"""PETS on the twin: probabilistic ensemble forward model + trajectory-sampling planner.

Chua et al. 2018. Our forward model + CEM planner is PETS without the uncertainty.
Adding it asks two things the deterministic model cannot:

  * Does the model KNOW where it is unsure? Predicted variance should rise in the
    fast / fallen regimes where one-step gyro change is mostly contact chatter
    (R^2 ~ 0.2 there). That is calibration, measured, not assumed.
  * Does planning through sampled particles (mean cost over particles, TS-inf) beat
    planning through the mean? It should stop chasing noise it cannot predict.

Model: E nets, each 128x2 swish, each emitting mean and log-variance of the 9-dim
delta, trained with Gaussian NLL on a bootstrap resample. predict() returns the
ensemble mean so every existing evaluation still runs unchanged.
"""
from __future__ import annotations
import argparse, json, time
import numpy as np, torch, torch.nn as nn
from growbot_cerebellum.forward import MLP, make_windows, rollout_error, K
from growbot_cerebellum.planner import Imagination, angle_cost, pick_targets, run_episode, rpy_to_quat
from growbot_cerebellum.sim import GrowBotSim
import mujoco


class ProbEnsemble:
    name = "pets"
    def __init__(self, E=5, hidden=128, epochs=40, lr=2e-3, seed=0, batch=1024):
        self.E, self.hidden, self.epochs, self.lr, self.seed, self.batch = E, hidden, epochs, lr, seed, batch

    def fit(self, X, Y, log=False):
        torch.manual_seed(self.seed); rng = np.random.default_rng(self.seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.ymu, self.ysd = Y.mean(0), Y.std(0) + 1e-6
        Xt = torch.tensor((X - self.mu) / self.sd); Yt = torch.tensor((Y - self.ymu) / self.ysd)
        n, d_in, d_out = len(Xt), X.shape[1], Y.shape[1]
        self.nets = []
        for e in range(self.E):
            net = nn.Sequential(nn.Linear(d_in, self.hidden), nn.SiLU(), nn.Linear(self.hidden, self.hidden), nn.SiLU(),
                                nn.Linear(self.hidden, 2 * d_out))
            opt = torch.optim.Adam(net.parameters(), lr=self.lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
            boot = torch.tensor(rng.integers(0, n, n))                     # bootstrap resample per member
            for ep in range(self.epochs):
                perm = boot[torch.randperm(n)]; tot = 0.0
                for i in range(0, n, self.batch):
                    idx = perm[i:i + self.batch]
                    out = net(Xt[idx]); mean, logvar = out[:, :d_out], out[:, d_out:].clamp(-8, 4)
                    nll = 0.5 * (((Yt[idx] - mean) ** 2) * torch.exp(-logvar) + logvar).mean()
                    opt.zero_grad(); nll.backward(); opt.step(); tot += float(nll) * len(idx)
                sched.step()
                if log and (ep + 1) % 20 == 0: print(f"    member {e} epoch {ep + 1:>3}  nll {tot / n:.4f}", flush=True)
            self.nets.append(net)
        self.n_params = sum(p.numel() for p in self.nets[0].parameters()) * self.E
        return self

    def _dist(self, X):
        """(E, N, 9) means and stds in output units."""
        with torch.no_grad():
            Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
            outs = [net(Xt) for net in self.nets]
        d = outs[0].shape[1] // 2
        means = torch.stack([o[:, :d] for o in outs]).numpy() * self.ysd + self.ymu
        stds = torch.stack([torch.exp(0.5 * o[:, d:].clamp(-8, 4)) for o in outs]).numpy() * self.ysd
        return means, stds

    def predict(self, X):
        m, _ = self._dist(X); return m.mean(0)

    def uncertainty(self, X):
        """aleatoric (mean member std) and epistemic (std of member means), per row, averaged over dims."""
        m, s = self._dist(X)
        return s.mean(0).mean(1), m.std(0).mean(1)


class PETSImagination:
    """TS-inf: each particle keeps one ensemble member for the whole rollout and samples the Gaussian each step."""
    def __init__(self, ens, K_, particles=8, seed=0):
        self.ens, self.K, self.P = ens, K_, particles; self.rng = np.random.default_rng(seed)

    def rollout(self, F_hist, A_hist, plans):
        N, H, _ = plans.shape; P = self.P; fdim = F_hist.shape[1]
        win = np.zeros((N * P, self.K, fdim + 2), np.float32)
        win[:, :, :fdim] = F_hist[None]; win[:, :, fdim:] = A_hist[None]
        cur = np.repeat(F_hist[0][None], N * P, 0)
        member = np.tile(self.rng.integers(0, self.ens.E, P), N)          # (N*P,) member id per particle
        out = np.zeros((N * P, H, fdim), np.float32)
        for h in range(H):
            win[:, 0, fdim:] = np.repeat(plans[:, h], P, 0)
            means, stds = self.ens._dist(win.reshape(N * P, -1))           # (E, N*P, 9)
            mu = means[member, np.arange(N * P)]; sd = stds[member, np.arange(N * P)]
            cur = cur + mu + sd * self.rng.standard_normal(mu.shape).astype(np.float32)
            for a in range(3):
                n = np.sqrt(cur[:, a] ** 2 + cur[:, a + 3] ** 2) + 1e-9; cur[:, a] /= n; cur[:, a + 3] /= n
            out[:, h] = cur
            win = np.roll(win, 1, axis=1); win[:, 0, :fdim] = cur
        return out.reshape(N, P, H, fdim)


def pets_cost(imag_out, target):
    """mean over particles of the per-particle roll/pitch cost -> (N,)"""
    N, P, H, _ = imag_out.shape
    return angle_cost(imag_out.reshape(N * P, H, -1), target).reshape(N, P).mean(1)


def cem_plan_pets(imag, F_hist, A_hist, target, H, n=192, iters=3, elite=24, rng=None, init_mean=None, smooth=0.6):
    rng = rng or np.random.default_rng(0)
    mean = np.zeros((H, 2), np.float32) if init_mean is None else init_mean.copy()
    std = np.full((H, 2), 0.6, np.float32)
    for _ in range(iters):
        raw = rng.normal(size=(n, H, 2)).astype(np.float32) * std + mean
        for h in range(1, H): raw[:, h] = smooth * raw[:, h - 1] + (1 - smooth) * raw[:, h]
        plans = np.clip(raw, -1.4, 1.4)
        cost = pets_cost(imag.rollout(F_hist, A_hist, plans), target)
        best = plans[np.argsort(cost)[:elite]]
        mean = best.mean(0); std = best.std(0) + 0.02
    return mean


class _PlannerShim:
    """Adapts cem_plan_pets to mimic.run_episode, which calls cem_plan(imag, F, A, tgt, H, rng=, init_mean=)."""
    def __init__(self, imag): self.imag = imag


def run_mimic(model_imag, planner_fn, targets, K_, H_plan, replan, seed):
    import mimic as M
    saved = M.cem_plan; M.cem_plan = planner_fn
    try:
        rmse, within = [], []
        for i, tgt in enumerate(targets):
            sim = GrowBotSim(seed=1000 + i); sim.reset(tilt=0.0)
            sim.d.qpos[3:7] = rpy_to_quat(*tgt[0, :3]); sim.d.qvel[3:6] = tgt[0, 3:]; mujoco.mj_forward(sim.m, sim.d)
            r = run_episode(sim, model_imag, tgt, K_, H_plan, replan, np.random.default_rng(seed + i))
            rmse.append(r["rmse"]); within.append(r["within_0.2"])
    finally:
        M.cem_plan = saved
    return float(np.mean(rmse)), float(np.std(rmse)), float(np.mean(within))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--E", type=int, default=5)
    ap.add_argument("--particles", type=int, default=8); ap.add_argument("--n-targets", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tr = np.load("data/train.npz"); te = np.load("data/test.npz")
    Xtr, Ytr, *_ = make_windows(tr["obs"], tr["act"], tr["next_obs"], tr["done"], K)
    Xte, Yte, *_, valid = make_windows(te["obs"], te["act"], te["next_obs"], te["done"], K)
    res = {}

    t0 = time.time(); det = MLP(hidden=128, epochs=args.epochs, seed=args.seed).fit(Xtr, Ytr); t_det = time.time() - t0
    t0 = time.time(); ens = ProbEnsemble(E=args.E, epochs=args.epochs, seed=args.seed).fit(Xtr, Ytr, log=True); t_ens = time.time() - t0
    print(f"\ndeterministic MLP {det.n_params:,} params {t_det:.0f}s | ensemble {ens.n_params:,} params {t_ens:.0f}s")

    # ---- 1. accuracy of the ensemble mean vs the single net (existing metric) ----
    print("\n1) open-loop rollout, within 0.2 rad (ensemble MEAN vs single MLP)")
    for name, m in (("single MLP", det), ("ensemble mean", ens)):
        ro = rollout_error(m, te["obs"], te["act"], te["done"], K, [5, 25], seed=args.seed)
        res[f"rollout/{name}"] = {str(h): ro[h]["within_0.2rad"] for h in (5, 25)}
        print(f"   {name:<14} @100ms {ro[5]['within_0.2rad']*100:5.1f}%   @500ms {ro[25]['within_0.2rad']*100:5.1f}%")

    # ---- 2. calibration: does predicted uncertainty track the regimes? ----
    print("\n2) predicted uncertainty by regime (one-step, test set)")
    obs = te["obs"][valid]; gyro_mag = np.linalg.norm(obs[:, 3:], axis=1)
    fallen = (np.abs(obs[:, 0]) > 1.2) | (np.abs(obs[:, 1]) > 1.2)
    ale, epi = ens.uncertainty(Xte)
    err = np.abs(ens.predict(Xte) - Yte).mean(1)
    regimes = {"calm (|gyro|<1)": (gyro_mag < 1) & ~fallen, "moderate": (gyro_mag >= 1) & (gyro_mag <= 3) & ~fallen,
               "fast (|gyro|>3)": (gyro_mag > 3) & ~fallen, "fallen": fallen}
    print(f"   {'regime':<18}{'n':>7}{'aleatoric std':>15}{'epistemic std':>15}{'actual |err|':>14}")
    res["calibration"] = {}
    for name, sel in regimes.items():
        res["calibration"][name] = {"aleatoric": float(ale[sel].mean()), "epistemic": float(epi[sel].mean()), "err": float(err[sel].mean())}
        print(f"   {name:<18}{sel.sum():>7}{ale[sel].mean():>15.4f}{epi[sel].mean():>15.4f}{err[sel].mean():>14.4f}")
    corr = float(np.corrcoef(ale + epi, err)[0, 1])
    res["calibration"]["corr_total_std_vs_err"] = corr
    print(f"   corr(predicted std, actual |err|) over all rows: {corr:.3f}")

    # ---- 3. mimic game: mean-planner vs particle-planner ----
    print("\n3) mimic game, 40 targets")
    rng = np.random.default_rng(args.seed); targets, _ = pick_targets(te, 100, args.n_targets, rng)
    from mimic import cem_plan as det_cem
    r_det = run_mimic(Imagination(det, K), det_cem, targets, K, 15, 5, args.seed)
    r_ensmean = run_mimic(Imagination(ens, K), det_cem, targets, K, 15, 5, args.seed)
    r_pets = run_mimic(PETSImagination(ens, K, args.particles, args.seed), cem_plan_pets, targets, K, 15, 5, args.seed)
    for name, r in (("single MLP, mean plan", r_det), ("ensemble mean, mean plan", r_ensmean), (f"PETS, {args.particles} particles", r_pets)):
        res[f"mimic/{name}"] = {"rmse": r[0], "sd": r[1], "within": r[2]}
        print(f"   {name:<28} RMSE {r[0]:.3f} ± {r[1]:.3f}   within 0.2 rad {r[2]*100:5.1f}%")
    json.dump(res, open("results/pets.json", "w"), indent=1)

if __name__ == "__main__":
    main()
