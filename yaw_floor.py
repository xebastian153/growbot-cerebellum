"""Is the twin's own yaw error a model limit, an information limit, or noise?

The forward model's weakest axis is yaw even in-sim (README table: 59% within
0.2 rad @1 s vs ~77% roll/pitch). Three hypotheses, one decision rule stated
before the numbers:

  (a) model-limited        yaw improves materially when data or capacity grows.
  (b) information-limited  yaw is flat under scaling, but a model that also sees
                           privileged sim state (contact forces, linear velocity,
                           joint state -- nothing a phone IMU has) improves
                           materially: the signal exists, the observation lacks it.
  (c) intrinsically noisy  flat under scaling AND under privileged input: contact
                           chatter is aleatoric at this timescale, and planning
                           should not chase it.

"Materially" is fixed up front: an improvement in yaw within-0.2 rad @1 s larger
than max(3.0 points, 2x the seed spread of the baseline across three MLP init
seeds). The privileged probe is a DIAGNOSTIC, not a deployable model: at rollout
its privileged channel is teacher-forced from the recorded truth while the IMU
channel runs on its own predictions, so it answers "is the information there?"
and nothing else.

Data discipline: one 4x collection (seed 0) whose 400k-step prefix is asserted
byte-equal to the published data/train.npz; all data scales are nested prefixes
of it. Evaluation uses a privileged re-collection of seed 1 asserted equal to
data/test.npz, so every number is comparable with the published table. All
conditions share the evaluation starts (same mask, same rng).
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import mujoco
import numpy as np

from growbot_cerebellum.sim import GrowBotSim, Excitation, CTRL_HZ, OBS_DIM, ACT_DIM
from growbot_cerebellum.forward import MLP, encode_obs, decode_obs

K = 5
HORIZONS = (25, 50)                     # 500 ms and 1 s
N_STARTS = 2000
PRIV_DIM = 18                           # z, quat(4), joints(2), qvel(8), foot forces(2), ncon


# ----------------------------------------------------------------------
# collection with privileged state
# ----------------------------------------------------------------------

def _priv(sim):
    """Privileged features at the current sim state: what a phone IMU cannot see.

    z + quaternion + servo joint angles (qpos minus world x,y, which carry no
    dynamics information), full qvel (linear velocity is the part the IMU lacks;
    angular repeats the gyro), per-foot normal-force sum, and contact count.
    """
    d, m = sim.d, sim.m
    feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
            for g in ("lower_leg_1", "lower_leg_2")]
    f = np.zeros(2)
    buf = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        for j, gid in enumerate(feet):
            if c.geom1 == gid or c.geom2 == gid:
                mujoco.mj_contactForce(m, d, i, buf)
                f[j] += abs(buf[0])          # normal component in contact frame
    return np.concatenate([d.qpos[2:9], d.qvel[:8], f, [d.ncon]]).astype(np.float32)


def collect_priv(n_steps, seed=0, push_prob=0.01, episode_s=8.0, log_every=0):
    """growbot_sim.collect with a privileged channel recorded alongside.

    The loop is a line-for-line mirror of collect() (same rng draws in the same
    order) so a given seed produces the identical stream; main() asserts the
    seed-0/seed-1 outputs equal the published data/*.npz to prove it. Privileged
    readout makes no rng calls.
    """
    sim = GrowBotSim(seed, body="walk")
    exc = Excitation(sim.rng)
    O = np.zeros((n_steps, OBS_DIM), np.float32)
    A = np.zeros((n_steps, ACT_DIM), np.float32)
    O2 = np.zeros((n_steps, OBS_DIM), np.float32)
    P = np.zeros((n_steps, PRIV_DIM), np.float32)
    D = np.zeros(n_steps, bool)
    modes = []
    o = sim.reset(tilt=0.3)
    prev = np.zeros(2, np.float32)
    ep_len = int(episode_s * CTRL_HZ)
    def fresh():
        return sim.reset(tilt=1.0 if sim.rng.random() < 0.2 else 0.3)
    t_ep = 0
    for i in range(n_steps):
        P[i] = _priv(sim)
        a = exc(o, prev)
        if sim.rng.random() < push_prob:
            sim.push()
        o2 = sim.step(a)
        O[i], A[i], O2[i] = o, a, o2
        modes.append(exc.mode)
        t_ep += 1
        end = t_ep >= ep_len or (sim.fallen() and sim.rng.random() < 0.02)
        D[i] = end
        if end:
            o = fresh(); prev = np.zeros(2, np.float32); t_ep = 0
            exc.new_segment()
        else:
            o, prev = o2, a
        if log_every and (i + 1) % log_every == 0:
            print(f"  {i + 1}/{n_steps}", flush=True)
    return O, A, O2, P, D, np.array(modes)


# ----------------------------------------------------------------------
# windows over an arbitrary per-tick feature matrix (generalizes make_windows)
# ----------------------------------------------------------------------

def windows_feat(feat, act, F, F2, done, K):
    """Same validity rules as forward.make_windows; the slot content is `feat`."""
    N = len(feat)
    fdim = feat.shape[1]
    X = np.zeros((N, K * (fdim + act.shape[1])), np.float32)
    valid = ~done
    nodone_back = np.ones(N, bool)
    for k in range(K):
        idx = np.arange(N) - k
        if k > 0:
            nodone_back = nodone_back & (idx >= 0) & ~done[np.clip(idx, 0, N - 1)]
            valid = valid & nodone_back
        valid = valid & (idx >= 0)
        idx = np.clip(idx, 0, N - 1)
        X[:, k * (fdim + 2):(k + 1) * (fdim + 2)] = np.concatenate([feat[idx], act[idx]], axis=1)
    Y = F2 - F
    return X[valid], Y[valid]


# ----------------------------------------------------------------------
# rollout returning per-start per-axis errors (one evaluator for every model)
# ----------------------------------------------------------------------

def rollout_axis_errors(model, obs, act, done, priv=None, K=K, horizons=HORIZONS,
                        n_starts=N_STARTS, seed=0):
    """Open-loop rollout; IMU channel imagined, privileged channel (if any)
    teacher-forced from the recorded truth. Returns starts and, per horizon,
    the (n_starts, 3) absolute angle errors, so callers can slice by regime.
    Start selection depends only on (done, K, max horizon): identical for every
    model evaluated on the same collection."""
    rng = np.random.default_rng(seed)
    N = len(obs)
    F = encode_obs(obs)
    feat = F if priv is None else np.concatenate([F, priv], axis=1)
    fdim = feat.shape[1]
    Hmax = max(horizons)
    ok = np.ones(N, bool)
    for j in range(K):
        ok &= np.roll(~done, j + 1)
    for j in range(Hmax):
        ok &= np.roll(~done, -j)
    ok[:K] = False; ok[N - Hmax - 1:] = False
    starts = rng.choice(np.flatnonzero(ok), size=min(n_starts, ok.sum()), replace=False)

    win = np.zeros((len(starts), K, fdim + 2), np.float32)
    for k in range(K):
        win[:, k, :fdim] = feat[starts - k]
        win[:, k, fdim:] = act[starts - k]
    cur = F[starts].copy()
    out = {}
    for h in range(1, Hmax + 1):
        win[:, 0, fdim:] = act[starts + h - 1]
        d = model.predict(win.reshape(len(starts), -1))
        cur = cur + d
        for a in range(3):
            n = np.sqrt(cur[:, a] ** 2 + cur[:, a + 3] ** 2) + 1e-9
            cur[:, a] /= n; cur[:, a + 3] /= n
        win = np.roll(win, 1, axis=1)
        win[:, 0, :9] = cur
        if priv is not None:
            win[:, 0, 9:fdim] = priv[starts + h]     # teacher-forced truth
        if h in horizons:
            pa = decode_obs(cur)[:, :3]
            ta = decode_obs(F[starts + h])[:, :3]
            out[h] = np.abs(np.arctan2(np.sin(pa - ta), np.cos(pa - ta)))
    return starts, out


def summarize(errs):
    return {"within_0.2rad_axis": {a: float((errs[:, i] < 0.2).mean())
                                   for i, a in enumerate(("roll", "pitch", "yaw"))},
            "rmse_axis_rad": {a: float(np.sqrt((errs[:, i] ** 2).mean()))
                              for i, a in enumerate(("roll", "pitch", "yaw"))}}


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--base-steps", type=int, default=400_000,
                    help="1x dataset size; scales are nested prefixes of a 4x collection")
    ap.add_argument("--eval-steps", type=int, default=60_000)
    args = ap.parse_args()
    t00 = time.time()

    # ---- collections: one 4x train (seed 0, prefix == data/train.npz) + eval (seed 1 == data/test.npz)
    print(f"collecting 4x train ({4 * args.base_steps:,} steps, seed 0) with privileged state...", flush=True)
    O, A, O2, P, D, M = collect_priv(4 * args.base_steps, seed=0, log_every=400_000)
    print(f"collecting eval ({args.eval_steps:,} steps, seed 1)...", flush=True)
    Oe, Ae, O2e, Pe, De, Me = collect_priv(args.eval_steps, seed=1)

    tr = np.load("data/train.npz"); te = np.load("data/test.npz")
    n0 = len(tr["obs"])
    assert np.allclose(O[:n0], tr["obs"]) and np.array_equal(D[:n0], tr["done"]), \
        "collect_priv diverged from collect(): seed-0 prefix != data/train.npz"
    assert np.allclose(Oe, te["obs"]), "eval collection != data/test.npz"
    print("prefix checks: 4x collection extends data/train.npz; eval == data/test.npz", flush=True)

    F, F2 = encode_obs(O), encode_obs(O2)
    results = {"config": {"K": K, "epochs": args.epochs, "n_starts": N_STARTS,
                          "horizons_ms": [h * 1000 // CTRL_HZ for h in HORIZONS],
                          "train_seed": 0, "eval_seed": 1, "base_steps": args.base_steps,
                          "eval_steps": args.eval_steps, "priv_dim": PRIV_DIM,
                          "rule": "material = yaw within-0.2rad gain @1s > max(3.0 pts, 2x baseline seed spread)"},
               "conditions": {}}

    def train_eval(tag, steps, hidden, mlp_seed=0, use_priv=False):
        t0 = time.time()
        sl = slice(0, steps)
        feat = F[sl] if not use_priv else np.concatenate([F[sl], P[sl]], axis=1)
        X, Y = windows_feat(feat, A[sl], F[sl], F2[sl], D[sl], K)
        m = MLP(hidden=hidden, epochs=args.epochs, seed=mlp_seed).fit(X, Y)
        _, errs = rollout_axis_errors(m, Oe, Ae, De, priv=Pe if use_priv else None)
        row = {"steps": steps, "hidden": hidden, "params": m.n_params, "mlp_seed": mlp_seed,
               "privileged": use_priv, "fit_s": round(time.time() - t0, 1),
               "horizons": {str(h * 20) + "ms": summarize(errs[h]) for h in HORIZONS}}
        results["conditions"][tag] = row
        y5, y10 = (row["horizons"]["500ms"]["within_0.2rad_axis"]["yaw"],
                   row["horizons"]["1000ms"]["within_0.2rad_axis"]["yaw"])
        rp = row["horizons"]["1000ms"]["within_0.2rad_axis"]
        print(f"{tag:<28} {steps:>9,} steps  h{hidden:<4} {m.n_params:>7,}p  "
              f"yaw {y5 * 100:5.1f}% @500ms {y10 * 100:5.1f}% @1s   "
              f"roll {rp['roll'] * 100:5.1f}%  pitch {rp['pitch'] * 100:5.1f}% @1s   "
              f"({time.time() - t0:.0f}s)", flush=True)
        return m, errs

    b = args.base_steps
    print("\n-- baseline seed spread (1x data, 128 hidden, MLP seeds 0/1/2) --", flush=True)
    base_models = {}
    for s in (0, 1, 2):
        m, errs = train_eval(f"baseline seed {s}", b, 128, mlp_seed=s)
        if s == 0:
            base_models["m"], base_models["errs"] = m, errs
    yaws = [results["conditions"][f"baseline seed {s}"]["horizons"]["1000ms"]
            ["within_0.2rad_axis"]["yaw"] for s in (0, 1, 2)]
    spread = max(yaws) - min(yaws)
    material = max(0.03, 2 * spread)
    results["config"]["baseline_yaw_1s"] = yaws
    results["config"]["seed_spread"] = spread
    results["config"]["material_threshold"] = material
    print(f"baseline yaw @1s: {[f'{y * 100:.1f}%' for y in yaws]}  spread {spread * 100:.1f} pts"
          f"  -> material threshold {material * 100:.1f} pts", flush=True)

    print("\n-- data scaling (128 hidden) --", flush=True)
    for mult in (0.5, 2, 4):
        train_eval(f"data {mult}x", int(b * mult), 128)

    print("\n-- capacity scaling (1x data) --", flush=True)
    for h in (90, 192, 288):
        train_eval(f"capacity h{h}", b, h)

    print("\n-- privileged probe (1x data, 128 hidden; diagnostic only) --", flush=True)
    _, perrs = train_eval("privileged", b, 128, use_priv=True)

    # ---- regime split of the baseline-vs-privileged delta (CONVENTIONS: no hidden regimes)
    starts, _ = rollout_axis_errors(base_models["m"], Oe, Ae, De)   # same starts everywhere
    fallen = (np.abs(Oe[starts, 0]) > 1.2) | (np.abs(Oe[starts, 1]) > 1.2)
    fast = (np.linalg.norm(Oe[starts, 3:], axis=1) > 3.0) & ~fallen
    calm = ~fallen & ~fast
    regimes = {"calm": calm, "fast": fast, "fallen": fallen}
    print("\n-- yaw within 0.2 rad @1 s by start regime --", flush=True)
    results["regime_split_1s_yaw"] = {}
    for name, sel in regimes.items():
        if sel.sum() < 50:
            continue
        bm = float((base_models["errs"][50][sel, 2] < 0.2).mean())
        pm = float((perrs[50][sel, 2] < 0.2).mean())
        results["regime_split_1s_yaw"][name] = {"n": int(sel.sum()), "baseline": bm, "privileged": pm}
        print(f"  {name:<8} n={sel.sum():<5} baseline {bm * 100:5.1f}%   privileged {pm * 100:5.1f}%", flush=True)

    # ---- verdict by the pre-stated rule
    ymean = float(np.mean(yaws))
    def gain(tag):
        return results["conditions"][tag]["horizons"]["1000ms"]["within_0.2rad_axis"]["yaw"] - ymean
    gains = {"data 4x": gain("data 4x"), "capacity h288": gain("capacity h288"),
             "privileged": gain("privileged")}
    verdict = {k: ("material" if g > material else "not material") for k, g in gains.items()}
    results["gains_vs_baseline_mean"] = gains
    results["verdict"] = verdict
    print("\ngains on yaw @1s vs baseline mean "
          + "  ".join(f"{k}: {g * +100:+.1f} pts ({verdict[k]})" for k, g in gains.items()), flush=True)

    Path("results").mkdir(exist_ok=True)
    with open("results/yaw_floor.json", "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nsaved results/yaw_floor.json   total {(time.time() - t00) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
