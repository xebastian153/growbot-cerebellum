"""Forward models for the GrowBot body: predict the next IMU from history + action.

Three competitors, so the result cannot flatter itself:

  persistence  next = now. If a model does not beat this it has learned nothing.
  linear       GCML-style forward map: delta = A x + B u, fitted by least squares
               (the closed form of the delta rules of equations 12-13). Says how
               far the paper's linear assumption reaches on real body physics.
  mlp          a small nonlinear net, the "cerebellum" the video says is missing.

All models predict the *change* in the IMU over one 20 ms tick from a window of
the last K observations and actions. Multi-step accuracy comes from rolling the
model forward on its own predictions -- the same bootstrapping that made the
crafting planner drift, now measured on physics.

Angles are represented as (sin, cos) so yaw wrapping and roll going through +-pi
in a fall do not produce spurious jumps.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).parent
CTRL_HZ = 50


# ----------------------------------------------------------------------
# features
# ----------------------------------------------------------------------

def encode_obs(o):
    """(N,6) rpy+gyro -> (N,9): sin/cos of the three angles + gyro."""
    ang, gyro = o[..., :3], o[..., 3:]
    return np.concatenate([np.sin(ang), np.cos(ang), gyro], axis=-1)


def decode_obs(f):
    ang = np.arctan2(f[..., :3], f[..., 3:6])
    return np.concatenate([ang, f[..., 6:]], axis=-1)


def make_windows(obs, act, next_obs, done, K):
    """Stack K past (obs, act) pairs into one input; target is next encoded obs.

    Windows never cross an episode boundary.
    """
    N = len(obs)
    F = encode_obs(obs)
    F2 = encode_obs(next_obs)
    fdim = F.shape[1]
    X = np.zeros((N, K * (fdim + act.shape[1])), np.float32)
    # A row is valid iff (a) its own transition does not cross a cut -- under the
    # parser's convention done[t] means next_obs[t] is interpolated inside a gap,
    # so Y[t] would be a delta the physics never produced -- and (b) no cut lies
    # inside the K-step history. (a) was missing until a review caught it: the
    # third relative of the same cut-boundary bug (servo transient, gap endpoints).
    valid = ~done
    nodone_back = np.ones(N, bool)
    for k in range(K):
        idx = np.arange(N) - k
        if k > 0:
            nodone_back = nodone_back & (idx >= 0) & ~done[np.clip(idx, 0, N - 1)]
            valid = valid & nodone_back
        valid = valid & (idx >= 0)
        idx = np.clip(idx, 0, N - 1)
        X[:, k * (fdim + 2):(k + 1) * (fdim + 2)] = np.concatenate([F[idx], act[idx]], axis=1)
    Y = F2 - F                     # predict the delta
    return X[valid], Y[valid], F[valid], F2[valid], valid


# ----------------------------------------------------------------------
# models
# ----------------------------------------------------------------------

class Persistence:
    name = "persistence"
    def fit(self, X, Y): return self
    def predict(self, X): return np.zeros((len(X), 9), np.float32)


class Linear:
    """delta = X @ W + b, least squares. Equivalent to the converged GCML delta rules."""
    name = "linear"
    def fit(self, X, Y, ridge=1e-3):
        Xb = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)
        A = Xb.T @ Xb + ridge * np.eye(Xb.shape[1], dtype=np.float32)
        A[-1, -1] -= ridge                      # do not penalise the bias
        self.W = np.linalg.solve(A, Xb.T @ Y)
        return self
    def predict(self, X):
        Xb = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)
        return Xb @ self.W


class MLP:
    name = "mlp"
    def __init__(self, hidden=128, layers=2, epochs=30, lr=2e-3, seed=0):
        self.hidden, self.layers, self.epochs, self.lr, self.seed = hidden, layers, epochs, lr, seed

    def fit(self, X, Y, log=False):
        torch.manual_seed(self.seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.ymu, self.ysd = Y.mean(0), Y.std(0) + 1e-6
        Xt = torch.tensor((X - self.mu) / self.sd)
        Yt = torch.tensor((Y - self.ymu) / self.ysd)
        dims = [X.shape[1]] + [self.hidden] * self.layers + [Y.shape[1]]
        mods = []
        for i in range(len(dims) - 1):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                mods.append(nn.SiLU())
        self.net = nn.Sequential(*mods)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        n = len(Xt); bs = 1024
        for ep in range(self.epochs):
            perm = torch.randperm(n); tot = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                loss = ((self.net(Xt[idx]) - Yt[idx]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss) * len(idx)
            sched.step()
            if log and (ep + 1) % 10 == 0:
                print(f"    epoch {ep + 1:>3}  loss {tot / n:.4f}", flush=True)
        self.n_params = sum(p.numel() for p in self.net.parameters())
        return self

    def predict(self, X):
        with torch.no_grad():
            Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
            return self.net(Xt).numpy() * self.ysd + self.ymu


# ----------------------------------------------------------------------
# evaluation: one step, and rolled out on its own predictions
# ----------------------------------------------------------------------

def rollout_error(model, obs, act, done, K, horizons, n_starts=2000, seed=0, start_mask=None):
    """Open-loop imagination: feed the model its own predictions for H steps.

    Returns per-horizon RMSE on (roll, pitch) in radians and on gyro in rad/s,
    plus the fraction of starts where the imagined roll/pitch stays within 0.2 rad
    of the truth at that horizon.

    `start_mask` (bool per tick, default None = every eligible tick) restricts the
    starts to a subset -- one regime, one balance state -- so the same rollout maths
    can be split by what the body was doing at the start instead of re-implemented.
    None reproduces every earlier caller bit-identically.
    """
    rng = np.random.default_rng(seed)
    N = len(obs)
    F = encode_obs(obs)
    Hmax = max(horizons)
    # candidate starts: full K-window before, full Hmax after, no episode end inside
    ok = np.ones(N, bool)
    for j in range(0, K):
        ok &= np.roll(~done, j + 1)      # no done in the K-1 steps before t
    for j in range(0, Hmax):
        ok &= np.roll(~done, -j)         # no done in t..t+Hmax-1
    ok[:K] = False; ok[N - Hmax - 1:] = False
    if start_mask is not None:
        ok &= np.asarray(start_mask, bool)
    if not ok.any():
        return {h: None for h in horizons}
    starts = rng.choice(np.flatnonzero(ok), size=min(n_starts, ok.sum()), replace=False)

    fdim = F.shape[1]
    # build the initial windows for all starts
    win = np.zeros((len(starts), K, fdim + 2), np.float32)
    for k in range(K):
        win[:, k, :fdim] = F[starts - k]
        win[:, k, fdim:] = act[starts - k]
    cur = F[starts].copy()
    out = {}
    for h in range(1, Hmax + 1):
        # slot 0 holds the action executed during this tick: the one actually taken
        win[:, 0, fdim:] = act[starts + h - 1]
        X = win.reshape(len(starts), -1)
        d = model.predict(X)
        cur = cur + d
        # renormalise the (sin, cos) pairs so angles stay on the circle
        for a in range(3):
            n = np.sqrt(cur[:, a] ** 2 + cur[:, a + 3] ** 2) + 1e-9
            cur[:, a] /= n; cur[:, a + 3] /= n
        # shift the window: newest first
        win = np.roll(win, 1, axis=1)
        win[:, 0, :fdim] = cur
        if h in horizons:
            truth = F[starts + h]
            pred_ang = decode_obs(cur)[:, :3]; true_ang = decode_obs(truth)[:, :3]
            ang_err = np.arctan2(np.sin(pred_ang - true_ang), np.cos(pred_ang - true_ang))
            gyro_err = cur[:, 6:] - truth[:, 6:]
            out[h] = {
                "n_starts": int(len(starts)),
                # headline (roll/pitch) kept for continuity with every published table
                "rmse_rollpitch_rad": float(np.sqrt((ang_err[:, :2] ** 2).mean())),
                "rmse_gyro_rads": float(np.sqrt((gyro_err ** 2).mean())),
                "within_0.2rad": float((np.abs(ang_err[:, :2]).max(1) < 0.2).mean()),
                # per-axis: the spin gap lives in yaw, so the tool that will measure
                # it must see it -- one RMSE and one within per angle
                "rmse_axis_rad": {a: float(np.sqrt((ang_err[:, i] ** 2).mean()))
                                  for i, a in enumerate(("roll", "pitch", "yaw"))},
                "within_0.2rad_axis": {a: float((np.abs(ang_err[:, i]) < 0.2).mean())
                                       for i, a in enumerate(("roll", "pitch", "yaw"))},
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=5, help="history window (ticks)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 25, 50])
    args = ap.parse_args()

    tr = np.load(HERE / "data" / "train.npz"); te = np.load(HERE / "data" / "test.npz")
    Xtr, Ytr, _, _, _ = make_windows(tr["obs"], tr["act"], tr["next_obs"], tr["done"], args.K)
    Xte, Yte, _, _, _ = make_windows(te["obs"], te["act"], te["next_obs"], te["done"], args.K)
    print(f"windows K={args.K}: train {len(Xtr):,}  test {len(Xte):,}  "
          f"input dim {Xtr.shape[1]}  target dim {Ytr.shape[1]}")

    models = [Persistence(), Linear(), MLP(hidden=args.hidden, epochs=args.epochs)]
    results = {}
    for m in models:
        t0 = time.time()
        m.fit(Xtr, Ytr, log=True) if isinstance(m, MLP) else m.fit(Xtr, Ytr)
        one = m.predict(Xte)
        rmse1 = float(np.sqrt(((one - Yte) ** 2).mean()))
        ro = rollout_error(m, te["obs"], te["act"], te["done"], args.K, args.horizons)
        results[m.name] = {"one_step_rmse_delta": rmse1, "rollout": ro,
                           "fit_s": round(time.time() - t0, 1),
                           "params": getattr(m, "n_params", None)}
        print(f"\n{m.name:<12} one-step delta RMSE {rmse1:.4f}   fit {time.time() - t0:.1f}s"
              + (f"   params {m.n_params:,}" if hasattr(m, "n_params") else ""))
        print(f"  {'horizon':>8}{'ms':>6}{'roll/pitch RMSE':>18}{'yaw RMSE':>11}{'gyro RMSE':>12}{'within 0.2':>12}{'yaw w0.2':>10}")
        for h in args.horizons:
            r = ro[h]
            print(f"  {h:>8}{h * 1000 // CTRL_HZ:>6}{r['rmse_rollpitch_rad']:>18.4f}"
                  f"{r['rmse_axis_rad']['yaw']:>11.4f}{r['rmse_gyro_rads']:>12.3f}"
                  f"{r['within_0.2rad'] * 100:>11.1f}%{r['within_0.2rad_axis']['yaw'] * 100:>9.1f}%")

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"forward_K{args.K}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()


def by_regime(model, te, K, h=5):
    """Rollout error at horizon h, split by what the body was doing at the start."""
    obs, act, done, mode = te["obs"], te["act"], te["done"], te["mode"]
    F = encode_obs(obs); N = len(obs); fdim = F.shape[1]
    ok = np.ones(N, bool)
    for j in range(K): ok &= np.roll(~done, j + 1)
    for j in range(h): ok &= np.roll(~done, -j)
    ok[:K] = False; ok[N - h - 1:] = False
    fallen = (np.abs(obs[:, 0]) > 1.2) | (np.abs(obs[:, 1]) > 1.2)
    fast = np.linalg.norm(obs[:, 3:], axis=1) > 3.0
    regimes = {"policy walking": (mode == "policy") & ~fallen & ~fast,
               "sine gait": (mode == "sine") & ~fallen & ~fast,
               "keyframe/OU": np.isin(mode, ["keyframe", "ou"]) & ~fallen & ~fast,
               "still": (mode == "still") & ~fallen & ~fast,
               "fast (|gyro|>3)": fast & ~fallen,
               "fallen": fallen}
    rows = []
    for name, sel in regimes.items():
        idx = np.flatnonzero(ok & sel)
        if len(idx) < 50: continue
        idx = np.random.default_rng(0).choice(idx, size=min(1500, len(idx)), replace=False)
        win = np.zeros((len(idx), K, fdim + 2), np.float32)
        for k in range(K):
            win[:, k, :fdim] = F[idx - k]; win[:, k, fdim:] = act[idx - k]
        cur = F[idx].copy()
        for s in range(1, h + 1):
            cur = cur + model.predict(win.reshape(len(idx), -1))
            for a in range(3):
                n = np.sqrt(cur[:, a] ** 2 + cur[:, a + 3] ** 2) + 1e-9
                cur[:, a] /= n; cur[:, a + 3] /= n
            win = np.roll(win, 1, axis=1); win[:, 0, :fdim] = cur; win[:, 0, fdim:] = act[idx + s]
        pa, ta = decode_obs(cur)[:, :2], decode_obs(F[idx + h])[:, :2]
        e = np.arctan2(np.sin(pa - ta), np.cos(pa - ta))
        rows.append((name, len(idx), float(np.sqrt((e ** 2).mean())), float((np.abs(e).max(1) < 0.2).mean())))
    return rows

