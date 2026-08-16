"""Sim-to-real proxy: train the forward model on the nominal Olie body, then measure it
on the 13 domain-randomisation corners the project itself uses (dr_sweep_spin.py),
and see how much an online residual learned from prediction error recovers.

The claim under test is brit's framing, "continual correction instead of better sim":
a forward model that is wrong about a new body should become right by watching its
own error on that body, on device, without retraining. Until a real IMU log exists,
the DR corners are the closest thing to "a different body" the project already
believes in.

Online residual: a linear map from the model's own input window to a correction of
its output, updated every tick by normalised LMS on the observed prediction error.
That is the cheapest learner that runs on a phone (one matrix, one outer product per
tick) and it is exactly the local delta rule of the GCML paper's equations 12-13,
which is why it is the first thing to try -- if linear residuals close most of the
gap, no on-device backprop is needed.

For each corner, three numbers at a 100 ms open-loop horizon:
  frozen     the nominal model, as shipped
  adapted    the same model plus the online residual after `warm_s` seconds of
             experience on the new body (updated only from what it could observe)
  oracle     a model trained on that corner's own data -- the ceiling
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "sim"))
from growbot_sim import DR, collect                       # noqa: E402
from forward import MLP, make_windows, encode_obs, decode_obs   # noqa: E402

HERE = Path(__file__).parent
K = 5

# 13 corners: nominal, each factor at both extremes (5x2 = 10), and two worst-case combos
def corners():
    lo = {k: v[0] for k, v in DR.items()}; hi = {k: v[1] for k, v in DR.items()}
    def mk(name, mass=1.0, dcom=(0, 0, 0), leg=1.0, gain=1.0, fric=None):
        return name, dict(mass_scale=mass, dcom=dcom, leg_scale=leg, gain_mult=gain, friction=fric)
    return [
        mk("nominal"),
        mk("mass -", mass=lo["mass_scale"]), mk("mass +", mass=hi["mass_scale"]),
        mk("com back/low", dcom=(lo["dcom_x"], 0, lo["dcom_z"])), mk("com fwd/high", dcom=(hi["dcom_x"], 0, hi["dcom_z"])),
        mk("leg -", leg=lo["leg_scale"]), mk("leg +", leg=hi["leg_scale"]),
        mk("gain -", gain=lo["gain_mult"]), mk("gain +", gain=hi["gain_mult"]),
        mk("friction -", fric=lo["friction"]), mk("friction +", fric=hi["friction"]),
        mk("worst A (heavy, long, weak, slippery)", mass=hi["mass_scale"], dcom=(hi["dcom_x"], 0, hi["dcom_z"]),
           leg=hi["leg_scale"], gain=lo["gain_mult"], fric=lo["friction"]),
        mk("worst B (light, short, strong, grippy)", mass=lo["mass_scale"], dcom=(lo["dcom_x"], 0, lo["dcom_z"]),
           leg=lo["leg_scale"], gain=hi["gain_mult"], fric=hi["friction"]),
    ]


class OnlineResidual:
    """delta_out += R @ x, R updated by normalised LMS on the observed one-step error."""
    def __init__(self, in_dim, out_dim, eta=0.05):
        self.R = np.zeros((out_dim, in_dim), np.float32)
        self.eta = eta
    def correct(self, X):
        return X @ self.R.T
    def update(self, x, err):
        # err = observed_delta - (model_delta + correction); x = the model input window
        self.R += self.eta * np.outer(err, x) / (float(x @ x) + 1.0)


def horizon_within(model, obs, act, done, residual=None, h=5, n_starts=1500, seed=0):
    """Open-loop imagination for h ticks; fraction of starts within 0.2 rad roll/pitch."""
    rng = np.random.default_rng(seed)
    F = encode_obs(obs); N = len(obs); fdim = F.shape[1]
    ok = np.ones(N, bool)
    for j in range(K): ok &= np.roll(~done, j + 1)
    for j in range(h): ok &= np.roll(~done, -j)
    ok[:K] = False; ok[N - h - 1:] = False
    starts = rng.choice(np.flatnonzero(ok), size=min(n_starts, ok.sum()), replace=False)
    win = np.zeros((len(starts), K, fdim + 2), np.float32)
    for k in range(K):
        win[:, k, :fdim] = F[starts - k]; win[:, k, fdim:] = act[starts - k]
    cur = F[starts].copy()
    for s in range(1, h + 1):
        win[:, 0, fdim:] = act[starts + s - 1]
        X = win.reshape(len(starts), -1)
        d = model.predict(X)
        if residual is not None:
            d = d + residual.correct(X)
        cur = cur + d
        for a in range(3):
            n = np.sqrt(cur[:, a] ** 2 + cur[:, a + 3] ** 2) + 1e-9
            cur[:, a] /= n; cur[:, a + 3] /= n
        win = np.roll(win, 1, axis=1); win[:, 0, :fdim] = cur
    pa, ta = decode_obs(cur)[:, :2], decode_obs(F[starts + h])[:, :2]
    e = np.arctan2(np.sin(pa - ta), np.cos(pa - ta))
    return float((np.abs(e).max(1) < 0.2).mean()), float(np.sqrt((e ** 2).mean()))


def adapt_online(model, obs, act, next_obs, done, warm_ticks, eta):
    """Stream the first warm_ticks of experience through the residual, one tick at a time."""
    X, Y, *_ = make_windows(obs[:warm_ticks], act[:warm_ticks], next_obs[:warm_ticks], done[:warm_ticks], K)
    res = OnlineResidual(X.shape[1], Y.shape[1], eta)
    base = model.predict(X)              # frozen model's predictions on this stream
    for i in range(len(X)):
        err = Y[i] - (base[i] + res.correct(X[i:i + 1])[0])
        res.update(X[i], err)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--warm-s", type=float, nargs="+", default=[10, 60, 300],
                    help="seconds of experience on the new body before measuring")
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--corner-steps", type=int, default=30000, help="ticks collected per corner (600 s)")
    args = ap.parse_args()

    tr = np.load(HERE / "data" / "olie_train.npz")
    Xtr, Ytr, *_ = make_windows(tr["obs"], tr["act"], tr["next_obs"], tr["done"], K)
    print("training nominal forward model...", flush=True)
    nominal = MLP(hidden=128, epochs=args.epochs).fit(Xtr, Ytr)

    rows = []
    hdr = f"{'corner':<40}{'frozen':>8}" + "".join(f"{'adapt ' + str(int(w)) + 's':>11}" for w in args.warm_s) + f"{'oracle':>9}"
    print("\n" + hdr); print("-" * len(hdr))
    t0 = time.time()
    for name, dr in corners():
        # experience on this body: first part is the adaptation stream, the rest is held out
        O, A, O2, D, _ = collect(args.corner_steps, seed=hash(name) % 10000, body="olie", dr=dr)
        cut = int(max(args.warm_s) * 50)
        held = slice(cut, None)
        frozen, _ = horizon_within(nominal, O[held], A[held], D[held])
        adapted = []
        for w in args.warm_s:
            res = adapt_online(nominal, O, A, O2, D, int(w * 50), args.eta)
            adapted.append(horizon_within(nominal, O[held], A[held], D[held], residual=res)[0])
        # oracle: a model trained on this corner's own data (same budget as the biggest warm-up)
        Xc, Yc, *_ = make_windows(O[:cut], A[:cut], O2[:cut], D[:cut], K)
        oracle_m = MLP(hidden=128, epochs=20).fit(Xc, Yc)
        oracle, _ = horizon_within(oracle_m, O[held], A[held], D[held])
        rows.append({"corner": name, "frozen": frozen, "adapted": dict(zip([str(w) for w in args.warm_s], adapted)),
                     "oracle": oracle})
        print(f"{name:<40}{frozen * 100:>7.1f}%" + "".join(f"{a * 100:>10.1f}%" for a in adapted) + f"{oracle * 100:>8.1f}%", flush=True)
    print(f"\n{time.time() - t0:.0f}s")

    fr = np.array([r["frozen"] for r in rows[1:]]); orc = np.array([r["oracle"] for r in rows[1:]])
    print(f"\nnon-nominal corners, mean within-0.2rad @100ms:  frozen {fr.mean() * 100:.1f}%  "
          + "  ".join(f"adapt {w}s {np.mean([r['adapted'][str(w)] for r in rows[1:]]) * 100:.1f}%" for w in args.warm_s)
          + f"  oracle {orc.mean() * 100:.1f}%")
    gap = orc - fr
    rec = {w: (np.array([r["adapted"][str(w)] for r in rows[1:]]) - fr) / np.where(gap > 1e-6, gap, np.nan) for w in args.warm_s}
    print("fraction of the frozen->oracle gap recovered by the online residual: "
          + "  ".join(f"{w}s: {np.nanmean(v) * 100:.0f}%" for w, v in rec.items()))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "sim2real_proxy.json").write_text(json.dumps({"rows": rows, "config": vars(args)}, indent=1))


if __name__ == "__main__":
    main()
