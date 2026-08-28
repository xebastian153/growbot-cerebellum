"""Sim-to-real proxy: train the forward model on the nominal Olie body, then measure it
on the 13 domain-randomisation corners the project itself uses (dr_sweep_spin.py),
and see how much an online residual learned from prediction error recovers.

The claim under test: continual correction instead of a better simulator --
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
  oracle     a model trained on that corner's own data. It is a LOWER BOUND on what a
             better-matched model recovers, not a ceiling: it sees `warm_s` seconds of
             that corner at 20 epochs against the frozen model's 400 k ticks at 60, and
             on the nominal body it scores BELOW the frozen model. `body_params.py`
             publishes that deficit per axis at 500 ms.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from growbot_cerebellum import provenance
from growbot_cerebellum.paths import DATA, RESULTS
from growbot_cerebellum.sim import collect
from growbot_cerebellum.forward import MLP, make_windows, K
from growbot_cerebellum.sim2real import corners, horizon_within, adapt_online


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--warm-s", type=float, nargs="+", default=[10, 60, 300],
                    help="seconds of experience on the new body before measuring")
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--corner-steps", type=int, default=30000, help="ticks collected per corner (600 s)")
    args = ap.parse_args()

    tr = np.load(DATA / "olie_train.npz")
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
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "sim2real_proxy.json").write_text(json.dumps(
        {"rows": rows, "config": vars(args),
         "provenance": provenance(seeds={"corner": "hash(name) % 10000 -- str hash, randomised per process "
                                                   "unless PYTHONHASHSEED is set"})}, indent=1))


if __name__ == "__main__":
    main()
