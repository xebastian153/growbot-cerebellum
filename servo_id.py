"""Identify the servo's dynamics from IMU + commands only, through the frozen forward model.

GrowBot's servos have no position feedback, so Hwangbo-style actuator nets (which
train on measured horn angle) are not directly available. But the forward model
already knows what the body does for a given horn angle. So: propose a servo model
(delay, slew limit, deadband), replay the commanded angles through it to get an
estimated horn angle, feed that to the frozen forward model, and score the one-step
prediction error on a real log. The best hypothesis is the identified servo.

Here the "real" log is the twin with a hidden realistic servo. The recovered
parameters and the held-out gain say whether the idea works before a real log exists.
The identification itself -- `identify`, `confidence_band`, `determined_sets`,
`default_grid`, `argmin_interior`, the per-side variants -- is `growbot_cerebellum.servo_id`.
"""
from __future__ import annotations
import argparse, json, time
import numpy as np

from growbot_cerebellum import provenance
from growbot_cerebellum.paths import DATA, RESULTS
from growbot_cerebellum.sim import ServoModel, collect
from growbot_cerebellum.forward import MLP, make_windows, K
from growbot_cerebellum.sim2real import horizon_within
from growbot_cerebellum.servo_id import (identify, realized_from_commands, confidence_band, determined_sets,
                                         default_grid, argmin_interior)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--log-steps", type=int, default=30000, help="600 s: first half fits, second half evaluates")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--true-delay", type=int, default=2)
    ap.add_argument("--true-slew", type=float, default=5.0)
    ap.add_argument("--true-deadband-deg", type=float, default=2.0)
    args = ap.parse_args()

    tr = np.load(DATA / "train.npz")
    Xtr, Ytr, *_ = make_windows(tr["obs"], tr["act"], tr["next_obs"], tr["done"], K)
    nominal = MLP(hidden=128, epochs=args.epochs).fit(Xtr, Ytr)

    TRUE = dict(delay_ticks=args.true_delay, slew_rad_s=args.true_slew, deadband=np.deg2rad(args.true_deadband_deg))
    O, A, O2, D, _, R_true = collect(args.log_steps, seed=args.seed, body="walk",
                                     servo=ServoModel(**TRUE), return_realized=True)
    half = args.log_steps // 2
    fit, held = slice(0, half), slice(half, None)

    grid = default_grid()
    t0 = time.time()
    scores, best = identify(nominal, O[fit], A[fit], O2[fit], D[fit], grid)
    ideal_err = [e for e, kw in scores if kw["delay_ticks"] == 0 and kw["slew_rad_s"] is None and kw["deadband"] == 0.0][0]
    print(f"{len(grid)} hypotheses on {half / 50:.0f} s of IMU+commands: {time.time() - t0:.0f}s")
    print("grid resolution: delay in 20 ms steps, slew from an enumerated set, deadband 0-4 deg;")
    print("identified values are grid points, not continuous estimates")
    print("best 5:")
    for e, kw in scores[:5]:
        print(f"  err {e:.4f}  delay {kw['delay_ticks']}  slew {kw['slew_rad_s']}  deadband {np.rad2deg(kw['deadband']):.0f} deg")
    print(f"ideal-servo hypothesis err {ideal_err:.4f}   true: delay {TRUE['delay_ticks']}, slew {TRUE['slew_rad_s']}, "
          f"deadband {args.true_deadband_deg:.0f} deg")
    interior, interior_why = argmin_interior(best, grid)
    print(f"  {interior_why}")

    # diagnostics for the day the real servo leaves the model family ------------
    halfA, halfB = slice(0, half // 2), slice(half // 2, half)
    scoresA, bestA = identify(nominal, O[halfA], A[halfA], O2[halfA], D[halfA], grid)
    scoresB, bestB = identify(nominal, O[halfB], A[halfB], O2[halfB], D[halfB], grid)
    agree = bestA["delay_ticks"] == bestB["delay_ticks"] and bestA["slew_rad_s"] == bestB["slew_rad_s"]
    print(f"split-half stability: A=(delay {bestA['delay_ticks']}, slew {bestA['slew_rad_s']})  "
          f"B=(delay {bestB['delay_ticks']}, slew {bestB['slew_rad_s']})  "
          f"{'AGREE' if agree else 'DISAGREE -- log too short or servo outside the model family'}")
    band = confidence_band(scoresA, scoresB)
    delay_set, slew_set = determined_sets(scores, best, grid, band)
    def show(name, sset, unit=""):
        if len(sset) == 1:
            print(f"{name} determined: {sset[0]}{unit}")
        else:
            print(f"{name} ∈ {{{', '.join(str(v) for v in sset)}}}{unit} -- separation below the "
                  f"confidence band (±{band:.4f}); report the set, not the argmin")
    show("delay", delay_set, " ticks")
    show("slew", slew_set, " rad/s")
    held_scores, _ = identify(nominal, O[held], A[held], O2[held], D[held],
                              [(best["delay_ticks"], best["slew_rad_s"], best["deadband"]),
                               (0, None, 0.0)])
    by_kw = {(kw["delay_ticks"], kw["slew_rad_s"]): e for e, kw in held_scores}
    print(f"held-out one-step err: best {by_kw[(best['delay_ticks'], best['slew_rad_s'])]:.4f}  "
          f"ideal {by_kw[(0, None)]:.4f}  (fit: best {scores[0][0]:.4f}  ideal {ideal_err:.4f}; "
          f"divergence between fit and held-out = fitting noise)")
    slew_family = sorted((e, kw["slew_rad_s"]) for e, kw in scores
                         if kw["delay_ticks"] == best["delay_ticks"] and kw["deadband"] == best["deadband"])
    print("slew separability at the identified delay: "
          + "  ".join(f"{sl}:{e:.4f}" for e, sl in slew_family)
          + "   (near-ties here mean the excitation never hit the slew limit)")

    R_est = realized_from_commands(A, D, best)
    out = {"true": {**TRUE, "deadband": float(TRUE["deadband"])}, "identified": {**best, "deadband": float(best["deadband"])},
           "ideal_err": ideal_err, "best_err": scores[0][0],
           "split_half_agree": agree, "confidence_band": band, "argmin_interior": interior,
           "delay_determined_set": delay_set, "slew_determined_set": [v for v in slew_set],
           "held_out_err": {"best": by_kw[(best["delay_ticks"], best["slew_rad_s"])], "ideal": by_kw[(0, None)]},
           "held_out": {}}
    print("\nheld-out half, within 0.2 rad:")
    for h in (5, 25):
        c = horizon_within(nominal, O[held], A[held], D[held], h=h)[0]
        e = horizon_within(nominal, O[held], R_est[held], D[held], h=h)[0]
        t = horizon_within(nominal, O[held], R_true[held], D[held], h=h)[0]
        out["held_out"][f"{h * 20}ms"] = {"commanded": c, "identified": e, "true_horn": t}
        print(f"  {h * 20:>3} ms   commanded {c * 100:5.1f}%   identified servo {e * 100:5.1f}%   true horn angle {t * 100:5.1f}%")
    out["mean_abs_horn_err_rad"] = float(np.abs(R_est[held] - R_true[held]).mean())
    out["provenance"] = provenance(seeds=args.seed)
    json.dump(out, open(RESULTS / "servo_id.json", "w"), indent=1)


if __name__ == "__main__":
    main()
