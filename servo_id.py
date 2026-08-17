"""Identify the servo's dynamics from IMU + commands only, through the frozen forward model.

GrowBot's servos have no position feedback, so Hwangbo-style actuator nets (which
train on measured horn angle) are not directly available. But the forward model
already knows what the body does for a given horn angle. So: propose a servo model
(delay, slew limit, deadband), replay the commanded angles through it to get an
estimated horn angle, feed that to the frozen forward model, and score the one-step
prediction error on a real log. The best hypothesis is the identified servo.

Here the "real" log is the twin with a hidden realistic servo. The recovered
parameters and the held-out gain say whether the idea works before a real log exists.
"""
from __future__ import annotations
import argparse, itertools, json, sys, time
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "sim")
from growbot_sim import ServoModel, collect
from forward import MLP, make_windows
from sim2real_proxy import horizon_within, K


def realized_from_commands(A, D, kw):
    """Replay a candidate servo over commanded angles; reset at episode ends."""
    sv = ServoModel(**kw); out = np.zeros_like(A); sv.reset()
    for i in range(len(A)):
        out[i] = sv(A[i], 1 / 50)
        if D[i]:
            sv.reset()
    return out


def identify(model, O, A, O2, D, grid):
    """Return (sorted [(err, kw)], best kw) by one-step forward error."""
    scores = []
    for d, s, db in grid:
        kw = dict(delay_ticks=d, slew_rad_s=s, deadband=db)
        Rc = realized_from_commands(A, D, kw)
        X, Y, *_ = make_windows(O, Rc, O2, D, K)
        scores.append((float(((model.predict(X) - Y) ** 2).mean()), kw))
    scores.sort(key=lambda x: x[0])
    return scores, scores[0][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--log-steps", type=int, default=30000, help="600 s: first half fits, second half evaluates")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--true-delay", type=int, default=2)
    ap.add_argument("--true-slew", type=float, default=5.0)
    ap.add_argument("--true-deadband-deg", type=float, default=2.0)
    args = ap.parse_args()

    tr = np.load("data/train.npz")
    Xtr, Ytr, *_ = make_windows(tr["obs"], tr["act"], tr["next_obs"], tr["done"], K)
    nominal = MLP(hidden=128, epochs=args.epochs).fit(Xtr, Ytr)

    TRUE = dict(delay_ticks=args.true_delay, slew_rad_s=args.true_slew, deadband=np.deg2rad(args.true_deadband_deg))
    O, A, O2, D, _ = collect(args.log_steps, seed=args.seed, body="walk", servo=ServoModel(**TRUE))
    R_true = collect.last_realized
    half = args.log_steps // 2
    fit, held = slice(0, half), slice(half, None)

    grid = list(itertools.product([0, 1, 2, 3], [3.0, 4.0, 5.0, 6.0, 8.0, None],
                                  [0.0, np.deg2rad(1), np.deg2rad(2), np.deg2rad(4)]))
    t0 = time.time()
    scores, best = identify(nominal, O[fit], A[fit], O2[fit], D[fit], grid)
    ideal_err = [e for e, kw in scores if kw["delay_ticks"] == 0 and kw["slew_rad_s"] is None and kw["deadband"] == 0.0][0]
    print(f"{len(grid)} hypotheses on {half / 50:.0f} s of IMU+commands: {time.time() - t0:.0f}s")
    print("best 5:")
    for e, kw in scores[:5]:
        print(f"  err {e:.4f}  delay {kw['delay_ticks']}  slew {kw['slew_rad_s']}  deadband {np.rad2deg(kw['deadband']):.0f} deg")
    print(f"ideal-servo hypothesis err {ideal_err:.4f}   true: delay {TRUE['delay_ticks']}, slew {TRUE['slew_rad_s']}, "
          f"deadband {args.true_deadband_deg:.0f} deg")

    R_est = realized_from_commands(A, D, best)
    out = {"true": {**TRUE, "deadband": float(TRUE["deadband"])}, "identified": {**best, "deadband": float(best["deadband"])},
           "ideal_err": ideal_err, "best_err": scores[0][0], "held_out": {}}
    print("\nheld-out half, within 0.2 rad:")
    for h in (5, 25):
        c = horizon_within(nominal, O[held], A[held], D[held], h=h)[0]
        e = horizon_within(nominal, O[held], R_est[held], D[held], h=h)[0]
        t = horizon_within(nominal, O[held], R_true[held], D[held], h=h)[0]
        out["held_out"][f"{h * 20}ms"] = {"commanded": c, "identified": e, "true_horn": t}
        print(f"  {h * 20:>3} ms   commanded {c * 100:5.1f}%   identified servo {e * 100:5.1f}%   true horn angle {t * 100:5.1f}%")
    out["mean_abs_horn_err_rad"] = float(np.abs(R_est[held] - R_true[held]).mean())
    json.dump(out, open("results/servo_id.json", "w"), indent=1)


if __name__ == "__main__":
    main()
