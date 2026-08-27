"""When the real servo is OUTSIDE the family servo_id searches, what does it say?

servo_id.py assumes the servo is a (delay, slew, deadband) triple. A real MG90S
under load is not quite: slew falls as torque demand rises, and a sagging battery
drags the whole servo slower over a session. This script hides two such
out-of-family servos inside the twin, runs the exact day-of-log identification
against them, and asks two questions:

  1. Honesty: does identification fail loudly (split-half DISAGREE, wide
     determined sets) or produce a tight set around a wrong answer? The
     diagnostics servo_id prints are the only warning a real log would give;
     this measures whether they fire when the family is wrong.
  2. Fallback: how much of the remaining gap does a residual model close --
     fit on the identification half, on top of the identified servo, from
     signals a real log provides (IMU + commands only)?

The in-family servo runs through the identical pipeline as control, same seed.
Identification uses the first half of the log; every reported gap number comes
from the second half only (the gap* circularity rule).
"""
from __future__ import annotations
import argparse, itertools, json, time
import numpy as np
from growbot_cerebellum import provenance
from growbot_cerebellum.sim import ServoModel, collect
from growbot_cerebellum.forward import MLP, make_windows
from growbot_cerebellum.servo_id import identify, realized_from_commands, confidence_band, determined_sets
from growbot_cerebellum.forward import K
from growbot_cerebellum.sim2real import horizon_within


class LoadDependentServo(ServoModel):
    """Slew limit that falls with tracking error, a proxy for load-dependent torque.

    slew_eff = slew0 / (1 + k_load * |target - pos|), per servo. Small corrections
    move at slew0; a large swing (high torque demand) crawls. No single grid slew
    reproduces it: the effective limit spans ~2.7-8 rad/s across the operating range.
    """
    def __init__(self, delay_ticks=0, slew0=8.0, k_load=2.0, deadband=0.0):
        super().__init__(delay_ticks=delay_ticks, slew_rad_s=None, deadband=deadband)
        self.slew0, self.k_load = float(slew0), float(k_load)

    def __call__(self, target, dt):
        self.q.append(np.array(target, np.float32))
        err = self.q[0] - self.pos
        if self.db > 0:
            err = np.where(np.abs(err) < self.db, 0.0, err)
        lim = self.slew0 / (1.0 + self.k_load * np.abs(err)) * dt
        self.pos = self.pos + np.clip(err, -lim, lim)
        return self.pos


class VoltageSagServo(ServoModel):
    """Slew that drifts linearly over the session (fresh -> low battery).

    Violates the stationarity every grid hypothesis assumes. The age counter
    survives episode resets on purpose: batteries do not recharge between falls.
    """
    def __init__(self, delay_ticks=0, slew_start=6.0, slew_end=3.0, total_calls=30000, deadband=0.0):
        super().__init__(delay_ticks=delay_ticks, slew_rad_s=float(slew_start), deadband=deadband)
        self.s0, self.s1, self.total, self.age = float(slew_start), float(slew_end), int(total_calls), 0

    def __call__(self, target, dt):
        self.slew = self.s0 + (self.s1 - self.s0) * min(self.age / self.total, 1.0)
        self.age += 1
        return super().__call__(target, dt)


class LinearResidual:
    """Ridge least-squares map from the model's input window to its one-step error."""
    name = "linear"
    def fit(self, X, E, ridge=1e-3):
        Xb = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)
        A = Xb.T @ Xb + ridge * np.eye(Xb.shape[1], dtype=np.float32)
        A[-1, -1] -= ridge
        self.W = np.linalg.solve(A, Xb.T @ E)
        return self
    def correct(self, X):
        return np.concatenate([X, np.ones((len(X), 1), np.float32)], 1) @ self.W


class MLPResidual:
    """Small nonlinear residual; same interface, only tried when linear falls short."""
    name = "mlp"
    def __init__(self, hidden=64, epochs=20):
        self.m = MLP(hidden=hidden, layers=2, epochs=epochs)
    def fit(self, X, E):
        self.m.fit(X, E); return self
    def correct(self, X):
        return self.m.predict(X)


def extend_cuts(D, delay):
    D_ext = D.copy()
    for j in range(1, delay + 1):
        D_ext[j:] |= D[:-j]
    return D_ext


def fit_residual(cls, model, O, A_used, O2, D, delay):
    """Residual on the identified twin's one-step error, identification half only."""
    X, Y, *_ = make_windows(O, A_used, O2, extend_cuts(D, delay), K)
    E = Y - model.predict(X)
    return cls().fit(X, E)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--log-steps", type=int, default=30000, help="600 s: first half identifies, second half evaluates")
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    tr = np.load("data/train.npz")
    Xtr, Ytr, *_ = make_windows(tr["obs"], tr["act"], tr["next_obs"], tr["done"], K)
    print("training nominal forward model...", flush=True)
    nominal = MLP(hidden=128, epochs=args.epochs).fit(Xtr, Ytr)

    db2 = float(np.deg2rad(2.0))
    variants = [
        ("in-family control (delay 2, slew 5, db 2 deg)",
         lambda: ServoModel(delay_ticks=2, slew_rad_s=5.0, deadband=db2)),
        ("load-dependent slew (delay 2, slew0 8 / (1 + 2|err|), db 2 deg)",
         lambda: LoadDependentServo(delay_ticks=2, slew0=8.0, k_load=2.0, deadband=db2)),
        ("voltage sag (delay 2, slew 6 -> 3 over the session, db 2 deg)",
         lambda: VoltageSagServo(delay_ticks=2, slew_start=6.0, slew_end=3.0,
                                 total_calls=args.log_steps, deadband=db2)),
    ]
    grid = list(itertools.product([0, 1, 2, 3], [3.0, 4.0, 5.0, 6.0, 8.0, None],
                                  [0.0, np.deg2rad(1), np.deg2rad(2), np.deg2rad(4)]))
    out = {"config": {"seed": args.seed, "log_steps": args.log_steps, "epochs": args.epochs,
                      "grid": "delay {0..3} ticks x slew {3,4,5,6,8,none} rad/s x deadband {0,1,2,4} deg",
                      "n_starts": 1500, "horizons_ms": [100, 500]}, "variants": {}}

    for name, mk in variants:
        print(f"\n=== {name} ===", flush=True)
        O, A, O2, D, _, R_true = collect(args.log_steps, seed=args.seed, body="walk",
                                         servo=mk(), return_realized=True)
        half = args.log_steps // 2
        fit, held = slice(0, half), slice(half, None)

        t0 = time.time()
        scores, best = identify(nominal, O[fit], A[fit], O2[fit], D[fit], grid)
        print(f"{len(grid)} hypotheses on {half / 50:.0f} s: {time.time() - t0:.0f}s   best 3:")
        for e, kw in scores[:3]:
            print(f"  err {e:.4f}  delay {kw['delay_ticks']}  slew {kw['slew_rad_s']}  "
                  f"deadband {np.rad2deg(kw['deadband']):.0f} deg")

        # the honesty diagnostics, exactly as the day-of-log chain prints them
        halfA, halfB = slice(0, half // 2), slice(half // 2, half)
        scoresA, bestA = identify(nominal, O[halfA], A[halfA], O2[halfA], D[halfA], grid)
        scoresB, bestB = identify(nominal, O[halfB], A[halfB], O2[halfB], D[halfB], grid)
        agree = bestA["delay_ticks"] == bestB["delay_ticks"] and bestA["slew_rad_s"] == bestB["slew_rad_s"]
        print(f"split-half stability: A=(delay {bestA['delay_ticks']}, slew {bestA['slew_rad_s']})  "
              f"B=(delay {bestB['delay_ticks']}, slew {bestB['slew_rad_s']})  "
              f"{'AGREE' if agree else 'DISAGREE -- log too short or servo outside the model family'}")
        band = confidence_band(scoresA, scoresB)
        delay_set, slew_set = determined_sets(scores, best, grid, band)
        print(f"delay set {delay_set}  slew set {slew_set}  (band ±{band:.4f})")
        held_scores, _ = identify(nominal, O[held], A[held], O2[held], D[held],
                                  [(best["delay_ticks"], best["slew_rad_s"], best["deadband"])])
        print(f"one-step err: fit {scores[0][0]:.4f}  held-out {held_scores[0][0]:.4f}")

        # gap attribution, held-out half only
        R_est = realized_from_commands(A, D, best)
        res = fit_residual(LinearResidual, nominal, O[fit], R_est[fit], O2[fit], D[fit], best["delay_ticks"])
        rows = {}
        for h in (5, 25):
            c, _ = horizon_within(nominal, O[held], A[held], D[held], h=h)
            e, _ = horizon_within(nominal, O[held], R_est[held], D[held], h=h)
            r, _ = horizon_within(nominal, O[held], R_est[held], D[held], residual=res, h=h)
            t, _ = horizon_within(nominal, O[held], R_true[held], D[held], h=h)
            rows[f"{h * 20}ms"] = {"commanded": c, "identified": e,
                                   "identified_plus_residual": r, "true_horn": t}
            print(f"  {h * 20:>3} ms within 0.2 rad:  commanded {c * 100:5.1f}%   identified {e * 100:5.1f}%   "
                  f"+residual {r * 100:5.1f}%   true horn {t * 100:5.1f}%")
        residual_kind = "linear"
        closable = rows["500ms"]["true_horn"] - rows["500ms"]["identified"]
        gained = rows["500ms"]["identified_plus_residual"] - rows["500ms"]["identified"]
        if closable > 0.01 and gained < 0.5 * closable:
            print("  linear residual closed under half the closable gap; trying a small MLP residual")
            res2 = fit_residual(MLPResidual, nominal, O[fit], R_est[fit], O2[fit], D[fit], best["delay_ticks"])
            for h in (5, 25):
                r2, _ = horizon_within(nominal, O[held], R_est[held], D[held], residual=res2, h=h)
                rows[f"{h * 20}ms"]["identified_plus_mlp_residual"] = r2
                print(f"  {h * 20:>3} ms +MLP residual {r2 * 100:5.1f}%")
            residual_kind = "linear+mlp"

        out["variants"][name] = {
            "identified": {**best, "deadband": float(best["deadband"])},
            "split_half": {"A": {"delay": bestA["delay_ticks"], "slew": bestA["slew_rad_s"]},
                           "B": {"delay": bestB["delay_ticks"], "slew": bestB["slew_rad_s"]},
                           "agree": agree},
            "confidence_band": band, "delay_determined_set": delay_set,
            "slew_determined_set": list(slew_set),
            "one_step_err": {"fit": scores[0][0], "held_out": held_scores[0][0]},
            "mean_abs_horn_err_rad": float(np.abs(R_est[held] - R_true[held]).mean()),
            "residual": residual_kind, "held_out": rows,
        }

    out["provenance"] = provenance(seeds=args.seed)
    json.dump(out, open("results/model_mismatch.json", "w"), indent=1)
    print("\nsaved results/model_mismatch.json")


if __name__ == "__main__":
    main()
