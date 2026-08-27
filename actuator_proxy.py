"""Sim-to-real proxy, second attempt: perturb the actuator's DYNAMICS, not the body's parameters.

sim2real_proxy.py changed mass, CoM, leg length, servo gain and friction and found
nothing for an online residual to correct: those factors barely reach the IMU at
100 ms. The actuator-net literature (Hwangbo 2019) says what breaks transfer is the
actuator's dynamics -- latency, slew limit, deadband -- and MuJoCo's position actuator
has none of them. sim/growbot_sim.ServoModel adds them between command and PD.

Same protocol as before, on the walk body: forward model trained on the ideal servo,
measured on servo variants; frozen vs online normalised-LMS residual (warm 10/60/300 s)
vs an oracle trained on that variant. Plus the per-axis one-step BIAS, because a
systematic error is what a linear residual can learn and the DR proxy had none.
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
from growbot_cerebellum.sim import ServoModel, collect
from growbot_cerebellum.forward import MLP, make_windows, K
from growbot_cerebellum.sim2real import horizon_within, adapt_online

D2R = np.deg2rad
VARIANTS = [
    ("ideal servo (nominal)", None),
    ("delay 1 tick (20 ms)", dict(delay_ticks=1)),
    ("delay 2 ticks (40 ms)", dict(delay_ticks=2)),
    ("slew 8 rad/s (light load)", dict(slew_rad_s=8.0)),
    ("slew 4 rad/s (heavy load)", dict(slew_rad_s=4.0)),
    ("deadband 2 deg", dict(deadband=D2R(2))),
    ("deadband 4 deg", dict(deadband=D2R(4))),
    ("realistic A: 1t + 8 rad/s + 1 deg", dict(delay_ticks=1, slew_rad_s=8.0, deadband=D2R(1))),
    ("realistic B: 2t + 5 rad/s + 2 deg", dict(delay_ticks=2, slew_rad_s=5.0, deadband=D2R(2))),
]

def bias_by_axis(model, O, A, O2, D):
    X, Y, *_ = make_windows(O, A, O2, D, K)
    e = model.predict(X) - Y
    return {"gyro_roll": float(e[:, 6].mean()), "gyro_pitch": float(e[:, 7].mean()),
            "gyro_yaw": float(e[:, 8].mean()), "rmse_gyro": float(np.sqrt((e[:, 6:] ** 2).mean()))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--warm-s", type=float, nargs="+", default=[10, 60, 300])
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--corner-steps", type=int, default=30000)
    ap.add_argument("--horizon", type=int, default=5)
    args = ap.parse_args()

    tr = np.load("data/train.npz")
    Xtr, Ytr, *_ = make_windows(tr["obs"], tr["act"], tr["next_obs"], tr["done"], K)
    print("training nominal (ideal-servo) forward model...", flush=True)
    nominal = MLP(hidden=128, epochs=args.epochs).fit(Xtr, Ytr)

    rows = []
    hdr = (f"{'servo variant':<36}{'frozen':>8}" + "".join(f"{'adapt ' + str(int(w)) + 's':>11}" for w in args.warm_s)
           + f"{'oracle':>9}{'yaw bias':>10}{'pitch bias':>11}")
    print("\n" + hdr); print("-" * len(hdr))
    t0 = time.time()
    for name, kw in VARIANTS:
        servo = ServoModel(**kw) if kw else None
        O, A, O2, D, _ = collect(args.corner_steps, seed=hash(name) % 10000, body="walk", servo=servo)
        cut = int(max(args.warm_s) * 50); held = slice(cut, None)
        frozen, _ = horizon_within(nominal, O[held], A[held], D[held], h=args.horizon)
        adapted = []
        for w in args.warm_s:
            res = adapt_online(nominal, O, A, O2, D, int(w * 50), args.eta)
            adapted.append(horizon_within(nominal, O[held], A[held], D[held], residual=res, h=args.horizon)[0])
        Xc, Yc, *_ = make_windows(O[:cut], A[:cut], O2[:cut], D[:cut], K)
        oracle, _ = horizon_within(MLP(hidden=128, epochs=20).fit(Xc, Yc), O[held], A[held], D[held], h=args.horizon)
        b = bias_by_axis(nominal, O[held], A[held], O2[held], D[held])
        rows.append({"variant": name, "servo": kw, "frozen": frozen,
                     "adapted": dict(zip([str(w) for w in args.warm_s], adapted)), "oracle": oracle, "bias": b})
        print(f"{name:<36}{frozen*100:>7.1f}%" + "".join(f"{a*100:>10.1f}%" for a in adapted)
              + f"{oracle*100:>8.1f}%{b['gyro_yaw']:>+10.3f}{b['gyro_pitch']:>+11.3f}", flush=True)
    print(f"\n{time.time()-t0:.0f}s")
    non = rows[1:]
    fr = np.array([r["frozen"] for r in non]); orc = np.array([r["oracle"] for r in non])
    print(f"\nnon-ideal variants, mean within-0.2rad @{args.horizon*20}ms:  frozen {fr.mean()*100:.1f}%  "
          + "  ".join(f"adapt {w}s {np.mean([r['adapted'][str(w)] for r in non])*100:.1f}%" for w in args.warm_s)
          + f"  oracle {orc.mean()*100:.1f}%")
    gap = orc - fr
    for w in args.warm_s:
        ad = np.array([r["adapted"][str(w)] for r in non])
        rec = (ad - fr) / np.where(gap > 0.005, gap, np.nan)
        print(f"  gap frozen->oracle recovered by residual after {w:.0f}s: {np.nanmean(rec)*100:.0f}%  (over variants with a gap > 0.5 pt)")
    json.dump({"rows": rows, "config": vars(args)}, open("results/actuator_proxy.json", "w"), indent=1)

if __name__ == "__main__":
    main()
