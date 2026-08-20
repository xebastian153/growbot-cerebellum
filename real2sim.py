"""Real2Sim loop closure: the identified servo goes back into the twin, the twin
retrains its forward model, and the result is scored on the same real logs.

The real-log report identified the servo as delay 4 ticks (80 ms at the 50 Hz
caller), slew 2.0 rad/s, deadband 2 deg -- but split-half DISAGREED (A: delay 4 /
slew 3.0, B: delay 6 / slew 1.5), so a single point would overclaim. This
experiment therefore runs the WHOLE loop -- collect with the servo in the twin,
train the standard forward model, evaluate on the real logs -- at three points
spanning the determined uncertainty, plus a nominal-servo control through the
identical pipeline.

Decision rule, stated before the numbers: a config's closure on an axis at
500 ms is material when it exceeds max(3.0 pts, 2x the control's seed spread on
that axis). If all three corrected configs close materially on the axes the
actuator explained in attribution (roll, yaw), the Real2Sim loop is validated
robustly to the identification uncertainty; if they diverge, the honest
conclusion is that the gesture capture must land before claiming closure.
Pitch is expected to stay open either way: the actuator explained none of its
gap in the held-out attribution, so an open pitch here confirms the coverage
hypothesis (the tipped walk's sit-to-stand has no counterpart in twin data)
rather than contradicting the loop.

Discipline notes: identification used the FIRST half of the concatenated log,
so every real-log number here is evaluated on the held-out second half only --
the same no-shared-ticks rule as gap_report --servo-id. The corrected twins
train on COMMANDED actions (the horn lags inside the twin), because commanded
angles are all a real log carries. On a slower servo the walk policy's realized
gait changes; that distribution shift is part of the loop, not a bug, and the
realized-motion sanity of each collection is reported as context.
"""
from __future__ import annotations
import json, sys, time
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "sim")

from growbot_sim import ServoModel, collect
from forward import MLP, make_windows
from sim2real_proxy import K
from gap_report import evaluate_axes, AXES
from imulog import parse, run_preflight

LOGS = ["imu-walk-1-2026-08-20T17-50-14-713Z.json",
        "imu-walk-3-2026-08-20T17-38-19-478Z.json"]
TRAIN_STEPS, TEST_STEPS = 400_000, 60_000
TRAIN_SEED, TEST_SEED = 0, 1            # the published data protocol (README)
EPOCHS, HIDDEN = 80, 128                # the published real-log model (real_log_report)
HORIZONS = [5, 25]                      # 100 / 500 ms
TIP_WINDOW_S = 1.5
DB2 = float(np.deg2rad(2))              # argmin deadband, shared by all corrected configs

# delay_ticks count CALLS at GrowBotSim.step's 50 Hz (1 tick = 20 ms)
CONFIGS = {
    "control nominal": None,
    "argmin d4 s2.0 (80 ms, 2.0 rad/s)": dict(delay_ticks=4, slew_rad_s=2.0, deadband=DB2),
    "half-A d4 s3.0 (80 ms, 3.0 rad/s)": dict(delay_ticks=4, slew_rad_s=3.0, deadband=DB2),
    "half-B d6 s1.5 (120 ms, 1.5 rad/s)": dict(delay_ticks=6, slew_rad_s=1.5, deadband=DB2),
}
CONTROL_EXTRA_SEEDS = [1, 2]            # extra MLP seeds on the control, for the spread


class Tee:
    def __init__(self, path):
        self.f = open(path, "w")
    def write(self, s):
        sys.__stdout__.write(s); self.f.write(s)
    def flush(self):
        sys.__stdout__.flush()
        if not self.f.closed:
            self.f.flush()


def load_real_heldout():
    """The concatenated real log, cut per file, tip window labelled, second half."""
    parts = []
    for f in LOGS:
        if not run_preflight(f):
            raise SystemExit(f"preflight FAIL on {f}")
        O, A, O2, D, header, mode = parse(f)
        label = np.array(mode, dtype=object)
        if header.get("end_why") == "tipped":
            label[-int(TIP_WINDOW_S * 50):] = "tip_onset"
        parts.append((O, A, O2, D, label.astype(str)))
    O, A, O2, D, label = (np.concatenate(x) for x in zip(*parts))
    half = len(O) // 2                  # identification used [0:half]; we never touch it
    held = slice(half, None)
    return O[held], A[held], D[held], label[held], half, len(O)


def gait_sanity(A, R, O, mode):
    """Does the policy still walk on this servo? Realized-motion context numbers."""
    pol = mode == "policy"
    track = float(np.abs(R[pol] - A[pol]).mean())
    return {"policy_ticks": int(pol.sum()),
            "horn_tracking_err_rad": round(track, 4),
            "cmd_amplitude_rad": round(float(np.abs(A[pol]).mean()), 3),
            "horn_amplitude_rad": round(float(np.abs(R[pol]).mean()), 3),
            "gyro_norm_mean": round(float(np.linalg.norm(O[pol, 3:], axis=1).mean()), 3),
            "fallen_frac": round(float(((np.abs(O[pol, 0]) > 1.2) |
                                        (np.abs(O[pol, 1]) > 1.2)).mean()), 4)}


def main():
    sys.stdout = tee = Tee("results/logs/real2sim.txt")
    t0 = time.time()
    Oh, Ah, Dh, labelh, half, total = load_real_heldout()
    print(f"\nreal log: {total} ticks concatenated, identification half = [0:{half}], "
          f"evaluation = held-out [{half}:{total}] ({len(Oh)} ticks, {len(Oh)/50:.1f} s)")

    report = {"conditions": {
        "logs": LOGS, "train_steps": TRAIN_STEPS, "test_steps": TEST_STEPS,
        "train_seed": TRAIN_SEED, "test_seed": TEST_SEED, "epochs": EPOCHS,
        "hidden": HIDDEN, "K": K, "horizons_ms": [h * 20 for h in HORIZONS],
        "tip_window_s": TIP_WINDOW_S, "held_out_ticks": int(len(Oh)),
        "deadband_deg": 2.0,
        "delay_unit": "ticks at the 50 Hz caller (1 tick = 20 ms)",
        "rule": "material closure @500ms = gain over control > max(3.0 pts, 2x control seed spread)",
    }, "configs": {}}

    control_train = None
    control_real_by_seed = {}
    for name, kw in CONFIGS.items():
        print(f"\n== {name}")
        servo_tr = ServoModel(**kw) if kw else None
        servo_te = ServoModel(**kw) if kw else None
        O, A, O2, D, M, R = collect(TRAIN_STEPS, TRAIN_SEED, servo=servo_tr,
                                    return_realized=True)
        te = {}
        te["obs"], te["act"], te["next_obs"], te["done"], te["mode"] = collect(
            TEST_STEPS, TEST_SEED, servo=servo_te)
        sanity = gait_sanity(A, R, O, M)
        print(f"  gait sanity (policy mode): {sanity}")
        if kw is None:
            # the identical pipeline must reproduce the published data exactly
            tr_pub = np.load("data/train.npz")
            assert np.array_equal(O, tr_pub["obs"]) and np.array_equal(A, tr_pub["act"]), \
                "control collection does not reproduce data/train.npz"
            print("  control collection asserted equal to data/train.npz")
            control_train = (O, A, O2, D)

        Xtr, Ytr, *_ = make_windows(O, A, O2, D, K)
        model = MLP(hidden=HIDDEN, epochs=EPOCHS, seed=0).fit(Xtr, Ytr)
        floor = evaluate_axes(model, te["obs"], te["act"], te["done"],
                              te["mode"].astype(str), HORIZONS)
        real = evaluate_axes(model, Oh, Ah, Dh, labelh, HORIZONS)
        if kw is None:
            control_real_by_seed[0] = real
        cfg = {"servo": kw, "gait_sanity": sanity,
               "twin_floor_policy": {str(h * 20): {ax: floor["policy"][h][ax]["within"]
                                                   for ax in AXES} for h in HORIZONS},
               "real_heldout_all": {str(h * 20): {ax: real["all"][h][ax]["within"]
                                                  for ax in AXES} for h in HORIZONS}}
        report["configs"][name] = cfg
        for h in HORIZONS:
            row = "  ".join(f"{ax} {real['all'][h][ax]['within']*100:5.1f} "
                            f"(floor {floor['policy'][h][ax]['within']*100:5.1f})"
                            for ax in AXES)
            print(f"  real held-out @{h*20:>3}ms  {row}")

    print("\n== control seed spread (extra MLP seeds on the control data)")
    O, A, O2, D = control_train
    Xtr, Ytr, *_ = make_windows(O, A, O2, D, K)
    for s in CONTROL_EXTRA_SEEDS:
        m = MLP(hidden=HIDDEN, epochs=EPOCHS, seed=s).fit(Xtr, Ytr)
        control_real_by_seed[s] = evaluate_axes(m, Oh, Ah, Dh, labelh, HORIZONS)
    spread = {}
    for ax in AXES:
        vals = [control_real_by_seed[s]["all"][25][ax]["within"]
                for s in control_real_by_seed]
        spread[ax] = float(max(vals) - min(vals))
        print(f"  {ax}: seeds {['%.1f' % (v*100) for v in vals]} -> spread {spread[ax]*100:.1f} pts")
    report["control_seed_spread_500ms"] = spread

    print("\n== closure @500 ms vs control (within 0.2 rad; threshold = "
          "max(3.0, 2x spread) pts)")
    ctrl = report["configs"]["control nominal"]["real_heldout_all"]["500"]
    verdict = {}
    for name, cfg in report["configs"].items():
        if name == "control nominal":
            continue
        row, v = "", {}
        for ax in AXES:
            gain = cfg["real_heldout_all"]["500"][ax] - ctrl[ax]
            thr = max(0.03, 2 * spread[ax])
            v[ax] = {"gain_pts": round(gain * 100, 1), "threshold_pts": round(thr * 100, 1),
                     "material": bool(gain > thr)}
            row += f"  {ax} {gain*100:+5.1f} (thr {thr*100:.1f}){' MATERIAL' if gain > thr else ''}"
        verdict[name] = v
        print(f"  {name:<38}{row}")
    report["closure_verdict"] = verdict

    mat = {ax: [v[ax]["material"] for v in verdict.values()] for ax in AXES}
    if all(mat["roll"]) or all(mat["yaw"]):
        axes_ok = [ax for ax in ("roll", "yaw") if all(mat[ax])]
        conclusion = (f"loop VALIDATED robustly to the identification uncertainty on "
                      f"{'+'.join(axes_ok)}: every config in the determined band closes "
                      f"materially there")
    elif any(any(v[ax]["material"] for ax in AXES) for v in verdict.values()):
        conclusion = ("MIXED: closure depends on where in the determined band the servo "
                      "sits -- the gesture capture must land before claiming closure")
    else:
        conclusion = ("NOT CLOSED: retraining on the identified servo does not transfer "
                      "its held-out replay gains -- the gesture capture must land first")
    if not any(v["pitch"]["material"] for v in verdict.values()):
        conclusion += ("; pitch stays open in every config, as the attribution predicted "
                       "-- the coverage hypothesis (sit-to-stand) stands")
    report["conclusion"] = conclusion
    print(f"\n  {conclusion}")

    json.dump(report, open("results/real2sim.json", "w"), indent=1)
    print(f"\nwrote results/real2sim.json   total {(time.time()-t0)/60:.1f} min")
    tee.f.close()


if __name__ == "__main__":
    main()
