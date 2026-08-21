"""Real2Sim loop closure: the identified servo goes back into the twin, the twin
retrains its forward model, and the result is scored on the real walking log.

ONE FILE. The earlier version of this experiment scored a concatenation of walk-1
and walk-3, and roughly half of that held-out slice was walk-3 -- a motionless
body under swinging commands followed by a fall. Retraining a twin to predict a
robot that is not moving is not loop closure, and the numbers it produced are
withdrawn. walk-3 also runs at a different agent gain and rests at a different
attitude, so it is a different experiment however it is scored. Here the
identification and the evaluation are both walk-1, first half and held-out second
half, and only its WALKING segment carries the walking claim.

What is being tested, and what it cannot test. servo_id leaves the delay
completely undetermined on this log (the determined set is the entire grid,
0-6 ticks) and the slew determined only to a band. A "corrected" config therefore
changes delay, slew AND deadband at once against a control that has none of them,
so a gain over the control says "some actuator model helps", not "this actuator
model is right". The SMOOTHING-ONLY cell below is the one lever that separates
them: delay 0, slew and deadband at the argmin values. It sits inside the
determined band, so it is a legitimate hypothesis about the same servo, and it
carries no latency at all. If it closes as much as the delayed configs, the
closure is attributable to action smoothing and the delay is doing nothing that
this log can see.

Decision rule, stated before the numbers: a config's closure on an axis at 500 ms
is material when it exceeds max(3.0 pts, 2x the control's seed spread on that
axis). The verdict text is COMPUTED from which cells pass and from how much of the
determined band the tested configs actually cover -- the previous version emitted
"robust to the identification uncertainty" from three sampled points out of a
seven-wide delay set, which is a claim about the band made from a sample of it.

Discipline notes: identification used the FIRST half of walk-1, so every real-log
number here is evaluated on the held-out second half only -- the same
no-shared-ticks rule as gap_report --servo-id. The corrected twins train on
COMMANDED actions (the horn lags inside the twin), because commanded angles are
all a real log carries. On a slower servo the walk policy's realized gait changes;
that distribution shift is part of the loop, not a bug, and the realized-motion
sanity of each collection is reported as context.
"""
from __future__ import annotations
import json, sys, time
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "sim")

from growbot_sim import ServoModel, collect
from forward import MLP, make_windows
from sim2real_proxy import K
from gap_report import evaluate_axes, twin_regimes, REGIME_MAP, AXES
from imulog import parse, run_preflight

LOG = "imu-walk-1-2026-08-20T17-50-14-713Z.json"     # the only real file that walks
EXCLUDED = {"imu-walk-3-2026-08-20T17-38-19-478Z.json":
            "no walking segment (2.6 s motionless under active commands, then a fall), "
            "different agent gain (0.8 vs null) and a different resting attitude"}
TRAIN_STEPS, TEST_STEPS = 400_000, 60_000
TRAIN_SEED, TEST_SEED = 0, 1            # the published data protocol (README)
EPOCHS, HIDDEN = 80, 128                # the published real-log model (real_log_report)
HORIZONS = [5, 25]                      # 100 / 500 ms
DB2 = float(np.deg2rad(2))              # argmin deadband, shared by all corrected configs

# The determined sets real_log_report.py reports for walk-1 alone. Written down here
# so the coverage arithmetic below is against the identification's own output and not
# against whichever points happened to be tried.
DETERMINED_DELAY = [0, 1, 2, 3, 4, 5, 6]
DETERMINED_SLEW = [1.5, 2.0, 3.0, 4.0]

# delay_ticks count CALLS at GrowBotSim.step's 50 Hz (1 tick = 20 ms)
CONFIGS = {
    "control nominal": None,
    "argmin d5 s2.0 (100 ms, 2.0 rad/s)": dict(delay_ticks=5, slew_rad_s=2.0, deadband=DB2),
    "half-A d6 s5.0 (120 ms, 5.0 rad/s)": dict(delay_ticks=6, slew_rad_s=5.0, deadband=DB2),
    "half-B d6 s2.0 (120 ms, 2.0 rad/s)": dict(delay_ticks=6, slew_rad_s=2.0, deadband=DB2),
    "smoothing only d0 s2.0 (no delay)": dict(delay_ticks=0, slew_rad_s=2.0, deadband=DB2),
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
    """walk-1, segment-labelled by imulog's segmenter, held-out second half."""
    if not run_preflight(LOG):
        raise SystemExit(f"preflight FAIL on {LOG}")
    O, A, O2, D, header, mode = parse(LOG)
    half = len(O) // 2                  # identification used [0:half]; we never touch it
    held = slice(half, None)
    return O[held], A[held], D[held], mode[held], half, len(O), header


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


def band_coverage(configs):
    """How much of the determined (delay, slew) band the tested configs actually visit."""
    tested = [(kw["delay_ticks"], kw["slew_rad_s"]) for kw in configs.values() if kw]
    d = sorted({t[0] for t in tested}); s = sorted({t[1] for t in tested})
    return {"delay_tested": d, "delay_determined": DETERMINED_DELAY,
            "delay_covered": bool(set(DETERMINED_DELAY) <= set(d)),
            "delay_fraction": round(len(set(d) & set(DETERMINED_DELAY)) / len(DETERMINED_DELAY), 2),
            "slew_tested": s, "slew_determined": DETERMINED_SLEW,
            "slew_covered": bool(set(DETERMINED_SLEW) <= set(s)),
            "slew_fraction": round(len(set(s) & set(DETERMINED_SLEW)) / len(DETERMINED_SLEW), 2)}


def main():
    sys.stdout = tee = Tee("results/logs/real2sim.txt")
    t0 = time.time()
    Oh, Ah, Dh, labelh, half, total, header = load_real_heldout()
    segs = {m: int((labelh == m).sum()) for m in sorted(set(labelh))}
    print(f"\nreal log: {LOG}")
    for f, why in EXCLUDED.items():
        print(f"  EXCLUDED {f}: {why}")
    print(f"  {total} ticks, identification half = [0:{half}], evaluation = held-out "
          f"[{half}:{total}] ({len(Oh)} ticks, {len(Oh) / 50:.1f} s), segments {segs}")

    cover = band_coverage(CONFIGS)
    print(f"\ndetermined band from walk-1 alone: delay {DETERMINED_DELAY} ticks, "
          f"slew {DETERMINED_SLEW} rad/s")
    print(f"  tested delays {cover['delay_tested']} -> {cover['delay_fraction'] * 100:.0f}% of the "
          f"delay set; tested slews {cover['slew_tested']} -> {cover['slew_fraction'] * 100:.0f}% "
          f"of the slew set")

    report = {"conditions": {
        "log": LOG, "excluded": EXCLUDED,
        "train_steps": TRAIN_STEPS, "test_steps": TEST_STEPS,
        "train_seed": TRAIN_SEED, "test_seed": TEST_SEED, "epochs": EPOCHS,
        "hidden": HIDDEN, "K": K, "horizons_ms": [h * 20 for h in HORIZONS],
        "held_out_ticks": int(len(Oh)), "held_out_segments": segs,
        "deadband_deg": 2.0,
        "delay_unit": "ticks at the 50 Hz caller (1 tick = 20 ms)",
        "rule": "material closure @500ms = gain over control > max(3.0 pts, 2x control seed spread)",
    }, "band_coverage": cover, "configs": {}}

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
        tw_mode, _ = twin_regimes(te["obs"], te["mode"].astype(str))
        floor = evaluate_axes(model, te["obs"], te["act"], te["done"], tw_mode, HORIZONS)
        real = evaluate_axes(model, Oh, Ah, Dh, labelh, HORIZONS)
        if kw is None:
            control_real_by_seed[0] = real
        cfg = {"servo": kw, "gait_sanity": sanity,
               "twin_floor_policy": {str(h * 20): {ax: floor["policy"][h][ax]["within"]
                                                   for ax in AXES} for h in HORIZONS},
               "real_heldout": {reg: {str(h * 20): {ax: real[reg][h][ax]["within"]
                                                    for ax in AXES} for h in HORIZONS}
                                for reg in real}}
        for reg in real:
            cfg["real_heldout"][reg]["n"] = real[reg]["n"]
        report["configs"][name] = cfg
        for reg in real:
            for h in HORIZONS:
                row = "  ".join(f"{ax} {real[reg][h][ax]['within'] * 100:5.1f}" for ax in AXES)
                fl = REGIME_MAP.get(reg, "all") if reg != "all" else "all"
                fl = fl if fl in floor else "all"
                frow = "  ".join(f"{floor[fl][h][ax]['within'] * 100:5.1f}" for ax in AXES)
                print(f"  real held-out {reg:<8} @{h * 20:>3}ms  {row}   (twin {fl}: {frow})")

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
        print(f"  {ax}: seeds {['%.1f' % (v * 100) for v in vals]} -> spread {spread[ax] * 100:.1f} pts")
    report["control_seed_spread_500ms"] = spread

    print("\n== closure @500 ms vs control (all row; within 0.2 rad; threshold = "
          "max(3.0, 2x spread) pts)")
    ctrl = report["configs"]["control nominal"]["real_heldout"]["all"]["500"]
    verdict = {}
    for name, cfg in report["configs"].items():
        if name == "control nominal":
            continue
        row, v = "", {}
        for ax in AXES:
            gain = cfg["real_heldout"]["all"]["500"][ax] - ctrl[ax]
            thr = max(0.03, 2 * spread[ax])
            v[ax] = {"gain_pts": round(gain * 100, 1), "threshold_pts": round(thr * 100, 1),
                     "material": bool(gain > thr)}
            row += f"  {ax} {gain * 100:+5.1f} (thr {thr * 100:.1f}){' MATERIAL' if gain > thr else ''}"
        verdict[name] = v
        print(f"  {name:<38}{row}")
    report["closure_verdict"] = verdict

    # ---- the conclusion, computed ---------------------------------------------------
    smooth_name = next(n for n in CONFIGS if n.startswith("smoothing only"))
    delayed = [n for n in verdict if n != smooth_name]
    closes = {ax: [n for n in verdict if verdict[n][ax]["material"]] for ax in AXES}
    axes_all = [ax for ax in AXES if len(closes[ax]) == len(verdict)]
    axes_some = [ax for ax in AXES if closes[ax] and ax not in axes_all]
    axes_none = [ax for ax in AXES if not closes[ax]]

    parts = []
    if axes_all:
        parts.append(f"closure holds at EVERY tested point on {'+'.join(axes_all)}")
    if axes_some:
        parts.append(f"closure holds at some but not all tested points on "
                     f"{'+'.join(axes_some)} ({ {ax: closes[ax] for ax in axes_some} })")
    if axes_none:
        parts.append(f"no tested point closes {'+'.join(axes_none)}")
    if not cover["delay_covered"] or not cover["slew_covered"]:
        parts.append(f"this says nothing about the BAND: the tested configs visit "
                     f"{cover['delay_fraction'] * 100:.0f}% of the determined delay set and "
                     f"{cover['slew_fraction'] * 100:.0f}% of the determined slew set, so "
                     f"'robust to the identification uncertainty' is not a claim these cells "
                     f"support")
    else:
        parts.append("the tested configs cover the whole determined band")

    # What the smoothing-only cell shows. Every comparison below is made against the
    # SAME materiality threshold the cells themselves are judged by: a difference
    # smaller than the threshold is not a difference, whichever direction it points.
    sm = verdict[smooth_name]
    for ax in AXES:
        gains_delayed = {n: verdict[n][ax]["gain_pts"] for n in delayed}
        best = max(gains_delayed.values()); worst = min(gains_delayed.values())
        n_close = sum(1 for n in delayed if verdict[n][ax]["material"])
        sm_gain = sm[ax]["gain_pts"]
        thr = sm[ax]["threshold_pts"]
        span = (f"the {len(delayed)} delayed cells span {worst:+.1f} to {best:+.1f} pts "
                f"({n_close} of {len(delayed)} material)")
        if best - sm_gain <= thr:
            parts.append(f"on {ax} the zero-delay smoothing-only cell closes {sm_gain:+.1f} pts and "
                         f"the best delayed cell {best:+.1f} -- a difference of {best - sm_gain:+.1f}, "
                         f"INSIDE the {thr:.1f}-pt threshold, so this log does not separate "
                         f"'the identified dynamics' from 'any action smoothing' there; {span}")
        elif sm[ax]["material"]:
            parts.append(f"on {ax} smoothing alone closes {sm_gain:+.1f} pts and the best delayed "
                         f"cell {best:+.1f} -- {best - sm_gain:+.1f} more, above the {thr:.1f}-pt "
                         f"threshold, so part of the closure needs the latency and part does not; "
                         f"{span}")
        elif n_close:
            parts.append(f"on {ax} no zero-delay cell clears the threshold ({sm_gain:+.1f} pts) "
                         f"while the best delayed cell reaches {best:+.1f} -- {best - sm_gain:+.1f} "
                         f"more, above the {thr:.1f}-pt threshold; {span}")
        else:
            parts.append(f"on {ax} nothing closes: smoothing only {sm_gain:+.1f} pts, {span}")
    parts.append("delay remains UNIDENTIFIED on this log (determined set "
                 f"{DETERMINED_DELAY} ticks, the whole grid); deadband is never varied on its "
                 f"own in these cells, so its contribution is untested")
    conclusion = "; ".join(parts)
    report["conclusion"] = conclusion
    print(f"\n  {conclusion}")

    json.dump(report, open("results/real2sim.json", "w"), indent=1)
    print(f"\nwrote results/real2sim.json   total {(time.time() - t0) / 60:.1f} min")
    tee.f.close()


if __name__ == "__main__":
    main()
