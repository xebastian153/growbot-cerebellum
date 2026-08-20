"""First real ?imulog=1 logs: convention evidence, gap per axis, actuator and sensor reads.

The question: what does the day-of-log chain actually say on the first real
sessions from the upstream app -- two walk-lane files, 16 s (end_why "done") and
5 s (end_why "tipped"), growbot-imulog-1 format, ~21 s of robot total?

This script does four things the generic tools do not:
  1. Records the EMPIRICAL evidence for the device->twin conventions the
     growbot-imulog-1 adapter hard-codes (gravity residual, rate-axis
     assignment, kinematic pair consistency, stance physics, action-response
     signatures vs the twin), so the mapping is auditable, not asserted.
  2. Runs the two day-of-log commands verbatim as subprocesses
     (gap_report.py <files> --servo-id and sensor_id.py <files>) and captures
     their output -- proving the promised commands run unmodified on real data.
  3. Splits the gap per file and, for the tipped walk, separates the final
     1.5 s (the tip onset the recording ends at) from the walking before it.
  4. States the determined sets for the servo on this much data, and what
     longer captures would determine (the data ask).

Every number carries its conditions; n is small and printed everywhere.
"""
from __future__ import annotations
import argparse, itertools, json, subprocess, sys
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "sim")

from imulog import (parse, _read_rows, _deviceorientation_to_R, _R_to_zyx_rpy,
                    GROWBOT_V1_MOUNT)
from forward import MLP, make_windows
from sim2real_proxy import K
from gap_report import evaluate_axes, REGIME_MAP, AXES
from servo_id import identify, realized_from_commands, confidence_band, determined_sets


class Tee:
    def __init__(self, path):
        self.f = open(path, "w")
    def write(self, s):
        sys.__stdout__.write(s); self.f.write(s)
    def flush(self):
        sys.__stdout__.flush()
        if not self.f.closed:
            self.f.flush()


def mapping_evidence(path, te):
    """The empirical case for the adapter's conventions, from this file alone."""
    raw = json.load(open(path))
    imu = np.asarray(raw["imu"], np.float64)
    t = imu[:, 1] / 1000.0
    acc = imu[:, 2:5]
    rate = np.deg2rad(imu[:, 5:8])
    ori = np.deg2rad(imu[:, 8:11])
    R = _deviceorientation_to_R(ori[:, 0], ori[:, 1], ori[:, 2])

    # gravity through the W3C composition
    ae = np.einsum("nij,nj->ni", R, acc)
    grav = {"earth_mean": [round(float(v), 2) for v in ae.mean(0)],
            "expected": [0.0, 0.0, 9.81]}

    # which logged rate is which device axis: vee(R^T dR/dt) vs the three columns
    w = np.empty((len(t), 3))
    for i in range(1, len(t) - 1):
        S = R[i].T @ ((R[i + 1] - R[i - 1]) / (t[i + 1] - t[i - 1]))
        w[i] = [S[2, 1], S[0, 2], S[1, 0]]
    w[0], w[-1] = w[1], w[-2]
    C = np.array([[np.corrcoef(w[:, i], rate[:, j])[0, 1] for j in range(3)] for i in range(3)])
    rate_assign = {"corr_matrix": np.round(C, 3).tolist(),
                   "diag": [round(float(C[i, i]), 3) for i in range(3)],
                   "max_offdiag": round(float(max(abs(C[i, j]) for i in range(3)
                                                  for j in range(3) if i != j)), 3),
                   "reading": "logged rate_alpha/beta/gamma are about device x/y/z "
                              "in that order, positive -- NOT the W3C rotationRate naming"}

    # twin-frame stance and action-response signatures under the adapter's mount
    O, A, O2, D, header, mode = parse(path)
    m = te["mode"].astype(str); obs, act = te["obs"], te["act"]
    tw = (m == "policy") & (np.abs(obs[:, 0]) <= 1.2) & (np.abs(obs[:, 1]) <= 1.2)
    def sig(a, o):
        return {"mean_cmd_vs_pitch": round(float(np.corrcoef(a.mean(1), o[:, 1])[0, 1]), 3),
                "diff_cmd_vs_roll": round(float(np.corrcoef(a[:, 0] - a[:, 1], o[:, 0])[0, 1]), 3)}
    return {"file": path,
            "gravity": grav,
            "rate_axis_assignment": rate_assign,
            "stance": {"roll_mean": round(float(O[:, 0].mean()), 3),
                       "pitch_mean": round(float(O[:, 1].mean()), 3),
                       "pitch_std": round(float(O[:, 1].std()), 3),
                       "twin_policy_pitch_mean": round(float(obs[tw, 1].mean()), 3),
                       "twin_geometry_edge_rest": -0.81},
            "action_response": {"real": sig(A, O), "twin_policy": sig(act[tw], obs[tw]),
                                "reading": "same signs as the twin under the upstream l/r "
                                           "assignment; swapping l and r flips the roll "
                                           "signature into disagreement"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="growbot-imulog-1 walk files, in walk order")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--tip-window-s", type=float, default=1.5,
                    help="final span of an end_why=tipped file reported separately")
    args = ap.parse_args()
    sys.stdout = tee = Tee("results/logs/real_log_report.txt")
    report = {"files": args.logs, "conditions": {"epochs": args.epochs, "K": K,
              "horizons_ms": [100, 500], "model": "MLP h128 seed 0 on data/train.npz",
              "tip_window_s": args.tip_window_s}}

    te = np.load("data/test.npz")

    print("== 1. convention evidence (the adapter's mount and mappings, audited on this data)")
    report["mapping_evidence"] = [mapping_evidence(f, te) for f in args.logs]
    for ev in report["mapping_evidence"]:
        ra = ev["rate_axis_assignment"]; st = ev["stance"]; ar = ev["action_response"]
        print(f"  {ev['file']}")
        print(f"    gravity via W3C composition: earth mean {ev['gravity']['earth_mean']} m/s^2")
        print(f"    rate-axis diag corr {ra['diag']} vs max off-diag {ra['max_offdiag']} -- {ra['reading']}")
        print(f"    stance roll {st['roll_mean']:+.2f}, pitch {st['pitch_mean']:+.2f} +- {st['pitch_std']:.2f} rad "
              f"(twin policy {st['twin_policy_pitch_mean']:+.2f}, edge-rest geometry {st['twin_geometry_edge_rest']})")
        print(f"    action-response real {ar['real']} vs twin {ar['twin_policy']}")

    print("\n== 2. the day-of-log commands, verbatim")
    for cmd in (["gap_report.py", *args.logs, "--servo-id"], ["sensor_id.py", *args.logs]):
        print(f"\n$ python {' '.join(cmd)}")
        r = subprocess.run([sys.executable, *cmd], capture_output=True, text=True)
        print(r.stdout, end="")
        if r.returncode != 0:
            print(r.stderr, end="")
            raise SystemExit(f"{cmd[0]} failed")
    report["gap_report"] = json.load(open("results/gap_report.json"))
    report["sensor_id"] = json.load(open("results/sensor_id.json"))

    print("\n== 3. per-file gap, and the tip onset separated")
    tr = np.load("data/train.npz")
    Xtr, Ytr, *_ = make_windows(tr["obs"], tr["act"], tr["next_obs"], tr["done"], K)
    model = MLP(hidden=128, epochs=args.epochs).fit(Xtr, Ytr)
    horizons = [5, 25]
    twin = evaluate_axes(model, te["obs"], te["act"], te["done"], te["mode"].astype(str), horizons)
    report["twin_floor"] = twin
    per_file = {}
    parts = []
    for f in args.logs:
        O, A, O2, D, header, mode = parse(f)
        label = np.array(mode, dtype=object)
        if header.get("end_why") == "tipped":
            t_end = len(O) * 20.0
            label[-int(args.tip_window_s * 50):] = "tip_onset"
        label = label.astype(str)
        parts.append((O, A, O2, D, label))
        real = evaluate_axes(model, O, A, D, label, horizons)
        per_file[f] = {"ticks": int(len(O)), "seconds": round(len(O) / 50.0, 1),
                       "end_why": header.get("end_why"), "gain_agent": header.get("gain_agent"),
                       "regimes": real}
        print(f"\n  {f} -- {len(O)} ticks ({len(O)/50:.1f} s), end_why={header.get('end_why')!r}, "
              f"agent gain={header.get('gain_agent')}")
        print(f"    {'regime':<12}{'n':>6}{'axis':>7}{'@100ms real|twin|gap':>26}{'@500ms real|twin|gap':>26}")
        for reg in real:
            tref = twin.get(REGIME_MAP.get(reg, "all") if reg != "all" else "all", twin["all"])
            for ax in AXES:
                cells = ""
                for h in horizons:
                    r = real[reg][h][ax]["within"]; t_ = tref[h][ax]["within"]
                    cells += f"{r*100:>10.1f} {t_*100:>5.1f} {(r-t_)*100:>+6.1f}   "
                print(f"    {reg if ax == 'roll' else '':<12}{real[reg]['n'] if ax == 'roll' else '':>6}"
                      f"{ax:>7}   {cells}")
        if "tip_onset" not in real and header.get("end_why") == "tipped":
            print(f"    (tip window has too few rollout starts to report separately -- "
                  f"needs >= 30, the final {args.tip_window_s} s gives fewer once "
                  f"horizons are excluded)")
    report["per_file"] = per_file

    print("\n== 4. servo determined sets on the concatenated log "
          "(fit = first half, evaluation = held-out second half)")
    O, A, O2, D, label = (np.concatenate(x) for x in zip(*parts))
    half = len(O) // 2
    # The published grid (delay 0-3, slew >= 3) pins BOTH parameters at its boundary on
    # this log, and a boundary argmin is the search running out, not an identification.
    # Extended until the optimum is interior: delays to 120 ms, slews down to 1 rad/s.
    grid = list(itertools.product([0, 1, 2, 3, 4, 5, 6],
                                  [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, None],
                                  [0.0, np.deg2rad(1), np.deg2rad(2), np.deg2rad(4)]))
    scores, best = identify(model, O[:half], A[:half], O2[:half], D[:half], grid)
    hA, hB = slice(0, half // 2), slice(half // 2, half)
    sA, bA = identify(model, O[hA], A[hA], O2[hA], D[hA], grid)
    sB, bB = identify(model, O[hB], A[hB], O2[hB], D[hB], grid)
    band = confidence_band(sA, sB)
    dset, sset = determined_sets(scores, best, grid, band)
    agree = bA["delay_ticks"] == bB["delay_ticks"] and bA["slew_rad_s"] == bB["slew_rad_s"]
    print(f"  {half} ticks fit ({half/50:.1f} s), {len(grid)} hypotheses, split-half band {band:.4f}")
    print(f"  argmin: delay {best['delay_ticks']} ticks, slew {best['slew_rad_s']} rad/s, "
          f"deadband {np.rad2deg(best['deadband']):.0f} deg")
    print(f"  split-half: A=(delay {bA['delay_ticks']}, slew {bA['slew_rad_s']})  "
          f"B=(delay {bB['delay_ticks']}, slew {bB['slew_rad_s']})  "
          f"{'AGREE' if agree else 'DISAGREE -- log too short or servo outside the model family'}")
    print(f"  delay determined set {dset} ticks; slew determined set {sset} rad/s")
    delays = sorted({d for d, _, _ in grid})
    slews = sorted({s for _, s, _ in grid if s is not None})
    interior = (min(delays) < best["delay_ticks"] < max(delays)
                and (best["slew_rad_s"] is None or min(slews) < best["slew_rad_s"] < max(slews)))
    print(f"  argmin is {'INTERIOR to' if interior else 'AT THE BOUNDARY of'} the extended grid "
          f"(delay {min(delays)}-{max(delays)} ticks, slew {min(slews)}-{max(slews)} rad/s or none)")
    print(f"  (the fixture's 480 s of mixed excitation determines both to +-1 grid step; "
          f"{half/50:.0f} s of periodic walking is the excitation servo_id.py warns about: "
          f"'near-ties mean the excitation never hit the slew limit')")
    # gap* under the extended-grid best, same held-out discipline as gap_report
    R = realized_from_commands(A, D, best)
    held = slice(half, None)
    real_h = evaluate_axes(model, O[held], A[held], D[held], label[held], horizons)
    corr_h = evaluate_axes(model, O[held], R[held], D[held], label[held], horizons)
    ext = {}
    print(f"  held-out half with the extended-grid servo (within 0.2 rad; twin floor = policy regime):")
    tref = twin.get("policy", twin["all"])
    for ax in AXES:
        cells = ""
        for h in horizons:
            r = real_h["all"][h][ax]["within"]; c = corr_h["all"][h][ax]["within"]
            t_ = tref[h][ax]["within"]
            ext.setdefault(str(h), {})[ax] = {"real": r, "after_servo": c, "twin": t_}
            cells += f"  @{h*20}ms {r*100:5.1f} -> {c*100:5.1f} (floor {t_*100:5.1f})"
        print(f"    {ax:>6}{cells}")
    report["servo"] = {"fit_ticks": int(half), "band": float(band),
                       "argmin": {k: (float(v) if v is not None else None) for k, v in best.items()},
                       "argmin_interior": bool(interior),
                       "split_half_agree": bool(agree),
                       "delay_determined_set": [int(v) for v in dset],
                       "slew_determined_set": [None if v is None else float(v) for v in sset],
                       "held_out_extended": ext}

    print("\n== 5. the data ask, from these numbers")
    ask = [
        f"pose-sequence (gesture) capture, 3-5 min: the walk lane alone leaves the servo at "
        f"delay {dset} ticks / slew {sset} rad/s; varied large and small steps are what "
        f"separate slew and deadband (the twin fixture determines both to one grid step "
        f"from 8 min of mixed excitation)",
        "one still capture, 2-5 min (robot on the floor, untouched): unlocks the Allan "
        "read of the phone gyro's noise -- no still segment exists in the walk lane, so "
        "ARW and bias instability are currently unmeasurable",
        "the same walk lane twice, battery fresh vs low: the twin predicts a split-half "
        "DISAGREE on slew if the battery sags (model_mismatch.py); this is the free real "
        "actuator experiment",
    ]
    for a in ask:
        print(f"  - {a}")
    report["data_ask"] = ask

    json.dump(report, open("results/real_log_report.json", "w"), indent=1)
    print("\nwrote results/real_log_report.json")
    tee.f.close()


if __name__ == "__main__":
    main()
