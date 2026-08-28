"""First real ?imulog=1 logs: convention evidence, gap per segment, actuator and sensor reads.

The question: what does the day-of-log chain actually say on the first real
sessions from the upstream app -- two walk-lane files, 16 s (end_why "done") and
5 s (end_why "tipped"), growbot-imulog-1 format, ~21 s of robot total?

PER FILE, and that is not a formatting choice. The two logs are not two samples of
one experiment:
  - walk-1 runs at agent gain null, walk-3 at 0.8, so the commands that reached the
    two bodies are scaled differently;
  - walk-1's body rests at pitch -0.74 rad, walk-3's flat at 0.00 -- the phone is
    not in the same place, so every attitude-referenced number means something
    different in each;
  - walk-1 walks for 15 s; walk-3 never walks at all. Its first 2.6 s are a
    motionless body under swinging commands, and its last 2.2 s are a fall.
An aggregate over the pair is an average of two different things. The earlier
version of this report published exactly that, and this one does not: walk-1 is
the identification and held-out file, walk-3 is reported separately by segment,
and nothing is pooled.

This script does five things the generic tools do not:
  1. Records the EMPIRICAL evidence for the device->twin conventions the
     growbot-imulog-1 adapter hard-codes, with every reading COMPUTED from the
     numbers beside it rather than asserted in a string: the rate-axis
     assignment is a stated separation test that walk-3 fails, the l/r
     assignment is scored against its own swapped alternative, and the twin's
     resting geometry is measured by running the twin.
  2. Runs the two day-of-log commands verbatim as subprocesses, once per file
     (gap_report.py <file> --servo-id and sensor_id.py <file>) -- proving the
     promised commands run unmodified on real data.
  3. Splits the gap per file AND per segment, against the twin floor each
     segment's own regime maps to.
  4. States the determined sets for the servo on walk-1 alone, and what longer
     captures would determine (the data ask).
  5. Tests, rather than assumes, whether header.gain is already inside the logged
     command values -- the two files differ in exactly that field and in nothing
     else that scales a command, so their amplitude ratio is the experiment.

Every number carries its conditions; n is small and printed everywhere.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
import numpy as np

from growbot_cerebellum import provenance
from growbot_cerebellum.paths import ROOT, DATA, RESULTS, LOGS, under_root
from growbot_cerebellum.imulog import parse, _deviceorientation_to_R, CTRL_HZ, rest_attitude
from growbot_cerebellum.forward import MLP, make_windows, K, AXES
from growbot_cerebellum.gap import evaluate_axes, twin_regimes, REGIME_MAP
from growbot_cerebellum.servo_id import identify, realized_from_commands, confidence_band, determined_sets, default_grid, argmin_interior
from growbot_cerebellum.sensor_id import default_out_path as sensor_out_path
from growbot_cerebellum.tee import Tee

# The rate-axis assignment claim is "the logged rates are the device x/y/z body
# rates, in that order". The test of that claim is diagonal dominance of the
# correlation matrix against vee(R^T dR/dt): every diagonal term must beat the
# largest off-diagonal by this factor. 1.5 is a stated threshold, not a fitted
# one -- below it the assignment is not established BY THIS FILE, whatever a
# neighbouring file says.
RATE_SEPARATION_FACTOR = 1.5
GAIN_RATIO_BOOTSTRAP = 4000     # resamples for the agent-gain amplitude ratio CI
GAIN_AMPLITUDE_PCT = 95         # the amplitude statistic compared between files



def twin_rest_pitch(settle_s=5.0):
    """The twin's own resting attitude, MEASURED: reset level, hold neutral, read rpy.

    The published version of this report carried an edge-rest figure of -0.81 rad
    as a bare constant with no procedure attached. This runs the procedure. What
    the number supports is narrow and worth stating: it says which SIGN of the
    mount's x axis is physically possible for a body at rest, not what the stance
    during a walk should be.
    """
    from growbot_sim import GrowBotSim
    sim = GrowBotSim(0)
    o = sim.reset(tilt=0.0)
    for _ in range(int(settle_s * CTRL_HZ)):
        o = sim.step(np.zeros(2, np.float32))
    return {"procedure": f"GrowBotSim(seed 0).reset(tilt=0) then {settle_s:.0f} s of "
                         f"neutral commands at {CTRL_HZ} Hz, final rpy",
            "roll": round(float(o[0]), 3), "pitch": round(float(o[1]), 3),
            "residual_gyro": round(float(np.abs(o[3:]).max()), 4)}


def command_amplitude(path, pct=GAIN_AMPLITUDE_PCT):
    """|command - neutral| in servo degrees, over the rows where the agent is driving.

    Raw pose columns, before any calibration inversion: the question is what the
    APP SENT, and the cal is identical in the two files anyway.
    """
    raw = json.load(open(path))
    pose = np.asarray(raw["pose"], np.float64)
    ok = pose[:, 4] != 0
    dev = np.concatenate([np.abs(pose[ok, 2] - 90.0), np.abs(pose[ok, 3] - 90.0)])
    dev = dev[dev > 0]                    # the neutral prefix before the walk starts
    return dev


def agent_gain_test(paths, headers, rng=None):
    """Is header.gain already inside the logged command values, or applied later?

    The adapter's docstring asserted that it is, with nothing behind the claim.
    The two files are the experiment: identical body, identical cal (signs, offs,
    cal gain 0.99, turn 0), identical gait and policy -- and agent gain null vs
    0.8. If the agent gain were applied downstream of the log, both files would
    show the same command amplitude and the recovered horn angles for walk-3
    would be 1/0.8 = 1.25x too large. If it is already inside the values, walk-3's
    amplitudes are 0.8x walk-1's.

    The statistic is the p95 of |command - 90| (a walk policy's output is bounded,
    so its upper envelope is what a gain scales), and the interval is a
    percentile bootstrap over the pooled l/r rows of each file.
    """
    rng = rng or np.random.default_rng(0)
    gains = [h.get("gain_agent") for h in headers]
    if len(paths) != 2 or sorted(str(g) for g in gains) != sorted(["None", "0.8"]):
        return {"determined": False,
                "reason": f"needs exactly one null-gain and one 0.8-gain file; got {gains}"}
    i_ref = [i for i, g in enumerate(gains) if g is None][0]
    i_gain = 1 - i_ref
    g = float(gains[i_gain])
    a_ref, a_gain = command_amplitude(paths[i_ref]), command_amplitude(paths[i_gain])
    stat = lambda x: float(np.percentile(x, GAIN_AMPLITUDE_PCT))
    ratio = stat(a_ref) / stat(a_gain)
    boot = np.array([stat(rng.choice(a_ref, len(a_ref))) / stat(rng.choice(a_gain, len(a_gain)))
                     for _ in range(GAIN_RATIO_BOOTSTRAP)])
    lo, hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    baked, applied_later = 1.0 / g, 1.0        # the two hypotheses, as ratios
    in_baked = lo <= baked <= hi
    in_later = lo <= applied_later <= hi
    if in_baked and not in_later:
        verdict = (f"BAKED IN: the ratio is consistent with 1/{g} = {baked:.2f} and excludes "
                   f"1.00, so header.gain is already inside the logged values and only "
                   f"cal.gain takes part in the inversion")
    elif in_later and not in_baked:
        verdict = (f"APPLIED LATER: the ratio is consistent with 1.00 and excludes 1/{g} = "
                   f"{baked:.2f} -- the adapter must divide walk-{g} commands by {g}, and "
                   f"every command-derived number for that file is currently {baked:.2f}x off")
    else:
        verdict = (f"UNDETERMINED: the interval covers "
                   f"{'both' if in_baked and in_later else 'neither'} hypothesis "
                   f"({baked:.2f} = baked in, 1.00 = applied later) -- the docstring claim "
                   f"stays 'assumed, untested' and every command-derived number from the "
                   f"gain-{g} file carries that caveat")
    return {"determined": bool(in_baked != in_later),
            "reference_file": paths[i_ref], "gain_file": paths[i_gain], "gain": g,
            "statistic": f"p{GAIN_AMPLITUDE_PCT} of |command - 90| deg, l and r pooled",
            "n": [int(len(a_ref)), int(len(a_gain))],
            "amplitude_deg": [round(stat(a_ref), 2), round(stat(a_gain), 2)],
            "ratio": round(float(ratio), 3), "ci95": [round(lo, 3), round(hi, 3)],
            "predicted_if_baked_in": round(baked, 3), "predicted_if_applied_later": 1.0,
            "verdict": verdict}


def mapping_evidence(path, te, tw_mode, twin_rest):
    """The empirical case for the adapter's conventions, from this file alone.

    Every 'reading' below is derived from the numbers next to it. A reading that
    the file's own numbers do not support says so, for that file, in those words.
    """
    raw = json.load(open(path))
    header_raw = raw.get("header", {})
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
    diag = [float(C[i, i]) for i in range(3)]
    offd = float(max(abs(C[i, j]) for i in range(3) for j in range(3) if i != j))
    sep = min(diag) >= offd * RATE_SEPARATION_FACTOR
    worst = int(np.argmin(diag))
    rate_assign = {"corr_matrix": np.round(C, 3).tolist(),
                   "diag": [round(v, 3) for v in diag],
                   "max_offdiag": round(offd, 3),
                   "separation_factor_required": RATE_SEPARATION_FACTOR,
                   "separation_observed": round(min(diag) / max(offd, 1e-9), 2),
                   "separated": bool(sep),
                   "reading": (
                       f"logged rate_alpha/beta/gamma ARE the device x/y/z body rates in that "
                       f"order, positive -- every diagonal beats the largest off-diagonal "
                       f"({offd:.3f}) by at least {RATE_SEPARATION_FACTOR}x on this file"
                       if sep else
                       f"NOT SEPARATED ON THIS FILE: the weakest diagonal "
                       f"({'xyz'[worst]}, {diag[worst]:.3f}) is only "
                       f"{min(diag) / max(offd, 1e-9):.2f}x the largest off-diagonal "
                       f"({offd:.3f}), below the {RATE_SEPARATION_FACTOR}x this report "
                       f"requires. The assignment is not established BY THIS FILE; it rests "
                       f"on the file(s) where the test passes")}

    O, A, O2, D, header, mode = parse(path)
    ti = np.arange(len(O)) * (1000.0 / CTRL_HZ)
    rest = rest_attitude(ti, O[:, :3], O[:, 3:])
    walking = mode == "walking"
    m = tw_mode; obs, act = te["obs"], te["act"]
    tw = (m == "policy") & (np.abs(obs[:, 0]) <= 1.2) & (np.abs(obs[:, 1]) <= 1.2)

    def sig(a, o):
        return {"mean_cmd_vs_pitch": round(float(np.corrcoef(a.mean(1), o[:, 1])[0, 1]), 3),
                "diff_cmd_vs_roll": round(float(np.corrcoef(a[:, 0] - a[:, 1], o[:, 0])[0, 1]), 3)}

    # Stance is a claim about a WALKING body. On a file with no walking segment it
    # is not evidence for anything, and the mount it would support is the mount of
    # a different placement.
    n_walk = int(walking.sum())
    if n_walk >= 50:
        stance = {"n_walking_ticks": n_walk,
                  "roll_mean": round(float(O[walking, 0].mean()), 3),
                  "pitch_mean": round(float(O[walking, 1].mean()), 3),
                  "pitch_std": round(float(O[walking, 1].std()), 3),
                  "twin_policy_pitch_mean": round(float(obs[tw, 1].mean()), 3),
                  "twin_policy_pitch_std": round(float(obs[tw, 1].std()), 3),
                  "twin_measured_rest": twin_rest,
                  "rest_attitude": None if rest is None else [round(rest[0], 3), round(rest[1], 3)],
                  "reading": "walking stance under this mount, from THIS file's placement only"}
    else:
        stance = {"n_walking_ticks": n_walk,
                  "rest_attitude": None if rest is None else [round(rest[0], 3), round(rest[1], 3)],
                  "twin_measured_rest": twin_rest,
                  "reading": (f"NO STANCE EVIDENCE FROM THIS FILE: only {n_walk} walking ticks. "
                              f"Its body rests at pitch "
                              f"{'unknown' if rest is None else f'{rest[1]:+.2f}'} rad, a "
                              f"different placement from the file that does walk, so its "
                              f"attitudes are not evidence about the walking mount")}

    # The l/r assignment against its own alternative, computed rather than claimed.
    # Swapping l and r swaps which logged column becomes a_right: the MEAN is
    # unchanged, so the pitch signature cannot decide this, and only the roll
    # signature can. Saying so is part of the evidence.
    if n_walk >= 50:
        real_sig = sig(A[walking], O[walking])
        swapped = sig(A[walking][:, ::-1], O[walking])
    else:
        real_sig = sig(A, O); swapped = sig(A[:, ::-1], O)
    twin_sig = sig(act[tw], obs[tw])
    agrees = lambda s: (np.sign(s["mean_cmd_vs_pitch"]) == np.sign(twin_sig["mean_cmd_vs_pitch"]),
                        np.sign(s["diff_cmd_vs_roll"]) == np.sign(twin_sig["diff_cmd_vs_roll"]))
    a_real, a_swap = agrees(real_sig), agrees(swapped)
    if n_walk < 50:
        reading = ("NO ACTION-RESPONSE EVIDENCE FROM THIS FILE: it has no walking segment, so "
                   "these correlations are between the commands and a body that was first "
                   "motionless and then falling. They are printed so that nobody reads the "
                   "adapter's l/r assignment out of them")
    elif all(a_real) and not a_swap[1]:
        reading = (f"the upstream l/r assignment agrees with the twin on both signatures and the "
                   f"swap disagrees on roll ({swapped['diff_cmd_vs_roll']:+.3f} vs the twin's "
                   f"{twin_sig['diff_cmd_vs_roll']:+.3f}); the pitch signature is INVARIANT under "
                   f"the swap (the mean of the two commands does not change) and decides nothing")
    else:
        reading = (f"INCONCLUSIVE on this file: upstream agrees {list(map(bool, a_real))}, swapped "
                   f"agrees {list(map(bool, a_swap))} on (pitch, roll) -- and the pitch signature "
                   f"is invariant under the swap, so only the roll column can decide it")
    action_response = {
        "real": real_sig, "swapped_lr": swapped, "twin_policy": twin_sig,
        "n_ticks": n_walk if n_walk >= 50 else int(len(O)),
        "over_walking_ticks": bool(n_walk >= 50),
        "upstream_signs_agree": [bool(a_real[0]), bool(a_real[1])],
        "swapped_signs_agree": [bool(a_swap[0]), bool(a_swap[1])],
        "reading": reading}

    return {"file": path,
            "walk": header_raw.get("walk"), "end_why": header_raw.get("end_why"),
            "gain_agent": header_raw.get("gain"),
            "segments": header.get("segments"),
            "gravity": grav,
            "rate_axis_assignment": rate_assign,
            "stance": stance,
            "action_response": action_response}


def print_gap_table(real, twin, horizons, indent="    "):
    print(f"{indent}{'segment':<10}{'twin floor':<12}{'n':>6}{'axis':>7}"
          f"{'@100ms real|twin|gap':>26}{'@500ms real|twin|gap':>26}")
    for reg in real:
        tname = REGIME_MAP.get(reg, "all") if reg != "all" else "all"
        if tname not in twin:
            tname = "all"
        tref = twin[tname]
        for ax in AXES:
            cells = ""
            for h in horizons:
                r = real[reg][h][ax]["within"]; t_ = tref[h][ax]["within"]
                cells += f"{r * 100:>10.1f} {t_ * 100:>5.1f} {(r - t_) * 100:>+6.1f}   "
            print(f"{indent}{reg if ax == 'roll' else '':<10}"
                  f"{(tname if ax == 'roll' else ''):<12}"
                  f"{real[reg]['n'] if ax == 'roll' else '':>6}{ax:>7}   {cells}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="growbot-imulog-1 walk files, in walk order")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--id-file", default=None,
                    help="the file the servo is identified on and evaluated held-out on; "
                         "default: the first file with a walking segment")
    args = ap.parse_args()
    # Taken before the run log opens (Tee truncates a tracked file -> "dirty").
    prov = provenance(seeds={"mlp": 0})
    sys.stdout = tee = Tee(LOGS / "real_log_report.txt")
    report = {"files": args.logs, "conditions": {"epochs": args.epochs, "K": K,
              "horizons_ms": [100, 500], "model": "MLP h128 seed 0 on data/train.npz",
              "analysis": "per file; nothing pooled across files"}}

    te = np.load(DATA / "test.npz")
    tw_mode, tw_rest = twin_regimes(te["obs"], te["mode"].astype(str))
    twin_rest = twin_rest_pitch()

    print("== 0. what the two files are, before anything is compared")
    parsed = {}
    for f in args.logs:
        O, A, O2, D, header, mode = parse(f)
        ti = np.arange(len(O)) * (1000.0 / CTRL_HZ)
        rest = rest_attitude(ti, O[:, :3], O[:, 3:])
        parsed[f] = (O, A, O2, D, header, mode, rest)
        segs = header.get("segments") or []
        t0 = segs[0][0] if segs else 0.0
        print(f"  {f}")
        print(f"    walk {header.get('walk')}, end_why={header.get('end_why')!r}, agent gain="
              f"{header.get('gain_agent')}, {len(O)} ticks ({len(O) / 50:.1f} s)")
        print("    rest attitude: " + ("none (no still segment)" if rest is None else
              f"roll {rest[0]:+.2f}, pitch {rest[1]:+.2f} rad"))
        print("    segments: " + " | ".join(f"{(t - t0) / 1000:.2f}s {n.removesuffix('_start')}"
                                             for t, n in segs))
        print(f"    regimes: { {m: int((mode == m).sum()) for m in sorted(set(mode))} }")
    rests = [parsed[f][6] for f in args.logs]
    gains = [parsed[f][4].get("gain_agent") for f in args.logs]
    if len(args.logs) > 1:
        print(f"\n  These files are NOT pooled. Agent gains {gains} differ, and the resting "
              f"attitudes\n  {[None if r is None else (round(r[0], 2), round(r[1], 2)) for r in rests]} "
              f"put the phone in different places. gap_report.py refuses to\n  concatenate them; "
              f"so does this report. Every table below names one file.")
    report["files_summary"] = {f: {"walk": parsed[f][4].get("walk"),
                                   "end_why": parsed[f][4].get("end_why"),
                                   "gain_agent": parsed[f][4].get("gain_agent"),
                                   "ticks": int(len(parsed[f][0])),
                                   "rest_attitude": None if parsed[f][6] is None else
                                                    [round(v, 3) for v in parsed[f][6]],
                                   "segments": parsed[f][4].get("segments"),
                                   "regimes": {m: int((parsed[f][5] == m).sum())
                                               for m in sorted(set(parsed[f][5]))}}
                              for f in args.logs}

    print("\n== 1. convention evidence (the adapter's mount and mappings, audited on this data)")
    print(f"  twin resting geometry, MEASURED: roll {twin_rest['roll']:+.3f}, pitch "
          f"{twin_rest['pitch']:+.3f} rad ({twin_rest['procedure']}; residual gyro "
          f"{twin_rest['residual_gyro']:.4f} rad/s)")
    report["twin_measured_rest"] = twin_rest
    report["mapping_evidence"] = [mapping_evidence(f, te, tw_mode, twin_rest) for f in args.logs]
    for ev in report["mapping_evidence"]:
        ra = ev["rate_axis_assignment"]; st = ev["stance"]; ar = ev["action_response"]
        print(f"\n  {ev['file']}")
        print(f"    gravity via W3C composition: earth mean {ev['gravity']['earth_mean']} m/s^2 "
              f"(expected {ev['gravity']['expected']})")
        print(f"    rate-axis diag corr {ra['diag']} vs max off-diag {ra['max_offdiag']}; "
              f"separation {ra['separation_observed']}x, need {ra['separation_factor_required']}x")
        print(f"      -> {ra['reading']}")
        if st["n_walking_ticks"] >= 50:
            print(f"    stance over {st['n_walking_ticks']} WALKING ticks: roll "
                  f"{st['roll_mean']:+.2f}, pitch {st['pitch_mean']:+.2f} +- {st['pitch_std']:.2f} rad "
                  f"(twin policy {st['twin_policy_pitch_mean']:+.2f} +- {st['twin_policy_pitch_std']:.2f}, "
                  f"twin measured rest {twin_rest['pitch']:+.2f})")
        else:
            print(f"    stance: {st['reading']}")
        print(f"    action-response over {ar['n_ticks']} ticks:")
        print(f"      upstream l/r {ar['real']}  swapped {ar['swapped_lr']}  twin {ar['twin_policy']}")
        print(f"      -> {ar['reading']}")

    print("\n== 2. the day-of-log commands, verbatim, ONE FILE AT A TIME")
    report["gap_report"] = {}
    report["sensor_id"] = {}
    for f in args.logs:
        # sensor_id writes one artifact per input; ask for the path explicitly rather
        # than relying on a fixed name that a second file would overwrite.
        sid_out = str(under_root(sensor_out_path([f])))
        gap_out = str(under_root(sensor_out_path([f], "gap_report")))
        for cmd, out, key in (([str(ROOT / "gap_report.py"), f, "--servo-id", "--out", gap_out], gap_out, "gap_report"),
                              ([str(ROOT / "sensor_id.py"), f, "--out", sid_out], sid_out, "sensor_id")):
            print(f"\n$ python {' '.join(cmd)}")
            r = subprocess.run([sys.executable, *cmd], capture_output=True, text=True)
            print(r.stdout, end="")
            if r.returncode != 0:
                print(r.stderr, end="")
                raise SystemExit(f"{cmd[0]} failed on {f}")
            report[key][f] = json.load(open(out))

    print("\n== 3. per-file gap, per segment, against the twin floor each segment maps to")
    tr = np.load(DATA / "train.npz")
    Xtr, Ytr, *_ = make_windows(tr["obs"], tr["act"], tr["next_obs"], tr["done"], K)
    model = MLP(hidden=128, epochs=args.epochs).fit(Xtr, Ytr)
    horizons = [5, 25]
    twin = evaluate_axes(model, te["obs"], te["act"], te["done"], tw_mode, horizons)
    report["twin_floor"] = twin
    print(f"  twin floor regimes (rest roll {tw_rest[0]:+.2f}, pitch {tw_rest[1]:+.2f} rad): "
          f"{ {m: twin[m]['n'] for m in twin} }")
    per_file = {}
    for f in args.logs:
        O, A, O2, D, header, mode, rest = parsed[f]
        real = evaluate_axes(model, O, A, D, mode, horizons)
        per_file[f] = {"ticks": int(len(O)), "seconds": round(len(O) / 50.0, 1),
                       "end_why": header.get("end_why"), "gain_agent": header.get("gain_agent"),
                       "twin_floor_used": {reg: (REGIME_MAP.get(reg, "all") if reg != "all" else "all")
                                           for reg in real},
                       "regimes": real}
        print(f"\n  {f} -- {len(O)} ticks ({len(O) / 50:.1f} s), end_why="
              f"{header.get('end_why')!r}, agent gain={header.get('gain_agent')}")
        print_gap_table(real, twin, horizons)
        for reg in sorted(set(mode) - set(real)):
            print(f"    ({reg}: {int((mode == reg).sum())} ticks, fewer than the 30 rollout "
                  f"starts evaluate_axes requires -- not reported)")
        # A still segment with a large gap is not a model failure and must not be read as
        # one: the model is fed commands, predicts the motion those commands imply, and
        # the body produced none. The size of that gap is a measurement OF THE ROBOT.
        if "still" in real:
            worst = min(real["still"][25][ax]["within"] for ax in AXES)
            drive = np.abs(A[mode == "still"]).max() if (mode == "still").any() else 0.0
            if worst < 0.5 and np.rad2deg(drive) >= 5.0:
                ax = min(AXES, key=lambda a: real["still"][25][a]["within"])
                print(f"    NOTE: the still segment scores {worst * 100:.1f}% on {ax} at 500 ms "
                      f"while the commands reach {np.rad2deg(drive):.0f} deg of horn swing. The "
                      f"model predicts the motion those commands imply; the body produced none. "
                      f"That gap measures the ROBOT (phone off the body, body off the ground, or "
                      f"servos not moving), not the forward model.")
    report["per_file"] = per_file

    # --- which file carries the identification --------------------------------------
    id_file = args.id_file
    if id_file is None:
        walkers = [f for f in args.logs if int((parsed[f][5] == "walking").sum()) >= 200]
        if not walkers:
            raise SystemExit("no file has a walking segment long enough to identify a servo on")
        id_file = walkers[0]
    print(f"\n== 4. servo determined sets on {id_file} ALONE "
          f"(fit = its first half, evaluation = its held-out second half)")
    other = [f for f in args.logs if f != id_file]
    if other:
        print(f"  The other file(s) {other} are NOT in this fit and NOT in this evaluation. "
              f"Concatenating\n  them would identify one servo from two different action "
              f"distributions and score it on a body\n  that spent its recording motionless "
              f"and then falling.")
    O, A, O2, D, header, mode, rest = parsed[id_file]
    label = mode
    half = len(O) // 2
    # servo_id.default_grid() is now this range: the widening this report needed was
    # folded back into the shared definition, so the CLI cannot pin at a boundary the
    # real log is known to exceed.
    grid = default_grid()
    scores, best = identify(model, O[:half], A[:half], O2[:half], D[:half], grid)
    hA, hB = slice(0, half // 2), slice(half // 2, half)
    sA, bA = identify(model, O[hA], A[hA], O2[hA], D[hA], grid)
    sB, bB = identify(model, O[hB], A[hB], O2[hB], D[hB], grid)
    band = confidence_band(sA, sB)
    dset, sset = determined_sets(scores, best, grid, band)
    agree = bA["delay_ticks"] == bB["delay_ticks"] and bA["slew_rad_s"] == bB["slew_rad_s"]
    print(f"  {half} ticks fit ({half / 50:.1f} s), {len(grid)} hypotheses, split-half band {band:.4f}")
    print(f"  argmin: delay {best['delay_ticks']} ticks, slew {best['slew_rad_s']} rad/s, "
          f"deadband {np.rad2deg(best['deadband']):.0f} deg")
    print(f"  split-half: A=(delay {bA['delay_ticks']}, slew {bA['slew_rad_s']})  "
          f"B=(delay {bB['delay_ticks']}, slew {bB['slew_rad_s']})  "
          f"{'AGREE' if agree else 'DISAGREE -- log too short or servo outside the model family'}")
    print(f"  delay determined set {dset} ticks; slew determined set {sset} rad/s")
    interior, interior_why = argmin_interior(best, grid)
    print(f"  {interior_why}")
    print(f"  (the fixture's 480 s of mixed excitation determines both to +-1 grid step; "
          f"{half / 50:.0f} s of periodic walking is the excitation servo_id.py warns about: "
          f"'near-ties mean the excitation never hit the slew limit')")
    R = realized_from_commands(A, D, best)
    held = slice(half, None)
    real_h = evaluate_axes(model, O[held], A[held], D[held], label[held], horizons)
    corr_h = evaluate_axes(model, O[held], R[held], D[held], label[held], horizons)
    ext = {}
    print(f"  held-out half ({len(O) - half} ticks) with the extended-grid servo "
          f"(within 0.2 rad; twin floor per segment):")
    for reg in real_h:
        tname = REGIME_MAP.get(reg, "all") if reg != "all" else "all"
        tref = twin.get(tname, twin["all"])
        for ax in AXES:
            cells = ""
            for h in horizons:
                r = real_h[reg][h][ax]["within"]; c = corr_h[reg][h][ax]["within"]
                t_ = tref[h][ax]["within"]
                ext.setdefault(reg, {}).setdefault(str(h), {})[ax] = {
                    "real": r, "after_servo": c, "twin": t_, "twin_regime": tname}
                cells += f"  @{h * 20}ms {r * 100:5.1f} -> {c * 100:5.1f} (floor {t_ * 100:5.1f})"
            print(f"    {reg if ax == 'roll' else '':<10}{ax:>6}{cells}")
    report["servo"] = {"file": id_file, "fit_ticks": int(half), "band": float(band),
                       "argmin": {k: (float(v) if v is not None else None) for k, v in best.items()},
                       "argmin_interior": bool(interior),
                       "split_half_agree": bool(agree),
                       "delay_determined_set": [int(v) for v in dset],
                       "slew_determined_set": [None if v is None else float(v) for v in sset],
                       "held_out_by_segment": ext}

    print("\n== 5. is header.gain already inside the logged commands? (the test, not the assumption)")
    gt = agent_gain_test(args.logs, [parsed[f][4] for f in args.logs])
    report["agent_gain_test"] = gt
    if "reason" in gt:
        print(f"  not runnable: {gt['reason']}")
    else:
        print(f"  statistic: {gt['statistic']}")
        print(f"  {gt['reference_file']} (gain null): {gt['amplitude_deg'][0]:.2f} deg, n={gt['n'][0]}")
        print(f"  {gt['gain_file']} (gain {gt['gain']}): {gt['amplitude_deg'][1]:.2f} deg, n={gt['n'][1]}")
        print(f"  ratio {gt['ratio']:.3f}, bootstrap 95% CI [{gt['ci95'][0]:.3f}, {gt['ci95'][1]:.3f}] "
              f"({GAIN_RATIO_BOOTSTRAP} resamples)")
        print(f"  hypotheses: baked into the log -> {gt['predicted_if_baked_in']:.2f}; "
              f"applied downstream -> {gt['predicted_if_applied_later']:.2f}")
        print(f"  -> {gt['verdict']}")

    print("\n== 6. the data ask, from these numbers")
    still_s = {}
    for f in args.logs:
        segs = parsed[f][4].get("segments") or []
        t0 = segs[0][0] if segs else 0.0
        tend = t0 + len(parsed[f][0]) * 20.0
        for i, (t, n) in enumerate(segs):
            if n == "still_start":
                t1 = segs[i + 1][0] if i + 1 < len(segs) else tend
                still_s[f] = max(still_s.get(f, 0.0), (t1 - t) / 1000.0)
    longest_still = max(still_s.values(), default=0.0)
    ask = [
        f"pose-sequence (gesture) capture, 3-5 min: {id_file} alone leaves the servo at "
        f"delay {dset} ticks / slew {sset} rad/s; varied large and small steps are what "
        f"separate slew and deadband (the twin fixture determines both to one grid step "
        f"from 8 min of mixed excitation)",
        f"one still capture, 2-5 min (robot on the floor, untouched). A still segment DOES "
        f"exist -- the longest here is {longest_still:.1f} s -- but Allan deviation needs "
        f"tens of taus with many independent clusters each, and at 60 Hz that means minutes, "
        f"not seconds. ARW and bias instability stay unmeasurable because the still segments "
        f"are TOO SHORT, not because none was recorded",
        f"a walk lane where the robot actually walks the whole time: walk-3 contributed "
        f"{sum(1 for m in parsed[args.logs[-1]][5] if m == 'walking') / 50:.1f} s of walking "
        f"out of {len(parsed[args.logs[-1]][0]) / 50:.1f} s -- 2.6 s of it is a motionless body "
        f"under swinging commands, which is a hardware or placement report, not walk data",
        "the same walk lane twice, battery fresh vs low: the twin predicts a split-half "
        "DISAGREE on slew if the battery sags (model_mismatch.py); this is the free real "
        "actuator experiment",
    ]
    for a in ask:
        print(f"  - {a}")
    report["data_ask"] = ask

    report["provenance"] = prov
    json.dump(report, open(RESULTS / "real_log_report.json", "w"), indent=1)
    print("\nwrote results/real_log_report.json")
    tee.f.close()


if __name__ == "__main__":
    main()
