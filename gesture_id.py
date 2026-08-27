"""What an `act` (gesture) capture can and cannot identify about the servo.

We asked the maintainer for varied excitation because ~15 s of periodic walking left
the servo delay undetermined over the whole grid: a gait moves the horns through one
narrow band at one rhythm, which is the worst case for separating a latency from a
rate limit. A gesture lane moves them through +-86 deg of horn travel in steps of
20-111 deg, so on excitation alone it is the better file.

It answers less anyway, and this script measures how much less rather than explaining
why. What the log carries about each keyframe is a TARGET POSE and a send time. What
it does not carry is the duration of the act that reaches it: the `act` verb has one,
the glide engine plays it out, and no field records it. `realized_from_commands`
replays a STEP and lets the candidate servo's own slew shape the trajectory, so the
command it scores hypotheses against is not the command the body received -- for any
act whose duration is longer than a tick.

RETRACTED, and kept here so the record shows the correction. This module previously
derived a "glide rate" of 1.0 rad/s from the header's documented example
{l:130, r:50, ms:700}, called it the engine's ramp, and reported that the identified
slew measured it. Three things are wrong with that. {l:130, r:50} is an absolute
TARGET POSE pair (90+40 and 90-40), not a 40 deg move: it is a 40 deg move only from
neutral, and the header does not say where it starts. The example documents the sit
fold that happens AFTER recording ends -- the same field, and the same misreading,
that `docs/EXPERIMENTS.md` already retracts a coverage conclusion over. And the
derivation dropped cal.gain, which every other conversion in this repository applies.
The identified slew 1.0 rad/s "confirming" the derived 1.0 rad/s confirmed nothing
either way: 1.0 is min(slew) on the shared grid, and this file's slew determined set
is the whole grid, so any file that separates no hypothesis lands its argmin there.

What is reported instead: the keyframes as the validated adapter reads them, and the
act duration as an OPEN question with both readings the log admits, stated as
unresolved. Under a constant-rate engine the horn is never asked to move fast; under
a constant-duration engine the largest steps are commanded at or above the walk lane's
own slew band. The log separates neither, and the identification result -- both
parameters undetermined over the entire grid, at a band 19x the walk lane's -- does
not depend on which is true.
"""
from __future__ import annotations
import json
import numpy as np
from growbot_cerebellum import provenance
from growbot_cerebellum.forward import MLP, make_windows
from growbot_cerebellum.imulog import parse, run_preflight, _read_rows, _convert_growbot_v1
from growbot_cerebellum.sensor_id import default_out_path
from growbot_cerebellum.forward import K
from growbot_cerebellum.servo_id import identify, confidence_band, determined_sets, default_grid, argmin_interior

LOG = "SEND-gesture-3.6min.json"
WALK = "imu-walk-1-2026-08-20T17-50-14-713Z.json"
DOC_ACT = {"l": 130.0, "r": 50.0, "ms": 700.0}      # the header's one documented act
DOC_ACT_WHERE = ("header.post_walk: the sit fold that happens AFTER recording ends, on walks "
                 "that end 'done'. It is the only act in the log with an ms, it is not a member "
                 "of this session, and it states a target pose pair, not a displacement")


def documented_act_horn_deg(path):
    """|horn travel| of the documented act, MEASURED FROM NEUTRAL, through the same
    adapter the log's own keyframes go through.

    The pose pair is fed as a two-row command stream (neutral, then the documented
    pose) over this file's own header, so the calibration inversion -- signs, offsets,
    cal.gain, turn -- is the validated one in `imulog._convert_growbot_v1` and not a
    second copy of the same arithmetic living here. "From neutral" is an ASSUMPTION,
    not a reading: the header gives no start pose for the example.
    """
    obj = json.load(open(path))
    imu = obj["imu"][:64]
    probe = {"header": obj["header"], "imu": imu,
             "pose": [[0, float(imu[0][1]), 90.0, 90.0, 1],
                      [1, float(imu[8][1]), DOC_ACT["l"], DOC_ACT["r"], 1]]}
    _, _, _, _, cmd_v, _ = _convert_growbot_v1(probe)
    cmd = np.rad2deg(np.asarray(cmd_v, float))
    return float(np.abs(cmd[1] - cmd[0]).max())


def keyframe_stats(path):
    """What the pose stream says, read through the validated adapter.

    Reading `obj["pose"]` directly (as this did) bypasses three checks the adapter
    owns: the declared field order the converter indexes by position, the send_ok
    filter that drops commands which never reached the body, and the empty-pose
    reshape. Those are the parser's contract with the file; a second reader that
    skips them is a second implementation of the same math with no equivalence test.
    """
    header, (imu_t, imu_v, cmd_t, cmd_v), _ = _read_rows(path)
    ct = np.asarray(cmd_t, float)
    deg = np.rad2deg(np.asarray(cmd_v, float))          # horn degrees, twin order [right, left]
    step = np.abs(np.diff(deg, axis=0)).max(1)          # per-keyframe move, larger horn
    dtk = np.diff(ct)                                   # gap AFTER each keyframe
    return {"source": "imulog._read_rows (declared field order checked, send_ok filtered, "
                      "cal inverted) -- horn degrees, not raw servo degrees",
            "n_keyframes": int(len(ct)),
            "n_pose_rows_in_file": int(header.get("n_pose_rows", len(ct))),
            "send_ok_dropped": int(header.get("send_ok_dropped", 0)),
            "span_s": float((ct[-1] - ct[0]) / 1000.0),
            "step_deg": {"median": float(np.median(step)), "max": float(step.max()),
                         "min": float(step.min()), "n": int(len(step)),
                         "n_zero": int((step == 0).sum())},
            "horn_cmd_deg": {"min": float(deg.min()), "max": float(deg.max())},
            "inter_keyframe_s": {"min": float(dtk.min() / 1000), "p25": float(np.percentile(dtk, 25) / 1000),
                                 "median": float(np.median(dtk) / 1000), "max": float(dtk.max() / 1000)},
            "_step": step, "_dtk": dtk}


def act_duration_readings(ks, doc_step_deg, walk_slew_determined):
    """The two readings of the unrecorded act duration, both stated, neither resolved.

    The log records send times and target poses. Any statement about what the body was
    COMMANDED to do between two keyframes is an assumption about the engine, so each
    reading carries its assumption, where the assumption comes from, and what it would
    have to be wrong about.
    """
    step, dtk = ks["_step"], ks["_dtk"]
    rate = float(np.deg2rad(doc_step_deg) / (DOC_ACT["ms"] / 1000.0))
    need_ms = 1000.0 * np.deg2rad(step) / rate
    # an act commanded AT keyframe i+1 covers step[i] and is interrupted when the next
    # keyframe arrives before it ends, i.e. dtk[i+1]. Pairing need_ms[i] with dtk[i] --
    # as this did -- compares a glide with the interval BEFORE its own command.
    cut_rate = float(np.mean(need_ms[:-1] > dtk[1:]))
    implied = np.deg2rad(step) / (DOC_ACT["ms"] / 1000.0)
    lo, hi = min(s for s in walk_slew_determined if s is not None), \
        max(s for s in walk_slew_determined if s is not None)
    return {
        "recorded_in_log": False,
        "documented_example": {**DOC_ACT, "where": DOC_ACT_WHERE,
                               "horn_travel_from_neutral_deg": doc_step_deg},
        "resolved": False,
        "readings": [
            {"name": "constant rate",
             "assumption": "every act plays at one fixed angular rate, and the documented "
                           "example starts from neutral so that rate is its travel over its ms",
             "rate_rad_s": rate,
             "uncertainty": "unverifiable from this log: no act in the session carries an ms, "
                            "the example's start pose is not stated, and the example is the "
                            "post-recording fold rather than one of these keyframes",
             "implied_act_ms": {"median": float(np.median(need_ms)), "max": float(need_ms.max())},
             "implied_act_ms_note": "the median is 700 ms BY CONSTRUCTION -- the median step is "
                                    "the same move the rate is defined from",
             "frac_acts_interrupted": cut_rate},
            {"name": "constant duration",
             "assumption": "every act takes the documented 700 ms whatever the step, so the "
                           "commanded rate scales with the move",
             "uncertainty": "equally unverifiable, and equally consistent with every field the "
                            "log carries",
             "implied_rate_rad_s": {"median": float(np.median(implied)), "max": float(implied.max())},
             "n_steps_at_or_above_walk_slew": int((implied >= lo).sum()),
             "n_steps": int(len(implied)),
             "walk_slew_determined_rad_s": [None if s is None else float(s)
                                            for s in walk_slew_determined],
             "frac_acts_interrupted": float(np.mean(dtk[1:] < DOC_ACT["ms"]))},
        ],
        "consequence": (f"the two readings disagree about whether the horn is ever commanded at "
                        f"its own limit -- under the first it never reaches {lo} rad/s, under the "
                        f"second {int((implied >= lo).sum())} of {len(implied)} steps are at or "
                        f"above the {lo}-{hi} rad/s the walk lane determines. This file separates "
                        f"neither, so neither is published as a property of the robot"),
    }


def run(model, path, grid, label):
    O, A, O2, D, h, mode = parse(path)
    n = len(O); half = n // 2
    fit = slice(0, half)
    scores, best = identify(model, O[fit], A[fit], O2[fit], D[fit], grid)
    q = half // 2
    sA, _ = identify(model, O[0:q], A[0:q], O2[0:q], D[0:q], grid)
    sB, _ = identify(model, O[q:half], A[q:half], O2[q:half], D[q:half], grid)
    band = confidence_band(sA, sB)
    dset, sset = determined_sets(scores, best, grid, band)
    interior, why = argmin_interior(best, grid)
    all_delays = sorted({d for d, _, _ in grid})
    all_slews = sorted({s for _, s, _ in grid}, key=lambda v: (v is None, v))
    return {"log": path, "label": label, "ticks": int(n), "fit_ticks": int(half),
            "argmin": {k: (None if v is None else float(v)) for k, v in best.items()},
            "band": float(band), "delay_determined": [int(d) for d in dset],
            "slew_determined": [None if s is None else float(s) for s in sset],
            "argmin_interior": bool(interior), "interior_why": why,
            # a slew argmin equal to the grid's smallest slew is where a file that
            # separates nothing necessarily lands, so the value is reported with that
            # fact attached rather than read as a measurement
            "grid_slew_min_rad_s": float(min(s for s in all_slews if s is not None)),
            "argmin_slew_is_grid_min": bool(best["slew_rad_s"] is not None
                                            and best["slew_rad_s"] == min(s for s in all_slews
                                                                          if s is not None)),
            "delay_undetermined_whole_grid": bool(list(dset) == list(all_delays)),
            "slew_undetermined_whole_grid": bool(list(sset) == list(all_slews)),
            "split_half": {"A": {k: (None if v is None else float(v)) for k, v in sA[0][1].items()},
                           "B": {k: (None if v is None else float(v)) for k, v in sB[0][1].items()},
                           "agree": sA[0][1] == sB[0][1]}}


def main(log=LOG, walk=WALK):
    for f in (log, walk):
        if not run_preflight(f):
            raise SystemExit(f"preflight FAIL on {f}")
    tr = np.load("data/train.npz")
    Xtr, Ytr, *_ = make_windows(tr["obs"], tr["act"], tr["next_obs"], tr["done"], K)
    model = MLP(hidden=128, epochs=80).fit(Xtr, Ytr)
    grid = default_grid()

    ks = keyframe_stats(log)
    print("\n== the keyframes, through the parser's own adapter")
    print(f"  {ks['n_keyframes']} keyframes over {ks['span_s']:.0f} s "
          f"({ks['n_pose_rows_in_file']} pose rows, {ks['send_ok_dropped']} dropped on send_ok)")
    print(f"  horn command {ks['horn_cmd_deg']['min']:.0f} to {ks['horn_cmd_deg']['max']:.0f} deg "
          f"off neutral; steps median {ks['step_deg']['median']:.1f} deg, max "
          f"{ks['step_deg']['max']:.1f} deg ({ks['step_deg']['n_zero']} of {ks['step_deg']['n']} "
          f"repeat the previous pose)")
    print(f"  inter-keyframe gap median {ks['inter_keyframe_s']['median']:.2f} s, min "
          f"{ks['inter_keyframe_s']['min']:.2f} s, max {ks['inter_keyframe_s']['max']:.2f} s")

    print("\n== identification, same grid and protocol on both files")
    runs = []
    for path, label in ((log, "gesture (act)"), (walk, "walk-1 (official)")):
        r = run(model, path, grid, label)
        runs.append(r)
        a = r["argmin"]
        print(f"\n  {label}: {r['ticks']} ticks, identify on the first {r['fit_ticks']}")
        print(f"    argmin delay {a['delay_ticks']:.0f} ticks, slew {a['slew_rad_s']}, "
              f"deadband {np.rad2deg(a['deadband']):.0f} deg"
              + ("" if r["argmin_interior"] else "   [AT THE GRID BOUNDARY]"))
        print(f"    determined: delay {r['delay_determined']}  slew {r['slew_determined']}  "
              f"(band {r['band']:.5f})")
        if r["argmin_slew_is_grid_min"] and r["slew_undetermined_whole_grid"]:
            print(f"    the slew argmin is min(grid) = {r['grid_slew_min_rad_s']} rad/s and the "
                  f"determined set is the whole grid: that value is where a file which separates "
                  f"no hypothesis lands, not a measurement of the horn")
        print(f"    split-half: A={r['split_half']['A']['delay_ticks']:.0f}/"
              f"{r['split_half']['A']['slew_rad_s']}  B={r['split_half']['B']['delay_ticks']:.0f}/"
              f"{r['split_half']['B']['slew_rad_s']}  "
              f"{'AGREE' if r['split_half']['agree'] else 'DISAGREE'}")

    g, w = runs[0], runs[1]
    doc_step = documented_act_horn_deg(log)
    dur = act_duration_readings(ks, doc_step, w["slew_determined"])
    print("\n== the act duration the log does not record")
    print(f"  the header's one documented act is {DOC_ACT} -- {DOC_ACT_WHERE}")
    print(f"  read through the adapter, that pose is {doc_step:.2f} deg of horn travel FROM "
          f"NEUTRAL (an assumption: the example states no start pose)")
    for rd in dur["readings"]:
        if rd["name"] == "constant rate":
            print(f"  reading '{rd['name']}': {rd['rate_rad_s']:.4f} rad/s; a median step would "
                  f"take {rd['implied_act_ms']['median']:.0f} ms ({rd['implied_act_ms_note']}), "
                  f"the largest {rd['implied_act_ms']['max']:.0f} ms")
        else:
            print(f"  reading '{rd['name']}': commanded rate median "
                  f"{rd['implied_rate_rad_s']['median']:.2f} rad/s, max "
                  f"{rd['implied_rate_rad_s']['max']:.2f} rad/s; "
                  f"{rd['n_steps_at_or_above_walk_slew']} of {rd['n_steps']} steps at or above "
                  f"the walk lane's {rd['walk_slew_determined_rad_s']} rad/s")
        print(f"    assumption: {rd['assumption']}")
        print(f"    uncertainty: {rd['uncertainty']}")
        print(f"    {rd['frac_acts_interrupted'] * 100:.0f}% of acts are interrupted by the next "
              f"keyframe under this reading")
    print(f"  UNRESOLVED: {dur['consequence']}")

    out = {"keyframes": {k: v for k, v in ks.items() if not k.startswith("_")},
           "act_duration": dur, "runs": runs}
    out["reading"] = {
        "slew_interpretable": False,
        "slew_reason": (f"the argmin slew {g['argmin']['slew_rad_s']} rad/s is min(grid) and the "
                        f"determined set is every slew on the grid including 'no limit at all', at "
                        f"a band of {g['band']:.4f}: this file separates no slew hypothesis from any "
                        f"other, so its argmin is where the search stops, not what the horn does"),
        "delay_interpretable": False,
        "delay_reason": (f"the delay determined set is {g['delay_determined']} -- the whole grid -- "
                         f"and the split halves disagree"),
        "vs_walk": (f"the capture asked for determines strictly less than the walking file it was "
                    f"meant to improve on: band {g['band']:.4f} vs {w['band']:.4f} "
                    f"({g['band'] / w['band']:.0f}x wider), both parameters the entire grid vs "
                    f"delay {w['delay_determined']} and slew {w['slew_determined']}"),
        "cause_not_established": ("the log records target poses and send times but no act duration, "
                                  "so the command every hypothesis is scored against is not the "
                                  "command the body received. That is enough to explain an "
                                  "undetermined result and is NOT established as its cause by this "
                                  "file: what would settle it is the ms per act, or the realized "
                                  "30 Hz command stream"),
        "delay_tightened_vs_walk": len(g["delay_determined"]) < len(w["delay_determined"]),
    }
    print("\n== reading")
    print(f"  slew: {out['reading']['slew_reason']}")
    print(f"  delay: {out['reading']['delay_reason']}")
    print(f"  vs walk-1: {out['reading']['vs_walk']}")
    print(f"  cause: {out['reading']['cause_not_established']}")
    path = default_out_path([log], "gesture_id")
    out["provenance"] = provenance(seeds={"mlp": 0})
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="What an `act` (gesture) capture can and cannot identify about the servo: "
                    "keyframe statistics through the parser's adapter, the same identification "
                    "grid on the gesture file and on walk-1, and the act-duration readings the "
                    "log leaves open. Writes results/gesture_id_<stem>.json and prints the report "
                    "(tee it to results/logs/gesture_id.txt). Needs the untracked maintainer logs.")
    ap.add_argument("--log", default=LOG, help=f"gesture (act) capture (default {LOG})")
    ap.add_argument("--walk", default=WALK, help=f"walking capture to compare against (default {WALK})")
    a = ap.parse_args()
    main(a.log, a.walk)
