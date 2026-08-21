"""RETRACTED. Coverage experiment: does adding sit<->stand transitions to twin
training data close the real pitch gap that servo correction never touched?

*** This experiment is withdrawn. It is invalid twice over, and neither defect is
*** repairable by rerunning it as written. The code is kept, and the published
*** results file is kept with its conclusion marked retracted, because the record
*** of a wrong result belongs in the repository next to the correction.
***
*** 1. THE PREMISE IS FALSE. Neither real log contains a sit-to-stand. The header
***    field this experiment was built on says so in its own words: "legged done
***    walks fold to a sit act {l:130,r:50,ms:700} AFTER recording ends". The fold
***    happens after the recording, and it happens on walks that end "done" --
***    walk-3 ends "tipped" and never sits at all. The -1.0 rad pitch tail that
***    was read as a sit is a FALL: a 5 g event at t=73044 ms, rate_alpha 124-200
***    deg/s, ori_beta 1.8 -> 56 deg in 0.6 s, and the recording stops with the
***    body still down. The sit pose {l:130, r:50} appears in neither file's pose
***    stream. There was no missing motion to supply.
***
*** 2. THE MANIPULATION WAS NULL. The sanity precondition asked whether the
***    synthesized transitions reach walk-3's pitch excursions. They do -- and so
***    does the STANDARD data, which the same run measured and then never
***    compared: standard pitch spans -1.570 to +1.570 rad, transition pitch
***    -1.568 to +1.459, walk-3 -1.013 to +0.014. The augmented cells added no
***    pitch range the control did not already have, so the 2x2 varied nothing on
***    the axis it was built to test and its flat pitch measures nothing. The
***    precondition as written could not fail: it compared the treatment with the
***    target and skipped the control. s_std was computed, printed, and left out
***    of the test.
***
*** The check below has been replaced with one that would have caught this (the
*** treatment must add range the CONTROL lacks, and the target must lie outside
*** the control's range at all), and a premise check that refuses to run when the
*** logs do not contain the pose the synthesis is built around. Nothing else is
*** changed: the numbers this file produced stand as what it produced.

Original design: a 2x2 factorial so the coverage effect and the actuator effect
separate --

                        standard data          +transition data
    nominal servo       control (= baseline)   coverage-only
    corrected servo     real2sim replication   coverage + corrected

The corrected servo is real2sim's argmin (delay 4 ticks = 80 ms at the 50 Hz
caller, slew 2.0 rad/s, deadband 2 deg). Transition data is synthesized with the
keyframe machinery around the REAL sit pose documented in the log header (act
{l:130, r:50}, pushed through the same cal inversion the adapter uses), glided at
varied speeds, mixed with walking bursts so sit->stand->walk chains -- the exact
shape the tipped walk contains -- are in the data. Mix ratio 75/25: augmented
cells train on 300k standard + 100k transition ticks; standard cells on 400k.
All cells share collection seeds (standard: seed 0; transitions: seed 10) and the
MLP seed (0); the augmented cells' standard part is the exact 300k prefix of the
standard cells' collection (same rng stream), cut at the splice.

Decision rule, stated before the numbers: a cell's gain over the control on an
axis at 500 ms is material when it exceeds max(3.0 pts, 2x the control seed
spread measured by real2sim on the same held-out ticks) -- thresholds 7.2 / 3.0 /
3.8 pts for roll / pitch / yaw, from results/real2sim.json. Expectations:
  - coverage should move PITCH if the hypothesis is right (walk-3's walking part,
    which contains the sit-to-stand, most of all) and leave roll/yaw to the servo;
  - the corrected-servo replication should reproduce real2sim's roll/yaw closure;
  - the combined cell tests additivity: roll/yaw from the servo, pitch from
    coverage.
A flat pitch under coverage -- with the sanity precondition below satisfied --
kills the hypothesis, and that negative is published with equal prominence.

Sanity preconditions, as they should always have read (checked before any
conclusion is drawn, and both of them fail on this data):
  premise      the pose the synthesis is built around must actually occur in the
               logs the experiment scores. A coverage hole is a motion the real
               body performed and the twin never saw; if the real body never
               performed it, there is nothing to cover.
  manipulation the transition collection must reach pitch the STANDARD collection
               does not, and the real target must lie outside the standard
               collection's range. Comparing the treatment against the target
               while ignoring the control tests nothing.

Discipline: identification used the first half of the concatenated real log, so
every real-log number is the held-out second half only (535 ticks, 10.7 s). That
concatenation is itself withdrawn -- see real2sim.py, which now scores walk-1
alone -- and is left here only so this file reproduces what it published.
"""
from __future__ import annotations
import json, sys, time
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "sim")

from growbot_sim import GrowBotSim, ServoModel, WalkPolicy, collect, CTRL_HZ
from forward import MLP, make_windows
from sim2real_proxy import K
from gap_report import evaluate_axes, AXES
from imulog import parse, run_preflight

LOGS = ["imu-walk-1-2026-08-20T17-50-14-713Z.json",
        "imu-walk-3-2026-08-20T17-38-19-478Z.json"]
TRAIN_STEPS, TEST_STEPS = 400_000, 60_000
STD_STEPS, TRANS_STEPS = 300_000, 100_000       # the 75/25 mix of the augmented cells
TRAIN_SEED, TEST_SEED, TRANS_SEED = 0, 1, 10
EPOCHS, HIDDEN = 80, 128
HORIZONS = [5, 25]                              # 100 / 500 ms
TIP_WINDOW_S = 1.5
SIT_ACT_DEG = {"l": 130.0, "r": 50.0}           # header.post_walk act, servo degrees
SERVO_KW = dict(delay_ticks=4, slew_rad_s=2.0, deadband=float(np.deg2rad(2)))



def thresholds():
    """2x seed spread on the control, measured by real2sim on the same held-out ticks
    (results/real2sim.json control_seed_spread_500ms); rule = max(3.0 pts, 2x spread).

    Read inside main(), never at import: a module-level json.load makes importing this
    file fail on any checkout where results/real2sim.json is absent or stale, which is
    exactly the checkout someone runs the experiments in.
    """
    spread = json.load(open("results/real2sim.json"))["control_seed_spread_500ms"]
    return {ax: max(0.03, 2 * spread[ax]) for ax in AXES}


class Tee:
    def __init__(self, path):
        self.f = open(path, "w")
    def write(self, s):
        sys.__stdout__.write(s); self.f.write(s)
    def flush(self):
        sys.__stdout__.flush()
        if not self.f.closed:
            self.f.flush()


def sit_pose_from_header(header):
    """The real sit pose {l:130, r:50} through the adapter's exact cal inversion.

    parse() exposes the cal as l_sign/r_sign/l_off/r_off/gain/turn; the upstream
    servo map (sim/growbot_policy.js) is l = 90 + L_OFF + L_SIGN*deg(aL)*gain + turn,
    r = 90 + R_OFF + R_SIGN*deg(aR)*gain - turn, action order [a0=right, a1=left].
    With this log's cal (signs -1/-1, offs 0, gain 0.99, turn 0) the sit lands at
    [+0.705, -0.705] rad.
    """
    ls, rs = header["l_sign"], header["r_sign"]
    lo, ro = header["l_off"], header["r_off"]
    g, turn = header["gain"], header["turn"]
    a_right = np.deg2rad((SIT_ACT_DEG["r"] - 90 - ro + turn) / (rs * g))
    a_left = np.deg2rad((SIT_ACT_DEG["l"] - 90 - lo - turn) / (ls * g))
    return np.array([a_right, a_left], np.float32)


def load_real_heldout():
    """Concatenated real log with per-file labels, held-out second half only."""
    parts, sit = [], None
    for f in LOGS:
        if not run_preflight(f):
            raise SystemExit(f"preflight FAIL on {f}")
        O, A, O2, D, header, mode = parse(f)
        tag = f"official_w{header.get('walk')}"
        label = np.full(len(O), tag, dtype=object)
        if header.get("end_why") == "tipped":
            label[-int(TIP_WINDOW_S * 50):] = "tip_onset"
        if sit is None:
            sit = sit_pose_from_header(header)
        parts.append((O, A, O2, D, label.astype(str)))
    O, A, O2, D, label = (np.concatenate(x) for x in zip(*parts))
    half = len(O) // 2                  # identification used [0:half]; never touched here
    held = slice(half, None)
    return O[held], A[held], D[held], label[held], half, len(O), sit


class TransitionExcitation:
    """Sit<->stand<->walk sequences: glide to a pose, hold it, sometimes walk.

    Pose family (probabilities in _pick_pose): 40 % sit variants -- the real fold
    scaled 0.75-1.20 with small noise, so the depths bracet the logged one; 25 %
    stand/neutral; 20 % walk stance; 15 % anywhere, for breadth. Glides are linear
    in command space over 0.3-1.5 s (the app's own /act glide, 700 ms, sits inside
    the range); holds last 0.5-2 s; walking bursts (20 % of segments) run the
    shipped policy 1-3 s so sit->stand->walk chains occur.
    """
    def __init__(self, rng, sit_pose):
        self.rng, self.sit = rng, sit_pose
        try:
            self.policy = WalkPolicy()
        except Exception:
            self.policy = None
        self.mode = "hold"; self.t = 0; self.len = 0
        self.a0 = np.zeros(2, np.float32); self.a1 = np.zeros(2, np.float32)
        self.energy = 1.0

    def _pick_pose(self):
        r = self.rng.random()
        if r < 0.40:
            p = self.sit * self.rng.uniform(0.75, 1.20) + self.rng.normal(0, 0.08, 2)
        elif r < 0.65:
            p = self.rng.normal(0, 0.10, 2)
        elif r < 0.85:
            p = self.rng.uniform(-0.45, 0.45, 2)
        else:
            p = self.rng.uniform(-1.2, 1.2, 2)
        return np.clip(p, -1.57, 1.57).astype(np.float32)

    def new_segment(self, prev):
        r = self.rng.random()
        if self.policy is not None and r < 0.20:
            self.mode = "policy"
            self.len = int(self.rng.uniform(1.0, 3.0) * CTRL_HZ)
            self.energy = self.rng.uniform(0.5, 1.0)
            self.policy.reset()
        elif r < 0.60:
            self.mode = "glide"
            self.a0, self.a1 = prev.astype(np.float32).copy(), self._pick_pose()
            self.len = max(1, int(self.rng.uniform(0.3, 1.5) * CTRL_HZ))
        else:
            self.mode = "hold"
            self.len = int(self.rng.uniform(0.5, 2.0) * CTRL_HZ)
        self.t = 0

    def __call__(self, obs, prev):
        if self.t >= self.len:
            self.new_segment(prev)
        self.t += 1
        if self.mode == "policy":
            a = self.energy * self.policy(obs)
        elif self.mode == "glide":
            a = self.a0 + (self.a1 - self.a0) * (self.t / self.len)
        else:
            a = prev
        return np.clip(a, -1.57, 1.57).astype(np.float32)


def collect_transitions(n_steps, seed, sit_pose, servo=None):
    """collect()'s episode/push/reset protocol with TransitionExcitation actions."""
    sim = GrowBotSim(seed, servo=servo)
    exc = TransitionExcitation(sim.rng, sit_pose)
    O = np.zeros((n_steps, 6), np.float32); A = np.zeros((n_steps, 2), np.float32)
    O2 = np.zeros((n_steps, 6), np.float32); D = np.zeros(n_steps, bool)
    modes = []
    o = sim.reset(tilt=0.3)
    prev = np.zeros(2, np.float32)
    ep_len = int(8.0 * CTRL_HZ)
    t_ep = 0
    for i in range(n_steps):
        a = exc(o, prev)
        if sim.rng.random() < 0.01:
            sim.push()
        o2 = sim.step(a)
        O[i], A[i], O2[i] = o, a, o2
        modes.append("policy" if exc.mode == "policy" else "transition")
        t_ep += 1
        end = t_ep >= ep_len or (sim.fallen() and sim.rng.random() < 0.02)
        D[i] = end
        if end:
            o = sim.reset(tilt=1.0 if sim.rng.random() < 0.2 else 0.3)
            prev = np.zeros(2, np.float32); t_ep = 0
            exc.new_segment(prev)
        else:
            o, prev = o2, a
    D[-1] = True
    return O, A, O2, D, np.array(modes)


def pitch_stats(O, name):
    p = O[:, 1]
    st = {"min": float(p.min()), "p1": float(np.percentile(p, 1)),
          "p50": float(np.percentile(p, 50)), "p99": float(np.percentile(p, 99)),
          "max": float(p.max())}
    print(f"  pitch {name:<24} min {st['min']:+.2f}  p1 {st['p1']:+.2f}  "
          f"p50 {st['p50']:+.2f}  p99 {st['p99']:+.2f}  max {st['max']:+.2f} rad")
    return st


def splice(std, trans):
    """300k standard prefix + 100k transitions, cut at the seam."""
    Os, As, O2s, Ds, Ms = std
    Ot, At, O2t, Dt, Mt = trans
    Ds = Ds[:STD_STEPS].copy(); Ds[-1] = True
    return (np.concatenate([Os[:STD_STEPS], Ot]), np.concatenate([As[:STD_STEPS], At]),
            np.concatenate([O2s[:STD_STEPS], O2t]), np.concatenate([Ds, Dt]),
            np.concatenate([Ms[:STD_STEPS], Mt]))


def main():
    sys.stdout = tee = Tee("results/logs/coverage.txt")
    t0 = time.time()
    THRESH = thresholds()
    Oh, Ah, Dh, labelh, half, total, sit = load_real_heldout()
    print(f"\nreal log: {total} ticks, evaluation = held-out [{half}:{total}] "
          f"({len(Oh)} ticks, {len(Oh)/50:.1f} s), labels "
          f"{ {m: int((labelh == m).sum()) for m in sorted(set(labelh))} }")
    print(f"sit pose through the adapter cal: [{sit[0]:+.3f}, {sit[1]:+.3f}] rad "
          f"(action order [right, left])")

    report = {"conditions": {
        "logs": LOGS, "train_steps": TRAIN_STEPS, "std_steps": STD_STEPS,
        "trans_steps": TRANS_STEPS, "mix": "augmented cells: 75% standard / 25% transition",
        "train_seed": TRAIN_SEED, "trans_seed": TRANS_SEED, "test_seed": TEST_SEED,
        "epochs": EPOCHS, "hidden": HIDDEN, "K": K, "mlp_seed": 0,
        "horizons_ms": [h * 20 for h in HORIZONS], "held_out_ticks": int(len(Oh)),
        "corrected_servo": {**SERVO_KW, "note": "real2sim argmin; delay in ticks at the 50 Hz caller"},
        "sit_pose_rad": [float(sit[0]), float(sit[1])],
        "rule": "material @500ms = gain over control > max(3.0 pts, 2x real2sim control seed spread)",
        "thresholds_pts": {ax: round(THRESH[ax] * 100, 1) for ax in AXES},
    }, "cells": {}}

    # -- collections (standard reused across cells: the augmented standard part is
    #    the exact 300k prefix of the same rng stream)
    print("\n== collections")
    std_nom = collect(TRAIN_STEPS, TRAIN_SEED, return_realized=False)
    tr_pub = np.load("data/train.npz")
    assert np.array_equal(std_nom[0], tr_pub["obs"]) and np.array_equal(std_nom[1], tr_pub["act"]), \
        "nominal standard collection does not reproduce data/train.npz"
    print("  nominal standard collection asserted equal to data/train.npz")
    std_cor = collect(TRAIN_STEPS, TRAIN_SEED, servo=ServoModel(**SERVO_KW))
    trans_nom = collect_transitions(TRANS_STEPS, TRANS_SEED, sit, servo=None)
    trans_cor = collect_transitions(TRANS_STEPS, TRANS_SEED, sit, servo=ServoModel(**SERVO_KW))

    print("\n== sanity preconditions")
    # (a) PREMISE. The sit pose the synthesis is built around must occur in the logs
    # this experiment scores. It does not: the header says the fold happens after
    # recording ends, and the pose stream agrees. The original version never asked.
    MARGIN = 0.05
    SIT_TOL_RAD, SIT_DWELL_TICKS = 0.15, 15      # 0.3 s held at the pose, at 50 Hz
    sit_seen = {}
    for f in LOGS:
        _, A_f, _, _, h_f, _ = parse(f)
        d = np.abs(A_f - sit[None, :]).max(1)
        near = d <= SIT_TOL_RAD
        # A HELD pose, not a passing one. An alternating gait sweeps its whole command
        # range, so isolated ticks near the sit pose are the walk going past it; a
        # sit-to-stand is a dwell. Counting bare ticks would have called walk-1's 8
        # scattered near-misses a sit.
        run = best = 0
        for v in near:
            run = run + 1 if v else 0
            best = max(best, run)
        sit_seen[f] = {"closest_rad": round(float(d.min()), 3),
                       "ticks_within_tol": int(near.sum()),
                       "longest_dwell_ticks": int(best),
                       "dwell_required": SIT_DWELL_TICKS,
                       "end_why": h_f.get("end_why")}
        print(f"  premise: {f} end_why={h_f.get('end_why')!r}, closest command to the sit pose "
              f"{d.min():.3f} rad away, {int(near.sum())} ticks within {SIT_TOL_RAD} rad, "
              f"longest HELD run {best} ticks (need {SIT_DWELL_TICKS} = 0.3 s)")
    premise = any(v["longest_dwell_ticks"] >= SIT_DWELL_TICKS for v in sit_seen.values())
    print(f"  -> the sit pose {'occurs' if premise else 'NEVER OCCURS'} in these logs: the "
          f"coverage premise is {'live' if premise else 'FALSE, and no result below means anything'}")

    # (b) MANIPULATION. The treatment must add pitch range the CONTROL lacks, and the
    # real target must lie outside the control's range at all -- otherwise the 2x2
    # varies nothing on the axis it exists to test. Comparing the treatment against
    # the target and skipping the control (what the original check did) cannot fail.
    print("\n  manipulation: does +transitions add pitch the standard data does not have?")
    s_w3 = pitch_stats(Oh[np.char.startswith(labelh.astype(str), "official_w3") |
                          (labelh == "tip_onset")], "walk-3 (held-out)")
    s_std = pitch_stats(std_nom[0], "standard 400k")
    s_trn = pitch_stats(trans_nom[0], "transitions 100k")
    adds_range = (s_trn["min"] < s_std["min"] - MARGIN) or (s_trn["max"] > s_std["max"] + MARGIN)
    target_outside = (s_w3["min"] < s_std["min"]) or (s_w3["max"] > s_std["max"])
    covered = bool(premise and adds_range and target_outside)
    report["sanity"] = {"walk3_pitch": s_w3, "standard_pitch": s_std,
                        "transition_pitch": s_trn,
                        "premise_sit_pose_in_logs": bool(premise), "sit_pose_search": sit_seen,
                        "transitions_add_range_over_standard": bool(adds_range),
                        "target_outside_standard_range": bool(target_outside),
                        "covered": covered}
    print(f"  transitions add range beyond standard: {adds_range} "
          f"(standard [{s_std['min']:+.2f}, {s_std['max']:+.2f}], "
          f"transitions [{s_trn['min']:+.2f}, {s_trn['max']:+.2f}])")
    print(f"  walk-3's pitch lies outside standard's range: {target_outside} "
          f"(walk-3 [{s_w3['min']:+.2f}, {s_w3['max']:+.2f}])")
    print(f"  -> a negative below is {'interpretable' if covered else 'NOT interpretable'}")

    cells = {
        "control (nominal, standard)": (None, std_nom),
        "coverage (nominal, +transitions)": (None, splice(std_nom, trans_nom)),
        "corrected (argmin servo, standard)": (SERVO_KW, std_cor),
        "coverage+corrected": (SERVO_KW, splice(std_cor, trans_cor)),
    }
    for name, (kw, data) in cells.items():
        print(f"\n== {name}")
        O, A, O2, D, M = data
        te = {}
        te["obs"], te["act"], te["next_obs"], te["done"], te["mode"] = collect(
            TEST_STEPS, TEST_SEED, servo=ServoModel(**kw) if kw else None)
        Xtr, Ytr, *_ = make_windows(O, A, O2, D, K)
        model = MLP(hidden=HIDDEN, epochs=EPOCHS, seed=0).fit(Xtr, Ytr)
        floor = evaluate_axes(model, te["obs"], te["act"], te["done"],
                              te["mode"].astype(str), HORIZONS)
        real = evaluate_axes(model, Oh, Ah, Dh, labelh, HORIZONS)
        cell = {"servo": kw, "train_ticks": int(len(O)),
                "twin_floor_policy": {str(h * 20): {ax: floor["policy"][h][ax]["within"]
                                                    for ax in AXES} for h in HORIZONS},
                "real_heldout": {reg: {str(h * 20): {ax: real[reg][h][ax]["within"]
                                                     for ax in AXES} for h in HORIZONS}
                                 for reg in real}}
        for reg in real:
            cell["real_heldout"].setdefault(reg, {})["n"] = real[reg]["n"]
        report["cells"][name] = cell
        for h in HORIZONS:
            row = "  ".join(f"{ax} {real['all'][h][ax]['within']*100:5.1f}" for ax in AXES)
            print(f"  real held-out (all) @{h*20:>3}ms  {row}")
        for reg in sorted(set(real) - {"all"}):
            row = "  ".join(f"{ax} {real[reg][25][ax]['within']*100:5.1f}" for ax in AXES)
            print(f"    {reg:<14} n={real[reg]['n']:<4} @500ms  {row}")

    print("\n== closure @500 ms vs control (all row; within 0.2 rad; "
          "threshold = max(3.0, 2x real2sim control spread) pts)")
    ctrl = report["cells"]["control (nominal, standard)"]["real_heldout"]["all"]["500"]
    verdict = {}
    for name, cell in report["cells"].items():
        if name.startswith("control"):
            continue
        row, v = "", {}
        for ax in AXES:
            gain = cell["real_heldout"]["all"]["500"][ax] - ctrl[ax]
            v[ax] = {"gain_pts": round(gain * 100, 1),
                     "threshold_pts": round(THRESH[ax] * 100, 1),
                     "material": bool(gain > THRESH[ax])}
            row += f"  {ax} {gain*100:+5.1f}{' MATERIAL' if v[ax]['material'] else ''}"
        verdict[name] = v
        print(f"  {name:<36}{row}")
    report["closure_verdict"] = verdict

    cov = verdict["coverage (nominal, +transitions)"]
    cor = verdict["corrected (argmin servo, standard)"]
    both = verdict["coverage+corrected"]
    addv = {ax: {"expected_pts": round(cov[ax]["gain_pts"] + cor[ax]["gain_pts"], 1),
                 "observed_pts": both[ax]["gain_pts"]} for ax in AXES}
    report["additivity"] = addv
    print("\n== additivity (combined vs coverage-gain + servo-gain)")
    for ax in AXES:
        print(f"  {ax}: expected {addv[ax]['expected_pts']:+.1f}  observed {addv[ax]['observed_pts']:+.1f}")

    pitch_moved = cov["pitch"]["material"] or both["pitch"]["material"]
    if not covered:
        why = []
        if not premise:
            why.append("the sit pose the transitions are built around occurs in NEITHER log "
                       "(the header puts the fold after the recording ends, and the -1.0 rad "
                       "tail is a fall), so there is no missing motion to supply")
        if not adds_range:
            why.append(f"the transitions add no pitch range the standard data lacks "
                       f"(standard [{s_std['min']:+.2f}, {s_std['max']:+.2f}] vs transitions "
                       f"[{s_trn['min']:+.2f}, {s_trn['max']:+.2f}]), so the 2x2 varies nothing "
                       f"on the axis it exists to test")
        if not target_outside:
            why.append(f"walk-3's pitch [{s_w3['min']:+.2f}, {s_w3['max']:+.2f}] already lies "
                       f"inside the standard data's range, so a coverage hole of this kind "
                       f"cannot exist")
        conclusion = ("sanity preconditions FAILED -- NO CONCLUSION about the coverage "
                      "hypothesis can be drawn from these cells: " + "; ".join(why))
    elif pitch_moved:
        conclusion = ("coverage hypothesis CONFIRMED: transition data moves pitch materially "
                      "where servo correction never did")
    else:
        conclusion = ("coverage hypothesis KILLED (negative): transition data reaches walk-3's "
                      "excursions yet pitch stays under threshold -- the pitch gap is not a "
                      "training-coverage hole of this kind")
    if covered and cor["roll"]["material"] and cor["yaw"]["material"]:
        conclusion += "; corrected-servo replication reproduces real2sim's roll/yaw closure"
    report["conclusion"] = conclusion
    # Additivity is a difference of differences on a manipulation that changed nothing;
    # and the differences here (1.5 and 1.2 pts) sit inside one control seed spread.
    report["additivity_caveat"] = (
        "the observed-vs-expected differences are within one control MLP seed spread on "
        "these held-out ticks, so this supports 'consistent with additivity within noise', "
        "never 'the two effects are additive'")
    print(f"\n  {conclusion}")

    json.dump(report, open("results/coverage.json", "w"), indent=1)
    print(f"\nwrote results/coverage.json   total {(time.time()-t0)/60:.1f} min")
    tee.f.close()


if __name__ == "__main__":
    main()
