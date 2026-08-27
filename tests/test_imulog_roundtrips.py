"""The round-trip suite: hidden secrets through the parser, the segmenter, the identification
and the sensor-side estimators, and every one of them must come back.

This is the suite `python imulog.py` (no arguments) runs; it used to live in that file's
`__main__`. Every assertion and every tolerance is the one that lived there -- a
tolerance loosened here is a secret that stopped being tested. Marked `slow` (~5 min on
CPU): `pytest -m "not slow"` skips it, `pytest` runs it, and CI runs it.

Fixtures are module-scoped so the 600 s twin session and the 60-epoch forward model are
generated once and shared, as they were when this was one script.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from growbot_cerebellum.imulog import (parse, fixture, _selfcheck, _dialect_side_check, _jsonl_to_csv,
                                       _jsonl_to_growbot_v1, _read_rows, GAIT_DRIVEN_LABEL)
from growbot_cerebellum.forward import MLP, make_windows, K
from growbot_cerebellum.sim2real import horizon_within
from growbot_cerebellum.sim import collect
from growbot_cerebellum.servo_id import (identify, realized_from_commands, confidence_band, determined_sets,
                                         default_grid, identify_per_side, PerSideServo, slower_side,
                                         sim_side_columns, check_side_convention)
from growbot_cerebellum.sensor_id import (dt_stats, allan_deviation, filter_lag, still_windows,
                                          euler_rates_to_body, verify_still, segment_rate, BODY_AXES)

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "train.npz"
TRUE = dict(delay_ms=40, slew_rad_s=5.0, deadband=np.deg2rad(2))


# ----------------------------------------------------------------------------
# shared fixtures: one 600 s session, one parse, one forward model
# ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("imulog")


@pytest.fixture(scope="module")
def walk_jsonl(tmp):
    path = tmp / "imulog_fixture.jsonl"
    print("generating 600 s fixture (hidden servo: delay 40 ms, slew 5 rad/s, deadband 2 deg)...", flush=True)
    fixture(str(path), seconds=600, servo_ms=TRUE, seed=3)
    return path


@pytest.fixture(scope="module")
def parsed(walk_jsonl):
    O, A, O2, D, header, mode = parse(str(walk_jsonl))
    print(f"parsed: {len(O):,} ticks at 50 Hz, {int(D.sum())} episode splits, "
          f"regimes {dict(Counter(mode))}")
    return O, A, O2, D, header, mode


@pytest.fixture(scope="module")
def walk_v1(tmp, walk_jsonl):
    path = tmp / "imulog_fixture_v1.json"
    _jsonl_to_growbot_v1(str(walk_jsonl), str(path))
    return path


@pytest.fixture(scope="module")
def parsed_v1(walk_v1):
    return parse(str(walk_v1))


@pytest.fixture(scope="module")
def model():
    if not TRAIN.exists():
        pytest.skip(f"{TRAIN} missing: regenerate it with the sim/growbot_sim.py commands in the README")
    tr = np.load(TRAIN)
    Xtr, Ytr, *_ = make_windows(tr["obs"], tr["act"], tr["next_obs"], tr["done"], K)
    return MLP(hidden=128, epochs=60).fit(Xtr, Ytr)


@pytest.fixture(scope="module")
def grid():
    # the grid that SHIPS, not a copy of it: there were three copies of a narrow grid
    # (here, servo_id's CLI, gap_report's), and the one the real log needed was a fourth,
    # written inline in the real-log report. A test that exercises a private copy cannot
    # catch a default that pins at its own boundary.
    return default_grid()


def _slews(grid):
    return sorted({s for _, s, _ in grid}, key=lambda v: (v is None, v))


@pytest.fixture(scope="module")
def shared_fit(model, parsed, grid):
    """The shared (both-horn) identification on the first half, plus the split-half band."""
    O, A, O2, D, _, _ = parsed
    half = len(O) // 2
    fit = slice(0, half)
    scores, best = identify(model, O[fit], A[fit], O2[fit], D[fit], grid)
    hA, hB = slice(0, half // 2), slice(half // 2, half)
    sA, _ = identify(model, O[hA], A[hA], O2[hA], D[hA], grid)
    sB, _ = identify(model, O[hB], A[hB], O2[hB], D[hB], grid)
    band = confidence_band(sA, sB)
    return dict(half=half, scores=scores, best=best, band=band)


# ----------------------------------------------------------------------------
# the parser's dialects
# ----------------------------------------------------------------------------

def test_trim_convention():
    _selfcheck()


def test_dialect_side_convention(tmp):
    _dialect_side_check(str(tmp / "_imulog_side_check.jsonl"))


def test_csv_fallback(tmp, walk_jsonl, parsed):
    O, A, O2, D, header, mode = parsed
    csv = tmp / "imulog_fixture.csv"
    _jsonl_to_csv(str(walk_jsonl), str(csv))
    Oc, Ac, O2c, Dc, hc, mc = parse(str(csv))
    same = (np.allclose(O, Oc, atol=1e-4) and np.allclose(A, Ac, atol=1e-4)
            and (D == Dc).all() and (mode == mc).all())
    print(f"CSV fallback: {'PASS — identical arrays from both formats' if same else 'FAIL'}")
    assert same


def test_growbot_v1_roundtrip(parsed, parsed_v1):
    # growbot-imulog-1: the same session through the real upstream format. Twin rpy ->
    # mount -> W3C ZX'Y'' degrees -> back, body rates -> device rates -> back, actions
    # -> upstream servo map (signs, offsets, gain, nonzero turn) -> back. A wrong
    # composition order, a wrong extraction branch, a swapped l/r or an unfolded turn
    # all break the equality; the mount itself is validated against the real logs
    # (gravity, stance, action-response signatures), not here, because M M^T = I makes
    # any mount self-consistent in a round trip.
    O, A, O2, D, header, mode = parsed
    Og, Ag, O2g, Dg, hg, mg = parsed_v1
    ang_err = float(np.abs(np.arctan2(np.sin(O[:, :3] - Og[:, :3]), np.cos(O[:, :3] - Og[:, :3]))).max())
    gyro_err = float(np.abs(O[:, 3:] - Og[:, 3:]).max())
    act_err = float(np.abs(A - Ag).max())
    same_g = (ang_err < 1e-4 and gyro_err < 1e-4 and act_err < 1e-4 and (D == Dg).all()
              and hg.get("send_ok_dropped") == 0)
    print(f"growbot-imulog-1: {'PASS' if same_g else 'FAIL'} — max errs angle {ang_err:.1e} rad, "
          f"gyro {gyro_err:.1e} rad/s, action {act_err:.1e} rad through mount + W3C angles + servo cal "
          f"(signs, offsets, gain 0.99, turn 1.5)")
    assert same_g, "growbot-imulog-1 dialect does not round-trip"


def test_empty_command_capture(tmp, walk_v1, parsed_v1):
    # The still lane sends nothing, so its pose array is empty. Strip every command from
    # a file we already know the answer for: BOTH IMU halves must survive -- the fused
    # orientation as well as the body rates, since the pose array is the only thing
    # removed -- while the commands become neutral and the record keeps a still segment.
    # A parser that rejected the empty array, or invented a command to fill it, fails.
    Og, Ag, O2g, Dg, hg, mg = parsed_v1
    with open(walk_v1) as fh:
        _nc = json.load(fh)
    _nc["pose"] = []
    _nc["header"] = dict(_nc["header"], gait="still", end_why="tap")
    nocmd = tmp / "imulog_nocmd_v1.json"
    with open(nocmd, "w") as fh:
        json.dump(_nc, fh)
    On, An, O2n, Dn, hn, mn = parse(str(nocmd))
    ori_err_n = float(np.abs(np.arctan2(np.sin(On[:, :3] - Og[:, :3]),
                                        np.cos(On[:, :3] - Og[:, :3]))).max())
    gyro_err_n = float(np.abs(On[:, 3:] - Og[:, 3:]).max())
    # 'walking'/'acting' mean the agent was commanding the servos, and this file records
    # none. With an empty pose array the driven branch is unreachable by construction --
    # cmd_active is identically False -- so its absence here proves nothing on its own.
    # What binds is the CONTRAST with the same IMU under its commands: the source file
    # must carry driven ticks, this one must carry none, and the ticks that lost their
    # label must become 'unknown' rather than 'still', i.e. the label tracks the command
    # stream and never the motion.
    _driven = set(GAIT_DRIVEN_LABEL.values()) | {"walking"}
    mg_arr, mn_arr = np.asarray(mg), np.asarray(mn)
    was_driven = np.isin(mg_arr, list(_driven))
    lost = set(np.unique(mn_arr[was_driven])) if was_driven.any() else set()
    nocmd_ok = (len(On) == len(Og) and gyro_err_n < 1e-9 and ori_err_n < 1e-9
                and float(np.abs(An).max()) == 0.0
                and was_driven.any() and not (set(np.unique(mn_arr)) & _driven)
                and lost == {"unknown"}
                and "still" in set(np.unique(mn_arr)) and hn.get("n_pose_rows") == 0)
    print(f"empty-command capture: {'PASS' if nocmd_ok else 'FAIL'} — {len(On)} ticks, IMU identical "
          f"on both halves (max orientation err {ori_err_n:.1e} rad, gyro {gyro_err_n:.1e} rad/s), "
          f"actions all neutral; the {int(was_driven.sum())} driven ticks of the same file WITH "
          f"commands all become {sorted(lost)} here (labels {sorted(set(np.unique(mn_arr)))})")
    assert nocmd_ok, "a capture with no commands must parse, hold neutral and read as still"


# ----------------------------------------------------------------------------
# growbot-imulog-1 segmentation: the regimes the format does not carry
# ----------------------------------------------------------------------------

def test_segmentation(tmp):
    # Hidden secrets: a 10 s motionless prefix, driven walking after it, and a tip at
    # 34.0 s that the body never gets up from. The file the segmenter sees carries no
    # event row at all -- only end_why in the header -- so every boundary below has to
    # come out of the IMU and pose streams.
    SEG = dict(still_lead_s=10.0, tip_at_s=34.0, seconds=45.0, seed=11)
    print(f"\ngenerating {SEG['seconds']:.0f} s segmentation fixture (hidden: still prefix "
          f"{SEG['still_lead_s']:.0f} s, tip at {SEG['tip_at_s']:.0f} s)...", flush=True)
    seg_jsonl = tmp / "imulog_seg_fixture.jsonl"
    seg_v1 = tmp / "imulog_seg_v1.json"
    fixture(str(seg_jsonl), seconds=SEG["seconds"], seed=SEG["seed"],
            still_lead_s=SEG["still_lead_s"], tip_at_s=SEG["tip_at_s"])
    _jsonl_to_growbot_v1(str(seg_jsonl), str(seg_v1), end_why="tipped")
    hs, (its, ivs, _, _), evs = _read_rows(str(seg_v1))
    t0 = its[0]
    segs = [((t - t0) / 1000.0, n.removesuffix("_start")) for t, n in evs]
    span = [(a, b, n) for (a, n), (b, _) in zip(segs, segs[1:] + [((its[-1] - t0) / 1000.0, "")])]
    print("  segments: " + " | ".join(f"{a:.1f}-{b:.1f}s {n}" for a, b, n in span))
    first_end = span[0][1]
    fall = [s for s in span if s[2] == "fall"]
    still_ok = span[0][2] == "still" and abs(first_end - SEG["still_lead_s"]) <= 0.5
    walk_ok = any(n == "walking" and b - a >= 2.0 for a, b, n in span)
    fall_ok = (len(fall) == 1 and fall[0] is span[-1]
               and abs(fall[0][0] - SEG["tip_at_s"]) <= 1.0)
    print(f"  still prefix {span[0][2]!r} ends {first_end:.2f} s (injected "
          f"{SEG['still_lead_s']:.1f}): {'ok' if still_ok else 'WRONG'}")
    print(f"  a driven walking segment of >= 2 s exists: {'ok' if walk_ok else 'MISSING'}")
    print(f"  fall {'at %.2f s to the end' % fall[0][0] if fall else 'NOT FOUND'} (injected tip "
          f"{SEG['tip_at_s']:.1f}): {'ok' if fall_ok else 'WRONG'}")
    assert still_ok, f"the motionless prefix did not come out as one still segment: {span[:2]}"
    assert walk_ok, f"no driven walking segment recovered: {span}"
    assert fall_ok, f"the injected tip did not come out as the final fall segment: {span}"
    # the labels must survive their own physics check, or they are decoration.
    # PER SEGMENT, never pooled: two still segments held at different poses pool into
    # the offset between them, which verify_still's docstring calls a tilt, not motion.
    checks = {"still": [], "walking": []}
    for a, b, n in span:
        if n not in checks: continue
        m = ((its - t0) / 1000.0 >= a) & ((its - t0) / 1000.0 < b)
        if m.sum() >= 20:
            checks[n].append(verify_still(ivs[m, :3], ivs[m, 3:])["still"])
    print(f"  labelled still verifies still on {sum(checks['still'])}/{len(checks['still'])} "
          f"segments; labelled walking on {sum(checks['walking'])}/{len(checks['walking'])} "
          f"(must be 0)")
    assert checks["still"] and all(checks["still"]), "a labelled still segment is not still"
    assert not any(checks["walking"]), "a labelled walking segment verifies as still"
    # negative control: the SAME bytes with end_why 'done' must produce NO fall. The
    # class is gated on the header's own claim, never invented from attitude alone.
    seg_done = tmp / "imulog_seg_v1_done.json"
    _jsonl_to_growbot_v1(str(seg_jsonl), str(seg_done), end_why="done")
    _, _, ev_done = _read_rows(str(seg_done))
    n_fall_done = sum(1 for _, n in ev_done if n == "fall_start")
    print(f"  same session with end_why='done': {n_fall_done} fall segments (must be 0)")
    assert n_fall_done == 0, "a fall was invented on a session the header does not call tipped"
    print("SEGMENTATION PASS - still prefix, driven walking and the tip recovered from a "
          "format that carries no event rows")


def test_cut_coherence(parsed):
    # permanent detector for the cut-boundary bug family: no valid window may have a
    # target that crosses a cut, and every cut must cost at least one window
    O, A, O2, D, _, _ = parsed
    *_, valid = make_windows(O, A, O2, D, K)
    assert not (valid & D).any(), "a window whose target crosses a cut leaked into make_windows"
    print(f"cut coherence: {int(D.sum())} cut transitions, none inside valid windows "
          f"({int(valid.sum()):,} valid of {len(D):,})")


# ----------------------------------------------------------------------------
# servo identification on the parsed log
# ----------------------------------------------------------------------------

def test_servo_determined_sets(model, parsed, grid, shared_fit):
    O, A, O2, D, _, _ = parsed
    half, scores, best, band = shared_fit["half"], shared_fit["scores"], shared_fit["best"], shared_fit["band"]
    held = slice(half, None)
    print("\nservo identification on the PARSED log (top 3):")
    for e, kw in scores[:3]:
        print(f"  err {e:.4f}  delay {kw['delay_ticks']}  slew {kw['slew_rad_s']}  deadband {np.rad2deg(kw['deadband']):.0f} deg")
    R_est = realized_from_commands(A, D, best)
    for h in (5, 25):
        c = horizon_within(model, O[held], A[held], D[held], h=h)[0]
        e = horizon_within(model, O[held], R_est[held], D[held], h=h)[0]
        print(f"  {h*20:>3} ms  within 0.2 rad: commanded {c*100:5.1f}%  identified servo {e*100:5.1f}%")
    # Acceptance is servo_id's own standard -- the determined SET, not the argmin.
    # The valley here is nearly flat (the top three hypotheses differ by ~2e-4 on
    # errors of ~0.4), so the argmin is a coin flip, and asserting it was a test that
    # passed for the wrong reason: it leaned on the ~30 rows the fixture used to
    # backfill at 200 Hz after every repositioning gap, right where the post-reset
    # transient is most informative. With that artifact gone the argmin lands one
    # grid step off the injected delay on two of the five seeds tested (0/3/5/7/11),
    # while the determined set still contains the truth on all five.
    #
    # The bound asserted below is what held on all five seeds, not what looks tidy:
    #   delay  the injected 2 ticks is IN the set, and the set never leaves 2 +-1
    #          grid step (measured sets: [1,2,3] [1,2] [2] [1,2] [1,2,3])
    #   slew   every member of the set is within one grid step of the injected
    #          5.0 rad/s (measured: [4,5,6] [4,5] [4,5,6] [4] [4,5,6]) -- containment
    #          is NOT asserted, because seed 7 determines [4.0] and excludes the
    #          truth. The 200 Hz burst was the fixture's most slew-informative data;
    #          without it this log resolves slew to one grid step, and claiming more
    #          would be the same mistake in a new place.
    # Both clauses still exclude the answers that would make the test vacuous: delay
    # 0 (no servo at all) and slew 3.0, 8.0 or None (no slew limit).
    delay_set, slew_set = determined_sets(scores, best, grid, band)
    true_ticks = round(TRUE["delay_ms"] / 20)
    slews = _slews(grid)
    si = slews.index(TRUE["slew_rad_s"])
    near_slew = set(slews[max(0, si - 1):si + 2])
    delay_ok = (true_ticks in delay_set
                and set(delay_set) <= {true_ticks - 1, true_ticks, true_ticks + 1})
    slew_ok = bool(slew_set) and set(slew_set) <= near_slew
    print(f"\ndetermined sets at the split-half band {band:.5f} (argmin was delay "
          f"{best['delay_ticks']}, slew {best['slew_rad_s']}):")
    print(f"  delay {delay_set} ticks -- injected {true_ticks} "
          f"{'inside the set' if true_ticks in delay_set else 'OUTSIDE the set'}, "
          f"set within +-1 grid step: {set(delay_set) <= {true_ticks - 1, true_ticks, true_ticks + 1}}")
    print(f"  slew  {slew_set} rad/s -- injected {TRUE['slew_rad_s']} "
          f"{'inside the set' if TRUE['slew_rad_s'] in slew_set else 'outside the set'}, "
          f"every member within one grid step: {slew_ok}")
    assert delay_ok, f"injected delay {true_ticks} ticks not determined: set {delay_set}"
    assert slew_ok, f"injected slew {TRUE['slew_rad_s']} rad/s not determined: set {slew_set}"


def test_per_side_symmetric_fixture(model, parsed, grid, shared_fit):
    # Per-side identification must not invent an asymmetry the fixture does not have.
    # One servo drives both horns here, so a per-side search that lands on two different
    # answers is reading noise, and the coordinate descent must come back to the shared
    # solution's neighbourhood rather than away from it.
    #
    # This bound is ASSERTED, not printed. It used to be folded into the PASS/FAIL string
    # and then dropped, so the process exited 0 on a failure and the line above it was the
    # only trace. It also constrained the two delays only, while the thing the report
    # publishes as the asymmetry is the SLEW; and its second clause,
    # best_err <= scores[0][0] + 1e-9, was true by construction -- the descent starts AT
    # the shared solution and only accepts strict improvements, so it could never fail.
    # That clause is replaced by one that can: on a symmetric fixture the per-side fit
    # must not SEPARATE from the shared fit, i.e. its gain must not exceed the log's own
    # confidence band. That is the repo's own criterion for "this log can see it", pointed
    # at the claim per-side actually makes.
    O, A, O2, D, _, _ = parsed
    half, scores, best, band = shared_fit["half"], shared_fit["scores"], shared_fit["best"], shared_fit["band"]
    fit = slice(0, half)
    true_ticks = round(TRUE["delay_ms"] / 20)
    slews = _slews(grid)
    si = slews.index(TRUE["slew_rad_s"])
    near_slew = set(slews[max(0, si - 1):si + 2])
    kw_l, kw_r, ps_info = identify_per_side(model, O[fit], A[fit], O2[fit], D[fit], grid, best)
    ps_delays = {kw_l["delay_ticks"], kw_r["delay_ticks"]}
    ps_slews = {kw_l["slew_rad_s"], kw_r["slew_rad_s"]}
    ps_delay_ok = ps_delays <= {true_ticks - 1, true_ticks, true_ticks + 1}
    ps_slew_ok = ps_slews <= near_slew
    ps_gain = scores[0][0] - ps_info["best_err"]
    ps_gain_ok = ps_gain <= band
    print(f"  per side: L(delay {kw_l['delay_ticks']}, slew {kw_l['slew_rad_s']})  "
          f"R(delay {kw_r['delay_ticks']}, slew {kw_r['slew_rad_s']})  "
          f"in {ps_info['evaluations']} evaluations; err {ps_info['best_err']:.5f} vs shared "
          f"{scores[0][0]:.5f}")
    print(f"    both delays within one grid step of the single injected servo: {ps_delay_ok}; "
          f"both slews: {ps_slew_ok}")
    print(f"    per-side fit gain over shared {ps_gain:.5f} vs band {band:.5f} -- "
          f"{'not separated, as a symmetric fixture requires' if ps_gain_ok else 'SEPARATED: an asymmetry was invented'}")
    assert ps_delay_ok, (f"per-side delays {sorted(ps_delays)} left the injected "
                         f"{true_ticks} +-1 grid step")
    assert ps_slew_ok, (f"per-side slews {sorted(ps_slews, key=lambda v: (v is None, v))} "
                        f"left one grid step of the injected {TRUE['slew_rad_s']}: "
                        f"an asymmetry the fixture does not have")
    assert ps_gain_ok, (f"per-side fit beats the shared fit by {ps_gain:.5f} > band {band:.5f} "
                        f"on a fixture with ONE servo driving both horns: invented asymmetry")


def test_per_side_attribution(model, grid):
    # --- per-side ATTRIBUTION: which horn, not just how much -----------------------
    # The check above is symmetric, so it passes unchanged if the left and right labels
    # are swapped -- and they were: realized_per_side put the left triple on action
    # column 0, which imulog.parse and the twin both define as the RIGHT leg, so every
    # published per-side attribution was inverted while every error number stayed
    # identical. Only an ASYMMETRIC fixture can catch that.
    #
    # And only one anchored on GROUND TRUTH. An asymmetric fixture injected through
    # servo_id.RIGHT_COL / LEFT_COL and then labelled by the identification -- which
    # reads the same two constants -- tests self-consistency: set them to 1, 0 and the
    # injection moves with the label, so the guard stays green while every published
    # attribution inverts. The physical convention lives in the twin's XML
    # (right_leg -> joint_1 -> actuator servo_1 = ctrl[0]) and in the policy head
    # (a = np.tanh(x[:2])  # [aRight, aLeft]), not in the constants. So both of these,
    # neither redundant with the other:
    #   (a) the constants are asserted against the XML, read independently by
    #       servo_id.sim_side_columns;
    #   (b) the crippled horn is injected BY COLUMN, on the column that XML says is the
    #       right leg -- so under a reversed constant the slow horn physically stays on
    #       the right and the identification hands it back as 'left', and side_ok fails.
    slews = _slews(grid)
    conv_ok, conv_why = check_side_convention("walk")
    cols = sim_side_columns("walk")
    ASYM = dict(n=6000, seed=5,
                fast=dict(delay_ticks=0, slew_rad_s=None, deadband=0.0),
                slow=dict(delay_ticks=5, slew_rad_s=1.5, deadband=0.0),
                slow_horn="right")
    fast_horn = "left" if ASYM["slow_horn"] == "right" else "right"
    print(f"\n  {conv_why}")
    assert conv_ok, conv_why
    print(f"generating {ASYM['n'] / 50:.0f} s asymmetric fixture (hidden: {ASYM['slow_horn']} horn "
          f"delay {ASYM['slow']['delay_ticks']} ticks / slew {ASYM['slow']['slew_rad_s']} rad/s, "
          f"the other horn ideal; injected on action column {cols[ASYM['slow_horn']]}, "
          f"which the XML says drives the {ASYM['slow_horn']} leg)...", flush=True)
    per_side_servo = PerSideServo(by_column={cols[ASYM["slow_horn"]]: ASYM["slow"],
                                            cols[fast_horn]: ASYM["fast"]})
    Oy, Ay, O2y, Dy, _ = collect(ASYM["n"], seed=ASYM["seed"], body="walk", servo=per_side_servo)
    _, best_y = identify(model, Oy, Ay, O2y, Dy, grid)
    kw_ly, kw_ry, _ = identify_per_side(model, Oy, Ay, O2y, Dy, grid, best_y)
    got_slow = slower_side(kw_ly, kw_ry)
    slow_kw = kw_ry if ASYM["slow_horn"] == "right" else kw_ly
    si_a = slews.index(ASYM["slow"]["slew_rad_s"])
    near_slow = set(slews[max(0, si_a - 1):si_a + 2])
    side_ok = got_slow == ASYM["slow_horn"]
    slow_slew_ok = slow_kw["slew_rad_s"] in near_slow
    print(f"  identified: L(delay {kw_ly['delay_ticks']}, slew {kw_ly['slew_rad_s']})  "
          f"R(delay {kw_ry['delay_ticks']}, slew {kw_ry['slew_rad_s']})")
    print(f"  slower horn identified as {got_slow!r}, injected on {ASYM['slow_horn']!r}: "
          f"{'ok' if side_ok else 'WRONG SIDE -- the left/right labels are inverted'}")
    print(f"  that horn's slew {slow_kw['slew_rad_s']} within one grid step of the injected "
          f"{ASYM['slow']['slew_rad_s']}: {slow_slew_ok}")
    assert side_ok, (f"the slow horn was injected on action column "
                     f"{cols[ASYM['slow_horn']]}, which the XML says is the "
                     f"{ASYM['slow_horn']} leg, and came back on the {got_slow}: "
                     f"per-side left/right attribution is inverted")
    assert slow_slew_ok, (f"the crippled horn's slew came back {slow_kw['slew_rad_s']}, more "
                          f"than one grid step from the injected {ASYM['slow']['slew_rad_s']}")
    print("\nROUND-TRIP PASS "
          "- delay and slew determined to one grid step through 60/30 Hz jittered sampling, "
          "the per-side search stays on the symmetric answer, and on a deliberately "
          "asymmetric fixture -- injected on the action column the twin's XML says is the "
          "right leg, not on the column servo_id's constants say it is -- the slow horn "
          "comes back on the side it was injected on")


# ----------------------------------------------------------------------------
# sensor-side round-trip: the same standard, applied to sensor_id
# ----------------------------------------------------------------------------

def test_euler_rates_to_body_kinematics():
    # (1) the euler-rate -> body-rate map, against an INDEPENDENT derivation.
    # A pure delay commutes with any static mixing of the axes, so a sign-flipped or
    # transposed kinematic map still recovers the injected lag below: that assert
    # proves the correlation works, never that the kinematics are right. Ground truth
    # here is w^ = R^T dR/dt on an analytic trajectory, which shares no line of code
    # with the formula under test.
    def _R(phi, th, psi):
        cz, sz = np.cos(psi), np.sin(psi)
        cy, sy = np.cos(th), np.sin(th)
        cx, sx = np.cos(phi), np.sin(phi)
        return (np.array([[cz, -sz, 0.], [sz, cz, 0.], [0., 0., 1.]])
                @ np.array([[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]])
                @ np.array([[1., 0., 0.], [0., cx, -sx], [0., sx, cx]]))

    def _traj(tt):                       # |pitch| to 1.0 rad, yaw winding through +-pi
        tt = np.atleast_1d(np.asarray(tt, np.float64))
        return np.stack([0.9 * np.sin(2 * np.pi * 0.31 * tt + 0.4),
                         1.0 * np.sin(2 * np.pi * 0.23 * tt),
                         np.pi * tt], 1)

    dtk = 1 / 60.0
    tt = np.arange(0.0, 20.0, dtk)
    h = 1e-6
    w_true = np.empty((len(tt), 3))
    for i, ti in enumerate(tt):
        S = _R(*_traj(ti)[0]).T @ ((_R(*_traj(ti + h)[0]) - _R(*_traj(ti - h)[0])) / (2 * h))
        w_true[i] = [S[2, 1], S[0, 2], S[1, 0]]
    ang_true = _traj(tt)
    wrapped = np.arctan2(np.sin(ang_true), np.cos(ang_true))      # the map must unwrap itself
    w_pred = euler_rates_to_body(wrapped, dtk)
    inner = slice(3, -3)                 # np.gradient is one-sided at the ends
    kerr = np.abs(w_pred[inner] - w_true[inner]).max(0)
    print(f"\neuler-rate -> body-rate map vs R^T dR/dt, analytic trajectory "
          f"({tt[-1] * 0.5:.0f} yaw turns, |pitch| <= 1.0 rad, dt {dtk * 1000:.1f} ms):")
    print("  max |err| " + "  ".join(f"{BODY_AXES[a]} {kerr[a]:.1e}" for a in range(3))
          + " rad/s   (np.gradient truncation; a sign or convention flip is O(1))")
    assert (kerr < 5e-3).all(), f"euler->body kinematics wrong on some axis: {kerr}"


def test_sensor_secrets(tmp):
    SENSOR = dict(fused_lag_ms=60.0, gyro_arw=2e-3, still_lead_s=120.0, imu_slow=(300.0, 30.0, 3.0))
    print("\ngenerating 600 s sensor fixture (hidden: fused-filter lag 60 ms, gyro ARW 2e-3 "
          "rad/s/sqrt(Hz), 30 s of 3x-slow IMU at 300 s)...", flush=True)
    path = tmp / "imulog_sensor_fixture.jsonl"
    fixture(str(path), seconds=600, seed=4, **SENSOR)
    _, (it, iv, ct, cv), ev = _read_rows(str(path))
    st = dt_stats(it)
    print(f"dt stats: median {st['median_ms']:.1f} ms  p99 {st['p99_ms']:.1f} ms  "
          f"max {st['max_ms']:.0f} ms  jitter_warn {st['jitter_warn']}")
    assert st["jitter_warn"], "dt_stats missed the injected 3x IMU slowdown"
    res = filter_lag(iv[:, :3], it, iv[:, 3:], it)
    print(f"  grid {res['grid_dt_ms']:.1f} ms, {res['segments_used']} segments used / "
          f"{res['segments_dropped']} dropped, {res['excluded_ms'] / 1000:.1f} s excluded at gaps, "
          f"gimbal mask keeps {res['gimbal_kept_frac'] * 100:.1f}% of {res['grid_samples']:,} samples")
    for r in res["axes"]:
        print(f"  filter lag {r['axis']:>3}: " + (f"{r['lag_ms']:+6.1f} ms (corr {r['corr']:.2f})"
              if r["determined"] else f"undetermined (corr {r['corr']:.2f}) -- {r['reason']}"))
    lags = {r["axis"]: r for r in res["axes"]}
    det = [r for r in res["axes"] if r["determined"]]
    assert len(det) >= 2, f"filter lag determined on only {len(det)} of 3 axes"
    # wz carries the yaw wrap: if unwrapping or the segment split is wrong, this is the
    # axis that breaks, and 'two of three determined' would have hidden it.
    assert lags["wz"]["determined"], (f"wz undetermined (corr {lags['wz']['corr']:.2f}) -- yaw is "
                                      f"the only wrap-sensitive axis and it must be recovered")
    for r in det:
        assert abs(r["lag_ms"] - SENSOR["fused_lag_ms"]) <= 10.0, \
            f"{r['axis']} lag {r['lag_ms']:.1f} ms vs injected {SENSOR['fused_lag_ms']:.0f} ms"
        assert r["corr"] >= 0.6, f"{r['axis']} reported a lag on corr {r['corr']:.2f}"
        assert not r["boundary"], f"{r['axis']} peak pegged at the search boundary"

    t0, t1 = max(still_windows(ev, it[-1]), key=lambda w: w[1] - w[0])
    sel = (it >= t0) & (it < t1)
    vs = verify_still(iv[sel, :3], iv[sel, 3:])
    fs, rate_ok, rate_why = segment_rate(it[sel])
    print(f"  still segment {(t1 - t0) / 1000:.0f} s: gyro RMS {vs['gyro_rms']:.4f} rad/s, "
          f"max per-axis roll/pitch std {vs['ang_std']:.4f} rad, still={vs['still']}; "
          f"fs {fs:.2f} Hz, one-rate check {'ok' if rate_ok else rate_why}")
    assert vs["still"], "the fixture's labelled still segment must verify as still"
    assert rate_ok, f"the still segment must support one fs: {rate_why}"
    # the gate itself, on timestamps built for it: healthy jitter must pass, and a
    # stall inside the segment must not, or Allan would integrate at the wrong rate
    rr = np.random.default_rng(1)
    clean = np.cumsum(np.full(4000, 16.7) * (1 + rr.normal(0, 0.04, 4000))) + rr.normal(0, 2.0, 4000)
    stalled = clean + np.concatenate([np.zeros(3000), np.arange(1000) * 33.4])
    _, ok_clean, _ = segment_rate(np.sort(clean))
    _, ok_stall, _ = segment_rate(np.sort(stalled))
    print(f"  one-rate gate: healthy 60 Hz jitter -> {'accepted' if ok_clean else 'REFUSED'}; "
          f"3x stall over a quarter -> {'ACCEPTED' if ok_stall else 'refused'}")
    assert ok_clean, "the one-rate gate refuses ordinary phone jitter"
    assert not ok_stall, "the one-rate gate misses a 3x stall inside the segment"
    for a, r in enumerate(allan_deviation(iv[sel, 3:], fs)):
        n_est = r["arw"]; b_est = r["bias_instability"]
        print(f"  ARW {BODY_AXES[a]}: {'undetermined' if n_est is None else f'{n_est:.2e}'} "
              f"rad/s/sqrt(Hz) (injected {SENSOR['gyro_arw']:.0e}, fitted log-log slope "
              f"{r['arw_slope']:+.3f} vs the -1/2 law it is read with); bias instability "
              + ("undetermined" if b_est is None else f"{b_est:.2e} rad/s"))
        assert n_est is not None and abs(n_est / SENSOR["gyro_arw"] - 1) <= 0.20, \
            f"axis {a} ARW {n_est} vs injected {SENSOR['gyro_arw']}"
        assert r["bias_instability"] is None, \
            f"white-only gyro reported a bias instability ({r['bias_instability']:.2e})"


def test_bias_instability_negative_control(tmp):
    # (2) negative control for bias instability: angle random walk PLUS rate random
    # walk and no flicker floor anywhere. The Allan curve has a real interior minimum
    # -- the ARW/RRW crossover -- and B = adev_min / 0.664 there would be a number for
    # a noise process the sensor does not have. It must come back undetermined while
    # the ARW, which IS in the data, is still recovered.
    RRW = dict(gyro_arw=2e-3, gyro_rrw=3.5e-4, still_lead_s=150.0)
    print("\ngenerating 160 s ARW+RRW fixture (crossover minimum, no flicker floor)...", flush=True)
    path = tmp / "imulog_rrw_fixture.jsonl"
    fixture(str(path), seconds=160, seed=5, **RRW)
    _, (it2, iv2, _, _), ev2 = _read_rows(str(path))
    t0, t1 = max(still_windows(ev2, it2[-1]), key=lambda w: w[1] - w[0])
    sel2 = (it2 >= t0) & (it2 < t1)
    fs2 = 1000.0 / float(np.median(np.diff(it2[sel2])))
    print(f"  still segment {(t1 - t0) / 1000:.0f} s, {int(sel2.sum()):,} samples at {fs2:.1f} Hz")
    for a, r in enumerate(allan_deviation(iv2[sel2, 3:], fs2)):
        n_est = r["arw"]; b_est = r["bias_instability"]
        print(f"  {BODY_AXES[a]}: ARW {'undetermined' if n_est is None else f'{n_est:.2e}'} "
              f"(injected {RRW['gyro_arw']:.0e}, fitted slope {r['arw_slope']:+.3f});  bias instability "
              + (f"undetermined -- {r['bias_reason']}" if b_est is None else f"{b_est:.2e} rad/s"))
        assert r["bias_instability"] is None, \
            f"axis {a}: an ARW/RRW crossover was converted into a bias instability"
        assert n_est is not None and abs(n_est / RRW["gyro_arw"] - 1) <= 0.20, \
            f"axis {a} ARW {n_est} vs injected {RRW['gyro_arw']} under a rate random walk"
    print("\nSENSOR ROUND-TRIP PASS - kinematics, fused-filter lag (with the yaw axis), gyro "
          "noise density, the refused bias instability and the timing stall, from the file alone")
