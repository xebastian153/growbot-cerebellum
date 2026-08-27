"""Does the identification actually get better? Four changes, ablated on the real log.

servo_id.py identifies the servo from commands + IMU through the frozen forward model.
Three findings from the literature and one from the code say its scoring is weaker than
it needs to be, and this script measures whether fixing them moves anything on the one
real file that walks:

  aligned    the phone's fused orientation trails its own gyro by about 13 ms, measured
             independently by sensor_id.filter_lag. The twin emits both channels from the
             same instant, so a real observation vector is internally inconsistent in a way
             no training vector is. imulog.parse(ang_lead_ms=...) advances the angle
             channels to meet the gyro. Observation delay and command delay are formally
             interchangeable, so whatever this removes was previously being charged to the
             servo -- the un-aligned delay is a lump of both.

  multi      the score is a one-step forward error. A delay is exactly the kind of defect
             a one-step score barely sees, because it is given the truth again at every
             tick. Clip rollouts with uniformly sampled horizons let the error accumulate.

  per-side   the grid fits ONE (delay, slew, deadband) for both servos. Nothing says the
             two are the same part, and a shared fit can only average them.

There is no ground truth on a real log, so nothing here is scored against a known answer.
The fit uses the first half, every number is read off the held-out second half, and the
question asked of each variant is the same one servo_id already asks: does the identified
servo predict the held-out data better than the raw commands do, and does the log separate
the parameters well enough for the answer to mean anything.
"""
from __future__ import annotations
import argparse, json, sys, time
import numpy as np
from growbot_cerebellum.imulog import parse, CTRL_HZ
from growbot_cerebellum.forward import MLP, make_windows, K, AXES
from growbot_cerebellum.gap import evaluate_axes
from growbot_cerebellum.servo_id import identify, identify_per_side, realized_from_commands, realized_per_side, confidence_band, determined_sets, default_grid, argmin_interior, slower_side
from growbot_cerebellum.tee import Tee

HORIZONS = [5, 25]                      # 100 ms and 500 ms, as everywhere else
CLIPS = {"min_ticks": 2, "max_ticks": 40, "n_starts": 400}   # 40 ms .. 800 ms clips



def measured_lag(path):
    """Per-axis fused-orientation lag in ms, in (roll, pitch, yaw) order.

    sensor_id reports lags on BODY axes wx/wy/wz -- the fused angles pushed through the
    euler-rate kinematics. Applying them to the angle channels uses wx->roll, wy->pitch,
    wz->yaw, which is exact only while the robot is upright. On this log the three lags
    differ by about 1 ms, far below the 20 ms grid, so the mapping choice cannot matter
    at the resolution the identification works at. Undetermined axes fall back to the
    mean of the determined ones; if none is determined there is nothing to align with.
    """
    d = json.load(open(path))
    ax = d["filter_lag"]["axes"]
    ok = [r["lag_ms"] for r in ax if r["determined"]]
    if not ok:
        return None, {"determined": 0}
    fill = float(np.mean(ok))
    lag = [float(r["lag_ms"]) if r["determined"] else fill for r in ax]
    return lag, {"determined": len(ok), "per_axis_ms": lag,
                 "axes_reported": [r["axis"] for r in ax],
                 "spread_ms": float(max(ok) - min(ok)),
                 "split_half": d["filter_lag"].get("split_half")}


def held_out_gain(model, O, A, R, D, mode, held):
    """within-0.2 rad on the held-out half, commands vs identified servo, per axis."""
    raw = evaluate_axes(model, O[held], A[held], D[held], mode[held], HORIZONS)
    cor = evaluate_axes(model, O[held], R[held], D[held], mode[held], HORIZONS)
    out = {}
    for reg in raw:
        if reg not in cor:
            continue
        for h in HORIZONS:
            for ax in AXES:
                out.setdefault(reg, {}).setdefault(str(h * 20), {})[ax] = {
                    "commands": raw[reg][h][ax]["within"],
                    "servo": cor[reg][h][ax]["within"],
                    "gain_pts": 100 * (cor[reg][h][ax]["within"] - raw[reg][h][ax]["within"])}
    return out


def run_variant(name, model, O, A, O2, D, mode, grid, clips, per_side, notes):
    """One row of the ablation: identify on the first half, read the second half."""
    t0 = time.time()
    half = len(O) // 2
    fit = slice(0, half)
    held = slice(half, None)
    scores, best = identify(model, O[fit], A[fit], O2[fit], D[fit], grid, clips=clips)
    hA, hB = slice(0, half // 2), slice(half // 2, half)
    sA, bA = identify(model, O[hA], A[hA], O2[hA], D[hA], grid, clips=clips)
    sB, bB = identify(model, O[hB], A[hB], O2[hB], D[hB], grid, clips=clips)
    band = confidence_band(sA, sB)
    dset, sset = determined_sets(scores, best, grid, band)
    agree = (bA["delay_ticks"] == bB["delay_ticks"] and bA["slew_rad_s"] == bB["slew_rad_s"])
    interior, interior_why = argmin_interior(best, grid)

    row = {"variant": name, "notes": notes, "fit_ticks": int(half),
           "clips": clips, "per_side": bool(per_side),
           "argmin": {**best, "deadband": float(best["deadband"])},
           "argmin_interior": interior, "band": float(band),
           "split_half": {"A": {"delay": bA["delay_ticks"], "slew": bA["slew_rad_s"]},
                          "B": {"delay": bB["delay_ticks"], "slew": bB["slew_rad_s"]},
                          "agree": bool(agree)},
           "delay_determined_set": [int(v) for v in dset],
           "slew_determined_set": [None if v is None else float(v) for v in sset]}

    if per_side:
        kw_l, kw_r, info = identify_per_side(model, O[fit], A[fit], O2[fit], D[fit],
                                             grid, best, clips=clips)
        R = realized_per_side(A, D, kw_l, kw_r)
        # The per-side solution needs its OWN interiority verdict. `interior` above is
        # about the shared argmin and says nothing about where the two per-side argmins
        # landed; servo_id's rule -- "a boundary argmin is the search running out, not
        # an identification" -- applies to every argmin this script publishes.
        li, lwhy = argmin_interior(kw_l, grid)
        ri, rwhy = argmin_interior(kw_r, grid)
        # Does the per-side fit separate itself from the shared fit, by the same band
        # every other claim here is cut with? A fit improvement no larger than the
        # band is not an improvement this log can see.
        fit_gain = float(scores[0][0] - info["best_err"])
        # One verdict, computed once, with MARGINAL taking precedence -- because the
        # printed prose branched on marginal first while the JSON published a bare
        # `separated: true` beside `marginal: true` at a ratio of 1.008. A consumer
        # reading the artifact got exactly the boolean the prose had retracted. So
        # `separated` now means "clear of the band", not "one part in a thousand above
        # it", and `verdict` is the single field to read.
        #
        # `band` can be exactly 0.0: confidence_band is a MAD, so it collapses whenever
        # more than half the hypotheses tie across the two halves (a segment whose
        # commands never leave neutral does this). Then there is no noise scale, `ratio`
        # is null, and neither boolean can be true. That is a reported condition -- it
        # used to be a TypeError inside the `:.2f` below, which aborted the run before
        # json.dump and lost every variant's output rather than degrading one field.
        marginal = bool(band > 0 and 0.9 * band <= fit_gain <= 1.1 * band)
        separated = bool(band > 0 and fit_gain > band and not marginal)
        verdict = ("band_zero" if not band > 0 else
                   "marginal" if marginal else
                   "separated" if separated else "not_separated")
        row["per_side_solution"] = {
            "left": {**kw_l, "deadband": float(kw_l["deadband"])},
            "right": {**kw_r, "deadband": float(kw_r["deadband"])},
            "evaluations": info["evaluations"], "best_err": info["best_err"],
            "shared_err": scores[0][0],
            "left_argmin_interior": bool(li), "right_argmin_interior": bool(ri),
            "argmin_interior": bool(li and ri),
            "interior_why": {"left": lwhy, "right": rwhy},
            "slower_side": slower_side(kw_l, kw_r),
            "fit_gain_over_shared": fit_gain,
            "fit_gain_vs_band": {
                "gain": fit_gain, "band": float(band),
                "ratio": float(fit_gain / band) if band > 0 else None,
                # A bare "separated" boolean at gain/band = 1.01 reads like a result and
                # is a coin flip. Anything inside 10% of the band is reported as MARGINAL,
                # because the band is itself a noise estimate from two halves and is not
                # known to that precision -- and MARGINAL excludes SEPARATED, in the
                # artifact as well as in the prose.
                "separated": separated, "marginal": marginal,
                "band_zero": bool(not band > 0),
                "verdict": verdict}}
        # The per-side fit's own split-half stability. The `split_half` above is the
        # SHARED fit's; the claim being published ("which horn is slower") is a per-side
        # claim, so it needs a per-side test. Each half is re-fitted from its own shared
        # argmin, exactly as the full fit is.
        lA, rA, _ = identify_per_side(model, O[hA], A[hA], O2[hA], D[hA], grid, bA, clips=clips)
        lB, rB, _ = identify_per_side(model, O[hB], A[hB], O2[hB], D[hB], grid, bB, clips=clips)
        slowA, slowB = slower_side(lA, rA), slower_side(lB, rB)
        def _pair(l, r):
            return {"left": {"delay": l["delay_ticks"], "slew": l["slew_rad_s"]},
                    "right": {"delay": r["delay_ticks"], "slew": r["slew_rad_s"]}}
        # slower_agree is the only surviving support for "the right horn is slower", so it
        # is published WITH its noise floor rather than as a bare true. It is two halves
        # agreeing on one of three outcomes {left, right, neither}: under a null with no
        # real asymmetry, and ties rare on a 252-point grid, the two halves land on the
        # same non-'neither' side about one time in two. That is a coin flip -- the exact
        # standard applied a few lines above to reject a gain/band ratio of 1.01 -- so the
        # flag is worth about one bit and cannot be quoted as a confirmed asymmetry.
        agree_note = ("two halves, each landing on one of {left, right, neither}: under a "
                      "no-asymmetry null they agree on the same side roughly 1 time in 2, "
                      "so this flag is about one bit of evidence, not a confirmed asymmetry")
        row["per_side_solution"]["split_half"] = {
            "A": {**_pair(lA, rA), "slower": slowA},
            "B": {**_pair(lB, rB), "slower": slowB},
            "slower_agree": bool(slowA == slowB and slowA != "neither"),
            "slower_agree_null_p": 0.5,
            "slower_agree_note": agree_note,
            "argmin_agree": bool(_pair(lA, rA) == _pair(lB, rB))}
        # Each side's CONDITIONAL slice, with the OTHER side held at its solution: on
        # a per-side search a side's separability is conditional on its partner. These
        # are one-dimensional slices through a joint surface, each centred on its own
        # argmin and cut with the band from the SHARED sweeps -- not joint determined
        # sets; see the "what the per-side split is and is not evidence for" block below
        # for what their disjointness is worth.
        for key in ("left", "right"):
            ss = info["side_scores"][key]
            ds, sset_side = determined_sets(ss, ss[0][1], grid, band)
            row["per_side_solution"][key + "_delay_conditional"] = [int(v) for v in ds]
            row["per_side_solution"][key + "_slew_conditional"] = [
                None if v is None else float(v) for v in sset_side]
    else:
        R = realized_from_commands(A, D, best)

    row["held_out"] = held_out_gain(model, O, A, R, D, mode, held)
    row["seconds"] = round(time.time() - t0, 1)
    return row


def fmt_row(r):
    a = r["argmin"]
    who = f"delay {a['delay_ticks']}, slew {a['slew_rad_s']}, db {np.rad2deg(a['deadband']):.0f} deg"
    if not r["argmin_interior"]:
        who += " [boundary]"
    if r["per_side"]:
        ps = r["per_side_solution"]
        l, rr = ps["left"], ps["right"]
        # '!' marks an argmin sitting on the grid's edge -- the search ran out there
        who = (f"L(delay {l['delay_ticks']}, slew {l['slew_rad_s']})"
               f"{'' if ps['left_argmin_interior'] else '!'}  "
               f"R(delay {rr['delay_ticks']}, slew {rr['slew_rad_s']})"
               f"{'' if ps['right_argmin_interior'] else '!'}")
    w = r["held_out"].get("walking", r["held_out"].get("all", {}))
    g500 = w.get("500", {})
    gains = "  ".join(f"{ax} {g500[ax]['gain_pts']:+5.1f}" for ax in AXES if ax in g500)
    return (f"  {r['variant']:<22}{who:<46}"
            f"delay set {str(r['delay_determined_set']):<17}"
            f"{'AGREE' if r['split_half']['agree'] else 'DISAGREE':<9}{gains}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="the growbot-imulog-1 file with a walking segment")
    ap.add_argument("--sensor-json", default=None,
                    help="sensor_id output for this log; default results/sensor_id_<stem>.json")
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()
    sys.stdout = Tee("results/logs/identification_ablation.txt")

    from sensor_id import default_out_path
    sensor_json = args.sensor_json or default_out_path([args.log])
    lag, lag_meta = measured_lag(sensor_json)

    print("== identification ablation on one real file")
    print(f"  log {args.log}")
    print(f"  sensor lag from {sensor_json}: "
          + ("none determined -- alignment variants skipped" if lag is None else
             f"{[round(v, 1) for v in lag]} ms on {lag_meta['axes_reported']} "
             f"(spread {lag_meta['spread_ms']:.1f} ms, {lag_meta['determined']}/3 determined)"))
    print("  applied as (roll, pitch, yaw); exact only upright, and the spread is far below")
    print("  the 20 ms grid, so the mapping cannot change an answer at this resolution")

    tr = np.load("data/train.npz")
    Xtr, Ytr, *_ = make_windows(tr["obs"], tr["act"], tr["next_obs"], tr["done"], K)
    model = MLP(hidden=128, epochs=args.epochs).fit(Xtr, Ytr)
    grid = default_grid()

    O, A, O2, D, header, mode = parse(args.log)
    print(f"\n  {len(O)} ticks ({len(O) / CTRL_HZ:.1f} s), regimes "
          f"{ {m: int((mode == m).sum()) for m in sorted(set(mode))} }")
    print(f"  {len(grid)} hypotheses; fit = first half, every number below = held-out half")
    print(f"  clips for the multi-horizon score: {CLIPS['min_ticks']}-{CLIPS['max_ticks']} ticks "
          f"({CLIPS['min_ticks'] * 20}-{CLIPS['max_ticks'] * 20} ms), {CLIPS['n_starts']} starts")

    aligned = None
    if lag is not None:
        aligned = parse(args.log, ang_lead_ms=lag)
        print(f"  alignment moves the angle channels by at most "
              f"{np.abs(np.arctan2(np.sin(aligned[0][:, :3] - O[:, :3]), np.cos(aligned[0][:, :3] - O[:, :3]))).max():.3f} rad "
              f"and adds {int(aligned[3].sum() - D.sum())} done ticks")

    variants = []
    plan = [("baseline", O, A, O2, D, mode, None, False,
             "one-step score, shared servo, raw parse -- what servo_id.py does today"),
            ("+multi-horizon", O, A, O2, D, mode, CLIPS, False,
             "clip rollouts instead of one-step"),
            ("+per-side", O, A, O2, D, mode, None, True,
             "one (delay, slew, deadband) per servo, coordinate descent from the shared fit")]
    if aligned is not None:
        Oa, Aa, O2a, Da, _, modea = aligned
        plan.insert(1, ("+aligned", Oa, Aa, O2a, Da, modea, None, False,
                        "angle channels advanced to meet the gyro"))
        plan.append(("+all three", Oa, Aa, O2a, Da, modea, CLIPS, True,
                     "aligned + multi-horizon + per-side"))

    print(f"\n  running {len(plan)} variants ...")
    for name, o, a, o2, d, m, clips, ps, notes in plan:
        r = run_variant(name, model, o, a, o2, d, m, grid, clips, ps, notes)
        variants.append(r)
        print(f"    {name:<16} {r['seconds']:6.1f}s   argmin delay {r['argmin']['delay_ticks']}, "
              f"slew {r['argmin']['slew_rad_s']}")

    print("\n== the table (held-out gains are walking-segment, 500 ms, within 0.2 rad)")
    print(f"  {'variant':<22}{'identified':<46}{'delay set':<17}{'halves':<9}"
          + "  ".join(f"{ax:>9}" for ax in AXES))
    for r in variants:
        print(fmt_row(r))
    print("  '!' on a per-side entry = that horn's argmin sits on the grid's edge")

    print("\n== what the per-side split is and is not evidence for")
    print("  Read before quoting any per-side number:")
    print("  - the two per-side sweeps are ONE-DIMENSIONAL CONDITIONAL slices. Each is")
    print("    taken with the partner frozen at its coordinate-descent optimum and each")
    print("    is centred on its own argmin, cut with the band from the SHARED sweeps.")
    print("    Their disjointness therefore restates 'the two argmins differ by more than")
    print("    the band'; it is not independent evidence for an asymmetry.")
    print("  - the per-side fit is only separated from the shared fit when its fit gain")
    print("    exceeds that same band. Below the band, 'per-side fits better' is noise.")
    print("  - 'both halves agree on which horn is slower' is two halves picking the same")
    print("    one of {left, right, neither}. Under a null with no real asymmetry that")
    print("    agreement comes up about half the time, so it is one coin flip's worth of")
    print("    evidence -- the same standard that rejects a gain/band ratio of 1.01 here.")
    for r in variants:
        if not r["per_side"]:
            continue
        ps = r["per_side_solution"]
        l, rr, sh = ps["left"], ps["right"], ps["split_half"]
        fg = ps["fit_gain_vs_band"]
        print(f"\n  {r['variant']}")
        print(f"    solution         L(delay {l['delay_ticks']}, slew {l['slew_rad_s']})"
              f"  R(delay {rr['delay_ticks']}, slew {rr['slew_rad_s']})"
              f"   -> slower horn: {ps['slower_side']}")
        print(f"    argmin interior  left {ps['left_argmin_interior']}, "
              f"right {ps['right_argmin_interior']}")
        print(f"      left  {ps['interior_why']['left']}")
        print(f"      right {ps['interior_why']['right']}")
        print(f"    conditional slices  left  delay {ps['left_delay_conditional']}  "
              f"slew {ps['left_slew_conditional']}")
        print(f"                        right delay {ps['right_delay_conditional']}  "
              f"slew {ps['right_slew_conditional']}")
        # `ratio` is null when the band collapsed to zero, so it is formatted through a
        # guard rather than straight into ':.2f' -- that raised TypeError and killed the
        # run before the artifact was written.
        ratio = "n/a" if fg["ratio"] is None else f"{fg['ratio']:.2f}"
        if fg["verdict"] == "band_zero":
            sep = ("BAND ZERO -- the two fit halves ranked the hypotheses too nearly "
                   "identically for a MAD to see (more than half the differences tie), so "
                   "this log offers no noise scale to cut the gain with and no separation "
                   "verdict is possible")
        elif fg["verdict"] == "marginal":
            sep = (f"MARGINAL -- gain/band = {ratio}, i.e. the per-side fit sits "
                   f"ON the band rather than clear of it; not a separation this log can "
                   f"be said to show")
        elif fg["verdict"] == "separated":
            sep = f"SEPARATED (gain/band = {ratio})"
        else:
            sep = (f"NOT SEPARATED (gain/band = {ratio}) -- the per-side fit is "
                   f"not distinguishable from the shared one on this log")
        print(f"    fit gain over shared {fg['gain']:.4f} vs band {fg['band']:.4f}: {sep}")
        print(f"    per-side split-half  A slower={sh['A']['slower']}  "
              f"B slower={sh['B']['slower']}  "
              f"-> {'AGREE' if sh['slower_agree'] else 'DISAGREE'} on which horn is slower; "
              f"argmins {'agree' if sh['argmin_agree'] else 'disagree'}")
        print(f"      noise floor  {sh['slower_agree_note']}")

    base = variants[0]
    al = next((v for v in variants if v["variant"] == "+aligned"), None)
    print("\n== what the sensor lag was costing the servo")
    if al is None:
        print("  no determined sensor lag on this file: nothing to decompose")
        decomp = None
    else:
        d0, d1 = base["argmin"]["delay_ticks"], al["argmin"]["delay_ticks"]
        lag_ticks = float(np.mean(lag)) / (1000.0 / CTRL_HZ)
        decomp = {"lumped_delay_ticks": d0, "aligned_delay_ticks": d1,
                  "sensor_lag_ms": float(np.mean(lag)), "sensor_lag_ticks": lag_ticks,
                  "delta_ticks": d0 - d1}
        print(f"  un-aligned (lumped) delay   {d0} ticks = {d0 * 20} ms")
        print(f"  aligned (actuator + gyro)   {d1} ticks = {d1 * 20} ms")
        print(f"  measured sensor lag         {np.mean(lag):.1f} ms = {lag_ticks:.2f} ticks")
        print(f"  the grid step is 20 ms, so a {np.mean(lag):.0f} ms correction can only move the")
        print(f"  argmin by a whole tick or not at all: it moved by {d0 - d1}.")
        print("  What is left after alignment is the actuator plus whatever absolute lag the")
        print("  gyro itself carries, which this log cannot measure. The decomposition is")
        print("  therefore a bound, not a split: the actuator is at most the aligned number.")

    out = {"log": args.log, "conditions": {
               "epochs": args.epochs, "K": K, "horizons_ms": [h * 20 for h in HORIZONS],
               "grid_size": len(grid), "clips": CLIPS, "ticks": int(len(O)),
               "fit": "first half", "evaluation": "held-out second half",
               "model": "MLP h128 on data/train.npz",
               "sensor_json": sensor_json, "sensor_lag_ms": lag, "sensor_lag_meta": lag_meta},
           "variants": variants, "delay_decomposition": decomp}
    with open("results/identification_ablation.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\nwrote results/identification_ablation.json")


if __name__ == "__main__":
    main()
