"""The day-of-log deliverable: gap per regime and per axis, with the twin floor built in.

The forward model's error on a real log is NOT the sim-to-real gap: the model has a
floor on the twin's own held-out physics, and that floor differs by axis (yaw is the
hard one even in sim) and by regime. Quoting the raw real-log error would misread
floor as gap. This report therefore prints three numbers per cell:

    real        the model's error on the log
    twin        the same model, same axis, same horizon, matched regime, on twin test
    gap         real - twin  (positive = worse than the floor explains)

--servo-id adds a fourth column: the real-log error after replaying the commands
through the servo identified from the log itself (servo_id.py), i.e. how much of the
gap the actuator explains. It is REFUSED, with the reason written into the artifact,
in the two cases where that column would be a number about nothing: an identification
half with no body motion, and an argmin at the grid boundary whose determined sets are
the entire grid. In both the identification is not a measurement -- of the body in the
first case, of the score surface in the second -- and a gap* computed from it still
reads like a closable actuator gap.

Regimes come from the log's event rows. Real session names map to the twin's
excitation regimes conservatively (REGIME_MAP); unmapped names fall back to the
twin's overall row, printed as such.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from growbot_cerebellum import provenance
from growbot_cerebellum.paths import DATA, under_root
from growbot_cerebellum.forward import MLP, make_windows, K, AXES
from growbot_cerebellum.sensor_id import default_out_path
from growbot_cerebellum.imulog import (parse, run_preflight, rest_attitude, CTRL_HZ,
                                       SEG_FALL_EXCURSION_RAD, STILL_GYRO_RMS_MAX)
from growbot_cerebellum.servo_id import (identify, realized_from_commands, default_grid, argmin_interior,
                                         confidence_band, determined_sets)
from growbot_cerebellum.gap import REGIME_MAP, REST_MISMATCH_RAD, twin_regimes, evaluate_axes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="+", help="?imulog=1 session file(s); per-walk files concatenate with a cut between")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--horizons", type=int, nargs="+", default=[5, 25])
    ap.add_argument("--servo-id", action="store_true", help="add the after-identified-servo column")
    ap.add_argument("--out", default=None,
                    help="output JSON; default results/gap_report_<input stem>.json, so analysing "
                         "two files keeps both instead of overwriting one")
    args = ap.parse_args()
    out_path = under_root(args.out or default_out_path(args.log, "gap_report"))

    parts, header, first, rest0 = [], None, None, None
    for f in args.log:
        print(f"--- {f}")
        if not run_preflight(f):
            raise SystemExit(f"preflight FAIL on {f}: fix the contract before analysing")
        Oi, Ai, O2i, Di, hi, mi = parse(f)
        ti = np.arange(len(Oi)) * (1000.0 / CTRL_HZ)
        resti = rest_attitude(ti, Oi[:, :3], Oi[:, 3:])
        if header is None:
            header, first, rest0 = hi, f, resti
        else:
            # Concatenating two files asserts they are the same experiment. The trim
            # keys below were already checked; these two were not, and both of them
            # were violated by the very first pair of real logs this report was run on.
            for k in ("imu_units", "pose_units", "trims_in_values", "l_sign", "r_sign",
                      "l_off", "r_off", "gain", "gain_agent"):
                if hi.get(k) != header.get(k):
                    raise SystemExit(
                        f"header mismatch across files on {k!r}: {header.get(k)} in {first} vs "
                        f"{hi.get(k)} in {f}. These are different experiments; run them "
                        f"separately (agent gain scales the commands the body actually got, "
                        f"so pooling the two mixes two action distributions into one row).")
            # Resting attitude. Two files whose bodies rest tens of degrees apart are in
            # different mounting or placement states, and every attitude-referenced
            # number -- stance, excursion, the fall threshold, the gap per axis -- means
            # something different in each. The header cannot say so; the data can.
            if rest0 is not None and resti is not None:
                d = max(abs(np.arctan2(np.sin(resti[a] - rest0[a]),
                                       np.cos(resti[a] - rest0[a]))) for a in range(2))
                if d > REST_MISMATCH_RAD:
                    raise SystemExit(
                        f"resting attitude mismatch across files: {first} rests at "
                        f"(roll {rest0[0]:+.2f}, pitch {rest0[1]:+.2f}) rad, {f} at "
                        f"(roll {resti[0]:+.2f}, pitch {resti[1]:+.2f}) -- {np.rad2deg(d):.0f} deg "
                        f"apart, above the {np.rad2deg(REST_MISMATCH_RAD):.0f} deg limit. The "
                        f"phone is not in the same place on the two bodies; run them separately.")
            elif (rest0 is None) != (resti is None):
                raise SystemExit(
                    f"one of {first} / {f} contains no still segment at all, so their resting "
                    f"attitudes cannot be compared -- refusing to concatenate on the assumption "
                    f"that they match.")
        parts.append((Oi, Ai, O2i, Di, mi))     # parse sets D[-1]=True: automatic cut at file boundary
    O, A, O2, D, mode = (np.concatenate(x) for x in zip(*parts))
    print(f"log: {len(args.log)} file(s), {len(O):,} ticks, {int(D.sum())} cuts, "
          f"surface={header.get('surface', '?')}, "
          f"regimes={ {m: int((mode == m).sum()) for m in sorted(set(mode))} }")

    tr = np.load(DATA / "train.npz"); te = np.load(DATA / "test.npz")
    Xtr, Ytr, *_ = make_windows(tr["obs"], tr["act"], tr["next_obs"], tr["done"], K)
    model = MLP(hidden=128, epochs=args.epochs).fit(Xtr, Ytr)
    tw_mode, tw_rest = twin_regimes(te["obs"], te["mode"].astype(str))
    print(f"twin floor regimes (rest attitude roll {tw_rest[0]:+.2f}, pitch {tw_rest[1]:+.2f} rad; "
          f"'fallen' = excursion > {SEG_FALL_EXCURSION_RAD} rad): "
          f"{ {m: int((tw_mode == m).sum()) for m in sorted(set(tw_mode))} }")
    twin = evaluate_axes(model, te["obs"], te["act"], te["done"], tw_mode, args.horizons)

    corrected = None
    servo_block = None
    if args.servo_id:
        # Identification and attribution must not share ticks: the servo is fitted on
        # the first half, and EVERY log column (real, gap, gap*) is evaluated on the
        # held-out second half, so gap* cannot credit itself for what it fitted.
        half = len(O) // 2
        # A servo is only visible in how the body RESPONDS to a command. Where the body
        # does not move, every hypothesis replays to the same absent response, the argmin
        # is noise, and the gap* it produces is a number about nothing -- yet it reads
        # like a closable actuator gap. One of the real logs is exactly this: its first
        # half is a motionless phone under commands swinging tens of degrees. Refusing
        # the column, and saying why, is the honest output.
        fit_rms = float(np.sqrt((O[:half, 3:] ** 2).sum(1).mean()))
        if fit_rms <= STILL_GYRO_RMS_MAX:
            servo_block = {"identified": None, "refused": True,
                           "reason": "identification half has no body motion to identify from",
                           "fit_gyro_rms_rad_s": fit_rms,
                           "still_gyro_rms_max": STILL_GYRO_RMS_MAX,
                           "fit_ticks": half}
            print(f"servo identification REFUSED: the first half ({half:,} ticks) has body-rate "
                  f"RMS {fit_rms:.3f} rad/s, at or below the stillness threshold "
                  f"{STILL_GYRO_RMS_MAX} -- the body does not respond there, so no servo is "
                  f"identifiable and no gap* column is reported for this file")
            args.servo_id = False
    if args.servo_id:
        grid = default_grid()          # one definition, shared with servo_id and the real-log report
        scores, best = identify(model, O[:half], A[:half], O2[:half], D[:half], grid)
        interior, interior_why = argmin_interior(best, grid)
        # A boundary argmin whose determined sets span the whole grid is the search
        # running out, not an identification: every hypothesis on the grid fits within
        # the band, and the winner is the edge the enumeration stops at. Replaying THAT
        # through the commands produces a gap* column that reads like a closable actuator
        # gap and is a number about nothing -- the same failure the stillness refusal
        # above exists to prevent, arriving through the score surface instead of through
        # the body. The two conditions are required together: a boundary argmin that the
        # log still separates from its neighbours is a reported condition, not a refusal.
        q = half // 2
        sA, _ = identify(model, O[0:q], A[0:q], O2[0:q], D[0:q], grid)
        sB, _ = identify(model, O[q:half], A[q:half], O2[q:half], D[q:half], grid)
        band = confidence_band(sA, sB)
        dset, sset = determined_sets(scores, best, grid, band)
        all_delays = sorted({d for d, _, _ in grid})
        all_slews = sorted({s for _, s, _ in grid}, key=lambda v: (v is None, v))
        undetermined = (list(dset) == list(all_delays)) and (list(sset) == list(all_slews))
        print(f"identified servo: delay {best['delay_ticks']} ticks, slew {best['slew_rad_s']} rad/s "
              f"(grid points; run servo_id.py for determined-set diagnostics)")
        print(f"  {interior_why}")
        print(f"  determined at band {band:.5f}: delay {dset}, slew {sset}")
        if (not interior) and undetermined:
            servo_block = {"identified": None, "refused": True,
                           "reason": ("the argmin is at the grid boundary AND both determined sets "
                                      "are the entire grid: the log separates no hypothesis from "
                                      "any other, so the identified servo is the edge of the search, "
                                      "not a measurement"),
                           "argmin": {k: (None if v is None else float(v)) for k, v in best.items()},
                           "argmin_interior": False, "interior_why": interior_why,
                           "band": float(band),
                           "delay_determined": [int(d) for d in dset],
                           "slew_determined": [None if s is None else float(s) for s in sset],
                           "fit_gyro_rms_rad_s": fit_rms, "fit_ticks": half}
            print(f"servo identification REFUSED: {servo_block['reason']} -- no gap* column is "
                  f"reported for this file")
            args.servo_id = False
    if args.servo_id:
        R = realized_from_commands(A, D, best)          # replayed over the full log: servo state is continuous
        held = slice(half, None)
        print(f"evaluation restricted to the held-out half ({len(O) - half:,} ticks); "
              f"identification used the first half")
        real = evaluate_axes(model, O[held], A[held], D[held], mode[held], args.horizons)
        corrected = evaluate_axes(model, O[held], R[held], D[held], mode[held], args.horizons)
        servo_block = {"identified": {k: (None if v is None else float(v) if k != "delay_ticks"
                                          else int(v)) for k, v in best.items()},
                       "refused": False, "argmin_interior": bool(interior),
                       "fit_gyro_rms_rad_s": fit_rms, "fit_ticks": half}
    else:
        real = evaluate_axes(model, O, A, D, mode, args.horizons)

    hs = args.horizons
    print("\nwithin 0.2 rad; gap = real - twin floor (matched regime); "
          + ("gap* = after identified servo; " if corrected else "") + "negative gap = worse than floor")
    print(f"{'regime':<10}{'n':>6}{'axis':>7}" + "".join(
        f"{'@' + str(h * 20) + 'ms':>{22 + (7 if corrected else 0)}}" for h in hs))
    print("-" * (23 + len(hs) * (22 + (7 if corrected else 0))))
    report = {}
    for reg in real:
        tref_name = REGIME_MAP.get(reg, "all") if reg != "all" else "all"
        tref = twin.get(tref_name, twin["all"])
        # Every published regime row carries its own denominator, and the twin row it is
        # differenced against carries its. n was computed and printed but never written,
        # so a reader of the artifact could not tell a 3,454-start regime from a 187-start
        # one -- and 'gap' is a difference of two rates whose noise is set by both.
        report.setdefault(reg, {})["n"] = int(real[reg]["n"])
        report[reg]["twin_regime"] = tref_name
        report[reg]["twin_n"] = int(tref["n"])
        for ax in AXES:
            line = f"{reg if ax == 'roll' else '':<10}{real[reg]['n'] if ax == 'roll' else '':>6}{ax:>7}"
            for h in hs:
                r = real[reg][h][ax]["within"]; tw = tref[h][ax]["within"]; g = r - tw
                line += f"{r * 100:>7.1f}%{tw * 100:>6.1f}%{g * 100:>+6.1f}"
                if corrected:
                    line += f"{(corrected[reg][h][ax]['within'] - tw) * 100:>+7.1f}"
                report.setdefault(reg, {}).setdefault(str(h), {})[ax] = {
                    "real": r, "twin": tw, "gap": g,
                    **({"gap_after_servo": corrected[reg][h][ax]["within"] - tw} if corrected else {})}
            print(line)
    json.dump({"header": {k: v for k, v in header.items() if not isinstance(v, (list, dict))},
               **({"servo_id": servo_block} if servo_block else {}),
               "report": report, "provenance": provenance(seeds={"mlp": 0})}, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
