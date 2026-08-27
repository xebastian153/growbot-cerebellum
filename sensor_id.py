"""Characterise the SENSOR side of the sim-to-real gap from a ?imulog=1 log alone.

servo_id.py identifies the actuator between command and body (delay, slew,
deadband). This module is the symmetric counterpart on the observation side:
the phone does not report ground truth, it reports the output of a sensor-fusion
filter running on jittered timestamps. Three measurements, all from the same
session file the actuator analysis uses, no extra hardware:

  dt_stats()         clock jitter per stream -- the forward model assumes a
                     fixed tick; the timestamps say how true that is.
  filter_lag()       per-axis lag of the fused orientation behind the raw gyro,
                     by cross-correlating the fused angles (pushed through the
                     euler-rate kinematics to predicted BODY rates) against the
                     gyro. Positive = orientation lags the gyro.
  allan_deviation()  overlapping Allan deviation of the still-segment gyro:
                     angle random walk, and bias instability when the curve
                     shows a genuine flicker plateau -- noise parameters
                     measured from the phone, to inject into the twin instead
                     of invented gaussians.

Every number here is reported with the conditions that produced it, and every
condition that could fake a number is checked instead of assumed: correlations
never cross a dropout or a multi-file seam, grid samples near gimbal lock are
dropped, a peak pegged at the edge of the search window is reported as "at least
the window" rather than as a value, a segment labelled still is verified still
before Allan touches it, and Allan is refused outright when the timing inside
that segment is not uniform enough for a single fs. Undetermined is a result.

Validated like the parser: imulog.py's fixture can hide a known fused-filter
lag, a known angle-random-walk density, a rate random walk with no flicker floor
and a timing stall; the round-trip suite (tests/test_imulog_roundtrips.py) must recover the first two, refuse
the third and flag the fourth, or fail.

The estimators live in `growbot_cerebellum.sensor_id`; this file is the command.
"""
from __future__ import annotations
import argparse, json
import numpy as np

from growbot_cerebellum import provenance
from growbot_cerebellum.imulog import _read_rows, run_preflight, GAP_MS, STILL_GYRO_RMS_MAX, STILL_ANG_STD_MAX
from growbot_cerebellum.sensor_id import (BODY_AXES, GIMBAL_PITCH_RAD, MIN_REGIME_SAMPLES, ARW_TAIL_TRIM_MS,
                                          _stats_from_diffs, dt_stats, still_windows, regime_spans, verify_still, segment_rate,
                                          allan_deviation, filter_lag, default_out_path, apply_split_half)


def _fmt_lag(r, band=None, verdict=""):
    """One report line for one axis: the number with its band, or the reason it has none."""
    if not r["determined"]:
        tag = "boundary" if r.get("boundary") else "undetermined"
        gate = r.get("gate_failed")
        tag += f" [{gate} gate]" if gate else ""
        corr = "" if not np.isfinite(r["corr"]) else f" (peak corr {r['corr']:.2f})"
        return f"{tag}{corr} -- {r['reason']}"
    b = "" if band is None else f" +- {band:.1f}"
    return (f"{r['lag_ms']:+7.1f}{b} ms   peak corr {r['corr']:.2f}"
            + (f"   split-half {verdict}" if verdict else ""))


def main():
    ap = argparse.ArgumentParser(description="sensor-side characterisation of a ?imulog=1 session")
    ap.add_argument("log", nargs="+", help="session file(s); clocks are made sequential across files")
    ap.add_argument("--min-still-s", type=float, default=60.0,
                    help="still segment length below which Allan results are reported as undetermined-prone")
    ap.add_argument("--max-lag-ms", type=float, default=500.0, help="cross-correlation search half-width")
    ap.add_argument("--min-corr", type=float, default=0.5, help="peak correlation below which a lag is not reported")
    ap.add_argument("--out", default=None,
                    help="output JSON; default results/sensor_id_<input stem>.json, so two "
                         "inputs never clobber one artifact")
    args = ap.parse_args()
    out_path = args.out or default_out_path(args.log)

    SEAM_MS = 1000.0                    # artificial gap inserted between files
    header = None
    imu_t, imu_v = [], []
    stills, regimes, imu_d, cmd_d = [], {}, [], []
    offset = 0.0
    for f in args.log:
        print(f"--- {f}")
        if not run_preflight(f):
            raise SystemExit(f"preflight FAIL on {f}: fix the contract before analysing")
        hi, (it, iv, ct, cv), events = _read_rows(f)
        if header is None:
            header = hi
        else:
            for k in ("imu_units", "pose_units", "trims_in_values", "l_sign", "r_sign", "l_off", "r_off", "gain"):
                if hi.get(k) != header.get(k):
                    raise SystemExit(f"header mismatch across files on {k!r}: {header.get(k)} vs {hi.get(k)} in {f}")
        if header.get("imu_units", "rad") == "deg":
            iv = np.deg2rad(iv)
        shift = offset - it[0]
        imu_t.append(it + shift); imu_v.append(iv)
        imu_d.append(np.diff(it)); cmd_d.append(np.diff(ct))   # per-file diffs: no seam dt
        stills += [(t0 + shift, t1 + shift) for t0, t1 in still_windows(events, it[-1])]
        for name, spans in regime_spans(events, it[-1]).items():
            regimes.setdefault(name, []).extend((t0 + shift, t1 + shift) for t0, t1 in spans)
        offset = it[-1] + shift + SEAM_MS
    imu_t = np.concatenate(imu_t); imu_v = np.concatenate(imu_v)

    print(f"\nclock ({len(args.log)} file(s); dt within files only, {SEAM_MS:.0f} ms seam between files)")
    dt_by_stream = {}
    for name, d in (("imu", np.concatenate(imu_d)), ("cmd", np.concatenate(cmd_d))):
        if len(d) == 0:                       # a still capture sends nothing: no cadence to report
            dt_by_stream[name] = None
            print(f"  {name}  no rows -- nothing was sent to the servos in this session")
            continue
        st = _stats_from_diffs(d)
        dt_by_stream[name] = st
        print(f"  {name}  dt ms: median {st['median_ms']:.1f}  p95 {st['p95_ms']:.1f}  "
              f"p99 {st['p99_ms']:.1f}  max {st['max_ms']:.0f}  "
              f">20% off-median {st['frac_dev20'] * 100:.1f}%"
              + ("   JITTER above 1.5x median at p99" if st["jitter_warn"] else ""))

    # --- fused-orientation lag ------------------------------------------------
    kw = dict(max_lag_ms=args.max_lag_ms, min_corr=args.min_corr)
    res = filter_lag(imu_v[:, :3], imu_t, imu_v[:, 3:], imu_t, **kw)
    mid = 0.5 * (imu_t[0] + imu_t[-1])
    halves = []
    for sel in ((imu_t < mid), (imu_t >= mid)):
        halves.append(filter_lag(imu_v[sel, :3], imu_t[sel], imu_v[sel, 3:], imu_t[sel], **kw))
    print("\nfused-orientation lag behind the raw gyro (cross-correlation, whole session)")
    print("  axes are BODY rates wx/wy/wz -- the fused ZYX angles through the euler-rate")
    print("  kinematics, NOT gap_report's roll/pitch/yaw angles (wx = roll rate only upright)")
    print(f"  grid {res['grid_dt_ms']:.1f} ms | {res['segments_used']} segments used, "
          f"{res['segments_dropped']} dropped | {res['excluded_ms'] / 1000:.1f} s excluded at "
          f"gaps/seams, {res['warmup_excluded_ms'] / 1000:.1f} s to post-gap filter warm-up")
    print(f"  gimbal mask |pitch|>{res['gimbal_pitch_rad']:.1f} rad keeps "
          f"{res['gimbal_kept_frac'] * 100:.1f}% of {res['grid_samples']:,} grid samples; "
          f"{res['usable_samples']:,} ({res['usable_frac'] * 100:.1f}%) enter the correlation")
    print(f"  search +-{res['max_lag_ms']:.0f} ms, min corr {res['min_corr']:.2f}; band from the "
          f"split halves (|A - B| / 2), as in servo_id.py")
    split = []
    for a, r in enumerate(res["axes"]):
        ra, rb = halves[0]["axes"][a], halves[1]["axes"][a]
        band = verdict = None
        if r["determined"] and ra["determined"] and rb["determined"]:
            band = abs(ra["lag_ms"] - rb["lag_ms"]) / 2.0
            verdict = "AGREE" if 2 * band <= res["grid_dt_ms"] else \
                      "DISAGREE -- halves differ by more than one grid sample"
        elif r["determined"]:
            verdict = "DISAGREE -- undetermined on " + \
                      ("the first half" if not ra["determined"] else "the second half")
        split.append({"axis": r["axis"], "half_a": ra["lag_ms"] if ra["determined"] else None,
                      "half_b": rb["lag_ms"] if rb["determined"] else None,
                      "band_ms": band, "verdict": verdict})
    # both gates before any number is printed: correlation AND split-half stability
    apply_split_half(res, split)
    for r, s in zip(res["axes"], split):
        print(f"  {r['axis']:>5}  " + _fmt_lag(r, s["band_ms"], s["verdict"] or ""))

    # --- lag per regime: an aggregate that hides a regime difference is not a lag
    per_regime = {}
    big = {n: s for n, s in regimes.items()
           if int(sum(((imu_t >= t0) & (imu_t < t1)).sum() for t0, t1 in s)) >= MIN_REGIME_SAMPLES}
    if big:
        print(f"\n  per regime (>= {MIN_REGIME_SAMPLES} IMU rows); lag ms (peak corr), "
              f"'--' = undetermined")
        print(f"    {'regime':<12}{'rows':>8}   " + "".join(f"{a:>16}" for a in BODY_AXES))
        for name in sorted(big):
            m = np.zeros(len(imu_t), bool)
            for t0, t1 in big[name]:
                m |= (imu_t >= t0) & (imu_t < t1)
            rr = filter_lag(imu_v[m, :3], imu_t[m], imu_v[m, 3:], imu_t[m], **kw)
            per_regime[name] = {"rows": int(m.sum()), **rr}
            cells = "".join(
                (f"{x['lag_ms']:+9.1f} ({x['corr']:.2f})" if x["determined"]
                 else f"{'--':>9} ({x['corr']:.2f})" if np.isfinite(x["corr"]) else f"{'--':>16}")
                for x in rr["axes"])
            print(f"    {name:<12}{int(m.sum()):>8}   {cells}")

    # --- Allan deviation on a verified-still segment ---------------------------
    allan = None
    still_meta = None
    tail = []
    if stills:
        # Allan integrates the rate into an angle assuming a single uniform sample
        # period, so a dropout inside the segment is not a small blemish: the missing
        # samples silently vanish from the time axis and the integrated angle acquires
        # a step the sensor never produced, biasing every tau above the gap. Split the
        # candidate windows at dropouts first and take the longest CONTIGUOUS run.
        contiguous = []
        for w0, w1 in stills:
            m = np.flatnonzero((imu_t >= w0) & (imu_t < w1))
            if len(m) < 2:
                continue
            cuts = np.flatnonzero(np.diff(imu_t[m]) > GAP_MS)
            for a, b in zip(np.r_[0, cuts + 1], np.r_[cuts + 1, len(m)]):
                if b - a >= 2:
                    contiguous.append((float(imu_t[m[a]]), float(imu_t[m[b - 1]])))
        if not contiguous:
            contiguous = list(stills)
        t0, t1 = max(contiguous, key=lambda w: w[1] - w[0])
        whole = max(stills, key=lambda w: w[1] - w[0])
        if (whole[1] - whole[0]) - (t1 - t0) > GAP_MS:
            print(f"\n  the longest still window spans {(whole[1] - whole[0]) / 1000:.0f} s but "
                  f"contains a dropout: Allan uses its longest gap-free run "
                  f"({(t1 - t0) / 1000:.0f} s) instead, because the estimator assumes one "
                  f"uniform sample period and missing samples would bias the long taus")
        sel = (imu_t >= t0) & (imu_t <= t1)
        span = (t1 - t0) / 1000.0
        seg_dt = dt_stats(imu_t[sel])
        vs = verify_still(imu_v[sel, :3], imu_v[sel, 3:])
        print(f"\nAllan deviation on the longest still/idle segment "
              f"({span:.0f} s, {int(sel.sum()):,} samples, t = {t0 / 1000:.1f}..{t1 / 1000:.1f} s)")
        print(f"  segment dt ms: median {seg_dt['median_ms']:.1f}  p95 {seg_dt['p95_ms']:.1f}  "
              f"p99 {seg_dt['p99_ms']:.1f}  max {seg_dt['max_ms']:.0f}"
              + ("  (p99 above 1.5x median: reported, not a gate)" if seg_dt["jitter_warn"] else "")
              + f"; stillness check: gyro RMS {vs['gyro_rms']:.3f} rad/s (max {vs['gyro_rms_max']}), "
              f"max per-axis roll/pitch std {vs['ang_std']:.4f} rad (max {vs['ang_std_max']})")
        reasons = []
        if span < args.min_still_s:
            print(f"  segment shorter than {args.min_still_s:.0f} s -- expect undetermined "
                  f"parameters; reporting only what the curve supports")
        fs, rate_ok, rate_why = segment_rate(imu_t[sel])
        if not rate_ok:
            reasons.append(rate_why)
        if not vs["still"]:
            reasons.append(f"the segment labelled still is not still: gyro RMS {vs['gyro_rms']:.3f} "
                           f"rad/s, max per-axis roll/pitch std {vs['ang_std']:.4f} rad")
        # The peak body rate inside the analysed segment, and the session's own peak with
        # its timestamp. A statement about "the peak is a tap, not a disturbance" is a
        # claim about WHICH samples entered the fit, so both numbers are recorded rather
        # than eyeballed off a plot.
        seg_w = np.abs(imu_v[sel, 3:])
        all_w = np.abs(imu_v[:, 3:]).max(1)
        i_peak = int(np.argmax(all_w))
        still_meta = {"t0_ms": float(t0), "t1_ms": float(t1), "span_s": float(span),
                      "samples": int(sel.sum()), "fs_hz": float(fs), "rate_ok": bool(rate_ok),
                      "dt": seg_dt, "stillness": vs,
                      "gyro_abs_max_rad_s": float(seg_w.max()),
                      "gyro_abs_max_t_ms": float(imu_t[sel][int(np.argmax(seg_w.max(1)))]),
                      "session_gyro_abs_max_rad_s": float(all_w[i_peak]),
                      "session_gyro_abs_max_t_ms": float(imu_t[i_peak]),
                      "session_gyro_abs_max_in_segment": bool(t0 <= imu_t[i_peak] <= t1),
                      "undetermined_reasons": reasons}
        print(f"  peak |body rate| inside the segment {np.rad2deg(still_meta['gyro_abs_max_rad_s']):.2f} deg/s; "
              f"session peak {np.rad2deg(still_meta['session_gyro_abs_max_rad_s']):.2f} deg/s at "
              f"t = {still_meta['session_gyro_abs_max_t_ms'] / 1000:.2f} s, "
              f"{'inside' if still_meta['session_gyro_abs_max_in_segment'] else 'outside'} the analysed segment")
        if np.isfinite(fs):
            allan = allan_deviation(imu_v[sel, 3:], fs)
            print(f"  fs {fs:.2f} Hz from the segment's median dt")
            for a, r in enumerate(allan):
                if reasons:
                    print(f"  {BODY_AXES[a]:>5}  ARW and bias instability undetermined -- {reasons[0]}")
                    continue
                sl = ("slope --" if r["arw_slope"] is None
                      else f"fitted log-log slope {r['arw_slope']:+.3f} over tau "
                           f"{r['arw_slope_window_s'][0]}-{r['arw_slope_window_s'][1]} s, "
                           f"accepted in [{r['arw_slope_accept'][0]}, {r['arw_slope_accept'][1]}]")
                arw = (f"ARW {r['arw']:.2e} rad/s/sqrt(Hz) ({sl})" if r["arw"] is not None
                       else f"ARW undetermined (no -1/2 slope around tau = 1 s: {sl})")
                bi = (f"bias instability {r['bias_instability']:.2e} rad/s"
                      if r["bias_instability"] is not None
                      else f"bias instability undetermined ({r['bias_reason']})")
                print(f"  {BODY_AXES[a]:>5}  {arw};  {bi}")
            # Tail sensitivity. The segment ends where the record does, and a session
            # that ends on a tap puts its largest body rate in the last samples of the
            # fit. Trimming is not free -- a hand-chosen cut is a free parameter, so the
            # published value stays the one the segment rule produces -- but an ARW that
            # only survives with those samples in it is a different claim from one that
            # does not move. The trims are stated, not searched.
            for trim in ([] if reasons else ARW_TAIL_TRIM_MS):
                s2 = (imu_t >= t0) & (imu_t <= t1 - trim)
                if int(s2.sum()) < 64:
                    continue
                fs2, ok2, _ = segment_rate(imu_t[s2])
                if not np.isfinite(fs2):
                    continue
                r2 = allan_deviation(imu_v[s2, 3:], fs2)
                tail.append({"trim_ms": float(trim), "samples": int(s2.sum()),
                             "fs_hz": float(fs2), "rate_ok": bool(ok2),
                             "peak_gyro_rad_s": float(np.abs(imu_v[s2, 3:]).max()),
                             "axes": [{"axis": BODY_AXES[a], "arw": x["arw"],
                                       "arw_slope": x["arw_slope"]} for a, x in enumerate(r2)]})
            for tr in tail:
                cells = "  ".join(
                    f"{c['axis']} "
                    + ("undetermined" if c["arw"] is None else f"{c['arw']:.2e}")
                    + (" (slope --)" if c["arw_slope"] is None
                       else f" (slope {c['arw_slope']:+.3f})") for c in tr["axes"])
                print(f"  last {tr['trim_ms'] / 1000:.1f} s trimmed ({tr['samples']:,} samples, peak "
                      f"{np.rad2deg(tr['peak_gyro_rad_s']):.2f} deg/s):  {cells}")
        else:
            print(f"  Allan skipped -- {reasons[0]}")
    else:
        print("\nno still/idle segment labelled in the log -- Allan analysis skipped; "
              "record one still segment per session to characterise the gyro noise "
              "(a '_stop' event does not count: dead time is not verified stillness)")

    nulled = bool(still_meta and still_meta["undetermined_reasons"])
    out = {"files": args.log,
           "conditions": {"n_files": len(args.log), "seam_ms": SEAM_MS, "gap_ms": GAP_MS,
                          "max_lag_ms": args.max_lag_ms, "min_corr": args.min_corr,
                          "min_still_s": args.min_still_s,
                          "gimbal_pitch_rad": GIMBAL_PITCH_RAD,
                          "still_gyro_rms_max": STILL_GYRO_RMS_MAX,
                          "still_ang_std_max": STILL_ANG_STD_MAX,
                          "imu_rows": int(len(imu_t)),
                          "session_span_s": float((imu_t[-1] - imu_t[0]) / 1000.0)},
           "dt": dt_by_stream,
           "filter_lag": {**res, "split_half": split},
           "filter_lag_per_regime": per_regime,
           "still_segment": still_meta,
           "allan": None if allan is None else [
               {"axis": BODY_AXES[a],
                "arw": None if nulled else r["arw"],
                "arw_slope": r["arw_slope"],
                "arw_slope_window_s": list(r["arw_slope_window_s"]),
                "arw_slope_accept": list(r["arw_slope_accept"]),
                "bias_instability": None if nulled else r["bias_instability"],
                "bias_reason": still_meta["undetermined_reasons"][0] if nulled else r["bias_reason"],
                "tau_s": r["tau"].tolist(), "adev": r["adev"].tolist()}
               for a, r in enumerate(allan)],
           "allan_tail_trim": tail}
    out["provenance"] = provenance(seeds=None)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
