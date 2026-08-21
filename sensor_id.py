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
and a timing stall; the __main__ test there must recover the first two, refuse
the third and flag the fourth, or fail.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from imulog import GAP_MS          # the same dropout threshold parse() cuts episodes at
from imulog import STILL_GYRO_RMS_MAX, STILL_ANG_STD_MAX   # one definition, three users

BODY_AXES = ("wx", "wy", "wz")       # Body rates: what the gyro measures, and what the
                                     # fused ZYX angles predict through the kinematics.
                                     # NOT gap_report's ("roll", "pitch", "yaw"), which
                                     # are orientation ANGLES; the two coincide only
                                     # while the robot is upright.
GIMBAL_PITCH_RAD = 1.2               # |pitch| beyond this is gimbal territory (and is
                                     # also what fall_recovery calls fallen())
                                     # STILL_GYRO_RMS_MAX / STILL_ANG_STD_MAX are imported
                                     # from imulog above: one definition shared by the
                                     # preflight, this module, and the v1 segmenter.
MIN_REGIME_SAMPLES = 600             # 10 s at 60 Hz: below this a per-regime lag is noise
ALLAN_RATE_TOL = 0.05                # mean/median dt mismatch above which one fs is a fiction


def _stats_from_diffs(d):
    d = np.asarray(d, np.float64)
    med = float(np.median(d))
    return {"median_ms": med,
            "p95_ms": float(np.percentile(d, 95)),
            "p99_ms": float(np.percentile(d, 99)),
            "max_ms": float(d.max()),
            "frac_dev20": float((np.abs(d - med) > 0.2 * med).mean()),
            "jitter_warn": bool(np.percentile(d, 99) > 1.5 * med)}


def dt_stats(ts):
    """Timestamp-jitter statistics for one stream (timestamps in ms).

    jitter_warn (p99 > 1.5x median) marks timing rough enough to degrade a
    fixed-dt forward model; it is a report, never an analysis gate -- except for
    Allan variance, which assumes one sampling rate and is refused when the
    selected segment's own timing fails this check.
    """
    ts = np.asarray(ts, np.float64)
    if len(ts) < 3:
        return {"median_ms": np.nan, "p95_ms": np.nan, "p99_ms": np.nan,
                "max_ms": np.nan, "frac_dev20": np.nan, "jitter_warn": False}
    return _stats_from_diffs(np.diff(ts))


def still_windows(events, t_end):
    """(t0, t1) spans OPENED by a literal still/idle event, and by nothing else.

    Preflight's seg_mask semantics exactly: 'still_start' / 'idle_start' (or a bare
    'still' / 'idle' row) opens a span that the next event of any kind closes, and
    an 'X_stop' event closes but never opens one. That last clause is the whole
    point. Mapping every '_stop' to "idle" turns the dead time after 'walk_stop' --
    the operator picking the robot up and carrying it to the next spot -- into the
    longest "still" segment in the session, and preflight's stillness check does
    not cover those spans, so nothing downstream would have contradicted it. Allan
    variance on carried-robot data is not gyro noise.
    """
    ev = sorted(events)
    out = []
    for i, (t0, n) in enumerate(ev):
        if n.endswith("_stop"):
            continue
        if n.removesuffix("_start") in ("still", "idle"):
            t1 = ev[i + 1][0] if i + 1 < len(ev) else t_end
            if t1 > t0:
                out.append((float(t0), float(t1)))
    return out


def regime_spans(events, t_end):
    """{regime: [(t0, t1), ...]} from event rows, same semantics as still_windows.

    'X_start' opens regime X until the next event; an 'X_stop' closes X and opens
    nothing, because unlabelled dead time is not a regime.
    """
    ev = sorted(events)
    out = {}
    for i, (t0, n) in enumerate(ev):
        if n.endswith("_stop"):
            continue
        name = n.removesuffix("_start")
        t1 = ev[i + 1][0] if i + 1 < len(ev) else t_end
        if t1 > t0:
            out.setdefault(name, []).append((float(t0), float(t1)))
    return out


def verify_still(ang, gyro):
    """Is a segment labelled still actually still? Same thresholds as preflight.

    The label is a claim by whoever wrote the log, and labels are wrong in ways
    that are invisible in the parameters they poison. A regime named 'still' can
    mean only that the COMMAND was held (the twin's Excitation does exactly that:
    its 'still' mode repeats the previous action, and a pushed or falling body
    under a held command is anything but still), and a real session's still marker
    can be a phone lying on a table someone leans on. Allan variance cannot tell
    motion from noise, so the segment is measured before it is used.

    The orientation number is the LARGEST PER-AXIS std, not the std of roll and
    pitch pooled together. Pooled, the statistic is dominated by the offset between
    the two axes -- a robot resting motionless at a constant -0.60 rad pitch scores
    0.30 rad, twelve times the threshold, purely for standing on a slope. It
    measures a tilt, and a tilt is not motion.
    """
    ang = np.asarray(ang, np.float64); gyro = np.asarray(gyro, np.float64)
    if len(gyro) < 2:
        return {"still": False, "gyro_rms": float("nan"), "ang_std": float("nan"),
                "gyro_rms_max": STILL_GYRO_RMS_MAX, "ang_std_max": STILL_ANG_STD_MAX,
                "n": int(len(gyro))}
    grms = float(np.sqrt((gyro ** 2).mean()))
    astd = float(max(ang[:, 0].std(), ang[:, 1].std()))
    return {"still": bool(grms <= STILL_GYRO_RMS_MAX and astd <= STILL_ANG_STD_MAX),
            "gyro_rms": grms, "ang_std": astd,
            "gyro_rms_max": STILL_GYRO_RMS_MAX, "ang_std_max": STILL_ANG_STD_MAX,
            "n": int(len(gyro))}


def segment_rate(ts, tol=ALLAN_RATE_TOL):
    """(fs, ok, reason): can ONE sampling rate describe this segment's timestamps?

    Allan variance integrates the gyro at a fixed fs. A stall inside the segment
    means the true rate is two rates, and every tau is then wrong by the ratio,
    which lands on ARW as sqrt(rate ratio). A median dt of zero (duplicate
    timestamps) is worse than wrong: 1000/median divides by zero.

    The test is the segment's MEAN dt against its MEDIAN dt. Ordinary phone jitter
    is roughly symmetric and leaves the two within about 1%; a stall drags the mean
    while barely moving the median (the fixture's 3x stall over a quarter of a
    segment: 1.38). p99 > 1.5x median -- dt_stats's whole-session jitter flag -- is
    deliberately NOT the gate here: at 60 Hz with a couple of ms of timestamp noise
    it fires on healthy segments, and refusing a number that can be measured
    correctly is its own kind of dishonesty. It is still reported.
    """
    ts = np.asarray(ts, np.float64)
    if len(ts) < 3:
        return float("nan"), False, f"only {len(ts)} samples in the segment"
    d = np.diff(ts)
    med = float(np.median(d))
    if not np.isfinite(med) or med <= 0:
        return float("nan"), False, ("the segment's median dt is not positive (duplicate or "
                                     "non-increasing timestamps): no sampling rate follows from it")
    ratio = float(d.mean() / med)
    if abs(ratio - 1.0) > tol:
        return (1000.0 / med, False,
                f"the segment is not one sampling rate: mean dt {d.mean():.1f} ms vs median "
                f"{med:.1f} ms ({ratio:.2f}x, tolerance {1 + tol:.2f}x) -- a stall inside the "
                f"segment biases ARW by sqrt(rate ratio)")
    return 1000.0 / med, True, ""


def allan_deviation(gyro, fs, n_taus=40):
    """Overlapping Allan deviation per axis on still gyro data at rate fs (Hz).

    Returns one dict per axis: tau (s), adev (rad/s), and two extracted
    parameters with their determinability stated rather than guessed:
      arw               angle random walk N (rad/s/sqrt(Hz)); the -1/2-slope
                        law read at tau = 1 s, only if the local log-log slope
                        around 1 s is -1/2 within [-0.65, -0.35], else None.
      bias_instability  B (rad/s) = adev_min / 0.664, only if the minimum sits on
                        a flicker PLATEAU (see _bias) -- the 1/0.664 conversion is
                        a flicker-noise identity and means nothing at the bottom of
                        an ARW / rate-random-walk crossover, which is what a short
                        segment usually shows. bias_reason says why, when None.
    """
    g = np.asarray(gyro, np.float64)
    if g.ndim == 1:
        g = g[:, None]
    N = len(g)
    out = []
    if N < 32:
        return [{"tau": np.array([]), "adev": np.array([]), "arw": None,
                 "bias_instability": None, "bias_reason": "fewer than 32 samples"}
                for _ in range(g.shape[1])]
    theta = np.cumsum(g, 0) / fs
    ms = np.unique(np.round(np.logspace(0, np.log10(N // 4), n_taus)).astype(int))
    ms = ms[ms >= 1]
    tau = ms / fs
    for a in range(g.shape[1]):
        adev = np.empty(len(ms))
        th = theta[:, a]
        for j, m in enumerate(ms):
            d = th[2 * m:] - 2 * th[m:-m] + th[:-2 * m]
            adev[j] = np.sqrt((d ** 2).mean() / (2 * tau[j] ** 2))
        b, why = _bias(tau, adev)
        out.append({"tau": tau, "adev": adev, "arw": _arw(tau, adev),
                    "bias_instability": b, "bias_reason": why})
    return out


def _arw(tau, adev):
    sel = (tau >= 0.33) & (tau <= 3.0)
    if sel.sum() < 4:
        return None
    lt, ls = np.log10(tau[sel]), np.log10(adev[sel])
    slope = float(np.polyfit(lt, ls, 1)[0])
    if not (-0.65 <= slope <= -0.35):
        return None
    return float(10 ** np.mean(ls + 0.5 * lt))     # the -1/2 law through the window, at tau=1


def _bias(tau, adev, flat_frac=0.10, min_decade=5.0):
    """(B, reason): bias instability only where a flicker FLOOR actually exists.

    Returns (None, why) unless the minimum is interior AND flat over a real stretch
    of tau: the contiguous run of taus whose adev is within `flat_frac` of the
    minimum must span at least a factor `min_decade` in tau.

    The earlier gate -- |log-log slope| < 0.2 on the five points centred on the
    argmin -- could not fail. Any smooth minimum has zero slope at its own minimum,
    so the test asked whether a minimum is a minimum. An ARW / rate-random-walk
    crossover (slope -1/2 meeting slope +1/2, no flicker anywhere in the sensor)
    passed it and was converted by /0.664, an identity that holds only for flicker
    noise, into a bias instability the gyro does not have. A crossover is a narrow
    V: within 10% of its minimum it spans about 3.2x in tau. A flicker floor is
    flat over roughly a decade. That is the difference this measures.
    """
    if len(adev) < 5:
        return None, "too few tau points"
    i = int(np.argmin(adev))
    if i < 2 or i > len(adev) - 3:
        return None, f"minimum at the edge of the tau range ({tau[i]:.2g} s): not a minimum"
    near = adev <= (1.0 + flat_frac) * adev[i]
    lo = i
    while lo > 0 and near[lo - 1]:
        lo -= 1
    hi = i
    while hi < len(near) - 1 and near[hi + 1]:
        hi += 1
    span = float(tau[hi] / tau[lo])
    if span < min_decade:
        return None, (f"minimum at tau {tau[i]:.2g} s is a {span:.1f}x-wide notch, not a "
                      f"flicker plateau (needs {min_decade:.0f}x within {flat_frac:.0%}): "
                      f"looks like an ARW / rate-random-walk crossover")
    return float(adev[i] / 0.664), ""


def euler_rates_to_body(ang, dt_s):
    """Fused ZYX Euler angles on a uniform grid -> predicted BODY rates (n, 3).

    ang is (n, 3) = (roll phi, pitch theta, yaw psi) in radians, possibly wrapped;
    dt_s is the grid step in seconds. ZYX (yaw-pitch-roll) kinematics:

        wx = phi'                       - psi' sin(theta)
        wy =        th' cos(phi)        + psi' cos(theta) sin(phi)
        wz =       -th' sin(phi)        + psi' cos(theta) cos(phi)

    Differentiating roll and calling it wx is true only near upright, which is not
    where this robot spends its time, so the map is applied before anything is
    compared against the gyro. Unwrapping happens here, per caller-supplied
    segment: it must never be applied across a dropout, where a jump in the angle
    is a real jump and not a wrap. imulog.py's __main__ checks this function
    against body rates derived independently from R^T dR/dt, because a sign flip
    here is invisible in a lag estimate (a pure delay commutes with any static
    mixing of the axes).
    """
    ang = np.unwrap(np.asarray(ang, np.float64), axis=0)
    dphi, dth, dpsi = (np.gradient(ang[:, a], dt_s) for a in range(3))
    phi, th = ang[:, 0], ang[:, 1]
    return np.stack([dphi - dpsi * np.sin(th),
                     dth * np.cos(phi) + dpsi * np.cos(th) * np.sin(phi),
                     -dth * np.sin(phi) + dpsi * np.cos(th) * np.cos(phi)], 1)


def _segments(ts, gap_ms):
    """[(i0, i1)] index ranges of one stream, split wherever dt exceeds gap_ms."""
    if len(ts) < 2:
        return []
    cuts = np.flatnonzero(np.diff(ts) > gap_ms) + 1
    edges = np.concatenate(([0], cuts, [len(ts)])).astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b - a >= 2]


def filter_lag(fused_angles, ts_f, gyro, ts_g, max_lag_ms=500.0, min_corr=0.5,
               gap_ms=GAP_MS, gimbal_rad=GIMBAL_PITCH_RAD, mask_margin=3):
    """Per-axis lag (ms) of the fused orientation behind the raw gyro.

    Euler-angle rates are NOT body rates once the body tilts, so the fused ZYX
    angles are pushed through euler_rates_to_body() first, and the predicted body
    rates are cross-correlated axis-to-axis against the raw gyro on a uniform grid
    at the gyro's median dt, with parabolic refinement of the peak. Positive lag =
    the orientation stream is late.

    What the correlation refuses to do, and why:

      seams and dropouts   the grid is built per contiguous segment, splitting
                           wherever the gyro dt exceeds gap_ms -- the same
                           threshold parse() cuts episodes at, which also catches
                           the artificial seam between concatenated files.
                           Interpolating and differentiating across a 500 ms
                           dropout invents an enormous phantom body rate at a
                           position that has nothing to do with the filter, and
                           np.unwrap across one invents a false 2pi correction.
                           The correlation sums (numerator and both norms) are
                           accumulated across segments and normalised at the end,
                           so the segments contribute jointly without any of them
                           spanning a gap.
      filter warm-up       the first max_lag_ms of every segment is dropped. Right
                           after a discontinuity a causal fusion filter's output is
                           a blend of data from both sides of it, for as long as its
                           memory -- which is the quantity being measured, bounded
                           by max_lag_ms. Those samples are a function of data the
                           file does not contain, and on imulog.py's fixture they
                           are what drags an injected 60 ms lag to +79..+91 ms
                           across the three axes. Same guard, same reason, as the
                           post-cut ticks servo_id.identify() excludes on the
                           actuator side.
      gimbal lock          near |pitch| = pi/2 the ZYX map is singular: roll and
                           yaw jump by ~pi (np.unwrap removes 2pi jumps, not pi
                           ones) and 1/cos(pitch) amplifies interpolation noise
                           without bound. This robot falls routinely, so grid
                           samples with |pitch| > gimbal_rad, plus mask_margin
                           samples of guard around each masked run, are dropped
                           from the sums. An axis with less than half the grid
                           left is undetermined, not estimated.
      rail peaks           an argmax at the edge of the +-max_lag_ms search is not
                           a lag, it is the search running out; it is reported as
                           boundary=True and "at least the window", never as a
                           number.

    Returns a dict: "axes" (one entry per body axis) plus the conditions the
    numbers were produced under -- grid dt, segments used and dropped, time
    excluded at gaps and to warm-up, the gimbal-kept and usable fractions, and the
    search parameters.
    """
    ts_f = np.asarray(ts_f, np.float64); ts_g = np.asarray(ts_g, np.float64)
    ang = np.asarray(fused_angles, np.float64); gyr = np.asarray(gyro, np.float64)

    def undetermined(reason, dtg=np.nan, meta=None):
        base = {"axes": [{"axis": BODY_AXES[a], "lag_ms": None, "corr": float("nan"),
                          "determined": False, "boundary": False, "reason": reason}
                         for a in range(3)],
                "grid_dt_ms": float(dtg), "max_lag_ms": float(max_lag_ms),
                "min_corr": float(min_corr), "gap_ms": float(gap_ms),
                "gimbal_pitch_rad": float(gimbal_rad),
                "segments_used": 0, "segments_dropped": 0, "excluded_ms": 0.0,
                "warmup_excluded_ms": 0.0, "grid_samples": 0, "usable_samples": 0,
                "usable_frac": float("nan"), "gimbal_kept_frac": float("nan")}
        base.update(meta or {})
        return base

    if len(ts_g) < 8 or len(ts_f) < 8:
        return undetermined(f"only {len(ts_g)} gyro / {len(ts_f)} orientation rows")
    d = np.diff(ts_g)
    within = d[(d > 0) & (d <= gap_ms)]
    if not len(within):
        return undetermined("no two consecutive gyro samples inside the gap threshold")
    dtg = float(np.median(within))
    L = max(1, int(round(max_lag_ms / dtg)))

    pieces = []
    grid_n = gim_n = good_n = 0
    used = dropped = 0
    excluded_ms = float(d[d > gap_ms].sum())            # time inside dropouts and seams
    warmup_ms = 0.0
    for i0, i1 in _segments(ts_g, gap_ms):
        t0 = max(float(ts_g[i0]), float(ts_f[0])); t1 = min(float(ts_g[i1 - 1]), float(ts_f[-1]))
        fsel = (ts_f >= ts_g[i0] - dtg) & (ts_f <= ts_g[i1 - 1] + dtg)
        grid = np.arange(t0, t1, dtg)
        # A segment must outlast the warm-up guard by at least one search window,
        # or it has nothing left to correlate; it is dropped and counted, never
        # allowed to contribute pairs to the small lags only.
        if len(grid) < 2 * L + 4 or int(fsel.sum()) < 4:
            dropped += 1; excluded_ms += max(0.0, t1 - t0); continue
        used += 1
        a_un = np.unwrap(ang[fsel], axis=0)             # unwrap INSIDE the segment only
        a_seg = np.stack([np.interp(grid, ts_f[fsel], a_un[:, a]) for a in range(3)], 1)
        w_seg = euler_rates_to_body(a_seg, dtg / 1000.0)
        g_seg = np.stack([np.interp(grid, ts_g[i0:i1], gyr[i0:i1, a]) for a in range(3)], 1)
        bad = np.abs(a_seg[:, 1]) > gimbal_rad
        if bad.any() and mask_margin > 0:               # guard band around each masked run
            bad = np.convolve(bad.astype(np.float64), np.ones(2 * mask_margin + 1), "same") > 0
        good = ~bad
        grid_n += len(grid); gim_n += int(good.sum())
        good[:L] = False                                # filter warm-up after the discontinuity
        good[-1] = False                                # np.gradient is one-sided at the end
        warmup_ms += (L + 1) * dtg
        good_n += int(good.sum())
        pieces.append((w_seg, g_seg, good.astype(np.float64)))

    meta = {"grid_dt_ms": dtg, "max_lag_ms": float(max_lag_ms), "min_corr": float(min_corr),
            "gap_ms": float(gap_ms), "gimbal_pitch_rad": float(gimbal_rad),
            "segments_used": used, "segments_dropped": dropped,
            "excluded_ms": float(excluded_ms), "warmup_excluded_ms": float(warmup_ms),
            "grid_samples": int(grid_n), "usable_samples": int(good_n),
            "usable_frac": float(good_n / grid_n) if grid_n else float("nan"),
            "gimbal_kept_frac": float(gim_n / grid_n) if grid_n else float("nan")}
    if not pieces or good_n < 4 * L:
        return undetermined(f"only {good_n} usable grid samples left after the gap split, the "
                            f"warm-up guard and the gimbal mask", dtg, meta)

    mean_w = np.zeros(3); mean_g = np.zeros(3)
    tot = sum(float(v.sum()) for _, _, v in pieces)
    for a in range(3):
        mean_w[a] = sum(float((w[:, a] * v).sum()) for w, g, v in pieces) / tot
        mean_g[a] = sum(float((g[:, a] * v).sum()) for w, g, v in pieces) / tot

    num = np.zeros((3, 2 * L + 1)); nx = np.zeros((3, 2 * L + 1)); ny = np.zeros((3, 2 * L + 1))
    pairs = np.zeros(2 * L + 1)
    for w, g, v in pieces:
        n = len(v)
        for j, k in enumerate(range(-L, L + 1)):
            sx, sy = (slice(k, n), slice(0, n - k)) if k >= 0 else (slice(0, n + k), slice(-k, n))
            if sx.stop - sx.start < 2:
                continue
            vv = v[sx] * v[sy]                      # a pair counts only if BOTH ends survive
            pairs[j] += float(vv.sum())
            for a in range(3):
                x = (w[sx, a] - mean_w[a]) * vv
                y = g[sy, a] - mean_g[a]
                num[a, j] += float(x @ y)
                nx[a, j] += float(((w[sx, a] - mean_w[a]) ** 2 * vv).sum())
                ny[a, j] += float((y ** 2 * vv).sum())

    # lags supported by less than a quarter of the zero-lag pairs cannot win the
    # argmax: with many short segments their correlation is an artefact of the count
    eligible = pairs >= 0.25 * pairs[L]
    out = []
    for a in range(3):
        rho = num[a] / (np.sqrt(nx[a] * ny[a]) + 1e-12)
        r = np.where(eligible, rho, -np.inf)
        i = int(np.argmax(r))
        k = float(i - L)
        railed = bool(i == 0 or i == 2 * L)
        starved = bool(not eligible[min(i + 1, 2 * L)] or not eligible[max(i - 1, 0)])
        boundary = railed or starved
        if 0 < i < 2 * L and not boundary:              # parabolic refinement around the peak
            denom = rho[i - 1] - 2 * rho[i] + rho[i + 1]
            if abs(denom) > 1e-12:
                k = k + 0.5 * (rho[i - 1] - rho[i + 1]) / denom
        peak = float(rho[i])
        lag = float(k * dtg)
        reason = ""
        det = peak >= min_corr
        if meta["gimbal_kept_frac"] < 0.5:
            det = False
            reason = (f"only {meta['gimbal_kept_frac']:.0%} of the grid survives the "
                      f"|pitch| > {gimbal_rad:.2f} rad gimbal mask")
        elif railed:
            det = False
            reason = (f"peak pegged at the edge of the +-{max_lag_ms:.0f} ms search window: "
                      f"the lag is >= the window, not {lag:+.1f} ms")
        elif starved:
            det = False
            reason = (f"peak at {lag:+.1f} ms sits where the segments stop overlapping (fewer "
                      f"than a quarter of the zero-lag pairs): the search ran out of data, "
                      f"not out of correlation")
        elif not det:
            reason = (f"peak corr {peak:.2f} below min_corr {min_corr:.2f} -- euler/body-rate "
                      f"mismatch, or too little motion on this axis")
        out.append({"axis": BODY_AXES[a], "lag_ms": lag, "corr": peak,
                    "determined": bool(det), "boundary": boundary, "reason": reason})
    return {"axes": out, **meta}


def _fmt_lag(r, band=None, verdict=""):
    """One report line for one axis: the number with its band, or the reason it has none."""
    if not r["determined"]:
        tag = "boundary" if r.get("boundary") else "undetermined"
        corr = "" if not np.isfinite(r["corr"]) else f" (peak corr {r['corr']:.2f})"
        return f"{tag}{corr} -- {r['reason']}"
    b = "" if band is None else f" +- {band:.1f}"
    return (f"{r['lag_ms']:+7.1f}{b} ms   peak corr {r['corr']:.2f}"
            + (f"   split-half {verdict}" if verdict else ""))


def main():
    from imulog import _read_rows, run_preflight
    ap = argparse.ArgumentParser(description="sensor-side characterisation of a ?imulog=1 session")
    ap.add_argument("log", nargs="+", help="session file(s); clocks are made sequential across files")
    ap.add_argument("--min-still-s", type=float, default=60.0,
                    help="still segment length below which Allan results are reported as undetermined-prone")
    ap.add_argument("--max-lag-ms", type=float, default=500.0, help="cross-correlation search half-width")
    ap.add_argument("--min-corr", type=float, default=0.5, help="peak correlation below which a lag is not reported")
    args = ap.parse_args()

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
        print(f"  {r['axis']:>5}  " + _fmt_lag(r, band, verdict or ""))

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
    if stills:
        t0, t1 = max(stills, key=lambda w: w[1] - w[0])
        sel = (imu_t >= t0) & (imu_t < t1)
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
        still_meta = {"t0_ms": float(t0), "t1_ms": float(t1), "span_s": float(span),
                      "samples": int(sel.sum()), "fs_hz": float(fs), "rate_ok": bool(rate_ok),
                      "dt": seg_dt, "stillness": vs, "undetermined_reasons": reasons}
        if np.isfinite(fs):
            allan = allan_deviation(imu_v[sel, 3:], fs)
            print(f"  fs {fs:.2f} Hz from the segment's median dt")
            for a, r in enumerate(allan):
                if reasons:
                    print(f"  {BODY_AXES[a]:>5}  ARW and bias instability undetermined -- {reasons[0]}")
                    continue
                arw = (f"ARW {r['arw']:.2e} rad/s/sqrt(Hz)" if r["arw"] is not None
                       else "ARW undetermined (no -1/2 slope around tau = 1 s)")
                bi = (f"bias instability {r['bias_instability']:.2e} rad/s"
                      if r["bias_instability"] is not None
                      else f"bias instability undetermined ({r['bias_reason']})")
                print(f"  {BODY_AXES[a]:>5}  {arw};  {bi}")
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
                "bias_instability": None if nulled else r["bias_instability"],
                "bias_reason": still_meta["undetermined_reasons"][0] if nulled else r["bias_reason"],
                "tau_s": r["tau"].tolist(), "adev": r["adev"].tolist()}
               for a, r in enumerate(allan)]}
    with open("results/sensor_id.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote results/sensor_id.json")


if __name__ == "__main__":
    main()
