"""Characterise the SENSOR side of the sim-to-real gap from a ?imulog=1 log alone.

servo_id.py identifies the actuator between command and body (delay, slew,
deadband). This module is the symmetric counterpart on the observation side:
the phone does not report ground truth, it reports the output of a sensor-fusion
filter running on jittered timestamps. Three measurements, all from the same
session file the actuator analysis uses, no extra hardware:

  dt_stats()         clock jitter per stream -- the forward model assumes a
                     fixed tick; the timestamps say how true that is.
  filter_lag()       per-axis lag of the fused orientation behind the raw gyro,
                     by cross-correlating the differentiated fused angles
                     against the gyro. Positive = orientation lags the gyro.
  allan_deviation()  overlapping Allan deviation of the still-segment gyro:
                     angle random walk, and bias instability when the curve
                     shows an interior minimum -- noise parameters measured
                     from the phone, to inject into the twin instead of
                     invented gaussians.

Validated like the parser: imulog.py's fixture can hide a known fused-filter
lag, a known angle-random-walk density and a timing stall; the __main__ test
there must recover all three or fail.
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np

AXES = ("roll", "pitch", "yaw")


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
    fixed-dt forward model; it is a report, never an analysis gate.
    """
    ts = np.asarray(ts, np.float64)
    if len(ts) < 3:
        return {"median_ms": np.nan, "p95_ms": np.nan, "p99_ms": np.nan,
                "max_ms": np.nan, "frac_dev20": np.nan, "jitter_warn": False}
    return _stats_from_diffs(np.diff(ts))


def still_windows(events, t_end):
    """(t0, t1) spans labelled still/idle, same event semantics as preflight:
    'X_start' opens X until the next event, 'X_stop' returns to idle."""
    ev = sorted(events); out = []
    for i, (t0, n) in enumerate(ev):
        name = "idle" if n.endswith("_stop") else n.removesuffix("_start")
        if name in ("still", "idle"):
            t1 = ev[i + 1][0] if i + 1 < len(ev) else t_end
            if t1 > t0:
                out.append((float(t0), float(t1)))
    return out


def allan_deviation(gyro, fs, n_taus=40):
    """Overlapping Allan deviation per axis on still gyro data at rate fs (Hz).

    Returns one dict per axis: tau (s), adev (rad/s), and two extracted
    parameters with their determinability stated rather than guessed:
      arw               angle random walk N (rad/s/sqrt(Hz)); the -1/2-slope
                        law read at tau = 1 s, only if the local log-log slope
                        around 1 s is -1/2 within [-0.65, -0.35], else None.
      bias_instability  B (rad/s) = adev_min / 0.664, only if the curve has an
                        interior, locally flat minimum inside the tau range --
                        a short still segment usually cannot determine B, and
                        the honest answer is None, not the edge value.
    """
    g = np.asarray(gyro, np.float64)
    if g.ndim == 1:
        g = g[:, None]
    N = len(g)
    out = []
    if N < 32:
        return [{"tau": np.array([]), "adev": np.array([]),
                 "arw": None, "bias_instability": None} for _ in range(g.shape[1])]
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
        out.append({"tau": tau, "adev": adev,
                    "arw": _arw(tau, adev), "bias_instability": _bias(tau, adev)})
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


def _bias(tau, adev):
    i = int(np.argmin(adev))
    if i < 2 or i > len(adev) - 3:
        return None                                # minimum at an edge is not a minimum
    lt, ls = np.log10(tau[i - 2:i + 3]), np.log10(adev[i - 2:i + 3])
    if abs(float(np.polyfit(lt, ls, 1)[0])) > 0.2:
        return None
    return float(adev[i] / 0.664)


def filter_lag(fused_angles, ts_f, gyro, ts_g, max_lag_ms=500.0, min_corr=0.5):
    """Per-axis lag (ms) of the fused orientation behind the raw gyro.

    Euler-angle rates are NOT body rates once the body tilts -- differentiating
    roll and correlating it against gyro x washes out exactly in the fallen and
    tumbling segments this robot lives in. So the fused ZYX angles are pushed
    through the euler-rate kinematics to PREDICTED body rates first
    (w_x = phi' - psi' sin(theta), etc.), and those are cross-correlated
    axis-to-axis against the raw gyro on a uniform grid at the gyro's median
    dt, with parabolic refinement of the peak. Positive lag = the orientation
    stream is late. The peak correlation is the confidence; below min_corr
    (frame or mount convention mismatch, or too little motion on the axis) the
    axis is reported undetermined instead of printing a meaningless lag.
    """
    ts_f = np.asarray(ts_f, np.float64); ts_g = np.asarray(ts_g, np.float64)
    ang = np.unwrap(np.asarray(fused_angles, np.float64), axis=0)
    gyr = np.asarray(gyro, np.float64)
    dtg = float(np.median(np.diff(ts_g)))
    lo, hi = max(ts_f[0], ts_g[0]), min(ts_f[-1], ts_g[-1])
    grid = np.arange(lo, hi, dtg)
    ag = np.stack([np.interp(grid, ts_f, ang[:, a]) for a in range(3)], 1)
    dphi, dth, dpsi = (np.gradient(ag[:, a], dtg / 1000.0) for a in range(3))
    phi, th = ag[:, 0], ag[:, 1]
    w_pred = np.stack([dphi - dpsi * np.sin(th),
                       dth * np.cos(phi) + dpsi * np.cos(th) * np.sin(phi),
                       -dth * np.sin(phi) + dpsi * np.cos(th) * np.cos(phi)], 1)
    L = max(1, int(round(max_lag_ms / dtg)))
    out = []
    for a in range(3):
        ra = w_pred[:, a]
        ga = np.interp(grid, ts_g, gyr[:, a])
        ra = ra - ra.mean(); ga = ga - ga.mean()
        n = len(grid)
        rho = np.empty(2 * L + 1)
        for j, k in enumerate(range(-L, L + 1)):
            x, y = (ra[k:], ga[:n - k]) if k >= 0 else (ra[:n + k], ga[-k:])
            rho[j] = x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)
        i = int(np.argmax(rho))
        k = i - L
        if 0 < i < 2 * L:                          # parabolic refinement around the peak
            denom = rho[i - 1] - 2 * rho[i] + rho[i + 1]
            if abs(denom) > 1e-12:
                k = k + 0.5 * (rho[i - 1] - rho[i + 1]) / denom
        peak = float(rho[i])
        out.append({"axis": AXES[a], "lag_ms": float(k * dtg), "corr": peak,
                    "determined": peak >= min_corr})
    return out


def main():
    from imulog import _read_rows, run_preflight
    ap = argparse.ArgumentParser(description="sensor-side characterisation of a ?imulog=1 session")
    ap.add_argument("log", nargs="+", help="session file(s); clocks are made sequential across files")
    ap.add_argument("--min-still-s", type=float, default=60.0,
                    help="still segment length below which Allan results are reported as undetermined-prone")
    args = ap.parse_args()

    header = None
    imu_t, imu_v, cmd_t = [], [], []
    stills, imu_d, cmd_d = [], [], []
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
        imu_t.append(it + shift); imu_v.append(iv); cmd_t.append(ct + shift)
        imu_d.append(np.diff(it)); cmd_d.append(np.diff(ct))   # per-file diffs: no seam dt
        stills += [(t0 + shift, t1 + shift) for t0, t1 in still_windows(events, it[-1])]
        offset = it[-1] + shift + 1000.0
    imu_t = np.concatenate(imu_t); imu_v = np.concatenate(imu_v)

    print(f"\nclock ({len(args.log)} file(s); dt within files only)")
    for name, d in (("imu", np.concatenate(imu_d)), ("cmd", np.concatenate(cmd_d))):
        st = _stats_from_diffs(d)
        print(f"  {name}  dt ms: median {st['median_ms']:.1f}  p95 {st['p95_ms']:.1f}  "
              f"p99 {st['p99_ms']:.1f}  max {st['max_ms']:.0f}  "
              f">20% off-median {st['frac_dev20'] * 100:.1f}%"
              + ("   JITTER above 1.5x median at p99" if st["jitter_warn"] else ""))
        if name == "imu":
            imu_stats = st

    print("\nfused-orientation lag behind the raw gyro (cross-correlation, whole session)")
    lags = filter_lag(imu_v[:, :3], imu_t, imu_v[:, 3:], imu_t)
    for r in lags:
        if r["determined"]:
            print(f"  {r['axis']:>5}  {r['lag_ms']:+7.1f} ms   peak corr {r['corr']:.2f}")
        else:
            print(f"  {r['axis']:>5}  undetermined -- peak corr {r['corr']:.2f} below 0.5; "
                  f"euler/body rate mismatch or too little motion on this axis")

    allan = None
    if stills:
        t0, t1 = max(stills, key=lambda w: w[1] - w[0])
        sel = (imu_t >= t0) & (imu_t < t1)
        span = (t1 - t0) / 1000.0
        print(f"\nAllan deviation on the longest still segment ({span:.0f} s, {int(sel.sum()):,} samples)")
        if span < args.min_still_s:
            print(f"  segment shorter than {args.min_still_s:.0f} s -- expect undetermined parameters; "
                  f"reporting only what the curve supports")
        fs = 1000.0 / float(np.median(np.diff(imu_t[sel])))
        allan = allan_deviation(imu_v[sel, 3:], fs)
        for a, r in enumerate(allan):
            arw = (f"ARW {r['arw']:.2e} rad/s/sqrt(Hz)" if r["arw"] is not None
                   else "ARW undetermined (no -1/2 slope around tau = 1 s)")
            bi = (f"bias instability {r['bias_instability']:.2e} rad/s" if r["bias_instability"] is not None
                  else "bias instability undetermined (no interior minimum in the tau range)")
            print(f"  {AXES[a]:>5}  {arw};  {bi}")
    else:
        print("\nno still/idle segment labelled in the log -- Allan analysis skipped; "
              "record one still segment per session to characterise the gyro noise")

    out = {"files": args.log, "imu_dt": imu_stats,
           "filter_lag": lags,
           "allan": None if allan is None else [
               {"axis": AXES[a], "arw": r["arw"], "bias_instability": r["bias_instability"],
                "tau_s": r["tau"].tolist(), "adev": r["adev"].tolist()}
               for a, r in enumerate(allan)]}
    json.dump(out, open("results/sensor_id_log.json", "w"), indent=1)
    print("\nwrote results/sensor_id_log.json")


if __name__ == "__main__":
    main()
