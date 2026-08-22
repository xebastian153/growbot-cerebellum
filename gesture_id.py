"""What an `act` (gesture) capture can and cannot identify about the servo.

We asked the maintainer for varied excitation because ~15 s of periodic walking left
the servo delay undetermined over the whole grid: a gait moves the horns through one
narrow band at one rhythm, which is the worst case for separating a latency from a
rate limit. A gesture lane moves them through 5-175 deg in steps of 40-110 deg, so on
excitation alone it is the better file.

It also introduces a confound the walk lane does not have, and the confound is
decisive rather than cosmetic. The `act` verb carries a DURATION: the header documents
its own example as {l:130, r:50, ms:700}, a 40 deg move in 700 ms, i.e. a commanded
ramp of 1.0 rad/s. The glide engine between brain and servo plays that ramp out, but
the log records only the keyframe endpoints -- 53 rows for 217 s -- with no `ms`. So
`realized_from_commands`, which replays a STEP and lets the candidate servo's own slew
shape the trajectory, is being handed a command the body never received.

The consequence is not that the answer is noisy. It is that the answer is a
measurement of the wrong thing: at 1.0 rad/s the commanded ramp is SLOWER than the
2.0-3.0 rad/s slew the walk lane identified, so the horn never reaches its own limit
and whatever slew best explains this file is the glide engine's rate. A delay is a
different matter -- a pure time shift that a ramp does not hide -- which is why this
script reports the two separately instead of a single triple.
"""
from __future__ import annotations
import json, sys
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "sim")
from forward import MLP, make_windows
from imulog import parse, run_preflight
from sensor_id import default_out_path
from sim2real_proxy import K
from servo_id import (identify, confidence_band, determined_sets, default_grid,
                      argmin_interior)

LOG = "SEND-gesture-3.6min.json"
WALK = "imu-walk-1-2026-08-20T17-50-14-713Z.json"
DOC_ACT_DEG, DOC_ACT_MS = 40.0, 700.0     # the header's own documented act


def glide_rate_rad_s():
    return float(np.deg2rad(DOC_ACT_DEG) / (DOC_ACT_MS / 1000.0))


def keyframe_stats(path):
    obj = json.load(open(path))
    p = np.asarray(obj["pose"], float)
    step = np.abs(np.diff(p[:, 2:4], axis=0)).max(1)
    dtk = np.diff(p[:, 1])
    rate = glide_rate_rad_s()
    # how long the glide engine needs for each commanded step, at the documented rate
    need_ms = 1000.0 * np.deg2rad(step) / rate
    return {"n_keyframes": int(len(p)),
            "span_s": float((p[-1, 1] - p[0, 1]) / 1000.0),
            "step_deg": {"median": float(np.median(step)), "max": float(step.max())},
            "inter_keyframe_s": {"min": float(dtk.min() / 1000), "p25": float(np.percentile(dtk, 25) / 1000),
                                 "median": float(np.median(dtk) / 1000), "max": float(dtk.max() / 1000)},
            "glide_rate_rad_s": rate,
            "glide_ms_needed": {"median": float(np.median(need_ms)), "max": float(need_ms.max())},
            "frac_keyframes_arriving_mid_glide": float(np.mean(need_ms[:len(dtk)] > dtk))}


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
    return {"log": path, "label": label, "ticks": int(n), "fit_ticks": int(half),
            "argmin": {k: (None if v is None else float(v)) for k, v in best.items()},
            "band": float(band), "delay_determined": [int(d) for d in dset],
            "slew_determined": [None if s is None else float(s) for s in sset],
            "argmin_interior": bool(interior), "interior_why": why,
            "split_half": {"A": {k: (None if v is None else float(v)) for k, v in sA[0][1].items()},
                           "B": {k: (None if v is None else float(v)) for k, v in sB[0][1].items()},
                           "agree": sA[0][1] == sB[0][1]}}


def main():
    for f in (LOG, WALK):
        if not run_preflight(f):
            raise SystemExit(f"preflight FAIL on {f}")
    tr = np.load("data/train.npz")
    Xtr, Ytr, *_ = make_windows(tr["obs"], tr["act"], tr["next_obs"], tr["done"], K)
    model = MLP(hidden=128, epochs=80).fit(Xtr, Ytr)
    grid = default_grid()

    ks = keyframe_stats(LOG)
    print("\n== the glide, from the header's own documented act")
    print(f"  {DOC_ACT_DEG:.0f} deg in {DOC_ACT_MS:.0f} ms = {ks['glide_rate_rad_s']:.2f} rad/s commanded ramp")
    print(f"  {ks['n_keyframes']} keyframes over {ks['span_s']:.0f} s; steps median "
          f"{ks['step_deg']['median']:.0f} deg, max {ks['step_deg']['max']:.0f} deg")
    print(f"  at that rate a median step needs {ks['glide_ms_needed']['median']:.0f} ms, "
          f"the largest {ks['glide_ms_needed']['max']:.0f} ms")
    print(f"  inter-keyframe gap median {ks['inter_keyframe_s']['median']:.2f} s, min "
          f"{ks['inter_keyframe_s']['min']:.2f} s")
    print(f"  {ks['frac_keyframes_arriving_mid_glide'] * 100:.0f}% of keyframes arrive while the "
          f"previous glide is still playing")

    out = {"glide": ks, "runs": []}
    print("\n== identification, same grid and protocol on both files")
    for path, label in ((LOG, "gesture (act)"), (WALK, "walk-1 (official)")):
        r = run(model, path, grid, label)
        out["runs"].append(r)
        a = r["argmin"]
        print(f"\n  {label}: {r['ticks']} ticks, identify on the first {r['fit_ticks']}")
        print(f"    argmin delay {a['delay_ticks']:.0f} ticks, slew {a['slew_rad_s']}, "
              f"deadband {np.rad2deg(a['deadband']):.0f} deg"
              + ("" if r["argmin_interior"] else "   [AT THE GRID BOUNDARY]"))
        print(f"    determined: delay {r['delay_determined']}  slew {r['slew_determined']}  "
              f"(band {r['band']:.5f})")
        print(f"    split-half: A={r['split_half']['A']['delay_ticks']:.0f}/"
              f"{r['split_half']['A']['slew_rad_s']}  B={r['split_half']['B']['delay_ticks']:.0f}/"
              f"{r['split_half']['B']['slew_rad_s']}  "
              f"{'AGREE' if r['split_half']['agree'] else 'DISAGREE'}")

    g, w = out["runs"][0], out["runs"][1]
    out["reading"] = {
        "slew_interpretable": False,
        "slew_reason": (f"the identified slew {g['argmin']['slew_rad_s']} rad/s is the glide engine's "
                        f"commanded ramp ({ks['glide_rate_rad_s']:.2f} rad/s from the header's own act), "
                        f"not the horn's limit: the ramp is slower than the {w['slew_determined']} rad/s "
                        f"the walk lane determines, so the horn is never asked to move at its limit"),
        "delay_tightened_vs_walk": len(g["delay_determined"]) < len(w["delay_determined"]),
    }
    print("\n== reading")
    print(f"  slew from this file is NOT the servo's: {out['reading']['slew_reason']}")
    print(f"  delay set: gesture {g['delay_determined']} vs walk-1 {w['delay_determined']} -- "
          f"{'tighter' if out['reading']['delay_tightened_vs_walk'] else 'not tighter'}")
    path = default_out_path([LOG], "gesture_id")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
