"""`confidence_band` and `determined_sets` exactly as growbot_cerebellum/servo_id.py shipped
them at fb73537, BEFORE they were split into `band_from_errors` / `within_band`. A golden:
the refactored functions must reproduce these on the servo grid. Do not edit; if the math
is meant to change, the pin moves with a stated reason."""
import numpy as np


def _key(kw):
    return kw["delay_ticks"], kw["slew_rad_s"], round(float(kw["deadband"]), 5)


def confidence_band(scoresA, scoresB):
    eA = {_key(kw): e for e, kw in scoresA}
    eB = {_key(kw): e for e, kw in scoresB}
    d = np.array([eA[k] - eB[k] for k in eA])
    return float(1.4826 * np.median(np.abs(d - np.median(d)))) / 2.0


def determined_sets(scores, best, grid, band):
    fit_err = {_key(kw): e for e, kw in scores}
    best_e = scores[0][0]
    db = round(float(best["deadband"]), 5)

    def determined(values, fixed):
        keep = {v for v in values if fit_err.get(fixed(v), np.inf) - best_e <= band}
        return sorted(keep, key=lambda v: (v is None, v))

    delays = sorted({d for d, _, _ in grid})
    slews = sorted({s for _, s, _ in grid}, key=lambda v: (v is None, v))
    return (determined(delays, lambda v: (v, best["slew_rad_s"], db)),
            determined(slews, lambda v: (best["delay_ticks"], v, db)))
