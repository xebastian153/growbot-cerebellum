"""com_id scores with servo_id's machinery, not a copy of it (AGENTS.md: two
implementations of the same math get an equivalence test).

`one_step_scorer` must return the number `servo_id.identify` returns for the same
hypothesis on the same log, and the band / determined-set rules com_id imports must give
`confidence_band` / `determined_sets` back on the servo grid."""
from __future__ import annotations
import itertools
import numpy as np

from growbot_cerebellum.forward import Linear, K
from growbot_cerebellum.servo_id import (identify, realized_from_commands, confidence_band,
                                         determined_sets, band_from_errors, within_band, _key)
import com_id


def _log(seed, n=600):
    rng = np.random.default_rng(seed)
    O = rng.normal(size=(n, 6)).astype(np.float32)
    A = np.clip(rng.normal(scale=0.4, size=(n, 2)), -1, 1).astype(np.float32)
    O2 = O + 0.05 * rng.normal(size=O.shape).astype(np.float32)
    D = np.zeros(n, bool)
    D[::97] = True                                    # cuts, so the transient guard has work to do
    return O, A, O2, D


def _model(O, A, O2, D):
    from growbot_cerebellum.forward import make_windows
    X, Y, *_ = make_windows(O, A, O2, D, K)
    return Linear().fit(X, Y)


GRID = list(itertools.product([0, 2, 4], [2.0, None], [0.0, float(np.deg2rad(2))]))


def test_one_step_scorer_equals_identify():
    O, A, O2, D = _log(0)
    m = _model(O, A, O2, D)
    scores, _ = identify(m, O, A, O2, D, GRID)
    ref = {_key(kw): e for e, kw in scores}
    score = com_id.one_step_scorer(O, A, O2, D, max(d for d, _, _ in GRID))
    for d, s, db in GRID:
        kw = dict(delay_ticks=d, slew_rad_s=s, deadband=db)
        R = realized_from_commands(A, D, kw)
        assert score(m, R) == ref[_key(kw)], kw


def test_ideal_servo_scorer_is_identify_at_delay_zero():
    # identification A scores every candidate with max_delay 0 and R = A; that is
    # identify() on the one-hypothesis grid (0, None, 0), whose realised command is A itself
    O, A, O2, D = _log(1)
    m = _model(O, A, O2, D)
    scores, _ = identify(m, O, A, O2, D, [(0, None, 0.0)])
    # ServoModel replays in float64 and casts back, so the ideal replay matches A to the
    # last float32 digit, not bit-exactly; the scores agree to the same order
    R = realized_from_commands(A, D, dict(delay_ticks=0, slew_rad_s=None, deadband=0.0))
    assert np.allclose(R, A, atol=1e-6) and not np.array_equal(R, A)
    assert abs(com_id.one_step_scorer(O, A, O2, D, 0)(m, A) - scores[0][0]) < 1e-6


def test_band_and_within_band_equal_servo_id_on_the_servo_grid():
    O, A, O2, D = _log(2)
    m = _model(O, A, O2, D)
    h = len(O) // 2
    sA, _ = identify(m, O[:h], A[:h], O2[:h], D[:h], GRID)
    sB, _ = identify(m, O[h:], A[h:], O2[h:], D[h:], GRID)
    eA = {_key(kw): e for e, kw in sA}
    eB = {_key(kw): e for e, kw in sB}
    band = confidence_band(sA, sB)
    assert band == band_from_errors(eA, eB)
    best = sA[0][1]
    delays_ref, slews_ref = determined_sets(sA, best, GRID, band)
    db = round(float(best["deadband"]), 5)
    delays = within_band(eA, sA[0][0], band, sorted({d for d, _, _ in GRID}), lambda v: (v, best["slew_rad_s"], db))
    slews = within_band(eA, sA[0][0], band, sorted({s for _, s, _ in GRID}, key=lambda v: (v is None, v)),
                        lambda v: (best["delay_ticks"], v, db))
    assert delays == delays_ref and slews == slews_ref


def test_joint_verdict_marks_two_point_axes_as_boundary():
    # slew and deadband carry two values each, so no joint argmin is interior on them
    err = {k: 1.0 for k in com_id.joint_grid()}
    err[(0.0, 3, 2.0, 0.0)] = 0.0                     # interior on x and delay, boundary on the rest
    v = com_id.joint_verdict(err, 0.5, 0.0, dict(delay_ticks=3, slew_rad_s=2.0, deadband=0.0))
    assert v["argmin_interior_x"] and v["argmin_interior_delay"]
    assert v["argmin_interior"] is False and v["two_point_axes"] == ["slew", "deadband"]
    assert v["argmin_deadband_correct"] and v["deadband_determined_deg"] == [0.0]


def test_summarize_excludes_below_noise_seeds():
    s = com_id.summarize([True, False, True], counted=[True, False, True])
    assert s["resolved"] and s["value"] is True and s["n_counted"] == 2
    assert com_id.summarize([True, False], counted=[False, False])["resolved"] is False
