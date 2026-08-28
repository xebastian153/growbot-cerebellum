"""com_id scores with servo_id's machinery, not a copy of it (AGENTS.md: two
implementations of the same math get an equivalence test).

`one_step_scorer` must return the number `servo_id.identify` returns for the same
hypothesis on the same log; the refactored `confidence_band` / `determined_sets` (now thin
wrappers over `band_from_errors` / `within_band`) must reproduce the pre-refactor
implementations pinned in `tests/_servo_id_pre_refactor.py`; and the noise floor com_id
consumes in both identifications is one function of the stream."""
from __future__ import annotations
import itertools
import numpy as np

from growbot_cerebellum.forward import Linear, K
from growbot_cerebellum.servo_id import (identify, realized_from_commands, confidence_band,
                                         determined_sets, _key)
import com_id
import _servo_id_pre_refactor as golden


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


def test_refactored_band_and_sets_reproduce_the_pre_refactor_golden():
    # confidence_band / determined_sets delegate to band_from_errors / within_band since the
    # com_id refactor; comparing them with themselves proves nothing, so the pin is the code
    # as it shipped BEFORE the refactor, on scores with a real tail (slow slews fit badly)
    for seed in (2, 3, 4):
        O, A, O2, D = _log(seed)
        m = _model(O, A, O2, D)
        h = len(O) // 2
        sA, _ = identify(m, O[:h], A[:h], O2[:h], D[:h], GRID)
        sB, _ = identify(m, O[h:], A[h:], O2[h:], D[h:], GRID)
        band = confidence_band(sA, sB)
        assert band == golden.confidence_band(sA, sB)
        assert band > 0
        best = sA[0][1]
        for b in (band, 3 * band, 0.0):
            assert determined_sets(sA, best, GRID, b) == golden.determined_sets(sA, best, GRID, b)


def test_golden_is_not_the_current_code():
    # the pin only guards if it can disagree: a deliberately wrong band must be caught
    O, A, O2, D = _log(5)
    m = _model(O, A, O2, D)
    h = len(O) // 2
    sA, _ = identify(m, O[:h], A[:h], O2[:h], D[:h], GRID)
    sB, _ = identify(m, O[h:], A[h:], O2[h:], D[h:], GRID)
    assert golden.confidence_band(sA, sB) != golden.confidence_band(sA, sA)
    wide = golden.determined_sets(sA, sA[0][1], GRID, 1e9)
    assert wide == (sorted({d for d, _, _ in GRID}), [2.0, None])


def test_noise_floor_is_one_function_of_the_stream():
    # identification A and B both call noise_floor() on the same (O, A, O2, D); the number
    # is |err(nominal) - err(second)| with R = A and no cut extension, whichever section asks
    O, A, O2, D = _log(6)
    nominal, second = _model(O, A, O2, D), _model(*_log(7))
    n = com_id.noise_floor(O, A, O2, D, nominal, second)
    sc = com_id.one_step_scorer(O, A, O2, D, 0)
    assert n == abs(sc(nominal, A) - sc(second, A)) and n > 0
    assert com_id.noise_floor(O, A, O2, D, nominal, second) == n
    assert com_id.noise_floor(O, A, O2, D, nominal, nominal) == 0.0
    # the exclusion predicate as stated: ratio >= 1, or a band of 0
    assert com_id.below_noise(n, 2 * n) == (0.5, False)
    assert com_id.below_noise(n, n) == (1.0, True)
    assert com_id.below_noise(n, 0.0) == (None, True)


def test_joint_verdict_marks_two_point_axes_as_boundary():
    # slew and deadband carry two values each, so no joint argmin is interior on them
    err = {k: 1.0 for k in com_id.joint_grid()}
    err[(0.0, 3, 2.0, 0.0)] = 0.0                     # interior on x and delay, boundary on the rest
    v = com_id.joint_verdict(err, 0.5, 0.0, dict(delay_ticks=3, slew_rad_s=2.0, deadband=0.0))
    assert v["argmin_interior_x"] and v["argmin_interior_delay"]
    assert v["argmin_interior"] is False and v["two_point_axes"] == ["slew", "deadband"]
    assert v["argmin_deadband_correct"] and v["deadband_determined_deg"] == [0.0]


def test_summarize_excludes_below_noise_seeds_and_needs_two_counted():
    s = com_id.summarize([True, False, True], counted=[True, False, True])
    assert s["resolved"] and s["value"] is True and s["n_counted"] == 2
    assert com_id.summarize([True, False], counted=[False, False])["resolved"] is False
    # one counted seed resolves nothing, whichever way it points
    one = com_id.summarize([True, True, True], counted=[False, True, False])
    assert one["n_counted"] == 1 and one["resolved"] is False and one["value"] is None
    assert com_id.summarize([False, False, False])["resolved"] and com_id.summarize([False, False, False])["value"] is False
