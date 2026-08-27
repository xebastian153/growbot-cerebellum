"""Smoke tests for growbot_cerebellum.forward: window validity across cuts, rollout shape
and determinism. No simulator, no training -- seconds."""
from __future__ import annotations
import numpy as np

from growbot_cerebellum.forward import make_windows, rollout_error, encode_obs, decode_obs, Persistence, K


def _stream(n, seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(n, 6)).astype(np.float32) * 0.3
    act = rng.uniform(-1, 1, size=(n, 2)).astype(np.float32)
    next_obs = np.roll(obs, -1, axis=0)
    return obs, act, next_obs


def test_make_windows_never_crosses_a_cut():
    # D[i] marks the transition i -> i+1 as a cut (AGENTS.md): a valid row needs its own
    # transition intact AND no cut inside its K-step history.
    n = 200
    obs, act, next_obs = _stream(n)
    done = np.zeros(n, bool)
    done[[30, 31, 100, 150]] = True
    X, Y, F, F2, valid = make_windows(obs, act, next_obs, done, K)
    assert X.shape == (valid.sum(), K * 11) and Y.shape == (valid.sum(), 9)
    assert not (valid & done).any(), "a window whose target crosses a cut is valid"
    for t in np.flatnonzero(valid):
        assert t >= K - 1
        assert not done[max(0, t - K + 1):t].any(), f"row {t} has a cut inside its history"
    # every cut costs windows: the K rows after a cut are invalid, and the cut row itself
    for c in (30, 100, 150):
        assert not valid[c:c + K].any()
    assert valid[:K - 1].sum() == 0


def test_encode_decode_roundtrip():
    obs, *_ = _stream(50)
    back = decode_obs(encode_obs(obs))
    assert np.allclose(back, obs, atol=1e-6)


def test_rollout_error_shape_and_determinism():
    n = 2000
    obs, act, next_obs = _stream(n, seed=1)
    done = np.zeros(n, bool); done[::400] = True
    horizons = (1, 5, 25)
    a = rollout_error(Persistence(), obs, act, done, K, horizons, n_starts=200, seed=7)
    b = rollout_error(Persistence(), obs, act, done, K, horizons, n_starts=200, seed=7)
    assert set(a) == set(horizons)
    for h in horizons:
        r = a[h]
        assert r["n_starts"] == 200
        assert set(r["within_0.2rad_axis"]) == {"roll", "pitch", "yaw"}
        assert 0.0 <= r["within_0.2rad"] <= 1.0
        assert a[h] == b[h], "same seed, different rollout"
    c = rollout_error(Persistence(), obs, act, done, K, horizons, n_starts=200, seed=8)
    assert c[25] != a[25], "different seeds picked the same starts"


def test_rollout_error_respects_start_mask():
    n = 1000
    obs, act, next_obs = _stream(n, seed=2)
    done = np.zeros(n, bool)
    mask = np.zeros(n, bool); mask[100:140] = True
    r = rollout_error(Persistence(), obs, act, done, K, (5,), n_starts=500, seed=0, start_mask=mask)
    assert r[5]["n_starts"] == 40
    empty = rollout_error(Persistence(), obs, act, done, K, (5,), start_mask=np.zeros(n, bool))
    assert empty == {5: None}
