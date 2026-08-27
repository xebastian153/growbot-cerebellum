"""evaluate_axes never scores a window that crosses a cut.

Each episode is a distinct constant; a persistence model imagines the start forever, so
the truth at start+h equals the imagined value iff no cut lies in [start, start+h]. A
window that crossed a cut would read a different constant and score below 1.0."""
from __future__ import annotations
import numpy as np

from growbot_cerebellum.forward import Persistence, K
from growbot_cerebellum.gap import evaluate_axes, twin_regimes, REGIME_MAP


def test_evaluate_axes_never_crosses_a_cut():
    n_ep, ep_len = 12, 80
    O = np.zeros((n_ep * ep_len, 6), np.float32)
    D = np.zeros(n_ep * ep_len, bool)
    for e in range(n_ep):
        O[e * ep_len:(e + 1) * ep_len, :3] = 0.5 * (e - n_ep / 2) / n_ep     # one attitude per episode
        D[(e + 1) * ep_len - 1] = True
    A = np.zeros((len(O), 2), np.float32)
    mode = np.array(["policy"] * len(O))
    horizons = (5, 25)
    out = evaluate_axes(Persistence(), O, A, D, mode, horizons, n_starts=10_000, seed=0)
    assert set(out) == {"all", "policy"}
    for reg in out:
        for h in horizons:
            for ax in ("roll", "pitch", "yaw"):
                assert out[reg][h][ax]["within"] == 1.0, (reg, h, ax)
                assert out[reg][h][ax]["rmse"] == 0.0
    # every eligible start was used: (ep_len - K - Hmax) per episode, minus the one start the
    # stream's own end-guard (ok[N - Hmax - 1:] = False) removes from the last episode
    assert out["all"]["n"] == n_ep * (ep_len - K - max(horizons)) - 1


def test_regime_map_only_borrows_floors_it_can_defend():
    assert REGIME_MAP["walking"] == "policy"
    assert "impact" not in REGIME_MAP and "unknown" not in REGIME_MAP
    obs = np.zeros((300, 6), np.float32)
    mode = np.array(["still"] * 300)
    m, rest = twin_regimes(obs, mode)
    assert list(m) == ["still"] * 300
