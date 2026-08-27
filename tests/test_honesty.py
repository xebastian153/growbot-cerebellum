"""The honesty helpers on hand-built cases: means never travel alone, and a verdict is
material by the mean AND resolved only when the seeds separate."""
from __future__ import annotations
import numpy as np

from growbot_cerebellum.honesty import seed_stat, seed_separation, decide_per_metric, metric_keys


def test_seed_stat_carries_the_seeds():
    s = seed_stat([0.80, 0.82, 0.81], scale=100.0)
    assert s == {"mean": 81.0, "per_seed": [80.0, 82.0, 81.0], "spread": 2.0}


def test_seed_separation_is_the_gap_between_closest_seeds():
    assert np.isclose(seed_separation([0.70, 0.71, 0.72], [0.80, 0.81, 0.82]), -8.0)
    assert np.isclose(seed_separation([0.90, 0.91, 0.92], [0.80, 0.81, 0.82]), 8.0)
    assert seed_separation([0.75, 0.81, 0.90], [0.80, 0.81, 0.82]) == 0.0


def _row(name, seeds):
    st = lambda: {"mean": float(np.mean(seeds)), "per_seed": list(seeds), "spread": float(max(seeds) - min(seeds))}
    return {"corner": name, "legacy_100ms": st(),
            "axis": {"5": {a: st() for a in ("roll", "pitch", "yaw")}}}


def test_decide_per_metric_material_and_resolved_are_separate_marks():
    rows = [_row("nominal", [0.80, 0.81, 0.82]),
            _row("moved", [0.70, 0.71, 0.72]),           # -10 pts, seeds separate by 8
            _row("flat", [0.76, 0.80, 0.84]),            # mean unchanged, wide spread
            _row("material only", [0.72, 0.74, 0.83])]   # -4.7 pts past the 4-pt bar, but one seed overlaps nominal
    thresh, v = decide_per_metric(rows, horizons=[5], floor=0.03)
    assert set(thresh) == set(metric_keys([5]))
    assert all(np.isclose(t, 0.04) for t in thresh.values())      # 2 x the 2-pt nominal spread beats the 3-pt floor
    moved = v["moved"]["axis"]["roll@5"]
    assert moved["material"] and moved["resolved"] and moved["reported_as_moved"]
    assert np.isclose(moved["delta_pts"], -10.0) and np.isclose(moved["seed_separation_pts"], -8.0)
    flat = v["flat"]["axis"]["roll@5"]
    assert not flat["material"] and not flat["resolved"] and not flat["reported_as_moved"]
    assert flat["spread_reaches_threshold"]
    only = v["material only"]["axis"]["roll@5"]
    assert only["material"] and not only["resolved"] and not only["reported_as_moved"]
    assert only["per_seed_delta_pts"] == [float((s - 0.81) * 100) for s in (0.72, 0.74, 0.83)]
