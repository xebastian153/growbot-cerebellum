"""The honesty machinery every corner experiment starts from.

Nothing here is published as a bare mean. `seed_stat` carries the seeds and their spread;
`score_corners` collects and scores corners with the shared evaluation primitives;
`decide` and `decide_per_metric` turn a table into verdicts whose bars come from the
nominal spread of the SAME quantity, and whose `resolved` flag says whether the seeds
separate at all. AGENTS.md, "Publishing a number", is the reason each of these exists.
"""
from __future__ import annotations
import numpy as np
from .sim import collect
from .forward import rollout_error, K, AXES
from .sim2real import horizon_within


def seed_stat(vals, scale=1.0):
    """mean, the three seeds themselves, and their spread. Nothing in the partition is
    published as a bare mean, for the same reason no axis verdict is."""
    v = [float(x) * scale for x in vals]
    return {"mean": float(np.mean(v)), "per_seed": v, "spread": float(max(v) - min(v))}


def score_corners(nominal, corner_list, seeds, corner_steps, horizons, body="olie", extra=None):
    """One row per corner: the published legacy metric plus per-axis within-0.2rad and
    per-axis RMSE at each horizon, seeds shared across corners. The evaluation primitives
    are sim2real_proxy's and forward's; nothing here re-implements them.

    Seeds are shared in the sense that corner k and corner l are collected from the same
    seed, so they start from the same initial condition -- NOT in the sense of common
    random numbers. `collect()` consumes `sim.rng` on a schedule the corner's own behaviour
    sets: one draw every tick for the push test (`sim.rng.random() < push_prob`, taken
    whether or not a push happens), mode-dependent draws inside `Excitation`, one draw per
    episode in `fresh()`, and a further draw only when the body is already fallen. Two
    corners therefore diverge at the first tick where their dynamics differ and never
    realign -- the desynchronisation does not need a fall to start it. Paired-difference
    statistics do not apply; the per-corner seed spread reported beside every verdict is
    the honest uncertainty.

    `extra(nominal, O, A, O2, D, M) -> dict` (default None) is merged into that seed's
    record. It exists because the mode channel `collect()` returns is what a fall-rate /
    regime / oracle partition needs, and this function used to drop it on the floor.
    """
    rows = []
    for name, dr, group in corner_list:
        per_seed = []
        for sd in seeds:
            O, A, O2, D, M = collect(corner_steps, seed=sd, body=body, dr=dr)
            legacy = horizon_within(nominal, O, A, D, h=5, seed=0)[0]
            ro = rollout_error(nominal, O, A, D, K, horizons, seed=0)
            rec = {
                "seed": sd, "n_ticks": int(len(O)),
                "legacy_within_0.2rad_rollpitch_100ms": legacy,
                "per_axis": {str(h): {"within_0.2rad_axis": ro[h]["within_0.2rad_axis"],
                                      "rmse_axis_rad": ro[h]["rmse_axis_rad"]} for h in horizons},
            }
            if extra is not None:
                rec.update(extra(nominal, O, A, O2, D, M))
            per_seed.append(rec)
        def agg(get):
            v = np.array([get(s) for s in per_seed], float)
            return {"mean": float(v.mean()), "spread": float(v.max() - v.min()),
                    "per_seed": [float(x) for x in v]}
        rows.append({
            "corner": name, "group": group, "dr": {k: v for k, v in dr.items() if k != "mass_scale"} if dr.get("mass_scale", 1.0) == 1.0 else dict(dr),
            "seeds": list(seeds), "per_seed": per_seed,
            "legacy_100ms": agg(lambda s: s["legacy_within_0.2rad_rollpitch_100ms"]),
            "axis": {str(h): {a: agg(lambda s, h=h, a=a: s["per_axis"][str(h)]["within_0.2rad_axis"][a])
                              for a in ("roll", "pitch", "yaw")} for h in horizons},
            "rmse": {str(h): {a: agg(lambda s, h=h, a=a: s["per_axis"][str(h)]["rmse_axis_rad"][a])
                              for a in ("roll", "pitch", "yaw")} for h in horizons},
        })
        print(f"  collected+scored: {name}", flush=True)
    return rows


def decide(rows, horizons, seeds):
    """Threshold from the nominal row's seed spread; verdict per corner, axis and horizon."""
    nom = rows[0]
    spread = max(nom["legacy_100ms"]["spread"],
                 max(nom["axis"][str(h)][a]["spread"] for h in horizons
                     for a in ("roll", "pitch", "yaw")))
    thresh = max(0.03, 2.0 * spread)
    print(f"\n  nominal seed spread (worst metric, {len(seeds)} seeds): "
          f"{spread * 100:.2f} pts -> material threshold {thresh * 100:.2f} pts")
    verdicts = {}
    for r in rows[1:]:
        d = r["legacy_100ms"]["mean"] - nom["legacy_100ms"]["mean"]
        mats = {}
        for h in horizons:
            for ax in ("roll", "pitch", "yaw"):
                dd = r["axis"][str(h)][ax]["mean"] - nom["axis"][str(h)][ax]["mean"]
                mats[f"{ax}@{h}"] = {"delta_pts": float(dd * 100), "material": bool(abs(dd) > thresh)}
        verdicts[r["corner"]] = {"legacy_delta_pts": float(d * 100),
                                 "legacy_material": bool(abs(d) > thresh), "axis": mats}
    return spread, thresh, verdicts


# ----------------------------------------------------------------------
# decision rule
# ----------------------------------------------------------------------
def metric_keys(horizons):
    return ["legacy@5"] + [f"{a}@{h}" for h in horizons for a in AXES]


def nominal_spread(nom, key):
    if key == "legacy@5":
        return nom["legacy_100ms"]["spread"]
    ax, h = key.split("@")
    return nom["axis"][h][ax]["spread"]


def corner_value(r, key):
    if key == "legacy@5":
        return r["legacy_100ms"]
    ax, h = key.split("@")
    return r["axis"][h][ax]


def seed_separation(corner_seeds, nominal_seeds):
    """Signed gap between the two closest seeds of the corner and of nominal, in points.

    The seeds are NOT paired: `collect()` draws `sim.rng.random()` every tick for the push
    test, mode-dependently inside `Excitation` and once per episode in `fresh()`, so two
    corners diverge at the first tick their dynamics differ and never realign. The honest
    question a 3-seed run can answer is whether the two sets of seeds separate at all.
    0.0 means they overlap.
    """
    c, n = np.asarray(corner_seeds, float), np.asarray(nominal_seeds, float)
    if c.max() < n.min():
        return float((c.max() - n.min()) * 100)
    if c.min() > n.max():
        return float((c.min() - n.max()) * 100)
    return 0.0


def decide_per_metric(rows, horizons, floor=0.03):
    """Threshold per metric and horizon, from the nominal seed spread OF THAT METRIC.

    The earlier rule took the worst nominal spread across every metric (yaw at 100 ms,
    2.30 pts) and applied the resulting 4.60-pt bar to all of them, which makes a null on
    a quiet metric far too easy to declare. Each metric now carries its own bar,
    max(3.0 pts, 2x that metric's nominal spread).

    Every verdict is published beside two things the mean alone hides:
      spread    the seed spread of the corner it was measured on. A corner whose own
                spread reaches its threshold is decided at a precision worse than the bar
                deciding it.
      resolved  whether the corner's three seeds separate from nominal's three by more
                than that bar. This is the claim that survives an unpaired 3-seed run: a
                mean beyond the bar with overlapping seed ranges is not resolved, and a
                wide spread whose every seed still clears the bar on the same side is.
    An axis is reported as moved only when it is BOTH material and resolved.
    """
    nom = rows[0]
    keys = metric_keys(horizons)
    thresh = {k: max(floor, 2.0 * nominal_spread(nom, k)) for k in keys}
    verdicts = {}
    for r in rows[1:]:
        axis, legacy = {}, None
        for k in keys:
            c, n = corner_value(r, k), corner_value(nom, k)
            dd = c["mean"] - n["mean"]
            sep = seed_separation(c["per_seed"], n["per_seed"])
            rec = {"delta_pts": float(dd * 100), "threshold_pts": float(thresh[k] * 100),
                   "corner_spread_pts": float(c["spread"] * 100),
                   "per_seed_delta_pts": [float((v - n["mean"]) * 100) for v in c["per_seed"]],
                   "material": bool(abs(dd) > thresh[k]),
                   "spread_reaches_threshold": bool(c["spread"] >= thresh[k]),
                   "seed_separation_pts": sep,
                   "resolved": bool(abs(sep) > thresh[k] * 100)}
            rec["reported_as_moved"] = bool(rec["material"] and rec["resolved"])
            if k == "legacy@5":
                legacy = rec
            else:
                axis[k] = rec
        verdicts[r["corner"]] = {"legacy_delta_pts": legacy["delta_pts"],
                                 "legacy_material": legacy["material"],
                                 "legacy_resolved": legacy["resolved"],
                                 "legacy_spread_reaches_threshold": legacy["spread_reaches_threshold"],
                                 "legacy_corner_spread_pts": legacy["corner_spread_pts"],
                                 "axis": axis}
    return thresh, verdicts
