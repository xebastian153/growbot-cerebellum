"""Contact friction: the twin has no torsional friction, and the DR negative could not
have seen one.

This repo publishes that body-parameter domain randomisation is invisible in the IMU
(`sim2real_proxy.py`, 13 corners). Two holes in that negative, both on the spin axis:

  1. `perturb()` set `geom_friction[:, 0]` -- SLIDING friction. Torsional (column 1)
     and rolling (column 2) were never varied.
  2. `sim2real_proxy.horizon_within` scores `decode_obs(...)[:, :2]` -- roll and pitch.
     Yaw is not in the metric at all.

Sweeping columns 1 and 2 would have changed nothing regardless, because both bodies
ship with `condim="3"`. Under condim 3 MuJoCo builds a 3-dimensional contact -- one
normal plus two tangential (sliding) directions -- and the torsional and rolling
coefficients are simply not part of the solve. Torsional needs condim 4, rolling
needs condim 6. `condim_audit()` measures this rather than asserting it.

So the honest question is not "did we forget to sweep two columns" but "does the
mechanism the twin cannot represent matter". The corners below answer it by turning
the mechanism on and varying it, and by scoring yaw as well as roll and pitch.

Ranges are anchored, not invented: the XML declares `friction="1.2 0.1 0.1"`, and
MuJoCo's own defaults are `1 0.005 0.0001`. The sweep uses the XML value, the MuJoCo
default, and one decade above the XML value.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "sim"))
from growbot_sim import collect                                   # noqa: E402
from forward import MLP, make_windows, rollout_error              # noqa: E402
from sim2real_proxy import horizon_within, K                      # noqa: E402

HERE = Path(__file__).parent
XML_SLIDING, XML_TORSIONAL, XML_ROLLING = 1.2, 0.1, 0.1           # sim/growbot_*_body.xml
MJ_TORSIONAL, MJ_ROLLING = 0.005, 0.0001                          # MuJoCo defaults
NOOP = {"mass_scale": 1.0}    # truthy so GrowBotSim actually calls perturb()


def corners():
    """(label, dr, group). `group` says what the row is for."""
    def mk(name, group, **kw):
        return name, {**NOOP, **kw}, group
    return [
        mk("nominal (as shipped, condim 3)", "reference"),
        # the published negative's own friction corners, same protocol, for comparability
        mk(f"sliding {0.6} (published corner)", "sliding", friction=0.6),
        mk(f"sliding {1.4} (published corner)", "sliding", friction=1.4),
        # torsional inert at condim 3 -- included to show the null in the same table
        mk("torsional 1.0 at condim 3 (inert)", "inert", friction_torsional=1.0),
        mk("rolling 1.0 at condim 3 (inert)", "inert", friction_rolling=1.0),
        # mechanism ON: torsional only
        mk(f"condim 4, torsional {MJ_TORSIONAL} (MuJoCo default)", "torsional",
           condim=4, friction_torsional=MJ_TORSIONAL),
        mk(f"condim 4, torsional {XML_TORSIONAL} (XML value)", "torsional",
           condim=4, friction_torsional=XML_TORSIONAL),
        mk("condim 4, torsional 1.0 (decade above XML)", "torsional",
           condim=4, friction_torsional=1.0),
        # mechanism ON: torsional + rolling
        mk(f"condim 6, torsional {XML_TORSIONAL} rolling {XML_ROLLING} (XML values)", "rolling",
           condim=6, friction_torsional=XML_TORSIONAL, friction_rolling=XML_ROLLING),
        mk(f"condim 6, MuJoCo default contact ({MJ_TORSIONAL}/{MJ_ROLLING})", "rolling",
           condim=6, friction_torsional=MJ_TORSIONAL, friction_rolling=MJ_ROLLING),
    ]


def condim_audit(steps=6000, seed=0, body="olie"):
    """Measure whether torsional/rolling do anything, per condim. No model involved.

    A sliding-friction change is the positive control: if that moves the trajectory
    and a x100 torsional change does not, the coefficient is not in the solve.
    """
    base = collect(steps, seed=seed, body=body, dr=NOOP)[0]
    probes = [
        ("sliding 0.6 (positive control)", dict(friction=0.6)),
        ("torsional x10 at condim 3", dict(friction_torsional=1.0)),
        ("torsional x100 at condim 3", dict(friction_torsional=10.0)),
        ("rolling x100 at condim 3", dict(friction_rolling=10.0)),
        ("MuJoCo default contact at condim 3", dict(friction_torsional=MJ_TORSIONAL,
                                                    friction_rolling=MJ_ROLLING)),
        ("torsional x100 at condim 4", dict(condim=4, friction_torsional=10.0)),
        ("torsional MuJoCo default at condim 4", dict(condim=4, friction_torsional=MJ_TORSIONAL)),
        ("rolling x100 at condim 6", dict(condim=6, friction_rolling=10.0)),
    ]
    out = []
    for label, kw in probes:
        o = collect(steps, seed=seed, body=body, dr={**NOOP, **kw})[0]
        out.append({"probe": label, "max_abs_obs_delta": float(np.abs(o - base).max()),
                    "identical": bool(np.array_equal(o, base)), "kw": {k: v for k, v in kw.items()}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--corner-steps", type=int, default=30000, help="ticks per corner-seed (600 s)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--audit-steps", type=int, default=6000)
    ap.add_argument("--horizons", type=int, nargs="+", default=[5, 25])
    args = ap.parse_args()
    t_start = time.time()

    print("=" * 78)
    print("PART A -- does the coefficient act at all? (physics only, no model)")
    print("=" * 78)
    audit = condim_audit(steps=args.audit_steps)
    for r in audit:
        verdict = "NO EFFECT (bit-identical)" if r["identical"] else f"acts, max|d obs| {r['max_abs_obs_delta']:.3e}"
        print(f"  {r['probe']:<42} {verdict}")
    inert3 = [r for r in audit if "condim 3" in r["probe"] and "control" not in r["probe"]]
    print(f"\n  torsional/rolling probes at condim 3 that are bit-identical: "
          f"{sum(r['identical'] for r in inert3)}/{len(inert3)}")

    print("\n" + "=" * 78)
    print("PART B -- decision rule, stated before the numbers")
    print("=" * 78)
    print("  Protocol is sim2real_proxy's: the frozen nominal forward model, trained once on")
    print("  data/olie_train.npz, evaluated open-loop on each corner's own held-out stream.")
    print("  Two metrics per corner:")
    print("    within_0.2rad @100ms (roll/pitch only) -- the exact metric of the published")
    print("      negative, kept so the new corners and the old ones are comparable;")
    print("    within_0.2rad per axis @100/500ms -- adds YAW, which that metric never scored.")
    print("  Seeds are shared across every corner. Material = a shift from nominal larger")
    print("  than max(3.0 pts, 2x the nominal seed spread), the rule yaw_floor and real2sim use.")

    tr = np.load(HERE / "data" / "olie_train.npz")
    Xtr, Ytr, *_ = make_windows(tr["obs"], tr["act"], tr["next_obs"], tr["done"], K)
    print(f"\n  training the frozen nominal model ({args.epochs} epochs)...", flush=True)
    nominal = MLP(hidden=128, epochs=args.epochs).fit(Xtr, Ytr)

    rows = []
    for name, dr, group in corners():
        per_seed = []
        for sd in args.seeds:
            O, A, O2, D, _ = collect(args.corner_steps, seed=sd, body="olie", dr=dr)
            legacy = horizon_within(nominal, O, A, D, h=5, seed=0)[0]
            ro = rollout_error(nominal, O, A, D, K, args.horizons, seed=0)
            per_seed.append({
                "seed": sd, "n_ticks": int(len(O)),
                "legacy_within_0.2rad_rollpitch_100ms": legacy,
                "per_axis": {str(h): {"within_0.2rad_axis": ro[h]["within_0.2rad_axis"],
                                      "rmse_axis_rad": ro[h]["rmse_axis_rad"]} for h in args.horizons},
            })
        def agg(get):
            v = np.array([get(s) for s in per_seed], float)
            return {"mean": float(v.mean()), "spread": float(v.max() - v.min())}
        rows.append({
            "corner": name, "group": group, "dr": {k: v for k, v in dr.items() if k != "mass_scale"},
            "seeds": args.seeds, "per_seed": per_seed,
            "legacy_100ms": agg(lambda s: s["legacy_within_0.2rad_rollpitch_100ms"]),
            "axis": {str(h): {a: agg(lambda s, h=h, a=a: s["per_axis"][str(h)]["within_0.2rad_axis"][a])
                              for a in ("roll", "pitch", "yaw")} for h in args.horizons},
        })
        print(f"  collected+scored: {name}", flush=True)

    nom = rows[0]
    spread = max(nom["legacy_100ms"]["spread"],
                 max(nom["axis"][str(h)][a]["spread"] for h in args.horizons
                     for a in ("roll", "pitch", "yaw")))
    thresh = max(0.03, 2.0 * spread)
    print(f"\n  nominal seed spread (worst metric, {len(args.seeds)} seeds): "
          f"{spread * 100:.2f} pts -> material threshold {thresh * 100:.2f} pts")

    print("\n" + "=" * 78)
    print("PART B -- results (within 0.2 rad, % of starts; delta vs nominal in pts)")
    print("=" * 78)
    hdr = f"{'corner':<52}{'legacy':>9}" + "".join(
        f"{ax[:4] + '@' + str(h):>11}" for h in args.horizons for ax in ("roll", "pitch", "yaw"))
    print(hdr); print("-" * len(hdr))
    for r in rows:
        line = f"{r['corner']:<52}{r['legacy_100ms']['mean'] * 100:>8.1f}%"
        for h in args.horizons:
            for ax in ("roll", "pitch", "yaw"):
                line += f"{r['axis'][str(h)][ax]['mean'] * 100:>10.1f}%"
        print(line)
    print("\ndelta vs nominal (pts), material marked *")
    print(hdr); print("-" * len(hdr))
    verdicts = {}
    for r in rows[1:]:
        line = f"{r['corner']:<52}"
        d = r["legacy_100ms"]["mean"] - nom["legacy_100ms"]["mean"]
        line += f"{d * 100:>+8.1f}{'*' if abs(d) > thresh else ' '}"
        mats = {}
        for h in args.horizons:
            for ax in ("roll", "pitch", "yaw"):
                dd = r["axis"][str(h)][ax]["mean"] - nom["axis"][str(h)][ax]["mean"]
                mats[f"{ax}@{h}"] = {"delta_pts": float(dd * 100), "material": bool(abs(dd) > thresh)}
                line += f"{dd * 100:>+10.1f}{'*' if abs(dd) > thresh else ' '}"
        print(line)
        verdicts[r["corner"]] = {"legacy_delta_pts": float(d * 100),
                                 "legacy_material": bool(abs(d) > thresh), "axis": mats}

    def any_material(group, axis_prefix=None):
        out = []
        for r in rows[1:]:
            if r["group"] != group:
                continue
            v = verdicts[r["corner"]]
            hit = [k for k, m in v["axis"].items() if m["material"] and
                   (axis_prefix is None or k.startswith(axis_prefix))]
            if hit or (axis_prefix is None and v["legacy_material"]):
                out.append((r["corner"], hit))
        return out

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    inert_hits = any_material("inert")
    tor_hits = any_material("torsional")
    tor_yaw = any_material("torsional", "yaw")
    roll_hits = any_material("rolling")
    print(f"  torsional/rolling AT condim 3 (as shipped): "
          f"{'moves the IMU' if inert_hits else 'no material effect on any axis -- inert, as Part A predicts'}")
    print(f"  torsional with the mechanism ON (condim 4): "
          f"{'moves the IMU on ' + ', '.join(sorted({k for _, h in tor_hits for k in h})) if tor_hits else 'no material effect'}")
    print(f"    of which YAW specifically: {'YES -- ' + ', '.join(c for c, _ in tor_yaw) if tor_yaw else 'no'}")
    print(f"  torsional+rolling (condim 6): "
          f"{'moves the IMU on ' + ', '.join(sorted({k for _, h in roll_hits for k in h})) if roll_hits else 'no material effect'}")

    out = {"config": vars(args),
           "xml_friction": {"sliding": XML_SLIDING, "torsional": XML_TORSIONAL, "rolling": XML_ROLLING},
           "mujoco_default_friction": {"sliding": 1.0, "torsional": MJ_TORSIONAL, "rolling": MJ_ROLLING},
           "shipped_condim": 3,
           "decision_rule": {"nominal_seed_spread": float(spread),
                             "threshold": float(thresh),
                             "text": "material = |delta vs nominal| > max(3.0 pts, 2x nominal seed spread)"},
           "part_a_condim_audit": audit,
           "rows": rows, "verdicts": verdicts,
           "runtime_s": float(time.time() - t_start)}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "contact_friction.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote results/contact_friction.json   total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
