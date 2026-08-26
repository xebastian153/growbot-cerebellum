"""Body parameters at the horizons the published negative never scored, with the
centre-of-mass result partitioned into stream difficulty and model mismatch.

This repo publishes that body-parameter domain randomisation is invisible in the IMU
(`sim2real_proxy.py`, 13 corners) -- but that negative was scored with
`horizon_within(..., h=5)`: 100 ms only, roll and pitch only. `contact_friction.py`
closed the same limit for friction by scoring at 100 AND 500 ms with yaw. Mass, centre
of mass and leg length were never scored there. This script does that, on the identical
protocol, and adds five one-factor corners: +-75 g on the base body alone and the three
published CoM endpoints one axis at a time.

What the +-75 g corners are, and are not. The twin carries NO phone: its 427 g base is
battery plus structure (`sim/growbot_olie_body.xml`, lines 16-19). `base_mass_delta`
adds or removes mass at the base body's EXISTING centre of mass and leaves `body_ipos`
untouched, so those two corners isolate mass at a fixed CoM -- they are not a phone
swap, and no corner in this file mounts a phone. A 200 g phone on this body is +200 g on
480 g, a whole-body mass_scale of ~1.42, outside every corner tested here and outside
the published DR range (0.80-1.25). The CoM corners are the published DR endpoints,
anchored to the DR ranges and nothing else.

Why the CoM result needs a partition. The nominal whole-body CoM sits BEHIND the foot
support box; +3 cm moves it INSIDE. That is a change of balance regime, not only a
change of body parameter, so a drop in prediction accuracy at that corner can be the
stream getting genuinely harder rather than the frozen model being wrong about a new
body. Every corner therefore reports its fall rate, its regime mix, its per-regime
accuracy, and an oracle (a model trained on that corner's own data) at 500 ms, which
splits the drop exactly:

    frozen_c - frozen_nom  =  (oracle_c - oracle_nom)  +  [mismatch_c - mismatch_nom]
                              ^ intrinsic difficulty     ^ what training on the body fixes

Two limits, stated up front and repeated in the write-up:
  - this is forward-model PREDICTION accuracy, not policy TRANSFER;
  - every real-log number in this repo comes from ONE unit and ONE phone. This sweep
    says what the twin predicts across units, not what a second real robot does.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "sim"))
from growbot_sim import DR, GrowBotSim                            # noqa: E402
from forward import MLP, make_windows, rollout_error              # noqa: E402
from sim2real_proxy import K, corners as published_corners        # noqa: E402
from contact_friction import score_corners, decide                # noqa: E402

HERE = Path(__file__).parent
BASE_DELTA_KG = 0.075      # +-75 g added at the base's existing CoM: a mass isolation
PHONE_KG = 0.200           # a real phone, for the anchor arithmetic only -- never a corner
FALL_RAD = 1.2             # GrowBotSim.fallen(): |roll| > 1.2 or |pitch| > 1.2
FAST_RADS = 3.0            # forward.by_regime's own "fast" cut
MIN_REGIME_TICKS = 50      # forward.by_regime's own floor
AXES = ("roll", "pitch", "yaw")


def corners():
    """(label, dr, group). The 13 published corners come from sim2real_proxy's own
    constructor so the old table and this one are the same points; five are added."""
    lo = {k: v[0] for k, v in DR.items()}; hi = {k: v[1] for k, v in DR.items()}
    out = []
    for name, dr in published_corners():
        if name == "nominal":
            group = "reference"
        elif name.startswith(("mass", "com", "leg")):
            group = "body"
        elif name.startswith("gain"):
            group = "gain"
        elif name.startswith("friction"):
            group = "sliding"
        else:
            group = "worst"
        out.append((name if name != "nominal" else "nominal (published)", dr, group))
    base = dict(mass_scale=1.0)
    out += [
        ("base -75 g at a fixed CoM", {**base, "base_mass_delta": -BASE_DELTA_KG}, "mass_fixed_com"),
        ("base +75 g at a fixed CoM", {**base, "base_mass_delta": +BASE_DELTA_KG}, "mass_fixed_com"),
        ("com x -0.03 only (DR endpoint, back)", {**base, "dcom": (lo["dcom_x"], 0.0, 0.0)}, "body"),
        ("com x +0.03 only (DR endpoint, forward)", {**base, "dcom": (hi["dcom_x"], 0.0, 0.0)}, "body"),
        ("com z +0.015 only (DR endpoint, higher)", {**base, "dcom": (0.0, 0.0, hi["dcom_z"])}, "body"),
    ]
    return out


# ----------------------------------------------------------------------
# geometry: where the centre of mass sits relative to the feet
# ----------------------------------------------------------------------
def balance_geometry(body="olie"):
    """Whole-body CoM along the body axis vs the foot support box, measured from the model.

    The support box is the x-extent of the two leg geoms (they sit at x = 0), i.e. what
    the body can put weight on without tipping. A CoM outside it is held up by contact
    torque; a CoM inside it is statically supported. Crossing that line is a change of
    balance regime, which is why the CoM corners cannot be read as model error alone.
    """
    def com_x(dr):
        s = GrowBotSim(seed=0, body=body, dr=dr)
        mujoco.mj_forward(s.m, s.d)
        base = mujoco.mj_name2id(s.m, mujoco.mjtObj.mjOBJ_BODY, "base_body")
        com = (s.d.xipos * s.m.body_mass[:, None]).sum(0) / s.m.body_mass.sum()
        return float(s.m.body_ipos[base, 0]), float(com[0] - s.d.xpos[base, 0])

    m = GrowBotSim(seed=0, body=body).m
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "lower_leg_1")
    half_x = float(m.geom_size[gid, 0])
    torso_half_x = float(m.geom_size[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "torso_geom"), 0])
    out = {"support_box_half_x_m": half_x, "torso_half_length_m": torso_half_x, "corners": {}}
    for label, dcom in (("nominal", 0.0), ("com x -0.03", DR["dcom_x"][0]), ("com x +0.03", DR["dcom_x"][1])):
        ipos_x, com_rel = com_x({"mass_scale": 1.0, "dcom": (dcom, 0.0, 0.0)})
        out["corners"][label] = {"base_ipos_x_m": ipos_x, "whole_body_com_x_m": com_rel,
                                 "inside_support_box": bool(abs(com_rel) <= half_x),
                                 "margin_outside_box_mm": float((abs(com_rel) - half_x) * 1000)}
    return out


def base_mass_kg(body="olie"):
    m = GrowBotSim(seed=0, body=body).m
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_body")
    return (float(m.body_mass[base]), float(m.body_mass.sum()),
            float(m.body_ipos[base, 0]), float(m.body_ipos[base, 2]))


# ----------------------------------------------------------------------
# the partition: fall rate, regime mix, per-regime accuracy, oracle
# ----------------------------------------------------------------------
def regime_masks(O, M):
    """forward.by_regime's own six buckets, applied to a corner's stream."""
    fallen = (np.abs(O[:, 0]) > FALL_RAD) | (np.abs(O[:, 1]) > FALL_RAD)
    fast = np.linalg.norm(O[:, 3:], axis=1) > FAST_RADS
    return fallen, {
        "policy walking": (M == "policy") & ~fallen & ~fast,
        "sine gait": (M == "sine") & ~fallen & ~fast,
        "keyframe/OU": np.isin(M, ["keyframe", "ou"]) & ~fallen & ~fast,
        "still": (M == "still") & ~fallen & ~fast,
        "fast (|gyro|>3)": fast & ~fallen,
        "fallen": fallen,
    }


def make_partition(h, oracle_epochs):
    """Build the per-seed hook `score_corners` calls with the mode channel it used to drop.

    h is the horizon (ticks) the partition is measured at -- 25 = 500 ms, where the CoM
    result lives. The oracle is sim2real_proxy's construction: a model trained on the
    corner's OWN first half, with frozen and oracle both scored on the held-out second
    half so the two are comparable.
    """
    def partition(nominal, O, A, O2, D, M, dr, body, horizons, seed):
        fallen, masks = regime_masks(O, M)
        per_regime = {}
        for name, sel in masks.items():
            if sel.sum() < MIN_REGIME_TICKS:
                continue
            r = rollout_error(nominal, O, A, D, K, [h], n_starts=1500, seed=0, start_mask=sel)[h]
            if r is None:
                continue
            per_regime[name] = {"n_starts": r["n_starts"], "within_0.2rad_axis": r["within_0.2rad_axis"]}

        cut = len(O) // 2
        held = slice(cut, None)
        Xc, Yc, *_ = make_windows(O[:cut], A[:cut], O2[:cut], D[:cut], K)
        oracle_m = MLP(hidden=128, epochs=oracle_epochs).fit(Xc, Yc)
        froz = rollout_error(nominal, O[held], A[held], D[held], K, [h], seed=0)[h]
        orac = rollout_error(oracle_m, O[held], A[held], D[held], K, [h], seed=0)[h]
        return {
            "fall_rate": float(fallen.mean()),
            "regime_mix": {k: float(v.mean()) for k, v in masks.items()},
            "regime_within_0.2rad_axis": per_regime,
            "held_out": {"n_train_windows": int(len(Xc)), "n_starts": froz["n_starts"],
                         "frozen_within_0.2rad_axis": froz["within_0.2rad_axis"],
                         "oracle_within_0.2rad_axis": orac["within_0.2rad_axis"]},
        }
    return partition


def partition_table(rows, h):
    """Per-corner means over seeds, and the exact split of each corner's frozen drop.

        frozen_c - frozen_n = (oracle_c - oracle_n) + [(frozen_c - oracle_c) - (frozen_n - oracle_n)]
                               intrinsic difficulty    model mismatch the corner adds
    """
    def mean_over_seeds(r, get):
        return float(np.mean([get(s) for s in r["per_seed"]]))

    nom = rows[0]
    out = {}
    for r in rows:
        froz = {a: mean_over_seeds(r, lambda s, a=a: s["held_out"]["frozen_within_0.2rad_axis"][a]) for a in AXES}
        orac = {a: mean_over_seeds(r, lambda s, a=a: s["held_out"]["oracle_within_0.2rad_axis"][a]) for a in AXES}
        reg = {}
        for name in rows[0]["per_seed"][0]["regime_within_0.2rad_axis"]:
            vals = [s["regime_within_0.2rad_axis"][name] for s in r["per_seed"]
                    if name in s["regime_within_0.2rad_axis"]]
            if not vals:
                continue
            reg[name] = {"seeds_present": len(vals),
                         "n_starts_mean": float(np.mean([v["n_starts"] for v in vals])),
                         **{a: float(np.mean([v["within_0.2rad_axis"][a] for v in vals])) for a in AXES}}
        out[r["corner"]] = {
            "fall_rate": mean_over_seeds(r, lambda s: s["fall_rate"]),
            "regime_mix": {k: mean_over_seeds(r, lambda s, k=k: s["regime_mix"][k])
                           for k in r["per_seed"][0]["regime_mix"]},
            "regime_within_0.2rad_axis": reg,
            "held_out_frozen": froz, "held_out_oracle": orac,
        }
    nomp = out[nom["corner"]]
    for name, p in out.items():
        p["split_vs_nominal_pts"] = {}
        for a in AXES:
            drop = (p["held_out_frozen"][a] - nomp["held_out_frozen"][a]) * 100
            intrinsic = (p["held_out_oracle"][a] - nomp["held_out_oracle"][a]) * 100
            mismatch = drop - intrinsic
            p["split_vs_nominal_pts"][a] = {
                "frozen_drop": drop, "intrinsic": intrinsic, "model_mismatch": mismatch,
                "intrinsic_share": (float(intrinsic / drop) if abs(drop) > 1e-9 else None)}
    return out


# ----------------------------------------------------------------------
# decision rule
# ----------------------------------------------------------------------
def metric_keys(horizons):
    return ["legacy@5"] + [f"{a}@{h}" for h in horizons for a in AXES]


def nominal_spread(nom, key, horizons):
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

    The seeds are NOT paired -- `collect()` consumes the RNG only when the body has
    fallen, so the streams desynchronise -- so the honest question a 3-seed run can
    answer is whether the two sets of seeds separate at all. 0.0 means they overlap.
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
    thresh = {k: max(floor, 2.0 * nominal_spread(nom, k, horizons)) for k in keys}
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


def print_threshold_block(thresh, rows, horizons, legacy_thresh):
    nom = rows[0]
    print(f"\n  per-metric thresholds (max(3.0 pts, 2x the nominal spread of that metric)),")
    print(f"  against the single worst-case bar this section used before, {legacy_thresh * 100:.2f} pts:")
    print(f"    {'metric':<12}{'nominal spread':>16}{'threshold':>12}{'was':>8}")
    for k in metric_keys(horizons):
        print(f"    {k:<12}{nominal_spread(nom, k, horizons) * 100:>15.2f}p{thresh[k] * 100:>11.2f}p"
              f"{legacy_thresh * 100:>7.2f}p")


def print_result_tables(rows, verdicts, horizons):
    print("\n" + "=" * 100)
    print("RESULTS (within 0.2 rad, % of starts)")
    print("=" * 100)
    hdr = f"{'corner':<42}{'legacy':>9}" + "".join(
        f"{ax[:4] + '@' + str(h):>11}" for h in horizons for ax in AXES)
    print(hdr); print("-" * len(hdr))
    for r in rows:
        line = f"{r['corner']:<42}{r['legacy_100ms']['mean'] * 100:>8.1f}%"
        for h in horizons:
            for ax in AXES:
                line += f"{r['axis'][str(h)][ax]['mean'] * 100:>10.1f}%"
        print(line)

    print("\ndelta vs nominal (pts) +- this corner's own seed spread, then two marks: material, resolved")
    hdr2 = f"{'corner':<42}{'legacy':>16}" + "".join(f"{ax[:4] + '@' + str(h):>18}" for h in horizons for ax in AXES)
    print(hdr2); print("-" * len(hdr2))
    def marks(m):
        return ("*" if m["material"] else " ") + ("R" if m["resolved"] else "?")
    for r in rows[1:]:
        v = verdicts[r["corner"]]
        line = (f"{r['corner']:<42}{v['legacy_delta_pts']:>+9.1f}+-{v['legacy_corner_spread_pts']:<3.1f}"
                + ("*" if v["legacy_material"] else " ") + ("R" if v["legacy_resolved"] else "?"))
        for h in horizons:
            for ax in AXES:
                m = v["axis"][f"{ax}@{h}"]
                line += f"{m['delta_pts']:>+11.1f}+-{m['corner_spread_pts']:<4.1f}{marks(m)}"
        print(line)
    print("  * = |mean delta| > that metric's threshold.")
    print("  R = this corner's three seeds separate from nominal's three by more than that")
    print("      threshold;  ? = they do not, so three seeds do not resolve the row -- whichever")
    print("      side of the bar the mean sits on. An axis is reported as moved only when both.")
    flagged = [(r["corner"], k) for r in rows[1:] for k, m in verdicts[r["corner"]]["axis"].items()
               if m["spread_reaches_threshold"]]
    print(f"  rows whose own seed spread reaches their threshold: {len(flagged)} of "
          f"{sum(len(verdicts[r['corner']]['axis']) for r in rows[1:])}")


def print_rmse_table(rows, h, groups=("body", "mass_fixed_com")):
    """within-0.2rad is a threshold crossing; RMSE is the error magnitude behind it. A
    corner that moves one and not the other has not moved the axis."""
    nom = rows[0]
    print(f"\nwithin-0.2rad vs RMSE at {h * 20} ms, delta vs nominal (nominal RMSE "
          + ", ".join(f"{a} {nom['rmse'][str(h)][a]['mean']:.3f}" for a in AXES) + " rad)")
    hdr = f"{'corner':<42}" + "".join(f"{a + ' within':>13}{a + ' RMSE':>13}" for a in AXES)
    print(hdr); print("-" * len(hdr))
    for r in rows[1:]:
        if r["group"] not in groups:
            continue
        line = f"{r['corner']:<42}"
        for a in AXES:
            dw = (r["axis"][str(h)][a]["mean"] - nom["axis"][str(h)][a]["mean"]) * 100
            n_rm = nom["rmse"][str(h)][a]["mean"]; c_rm = r["rmse"][str(h)][a]["mean"]
            line += f"{dw:>+12.1f}p{(c_rm / n_rm - 1) * 100:>+12.1f}%"
        print(line)


def print_partition(part, rows, h):
    print("\n" + "=" * 100)
    print(f"PARTITION at {h * 20} ms -- is the drop the body, or the model?")
    print("=" * 100)
    print(f"  {'corner':<42}{'fall %':>8}{'walk %':>8}{'fast %':>8}"
          + "".join(f"{'pitch ' + s:>13}" for s in ("frozen", "oracle", "intrins.")))
    print("  " + "-" * 96)
    for r in rows:
        p = part[r["corner"]]
        sp = p["split_vs_nominal_pts"]["pitch"]
        share = sp["intrinsic_share"]
        print(f"  {r['corner']:<42}{p['fall_rate'] * 100:>7.1f}%"
              f"{p['regime_mix']['policy walking'] * 100:>7.1f}%{p['regime_mix']['fast (|gyro|>3)'] * 100:>7.1f}%"
              f"{p['held_out_frozen']['pitch'] * 100:>12.1f}%{p['held_out_oracle']['pitch'] * 100:>12.1f}%"
              + (f"{share * 100:>12.0f}%" if share is not None and abs(sp["frozen_drop"]) >= 1.0 else f"{'--':>13}"))
    print("  frozen/oracle are scored on each corner's held-out second half; intrins. = the share")
    print("  of that corner's frozen drop that a model trained on the corner's own data also pays.")


def print_regime_split(part, corner, ref="nominal (published)"):
    a = part.get(corner); b = part.get(ref)
    if not a or not b:
        return
    print(f"\n  per-regime pitch within-0.2rad, {corner} vs {ref}:")
    print(f"    {'regime':<20}{'nominal':>10}{'corner':>10}{'delta':>10}{'n starts':>10}")
    for name, v in a["regime_within_0.2rad_axis"].items():
        if name not in b["regime_within_0.2rad_axis"]:
            continue
        n = b["regime_within_0.2rad_axis"][name]
        print(f"    {name:<20}{n['pitch'] * 100:>9.1f}%{v['pitch'] * 100:>9.1f}%"
              f"{(v['pitch'] - n['pitch']) * 100:>+9.1f}p{v['n_starts_mean']:>10.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--corner-steps", type=int, default=30000, help="ticks per corner-seed (600 s)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--horizons", type=int, nargs="+", default=[5, 25])
    ap.add_argument("--oracle-epochs", type=int, default=20, help="oracle fit budget, as in sim2real_proxy")
    args = ap.parse_args()
    t_start = time.time()
    h_part = max(args.horizons)

    bm, tm, ipos_x, ipos_z = base_mass_kg()
    geo = balance_geometry()
    phone_shift_m = DR["dcom_x"][1] * (bm + PHONE_KG) / PHONE_KG
    print("=" * 100)
    print("ANCHORS")
    print("=" * 100)
    print(f"  base body {bm * 1000:.0f} g, whole body {tm * 1000:.0f} g (twin XML, olie body)")
    print(f"  the twin carries NO phone: the base mass is battery + structure, its CoM at "
          f"x {ipos_x * 1000:+.1f} mm, z {ipos_z * 1000:+.1f} mm (below the torso mid-plane)")
    print(f"  +-{BASE_DELTA_KG * 1000:.0f} g corners add mass AT that CoM (body_ipos untouched) = whole-body "
          f"mass_scale {(tm - BASE_DELTA_KG) / tm:.3f} / {(tm + BASE_DELTA_KG) / tm:.3f}; "
          f"published DR ends {DR['mass_scale'][0]} / {DR['mass_scale'][1]}")
    print(f"  a real {PHONE_KG * 1000:.0f} g phone would be mass_scale {(tm + PHONE_KG) / tm:.2f} -- outside every "
          f"corner here and outside the DR range; no corner in this file mounts a phone")
    print(f"  com corners are the published DR endpoints: dcom_x {DR['dcom_x']} m, dcom_z {DR['dcom_z']} m; "
          f"leg_scale {DR['leg_scale']}")
    print(f"  for scale only: with mass fixed, {DR['dcom_x'][1] * 1000:.0f} mm of base-CoM shift is what moving a "
          f"{PHONE_KG * 1000:.0f} g phone {phone_shift_m * 1000:.0f} mm would do -- beyond the "
          f"{geo['torso_half_length_m'] * 1000:.0f} mm torso half-length")
    print("\n  balance geometry (measured from the model, not asserted):")
    print(f"    foot support box: x +-{geo['support_box_half_x_m'] * 1000:.1f} mm (both leg geoms sit at x = 0)")
    for label, c in geo["corners"].items():
        where = "INSIDE the support box" if c["inside_support_box"] else \
                f"outside it by {c['margin_outside_box_mm']:.1f} mm"
        print(f"    {label:<14} whole-body CoM x {c['whole_body_com_x_m'] * 1000:>+7.2f} mm -> {where}")
    print("    so +3 cm is not only a parameter change: it moves the body into a different")
    print("    balance regime, which the partition below is there to separate from model error.")

    print("\n" + "=" * 100)
    print("DECISION RULE, stated before the numbers")
    print("=" * 100)
    print("  Protocol is contact_friction's, i.e. sim2real_proxy's: the frozen nominal forward")
    print("  model trained once on data/olie_train.npz, evaluated open-loop on each corner's own")
    print("  stream. Corners share SEEDS, i.e. the same initial condition -- not common random")
    print("  numbers: collect() draws sim.rng.random() only when the body has fallen, so streams")
    print("  desynchronise as soon as fall behaviour differs. Differences are unpaired, and the")
    print("  per-corner seed spread is published beside every verdict.")
    print("  Metrics per corner:")
    print("    within_0.2rad @100ms (roll/pitch only) -- the exact metric of the published negative;")
    print("    within_0.2rad per axis @100/500ms -- adds 500 ms and YAW, which it never scored;")
    print("    RMSE per axis, beside it -- a threshold crossing with unchanged error magnitude is")
    print("      not a moved axis;")
    print(f"    fall rate, regime mix, per-regime accuracy and an oracle at {h_part * 20} ms -- the body/model split.")
    print("  Material = |mean delta| > max(3.0 pts, 2x the nominal seed spread OF THAT METRIC).")
    print("  Resolved = this corner's three seeds separate from nominal's three by more than that")
    print("    same bar. An axis is reported as moved only when it is material AND resolved; a wide")
    print("    corner spread is published beside every verdict, material or null.")
    print("  LIMITS: prediction accuracy, not policy transfer; and the twin's prediction across")
    print("  units, not a second real robot -- every real-log number here is one unit, one phone.")

    tr = np.load(HERE / "data" / "olie_train.npz")
    Xtr, Ytr, *_ = make_windows(tr["obs"], tr["act"], tr["next_obs"], tr["done"], K)
    print(f"\n  training the frozen nominal model ({args.epochs} epochs)...", flush=True)
    nominal = MLP(hidden=128, epochs=args.epochs).fit(Xtr, Ytr)

    rows = score_corners(nominal, corners(), args.seeds, args.corner_steps, args.horizons,
                         extra=make_partition(h_part, args.oracle_epochs))

    legacy_spread, legacy_thresh, legacy_verdicts = decide(rows, args.horizons, args.seeds)
    thresh, verdicts = decide_per_metric(rows, args.horizons)
    print_threshold_block(thresh, rows, args.horizons, legacy_thresh)

    flips = []
    for corner, v in verdicts.items():
        lv = legacy_verdicts[corner]
        for k, m in v["axis"].items():
            was = lv["axis"][k]["material"]
            if was != m["material"]:
                flips.append((corner, k, m["delta_pts"], was, m["material"]))
        if lv["legacy_material"] != v["legacy_material"]:
            flips.append((corner, "legacy@5", v["legacy_delta_pts"], lv["legacy_material"], v["legacy_material"]))
    print(f"\n  verdicts that flip under the per-metric rule: {len(flips)}")
    for corner, k, d, was, now in flips:
        print(f"    {corner:<42} {k:<10} {d:>+6.1f} pts   {'material' if was else 'flat'} -> "
              f"{'material' if now else 'flat'}")

    print_result_tables(rows, verdicts, args.horizons)
    print_rmse_table(rows, h_part)

    part = partition_table(rows, h_part)
    print_partition(part, rows, h_part)
    print_regime_split(part, "com x +0.03 only (DR endpoint, forward)")

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    for group, label in (("body", "mass / centre of mass / leg length"),
                         ("mass_fixed_com", "+-75 g at a fixed CoM"),
                         ("worst", "published worst-case combinations"), ("gain", "servo gain"),
                         ("sliding", "sliding friction")):
        moved, unresolved = [], []
        for r in rows[1:]:
            if r["group"] != group:
                continue
            v = verdicts[r["corner"]]
            mv = [k for k, m in v["axis"].items() if m["reported_as_moved"]]
            un = [k for k, m in v["axis"].items() if m["material"] and not m["resolved"]]
            if mv:
                moved.append(f"{r['corner']} [{', '.join(mv)}]")
            if un:
                unresolved.append(f"{r['corner']} [{', '.join(un)}]")
        print(f"  {label:<40} "
              + ("moves nothing on any axis at 100 or 500 ms" if not moved else
                 "moves: " + ", ".join(moved)))
        if unresolved:
            print(f"  {'':<40} material by the mean but unresolved at 3 seeds: " + ", ".join(unresolved))

    com_fwd = "com x +0.03 only (DR endpoint, forward)"
    sp = part[com_fwd]["split_vs_nominal_pts"]["pitch"]
    per_seed_pitch = np.array(rows[[r["corner"] for r in rows].index(com_fwd)]["axis"][str(h_part)]["pitch"]["per_seed"])
    nom_pitch = rows[0]["axis"][str(h_part)]["pitch"]["mean"]
    d_seeds = (per_seed_pitch - nom_pitch) * 100
    print(f"\n  the headline corner, {com_fwd}:")
    print(f"    pitch @{h_part * 20} ms {(np.mean(per_seed_pitch) - nom_pitch) * 100:+.1f} pts "
          f"(per seed {', '.join(f'{d:+.1f}' for d in sorted(d_seeds))}), fall rate "
          f"{part[com_fwd]['fall_rate'] * 100:.1f}% vs {part[rows[0]['corner']]['fall_rate'] * 100:.1f}% nominal")
    print(f"    of the held-out frozen drop ({sp['frozen_drop']:+.1f} pts), a model trained on this corner's own"
          f" data still pays {sp['intrinsic']:+.1f} pts"
          + (f" = {sp['intrinsic_share'] * 100:.0f}% intrinsic stream difficulty, "
             f"{100 - sp['intrinsic_share'] * 100:.0f}% model mismatch" if sp["intrinsic_share"] is not None else ""))

    out = {"config": vars(args),
           "anchors": {"base_mass_kg": bm, "whole_body_mass_kg": tm,
                       "base_com_x_m": ipos_x, "base_com_z_m": ipos_z,
                       "base_mass_delta_kg": BASE_DELTA_KG,
                       "base_delta_as_whole_body_mass_scale": [(tm - BASE_DELTA_KG) / tm, (tm + BASE_DELTA_KG) / tm],
                       "reference_phone_kg": PHONE_KG,
                       "reference_phone_as_whole_body_mass_scale": (tm + PHONE_KG) / tm,
                       "phone_shift_equivalent_to_dcom_x_max_m": phone_shift_m,
                       "no_corner_mounts_a_phone": True,
                       "balance_geometry": geo,
                       "published_DR": {k: list(v) for k, v in DR.items()}},
           "decision_rule": {"per_metric_threshold": {k: float(v) for k, v in thresh.items()},
                             "nominal_seed_spread_per_metric": {k: float(nominal_spread(rows[0], k, args.horizons))
                                                                for k in metric_keys(args.horizons)},
                             "text": "material = |mean delta vs nominal| > max(3.0 pts, 2x the nominal seed "
                                     "spread of that metric and horizon); resolved = the corner's seeds separate "
                                     "from nominal's by more than that same bar; an axis is reported as moved only "
                                     "when it is both, and every verdict carries the corner's own seed spread",
                             "superseded_rule": {"text": "material = |delta| > max(3.0 pts, 2x the WORST nominal "
                                                         "seed spread across all metrics)",
                                                 "nominal_seed_spread": float(legacy_spread),
                                                 "threshold": float(legacy_thresh)},
                             "flips_vs_superseded_rule": [{"corner": c, "metric": k, "delta_pts": d,
                                                           "was_material": w, "now_material": n}
                                                          for c, k, d, w, n in flips],
                             "pairing": "seeds shared = same initial condition, NOT common random numbers; "
                                        "collect() consumes sim.rng only when fallen, so streams desynchronise",
                             "limits": ["forward-model prediction accuracy, not policy transfer",
                                        "twin prediction across units; every real-log number is one unit, one phone"]},
           "rows": rows, "verdicts": verdicts, "verdicts_superseded_rule": legacy_verdicts,
           "partition": {"horizon_ticks": h_part, "horizon_ms": h_part * 20,
                         "oracle": "MLP(128) trained on the corner's own first half; frozen and oracle both "
                                   "scored on the held-out second half",
                         "fallen_definition": "GrowBotSim.fallen(): |roll| > 1.2 or |pitch| > 1.2 rad",
                         "by_corner": part},
           "runtime_s": float(time.time() - t_start)}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "body_params.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote results/body_params.json   total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
