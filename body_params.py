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
body. Every corner therefore reports its fall rate (the fraction of TICKS spent fallen,
not how often the body tips), its regime mix, its per-regime accuracy, and an oracle (a
model trained on that corner's own data) at 500 ms, which splits the drop exactly:

    frozen_c - frozen_nom  =  (oracle_c - oracle_nom)  +  [mismatch_c - mismatch_nom]
                              ^ intrinsic difficulty     ^ what training on the body fixes

The oracle is a small-data model -- 14741 windows and 20 epochs against the frozen
model's 400 k ticks and 60 epochs -- and on the NOMINAL body it scores below the frozen
model on all three axes. It is therefore a LOWER BOUND on what a better-matched model
recovers, not a ceiling, and that nominal deficit is published per axis beside every
split so an intrinsic figure of the same order can be read for what it is. Every
partition quantity is published per seed with its spread and with the same
material/resolved marks the axis verdicts carry; no partition number appears as a bare
mean, and the shares appear as a range across seeds rather than a point estimate.

Two limits, stated up front and repeated in the write-up:
  - this is forward-model PREDICTION accuracy, not policy TRANSFER;
  - every real-log number in this repo comes from ONE unit and ONE phone. This sweep
    says what the twin predicts across units, not what a second real robot does.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from growbot_cerebellum.sim import DR, GrowBotSim, quat_to_rpy
from growbot_cerebellum.forward import MLP, make_windows, rollout_error, K, AXES
from growbot_cerebellum.sim2real import corners as published_corners
from growbot_cerebellum.honesty import (seed_stat, score_corners, decide, metric_keys, nominal_spread,
                                        seed_separation, decide_per_metric)

HERE = Path(__file__).parent
BASE_DELTA_KG = 0.075      # +-75 g added at the base's existing CoM: a mass isolation
PHONE_KG = 0.200           # a real phone, for the anchor arithmetic only -- never a corner
FALL_RAD = 1.2             # GrowBotSim.fallen(): |roll| > 1.2 or |pitch| > 1.2
FAST_RADS = 3.0            # forward.by_regime's own "fast" cut
MIN_REGIME_TICKS = 50      # forward.by_regime's own floor


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
    """Whole-body CoM along the body x axis vs the foot support box, BOTH in the base body
    frame at the reference pose (every joint at zero).

    The support box is the x-extent of the two leg geoms (they sit at x = 0), i.e. what the
    body can put weight on without tipping. A CoM outside it is held up by contact torque; a
    CoM inside it is statically supported. Crossing that line is a change of balance regime,
    which is why the CoM corners cannot be read as model error alone.

    CORRECTED, and the correction moves the published millimetres. The first version of this
    function read the whole-body CoM in WORLD x -- `d.xipos` -- while comparing it against a
    body-frame half-extent, and it read it at whatever pose `GrowBotSim.__init__` had left
    the body in, because `reset()` steps 0.5 s of physics before anything is measured. The
    nominal and -3 cm bodies settle rotated about -34.5 deg in pitch, so those two rows were
    a body-frame offset projected onto a world axis: nominal came out -19.30 mm instead of
    -27.42, and -3 cm -41.34 instead of -54.12. The +3 cm body settles level (-0.1 deg),
    which is why its -0.71 mm was already right and why the artefact was invisible in the one
    row the section leans on. Both quantities are now measured in the same frame.

    Pose dependence of the corrected number is small and is published with it: leg swing is
    the only pose freedom that moves the body-frame CoM, and the two legs carry 11.0 % of the
    mass with their own CoM 37 mm below their hinges, so a full +-90 deg swing moves it by at
    most ~4 mm; at the pose the sim actually settles into it moves it by 0.05 mm. The
    qualitative reading is unchanged by the correction: nominal and -3 cm sit outside the
    box, +3 cm sits inside it.
    """
    def measure(dr):
        m = GrowBotSim(seed=0, body=body, dr=dr).m
        base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_body")

        def body_frame_com_x(mm, dd):
            R = dd.xmat[base].reshape(3, 3)
            com = (dd.xipos * mm.body_mass[:, None]).sum(0) / mm.body_mass.sum()
            return float((R.T @ (com - dd.xpos[base]))[0])

        ref_d = mujoco.MjData(m)                 # reference pose: joints at zero, no stepping
        mujoco.mj_resetData(m, ref_d)
        mujoco.mj_kinematics(m, ref_d)
        ref = body_frame_com_x(m, ref_d)

        s = GrowBotSim(seed=0, body=body, dr=dr)  # the pose reset()'s 0.5 s of stepping leaves
        mujoco.mj_forward(s.m, s.d)
        com_w = (s.d.xipos * s.m.body_mass[:, None]).sum(0) / s.m.body_mass.sum()
        return {"base_ipos_x_m": float(m.body_ipos[base, 0]),
                "whole_body_com_x_m": ref,
                "whole_body_com_x_at_settled_pose_m": body_frame_com_x(s.m, s.d),
                "superseded_world_x_at_settled_pose_m": float(com_w[0] - s.d.xpos[base, 0]),
                "settled_pitch_rad": float(quat_to_rpy(s.d.qpos[3:7])[1])}

    m = GrowBotSim(seed=0, body=body).m
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "lower_leg_1")
    half_x = float(m.geom_size[gid, 0])
    torso_half_x = float(m.geom_size[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "torso_geom"), 0])
    leg_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_leg")
    leg_frac = float(2 * m.body_mass[leg_bid] / m.body_mass.sum())
    leg_arm = abs(float(m.body_ipos[leg_bid, 2]))
    out = {"frame": "base body frame, every joint at zero (the frame the support box is "
                    "defined in); the superseded reading was world x at the settled pose",
           "support_box_half_x_m": half_x, "torso_half_length_m": torso_half_x,
           "leg_mass_fraction": leg_frac,
           "max_leg_swing_com_shift_mm": float(leg_frac * leg_arm * 1000),
           "corners": {}}
    for label, dcom in (("nominal", 0.0), ("com x -0.03", DR["dcom_x"][0]), ("com x +0.03", DR["dcom_x"][1])):
        c = measure({"mass_scale": 1.0, "dcom": (dcom, 0.0, 0.0)})
        com_rel = c["whole_body_com_x_m"]
        c["inside_support_box"] = bool(abs(com_rel) <= half_x)
        c["margin_outside_box_mm"] = float((abs(com_rel) - half_x) * 1000)
        c["settled_pose_shift_mm"] = float((c["whole_body_com_x_at_settled_pose_m"] - com_rel) * 1000)
        c["superseded_world_reading_error_mm"] = float(
            (c["superseded_world_x_at_settled_pose_m"] - com_rel) * 1000)
        out["corners"][label] = c
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
    def partition(nominal, O, A, O2, D, M):
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


def partition_table(rows, floor=0.03):
    """Per-SEED partition quantities with their spread, and the exact split of each corner's
    frozen drop, decided by the same two marks the axis table uses.

        frozen_c - frozen_n = (oracle_c - oracle_n) + [(frozen_c - oracle_c) - (frozen_n - oracle_n)]
                               intrinsic difficulty    model mismatch the corner adds

    CORRECTED. The first published version of this table averaged three seeds and reported
    nothing else -- no spread, no per-seed values, no verdict -- which is precisely the
    defect the axis verdicts above were rewritten to remove, reintroduced one section later.
    Every quantity here now carries its three seeds and their spread, and every split
    carries:

      threshold  max(3.0 pts, 2x the NOMINAL seed spread of that same quantity). The oracle
                 is a small-data model and its own nominal spread is much wider than the
                 frozen model's, so the bar the intrinsic component has to clear is wider
                 too -- that is the point of deriving each bar from its own quantity.
      resolved   the corner's three seeds separate from nominal's three by more than that
                 bar (`seed_separation`, the same unpaired criterion the axis table uses).

    The per-seed shares are seed-index readings -- corner seed i against nominal seed i --
    not paired differences: the streams desynchronise from the first tick where the bodies
    differ (see `score_corners`). They are published as a RANGE for that reason, and a share
    is only formed where that seed's frozen drop is at least 1 pt, since a share of a drop
    near zero is not a quantity.
    """
    nom = rows[0]

    def held(r, kind, a):
        return [sd["held_out"][kind][a] for sd in r["per_seed"]]

    regime_names = sorted({n for row in rows for sd in row["per_seed"]
                           for n in sd["regime_within_0.2rad_axis"]})
    out = {}
    for r in rows:
        reg = {}
        for name in regime_names:
            vals = [sd["regime_within_0.2rad_axis"][name] for sd in r["per_seed"]
                    if name in sd["regime_within_0.2rad_axis"]]
            if not vals:
                # reported, not silently dropped: a regime below MIN_REGIME_TICKS on every
                # seed of this corner is a fact about the corner, not an absent row
                reg[name] = {"seeds_present": 0, "absent_reason":
                             f"under {MIN_REGIME_TICKS} ticks on every seed"}
                continue
            reg[name] = {"seeds_present": len(vals),
                         "n_starts": seed_stat([v["n_starts"] for v in vals]),
                         **{a: seed_stat([v["within_0.2rad_axis"][a] for v in vals]) for a in AXES}}
        out[r["corner"]] = {
            "fall_rate": seed_stat([sd["fall_rate"] for sd in r["per_seed"]]),
            "regime_mix": {k: seed_stat([sd["regime_mix"][k] for sd in r["per_seed"]])
                           for k in r["per_seed"][0]["regime_mix"]},
            "regime_within_0.2rad_axis": reg,
            "held_out_frozen": {a: seed_stat(held(r, "frozen_within_0.2rad_axis", a)) for a in AXES},
            "held_out_oracle": {a: seed_stat(held(r, "oracle_within_0.2rad_axis", a)) for a in AXES},
        }

    nomp = out[nom["corner"]]
    # every bar comes from the nominal spread of the quantity it decides
    bar = {kind: {a: max(floor, 2.0 * nomp[kind][a]["spread"]) * 100 for a in AXES}
           for kind in ("held_out_frozen", "held_out_oracle")}
    bar_fall = max(floor, 2.0 * nomp["fall_rate"]["spread"]) * 100
    # the oracle's own deficit on the NOMINAL body, per axis: the yardstick's error, which
    # every intrinsic figure below has to be read against
    oracle_deficit = {a: float((nomp["held_out_frozen"][a]["mean"]
                                - nomp["held_out_oracle"][a]["mean"]) * 100) for a in AXES}

    for name, part in out.items():
        fsep = seed_separation(part["fall_rate"]["per_seed"], nomp["fall_rate"]["per_seed"])
        part["fall_rate_vs_nominal"] = {
            "delta_pts": float((part["fall_rate"]["mean"] - nomp["fall_rate"]["mean"]) * 100),
            "threshold_pts": bar_fall, "seed_separation_pts": fsep,
            "resolved": bool(abs(fsep) > bar_fall)}
        part["oracle_deficit_on_nominal_body_pts"] = oracle_deficit
        part["split_vs_nominal_pts"] = {}
        for a in AXES:
            fz = part["held_out_frozen"][a]["per_seed"]; nfz = nomp["held_out_frozen"][a]["per_seed"]
            oc = part["held_out_oracle"][a]["per_seed"]; noc = nomp["held_out_oracle"][a]["per_seed"]
            drop = [(c - n) * 100 for c, n in zip(fz, nfz)]
            intr = [(c - n) * 100 for c, n in zip(oc, noc)]
            mism = [d - i for d, i in zip(drop, intr)]
            share = [(float(i / d) if abs(d) >= 1.0 else None) for d, i in zip(drop, intr)]
            known = [x for x in share if x is not None]
            dsep = seed_separation(fz, nfz); isep = seed_separation(oc, noc)
            mean_drop = float(np.mean(drop)); mean_intr = float(np.mean(intr))
            part["split_vs_nominal_pts"][a] = {
                "frozen_drop": seed_stat(drop), "intrinsic": seed_stat(intr),
                "model_mismatch": seed_stat(mism),
                "intrinsic_share_per_seed": share,
                "intrinsic_share_range": ([min(known), max(known)] if known else None),
                "intrinsic_share_of_the_means": (float(mean_intr / mean_drop)
                                                 if abs(mean_drop) >= 1.0 else None),
                "frozen_drop_threshold_pts": bar["held_out_frozen"][a],
                "frozen_drop_seed_separation_pts": dsep,
                "frozen_drop_resolved": bool(abs(dsep) > bar["held_out_frozen"][a]),
                "intrinsic_threshold_pts": bar["held_out_oracle"][a],
                "intrinsic_seed_separation_pts": isep,
                "intrinsic_resolved": bool(abs(isep) > bar["held_out_oracle"][a]),
                "intrinsic_vs_oracle_deficit_ratio": (abs(mean_intr) / oracle_deficit[a]
                                                      if oracle_deficit[a] > 1e-9 else None),
            }
    return out


def print_threshold_block(thresh, rows, horizons, legacy_thresh):
    nom = rows[0]
    print("\n  per-metric thresholds (max(3.0 pts, 2x the nominal spread of that metric)),")
    print(f"  against the single worst-case bar this section used before, {legacy_thresh * 100:.2f} pts:")
    print(f"    {'metric':<12}{'nominal spread':>16}{'threshold':>12}{'was':>8}")
    for k in metric_keys(horizons):
        print(f"    {k:<12}{nominal_spread(nom, k) * 100:>15.2f}p{thresh[k] * 100:>11.2f}p"
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


def print_partition(part, rows, h, axis="pitch"):
    nomp = part[rows[0]["corner"]]
    print("\n" + "=" * 118)
    print(f"PARTITION at {h * 20} ms -- is the drop the body, or the model?   ({axis}, mean +- 3-seed spread)")
    print("=" * 118)
    print(f"  {'corner':<42}{'fall %':>13}{'fast %':>13}{'frozen':>14}{'oracle':>14}"
          f"{'drop':>10}{'intrinsic':>12}{'share (3 seeds)':>18}{'res':>4}")
    print("  " + "-" * 124)
    for r in rows:
        pt = part[r["corner"]]
        sp = pt["split_vs_nominal_pts"][axis]
        rng = sp["intrinsic_share_range"]
        share = ("--" if rng is None else
                 (f"{rng[0] * 100:.0f}%" if rng[0] == rng[1]
                  else f"{rng[0] * 100:.0f} to {rng[1] * 100:.0f}%"))
        mark = ("" if r is rows[0] else
                ("R" if sp["frozen_drop_resolved"] else "?") + ("R" if sp["intrinsic_resolved"] else "?"))
        print(f"  {r['corner']:<42}"
              f"{pt['fall_rate']['mean'] * 100:>8.1f}+-{pt['fall_rate']['spread'] * 100:<4.1f}"
              f"{pt['regime_mix']['fast (|gyro|>3)']['mean'] * 100:>8.1f}+-{pt['regime_mix']['fast (|gyro|>3)']['spread'] * 100:<4.1f}"
              f"{pt['held_out_frozen'][axis]['mean'] * 100:>9.1f}+-{pt['held_out_frozen'][axis]['spread'] * 100:<4.1f}"
              f"{pt['held_out_oracle'][axis]['mean'] * 100:>9.1f}+-{pt['held_out_oracle'][axis]['spread'] * 100:<4.1f}"
              f"{sp['frozen_drop']['mean']:>+10.1f}{sp['intrinsic']['mean']:>+12.1f}"
              f"{share:>18}{mark:>4}")
    print("  frozen/oracle are scored on each corner's held-out second half; intrinsic = the part of")
    print("  the drop a model trained on the corner's own data also pays; share = the RANGE of the")
    print("  three per-seed shares, never a point estimate. The two marks are drop-resolved and")
    bars_f = "/".join(f"{a} {nomp['split_vs_nominal_pts'][a]['frozen_drop_threshold_pts']:.1f}" for a in AXES)
    bars_o = "/".join(f"{a} {nomp['split_vs_nominal_pts'][a]['intrinsic_threshold_pts']:.1f}" for a in AXES)
    print(f"  intrinsic-resolved against their own bars ({bars_f} pts frozen,")
    print(f"  {bars_o} pts oracle) -- R = separated at 3 seeds, ? = not.")
    print("  The oracle's own deficit on the NOMINAL body, the yardstick's error: "
          + ", ".join(f"{a} {nomp['oracle_deficit_on_nominal_body_pts'][a]:+.1f}" for a in AXES)
          + " pts;")
    print("  any intrinsic figure of that order is inside the yardstick, not a measurement of the body.")


def print_split_detail(part, corner):
    """Every partition number of one corner on all three axes, per seed. No bare means."""
    pt = part.get(corner)
    if pt is None:
        return
    print(f"\n  the split at {corner}, per seed (corner seed i vs nominal seed i -- seed-index")
    print("  readings of unpaired streams, not paired differences):")
    print(f"    {'axis':<7}{'drop (3 seeds)':>34}{'intrinsic (3 seeds)':>34}{'share':>24}{'resolved':>26}")
    for a in AXES:
        sp = pt["split_vs_nominal_pts"][a]
        d = sp["frozen_drop"]; i = sp["intrinsic"]
        sh = ", ".join("--" if x is None else f"{x * 100:.0f}%" for x in sp["intrinsic_share_per_seed"])
        res = ("drop " + ("yes" if sp["frozen_drop_resolved"] else "NO")
               + ", intrinsic " + ("yes" if sp["intrinsic_resolved"] else "NO"))
        print(f"    {a:<7}{d['mean']:>+8.1f} [{', '.join(f'{v:+.1f}' for v in d['per_seed'])}]"
              f"{i['mean']:>+12.1f} [{', '.join(f'{v:+.1f}' for v in i['per_seed'])}]"
              f"{sh:>24}{res:>26}")
    for a in AXES:
        sp = pt["split_vs_nominal_pts"][a]
        ratio = sp["intrinsic_vs_oracle_deficit_ratio"]
        if ratio is not None:
            print(f"    {a}: |intrinsic| {abs(sp['intrinsic']['mean']):.1f} pts is "
                  f"{ratio:.1f}x the oracle's own {pt['oracle_deficit_on_nominal_body_pts'][a]:+.1f}-pt "
                  f"deficit on the nominal body")


def print_regime_split(part, corner, ref="nominal (published)"):
    a = part.get(corner); b = part.get(ref)
    if not a or not b:
        return
    print(f"\n  per-regime pitch within-0.2rad, {corner} vs {ref} (mean +- 3-seed spread):")
    print(f"    {'regime':<20}{'nominal':>16}{'corner':>16}{'delta':>10}{'n starts':>10}")
    for name, v in a["regime_within_0.2rad_axis"].items():
        n = b["regime_within_0.2rad_axis"].get(name)
        if n is None or not v.get("seeds_present") or not n.get("seeds_present"):
            miss = v.get("absent_reason") or (n or {}).get("absent_reason") or "absent"
            print(f"    {name:<20}{miss:>52}")
            continue
        print(f"    {name:<20}{n['pitch']['mean'] * 100:>11.1f}+-{n['pitch']['spread'] * 100:<4.1f}"
              f"{v['pitch']['mean'] * 100:>11.1f}+-{v['pitch']['spread'] * 100:<4.1f}"
              f"{(v['pitch']['mean'] - n['pitch']['mean']) * 100:>+9.1f}p{v['n_starts']['mean']:>10.0f}")
    print("    A quiet bucket that collapses harder than the fast one is not by itself evidence of")
    print("    intrinsic difficulty: a frozen model carrying a static offset at the corner's new")
    print("    resting pitch would look the same here. This frozen-only table cannot separate them.")


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
    print("\n  balance geometry, base body frame, every joint at zero -- the frame the support")
    print("  box is defined in (CORRECTED: the first published version of these millimetres read")
    print("  the CoM in WORLD x at the pose reset()'s 0.5 s of stepping leaves, and compared it")
    print("  against a body-frame box):")
    print(f"    foot support box: x +-{geo['support_box_half_x_m'] * 1000:.1f} mm (both leg geoms sit at x = 0)")
    for label, c in geo["corners"].items():
        where = (f"INSIDE the support box by {-c['margin_outside_box_mm']:.1f} mm"
                 if c["inside_support_box"] else f"outside it by {c['margin_outside_box_mm']:.1f} mm")
        print(f"    {label:<14} whole-body CoM x {c['whole_body_com_x_m'] * 1000:>+7.2f} mm -> {where}")
        print(f"    {'':<14}   settles at pitch {np.rad2deg(c['settled_pitch_rad']):>+6.1f} deg; same measurement at that "
              f"settled pose {c['whole_body_com_x_at_settled_pose_m'] * 1000:+.2f} mm "
              f"({c['settled_pose_shift_mm']:+.2f} mm), superseded world-x reading "
              f"{c['superseded_world_x_at_settled_pose_m'] * 1000:+.2f} mm "
              f"({c['superseded_world_reading_error_mm']:+.2f} mm)")
    print(f"    pose dependence is bounded: the legs are {geo['leg_mass_fraction'] * 100:.1f} % of the mass, so a full")
    print(f"    +-90 deg swing moves the body-frame CoM by at most {geo['max_leg_swing_com_shift_mm']:.1f} mm; the rows above move")
    print("    it by less than 0.1 mm. The whole world-vs-body gap is base rotation, not leg swing.")
    print("    so +3 cm is not only a parameter change: it moves the body into a different")
    print("    balance regime, which the partition below is there to separate from model error.")

    print("\n" + "=" * 100)
    print("DECISION RULE, stated before the numbers")
    print("=" * 100)
    print("  Protocol is contact_friction's, i.e. sim2real_proxy's: the frozen nominal forward")
    print("  model trained once on data/olie_train.npz, evaluated open-loop on each corner's own")
    print("  stream. Corners share SEEDS, i.e. the same initial condition -- not common random")
    print("  numbers: collect() draws sim.rng every tick for the push test, mode-dependently in")
    print("  Excitation and once per episode in fresh(), so two corners diverge at the first tick")
    print("  their dynamics differ and never realign. Differences are unpaired, and the per-corner")
    print("  seed spread is published beside every verdict.")
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

    part = partition_table(rows)
    print_partition(part, rows, h_part)
    print_split_detail(part, "com x +0.03 only (DR endpoint, forward)")
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
    nomp = part[rows[0]["corner"]]
    per_seed_pitch = np.array(rows[[r["corner"] for r in rows].index(com_fwd)]["axis"][str(h_part)]["pitch"]["per_seed"])
    nom_pitch = rows[0]["axis"][str(h_part)]["pitch"]["mean"]
    d_seeds = (per_seed_pitch - nom_pitch) * 100
    fr = part[com_fwd]["fall_rate"]; nfr = nomp["fall_rate"]; frv = part[com_fwd]["fall_rate_vs_nominal"]
    rng = sp["intrinsic_share_range"]
    print(f"\n  the headline corner, {com_fwd}:")
    print(f"    pitch @{h_part * 20} ms {(np.mean(per_seed_pitch) - nom_pitch) * 100:+.1f} pts "
          f"(per seed {', '.join(f'{d:+.1f}' for d in sorted(d_seeds))})")
    print(f"    of the held-out frozen drop ({sp['frozen_drop']['mean']:+.1f} pts, per seed "
          f"{', '.join(f'{v:+.1f}' for v in sp['frozen_drop']['per_seed'])}), a model trained on this")
    print(f"    corner's own data still pays {sp['intrinsic']['mean']:+.1f} pts (per seed "
          f"{', '.join(f'{v:+.1f}' for v in sp['intrinsic']['per_seed'])}) --")
    print(f"    intrinsic share {rng[0] * 100:.0f}-{rng[1] * 100:.0f} % across the three seeds "
          f"({', '.join(f'{x * 100:.0f}%' for x in sp['intrinsic_share_per_seed'])}); the pairing is "
          f"{abs(sp['intrinsic']['mean']):.1f} of the")
    print(f"    {abs(sp['frozen_drop']['mean']):.1f} HELD-OUT points, which is a different quantity from the "
          f"full-stream {abs((np.mean(per_seed_pitch) - nom_pitch) * 100):.1f}-pt table figure.")
    print(f"    resolved at 3 seeds: drop {'YES' if sp['frozen_drop_resolved'] else 'NO'} "
          f"(separation {sp['frozen_drop_seed_separation_pts']:+.1f} vs bar {sp['frozen_drop_threshold_pts']:.1f}), "
          f"intrinsic {'YES' if sp['intrinsic_resolved'] else 'NO'} "
          f"(separation {sp['intrinsic_seed_separation_pts']:+.1f} vs bar {sp['intrinsic_threshold_pts']:.1f})")
    print(f"    time in the fallen state {fr['mean'] * 100:.1f}+-{fr['spread'] * 100:.1f}% vs "
          f"{nfr['mean'] * 100:.1f}+-{nfr['spread'] * 100:.1f}% nominal, unresolved at 3 seeds "
          f"(separation {frv['seed_separation_pts']:+.1f} vs bar {frv['threshold_pts']:.1f}) -- and note this is")
    print("    the FRACTION OF TICKS spent fallen, not how often the body falls; it does not say the")
    print("    body tips over no more often.")
    print("    the oracle is a small-data model, so it is a LOWER BOUND on what a better-matched model")
    print("    recovers, not a ceiling: on the nominal body it trails the frozen model by "
          + ", ".join(f"{a} {nomp['oracle_deficit_on_nominal_body_pts'][a]:.1f}" for a in AXES) + " pts.")

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
                             "nominal_seed_spread_per_metric": {k: float(nominal_spread(rows[0], k))
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
                                        "collect() draws sim.rng every tick for the push test, mode-dependently "
                                        "in Excitation and once per episode in fresh(), so two corners diverge at "
                                        "the first tick their dynamics differ and never realign",
                             "limits": ["forward-model prediction accuracy, not policy transfer",
                                        "twin prediction across units; every real-log number is one unit, one phone"]},
           "rows": rows, "verdicts": verdicts, "verdicts_superseded_rule": legacy_verdicts,
           "partition": {"horizon_ticks": h_part, "horizon_ms": h_part * 20,
                         "oracle": "MLP(128) trained on the corner's own first half (14741 windows, 20 epochs); "
                                   "frozen and oracle both scored on the held-out second half. It is a "
                                   "small-data model, so it is a LOWER BOUND on recoverable mismatch, NOT a "
                                   "ceiling: on the nominal body it scores BELOW the frozen model on every "
                                   "axis, and the size of that deficit is published per axis as "
                                   "oracle_deficit_on_nominal_body_pts",
                         "fallen_definition": "GrowBotSim.fallen(): |roll| > 1.2 or |pitch| > 1.2 rad; "
                                              "fall_rate is the FRACTION OF TICKS in that state, not the "
                                              "frequency of fall events",
                         "seeds": "every quantity carries its three seeds and their spread; thresholds are "
                                  "max(3.0 pts, 2x the nominal spread of that same quantity) and 'resolved' "
                                  "is the same unpaired seed-separation criterion the axis verdicts use",
                         "denominators": "frozen_drop/intrinsic/model_mismatch are HELD-OUT-half quantities; "
                                         "the axis table's delta_pts is the full-stream figure. They are "
                                         "different quantities and must not share a denominator",
                         "by_corner": part},
           "runtime_s": float(time.time() - t_start)}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "body_params.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote results/body_params.json   total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
