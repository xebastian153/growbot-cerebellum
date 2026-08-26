"""Body parameters at the horizons the published negative never scored.

Every GrowBot is assembled by a different person around a different phone. Phones
span roughly 150-250 g on a base body of ~427 g (`sim/growbot_body.xml`), so units
differ by about +-10-20 % in mass, and the centre of mass moves with the phone's
length. This repo publishes that body-parameter domain randomisation is invisible in
the IMU (`sim2real_proxy.py`, 13 corners) -- but that negative was scored with
`horizon_within(..., h=5)`: 100 ms only, roll and pitch only. `contact_friction.py`
closed the same limit for friction by scoring at 100 AND 500 ms with yaw. Mass,
centre of mass and leg length were never scored there. This script does that, on the
identical protocol, and adds two corners that are the phone question asked directly:
+-75 g on the base body alone, legs untouched.

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
from forward import MLP, make_windows                             # noqa: E402
from sim2real_proxy import K, corners as published_corners        # noqa: E402
from contact_friction import score_corners, decide, print_tables, any_material  # noqa: E402

HERE = Path(__file__).parent
PHONE_DELTA_KG = 0.075   # +-75 g on the base: brackets a 125-275 g phone around a 200 g one


def base_mass_kg(body="olie"):
    m = GrowBotSim(seed=0, body=body).m
    return float(m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_body")]), float(m.body_mass.sum())


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
        ("phone -75 g on the base (light phone)", {**base, "base_mass_delta": -PHONE_DELTA_KG}, "phone"),
        ("phone +75 g on the base (heavy phone)", {**base, "base_mass_delta": +PHONE_DELTA_KG}, "phone"),
        ("com x -0.03 only (phone shifted back)", {**base, "dcom": (lo["dcom_x"], 0.0, 0.0)}, "body"),
        ("com x +0.03 only (phone shifted forward)", {**base, "dcom": (hi["dcom_x"], 0.0, 0.0)}, "body"),
        ("com z +0.015 only (phone mounted higher)", {**base, "dcom": (0.0, 0.0, hi["dcom_z"])}, "body"),
    ]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--corner-steps", type=int, default=30000, help="ticks per corner-seed (600 s)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--horizons", type=int, nargs="+", default=[5, 25])
    args = ap.parse_args()
    t_start = time.time()

    bm, tm = base_mass_kg()
    print("=" * 78)
    print("ANCHORS")
    print("=" * 78)
    print(f"  base body {bm * 1000:.0f} g, whole body {tm * 1000:.0f} g (twin XML, olie body)")
    print(f"  phone corners: +-{PHONE_DELTA_KG * 1000:.0f} g on the base alone = whole-body mass_scale "
          f"{(tm - PHONE_DELTA_KG) / tm:.3f} / {(tm + PHONE_DELTA_KG) / tm:.3f}; the published DR ends are "
          f"{DR['mass_scale'][0]} / {DR['mass_scale'][1]} on every body")
    print(f"  com corners: dcom_x {DR['dcom_x']} m, dcom_z {DR['dcom_z']} m; leg_scale {DR['leg_scale']}")

    print("\n" + "=" * 78)
    print("DECISION RULE, stated before the numbers")
    print("=" * 78)
    print("  Protocol is contact_friction's, i.e. sim2real_proxy's: the frozen nominal forward")
    print("  model trained once on data/olie_train.npz, evaluated open-loop on each corner's own")
    print("  stream, seeds shared across every corner. Two metrics per corner:")
    print("    within_0.2rad @100ms (roll/pitch only) -- the exact metric of the published negative;")
    print("    within_0.2rad per axis @100/500ms -- adds 500 ms and YAW, which it never scored.")
    print("  Material = a shift from nominal larger than max(3.0 pts, 2x the nominal seed spread).")
    print("  LIMITS: prediction accuracy, not policy transfer; and the twin's prediction across")
    print("  units, not a second real robot -- every real-log number here is one unit, one phone.")

    tr = np.load(HERE / "data" / "olie_train.npz")
    Xtr, Ytr, *_ = make_windows(tr["obs"], tr["act"], tr["next_obs"], tr["done"], K)
    print(f"\n  training the frozen nominal model ({args.epochs} epochs)...", flush=True)
    nominal = MLP(hidden=128, epochs=args.epochs).fit(Xtr, Ytr)

    rows = score_corners(nominal, corners(), args.seeds, args.corner_steps, args.horizons)
    spread, thresh, verdicts = decide(rows, args.horizons, args.seeds)
    print_tables(rows, verdicts, thresh, args.horizons, title="RESULTS")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    for group, label in (("body", "mass / centre of mass / leg length"), ("phone", "+-75 g phone on the base"),
                         ("worst", "published worst-case combinations"), ("gain", "servo gain"),
                         ("sliding", "sliding friction")):
        hits = any_material(rows, verdicts, group)
        hits500 = any_material(rows, verdicts, group, None)
        yaw = [c for c, h in hits if any(k.startswith("yaw") for k in h)]
        h500 = sorted({k for _, h in hits for k in h if k.endswith("@25")})
        print(f"  {label:<40} "
              + ("no material effect on any axis at 100 or 500 ms" if not hits else
                 f"material: {', '.join(c + ' [' + ', '.join(h) + ']' for c, h in hits)}"))
    body_hits = any_material(rows, verdicts, "body") + any_material(rows, verdicts, "phone")
    print(f"\n  the published negative at 500 ms and on yaw: "
          + ("HOLDS -- its stated limit (100 ms, roll/pitch only) is now closed for mass, CoM and leg length"
             if not body_hits else "DOES NOT HOLD -- see material rows above; README/EXPERIMENTS/READING corrected"))

    out = {"config": vars(args),
           "anchors": {"base_mass_kg": bm, "whole_body_mass_kg": tm, "phone_delta_kg": PHONE_DELTA_KG,
                       "phone_as_whole_body_mass_scale": [(tm - PHONE_DELTA_KG) / tm, (tm + PHONE_DELTA_KG) / tm],
                       "published_DR": {k: list(v) for k, v in DR.items()}},
           "decision_rule": {"nominal_seed_spread": float(spread), "threshold": float(thresh),
                             "text": "material = |delta vs nominal| > max(3.0 pts, 2x nominal seed spread)",
                             "limits": ["forward-model prediction accuracy, not policy transfer",
                                        "twin prediction across units; every real-log number is one unit, one phone"]},
           "rows": rows, "verdicts": verdicts,
           "runtime_s": float(time.time() - t_start)}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "body_params.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote results/body_params.json   total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
