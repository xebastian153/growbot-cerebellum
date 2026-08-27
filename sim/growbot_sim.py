"""Collect twin data: `python sim/growbot_sim.py --steps N --seed S --out data/x.npz`.

The twin itself lives in `growbot_cerebellum.sim` (GrowBotSim, ServoModel, perturb,
collect); this file is the documented data-collection command and nothing else. The
vendored body XMLs and policy weights stay in this directory (see NOTICE).
"""
from __future__ import annotations
import argparse, time
from pathlib import Path

import numpy as np

from growbot_cerebellum.sim import BODIES, CTRL_HZ, collect

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data" / "growbot_50hz.npz"))
    ap.add_argument("--body", default="walk", choices=list(BODIES))
    args = ap.parse_args()
    t0 = time.time()
    O, A, O2, D, M = collect(args.steps, args.seed, log_every=50_000, body=args.body)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, obs=O, act=A, next_obs=O2, done=D, mode=M)
    dt = time.time() - t0
    print(f"{args.steps} steps = {args.steps / CTRL_HZ / 60:.1f} sim-minutes in {dt:.0f}s "
          f"({args.steps / dt / CTRL_HZ:.0f}x realtime)")
    print("modes:", {m: int((M == m).sum()) for m in np.unique(M)})
    print("episodes:", int(D.sum()))
    r, p = O[:, 0], O[:, 1]
    print(f"roll  range [{r.min():+.2f}, {r.max():+.2f}]  pitch range [{p.min():+.2f}, {p.max():+.2f}]")
    print(f"fallen frames (|roll| or |pitch| > 1.2): {np.mean((abs(r) > 1.2) | (abs(p) > 1.2)) * 100:.1f}%")
    print("saved", args.out)


if __name__ == "__main__":
    main()
