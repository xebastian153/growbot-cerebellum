"""The mimic game: reproduce a target IMU trajectory by planning through imagination.

In the video the robot is moved by hand, asked to recreate the motion, and
fails -- it can read its sensors but cannot imagine what its legs will do. Here
the forward model supplies that imagination and a planner uses it the way GCML
does: propose an action sequence, roll it forward in the model, score against the
target, keep the best. The plan is then executed open-loop in the real twin and
scored against the same target. Model quality shows up directly as how well the
executed motion matches.

Planner: cross-entropy method (CEM) over a chunk of H actions -- Bernstein's
"action chunk". Sampling N candidates and rolling them through the model is one
batched forward pass, cheap enough to run every tick on a phone.

Targets come from held-out test episodes: a real IMU trace the twin produced
under actions the planner never sees. Three planners are compared:
  persistence  imagination says "nothing changes" -> effectively random plans
  linear       GCML-style forward model
  mlp          the nonlinear forward model
plus a "hold still" no-op baseline. Score is roll/pitch RMSE between executed and
target trace over the horizon, and the fraction of ticks within 0.2 rad.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from growbot_cerebellum.sim import GrowBotSim, CTRL_HZ
from growbot_cerebellum.planner import Imagination, run_episode, pick_targets, rpy_to_quat
from growbot_cerebellum.forward import Persistence, Linear, MLP, make_windows

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--T", type=int, default=100, help="target length in ticks (2 s)")
    ap.add_argument("--H", type=int, default=15, help="plan horizon (300 ms)")
    ap.add_argument("--replan", type=int, default=5)
    ap.add_argument("--n-targets", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tr = np.load(HERE / "data" / "train.npz"); te = np.load(HERE / "data" / "test.npz")
    Xtr, Ytr, *_ = make_windows(tr["obs"], tr["act"], tr["next_obs"], tr["done"], args.K)
    print("fitting forward models...", flush=True)
    models = {"persistence": Persistence().fit(Xtr, Ytr),
              "linear": Linear().fit(Xtr, Ytr),
              "mlp": MLP(hidden=128, epochs=args.epochs).fit(Xtr, Ytr)}

    rng = np.random.default_rng(args.seed)
    targets, modes = pick_targets(te, args.T, args.n_targets, rng)
    print(f"{len(targets)} target traces of {args.T / CTRL_HZ:.1f}s, modes: "
          f"{dict(zip(*np.unique(modes, return_counts=True)))}")

    results = {"hold still": []}
    for name in models: results[name] = []
    per_target = []
    t0 = time.time()
    for i, tgt in enumerate(targets):
        row = {"mode": str(modes[i])}
        for name in ["hold still"] + list(models):
            sim = GrowBotSim(seed=1000 + i)
            # start the twin from the target's own initial pose so the task is reproduction, not recovery
            sim.reset(tilt=0.0)
            sim.d.qpos[3:7] = rpy_to_quat(*tgt[0, :3]); sim.d.qvel[3:6] = tgt[0, 3:]
            import mujoco; mujoco.mj_forward(sim.m, sim.d)
            if name == "hold still":
                r = run_episode(sim, None, tgt, args.K, args.H, args.replan, rng, no_op=True)
            else:
                r = run_episode(sim, Imagination(models[name], args.K), tgt, args.K, args.H,
                                args.replan, np.random.default_rng(args.seed + i))
            results[name].append(r["rmse"]); row[name] = r["rmse"]
            row[name + "_within"] = r["within_0.2"]
        per_target.append(row)
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{len(targets)}  " + "  ".join(
                f"{n}: {np.mean(results[n]):.3f}" for n in results), flush=True)
    print(f"done in {time.time() - t0:.0f}s\n")

    print(f"{'planner':<14}{'roll/pitch RMSE':>18}{'within 0.2 rad':>16}{'wins vs hold':>14}")
    print("-" * 62)
    hold = np.array(results["hold still"])
    for name in results:
        arr = np.array(results[name])
        within = np.mean([r[name + "_within"] for r in per_target])
        wins = np.mean(arr < hold) if name != "hold still" else float("nan")
        print(f"{name:<14}{arr.mean():>12.3f} ± {arr.std():<4.3f}{within * 100:>14.1f}%"
              + (f"{wins * 100:>13.0f}%" if name != "hold still" else ""))

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "mimic.json").write_text(json.dumps(
        {"summary": {n: {"rmse_mean": float(np.mean(v)), "rmse_sd": float(np.std(v))}
                     for n, v in results.items()}, "per_target": per_target,
         "config": vars(args)}, indent=1))



if __name__ == "__main__":
    main()
