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
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "sim"))
from growbot_sim import GrowBotSim, CTRL_HZ                     # noqa: E402
from forward import (Persistence, Linear, MLP, make_windows,     # noqa: E402
                     encode_obs, decode_obs)

HERE = Path(__file__).parent


class Imagination:
    """Batched open-loop rollout of a forward model from a shared history window."""

    def __init__(self, model, K):
        self.model, self.K = model, K

    def rollout(self, F_hist, A_hist, plans):
        """F_hist: (K,9) newest-first, A_hist: (K,2) newest-first, plans: (N,H,2).
        Returns imagined encoded obs (N,H,9)."""
        N, H, _ = plans.shape
        fdim = F_hist.shape[1]
        win = np.zeros((N, self.K, fdim + 2), np.float32)
        win[:, :, :fdim] = F_hist[None]
        win[:, :, fdim:] = A_hist[None]
        cur = np.repeat(F_hist[0][None], N, axis=0)
        out = np.zeros((N, H, fdim), np.float32)
        for h in range(H):
            # the action of *this* tick sits in slot 0 alongside the current obs
            win[:, 0, fdim:] = plans[:, h]
            d = self.model.predict(win.reshape(N, -1))
            cur = cur + d
            for a in range(3):
                n = np.sqrt(cur[:, a] ** 2 + cur[:, a + 3] ** 2) + 1e-9
                cur[:, a] /= n; cur[:, a + 3] /= n
            out[:, h] = cur
            win = np.roll(win, 1, axis=1)
            win[:, 0, :fdim] = cur
        return out


def angle_cost(imagined, target):
    """Mean squared roll/pitch angle error over the horizon, per candidate."""
    pa = decode_obs(imagined)[..., :2]
    ta = decode_obs(target)[None, :, :2]
    e = np.arctan2(np.sin(pa - ta), np.cos(pa - ta))
    return (e ** 2).mean(axis=(1, 2))


def cem_plan(imag, F_hist, A_hist, target, H, n=256, iters=4, elite=32, rng=None,
             init_mean=None, smooth=0.6):
    rng = rng or np.random.default_rng(0)
    mean = np.zeros((H, 2), np.float32) if init_mean is None else init_mean.copy()
    std = np.full((H, 2), 0.6, np.float32)
    for _ in range(iters):
        raw = rng.normal(size=(n, H, 2)).astype(np.float32) * std + mean
        # actions are servo targets; smooth candidates so plans are physically sane
        for h in range(1, H):
            raw[:, h] = smooth * raw[:, h - 1] + (1 - smooth) * raw[:, h]
        plans = np.clip(raw, -1.4, 1.4)
        cost = angle_cost(imag.rollout(F_hist, A_hist, plans), target)
        best = plans[np.argsort(cost)[:elite]]
        mean = best.mean(0); std = best.std(0) + 0.02
    return mean


def run_episode(sim, imag, target_obs, K, H_plan, replan_every, rng, no_op=False):
    """Execute in the twin, replanning every `replan_every` ticks with a receding horizon."""
    T = len(target_obs)
    F_t = encode_obs(target_obs)
    executed = np.zeros((T, 6), np.float32)
    o = sim.obs()
    F_hist = np.repeat(encode_obs(o)[None], K, axis=0)
    A_hist = np.zeros((K, 2), np.float32)
    plan = np.zeros((H_plan, 2), np.float32)
    t = 0
    while t < T:
        if not no_op and t % replan_every == 0:
            H = min(H_plan, T - t)
            tgt = F_t[t:t + H]
            init = np.concatenate([plan[replan_every:], np.repeat(plan[-1:], replan_every, 0)])[:H]
            plan = np.zeros((H_plan, 2), np.float32)
            plan[:H] = cem_plan(imag, F_hist, A_hist, tgt, H, rng=rng, init_mean=init)
        a = np.zeros(2, np.float32) if no_op else plan[t % replan_every]
        o = sim.step(a)
        executed[t] = o
        F_hist = np.roll(F_hist, 1, axis=0); F_hist[0] = encode_obs(o)
        A_hist = np.roll(A_hist, 1, axis=0); A_hist[0] = a
        t += 1
    ea, ta = executed[:, :2], target_obs[:, :2]
    e = np.arctan2(np.sin(ea - ta), np.cos(ea - ta))
    return {"rmse": float(np.sqrt((e ** 2).mean())),
            "within_0.2": float((np.abs(e).max(1) < 0.2).mean()),
            "executed": executed}


def pick_targets(te, T, n, rng, min_motion=0.15):
    """Held-out segments with real motion (not still, not fallen the whole time)."""
    obs, done = te["obs"], te["done"]
    N = len(obs)
    starts = []
    tries = 0
    while len(starts) < n and tries < 20000:
        tries += 1
        s = int(rng.integers(0, N - T - 1))
        seg = obs[s:s + T]
        if done[s:s + T - 1].any():
            continue
        if (np.abs(seg[:, :2]) > 1.2).mean() > 0.3:
            continue
        if np.ptp(seg[:, 0]) + np.ptp(seg[:, 1]) < min_motion:
            continue
        starts.append(s)
    return [obs[s:s + T].copy() for s in starts], [te["mode"][s] for s in starts]


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


def rpy_to_quat(roll, pitch, yaw):
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


if __name__ == "__main__":
    main()
