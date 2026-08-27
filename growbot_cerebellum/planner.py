"""Imagination and the CEM planner: search through the forward model for the action
chunk that reproduces a target motion. `mimic.py` plays the game; `fall_recovery.py` plans
from fallen states with the same two pieces.
"""
from __future__ import annotations
import numpy as np
from .forward import encode_obs, decode_obs


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


def rpy_to_quat(roll, pitch, yaw):
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])
