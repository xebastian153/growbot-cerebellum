"""Fall recovery through imagination: a natural use of the mimic module.

Set the target to a steady upright pose, start from a fallen or tipped body, and let
the planner (frozen forward model + CEM over 300 ms chunks, replan every 100 ms) try
to converge. Score against two baselines that need no model: hold still (servos to
neutral) and a scripted "wiggle" (alternating legs, the reflex a person would code).

Success = |roll| < 0.35 and |pitch| < 0.35 rad held for 0.5 s within 4 s.
Start states are sampled by tipping the body from rest with random pushes and hard
initial leans until it is past 1.0 rad on roll or pitch, so they are real fallen
configurations the physics produced, not hand-placed ones.
"""
from __future__ import annotations
import argparse, json, time
import numpy as np, mujoco
from growbot_cerebellum import provenance
from growbot_cerebellum.sim import GrowBotSim, CTRL_HZ
from growbot_cerebellum.forward import MLP, make_windows, encode_obs, K
from growbot_cerebellum.planner import Imagination, cem_plan

UP = np.array([0.0, -0.6, 0.0], np.float32)   # the settled stance the twin rests in (pitch ~ -0.6 rad)

def upright(o): return abs(o[0]) < 0.35 and abs(o[1] - UP[1]) < 0.35

def make_fallen(sim, rng, tries=50):
    for _ in range(tries):
        sim.reset(tilt=1.2)
        for _ in range(int(rng.integers(5, 40))):
            if rng.random() < 0.3: sim.push(scale=1.5)
            sim.step(rng.uniform(-1.2, 1.2, 2))
        o = sim.obs()
        if abs(o[0]) > 1.0 or abs(o[1] - UP[1]) > 1.0:
            return o
    return sim.obs()

def run(sim, policy, T, K_, rng, imag=None, H=15, replan=5):
    """policy: 'plan' | 'still' | 'wiggle'. Returns (recovered, ticks_to_recover)."""
    o = sim.obs(); F_hist = np.repeat(encode_obs(o)[None], K_, 0); A_hist = np.zeros((K_, 2), np.float32)
    plan = np.zeros((H, 2), np.float32); good = 0
    tgt_obs = np.tile(np.concatenate([UP, np.zeros(3, np.float32)]), (H, 1)).astype(np.float32)
    F_t = encode_obs(tgt_obs)
    for t in range(T):
        if policy == "still": a = np.zeros(2, np.float32)
        elif policy == "wiggle": a = np.array([np.sin(t / 3.0), -np.sin(t / 3.0)], np.float32) * 1.0
        else:
            if t % replan == 0:
                init = np.concatenate([plan[replan:], np.repeat(plan[-1:], replan, 0)])[:H]
                plan = cem_plan(imag, F_hist, A_hist, F_t, H, rng=rng, init_mean=init, n=192, iters=3)
            a = plan[t % replan]
        o = sim.step(a)
        F_hist = np.roll(F_hist, 1, 0); F_hist[0] = encode_obs(o)
        A_hist = np.roll(A_hist, 1, 0); A_hist[0] = a
        good = good + 1 if upright(o) else 0
        if good >= int(0.5 * CTRL_HZ): return True, t + 1
    return False, T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--T", type=int, default=200, help="4 s budget"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", choices=["tipped", "side", "back"], default=None, help="sample starts from one severity bucket only")
    args = ap.parse_args()
    tr = np.load("data/train.npz")
    Xtr, Ytr, *_ = make_windows(tr["obs"], tr["act"], tr["next_obs"], tr["done"], K)
    model = MLP(hidden=128, epochs=args.epochs).fit(Xtr, Ytr); imag = Imagination(model, K)
    rng = np.random.default_rng(args.seed)
    # sample fallen states once, then replay each policy from the SAME state via saved qpos/qvel
    def bucket_of(o):
        r = abs(o[0]); return "tipped" if r < 1.2 else ("side" if r < 2.4 else "back")
    starts = []; i = 0
    while len(starts) < args.n and i < 20 * args.n:
        sim = GrowBotSim(seed=5000 + i); o = make_fallen(sim, rng); i += 1
        if args.only and bucket_of(o) != args.only: continue
        starts.append((sim.d.qpos.copy(), sim.d.qvel.copy(), sim.obs()))
    print(f"{args.n} fallen starts: mean |roll| {np.mean([abs(s[2][0]) for s in starts]):.2f}, mean |pitch-UP| {np.mean([abs(s[2][1]-UP[1]) for s in starts]):.2f} rad")
    # severity buckets: tipped (|roll| < 1.2, i.e. still on its feet-ish, pitched or leaning),
    # side (1.2 <= |roll| < 2.4), back (|roll| >= 2.4)
    def bucket(o):
        r = abs(o[0]); return "tipped" if r < 1.2 else ("side" if r < 2.4 else "back")
    buckets = [bucket(s[2]) for s in starts]
    print("starts by bucket:", {b: buckets.count(b) for b in ("tipped", "side", "back")})
    res = {}
    t0 = time.time()
    outcomes = {p: [] for p in ("still", "wiggle", "plan")}
    for policy in outcomes:
        for i, (qp, qv, _) in enumerate(starts):
            sim = GrowBotSim(seed=5000 + i); sim.d.qpos[:] = qp; sim.d.qvel[:] = qv; mujoco.mj_forward(sim.m, sim.d)
            ok, tt = run(sim, policy, args.T, K, np.random.default_rng(args.seed + i), imag=imag if policy == "plan" else None)
            outcomes[policy].append((buckets[i], ok, tt))
    print(f"\n{'policy':<8}" + "".join(f"{b:>18}" for b in ("tipped", "side", "back")) + f"{'all':>10}")
    print("-" * 66)
    for policy, rows in outcomes.items():
        line = f"{policy:<8}"; res[policy] = {}
        for b in ("tipped", "side", "back"):
            sel = [ok for bb, ok, _ in rows if bb == b]
            v = float(np.mean(sel)) if sel else float("nan"); res[policy][b] = v
            line += f"{v*100:>13.1f}% n={len(sel):<3}" if sel else f"{'n/a':>18}"
        allv = float(np.mean([ok for _, ok, _ in rows])); res[policy]["all"] = allv
        print(line + f"{allv*100:>9.1f}%")
    print(f"{time.time()-t0:.0f}s")
    res["provenance"] = provenance(seeds={"rng": args.seed, "sim": "5000 + i"})
    json.dump(res, open("results/fall_recovery.json", "w"), indent=1)

if __name__ == "__main__":
    main()
