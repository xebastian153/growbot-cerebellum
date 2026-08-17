"""PETS vs deterministic planner on fall recovery -- the noisy regime where uncertainty should matter."""
import sys, json, numpy as np, mujoco; sys.path.insert(0,'.'); sys.path.insert(0,'sim')
import fall_recovery as FR
from forward import MLP, make_windows
from sim2real_proxy import K
from mimic import Imagination
from pets import ProbEnsemble, PETSImagination, cem_plan_pets
from growbot_sim import GrowBotSim
tr = np.load("data/train.npz"); Xtr, Ytr, *_ = make_windows(tr["obs"], tr["act"], tr["next_obs"], tr["done"], K)
det = MLP(hidden=128, epochs=40).fit(Xtr, Ytr); ens = ProbEnsemble(E=5, epochs=40).fit(Xtr, Ytr)
rng = np.random.default_rng(0)
def bucket_of(o): r = abs(o[0]); return "tipped" if r < 1.2 else ("side" if r < 2.4 else "back")
starts=[]; i=0
while len(starts) < 60 and i < 1200:
    sim = GrowBotSim(seed=5000+i); o = FR.make_fallen(sim, rng); i+=1
    if bucket_of(o) != "tipped": continue
    starts.append((sim.d.qpos.copy(), sim.d.qvel.copy()))
def trial(policy, imag, planner=None):
    saved = FR.cem_plan
    if planner: FR.cem_plan = planner
    try:
        rec=[]
        for j,(qp,qv) in enumerate(starts):
            sim = GrowBotSim(seed=5000+j); sim.d.qpos[:]=qp; sim.d.qvel[:]=qv; mujoco.mj_forward(sim.m, sim.d)
            ok,_ = FR.run(sim, policy, 200, K, np.random.default_rng(j), imag=imag); rec.append(ok)
    finally: FR.cem_plan = saved
    return float(np.mean(rec))
res = {}
print("60 tipped starts, 4 s budget:", flush=True)
for name, args in [("hold still", ('still', None)), ("plan, single MLP (mean)", ('plan', Imagination(det, K))),
                   ("plan, ensemble mean", ('plan', Imagination(ens, K)))]:
    res[name] = trial(*args); print(f"  {name:<28} {res[name]*100:5.1f}%", flush=True)
for P in (8, 16):
    name = f"plan, PETS {P} particles"; res[name] = trial('plan', PETSImagination(ens, K, P, 0), cem_plan_pets)
    print(f"  {name:<28} {res[name]*100:5.1f}%", flush=True)
json.dump(res, open("results/pets_fall.json", "w"), indent=1)
