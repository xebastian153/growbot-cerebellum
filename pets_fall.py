"""PETS vs deterministic planner on fall recovery -- the noisy regime where uncertainty should matter.

Same 60 "tipped" starts and 4 s budget as fall_recovery.py; hold-still, a single MLP,
the ensemble mean, and PETS with 8 and 16 particles. Writes results/pets_fall.json.
"""
import argparse, json, numpy as np, mujoco
import fall_recovery as FR
from growbot_cerebellum import provenance
from growbot_cerebellum.paths import DATA, RESULTS
from growbot_cerebellum.forward import MLP, make_windows, K
from growbot_cerebellum.planner import Imagination
from pets import ProbEnsemble, PETSImagination, cem_plan_pets
from growbot_cerebellum.sim import GrowBotSim


def bucket_of(o): r = abs(o[0]); return "tipped" if r < 1.2 else ("side" if r < 2.4 else "back")


def main():
    ap = argparse.ArgumentParser(
        description="Does planning through PETS particles help fall recovery, the regime where the "
                    "ensemble's uncertainty is largest? Trains one MLP and a 5-net probabilistic "
                    "ensemble, then scores hold-still / single MLP / ensemble mean / PETS 8 and 16 "
                    "particles on 60 tipped starts with a 4 s budget. Writes results/pets_fall.json "
                    "(tee the output to results/logs/pets_fall.txt).")
    ap.parse_args()

    tr = np.load(DATA / "train.npz"); Xtr, Ytr, *_ = make_windows(tr["obs"], tr["act"], tr["next_obs"], tr["done"], K)
    det = MLP(hidden=128, epochs=40).fit(Xtr, Ytr); ens = ProbEnsemble(E=5, epochs=40).fit(Xtr, Ytr)
    rng = np.random.default_rng(0)
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
    res["provenance"] = provenance(seeds={"sim": "5000 + i"})
    json.dump(res, open(RESULTS / "pets_fall.json", "w"), indent=1)


if __name__ == "__main__":
    main()
