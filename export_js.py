"""Export the forward model in the GrowBot policy convention: a weights JSON plus
reference vectors, so growbot_forward.js can be verified against the trained net.

Output layout mirrors policy_85mm.json (mean/std normalisation, layers with W/b/act)
so a reader of one file understands the other.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from forward import MLP, make_windows, encode_obs

HERE = Path(__file__).parent
OUT = HERE / "forward-model"
K = 5


def main():
    ap = argparse.ArgumentParser(
        description="Train the forward model (128x2, K=5, 80 epochs) on data/train.npz and export "
                    "it in the GrowBot policy-JSON convention, plus reference vectors for the JS "
                    "equivalence test. Writes <out>/forward_85mm.json and "
                    "<out>/reference_vectors.json; the shipped copies live in forward-model/. "
                    "Training is seeded (torch and numpy) so the export is reproducible.")
    ap.add_argument("--seed", type=int, default=0, help="training seed (default 0)")
    ap.add_argument("--out", default=str(OUT), help=f"output directory (default {OUT})")
    args = ap.parse_args()
    out = Path(args.out)
    tr = np.load(HERE / "data" / "train.npz")
    te = np.load(HERE / "data" / "test.npz")
    Xtr, Ytr, *_ = make_windows(tr["obs"], tr["act"], tr["next_obs"], tr["done"], K)
    Xte, Yte, *_ = make_windows(te["obs"], te["act"], te["next_obs"], te["done"], K)

    m = MLP(hidden=128, layers=2, epochs=80, seed=args.seed).fit(Xtr, Ytr)   # seeds torch inside

    layers = []
    lin = [mod for mod in m.net if hasattr(mod, "weight")]
    for i, mod in enumerate(lin):
        layers.append({
            "W": mod.weight.detach().numpy().T.round(7).tolist(),   # (in, out) like the policy file
            "b": mod.bias.detach().numpy().round(7).tolist(),
            "act": "swish" if i < len(lin) - 1 else "linear",
        })

    doc = {
        "note": ("GrowBot 85mm forward model (physical imagination). Predicts the change in the "
                 "phone IMU over one 20 ms tick from the last 5 (imu, action) pairs, newest first. "
                 "imu9 = [sin r, sin p, sin y, cos r, cos p, cos y, gyro_r, gyro_p, gyro_y]; "
                 "action2 = [aRight, aLeft] radians of leg swing, same units the walk policy emits. "
                 "Trained on the MuJoCo twin in policy/Harsh_policies/DR_RMA_EXPORT."),
        "obs_size": int(Xtr.shape[1]),
        "out_size": int(Ytr.shape[1]),
        "history": K,
        "tick_ms": 20,
        "activation": "swish",
        "in_mean": m.mu.round(7).tolist(),
        "in_std": m.sd.round(7).tolist(),
        "out_mean": m.ymu.round(7).tolist(),
        "out_std": m.ysd.round(7).tolist(),
        "layers": layers,
    }
    out.mkdir(exist_ok=True)
    (out / "forward_85mm.json").write_text(json.dumps(doc, separators=(",", ":")))

    # reference vectors: raw windows in, predicted delta out, plus a 25-step rollout
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(Xte), size=64, replace=False)
    single = {"x": Xte[idx].round(6).tolist(), "y": m.predict(Xte[idx]).round(6).tolist()}

    # rollout reference: one start, 25 known actions, imagined encoded obs each step
    obs, act, done = te["obs"], te["act"], te["done"]
    F = encode_obs(obs)
    ok = np.ones(len(obs), bool)
    for j in range(K): ok &= np.roll(~done, j + 1)
    for j in range(25): ok &= np.roll(~done, -j)
    ok[:K] = False; ok[-30:] = False
    s = int(rng.choice(np.flatnonzero(ok)))
    # history is (obs, action) pairs newest first; the action in slot 0 is the one
    # executed *during* the current tick, so imagining a plan overwrites slot 0
    # with plan[h] before each prediction -- same convention as mimic.Imagination
    hist_f = F[s - np.arange(K)]; hist_a = act[s - np.arange(K)]
    win = np.concatenate([hist_f, hist_a], 1).reshape(1, -1).astype(np.float32).copy()
    cur = F[s].copy(); traj = []
    plan = act[s:s + 25]
    for h in range(25):
        win = win.reshape(K, 11); win[0, 9:] = plan[h]; win = win.reshape(1, -1)
        cur = cur + m.predict(win)[0]
        for a in range(3):
            n = np.sqrt(cur[a] ** 2 + cur[a + 3] ** 2) + 1e-9
            cur[a] /= n; cur[a + 3] /= n
        traj.append(cur.round(6).tolist())
        win = np.roll(win.reshape(K, 11), 1, axis=0)
        win[0, :9] = cur; win[0, 9:] = 0.0
        win = win.reshape(1, -1)
    rollout = {"hist_imu9": hist_f.round(6).tolist(), "hist_action": hist_a.round(6).tolist(),
               "plan": plan.round(6).tolist(), "imagined": traj}
    (out / "reference_vectors.json").write_text(json.dumps({"single": single, "rollout": rollout}))

    kb = (out / "forward_85mm.json").stat().st_size / 1024
    print(f"wrote {out}/forward_85mm.json ({kb:.0f} KB, {m.n_params:,} params) and reference_vectors.json")


if __name__ == "__main__":
    main()
