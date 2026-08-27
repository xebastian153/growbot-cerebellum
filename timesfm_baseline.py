"""TimesFM 2.5 as an action-blind forecasting baseline on the twin IMU.

Question: how much of the next 100-500 ms is predictable from the sensor history
alone, by the strongest generic forecaster available, versus a small model that
knows what the legs were told to do? Same windows, same metric as forward.py.
"""
import time, json, argparse
import numpy as np, timesfm
from pathlib import Path
from growbot_cerebellum.forward import make_windows, MLP, Linear, Persistence, encode_obs, decode_obs

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=400); ap.add_argument("--ctx", type=int, default=256)
ap.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 25]); args = ap.parse_args()
K = 5; Hmax = max(args.horizons)

te = np.load("data/test.npz"); tr = np.load("data/train.npz")
obs, act, done = te["obs"], te["act"], te["done"]; N = len(obs)
ok = np.ones(N, bool)
for j in range(args.ctx): ok &= np.roll(~done, j + 1)      # full context inside one episode
for j in range(Hmax): ok &= np.roll(~done, -j)
ok[:args.ctx] = False; ok[N - Hmax - 1:] = False
rng = np.random.default_rng(0)
starts = rng.choice(np.flatnonzero(ok), size=min(args.n, ok.sum()), replace=False)
print(f"{len(starts)} windows, ctx {args.ctx} ticks, horizons {args.horizons}")

# --- TimesFM, per-channel univariate, action-blind ---
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
model.compile(timesfm.ForecastConfig(max_context=max(args.ctx, 64), max_horizon=Hmax, normalize_inputs=True, use_continuous_quantile_head=False))
t0 = time.time(); tfm = np.zeros((len(starts), Hmax, 6), np.float32)
B = 32
for i in range(0, len(starts), B):
    idx = starts[i:i + B]
    inputs = [obs[s - args.ctx:s, j].astype(np.float32) for s in idx for j in range(6)]
    point, _ = model.forecast(horizon=Hmax, inputs=inputs)
    tfm[i:i + len(idx)] = np.asarray(point).reshape(len(idx), 6, Hmax).transpose(0, 2, 1)
    if (i // B) % 3 == 0: print(f"  {i + len(idx)}/{len(starts)}  {time.time() - t0:.0f}s", flush=True)
print(f"timesfm done in {time.time() - t0:.0f}s ({(time.time() - t0) / len(starts) * 1000:.0f} ms/window)")

# --- our models, action-aware, same starts (uses forward.rollout convention) ---
Xtr, Ytr, *_ = make_windows(tr["obs"], tr["act"], tr["next_obs"], tr["done"], K)
ours = {"persistence": Persistence().fit(Xtr, Ytr), "linear": Linear().fit(Xtr, Ytr), "mlp": MLP(hidden=128, epochs=80).fit(Xtr, Ytr)}
F = encode_obs(obs); fdim = F.shape[1]
def rollout(m):
    win = np.zeros((len(starts), K, fdim + 2), np.float32)
    for k in range(K): win[:, k, :fdim] = F[starts - k]; win[:, k, fdim:] = act[starts - k]
    cur = F[starts].copy(); out = np.zeros((len(starts), Hmax, 6), np.float32)
    for h in range(1, Hmax + 1):
        win[:, 0, fdim:] = act[starts + h - 1]
        cur = cur + m.predict(win.reshape(len(starts), -1))
        for a in range(3):
            n = np.sqrt(cur[:, a] ** 2 + cur[:, a + 3] ** 2) + 1e-9; cur[:, a] /= n; cur[:, a + 3] /= n
        out[:, h - 1] = decode_obs(cur); win = np.roll(win, 1, axis=1); win[:, 0, :fdim] = cur
    return out
preds = {"timesfm 2.5 (action-blind)": tfm, **{k: rollout(m) for k, m in ours.items()}}

def score(P, h):
    pa = P[:, h - 1, :2]; ta = obs[starts + h, :2]
    e = np.arctan2(np.sin(pa - ta), np.cos(pa - ta))
    return float(np.sqrt((e ** 2).mean())), float((np.abs(e).max(1) < 0.2).mean())
print(f"\n{'model':<28}" + "".join(f"{'@' + str(h * 20) + 'ms RMSE':>13}{'within.2':>10}" for h in args.horizons))
print("-" * (28 + 23 * len(args.horizons)))
res = {}
for name, P in preds.items():
    row = [score(P, h) for h in args.horizons]; res[name] = row
    print(f"{name:<28}" + "".join(f"{r:>13.4f}{w * 100:>9.1f}%" for r, w in row))
Path("results").mkdir(exist_ok=True); Path("results/timesfm_baseline.json").write_text(json.dumps({"horizons": args.horizons, "n": len(starts), "ctx": args.ctx, "rows": res}, indent=1))
