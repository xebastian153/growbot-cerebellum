"""Does metadata conditioning help the forward model? (the pi0.7 metadata finding, tested here)

The pi0.7 technical report found that prompting the policy with metadata about the data
(quality, episode length) turned low-quality data from harmful into helpful. The
forward model has an analogue that costs nothing at deployment: the harness knows
which verb is running (walk policy, gesture, rest) and which body it is on. Append
that as one-hot input and ask three questions on the same windows and metric:

  Q1  single body: does knowing the excitation mode help?
  Q2  two bodies pooled: does body+mode metadata let walk and Olie data share a model
      without hurting either, versus one model per body?
  Q3  the pi0.7 curve: add progressively "less informative" excitation (OU jitter, still)
      to a clean set (policy + sine + keyframe). Without metadata, does it hurt?
      With metadata, does it help?
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
from growbot_cerebellum.forward import MLP, make_windows, encode_obs, decode_obs

MODES = ["keyframe", "sine", "ou", "policy", "still"]
CLEAN = ["policy", "sine", "keyframe"]
K = 5

def load(name):
    d = np.load(f"data/{name}.npz")
    return d["obs"], d["act"], d["next_obs"], d["done"], d["mode"].astype(str)

def meta_cols(mode, body, use_mode, use_body):
    cols = []
    if use_mode: cols.append(np.stack([(mode == m).astype(np.float32) for m in MODES], 1))
    if use_body: cols.append(np.stack([(body == b).astype(np.float32) for b in ("walk", "olie")], 1))
    return np.concatenate(cols, 1) if cols else np.zeros((len(mode), 0), np.float32)

def windows(sets, use_mode, use_body, keep_modes=None):
    Xs, Ys = [], []
    for body, (O, A, O2, D, mode) in sets.items():
        X, Y, *_, valid = make_windows(O, A, O2, D, K)
        m = mode[valid]; b = np.full(len(m), body)
        if keep_modes is not None:
            sel = np.isin(m, keep_modes); X, Y, m, b = X[sel], Y[sel], m[sel], b[sel]
        Xs.append(np.concatenate([X, meta_cols(m, b, use_mode, use_body)], 1)); Ys.append(Y)
    return np.concatenate(Xs), np.concatenate(Ys)

def evaluate(model, body, data, use_mode, use_body, horizons=(5, 25), n=2000, seed=0):
    O, A, D, mode = data[0], data[1], data[3], data[4]
    F = encode_obs(O); N = len(O); fdim = F.shape[1]; Hmax = max(horizons)
    ok = np.ones(N, bool)
    for j in range(K): ok &= np.roll(~D, j + 1)
    for j in range(Hmax): ok &= np.roll(~D, -j)
    ok[:K] = False; ok[N - Hmax - 1:] = False
    rng = np.random.default_rng(seed)
    starts = rng.choice(np.flatnonzero(ok), size=min(n, ok.sum()), replace=False)
    meta = meta_cols(mode[starts], np.full(len(starts), body), use_mode, use_body)
    win = np.zeros((len(starts), K, fdim + 2), np.float32)
    for k in range(K): win[:, k, :fdim] = F[starts - k]; win[:, k, fdim:] = A[starts - k]
    cur = F[starts].copy(); out = {}
    for h in range(1, Hmax + 1):
        win[:, 0, fdim:] = A[starts + h - 1]
        X = np.concatenate([win.reshape(len(starts), -1), meta], 1)
        cur = cur + model.predict(X)
        for a in range(3):
            nn_ = np.sqrt(cur[:, a] ** 2 + cur[:, a + 3] ** 2) + 1e-9; cur[:, a] /= nn_; cur[:, a + 3] /= nn_
        win = np.roll(win, 1, axis=1); win[:, 0, :fdim] = cur
        if h in horizons:
            pa, ta = decode_obs(cur)[:, :2], decode_obs(F[starts + h])[:, :2]
            e = np.arctan2(np.sin(pa - ta), np.cos(pa - ta))
            out[h] = float((np.abs(e).max(1) < 0.2).mean())
    return out

def fit(X, Y, epochs, seed):
    return MLP(hidden=128, epochs=epochs, seed=seed).fit(X, Y)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = ap.parse_args()
    walk = {"train": load("train"), "test": load("test")}
    olie = {"train": load("olie_train"), "test": load("olie_test")}
    results = {}
    def report(tag, rows):
        results[tag] = rows
        print(f"\n== {tag} ==")
        for name, r in rows.items():
            print(f"  {name:<44}" + "  ".join(f"{k}: {v[5]*100:5.1f}% / {v[25]*100:5.1f}%" for k, v in r.items()))
    t0 = time.time()

    for seed in args.seeds:
        print(f"\n######## seed {seed} ########", flush=True)
        # ---------------- Q1: single body, +mode ----------------
        rows = {}
        for use_mode in (False, True):
            X, Y = windows({"walk": walk["train"]}, use_mode, False)
            m = fit(X, Y, args.epochs, seed)
            rows[f"walk only, meta={'mode' if use_mode else 'none'}"] = {"walk": evaluate(m, "walk", walk["test"], use_mode, False)}
        report(f"Q1 seed{seed}: single body (within 0.2 rad @100ms / @500ms)", rows)

        # ---------------- Q2: pooled bodies ----------------
        rows = {}
        for body, d in (("walk", walk), ("olie", olie)):
            X, Y = windows({body: d["train"]}, False, False); m = fit(X, Y, args.epochs, seed)
            rows[f"per-body model ({body})"] = {body: evaluate(m, body, d["test"], False, False)}
        for use_mode, use_body in ((False, False), (False, True), (True, True)):
            X, Y = windows({"walk": walk["train"], "olie": olie["train"]}, use_mode, use_body)
            m = fit(X, Y, args.epochs, seed)
            tag = f"pooled, meta={'+'.join([n for n, u in (('mode', use_mode), ('body', use_body)) if u]) or 'none'}"
            rows[tag] = {"walk": evaluate(m, "walk", walk["test"], use_mode, use_body),
                         "olie": evaluate(m, "olie", olie["test"], use_mode, use_body)}
        report(f"Q2 seed{seed}: two bodies pooled", rows)

        # ---------------- Q3: the pi0.7 curve ----------------
        rows = {}
        for use_mode in (False, True):
            for label, keep in (("clean (policy+sine+keyframe)", CLEAN), ("clean + OU", CLEAN + ["ou"]), ("clean + OU + still (all)", MODES)):
                X, Y = windows({"walk": walk["train"]}, use_mode, False, keep_modes=keep)
                m = fit(X, Y, args.epochs, seed)
                # evaluate on ALL test modes and on the clean subset only
                O, A, O2, D, mode = walk["test"]
                sel = np.isin(mode, CLEAN)
                # a clean-only test set: mask done so windows never cross into other modes
                D2 = D | ~sel
                rows[f"meta={'mode' if use_mode else 'none'} | train {label} ({len(X):,})"] = {
                    "all-test": evaluate(m, "walk", walk["test"], use_mode, False),
                    "clean-test": evaluate(m, "walk", (O, A, O2, D2, mode), use_mode, False)}
        report(f"Q3 seed{seed}: adding less-informative excitation", rows)
    print(f"\n{time.time()-t0:.0f}s")
    json.dump(results, open("results/metadata_experiment.json", "w"), indent=1)

if __name__ == "__main__":
    main()
