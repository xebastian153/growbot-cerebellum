"""Multi-step training loss: does teaching the forward model to roll itself out reduce drift?

The shipped model is trained on one-tick deltas and then unrolled 25 ticks at test
time on its own predictions; it never sees its errors compound. SPR's point:
train through the unroll. Same 128x2 MLP, same predict(X) interface, same evaluation
(forward.rollout_error at 100 / 500 ms), only the loss changes:

  H=1   one-step (baseline, what ships)
  H=5   unroll 5 ticks (100 ms) with gradient through the rollout, loss on every step
  H=10  unroll 10 ticks (200 ms)

Sequences never cross episode boundaries. Angles are (sin, cos) and re-normalised
after each imagined step exactly as at inference, so train and test see the same map.
"""
from __future__ import annotations
import argparse, json, sys, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, ".")
from forward import encode_obs, rollout_error
from sim2real_proxy import K

def sequences(obs, act, next_obs, done, H):
    """windows for multi-step training: hist F (N,K,9), hist A (N,K,2), future A (N,H,2), future F (N,H,9)."""
    F = encode_obs(obs); F2 = encode_obs(next_obs); N = len(obs)
    ok = np.ones(N, bool)
    for j in range(K): ok &= np.roll(~done, j + 1)      # no episode end inside the K-1 ticks before t
    for j in range(H): ok &= np.roll(~done, -j)         # none inside t .. t+H-1  (t+H-1 -> t+H is the last transition)
    ok[:K] = False; ok[N - H:] = False
    idx = np.flatnonzero(ok)
    hf = np.stack([F[idx - k] for k in range(K)], 1)            # newest first
    ha = np.stack([act[idx - k] for k in range(K)], 1)
    fa = np.stack([act[idx + h] for h in range(H)], 1)          # action executed during tick t+h
    ff = np.stack([F2[idx + h] for h in range(H)], 1)           # obs after tick t+h
    return hf.astype(np.float32), ha.astype(np.float32), fa.astype(np.float32), ff.astype(np.float32)

class MultiStepMLP:
    """Same net and predict() as forward.MLP; trained through an H-step unroll."""
    name = "multistep"
    def __init__(self, H=5, hidden=128, epochs=40, lr=2e-3, seed=0, batch=512):
        self.H, self.hidden, self.epochs, self.lr, self.seed, self.batch = H, hidden, epochs, lr, seed, batch

    def _build(self, in_dim, out_dim):
        torch.manual_seed(self.seed)
        self.net = nn.Sequential(nn.Linear(in_dim, self.hidden), nn.SiLU(),
                                 nn.Linear(self.hidden, self.hidden), nn.SiLU(),
                                 nn.Linear(self.hidden, out_dim))
        self.n_params = sum(p.numel() for p in self.net.parameters())

    def _step(self, win_flat):
        x = (win_flat - self.mu_t) / self.sd_t
        return self.net(x) * self.ysd_t + self.ymu_t

    def fit(self, obs, act, next_obs, done, log=False):
        hf, ha, fa, ff = sequences(obs, act, next_obs, done, self.H)
        # normalisation stats from one-step windows (same as forward.MLP)
        X1 = np.concatenate([hf, ha], 2).reshape(len(hf), -1)
        Y1 = ff[:, 0] - hf[:, 0]
        self.mu, self.sd = X1.mean(0), X1.std(0) + 1e-6
        self.ymu, self.ysd = Y1.mean(0), Y1.std(0) + 1e-6
        self.mu_t, self.sd_t = torch.tensor(self.mu), torch.tensor(self.sd)
        self.ymu_t, self.ysd_t = torch.tensor(self.ymu), torch.tensor(self.ysd)
        self._build(X1.shape[1], Y1.shape[1])
        hf_t, ha_t, fa_t, ff_t = map(torch.tensor, (hf, ha, fa, ff))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        n = len(hf_t)
        for ep in range(self.epochs):
            perm = torch.randperm(n); tot = 0.0
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]; B = len(idx)
                win = torch.cat([hf_t[idx], ha_t[idx]], 2)          # (B,K,11) newest first
                cur = hf_t[idx, 0].clone()
                loss = 0.0
                for h in range(self.H):
                    win = win.clone(); win[:, 0, 9:] = fa_t[idx, h]
                    d = self._step(win.reshape(B, -1))
                    cur = cur + d
                    # renormalise (sin,cos) pairs, differentiably
                    s, c = cur[:, :3], cur[:, 3:6]
                    nrm = torch.sqrt(s * s + c * c) + 1e-9
                    cur = torch.cat([s / nrm, c / nrm, cur[:, 6:]], 1)
                    tgt = ff_t[idx, h]
                    loss = loss + (((cur - tgt) / self.ysd_t) ** 2).mean()
                    win = torch.roll(win, 1, dims=1); win = win.clone(); win[:, 0, :9] = cur; win[:, 0, 9:] = 0.0
                loss = loss / self.H
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss) * B
            sched.step()
            if log and (ep + 1) % 10 == 0:
                print(f"    H={self.H} epoch {ep + 1:>3}  loss {tot / n:.4f}", flush=True)
        return self

    def predict(self, X):
        with torch.no_grad():
            return self._step(torch.tensor(X, dtype=torch.float32)).numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--Hs", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 25, 50])
    args = ap.parse_args()
    tr = np.load("data/train.npz"); te = np.load("data/test.npz")
    res = {H: {h: [] for h in args.horizons} for H in args.Hs}
    times = {}
    for seed in args.seeds:
        for H in args.Hs:
            t0 = time.time()
            m = MultiStepMLP(H=H, epochs=args.epochs, seed=seed).fit(tr["obs"], tr["act"], tr["next_obs"], tr["done"], log=(seed == args.seeds[0]))
            times[H] = time.time() - t0
            ro = rollout_error(m, te["obs"], te["act"], te["done"], K, args.horizons, seed=seed)
            for h in args.horizons: res[H][h].append(ro[h]["within_0.2rad"])
            print(f"seed {seed} H={H:<2} " + "  ".join(f"@{h*20}ms {ro[h]['within_0.2rad']*100:5.1f}%" for h in args.horizons) + f"   ({times[H]:.0f}s)", flush=True)
    print(f"\n{'train unroll':<14}" + "".join(f"{'@' + str(h*20) + ' ms':>16}" for h in args.horizons) + f"{'fit s':>8}   within 0.2 rad, mean ± sd over {len(args.seeds)} seeds")
    print("-" * (14 + 16 * len(args.horizons) + 8))
    for H in args.Hs:
        print(f"H={H:<12}" + "".join(f"{np.mean(res[H][h])*100:>10.1f} ± {np.std(res[H][h])*100:<3.1f}" for h in args.horizons) + f"{times[H]:>8.0f}")
    json.dump({"epochs": args.epochs, "seeds": args.seeds, "horizons": args.horizons,
               "within": {str(H): {str(h): v for h, v in d.items()} for H, d in res.items()}}, open("results/multistep.json", "w"), indent=1)

if __name__ == "__main__":
    main()
