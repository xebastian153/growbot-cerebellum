"""Train and score the three forward models on the twin data: persistence, linear, MLP.

Writes results/forward_K<K>.json -- the artifact behind the headline table. The models,
`make_windows`, `rollout_error` and `by_regime` live in `growbot_cerebellum.forward`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from growbot_cerebellum import provenance
from growbot_cerebellum.forward import CTRL_HZ, MLP, Linear, Persistence, by_regime, make_windows, rollout_error

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=5, help="history window (ticks)")
    ap.add_argument("--epochs", type=int, default=80, help="the value behind every published number")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 25, 50])
    ap.add_argument("--regime-horizons", type=int, nargs="+", default=[5, 25],
                    help="horizons of the per-regime table (100 ms and 500 ms)")
    args = ap.parse_args()

    tr = np.load(HERE / "data" / "train.npz"); te = np.load(HERE / "data" / "test.npz")
    Xtr, Ytr, _, _, _ = make_windows(tr["obs"], tr["act"], tr["next_obs"], tr["done"], args.K)
    Xte, Yte, _, _, _ = make_windows(te["obs"], te["act"], te["next_obs"], te["done"], args.K)
    print(f"windows K={args.K}: train {len(Xtr):,}  test {len(Xte):,}  "
          f"input dim {Xtr.shape[1]}  target dim {Ytr.shape[1]}")

    models = [Persistence(), Linear(), MLP(hidden=args.hidden, epochs=args.epochs)]
    results = {}
    for m in models:
        t0 = time.time()
        m.fit(Xtr, Ytr, log=True) if isinstance(m, MLP) else m.fit(Xtr, Ytr)
        one = m.predict(Xte)
        rmse1 = float(np.sqrt(((one - Yte) ** 2).mean()))
        ro = rollout_error(m, te["obs"], te["act"], te["done"], args.K, args.horizons)
        results[m.name] = {"one_step_rmse_delta": rmse1, "rollout": ro,
                           "fit_s": round(time.time() - t0, 1),
                           "params": getattr(m, "n_params", None)}
        print(f"\n{m.name:<12} one-step delta RMSE {rmse1:.4f}   fit {time.time() - t0:.1f}s"
              + (f"   params {m.n_params:,}" if hasattr(m, "n_params") else ""))
        print(f"  {'horizon':>8}{'ms':>6}{'roll/pitch RMSE':>18}{'yaw RMSE':>11}{'gyro RMSE':>12}{'within 0.2':>12}{'yaw w0.2':>10}")
        for h in args.horizons:
            r = ro[h]
            print(f"  {h:>8}{h * 1000 // CTRL_HZ:>6}{r['rmse_rollpitch_rad']:>18.4f}"
                  f"{r['rmse_axis_rad']['yaw']:>11.4f}{r['rmse_gyro_rads']:>12.3f}"
                  f"{r['within_0.2rad'] * 100:>11.1f}%{r['within_0.2rad_axis']['yaw'] * 100:>9.1f}%")

    # Per regime: the same rollout split by what the body was doing at the start
    # (by_regime samples up to 1500 starts per regime with a fixed seed).
    results["by_regime"] = {}
    for h in args.regime_horizons:
        table = {m.name: {name: {"n": n, "rmse_rollpitch_rad": rmse, "within_0.2rad": w}
                          for name, n, rmse, w in by_regime(m, te, args.K, h=h)}
                 for m in models}
        results["by_regime"][str(h)] = table
        regimes = list(next(iter(table.values())).keys())
        print(f"\nby regime @ {h * 1000 // CTRL_HZ} ms, within 0.2 rad (n starts)")
        print(f"  {'regime':<18}{'n':>6}" + "".join(f"{m.name:>14}" for m in models))
        for name in regimes:
            n = table[models[0].name][name]["n"]
            print(f"  {name:<18}{n:>6}" + "".join(f"{table[m.name][name]['within_0.2rad'] * 100:>13.1f}%" for m in models))

    (HERE / "results").mkdir(exist_ok=True)
    results["provenance"] = provenance(seeds={"mlp": 0, "rollout": 0, "regime_starts": 0})
    (HERE / "results" / f"forward_K{args.K}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
