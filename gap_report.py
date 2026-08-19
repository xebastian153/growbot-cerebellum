"""The day-of-log deliverable: gap per regime and per axis, with the twin floor built in.

The forward model's error on a real log is NOT the sim-to-real gap: the model has a
floor on the twin's own held-out physics, and that floor differs by axis (yaw is the
hard one even in sim) and by regime. Quoting the raw real-log error would misread
floor as gap. This report therefore prints three numbers per cell:

    real        the model's error on the log
    twin        the same model, same axis, same horizon, matched regime, on twin test
    gap         real - twin  (positive = worse than the floor explains)

--servo-id adds a fourth column: the real-log error after replaying the commands
through the servo identified from the log itself (servo_id.py), i.e. how much of the
gap the actuator explains.

Regimes come from the log's event rows. Real session names map to the twin's
excitation regimes conservatively (REGIME_MAP); unmapped names fall back to the
twin's overall row, printed as such.
"""
from __future__ import annotations
import argparse, itertools, json, sys
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "sim")
from forward import MLP, make_windows, encode_obs, decode_obs
from sim2real_proxy import K
from imulog import parse
from servo_id import identify, realized_from_commands

AXES = ("roll", "pitch", "yaw")
REGIME_MAP = {"walk": "policy", "spin": "policy", "gesture": "keyframe",
              "handling": "ou", "idle": "still", "still": "still",
              "policy": "policy", "sine": "sine", "keyframe": "keyframe", "ou": "ou"}


def evaluate_axes(model, O, A, D, mode, horizons, n_starts=4000, seed=0):
    """Open-loop rollout; per (regime, horizon, axis) within-0.2 and RMSE. 'all' included."""
    rng = np.random.default_rng(seed)
    F = encode_obs(O); N = len(O); fdim = F.shape[1]; Hmax = max(horizons)
    ok = np.ones(N, bool)
    for j in range(K): ok &= np.roll(~D, j + 1)
    for j in range(Hmax): ok &= np.roll(~D, -j)
    ok[:K] = False; ok[N - Hmax - 1:] = False
    cand = np.flatnonzero(ok)
    starts = rng.choice(cand, size=min(n_starts, len(cand)), replace=False)
    win = np.zeros((len(starts), K, fdim + 2), np.float32)
    for k in range(K):
        win[:, k, :fdim] = F[starts - k]; win[:, k, fdim:] = A[starts - k]
    cur = F[starts].copy()
    err_at = {}
    for h in range(1, Hmax + 1):
        win[:, 0, fdim:] = A[starts + h - 1]
        cur = cur + model.predict(win.reshape(len(starts), -1))
        for a in range(3):
            n = np.sqrt(cur[:, a] ** 2 + cur[:, a + 3] ** 2) + 1e-9
            cur[:, a] /= n; cur[:, a + 3] /= n
        win = np.roll(win, 1, axis=1); win[:, 0, :fdim] = cur
        if h in horizons:
            pa, ta = decode_obs(cur)[:, :3], decode_obs(F[starts + h])[:, :3]
            err_at[h] = np.arctan2(np.sin(pa - ta), np.cos(pa - ta))
    regs = mode[starts]
    out = {}
    for reg in ["all", *sorted(set(regs))]:
        sel = np.ones(len(starts), bool) if reg == "all" else (regs == reg)
        if sel.sum() < 30: continue
        out[reg] = {"n": int(sel.sum())}
        for h in horizons:
            e = err_at[h][sel]
            out[reg][h] = {ax: {"within": float((np.abs(e[:, i]) < 0.2).mean()),
                                "rmse": float(np.sqrt((e[:, i] ** 2).mean()))}
                           for i, ax in enumerate(AXES)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="?imulog=1 session file (jsonl or csv)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--horizons", type=int, nargs="+", default=[5, 25])
    ap.add_argument("--servo-id", action="store_true", help="add the after-identified-servo column")
    args = ap.parse_args()

    O, A, O2, D, header, mode = parse(args.log)
    print(f"log: {len(O):,} ticks, {int(D.sum())} cuts, surface={header.get('surface', '?')}, "
          f"regimes={ {m: int((mode == m).sum()) for m in sorted(set(mode))} }")

    tr = np.load("data/train.npz"); te = np.load("data/test.npz")
    Xtr, Ytr, *_ = make_windows(tr["obs"], tr["act"], tr["next_obs"], tr["done"], K)
    model = MLP(hidden=128, epochs=args.epochs).fit(Xtr, Ytr)
    twin = evaluate_axes(model, te["obs"], te["act"], te["done"], te["mode"].astype(str), args.horizons)
    real = evaluate_axes(model, O, A, D, mode, args.horizons)

    corrected = None
    if args.servo_id:
        half = len(O) // 2
        grid = list(itertools.product([0, 1, 2, 3], [3.0, 4.0, 5.0, 6.0, 8.0, None],
                                      [0.0, np.deg2rad(2)]))
        _, best = identify(model, O[:half], A[:half], O2[:half], D[:half], grid)
        print(f"identified servo: delay {best['delay_ticks']} ticks, slew {best['slew_rad_s']} rad/s "
              f"(grid points; run servo_id.py for determined-set diagnostics)")
        R = realized_from_commands(A, D, best)
        corrected = evaluate_axes(model, O, R, D, mode, args.horizons)

    hs = args.horizons
    cols = "".join(f"{'real':>8}{'twin':>7}{'gap':>7}" + (f"{'gap*':>7}" if corrected else "") for _ in hs)
    print(f"\nwithin 0.2 rad; gap = real - twin floor (matched regime); "
          + ("gap* = after identified servo; " if corrected else "") + "negative gap = worse than floor")
    print(f"{'regime':<10}{'n':>6}{'axis':>7}" + "".join(
        f"{'@' + str(h * 20) + 'ms':>{22 + (7 if corrected else 0)}}" for h in hs))
    print("-" * (23 + len(hs) * (22 + (7 if corrected else 0))))
    report = {}
    for reg in real:
        tref_name = REGIME_MAP.get(reg, "all") if reg != "all" else "all"
        tref = twin.get(tref_name, twin["all"])
        for ax in AXES:
            line = f"{reg if ax == 'roll' else '':<10}{real[reg]['n'] if ax == 'roll' else '':>6}{ax:>7}"
            for h in hs:
                r = real[reg][h][ax]["within"]; tw = tref[h][ax]["within"]; g = r - tw
                line += f"{r * 100:>7.1f}%{tw * 100:>6.1f}%{g * 100:>+6.1f}"
                if corrected:
                    line += f"{(corrected[reg][h][ax]['within'] - tw) * 100:>+7.1f}"
                report.setdefault(reg, {}).setdefault(str(h), {})[ax] = {
                    "real": r, "twin": tw, "gap": g,
                    **({"gap_after_servo": corrected[reg][h][ax]["within"] - tw} if corrected else {})}
            print(line)
    json.dump({"header": {k: v for k, v in header.items() if not isinstance(v, (list, dict))},
               "report": report}, open("results/gap_report.json", "w"), indent=1)
    print("\nwrote results/gap_report.json")


if __name__ == "__main__":
    main()
