"""Per-regime, per-axis open-loop scoring against the twin floor.

`evaluate_axes` is the one rollout every real-log report and every real2sim cell scores
with; `twin_regimes` adds the attitude-derived 'fallen' class the twin's excitation
labels do not carry; REGIME_MAP says which twin regime a real session name is allowed
to borrow its floor from.
"""
from __future__ import annotations
import numpy as np
from .forward import encode_obs, decode_obs, K, AXES
from .imulog import CTRL_HZ, rest_attitude, attitude_excursion, SEG_FALL_EXCURSION_RAD

REGIME_MAP = {"walk": "policy", "spin": "policy", "gesture": "keyframe",
              "handling": "ou", "idle": "still", "still": "still",
              "policy": "policy", "sine": "sine", "keyframe": "keyframe", "ou": "ou",
              # growbot-imulog-1 regimes, synthesized from the data by imulog's
              # segmenter. Only 'walking' -- commands active AND the body responding --
              # earns the twin's policy floor. 'impact' and 'unknown' are deliberately
              # ABSENT: they have no twin counterpart, so they fall back to the twin's
              # overall row and are printed as such rather than credited to a regime.
              "walking": "policy", "acting": "keyframe", "fall": "fallen",
              # a gait name from a file that DOES carry event rows; the nearest twin
              # regime is policy walking (the same alternating gait family). For
              # growbot-imulog-1 this is now unreachable: the segmenter labels every
              # tick, so header.gait is no longer anyone's default.
              "official": "policy"}
REST_MISMATCH_RAD = 0.35     # 20 deg. Above this, two logs are not the same setup.


def twin_regimes(obs, mode):
    """Twin excitation labels, plus the attitude-derived 'fallen' class.

    The twin's own mode labels say what the EXCITATION was doing, never what the
    body was doing, so there is no twin regime to compare a real fall against.
    This adds one: a tick whose attitude has left the twin's resting attitude by
    more than SEG_FALL_EXCURSION_RAD, the same threshold and the same
    excursion-from-rest measure the real fall is detected with.

    Rest is measured from the twin's own still ticks rather than assumed upright,
    because this body does not rest upright: at neutral commands it settles at
    pitch about -0.6 rad, and calling that 0.6 rad of fall would be an artefact of
    the assumption, not a property of the data.

    The threshold is looser than GrowBotSim.fallen()'s 1.2 rad, and deliberately:
    the real fall in these logs tops out at 0.99 rad of excursion, so 1.2 would
    leave the class empty and send the fall silently back to the twin's overall row.
    """
    obs = np.asarray(obs)
    t = np.arange(len(obs)) * (1000.0 / CTRL_HZ)
    rest = rest_attitude(t, obs[:, :3], obs[:, 3:])
    m = np.array(mode, dtype=object)
    if rest is not None:
        m[attitude_excursion(obs[:, :3], rest) > SEG_FALL_EXCURSION_RAD] = "fallen"
    return m.astype(str), rest


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
