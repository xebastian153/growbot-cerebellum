"""The sim-to-real proxy primitives: the 13 DR corners, the online residual, and
`horizon_within` -- the within-0.2 rad open-loop score every corner experiment reuses.

`sim2real_proxy.py` at the repository root is the experiment that first used them;
`contact_friction.py`, `body_params.py`, `servo_id` and `model_mismatch.py` score with
the same function so no experiment re-implements the metric.
"""
from __future__ import annotations
import numpy as np
from .sim import DR
from .forward import make_windows, encode_obs, decode_obs, K

# 13 corners: nominal, each factor at both extremes (5x2 = 10), and two worst-case combos
def corners():
    lo = {k: v[0] for k, v in DR.items()}; hi = {k: v[1] for k, v in DR.items()}
    def mk(name, mass=1.0, dcom=(0, 0, 0), leg=1.0, gain=1.0, fric=None):
        return name, dict(mass_scale=mass, dcom=dcom, leg_scale=leg, gain_mult=gain, friction=fric)
    return [
        mk("nominal"),
        mk("mass -", mass=lo["mass_scale"]), mk("mass +", mass=hi["mass_scale"]),
        mk("com back/low", dcom=(lo["dcom_x"], 0, lo["dcom_z"])), mk("com fwd/high", dcom=(hi["dcom_x"], 0, hi["dcom_z"])),
        mk("leg -", leg=lo["leg_scale"]), mk("leg +", leg=hi["leg_scale"]),
        mk("gain -", gain=lo["gain_mult"]), mk("gain +", gain=hi["gain_mult"]),
        mk("friction -", fric=lo["friction"]), mk("friction +", fric=hi["friction"]),
        mk("worst A (heavy, long, weak, slippery)", mass=hi["mass_scale"], dcom=(hi["dcom_x"], 0, hi["dcom_z"]),
           leg=hi["leg_scale"], gain=lo["gain_mult"], fric=lo["friction"]),
        mk("worst B (light, short, strong, grippy)", mass=lo["mass_scale"], dcom=(lo["dcom_x"], 0, lo["dcom_z"]),
           leg=lo["leg_scale"], gain=hi["gain_mult"], fric=hi["friction"]),
    ]


class OnlineResidual:
    """delta_out += R @ x, R updated by normalised LMS on the observed one-step error."""
    def __init__(self, in_dim, out_dim, eta=0.05):
        self.R = np.zeros((out_dim, in_dim), np.float32)
        self.eta = eta
    def correct(self, X):
        return X @ self.R.T
    def update(self, x, err):
        # err = observed_delta - (model_delta + correction); x = the model input window
        self.R += self.eta * np.outer(err, x) / (float(x @ x) + 1.0)


def horizon_within(model, obs, act, done, residual=None, h=5, n_starts=1500, seed=0):
    """Open-loop imagination for h ticks; fraction of starts within 0.2 rad roll/pitch."""
    rng = np.random.default_rng(seed)
    F = encode_obs(obs); N = len(obs); fdim = F.shape[1]
    ok = np.ones(N, bool)
    for j in range(K): ok &= np.roll(~done, j + 1)
    for j in range(h): ok &= np.roll(~done, -j)
    ok[:K] = False; ok[N - h - 1:] = False
    starts = rng.choice(np.flatnonzero(ok), size=min(n_starts, ok.sum()), replace=False)
    win = np.zeros((len(starts), K, fdim + 2), np.float32)
    for k in range(K):
        win[:, k, :fdim] = F[starts - k]; win[:, k, fdim:] = act[starts - k]
    cur = F[starts].copy()
    for s in range(1, h + 1):
        win[:, 0, fdim:] = act[starts + s - 1]
        X = win.reshape(len(starts), -1)
        d = model.predict(X)
        if residual is not None:
            d = d + residual.correct(X)
        cur = cur + d
        for a in range(3):
            n = np.sqrt(cur[:, a] ** 2 + cur[:, a + 3] ** 2) + 1e-9
            cur[:, a] /= n; cur[:, a + 3] /= n
        win = np.roll(win, 1, axis=1); win[:, 0, :fdim] = cur
    pa, ta = decode_obs(cur)[:, :2], decode_obs(F[starts + h])[:, :2]
    e = np.arctan2(np.sin(pa - ta), np.cos(pa - ta))
    return float((np.abs(e).max(1) < 0.2).mean()), float(np.sqrt((e ** 2).mean()))


def adapt_online(model, obs, act, next_obs, done, warm_ticks, eta):
    """Stream the first warm_ticks of experience through the residual, one tick at a time."""
    X, Y, *_ = make_windows(obs[:warm_ticks], act[:warm_ticks], next_obs[:warm_ticks], done[:warm_ticks], K)
    res = OnlineResidual(X.shape[1], Y.shape[1], eta)
    base = model.predict(X)              # frozen model's predictions on this stream
    for i in range(len(X)):
        err = Y[i] - (base[i] + res.correct(X[i:i + 1])[0])
        res.update(X[i], err)
    return res
