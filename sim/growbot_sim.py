"""The GrowBot digital twin, driven at 50 Hz, with the phone-style IMU read out.

Body model, actuator gains and IMU convention come from the project's own
export (policy/Harsh_policies/DR_RMA_EXPORT). Control ticks at 50 Hz, the rate
the video gives for the reflex loop and the firmware glide engine.

Observation (6): roll, pitch, yaw, gyro_x, gyro_y, gyro_z -- the phone IMU as
the walk policy consumes it. Yaw is wrapped to (-pi, pi]; models should learn
deltas so the wrap does not matter.
Action (2): target angles for the two servos in radians, ctrlrange [-1.57, 1.57].
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).parent
XML = HERE / "growbot_body.xml"
POLICY = HERE / "policy_85mm.json"

CTRL_HZ = 50
OBS_DIM = 6
ACT_DIM = 2


def quat_to_rpy(q):
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


class GrowBotSim:
    def __init__(self, seed=0):
        self.m = mujoco.MjModel.from_xml_path(str(XML))
        self.d = mujoco.MjData(self.m)
        self.nframes = int(round(1.0 / (CTRL_HZ * self.m.opt.timestep)))
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, drop_height=0.06, tilt=0.0):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[2] = drop_height
        if tilt:
            # random initial lean about x/y so falls and recoveries are in the data
            ax = self.rng.normal(size=3); ax /= np.linalg.norm(ax)
            ang = self.rng.uniform(-tilt, tilt)
            self.d.qpos[3:7] = [np.cos(ang / 2), *(np.sin(ang / 2) * ax)]
        self.d.ctrl[:] = 0.0
        mujoco.mj_forward(self.m, self.d)
        for _ in range(int(0.5 * CTRL_HZ) * self.nframes):
            mujoco.mj_step(self.m, self.d)
        return self.obs()

    def obs(self):
        rpy = quat_to_rpy(self.d.qpos[3:7])
        gyro = self.d.qvel[3:6].copy()
        return np.concatenate([rpy, gyro]).astype(np.float32)

    def step(self, action):
        self.d.ctrl[:] = np.clip(action, -1.57, 1.57)
        for _ in range(self.nframes):
            mujoco.mj_step(self.m, self.d)
        return self.obs()

    def push(self, scale=1.0):
        """External kick, like a hand shoving or flicking the body."""
        self.d.qvel[0:3] += self.rng.normal(scale=scale, size=3) * [1, 1, 0.5]
        self.d.qvel[3:6] += self.rng.normal(scale=scale * 12, size=3)

    def fallen(self):
        r, p, _ = quat_to_rpy(self.d.qpos[3:7])
        return abs(r) > 1.2 or abs(p) > 1.2


# ----------------------------------------------------------------------
class WalkPolicy:
    """The shipped policy_85mm.json, run in numpy (same net as growbot_policy.js)."""

    def __init__(self, path=POLICY):
        p = json.loads(Path(path).read_text())
        self.mean = np.array(p["mean"], dtype=np.float32)
        self.std = np.array(p["std"], dtype=np.float32)
        self.layers = [(np.array(L["W"], dtype=np.float32), np.array(L["b"], dtype=np.float32))
                       for L in p["layers"]]
        self.act = p["activation"]
        self.hist = np.zeros((5, 2), dtype=np.float32)

    def _act(self, x):
        if self.act == "swish":
            return x / (1 + np.exp(-x))
        return np.tanh(x)

    def __call__(self, obs6):
        x = np.concatenate([obs6, self.hist.reshape(-1)])
        x = (x - self.mean) / (self.std + 1e-8)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = self._act(x)
        # PPO head emits [loc(2), log_std(2)]; the JS runner takes tanh(loc) only
        a = np.tanh(x[:2])  # [aRight, aLeft] in -1..1, radians of swing
        self.hist = np.roll(self.hist, 1, axis=0); self.hist[0] = a
        # XML: joint_1 is right_leg, joint_2 is left_leg
        return np.array([a[0], a[1]], dtype=np.float32)

    def reset(self):
        self.hist[:] = 0


# ----------------------------------------------------------------------
class Excitation:
    """Action generators that together cover what a real body experiences."""

    def __init__(self, rng):
        self.rng = rng
        self.mode = "hold"; self.t = 0; self.target = np.zeros(2); self.phase = 0.0
        self.policy = None
        try:
            self.policy = WalkPolicy()
        except Exception:
            self.policy = None

    def new_segment(self):
        modes = ["keyframe", "sine", "ou", "policy", "still"]
        w = [0.30, 0.20, 0.20, 0.20, 0.10] if self.policy else [0.375, 0.25, 0.25, 0.0, 0.125]
        self.mode = self.rng.choice(modes, p=w)
        self.t = 0
        self.len = int(self.rng.uniform(1.0, 4.0) * CTRL_HZ)
        self.target = self.rng.uniform(-1.2, 1.2, 2)
        self.freq = self.rng.uniform(0.5, 4.0)
        self.amp = self.rng.uniform(0.2, 1.2)
        self.phase = self.rng.uniform(0, 2 * np.pi)
        self.energy = self.rng.uniform(0.3, 1.0)
        self.state = np.zeros(2)
        if self.policy:
            self.policy.reset()

    def __call__(self, obs, prev_action):
        if self.t >= getattr(self, "len", 0):
            self.new_segment()
        self.t += 1
        m = self.mode
        if m == "still":
            a = prev_action
        elif m == "keyframe":
            # jump to a pose and hold; occasionally re-target (what /act glides do)
            if self.rng.random() < 0.05:
                self.target = self.rng.uniform(-1.2, 1.2, 2)
            a = 0.7 * prev_action + 0.3 * self.target
        elif m == "sine":
            ph = 2 * np.pi * self.freq * self.t / CTRL_HZ + self.phase
            a = self.amp * np.array([np.sin(ph), np.sin(ph + np.pi)])   # alternating gait shape
        elif m == "ou":
            self.state += -0.15 * self.state + 0.25 * self.rng.normal(size=2)
            a = np.clip(self.state, -1.2, 1.2)
        else:  # policy
            a = self.energy * self.policy(obs)
        return np.clip(a, -1.57, 1.57).astype(np.float32)


def collect(n_steps, seed=0, push_prob=0.01, episode_s=8.0, log_every=0):
    """(obs_t, act_t, obs_t+1, done_t) at 50 Hz. done marks the last step of an episode."""
    sim = GrowBotSim(seed)
    exc = Excitation(sim.rng)
    O = np.zeros((n_steps, OBS_DIM), np.float32)
    A = np.zeros((n_steps, ACT_DIM), np.float32)
    O2 = np.zeros((n_steps, OBS_DIM), np.float32)
    D = np.zeros(n_steps, bool)
    modes = []
    o = sim.reset(tilt=0.3)
    prev = np.zeros(2, np.float32)
    ep_len = int(episode_s * CTRL_HZ)
    def fresh():
        # one episode in five starts from a hard lean so tips and recoveries are covered
        return sim.reset(tilt=1.0 if sim.rng.random() < 0.2 else 0.3)
    t_ep = 0
    for i in range(n_steps):
        a = exc(o, prev)
        if sim.rng.random() < push_prob:
            sim.push()
        o2 = sim.step(a)
        O[i], A[i], O2[i] = o, a, o2
        modes.append(exc.mode)
        t_ep += 1
        end = t_ep >= ep_len or (sim.fallen() and sim.rng.random() < 0.02)
        D[i] = end
        if end:
            o = fresh(); prev = np.zeros(2, np.float32); t_ep = 0
            exc.new_segment()
        else:
            o, prev = o2, a
        if log_every and (i + 1) % log_every == 0:
            print(f"  {i + 1}/{n_steps}", flush=True)
    return O, A, O2, D, np.array(modes)


if __name__ == "__main__":
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE.parent / "data" / "growbot_50hz.npz"))
    args = ap.parse_args()
    t0 = time.time()
    O, A, O2, D, M = collect(args.steps, args.seed, log_every=50_000)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, obs=O, act=A, next_obs=O2, done=D, mode=M)
    dt = time.time() - t0
    print(f"{args.steps} steps = {args.steps / CTRL_HZ / 60:.1f} sim-minutes in {dt:.0f}s "
          f"({args.steps / dt / CTRL_HZ:.0f}x realtime)")
    print("modes:", {m: int((M == m).sum()) for m in np.unique(M)})
    print("episodes:", int(D.sum()))
    r, p = O[:, 0], O[:, 1]
    print(f"roll  range [{r.min():+.2f}, {r.max():+.2f}]  pitch range [{p.min():+.2f}, {p.max():+.2f}]")
    print(f"fallen frames (|roll| or |pitch| > 1.2): {np.mean((abs(r) > 1.2) | (abs(p) > 1.2)) * 100:.1f}%")
    print("saved", args.out)
