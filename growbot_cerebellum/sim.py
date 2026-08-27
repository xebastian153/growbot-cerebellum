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

# The vendored GrowBot assets (body XMLs, policy weights) stay under sim/ at the repository
# root, where NOTICE and the licence point; this module only moved.
HERE = Path(__file__).resolve().parent.parent / "sim"
BODIES = {"walk": HERE / "growbot_body.xml", "olie": HERE / "growbot_olie_body.xml"}
XML = BODIES["walk"]
POLICY = HERE / "policy_85mm.json"

# Domain-randomisation ranges, copied from the upstream GrowBot file
# policy/Harsh_policies/SPIN_IN_PLACE_OLIE_EXPORT/dr_sweep_spin.py (not vendored here) so
# "a different body" means what the project already means by it.
DR = {"mass_scale": (0.80, 1.25), "dcom_x": (-0.030, 0.030), "dcom_y": (-0.015, 0.015),
      "dcom_z": (-0.010, 0.015), "leg_scale": (0.85, 1.15), "gain_mult": (0.75, 1.25),
      "friction": (0.6, 1.4)}

CTRL_HZ = 50
OBS_DIM = 6
ACT_DIM = 2


def quat_to_rpy(q):
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def perturb(m, mass_scale=1.0, dcom=(0.0, 0.0, 0.0), leg_scale=1.0, gain_mult=1.0, friction=None,
            friction_torsional=None, friction_rolling=None, condim=None, base_mass_delta=0.0):
    """Apply one DR corner to a loaded model. Same edits as dr_sweep_spin.build_model.

    `friction` is MuJoCo's SLIDING coefficient, geom_friction column 0 -- the only one
    the upstream sweep ever varied. The torsional (column 1) and rolling (column 2)
    coefficients are separate arguments because they are separate physics, and because
    the bodies ship with `condim="3"`, under which MuJoCo applies column 0 alone and
    ignores the other two entirely: setting them at condim 3 changes nothing at all
    (measured, see contact_friction.py). `condim=4` activates torsional, `condim=6`
    activates rolling as well. All three new arguments default to None = leave the
    model as loaded, so every existing caller reproduces bit-identically.
    `base_mass_delta` (kg) is added to the base body only: the base mass is scaled to
    mass + delta and its inertia is scaled by the same ratio, `body_ipos` is left
    untouched, and the legs do not change. It is therefore mass added at the base's
    EXISTING centre of mass -- a mass isolation, not a payload mounted somewhere.
    Default 0.0 = no-op.
    """
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_body")
    legs = [(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g), mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b))
            for g, b in (("lower_leg_1", "right_leg"), ("lower_leg_2", "left_leg"))]
    half0 = float(m.geom_size[legs[0][0], 2]); hx = float(m.geom_size[legs[0][0], 0]); hy = float(m.geom_size[legs[0][0], 1])
    leg_mass0 = float(m.body_mass[legs[0][1]])
    m.body_mass[:] *= mass_scale
    m.body_inertia[:] *= mass_scale
    if base_mass_delta:
        r = (float(m.body_mass[base]) + base_mass_delta) / float(m.body_mass[base])
        m.body_mass[base] *= r
        m.body_inertia[base] *= r
    m.body_ipos[base] += np.array(dcom)
    for gid, bid in legs:
        half = half0 * leg_scale
        m.geom_size[gid, 2] = half
        m.geom_pos[gid, 2] = -half
        mm = leg_mass0 * leg_scale * mass_scale
        m.body_mass[bid] = mm
        a, b, c = 2 * hx, 2 * hy, 2 * half
        m.body_inertia[bid] = mm / 12.0 * np.array([b * b + c * c, a * a + c * c, a * a + b * b])
        m.body_ipos[bid, 2] = -half
    m.actuator_gainprm[:] *= gain_mult
    m.actuator_biasprm[:] *= gain_mult
    if condim is not None:
        m.geom_condim[:] = int(condim)
    if friction is not None:
        m.geom_friction[:, 0] = friction
    if friction_torsional is not None:
        m.geom_friction[:, 1] = friction_torsional
    if friction_rolling is not None:
        m.geom_friction[:, 2] = friction_rolling
    mujoco.mj_setConst(m, mujoco.MjData(m))
    return m


class ServoModel:
    """A cheap hobby servo between the command and MuJoCo's ideal PD actuator.

    MuJoCo's position actuator reaches any target instantly up to its force limit.
    Real MG90S-class servos do not: the command arrives late (serial + PWM frame),
    the horn cannot turn faster than a slew limit, and small errors inside a
    deadband are ignored. None of that is in the twin the forward model learned.

      delay_ticks  command latency in CALLS -- the caller's rate sets the unit.
                   GrowBotSim.step calls once per 50 Hz tick (1 call = 20 ms);
                   imulog.fixture calls at physics rate (1 call = 5 ms) and
                   converts from milliseconds for exactly that reason. This
                   ambiguity has produced one real bug already; when in doubt,
                   express the delay in ms and divide by the caller's dt.
      slew_rad_s   maximum horn speed (MG90S ~ 0.1 s / 60 deg no load = 10.5 rad/s;
                   under load 3-6 rad/s)
      deadband     radians of error the servo ignores
    """
    def __init__(self, delay_ticks=0, slew_rad_s=None, deadband=0.0):
        self.delay, self.slew, self.db = int(delay_ticks), slew_rad_s, float(deadband)
        self.reset()
    def reset(self, pos=None):
        from collections import deque
        self.pos = np.zeros(2, np.float32) if pos is None else np.array(pos, np.float32)
        self.q = deque([self.pos.copy()] * (self.delay + 1), maxlen=self.delay + 1)
    def __call__(self, target, dt):
        self.q.append(np.array(target, np.float32))
        tgt = self.q[0]
        err = tgt - self.pos
        if self.db > 0:
            err = np.where(np.abs(err) < self.db, 0.0, err)
        if self.slew is not None:
            err = np.clip(err, -self.slew * dt, self.slew * dt)
        self.pos = self.pos + err
        return self.pos


class GrowBotSim:
    def __init__(self, seed=0, body="walk", dr=None, servo=None):
        self.m = mujoco.MjModel.from_xml_path(str(BODIES[body]))
        if dr:
            perturb(self.m, **dr)
        self.d = mujoco.MjData(self.m)
        self.nframes = int(round(1.0 / (CTRL_HZ * self.m.opt.timestep)))
        self.rng = np.random.default_rng(seed)
        self.servo = servo          # ServoModel or None (ideal)
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
        if self.servo is not None:
            self.servo.reset()
        mujoco.mj_forward(self.m, self.d)
        for _ in range(int(0.5 * CTRL_HZ) * self.nframes):
            mujoco.mj_step(self.m, self.d)
        return self.obs()

    def obs(self):
        rpy = quat_to_rpy(self.d.qpos[3:7])
        gyro = self.d.qvel[3:6].copy()
        return np.concatenate([rpy, gyro]).astype(np.float32)

    def step(self, action):
        a = np.clip(action, -1.57, 1.57)
        if self.servo is not None:
            a = self.servo(a, 1.0 / CTRL_HZ)
        self.d.ctrl[:] = a
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


def collect(n_steps, seed=0, push_prob=0.01, episode_s=8.0, log_every=0, body="walk", dr=None, servo=None,
            return_realized=False):
    """(obs_t, act_t, obs_t+1, done_t) at 50 Hz. done marks the last step of an episode.
    act_t is the COMMANDED angle; with a ServoModel the horn lags it.
    return_realized=True appends the realized horn angle to the tuple (preferred over
    the legacy collect.last_realized attribute, which later calls overwrite)."""
    sim = GrowBotSim(seed, body=body, dr=dr, servo=servo)
    exc = Excitation(sim.rng)
    O = np.zeros((n_steps, OBS_DIM), np.float32)
    A = np.zeros((n_steps, ACT_DIM), np.float32)
    O2 = np.zeros((n_steps, OBS_DIM), np.float32)
    D = np.zeros(n_steps, bool)
    R = np.zeros((n_steps, ACT_DIM), np.float32)   # realized horn angle (== command with an ideal servo)
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
        R[i] = sim.servo.pos if sim.servo is not None else np.clip(a, -1.57, 1.57)
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
    collect.last_realized = R          # legacy side channel; prefer return_realized=True
    if return_realized:
        return O, A, O2, D, np.array(modes), R
    return O, A, O2, D, np.array(modes)

