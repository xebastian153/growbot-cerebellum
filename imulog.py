"""Parser for GrowBot ?imulog=1 sessions, and the fixture that validates it.

The upstream ?imulog=1 format: two streams at native rates, measured
timestamps on one shared clock. IMU as the phone delivers it (~60 Hz), l/r poses
at actual send time (~30 Hz). A header carries units, axis and mount convention,
whether gravity is included, gait name and stride gain, calibration trims, build
version, wall-clock anchor.

This module does three things:
  parse()    log file -> the 50 Hz (obs, act, next_obs, done) arrays every
             experiment in this repo consumes. Orientation interpolated on
             (sin, cos), gyro linearly, commands zero-order-held (a command is
             in force until the next one). IMU gaps > `gap_ms` split episodes.
  fixture()  generates a synthetic session from the twin with jittered 60/30 Hz
             sampling and a hidden ServoModel, in the same file format.
  __main__   round-trip test: fixture -> parse -> one-step forward error and
             servo identification must survive the resampling.

Accepted file format (one JSON object per line):
  line 1  {"header": {...}}   free-form; fields used if present:
          imu_units ("rad"|"deg"), pose_units ("deg" default, 90 = neutral),
          l_sign/r_sign/l_off/r_off/gain (trims; default +1/+1/0/0/1),
          trims_in_values (bool, default true), gait, surface
  rows    {"t": <ms>, "s": "imu", "o": [roll, pitch, yaw, gr, gp, gy]}
          {"t": <ms>, "s": "cmd", "l": <deg>, "r": <deg>}
          {"t": <ms>, "s": "ev",  "name": "walk_start" | ...}   (optional)

The upstream growbot-imulog-1 dialect carries NO event rows, so this module
synthesizes them from the data (segment_growbot_v1: still / walking / impact /
fall). Without that, every tick of every real file inherits header.gait and is
scored against the twin's walking floor whatever the body was doing -- which is
how a motionless phone and a fall once came to be compared with a walk.
CSV fallback (auto-detected): columns t,s,roll,pitch,yaw,gr,gp,gy,l,r with the
same meanings; an optional first line `# {json}` carries the header.

Trim convention. The upstream runner sends `l = 90 + off + sign*deg(a)*gain`,
so what reaches the servo already contains the trims. `trims_in_values: true`
(the default) means the logged numbers are those as-sent values and the parser
INVERTS the trims: a = deg2rad((v - 90 - off) / (sign * gain)). `false` means
the log carries pre-trim model commands and the trims in the header are ignored:
a = deg2rad(v - 90). Getting this flag wrong is a constant offset/scale on every
action -- the kind of error the forward model absorbs silently and that poisons
servo identification, which is why it is explicit rather than guessed. A turn
bias mixed into l/r cannot be inverted from the header alone; if turn is active
during logging it must be logged per-sample or the session flagged.
"""
from __future__ import annotations
import json, sys
import numpy as np

CTRL_HZ = 50
GAP_MS = 100.0          # IMU dt above this is a dropout: parse() cuts the episode there,
                        # and sensor_id refuses to interpolate across it

# Stillness thresholds. Defined HERE and imported by sensor_id.verify_still, so
# "still" means one thing in the preflight, in the analysis, and in the
# growbot-imulog-1 segmenter below -- three places that would otherwise drift.
STILL_GYRO_RMS_MAX = 0.15      # rad/s, RMS over the three body rates
STILL_ANG_STD_MAX = 0.05       # rad, largest PER-AXIS roll/pitch std (a tilt is not motion)

# ----------------------------------------------------------------------------
# growbot-imulog-1: the upstream app's own per-walk format (one JSON object with
# header + imu[] + pose[] arrays), converted here to the internal representation.
#
# Mount rotation, device frame -> twin body frame. Columns are the twin axes in
# device coordinates: twin x = +device y (long axis of the phone), twin y =
# -device x, twin z = +device z (out of the screen). Established from the real
# logs, not from a datasheet:
#   - gravity: composing R = Rz(alpha) Rx(beta) Ry(gamma) (the W3C deviceorientation
#     intrinsic ZX'Y'' order) maps the logged accelerometer to earth mean
#     [-0.2, 0.4, 9.9] m/s^2 -- the composition is right;
#   - stance: under this mount the walking stance reads pitch -0.74 +- 0.16 rad,
#     roll -0.06 -- the twin walks at pitch -0.52 +- 0.27 and its geometry rests
#     the -x phone edge at -0.81 rad; the opposite x sign reads +0.74 and is out;
#   - action response: corr(mean leg command, pitch) and corr(right-left
#     differential, roll) match the twin's signs on both files under this mount
#     and the upstream l/r assignment; swapping l and r flips the roll signature
#     into disagreement.
GROWBOT_V1_MOUNT = np.array([[0.0, -1.0, 0.0],
                             [1.0,  0.0, 0.0],
                             [0.0,  0.0, 1.0]])
GROWBOT_V1_UNITS = {"accel": "m/s^2", "rate": "deg/s", "t": "ms", "ori": "deg"}
# The calibration this adapter's evidence was established under. Upstream ships at
# least one other build with IMU_SIGN [1, -1, 1]; applying the mount above to a
# sign-flipped device stream silently inverts pitch, and nothing downstream would
# say so. SWAP exchanges the l/r pose columns for a mirrored chassis. Neither can
# be validated from the file, so a non-default value is refused rather than guessed.
GROWBOT_V1_CAL_DEFAULTS = {"IMU_SIGN": [1, 1, 1], "SWAP": 0}


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    z = np.zeros_like(a); o = np.ones_like(a)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1),
                     np.stack([z, z, o], -1)], -2)


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    z = np.zeros_like(a); o = np.ones_like(a)
    return np.stack([np.stack([o, z, z], -1), np.stack([z, c, -s], -1),
                     np.stack([z, s, c], -1)], -2)


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    z = np.zeros_like(a); o = np.ones_like(a)
    return np.stack([np.stack([c, z, s], -1), np.stack([z, o, z], -1),
                     np.stack([-s, z, c], -1)], -2)


def _deviceorientation_to_R(alpha_rad, beta_rad, gamma_rad):
    """W3C deviceorientation intrinsic Z-X'-Y'': R(device->earth) = Rz(a) Rx(b) Ry(g)."""
    return _rz(alpha_rad) @ _rx(beta_rad) @ _ry(gamma_rad)


def _R_to_deviceorientation(R):
    """Inverse of _deviceorientation_to_R (any branch recomposes to the same R)."""
    beta = np.arcsin(np.clip(R[..., 2, 1], -1.0, 1.0))
    gamma = np.arctan2(-R[..., 2, 0], R[..., 2, 2])
    alpha = np.arctan2(-R[..., 0, 1], R[..., 1, 1])
    return alpha, beta, gamma


def _R_to_zyx_rpy(R):
    """Rotation matrix -> (roll, pitch, yaw), same ZYX convention as the twin's quat_to_rpy."""
    roll = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    pitch = -np.arcsin(np.clip(R[..., 2, 0], -1.0, 1.0))
    yaw = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    return np.stack([roll, pitch, yaw], -1)


# ----------------------------------------------------------------------------
# Motion-based segmentation for growbot-imulog-1.
#
# The format carries NO event rows. Before this existed the adapter returned an
# empty event list, so _mode_per_tick fell back to header.gait -- every tick of
# every file was labelled "official" and mapped to the twin's POLICY-WALKING
# floor, and preflight's physics-on-labelled-segments block was unreachable on
# real files. That is how a motionless phone and a fall came to be scored against
# a walking floor. The regimes are therefore synthesized from the DATA.
#
# The four classes, and what each one claims:
#   still     the BODY is not moving, by sensor_id.verify_still's own thresholds
#             applied in a rolling window. It claims nothing about the commands:
#             a still window under active commands is a real and important state
#             (unattached phone, robot off the ground, dead servos), reported by
#             preflight rather than hidden inside a "walking" label.
#   walking   commands are active AND the body is responding (the window fails
#             the stillness test). This is the only class mapped to the twin's
#             policy floor.
#   impact    an acceleration spike at a stillness boundary -- the transition
#             event itself. The magnitude alone cannot carry this: walk-1's foot
#             strikes reach 6.9 g in the middle of ordinary walking, higher than
#             walk-3's 5.1 g collision, so the class is defined by WHERE the
#             spike sits, not by how large it is.
#   fall      only in a file whose header says end_why == "tipped": the attitude
#             leaves the file's OWN resting attitude by more than
#             SEG_FALL_EXCURSION_RAD and never comes back before the recording
#             ends, with a body-rate onset to match. Excursion is measured from
#             each file's own rest because the two real logs sit in different
#             mounting/placement states -- walk-1 rests at pitch -0.74 rad and
#             walk-3 flat at 0.00, so any ABSOLUTE tilt threshold would label one
#             of them wrongly.
#   unknown   everything the three tests above do not explain (moving with no
#             command, or a window that is neither still nor driven). Left
#             unmapped, so it is scored against the twin's overall row and
#             printed as such rather than silently credited to a regime.
SEG_WIN_MS = 400.0             # rolling window for the stillness / activity tests
SEG_MIN_MS = 240.0             # runs shorter than this are absorbed (impact is exempt)
SEG_IMPACT_G = 3.0             # |accel| / g at a stillness boundary = impact
SEG_CMD_ACTIVE_DEG = 5.0       # |command - 90| above which the agent is driving the servos
SEG_FALL_WIN_MS = 500.0        # window the fall excursion is held over (a fall oscillates)
SEG_FALL_EXCURSION_RAD = 0.7   # 40 deg from the file's own rest attitude. Between walk-1's
                               # largest walking excursion (0.50) and walk-3's fall (0.99).
SEG_FALL_RELEASE_RAD = 0.35    # the fall is over only if the body comes back INSIDE this.
                               # A fallen body oscillates: walk-3's tail dips to 0.62 rad of
                               # excursion and rises again, so a single-threshold test would
                               # place the onset at the last dip, two seconds after the
                               # collision that caused it. Hysteresis, not a tighter threshold.
SEG_FALL_RATE_RAD_S = 1.5      # body rate the departure must contain somewhere
TIP_SETTLE_S = 5.0             # fixture only: quiet walking before the injected tip


def _win_bounds(t, win_ms):
    """[lo, hi) index bounds of a centred +-win_ms/2 window around every sample."""
    h = win_ms / 2.0
    return np.searchsorted(t, t - h, "left"), np.searchsorted(t, t + h, "right")


def _win_mean(x, lo, hi):
    """Windowed mean of x (1-D) for every sample, via one cumulative sum."""
    c = np.concatenate([[0.0], np.cumsum(np.asarray(x, np.float64))])
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def _win_any(flag, lo, hi):
    """True where any sample inside the window has flag set."""
    c = np.concatenate([[0], np.cumsum(np.asarray(flag, np.int64))])
    return (c[hi] - c[lo]) > 0


def rolling_still(t, rpy, gyro, win_ms=SEG_WIN_MS):
    """Per-sample stillness, verify_still's two thresholds in a rolling window.

    Returns (still, gyro_rms, ang_std). The angle statistic is the LARGEST
    PER-AXIS roll/pitch std, never the pooled std, for the reason
    sensor_id.verify_still documents: pooled, a motionless robot standing on a
    slope scores its own tilt and fails.
    """
    t = np.asarray(t, np.float64)
    lo, hi = _win_bounds(t, win_ms)
    grms = np.sqrt(_win_mean((np.asarray(gyro, np.float64) ** 2).sum(1), lo, hi))
    astd = np.zeros(len(t))
    for a in range(2):                       # roll, pitch
        x = np.asarray(rpy, np.float64)[:, a]
        m = _win_mean(x, lo, hi)
        astd = np.maximum(astd, np.sqrt(np.maximum(_win_mean(x ** 2, lo, hi) - m ** 2, 0.0)))
    return (grms <= STILL_GYRO_RMS_MAX) & (astd <= STILL_ANG_STD_MAX), grms, astd


def rest_attitude(t, rpy, gyro, win_ms=SEG_WIN_MS, first_run=False):
    """(roll, pitch) the body rests at: the median over its verified-still samples.

    None when the record contains no still sample at all -- the honest answer,
    and the caller must not invent a rest attitude for a body that never rested.

    first_run=True takes only the FIRST still run instead of pooling all of them.
    A record that ends tipped over ends motionless too, so its later still samples
    are the FALL, not the rest: pooled, a session with a 10 s upright prefix and a
    10 s upside-down tail returns a median attitude the body never held, and the
    fall it is supposed to detect vanishes into its own reference.
    """
    still, _, _ = rolling_still(t, rpy, gyro, win_ms)
    if not still.any():
        return None
    r = np.asarray(rpy, np.float64)
    sel = still
    if first_run:
        for a, b, on in _runs(still.astype(int)):
            if on and t[b - 1] - t[a] >= SEG_MIN_MS:
                sel = np.zeros(len(still), bool); sel[a:b] = True
                break
        else:
            return None
    return float(np.median(r[sel, 0])), float(np.median(r[sel, 1]))


def attitude_excursion(rpy, rest):
    """Largest per-axis roll/pitch departure from a rest attitude, wrapped to +-pi."""
    r = np.asarray(rpy, np.float64)
    d = np.stack([np.arctan2(np.sin(r[:, a] - rest[a]), np.cos(r[:, a] - rest[a]))
                  for a in range(2)], 1)
    return np.abs(d).max(1)


def _runs(label):
    """[(start, stop_exclusive, name)] for every maximal run of equal labels."""
    out, i = [], 0
    while i < len(label):
        j = i
        while j + 1 < len(label) and label[j + 1] == label[i]:
            j += 1
        out.append((i, j + 1, label[i]))
        i = j + 1
    return out


def segment_growbot_v1(imu_t, rpy, gyro, acc, cmd_t, cmd_lr, end_why=None):
    """Synthesize regime events for a growbot-imulog-1 record from its own data.

    Returns [(t_ms, "<regime>_start"), ...] in _mode_per_tick's dialect, one per
    segment, covering the record from its first sample. Thresholds are the stated
    SEG_* constants above; every one of them is a number this file's docstring
    justifies, not a tuned knob.

    Order of precedence, loosest claim first: still -> walking -> fall -> impact.
    The impact windows sit on top because they ARE the boundaries the other
    classes meet at, and runs shorter than SEG_MIN_MS are absorbed into their
    predecessor so a single noisy sample cannot manufacture a regime.
    """
    imu_t = np.asarray(imu_t, np.float64)
    rpy = np.asarray(rpy, np.float64)
    gyro = np.asarray(gyro, np.float64)
    n = len(imu_t)
    if n < 8:
        return []
    lo, hi = _win_bounds(imu_t, SEG_WIN_MS)
    still, _, _ = rolling_still(imu_t, rpy, gyro)

    # commands, zero-order held onto the IMU clock, then "active anywhere in the window"
    if len(cmd_t):
        k = np.clip(np.searchsorted(np.asarray(cmd_t, np.float64), imu_t, "right") - 1, 0, None)
        dev = np.abs(np.asarray(cmd_lr, np.float64)[k] - 90.0).max(1)
        dev[np.searchsorted(np.asarray(cmd_t, np.float64), imu_t, "right") == 0] = 0.0
    else:
        dev = np.zeros(n)
    cmd_active = _win_any(dev >= SEG_CMD_ACTIVE_DEG, lo, hi)

    label = np.full(n, "unknown", dtype=object)
    label[still] = "still"
    label[(~still) & cmd_active] = "walking"

    # fall: only where the header says the session ended tipped, and only when the
    # attitude leaves this file's own rest and stays away until the recording ends
    if str(end_why) == "tipped":
        rest = rest_attitude(imu_t, rpy, gyro, first_run=True)
        if rest is not None:
            exc = attitude_excursion(rpy, rest)
            flo, fhi = _win_bounds(imu_t, SEG_FALL_WIN_MS)
            departed = _win_any(exc > SEG_FALL_RELEASE_RAD, flo, fhi)
            tipped = _win_any(exc > SEG_FALL_EXCURSION_RAD, flo, fhi)
            # the fall is the FINAL departure that never comes back: the last run of
            # `departed` reaching the end of the record, which must also have tipped
            # past the full threshold and must contain a real body-rate event -- a
            # body that only leans slowly has not fallen.
            if departed[-1] and tipped[-1]:
                at_rest = np.flatnonzero(~departed)
                start = int(at_rest[-1]) + 1 if len(at_rest) else 0
                if start < n and np.abs(gyro[start:]).max(initial=0.0) >= SEG_FALL_RATE_RAD_S:
                    label[start:] = "fall"

    # impact: an acceleration spike sitting on a stillness boundary
    if acc is not None and len(acc) == n:
        spike = np.linalg.norm(np.asarray(acc, np.float64), axis=1) >= SEG_IMPACT_G * 9.81
        edge = np.zeros(n, bool)
        edge[1:] = still[:-1] != still[1:]
        near_edge = _win_any(edge, lo, hi)
        # the spike and the SEG_WIN_MS transient AFTER it -- forward, not centred, so an
        # impact never eats the tail of the still segment that led up to it
        imp_hi = np.searchsorted(imu_t, imu_t + SEG_WIN_MS, "right")
        for i in np.flatnonzero(spike & near_edge):
            label[i:imp_hi[i]] = "impact"

    # absorb runs too short to be a regime (impact is exempt: it is an event)
    runs = _runs(label)
    for a, b, name in runs:
        if name == "impact" or a == 0:
            continue
        if imu_t[b - 1] - imu_t[a] < SEG_MIN_MS:
            label[a:b] = label[a - 1]

    ev = [(float(imu_t[a]), f"{name}_start") for a, _, name in _runs(label)]
    ev[0] = (float(imu_t[0]), ev[0][1])
    return ev


def _convert_growbot_v1(obj):
    """One growbot-imulog-1 object -> (header, imu_t, imu_v, cmd_t, cmd_v, events).

    Everything leaves in the twin's dialect: orientation as ZYX roll/pitch/yaw in
    radians, gyro as body rates in rad/s, commands as horn radians in the twin's
    action order [a0 = right leg, a1 = left leg].

    The pieces, each with the reason it is what it is:
      rate fields    logged as rate_alpha/beta/gamma but EMPIRICALLY they are the
                     body rates about device x/y/z in that order -- correlating
                     vee(R^T dR/dt) from the orientation stream against the three
                     logged rates on a real walk assigns them diagonally at
                     0.70-0.89 with off-diagonals below 0.20. The W3C names
                     (alpha about z) do not describe this app's output; the data
                     does.
      commands       the upstream servo map (sim/growbot_policy.js) is
                     l = 90 + L_OFF + L_SIGN*deg(a_left)*gain + turn and
                     r = 90 + R_OFF + R_SIGN*deg(a_right)*gain - turn, with
                     a_right = action[0] (joint_1) and a_left = action[1]
                     (joint_2). Inverted exactly, per side, turn folded per side.
                     header.gain (the agent's walk gain, sometimes null) scales
                     the ACTION before this map and is therefore already inside
                     the logged values; only cal.gain takes part in the
                     inversion. The two must never be confused: the internal
                     'gain' key is cal.gain, the agent's is kept as 'gain_agent'.
      send_ok        a row with send_ok = 0 is a command that never reached the
                     body, so the previous command stayed in force: the row is
                     dropped from the stream (zero-order hold then does the right
                     thing) and counted in the header.
      IMU_SIGN/SWAP  refused unless default -- see GROWBOT_V1_CAL_DEFAULTS.
      events         the format carries none, so they are SYNTHESIZED from the
                     data by segment_growbot_v1 (still / walking / impact / fall).
                     Without them every tick inherits header.gait and is scored
                     against the twin's walking floor whatever the body was doing.
    """
    h = dict(obj.get("header", {}))
    for k, want in GROWBOT_V1_UNITS.items():
        got = str(h.get("units", {}).get(k, ""))
        if not got.startswith(want):
            raise ValueError(f"growbot-imulog-1 units[{k!r}] = {got!r}, expected {want!r}: "
                             f"refusing to guess a conversion")
    want_imu = ["seq", "t_ms", "ax", "ay", "az", "rate_alpha", "rate_beta", "rate_gamma",
                "ori_alpha", "ori_beta", "ori_gamma"]
    want_pose = ["seq", "t_ms", "l", "r", "send_ok"]
    if h.get("imu_fields") != want_imu or h.get("pose_fields") != want_pose:
        raise ValueError(f"growbot-imulog-1 field order changed: imu {h.get('imu_fields')}, "
                         f"pose {h.get('pose_fields')} -- the converter indexes by position")
    imu = np.asarray(obj["imu"], np.float64)
    pose = np.asarray(obj["pose"], np.float64)
    if imu.ndim != 2 or imu.shape[1] != len(want_imu) or pose.ndim != 2 or pose.shape[1] != len(want_pose):
        raise ValueError(f"growbot-imulog-1 row shapes {imu.shape}/{pose.shape} do not match the declared fields")

    M = GROWBOT_V1_MOUNT
    ori = np.deg2rad(imu[:, 8:11])
    R = _deviceorientation_to_R(ori[:, 0], ori[:, 1], ori[:, 2]) @ M
    rpy = _R_to_zyx_rpy(R)
    gyro = np.deg2rad(imu[:, 5:8]) @ M              # M^T w_device, rates about device x/y/z
    imu_t = imu[:, 1].astype(np.float64)
    imu_v = np.concatenate([rpy, gyro], 1).astype(np.float32)

    cal = dict(h.get("cal", {}))
    for k, default in GROWBOT_V1_CAL_DEFAULTS.items():
        got = cal.get(k, default)
        if list(np.atleast_1d(got)) != list(np.atleast_1d(default)):
            raise ValueError(
                f"growbot-imulog-1 cal[{k!r}] = {got!r}, expected the default {default!r}. "
                f"The mount, rate-axis assignment and l/r assignment this adapter applies "
                f"were established on the default calibration; under {k} = {got!r} they are "
                f"a silent sign or side flip that no downstream number would reveal. "
                f"Refusing to guess: re-derive the conventions on a log from that build.")
    ls, rs = float(cal.get("L_SIGN", 1)), float(cal.get("R_SIGN", 1))
    lo, ro = float(cal.get("L_OFF", 0)), float(cal.get("R_OFF", 0))
    g, turn = float(cal.get("gain", 1.0)), float(cal.get("turn", 0.0))
    ok = pose[:, 4] != 0
    a_right = np.deg2rad((pose[ok, 3] - 90 - ro + turn) / (rs * g))
    a_left = np.deg2rad((pose[ok, 2] - 90 - lo - turn) / (ls * g))
    cmd_t = pose[ok, 1].astype(np.float64)
    cmd_v = np.stack([a_right, a_left], 1).astype(np.float32)

    header = {"format": h.get("format"), "imu_units": "rad", "pose_units": "rad",
              "trims_in_values": True,
              "l_sign": ls, "r_sign": rs, "l_off": lo, "r_off": ro, "gain": g, "turn": turn,
              "gait": h.get("gait", "unknown"), "gain_agent": h.get("gain"),
              "walk": h.get("walk"), "end_why": h.get("end_why"),
              "body_id": h.get("body_id"), "app": h.get("app"), "anchor": h.get("anchor"),
              "send_ok_dropped": int((~ok).sum()), "n_pose_rows": int(len(pose)),
              "dropped_imu": h.get("dropped_imu"), "dropped_pose": h.get("dropped_pose")}
    events = segment_growbot_v1(imu_t, rpy, gyro, imu[:, 2:5],
                                pose[ok, 1].astype(np.float64), pose[ok, 2:4],
                                h.get("end_why"))
    header["segments"] = [(round(t, 1), n) for t, n in events]
    return header, imu_t, list(imu_v), cmd_t, list(cmd_v), events


def _read_rows(path):
    """growbot-imulog-1 (one JSON object), JSONL, or CSV -- sniffed from the first line."""
    header, imu_t, imu_v, cmd_t, cmd_v, events = {}, [], [], [], [], []
    with open(path) as f:
        first = ""
        for first in f:
            if first.strip(): break
        rest = f
        s = first.strip()
        handled = False
        if s.startswith("{") and '"imu"' in s and '"pose"' in s:   # one-object growbot-imulog-1
            obj = json.loads(s)
            if str(obj.get("header", {}).get("format", "")).startswith("growbot-imulog"):
                header, imu_t, imu_v, cmd_t, cmd_v, events = _convert_growbot_v1(obj)
                imu_t = list(imu_t); cmd_t = list(cmd_t)
                handled = True
        if handled:
            pass
        elif s.startswith("{"):                     # JSON-lines
            for line in [first, *rest]:
                line = line.strip()
                if not line: continue
                row = json.loads(line)
                if "header" in row: header = row["header"]; continue
                if row.get("s") == "imu": imu_t.append(row["t"]); imu_v.append(row["o"])
                elif row.get("s") == "cmd": cmd_t.append(row["t"]); cmd_v.append([row["l"], row["r"]])
                elif row.get("s") == "ev": events.append((row["t"], row.get("name", "")))
        else:                                       # CSV, optional `# {json}` header line
            import csv, io
            if s.startswith("#"):
                maybe = s.lstrip("# ").strip()
                if maybe.startswith("{"): header = json.loads(maybe)
                first = next(rest, "")
            reader = csv.DictReader(io.StringIO(first + "".join(rest)))
            for row in reader:
                if row.get("s") == "imu":
                    imu_t.append(float(row["t"]))
                    imu_v.append([float(row[k]) for k in ("roll", "pitch", "yaw", "gr", "gp", "gy")])
                elif row.get("s") == "cmd":
                    cmd_t.append(float(row["t"])); cmd_v.append([float(row["l"]), float(row["r"])])
                elif row.get("s") == "ev":
                    events.append((float(row["t"]), row.get("roll", "")))
    imu_t = np.asarray(imu_t, np.float64); imu_v = np.asarray(imu_v, np.float32)
    cmd_t = np.asarray(cmd_t, np.float64); cmd_v = np.asarray(cmd_v, np.float32)
    # One timestamp order for every consumer, established here and nowhere else.
    # np.interp with non-monotonic xp returns nonsense, and a consumer that sorts
    # only the timestamps decouples them from their values; so the rows are sorted
    # stably by t per stream, values moving with their timestamps. How many
    # inversions the file had is reported through the header so preflight can say
    # so without re-reading the file.
    n_imu = int((np.diff(imu_t) < 0).sum()) if len(imu_t) > 1 else 0
    if n_imu:
        o = np.argsort(imu_t, kind="stable"); imu_t, imu_v = imu_t[o], imu_v[o]
    n_cmd = int((np.diff(cmd_t) < 0).sum()) if len(cmd_t) > 1 else 0
    if n_cmd:
        o = np.argsort(cmd_t, kind="stable"); cmd_t, cmd_v = cmd_t[o], cmd_v[o]
    events = sorted(events, key=lambda e: e[0])
    header = {**header, "_sorted_on_read": {"imu_inversions": n_imu, "cmd_inversions": n_cmd}}
    return header, (imu_t, imu_v, cmd_t, cmd_v), events


def _commands_to_rad(cmd_v, header):
    """Logged l/r -> model-space swing in radians, honouring the trim convention."""
    if header.get("pose_units", "deg") != "deg":
        return cmd_v.astype(np.float32)
    if header.get("trims_in_values", True):
        ls, rs = header.get("l_sign", 1.0), header.get("r_sign", 1.0)
        lo, ro = header.get("l_off", 0.0), header.get("r_off", 0.0)
        g = header.get("gain", 1.0)
        return np.stack([np.deg2rad((cmd_v[:, 0] - 90 - lo) / (ls * g)),
                         np.deg2rad((cmd_v[:, 1] - 90 - ro) / (rs * g))], 1).astype(np.float32)
    return np.deg2rad(cmd_v - 90.0).astype(np.float32)


def _mode_per_tick(events, grid, header):
    """Regime label per grid tick from event rows: 'X_start' opens regime X until the
    next event; 'X_stop' returns to 'idle'. No events -> the header's gait, else 'unknown'."""
    default = str(header.get("gait", "unknown"))
    mode = np.full(len(grid), default, dtype=object)
    if not events:
        return np.asarray(mode, dtype=str)
    ev = sorted(events)
    times = np.array([e[0] for e in ev]); names = [e[1] for e in ev]
    idx = np.searchsorted(times, grid, side="right") - 1
    for i, j in enumerate(idx):
        if j >= 0:
            n = names[j]
            mode[i] = "idle" if n.endswith("_stop") else n.removesuffix("_start")
    return np.asarray(mode, dtype=str)


def parse(path, gap_ms=GAP_MS, ang_lead_ms=None):
    """Resample a session onto the 50 Hz grid.

    ang_lead_ms: three per-axis milliseconds by which to ADVANCE the fused-orientation
    channels, read off their own timeline (roll, pitch, yaw). The phone reports a
    filtered orientation that trails the raw gyro -- sensor_id.filter_lag measures by
    how much -- while the twin emits both from the same instant, so a real observation
    vector is internally inconsistent in a way no training vector ever is. Advancing the
    angles aligns the two channels TO EACH OTHER; it does not make either absolute,
    because the gyro's own lag is not observable from the log. The shift is sub-tick
    (about 13 ms against a 20 ms grid), which is why it belongs here, in the resampling,
    and not in a shift of grid samples. Default None reproduces the unaligned parse
    exactly.
    """
    header, (imu_t, imu_v, cmd_t, cmd_v), events = _read_rows(path)
    if header.get("imu_units", "rad") == "deg": imu_v = np.deg2rad(imu_v)
    cmd_v = _commands_to_rad(cmd_v, header)
    # 50 Hz grid, episodes split at IMU gaps
    dt = 1000.0 / CTRL_HZ
    grid = np.arange(imu_t[0], imu_t[-1] - dt, dt)
    gap_after = np.searchsorted(imu_t, grid, side="right")
    prev_i = np.clip(gap_after - 1, 0, len(imu_t) - 1)
    next_i = np.clip(gap_after, 0, len(imu_t) - 1)
    in_gap = (imu_t[next_i] - imu_t[prev_i]) > gap_ms
    # orientation on (sin, cos); gyro linear
    ang, gyro = imu_v[:, :3], imu_v[:, 3:]
    lead = np.zeros(3) if ang_lead_ms is None else np.asarray(ang_lead_ms, float)
    at = grid[:, None] + lead                       # per-axis read times for the angles
    s = np.stack([np.interp(at[:, a], imu_t, np.sin(ang[:, a])) for a in range(3)], 1)
    c = np.stack([np.interp(at[:, a], imu_t, np.cos(ang[:, a])) for a in range(3)], 1)
    ang_g = np.arctan2(s, c)
    gyro_g = np.stack([np.interp(grid, imu_t, gyro[:, a]) for a in range(3)], 1)
    if ang_lead_ms is not None:
        # a shifted read can land inside a dropout the unshifted one missed, or past the
        # ends of the recording where np.interp would silently hold the edge value
        ga = np.searchsorted(imu_t, at, side="right")
        pa = np.clip(ga - 1, 0, len(imu_t) - 1)
        na = np.clip(ga, 0, len(imu_t) - 1)
        in_gap |= ((imu_t[na] - imu_t[pa]) > gap_ms).any(1)
        in_gap |= (at < imu_t[0]).any(1) | (at > imu_t[-1]).any(1)
    obs = np.concatenate([ang_g, gyro_g], 1).astype(np.float32)
    # commands: zero-order hold (last command at or before grid time; neutral before the first)
    idx = np.searchsorted(cmd_t, grid, side="right") - 1
    act = np.where(idx[:, None] >= 0, cmd_v[np.clip(idx, 0, None)], 0.0).astype(np.float32)
    O, A, O2 = obs[:-1], act[:-1], obs[1:]
    # a transition is invalid if EITHER endpoint was interpolated inside a gap
    D = (in_gap[:-1] | in_gap[1:]).copy(); D[-1] = True
    mode = _mode_per_tick(events, grid, header)[:-1]
    return O, A, O2, D, header, mode


def fixture(path, seconds=600, servo_ms=None, seed=0, imu_hz=60.0, cmd_hz=30.0, jitter_ms=2.0,
            push_prob_s=0.5, fused_lag_ms=None, gyro_arw=None, gyro_rrw=None, still_lead_s=0.0,
            imu_slow=None, tip_at_s=None):
    """Synthetic ?imulog=1 session from the twin: physics at 200 Hz, jittered sampling.

    servo_ms: dict(delay_ms=, slew_rad_s=, deadband=) -- delay given in MILLISECONDS
    and converted to physics-rate calls here, because ServoModel counts calls and the
    fixture steps it at physics rate, not at 50 Hz. (The first version passed 50 Hz
    ticks straight through and simulated a 10 ms delay while believing it was 40; the
    round-trip test below is what caught it.)

    Sensor-side secrets, all off by default so the servo round-trip is untouched:
    fused_lag_ms   emit orientation through a causal boxcar FIR on (sin, cos) at
                   physics rate. Linear phase, so its group delay is (W-1)/2 *
                   phys_dt across the passband -- the caveats are that the boxcar
                   has nulls at multiples of 1/(W*phys_dt) (no phase information
                   survives there) and that the arctan2 read of the filtered
                   (sin, cos) is nonlinear, so the exact-lag claim holds for
                   in-band content, not for every frequency.
    gyro_arw       white noise on the emitted gyro with angle-random-walk density
                   N (rad/s/sqrt(Hz)); per-sample std is N*sqrt(imu_hz).
    gyro_rrw       rate random walk added to the emitted gyro: a cumulative sum of
                   white noise with coefficient K (rad/s/sqrt(s)), per-sample step
                   std K*sqrt(1/imu_hz). Its Allan slope is +1/2, so with gyro_arw
                   the curve has a V-shaped ARW/RRW crossover and NO flicker floor
                   -- the negative control for bias-instability extraction.
    still_lead_s   hold neutral commands for this long first (reset untilted, no
                   pushes), labelled still from t=2 s -- the segment Allan needs.
    imu_slow       (start_s, dur_s, factor): IMU period multiplied by factor in
                   that window -- the timing stall dt_stats must flag.
    tip_at_s       shove the body hard at this time and never reposition it
                   again, so the session ends tipped over: the fall the v1
                   segmenter has to find. Random pushes stop at the same moment,
                   so the tail is a fall and not a shaking.
    """
    sys.path.insert(0, "sim")
    from growbot_sim import GrowBotSim, ServoModel, Excitation
    import mujoco
    rng = np.random.default_rng(seed)
    servo = None
    if servo_ms:
        phys_dt_s = 0.005
        servo = ServoModel(delay_ticks=round(servo_ms.get("delay_ms", 0) / 1000 / phys_dt_s),
                           slew_rad_s=servo_ms.get("slew_rad_s"), deadband=servo_ms.get("deadband", 0.0))
    sim = GrowBotSim(seed=seed, servo=servo)
    exc = Excitation(sim.rng)
    phys_dt = sim.m.opt.timestep                      # 0.005 s
    rows, t = [], 0.0
    next_imu = 0.0; next_cmd = 0.0; cur_cmd = np.zeros(2, np.float32)
    # Non-trivial trims, declared and applied at emission, so the full parse()
    # round-trip exercises the inversion path -- not only _commands_to_rad in isolation.
    trims = dict(l_sign=-1.0, r_sign=1.0, l_off=3.0, r_off=-2.0, gain=1.2)
    header = {"imu_units": "rad", "pose_units": "deg", "trims_in_values": True, **trims,
              "gait": "mixed", "surface": "twin",
              "build": "fixture", "note": "synthetic session for parser validation"}
    rows.append({"header": header})
    if still_lead_s > 0:
        rows.append({"t": 2000.0, "s": "ev", "name": "still_start"})   # 2 s settle stays unlabelled
    obs = sim.reset(tilt=0.0 if still_lead_s > 0 else 0.3); prev = np.zeros(2, np.float32)
    lead_done = still_lead_s <= 0
    if fused_lag_ms is not None:                    # boxcar FIR on (sin, cos): lag (W-1)/2*phys_dt
        W = int(round(2 * fused_lag_ms / 1000 / phys_dt)) + 1
        hist = np.zeros((W, 6)); nh = 0
    rrw = np.zeros(3)                               # rate-random-walk state (gyro_rrw)
    tipped = settled = False
    n_steps = int(seconds / phys_dt)
    for i in range(n_steps):
        if t >= next_cmd:
            if t < still_lead_s:
                a = cur_cmd = np.zeros(2, np.float32)
            else:
                last_mode = exc.mode
                a = exc(obs, prev); prev = cur_cmd = a
                if exc.mode != last_mode or not lead_done:
                    rows.append({"t": round(t * 1000, 2), "s": "ev", "name": f"{exc.mode}_start"})
                lead_done = True
            l_sent = 90 + trims["l_off"] + trims["l_sign"] * np.rad2deg(float(a[0])) * trims["gain"]
            r_sent = 90 + trims["r_off"] + trims["r_sign"] * np.rad2deg(float(a[1])) * trims["gain"]
            rows.append({"t": round(t * 1000 + rng.normal(0, jitter_ms), 2), "s": "cmd",
                         "l": round(l_sent, 2), "r": round(r_sent, 2)})
            next_cmd += 1.0 / cmd_hz * (1 + rng.normal(0, 0.03))
        aa = cur_cmd
        if sim.servo is not None: aa = sim.servo(np.clip(cur_cmd, -1.57, 1.57), phys_dt)
        sim.d.ctrl[:] = np.clip(aa, -1.57, 1.57)
        mujoco.mj_step(sim.m, sim.d)
        t += phys_dt
        if fused_lag_ms is not None:
            o6 = sim.obs()
            hist[nh % W] = np.concatenate([np.sin(o6[:3]), np.cos(o6[:3])]); nh += 1
        if tip_at_s is not None and not settled and t >= tip_at_s - TIP_SETTLE_S:
            # stand the body back up and leave it alone for TIP_SETTLE_S, so the tip
            # below is unambiguously THE fall of this session and not a random push
            # that happened to land first
            obs = sim.reset(tilt=0.0); prev = np.zeros(2, np.float32)
            settled = True
        if tip_at_s is not None and not tipped and t >= tip_at_s:
            # a roll-over impulse, not a teleport: the body physically turns past its
            # own tipping point, so the fall carries the rate onset and the impact the
            # segmenter is supposed to find. 20 rad/s about x tips this body and it
            # does not get back up (two legs, no arms).
            sim.d.qvel[3] += 20.0; sim.d.qvel[2] += 1.0
            tipped = True
        quiet = tip_at_s is not None and t >= tip_at_s - TIP_SETTLE_S
        if t >= still_lead_s and not quiet and rng.random() < push_prob_s * phys_dt:
            sim.push()
        if t >= next_imu:
            obs = sim.obs()
            emit = obs
            if fused_lag_ms is not None:
                m = hist[:min(nh, W)].mean(0)
                emit = np.concatenate([np.arctan2(m[:3], m[3:]), obs[3:]])
            if gyro_arw is not None or gyro_rrw is not None:
                emit = emit.copy()
                if gyro_arw is not None:
                    emit[3:] += rng.normal(0, gyro_arw * np.sqrt(imu_hz), 3)
                if gyro_rrw is not None:
                    rrw += rng.normal(0, gyro_rrw / np.sqrt(imu_hz), 3)
                    emit[3:] += rrw
            rows.append({"t": round(t * 1000 + rng.normal(0, jitter_ms), 2), "s": "imu",
                         "o": [round(float(v), 5) for v in emit]})
            period = 1.0 / imu_hz * (1 + rng.normal(0, 0.03))
            if imu_slow is not None and imu_slow[0] <= t < imu_slow[0] + imu_slow[1]:
                period *= imu_slow[2]
            next_imu += period
        if sim.fallen() and not quiet and rng.random() < 0.002:
            obs = sim.reset(tilt=0.3); prev = np.zeros(2, np.float32)
            if sim.servo is not None: sim.servo.reset()
            t += 0.5  # a real gap: the app was repositioned
            # The clock jumped, so the emission schedules jump with it. Left behind,
            # both emitters fire on every physics step until they catch up and
            # backfill ~30 rows at 200 Hz -- a rate no real session has, right where
            # the post-reset transient carries the most servo information. The
            # round-trip below leaned on exactly that: with the burst removed, the
            # argmin lands one grid step off the injected delay on half the seeds,
            # which is why the acceptance rule is now servo_id's determined set
            # rather than the argmin.
            next_imu = max(next_imu, t); next_cmd = max(next_cmd, t)
    # rows may be slightly out of order after jitter -- sort like a real file would be
    head, body = rows[0], sorted(rows[1:], key=lambda r: r["t"])
    with open(path, "w") as f:
        f.write(json.dumps(head) + "\n")
        for r in body: f.write(json.dumps(r) + "\n")
    return path


def _selfcheck():
    """Cheap invariants that need no simulation: trim inversion and CSV equality."""
    rng = np.random.default_rng(0)
    a = rng.uniform(-1.2, 1.2, (50, 2)).astype(np.float32)
    trims = dict(l_sign=-1.0, r_sign=1.0, l_off=3.0, r_off=-2.0, gain=1.2)
    sent = np.stack([90 + trims["l_off"] + trims["l_sign"] * np.rad2deg(a[:, 0]) * trims["gain"],
                     90 + trims["r_off"] + trims["r_sign"] * np.rad2deg(a[:, 1]) * trims["gain"]], 1)
    back = _commands_to_rad(sent, {**trims, "trims_in_values": True})
    assert np.allclose(back, a, atol=1e-5), "as-sent inversion failed"
    pre = 90 + np.rad2deg(a)
    back2 = _commands_to_rad(pre, {**trims, "trims_in_values": False})
    assert np.allclose(back2, a, atol=1e-5), "pre-trim path must ignore header trims"
    print("trim convention check: PASS (as-sent inverted, pre-trim ignores trims)")


def _jsonl_to_csv(src, dst):
    rows = [json.loads(l) for l in open(src) if l.strip()]
    with open(dst, "w") as f:
        f.write("# " + json.dumps(rows[0]["header"]) + "\n")
        f.write("t,s,roll,pitch,yaw,gr,gp,gy,l,r\n")
        for r in rows[1:]:
            if r["s"] == "imu":
                f.write(f"{r['t']},imu," + ",".join(str(v) for v in r["o"]) + ",,\n")
            elif r["s"] == "cmd":
                f.write(f"{r['t']},cmd,,,,,,,{r['l']},{r['r']}\n")
            elif r["s"] == "ev":
                f.write(f"{r['t']},ev,{r['name']},,,,,,,\n")


def _jsonl_to_growbot_v1(src, dst, cal=None, gait=None, end_why="done"):
    """Re-emit an internal-dialect session as growbot-imulog-1, inverting every conversion.

    The equivalence standard of _jsonl_to_csv, applied to the real upstream format:
    the same physical session written in both dialects must parse to the same arrays.
    The emission runs the exact inverse of _convert_growbot_v1 -- twin rpy back
    through the mount to W3C deviceorientation degrees, twin body rates back to
    device rates, twin actions back through the upstream servo map with a cal of
    its own (signs, offsets, gain AND a nonzero turn, so the per-side turn folding
    is exercised) -- and the parser must undo all of it.
    """
    cal = cal or {"L_SIGN": -1, "R_SIGN": -1, "L_OFF": 2.0, "R_OFF": -3.0,
                  "IMU_SIGN": [1, 1, 1], "gain": 0.99, "turn": 1.5, "SWAP": 0}
    rows = [json.loads(l) for l in open(src) if l.strip()]
    h = rows[0]["header"]
    M = GROWBOT_V1_MOUNT
    imu, pose = [], []
    iseq = pseq = 0
    for r in rows[1:]:
        if r["s"] == "imu":
            o = np.asarray(r["o"], np.float64)
            R_t = _rz(np.float64(o[2])) @ _ry(np.float64(o[1])) @ _rx(np.float64(o[0]))
            R_d = R_t @ M.T
            al, be, ga = _R_to_deviceorientation(R_d)
            w_dev = M @ o[3:]
            acc = R_d.T @ np.array([0.0, 0.0, 9.81])
            imu.append([iseq, r["t"], *np.round(acc, 6),
                        *np.round(np.rad2deg(w_dev), 6),
                        round(float(np.rad2deg(al)) % 360.0, 6),
                        round(float(np.rad2deg(be)), 6), round(float(np.rad2deg(ga)), 6)])
            iseq += 1
        elif r["s"] == "cmd":
            # internal fixture convention: column l carries action[0], r carries action[1]
            a0 = np.deg2rad((r["l"] - 90 - h["l_off"]) / (h["l_sign"] * h["gain"]))
            a1 = np.deg2rad((r["r"] - 90 - h["r_off"]) / (h["r_sign"] * h["gain"]))
            l = 90 + cal["L_OFF"] + cal["L_SIGN"] * np.rad2deg(a1) * cal["gain"] + cal["turn"]
            r_ = 90 + cal["R_OFF"] + cal["R_SIGN"] * np.rad2deg(a0) * cal["gain"] - cal["turn"]
            pose.append([pseq, r["t"], round(float(l), 6), round(float(r_), 6), 1])
            pseq += 1
    out = {"header": {"format": "growbot-imulog-1", "app": "fixture",
                      "walk": 0, "end_why": end_why,
                      "units": {"accel": "m/s^2", "rate": "deg/s",
                                "ori": "deg (deviceorientation alpha,beta,gamma)",
                                "pose": "servo/wheel command deg 0-180, 90 = neutral/stopped",
                                "t": "ms, performance.now() monotonic"},
                      "gravity_included": True,
                      "axes": "device frame, DeviceMotionEvent convention",
                      "imu_fields": ["seq", "t_ms", "ax", "ay", "az", "rate_alpha",
                                     "rate_beta", "rate_gamma", "ori_alpha", "ori_beta", "ori_gamma"],
                      "pose_fields": ["seq", "t_ms", "l", "r", "send_ok"],
                      "gait": gait if gait is not None else h.get("gait", "unknown"),
                      "gain": None, "cal": cal, "body_id": "fixture-2leg", "wheels": 0,
                      "dropped_imu": 0, "dropped_pose": 0},
           "imu": imu, "pose": pose}
    with open(dst, "w") as f:
        f.write(json.dumps(out))


def preflight(path):
    """Contract checks that need no ground truth, run BEFORE any analysis.

    The round-trip test proves that information survives the pipeline IF the file
    speaks the twin's dialect; it cannot prove the phone speaks it, because the
    fixture shares every convention with the parser by construction. This closes
    what is closable from the file alone: units and rates fail hard, physics on
    labelled segments warns, and mount-dependent signs are reported for a human
    to confirm -- the preflight cannot know how the phone is mounted.

    Returns (ok, findings); ok is False only on FAIL-level findings.
    """
    header, (imu_t, imu_v, cmd_t, cmd_v), events = _read_rows(path)
    F, out = True, []
    def fail(m): out.append(("FAIL", m)); return False
    def warn(m): out.append(("WARN", m))
    def info(m): out.append(("ok", m))

    if len(imu_t) < 100 or len(cmd_t) < 10:
        return fail(f"too few rows (imu {len(imu_t)}, cmd {len(cmd_t)})"), out
    # --- timestamps: order, units, one clock ---
    order = header.get("_sorted_on_read", {})
    for name, k in (("IMU", "imu_inversions"), ("command", "cmd_inversions")):
        if order.get(k, 0):
            warn(f"{name} timestamps not sorted in the file ({order[k]} inversions) -- sorted on read, "
                 f"values moved with their timestamps")
    dt_i = float(np.median(np.diff(imu_t))); dt_c = float(np.median(np.diff(cmd_t)))
    if dt_i < 1.0:
        F = fail(f"IMU median dt = {dt_i:.4f}: timestamps look like SECONDS, expected milliseconds")
    else:
        info(f"effective rates: IMU {1000 / dt_i:.1f} Hz, commands {1000 / dt_c:.1f} Hz")
        if not (10 <= 1000 / dt_i <= 250): warn(f"IMU rate {1000 / dt_i:.1f} Hz far from the expected ~60")
        if not (5 <= 1000 / dt_c <= 100): warn(f"command rate {1000 / dt_c:.1f} Hz far from the expected ~30")
    from sensor_id import dt_stats, verify_still
    for name, ts in (("IMU", imu_t), ("command", cmd_t)):
        st = dt_stats(ts)
        line = (f"{name} dt: median {st['median_ms']:.1f} ms, p95 {st['p95_ms']:.1f}, "
                f"p99 {st['p99_ms']:.1f}, max {st['max_ms']:.0f}; "
                f"{st['frac_dev20'] * 100:.1f}% of ticks >20% off the median")
        if st["jitter_warn"]:
            warn(line + " -- p99 above 1.5x median: timing jitter degrades the fixed-dt "
                 "forward model (reported, never gated)")
        else:
            info(line)
        # Dropouts inside one file. A stall shorter than p99 hides in the quantiles
        # above but still splits an episode in parse() and a correlation segment in
        # sensor_id, so it is counted explicitly. Reported, never a FAIL: a real
        # Bluetooth session has these.
        d = np.diff(ts)
        n_gap = int((d > GAP_MS).sum())
        if n_gap:
            # What a gap costs depends on WHICH stream has it, and the two are not the
            # same. parse() cuts episodes on IMU gaps only; a command gap costs nothing
            # of the sort, because commands are zero-order held -- the last command
            # stays in force, which is what the robot actually did. Saying "these cut
            # episodes" of both streams was simply false for one of them.
            consequence = ("these cut episodes in parse() and are excluded from "
                           "sensor_id's interpolation, they are not interpolated across"
                           if name == "IMU" else
                           "the last command is zero-order held across them, which is what "
                           "the body did; they do not cut episodes, but a held command is "
                           "an assumption about a stretch the log does not observe")
            warn(f"{name} stream has {n_gap} dt above the {GAP_MS:.0f} ms gap threshold "
                 f"(max {d.max():.0f} ms) -- {consequence}")
    lo, hi = max(imu_t[0], cmd_t[0]), min(imu_t[-1], cmd_t[-1])
    overlap = max(0.0, hi - lo) / max(imu_t[-1] - imu_t[0], 1e-9)
    if overlap < 0.5:
        F = fail(f"IMU and command timestamp ranges overlap only {overlap * 100:.0f}% -- different clocks?")
    # --- units in practice, not in the header ---
    ang = np.abs(imu_v[:, :3])
    if header.get("imu_units", "rad") == "rad" and float(np.percentile(ang, 99)) > 7.0:
        F = fail(f"header says radians but 99th pct |angle| = {np.percentile(ang, 99):.1f} -- degrees in practice?")
    gyro99 = float(np.percentile(np.abs(imu_v[:, 3:]), 99))
    if gyro99 > 50:
        warn(f"99th pct |gyro| = {gyro99:.0f}: deg/s suspected (rad/s rarely exceeds ~20 on this body)")
    pose_rng = float(cmd_v.min()), float(cmd_v.max())
    if header.get("pose_units", "deg") == "deg" and (pose_rng[0] < -10 or pose_rng[1] > 190):
        F = fail(f"pose range {pose_rng} incompatible with degrees around 90 = neutral")
    if header.get("pose_units", "deg") == "deg" and pose_rng[1] < 3.2:
        F = fail(f"pose range {pose_rng} looks like RADIANS but the header says degrees")
    if header.get("pose_units") == "rad" and max(abs(pose_rng[0]), abs(pose_rng[1])) > 1.65:
        F = fail(f"pose range {pose_rng} exceeds the +-1.57 rad ctrlrange -- degrees in practice, "
                 f"or a wrong calibration inversion")
    if "trims_in_values" not in header:
        warn("header omits trims_in_values -- parser will assume as-sent; ask the emitter to state it")
    if str(header.get("format", "")).startswith("growbot-imulog"):
        info(f"growbot-imulog walk {header.get('walk')}, end_why={header.get('end_why')!r}, "
             f"gait={header.get('gait')!r}, agent gain={header.get('gain_agent')}, "
             f"cal gain={header.get('gain')}, send_ok dropped {header.get('send_ok_dropped', 0)} "
             f"of {header.get('n_pose_rows')} pose rows, app-side drops "
             f"imu={header.get('dropped_imu')} pose={header.get('dropped_pose')}")
    # --- physics on labelled segments ---
    def seg_spans(name):
        out = []; ev = sorted(events)
        for i, (t0, n) in enumerate(ev):
            if n.removesuffix("_start") == name and not n.endswith("_stop"):
                t1 = ev[i + 1][0] if i + 1 < len(ev) else imu_t[-1] + 1.0
                out.append((t0, t1))
        return out

    def seg_mask(name):
        m = np.zeros(len(imu_t), bool)
        for t0, t1 in seg_spans(name):
            m |= (imu_t >= t0) & (imu_t < t1)
        return m
    # Each still segment is verified SEPARATELY. Pooled, two motionless segments held
    # at different poses report the offset between them as motion -- the same mistake
    # verify_still's docstring rejects for the roll/pitch pair, one level up.
    still_spans = seg_spans("still") + seg_spans("idle")
    verdicts = []
    for t0, t1 in still_spans:
        m = (imu_t >= t0) & (imu_t < t1)
        if m.sum() > 50:
            verdicts.append(((t1 - t0) / 1000.0, verify_still(imu_v[m, :3], imu_v[m, 3:])))
    if verdicts:
        bad = [(d, v) for d, v in verdicts if not v["still"]]
        worst = max(verdicts, key=lambda dv: dv[1]["gyro_rms"])[1]
        nums = (f"worst gyro RMS {worst['gyro_rms']:.3f} rad/s (max {worst['gyro_rms_max']}), "
                f"max per-axis roll/pitch std {worst['ang_std']:.4f} rad (max {worst['ang_std_max']})")
        if bad:
            warn(f"{len(bad)} of {len(verdicts)} 'still' segments are not still: {nums} "
                 f"-- mislabelled segments or a unit/axis problem")
        else:
            info(f"{len(verdicts)} 'still' segment(s), longest {max(d for d, _ in verdicts):.1f} s, "
                 f"all check out ({nums})")
        # A still body under active commands is not a quiet moment: it is a body that
        # is not doing what it is told. walk-3 spends its first 2.6 s exactly there,
        # commands swinging +-30 deg against an orientation frozen byte-identical.
        if len(cmd_t):
            for t0, t1 in still_spans:
                m = (imu_t >= t0) & (imu_t < t1)
                if m.sum() <= 50: continue
                if not len(cmd_v): continue
                k = np.clip(np.searchsorted(cmd_t, imu_t[m], "right") - 1, 0, None)
                # cmd_v is whatever the file speaks: raw servo degrees around 90, or
                # (growbot-imulog-1) horn radians around 0. Compare in the file's units.
                if header.get("pose_units", "deg") == "deg":
                    drive = float(np.abs(cmd_v[k] - 90.0).max())
                else:
                    drive = float(np.rad2deg(np.abs(cmd_v[k]).max()))
                if drive >= SEG_CMD_ACTIVE_DEG:
                    warn(f"still segment {(t1 - t0) / 1000:.1f} s at t={t0:.0f} ms runs under "
                         f"ACTIVE commands (peak {drive:.0f} deg off neutral): the body is not "
                         f"responding to what it is sent -- phone not on the robot, robot not on "
                         f"the ground, or the servos are not moving. Nothing downstream can tell "
                         f"this from a quiet moment, so it is said here")
    for name in ("spin", "spin_ccw", "spin_cw"):
        m = seg_mask(name)
        if m.sum() > 50:
            yz = float(imu_v[m, 5].mean())
            info(f"'{name}' segment mean yaw rate {yz:+.2f} rad/s -- CONFIRM the sign matches the "
                 f"commanded direction (mount convention; the file alone cannot decide this)")
    return F, out


def run_preflight(path):
    ok, findings = preflight(path)
    for lvl, msg in findings:
        print(f"  [{lvl:>4}] {msg}")
    print(f"preflight: {'PASS' if ok else 'FAIL -- fix the contract before analysing'}")
    return ok


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1:                 # imulog.py <file> = standalone preflight
        raise SystemExit(0 if run_preflight(_sys.argv[1]) else 1)

    import itertools, time
    from forward import MLP, make_windows
    from sim2real_proxy import K, horizon_within
    from servo_id import (identify, realized_from_commands, confidence_band, determined_sets,
                          default_grid, realized_per_side, identify_per_side)

    _selfcheck()

    TRUE = dict(delay_ms=40, slew_rad_s=5.0, deadband=np.deg2rad(2))
    print("generating 600 s fixture (hidden servo: delay 40 ms, slew 5 rad/s, deadband 2 deg)...", flush=True)
    fixture("/tmp/imulog_fixture.jsonl", seconds=600, servo_ms=TRUE, seed=3)
    O, A, O2, D, header, mode = parse("/tmp/imulog_fixture.jsonl")
    from collections import Counter
    print(f"parsed: {len(O):,} ticks at 50 Hz, {int(D.sum())} episode splits, "
          f"regimes {dict(Counter(mode))}")
    _jsonl_to_csv("/tmp/imulog_fixture.jsonl", "/tmp/imulog_fixture.csv")
    Oc, Ac, O2c, Dc, hc, mc = parse("/tmp/imulog_fixture.csv")
    same = (np.allclose(O, Oc, atol=1e-4) and np.allclose(A, Ac, atol=1e-4)
            and (D == Dc).all() and (mode == mc).all())
    print(f"CSV fallback: {'PASS — identical arrays from both formats' if same else 'FAIL'}")
    assert same
    # growbot-imulog-1: the same session through the real upstream format. Twin rpy ->
    # mount -> W3C ZX'Y'' degrees -> back, body rates -> device rates -> back, actions
    # -> upstream servo map (signs, offsets, gain, nonzero turn) -> back. A wrong
    # composition order, a wrong extraction branch, a swapped l/r or an unfolded turn
    # all break the equality; the mount itself is validated against the real logs
    # (gravity, stance, action-response signatures), not here, because M M^T = I makes
    # any mount self-consistent in a round trip.
    _jsonl_to_growbot_v1("/tmp/imulog_fixture.jsonl", "/tmp/imulog_fixture_v1.json")
    Og, Ag, O2g, Dg, hg, mg = parse("/tmp/imulog_fixture_v1.json")
    ang_err = float(np.abs(np.arctan2(np.sin(O[:, :3] - Og[:, :3]), np.cos(O[:, :3] - Og[:, :3]))).max())
    gyro_err = float(np.abs(O[:, 3:] - Og[:, 3:]).max())
    act_err = float(np.abs(A - Ag).max())
    same_g = (ang_err < 1e-4 and gyro_err < 1e-4 and act_err < 1e-4 and (D == Dg).all()
              and hg.get("send_ok_dropped") == 0)
    print(f"growbot-imulog-1: {'PASS' if same_g else 'FAIL'} — max errs angle {ang_err:.1e} rad, "
          f"gyro {gyro_err:.1e} rad/s, action {act_err:.1e} rad through mount + W3C angles + servo cal "
          f"(signs, offsets, gain 0.99, turn 1.5)")
    assert same_g, "growbot-imulog-1 dialect does not round-trip"

    # --- growbot-imulog-1 segmentation: the regimes the format does not carry -------
    # Hidden secrets: a 10 s motionless prefix, driven walking after it, and a tip at
    # 34.0 s that the body never gets up from. The file the segmenter sees carries no
    # event row at all -- only end_why in the header -- so every boundary below has to
    # come out of the IMU and pose streams.
    SEG = dict(still_lead_s=10.0, tip_at_s=34.0, seconds=45.0, seed=11)
    print(f"\ngenerating {SEG['seconds']:.0f} s segmentation fixture (hidden: still prefix "
          f"{SEG['still_lead_s']:.0f} s, tip at {SEG['tip_at_s']:.0f} s)...", flush=True)
    fixture("/tmp/imulog_seg_fixture.jsonl", seconds=SEG["seconds"], seed=SEG["seed"],
            still_lead_s=SEG["still_lead_s"], tip_at_s=SEG["tip_at_s"])
    _jsonl_to_growbot_v1("/tmp/imulog_seg_fixture.jsonl", "/tmp/imulog_seg_v1.json",
                         end_why="tipped")
    from sensor_id import verify_still
    hs, (its, ivs, _, _), evs = _read_rows("/tmp/imulog_seg_v1.json")
    t0 = its[0]
    segs = [((t - t0) / 1000.0, n.removesuffix("_start")) for t, n in evs]
    span = [(a, b, n) for (a, n), (b, _) in zip(segs, segs[1:] + [((its[-1] - t0) / 1000.0, "")])]
    print("  segments: " + " | ".join(f"{a:.1f}-{b:.1f}s {n}" for a, b, n in span))
    first_end = span[0][1]
    fall = [s for s in span if s[2] == "fall"]
    still_ok = span[0][2] == "still" and abs(first_end - SEG["still_lead_s"]) <= 0.5
    walk_ok = any(n == "walking" and b - a >= 2.0 for a, b, n in span)
    fall_ok = (len(fall) == 1 and fall[0] is span[-1]
               and abs(fall[0][0] - SEG["tip_at_s"]) <= 1.0)
    print(f"  still prefix {span[0][2]!r} ends {first_end:.2f} s (injected "
          f"{SEG['still_lead_s']:.1f}): {'ok' if still_ok else 'WRONG'}")
    print(f"  a driven walking segment of >= 2 s exists: {'ok' if walk_ok else 'MISSING'}")
    print(f"  fall {'at %.2f s to the end' % fall[0][0] if fall else 'NOT FOUND'} (injected tip "
          f"{SEG['tip_at_s']:.1f}): {'ok' if fall_ok else 'WRONG'}")
    assert still_ok, f"the motionless prefix did not come out as one still segment: {span[:2]}"
    assert walk_ok, f"no driven walking segment recovered: {span}"
    assert fall_ok, f"the injected tip did not come out as the final fall segment: {span}"
    # the labels must survive their own physics check, or they are decoration.
    # PER SEGMENT, never pooled: two still segments held at different poses pool into
    # the offset between them, which verify_still's docstring calls a tilt, not motion.
    checks = {"still": [], "walking": []}
    for a, b, n in span:
        if n not in checks: continue
        m = ((its - t0) / 1000.0 >= a) & ((its - t0) / 1000.0 < b)
        if m.sum() >= 20:
            checks[n].append(verify_still(ivs[m, :3], ivs[m, 3:])["still"])
    print(f"  labelled still verifies still on {sum(checks['still'])}/{len(checks['still'])} "
          f"segments; labelled walking on {sum(checks['walking'])}/{len(checks['walking'])} "
          f"(must be 0)")
    assert checks["still"] and all(checks["still"]), "a labelled still segment is not still"
    assert not any(checks["walking"]), "a labelled walking segment verifies as still"
    # negative control: the SAME bytes with end_why 'done' must produce NO fall. The
    # class is gated on the header's own claim, never invented from attitude alone.
    _jsonl_to_growbot_v1("/tmp/imulog_seg_fixture.jsonl", "/tmp/imulog_seg_v1_done.json",
                         end_why="done")
    _, _, ev_done = _read_rows("/tmp/imulog_seg_v1_done.json")
    n_fall_done = sum(1 for _, n in ev_done if n == "fall_start")
    print(f"  same session with end_why='done': {n_fall_done} fall segments (must be 0)")
    assert n_fall_done == 0, "a fall was invented on a session the header does not call tipped"
    print("SEGMENTATION PASS - still prefix, driven walking and the tip recovered from a "
          "format that carries no event rows")
    # permanent detector for the cut-boundary bug family: no valid window may have a
    # target that crosses a cut, and every cut must cost at least one window
    *_, valid = make_windows(O, A, O2, D, K)
    assert not (valid & D).any(), "a window whose target crosses a cut leaked into make_windows"
    print(f"cut coherence: {int(D.sum())} cut transitions, none inside valid windows "
          f"({int(valid.sum()):,} valid of {len(D):,})")

    tr = np.load("data/train.npz")
    Xtr, Ytr, *_ = make_windows(tr["obs"], tr["act"], tr["next_obs"], tr["done"], K)
    model = MLP(hidden=128, epochs=60).fit(Xtr, Ytr)

    half = len(O) // 2
    fit, held = slice(0, half), slice(half, None)
    # the grid that SHIPS, not a copy of it: there were three copies of a narrow grid
    # (here, servo_id's CLI, gap_report's), and the one the real log needed was a fourth,
    # written inline in the real-log report. A test that exercises a private copy cannot
    # catch a default that pins at its own boundary.
    grid = default_grid()
    scores, best = identify(model, O[fit], A[fit], O2[fit], D[fit], grid)
    print("\nservo identification on the PARSED log (top 3):")
    for e, kw in scores[:3]:
        print(f"  err {e:.4f}  delay {kw['delay_ticks']}  slew {kw['slew_rad_s']}  deadband {np.rad2deg(kw['deadband']):.0f} deg")
    R_est = realized_from_commands(A, D, best)
    for h in (5, 25):
        c = horizon_within(model, O[held], A[held], D[held], h=h)[0]
        e = horizon_within(model, O[held], R_est[held], D[held], h=h)[0]
        print(f"  {h*20:>3} ms  within 0.2 rad: commanded {c*100:5.1f}%  identified servo {e*100:5.1f}%")
    # Acceptance is servo_id's own standard -- the determined SET, not the argmin.
    # The valley here is nearly flat (the top three hypotheses differ by ~2e-4 on
    # errors of ~0.4), so the argmin is a coin flip, and asserting it was a test that
    # passed for the wrong reason: it leaned on the ~30 rows the fixture used to
    # backfill at 200 Hz after every repositioning gap, right where the post-reset
    # transient is most informative. With that artifact gone the argmin lands one
    # grid step off the injected delay on two of the five seeds tested (0/3/5/7/11),
    # while the determined set still contains the truth on all five.
    #
    # The bound asserted below is what held on all five seeds, not what looks tidy:
    #   delay  the injected 2 ticks is IN the set, and the set never leaves 2 +-1
    #          grid step (measured sets: [1,2,3] [1,2] [2] [1,2] [1,2,3])
    #   slew   every member of the set is within one grid step of the injected
    #          5.0 rad/s (measured: [4,5,6] [4,5] [4,5,6] [4] [4,5,6]) -- containment
    #          is NOT asserted, because seed 7 determines [4.0] and excludes the
    #          truth. The 200 Hz burst was the fixture's most slew-informative data;
    #          without it this log resolves slew to one grid step, and claiming more
    #          would be the same mistake in a new place.
    # Both clauses still exclude the answers that would make the test vacuous: delay
    # 0 (no servo at all) and slew 3.0, 8.0 or None (no slew limit).
    hA, hB = slice(0, half // 2), slice(half // 2, half)
    sA, _ = identify(model, O[hA], A[hA], O2[hA], D[hA], grid)
    sB, _ = identify(model, O[hB], A[hB], O2[hB], D[hB], grid)
    band = confidence_band(sA, sB)
    delay_set, slew_set = determined_sets(scores, best, grid, band)
    true_ticks = round(TRUE["delay_ms"] / 20)
    slews = sorted({s for _, s, _ in grid}, key=lambda v: (v is None, v))
    si = slews.index(TRUE["slew_rad_s"])
    near_slew = set(slews[max(0, si - 1):si + 2])
    delay_ok = (true_ticks in delay_set
                and set(delay_set) <= {true_ticks - 1, true_ticks, true_ticks + 1})
    slew_ok = bool(slew_set) and set(slew_set) <= near_slew
    print(f"\ndetermined sets at the split-half band {band:.5f} (argmin was delay "
          f"{best['delay_ticks']}, slew {best['slew_rad_s']}):")
    print(f"  delay {delay_set} ticks -- injected {true_ticks} "
          f"{'inside the set' if true_ticks in delay_set else 'OUTSIDE the set'}, "
          f"set within +-1 grid step: {set(delay_set) <= {true_ticks - 1, true_ticks, true_ticks + 1}}")
    print(f"  slew  {slew_set} rad/s -- injected {TRUE['slew_rad_s']} "
          f"{'inside the set' if TRUE['slew_rad_s'] in slew_set else 'outside the set'}, "
          f"every member within one grid step: {slew_ok}")
    # Per-side identification must not invent an asymmetry the fixture does not have.
    # One servo drives both horns here, so a per-side search that lands on two different
    # answers is reading noise, and the coordinate descent must come back to the shared
    # solution's neighbourhood rather than away from it.
    #
    # This bound is ASSERTED, not printed. It used to be folded into the PASS/FAIL string
    # and then dropped, so the process exited 0 on a failure and the line above it was the
    # only trace. It also constrained the two delays only, while the thing the report
    # publishes as the asymmetry is the SLEW; and its second clause,
    # best_err <= scores[0][0] + 1e-9, was true by construction -- the descent starts AT
    # the shared solution and only accepts strict improvements, so it could never fail.
    # That clause is replaced by one that can: on a symmetric fixture the per-side fit
    # must not SEPARATE from the shared fit, i.e. its gain must not exceed the log's own
    # confidence band. That is the repo's own criterion for "this log can see it", pointed
    # at the claim per-side actually makes.
    kw_l, kw_r, ps_info = identify_per_side(model, O[fit], A[fit], O2[fit], D[fit], grid, best)
    ps_delays = {kw_l["delay_ticks"], kw_r["delay_ticks"]}
    ps_slews = {kw_l["slew_rad_s"], kw_r["slew_rad_s"]}
    ps_delay_ok = ps_delays <= {true_ticks - 1, true_ticks, true_ticks + 1}
    ps_slew_ok = ps_slews <= near_slew
    ps_gain = scores[0][0] - ps_info["best_err"]
    ps_gain_ok = ps_gain <= band
    ps_ok = ps_delay_ok and ps_slew_ok and ps_gain_ok
    print(f"  per side: L(delay {kw_l['delay_ticks']}, slew {kw_l['slew_rad_s']})  "
          f"R(delay {kw_r['delay_ticks']}, slew {kw_r['slew_rad_s']})  "
          f"in {ps_info['evaluations']} evaluations; err {ps_info['best_err']:.5f} vs shared "
          f"{scores[0][0]:.5f}")
    print(f"    both delays within one grid step of the single injected servo: {ps_delay_ok}; "
          f"both slews: {ps_slew_ok}")
    print(f"    per-side fit gain over shared {ps_gain:.5f} vs band {band:.5f} -- "
          f"{'not separated, as a symmetric fixture requires' if ps_gain_ok else 'SEPARATED: an asymmetry was invented'}")
    assert delay_ok, f"injected delay {true_ticks} ticks not determined: set {delay_set}"
    assert slew_ok, f"injected slew {TRUE['slew_rad_s']} rad/s not determined: set {slew_set}"
    assert ps_delay_ok, (f"per-side delays {sorted(ps_delays)} left the injected "
                         f"{true_ticks} +-1 grid step")
    assert ps_slew_ok, (f"per-side slews {sorted(ps_slews, key=lambda v: (v is None, v))} "
                        f"left one grid step of the injected {TRUE['slew_rad_s']}: "
                        f"an asymmetry the fixture does not have")
    assert ps_gain_ok, (f"per-side fit beats the shared fit by {ps_gain:.5f} > band {band:.5f} "
                        f"on a fixture with ONE servo driving both horns: invented asymmetry")

    # --- per-side ATTRIBUTION: which horn, not just how much -----------------------
    # The check above is symmetric, so it passes unchanged if the left and right labels
    # are swapped -- and they were: realized_per_side put the left triple on action
    # column 0, which imulog.parse and the twin both define as the RIGHT leg, so every
    # published per-side attribution was inverted while every error number stayed
    # identical. Only an ASYMMETRIC fixture can catch that.
    #
    # And only one anchored on GROUND TRUTH. An asymmetric fixture injected through
    # servo_id.RIGHT_COL / LEFT_COL and then labelled by the identification -- which
    # reads the same two constants -- tests self-consistency: set them to 1, 0 and the
    # injection moves with the label, so the guard stays green while every published
    # attribution inverts. The physical convention lives in the twin's XML
    # (right_leg -> joint_1 -> actuator servo_1 = ctrl[0]) and in the policy head
    # (a = np.tanh(x[:2])  # [aRight, aLeft]), not in the constants. So both of these,
    # neither redundant with the other:
    #   (a) the constants are asserted against the XML, read independently by
    #       servo_id.sim_side_columns;
    #   (b) the crippled horn is injected BY COLUMN, on the column that XML says is the
    #       right leg -- so under a reversed constant the slow horn physically stays on
    #       the right and the identification hands it back as 'left', and side_ok fails.
    from growbot_sim import collect as _collect
    from servo_id import PerSideServo, slower_side, sim_side_columns, check_side_convention
    conv_ok, conv_why = check_side_convention("walk")
    cols = sim_side_columns("walk")
    ASYM = dict(n=6000, seed=5,
                fast=dict(delay_ticks=0, slew_rad_s=None, deadband=0.0),
                slow=dict(delay_ticks=5, slew_rad_s=1.5, deadband=0.0),
                slow_horn="right")
    fast_horn = "left" if ASYM["slow_horn"] == "right" else "right"
    print(f"\n  {conv_why}")
    assert conv_ok, conv_why
    print(f"generating {ASYM['n'] / 50:.0f} s asymmetric fixture (hidden: {ASYM['slow_horn']} horn "
          f"delay {ASYM['slow']['delay_ticks']} ticks / slew {ASYM['slow']['slew_rad_s']} rad/s, "
          f"the other horn ideal; injected on action column {cols[ASYM['slow_horn']]}, "
          f"which the XML says drives the {ASYM['slow_horn']} leg)...", flush=True)
    per_side_servo = PerSideServo(by_column={cols[ASYM["slow_horn"]]: ASYM["slow"],
                                            cols[fast_horn]: ASYM["fast"]})
    Oy, Ay, O2y, Dy, _ = _collect(ASYM["n"], seed=ASYM["seed"], body="walk", servo=per_side_servo)
    _, best_y = identify(model, Oy, Ay, O2y, Dy, grid)
    kw_ly, kw_ry, _ = identify_per_side(model, Oy, Ay, O2y, Dy, grid, best_y)
    got_slow = slower_side(kw_ly, kw_ry)
    slow_kw = kw_ry if ASYM["slow_horn"] == "right" else kw_ly
    si_a = slews.index(ASYM["slow"]["slew_rad_s"])
    near_slow = set(slews[max(0, si_a - 1):si_a + 2])
    side_ok = got_slow == ASYM["slow_horn"]
    slow_slew_ok = slow_kw["slew_rad_s"] in near_slow
    print(f"  identified: L(delay {kw_ly['delay_ticks']}, slew {kw_ly['slew_rad_s']})  "
          f"R(delay {kw_ry['delay_ticks']}, slew {kw_ry['slew_rad_s']})")
    print(f"  slower horn identified as {got_slow!r}, injected on {ASYM['slow_horn']!r}: "
          f"{'ok' if side_ok else 'WRONG SIDE -- the left/right labels are inverted'}")
    print(f"  that horn's slew {slow_kw['slew_rad_s']} within one grid step of the injected "
          f"{ASYM['slow']['slew_rad_s']}: {slow_slew_ok}")
    assert side_ok, (f"the slow horn was injected on action column "
                     f"{cols[ASYM['slow_horn']]}, which the XML says is the "
                     f"{ASYM['slow_horn']} leg, and came back on the {got_slow}: "
                     f"per-side left/right attribution is inverted")
    assert slow_slew_ok, (f"the crippled horn's slew came back {slow_kw['slew_rad_s']}, more "
                          f"than one grid step from the injected {ASYM['slow']['slew_rad_s']}")

    ok = conv_ok and delay_ok and slew_ok and ps_ok and side_ok and slow_slew_ok
    print("\nROUND-TRIP", "PASS" if ok else "FAIL",
          "- delay and slew determined to one grid step through 60/30 Hz jittered sampling, "
          "the per-side search stays on the symmetric answer, and on a deliberately "
          "asymmetric fixture -- injected on the action column the twin's XML says is the "
          "right leg, not on the column servo_id's constants say it is -- the slow horn "
          "comes back on the side it was injected on"
          if ok else "")
    assert ok

    # --- sensor-side round-trip: the same standard, applied to sensor_id.py --------
    from sensor_id import (dt_stats, allan_deviation, filter_lag, still_windows,
                           euler_rates_to_body, verify_still, segment_rate, BODY_AXES)

    # (1) the euler-rate -> body-rate map, against an INDEPENDENT derivation.
    # A pure delay commutes with any static mixing of the axes, so a sign-flipped or
    # transposed kinematic map still recovers the injected lag below: that assert
    # proves the correlation works, never that the kinematics are right. Ground truth
    # here is w^ = R^T dR/dt on an analytic trajectory, which shares no line of code
    # with the formula under test.
    def _R(phi, th, psi):
        cz, sz = np.cos(psi), np.sin(psi)
        cy, sy = np.cos(th), np.sin(th)
        cx, sx = np.cos(phi), np.sin(phi)
        return (np.array([[cz, -sz, 0.], [sz, cz, 0.], [0., 0., 1.]])
                @ np.array([[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]])
                @ np.array([[1., 0., 0.], [0., cx, -sx], [0., sx, cx]]))

    def _traj(tt):                       # |pitch| to 1.0 rad, yaw winding through +-pi
        tt = np.atleast_1d(np.asarray(tt, np.float64))
        return np.stack([0.9 * np.sin(2 * np.pi * 0.31 * tt + 0.4),
                         1.0 * np.sin(2 * np.pi * 0.23 * tt),
                         np.pi * tt], 1)

    dtk = 1 / 60.0
    tt = np.arange(0.0, 20.0, dtk)
    h = 1e-6
    w_true = np.empty((len(tt), 3))
    for i, ti in enumerate(tt):
        S = _R(*_traj(ti)[0]).T @ ((_R(*_traj(ti + h)[0]) - _R(*_traj(ti - h)[0])) / (2 * h))
        w_true[i] = [S[2, 1], S[0, 2], S[1, 0]]
    ang_true = _traj(tt)
    wrapped = np.arctan2(np.sin(ang_true), np.cos(ang_true))      # the map must unwrap itself
    w_pred = euler_rates_to_body(wrapped, dtk)
    inner = slice(3, -3)                 # np.gradient is one-sided at the ends
    kerr = np.abs(w_pred[inner] - w_true[inner]).max(0)
    print(f"\neuler-rate -> body-rate map vs R^T dR/dt, analytic trajectory "
          f"({tt[-1] * 0.5:.0f} yaw turns, |pitch| <= 1.0 rad, dt {dtk * 1000:.1f} ms):")
    print("  max |err| " + "  ".join(f"{BODY_AXES[a]} {kerr[a]:.1e}" for a in range(3))
          + " rad/s   (np.gradient truncation; a sign or convention flip is O(1))")
    assert (kerr < 5e-3).all(), f"euler->body kinematics wrong on some axis: {kerr}"

    SENSOR = dict(fused_lag_ms=60.0, gyro_arw=2e-3, still_lead_s=120.0, imu_slow=(300.0, 30.0, 3.0))
    print("\ngenerating 600 s sensor fixture (hidden: fused-filter lag 60 ms, gyro ARW 2e-3 "
          "rad/s/sqrt(Hz), 30 s of 3x-slow IMU at 300 s)...", flush=True)
    fixture("/tmp/imulog_sensor_fixture.jsonl", seconds=600, seed=4, **SENSOR)
    _, (it, iv, ct, cv), ev = _read_rows("/tmp/imulog_sensor_fixture.jsonl")
    st = dt_stats(it)
    print(f"dt stats: median {st['median_ms']:.1f} ms  p99 {st['p99_ms']:.1f} ms  "
          f"max {st['max_ms']:.0f} ms  jitter_warn {st['jitter_warn']}")
    assert st["jitter_warn"], "dt_stats missed the injected 3x IMU slowdown"
    res = filter_lag(iv[:, :3], it, iv[:, 3:], it)
    print(f"  grid {res['grid_dt_ms']:.1f} ms, {res['segments_used']} segments used / "
          f"{res['segments_dropped']} dropped, {res['excluded_ms'] / 1000:.1f} s excluded at gaps, "
          f"gimbal mask keeps {res['gimbal_kept_frac'] * 100:.1f}% of {res['grid_samples']:,} samples")
    for r in res["axes"]:
        print(f"  filter lag {r['axis']:>3}: " + (f"{r['lag_ms']:+6.1f} ms (corr {r['corr']:.2f})"
              if r["determined"] else f"undetermined (corr {r['corr']:.2f}) -- {r['reason']}"))
    lags = {r["axis"]: r for r in res["axes"]}
    det = [r for r in res["axes"] if r["determined"]]
    assert len(det) >= 2, f"filter lag determined on only {len(det)} of 3 axes"
    # wz carries the yaw wrap: if unwrapping or the segment split is wrong, this is the
    # axis that breaks, and 'two of three determined' would have hidden it.
    assert lags["wz"]["determined"], (f"wz undetermined (corr {lags['wz']['corr']:.2f}) -- yaw is "
                                      f"the only wrap-sensitive axis and it must be recovered")
    for r in det:
        assert abs(r["lag_ms"] - SENSOR["fused_lag_ms"]) <= 10.0, \
            f"{r['axis']} lag {r['lag_ms']:.1f} ms vs injected {SENSOR['fused_lag_ms']:.0f} ms"
        assert r["corr"] >= 0.6, f"{r['axis']} reported a lag on corr {r['corr']:.2f}"
        assert not r["boundary"], f"{r['axis']} peak pegged at the search boundary"

    t0, t1 = max(still_windows(ev, it[-1]), key=lambda w: w[1] - w[0])
    sel = (it >= t0) & (it < t1)
    vs = verify_still(iv[sel, :3], iv[sel, 3:])
    fs, rate_ok, rate_why = segment_rate(it[sel])
    print(f"  still segment {(t1 - t0) / 1000:.0f} s: gyro RMS {vs['gyro_rms']:.4f} rad/s, "
          f"max per-axis roll/pitch std {vs['ang_std']:.4f} rad, still={vs['still']}; "
          f"fs {fs:.2f} Hz, one-rate check {'ok' if rate_ok else rate_why}")
    assert vs["still"], "the fixture's labelled still segment must verify as still"
    assert rate_ok, f"the still segment must support one fs: {rate_why}"
    # the gate itself, on timestamps built for it: healthy jitter must pass, and a
    # stall inside the segment must not, or Allan would integrate at the wrong rate
    rr = np.random.default_rng(1)
    clean = np.cumsum(np.full(4000, 16.7) * (1 + rr.normal(0, 0.04, 4000))) + rr.normal(0, 2.0, 4000)
    stalled = clean + np.concatenate([np.zeros(3000), np.arange(1000) * 33.4])
    _, ok_clean, _ = segment_rate(np.sort(clean))
    _, ok_stall, _ = segment_rate(np.sort(stalled))
    print(f"  one-rate gate: healthy 60 Hz jitter -> {'accepted' if ok_clean else 'REFUSED'}; "
          f"3x stall over a quarter -> {'ACCEPTED' if ok_stall else 'refused'}")
    assert ok_clean, "the one-rate gate refuses ordinary phone jitter"
    assert not ok_stall, "the one-rate gate misses a 3x stall inside the segment"
    for a, r in enumerate(allan_deviation(iv[sel, 3:], fs)):
        n_est = r["arw"]; b_est = r["bias_instability"]
        print(f"  ARW {BODY_AXES[a]}: {'undetermined' if n_est is None else f'{n_est:.2e}'} "
              f"rad/s/sqrt(Hz) (injected {SENSOR['gyro_arw']:.0e}); bias instability "
              + ("undetermined" if b_est is None else f"{b_est:.2e} rad/s"))
        assert n_est is not None and abs(n_est / SENSOR["gyro_arw"] - 1) <= 0.20, \
            f"axis {a} ARW {n_est} vs injected {SENSOR['gyro_arw']}"
        assert r["bias_instability"] is None, \
            f"white-only gyro reported a bias instability ({r['bias_instability']:.2e})"

    # (2) negative control for bias instability: angle random walk PLUS rate random
    # walk and no flicker floor anywhere. The Allan curve has a real interior minimum
    # -- the ARW/RRW crossover -- and B = adev_min / 0.664 there would be a number for
    # a noise process the sensor does not have. It must come back undetermined while
    # the ARW, which IS in the data, is still recovered.
    RRW = dict(gyro_arw=2e-3, gyro_rrw=3.5e-4, still_lead_s=150.0)
    print("\ngenerating 160 s ARW+RRW fixture (crossover minimum, no flicker floor)...", flush=True)
    fixture("/tmp/imulog_rrw_fixture.jsonl", seconds=160, seed=5, **RRW)
    _, (it2, iv2, _, _), ev2 = _read_rows("/tmp/imulog_rrw_fixture.jsonl")
    t0, t1 = max(still_windows(ev2, it2[-1]), key=lambda w: w[1] - w[0])
    sel2 = (it2 >= t0) & (it2 < t1)
    fs2 = 1000.0 / float(np.median(np.diff(it2[sel2])))
    print(f"  still segment {(t1 - t0) / 1000:.0f} s, {int(sel2.sum()):,} samples at {fs2:.1f} Hz")
    for a, r in enumerate(allan_deviation(iv2[sel2, 3:], fs2)):
        n_est = r["arw"]; b_est = r["bias_instability"]
        print(f"  {BODY_AXES[a]}: ARW {'undetermined' if n_est is None else f'{n_est:.2e}'} "
              f"(injected {RRW['gyro_arw']:.0e});  bias instability "
              + (f"undetermined -- {r['bias_reason']}" if b_est is None else f"{b_est:.2e} rad/s"))
        assert r["bias_instability"] is None, \
            f"axis {a}: an ARW/RRW crossover was converted into a bias instability"
        assert n_est is not None and abs(n_est / RRW["gyro_arw"] - 1) <= 0.20, \
            f"axis {a} ARW {n_est} vs injected {RRW['gyro_arw']} under a rate random walk"

    print("\nSENSOR ROUND-TRIP PASS - kinematics, fused-filter lag (with the yaw axis), gyro "
          "noise density, the refused bias instability and the timing stall, from the file alone")
