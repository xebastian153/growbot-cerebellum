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


def _read_rows(path):
    """JSONL or CSV, sniffed from the first non-empty line."""
    header, imu_t, imu_v, cmd_t, cmd_v = {}, [], [], [], []
    with open(path) as f:
        first = ""
        for first in f:
            if first.strip(): break
        rest = f
        s = first.strip()
        if s.startswith("{"):                       # JSON-lines
            for line in [first, *rest]:
                line = line.strip()
                if not line: continue
                row = json.loads(line)
                if "header" in row: header = row["header"]; continue
                if row.get("s") == "imu": imu_t.append(row["t"]); imu_v.append(row["o"])
                elif row.get("s") == "cmd": cmd_t.append(row["t"]); cmd_v.append([row["l"], row["r"]])
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
    return header, (np.asarray(imu_t, np.float64), np.asarray(imu_v, np.float32),
                    np.asarray(cmd_t, np.float64), np.asarray(cmd_v, np.float32))


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


def parse(path, gap_ms=100.0):
    header, (imu_t, imu_v, cmd_t, cmd_v) = _read_rows(path)
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
    s = np.stack([np.interp(grid, imu_t, np.sin(ang[:, a])) for a in range(3)], 1)
    c = np.stack([np.interp(grid, imu_t, np.cos(ang[:, a])) for a in range(3)], 1)
    ang_g = np.arctan2(s, c)
    gyro_g = np.stack([np.interp(grid, imu_t, gyro[:, a]) for a in range(3)], 1)
    obs = np.concatenate([ang_g, gyro_g], 1).astype(np.float32)
    # commands: zero-order hold (last command at or before grid time; neutral before the first)
    idx = np.searchsorted(cmd_t, grid, side="right") - 1
    act = np.where(idx[:, None] >= 0, cmd_v[np.clip(idx, 0, None)], 0.0).astype(np.float32)
    O, A, O2 = obs[:-1], act[:-1], obs[1:]
    # a transition is invalid if EITHER endpoint was interpolated inside a gap
    D = (in_gap[:-1] | in_gap[1:]).copy(); D[-1] = True
    return O, A, O2, D, header


def fixture(path, seconds=600, servo_ms=None, seed=0, imu_hz=60.0, cmd_hz=30.0, jitter_ms=2.0,
            push_prob_s=0.5):
    """Synthetic ?imulog=1 session from the twin: physics at 200 Hz, jittered sampling.

    servo_ms: dict(delay_ms=, slew_rad_s=, deadband=) -- delay given in MILLISECONDS
    and converted to physics-rate calls here, because ServoModel counts calls and the
    fixture steps it at physics rate, not at 50 Hz. (The first version passed 50 Hz
    ticks straight through and simulated a 10 ms delay while believing it was 40; the
    round-trip test below is what caught it.)
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
    obs = sim.reset(tilt=0.3); prev = np.zeros(2, np.float32)
    n_steps = int(seconds / phys_dt)
    for i in range(n_steps):
        if t >= next_cmd:
            a = exc(obs, prev); prev = cur_cmd = a
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
        if rng.random() < push_prob_s * phys_dt:
            sim.push()
        if t >= next_imu:
            obs = sim.obs()
            rows.append({"t": round(t * 1000 + rng.normal(0, jitter_ms), 2), "s": "imu",
                         "o": [round(float(v), 5) for v in obs]})
            next_imu += 1.0 / imu_hz * (1 + rng.normal(0, 0.03))
        if sim.fallen() and rng.random() < 0.002:
            obs = sim.reset(tilt=0.3); prev = np.zeros(2, np.float32)
            if sim.servo is not None: sim.servo.reset()
            t += 0.5  # a real gap: the app was repositioned
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


if __name__ == "__main__":
    import itertools, time
    from forward import MLP, make_windows
    from sim2real_proxy import K, horizon_within
    from servo_id import identify, realized_from_commands

    _selfcheck()

    TRUE = dict(delay_ms=40, slew_rad_s=5.0, deadband=np.deg2rad(2))
    print("generating 600 s fixture (hidden servo: delay 40 ms, slew 5 rad/s, deadband 2 deg)...", flush=True)
    fixture("/tmp/imulog_fixture.jsonl", seconds=600, servo_ms=TRUE, seed=3)
    O, A, O2, D, header = parse("/tmp/imulog_fixture.jsonl")
    print(f"parsed: {len(O):,} ticks at 50 Hz, {int(D.sum())} episode splits, header gait={header.get('gait')}")
    _jsonl_to_csv("/tmp/imulog_fixture.jsonl", "/tmp/imulog_fixture.csv")
    Oc, Ac, O2c, Dc, hc = parse("/tmp/imulog_fixture.csv")
    same = np.allclose(O, Oc, atol=1e-4) and np.allclose(A, Ac, atol=1e-4) and (D == Dc).all()
    print(f"CSV fallback: {'PASS — identical arrays from both formats' if same else 'FAIL'}")
    assert same
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
    grid = list(itertools.product([0, 1, 2, 3], [3.0, 4.0, 5.0, 6.0, 8.0, None],
                                  [0.0, np.deg2rad(1), np.deg2rad(2), np.deg2rad(4)]))
    scores, best = identify(model, O[fit], A[fit], O2[fit], D[fit], grid)
    print("\nservo identification on the PARSED log (top 3):")
    for e, kw in scores[:3]:
        print(f"  err {e:.4f}  delay {kw['delay_ticks']}  slew {kw['slew_rad_s']}  deadband {np.rad2deg(kw['deadband']):.0f} deg")
    R_est = realized_from_commands(A, D, best)
    for h in (5, 25):
        c = horizon_within(model, O[held], A[held], D[held], h=h)[0]
        e = horizon_within(model, O[held], R_est[held], D[held], h=h)[0]
        print(f"  {h*20:>3} ms  within 0.2 rad: commanded {c*100:5.1f}%  identified servo {e*100:5.1f}%")
    ok = best["delay_ticks"] == round(TRUE["delay_ms"] / 20) and best["slew_rad_s"] == TRUE["slew_rad_s"]
    print("\nROUND-TRIP", "PASS" if ok else "FAIL", "- delay and slew recovered through 60/30 Hz jittered sampling" if ok else "")
