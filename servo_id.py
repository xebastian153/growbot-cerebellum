"""Identify the servo's dynamics from IMU + commands only, through the frozen forward model.

GrowBot's servos have no position feedback, so Hwangbo-style actuator nets (which
train on measured horn angle) are not directly available. But the forward model
already knows what the body does for a given horn angle. So: propose a servo model
(delay, slew limit, deadband), replay the commanded angles through it to get an
estimated horn angle, feed that to the frozen forward model, and score the one-step
prediction error on a real log. The best hypothesis is the identified servo.

Here the "real" log is the twin with a hidden realistic servo. The recovered
parameters and the held-out gain say whether the idea works before a real log exists.
"""
from __future__ import annotations
import argparse, itertools, json, sys, time
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "sim")
from growbot_sim import ServoModel, collect
from forward import MLP, make_windows, encode_obs
from sim2real_proxy import horizon_within, K


def default_grid():
    """The hypothesis grid every caller shares.

    One definition, because there were two and they drifted: the published grid
    (delay 0-3, slew >= 3 rad/s) pins BOTH parameters at its own boundary on the
    real robot's log, where the argmin sits at delay 5 and slew 2. A boundary
    argmin is the search running out, not an identification, so the range now
    reaches delay 120 ms and slew 1 rad/s -- far enough that the real optimum is
    interior and the report can say so.
    """
    return list(itertools.product([0, 1, 2, 3, 4, 5, 6],
                                  [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, None],
                                  [0.0, np.deg2rad(1), np.deg2rad(2), np.deg2rad(4)]))


def argmin_interior(best, grid):
    """(is_interior, description). A boundary argmin is a reported condition, not a footnote.

    'slew None' means NO slew limit: that is the open end of the grid, not a point
    inside it. Counting it as interior was a test that could not fail on the one
    answer it most needed to catch -- "the search ran out and picked no limit at all".

    All THREE searched axes count, deadband included. The grid is a product of delay,
    slew and deadband, so an argmin pinned at deadband 0 (or at the widest deadband on
    the grid) is the search running out on that axis exactly as it is on the other two.
    Reading only delay and slew published one argmin as INTERIOR while its deadband sat
    at min(deadbands) -- the same failure this function exists to report, on the one
    axis it was not looking at.
    """
    delays = sorted({d for d, _, _ in grid})
    slews = sorted({s for _, s, _ in grid if s is not None})
    deadbands = sorted({round(float(db), 5) for _, _, db in grid})
    db = round(float(best["deadband"]), 5)
    interior = (min(delays) < best["delay_ticks"] < max(delays)
                and best["slew_rad_s"] is not None
                and min(slews) < best["slew_rad_s"] < max(slews)
                and min(deadbands) < db < max(deadbands))
    return bool(interior), (
        f"argmin is {'INTERIOR to' if interior else 'AT THE BOUNDARY of'} the grid "
        f"(delay {min(delays)}-{max(delays)} ticks, slew {min(slews)}-{max(slews)} rad/s "
        f"or none, deadband {np.rad2deg(min(deadbands)):.0f}-{np.rad2deg(max(deadbands)):.0f} deg; "
        f"'none' = no slew limit counts as the boundary, not as an interior point)")


# Which action column is which horn. NOT (left, right): the parser stacks the two
# commands right-first --
#   imulog.py:  cmd_v = np.stack([a_right, a_left], 1)      (growbot-imulog-1 adapter)
# and the twin agrees, in its policy head and in its XML --
#   growbot_sim.py:  a = np.tanh(x[:2])  # [aRight, aLeft]
#                    # XML: joint_1 is right_leg, joint_2 is left_leg
# So column 0 is the RIGHT horn and column 1 is the LEFT one. Reading that pair the
# other way round costs nothing numerically and publishes every per-side parameter
# under its partner's name, which is a defect no error metric can show: the fit is
# identical, only the attribution is inverted. Every label in this module and in its
# callers is derived from these two constants, never from the column order.
RIGHT_COL, LEFT_COL = 0, 1


def sim_side_columns(body="walk"):
    """{'right': col, 'left': col}, read out of the twin's own model. GROUND TRUTH.

    Derived WITHOUT reference to RIGHT_COL / LEFT_COL, which is the whole point: the two
    constants above are an assertion about the physical robot, and an assertion that is
    only ever checked against itself is not checked at all. MuJoCo orders `data.ctrl` by
    actuator and `GrowBotSim.step` writes the servo output straight into `d.ctrl`, so
    action column i IS actuator i. Follow actuator -> transmitted joint -> that joint's
    body, and the BODY NAME says which leg it drives:

        <actuator><position name="servo_1" joint="joint_1"/>   (index 0)
        <body name="right_leg"><joint name="joint_1" .../></body>

    Flip RIGHT_COL / LEFT_COL and this function's answer does not move, because the XML
    did not move. That is what makes it usable as the reference the constants are held
    against, and as the anchor an asymmetric fixture injects on.
    """
    import mujoco
    from growbot_sim import BODIES
    m = mujoco.MjModel.from_xml_path(str(BODIES[body]))
    cols = {}
    for i in range(m.nu):
        jid = int(m.actuator_trnid[i, 0])
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.jnt_bodyid[jid])) or ""
        for side in ("right", "left"):
            if name.startswith(side):
                cols[side] = i
    if set(cols) != {"right", "left"} or cols["right"] == cols["left"]:
        raise RuntimeError(f"cannot read the left/right actuator convention out of "
                           f"{BODIES[body]}: got {cols}")
    return cols


def check_side_convention(body="walk"):
    """(ok, description): RIGHT_COL / LEFT_COL against sim_side_columns's ground truth."""
    cols = sim_side_columns(body)
    ok = RIGHT_COL == cols["right"] and LEFT_COL == cols["left"]
    return bool(ok), (
        f"action columns from {body} XML (actuator -> joint -> body): "
        f"right={cols['right']}, left={cols['left']}; servo_id constants: "
        f"RIGHT_COL={RIGHT_COL}, LEFT_COL={LEFT_COL} -- "
        f"{'agree' if ok else 'DISAGREE: every published per-side attribution is inverted'}")


def realized_from_commands(A, D, kw):
    """Replay a candidate servo over commanded angles; reset at episode ends."""
    sv = ServoModel(**kw); out = np.zeros_like(A); sv.reset()
    for i in range(len(A)):
        out[i] = sv(A[i], 1 / 50)
        if D[i]:
            sv.reset()
    return out


def realized_per_side(A, D, kw_l, kw_r):
    """Replay two independent servos, one per side, over the commanded angles.

    ServoModel is elementwise across the two components (its queue delays the whole
    target vector, the slew clip and the deadband apply per component), so running
    one model per side over the full command vector and keeping that side's column
    is exactly per-side parameters -- no change to the model itself.

    kw_l is the LEFT horn's triple and kw_r the RIGHT one; they land on LEFT_COL and
    RIGHT_COL respectively, which is column 1 and column 0. See those constants.
    """
    out = np.zeros_like(A)
    for col, kw in ((LEFT_COL, kw_l), (RIGHT_COL, kw_r)):
        out[:, col] = realized_from_commands(A, D, kw)[:, col]
    return out


def slower_side(kw_l, kw_r):
    """Which horn a per-side fit calls slower: lower slew limit first, then longer delay.

    'slew None' is NO slew limit, i.e. the fastest hypothesis on the grid, so it sorts
    at the fast end rather than raising on a comparison with a number. Returns
    'left', 'right' or 'neither' -- the last one when the two triples tie, where the
    fit has found no asymmetry to attribute.
    """
    def rank(kw):                                   # bigger = slower
        s = kw["slew_rad_s"]
        return (-(np.inf if s is None else float(s)), kw["delay_ticks"])
    rl, rr = rank(kw_l), rank(kw_r)
    if rl == rr:
        return "neither"
    return "left" if rl > rr else "right"


class PerSideServo:
    """Two independent ServoModels behind one GrowBotSim-compatible servo.

    Two ways to say which model goes where:

      PerSideServo(kw_l=..., kw_r=...)   by HORN NAME, through LEFT_COL / RIGHT_COL.
      PerSideServo(by_column={col: kw})  by ACTION COLUMN, touching neither constant.

    The second exists for the regression guard. It injects a known asymmetry into the
    twin and asks the identification which horn it comes back on; if the injection is
    placed through the same two constants that label the answer, the test is
    self-consistent and stays green under a swap of the pair. Anchored on the column
    that `sim_side_columns` reads out of the XML instead, the injected side is a
    physical fact about the model, and a reversed constant makes the identification hand
    the slow triple back under the wrong name -- which is the failure being guarded.

    A symmetric fixture cannot catch a label swap either way, because under a swap it
    produces exactly the same answer.
    """

    def __init__(self, kw_l=None, kw_r=None, by_column=None):
        if by_column is not None:
            if kw_l is not None or kw_r is not None:
                raise ValueError("pass the two horn triples OR by_column, not both")
            self.sv = {int(c): ServoModel(**kw) for c, kw in by_column.items()}
            if set(self.sv) != {0, 1}:
                raise ValueError(f"by_column must cover both action columns, got {sorted(self.sv)}")
        else:
            self.sv = {LEFT_COL: ServoModel(**kw_l), RIGHT_COL: ServoModel(**kw_r)}
        self.reset()

    def reset(self, pos=None):
        for s in self.sv.values():
            s.reset(pos)
        self.pos = np.zeros(2, np.float32)

    def __call__(self, target, dt):
        out = np.zeros(2, np.float32)
        for col, s in self.sv.items():
            out[col] = s(target, dt)[col]
        self.pos = out
        return out


def _extend_cuts(D, max_delay):
    """Cut mask widened by the largest candidate delay: the servo transient guard."""
    D_ext = D.copy()
    for j in range(1, max_delay + 1):
        D_ext[j:] |= D[:-j]
    return D_ext


def _clip_scorer(model, O, D_ext, clips, seed=0):
    """Multi-horizon scorer: open-loop rollouts inside clips, re-anchored at clip start.

    A one-step score asks only "given the truth now, is the next tick right"; a servo
    delay shows up as an error that accumulates, so a one-step cost sees the weakest
    version of the signature it is trying to read. Clip horizons are sampled uniformly
    rather than fixed: short clips carry no long-horizon information, long ones drown in
    open-loop divergence, and sampling gets both without choosing between them.

    Normalisation is per component, as in the one-step path and for the same reason (the
    gyro's irreducible contact variance must not dilute the angle components), but the
    scale is the std of the target STATES rather than of one-tick deltas, because the
    quantity being compared here is a state after h steps, not a delta.
    """
    lo, hi, n_starts = clips["min_ticks"], clips["max_ticks"], clips["n_starts"]
    F = encode_obs(O)
    N, fdim = len(O), F.shape[1]
    ok = np.ones(N, bool)
    for j in range(K):
        ok &= np.roll(~D_ext, j + 1)
    for j in range(hi):
        ok &= np.roll(~D_ext, -j)
    ok[:K] = False
    ok[N - hi - 1:] = False
    cand = np.flatnonzero(ok)
    if len(cand) == 0:
        raise ValueError(f"no clip start survives K={K} history and {hi} ticks of horizon "
                         f"in {N} ticks: shorten max_ticks")
    rng = np.random.default_rng(seed)
    starts = rng.choice(cand, size=min(n_starts, len(cand)), replace=False)
    horiz = rng.integers(lo, hi + 1, size=len(starts))
    fstd = F.std(0) + 1e-8

    def score(R):
        win = np.zeros((len(starts), K, fdim + 2), np.float32)
        for k in range(K):
            win[:, k, :fdim] = F[starts - k]
            win[:, k, fdim:] = R[starts - k]
        cur = F[starts].copy()
        tot, cnt = 0.0, 0
        for h in range(1, hi + 1):
            win[:, 0, fdim:] = R[starts + h - 1]
            cur = cur + model.predict(win.reshape(len(starts), -1))
            for a in range(3):                      # keep (sin, cos) on the unit circle
                n = np.sqrt(cur[:, a] ** 2 + cur[:, a + 3] ** 2) + 1e-9
                cur[:, a] /= n
                cur[:, a + 3] /= n
            win = np.roll(win, 1, axis=1)
            win[:, 0, :fdim] = cur
            live = horiz >= h
            if not live.any():
                break
            e = (cur[live] - F[starts[live] + h]) / fstd
            tot += float((e ** 2).sum())
            cnt += int(live.sum()) * fdim
        return tot / max(cnt, 1)

    return score, {"starts": int(len(starts)), "min_ticks": lo, "max_ticks": hi,
                   "mean_horizon_ticks": float(horiz.mean())}


def identify(model, O, A, O2, D, grid, clips=None, seed=0):
    """Return (sorted [(err, kw)], best kw). One-step forward error by default.

    Two guards for real logs: (1) windows within max-grid-delay ticks after an
    episode cut are excluded, because the replayed servo's state there is its
    reset value, not something identified -- with many Bluetooth micro-gaps that
    transient would otherwise enter the score as valid data; (2) the residual is
    normalised per output component, so the gyro's large irreducible contact
    variance (a floor common to every hypothesis) does not dilute the angle
    components where the servo signature actually lives.

    clips: pass dict(min_ticks=, max_ticks=, n_starts=) to score multi-horizon
    rollouts instead of one-step error (see _clip_scorer). Both guards are kept.
    """
    D_ext = _extend_cuts(D, max(d for d, _, _ in grid))
    if clips is None:
        _, Y0, *_ = make_windows(O, A, O2, D_ext, K)   # Y is hypothesis-independent
        ystd = Y0.std(0) + 1e-8

        def score(R):
            X, Y, *_ = make_windows(O, R, O2, D_ext, K)
            return float((((model.predict(X) - Y) / ystd) ** 2).mean())
    else:
        score, _ = _clip_scorer(model, O, D_ext, clips, seed=seed)
    scores = []
    for d, s, db in grid:
        kw = dict(delay_ticks=d, slew_rad_s=s, deadband=db)
        Rc = realized_from_commands(A, D, kw)       # reset on the true cuts
        scores.append((score(Rc), kw))
    scores.sort(key=lambda x: x[0])
    return scores, scores[0][1]


def identify_per_side(model, O, A, O2, D, grid, shared, clips=None, seed=0, rounds=3):
    """Per-side (delay, slew, deadband) by coordinate descent from the shared solution.

    Two independent triples over this grid is ~63k hypotheses; brute force is not the
    point. Coordinate descent starts at the shared answer -- which is by construction a
    feasible point and usually a good one -- and alternately re-optimises one side with
    the other held, until a full round buys nothing.

    Returns (kw_l, kw_r, info), left horn first. The descent is indexed by ACTION
    COLUMN throughout and the two horn names are attached once, at the return and in
    info["side_scores"], from RIGHT_COL / LEFT_COL -- so no caller has to know the
    column order, and no caller can re-invert it.

    info carries the evaluation count and the per-side sweeps, each measured with the
    OTHER side held at its solution: the honest reading of a per-side search, where a
    side's separability depends on its partner. Those sweeps are one-dimensional
    CONDITIONAL slices, not joint sets -- see the caveat in identification_ablation.
    """
    D_ext = _extend_cuts(D, max(d for d, _, _ in grid))
    if clips is None:
        _, Y0, *_ = make_windows(O, A, O2, D_ext, K)
        ystd = Y0.std(0) + 1e-8

        def score(R):
            X, Y, *_ = make_windows(O, R, O2, D_ext, K)
            return float((((model.predict(X) - Y) / ystd) ** 2).mean())
    else:
        score, _ = _clip_scorer(model, O, D_ext, clips, seed=seed)

    def as_kw(t):
        return dict(delay_ticks=t[0], slew_rad_s=t[1], deadband=t[2])

    start = (shared["delay_ticks"], shared["slew_rad_s"], shared["deadband"])
    cur = {RIGHT_COL: start, LEFT_COL: start}

    def realized(state):
        return realized_per_side(A, D, as_kw(state[LEFT_COL]), as_kw(state[RIGHT_COL]))

    best_e = score(realized(cur))
    evals, col_scores = 1, {RIGHT_COL: None, LEFT_COL: None}
    for _ in range(rounds):
        improved = False
        for col in (RIGHT_COL, LEFT_COL):       # column order: 0 then 1
            trials = []
            for t in grid:
                trials.append((score(realized({**cur, col: t})), t))
                evals += 1
            trials.sort(key=lambda x: x[0])
            col_scores[col] = trials
            if trials[0][0] < best_e - 1e-12:
                best_e, cur[col], improved = trials[0][0], trials[0][1], True
        if not improved:
            break
    # expose the per-side sweeps in the (err, kw) shape determined_sets() consumes,
    # keyed by HORN NAME so a caller cannot re-derive the mapping and get it wrong
    info = {"evaluations": evals, "best_err": best_e,
            "side_scores": {name: [(e, as_kw(t)) for e, t in (col_scores[c] or [])]
                            for name, c in (("left", LEFT_COL), ("right", RIGHT_COL))}}
    return as_kw(cur[LEFT_COL]), as_kw(cur[RIGHT_COL]), info


def _key(kw):
    return kw["delay_ticks"], kw["slew_rad_s"], round(float(kw["deadband"]), 5)


def confidence_band(scoresA, scoresB):
    """Estimator noise at the full fit size, from two independent halves of it.

    Each half scores every hypothesis independently, so the spread of (errA - errB)
    measures the noise at half the data; the full fit uses twice as much, so its noise
    is about that spread / 2. This is the number a separation must beat before an argmin
    means anything.

    The spread is a ROBUST scale (1.4826 * MAD), not a standard deviation, because a
    standard deviation made the band depend on which hypotheses were enumerated. Adding
    slow-slew candidates to the grid -- candidates added precisely in order to rule them
    out -- fits them badly, and their large, noisy errors inflated std and widened every
    determined set. Measured on the fixture, same data and same argmin: widening the grid
    from 96 to 252 hypotheses moved the std band 0.00141 -> 0.00457 and the delay set
    [1, 2] -> [0, 1, 2, 3], which admits delay 0, i.e. "no servo at all". A determined set
    that grows because someone enumerated a worse hypothesis is not measuring the log.
    1.4826 * MAD is the standard consistent estimator of sigma for gaussian samples, so on
    a grid without that tail it agrees with the old number (0.00140 vs 0.00141 measured);
    it only differs where the tail exists, which is where std was wrong.
    """
    eA = {_key(kw): e for e, kw in scoresA}
    eB = {_key(kw): e for e, kw in scoresB}
    d = np.array([eA[k] - eB[k] for k in eA])
    return float(1.4826 * np.median(np.abs(d - np.median(d)))) / 2.0


def determined_sets(scores, best, grid, band):
    """(delay_set, slew_set): the grid values this log actually separates.

    A value stays in the set when holding the other parameters at `best` costs no
    more than `band` over the best error -- i.e. the log cannot tell it apart from
    the winner. A one-element set is an identification; a longer one is the honest
    answer, and reporting its argmin as though it were the answer is the failure
    mode this exists to prevent.
    """
    fit_err = {_key(kw): e for e, kw in scores}
    best_e = scores[0][0]
    db = round(float(best["deadband"]), 5)

    def determined(values, fixed):
        # ordered with the same key the candidate list uses: 'None' (no slew limit) is a
        # legal member, and plain sorted() raises the moment it lands in a set beside a
        # number. That is exactly the under-determined case this function exists to
        # report -- the log could not rule out "no slew limit at all" -- so the crash was
        # waiting for the one answer it most needed to deliver.
        keep = {v for v in values if fit_err.get(fixed(v), np.inf) - best_e <= band}
        return sorted(keep, key=lambda v: (v is None, v))

    delays = sorted({d for d, _, _ in grid})
    slews = sorted({s for _, s, _ in grid}, key=lambda v: (v is None, v))
    return (determined(delays, lambda v: (v, best["slew_rad_s"], db)),
            determined(slews, lambda v: (best["delay_ticks"], v, db)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--log-steps", type=int, default=30000, help="600 s: first half fits, second half evaluates")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--true-delay", type=int, default=2)
    ap.add_argument("--true-slew", type=float, default=5.0)
    ap.add_argument("--true-deadband-deg", type=float, default=2.0)
    args = ap.parse_args()

    tr = np.load("data/train.npz")
    Xtr, Ytr, *_ = make_windows(tr["obs"], tr["act"], tr["next_obs"], tr["done"], K)
    nominal = MLP(hidden=128, epochs=args.epochs).fit(Xtr, Ytr)

    TRUE = dict(delay_ticks=args.true_delay, slew_rad_s=args.true_slew, deadband=np.deg2rad(args.true_deadband_deg))
    O, A, O2, D, _, R_true = collect(args.log_steps, seed=args.seed, body="walk",
                                     servo=ServoModel(**TRUE), return_realized=True)
    half = args.log_steps // 2
    fit, held = slice(0, half), slice(half, None)

    grid = default_grid()
    t0 = time.time()
    scores, best = identify(nominal, O[fit], A[fit], O2[fit], D[fit], grid)
    ideal_err = [e for e, kw in scores if kw["delay_ticks"] == 0 and kw["slew_rad_s"] is None and kw["deadband"] == 0.0][0]
    print(f"{len(grid)} hypotheses on {half / 50:.0f} s of IMU+commands: {time.time() - t0:.0f}s")
    print("grid resolution: delay in 20 ms steps, slew from an enumerated set, deadband 0-4 deg;")
    print("identified values are grid points, not continuous estimates")
    print("best 5:")
    for e, kw in scores[:5]:
        print(f"  err {e:.4f}  delay {kw['delay_ticks']}  slew {kw['slew_rad_s']}  deadband {np.rad2deg(kw['deadband']):.0f} deg")
    print(f"ideal-servo hypothesis err {ideal_err:.4f}   true: delay {TRUE['delay_ticks']}, slew {TRUE['slew_rad_s']}, "
          f"deadband {args.true_deadband_deg:.0f} deg")
    interior, interior_why = argmin_interior(best, grid)
    print(f"  {interior_why}")

    # diagnostics for the day the real servo leaves the model family ------------
    halfA, halfB = slice(0, half // 2), slice(half // 2, half)
    scoresA, bestA = identify(nominal, O[halfA], A[halfA], O2[halfA], D[halfA], grid)
    scoresB, bestB = identify(nominal, O[halfB], A[halfB], O2[halfB], D[halfB], grid)
    agree = bestA["delay_ticks"] == bestB["delay_ticks"] and bestA["slew_rad_s"] == bestB["slew_rad_s"]
    print(f"split-half stability: A=(delay {bestA['delay_ticks']}, slew {bestA['slew_rad_s']})  "
          f"B=(delay {bestB['delay_ticks']}, slew {bestB['slew_rad_s']})  "
          f"{'AGREE' if agree else 'DISAGREE -- log too short or servo outside the model family'}")
    band = confidence_band(scoresA, scoresB)
    delay_set, slew_set = determined_sets(scores, best, grid, band)
    def show(name, sset, unit=""):
        if len(sset) == 1:
            print(f"{name} determined: {sset[0]}{unit}")
        else:
            print(f"{name} ∈ {{{', '.join(str(v) for v in sset)}}}{unit} -- separation below the "
                  f"confidence band (±{band:.4f}); report the set, not the argmin")
    show("delay", delay_set, " ticks")
    show("slew", slew_set, " rad/s")
    held_scores, _ = identify(nominal, O[held], A[held], O2[held], D[held],
                              [(best["delay_ticks"], best["slew_rad_s"], best["deadband"]),
                               (0, None, 0.0)])
    by_kw = {(kw["delay_ticks"], kw["slew_rad_s"]): e for e, kw in held_scores}
    print(f"held-out one-step err: best {by_kw[(best['delay_ticks'], best['slew_rad_s'])]:.4f}  "
          f"ideal {by_kw[(0, None)]:.4f}  (fit: best {scores[0][0]:.4f}  ideal {ideal_err:.4f}; "
          f"divergence between fit and held-out = fitting noise)")
    slew_family = sorted((e, kw["slew_rad_s"]) for e, kw in scores
                         if kw["delay_ticks"] == best["delay_ticks"] and kw["deadband"] == best["deadband"])
    print("slew separability at the identified delay: "
          + "  ".join(f"{sl}:{e:.4f}" for e, sl in slew_family)
          + "   (near-ties here mean the excitation never hit the slew limit)")

    R_est = realized_from_commands(A, D, best)
    out = {"true": {**TRUE, "deadband": float(TRUE["deadband"])}, "identified": {**best, "deadband": float(best["deadband"])},
           "ideal_err": ideal_err, "best_err": scores[0][0],
           "split_half_agree": agree, "confidence_band": band, "argmin_interior": interior,
           "delay_determined_set": delay_set, "slew_determined_set": [v for v in slew_set],
           "held_out_err": {"best": by_kw[(best["delay_ticks"], best["slew_rad_s"])], "ideal": by_kw[(0, None)]},
           "held_out": {}}
    print("\nheld-out half, within 0.2 rad:")
    for h in (5, 25):
        c = horizon_within(nominal, O[held], A[held], D[held], h=h)[0]
        e = horizon_within(nominal, O[held], R_est[held], D[held], h=h)[0]
        t = horizon_within(nominal, O[held], R_true[held], D[held], h=h)[0]
        out["held_out"][f"{h * 20}ms"] = {"commanded": c, "identified": e, "true_horn": t}
        print(f"  {h * 20:>3} ms   commanded {c * 100:5.1f}%   identified servo {e * 100:5.1f}%   true horn angle {t * 100:5.1f}%")
    out["mean_abs_horn_err_rad"] = float(np.abs(R_est[held] - R_true[held]).mean())
    json.dump(out, open("results/servo_id.json", "w"), indent=1)


if __name__ == "__main__":
    main()
