"""Can a centre-of-mass shift be identified from IMU + commands alone -- and is it
separable from the servo, or do the two trade off?

`body_params.py` found that a 3 cm centre-of-mass shift is what a builder's phone placement
changes, and that it costs the frozen model 33.8 pts of 500 ms pitch. The day-of-log chain
identifies the SERVO from IMU + commands through the frozen model (`servo_id.py`); this
script asks whether it can do the same for the centre of mass, and -- the question that
matters more -- whether a CoM shift and a servo delay look alike from the IMU. If they do,
the servo identified on the real log may carry centre-of-mass inside it.

How a CoM hypothesis is scored. A servo hypothesis transforms the COMMANDS, so servo_id
scores every candidate through one frozen model. A CoM hypothesis transforms the BODY: there
is no command-side knob, so a hypothesis IS a body, and a body is represented by a forward
model trained on it. Every candidate is therefore a model trained the same way (same
collection length, same collection seed, same epochs, same torch seed) on the twin with that
CoM, and scored by its normalised one-step error on the hidden log -- exactly the error
`servo_id.identify` uses, with the model varying instead of the realised command.

Two consequences are published rather than hidden. (1) Candidate models carry training
noise that split-half cannot see (the same models score both halves), so a SECOND nominal
candidate trained on a different collection seed is scored on every hidden log, and the gap
between the two nominal candidates is printed beside every band as the noise floor of the
method. (2) The grid extends one step beyond the published DR endpoints on both axes, so a
truth AT an endpoint is interior to the grid and an argmin on the grid edge still means the
search ran out.

The confound test puts both knobs in one grid: candidate models along dcom_x (dcom_z = 0)
times servo (delay, slew, deadband) hypotheses applied to the commands, and scores the joint
grid on three hidden bodies -- shifted CoM with an ideal servo, nominal CoM with the real-log
servo candidate (delay 5, slew 2.0, deadband 2 deg), and both at once. If the within-band set
of the joint grid runs along a CoM/delay diagonal instead of sitting on a point, the log
cannot separate them.

Decision rule, stated before the numbers (the same machinery as servo_id / body_params):
  band        1.4826 * MAD / 2 of (err_A - err_B) over every hypothesis, A/B = the two
              quarters of the fit half (`servo_id.confidence_band`'s estimator).
  determined  the grid values within `band` of the best error with the other parameters
              held at the argmin (`servo_id.determined_sets`' rule); a one-element set
              containing the truth is an identification, a longer set is the honest answer.
  per seed    argmin correct / truth in set / set == {truth}; three hidden seeds per body.
  resolved    the per-seed verdict is the same on all three seeds. Seeds are NOT paired
              (collect() draws sim.rng every tick for the push test, mode-dependently inside
              Excitation and once per episode in fresh()), so three seeds answer "does the
              verdict hold on every seed", not "is the mean significant".
  null case   the nominal hidden body MUST identify to (0, 0) on every seed. Hard assertion.
  noise floor |err(nominal candidate) - err(second nominal candidate)| on the same hidden
              log; a CoM separation smaller than this is not evidence.
  mechanism   every sentence in the write-up that explains a result carries the same
              resolved / unresolved mark as the numbers it rests on.

Limits, verbatim from body_params: this is forward-model *prediction* accuracy, not policy
*transfer*; and every real-log number in this repo comes from **one** unit and **one** phone,
so this run says what the twin can identify, not what a second real robot does. Body is the
walk twin (the real-log body, the one `real2sim` and `servo_id` use), with the published DR
centre-of-mass values applied to it -- `body_params`' rows are on the olie body; the
confound question concerns the walk-body real log, so the walk body is the one that matters
here.
"""
from __future__ import annotations
import argparse, itertools, json, sys, time
import numpy as np
from growbot_cerebellum.sim import collect, ServoModel, DR
from growbot_cerebellum.forward import MLP, make_windows, K
from growbot_cerebellum.servo_id import realized_from_commands, _extend_cuts
from growbot_cerebellum.honesty import seed_stat

BODY = "walk"
CAND_SEED, NOISE_SEED = 1000, 1001          # candidate collections; never a hidden-log seed
HIDDEN_SEEDS = [0, 1, 2]
X_STEP, Z_LO, Z_HI = 0.015, DR["dcom_z"][0], DR["dcom_z"][1]
X_GRID = [-0.045, -0.030, -0.015, 0.0, 0.015, 0.030, 0.045]          # DR ends +-0.030, one step beyond
Z_GRID = [-0.020, Z_LO, 0.0, Z_HI, 0.030]                              # DR ends -0.010 / +0.015, one step beyond
DELAYS = [0, 1, 2, 3, 4, 5, 6]
SLEWS = [2.0, None]
DBS = [0.0, float(np.deg2rad(2))]
SERVO_REAL = dict(delay_ticks=5, slew_rad_s=2.0, deadband=float(np.deg2rad(2)))   # real-log argmin
HIDDEN_COM = [("nominal (null case)", (0.0, 0.0, 0.0)),
              ("com x -0.03", (DR["dcom_x"][0], 0.0, 0.0)),
              ("com x +0.03", (DR["dcom_x"][1], 0.0, 0.0)),
              ("com z +0.015", (0.0, 0.0, Z_HI))]
HIDDEN_CONFOUND = [("com x +0.03, ideal servo", (DR["dcom_x"][1], 0.0, 0.0), None),
                   ("nominal CoM, real-log servo", (0.0, 0.0, 0.0), SERVO_REAL),
                   ("com x +0.03, real-log servo", (DR["dcom_x"][1], 0.0, 0.0), SERVO_REAL)]


def dr(dcom):
    return dict(mass_scale=1.0, dcom=tuple(float(v) for v in dcom))


def train_candidate(dcom, steps, epochs, seed):
    O, A, O2, D, _ = collect(steps, seed=seed, body=BODY, dr=dr(dcom))
    X, Y, *_ = make_windows(O, A, O2, D, K)
    return MLP(hidden=128, epochs=epochs, seed=0).fit(X, Y)


def one_step_scorer(O, A, O2, D, max_delay):
    """Normalised one-step error, the servo_id.identify scorer with the model as the argument.
    Windows within max_delay ticks after a cut are excluded (the replayed servo's transient),
    and the residual is normalised per component by a hypothesis-independent ystd."""
    D_ext = _extend_cuts(D, max_delay) if max_delay > 0 else D
    _, Y0, *_ = make_windows(O, A, O2, D_ext, K)
    ystd = Y0.std(0) + 1e-8

    def score(model, R):
        X, Y, *_ = make_windows(O, R, O2, D_ext, K)
        return float((((model.predict(X) - Y) / ystd) ** 2).mean())
    return score


def band_of(errA, errB):
    d = np.array([errA[k] - errB[k] for k in errA])
    return float(1.4826 * np.median(np.abs(d - np.median(d)))) / 2.0


def determined(err, best_key, best_e, band, axis, values, key_at):
    """Values of one parameter within `band` of the best error, the others held at the argmin."""
    return [v for v in values if err.get(key_at(best_key, axis, v), np.inf) - best_e <= band]


# ----------------------------------------------------------------------
# identification A: CoM alone (ideal servo)
# ----------------------------------------------------------------------
def identify_com(models, O, A, O2, D):
    """scores {(x, z): err} on (O, A, O2, D) with R = A (ideal servo)."""
    score = one_step_scorer(O, A, O2, D, 0)
    return {xz: score(m, A) for xz, m in models.items()}


def com_verdict(err, band, truth):
    best_key = min(err, key=err.get); best_e = err[best_key]
    at = lambda bk, axis, v: ((v, bk[1]) if axis == "x" else (bk[0], v))
    xs = determined(err, best_key, best_e, band, "x", X_GRID, at)
    zs = determined(err, best_key, best_e, band, "z", Z_GRID, at)
    interior = (min(X_GRID) < best_key[0] < max(X_GRID)) and (min(Z_GRID) < best_key[1] < max(Z_GRID))
    tx, tz = truth[0], truth[2]
    return {"argmin": {"dcom_x": best_key[0], "dcom_z": best_key[1]}, "best_err": best_e,
            "x_determined": xs, "z_determined": zs, "argmin_interior": bool(interior),
            "argmin_correct": bool(abs(best_key[0] - tx) < 1e-9 and abs(best_key[1] - tz) < 1e-9),
            "truth_in_sets": bool(any(abs(v - tx) < 1e-9 for v in xs) and any(abs(v - tz) < 1e-9 for v in zs)),
            "identified": bool(len(xs) == 1 and len(zs) == 1 and abs(xs[0] - tx) < 1e-9 and abs(zs[0] - tz) < 1e-9),
            "nearest_wrong_x_gap": float(min(err[(v, best_key[1])] for v in X_GRID if abs(v - best_key[0]) > 1e-9) - best_e),
            "nearest_wrong_z_gap": float(min(err[(best_key[0], v)] for v in Z_GRID if abs(v - best_key[1]) > 1e-9) - best_e)}


# ----------------------------------------------------------------------
# identification B: joint CoM x servo grid (the confound)
# ----------------------------------------------------------------------
def joint_grid():
    return list(itertools.product(X_GRID, DELAYS, SLEWS, DBS))


def identify_joint(models_x, O, A, O2, D):
    score = one_step_scorer(O, A, O2, D, max(DELAYS))
    err = {}
    real = {}
    for d, s, db in itertools.product(DELAYS, SLEWS, DBS):
        real[(d, s, db)] = realized_from_commands(A, D, dict(delay_ticks=d, slew_rad_s=s, deadband=db))
    for x in X_GRID:
        m = models_x[x]
        for d, s, db in itertools.product(DELAYS, SLEWS, DBS):
            err[(x, d, s, db)] = score(m, real[(d, s, db)])
    return err


def joint_verdict(err, band, truth_x, truth_servo):
    best_key = min(err, key=err.get); best_e = err[best_key]
    x0, d0, s0, db0 = best_key
    xs = [v for v in X_GRID if err[(v, d0, s0, db0)] - best_e <= band]
    ds = [v for v in DELAYS if err[(x0, v, s0, db0)] - best_e <= band]
    ss = [v for v in SLEWS if err[(x0, d0, v, db0)] - best_e <= band]
    within = sorted([k for k, e in err.items() if e - best_e <= band], key=lambda k: err[k])
    # the ridge: for each dcom_x, the best servo hypothesis and its error. A delay that
    # drifts with dcom_x is the trade-off, a delay that stays put is separability.
    ridge = []
    for x in X_GRID:
        sub = {k: e for k, e in err.items() if abs(k[0] - x) < 1e-9}
        kb = min(sub, key=sub.get)
        ridge.append({"dcom_x": x, "best_delay": kb[1], "best_slew": kb[2], "best_deadband_deg": float(np.rad2deg(kb[3])),
                      "err": sub[kb], "err_minus_best": sub[kb] - best_e, "within_band": bool(sub[kb] - best_e <= band)})
    ts = truth_servo or dict(delay_ticks=0, slew_rad_s=None, deadband=0.0)
    pairs = sorted({(k[0], k[1]) for k in within})
    xs_in, ds_in = sorted({p[0] for p in pairs}), sorted({p[1] for p in pairs})
    # diagonal: the within-band set spans more than one dcom_x AND more than one delay,
    # and the delay that wins moves with dcom_x
    ridge_delays = [r["best_delay"] for r in ridge if r["within_band"]]
    return {"argmin": {"dcom_x": x0, "delay_ticks": d0, "slew_rad_s": s0, "deadband_deg": float(np.rad2deg(db0))},
            "best_err": best_e,
            "x_determined": xs, "delay_determined": ds, "slew_determined": ss,
            "within_band_count": len(within), "within_band_x_values": xs_in, "within_band_delay_values": ds_in,
            "within_band_pairs_x_delay": [list(p) for p in pairs],
            "ridge": ridge,
            "argmin_x_correct": bool(abs(x0 - truth_x) < 1e-9),
            "argmin_delay_correct": bool(d0 == ts["delay_ticks"]),
            "truth_x_in_set": bool(any(abs(v - truth_x) < 1e-9 for v in xs)),
            "truth_delay_in_set": bool(ts["delay_ticks"] in ds),
            "diagonal": bool(len(xs_in) > 1 and len(ds_in) > 1 and len(set(ridge_delays)) > 1)}


# ----------------------------------------------------------------------
def summarize(flags):
    """resolved = the same boolean on every seed; value = that boolean if resolved."""
    return {"per_seed": flags, "resolved": bool(all(flags) or not any(flags)),
            "value": (bool(flags[0]) if (all(flags) or not any(flags)) else None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand-steps", type=int, default=100_000, help="ticks per candidate body (2000 s)")
    ap.add_argument("--cand-epochs", type=int, default=30)
    ap.add_argument("--hidden-steps", type=int, default=30_000, help="600 s per hidden log: first half fits")
    ap.add_argument("--shipped-epochs", type=int, default=60, help="the published model on data/train.npz")
    args = ap.parse_args()
    t_start = time.time()

    print("=" * 100)
    print("DECISION RULE, stated before the numbers")
    print("=" * 100)
    print("  a CoM hypothesis is a body, so a candidate is a forward model trained on that body:")
    print(f"  {len(X_GRID) * len(Z_GRID)} candidates, dcom_x {X_GRID} m x dcom_z {Z_GRID} m, each trained on")
    print(f"  {args.cand_steps} ticks collected with seed {CAND_SEED} on the {BODY} body, MLP(128) {args.cand_epochs} epochs,")
    print("  torch seed 0. The grid extends one step beyond the published DR endpoints on both axes,")
    print("  so a truth at an endpoint is interior and a grid-edge argmin means the search ran out.")
    print("  score = servo_id.identify's normalised one-step error, on the first half of each hidden")
    print(f"  log ({args.hidden_steps // 2} ticks); band = 1.4826*MAD/2 of the A/B quarter errors; determined")
    print("  sets hold the other parameter at the argmin; a verdict is RESOLVED when all three")
    print("  hidden seeds agree. Seeds are not paired (collect() draws sim.rng every tick for the")
    print("  push test, mode-dependently in Excitation, once per episode in fresh()).")
    print(f"  noise floor: a second nominal candidate (collection seed {NOISE_SEED}) is scored on every")
    print("  hidden log; |err(nominal) - err(second nominal)| is the method's own training noise.")
    print("  null case: the nominal hidden body must identify to (0, 0) on every seed -- asserted.")
    print(f"  confound grid: dcom_x ({len(X_GRID)}) x delay {DELAYS} x slew {SLEWS} x deadband {[0, 2]} deg")
    print(f"  = {len(joint_grid())} hypotheses, scored on {len(HIDDEN_CONFOUND)} hidden bodies x 3 seeds.")
    print("  limits: prediction accuracy, not policy transfer; every real-log number in this repo")
    print("  comes from ONE unit and ONE phone.")

    # ---- candidates ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"CANDIDATE MODELS ({len(X_GRID) * len(Z_GRID)} bodies + 1 noise-floor model + the shipped model)")
    print("=" * 100)
    models = {}
    t0 = time.time()
    for x, z in itertools.product(X_GRID, Z_GRID):
        models[(x, z)] = train_candidate((x, 0.0, z), args.cand_steps, args.cand_epochs, CAND_SEED)
        print(f"  trained dcom_x {x:+.3f} dcom_z {z:+.3f}   ({time.time() - t0:.0f}s)", flush=True)
    noise_model = train_candidate((0.0, 0.0, 0.0), args.cand_steps, args.cand_epochs, NOISE_SEED)
    tr = np.load("data/train.npz")
    Xtr, Ytr, *_ = make_windows(tr["obs"], tr["act"], tr["next_obs"], tr["done"], K)
    shipped = MLP(hidden=128, epochs=args.shipped_epochs).fit(Xtr, Ytr)
    print(f"  candidates ready in {time.time() - t0:.0f}s", flush=True)
    models_x = {x: models[(x, 0.0)] for x in X_GRID}

    half = args.hidden_steps // 2
    q = half // 2
    fit, held = slice(0, half), slice(half, None)
    qa, qb = slice(0, q), slice(q, half)

    # ---- identification A --------------------------------------------------------------
    print("\n" + "=" * 100)
    print("IDENTIFICATION A -- centre of mass alone, ideal servo")
    print("=" * 100)
    print(f"  {'hidden body':<22}{'seed':>5}{'argmin (x, z)':>18}{'x set':>34}{'z set':>30}{'band':>10}{'noise':>9}{'xgap':>9}{'zgap':>9}  verdict")
    resA = {}
    for name, dcom in HIDDEN_COM:
        rows = []
        for sd in HIDDEN_SEEDS:
            O, A, O2, D, _ = collect(args.hidden_steps, seed=sd, body=BODY, dr=dr(dcom))
            err = identify_com(models, O[fit], A[fit], O2[fit], D[fit])
            eA = identify_com(models, O[qa], A[qa], O2[qa], D[qa])
            eB = identify_com(models, O[qb], A[qb], O2[qb], D[qb])
            band = band_of(eA, eB)
            v = com_verdict(err, band, dcom)
            vA, vB = com_verdict(eA, band, dcom), com_verdict(eB, band, dcom)
            split_agree = vA["argmin"] == vB["argmin"]
            sc_fit = one_step_scorer(O[fit], A[fit], O2[fit], D[fit], 0)
            noise = abs(sc_fit(models[(0.0, 0.0)], A[fit]) - sc_fit(noise_model, A[fit]))
            sc_held = one_step_scorer(O[held], A[held], O2[held], D[held], 0)
            bx, bz = v["argmin"]["dcom_x"], v["argmin"]["dcom_z"]
            held_err = {"argmin_candidate": sc_held(models[(bx, bz)], A[held]),
                        "nominal_candidate": sc_held(models[(0.0, 0.0)], A[held]),
                        "truth_candidate": sc_held(models[(dcom[0], dcom[2])], A[held]),
                        "shipped_model": sc_held(shipped, A[held])}
            rows.append({"seed": sd, **v, "band": band, "split_half_argmin_agree": bool(split_agree),
                         "split_half_argmins": [vA["argmin"], vB["argmin"]],
                         "noise_floor": float(noise), "noise_floor_vs_band": float(noise / band) if band > 0 else None,
                         "held_out_one_step_err": held_err,
                         "errors": {f"{k[0]:+.3f},{k[1]:+.3f}": e for k, e in err.items()}})
            verdict = ("IDENTIFIED" if v["identified"] else "truth in set" if v["truth_in_sets"] else "WRONG")
            print(f"  {name:<22}{sd:>5}{f'({bx:+.3f}, {bz:+.3f})':>18}{str([f'{u:+.3f}' for u in v['x_determined']]):>34}"
                  f"{str([f'{u:+.3f}' for u in v['z_determined']]):>30}{band:>10.5f}{noise:>9.5f}"
                  f"{v['nearest_wrong_x_gap']:>9.5f}{v['nearest_wrong_z_gap']:>9.5f}  {verdict}"
                  f"{'' if v['argmin_interior'] else ' [boundary]'}{'' if split_agree else ' split-half DISAGREE'}")
        resA[name] = {"truth": {"dcom_x": dcom[0], "dcom_z": dcom[2]}, "per_seed": rows,
                      "argmin_correct": summarize([r["argmin_correct"] for r in rows]),
                      "truth_in_sets": summarize([r["truth_in_sets"] for r in rows]),
                      "identified": summarize([r["identified"] for r in rows]),
                      "split_half_agree": summarize([r["split_half_argmin_agree"] for r in rows]),
                      "argmin_interior": summarize([r["argmin_interior"] for r in rows]),
                      "band": seed_stat([r["band"] for r in rows]),
                      "noise_floor": seed_stat([r["noise_floor"] for r in rows]),
                      "nearest_wrong_x_gap": seed_stat([r["nearest_wrong_x_gap"] for r in rows]),
                      "nearest_wrong_z_gap": seed_stat([r["nearest_wrong_z_gap"] for r in rows]),
                      "x_set_size": seed_stat([len(r["x_determined"]) for r in rows]),
                      "z_set_size": seed_stat([len(r["z_determined"]) for r in rows]),
                      "held_out_shipped_minus_truth": seed_stat([r["held_out_one_step_err"]["shipped_model"]
                                                                 - r["held_out_one_step_err"]["truth_candidate"] for r in rows])}
    null = resA["nominal (null case)"]
    null_ok = bool(null["argmin_correct"]["resolved"] and null["argmin_correct"]["value"])
    # a failed null case is a RESULT about the method, so the run records it and finishes;
    # the hard assertion is the non-zero exit at the end, after the artifact is written
    print("  null case: nominal hidden body identifies to (0, 0) on every seed -- "
          + ("PASS" if null_ok else f"FAIL {null['argmin_correct']['per_seed']}"))

    # ---- identification B --------------------------------------------------------------
    print("\n" + "=" * 100)
    print("IDENTIFICATION B -- the confound: joint dcom_x x servo grid")
    print("=" * 100)
    resB = {}
    for name, dcom, servo in HIDDEN_CONFOUND:
        rows = []
        for sd in HIDDEN_SEEDS:
            sv = ServoModel(**servo) if servo else None
            O, A, O2, D, _ = collect(args.hidden_steps, seed=sd, body=BODY, dr=dr(dcom), servo=sv)
            err = identify_joint(models_x, O[fit], A[fit], O2[fit], D[fit])
            eA = identify_joint(models_x, O[qa], A[qa], O2[qa], D[qa])
            eB = identify_joint(models_x, O[qb], A[qb], O2[qb], D[qb])
            band = band_of(eA, eB)
            v = joint_verdict(err, band, dcom[0], servo)
            vA, vB = joint_verdict(eA, band, dcom[0], servo), joint_verdict(eB, band, dcom[0], servo)
            rows.append({"seed": sd, **v, "band": band,
                         "split_half_argmins": [vA["argmin"], vB["argmin"]],
                         "split_half_x_agree": bool(vA["argmin"]["dcom_x"] == vB["argmin"]["dcom_x"]),
                         "split_half_delay_agree": bool(vA["argmin"]["delay_ticks"] == vB["argmin"]["delay_ticks"])})
            a = v["argmin"]
            print(f"  {name:<30} seed {sd}: argmin dcom_x {a['dcom_x']:+.3f} delay {a['delay_ticks']} slew {a['slew_rad_s']} "
                  f"db {a['deadband_deg']:.0f} deg | x set {[f'{u:+.3f}' for u in v['x_determined']]} "
                  f"delay set {v['delay_determined']} | within band: {v['within_band_count']} hyps over "
                  f"x {[f'{u:+.3f}' for u in v['within_band_x_values']]} x delay {v['within_band_delay_values']} "
                  f"| band {band:.5f} | {'DIAGONAL' if v['diagonal'] else 'point'}")
            print("      ridge (best servo per dcom_x): " + ", ".join(
                f"x{r['dcom_x']:+.3f}->d{r['best_delay']}{'*' if r['within_band'] else ''}" for r in v["ridge"]))
        truth_servo = servo or dict(delay_ticks=0, slew_rad_s=None, deadband=0.0)
        resB[name] = {"truth": {"dcom_x": dcom[0], "servo": {k: (None if v is None else float(v)) for k, v in truth_servo.items()}},
                      "per_seed": rows,
                      "argmin_x_correct": summarize([r["argmin_x_correct"] for r in rows]),
                      "argmin_delay_correct": summarize([r["argmin_delay_correct"] for r in rows]),
                      "truth_x_in_set": summarize([r["truth_x_in_set"] for r in rows]),
                      "truth_delay_in_set": summarize([r["truth_delay_in_set"] for r in rows]),
                      "diagonal": summarize([r["diagonal"] for r in rows]),
                      "split_half_x_agree": summarize([r["split_half_x_agree"] for r in rows]),
                      "split_half_delay_agree": summarize([r["split_half_delay_agree"] for r in rows]),
                      "band": seed_stat([r["band"] for r in rows]),
                      "within_band_count": seed_stat([r["within_band_count"] for r in rows]),
                      "x_set_size": seed_stat([len(r["x_determined"]) for r in rows]),
                      "delay_set_size": seed_stat([len(r["delay_determined"]) for r in rows])}

    # ---- verdict -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("VERDICT (resolved = the same on all three seeds)")
    print("=" * 100)
    for name, r in resA.items():
        print(f"  A {name:<22} identified {r['identified']['per_seed']} resolved={r['identified']['resolved']} | "
              f"truth in sets {r['truth_in_sets']['per_seed']} | x-set size {r['x_set_size']['per_seed']} "
              f"z-set size {r['z_set_size']['per_seed']} | band {r['band']['mean']:.5f} +-{r['band']['spread'] / 2:.5f} "
              f"| noise floor {r['noise_floor']['mean']:.5f} ({r['noise_floor']['mean'] / max(r['band']['mean'], 1e-12):.1f}x band)")
    for name, r in resB.items():
        print(f"  B {name:<30} x correct {r['argmin_x_correct']['per_seed']} delay correct {r['argmin_delay_correct']['per_seed']} "
              f"| truth x in set {r['truth_x_in_set']['per_seed']} truth delay in set {r['truth_delay_in_set']['per_seed']} "
              f"| diagonal {r['diagonal']['per_seed']} resolved={r['diagonal']['resolved']} "
              f"| within-band hyps {r['within_band_count']['per_seed']}")

    out = {"config": {**vars(args), "body": BODY, "candidate_seed": CAND_SEED, "noise_seed": NOISE_SEED,
                      "hidden_seeds": HIDDEN_SEEDS, "K": K,
                      "x_grid_m": X_GRID, "z_grid_m": Z_GRID, "delays": DELAYS, "slews": SLEWS,
                      "deadbands_deg": [float(np.rad2deg(d)) for d in DBS], "joint_hypotheses": len(joint_grid()),
                      "real_log_servo_candidate": {k: (None if v is None else float(v)) for k, v in SERVO_REAL.items()},
                      "published_DR": {"dcom_x": DR["dcom_x"], "dcom_z": DR["dcom_z"]}},
           "decision_rule": {"band": "1.4826*MAD/2 of err_A - err_B over every hypothesis, A/B = quarters of the fit half",
                             "determined": "values within band of the best error, other parameters at the argmin",
                             "resolved": "the per-seed boolean is the same on all three hidden seeds",
                             "seeds": "NOT paired: collect() draws sim.rng every tick for the push test, mode-dependently "
                                      "inside Excitation and once per episode in fresh(); three seeds answer whether the "
                                      "verdict holds on every seed, not whether a mean is significant",
                             "noise_floor": "|err(nominal candidate) - err(second nominal candidate, other collection seed)| "
                                            "on the same hidden log; a separation below it is not evidence",
                             "null_case": "nominal hidden body must identify to (0, 0) on every seed (asserted)"},
           "null_case_pass": null_ok,
           "identification_com": resA, "confound": resB, "runtime_s": time.time() - t_start}
    with open("results/com_id.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote results/com_id.json   total {time.time() - t_start:.0f}s")
    if not null_ok:
        print("NULL CASE FAILED -- the nominal hidden body did not identify to (0, 0) on every seed; "
              "see identification_com['nominal (null case)'] in the artifact")
        sys.exit(1)


if __name__ == "__main__":
    main()
