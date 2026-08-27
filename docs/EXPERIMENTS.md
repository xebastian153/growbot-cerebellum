# Experiments

Every number below is reproducible with the commands in the README; the
machine-readable source is `results/`. Conventions in `CONVENTIONS.md`.

## Contents

- [At a glance](#at-a-glance)
- [Forward model](#forward-model)
- [Mimic game](#mimic-game)
- [JS runner](#js-runner)
- [Sim-to-real proxy — negative](#sim-to-real-proxy--negative)
- [Contact friction — the twin has no torsional friction, and the DR negative could not have seen one](#contact-friction--the-twin-has-no-torsional-friction-and-the-dr-negative-could-not-have-seen-one)
  - [Part A — the coefficients were inert, so the missing sweep could not have mattered](#part-a--the-coefficients-were-inert-so-the-missing-sweep-could-not-have-mattered)
  - [Part B — decision rule, stated before the numbers](#part-b--decision-rule-stated-before-the-numbers)
  - [Results (within 0.2 rad, % of starts, 30 000 ticks per corner-seed, 3 seeds)](#results-within-02-rad--of-starts-30-000-ticks-per-corner-seed-3-seeds)
  - [What this settles](#what-this-settles)
- [Body parameters at 500 ms — a 3 cm centre-of-mass shift costs 33.8 pts of pitch, and 37–55 % of the held-out drop is the body, not the model](#body-parameters-at-500-ms--a-3-cm-centre-of-mass-shift-costs-338-pts-of-pitch-and-3755--of-the-held-out-drop-is-the-body-not-the-model)
  - [Is the drop the model, or the body?](#is-the-drop-the-model-or-the-body)
- [Centre-of-mass identifiability — the method that identifies the servo cannot identify the centre of mass (negative), and a slow servo leaves the delay undetermined even with the right body in the grid](#centre-of-mass-identifiability--the-method-that-identifies-the-servo-cannot-identify-the-centre-of-mass-negative-and-a-slow-servo-leaves-the-delay-undetermined-even-with-the-right-body-in-the-grid)
  - [A. Centre of mass alone (ideal servo) — the null case fails on every seed](#a-centre-of-mass-alone-ideal-servo--the-null-case-fails-on-every-seed)
  - [B. The confound — joint dcom_x × servo grid (196 hypotheses)](#b-the-confound--joint-dcom_x--servo-grid-196-hypotheses)
- [Yaw floor — mostly noise, not model: scaling is flat and privileged state adds little (negative)](#yaw-floor--mostly-noise-not-model-scaling-is-flat-and-privileged-state-adds-little-negative)
- [Actuator dynamics — the sim-to-real signature that is there, and how to recover it](#actuator-dynamics--the-sim-to-real-signature-that-is-there-and-how-to-recover-it)
- [Model mismatch — wrong-family identification recovers ~90 % of the gap; split-half catches drift, not shape](#model-mismatch--wrong-family-identification-recovers-90--of-the-gap-split-half-catches-drift-not-shape)
- [The first real logs — read per file, per segment](#the-first-real-logs--read-per-file-per-segment)
- [Real2Sim loop closure — an actuator model helps on the real walk; which actuator model is not identified](#real2sim-loop-closure--an-actuator-model-helps-on-the-real-walk-which-actuator-model-is-not-identified)
- [Identification ablation — per-side gains the most and proves the least, multi-horizon backfires, and the delay was over-charged by a tick](#identification-ablation--per-side-gains-the-most-and-proves-the-least-multi-horizon-backfires-and-the-delay-was-over-charged-by-a-tick)
- [The gesture and still captures — the still lane pays out, the gesture lane cannot](#the-gesture-and-still-captures--the-still-lane-pays-out-the-gesture-lane-cannot)
  - [The still lane: the phone's gyro noise, measured](#the-still-lane-the-phones-gyro-noise-measured)
  - [The gesture lane: the file determines nothing, and the reason is not established](#the-gesture-lane-the-file-determines-nothing-and-the-reason-is-not-established)
- [Coverage — RETRACTED: the experiment was invalid twice over](#coverage--retracted-the-experiment-was-invalid-twice-over)
- [Multi-step training loss — small, real gain](#multi-step-training-loss--small-real-gain)
- [Fall recovery through imagination — a feature, with a low physical ceiling](#fall-recovery-through-imagination--a-feature-with-a-low-physical-ceiling)
- [PETS — the model knows where it is unsure; planning through that knowledge does not help](#pets--the-model-knows-where-it-is-unsure-planning-through-that-knowledge-does-not-help)
- [Metadata conditioning — negative](#metadata-conditioning--negative)
- [TimesFM 2.5 baseline](#timesfm-25-baseline)

## At a glance

One row per experiment, the verdict with its numbers; every figure resolves to a file under
`results/`, and the section it links to carries the conditions.

| experiment | verdict |
|---|---|
| Forward model | 96.0 % within 0.2 rad at 100 ms, 82.7 % at 500 ms; yaw is the hard axis (59 % vs 77 % at 1 s); the gain over baselines concentrates in fast motion (41 → 86 %) and falls (58 → 89 %) |
| Mimic game | planning without a model is worse than doing nothing; with it, error halves (0.210 → 0.095 rad) and 39/40 traces beat hold-still |
| JS runner | float32-equivalent to the trained net; the equivalence test caught a real convention bug |
| Body-parameter DR proxy | **negative at 100 ms** — mass/CoM/leg/gain/*sliding* friction never reach the IMU there; contact chatter dominates the gyro. Re-scored at 500 ms below |
| Body parameters at 500 ms | shifting the base **centre of mass** 3 cm forward drops 500 ms pitch predictability by 33.8 pts (38.5 / 34.9 / 28.1 across seeds; RMSE +129 %), 3 cm back by 14.2, and leg 1.15 by 7.4 — while mass moves nothing, neither −20 % / +25 % on the whole body nor ±75 g at a fixed centre of mass, and neither do gain or sliding friction. At +3 cm the whole-body CoM crosses into the foot support box (body-frame x ±10.5 mm; the CoM goes from 16.9 mm behind it at nominal to 0.7 mm inside), and on the **held-out half** a model trained on that body still pays 13.5 of the 30.2 pts there — **37–55 % across the three seeds**, a range and not a point estimate, and a different quantity from the 33.8-pt full-stream figure. The drop is resolved by seeds; the split is not, and the oracle behind it is a lower bound, not a ceiling. **No corner here mounts a phone**: the twin carries none, and a 200 g one would be mass_scale 1.42, outside the sweep |
| Centre-of-mass identifiability | **negative** — the one-step score that identifies the servo cannot identify the centre of mass from IMU + commands: 0 of 12 body-seeds identified, the nominal body reads as ±1.5 cm off on every seed (null case fails, asserted), and the shipped model beats every body-matched candidate on held-out data. What holds, 3 of 3: a CoM shift does **not** read as a servo delay (joint grid recovers +0.030 m and delay 0 with an ideal servo). What does not resolve: with the real-log servo in the loop the delay set is the whole grid on all 6 seeds and a 1.5 cm CoM offset enters the argmin on 2 of 3 — the servo identified on the real log may carry a little body inside it |
| Contact friction | the twin has **no torsional friction**: at the shipped `condim=3` the XML's torsional and rolling coefficients are inert (bit-identical under a ×100 change). Switched on, torsional still moves nothing on any axis, yaw included — the DR negative survives a proper test. The XML's own unapplied rolling value costs 22.6 pts of yaw at 500 ms if enabled |
| Actuator dynamics | the sim-to-real signature that *is* there: a slew-limited servo opens a 3–4 pt gap; the servo is identifiable from IMU + commands alone |
| Multi-step training loss | +1.3 pts at 500 ms, consistent across seeds, saturates by H=5 |
| Fall recovery | planner doubles hold-still (18 → 37 %) on recoverable falls; ceiling is the two-legged body |
| PETS | uncertainty is well calibrated per regime; planning through it is neutral-to-harmful |
| Metadata conditioning | **negative** — a forward model has no quality axis for a tag to separate; one model serves two bodies regardless |
| TimesFM 2.5 baseline | a 200M-param action-blind forecaster ties persistence; the information is in the action |
| `?imulog=1` parser | round-trip validated: a hidden servo survives 60/30 Hz jittered sampling into the 50 Hz arrays, determined to one grid step — the injected 40 ms delay is inside the determined set on all 5 seeds tested and the set never leaves ±20 ms; slew resolves to within one grid step of the injected 5 rad/s (the set contains it on 4 of 5). The argmin alone is a coin flip here and is no longer the acceptance rule |
| Yaw floor | **negative** — the twin's own yaw weakness (58.6 % @1 s vs ~83 % roll/pitch) is not a model limit: 4× data +0.9 pts, 4× capacity +1.3, and a teacher-forced probe fed true contact forces and linear velocity +3.1 — all under the pre-stated 4.6-pt materiality threshold (2× seed spread). Contact chatter is aleatoric at 20 ms; planning should not chase it. Sim-only, 3 MLP seeds |
| Model mismatch | out-of-family servos (load-dependent slew, voltage sag) identify to their nearest grid point and still recover 90–94 % of the closable held-out gap at 500 ms; split-half DISAGREE fires on drift but is blind to stationary mismatch; a linear residual on top adds nothing (**negative**) — seed 777, sim-only |
| Real logs (2 sessions, 21 s) | read **per file, per segment** — the two logs differ in agent gain and rest 43° apart, so they are never pooled. `growbot-imulog-1` carries no event rows, so the regimes are synthesized from the data: walk-1 = still 1.1 s + walking 15 s; walk-3 = still 2.6 s + impact + fall, and **no walking at all**. walk-1's walking gap @500 ms is −36.9 / −37.1 / −43.0 pts vs the twin's policy floor (−0.4 to +2.2 @100 ms). walk-3's *motionless* segment reads 4.8 % on pitch while the commands swing ±34° — that number measures the robot, not the model. Servo on walk-1 alone: argmin delay 100 ms / slew 2.0, split-half DISAGREE, delay determined set [2 … 6] ticks and slew [2.0, 3.0] rad/s. Fusion lag +12.8 to +13.8 ms, split-half AGREE (walk-1 only). Measured, not assumed: `header.gain` is baked into the logged commands (amplitude ratio 1.230, CI [1.171, 1.281] against 1.25 vs 1.00) |
| Real2Sim loop closure | an actuator model helps on the real walk; **which** one is not identified. On walk-1's held-out half, retraining the twin against a servo from the determined band closes roll at every tested point — but the two cells inside both determined sets spread +16.0 to +33.4 pts, and the tested cells cover only 40 % of the determined delay set and 50 % of the slew set, so "robust to the identification uncertainty" is **retracted**. (The wider +4.5 to +33.4 range across all four cells is not that swing: its low end is `half-A` at slew 5.0, outside the slew set [2.0, 3.0], so the identification does tell that one apart.) A zero-delay smoothing-only cell closes yaw to within 2.7 pts of the best delayed cell — under the 10.7-pt threshold — so this log does not separate identified dynamics from plain action smoothing on yaw; on roll and pitch the latency does carry more. That cell is **outside** the determined band (delay 0 ∉ [2 … 6]), so it is an action-smoothing control rather than a rival hypothesis about the same servo — the earlier "inside the band" reading came from a hand-copied mirror of the determined sets that had gone stale; `real2sim` now reads them from `results/real_log_report.json` and fails hard if it cannot. Delay is determined only to [2 … 6] ticks (40–120 ms), deadband untested, 405 held-out ticks |
| Coverage | **RETRACTED** — invalid twice over. The premise was false (no sit-to-stand exists in either log: the header puts the fold *after* recording ends, and the −1.0 rad tail is a **fall**), and the manipulation was null (the standard data already spans pitch −1.570 to +1.570 against the transitions' −1.568 to +1.459 and walk-3's −1.013 to +0.014 — the sanity check compared the treatment with the target and skipped the control, so it could not fail). The additivity claim went with it: its differences sit inside one control seed spread. Numbers kept in `results/coverage.json`, conclusion marked retracted; the sanity check is fixed so a rerun would be honest |
| Identification ablation | four changes to how the servo is identified, on walk-1's held-out half. **Per-side servos give the largest gain and the weakest claim to it**: fitting one triple per horn instead of one for both more than doubles the roll gain, +3.5 → +8.0 pts (real, and it reproduces from the JSON) — but the *fit* improvement behind it, 0.0064, is a ratio of 1.01 against its own confidence band of 0.0063, i.e. on the noise floor; both per-side argmins sit on the grid **boundary** (delay 6 = max, slew 1.0 = min); and the "disjoint" slew sets are one-dimensional conditional slices, each swept with the partner frozen and cut with the *shared* band, so their disjointness restates the argmins rather than confirming them. What does survive a test: re-fitting per-side on each half puts the slower horn on the **right** both times — quoted with its noise floor, because it is now the only support left for that attribution: two halves each picking one of {left, right, neither} agree under a no-asymmetry null roughly 1 time in 2, so the flag is about one coin flip's worth of evidence, the same standard that rejects the 1.01 fit ratio above. (The L/R labels were **inverted** before this revision — action column 0 is the right leg, not the left — so every earlier left/right attribution here was backwards; no error, gain or set changes, only which horn owns which triple.) **Aligning the observation channels** (advancing the fused angles by the measured 13.2 ms so they meet the gyro) moves the identified delay 5 → 4 ticks and narrows its set, with no change in held-out accuracy: the correction is to the attribution, not the prediction, and 80 ms is now an upper bound on the actuator rather than its value. **Multi-horizon scoring backfired** — argmin to the grid's low boundary, delay set widens to everything, band ×14, roll gain negative: the published horizon ablation does not transfer at 16 s. Every variant still splits DISAGREE; none of this fixes the excitation |
| Sensor characterisation | round-trip validated: a hidden 60 ms fusion-filter lag recovered on 3 of 3 body-rate axes (+61.5 / +61.6 / +62.1 ms, peak corr 0.89–0.92; 60.4–63.5 ms across 5 seeds), gyro noise density within 20 %, an injected timing stall flagged, and a bias instability refused on a gyro that has none — all from the file alone |

**The twin is faithful.** The shipped `policy_85mm.json`, ported to numpy, walks in it
(0.138 m in 5 s, no fall). It is the same net that drives the physical robot.

## Forward model

Held-out episodes, model rolled forward on its own predictions, fraction of starts whose
imagined roll/pitch stays within 0.2 rad of the truth. 128×2 swish MLP, 24,841 params, K=5.

| horizon | persistence | linear (GCML-style) | **MLP** |
|---|---|---|---|
| 100 ms | 85.9 % | 93.5 % | **96.0 %** |
| 500 ms | 59.0 % | 75.0 % | **82.7 %** |

Per axis (now reported separately in `results/forward_K5.json`): yaw is consistently the
hardest angle — MLP at 1 s sits at 59 % within 0.2 rad against 77 % for roll/pitch —
consistent with contact dominating the yaw gyro. By regime at 100 ms the three tie on calm gaits (~95 %); the model earns its keep under
fast motion (persistence 41 → MLP 86 %) and while tipping or fallen (58 → 89 %) —
exactly where the video locates the gap. Capacity sweep saturated (192×2, 128×3 add nothing).

Reproduce: `.venv/bin/python forward.py` → `results/forward_K5.json` (80 epochs, the CLI default;
`results/logs/forward_K5.txt` is that run's redirected stdout). Total 33 s wall, 6 cores.

## Mimic game

Reproduce a held-out 2 s motion by planning through imagination and executing in the twin.
40 traces, receding-horizon CEM, replan every 100 ms.

| planner | roll/pitch RMSE | beats holding still |
|---|---|---|
| hold still | 0.210 | — |
| plan without a forward model | 0.220 | 42 % |
| plan with the linear model | 0.142 | 88 % |
| **plan with the MLP** | **0.095** | **98 %** |

Planning without imagination is worse than doing nothing — the failure in the video.
Replanning every 100 ms is the optimum (every tick chases noise; pure 2 s imagination
drifts to 0.161 but still beats rest). 100 ms is also the sensory-delay figure in the video.

## JS runner

`node forward-model/test_forward.mjs`: single step 6.1e-6, 25-tick rollout 9.2e-6 versus
PyTorch (the reference vectors and weights regenerated from the current `data/train.npz`, seed 0;
the shipped weights score 96.0 / 82.7 / 77.4 % at 100 ms / 500 ms / 1 s on the held-out set,
`results/export_js.json`); planner beats hold-still on a reachable target with a seeded RNG. This test caught
a real off-by-one in the Python evaluation (which action sits in history slot 0) before
it could reach a phone.

Reproduce: `.venv/bin/python export_js.py` → `forward-model/forward_85mm.json`,
`forward-model/reference_vectors.json`, `results/export_js.json`; then
`node forward-model/test_forward.mjs`. Total ~40 s.

## Sim-to-real proxy — negative

Forward model trained on the nominal Olie body, measured on the 13 domain-randomisation
corners from GrowBot's `policy/Harsh_policies/SPIN_IN_PLACE_OLIE_EXPORT/dr_sweep_spin.py` —
not in this repository; its ranges are copied into `sim/growbot_sim.DR` — (mass 0.8–1.25, CoM ±3 cm, leg 0.85–1.15, gain 0.75–1.25,
friction 0.6–1.4), with an online linear residual learning from prediction error.
**Nothing to correct:** frozen 93.9 % across corners vs 93.7 % nominal, per-corner yaw
bias ±0.02, residual only adds noise. Tick-to-tick gyro change is mostly unpredictable in
the twin itself (R² ≈ 0.2, ≈ 0.05 when calm): foot–floor contact chatter, not model error.
So the project's DR does not show up in the IMU at 100 ms — consistent with its good walk
transfer — and the spin gap is unlikely to be mass/CoM/leg/gain. Contact is the untested
factor, and contact drives yaw, which drives spin.

Three limits of this section, all established later: the corner labelled "friction
0.6–1.4" varies **sliding** friction alone (`geom_friction[:, 0]`); the metric scores
roll and pitch only — yaw is not in it; and it scores 100 ms only. *Contact friction*
below addresses the first two and leaves the negative standing. *Body parameters at
500 ms* addresses the third and does not: mass — whether −20 % / +25 % on the whole body
or ±75 g at a fixed centre of mass — gain and sliding friction stay invisible at 500 ms,
but the centre of mass is material there, and so is leg length 1.15 on pitch (−7.4). Not
on every axis, though: of the five CoM corners, three move all three axes, CoM back/low
moves pitch alone and CoM z +0.015 roll alone. And of the largest drop, 3 cm forward on
pitch, 37–55 % across three seeds is the stream itself getting harder rather than the frozen
model being wrong — a model trained on that body pays 13.5 of the 30.2 **held-out** points
too, a different quantity from the 33.8-pt full-stream figure.

## Contact friction — the twin has no torsional friction, and the DR negative could not have seen one

The sim-to-real proxy above concludes with "contact is the untested factor, and contact
drives yaw, which drives spin". This tests it, and starts by finding two holes in the
negative itself, both on the spin axis:

1. `perturb()` set `geom_friction[:, 0]` — **sliding** friction. Torsional (column 1) and
   rolling (column 2) were never varied. The published corner labelled "friction 0.6–1.4"
   is a sliding-friction corner.
2. `sim2real_proxy.horizon_within` scores `decode_obs(...)[:, :2]` — roll and pitch. **Yaw
   was not in the metric at all**, on the axis the same section names as the one that
   matters.

### Part A — the coefficients were inert, so the missing sweep could not have mattered

Both bodies ship with `condim="3"`. Under condim 3 MuJoCo solves a three-dimensional
contact — one normal, two tangential — and the torsional and rolling coefficients are not
in the solve. Measured rather than asserted, 6000 ticks against the unperturbed stream:

| probe | effect on the trajectory |
|---|---|
| sliding 0.6 (positive control) | acts, max abs delta obs **26.7** |
| torsional ×10 at condim 3 | **bit-identical** |
| torsional ×100 at condim 3 | **bit-identical** |
| rolling ×100 at condim 3 | **bit-identical** |
| MuJoCo default contact at condim 3 | **bit-identical** |
| torsional ×100 at condim 4 | acts, 19.0 |
| torsional at MuJoCo's default, condim 4 | acts, 28.8 |
| rolling ×100 at condim 6 | acts, 22.7 |

So the `0.1 0.1` the XML declares for torsional and rolling has never been applied. The
body has no torsional friction — not a badly tuned one, none — and sweeping those two
columns would have changed nothing. The honest question is therefore not "did we forget
two columns" but "does the mechanism the twin cannot represent matter".

### Part B — decision rule, stated before the numbers

Protocol is the proxy's: the frozen nominal forward model trained once on
`data/olie_train.npz`, evaluated open-loop on each corner's own stream, seeds shared across
every corner. Two metrics — the proxy's own `within_0.2rad` at 100 ms over roll/pitch, kept
so old and new corners are comparable, and `within_0.2rad` **per axis** at 100 and 500 ms,
which adds yaw. Ranges are anchored on the XML's `friction="1.2 0.1 0.1"` and MuJoCo's
defaults `1 0.005 0.0001`, plus one decade above the XML value; nothing invented.
Material = a shift from nominal larger than `max(3.0 pts, 2× nominal seed spread)`, the
rule `yaw_floor.py` and `real2sim.py` use. Measured nominal spread over 3 seeds: **2.30 pts**
→ threshold **4.60 pts**.

### Results (within 0.2 rad, % of starts, 30 000 ticks per corner-seed, 3 seeds)

| corner | legacy 100 ms | yaw @100 ms | yaw @500 ms | Δ yaw @500 ms |
|---|---|---|---|---|
| nominal (as shipped, condim 3) | 94.6 % | 96.2 % | 75.6 % | — |
| sliding 0.6 (published corner) | 94.3 % | 95.8 % | 74.2 % | −1.4 |
| sliding 1.4 (published corner) | 94.4 % | 95.2 % | 74.4 % | −1.2 |
| torsional 1.0 at condim 3 | 94.6 % | 96.2 % | 75.6 % | **+0.0** |
| rolling 1.0 at condim 3 | 94.6 % | 96.2 % | 75.6 % | **+0.0** |
| condim 4, torsional 0.005 (MuJoCo default) | 94.2 % | 96.4 % | 76.0 % | +0.4 |
| condim 4, torsional 0.1 (XML value) | 93.8 % | 96.2 % | 74.9 % | −0.7 |
| condim 4, torsional 1.0 (decade above XML) | 94.3 % | 96.6 % | 72.2 % | −3.5 |
| condim 6, torsional 0.1 rolling 0.1 (XML values) | 89.6 % | 94.2 % | 53.0 % | **−22.6** |
| condim 6, MuJoCo default contact | 94.5 % | 95.8 % | 75.0 % | −0.7 |
| condim 6, torsional 0.1 rolling 0.0001 (rolling OFF) | 94.3 % | 95.9 % | 74.1 % | −1.6 |
| condim 6, torsional 0.005 rolling 0.1 (torsional OFF) | 88.6 % | 92.8 % | 48.6 % | **−27.1** |

The two condim-3 rows are exactly `+0.0` on every axis and every horizon — the same
bit-identity Part A measured, now visible in the metric.

### What this settles

- **The published negative survives, and is stronger for having been tested properly.**
  With torsional friction actually switched on (condim 4) and swept from MuJoCo's default
  through a decade above the XML value, nothing is material on any axis — yaw included.
  The largest yaw excursion is −3.5 pts against a 4.60-pt threshold.
- **The hole was real but benign.** The two unswept columns could not have moved anything
  at the shipped contact dimension.
- **One thing does move the body, and it is the coefficient nobody applied.** At condim 6
  the XML's own rolling value of 0.1 — 1000× MuJoCo's default — costs 22.6 pts of yaw and
  41.7 of pitch at 500 ms. The isolation is direct, not inferred: at condim 6 with the XML's torsional value
  and rolling switched off (0.0001), yaw moves −1.6 pts — flat; with torsional switched
  off (0.005) and the XML's rolling value kept, yaw moves −27.1 pts, more than the XML
  row itself. The mover is the rolling coefficient, not the contact dimension. Anyone who raises `condim` to model torsional
  friction will silently switch on that rolling resistance as well and get a different
  robot.
- **What it does not settle.** This metric is forward-model *prediction* accuracy, not
  policy *transfer*. A contact term can decide whether a spin policy survives the trip to
  hardware while barely changing how predictable the body is; those are different
  questions and this experiment answers only the second.
- **For `yaw_floor.py`**: its aleatoric yaw ceiling was measured at condim 3, i.e. with
  both coefficients inert. Switching torsional on does not move predictability materially,
  so that conclusion is robust to *this* parameter. It remains conditional on the contact
  model in general — the condim-6 row shows the contact model can move yaw by 22 pts — and
  a re-test under a contact model chosen for physical realism rather than inherited default
  would be worth having.
- One honesty note the table carries: the `condim 4, torsional 0.005` row has a 5.40-pt
  seed spread of its own, wider than the 4.60-pt threshold, so its +0.4 is not evidence in
  either direction.

Reproduce: `.venv/bin/python contact_friction.py` → `results/contact_friction.json`,
`results/logs/contact_friction.txt`. Total 140 s.

## Body parameters at 500 ms — a 3 cm centre-of-mass shift costs 33.8 pts of pitch, and 37–55 % of the held-out drop is the body, not the model

Shifting the base centre of mass 3 cm forward drops 500 ms pitch predictability by 33.8 pts
(38.5 / 34.9 / 28.1 across three seeds), while ±75 g at a fixed centre of mass moves nothing
on any axis. At +3 cm the twin's whole-body CoM crosses into the foot support box, and on the
**held-out half** a model trained on that body still pays 13.5 of the 30.2 points there — so
part of that drop is the body itself, not the model.

Three things about that last sentence, the first two of which an earlier version of this
section got wrong. **The 13.5 and the 30.2 are held-out-half points, and the 33.8 above them
is the full-stream table figure**: they are different quantities measured on different data,
and one is never divided into the other. **The share is not one number**: per seed it is
37 / 42 / 55 %, so it is published as a range — the point estimate 45 % this section carried
is the ratio of the two means, and standing alone it hid a spread wider than most of the
effects this file calls null. And **three seeds resolve the drop but not the split**: the
corner's frozen seeds separate from nominal's by 22.6 pts against a 4.60-pt bar, while the
oracle's separate by 9.4 against the 11.70-pt bar its own nominal spread sets. The direction
of the split is supported; its magnitude is a measurement this run makes, not one it pins
down.

The *Sim-to-real proxy* section above found body-parameter randomisation invisible in the
IMU, but it scored one horizon and two axes: `horizon_within(..., h=5)`, 100 ms, roll and
pitch. *Contact friction* closed that limit for friction. This section closes it for the
parameters a builder actually changes — mass, centre of mass, leg length — on the identical
protocol (`body_params.py`, reusing `contact_friction.py`'s scoring and `sim2real_proxy.py`'s
own 13 corners so the old table and this one are the same points), at 100 **and** 500 ms, yaw
included.

**Anchors, and what the ±75 g corners are not.** The twin's base body is 427 g and the whole
body 480 g, and **the twin carries no phone at all**: that 427 g is battery plus structure
(`sim/growbot_olie_body.xml`, lines 16–19), with its centre of mass at x −30.8 mm, z −3.0 mm,
i.e. at the battery, below the torso mid-plane. `base_mass_delta` adds or removes mass **at
that existing centre of mass**, leaving `body_ipos` untouched, so the ±75 g corners
(whole-body mass_scale 0.844 / 1.156) isolate mass at a fixed CoM. They are not a phone swap,
and **no corner in this file represents a mounted phone**: a real 200 g phone would be +200 g
on 480 g, a whole-body mass_scale of 1.42, outside every corner tested here and outside the
published DR range (0.80–1.25) as well. The CoM corners are the published DR endpoints — CoM
x ±0.03 m, z −0.01 / +0.015 m, mass 0.80 / 1.25, leg 0.85 / 1.15 — anchored to that sweep and
to nothing else. For scale only: with mass fixed, 30 mm of base-CoM shift is what moving a
200 g phone 94 mm would do, beyond the 84 mm torso half-length.

**Balance geometry, and a correction to the millimetres.** The two leg geoms sit at x = 0 with
a half-x of 10.5 mm, so the feet can support the body over x ±10.5 mm — a **body-frame**
extent. Measured in that same frame, at the reference pose with every joint at zero, the
nominal whole-body CoM sits at x **−27.42 mm**, 16.9 mm **behind** the box. The +3 cm corner
moves it to **−0.71 mm**, **inside** it with 9.8 mm to spare; the −3 cm corner moves it to
**−54.12 mm**, 43.6 mm outside.

The numbers this section published first — −19.30, −0.71 and −41.34 mm — were wrong for two
of the three rows, and the error was a pose artefact, not a physics disagreement.
`balance_geometry()` read the CoM in **world** x while comparing it against a body-frame box,
and it read it at whatever pose `GrowBotSim.__init__` had left the body in, because `reset()`
steps 0.5 s of physics before anything is measured. The nominal and −3 cm bodies settle
**rotated −34.5° in pitch**, so those two rows were a body-frame offset projected onto a world
axis: they read 8.1 and 12.8 mm short. The +3 cm body settles level (−0.1°), which is why its
−0.71 mm was already right and why the artefact was invisible in the one row the argument
leans on. Pose dependence of the corrected number is small and bounded: the legs are 11.0 % of
the mass with their own CoM 37 mm below their hinges, so a full ±90° swing moves the
body-frame CoM by at most 4.1 mm, and at the settled pose it moves it by less than 0.1 mm.
The whole world-versus-body gap is base rotation.

**The qualitative reading survives the correction, and it is the only part of it this section
uses**: nominal and −3 cm sit outside the support box, +3 cm sits inside it. So +3 cm is not
only a parameter change, it is a change of balance regime, and the two directions are not
symmetric — which is exactly what the table below shows (−33.8 pts of pitch forward against
−14.2 back for the same 3 cm) and what the partition after it is there to separate from model
error. What the correction does remove is any reading of the *margins* as small: the nominal
CoM is 16.9 mm outside a 10.5 mm box, not 8.8.

**Decision rule, stated before the numbers, and changed from the earlier run.** Material =
|mean Δ vs nominal| > max(3.0 pts, 2× the nominal seed spread **of that metric and horizon**).
The earlier version of this section took the worst nominal spread across all metrics (yaw at
100 ms, 2.30 pts) and applied the resulting 4.60-pt bar to every metric and every horizon,
which makes a null on a quiet metric far too easy to declare. Per metric the nominal spreads
are 0.87 (legacy), 1.50 / 0.25 / 2.30 (roll / pitch / yaw at 100 ms) and 1.50 / 1.15 / 1.25 at
500 ms, so every bar is now 3.00 pts except yaw at 100 ms, which stays at 4.60. Six verdicts
flip under the new rule, all from flat to material: CoM back/low roll −4.6 and yaw −4.3,
leg 0.85 pitch −4.0, leg 1.15 yaw −3.0, worst B roll −3.7 and pitch −3.5. **All six are
unresolved at three seeds** (separations −0.2 to −2.4 pts against their own 3.00-pt bars) and
**none of the six is reported as moved**. The flip says the earlier bar was too generous, not
that six new effects were found.

Two further corrections to how these numbers are read:

- **The seeds are not paired.** Corners share seeds in the sense that they start from the
  same initial condition — not in the sense of common random numbers. This section previously
  gave the wrong reason for that ("`collect()` draws `sim.rng.random()` only when the body has
  fallen"), and the code says otherwise: the push test draws `sim.rng.random()` on **every
  tick**, whether or not a push follows; `Excitation` draws mode-dependently; `fresh()` draws
  once per episode; and the fall check adds a further draw only when the body is *already*
  fallen. Two corners therefore diverge at the first tick where their dynamics differ and
  never realign — desynchronisation does not need a fall to start it. The conclusion is the
  one it always was, and it holds for stronger reasons: paired-difference reasoning does not
  apply here.
- **Every verdict is published beside the seed spread of the corner it was measured on**, the
  honesty note the *Contact friction* table already carries, and a verdict is reported as
  moving an axis only when it is material **and resolved**: the corner's three seeds separate
  from nominal's three by more than that same bar. A mean past the bar with overlapping seed
  ranges is not resolved at three seeds, whichever side of the bar it fell on. 20 of the 102
  axis rows have a seed spread that reaches their own threshold.

**At 100 ms nothing moves.** The largest shift on any axis in any corner is 2.7 pts and the
largest on the published metric is 2.7 pts, both under the bar, and no corner separates from
nominal by seeds. The original negative reproduces exactly at the horizon it was measured.

**At 500 ms, within 0.2 rad, Δ vs nominal in pts ± that corner's own 3-seed spread** (nominal
85.2 / 86.9 / 75.6 on roll / pitch / yaw). **Bold** = material *and* resolved, i.e. reported
as moved; `?` = material by the mean but not resolved at three seeds. Every row carries its
own spread whether it moved or not:

| corner | legacy 100 ms | roll | pitch | yaw |
|---|---|---|---|---|
| mass 0.80 | −0.6 ±0.4 | −0.8 ±3.0 | −0.6 ±2.6 | −0.5 ±0.6 |
| mass 1.25 | −0.2 ±2.0 | +2.5 ±0.8 | +0.8 ±1.6 | +1.5 ±1.0 |
| base −75 g at a fixed CoM | +0.5 ±0.3 | +0.2 ±1.1 | −0.5 ±2.5 | −1.6 ±3.7 |
| base +75 g at a fixed CoM | −0.2 ±0.7 | +1.4 ±2.3 | −0.3 ±1.1 | −0.9 ±0.7 |
| leg 0.85 | −0.0 ±0.2 | +2.7 ±3.1 | −4.0 ±2.5 `?` | −2.7 ±1.8 |
| leg 1.15 | −0.4 ±0.9 | −1.6 ±3.5 | **−7.4 ±3.7** | −3.0 ±1.1 `?` |
| gain 0.75 | −0.3 ±0.7 | +1.0 ±1.5 | +0.4 ±0.8 | −0.0 ±0.5 |
| gain 1.25 | −1.1 ±1.4 | −0.5 ±5.6 | −0.3 ±4.8 | −2.1 ±3.2 |
| sliding friction 0.6 | −0.3 ±0.5 | +1.3 ±3.3 | +0.4 ±2.9 | −1.4 ±2.6 |
| sliding friction 1.4 | −0.3 ±0.5 | −0.1 ±1.2 | −0.4 ±1.6 | −1.2 ±2.6 |
| CoM back/low (published corner) | −0.7 ±2.1 | −4.6 ±3.6 `?` | **−10.6 ±4.3** | −4.3 ±2.0 `?` |
| CoM fwd/high (published corner) | −2.7 ±0.9 | **−13.1 ±3.8** | **−29.4 ±5.1** | **−10.9 ±1.9** |
| CoM x −0.03 only | −2.5 ±2.7 | **−7.8 ±2.0** | **−14.2 ±2.5** | **−8.1 ±3.7** |
| CoM x +0.03 only | −1.6 ±1.5 | **−12.4 ±4.4** | **−33.8 ±10.4** | **−12.2 ±6.2** |
| CoM z +0.015 only | −0.7 ±1.7 | **−5.1 ±1.0** | −4.8 ±1.7 `?` | −1.4 ±1.1 |
| worst A (heavy, long, weak, slippery) | −2.6 ±1.3 | **−13.1 ±2.2** | **−26.7 ±2.8** | **−10.1 ±4.6** |
| worst B (light, short, strong, grippy) | −0.7 ±1.0 | −3.7 ±5.9 `?` | −3.5 ±3.5 `?` | **−6.4 ±2.8** |

**The same rows against RMSE.** within-0.2 rad is a threshold crossing; RMSE is the error
magnitude behind it, and an axis that moves one without the other has not moved. Δ RMSE vs
nominal (nominal 0.462 / 0.238 / 0.519 rad on roll / pitch / yaw at 500 ms):

| corner | roll within / RMSE | pitch within / RMSE | yaw within / RMSE |
|---|---|---|---|
| base −75 g at a fixed CoM | +0.2 pts / −8.3 % | −0.5 pts / −3.4 % | −1.6 pts / +7.4 % |
| base +75 g at a fixed CoM | +1.4 pts / −6.3 % | −0.3 pts / +2.9 % | −0.9 pts / +3.9 % |
| CoM back/low | −4.6 pts / −7.5 % | −10.6 pts / +5.8 % | −4.3 pts / +6.2 % |
| CoM fwd/high | −13.1 pts / +33.0 % | −29.4 pts / +123.3 % | −10.9 pts / +15.0 % |
| CoM x −0.03 only | −7.8 pts / +26.0 % | −14.2 pts / +33.2 % | −8.1 pts / +30.8 % |
| CoM x +0.03 only | −12.4 pts / +4.1 % | −33.8 pts / +129.4 % | −12.2 pts / +7.5 % |
| CoM z +0.015 only | −5.1 pts / +40.4 % | −4.8 pts / +11.6 % | −1.4 pts / +27.7 % |

On the headline corner the two metrics agree on **pitch only**: RMSE more than doubles
(+129.4 %) where within-0.2 rad loses 33.8 pts, while roll and yaw cross the threshold with an
error magnitude that barely moves (+4.1 % and +7.5 %). The claim for CoM x +0.03 is therefore
made about pitch. The −3 cm corner is corroborated on all three axes (+26.0 / +33.2 / +30.8 %),
and so is CoM z on roll (+40.4 %).

### Is the drop the model, or the body?

The frozen model is scored on each corner's own stream, so a corner that is intrinsically
harder to predict costs points even for a model that knows the body perfectly. `score_corners`
now keeps the mode channel `collect()` returns, so every corner also reports its fall rate
(the sim's own `fallen()`: |roll| or |pitch| > 1.2 rad), its regime mix, its per-regime
accuracy, and an **oracle** — a model trained on that corner's own first half, with frozen and
oracle both scored on the held-out second half. That splits each drop exactly:

    frozen_c − frozen_nom  =  (oracle_c − oracle_nom)  +  [mismatch_c − mismatch_nom]
                               intrinsic difficulty       model mismatch the corner adds

**Everything below is per seed, with its spread and its verdict.** The first version of this
table published three-seed means and nothing else — no spread, no per-seed values, no
resolution mark — which is exactly the defect the axis verdicts above were rewritten to
remove, reintroduced one section later. Each quantity now carries the same two questions the
axis table asks: is the mean past a bar of max(3.0 pts, 2× the **nominal** spread of that same
quantity), and do the corner's three seeds separate from nominal's three by more than that bar
(`seed_separation`, unpaired). The bars differ by quantity because the spreads do: on pitch the
frozen bar is 4.60 pts and the oracle bar is **11.70**, because the oracle is a small-data model
whose own nominal seeds run 81.9 / 82.6 / 87.7.

Fall rate, frozen and oracle held-out pitch, mean ± 3-seed spread (nominal fall rate
9.7 ± 2.9 %, fast 11.0 ± 1.0 %, frozen 86.1 ± 2.3 %, oracle 84.0 ± 5.8 %):

| corner | fall rate | fast (\|gyro\|>3) | frozen pitch | oracle pitch | drop | intrinsic | intrinsic share, per seed | drop resolved | intrinsic resolved |
|---|---|---|---|---|---|---|---|---|---|
| nominal (published) | 9.7 ±2.9 % | 11.0 ±1.0 % | 86.1 ±2.3 % | 84.0 ±5.8 % | — | — | — | — | — |
| CoM back/low | 4.4 ±0.1 % | 9.1 ±0.9 % | 77.8 ±2.3 % | 87.4 ±4.0 % | −8.3 ±1.3 | +3.4 ±4.0 | −72 / −38 / −17 % | yes | no |
| CoM fwd/high | 16.0 ±2.4 % | 12.1 ±0.5 % | 56.4 ±14.0 % | 70.7 ±2.2 % | −29.7 ±12.9 | −13.4 ±5.3 | 32 / 39 / 75 % | yes | no |
| CoM x −0.03 only | 12.0 ±4.0 % | 7.4 ±0.8 % | 72.4 ±4.2 % | 82.8 ±3.5 % | −13.6 ±3.1 | −1.3 ±2.9 | −1 / 8 / 23 % | yes | no |
| CoM x +0.03 only | 10.1 ±2.6 % | 14.0 ±1.1 % | 55.9 ±12.0 % | 70.6 ±3.2 % | −30.2 ±13.2 | −13.5 ±8.3 | 37 / 42 / 55 % | yes | **no** |
| CoM z +0.015 only | 16.2 ±2.0 % | 8.2 ±1.0 % | 83.1 ±1.2 % | 81.7 ±0.9 % | −3.0 ±2.0 | −2.4 ±5.6 | 17 / 28 / 139 % | no | no |
| leg 1.15 | 10.4 ±2.8 % | 9.4 ±0.3 % | 80.3 ±5.8 % | 78.9 ±5.2 % | −5.7 ±3.6 | −5.1 ±3.4 | 39 / 105 / 161 % | no | no |
| base +75 g at a fixed CoM | 9.0 ±2.7 % | 11.3 ±0.5 % | 86.6 ±1.0 % | 83.5 ±1.3 % | +0.6 ±2.3 | −0.5 ±7.1 | 117 % (2 seeds under 1 pt) | no | no |
| worst A | 14.0 ±4.0 % | 10.6 ±0.4 % | 59.9 ±4.3 % | 73.2 ±2.2 % | −26.1 ±5.5 | −10.8 ±6.4 | 29 / 42 / 54 % | yes | no |

The per-seed shares are seed-**index** readings — corner seed *i* against nominal seed *i* —
not paired differences, for the reason given in the decision-rule bullets above; a share is
only formed where that seed's own drop reaches 1 pt. A positive intrinsic figure against a
negative drop, as at CoM back/low, means the oracle did *better* on that corner than on
nominal: the whole drop there is mismatch and then some.

**The headline corner, all three axes, per seed.** At CoM x +0.03 the held-out pitch drop is
−30.2 pts (−22.6 / −35.8 / −32.2) and the part of it a model trained on that body still pays
is −13.5 (−9.4 / −13.3 / −17.7), leaving −16.8 of mismatch. Exactly: −13.47 and −16.75 against
a drop of −30.22 — the one-decimal figures do not re-add, and the section quotes them rather
than a tidier pair that would. The intrinsic share is **37 / 42 / 55 % across the three
seeds**; the 45 % this section used to print is the ratio of the two means and is not
reported alone any more.

| axis | drop (3 seeds) | intrinsic (3 seeds) | mismatch | share | drop resolved | intrinsic resolved | oracle's own deficit |
|---|---|---|---|---|---|---|---|
| roll | −12.6 (−10.2 / −12.9 / −14.8) | −6.3 (−7.2 / −4.0 / −7.8) | −6.3 | 31 / 52 / 71 % | yes (−9.8 vs 6.1) | no (−3.4 vs 8.2) | −4.4 |
| pitch | −30.2 (−22.6 / −35.8 / −32.2) | −13.5 (−9.4 / −13.3 / −17.7) | −16.8 | 37 / 42 / 55 % | yes (−22.6 vs 4.6) | no (−9.4 vs 11.7) | −2.0 |
| yaw | −11.2 (−6.9 / −13.6 / −13.3) | −9.6 (−6.9 / −11.9 / −10.0) | −1.6 | 76 / 88 / 100 % | yes (−6.9 vs 3.6) | yes (−6.8 vs 4.7) | −6.7 |

**What three seeds resolve here, and what they do not.** The *drop* is resolved on all three
axes: the corner's seeds and nominal's do not overlap, by a wide margin on pitch. The *split*
is not. On pitch the intrinsic component separates by 9.4 pts against an 11.70-pt bar derived
from the oracle's own nominal spread, so **the direction of the split is supported and its
magnitude is not resolved at three seeds** — 37–55 % is the range this run measured, not a
band it excludes the outside of. Only yaw's intrinsic component clears its bar, and yaw comes
with the caveat in the next paragraph. A fourth seed is the cheap fix and this run does not
have one.

**Read every intrinsic figure against the oracle's own error.** The oracle is not a ceiling
(see the paragraph after next), and on the *nominal* body it trails the frozen model by
**4.4 / 2.0 / 6.7 pts** on roll / pitch / yaw — on nominal seed 2 it even *beats* it on pitch,
87.7 against 87.2. The intrinsic figure at +3 cm is 6.6× that deficit on pitch, which is why
the pitch split is the one this section makes a claim about. On roll (−6.3 against a 4.4-pt
deficit) and yaw (−9.6 against 6.7) it is only about 1.4× the yardstick's own error, so those
two splits are **reported and not claimed**, yaw's resolution notwithstanding. Three
small-drop rows in the table above have to be read against that yardstick as well, and none of
them has a resolved split: base +75 g (intrinsic −0.5, 0.3× the pitch deficit) and CoM z
+0.015 (−2.4, 1.2×) sit at or under it outright, and leg 1.15 (−5.1) is 2.5× it but separates
by seeds on neither its drop nor its split. No split is claimed for any of the three.

**Fall rate is time, not events.** `fall_rate` is the fraction of *ticks* in which
`fallen()` is true, not how often the body tips over. At +3 cm it reads 10.1 ± 2.6 % against
9.7 ± 2.9 % nominal — overlapping seed ranges, unresolved — so this run says the body does not
spend materially more time down, and it says nothing at all about how often it goes down. What
does shift on the means is time in fast motion, 14.0 ± 1.1 % against 11.0 ± 1.0 % — material by 0.07 pt against its 3.0-pt bar and, like fall rate, unresolved at three seeds (seed separation 1.9 pts), so it is reported here and not claimed. Two corners read higher
(CoM fwd/high 16.0 ± 2.4 %, CoM z 16.2 ± 2.0 %) and CoM back/low reads lower (4.4 ± 0.1 %)
while its oracle is 3.4 pts **better** than nominal's — that corner's drop is mismatch, not
difficulty — but none of those three separates from nominal by seeds either.

Per-regime pitch within-0.2 rad at 500 ms, CoM x +0.03 against nominal, mean ± 3-seed spread
(1 500 starts per cell, 1 492 for the fallen cell):

| regime | nominal | CoM x +0.03 | Δ |
|---|---|---|---|
| policy walking | 88.0 ±4.2 % | 54.0 ±16.7 % | −34.0 |
| sine gait | 88.3 ±1.1 % | 50.2 ±6.5 % | −38.1 |
| keyframe/OU | 87.9 ±3.4 % | 52.4 ±7.6 % | −35.5 |
| still | 90.7 ±3.3 % | 46.6 ±11.6 % | −44.1 |
| fast (\|gyro\|>3) | 77.8 ±4.1 % | 50.2 ±6.6 % | −27.6 |
| fallen | 79.1 ±9.9 % | 70.7 ±8.1 % | −8.4 |

The drop is present in every bucket, so it is not an artefact of the fallen bucket growing.
But **the bucket that collapses hardest is `still`, at −44.1 pts — more than `fast`, at
−27.6** — and that ordering deserves to be read against the section's own thesis rather than
past it. A quiet, not-fallen, not-fast bucket losing the most is at least as consistent with a
**static offset**: the +3 cm body rests at a different pitch — the geometry probe above settles
the nominal body at −34.5° and the +3 cm body level, −0.1° — and a frozen model that is simply
wrong about where "at rest" is would fail hardest exactly where the body sits still and the
error has nothing to wash it out. That is model mismatch, not intrinsic difficulty. This table
scores the **frozen model only** and therefore cannot separate the two; the oracle column in
the partition above is where the separation is attempted, and on pitch it is the unresolved
half of the split. The honest statement is that the regime table rules out one alternative
explanation (the fallen bucket) and does not adjudicate this one.

**The oracle is a lower bound, not a ceiling.** It is trained on 14 741 windows for 20 epochs
against the frozen model's 400 k ticks and 60 epochs, and on the *nominal* body it scores
4.4 / 2.0 / 6.7 pts **below** the frozen model on roll / pitch / yaw. A better model trained on
the corner would score higher than it does, so what it measures is a floor on how much of the
drop training can recover, never the most. The split subtracts the nominal baseline, which is
why it measures the *change* in difficulty and the *change* in mismatch rather than either in
absolute terms. `sim2real_proxy.py` builds the same oracle at 100 ms and its docstring now
says the same thing — it used to call this construction "the ceiling", which was wrong in
both files for the same reason.

**What holds.** Mass does not reach the IMU at 500 ms: −20 % / +25 % on the whole body and
±75 g at a fixed centre of mass both stay inside the bar on every axis and never separate by
seeds. Neither does servo gain nor sliding friction. For the mass half of the question that
motivated the sweep — is the twin's prediction sensitive to how much the body weighs — the
published negative survives at the horizon where the real gap lives. Nulls are nulls at this
precision, not proofs of flatness: gain 1.25 carries seed
spreads of 5.6 / 4.8 / 3.2 pts and base −75 g a yaw spread of 3.7, all at or above their own
3.00-pt bar.

**What does not hold.** The **centre of mass** reaches the IMU at 500 ms, and hard. Moving it
3 cm along the body axis alone costs 33.8 pts of pitch forward and 14.2 back, with roll and yaw
material and resolved in both directions but corroborated by RMSE only in the back direction.
The published combined corner CoM fwd/high moves all three axes; CoM back/low moves pitch
alone (−10.6) and CoM z +0.015 roll alone (−5.1), so "material on every axis" is false for two
of the five CoM corners. Leg length 1.15 is material and resolved on pitch (−7.4) and belongs
in this list with them; leg 0.85 reaches the bar on pitch (−4.0) but three seeds do not resolve
it. The published worst-case corners inherit the CoM: worst A moves all three axes, worst B
yaw. So the sentence "body parameters never reach the IMU" was true of the horizon and axes it
was measured on, and is false at 500 ms for the centre of mass and for leg length.

**What it means for a builder.** The cerebellum shipped in this repo was trained at one centre
of mass. Its 100 ms predictions survive every corner tested here. Its 500 ms pitch predictions
do not survive a centre of mass 3 cm further forward or back — and on the held-out half
somewhere between a third and a half of that particular loss (37–55 % across three seeds) is
the body being harder to predict there rather than the model being stale. The oracle that
measures that is a small-data model and therefore a **lower bound**: a better-matched model
recovers *at least* the remainder, and this run does not say where its ceiling is. Nor does
this run resolve the split itself at three seeds — only its direction. Since the real logs
come from one unit, this is a prediction of the twin about other units, not a measurement of
them — the same limit as every DR number in this file.

**Two limits, verbatim.** This is forward-model *prediction* accuracy, not policy *transfer*;
and every real-log number in this repo comes from **one** unit and **one** phone, so this
sweep says what the twin predicts across units, not what a second real robot does.

Reproduce: `.venv/bin/python body_params.py` → `results/body_params.json`,
`results/logs/body_params.txt`. Total 235 s.

## Centre-of-mass identifiability — the method that identifies the servo cannot identify the centre of mass (negative), and a slow servo leaves the delay undetermined even with the right body in the grid

**What was asked.** *Body parameters at 500 ms* found that a 3 cm centre-of-mass shift is what
a builder's phone placement changes and that it costs the frozen model 33.8 pts of pitch. The
day-of-log chain identifies the *servo* from IMU + commands through the frozen model
(`servo_id.py`). Can it identify the centre of mass the same way — and, the question that
matters more, do a CoM shift and a servo delay look alike from the IMU? If they do, the servo
identified on the real log may carry centre-of-mass inside it.

**How a CoM hypothesis is scored** (`com_id.py`). A servo hypothesis transforms the commands,
so `servo_id` scores every candidate through one frozen model. A CoM hypothesis transforms
the *body*, and there is no command-side knob, so a hypothesis **is** a body, represented by
a forward model trained on it: 35 candidates on a dcom_x × dcom_z grid — dcom_x in {−0.045,
−0.030, −0.015, 0, 0.015, 0.030, 0.045} m, dcom_z in {−0.020, −0.010, 0, 0.015, 0.030} m,
one step beyond the published DR endpoints on both axes so that a truth at an endpoint is
interior — each trained identically (100000 ticks collected with seed 1000 on the walk body,
MLP(128), 30 epochs, torch seed 0) and scored by `servo_id.identify`'s normalised one-step
error on the first half of a 30000-tick hidden log. Two things this design publishes
rather than hides: candidate models carry training noise that split-half cannot see, so a
second nominal candidate (collection seed 1001) is scored on every hidden log and the gap
between the two nominal candidates is the method's **noise floor**, printed beside every
band; and the walk body is used, not the olie rows of *Body parameters*, because the confound
question is about the walk-body real log.

**Decision rule, stated before the numbers.** Band = 1.4826·MAD/2 of the A/B quarter errors
over every hypothesis (`servo_id.confidence_band`'s estimator); determined set = the values
within the band of the best error with the other parameter at the argmin; a verdict is
**resolved** when all three hidden seeds (0, 1, 2; 30000 ticks each) agree. Seeds are not
paired — `collect()` draws `sim.rng` every tick for the push test, mode-dependently inside
`Excitation` and once per episode in `fresh()` — so three seeds answer whether a verdict holds
on every seed, not whether a mean is significant. **Null case, asserted:** the nominal hidden
body must identify to (0, 0) on every seed; a failure is recorded, the artifact is written,
and the script exits 1.

### A. Centre of mass alone (ideal servo) — the null case fails on every seed

| hidden body (truth) | argmin (dcom_x, dcom_z) per seed | x set size | z set size | band | noise floor | truth in sets | identified |
|---|---|---|---|---|---|---|---|
| nominal (0, 0) | (−0.015, 0) / (−0.015, −0.010) / (+0.015, 0) | 5 / 3 / 3 | 4 / 1 / 4 | 0.060 / 0.016 / 0.019 | 0.005 / 0.007 / 0.009 | 2 of 3 | **0 of 3** |
| com x −0.03 | (−0.045, 0) × 3, all on the grid edge | 3 / 1 / 3 | 3 / 1 / 3 | 0.043 / 0.017 / 0.033 | 0.009 / 0.002 / 0.002 | 2 of 3 | 0 of 3 |
| com x +0.03 | (+0.030, 0) / (+0.015, +0.015) / (+0.030, +0.015) | 2 / 1 / 1 | 2 / 1 / 1 | 0.010 / 0.011 / 0.013 | 0.009 / 0.002 / 0.016 | 1 of 3 | 0 of 3 |
| com z +0.015 | (−0.015, +0.030) / (0, +0.030) / (+0.015, +0.030), z on the edge | 3 / 2 / 3 | 2 / 2 / 2 | 0.014 / 0.008 / 0.018 | 0.003 / 0.012 / 0.006 | **3 of 3** | 0 of 3 |

**The centre of mass is not identifiable from IMU + commands by this method at this data
scale** — resolved, 0 of 12 body-seeds identified, and the null case fails on all three seeds:
the nominal body reads as ±1.5 cm off on every seed. That is not the candidate models' noise:
the noise floor is 0.2× the band on the nominal body (0.007 against 0.032 mean) and at most
0.8× on any body. It is the score itself — a 1.5 cm step changes the one-step error by
0.003–0.024 (nearest-wrong gaps), which is inside the band on nearly every seed. Split-half
argmins disagree on 11 of 12 body-seeds.

What survives is coarser, and it is marked: **a reading at or beyond 3 cm was produced only by
shifted bodies** — 8 of 9 shifted-body seeds and 0 of 3 null seeds — so the method is a sign
test with a ±1.5 cm false-positive band, not an identification (resolved on com x −0.03 and
com z +0.015, 3 of 3 each; unresolved on com x +0.03, 2 of 3). The truth landed inside the
determined sets on all three seeds only for com z +0.015.

Why the resolution is this poor is *inferred, not resolved*: on the held-out half the
**shipped** model (400000 ticks, 60 epochs, nominal body) beats even the truth-matched
candidate on every seed of every hidden body (one-step error lower by 0.07 / 0.04 / 0.03 /
0.05 on average for the four bodies), so at 100000 ticks and 30 epochs the candidates are
dominated by their data budget, not by the body they were trained on. A candidate grid at the
shipped model's budget would cost ~35× this run's training and is not what this run did.

### B. The confound — joint dcom_x × servo grid (196 hypotheses)

Candidate models along dcom_x (dcom_z = 0) × servo (delay 0–6 ticks, slew {2.0, none},
deadband {0, 2}°) applied to the commands, scored on three hidden bodies.

| hidden body | argmin (dcom_x, delay) per seed | x set | delay set | within band | shape |
|---|---|---|---|---|---|
| com x +0.03, **ideal servo** | (+0.030, 0) × 3 | {+0.015, +0.030} / {+0.030} / {+0.015, +0.030, +0.045} | {0} / {0} / {0, 1} | 4 / 2 / 8 | **point, 3 of 3** |
| nominal CoM, **real-log servo** (delay 5, slew 2.0, 2°) | (−0.015, 5) / (−0.015, 6) / (+0.015, 5) | {−0.015} / {−0.030 … +0.015} / {+0.015} | **whole grid [0 … 6] × 3** | 14 / 56 / 14 | diagonal 1 of 3 |
| com x +0.03, real-log servo | (+0.045, 6) / (+0.030, 5) / (+0.030, 5) | {+0.015 … +0.045} / {+0.030} / {+0.015 … +0.045} | **whole grid [0 … 6] × 3** | 34 / 14 / 32 | diagonal 2 of 3 |

Two readings, one resolved and one not.

**A centre-of-mass shift does not masquerade as a servo delay — resolved, 3 of 3.** With an
ideal servo in the loop, the joint grid recovers dcom_x = +0.030 *and* delay 0 on every seed,
the delay set is {0} or {0, 1}, and the within-band set is a point (2–8 hypotheses, one
delay). The IMU signature of a shifted body is not the signature of a late servo.

**A slow servo leaves the delay undetermined and lets the CoM reading drift — unresolved.**
With the real-log servo candidate in the loop the delay set is the entire grid on all six
seeds, whether or not the body is shifted, and the within-band set grows to 14–56 hypotheses
spanning several dcom_x values and every delay; it is a diagonal on 3 of those 6 seeds and a
point on the other 3, so the trade-off is **not resolved** at three seeds. On the unshifted
body the argmin lands 1.5 cm off on every seed (truth in the x set 1 of 3); on the shifted
body the argmin is right on 2 of 3 and the truth is in the x set on 3 of 3.

**What this does to the servo reading on the real log.** The real walk-1 log (`servo_id`, the
*Real2Sim* section) identified delay 5 with the delay set [2 … 6]. This run reproduces that
shape in the twin: with a hidden delay-5 servo and the same 300 s of walking, the delay set is
the whole grid *even with the correct body model in the grid*, and a 1.5 cm CoM offset is read
into the argmin on 2 of 3 seeds. So the servo identified on the real log **may carry a small
centre-of-mass offset inside it, and this run cannot rule that in or out** — what it does rule
out, 3 of 3, is the reverse: the real log's servo signature is not a disguised CoM shift.

**Two limits, verbatim.** This is forward-model *prediction* accuracy, not policy *transfer*;
and every real-log number in this repo comes from **one** unit and **one** phone, so this run
says what the twin can identify, not what a second real robot does.

Reproduce: `.venv/bin/python com_id.py` → `results/com_id.json`, `results/logs/com_id.txt`
(exits 1 by design: the null case fails). Total 679 s.

## Yaw floor — mostly noise, not model: scaling is flat and privileged state adds little (negative)

The forward model's weakest axis is yaw in the twin itself (58–60 % within 0.2 rad @1 s
against ~83 % roll/pitch). Before treating that as a modelling problem, `yaw_floor.py`
decomposes it: is yaw (a) model-limited, (b) information-limited, or (c) intrinsically
stochastic at this timescale? The decision rule was fixed before the numbers: a gain is
material only above max(3.0 pts, 2× the baseline seed spread) on yaw within 0.2 rad @1 s.
Three MLP init seeds at the standard 400 k / 128×2 configuration spread 2.3 pts
(57.5 / 58.4 / 59.8 %), so the threshold is 4.6 pts.

All data scales are nested prefixes of one 1.6 M-step seed-0 collection whose 400 k prefix
is asserted equal to `data/train.npz`; evaluation is a privileged re-collection of seed 1
asserted equal to `data/test.npz` (2000 shared starts, so every number is comparable with
the forward-model table above).

| condition | yaw @500 ms | yaw @1 s | gain @1 s vs baseline mean 58.6 % |
|---|---|---|---|
| data 0.5× / 1× / 2× / 4× | 76.3 / 77.1 / 78.3 / 77.5 % | 58.1 / 58.4 / 60.9 / 59.4 % | 4×: **+0.9 (not material)** |
| capacity 14 k / 25 k / 50 k / 102 k params | 77.0 / 77.1 / 77.8 / 78.8 % | 59.1 / 58.4 / 60.2 / 59.8 % | 4×: **+1.3 (not material)** |
| privileged probe (diagnostic) | 79.4 % | 61.7 % | **+3.1 (not material)** |

The privileged probe gives the model 18 features a phone IMU cannot see — linear velocity,
full joint state, per-foot contact normal forces, contact count — teacher-forced from the
recorded truth at every rollout step, so it is an upper bound on "the information exists",
not a deployable model. Even that ceiling moves yaw only +3.1 pts, inside what seed noise
can produce. Roll/pitch are flat everywhere (83–85 %). By start regime @1 s, privileged
helps calm starts (59.2 → 63.1 %, n=1715) and does nothing consistent elsewhere
(fast 52.7 → 50.0 %, n=220; fallen 56.9 → 63.1 %, n=65 — too few starts to weigh).

**Verdict (c): the yaw floor is dominated by contact chatter that is aleatoric at 20 ms
resolution** — even knowing the true contact forces barely helps. Consistent with the DR
proxy's R² ≈ 0.2 on yaw gyro, with PETS placing its widest uncertainty exactly here, and
with the 100 ms replan optimum: the planner should not chase this noise. Sim-only, one
collection seed, three MLP seeds; the largest gain anywhere (+3.1, privileged) is the one
to re-test first if more seeds ever tighten the threshold.

## Actuator dynamics — the sim-to-real signature that *is* there, and how to recover it

Second attempt at a proxy, this time perturbing the actuator's **dynamics** rather than the
body's parameters (`sim/growbot_sim.ServoModel`: command latency, slew-rate limit, deadband,
inserted between the command and MuJoCo's ideal PD). Controlled, same three seeds per variant,
within 0.2 rad at 500 ms:

| servo | commanded angle → model | **realized** angle → model |
|---|---|---|
| ideal | 81.1 ± 2.0 | 81.1 ± 2.0 |
| delay 40 ms | 82.2 ± 1.1 | 82.7 ± 0.9 |
| deadband 4° | 80.6 ± 2.9 | 80.8 ± 2.7 |
| **slew 4 rad/s (heavy load)** | **77.1 ± 0.7** | **80.4 ± 0.9** |
| **realistic (2 ticks + 5 rad/s + 2°)** | **78.6 ± 2.4** | **82.5 ± 2.4** |

Latency and deadband the model tolerates; a slew limit opens a 3–4 point gap that a linear
output residual cannot close (its least-squares ceiling is no better than frozen). Giving the
model the **realized horn angle** instead of the command closes it completely (+3.9 ± 0.1) —
the forward model is right, its input is wrong. That is Hwangbo's actuator-net finding.

GrowBot has no servo position feedback, so `servo_id.py` inverts it: propose a servo model,
replay the commands through it, feed the estimate to the frozen forward model, keep the
hypothesis with the lowest one-step error on 300 s of **IMU + commands only**. Two hidden
servos, 252 hypotheses, 16 s of compute per identification: both argmins land exactly on
the hidden values (delay and slew; deadband is the one parameter the IMU cannot see), split
halves agree, and the held-out gap closes to the true-horn-angle value (80.4 → 84.0 % for
the realistic servo, 77.9 → 83.1 % for delay 1 / slew 4). Under the confidence band the
determined sets are wider than the argmin — delay [1 … 3] ticks and slew [4 … 6] rad/s for
the first, [0 … 3] and [3 … 6] for the second — so 300 s of the twin's own walk pins each
parameter to about one grid step either side, not to a point. The forward model doubles as
the position sensor the robot does not have. Sim-only, same-model-class caveat applies; the point is that a real IMU
log of a few minutes is enough data to run this.

`sensor_id.py` asks the symmetric question about the observation side (fusion-filter lag,
gyro noise character, clock jitter), validated by the same hidden-secret round-trip suite (`tests/test_imulog_roundtrips.py`).

Reproduce: `.venv/bin/python actuator_proxy.py` → `results/actuator_proxy.json`,
`results/logs/actuator_proxy.txt`. `.venv/bin/python servo_id.py` (realistic servo: delay 2,
slew 5, deadband 2°) → `results/servo_id.json`, `results/logs/servo_id.txt`;
`.venv/bin/python servo_id.py --true-delay 1 --true-slew 4.0 --true-deadband-deg 1` →
`results/logs/servo_id_slew4.txt` (the JSON holds whichever ran last; the shipped one is the
realistic servo). About 1 min each.

## Model mismatch — wrong-family identification recovers ~90 % of the gap; split-half catches drift, not shape

`servo_id.py` searches a (delay, slew, deadband) family. A real servo under load is not in
that family, so `model_mismatch.py` hides two out-of-family servos in the twin and runs the
identical identification against them: a **load-dependent slew** (8 / (1 + 2·|err|) rad/s —
the effective limit spans ~2.7–8 across the operating range, so no single grid point
reproduces it) and a **voltage sag** (slew drifting 6 → 3 rad/s over the session, the age
counter surviving episode resets). The in-family servo (delay 2, slew 5) runs through the
same pipeline as control. One seed (777), 600 s logs, identification on the first half,
every number below from the second half only; within 0.2 rad, n = 1500 starts per cell.

| held-out, within 0.2 rad | horizon | commanded | identified | + linear residual | true horn (floor) |
|---|---|---|---|---|---|
| in-family control | 100 ms | 94.8 | 95.5 | 95.5 | 95.5 |
|  | 500 ms | 80.4 | 84.0 | 83.7 | 84.0 |
| load-dependent slew | 100 ms | 94.1 | 95.8 | 95.8 | 95.7 |
|  | 500 ms | 75.9 | 82.5 | 82.4 | 83.3 |
| voltage sag | 100 ms | 93.8 | 95.3 | 95.0 | 95.4 |
|  | 500 ms | 74.9 | 81.6 | 81.5 | 82.0 |

Three findings:

- **Graceful degradation.** The wrong-family point approximation (load-dependent identifies
  as slew 4.0, sag as 5.0, both with the true delay 2) recovers 90 % and 94 % of the closable
  commanded → true-horn gap at 500 ms. Being outside the family cost 0.8 and 0.4 points
  against the oracle floor — the grid's nearest point is a good servo model even when no grid
  point is the servo.
- **The honesty diagnostics split by time, so they catch drift, not shape.** Voltage sag
  fires split-half DISAGREE (A: slew 6, B: slew 5) — non-stationarity is caught, exactly the
  battery signature a real fresh-vs-low session would show. Load-dependent slew passes AGREE
  on the same wrong point in both halves: a time-stationary mismatch presents identical
  statistics to both halves, so a time-split diagnostic cannot see it, and nothing else in
  the report flags it. The saving grace is the previous finding: the undetected wrong answer
  was also the nearly harmless one.
- **Residual on top of identification — negative.** A linear residual (ridge least squares
  from the model's input window to its one-step error, fit on the identification half only)
  moves the held-out numbers by −0.1 to −0.3 points everywhere. After identification at most
  0.8 points remained to close, and the residual closed none of them. At these mismatch
  magnitudes the fallback is unnecessary; whether it earns its place under larger mismatch
  is untested.

Sim-only, single seed, and both mismatch shapes are guesses at what a loaded MG90S does —
a real log decides whether they were the right guesses.

Reproduce: `.venv/bin/python model_mismatch.py` → `results/model_mismatch.json`,
`results/logs/model_mismatch.txt`.

## The first real logs — read per file, per segment

Two `?imulog=1` sessions from the upstream app, `growbot-imulog-1` format: walk-1,
16.2 s, `end_why: done`, agent gain null; walk-3, 5.2 s, `end_why: tipped`, agent gain
0.8. `real_log_report.py` runs the whole day-of-log chain on them.

**They are not two samples of one experiment, and are never pooled.** The agent gains
differ, so the commands that reached the two bodies are scaled differently. The resting
attitudes differ by 43° — walk-1's body rests at pitch −0.74 rad, walk-3's flat at
+0.01 — so the phone is not in the same place and every attitude-referenced number means
something different in each. `gap_report.py` now refuses to concatenate files that
disagree on either. An earlier version of this report published aggregates over the pair;
they are withdrawn.

**What the segmenter finds.** `growbot-imulog-1` carries no event rows, so `growbot_cerebellum/imulog.py`
synthesizes regimes from the data (rolling stillness by `sensor_id.verify_still`'s own
thresholds, command activity, acceleration spikes at stillness boundaries, and a fall
defined as an excursion from *the file's own* rest attitude that never returns). Before
this, every tick of both files inherited `header.gait` = "official" and was scored
against the twin's **walking** floor.

| file | segments |
|---|---|
| walk-1 | still 0.00–1.15 s, **walking** 1.15–16.2 s |
| walk-3 | still 0.00–2.60 s, impact 2.60–3.02 s, **fall** 3.02–5.2 s |

walk-3 contains **no walking at all**. Its first 2.6 s are a body frozen byte-identical
in orientation (gyro RMS 0.003 rad/s) while the commands swing ±34° — preflight now warns
about exactly that — and its last 2.2 s are the fall. A 5 g event at t=73044 ms separates
them.

**walk-1, gap per segment** (within 0.2 rad; twin floor is the regime each segment maps
to; `n` is rollout starts):

| walk-1 @500 ms | twin floor | n | roll real/twin/gap | pitch | yaw |
|---|---|---|---|---|---|
| still | still | 53 | 84.9 / 90.1 / −5.1 | 86.8 / 89.0 / −2.2 | 77.4 / 80.1 / −2.7 |
| walking | policy | 725 | 50.8 / 87.7 / **−36.9** | 50.1 / 87.2 / **−37.1** | 33.7 / 76.7 / **−43.0** |

At 100 ms the same rows read −0.4 to +2.2 — the model is right about the next 100 ms of a
real robot and wrong about the next 500. Roll and pitch sit at nearly the same distance
below their floors, which is not the signature of a pitch-specific problem.

**walk-3, per segment.** Reported separately and never mixed into walk-1:

| walk-3 @500 ms | twin floor | n | roll real/twin/gap | pitch | yaw |
|---|---|---|---|---|---|
| still | still | 125 | 91.2 / 90.1 / +1.1 | 4.8 / 89.0 / **−84.2** | 67.2 / 80.1 / −12.9 |
| impact | (none) | 21 | not reported | not reported | not reported |
| fall | fallen | 84 | 57.1 / 72.4 / −15.2 | 40.5 / 67.2 / −26.7 | 61.9 / 66.1 / −4.2 |

(`impact` has 21 ticks, below the 30 rollout starts `evaluate_axes` requires, so it is
left out rather than quoted on 21 samples. The `fallen` twin floor is new: the twin's
excitation labels say what the *excitation* was doing, never what the body was doing, so
there was no twin regime to compare a real fall against. It is defined by the same
excursion-from-rest measure and threshold the real fall is detected with.)

The −84.2 on a *motionless* segment is not a model failure. The model is fed commands
reaching 34° of horn swing, predicts the motion those commands imply, and the body
produced none. That number measures the robot — phone off the body, body off the ground,
or servos not moving — and it is the most useful thing walk-3 contains.

**Servo identification, walk-1 alone** (fit on its first half, evaluated on its held-out
second half, 252 hypotheses on a grid extended until the argmin is interior): argmin
delay 5 ticks (100 ms), slew 2.0 rad/s, deadband 2°; split-half **DISAGREE**
(A: 6 / 5.0, B: 6 / 2.0). The determined sets are delay **[2, 3, 4, 5, 6] ticks** (5 of the
grid's 7 values, 40–120 ms) and slew **[2.0, 3.0] rad/s**. Eight seconds of periodic
walking is the excitation `servo_id.py` warns about: the argmin is not an identification
at that width, but the set is no longer the whole grid.

> These two sets are quoted from `results/real_log_report.json` (`servo.delay_determined_set`,
> `servo.slew_determined_set`). An earlier revision of this section published delay
> **[0 … 6] — the entire grid** — and slew **[1.5, 2.0, 3.0, 4.0]**, which were the sets
> before `confidence_band` moved from a standard deviation to 1.4826·MAD. That change
> narrowed both sets and nothing downstream noticed, because `real2sim.py` held a
> hand-copied mirror of them. The copy is gone (`real2sim.determined_band` reads the
> artifact and fails hard if it cannot), and every number below that depends on the sets
> is recomputed against them.
> *Caveat added later (`com_id.py`): in the twin, a hidden delay-5 servo with the same 300 s of
> walking leaves the delay set at the whole grid even with the correct body model in the grid, and
> a 1.5 cm centre-of-mass offset is read into the argmin on 2 of 3 seeds — the identified servo may
> carry a small CoM offset inside it; unresolved. The reverse is ruled out 3 of 3: a CoM shift does
> not read as a servo delay. See *Centre-of-mass identifiability*.*

**Fusion-filter lag, walk-1 only**: wx +13.0 ± 0.1 ms (corr 0.87), wy +13.8 ± 0.2 (0.95),
wz +12.8 ± 0.7 (0.96), split-half AGREE on all three. walk-3 gives +17.9 ± 10.1 / +20.7 ±
10.5 / +19.5 ms with split-half **DISAGREE** on every axis — 5 s, half of it motionless,
is not enough, and it is reported as undetermined rather than averaged in.

**Is `header.gain` already inside the logged commands?** The adapter asserted it was.
The two files are the experiment: same body, same cal, same gait, agent gain null vs 0.8.
If the gain were applied downstream of the log both files would show the same command
amplitude; if it is already inside, walk-3's are 0.8× walk-1's. Statistic: p95 of
|command − 90|, l and r pooled; percentile bootstrap, 4000 resamples.

| | p95 |command − 90| | n |
|---|---|---|
| walk-1 (gain null) | 51.00° | 734 |
| walk-3 (gain 0.8) | 41.46° | 198 |
| **ratio** | **1.230, 95 % CI [1.171, 1.281]** | |

1/0.8 = 1.25 is inside the interval; 1.00 is far outside. **The gain is baked into the
logged values**, so only `cal.gain` takes part in the inversion — now measured rather
than assumed.

**Allan / gyro noise: still undetermined, but for a stated reason.** A still segment
*does* exist in each file — 1.1 s in walk-1, 2.6 s in walk-3 — and both are far under
what Allan deviation needs (tens of taus with many independent clusters each: minutes at
60 Hz, not seconds). The data ask stands, and now says the right thing: the still
segments are **too short**, not absent.

## Real2Sim loop closure — an actuator model helps on the real walk; *which* actuator model is not identified

> **This section replaces an earlier one.** The previous version scored a
> concatenation of walk-1 and walk-3 and concluded the loop was "validated robustly to
> the identification uncertainty". Both halves of that are withdrawn. Roughly half the
> old held-out slice was walk-3 — 2.6 s of a motionless body under swinging commands,
> then a fall — so a large part of what the corrected twins were credited with
> predicting was a robot that was not moving. And "robustly to the identification
> uncertainty" was inferred from three sampled points out of a seven-wide determined
> delay set, which is a claim about a band made from a sample of it. Every number below
> is new: walk-1 only, and the verdict text is computed from which cells pass and from
> how much of the band they cover.
>
> **The band numbers in this section were corrected again.** They were computed against
> `real2sim.py`'s hand-copied mirror of the determined sets, which had gone stale against
> `results/real_log_report.json`. The coverage figures move 43 % → **40 %** on delay and
> 25 % → **50 %** on slew, and the smoothing-only cell moves from inside the determined
> band to outside it. The measured percentages in the table are unchanged — they never
> depended on the sets — but two of the readings did, and both are rewritten below.

`servo_id.py` on walk-1 alone leaves the servo at delay **[2, 3, 4, 5, 6] ticks** (40–120
ms, 5 of the grid's 7 values) and slew **[2.0, 3.0] rad/s**, with
split-half DISAGREE (A: delay 6 / slew 5.0, B: delay 6 / slew 2.0). `real2sim.py` runs
the whole loop at four points plus a nominal control through the identical pipeline:
collect 400 k ticks with the servo inside the twin (seed 0), train the standard model
(128×2, K=5, 80 epochs, seed 0), evaluate on walk-1's held-out half. The corrected twins
train on **commanded** actions — the horn lags inside the twin — because commanded
angles are all a real log carries.

The fourth point is the one the previous design lacked. A "corrected" config changes
delay, slew *and* deadband at once against a control that has none of them, so its gain
says "some actuator model helps", not "this one is right". The **smoothing-only** cell
(delay 0, slew 2.0, deadband 2°) carries no latency at all, so it still separates
identified dynamics from plain action smoothing.

What it no longer does is sit **inside** the determined band. Delay 0 is not in
[2, 3, 4, 5, 6], so this cell is an action-smoothing *control*, not a rival hypothesis
about the same servo. The difference matters for exactly one reading: a tie between it
and a delayed cell used to mean "the identified servo could have been its own zero-delay
member", and it now means only "smoothing alone gets you as far here". (`half-A`, at slew
5.0, is outside the band too — 5.0 is not in [2.0, 3.0].) Whether each cell is in band is
now computed per cell by `real2sim.in_band` and published in
`results/real2sim.json → band_coverage.cells_in_band`, rather than asserted in prose.

Discipline: identification used the first half of walk-1, so every real-log number is
its held-out second half only (405 ticks, 8.1 s — n is small and quoted). The decision
rule preceded the numbers: closure on an axis at 500 ms is material when it beats
max(3.0 pts, 2× the control's spread across 3 MLP seeds). The control collection is
asserted array-equal to `data/train.npz`.

| walk-1 held-out, within 0.2 rad @500 ms | roll | pitch | yaw |
|---|---|---|---|
| control (nominal servo) | 43.0 | 51.3 | 33.2 |
| argmin — delay 100 ms, slew 2.0 | 59.1 **(+16.0)** | 54.3 (+2.9) | 45.7 **(+12.6)** |
| half-A — delay 120 ms, slew 5.0 | 47.6 **(+4.5)** | 50.0 (−1.3) | 36.6 (+3.5) |
| half-B — delay 120 ms, slew 2.0 | 76.5 **(+33.4)** | 58.6 **(+7.2)** | 45.5 **(+12.3)** |
| **smoothing only — delay 0, slew 2.0** | 54.0 **(+11.0)** | 52.1 (+0.8) | 43.0 (+9.9) |
| materiality threshold (2× seed spread) | 3.0 | 3.0 | 10.7 |

The yaw threshold is 10.7 pts because the control's own spread across three MLP seeds is
5.3 pts on these 405 ticks. That is the honest cost of a small held-out slice, and it is
why yaw's +9.9 and +12.6 are not treated as different from each other.

Read carefully, this is a weaker and more specific result than the one it replaces:

- **Roll closes at every tested point**, and among the configs the identification
  genuinely **cannot** tell apart the spread is **+16.0 to +33.4 pts** — a 17.4-point
  swing between `argmin` and `half-B`, the only two cells inside both determined sets.
  The choice of servo inside the band matters enormously, and the band is not narrowed.
  (This bullet used to quote the full four-point range, +4.5 to +33.4, as that swing. It
  is not: the +4.5 end is `half-A` at slew 5.0, which the slew set [2.0, 3.0] excludes —
  as this section states above and `band_coverage.cells_in_band` records — so the
  identification does tell that cell apart. The four-point range is still the spread
  across everything tested; it is not a statement about the band.)
- **The tested configs visit 40 % of the determined delay set and 50 % of the slew set.**
  Nothing here supports "robust to the identification uncertainty"; that phrase is
  retracted. Only two of the four corrected cells (`argmin`, `half-B`) are inside both
  determined sets at all.
- **The smoothing-only cell splits the axes, and the split is the finding — with a
  weaker reading on yaw than this section used to give it.** On **yaw** it closes +9.9
  against the best delayed cell's +12.6 — a 2.7-pt difference, inside the 10.7-pt
  threshold — so *this log does not separate "the identified dynamics" from "any action
  smoothing"* on yaw. That much stands: it is a statement about two measured gains and
  the threshold, and none of the three numbers depends on the determined sets. What does
  not stand is the stronger gloss the cell used to carry. Because delay 0 is **outside**
  the determined set, the yaw tie no longer says "the servo could just as well be the
  zero-delay member of its own band"; it says only that on yaw, 8.1 s of held-out walking
  cannot tell a zero-delay smoother from the identified servo. On **roll** the best
  delayed cell beats it by 22.4 pts and on **pitch** by 6.4, both above threshold, so
  there the latency is carrying something smoothing alone does not.
- **Delay is not identified to a point on this log** — the determined set is
  [2, 3, 4, 5, 6] ticks, 5 of the grid's 7 values, spanning 40–120 ms — so none of the
  above says the servo's real latency is any particular number. (This bullet previously
  read "the determined set is the whole grid", which was the stale copy.)
- **Deadband is never varied on its own**, so its contribution is untested. The cells
  are not a factorial.
- **The identified servo may carry a small centre-of-mass offset inside it — unresolved.** In the
  twin (`com_id.py`), a hidden delay-5 servo with the same 300 s of walking leaves the delay set at
  the whole grid even with the correct body model available, and a 1.5 cm CoM offset is read into
  the joint argmin on 2 of 3 seeds. The reverse is ruled out 3 of 3: a CoM shift with an ideal servo
  recovers delay 0. So the closure numbers above stand as measured; what they attribute to *the
  servo* could include a little body, and this run cannot separate the two at this data scale.

Pitch is no longer read as a coverage hole — see the retraction below. On walk-1, the
only file that walks, pitch and roll sit at similar distances below their twin floors,
which is not the signature a pitch-specific missing-motion hole would leave.

Context, not caveat: on the slower servos the policy's realized gait shrinks (horn
amplitude 0.39 → 0.22–0.27 rad, mean gyro 1.70 → 1.00–1.22 rad/s, falls roughly
unchanged) — the same policy driven through the identified servo walks noticeably
more gently than the twin pretends, which is itself an argument for retraining the
walk policy against the corrected twin upstream. It is also a confound worth naming:
the corrected twins move 40–45 % less than the control, so part of any gain may be a
gentler training distribution rather than a better actuator model.

Reproduce: `.venv/bin/python real2sim.py` → `results/real2sim.json`, `results/logs/real2sim.txt`.
Needs the untracked `imu-walk-1-*.json` and `results/real_log_report.json`, which
`.venv/bin/python real_log_report.py imu-walk-1-*.json imu-walk-3-*.json` writes; the
determined band is read from that file, so the order is `real_log_report` → `real2sim` →
`coverage` (the retracted 2×2 reads `real2sim.json` for its thresholds).

## Identification ablation — per-side gains the most and proves the least, multi-horizon backfires, and the delay was over-charged by a tick

`identification_ablation.py`, on `imu-walk-1` (809 ticks, 16.2 s, the only real file that
walks). Four changes to how the servo is identified, each from a specific reading of the
code or the literature, each measured the way `servo_id` already asks to be measured: fit
on the first half, every number from the held-out second half, and the determined set
reported rather than the argmin. There is no ground truth on a real log, so nothing here
is scored against a known answer — the question is whether the identified servo predicts
held-out data better than the raw commands, and whether the log separates the parameters
well enough for the answer to mean anything.

The decision the table is read against was fixed before running: a change earns its place
if it improves the held-out gain, narrows a determined set, or corrects an attribution —
and a change that does none of those is reported as a negative at the same size.

| variant | identified | delay set | held-out gain @500 ms (roll / pitch / yaw, pts) |
|---|---|---|---|
| baseline (one-step, shared) | delay 5, slew 2.0 | [2, 3, 4, 5, 6] | **+3.5** / −1.3 / **+7.8** |
| + aligned observations | delay 4, slew 2.0 | [3, 4, 5, 6] | +3.5 / −1.1 / +7.2 |
| + multi-horizon score | delay 1, slew 1.0 ⚠ | [0 … 6] | **−2.4** / −0.3 / +10.4 |
| + per-side servos | L(6, 6.0) ⚠ R(6, 1.0) ⚠ | [2, 3, 4, 5, 6] | **+8.0** / +0.3 / +8.0 |
| + all three | L(5, 4.0) ⚠ R(6, 1.0) ⚠ | [0 … 6] | +6.4 / −1.3 / +8.3 |

⚠ marks an argmin sitting on the grid's **boundary** — by this repo's own rule
(`servo_id.argmin_interior`) that is the search running out, not an identification. In
the `+ per-side` row both argmins are at delay 6 = max(grid delays) and the slower horn
at slew 1.0 = min(grid slews). The shared argmins in rows 1–2 are interior; nothing else
in this table is. Boundary status is checked per side and published in
`results/identification_ablation.json → per_side_solution.left/right_argmin_interior`.

The rule counts **all three** searched axes, deadband included, and `+ all three`'s left
horn is the reason that had to be said out loud: it carries an interior delay (5) and an
interior slew (4.0) but sits at deadband 0 = min(grid deadbands), and the check —
reading only delay and slew — published it as INTERIOR and left it unmarked here. The
grid is a product of the three, so an argmin pinned on the deadband axis is the search
running out exactly as it is on the other two. Fixed and re-run: that horn now reads
`left_argmin_interior: false`, prints with a `!` in
`results/logs/identification_ablation.txt`, and carries its ⚠ in the table above.
Nothing else moved — no error, no gain, no determined set.

> **The per-side L/R labels in this table were inverted before this revision**, and every
> published left/right attribution with them. `servo_id.realized_per_side` put the *left*
> triple on action column 0, but column 0 is the **right** leg: `imulog.parse` stacks
> `np.stack([a_right, a_left], 1)`, and the twin agrees (`a = np.tanh(x[:2])  # [aRight,
> aLeft]`, `joint_1 is right_leg`). The error was invisible to every metric — swapping two
> labels changes no error, no gain and no determined set, only who gets the credit — so
> the fix changes no number in this table, only which horn each triple belongs to. The
> slow horn is the **right** one, not the left. A regression guard now runs an
> asymmetric fixture (one horn deliberately crippled) through the identification and
> asserts the slow triple comes back on the side it was injected on; a symmetric fixture
> cannot catch a label swap, which is why the old round-trip passed throughout.
>
> **What that guard proves, precisely.** Its first version proved only self-consistency:
> it injected the crippled horn through `servo_id.RIGHT_COL / LEFT_COL` and then read the
> answer's label off the same two constants, so setting them to `1, 0` moved the
> injection along with the label and the guard stayed green while every published
> attribution inverted. It is now bound to ground truth instead, two ways: the constants
> are asserted against the twin's own XML, read independently by
> `servo_id.sim_side_columns` (actuator `servo_1` → `joint_1` → body `right_leg` ⇒ action
> column 0 is the right leg), and the crippled horn is injected **by action column** on
> the column that XML names, not on whatever column the constants currently name.
> Verified by reversing the pair: with `RIGHT_COL, LEFT_COL = 1, 0` the suite exits 1 on
> the convention assert, and with that assert bypassed it exits 1 again on the
> attribution assert (`identified: L(delay 6, slew 1.5) R(delay 0, slew None)` — the slow
> horn injected on the right, handed back as the left). Restored, it passes.

Gains are (identified servo − raw commands) on the same data, which is what makes the
column comparable across rows: the aligned variants are scored on aligned observations, so
their absolute percentages are not directly comparable to the unaligned ones, but the gain
is. Every variant splits DISAGREE across the two fit halves. None of these changes fixes
that, and none was expected to: 16 s of periodic walking is the excitation problem, not the
scoring problem.

**The sensor lag was being charged to the servo, and alignment gives it back.** The phone's
fused orientation trails its own gyro by 13.2 ms on this file, measured independently by
`sensor_id.filter_lag` and stable across halves. Observation delay and command delay are
formally interchangeable, so a delay identified from commands and IMU is necessarily a lump
of both. Advancing the angle channels to meet the gyro (`imulog.parse(ang_lead_ms=...)`, a
sub-tick correction of at most 0.099 rad on this log) moves the identified delay from 5
ticks to 4, and narrows the delay determined set from [2, 3, 4, 5, 6] to [3, 4, 5, 6]. The
held-out gain does not move (roll +3.5 either way). That is the honest shape of this
result: alignment does not predict better, it stops the servo being billed for the phone.
What remains after alignment is the actuator plus whatever absolute lag the gyro itself
carries, which this log cannot measure — so 80 ms is an upper bound on the actuator, not a
split.

**Per-side has the largest held-out gain, and the weakest claim to it of anything in this
table.** Fitting one (delay, slew, deadband) per servo instead of one for both more than
doubles the held-out roll gain, +3.5 → +8.0 pts, and helps at 100 ms too (+0.5 → +1.9).
That gain is real: it is measured on the held-out half and it reproduces from
`results/identification_ablation.json`. The search is coordinate descent from the shared
solution — the full product of two triples is ~63k hypotheses and brute force is not the
point — converging in 1009 evaluations, on L(delay 6, slew 6.0) and R(delay 6, slew 1.0):
the **right** horn is the slower one.

Three caveats, and none of them is a footnote:

- **The fit improvement is not separated from the shared fit.** Per-side beats the shared
  fit by **0.0064** on a confidence band of **0.0063** — a ratio of 1.01. By the criterion
  this repo applies to every other number, a separation that small is not one, and the
  script now reports it as `MARGINAL` rather than as a boolean — in the artifact as well
  as in the prose. That distinction had to be repaired once: the printed line branched on
  marginal first, but the JSON emitted `separated: true` beside `marginal: true`, so a
  consumer reading `fit_gain_vs_band.separated` got back exactly the boolean this bullet
  retracts. `separated` now means *clear of* the band and is false here; the single field
  to read is `fit_gain_vs_band.verdict` (`separated` / `marginal` / `not_separated` /
  `band_zero`). So "per-side fits better"
  and "per-side is the only change that improved prediction" are two different claims: the
  *held-out* gain is +8.0 pts and stands, while the *fit* improvement that motivates the
  extra three parameters sits on the noise floor. The earlier version of this section
  asserted the second while quoting the first.
- **The disjointness of the two slew sets is not independent evidence.** Those sets —
  [3.0 … 8.0, none] on the left against [1.0, 1.5, 2.0] on the right — are
  one-dimensional *conditional slices*: each is swept with the partner frozen at its own
  coordinate-descent optimum, each is centred on its own argmin, and both are cut with a
  band computed from the **shared** sweeps. Saying they are disjoint therefore restates
  "the two argmins differ by more than the band". It is a restatement of the argmins, not
  a second measurement confirming them, and this document previously read it as the
  latter. The JSON keys were renamed `left/right_slew_conditional` (from
  `..._determined`) so the artifact cannot be quoted as a determined set either. The
  delay slices are wide on both horns and overlap completely — [4, 5, 6] on the left,
  [1 … 6] on the right — so no delay asymmetry is claimed in either direction.
- **Both per-side argmins sit on the grid boundary** (delay 6 = max, slew 1.0 = min), so
  by `servo_id`'s own rule the per-side search ran out rather than identified.

What *does* survive is the direction, and on a test it did not have before. The per-side
fit is now re-run independently on each fit half, and both halves put the slower horn on
the **right** (`per_side_solution.split_half.slower_agree = true`), even though their
argmins disagree and every variant in the table still splits DISAGREE on the shared fit.
That is the honest form of the claim: *which* horn is slower reproduces across halves;
*how much* slower does not, is not pinned by the conditional slices, and rests on argmins
at the edge of the grid.

**And that flag carries its own noise floor, which is low.** With the fit separation
retracted as `MARGINAL`, the boundary argmins reported and the disjointness withdrawn,
`slower_agree` is the *only* support left for "the right horn is slower" — so it has to
be quoted with what it is worth. It is two halves each landing on one of three outcomes
{left, right, neither}; under a null with no real asymmetry, and ties rare on this grid,
they agree on the same non-`neither` side roughly **1 time in 2**. That is a coin flip —
the same standard this very script applies to reject a gain/band ratio of 1.01 — so the
flag is about one bit of evidence, not a confirmation. It is now published with that
floor beside it (`split_half.slower_agree_null_p`, `split_half.slower_agree_note`, and a
`noise floor` line in `results/logs/identification_ablation.txt`). The direction survives
as the *least weak* thing here, not as a demonstrated asymmetry.

**Multi-horizon scoring backfired here — a clean negative against the literature's
expectation.** Replacing the one-step error with clip rollouts (uniformly sampled 2–40 tick
horizons, 400 starts) sends the argmin to delay 1 and slew 1.0, which is the grid's *low
boundary*; the delay determined set widens to the entire grid, i.e. it determines nothing;
the confidence band explodes 0.0063 → 0.0871; and the held-out roll gain goes negative.
Yaw improves (+10.4) but on its own that is not enough to accept a variant that determines
nothing else. The likely reason is scale rather than principle: with ~400 ticks to fit and
clips reaching 800 ms, open-loop divergence dominates the score instead of the servo
signature, where the published horizon ablations that motivated this ran on far more data.
On this log the one-step score is the better instrument, and the combined variant inherits
the damage (band 0.081, sets spanning the whole grid) while keeping most of per-side's
roll gain.

Two defects surfaced while running this, both fixed and both in the same family — a
statistic that looked stable only because nothing had stressed it:

- **The confidence band depended on which hypotheses were enumerated.** It was a standard
  deviation over the A/B error differences of every hypothesis, so adding slow-slew
  candidates — added precisely in order to rule them out — inflated it through their large,
  noisy errors. On identical fixture data with an identical argmin, widening the grid from
  96 to 252 hypotheses moved the band 0.00141 → 0.00457 and the delay determined set
  [1, 2] → [0, 1, 2, 3], admitting delay 0, i.e. "no servo at all". The band is now a robust
  scale (1.4826 · MAD), which is a no-op where there is no tail (0.00140 vs 0.00141 on the
  old grid) and holds the set at [1, 2, 3] on the wide one.
- **`determined_sets` crashed when it had to report "no slew limit" beside a number** —
  a plain `sorted()` over a set containing `None`. It had never fired because that answer
  had never survived into a set, which is to say the crash was waiting for exactly the
  under-determined case the function exists to report.

Reproduce: `.venv/bin/python identification_ablation.py imu-walk-1-*.json` →
`results/identification_ablation.json`, `results/logs/identification_ablation.txt`. The
channel-alignment variant reads the measured fusion lag from `results/sensor_id_<stem>.json`,
so `.venv/bin/python sensor_id.py imu-walk-1-*.json` runs first.

## The gesture and still captures — the still lane pays out, the gesture lane cannot

Two captures were requested to break the identification deadlock: `~15 s` of periodic
walking had left the servo delay undetermined over the whole grid, and no still segment
long enough for an Allan read existed in the walk lane. The maintainer sent a 3.6 min
gesture session (`gait: "act"`, 53 pose rows over 216 s, 10,872 ticks at 50 Hz) and a
75.7 s still capture (`gait: "still"`, 4,471 IMU rows, **an empty pose array**), with
the warning that in act
files "the pose rows are the commanded keyframe schedule (send time plus cumulative ms
offsets), not the 30Hz stream".

### The still lane: the phone's gyro noise, measured

`sensor_id.py SEND-still-76s.json`. The record is genuinely still: one 75.7 s still
segment, and over the 66 s window the Allan read uses, gyro RMS **0.0034 rad/s**
against the 0.15 threshold and a largest per-axis roll/pitch standard deviation of
**0.0014 rad** (0.08°) against 0.05.

One dropout — the session's largest IMU gap, **1051 ms**, which is why the analysed run
starts at t = 9.7 s rather than at 0 — sits inside that segment, and it mattered more
than its size suggests. Allan integrates the rate into an angle assuming a single
uniform sample period, so ~61 missing samples do not blur the curve, they insert a step
the sensor never produced and bias every tau above the gap. The estimator therefore
splits the still windows at dropouts and reads the longest gap-free run:

| | ARW (rad/s/√Hz) | fitted log-log slope | last 0.5 s trimmed | last 1.0 s trimmed |
|---|---|---|---|---|
| wx | **6.42e-04** | −0.624 | **undetermined** (−0.650) | **undetermined** (−0.678) |
| wy | **3.05e-04** | −0.554 | 2.88e-04 (−0.494) | 2.86e-04 (−0.505) |
| wz | **1.26e-04** | −0.625 | 1.26e-04 (−0.635) | 1.25e-04 (−0.645) |

Conditions: 66 s gap-free run, 3,956 samples, fs 59.88 Hz, gyro RMS 0.0034 rad/s; the
law is read over tau 0.33–3.0 s and a slope is accepted in [−0.65, −0.35].

**The slope is part of the number, so it is published with it.** The estimator reads the
ARW off an *assumed* −1/2 law and then reports the intercept, discarding the slope it
measured — so a value quoted alone hides how far the curve was from the law it was read
with. `wx` fits −0.624 — 83 % of the way from the assumed −1/2 to the edge of the
acceptance window — and its curve *rises* from
adev 2.53e-03 at tau 0.0167 s to 2.76e-03 at 0.0501 s, which is not what white noise
does. `wy` is the only axis near the middle of the window at −0.554.

**Correction — where the peak actually is.** This section previously said the 7.1 deg/s
peak was the taps at 0–2.2 s and 71.7–73 s, "not a disturbance in the body of the
capture". That was wrong about which samples entered the fit: 7.1 deg/s is the **last
sample of the Allan segment** (t = 75.70 s), the tap that ends the recording, and the
segment's own peak is therefore the session's. Whether to exclude it is now measured
rather than argued, at two stated trims: dropping the last 0.5 s (30 samples) moves
`wy` by −6 % and `wz` by −0.3 %, and takes `wx`'s slope to −0.650, where the gate
rejects it. So **wx's ARW does not survive a half-second trim** and is quoted as
conditional on those samples. The published values are the untrimmed ones because the
segment rule is mechanical — longest gap-free still run — and a hand-chosen cut is a
free parameter; the trims are recorded beside them in
`results/sensor_id_SEND-still-76s.json` (`allan_tail_trim`) so the reader is not asked
to take the choice on trust.

**Bias instability stays undetermined on all three axes**, for two different reasons the
artifact records separately: on `wx` and `wy` the minimum sits at the edge of the tau
range (~17 s), i.e. 76 s cannot show a flicker floor if this sensor has one; on `wz` the
minimum is interior, at tau 6.8 s, and is refused as a 1.4×-wide notch rather than a
plateau (the gate needs 5× within 10 %) — an ARW / rate-random-walk crossover, not a
floor.

The fusion-filter lag could **not** be replicated from this file, and could not have
been: peak correlations of 0.17 / 0.29 / 0.13 are far below the 0.50 gate, because a
motionless body gives the cross-correlation nothing to lock onto. The walk lane's
+13.0 / +13.8 / +12.8 ms stands unreplicated rather than contradicted — measuring it
needs motion, which is exactly what a still capture excludes.

### The gesture lane: the file determines nothing, and the reason is not established

> **Retraction.** This section previously explained the gesture lane's failure: the
> `act` verb glides at **1.00 rad/s**, derived from the header's documented example
> `{l:130, r:50, ms:700}`; that ramp is slower than the horn's own slew, so "the horn is
> never asked to move at its own limit" and the identified slew "is a measurement of the
> glide engine", which "the identification confirms exactly". Every step of that is
> withdrawn. The derivation, the confirmation and the conclusion each fail on their own,
> and they fail in a way this repository has already documented once: the header field
> they rest on is `post_walk`, the same field the coverage retraction below is about.
> The measured numbers are preserved and regenerated in
> `results/gesture_id_SEND-gesture-3_6min.json`; what is deleted is what was inferred
> from them.
>
> **The example is a target pose, not a move.** `{l:130, r:50}` is an absolute pose pair
> — 90+40 and 90−40 — so it is "a 40° move" only from neutral, and the header states no
> start pose. Read through the parser's own calibration inversion it is 40.40° of horn
> travel from neutral, not 40°, because the derivation dropped `cal.gain` (0.99) that
> every other conversion in this repository applies. The rate that follows from it is
> **1.0074 rad/s**, not the 0.9973 published in the artifact.
>
> **The example is not from this session.** `post_walk` documents the sit fold that
> happens **after recording ends**, on walks that end `done`. This is the second
> conclusion in this repository built on that field, and the second to be withdrawn for
> the same reason: the act it documents is not in the record it was read into.
>
> **The confirmation was a grid artifact.** 1.0 rad/s is `min(slews)` in
> `servo_id.default_grid()`. The gesture argmin sits on the grid boundary on all three
> axes (`argmin_interior: false`) and its slew determined set is the *entire* grid,
> "no slew limit" included. Any file that separates no slew hypothesis lands its argmin
> at 1.0 whatever the engine does, so the agreement between the derived 1.0 and the
> identified 1.0 carried no information — and, taken at face value, the two numbers were
> not equal anyway.

`gesture_id.py`, same grid and protocol on both files, identify on the first half:

| | argmin | delay determined | slew determined | band | split-half |
|---|---|---|---|---|---|
| gesture (act) | delay 0, slew 1.0 ⚠ | [0…6] — the whole grid | [1.0 … none] — the whole grid | 0.1203 | DISAGREE |
| walk-1 (official) | delay 5, slew 2.0 | [2, 3, 4, 5, 6] | [2.0, 3.0] | 0.0063 | DISAGREE |

⚠ at the grid boundary, on delay, slew and deadband alike.

**The capture we asked for determines strictly less than the walking file it was meant
to improve on.** Its confidence band is 19× wider (0.1203 against 0.0063) and both
parameters come back as the entire grid. That is the result. The excitation really is
wider — 53 keyframes over 216 s, horn commands spanning ±85.9° off neutral, steps of
median 40.4° and up to 111.1° (7 of 52 repeat the previous pose), against a gait's one
narrow band.

**What the log does not record, and what follows from that.** Each pose row carries a
target and a send time. The `act` verb has a duration; no field holds it. So any
statement about what the body was *commanded* between two keyframes is an assumption
about the engine, and the log admits two that it cannot separate:

| reading | assumption | commanded rate | acts interrupted by the next keyframe |
|---|---|---|---|
| constant rate | every act plays at one fixed rate, the documented example starting from neutral | 1.0074 rad/s throughout | 47 % |
| constant duration | every act takes the documented 700 ms whatever the step | median 1.01, max **2.77 rad/s**; 15 of 52 steps at or above the walk lane's 2.0–3.0 | 59 % |

Under the first, the horn is never asked to move at its limit. Under the second, 15 of
52 commanded ramps are at or above the whole slew band the walk lane determines. The
published claim that the horn is never driven at its own limit assumed the first without
saying so, and the second is as consistent with every field this file carries. Neither is
resolved here, so neither is published as a property of the robot; both are recorded, with
their assumptions, in the artifact's `act_duration` block. (Under the first reading the
median act takes 700 ms *by construction* — the median step is the same move the rate is
defined from, so that number tests nothing.)

Two further properties of the file are measured rather than inferred. The preflight
reports **15 stretches, 1.0–14.7 s each and about 130 s in total**, in which the body does
not respond while commands sit up to **86°** off neutral — the same words it used for the
tipped walk (`results/logs/gesture_id.txt`). And of the 4,000 starts `gap_report.py`
samples, **3,158 (79 %) fall in `still` against 422 (11 %) in `acting`**
(`results/gap_report_SEND-gesture-3_6min.json`, per-regime `n`).

**The competing explanation, which is not excluded.** `gap_report` says of a motionless
identification half that "every hypothesis replays to the same absent response, the argmin
is noise" — and refuses the after-servo column when it happens. That is exactly this
file's shape: a mostly-motionless record. The undetermined result may be the missing act
duration, and it may be that four fifths of the record is a body that is not moving; the
two are not separated here, and the earlier text named only the first. Since this
revision, `gap_report --servo-id` also refuses the gap* column on a boundary argmin whose
determined sets span the whole grid, which is why this file's artifact now carries a
refusal where it used to carry a full `gap_after_servo` column computed from delay 0 and
slew 1.0.

The reading is: **this file determines neither delay nor slew**, at a band 19× the walk
lane's, and what it would take to say why is data the log does not carry — the `ms` per
act, or the realized 30 Hz command stream. Either one turns the same 3.6 minutes into a
record that could answer the question, and neither is an argument that the answer would
then be the glide engine.

Reproduce: `.venv/bin/python gesture_id.py` → `results/gesture_id_SEND-gesture-3_6min.json`,
`results/logs/gesture_id.txt`. Needs the untracked `SEND-gesture-3.6min.json` and
`imu-walk-1-*.json`.

## Coverage — **RETRACTED**: the experiment was invalid twice over

> **Retraction.** This section previously published a negative result — "synthesized
> sit↔stand transitions move pitch nowhere, so the pitch gap is not a coverage hole".
> That result is withdrawn. Not because the sign flipped, but because the experiment
> could not have measured what it claimed to measure. Both defects were in the design,
> both were visible in the run's own output, and the numbers are preserved in
> `results/coverage.json` (conclusion field marked retracted) so the record shows the
> correction rather than a silent deletion.

**Defect 1 — the premise is false. There is no sit-to-stand in the logs.**
The hypothesis was built on one header field, and the field says the opposite of what
was read into it:

> `post_walk`: "legged **done** walks fold to a sit act {l:130,r:50,ms:700} **AFTER
> recording ends** (act verb, not pose — documented so the tail is explained, never
> mistaken for mid-stride)"

The fold happens after the recording, and only on walks that end `done`. walk-3 ends
`tipped`. The sit pose {l:130, r:50} appears in neither file's pose stream — the closest
command in either log is far from it. The −1.0 rad pitch tail that was read as a sit is
a **fall**: a 5 g event at t=73044 ms, `rate_alpha` 124–200 °/s, `ori_beta` 1.8° → 56°
in 0.6 s, and the recording stops with the body still down. There was no missing motion
to supply, so there was no coverage hole to test.

**Defect 2 — the manipulation was null, and the sanity check could not fail.**
The precondition asked whether the synthesized transitions reach walk-3's pitch
excursions. They do. So does the **standard** data — which the same run measured,
printed, and never compared against:

| pitch range, rad | min | max |
|---|---|---|
| standard 400 k (the control) | −1.570 | +1.570 |
| synthesized transitions 100 k (the treatment) | −1.568 | +1.459 |
| walk-3 held-out (the target) | −1.013 | +0.014 |

The treatment adds no pitch range the control did not already have, and the target lies
entirely **inside** the control's range. The augmented cells therefore varied nothing on
the axis the 2×2 existed to test, and their flat pitch measures nothing. The check as
written compared the treatment with the target and skipped the control, so it could only
ever return "covered". `s_std` was computed and left out of the test.

A third problem compounds both: roughly half the held-out slice was walk-3, which is
2.6 s of a motionless body under swinging commands followed by a fall — see the Real2Sim
section for why that file is no longer scored as a walk.

**What is retracted.** The negative verdict on the coverage hypothesis; the claim that
the synthesized data "demonstrably contains the missing motions"; the free replication of
real2sim's roll/yaw closure (it replicates a number that is itself superseded); and the
additivity claim — its observed-vs-expected differences (1.5 and 1.2 pts) sit inside one
control seed spread, so at best it was ever "consistent with additivity within noise".

**What survives.** Nothing about the pitch gap's cause. The open question is simply open
again, and it is now better posed: on walk-1, the only file that walks, pitch and roll sit
at almost the same distance below their twin floors, which is not the signature a
pitch-specific coverage hole would leave.

`coverage.py` is kept and its sanity precondition replaced with one that would have caught
this — the treatment must add range the **control** lacks, the target must lie outside the
control's range, and the pose the synthesis is built around must actually occur in the logs
— so that a rerun is honest if the experiment is ever justified again.

Reproduce: `.venv/bin/python coverage.py` → `results/coverage.json`, `results/logs/coverage.txt`.
The script writes the `retracted` / `retraction` / `conclusion_retracted` fields itself, so a
rerun cannot drop the withdrawal. Needs both untracked walk logs and `results/real2sim.json`.

## Multi-step training loss — small, real gain

Same 128×2 net, trained through an H-tick unroll with loss on every step (SPR-style),
evaluated with the usual open-loop rollout. 3 seeds, 40 epochs:

| train unroll | @100 ms | @500 ms | @1000 ms | fit |
|---|---|---|---|---|
| H=1 (one-step, ships) | 95.8 ± 0.3 | 82.6 ± 0.1 | 77.9 ± 0.3 | 35 s |
| H=5 | 96.0 ± 0.3 | 83.8 ± 0.2 | 78.7 ± 0.4 | 166 s |
| H=10 | 95.6 ± 0.2 | **84.0 ± 0.2** | **78.8 ± 0.3** | 292 s |

+1.2–1.4 points at 500 ms, consistent across seeds (σ ≈ 0.2), nothing at 100 ms, at 5–8×
the training cost. Real, modest, and it saturates by H=5. It backs the "multi-step
consistency at training time" lever without making it a big one for this body.

Reproduce: `.venv/bin/python multistep.py` → `results/multistep.json`, `results/logs/multistep.txt`.

## Fall recovery through imagination — a feature, with a low physical ceiling

Fall recovery as a use of the mimic module: target = the upright resting stance, start from
fallen states the physics produced (pushes + hard leans), planner vs two model-free
baselines, 4 s budget, success = upright for 0.5 s. 60 "tipped" starts (|roll| < 1.2 rad,
the recoverable bucket):

| policy | recovered |
|---|---|
| hold still | 18.3 % |
| scripted wiggle | 28.3 % |
| **plan with the forward model** | **36.7 %** |

Twice hold-still and +8 over the reflex a person would code; the same ordering holds on
side and back falls at lower rates (90 mixed starts: 30.0 % vs 18.9 / 20.0 %). The
ceiling is the body, not the model: two legs cannot right most falls. Worth having as a
verb; not a headline.

Reproduce: `.venv/bin/python fall_recovery.py` → `results/fall_recovery.json`.

## PETS — the model knows where it is unsure; planning through that knowledge does not help

Ensemble of 5 probabilistic nets (mean + log-variance, Gaussian NLL, bootstrap), TS-∞
particle planner. Three findings, in decreasing order of usefulness:

*Calibration, at the regime level, is clean.* Predicted aleatoric std by regime: calm
0.21 → moderate 0.23 → **fast 0.34** → fallen 0.28; actual mean |error| 0.05 → 0.10 →
**0.18** → 0.12 — same ordering. Epistemic std ×4 from calm to fast. The model knows
where the contact chatter is. Per-tick correlation of predicted std with error is only
0.18: it captures the regime, not the individual bounce, consistent with that bounce being
irreducible.

*Accuracy: nothing.* Ensemble mean 82.5 % vs single net 82.0 % at 500 ms.

*Planning through particles: nothing on mimic, harmful on fall recovery.* Mimic 40 targets:
single MLP 0.095, ensemble mean 0.092, PETS-8 0.096. Fall recovery, 60 tipped starts:
hold still 18.3 %, single MLP 30.0 %, ensemble mean 28.3 %, **PETS-8 21.7 %, PETS-16 18.3 %**
— monotonically worse with more particles. Averaging cost over noise the model cannot
predict flattens the differences between plans and CEM stops finding the good ones. The
uncertainty is worth having as a *signal* (e.g. to tell the harness when imagination is
not to be trusted); it is not worth planning through on this body.

Fall-recovery rates move ±5 pts with the model's training seed (30.0 % here vs 36.7 %
above, same starts); orderings hold, absolute numbers carry that margin.

Reproduce: `.venv/bin/python pets.py` → `results/pets.json`, `results/logs/pets.txt`;
`.venv/bin/python pets_fall.py` → `results/pets_fall.json`, `results/logs/pets_fall.txt`.

## Metadata conditioning — negative

Excitation-mode and body one-hots as extra input (the π0.7 idea). One body: no change.
Two bodies pooled, 3 seeds: per-body 95.5±0.3, pooled 95.3±0.3, pooled+body 95.5±0.3 —
within noise. Adding "less informative" excitation never hurts without metadata; with
metadata, unseen labels at test time collapse the model (64.7 %). The π0.7 effect concerns
quality-heterogeneous imitation data; a forward model predicts physics and has no quality
axis to separate. Useful side result: one model serves two bodies at no cost.

## TimesFM 2.5 baseline

Google's 200M-param zero-shot forecaster on the six IMU channels, same 400 windows,
action-blind: 85.0 % @100 ms and 55.0 % @500 ms — ties persistence (82.0 / 55.2 %) and
loses to the 25k-param action-conditioned MLP (96.0 / 79.0 %). The information is in the
action, not the sensor history. Forecaster ≠ world action model. Also ~1 s per window on CPU.

