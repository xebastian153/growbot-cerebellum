# Experiments

Every number below is reproducible with the commands in the README; the
machine-readable source is `results/`. Conventions in `CONVENTIONS.md`.

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

`node forward-model/test_forward.mjs`: single step 7.5e-6, 25-tick rollout 5.0e-6 versus
PyTorch; planner beats hold-still on a reachable target with a seeded RNG. This test caught
a real off-by-one in the Python evaluation (which action sits in history slot 0) before
it could reach a phone.

## Sim-to-real proxy — negative

Forward model trained on the nominal Olie body, measured on the 13 domain-randomisation
corners from `dr_sweep_spin.py` (mass 0.8–1.25, CoM ±3 cm, leg 0.85–1.15, gain 0.75–1.25,
friction 0.6–1.4), with an online linear residual learning from prediction error.
**Nothing to correct:** frozen 93.9 % across corners vs 93.7 % nominal, per-corner yaw
bias ±0.02, residual only adds noise. Tick-to-tick gyro change is mostly unpredictable in
the twin itself (R² ≈ 0.2, ≈ 0.05 when calm): foot–floor contact chatter, not model error.
So the project's DR does not show up in the IMU at 100 ms — consistent with its good walk
transfer — and the spin gap is unlikely to be mass/CoM/leg/gain. Contact is the untested
factor, and contact drives yaw, which drives spin.

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
servos, 96 hypotheses, 6 s of compute: both identified exactly (delay and slew; deadband is
the one parameter the IMU cannot see), and the held-out gap closes to the true-horn-angle
value (80.8 → 83.9 %, 75.9 → 80.8 %). The forward model doubles as the position sensor the
robot does not have. Sim-only, same-model-class caveat applies; the point is that a real IMU
log of a few minutes is enough data to run this.

`sensor_id.py` asks the symmetric question about the observation side (fusion-filter lag,
gyro noise character, clock jitter), validated by the same hidden-secret round-trip in `imulog.py`.

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

## Real2Sim loop closure — the identified servo, fed back, closes roll and yaw on the real logs

The real-log report identified the servo as delay 4 ticks (80 ms at the 50 Hz caller),
slew 2.0 rad/s, deadband 2° — with split-half DISAGREE (A: 4 / 3.0, B: 6 / 1.5), so a
single point would overclaim. `real2sim.py` therefore runs the whole loop at three
points spanning the determined band, plus a nominal control through the identical
pipeline: collect 400 k ticks with the servo inside the twin (seed 0), train the
standard model (128×2, K=5, 80 epochs, seed 0), evaluate on the real logs. The
corrected twins train on **commanded** actions — the horn lags inside the twin —
because commanded angles are all a real log carries.

Discipline: identification used the first half of the concatenated log, so every
real-log number here is the held-out second half only (535 ticks, 10.7 s — n is small
and quoted). The decision rule preceded the numbers: closure on an axis at 500 ms is
material when it beats max(3.0 pts, 2× the control's spread across 3 MLP seeds). The
control collection is asserted array-equal to `data/train.npz`, and its real-log
numbers reproduce the published baseline exactly (60.5 / 40.1 / 43.2 at 500 ms).

| real log held-out, within 0.2 rad @500 ms | roll | pitch | yaw |
|---|---|---|---|
| control (nominal servo) | 60.5 | 40.1 | 43.2 |
| argmin — delay 80 ms, slew 2.0 | 79.1 **(+18.6)** | 40.5 (+0.4) | 58.9 **(+15.6)** |
| half-A — delay 80 ms, slew 3.0 | 79.5 **(+19.0)** | 39.9 (−0.2) | 56.3 **(+13.1)** |
| half-B — delay 120 ms, slew 1.5 | 88.2 **(+27.6)** | 42.6 (+2.5) | 63.3 **(+20.0)** |
| materiality threshold (2× seed spread) | 7.2 | 3.0 | 3.8 |

Every config in the determined band closes roll and yaw materially — the loop is
**validated robustly to the identification uncertainty**: it does not matter where in
the DISAGREE band the true servo sits, retraining against any of it transfers. The
slowest config reaches its own twin floor on roll (88.2 vs 86.1), and closure grows
monotonically toward the slow end of the band — weak evidence the truth sits there,
for the gesture capture to settle. At 100 ms every config stays at the floor (96–100 %),
so the correction costs nothing at short horizon.

Pitch does not move (+0.4 / −0.2 / +2.5, all under threshold) — exactly what the
held-out attribution predicted, and the strongest evidence yet for the coverage
hypothesis: the tipped walk's sit-to-stand has no counterpart in twin training data,
and no servo model can supply missing motions.

Context, not caveat: on the slower servos the policy's realized gait shrinks (horn
amplitude 0.39 → 0.21–0.25 rad, mean gyro 1.7 → 0.9–1.2 rad/s, falls roughly
unchanged) — the same policy driven through the identified servo walks noticeably
more gently than the twin pretends, which is itself an argument for retraining the
walk policy against the corrected twin upstream.

## Coverage — transition data does not close the pitch gap (negative)

The Real2Sim section above read the immovable pitch as "the sit-to-stand coverage
hole": the tipped walk contains a sit-to-stand, twin training data contains nothing
like it, and no servo model can supply missing motions. `coverage.py` tests that
reading directly — and kills it.

Design: a 2×2 factorial so the coverage effect and the actuator effect separate —
{nominal, corrected-argmin servo} × {400 k standard ticks, 300 k standard + 100 k
synthesized transitions}. The transitions are built around the REAL sit pose from
the log header (act {l:130, r:50} through the adapter's cal inversion →
[+0.705, −0.705] rad), glided over 0.3–1.5 s (the app's own 700 ms /act glide sits
inside the range), held, scaled 0.75–1.20 in depth, and mixed with shipped-policy
walking bursts so sit→stand→walk chains — the exact shape the tipped walk
contains — are in the data. Shared seeds everywhere (standard: seed 0, asserted
array-equal to `data/train.npz`; transitions: seed 10; MLP seed 0); the augmented
cells' standard part is the exact 300 k prefix of the standard cells' stream, cut
at the splice. The decision rule preceded the numbers: material = gain over the
control > max(3.0 pts, 2× the control seed spread real2sim measured) = 7.2 / 3.0 /
3.8 pts for roll / pitch / yaw.

Sanity precondition, so the negative is interpretable: the synthesized data must
actually reach walk-3's pitch excursions. It does — transition pitch spans −1.57
to +1.46 rad against walk-3's −1.01 to +0.01 (the −1.0 tail is the sit itself).

| real log held-out, within 0.2 rad @500 ms | roll | pitch | yaw |
|---|---|---|---|
| control (nominal, standard) | 60.5 | 40.1 | 43.2 |
| coverage (nominal, +transitions) | 60.5 (+0.0) | 39.2 (−0.8) | 47.5 (+4.2)* |
| corrected (argmin servo, standard) | 79.1 **(+18.6)** | 40.5 (+0.4) | 58.9 **(+15.6)** |
| coverage + corrected | 77.6 **(+17.1)** | 39.7 (−0.4) | 61.8 **(+18.6)** |
| materiality threshold | 7.2 | 3.0 | 3.8 |

Pitch never moves — not in the aggregate, and not where the sit-to-stand lives:
walk-3's walking part (n=181) reads 4.4 / 3.3 / 4.4 / 4.4 % across the four cells.
Training data that demonstrably contains the missing motions changes nothing, so
**the pitch gap is not a training-coverage hole of this kind**. What the 2×2
excludes: missing motions in the command/pose repertoire. What it leaves open, as
questions: the twin never rests its body on the ground the way a seated robot
does (a contact configuration no pose sequence supplies), and the body model
itself may be wrong in the folded regime. (*) Coverage alone nudges yaw +4.2
against a 3.8 threshold — barely material, and worth exactly that much.

Two replications came free: the corrected-servo cell reproduces real2sim's
closure (+18.6 / +0.4 / +15.6, same numbers to the decimal — same seeds, same
code path), and additivity holds (combined observed +17.1 / −0.4 / +18.6 vs
expected-from-sum +18.6 / −0.4 / +19.8): the servo effect and the coverage
(non-)effect are independent. At 100 ms every cell sits at 94–100 % on every
axis. Held-out half only: 535 ticks, 10.7 s; labels official_w1 n=244,
official_w3 n=181, tip_onset n=49.

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

