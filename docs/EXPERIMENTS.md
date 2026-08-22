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

**What the segmenter finds.** `growbot-imulog-1` carries no event rows, so `imulog.py`
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

- **Roll closes at every tested point**, but the four points spread from +4.5 to +33.4
  pts — a 29-point swing between configs that the identification cannot tell apart. The
  choice of servo inside the band matters enormously, and the band is not narrowed.
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
| + all three | L(5, 4.0) R(6, 1.0) ⚠ | [0 … 6] | +6.4 / −1.3 / +8.3 |

⚠ marks an argmin sitting on the grid's **boundary** — by this repo's own rule
(`servo_id.argmin_interior`) that is the search running out, not an identification. Both
per-side argmins are at delay 6 = max(grid delays), and the slower horn at slew 1.0 =
min(grid slews). The shared argmins in rows 1–2 are interior; nothing else in this table
is. Boundary status is checked per side and published in
`results/identification_ablation.json → per_side_solution.left/right_argmin_interior`.

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
  script now reports it as `MARGINAL` rather than as a boolean. So "per-side fits better"
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

