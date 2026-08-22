# Reading list, tied to what we measured

Not a survey. Each thread answers a question this project ran into and could not
settle with the twin alone. Priority order.

## 1. The servos have no position feedback  →  learned actuator models

**Why:** the GrowBot servos report no position. Cheap positional
servos have latency, deadband, load-dependent slew. Our DR corners (gain ×0.75–1.25)
did not move the IMU at 100 ms, so if the spin gap is in the actuator it is in its
*dynamics*, not its gain.

- Hwangbo et al., *Learning agile and dynamic motor skills for legged robots*, Science
  Robotics 2019. The "actuator net": an MLP from recent position errors + velocity to
  torque, trained on real data, dropped into the simulator. The trick that made ANYmal
  transfer. Small model, short real log. Directly transplantable to a 2-servo body once
  a real log exists. https://pubmed.ncbi.nlm.nih.gov/33137755/
- Yamamori et al., *Actuator Reality Shaping for Zero-Shot Sim-to-Real Robot Learning*,
  arXiv Jul 2026. The inverse move: shape the real actuator's closed loop to match the
  ideal second-order model the sim assumes, via a 2-DoF feedforward/feedback controller
  per joint. Firmware-level, no learned model. Worth knowing because the Pico glide
  engine already sits between brain and servo. https://arxiv.org/abs/2607.02205
- Dao & Fern, *Simulator Adaptation for Sim-to-Real Learning of Legged Locomotion via
  Proprioceptive Distribution Matching*, arXiv Apr 2026. The closest published relative
  of this repo's loop, and it differs on the one axis we are weakest: instead of matching
  trajectories tick by tick, it compares sim and hardware as *distributions* — averaged 1D
  Wasserstein over the marginals of joint positions, velocities and actions — so no time
  alignment, no resets, no motion capture. Their non-privileged time-aligned baseline
  degrades to 86% parameter error under torque noise where the distributional cost holds
  at 19%. Two results carry directly: which simulator modification wins depends on the
  gap's shape (static parameters win a parameter shift and *fail catastrophically* on an
  out-of-family spring, where residual actuator/action models are required), and their
  failure section states the precondition our own logs violated — the hardware data must
  come from the region of state space where the policy actually works, or the optimizer
  matches the observed distribution with modifications that are not physically meaningful.
  Their real run used ~4.3 min of hardware data. https://arxiv.org/abs/2604.11090
- Sobanbabu et al., *Sampling-Based System Identification with Active Exploration for
  Legged Robot Sim2Real Learning*, arXiv May 2025. Same shape as `servo_id.py` — sampling
  over candidate parameters, scoring simulated rollouts against real trajectories, no
  differentiable simulator and no torque sensing — plus the piece this repo lacks: the
  data is *designed*. Stage 2 optimizes the command sequence of a multi-behavioral policy
  to maximize Fisher Information, i.e. it computes which motions make a parameter
  identifiable instead of hoping the recording contains them. Their own evidence that this
  is the binding constraint: plain identification lost on the attitude-tracking task
  "likely due to insufficient excitation of attitude-related system parameters", and the
  designed excitation won every task. Two ablations bear directly on this repo's
  identification, and both point at choices it currently makes: uniformly sampled clip
  horizons (0.05–2 s) beat any fixed horizon, because short horizons "fail to capture
  long-term temporal dependencies" — `servo_id.identify` scores a one-step error; and
  per-joint actuator parameters beat one shared set by a wide margin (0.55 vs 0.74
  normalized error), where the shared version was worse than no model at all on one task —
  `servo_id`'s grid fits a single (delay, slew, deadband) triple for both servos.
  Both were tried (`identification_ablation.py`): per-side more than doubled the held-out
  roll gain, while the multi-horizon score backfired on 16 s of walking — the argmin ran
  to the grid's low boundary and the delay set widened to everything. The horizon ablation
  does not transfer at this data scale; the per-joint one transfers as a *gain*, but not
  as an identification. This entry previously said per-side "separated the two horns'
  slews into disjoint determined sets". It does not. Those two sets are one-dimensional
  conditional slices, each swept with the partner frozen at its own optimum and cut with
  the band from the shared sweeps, so their disjointness restates "the two argmins differ
  by more than the band" rather than confirming it; both per-side argmins also sit on the
  grid boundary, and the per-side fit improvement (0.0064) is a ratio of 1.01 against its
  own band (0.0063). What holds is narrower and was tested separately: re-fitting per-side
  on each half puts the slower horn on the same side both times — the **right** one. (The
  left/right labels in this repo were inverted until that check was written; action column
  0 is the right leg.)
  https://arxiv.org/abs/2505.14266

**Tested:** `actuator_proxy.py` / `servo_id.py` / `real2sim.py`. A slew-limited servo opens
a 3–4 pt gap at 500 ms that an output residual cannot close and the realized horn angle
closes completely; the servo's delay and slew are identifiable from 300 s of IMU + commands
through the frozen forward model (two hidden servos, both recovered). On the real log the
thread paid out and showed its limit in the same run: slew is determined to a set below the
simulator's assumption, delay is not determined at all by ~15 s of periodic walking, and
feeding the candidates back into the twin closes roll materially at every tested point.
Note what this repo does *not* have that the entry above assumes throughout: joint
encoders. Its proprioception is IMU plus the commands it issued, so a distributional cost
here would compare IMU-feature marginals only — the action marginals are identical by
construction when the logged commands are replayed.

## 2. Learn from the real error on device  →  feedback-error learning, adaptation

**Why:** the eventual goal is continual on-device correction rather than a
higher-fidelity simulator. Our DR proxy found no
systematic error to correct in the twin, so the real signal has to come from the body.

- Kawato, *Feedback-error-learning*, 1987–90 (e.g. Neural Networks 1988). A fixed
  feedback controller keeps things stable; its output *is* the training error for an
  adaptive feedforward inverse model. Cheapest on-device learner there is, and the
  cerebellar theory the GrowBot launch video leans on. Same lineage as the GCML paper's inverse
  model. https://www.sciencedirect.com/science/article/abs/pii/0893608088900305
- Kumar, Fu, Pathak, Malik, *RMA: Rapid Motor Adaptation for Legged Robots*, RSS 2021.
  Base policy + adaptation module that infers environment latents from recent history,
  in fractions of a second, trained only in sim. The upstream training already uses
  the privileged critic; the adaptation module is the half not yet built.
  https://www.researchgate.net/publication/353116578
- Nagabandi et al., *Learning to Adapt in Dynamic, Real-World Environments Through
  Meta-RL*, ICLR 2019. Meta-learn a dynamics-model prior that adapts online from the
  last few steps. The model-based cousin of RMA; what our online residual was trying to
  be, done properly. https://arxiv.org/abs/1803.11347
- Yin et al., *Rapidly Adapting Policies to the Real World via Simulation-Guided
  Fine-Tuning*, arXiv Feb 2025. Use the sim value function to guide real-world
  exploration; up to 10× fewer real samples where direct transfer fails. Relevant once
  spin has real data and a reward. https://arxiv.org/abs/2502.02705

## 3. Gyro change is mostly unpredictable  →  aleatoric vs epistemic uncertainty

**Why:** R² 0.20 on yaw-gyro in the twin itself; 0.05 when calm. Before adding
capacity, know how much is irreducible contact noise, and stop the planner from
chasing it (replan-every-tick was worse than every 100 ms).

- Chua, Calandra, McAllister, Levine, *Deep RL in a Handful of Trials using
  Probabilistic Dynamics Models* (PETS), NeurIPS 2018. Ensemble of probabilistic nets:
  variance within a net = aleatoric, disagreement across nets = epistemic; trajectory
  sampling through both for MPC. Our forward model + CEM planner is PETS without the
  uncertainty; adding it is the natural next step and directly explains the 100 ms
  optimum. https://arxiv.org/abs/1805.12114

**Tested:** `pets.py`. Regime-level calibration is good (predicted std tracks the
error ordering calm < moderate < fallen < fast; epistemic ×4 calm→fast); per-tick
correlation 0.18. Planning through particles: no gain on mimic, monotonically worse on
fall recovery (30 → 22 → 18 % with 0 → 8 → 16 particles). Keep the uncertainty as a
signal, not as a planning input, on this body.

## 4. Contact drives yaw and nobody senses it  →  proprioceptive contact estimation

**Why:** the untested factor in their DR. No foot sensors on GrowBot; the phone IMU is
the only sense. Legged robotics has done contact detection from proprioception alone.

- *OCELOT: Odometry and Contact Estimation for Legged Robots*, arXiv May 2026.
  https://arxiv.org/pdf/2605.21863
- *Contact-Anchored Proprioceptive Odometry for Legged and Wheel-Legged Robots*,
  arXiv Feb 2026. https://arxiv.org/pdf/2602.17393
- *Four Simple Proprioceptive Estimators for Legged Robots*, arXiv May 2026 (a
  baseline-first paper, in the spirit of what we did). https://arxiv.org/html/2605.23100
- Learning-based contact estimators run at 830 Hz on a Jetson from IMU + encoders
  (see MDPI Information 16(6):479, 2025). GrowBot has no encoders, so the question is
  what survives with IMU only. https://www.mdpi.com/2078-2489/16/6/479

## 5. Their V0: next-state prediction as an auxiliary loss

**Why:** an auxiliary next-state head at training time is the natural V0;
Fast-WAM reaches the same conclusion at robot-lab scale.

- Yuan, Dong, Liu, Zhao, *Fast-WAM: Do World Action Models Need Test-time Future
  Imagination?*, arXiv Mar 2026. Removing training-time prediction
  hurts far more than removing test-time imagination; 4× faster at 190 ms latency.
  Supports V0 over the mimic planner as the *policy* path.
  https://arxiv.org/abs/2603.16666
- Schwarzer et al., *Data-Efficient RL with Self-Predictive Representations* (SPR),
  ICLR 2021. Predict your own latent K steps ahead as an auxiliary loss; +55% on
  Atari-100k. The multi-step version of the aux head, which our 96→84% decay from
  100 to 500 ms says is the next lever. https://arxiv.org/abs/2007.05929

## 6. Learn the world model on the real robot  →  DayDreamer

- Wu, Escontrela, Hafner, Goldberg, Abbeel, *DayDreamer: World Models for Physical
  Robot Learning*, CoRL 2022. Dreamer on real robots, no simulator; the reference in
  the GrowBot launch video and the ceiling for "learn from raw experience". Says what a real data
  budget looks like. https://arxiv.org/abs/2206.14176

## 7. The phone is not a truth sensor  →  observation delay and measured IMU noise

**Why:** every other thread treats the path from the body to the model as transparent.
It is not. A consumer phone reports the output of a sensor-fusion filter, which has its
own lag, and a MEMS gyro whose noise the twin replaces with nothing. The twin trains on
MuJoCo's exact state, so a second unmodelled dynamic system sits between the world and
what the model sees.

- Sintes, Bušić, Zhu, *Structural Equivalence and Learning Dynamics in Delayed MARL*,
  arXiv May 2026. Observation delay and action delay are formally equivalent: they admit
  identical policy sets and induce identically distributed trajectories, and any mixed
  configuration reduces to a pure observation-delay system. The consequence here is
  concrete — from commands and IMU alone the servo's latency and the fusion filter's lag
  are one lumped quantity, and `sim/growbot_sim.ServoModel` assigns all of it to the
  command side (its queue delays the target; nothing delays the observation). The same
  paper's experiments warn that equivalent optima do not imply equivalent learning
  dynamics, which matters when identified parameters are handed to policy training.
  https://arxiv.org/abs/2605.04345
- Ji, Mun, Kim, Hwangbo, *Concurrent Training of a Control Policy and a State Estimator*,
  RA-L 2022. Trains the estimator jointly with the policy, on the premise that the
  estimator is part of the plant rather than a clean preprocessing step.
  https://arxiv.org/abs/2202.05481
- The Allan-variance characterization of a MEMS gyro — angle random walk, rate random
  walk, bias instability from a stationary segment — is the standard way to give a
  simulator the sensor's real noise instead of a guessed Gaussian. Reference model:
  https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model ; a worked mobile-robot
  treatment: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7506677/

**Measured:** `sensor_id.py`. On the one real file that walks, the fused orientation lags
the raw gyro by +13.0 / +13.8 / +12.8 ms on wx/wy/wz, split-half AGREE, peak correlation
0.87–0.96, and the walking regime alone gives the same answer (+13.2 / +13.9 / +12.7).
That is about two thirds of a 50 Hz tick. It used to sit inside whatever delay `servo_id`
charged to the servo; `imulog.parse(ang_lead_ms=...)` now advances the angle channels to
meet the gyro, and doing so moves the identified delay from 5 ticks to 4 and narrows its
determined set, with no change to held-out accuracy — the correction is to the attribution,
not to the prediction (`identification_ablation.py`). What is left is the actuator plus the
gyro's own absolute lag, which this log cannot measure, so 80 ms is an upper bound on the
actuator rather than its value. Allan deviation stays undetermined — the longest still
segment in the walk lane is about 1 s, and the parameters need minutes.

## What is *not* on this list, and why

- Time-series foundation models (TimesFM, Chronos). Measured: TimesFM 2.5 zero-shot
  ties persistence on the twin IMU (85.0 % vs 82.0 % within 0.2 rad @100 ms, 55.0 % vs
  55.2 % @500 ms) and loses to a 25k-param action-conditioned MLP (96 / 79 %). The
  information is in the action, not the sensor history. Forecaster ≠ world action model.
- JEPA-style latent world models. Complexity before the simple thing is validated.

## How the field closes the sim-to-real gap (documented practice)

Two philosophies and a bridge; within the first, one pattern repeats in every recipe
that transfers well.

**Sim-first, gap-closing machinery:**
- ETH Zurich / ANYbotics — the open standard: an *actuator network* learned from real
  data (delays, hysteresis, saturation) inserted into the simulator, plus friction/mass
  randomization, noisy observations and random pushes during training; zero-shot on mud,
  snow, rubble. https://github.com/leggedrobotics/legged_gym
- Boston Dynamics / RAI Institute — open Spot pipeline reaches 5.2 m/s zero-shot by
  "closely modeling hardware-specific dynamics, including actuator delays and joint
  torque limits, using custom actuator classes". https://arxiv.org/abs/2504.17857
- DeepMind, OP3 soccer (Science Robotics 2024) — closest to this project's scale (small
  servo robots, falls): system identification first, then *targeted* randomization —
  floor friction, joint orientation, masses, control-loop latency — plus perturbations
  and high-frequency control; zero-shot transfer.
  https://www.science.org/doi/10.1126/scirobotics.adi8022
- OpenAI, Dactyl — Automatic Domain Randomization: the randomization widens itself as
  the policy improves, a difficulty curriculum; the hand solved the cube in a rubber
  glove it never trained with. https://arxiv.org/abs/1910.07113
- Agility Robotics (Digit) / OSU (Cassie) — a bespoke high-fidelity simulator (closed
  kinematic chains Isaac Gym cannot express) with dynamics, terrain and *delay*
  randomization; Cassie was the first end-to-end DRL sim-to-real biped (2020).
  Survey: https://arxiv.org/abs/2404.17070

**Real-data-first, sidestepping the gap:** Tesla (Optimus) and Figure (Helix) train
VLAs primarily on teleoperation and human video, simulation as a supplement;
Physical Intelligence runs fleets plus RL post-training. No gap to close because the
data was never synthetic — the cost is fleets and teleoperators.

**The bridge:** 1X learns a *world model from the robot's own real video* and trains
and evaluates policies inside it — the simulator itself is learned from reality.
https://techcrunch.com/2026/01/13/neo-humanoid-maker-1x-releases-world-model-to-help-bots-learn-what-they-see

**The pattern, and how it reads against this repo's measurements:** every sim-first
recipe that transfers well models the ACTUATOR first (learned actuator nets, custom
actuator classes, delay randomization, system ID) and randomizes *targeted* factors
second; none wins by blanket-randomizing mass and friction. That is what the twin
measured independently: body parameters never reach the IMU, actuator delay and slew
open the gap, identify it and feed it back. The Real2Sim loop is standard practice at
the top of the field. One caveat separates this project: the industry's system ID uses
encoders and torque sensors; `servo_id.py` identifies the actuator from IMU + commands
alone, because a $30 body carries nothing else — a variant none of the above documents.

## From talks and tools

Things that came in sideways and left something behind.

- **Chelsea Finn, Physical Intelligence talk (π0.7).** Four transferable
  conclusions, none of the models: (1) test-time imagination "helps, perhaps not
  as much as you'd expect" — the same finding as Fast-WAM and our twin, so the
  aux-head V0 is the policy path and the mimic planner is a feature; (2) memory at
  multiple timescales (short raw + long text summaries) is the structure GrowBot's
  harness already has; (3) a *general* value function that predicts progress
  across tasks is what a value function should be — our pose cost is not one, and
  GCML's `W(s*−s)` is a linear, cheap relative of the idea; (4) metadata prompting
  turns low-quality data from harmful to helpful — cheap to try on our forward
  model (condition on excitation mode / body) and worth considering before the
  dream digests GrowBot's experience. Action space = target joint positions + PD,
  same as GrowBot and our model; she says it is not the bottleneck.
- **TimesFM 2.5 (Google).** Measured, not read: 200M-param zero-shot forecaster
  ties persistence on the twin IMU (85.0 vs 82.0 % @100 ms, 55.0 vs 55.2 % @500 ms),
  loses to the 25k-param action-conditioned MLP (96 / 79 %). Its covariate path is
  in-context *linear* regression, so even with actions it would be the linear
  baseline we already have. Forecaster ≠ world action model. 1 s/window on CPU.
- **Soup (LLM fine-tuning from one YAML, layer streaming on 4 GB GPUs).** Not for
  this problem. Adjacent use if it ever comes up: fine-tune a small local model on
  GrowBot verb logs so the harness emits well-formed `gesture`/`walk`/`say` without
  an API. Its self-auditing measurement style is the standard to imitate.

### Metadata conditioning: tested, does not help here

`metadata_experiment.py`, 60 epochs, within 0.2 rad @100 / @500 ms:

- Q1 walk only, +excitation-mode one-hot: 95.9/82.2 → 95.8/82.8. Nothing.
- Q2 walk+Olie pooled, 3 seeds: per-body 95.5±0.3, pooled no meta 95.3±0.3, pooled
  +body 95.5±0.3 (walk @100). All within noise. Pooling neither hurts nor needs a
  body tag at this scale.
- Q3 the pi0.7 curve does not appear: adding OU and still to a clean set does not hurt
  without metadata (95.7 → 95.9); with metadata, training on clean modes and testing
  on all collapses to 64.7% because unseen mode labels arrive at test time — a
  deployment hazard, not a gain.

Why: the pi0.7 effect is about *quality-heterogeneous* data (clumsy demos, failed
rollouts) where a tag lets a policy avoid imitating the bad. A forward model does
not imitate; it predicts physics, and the physics of an OU jitter is as true as a
gait's. No quality axis, nothing for the tag to separate. Where the idea could still
bite is real-vs-sim data on the same body — that is the log we are waiting for.
