# Reading list, tied to what we measured

Not a survey. Each thread answers a question this project ran into and could not
settle with the twin alone. Priority order.

## 1. The servos have no position feedback  →  learned actuator models

**Why:** brit: "we don't have positional feedback on our servos". Cheap positional
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

## 2. Learn from the real error on device  →  feedback-error learning, adaptation

**Why:** brit's "continual correction instead of better sim". Our DR proxy found no
systematic error to correct in the twin, so the real signal has to come from the body.

- Kawato, *Feedback-error-learning*, 1987–90 (e.g. Neural Networks 1988). A fixed
  feedback controller keeps things stable; its output *is* the training error for an
  adaptive feedforward inverse model. Cheapest on-device learner there is, and the
  cerebellar theory brit's video leans on. Same lineage as the GCML paper's inverse
  model. https://www.sciencedirect.com/science/article/abs/pii/0893608088900305
- Kumar, Fu, Pathak, Malik, *RMA: Rapid Motor Adaptation for Legged Robots*, RSS 2021.
  Base policy + adaptation module that infers environment latents from recent history,
  in fractions of a second, trained only in sim. Harsh already uses the privileged
  critic; the adaptation module is the missing half he named.
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

**Why:** dsevero's plan and Fast-WAM's finding are the same finding at two scales.

- Yuan, Dong, Liu, Zhao, *Fast-WAM: Do World Action Models Need Test-time Future
  Imagination?*, arXiv Mar 2026 (dsevero's link). Removing training-time prediction
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
  brit's video and the ceiling for "learn from raw experience". Says what a real data
  budget looks like. https://arxiv.org/abs/2206.14176

## What is *not* on this list, and why

- Time-series foundation models (TimesFM, Chronos). Measured: TimesFM 2.5 zero-shot
  ties persistence on the twin IMU (85.0 % vs 82.0 % within 0.2 rad @100 ms, 55.0 % vs
  55.2 % @500 ms) and loses to a 25k-param action-conditioned MLP (96 / 79 %). The
  information is in the action, not the sensor history. Forecaster ≠ world action model.
- JEPA-style latent world models. dsevero's call: adds complexity before the simple
  thing is validated. Agree.
