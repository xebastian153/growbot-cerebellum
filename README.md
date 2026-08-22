# growbot-cerebellum

A forward model — "physical imagination" — for the [GrowBot](https://github.com/britcruise9/GrowBot)
two-servo body, trained on the project's own MuJoCo digital twin, packaged in the walk
policy's convention (pure-JS runner + weights, verified against the trained net), and a
set of measured experiments around it.

The launch video names the wall: the creature can read the last second of its IMU but
cannot imagine the next one, so it fails the mimic game and cannot learn fine motor
skill from experience. The missing piece is a cerebellum — a fast model that predicts
the sensory outcome of an action, so the error against reality can drive learning.
This repository builds the predicting half in simulation and measures, honestly, what
it can and cannot do.

**Status:** proposed and accepted in [GrowBot #6](https://github.com/britcruise9/GrowBot/issues/6),
submitted as [GrowBot PR #7](https://github.com/britcruise9/GrowBot/pull/7) (`policy/forward/`).
Sim-only until a real IMU log exists.

---

## Contents

| path | what |
|---|---|
| `sim/growbot_sim.py` | the twin at 50 Hz: phone-style IMU (rpy + gyro), numpy port of the shipped walk policy, excitation mix, pushes, two bodies (85 mm walk, Olie spin), domain-randomisation corners, data collection |
| `forward.py` | forward models (persistence / linear / MLP), one-step and open-loop rollout evaluation, per-regime breakdown |
| `mimic.py` | CEM planner over 300 ms action chunks that imagines with a forward model; executes in the twin against held-out target motions |
| `export_js.py` | exports weights + reference vectors for the JS runner |
| `forward-model/` | **the PR payload**: `growbot_forward.js`, `growbot_planner.js`, `forward_85mm.json`, `test_forward.mjs`, README |
| `sim2real_proxy.py` | frozen vs online-residual vs oracle across the project's 13 DR corners |
| `multistep.py` | H-step unrolled training loss vs one-step |
| `pets.py`, `pets_fall.py` | probabilistic ensemble + particle planner: calibration by regime, mimic, fall recovery |
| `fall_recovery.py` | planner vs hold-still vs scripted wiggle from real fallen states, by severity |
| `actuator_proxy.py`, `servo_id.py` | actuator-dynamics proxy (latency / slew / deadband) and servo identification from IMU + commands through the frozen model |
| `model_mismatch.py` | identification stress test: out-of-family servos (load-dependent slew, voltage sag) vs the grid, plus a residual fallback |
| `yaw_floor.py` | yaw-floor decomposition: data/capacity scaling vs a teacher-forced privileged-state probe (contact forces, linear velocity) |
| `sensor_id.py` | the sensor side of the same question: fused-orientation lag behind the raw gyro, still-segment Allan deviation (measured gyro noise for the twin), clock-jitter stats per stream |
| `metadata_experiment.py` | does conditioning on excitation mode / body help? (π0.7 analogue) |
| `timesfm_baseline.py` | Google TimesFM 2.5 zero-shot as an action-blind baseline |
| `gap_report.py` | the day-of-log command: gap per regime and axis as real − twin floor, optional after-identified-servo column |
| `real_log_report.py` | the full day-of-log run on the maintainer's real walk files, **per file**: computed adapter evidence, per-segment gap, extended-grid servo identification, the agent-gain differential test, sensor-side numbers (needs the untracked walk logs) |
| `real2sim.py` | the loop closed: identified servo → twin's `ServoModel` → retrain → score on walk-1's held-out half, at three points of the identification band, plus a zero-delay smoothing-only cell and a nominal control (needs the untracked walk logs) |
| `coverage.py` | **retracted** 2×2 factorial: {nominal, identified servo} × {standard, +sit↔stand transition data}. Kept with its sanity precondition fixed; see the retraction in `docs/EXPERIMENTS.md` |
| `identification_ablation.py` | four changes to how the servo is identified — observation/command channel alignment, multi-horizon scoring, per-side servos — each scored on walk-1's held-out half (needs the untracked walk logs) |
| `imulog.py` | parser for GrowBot `?imulog=1` sessions (two native-rate streams → 50 Hz arrays; regimes from event rows, or **synthesized from the data** for `growbot-imulog-1`, which carries none: still / walking / impact / fall) + preflight (`imulog.py <file>`: units, rates, clock, per-segment still physics, still-under-active-commands — validated against deliberately corrupted fixtures) + round-trip tests that recover a hidden servo through 60/30 Hz jittered sampling and a hidden still prefix, walk and tip through the segmenter |
| `results/` | every number below, as JSON; raw run logs in `results/logs/` |
| `docs/READING.md` | literature tied to each open question, plus what talks and tools left behind |
| `docs/EXPERIMENTS.md` | full write-ups of every experiment |
| `docs/CONVENTIONS.md` | the rules this repository holds itself to |
| `docs/ISSUE-6.md` | the proposal text as opened upstream |

## Setup

```bash
uv venv --python 3.11 .venv
VIRTUAL_ENV=.venv uv pip install --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
  mujoco numpy 'torch==2.5.1'
```

CPU is enough for everything. The twin steps at ~178× realtime headless. `data/*.npz` is
gitignored and regenerated by the commands below.

## Reproduce

```bash
# data: 400k ticks train (seed 0), 60k test (seed 1), per body
.venv/bin/python sim/growbot_sim.py --steps 400000 --seed 0  --out data/train.npz
.venv/bin/python sim/growbot_sim.py --steps 60000  --seed 1  --out data/test.npz
.venv/bin/python sim/growbot_sim.py --body olie --steps 400000 --seed 10 --out data/olie_train.npz
.venv/bin/python sim/growbot_sim.py --body olie --steps 60000  --seed 11 --out data/olie_test.npz

.venv/bin/python forward.py                # forward models, horizons, regimes   → results/forward_K5.json
.venv/bin/python mimic.py                  # mimic game, 40 targets              → results/mimic.json
.venv/bin/python export_js.py && (cd forward-model && node test_forward.mjs)      # PASS
.venv/bin/python sim2real_proxy.py         # DR corners                          → results/sim2real_proxy.json
.venv/bin/python metadata_experiment.py    # metadata conditioning               → results/metadata_experiment.json
.venv/bin/python yaw_floor.py              # yaw floor decomposition             → results/yaw_floor.json
VIRTUAL_ENV=.venv uv pip install timesfm && .venv/bin/python timesfm_baseline.py   # → results/timesfm_baseline.json
```

On a real `?imulog=1` session the whole analysis is:

```bash
.venv/bin/python imulog.py session.jsonl            # preflight: units, rates, clock, jitter, per-segment physics
.venv/bin/python sensor_id.py session.jsonl         # sensor side: fusion-filter lag, gyro Allan noise, dt stats -> results/sensor_id_<stem>.json
.venv/bin/python gap_report.py session.jsonl --servo-id   # gap per regime and axis vs the twin floor
```

One session per command. `gap_report.py` accepts several files but refuses to concatenate
any that disagree on agent gain or resting attitude: those are different experiments, and
pooling them averages two things into a number that describes neither.

## Results at a glance

Full write-ups with conditions and per-regime splits: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

| experiment | verdict |
|---|---|
| Forward model | 96.0 % within 0.2 rad at 100 ms, 82.7 % at 500 ms; yaw is the hard axis (59 % vs 77 % at 1 s); the gain over baselines concentrates in fast motion (41 → 86 %) and falls (58 → 89 %) |
| Mimic game | planning without a model is worse than doing nothing; with it, error halves (0.210 → 0.095 rad) and 39/40 traces beat hold-still |
| JS runner | float32-equivalent to the trained net; the equivalence test caught a real convention bug |
| Body-parameter DR proxy | **negative** — mass/CoM/leg/gain/friction never reach the IMU at 100 ms; contact chatter dominates the gyro |
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
| Sensor characterisation | round-trip validated: a hidden 60 ms fusion-filter lag recovered on 3 of 3 body-rate axes (+61.5 / +61.6 / +62.1 ms, peak corr 0.89–0.92; 60.4–63.5 ms across 5 seeds), gyro noise density within 8%, an injected timing stall flagged, and a bias instability refused on a gyro that has none — all from the file alone |

## What holds, what doesn't

**Holds.** A small action-conditioned forward model gives the body a usable physical
imagination: right about the shape of the next 100–500 ms, especially under the shakes
and falls the video says the creature cannot picture; good enough to win the mimic game;
small enough to run in the phone browser.

**Doesn't, yet.** Everything here has learned MuJoCo. Two experiments designed to stand
in for "a different body" came back flat (body parameters, metadata) — the twin's physics is
too clean for output-side correction to have anything to correct. The third, actuator
dynamics, found the real signature: the gap lives on the *input* side (where the horn actually
is), and it is recoverable from IMU + commands alone. The learning
half — compare prediction with the *real* IMU on device and update from the error — needs
a body and a log. That is the next step.

## Third-party files and licence

`sim/growbot_body.xml`, `sim/growbot_olie_body.xml`, `sim/policy_85mm.json` and
`sim/growbot_policy.js` are copied from
[britcruise9/GrowBot](https://github.com/britcruise9/GrowBot) (`policy/` and
`policy/Harsh_policies/`) and remain under their **PolyForm Noncommercial 1.0.0**
licence (`sim/LICENSE.growbot`). `forward-model/forward_85mm.json` was trained on data
generated with that twin and is offered upstream under the same terms.

Everything else in this repository is by Sebastián Díaz and is released under the
**PolyForm Noncommercial License 1.0.0** as well — the same terms as GrowBot, so code
can move between the two without relicensing. See `LICENSE` and `NOTICE`.
