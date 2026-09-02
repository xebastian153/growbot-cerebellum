# growbot-cerebellum

A forward model — "physical imagination" — for the [GrowBot](https://github.com/britcruise9/GrowBot)
two-servo body, trained on the project's own MuJoCo digital twin, packaged in the walk
policy's convention (pure-JS runner + weights, verified against the trained net), and a
set of measured experiments around it.

The launch video names the wall: the creature can read the last second of its IMU but
cannot imagine the next one, so it fails the mimic game and cannot learn fine motor
skill from experience. The missing piece is a cerebellum — a fast model that predicts
the sensory outcome of an action, so the error against reality can drive learning.
This repository builds the predicting half in simulation, measures what it can and
cannot do, and reads the first real IMU logs against it.

**Status:** proposed and accepted in [GrowBot #6](https://github.com/britcruise9/GrowBot/issues/6),
submitted as [GrowBot PR #7](https://github.com/britcruise9/GrowBot/pull/7) (`policy/forward/`).
Two real walk sessions, one still capture and one gesture capture have been read; every
real number here comes from one unit and one phone.

## Setup

```bash
uv venv --python 3.11 .venv
VIRTUAL_ENV=.venv uv pip install --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
  -e ".[dev]"
.venv/bin/pytest -m "not slow"                  # 24 tests, one second
.venv/bin/pytest                                # + the round-trip suite, ~5 min (same as `python imulog.py`)
```

CPU is enough for everything; `requirements.lock` is the exact freeze that produced
`results/`. The twin data is gitignored and regenerated with:

```bash
.venv/bin/python sim/growbot_sim.py --steps 400000 --seed 0  --out data/train.npz
.venv/bin/python sim/growbot_sim.py --steps 60000  --seed 1  --out data/test.npz
.venv/bin/python sim/growbot_sim.py --body olie --steps 400000 --seed 10 --out data/olie_train.npz
.venv/bin/python sim/growbot_sim.py --body olie --steps 60000  --seed 11 --out data/olie_test.npz
```

`sha256sum -c data/twin.sha256` then says whether your machine regenerated the same stream
the published numbers come from — the twin is chaotic and the stream is CPU-dependent, so
on a different CPU every number moves at the second digit (see CONTRIBUTING.md).

Each experiment is one script with a `Reproduce:` line in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md);
`python <script>.py --help` states its question and never runs anything.

## On a real `?imulog=1` session

```bash
.venv/bin/python imulog.py session.json                  # preflight: units, rates, clock, jitter, per-segment physics
.venv/bin/python sensor_id.py session.json               # sensor side: fusion-filter lag, gyro Allan noise, dt stats
.venv/bin/python gap_report.py session.json --servo-id   # gap per regime and axis vs the twin floor, servo identified
```

One session per command; `gap_report.py` refuses to pool files that differ in agent gain
or resting attitude. Real logs are the maintainers' data and are never committed.

## Results

One line per experiment; the full write-up with conditions, per-seed spreads and the
decision rule stated before the run is behind each link, and every number resolves to a
file under `results/`.

| experiment | verdict |
|---|---|
| [Forward model](docs/EXPERIMENTS.md#forward-model) | 96.0 % within 0.2 rad at 100 ms, 82.7 % at 500 ms; over the linear baseline the gain is in fast motion (73.5 → 87.3 %) and falls (75.4 → 90.1 %) |
| [Mimic game](docs/EXPERIMENTS.md#mimic-game) | planning without a model is worse than doing nothing (0.220 vs 0.210 rad); with it, 0.095 rad ᴸ |
| [JS runner](docs/EXPERIMENTS.md#js-runner) | float32-equivalent to the trained net; the equivalence test caught a real convention bug |
| [Sim-to-real proxy](docs/EXPERIMENTS.md#sim-to-real-proxy--negative) | **negative at 100 ms** — mass, CoM, leg, gain and sliding friction never reach the IMU there ᴸ |
| [Body parameters at 500 ms](docs/EXPERIMENTS.md#body-parameters-at-500-ms--a-3-cm-centre-of-mass-shift-costs-338-pts-of-pitch-and-3755--of-the-held-out-drop-is-the-body-not-the-model) | a 3 cm centre-of-mass shift costs 33.8 pts of pitch (28.1 / 34.9 / 38.5 across seeds); mass alone moves nothing ᴸ |
| [Centre-of-mass identifiability](docs/EXPERIMENTS.md#centre-of-mass-identifiability--the-method-that-identifies-the-servo-cannot-identify-the-centre-of-mass-negative-and-a-slow-servo-leaves-the-delay-undetermined-even-with-the-right-body-in-the-grid) | **negative** — the null case fails on every counted seed; whether a CoM shift reads as a servo delay is unresolved (one counted seed) |
| [Contact friction](docs/EXPERIMENTS.md#contact-friction--the-twin-has-no-torsional-friction-and-the-dr-negative-could-not-have-seen-one) | the twin has **no torsional friction** at the shipped `condim=3`; its inert rolling value would cost 22.6 pts of yaw at 500 ms if switched on ᴸ |
| [Actuator dynamics](docs/EXPERIMENTS.md#actuator-dynamics--the-sim-to-real-signature-that-is-there-and-how-to-recover-it) | a hidden servo is identified from IMU + commands alone; held-out 500 ms 80.4 → 84.0 % |
| [Model mismatch](docs/EXPERIMENTS.md#model-mismatch--wrong-family-identification-recovers-90--of-the-gap-split-half-catches-drift-not-shape) | out-of-family servos identify to their nearest grid point; split-half catches drift, not shape; a linear residual adds nothing (**negative**) ᴸ |
| [Yaw floor](docs/EXPERIMENTS.md#yaw-floor--mostly-noise-not-model-scaling-is-flat-and-privileged-state-adds-little-negative) | **negative** — 4× data +0.9, 4× capacity +1.3, privileged contact state +3.1 pts, none material ᴸ |
| [Multi-step training loss](docs/EXPERIMENTS.md#multi-step-training-loss--small-real-gain) | +1.2–1.4 pts at 500 ms, consistent across seeds, saturates by H=5 ᴸ |
| [Fall recovery](docs/EXPERIMENTS.md#fall-recovery-through-imagination--a-feature-with-a-low-physical-ceiling) | planner doubles hold-still (18 → 37 %); the ceiling is the two-legged body ᴸ |
| [PETS](docs/EXPERIMENTS.md#pets--the-model-knows-where-it-is-unsure-planning-through-that-knowledge-does-not-help) | uncertainty calibrated per regime; planning through it is neutral-to-harmful ᴸ |
| [Metadata conditioning](docs/EXPERIMENTS.md#metadata-conditioning--negative) | **negative** — one model serves two bodies regardless of the tag ᴸ |
| [TimesFM 2.5 baseline](docs/EXPERIMENTS.md#timesfm-25-baseline) | a 200M-param action-blind forecaster ties persistence; the information is in the action ᴸ |
| [The first real logs](docs/EXPERIMENTS.md#the-first-real-logs--read-per-file-per-segment) | at 100 ms every baseline transfers, persistence included; at 500 ms the walk sits ~40 pts under the twin and persistence beats the model on roll (89.8 vs 50.8) |
| [Real2Sim loop closure](docs/EXPERIMENTS.md#real2sim-loop-closure--an-actuator-model-helps-on-the-real-walk-which-actuator-model-is-not-identified) | an actuator model helps on the real walk (roll +4.5 to +33.4 pts across the tested cells); **which** one is not identified ᴸ |
| [Identification ablation](docs/EXPERIMENTS.md#identification-ablation--per-side-gains-the-most-and-proves-the-least-multi-horizon-backfires-and-the-delay-was-over-charged-by-a-tick) | aligning the observation channels moves the delay 5 → 4 ticks (80 ms is an upper bound); per-side gains the most and proves the least; every variant splits DISAGREE ᴸ |
| [Gesture and still captures](docs/EXPERIMENTS.md#the-gesture-and-still-captures--the-still-lane-pays-out-the-gesture-lane-cannot) | the still lane measures the phone's gyro noise (ARW 6.42e-04 / 3.05e-04 / 1.26e-04 rad/s/√Hz); the gesture lane determines nothing ᴸ |
| [Coverage](docs/EXPERIMENTS.md#coverage--retracted-the-experiment-was-invalid-twice-over) | **retracted** — false premise and a null manipulation; kept in place with both defects named ᴸ |
| [`?imulog=1` parser](tests/test_imulog_roundtrips.py) | round-trip validated: a hidden servo through 60/30 Hz jittered sampling and three dialects, a hidden still prefix, walk and tip through the segmenter |
| [Sensor characterisation](tests/test_imulog_roundtrips.py) | a hidden 60 ms fusion lag recovered on all three body-rate axes within 10 ms, gyro noise density within 20 %, a timing stall flagged, a bias instability refused |

<sup>ᴸ</sup> artifact written before the `provenance` block existed (15 of 20 rows): the
numbers are unchanged and still resolve to `results/<name>.json`, but that file does not record
the commit, library versions or data stream that produced it. Artifacts without the mark do.

## What holds, what doesn't

**Holds.** A small action-conditioned forward model gives the body a usable physical
imagination in the twin — right about the shape of the next 100–500 ms, good enough to
win the mimic game, small enough for the phone browser. On the real walk it is as good as
the twin at 100 ms — and so is predicting no change at all: the real body moves little in
100 ms, so that number is not evidence for the model.

**Doesn't, yet.** At 500 ms the real walk sits ~40 pts under the twin's floor on every
axis, and persistence beats the model by 39 pts on roll: the model imagines twin-sized
swings the slower real body does not make. The actuator explains part of it and is
recoverable from IMU + commands; the rest is not identified, and the learning half —
compare prediction with the *real* IMU on device and update from the error — needs a body
and a longer log.

## Contents

| path | what |
|---|---|
| `growbot_cerebellum/` | the library: `sim` (twin, `ServoModel`, `perturb`, `collect`), `forward` (models, windows, rollouts), `servo_id`, `sensor_id`, `imulog` (parser, preflight, segmenter, fixture), `gap`, `sim2real`, `honesty`, `planner`, `provenance` |
| `sim/` | the vendored body XMLs and walk policy (see NOTICE) and the data-collection command |
| `*.py` | one experiment per script, a thin CLI over the package; `results/<name>.json` is its artifact, `results/logs/` its run log |
| `forward-model/` | **the PR payload**: `growbot_forward.js`, `growbot_planner.js`, `forward_85mm.json`, `test_forward.mjs`, README |
| `tests/` | the round-trip suite (`slow`) and the fast checks; `.github/workflows/ci.yml` runs both |
| `docs/EXPERIMENTS.md` | every write-up, with a table of contents and the at-a-glance table |
| `docs/READING.md` | literature and repositories, each with a cross-check against this code |
| `docs/CORRECTIONS.md` | every retraction and correction, dated by commit; the experiment sections state current belief and link here |
| `docs/CONVENTIONS.md`, `AGENTS.md`, `CONTRIBUTING.md` | the documentation standard, the invariants with their scars, and how to set up and add an experiment |
| `docs/ISSUE-6.md` | the proposal text as opened upstream |

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
