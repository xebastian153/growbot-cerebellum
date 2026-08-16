# growbot-cerebellum

A forward model — "physical imagination" — for the GrowBot two-servo body,
trained on the project's own MuJoCo digital twin, and a planner that uses it to
play the mimic game the robot fails in the launch video.

The video's diagnosis: the robot can read the last second of its IMU but cannot
imagine the next one, so it cannot learn fine motor skill from experience. The
missing piece is a cerebellum — a fast model that predicts the sensory outcome of
an action, compares it with what happened, and uses the error to improve. This
repository builds and measures that piece in simulation.

Everything runs on CPU. The twin steps at ~178× realtime headless.

## Setup

```
uv venv --python 3.11 .venv
VIRTUAL_ENV=.venv uv pip install --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
  mujoco numpy 'torch==2.5.1'
.venv/bin/python sim/growbot_sim.py --steps 400000 --seed 0 --out data/train.npz
.venv/bin/python sim/growbot_sim.py --steps 60000  --seed 1 --out data/test.npz
.venv/bin/python forward.py
.venv/bin/python mimic.py
```

## Files

| file | what |
|---|---|
| `sim/growbot_body.xml` | the twin, from GrowBot `policy/Harsh_policies/DR_RMA_EXPORT` (PolyForm NC) |
| `sim/policy_85mm.json`, `growbot_policy.js` | the shipped walk policy, for reference and as an excitation source |
| `sim/growbot_sim.py` | twin at 50 Hz, phone-style IMU (rpy + gyro), numpy port of the walk policy, excitation mix, data collection |
| `forward.py` | persistence / linear / MLP forward models; one-step and open-loop rollout evaluation; per-regime breakdown |
| `mimic.py` | CEM planner over action chunks that imagines with a forward model; executes in the twin against held-out targets |

## What was measured

**The twin is faithful.** The shipped `policy_85mm.json`, ported to numpy, walks in
it: 0.138 m in 5 s, no fall. Same net that drives the physical robot.

**Forward models, 100 ms open-loop horizon, % of starts within 0.2 rad of truth:**

| regime | persistence | linear (GCML-style) | MLP 25k params |
|---|---|---|---|
| calm walking / gaits | ~91–93 % | ~95 % | ~96 % |
| fast motion (\|gyro\| > 3 rad/s) | 41 % | 74 % | **86 %** |
| fallen / tipping | 58 % | 75 % | **89 %** |

Where the body moves gently, persistence already wins and linearity suffices.
Where it shakes, bounces or falls — exactly where the video locates the gap —
the linear map is half-way and the nonlinear model roughly doubles persistence.

**Mimic game, 40 held-out 2 s traces, receding-horizon CEM, replan every 100 ms:**

| planner | roll/pitch RMSE | within 0.2 rad | beats holding still |
|---|---|---|---|
| hold still | 0.210 | 66 % | — |
| persistence (no imagination) | 0.220 | 61 % | 42 % |
| linear | 0.142 | 84 % | 88 % |
| **MLP** | **0.095** | **90 %** | **98 %** |

Planning without a forward model is worse than doing nothing — the failure the
video shows. With the nonlinear one the error halves and 39 of 40 traces improve.

**Closing the loop:** replanning every 100 ms is the optimum (0.095); every tick
is slightly worse (0.103, chasing noise); pure 2 s imagination still beats holding
still (0.161 vs 0.210) but drifts — the same bootstrapping drift measured in the
GCML crafting testbed, milder here because the horizon is short.

## Caveats

Simulation only. The twin is the one the gait was trained in, but a forward model
trained on MuJoCo has learned MuJoCo. The point of a cerebellum is to keep learning
from the *real* prediction error, and that needs the physical body.
