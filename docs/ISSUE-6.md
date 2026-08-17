# Proposal: an open forward model (physical imagination) for the 85 mm body

The launch video puts its finger on the wall the creature hits: it can read the
last second of its IMU but cannot imagine the next one, so it fails the mimic
game and cannot learn fine motor skill from its own experience. The piece it
names as missing is a cerebellum — a fast model that predicts the sensory outcome
of an action, so the error against reality can drive learning.

I built the predicting half of that, on the MuJoCo twin in
`policy/Harsh_policies/DR_RMA_EXPORT`, and packaged it in the walk policy's own
convention (pure JS runner + weights JSON, verified against the trained net).
Before opening a PR I'd like to check it is wanted and where it should live.

## What it is

- A 24,841-param net (`128×2`, swish — same size class as `policy_85mm.json`).
- Input: last 5 `(imu6, [aRight, aLeft])` pairs, newest first — same frame,
  calibration and action units the walk policy already uses.
- Output: the change in the IMU over one 20 ms tick. Roll it on its own output
  to imagine a plan; a small CEM helper turns that into `mimic(targetTrace)`.
- `node test_forward.mjs` checks the JS against PyTorch: 7.5e-6 single step,
  5.0e-6 over a 25-tick rollout.

## What it does, measured in the twin

Held-out episodes, imagined roll/pitch within 0.2 rad of truth:

| horizon | nothing-changes baseline | linear | this model |
|---|---|---|---|
| 100 ms | 85.9 % | 93.5 % | **95.9 %** |
| 500 ms | 59.0 % | 75.0 % | **83.6 %** |

The gain is where the video locates the gap — fast motion 41 → 86 %, tipping or
fallen 58 → 89 %. On calm walking every baseline already does fine.

Mimic game (reproduce a held-out 2 s motion by planning through imagination and
executing in the twin), 40 traces: hold-still 0.210 rad RMSE, planning *without*
a forward model 0.220 (worse than doing nothing — the failure in the video), with
this model **0.095**, beating hold-still on 39/40. Replanning every 100 ms is the
optimum; that matches the sensory-delay number in the video.

## What it isn't

Sim-only. It has learned MuJoCo's physics; the sim-to-real gap is the same one
the walk policy crosses. The *learning* half — comparing prediction to the real
IMU on-device and updating from the error — needs a body and is the natural
follow-up. Nothing here touches the firmware, the protocol or the agent harness.

Source, data generation, evaluation and the mimic-game harness: https://github.com/xebastian153/growbot-cerebellum

## Questions for you

1. Is this wanted in the repo? It's a new capability, not a fix.
2. Where should it live — `policy/forward/` beside the walk policy, or its own
   top-level folder like `cerebellum/`?
3. Would you rather see it first as a verb in the harness (`mimic`), or as the
   standalone runner + weights so ports can use it?

PR would be ~230 lines of JS + docs plus a 500 KB weights JSON. Happy to split it
if you want the planner helper separate from the runner.
