# Draft issue for britcruise9/GrowBot — the body XMLs declare torsional and rolling friction that MuJoCo never applies

Status: draft, not posted. Numbers from `results/contact_friction.json` in
xebastian153/growbot-cerebellum (conditions stated with each).

## What

Both body files (`growbot_body.xml`, `growbot_olie_body.xml`) set the contact geoms to
`condim="3"` and declare `friction="<sliding> 0.1 0.1"`. Under `condim=3` MuJoCo solves a
three-dimensional contact — one normal, two tangential — and the torsional and rolling
coefficients are not part of the solve. The `0.1 0.1` has therefore never been applied:
the bodies have no torsional friction, not a badly tuned one.

## Evidence

Two measurements, stated separately.

The first three rows are a raw-stream audit: one 6 000-tick stream, same seed and excitation,
compared observation by observation against the shipped body (no forward model involved).
A sliding-friction change on the same audit moves the stream by up to 26.7 in observation
units, so the audit can see a change when there is one.

| change | raw observation stream vs the shipped body |
|---|---|
| torsional ×10 at condim 3 | bit-identical |
| torsional ×100 at condim 3 | bit-identical |
| rolling ×100 at condim 3 | bit-identical |

The remaining rows are forward-model prediction accuracy: 30 000 ticks per corner, three
seeds, the within-0.2 rad rate at 100 ms and 500 ms, each corner scored against the shipped
`condim=3` body (deltas in percentage points, roll / pitch / yaw; the pre-stated
materiality threshold is 4.60 pts):

| change (vs the shipped body) | 100 ms | 500 ms |
|---|---|---|
| condim 4, torsional 1.0 (torsional actually on) | −0.6 / −0.4 / +0.5 | −0.3 / −0.1 / −3.5 — under threshold |
| condim 6 with the XML's own `0.1 0.1` | −3.4 / −2.5 / −2.0 | **−28.1 / −41.7 / −22.6** |
| condim 6, torsional 0.1, rolling 0.0001 (rolling off) | −0.7 / −0.7 / −0.3 | +0.9 / −0.4 / −1.6 |
| condim 6, torsional 0.005, rolling 0.1 (torsional off) | −4.1 / −3.6 / −3.3 | **−29.2 / −42.3 / −27.1** |

So: the declared coefficients are inert today; switching torsional friction on changes
little; and the value that would change the robot is the declared **rolling** 0.1 — 1000×
MuJoCo's default of 0.0001 — which silently comes into force the moment anyone raises
`condim` to 6 to model torsional effects. That is the trap worth a comment in the XML.

The stated limit of this measurement: it is forward-model *prediction* accuracy on the
twin, not policy transfer. It says what the coefficients do to the simulated stream, not
whether the real spin failure is torsional.

## One-line options

1. Leave `condim="3"` and delete the two inert coefficients (`friction="<sliding>"`), so the
   file says what the solver does.
2. If torsional friction is wanted: `condim="4"` with an explicit torsional value, keeping
   rolling out of the solve.
3. If both are wanted: `condim="6"` **and** replace rolling `0.1` with a deliberate value
   (MuJoCo's default is `0.0001`); the current `0.1` is almost certainly a copy of the
   torsional number, not a choice.

Any DR sweep over the friction column should state which `condim` it ran under; at 3 the
torsional and rolling entries of the sweep were no-ops.
