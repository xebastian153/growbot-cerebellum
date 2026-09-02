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

Same seed, same excitation, 30 000 ticks per corner, three seeds; the comparison is the
forward model's within-0.2 rad rate at 100 / 500 ms against the shipped body:

| change | effect on the simulated stream |
|---|---|
| torsional ×10 at condim 3 | bit-identical |
| torsional ×100 at condim 3 | bit-identical |
| rolling ×100 at condim 3 | bit-identical |
| torsional 0.005 → 1.0 with condim 4 (torsional actually on) | yaw at 500 ms moves at most −3.5 pts, under the 4.60-pt pre-stated threshold |
| condim 6 with the XML's own `0.1 0.1` | yaw −22.6 pts, pitch −41.7 pts at 500 ms |
| condim 6, torsional 0.1, rolling 0.0001 (rolling off) | −1.6 |
| condim 6, torsional 0.005, rolling 0.1 (torsional off) | −27.1 |

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
