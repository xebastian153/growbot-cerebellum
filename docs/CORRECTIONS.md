# Corrections log

Every retraction and correction this repository has published, in one place, dated by the
commit that made it. The experiment sections in [EXPERIMENTS.md](EXPERIMENTS.md) state what
is currently believed and link here; the history is kept verbatim so that nothing is
erased, only relocated. Rule 7 in [AGENTS.md](../AGENTS.md): corrections are recorded,
never erased.

Read the entries as the repository's error record. The pattern across them is stated in
AGENTS.md rule 8: the code was right every time, and the story on top of it was wrong.

## 2026-08-20 — coverage retracted (3827c18)

The coverage section ([EXPERIMENTS.md](EXPERIMENTS.md#coverage--retracted-the-experiment-was-invalid-twice-over))
is kept in place as a retracted section: its banner and both defects (a false premise —
there is no sit-to-stand in either log — and a null manipulation) are the section's
current content, and `results/coverage.json` carries the withdrawal in its own fields.
Published 1f6f60e, retracted 3827c18.

## 2026-08-20 — real-log aggregates withdrawn (3827c18)

From the section *The first real logs*: "An earlier version of this report published
aggregates over the pair; they are withdrawn." The two files differ in agent gain and rest
attitude and are never pooled; every table since names one file.

## 2026-08-20 — Real2Sim rescored on walk-1 alone (3827c18)

Relocated verbatim from the head of *Real2Sim loop closure* (the second paragraph was added
at bfa8475, when the hand-copied mirror of the determined sets was removed):

**This section replaces an earlier one.** The previous version scored a
concatenation of walk-1 and walk-3 and concluded the loop was "validated robustly to
the identification uncertainty". Both halves of that are withdrawn. Roughly half the
old held-out slice was walk-3 — 2.6 s of a motionless body under swinging commands,
then a fall — so a large part of what the corrected twins were credited with
predicting was a robot that was not moving. And "robustly to the identification
uncertainty" was inferred from three sampled points out of a seven-wide determined
delay set, which is a claim about a band made from a sample of it. Every number below
is new: walk-1 only, and the verdict text is computed from which cells pass and from
how much of the band they cover.

**The band numbers in this section were corrected again.** They were computed against
`real2sim.py`'s hand-copied mirror of the determined sets, which had gone stale against
`results/real_log_report.json`. The coverage figures move 43 % → **40 %** on delay and
25 % → **50 %** on slew, and the smoothing-only cell moves from inside the determined
band to outside it. The measured percentages in the table are unchanged — they never
depended on the sets — but two of the readings did, and both are rewritten below.

## 2026-08-22 — determined sets unstaled (bfa8475)

Relocated verbatim from *The first real logs*, servo identification:

These two sets are quoted from `results/real_log_report.json` (`servo.delay_determined_set`,
`servo.slew_determined_set`). An earlier revision of this section published delay
**[0 … 6] — the entire grid** — and slew **[1.5, 2.0, 3.0, 4.0]**, which were the sets
before `confidence_band` moved from a standard deviation to 1.4826·MAD. That change
narrowed both sets and nothing downstream noticed, because `real2sim.py` held a
hand-copied mirror of them. The copy is gone (`real2sim.determined_band` reads the
artifact and fails hard if it cannot), and every number below that depends on the sets
is recomputed against them.

## 2026-08-22 — left/right attribution inverted (bfa8475, 81c8e88)

Relocated verbatim from *Identification ablation*:

**The per-side L/R labels in this table were inverted before this revision**, and every
published left/right attribution with them. `servo_id.realized_per_side` put the *left*
triple on action column 0, but column 0 is the **right** leg: `imulog.parse` stacks
`np.stack([a_right, a_left], 1)`, and the twin agrees (`a = np.tanh(x[:2])  # [aRight,
aLeft]`, `joint_1 is right_leg`). The error was invisible to every metric — swapping two
labels changes no error, no gain and no determined set, only who gets the credit — so
the fix changes no number in this table, only which horn each triple belongs to. The
slow horn is the **right** one, not the left. A regression guard now runs an
asymmetric fixture (one horn deliberately crippled) through the identification and
asserts the slow triple comes back on the side it was injected on; a symmetric fixture
cannot catch a label swap, which is why the old round-trip passed throughout.

**What that guard proves, precisely.** Its first version proved only self-consistency:
it injected the crippled horn through `servo_id.RIGHT_COL / LEFT_COL` and then read the
answer's label off the same two constants, so setting them to `1, 0` moved the
injection along with the label and the guard stayed green while every published
attribution inverted. It is now bound to ground truth instead, two ways: the constants
are asserted against the twin's own XML, read independently by
`servo_id.sim_side_columns` (actuator `servo_1` → `joint_1` → body `right_leg` ⇒ action
column 0 is the right leg), and the crippled horn is injected **by action column** on
the column that XML names, not on whatever column the constants currently name.
Verified by reversing the pair: with `RIGHT_COL, LEFT_COL = 1, 0` the suite exits 1 on
the convention assert, and with that assert bypassed it exits 1 again on the
attribution assert (`identified: L(delay 6, slew 1.5) R(delay 0, slew None)` — the slow
horn injected on the right, handed back as the left). Restored, it passes.

## 2026-08-26 — gesture glide-engine reading retracted (64136fd)

The retraction statement stays at the head of *The gesture lane* in EXPERIMENTS.md. The
three paragraphs that explain why each step failed are relocated here verbatim:

**The example is a target pose, not a move.** `{l:130, r:50}` is an absolute pose pair
— 90+40 and 90−40 — so it is "a 40° move" only from neutral, and the header states no
start pose. Read through the parser's own calibration inversion it is 40.40° of horn
travel from neutral, not 40°, because the derivation dropped `cal.gain` (0.99) that
every other conversion in this repository applies. The rate that follows from it is
**1.0074 rad/s**, not the 0.9973 published in the artifact.

**The example is not from this session.** `post_walk` documents the sit fold that
happens **after recording ends**, on walks that end `done`. This is the second
conclusion in this repository built on that field, and the second to be withdrawn for
the same reason: the act it documents is not in the record it was read into.

**The confirmation was a grid artifact.** 1.0 rad/s is `min(slews)` in
`servo_id.default_grid()`. The gesture argmin sits on the grid boundary on all three
axes (`argmin_interior: false`) and its slew determined set is the *entire* grid,
"no slew limit" included. Any file that separates no slew hypothesis lands its argmin
at 1.0 whatever the engine does, so the agreement between the derived 1.0 and the
identified 1.0 carried no information — and, taken at face value, the two numbers were
not equal anyway.

## 2026-08-26 — still lane: where the ARW peak actually is (64136fd)

From *The still lane*: "This section previously said the 7.1 deg/s peak was the taps at
0–2.2 s and 71.7–73 s, 'not a disturbance in the body of the capture'. That was wrong about
which samples entered the fit: 7.1 deg/s is the last sample of the Allan segment (t = 75.70 s),
the tap that ends the recording." The trim sensitivity that followed is a current
measurement and stays in the section.

## 2026-08-26 — body parameters: partition per seed, geometry in body frame, fast-motion shift unresolved (98a3231, 1787c6c, b45fdf2)

Two review rounds on `body_params.py`. Round 1 (98a3231): the CoM drop was split into body
and model, and the phone framing was dropped. Round 2 (1787c6c): the partition is published
per seed (37 / 42 / 55 %, not a bare 45 %), the balance geometry is measured in the body
frame (nominal CoM −27.42 mm, 16.9 mm behind the support box; the first figures −19.30 and
−41.34 mm were a pose artefact, world x read on a body rotated −34.5°), and the decision
rule uses each metric's own seed spread. b45fdf2: the fast-motion shift is reported as
unresolved, not as a movement. The corrected paragraphs stay in the section because the
corrected numbers are the current numbers; each is introduced there as a correction.

## 2026-08-27 — servo_id artifact regenerated on the current grid (0fb2b24)

`results/servo_id.json` predated the `argmin_interior` and determined-set fields and the
252-hypothesis grid. Regenerated: 252 hypotheses / 16 s (was 96 / 6 s); held-out 500 ms
80.4 → 84.0 % and 77.9 → 83.1 % (were 80.8 → 83.9 and 75.9 → 80.8); the argmins are still
exact and the halves agree, but under the band the determined sets are [1 … 3] / [4 … 6]
and [0 … 3] / [3 … 6] — "identified exactly" was true of the argmin only, and the text now
says so.

## 2026-08-27 — forward_K5: the published table came from 80 epochs, not the CLI default (9888f0d, fb73537)

`results/forward_K5.json` (the 96.0 / 82.7 table) was trained for 80 epochs while
`forward.py` defaulted to 30, so the documented command did not reproduce the published
table. The default is now 80; the artifact and its log were regenerated from one run at a
clean commit (every number identical; only `provenance` and `fit_s` changed). A first
regeneration recorded `git_dirty: true` because stdout had been redirected into the
tracked log before the provenance block was taken — the Reproduce line now says to
redirect outside the repository.

## 2026-08-27 — centre-of-mass identifiability: sign test removed, noise floor consumed, "3 of 3" withdrawn (58ec2e9, 6bdcd7c)

Round 1 (58ec2e9): the "8 of 9 shifted-body seeds / 0 of 3 null seeds" sign test was a
threshold chosen after the table and is removed; the noise floor is consumed per seed with
a pre-stated rule (2 of 12 body-seeds excluded); the joint verdict marks interiority on
all four axes and publishes the deadband set; means carry their per-seed values. Round 2
(6bdcd7c): the below-noise rule had been applied to identification A only; applied to the
joint grid it excludes two of the three seeds behind "a +3 cm shift does not read as a
servo delay, resolved 3 of 3" — that claim is withdrawn wherever it was echoed and the row
is unresolved on one counted seed. The section's own sentences say so in place.
