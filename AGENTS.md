# AGENTS.md

A forward model — a cerebellum — for a $30 phone-brained biped, trained in a MuJoCo
twin and measured against the real body's IMU logs. Everything below exists because
skipping it once produced a number that looked right and was not. Read it before
touching a script, and before letting any number out of this repository.

## Commands

```bash
.venv/bin/pytest -m "not slow"                   # 17 fast checks, one second
.venv/bin/pytest                                 # + the round-trip suite: 8 round-trips, hidden secrets, must PASS before and after every change
.venv/bin/python imulog.py                       # the same suite, run through the day-of-log command
.venv/bin/python imulog.py session.json          # preflight a real log: units, rates, clock, stillness — FAIL gates everything downstream
.venv/bin/python gap_report.py session.json --servo-id   # gap per regime and axis (real − twin floor), servo identified on the first half only
.venv/bin/python sensor_id.py session.json       # sensor side: fusion-filter lag, Allan noise, dt jitter
.venv/bin/python real_log_report.py a.json b.json        # the whole day-of-log chain, one file at a time
.venv/bin/python <experiment>.py --help          # every experiment is one script with a docstring stating its question
.venv/bin/ruff check .                           # lint; must be clean
```

The suite takes about five minutes on CPU (4:40 measured). A change that touches
`growbot_cerebellum/` is not done until it is green.

## Repo map

- `growbot_cerebellum/` — the library every script imports:
  - `sim.py` — the twin at 50 Hz, `ServoModel` (delay / slew / deadband in front of
    MuJoCo's ideal PD), `perturb()` for body corners, `collect()`. The body XMLs and the
    walk policy it loads stay in `sim/`.
  - `forward.py` — models, `make_windows`, `rollout_error`, `by_regime`, `K`, `AXES`.
  - `imulog.py` — parser for the real `growbot-imulog-1` format, preflight, segmenter,
    and the fixture that hides secrets for the suite.
  - `servo_id.py` / `sensor_id.py` — identification of the actuator and characterization
    of the phone, both from IMU + commands only (this body has no encoder).
  - `gap.py` (`evaluate_axes`, `twin_regimes`), `sim2real.py` (`horizon_within`, the DR
    corners), `honesty.py` (`seed_stat`, `score_corners`, `decide_per_metric`),
    `planner.py` (`Imagination`, `cem_plan`), `tee.py`, `provenance.py`.
- One script per experiment at the root, a thin CLI over the package (see the README
  table); `results/<name>.json` is the machine-readable source for every published
  number and carries a `provenance` block (commit, versions, seeds, argv), `results/logs/`
  the run log. `gap_report.py`, `real_log_report.py`, `real2sim.py` are the day-of-log
  chain and the loop back into the twin.
- `tests/` — `test_imulog_roundtrips.py` is the suite (marked `slow`); the rest run in a
  second. `.github/workflows/ci.yml` runs lint, both, and the JS equivalence test.
- `docs/EXPERIMENTS.md` write-ups, `docs/READING.md` literature with code cross-checks,
  `docs/CONVENTIONS.md` the documentation standard this file extends.

## Invariants — do not break these

- **Action column 0 is the RIGHT leg, column 1 the LEFT.** The twin's XML says so
  (`right_leg → joint_1 → servo_1`), and `servo_id.sim_side_columns` reads it from the
  model. Every left/right label is derived from that, never from a constant. *Scar:*
  a per-side identification shipped with the horns swapped, and a symmetric fixture
  could not see it — the guard now injects an asymmetric servo on the XML's column and
  fails if the constants disagree with the model.
- **`D[i]` marks the transition i→i+1 as a cut, not the state i.** Forward-validity
  guards cover offsets `0..h−1` to read a target at offset `h`; this is correct and
  was adjudicated once — do not re-litigate without a failing test.
- **`ServoModel.delay_ticks` counts CALLS.** The caller's rate sets the unit: 50 Hz in
  `GrowBotSim.step`, physics rate in the fixture. Express delays in ms and divide by
  the caller's dt. *Scar:* "2 ticks" once simulated 10 ms instead of 40 ms.
- **The bodies ship `condim="3"`.** Torsional and rolling friction are not in the
  solve at that setting — the XML's `0.1 0.1` are inert, and ×100 changes are
  bit-identical. Raising `condim` switches on a rolling coefficient 1000× MuJoCo's
  default that alone costs ~27 pts of yaw predictability. Never randomize a column
  you have not proven active.
- **`perturb()` edits a model loaded fresh per `GrowBotSim` instance.** DR cannot
  accumulate across resets here; keep it that way if you ever reuse a model object.
- **Real logs are never committed.** They are the maintainer's data; `.gitignore`
  covers them. Identification happens on the first half of a file, evaluation on the
  second — never on shared ticks.
- **`send_ok` means transmitted, not actuated.** A file can carry commands while the
  body reads motionless; the segmenter labels that `still` and `gap_report` refuses to
  identify a servo from it. *Scar:* a "walk" file that contained no walking was scored
  against the twin's walking floor and published as a −80 pt pitch gap.
- **The `act` verb glides, and the glide duration is not in the log.** A slew
  identified from a gesture capture is not the horn's limit; report it as
  undetermined until the per-act `ms` or the realized horn stream is logged.
- **The observation vector mixes channels with different lags.** Fused orientation
  trails the raw gyro by ~13 ms; `parse(ang_lead_ms=...)` aligns them. Observation
  delay and action delay are formally equivalent, so an identified servo delay is a
  lump until the sensor lag is removed.

## Publishing a number — the rules with scars

1. **A run's printed output is not evidence; the artifact is.** Before a figure is
   quoted anywhere, it resolves to `results/<name>.json`. Sweep every numeric token in
   a rewritten section and delete the ones that do not. *Scar:* an "801 ms median
   time-to-peak" reached a maintainer without any code that computed it.
2. **Means never travel alone.** Every quantity carries its per-seed values and spread;
   a verdict is *material* by the mean AND *resolved* only when the three seeds
   separate from nominal's three by more than the bar. Bars come from the nominal
   spread of the SAME quantity and horizon. *Scar:* a "45 %" headline had a per-seed
   range of 37–55 %, and its bar came from a different metric at a different horizon.
3. **Mechanism sentences carry the same mark as numbers.** "Teeters more", "falls
   more", "what moves is X" are claims; each needs its resolved/unresolved tag or it
   is deleted. *Scar:* the numbers survived two review rounds; the mechanism words did
   not.
4. **Pre-state the decision rule**, in the script and in the write-up, before the run.
   A threshold chosen after seeing the table is not a threshold.
5. **Refuse, do not caveat.** When a quantity cannot be measured — a servo on a
   motionless half, a slew on a gliding act — the artifact records the refusal and its
   reason instead of a number with a footnote. A number with a footnote still gets
   quoted.
6. **A correction is a measurement too.** The number you add to answer a reviewer is
   subject to every rule above. *Scar:* the round that fixed "means without spread"
   introduced a mean without spread.
7. **Corrections are recorded, never erased**, and a section built on a false premise
   is retracted in place with both defects named (see the coverage section).
8. **Adversarial review before publication.** Two independent, blind readers of the
   diff, with the raw files; only findings both confirm are fixed; at most two rounds.
   Four such reviews each found real defects that the hidden-secret suite could not:
   the suite tests the code against the author's assumptions about the data, and the
   defects lived in the assumptions.

## Building a new experiment — the workflow

1. Reuse the evaluation path (`horizon_within`, `rollout_error`, `evaluate_axes`,
   `servo_id.identify`); never reimplement scoring. Two implementations of the same
   math get an equivalence test.
2. Start from `body_params.py`'s honesty machinery: `seed_stat`, per-quantity bars,
   `resolved`, `argmin_interior`. A boundary argmin is a reported condition, not a
   footnote.
3. Include the null case as a hard check: a hidden *nominal* body or servo must
   identify to nominal, or the method is broken before the interesting cases run.
   Write the artifact first, then fail — a failed null case is a result.
4. Shared seeds share the initial condition only: `collect()` draws every tick for
   pushes, per mode in `Excitation`, per episode in `fresh()`, and once more when
   fallen, so streams desynchronize. Say so; do not call it a paired design.
5. Report both horizons, all three axes, per regime when regimes differ, and the
   twin floor next to every real number.
6. Write the section with the stated limits verbatim: this metric is forward-model
   *prediction*, not policy *transfer*; and every real number in this repository comes
   from ONE unit and ONE phone.

## Reading papers and repositories

A source enters `docs/READING.md` only with a cross-check against this code: what it
claims, the number that matters, and what does or does not transfer to a body with no
encoders. The two most useful checks so far were disagreements — a published horizon
ablation that backfired on 16 s of walking, and a bench-based actuator identification
this body cannot run as published.
