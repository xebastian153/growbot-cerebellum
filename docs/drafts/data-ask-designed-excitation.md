# Draft data request — one 3–5 minute log with designed excitation

Status: draft, not sent. Format: `growbot-imulog-1` (`?imulog=1`), exactly as the app writes
it today: `header` + `imu[]` + `pose[]`, `imu_fields` = seq, t_ms, ax, ay, az, rate_alpha,
rate_beta, rate_gamma, ori_alpha, ori_beta, ori_gamma; `pose_fields` = seq, t_ms, l, r,
send_ok. No new fields are needed for this ask.

## Why

Every unresolved identification in xebastian153/growbot-cerebellum shares one condition:
periodic walking under-excites the servo. Excitation is necessary, not shown sufficient — in
the twin, 300 s of mixed excitation with the correct body model in the grid still leaves a
hidden delay-5 servo's delay set at the whole grid (`com_id`, 4 of 4 counted seeds), so the
test below can fail for that reason too. On the 16 s real walk the delay is
determined only to [2 … 6] ticks (40–120 ms) and the slew to [2.0, 3.0] rad/s, the split
halves disagree, and the twin reproduces that outcome with the correct servo in the grid.
The twin fixture that determines both to one grid step uses 8 minutes of mixed
excitation — steps of different sizes, holds, slow and fast sweeps; 8 minutes is the
round-trip suite's figure, not a human session's. The repository's own data ask is 3–5
minutes. Walking is one frequency at one amplitude.

## The recording

Robot on the floor it usually walks on, phone mounted as usual, battery fresh. One
session, 3–5 minutes, the app logging as for a walk. The pose sequence, sent as ordinary
`l`/`r` targets (degrees, 90 = neutral), each block repeated twice:

1. **Still**: neutral (90, 90) held for 30 s at the start and 30 s at the end.
2. **Steps, one leg at a time**: from neutral, `l` to 90±10, hold 1 s, back to 90, hold 1 s;
   then ±20, ±30, ±40. Same for `r` with `l` at 90. Small and large steps are what separate
   slew from deadband.
3. **Chirp, one leg at a time**: `l` swinging ±25° about 90 with the period going from 2 s
   down to 0.3 s over 20 s; `r` at 90. Then the same on `r`. A sweep that crosses the
   horn's own rate limit is what pins the slew.
4. **Both legs, alternating** (the walking gait) for 20 s, then **both legs in phase** for
   20 s: the alternating block decides left/right attribution by its roll signature (the
   pitch signature is the mean of the two commands and does not change under an l/r swap).
5. **Holds at angle**: (60, 60), (120, 120), (60, 120), (120, 60), each held 3 s.

If the app can log the time each target was sent to the servo, `t_ms` on the pose row
already carries it; nothing else is required. If a later logger version adds the horn's
actual angle per row, the glide of the `act` verb becomes a measurement instead of an
identification — but this ask does not depend on that.

## What comes back

The same day-of-log chain the repository documents (`imulog.py` preflight, `sensor_id.py`,
`gap_report.py --servo-id`), plus the per-side identification, with the determined sets
before and after. The claim to be tested is stated in advance: with this excitation the
delay set narrows to at most three adjacent ticks and the split halves agree. If they do
not, either the servo is outside the model family or the one-step score cannot separate it
even under this excitation (the `com_id` result above) — which of the two is then the
next question, and that is a result too.
