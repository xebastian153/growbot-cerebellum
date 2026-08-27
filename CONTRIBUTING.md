# Contributing

This repository is a set of experiments, each one script, each with a written-up
result that resolves to a file under `results/`. Read `AGENTS.md` first: it holds the
invariants and the publishing rules, each with the mistake that produced it.

## Setup

```bash
uv venv --python 3.11 .venv
VIRTUAL_ENV=.venv uv pip install --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
  -r requirements.txt
```

`requirements.txt` is the direct set; `requirements.lock` is the exact freeze that produced
the shipped artifacts (use it when a number has to be reproduced to the decimal). CPU only.
Node 18 or newer for the JS test. Then regenerate the gitignored twin data — the four
`sim/growbot_sim.py` commands at the top of the README's "Reproduce" block.

## The one command

```bash
.venv/bin/python imulog.py          # the round-trip suite, ~5 min, must print PASS on every line
```

It generates a 600 s twin session with a hidden servo and hidden sensor secrets, pushes it
through every accepted log dialect, the segmenter, per-side identification and the
sensor-side estimators, and asserts the secrets come back. A change that touches
`imulog.py`, `servo_id.py`, `sensor_id.py` or `sim/` is not done until it is green, before
and after. `imulog.py --selftest` is the same thing spelled out; `imulog.py FILE` is the
preflight on a real log and writes nothing.

The JS runner has its own equivalence test against the trained PyTorch net, float32
tolerance:

```bash
cd forward-model && node test_forward.mjs      # PASS
```

`export_js.py` regenerates `forward-model/forward_85mm.json` and the reference vectors from
`data/train.npz`; training is seeded, so two runs on the same data are byte-identical.

## `--help` never runs anything

Every script parses its arguments before it opens a file for writing or starts a
collection: `python <script>.py --help` prints the question the script answers and what
it writes, and exits. Keep it that way — a script whose import has side effects has once
retrained the shipped weights during a documentation sweep. `ruff check .` is the linter
(`ruff.toml`; lint only, nothing reformats). At the commit that introduced the config it
reports 18 findings — unused imports, empty f-strings, unused locals — left for a cleanup
commit rather than mixed into an unrelated change.

## Adding an experiment

The workflow is in `AGENTS.md` ("Building a new experiment"); the rules the write-up has
to satisfy are in `docs/CONVENTIONS.md`. In short: one script, a docstring that states the
question, a working `--help`, output to `results/<name>.json` and a run log in
`results/logs/<name>.txt`, a `Reproduce:` line in `docs/EXPERIMENTS.md`, and every figure in
the text traceable to that artifact. Decision rules are written before the numbers. Negative
results get the same prominence as positive ones.

Commits are work units with conventional prefixes; the message states the finding with its
numbers. Nothing that a listed command regenerates is tracked (`data/*.npz`).

## Real logs

The real-data scripts (`real_log_report.py`, `real2sim.py`, `coverage.py`,
`identification_ablation.py`, `gesture_id.py`) run on `?imulog=1` sessions from the GrowBot
web app, which records one file per walk when the page is opened with `?imulog=1`. The
format, `growbot-imulog-1`, is a single JSON object: a `header` (units, mount, gait, gain,
calibration trims, build), an `imu` array at the phone's native rate (device orientation in
degrees plus body rates) and a `pose` array with each servo command at its send time,
carrying `send_ok` — which means transmitted, not actuated. `imulog.py` also accepts the
one-row-per-line JSONL dialect and a CSV fallback described in its docstring; `imulog.py
FILE` tells you whether a file is usable before anything is computed from it.

This repository ships **no real logs**. They are the maintainers' data, gitignored by
name pattern (`imu-walk-*.json`, `SEND-*.json`), and every result computed from one says
which file it came from. To run those scripts, obtain a session from the upstream project
and place it in the repository root.
