"""Where the experiment scripts read and write, anchored on the repository, not the cwd.

    ROOT     the repository (the directory that holds the scripts, data/, results/)
    DATA     ROOT / "data"     — the gitignored twin streams (README's four commands)
    RESULTS  ROOT / "results"  — one JSON per experiment, the source of every published number
    LOGS     RESULTS / "logs"  — the mirrored stdout of the run that wrote the JSON

Every root script used to spell "data/train.npz" and "results/<name>.json" relative to the
cwd, so a documented command run from any other directory failed to find the data or wrote
its artifact into the wrong tree, silently.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
LOGS = RESULTS / "logs"


def under_root(path) -> Path:
    """A path from the command line: absolute stays as given, relative is taken from ROOT."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
