"""The provenance block every artifact under results/ carries.

A number without the commit, the library versions and the seeds that produced it cannot
be reproduced to the decimal, and a stale artifact cannot be told from a fresh one. The
shipped forward-model weights were once found to have been trained on a data file that
was no longer the one on disk; nothing in the JSON could say so. This block can.
"""
from __future__ import annotations
import datetime as _dt
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _version(dist):
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def provenance(seeds=None, **extra):
    """{git_commit, git_dirty, python, numpy, torch, mujoco, argv, seeds, timestamp, ...}.

    `seeds` is whatever the caller randomised on (an int, a list, or a dict of named
    seeds); it is stored as given. `extra` keys are merged verbatim so a script can add
    what only it knows (a data file, a hypothesis grid size).
    """
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "numpy": _version("numpy"),
        "torch": _version("torch"),
        "mujoco": _version("mujoco"),
        "argv": list(sys.argv),
        "seeds": seeds,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }
