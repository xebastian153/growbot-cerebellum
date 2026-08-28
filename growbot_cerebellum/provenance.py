"""The provenance block every script writes into its artifact under results/.

A number without the commit, the library versions and the seeds that produced it cannot
be reproduced to the decimal, and a stale artifact cannot be told from a fresh one. The
shipped forward-model weights were once found to have been trained on a data file that
was no longer the one on disk; nothing in the JSON could say so. This block can.

Every writer has called it since the package was introduced; artifacts committed before
that were not regenerated (expensive, or dependent on real logs that are not in the
repository) and carry no `provenance` key. An artifact without the key predates it.
`git_commit` and `git_dirty` are `null` when git cannot answer (no binary, no repository,
timeout) -- never a value that could be mistaken for a clean checkout.
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
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    digests = _data_digests(extra.get("data"))
    if digests:
        extra["data_sha256"] = digests
    return {
        "git_commit": commit,
        "git_dirty": None if commit is None or status is None else bool(status),
        "python": platform.python_version(),
        "numpy": _version("numpy"),
        "torch": _version("torch"),
        "mujoco": _version("mujoco"),
        "argv": list(sys.argv),
        "seeds": seeds,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }


def _data_digests(data):
    """sha256 (first 16 hex) of every existing file named in `data` (a path or a dict of
    paths), so an artifact says which twin stream it was computed on — the stream is
    CPU-dependent, and a version pin alone does not identify it."""
    import hashlib
    paths = data.values() if isinstance(data, dict) else [data] if isinstance(data, str) else []
    out = {}
    for path in paths:
        f = ROOT / path if not Path(path).is_absolute() else Path(path)
        if isinstance(path, str) and f.is_file():
            out[path] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    return out

