"""The shipped JS runner against the shipped reference vectors, through node when it is
installed (skipped cleanly otherwise). This is the PR payload's own equivalence test."""
from __future__ import annotations
import shutil
import subprocess
from pathlib import Path

import pytest

FM = Path(__file__).resolve().parent.parent / "forward-model"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_runner_matches_reference_vectors():
    r = subprocess.run(["node", "test_forward.mjs"], cwd=FM, capture_output=True, text=True, timeout=120)
    print(r.stdout)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS" in r.stdout
