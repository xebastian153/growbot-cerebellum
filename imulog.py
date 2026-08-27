"""Preflight a GrowBot ?imulog=1 session, or run the round-trip suite.

    python imulog.py session.json     preflight: units, rates, clock, jitter, per-segment
                                      still physics; exit 0 on PASS, 1 on FAIL; writes nothing
    python imulog.py                  the round-trip suite (same as --selftest): a 600 s twin
                                      fixture with a hidden servo through the JSONL, CSV and
                                      growbot-imulog-1 dialects, the segmenter, per-side
                                      identification and the sensor-side secrets; ~5 min

The parser, preflight, segmenter and fixture live in `growbot_cerebellum.imulog`; the
suite is `tests/test_imulog_roundtrips.py` (pytest, marked `slow`) and this command runs
exactly that file, so there is one set of assertions, not two.
"""
from __future__ import annotations
import argparse
from pathlib import Path

from growbot_cerebellum.imulog import run_preflight

SUITE = Path(__file__).resolve().parent / "tests" / "test_imulog_roundtrips.py"


def main():
    ap = argparse.ArgumentParser(
        description="Parser for GrowBot ?imulog=1 sessions. With a FILE: run the preflight on it "
                    "(units, rates, clock, jitter, per-segment still physics) and exit 0 on PASS, "
                    "1 on FAIL; nothing is written. With no FILE (or --selftest): run the "
                    "round-trip suite -- a 600 s twin fixture with a hidden servo through the "
                    "JSONL, CSV and growbot-imulog-1 dialects, the segmenter, per-side "
                    "identification and the sensor-side secrets; ~5 min, writes only under pytest's "
                    "temporary directory. The suite must be green before and after any change to "
                    "growbot_cerebellum/.")
    ap.add_argument("log", nargs="?", metavar="FILE", help="session file to preflight")
    ap.add_argument("--selftest", action="store_true",
                    help="run the round-trip suite (the default when no FILE is given)")
    args = ap.parse_args()
    if args.log is not None and args.selftest:
        ap.error("give a FILE to preflight or --selftest, not both")
    if args.log is not None:                 # imulog.py <file> = standalone preflight
        raise SystemExit(0 if run_preflight(args.log) else 1)
    import pytest
    raise SystemExit(pytest.main(["-q", "-s", "-p", "no:cacheprovider", str(SUITE)]))


if __name__ == "__main__":
    main()
