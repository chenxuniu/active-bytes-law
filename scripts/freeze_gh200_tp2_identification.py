#!/usr/bin/env python3
"""Freeze TP=2 coefficients and the disjoint calibration envelope."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.tp2_identification import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
