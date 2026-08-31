#!/usr/bin/env python3
"""Align a frozen pilot repeat with scoped telemetry."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.pilot_alignment import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
