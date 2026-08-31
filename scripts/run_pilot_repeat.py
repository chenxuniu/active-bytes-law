#!/usr/bin/env python3
"""Run one locked pilot repeat from a source checkout."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.pilot_repeat import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
