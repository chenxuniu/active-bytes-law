#!/usr/bin/env python3
"""Run one frozen full FP8 KV-only calibration."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.full_calibration import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
