#!/usr/bin/env python3
"""Compare two non-paper FP8 KV calibration-doctor artifacts."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.calibration_compare import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
