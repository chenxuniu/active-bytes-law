#!/usr/bin/env python3
"""Run the non-paper offline FP8 KV-cache calibration doctor."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.kv_calibration import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
