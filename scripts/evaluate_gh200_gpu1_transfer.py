#!/usr/bin/env python3
"""Evaluate the frozen GPU1 same-SKU transfer campaign without refitting."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.duration_v2_evaluation import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
