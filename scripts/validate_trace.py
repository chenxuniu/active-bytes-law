#!/usr/bin/env python3
"""Validate one static decode-only trace from a source checkout."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.validation import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
