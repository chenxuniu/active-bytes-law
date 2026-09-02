#!/usr/bin/env python3
"""Verify the content-addressed GH200 evaluation release."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.evaluation_release import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
