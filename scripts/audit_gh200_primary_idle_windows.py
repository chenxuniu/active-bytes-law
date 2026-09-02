#!/usr/bin/env python3
"""Audit pre/post-decode power samples for the frozen GH200 primary runs."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.idle_window_audit import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
