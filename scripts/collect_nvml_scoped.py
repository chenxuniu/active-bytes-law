#!/usr/bin/env python3
"""Collect and audit scoped NVML power from a source checkout."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.nvml_scoped import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
