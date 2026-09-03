#!/usr/bin/env python3
"""Freeze the prespecified Qwen2.5-14B duration-model identification."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.model_replication import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
