#!/usr/bin/env python3
"""Evaluate the released Qwen2.5-14B holdout without refitting."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.model_replication_evaluation import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
