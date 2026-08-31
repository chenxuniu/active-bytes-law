#!/usr/bin/env python3
"""Audit a frozen pilot cell's resolved vLLM runtime."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.runtime_audit import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
