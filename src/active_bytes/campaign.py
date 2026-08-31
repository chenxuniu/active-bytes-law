"""Deterministic expansion and locking of experiment campaigns."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


ALGORITHM_VERSION = "active-bytes-campaign-v2"
ALLOWED_SPLITS = {
    "pilot",
    "coefficient-fit",
    "residual-calibration",
    "evaluation",
    "architecture-holdout",
    "placebo",
    "dynamic",
    "weight-treatment",
    "profiler-anchor",
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _slug(value: Any) -> str:
    text = str(value).lower().replace(".", "p")
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:32] or "value"


def _axis_combinations(axes: Mapping[str, list[Any]]) -> Iterable[dict[str, Any]]:
    names = sorted(axes)
    if not names:
        yield {}
        return
    values: list[list[Any]] = []
    for name in names:
        axis = axes[name]
        if not isinstance(axis, list) or not axis:
            raise ValueError(f"axis {name!r} must be a non-empty array")
        values.append(axis)
    for combination in itertools.product(*values):
        yield dict(zip(names, combination))


def _cells_from_block(block: Mapping[str, Any], defaults: Mapping[str, Any]) -> list[dict[str, Any]]:
    block_id = block.get("block_id")
    split = block.get("split")
    if not isinstance(block_id, str) or not block_id:
        raise ValueError("every block needs a non-empty block_id")
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"block {block_id!r} has unsupported split {split!r}")
    constants = block.get("constants", {})
    if not isinstance(constants, Mapping):
        raise ValueError(f"block {block_id!r} constants must be an object")

    generated: list[dict[str, Any]] = []
    if "cells" in block:
        if "axes" in block:
            raise ValueError(f"block {block_id!r} cannot define both cells and axes")
        source_cells = block["cells"]
        if not isinstance(source_cells, list) or not source_cells:
            raise ValueError(f"block {block_id!r} cells must be a non-empty array")
        combinations = source_cells
    else:
        combinations = list(_axis_combinations(block.get("axes", {})))

    for combination in combinations:
        if not isinstance(combination, Mapping):
            raise ValueError(f"block {block_id!r} contains a non-object cell")
        parameters = dict(defaults)
        parameters.update(constants)
        parameters.update(combination)
        repetitions = parameters.pop("repetitions", None)
        if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions <= 0:
            raise ValueError(f"block {block_id!r} needs a positive repetitions value")
        explicit_id = parameters.pop("cell_id", None)
        condition_signature = sha256_json(parameters)
        signature = {"split": split, "parameters": parameters}
        digest = sha256_json(signature)[:10]
        if explicit_id is None:
            cell_id = f"{_slug(block_id)}-{digest}"
        elif not isinstance(explicit_id, str) or not explicit_id:
            raise ValueError(f"block {block_id!r} has an invalid cell_id")
        else:
            cell_id = _slug(explicit_id)
        generated.append(
            {
                "cell_id": cell_id,
                "block_id": block_id,
                "split": split,
                "repetitions": repetitions,
                "parameters": parameters,
                "cell_signature_sha256": sha256_json(signature),
                "condition_signature_sha256": condition_signature,
            }
        )
    return generated


def expand_campaign(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != 1:
        raise ValueError("campaign schema_version must be 1")
    campaign_id = config.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise ValueError("campaign_id must be a non-empty string")
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    defaults = config.get("defaults", {})
    randomize_cell_order = config.get("randomize_cell_order", True)
    blocks = config.get("blocks")
    if not isinstance(defaults, Mapping):
        raise ValueError("defaults must be an object")
    if not isinstance(randomize_cell_order, bool):
        raise ValueError("randomize_cell_order must be a boolean")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("blocks must be a non-empty array")

    cells: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, Mapping):
            raise ValueError("each block must be an object")
        cells.extend(_cells_from_block(block, defaults))

    seen_cells: dict[str, str] = {}
    seen_signatures: dict[str, str] = {}
    for cell in cells:
        cell_id = cell["cell_id"]
        signature = cell["cell_signature_sha256"]
        condition_signature = cell["condition_signature_sha256"]
        if cell_id in seen_cells:
            raise ValueError(f"duplicate cell_id {cell_id!r}")
        if (
            condition_signature in seen_signatures
            and seen_signatures[condition_signature] != cell["split"]
        ):
            raise ValueError("the same condition signature appears in multiple splits")
        seen_cells[cell_id] = signature
        seen_signatures[condition_signature] = cell["split"]

    base = list(cells)
    if randomize_cell_order:
        random.Random(seed).shuffle(base)
    maximum_repeats = max(cell["repetitions"] for cell in cells)
    run_order: list[dict[str, Any]] = []
    order = 0
    seen_runs: set[str] = set()
    for repeat in range(1, maximum_repeats + 1):
        eligible = [cell for cell in base if cell["repetitions"] >= repeat]
        if eligible:
            offset = (repeat - 1) % len(eligible)
            eligible = eligible[offset:] + eligible[:offset]
        for cell in eligible:
            run_id = f"ab1-{_slug(cell['split'])}-{cell['cell_id']}-r{repeat:02d}"
            if run_id in seen_runs:
                raise ValueError(f"duplicate run_id {run_id!r}")
            seen_runs.add(run_id)
            run_order.append(
                {
                    "order": order,
                    "run_id": run_id,
                    "cell_id": cell["cell_id"],
                    "split": cell["split"],
                    "repeat": repeat,
                    "parameters": cell["parameters"],
                }
            )
            order += 1

    lock: dict[str, Any] = {
        "schema_version": 1,
        "algorithm_version": ALGORITHM_VERSION,
        "campaign_id": campaign_id,
        "seed": seed,
        "cell_order_policy": (
            "seeded-shuffle" if randomize_cell_order else "declared-order-with-repeat-rotation"
        ),
        "source_sha256": sha256_json(config),
        "cell_count": len(cells),
        "run_count": len(run_order),
        "cells": cells,
        "run_order": run_order,
    }
    lock["lock_sha256"] = sha256_json(lock)
    return lock


def write_lock(path: Path, lock: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen lock: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(lock, indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(f"{hashlib.sha256(payload.encode()).hexdigest()}  {path.name}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    lock = expand_campaign(config)
    write_lock(args.output, lock)
    print(
        f"froze {lock['cell_count']} cells and {lock['run_count']} runs: "
        f"{args.output} ({lock['lock_sha256']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
