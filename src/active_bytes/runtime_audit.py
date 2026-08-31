"""Audit the resolved vLLM cache tensors and resident weight storage."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from .batch_doctor import balanced_prompt_lengths
from .decode_doctor import _atomic_write_json


def validate_cache_dtype_contract(
    observed_dtypes: Iterable[str], *, declared_dtype: str
) -> dict[str, Any]:
    observed = sorted(set(observed_dtypes))
    accepted = {
        "bf16": {"bfloat16", "torch.bfloat16"},
        "fp8_e4m3": {
            "float8_e4m3fn",
            "torch.float8_e4m3fn",
            "uint8",
            "torch.uint8",
        },
    }
    if declared_dtype not in accepted:
        raise ValueError(f"unsupported declared KV cache dtype {declared_dtype!r}")
    reasons: list[str] = []
    if not observed:
        reasons.append("no GPU KV cache tensors were discovered")
    elif any(dtype not in accepted[declared_dtype] for dtype in observed):
        reasons.append(
            f"resolved KV tensor dtypes {observed} do not match {declared_dtype}"
        )
    return {
        "declared_kv_cache_dtype": declared_dtype,
        "observed_kv_tensor_dtypes": observed,
        "qc_reasons": reasons,
        "qc_pass": not reasons,
    }


def _flatten_tensors(value: Any, torch: Any) -> Iterable[Any]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _flatten_tensors(child, torch)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _flatten_tensors(child, torch)


def _known_cache_roots(worker: Any) -> list[tuple[str, Any]]:
    roots: list[tuple[str, Any]] = []
    for name in ("gpu_cache", "kv_cache", "kv_caches"):
        value = getattr(worker, name, None)
        if value is not None:
            roots.append((f"driver_worker.{name}", value))
    cache_engines = getattr(worker, "cache_engine", None)
    if isinstance(cache_engines, (list, tuple)):
        for index, cache_engine in enumerate(cache_engines):
            for name in ("gpu_cache", "kv_cache", "kv_caches"):
                value = getattr(cache_engine, name, None)
                if value is not None:
                    roots.append((f"cache_engine[{index}].{name}", value))
    elif cache_engines is not None:
        for name in ("gpu_cache", "kv_cache", "kv_caches"):
            value = getattr(cache_engines, name, None)
            if value is not None:
                roots.append((f"cache_engine.{name}", value))
    model_runner = getattr(worker, "model_runner", None)
    for name in ("gpu_cache", "kv_cache", "kv_caches"):
        value = getattr(model_runner, name, None)
        if value is not None:
            roots.append((f"model_runner.{name}", value))
    return roots


def _cache_tensor_report(engine: Any, torch: Any) -> dict[str, Any]:
    executor = getattr(engine, "model_executor", None)
    worker = getattr(executor, "driver_worker", None)
    roots = _known_cache_roots(worker)
    discovered: list[Any] = []
    seen: set[int] = set()
    for _, root in roots:
        for tensor in _flatten_tensors(root, torch):
            identity = id(tensor)
            if identity not in seen:
                seen.add(identity)
                discovered.append(tensor)
    tensors = discovered
    rows = [
        {
            "dtype": str(tensor.dtype),
            "device_type": tensor.device.type,
            "shape": list(tensor.shape),
            "numel": tensor.numel(),
            "element_size": tensor.element_size(),
            "logical_nbytes": tensor.numel() * tensor.element_size(),
        }
        for tensor in tensors
    ]
    return {
        "tensor_count": len(rows),
        "gpu_tensor_count": sum(row["device_type"] == "cuda" for row in rows),
        "logical_nbytes": sum(row["logical_nbytes"] for row in rows),
        "dtypes": sorted({row["dtype"] for row in rows}),
        "inspected_roots": [name for name, _ in roots],
        "tensors": rows,
    }


def _weight_storage_report(engine: Any) -> dict[str, Any]:
    executor = getattr(engine, "model_executor", None)
    worker = getattr(executor, "driver_worker", None)
    model_runner = getattr(worker, "model_runner", None)
    model = getattr(model_runner, "model", None)
    if model is None:
        raise RuntimeError("cannot locate the loaded model on the V0 driver worker")

    storage_ids: dict[tuple[str, int], str] = {}
    storage_sizes: dict[str, int] = {}
    inventory: list[dict[str, Any]] = []
    parameters = list(model.named_parameters())
    parameter_names = {name for name, _ in parameters}
    tensors = parameters + list(model.named_buffers())
    for name, tensor in tensors:
        storage = tensor.untyped_storage()
        physical_key = (tensor.device.type, int(storage.data_ptr()))
        storage_id = storage_ids.setdefault(
            physical_key, f"storage-{len(storage_ids):06d}"
        )
        storage_nbytes = int(storage.nbytes())
        previous = storage_sizes.setdefault(storage_id, storage_nbytes)
        if previous != storage_nbytes:
            raise RuntimeError("one physical storage reported conflicting sizes")
        inventory.append(
            {
                "name": name,
                "kind": "parameter" if name in parameter_names else "buffer",
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "physical_storage_id": storage_id,
                "storage_nbytes": storage_nbytes,
            }
        )
    canonical = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    return {
        "tensor_count": len(inventory),
        "unique_storage_count": len(storage_sizes),
        "unique_storage_bytes": sum(storage_sizes.values()),
        "inventory_sha256": hashlib.sha256(canonical).hexdigest(),
        "inventory": inventory,
    }


def _locked_run(lock: Mapping[str, Any], run_id: str) -> Mapping[str, Any]:
    matching = [run for run in lock.get("run_order", []) if run.get("run_id") == run_id]
    if len(matching) != 1:
        raise ValueError(f"run ID {run_id!r} does not occur exactly once in the lock")
    return matching[0]


def resolve_runtime_contract(
    parameters: Mapping[str, Any],
    *,
    compatibility_attention_backend: str | None = None,
    compatibility_kv_cache_dtype: str | None = None,
) -> dict[str, Any]:
    """Resolve either the frozen contract or a clearly non-paper compatibility probe.

    A compatibility probe may borrow a frozen cell's model and geometry, but it
    must never be represented as execution of that cell.  Requiring both
    overrides prevents an accidental one-factor mutation of a preregistered run.
    """
    compatibility_values = (
        compatibility_attention_backend,
        compatibility_kv_cache_dtype,
    )
    if any(value is not None for value in compatibility_values) and not all(
        value is not None for value in compatibility_values
    ):
        raise ValueError(
            "compatibility attention backend and KV cache dtype must be supplied together"
        )
    compatibility_mode = all(value is not None for value in compatibility_values)
    if compatibility_mode:
        declared_kv_dtype = str(compatibility_kv_cache_dtype)
        attention_backend = str(compatibility_attention_backend)
    else:
        declared_kv_dtype = str(parameters["kv_cache_dtype"])
        attention_backend = str(parameters["attention_backend"])
    if declared_kv_dtype not in {"bf16", "fp8_e4m3"}:
        raise ValueError(f"unsupported declared KV cache dtype {declared_kv_dtype!r}")
    return {
        "compatibility_mode": compatibility_mode,
        "attention_backend": attention_backend,
        "declared_kv_cache_dtype": declared_kv_dtype,
        "requested_kv_cache_dtype": (
            "auto" if declared_kv_dtype == "bf16" else declared_kv_dtype
        ),
    }


def run_runtime_audit(
    *,
    campaign_lock: Path,
    run_id: str,
    gpu_memory_utilization: float,
    compatibility_attention_backend: str | None = None,
    compatibility_kv_cache_dtype: str | None = None,
) -> dict[str, Any]:
    if not 0 < gpu_memory_utilization < 1:
        raise ValueError("gpu_memory_utilization must be between zero and one")
    lock = json.loads(campaign_lock.read_text(encoding="utf-8"))
    run = _locked_run(lock, run_id)
    parameters = run["parameters"]
    batch = parameters["target_batch"]
    measured = parameters["metered_decode_tokens_per_request"]
    target_mean = parameters["target_mean_attended_history_tokens"]
    prompts = balanced_prompt_lengths(
        target_mean_attended_history_tokens=target_mean,
        batch=batch,
        measured_decode_tokens=measured,
    )

    contract = resolve_runtime_contract(
        parameters,
        compatibility_attention_backend=compatibility_attention_backend,
        compatibility_kv_cache_dtype=compatibility_kv_cache_dtype,
    )
    attention_backend = os.environ.get("VLLM_ATTENTION_BACKEND")
    expected_backend = contract["attention_backend"]
    if attention_backend != expected_backend:
        raise RuntimeError(
            f"VLLM_ATTENTION_BACKEND must be {expected_backend}, got {attention_backend!r}"
        )

    torch = importlib.import_module("torch")
    vllm = importlib.import_module("vllm")
    EngineArgs = getattr(vllm, "EngineArgs")
    LLMEngine = getattr(vllm, "LLMEngine")
    if "vllm.engine.llm_engine" not in LLMEngine.__module__:
        raise RuntimeError("runtime audit requires the V0 LLMEngine")
    declared_kv_dtype = contract["declared_kv_cache_dtype"]
    requested_kv_dtype = contract["requested_kv_cache_dtype"]
    engine = LLMEngine.from_engine_args(
        EngineArgs(
            model=parameters["model"],
            revision=parameters["model_revision"],
            tokenizer_revision=parameters["model_revision"],
            dtype="bfloat16",
            kv_cache_dtype=requested_kv_dtype,
            seed=2027,
            max_model_len=max(prompts) + measured + 9,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_batched_tokens=sum(prompts),
            max_num_seqs=batch,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            swap_space=0,
            cpu_offload_gb=0,
            enforce_eager=False,
            disable_async_output_proc=True,
            disable_log_stats=True,
            speculative_config=None,
        )
    )
    cache = _cache_tensor_report(engine, torch)
    cache_contract = validate_cache_dtype_contract(
        cache["dtypes"], declared_dtype=declared_kv_dtype
    )
    weights = _weight_storage_report(engine)
    reasons = list(cache_contract["qc_reasons"])
    if cache["gpu_tensor_count"] != cache["tensor_count"]:
        reasons.append("one or more discovered KV cache tensors are not on the GPU")
    report = {
        "schema_version": 1,
        "measurement": (
            "vllm-backend-compatibility-audit"
            if contract["compatibility_mode"]
            else "vllm-frozen-runtime-audit"
        ),
        "non_paper_measurement": True,
        "frozen_run_execution": not contract["compatibility_mode"],
        "campaign_lock_sha256": lock["lock_sha256"],
        "runtime": {
            "vllm_version": vllm.__version__,
            "engine_module": LLMEngine.__module__,
            "attention_backend": attention_backend,
            "weight_dtype": "bfloat16",
            "requested_kv_cache_dtype": requested_kv_dtype,
        },
        "geometry": {
            "batch": batch,
            "target_mean_attended_history_tokens": target_mean,
            "metered_decode_tokens_per_request": measured,
            "prompt_lengths": prompts,
        },
        "cache": cache,
        "cache_contract": cache_contract,
        "weights": weights,
        "qc_reasons": reasons,
        "qc_pass": not reasons,
    }
    if contract["compatibility_mode"]:
        report["geometry_template_run_id"] = run_id
        report["compatibility_contract"] = {
            "purpose": "backend-and-cache-dtype-feasibility-only",
            "may_enter_paper_outcomes": False,
            "requires_new_preregistration_before_energy_measurement": True,
            "candidate_attention_backend": expected_backend,
            "candidate_kv_cache_dtype": declared_kv_dtype,
        }
    else:
        report["run_id"] = run_id
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-lock", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--compatibility-attention-backend")
    parser.add_argument(
        "--compatibility-kv-cache-dtype", choices=("bf16", "fp8_e4m3")
    )
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)
    report = run_runtime_audit(
        campaign_lock=args.campaign_lock,
        run_id=args.run_id,
        gpu_memory_utilization=args.gpu_memory_utilization,
        compatibility_attention_backend=args.compatibility_attention_backend,
        compatibility_kv_cache_dtype=args.compatibility_kv_cache_dtype,
    )
    _atomic_write_json(args.output_json, report)
    summary = {
        "qc_pass": report["qc_pass"],
        "qc_reasons": report["qc_reasons"],
        "campaign_lock_sha256": report["campaign_lock_sha256"],
        "runtime": report["runtime"],
        "geometry": report["geometry"],
        "cache": {key: value for key, value in report["cache"].items() if key != "tensors"},
        "cache_contract": report["cache_contract"],
        "weights": {key: value for key, value in report["weights"].items() if key != "inventory"},
        "output_json": str(args.output_json),
    }
    if report["frozen_run_execution"]:
        summary["run_id"] = report["run_id"]
    else:
        summary["geometry_template_run_id"] = report["geometry_template_run_id"]
        summary["compatibility_contract"] = report["compatibility_contract"]
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report["qc_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
