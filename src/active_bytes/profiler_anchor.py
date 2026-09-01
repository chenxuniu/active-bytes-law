"""Run one frozen, NVTX-scoped, single-iteration V1 traffic anchor."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from .accounting import active_bytes
from .batch_doctor import balanced_prompt_lengths
from .decode_doctor import _atomic_write_json, _normal_token_id
from .pilot_repeat import (
    _model_geometry,
    _observe_engine_step,
    _scheduler_snapshot,
    validate_episode_observations,
    validate_scheduler_delta,
)
from .runtime_audit import (
    _cache_tensor_report,
    _locked_run,
    _weight_storage_report,
    validate_cache_dtype_contract,
)


def run_profiler_anchor(
    *, campaign_lock: Path, run_id: str, gpu_memory_utilization: float
) -> dict[str, Any]:
    lock = json.loads(campaign_lock.read_text(encoding="utf-8"))
    run = _locked_run(lock, run_id)
    parameters = run["parameters"]
    if run["split"] != "profiler-anchor":
        raise ValueError("V1 anchor runner accepts only profiler-anchor runs")
    if int(parameters["metered_decode_tokens_per_request"]) != 1:
        raise ValueError("V1 range contract requires exactly one decode iteration")
    if parameters.get("profiler_replay_mode") != "app-range":
        raise ValueError("V1 profiler contract requires application-range replay")
    locked_utilization = float(parameters["gpu_memory_utilization"])
    if not math.isclose(
        locked_utilization, gpu_memory_utilization, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("gpu_memory_utilization does not match the frozen lock")
    expected_backend = str(parameters["attention_backend"])
    if os.environ.get("VLLM_ATTENTION_BACKEND") != expected_backend:
        raise RuntimeError(f"VLLM_ATTENTION_BACKEND must be {expected_backend}")

    batch = int(parameters["target_batch"])
    target_mean = int(parameters["target_mean_attended_history_tokens"])
    prompt_lengths = balanced_prompt_lengths(
        target_mean_attended_history_tokens=target_mean,
        batch=batch,
        measured_decode_tokens=1,
    )

    torch = importlib.import_module("torch")
    vllm = importlib.import_module("vllm")
    EngineArgs = getattr(vllm, "EngineArgs")
    LLMEngine = getattr(vllm, "LLMEngine")
    SamplingParams = getattr(vllm, "SamplingParams")
    if "vllm.engine.llm_engine" not in LLMEngine.__module__:
        raise RuntimeError("V1 anchor requires the frozen V0 LLMEngine")

    engine = LLMEngine.from_engine_args(
        EngineArgs(
            model=parameters["model"],
            revision=parameters["model_revision"],
            tokenizer_revision=parameters["model_revision"],
            dtype="bfloat16",
            kv_cache_dtype="auto",
            seed=2027,
            max_model_len=max(prompt_lengths) + 10,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_batched_tokens=sum(prompt_lengths),
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
        cache["dtypes"], declared_dtype="bf16"
    )
    if not cache_contract["qc_pass"]:
        raise RuntimeError("resolved KV cache dtype failed the BF16 contract")
    weights = _weight_storage_report(engine)
    geometry = _model_geometry(engine, "bf16")
    mechanism = active_bytes(
        weights["unique_storage_bytes"],
        geometry["kv_bytes_per_historical_token"],
        batch,
        target_mean,
    ).to_dict()

    tokenizer = engine.get_tokenizer()
    prompt_token_id = _normal_token_id(tokenizer)
    sampling = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        seed=2027,
        ignore_eos=True,
        min_tokens=2,
        max_tokens=2,
        detokenize=False,
    )
    request_ids = [f"{run_id}-q{index:03d}" for index in range(batch)]
    prompts = dict(zip(request_ids, prompt_lengths))
    for request_id, prompt_tokens in prompts.items():
        engine.add_request(
            request_id,
            {"prompt_token_ids": [prompt_token_id] * prompt_tokens},
            sampling,
        )

    scheduler_before = _scheduler_snapshot(engine)
    previous_counts = {request_id: 0 for request_id in request_ids}
    bootstrap = _observe_engine_step(
        engine,
        step_id=0,
        phase="bootstrap-unprofiled",
        previous_counts=previous_counts,
        prompt_tokens_by_request=prompts,
    )
    previous_counts = dict(bootstrap["cumulative_output_tokens_by_request"])
    if set(previous_counts) != set(request_ids) or any(
        value != 1 for value in previous_counts.values()
    ):
        raise RuntimeError("bootstrap did not produce one token for every request")

    torch.cuda.synchronize()
    range_name = str(parameters["nvtx_range"])
    torch.cuda.nvtx.range_push(range_name)
    start_ns = time.monotonic_ns()
    try:
        measured = _observe_engine_step(
            engine,
            step_id=1,
            phase="decode-profiled",
            previous_counts=previous_counts,
            prompt_tokens_by_request=prompts,
        )
        torch.cuda.synchronize()
        end_ns = time.monotonic_ns()
    finally:
        torch.cuda.nvtx.range_pop()

    scheduler_after = _scheduler_snapshot(engine)
    validation = validate_episode_observations(
        [bootstrap, measured], request_ids=request_ids, measured_decode_tokens=1
    )
    scheduler_validation = validate_scheduler_delta(
        scheduler_before, scheduler_after
    )
    reasons = list(validation["qc_reasons"]) + list(
        scheduler_validation["qc_reasons"]
    )
    useful_tokens = batch
    return {
        "schema_version": 1,
        "measurement": "gh200-v1-single-iteration-profiler-anchor",
        "paper_candidate_measurement": True,
        "energy_measurement": False,
        "campaign_lock_sha256": lock["lock_sha256"],
        "run": {
            key: run[key]
            for key in ("run_id", "cell_id", "split", "repeat", "order")
        },
        "runtime": {
            "vllm_version": vllm.__version__,
            "engine_module": LLMEngine.__module__,
            "attention_backend": expected_backend,
            "weight_dtype": "bfloat16",
            "kv_cache_dtype": "bfloat16",
            "gpu_memory_utilization": gpu_memory_utilization,
        },
        "profile_range": {
            "name": range_name,
            "replay_mode": parameters["profiler_replay_mode"],
            "decode_iterations": 1,
            "metered_useful_tokens": useful_tokens,
            "start_monotonic_ns": start_ns,
            "end_monotonic_ns": end_ns,
            "duration_seconds_under_profiler": (end_ns - start_ns) / 1e9,
        },
        "geometry": {
            "batch": batch,
            "target_mean_attended_history_tokens": target_mean,
            "prompt_lengths": prompt_lengths,
        },
        "model_geometry": geometry,
        "active_bytes": mechanism,
        "uncorrected_obligation_totals": {
            "read_bytes": mechanism["active_bytes_read"] * useful_tokens,
            "read_write_bytes": mechanism["active_bytes_read_write"]
            * useful_tokens,
            "kv_write_bytes": mechanism["kv_write_bytes_per_token"]
            * useful_tokens,
        },
        "weights": {
            key: value for key, value in weights.items() if key != "inventory"
        },
        "cache": {key: value for key, value in cache.items() if key != "tensors"},
        "cache_contract": cache_contract,
        "observations": [bootstrap, measured],
        "validation": validation,
        "scheduler_validation": scheduler_validation,
        "qc_reasons": reasons,
        "qc_pass": not reasons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-lock", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu-memory-utilization", required=True, type=float)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)
    report = run_profiler_anchor(
        campaign_lock=args.campaign_lock,
        run_id=args.run_id,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    _atomic_write_json(args.output_json, report)
    print(
        json.dumps(
            {
                "qc_pass": report["qc_pass"],
                "qc_reasons": report["qc_reasons"],
                "run": report["run"],
                "profile_range": report["profile_range"],
                "geometry": report["geometry"],
                "active_bytes": report["active_bytes"],
                "output_json": str(args.output_json),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["qc_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
