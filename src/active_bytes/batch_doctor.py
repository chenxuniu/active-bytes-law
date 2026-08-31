"""Verify a synchronized multi-request bootstrap and pure-decode barrier."""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from .decode_doctor import _atomic_write_json, _normal_token_id


def validate_batch_observations(
    observations: Iterable[Mapping[str, Any]],
    *,
    batch: int,
    measured_decode_tokens: int,
) -> dict[str, Any]:
    rows = list(observations)
    request_ids = [f"batch-doctor-{index}" for index in range(batch)]
    expected_steps = 1 + measured_decode_tokens
    reasons: list[str] = []
    if len(rows) != expected_steps:
        reasons.append(f"observed {len(rows)} steps; expected {expected_steps}")
    for step_id, row in enumerate(rows):
        counts = row.get("cumulative_output_tokens_by_request", {})
        finished = row.get("finished_by_request", {})
        if set(counts) != set(request_ids):
            reasons.append(f"step {step_id} request membership is not the full batch")
            continue
        expected_count = step_id + 1
        if any(counts[request_id] != expected_count for request_id in request_ids):
            reasons.append(
                f"step {step_id} cumulative counts are not all {expected_count}"
            )
        should_finish = step_id == measured_decode_tokens
        if any(bool(finished.get(request_id)) != should_finish for request_id in request_ids):
            reasons.append(f"step {step_id} finished flags violate the common barrier")
        if step_id > 0:
            useful = row.get("useful_tokens_by_request", {})
            if set(useful) != set(request_ids) or any(
                useful[request_id] != 1 for request_id in request_ids
            ):
                reasons.append(f"step {step_id} did not add one token per request")
    timestamps = [
        (row.get("monotonic_start_ns"), row.get("monotonic_end_ns")) for row in rows
    ]
    if any(
        not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end <= start
        for start, end in timestamps
    ):
        reasons.append("one or more step timestamps are invalid")
    elif any(
        timestamps[index][1] > timestamps[index + 1][0]
        for index in range(len(timestamps) - 1)
    ):
        reasons.append("step timestamps overlap or are out of order")
    return {
        "schema_version": 1,
        "qc_pass": not reasons,
        "qc_reasons": reasons,
        "batch": batch,
        "expected_steps": expected_steps,
        "observed_steps": len(rows),
        "request_ids": request_ids,
    }


def _observe_step(
    engine: Any,
    *,
    step_id: int,
    phase: str,
    previous_counts: Mapping[str, int],
    prompt_tokens: int,
) -> dict[str, Any]:
    start_ns = time.monotonic_ns()
    outputs = engine.step()
    end_ns = time.monotonic_ns()
    counts: dict[str, int] = {}
    finished: dict[str, bool] = {}
    useful: dict[str, int] = {}
    for output in outputs:
        if len(output.outputs) != 1:
            raise RuntimeError("batch doctor requires one sampled sequence per request")
        count = len(output.outputs[0].token_ids)
        counts[output.request_id] = count
        finished[output.request_id] = bool(output.finished)
        useful[output.request_id] = count - previous_counts.get(output.request_id, 0)
    return {
        "step_id": step_id,
        "phase": phase,
        "monotonic_start_ns": start_ns,
        "monotonic_end_ns": end_ns,
        "cumulative_output_tokens_by_request": counts,
        "useful_tokens_by_request": useful,
        "finished_by_request": finished,
        "expected_attended_history_tokens": (
            None if step_id == 0 else prompt_tokens + step_id - 1
        ),
    }


def run_batch_doctor(
    *,
    model: str,
    model_revision: str,
    prompt_tokens: int,
    batch: int,
    measured_decode_tokens: int,
    gpu_memory_utilization: float,
    seed: int,
) -> dict[str, Any]:
    if prompt_tokens <= 0 or batch <= 0 or measured_decode_tokens <= 0:
        raise ValueError("prompt, batch, and decode counts must be positive")
    if not 0 < gpu_memory_utilization < 1:
        raise ValueError("gpu_memory_utilization must be between zero and one")
    vllm = importlib.import_module("vllm")
    EngineArgs = getattr(vllm, "EngineArgs")
    LLMEngine = getattr(vllm, "LLMEngine")
    SamplingParams = getattr(vllm, "SamplingParams")
    if "vllm.engine.llm_engine" not in LLMEngine.__module__:
        raise RuntimeError("batch doctor requires the V0 LLMEngine")

    total_output_tokens = 1 + measured_decode_tokens
    max_model_len = prompt_tokens + total_output_tokens + 8
    max_num_batched_tokens = batch * prompt_tokens
    engine = LLMEngine.from_engine_args(
        EngineArgs(
            model=model,
            revision=model_revision,
            tokenizer_revision=model_revision,
            dtype="bfloat16",
            kv_cache_dtype="auto",
            seed=seed,
            max_model_len=max_model_len,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_batched_tokens=max_num_batched_tokens,
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
    prompt_token_id = _normal_token_id(engine.get_tokenizer())
    prompt = {"prompt_token_ids": [prompt_token_id] * prompt_tokens}
    params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        seed=seed,
        ignore_eos=True,
        min_tokens=total_output_tokens,
        max_tokens=total_output_tokens,
        detokenize=False,
    )
    request_ids = [f"batch-doctor-{index}" for index in range(batch)]
    for request_id in request_ids:
        engine.add_request(request_id, prompt, params)

    observations: list[dict[str, Any]] = []
    previous_counts = {request_id: 0 for request_id in request_ids}
    bootstrap = _observe_step(
        engine,
        step_id=0,
        phase="bootstrap-unmetered",
        previous_counts=previous_counts,
        prompt_tokens=prompt_tokens,
    )
    observations.append(bootstrap)
    counts = bootstrap["cumulative_output_tokens_by_request"]
    bootstrap_pass = set(counts) == set(request_ids) and all(
        counts[request_id] == 1 for request_id in request_ids
    )
    if bootstrap_pass:
        previous_counts = dict(counts)
        for measured_index in range(measured_decode_tokens):
            row = _observe_step(
                engine,
                step_id=measured_index + 1,
                phase="decode-metered",
                previous_counts=previous_counts,
                prompt_tokens=prompt_tokens,
            )
            observations.append(row)
            previous_counts = dict(row["cumulative_output_tokens_by_request"])

    validation = validate_batch_observations(
        observations,
        batch=batch,
        measured_decode_tokens=measured_decode_tokens,
    )
    return {
        "schema_version": 1,
        "measurement": "vllm-batch-barrier-doctor",
        "non_paper_measurement": True,
        "runtime": {
            "vllm_version": vllm.__version__,
            "engine_module": LLMEngine.__module__,
            "model": model,
            "model_revision": model_revision,
            "dtype": "bfloat16",
            "kv_cache_dtype": "auto",
            "enforce_eager": False,
            "prefix_caching": False,
            "chunked_prefill": False,
            "speculation": False,
            "cpu_offload_gb": 0,
            "swap_space_gb": 0,
            "max_num_batched_tokens": max_num_batched_tokens,
            "max_num_seqs": batch,
        },
        "geometry": {
            "prompt_tokens": prompt_tokens,
            "batch": batch,
            "bootstrap_tokens_per_request": 1,
            "metered_decode_tokens_per_request": measured_decode_tokens,
            "metered_useful_tokens": batch * measured_decode_tokens,
        },
        "observations": observations,
        "validation": validation,
        "qc_pass": validation["qc_pass"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--measured-decode-tokens", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)
    report = run_batch_doctor(
        model=args.model,
        model_revision=args.model_revision,
        prompt_tokens=args.prompt_tokens,
        batch=args.batch,
        measured_decode_tokens=args.measured_decode_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    _atomic_write_json(args.output_json, report)
    summary = {
        "qc_pass": report["qc_pass"],
        "qc_reasons": report["validation"]["qc_reasons"],
        "runtime": report["runtime"],
        "geometry": report["geometry"],
        "observed_steps": report["validation"]["observed_steps"],
        "output_json": str(args.output_json),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report["qc_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
