"""Execute one frozen pilot repeat as exact synchronized decode episodes."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from .accounting import active_bytes, kv_bytes_per_token
from .batch_doctor import balanced_prompt_lengths
from .decode_doctor import (
    _atomic_write_json,
    _normal_token_id,
    _wait_for_external_start_gate,
)
from .runtime_audit import (
    _cache_tensor_report,
    _locked_run,
    _weight_storage_report,
    validate_cache_dtype_contract,
)


def validate_episode_observations(
    observations: Iterable[Mapping[str, Any]],
    *,
    request_ids: list[str],
    measured_decode_tokens: int,
) -> dict[str, Any]:
    rows = list(observations)
    expected_steps = 1 + measured_decode_tokens
    expected_set = set(request_ids)
    reasons: list[str] = []
    if len(rows) != expected_steps:
        reasons.append(f"observed {len(rows)} steps; expected {expected_steps}")
    for step_id, row in enumerate(rows):
        counts = row.get("cumulative_output_tokens_by_request", {})
        useful = row.get("useful_tokens_by_request", {})
        finished = row.get("finished_by_request", {})
        if set(counts) != expected_set:
            reasons.append(f"step {step_id} request membership changed")
            continue
        expected_count = step_id + 1
        if any(counts[request_id] != expected_count for request_id in request_ids):
            reasons.append(f"step {step_id} cumulative token counts are not synchronized")
        if step_id > 0 and (
            set(useful) != expected_set
            or any(useful[request_id] != 1 for request_id in request_ids)
        ):
            reasons.append(f"step {step_id} did not add exactly one token per request")
        should_finish = step_id == measured_decode_tokens
        if any(bool(finished.get(request_id)) != should_finish for request_id in request_ids):
            reasons.append(f"step {step_id} finish flags violate the common barrier")
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
        reasons.append("step timestamps overlap or move backward")
    return {
        "schema_version": 1,
        "expected_steps": expected_steps,
        "observed_steps": len(rows),
        "metered_useful_tokens": (
            len(request_ids) * measured_decode_tokens if not reasons else None
        ),
        "qc_reasons": reasons,
        "qc_pass": not reasons,
    }


def _scheduler_snapshot(engine: Any) -> dict[str, Any]:
    schedulers = getattr(engine, "scheduler", None)
    if schedulers is None:
        return {"observable": False, "reason": "engine.scheduler is unavailable"}
    if not isinstance(schedulers, (list, tuple)):
        schedulers = [schedulers]
    rows: list[dict[str, Any]] = []
    for index, scheduler in enumerate(schedulers):
        preemptions = getattr(scheduler, "num_cumulative_preemption", None)
        swapped = getattr(scheduler, "swapped", None)
        try:
            swapped_count = len(swapped) if swapped is not None else None
        except TypeError:
            swapped_count = None
        rows.append(
            {
                "scheduler_index": index,
                "cumulative_preemptions": preemptions,
                "swapped_request_count": swapped_count,
            }
        )
    observable = bool(rows) and all(
        isinstance(row["cumulative_preemptions"], int)
        and isinstance(row["swapped_request_count"], int)
        for row in rows
    )
    return {"observable": observable, "schedulers": rows}


def validate_scheduler_delta(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    reasons: list[str] = []
    if not before.get("observable") or not after.get("observable"):
        reasons.append("scheduler preemption/swap counters are not observable")
    else:
        before_rows = before["schedulers"]
        after_rows = after["schedulers"]
        if len(before_rows) != len(after_rows):
            reasons.append("scheduler count changed")
        else:
            for left, right in zip(before_rows, after_rows):
                if right["cumulative_preemptions"] != left["cumulative_preemptions"]:
                    reasons.append("cumulative preemption count changed")
                if right["swapped_request_count"] != 0:
                    reasons.append("one or more requests are swapped")
    return {"qc_reasons": reasons, "qc_pass": not reasons}


def _observe_engine_step(
    engine: Any,
    *,
    step_id: int,
    phase: str,
    previous_counts: Mapping[str, int],
    prompt_tokens_by_request: Mapping[str, int],
) -> dict[str, Any]:
    start_ns = time.monotonic_ns()
    outputs = engine.step()
    end_ns = time.monotonic_ns()
    counts: dict[str, int] = {}
    finished: dict[str, bool] = {}
    useful: dict[str, int] = {}
    for output in outputs:
        if len(output.outputs) != 1:
            raise RuntimeError("pilot repeat requires one sampled sequence per request")
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
        "attended_history_tokens_by_request": (
            None
            if step_id == 0
            else {
                request_id: prompt_tokens + step_id - 1
                for request_id, prompt_tokens in prompt_tokens_by_request.items()
            }
        ),
    }


def _model_geometry(engine: Any, declared_kv_dtype: str) -> dict[str, Any]:
    model_config = getattr(engine, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        raise RuntimeError("cannot locate the Hugging Face model configuration")
    layers = int(hf_config.num_hidden_layers)
    attention_heads = int(hf_config.num_attention_heads)
    kv_heads = int(hf_config.num_key_value_heads)
    hidden_size = int(hf_config.hidden_size)
    configured_head_dim = getattr(hf_config, "head_dim", None)
    if configured_head_dim is None:
        if hidden_size % attention_heads:
            raise RuntimeError("hidden size is not divisible by the attention-head count")
        head_dim = hidden_size // attention_heads
    else:
        head_dim = int(configured_head_dim)
    bytes_per_element = 2 if declared_kv_dtype == "bf16" else 1
    return {
        "num_hidden_layers": layers,
        "num_attention_heads": attention_heads,
        "num_key_value_heads": kv_heads,
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "kv_bytes_per_element": bytes_per_element,
        "kv_bytes_per_historical_token": kv_bytes_per_token(
            layers, kv_heads, head_dim, bytes_per_element
        ),
    }


def run_pilot_repeat(
    *,
    campaign_lock: Path,
    run_id: str,
    gpu_memory_utilization: float,
    ready_file: Path,
    start_gate_file: Path,
    gate_timeout_seconds: float,
) -> dict[str, Any]:
    lock = json.loads(campaign_lock.read_text(encoding="utf-8"))
    run = _locked_run(lock, run_id)
    parameters = run["parameters"]
    locked_gpu_memory_utilization = parameters.get("gpu_memory_utilization")
    if locked_gpu_memory_utilization is not None and not math.isclose(
        float(locked_gpu_memory_utilization),
        gpu_memory_utilization,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "gpu_memory_utilization does not match the frozen run contract"
        )
    batch = int(parameters["target_batch"])
    measured = int(parameters["metered_decode_tokens_per_request"])
    target_mean = int(parameters["target_mean_attended_history_tokens"])
    minimum_episode_seconds = float(parameters["minimum_episode_decode_seconds"])
    minimum_repeat_seconds = float(parameters["minimum_decode_seconds_per_repeat"])
    prompt_lengths = balanced_prompt_lengths(
        target_mean_attended_history_tokens=target_mean,
        batch=batch,
        measured_decode_tokens=measured,
    )
    expected_backend = parameters["attention_backend"]
    if os.environ.get("VLLM_ATTENTION_BACKEND") != expected_backend:
        raise RuntimeError(f"VLLM_ATTENTION_BACKEND must be {expected_backend}")

    torch = importlib.import_module("torch")
    pynvml = importlib.import_module("pynvml")
    vllm = importlib.import_module("vllm")
    EngineArgs = getattr(vllm, "EngineArgs")
    LLMEngine = getattr(vllm, "LLMEngine")
    SamplingParams = getattr(vllm, "SamplingParams")
    if "vllm.engine.llm_engine" not in LLMEngine.__module__:
        raise RuntimeError("pilot repeat requires the V0 LLMEngine")
    declared_kv_dtype = parameters["kv_cache_dtype"]
    requested_kv_dtype = "auto" if declared_kv_dtype == "bf16" else declared_kv_dtype
    engine = LLMEngine.from_engine_args(
        EngineArgs(
            model=parameters["model"],
            revision=parameters["model_revision"],
            tokenizer_revision=parameters["model_revision"],
            dtype="bfloat16",
            kv_cache_dtype=requested_kv_dtype,
            seed=2027,
            max_model_len=max(prompt_lengths) + measured + 9,
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
        cache["dtypes"], declared_dtype=declared_kv_dtype
    )
    if not cache_contract["qc_pass"]:
        raise RuntimeError("resolved KV cache tensor dtype failed the frozen contract")
    weights = _weight_storage_report(engine)
    model_geometry = _model_geometry(engine, declared_kv_dtype)
    mechanism = active_bytes(
        weights["unique_storage_bytes"],
        model_geometry["kv_bytes_per_historical_token"],
        batch,
        target_mean,
    ).to_dict()
    external_gate = _wait_for_external_start_gate(
        ready_file=ready_file,
        start_gate_file=start_gate_file,
        timeout_seconds=gate_timeout_seconds,
    )

    tokenizer = engine.get_tokenizer()
    prompt_token_id = _normal_token_id(tokenizer)
    total_output_tokens = measured + 1
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        seed=2027,
        ignore_eos=True,
        min_tokens=total_output_tokens,
        max_tokens=total_output_tokens,
        detokenize=False,
    )
    episodes: list[dict[str, Any]] = []
    cumulative_decode_seconds = 0.0
    cumulative_useful_tokens = 0
    overall_reasons: list[str] = []
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        episode_id = 0
        while cumulative_decode_seconds < minimum_repeat_seconds:
            if episode_id >= 20:
                raise RuntimeError("repeat did not reach its duration target in 20 episodes")
            request_ids = [f"{run_id}-e{episode_id:02d}-q{index:03d}" for index in range(batch)]
            prompt_tokens_by_request = dict(zip(request_ids, prompt_lengths))
            for request_id, prompt_tokens in prompt_tokens_by_request.items():
                engine.add_request(
                    request_id,
                    {"prompt_token_ids": [prompt_token_id] * prompt_tokens},
                    sampling_params,
                )
            scheduler_before = _scheduler_snapshot(engine)
            observations: list[dict[str, Any]] = []
            previous_counts = {request_id: 0 for request_id in request_ids}
            bootstrap = _observe_engine_step(
                engine,
                step_id=0,
                phase="bootstrap-unmetered",
                previous_counts=previous_counts,
                prompt_tokens_by_request=prompt_tokens_by_request,
            )
            observations.append(bootstrap)
            if set(bootstrap["cumulative_output_tokens_by_request"]) != set(request_ids):
                raise RuntimeError("bootstrap did not admit the complete frozen batch")
            previous_counts = dict(bootstrap["cumulative_output_tokens_by_request"])
            if any(value != 1 for value in previous_counts.values()):
                raise RuntimeError("bootstrap did not produce exactly one token per request")

            torch.cuda.synchronize()
            counter_start_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            go_ns = time.monotonic_ns()
            for measured_index in range(measured):
                row = _observe_engine_step(
                    engine,
                    step_id=measured_index + 1,
                    phase="decode-metered",
                    previous_counts=previous_counts,
                    prompt_tokens_by_request=prompt_tokens_by_request,
                )
                observations.append(row)
                previous_counts = dict(row["cumulative_output_tokens_by_request"])
            torch.cuda.synchronize()
            done_ns = time.monotonic_ns()
            counter_end_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            scheduler_after = _scheduler_snapshot(engine)
            validation = validate_episode_observations(
                observations,
                request_ids=request_ids,
                measured_decode_tokens=measured,
            )
            scheduler_validation = validate_scheduler_delta(
                scheduler_before, scheduler_after
            )
            decode_seconds = (done_ns - go_ns) / 1e9
            episode_reasons = list(validation["qc_reasons"]) + list(
                scheduler_validation["qc_reasons"]
            )
            if decode_seconds < minimum_episode_seconds:
                episode_reasons.append(
                    f"decode duration {decode_seconds:.6f}s is below {minimum_episode_seconds:.6f}s"
                )
            if counter_end_mj <= counter_start_mj:
                episode_reasons.append("module cumulative energy counter did not advance")
            useful_tokens = batch * measured
            episodes.append(
                {
                    "episode_id": episode_id,
                    "request_ids": request_ids,
                    "prompt_tokens_by_request": prompt_tokens_by_request,
                    "boundary": {"go_monotonic_ns": go_ns, "done_monotonic_ns": done_ns},
                    "decode_seconds": decode_seconds,
                    "metered_useful_tokens": useful_tokens,
                    "module_counter_start_mj": counter_start_mj,
                    "module_counter_end_mj": counter_end_mj,
                    "module_counter_joules": (counter_end_mj - counter_start_mj) / 1000.0,
                    "scheduler_before": scheduler_before,
                    "scheduler_after": scheduler_after,
                    "scheduler_validation": scheduler_validation,
                    "observations": observations,
                    "validation": validation,
                    "qc_reasons": episode_reasons,
                    "qc_pass": not episode_reasons,
                }
            )
            overall_reasons.extend(
                f"episode {episode_id}: {reason}" for reason in episode_reasons
            )
            cumulative_decode_seconds += decode_seconds
            cumulative_useful_tokens += useful_tokens
            episode_id += 1
    finally:
        pynvml.nvmlShutdown()

    if cumulative_decode_seconds < minimum_repeat_seconds:
        overall_reasons.append("repeat did not reach the frozen decode-duration target")
    return {
        "schema_version": 1,
        "measurement": (
            "frozen-pilot-decode-repeat"
            if run["split"] in {"pilot", "placebo"}
            else "frozen-static-decode-repeat"
        ),
        "paper_candidate_measurement": True,
        "campaign_lock_sha256": lock["lock_sha256"],
        "run": {key: run[key] for key in ("run_id", "cell_id", "split", "repeat", "order")},
        "runtime": {
            "vllm_version": vllm.__version__,
            "engine_module": LLMEngine.__module__,
            "attention_backend": expected_backend,
            "weight_dtype": "bfloat16",
            "declared_kv_cache_dtype": declared_kv_dtype,
            "requested_kv_cache_dtype": requested_kv_dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
        },
        "geometry": {
            "batch": batch,
            "target_mean_attended_history_tokens": target_mean,
            "metered_decode_tokens_per_request": measured,
            "prompt_lengths": prompt_lengths,
            "episode_count": len(episodes),
            "cumulative_decode_seconds": cumulative_decode_seconds,
            "cumulative_metered_useful_tokens": cumulative_useful_tokens,
        },
        "model_geometry": model_geometry,
        "active_bytes": mechanism,
        "cache": {key: value for key, value in cache.items() if key != "tensors"},
        "cache_contract": cache_contract,
        "weights": {key: value for key, value in weights.items() if key != "inventory"},
        "external_start_gate": external_gate,
        "episodes": episodes,
        "qc_reasons": overall_reasons,
        "qc_pass": not overall_reasons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-lock", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--ready-file", required=True, type=Path)
    parser.add_argument("--start-gate-file", required=True, type=Path)
    parser.add_argument("--gate-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)
    report = run_pilot_repeat(
        campaign_lock=args.campaign_lock,
        run_id=args.run_id,
        gpu_memory_utilization=args.gpu_memory_utilization,
        ready_file=args.ready_file,
        start_gate_file=args.start_gate_file,
        gate_timeout_seconds=args.gate_timeout_seconds,
    )
    _atomic_write_json(args.output_json, report)
    summary = {
        "qc_pass": report["qc_pass"],
        "qc_reasons": report["qc_reasons"],
        "campaign_lock_sha256": report["campaign_lock_sha256"],
        "run": report["run"],
        "runtime": report["runtime"],
        "geometry": report["geometry"],
        "model_geometry": report["model_geometry"],
        "active_bytes": report["active_bytes"],
        "cache": report["cache"],
        "weights": report["weights"],
        "episodes": [
            {
                key: episode[key]
                for key in (
                    "episode_id",
                    "decode_seconds",
                    "metered_useful_tokens",
                    "module_counter_joules",
                    "qc_pass",
                    "qc_reasons",
                )
            }
            for episode in report["episodes"]
        ],
        "output_json": str(args.output_json),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report["qc_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
