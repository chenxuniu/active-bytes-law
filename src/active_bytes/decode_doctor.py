"""Run and validate a minimal vLLM pure-decode boundary doctor."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


def validate_doctor_observations(
    observations: Iterable[Mapping[str, Any]],
    *,
    bootstrap_tokens: int,
    measured_decode_tokens: int,
) -> dict[str, Any]:
    """Validate cumulative output growth across a bootstrap/decode boundary."""

    rows = list(observations)
    expected_rows = 1 + measured_decode_tokens
    reasons: list[str] = []
    if len(rows) != expected_rows:
        reasons.append(f"observed {len(rows)} steps; expected {expected_rows}")
    expected_counts = [bootstrap_tokens + index for index in range(expected_rows)]
    observed_counts = [row.get("cumulative_output_tokens") for row in rows]
    if observed_counts != expected_counts:
        reasons.append(
            f"cumulative output counts {observed_counts} != expected {expected_counts}"
        )
    step_ids = [row.get("step_id") for row in rows]
    if step_ids != list(range(expected_rows)):
        reasons.append("step IDs are not contiguous from zero")
    request_ids = [row.get("request_id") for row in rows]
    if request_ids and any(value != request_ids[0] for value in request_ids):
        reasons.append("request membership changed")
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
    if rows:
        if bool(rows[0].get("finished")):
            reasons.append("request finished during the bootstrap step")
        if not bool(rows[-1].get("finished")):
            reasons.append("request did not finish after the final measured step")
    return {
        "schema_version": 1,
        "qc_pass": not reasons,
        "qc_reasons": reasons,
        "expected_steps": expected_rows,
        "observed_steps": len(rows),
        "expected_cumulative_output_tokens": expected_counts,
        "observed_cumulative_output_tokens": observed_counts,
        "metered_useful_tokens": measured_decode_tokens if not reasons else None,
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _normal_token_id(tokenizer: Any) -> int:
    token_ids = tokenizer.encode(" measurement", add_special_tokens=False)
    if not token_ids:
        raise RuntimeError("tokenizer produced no ordinary token for doctor prompt")
    return int(token_ids[-1])


def _single_request_output(outputs: list[Any], request_id: str) -> Any:
    matching = [output for output in outputs if output.request_id == request_id]
    if len(matching) != 1:
        raise RuntimeError(
            f"engine step returned {len(matching)} outputs for request {request_id}"
        )
    output = matching[0]
    if len(output.outputs) != 1:
        raise RuntimeError("doctor requires exactly one sampled sequence")
    return output


def _wait_for_external_start_gate(
    *,
    ready_file: Path | None,
    start_gate_file: Path | None,
    timeout_seconds: float,
) -> dict[str, Any] | None:
    if ready_file is None and start_gate_file is None:
        return None
    if ready_file is None or start_gate_file is None:
        raise ValueError("ready-file and start-gate-file must be supplied together")
    if timeout_seconds <= 0:
        raise ValueError("gate timeout must be positive")
    ready_ns = time.monotonic_ns()
    _atomic_write_json(
        ready_file,
        {
            "schema_version": 1,
            "state": "ENGINE_READY",
            "ready_monotonic_ns": ready_ns,
        },
    )
    deadline = time.monotonic() + timeout_seconds
    while not start_gate_file.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"start gate {start_gate_file} did not appear within {timeout_seconds}s"
            )
        time.sleep(0.05)
    released_ns = time.monotonic_ns()
    return {
        "ready_file": str(ready_file),
        "start_gate_file": str(start_gate_file),
        "ready_monotonic_ns": ready_ns,
        "released_monotonic_ns": released_ns,
        "wait_seconds": (released_ns - ready_ns) / 1e9,
    }


def run_vllm_doctor(
    *,
    model: str,
    model_revision: str | None,
    prompt_tokens: int,
    measured_decode_tokens: int,
    gpu_memory_utilization: float,
    seed: int,
    enforce_eager: bool,
    minimum_counter_duration_seconds: float,
    ready_file: Path | None,
    start_gate_file: Path | None,
    gate_timeout_seconds: float,
) -> dict[str, Any]:
    """Execute one bootstrap step followed by exact pure-decode engine steps."""

    if prompt_tokens <= 0 or measured_decode_tokens <= 0:
        raise ValueError("prompt and measured decode token counts must be positive")
    if not 0 < gpu_memory_utilization < 1:
        raise ValueError("gpu_memory_utilization must be between zero and one")
    if minimum_counter_duration_seconds < 0:
        raise ValueError("minimum counter duration cannot be negative")

    torch = importlib.import_module("torch")
    pynvml = importlib.import_module("pynvml")
    vllm = importlib.import_module("vllm")
    EngineArgs = getattr(vllm, "EngineArgs")
    LLMEngine = getattr(vllm, "LLMEngine")
    SamplingParams = getattr(vllm, "SamplingParams")

    if "vllm.engine.llm_engine" not in LLMEngine.__module__:
        raise RuntimeError(
            f"doctor requires the V0 LLMEngine; received {LLMEngine.__module__}"
        )

    bootstrap_tokens = 1
    total_output_tokens = bootstrap_tokens + measured_decode_tokens
    max_model_len = prompt_tokens + total_output_tokens + 8
    engine_args = EngineArgs(
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
        max_num_batched_tokens=max_model_len,
        max_num_seqs=1,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        swap_space=0,
        cpu_offload_gb=0,
        enforce_eager=enforce_eager,
        disable_async_output_proc=True,
        disable_log_stats=True,
        speculative_config=None,
    )
    engine = LLMEngine.from_engine_args(engine_args)
    tokenizer = engine.get_tokenizer()
    prompt_token_id = _normal_token_id(tokenizer)
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
    request_id = "decode-doctor-0"
    engine.add_request(request_id, prompt, params)
    external_gate = _wait_for_external_start_gate(
        ready_file=ready_file,
        start_gate_file=start_gate_file,
        timeout_seconds=gate_timeout_seconds,
    )

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        observations: list[dict[str, Any]] = []

        bootstrap_start_ns = time.monotonic_ns()
        bootstrap_output = _single_request_output(engine.step(), request_id)
        bootstrap_end_ns = time.monotonic_ns()
        bootstrap_count = len(bootstrap_output.outputs[0].token_ids)
        observations.append(
            {
                "step_id": 0,
                "phase": "bootstrap-unmetered",
                "request_id": request_id,
                "monotonic_start_ns": bootstrap_start_ns,
                "monotonic_end_ns": bootstrap_end_ns,
                "cumulative_output_tokens": bootstrap_count,
                "useful_tokens_this_step": bootstrap_count,
                "finished": bool(bootstrap_output.finished),
            }
        )
        if bootstrap_count != bootstrap_tokens or bootstrap_output.finished:
            raise RuntimeError(
                "bootstrap gate failed: first engine step must produce exactly one token"
            )

        torch.cuda.synchronize()
        decode_ready_ns = time.monotonic_ns()
        counter_read_start_ns = time.monotonic_ns()
        counter_start_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        go_ns = time.monotonic_ns()

        previous_count = bootstrap_count
        for measured_index in range(measured_decode_tokens):
            step_start_ns = time.monotonic_ns()
            request_output = _single_request_output(engine.step(), request_id)
            step_end_ns = time.monotonic_ns()
            cumulative_count = len(request_output.outputs[0].token_ids)
            observations.append(
                {
                    "step_id": measured_index + 1,
                    "phase": "decode-metered",
                    "request_id": request_id,
                    "monotonic_start_ns": step_start_ns,
                    "monotonic_end_ns": step_end_ns,
                    "cumulative_output_tokens": cumulative_count,
                    "useful_tokens_this_step": cumulative_count - previous_count,
                    "finished": bool(request_output.finished),
                }
            )
            previous_count = cumulative_count

        torch.cuda.synchronize()
        decode_done_ns = time.monotonic_ns()
        counter_end_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        counter_read_end_ns = time.monotonic_ns()
    finally:
        pynvml.nvmlShutdown()

    validation = validate_doctor_observations(
        observations,
        bootstrap_tokens=bootstrap_tokens,
        measured_decode_tokens=measured_decode_tokens,
    )
    token_boundary_qc_pass = validation["qc_pass"]
    decode_seconds = (decode_done_ns - go_ns) / 1e9
    boundary_reasons: list[str] = []
    ordered = (
        decode_ready_ns
        <= counter_read_start_ns
        <= go_ns
        <= decode_done_ns
        <= counter_read_end_ns
    )
    if not ordered:
        boundary_reasons.append("decode/counter boundary timestamps are out of order")
    if counter_end_mj < counter_start_mj:
        boundary_reasons.append("cumulative energy counter decreased")
    elif counter_end_mj == counter_start_mj:
        boundary_reasons.append(
            "cumulative energy counter did not advance; the interval is below "
            "the validated counter-observation duration"
        )
    if decode_seconds < minimum_counter_duration_seconds:
        boundary_reasons.append(
            f"decode interval {decode_seconds:.6f}s is below required counter "
            f"duration {minimum_counter_duration_seconds:.6f}s"
        )
    if any(row["useful_tokens_this_step"] != 1 for row in observations[1:]):
        boundary_reasons.append("a metered step did not produce exactly one useful token")

    validation["qc_reasons"].extend(boundary_reasons)
    validation["qc_pass"] = validation["qc_pass"] and not boundary_reasons
    module_energy_joules = (counter_end_mj - counter_start_mj) / 1000.0
    return {
        "schema_version": 1,
        "measurement": "vllm-decode-boundary-doctor",
        "non_paper_measurement": True,
        "runtime": {
            "vllm_version": vllm.__version__,
            "engine_module": LLMEngine.__module__,
            "model": model,
            "model_revision": model_revision,
            "dtype": "bfloat16",
            "kv_cache_dtype": "auto",
            "enforce_eager": enforce_eager,
            "prefix_caching": False,
            "chunked_prefill": False,
            "speculation": False,
            "cpu_offload_gb": 0,
            "swap_space_gb": 0,
        },
        "geometry": {
            "prompt_tokens": prompt_tokens,
            "bootstrap_tokens": bootstrap_tokens,
            "metered_decode_tokens": measured_decode_tokens,
            "requested_output_tokens": total_output_tokens,
            "batch": 1,
        },
        "boundary": {
            "decode_ready_monotonic_ns": decode_ready_ns,
            "counter_read_start_monotonic_ns": counter_read_start_ns,
            "go_monotonic_ns": go_ns,
            "decode_done_monotonic_ns": decode_done_ns,
            "counter_read_end_monotonic_ns": counter_read_end_ns,
        },
        "external_start_gate": external_gate,
        "energy": {
            "scope": "module-on-validated-GH200-stack",
            "counter_start_mj": counter_start_mj,
            "counter_end_mj": counter_end_mj,
            "module_energy_joules": module_energy_joules,
            "decode_seconds": decode_seconds,
            "minimum_counter_duration_seconds": minimum_counter_duration_seconds,
            "module_joules_per_token_non_paper": (
                module_energy_joules / measured_decode_tokens
            ),
        },
        "observations": observations,
        "validation": validation,
        "token_boundary_qc_pass": token_boundary_qc_pass,
        "energy_counter_observed": counter_end_mj > counter_start_mj,
        "qc_pass": validation["qc_pass"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-revision")
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--measured-decode-tokens", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--runtime-mode", choices=("eager", "graph"), default="eager"
    )
    parser.add_argument("--minimum-counter-duration-seconds", type=float, default=0.0)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--start-gate-file", type=Path)
    parser.add_argument("--gate-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--print-summary-only", action="store_true")
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)
    report = run_vllm_doctor(
        model=args.model,
        model_revision=args.model_revision,
        prompt_tokens=args.prompt_tokens,
        measured_decode_tokens=args.measured_decode_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        enforce_eager=args.runtime_mode == "eager",
        minimum_counter_duration_seconds=args.minimum_counter_duration_seconds,
        ready_file=args.ready_file,
        start_gate_file=args.start_gate_file,
        gate_timeout_seconds=args.gate_timeout_seconds,
    )
    _atomic_write_json(args.output_json, report)
    printed = report
    if args.print_summary_only:
        printed = {
            "qc_pass": report["qc_pass"],
            "token_boundary_qc_pass": report["token_boundary_qc_pass"],
            "energy_counter_observed": report["energy_counter_observed"],
            "runtime": report["runtime"],
            "geometry": report["geometry"],
            "energy": report["energy"],
            "qc_reasons": report["validation"]["qc_reasons"],
            "output_json": str(args.output_json),
        }
    print(json.dumps(printed, indent=2, sort_keys=True))
    return 0 if report["qc_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
