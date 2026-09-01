"""Audit a calibrated FP8-KV checkpoint in the frozen vLLM image."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

from .batch_doctor import _observe_step, validate_batch_observations
from .decode_doctor import _atomic_write_json, _normal_token_id
from .full_calibration import contract_sha256
from .runtime_audit import (
    _cache_tensor_report,
    _kv_scale_report,
    _weight_storage_report,
    validate_cache_dtype_contract,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_serialized_checkpoint(
    checkpoint: Path,
    calibration_report: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    required = (
        "config.json",
        "model.safetensors.index.json",
        "calibration-contract.json",
        "recipe.yaml",
    )
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        reasons.append(f"checkpoint is missing required files: {missing}")
        return {
            "qc_pass": False,
            "qc_reasons": reasons,
            "missing_files": missing,
        }

    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    index = json.loads(
        (checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    embedded_contract = json.loads(
        (checkpoint / "calibration-contract.json").read_text(encoding="utf-8")
    )
    quantization = config.get("quantization_config", {})
    kv_scheme = quantization.get("kv_cache_scheme", {})
    expected_scheme = embedded_contract.get("recipe", {}).get("kv_cache", {})
    observed_scheme = {
        key: kv_scheme.get(key)
        for key in ("num_bits", "type", "strategy", "symmetric", "dynamic", "observer")
    }
    if quantization.get("quant_method") != "compressed-tensors":
        reasons.append("checkpoint quantization method is not compressed-tensors")
    if quantization.get("quantization_status") != "frozen":
        reasons.append("checkpoint quantization status is not frozen")
    if observed_scheme != expected_scheme:
        reasons.append("serialized KV-cache scheme differs from the frozen contract")

    non_kv_quantization: list[str] = []
    for group_name, group in quantization.get("config_groups", {}).items():
        for field in ("weights", "input_activations", "output_activations"):
            if group.get(field) is not None:
                non_kv_quantization.append(f"{group_name}.{field}")
    if non_kv_quantization:
        reasons.append("checkpoint config contains non-KV quantization schemes")

    weight_names = set(index.get("weight_map", {}))
    k_scale_names = sorted(name for name in weight_names if name.endswith(".k_scale"))
    v_scale_names = sorted(name for name in weight_names if name.endswith(".v_scale"))
    expected_layers = embedded_contract.get("expected_model_invariants", {}).get(
        "attention_layers"
    )
    if len(k_scale_names) != expected_layers or len(v_scale_names) != expected_layers:
        reasons.append("checkpoint index does not contain one K/V scale pair per layer")

    manifest = calibration_report.get("checkpoint", {}).get("files", [])
    size_mismatches = []
    for row in manifest:
        path = checkpoint / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            size_mismatches.append(row["path"])
    if size_mismatches:
        reasons.append("one or more checkpoint files differ from manifest sizes")

    expected_contract_hash = calibration_report.get("contract", {}).get(
        "canonical_sha256"
    )
    observed_contract_hash = contract_sha256(embedded_contract)
    if observed_contract_hash != expected_contract_hash:
        reasons.append("embedded contract does not match the calibration report")

    return {
        "qc_pass": not reasons,
        "qc_reasons": reasons,
        "quantization_method": quantization.get("quant_method"),
        "quantization_status": quantization.get("quantization_status"),
        "kv_cache_scheme": observed_scheme,
        "non_kv_quantization": non_kv_quantization,
        "k_scale_tensor_count": len(k_scale_names),
        "v_scale_tensor_count": len(v_scale_names),
        "k_scale_tensor_names": k_scale_names,
        "v_scale_tensor_names": v_scale_names,
        "manifest_file_count": len(manifest),
        "size_mismatches": size_mismatches,
        "embedded_contract_canonical_sha256": observed_contract_hash,
    }


def _run_decode_smoke(engine: Any, vllm: Any, *, seed: int) -> dict[str, Any]:
    SamplingParams = getattr(vllm, "SamplingParams")
    request_id = "checkpoint-load-audit-0"
    prompt_tokens = 32
    measured_decode_tokens = 8
    total_output_tokens = 1 + measured_decode_tokens
    prompt_token_id = _normal_token_id(engine.get_tokenizer())
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
    engine.add_request(
        request_id,
        {"prompt_token_ids": [prompt_token_id] * prompt_tokens},
        params,
    )
    prompt_map = {request_id: prompt_tokens}
    observations = []
    previous = {request_id: 0}
    for step_id in range(total_output_tokens):
        row = _observe_step(
            engine,
            step_id=step_id,
            phase="bootstrap-unmetered" if step_id == 0 else "decode-smoke",
            previous_counts=previous,
            prompt_tokens_by_request=prompt_map,
        )
        observations.append(row)
        previous = dict(row["cumulative_output_tokens_by_request"])
    validation = validate_batch_observations(
        observations,
        batch=1,
        measured_decode_tokens=measured_decode_tokens,
    )
    return {
        "prompt_tokens": prompt_tokens,
        "bootstrap_tokens": 1,
        "measured_decode_tokens": measured_decode_tokens,
        "observed_steps": len(observations),
        "validation": validation,
        "observations": observations,
    }


def run_checkpoint_load_audit(
    *,
    checkpoint: Path,
    calibration_report_path: Path,
    expected_calibration_report_sha256: str,
    gpu_memory_utilization: float,
    seed: int,
    inference_image_digest: str | None,
) -> dict[str, Any]:
    if not 0 < gpu_memory_utilization < 1:
        raise ValueError("gpu memory utilization must be between zero and one")
    observed_report_hash = _sha256(calibration_report_path)
    if observed_report_hash != expected_calibration_report_sha256:
        raise ValueError("full calibration report SHA-256 does not match")
    calibration_report = json.loads(
        calibration_report_path.read_text(encoding="utf-8")
    )
    serialized = validate_serialized_checkpoint(checkpoint, calibration_report)
    if not serialized["qc_pass"]:
        raise ValueError("; ".join(serialized["qc_reasons"]))
    contract = json.loads(
        (checkpoint / "calibration-contract.json").read_text(encoding="utf-8")
    )
    expected_image = contract.get("runtime", {}).get("base_inference_image_digest")
    if not inference_image_digest or inference_image_digest != expected_image:
        raise ValueError("frozen inference image digest does not match the contract")
    if os.environ.get("VLLM_ATTENTION_BACKEND") != "FLASHINFER":
        raise ValueError("VLLM_ATTENTION_BACKEND must be FLASHINFER")

    torch = importlib.import_module("torch")
    vllm = importlib.import_module("vllm")
    EngineArgs = getattr(vllm, "EngineArgs")
    LLMEngine = getattr(vllm, "LLMEngine")
    if "vllm.engine.llm_engine" not in LLMEngine.__module__:
        raise RuntimeError("checkpoint load audit requires the V0 LLMEngine")
    engine = LLMEngine.from_engine_args(
        EngineArgs(
            model=str(checkpoint),
            tokenizer=str(checkpoint),
            dtype="bfloat16",
            kv_cache_dtype="fp8_e4m3",
            calculate_kv_scales=False,
            seed=seed,
            max_model_len=512,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_batched_tokens=512,
            max_num_seqs=1,
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
        cache["dtypes"], declared_dtype="fp8_e4m3"
    )
    weights = _weight_storage_report(engine)
    scales = _kv_scale_report(engine, torch)
    scale_suffixes = (".k_scale", ".v_scale", "._k_scale", "._v_scale")
    non_scale_parameters = [
        row
        for row in weights["inventory"]
        if row["kind"] == "parameter" and not row["name"].endswith(scale_suffixes)
    ]
    non_scale_parameter_dtypes = sorted({row["dtype"] for row in non_scale_parameters})
    decode = _run_decode_smoke(engine, vllm, seed=seed)

    reasons = list(cache_contract["qc_reasons"])
    if cache["gpu_tensor_count"] != cache["tensor_count"]:
        reasons.append("one or more KV cache tensors are not on the GPU")
    if non_scale_parameter_dtypes != ["torch.bfloat16"]:
        reasons.append("one or more non-scale model parameters are not BF16")
    expected_layers = contract["expected_model_invariants"]["attention_layers"]
    if scales["layer_count"] != expected_layers:
        reasons.append("vLLM did not load one K/V scale pair per attention layer")
    if not scales["finite_positive"]:
        reasons.append("one or more vLLM K/V scales are nonpositive or nonfinite")
    if scales["all_unity"]:
        reasons.append("vLLM loaded only unity K/V scales")
    if not decode["validation"]["qc_pass"]:
        reasons.extend(decode["validation"]["qc_reasons"])

    return {
        "schema_version": 1,
        "measurement": "frozen-vllm-fp8-kv-checkpoint-load-audit",
        "non_paper_measurement": True,
        "may_enter_paper_outcomes": False,
        "energy_measurement": False,
        "checkpoint": str(checkpoint),
        "calibration_report": {
            "path": str(calibration_report_path),
            "sha256": observed_report_hash,
        },
        "runtime": {
            "inference_image_digest": inference_image_digest,
            "vllm_version": vllm.__version__,
            "engine_module": LLMEngine.__module__,
            "attention_backend": os.environ.get("VLLM_ATTENTION_BACKEND"),
            "weight_dtype": "bfloat16",
            "kv_cache_dtype": "fp8_e4m3",
            "calculate_kv_scales": False,
        },
        "serialized_checkpoint": serialized,
        "cache": cache,
        "cache_contract": cache_contract,
        "weights": weights,
        "non_scale_parameter_dtypes": non_scale_parameter_dtypes,
        "kv_scales": scales,
        "decode_smoke": decode,
        "qc_reasons": reasons,
        "qc_pass": not reasons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--calibration-report", required=True, type=Path)
    parser.add_argument("--expected-calibration-report-sha256", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = run_checkpoint_load_audit(
            checkpoint=args.checkpoint,
            calibration_report_path=args.calibration_report,
            expected_calibration_report_sha256=(
                args.expected_calibration_report_sha256
            ),
            gpu_memory_utilization=args.gpu_memory_utilization,
            seed=args.seed,
            inference_image_digest=os.environ.get("TEL_INFERENCE_IMAGE_DIGEST"),
        )
    except Exception as error:
        traceback.print_exc()
        report = {
            "schema_version": 1,
            "measurement": "frozen-vllm-fp8-kv-checkpoint-load-audit",
            "non_paper_measurement": True,
            "may_enter_paper_outcomes": False,
            "energy_measurement": False,
            "qc_pass": False,
            "qc_reasons": [f"{type(error).__name__}: {error}"],
        }
    _atomic_write_json(args.output_json, report)
    summary = {
        key: value
        for key, value in report.items()
        if key not in {"cache", "weights", "kv_scales", "decode_smoke"}
    }
    if "cache" in report:
        summary["cache"] = {
            key: value for key, value in report["cache"].items() if key != "tensors"
        }
    if "weights" in report:
        summary["weights"] = {
            key: value
            for key, value in report["weights"].items()
            if key != "inventory"
        }
    if "kv_scales" in report:
        summary["kv_scales"] = {
            key: value
            for key, value in report["kv_scales"].items()
            if key != "layers"
        }
    if "decode_smoke" in report:
        summary["decode_smoke"] = {
            key: value
            for key, value in report["decode_smoke"].items()
            if key != "observations"
        }
    summary["output_json"] = str(args.output_json)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report["qc_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
