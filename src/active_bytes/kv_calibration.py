"""Reproducible, non-paper FP8 KV-cache calibration doctor."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import re
import traceback
from pathlib import Path
from typing import Any, Iterable

from .decode_doctor import _atomic_write_json


PINNED_DATASET_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"
REVISION = re.compile(r"^[0-9a-f]{40}$")
KV_STATE_SUFFIXES = {"k_scale", "v_scale", "k_zero_point", "v_zero_point"}
KV_OBSERVER = "minmax"


def validate_revision(value: str, *, label: str) -> str:
    if not REVISION.fullmatch(value):
        raise ValueError(f"{label} must be a full lowercase 40-hex revision")
    return value


def calibration_sample_digest(texts: Iterable[str]) -> str:
    canonical = json.dumps(
        list(texts), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _scalar(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    numel = getattr(value, "numel", None)
    if not callable(numel) or int(numel()) != 1:
        return None
    current = value
    for method in ("detach", "float", "cpu"):
        operation = getattr(current, method, None)
        if callable(operation):
            current = operation()
    item = getattr(current, "item", None)
    return float(item()) if callable(item) else None


def kv_scale_report(model: Any) -> dict[str, Any]:
    """Find calibrated K/V scales across both state entries and module attributes."""

    scales: dict[tuple[str, str], float] = {}
    state_iterators = (
        getattr(model, "named_parameters", None),
        getattr(model, "named_buffers", None),
    )
    for iterator in state_iterators:
        if not callable(iterator):
            continue
        for name, value in iterator():
            suffix = name.rsplit(".", 1)[-1]
            if suffix not in {"k_scale", "v_scale"}:
                continue
            scalar = _scalar(value)
            if scalar is not None:
                parent = name.rsplit(".", 1)[0]
                scales[(parent, suffix)] = scalar

    for module_name, module in model.named_modules():
        for suffix in ("k_scale", "v_scale"):
            for attribute in (suffix, f"_{suffix}"):
                scalar = _scalar(getattr(module, attribute, None))
                if scalar is not None:
                    scales.setdefault((module_name, suffix), scalar)
                    break

    parents = sorted({parent for parent, _ in scales})
    layers = [
        {
            "name": parent,
            "k_scale": scales.get((parent, "k_scale")),
            "v_scale": scales.get((parent, "v_scale")),
        }
        for parent in parents
    ]
    complete = [
        row for row in layers if row["k_scale"] is not None and row["v_scale"] is not None
    ]
    numeric = [
        value
        for row in complete
        for value in (row["k_scale"], row["v_scale"])
    ]
    return {
        "discovered_parent_count": len(layers),
        "complete_layer_count": len(complete),
        "finite_positive": bool(numeric)
        and all(math.isfinite(value) and value > 0 for value in numeric),
        "all_unity": bool(numeric)
        and all(math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-7) for value in numeric),
        "unique_k_scales": sorted({row["k_scale"] for row in complete}),
        "unique_v_scales": sorted({row["v_scale"] for row in complete}),
        "layers": layers,
    }


def _tensor_probe(tensor: Any) -> list[float]:
    flat = tensor.detach().reshape(-1)
    if int(flat.numel()) == 0:
        return []
    indexes = sorted({0, int(flat.numel()) // 2, int(flat.numel()) - 1})
    return [float(flat[index].float().cpu().item()) for index in indexes]


def parameter_probe(model: Any, *, names: set[str] | None = None) -> dict[str, Any]:
    """Cheaply detect accidental weight replacement without hashing every model byte."""

    rows = []
    for name, tensor in model.named_parameters():
        if names is not None and name not in names:
            continue
        rows.append(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "numel": int(tensor.numel()),
                "probe": _tensor_probe(tensor),
            }
        )
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {
        "parameter_count": len(rows),
        "logical_nbytes": sum(
            int(tensor.numel()) * int(tensor.element_size())
            for name, tensor in model.named_parameters()
            if names is None or name in names
        ),
        "probe_sha256": hashlib.sha256(canonical).hexdigest(),
        "names": [row["name"] for row in rows],
    }


def added_state_entries(model: Any, baseline: set[str]) -> list[str]:
    current = {
        name for name, _ in model.named_parameters()
    } | {name for name, _ in model.named_buffers()}
    return sorted(current - baseline)


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in (
        "llmcompressor",
        "compressed-tensors",
        "transformers",
        "datasets",
        "accelerate",
        "torch",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def run_kv_calibration_doctor(
    *,
    model_id: str,
    model_revision: str,
    dataset_id: str,
    dataset_revision: str,
    dataset_split: str,
    num_calibration_samples: int,
    max_sequence_length: int,
    seed: int,
) -> dict[str, Any]:
    validate_revision(model_revision, label="model revision")
    validate_revision(dataset_revision, label="dataset revision")
    if num_calibration_samples <= 0 or max_sequence_length <= 0:
        raise ValueError("sample count and maximum sequence length must be positive")

    torch = importlib.import_module("torch")
    datasets = importlib.import_module("datasets")
    transformers = importlib.import_module("transformers")
    llmcompressor = importlib.import_module("llmcompressor")
    quantization = importlib.import_module("compressed_tensors.quantization")
    modifiers = importlib.import_module("llmcompressor.modifiers.quantization")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        revision=model_revision,
        trust_remote_code=False,
    )
    raw = datasets.load_dataset(
        dataset_id,
        split=dataset_split,
        revision=dataset_revision,
    )
    if len(raw) < num_calibration_samples:
        raise ValueError(
            f"dataset contains {len(raw)} rows, fewer than requested "
            f"{num_calibration_samples}"
        )
    selected = raw.shuffle(seed=seed).select(range(num_calibration_samples))
    rendered_texts = [
        tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
        for row in selected
    ]
    tokenized_rows = [
        tokenizer(
            text,
            padding=False,
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=False,
        )
        for text in rendered_texts
    ]
    prepared = datasets.Dataset.from_list(tokenized_rows)
    token_lengths = [len(row["input_ids"]) for row in tokenized_rows]

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_revision,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=False,
    )
    baseline_parameter_names = {name for name, _ in model.named_parameters()}
    baseline_state_names = baseline_parameter_names | {
        name for name, _ in model.named_buffers()
    }
    before = parameter_probe(model)

    fp8_args = quantization.QuantizationArgs(
        num_bits=8,
        type="float",
        strategy="tensor",
        symmetric=True,
        dynamic=False,
        observer=KV_OBSERVER,
    )
    recipe = modifiers.QuantizationModifier(kv_cache_scheme=fp8_args)
    model = llmcompressor.oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=prepared,
        recipe=recipe,
        num_calibration_samples=num_calibration_samples,
        shuffle_calibration_samples=False,
        max_seq_length=max_sequence_length,
        pad_to_max_length=True,
        save_compressed=False,
        oneshot_device="cuda:0",
        model_revision=model_revision,
        output_dir=None,
        log_dir=None,
    )

    after = parameter_probe(model, names=baseline_parameter_names)
    added = added_state_entries(model, baseline_state_names)
    disallowed_added = [
        name for name in added if name.rsplit(".", 1)[-1] not in KV_STATE_SUFFIXES
    ]
    scales = kv_scale_report(model)
    expected_layers = int(getattr(model.config, "num_hidden_layers", 0))
    reasons: list[str] = []
    if before["parameter_count"] != after["parameter_count"]:
        reasons.append("one or more baseline parameters disappeared")
    if before["probe_sha256"] != after["probe_sha256"]:
        reasons.append("the baseline-parameter probe changed during KV-only calibration")
    if disallowed_added:
        reasons.append("calibration added state entries outside the K/V scale contract")
    if scales["complete_layer_count"] != expected_layers:
        reasons.append(
            f"discovered {scales['complete_layer_count']} complete K/V scale pairs; "
            f"expected {expected_layers}"
        )
    if not scales["finite_positive"]:
        reasons.append("one or more K/V scales are nonpositive or nonfinite")
    if scales["all_unity"]:
        reasons.append("every calibrated K/V scale is still 1.0")

    return {
        "schema_version": 1,
        "measurement": "offline-fp8-kv-calibration-doctor",
        "non_paper_measurement": True,
        "may_enter_paper_outcomes": False,
        "checkpoint_saved": False,
        "model": {"id": model_id, "revision": model_revision},
        "dataset": {
            "id": dataset_id,
            "revision": dataset_revision,
            "split": dataset_split,
            "seed": seed,
            "num_calibration_samples": num_calibration_samples,
            "max_sequence_length": max_sequence_length,
            "rendered_text_sha256": calibration_sample_digest(rendered_texts),
            "token_length_min": min(token_lengths),
            "token_length_median": sorted(token_lengths)[len(token_lengths) // 2],
            "token_length_max": max(token_lengths),
        },
        "runtime": {
            "packages": _package_versions(),
            "cuda": torch.version.cuda,
            "device": str(next(model.parameters()).device),
            "weight_dtype": str(next(model.parameters()).dtype),
        },
        "recipe": {
            "weights_quantized": False,
            "kv_cache": {
                "num_bits": 8,
                "type": "float",
                "strategy": "tensor",
                "symmetric": True,
                "dynamic": False,
                "observer": KV_OBSERVER,
            },
        },
        "baseline_parameters_before": {
            key: value for key, value in before.items() if key != "names"
        },
        "baseline_parameters_after": {
            key: value for key, value in after.items() if key != "names"
        },
        "added_state_entries": added,
        "disallowed_added_state_entries": disallowed_added,
        "kv_scales": scales,
        "qc_reasons": reasons,
        "qc_pass": not reasons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--dataset-revision", default=PINNED_DATASET_REVISION)
    parser.add_argument("--dataset-split", default="train_sft")
    parser.add_argument("--num-calibration-samples", type=int, default=8)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = run_kv_calibration_doctor(
            model_id=args.model,
            model_revision=args.model_revision,
            dataset_id=args.dataset,
            dataset_revision=args.dataset_revision,
            dataset_split=args.dataset_split,
            num_calibration_samples=args.num_calibration_samples,
            max_sequence_length=args.max_sequence_length,
            seed=args.seed,
        )
    except Exception as error:  # Preserve a machine-readable failure artifact.
        traceback.print_exc()
        report = {
            "schema_version": 1,
            "measurement": "offline-fp8-kv-calibration-doctor",
            "non_paper_measurement": True,
            "may_enter_paper_outcomes": False,
            "checkpoint_saved": False,
            "qc_pass": False,
            "qc_reasons": [f"{type(error).__name__}: {error}"],
        }
    _atomic_write_json(args.output_json, report)
    summary = {
        key: value
        for key, value in report.items()
        if key not in {"kv_scales", "added_state_entries"}
    }
    if "kv_scales" in report:
        summary["kv_scales"] = {
            key: value
            for key, value in report["kv_scales"].items()
            if key != "layers"
        }
    summary["output_json"] = str(args.output_json)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report["qc_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
