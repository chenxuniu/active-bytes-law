#!/usr/bin/env python3
"""Report the pinned calibration package and callable API surface."""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import json
import platform


PACKAGES = (
    "accelerate",
    "apex",
    "compressed-tensors",
    "datasets",
    "llmcompressor",
    "numpy",
    "torch",
    "transformers",
    "vllm",
)


def distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def signature(module_name: str, attribute: str) -> dict[str, str | None]:
    try:
        module = importlib.import_module(module_name)
        value = getattr(module, attribute)
    except (ImportError, AttributeError) as exc:
        return {"import": f"{module_name}.{attribute}", "error": repr(exc)}
    try:
        rendered = str(inspect.signature(value))
    except (TypeError, ValueError):
        rendered = None
    return {
        "import": f"{module_name}.{attribute}",
        "module": getattr(value, "__module__", None),
        "signature": rendered,
    }


def main() -> int:
    torch = importlib.import_module("torch")
    report = {
        "schema_version": 1,
        "measurement": "calibration-stack-api-audit",
        "non_paper_measurement": True,
        "platform": {
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count(),
        },
        "packages": {name: distribution_version(name) for name in PACKAGES},
        "api": [
            signature("llmcompressor", "oneshot"),
            signature(
                "llmcompressor.modifiers.quantization", "QuantizationModifier"
            ),
            signature("datasets", "load_dataset"),
            signature("transformers", "AutoModelForCausalLM"),
        ],
    }
    required = (
        report["packages"]["llmcompressor"] == "0.6.0.1"
        and report["packages"]["compressed-tensors"] == "0.10.2"
        and report["packages"]["transformers"] == "4.52.4"
        and report["packages"]["vllm"] is None
        and report["packages"]["apex"] is None
    )
    report["qc_pass"] = required and all("error" not in row for row in report["api"])
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qc_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
