#!/usr/bin/env python3
"""Collect a whitelist-only, non-identifying public system profile."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def run(arguments: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            arguments,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() or None


def os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key in {"NAME", "VERSION_ID"}:
                values[key.lower()] = value.strip().strip('"')
    return values


def cpu_profile() -> dict[str, Any]:
    model = platform.processor() or None
    physical_pairs: set[tuple[str, str]] = set()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        physical = None
        core = None
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines() + [""]:
            if not line:
                if physical is not None and core is not None:
                    physical_pairs.add((physical, core))
                physical = None
                core = None
                continue
            if ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            if key == "model name" and not model:
                model = value
            elif key == "physical id":
                physical = value
            elif key == "core id":
                core = value
    return {
        "model": model,
        "logical_cpus": os.cpu_count(),
        "physical_cores": len(physical_pairs) or None,
    }


def memory_gib() -> float | None:
    path = Path("/proc/meminfo")
    if not path.exists():
        return None
    match = re.search(r"^MemTotal:\s+(\d+)\s+kB$", path.read_text(), re.MULTILINE)
    return round(int(match.group(1)) / 1024 / 1024, 2) if match else None


def gpu_profile() -> list[dict[str, Any]]:
    fields = ["name", "memory.total", "power.limit", "mig.mode.current", "driver_version"]
    output = run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return []
    gpus: list[dict[str, Any]] = []
    for index, line in enumerate(output.splitlines()):
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(fields):
            continue
        memory = float(values[1]) if values[1].replace(".", "", 1).isdigit() else None
        power = float(values[2]) if values[2].replace(".", "", 1).isdigit() else None
        gpus.append(
            {
                "public_label": f"gpu-{index}",
                "name": values[0],
                "memory_mib": memory,
                "power_limit_w": power,
                "mig_mode": values[3],
                "driver_version": values[4],
            }
        )
    return gpus


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def version_line(arguments: list[str]) -> str | None:
    output = run(arguments)
    if not output:
        return None
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else None


def build_profile(args: argparse.Namespace) -> dict[str, Any]:
    public_git_commit = run(["git", "rev-parse", "HEAD"])
    dcgm_probe = run(["dcgmi", "dmon", "-i", "0", "-e", "156", "-c", "1"])
    profile: dict[str, Any] = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "privacy_contract": "whitelist-only; no host/user/network/device-unique identifiers",
        "operating_system": {
            **os_release(),
            "architecture": platform.machine(),
            "kernel_release": platform.release(),
        },
        "cpu": cpu_profile(),
        "host_memory_gib": memory_gib(),
        "gpus": gpu_profile(),
        "software": {
            "python": platform.python_version(),
            "cuda_toolkit": version_line(["nvcc", "--version"]),
            "dcgm": version_line(["dcgmi", "--version"]),
            "dcgm_total_energy_field_156_readable": dcgm_probe is not None,
            "nsight_compute": version_line(["ncu", "--version"]),
            "docker": version_line(["docker", "--version"]),
            "pytorch": package_version("torch"),
            "vllm": package_version("vllm"),
        },
        "public_repository_commit": public_git_commit,
        "container_image_digest": args.container_image_digest,
        "model_revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision,
    }
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--container-image-digest")
    parser.add_argument("--model-revision")
    parser.add_argument("--tokenizer-revision")
    args = parser.parse_args()
    profile = build_profile(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
