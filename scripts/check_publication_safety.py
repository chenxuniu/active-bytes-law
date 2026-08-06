#!/usr/bin/env python3
"""Reject common secrets and infrastructure identifiers before publication."""

from __future__ import annotations

import argparse
import ipaddress
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable


DENIED_SUFFIXES = {
    ".ncu-rep",
    ".nsys-rep",
    ".qdrep",
    ".safetensors",
    ".pt",
    ".pth",
    ".pem",
    ".key",
    ".zip",
}
SKIP_PARTS = {".git", "__pycache__", ".venv", "venv"}
MAX_TEXT_BYTES = 5 * 1024 * 1024

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email address", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)),
    ("MAC address", re.compile(r"\b(?:[0-9A-F]{2}:){5}[0-9A-F]{2}\b", re.I)),
    ("GPU UUID", re.compile(r"\bGPU-[0-9A-F-]{20,}\b", re.I)),
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("GitHub token", re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("bearer credential", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}", re.I)),
    ("absolute user path", re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+(?:/|\b)")),
    (
        "infrastructure identifier assignment",
        re.compile(
            r"\b(?:hostname|fqdn|bmc(?:_ip)?|serial(?:_number)?|lease_id|reservation_id|resource_id)\s*[:=]\s*[A-Za-z0-9._:-]+",
            re.I,
        ),
    ),
]

IPV4 = re.compile(r"(?<![0-9.])(?:\d{1,3}\.){3}\d{1,3}(?![0-9.])")
FQDN = re.compile(r"\b(?:[a-z0-9-]+\.)+nvidia\.(?:com|net)\b", re.I)
PUBLIC_NVIDIA_HOSTS = {"www.nvidia.com", "developer.nvidia.com", "docs.nvidia.com"}


def tracked_or_publishable_files(root: Path) -> list[Path]:
    try:
        output = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return [
            path
            for path in root.rglob("*")
            if path.is_file() and not any(part in SKIP_PARTS for part in path.parts)
        ]
    return [root / item.decode("utf-8") for item in output.split(b"\0") if item]


def scan_text(text: str) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    for label, pattern in PATTERNS:
        for match in pattern.finditer(text):
            findings.append((label, match.group(0)[:120]))
    for match in IPV4.finditer(text):
        candidate = match.group(0)
        try:
            address = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if address.is_private or address.is_loopback or address.is_link_local:
            findings.append(("non-public IPv4 address", candidate))
    for match in FQDN.finditer(text):
        candidate = match.group(0).lower()
        if candidate not in PUBLIC_NVIDIA_HOSTS:
            findings.append(("potential internal NVIDIA hostname", candidate))
    return findings


def scan_files(paths: Iterable[Path]) -> list[str]:
    messages: list[str] = []
    for path in paths:
        if not path.exists() or not path.is_file():
            messages.append(f"missing file: {path}")
            continue
        if path.name == Path(__file__).name:
            continue
        if path.suffix.lower() in DENIED_SUFFIXES or any(
            path.name.lower().endswith(suffix) for suffix in DENIED_SUFFIXES
        ):
            messages.append(f"denied binary/archive type: {path}")
            continue
        if path.stat().st_size > MAX_TEXT_BYTES:
            messages.append(f"unreviewed file larger than {MAX_TEXT_BYTES} bytes: {path}")
            continue
        raw = path.read_bytes()
        if b"\x00" in raw:
            messages.append(f"unreviewed binary file: {path}")
            continue
        text = raw.decode("utf-8", errors="replace")
        for label, excerpt in scan_text(text):
            messages.append(f"{path}: {label}: {excerpt!r}")
    return messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", nargs="*", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    paths = args.paths if args.paths else tracked_or_publishable_files(root)
    findings = scan_files(paths)
    if findings:
        print("publication safety check failed:", file=sys.stderr)
        for finding in findings:
            print(f"- {finding}", file=sys.stderr)
        return 2
    print(f"publication safety check passed ({len(paths)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
