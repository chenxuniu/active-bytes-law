"""Validate public manifest linkage, campaign membership, sizes, and hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_path(root: Path, uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme in {"http", "https", "doi"}:
        return None
    if parsed.scheme:
        raise ValueError(f"unsupported artifact URI scheme: {parsed.scheme}")
    path = Path(uri)
    if path.is_absolute():
        raise ValueError("public artifact URI must not be an absolute local path")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("public artifact URI escapes the repository root") from exc
    return resolved


def validate_manifest(manifest: Mapping[str, Any], repository_root: Path) -> dict[str, Any]:
    errors: list[str] = []
    required = (
        "run_id",
        "cell_id",
        "split",
        "repeat",
        "campaign_lock_sha256",
        "campaign_lock_uri",
        "artifacts",
    )
    for field in required:
        if field not in manifest:
            errors.append(f"manifest is missing {field}")
    if errors:
        return {"qc_pass": False, "errors": errors}

    try:
        lock_path = _local_path(repository_root, str(manifest["campaign_lock_uri"]))
    except ValueError as exc:
        errors.append(f"campaign lock: {exc}")
        lock_path = None
    if lock_path is None:
        errors.append("campaign lock must be a repository-local frozen artifact")
    elif not lock_path.exists():
        errors.append(f"campaign lock does not exist: {manifest['campaign_lock_uri']}")
    else:
        try:
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"campaign lock cannot be read: {exc}")
        else:
            if lock.get("lock_sha256") != manifest["campaign_lock_sha256"]:
                errors.append("campaign lock hash does not match manifest")
            matching = [
                run for run in lock.get("run_order", []) if run.get("run_id") == manifest["run_id"]
            ]
            if len(matching) != 1:
                errors.append("run_id does not occur exactly once in campaign lock")
            else:
                run = matching[0]
                for field in ("cell_id", "split", "repeat"):
                    if run.get(field) != manifest[field]:
                        errors.append(f"manifest {field} does not match campaign lock")

    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, list):
        errors.append("artifacts must be an array")
        artifacts = []
    roles: set[str] = set()
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping):
            errors.append(f"artifact {index} is not an object")
            continue
        missing = [field for field in ("role", "artifact_uri", "sha256", "bytes") if field not in artifact]
        if missing:
            errors.append(f"artifact {index} is missing {', '.join(missing)}")
            continue
        role = str(artifact["role"])
        if role in roles:
            errors.append(f"duplicate artifact role {role!r}")
        roles.add(role)
        try:
            path = _local_path(repository_root, str(artifact["artifact_uri"]))
        except ValueError as exc:
            errors.append(f"artifact {index}: {exc}")
            continue
        if path is None:
            continue
        if not path.exists():
            errors.append(f"artifact {index} does not exist: {artifact['artifact_uri']}")
            continue
        if path.stat().st_size != artifact["bytes"]:
            errors.append(f"artifact {index} byte size does not match")
        if sha256_file(path) != artifact["sha256"]:
            errors.append(f"artifact {index} SHA-256 does not match")
    return {"qc_pass": not errors, "errors": errors, "artifact_roles": sorted(roles)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = validate_manifest(manifest, args.repository_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qc_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
