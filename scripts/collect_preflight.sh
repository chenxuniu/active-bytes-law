#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PRIVATE_DIR="${AB_PRIVATE_DIR:-${REPO_ROOT}/artifacts/private}"
PUBLIC_DIR="${AB_PUBLIC_DIR:-${REPO_ROOT}/environment}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PRIVATE_OUT="${PRIVATE_DIR}/preflight-${STAMP}.txt"
PUBLIC_OUT="${PUBLIC_DIR}/system-profile.measured-${STAMP}.json"

mkdir -p "${PRIVATE_DIR}" "${PUBLIC_DIR}"
umask 077

capture() {
  local label="$1"
  shift
  printf '\n[%s]\n' "${label}"
  if command -v "$1" >/dev/null 2>&1; then
    "$@" || printf 'command failed with status %s\n' "$?"
  else
    printf 'command unavailable: %s\n' "$1"
  fi
}

{
  printf '[warning]\nPrivate raw node inventory. Do not commit or publish.\n'
  capture utc-date date -u --iso-8601=seconds
  capture kernel uname -a
  capture hostname hostname --fqdn
  capture repository-commit git -C "${REPO_ROOT}" rev-parse HEAD
  capture repository-status git -C "${REPO_ROOT}" status --short
  capture gpu-list nvidia-smi -L
  capture gpu-query nvidia-smi -q
  capture gpu-topology nvidia-smi topo -m
  capture dcgm-version dcgmi --version
  capture dcgm-discovery dcgmi discovery -l
  capture dcgm-energy-field dcgmi dmon -i 0 -e 156 -c 1
  capture ncu-version ncu --version
  capture ncu-metrics ncu --query-metrics
  capture docker-version docker --version
  if [[ -n "${AB_CONTAINER_IMAGE:-}" ]]; then
    capture container-image docker image inspect "${AB_CONTAINER_IMAGE}" \
      --format '{{.Id}} {{json .RepoDigests}}'
  fi
} >"${PRIVATE_OUT}" 2>&1

chmod 600 "${PRIVATE_OUT}"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "${PRIVATE_OUT}" >"${PRIVATE_OUT}.sha256"
else
  shasum -a 256 "${PRIVATE_OUT}" >"${PRIVATE_OUT}.sha256"
fi

PUBLIC_ARGS=(--output "${PUBLIC_OUT}")
if [[ -n "${AB_CONTAINER_IMAGE_DIGEST:-}" ]]; then
  PUBLIC_ARGS+=(--container-image-digest "${AB_CONTAINER_IMAGE_DIGEST}")
fi
if [[ -n "${AB_MODEL_REVISION:-}" ]]; then
  PUBLIC_ARGS+=(--model-revision "${AB_MODEL_REVISION}")
fi
if [[ -n "${AB_TOKENIZER_REVISION:-}" ]]; then
  PUBLIC_ARGS+=(--tokenizer-revision "${AB_TOKENIZER_REVISION}")
fi

python3 "${SCRIPT_DIR}/collect_public_system.py" "${PUBLIC_ARGS[@]}"
python3 "${SCRIPT_DIR}/check_publication_safety.py" --paths "${PUBLIC_OUT}"

printf 'private raw inventory: %s\n' "${PRIVATE_OUT}"
printf 'reviewable public profile: %s\n' "${PUBLIC_OUT}"
