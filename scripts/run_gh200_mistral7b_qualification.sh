#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [GPU_INDEX]" >&2
  exit 64
fi

gpu_index=${1:-0}
if [[ "$gpu_index" != "0" && "$gpu_index" != "1" ]]; then
  echo "GPU_INDEX must be 0 or 1" >&2
  exit 64
fi

repo_root=${TEL_REPO_ROOT:-/srv/token-energy-law/repo}
export TEL_QUALIFICATION_CAMPAIGN_LOCK="$repo_root/results/manifests/gh200-mistral7b-qualification-v1.lock.json"
export TEL_QUALIFICATION_CONTRACT="$repo_root/configs/addenda/gh200-mistral7b-qualification-v1.json"
export TEL_QUALIFICATION_RESULT_SUBDIR=mistral7b-v0p3
export TEL_QUALIFICATION_ID=gh200-mistral7b-architecture-replication-qualification-v1
export TEL_QUALIFICATION_NAME_PREFIX=tel-m7

exec "$repo_root/scripts/run_gh200_qwen14b_qualification.sh" "$gpu_index"
