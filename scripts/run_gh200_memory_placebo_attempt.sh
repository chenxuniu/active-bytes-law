#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 REPEAT_NUMBER [GPU_INDEX]" >&2
  exit 64
fi

repeat=$1
gpu_index=${2:-0}
case "$repeat" in
  1|01) repeat=01 ;;
  2|02) repeat=02 ;;
  3|03) repeat=03 ;;
  *)
    echo "repeat must be 1, 2, or 3" >&2
    exit 64
    ;;
esac

repo_root=${TEL_REPO_ROOT:-/srv/token-energy-law/repo}
export TEL_CAMPAIGN_LOCK="$repo_root/results/manifests/gh200-memory-placebo.lock.json"

exec "$repo_root/scripts/run_gh200_pilot_attempt.sh" \
  "ab1-placebo-allocator-u080-l4096-b8-bf16-r$repeat" \
  0.80 \
  "$gpu_index"
