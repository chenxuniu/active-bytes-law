#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 START_ORDER END_ORDER [GPU_INDEX]" >&2
  exit 64
fi

start_order=$1
end_order=$2
gpu_index=${3:-1}

if ! [[ "$start_order" =~ ^[0-9]+$ && "$end_order" =~ ^[0-9]+$ ]]; then
  echo "START_ORDER and END_ORDER must be integers in [0, 44]" >&2
  exit 64
fi
if (( start_order < 0 || end_order > 44 || start_order > end_order )); then
  echo "invalid order range: expected 0 <= START_ORDER <= END_ORDER <= 44" >&2
  exit 64
fi
if [[ "$gpu_index" != "1" ]]; then
  echo "the frozen same-SKU transfer campaign is bound to GPU index 1" >&2
  exit 64
fi

repo_root=${TEL_REPO_ROOT:-/srv/token-energy-law/repo}
export TEL_EXPECTED_GPU_INDEX=1
export TEL_V2_CAMPAIGN_LOCK="$repo_root/results/manifests/gh200-gpu1-same-sku-transfer.lock.json"
export TEL_V2_ATTEMPT_RUNNER="$repo_root/scripts/run_gh200_gpu1_transfer_attempt.sh"
export TEL_V2_RESULT_DOMAIN=gpu1-same-sku-transfer
export TEL_V2_BATCH_RESULT_DOMAIN=gpu1-same-sku-transfer-batch-runs
export TEL_V2_BATCH_MEASUREMENT=gh200-gpu1-same-sku-transfer-batch-execution-summary

exec "$repo_root/scripts/run_gh200_v2_duration_holdout_batch.sh" \
  "$start_order" "$end_order" "$gpu_index"
