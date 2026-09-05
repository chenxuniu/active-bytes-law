#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 START_ORDER END_ORDER" >&2
  exit 64
fi

repo_root=${TEL_REPO_ROOT:-/srv/token-energy-law/repo}
export TEL_TP2_BATCH_CAMPAIGN_LOCK="$repo_root/results/manifests/gh200-tp2-nvlink-identification.lock.json"
export TEL_TP2_BATCH_ATTEMPT_RUNNER="$repo_root/scripts/run_gh200_tp2_identification_attempt.sh"
export TEL_TP2_BATCH_RESULT_DOMAIN=tp2-identification
export TEL_TP2_BATCH_LOG_DOMAIN=tp2-identification-batch-runs
export TEL_TP2_BATCH_MEASUREMENT=gh200-tp2-identification-batch-execution-summary
export TEL_TP2_BATCH_MAXIMUM_ORDER=44

exec "$repo_root/scripts/run_gh200_tp2_batch_core.sh" "$1" "$2"
