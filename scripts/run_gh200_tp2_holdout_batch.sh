#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 START_ORDER END_ORDER" >&2
  exit 64
fi

repo_root=${TEL_REPO_ROOT:-/srv/token-energy-law/repo}
export TEL_TP2_BATCH_CAMPAIGN_LOCK="$repo_root/results/manifests/gh200-tp2-nvlink-holdout.lock.json"
export TEL_TP2_BATCH_ATTEMPT_RUNNER="$repo_root/scripts/run_gh200_tp2_holdout_attempt.sh"
export TEL_TP2_BATCH_RESULT_DOMAIN=tp2-holdout
export TEL_TP2_BATCH_LOG_DOMAIN=tp2-holdout-batch-runs
export TEL_TP2_BATCH_MEASUREMENT=gh200-tp2-holdout-batch-execution-summary
export TEL_TP2_BATCH_MAXIMUM_ORDER=29

exec "$repo_root/scripts/run_gh200_tp2_batch_core.sh" "$1" "$2"
