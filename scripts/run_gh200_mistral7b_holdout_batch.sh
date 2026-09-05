#!/usr/bin/env bash
set -euo pipefail

script_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
repo_root=${TEL_REPO_ROOT:-$script_root}
results_root=${TEL_RESULTS_ROOT:-/srv/token-energy-law/results}

export TEL_REPLICATION_NAME=Mistral-7B-Instruct-v0.3
export TEL_REPLICATION_RUN_COUNT=30
export TEL_REPLICATION_CAMPAIGN_LOCK="$repo_root/results/manifests/gh200-mistral7b-holdout.lock.json"
export TEL_REPLICATION_RESULT_DOMAIN=mistral7b-holdout
export TEL_REPLICATION_BATCH_DOMAIN=mistral7b-holdout-batch-runs
export TEL_REPLICATION_BATCH_MEASUREMENT=gh200-mistral7b-holdout-batch-execution-summary
export TEL_Q14_HOLDOUT_ATTEMPT_RUNNER="$repo_root/scripts/run_gh200_mistral7b_holdout_attempt.sh"

exec "$repo_root/scripts/run_gh200_qwen14b_holdout_batch.sh" "$@"
