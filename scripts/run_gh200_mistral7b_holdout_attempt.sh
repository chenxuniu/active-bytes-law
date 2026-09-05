#!/usr/bin/env bash
set -euo pipefail

script_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
repo_root=${TEL_REPO_ROOT:-$script_root}
results_root=${TEL_RESULTS_ROOT:-/srv/token-energy-law/results}

export TEL_REPLICATION_NAME=Mistral-7B-Instruct-v0.3
export TEL_REPLICATION_RUN_COUNT=30
export TEL_REPLICATION_CAMPAIGN_LOCK="$repo_root/results/manifests/gh200-mistral7b-holdout.lock.json"
export TEL_REPLICATION_ADDENDUM="$repo_root/configs/addenda/gh200-mistral7b-form-replication-v1.json"
export TEL_REPLICATION_RELEASE_RECORD="$repo_root/configs/addenda/gh200-mistral7b-holdout-release-v1.json"
export TEL_REPLICATION_IDENTIFICATION_FREEZE_DIR="$results_root/mistral7b-identification-freeze/20260905T050600Z"
export TEL_REPLICATION_RESULT_DOMAIN=mistral7b-holdout
export TEL_REPLICATION_VERIFICATION_DOMAIN=mistral7b-holdout-release-verifications
export TEL_REPLICATION_VERIFICATION_SCRIPT="$repo_root/scripts/verify_gh200_mistral7b_holdout_release.py"

exec "$repo_root/scripts/run_gh200_qwen14b_holdout_attempt.sh" "$@"
