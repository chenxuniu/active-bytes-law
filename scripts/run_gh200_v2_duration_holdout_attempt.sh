#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 RUN_ORDER [GPU_INDEX]" >&2
  exit 64
fi

run_order=$1
gpu_index=${2:-0}
if [[ "$gpu_index" != "0" ]]; then
  echo "the frozen GH200 V2 holdout is bound to GPU index 0" >&2
  exit 64
fi
if [[ ! "$run_order" =~ ^[0-9]+$ ]] || (( run_order < 0 || run_order >= 45 )); then
  echo "RUN_ORDER must be an integer from 0 through 44" >&2
  exit 64
fi

repo_root=${TEL_REPO_ROOT:-/srv/token-energy-law/repo}
results_root=${TEL_RESULTS_ROOT:-/srv/token-energy-law/results}
campaign_lock="$repo_root/results/manifests/gh200-v2-duration-holdout.lock.json"
execution_addendum="$repo_root/configs/addenda/gh200-primary-bf16-v1.json"
model_artifact="$repo_root/configs/addenda/gh200-v2-duration-holdout-v1.json"
result_domain=duration-v2-holdout

readarray -t resolved < <(python3 - "$campaign_lock" "$run_order" <<'PY'
import json
import sys
lock = json.load(open(sys.argv[1], encoding="utf-8"))
order = int(sys.argv[2])
matches = [row for row in lock["run_order"] if row["order"] == order]
if len(matches) != 1:
    raise SystemExit("run order does not occur exactly once in the frozen lock")
run = matches[0]
print(run["run_id"])
print(run["parameters"]["gpu_memory_utilization"])
print(run["parameters"]["v2_model_artifact_sha256"])
PY
)

run_id=${resolved[0]}
gpu_memory_utilization=${resolved[1]}
expected_model_sha=${resolved[2]}
observed_model_sha=$(sha256sum "$model_artifact" | awk '{print $1}')
if [[ "$observed_model_sha" != "$expected_model_sha" ]]; then
  echo "V2 model artifact does not match the frozen campaign lock" >&2
  exit 65
fi

export TEL_CAMPAIGN_LOCK="$campaign_lock"
export TEL_EXECUTION_ADDENDUM="$execution_addendum"
export TEL_RESULT_DOMAIN="$result_domain"
export TEL_REQUIRE_CLEAN_REPO=1
export TEL_EXTRA_CONTRACTS="$model_artifact"

echo "frozen_v2_order=$run_order"
echo "frozen_v2_run_id=$run_id"
echo "v2_model_artifact_sha256=$observed_model_sha"

exec "$repo_root/scripts/run_gh200_pilot_attempt.sh" \
  "$run_id" \
  "$gpu_memory_utilization" \
  "$gpu_index"
