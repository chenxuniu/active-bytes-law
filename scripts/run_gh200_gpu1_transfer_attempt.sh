#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 RUN_ORDER [GPU_INDEX]" >&2
  exit 64
fi

run_order=$1
gpu_index=${2:-1}
if [[ "$gpu_index" != "1" ]]; then
  echo "the frozen same-SKU transfer campaign is bound to GPU index 1" >&2
  exit 64
fi
if [[ ! "$run_order" =~ ^[0-9]+$ ]] || (( run_order < 0 || run_order >= 45 )); then
  echo "RUN_ORDER must be an integer from 0 through 44" >&2
  exit 64
fi

# The source and target devices share one node. Reject a co-run before opening
# the target outcome so activity on the sibling cannot become an unrecorded
# device-transfer confounder. A small allowance covers driver bookkeeping;
# the qualified idle state observed 0 MiB on both devices.
idle_inventory=$(nvidia-smi \
  --query-gpu=index,memory.used \
  --format=csv,noheader,nounits) || exit $?
python3 - "$idle_inventory" <<'PY' || exit 65
import sys

observed = {}
for line in sys.argv[1].splitlines():
    fields = [field.strip() for field in line.split(",")]
    if len(fields) == 2:
        observed[int(fields[0])] = float(fields[1])
if set(observed) != {0, 1}:
    raise SystemExit("expected exactly host GPU indexes 0 and 1")
busy = {index: used for index, used in observed.items() if used > 16.0}
if busy:
    raise SystemExit(f"both physical GPUs must be idle before transfer attempts: {busy}")
PY

repo_root=${TEL_REPO_ROOT:-/srv/token-energy-law/repo}
campaign_lock="$repo_root/results/manifests/gh200-gpu1-same-sku-transfer.lock.json"
execution_addendum="$repo_root/configs/addenda/gh200-primary-bf16-v1.json"
model_artifact="$repo_root/configs/addenda/gh200-v2-duration-holdout-v1.json"
transfer_addendum="$repo_root/configs/addenda/gh200-gpu1-same-sku-transfer-v1.json"

readarray -t resolved < <(python3 - "$campaign_lock" "$run_order" <<'PY'
import json
import sys

lock = json.load(open(sys.argv[1], encoding="utf-8"))
order = int(sys.argv[2])
matches = [row for row in lock["run_order"] if row["order"] == order]
if len(matches) != 1:
    raise SystemExit("run order does not occur exactly once in the frozen lock")
run = matches[0]
p = run["parameters"]
print(run["run_id"])
print(p["gpu_memory_utilization"])
print(p["target_gpu_index"])
print(p["v2_model_artifact_sha256"])
print(p["device_transfer_addendum_sha256"])
PY
)

run_id=${resolved[0]}
gpu_memory_utilization=${resolved[1]}
locked_gpu_index=${resolved[2]}
expected_model_sha=${resolved[3]}
expected_transfer_sha=${resolved[4]}

if [[ "$locked_gpu_index" != "$gpu_index" ]]; then
  echo "host GPU index does not match the frozen campaign" >&2
  exit 65
fi
observed_model_sha=$(sha256sum "$model_artifact" | awk '{print $1}')
observed_transfer_sha=$(sha256sum "$transfer_addendum" | awk '{print $1}')
if [[ "$observed_model_sha" != "$expected_model_sha" ]]; then
  echo "frozen duration model does not match the campaign lock" >&2
  exit 65
fi
if [[ "$observed_transfer_sha" != "$expected_transfer_sha" ]]; then
  echo "same-SKU transfer addendum does not match the campaign lock" >&2
  exit 65
fi

export TEL_CAMPAIGN_LOCK="$campaign_lock"
export TEL_EXECUTION_ADDENDUM="$execution_addendum"
export TEL_RESULT_DOMAIN=gpu1-same-sku-transfer
export TEL_REQUIRE_CLEAN_REPO=1
export TEL_EXTRA_CONTRACTS="$model_artifact:$transfer_addendum"

echo "frozen_transfer_order=$run_order"
echo "frozen_transfer_run_id=$run_id"
echo "target_gpu_index=$gpu_index"
echo "coefficient_refit_allowed=false"

exec "$repo_root/scripts/run_gh200_pilot_attempt.sh" \
  "$run_id" \
  "$gpu_memory_utilization" \
  "$gpu_index"
