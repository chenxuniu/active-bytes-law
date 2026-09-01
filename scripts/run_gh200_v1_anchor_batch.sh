#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 START_ORDER END_ORDER [GPU_INDEX]" >&2
  exit 64
fi

start_order=$1
end_order=$2
gpu_index=${3:-0}

for value in "$start_order" "$end_order"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]] || (( value < 0 || value >= 60 )); then
    echo "orders must be integers from 0 through 59" >&2
    exit 64
  fi
done
if (( start_order > end_order )); then
  echo "START_ORDER must not exceed END_ORDER" >&2
  exit 64
fi
if [[ "$gpu_index" != "0" ]]; then
  echo "the frozen GH200 V1 anchor campaign is bound to GPU index 0" >&2
  exit 64
fi

repo_root=${TEL_REPO_ROOT:-/srv/token-energy-law/repo}
results_root=${TEL_RESULTS_ROOT:-/srv/token-energy-law/results}
campaign_lock="$repo_root/results/manifests/gh200-v1-anchors.lock.json"
attempt_runner="$repo_root/scripts/run_gh200_v1_anchor_attempt.sh"
result_domain=v1-profiler-anchors
locked_campaign_sha=$(python3 - "$campaign_lock" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["lock_sha256"])
PY
)

if [[ -n "$(git -C "$repo_root" status --short)" ]]; then
  echo "V1 batch execution requires a clean repository checkout" >&2
  git -C "$repo_root" status --short >&2
  exit 65
fi
if [[ ! -x "$attempt_runner" ]]; then
  echo "attempt runner is missing or not executable: $attempt_runner" >&2
  exit 65
fi

batch_tag="$(date -u +%Y%m%dT%H%M%SZ)-$$"
batch_dir="$results_root/v1-batch-runs/orders-${start_order}-${end_order}/batch-$batch_tag"
mkdir -p "$batch_dir"
batch_log="$batch_dir/batch.events.log"
batch_summary="$batch_dir/batch.summary.json"
touch "$batch_log"

emit() {
  printf '%s\n' "$*" | tee -a "$batch_log"
}

run_id_for_order() {
  python3 - "$campaign_lock" "$1" <<'PY'
import json
import sys

lock = json.load(open(sys.argv[1], encoding="utf-8"))
order = int(sys.argv[2])
run = next((row for row in lock["run_order"] if row["order"] == order), None)
if run is None:
    raise SystemExit(f"order {order} is absent from the frozen campaign lock")
print(run["run_id"])
PY
}

accepted_attempt_exists() {
  python3 - "$1" "$2" "$3" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected_run_id = sys.argv[2]
expected_campaign_sha = sys.argv[3]
for path in sorted(root.glob("attempt-*/traffic.json")):
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        if (
            report.get("qc_pass") is True
            and report.get("run", {}).get("run_id") == expected_run_id
            and report.get("campaign_lock_sha256") == expected_campaign_sha
        ):
            raise SystemExit(0)
    except (OSError, ValueError):
        pass
raise SystemExit(1)
PY
}

latest_attempt_path() {
  if [[ ! -d "$1" ]]; then
    return 0
  fi
  find "$1" -maxdepth 1 -type d -name 'attempt-*' | sort | tail -1
}

record_batch_failure() {
  local marker_path=$1
  local failed_order=$2
  local failed_run_id=$3
  local runner_rc=$4
  local attempt_path=$5
  python3 - \
    "$marker_path" \
    "$failed_order" \
    "$failed_run_id" \
    "$runner_rc" \
    "$attempt_path" \
    "$locked_campaign_sha" \
    "$(git -C "$repo_root" rev-parse HEAD)" <<'PY'
import json
import os
import pathlib
import sys
from datetime import datetime, timezone

path = pathlib.Path(sys.argv[1])
report = {
    "schema_version": 1,
    "measurement": "gh200-v1-recorded-batch-failure",
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "order": int(sys.argv[2]),
    "run_id": sys.argv[3],
    "runner_return_code": int(sys.argv[4]),
    "attempt_path": sys.argv[5],
    "campaign_lock_sha256": sys.argv[6],
    "repository_commit": sys.argv[7],
    "accepted_outcome": False,
    "batch_may_continue": True,
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
}

write_batch_summary() {
  local status=$1
  local failed_csv=$2
  local accepted_count=$3
  local skipped_count=$4
  python3 - \
    "$batch_summary" \
    "$status" \
    "$start_order" \
    "$end_order" \
    "$failed_csv" \
    "$accepted_count" \
    "$skipped_count" \
    "$locked_campaign_sha" \
    "$(git -C "$repo_root" rev-parse HEAD)" <<'PY'
import json
import os
import pathlib
import sys
from datetime import datetime, timezone

path = pathlib.Path(sys.argv[1])
failed = [int(value) for value in sys.argv[5].split(",") if value]
report = {
    "schema_version": 1,
    "measurement": "gh200-v1-batch-execution-summary",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": sys.argv[2],
    "start_order": int(sys.argv[3]),
    "end_order": int(sys.argv[4]),
    "failed_orders": failed,
    "accepted_in_this_invocation": int(sys.argv[6]),
    "already_accepted_and_skipped": int(sys.argv[7]),
    "campaign_lock_sha256": sys.argv[8],
    "repository_commit": sys.argv[9],
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
}

failed_orders_csv() {
  if (( ${#failed_orders[@]} == 0 )); then
    printf '%s' ""
    return
  fi
  local joined
  joined=$(IFS=,; echo "${failed_orders[*]}")
  printf '%s' "$joined"
}

emit "batch_start_order=$start_order"
emit "batch_end_order=$end_order"
emit "gpu_index=$gpu_index"
emit "repository_commit=$(git -C "$repo_root" rev-parse HEAD)"
emit "batch_dir=$batch_dir"

failed_orders=()
accepted_count=0
skipped_count=0

for ((order = start_order; order <= end_order; order++)); do
  run_id=$(run_id_for_order "$order")
  run_root="$results_root/$result_domain/$run_id"
  if accepted_attempt_exists "$run_root" "$run_id" "$locked_campaign_sha"; then
    emit "batch_order=$order run_id=$run_id status=already-accepted"
    ((skipped_count += 1))
    continue
  fi

  attempt_before=$(latest_attempt_path "$run_root")
  emit "batch_order=$order run_id=$run_id status=starting"
  if "$attempt_runner" "$order" "$gpu_index" recorded-gaps; then
    :
  else
    runner_rc=$?
    attempt_after=$(latest_attempt_path "$run_root")
    if [[ -z "$attempt_after" || "$attempt_after" == "$attempt_before" ]]; then
      emit "batch_order=$order run_id=$run_id status=fatal-infrastructure-or-contract-failure rc=$runner_rc" >&2
      failed_csv=$(failed_orders_csv)
      write_batch_summary "aborted-infrastructure-or-contract-failure" "$failed_csv" "$accepted_count" "$skipped_count"
      exit "$runner_rc"
    fi
    failure_marker="$run_root/batch-failure-${batch_tag}-order-${order}.json"
    record_batch_failure \
      "$failure_marker" \
      "$order" \
      "$run_id" \
      "$runner_rc" \
      "$attempt_after"
    failed_orders+=("$order")
    emit "batch_order=$order run_id=$run_id status=recorded-failure-continuing rc=$runner_rc marker=$failure_marker" >&2
    continue
  fi
  if ! accepted_attempt_exists "$run_root" "$run_id" "$locked_campaign_sha"; then
    attempt_after=$(latest_attempt_path "$run_root")
    failure_marker="$run_root/batch-failure-${batch_tag}-order-${order}.json"
    record_batch_failure \
      "$failure_marker" \
      "$order" \
      "$run_id" \
      65 \
      "$attempt_after"
    failed_orders+=("$order")
    emit "batch_order=$order run_id=$run_id status=missing-accepted-traffic-continuing marker=$failure_marker" >&2
    continue
  fi
  ((accepted_count += 1))
  emit "batch_order=$order run_id=$run_id status=accepted"
done

failed_csv=$(failed_orders_csv)
if (( ${#failed_orders[@]} > 0 )); then
  write_batch_summary "complete-with-recorded-failures" "$failed_csv" "$accepted_count" "$skipped_count"
  emit "batch_status=complete-with-recorded-failures"
  emit "batch_failed_orders=$failed_csv"
  emit "batch_summary=$batch_summary"
  exit 2
fi

write_batch_summary "complete" "" "$accepted_count" "$skipped_count"
emit "batch_status=complete"
emit "batch_completed_orders=${start_order}-${end_order}"
emit "batch_summary=$batch_summary"
