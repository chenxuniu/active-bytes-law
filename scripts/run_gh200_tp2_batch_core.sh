#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 START_ORDER END_ORDER" >&2
  exit 64
fi

start_order=$1
end_order=$2
repo_root=${TEL_REPO_ROOT:-/srv/token-energy-law/repo}
results_root=${TEL_RESULTS_ROOT:-/srv/token-energy-law/results}
campaign_lock=${TEL_TP2_BATCH_CAMPAIGN_LOCK:-}
attempt_runner=${TEL_TP2_BATCH_ATTEMPT_RUNNER:-}
result_domain=${TEL_TP2_BATCH_RESULT_DOMAIN:-}
batch_result_domain=${TEL_TP2_BATCH_LOG_DOMAIN:-}
measurement=${TEL_TP2_BATCH_MEASUREMENT:-gh200-tp2-batch-execution-summary}
maximum_order=${TEL_TP2_BATCH_MAXIMUM_ORDER:-}

for name in campaign_lock attempt_runner result_domain batch_result_domain maximum_order; do
  if [[ -z "${!name}" ]]; then
    echo "missing required TP=2 batch environment: $name" >&2
    exit 66
  fi
done
if ! [[ "$start_order" =~ ^[0-9]+$ && "$end_order" =~ ^[0-9]+$ ]]; then
  echo "START_ORDER and END_ORDER must be integers" >&2
  exit 64
fi
if ! [[ "$maximum_order" =~ ^[0-9]+$ ]]; then
  echo "TEL_TP2_BATCH_MAXIMUM_ORDER must be an integer" >&2
  exit 65
fi
if (( start_order < 0 || end_order > maximum_order || start_order > end_order )); then
  echo "invalid order range: expected 0 <= START_ORDER <= END_ORDER <= $maximum_order" >&2
  exit 64
fi
if [[ -n "$(git -C "$repo_root" status --short)" ]]; then
  echo "TP=2 batch execution requires a clean repository checkout" >&2
  git -C "$repo_root" status --short >&2
  exit 65
fi
if [[ ! -x "$attempt_runner" || ! -r "$campaign_lock" ]]; then
  echo "TP=2 attempt runner or campaign lock is unavailable" >&2
  exit 66
fi

locked_campaign_sha=$(python3 - "$campaign_lock" "$maximum_order" <<'PY'
import json
import sys

lock = json.load(open(sys.argv[1], encoding="utf-8"))
maximum = int(sys.argv[2])
if lock.get("run_count") != maximum + 1:
    raise SystemExit("maximum order disagrees with campaign run_count")
if {row["order"] for row in lock["run_order"]} != set(range(maximum + 1)):
    raise SystemExit("campaign orders are not contiguous")
print(lock["lock_sha256"])
PY
)

batch_tag="$(date -u +%Y%m%dT%H%M%SZ)-$$"
batch_dir="$results_root/$batch_result_domain/orders-${start_order}-${end_order}/batch-$batch_tag"
mkdir -p "$batch_dir"
batch_log="$batch_dir/batch.events.log"
batch_summary="$batch_dir/batch.summary.json"
touch "$batch_log"

emit() { printf '%s\n' "$*" | tee -a "$batch_log"; }

run_id_for_order() {
  python3 - "$campaign_lock" "$1" <<'PY'
import json
import sys
lock = json.load(open(sys.argv[1], encoding="utf-8"))
order = int(sys.argv[2])
run = next((row for row in lock["run_order"] if row["order"] == order), None)
if run is None:
    raise SystemExit(f"order {order} is absent from the frozen TP=2 lock")
print(run["run_id"])
PY
}

accepted_attempt_exists() {
  python3 - "$1" "$2" "$locked_campaign_sha" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
run_id = sys.argv[2]
campaign_sha = sys.argv[3]
accepted = []
for path in sorted(root.glob("attempt-*/alignment.json")):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        continue
    if (
        value.get("qc_pass") is True
        and value.get("measurement") == "aligned-frozen-tp2-decode-repeat"
        and value.get("run", {}).get("run_id") == run_id
        and value.get("campaign_lock_sha256") == campaign_sha
        and value.get("tensor_parallel_size") == 2
        and value.get("host_gpu_indices") == [0, 1]
    ):
        accepted.append(path)
raise SystemExit(0 if len(accepted) == 1 else 1)
PY
}

write_summary() {
  local status=$1
  local accepted_count=$2
  local skipped_count=$3
  local failed_csv=$4
  python3 - "$batch_summary" "$status" "$start_order" "$end_order" \
    "$accepted_count" "$skipped_count" "$failed_csv" "$locked_campaign_sha" \
    "$(git -C "$repo_root" rev-parse HEAD)" "$measurement" <<'PY'
import json
import os
import pathlib
import sys
from datetime import datetime, timezone

path = pathlib.Path(sys.argv[1])
failed = [int(value) for value in sys.argv[7].split(",") if value]
report = {
    "schema_version": 1,
    "measurement": sys.argv[10],
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": sys.argv[2],
    "start_order": int(sys.argv[3]),
    "end_order": int(sys.argv[4]),
    "accepted_in_this_invocation": int(sys.argv[5]),
    "already_accepted_and_skipped": int(sys.argv[6]),
    "failed_orders": failed,
    "campaign_lock_sha256": sys.argv[8],
    "repository_commit": sys.argv[9],
    "failure_policy": "continue-preserve-failed-attempts",
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
}

emit "batch_start_order=$start_order"
emit "batch_end_order=$end_order"
emit "host_gpu_indices=0,1"
emit "tensor_parallel_size=2"
emit "repository_commit=$(git -C "$repo_root" rev-parse HEAD)"
emit "batch_dir=$batch_dir"
emit "failure_policy=continue-preserve-failed-attempts"

accepted_count=0
skipped_count=0
failed_orders=()
for ((order = start_order; order <= end_order; order++)); do
  run_id=$(run_id_for_order "$order")
  run_root="$results_root/$result_domain/$run_id"
  if accepted_attempt_exists "$run_root" "$run_id"; then
    emit "batch_order=$order run_id=$run_id status=already-accepted"
    ((skipped_count += 1))
    continue
  fi
  emit "batch_order=$order run_id=$run_id status=starting"
  if "$attempt_runner" "$order"; then
    if accepted_attempt_exists "$run_root" "$run_id"; then
      ((accepted_count += 1))
      emit "batch_order=$order run_id=$run_id status=accepted"
    else
      failed_orders+=("$order")
      emit "batch_order=$order run_id=$run_id status=missing-accepted-alignment-preserved-continuing"
    fi
  else
    runner_rc=$?
    failed_orders+=("$order")
    emit "batch_order=$order run_id=$run_id status=failed-preserved-continuing rc=$runner_rc"
  fi
done

failed_csv=$(IFS=,; echo "${failed_orders[*]}")
if (( ${#failed_orders[@]} == 0 )); then
  status=complete
else
  status=complete-with-preserved-failures
fi
write_summary "$status" "$accepted_count" "$skipped_count" "$failed_csv"
emit "batch_status=$status"
if [[ -n "$failed_csv" ]]; then
  emit "batch_failed_orders=$failed_csv"
fi
emit "batch_completed_orders=${start_order}-${end_order}"
emit "batch_summary=$batch_summary"
if [[ -n "$failed_csv" ]]; then
  exit 2
fi
