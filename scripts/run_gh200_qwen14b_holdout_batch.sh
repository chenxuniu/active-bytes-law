#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 START_ORDER END_ORDER [GPU_INDEX]" >&2
  exit 64
fi

start_order=$1
end_order=$2
gpu_index=${3:-0}
replication_name=${TEL_REPLICATION_NAME:-Qwen2.5-14B}
run_count=${TEL_REPLICATION_RUN_COUNT:-30}
for value in "$start_order" "$end_order"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]] || (( value < 0 || value >= run_count )); then
    echo "orders must be integers from 0 through $((run_count - 1))" >&2
    exit 64
  fi
done
if (( start_order > end_order )); then
  echo "START_ORDER must not exceed END_ORDER" >&2
  exit 64
fi
if [[ "$gpu_index" != "0" ]]; then
  echo "the released $replication_name holdout is bound to GPU index 0" >&2
  exit 64
fi

repo_root=${TEL_REPO_ROOT:-/srv/token-energy-law/repo}
results_root=${TEL_RESULTS_ROOT:-/srv/token-energy-law/results}
campaign_lock=${TEL_REPLICATION_CAMPAIGN_LOCK:-$repo_root/results/manifests/gh200-qwen2p5-14b-holdout.lock.json}
attempt_runner=${TEL_Q14_HOLDOUT_ATTEMPT_RUNNER:-$repo_root/scripts/run_gh200_qwen14b_holdout_attempt.sh}
result_domain=${TEL_REPLICATION_RESULT_DOMAIN:-qwen14b-holdout}
batch_domain=${TEL_REPLICATION_BATCH_DOMAIN:-qwen14b-holdout-batch-runs}
batch_measurement=${TEL_REPLICATION_BATCH_MEASUREMENT:-gh200-qwen2p5-14b-holdout-batch-execution-summary}
locked_campaign_sha=$(python3 - "$campaign_lock" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["lock_sha256"])
PY
)

if [[ -n "$(git -C "$repo_root" status --short)" ]]; then
  echo "$replication_name holdout requires a clean repository checkout" >&2
  git -C "$repo_root" status --short >&2
  exit 65
fi
if [[ ! -x "$attempt_runner" ]]; then
  echo "attempt runner is missing or not executable: $attempt_runner" >&2
  exit 65
fi

sudo -v || exit $?
(
  while true; do
    sleep 60
    sudo -n -v || exit 1
  done
) &
sudo_keepalive_pid=$!
cleanup() {
  kill "$sudo_keepalive_pid" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

batch_tag="$(date -u +%Y%m%dT%H%M%SZ)-$$"
batch_dir="$results_root/$batch_domain/orders-${start_order}-${end_order}/batch-$batch_tag"
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
for path in sorted(root.glob("attempt-*/alignment.json")):
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

write_summary() {
  python3 - "$batch_summary" "$start_order" "$end_order" "$accepted_count" \
    "$skipped_count" "$failed_orders_csv" "$locked_campaign_sha" \
    "$(git -C "$repo_root" rev-parse HEAD)" <<'PY'
import json
import os
import pathlib
import sys
from datetime import datetime, timezone

path = pathlib.Path(sys.argv[1])
failed = [int(value) for value in sys.argv[6].split(",") if value]
report = {
    "schema_version": 1,
    "measurement": os.environ["TEL_BATCH_MEASUREMENT"],
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "complete" if not failed else "complete-with-preserved-failures",
    "start_order": int(sys.argv[2]),
    "end_order": int(sys.argv[3]),
    "accepted_in_this_invocation": int(sys.argv[4]),
    "already_accepted_and_skipped": int(sys.argv[5]),
    "failed_orders": failed,
    "campaign_lock_sha256": sys.argv[7],
    "repository_commit": sys.argv[8],
    "failure_policy": "continue-preserve-failed-attempts",
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
}

emit "batch_start_order=$start_order"
emit "batch_end_order=$end_order"
emit "gpu_index=$gpu_index"
emit "repository_commit=$(git -C "$repo_root" rev-parse HEAD)"
emit "batch_dir=$batch_dir"
emit "failure_policy=continue-preserve-failed-attempts"

export TEL_BATCH_MEASUREMENT="$batch_measurement"

accepted_count=0
skipped_count=0
failed_orders_csv=""
for ((order = start_order; order <= end_order; order++)); do
  run_id=$(run_id_for_order "$order")
  run_root="$results_root/$result_domain/$run_id"
  if accepted_attempt_exists "$run_root" "$run_id" "$locked_campaign_sha"; then
    emit "batch_order=$order run_id=$run_id status=already-accepted"
    ((skipped_count += 1))
    continue
  fi
  emit "batch_order=$order run_id=$run_id status=starting"
  if "$attempt_runner" "$order" "$gpu_index"; then
    runner_rc=0
  else
    runner_rc=$?
  fi
  if (( runner_rc == 0 )) && accepted_attempt_exists "$run_root" "$run_id" "$locked_campaign_sha"; then
    ((accepted_count += 1))
    emit "batch_order=$order run_id=$run_id status=accepted"
  else
    failed_orders_csv="${failed_orders_csv:+$failed_orders_csv,}$order"
    emit "batch_order=$order run_id=$run_id status=failed-preserved-continuing rc=$runner_rc"
  fi
done

write_summary
if [[ -z "$failed_orders_csv" ]]; then
  emit "batch_status=complete"
else
  emit "batch_status=complete-with-preserved-failures"
  emit "batch_failed_orders=$failed_orders_csv"
fi
emit "batch_completed_orders=${start_order}-${end_order}"
emit "batch_summary=$batch_summary"
[[ -z "$failed_orders_csv" ]]
