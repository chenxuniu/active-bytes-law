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

echo "batch_start_order=$start_order"
echo "batch_end_order=$end_order"
echo "gpu_index=$gpu_index"
echo "repository_commit=$(git -C "$repo_root" rev-parse HEAD)"

for ((order = start_order; order <= end_order; order++)); do
  run_id=$(run_id_for_order "$order")
  run_root="$results_root/$result_domain/$run_id"
  if accepted_attempt_exists "$run_root" "$run_id" "$locked_campaign_sha"; then
    echo "batch_order=$order run_id=$run_id status=already-accepted"
    continue
  fi

  echo "batch_order=$order run_id=$run_id status=starting"
  if "$attempt_runner" "$order" "$gpu_index"; then
    :
  else
    runner_rc=$?
    echo "batch_order=$order run_id=$run_id status=failed rc=$runner_rc" >&2
    exit "$runner_rc"
  fi
  if ! accepted_attempt_exists "$run_root" "$run_id" "$locked_campaign_sha"; then
    echo "batch_order=$order run_id=$run_id status=missing-accepted-traffic" >&2
    exit 65
  fi
  echo "batch_order=$order run_id=$run_id status=accepted"
done

echo "batch_status=complete"
echo "batch_completed_orders=${start_order}-${end_order}"
