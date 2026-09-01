#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 RUN_ORDER [GPU_INDEX]" >&2
  exit 64
fi

run_order=$1
gpu_index=${2:-0}
if [[ ! "$run_order" =~ ^[0-9]+$ ]] || (( run_order < 0 || run_order >= 45 )); then
  echo "RUN_ORDER must be an integer from 0 through 44" >&2
  exit 64
fi

repo_root=${TEL_REPO_ROOT:-/srv/token-energy-law/repo}
results_root=${TEL_RESULTS_ROOT:-/srv/token-energy-law/results}
campaign_lock="$repo_root/results/manifests/gh200-primary-bf16.lock.json"
execution_addendum="$repo_root/configs/addenda/gh200-primary-bf16-v1.json"
result_domain=primary-identification

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
if order > 0:
    previous = next(row for row in lock["run_order"] if row["order"] == order - 1)
    print(previous["run_id"])
else:
    print("")
PY
)

run_id=${resolved[0]}
gpu_memory_utilization=${resolved[1]}
previous_run_id=${resolved[2]}

accepted_attempt_exists() {
  python3 - "$1" <<'PY'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
for path in root.glob("attempt-*/alignment.json"):
    try:
        if json.loads(path.read_text(encoding="utf-8")).get("qc_pass") is True:
            raise SystemExit(0)
    except (OSError, ValueError):
        pass
raise SystemExit(1)
PY
}

current_dir="$results_root/$result_domain/$run_id"
if accepted_attempt_exists "$current_dir"; then
  echo "run order $run_order already has an accepted attempt: $run_id" >&2
  exit 65
fi
if [[ -n "$previous_run_id" ]]; then
  previous_dir="$results_root/$result_domain/$previous_run_id"
  if ! accepted_attempt_exists "$previous_dir"; then
    echo "previous frozen run is not yet accepted: $previous_run_id" >&2
    exit 65
  fi
fi

export TEL_CAMPAIGN_LOCK="$campaign_lock"
export TEL_EXECUTION_ADDENDUM="$execution_addendum"
export TEL_RESULT_DOMAIN="$result_domain"
export TEL_REQUIRE_CLEAN_REPO=1

echo "frozen_run_order=$run_order"
echo "frozen_run_id=$run_id"

exec "$repo_root/scripts/run_gh200_pilot_attempt.sh" \
  "$run_id" \
  "$gpu_memory_utilization" \
  "$gpu_index"
