#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]] || (( $1 < 0 || $1 > 44 )); then
  echo "usage: $0 RUN_ORDER  # RUN_ORDER is 0 through 44" >&2
  exit 64
fi

run_order=$1
repo_root=${TEL_REPO_ROOT:-/srv/token-energy-law/repo}
results_root=${TEL_RESULTS_ROOT:-/srv/token-energy-law/results}
campaign_lock="$repo_root/results/manifests/gh200-tp2-nvlink-identification.lock.json"
qualification_lock="$repo_root/results/manifests/gh200-tp2-nvlink-qualification.lock.json"
addendum="$repo_root/configs/addenda/gh200-tp2-nvlink-v1.json"

python3 - "$qualification_lock" "$results_root/tp2-qualification" <<'PY'
import json
import pathlib
import sys

lock = json.load(open(sys.argv[1], encoding="utf-8"))
root = pathlib.Path(sys.argv[2])
issues = []
for run in lock["run_order"]:
    accepted = []
    for path in sorted((root / run["run_id"]).glob("attempt-*/alignment.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            value.get("qc_pass") is True
            and value.get("paper_candidate_measurement") is False
            and value.get("campaign_lock_sha256") == lock["lock_sha256"]
            and value.get("run", {}).get("run_id") == run["run_id"]
            and value.get("tensor_parallel_size") == 2
            and value.get("host_gpu_indices") == [0, 1]
        ):
            accepted.append(path)
    if len(accepted) != 1:
        issues.append(f"{run['run_id']}: expected exactly one accepted qualification; found {len(accepted)}")
if issues:
    raise SystemExit("TP=2 qualification gate failed: " + "; ".join(issues))
PY

readarray -t resolved < <(python3 - "$campaign_lock" "$run_order" <<'PY'
import json
import sys
lock = json.load(open(sys.argv[1], encoding="utf-8"))
order = int(sys.argv[2])
run = next(row for row in lock["run_order"] if row["order"] == order)
print(run["run_id"])
print(run["parameters"]["gpu_memory_utilization"])
PY
)
run_id=${resolved[0]}
gpu_memory_utilization=${resolved[1]}

if python3 - "$results_root/tp2-identification/$run_id" "$run_id" <<'PY'
import json
import pathlib
import sys
accepted = []
for path in pathlib.Path(sys.argv[1]).glob("attempt-*/alignment.json"):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("qc_pass") is True and value.get("run", {}).get("run_id") == sys.argv[2]:
            accepted.append(path)
    except (OSError, ValueError):
        pass
raise SystemExit(0 if accepted else 1)
PY
then
  echo "identification order $run_order already has an accepted attempt: $run_id" >&2
  exit 65
fi

export TEL_TP2_CAMPAIGN_LOCK="$campaign_lock"
export TEL_TP2_EXECUTION_ADDENDUM="$addendum"
export TEL_TP2_RESULT_DOMAIN=tp2-identification
export TEL_TP2_PAPER_CANDIDATE=1
export TEL_TP2_EXTRA_CONTRACTS="$qualification_lock"

echo "tp2_identification_order=$run_order"
echo "tp2_identification_run_id=$run_id"
echo "primary_outcome=sum_two_gpu_boards_joules_per_useful_token"

exec "$repo_root/scripts/run_gh200_tp2_attempt_core.sh" \
  "$run_id" "$gpu_memory_utilization"
