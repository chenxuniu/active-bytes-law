#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-2]$ ]]; then
  echo "usage: $0 RUN_ORDER  # RUN_ORDER is 0, 1, or 2" >&2
  exit 64
fi

run_order=$1
repo_root=${TEL_REPO_ROOT:-/srv/token-energy-law/repo}
results_root=${TEL_RESULTS_ROOT:-/srv/token-energy-law/results}
campaign_lock="$repo_root/results/manifests/gh200-tp2-nvlink-qualification.lock.json"
addendum="$repo_root/configs/addenda/gh200-tp2-nvlink-v1.json"
source_model="$repo_root/configs/addenda/gh200-v2-duration-holdout-v1.json"

(
  cd "$repo_root/configs/addenda"
  sha256sum -c gh200-tp2-nvlink-v1.json.sha256
)
(
  cd "$repo_root/results/manifests"
  sha256sum -c gh200-tp2-nvlink-qualification.lock.json.sha256
)

preflight=""
if [[ -d "$results_root/tp2-preflight" ]]; then
  preflight=$(find "$results_root/tp2-preflight" -mindepth 2 -maxdepth 2 \
    -name preflight.log -type f 2>/dev/null | sort | tail -1)
fi
if [[ -z "$preflight" ]]; then
  echo "run check_gh200_tp2_preflight.sh after restoring the node, before qualification" >&2
  exit 66
fi
if [[ ! -r "$(dirname "$preflight")/artifacts.sha256" ]]; then
  echo "latest TP=2 preflight lacks its artifact manifest" >&2
  exit 66
fi
sha256sum -c "$(dirname "$preflight")/artifacts.sha256" >/dev/null
python3 - "$preflight" "$(git -C "$repo_root" rev-parse HEAD)" <<'PY'
import sys

text = open(sys.argv[1], encoding="utf-8").read()
required = [
    f"repository_commit={sys.argv[2]}",
    "host_gpu_indices=0,1",
    "tensor_parallel_size=2",
    "power_limit_w_per_gpu=700",
    "dvfs_manipulation=false",
    "visible_devices: 2",
]
missing = [value for value in required if value not in text]
if missing:
    raise SystemExit(f"latest TP=2 preflight does not match this checkout: {missing}")
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

if python3 - "$results_root/tp2-qualification/$run_id" "$run_id" <<'PY'
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
  echo "qualification order $run_order already has an accepted attempt: $run_id" >&2
  exit 65
fi

export TEL_TP2_CAMPAIGN_LOCK="$campaign_lock"
export TEL_TP2_EXECUTION_ADDENDUM="$addendum"
export TEL_TP2_RESULT_DOMAIN=tp2-qualification
export TEL_TP2_PAPER_CANDIDATE=0
export TEL_TP2_EXTRA_CONTRACTS="$source_model:$preflight"

echo "tp2_qualification_order=$run_order"
echo "tp2_qualification_run_id=$run_id"
echo "paper_candidate_measurement=false"

exec "$repo_root/scripts/run_gh200_tp2_attempt_core.sh" \
  "$run_id" "$gpu_memory_utilization"
