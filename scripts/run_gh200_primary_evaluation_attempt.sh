#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 RUN_ORDER [GPU_INDEX]" >&2
  exit 64
fi

run_order=$1
gpu_index=${2:-0}
if [[ "$gpu_index" != "0" ]]; then
  echo "the frozen GH200 evaluation campaign is bound to GPU index 0" >&2
  exit 64
fi
if [[ ! "$run_order" =~ ^[0-9]+$ ]] || (( run_order < 0 || run_order >= 30 )); then
  echo "RUN_ORDER must be an integer from 0 through 29" >&2
  exit 64
fi

repo_root=${TEL_REPO_ROOT:-/srv/token-energy-law/repo}
results_root=${TEL_RESULTS_ROOT:-/srv/token-energy-law/results}
campaign_lock="$repo_root/results/manifests/gh200-primary-bf16-evaluation.lock.json"
execution_addendum="$repo_root/configs/addenda/gh200-primary-bf16-v1.json"
release_record="$repo_root/configs/addenda/gh200-primary-bf16-evaluation-release-v1.json"
freeze_dir=${TEL_IDENTIFICATION_FREEZE_DIR:-/srv/token-energy-law/results/primary-identification-freeze/20260902T170315Z}
result_domain=primary-evaluation

verification_tag=$(date -u +%Y%m%dT%H%M%SZ)
verification_dir="$results_root/primary-evaluation-release-verifications"
verification_json="$verification_dir/release-verification-${verification_tag}-order-${run_order}.json"
mkdir -p "$verification_dir"

python3 "$repo_root/scripts/verify_gh200_evaluation_release.py" \
  --release-record "$release_record" \
  --identification-freeze-dir "$freeze_dir" \
  --evaluation-lock "$campaign_lock" \
  --output-json "$verification_json"

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
locked_campaign_sha=$(python3 - "$campaign_lock" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["lock_sha256"])
PY
)

accepted_attempt_exists() {
  python3 - "$1" "$2" "$locked_campaign_sha" <<'PY'
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

current_dir="$results_root/$result_domain/$run_id"
if accepted_attempt_exists "$current_dir" "$run_id"; then
  echo "run order $run_order already has an accepted attempt: $run_id" >&2
  exit 65
fi
if [[ -n "$previous_run_id" ]]; then
  previous_dir="$results_root/$result_domain/$previous_run_id"
  if ! accepted_attempt_exists "$previous_dir" "$previous_run_id"; then
    echo "previous frozen evaluation run is not yet accepted: $previous_run_id" >&2
    exit 65
  fi
fi

export TEL_CAMPAIGN_LOCK="$campaign_lock"
export TEL_EXECUTION_ADDENDUM="$execution_addendum"
export TEL_RESULT_DOMAIN="$result_domain"
export TEL_REQUIRE_CLEAN_REPO=1
export TEL_EXTRA_CONTRACTS="$release_record:$freeze_dir/coefficient-artifact.json:$freeze_dir/discrepancy-envelope.json:$freeze_dir/accepted-runs.csv:$freeze_dir/identification-freeze-summary.json:$verification_json"

echo "frozen_evaluation_order=$run_order"
echo "frozen_evaluation_run_id=$run_id"
echo "evaluation_release_verification=$verification_json"

exec "$repo_root/scripts/run_gh200_pilot_attempt.sh" \
  "$run_id" \
  "$gpu_memory_utilization" \
  "$gpu_index"
