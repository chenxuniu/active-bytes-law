#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 RUN_ORDER [GPU_INDEX]" >&2
  exit 64
fi

run_order=$1
gpu_index=${2:-0}
replication_name=${TEL_REPLICATION_NAME:-Qwen2.5-14B}
run_count=${TEL_REPLICATION_RUN_COUNT:-30}
if [[ "$gpu_index" != "0" ]]; then
  echo "the released $replication_name holdout is bound to GPU index 0" >&2
  exit 64
fi
if [[ ! "$run_order" =~ ^[0-9]+$ ]] || (( run_order < 0 || run_order >= run_count )); then
  echo "RUN_ORDER must be an integer from 0 through $((run_count - 1))" >&2
  exit 64
fi

repo_root=${TEL_REPO_ROOT:-/srv/token-energy-law/repo}
results_root=${TEL_RESULTS_ROOT:-/srv/token-energy-law/results}
campaign_lock=${TEL_REPLICATION_CAMPAIGN_LOCK:-$repo_root/results/manifests/gh200-qwen2p5-14b-holdout.lock.json}
execution_addendum=${TEL_REPLICATION_ADDENDUM:-$repo_root/configs/addenda/gh200-qwen2p5-14b-form-replication-v1.json}
release_record=${TEL_REPLICATION_RELEASE_RECORD:-$repo_root/configs/addenda/gh200-qwen2p5-14b-holdout-release-v1.json}
freeze_dir=${TEL_REPLICATION_IDENTIFICATION_FREEZE_DIR:-${TEL_Q14_IDENTIFICATION_FREEZE_DIR:-$results_root/qwen14b-identification-freeze/20260904T161637Z}}
result_domain=${TEL_REPLICATION_RESULT_DOMAIN:-qwen14b-holdout}
verification_domain=${TEL_REPLICATION_VERIFICATION_DOMAIN:-qwen14b-holdout-release-verifications}
verification_script=${TEL_REPLICATION_VERIFICATION_SCRIPT:-$repo_root/scripts/verify_gh200_qwen14b_holdout_release.py}

(
  cd "$(dirname "$execution_addendum")"
  sha256sum -c "$(basename "$execution_addendum").sha256"
  sha256sum -c "$(basename "$release_record").sha256"
)
(
  cd "$(dirname "$campaign_lock")"
  sha256sum -c "$(basename "$campaign_lock").sha256"
)

verification_tag=$(date -u +%Y%m%dT%H%M%SZ)
verification_dir="$results_root/$verification_domain"
verification_json="$verification_dir/release-verification-${verification_tag}-order-${run_order}.json"
mkdir -p "$verification_dir"

python3 "$verification_script" \
  --release-record "$release_record" \
  --identification-freeze-dir "$freeze_dir" \
  --holdout-lock "$campaign_lock" \
  --form-replication-addendum "$execution_addendum" \
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
parameters = run["parameters"]
if parameters.get("execution_state") != "sealed-unreleased":
    raise SystemExit("source holdout lock does not preserve its sealed state")
if parameters.get("requires_frozen_identification_release") is not True:
    raise SystemExit("source holdout lock does not require a release")
print(run["run_id"])
print(parameters["gpu_memory_utilization"])
PY
)

run_id=${resolved[0]}
gpu_memory_utilization=${resolved[1]}
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

export TEL_CAMPAIGN_LOCK="$campaign_lock"
export TEL_EXECUTION_ADDENDUM="$execution_addendum"
export TEL_RESULT_DOMAIN="$result_domain"
export TEL_REQUIRE_CLEAN_REPO=1
export TEL_EXTRA_CONTRACTS="$release_record:$freeze_dir/coefficient-artifact.json:$freeze_dir/discrepancy-envelope.json:$freeze_dir/accepted-runs.csv:$freeze_dir/identification-freeze-summary.json:$verification_json"

echo "released_replication_name=$replication_name"
echo "released_replication_holdout_order=$run_order"
echo "released_replication_holdout_run_id=$run_id"
echo "holdout_release_verification=$verification_json"

exec "$repo_root/scripts/run_gh200_pilot_attempt.sh" \
  "$run_id" \
  "$gpu_memory_utilization" \
  "$gpu_index"
