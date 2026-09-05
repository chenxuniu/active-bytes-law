#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]] || (( $1 < 0 || $1 > 29 )); then
  echo "usage: $0 RUN_ORDER  # RUN_ORDER is 0 through 29" >&2
  exit 64
fi

run_order=$1
repo_root=${TEL_REPO_ROOT:-/srv/token-energy-law/repo}
results_root=${TEL_RESULTS_ROOT:-/srv/token-energy-law/results}
campaign_lock="$repo_root/results/manifests/gh200-tp2-nvlink-holdout.lock.json"
identification_lock="$repo_root/results/manifests/gh200-tp2-nvlink-identification.lock.json"
addendum="$repo_root/configs/addenda/gh200-tp2-nvlink-v1.json"
release_record="$repo_root/configs/addenda/gh200-tp2-nvlink-holdout-release-v1.json"
freeze_dir=${TEL_TP2_IDENTIFICATION_FREEZE_DIR:-}

if [[ -z "$freeze_dir" ]]; then
  echo "TEL_TP2_IDENTIFICATION_FREEZE_DIR is required for sealed holdout execution" >&2
  exit 66
fi
if [[ ! -r "$release_record" || ! -r "$release_record.sha256" ]]; then
  echo "TP=2 holdout remains sealed: content-addressed release record is absent" >&2
  exit 66
fi
read -r _ expected_release_name <"$release_record.sha256"
if [[ "$expected_release_name" != "$(basename "$release_record")" ]]; then
  echo "TP=2 release sidecar filename mismatch" >&2
  exit 65
fi

verification_tag=$(date -u +%Y%m%dT%H%M%SZ)
verification_dir="$results_root/tp2-holdout-release-verifications"
verification_json="$verification_dir/release-verification-${verification_tag}-order-${run_order}.json"
mkdir -p "$verification_dir"
python3 "$repo_root/scripts/verify_gh200_tp2_holdout_release.py" \
  --release-record "$release_record" \
  --identification-freeze-dir "$freeze_dir" \
  --identification-lock "$identification_lock" \
  --holdout-lock "$campaign_lock" \
  --execution-addendum "$addendum" \
  --output-json "$verification_json"

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

if python3 - "$results_root/tp2-holdout/$run_id" "$run_id" <<'PY'
import json
import pathlib
import sys

accepted = []
for path in pathlib.Path(sys.argv[1]).glob("attempt-*/alignment.json"):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        continue
    if (
        value.get("qc_pass") is True
        and value.get("measurement") == "aligned-frozen-tp2-decode-repeat"
        and value.get("run", {}).get("run_id") == sys.argv[2]
    ):
        accepted.append(path)
raise SystemExit(0 if accepted else 1)
PY
then
  echo "holdout order $run_order already has an accepted attempt: $run_id" >&2
  exit 65
fi

export TEL_TP2_CAMPAIGN_LOCK="$campaign_lock"
export TEL_TP2_EXECUTION_ADDENDUM="$addendum"
export TEL_TP2_RESULT_DOMAIN=tp2-holdout
export TEL_TP2_PAPER_CANDIDATE=1
export TEL_TP2_EXTRA_CONTRACTS="$identification_lock:$release_record:$freeze_dir/coefficient-artifact.json:$freeze_dir/discrepancy-envelope.json:$freeze_dir/accepted-runs.csv:$freeze_dir/identification-freeze-summary.json:$verification_json"

echo "tp2_holdout_order=$run_order"
echo "tp2_holdout_run_id=$run_id"
echo "holdout_release_verification=$verification_json"

exec "$repo_root/scripts/run_gh200_tp2_attempt_core.sh" \
  "$run_id" "$gpu_memory_utilization"
