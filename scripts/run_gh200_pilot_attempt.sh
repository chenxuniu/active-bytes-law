#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 RUN_ID GPU_MEMORY_UTILIZATION [GPU_INDEX]" >&2
  exit 64
fi

run_id=$1
gpu_memory_utilization=$2
gpu_index=${3:-0}
repo_root=${TEL_REPO_ROOT:-/srv/token-energy-law/repo}
results_root=${TEL_RESULTS_ROOT:-/srv/token-energy-law/results}
hf_cache=${TEL_HF_CACHE:-/srv/token-energy-law/hf-cache}
container_home=${TEL_CONTAINER_HOME:-/srv/token-energy-law/container-home}
campaign_lock=${TEL_CAMPAIGN_LOCK:-$repo_root/results/manifests/pilot.lock.json}
execution_addendum=${TEL_EXECUTION_ADDENDUM:-$repo_root/configs/addenda/gh200-pilot-memory-admission-v1.json}
result_domain=${TEL_RESULT_DOMAIN:-pilot}
require_clean_repo=${TEL_REQUIRE_CLEAN_REPO:-0}

for path in "$campaign_lock" "$execution_addendum"; do
  if [[ ! -r "$path" ]]; then
    echo "required frozen contract is missing: $path" >&2
    exit 66
  fi
done

if [[ ! "$result_domain" =~ ^[a-z0-9][a-z0-9._-]*$ ]]; then
  echo "invalid result domain: $result_domain" >&2
  exit 65
fi
if [[ "$require_clean_repo" == "1" ]] && [[ -n "$(git -C "$repo_root" status --short)" ]]; then
  echo "primary execution requires a clean repository checkout" >&2
  git -C "$repo_root" status --short >&2
  exit 65
fi

run_contract=$(python3 - "$campaign_lock" "$run_id" <<'PY'
import json
import sys

lock = json.load(open(sys.argv[1], encoding="utf-8"))
matches = [row for row in lock["run_order"] if row["run_id"] == sys.argv[2]]
if len(matches) != 1:
    raise SystemExit("run ID does not occur exactly once in the frozen lock")
run = matches[0]
p = run["parameters"]
print("\t".join([
    run["cell_id"],
    str(p["target_mean_attended_history_tokens"]),
    str(p["target_batch"]),
    p["kv_cache_dtype"],
    p["attention_backend"],
    p["container_image"],
    str(p.get("power_limit_w", "")),
]))
PY
) || exit $?

IFS=$'\t' read -r cell_id target_mean target_batch kv_dtype attention_backend image locked_power_limit_w <<<"$run_contract"

case "$campaign_lock" in
  "$repo_root"/*) campaign_lock_relative=${campaign_lock#"$repo_root"/} ;;
  *)
    echo "campaign lock must be inside the repository: $campaign_lock" >&2
    exit 65
    ;;
esac

if [[ "$kv_dtype" != "bf16" ]]; then
  echo "this orchestrator refuses FP8 pilot energy until a promoted runtime exists" >&2
  exit 65
fi
if [[ "$cell_id" == "p-c-l16384-b32-bf16" && "$gpu_memory_utilization" != "0.8" && "$gpu_memory_utilization" != "0.80" ]]; then
  echo "P-C requires gpu_memory_utilization=0.80 under the frozen addendum" >&2
  exit 65
fi
if [[ "$cell_id" == "p-a-l4096-b8-bf16" && "$gpu_memory_utilization" != "0.5" && "$gpu_memory_utilization" != "0.50" ]]; then
  echo "P-A frozen pilot repeats require gpu_memory_utilization=0.50" >&2
  exit 65
fi

locked_utilization=$(python3 - "$campaign_lock" "$run_id" <<'PY'
import json
import sys
lock = json.load(open(sys.argv[1], encoding="utf-8"))
run = next(row for row in lock["run_order"] if row["run_id"] == sys.argv[2])
value = run["parameters"].get("gpu_memory_utilization")
print("" if value is None else value)
PY
)
if [[ -n "$locked_utilization" ]] && ! python3 - "$locked_utilization" "$gpu_memory_utilization" <<'PY'
import math
import sys
raise SystemExit(0 if math.isclose(float(sys.argv[1]), float(sys.argv[2]), rel_tol=0.0, abs_tol=1e-12) else 1)
PY
then
  echo "gpu_memory_utilization does not match the campaign lock" >&2
  exit 65
fi

if [[ -n "$locked_power_limit_w" ]]; then
  observed_power_limit_w=$(nvidia-smi -i "$gpu_index" \
    --query-gpu=power.limit --format=csv,noheader,nounits)
  if ! python3 - "$locked_power_limit_w" "$observed_power_limit_w" <<'PY'
import math
import sys
raise SystemExit(
    0
    if math.isclose(float(sys.argv[1]), float(sys.argv[2]), rel_tol=0.0, abs_tol=0.01)
    else 1
)
PY
  then
    echo "GPU power limit does not match the frozen run contract: expected ${locked_power_limit_w} W, observed ${observed_power_limit_w} W" >&2
    exit 65
  fi
fi

locked_addendum_sha=$(python3 - "$campaign_lock" "$run_id" <<'PY'
import json
import sys
lock = json.load(open(sys.argv[1], encoding="utf-8"))
run = next(row for row in lock["run_order"] if row["run_id"] == sys.argv[2])
value = run["parameters"].get("execution_addendum_sha256")
print("" if value is None else value)
PY
)
if [[ -n "$locked_addendum_sha" ]]; then
  observed_addendum_sha=$(sha256sum "$execution_addendum" | awk '{print $1}')
  if [[ "$observed_addendum_sha" != "$locked_addendum_sha" ]]; then
    echo "execution addendum does not match the frozen run contract" >&2
    exit 65
  fi
fi

tag=$(date -u +%Y%m%dT%H%M%SZ)
attempt_dir="$results_root/$result_domain/$run_id/attempt-$tag"
relative_attempt="$result_domain/$run_id/attempt-$tag"
mkdir -p "$attempt_dir"

ready_file="$attempt_dir/engine.ready.json"
gate_file="$attempt_dir/release"
repeat_json="$attempt_dir/repeat.json"
runner_log="$attempt_dir/runner.log"
telemetry_jsonl="$attempt_dir/telemetry.jsonl"
telemetry_summary="$attempt_dir/telemetry.summary.json"
telemetry_ready="$attempt_dir/telemetry.ready.json"
telemetry_stop="$attempt_dir/telemetry.stop"
collector_log="$attempt_dir/collector.log"
alignment_json="$attempt_dir/alignment.json"
alignment_log="$attempt_dir/alignment.log"
runner_name="tel-runner-$tag"
collector_name="tel-collector-$tag"

cleanup() {
  sudo docker rm -f "$runner_name" "$collector_name" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

git -C "$repo_root" rev-parse HEAD >"$attempt_dir/repository.commit.txt"
sha256sum "$campaign_lock" "$execution_addendum" >"$attempt_dir/contracts.sha256.txt"
nvidia-smi -i "$gpu_index" \
  --query-gpu=index,name,memory.used,memory.free,power.draw,power.limit,temperature.gpu \
  --format=csv >"$attempt_dir/gpu.before.csv"

sudo -v || exit $?

sudo docker run --rm \
  --name "$runner_name" \
  --ipc=host \
  --gpus "device=$gpu_index" \
  -e HOME=/workspace/home \
  -e USER=root \
  -e LOGNAME=root \
  -e HF_HOME=/workspace/hf-cache \
  -e VLLM_USE_V1=0 \
  -e VLLM_ATTENTION_BACKEND="$attention_backend" \
  -v "$repo_root:/workspace/active-bytes-law:ro" \
  -v "$results_root:/workspace/results" \
  -v "$hf_cache:/workspace/hf-cache" \
  -v "$container_home:/workspace/home" \
  --entrypoint python3 \
  "$image" \
  /workspace/active-bytes-law/scripts/run_pilot_repeat.py \
  --campaign-lock "/workspace/active-bytes-law/$campaign_lock_relative" \
  --run-id "$run_id" \
  --gpu-memory-utilization "$gpu_memory_utilization" \
  --ready-file "/workspace/results/$relative_attempt/engine.ready.json" \
  --start-gate-file "/workspace/results/$relative_attempt/release" \
  --gate-timeout-seconds 1800 \
  --output-json "/workspace/results/$relative_attempt/repeat.json" \
  >"$runner_log" 2>&1 &
runner_pid=$!

ready_deadline=$((SECONDS + 1200))
while [[ ! -f "$ready_file" ]]; do
  if ! kill -0 "$runner_pid" 2>/dev/null; then
    wait "$runner_pid"
    runner_rc=$?
    echo "runner exited before ENGINE_READY (rc=$runner_rc)" >&2
    tail -120 "$runner_log" >&2
    exit "$runner_rc"
  fi
  if (( SECONDS >= ready_deadline )); then
    echo "timed out waiting for ENGINE_READY" >&2
    exit 124
  fi
  sleep 2
done

sudo docker run --rm \
  --name "$collector_name" \
  --gpus "device=$gpu_index" \
  -v "$repo_root:/workspace/active-bytes-law:ro" \
  -v "$results_root:/workspace/results" \
  --entrypoint python3 \
  "$image" \
  /workspace/active-bytes-law/scripts/collect_nvml_scoped.py \
  --duration-seconds 180 \
  --interval-ms 10 \
  --maximum-gap-ms 50 \
  --module-counter-error-limit 0.02 \
  --ready-file "/workspace/results/$relative_attempt/telemetry.ready.json" \
  --stop-file "/workspace/results/$relative_attempt/telemetry.stop" \
  --output-jsonl "/workspace/results/$relative_attempt/telemetry.jsonl" \
  --summary-json "/workspace/results/$relative_attempt/telemetry.summary.json" \
  >"$collector_log" 2>&1 &
collector_pid=$!

collector_deadline=$((SECONDS + 60))
while [[ ! -f "$telemetry_ready" ]]; do
  if ! kill -0 "$collector_pid" 2>/dev/null; then
    wait "$collector_pid"
    collector_rc=$?
    echo "collector exited before TELEMETRY_READY (rc=$collector_rc)" >&2
    tail -120 "$collector_log" >&2
    exit "$collector_rc"
  fi
  if (( SECONDS >= collector_deadline )); then
    echo "timed out waiting for TELEMETRY_READY" >&2
    exit 124
  fi
  sleep 1
done

touch "$gate_file"
wait "$runner_pid"
runner_rc=$?
sleep 1
touch "$telemetry_stop"
wait "$collector_pid"
collector_rc=$?

sudo chown -R "$(id -u):$(id -g)" "$attempt_dir"

alignment_rc=125
if [[ $runner_rc -eq 0 && -s "$repeat_json" && -s "$telemetry_jsonl" ]]; then
  python3 "$repo_root/scripts/align_pilot_repeat.py" \
    --telemetry-jsonl "$telemetry_jsonl" \
    --repeat-json "$repeat_json" \
    --gpu-index 0 \
    --maximum-gap-ms 50 \
    --module-counter-error-limit 0.02 \
    --output-json "$alignment_json" \
    >"$alignment_log" 2>&1
  alignment_rc=$?
fi

nvidia-smi -i "$gpu_index" \
  --query-gpu=index,name,memory.used,memory.free,power.draw,power.limit,temperature.gpu \
  --format=csv >"$attempt_dir/gpu.after.csv"

find "$attempt_dir" -maxdepth 1 -type f ! -name artifacts.sha256 -print0 \
  | sort -z \
  | xargs -0 sha256sum >"$attempt_dir/artifacts.sha256"

{
  echo "run_id=$run_id"
  echo "cell_id=$cell_id"
  echo "target_mean_attended_history_tokens=$target_mean"
  echo "target_batch=$target_batch"
  echo "gpu_memory_utilization=$gpu_memory_utilization"
  echo "runner_rc=$runner_rc"
  echo "collector_rc=$collector_rc"
  echo "alignment_rc=$alignment_rc"
  echo "attempt_dir=$attempt_dir"
} | tee "$attempt_dir/status.txt"

if [[ $runner_rc -ne 0 || $alignment_rc -ne 0 ]]; then
  echo "attempt retained but not accepted; inspect runner.log and alignment.log" >&2
  exit 2
fi
if [[ $collector_rc -ne 0 && $collector_rc -ne 2 ]]; then
  echo "collector failed before producing an interpretable aligned attempt" >&2
  exit 2
fi

python3 - "$alignment_json" <<'PY'
import json
import sys

x = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps({
    "qc_pass": x["qc_pass"],
    "qc_reasons": x["qc_reasons"],
    "run": x["run"],
    "episode_count": x["episode_count"],
    "totals": x["totals"],
    "active_bytes": x["active_bytes"],
}, indent=2, sort_keys=True))
PY
