#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 RUN_ORDER [GPU_INDEX]" >&2
  exit 64
fi

run_order=$1
gpu_index=${2:-0}
if [[ "$gpu_index" != "0" ]]; then
  echo "the frozen GH200 V1 anchor campaign is bound to GPU index 0" >&2
  exit 64
fi
if [[ ! "$run_order" =~ ^[0-9]+$ ]] || (( run_order < 0 || run_order >= 60 )); then
  echo "RUN_ORDER must be an integer from 0 through 59" >&2
  exit 64
fi

repo_root=${TEL_REPO_ROOT:-/srv/token-energy-law/repo}
results_root=${TEL_RESULTS_ROOT:-/srv/token-energy-law/results}
hf_cache=${TEL_HF_CACHE:-/srv/token-energy-law/hf-cache}
container_home=${TEL_CONTAINER_HOME:-/srv/token-energy-law/container-home}
campaign_lock="$repo_root/results/manifests/gh200-v1-anchors.lock.json"
execution_addendum="$repo_root/configs/addenda/gh200-primary-bf16-v1.json"
profiler_amendment="$repo_root/configs/addenda/gh200-v1-profiler-replay-amendment-v1.json"
result_domain=v1-profiler-anchors

if [[ -n "$(git -C "$repo_root" status --short)" ]]; then
  echo "V1 execution requires a clean repository checkout" >&2
  git -C "$repo_root" status --short >&2
  exit 65
fi
(
  cd "$repo_root/configs/addenda"
  sha256sum -c \
    gh200-primary-bf16-v1.json.sha256 \
    gh200-v1-profiler-replay-amendment-v1.json.sha256
)
(
  cd "$repo_root/results/manifests"
  sha256sum -c gh200-v1-anchors.lock.json.sha256
)

readarray -t resolved < <(python3 - "$campaign_lock" "$run_order" <<'PY'
import json
import sys
lock = json.load(open(sys.argv[1], encoding="utf-8"))
order = int(sys.argv[2])
run = next(row for row in lock["run_order"] if row["order"] == order)
p = run["parameters"]
print(run["run_id"])
print(p["container_image"])
print(p["attention_backend"])
print(p["gpu_memory_utilization"])
print(p["power_limit_w"])
print(p["nvtx_range"])
print(",".join(p["profiler_metrics"]))
print(p["profiler_replay_mode"])
print(p["profiler_replay_amendment_sha256"])
print(p["execution_addendum_sha256"])
if order > 0:
    print(next(row for row in lock["run_order"] if row["order"] == order - 1)["run_id"])
else:
    print("")
PY
)

run_id=${resolved[0]}
image=${resolved[1]}
attention_backend=${resolved[2]}
gpu_memory_utilization=${resolved[3]}
locked_power_limit_w=${resolved[4]}
nvtx_range=${resolved[5]}
metrics=${resolved[6]}
profiler_replay_mode=${resolved[7]}
locked_profiler_amendment_sha=${resolved[8]}
locked_addendum_sha=${resolved[9]}
previous_run_id=${resolved[10]}

if [[ "$profiler_replay_mode" != "app-range" ]]; then
  echo "V1 profiler replay mode must be app-range after the frozen amendment" >&2
  exit 65
fi

observed_addendum_sha=$(sha256sum "$execution_addendum" | awk '{print $1}')
if [[ "$observed_addendum_sha" != "$locked_addendum_sha" ]]; then
  echo "execution addendum does not match the V1 campaign lock" >&2
  exit 65
fi
observed_profiler_amendment_sha=$(sha256sum "$profiler_amendment" | awk '{print $1}')
if [[ "$observed_profiler_amendment_sha" != "$locked_profiler_amendment_sha" ]]; then
  echo "profiler replay amendment does not match the V1 campaign lock" >&2
  exit 65
fi
observed_power_limit_w=$(nvidia-smi -i "$gpu_index" \
  --query-gpu=power.limit --format=csv,noheader,nounits)
if ! python3 - "$locked_power_limit_w" "$observed_power_limit_w" <<'PY'
import math
import sys
raise SystemExit(0 if math.isclose(float(sys.argv[1]), float(sys.argv[2]), rel_tol=0.0, abs_tol=0.01) else 1)
PY
then
  echo "GPU power limit does not match the V1 lock: expected ${locked_power_limit_w} W, observed ${observed_power_limit_w} W" >&2
  exit 65
fi

accepted_attempt_exists() {
  python3 - "$1" <<'PY'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
for path in root.glob("attempt-*/traffic.json"):
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
if [[ -n "$previous_run_id" ]] && ! accepted_attempt_exists \
  "$results_root/$result_domain/$previous_run_id"; then
  echo "previous frozen V1 run is not yet accepted: $previous_run_id" >&2
  exit 65
fi

tag=$(date -u +%Y%m%dT%H%M%SZ)
attempt_dir="$current_dir/attempt-$tag"
relative_attempt="$result_domain/$run_id/attempt-$tag"
mkdir -p "$attempt_dir"
anchor_json="$attempt_dir/anchor.json"
ncu_csv="$attempt_dir/ncu.csv"
traffic_json="$attempt_dir/traffic.json"
runner_log="$attempt_dir/runner.log"
ncu_report_prefix="/workspace/results/$relative_attempt/anchor-profile"

git -C "$repo_root" rev-parse HEAD >"$attempt_dir/repository.commit.txt"
sha256sum \
  "$campaign_lock" \
  "$execution_addendum" \
  "$profiler_amendment" \
  >"$attempt_dir/contracts.sha256.txt"
nvidia-smi -i "$gpu_index" \
  --query-gpu=index,name,memory.used,memory.free,power.draw,power.limit,temperature.gpu \
  --format=csv >"$attempt_dir/gpu.before.csv"

sudo -v
set +e
sudo docker run --rm \
  --ipc=host \
  --gpus "device=$gpu_index" \
  --cap-add SYS_ADMIN \
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
  --entrypoint ncu \
  "$image" \
  --target-processes all \
  --nvtx \
  --nvtx-include "${nvtx_range}/" \
  --replay-mode "$profiler_replay_mode" \
  --cache-control none \
  --clock-control none \
  --metrics "$metrics" \
  --page raw \
  --csv \
  --log-file "/workspace/results/$relative_attempt/ncu.csv" \
  --export "$ncu_report_prefix" \
  --force-overwrite \
  python3 /workspace/active-bytes-law/scripts/run_profiler_anchor.py \
  --campaign-lock /workspace/active-bytes-law/results/manifests/gh200-v1-anchors.lock.json \
  --run-id "$run_id" \
  --gpu-memory-utilization "$gpu_memory_utilization" \
  --output-json "/workspace/results/$relative_attempt/anchor.json" \
  >"$runner_log" 2>&1
runner_rc=$?
set -e
sudo chown -R "$(id -u):$(id -g)" "$attempt_dir"
if (( runner_rc != 0 )); then
  echo "V1 profiler runner failed (rc=$runner_rc)" >&2
  if [[ -s "$ncu_csv" ]]; then
    echo "=== Nsight Compute diagnostic ===" >&2
    tail -160 "$ncu_csv" >&2
  fi
  echo "=== application diagnostic ===" >&2
  tail -160 "$runner_log" >&2
  exit "$runner_rc"
fi

python3 "$repo_root/scripts/parse_ncu_traffic.py" \
  --anchor-json "$anchor_json" \
  --ncu-csv "$ncu_csv" \
  --output-json "$traffic_json"

nvidia-smi -i "$gpu_index" \
  --query-gpu=index,name,memory.used,memory.free,power.draw,power.limit,temperature.gpu \
  --format=csv >"$attempt_dir/gpu.after.csv"
find "$attempt_dir" -maxdepth 1 -type f ! -name 'artifacts.sha256.txt' -print0 \
  | sort -z \
  | xargs -0 sha256sum >"$attempt_dir/artifacts.sha256.txt"

echo "frozen_run_order=$run_order"
echo "frozen_run_id=$run_id"
echo "attempt_dir=$attempt_dir"
python3 -m json.tool "$traffic_json"
