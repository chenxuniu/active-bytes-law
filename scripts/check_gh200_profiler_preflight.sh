#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [GPU_INDEX]" >&2
  exit 64
fi

gpu_index=${1:-0}
repo_root=${TEL_REPO_ROOT:-/srv/token-energy-law/repo}
results_root=${TEL_RESULTS_ROOT:-/srv/token-energy-law/results}
image="nvcr.io/nvidia/vllm@sha256:15f380ad9c32f0ac57ac16e4b778c6f733c88b9ffe3a936035d0a59ad17b1aab"
expected_power_limit_w=700
addendum="$repo_root/configs/addenda/gh200-primary-bf16-v1.json"
identification_lock="$repo_root/results/manifests/gh200-primary-bf16.lock.json"
evaluation_lock="$repo_root/results/manifests/gh200-primary-bf16-evaluation.lock.json"

if [[ -n "$(git -C "$repo_root" status --short)" ]]; then
  echo "profiler preflight requires a clean repository checkout" >&2
  git -C "$repo_root" status --short >&2
  exit 65
fi

(
  cd "$repo_root/configs/addenda"
  sha256sum -c gh200-primary-bf16-v1.json.sha256
)
(
  cd "$repo_root/results/manifests"
  sha256sum -c \
    gh200-primary-bf16.lock.json.sha256 \
    gh200-primary-bf16-evaluation.lock.json.sha256
)

for path in "$addendum" "$identification_lock" "$evaluation_lock"; do
  if [[ ! -r "$path" ]]; then
    echo "missing frozen primary artifact: $path" >&2
    exit 66
  fi
done

tag=$(date -u +%Y%m%dT%H%M%SZ)
output_dir="$results_root/profiler-preflight/$tag"
mkdir -p "$output_dir"
output_log="$output_dir/preflight.log"

sudo -v
{
  echo "preflight_utc=$tag"
  echo "repository_commit=$(git -C "$repo_root" rev-parse HEAD)"
  echo "gpu_index=$gpu_index"
  echo "image=$image"
  echo "memory_hotplug_mode=$(cat /sys/devices/system/memory/auto_online_blocks)"
  nvidia-smi -i "$gpu_index" \
    --query-gpu=index,uuid,name,driver_version,memory.total,memory.free,power.limit,persistence_mode,temperature.gpu \
    --format=csv
  actual_power_limit_w=$(nvidia-smi -i "$gpu_index" \
    --query-gpu=power.limit --format=csv,noheader,nounits)
  python3 - "$expected_power_limit_w" "$actual_power_limit_w" <<'PY'
import math
import sys
expected = float(sys.argv[1])
actual = float(sys.argv[2])
if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=0.01):
    raise SystemExit(
        f"power-limit contract failed: expected {expected:.2f} W, observed {actual:.2f} W"
    )
print(f"power_limit_contract=pass expected_w={expected:.2f} observed_w={actual:.2f}")
PY
  sudo docker image inspect "$image" \
    --format 'image_id={{.Id}} architecture={{.Architecture}} created={{.Created}} repo_digests={{json .RepoDigests}}'
  if [[ -r /proc/driver/nvidia/params ]]; then
    grep -E 'RmProfilingAdminOnly|RestrictProfiling' /proc/driver/nvidia/params || true
  fi
  sudo docker run --rm \
    --gpus "device=$gpu_index" \
    --cap-add SYS_ADMIN \
    --entrypoint /bin/bash \
    "$image" \
    -lc '
      set -euo pipefail
      uname -m
      command -v ncu
      ncu --version
      metric_inventory=$(ncu --query-metrics-mode suffix \
        --metrics dram__bytes_read,dram__bytes_write)
      metric_names=$(printf "%s\n" "$metric_inventory" \
        | awk "\$1 ~ /^dram__bytes_(read|write)\./ {print \$1}")
      grep -Fx "dram__bytes_read.sum" <<<"$metric_names"
      grep -Fx "dram__bytes_write.sum" <<<"$metric_names"
      echo "dram_metric_contract=pass"
      ncu \
        --target-processes all \
        --launch-count 1 \
        --cache-control none \
        --metrics dram__bytes_read.sum,dram__bytes_write.sum \
        --csv \
        python3 -c "import torch; x=torch.ones(1048576,device=\"cuda\"); y=x+1; torch.cuda.synchronize(); print(float(y[0]))"
      echo "profiling_permission_smoke=pass"
      python3 - <<"PY"
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
PY
    '
} 2>&1 | tee "$output_log"

sha256sum "$output_log" >"$output_log.sha256"
echo "profiler_preflight_log=$output_log"
echo "profiler_preflight_sha256=$(awk '{print $1}' "$output_log.sha256")"
