#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "usage: $0" >&2
  exit 64
fi

repo_root=${TEL_REPO_ROOT:-/srv/token-energy-law/repo}
results_root=${TEL_RESULTS_ROOT:-/srv/token-energy-law/results}
addendum="$repo_root/configs/addenda/gh200-tp2-nvlink-v1.json"
qualification_lock="$repo_root/results/manifests/gh200-tp2-nvlink-qualification.lock.json"
image="nvcr.io/nvidia/vllm@sha256:15f380ad9c32f0ac57ac16e4b778c6f733c88b9ffe3a936035d0a59ad17b1aab"

if [[ -n "$(git -C "$repo_root" status --short)" ]]; then
  echo "TP=2 preflight requires a clean repository checkout" >&2
  git -C "$repo_root" status --short >&2
  exit 65
fi
(
  cd "$repo_root/configs/addenda"
  sha256sum -c gh200-tp2-nvlink-v1.json.sha256
)
(
  cd "$repo_root/results/manifests"
  sha256sum -c gh200-tp2-nvlink-qualification.lock.json.sha256
)

inventory=$(nvidia-smi \
  --query-gpu=index,memory.used \
  --format=csv,noheader,nounits)
python3 - "$inventory" <<'PY'
import sys

rows = {}
for line in sys.argv[1].splitlines():
    fields = [field.strip() for field in line.split(",")]
    if len(fields) == 2:
        rows[int(fields[0])] = float(fields[1])
if set(rows) != {0, 1}:
    raise SystemExit("preflight requires exactly host GPU indexes 0 and 1")
busy = {index: used for index, used in rows.items() if used > 16.0}
if busy:
    raise SystemExit(f"both GPUs must be idle before resetting clocks: {busy}")
PY

# This is restoration to the default clock policy after any unrelated DVFS
# work.  It is not a clock treatment in this campaign.
sudo -v
sudo nvidia-smi -i 0 -rgc
sudo nvidia-smi -i 1 -rgc
sudo nvidia-smi -i 0 -rmc || true
sudo nvidia-smi -i 1 -rmc || true
sudo nvidia-smi -i 0 -pl 700
sudo nvidia-smi -i 1 -pl 700
sudo nvidia-smi -pm 1

tag=$(date -u +%Y%m%dT%H%M%SZ)
preflight_dir="$results_root/tp2-preflight/$tag"
mkdir -p "$preflight_dir"
log="$preflight_dir/preflight.log"

{
  echo "preflight_utc=$tag"
  echo "repository_commit=$(git -C "$repo_root" rev-parse HEAD)"
  echo "host_gpu_indices=0,1"
  echo "tensor_parallel_size=2"
  echo "power_limit_w_per_gpu=700"
  echo "dvfs_manipulation=false"
  echo "clock_policy=reset-default-no-lock-no-sweep"
  echo "memory_hotplug_mode=$(cat /sys/devices/system/memory/auto_online_blocks)"
  nvidia-smi -i 0,1 \
    --query-gpu=index,uuid,name,driver_version,memory.used,memory.free,power.limit,persistence_mode,compute_mode,mig.mode.current,temperature.gpu \
    --format=csv
  nvidia-smi topo -m
  sudo docker image inspect "$image" \
    --format 'image_id={{.Id}} architecture={{.Architecture}} created={{.Created}} repo_digests={{json .RepoDigests}}'
  sudo docker run --rm \
    --gpus '"device=0,1"' \
    --entrypoint python3 \
    "$image" \
    -c 'import torch; assert torch.cuda.device_count() == 2; print("visible_devices:", torch.cuda.device_count()); print("device0:", torch.cuda.get_device_name(0)); print("device1:", torch.cuda.get_device_name(1)); print("capability0:", torch.cuda.get_device_capability(0)); print("capability1:", torch.cuda.get_device_capability(1))'
} 2>&1 | tee "$log"

python3 - "$log" <<'PY'
import sys

text = open(sys.argv[1], encoding="utf-8").read()
required = [
    "host_gpu_indices=0,1",
    "tensor_parallel_size=2",
    "power_limit_w_per_gpu=700",
    "dvfs_manipulation=false",
    "memory_hotplug_mode=online_movable",
    "GPU0",
    "GPU1",
    "NV18",
    "visible_devices: 2",
]
missing = [value for value in required if value not in text]
if missing:
    raise SystemExit(f"TP=2 preflight is missing required evidence: {missing}")
PY

sha256sum "$addendum" "$qualification_lock" "$log" \
  >"$preflight_dir/artifacts.sha256"
echo "tp2_preflight_status=pass"
echo "tp2_preflight_dir=$preflight_dir"
