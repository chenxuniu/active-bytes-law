#!/usr/bin/env bash
set -uo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [GPU_INDEX]" >&2
  exit 64
fi

gpu_index=${1:-0}
if [[ "$gpu_index" != "0" && "$gpu_index" != "1" ]]; then
  echo "GPU_INDEX must be 0 or 1" >&2
  exit 64
fi

repo_root=${TEL_REPO_ROOT:-/srv/token-energy-law/repo}
results_root=${TEL_RESULTS_ROOT:-/srv/token-energy-law/results}
hf_cache=${TEL_HF_CACHE:-/srv/token-energy-law/hf-cache}
container_home=${TEL_CONTAINER_HOME:-/srv/token-energy-law/container-home}
campaign_lock=${TEL_QUALIFICATION_CAMPAIGN_LOCK:-"$repo_root/results/manifests/gh200-qwen2p5-14b-qualification-v2.lock.json"}
qualification_contract=${TEL_QUALIFICATION_CONTRACT:-"$repo_root/configs/addenda/gh200-qwen2p5-14b-qualification-v2.json"}
qualification_result_subdir=${TEL_QUALIFICATION_RESULT_SUBDIR:-qwen2p5-14b}
qualification_id=${TEL_QUALIFICATION_ID:-gh200-qwen2p5-14b-form-replication-qualification-v2}
qualification_name_prefix=${TEL_QUALIFICATION_NAME_PREFIX:-tel-q14}

case "$campaign_lock" in
  "$repo_root"/*) campaign_lock_relative=${campaign_lock#"$repo_root/"} ;;
  *)
    echo "qualification campaign lock must be below the repository root" >&2
    exit 65
    ;;
esac

if [[ -n "$(git -C "$repo_root" status --short)" ]]; then
  echo "qualification requires a clean repository checkout" >&2
  git -C "$repo_root" status --short >&2
  exit 65
fi

(
  cd "$repo_root/configs/addenda" || exit 66
  sha256sum -c gh200-qwen2p5-14b-qualification-v2.json.sha256
) || exit $?
(
  cd "$repo_root/results/manifests" || exit 66
  sha256sum -c gh200-qwen2p5-14b-qualification-v2.lock.json.sha256
) || exit $?

run_contract=$(python3 - "$campaign_lock" <<'PY'
import json
import sys

lock = json.load(open(sys.argv[1], encoding="utf-8"))
if len(lock["run_order"]) != 1:
    raise SystemExit("qualification lock must contain exactly one run")
run = lock["run_order"][0]
p = run["parameters"]
print("\t".join([
    run["run_id"],
    p["model"],
    p["model_revision"],
    str(p["target_mean_attended_history_tokens"]),
    str(p["target_batch"]),
    str(p["metered_decode_tokens_per_request"]),
    str(p["gpu_memory_utilization"]),
    p["attention_backend"],
    p["container_image"],
    p["driver_version"],
    str(p["power_limit_w"]),
    p["execution_addendum_sha256"],
]))
PY
) || exit $?

IFS=$'\t' read -r \
  run_id model model_revision target_mean target_batch measured_tokens \
  gpu_memory_utilization attention_backend image driver_version power_limit_w \
  expected_contract_sha <<<"$run_contract"

observed_contract_sha=$(sha256sum "$qualification_contract" | awk '{print $1}')
if [[ "$observed_contract_sha" != "$expected_contract_sha" ]]; then
  echo "qualification contract does not match the frozen lock" >&2
  exit 65
fi

hotplug_mode=$(cat /sys/devices/system/memory/auto_online_blocks)
if [[ "$hotplug_mode" != "online_movable" ]]; then
  echo "memory hotplug mode must be online_movable, got $hotplug_mode" >&2
  exit 65
fi

gpu_state=$(nvidia-smi -i "$gpu_index" \
  --query-gpu=name,driver_version,memory.used,power.limit,persistence_mode \
  --format=csv,noheader,nounits) || exit $?
python3 - "$gpu_state" "$driver_version" "$power_limit_w" <<'PY' || exit $?
import math
import sys

fields = [field.strip() for field in sys.argv[1].split(",")]
if len(fields) != 5:
    raise SystemExit("unexpected nvidia-smi preflight output")
name, driver, memory_used, power_limit, persistence = fields
reasons = []
if "GH200 144G HBM3e" not in name:
    reasons.append(f"unexpected GPU: {name}")
if driver != sys.argv[2]:
    reasons.append(f"driver {driver} != {sys.argv[2]}")
if float(memory_used) > 1024:
    reasons.append(f"GPU is occupied: {memory_used} MiB in use")
if not math.isclose(float(power_limit), float(sys.argv[3]), rel_tol=0.0, abs_tol=0.01):
    reasons.append(f"power limit {power_limit} W != {sys.argv[3]} W")
if persistence != "Enabled":
    reasons.append(f"persistence mode is {persistence}")
if reasons:
    raise SystemExit("; ".join(reasons))
PY

expected_image_id=${image#*@}
observed_image_id=$(sudo docker image inspect "$image" --format '{{.Id}}') || exit $?
observed_architecture=$(sudo docker image inspect "$image" --format '{{.Architecture}}') || exit $?
if [[ "$observed_image_id" != "$expected_image_id" || "$observed_architecture" != "arm64" ]]; then
  echo "container identity or architecture does not match the frozen contract" >&2
  exit 65
fi

tag=$(date -u +%Y%m%dT%H%M%SZ)
qualification_dir="$results_root/model-qualification/$qualification_result_subdir/qualification-$tag"
relative_dir="model-qualification/$qualification_result_subdir/qualification-$tag"
mkdir -p "$qualification_dir" "$hf_cache" "$container_home"

doctor_json="$qualification_dir/batch-doctor.json"
doctor_log="$qualification_dir/batch-doctor.log"
runtime_json="$qualification_dir/runtime-audit.json"
runtime_log="$qualification_dir/runtime-audit.log"
summary_json="$qualification_dir/qualification-summary.json"
doctor_name="$qualification_name_prefix-doctor-$tag"
runtime_name="$qualification_name_prefix-runtime-$tag"

cleanup() {
  sudo docker rm -f "$doctor_name" "$runtime_name" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

git -C "$repo_root" rev-parse HEAD >"$qualification_dir/repository.commit.txt"
sha256sum "$qualification_contract" "$campaign_lock" \
  >"$qualification_dir/contracts.sha256.txt"
nvidia-smi -i "$gpu_index" \
  --query-gpu=index,uuid,name,driver_version,memory.used,memory.free,power.draw,power.limit,persistence_mode,temperature.gpu \
  --format=csv >"$qualification_dir/gpu.before.csv"

sudo -v || exit $?

echo "stage=batch-doctor status=starting"
sudo docker run --rm \
  --name "$doctor_name" \
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
  /workspace/active-bytes-law/scripts/run_batch_doctor.py \
  --model "$model" \
  --model-revision "$model_revision" \
  --target-mean-attended-history-tokens "$target_mean" \
  --batch "$target_batch" \
  --measured-decode-tokens "$measured_tokens" \
  --gpu-memory-utilization "$gpu_memory_utilization" \
  --seed 2027 \
  --output-json "/workspace/results/$relative_dir/batch-doctor.json" \
  >"$doctor_log" 2>&1
doctor_rc=$?
sudo chown -R "$(id -u):$(id -g)" "$qualification_dir"
if [[ $doctor_rc -ne 0 || ! -s "$doctor_json" ]]; then
  echo "stage=batch-doctor status=failed rc=$doctor_rc" | tee "$qualification_dir/status.txt" >&2
  tail -120 "$doctor_log" >&2
  exit 2
fi
echo "stage=batch-doctor status=accepted"

echo "stage=runtime-audit status=starting"
sudo docker run --rm \
  --name "$runtime_name" \
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
  /workspace/active-bytes-law/scripts/run_runtime_audit.py \
  --campaign-lock "/workspace/active-bytes-law/$campaign_lock_relative" \
  --run-id "$run_id" \
  --gpu-memory-utilization "$gpu_memory_utilization" \
  --output-json "/workspace/results/$relative_dir/runtime-audit.json" \
  >"$runtime_log" 2>&1
runtime_rc=$?
sudo chown -R "$(id -u):$(id -g)" "$qualification_dir"
if [[ $runtime_rc -ne 0 || ! -s "$runtime_json" ]]; then
  echo "stage=runtime-audit status=failed rc=$runtime_rc" | tee "$qualification_dir/status.txt" >&2
  tail -120 "$runtime_log" >&2
  exit 2
fi
echo "stage=runtime-audit status=accepted"

python3 "$repo_root/scripts/evaluate_model_qualification.py" \
  --qualification-contract "$qualification_contract" \
  --campaign-lock "$campaign_lock" \
  --doctor-json "$doctor_json" \
  --runtime-audit-json "$runtime_json" \
  --output-json "$summary_json"
qualification_rc=$?

nvidia-smi -i "$gpu_index" \
  --query-gpu=index,uuid,name,driver_version,memory.used,memory.free,power.draw,power.limit,persistence_mode,temperature.gpu \
  --format=csv >"$qualification_dir/gpu.after.csv"

{
  echo "qualification_id=$qualification_id"
  echo "run_id=$run_id"
  echo "gpu_index=$gpu_index"
  echo "doctor_rc=$doctor_rc"
  echo "runtime_rc=$runtime_rc"
  echo "qualification_rc=$qualification_rc"
  echo "qualification_dir=$qualification_dir"
} >"$qualification_dir/status.txt"

find "$qualification_dir" -maxdepth 1 -type f ! -name artifacts.sha256 -print0 \
  | sort -z \
  | xargs -0 sha256sum >"$qualification_dir/artifacts.sha256"

if [[ $qualification_rc -ne 0 ]]; then
  echo "qualification failed; artifacts were preserved at $qualification_dir" >&2
  exit 2
fi

echo "qualification_status=pass"
echo "qualification_dir=$qualification_dir"
