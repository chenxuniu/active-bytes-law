# FP8 KV calibration environment

This image is a derived, calibration-only environment. It is never used to
collect paper energy or latency outcomes. The base inference image remains the
immutable NGC digest frozen in the campaign.

The dependency set is the successful resolver plan paired with
`compressed-tensors==0.10.2` already present in the base. In particular, LLM
Compressor 0.6.0.1 requires Transformers at or below 4.52.4, so the calibration
image deliberately uses 4.52.4 while the frozen inference image continues to
use 4.55.2. The calibration image removes vLLM entirely so that the incompatible
inference dependency cannot be invoked accidentally. It also removes the base
image's partial aarch64 Apex build: that package advertises Apex availability
but does not contain `apex.amp`, causing Transformers to fail during optional
AMP discovery. Calibration uses native PyTorch and does not require Apex.
`pip check` remains a hard build gate. Every produced checkpoint must be loaded
and audited again in the frozen inference image before it can enter a new
campaign lock.

Build without changing the base image:

```bash
docker build \
  --file calibration/Dockerfile \
  --tag token-energy-law-calibration:0.3 \
  .
```

Then capture the image ID, package inventory, and API surface before opening a
model or dataset. The repository must be mounted at
`/workspace/active-bytes-law`:

```bash
docker image inspect token-energy-law-calibration:0.3 \
  --format 'image_id={{.Id}} architecture={{.Architecture}} created={{.Created}}'

docker run --rm \
  -v "$PWD:/workspace/active-bytes-law:ro" \
  --entrypoint python3 \
  token-energy-law-calibration:0.3 \
  /workspace/active-bytes-law/scripts/inspect_calibration_stack.py
```

This is a smoke test only. Dataset download and calibration remain closed until
the reported API is reviewed.

After that audit passes, run the calibration doctor before producing a full
checkpoint. The doctor pins both the model and dataset revisions, deterministically
selects eight UltraChat records, calibrates tensor-wise FP8 K/V scales at a short
sequence length, and does not save a checkpoint. It fails unless all expected
attention layers expose finite positive K/V scales, the scales are not all 1.0,
and a probe of the original BF16 parameters remains unchanged. Its output is
explicitly non-paper data and no energy collector may run beside it.

```bash
docker run --rm \
  --ipc=host \
  --gpus '"device=0"' \
  -e HOME=/workspace/home \
  -e HF_HOME=/workspace/hf-cache \
  -v "$PWD:/workspace/active-bytes-law:ro" \
  -v /path/to/results:/workspace/results \
  -v /path/to/hf-cache:/workspace/hf-cache \
  -v /path/to/container-home:/workspace/home \
  --entrypoint python3 \
  token-energy-law-calibration:0.3 \
  /workspace/active-bytes-law/scripts/run_kv_calibration_doctor.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --model-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --dataset-revision 8049631c405ae6576f93f445c6b8166f76f5505a \
  --num-calibration-samples 8 \
  --max-sequence-length 256 \
  --seed 2027 \
  --output-json /workspace/results/calibration-doctor.json
```

Do not promote the short doctor to the full calibration simply by changing its
arguments. The full 512-sample, 2048-token checkpoint requires a separately
recorded execution contract and a post-save load audit in the frozen inference
image.

Run the short doctor twice under the same contract before opening the full
calibration. Compare the two complete JSON artifacts, including the rendered
sample digest, package contract, baseline-parameter probe, layer membership,
and all 56 K/V scale values:

```bash
python3 scripts/compare_kv_calibration_doctors.py \
  --first-json /path/to/doctor-r01.json \
  --second-json /path/to/doctor-r02.json \
  --output-json /path/to/doctor-repeat-comparison.json
```

The comparison uses a relative scale tolerance of `1e-6` and an absolute
tolerance of `1e-8`. A passing comparison is still non-paper evidence; it only
qualifies the calibration pipeline for a separately locked full run.
