# GH200 GPU1 same-SKU transfer

This campaign tests whether the Qwen2.5-7B duration-augmented Token-Energy
model frozen on GPU index 0 transfers without refitting to GPU index 1, the
second physical GH200 in the same node. It is a device-replication experiment,
not a cross-SKU, cross-node, tensor-parallel, or advance-duration-prediction
claim.

The public transfer addendum records the pre-outcome GPU1 qualification, the
source model digest, the target GPU index, the nine coordinates, and the
primary gates. The lock contains 45 randomized runs: nine cells, five repeats
per cell. Every target-device outcome must be collected from a clean checkout
at or after the commit that introduces the lock.

## Verify the frozen contracts

```bash
cd /srv/token-energy-law/repo

git pull --ff-only
git status --short
git rev-parse HEAD

(
  cd configs/addenda
  sha256sum -c \
    gh200-primary-bf16-v1.json.sha256 \
    gh200-v2-duration-holdout-v1.json.sha256 \
    gh200-gpu1-same-sku-transfer-v1.json.sha256
)

(
  cd results/manifests
  sha256sum -c gh200-gpu1-same-sku-transfer.lock.json.sha256
)
```

The checkout must be clean. GPU index 1 must be idle, in persistence mode,
using the frozen 700 W limit, and the GH200 memory hotplug mode must remain
`online_movable`. Do not run another workload on GPU 0 during this campaign;
the attempt wrapper rejects model allocations on either physical GPU before
opening each outcome.

## Open exactly one target-device outcome

Run order 0 is the execution qualification as well as the first preserved
confirmatory outcome. It is not inspected until after its alignment gate has
completed.

```bash
cd /srv/token-energy-law/repo
sudo -v
./scripts/run_gh200_gpu1_transfer_attempt.sh 0 1
```

An accepted result reports `qc_pass: true`, `runner_rc=0`, and
`alignment_rc=0`. The collector may return 2 because its whole-lifetime gap
gate includes initialization intervals; the aligned decode interval remains
the acceptance authority.

## Run the remaining randomized orders under tmux

```bash
tmux new -s tel-gpu1-transfer

cd /srv/token-energy-law/repo
sudo -v
./scripts/run_gh200_gpu1_transfer_batch.sh 1 44 1
```

Detach with `Ctrl-b`, then `d`. The batch driver refreshes the sudo ticket,
continues after preserved run-specific failures, and stores every attempt in a
run-specific directory. Re-running the same range skips accepted attempts and
retries only missing orders.

Without attaching to tmux, locate and inspect the persistent event log:

```bash
TEL_GPU1_BATCH_DIR=$(find \
  /srv/token-energy-law/results/gpu1-same-sku-transfer-batch-runs/orders-1-44 \
  -mindepth 1 -maxdepth 1 -type d -name 'batch-*' | sort | tail -1)

tail -40 "$TEL_GPU1_BATCH_DIR/batch.events.log"
```

Completion requires `batch_status=complete`. A
`complete-with-preserved-failures` status is not a scientific pass; rerun the
same range after diagnosing the listed orders.

## Evaluate without refitting

Only after all 45 alignments are accepted:

```bash
TEL_GPU1_ANALYSIS_TAG=$(date -u +%Y%m%dT%H%M%SZ)
TEL_GPU1_ANALYSIS_DIR="/srv/token-energy-law/results/gpu1-same-sku-transfer-analysis/${TEL_GPU1_ANALYSIS_TAG}"

python3 scripts/evaluate_gh200_gpu1_transfer.py \
  --campaign-lock results/manifests/gh200-gpu1-same-sku-transfer.lock.json \
  --model-artifact configs/addenda/gh200-v2-duration-holdout-v1.json \
  --required-artifact configs/addenda/gh200-gpu1-same-sku-transfer-v1.json \
  --required-lock-sha-field device_transfer_addendum_sha256 \
  --expected-host-gpu-index 1 \
  --result-domain gpu1-same-sku-transfer \
  --measurement gh200-gpu1-same-sku-zero-refit-transfer-evaluation \
  --output-prefix gpu1-transfer \
  --scientific-gate-name same_sku_device_transfer_pass \
  --results-root /srv/token-energy-law/results \
  --output-dir "$TEL_GPU1_ANALYSIS_DIR"
```

The primary decision requires all nine cell means below 10% absolute relative
error, median error at most 5%, and maximum error at most 10%. The evaluator
fails if any accepted alignment does not record host GPU index 1, if either
frozen artifact digest changes, or if any run is absent. It never refits the
four coefficients.

## Preserve the evidence

```bash
mkdir -p /srv/token-energy-law/backups
TEL_GPU1_BACKUP_TAG=$(date -u +%Y%m%dT%H%M%SZ)

tar -czf \
  "/srv/token-energy-law/backups/gh200-gpu1-transfer-${TEL_GPU1_BACKUP_TAG}.tar.gz" \
  -C /srv/token-energy-law/results \
  gpu1-same-sku-transfer \
  gpu1-same-sku-transfer-batch-runs/orders-1-44 \
  "gpu1-same-sku-transfer-analysis/${TEL_GPU1_ANALYSIS_TAG}"

sha256sum \
  "/srv/token-energy-law/backups/gh200-gpu1-transfer-${TEL_GPU1_BACKUP_TAG}.tar.gz" \
  | tee \
  "/srv/token-energy-law/backups/gh200-gpu1-transfer-${TEL_GPU1_BACKUP_TAG}.tar.gz.sha256"
```
