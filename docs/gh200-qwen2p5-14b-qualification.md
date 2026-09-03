# GH200 Qwen2.5-14B qualification

This is the first gate for the second-model replication. It is deliberately
not an energy experiment and cannot enter paper outcomes. It checks the pinned
model revision, the largest simultaneously admitted identification geometry, the exact
1024-token decode barrier, the resolved FlashAttention/BF16 runtime, resident
weight storage, and logical KV geometry. Only a passing report authorizes us to
freeze the 14B identification and unopened holdout campaigns.

Qualification v1 requested batch 20 at a mean attended history of 16,384.
The model and backend loaded, but vLLM reported only 16.08x maximum concurrency
at the frozen maximum length; the full batch therefore could not cross a common
bootstrap barrier. No energy outcome was measured. The content-addressed v2
contract changes only the batch from 20 to 16 and preserves that failure as its
design basis.

## Restore and inspect the node

The node may have been used for a DVFS experiment. Restore the standard clock
and power state before qualification, then inspect it:

```bash
sudo nvidia-smi -i 0 -rgc
sudo nvidia-smi -i 0 -rmc
sudo nvidia-smi -i 0 -pl 700
sudo nvidia-smi -pm 1

nvidia-smi -i 0 \
  --query-gpu=index,uuid,name,driver_version,memory.used,memory.free,power.limit,persistence_mode,temperature.gpu,clocks.current.sm,clocks.current.memory \
  --format=csv
```

Do not continue if GPU 0 is occupied, the power limit is not 700 W, or the
memory hotplug mode is not `online_movable`.

## Run exactly one qualification

```bash
cd /srv/token-energy-law/repo
git pull --ff-only
git status --short
git rev-parse HEAD

./scripts/run_gh200_qwen14b_qualification.sh 0
```

The first run may download about 30 GB of model data. The script loads the
model twice on purpose: the first load exercises the exact decode barrier; the
second independently inventories the resolved weight and KV-cache storage.
Both stages preserve their JSON and logs under
`/srv/token-energy-law/results/model-qualification/qwen2p5-14b/`.

Success ends with:

```text
qualification_status=pass
qualification_dir=...
```

On failure, do not reduce batch, context, token count, or memory utilization.
Return the final 120 lines of the failed stage log and the path printed by the
script. Any changed geometry requires review and a new content-addressed
qualification contract.

## Inspect and back up a pass

```bash
TEL_Q14_DIR=$(find \
  /srv/token-energy-law/results/model-qualification/qwen2p5-14b \
  -mindepth 1 -maxdepth 1 -type d -name 'qualification-*' \
  | sort | tail -1)

python3 -m json.tool "$TEL_Q14_DIR/qualification-summary.json"
sha256sum -c "$TEL_Q14_DIR/artifacts.sha256"

mkdir -p /srv/token-energy-law/backups
TEL_Q14_TAG=$(date -u +%Y%m%dT%H%M%SZ)
tar -czf \
  "/srv/token-energy-law/backups/qwen14b-qualification-${TEL_Q14_TAG}.tar.gz" \
  -C /srv/token-energy-law/results \
  "${TEL_Q14_DIR#/srv/token-energy-law/results/}"
sha256sum \
  "/srv/token-energy-law/backups/qwen14b-qualification-${TEL_Q14_TAG}.tar.gz" \
  | tee \
  "/srv/token-energy-law/backups/qwen14b-qualification-${TEL_Q14_TAG}.tar.gz.sha256"
```
