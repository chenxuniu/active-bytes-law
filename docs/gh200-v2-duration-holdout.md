# GH200 V2 duration holdout

This campaign is a new confirmatory test of the post-P2 model

```text
E_token = c + alpha Q_weight + beta Q_KV + P_time D/N_token.
```

The model was developed after inspecting all 75 earlier primary and evaluation
runs. Those data are therefore development data. The 45 runs in this campaign
use nine previously unopened batch/context coordinates and are the first data
eligible to confirm or reject V2. The frozen model, source hashes, gates, and
claim limits are in `configs/addenda/gh200-v2-duration-holdout-v1.json`.

## Before starting

The campaign must not inherit settings from a DVFS experiment. On the GH200
host, restore the standard state and verify the frozen files:

```bash
cd /srv/token-energy-law/repo
git pull --ff-only
git status --short
git rev-parse HEAD

sudo nvidia-smi -i 0 -rgc
sudo nvidia-smi -i 0 -rmc
sudo nvidia-smi -i 0 -pl 700
sudo nvidia-smi -pm 1

nvidia-smi -i 0 \
  --query-gpu=index,uuid,name,memory.used,memory.free,power.limit,persistence_mode,temperature.gpu,clocks.current.sm,clocks.current.memory \
  --format=csv

(
  cd configs/addenda
  sha256sum -c \
    gh200-primary-bf16-v1.json.sha256 \
    gh200-v2-duration-holdout-v1.json.sha256
)
(
  cd results/manifests
  sha256sum -c gh200-v2-duration-holdout.lock.json.sha256
)
```

Stop if the repository is dirty, GPU 0 is occupied, the power limit is not
700 W, or a digest fails. A reset-clock command reporting that no application
clock was set is harmless; any other error should be inspected.

## Run in tmux

Create and attach the session. The batch driver asks for the sudo password once
and then refreshes that ticket every 60 seconds in the same tmux pane.

```bash
tmux new -s tel-v2-holdout
cd /srv/token-energy-law/repo
./scripts/run_gh200_v2_duration_holdout_batch.sh 0 44 0
```

After entering the sudo password and seeing `batch_order=0 ... status=starting`,
detach with `Ctrl-b`, then `d`. If the key binding does not work, open a second
SSH connection and run:

```bash
tmux detach-client -s tel-v2-holdout
```

The SSH connection may then close. Each attempt has its own directory under
`/srv/token-energy-law/results/duration-v2-holdout/`. A failed attempt is
preserved and recorded, and the driver continues to the next order. Rerunning
the same `0 44` command skips every accepted order and retries only missing or
failed orders.

## Check progress

```bash
tmux capture-pane -pt tel-v2-holdout -S -60

TEL_V2_BATCH_DIR=$(find \
  /srv/token-energy-law/results/duration-v2-holdout-batch-runs/orders-0-44 \
  -mindepth 1 -maxdepth 1 -type d -name 'batch-*' \
  | sort | tail -1)

tail -40 "$TEL_V2_BATCH_DIR/batch.events.log"
```

Completion is explicit:

```text
batch_status=complete
batch_completed_orders=0-44
```

`complete-with-preserved-failures` means the queue reached order 44 but the
listed orders must be retried. Do not interpret the outcomes until all 45
orders have accepted alignments.
