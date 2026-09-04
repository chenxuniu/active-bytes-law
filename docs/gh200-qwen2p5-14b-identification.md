# GH200 Qwen2.5-14B identification

This campaign is the first paper-eligible second-model experiment. It asks a
narrow question: after measuring new coefficients for Qwen2.5-14B on the same
GH200 runtime stratum, does the duration-augmented Token-Energy form remain
predictive at coordinates that were not used for fitting? It does not test
coefficient equality across models or hardware generality.

The identification lock contains 45 independent process-level runs: 30 runs
over six coefficient-fit cells and 15 runs over three residual-calibration
cells. A separate 30-run holdout lock is already content-addressed, but remains
`sealed-unreleased`. The source lock remains sealed; the later release record
is a separate, content-addressed authorization rather than a mutation of that
lock.

## Frozen analysis

The run-level fit is

```text
E_token = intercept
        + alpha_weight * weight_decimal_GB_per_token
        + beta_kv * kv_read_write_decimal_GB_per_token
        + p_time_W * decode_seconds_per_useful_token.
```

`p_time_W` is a retrospective nuisance coefficient. It absorbs duration-linked
board energy in the frozen runtime and is not interpreted as literal idle
power, a DVFS effect, or an advance latency predictor. The fit uses only the 30
coefficient-fit runs, unweighted OLS, and HC3 covariance. The 15 batch-8 runs
construct a separately reported Bonferroni Student-t residual envelope.

## Restore and verify the node

```bash
sudo nvidia-smi -i 0 -rgc
sudo nvidia-smi -i 0 -rmc || true
sudo nvidia-smi -i 0 -pl 700
sudo nvidia-smi -pm 1

cd /srv/token-energy-law/repo
git pull --ff-only
git status --short
git rev-parse HEAD

(
  cd configs/addenda
  sha256sum -c gh200-qwen2p5-14b-form-replication-v1.json.sha256
)
(
  cd results/manifests
  sha256sum -c gh200-qwen2p5-14b-identification.lock.json.sha256
  sha256sum -c gh200-qwen2p5-14b-holdout.lock.json.sha256
)

nvidia-smi -i 0 \
  --query-gpu=index,name,driver_version,memory.used,memory.free,power.limit,persistence_mode,temperature.gpu \
  --format=csv
cat /sys/devices/system/memory/auto_online_blocks
```

Continue only with a clean checkout, GPU 0 free, a 700 W power limit,
persistence enabled, and `online_movable`. The attempt runner independently
checks the checksums and the exact successful qualification artifacts.

## Validate order 0 before background execution

Run one frozen outcome through the complete measurement path:

```bash
cd /srv/token-energy-law/repo
./scripts/run_gh200_qwen14b_identification_attempt.sh 0 0
```

Accept it only if the final alignment reports `qc_pass: true`, the run ID and
order match the lock, and both runner and alignment complete. A collector exit
code of 2 can reflect a gap outside every aligned decode interval; the final
alignment, not the whole-lifetime collector status, is the run acceptance
record.

## Run orders 1--44 under tmux

After order 0 is accepted:

```bash
tmux new -s tel-q14-identification
cd /srv/token-energy-law/repo
sudo -v
./scripts/run_gh200_qwen14b_identification_batch.sh 1 44 0
```

Detach with `Ctrl-b`, then `d`. If the key sequence is unavailable, open a
second SSH session and run:

```bash
tmux detach-client -s tel-q14-identification
```

The driver stops at the first unsuccessful scientific run, preserves all
artifacts, and retries that same order when the inclusive command is run again.
Accepted attempts are never overwritten. Inspect progress with:

```bash
tmux capture-pane -pt tel-q14-identification -S -80
find /srv/token-energy-law/results/qwen14b-identification-batch-runs \
  -name batch.events.log -print | sort | tail -1
```

The 45-run identification is expected to take about 7--9 hours on this node;
reserve a 10--12 hour window for model loads, cooling, and one recoverable
retry. The sealed 30-run holdout is a later 5--6 hour stage and must not be
opened merely because the identification batch completes.

## Freeze the identification artifacts

After all 45 locked orders have exactly one accepted alignment:

```bash
TEL_Q14_FREEZE_TAG=$(date -u +%Y%m%dT%H%M%SZ)
TEL_Q14_FREEZE_DIR="/srv/token-energy-law/results/qwen14b-identification-freeze/${TEL_Q14_FREEZE_TAG}"

python3 scripts/freeze_gh200_qwen14b_identification.py \
  --campaign-lock results/manifests/gh200-qwen2p5-14b-identification.lock.json \
  --form-replication-addendum configs/addenda/gh200-qwen2p5-14b-form-replication-v1.json \
  --results-root /srv/token-energy-law/results \
  --output-dir "$TEL_Q14_FREEZE_DIR"
```

This emits a coefficient artifact, residual envelope, accepted-run table, and
freeze summary. A technical pass is not itself a scientific pass: holdout
release additionally requires positive lower bounds for both traffic slopes
and a finite nonnegative duration coefficient. Even then, a new release record
must bind the exact artifact hashes before any holdout command is created.

## Back up the raw identification evidence

```bash
mkdir -p /srv/token-energy-law/backups
TEL_Q14_BACKUP_TAG=$(date -u +%Y%m%dT%H%M%SZ)

tar -czf \
  "/srv/token-energy-law/backups/qwen14b-identification-${TEL_Q14_BACKUP_TAG}.tar.gz" \
  -C /srv/token-energy-law/results \
  qwen14b-identification \
  qwen14b-identification-batch-runs \
  qwen14b-identification-freeze

sha256sum \
  "/srv/token-energy-law/backups/qwen14b-identification-${TEL_Q14_BACKUP_TAG}.tar.gz" \
  | tee \
  "/srv/token-energy-law/backups/qwen14b-identification-${TEL_Q14_BACKUP_TAG}.tar.gz.sha256"
```

Raw telemetry and logs remain outside Git. Only reviewed, sanitized aggregate
artifacts should later enter the public repository.

## Released holdout

Identification completed with all 45 runs accepted. The two traffic-slope
lower bounds were positive, the retrospective duration coefficient was finite
and nonnegative, and the freeze summary marked the holdout as a release
candidate. Release record
`configs/addenda/gh200-qwen2p5-14b-holdout-release-v1.json` binds the exact
coefficient, residual-envelope, accepted-run, and freeze-summary hashes.

Before measuring a holdout outcome, verify that release against the node-local
freeze directory:

```bash
TEL_Q14_FREEZE_DIR=/srv/token-energy-law/results/qwen14b-identification-freeze/20260904T161637Z
TEL_Q14_VERIFY_DIR=/srv/token-energy-law/results/qwen14b-holdout-release-verifications/manual
mkdir -p "$TEL_Q14_VERIFY_DIR"

python3 scripts/verify_gh200_qwen14b_holdout_release.py \
  --release-record configs/addenda/gh200-qwen2p5-14b-holdout-release-v1.json \
  --identification-freeze-dir "$TEL_Q14_FREEZE_DIR" \
  --holdout-lock results/manifests/gh200-qwen2p5-14b-holdout.lock.json \
  --form-replication-addendum configs/addenda/gh200-qwen2p5-14b-form-replication-v1.json \
  --output-json "$TEL_Q14_VERIFY_DIR/release-verification.json"
```

Validate order 0 manually:

```bash
./scripts/run_gh200_qwen14b_holdout_attempt.sh 0 0
```

Only after its final alignment passes, execute orders 1--29 under tmux:

```bash
tmux new -s tel-q14-holdout
cd /srv/token-energy-law/repo
sudo -v
./scripts/run_gh200_qwen14b_holdout_batch.sh 1 29 0
```

The batch preserves failed attempts, continues to later frozen orders, and
returns nonzero if any order remains unaccepted. Rerunning the same inclusive
range skips accepted attempts and retries the missing orders.

After all 30 holdout runs are accepted, evaluate without refitting:

```bash
TEL_Q14_EVAL_TAG=$(date -u +%Y%m%dT%H%M%SZ)
TEL_Q14_EVAL_DIR="/srv/token-energy-law/results/qwen14b-holdout-analysis/${TEL_Q14_EVAL_TAG}"

python3 scripts/evaluate_gh200_qwen14b_holdout.py \
  --campaign-lock results/manifests/gh200-qwen2p5-14b-holdout.lock.json \
  --release-record configs/addenda/gh200-qwen2p5-14b-holdout-release-v1.json \
  --identification-freeze-dir "$TEL_Q14_FREEZE_DIR" \
  --form-replication-addendum configs/addenda/gh200-qwen2p5-14b-form-replication-v1.json \
  --results-root /srv/token-energy-law/results \
  --output-dir "$TEL_Q14_EVAL_DIR"
```

The primary result requires all six cell means to have at most 10% absolute
relative error and the median cell error to be at most 5%. Frozen residual-band
coverage is reported descriptively because the parent addendum did not make it
a primary gate. No outcome from this campaign authorizes universal
coefficients, cross-hardware transfer, DVFS conclusions, or a causal
interpretation of the duration term.
