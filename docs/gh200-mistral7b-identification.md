# GH200 Mistral-7B identification and sealed holdout

The non-energy qualification passed before any Mistral energy outcome was
collected. The result audits 14,504,435,712 unique weight bytes, 131,072 logical
KV bytes per attended token, BF16 cache tensors on GPU, and a 30.853-second
high-KV decode at mean attended history 16,384 and batch 16. The exact
qualification artifacts are bound into
`gh200-mistral7b-form-replication-v1.json`.

## Scientific question

Mistral-7B has approximately the Qwen2.5-7B weight scale but 2.286 times its
logical KV bytes per attended token. This campaign therefore asks whether the
same four-term Token-Energy form can be independently identified for a model
with different architecture and KV geometry on the unchanged GH200 stack.
It does not test zero-shot coefficient transfer.

## Frozen split

Identification contains 45 independent process-level runs:

- 30 coefficient-fit runs: histories 4,096, 10,240, and 16,384 crossed with
  batches 4 and 16, with five repetitions;
- 15 residual-calibration runs at the same histories and batch 8, with five
  repetitions.

The unopened holdout is already content-addressed but remains
`sealed-unreleased`: histories 6,144, 12,288, and 14,336 crossed with batches 6
and 12, with five repetitions, for 30 runs. No holdout runner or release record
is provided at this stage. Identification results cannot change those
coordinates or primary gates.

The primary outcome is gross scope-0 GPU-board joules per useful token. The
frozen model is

```text
E_token = intercept
        + alpha_weight * weight_decimal_GB_per_token
        + beta_kv * KV_read_write_decimal_GB_per_token
        + p_time_W * decode_seconds_per_useful_token.
```

The duration term is retrospective and non-causal. No DVFS treatment is part
of this campaign.

## Verify the frozen contracts

```bash
cd /srv/token-energy-law/repo
git pull --ff-only
git status --short
git rev-parse HEAD

(
  cd configs/addenda
  sha256sum -c gh200-mistral7b-form-replication-v1.json.sha256
)

(
  cd results/manifests
  sha256sum -c \
    gh200-mistral7b-identification.lock.json.sha256 \
    gh200-mistral7b-holdout.lock.json.sha256
)
```

The repository must be clean. GPU 0 must remain at the frozen 700 W power
limit, with persistence enabled and memory hotplug mode `online_movable`.

Before collecting energy outcomes, preserve the qualification evidence outside
the results tree:

```bash
mkdir -p /srv/token-energy-law/backups
TEL_M7_QUAL_BACKUP_TAG=$(date -u +%Y%m%dT%H%M%SZ)

tar -czf \
  "/srv/token-energy-law/backups/mistral7b-qualification-${TEL_M7_QUAL_BACKUP_TAG}.tar.gz" \
  -C /srv/token-energy-law/results \
  model-qualification/mistral7b-v0p3/qualification-20260904T225306Z

sha256sum \
  "/srv/token-energy-law/backups/mistral7b-qualification-${TEL_M7_QUAL_BACKUP_TAG}.tar.gz" \
  | tee \
  "/srv/token-energy-law/backups/mistral7b-qualification-${TEL_M7_QUAL_BACKUP_TAG}.tar.gz.sha256"
```

## Run one identification attempt first

```bash
./scripts/run_gh200_mistral7b_identification_attempt.sh 0 0
```

Accept it only when `runner_rc=0`, `alignment_rc=0`, and the printed alignment
has `qc_pass=true`. The global collector may report a gap outside every aligned
decode interval; the run-level decision uses the prespecified aligned interval
QC and module-counter agreement.

## Run the remaining identification orders under tmux

Start an attached session so `sudo -v` can be refreshed before detaching:

```bash
tmux new -s tel-mistral-identification

cd /srv/token-energy-law/repo
sudo -v
./scripts/run_gh200_mistral7b_identification_batch.sh 1 44 0
```

From another SSH connection, detach or monitor the session:

```bash
tmux detach-client -s tel-mistral-identification
tmux capture-pane -pt tel-mistral-identification -S -100
```

The batch stops at the first missing or rejected identification outcome,
preserves that attempt, and can be resumed with the same inclusive range after
diagnosis. Accepted attempts are detected and skipped. Every attempt has its
own directory, and every batch invocation has a persistent event log and JSON
summary.

Allow approximately four to seven hours for the 45-run identification on the
qualified node, depending on model-load cache behavior and the number of
episodes needed to exceed 30 decoded seconds per repeat.

## Freeze identification after all 45 runs

Do not inspect or execute the holdout first. After the batch reports all 45
orders accepted:

```bash
TEL_M7_FREEZE_TAG=$(date -u +%Y%m%dT%H%M%SZ)
TEL_M7_FREEZE_DIR="/srv/token-energy-law/results/mistral7b-identification-freeze/${TEL_M7_FREEZE_TAG}"

python3 scripts/freeze_gh200_mistral7b_identification.py \
  --campaign-lock results/manifests/gh200-mistral7b-identification.lock.json \
  --form-replication-addendum configs/addenda/gh200-mistral7b-form-replication-v1.json \
  --results-root /srv/token-energy-law/results \
  --output-dir "$TEL_M7_FREEZE_DIR"
```

Only a QC-passing freeze with positive familywise lower bounds for both traffic
slopes and a finite nonnegative time term becomes a holdout-release candidate.
A later commit must bind the exact coefficient, discrepancy-envelope,
accepted-run-table, and freeze-summary hashes before any of the 30 holdout runs
can execute.

## Claim boundary

Identification alone cannot establish architecture replication. A future
sealed-holdout pass may support the same functional form with separately
identified Mistral coefficients over this GH200 interpolation domain. It cannot
establish universal coefficients, zero-shot portability, cross-hardware
validity, prospective latency prediction, causal idle power, or DVFS effects.
