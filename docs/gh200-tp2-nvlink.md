# Two-GH200 TP=2/NVLink experiment

This protocol tests whether the duration-augmented Token-Energy functional form
can be re-identified in a distinct tensor-parallel execution stratum. It uses
Qwen2.5-7B on both GH200s in the qualified node, vLLM V0 tensor parallelism,
and the node's `NV18` inter-GPU path.

This is not a DVFS experiment. Both devices remain at the fixed 700 W power
limit. The preflight resets application clocks to the driver default only to
remove state left by unrelated experiments; it does not set, lock, or sweep a
clock. No DVFS result may be inferred from this campaign.

## Scientific question and estimand

The TP=2 law is fitted as its own stratum:

```text
E_pair/token = intercept_tp2
             + alpha_tp2 * full_model_weight_decimal_GB_per_token
             + beta_tp2  * full_model_KV_read_write_decimal_GB_per_token
             + p_time_tp2_W * decode_seconds_per_useful_token.
```

`E_pair/token` is the sum of both GH200 GPU-board, NVML scope-0,
instantaneous-power integrals divided by the common useful-token count. Each
integral uses the same runner `GO` and `DONE` monotonic boundaries. Device 0
and device 1 must each pass sampling-gap and module-counter agreement checks;
passing only after cancellation in the pair sum is forbidden.

The Active-Bytes coordinates use the audited full-model storage and logical
full-model KV obligation. They are not multiplied by two and are not replaced
by one worker shard. The sum of physical board energy is the response; the
full algorithmic byte obligation is the explanatory coordinate.

TP=2/NCCL/NVLink can alter the TP2-specific coefficients, duration, intercept,
or residuals. This fixed-TP design cannot identify a separate communication
energy coefficient, so none is introduced after outcomes are observed. The
single-GPU coefficients are descriptive comparison material, not TP=2 primary
predictions.

## Frozen stages

The checksummed addendum is
`configs/addenda/gh200-tp2-nvlink-v1.json`. Three independent campaign locks
enforce the following order:

1. **Qualification (non-paper):** three feasibility/meter-boundary runs at
   `(L, B) = (4096, 8), (10240, 16), (16384, 32)`. Their energy values never
   enter a fit, a gate, or a paper outcome.
2. **Identification:** 30 coefficient-fit runs over six cells and 15 residual-
   calibration runs over three disjoint cells. Every accepted paper run has at
   least 30 total decode seconds.
3. **Sealed holdout:** 30 runs over six previously unused `(L, B)` cells. This
   lock remains `sealed-unreleased` until a later content-addressed record
   binds the exact coefficient artifact, residual envelope, accepted-run
   table, identification summary, and backup digest.

The batch drivers preserve every failed attempt, continue to later frozen
orders, and return a nonzero status if failures remain. Re-running the same
inclusive range skips exactly one accepted alignment and retries missing
orders. Every attempt has its own timestamped directory; no accepted artifact
is overwritten.

## Tomorrow: first safe command

Do not start while another experiment owns either GPU. First update the node
checkout and inspect both devices:

```bash
cd /srv/token-energy-law/repo
git pull --ff-only
git status --short
git rev-parse HEAD

nvidia-smi \
  --query-gpu=index,memory.used,memory.free,power.draw,power.limit,persistence_mode,temperature.gpu \
  --format=csv
```

Stop if the repository is dirty, either GPU has more than 16 MiB allocated, a
power limit is not 700 W, or an unrelated process is still running. When both
GPUs are free, run the dedicated preflight:

```bash
./scripts/check_gh200_tp2_preflight.sh
```

It verifies the addendum and qualification lock, requires
`online_movable`, restores default application-clock policy, fixes both power
limits at 700 W, records topology, and proves a container sees exactly two
GH200s. A pass is an admission check, not an energy result.

## Qualification

Run one order manually before committing the node to a batch:

```bash
./scripts/run_gh200_tp2_qualification_attempt.sh 0
```

The accepted summary must report all of the following:

- `runner_rc=0` and `alignment_rc=0`;
- `qc_pass=true`;
- `tensor_parallel_size=2` and `host_gpu_indices=[0,1]`;
- device-level counter QC for both devices;
- a primary `gpu_joules_per_token` that is explicitly the two-board sum.

If order 0 passes, run the remaining two under tmux:

```bash
tmux new -s tel-tp2-qualification
cd /srv/token-energy-law/repo
sudo -v
./scripts/run_gh200_tp2_qualification_batch.sh 1 2
```

Detach from another SSH session if desired:

```bash
tmux detach-client -s tel-tp2-qualification
```

Monitor without attaching:

```bash
find /srv/token-energy-law/results/tp2-qualification-batch-runs \
  -name batch.events.log -type f -print | sort | tail -1
tail -30 /path/printed/above
```

Qualification must contain exactly one accepted alignment for each of its
three run IDs. The identification runner fails closed otherwise.

## Identification

After all three qualification orders pass, validate order 0 manually:

```bash
./scripts/run_gh200_tp2_identification_attempt.sh 0
```

Then launch the remaining frozen orders under tmux:

```bash
tmux new -s tel-tp2-identification
cd /srv/token-energy-law/repo
sudo -v
./scripts/run_gh200_tp2_identification_batch.sh 1 44
```

On this node, allow roughly 4--8 hours for 45 runs, but treat that as an
operational estimate rather than a protocol guarantee. Model startup and the
number of decode episodes determine the actual duration.

When the batch reports `batch_status=complete`, freeze the identification:

```bash
TEL_TP2_FREEZE_TAG=$(date -u +%Y%m%dT%H%M%SZ)
TEL_TP2_FREEZE_DIR="/srv/token-energy-law/results/tp2-identification-freeze/${TEL_TP2_FREEZE_TAG}"

python3 scripts/freeze_gh200_tp2_identification.py \
  --campaign-lock results/manifests/gh200-tp2-nvlink-identification.lock.json \
  --qualification-lock results/manifests/gh200-tp2-nvlink-qualification.lock.json \
  --execution-addendum configs/addenda/gh200-tp2-nvlink-v1.json \
  --results-root /srv/token-energy-law/results \
  --output-dir "$TEL_TP2_FREEZE_DIR"
```

Stop here and preserve the freeze directory. Do not run the holdout. A new
release record must be created and reviewed after the identification artifacts
exist and before any holdout outcome is observed.

## Released holdout (future, intentionally unavailable now)

The runner expects a later checksummed
`configs/addenda/gh200-tp2-nvlink-holdout-release-v1.json`. Without that exact
record, its reviewed digest compiled into the verifier, and
`TEL_TP2_IDENTIFICATION_FREEZE_DIR`, it exits before starting a GPU
container. The release verifier also rejects changed gates, coefficients,
residual ranges, locks, or artifact hashes.

After a reviewed release exists, validate order 0, then batch 1--29:

```bash
export TEL_TP2_IDENTIFICATION_FREEZE_DIR=/srv/token-energy-law/results/tp2-identification-freeze/EXACT_TAG
./scripts/run_gh200_tp2_holdout_attempt.sh 0
./scripts/run_gh200_tp2_holdout_batch.sh 1 29
```

Evaluate all 30 accepted outcomes without refitting:

```bash
TEL_TP2_EVAL_TAG=$(date -u +%Y%m%dT%H%M%SZ)
TEL_TP2_EVAL_DIR="/srv/token-energy-law/results/tp2-holdout-analysis/${TEL_TP2_EVAL_TAG}"

python3 scripts/evaluate_gh200_tp2_holdout.py \
  --campaign-lock results/manifests/gh200-tp2-nvlink-holdout.lock.json \
  --release-record configs/addenda/gh200-tp2-nvlink-holdout-release-v1.json \
  --identification-freeze-dir "$TEL_TP2_IDENTIFICATION_FREEZE_DIR" \
  --identification-lock results/manifests/gh200-tp2-nvlink-identification.lock.json \
  --execution-addendum configs/addenda/gh200-tp2-nvlink-v1.json \
  --results-root /srv/token-energy-law/results \
  --output-dir "$TEL_TP2_EVAL_DIR"
```

The primary holdout gate is fixed before outcomes: median cell absolute
relative error at most 5%, maximum at most 10%, and all six cells at most 10%.

## Claim boundary

A passing unopened holdout can support the duration-augmented form with newly
identified coefficients for the pinned Qwen2.5-7B, two-GH200, TP=2, NV18,
vLLM-V0/BF16 stratum over the tested interpolation domain. It cannot support
universal coefficients, TP1-to-TP2 zero-shot transfer, a separately identified
NVLink/NCCL energy term, another tensor-parallel degree, another node/SKU,
prospective latency without a duration model, causal idle-power semantics, or
any DVFS conclusion.
