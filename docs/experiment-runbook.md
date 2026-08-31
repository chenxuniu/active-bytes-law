# H100 experiment runbook

This is the stop/go execution contract for `active-bytes-v1`. It assumes one
exclusive, non-MIG H100 80 GB node with administrative access. One node is
enough for the P0 study: budget 50--65 H100 hours and reserve 80 hours for QC
reruns. Pilot, primary energy, and profiler runs are separate data domains.

## 0. Clone, test, and isolate private output

```bash
git clone https://github.com/chenxuniu/active-bytes-law.git
cd active-bytes-law
python3 -m unittest discover -s tests -v
python3 scripts/check_publication_safety.py
export AB_PRIVATE_DIR="$PWD/artifacts/private"
mkdir -p "$AB_PRIVATE_DIR"
bash scripts/collect_preflight.sh
```

The collector writes a complete local inventory only to the ignored private
directory and creates a whitelist-only public profile under `results/`.
Inspect the public JSON before committing it. Never commit the raw inventory.

Record the original GPU state before changing it:

```bash
nvidia-smi --query-gpu=index,name,persistence_mode,power.limit,clocks.current.sm,clocks.current.memory,mig.mode.current \
  --format=csv,noheader,nounits
sudo nvidia-smi -pm 1
sudo nvidia-smi -i 0 -pl 700
nvidia-smi --query-gpu=index,name,power.limit,mig.mode.current \
  --format=csv,noheader,nounits
```

Use the same clock policy as the previous controlled campaign. Do not begin by
locking clocks unless clock locking is itself a preregistered treatment. Save
the original state privately and restore it when the campaign finishes.

Freeze these versions before instrumentation: driver, CUDA, DCGM, Nsight
Compute, PyTorch, vLLM, attention backend, container digest, model revision,
tokenizer revision, and repository commit. A version label without a digest or
revision is insufficient.

## 1. Build the decode-only integration

Do not meter a client-wide request. Connect the experiment control layer to a
pinned offline vLLM `LLMEngine`/engine-step runner, or an equivalent server-side
patch, with this state machine:

```text
LOAD -> ADD_B_REQUESTS -> PREFILL_AND_BOOTSTRAP -> PARK_EACH_REQUEST
     -> ALL_B_PARKED -> CUDA_SYNC -> DECODE_READY
     -> HOST_READS_ENERGY_START -> GO -> ACTIVEBYTES_DECODE_NVTX
     -> EXACTLY_128_PURE_DECODE_STEPS -> CUDA_SYNC -> DECODE_DONE
     -> HOST_READS_ENERGY_END -> ACK -> ATOMIC_ARTIFACT_WRITE
```

For each request set:

```text
requested_api_output_tokens              = 129
unmetered_bootstrap_tokens_per_request   = 1
metered_decode_tokens_per_request        = 128
ignore_eos                               = true
temperature                              = 0
```

The first output is sampled by prefill and is not metered. It seeds 128 later
pure-decode steps. Each request must be parked immediately after its bootstrap
token. Open the common gate only when all `B` requests have exactly one output;
no request may start its second output early. This matters when long prompts
are prefetched in chunks or waves.

The canonical context variable is historical length **before** the current
token's KV write. The 128 values for an input of length `I` are
`I, I+1, ..., I+127`; their mean is `I+63.5`. If a runtime field includes the
current query token, retain it as `runtime_seq_len_raw` and store
`attended_history_tokens = runtime_seq_len_raw - 1`. Verify this conversion with
a one-request token-by-token doctor test before the pilot.

Required integration checks:

1. markers have a CUDA synchronization on both boundaries;
2. the host reads cumulative DCGM field 156 after `DECODE_READY` and before
   `GO`, then after `DECODE_DONE` and before `ACK`;
3. an NVTX push/pop range named `activebytes_decode` covers only metered steps;
4. every step emits the fields in `schemas/iteration-trace.schema.json`;
5. trace, energy, and manifest files use atomic rename and SHA-256;
6. prefix caching, speculation, CPU/KV offload, and swap are disabled;
7. the static identification runner admits the entire requested KV footprint;
   an infeasible cell is recorded as capacity-censored, not made feasible by
   offload.

## 2. Run a one-request instrumentation doctor

Use `I=32`, `B=1`, and eight metered steps in a non-paper doctor run. Inspect
the trace manually. Normalized historical lengths must be `32..39`, request
membership must be constant, and the energy marker must exclude prefill. Then
run the strict validator with `--measured-decode-tokens 8`.

For the pinned NVIDIA vLLM V0 engine, the repository doctor performs one
unmetered bootstrap `engine.step()`, synchronizes CUDA, reads the cumulative
energy counter, and then admits exactly eight metered `engine.step()` calls:

```bash
VLLM_USE_V1=0 python3 scripts/run_decode_doctor.py \
  --model Qwen/Qwen3-0.6B \
  --prompt-tokens 32 --measured-decode-tokens 8 \
  --output-json artifacts/private/decode-doctor.json
```

This small-model run verifies instrumentation only and is marked
`non_paper_measurement`. Repeat the doctor with the pinned study model and
runtime mode before opening pilot outcomes. The doctor rejects any first step
that does not produce exactly one bootstrap token, any metered step that does
not add exactly one cumulative output token, early/late completion, timestamp
disorder, or a decreasing cumulative-energy counter.

The Qwen study model is frozen at revision
`a09a35458c702b33eeacc393d103063234e8bc28`. Pass that value with
`--model-revision`; pass `--runtime-mode graph` for the study-mode doctor.

Do not continue until the doctor demonstrates a real bootstrap barrier. TTFT,
client timestamps, or subtraction of a separately timed prefill do not satisfy
the boundary contract.

## 3. Freeze the four-cell pilot

```bash
python3 scripts/expand_campaign.py configs/campaigns/pilot.json \
  --output results/manifests/pilot.lock.json
sha256sum results/manifests/pilot.lock.json
```

The pilot uses Qwen2.5-7B, BF16 weights, TP=PP=1, fixed backend and graph mode,
no prefix cache, speculation, offload, or swap:

| Cell | Actual context | Target batch | KV dtype | Repeats |
|---|---:|---:|---|---:|
| P-A | 4,096 | 8 | BF16 | 3 |
| P-B | 4,096 | 8 | FP8 | 3 |
| P-C | 16,384 | 32 | BF16 | 3 |
| P-D | 16,384 | 32 | FP8 | 3 |

Follow the order in the lock file. A repetition may contain multiple
independent 128-step episodes. Sum only decode-marker time, energy, and useful
tokens until that repetition contains at least 30 seconds of decode. Never add
prefill, restart, or between-episode idle to reach the duration threshold.

For each trace:

```bash
python3 scripts/validate_trace.py \
  --trace path/to/trace.jsonl \
  --batch TARGET_BATCH \
  --prompt-tokens ACTUAL_CONTEXT \
  --measured-decode-tokens 128 \
  --report path/to/trace.qc.json
```

After all repeats:

```bash
python3 scripts/summarize_energy.py \
  --input path/to/pilot-energy.jsonl \
  --output results/summaries/pilot-energy.json \
  --minimum-repeats 3 \
  --minimum-decode-seconds 30
```

The pilot passes only if every condition holds:

- each episode has exactly `128*B` metered useful tokens;
- conventional decode has one useful accepted token per active request per
  step and zero speculative rejection;
- `B_eff == B` and `L_bar == I+63.5` at accounting precision;
- no late entry/exit, preemption, swap, offload, recomputation, or prefix reuse;
- backend, graph mode, dtype, runtime weight inventory, and KV accounting are
  stable across matched repeats;
- cumulative-energy delta and boundary-aligned 100 ms power integration differ
  by at most 2%;
- repeat-level J/useful-token CV is at most 3%;
- one separate NCU anchor reports plausible, nonzero DRAM read and write bytes.

If CV exceeds 3%, extend each repeat to 60 seconds. If it still fails, use
8--10 repeats and diagnose temperature, clocks, throttling, and boundary noise.
If a 16K cell is infeasible, preserve the censored record and substitute the
nearest feasible BF16/FP8-matched `(L,B)` pair. Do not use swap or offload.

## 4. Freeze all confirmatory campaigns before opening outcomes

```bash
for name in qwen-core weight-treatment window-placebo llama-holdout dynamic; do
  python3 scripts/expand_campaign.py "configs/campaigns/${name}.json" \
    --output "results/manifests/${name}.lock.json"
done
python3 scripts/check_publication_safety.py
```

For the Qwen grid, first run a capacity admission check without measuring
energy. If 16K/B64 BF16 is infeasible, change `B_top` to 48 for **all** Qwen
treatments, create new campaign IDs, and freeze again. Keep the rejected B64
admission record.

The preregistered Qwen split is:

- coefficient fit: `B={4,32}` across all contexts and KV dtypes, 12 cells;
- residual calibration: `B=16`, 6 cells;
- evaluation: `B={8,B_top}`, 12 cells.

Pilot measurements never enter these splits; rerun coincident main cells.

## 5. Run the mechanism gates and primary grid

First inventory unique physical model storage and calculate `K_KV`. Packed
weights, scales, tied tensors, and views require storage-level deduplication.
Run the unit-tested accounting code against the saved inventory.

Discover installed Nsight metric names rather than copying names from another
release:

```bash
ncu --query-metrics | grep -E 'dram__bytes|lts__t_bytes|hit_rate|throughput'
```

Profile only the decode NVTX range. Select the installed equivalents of total
DRAM reads/writes, L2 traffic/hit rate, and SM/DRAM activity. The command shape
is:

```bash
ncu --target-processes all --nvtx --nvtx-include 'activebytes_decode/' \
  --replay-mode range --metrics 'DISCOVERED_METRICS' \
  --export 'artifacts/private/ncu/RUN_ID' RUNNER_COMMAND
```

Run 12--16 frozen anchors, three profiler repeats each. Profiler replay can
change graph mode and timing, so NCU runs are mechanism diagnostics only. Do
not use their energy or latency in primary estimates.

Then execute Qwen core in lock-file order: 30 cells x 5 repeats = 150 primary
energy records. Each repeat follows this exact sequence:

1. launch and verify the immutable server contract;
2. generate a tokenizer-verified exact-length prompt;
3. add `B` requests for 129 outputs and park all after bootstrap;
4. synchronize, emit `DECODE_READY`, and read field 156;
5. release `GO` and meter exactly 128 pure-decode steps per request;
6. synchronize, emit `DECODE_DONE`, and read field 156;
7. validate the trace immediately;
8. add independent episodes until decode-only duration is at least 30 seconds;
9. atomically write manifest, energy, trace, telemetry references, and hashes;
10. record QC, exclusion, or capacity-censoring status without deleting rows.

Randomized cyclic scheduling means one repeat of every cell per cycle. Do not
run all five repeats of one cell consecutively.

## 6. Run interventions and placebo

The paired BF16/FP8 KV grid is the primary byte intervention. Record the exact
FP8 scale/calibration artifact and any backend/kernel change; otherwise the
treatment is not a pure storage-byte intervention.

The weight treatment compares an engine-native FP8-weight path with the BF16
baseline at actual context 8K and batches `4,8,16,32,B_top`. Promote it as a
causal byte intervention only if physical-storage inventory, scales, kernels,
and runtime contract pass audit. Otherwise label it a confounded stress test.

The configured-window placebo holds actual context at 4K, uses batches
`8,16,32`, and varies `max_model_len={8K,16K,32K}`. Check actual geometry,
`B_eff`, `L_bar`, allocated/live KV blocks, backend, graph capture, scheduler,
power, and clocks. If changing the window changes allocator or kernel behavior,
report that mechanism rather than calling it a null placebo.

## 7. Freeze coefficients, then open holdout and dynamic outcomes

Fit the H100/Qwen coefficients and freeze the residual envelope before loading
Llama outcomes. The architecture holdout is Llama-3.1-8B, BF16 KV,
`L={2K,4K,8K}`, `B={4,8,16,32}`: 12 cells x 5 repeats. Do not refit coefficients.

Dynamic P0 uses a parked reservoir so the metered interval remains decode-only:
short=1K, long=8K, and deterministic 50/50 mixed prompts crossed with release
loads at 25%, 55%, and 85% of a separately measured stable decode capacity.
Freeze the release trace and seed. Analyze realized `B_eff` and token-weighted
`L_bar`, not offered load or configured batch. Real arrivals with interleaved
prefill are a mixed-phase extension, not part of the core decode law.

## 8. Final gates and publication

The 65 non-pilot cells yield 325 energy repeats: 150 Qwen core, 25 weight, 45
placebo, 60 Llama holdout, and 45 dynamic. Add 12 pilot runs and 36--48 separate
profiler runs.

Before promoting the strongest claim, evaluate the frozen gates in
[the analysis plan](analysis-plan.md). Preserve all failed and censored rows.
Publish small reviewed tables and manifests in Git; archive approved raw data
outside Git with URL, SHA-256, size, and schema version. Run the safety scanner
on the full Git index before every push.
