# Active-Bytes Law

Active-Bytes Law is a measurement and applied-modeling project for explaining
decode-time LLM inference energy with the bytes that are active for each useful
output token. Its central, testable accounting variable is

```text
A_read = M_w / B_eff + K_KV * L_bar,
```

where `M_w` is the unique resident model-weight storage, `B_eff` is the
trace-derived effective decode batch, `K_KV` is the model's key/value-cache
bytes per historical token, and `L_bar` is the useful-token-weighted mean
historical context. We also track the new-token KV write,
`A_rw = A_read + K_KV`, and the weight/KV parity ratio
`Pi_WK = B_eff * K_KV * L_bar / M_w`.

This repository is the public experiment contract and artifact index for the
paper. The completed Qwen2.5-7B/GH200 campaigns establish one
platform--model--runtime instantiation; they do not by themselves establish a
cross-model or cross-platform law. The next gate is a non-paper
Qwen2.5-14B/GH200 qualification. No 14B energy campaign is frozen or opened
until its model, geometry, decode boundary, and runtime-storage audits pass.

## What is ready

- deterministic campaign expansion and immutable SHA-256 lock files;
- Active-Bytes and runtime-storage accounting;
- strict trace validation for one unmetered bootstrap followed by an exact
  metered pure-decode interval, including the current 1024-token campaigns;
- DCGM cumulative-energy aggregation across episodes and CV across independent
  repeats;
- boundary-interpolated trapezoidal integration of sampled power telemetry;
- scoped NVML collection that keeps GH200 GPU-board and module energy separate;
- executable single- and multi-request vLLM decode-boundary doctors;
- campaign membership plus public artifact size/SHA-256 validation;
- a whitelist-only public system-profile collector;
- a publication-safety scanner and CI tests;
- preregistered pilot, core, intervention, placebo, holdout, dynamic, and NCU
  anchor configurations.

## Current evidence boundary

The vLLM V0 runner now exposes an `LLMEngine`/engine-step decode gate and emits
the trace described in [the measurement contract](docs/measurement-contract.md).
The primary energy interval begins only after prefill, CUDA synchronization,
and the `DECODE_READY -> GO` handshake; it ends after the exact frozen number of
useful decode iterations. A client-wide serving benchmark that contains prefill
is retrospective evidence, not confirmatory Active-Bytes data. The remaining
scientific boundary is external validity: a second model must reproduce the
functional form under newly identified coefficients before the paper can make
a cross-model claim.

## Start here on the measurement node

```bash
git clone https://github.com/chenxuniu/active-bytes-law.git
cd active-bytes-law
python3 -m unittest discover -s tests -v
bash scripts/collect_preflight.sh
python3 scripts/expand_campaign.py configs/campaigns/pilot.json \
  --output results/manifests/pilot.lock.json
```

Then implement or connect the decode-only runner, execute the four locked pilot
cells, and validate every trace before looking at the main outcomes:

```bash
python3 scripts/validate_trace.py \
  --trace TRACE.jsonl --batch 8 --prompt-tokens 4096 \
  --measured-decode-tokens 128 --report TRACE.qc.json

python3 scripts/summarize_energy.py \
  --input ENERGY.jsonl --output results/summaries/pilot-energy.json \
  --minimum-repeats 3 --minimum-decode-seconds 30

python3 scripts/integrate_power.py \
  --input POWER.jsonl --start-ns START --end-ns END \
  --output POWER-INTEGRAL.json

python3 scripts/validate_manifest.py \
  --manifest RUN-MANIFEST.json --repository-root "$PWD"
```

The exact command order, stop/go gates, and expected GPU time are in the
[step-by-step experiment runbook](docs/experiment-runbook.md).

For Grace Hopper, first run the [GH200 scoped-power meter audit](docs/gh200-meter-audit.md).
The cumulative energy counter on the validated stack follows module power,
while the mechanism-facing primary outcome integrates GPU-board power. These
boundaries and their idle baselines must never be mixed.

The accepted loaded audit uses 10 ms requested instantaneous-power telemetry;
the one-second average field is diagnostic only because its boundary lag
exceeded the preregistered 2% agreement gate.

After the meter gate passes, run `scripts/run_decode_doctor.py` under the pinned
vLLM V0 engine. Its 32-token prompt, one unmetered bootstrap token, and eight
metered engine steps are instrumentation checks, not paper measurements.

## Campaigns

| Configuration | Role | Cells | Repeats |
|---|---|---:|---:|
| `pilot.json` | boundary/counter acceptance | 4 | 3 |
| `qwen-core.json` | BF16/FP8 KV identification grid | 30 | 5 |
| `weight-treatment.json` | weight-byte stress intervention | 5 | 5 |
| `window-placebo.json` | configured-window placebo | 9 | 5 |
| `llama-holdout.json` | architecture holdout | 12 | 5 |
| `dynamic.json` | realized-state validation | 9 | 5 |
| `ncu-anchors.json` | physical-traffic mechanism anchors | 16 | 3 |
| `gh200-v1-anchors.json` | frozen GH200 BF16 V1 application-range-replay anchors | 12 | 5 |
| `gh200-primary-bf16.json` | GH200 BF16 coefficient/discrepancy identification | 9 | 5 |
| `gh200-primary-bf16-evaluation.json` | separately sealed GH200 BF16 evaluation | 6 | 5 |
| `gh200-v2-duration-holdout.json` | later unopened duration-augmented holdout | 9 | 5 |
| `gh200-qwen2p5-14b-qualification.json` | non-paper second-model qualification | 1 | 1 |

The older generic grids remain design provenance.  The currently executable
primary path is the checksummed GH200 BF16 V1/identification/evaluation set
above.  NCU runs are mechanism measurements and never enter the primary energy
or latency estimates; evaluation remains sealed until the identification fit
and discrepancy envelope are frozen.

After validating the first V1 sweep manually, run a resumable inclusive range
with `scripts/run_gh200_v1_anchor_batch.sh START_ORDER END_ORDER 0`. The driver
skips accepted attempts and records run-specific failures before continuing
with an explicit execution gap. Infrastructure or contract failures that occur
before an attempt directory exists still abort the batch. Every invocation has
a persistent batch log and JSON summary. Re-running the same range skips accepted
outcomes and retries missing ones. Valid order values are 0 through 59.

Run an inclusive range of the frozen GH200 primary BF16 identification campaign
with `scripts/run_gh200_primary_bf16_batch.sh START_ORDER END_ORDER 0`. This
driver skips accepted alignments and survives SSH loss when launched under tmux.
Unlike the mechanism-only profiler batch, a failed confirmatory energy run stops
the sequence; rerun the same command after diagnosis to retry that exact order.

After all 45 identification runs are accepted, audit the telemetry available
outside their exact decode boundaries with
`scripts/audit_gh200_primary_idle_windows.py`.  This diagnostic is deliberately
non-promotional: it records whether pre/post samples exist, but cannot choose a
post-outcome idle estimator or convert gross board energy into an eligible
idle-corrected result.  Freeze that policy, or explicitly keep gross board
J/token as the estimand, before fitting coefficients and before releasing the
separate evaluation campaign.

The frozen GH200 identification campaign selected gross scope-0 GPU-board
J/useful-token as its primary outcome.  After the idle-window availability
audit, freeze the 30-run OLS--HC3 coefficient artifact and the disjoint 15-run
Bonferroni-$t$ discrepancy envelope with
`scripts/freeze_gh200_primary_identification.py`.  The command emits immutable
digests but does not itself open evaluation; a separate release record must bind
both digests first.

The content-addressed evaluation release is
`configs/addenda/gh200-primary-bf16-evaluation-release-v1.json`. It binds the
identification coefficient and discrepancy artifacts before opening the 30
held-out batch-8/batch-24 runs. Execute them only through
`scripts/run_gh200_primary_evaluation_attempt.sh` or the resumable batch driver
`scripts/run_gh200_primary_evaluation_batch.sh`; both fail closed when any
released digest changes.
After all 30 runs pass, execute `scripts/evaluate_gh200_primary_held_out.py`.
It never refits: it reports six cell means, empirical frozen-envelope coverage,
relative width/error, and the prespecified HC3 batch/context residual trends.

After all 60 anchors pass, audit artifact hashes, repetition completeness, cell
confidence intervals, and the raw two-component traffic-law fit with
`scripts/aggregate_gh200_v1_traffic.py`. This aggregate intentionally does not
apply the separately frozen cache/residency correction and is not itself a formal
V1 decision.

Before collecting any Qwen2.5-14B energy outcome, run the
[second-model qualification](docs/gh200-qwen2p5-14b-qualification.md). It checks
the immutable model revision, exact high-KV geometry, runtime weight storage,
and logical KV bytes. A pass authorizes campaign design only; identification
and holdout locks are created afterward so that the held-out coordinates remain
genuinely unopened.

## Data policy

Git stores configs, lock files, sanitized manifests, cell-level summaries,
figures, checksums, and small approved trace excerpts. Model weights, raw power
streams, full logs, and profiler binaries stay outside Git. A paper release
should archive approved raw data in a DOI-bearing repository and list its URL,
byte size, schema version, and SHA-256 here.

Raw preflight output can expose hostnames, device UUIDs, serial numbers, private
addresses, local paths, and lease metadata. It is written only under
`artifacts/private/`, which is ignored. Before every commit run:

```bash
python3 scripts/check_publication_safety.py
```

See [publication safety](docs/publication-safety.md) and confirm that public
release complies with all applicable institutional, employer, model-license,
and dataset-review requirements.

## Repository map

```text
configs/campaigns/   preregistered experiment definitions
docs/                measurement, execution, analysis, and release contracts
environment/         sanitized, non-identifying system declarations
results/             public manifests, summaries, figures, and checksums
schemas/             machine-readable artifact schemas
scripts/             zero-dependency command-line entry points
src/active_bytes/    accounting, campaign, trace, and energy logic
tests/                standard-library unit tests
```

## Reproduce the control layer

```bash
python3 -m unittest discover -s tests -v
for config in configs/campaigns/*.json; do
  out="/tmp/$(basename "${config%.json}").lock.json"
  python3 scripts/expand_campaign.py "$config" --output "$out"
done
python3 scripts/check_publication_safety.py
```

The project currently uses the MIT license. Experimental data released later
must carry an explicit data license and provenance statement.

## First publication

The repository owner can perform the one-time publication after authenticating
GitHub CLI. The script refuses a dirty worktree, an existing `origin`, an
existing same-name repository, failed tests, or a failed safety scan:

```bash
gh auth login -h github.com -p https --web --clipboard --skip-ssh-key
./scripts/publish_public_repo.sh
```
