# GH200 Mistral-7B architecture qualification

This is a non-energy feasibility gate for the next external-validity study.
It must run before any paper-eligible Mistral energy outcome is collected.

## Why this model

`mistralai/Mistral-7B-Instruct-v0.3` is close to the audited Qwen2.5-7B
weight scale but changes the architecture and KV geometry. Its 32 layers,
8 KV heads, 128-dimensional heads, and BF16 cache imply

```text
K_KV = 2 (K,V) * 32 layers * 8 KV heads * 128 dimensions * 2 bytes
     = 131072 bytes per attended token.
```

The audited Qwen2.5-7B value is 57344 bytes, so this model provides 2.286x
larger logical KV traffic per historical token at similar weight scale. This
is a more informative architecture replication than adding another Qwen size.

The frozen model revision is
`c170c708c41dac9275d15a8fff4eca08d52bab71`. The public model repository is
ungated and reports Apache-2.0 licensing. The model revision, license, and
publication eligibility must still be reviewed by the authors before public
release.

## Frozen qualification

The single qualification cell targets mean attended history 16384, batch 16,
and 1024 metered decode tokens per request. It uses the existing GH200 runtime
stratum: one GH200 144GB HBM3e, 700 W board limit, the pinned vLLM 25.09 ARM64
container, FlashAttention, BF16 weights, and BF16 KV cache.

The run performs a batch doctor and a runtime inventory. It does not start the
NVML energy collector and cannot enter paper energy outcomes. A pass requires:

- the exact model revision and declared architecture;
- exact attended-history and batch geometry;
- at least five seconds of pure decode;
- all discovered KV tensors resident on GPU in BF16;
- unique weight storage between 13.5 and 16.5 decimal GB;
- the frozen container, driver, power limit, and memory-hotplug state.

## Run on the GH200 node

```bash
cd /srv/token-energy-law/repo
git pull --ff-only
git status --short
git rev-parse HEAD

(
  cd configs/addenda
  sha256sum -c gh200-mistral7b-qualification-v1.json.sha256
)
(
  cd results/manifests
  sha256sum -c gh200-mistral7b-qualification-v1.lock.json.sha256
)

sudo nvidia-smi -i 0 -pl 700
sudo nvidia-smi -pm 1
./scripts/run_gh200_mistral7b_qualification.sh 0
```

The first run may download roughly 15--29 GB depending on Hugging Face cache
deduplication. Preserve the complete qualification directory and its
`artifacts.sha256` file. Do not collect Mistral energy after a pass until a new
content-addressed identification/holdout design has been reviewed and frozen.

## Interpretation

A passing qualification establishes feasibility only. It does not support the
Token-Energy functional form on Mistral, coefficient equality, uncertainty
transfer, cross-hardware generalization, or a causal duration interpretation.
Those claims require a new identification stage followed by a separately
sealed no-refit holdout.
