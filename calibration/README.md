# FP8 KV calibration environment

This image is a derived, calibration-only environment. It is never used to
collect paper energy or latency outcomes. The base inference image remains the
immutable NGC digest frozen in the campaign.

The dependency set is the successful resolver plan paired with
`compressed-tensors==0.10.2` already present in the base. In particular, LLM
Compressor 0.6.0.1 requires Transformers at or below 4.52.4, so the calibration
image deliberately uses 4.52.4 while the frozen inference image continues to
use 4.55.2. The calibration image removes vLLM entirely so that the incompatible
inference dependency cannot be invoked accidentally and `pip check` remains a
hard build gate. Every produced checkpoint must be loaded and audited again in
the frozen inference image before it can enter a new campaign lock.

Build without changing the base image:

```bash
docker build \
  --file calibration/Dockerfile \
  --tag token-energy-law-calibration:0.2 \
  .
```

Then capture the image ID, package inventory, and API surface before opening a
model or dataset. The repository must be mounted at
`/workspace/active-bytes-law`:

```bash
docker image inspect token-energy-law-calibration:0.2 \
  --format 'image_id={{.Id}} architecture={{.Architecture}} created={{.Created}}'

docker run --rm \
  -v "$PWD:/workspace/active-bytes-law:ro" \
  --entrypoint python3 \
  token-energy-law-calibration:0.2 \
  /workspace/active-bytes-law/scripts/inspect_calibration_stack.py
```

This is a smoke test only. Dataset download and calibration remain closed until
the reported API is reviewed.
