# GH200 scoped-power meter audit

GH200 exposes two distinct energy boundaries. NVML power field IDs 185
(one-second average) and 186 (instantaneous) use `scopeId=0` for the GPU board
and `scopeId=1` for the Grace Hopper module. The module includes the GPU,
supported NVIDIA CPU, and other module components.

On the validated dual-GH200 144 GB stack (driver 580.173.02),
`nvmlDeviceGetTotalEnergyConsumption()` tracks module-scope power. An idle
30-second audit found counter-derived averages of 150.224 W and 147.587 W;
synchronized module-power observations were 150.68 W and 147.76 W. Legacy
`nvmlDeviceGetPowerUsage()` instead matched GPU-scope observations near
94--97 W. Comparing those unlike boundaries creates a spurious 35--38%
disagreement.

Run a fresh, synchronized 60-second audit before model experiments:

```bash
python3 scripts/collect_nvml_scoped.py \
  --duration-seconds 60 --interval-ms 100 \
  --output-jsonl artifacts/private/gh200-idle-scoped.jsonl \
  --summary-json artifacts/private/gh200-idle-scoped.summary.json
```

The audit passes only when sampling gaps are at most 250 ms and the integral of
module average power agrees with the cumulative counter within 2%. A large
GPU-vs-module difference is recorded as `cross_scope_difference`; it is not a
meter failure.

For GH200 studies, report both outcomes without mixing baselines:

- primary mechanism outcome: GPU-board joules per useful decode token, from
  boundary-aligned integration of scope-0 power;
- deployment outcome: module joules per useful decode token, from the
  cumulative counter, checked against scope-1 power integration.

Measure bare idle only to qualify the sensor. Experimental subtraction, if
used, requires a matched loaded-idle baseline with the same resident model,
runtime, clocks, and power policy.
