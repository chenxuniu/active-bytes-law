# Measurement contract

## Estimand and accounting

For every useful output token in a metered decode interval:

```text
B_eff  = total metered useful tokens / decode iterations
L_bar  = sum attended historical-token positions / total metered useful tokens
A_read = M_w / B_eff + K_KV * L_bar
A_rw   = A_read + K_KV
Pi_WK  = B_eff * K_KV * L_bar / M_w
```

`L_bar` is token-weighted, not request-weighted. `M_w` is unique physical
storage actually resident in the runtime, not a parameter-count estimate.
`K_KV` contains K and V across all local layers, KV heads, head dimension,
bytes per element, replication, and layout padding.

Logical active bytes are a mechanism-aligned explanatory variable. They are
not automatically equal to physical HBM traffic because cache reuse, fusion,
layout, and kernel choice intervene. NCU anchors test that relationship.

## Decode boundary

All requests produce exactly one bootstrap output outside the energy interval
and are then parked. Both interval boundaries use `torch.cuda.synchronize()`.
The host applies the validated platform-specific meter after the ready marker
and after the done marker. Counter values are converted from millijoules to
joules explicitly; their scope must be declared rather than inferred.

```text
requested API outputs/request:      1025
unmetered bootstrap outputs/request:   1
metered pure-decode outputs/request: 1024
metered useful outputs/episode:      1024 * B
```

## Static identification trace

Each JSONL row represents one pure-decode engine iteration. In addition to the
schema-required fields, retain raw runtime sequence length when available. For
iteration `t=0..1023`, request `r` has normalized historical length `I_r+t`.
Prompt lengths use a balanced integer construction so their request mean plus
`511.5` equals the cell's target mean attended history exactly. For example,
the 4K, `B=8` cell uses four 3,584-token and four 3,585-token prompts.

The static validator rejects:

- an iteration count other than the declared 1024;
- non-contiguous IDs or non-monotonic timestamps;
- changing request membership or one-token-per-request violations;
- mismatch between active requests, useful-token keys, and attended lengths;
- any speculation, late entry/exit, preemption, swap, recomputation, prefix
  reuse, or offload;
- backend, graph, weight dtype, or KV dtype drift;
- `B_eff != B` or `L_bar` unequal to the preregistered target.

## Energy aggregation

A repeat may contain multiple independent episodes. The repeat estimate is

```text
J/token = sum episode counter deltas / sum episode useful tokens.
```

Never average episode J/token values. Repeat until cumulative decode-only time
is at least 30 seconds. Coefficient of variation and confidence intervals use
independent repeat estimates, not episodes, tokens, or telemetry samples as
replicates.

Boundary-aligned 100 ms power telemetry is integrated with the trapezoidal
rule. Samples must bracket the decode interval; interpolation is used at each
boundary. The pilot requires cumulative-counter and integrated-power energy to
agree within 2% after declared boundary uncertainty.

### GH200 scoped-power specialization

On the validated Grace Hopper stack, sample scoped NVML instantaneous power at
10 ms requested cadence and reject any trace whose maximum realized gap exceeds
50 ms. Scope 0 (GPU board, including associated memory circuitry) is the
primary mechanism-facing energy boundary. Scope 1 is the Grace Hopper module
boundary and must agree with the module cumulative-energy counter within 2% on
a continuous decode-only audit of at least 5 seconds.

Do not use the one-second average-power field for boundary-aligned energy on
short decode intervals. In the loaded audit it lagged the cumulative module
counter by 5.65%, while instantaneous module-power integration differed by
0.81%. Average power remains a steady-state diagnostic only. The bare-idle and
loaded-idle baselines are boundary-specific; never subtract module idle from
GPU-board energy or vice versa.

## Profiler separation

NCU runs share cell settings and the `activebytes_decode` NVTX range but have
their own run IDs and domain. Replay, graph disabling, or kernel changes are
recorded. Profiled latency and energy never enter the primary estimates.

## Artifact linkage

Every public run manifest links its campaign lock hash, model/container/code
revisions, sanitized device fingerprint, runtime tensor inventory summary,
requested and trace-derived geometry, energy/trace/telemetry hashes, and QC or
censoring reason. Private artifacts may carry local identifiers but must never
be referenced by an absolute path in a public manifest.
