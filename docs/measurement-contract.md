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
The host reads cumulative DCGM total energy after the ready marker and after the
done marker. The delta is converted from millijoules to joules explicitly.

```text
requested API outputs/request:       129
unmetered bootstrap outputs/request:   1
metered pure-decode outputs/request:  128
metered useful outputs/episode:       128 * B
```

## Static identification trace

Each JSONL row represents one pure-decode engine iteration. In addition to the
schema-required fields, retain raw runtime sequence length when available. For
iteration `t=0..127`, the normalized historical length is `I+t`.

The static validator rejects:

- an iteration count other than the declared 128;
- non-contiguous IDs or non-monotonic timestamps;
- changing request membership or one-token-per-request violations;
- mismatch between active requests, useful-token keys, and attended lengths;
- any speculation, late entry/exit, preemption, swap, recomputation, prefix
  reuse, or offload;
- backend, graph, weight dtype, or KV dtype drift;
- `B_eff != B` or `L_bar != I+63.5`.

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
