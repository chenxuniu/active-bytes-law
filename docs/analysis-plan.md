# Preregistered analysis plan

## Identification split

For the 30-cell Qwen KV grid, freeze the top feasible batch before outcomes:

- coefficient fit: batches 4 and 32 across three contexts and two KV dtypes
  (12 cells);
- residual calibration: batch 16 (6 cells);
- evaluation: batches 8 and `B_top` (12 cells).

The pilot is used only for instrumentation and threshold selection. Coincident
main cells are rerun. Stream, GEMM, and attention microbenchmarks diagnose
mechanisms but are not mixed into the serving regression without a separately
frozen normalization rule.

## Model ladder

1. Report coefficient-free `A_read`, `A_rw`, and `Pi_WK`.
2. Fit the simplest preregistered serving relation on coefficient-fit cells,
   beginning with an intercept plus separate weight and KV active-byte terms.
3. Freeze coefficients on Qwen/H100.
4. Use residual-calibration cells to freeze the prediction envelope.
5. Open Qwen evaluation cells, then the configured-window placebo.
6. Open Llama holdout with no coefficient refit.
7. Evaluate dynamic traces with realized state variables.

Report uncertainty across independent repeats. A bootstrap or t interval must
resample at the repeat/cell level appropriate to the estimand, never treating
tokens or telemetry points as independent repetitions.

## Promotion gates

- energy 95% CI half-width no greater than `max(3%, 3*pilot_CV)`;
- no unexplained repeated one-sided violation of the physical-traffic lower
  bound;
- physical/logical traffic ratio median at most 1.25 and P90 at most 1.5 for
  preregistered counter anchors;
- configured-window placebo equivalence within `max(3%, 2*repeat_CV)`;
- paired byte-intervention effect has the predicted sign and its interval does
  not cross zero;
- predicted crossover is within one adjacent batch-doubling interval;
- nominal 95% frozen-envelope coverage is at least 90%, reported together with
  envelope width;
- Llama holdout median point error at most 15%, with no systematic residual
  trend against batch or context.

Failure of a strong gate does not justify deleting a cell. Demote the claim:
from law to bounded model, from causal intervention to confounded stress test,
or from portable predictor to device-specific characterization.
