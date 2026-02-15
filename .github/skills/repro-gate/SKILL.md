---
name: repro-gate
description: Enforce reproducibility with seeds, config echo, and metric comparison.
---

# Repro Gate

## When to use
Use before publishing results or comparing experiments.

## Inputs
- Seed(s)
- Config parameters
- Baseline metrics (optional)

## Steps
1. Set seeds for all relevant libraries.
2. Echo configuration and environment info.
3. Re-run critical cells in order.
4. Compare metrics to baseline with tolerance.

## Outputs
- Repro status (pass/fail)
- Metric deltas
- Config snapshot

## Failure conditions
- Metrics outside tolerance
- Non-deterministic output across runs
