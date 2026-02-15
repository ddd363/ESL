---
name: backtest-sanity
description: Compare backtest outputs to baseline and validate key risk metrics.
---

# Backtest Sanity

## When to use
Use after strategy changes or data updates.

## Inputs
- Current backtest results
- Baseline results (if available)

## Steps
1. Compare returns, drawdown, volatility, and hit rate.
2. Validate transaction counts and turnover.
3. Check for lookahead or leakage artifacts.
4. Summarize deltas against baseline.

## Outputs
- Comparison table
- Pass/fail assessment

## Failure conditions
- Significant degradation beyond tolerance
- Evidence of leakage
