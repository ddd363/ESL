---
name: viz-check
description: Generate standard plots and verify labels, scales, and outputs.
---

# Visualization Check

## When to use
Use after plotting or reporting visuals.

## Inputs
- Dataset
- Plot specifications or expected plot types

## Steps
1. Generate plots from the provided spec.
2. Verify axis labels, titles, and scales.
3. Check for empty or NaN-only plots.
4. Save figures if required.

## Outputs
- Saved plots (if applicable)
- Visual checklist (labels, scales, data presence)

## Failure conditions
- Missing labels or titles
- Invalid axes or empty plots
