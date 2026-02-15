---
name: data-sanity-check
description: Validate dataframes or arrays for schema, nulls, ranges, and duplicates.
---

# Data Sanity Check

## When to use
Use after data loading, feature generation, or merging datasets.

## Inputs
- DataFrame(s) or array(s)
- Expected schema (optional)

## Steps
1. Report shapes and dtypes.
2. Check null counts and percentages.
3. Validate value ranges for critical columns.
4. Detect duplicates on key columns.
5. Flag unexpected columns or missing required fields.

## Outputs
- Summary table (shape, dtypes)
- Anomaly report (nulls, range violations, duplicates)

## Failure conditions
- Critical nulls or schema mismatch
- Out-of-range values beyond tolerance
