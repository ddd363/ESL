---
name: feature-pipeline-audit
description: Trace feature generation for leakage, alignment, and schema consistency.
---

# Feature Pipeline Audit

## When to use
Use after feature engineering or pipeline changes.

## Inputs
- Raw data
- Feature generation cells or functions

## Steps
1. Trace inputs and outputs for each feature step.
2. Validate alignment of features with targets.
3. Check for leakage (future data, target contamination).
4. Confirm stable schema and ordering.

## Outputs
- Feature list with provenance
- Leakage check result

## Failure conditions
- Leakage risk detected
- Misalignment between features and labels
