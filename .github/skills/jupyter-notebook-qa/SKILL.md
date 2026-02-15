---
name: jupyter-notebook-qa
description: Run Jupyter notebooks end-to-end, capture errors, and verify expected outputs.
---

# Jupyter Notebook QA

## When to use
Use for validating notebooks after changes or before sharing results.

## Inputs
- Notebook path
- Target cells or full run
- Expected outputs or checkpoints

## Steps
1. Run notebook cells top-to-bottom in order.
2. Capture any exceptions and stop at the first failure.
3. Verify key outputs (tables, metrics, plots) against expectations.
4. Record the failing cell number and error summary.
5. If failures occur, fix and re-run from the start.

## Outputs
- Pass/fail summary
- List of failing cell numbers (if any)
- Verified output notes

## Failure conditions
- Any exception
- Missing or incorrect expected output

## Constraints
- Never change the selected kernel.
- Install dependencies inside notebook cells only.
