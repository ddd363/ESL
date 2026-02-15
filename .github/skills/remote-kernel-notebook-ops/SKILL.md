---
name: remote-kernel-notebook-ops
description: Operate safely in Jupyter notebooks that use a fixed remote kernel (no kernel changes).
---

# Remote Kernel Notebook Ops

## When to use
Use for any notebook work where the kernel is remote and must not be changed.

## Inputs
- Notebook path
- Target cells or full run
- Required dependencies (if any)

## Steps
1. Confirm the notebook uses a fixed remote kernel; do not change it.
2. Install dependencies only inside notebook cells (never via shell).
3. Run cells in order unless explicitly instructed otherwise.
4. Capture errors immediately; fix and re-run from the top if needed.
5. Avoid terminal-based notebook commands (no `jupyter`, no kernel switches).

## Outputs
- Executed notebook state
- Error log (if any)
- Confirmation of kernel integrity (unchanged)

## Failure conditions
- Kernel change attempted
- Dependency install outside notebook cells
- Out-of-order execution without instruction
