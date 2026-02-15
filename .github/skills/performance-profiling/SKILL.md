---
name: performance-profiling
description: Measure runtime and memory hotspots in notebook workflows.
---

# Performance Profiling

## When to use
Use when notebooks run slowly or memory spikes occur.

## Inputs
- Notebook path
- Target cells or sections

## Steps
1. Time each target cell or section.
2. Capture peak memory usage if available.
3. Identify hotspots and annotate causes.
4. Propose focused optimizations.

## Outputs
- Timing/memory report
- Ranked hotspots

## Failure conditions
- Runtime or memory exceeds defined limits
