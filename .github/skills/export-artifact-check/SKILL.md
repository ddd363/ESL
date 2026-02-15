---
name: export-artifact-check
description: Validate exported files for existence, size, schema, and timestamps.
---

# Export & Artifact Check

## When to use
Use after saving models, metrics, or datasets.

## Inputs
- Output paths
- Expected schema or file formats

## Steps
1. Confirm file existence and size.
2. Validate schema or headers.
3. Check timestamps for freshness.
4. Log a manifest of artifacts.

## Outputs
- Artifact manifest
- Integrity status

## Failure conditions
- Missing, empty, or malformed artifacts
