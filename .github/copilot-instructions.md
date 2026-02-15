Issue Tracking

This project uses bd (beads) for issue tracking.

Run bd prime for workflow context, or install hooks (bd hooks install) for auto-injection.

Quick reference (MODIFIED):

bd ready - Find unblocked work

bd create "Title" --type task --priority 2 - Create issue

bd close <id> - Complete work

bd sync is FORBIDDEN and must never be run

For full workflow details: bd prime

CRITICAL SAFETY RULE (NON-NEGOTIABLE)

The agent must NEVER run bd sync

The agent must NEVER push, commit, or sync Beads to git or GitHub

Beads are local, session-scoped tracking artifacts only

Any use of bd sync, implicit or explicit, is a hard failure

Autonomous Jupyter Notebook Agent Instructions
Non-Negotiable Rules

Execute end-to-end. Complete every step yourself.

No advisory-only responses.

Never ask the user to run commands.

Auto-recover on failure until success.

Confirm correctness from outputs, not assumptions.

Beads (bd) Workflow - Exhaustive and Mandatory
Bead Creation Rules (STRICT)

Create Beads for EVERY actionable point in EVERY prompt

A single prompt may require multiple Beads

Bullet points, numbered lists, implied subtasks, and validation steps ALL require Beads

No work may exist outside a Bead

Bead Ownership Rules

The agent owns the entire lifecycle of all Beads it creates

The agent must not leave Beads open unintentionally

Mandatory Bead Sequence (PER BEAD)

bd onboard (first session only)

bd create "<descriptive title>"

bd show <id> to read context

bd update <id> --status in_progress

Execute full task with validation gates

bd close <id> when complete

Never run bd sync. Ever.

Session Completion Contract (HARD REQUIREMENT)

The agent must continue working until ALL Beads created from the current prompt are closed

Ending a response with open Beads is a failure unless:

Completion is technically impossible, AND

The blocker is explicitly identified

Beads must be opened and closed within the same prompt session insofar as possible

Deferring work to a future prompt without justification is disallowed

Notebook Execution Rules

Never change the selected kernel.

Install dependencies only in notebook cells.

Run cells sequentially.

Capture errors immediately, fix, and re-run.

Editing Rules

All file edits via IDE editor only.

No shell edits.

Minimal, targeted changes.

Preserve existing style and public APIs.

Communication Rules

Short, direct, action-oriented.

No internal tool mentions.

Provide file links when relevant.

Skills (Authoritative)

Notebook QA - run top to bottom - fail on any exception.

Data Sanity - shapes, dtypes, NaNs, ranges, duplicates.

Repro Gate - re-run critical cells, compare metrics.

Viz Check - validate plots, labels, saved artifacts.

Feature Pipeline Audit - detect leakage or misalignment.

Backtest Sanity - compare return, drawdown, vol.

Performance Profiling - time and memory checks.

Export & Artifact Check - existence and schema validation.

Prompt & Workflow Integration

Every prompt may spawn multiple Beads

All work is strictly scoped to those Beads

Bead closure means the agent asserts correctness

Review may reopen Beads

OpenSpec references included when relevant

OpenSpec updates only on scope or feasibility change

Iteration Loop

Receive prompt
→ create ALL required Beads
→ execute work
→ validate outputs
→ close ALL Beads
→ only then end response

Landing the Plane (Session Completion)

Ensure no Beads remain open

File Beads for remaining future work only if blocked

Run quality gates if code changed

Clean up local state

Do NOT sync Beads

Hand off context for next session