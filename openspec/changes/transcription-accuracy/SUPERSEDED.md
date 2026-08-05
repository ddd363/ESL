# Superseded

This change is superseded by the transcription-accuracy direction approved on
2026-08-03. Do not implement its task list unchanged.

The replacement work:

- pins Deepgram `nova-3` and English explicitly;
- uses Nova-3 `keyterm` prompting only for curated, evaluated vocabulary;
- treats endpointing and smart formatting as measured experiments, not fixed
  assumptions;
- sends only finalized Deepgram text to OpenAI analysis;
- preserves raw ASR text and never filters, replaces, or votes on words from
  confidence scores;
- provides `tools/evaluate_transcription.py` and `evaluation/` templates for
  WER and manually adjudicated false learner-error measurements.

The existing tasks conflict with this direction by requiring legacy
`keywords` for Nova-3 and proposing confidence-driven transcript handling.
Replicate configuration/UI changes in this old proposal require a separate,
validated proposal before implementation.

The local `openspec` command could not be used to archive this change because
its launcher cannot locate a Node runtime in this environment.