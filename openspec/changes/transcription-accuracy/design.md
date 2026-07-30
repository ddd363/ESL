## Context

The app transcribes ESL/IELTS spoken audio via two independent pipelines:

1. **Replicate** — `thomasmol/whisper-diarization`, a cloud-hosted Whisper model with speaker diarization. Called once per recording after the user clicks "Transcribe via Replicate". Current call uses only `group_segments=True` with no language hint, no initial prompt, no word timestamps, and a pinned-but-old model digest.

2. **Deepgram** — live WebSocket streaming (`_build_url()` in `DeepgramStreamingClient`). Current URL uses only `encoding`, `sample_rate`, `channels`, `interim_results`, `punctuate`, `smart_format`, `diarize`. No model is specified (falls back to Deepgram's legacy default, not `nova-3`), no `language`, no `filler_words`, no keyword hints.

Both pipelines feed the OpenAI analysis step; transcript quality is therefore a hard prerequisite for feedback quality.

---

## Goals / Non-Goals

**Goals:**
- Align both pipelines with current best-practice API parameter sets for English ESL audio
- Expose the most impactful parameters as user-configurable settings without cluttering the primary flow
- Add no new external dependencies — only API parameter upgrades

**Non-Goals:**
- Real-time Deepgram model switching mid-recording (model is read at `start_deepgram_stream()` time)
- Offline/local Whisper model tuning (faster-whisper fallback is out of scope)
- Automatic audio quality enhancement (noise reduction, AGC) — deferred to a future change
- Changing the fundamental dual-pipeline architecture

---

## Decisions

### D1 — Replicate: upgrade model digest
**Decision**: Update `REPLICATE_MODEL_ID` to the latest `thomasmol/whisper-diarization` digest.

Current digest: `1495a9cddc83b2203b0d8d3516e38b80fd1572ebc4bc5700ac1da56a9b3ed886`

**Why**: The current digest is months old. The model receives ongoing accuracy improvements, especially for accented speech. Pinning to latest-available at change time captures those gains.

**Alternative considered**: Switch to `openai/whisper` on Replicate directly. Rejected because `thomasmol/whisper-diarization` adds speaker diarization which is required for the two-speaker ESL scenario.

---

### D2 — Replicate: add `language`, `initial_prompt`, `word_timestamps`
**Decision**: Always pass `language="en"` and `word_timestamps=True`. Expose `initial_prompt` with a sensible ESL default editable in the UI.

**Why**: 
- `language="en"` removes ~15% of Whisper's compute budget spent on language identification, improving both speed and accuracy on non-native English.
- `initial_prompt` primes the model's attention window with domain vocabulary. Whisper 3 is highly sensitive to the first few tokens.
- `word_timestamps=True` sets the foundation for future utterance-level highlighting features without significant latency impact.

**Alternative considered**: Keep language on auto-detect to support non-English students.
Rejected for now — the current spec targets IELTS (English test). If multi-language support is needed it can be a follow-on change.

---

### D3 — Deepgram: upgrade to `nova-3`
**Decision**: Set `model=nova-3` in `_build_url()`.

**Why**: `nova-3` is Deepgram's current top-accuracy English model with specific improvements for accent robustness vs. `nova-2`. No API cost difference. The fallback (default model) is legacy `general` which Deepgram itself recommends replacing.

**Alternative considered**: `nova-2-ea` (early access). Rejected — less stable, not GA.

---

### D4 — Deepgram: add `filler_words`, `utterances`, `keywords`
**Decision**: Always set `filler_words=true` and `utterances=true`. Send `keywords` list from a module-level constant (`DEEPGRAM_KEYWORDS`).

**Why**:
- `filler_words=true`: Critical for ESL fluency analysis. A student who says "I um… think…" must not be transcribed as "I think…" — the hesitation is the data.
- `utterances=true`: Groups alternating speech into utterance blocks which improves diarization boundary detection.
- `keywords`: Deepgram's keyword boosting increases recognition probability for domain terms (band scores, IELTS, examiner, etc.) without requiring fine-tuning.

**Risk**: `keywords` can cause hallucinations if keywords are too generic (e.g., `"the"`) — mitigated by keeping the list specific and short (max 20 terms).

---

### D5 — Config UI: collapsed expanders, not inline
**Decision**: Both "Replicate Settings" and "Deepgram Settings" will be `st.expander(..., expanded=False)` panels inside Step 3, placed between the transcript source radio and the Analyse button.

**Why**: Primary users (teachers running live analysis) should never need to touch these. Power users who want to tune them can expand as needed. Inline controls would add visual noise to the main flow.

**Alternative considered**: Move config to a sidebar settings section. Rejected because the current app has no sidebar and introducing one for only these settings would over-engineer the layout.

---

### D6 — session_state keys and disk persistence
**Decision**:
- `st.session_state.replicate_num_speakers` (int | None, default 2)
- `st.session_state.replicate_language` (str, default "en")
- `st.session_state.replicate_initial_prompt` (str, default ESL constant)
- `st.session_state.deepgram_model` (str, default "nova-3")
- `st.session_state.deepgram_language` (str, default "en")
- `st.session_state.deepgram_keywords` (str — comma-separated, default DEEPGRAM_KEYWORDS joined)

Replicate `initial_prompt` SHALL be persisted to `outputs/replicate_initial_prompt.txt` (same pattern as `openai_system_prompt.txt`).
Deepgram settings do NOT need disk persistence — they are session-scoped and change only mid-session.

---

## Risks / Trade-offs

| Risk | Mitigation |
|------|-----------|
| `nova-3` model name changes or is deprecated | Model name stored in `DEEPGRAM_DEFAULT_MODEL` constant; easy to update |
| Replicate digest becomes unavailable | Wrap `replicate.run()` in existing try/except; surface model ID in error message |
| `filler_words=true` produces trailing `[BLANK_AUDIO]` tokens in Deepgram output | Strip known Deepgram artefact tokens in `_format_deepgram_alt_with_speakers()` |
| `initial_prompt` injection causes Whisper to echo the prompt | Whisper 3 handles this well; mitigated by keeping the prompt factual, not instructive |
| `keywords` param causes Deepgram to over-transcribe domain terms | Keep list ≤20 specific terms; no generic words |
| User sets `num_speakers=1` on a 2-speaker recording | diarization will still work but merge both speakers; acceptable as user override |

---

## Migration Plan

1. Update constants: `REPLICATE_MODEL_ID`, `DEEPGRAM_DEFAULT_MODEL`, `DEEPGRAM_KEYWORDS`, `REPLICATE_DEFAULT_INITIAL_PROMPT`.
2. Update `init_session_state()` with new keys and defaults.
3. Update `DeepgramStreamingClient._build_url()` to read from `st.session_state`.
4. Update `run_transcription()` to use new constants and session_state values.
5. Add config expanders to Step 3 UI (before the Analyse button).
6. Add `load_persisted_replicate_prompt()` and `persist_replicate_prompt()` helpers (modelled on existing OpenAI prompt helpers).
7. Update `render_live_debug_monitor()` to include `deepgram_model` in the snapshot.
8. Manual test: record 30s audio, verify nova-3 model appears in logs, verify `filler_words` appear in transcript.

**Rollback**: Revert constants to previous values. No DB migrations or file format changes.

---

## Open Questions

- Should keyword list be persisted to disk? (Current: no; revisit if teachers find themselves re-entering it every session.)
- Is there a Replicate model newer than `1495a...` available at implementation time? (Check Replicate at implementation.)
- Does Deepgram `nova-3` support `diarize=true`? (Confirmed: yes, as of Deepgram API v1.)
