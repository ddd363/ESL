## 1. Constants and Configuration

- [ ] 1.1 Look up the latest available `thomasmol/whisper-diarization` digest on Replicate and update `REPLICATE_MODEL_ID` constant in `app.py`
- [ ] 1.2 Add `DEEPGRAM_DEFAULT_MODEL = "nova-3"` constant to `app.py`
- [ ] 1.3 Add `DEEPGRAM_KEYWORDS` list constant to `app.py` with the default 10-term IELTS vocabulary set
- [ ] 1.4 Add `REPLICATE_DEFAULT_INITIAL_PROMPT` constant to `app.py` with the ESL/IELTS prompt string
- [ ] 1.5 Add `REPLICATE_INITIAL_PROMPT_PATH` constant pointing to `outputs/replicate_initial_prompt.txt`

## 2. Session State

- [ ] 2.1 Add `replicate_num_speakers` (default `2`) to `init_session_state()`
- [ ] 2.2 Add `replicate_language` (default `"en"`) to `init_session_state()`
- [ ] 2.3 Add `replicate_initial_prompt` (loads from disk, falls back to `REPLICATE_DEFAULT_INITIAL_PROMPT`) to `init_session_state()`
- [ ] 2.4 Add `deepgram_model` (default `DEEPGRAM_DEFAULT_MODEL`) to `init_session_state()`
- [ ] 2.5 Add `deepgram_language` (default `"en"`) to `init_session_state()`
- [ ] 2.6 Add `deepgram_keywords` (default `", ".join(DEEPGRAM_KEYWORDS)`) to `init_session_state()`

## 3. Replicate Transcription Pipeline

- [ ] 3.1 Add `load_persisted_replicate_prompt()` helper (reads `REPLICATE_INITIAL_PROMPT_PATH`, falls back to constant)
- [ ] 3.2 Add `persist_replicate_prompt()` helper (writes `st.session_state.replicate_initial_prompt` to disk)
- [ ] 3.3 Update `run_transcription()` to always pass `language=st.session_state.get("replicate_language", "en")` in the Replicate input payload
- [ ] 3.4 Update `run_transcription()` to always pass `word_timestamps=True` in the Replicate input payload
- [ ] 3.5 Update `run_transcription()` to pass `initial_prompt=st.session_state.get("replicate_initial_prompt", REPLICATE_DEFAULT_INITIAL_PROMPT)` in the payload
- [ ] 3.6 Update `run_transcription()` `num_speakers` logic: read from `st.session_state.get("replicate_num_speakers", 2)`; omit key entirely when value is `None`
- [ ] 3.7 Verify `convert_audio_path_for_analysis()` is called unconditionally before `replicate.run()` in `run_transcription()` (it already is; confirm in code and add inline comment)
- [ ] 3.8 Update `replicate_call_attempt` log event to include `language`, `word_timestamps`, `initial_prompt_chars` fields

## 4. Deepgram Streaming Pipeline

- [ ] 4.1 Add `model` parameter to `DeepgramStreamingClient.__init__()` (default `DEEPGRAM_DEFAULT_MODEL`); store as `self.model`
- [ ] 4.2 Update `_build_url()` to include `model=self.model` param
- [ ] 4.3 Update `_build_url()` to include `language=self.language` param (store `language` on `self`, default `"en"`)
- [ ] 4.4 Update `_build_url()` to include `filler_words=true` param
- [ ] 4.5 Update `_build_url()` to include `utterances=true` param
- [ ] 4.6 Update `_build_url()` to append `keywords=<term>` for each term in `self.keywords` list
- [ ] 4.7 Update `_on_open` / `deepgram_stream_config` log event to include `model`, `language`, `filler_words`, `utterances`, `keywords_count` fields
- [ ] 4.8 Update `start_deepgram_stream()` to read `model`, `language`, `keywords` from `st.session_state` when constructing `DeepgramStreamingClient`
- [ ] 4.9 Add `[BLANK_AUDIO]` stripping in `_format_deepgram_alt_with_speakers()` to remove Deepgram artefact tokens

## 5. UI — Replicate Settings Expander (Step 3)

- [ ] 5.1 Add `st.expander("Replicate Settings", expanded=False)` block in Step 3, after the "Transcribe via Replicate" button
- [ ] 5.2 Inside expander: add `st.selectbox` for language (`en` / `auto`), keyed `replicate_language`
- [ ] 5.3 Inside expander: add `st.selectbox` for number of speakers (`Auto`, `1`, `2`, `3`, `4`), mapped to `None` or int, keyed `replicate_num_speakers`
- [ ] 5.4 Inside expander: add `st.text_area` for initial prompt, keyed `replicate_initial_prompt`, with `on_change=persist_replicate_prompt`

## 6. UI — Deepgram Settings Expander (Step 3)

- [ ] 6.1 Add `st.expander("Deepgram Settings", expanded=False)` block in Step 3, after the Replicate Settings expander
- [ ] 6.2 Inside expander: add `st.selectbox` for model (`nova-3`, `nova-2`, `enhanced`, `base`), keyed `deepgram_model`
- [ ] 6.3 Inside expander: add `st.selectbox` for language (`en`, `auto`), keyed `deepgram_language`
- [ ] 6.4 Inside expander: add `st.text_area` for keywords (comma-separated), keyed `deepgram_keywords`, with help text explaining format

## 7. Debug Monitor

- [ ] 7.1 Add `deepgram_model` field to the `render_live_debug_monitor()` snapshot dict (read from `st.session_state.get("deepgram_model")` or streamer)
- [ ] 7.2 Add `replicate_model_id` field to the Advanced Diagnostics `st.write()` dict

## 8. Validation

- [ ] 8.1 Run `python -m py_compile app.py` — confirm no syntax errors
- [ ] 8.2 Start the app and record ~30 seconds of speech; confirm `deepgram_stream_config` log shows `model=nova-3`, `language=en`, `filler_words=True`
- [ ] 8.3 Click "Transcribe via Replicate"; confirm `replicate_call_attempt` log shows `language=en`, `word_timestamps=True`, `initial_prompt_chars > 0`
- [ ] 8.4 Verify Replicate transcript includes speaker labels and at least one filler word (`uh`/`um`) appears in the Deepgram transcript if present in audio
- [ ] 8.5 Verify the Replicate Settings expander and Deepgram Settings expander render correctly in Step 3 and that changes to values are reflected in subsequent transcription calls
