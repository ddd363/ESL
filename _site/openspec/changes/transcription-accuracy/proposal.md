## Why

Both the Replicate (whisper-diarization) and Deepgram (live streaming) pipelines are producing inaccurate transcripts when processing ESL/IELTS speaker audio — missing words, mistranscribing non-native accents, hallucinating filler content, and using suboptimal default models.
Accuracy improvements are a prerequisite for reliable OpenAI feedback, since the analysis is only as good as the transcript it receives.

## What Changes

- **Replicate pipeline**: pin to the latest `thomasmol/whisper-diarization` model digest, set `language=en`, inject an ESL-domain initial prompt, enable `word_timestamps`, and set `num_speakers=2` as a configurable default (teacher + student).
- **Deepgram streaming pipeline**: upgrade from the implicit default model to `nova-3`, set `language=en`, add `filler_words=true`, add `utterances=true`, add `keywords` for common IELTS vocabulary, and surface model/tier in the debug panel.
- **Audio pre-processing**: enforce 16 kHz mono WAV normalisation before both pipelines (Replicate already does this via `convert_audio_path_for_analysis`; Deepgram streams raw PCM from the browser recorder which may vary).
- **Config surface**: expose Replicate `num_speakers`, `language`, and `initial_prompt` as editable fields in the Streamlit UI (collapsed expander, similar to the OpenAI prompt editor).
- **Config surface**: expose Deepgram `model` and `language` as selectable options in the UI.

## Capabilities

### New Capabilities

- `replicate-transcription`: Parameters, model version management, audio normalisation, and domain-prompt injection for the Replicate whisper-diarization call.
- `deepgram-streaming`: WebSocket connection parameters, model selection, language, keywords, and utterance settings for the Deepgram live streaming client.

### Modified Capabilities

<!-- No existing specs — both capabilities are new to specs/ -->

## Impact

- **app.py** — `run_transcription()` (Replicate call, lines ~1731–1825), `DeepgramStreamingClient._build_url()` (lines ~425–433), `run_transcription_background_job()` (lines ~1295–1331)
- **UI** — Step 3 "Run Analysis" section: add collapsible config panels for Replicate and Deepgram settings
- **session_state** — new keys: `replicate_num_speakers`, `replicate_language`, `replicate_initial_prompt`, `deepgram_model`, `deepgram_language`
- **No breaking changes to public API or file formats**
