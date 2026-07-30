## ADDED Requirements

### Requirement: Pinned latest model version
The system SHALL use the latest pinned `thomasmol/whisper-diarization` model digest.
The digest SHALL be stored as a named constant (`REPLICATE_MODEL_ID`) at the top of `app.py` so upgrades require a single-line change.

#### Scenario: Model version is explicit and pinned
- **WHEN** the app runs `run_transcription()`
- **THEN** the Replicate call uses a fully-qualified `owner/model:digest` string, not a floating tag

---

### Requirement: Language forced to English
The system SHALL pass `language="en"` to the Replicate input payload so Whisper does not waste tokens on language auto-detection, which degrades accuracy on non-native English speakers.

#### Scenario: Language parameter is always set
- **WHEN** `run_transcription()` constructs the input payload
- **THEN** the payload contains `{"language": "en", ...}`

#### Scenario: Language parameter is sent to Replicate
- **WHEN** the Replicate call succeeds
- **THEN** the event log entry `replicate_call_attempt` includes `language="en"`

---

### Requirement: ESL domain initial prompt
The system SHALL send an `initial_prompt` string to Whisper that primes it for ESL/IELTS spoken English.
The default prompt SHALL be: `"IELTS speaking test. Two speakers: examiner and candidate. Spoken English, complete sentences."`.
The current value SHALL be readable from `st.session_state.replicate_initial_prompt` and persisted to disk alongside the OpenAI prompt.

#### Scenario: Default prompt is applied when none is configured
- **WHEN** a user has not customised the initial prompt
- **THEN** `run_transcription()` sends the default ESL prompt

#### Scenario: Custom prompt is applied when configured
- **WHEN** a user writes a custom value in the Replicate config expander and saves
- **THEN** `run_transcription()` sends the user-defined prompt

---

### Requirement: Configurable speaker count
The system SHALL expose a `num_speakers` setting (default `2`) that is passed to the Replicate call.
When set to `None` (auto), the parameter SHALL be omitted entirely from the payload.

#### Scenario: Two-speaker default is applied
- **WHEN** `num_speakers` is 2
- **THEN** the Replicate payload includes `{"num_speakers": 2, ...}`

#### Scenario: Auto-detect is applied when none is selected
- **WHEN** `num_speakers` is `None`
- **THEN** the Replicate payload does NOT include a `num_speakers` key

---

### Requirement: Word timestamps enabled
The system SHALL pass `word_timestamps=True` to the Replicate call.
Word-level timestamps improve downstream diarization alignment and enable future highlighting features.

#### Scenario: Word timestamps are included in payload
- **WHEN** `run_transcription()` builds the input payload
- **THEN** the payload contains `{"word_timestamps": True, ...}`

---

### Requirement: Audio normalised to 16 kHz mono before submission
The system SHALL ensure the WAV file passed to Replicate is 16 kHz mono before the call.
This is already performed by `convert_audio_path_for_analysis()`; this requirement formalises that it MUST be called unconditionally in `run_transcription()`.

#### Scenario: Audio is always converted before Replicate call
- **WHEN** `run_transcription()` is called with any valid audio path
- **THEN** `convert_audio_path_for_analysis()` is called and its output path is what is passed to `replicate.run()`

---

### Requirement: Replicate config UI in Step 3
The system SHALL expose a collapsed expander "Replicate Settings" in Step 3 of the UI containing:
- Language selector (`en` default, with at least `en` / `auto` options)
- Number of speakers selector (`2` default, `1`–`4` + Auto)
- Editable initial prompt textarea

#### Scenario: Settings are visible but collapsed by default
- **WHEN** the app renders Step 3
- **THEN** the "Replicate Settings" expander is present and collapsed

#### Scenario: Settings changes persist across reruns
- **WHEN** a user changes the initial prompt and the page reruns
- **THEN** the new value is preserved in session_state and disk
