## ADDED Requirements

### Requirement: Upgrade to nova-3 model
The system SHALL pass `model=nova-3` in the Deepgram WebSocket URL parameters.
`nova-3` is Deepgram's highest-accuracy English model as of 2026 and significantly outperforms the legacy default on accented, non-native speech.

#### Scenario: nova-3 is present in the WebSocket URL
- **WHEN** `DeepgramStreamingClient._build_url()` is called
- **THEN** the URL contains `model=nova-3`

#### Scenario: Model name is logged at connection
- **WHEN** `_on_open` fires
- **THEN** the `deepgram_stream_config` event log entry includes `model="nova-3"`

---

### Requirement: Language forced to English
The system SHALL pass `language=en` in the Deepgram WebSocket URL.
Without this, Deepgram's language-detection overhead reduces accuracy on non-native speakers.

#### Scenario: Language parameter is present in the WebSocket URL
- **WHEN** `_build_url()` is called
- **THEN** the URL contains `language=en`

---

### Requirement: Filler words captured
The system SHALL pass `filler_words=true` in the Deepgram URL so that hesitation markers (`uh`, `um`, `er`) are transcribed rather than silently dropped.
These are linguistically significant for ESL fluency assessment.

#### Scenario: Filler words parameter is present
- **WHEN** `_build_url()` is called
- **THEN** the URL contains `filler_words=true`

---

### Requirement: Utterance detection enabled
The system SHALL pass `utterances=true` in the Deepgram URL.
Utterance boundaries improve sentence segmentation and downstream diarization quality.

#### Scenario: Utterances parameter is present
- **WHEN** `_build_url()` is called
- **THEN** the URL contains `utterances=true`

---

### Requirement: IELTS keyword hints
The system SHALL pass a `keywords` list of high-frequency IELTS topic vocabulary to Deepgram to boost recognition of domain-specific terms.
The default keyword list SHALL be stored as a module-level constant `DEEPGRAM_KEYWORDS` and be user-editable via the UI.

Default list (minimum viable set):
`["IELTS", "examiner", "candidate", "fluency", "coherence", "vocabulary", "grammar", "pronunciation", "band", "task"]`

#### Scenario: Keywords are appended to the WebSocket URL
- **WHEN** `_build_url()` is called and `DEEPGRAM_KEYWORDS` is non-empty
- **THEN** the URL contains one `keywords=<word>` parameter per keyword

#### Scenario: Empty keyword list produces no keyword params
- **WHEN** `DEEPGRAM_KEYWORDS` is empty
- **THEN** the URL contains no `keywords=` parameter

---

### Requirement: Deepgram config UI in Step 3
The system SHALL expose a collapsed expander "Deepgram Settings" in Step 3 of the UI containing:
- Model selector (`nova-3` default; options: `nova-3`, `nova-2`, `enhanced`, `base`)
- Language selector (`en` default; options: `en`, `auto`)
- Keywords textarea (comma-separated, pre-filled with default list)

#### Scenario: Settings are visible but collapsed by default
- **WHEN** the app renders Step 3
- **THEN** the "Deepgram Settings" expander is present and collapsed

#### Scenario: Selected model is used in next recording session
- **WHEN** a user selects a different model and starts a new recording
- **THEN** `DeepgramStreamingClient` is initialised with the new model value in the URL

---

### Requirement: Deepgram model surfaced in debug monitor
The system SHALL display the active Deepgram model name in the Live Debug Monitor snapshot so users can confirm which model is in use.

#### Scenario: Debug monitor shows active model
- **WHEN** the Live Debug Monitor is open
- **THEN** the snapshot includes a `deepgram_model` field showing the currently configured model name
