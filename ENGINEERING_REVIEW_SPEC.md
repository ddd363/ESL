# Engineering Review Spec

## 1. Purpose

This document describes the application in its current repository state so another engineer can review, maintain, or extend it without reverse-engineering the codebase first.

The repository contains two related but independently deployable products:

1. A Streamlit speech-analysis app for ESL and IELTS speaking feedback.
2. A Jekyll marketing site with Decap CMS editing support and an optional Cloudflare OAuth worker for hosted CMS login.

These products share branding and content context, but they do not currently share a runtime, deployment target, or configuration system.

## 2. Product Scope

### 2.1 Streamlit app

The Streamlit app records or ingests spoken audio, transcribes it through cloud transcription providers, and generates structured teacher-facing feedback.

Primary use case:

- Capture speech from a learner or upload an existing recording.
- Produce diarized transcript output.
- Run grammar and fluency analysis.
- Present structured IELTS-style feedback.

### 2.2 Jekyll site

The static site is a coaching website with content pages for home, methodology, qualifications, and contact.

Primary use case:

- Market the coaching service.
- Allow non-technical content editing through Decap CMS.
- Publish to GitHub Pages under the `/ESL` base path.

### 2.3 Non-goals in current repository state

The repository does not currently implement:

- A unified deployed product that embeds the Streamlit app into the Jekyll site.
- A shared authentication layer between the speech-analysis app and the CMS site.
- A production-grade backend service for persistent users, jobs, or analytics.
- Automated tests for the major runtime paths.

## 3. Top-Level Architecture

### 3.1 System split

The repo is best understood as four implementation surfaces:

1. `app.py`: the Streamlit application and nearly all runtime logic for speech capture, transcription, analysis, and debugging.
2. `components/audio_recorder/index.html`: a custom browser recorder component used by the Streamlit app.
3. Jekyll site files (`index.html`, `contact.html`, `methodology.html`, `qualifications.html`, `_layouts`, `_includes`, `_data`, `assets/css/main.css`): the static marketing site.
4. `cloudflare-worker/worker.js`: an optional OAuth proxy used by hosted Decap CMS login.

### 3.2 Deployment split

The static site is configured for GitHub Pages via Jekyll. The speech-analysis app is not part of that build.

Evidence:

- `_config.yml` sets `baseurl: "/ESL"` and excludes `app.py`, `components`, `audio`, `outputs`, and `requirements.txt` from the Jekyll build.
- The Jekyll-generated output is committed under `_site/`, including a duplicated Cloudflare worker directory.

Operationally, this means a reviewer should treat the Streamlit app as a separate deployment target even though it lives in the same repo.

## 4. Streamlit App Specification

### 4.1 Primary responsibilities

The Streamlit app is responsible for:

- accepting audio from upload, saved local recordings, or live microphone capture,
- converting audio into a provider-compatible format,
- streaming live transcription through Deepgram,
- running batch diarized transcription through Replicate,
- performing OpenAI-based issue extraction,
- performing LanguageTool grammar checks,
- rendering transcripts, diagnostics, and feedback,
- persisting a small set of artifacts locally.

### 4.2 Runtime model

The app is a single-file Streamlit application with module-level globals and large `st.session_state` usage.

Important runtime characteristics:

- State is mostly held in `st.session_state`.
- Background jobs use `concurrent.futures.ThreadPoolExecutor`.
- Live Deepgram streaming uses a custom `websocket-client` integration and a background thread model.
- Event logging is append-only to `/tmp/esl_app_events.log`.
- Prompt persistence is filesystem-based under `outputs/`.

There is no database, message queue, or persistent service boundary.

### 4.3 User flows

#### Flow A: Upload audio

1. User selects `Upload`.
2. The app accepts `.wav` or `.m4a`.
3. `.m4a` is converted to mono 16 kHz WAV using `pydub`.
4. The resulting audio path is used for transcription.

#### Flow B: Load saved recording

1. User selects `Load`.
2. The app enumerates supported files under `audio/`.
3. The selected file is reused or converted for analysis.

#### Flow C: Record live audio

1. User selects `Record`.
2. The custom HTML component requests microphone access.
3. The component emits recorder events back to Streamlit.
4. The app starts a Deepgram streaming client when recording begins.
5. PCM16 chunks are sent from browser to Streamlit and then to Deepgram.
6. Live interim and final transcript state is rendered from Deepgram snapshots.
7. On stop, the app sends `CloseStream`, waits for final Deepgram frames, persists transcript state, and stores a WAV recording.

#### Flow D: Replicate transcription

1. The app prepares a WAV input.
2. `run_transcription()` sends the file to Replicate using a pinned `thomasmol/whisper-diarization` model digest.
3. The result is converted into a clean speaker-labeled transcript and a human-readable diarization view.
4. The transcript becomes available for analysis and display.

#### Flow E: Analysis and feedback

1. The app chooses transcript text.
2. `run_error_detection_background()` sends the transcript to OpenAI using raw HTTP requests.
3. The response is constrained to a JSON schema with an `issues` array.
4. If strict OpenAI mode is disabled or a non-auth failure occurs, a local rule-based fallback can be used.
5. Feedback is rendered as Markdown tables.

#### Flow F: LanguageTool annotation

1. A public LanguageTool API request is submitted asynchronously.
2. Matches are stored in session state.
3. The transcript is re-rendered with issue annotations and optional corrections.

### 4.4 External dependencies and integrations

The app currently integrates with four external services:

1. Deepgram WebSocket transcription API.
2. Replicate transcription/diarization API.
3. OpenAI chat completions API.
4. Public LanguageTool HTTP API.

Required environment variables:

- `DEEPGRAM_API_KEY`
- `REPLICATE_API_TOKEN`
- `OPENAI_API_KEY`

Optional runtime configuration:

- `WHISPER_MODEL_SIZE` for the local `faster_whisper` path.

### 4.5 Audio handling specification

Supported input extensions are `.wav` and `.m4a`.

Normalization rules:

- `.m4a` input is converted to `.wav`.
- conversion targets mono, 16 kHz for Whisper compatibility,
- live recording emits PCM16 chunks for Deepgram and a WAV payload for local retention.

Output storage:

- saved recordings go under `audio/`,
- transcripts and prompt artifacts go under `outputs/`,
- event logs go to `/tmp/esl_app_events.log`.

### 4.6 Session state model

`init_session_state()` establishes a large UI and workflow state surface.

Key categories:

- audio selection and recorder state,
- live Deepgram transcript state,
- batch Replicate transcript state,
- analysis and feedback state,
- background job futures and timers,
- LanguageTool annotation state,
- debugging toggles.

This is functionally adequate for a single-user Streamlit session, but a reviewer should note that business logic and UI state are tightly coupled.

### 4.7 Concurrency model

The app uses multiple concurrency mechanisms:

- `ThreadPoolExecutor(max_workers=4)` for background jobs,
- websocket background thread for Deepgram,
- keepalive thread for Deepgram,
- Streamlit reruns as the orchestration mechanism.

The design intention is non-blocking UX rather than strict isolation. This means reviewers should pay attention to race conditions between recorder events, reruns, and transcript persistence.

### 4.8 Error handling model

The app is defensive and log-heavy, but mostly not strongly typed.

Patterns used:

- service exceptions are often caught and turned into user-facing status strings,
- logging is done with `log_event(...)`,
- background jobs time out at `JOB_TIMEOUT_S = 120`,
- Deepgram health checks restart the streaming engine if disconnection persists,
- OpenAI auth failures can disable feedback entirely depending on `STRICT_OPENAI_FEEDBACK`.

### 4.9 Diagnostics and observability

The app includes a notable amount of operational diagnostics for a single-file Streamlit app.

Available diagnostics:

- structured event log lines in `/tmp/esl_app_events.log`,
- live debug monitor UI,
- transcript byte and event counters,
- latency capture for Deepgram and async jobs,
- environment presence snapshots for external API keys.

This suggests active debugging of transcription pipeline reliability has been a recent development priority.

### 4.10 Dormant or partially integrated paths

There are code paths that appear present but not central to the current main flow:

- `run_local_transcription()` provides a `faster_whisper` fallback but the main `run_transcription()` path hard-requires a Replicate token before proceeding.
- The `deepgram-sdk` and `openai` Python packages appear in `requirements.txt`, but `app.py` currently uses `websocket-client` and raw `requests` instead.
- `language-tool-python` appears in `requirements.txt`, while the app uses the public LanguageTool HTTP endpoint rather than the library.

These should be treated as implementation drift rather than active features until confirmed otherwise.

## 5. Custom Audio Recorder Component Specification

### 5.1 Responsibilities

The custom recorder component is a plain HTML and JavaScript page mounted into Streamlit as a custom component.

It is responsible for:

- requesting microphone permissions,
- enumerating microphones,
- recording browser audio,
- drawing a waveform,
- emitting recorder lifecycle events,
- chunking live PCM16 audio for streaming,
- returning a WAV payload at stop.

### 5.2 Interface contract with Streamlit

The component emits structured events with fields such as:

- `event`
- `t`
- `seq`
- `sample_rate`
- `pcm16_base64`
- `wav_base64`

Notable events include:

- `start_clicked`
- `recording_started`
- `recording_chunk`
- microphone enumeration events
- microphone error events

### 5.3 Recording strategy

The component uses:

- `getUserMedia()` for microphone access,
- `AudioContext`,
- `ScriptProcessorNode`,
- a half-second live chunk interval for Deepgram chunk emission.

Reviewer note: `ScriptProcessorNode` is legacy Web Audio API surface. It still works broadly, but an engineer reviewing for longevity or browser-performance correctness should consider migration to `AudioWorklet` in a future change.

## 6. Static Site Specification

### 6.1 Responsibilities

The Jekyll site is responsible for branded public content, not application execution.

Pages:

- home,
- qualifications,
- methodology,
- contact.

### 6.2 Rendering model

The site uses:

- page front matter,
- a single default layout,
- header and footer includes,
- YAML-backed content in `_data/home.yml`,
- a single global stylesheet in `assets/css/main.css`.

### 6.3 Content model

The home data file currently contains:

- brand and contact metadata,
- hero copy,
- authority cards,
- video embeds,
- method section,
- work section,
- process steps,
- testimonials,
- qualifications page content,
- methodology page content.

The site is therefore data-driven in content, but not in site structure.

### 6.4 Visual implementation

The CSS defines a teal and dark-blue brand system with:

- sticky header,
- hero card layout,
- card grids,
- responsive mobile navigation,
- light and contrast section variants.

There is no build pipeline beyond Jekyll itself; styling is handwritten CSS.

## 7. Decap CMS Specification

### 7.1 Purpose

The repository includes Decap CMS so non-technical edits can be made through `/admin`.

Current admin surface:

- `admin/index.html` loads Decap CMS from CDN.
- `admin/config.yml` points to GitHub backend and enables `local_backend: true`.

### 7.2 Local editing workflow

The intended local workflow described in `README.md` is:

1. Serve the static files locally.
2. Run `npx decap-server`.
3. Open `/admin/`.
4. Edit `_data/home.yml` through Decap.

### 7.3 Hosted editing workflow

The hosted workflow is optional and depends on the Cloudflare worker as an OAuth helper.

## 8. Cloudflare Worker Specification

### 8.1 Responsibilities

The worker implements a minimal GitHub OAuth relay for Decap CMS.

Endpoints:

- `/auth`: redirects to GitHub OAuth authorize.
- `/callback`: exchanges the code for an access token and posts it back to the opener window.

### 8.2 Security and operational constraints

The worker is intentionally minimal and currently assumes deployment-time secret configuration.

Important characteristics:

- `CLIENT_ID` and `CLIENT_SECRET` are placeholders in source.
- returned tokens are passed via `window.opener.postMessage(..., "*")`,
- the generated `state` value is not persisted or validated,
- there is no origin restriction on the callback postMessage,
- there is no explicit repo-level authorization logic.

This is sufficient for a basic CMS login bridge, but it is not hardened.

## 9. Configuration and Secrets

### 9.1 Runtime secrets

Speech-analysis runtime secrets:

- `DEEPGRAM_API_KEY`
- `REPLICATE_API_TOKEN`
- `OPENAI_API_KEY`

CMS/OAuth runtime secrets:

- GitHub OAuth client ID
- GitHub OAuth client secret

### 9.2 Repository configuration

Relevant configuration files:

- `_config.yml` for Jekyll site build behavior,
- `admin/config.yml` for Decap CMS,
- `requirements.txt` for Python runtime packages,
- `Gemfile` and vendored bundle for Jekyll dependencies.

## 10. Data and Artifact Model

### 10.1 Persistent repository data

- `_data/home.yml`: public site content source.
- `audio/`: saved recordings.
- `outputs/transcript.txt`, `outputs/transcript_raw.json`, `outputs/segments.csv`: transcription artifacts.

### 10.2 Local ephemeral data

- `/tmp/esl_app_events.log`: app event log.
- Streamlit session state in-process memory.

### 10.3 Generated site output

- `_site/` contains built static artifacts committed into the repository.

Because `_site/` duplicates source-derived content, reviewers should confirm whether committed build output is intentional or a convenience artifact.

## 11. Deployment Model

### 11.1 Static site

Expected deployment target:

- GitHub Pages project site under `https://ddd363.github.io/ESL`.

Build assumptions:

- Jekyll renders from source files.
- `baseurl` must remain `/ESL` for the Pages path unless hosting changes.

### 11.2 Streamlit app

No explicit deployment manifest for the Streamlit app is present in the repository.

A reviewer should assume one of these is currently true:

1. The app is run locally or ad hoc for development/demo use.
2. The deployment configuration lives outside this repository.

The lack of deployment documentation for the app is itself an engineering review point.

## 12. Review Findings and Known Risks

### 12.1 CMS schema drift

`admin/config.yml` does not fully describe the content shape used by the templates and `_data/home.yml`.

Examples of fields present in templates or data but not clearly modeled in the CMS config:

- `hero.positioning_line`
- icon fields on multiple list items
- `method` section on the homepage
- `testimonials`

Impact:

- CMS edits may omit important content fields,
- a non-technical editor may not be able to manage all rendered content,
- future edits risk partial overwrite or content drift.

### 12.2 Product boundary ambiguity

The repo README primarily documents the Jekyll site and CMS workflow, while `app.py` contains a large, separate product.

Impact:

- onboarding cost is higher,
- deployment ownership is unclear,
- another engineer may incorrectly assume the speech-analysis app is part of the Pages build.

### 12.3 Single-file application complexity

`app.py` contains UI composition, provider integrations, concurrency, error handling, persistence, and diagnostics in one module.

Impact:

- difficult targeted testing,
- high cognitive load,
- greater risk of regression during feature work.

### 12.4 Security hardening gaps in OAuth worker

The worker lacks validated `state`, origin restriction, and more restrictive token handoff.

Impact:

- acceptable for low-friction internal use,
- weak for hardened public deployment.

### 12.5 Dependency drift

`requirements.txt` includes packages that do not match the primary code paths.

Impact:

- larger than necessary environment,
- ambiguity about supported fallback modes,
- harder maintenance when dependency updates are needed.

### 12.6 Limited automated verification

No test suite or explicit CI verification for the speech-analysis app was found in the repository surfaces reviewed.

Impact:

- runtime regressions will primarily be caught manually,
- third-party API changes pose outsized risk.

### 12.7 Committed build output

`_site/` duplicates source content and includes built artifacts.

Impact:

- larger diffs,
- easier accidental drift between source and generated output,
- review noise.

## 13. In-Flight Change Context

The repository contains an active OpenSpec change under `openspec/changes/transcription-accuracy/`.

That change is focused on improving transcription quality by tuning Replicate and Deepgram parameters rather than changing the architecture itself.

Reviewer implication:

- current transcription behavior is under active refinement,
- any review of transcript quality should account for this pending work,
- architecture observations in this document describe current implementation, not the final target state of that change.

## 14. Recommended Review Checklist

Another engineer reviewing this app should evaluate the system in this order:

1. Confirm whether the Streamlit app is an actively deployed product or a development tool.
2. Validate end-to-end audio flows: upload, record, load saved recording.
3. Validate external API contract assumptions for Deepgram, Replicate, OpenAI, and LanguageTool.
4. Review `st.session_state` and background-thread interactions for race conditions.
5. Decide whether `app.py` should remain monolithic or be split into modules.
6. Reconcile `admin/config.yml` with `_data/home.yml` and the templates.
7. Decide whether `_site/` should remain versioned.
8. Harden the Cloudflare OAuth worker if hosted CMS editing is a production requirement.

## 15. Suggested Near-Term Engineering Work

1. Add a dedicated architecture section to `README.md` that separates the Streamlit app from the Jekyll site.
2. Align `admin/config.yml` with the actual rendered data schema.
3. Extract provider clients and transcript-processing logic from `app.py` into modules.
4. Add a smoke-test path for non-UI logic such as transcript formatting and fallback feedback generation.
5. Clarify deployment instructions for the Streamlit app.

## 16. Source Files Reviewed

This spec was derived from the current implementation in these primary files:

- `app.py`
- `components/audio_recorder/index.html`
- `README.md`
- `_config.yml`
- `requirements.txt`
- `index.html`
- `contact.html`
- `methodology.html`
- `qualifications.html`
- `_layouts/default.html`
- `_includes/header.html`
- `_includes/footer.html`
- `_data/home.yml`
- `assets/css/main.css`
- `admin/index.html`
- `admin/config.yml`
- `cloudflare-worker/worker.js`
- `cloudflare-worker/README.md`
- `openspec/changes/transcription-accuracy/design.md`
- `openspec/changes/transcription-accuracy/tasks.md`
