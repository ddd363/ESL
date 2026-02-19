import base64
import html
import hashlib
import io
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

import replicate
import requests
import streamlit as st
import streamlit.components.v1 as components
from pydub import AudioSegment


OPENAI_MODEL = "gpt-5.2"
STRICT_OPENAI_FEEDBACK = True
AUDIO_DIR = "audio"
APP_EVENT_LOG = "/tmp/esl_app_events.log"
SUPPORTED_EXTS = {".wav", ".m4a"}
RECORDER_COMPONENT_DIR = os.path.join(
    os.path.dirname(__file__), "components", "audio_recorder"
)
audio_recorder_component = components.declare_component(
    "audio_recorder", path=RECORDER_COMPONENT_DIR
)


def log_event(event, **fields):
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    try:
        with open(APP_EVENT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def read_recent_events(limit=20):
    if not os.path.exists(APP_EVENT_LOG):
        return []
    try:
        with open(APP_EVENT_LOG, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return [line.strip() for line in lines[-limit:] if line.strip()]
    except Exception:
        return []


def masked_key_prefix(value):
    if not value:
        return "<missing>"
    return f"{value[:8]}***"


def extract_status_code_from_text(message):
    match = re.search(r"status:\s*(\d{3})", message or "")
    if match:
        return int(match.group(1))
    return None


def show_diagnostic_banner():
    st.info("API call attempted — see logs for details.")

# Cell 0c: Replicate diarization (Whisper + diarization as a service)
# Set REPLICATE_API_TOKEN in your environment before running.

# JSON Schema for the model response (strict mode requires additionalProperties=false and required to include all keys in properties)
schema = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                },
                "required": ["message"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["issues"],
    "additionalProperties": False,
}


def init_session_state():
    if "audio_source" not in st.session_state:
        st.session_state.audio_source = "Upload"
    if "is_recording" not in st.session_state:
        st.session_state.is_recording = False
    if "recorded_audio_bytes" not in st.session_state:
        st.session_state.recorded_audio_bytes = None
    if "recorded_audio_path" not in st.session_state:
        st.session_state.recorded_audio_path = None
    if "recorder_nonce" not in st.session_state:
        st.session_state.recorder_nonce = 0
    if "last_recorded_audio_digest" not in st.session_state:
        st.session_state.last_recorded_audio_digest = None
    if "loaded_audio_path" not in st.session_state:
        st.session_state.loaded_audio_path = None
    if "allow_rule_based_fallback_on_openai_failure" not in st.session_state:
        st.session_state.allow_rule_based_fallback_on_openai_failure = False
    if "audio_ready" not in st.session_state:
        st.session_state.audio_ready = False
    if "transcript" not in st.session_state:
        st.session_state.transcript = ""
    if "feedback" not in st.session_state:
        st.session_state.feedback = []
    if "diarization" not in st.session_state:
        st.session_state.diarization = ""
    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False
    if "transcript_source" not in st.session_state:
        st.session_state.transcript_source = "unknown"
    if "feedback_source" not in st.session_state:
        st.session_state.feedback_source = "unknown"
    if "recording_started_at" not in st.session_state:
        st.session_state.recording_started_at = None
    if "active_audio_signature" not in st.session_state:
        st.session_state.active_audio_signature = None


def render_audio_recorder():
    return audio_recorder_component(key=f"audio_recorder_{st.session_state.recorder_nonce}")


def recording_ready():
    # Single source of truth for recorded audio readiness.
    return bool(st.session_state.get("recorded_audio_bytes")) or bool(
        st.session_state.get("recorded_audio_path")
    )


def get_recorded_audio_path():
    recorded_path = st.session_state.get("recorded_audio_path")
    if recorded_path and os.path.exists(recorded_path):
        return recorded_path
    return None


def clear_recording_state(delete_file=False):
    recorded_path = st.session_state.get("recorded_audio_path")
    st.session_state.recorded_audio_bytes = None
    st.session_state.recorded_audio_path = None
    st.session_state.is_recording = False
    st.session_state.last_recorded_audio_digest = None
    st.session_state.recorder_nonce += 1
    log_event(
        "clear_recording_state",
        delete_file=delete_file,
        recorded_path=recorded_path,
        recorder_nonce=st.session_state.recorder_nonce,
    )

    if delete_file:
        candidates = [recorded_path]
        for path in candidates:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


def reset_analysis_state():
    st.session_state.transcript = ""
    st.session_state.feedback = []
    st.session_state.diarization = ""
    st.session_state.analysis_complete = False
    st.session_state.transcript_source = "unknown"
    st.session_state.feedback_source = "unknown"


def reset_for_rerecord():
    clear_recording_state(delete_file=False)
    reset_analysis_state()
    st.session_state.audio_ready = False
    st.session_state.active_audio_signature = None


def list_saved_recordings():
    if not os.path.isdir(AUDIO_DIR):
        return []
    candidates = []
    for filename in os.listdir(AUDIO_DIR):
        path = os.path.join(AUDIO_DIR, filename)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(filename)[1].lower() not in SUPPORTED_EXTS:
            continue
        candidates.append(path)
    return sorted(candidates, key=os.path.getmtime, reverse=True)


def build_timestamped_recording_path():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    candidate_dt = datetime.now()
    while True:
        filename = f"recording_{candidate_dt.strftime('%Y%m%d_%H%M%S')}.wav"
        recorded_path = os.path.join(AUDIO_DIR, filename)
        if not os.path.exists(recorded_path):
            return recorded_path
        candidate_dt = candidate_dt + timedelta(seconds=1)


def convert_audio_path_for_analysis(input_path):
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".m4a":
        audio_path = os.path.splitext(input_path)[0] + ".wav"
        audio = AudioSegment.from_file(input_path, format="m4a")
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(audio_path, format="wav")
        return audio_path
    if ext == ".wav":
        return input_path
    raise ValueError("Unsupported file type. Use .wav or .m4a.")


def load_and_convert_audio(file_bytes, filename):
    # Cell 1: Load most recent audio file from ./audio (.wav or .m4a)
    # Prefer in-memory upload if present
    if file_bytes and filename:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTS:
            raise ValueError("Unsupported file type. Use .wav or .m4a.")

        os.makedirs(AUDIO_DIR, exist_ok=True)
        if ext == ".m4a":
            audio_path = os.path.join(AUDIO_DIR, os.path.splitext(filename)[0] + ".wav")
            audio = AudioSegment.from_file(io.BytesIO(file_bytes), format="m4a")
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(audio_path, format="wav")
        else:
            audio_path = os.path.join(AUDIO_DIR, filename)
            with open(audio_path, "wb") as f:
                f.write(file_bytes)

        st.write(f"Audio loaded from upload: {audio_path}")
        return audio_path

    if not os.path.isdir(AUDIO_DIR):
        raise FileNotFoundError(f"Directory not found: {AUDIO_DIR}")

    candidates = [
        os.path.join(AUDIO_DIR, f)
        for f in os.listdir(AUDIO_DIR)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
        and os.path.isfile(os.path.join(AUDIO_DIR, f))
    ]

    if not candidates:
        raise FileNotFoundError(
            "No .wav or .m4a files found in ./audio. Add a file and try again."
        )

    input_path = max(candidates, key=os.path.getmtime)

    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".m4a":
        audio_path = os.path.splitext(input_path)[0] + ".wav"
        audio = AudioSegment.from_file(input_path, format="m4a")
        # Convert to mono/16k for best Whisper results
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(audio_path, format="wav")
    elif ext == ".wav":
        audio_path = input_path
    else:
        raise ValueError("Unsupported file type. Use .wav or .m4a.")

    st.write(f"Audio loaded: {audio_path}")
    return audio_path


def get_active_audio(uploaded_file):
    source = st.session_state.audio_source
    log_event("get_active_audio", source=source, recording_ready=recording_ready())
    if source == "Upload":
        if not uploaded_file:
            log_event("upload_missing_file")
            st.error("Please upload a .wav or .m4a file first.")
            st.stop()
        return load_and_convert_audio(uploaded_file.getvalue(), uploaded_file.name)
    if source == "Record":
        if not recording_ready():
            log_event("record_missing_audio")
            st.error("Please record audio first.")
            st.stop()
        recorded_path = get_recorded_audio_path()
        log_event("record_audio_selected", recorded_path=recorded_path)
        st.write(f"Audio loaded from recording: {recorded_path}")
        return recorded_path
    if source == "Load":
        selected_path = st.session_state.get("loaded_audio_path")
        if not selected_path:
            log_event("load_missing_file")
            st.error("Please select a saved recording first.")
            st.stop()
        if not os.path.exists(selected_path):
            log_event("load_missing_path", selected_path=selected_path)
            st.error("Selected saved recording was not found. Please select another file.")
            st.stop()
        audio_path = convert_audio_path_for_analysis(selected_path)
        log_event("load_audio_selected", selected_path=selected_path, audio_path=audio_path)
        st.write(f"Audio loaded from saved file: {audio_path}")
        return audio_path
    log_event("unknown_audio_source", source=source)
    st.error("Unknown audio source.")
    st.stop()


def _format_time(sec):
    m = int(sec // 60)
    s = sec % 60
    return f"{m:02d}:{s:05.2f}"


def _normalize_text(t):
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"[\W_]+", "", t)
    return t


def _dedupe_sentences(text):
    # Remove consecutive duplicate sentences after splitting on punctuation
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out = []
    last_norm = None
    for p in parts:
        if not p:
            continue
        norm = _normalize_text(p)
        if norm and norm == last_norm:
            continue
        out.append(p)
        last_norm = norm
    return " ".join(out)


def _merge_segments(segments):
    merged = []
    last_text_norm = None
    for seg in segments:
        speaker = seg.get("speaker") or "UNKNOWN"
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        # If speaker is UNKNOWN, stick with previous speaker when possible
        if speaker == "UNKNOWN" and merged:
            speaker = merged[-1]["speaker"]
        text = _dedupe_sentences(text)
        text_norm = _normalize_text(text)
        if text_norm and text_norm == last_text_norm:
            # Skip exact repeat chunk
            continue
        if merged and merged[-1]["speaker"] == speaker:
            # Merge consecutive same-speaker chunks
            merged[-1]["end"] = end
            merged[-1]["text"] += " " + text
        else:
            merged.append({"speaker": speaker, "start": start, "end": end, "text": text})
        last_text_norm = text_norm
    return merged


def _pretty_print_replicate(output):
    if not output:
        return "No Replicate output available. Run the Replicate diarization cell first."
    # Replicate returns a dict with `segments` or a list in some cases
    segments = None
    if isinstance(output, dict):
        segments = output.get("segments")
    elif isinstance(output, list):
        segments = output
    if not segments:
        return "No segments found in Replicate output."
    merged = _merge_segments(segments)
    if not merged:
        return "No usable segments after merging."
    lines = ["HUMAN-READABLE DIARIZATION:\n"]
    last_speaker = None
    for seg in merged:
        speaker = seg["speaker"]
        start = _format_time(seg.get("start", 0.0))
        end = _format_time(seg.get("end", 0.0))
        text = seg["text"]
        if speaker != last_speaker:
            lines.append(f"[{speaker}] {start}–{end}: {text}")
            last_speaker = speaker
        else:
            lines.append(text)
    return "\n".join(lines)


def run_local_transcription(wav_path):
    from faster_whisper import WhisperModel

    model_size = os.getenv("WHISPER_MODEL_SIZE", "tiny")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments_iter, _ = model.transcribe(wav_path, vad_filter=True)

    transcript_parts = []
    diarization_lines = ["HUMAN-READABLE DIARIZATION:\n"]
    fallback_segments = []

    for seg in segments_iter:
        text = (seg.text or "").strip()
        if not text:
            continue
        transcript_parts.append(text)
        diarization_lines.append(
            f"[SPEAKER_00] {_format_time(seg.start)}–{_format_time(seg.end)}: {text}"
        )
        fallback_segments.append(
            {
                "speaker": "SPEAKER_00",
                "start": float(seg.start),
                "end": float(seg.end),
                "text": text,
            }
        )

    transcript = " ".join(transcript_parts).strip()
    diarization_text = "\n".join(diarization_lines).strip()

    if not transcript:
        raise RuntimeError("Local transcription produced no text.")

    return {"segments": fallback_segments}, transcript, diarization_text


def run_transcription(wav_path):
    import os

    # Ensure audio is prepared (run Cell 1 first to set AUDIO_PATH)
    if not wav_path:
        raise RuntimeError("AUDIO_PATH not set. Run the audio load cell first.")

    replicate_token = os.environ.get("REPLICATE_API_TOKEN")
    log_event(
        "replicate_env_snapshot",
        python_executable=sys.executable,
        conda_env=os.environ.get("CONDA_DEFAULT_ENV"),
        replicate_key_present=bool(replicate_token),
        replicate_key_prefix=masked_key_prefix(replicate_token),
    )
    if not replicate_token:
        raise RuntimeError("REPLICATE_API_TOKEN not set. Add it in this cell and re-run.")

    # Replicate diarization settings
    num_speakers = 2  # set to None to autodetect
    group_segments = True  # merge short same-speaker segments

    # Run diarization on Replicate (pin to a model version)
    model_id = "thomasmol/whisper-diarization:1495a9cddc83b2203b0d8d3516e38b80fd1572ebc4bc5700ac1da56a9b3ed886"
    try:
        log_event(
            "replicate_call_attempt",
            model_id=model_id,
            output="json",
            group_segments=group_segments,
            num_speakers=num_speakers,
            endpoint="replicate.run(model_id, input=...)",
        )
        with open(wav_path, "rb") as f:
            input_payload = {
                "file": f,
                "output": "json",
                "group_segments": group_segments,
            }
            if num_speakers:
                input_payload["num_speakers"] = num_speakers
            replicate_output = replicate.run(
                model_id,
                input=input_payload,
            )
        log_event(
            "replicate_call_result",
            inferred_http_status=200,
            response_type=type(replicate_output).__name__,
            response=replicate_output,
        )
    except Exception as e:
        message = str(e)
        status_code = extract_status_code_from_text(message)
        log_event(
            "replicate_call_error",
            status_code=status_code,
            error=message,
            error_type=type(e).__name__,
        )
        if "Invalid token" in message or "status: 401" in message:
            raise RuntimeError(
                "Replicate authentication failed. Set a valid REPLICATE_API_TOKEN in the environment."
            ) from e
        raise RuntimeError(f"Replicate diarization failed: {message}") from e

    transcript = _extract_transcript_from_replicate(replicate_output)
    diarization_text = _pretty_print_replicate(replicate_output)
    return replicate_output, transcript, diarization_text


def _extract_transcript_from_replicate(output):
    if isinstance(output, dict):
        if "text" in output and isinstance(output["text"], str):
            return output["text"].strip()
        segments = output.get("segments")
        if isinstance(segments, list):
            parts = []
            for seg in segments:
                if isinstance(seg, dict):
                    t = (seg.get("text") or "").strip()
                    if t:
                        parts.append(t)
            if parts:
                return " ".join(parts)
    return None


def get_issue_type(match):
    if isinstance(match, dict):
        return match.get("type", "UNKNOWN")
    if hasattr(match, "ruleId"):
        return match.ruleId
    if hasattr(match, "rule_id"):
        return match.rule_id
    if hasattr(match, "rule"):
        rule = getattr(match, "rule", None)
        if isinstance(rule, dict):
            return rule.get("id", "UNKNOWN")
        rule_id = getattr(rule, "id", None)
        if rule_id:
            return rule_id
    return "UNKNOWN"


def _md_escape(text):
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _truncate_words(text, max_words=18):
    words = (text or "").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + " ..."


def build_local_feedback_issues(transcript):
    text = (transcript or "").strip()
    if not text:
        return []

    lower = text.lower()
    rows = []

    if re.search(r"\b\w+\s+\w+\s+\w+\s+\w+\s+\w+\s+\w+\s+\w+\s+\w+\s+\w+\s+\w+", text) and not re.search(r"[.!?]", text):
        phrase = _truncate_words(text)
        rows.append(
            (
                "Coherence / sentence boundaries",
                f'"{_md_escape(phrase)}"; context: "{_md_escape(phrase)}"',
                "Split ideas into shorter clauses with punctuation.",
                "Long run-on delivery can reduce coherence and listener processing.",
            )
        )

    repeated = re.search(r"\b(\w+)\s+\1\b", lower)
    if repeated:
        phrase = repeated.group(0)
        rows.append(
            (
                "Repetition / fluency",
                f'"{_md_escape(phrase)}"; context: "{_md_escape(_truncate_words(text))}"',
                "Use one instance of the repeated word unless emphasis is intended.",
                "Unnecessary repetition can reduce fluency and clarity.",
            )
        )

    fillers = re.findall(r"\b(um+|uh+|you know|like)\b", lower)
    if fillers:
        phrase = ", ".join(sorted(set(fillers)))
        rows.append(
            (
                "Fillers",
                f'"{_md_escape(phrase)}"; context: "{_md_escape(_truncate_words(text))}"',
                "Pause briefly instead of using filler words.",
                "Frequent fillers can lower perceived fluency.",
            )
        )

    if not rows:
        phrase = _truncate_words(text)
        rows.append(
            (
                "Grammar and phrasing (local fallback)",
                f'"{_md_escape(phrase)}"; context: "{_md_escape(phrase)}"',
                "Refine verb forms and sentence structure for clearer meaning.",
                "Fallback mode is conservative; review this excerpt for high-impact IELTS issues.",
            )
        )

    lines = [
        "Title: Rule-Based Feedback Table (Fallback, Non-AI)",
        "",
        "| Error Group | Student Examples | Better Versions | Explanation |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {_md_escape(row[0])} | {_md_escape(row[1])} | {_md_escape(row[2])} | {_md_escape(row[3])} |")

    return [{"message": "\n".join(lines)}]


def run_error_detection(transcript, emit_ui=True):
    # 2) Call OpenAI (only if we have transcript + key)
    matches = []
    if not transcript:
        if emit_ui:
            st.text(
                "Transcript not available yet. Run the diarization/transcription cell(s) first, then re-run this cell."
            )
        return matches

    api_key = os.environ.get("OPENAI_API_KEY")
    allow_rule_based_fallback = bool(
        st.session_state.get("allow_rule_based_fallback_on_openai_failure", False)
    )
    log_event(
        "openai_env_snapshot",
        python_executable=sys.executable,
        conda_env=os.environ.get("CONDA_DEFAULT_ENV"),
        openai_key_present=bool(api_key),
        openai_key_prefix=masked_key_prefix(api_key),
        strict_openai_feedback=STRICT_OPENAI_FEEDBACK,
        allow_rule_based_fallback=allow_rule_based_fallback,
        key_matches_os_getenv=(api_key == os.getenv("OPENAI_API_KEY")),
    )
    if not api_key:
        st.session_state["feedback_source"] = "openai_unavailable"
        if emit_ui:
            st.error(
                "Feedback disabled: OpenAI authentication failed (invalid or missing API key)."
            )
        log_event(
            "openai_feedback_unavailable",
            failure_type="auth_missing_key",
            status_code=None,
            strict_openai_feedback=STRICT_OPENAI_FEEDBACK,
            allow_rule_based_fallback=allow_rule_based_fallback,
        )
        if STRICT_OPENAI_FEEDBACK and not allow_rule_based_fallback:
            return []
        st.session_state["feedback_source"] = "local_rule_based_fallback"
        return build_local_feedback_issues(transcript)

    system_msg = """
You are an ESL grammar and fluency analyst focused on IELTS Speaking assessment.

Input: a transcript with multiple speakers labeled [SPEAKER_00], [SPEAKER_01], etc.

TASK
1) Identify the second speaker in order of first appearance.
2) Analyze ONLY that speaker’s speech.
3) Produce a teacher-facing feedback TABLE that groups common errors and prioritizes them by IELTS impact.

NON-NEGOTIABLE RULES
- Use ONLY errors that explicitly appear in the transcript.
- Quote the student’s exact words for every example.
- Do NOT invent, generalize, or paraphrase student language.
- Group similar errors together under a clear error category.
- Explanations must reflect the REAL grammatical or lexical issue.
- Do NOT mention grammar forms that are not used in the correction.
- Prioritize reoccuring errors that most affect IELTS bands (Coherence, Lexical Resource, Grammar).
- DO NOT prioritise fillers or repetition


OUTPUT FORMAT (STRICT TABLE)

Title: On-the-Fly Feedback Table (Speaker 00)

Table columns (exactly these, in this order):
1) Error Group (sorted from highest to lowest IELTS impact)
2) Student Examples (exact phrases from transcript)
3) Better Versions
4) Explanation (clear, teacher-ready, 1–2 short lines)

STUDENT EXAMPLES COLUMN (IMPORTANT)
- For each row, include:
  (a) the exact problematic phrase in quotes, AND
  (b) a short verbatim context snippet
- Both must be exact transcript text (no cleanup, no paraphrase).

SORTING RULE
- Order rows by highest IELTS impact first:
  1) Coherence / clause structure / logic
  2) Collocation and word choice
  3) Fixed phrases / prepositions
  4) Verb forms and agreement
  5) Pronouns, fillers, repetition

STYLE CONSTRAINTS
- Concise but clear explanations
- No paragraphs outside the table
- No teaching activities or advice
- Teacher-facing language suitable for quick explanation in class
        """
    user_msg = f"Transcript:\n{transcript}"

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "esl_issues", "schema": schema, "strict": True},
        },
    }

    r = None
    try:
        endpoint = "https://api.openai.com/v1/chat/completions"
        log_event(
            "openai_call_attempt",
            model=OPENAI_MODEL,
            endpoint=endpoint,
            payload={
                "model": payload.get("model"),
                "response_format": payload.get("response_format"),
            },
        )
        r = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        log_event("openai_call_http_status", status_code=r.status_code)
        r.raise_for_status()
        resp = r.json()
        log_event("openai_call_response_json", response=resp)

        if "choices" in resp and len(resp["choices"]) > 0:
            content = resp["choices"][0].get("message", {}).get("content")
            if isinstance(content, str):
                content_str = content
            else:
                content_str = json.dumps(content, ensure_ascii=False)
            data = json.loads(content_str) if isinstance(content_str, str) else {}
            if isinstance(data, dict):
                matches = data.get("issues", []) or []
                st.session_state["feedback_source"] = "openai"
            else:
                matches = []
        else:
            raise ValueError(
                f"Unexpected response format from OpenAI: {json.dumps(resp, indent=2)}"
            )
    except requests.HTTPError as e:
        status_code = getattr(e.response, "status_code", None)
        is_auth_failure = status_code in {401, 403}
        log_event(
            "openai_call_http_error",
            status_code=status_code,
            failure_type="auth_http" if is_auth_failure else "http_error",
            error=str(e),
            endpoint="https://api.openai.com/v1/chat/completions",
            model=OPENAI_MODEL,
        )
        if is_auth_failure:
            st.session_state["feedback_source"] = "openai_unavailable"
            if emit_ui:
                st.error(
                    "Feedback disabled: OpenAI authentication failed (invalid or missing API key)."
                )
            if STRICT_OPENAI_FEEDBACK and not allow_rule_based_fallback:
                return []
            st.session_state["feedback_source"] = "local_rule_based_fallback"
            return build_local_feedback_issues(transcript)
        if emit_ui:
            st.warning("Feedback may be degraded: temporary API failure; using rule-based fallback.")
            st.text("OpenAI request failed:")
            st.text(str(e))
        try:
            if emit_ui:
                st.text(getattr(r, "text", ""))
            log_event("openai_call_raw_text", raw_text=getattr(r, "text", ""))
        except Exception:
            pass
        matches = build_local_feedback_issues(transcript)
        st.session_state["feedback_source"] = "local_rule_based_fallback"
    except Exception as e:
        log_event(
            "openai_call_exception",
            failure_type="temporary_api_failure",
            error=str(e),
            error_type=type(e).__name__,
        )
        if emit_ui:
            st.warning("Feedback may be degraded: temporary API failure; using rule-based fallback.")
            st.text("OpenAI request failed:")
            st.text(str(e))
        try:
            if emit_ui:
                st.text(getattr(r, "text", ""))
            log_event("openai_call_raw_text", raw_text=getattr(r, "text", ""))
        except Exception:
            pass
        matches = build_local_feedback_issues(transcript)
        st.session_state["feedback_source"] = "local_rule_based_fallback"

    return matches


def render_feedback_table(issues):
    # Expected shape: list[dict] with a 'message' field containing Markdown.
    if isinstance(issues, list) and issues and isinstance(issues[0], dict):
        parts = []
        for item in issues:
            if not isinstance(item, dict):
                continue
            md = (item.get("message") or "").strip()
            if md:
                parts.append(md)
        md_all = "\n\n".join(parts).strip()
        if md_all:
            if "<table" in md_all.lower():
                components.html(md_all, scrolling=True)
            else:
                st.markdown(md_all)
        else:
            st.text("No markdown content to render.")
    elif isinstance(issues, list) and not issues:
        st.text("No issues returned.")
    else:
        st.markdown("```\n" + str(issues) + "\n```")


def render_app_styles():
        # Determine theme base (light/dark) via Streamlit theme option where available
        try:
                theme_base = st.get_option("theme.base")
        except Exception:
                theme_base = None

        is_dark = False
        if theme_base:
                is_dark = str(theme_base).lower() == "dark"

        # Fallback colors (avoid #000000 and #ffffff hardcoding)
        card_bg = "#1e1e1e" if is_dark else "#ffffff"
        border_col = "#2b2b2b" if is_dark else "#e6e6e6"
        text_col = "#e6e6e6" if is_dark else "#1a1a1a"
        secondary_col = "#b3b3b3" if is_dark else "#555555"
        json_bg = "#111111" if is_dark else "#f5f5f5"

        css = f"""
        <style>
            html, body, [class*=\"css\"] {{
                font-family: Inter, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif;
            }}
            .section-gap {{
                margin-top: 1.1rem;
                margin-bottom: 0.8rem;
            }}
            .result-card {{
                background: {card_bg};
                border: 1px solid {border_col};
                border-radius: 12px;
                padding: 1rem 1rem 0.8rem 1rem;
                margin-bottom: 0.9rem;
                color: {text_col};
            }}
            .clean-text-block {{
                background: transparent;
                border: 1px solid {border_col};
                border-radius: 10px;
                padding: 0.9rem;
                line-height: 1.55;
                white-space: pre-wrap;
                color: {text_col};
            }}
            .subtle-file {{
                color: {secondary_col};
                font-size: 0.88rem;
                margin-top: 0.35rem;
            }}
            /* Tables: ensure transparent bg and proper contrast */
            div[data-testid=\"stMarkdownContainer\"] table,
            table {{
                background-color: transparent !important;
                color: {text_col} !important;
                border-collapse: collapse;
                width: 100%;
            }}
            table th, table td {{
                border: 1px solid {border_col};
                padding: 10px;
                vertical-align: top;
            }}
            table th {{
                font-weight: 600;
                color: {text_col};
            }}
            /* JSON block */
            .json-block {{
                background: {json_bg};
                color: {text_col};
                padding: 0.8rem;
                border-radius: 8px;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, \"Roboto Mono\", \"Courier New\", monospace;
                max-height: 36vh;
                overflow: auto;
                border: 1px solid {border_col};
            }}
            /* Status badge */
            .status-badge {{
                display: inline-block;
                padding: 6px 10px;
                border-radius: 999px;
                border: 1px solid {border_col};
                background: transparent;
                color: {text_col};
            }}
        </style>
        """

        st.markdown(css, unsafe_allow_html=True)


def _parse_markdown_table(md_text):
    lines = [line.strip() for line in (md_text or "").splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]
    if len(table_lines) < 3:
        return None
    header = table_lines[0]
    separator = table_lines[1]
    rows = table_lines[2:]
    return header, separator, rows


def build_sorted_feedback_markdown(issues):
    if not (isinstance(issues, list) and issues):
        return ""
    md_candidates = []
    for issue in issues:
        if isinstance(issue, dict):
            message = (issue.get("message") or "").strip()
            if message:
                md_candidates.append(message)
    if not md_candidates:
        return ""

    primary_md = md_candidates[0]
    parsed = _parse_markdown_table(primary_md)
    if not parsed:
        return primary_md

    header, separator, rows = parsed

    def _row_key(row):
        cells = [c.strip() for c in row.strip("|").split("|")]
        return cells[0].lower() if cells else ""

    rows_sorted = sorted(rows, key=_row_key)
    title_lines = [
        line for line in primary_md.splitlines() if not (line.strip().startswith("|") and line.strip().endswith("|"))
    ]
    title = "\n".join([line for line in title_lines if line.strip()]).strip()
    table = "\n".join([header, separator] + rows_sorted)
    if title:
        return f"{title}\n\n{table}"
    return table


render_app_styles()
init_session_state()

log_event(
    "app_rerun",
    audio_source=st.session_state.get("audio_source"),
    recording_ready=recording_ready(),
)

with st.container():
    st.title("Instant Spoken English Feedback")
    st.caption("Record your speech and receive structured IELTS-style feedback.")

st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

uploaded = None
audio_preview_data = None
audio_preview_path = None
audio_filename = ""
analysis_clicked = False

with st.container():
    st.markdown("## Action Section")
    st.markdown("**Step 1: Record or Upload**")
    audio_source = st.radio(
        "Choose audio input",
        ["Upload", "Load", "Record"],
        index=["Upload", "Load", "Record"].index(st.session_state.get("audio_source", "Upload")),
        key="audio_source",
        horizontal=True,
    )

    if audio_source == "Upload":
        uploaded = st.file_uploader("Upload Audio (.wav / .m4a)", type=["wav", "m4a"])
        if uploaded:
            audio_preview_data = uploaded.getvalue()
            audio_filename = uploaded.name
            signature = f"upload:{uploaded.name}:{len(audio_preview_data)}"
        else:
            signature = None
    elif audio_source == "Load":
        saved_recordings = list_saved_recordings()
        saved_options = [""] + saved_recordings
        selected_index = 0
        if (
            st.session_state.get("loaded_audio_path")
            and st.session_state["loaded_audio_path"] not in saved_recordings
        ):
            st.session_state.loaded_audio_path = None
        if st.session_state.get("loaded_audio_path") in saved_recordings:
            selected_index = saved_options.index(st.session_state["loaded_audio_path"])
        elif saved_recordings:
            selected_index = 1
        selected_saved = st.selectbox(
            "Select saved recording",
            options=saved_options,
            index=selected_index,
            format_func=(
                lambda path: "— Select a saved recording —"
                if not path
                else os.path.basename(path)
            ),
        )
        st.session_state.loaded_audio_path = selected_saved or None
        selected_path = st.session_state.get("loaded_audio_path")
        if selected_path and os.path.exists(selected_path):
            audio_preview_path = selected_path
            audio_filename = os.path.basename(selected_path)
            signature = f"load:{selected_path}:{os.path.getmtime(selected_path)}"
        else:
            signature = None
        log_event(
            "load_selection_changed",
            selected_path=selected_path,
            options_count=len(saved_recordings),
        )
    else:
        recorder_value = render_audio_recorder()
        recorder_type = type(recorder_value).__name__
        log_event(
            "recorder_value_seen",
            recorder_type=recorder_type,
            has_value=recorder_value is not None,
        )
        wav_bytes = None
        recorder_event_name = None

        if isinstance(recorder_value, (bytes, bytearray)):
            wav_bytes = bytes(recorder_value)
        elif isinstance(recorder_value, str):
            try:
                recorder_payload = json.loads(recorder_value)
            except json.JSONDecodeError:
                recorder_payload = None
            if isinstance(recorder_payload, dict):
                recorder_event_name = recorder_payload.get("event")
                if recorder_event_name:
                    log_event(
                        "recorder_component_event",
                        event_name=recorder_event_name,
                        payload=recorder_payload,
                    )
                wav_base64 = recorder_payload.get("wav_base64")
                if wav_base64:
                    wav_bytes = base64.b64decode(wav_base64)
        elif isinstance(recorder_value, dict):
            recorder_event_name = recorder_value.get("event")
            if recorder_event_name:
                log_event(
                    "recorder_component_event",
                    event_name=recorder_event_name,
                    payload=recorder_value,
                )
            wav_base64 = recorder_value.get("wav_base64")
            if wav_base64:
                wav_bytes = base64.b64decode(wav_base64)

        if recorder_event_name in {"recording_started", "start_clicked"}:
            st.session_state.is_recording = True
            st.session_state.recording_started_at = datetime.now(timezone.utc)
        if recorder_event_name in {"recording_complete", "stop_clicked"}:
            st.session_state.is_recording = False

        if wav_bytes:
            digest = hashlib.sha256(wav_bytes).hexdigest()
            if digest == st.session_state.get("last_recorded_audio_digest"):
                log_event("record_payload_duplicate", digest=digest)
                wav_bytes = None
            else:
                st.session_state.last_recorded_audio_digest = digest
                log_event("record_payload_received", bytes_len=len(wav_bytes), digest=digest)

        if wav_bytes:
            recorded_path = build_timestamped_recording_path()
            with open(recorded_path, "wb") as f:
                f.write(wav_bytes)
            st.session_state.recorded_audio_bytes = wav_bytes
            st.session_state.recorded_audio_path = recorded_path
            st.session_state.loaded_audio_path = recorded_path
            st.session_state.is_recording = False
            log_event(
                "record_saved",
                recorded_path=recorded_path,
                saved_filename=os.path.basename(recorded_path),
                bytes_len=len(wav_bytes),
            )

        if st.session_state.get("is_recording"):
            elapsed = 0
            if st.session_state.get("recording_started_at"):
                elapsed_delta = datetime.now(timezone.utc) - st.session_state.recording_started_at
                elapsed = max(0, int(elapsed_delta.total_seconds()))
            st.info(f"Recording... {elapsed // 60:02d}:{elapsed % 60:02d}")
        elif recording_ready():
            st.success("Recording saved ✓")

        if recording_ready():
            if st.session_state.recorded_audio_bytes:
                audio_preview_data = st.session_state.recorded_audio_bytes
            else:
                audio_preview_path = get_recorded_audio_path()
            record_path = st.session_state.get("recorded_audio_path")
            audio_filename = os.path.basename(record_path) if record_path else "recording.wav"
            signature = f"record:{record_path}"
        else:
            signature = None

        if recording_ready() and st.button("Re-record"):
            log_event("rerecord_clicked")
            reset_for_rerecord()
            st.rerun()

    st.markdown("**Step 2: Review Audio**")
    if audio_preview_data:
        st.audio(audio_preview_data)
        st.markdown(
            f'<div class="subtle-file">{html.escape(audio_filename)}</div>',
            unsafe_allow_html=True,
        )
    elif audio_preview_path and os.path.exists(audio_preview_path):
        st.audio(audio_preview_path, format="audio/wav")
        st.markdown(
            f'<div class="subtle-file">{html.escape(audio_filename)}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("No audio selected yet.")

    st.session_state.audio_ready = bool(audio_preview_data) or bool(
        audio_preview_path and os.path.exists(audio_preview_path)
    )

    if signature != st.session_state.get("active_audio_signature"):
        st.session_state.active_audio_signature = signature
        if signature:
            reset_analysis_state()

    st.markdown("**Step 3: Run Analysis**")
    analysis_clicked = st.button(
        "RUN ANALYSIS",
        type="primary",
        disabled=not st.session_state.audio_ready,
    )

if analysis_clicked:
    log_event(
        "run_analysis_clicked",
        audio_source=st.session_state.get("audio_source"),
        recording_ready=recording_ready(),
        recorded_path=st.session_state.get("recorded_audio_path"),
    )
    with st.status("Running analysis...", expanded=True) as status:
        status.write("Transcribing audio...")
        try:
            audio_path = get_active_audio(uploaded)
            log_event("audio_path_selected", audio_path=audio_path)
        except Exception as e:
            log_event("audio_path_error", error=str(e))
            status.update(label="Analysis failed", state="error")
            st.error(str(e))
            st.stop()

        try:
            replicate_output, transcript, diarization_text = run_transcription(audio_path)
            st.session_state["transcript_source"] = "replicate"
            log_event("transcription_replicate_ok", transcript_chars=len(transcript or ""))
        except Exception as e:
            log_event("transcription_replicate_error", error=str(e))
            try:
                replicate_output, transcript, diarization_text = run_local_transcription(
                    audio_path
                )
                st.session_state["transcript_source"] = "local"
                log_event("transcription_local_ok", transcript_chars=len(transcript or ""))
            except Exception as fallback_error:
                log_event("transcription_local_error", error=str(fallback_error))
                status.update(label="Analysis failed", state="error")
                st.error(str(fallback_error))
                st.stop()

        status.write("Generating feedback...")
        issues = run_error_detection(transcript, emit_ui=False)
        log_event("feedback_generated", issues_count=len(issues) if isinstance(issues, list) else -1)

        st.session_state.transcript = transcript or ""
        st.session_state.feedback = issues if isinstance(issues, list) else []
        st.session_state.diarization = diarization_text or ""
        st.session_state.replicate_output = replicate_output
        st.session_state.analysis_complete = True

        status.update(label="Analysis complete ✓", state="complete")

    st.success("Analysis complete ✓")

st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

with st.container():
    st.markdown("## Results Section")
    if st.session_state.analysis_complete:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("### Transcript")
        transcript_text = st.session_state.get("transcript") or ""
        st.markdown(
            f'<div class="clean-text-block">{html.escape(transcript_text)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("### Feedback Summary")
        feedback_md = build_sorted_feedback_markdown(st.session_state.get("feedback", []))
        if feedback_md:
            st.markdown(feedback_md)
        else:
            st.caption("No feedback available.")
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Detailed Output", expanded=False):
            st.markdown("#### Diarization")
            diarization_text = st.session_state.get("diarization") or ""
            if diarization_text:
                st.markdown(
                    f'<div class="clean-text-block">{html.escape(diarization_text)}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No diarization output.")
            st.markdown("#### Raw Structured Output")
            st.json(st.session_state.get("replicate_output", {}))
    else:
        st.caption("Run analysis to view transcript and feedback.")

with st.expander("Advanced Diagnostics", expanded=False):
    st.checkbox(
        "Allow rule-based fallback when OpenAI fails",
        key="allow_rule_based_fallback_on_openai_failure",
    )
    st.write(
        {
            "transcript_source": st.session_state.get("transcript_source", "unknown"),
            "feedback_source": st.session_state.get("feedback_source", "unknown"),
            "python_executable": sys.executable,
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "openai_key_present": bool(os.environ.get("OPENAI_API_KEY")),
            "replicate_key_present": bool(os.environ.get("REPLICATE_API_TOKEN")),
            "event_log_path": APP_EVENT_LOG,
        }
    )
    recent_events = read_recent_events(limit=12)
    if recent_events:
        st.code("\n".join(recent_events), language="json")
