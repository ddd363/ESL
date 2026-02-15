import base64
import binascii
import io
import json
import os
import re

import replicate
import requests
import streamlit as st
import streamlit.components.v1 as components
from pydub import AudioSegment


OPENAI_MODEL = "gpt-5.2"
AUDIO_DIR = "audio"
RECORDED_AUDIO_FILENAME = "recorded_audio.wav"
SUPPORTED_EXTS = {".wav", ".m4a"}
RECORDER_COMPONENT_DIR = os.path.join(
    os.path.dirname(__file__), "components", "audio_recorder"
)
audio_recorder_component = components.declare_component(
    "audio_recorder", path=RECORDER_COMPONENT_DIR
)

# Cell 0c: Replicate diarization (Whisper + diarization as a service)
# Put your Replicate API key here (or set in environment before running):
os.environ["REPLICATE_API_TOKEN"] = "r8_UdBzdYuIUsDW5MWGvuvgchX7FrxJvho3Vj88j"

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


def render_audio_recorder():
        return audio_recorder_component()


def recording_ready():
    # Single source of truth for recorded audio readiness.
    return bool(st.session_state.get("recorded_audio_bytes")) or bool(
        get_recorded_audio_path()
    )


def get_recorded_audio_path():
    recorded_path = st.session_state.get("recorded_audio_path")
    if recorded_path and os.path.exists(recorded_path):
        return recorded_path
    fallback_path = os.path.join(AUDIO_DIR, RECORDED_AUDIO_FILENAME)
    if os.path.exists(fallback_path):
        return fallback_path
    return None


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
    if source == "Upload":
        if not uploaded_file:
            st.error("Please upload a .wav or .m4a file first.")
            st.stop()
        return load_and_convert_audio(uploaded_file.getvalue(), uploaded_file.name)
    if source == "Record":
        if not recording_ready():
            st.error("Please record audio first.")
            st.stop()
        recorded_path = get_recorded_audio_path()
        st.write(f"Audio loaded from recording: {recorded_path}")
        return recorded_path
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


def run_transcription(wav_path):
    import os

    # Ensure audio is prepared (run Cell 1 first to set AUDIO_PATH)
    if not wav_path:
        raise RuntimeError("AUDIO_PATH not set. Run the audio load cell first.")

    if not os.environ.get("REPLICATE_API_TOKEN"):
        raise RuntimeError("REPLICATE_API_TOKEN not set. Add it in this cell and re-run.")

    # Replicate diarization settings
    num_speakers = 2  # set to None to autodetect
    group_segments = True  # merge short same-speaker segments

    # Run diarization on Replicate (pin to a model version)
    model_id = "thomasmol/whisper-diarization:1495a9cddc83b2203b0d8d3516e38b80fd1572ebc4bc5700ac1da56a9b3ed886"
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
        rule = match.rule
        if isinstance(rule, dict) and "id" in rule:
            return rule["id"]
        if hasattr(rule, "id"):
            return rule.id
    return "UNKNOWN"


def run_error_detection(transcript):
    # 2) Call OpenAI (only if we have transcript + key)
    matches = []
    if not transcript:
        st.text(
            "Transcript not available yet. Run the diarization/transcription cell(s) first, then re-run this cell."
        )
        return matches

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not set.")
        st.stop()

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

    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        r.raise_for_status()
        resp = r.json()

        if "choices" in resp and len(resp["choices"]) > 0:
            content = resp["choices"][0].get("message", {}).get("content")
            if isinstance(content, str):
                content_str = content
            else:
                content_str = json.dumps(content, ensure_ascii=False)
            data = json.loads(content_str) if isinstance(content_str, str) else {}
            if isinstance(data, dict):
                matches = data.get("issues", []) or []
            else:
                matches = []
        else:
            raise ValueError(
                f"Unexpected response format from OpenAI: {json.dumps(resp, indent=2)}"
            )
    except Exception as e:
        st.text("OpenAI request failed:")
        st.text(str(e))
        try:
            st.text(getattr(r, "text", ""))
        except Exception:
            pass
        matches = []

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


st.title("ESL Speech Analysis (Local)")

init_session_state()
audio_source = st.radio("Audio source", ["Upload", "Record"], index=0, key="audio_source")

uploaded = None
if audio_source == "Upload":
    uploaded = st.file_uploader("Upload Audio (.wav / .m4a)", type=["wav", "m4a"])
else:
    recorder_value = render_audio_recorder()
    wav_bytes = None
    if isinstance(recorder_value, (bytes, bytearray)):
        wav_bytes = bytes(recorder_value)
    elif isinstance(recorder_value, str):
        try:
            recorder_payload = json.loads(recorder_value)
        except json.JSONDecodeError:
            recorder_payload = None
        if isinstance(recorder_payload, dict):
            wav_base64 = recorder_payload.get("wav_base64")
            if wav_base64:
                wav_bytes = base64.b64decode(wav_base64)
    elif isinstance(recorder_value, dict):
        wav_base64 = recorder_value.get("wav_base64")
        if wav_base64:
            wav_bytes = base64.b64decode(wav_base64)

    if wav_bytes:
        os.makedirs(AUDIO_DIR, exist_ok=True)
        recorded_path = os.path.join(AUDIO_DIR, RECORDED_AUDIO_FILENAME)
        with open(recorded_path, "wb") as f:
            f.write(wav_bytes)
        st.session_state.recorded_audio_bytes = wav_bytes
        st.session_state.recorded_audio_path = recorded_path
        st.session_state.is_recording = False
    if recording_ready():
        st.text("Recording complete")
        st.markdown("**Playback**")
        if st.session_state.recorded_audio_bytes:
            st.audio(st.session_state.recorded_audio_bytes, format="audio/wav")
        else:
            playback_path = get_recorded_audio_path()
            if playback_path:
                st.audio(playback_path, format="audio/wav")
        if st.button("Re-record"):
            st.session_state.recorded_audio_bytes = None
            st.session_state.recorded_audio_path = None
            st.session_state.is_recording = False
run = st.button("Run Analysis")

if run:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not set.")
        st.stop()

    with st.spinner("Processing audio..."):
        audio_path = get_active_audio(uploaded)
        replicate_output, transcript, diarization_text = run_transcription(audio_path)

    st.markdown("---")
    st.markdown("### Transcript + Diarization")
    if diarization_text:
        st.text_area("Diarization", diarization_text, height=260)
    if transcript:
        st.text_area("Transcript", transcript, height=180)

    st.markdown("---")
    st.markdown("### Feedback Table")
    issues = run_error_detection(transcript)
    render_feedback_table(issues)
