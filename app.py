import base64
import concurrent.futures
import difflib
import html
import hashlib
import io
import json
import math
import os
import re
import sys
import threading
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from queue import Queue, Empty
from time import perf_counter

import replicate
import requests
import streamlit as st
import streamlit.components.v1 as components
from pydub import AudioSegment

from feedback_aggregator import aggregate_feedback

st.set_page_config(
    page_title="Instant Spoken English Feedback",
    layout="wide",
    initial_sidebar_state="collapsed",
)

try:
    import websocket
except Exception:
    websocket = None

# ---------------------------------------------------------------------------
# Module-level thread-safe queues for non-blocking Deepgram streaming.
# These persist across Streamlit reruns (module scope, not session scope).
# ---------------------------------------------------------------------------
audio_queue: Queue = Queue()
transcript_queue: Queue = Queue()
ACTIVE_DEEPGRAM_STREAMERS = set()
ACTIVE_DEEPGRAM_STREAMERS_LOCK = threading.Lock()

OPENAI_MODEL = "gpt-5.2"
STRICT_OPENAI_FEEDBACK = True
DEFAULT_OPENAI_SYSTEM_PROMPT = """
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
""".strip()
AUDIO_DIR = "audio"
APP_EVENT_LOG = "/tmp/esl_app_events.log"
OPENAI_PROMPT_PATH = os.path.join("outputs", "openai_system_prompt.txt")
SUPPORTED_EXTS = {".wav", ".m4a"}
RECORDER_COMPONENT_DIR = os.path.join(
    os.path.dirname(__file__), "components", "audio_recorder"
)
audio_recorder_component = components.declare_component(
    "audio_recorder", path=RECORDER_COMPONENT_DIR
)
OPENAI_HTTP = requests.Session()
DEEPGRAM_RESTART_MIN_INTERVAL_S = 1.0
JOB_TIMEOUT_S = 120
BACKGROUND_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4)
DEEPGRAM_DEFAULT_MODEL = os.environ.get("DEEPGRAM_MODEL", "nova-3")
DEEPGRAM_DEFAULT_LANGUAGE = os.environ.get("DEEPGRAM_LANGUAGE", "en")
DEEPGRAM_KEYTERMS = tuple(
    term.strip() for term in os.environ.get("DEEPGRAM_KEYTERMS", "").split(",") if term.strip()
)
DEEPGRAM_ENDPOINTING_MS = os.environ.get("DEEPGRAM_ENDPOINTING_MS")
DEEPGRAM_UTTERANCE_END_MS = os.environ.get("DEEPGRAM_UTTERANCE_END_MS", "1000")
DEEPGRAM_SMART_FORMAT = os.environ.get("DEEPGRAM_SMART_FORMAT", "true").lower() == "true"
LOW_CONFIDENCE_THRESHOLD = 0.75
DEFAULT_TURN_SILENCE_TIMEOUT_S = 2.0
DEFAULT_TURN_MIN_WORDS = 10
DEFAULT_TURN_ANALYSIS_CAP = 20
DEFAULT_TURN_MAX_FLAGGED_RATIO = 0.50
TURN_ANALYSIS_MODEL = os.environ.get("TURN_ANALYSIS_MODEL", "gpt-4.1-mini")
DEV_MODE = (
    os.environ.get("ESL_DEV_MODE", "").lower() in {"1", "true", "yes"}
    or st.query_params.get("dev") == "1"
    or st.query_params.get("debug") == "1"
)
TURN_ANALYSIS_SYSTEM_PROMPT = """
You are an IELTS Speaking turn analyst. Analyze only the supplied candidate turn.
Words in parentheses are possible ASR artifacts, not candidate errors; do not penalize
or correct them. Return concise JSON only. Use exact phrases from the turn for every
observation or error. For each highlight use: "exact quote | category: lexical range,
cohesion, narrative control, register, or fluency | why: one short teacher-ready reason".
Do not write paragraphs.
""".strip()
TURN_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "provisional_scores": {
            "type": "object",
            "properties": {
                "fluency_coherence": {"type": "number"},
                "lexical_resource": {"type": "number"},
                "grammar": {"type": "number"},
            },
            "required": ["fluency_coherence", "lexical_resource", "grammar"],
            "additionalProperties": False,
        },
        "observations": {"type": "array", "items": {"type": "string"}},
        "errors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "phrase": {"type": "string"},
                    "category": {"type": "string"},
                    "suggestion": {"type": "string"},
                },
                "required": ["phrase", "category", "suggestion"],
                "additionalProperties": False,
            },
        },
        "highlights": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["provisional_scores", "observations", "errors", "highlights"],
    "additionalProperties": False,
}


def _turn_text(words, annotate=True):
    parts = []
    for word in words:
        token = (word.get("word") or "").strip()
        if not token:
            continue
        if annotate:
            token = _mark_low_confidence_text(
                token, word.get("confidence"), word.get("flag_reasons")
            )
        parts.append(token)
    return " ".join(parts).strip()


def _speaker_durations(words):
    durations = {}
    for word in words or []:
        speaker = word.get("speaker")
        start, end = word.get("start"), word.get("end")
        if speaker is None or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        durations[speaker] = durations.get(speaker, 0.0) + max(0.0, end - start)
    return durations


def _candidate_turns(words, candidate_speaker, audio_position_s, silence_timeout_s=DEFAULT_TURN_SILENCE_TIMEOUT_S, force_last_close=False):
    """Build closed candidate turns from finalized, diarized Deepgram words."""
    segments = []
    for word in words or []:
        if not isinstance(word, dict) or not (word.get("word") or "").strip():
            continue
        speaker = word.get("speaker")
        if segments and segments[-1]["speaker"] == speaker:
            segments[-1]["words"].append(word)
        else:
            segments.append({"speaker": speaker, "words": [word]})

    turns = []
    index = 0
    segment_index = 0
    while segment_index < len(segments):
        segment = segments[segment_index]
        if segment["speaker"] != candidate_speaker:
            segment_index += 1
            continue
        examiner_context = ""
        if segment_index and segments[segment_index - 1]["speaker"] != candidate_speaker:
            examiner_context = _turn_text(segments[segment_index - 1]["words"], annotate=True)
        turn_words = list(segment["words"])
        next_index = segment_index + 1
        # A brief examiner utterance between two candidate segments belongs to one turn.
        if (
            next_index + 1 < len(segments)
            and segments[next_index]["speaker"] != candidate_speaker
            and segments[next_index + 1]["speaker"] == candidate_speaker
        ):
            bridge = segments[next_index]["words"]
            bridge_start = bridge[0].get("start")
            bridge_end = bridge[-1].get("end")
            if isinstance(bridge_start, (int, float)) and isinstance(bridge_end, (int, float)) and bridge_end - bridge_start < 1.5:
                turn_words.extend(segments[next_index + 1]["words"])
                next_index += 2
        end = turn_words[-1].get("end")
        closed = next_index < len(segments)
        if force_last_close and next_index == len(segments):
            closed = True
        # Wait for a brief examiner bridge to either lead back to the candidate
        # (merge above) or age out, preventing a premature partial analysis.
        if (
            closed
            and next_index == segment_index + 1
            and next_index < len(segments)
            and segments[next_index]["speaker"] != candidate_speaker
        ):
            bridge_words = segments[next_index]["words"]
            bridge_start = bridge_words[0].get("start")
            bridge_end = bridge_words[-1].get("end")
            if (
                isinstance(bridge_start, (int, float))
                and isinstance(bridge_end, (int, float))
                and bridge_end - bridge_start < 1.5
            ):
                closed = audio_position_s - bridge_end >= 1.5
        if not closed and isinstance(end, (int, float)):
            closed = audio_position_s - end >= silence_timeout_s
        text = _turn_text(turn_words, annotate=True)
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        start_key = f"{float(turn_words[0].get('start') or 0):.3f}"
        turns.append({
            "index": index,
            "key": f"{start_key}:{text_hash}",
            "words": turn_words,
            "text": text,
            "start": turn_words[0].get("start"),
            "end": end,
            "closed": closed,
            "examiner_context": examiner_context,
        })
        index += 1
        segment_index = next_index
    return turns


def _clean_word_text(record):
    return str(record.get("word") or record.get("punctuated_word") or "").strip()


def _processed_turns(words, candidate_speaker, audio_position_s, *, force_last_close=False):
    """Build immutable, all-speaker turns from finalized Deepgram words.

    The returned ``clean_text`` is the only representation eligible for model
    analysis. Display uncertainty remains word metadata, never inline markup.
    """
    segments = []
    for record in words or []:
        if not isinstance(record, dict) or not _clean_word_text(record):
            continue
        speaker = record.get("speaker")
        if segments and segments[-1]["speaker"] == speaker:
            segments[-1]["records"].append(record)
        else:
            segments.append({"speaker": speaker, "records": [record]})

    processed = []
    for index, segment in enumerate(segments):
        records = segment["records"]
        start = records[0].get("start")
        end = records[-1].get("end")
        closed = index < len(segments) - 1 or force_last_close
        if not closed and isinstance(end, (int, float)):
            closed = audio_position_s - end >= DEFAULT_TURN_SILENCE_TIMEOUT_S
        clean_text = " ".join(_clean_word_text(record) for record in records).strip()
        if not clean_text:
            continue
        words_payload = [
            {
                "text": _clean_word_text(record),
                "start": record.get("start"),
                "end": record.get("end"),
                "confidence": record.get("confidence"),
                "is_final": True,
                "low_confidence": isinstance(record.get("confidence"), (int, float)) and record["confidence"] < _confidence_flag_threshold(),
                "is_duplicate_token": "duplicate" in (record.get("flag_reasons") or ()),
                "flag_reasons": list(record.get("flag_reasons") or ()),
            }
            for record in records
        ]
        text_hash = hashlib.sha256(clean_text.lower().encode("utf-8")).hexdigest()[:16]
        start_key = f"{float(start or 0):.3f}"
        processed.append({
            "key": f"{start_key}:{text_hash}",
            "speaker": segment["speaker"],
            "speaker_label": _speaker_label(segment["speaker"]) or "Unknown speaker",
            "is_candidate": segment["speaker"] == candidate_speaker,
            "t_start": start,
            "t_end": end,
            "start": start,
            "end": end,
            "words": words_payload,
            "clean_text": clean_text,
            # Compatibility with worker/logging paths. It is deliberately clean.
            "text": clean_text,
            "closed": closed,
        })
    return processed


def _turn_word_key(word):
    return re.sub(r"[^a-z0-9']+", "", str(word.get("text") or word.get("word") or "").lower())


def _turn_delta_words(previous_words, current_words, max_window=8):
    """Return only newly finalised words from a revised cumulative segment."""
    previous_keys = [_turn_word_key(word) for word in previous_words]
    current_keys = [_turn_word_key(word) for word in current_words]
    prefix = 0
    while prefix < len(previous_keys) and prefix < len(current_keys) and previous_keys[prefix] == current_keys[prefix]:
        prefix += 1
    if prefix == len(previous_keys):
        return list(current_words[prefix:])
    for window in range(min(max_window, len(previous_keys), len(current_keys)), 0, -1):
        if previous_keys[-window:] == current_keys[:window]:
            return list(current_words[window:])
    return list(current_words[prefix:])


class IncrementalTurnAnalysisWorker:
    """Serial OpenAI worker; its lock-protected state is safe to render from Streamlit."""
    def __init__(self):
        self.queue = Queue()
        self.lock = threading.Lock()
        self.results = {}
        self.submitted = set()
        self.cancelled = set()
        self.accepting = True
        self.stats = {"turns_closed": 0, "turns_analysed": 0, "turns_gated_out": 0, "examiner_turns_ignored": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, turn):
        with self.lock:
            if not self.accepting or turn["key"] in self.submitted:
                return False
            self.submitted.add(turn["key"])
            self.results[turn["key"]] = {"status": "queued", "turn": turn}
        self.queue.put(turn)
        log_event("turn_analysis_queued", turn_key=turn["key"], words=len(turn["text"].split()))
        return True

    def snapshot(self):
        with self.lock:
            return dict(self.results), dict(self.stats)

    def stop_pending(self):
        """Prevent new work and mark queued, unstarted turns as cancelled."""
        with self.lock:
            self.accepting = False
            self.cancelled.update(
                key for key, item in self.results.items()
                if item.get("status") in {"queued", "analysing"}
            )
        cancelled = 0
        while True:
            try:
                turn = self.queue.get_nowait()
            except Empty:
                break
            with self.lock:
                item = self.results.get(turn["key"])
                if item and item.get("status") == "queued":
                    item.update({"status": "cancelled", "error": "Recording ended before analysis started."})
                    cancelled += 1
        if cancelled:
            log_event("turn_analysis_queue_drained", cancelled=cancelled)

    def close_input(self):
        """Stop accepting new turns without discarding analyses already queued."""
        with self.lock:
            self.accepting = False
        log_event("turn_analysis_input_closed")

    def resume(self):
        with self.lock:
            self.accepting = True

    def retry_failed(self):
        """Requeue failed or cancelled analyses without duplicating successful turns."""
        retries = []
        with self.lock:
            for key, item in self.results.items():
                if item.get("status") not in {"cancelled", "error"}:
                    continue
                self.cancelled.discard(key)
                item.update({"status": "queued", "error": ""})
                retries.append(item["turn"])
            if retries:
                self.accepting = True
        for turn in retries:
            self.queue.put(turn)
        if retries:
            log_event("turn_analysis_retried", count=len(retries))
        return len(retries)

    def _run(self):
        while True:
            turn = self.queue.get()
            usage = {}
            raw_body = ""
            with self.lock:
                if turn["key"] in self.cancelled:
                    item = self.results.get(turn["key"])
                    if item:
                        item.update({"status": "cancelled", "error": "Analysis cancelled."})
                    continue
                self.results[turn["key"]]["status"] = "analysing"
            log_event("turn_analysis_request_sent", turn_key=turn["key"], model=TURN_ANALYSIS_MODEL)
            try:
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise RuntimeError("OPENAI_API_KEY is not set")
                context = _truncate_words(turn.get("examiner_context", ""), max_words=40)
                clean_text = (turn.get("clean_text") or turn.get("text") or "").strip()
                if not clean_text:
                    raise RuntimeError("Closed candidate turn has no clean transcript text.")
                user_msg = f"Examiner context: {context or '[none]'}\nCandidate turn: {clean_text}"
                response = OPENAI_HTTP.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": TURN_ANALYSIS_MODEL,
                        "messages": [{"role": "system", "content": TURN_ANALYSIS_SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
                        "response_format": {"type": "json_schema", "json_schema": {"name": "turn_analysis", "schema": TURN_ANALYSIS_SCHEMA, "strict": True}},
                        "max_completion_tokens": 800,
                    },
                    timeout=60,
                )
                raw_body = response.text or ""
                log_event(
                    "turn_analysis_http_response",
                    turn_key=turn["key"],
                    status_code=response.status_code,
                    raw_body_preview=raw_body[:1500],
                )
                response.raise_for_status()
                payload = json.loads(raw_body) if raw_body.strip() else {}
                usage = payload.get("usage") or {}
                choice = (payload.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                content = message.get("content")
                finish_reason = choice.get("finish_reason")
                log_event(
                    "turn_analysis_response_received",
                    turn_key=turn["key"],
                    status_code=response.status_code,
                    finish_reason=finish_reason,
                    usage=usage,
                    content_preview=(content or "")[:500],
                    response_preview=response.text[:1000],
                )
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError(
                        f"OpenAI returned empty completion (finish_reason={finish_reason!r}, "
                        f"status={response.status_code}, usage={usage})"
                    )
                data = json.loads(content)
                with self.lock:
                    item = self.results[turn["key"]]
                    if turn["key"] in self.cancelled:
                        item.update({"status": "cancelled", "error": "Analysis cancelled while request was in flight.", "usage": usage})
                        self.stats["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                        self.stats["completion_tokens"] += int(usage.get("completion_tokens") or 0)
                        log_event("turn_analysis_result_discarded", turn_key=turn["key"], usage=usage)
                        continue
                    item.update({"status": "complete", "analysis": data, "usage": usage})
                    self.stats["turns_analysed"] += 1
                    self.stats["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                    self.stats["completion_tokens"] += int(usage.get("completion_tokens") or 0)
                log_event("turn_analysis_result_stored", turn_key=turn["key"], prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"))
            except Exception as error:
                with self.lock:
                    item = self.results[turn["key"]]
                    if turn["key"] in self.cancelled:
                        item.update({"status": "cancelled", "error": "Analysis cancelled while request was in flight.", "usage": usage})
                        self.stats["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                        self.stats["completion_tokens"] += int(usage.get("completion_tokens") or 0)
                        log_event("turn_analysis_result_discarded", turn_key=turn["key"], usage=usage)
                        continue
                    item.update({
                        "status": "error",
                        "error": str(error)[:500],
                        "usage": usage,
                        "response_preview": raw_body[:500],
                    })
                    self.stats["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                    self.stats["completion_tokens"] += int(usage.get("completion_tokens") or 0)
                log_event("turn_analysis_failed", turn_key=turn["key"], error=str(error), usage=usage)


def _fmt_seconds(value):
    return f"{float(value):.2f}s"


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


def read_recent_event_objects(limit=80):
    raw_lines = read_recent_events(limit=limit)
    parsed = []
    for line in raw_lines:
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                parsed.append(obj)
        except Exception:
            continue
    return parsed


def _fmt_debug_value(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def build_live_debug_lines(limit=80, only_important=False):
    events = read_recent_event_objects(limit=limit)
    if only_important:
        keep_tokens = (
            "deepgram",
            "recorder",
            "error",
            "exception",
            "warning",
            "timeout",
            "analysis",
            "transcription",
            "engine_restart",
            "websocket_recreated",
            "audio_generator",
            "keepalive",
            "health",
            "deepgram_not_emitting",
            "deepgram_stream_config",
            "deepgram_raw_message",
            "deepgram_result_parsed",
            "deepgram_result_no_alternatives",
            "deepgram_chunk_missing",
            "deepgram_audio_milestone",
            "deepgram_final_flush",
        )
        filtered = []
        for evt in events:
            event_name = (evt.get("event") or "").lower()
            if any(token in event_name for token in keep_tokens):
                filtered.append(evt)
        events = filtered

    lines = []
    for evt in events:
        ts = evt.get("ts", "")
        event_name = evt.get("event", "unknown")
        key_parts = []
        for k in (
            "event_name",
            "connected",
            "error",
            "status_code",
            "sample_rate",
            "latency_s",
            "idle_s",
            "timeout_s",
            "issues_count",
            "transcript_chars",
            "bytes_sent_total",
            "transcript_events_received",
            "is_final",
            "transcript_len",
            "words_count",
            "confidence",
            "msg_type",
            "hint",
            "n",
            "preview",
        ):
            if k in evt:
                key_parts.append(f"{k}={_fmt_debug_value(evt.get(k))}")
        suffix = f" | {'; '.join(key_parts)}" if key_parts else ""
        lines.append(f"{ts} | {event_name}{suffix}")
    return lines


def render_live_debug_monitor():
    with st.expander("Live Debug Monitor", expanded=False):
        controls_left, controls_mid, controls_right = st.columns([1, 1, 2])
        with controls_left:
            auto_refresh = st.checkbox("Auto-refresh", key="debug_log_autorefresh")
        with controls_mid:
            important_only = st.checkbox("Important only", key="debug_log_important_only")
        with controls_right:
            log_lines = st.slider("Log lines", min_value=20, max_value=200, value=80, step=10, key="debug_log_lines")

        streamer = st.session_state.get("deepgram_streamer")
        if streamer:
            try:
                snapshot = streamer.snapshot()
            except Exception as e:
                snapshot = {"error": str(e)}
        else:
            snapshot = {}

        stream_health = {
            "is_recording": bool(st.session_state.get("is_recording")),
            "streaming_active": bool(st.session_state.get("streaming_active")),
            "worker_started": bool(st.session_state.get("worker_started")),
            "deepgram_streamer_present": bool(streamer),
            "deepgram_is_active": bool(streamer.is_active) if streamer else False,
            "deepgram_is_connecting": bool(streamer.is_connecting) if streamer else False,
            "deepgram_error": snapshot.get("error"),
            "deepgram_disconnected_for_s": (
                round(perf_counter() - streamer.disconnected_since, 1)
                if streamer and streamer.disconnected_since is not None
                else None
            ),
            "deepgram_latency_s": snapshot.get("latency_s"),
            "deepgram_final_chars": len(snapshot.get("final") or ""),
            "deepgram_interim_chars": len(snapshot.get("interim") or ""),
            "deepgram_live_chars": len(snapshot.get("live") or ""),
            # Session-state transcript chars — non-zero even after streamer is cleared
            "deepgram_session_live_chars": len(st.session_state.get("deepgram_live_transcript") or ""),
            "deepgram_session_final_chars": len(st.session_state.get("deepgram_final_transcript") or ""),
            "deepgram_accumulated_chars": len(st.session_state.get("deepgram_accumulated_transcript") or ""),
            "pending_chunks": len(getattr(streamer, "pending_chunks", []) or []) if streamer else 0,
            "deepgram_bytes_sent_total": snapshot.get("bytes_sent_total", 0),
            "deepgram_chunks_received": snapshot.get("chunks_received", 0),
            "deepgram_transcript_events_received": snapshot.get("transcript_events_received", 0),
            "deepgram_stream_sample_rate": snapshot.get("sample_rate"),
            "transcript_queue_size": transcript_queue.qsize(),
            "last_recorder_event_id": st.session_state.get("last_recorder_event_id", ""),
            "last_recorder_event_t": st.session_state.get("last_recorder_event_t", 0),
            "deepgram_started_at": st.session_state.get("deepgram_started_at"),
            "deepgram_streaming_latency": st.session_state.get("deepgram_streaming_latency"),
            "deepgram_total_time": st.session_state.get("deepgram_total_time"),
            "openai_key_present": bool(os.environ.get("OPENAI_API_KEY")),
            "replicate_key_present": bool(os.environ.get("REPLICATE_API_TOKEN")),
            "deepgram_key_present": bool(os.environ.get("DEEPGRAM_API_KEY")),
        }
        st.json(stream_health)

        debug_lines = build_live_debug_lines(limit=log_lines, only_important=important_only)
        if debug_lines:
            st.code("\n".join(debug_lines), language="text")
        else:
            st.caption("No events available yet.")

        if auto_refresh:
            components.html(
                """
                <script>
                    setTimeout(function () {
                        window.parent.location.reload();
                    }, 1000);
                </script>
                """,
                height=0,
            )


def render_error_log():
    """Expose actionable runtime failures without exposing all development controls."""
    failure_tokens = ("error", "failed", "exception", "timeout", "cancelled")
    failures = [
        event for event in read_recent_event_objects(limit=100)
        if any(token in str(event.get("event") or "").lower() for token in failure_tokens)
    ]
    if not failures:
        return
    with st.expander(f"Error log ({len(failures)})", expanded=False):
        st.code(
            "\n".join(json.dumps(event, ensure_ascii=False) for event in failures[-20:]),
            language="json",
        )


def load_persisted_openai_prompt():
    try:
        if os.path.exists(OPENAI_PROMPT_PATH):
            with open(OPENAI_PROMPT_PATH, "r", encoding="utf-8") as f:
                saved_prompt = f.read().strip()
            if saved_prompt:
                return saved_prompt
    except Exception as e:
        log_event("openai_prompt_load_error", error=str(e), path=OPENAI_PROMPT_PATH)
    return DEFAULT_OPENAI_SYSTEM_PROMPT


def persist_openai_prompt():
    prompt = (st.session_state.get("openai_system_prompt") or "").strip()
    if not prompt:
        prompt = DEFAULT_OPENAI_SYSTEM_PROMPT
    try:
        os.makedirs(os.path.dirname(OPENAI_PROMPT_PATH), exist_ok=True)
        with open(OPENAI_PROMPT_PATH, "w", encoding="utf-8") as f:
            f.write(prompt)
        log_event("openai_prompt_saved", path=OPENAI_PROMPT_PATH, chars=len(prompt))
    except Exception as e:
        log_event("openai_prompt_save_error", error=str(e), path=OPENAI_PROMPT_PATH)


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


def _speaker_label(value):
    try:
        idx = int(value)
    except Exception:
        return ""
    if 0 <= idx < 26:
        return f"Speaker {chr(ord('A') + idx)}"
    return f"Speaker {idx + 1}"


def _confidence_flag_threshold():
    try:
        return float(st.session_state.get("low_confidence_threshold", LOW_CONFIDENCE_THRESHOLD))
    except Exception:
        return LOW_CONFIDENCE_THRESHOLD


def _enabled_deepgram_flag_reasons(confidence, flag_reasons=()):
    reasons = []
    if isinstance(confidence, (int, float)) and confidence < _confidence_flag_threshold():
        reasons.append("low_confidence")
    stored_reasons = set(flag_reasons or [])
    if st.session_state.get("flag_duplicate_tokens", True) and "duplicate" in stored_reasons:
        reasons.append("duplicate")
    if st.session_state.get("flag_interim_churn", False) and "churn" in stored_reasons:
        reasons.append("churn")
    return reasons


def _deepgram_word_key(value):
    return "".join(re.findall(r"[\w']+", (value or "").lower()))


def _finalized_deepgram_words(
    words, previous_speaker=None, previous_word=None, interim_text=""
):
    """Normalize finalized Deepgram words while preserving per-word diarization."""
    records = []
    for item in words or []:
        if not isinstance(item, dict):
            continue
        token = (item.get("punctuated_word") or item.get("word") or "").strip()
        if not token:
            continue
        speaker = item.get("speaker")
        if speaker is None:
            speaker = previous_speaker
        else:
            previous_speaker = speaker
        confidence = item.get("confidence")
        flag_reasons = []
        records.append({
            "word": token,
            "confidence": confidence,
            "speaker": speaker,
            "start": item.get("start"),
            "end": item.get("end"),
            "flag_reasons": flag_reasons,
        })

    previous_token = _deepgram_word_key(previous_word)
    for record in records:
        token = _deepgram_word_key(record.get("word"))
        if token and token == previous_token:
            record["flag_reasons"].append("duplicate")
        if token:
            previous_token = token

    interim_tokens = re.findall(r"[\w']+", (interim_text or "").lower())
    final_tokens = [_deepgram_word_key(record.get("word")) for record in records]
    for tag, _i1, _i2, j1, j2 in difflib.SequenceMatcher(
        a=interim_tokens, b=final_tokens, autojunk=False
    ).get_opcodes():
        if tag != "equal":
            for index in range(j1, j2):
                if final_tokens[index]:
                    records[index]["flag_reasons"].append("churn")
    return records


def _render_deepgram_transcript(words, annotate=True, interim_text=""):
    """Render finalized words once, adding labels only at speaker changes."""
    chunks = []
    no_speaker = object()
    current_speaker = no_speaker
    has_word = False
    for record in words or []:
        if not isinstance(record, dict):
            continue
        token = (record.get("word") or record.get("punctuated_word") or "").strip()
        if not token:
            continue
        speaker = record.get("speaker")
        if speaker is None and current_speaker is not no_speaker:
            speaker = current_speaker
        if speaker != current_speaker:
            label = _speaker_label(speaker)
            if label:
                if has_word:
                    chunks.append("\n\n")
                chunks.append(f"{label}: ")
            elif has_word:
                chunks.append(" ")
            current_speaker = speaker
        elif has_word:
            chunks.append(" ")
        if annotate:
            token = _mark_low_confidence_text(
                token, record.get("confidence"), record.get("flag_reasons")
            )
        chunks.append(token)
        has_word = True

    rendered = "".join(chunks).strip()
    interim = (interim_text or "").strip()
    return " ".join(part for part in [rendered, interim] if part).strip()


def _mark_low_confidence_text(text, confidence, flag_reasons=()):
    reasons = _enabled_deepgram_flag_reasons(confidence, flag_reasons)
    if not reasons:
        return text
    return re.sub(r"\b[\w']+\b", lambda match: f"({match.group(0)})", text)


def _deepgram_flagged_word_count(words):
    total = 0
    flagged = 0
    for record in words or []:
        if not isinstance(record, dict) or not (record.get("word") or record.get("punctuated_word")):
            continue
        total += 1
        if _enabled_deepgram_flag_reasons(record.get("confidence"), record.get("flag_reasons")):
            flagged += 1
    return flagged, total


def _replicate_segment_confidence(segment):
    for key in ("confidence", "probability"):
        confidence = segment.get(key)
        if isinstance(confidence, (int, float)):
            return confidence

    avg_logprob = segment.get("avg_logprob")
    if isinstance(avg_logprob, (int, float)):
        return math.exp(avg_logprob)
    return None


def _has_low_confidence_replicate_segment(output):
    return any(
        (confidence := _replicate_segment_confidence(segment)) is not None
        and confidence < _confidence_flag_threshold()
        for segment in _extract_replicate_segments(output)
        if isinstance(segment, dict)
    )


class DeepgramStreamingClient:
    def __init__(
        self,
        api_key,
        sample_rate,
        model=DEEPGRAM_DEFAULT_MODEL,
        language=DEEPGRAM_DEFAULT_LANGUAGE,
        keyterms=DEEPGRAM_KEYTERMS,
        endpointing_ms=DEEPGRAM_ENDPOINTING_MS,
        utterance_end_ms=DEEPGRAM_UTTERANCE_END_MS,
        smart_format=DEEPGRAM_SMART_FORMAT,
    ):
        self.api_key = api_key
        self.sample_rate = int(sample_rate) if sample_rate else 16000
        self.model = model
        self.language = language
        self.keyterms = tuple(keyterms)
        self.endpointing_ms = endpointing_ms
        self.utterance_end_ms = utterance_end_ms
        self.smart_format = bool(smart_format)
        self.ws = None
        self.thread = None
        self.keepalive_thread = None
        self.audio_sender_thread = None
        self.stop_event = threading.Event()
        self.connected_event = threading.Event()
        self.lock = threading.Lock()
        self.started_at = perf_counter()
        self.last_audio_sent_at = self.started_at
        self._heartbeat_last = self.started_at
        self.disconnected_since = None  # perf_counter() timestamp when socket went down
        self.first_result_at = None
        self.final_words = []
        self.interim_text = ""
        self.error = None
        self.pending_chunks = []
        self.max_pending_chunks = 40
        # Running byte counter — incremented only when ws_app.send() succeeds.
        # Visible in snapshot() and the debug monitor; zero means Deepgram
        # never received any audio.
        self.bytes_sent_total = 0
        # Log a milestone event every LOG_BYTES_MILESTONE bytes sent.
        self._bytes_logged_milestone = 0
        # Count of audio chunks received from the JS recorder component.
        # If this grows but bytes_sent_total stays zero, the WS send path is broken.
        self.chunks_received = 0
        # Count of Results-type messages received back from Deepgram.
        # If bytes_sent_total > 1 MB but this is still 0, Deepgram is not
        # returning any transcript events at all (encoding/channel mismatch).
        self.transcript_events_received = 0
        # Throttle raw-message logging: capture only the first N frames.
        self._raw_messages_logged = 0
        # Guard so the DEEPGRAM_NOT_EMITTING_TRANSCRIPTS hard-fail logs once.
        self._no_transcripts_warned = False
        # Signalled when Deepgram delivers the first is_final=True frame after
        # CloseStream is sent, confirming all final transcripts have arrived.
        self.final_flush_event = threading.Event()
        self.turn_boundary_sequence = 0

    def _build_url(self):
        params = [
            ("model", self.model),
            ("language", self.language),
            ("encoding", "linear16"),
            ("sample_rate", str(self.sample_rate)),
            ("channels", "1"),
            ("interim_results", "true"),
            ("punctuate", "true"),
            ("smart_format", str(self.smart_format).lower()),
            ("diarize", "true"),
        ]
        params.extend(("keyterm", term) for term in self.keyterms)
        if self.endpointing_ms:
            params.append(("endpointing", str(self.endpointing_ms)))
        if self.utterance_end_ms:
            params.extend(
                [
                    ("utterance_end_ms", str(self.utterance_end_ms)),
                    ("vad_events", "true"),
                ]
            )
        return "wss://api.deepgram.com/v1/listen?" + urllib.parse.urlencode(params)

    def _on_open(self, _ws):
        with self.lock:
            self.error = None
            self.disconnected_since = None
        self.connected_event.set()
        self._flush_pending_chunks()
        # ── Log full connection config so any encoding/channel mismatch is visible ──
        log_event(
            "deepgram_stream_config",
            model=self.model,
            language=self.language,
            encoding="linear16",
            sample_rate=self.sample_rate,
            channels=1,
            interim_results=True,
            punctuate=True,
            smart_format=self.smart_format,
            diarize=True,
            keyterms_count=len(self.keyterms),
            endpointing_ms=self.endpointing_ms,
            utterance_end_ms=self.utterance_end_ms,
        )
        log_event("deepgram_stream_open", sample_rate=self.sample_rate)

    def _on_error(self, _ws, err):
        self.error = str(err)
        with self.lock:
            if self.disconnected_since is None:
                self.disconnected_since = perf_counter()
        self.connected_event.clear()
        log_event("deepgram_stream_error", error=self.error)

    def _on_close(self, _ws, status_code, message):
        with self.lock:
            if self.disconnected_since is None:
                self.disconnected_since = perf_counter()
        self.connected_event.clear()
        self.stop_event.set()
        log_event(
            "deepgram_stream_close",
            status_code=status_code,
            message=message,
            keepalive_thread_id=(self.keepalive_thread.ident if self.keepalive_thread else None),
        )

    def _queue_chunk(self, chunk_bytes):
        self.pending_chunks.append(chunk_bytes)
        if len(self.pending_chunks) > self.max_pending_chunks:
            self.pending_chunks = self.pending_chunks[-self.max_pending_chunks :]

    def _flush_pending_chunks(self):
        with self.lock:
            if not self.connected_event.is_set() or not self.pending_chunks:
                return
            queued = self.pending_chunks
            self.pending_chunks = []
        for chunk_bytes in queued:
            self.send_pcm16(chunk_bytes)

    def _send_keepalive_if_idle(self, now):
        with self.lock:
            idle_s = now - self.last_audio_sent_at
            ws_app = self.ws
        if idle_s < 3.0 or not ws_app:
            return False
        try:
            ws_app.send(json.dumps({"type": "KeepAlive"}))
            with self.lock:
                # A successful keepalive starts a new idle interval.
                self.last_audio_sent_at = now
            log_event(
                "deepgram_keepalive_sent",
                idle_s=round(idle_s, 1),
                keepalive_thread_id=threading.get_ident(),
            )
            return True
        except Exception as e:
            self.error = str(e)
            with self.lock:
                if self.disconnected_since is None:
                    self.disconnected_since = perf_counter()
            self.connected_event.clear()
            log_event("deepgram_stream_keepalive_error", error=self.error)
            return False

    def _start_keepalive(self):
        if self.keepalive_thread and self.keepalive_thread.is_alive():
            return

        def _loop():
            while not self.stop_event.is_set():
                time.sleep(0.5)
                if self.stop_event.is_set():
                    break

                now = perf_counter()

                # ── Heartbeat log every 5s ──
                if now - self._heartbeat_last >= 5.0:
                    with self.lock:
                        disc = self.disconnected_since
                    disc_s = round(now - disc, 1) if disc is not None else None
                    log_event(
                        "deepgram_worker_heartbeat",
                        connected=self.connected_event.is_set(),
                        disconnected_s=disc_s,
                    )
                    self._heartbeat_last = now

                # ── Only attempt KeepAlive when the socket is actually open ──
                if not self.connected_event.is_set():
                    continue

                # ── Hard-fail diagnostic: audio flowing but zero Results back ──
                if (
                    not self._no_transcripts_warned
                    and self.bytes_sent_total > 1_000_000
                    and self.transcript_events_received == 0
                ):
                    self._no_transcripts_warned = True
                    log_event(
                        "DEEPGRAM_NOT_EMITTING_TRANSCRIPTS",
                        bytes_sent_total=self.bytes_sent_total,
                        transcript_events_received=0,
                        sample_rate=self.sample_rate,
                        hint="Check encoding/sample_rate/channels. Deepgram accepted audio but returned no Results.",
                    )

                # Keepalive only after a genuine audio gap. The helper resets
                # the activity clock after sending, preventing loop-tick bursts.
                self._send_keepalive_if_idle(now)

        self.keepalive_thread = threading.Thread(
            target=_loop, daemon=True, name="deepgram-keepalive"
        )
        self.keepalive_thread.start()

    def _on_message(self, _ws, message):
        try:
            payload = json.loads(message)
        except Exception:
            return

        if self.first_result_at is None:
            self.first_result_at = perf_counter()

        # ── Raw message capture (first 15 frames) for parsing-layer diagnosis ──
        if self._raw_messages_logged < 15:
            self._raw_messages_logged += 1
            log_event(
                "deepgram_raw_message",
                n=self._raw_messages_logged,
                msg_type=payload.get("type") if isinstance(payload, dict) else "non-dict",
                preview=(message[:400] if isinstance(message, str) else repr(message)[:400]),
            )

        if payload.get("type") == "UtteranceEnd":
            with self.lock:
                self.turn_boundary_sequence += 1
            transcript_queue.put("")
            log_event("deepgram_utterance_end", sequence=self.turn_boundary_sequence)
            return

        if payload.get("type") == "Results":
            self.transcript_events_received += 1
            channel = payload.get("channel") or {}
            alternatives = channel.get("alternatives") or []
            is_final = bool(payload.get("is_final"))
            speech_final = bool(payload.get("speech_final"))
            # Emit signal on every is_final frame regardless of transcript content.
            # Previously this was inside the `if not formatted: return` guard,
            # which meant empty-transcript is_final frames never unblocked close().
            if is_final:
                self.final_flush_event.set()
            if not alternatives:
                log_event(
                    "deepgram_result_no_alternatives",
                    n=self.transcript_events_received,
                    is_final=is_final,
                )
                return
            alt = alternatives[0]
            # Log parsed result metadata to diagnose empty-transcript root cause.
            transcript_raw = (alt.get("transcript") or "").strip()
            words_count = len(alt.get("words") or [])
            log_event(
                "deepgram_result_parsed",
                n=self.transcript_events_received,
                is_final=is_final,
                speech_final=speech_final,
                transcript_len=len(transcript_raw),
                words_count=words_count,
                confidence=alt.get("confidence"),
                bytes_sent_total=self.bytes_sent_total,
            )
            transcript = transcript_raw
            with self.lock:
                previous_speaker = (
                    self.final_words[-1].get("speaker") if self.final_words else None
                )
                previous_word = self.final_words[-1].get("word") if self.final_words else None
                previous_interim = self.interim_text
            finalized_words = _finalized_deepgram_words(
                alt.get("words"),
                previous_speaker=previous_speaker,
                previous_word=previous_word,
                interim_text=previous_interim,
            )
            if not transcript and not finalized_words:
                return
            with self.lock:
                if is_final:
                    self.final_words.extend(finalized_words)
                    self.interim_text = ""
                else:
                    self.interim_text = transcript
            if is_final and finalized_words:
                reason_counts = {}
                for word in finalized_words:
                    for reason in word.get("flag_reasons") or []:
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
                log_event(
                    "deepgram_finalized_words",
                    words_count=len(finalized_words),
                    flag_reason_counts=reason_counts,
                )
            if speech_final:
                log_event("deepgram_speech_final", action="append_only")
            # Push to module-level transcript_queue so the UI can drain it
            # and trigger st.rerun() without blocking the main thread.
            transcript_queue.put(transcript)

    def _start_audio_sender(self):
        """Dedicated background thread: drains audio_queue and sends PCM to Deepgram.

        Decouples the Streamlit render cycle from the websocket send path entirely.
        """
        if self.audio_sender_thread and self.audio_sender_thread.is_alive():
            return

        def _sender_loop():
            while not self.stop_event.is_set():
                try:
                    # Block briefly so we don't busy-spin when the queue is empty
                    chunk = audio_queue.get(timeout=0.05)
                except Empty:
                    time.sleep(0.01)
                    continue
                except Exception:
                    time.sleep(0.01)
                    continue

                if chunk is None:  # sentinel → stop
                    break
                try:
                    self.send_pcm16(chunk)
                except Exception as e:
                    log_event("deepgram_audio_sender_error", error=str(e))
                    time.sleep(0.01)

        self.audio_sender_thread = threading.Thread(target=_sender_loop, daemon=True)
        self.audio_sender_thread.start()

    def start(self):
        if websocket is None:
            self.error = "websocket_client_missing"
            log_event("deepgram_stream_unavailable", reason=self.error)
            return

        ws_url = self._build_url()
        headers = [f"Authorization: Token {self.api_key}"]
        self.ws = websocket.WebSocketApp(
            ws_url,
            header=headers,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        ws_app = self.ws
        if ws_app is None:
            self.error = "websocket_init_failed"
            log_event("deepgram_stream_unavailable", reason=self.error)
            return

        self.thread = threading.Thread(
            target=lambda: ws_app.run_forever(ping_interval=15, ping_timeout=8),
            daemon=True,
        )
        self.thread.start()
        self._start_keepalive()

    @property
    def is_active(self):
        with self.lock:
            return (
                self.ws is not None
                and self.connected_event.is_set()
                and self.error is None
                and not self.stop_event.is_set()
            )

    @property
    def is_connecting(self):
        with self.lock:
            return (
                self.ws is not None
                and not self.connected_event.is_set()
                and self.thread is not None
                and self.thread.is_alive()
                and not self.stop_event.is_set()
            )

    def send_pcm16(self, chunk_bytes):
        ws_app = self.ws
        if not chunk_bytes:
            return
        with self.lock:
            if self.stop_event.is_set():
                return
            if not ws_app:
                self._queue_chunk(chunk_bytes)
                return
            if not self.connected_event.is_set():
                self._queue_chunk(chunk_bytes)
                return
            try:
                ws_app.send(chunk_bytes, opcode=2)
                self.last_audio_sent_at = perf_counter()
                self.bytes_sent_total += len(chunk_bytes)
                # Log once per 10 KB milestone to avoid flooding the event log.
                _LOG_BYTES_MILESTONE = 10_000
                if self.bytes_sent_total - self._bytes_logged_milestone >= _LOG_BYTES_MILESTONE:
                    log_event(
                        "deepgram_audio_milestone",
                        bytes_sent_total=self.bytes_sent_total,
                    )
                    self._bytes_logged_milestone = (
                        (self.bytes_sent_total // _LOG_BYTES_MILESTONE) * _LOG_BYTES_MILESTONE
                    )
            except Exception as e:
                self.error = str(e)
                self.connected_event.clear()
                self._queue_chunk(chunk_bytes)
                log_event("deepgram_stream_send_error", error=self.error)

    def snapshot(self):
        with self.lock:
            final_words = list(self.final_words)
            interim_text = (self.interim_text or "").strip()
        final_text = _render_deepgram_transcript(final_words, annotate=False)
        live = "\n".join([part for part in [final_text, interim_text] if part]).strip()
        return {
            "final": final_text,
            "final_words": final_words,
            "interim": interim_text,
            "live": live,
            "latency_s": (
                (self.first_result_at - self.started_at)
                if self.first_result_at is not None
                else None
            ),
            "error": self.error,
            "bytes_sent_total": self.bytes_sent_total,
            "chunks_received": self.chunks_received,
            "transcript_events_received": self.transcript_events_received,
            "sample_rate": self.sample_rate,
            "turn_boundary_sequence": self.turn_boundary_sequence,
        }

    def close(self):
        total_s = perf_counter() - self.started_at
        self.stop_event.set()
        self.connected_event.clear()
        try:
            if self.ws:
                # Step 1: Tell Deepgram to flush and deliver final transcript frames.
                try:
                    self.ws.send(json.dumps({"type": "CloseStream"}))
                    log_event("deepgram_closestream_sent")
                except Exception:
                    pass
                # Step 2: Wait up to 3 s for the last is_final=True message before
                # tearing down the transport.  Without this wait, we close the
                # socket before Deepgram can send back the final results.
                got_final = self.final_flush_event.wait(timeout=3.0)
                log_event("deepgram_final_flush_wait", got_final=got_final)
                # Step 3: Now it is safe to close the underlying transport.
                try:
                    self.ws.close()
                except Exception:
                    pass
        finally:
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=1.2)
            if self.keepalive_thread and self.keepalive_thread.is_alive():
                self.keepalive_thread.join(timeout=1.2)
            with ACTIVE_DEEPGRAM_STREAMERS_LOCK:
                ACTIVE_DEEPGRAM_STREAMERS.discard(self)
            log_event(
                "deepgram_stream_stopped",
                keepalive_thread_id=(self.keepalive_thread.ident if self.keepalive_thread else None),
                keepalive_alive=bool(self.keepalive_thread and self.keepalive_thread.is_alive()),
            )
        return total_s

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
    if "session_mode" not in st.session_state:
        st.session_state.session_mode = "setup"
    if "live_feedback_snapshot" not in st.session_state:
        st.session_state.live_feedback_snapshot = None
    if "live_feedback_frozen_until" not in st.session_state:
        st.session_state.live_feedback_frozen_until = 0.0
    if "audio_source" not in st.session_state:
        st.session_state.audio_source = "Record"
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
    if "replicate_transcript" not in st.session_state:
        st.session_state.replicate_transcript = ""
    if "deepgram_live_transcript" not in st.session_state:
        st.session_state.deepgram_live_transcript = ""
    if "deepgram_final_transcript" not in st.session_state:
        st.session_state.deepgram_final_transcript = ""
    if "deepgram_final_words" not in st.session_state:
        st.session_state.deepgram_final_words = []
    if "analysis_transcript_choice" not in st.session_state:
        st.session_state.analysis_transcript_choice = "Replicate"
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
    if "recording_elapsed_seconds" not in st.session_state:
        st.session_state.recording_elapsed_seconds = 0
    if "active_audio_signature" not in st.session_state:
        st.session_state.active_audio_signature = None
    if "openai_system_prompt" not in st.session_state:
        st.session_state.openai_system_prompt = load_persisted_openai_prompt()
    if "analysis_timing" not in st.session_state:
        st.session_state.analysis_timing = {
            "started_at": None,
            "audio_prep_s": None,
            "transcription_s": None,
            "openai_s": None,
            "total_s": None,
        }
    if "deepgram_streamer" not in st.session_state:
        st.session_state.deepgram_streamer = None
    if "deepgram_started_at" not in st.session_state:
        st.session_state.deepgram_started_at = None
    if "deepgram_streaming_latency" not in st.session_state:
        st.session_state.deepgram_streaming_latency = None
    if "deepgram_total_time" not in st.session_state:
        st.session_state.deepgram_total_time = None
    if "last_recorder_event_t" not in st.session_state:
        st.session_state.last_recorder_event_t = 0
    if "last_recorder_event_id" not in st.session_state:
        st.session_state.last_recorder_event_id = ""
    if "deepgram_accumulated_transcript" not in st.session_state:
        st.session_state.deepgram_accumulated_transcript = ""
    if "deepgram_accumulated_words" not in st.session_state:
        st.session_state.deepgram_accumulated_words = []
    if "deepgram_last_restart_at" not in st.session_state:
        st.session_state.deepgram_last_restart_at = None
    if "deepgram_stop_requested_at" not in st.session_state:
        st.session_state.deepgram_stop_requested_at = None
    # Non-blocking streaming guards (used by background thread pattern)
    if "streaming_active" not in st.session_state:
        st.session_state.streaming_active = False
    if "worker_started" not in st.session_state:
        st.session_state.worker_started = False
    if "transcribing" not in st.session_state:
        st.session_state.transcribing = False
    if "transcribe_future" not in st.session_state:
        st.session_state.transcribe_future = None
    if "transcribe_started_at" not in st.session_state:
        st.session_state.transcribe_started_at = None
    if "transcribe_error" not in st.session_state:
        st.session_state.transcribe_error = ""
    if "analysing" not in st.session_state:
        st.session_state.analysing = False
    if "analysis_future" not in st.session_state:
        st.session_state.analysis_future = None
    if "analysis_started_at" not in st.session_state:
        st.session_state.analysis_started_at = None
    if "analysis_error" not in st.session_state:
        st.session_state.analysis_error = ""
    if "low_confidence_threshold" not in st.session_state:
        st.session_state.low_confidence_threshold = LOW_CONFIDENCE_THRESHOLD
    if "flag_duplicate_tokens" not in st.session_state:
        st.session_state.flag_duplicate_tokens = True
    if "flag_interim_churn" not in st.session_state:
        st.session_state.flag_interim_churn = False
    if "debug_log_autorefresh" not in st.session_state:
        st.session_state.debug_log_autorefresh = False
    if "debug_log_important_only" not in st.session_state:
        st.session_state.debug_log_important_only = True
    if "debug_log_lines" not in st.session_state:
        st.session_state.debug_log_lines = 80
    if "turn_analysis_worker" not in st.session_state:
        st.session_state.turn_analysis_worker = IncrementalTurnAnalysisWorker()
    if "live_turn_analysis_enabled" not in st.session_state:
        st.session_state.live_turn_analysis_enabled = True
    if "candidate_speaker_override" not in st.session_state:
        st.session_state.candidate_speaker_override = "Auto"
    if "turn_silence_timeout_s" not in st.session_state:
        st.session_state.turn_silence_timeout_s = DEFAULT_TURN_SILENCE_TIMEOUT_S
    if "turn_min_words" not in st.session_state:
        st.session_state.turn_min_words = DEFAULT_TURN_MIN_WORDS
    if "turn_analysis_cap" not in st.session_state:
        st.session_state.turn_analysis_cap = DEFAULT_TURN_ANALYSIS_CAP
    if "turn_max_flagged_ratio" not in st.session_state:
        st.session_state.turn_max_flagged_ratio = DEFAULT_TURN_MAX_FLAGGED_RATIO
    if "turn_closed_keys" not in st.session_state:
        st.session_state.turn_closed_keys = set()
    if "turn_ids_by_key" not in st.session_state:
        st.session_state.turn_ids_by_key = {}
    if "next_turn_id" not in st.session_state:
        st.session_state.next_turn_id = 0
    if "live_processed_turns" not in st.session_state:
        st.session_state.live_processed_turns = {}
    if "live_turn_buffers" not in st.session_state:
        st.session_state.live_turn_buffers = {}
    if "live_turn_order" not in st.session_state:
        st.session_state.live_turn_order = []
    if "live_transcript_scope" not in st.session_state:
        st.session_state.live_transcript_scope = "Everyone"
    if "live_transcript_view" not in st.session_state:
        st.session_state.live_transcript_view = "Turns"
    if "live_transcript_history_limit" not in st.session_state:
        st.session_state.live_transcript_history_limit = 8
    if "live_priority_criterion_filter" not in st.session_state:
        st.session_state.live_priority_criterion_filter = "All"
    if "live_feedback_statuses" not in st.session_state:
        st.session_state.live_feedback_statuses = {}
    if "turn_gated_keys" not in st.session_state:
        st.session_state.turn_gated_keys = set()
    if "ignored_speaker_segment_keys" not in st.session_state:
        st.session_state.ignored_speaker_segment_keys = set()
    if "turn_boundary_sequence_seen" not in st.session_state:
        st.session_state.turn_boundary_sequence_seen = 0
    if "use_legacy_full_transcript_analysis" not in st.session_state:
        st.session_state.use_legacy_full_transcript_analysis = False


def _current_session_mode():
    if st.session_state.get("is_recording") or st.session_state.get("streaming_active"):
        return "active"
    if recording_ready() or st.session_state.get("live_processed_turns"):
        return "review"
    return "setup"


def _word_diff_markup(said, target):
    said_words = str(said or "").split()
    target_words = str(target or "").split()
    matcher = difflib.SequenceMatcher(a=said_words, b=target_words, autojunk=False)
    said_markup = []
    target_markup = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            said_markup.extend(html.escape(word) for word in said_words[i1:i2])
            target_markup.extend(html.escape(word) for word in target_words[j1:j2])
        elif tag in {"delete", "replace"}:
            said_markup.extend(
                f'<del>{html.escape(word)}</del>' for word in said_words[i1:i2]
            )
        if tag in {"insert", "replace"}:
            target_markup.extend(
                f'<strong>{html.escape(word)}</strong>' for word in target_words[j1:j2]
            )
    return " ".join(said_markup), " ".join(target_markup)


def _priority_tag(category):
    category = str(category or "").lower()
    if "pronoun" in category or "relative" in category:
        return "relative pronoun"
    if "preposition" in category:
        return "preposition"
    if "collocation" in category or "word" in category:
        return "word choice"
    if "coher" in category or "cohes" in category:
        return "clause structure"
    return "grammar"


def _selected_candidate_speaker(words):
    durations = _speaker_durations(words)
    override = st.session_state.get("candidate_speaker_override", "Auto")
    if override != "Auto":
        try:
            return int(override)
        except (TypeError, ValueError):
            return override
    return max(durations, key=durations.get) if durations else None


def sync_live_turn_analysis(flush=False):
    """Persist closed all-speaker turns; queue clean candidate turns only."""
    words = st.session_state.get("deepgram_final_words") or []
    candidate = _selected_candidate_speaker(words)
    if candidate is None:
        return
    streamer = st.session_state.get("deepgram_streamer")
    audio_position = (
        perf_counter() - streamer.started_at if streamer else float("inf") if flush else 0.0
    )
    turns = _processed_turns(
        words,
        candidate,
        audio_position_s=audio_position,
        force_last_close=flush,
    )
    worker = st.session_state.turn_analysis_worker
    if not st.session_state.get("live_turn_analysis_enabled", True):
        worker.stop_pending()
    elif not worker.accepting and st.session_state.get("is_recording"):
        worker.resume()
    prior_turn = None
    for turn in turns:
        if not turn["closed"]:
            continue
        identity = f'{turn.get("speaker")}:{float(turn.get("start") or 0):.3f}'
        previous_words = st.session_state.live_turn_buffers.get(identity, [])
        delta_words = _turn_delta_words(previous_words, turn["words"])
        st.session_state.live_turn_buffers[identity] = list(turn["words"])
        if not delta_words:
            continue
        delta_text = " ".join(word["text"] for word in delta_words).strip()
        if not delta_text:
            continue
        turn = {
            **turn,
            "key": f'{identity}:{hashlib.sha256(delta_text.lower().encode("utf-8")).hexdigest()[:16]}',
            "words": delta_words,
            "text": delta_text,
            "clean_text": delta_text,
            "t_start": delta_words[0].get("start"),
            "t_end": delta_words[-1].get("end"),
            "start": delta_words[0].get("start"),
            "end": delta_words[-1].get("end"),
        }
        turn["id"] = st.session_state.next_turn_id
        st.session_state.next_turn_id += 1
        turn["examiner_context"] = (
            prior_turn.get("clean_text", "")
            if prior_turn and not prior_turn.get("is_candidate")
            else ""
        )
        st.session_state.live_processed_turns[turn["key"]] = turn
        st.session_state.live_turn_order.append(turn["key"])
        st.session_state.turn_ids_by_key[turn["key"]] = turn["id"]
        st.session_state.turn_closed_keys.add(turn["key"])
        worker.stats["turns_closed"] += 1
        prior_turn = turn
        if not turn["is_candidate"]:
            worker.stats["examiner_turns_ignored"] += 1
            log_event("speaker_turn_ignored", speaker=turn["speaker"], start=turn["t_start"])
            continue
        duration = None
        if isinstance(turn.get("t_start"), (int, float)) and isinstance(turn.get("t_end"), (int, float)):
            duration = round(turn["t_end"] - turn["t_start"], 3)
        log_event(
            "candidate_turn_closed",
            turn_key=turn["key"],
            candidate_speaker=candidate,
            words=len(turn["clean_text"].split()),
            duration_s=duration,
            text_preview=turn["text"][:200],
        )
        if not st.session_state.get("live_turn_analysis_enabled", True):
            st.session_state.turn_gated_keys.add(turn["key"])
            worker.stats["turns_gated_out"] += 1
            log_event(
                "turn_analysis_gated",
                turn_key=turn["key"],
                reason="disabled",
                words=len(turn["clean_text"].split()),
            )
            continue
        word_count = len(turn["clean_text"].split())
        flagged = sum(1 for word in turn["words"] if word.get("low_confidence"))
        total = len(turn["words"])
        flagged_ratio = flagged / total if total else 1.0
        if word_count < int(st.session_state.get("turn_min_words", DEFAULT_TURN_MIN_WORDS)):
            reason = "minimum_words"
        elif flagged_ratio > float(st.session_state.get("turn_max_flagged_ratio", DEFAULT_TURN_MAX_FLAGGED_RATIO)):
            reason = "asr_confidence"
        elif len(worker.submitted) >= int(st.session_state.get("turn_analysis_cap", DEFAULT_TURN_ANALYSIS_CAP)):
            reason = "session_cap"
        else:
            reason = ""
        if reason:
            st.session_state.turn_gated_keys.add(turn["key"])
            worker.stats["turns_gated_out"] += 1
            log_event("turn_analysis_gated", turn_key=turn["key"], reason=reason, words=word_count, flagged_ratio=round(flagged_ratio, 3))
        else:
            worker.submit(turn)


def reset_live_turn_analysis():
    existing = st.session_state.get("turn_analysis_worker")
    if existing:
        existing.stop_pending()
    st.session_state.turn_analysis_worker = IncrementalTurnAnalysisWorker()
    st.session_state.turn_closed_keys = set()
    st.session_state.turn_ids_by_key = {}
    st.session_state.next_turn_id = 0
    st.session_state.live_processed_turns = {}
    st.session_state.live_turn_buffers = {}
    st.session_state.live_turn_order = []
    st.session_state.turn_gated_keys = set()
    st.session_state.live_manual_priorities = set()
    st.session_state.live_feedback_statuses = {}
    st.session_state.ignored_speaker_segment_keys = set()
    st.session_state.turn_boundary_sequence_seen = 0


def render_audio_recorder(compact=False):
    return audio_recorder_component(
        compact=compact,
        key=f"audio_recorder_{st.session_state.recorder_nonce}",
    )


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


def _deepgram_has_key():
    return bool(os.environ.get("DEEPGRAM_API_KEY"))


def get_analysis_transcript(state, choice):
    """Return analysis-eligible transcript text and an optional availability message."""
    if choice == "Replicate":
        return (state.get("replicate_transcript") or "").strip(), ""
    if choice == "Deepgram":
        final_text = (state.get("deepgram_final_transcript") or "").strip()
        if final_text:
            return final_text, ""
        if (state.get("deepgram_live_transcript") or "").strip():
            return "", "Deepgram is finalizing the recording. Analysis will be available when final text arrives."
        return "", "No final Deepgram transcript is available."
    return "", "Select a transcript source for analysis."


def start_deepgram_stream(sample_rate):
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        st.session_state.deepgram_streamer = None
        return

    with ACTIVE_DEEPGRAM_STREAMERS_LOCK:
        stale_streamers = list(ACTIVE_DEEPGRAM_STREAMERS)
    for stale_streamer in stale_streamers:
        if stale_streamer is st.session_state.get("deepgram_streamer"):
            continue
        log_event("deepgram_stale_streamer_closing")
        try:
            stale_streamer.close()
        except Exception as error:
            log_event("deepgram_stale_streamer_close_error", error=str(error))
    
    # Close existing streamer if any, and save its final segments
    old_streamer = st.session_state.get("deepgram_streamer")
    if old_streamer:
        if old_streamer.is_active or old_streamer.is_connecting:
            log_event("deepgram_stream_reuse_blocked", reason="existing_streamer_live")
            return
        try:
            snap = old_streamer.snapshot()
            old_final = snap.get("final") or ""
            if old_final:
                acc = st.session_state.get("deepgram_accumulated_transcript", "")
                st.session_state.deepgram_accumulated_transcript = "\n".join([p for p in [acc, old_final] if p]).strip()
            old_words = snap.get("final_words") or []
            if old_words:
                st.session_state.deepgram_accumulated_words.extend(old_words)
            old_streamer.close()
        except Exception:
            pass

    try:
        streamer = DeepgramStreamingClient(api_key=api_key, sample_rate=sample_rate or 16000)
        streamer.start()
        with ACTIVE_DEEPGRAM_STREAMERS_LOCK:
            ACTIVE_DEEPGRAM_STREAMERS.add(streamer)
        st.session_state.deepgram_streamer = streamer
        st.session_state.deepgram_last_restart_at = perf_counter()
        st.session_state.deepgram_started_at = perf_counter()
        st.session_state.deepgram_streaming_latency = None
        st.session_state.deepgram_total_time = None
        st.session_state.streaming_active = True
        st.session_state.worker_started = True
        # Drain any stale transcript fragments from a previous session
        try:
            while True:
                transcript_queue.get_nowait()
        except Empty:
            pass
        log_event("deepgram_stream_start", sample_rate=sample_rate)
    except Exception as e:
        st.session_state.deepgram_streamer = None
        st.session_state.streaming_active = False
        st.session_state.worker_started = False
        log_event("deepgram_stream_start_error", error=str(e))


def check_deepgram_health():
    """Hard health check: if socket has been disconnected for >5s while recording,
    tear down the old client entirely and build a fresh one.

    Call this on every Streamlit rerun while is_recording=True.
    """
    if not st.session_state.get("is_recording"):
        return

    streamer = st.session_state.get("deepgram_streamer")
    if not streamer:
        return

    # Measure how long the socket has been down
    with streamer.lock:
        disc = streamer.disconnected_since
    if disc is None:
        return  # still connected — nothing to do

    down_s = perf_counter() - disc
    if down_s < 5.0:
        return  # give it a moment before forcing restart

    # Socket has been down >5s during active recording → full engine restart
    sample_rate = streamer.sample_rate
    log_event(
        "deepgram_engine_restart",
        reason="health_check_disconnected",
        disconnected_s=round(down_s, 1),
        sample_rate=sample_rate,
    )
    try:
        streamer.close()
    except Exception:
        pass
    st.session_state.deepgram_streamer = None

    # Drain stale queue entries so the fresh session starts clean
    try:
        while True:
            transcript_queue.get_nowait()
    except Empty:
        pass

    log_event("websocket_recreated", sample_rate=sample_rate)
    start_deepgram_stream(sample_rate)
    log_event("audio_generator_rebuilt", sample_rate=sample_rate)


def sync_deepgram_transcript_state():
    # Hard health check first — restarts the engine if socket has been dead >5s
    check_deepgram_health()

    streamer = st.session_state.get("deepgram_streamer")
    if not streamer:
        accumulated_words = st.session_state.get("deepgram_accumulated_words") or []
        st.session_state.deepgram_final_transcript = _render_deepgram_transcript(
            accumulated_words, annotate=False
        )
        st.session_state.deepgram_live_transcript = st.session_state.deepgram_final_transcript
        st.session_state.deepgram_final_words = list(accumulated_words)
        return
    
    snap = streamer.snapshot()
    accumulated_words = st.session_state.get("deepgram_accumulated_words") or []
    full_final_words = list(accumulated_words) + list(snap.get("final_words") or [])
    interim = snap.get("interim") or ""
    full_final = _render_deepgram_transcript(full_final_words, annotate=False)

    st.session_state.deepgram_live_transcript = _render_deepgram_transcript(
        full_final_words, annotate=False, interim_text=interim
    )
    st.session_state.deepgram_final_transcript = full_final
    st.session_state.deepgram_final_words = full_final_words
    
    if st.session_state.deepgram_streaming_latency is None and snap.get("latency_s") is not None:
        st.session_state.deepgram_streaming_latency = snap.get("latency_s")
        log_event("deepgram_first_result", latency_s=st.session_state.deepgram_streaming_latency)


def send_deepgram_chunk(chunk_bytes):
    """Send a PCM-16 audio chunk to Deepgram.

    Calls send_pcm16 directly — it is already protected by a threading.Lock
    and only touches the kernel socket buffer (microseconds), so it does NOT
    block the Streamlit render cycle in any meaningful way.
    The audio_queue / _start_audio_sender pattern was removed because it caused
    multiple client instances to compete for the same shared global queue,
    resulting in broken-pipe errors and the watchdog firing prematurely.
    """
    streamer = st.session_state.get("deepgram_streamer")
    if not streamer:
        return

    # If the streamer is dead but we're recording, restart it before sending
    if st.session_state.get("is_recording") and not streamer.is_active:
        if not streamer.is_connecting:
            now = perf_counter()
            last_restart_at = st.session_state.get("deepgram_last_restart_at")
            restart_allowed = (
                last_restart_at is None
                or (now - float(last_restart_at)) >= DEEPGRAM_RESTART_MIN_INTERVAL_S
            )
            if restart_allowed:
                log_event("deepgram_restarting_dead_stream", last_error=streamer.error)
                start_deepgram_stream(streamer.sample_rate)
                streamer = st.session_state.get("deepgram_streamer")
                if not streamer:
                    return

    if not chunk_bytes:
        log_event("deepgram_chunk_empty", bytes_len=0)
        return
    try:
        streamer.send_pcm16(chunk_bytes)
    except Exception as e:
        log_event("deepgram_chunk_send_exception", error=str(e))


def stop_deepgram_stream():
    streamer = st.session_state.get("deepgram_streamer")
    if not streamer:
        return
    try:
        st.session_state.deepgram_total_time = streamer.close()
    except Exception as e:
        log_event("deepgram_stream_stop_error", error=str(e))
    finally:
        sync_deepgram_transcript_state()
        sync_live_turn_analysis(flush=True)
        st.session_state.turn_analysis_worker.close_input()
        # ── Critical: persist the captured final into accumulated BEFORE the
        # streamer is set to None.  On the next rerun, sync_deepgram_transcript_state
        # finds no streamer and falls back to deepgram_accumulated_transcript.
        # Without this, every Stop wipes the live transcript.
        final = (st.session_state.get("deepgram_final_transcript") or "").strip()
        if final:
            st.session_state.deepgram_accumulated_transcript = final
            st.session_state.deepgram_accumulated_words = list(
                st.session_state.get("deepgram_final_words") or []
            )
            log_event("deepgram_transcript_persisted", final_chars=len(final))
        st.session_state.deepgram_streamer = None
        st.session_state.streaming_active = False
        st.session_state.worker_started = False


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
    st.session_state.replicate_transcript = ""
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
    st.session_state.deepgram_live_transcript = ""
    st.session_state.deepgram_final_transcript = ""
    st.session_state.deepgram_final_words = []
    st.session_state.deepgram_accumulated_transcript = ""
    st.session_state.deepgram_accumulated_words = []
    st.session_state.deepgram_streaming_latency = None
    st.session_state.deepgram_total_time = None
    reset_live_turn_analysis()


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


def deepgram_sidecar_path(audio_path):
    return f"{os.path.splitext(audio_path)[0]}.deepgram.json"


def save_deepgram_word_records(audio_path, words):
    if not audio_path:
        return
    try:
        with open(deepgram_sidecar_path(audio_path), "w", encoding="utf-8") as f:
            json.dump({"version": 1, "words": list(words or [])}, f, ensure_ascii=False)
        log_event("deepgram_word_records_saved", audio_path=audio_path, words_count=len(words or []))
    except Exception as e:
        log_event("deepgram_word_records_save_error", audio_path=audio_path, error=str(e))


def load_deepgram_word_records(audio_path):
    if not audio_path:
        return []
    try:
        with open(deepgram_sidecar_path(audio_path), "r", encoding="utf-8") as f:
            payload = json.load(f)
        words = payload.get("words", []) if isinstance(payload, dict) else payload
        return [dict(record) for record in words if isinstance(record, dict)]
    except FileNotFoundError:
        return []
    except Exception as e:
        log_event("deepgram_word_records_load_error", audio_path=audio_path, error=str(e))
        return []


def restore_deepgram_word_records(audio_path):
    words = load_deepgram_word_records(audio_path)
    st.session_state.deepgram_final_words = words
    st.session_state.deepgram_accumulated_words = list(words)
    rendered = _render_deepgram_transcript(words, annotate=False)
    st.session_state.deepgram_final_transcript = rendered
    st.session_state.deepgram_live_transcript = rendered
    return words


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
        return audio_path
    log_event("unknown_audio_source", source=source)
    st.error("Unknown audio source.")
    st.stop()


def get_active_audio_for_background(source, uploaded_bytes, uploaded_name, recorded_path, loaded_path):
    if source == "Upload":
        if not uploaded_bytes or not uploaded_name:
            raise RuntimeError("Please upload a .wav or .m4a file first.")
        return load_and_convert_audio(uploaded_bytes, uploaded_name)
    if source == "Record":
        if not recorded_path:
            raise RuntimeError("Please record audio first.")
        if not os.path.exists(recorded_path):
            raise RuntimeError("Recorded audio file not found. Please record again.")
        return recorded_path
    if source == "Load":
        if not loaded_path:
            raise RuntimeError("Please select a saved recording first.")
        if not os.path.exists(loaded_path):
            raise RuntimeError("Selected saved recording was not found. Please select another file.")
        return convert_audio_path_for_analysis(loaded_path)
    raise RuntimeError("Unknown audio source.")


def run_transcription_background_job(
    *,
    source,
    uploaded_bytes,
    uploaded_name,
    recorded_path,
    loaded_path,
):
    run_started = perf_counter()
    log_event("replicate_background_started", source=source)
    timings = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "audio_prep_s": None,
        "transcription_s": None,
        "openai_s": None,
        "total_s": None,
    }
    audio_prep_started = perf_counter()
    audio_path = get_active_audio_for_background(
        source=source,
        uploaded_bytes=uploaded_bytes,
        uploaded_name=uploaded_name,
        recorded_path=recorded_path,
        loaded_path=loaded_path,
    )
    timings["audio_prep_s"] = perf_counter() - audio_prep_started
    transcription_started = perf_counter()
    replicate_output, transcript, diarization_text = run_transcription(audio_path)
    timings["transcription_s"] = perf_counter() - transcription_started
    timings["total_s"] = perf_counter() - run_started
    return {
        "replicate_output": replicate_output,
        "transcript": transcript or "",
        "diarization": diarization_text or "",
        "timings": timings,
    }


def run_error_detection_background(transcript, system_prompt, allow_rule_based_fallback=False):
    matches = []
    if not transcript:
        return {
            "issues": [],
            "feedback_source": "openai_unavailable",
        }

    api_key = os.environ.get("OPENAI_API_KEY")
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
        log_event(
            "openai_feedback_unavailable",
            failure_type="auth_missing_key",
            status_code=None,
            strict_openai_feedback=STRICT_OPENAI_FEEDBACK,
            allow_rule_based_fallback=allow_rule_based_fallback,
        )
        if STRICT_OPENAI_FEEDBACK and not allow_rule_based_fallback:
            return {"issues": [], "feedback_source": "openai_unavailable"}
        return {
            "issues": build_local_feedback_issues(transcript),
            "feedback_source": "local_rule_based_fallback",
        }

    system_msg = (system_prompt or DEFAULT_OPENAI_SYSTEM_PROMPT).strip()
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
        r = OPENAI_HTTP.post(
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
        log_event(
            "openai_call_response_json",
            top_level_keys=list(resp.keys()) if isinstance(resp, dict) else [],
            choices_count=len(resp.get("choices", [])) if isinstance(resp, dict) else 0,
        )

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
        return {
            "issues": matches if isinstance(matches, list) else [],
            "feedback_source": "openai",
        }
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
        if is_auth_failure and STRICT_OPENAI_FEEDBACK and not allow_rule_based_fallback:
            return {"issues": [], "feedback_source": "openai_unavailable"}
        return {
            "issues": build_local_feedback_issues(transcript),
            "feedback_source": "local_rule_based_fallback",
        }
    except Exception as e:
        log_event(
            "openai_call_exception",
            failure_type="temporary_api_failure",
            error=str(e),
            error_type=type(e).__name__,
        )
        return {
            "issues": build_local_feedback_issues(transcript),
            "feedback_source": "local_rule_based_fallback",
        }


def run_openai_analysis_background_job(transcript, system_prompt, allow_rule_based_fallback=False):
    run_started = perf_counter()
    issues_payload = run_error_detection_background(
        transcript,
        system_prompt=system_prompt,
        allow_rule_based_fallback=allow_rule_based_fallback,
    )
    return {
        "issues": issues_payload.get("issues") if isinstance(issues_payload, dict) else [],
        "feedback_source": (
            issues_payload.get("feedback_source")
            if isinstance(issues_payload, dict)
            else "unknown"
        ),
        "openai_s": perf_counter() - run_started,
    }


def run_incremental_synthesis_background_job(turn_results, system_prompt, allow_rule_based_fallback=False):
    completed = [
        item.get("analysis")
        for item in turn_results.values()
        if isinstance(item, dict) and item.get("status") == "complete" and isinstance(item.get("analysis"), dict)
    ]
    if not completed:
        return {"issues": [], "feedback_source": "incremental_no_completed_turns", "openai_s": 0.0}
    synthesis_prompt = (
        (system_prompt or DEFAULT_OPENAI_SYSTEM_PROMPT).strip()
        + "\n\nSynthesize the supplied compact per-turn analyses only. Do not assume access to a transcript or invent examples."
    )
    return run_openai_analysis_background_job(
        json.dumps({"turn_analyses": completed}, ensure_ascii=False),
        synthesis_prompt,
        allow_rule_based_fallback,
    )


def poll_background_jobs():
    should_rerun = False

    if st.session_state.get("transcribing"):
        future = st.session_state.get("transcribe_future")
        started_at = st.session_state.get("transcribe_started_at")
        if future is not None and future.done():
            try:
                result = future.result()
                timings = result.get("timings") or {}
                st.session_state.transcript = result.get("transcript") or ""
                st.session_state.replicate_transcript = result.get("transcript") or ""
                st.session_state.diarization = result.get("diarization") or ""
                st.session_state.replicate_output = result.get("replicate_output")
                st.session_state["transcript_source"] = "replicate"
                st.session_state.analysis_complete = False
                st.session_state.analysis_timing = {
                    "started_at": timings.get("started_at"),
                    "audio_prep_s": timings.get("audio_prep_s"),
                    "transcription_s": timings.get("transcription_s"),
                    "openai_s": st.session_state.get("analysis_timing", {}).get("openai_s"),
                    "total_s": timings.get("total_s"),
                }
                st.session_state.transcribe_error = ""
                log_event(
                    "transcription_background_completed",
                    transcript_chars=len(st.session_state.transcript or ""),
                    total_s=timings.get("total_s"),
                )
            except Exception as e:
                st.session_state.transcribe_error = f"Replicate diarization failed: {e}"
                log_event("transcription_background_failed", error=str(e))
            st.session_state.transcribing = False
            st.session_state.transcribe_future = None
            st.session_state.transcribe_started_at = None
            should_rerun = True
        elif started_at is not None and (perf_counter() - float(started_at)) > JOB_TIMEOUT_S:
            st.session_state.transcribing = False
            st.session_state.transcribe_future = None
            st.session_state.transcribe_started_at = None
            st.session_state.transcribe_error = (
                f"Transcription timed out after {JOB_TIMEOUT_S}s."
            )
            log_event("transcription_background_timeout", timeout_s=JOB_TIMEOUT_S)
            should_rerun = True

    if st.session_state.get("analysing"):
        future = st.session_state.get("analysis_future")
        started_at = st.session_state.get("analysis_started_at")
        if future is not None and future.done():
            try:
                result = future.result()
                issues = result.get("issues") if isinstance(result, dict) else []
                st.session_state.feedback = issues if isinstance(issues, list) else []
                st.session_state.feedback_source = (
                    result.get("feedback_source")
                    if isinstance(result, dict)
                    else "unknown"
                )
                st.session_state.analysis_complete = True
                timings = dict(st.session_state.get("analysis_timing") or {})
                timings["openai_s"] = result.get("openai_s") if isinstance(result, dict) else None
                timings["total_s"] = result.get("openai_s") if isinstance(result, dict) else None
                if "started_at" not in timings or not timings.get("started_at"):
                    timings["started_at"] = datetime.now(timezone.utc).isoformat()
                st.session_state.analysis_timing = timings
                st.session_state.analysis_error = ""
                log_event(
                    "analysis_background_completed",
                    issues_count=len(st.session_state.feedback or []),
                    total_s=timings.get("total_s"),
                )
            except Exception as e:
                st.session_state.analysis_error = f"OpenAI analysis failed: {e}"
                log_event("analysis_background_failed", error=str(e))
            st.session_state.analysing = False
            st.session_state.analysis_future = None
            st.session_state.analysis_started_at = None
            should_rerun = True
        elif started_at is not None and (perf_counter() - float(started_at)) > JOB_TIMEOUT_S:
            st.session_state.analysing = False
            st.session_state.analysis_future = None
            st.session_state.analysis_started_at = None
            st.session_state.analysis_error = (
                f"OpenAI analysis timed out after {JOB_TIMEOUT_S}s."
            )
            log_event("analysis_background_timeout", timeout_s=JOB_TIMEOUT_S)
            should_rerun = True

    if should_rerun:
        st.rerun()


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


def _extract_replicate_segments(output):
    if isinstance(output, dict):
        segments = output.get("segments")
        if isinstance(segments, list):
            return segments
    elif isinstance(output, list):
        return output
    return []


def _build_clean_diarized_transcript(output):
    segments = _extract_replicate_segments(output)
    merged = _merge_segments(segments)
    if not merged:
        return ""

    speaker_map = {}
    next_index = 0
    lines = []

    for seg in merged:
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        raw_speaker = (seg.get("speaker") or "UNKNOWN").strip() or "UNKNOWN"
        if raw_speaker not in speaker_map:
            speaker_map[raw_speaker] = f"SPEAKER_{next_index:02d}"
            next_index += 1

        speaker_label = speaker_map[raw_speaker]
        lines.append(f"[{speaker_label}]: {text}")

    return "\n".join(lines).strip()


def _build_confidence_annotated_diarized_transcript(output):
    segments = _extract_replicate_segments(output)
    display_segments = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        display_segment = dict(segment)
        display_segment["text"] = _mark_low_confidence_text(
            (segment.get("text") or "").strip(),
            _replicate_segment_confidence(segment),
        )
        display_segments.append(display_segment)

    return _build_clean_diarized_transcript({"segments": display_segments})


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
                "avg_logprob": getattr(seg, "avg_logprob", None),
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
    num_speakers = None  # autodetect speaker count
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
            if num_speakers is not None:
                input_payload["num_speakers"] = num_speakers
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    replicate.run,
                    model_id,
                    input=input_payload,
                )
                replicate_output = future.result(timeout=JOB_TIMEOUT_S)
        segments_count = (
            len(replicate_output.get("segments", []))
            if isinstance(replicate_output, dict) and isinstance(replicate_output.get("segments"), list)
            else 0
        )
        log_event(
            "replicate_call_result",
            inferred_http_status=200,
            response_type=type(replicate_output).__name__,
            top_level_keys=list(replicate_output.keys()) if isinstance(replicate_output, dict) else [],
            segments_count=segments_count,
        )
        log_event(
            "replicate_success",
            segments_count=segments_count,
            response_type=type(replicate_output).__name__,
        )
    except Exception as e:
        message = str(e)
        if isinstance(e, concurrent.futures.TimeoutError):
            message = f"Replicate diarization timed out after {JOB_TIMEOUT_S}s"
        status_code = extract_status_code_from_text(message)
        log_event(
            "replicate_call_error",
            status_code=status_code,
            error=message,
            error_type=type(e).__name__,
        )
        log_event(
            "replicate_error",
            status_code=status_code,
            error=message,
            error_type=type(e).__name__,
        )
        if "Invalid token" in message or "status: 401" in message:
            raise RuntimeError(
                "Replicate authentication failed. Set a valid REPLICATE_API_TOKEN in the environment."
            ) from e
        raise RuntimeError(f"Replicate diarization failed: {message}") from e

    transcript = _build_clean_diarized_transcript(replicate_output)
    if not transcript:
        raise RuntimeError("Replicate diarization returned no usable segments for transcript formatting.")
    diarization_text = _pretty_print_replicate(replicate_output)
    return replicate_output, transcript, diarization_text


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


def run_error_detection(transcript, emit_ui=True, system_prompt=None):
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

    system_msg = (system_prompt or DEFAULT_OPENAI_SYSTEM_PROMPT).strip()
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
        r = OPENAI_HTTP.post(
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
        log_event(
            "openai_call_response_json",
            top_level_keys=list(resp.keys()) if isinstance(resp, dict) else [],
            choices_count=len(resp.get("choices", [])) if isinstance(resp, dict) else 0,
        )

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
        # Use Streamlit theme variables for stable dual-mode behavior
        card_bg = "var(--secondary-background-color)"
        border_col = "var(--secondary-background-color)"
        text_col = "var(--text-color)"
        secondary_col = "var(--text-color)"
        json_bg = "var(--secondary-background-color)"

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
            .result-card * {{
                color: {text_col} !important;
            }}
            .clean-text-block {{
                background: transparent;
                border: 1px solid {border_col};
                border-radius: 10px;
                padding: 0.9rem;
                line-height: 1.55;
                color: {text_col} !important;
            }}
            .clean-text-block * {{
                color: {text_col} !important;
            }}
            .subtle-file {{
                color: {secondary_col};
                font-size: 0.88rem;
                margin-top: 0.35rem;
            }}
            div[data-testid=\"stMarkdownContainer\"] table,
            table {{
                background-color: {card_bg} !important;
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
                color: {text_col} !important;
            }}
            table td {{
                color: {text_col} !important;
            }}
            /* JSON block */
            .json-block {{
                background: {json_bg};
                color: {text_col} !important;
                padding: 0.8rem;
                border-radius: 8px;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, \"Roboto Mono\", \"Courier New\", monospace;
                max-height: 36vh;
                overflow: auto;
                border: 1px solid {border_col};
            }}
            div[data-testid="stJson"] {{
                background: {json_bg} !important;
                color: {text_col} !important;
                border: 1px solid {border_col};
                border-radius: 8px;
            }}
            div[data-testid="stJson"] * {{
                color: {text_col} !important;
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
            .deepgram-confidence-text {{
                line-height: 1.8;
            }}
            .deepgram-confidence-medium {{
                border-bottom: 1px dotted {secondary_col};
            }}
            .deepgram-confidence-low {{
                border-bottom: 1px dotted #c46613;
            }}
            .live-console-pane {{
                height: calc(100vh - 10rem);
                min-height: 18rem;
                overflow-y: auto;
                overscroll-behavior: contain;
                padding: 0.9rem;
                border: 1px solid {border_col};
                border-radius: 8px;
                background: {card_bg};
            }}
            .live-console-pane:focus-visible {{ outline: 3px solid #175cd3; outline-offset: 2px; }}
                    .live-feedback-pane {{ position: sticky; top: 0.5rem; display: flex; flex-direction: column; min-height: 0; }}
                    .live-feedback-pane > div[aria-live] {{ scroll-snap-type: y mandatory; overflow-y: auto; flex: 1 1 auto; min-height: 0; }}
                    .live-feedback-pane .stContainer {{ scroll-snap-align: start; }}
            .live-transcript-turn {{
                margin: 0 0 1rem 0;
                line-height: 1.7;
            }}
            .live-transcript-turn.non-candidate {{ opacity: 0.56; margin-left: 1rem; }}
            .speaker-0 .live-turn-meta {{ color: #a52b1a; }}
            .speaker-1 .live-turn-meta {{ color: #176b61; }}
            .speaker-2 .live-turn-meta {{ color: #60408b; }}
            .live-turn-meta, .live-caption {{
                color: {secondary_col};
                font-size: 0.88rem;
            }}
            .asr-unsure {{ border-bottom: 2px dotted #8a4b08; }}
            .asr-unsure::after {{ content: " ?"; color: #8a4b08; font-size: 0.75em; }}
            .candidate-text {{ background: rgba(22, 110, 85, 0.08); }}
            .examiner-text {{ color: {secondary_col}; opacity: 0.68; }}
            .language-cited {{ background: #ffe4df; }}
            .live-interim {{ color: {secondary_col}; font-style: italic; }}
            .band-row {{ display: flex; gap: 0.7rem; margin-bottom: 0.9rem; }}
            .band-score {{ flex: 1; border-left: 4px solid #e85d4a; padding-left: 0.55rem; }}
            .band-score.weakest {{ border-left-color: #8a2a1e; background: rgba(138, 42, 30, 0.08); padding: 0.35rem 0.55rem; }}
            .band-score b {{ font-size: 1.45rem; }}
            .feedback-card {{ border-left: 4px solid #c4321c; padding: 0.75rem; margin: 0.8rem 0; background: rgba(196, 50, 28, 0.08); }}
            .feedback-card.hero {{ border-left-width: 6px; padding: 1rem; }}
            .feedback-card.hero strong {{ font-size: 1.18rem; }}
            .say-next {{ display: block; margin-top: 0.4rem; font-weight: 600; }}
            .strength-summary {{ margin: 0.6rem 0; color: {secondary_col}; font-size: 0.9rem; }}
            .feedback-card.strength {{ border-left: 4px dotted #166e55; background: rgba(22, 110, 85, 0.09); }}
            .feedback-card.covered {{ opacity: 0.55; border-left-color: #667085; background: rgba(102, 112, 133, 0.08); }}
            .category-chip, .count-badge {{ display: inline-block; font-size: 0.84rem; padding: 0.12rem 0.42rem; border-radius: 999px; background: rgba(196, 50, 28, 0.18); }}
            .feedback-card.strength .category-chip, .feedback-card.strength .count-badge {{ background: rgba(35, 133, 91, 0.18); }}
            .count-badge {{ float: right; }}
            .feedback-card[data-evidence-target] {{ cursor: pointer; }}
            .feedback-card[data-evidence-target]:focus-visible {{ outline: 3px solid #e85d4a; outline-offset: 2px; }}
            .priority-card {{ margin: 0.65rem 0; padding: 0.1rem 0.7rem 0.7rem; border: 1px solid {border_col}; border-left: 4px solid #c4321c; border-radius: 8px; }}
            .priority-judgement {{ display: flex; gap: 0.45rem; margin-top: 0.55rem; }}
            .priority-judgement button {{ min-height: 2.25rem; }}
            .priority-judgement button[aria-pressed="true"] {{ background: #166e55; color: white; border-color: #166e55; }}
            .coverage-line {{ margin: 0.55rem 0 0.8rem; font-size: 0.82rem; }}
            .coverage-bar {{ height: 0.38rem; margin-top: 0.28rem; border-radius: 99px; background: repeating-linear-gradient(135deg, #d1d5db 0, #d1d5db 4px, #e5e7eb 4px, #e5e7eb 8px); overflow: hidden; }}
            .coverage-complete {{ height: 100%; background: #23855b; }}
            .language-cited {{ background: #ffe4df; animation: evidence-flash 1.5s ease-out; }}
            @keyframes evidence-flash {{ from {{ background: #f7a89b; }} to {{ background: #ffe4df; }} }}
            .noise-line {{ color: {secondary_col}; font-size: 0.82rem; margin: 0.55rem 0; }}
            @keyframes feedback-pulse {{ from {{ background: rgba(232, 93, 74, 0.28); }} to {{ background: rgba(232, 93, 74, 0.08); }} }}
            .feedback-card.new {{ animation: feedback-pulse 1.8s ease-out; }}
            .diff-line {{ margin: 0.35rem 0; line-height: 1.45; }}
            .diff-line b {{ display: inline-block; width: 4.4rem; color: {secondary_col}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }}
            .diff-line del {{ color: #a52b1a; text-decoration-thickness: 2px; }}
            .diff-line strong {{ color: #166e55; font-weight: 700; }}
            .status-strip {{ position: static; margin-top: 0.8rem; background: {card_bg}; padding-top: 0.6rem; border-top: 1px solid {border_col}; font-size: 0.85rem; }}
            .session-bar {{ display: flex; align-items: center; gap: 0.8rem; flex-wrap: wrap; padding: 0.55rem 0.8rem; margin: 0 0 0.8rem; border: 1px solid {border_col}; border-radius: 6px; background: {card_bg}; font-size: 0.9rem; }}
            .session-bar {{ position: sticky; top: 0; z-index: 10; min-height: 3rem; }}
            .session-bar .status-slot {{ min-width: 7.5rem; white-space: nowrap; }}
            .session-bar .status-slot.wide {{ min-width: 10rem; }}
            .st-key-live-session-layout {{ height: calc(100dvh - 8.5rem); min-height: 28rem; }}
            .st-key-live-session-layout [data-testid="column"] {{ height: 100%; min-width: 0; overflow-y: auto; overscroll-behavior: contain; padding-right: 0.45rem; }}
            .st-key-live-session-layout [data-testid="column"] > div {{ min-height: 100%; }}
            .st-key-live-session-layout .live-console-pane {{ height: calc(100dvh - 15rem); min-height: 14rem; }}
            .st-key-live-session-layout [data-testid="stExpander"] {{ margin-bottom: 0.25rem; }}
            .st-key-live-session-layout [data-testid="stExpander"] summary {{ min-height: 2.5rem; align-items: center; }}
            .active-only-hidden {{ display: none !important; }}
            .recording-collapsed {{ display: flex; align-items: center; gap: 0.8rem; min-height: 2.5rem; padding: 0.25rem 0.5rem; margin-bottom: 0.35rem; font-size: 0.9rem; }}
            .session-state {{ font-weight: 700; }}
            .session-state.listening::before {{ content: "● "; color: #166e55; }}
            .session-state.paused::before {{ content: "● "; color: #b7791f; }}
            .session-state.disconnected::before {{ content: "● "; color: #c4321c; }}
            .session-state.listening::before {{ animation: live-dot 1.2s ease-in-out infinite; }}
            @keyframes live-dot {{ 50% {{ opacity: 0.35; }} }}
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


def _format_live_time(value):
    if not isinstance(value, (int, float)):
        return "--:--"
    total_seconds = max(0, int(value))
    return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"


def _live_turn_payloads():
    frozen_until = float(st.session_state.get("live_feedback_frozen_until") or 0)
    if time.monotonic() < frozen_until and st.session_state.get("live_feedback_snapshot"):
        return st.session_state.live_feedback_snapshot
    results, stats = st.session_state.turn_analysis_worker.snapshot()
    statuses = st.session_state.get("live_feedback_statuses", {})
    summary = aggregate_feedback(
        results.values(),
        previous_order=st.session_state.get("live_priority_order", ()),
        force_priority_phrases=st.session_state.get("live_manual_priorities", set()),
        exclude_phrases=[phrase for phrase, status in statuses.items() if status == "rejected"],
        low_confidence_threshold=_confidence_flag_threshold(),
    )
    for item in summary["priorities"]:
        item["teacher_status"] = statuses.get(item["phrase"])
    summary["priorities"].sort(key=lambda item: item.get("teacher_status") == "covered")
    snapshot = results, stats, summary
    st.session_state.live_feedback_snapshot = snapshot
    return snapshot


def _live_analysis_coverage(results):
    """Summarise how much closed candidate speech has a usable analysis."""
    processed = st.session_state.get("live_processed_turns") or {}
    candidate_turns = [
        turn for turn in processed.values()
        if turn.get("is_candidate") and turn.get("closed")
    ]
    total_duration = sum(
        max(0.0, float(turn.get("t_end") or 0) - float(turn.get("t_start") or 0))
        for turn in candidate_turns
    )
    analysed_keys = {
        key for key, item in results.items() if item.get("status") == "complete"
    }
    analysed_turns = [turn for turn in candidate_turns if turn.get("key") in analysed_keys]
    analysed_duration = sum(
        max(0.0, float(turn.get("t_end") or 0) - float(turn.get("t_start") or 0))
        for turn in analysed_turns
    )
    incomplete = [
        turn for turn in candidate_turns if turn.get("key") not in analysed_keys
    ]
    failures = [
        turn for turn in incomplete
        if results.get(turn.get("key"), {}).get("status") in {"cancelled", "error"}
    ]
    percent = (100 * analysed_duration / total_duration) if total_duration else 0
    analysed_starts = [turn.get("t_start") for turn in analysed_turns if isinstance(turn.get("t_start"), (int, float))]
    analysed_ends = [turn.get("t_end") for turn in analysed_turns if isinstance(turn.get("t_end"), (int, float))]
    return {
        "analysed_duration": analysed_duration,
        "total_duration": total_duration,
        "percent": percent,
        "complete_turns": len(analysed_turns),
        "incomplete_turns": len(incomplete),
        "failed_turns": len(failures),
        "is_sufficient": total_duration > 0 and percent >= 85,
        "analysed_start": min(analysed_starts) if analysed_starts else None,
        "analysed_end": max(analysed_ends) if analysed_ends else None,
    }


def _live_session_status(results):
    coverage = _live_analysis_coverage(results)
    pending = sum(
        1 for item in results.values() if item.get("status") in {"queued", "analysing"}
    )
    is_recording = bool(st.session_state.get("is_recording"))
    started_at = st.session_state.get("recording_started_at")
    elapsed = int(st.session_state.get("recording_elapsed_seconds") or 0)
    if is_recording and started_at:
        elapsed = max(0, int((datetime.now(timezone.utc) - started_at).total_seconds()))
    streamer = st.session_state.get("deepgram_streamer")
    disconnected = bool(is_recording and streamer and streamer.error)
    if disconnected:
        state, label = "disconnected", "Disconnected"
    elif is_recording:
        state, label = "listening", "Listening"
    else:
        state, label = "paused", "Paused"
    backlog = f"analysing {pending} turn{'s' if pending != 1 else ''}" if pending else "up to date"
    return {
        "state": state,
        "label": label,
        "elapsed": elapsed,
        "backlog": backlog,
        "coverage": coverage,
    }


def _render_live_transcript_pane():
    processed = st.session_state.get("live_processed_turns") or {}
    turn_order = st.session_state.get("live_turn_order") or []
    candidate = _selected_candidate_speaker(st.session_state.get("deepgram_final_words") or [])
    scope = st.session_state.get("live_transcript_scope", "Everyone")
    view = st.session_state.get("live_transcript_view", "Turns")
    streamer = st.session_state.get("deepgram_streamer")
    entries = []
    visible_turns = []
    for key in turn_order:
        turn = processed.get(key)
        if not turn:
            continue
        is_candidate = turn.get("speaker") == candidate
        if scope == "Candidate only" and not is_candidate:
            continue
        visible_turns.append((turn, is_candidate))
    history_limit = int(st.session_state.get("live_transcript_history_limit", 8))
    hidden_count = max(0, len(visible_turns) - history_limit)
    visible_turns = visible_turns[-history_limit:]
    if view == "Paragraph":
        paragraph_parts = []
        for turn, is_candidate in visible_turns:
            words = " ".join(
                f'<span class="paragraph-word {"candidate-text" if is_candidate else "examiner-text"}">{html.escape(str(word.get("text") or ""))}</span>'
                for word in turn["words"]
            )
            paragraph_parts.append(
                f'<span class="paragraph-turn" aria-label="{"Candidate" if is_candidate else "Teacher"} turn">{words}</span>'
            )
        if paragraph_parts:
            entries.append(f'<div class="live-transcript-turn paragraph-view">{" ".join(paragraph_parts)}</div>')
    for turn, is_candidate in visible_turns if view == "Turns" else []:
        turn_id = turn["id"]
        label = f'{"Candidate" if is_candidate else "Teacher"} - Turn {turn_id + 1}'
        rendered_words = []
        for word_index, word in enumerate(turn["words"]):
            token = html.escape(str(word.get("text") or ""))
            if _enabled_deepgram_flag_reasons(word.get("confidence"), word.get("flag_reasons")):
                confidence = word.get("confidence")
                token = f'<span class="asr-unsure" title="Transcription uncertainty, not a student error. Confidence: {confidence if confidence is not None else "unknown"}">{token}</span>'
            rendered_words.append(
                f'<span id="turn-{turn_id}-word-{word_index}" data-evidence-word="true">{token}</span>'
            )
        text = " ".join(rendered_words)
        entries.append(
            f'<div class="live-transcript-turn speaker-{turn.get("speaker", "unknown")} '
            f'{"candidate" if is_candidate else "non-candidate"}" id="turn-{turn_id}">'
            f'<div class="live-turn-meta">{label} | {_format_live_time(turn.get("t_start"))} - {_format_live_time(turn.get("t_end"))}</div>{text}</div>'
        )
    interim = streamer.snapshot().get("interim", "") if streamer else ""
    if interim:
        entries.append(f'<div class="live-interim">{html.escape(interim)}</div>')
    body = "".join(entries) or '<span class="live-caption">Waiting for finalised Deepgram words.</span>'
    jump_token = st.session_state.get("live_jump_token", 0)
    st.markdown(
        '<div class="live-caption">Legend: <span class="asr-unsure">dotted underline = transcription uncertainty, not an English error</span> | '
        '<span class="language-cited">coral highlight = cited feedback evidence</span></div>'
        f'<div class="live-console-pane" id="live-transcript" data-jump-token="{jump_token}" '
        f'role="log" aria-label="Live transcript" aria-live="polite" aria-relevant="additions" tabindex="0">{body}</div>',
        unsafe_allow_html=True,
    )
    st.button(
        "Jump to live",
        key="jump_to_live",
        help="Scroll the transcript to the newest turn.",
        on_click=lambda: st.session_state.__setitem__("live_jump_token", jump_token + 1),
    )
    if hidden_count:
        if st.button(f"Load {min(8, hidden_count)} earlier turn(s)", key="load_earlier_turns"):
            st.session_state.live_transcript_history_limit = history_limit + 8
            st.rerun()
    components.html(
        """
        <script>
        const doc = window.parent.document;
        const pane = doc.getElementById('live-transcript');
        if (pane) {
          const latestToken = pane.dataset.jumpToken;
          const nearBottom = () => pane.scrollHeight - pane.scrollTop - pane.clientHeight < 32;
          if (pane.dataset.lastJumpToken !== latestToken) {
            pane.scrollTop = pane.scrollHeight;
            pane.dataset.followLive = 'true';
            pane.dataset.lastJumpToken = latestToken;
          } else if (pane.dataset.followLive !== 'false') {
            pane.scrollTop = pane.scrollHeight;
          }
          pane.addEventListener('scroll', () => {
            pane.dataset.followLive = nearBottom() ? 'true' : 'false';
          }, { passive: true });
        }
                if (!doc.documentElement.dataset.evidenceNavigationBound) {
                    doc.documentElement.dataset.evidenceNavigationBound = 'true';
                    const showEvidence = (card) => {
                        const targetId = card.dataset.evidenceTarget;
                        const wordCount = Number(card.dataset.evidenceWords || 1);
                        const target = doc.getElementById(targetId);
                        const transcriptPane = doc.getElementById('live-transcript');
                        if (!target || !transcriptPane) return;
                        doc.querySelectorAll('[data-evidence-word].language-cited').forEach((node) => node.classList.remove('language-cited'));
                        let current = target;
                        for (let index = 0; current && index < wordCount; index += 1) {
                            current.classList.add('language-cited');
                            current = current.nextElementSibling;
                        }
                        target.scrollIntoView({ block: 'center', behavior: 'smooth' });
                        transcriptPane.dataset.followLive = 'false';
                    };
                    doc.addEventListener('click', (event) => {
                        const card = event.target.closest('.feedback-card[data-evidence-target]');
                        if (!card) return;
                        event.preventDefault();
                        showEvidence(card);
                    });
                    doc.addEventListener('keydown', (event) => {
                        if (event.key !== 'Enter' && event.key !== ' ') return;
                        const card = event.target.closest('.feedback-card[data-evidence-target]');
                        if (!card) return;
                        event.preventDefault();
                        showEvidence(card);
                    });
                    doc.addEventListener('keydown', (event) => {
                        const tagName = event.target.tagName;
                        if (event.ctrlKey || event.metaKey || ['INPUT', 'TEXTAREA', 'SELECT'].includes(tagName)) return;
                        const key = event.key.toLowerCase();
                        if (['e', 'c'].includes(key)) {
                            const show = doc.querySelectorAll('input[type="radio"]');
                            const target = [...show].find((input) => input.value === (key === 'c' ? 'Candidate only' : 'Everyone'));
                            if (target) { event.preventDefault(); target.click(); return; }
                        }
                        if (['p', 't'].includes(key)) {
                            const view = doc.querySelectorAll('input[type="radio"]');
                            const target = [...view].find((input) => input.value === (key === 'p' ? 'Paragraph' : 'Turns'));
                            if (target) { event.preventDefault(); target.click(); return; }
                        }
                        if (event.key.toLowerCase() === 'j') {
                            const transcriptPane = doc.getElementById('live-transcript');
                            if (!transcriptPane) return;
                            event.preventDefault();
                            transcriptPane.scrollTop = transcriptPane.scrollHeight;
                            transcriptPane.dataset.followLive = 'true';
                            transcriptPane.focus();
                            return;
                        }
                        if (!['1', '2', '3'].includes(event.key)) return;
                        const cards = doc.querySelectorAll('.feedback-card[data-evidence-target]:not(.strength)');
                        const card = cards[Number(event.key) - 1];
                        if (!card) return;
                        event.preventDefault();
                        showEvidence(card);
                        card.focus();
                    });
                }
        </script>
        """,
        height=0,
        )


def _live_criterion(category):
    category = str(category or "").strip().lower()
    if category in {"grammar", "verb form", "agreement"}:
        return "Grammatical Range & Accuracy"
    if category in {"cohesion", "coherence", "disfluency", "fluency", "narrative control"}:
        return "Fluency & Coherence"
    return "Lexical Resource"


def _validated_live_evidence(item, results):
    evidence = (item.get("evidence") or [None])[-1]
    if not evidence or evidence.get("turn_id") is None:
        return None
    source_turn = next(
        (
            result.get("turn")
            for result in results.values()
            if isinstance(result, dict)
            and isinstance(result.get("turn"), dict)
            and result["turn"].get("id") == evidence.get("turn_id")
        ),
        None,
    )
    evidence_time = evidence.get("t_start")
    turn_start, turn_end = (source_turn or {}).get("start"), (source_turn or {}).get("end")
    valid = (
        isinstance(evidence_time, (int, float))
        and isinstance(turn_start, (int, float))
        and isinstance(turn_end, (int, float))
        and turn_start <= evidence_time <= turn_end
    )
    if not valid:
        log_event(
            "live_evidence_timestamp_invalid",
            phrase=item.get("phrase"),
            evidence_time=evidence_time,
            turn_start=turn_start,
            turn_end=turn_end,
        )
        return None
    return evidence


def _render_live_feedback_pane():
    results, stats, summary = _live_turn_payloads()
    priorities = summary["priorities"]
    coverage = _live_analysis_coverage(results)
    st.session_state.live_priority_order = [item["phrase"] for item in priorities]
    completed = [item for item in results.values() if item.get("status") == "complete"]
    scores = {name: [] for name in ("fluency_coherence", "lexical_resource", "grammar")}
    for item in completed:
        payload = item.get("analysis") or {}
        for name in scores:
            value = (payload.get("provisional_scores") or {}).get(name)
            if isinstance(value, (int, float)):
                scores[name].append(float(value))
    previous = st.session_state.get("live_score_snapshot", {})
    current = {name: (sum(values) / len(values) if values else None) for name, values in scores.items()}
    st.session_state.live_score_snapshot = current
    valid_scores = {name: value for name, value in current.items() if value is not None}
    weakest_criterion = min(valid_scores, key=valid_scores.get) if valid_scores else None
    flagged_words, total_words = _deepgram_flagged_word_count(
        st.session_state.get("deepgram_final_words") or []
    )
    asr_reliability = 1 - (flagged_words / total_words) if total_words else 1.0
    score_confidence = min(1.0, len(completed) / 6) * (coverage["percent"] / 100) * asr_reliability
    confidence_label = "high" if score_confidence >= 0.75 else "medium" if score_confidence >= 0.45 else "low"
    score_values = [value for value in current.values() if value is not None]
    if score_values and coverage["is_sufficient"]:
        lower = math.floor(sum(score_values) / len(score_values) * 2) / 2
        band_summary = f"Est. band {lower:.1f}-{lower + 0.5:.1f} · flat · {confidence_label} confidence"
    else:
        band_summary = "Est. band partial · awaiting sufficient candidate speech"
    criterion_breakdown = " · ".join(
        f"{_live_criterion(name)} {value:.1f}" for name, value in current.items() if value is not None
    ) or "No criterion scores yet"
    st.session_state.seen_live_priorities = set(st.session_state.get("seen_live_priorities", set())) | {item["phrase"] for item in priorities}
    strengths = "".join(
        f'<div class="feedback-card strength"><strong>"{html.escape(item["phrase"][:120])}"</strong>'
        f'<span class="say-next">{html.escape(item.get("suggestion") or "This makes the message easy to follow.")}</span></div>'
        for item in summary["strengths"]
    ) or '<div class="live-caption">Positive evidence will appear as turns are analysed.</div>'
    if summary["noise_count"]:
        strengths += f'<div class="noise-line">ASR noise / disfluency: {summary["noise_count"]} uncertain token(s), excluded from priorities.</div>'
    streamer = st.session_state.get("deepgram_streamer")
    error_count = sum(1 for item in results.values() if item.get("status") == "error")
    priority_overflow = summary.get("priority_overflow", [])
    coverage_text = (
        f'Analysed {coverage["percent"]:.0f}% of candidate speech'
        if coverage["total_duration"]
        else "Waiting for analysed candidate speech"
    )
    coverage_bar = (
        f'<div class="coverage-line">{coverage_text}<div class="coverage-bar">'
        f'<div class="coverage-complete" style="width:{coverage["percent"]:.1f}%"></div></div></div>'
    )
    st.markdown(
        f'<div class="live-caption">{html.escape(band_summary)}</div>'
        f'<details class="band-breakdown"><summary>Criterion breakdown</summary><span class="live-caption">{html.escape(criterion_breakdown)}</span></details>'
        f'{coverage_bar}<strong>Priority areas</strong>',
        unsafe_allow_html=True,
    )
    if error_count:
        first_error = next(
            (
                str(item.get("error") or "")
                for item in results.values()
                if item.get("status") == "error"
            ),
            "",
        )
        if "401" in first_error or "invalid_api_key" in first_error.lower():
            st.error(
                "OpenAI rejected the API key. Replace OPENAI_API_KEY in the app environment, "
                "restart Streamlit, then retry the failed turns."
            )
        else:
            st.warning(f"{error_count} live analysis turn(s) failed. Review the details below and retry.")
    criterion_options = ["All", "Fluency & Coherence", "Lexical Resource", "Grammatical Range & Accuracy"]
    selected_criterion = st.selectbox(
        "Filter by IELTS criterion",
        criterion_options,
        key="live_priority_criterion_filter",
        label_visibility="collapsed",
    )
    visible_priorities = [
        item for item in priorities
        if selected_criterion == "All" or _live_criterion(item["category"]) == selected_criterion
    ]
    if visible_priorities:
        for item in visible_priorities:
            evidence = _validated_live_evidence(item, results)
            timestamp = _format_live_time(evidence.get("t_start")) if evidence else "unavailable"
            evidence_attrs = ""
            if evidence:
                evidence_attrs = (
                    f' data-evidence-target="turn-{evidence["turn_id"]}-word-{evidence["word_start"]}"'
                    f' data-evidence-words="{evidence["word_end"] - evidence["word_start"]}"'
                    ' tabindex="0" role="button"'
                )
            teacher_status = item.get("teacher_status")
            suggestion = str(item.get("suggestion") or "").strip().strip('"')
            said_markup, target_markup = _word_diff_markup(item["phrase"], suggestion)
            row_label = f'{item["phrase"]} -> {suggestion} · {timestamp}'
            with st.expander(row_label, expanded=False):
                st.markdown(
                    f'<div class="feedback-card{" covered" if teacher_status == "covered" else ""}"{evidence_attrs}>'
                    f'<span class="category-chip">{html.escape(_priority_tag(item["category"]))}</span>'
                    f'<span class="count-badge">{item["count"]}x</span>'
                    f'<div class="diff-line"><b>Said</b> {said_markup}</div>'
                    f'<div class="diff-line"><b>Target</b> {target_markup}</div>'
                    f'<span class="live-caption">{html.escape(_live_criterion(item["category"]))} · {timestamp}</span></div>',
                    unsafe_allow_html=True,
                )
                covered_col, rejected_col = st.columns(2)
                with covered_col:
                    if st.button(
                        "✓",
                        key=f'covered-{item["normalised"]}',
                        use_container_width=True,
                        help=f'Mark "{item["phrase"]}" as covered',
                        type="primary" if teacher_status == "covered" else "secondary",
                    ):
                        st.session_state.live_feedback_statuses = {**st.session_state.get("live_feedback_statuses", {}), item["phrase"]: "covered"}
                        st.session_state.live_feedback_frozen_until = time.monotonic() + 0.8
                        log_event("feedback_item_covered", phrase=item["phrase"], category=item["category"])
                        st.rerun()
                with rejected_col:
                    if st.button(
                        "✗",
                        key=f'rejected-{item["normalised"]}',
                        use_container_width=True,
                        help=f'Mark "{item["phrase"]}" as not an error',
                        type="primary" if teacher_status == "rejected" else "secondary",
                    ):
                        st.session_state.live_feedback_statuses = {**st.session_state.get("live_feedback_statuses", {}), item["phrase"]: "rejected"}
                        st.session_state.live_feedback_frozen_until = time.monotonic() + 0.8
                        log_event("feedback_item_rejected", phrase=item["phrase"], category=item["category"])
                        st.rerun()
    else:
        st.caption("Listening for recurring, teachable patterns.")
    st.markdown(f'<strong>Doing well</strong>{strengths}', unsafe_allow_html=True)
    if priority_overflow:
        with st.expander(f'+{len(priority_overflow)} more', expanded=False):
            for item in priority_overflow:
                st.markdown(f'{html.escape(_live_criterion(item["category"]))}: **{html.escape(item["phrase"])}** ({item["count"]}×)')
    if coverage["failed_turns"] and not st.session_state.get("is_recording"):
        if st.button(f'Retry {coverage["failed_turns"]} failed analysis turn(s)', key="retry_failed_turns"):
            st.session_state.turn_analysis_worker.retry_failed()
            st.rerun()
    if summary["emerging"]:
        with st.expander(f'Watchlist ({summary["emerging_count"]})', expanded=False):
            for item in summary["emerging"]:
                st.markdown(
                    f'**{html.escape(_live_criterion(item["category"]))}**: "{html.escape(item["phrase"])}"  '
                    f'\n\nSeen {item["count"]}× — needs 2 to become a priority · last at {_format_live_time(item["last_seen"])}  '
                    f'\n\nSuggested correction: {html.escape(item.get("suggestion") or "Review this phrase in context.")}'
                )
                if st.button(
                    "Add to priorities",
                    key=f'promote-pattern-{item["normalised"]}',
                ):
                    st.session_state.live_manual_priorities = set(
                        st.session_state.get("live_manual_priorities", set())
                    ) | {item["phrase"]}
                    st.rerun()


def render_live_session():
    results, _, summary = _live_turn_payloads()
    active = bool(st.session_state.get("is_recording") or st.session_state.get("streaming_active"))
    if not active and not results:
        return
    session_status = _live_session_status(results)
    coverage = session_status["coverage"]
    st.session_state.analysis_transcript_choice = "Deepgram"
    final_words = st.session_state.get("deepgram_final_words") or []
    speaker_ids = sorted({word.get("speaker") for word in final_words if word.get("speaker") is not None}, key=str)
    candidate_value = st.session_state.get("candidate_speaker_override", "Auto")
    candidate_name = "automatic" if candidate_value == "Auto" else _speaker_label(candidate_value)
    st.markdown(
        f'<div class="session-bar" role="status" aria-live="polite" aria-label="Session status">'
        f'<span class="session-state {session_status["state"]} status-slot">{session_status["label"]}</span>'
        f'<span class="status-slot">{_format_live_time(session_status["elapsed"])} elapsed</span>'
        f'<span class="status-slot wide">{session_status["backlog"]}</span>'
        f'<span class="status-slot wide">Analysed {coverage["percent"]:.0f}%</span>'
        f'<span class="status-slot wide">{html.escape(candidate_name)}</span></div>',
        unsafe_allow_html=True,
    )
    with st.popover("⋯", use_container_width=False):
        if speaker_ids:
            st.selectbox(
                "Candidate speaker",
                ["Auto"] + [str(speaker) for speaker in speaker_ids],
                format_func=lambda value: "Automatic (most speech)" if value == "Auto" else _speaker_label(value),
                key="candidate_speaker_override",
            )
        if st.button("Re-record", key="active_rerecord"):
            reset_for_rerecord()
            st.rerun()
    with st.container(key="live-session-layout"):
        transcript_col, feedback_col = st.columns([40, 60])
        with transcript_col:
            st.markdown("## Live Transcript")
            transcript_filter_col, transcript_view_col = st.columns(2)
            with transcript_filter_col:
                st.radio("Show", ["Everyone", "Candidate only"], key="live_transcript_scope", horizontal=True, captions=["Everyone (E)", "Candidate only (C)"])
            with transcript_view_col:
                st.radio("View", ["Paragraph", "Turns"], key="live_transcript_view", horizontal=True, captions=["Paragraph (P)", "Turns (T)"])
            render_live_transcript_fragment()
        with feedback_col:
            st.markdown("## What to say next")
            render_live_feedback_fragment()
    if results and not st.session_state.get("is_recording"):
        lesson_lines = ["Teacher brief", "", "Priority areas"]
        lesson_lines.extend(
            f'- {item["category"]}: "{item["phrase"]}" -> {item["suggestion"]} '
            f'(x{item["count"]}){f" [{item["teacher_status"]}]" if item.get("teacher_status") else ""}'
            for item in summary["priorities"]
        )
        lesson_lines.append("\nDoing well")
        lesson_lines.extend(f'- "{item["phrase"]}" (x{item["count"]})' for item in summary["strengths"])
        with st.container(key="live-session-footer"):
            with st.expander("Review Audio", expanded=False):
                if audio_preview_data:
                    st.audio(audio_preview_data)
                elif audio_preview_path and os.path.exists(audio_preview_path):
                    st.audio(audio_preview_path, format="audio/wav")
                else:
                    st.caption("Audio will be available after recording ends.")
            st.download_button(
                "Export lesson note",
                "\n".join(lesson_lines),
                file_name="teacher-brief.txt",
                mime="text/plain",
            )
            with st.expander("Full detail", expanded=False):
                for item in sorted(results.values(), key=lambda value: value["turn"].get("id", value["turn"].get("index", 0))):
                    turn = item.get("turn") or {}
                    st.markdown(f'**Turn {turn.get("id", turn.get("index", 0)) + 1}**: {item.get("status", "unknown")}')
                    if item.get("status") == "complete":
                        analysis = item.get("analysis") or {}
                        for observation in analysis.get("observations") or []:
                            st.write(f"- {observation}")
                        for error in analysis.get("errors") or []:
                            if isinstance(error, dict):
                                st.write(f'- {error.get("phrase", "")} -> {error.get("suggestion", "") }')


if hasattr(st, "fragment"):
    render_live_transcript_fragment = st.fragment(run_every=0.4)(_render_live_transcript_pane)
    render_live_feedback_fragment = st.fragment(run_every=1.5)(_render_live_feedback_pane)
else:
    render_live_transcript_fragment = _render_live_transcript_pane
    render_live_feedback_fragment = _render_live_feedback_pane


render_app_styles()
init_session_state()
st.session_state.session_mode = _current_session_mode()

log_event(
    "app_rerun",
    audio_source=st.session_state.get("audio_source"),
    recording_ready=recording_ready(),
)

# ── Page title ────────────────────────────────────────────────────────────────
with st.container():
    st.title("Instant Spoken English Feedback")

# =============================================================================
# STEP 1 — Audio Input
# =============================================================================
uploaded = None
audio_preview_data = None
audio_preview_path = None
audio_filename = ""

with st.container():
    audio_source = st.session_state.get("audio_source", "Record")
    if not (recording_ready() and audio_source == "Record"):
        st.markdown("### Step 1: Audio Input")
        audio_source = st.radio(
            "Choose audio input",
            ["Upload", "Load", "Record"],
            index=["Upload", "Load", "Record"].index(audio_source),
            key="audio_source",
            horizontal=True,
        )
    else:
        st.markdown(
            f'<div class="recording-collapsed"><span class="session-state paused">Paused</span>'
            f'<span>{_format_live_time(st.session_state.get("recording_elapsed_seconds", 0))}</span>'
            '<span>Microphone recording ready</span></div>',
            unsafe_allow_html=True,
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
            restore_deepgram_word_records(selected_path)
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

    elif audio_source == "Record" and not recording_ready():
        recorder_value = None if recording_ready() else render_audio_recorder(
            compact=st.session_state.get("session_mode") == "active"
        )
        recorder_type = type(recorder_value).__name__
        log_event(
            "recorder_value_seen",
            recorder_type=recorder_type,
            has_value=recorder_value is not None,
        )
        wav_bytes = None
        recorder_event_name = None
        recorder_payload = None

        if isinstance(recorder_value, (bytes, bytearray)):
            wav_bytes = bytes(recorder_value)
        else:
            if isinstance(recorder_value, str):
                try:
                    recorder_payload = json.loads(recorder_value)
                except json.JSONDecodeError:
                    recorder_payload = None
            elif isinstance(recorder_value, dict):
                recorder_payload = recorder_value
            else:
                recorder_payload = None

            if isinstance(recorder_payload, dict):
                recorder_event_name = recorder_payload.get("event")
                event_t = recorder_payload.get("t", 0)
                event_seq = recorder_payload.get("seq", "")
                event_id = f"{recorder_event_name}:{event_t}:{event_seq}"

                # Process each unique event payload once; don't drop events that share timestamp.
                is_new_event = bool(recorder_event_name) and (
                    event_id != st.session_state.get("last_recorder_event_id", "")
                )
                if is_new_event:
                    st.session_state.last_recorder_event_id = event_id
                    if isinstance(event_t, (int, float)):
                        st.session_state.last_recorder_event_t = max(
                            st.session_state.get("last_recorder_event_t", 0),
                            event_t,
                        )

                if is_new_event:
                    # Exclude large audio data from logs to avoid performance issues
                    log_payload = {k: v for k, v in recorder_payload.items() if k not in ["wav_base64", "pcm16_base64"]}
                    log_event(
                        "recorder_component_event",
                        event_name=recorder_event_name,
                        payload=log_payload,
                    )
                    wav_base64 = recorder_payload.get("wav_base64")
                    if wav_base64:
                        wav_bytes = base64.b64decode(wav_base64)
                else:
                    # This event was already processed in a previous Streamlit run
                    recorder_event_name = None
                    recorder_payload = None

        if recorder_event_name == "start_clicked":
            st.session_state.is_recording = True
            st.session_state.recording_started_at = datetime.now(timezone.utc)

        if recorder_event_name == "recording_started":
            st.session_state.is_recording = True
            st.session_state.recording_started_at = datetime.now(timezone.utc)
            st.session_state.recording_elapsed_seconds = 0
            st.session_state.deepgram_accumulated_transcript = ""
            st.session_state.deepgram_stop_requested_at = None
            reset_live_turn_analysis()
            sample_rate = None
            if isinstance(recorder_payload, dict):
                sample_rate = recorder_payload.get("sample_rate")
            start_deepgram_stream(sample_rate)

        if recorder_event_name == "stop_clicked":
            if st.session_state.get("recording_started_at"):
                st.session_state.recording_elapsed_seconds = max(
                    0,
                    int(
                        (
                            datetime.now(timezone.utc)
                            - st.session_state.recording_started_at
                        ).total_seconds()
                    ),
                )
            st.session_state.is_recording = False
            st.session_state.streaming_active = False
            st.session_state.deepgram_stop_requested_at = datetime.now(timezone.utc)
            log_event(
                "stop_pressed_preserving_transcript",
                deepgram_live_chars=len(st.session_state.get("deepgram_live_transcript") or ""),
                deepgram_final_chars=len(st.session_state.get("deepgram_final_transcript") or ""),
                deepgram_accumulated_chars=len(st.session_state.get("deepgram_accumulated_transcript") or ""),
                worker_started=bool(st.session_state.get("worker_started")),
            )

        if recorder_event_name == "recording_complete":
            st.session_state.is_recording = False
            st.session_state.deepgram_stop_requested_at = None
            stop_deepgram_stream()

        if recorder_event_name == "recording_chunk" and isinstance(recorder_payload, dict):
            pcm16_base64 = recorder_payload.get("pcm16_base64")
            chunk_sample_rate = recorder_payload.get("sample_rate")
            chunk_seq = recorder_payload.get("seq")
            if not pcm16_base64:
                log_event(
                    "deepgram_chunk_missing_b64",
                    seq=chunk_seq,
                    sample_rate=chunk_sample_rate,
                    payload_keys=list(recorder_payload.keys()),
                )
            else:
                try:
                    if (
                        not st.session_state.get("worker_started")
                        and not st.session_state.get("deepgram_streamer")
                        and st.session_state.get("is_recording")
                    ):
                        start_deepgram_stream(chunk_sample_rate)
                    chunk_bytes = base64.b64decode(pcm16_base64)
                    # Log every 10th decoded chunk to confirm the pipeline is flowing
                    # without flooding the event log.
                    streamer_for_count = st.session_state.get("deepgram_streamer")
                    if streamer_for_count is not None:
                        streamer_for_count.chunks_received += 1
                        _recv = streamer_for_count.chunks_received
                    else:
                        _recv = chunk_seq or 0
                    if _recv % 10 == 1:
                        log_event(
                            "deepgram_chunk_decoded",
                            seq=chunk_seq,
                            bytes_len=len(chunk_bytes),
                            sample_rate=chunk_sample_rate,
                            chunks_received=_recv,
                        )
                    send_deepgram_chunk(chunk_bytes)
                except Exception as e:
                    log_event("deepgram_chunk_decode_error", error=str(e))

        sync_deepgram_transcript_state()
        sync_live_turn_analysis()

        # ── Non-blocking UI refresh ──────────────────────────────────────────
        # transcript_queue is populated by the background WS thread whenever
        # Deepgram delivers a new result.  Drain it here (O(1) per rerun) and
        # call st.rerun() so the live transcript stays current without blocking
        # the main thread.  Forbidden: while-True, blocking recv in main thread.
        if st.session_state.get("is_recording"):
            _new_transcript = False
            try:
                while True:
                    transcript_queue.get_nowait()
                    _new_transcript = True
            except Empty:
                pass
            if _new_transcript:
                st.rerun()
        # ────────────────────────────────────────────────────────────────────

        if wav_bytes:
            digest = hashlib.sha256(wav_bytes).hexdigest()
            if digest == st.session_state.get("last_recorded_audio_digest"):
                log_event("record_payload_duplicate", digest=digest)
                wav_bytes = None
            else:
                st.session_state.last_recorded_audio_digest = digest
                log_event("record_payload_received", bytes_len=len(wav_bytes), digest=digest)

        if wav_bytes:
            if st.session_state.get("deepgram_streamer"):
                stop_deepgram_stream()
                st.session_state.deepgram_stop_requested_at = None
            recorded_path = build_timestamped_recording_path()
            with open(recorded_path, "wb") as f:
                f.write(wav_bytes)
            st.session_state.recorded_audio_bytes = wav_bytes
            st.session_state.recorded_audio_path = recorded_path
            st.session_state.loaded_audio_path = recorded_path
            st.session_state.is_recording = False
            save_deepgram_word_records(
                recorded_path, st.session_state.get("deepgram_final_words") or []
            )
            log_event(
                "record_saved",
                recorded_path=recorded_path,
                saved_filename=os.path.basename(recorded_path),
                bytes_len=len(wav_bytes),
            )

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

        if recording_ready():
            if st.button("Re-record"):
                log_event("rerecord_clicked")
                reset_for_rerecord()
                st.rerun()
    else:
        audio_preview_path = get_recorded_audio_path()
        audio_filename = os.path.basename(audio_preview_path) if audio_preview_path else "recording.wav"
        signature = f"record:{audio_preview_path}"
        if st.button("Re-record", key="compact_rerecord"):
            log_event("rerecord_clicked")
            reset_for_rerecord()
            st.rerun()

# ── Signature tracking → reset analysis on new audio ─────────────────────────
st.session_state.audio_ready = bool(audio_preview_data) or bool(
    audio_preview_path and os.path.exists(audio_preview_path)
)
if signature != st.session_state.get("active_audio_signature"):
    st.session_state.active_audio_signature = signature
    if signature:
        reset_analysis_state()

render_live_session()

# =============================================================================
# STEP 2 — Review Audio
# =============================================================================
st.markdown("---")
live_session_active = bool(
    st.session_state.get("is_recording")
    or st.session_state.get("streaming_active")
    or st.session_state.turn_analysis_worker.snapshot()[0]
)
with (
    st.expander("Review Audio", expanded=False)
    if live_session_active
    else st.container()
):
    st.markdown("### Step 2: Review Audio" if not live_session_active else "#### Review Audio")

    # Audio player + filename
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

    # Side-by-side transcript previews
    col_rep, col_dg = st.columns(2)
    with col_rep:
        st.markdown("**Replicate Transcript**")
        replicate_preview = (st.session_state.get("replicate_transcript") or "").strip()
        replicate_preview = _build_confidence_annotated_diarized_transcript(
            st.session_state.get("replicate_output")
        ) or replicate_preview
        st.markdown(
            f'<div class="clean-text-block">{html.escape(replicate_preview) if replicate_preview else "No Replicate transcript yet."}</div>',
            unsafe_allow_html=True,
        )
        if _has_low_confidence_replicate_segment(st.session_state.get("replicate_output")):
            st.caption("Words in parentheses come from a low-probability transcription segment.")
    with col_dg:
        st.markdown("**Deepgram Transcript**")
        final_words = st.session_state.get("deepgram_final_words") or []
        streamer = st.session_state.get("deepgram_streamer")
        interim_text = streamer.snapshot().get("interim") if streamer else ""
        deepgram_preview = _render_deepgram_transcript(
            final_words, annotate=True, interim_text=interim_text
        )
        st.markdown(
            f'<div class="clean-text-block">{html.escape(deepgram_preview) if deepgram_preview else "No Deepgram transcript yet."}</div>',
            unsafe_allow_html=True,
        )
        sensitivity_col, count_col = st.columns([3, 2])
        with sensitivity_col:
            if live_session_active:
                st.caption("ASR confidence threshold is in the Live Session toolbar.")
            else:
                st.slider(
                    "Flag words below confidence",
                    min_value=0.50,
                    max_value=1.00,
                    step=0.01,
                    key="low_confidence_threshold",
                )
        flagged_words, total_words = _deepgram_flagged_word_count(final_words)
        with count_col:
            percentage = (100 * flagged_words / total_words) if total_words else 0
            st.caption(f"{flagged_words} of {total_words} words flagged ({percentage:.0f}%)")
        duplicate_col, churn_col = st.columns(2)
        with duplicate_col:
            st.checkbox("Flag duplicate tokens", key="flag_duplicate_tokens")
        with churn_col:
            st.checkbox("Flag interim churn", key="flag_interim_churn")
        if final_words and not live_session_active:
            st.caption("Finalized word recognition confidence. This indicates ASR uncertainty, not English correctness.")
            speaker_ids = sorted({word.get("speaker") for word in final_words if word.get("speaker") is not None}, key=str)
            candidate_options = ["Auto"] + [str(speaker) for speaker in speaker_ids]
            st.selectbox(
                "Candidate speaker",
                candidate_options,
                format_func=lambda value: "Candidate: automatic (most speech)" if value == "Auto" else f"Candidate: {_speaker_label(value)}",
                key="candidate_speaker_override",
            )
        raw_deepgram = _render_deepgram_transcript(final_words, annotate=False)
        if DEV_MODE and raw_deepgram:
            with st.expander("Raw Deepgram transcript (debug)", expanded=False):
                st.markdown(
                    f'<div class="clean-text-block">{html.escape(raw_deepgram)}</div>',
                    unsafe_allow_html=True,
                )

# =============================================================================
# STEP 3 — Run Analysis
# =============================================================================
st.markdown("---")
with (
    st.expander("Analysis controls", expanded=False)
    if live_session_active
    else st.container()
):
    st.markdown("### Step 3: Run Analysis" if not live_session_active else "#### Analysis controls")

    analysis_control_col, legacy_control_col = st.columns(2)
    with analysis_control_col:
        st.checkbox("Live per-turn analysis", key="live_turn_analysis_enabled")
    with legacy_control_col:
        st.checkbox(
            "Use full-transcript analysis",
            key="use_legacy_full_transcript_analysis",
            disabled=live_session_active,
            help="Use the selected completed transcript for one report instead of live per-turn analysis.",
        )

    transcribe_clicked = st.button(
        "Re-transcribe with higher accuracy",
        type="secondary",
        disabled=not st.session_state.audio_ready or st.session_state.get("transcribing", False),
        help="Create a diarized transcript that can be used for feedback analysis.",
    )

    if live_session_active:
        st.caption("Transcript source is in the Live Session toolbar.")
    else:
        st.radio(
            "Use transcript for analysis:",
            ["Replicate", "Deepgram"],
            key="analysis_transcript_choice",
            horizontal=True,
        )

    if DEV_MODE:
        with st.expander("OpenAI Prompt (Editable)", expanded=False):
            st.text_area(
                "System prompt used for feedback generation",
                key="openai_system_prompt",
                height=260,
                on_change=persist_openai_prompt,
            )

    selected_for_analysis = st.session_state.get("analysis_transcript_choice", "Replicate")
    selected_text, transcript_availability = get_analysis_transcript(
        st.session_state, selected_for_analysis
    )
    turn_results, turn_stats = st.session_state.turn_analysis_worker.snapshot()
    has_incremental_input = any(item.get("status") == "complete" for item in turn_results.values())
    analyse_clicked = st.button(
        "Generate feedback report",
        disabled=(
            not (bool((selected_text or "").strip()) if st.session_state.get("use_legacy_full_transcript_analysis") else has_incremental_input)
            or st.session_state.get("analysing", False)
            or st.session_state.get("transcribing", False)
        ),
    )

    if turn_results:
        st.caption(
            f"Live turns: {turn_stats['turns_closed']} closed, {turn_stats['turns_analysed']} analysed, "
            f"{turn_stats['turns_gated_out']} candidate turns gated out, "
            f"{turn_stats['examiner_turns_ignored']} examiner turns ignored."
        )
        if DEV_MODE:
            with st.expander("Raw turn payloads (debug)", expanded=False):
                for item in sorted(
                    turn_results.values(),
                    key=lambda value: value["turn"].get("id", value["turn"].get("index", 0)),
                ):
                    turn = item["turn"]
                    st.write(
                        f"Turn {turn.get('id', turn.get('index', 0)) + 1} "
                        f"({_format_live_time(turn.get('start'))} - {_format_live_time(turn.get('end'))}): {item['status']}"
                    )
                    if item["status"] == "complete":
                        st.json(item["analysis"])
                    elif item["status"] == "error":
                        st.caption(item.get("error", "Analysis failed."))
                        if item.get("response_preview"):
                            st.code(item["response_preview"], language="json")

    if transcript_availability:
        st.caption(transcript_availability)

    if st.session_state.get("transcribing"):
        st.info("Transcription running...")
    if st.session_state.get("analysing"):
        st.info("OpenAI analysis running...")
    if st.session_state.get("transcribe_error"):
        st.error(st.session_state.get("transcribe_error"))
    if st.session_state.get("analysis_error"):
        st.error(st.session_state.get("analysis_error"))

# ── Background job polling & button handlers ──────────────────────────────────
poll_background_jobs()

if transcribe_clicked and not st.session_state.get("transcribing"):
    uploaded_bytes = uploaded.getvalue() if uploaded is not None else None
    uploaded_name = uploaded.name if uploaded is not None else None
    st.session_state.transcribe_error = ""
    st.session_state.analysis_complete = False
    st.session_state.feedback = []
    st.session_state.transcribing = True
    st.session_state.transcribe_started_at = perf_counter()
    st.session_state.transcribe_future = BACKGROUND_EXECUTOR.submit(
        run_transcription_background_job,
        source=st.session_state.get("audio_source"),
        uploaded_bytes=uploaded_bytes,
        uploaded_name=uploaded_name,
        recorded_path=get_recorded_audio_path(),
        loaded_path=st.session_state.get("loaded_audio_path"),
    )
    log_event(
        "transcription_background_started",
        source=st.session_state.get("audio_source"),
        timeout_s=JOB_TIMEOUT_S,
    )
    st.rerun()

if analyse_clicked and not st.session_state.get("analysing"):
    selected_for_analysis = st.session_state.get("analysis_transcript_choice", "Replicate")
    transcript_for_analysis, transcript_availability = get_analysis_transcript(
        st.session_state, selected_for_analysis
    )
    transcript_for_analysis = (transcript_for_analysis or "").strip()
    use_legacy = bool(st.session_state.get("use_legacy_full_transcript_analysis"))
    turn_results, _turn_stats = st.session_state.turn_analysis_worker.snapshot()
    has_incremental_input = any(item.get("status") == "complete" for item in turn_results.values())
    if use_legacy and not transcript_for_analysis:
        st.warning(transcript_availability or "No transcript available for selected source.")
    elif not use_legacy and not has_incremental_input:
        st.warning("No completed candidate-turn analyses are available yet.")
    else:
        st.session_state.analysis_error = ""
        st.session_state.analysis_complete = False
        st.session_state.transcript = transcript_for_analysis if use_legacy else "Incremental candidate-turn analyses"
        st.session_state["transcript_source"] = {
            "Replicate": "replicate",
            "Deepgram": "deepgram",
        }.get(selected_for_analysis, "unknown")
        st.session_state.analysing = True
        st.session_state.analysis_started_at = perf_counter()
        analysis_function = run_openai_analysis_background_job if use_legacy else run_incremental_synthesis_background_job
        analysis_input = transcript_for_analysis if use_legacy else turn_results
        st.session_state.analysis_future = BACKGROUND_EXECUTOR.submit(
            analysis_function,
            analysis_input,
            st.session_state.get("openai_system_prompt", DEFAULT_OPENAI_SYSTEM_PROMPT),
            bool(st.session_state.get("allow_rule_based_fallback_on_openai_failure", False)),
        )
        log_event(
            "analysis_background_started",
            transcript_choice=selected_for_analysis if use_legacy else "incremental_turns",
            transcript_chars=len(transcript_for_analysis) if use_legacy else 0,
            timeout_s=JOB_TIMEOUT_S,
        )
        st.rerun()

# ── Auto-rerun while background tasks are live so poll_background_jobs()
# can detect completion without requiring user interaction.
# IMPORTANT: use time.sleep + st.rerun() — NOT a JS location.reload().
# JS reload creates a brand-new Streamlit WebSocket session which resets
# session_state and orphans the transcribe_future permanently.
# st.rerun() stays within the same session, so session_state is preserved.
turn_results, _turn_stats = st.session_state.turn_analysis_worker.snapshot()
live_turn_work_pending = any(item.get("status") in {"queued", "analysing"} for item in turn_results.values())
if st.session_state.get("transcribing") or st.session_state.get("analysing") or live_turn_work_pending:
    time.sleep(2)
    st.rerun()

# =============================================================================
# STEP 4 — Feedback Output
# =============================================================================
st.markdown("---")
with st.container():
    st.markdown("### Step 4: Feedback Output")

    live_results, _, live_summary = _live_turn_payloads()
    has_live_feedback = bool(live_summary["priorities"] or live_summary["strengths"])
    if st.session_state.analysis_complete or bool(st.session_state.get("transcript")) or has_live_feedback:
        # Feedback table — shown first
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("#### Feedback Summary")
        if st.session_state.analysis_complete:
            feedback_md = build_sorted_feedback_markdown(st.session_state.get("feedback", []))
            if feedback_md:
                st.markdown(feedback_md)
            else:
                st.caption("No feedback available.")
        else:
            st.caption("The live teacher brief above is ready. Full-report generation is optional.")
        st.markdown("</div>", unsafe_allow_html=True)

        transcript_text = st.session_state.get("transcript") or ""
        if transcript_text:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("#### Transcript Used for Analysis")
            st.markdown(
                f'<div class="clean-text-block">{html.escape(transcript_text)}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # Processing time
        timing = st.session_state.get("analysis_timing") or {}
        if timing.get("total_s") is not None:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("#### Processing Time")
            st.markdown(
                "\n".join(
                    [
                        f"- Total: {_fmt_seconds(timing.get('total_s'))}",
                        f"- Audio prep: {_fmt_seconds(timing.get('audio_prep_s') or 0)}",
                        f"- Transcription: {_fmt_seconds(timing.get('transcription_s') or 0)}",
                        f"- OpenAI feedback: {_fmt_seconds(timing.get('openai_s') or 0)}",
                    ]
                )
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # Diarization & raw JSON in a single collapsible section
        with st.expander("Diarization & Raw Output", expanded=False):
            st.markdown("#### Audio Diarization Timeline")
            diarization_text = st.session_state.get("diarization") or ""
            if diarization_text:
                st.markdown(
                    f'<div class="clean-text-block">{html.escape(diarization_text)}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No diarization output.")
            st.markdown("#### Raw Structured JSON")
            if st.checkbox("Show raw structured JSON", key="show_raw_structured_output"):
                st.json(st.session_state.get("replicate_output", {}))
    else:
            st.caption("Live feedback will appear here as soon as a turn is analysed.")

# ── Debug & Diagnostics (collapsible) ────────────────────────────────────────
if st.session_state.get("session_mode") == "active":
        components.html(
                """
                <script>
                    const doc = window.parent.document;
                    const heading = (text) => [...doc.querySelectorAll('h1, h3')].find((node) => node.textContent.trim().startsWith(text));
                    const hideBlock = (text) => {
                        const node = heading(text);
                        if (!node) return;
                        const block = node.closest('[data-testid="stVerticalBlock"]');
                        if (block) block.classList.add('active-only-hidden');
                    };
                    hideBlock('Instant Spoken English Feedback');
                    hideBlock('Step 2: Review Audio');
                    hideBlock('Step 3: Run Analysis');
                    hideBlock('Step 4: Feedback Output');
                    const inputLabel = [...doc.querySelectorAll('[data-testid="stMarkdownContainer"]')].find((node) => node.textContent.trim() === 'Choose audio input');
                    if (inputLabel) inputLabel.closest('[data-testid="stVerticalBlock"]')?.classList.add('active-only-hidden');
                </script>
                """,
                height=0,
        )
render_error_log()

if DEV_MODE:
    render_live_debug_monitor()

if DEV_MODE:
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
                "deepgram_key_present": bool(os.environ.get("DEEPGRAM_API_KEY")),
                "deepgram_streaming_latency": st.session_state.get("deepgram_streaming_latency"),
                "deepgram_total_time": st.session_state.get("deepgram_total_time"),
                "event_log_path": APP_EVENT_LOG,
                "analysis_timing": st.session_state.get("analysis_timing", {}),
                "turn_analysis": {
                    **st.session_state.turn_analysis_worker.snapshot()[1],
                    "estimated_cost_usd": round(
                        (st.session_state.turn_analysis_worker.snapshot()[1]["prompt_tokens"] * 0.00000015)
                        + (st.session_state.turn_analysis_worker.snapshot()[1]["completion_tokens"] * 0.00000060),
                        6,
                    ),
                },
            }
        )
        recent_events = read_recent_events(limit=12)
        if recent_events:
            st.code("\n".join(recent_events), language="json")
