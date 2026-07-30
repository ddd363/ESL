import base64
import concurrent.futures
import html
import hashlib
import io
import json
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

# ---------------------------------------------------------------------------
# LanguageTool API (free public endpoint — no key required).
# ---------------------------------------------------------------------------
LT_API_URL = "https://api.languagetool.org/v2/check"
LT_API_TIMEOUT_S = 15
LT_HTTP = requests.Session()


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
    with st.expander("Live Debug Monitor", expanded=True):
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
        return "SPEAKER_00"
    return f"SPEAKER_{idx:02d}"


def _format_deepgram_alt_with_speakers(alternative):
    if not isinstance(alternative, dict):
        return ""
    words = alternative.get("words")
    if not isinstance(words, list) or not words:
        return (alternative.get("transcript") or "").strip()

    chunks = []
    current_speaker = None
    current_words = []
    for word in words:
        if not isinstance(word, dict):
            continue
        token = (word.get("punctuated_word") or word.get("word") or "").strip()
        if not token:
            continue
        speaker = word.get("speaker")
        if speaker is None:
            speaker = current_speaker if current_speaker is not None else 0
        if current_speaker is None:
            current_speaker = speaker
        if speaker != current_speaker and current_words:
            chunks.append(f"[{_speaker_label(current_speaker)}]: {' '.join(current_words)}")
            current_words = [token]
            current_speaker = speaker
        else:
            current_words.append(token)

    if current_words:
        chunks.append(f"[{_speaker_label(current_speaker)}]: {' '.join(current_words)}")

    if not chunks:
        return (alternative.get("transcript") or "").strip()
    return "\n".join(chunks).strip()


class DeepgramStreamingClient:
    def __init__(self, api_key, sample_rate):
        self.api_key = api_key
        self.sample_rate = int(sample_rate) if sample_rate else 16000
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
        self.final_segments = []
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

    def _build_url(self):
        params = {
            "encoding": "linear16",
            "sample_rate": str(self.sample_rate),
            "channels": "1",
            "interim_results": "true",
            "punctuate": "true",
            "smart_format": "true",
            "diarize": "true",
        }
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
            encoding="linear16",
            sample_rate=self.sample_rate,
            channels=1,
            interim_results=True,
            punctuate=True,
            smart_format=True,
            diarize=True,
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
        log_event("deepgram_stream_close", status_code=status_code, message=message)

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

                idle_s = now - self.last_audio_sent_at

                # ── KeepAlive ping every 3s of audio silence (Deepgram requires this) ──
                # NOTE: No idle-based forced disconnect. Deepgram stays open during silence.
                if idle_s < 3.0:
                    continue
                ws_app = self.ws
                if not ws_app:
                    continue
                try:
                    ws_app.send(json.dumps({"type": "KeepAlive"}))
                    log_event("deepgram_keepalive_sent", idle_s=round(idle_s, 1))
                except Exception as e:
                    self.error = str(e)
                    with self.lock:
                        if self.disconnected_since is None:
                            self.disconnected_since = perf_counter()
                    self.connected_event.clear()
                    log_event("deepgram_stream_keepalive_error", error=self.error)

        self.keepalive_thread = threading.Thread(target=_loop, daemon=True)
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
            formatted = _format_deepgram_alt_with_speakers(alt)
            if not formatted:
                return
            with self.lock:
                if is_final:
                    self.final_segments.append(formatted)
                    self.interim_text = ""
                else:
                    self.interim_text = formatted
            # Push to module-level transcript_queue so the UI can drain it
            # and trigger st.rerun() without blocking the main thread.
            transcript_queue.put(formatted)

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
            final_text = "\n".join([seg for seg in self.final_segments if seg]).strip()
            interim_text = (self.interim_text or "").strip()
        live = "\n".join([part for part in [final_text, interim_text] if part]).strip()
        return {
            "final": final_text,
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
        }

    def close(self):
        total_s = perf_counter() - self.started_at
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
            self.stop_event.set()
            self.connected_event.clear()
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=1.2)
            if self.keepalive_thread and self.keepalive_thread.is_alive():
                self.keepalive_thread.join(timeout=1.2)
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
    # LanguageTool grammar check
    if "lt_checking" not in st.session_state:
        st.session_state.lt_checking = False
    if "lt_future" not in st.session_state:
        st.session_state.lt_future = None
    if "lt_started_at" not in st.session_state:
        st.session_state.lt_started_at = None
    if "lt_error" not in st.session_state:
        st.session_state.lt_error = ""
    if "lt_matches" not in st.session_state:
        st.session_state.lt_matches = []
    if "replicate_transcript_corrected" not in st.session_state:
        st.session_state.replicate_transcript_corrected = ""
    if "lt_applied_corrections" not in st.session_state:
        st.session_state.lt_applied_corrections = {}
    if "lt_auto_triggered_for" not in st.session_state:
        st.session_state.lt_auto_triggered_for = ""
    if "debug_log_autorefresh" not in st.session_state:
        st.session_state.debug_log_autorefresh = False
    if "debug_log_important_only" not in st.session_state:
        st.session_state.debug_log_important_only = True
    if "debug_log_lines" not in st.session_state:
        st.session_state.debug_log_lines = 80


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


def _deepgram_has_key():
    return bool(os.environ.get("DEEPGRAM_API_KEY"))


def start_deepgram_stream(sample_rate):
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        st.session_state.deepgram_streamer = None
        return
    
    # Close existing streamer if any, and save its final segments
    old_streamer = st.session_state.get("deepgram_streamer")
    if old_streamer:
        try:
            snap = old_streamer.snapshot()
            old_final = snap.get("final") or ""
            if old_final:
                acc = st.session_state.get("deepgram_accumulated_transcript", "")
                st.session_state.deepgram_accumulated_transcript = "\n".join([p for p in [acc, old_final] if p]).strip()
            old_streamer.close()
        except Exception:
            pass

    try:
        streamer = DeepgramStreamingClient(api_key=api_key, sample_rate=sample_rate or 16000)
        streamer.start()
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
        st.session_state.deepgram_final_transcript = st.session_state.get("deepgram_accumulated_transcript", "")
        st.session_state.deepgram_live_transcript = st.session_state.deepgram_final_transcript
        return
    
    snap = streamer.snapshot()
    current_final = snap.get("final") or ""
    acc = st.session_state.get("deepgram_accumulated_transcript", "")
    full_final = "\n".join([p for p in [acc, current_final] if p]).strip()
    interim = snap.get("interim") or ""

    st.session_state.deepgram_live_transcript = "\n".join([p for p in [full_final, interim] if p]).strip()
    st.session_state.deepgram_final_transcript = full_final
    
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
        # ── Critical: persist the captured final into accumulated BEFORE the
        # streamer is set to None.  On the next rerun, sync_deepgram_transcript_state
        # finds no streamer and falls back to deepgram_accumulated_transcript.
        # Without this, every Stop wipes the live transcript.
        final = (st.session_state.get("deepgram_final_transcript") or "").strip()
        if final:
            st.session_state.deepgram_accumulated_transcript = final
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
    # Clear LanguageTool state with the transcript
    st.session_state.lt_matches = []
    st.session_state.lt_error = ""
    st.session_state.replicate_transcript_corrected = ""
    st.session_state.lt_applied_corrections = {}
    st.session_state.lt_auto_triggered_for = ""


def reset_for_rerecord():
    clear_recording_state(delete_file=False)
    reset_analysis_state()
    st.session_state.audio_ready = False
    st.session_state.active_audio_signature = None
    st.session_state.deepgram_live_transcript = ""
    st.session_state.deepgram_final_transcript = ""
    st.session_state.deepgram_streaming_latency = None
    st.session_state.deepgram_total_time = None


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


# ---------------------------------------------------------------------------
# LanguageTool background job
# ---------------------------------------------------------------------------
def run_languagetool_check(text):
    """POST text to the public LanguageTool API and return a dict of matches."""
    started = perf_counter()
    try:
        resp = LT_HTTP.post(
            LT_API_URL,
            data={"text": text, "language": "en-US"},
            timeout=LT_API_TIMEOUT_S,
        )
        resp.raise_for_status()
        data = resp.json()
        matches = data.get("matches") or []
        elapsed = perf_counter() - started
        # Summarise issue type distribution for diagnostics
        type_counts: dict = {}
        for m in matches:
            rule_cat = ((m.get("rule") or {}).get("category") or {}).get("id", "OTHER")
            type_counts[rule_cat] = type_counts.get(rule_cat, 0) + 1
        log_event(
            "lt_check_completed",
            matches_count=len(matches),
            type_counts=type_counts,
            latency_s=round(elapsed, 3),
            text_chars=len(text),
        )
        return {"matches": matches, "latency_s": elapsed}
    except Exception as e:
        elapsed = perf_counter() - started
        log_event("lt_check_error", error=str(e), latency_s=round(elapsed, 3))
        raise


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
                st.session_state.replicate_transcript_corrected = result.get("transcript") or ""
                st.session_state.lt_applied_corrections = {}
                st.session_state.lt_matches = []
                st.session_state.lt_error = ""
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
                # Auto-trigger LT check on the fresh Replicate transcript
                fresh_text = (st.session_state.replicate_transcript or "").strip()
                if fresh_text and not st.session_state.get("lt_checking"):
                    st.session_state.lt_checking = True
                    st.session_state.lt_started_at = perf_counter()
                    st.session_state.lt_auto_triggered_for = fresh_text
                    st.session_state.lt_future = BACKGROUND_EXECUTOR.submit(
                        run_languagetool_check, fresh_text
                    )
                    log_event("lt_auto_triggered", text_chars=len(fresh_text))
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

    # ── LanguageTool check polling ─────────────────────────────────────────────
    if st.session_state.get("lt_checking"):
        lt_future = st.session_state.get("lt_future")
        lt_started = st.session_state.get("lt_started_at")
        if lt_future is not None and lt_future.done():
            try:
                lt_result = lt_future.result()
                matches = lt_result.get("matches") or []
                st.session_state.lt_matches = matches
                st.session_state.lt_error = ""
                log_event(
                    "lt_poll_completed",
                    matches_count=len(matches),
                    latency_s=round(lt_result.get("latency_s", 0), 3),
                )
            except Exception as e:
                st.session_state.lt_error = f"LanguageTool check failed: {e}"
                st.session_state.lt_matches = []
                log_event("lt_poll_error", error=str(e))
            st.session_state.lt_checking = False
            st.session_state.lt_future = None
            st.session_state.lt_started_at = None
            should_rerun = True
        elif lt_started is not None and (perf_counter() - float(lt_started)) > LT_API_TIMEOUT_S + 5:
            st.session_state.lt_checking = False
            st.session_state.lt_future = None
            st.session_state.lt_started_at = None
            st.session_state.lt_error = "LanguageTool check timed out."
            log_event("lt_poll_timeout")
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


# ---------------------------------------------------------------------------
# LanguageTool annotation helpers
# ---------------------------------------------------------------------------
_LT_COLORS = {
    "SPELL": ("#9c27b0", "Spelling"),
    "TYPOS": ("#9c27b0", "Spelling"),
    "GRAMMAR": ("#d32f2f", "Grammar"),
    "STYLE": ("#f57c00", "Style"),
    "REDUNDANCY": ("#f57c00", "Style"),
    "PUNCTUATION": ("#1565c0", "Punctuation"),
    "COLLOQUIALISMS": ("#00796b", "Register"),
}


def _lt_issue_color(rule_category):
    cat = (rule_category or "").upper()
    for key, val in _LT_COLORS.items():
        if key in cat:
            return val
    return ("#795548", "Other")


def _build_lt_annotated_html(text, matches):
    """Return an HTML string with colored underlines for each LT match."""
    if not text:
        return ""
    if not matches:
        return f'<div class="clean-text-block">{html.escape(text)}</div>'

    # Sort by offset, drop overlapping matches (keep first)
    sorted_m = sorted(matches, key=lambda m: m.get("offset", 0))
    valid = []
    last_end = -1
    for m in sorted_m:
        offset = m.get("offset", 0)
        length = m.get("length", 0)
        if offset >= last_end and length > 0:
            valid.append(m)
            last_end = offset + length

    parts = []
    cursor = 0
    for m in valid:
        offset = m.get("offset", 0)
        length = m.get("length", 0)
        end = offset + length
        message = m.get("message") or ""
        replacements = [r.get("value", "") for r in (m.get("replacements") or [])[:3]]
        rule_cat = ((m.get("rule") or {}).get("category") or {}).get("id", "")
        color, label = _lt_issue_color(rule_cat)
        tooltip = f"{label}: {html.escape(message)}"
        if replacements:
            tooltip += f" → {html.escape(' / '.join(replacements))}"

        if offset > cursor:
            parts.append(html.escape(text[cursor:offset]))
        matched_text = html.escape(text[offset:end])
        parts.append(
            f'<span class="lt-match" '
            f'style="border-bottom: 2px solid {color}; color: {color};" '
            f'title="{tooltip}">{matched_text}</span>'
        )
        cursor = end
    if cursor < len(text):
        parts.append(html.escape(text[cursor:]))

    return f'<div class="clean-text-block lt-annotated">{" ".join(parts)}</div>'


def _apply_lt_corrections(original_text, matches, applied_dict):
    """Reconstruct text with selected corrections applied (by match index)."""
    if not applied_dict or not matches:
        return original_text
    # Build list of (offset, length, replacement) sorted by offset desc so
    # we splice from right-to-left to keep earlier offsets valid.
    patches = []
    for idx_s, replacement in applied_dict.items():
        try:
            idx = int(idx_s)
            m = matches[idx]
            patches.append((m.get("offset", 0), m.get("length", 0), replacement))
        except (IndexError, KeyError, ValueError):
            continue
    patches.sort(key=lambda p: p[0], reverse=True)
    result = original_text
    for offset, length, replacement in patches:
        result = result[:offset] + replacement + result[offset + length :]
    return result


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

# ── Page title ────────────────────────────────────────────────────────────────
with st.container():
    st.title("Instant Spoken English Feedback")
    st.caption("Record your speech and receive structured IELTS-style feedback.")

# =============================================================================
# STEP 1 — Audio Input
# =============================================================================
uploaded = None
audio_preview_data = None
audio_preview_path = None
audio_filename = ""

with st.container():
    st.markdown("### Step 1: Audio Input")
    audio_source = st.radio(
        "Choose audio input",
        ["Upload", "Load", "Record"],
        index=["Upload", "Load", "Record"].index(st.session_state.get("audio_source", "Record")),
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

    else:  # Record
        recorder_value = render_audio_recorder()
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
            st.session_state.deepgram_accumulated_transcript = ""
            st.session_state.deepgram_stop_requested_at = None
            sample_rate = None
            if isinstance(recorder_payload, dict):
                sample_rate = recorder_payload.get("sample_rate")
            start_deepgram_stream(sample_rate)

        if recorder_event_name == "stop_clicked":
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
            log_event(
                "record_saved",
                recorded_path=recorded_path,
                saved_filename=os.path.basename(recorded_path),
                bytes_len=len(wav_bytes),
            )

        # Live recording status + Deepgram live transcript inline
        if st.session_state.get("is_recording"):
            elapsed = 0
            if st.session_state.get("recording_started_at"):
                elapsed_delta = datetime.now(timezone.utc) - st.session_state.recording_started_at
                elapsed = max(0, int(elapsed_delta.total_seconds()))
            st.info(f"Recording... {elapsed // 60:02d}:{elapsed % 60:02d}")
            if not _deepgram_has_key():
                st.warning("DEEPGRAM_API_KEY missing — live transcript disabled. Recording continues normally.")
            live_text = (st.session_state.get("deepgram_live_transcript") or "").strip()
            st.markdown("**Live Transcript (Deepgram)**")
            st.markdown(
                f'<div class="clean-text-block">{html.escape(live_text) if live_text else "Listening..."}</div>',
                unsafe_allow_html=True,
            )
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

        if recording_ready():
            if st.button("Re-record"):
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

# =============================================================================
# STEP 2 — Review Audio
# =============================================================================
st.markdown("---")
with st.container():
    st.markdown("### Step 2: Review Audio")

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
        st.markdown(
            f'<div class="clean-text-block">{html.escape(replicate_preview) if replicate_preview else "No Replicate transcript yet."}</div>',
            unsafe_allow_html=True,
        )
    with col_dg:
        st.markdown("**Deepgram Transcript**")
        deepgram_preview = (
            st.session_state.get("deepgram_final_transcript")
            or st.session_state.get("deepgram_live_transcript")
            or ""
        ).strip()
        st.markdown(
            f'<div class="clean-text-block">{html.escape(deepgram_preview) if deepgram_preview else "No Deepgram transcript yet."}</div>',
            unsafe_allow_html=True,
        )

# =============================================================================
# STEP 2.5 — Grammar & Wording Check (LanguageTool)
# =============================================================================
st.markdown("---")
with st.container():
    st.markdown("### Step 2.5: Grammar & Wording Check")

    _lt_source_text = (st.session_state.get("replicate_transcript") or "").strip()
    _lt_matches = st.session_state.get("lt_matches") or []
    _lt_checking = st.session_state.get("lt_checking", False)
    _lt_error = st.session_state.get("lt_error") or ""
    _applied = st.session_state.get("lt_applied_corrections") or {}

    # ── Controls row ─────────────────────────────────────────────────────────
    lt_col_btn, lt_col_status = st.columns([1, 3])
    with lt_col_btn:
        lt_check_clicked = st.button(
            "Check Grammar",
            disabled=not bool(_lt_source_text) or _lt_checking,
            help="Send Replicate transcript to LanguageTool and highlight issues.",
        )
    with lt_col_status:
        if _lt_checking:
            st.info("Checking grammar… please wait.")
        elif _lt_error:
            st.error(_lt_error)
        elif _lt_matches:
            type_counts: dict = {}
            for _m in _lt_matches:
                _cat = ((_m.get("rule") or {}).get("category") or {}).get("id", "OTHER")
                type_counts[_cat] = type_counts.get(_cat, 0) + 1
            _summary = ", ".join(f"{v} {k.lower()}" for k, v in type_counts.items())
            st.success(f"{len(_lt_matches)} issue{'s' if len(_lt_matches) != 1 else ''} found: {_summary}")
        elif _lt_source_text:
            st.caption("No issues detected, or check has not run yet.")
        else:
            st.caption("Transcribe audio first to enable grammar check.")

    # ── Manual trigger handler ────────────────────────────────────────────────
    if lt_check_clicked and _lt_source_text and not st.session_state.get("lt_checking"):
        st.session_state.lt_checking = True
        st.session_state.lt_started_at = perf_counter()
        st.session_state.lt_matches = []
        st.session_state.lt_error = ""
        st.session_state.lt_applied_corrections = {}
        st.session_state.replicate_transcript_corrected = _lt_source_text
        st.session_state.lt_future = BACKGROUND_EXECUTOR.submit(
            run_languagetool_check, _lt_source_text
        )
        log_event("lt_manual_triggered", text_chars=len(_lt_source_text))
        st.rerun()

    # ── Annotated transcript ──────────────────────────────────────────────────
    if _lt_source_text:
        st.markdown("**Original transcript with issues highlighted** *(hover for detail)*")
        _annotated_html = _build_lt_annotated_html(_lt_source_text, _lt_matches)
        st.markdown(_annotated_html, unsafe_allow_html=True)

    # ── Suggestion panel ─────────────────────────────────────────────────────
    if _lt_matches:
        with st.expander(f"Show all {len(_lt_matches)} suggestions", expanded=False):
            for _i, _m in enumerate(_lt_matches):
                _offset = _m.get("offset", 0)
                _length = _m.get("length", 0)
                _word = _lt_source_text[_offset: _offset + _length]
                _message = _m.get("message") or ""
                _rule_cat = ((_m.get("rule") or {}).get("category") or {}).get("id", "")
                _color, _label = _lt_issue_color(_rule_cat)
                _replacements = [r.get("value", "") for r in (_m.get("replacements") or [])[:4]]

                _sc1, _sc2, _sc3 = st.columns([1, 3, 2])
                with _sc1:
                    st.markdown(
                        f'<span class="lt-cat-badge" style="background:{_color}22; color:{_color}; border:1px solid {_color}44;">'
                        f"{_label}</span>",
                        unsafe_allow_html=True,
                    )
                with _sc2:
                    _context = html.escape(
                        _lt_source_text[max(0, _offset - 20): _offset + _length + 20]
                    )
                    st.markdown(
                        f"**…{_context}…** — {html.escape(_message)}",
                        unsafe_allow_html=True,
                    )
                with _sc3:
                    if _replacements:
                        for _ri, _rep in enumerate(_replacements):
                            _btn_key = f"lt_apply_{_i}_{_ri}"
                            if st.button(
                                f'→ "{_rep}"',
                                key=_btn_key,
                                help=f"Replace '{_word}' with '{_rep}'",
                            ):
                                _applied_new = dict(st.session_state.lt_applied_corrections)
                                _applied_new[str(_i)] = _rep
                                st.session_state.lt_applied_corrections = _applied_new
                                st.session_state.replicate_transcript_corrected = (
                                    _apply_lt_corrections(
                                        _lt_source_text,
                                        _lt_matches,
                                        _applied_new,
                                    )
                                )
                                log_event(
                                    "lt_correction_applied",
                                    match_index=_i,
                                    original_word=_word,
                                    replacement=_rep,
                                )
                                st.rerun()
                    else:
                        st.caption("No suggestions")
                st.markdown("<hr style='margin:4px 0; border-color:rgba(128,128,128,0.2)'>", unsafe_allow_html=True)

    # ── Corrected transcript editor ───────────────────────────────────────────
    if _lt_source_text:
        _has_corrections = bool(_applied)
        st.markdown("**Corrected transcript** *(editable — used for Step 3 analysis)*")
        _corrected_display = st.session_state.get("replicate_transcript_corrected") or _lt_source_text

        _lt_edit_col, _lt_reset_col = st.columns([5, 1])
        with _lt_edit_col:
            _corrected_edited = st.text_area(
                "corrected_transcript_label",
                value=_corrected_display,
                height=140,
                key="replicate_transcript_corrected",
                label_visibility="collapsed",
            )
        with _lt_reset_col:
            if _has_corrections:
                if st.button("↺ Reset", help="Undo all applied corrections and restore original transcript"):
                    st.session_state.lt_applied_corrections = {}
                    st.session_state.replicate_transcript_corrected = _lt_source_text
                    log_event("lt_corrections_reset")
                    st.rerun()

        if _has_corrections:
            _napplied = len(_applied)
            st.caption(f"{_napplied} correction{'s' if _napplied != 1 else ''} applied. Step 3 will use the corrected version.")

# =============================================================================
# STEP 3 — Run Analysis
# =============================================================================
st.markdown("---")
with st.container():
    st.markdown("### Step 3: Run Analysis")

    transcribe_clicked = st.button(
        "Transcribe via Replicate",
        type="primary",
        disabled=not st.session_state.audio_ready or st.session_state.get("transcribing", False),
    )

    st.radio(
        "Use transcript for analysis:",
        ["Replicate", "Replicate (corrected)", "Deepgram"],
        key="analysis_transcript_choice",
        horizontal=True,
    )

    with st.expander("OpenAI Prompt (Editable)", expanded=False):
        st.text_area(
            "System prompt used for feedback generation",
            key="openai_system_prompt",
            height=260,
            on_change=persist_openai_prompt,
        )

    selected_for_analysis = st.session_state.get("analysis_transcript_choice", "Replicate")
    selected_text = (
        (
            st.session_state.get("replicate_transcript_corrected") or
            st.session_state.get("replicate_transcript", "")
        )
        if selected_for_analysis == "Replicate (corrected)"
        else (
            st.session_state.get("replicate_transcript", "")
            if selected_for_analysis == "Replicate"
            else (
                st.session_state.get("deepgram_final_transcript", "")
                or st.session_state.get("deepgram_live_transcript", "")
            )
        )
    )
    analyse_clicked = st.button(
        "Analyse via OpenAI",
        disabled=(
            not bool((selected_text or "").strip())
            or st.session_state.get("analysing", False)
            or st.session_state.get("transcribing", False)
        ),
    )

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
    transcript_for_analysis = (
        st.session_state.get("replicate_transcript", "")
        if selected_for_analysis == "Replicate"
        else (
            st.session_state.get("deepgram_final_transcript", "")
            or st.session_state.get("deepgram_live_transcript", "")
        )
    )
    transcript_for_analysis = (transcript_for_analysis or "").strip()
    if not transcript_for_analysis:
        st.warning("No transcript available for selected source.")
    else:
        st.session_state.analysis_error = ""
        st.session_state.analysis_complete = False
        st.session_state.transcript = transcript_for_analysis
        st.session_state["transcript_source"] = (
            "replicate" if selected_for_analysis == "Replicate" else "deepgram"
        )
        st.session_state.analysing = True
        st.session_state.analysis_started_at = perf_counter()
        st.session_state.analysis_future = BACKGROUND_EXECUTOR.submit(
            run_openai_analysis_background_job,
            transcript_for_analysis,
            st.session_state.get("openai_system_prompt", DEFAULT_OPENAI_SYSTEM_PROMPT),
            bool(st.session_state.get("allow_rule_based_fallback_on_openai_failure", False)),
        )
        log_event(
            "analysis_background_started",
            transcript_choice=selected_for_analysis,
            transcript_chars=len(transcript_for_analysis),
            timeout_s=JOB_TIMEOUT_S,
        )
        st.rerun()

# ── Auto-rerun while background tasks are live so poll_background_jobs()
# can detect completion without requiring user interaction.
# IMPORTANT: use time.sleep + st.rerun() — NOT a JS location.reload().
# JS reload creates a brand-new Streamlit WebSocket session which resets
# session_state and orphans the transcribe_future permanently.
# st.rerun() stays within the same session, so session_state is preserved.
if st.session_state.get("transcribing") or st.session_state.get("analysing") or st.session_state.get("lt_checking"):
    time.sleep(2)
    st.rerun()

# =============================================================================
# STEP 4 — Feedback Output
# =============================================================================
st.markdown("---")
with st.container():
    st.markdown("### Step 4: Feedback Output")

    if st.session_state.analysis_complete or bool(st.session_state.get("transcript")):
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
            st.caption("Transcript is ready. Feedback is still generating...")
        st.markdown("</div>", unsafe_allow_html=True)

        # Transcript used for analysis
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("#### Transcript Used for Analysis")
        transcript_text = st.session_state.get("transcript") or ""
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
        st.caption("Complete Step 3 to view feedback and transcript.")

# ── Debug & Diagnostics (collapsible) ────────────────────────────────────────
render_live_debug_monitor()

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
        }
    )
    recent_events = read_recent_events(limit=12)
    if recent_events:
        st.code("\n".join(recent_events), language="json")
