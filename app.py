"""Lesson Transcriber.

Live lesson transcription with two microphones (teacher + student), a single
timestamped transcript that autosaves during class, and an optional
post-lesson high-accuracy pass through Replicate (whisper-diarization).
"""

import base64
import collections
import concurrent.futures
import hashlib
import html
import json
import math
import os
import sys
import threading
import time
import urllib.parse
from datetime import datetime, timezone
from time import perf_counter

import httpx
import replicate
import streamlit as st
import streamlit.components.v1 as components

import live_runtime
from live_sentences import (
    in_feedback_window,
    index_words_to_sentences,
    low_confidence_words,
    newest_word_end,
    settled_sentences,
    split_sentences,
    update_feedback_windows,
)

st.set_page_config(
    page_title="Lesson Transcriber",
    layout="wide",
    initial_sidebar_state="collapsed",
)

try:
    import websocket
except Exception:
    websocket = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AUDIO_DIR = "audio"
APP_EVENT_LOG = os.environ.get("ESL_EVENT_LOG", "/tmp/esl_app_events.log")
LEGACY_AUDIO_EXTS = {".wav", ".m4a"}
LESSON_AUDIO_EXTS = {".wav", ".m4a", ".webm", ".ogg", ".mp4", ".mp3"}
SOURCE_LABELS = {"student": "Student", "teacher": "Teacher"}
AUTOSAVE_INTERVAL_S = 5.0
TURN_GAP_SECONDS = 8.0
TURN_MAX_SECONDS = 60.0
REPLICATE_TIMEOUT_S = 1800
LIVE_SETTLE_SECONDS = 3.0        # how far behind the newest word before a sentence is trusted
# Only single-word utterances are skipped - "Yes." / "Yeah." / "Yep." are 47% of
# the student's "sentences" and cannot carry a grammar error. At 5 this hid 81%
# of the transcript, so the counter reported work with nothing on screen to show
# for it. Set to 1 to underline literally every sentence.
LIVE_MIN_WORDS = 2
LIVE_LOW_CONFIDENCE = 0.7        # below this a word is flagged to the model as unreliable
LIVE_FEEDBACK_MAX_INFLIGHT = 3
LIVE_FEEDBACK_MAX_CALLS_PER_LESSON = 400
LIVE_FEEDBACK_CONTEXT_TURNS = 2
LIVE_FEEDBACK_TIMEOUT_S = 45     # DeepSeek is occasionally very slow to first byte
LIVE_FEEDBACK_MAX_ATTEMPTS = 3   # a timeout is transient; keep the sentence grey and retry
LIVE_FEEDBACK_MAX_TOKENS = 300
LIVE_FEEDBACK_SOURCE = "student"

DEEPGRAM_RESTART_MIN_INTERVAL_S = 1.0
DEEPGRAM_SEND_STALL_S = 4.0      # a write in flight this long means a wedged socket
DEEPGRAM_DEFAULT_MODEL = os.environ.get("DEEPGRAM_MODEL", "nova-3")
DEEPGRAM_DEFAULT_LANGUAGE = os.environ.get("DEEPGRAM_LANGUAGE", "en")
DEEPGRAM_KEYTERMS = tuple(
    term.strip() for term in os.environ.get("DEEPGRAM_KEYTERMS", "").split(",") if term.strip()
)
DEEPGRAM_SMART_FORMAT = os.environ.get("DEEPGRAM_SMART_FORMAT", "true").lower() == "true"

RECORDER_COMPONENT_DIR = os.path.join(
    os.path.dirname(__file__), "components", "audio_recorder"
)
audio_recorder_component = components.declare_component(
    "audio_recorder", path=RECORDER_COMPONENT_DIR
)
BACKGROUND_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2)
# Lives in live_runtime so it survives a rerun; app.py's module body re-executes
# about twice a second during a lesson.
LIVE_FEEDBACK_EXECUTOR = live_runtime.EXECUTOR

MIME_EXTENSIONS = {
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/mp4": ".m4a",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
}


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


def _format_clock(seconds):
    if not isinstance(seconds, (int, float)) or seconds < 0:
        seconds = 0
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Deepgram live streaming (one websocket per microphone source)
# ---------------------------------------------------------------------------
class DeepgramStreamingClient:
    def __init__(
        self,
        api_key,
        sample_rate,
        model=DEEPGRAM_DEFAULT_MODEL,
        language=DEEPGRAM_DEFAULT_LANGUAGE,
        keyterms=DEEPGRAM_KEYTERMS,
        smart_format=DEEPGRAM_SMART_FORMAT,
    ):
        self.api_key = api_key
        self.sample_rate = int(sample_rate) if sample_rate else 16000
        self.model = model
        self.language = language
        self.keyterms = tuple(keyterms)
        self.smart_format = bool(smart_format)
        self.ws = None
        self.thread = None
        self.keepalive_thread = None
        self.stop_event = threading.Event()
        self.connected_event = threading.Event()
        self.lock = threading.Lock()
        self.started_at = perf_counter()
        self.last_audio_sent_at = self.started_at
        self.disconnected_since = None
        self.final_words = []
        self.interim_text = ""
        self.error = None
        # Audio waiting to go out. The script thread only ever appends here;
        # a sender thread owns the socket. See send_pcm16.
        self.pending_chunks = collections.deque(maxlen=40)
        self.queue_lock = threading.Lock()
        self.queue_event = threading.Event()
        self.sender_thread = None
        self.bytes_enqueued_total = 0
        self.send_in_flight_since = None
        self.bytes_sent_total = 0
        self.chunks_received = 0
        self.transcript_events_received = 0
        # Signalled when Deepgram delivers the first is_final=True frame after
        # CloseStream is sent, confirming all final transcripts have arrived.
        self.final_flush_event = threading.Event()

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
        ]
        params.extend(("keyterm", term) for term in self.keyterms)
        return "wss://api.deepgram.com/v1/listen?" + urllib.parse.urlencode(params)

    def _on_open(self, _ws):
        with self.lock:
            self.error = None
            self.disconnected_since = None
        self.connected_event.set()
        self._flush_pending_chunks()
        log_event("deepgram_stream_open", sample_rate=self.sample_rate, model=self.model)

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
        log_event("deepgram_stream_close", status_code=status_code, message=message)

    def _flush_pending_chunks(self):
        """Wake the sender; it drains whatever is queued."""
        self.queue_event.set()

    def _send_keepalive_if_idle(self, now):
        """KeepAlive frames hold the socket open through lesson pauses."""
        with self.lock:
            idle_s = now - self.last_audio_sent_at
            ws_app = self.ws
        if idle_s < 3.0 or not ws_app:
            return
        try:
            self.send_in_flight_since = perf_counter()
            ws_app.send(json.dumps({"type": "KeepAlive"}))
            self.send_in_flight_since = None
            with self.lock:
                self.last_audio_sent_at = now
        except Exception as e:
            self.send_in_flight_since = None
            self.error = str(e)
            with self.lock:
                if self.disconnected_since is None:
                    self.disconnected_since = perf_counter()
            self.connected_event.clear()
            log_event("deepgram_keepalive_error", error=self.error)

    def _start_keepalive(self):
        if self.keepalive_thread and self.keepalive_thread.is_alive():
            return

        def _loop():
            while not self.stop_event.is_set():
                time.sleep(0.5)
                if self.stop_event.is_set():
                    break
                if not self.connected_event.is_set():
                    continue
                self._send_keepalive_if_idle(perf_counter())

        self.keepalive_thread = threading.Thread(
            target=_loop, daemon=True, name="deepgram-keepalive"
        )
        self.keepalive_thread.start()

    def _on_message(self, _ws, message):
        try:
            payload = json.loads(message)
        except Exception:
            return
        if not isinstance(payload, dict) or payload.get("type") != "Results":
            return
        self.transcript_events_received += 1
        is_final = bool(payload.get("is_final"))
        if is_final:
            self.final_flush_event.set()
        alternatives = (payload.get("channel") or {}).get("alternatives") or []
        if not alternatives:
            return
        alt = alternatives[0]
        transcript = (alt.get("transcript") or "").strip()
        if is_final:
            records = []
            for item in alt.get("words") or []:
                if not isinstance(item, dict):
                    continue
                token = (item.get("punctuated_word") or item.get("word") or "").strip()
                if not token:
                    continue
                records.append({
                    "word": token,
                    "start": item.get("start"),
                    "end": item.get("end"),
                    "confidence": item.get("confidence"),
                })
            with self.lock:
                if records:
                    self.final_words.extend(records)
                self.interim_text = ""
        elif transcript:
            with self.lock:
                self.interim_text = transcript

    def start(self):
        if websocket is None:
            self.error = "websocket_client_missing"
            log_event("deepgram_stream_unavailable", reason=self.error)
            return
        headers = [f"Authorization: Token {self.api_key}"]
        self.ws = websocket.WebSocketApp(
            self._build_url(),
            header=headers,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        ws_app = self.ws
        self.thread = threading.Thread(
            target=lambda: ws_app.run_forever(ping_interval=15, ping_timeout=8),
            daemon=True,
        )
        self.thread.start()
        self.sender_thread = threading.Thread(
            target=self._sender_loop, daemon=True, name="deepgram-sender"
        )
        self.sender_thread.start()
        self._start_keepalive()

    @property
    def is_active(self):
        # Read without the lock: a send that has been in flight far longer than
        # any healthy write means the socket is wedged, and the caller needs to
        # learn that even though connected_event still says "connected".
        stalled_since = self.send_in_flight_since
        if stalled_since is not None and perf_counter() - stalled_since > DEEPGRAM_SEND_STALL_S:
            return False
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
        """Hand audio to the sender thread. Called from the Streamlit script
        thread, so it must never block and never touch the socket.

        websocket send() is a blocking write with no send timeout. On a
        half-open TCP connection - a Wi-Fi switch, a sleeping laptop, Deepgram
        going quiet - it stalls for the OS retransmit timeout, minutes at worst.
        Doing that inline froze the whole app, and doing it while holding
        self.lock froze every render behind it too, because snapshot() wants the
        same lock twice a second.
        """
        if not chunk_bytes or self.stop_event.is_set():
            return
        with self.queue_lock:
            self.pending_chunks.append(chunk_bytes)
            self.bytes_enqueued_total += len(chunk_bytes)
        self.queue_event.set()

    def _sender_loop(self):
        while not self.stop_event.is_set():
            self.queue_event.wait(0.25)
            self.queue_event.clear()
            if self.stop_event.is_set():
                break
            ws_app = self.ws
            if not ws_app or not self.connected_event.is_set():
                continue
            while not self.stop_event.is_set():
                with self.queue_lock:
                    if not self.pending_chunks:
                        break
                    chunk_bytes = self.pending_chunks.popleft()
                # Sent outside every lock, so a stalled write blocks nothing else.
                self.send_in_flight_since = perf_counter()
                try:
                    ws_app.send(chunk_bytes, opcode=2)
                except Exception as e:
                    self.send_in_flight_since = None
                    with self.lock:
                        self.error = str(e)
                        if self.disconnected_since is None:
                            self.disconnected_since = perf_counter()
                    self.connected_event.clear()
                    log_event("deepgram_stream_send_error", error=str(e))
                    break
                self.send_in_flight_since = None
                with self.lock:
                    self.last_audio_sent_at = perf_counter()
                    self.bytes_sent_total += len(chunk_bytes)

    def audio_seconds_sent(self):
        """Audio time this stream covered, for the next stream's timestamp offset.

        Counts what was captured, not what the socket accepted: if a stalled
        connection loses queued chunks, the lesson clock must not shift.
        """
        captured = max(self.bytes_enqueued_total, self.bytes_sent_total)
        return captured / (2.0 * self.sample_rate) if self.sample_rate else 0.0

    def snapshot(self):
        with self.lock:
            return {
                "final_words": list(self.final_words),
                "interim": (self.interim_text or "").strip(),
                "error": self.error,
                "bytes_sent_total": self.bytes_sent_total,
                "chunks_received": self.chunks_received,
                "transcript_events_received": self.transcript_events_received,
                "sample_rate": self.sample_rate,
            }

    def close(self):
        self.stop_event.set()
        self.connected_event.clear()
        try:
            if self.ws:
                try:
                    self.ws.send(json.dumps({"type": "CloseStream"}))
                except Exception:
                    pass
                # Wait briefly for the final is_final frame so the last words
                # spoken before End are not lost.
                self.final_flush_event.wait(timeout=3.0)
                try:
                    self.ws.close()
                except Exception:
                    pass
        finally:
            self.queue_event.set()
            if self.sender_thread and self.sender_thread.is_alive():
                self.sender_thread.join(timeout=0.5)
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=1.2)
            if self.keepalive_thread and self.keepalive_thread.is_alive():
                self.keepalive_thread.join(timeout=1.2)
            log_event("deepgram_stream_stopped", bytes_sent_total=self.bytes_sent_total)


# ---------------------------------------------------------------------------
# Lesson stream management
# ---------------------------------------------------------------------------
def start_deepgram_stream(source, sample_rate):
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        return
    streamers = st.session_state.deepgram_streamers
    existing = streamers.get(source)
    if existing and (existing.is_active or existing.is_connecting):
        return
    if existing:
        # Already dead by the check above, so never wait on its socket here.
        _retire_streamer(source, existing, block=False)
    try:
        streamer = DeepgramStreamingClient(api_key=api_key, sample_rate=sample_rate or 16000)
        streamer.start()
        streamers[source] = streamer
        st.session_state.source_offsets.setdefault(source, 0.0)
        st.session_state.deepgram_last_restart_at = perf_counter()
        log_event("deepgram_stream_start", source=source, sample_rate=sample_rate)
    except Exception as e:
        streamers.pop(source, None)
        log_event("deepgram_stream_start_error", source=source, error=str(e))


def _retire_streamer(source, streamer, block=True):
    """Fold a stream's words into the lesson and advance the time offset.

    `block=False` closes the socket on a throwaway thread. close() waits up to
    3s for Deepgram's final frame and then joins two more threads for 1.2s each
    - over five seconds. That is fine when the lesson is ending, but on a
    mid-lesson restart it runs on the script thread, from handle_audio_chunk,
    and stalls the whole app: the page greys out and audio stops being read
    while we wait. A dying socket must never cost the lesson its transcription.
    """
    snap = streamer.snapshot()
    offset = st.session_state.source_offsets.get(source, 0.0)
    for word in snap["final_words"]:
        st.session_state.lesson_words.append({
            "source": source,
            "word": word["word"],
            "start": (word.get("start") or 0.0) + offset,
            "end": (word.get("end") or 0.0) + offset,
            "confidence": word.get("confidence"),
        })
    st.session_state.source_offsets[source] = offset + streamer.audio_seconds_sent()

    def _close():
        try:
            streamer.close()
        except Exception:
            pass

    if block:
        _close()
    else:
        threading.Thread(target=_close, daemon=True,
                         name=f"deepgram-close-{source}").start()


def restart_dead_stream(source, sample_rate):
    now = perf_counter()
    last_restart = st.session_state.get("deepgram_last_restart_at")
    if last_restart is not None and now - last_restart < DEEPGRAM_RESTART_MIN_INTERVAL_S:
        return
    streamer = st.session_state.deepgram_streamers.pop(source, None)
    if streamer:
        log_event("deepgram_stream_restart", source=source, last_error=streamer.error)
        _retire_streamer(source, streamer, block=False)
    start_deepgram_stream(source, sample_rate)


def end_all_streams():
    streamers = st.session_state.deepgram_streamers
    for source in list(streamers):
        _retire_streamer(source, streamers.pop(source))


def handle_audio_chunk(payload):
    source = payload.get("source") or "student"
    if source not in SOURCE_LABELS:
        source = "student"
    pcm16_base64 = payload.get("pcm16_base64")
    sample_rate = payload.get("sample_rate")
    if not pcm16_base64:
        return
    try:
        chunk_bytes = base64.b64decode(pcm16_base64)
    except Exception as e:
        log_event("deepgram_chunk_decode_error", source=source, error=str(e))
        return
    streamer = st.session_state.deepgram_streamers.get(source)
    if streamer is None:
        start_deepgram_stream(source, sample_rate)
        streamer = st.session_state.deepgram_streamers.get(source)
    elif not streamer.is_active and not streamer.is_connecting:
        restart_dead_stream(source, sample_rate)
        streamer = st.session_state.deepgram_streamers.get(source)
    if streamer is None:
        return
    streamer.chunks_received += 1
    streamer.send_pcm16(chunk_bytes)


# ---------------------------------------------------------------------------
# Transcript assembly, timestamps, autosave
# ---------------------------------------------------------------------------
def collect_lesson_words():
    """All finalized words so far: retired streams + live streams, sorted."""
    words = list(st.session_state.lesson_words)
    for source, streamer in st.session_state.deepgram_streamers.items():
        snap = streamer.snapshot()
        offset = st.session_state.source_offsets.get(source, 0.0)
        for word in snap["final_words"]:
            words.append({
                "source": source,
                "word": word["word"],
                "start": (word.get("start") or 0.0) + offset,
                "end": (word.get("end") or 0.0) + offset,
                "confidence": word.get("confidence"),
            })
    words.sort(key=lambda w: (w.get("start") or 0.0))
    return words


def group_turns(words):
    """Group consecutive same-speaker words into timestamped turns."""
    turns = []
    for word in words:
        token = (word.get("word") or "").strip()
        if not token:
            continue
        start = word.get("start") or 0.0
        current = turns[-1] if turns else None
        new_turn = (
            current is None
            or current["source"] != word["source"]
            or start - current["end"] > TURN_GAP_SECONDS
            or start - current["start"] > TURN_MAX_SECONDS
        )
        if new_turn:
            turns.append({
                "source": word["source"],
                "start": start,
                "end": word.get("end") or start,
                "tokens": [token],
                "words": [word],
            })
        else:
            current["tokens"].append(token)
            current["words"].append(word)
            current["end"] = word.get("end") or current["end"]
    for turn in turns:
        turn["text"] = " ".join(turn["tokens"])
    return turns


def interim_by_source():
    parts = {}
    for source, streamer in st.session_state.deepgram_streamers.items():
        snap = streamer.snapshot()
        if snap["interim"]:
            parts[source] = snap["interim"]
    return parts


def build_transcript_text(turns):
    lines = []
    for turn in turns:
        label = SOURCE_LABELS.get(turn["source"], str(turn["source"]).title())
        lines.append(f"[{_format_clock(turn['start'])}] {label}: {turn['text']}")
    return "\n\n".join(lines)


def lesson_transcript_path(lesson_dir):
    return os.path.join(lesson_dir, "transcript.txt")


def autosave_lesson(force=False):
    lesson_dir = st.session_state.get("lesson_dir")
    if not lesson_dir:
        return
    now = perf_counter()
    last = st.session_state.get("autosave_last_at")
    if not force and last is not None and now - last < AUTOSAVE_INTERVAL_S:
        return
    words = collect_lesson_words()
    if not words and not force:
        return
    turns = group_turns(words)
    header = (
        f"Lesson: {os.path.basename(lesson_dir)}\n"
        f"Saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    try:
        os.makedirs(lesson_dir, exist_ok=True)
        with open(lesson_transcript_path(lesson_dir), "w", encoding="utf-8") as f:
            f.write(header + build_transcript_text(turns) + "\n")
        with open(os.path.join(lesson_dir, "words.json"), "w", encoding="utf-8") as f:
            json.dump({"version": 2, "words": words}, f, ensure_ascii=False)
        feedback = st.session_state.get("live_feedback") or {}
        if feedback:
            with open(os.path.join(lesson_dir, "live_feedback.json"), "w", encoding="utf-8") as f:
                json.dump({"version": 1, "feedback": feedback}, f, ensure_ascii=False)
        st.session_state.autosave_last_at = now
        st.session_state.autosave_last_clock = datetime.now().strftime("%H:%M:%S")
    except Exception as e:
        log_event("autosave_error", error=str(e), lesson_dir=lesson_dir)


def save_lesson_audio(audio_payload):
    """Write the whole-lesson compressed audio delivered by the recorder."""
    lesson_dir = st.session_state.get("lesson_dir")
    if not lesson_dir or not isinstance(audio_payload, dict):
        return []
    saved = []
    os.makedirs(lesson_dir, exist_ok=True)
    for source, entry in audio_payload.items():
        if source not in SOURCE_LABELS or not isinstance(entry, dict):
            continue
        b64 = entry.get("b64")
        if not b64:
            continue
        mime = (entry.get("mime") or "audio/webm").split(";")[0].strip().lower()
        ext = MIME_EXTENSIONS.get(mime, ".webm")
        path = os.path.join(lesson_dir, f"{source}{ext}")
        try:
            with open(path, "wb") as f:
                f.write(base64.b64decode(b64))
            saved.append(path)
            log_event("lesson_audio_saved", path=path, bytes=os.path.getsize(path))
        except Exception as e:
            log_event("lesson_audio_save_error", source=source, error=str(e))
    return saved


def new_lesson_dir():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lesson_dir = os.path.join(AUDIO_DIR, f"lesson_{stamp}")
    suffix = 1
    while os.path.exists(lesson_dir):
        lesson_dir = os.path.join(AUDIO_DIR, f"lesson_{stamp}_{suffix}")
        suffix += 1
    os.makedirs(lesson_dir, exist_ok=True)
    return lesson_dir


def list_lesson_audio():
    """Student audio from lesson folders (newest first), then legacy loose files.

    Error analysis runs on the student file only, so teacher files stay on
    disk but are not offered for upload.
    """
    if not os.path.isdir(AUDIO_DIR):
        return []
    lesson_files = []
    legacy_files = []
    for name in os.listdir(AUDIO_DIR):
        path = os.path.join(AUDIO_DIR, name)
        if os.path.isdir(path) and name.startswith("lesson_"):
            for inner in sorted(os.listdir(path)):
                inner_path = os.path.join(path, inner)
                if (
                    os.path.isfile(inner_path)
                    and inner.startswith("student")
                    and os.path.splitext(inner)[1].lower() in LESSON_AUDIO_EXTS
                ):
                    lesson_files.append(inner_path)
        elif os.path.isfile(path) and os.path.splitext(name)[1].lower() in LEGACY_AUDIO_EXTS:
            legacy_files.append(path)
    lesson_files.sort(key=os.path.getmtime, reverse=True)
    legacy_files.sort(key=os.path.getmtime, reverse=True)
    return lesson_files + legacy_files


# ---------------------------------------------------------------------------
# Replicate (post-lesson accurate STUDENT transcript for error analysis)
#
# Only the student file is sent. It is a single speaker, so no diarization —
# plain Whisper large-v3, the most accurate open transcription model on
# Replicate. Per-segment probabilities are kept and shown so uncertain
# transcription is never mistaken for a student error.
# ---------------------------------------------------------------------------
REPLICATE_WHISPER_MODEL = (
    "openai/whisper:8099696689d249cf8b122d833c36ac3f75505c666a395ca40ef26f68e7d3d16e"
)


def _extract_replicate_segments(output):
    if isinstance(output, dict):
        segments = output.get("segments")
        if isinstance(segments, list):
            return segments
    elif isinstance(output, list):
        return output
    return []


def _segment_probability(seg):
    """Whisper reports avg_logprob per segment; exp() maps it to ~probability."""
    for key in ("confidence", "probability"):
        value = seg.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    avg_logprob = seg.get("avg_logprob")
    if isinstance(avg_logprob, (int, float)):
        return math.exp(avg_logprob)
    return None


def build_student_transcript(output):
    """Timestamped single-speaker transcript, confidence on every segment.

    This is one ASR viewpoint for downstream LLM synthesis, so probabilities
    are included everywhere — no thresholding or flagging here.
    """
    lines = []
    probabilities = []
    for seg in _extract_replicate_segments(output):
        if not isinstance(seg, dict):
            continue
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        stamp = _format_clock(seg.get("start", 0.0))
        prob = _segment_probability(seg)
        if prob is None:
            lines.append(f"[{stamp}] {text}")
            continue
        probabilities.append(prob)
        lines.append(f"[{stamp}] (confidence {prob:.2f}) {text}")
    if not lines:
        return ""
    header = [
        "Student transcript — Replicate openai/whisper large-v3 (single speaker)",
        "Confidence values are ASR certainty (exp of segment avg_logprob), not language correctness.",
    ]
    if probabilities:
        header.append(
            f"Segments: {len(probabilities)} — average confidence {sum(probabilities) / len(probabilities):.2f}"
        )
    return "\n".join(header) + "\n\n" + "\n\n".join(lines)


def run_replicate_transcription(audio_path):
    replicate_token = os.environ.get("REPLICATE_API_TOKEN")
    if not replicate_token:
        raise RuntimeError("REPLICATE_API_TOKEN not set. Add it to the environment and retry.")
    log_event("replicate_call_attempt", model_id=REPLICATE_WHISPER_MODEL, audio_path=audio_path)
    # Whisper large-v3 on a whole lesson can run for many minutes; the SDK's
    # default read timeout is far shorter, so pin it to REPLICATE_TIMEOUT_S.
    # A short connect timeout still fails fast on a dead network.
    client = replicate.Client(
        api_token=replicate_token,
        timeout=httpx.Timeout(REPLICATE_TIMEOUT_S, connect=10.0),
    )
    try:
        with open(audio_path, "rb") as f:
            output = client.run(
                REPLICATE_WHISPER_MODEL,
                input={
                    # This pinned version runs Whisper large-v3 (its only model).
                    "audio": f,
                    "language": "en",
                    "transcription": "plain text",
                    # Reduces Whisper's repetition/hallucination loops on
                    # silence-heavy single-mic audio — fewer false errors.
                    "condition_on_previous_text": False,
                },
            )
    except Exception as e:
        message = str(e)
        log_event("replicate_call_error", error=message, error_type=type(e).__name__)
        if "Invalid token" in message or "status: 401" in message:
            raise RuntimeError(
                "Replicate authentication failed. Set a valid REPLICATE_API_TOKEN."
            ) from e
        raise RuntimeError(f"Replicate transcription failed: {message}") from e
    transcript = build_student_transcript(output)
    if not transcript:
        raise RuntimeError("Replicate returned no usable segments.")
    log_event("replicate_success", segments_count=len(_extract_replicate_segments(output)))
    return output, transcript


# ---------------------------------------------------------------------------
# LLM synthesis viewpoints
#
# The live Deepgram transcript and the post-lesson Whisper transcript are
# independent ASR viewpoints on the same lesson. They are exported with full
# confidence data (word-level for Deepgram, segment-level for Whisper) so an
# external LLM can compare them and reconstruct what the student really said.
# ---------------------------------------------------------------------------
def load_lesson_words(lesson_dir):
    try:
        with open(os.path.join(lesson_dir, "words.json"), "r", encoding="utf-8") as f:
            payload = json.load(f)
        words = payload.get("words", []) if isinstance(payload, dict) else payload
        return [w for w in words if isinstance(w, dict)]
    except Exception:
        return []


def build_deepgram_confidence_transcript(words):
    """Deepgram viewpoint as text: turns with per-word confidence inline."""
    turns = group_turns(words)
    if not turns:
        return ""
    lines = [
        "Live lesson transcript — Deepgram nova-3 streaming (word-level confidence)",
        "Speakers come from separate microphones. Confidence values are ASR certainty, not language correctness.",
        "Format: word(confidence)",
    ]
    body = []
    for turn in turns:
        label = SOURCE_LABELS.get(turn["source"], str(turn["source"]).title())
        rendered = []
        for word in turn["words"]:
            conf = word.get("confidence")
            if isinstance(conf, (int, float)):
                rendered.append(f"{word['word']}({conf:.2f})")
            else:
                rendered.append(str(word["word"]))
        body.append(f"[{_format_clock(turn['start'])}] {label}: {' '.join(rendered)}")
    return "\n".join(lines) + "\n\n" + "\n\n".join(body)


def build_synthesis_bundle(lesson_dir, audio_path):
    """One JSON with every available ASR viewpoint, ready to hand to an LLM."""
    viewpoints = []
    words = load_lesson_words(lesson_dir) if lesson_dir else []
    if words:
        viewpoints.append({
            "id": "deepgram_live",
            "service": "Deepgram nova-3 (live streaming)",
            "granularity": "word",
            "speakers": "student and teacher, separated by microphone",
            "turns": [
                {
                    "speaker": SOURCE_LABELS.get(turn["source"], str(turn["source"])),
                    "start": turn["start"],
                    "end": turn["end"],
                    "text": turn["text"],
                    "words": [
                        {
                            "word": w["word"],
                            "start": w.get("start"),
                            "end": w.get("end"),
                            "confidence": w.get("confidence"),
                        }
                        for w in turn["words"]
                    ],
                }
                for turn in group_turns(words)
            ],
        })
    replicate_output = None
    if audio_path and os.path.exists(replicate_json_path(audio_path)):
        try:
            with open(replicate_json_path(audio_path), "r", encoding="utf-8") as f:
                replicate_output = json.load(f)
        except Exception:
            replicate_output = None
    if replicate_output is not None:
        segments = []
        for seg in _extract_replicate_segments(replicate_output):
            if not isinstance(seg, dict):
                continue
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            segments.append({
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": text,
                "probability": _segment_probability(seg),
                "avg_logprob": seg.get("avg_logprob"),
                "no_speech_prob": seg.get("no_speech_prob"),
                "compression_ratio": seg.get("compression_ratio"),
            })
        viewpoints.append({
            "id": "whisper_large_v3",
            "service": "Replicate openai/whisper large-v3 (post-lesson)",
            "granularity": "segment",
            "speakers": "student only (student microphone file)",
            "segments": segments,
        })
    if not viewpoints:
        return None
    return {
        "lesson": os.path.basename(lesson_dir) if lesson_dir else None,
        "generated": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Independent ASR viewpoints on the same ESL lesson. Compare them to "
            "reconstruct what the student actually said before doing error analysis. "
            "Confidence/probability values are ASR certainty, not language correctness."
        ),
        "viewpoints": viewpoints,
    }


def build_bundle_text(bundle, deepgram_text, whisper_text):
    """Plain-text rendering of the combined bundle: a header plus each ASR
    viewpoint in its human-readable, confidence-annotated form. The pieces are
    the same renderers shown/downloaded individually, joined into one file."""
    header = [
        f"Combined ASR bundle — lesson {bundle.get('lesson') or '(unknown)'}",
        f"Generated: {bundle.get('generated', '')}",
        (bundle.get("purpose") or "").strip(),
    ]
    sections = [line for line in header if line]
    if deepgram_text:
        sections.append("=" * 70 + "\n" + deepgram_text)
    if whisper_text:
        sections.append("=" * 70 + "\n" + whisper_text)
    return "\n\n".join(sections) + "\n"


def replicate_transcript_path(audio_path):
    return f"{os.path.splitext(audio_path)[0]}.replicate.txt"


def replicate_json_path(audio_path):
    return f"{os.path.splitext(audio_path)[0]}.replicate.json"


def run_replicate_job(audio_path):
    """Background job body; returns a plain dict for the polling loop."""
    try:
        output, transcript = run_replicate_transcription(audio_path)
        try:
            with open(replicate_json_path(audio_path), "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2, default=str)
            with open(replicate_transcript_path(audio_path), "w", encoding="utf-8") as f:
                f.write(transcript + "\n")
        except Exception as e:
            log_event("replicate_save_error", error=str(e))
        return {"ok": True, "audio_path": audio_path, "transcript": transcript}
    except Exception as e:
        return {"ok": False, "audio_path": audio_path, "error": str(e)}


# ---------------------------------------------------------------------------
# DeepSeek synthesis (reconcile the ASR viewpoints into one best transcript)
#
# The synthesis bundle (Deepgram word-level for both mics, Whisper
# segment-level for the student) is handed to DeepSeek, which weighs the
# viewpoints and reconstructs the single most likely verbatim transcript.
# Student turns are reconciled across both viewpoints; teacher turns exist
# only in Deepgram. Genuine student language errors are preserved on purpose —
# this transcript feeds error analysis, so the model resolves ASR uncertainty
# WITHOUT correcting the student's English.
# ---------------------------------------------------------------------------
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-reasoner")
# Output ceiling. deepseek-reasoner spends much of its budget on the hidden
# reasoning trace BEFORE the answer, so a full-lesson transcript needs the
# model's full 64000-token ceiling or it truncates (or returns nothing at all
# when reasoning alone exhausts a smaller budget). deepseek-chat has no reasoning
# trace but caps output at 8192 tokens and rejects anything larger, so its
# default stays under that — note chat cannot fit a long (~6000+ word) lesson in
# one shot. The default tracks the model; override with DEEPSEEK_MAX_TOKENS.
# Only generated tokens are billed.
_DEEPSEEK_DEFAULT_MAX_TOKENS = "64000" if "reasoner" in DEEPSEEK_MODEL else "8000"
DEEPSEEK_MAX_TOKENS = int(os.environ.get("DEEPSEEK_MAX_TOKENS", _DEEPSEEK_DEFAULT_MAX_TOKENS))

DEEPSEEK_SYSTEM_PROMPT = (
    "You reconstruct the single most accurate verbatim transcript of a recorded "
    "one-to-one ESL lesson between a Teacher and a Student.\n\n"
    "You are given independent automatic-speech-recognition (ASR) viewpoints of "
    "the same lesson as JSON:\n"
    "- 'deepgram_live': word-level, with a confidence per word, covering BOTH "
    "speakers (each on a separate microphone, so the speaker label is reliable).\n"
    "- 'whisper_large_v3' (may be absent): segment-level, with a probability per "
    "segment, covering the STUDENT only.\n\n"
    "Your job: merge these into one chronological transcript of what was really "
    "said. Where the viewpoints disagree on the student's words, choose the "
    "reading better supported by the evidence (higher confidence/probability, or "
    "agreement between the two viewpoints). Teacher words come only from "
    "'deepgram_live'; transcribe them faithfully.\n\n"
    "CRITICAL — this transcript is used to analyse the student's language errors, "
    "so you must NOT correct the student's English. Preserve their real mistakes "
    "exactly: wrong tense, missing/extra articles, wrong word choice, "
    "subject-verb disagreement, non-native word order. Only resolve ASR "
    "uncertainty (a word one engine misheard), never a learner mistake. When you "
    "cannot tell whether something is a mishearing or a genuine learner error, "
    "keep the student's apparent words. Do not paraphrase, translate, or add or "
    "invent content that neither viewpoint supports.\n\n"
    "OUTPUT FORMAT — output ONLY the transcript, nothing else. One turn per line, "
    "turns in time order, a blank line between turns, each line exactly:\n"
    "[mm:ss] Speaker: text\n"
    "where Speaker is exactly 'Teacher' or 'Student' and mm:ss is the turn's "
    "start time. No headers, no commentary, no confidence numbers, no markdown."
)


def deepseek_api_key():
    # Accept the conventional name, and the bare name in case it was set that way.
    return os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API")


def run_deepseek_synthesis(bundle):
    api_key = deepseek_api_key()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set. Add it to the environment and retry.")
    try:
        from openai import OpenAI
    except Exception as e:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "The 'openai' package is required for DeepSeek synthesis. Install it in the env."
        ) from e
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    user_content = (
        "Independent ASR viewpoints of one ESL lesson, as JSON. Reconcile them "
        "into the single most likely verbatim transcript, following your "
        "instructions exactly:\n\n" + json.dumps(bundle, ensure_ascii=False)
    )
    log_event(
        "deepseek_call_attempt",
        model=DEEPSEEK_MODEL,
        viewpoints=[v.get("id") for v in bundle.get("viewpoints", [])],
    )
    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": DEEPSEEK_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=DEEPSEEK_MAX_TOKENS,
            stream=False,
        )
    except Exception as e:
        message = str(e)
        log_event("deepseek_call_error", error=message, error_type=type(e).__name__)
        lowered = message.lower()
        if "401" in message or "authentication" in lowered or "invalid api key" in lowered:
            raise RuntimeError(
                "DeepSeek authentication failed. Set a valid DEEPSEEK_API_KEY."
            ) from e
        raise RuntimeError(f"DeepSeek synthesis failed: {message}") from e
    choice = response.choices[0]
    transcript = (choice.message.content or "").strip()
    if choice.finish_reason == "length" and not transcript:
        # The reasoning trace consumed the whole budget before any answer.
        raise RuntimeError(
            "DeepSeek hit its token limit while reasoning and returned no transcript. "
            "Raise DEEPSEEK_MAX_TOKENS and retry."
        )
    if not transcript:
        raise RuntimeError("DeepSeek returned an empty transcript.")
    if choice.finish_reason == "length":
        # Some answer came back but was cut off mid-transcript.
        raise RuntimeError(
            "DeepSeek transcript was truncated (hit the token limit). "
            "Raise DEEPSEEK_MAX_TOKENS and retry."
        )
    log_event("deepseek_success", chars=len(transcript))
    return transcript


def synthesis_transcript_path(audio_path):
    return f"{os.path.splitext(audio_path)[0]}.synthesis.txt"


def run_deepseek_job(lesson_dir, audio_path):
    """Background job body; returns a plain dict for the polling loop."""
    try:
        bundle = build_synthesis_bundle(lesson_dir, audio_path)
        if not bundle:
            return {
                "ok": False,
                "audio_path": audio_path,
                "error": "No ASR viewpoints available to synthesize yet.",
            }
        transcript = run_deepseek_synthesis(bundle)
        try:
            with open(synthesis_transcript_path(audio_path), "w", encoding="utf-8") as f:
                f.write(transcript + "\n")
        except Exception as e:
            log_event("deepseek_save_error", error=str(e))
        return {"ok": True, "audio_path": audio_path, "transcript": transcript}
    except Exception as e:
        return {"ok": False, "audio_path": audio_path, "error": str(e)}


# ---------------------------------------------------------------------------
# Live per-sentence feedback
#
# Opt-in lane, off by default. While the switch is on, each settled sentence of
# student speech goes to DeepSeek on its own and the result is cached by sentence
# id, so the teacher sees a marked sentence they can hover mid-lesson.
#
# Two facts from the recorded lessons drive the design. Deepgram punctuates
# aggressively, so 46% of "sentences" are a bare "Yeah." — hence LIVE_MIN_WORDS.
# And 17% of student words score under 0.70 confidence, so a naive prompt would
# spend most of its output reporting mishearings as learner errors — hence the
# uncertain-word list in the payload and the instructions built around it.
# ---------------------------------------------------------------------------
DEEPSEEK_LIVE_MODEL = os.environ.get("DEEPSEEK_LIVE_MODEL", "deepseek-chat")
LIVE_FEEDBACK_PROMPT_PATH = os.path.join("outputs", "live_feedback_prompt.txt")

DEFAULT_LIVE_FEEDBACK_PROMPT = (
    "You are an ESL teaching assistant watching a live one-to-one lesson. You are "
    "given ONE sentence a student just said, and you judge only that sentence.\n\n"
    "The text is automatic speech recognition output of SPOKEN English, not "
    "writing. That means:\n"
    "- Ignore punctuation and capitalisation entirely. They come from the "
    "recognizer, not the student.\n"
    "- Ignore repeated words, false starts, self-corrections and fillers "
    "('the the', 'how to how to', 'I be I begin'). That is normal speech "
    "disfluency or a transcription artefact, never a language error.\n"
    "- The 'uncertain_words' list holds words the recognizer itself was unsure "
    "of. Never build an error report on those words — if the only problem "
    "involves one of them, the sentence is fine.\n"
    "- A stray word from the other speaker sometimes bleeds into the sentence. "
    "Ignore anything that clearly is not part of the student's own utterance.\n\n"
    "Report ONLY a genuine learner language error: verb tense or form, "
    "subject-verb agreement, articles, prepositions, plurals, word order, or a "
    "wrong word choice. Report the single most important one, not a list. If the "
    "sentence is acceptable spoken English, or the only problems are disfluency "
    "or mis-transcription, say so and stop.\n\n"
    "Reply with json only, in exactly this shape:\n"
    '{"verdict": "ok" | "issue", "label": "...", "explanation": "...", '
    '"rewrites": ["...", "..."]}\n\n'
    "- 'verdict': 'issue' only when there is a real learner error to show.\n"
    "- 'label': the error type in two to four words, lowercase, e.g. "
    "'past tense', 'missing article'.\n"
    "- 'explanation': ONE short sentence, at most 15 words, saying plainly what "
    "is wrong. Write it for the teacher to read at a glance mid-lesson.\n"
    "- 'rewrites': one or two natural ways to say it, keeping the student's "
    "meaning and their spoken register. Never more than two.\n"
    "- When 'verdict' is 'ok', use an empty label, an empty explanation and an "
    "empty rewrites list."
)


def load_persisted_live_feedback_prompt():
    try:
        if os.path.exists(LIVE_FEEDBACK_PROMPT_PATH):
            with open(LIVE_FEEDBACK_PROMPT_PATH, "r", encoding="utf-8") as f:
                saved = f.read().strip()
            if saved:
                return saved
    except Exception as e:
        log_event("live_feedback_prompt_load_error", error=str(e), path=LIVE_FEEDBACK_PROMPT_PATH)
    return DEFAULT_LIVE_FEEDBACK_PROMPT


def persist_live_feedback_prompt():
    prompt = (st.session_state.get("live_feedback_prompt") or "").strip()
    if not prompt:
        prompt = DEFAULT_LIVE_FEEDBACK_PROMPT
        st.session_state.live_feedback_prompt = prompt
    try:
        # Write only on a real change. The callback is cheap to trigger and a
        # long-lived session should never end up rewriting this file in a loop.
        if os.path.exists(LIVE_FEEDBACK_PROMPT_PATH):
            with open(LIVE_FEEDBACK_PROMPT_PATH, "r", encoding="utf-8") as f:
                if f.read().strip() == prompt:
                    return
        os.makedirs(os.path.dirname(LIVE_FEEDBACK_PROMPT_PATH), exist_ok=True)
        with open(LIVE_FEEDBACK_PROMPT_PATH, "w", encoding="utf-8") as f:
            f.write(prompt)
        log_event("live_feedback_prompt_saved", chars=len(prompt))
    except Exception as e:
        log_event("live_feedback_prompt_save_error", error=str(e))


def parse_live_feedback_reply(raw):
    """Tolerant parse of the model's JSON, normalised to what the renderer needs."""
    text = (raw or "").strip()
    if text.startswith("```"):
        # Strip a fenced block; json_object mode should prevent this, but a hand
        # edited prompt can easily talk the model back into markdown.
        text = text.split("```")[1] if text.count("```") >= 2 else text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    verdict = str(data.get("verdict") or "").strip().lower()
    if verdict not in ("ok", "issue"):
        # Treat anything unrecognised as clean rather than showing a broken card.
        verdict = "issue" if data.get("rewrites") else "ok"
    rewrites = data.get("rewrites") or []
    if isinstance(rewrites, str):
        rewrites = [rewrites]
    rewrites = [str(r).strip() for r in rewrites if str(r).strip()][:2]
    explanation = str(data.get("explanation") or "").strip()
    if verdict == "issue" and not (rewrites or explanation):
        verdict = "ok"
    return {
        "verdict": verdict,
        "label": str(data.get("label") or "").strip(),
        "explanation": explanation,
        "rewrites": rewrites,
    }


def build_live_feedback_payload(sentence, turns):
    """What one sentence looks like to the model, plus the turns leading up to it.

    Context matters: tense and agreement are often only judgeable against the
    question that prompted the sentence.
    """
    context = []
    for turn in turns:
        if turn["start"] >= sentence["start"]:
            break
        context.append({
            "speaker": SOURCE_LABELS.get(turn["source"], turn["source"]),
            "text": turn["text"],
        })
    return {
        "speaker": SOURCE_LABELS.get(sentence["source"], sentence["source"]),
        "sentence": sentence["text"],
        "context": context[-LIVE_FEEDBACK_CONTEXT_TURNS:],
        "uncertain_words": low_confidence_words(sentence, LIVE_LOW_CONFIDENCE),
    }


def live_feedback_client(api_key):
    return live_runtime.openai_client(api_key, DEEPSEEK_BASE_URL, LIVE_FEEDBACK_TIMEOUT_S)


def run_live_feedback_job(sentence_id, system_prompt, model, payload):
    """Background job body. Pure: returns a plain dict, never touches st.*."""
    api_key = deepseek_api_key()
    if not api_key:
        return {"id": sentence_id, "ok": False, "error": "DEEPSEEK_API_KEY not set."}
    try:
        client = live_feedback_client(api_key)
    except Exception as e:  # pragma: no cover - dependency guard
        return {"id": sentence_id, "ok": False, "error": f"openai package missing: {e}"}
    log_event("live_feedback_call_attempt", sentence_id=sentence_id, model=model,
              words=len(payload.get("sentence", "").split()))
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=LIVE_FEEDBACK_MAX_TOKENS,
            stream=False,
        )
        result = parse_live_feedback_reply(response.choices[0].message.content)
    except Exception as e:
        message = str(e)
        log_event("live_feedback_call_error", sentence_id=sentence_id,
                  error=message, error_type=type(e).__name__)
        lowered = message.lower()
        if "401" in message or "authentication" in lowered or "invalid api key" in lowered:
            message = "DeepSeek authentication failed. Check DEEPSEEK_API_KEY."
        return {"id": sentence_id, "ok": False, "error": message}
    if result["verdict"] == "issue":
        log_event("live_feedback_issue", sentence_id=sentence_id, label=result["label"])
    result.update({"id": sentence_id, "ok": True})
    return result


def _record_live_feedback_failure(sentence_id, error):
    """Park a failed sentence for another go.

    DeepSeek occasionally stalls for a minute and then recovers. Burning that
    into a permanent state would leave the transcript speckled with sentences
    that were never really judged, so a failure stays "retry" - and renders
    grey, as still being checked - until it has used up its attempts.
    """
    previous = st.session_state.live_feedback.get(sentence_id) or {}
    attempts = int(previous.get("attempts") or 0) + 1
    terminal = attempts >= LIVE_FEEDBACK_MAX_ATTEMPTS
    st.session_state.live_feedback[sentence_id] = {
        "status": "error" if terminal else "retry",
        "error": error or "Live feedback failed.",
        "attempts": attempts,
    }
    if terminal:
        st.session_state.live_feedback_errors += 1
    log_event("live_feedback_retry" if not terminal else "live_feedback_gave_up",
              sentence_id=sentence_id, attempts=attempts)


def drain_live_feedback():
    """Move finished jobs into the cache. Main thread only."""
    futures = st.session_state.live_feedback_futures
    for sentence_id in [k for k, f in futures.items() if f.done()]:
        future = futures.pop(sentence_id)
        try:
            result = future.result()
        except Exception as e:
            _record_live_feedback_failure(sentence_id, str(e))
            continue
        if result.get("ok"):
            st.session_state.live_feedback[sentence_id] = {
                "status": "ok",
                "verdict": result["verdict"],
                "label": result["label"],
                "explanation": result["explanation"],
                "rewrites": result["rewrites"],
            }
        else:
            _record_live_feedback_failure(sentence_id, result.get("error"))


def live_feedback_is_on():
    """The switch, read from a mirror rather than from the widget key.

    The toggle is rendered in the main script body, but the window edges are
    detected inside a 0.5s fragment. A fragment run does not re-instantiate
    widgets outside it, and the widget key is not reliably readable there - a
    single false read closes the window permanently, which looks exactly like
    the feature having died. The mirror is written wherever the widget is
    actually drawn, and the switch can only change through a full rerun, so it
    is always current.
    """
    return bool(st.session_state.get("live_feedback_switch"))


def mirror_live_feedback_switch():
    """Call in the main script body, immediately after drawing the toggle."""
    st.session_state.live_feedback_switch = bool(st.session_state.get("live_feedback_on"))


def sync_live_feedback_window(now_audio):
    """Open or close the armed window as the switch flips.

    The switch selects a stretch of SPEECH, not a moment of dispatch. Windows
    are recorded on the lesson's audio clock and kept for the whole lesson, so
    what gets analysed depends on when a sentence was spoken - never on when the
    switch happened to be read.
    """
    on = live_feedback_is_on()
    was_on = bool(st.session_state.get("live_feedback_was_on"))
    if on is not was_on:
        update_feedback_windows(st.session_state.live_feedback_windows, on, was_on, now_audio)
        log_event("live_feedback_armed" if on else "live_feedback_disarmed",
                  at=round(now_audio, 2),
                  windows=len(st.session_state.live_feedback_windows))
    st.session_state.live_feedback_was_on = on


def dispatch_live_feedback(sentences, turns):
    """Submit every un-analysed sentence spoken inside an armed window.

    Deliberately NOT gated on the switch's current position. A sentence spoken
    just before the switch went off only settles a few seconds later; it still
    belongs to the window and must still be sent. Conversely nothing outside a
    window is ever sent, however long the switch is on afterwards.
    """
    if not deepseek_api_key():
        return
    windows = st.session_state.live_feedback_windows
    if not windows:
        return
    cache = st.session_state.live_feedback
    futures = st.session_state.live_feedback_futures
    prompt = st.session_state.get("live_feedback_prompt") or DEFAULT_LIVE_FEEDBACK_PROMPT
    armed = [s for s in sentences if in_feedback_window(s, windows)]
    for sentence in sorted(armed, key=lambda s: s["start"], reverse=True):
        if len(futures) >= LIVE_FEEDBACK_MAX_INFLIGHT:
            return
        if st.session_state.live_feedback_calls >= LIVE_FEEDBACK_MAX_CALLS_PER_LESSON:
            return
        sentence_id = sentence["id"]
        if sentence_id in futures:
            continue
        cached = cache.get(sentence_id)
        if cached is not None and cached.get("status") != "retry":
            continue
        if sentence["word_count"] < LIVE_MIN_WORDS:
            cache[sentence_id] = {"status": "skipped"}
            continue
        payload = build_live_feedback_payload(sentence, turns)
        futures[sentence_id] = LIVE_FEEDBACK_EXECUTOR.submit(
            run_live_feedback_job, sentence_id, prompt, DEEPSEEK_LIVE_MODEL, payload
        )
        st.session_state.live_feedback_calls += 1


def live_feedback_engaged():
    """True once the switch has been armed at all this lesson.

    Rendering and dispatch both key off this, NOT off the switch's current
    position: results already paid for stay on screen, and sentences from a
    closed window still finish processing.
    """
    return bool(st.session_state.live_feedback_windows)


def live_feedback_counts():
    """(checked, flagged, analysing, skipped) for the counter beside the switch.

    `skipped` is reported because it is the difference between what was heard
    and what got underlined; without it, a transcript full of unmarked "Yeah."
    reads as the feature being broken.
    """
    checked = flagged = skipped = retrying = 0
    for entry in st.session_state.live_feedback.values():
        status = entry.get("status")
        if status == "skipped":
            skipped += 1
            continue
        if status == "retry":
            retrying += 1
            continue
        if status != "ok":
            continue
        checked += 1
        if entry.get("verdict") == "issue":
            flagged += 1
    return (checked, flagged,
            len(st.session_state.live_feedback_futures) + retrying, skipped)


def reset_live_feedback():
    st.session_state.live_feedback = {}
    st.session_state.live_feedback_futures = {}
    st.session_state.live_feedback_windows = []
    # Cleared, not set from the switch: if the switch is still on, the next sync
    # opens a fresh window at the new lesson's zero.
    st.session_state.live_feedback_was_on = False
    st.session_state.live_feedback_calls = 0
    st.session_state.live_feedback_errors = 0


# ---------------------------------------------------------------------------
# Session state and recorder events
# ---------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "lesson_state": "idle",  # idle | recording | paused | ended
        "transcript_filter": "Both",  # on-screen display filter; never affects capture
        "lesson_dir": None,
        "lesson_started_at": None,
        "lesson_active_seconds": 0,
        "lesson_words": [],
        "source_offsets": {},
        "deepgram_streamers": {},
        "deepgram_last_restart_at": None,
        "autosave_last_at": None,
        "autosave_last_clock": None,
        "last_recorder_event_id": "",
        "replicate_future": None,
        "replicate_audio_path": None,
        "replicate_error": "",
        "replicate_transcript": "",
        "synthesis_future": None,
        "synthesis_error": "",
        "synthesis_transcript": "",
        "live_feedback_on": False,      # the toggle widget's own key
        "live_feedback_switch": False,  # mirror of it, safe to read from a fragment
        "live_feedback_windows": [],    # [[audio_start, audio_end_or_None], ...]
        "live_feedback_was_on": False,  # edge detector for the switch
        "live_feedback": {},            # sentence id -> cached result
        "live_feedback_futures": {},    # sentence id -> in-flight Future
        "live_feedback_calls": 0,
        "live_feedback_errors": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "live_feedback_prompt" not in st.session_state:
        st.session_state.live_feedback_prompt = load_persisted_live_feedback_prompt()


def parse_recorder_value(value):
    """Return the recorder event payload once per unique event, else None."""
    payload = None
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            payload = None
    elif isinstance(value, dict):
        payload = value
    if not isinstance(payload, dict) or not payload.get("event"):
        return None
    event_id = f"{payload.get('event')}:{payload.get('t', 0)}:{payload.get('seq', '')}:{payload.get('source', '')}"
    if event_id == st.session_state.get("last_recorder_event_id"):
        return None
    st.session_state.last_recorder_event_id = event_id
    return payload


def handle_recorder_event(payload):
    event = payload.get("event")

    if event == "lesson_started":
        st.session_state.lesson_state = "recording"
        st.session_state.lesson_dir = new_lesson_dir()
        st.session_state.lesson_started_at = datetime.now(timezone.utc)
        st.session_state.lesson_active_seconds = 0
        st.session_state.lesson_words = []
        st.session_state.source_offsets = {}
        st.session_state.autosave_last_at = None
        st.session_state.autosave_last_clock = None
        st.session_state.replicate_transcript = ""
        st.session_state.replicate_error = ""
        st.session_state.synthesis_transcript = ""
        st.session_state.synthesis_error = ""
        reset_live_feedback()
        source_rates = payload.get("sources") or {}
        for source, rate in source_rates.items():
            if source in SOURCE_LABELS:
                start_deepgram_stream(source, rate)
        log_event("lesson_started", lesson_dir=st.session_state.lesson_dir, sources=list(source_rates))

    elif event == "recording_chunk":
        if st.session_state.lesson_state == "recording":
            handle_audio_chunk(payload)

    elif event == "lesson_paused":
        st.session_state.lesson_state = "paused"
        st.session_state.lesson_active_seconds = payload.get("active_seconds") or 0
        autosave_lesson(force=True)
        log_event("lesson_paused", active_seconds=st.session_state.lesson_active_seconds)

    elif event == "lesson_resumed":
        st.session_state.lesson_state = "recording"
        log_event("lesson_resumed")

    elif event == "lesson_ended":
        st.session_state.lesson_active_seconds = payload.get("active_seconds") or 0
        saved_audio = save_lesson_audio(payload.get("audio") or {})
        end_all_streams()
        autosave_lesson(force=True)
        st.session_state.lesson_state = "ended"
        if saved_audio:
            # Default the post-lesson step to this lesson's student audio.
            student_audio = [p for p in saved_audio if os.path.basename(p).startswith("student")]
            st.session_state.replicate_audio_path = (student_audio or saved_audio)[0]
        log_event(
            "lesson_ended",
            lesson_dir=st.session_state.lesson_dir,
            active_seconds=st.session_state.lesson_active_seconds,
            audio_files=saved_audio,
        )

    elif event in ("mic_error", "lesson_start_error", "media_recorder_error", "lesson_audio_encode_error"):
        log_event("recorder_error", event_name=event, message=payload.get("message"))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def render_styles():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; }
        .lesson-status {
            display: flex; gap: 14px; align-items: center; flex-wrap: wrap;
            font-size: 0.9rem; color: #596273; margin-bottom: 4px;
        }
        .lesson-status .state {
            font-weight: 700; padding: 2px 10px; border-radius: 999px; font-size: 0.8rem;
        }
        .lesson-status .state.recording { background: #d8f3e7; color: #0f5743; }
        .lesson-status .state.paused { background: #fdf0d5; color: #8a5a10; }
        .lesson-status .state.ended { background: #e4e8ef; color: #384152; }
        .lesson-status .state.idle { background: #e4e8ef; color: #384152; }
        .transcript-pane {
            border: 1px solid #e0e4ea; border-radius: 10px;
            padding: 14px 18px; background: #fbfcfd;
            max-height: 60vh; overflow-y: auto;
        }
        .turn { margin: 0 0 10px 0; line-height: 1.5; }
        .turn .ts {
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 0.78rem; color: #8a93a3; margin-right: 8px;
        }
        .turn .speaker { font-weight: 700; margin-right: 6px; }
        .turn.student .speaker { color: #0f5743; }
        .turn.teacher .speaker { color: #175cd3; }
        .turn.teacher .text { color: #4a5468; }
        .turn.interim .text { color: #8a93a3; font-style: italic; }

        /* The whole progress indicator: the underline colour is the state.
           Grey = being checked, green = checked and fine, red = error found.
           No underline at all means the sentence was never sent, so the point
           the analysis reached is still readable at a glance. */
        /* The underline goes on the inner text span, not on .sent itself.
           text-decoration propagates into descendants and a descendant cannot
           cancel it, so an underlined wrapper drew lines straight through the
           feedback card's own text. Keeping the card a sibling of the underline
           is the only way to be rid of it. */
        .sent .sent-text {
            text-decoration: underline;
            text-decoration-thickness: 2px;
            text-underline-offset: 3px;
            -webkit-text-decoration-thickness: 2px;
        }
        .sent.pending .sent-text { text-decoration-color: #b6bec9; }
        .sent.checked .sent-text { text-decoration-color: #2e9e6b; }
        .sent.flagged .sent-text { text-decoration-color: #d92d20; }
        .sent.failed  .sent-text { text-decoration-color: #d9a441; }
        .sent.flagged, .sent.failed { cursor: help; }
        .sent.flagged:hover .sent-text, .sent.failed:hover .sent-text {
            background: rgba(217, 45, 32, 0.10);
        }

        /* The card in the transcript is only a data carrier - never shown.
           A single popup element, parked on <body> and positioned by script,
           does the display. It has to live outside the pane: the pane scrolls,
           and anything inside a scroll container is clipped at its edge and
           cannot be scrolled to, because scrolling drops the hover. */
        .fb-card { display: none; }

        #fb-popup {
            display: none; position: fixed; z-index: 9999;
            max-width: 420px; padding: 11px 13px 12px 13px;
            background: #ffffff; border: 1px solid #dfe3ea; border-radius: 9px;
            box-shadow: 0 8px 26px rgba(20, 28, 45, 0.18);
            font-size: 0.84rem; line-height: 1.45; color: #384152;
            text-align: left; pointer-events: none;
        }
        #fb-popup.show { display: block; }
        /* These style the popup's contents. The markup is lifted out of the
           hidden .fb-card, so the rules must target the popup, not the card. */
        #fb-popup .fb-label.failed { background: #fdf0d5; color: #8a5a10; }
        #fb-popup .fb-label {
            display: inline-block; margin: 0 0 7px 0; padding: 1px 8px;
            border-radius: 999px; background: #fee4e2; color: #b42318;
            font-size: 0.68rem; font-weight: 700; letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        #fb-popup .fb-said {
            display: block; color: #8a93a3; font-style: italic;
            margin-bottom: 6px; padding-left: 8px; border-left: 2px solid #e6e9ef;
        }
        #fb-popup .fb-exp { display: block; color: #596273; margin-bottom: 8px; }
        #fb-popup .fb-fix {
            display: block; color: #0f5743; margin-top: 4px; padding-left: 15px;
            text-indent: -15px;
        }
        #fb-popup .fb-fix::before { content: "→ "; color: #98a1b0; }
        .fb-counts {
            font-size: 0.78rem; color: #8a93a3; margin-top: 2px;
        }
        .fb-counts .flagged { color: #b42318; font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _turn_html(source, start, text, interim=False, segments=None):
    """One turn. With `segments` the text is broken into hoverable sentences;
    without it the markup is exactly what it has always been."""
    label = SOURCE_LABELS.get(source, str(source).title())
    classes = f"turn {html.escape(str(source))}" + (" interim" if interim else "")
    stamp = "..." if interim else _format_clock(start)
    body = "".join(_segment_html(seg) for seg in segments) if segments else html.escape(text)
    return (
        f'<div class="{classes}">'
        f'<span class="ts">[{stamp}]</span>'
        f'<span class="speaker">{html.escape(label)}:</span>'
        f'<span class="text">{body}</span>'
        f"</div>"
    )


def _segment_html(segment):
    """One sentence inside a turn.

    The underline colour is the progress indicator: grey while it is being
    checked, green once it came back clean, red when an error was found. A
    sentence with no underline was never sent (too short, or the switch was off
    when it went past), so the point the analysis reached stays visible.
    """
    text = html.escape(segment["text"])
    state = segment.get("state")
    if state not in ("flagged", "pending", "checked", "failed"):
        return f"{text} "
    if state in ("pending", "checked"):
        return f'<span class="sent {state}"><span class="sent-text">{text}</span></span> '

    parts = []
    if state == "failed":
        parts.append('<span class="fb-label failed">not checked</span>')
        parts.append(f'<span class="fb-exp">{html.escape(segment.get("error") or "Analysis failed.")}</span>')
        card = f'<span class="fb-card">{"".join(parts)}</span>'
        return (f'<span class="sent failed">'
                f'<span class="sent-text">{text}</span>{card}</span> ')

    if segment.get("label"):
        parts.append(f'<span class="fb-label">{html.escape(segment["label"])}</span>')
    # Cross-talk from the other mic splits a turn but not the sentence, so the
    # underline is sometimes only a scrap ("why"). Show the whole sentence then,
    # or the card has no context.
    whole = segment.get("sentence")
    if whole and whole != segment["text"]:
        parts.append(f'<span class="fb-said">{html.escape(whole)}</span>')
    if segment.get("explanation"):
        parts.append(f'<span class="fb-exp">{html.escape(segment["explanation"])}</span>')
    for rewrite in segment.get("rewrites") or []:
        parts.append(f'<span class="fb-fix">{html.escape(rewrite)}</span>')
    card = f'<span class="fb-card">{"".join(parts)}</span>'
    return (f'<span class="sent flagged">'
            f'<span class="sent-text">{text}</span>{card}</span> ')


def turn_segments(turn, word_to_sentence, sentences_by_id, feedback, in_flight):
    """Split one turn into runs of words belonging to the same sentence.

    A sentence can span two turns of the same speaker - the 2s sentence gap is
    shorter than the 8s turn gap, and interleaved cross-talk from the other mic
    splits the turn but not the sentence. Both halves then carry the same card,
    which is correct.
    """
    runs = []
    for word in turn["words"]:
        token = (word.get("word") or "").strip()
        if not token:
            continue
        sentence_id = word_to_sentence.get(id(word))
        if runs and runs[-1]["id"] == sentence_id:
            runs[-1]["tokens"].append(token)
        else:
            runs.append({"id": sentence_id, "tokens": [token]})
    segments = []
    for run in runs:
        segment = {"id": run["id"], "text": " ".join(run["tokens"]), "state": None}
        entry = feedback.get(run["id"]) if run["id"] else None
        if run["id"] and run["id"] in in_flight:
            segment["state"] = "pending"
        elif entry and entry.get("status") == "retry":
            segment["state"] = "pending"          # still being checked
        elif entry and entry.get("status") == "error":
            segment.update({"state": "failed", "error": entry.get("error")})
        elif entry and entry.get("status") == "ok":
            if entry.get("verdict") == "issue":
                sentence = sentences_by_id.get(run["id"]) or {}
                segment.update({
                    "state": "flagged",
                    "label": entry.get("label"),
                    "explanation": entry.get("explanation"),
                    "rewrites": entry.get("rewrites"),
                    "sentence": sentence.get("text"),
                })
            else:
                segment["state"] = "checked"
        # Anything else - never sent, or skipped as too short - stays unmarked.
        segments.append(segment)
    return segments


FEEDBACK_POPUP_SCRIPT = """
<script>
(function () {
  // st.markdown strips <script>, so the popup is installed from a component
  // iframe reaching into the parent document. Same origin, so this is allowed.
  var win = window.parent;
  var doc = win.document;

  // VERSION is a hash of this script. A plain "already installed" flag made the
  // script un-updatable: Streamlit hot-reloads app.py and re-renders this
  // component, but the guard bailed out and the OLD listeners stayed live, so
  // a fix here did nothing until the browser page itself was reloaded. Keying
  // on the script's own content means an edit replaces the old install.
  var VERSION = "__VERSION__";
  if (win.__fbPopup) {
    if (win.__fbPopup.version === VERSION) { return; }
    try { win.__fbPopup.teardown(); } catch (e) {}
  }

  var pop = doc.createElement("div");
  pop.id = "fb-popup";
  doc.body.appendChild(pop);

  function hide() { pop.className = ""; }

  // The transcript box. The popup is clamped to this, so it can never spill
  // outside the borders of the panel the transcript is displayed in.
  function transcriptBounds() {
    var pane = doc.querySelector(".transcript-pane");
    if (!pane) { return null; }
    var r = pane.getBoundingClientRect();
    return {left: r.left, right: r.right, top: r.top, bottom: r.bottom};
  }

  function show(sent) {
    var card = sent.querySelector(".fb-card");
    if (!card) { return; }
    pop.innerHTML = card.innerHTML;
    pop.className = "show";

    var box = transcriptBounds();
    if (!box) { hide(); return; }
    var pad = 8;
    // Never let the popup be wider than the box it must sit inside.
    pop.style.maxWidth = Math.max(220, Math.min(420, box.right - box.left - 2 * pad)) + "px";

    var s = sent.getBoundingClientRect();
    var w = pop.offsetWidth, h = pop.offsetHeight;

    var left = Math.min(Math.max(s.left, box.left + pad), box.right - w - pad);
    var top = s.bottom + 6;                       // below the sentence
    if (top + h > box.bottom - pad) {
      top = s.top - h - 6;                        // no room below: flip above it
    }
    // Unconditional clamp. Whatever the two preferences above worked out, the
    // popup ends up inside the transcript box - that is the hard requirement,
    // and preferred placement is only a preference.
    top = Math.min(top, box.bottom - h - pad);
    top = Math.max(top, box.top + pad);
    pop.style.left = Math.round(left) + "px";
    pop.style.top = Math.round(top) + "px";
  }

  // Delegated: the transcript's DOM is replaced twice a second, so per-element
  // listeners would not survive.
  function onOver(e) {
    var t = e.target;
    if (!t || !t.closest) { return; }
    var sent = t.closest(".sent.flagged, .sent.failed");
    if (sent) { show(sent); } else { hide(); }
  }
  doc.addEventListener("mouseover", onOver, true);

  // Deliberately NOT a capture-phase "mouseleave" on the document: that fires
  // for every nested element the pointer leaves, so it hid the popup the
  // instant it was shown. Moving onto anything else raises mouseover above,
  // whose else-branch hides it; this only covers leaving the page entirely.
  doc.documentElement.addEventListener("mouseleave", hide);
  win.addEventListener("scroll", hide, true);
  win.addEventListener("resize", hide);

  win.__fbPopup = {
    version: VERSION,
    teardown: function () {
      doc.removeEventListener("mouseover", onOver, true);
      doc.documentElement.removeEventListener("mouseleave", hide);
      win.removeEventListener("scroll", hide, true);
      win.removeEventListener("resize", hide);
      if (pop.parentNode) { pop.parentNode.removeChild(pop); }
    }
  };
})();
</script>
"""


def install_feedback_popup():
    """Zero-height component that installs the hover popup in the parent page."""
    script = FEEDBACK_POPUP_SCRIPT.replace(
        "__VERSION__", hashlib.md5(FEEDBACK_POPUP_SCRIPT.encode("utf-8")).hexdigest()[:10]
    )
    components.html(script, height=0)


def visible_sources():
    """Display filter only. Both mics always record and transcribe."""
    choice = st.session_state.get("transcript_filter", "Both")
    if choice == "Student only":
        return {"student"}
    if choice == "Teacher only":
        return {"teacher"}
    return {"student", "teacher"}


def _render_transcript_pane():
    words = collect_lesson_words()
    turns = group_turns(words)
    interim = interim_by_source() if st.session_state.lesson_state == "recording" else {}
    # Always drain, even with the switch just turned off, so in-flight jobs land
    # in the cache instead of leaking futures.
    drain_live_feedback()
    # Track the switch before any early return: arming during a silence has to
    # open the window there, or the next thing said falls outside it.
    now_audio = newest_word_end(words)
    sync_live_feedback_window(now_audio)
    if not turns and not interim:
        st.caption("The lesson transcript will appear here. Timestamps and speakers come from the two microphones.")
        return
    shown = visible_sources()
    visible_turns = [t for t in turns if t["source"] in shown]

    # Annotate once the switch has been armed at any point this lesson. Turning
    # it off must never erase feedback already gathered, nor abandon sentences
    # from the window that have not settled yet.
    if not (live_feedback_is_on() or live_feedback_engaged()):
        blocks = [_turn_html(t["source"], t["start"], t["text"]) for t in visible_turns]
        for source in ("student", "teacher"):
            if source in shown and interim.get(source):
                blocks.append(_turn_html(source, None, interim[source], interim=True))
        if not blocks:
            _render_empty_filter_note(turns, visible_turns)
            return
        st.markdown(f'<div class="transcript-pane">{"".join(blocks)}</div>',
                    unsafe_allow_html=True)
        _autosave_if_live()
        return

    # Sentences behind the settle boundary can no longer be re-shaped by a late
    # word from the other socket, so they are safe to analyse. Once the words
    # stop arriving nothing can shift, so the whole transcript settles.
    boundary = (now_audio - LIVE_SETTLE_SECONDS
                if st.session_state.lesson_state == "recording" else float("inf"))
    sentences = split_sentences(words, LIVE_FEEDBACK_SOURCE)
    dispatch_live_feedback(settled_sentences(sentences, boundary), turns)
    # Index EVERY sentence, not just the settled ones, or a sentence goes
    # unmarked for exactly the seconds it is in flight and the grey "being
    # checked" state is never seen - it appears already green or red.
    word_to_sentence = index_words_to_sentences(sentences)
    sentences_by_id = {sentence["id"]: sentence for sentence in sentences}
    feedback = st.session_state.live_feedback
    in_flight = set(st.session_state.live_feedback_futures)

    blocks = [
        _turn_html(turn["source"], turn["start"], turn["text"],
                   segments=turn_segments(turn, word_to_sentence, sentences_by_id,
                                          feedback, in_flight))
        for turn in visible_turns
    ]
    for source in ("student", "teacher"):
        if source in shown and interim.get(source):
            blocks.append(_turn_html(source, None, interim[source], interim=True))

    if not blocks:
        _render_empty_filter_note(turns, visible_turns)
        return
    st.markdown(f'<div class="transcript-pane">{"".join(blocks)}</div>',
                unsafe_allow_html=True)
    _autosave_if_live()


def _render_empty_filter_note(turns, visible_turns):
    hidden = len(turns) - len(visible_turns)
    st.caption(f"Nothing from this speaker yet ({hidden} turn(s) hidden by the display filter).")
    _autosave_if_live()


def _autosave_if_live():
    if st.session_state.lesson_state in ("recording", "paused"):
        autosave_lesson()


def _render_live_counts():
    if not (live_feedback_is_on() or live_feedback_engaged()):
        return
    checked, flagged, running, skipped = live_feedback_counts()
    paused = not live_feedback_is_on() and not running
    bits = [f"paused · {checked} checked" if paused else f"{checked} checked"]
    if flagged:
        bits.append(f'<span class="flagged">{flagged} flagged</span>')
    if running:
        bits.append(f"{running} analysing")
    if skipped:
        bits.append(f"{skipped} too short")
    if st.session_state.live_feedback_errors:
        bits.append(f"{st.session_state.live_feedback_errors} failed")
    st.markdown(f'<div class="fb-counts">{" · ".join(bits)}</div>', unsafe_allow_html=True)


if hasattr(st, "fragment"):
    render_transcript_fragment = st.fragment(run_every=0.5)(_render_transcript_pane)
    # Slower: this is a glanceable counter, not the transcript.
    render_live_counts_fragment = st.fragment(run_every=1.0)(_render_live_counts)
else:
    render_transcript_fragment = _render_transcript_pane
    render_live_counts_fragment = _render_live_counts


def render_status_bar():
    state = st.session_state.lesson_state
    state_labels = {
        "idle": "No lesson",
        "recording": "Recording",
        "paused": "Paused",
        "ended": "Lesson ended",
    }
    parts = [f'<span class="state {state}">{state_labels.get(state, state)}</span>']
    lesson_dir = st.session_state.get("lesson_dir")
    if lesson_dir:
        parts.append(f"<span>{html.escape(os.path.basename(lesson_dir))}</span>")
    if st.session_state.get("autosave_last_clock"):
        parts.append(f"<span>Autosaved {st.session_state.autosave_last_clock}</span>")
    missing_keys = []
    if not os.environ.get("DEEPGRAM_API_KEY"):
        missing_keys.append("DEEPGRAM_API_KEY")
    if not os.environ.get("REPLICATE_API_TOKEN"):
        missing_keys.append("REPLICATE_API_TOKEN")
    if not deepseek_api_key():
        missing_keys.append("DEEPSEEK_API_KEY")
    if missing_keys:
        parts.append(f'<span style="color:#b00020;">Missing: {", ".join(missing_keys)}</span>')
    st.markdown(f'<div class="lesson-status">{"".join(parts)}</div>', unsafe_allow_html=True)


def render_after_lesson():
    st.markdown("### After the lesson — student error analysis")
    st.caption(
        "Sends only the student audio to Replicate (Whisper large-v3, single speaker). "
        "Together with the live Deepgram transcript this gives independent ASR viewpoints, "
        "each with full confidence data, for an external LLM to synthesize."
    )
    audio_files = list_lesson_audio()
    if not audio_files:
        st.caption("No student audio yet. End a lesson and its audio will appear here.")
        return

    default_index = 0
    preferred = st.session_state.get("replicate_audio_path")
    if preferred in audio_files:
        default_index = audio_files.index(preferred)
    selected = st.selectbox(
        "Student audio",
        audio_files,
        index=default_index,
        format_func=lambda p: os.path.relpath(p, AUDIO_DIR),
    )
    st.session_state.replicate_audio_path = selected
    if selected and os.path.exists(selected):
        st.audio(selected)

    saved_transcript = replicate_transcript_path(selected)
    existing_transcript = ""
    if os.path.exists(saved_transcript):
        try:
            with open(saved_transcript, "r", encoding="utf-8") as f:
                existing_transcript = f.read().strip()
        except Exception:
            existing_transcript = ""

    running = st.session_state.replicate_future is not None
    if st.button(
        "Transcribe student audio (Whisper large-v3)",
        type="primary",
        disabled=running or not selected,
    ):
        st.session_state.replicate_error = ""
        st.session_state.replicate_transcript = ""
        st.session_state.replicate_future = BACKGROUND_EXECUTOR.submit(run_replicate_job, selected)
        st.rerun()

    if running:
        st.info("Replicate transcription running... this can take a few minutes for a whole lesson.")

    if st.session_state.replicate_error:
        st.error(st.session_state.replicate_error)

    display_transcript = st.session_state.replicate_transcript or existing_transcript
    if display_transcript:
        st.markdown("#### Student transcript (segment confidence included)")
        rendered_lines = [
            f"<p>{html.escape(line)}</p>"
            for line in display_transcript.splitlines()
            if line.strip()
        ]
        st.markdown(
            f'<div class="transcript-pane">{"".join(rendered_lines)}</div>',
            unsafe_allow_html=True,
        )

    lesson_dir = os.path.dirname(selected)
    if not os.path.basename(lesson_dir).startswith("lesson_"):
        lesson_dir = None
    lesson_words = load_lesson_words(lesson_dir) if lesson_dir else []
    deepgram_text = build_deepgram_confidence_transcript(lesson_words) if lesson_words else ""
    bundle = build_synthesis_bundle(lesson_dir, selected)
    stem = os.path.basename(lesson_dir) if lesson_dir else os.path.splitext(os.path.basename(selected))[0]

    # ── DeepSeek synthesis: the single deliverable ──────────────────────────
    # This synthesised transcript is the one output that matters — both
    # speakers, as close to the real utterances as possible, student errors
    # preserved. The raw ASR viewpoints below it are optional intermediates.
    st.markdown("#### Synthesised transcript — the final output")
    st.caption(
        "DeepSeek reconciles the Deepgram and Whisper viewpoints into one "
        "most-likely verbatim transcript, keeping both speakers. The student's "
        "own language errors are preserved for analysis — only ASR uncertainty "
        "is resolved."
    )
    saved_synthesis = synthesis_transcript_path(selected)
    existing_synthesis = ""
    if os.path.exists(saved_synthesis):
        try:
            with open(saved_synthesis, "r", encoding="utf-8") as f:
                existing_synthesis = f.read().strip()
        except Exception:
            existing_synthesis = ""

    synth_running = st.session_state.synthesis_future is not None
    if not deepseek_api_key():
        st.warning("DEEPSEEK_API_KEY is not set — add it to the environment to enable synthesis.")
    if st.button(
        "Synthesise best transcript (DeepSeek)",
        disabled=synth_running or not bundle or not deepseek_api_key(),
        help=None if bundle else "Needs at least one ASR viewpoint (run the live lesson and/or Whisper first).",
    ):
        st.session_state.synthesis_error = ""
        st.session_state.synthesis_transcript = ""
        st.session_state.synthesis_future = BACKGROUND_EXECUTOR.submit(
            run_deepseek_job, lesson_dir, selected
        )
        st.rerun()

    if synth_running:
        st.info("DeepSeek synthesis running... reconciling both viewpoints can take a minute.")

    if st.session_state.synthesis_error:
        st.error(st.session_state.synthesis_error)

    display_synthesis = st.session_state.synthesis_transcript or existing_synthesis
    if display_synthesis:
        rendered_lines = [
            f"<p>{html.escape(line)}</p>"
            for line in display_synthesis.splitlines()
            if line.strip()
        ]
        st.markdown(
            f'<div class="transcript-pane">{"".join(rendered_lines)}</div>',
            unsafe_allow_html=True,
        )
        st.download_button(
            "Download synthesised transcript",
            display_synthesis,
            file_name=f"{stem}_synthesis.txt",
            mime="text/plain",
            type="primary",
        )

    # ── Optional: the raw ASR viewpoints the synthesis was built from ───────
    if deepgram_text or display_transcript or bundle:
        with st.expander("Optional: intermediate ASR viewpoints", expanded=False):
            st.caption(
                "Not needed for normal use — these are the raw inputs the "
                "synthesis reconciles. Kept here in case you want to inspect them."
            )
            col_dg, col_wh, col_bundle = st.columns(3)
            with col_dg:
                if deepgram_text:
                    st.download_button(
                        "Deepgram live (word confidence)",
                        deepgram_text,
                        file_name=f"{stem}_deepgram_confidence.txt",
                        mime="text/plain",
                    )
                else:
                    st.caption("No live transcript for this file.")
            with col_wh:
                if display_transcript:
                    st.download_button(
                        "Whisper student (segment confidence)",
                        display_transcript,
                        file_name=f"{stem}_whisper_confidence.txt",
                        mime="text/plain",
                    )
                else:
                    st.caption("Run the Whisper transcription first.")
            with col_bundle:
                if bundle:
                    st.download_button(
                        "Combined bundle (TXT)",
                        build_bundle_text(bundle, deepgram_text, display_transcript),
                        file_name=f"{stem}_synthesis_bundle.txt",
                        mime="text/plain",
                    )
                else:
                    st.caption("Bundle appears once a viewpoint exists.")


def poll_replicate_job():
    future = st.session_state.replicate_future
    if future is None:
        return
    if not future.done():
        return
    st.session_state.replicate_future = None
    try:
        result = future.result()
    except Exception as e:
        st.session_state.replicate_error = str(e)
        return
    if result.get("ok"):
        st.session_state.replicate_transcript = result.get("transcript") or ""
    else:
        st.session_state.replicate_error = result.get("error") or "Replicate transcription failed."


def poll_synthesis_job():
    future = st.session_state.synthesis_future
    if future is None:
        return
    if not future.done():
        return
    st.session_state.synthesis_future = None
    try:
        result = future.result()
    except Exception as e:
        st.session_state.synthesis_error = str(e)
        return
    if result.get("ok"):
        st.session_state.synthesis_transcript = result.get("transcript") or ""
    else:
        st.session_state.synthesis_error = result.get("error") or "DeepSeek synthesis failed."


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
render_styles()
init_session_state()
poll_replicate_job()
poll_synthesis_job()

st.title("Lesson Transcriber")
render_status_bar()

recorder_value = audio_recorder_component(key="audio_recorder")
recorder_payload = parse_recorder_value(recorder_value)
if recorder_payload:
    handle_recorder_event(recorder_payload)

st.markdown("### Transcript")
filter_col, live_col, note_col = st.columns([2, 2, 3])
with filter_col:
    st.radio(
        "Show on screen",
        ["Both", "Student only", "Teacher only"],
        key="transcript_filter",
        horizontal=True,
    )
with live_col:
    _has_deepseek = bool(deepseek_api_key())
    st.toggle(
        "Live language feedback",
        key="live_feedback_on",
        disabled=not _has_deepseek,
        help=(
            "Sends each student sentence spoken while this is on to DeepSeek. "
            "Underline shows the state: grey while checking, green if clean, red "
            "if an error was found — hover a red one for the fix. Single-word "
            "replies (\"Yeah.\") are not sent, so they stay unmarked."
            if _has_deepseek
            else "Needs DEEPSEEK_API_KEY in the environment."
        ),
    )
    # Mirror the toggle here, where the widget actually exists.
    mirror_live_feedback_switch()
    render_live_counts_fragment()
with note_col:
    st.caption("Display only — both microphones always record, transcribe, and save to the lesson file.")
render_transcript_fragment()
install_feedback_popup()

st.markdown("---")
render_after_lesson()

with st.expander("Diagnostics", expanded=False):
    diag = {
        "lesson_state": st.session_state.lesson_state,
        "lesson_dir": st.session_state.get("lesson_dir"),
        "deepgram_key_present": bool(os.environ.get("DEEPGRAM_API_KEY")),
        "replicate_key_present": bool(os.environ.get("REPLICATE_API_TOKEN")),
        "deepseek_key_present": bool(deepseek_api_key()),
        "deepseek_model": DEEPSEEK_MODEL,
        "deepseek_live_model": DEEPSEEK_LIVE_MODEL,
        "live_feedback_on": st.session_state.live_feedback_on,
        "live_feedback_calls": st.session_state.live_feedback_calls,
        "live_feedback_errors": st.session_state.live_feedback_errors,
        "live_feedback_cached": len(st.session_state.live_feedback),
        "live_feedback_in_flight": len(st.session_state.live_feedback_futures),
        "python_executable": sys.executable,
    }
    for source, streamer in st.session_state.deepgram_streamers.items():
        snap = streamer.snapshot()
        diag[f"{source}_stream"] = {
            "active": streamer.is_active,
            "error": snap["error"],
            "bytes_sent": snap["bytes_sent_total"],
            "chunks_received": snap["chunks_received"],
            "transcript_events": snap["transcript_events_received"],
            "sample_rate": snap["sample_rate"],
        }
    st.json(diag)

with st.expander("Live feedback prompt (editable)", expanded=False):
    st.caption(
        "System prompt for the live per-sentence check. The app parses the reply as "
        'JSON shaped {"verdict": "ok"|"issue", "label": ..., "explanation": ..., '
        '"rewrites": [...]} — keep that contract or nothing will render. Saved to '
        f"{LIVE_FEEDBACK_PROMPT_PATH} and reused next time."
    )
    st.text_area(
        "Prompt",
        key="live_feedback_prompt",
        height=320,
        on_change=persist_live_feedback_prompt,
        label_visibility="collapsed",
    )
    prompt_col, rerun_col, model_col = st.columns([1, 1, 2])
    with prompt_col:
        if st.button("Reset to default"):
            st.session_state.live_feedback_prompt = DEFAULT_LIVE_FEEDBACK_PROMPT
            persist_live_feedback_prompt()
            st.rerun()
    with rerun_col:
        if st.button("Re-analyse all", help="Clear cached results so the edited prompt is applied to sentences already checked."):
            reset_live_feedback()
            st.rerun()
    with model_col:
        st.caption(f"Model: `{DEEPSEEK_LIVE_MODEL}` (set DEEPSEEK_LIVE_MODEL to change)")

# Keep polling while a background job is running so completion is detected
# without user interaction. st.rerun() preserves session_state.
if (
    st.session_state.replicate_future is not None
    or st.session_state.synthesis_future is not None
):
    time.sleep(2)
    st.rerun()
