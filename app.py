"""Lesson Transcriber.

Live lesson transcription with two microphones (teacher + student), a single
timestamped transcript that autosaves during class, and an optional
post-lesson high-accuracy pass through Replicate (whisper-diarization).
"""

import base64
import concurrent.futures
import html
import json
import os
import re
import sys
import threading
import time
import urllib.parse
from datetime import datetime, timezone
from time import perf_counter

import replicate
import streamlit as st
import streamlit.components.v1 as components

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
APP_EVENT_LOG = "/tmp/esl_app_events.log"
LEGACY_AUDIO_EXTS = {".wav", ".m4a"}
LESSON_AUDIO_EXTS = {".wav", ".m4a", ".webm", ".ogg", ".mp4", ".mp3"}
SOURCE_LABELS = {"student": "Student", "teacher": "Teacher"}
AUTOSAVE_INTERVAL_S = 5.0
TURN_GAP_SECONDS = 8.0
TURN_MAX_SECONDS = 60.0
REPLICATE_TIMEOUT_S = 1800
DEEPGRAM_RESTART_MIN_INTERVAL_S = 1.0
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
        self.pending_chunks = []
        self.max_pending_chunks = 40
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

    def _queue_chunk(self, chunk_bytes):
        self.pending_chunks.append(chunk_bytes)
        if len(self.pending_chunks) > self.max_pending_chunks:
            self.pending_chunks = self.pending_chunks[-self.max_pending_chunks:]

    def _flush_pending_chunks(self):
        with self.lock:
            if not self.connected_event.is_set() or not self.pending_chunks:
                return
            queued = self.pending_chunks
            self.pending_chunks = []
        for chunk_bytes in queued:
            self.send_pcm16(chunk_bytes)

    def _send_keepalive_if_idle(self, now):
        """KeepAlive frames hold the socket open through lesson pauses."""
        with self.lock:
            idle_s = now - self.last_audio_sent_at
            ws_app = self.ws
        if idle_s < 3.0 or not ws_app:
            return
        try:
            ws_app.send(json.dumps({"type": "KeepAlive"}))
            with self.lock:
                self.last_audio_sent_at = now
        except Exception as e:
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
            if not ws_app or not self.connected_event.is_set():
                self._queue_chunk(chunk_bytes)
                return
            try:
                ws_app.send(chunk_bytes, opcode=2)
                self.last_audio_sent_at = perf_counter()
                self.bytes_sent_total += len(chunk_bytes)
            except Exception as e:
                self.error = str(e)
                self.connected_event.clear()
                self._queue_chunk(chunk_bytes)
                log_event("deepgram_stream_send_error", error=self.error)

    def audio_seconds_sent(self):
        return self.bytes_sent_total / (2.0 * self.sample_rate) if self.sample_rate else 0.0

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
        _retire_streamer(source, existing)
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


def _retire_streamer(source, streamer):
    """Fold a stream's words into the lesson and advance the time offset."""
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
    try:
        streamer.close()
    except Exception:
        pass


def restart_dead_stream(source, sample_rate):
    now = perf_counter()
    last_restart = st.session_state.get("deepgram_last_restart_at")
    if last_restart is not None and now - last_restart < DEEPGRAM_RESTART_MIN_INTERVAL_S:
        return
    streamer = st.session_state.deepgram_streamers.pop(source, None)
    if streamer:
        log_event("deepgram_stream_restart", source=source, last_error=streamer.error)
        _retire_streamer(source, streamer)
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
            })
        else:
            current["tokens"].append(token)
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
    """Lesson-folder audio first (newest lesson first), then legacy loose files."""
    if not os.path.isdir(AUDIO_DIR):
        return []
    lesson_files = []
    legacy_files = []
    for name in os.listdir(AUDIO_DIR):
        path = os.path.join(AUDIO_DIR, name)
        if os.path.isdir(path) and name.startswith("lesson_"):
            for inner in sorted(os.listdir(path)):
                inner_path = os.path.join(path, inner)
                if os.path.isfile(inner_path) and os.path.splitext(inner)[1].lower() in LESSON_AUDIO_EXTS:
                    lesson_files.append(inner_path)
        elif os.path.isfile(path) and os.path.splitext(name)[1].lower() in LEGACY_AUDIO_EXTS:
            legacy_files.append(path)
    lesson_files.sort(key=os.path.getmtime, reverse=True)
    legacy_files.sort(key=os.path.getmtime, reverse=True)
    return lesson_files + legacy_files


# ---------------------------------------------------------------------------
# Replicate (post-lesson accurate transcript)
# ---------------------------------------------------------------------------
def _format_replicate_time(sec):
    m = int(sec // 60)
    s = sec % 60
    return f"{m:02d}:{s:05.2f}"


def _normalize_text(t):
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"[\W_]+", "", t)
    return t


def _dedupe_sentences(text):
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
        if not isinstance(seg, dict):
            continue
        speaker = seg.get("speaker") or "UNKNOWN"
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        if speaker == "UNKNOWN" and merged:
            speaker = merged[-1]["speaker"]
        text = _dedupe_sentences(text)
        text_norm = _normalize_text(text)
        if text_norm and text_norm == last_text_norm:
            continue
        if merged and merged[-1]["speaker"] == speaker:
            merged[-1]["end"] = end
            merged[-1]["text"] += " " + text
        else:
            merged.append({"speaker": speaker, "start": start, "end": end, "text": text})
        last_text_norm = text_norm
    return merged


def _extract_replicate_segments(output):
    if isinstance(output, dict):
        segments = output.get("segments")
        if isinstance(segments, list):
            return segments
    elif isinstance(output, list):
        return output
    return []


def build_replicate_transcript(output):
    """Timestamped speaker transcript from Replicate diarization output."""
    merged = _merge_segments(_extract_replicate_segments(output))
    lines = []
    for seg in merged:
        stamp = _format_clock(seg.get("start", 0.0))
        lines.append(f"[{stamp}] {seg['speaker']}: {seg['text']}")
    return "\n\n".join(lines)


def run_replicate_transcription(audio_path):
    replicate_token = os.environ.get("REPLICATE_API_TOKEN")
    if not replicate_token:
        raise RuntimeError("REPLICATE_API_TOKEN not set. Add it to the environment and retry.")
    model_id = "thomasmol/whisper-diarization:1495a9cddc83b2203b0d8d3516e38b80fd1572ebc4bc5700ac1da56a9b3ed886"
    log_event("replicate_call_attempt", model_id=model_id, audio_path=audio_path)
    try:
        with open(audio_path, "rb") as f:
            output = replicate.run(
                model_id,
                input={"file": f, "output": "json", "group_segments": True},
            )
    except Exception as e:
        message = str(e)
        log_event("replicate_call_error", error=message, error_type=type(e).__name__)
        if "Invalid token" in message or "status: 401" in message:
            raise RuntimeError(
                "Replicate authentication failed. Set a valid REPLICATE_API_TOKEN."
            ) from e
        raise RuntimeError(f"Replicate transcription failed: {message}") from e
    transcript = build_replicate_transcript(output)
    if not transcript:
        raise RuntimeError("Replicate returned no usable segments.")
    log_event("replicate_success", segments_count=len(_extract_replicate_segments(output)))
    return output, transcript


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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


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
        </style>
        """,
        unsafe_allow_html=True,
    )


def _turn_html(source, start, text, interim=False):
    label = SOURCE_LABELS.get(source, str(source).title())
    classes = f"turn {html.escape(str(source))}" + (" interim" if interim else "")
    stamp = "..." if interim else _format_clock(start)
    return (
        f'<div class="{classes}">'
        f'<span class="ts">[{stamp}]</span>'
        f'<span class="speaker">{html.escape(label)}:</span>'
        f'<span class="text">{html.escape(text)}</span>'
        f"</div>"
    )


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
    if not turns and not interim:
        st.caption("The lesson transcript will appear here. Timestamps and speakers come from the two microphones.")
        return
    shown = visible_sources()
    visible_turns = [t for t in turns if t["source"] in shown]
    blocks = [_turn_html(t["source"], t["start"], t["text"]) for t in visible_turns]
    for source in ("student", "teacher"):
        if source in shown and interim.get(source):
            blocks.append(_turn_html(source, None, interim[source], interim=True))
    if not blocks:
        hidden = len(turns) - len(visible_turns)
        st.caption(f"Nothing from this speaker yet ({hidden} turn(s) hidden by the display filter).")
        if st.session_state.lesson_state in ("recording", "paused"):
            autosave_lesson()
        return
    st.markdown(f'<div class="transcript-pane">{"".join(blocks)}</div>', unsafe_allow_html=True)
    if st.session_state.lesson_state in ("recording", "paused"):
        autosave_lesson()


if hasattr(st, "fragment"):
    render_transcript_fragment = st.fragment(run_every=0.5)(_render_transcript_pane)
else:
    render_transcript_fragment = _render_transcript_pane


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
    if missing_keys:
        parts.append(f'<span style="color:#b00020;">Missing: {", ".join(missing_keys)}</span>')
    st.markdown(f'<div class="lesson-status">{"".join(parts)}</div>', unsafe_allow_html=True)


def render_after_lesson():
    st.markdown("### After the lesson")
    audio_files = list_lesson_audio()
    if not audio_files:
        st.caption("No lesson audio yet. End a lesson and its audio will appear here.")
        return

    default_index = 0
    preferred = st.session_state.get("replicate_audio_path")
    if preferred in audio_files:
        default_index = audio_files.index(preferred)
    selected = st.selectbox(
        "Lesson audio",
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
        "Upload to Replicate for accurate transcript",
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
        st.markdown("#### Replicate transcript")
        st.markdown(
            f'<div class="transcript-pane">{"".join(f"<p>{html.escape(line)}</p>" for line in display_transcript.splitlines() if line.strip())}</div>',
            unsafe_allow_html=True,
        )
        st.download_button(
            "Download Replicate transcript",
            display_transcript,
            file_name=f"{os.path.splitext(os.path.basename(selected))[0]}_replicate_transcript.txt",
            mime="text/plain",
        )

    # Live transcript download for the currently selected lesson folder.
    lesson_dir = os.path.dirname(selected)
    live_path = lesson_transcript_path(lesson_dir)
    if os.path.basename(lesson_dir).startswith("lesson_") and os.path.exists(live_path):
        try:
            with open(live_path, "r", encoding="utf-8") as f:
                live_text = f.read()
            st.download_button(
                "Download live lesson transcript",
                live_text,
                file_name=f"{os.path.basename(lesson_dir)}_transcript.txt",
                mime="text/plain",
            )
        except Exception:
            pass


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


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
render_styles()
init_session_state()
poll_replicate_job()

st.title("Lesson Transcriber")
render_status_bar()

recorder_value = audio_recorder_component(key="audio_recorder")
recorder_payload = parse_recorder_value(recorder_value)
if recorder_payload:
    handle_recorder_event(recorder_payload)

st.markdown("### Transcript")
filter_col, note_col = st.columns([2, 3])
with filter_col:
    st.radio(
        "Show on screen",
        ["Both", "Student only", "Teacher only"],
        key="transcript_filter",
        horizontal=True,
    )
with note_col:
    st.caption("Display only — both microphones always record, transcribe, and save to the lesson file.")
render_transcript_fragment()

st.markdown("---")
render_after_lesson()

with st.expander("Diagnostics", expanded=False):
    diag = {
        "lesson_state": st.session_state.lesson_state,
        "lesson_dir": st.session_state.get("lesson_dir"),
        "deepgram_key_present": bool(os.environ.get("DEEPGRAM_API_KEY")),
        "replicate_key_present": bool(os.environ.get("REPLICATE_API_TOKEN")),
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

# Keep polling while a Replicate job is running so completion is detected
# without user interaction. st.rerun() preserves session_state.
if st.session_state.replicate_future is not None:
    time.sleep(2)
    st.rerun()
