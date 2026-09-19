"""Process-lifetime singletons, background audio ingestion daemon, and provider manager.

Runs independently of Streamlit reruns.
- AudioIngestionServer: Binds to 127.0.0.1:8765, accepts binary PCM chunks with sequence tracking.
- TrackRecorder: Authoritative on-disk sink (student.ogg, teacher.ogg) written BEFORE chunk ACK.
- ProviderManager: Isolated provider streaming (Deepgram, Gladia, AssemblyAI) with 1s watchdog,
  rolling pre-buffer replay, and transcript deduplication.
- Durable session manifest: Atomically maintains session_manifest.json for crash recovery.
"""

from __future__ import annotations

import os
import sys

def _load_env_files():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for name in (".env.project1.local", ".env.local", ".env"):
        p = os.path.join(base_dir, name)
        if not os.path.isfile(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            pass

_load_env_files()

import collections
import concurrent.futures
import http.server
import json
import threading
import time
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Dict, List, Optional, Set, Tuple

import lesson_audio
from deepgram_stream import DeepgramStreamingClient
from gladia_stream import GladiaStreamingClient
from assemblyai_stream import AssemblyAIStreamingClient
from live_consensus import build_consensus_words, normalize_token

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INGESTION_HOST = os.environ.get("ESL_INGESTION_HOST", "127.0.0.1")
INGESTION_PORT = int(os.environ.get("ESL_INGESTION_PORT", "8765"))
PREBUFFER_DURATION_S = 3.0
MAX_SEQS_HISTORY = 1000
WATCHDOG_INTERVAL_S = 1.0
APP_EVENT_LOG = os.environ.get("ESL_EVENT_LOG", "audio/events.log")

# Background thread pool for LLM live feedback calls
EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=3, thread_name_prefix="live-feedback"
)

_CLIENTS = {}
_CLIENTS_LOCK = threading.Lock()


def openai_client(api_key, base_url, timeout):
    """One client per key, reused for the life of the process."""
    key = (api_key, base_url, timeout)
    with _CLIENTS_LOCK:
        client = _CLIENTS.get(key)
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
            _CLIENTS[key] = client
        return client


def log_event(event: str, **fields):
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


# ---------------------------------------------------------------------------
# Active Lesson Session (Authoritative on-disk sink + Sequence tracking)
# ---------------------------------------------------------------------------
class TrackState:
    def __init__(self, source: str, sample_rate: int, recorder: lesson_audio.TrackRecorder):
        self.source = source
        self.sample_rate = sample_rate
        self.recorder = recorder
        self.samples_received = 0
        self.last_seq = -1
        self.expected_seq = 0
        self.received_seqs: collections.deque[int] = collections.deque(maxlen=MAX_SEQS_HISTORY)
        self.received_seqs_set: Set[int] = set()
        self.dropped_chunks = 0
        self.duplicate_chunks = 0
        self.rolling_prebuffer: collections.deque[Tuple[int, bytes, int]] = collections.deque(maxlen=30)
        self.last_chunk_at = perf_counter()


class ActiveLessonSession:
    def __init__(self, lesson_dir: str, student: str = "", capture_mode: str = "audio_worklet"):
        self.lesson_dir = lesson_dir
        self.student = student
        self.capture_mode = capture_mode
        self.state = "recording"  # recording, paused, ended
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.tracks: Dict[str, TrackState] = {}
        self.lock = threading.RLock()
        self.manifest_path = os.path.join(lesson_dir, "session_manifest.json")
        os.makedirs(self.lesson_dir, exist_ok=True)
        self.write_manifest()

    def add_track(self, source: str, sample_rate: int):
        with self.lock:
            if source in self.tracks:
                return self.tracks[source]
            os.makedirs(self.lesson_dir, exist_ok=True)
            recorder = lesson_audio.TrackRecorder(
                self.lesson_dir, source, sample_rate, on_event=log_event
            )
            track = TrackState(source, sample_rate, recorder)
            self.tracks[source] = track
            self.write_manifest()
            return track

    def get_audio_clock(self, source: Optional[str] = None) -> float:
        with self.lock:
            if source and source in self.tracks:
                t = self.tracks[source]
                return t.samples_received / float(t.sample_rate) if t.sample_rate else 0.0
            candidates = [0.0]
            for t in self.tracks.values():
                if t.sample_rate:
                    candidates.append(t.samples_received / float(t.sample_rate))
            return max(candidates)

    def write_manifest(self):
        try:
            payload = {
                "version": 1,
                "lesson_dir": self.lesson_dir,
                "student": self.student,
                "status": self.state,
                "started_at": self.started_at,
                "capture_mode": self.capture_mode,
                "audio_clock_s": round(self.get_audio_clock(), 2),
                "tracks": {
                    src: {
                        "sample_rate": t.sample_rate,
                        "samples_received": t.samples_received,
                        "last_seq": t.last_seq,
                        "dropped_chunks": t.dropped_chunks,
                        "duplicate_chunks": t.duplicate_chunks,
                    }
                    for src, t in self.tracks.items()
                },
            }
            os.makedirs(self.lesson_dir, exist_ok=True)
            tmp_path = f"{self.manifest_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.manifest_path)
        except Exception as e:
            log_event("manifest_write_error", error=str(e))

    def write_chunk(
        self, source: str, seq: int, chunk_bytes: bytes, sample_rate: int
    ) -> Tuple[str, bool]:
        """Write chunk to authoritative TrackRecorder path.

        Returns (status, ok):
        - ("duplicate", True): Chunk seq already received; ignored without error.
        - ("accepted", True): Chunk written to recorder.
        - ("error", False): Recorder failed or closed.
        """
        if self.state != "recording":
            return "paused_or_ended", True

        with self.lock:
            track = self.tracks.get(source)
            if track is None:
                track = self.add_track(source, sample_rate)

            # 1. Duplicate check
            if seq in track.received_seqs_set:
                track.duplicate_chunks += 1
                return "duplicate", True

            # 2. Sequence gap detection
            if track.last_seq >= 0 and seq > track.last_seq + 1:
                gap = seq - (track.last_seq + 1)
                track.dropped_chunks += gap
                log_event("chunk_gap_detected", source=source, expected=track.last_seq + 1, received=seq, gap=gap)

            track.last_seq = seq
            track.expected_seq = seq + 1
            if len(track.received_seqs) == track.received_seqs.maxlen:
                evicted = track.received_seqs.popleft()
                track.received_seqs_set.discard(evicted)
            track.received_seqs.append(seq)
            track.received_seqs_set.add(seq)
            track.last_chunk_at = perf_counter()

            # 3. Rolling prebuffer for provider watchdog replay
            track.rolling_prebuffer.append((seq, chunk_bytes, sample_rate))

            # 4. Authoritative write to TrackRecorder sink
            if track.recorder.closed or track.recorder.error:
                return "recorder_error", False

            track.recorder.write(chunk_bytes)
            samples_in_chunk = len(chunk_bytes) // 2
            track.samples_received += samples_in_chunk

        return "accepted", True

    def close(self):
        with self.lock:
            self.state = "ended"
            for track in self.tracks.values():
                try:
                    track.recorder.close()
                except Exception as e:
                    log_event("track_recorder_close_error", source=track.source, error=str(e))
            self.write_manifest()


# ---------------------------------------------------------------------------
# Provider Manager (Watchdogs, Rolling Replay, Deduplication)
# ---------------------------------------------------------------------------
class ProviderManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.deepgram_streamers: Dict[str, DeepgramStreamingClient] = {}
        self.gladia_streamers: Dict[str, GladiaStreamingClient] = {}
        self.assemblyai_streamers: Dict[str, AssemblyAIStreamingClient] = {}

        self.deepgram_offsets: Dict[str, float] = {}
        self.gladia_offsets: Dict[str, float] = {}
        self.assemblyai_offsets: Dict[str, float] = {}

        self.retired_words_deepgram: List[Dict[str, Any]] = []
        self.retired_words_gladia: List[Dict[str, Any]] = []
        self.retired_words_assemblyai: List[Dict[str, Any]] = []

        self.seen_word_keys: Set[Tuple[str, str, float]] = set()
        self.stop_event = threading.Event()
        self.watchdog_thread: Optional[threading.Thread] = None
        self.reconnect_counts = {"deepgram": 0, "gladia": 0, "assemblyai": 0}
        self._start_watchdog()

    def _start_watchdog(self):
        if self.watchdog_thread and self.watchdog_thread.is_alive():
            return

        def _watchdog_loop():
            while not self.stop_event.is_set():
                time.sleep(WATCHDOG_INTERVAL_S)
                if self.stop_event.is_set():
                    break
                self._check_and_recover_providers()

        self.watchdog_thread = threading.Thread(
            target=_watchdog_loop, daemon=True, name="provider-watchdog"
        )
        self.watchdog_thread.start()

    def _check_and_recover_providers(self):
        session = get_active_session()
        if not session or session.state != "recording":
            return

        with self.lock:
            # 1. Deepgram watchdog
            dg_key = os.environ.get("DEEPGRAM_API_KEY")
            if dg_key:
                for source, track in session.tracks.items():
                    streamer = self.deepgram_streamers.get(source)
                    if streamer is None or (not streamer.is_active and not streamer.is_connecting):
                        self._recover_streamer("deepgram", source, track, dg_key)

            # 2. Gladia watchdog
            gl_key = os.environ.get("GLADIA_API_KEY")
            if gl_key:
                for source, track in session.tracks.items():
                    streamer = self.gladia_streamers.get(source)
                    if streamer is None or (not streamer.is_active and not streamer.is_connecting):
                        self._recover_streamer("gladia", source, track, gl_key)

            # 3. AssemblyAI watchdog
            aai_key = os.environ.get("ASSEMBLYAI_API_KEY")
            if aai_key:
                for source, track in session.tracks.items():
                    streamer = self.assemblyai_streamers.get(source)
                    if streamer is None or (not streamer.is_active and not streamer.is_connecting):
                        self._recover_streamer("assemblyai", source, track, aai_key)

    def _recover_streamer(self, provider: str, source: str, track: TrackState, api_key: str):
        log_event(f"{provider}_stream_watchdog_restart", source=source)
        self.reconnect_counts[provider] = self.reconnect_counts.get(provider, 0) + 1

        # Retire dead streamer and record timeline offset
        if provider == "deepgram":
            old = self.deepgram_streamers.pop(source, None)
            if old:
                self._retire_deepgram_streamer(source, old)
            new_streamer = DeepgramStreamingClient(api_key, track.sample_rate, on_event=log_event)
            new_streamer.start()
            self.deepgram_streamers[source] = new_streamer
        elif provider == "gladia":
            old = self.gladia_streamers.pop(source, None)
            if old:
                self._retire_gladia_streamer(source, old)
            new_streamer = GladiaStreamingClient(api_key=api_key, sample_rate=track.sample_rate, on_event=log_event)
            new_streamer.start()
            self.gladia_streamers[source] = new_streamer
        elif provider == "assemblyai":
            old = self.assemblyai_streamers.pop(source, None)
            if old:
                self._retire_assemblyai_streamer(source, old)
            new_streamer = AssemblyAIStreamingClient(api_key=api_key, sample_rate=track.sample_rate, on_event=log_event)
            new_streamer.start()
            self.assemblyai_streamers[source] = new_streamer
        else:
            return

        # Replay rolling prebuffer to recover speech during the reconnect gap
        with track.recorder._lock:
            chunks_to_replay = list(track.rolling_prebuffer)
        for seq, chunk_bytes, s_rate in chunks_to_replay:
            new_streamer.send_pcm16(chunk_bytes)

    def _retire_deepgram_streamer(self, source: str, streamer: DeepgramStreamingClient):
        snap = streamer.snapshot()
        offset = self.deepgram_offsets.get(source, 0.0)
        for w in snap["final_words"]:
            st_time = (w.get("start") or 0.0) + offset
            end_time = (w.get("end") or 0.0) + offset
            token = w.get("word") or ""
            self.retired_words_deepgram.append({
                "source": source,
                "word": token,
                "start": st_time,
                "end": end_time,
                "confidence": w.get("confidence"),
            })
        self.deepgram_offsets[source] = offset + streamer.audio_seconds_sent()
        threading.Thread(target=lambda: streamer.close(), daemon=True, name=f"deepgram-close-{source}").start()

    def _retire_gladia_streamer(self, source: str, streamer: GladiaStreamingClient):
        snap = streamer.snapshot()
        offset = self.gladia_offsets.get(source, 0.0)
        for w in snap["final_words"]:
            st_time = (w.get("start") or 0.0) + offset
            end_time = (w.get("end") or 0.0) + offset
            token = w.get("word") or ""
            self.retired_words_gladia.append({
                "source": source,
                "word": token,
                "start": st_time,
                "end": end_time,
                "confidence": w.get("confidence"),
            })
        self.gladia_offsets[source] = offset + streamer.audio_seconds_sent()
        threading.Thread(target=lambda: streamer.close(), daemon=True, name=f"gladia-close-{source}").start()

    def _retire_assemblyai_streamer(self, source: str, streamer: AssemblyAIStreamingClient):
        snap = streamer.snapshot()
        offset = self.assemblyai_offsets.get(source, 0.0)
        for w in snap["final_words"]:
            st_time = (w.get("start") or 0.0) + offset
            end_time = (w.get("end") or 0.0) + offset
            token = w.get("word") or ""
            self.retired_words_assemblyai.append({
                "source": source,
                "word": token,
                "start": st_time,
                "end": end_time,
                "confidence": w.get("confidence"),
            })
        self.assemblyai_offsets[source] = offset + streamer.audio_seconds_sent()
        threading.Thread(target=lambda: streamer.close(), daemon=True, name=f"assemblyai-close-{source}").start()

    def start_streams(self, sources: Dict[str, int]):
        with self.lock:
            for source, sample_rate in sources.items():
                dg_key = os.environ.get("DEEPGRAM_API_KEY")
                if dg_key and source not in self.deepgram_streamers:
                    st = DeepgramStreamingClient(dg_key, sample_rate, on_event=log_event)
                    st.start()
                    self.deepgram_streamers[source] = st
                    self.deepgram_offsets.setdefault(source, 0.0)

                gl_key = os.environ.get("GLADIA_API_KEY")
                if gl_key and source not in self.gladia_streamers:
                    st = GladiaStreamingClient(api_key=gl_key, sample_rate=sample_rate, on_event=log_event)
                    st.start()
                    self.gladia_streamers[source] = st
                    self.gladia_offsets.setdefault(source, 0.0)

                aai_key = os.environ.get("ASSEMBLYAI_API_KEY")
                if aai_key and source not in self.assemblyai_streamers:
                    st = AssemblyAIStreamingClient(api_key=aai_key, sample_rate=sample_rate, on_event=log_event)
                    st.start()
                    self.assemblyai_streamers[source] = st
                    self.assemblyai_offsets.setdefault(source, 0.0)

    def dispatch_audio(self, source: str, chunk_bytes: bytes, sample_rate: int):
        """Dispatch audio chunk asynchronously to all active provider streamers."""
        with self.lock:
            # 1. Deepgram streamer auto-start / dispatch
            dg = self.deepgram_streamers.get(source)
            if dg is None:
                dg_key = os.environ.get("DEEPGRAM_API_KEY")
                if dg_key:
                    st = DeepgramStreamingClient(dg_key, sample_rate, on_event=log_event)
                    st.start()
                    self.deepgram_streamers[source] = st
                    self.deepgram_offsets.setdefault(source, 0.0)
                    dg = st
            if dg:
                dg.send_pcm16(chunk_bytes)

            # 2. Gladia streamer auto-start / dispatch
            gl = self.gladia_streamers.get(source)
            if gl is None:
                gl_key = os.environ.get("GLADIA_API_KEY")
                if gl_key:
                    st = GladiaStreamingClient(api_key=gl_key, sample_rate=sample_rate, on_event=log_event)
                    st.start()
                    self.gladia_streamers[source] = st
                    self.gladia_offsets.setdefault(source, 0.0)
                    gl = st
            if gl:
                gl.send_pcm16(chunk_bytes)

            # 3. AssemblyAI streamer auto-start / dispatch
            aai = self.assemblyai_streamers.get(source)
            if aai is None:
                aai_key = os.environ.get("ASSEMBLYAI_API_KEY")
                if aai_key:
                    st = AssemblyAIStreamingClient(api_key=aai_key, sample_rate=sample_rate, on_event=log_event)
                    st.start()
                    self.assemblyai_streamers[source] = st
                    self.assemblyai_offsets.setdefault(source, 0.0)
                    aai = st
            if aai:
                aai.send_pcm16(chunk_bytes)

    def collect_raw_words(self, provider: str) -> List[Dict[str, Any]]:
        with self.lock:
            if provider == "deepgram":
                words = list(self.retired_words_deepgram)
                for src, streamer in self.deepgram_streamers.items():
                    snap = streamer.snapshot()
                    off = self.deepgram_offsets.get(src, 0.0)
                    for w in snap["final_words"]:
                        words.append({
                            "source": src,
                            "word": w["word"],
                            "start": (w.get("start") or 0.0) + off,
                            "end": (w.get("end") or 0.0) + off,
                            "confidence": w.get("confidence"),
                        })
                words.sort(key=lambda x: (x.get("start") or 0.0))
                return words
            elif provider == "gladia":
                words = list(self.retired_words_gladia)
                for src, streamer in self.gladia_streamers.items():
                    snap = streamer.snapshot()
                    off = self.gladia_offsets.get(src, 0.0)
                    for w in snap["final_words"]:
                        words.append({
                            "source": src,
                            "word": w["word"],
                            "start": (w.get("start") or 0.0) + off,
                            "end": (w.get("end") or 0.0) + off,
                            "confidence": w.get("confidence"),
                        })
                words.sort(key=lambda x: (x.get("start") or 0.0))
                return words
            elif provider == "assemblyai":
                words = list(self.retired_words_assemblyai)
                for src, streamer in self.assemblyai_streamers.items():
                    snap = streamer.snapshot()
                    off = self.assemblyai_offsets.get(src, 0.0)
                    for w in snap["final_words"]:
                        words.append({
                            "source": src,
                            "word": w["word"],
                            "start": (w.get("start") or 0.0) + off,
                            "end": (w.get("end") or 0.0) + off,
                            "confidence": w.get("confidence"),
                        })
                words.sort(key=lambda x: (x.get("start") or 0.0))
                return words
            return []

    def get_interim_text(self) -> Dict[str, str]:
        with self.lock:
            res = {}
            for src, streamer in self.deepgram_streamers.items():
                snap = streamer.snapshot()
                if snap.get("interim"):
                    res[src] = snap["interim"]
            return res

    def stop_all(self):
        with self.lock:
            for src, s in list(self.deepgram_streamers.items()):
                self._retire_deepgram_streamer(src, s)
            self.deepgram_streamers.clear()

            for src, s in list(self.gladia_streamers.items()):
                self._retire_gladia_streamer(src, s)
            self.gladia_streamers.clear()

            for src, s in list(self.assemblyai_streamers.items()):
                self._retire_assemblyai_streamer(src, s)
            self.assemblyai_streamers.clear()

    def reset(self):
        with self.lock:
            self.stop_all()
            self.deepgram_offsets.clear()
            self.gladia_offsets.clear()
            self.assemblyai_offsets.clear()
            self.retired_words_deepgram.clear()
            self.retired_words_gladia.clear()
            self.retired_words_assemblyai.clear()
            self.seen_word_keys.clear()
            self.reconnect_counts = {"deepgram": 0, "gladia": 0, "assemblyai": 0}


# ---------------------------------------------------------------------------
# Global Session & Provider State
# ---------------------------------------------------------------------------
_GLOBAL_LOCK = threading.Lock()
_ACTIVE_SESSION: Optional[ActiveLessonSession] = None
_PROVIDER_MANAGER = ProviderManager()


def get_active_session() -> Optional[ActiveLessonSession]:
    with _GLOBAL_LOCK:
        return _ACTIVE_SESSION


def set_active_session(session: Optional[ActiveLessonSession]):
    global _ACTIVE_SESSION
    with _GLOBAL_LOCK:
        _ACTIVE_SESSION = session


def get_provider_manager() -> ProviderManager:
    return _PROVIDER_MANAGER


# ---------------------------------------------------------------------------
# HTTP Audio Ingestion Server
# ---------------------------------------------------------------------------
class AudioIngestionHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress noisy HTTP request logging to stderr
        pass

    def _send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Content-Length, X-Track-Id, X-Seq, X-Sample-Rate, X-Timestamp, Authorization, *",
        )
        self.send_header("Access-Control-Max-Age", "86400")

    def _send_json(self, status_code: int, data: Dict[str, Any]):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self._send_cors()
        self.end_headers()

    def do_GET(self):
        if self.path in ("/health", "/healthz"):
            self._send_json(200, {"status": "ok", "time": datetime.now(timezone.utc).isoformat()})
            return

        if self.path == "/status":
            session = get_active_session()
            pm = get_provider_manager()
            if not session:
                self._send_json(200, {"active": False, "lesson_state": "idle"})
                return

            status_payload = {
                "active": True,
                "lesson_dir": session.lesson_dir,
                "student": session.student,
                "lesson_state": session.state,
                "capture_mode": session.capture_mode,
                "audio_clock_s": round(session.get_audio_clock(), 2),
                "reconnect_counts": pm.reconnect_counts,
                "tracks": {
                    src: {
                        "sample_rate": t.sample_rate,
                        "samples_received": t.samples_received,
                        "last_seq": t.last_seq,
                        "dropped_chunks": t.dropped_chunks,
                        "duplicate_chunks": t.duplicate_chunks,
                    }
                    for src, t in session.tracks.items()
                },
            }
            self._send_json(200, status_payload)
            return

        self._send_json(404, {"error": "Not Found"})

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))

        # 1. Start Lesson
        if self.path == "/lesson/start":
            raw = self.rfile.read(content_length).decode("utf-8")
            try:
                body = json.loads(raw) if raw else {}
            except Exception:
                body = {}

            lesson_dir = body.get("lesson_dir") or f"audio/lesson_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            student = body.get("student", "")
            capture_mode = body.get("capture_mode", "audio_worklet")
            sources = body.get("sources", {"student": 48000, "teacher": 48000})

            session = get_active_session()
            if session and session.state == "recording":
                session.lesson_dir = lesson_dir
                session.student = student
                session.capture_mode = capture_mode
                for src, rate in sources.items():
                    if src not in session.tracks:
                        session.add_track(src, int(rate))
                session.write_manifest()
                pm = get_provider_manager()
                pm.start_streams(sources)
                self._send_json(200, {"ok": True, "lesson_dir": session.lesson_dir, "idempotent": True})
                return

            pm = get_provider_manager()
            pm.reset()

            new_session = ActiveLessonSession(lesson_dir=lesson_dir, student=student, capture_mode=capture_mode)
            for src, rate in sources.items():
                new_session.add_track(src, int(rate))
            set_active_session(new_session)

            pm = get_provider_manager()
            pm.start_streams(sources)

            log_event("lesson_started_ingestion", lesson_dir=lesson_dir, student=student, sources=sources)
            self._send_json(200, {"ok": True, "lesson_dir": lesson_dir})
            return

        # 2. Audio Chunk Ingestion
        if self.path == "/chunk":
            session = get_active_session()
            if not session or session.state != "recording":
                if content_length > 0:
                    self.rfile.read(content_length)
                self._send_json(200, {"ok": False, "ignored": "No active recording session"})
                return

            track_id = self.headers.get("X-Track-Id", "student")
            seq_hdr = self.headers.get("X-Seq")
            rate_hdr = self.headers.get("X-Sample-Rate")

            try:
                seq = int(seq_hdr) if seq_hdr is not None else 0
                sample_rate = int(rate_hdr) if rate_hdr else 48000
            except ValueError:
                if content_length > 0:
                    self.rfile.read(content_length)
                self._send_json(400, {"error": "Invalid headers"})
                return

            chunk_bytes = self.rfile.read(content_length)
            if not chunk_bytes:
                self._send_json(400, {"error": "Empty chunk"})
                return

            # Authoritative write to TrackRecorder path BEFORE acknowledging
            status, ok = session.write_chunk(track_id, seq, chunk_bytes, sample_rate)
            if not ok:
                self._send_json(500, {"error": f"TrackRecorder write failed: {status}", "seq": seq})
                return

            # Dispatch asynchronously to non-blocking provider streamers
            if status != "duplicate" and session.state == "recording":
                get_provider_manager().dispatch_audio(track_id, chunk_bytes, sample_rate)

            self._send_json(200, {
                "ok": True,
                "status": status,
                "seq": seq,
                "track": track_id,
                "samples": len(chunk_bytes) // 2,
            })
            return

        # 3. Pause
        if self.path == "/lesson/pause":
            session = get_active_session()
            if session:
                session.state = "paused"
                session.write_manifest()
            self._send_json(200, {"ok": True, "state": "paused"})
            return

        # 4. Resume
        if self.path == "/lesson/resume":
            session = get_active_session()
            if session:
                session.state = "recording"
                session.write_manifest()
            self._send_json(200, {"ok": True, "state": "recording"})
            return

        # 5. End Lesson
        if self.path == "/lesson/end":
            session = get_active_session()
            if session:
                session.close()
            pm = get_provider_manager()
            pm.stop_all()
            set_active_session(None)
            log_event("lesson_ended_ingestion")
            self._send_json(200, {"ok": True, "state": "ended"})
            return

        self._send_json(404, {"error": "Endpoint not found"})


class AudioIngestionServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


_SERVER_LOCK = threading.Lock()
_SERVER_INSTANCE: Optional[AudioIngestionServer] = None
_SERVER_THREAD: Optional[threading.Thread] = None


def ensure_ingestion_server(host: str = INGESTION_HOST, port: int = INGESTION_PORT) -> AudioIngestionServer:
    """Singleton guard ensuring exactly one background HTTP ingestion daemon runs per process."""
    global _SERVER_INSTANCE, _SERVER_THREAD
    with _SERVER_LOCK:
        if _SERVER_INSTANCE is not None:
            return _SERVER_INSTANCE

        try:
            server = AudioIngestionServer((host, port), AudioIngestionHandler)
            thread = threading.Thread(
                target=server.serve_forever, daemon=True, name="audio-ingestion-daemon"
            )
            thread.start()
            _SERVER_INSTANCE = server
            _SERVER_THREAD = thread
            log_event("ingestion_server_started", host=host, port=port)
            return server
        except Exception as e:
            log_event("ingestion_server_start_error", host=host, port=port, error=str(e))
            raise

def stop_ingestion_server():
    """Cleanly stop the ingestion daemon server if running."""
    global _SERVER_INSTANCE, _SERVER_THREAD
    with _SERVER_LOCK:
        if _SERVER_INSTANCE is not None:
            try:
                _SERVER_INSTANCE.shutdown()
                _SERVER_INSTANCE.server_close()
            except Exception:
                pass
            _SERVER_INSTANCE = None
            _SERVER_THREAD = None
