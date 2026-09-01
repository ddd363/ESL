"""Gladia V2 live streaming client (one websocket per microphone source).

Communicates with Gladia's real-time API (v2/live) using a two-step initialization:
1. HTTP POST to https://api.gladia.io/v2/live with x-gladia-key header to obtain a session WebSocket URL.
2. WebSocket connection for streaming PCM16 chunks and receiving word-level final and interim transcripts.

Safe for Streamlit:
- All socket interactions run on background threads.
- Non-blocking `send_pcm16` queues chunks to a bounded deque.
- Stalled socket detection prevents freezing the app.
- Never logs or exposes credentials.
"""

import collections
import json
import os
import threading
import time
import urllib.request
from time import perf_counter

try:
    import websocket
except ImportError:
    websocket = None

GLADIA_LIVE_INIT_URL = "https://api.gladia.io/v2/live"
GLADIA_DEFAULT_MODEL = "solaria-1"
GLADIA_DEFAULT_LANGUAGE = "en"
GLADIA_SEND_STALL_S = 4.0
GLADIA_INIT_TIMEOUT_S = 6.0


def _noop_log(event, **kwargs):
    pass


class GladiaStreamingClient:
    def __init__(
        self,
        api_key,
        sample_rate,
        model=GLADIA_DEFAULT_MODEL,
        language=GLADIA_DEFAULT_LANGUAGE,
        on_event=None,
    ):
        self.api_key = api_key
        self.sample_rate = int(sample_rate) if sample_rate else 16000
        self.model = model
        self.language = language
        self.on_event = on_event or _noop_log

        self.session_id = None
        self.ws_url = None
        self.ws = None
        self.thread = None
        self.stop_event = threading.Event()
        self.connected_event = threading.Event()
        self.lock = threading.Lock()
        self.started_at = perf_counter()
        self.last_audio_sent_at = self.started_at
        self.disconnected_since = None
        self.final_words = []
        self.interim_text = ""
        self.error = None
        self.seen_utterance_ids = set()

        # Audio waiting to go out via background sender thread
        self.pending_chunks = collections.deque(maxlen=40)
        self.queue_lock = threading.Lock()
        self.queue_event = threading.Event()
        self.sender_thread = None
        self.bytes_enqueued_total = 0
        self.send_in_flight_since = None
        self.bytes_sent_total = 0
        self.chunks_received = 0
        self.transcript_events_received = 0
        self.final_flush_event = threading.Event()

    def _log(self, event, **kwargs):
        # Credentials and tokens must never be logged
        safe_kwargs = {k: v for k, v in kwargs.items() if "key" not in k and "token" not in k}
        try:
            self.on_event(event, **safe_kwargs)
        except Exception:
            pass

    def _init_session(self):
        """HTTP POST to create a Gladia live session and retrieve the WebSocket URL."""
        payload = {
            "encoding": "wav/pcm",
            "sample_rate": self.sample_rate,
            "bit_depth": 16,
            "channels": 1,
            "language_config": {
                "languages": [self.language],
                "code_switching": False,
            },
            "messages_config": {
                "receive_partial_transcripts": True,
                "receive_final_transcripts": True,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            GLADIA_LIVE_INIT_URL,
            data=data,
            headers={
                "x-gladia-key": self.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=GLADIA_INIT_TIMEOUT_S) as resp:
            resp_body = resp.read().decode("utf-8")
            res_json = json.loads(resp_body)
            self.session_id = res_json.get("id")
            self.ws_url = res_json.get("url")
            if not self.ws_url:
                raise RuntimeError("Gladia init returned no WebSocket URL")

    def _on_open(self, _ws):
        with self.lock:
            self.error = None
            self.disconnected_since = None
        self.connected_event.set()
        self._flush_pending_chunks()
        self._log("gladia_stream_open", sample_rate=self.sample_rate)

    def _on_error(self, _ws, err):
        self.error = str(err)
        with self.lock:
            if self.disconnected_since is None:
                self.disconnected_since = perf_counter()
        self.connected_event.clear()
        self._log("gladia_stream_error", error=self.error)

    def _on_close(self, _ws, status_code, message):
        with self.lock:
            if self.disconnected_since is None:
                self.disconnected_since = perf_counter()
        self.connected_event.clear()
        self.stop_event.set()
        self._log("gladia_stream_close", status_code=status_code, message=message)

    def _flush_pending_chunks(self):
        self.queue_event.set()

    def _on_message(self, _ws, message):
        try:
            payload = json.loads(message)
        except Exception:
            return
        if not isinstance(payload, dict):
            return

        msg_type = payload.get("type")
        if msg_type != "transcript":
            return

        self.transcript_events_received += 1
        data = payload.get("data") or {}
        is_final = bool(data.get("is_final"))

        utterance = data.get("utterance") or {}
        transcript = (utterance.get("text") or "").strip()

        if is_final:
            utterance_id = data.get("id") or utterance.get("id")
            with self.lock:
                if utterance_id:
                    if utterance_id in self.seen_utterance_ids:
                        return
                    self.seen_utterance_ids.add(utterance_id)
            self.final_flush_event.set()

            records = []
            words_raw = utterance.get("words") or []
            for item in words_raw:
                if not isinstance(item, dict):
                    continue
                token = (item.get("word") or "").strip()
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
        if not self.api_key:
            self.error = "gladia_api_key_missing"
            self._log("gladia_stream_unavailable", reason=self.error)
            return

        if websocket is None:
            self.error = "websocket_client_missing"
            self._log("gladia_stream_unavailable", reason=self.error)
            return

        def _connect_and_run():
            try:
                self._init_session()
            except Exception as e:
                with self.lock:
                    self.error = f"session_init_error: {e}"
                self._log("gladia_stream_init_error", error=str(e))
                return

            if self.stop_event.is_set():
                return

            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            self.ws.run_forever(ping_interval=15, ping_timeout=8)

        self.thread = threading.Thread(target=_connect_and_run, daemon=True, name="gladia-ws")
        self.thread.start()

        self.sender_thread = threading.Thread(
            target=self._sender_loop, daemon=True, name="gladia-sender"
        )
        self.sender_thread.start()

    @property
    def is_active(self):
        stalled_since = self.send_in_flight_since
        if stalled_since is not None and perf_counter() - stalled_since > GLADIA_SEND_STALL_S:
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
                not self.connected_event.is_set()
                and self.thread is not None
                and self.thread.is_alive()
                and not self.stop_event.is_set()
            )

    def send_pcm16(self, chunk_bytes):
        """Hand audio to sender thread without blocking."""
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
                    self._log("gladia_stream_send_error", error=str(e))
                    break
                self.send_in_flight_since = None
                with self.lock:
                    self.last_audio_sent_at = perf_counter()
                    self.bytes_sent_total += len(chunk_bytes)

    def audio_seconds_sent(self):
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
        try:
            if self.ws and self.connected_event.is_set():
                # Drain queued audio chunks first
                drain_deadline = perf_counter() + 1.5
                while perf_counter() < drain_deadline:
                    with self.queue_lock:
                        if not self.pending_chunks:
                            break
                    time.sleep(0.05)
                try:
                    self.ws.send(json.dumps({"type": "stop_recording"}))
                except Exception:
                    pass
                self.final_flush_event.wait(timeout=3.0)
                try:
                    self.ws.close()
                except Exception:
                    pass
        finally:
            self.stop_event.set()
            self.connected_event.clear()
            self.queue_event.set()
            if self.sender_thread and self.sender_thread.is_alive():
                self.sender_thread.join(timeout=0.5)
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=1.2)
