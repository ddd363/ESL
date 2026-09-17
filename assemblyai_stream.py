"""AssemblyAI Universal-3.5 Pro Realtime live streaming client (one websocket per microphone source).

Communicates with AssemblyAI's real-time streaming API (v3/ws):
1. Connects to wss://streaming.assemblyai.com/v3/ws?sample_rate=16000&speech_model=universal-3-5-pro&mode=balanced
2. Uses raw `Authorization: <api_key>` header (no Bearer prefix).
3. Streams 16-bit PCM binary chunks (~100-250ms) and processes partial and finalized `Turn` events.
4. Converts word timestamps from milliseconds to seconds and captures word_is_final and turn_order metadata.
5. Sends `{"type": "Terminate"}` on clean closure.

Safe for Streamlit:
- All socket interactions run on background threads.
- Non-blocking `send_pcm16` queues chunks to a bounded deque (maxlen=600).
- Stalled socket detection prevents freezing the app.
- Never logs or exposes credentials.
"""

import collections
import json
import os
import threading
import time
import urllib.parse
from time import perf_counter

try:
    import websocket
except ImportError:
    websocket = None

ASSEMBLYAI_WS_BASE_URL = "wss://streaming.assemblyai.com/v3/ws"
ASSEMBLYAI_DEFAULT_MODEL = "universal-3-5-pro"
ASSEMBLYAI_DEFAULT_MODE = "balanced"
ASSEMBLYAI_SEND_STALL_S = 4.0


def _noop_log(event, **kwargs):
    pass


def _get_sslopt():
    sslopt = {}
    try:
        import certifi
        import ssl
        cafile = certifi.where()
        sslopt["ca_certs"] = cafile
        sslopt["context"] = ssl.create_default_context(cafile=cafile)
    except Exception:
        pass
    return sslopt


class AssemblyAIStreamingClient:
    def __init__(
        self,
        api_key,
        sample_rate=16000,
        model=ASSEMBLYAI_DEFAULT_MODEL,
        mode=ASSEMBLYAI_DEFAULT_MODE,
        on_event=None,
    ):
        self.api_key = api_key
        self.sample_rate = int(sample_rate) if sample_rate else 16000
        self.model = model
        self.mode = mode
        self.on_event = on_event or _noop_log

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
        self.seen_turn_keys = set()

        # Audio waiting to go out via background sender thread
        self.pending_chunks = collections.deque(maxlen=600)
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
        safe_kwargs = {k: v for k, v in kwargs.items() if "key" not in k and "token" not in k}
        try:
            self.on_event(event, **safe_kwargs)
        except Exception:
            pass

    def _build_ws_url(self):
        params = [
            ("sample_rate", str(self.sample_rate)),
            ("speech_model", str(self.model)),
            ("mode", str(self.mode)),
        ]
        return f"{ASSEMBLYAI_WS_BASE_URL}?{urllib.parse.urlencode(params)}"

    def _on_open(self, _ws):
        with self.lock:
            self.error = None
            self.disconnected_since = None
        self.connected_event.set()
        self._flush_pending_chunks()
        self._log("assemblyai_stream_open", sample_rate=self.sample_rate, model=self.model)

    def _on_error(self, _ws, err):
        self.error = str(err)
        with self.lock:
            if self.disconnected_since is None:
                self.disconnected_since = perf_counter()
        self.connected_event.clear()
        self._log("assemblyai_stream_error", error=self.error)

    def _on_close(self, _ws, status_code, message):
        with self.lock:
            if self.disconnected_since is None:
                self.disconnected_since = perf_counter()
        self.connected_event.clear()
        self.stop_event.set()
        self._log("assemblyai_stream_close", status_code=status_code, message=message)

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
        if msg_type != "Turn":
            return

        self.transcript_events_received += 1
        end_of_turn = bool(payload.get("end_of_turn"))
        transcript = (payload.get("transcript") or "").strip()
        turn_order = payload.get("turn_order")
        words_raw = payload.get("words") or []

        if end_of_turn:
            self.final_flush_event.set()
            records = []
            for item in words_raw:
                if not isinstance(item, dict):
                    continue
                token = (item.get("text") or item.get("word") or "").strip()
                if not token:
                    continue
                start_ms = item.get("start")
                end_ms = item.get("end")
                start_s = (start_ms / 1000.0) if isinstance(start_ms, (int, float)) else 0.0
                end_s = (end_ms / 1000.0) if isinstance(end_ms, (int, float)) else start_s
                records.append({
                    "word": token,
                    "start": start_s,
                    "end": end_s,
                    "confidence": item.get("confidence"),
                    "is_final": True,
                    "word_is_final": bool(item.get("word_is_final", True)),
                    "turn_order": turn_order,
                    "provider": "assemblyai",
                })
            with self.lock:
                if records:
                    first_t = round(records[0]["start"], 3)
                    last_t = round(records[-1]["end"], 3)
                    turn_key = (turn_order, first_t, last_t, len(records))
                    if turn_key in self.seen_turn_keys:
                        return
                    self.seen_turn_keys.add(turn_key)
                    self.final_words.extend(records)
                self.interim_text = ""
        elif transcript:
            with self.lock:
                self.interim_text = transcript

    def start(self):
        if not self.api_key:
            raise ValueError("ASSEMBLYAI_API_KEY is not set.")
        if websocket is None:
            raise RuntimeError("websocket-client library is not installed.")

        ws_url = self._build_ws_url()
        headers = [f"Authorization: {self.api_key}"]

        self.ws = websocket.WebSocketApp(
            ws_url,
            header=headers,
            on_open=self._on_open,
            on_error=self._on_error,
            on_close=self._on_close,
            on_message=self._on_message,
        )

        sslopt = _get_sslopt()
        self.thread = threading.Thread(
            target=lambda: self.ws.run_forever(sslopt=sslopt),
            daemon=True,
            name=f"assemblyai-client-{self.sample_rate}",
        )
        self.thread.start()

        self.sender_thread = threading.Thread(
            target=self._sender_loop,
            daemon=True,
            name=f"assemblyai-sender-{self.sample_rate}",
        )
        self.sender_thread.start()

    def send_pcm16(self, chunk_bytes):
        """Queue raw PCM16 bytes for background sending without blocking."""
        if not chunk_bytes or self.stop_event.is_set():
            return
        with self.queue_lock:
            self.pending_chunks.append(chunk_bytes)
            self.bytes_enqueued_total += len(chunk_bytes)
        self.queue_event.set()

    def _sender_loop(self):
        while not self.stop_event.is_set():
            self.queue_event.wait(timeout=0.1)
            self.queue_event.clear()

            while not self.stop_event.is_set():
                chunk_bytes = None
                with self.queue_lock:
                    if self.pending_chunks:
                        chunk_bytes = self.pending_chunks.popleft()
                if chunk_bytes is None:
                    break

                if not self.connected_event.wait(timeout=0.2):
                    continue

                ws_app = self.ws
                if not ws_app:
                    break

                try:
                    self.send_in_flight_since = perf_counter()
                    ws_app.send(chunk_bytes, opcode=websocket.ABNF.OPCODE_BINARY)
                    self.send_in_flight_since = None
                except Exception as e:
                    self.send_in_flight_since = None
                    with self.lock:
                        self.error = str(e)
                        if self.disconnected_since is None:
                            self.disconnected_since = perf_counter()
                    self.connected_event.clear()
                    self._log("assemblyai_stream_send_error", error=str(e))
                    break

                with self.lock:
                    self.last_audio_sent_at = perf_counter()
                    self.bytes_sent_total += len(chunk_bytes)

    @property
    def is_active(self):
        with self.lock:
            if not self.connected_event.is_set() or self.error or self.stop_event.is_set():
                return False
            if self.send_in_flight_since is not None:
                if perf_counter() - self.send_in_flight_since > ASSEMBLYAI_SEND_STALL_S:
                    return False
            return True

    @property
    def is_connecting(self):
        with self.lock:
            if self.stop_event.is_set() or self.error:
                return False
            return not self.connected_event.is_set() and (perf_counter() - self.started_at < 5.0)

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
        self.stop_event.set()
        self.connected_event.clear()
        try:
            if self.ws:
                try:
                    self.ws.send(json.dumps({"type": "Terminate"}))
                except Exception:
                    pass
                self.final_flush_event.wait(timeout=2.0)
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
            self._log("assemblyai_stream_stopped", bytes_sent_total=self.bytes_sent_total)
