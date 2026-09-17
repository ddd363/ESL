"""Deepgram live streaming client (one websocket per microphone source).

Communicates with Deepgram's streaming API (v1/listen) over WebSocket.
Safe for background threads and decoupled from Streamlit UI.
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

DEEPGRAM_DEFAULT_MODEL = os.environ.get("DEEPGRAM_MODEL", "nova-3")
DEEPGRAM_DEFAULT_LANGUAGE = os.environ.get("DEEPGRAM_LANGUAGE", "en")
DEEPGRAM_KEYTERMS = tuple(
    term.strip() for term in os.environ.get("DEEPGRAM_KEYTERMS", "").split(",") if term.strip()
)
DEEPGRAM_SMART_FORMAT = os.environ.get("DEEPGRAM_SMART_FORMAT", "false").lower() == "true"
DEEPGRAM_FILLER_WORDS = os.environ.get("DEEPGRAM_FILLER_WORDS", "true").lower() == "true"
DEEPGRAM_ENDPOINTING_MS = int(os.environ.get("DEEPGRAM_ENDPOINTING_MS", "400"))
DEEPGRAM_SEND_STALL_S = 4.0


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


class DeepgramStreamingClient:
    def __init__(
        self,
        api_key,
        sample_rate,
        model=DEEPGRAM_DEFAULT_MODEL,
        language=DEEPGRAM_DEFAULT_LANGUAGE,
        keyterms=DEEPGRAM_KEYTERMS,
        smart_format=DEEPGRAM_SMART_FORMAT,
        filler_words=DEEPGRAM_FILLER_WORDS,
        endpointing=DEEPGRAM_ENDPOINTING_MS,
        on_event=None,
    ):
        self.api_key = api_key
        self.sample_rate = int(sample_rate) if sample_rate else 16000
        self.model = model
        self.language = language
        self.keyterms = tuple(keyterms)
        self.smart_format = bool(smart_format)
        self.filler_words = bool(filler_words)
        self.endpointing = int(endpointing)
        self.on_event = on_event or _noop_log
        self.ws = None
        self.thread = None
        self.keepalive_thread = None
        self.stop_event = threading.Event()
        self.connected_event = threading.Event()
        self.lock = threading.Lock()
        self.pending_chunks = collections.deque(maxlen=600)
        self.queue_lock = threading.Lock()
        self.queue_event = threading.Event()
        self.started_at = perf_counter()
        self.last_audio_sent_at = self.started_at
        self.last_result_received_at = self.started_at
        self.disconnected_since = None
        self.final_words = []
        self.interim_text = ""
        self.error = None
        self.send_in_flight_since = None
        self.bytes_sent_total = 0
        self.bytes_enqueued_total = 0
        self.chunks_received = 0
        self.sender_thread = None
        self.transcript_events_received = 0
        self.final_flush_event = threading.Event()

    def _log(self, event, **kwargs):
        safe_kwargs = {k: v for k, v in kwargs.items() if "key" not in k and "token" not in k}
        try:
            self.on_event(event, **safe_kwargs)
        except Exception:
            pass

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
            ("filler_words", str(self.filler_words).lower()),
            ("endpointing", str(self.endpointing)),
        ]
        params.extend(("keyterm", term) for term in self.keyterms)
        return "wss://api.deepgram.com/v1/listen?" + urllib.parse.urlencode(params)

    def _on_open(self, _ws):
        with self.lock:
            self.error = None
            self.disconnected_since = None
        self.connected_event.set()
        self._flush_pending_chunks()
        self._log("deepgram_stream_open", sample_rate=self.sample_rate, model=self.model)

    def _on_error(self, _ws, err):
        self.error = str(err)
        with self.lock:
            if self.disconnected_since is None:
                self.disconnected_since = perf_counter()
        self.connected_event.clear()
        self._log("deepgram_stream_error", error=self.error)

    def _on_close(self, _ws, status_code, message):
        with self.lock:
            if self.disconnected_since is None:
                self.disconnected_since = perf_counter()
        self.connected_event.clear()
        self.stop_event.set()
        self._log("deepgram_stream_close", status_code=status_code, message=message)

    def _flush_pending_chunks(self):
        self.queue_event.set()

    def _send_keepalive_if_idle(self, now):
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
            self._log("deepgram_keepalive_error", error=self.error)

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
        with self.lock:
            self.last_result_received_at = perf_counter()
        is_final = bool(payload.get("is_final"))
        speech_final = bool(payload.get("speech_final"))
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
                    "is_final": True,
                    "speech_final": speech_final,
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
            self.error = "deepgram_api_key_missing"
            self._log("deepgram_stream_unavailable", reason=self.error)
            return
        if websocket is None:
            self.error = "websocket_client_missing"
            self._log("deepgram_stream_unavailable", reason=self.error)
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
        sslopt = _get_sslopt()
        self.thread = threading.Thread(
            target=lambda: ws_app.run_forever(ping_interval=15, ping_timeout=8, sslopt=sslopt),
            daemon=True,
            name="deepgram-ws",
        )
        self.thread.start()
        self.sender_thread = threading.Thread(
            target=self._sender_loop, daemon=True, name="deepgram-sender"
        )
        self.sender_thread.start()
        self._start_keepalive()

    @property
    def is_active(self):
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
                    self._log("deepgram_stream_send_error", error=str(e))
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
                drain_deadline = perf_counter() + 1.5
                while perf_counter() < drain_deadline:
                    with self.queue_lock:
                        if not self.pending_chunks:
                            break
                    time.sleep(0.05)
                try:
                    self.ws.send(json.dumps({"type": "CloseStream"}))
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
                self.thread.join(timeout=0.5)
            if self.keepalive_thread and self.keepalive_thread.is_alive():
                self.keepalive_thread.join(timeout=0.5)
