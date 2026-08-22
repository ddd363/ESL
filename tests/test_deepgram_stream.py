"""Guards on the audio path's behaviour when the network misbehaves.

`DeepgramStreamingClient` lives in app.py, which cannot be imported - importing
it runs the whole Streamlit script. The class is therefore sliced out of the
source and executed on its own. Ugly, but this is the code most worth pinning
down: a stall here freezes the entire lesson.
"""

import collections
import io
import json
import os
import threading
import time
import unittest
import urllib.parse
from time import perf_counter

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
STALL_S = 4.0


def load_client_class():
    with io.open(APP, encoding="utf-8") as f:
        source = f.read()
    start = source.index("class DeepgramStreamingClient:")
    end = source.index("# Lesson stream management")
    end = source.rindex("# ---", 0, end)
    namespace = {
        "threading": threading, "time": time, "json": json, "collections": collections,
        "os": os, "urllib": urllib, "perf_counter": perf_counter,
        "log_event": lambda *a, **k: None, "websocket": None,
        "DEEPGRAM_SEND_STALL_S": STALL_S, "DEEPGRAM_DEFAULT_MODEL": "nova-3",
        "DEEPGRAM_DEFAULT_LANGUAGE": "en", "DEEPGRAM_KEYTERMS": (),
        "DEEPGRAM_SMART_FORMAT": True,
    }
    exec(source[start:end], namespace)
    return namespace["DeepgramStreamingClient"]


class WedgedSocket:
    """A half-open TCP connection: the write is accepted and never completes."""

    def __init__(self):
        self.entered = threading.Event()

    def send(self, *args, **kwargs):
        self.entered.set()
        time.sleep(3600)


class DeadSocket:
    def send(self, *args, **kwargs):
        raise OSError("connection reset by peer")


class WedgedSocketTests(unittest.TestCase):
    """A stalled socket must cost the stream, never the app.

    websocket send() is a blocking write with no send timeout. Called inline
    from the Streamlit script thread - and worse, under the lock that snapshot()
    needs twice a second - a half-open connection froze the whole app: grey
    page, no transcription, no error, until the OS gave up minutes later.
    """

    @classmethod
    def setUpClass(cls):
        cls.Client = load_client_class()

    def make_wedged(self):
        client = self.Client(api_key="x", sample_rate=16000)
        client.ws = WedgedSocket()
        client.connected_event.set()
        client.sender_thread = threading.Thread(target=client._sender_loop, daemon=True)
        client.sender_thread.start()
        self.addCleanup(client.stop_event.set)
        return client

    def test_send_does_not_block_the_caller(self):
        client = self.make_wedged()
        worst = 0.0
        for _ in range(40):
            t0 = perf_counter()
            client.send_pcm16(b"\x00" * 1600)
            worst = max(worst, perf_counter() - t0)
        self.assertTrue(client.ws.entered.wait(2), "sender never reached the socket")
        self.assertLess(worst, 0.05, "send_pcm16 blocked the script thread")

    def test_snapshot_does_not_block_behind_a_stalled_send(self):
        client = self.make_wedged()
        client.send_pcm16(b"\x00" * 1600)
        self.assertTrue(client.ws.entered.wait(2))
        worst = 0.0
        for _ in range(40):
            t0 = perf_counter()
            client.snapshot()
            worst = max(worst, perf_counter() - t0)
        self.assertLess(worst, 0.05, "rendering blocked on the audio lock")

    def test_a_wedged_socket_eventually_reports_itself_dead(self):
        client = self.make_wedged()
        client.send_pcm16(b"\x00" * 1600)
        self.assertTrue(client.ws.entered.wait(2))
        self.assertTrue(client.is_active)          # still looks connected at first
        time.sleep(STALL_S + 0.3)
        # False is what makes handle_audio_chunk restart the stream.
        self.assertFalse(client.is_active)

    def test_queue_stays_bounded_while_wedged(self):
        client = self.make_wedged()
        for _ in range(500):
            client.send_pcm16(b"\x00" * 1600)
        self.assertLessEqual(len(client.pending_chunks), 40)

    def test_lesson_clock_advances_on_captured_audio(self):
        """The next stream's offset comes from this one, so dropped chunks must
        not rewind the transcript's timestamps."""
        client = self.make_wedged()
        for _ in range(100):
            client.send_pcm16(b"\x00" * 1600)
        self.assertEqual(client.bytes_sent_total, 0)     # nothing reached the wire
        self.assertGreater(client.audio_seconds_sent(), 0)

    def test_send_failure_marks_the_stream_disconnected(self):
        client = self.Client(api_key="x", sample_rate=16000)
        client.ws = DeadSocket()
        client.connected_event.set()
        client.sender_thread = threading.Thread(target=client._sender_loop, daemon=True)
        client.sender_thread.start()
        self.addCleanup(client.stop_event.set)
        client.send_pcm16(b"\x00" * 1600)
        for _ in range(50):
            if not client.connected_event.is_set():
                break
            time.sleep(0.05)
        self.assertFalse(client.connected_event.is_set())
        self.assertIsNotNone(client.error)
        self.assertFalse(client.is_active)


class QueueOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Client = load_client_class()

    def test_send_before_connection_is_queued_not_lost(self):
        client = self.Client(api_key="x", sample_rate=16000)
        client.send_pcm16(b"\x00" * 1600)
        self.assertEqual(len(client.pending_chunks), 1)

    def test_send_after_stop_is_dropped(self):
        client = self.Client(api_key="x", sample_rate=16000)
        client.stop_event.set()
        client.send_pcm16(b"\x00" * 1600)
        self.assertEqual(len(client.pending_chunks), 0)

    def test_empty_chunk_is_ignored(self):
        client = self.Client(api_key="x", sample_rate=16000)
        client.send_pcm16(b"")
        self.assertEqual(len(client.pending_chunks), 0)
