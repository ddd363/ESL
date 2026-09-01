"""Unit tests for GladiaStreamingClient.

Tests message parsing, non-blocking queueing, thread safety, and error handling.
"""

import json
import threading
import time
import unittest
from time import perf_counter

from gladia_stream import GladiaStreamingClient, GLADIA_SEND_STALL_S


class FakeSocket:
    def __init__(self):
        self.sent_messages = []
        self.closed = False

    def send(self, data, opcode=None):
        self.sent_messages.append((data, opcode))

    def close(self):
        self.closed = True


class WedgedSocket:
    """A socket where writes stall indefinitely."""

    def __init__(self):
        self.entered = threading.Event()

    def send(self, *args, **kwargs):
        self.entered.set()
        time.sleep(3600)

    def close(self):
        pass


class GladiaStreamingClientTests(unittest.TestCase):
    def test_missing_api_key_sets_error_and_does_not_crash(self):
        client = GladiaStreamingClient(api_key="", sample_rate=16000)
        client.start()
        self.assertEqual(client.error, "gladia_api_key_missing")
        self.assertFalse(client.is_active)

    def test_partial_and_final_message_parsing(self):
        client = GladiaStreamingClient(api_key="test_key", sample_rate=16000)

        # 1. Partial transcript
        partial_msg = json.dumps({
            "type": "transcript",
            "data": {
                "is_final": False,
                "utterance": {
                    "text": "I am thinking",
                },
            },
        })
        client._on_message(None, partial_msg)
        snap1 = client.snapshot()
        self.assertEqual(snap1["interim"], "I am thinking")
        self.assertEqual(snap1["final_words"], [])

        # 2. Final transcript
        final_msg = json.dumps({
            "type": "transcript",
            "data": {
                "is_final": True,
                "utterance": {
                    "text": "I am thinking about it.",
                    "start": 0.5,
                    "end": 2.0,
                    "confidence": 0.94,
                    "words": [
                        {"word": "I", "start": 0.5, "end": 0.7, "confidence": 0.98},
                        {"word": "am", "start": 0.7, "end": 0.9, "confidence": 0.95},
                        {"word": "thinking", "start": 0.9, "end": 1.3, "confidence": 0.92},
                        {"word": "about", "start": 1.3, "end": 1.6, "confidence": 0.90},
                        {"word": "it.", "start": 1.6, "end": 1.9, "confidence": 0.96},
                    ],
                },
            },
        })
        client._on_message(None, final_msg)
        snap2 = client.snapshot()
        self.assertEqual(snap2["interim"], "")  # cleared on final
        self.assertEqual(len(snap2["final_words"]), 5)
        self.assertEqual(snap2["final_words"][2]["word"], "thinking")
        self.assertEqual(snap2["final_words"][2]["start"], 0.9)
        self.assertEqual(snap2["final_words"][2]["end"], 1.3)
        self.assertEqual(snap2["final_words"][2]["confidence"], 0.92)

    def test_send_pcm16_queues_without_blocking(self):
        client = GladiaStreamingClient(api_key="test_key", sample_rate=16000)
        chunk = b"\x00" * 3200  # 100ms at 16kHz 16-bit
        start_t = perf_counter()
        for _ in range(5):
            client.send_pcm16(chunk)
        elapsed = perf_counter() - start_t
        self.assertLess(elapsed, 0.05)  # Enqueuing must be nearly instant
        self.assertEqual(len(client.pending_chunks), 5)
        self.assertEqual(client.bytes_enqueued_total, 3200 * 5)

    def test_audio_seconds_sent_calculation(self):
        client = GladiaStreamingClient(api_key="test_key", sample_rate=16000)
        # 16000 samples/sec * 2 bytes/sample = 32000 bytes/sec
        chunk = b"\x00" * 32000  # exactly 1 second
        client.send_pcm16(chunk)
        self.assertAlmostEqual(client.audio_seconds_sent(), 1.0, places=2)

    def test_final_utterance_deduplication(self):
        client = GladiaStreamingClient(api_key="test_key", sample_rate=16000)
        final_msg = json.dumps({
            "type": "transcript",
            "data": {
                "id": "utt_12345",
                "is_final": True,
                "utterance": {
                    "text": "Hello world.",
                    "words": [
                        {"word": "Hello", "start": 0.1, "end": 0.4, "confidence": 0.95},
                        {"word": "world.", "start": 0.4, "end": 0.8, "confidence": 0.95},
                    ],
                },
            },
        })
        # Send once
        client._on_message(None, final_msg)
        self.assertEqual(len(client.final_words), 2)
        # Send duplicate replayed final message
        client._on_message(None, final_msg)
        # Word count must remain 2, not 4
        self.assertEqual(len(client.final_words), 2)

    def test_close_drains_and_sends_stop_recording(self):
        client = GladiaStreamingClient(api_key="test_key", sample_rate=16000)
        fake_ws = FakeSocket()
        client.ws = fake_ws
        client.connected_event.set()
        client.close()
        # Ensure stop_recording was sent
        sent_types = [json.loads(m[0]).get("type") for m in fake_ws.sent_messages if m[1] is None or m[1] == 1 or isinstance(m[0], str)]
        self.assertIn("stop_recording", sent_types)
        self.assertTrue(fake_ws.closed)


    def test_wedged_socket_detection(self):
        client = GladiaStreamingClient(api_key="test_key", sample_rate=16000)
        client.ws = WedgedSocket()
        client.connected_event.set()

        # Simulate write that started long ago
        client.send_in_flight_since = perf_counter() - (GLADIA_SEND_STALL_S + 1.0)
        # is_active must detect stall and return False
        self.assertFalse(client.is_active)


if __name__ == "__main__":
    unittest.main()
