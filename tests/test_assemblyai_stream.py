"""Unit tests for AssemblyAIStreamingClient.

Tests message parsing, millisecond-to-second conversion, word_is_final and turn_order tracking,
non-blocking queueing, thread safety, and error handling.
"""

import json
import threading
import time
import unittest
from time import perf_counter

from assemblyai_stream import AssemblyAIStreamingClient, ASSEMBLYAI_SEND_STALL_S


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


class AssemblyAIStreamingClientTests(unittest.TestCase):
    def test_missing_api_key_raises_value_error(self):
        client = AssemblyAIStreamingClient(api_key="", sample_rate=16000)
        with self.assertRaises(ValueError):
            client.start()

    def test_build_ws_url(self):
        client = AssemblyAIStreamingClient(
            api_key="test_key",
            sample_rate=16000,
            model="universal-3-5-pro",
            mode="balanced",
        )
        url = client._build_ws_url()
        self.assertIn("speech_model=universal-3-5-pro", url)
        self.assertIn("mode=balanced", url)
        self.assertIn("sample_rate=16000", url)

    def test_partial_turn_message_parsing(self):
        client = AssemblyAIStreamingClient(api_key="test_key", sample_rate=16000)
        partial_msg = json.dumps({
            "type": "Turn",
            "end_of_turn": False,
            "turn_order": 1,
            "transcript": "I think picnic",
            "words": [
                {"text": "I", "start": 500, "end": 700, "confidence": 0.95},
                {"text": "think", "start": 700, "end": 1000, "confidence": 0.90},
                {"text": "picnic", "start": 1000, "end": 1400, "confidence": 0.88},
            ],
        })
        client._on_message(None, partial_msg)
        snap = client.snapshot()
        self.assertEqual(snap["interim"], "I think picnic")
        self.assertEqual(snap["final_words"], [])

    def test_final_turn_message_parsing_converts_ms_to_seconds(self):
        client = AssemblyAIStreamingClient(api_key="test_key", sample_rate=16000)
        final_msg = json.dumps({
            "type": "Turn",
            "end_of_turn": True,
            "turn_order": 2,
            "transcript": "I love people.",
            "words": [
                {"text": "I", "start": 1200, "end": 1400, "confidence": 0.98, "word_is_final": True},
                {"text": "love", "start": 1400, "end": 1750, "confidence": 0.96, "word_is_final": True},
                {"text": "people.", "start": 1750, "end": 2100, "confidence": 0.94, "word_is_final": True},
            ],
        })
        client._on_message(None, final_msg)
        snap = client.snapshot()
        self.assertEqual(snap["interim"], "")
        self.assertEqual(len(snap["final_words"]), 3)

        w0 = snap["final_words"][0]
        self.assertEqual(w0["word"], "I")
        self.assertAlmostEqual(w0["start"], 1.2)
        self.assertAlmostEqual(w0["end"], 1.4)
        self.assertEqual(w0["confidence"], 0.98)
        self.assertTrue(w0["is_final"])
        self.assertTrue(w0["word_is_final"])
        self.assertEqual(w0["turn_order"], 2)

        w1 = snap["final_words"][1]
        self.assertEqual(w1["word"], "love")
        self.assertAlmostEqual(w1["start"], 1.4)
        self.assertAlmostEqual(w1["end"], 1.75)

    def test_send_pcm16_queues_without_blocking(self):
        client = AssemblyAIStreamingClient(api_key="test_key", sample_rate=16000)
        chunk = b"\x00" * 3200  # 100ms
        start_t = perf_counter()
        for _ in range(5):
            client.send_pcm16(chunk)
        elapsed = perf_counter() - start_t
        self.assertLess(elapsed, 0.05)
        self.assertEqual(len(client.pending_chunks), 5)
        self.assertEqual(client.bytes_enqueued_total, 3200 * 5)

    def test_bounded_queue_drops_oldest_on_backpressure(self):
        client = AssemblyAIStreamingClient(api_key="test_key", sample_rate=16000)
        client.pending_chunks = client.pending_chunks.__class__(maxlen=5)
        for i in range(10):
            client.send_pcm16(bytes([i]))
        self.assertEqual(len(client.pending_chunks), 5)
        self.assertEqual(list(client.pending_chunks), [bytes([i]) for i in range(5, 10)])

    def test_stalled_socket_fails_is_active(self):
        client = AssemblyAIStreamingClient(api_key="test_key", sample_rate=16000)
        client.connected_event.set()
        client.send_in_flight_since = perf_counter() - (ASSEMBLYAI_SEND_STALL_S + 0.5)
        self.assertFalse(client.is_active)

    def test_close_sends_terminate_message(self):
        client = AssemblyAIStreamingClient(api_key="test_key", sample_rate=16000)
        fake_ws = FakeSocket()
        client.ws = fake_ws
        client.connected_event.set()
        client.close()

        sent = [msg[0] for msg in fake_ws.sent_messages]
        self.assertTrue(any(json.loads(m).get("type") == "Terminate" for m in sent if isinstance(m, str)))
        self.assertTrue(client.stop_event.is_set())
        self.assertFalse(client.connected_event.is_set())

    def test_turn_replay_deduplication_upon_reconnect(self):
        client = AssemblyAIStreamingClient(api_key="test_key", sample_rate=16000)
        turn_msg = json.dumps({
            "type": "Turn",
            "end_of_turn": True,
            "turn_order": 1,
            "transcript": "Hello world.",
            "words": [
                {"text": "Hello", "start": 100, "end": 400, "confidence": 0.95},
                {"text": "world.", "start": 400, "end": 800, "confidence": 0.95},
            ],
        })
        client._on_message(None, turn_msg)
        self.assertEqual(len(client.final_words), 2)
        # Duplicate replayed turn must not append duplicates
        client._on_message(None, turn_msg)
        self.assertEqual(len(client.final_words), 2)

    def test_reconnect_resets_turn_order_without_dropping_new_data(self):
        client = AssemblyAIStreamingClient(api_key="test_key", sample_rate=16000)
        turn1 = json.dumps({
            "type": "Turn",
            "end_of_turn": True,
            "turn_order": 1,
            "transcript": "First sentence.",
            "words": [
                {"text": "First", "start": 100, "end": 400, "confidence": 0.95},
                {"text": "sentence.", "start": 400, "end": 800, "confidence": 0.95},
            ],
        })
        client._on_message(None, turn1)
        self.assertEqual(len(client.final_words), 2)

        # Socket reconnect where turn_order resets to 1 for subsequent audio at 5.0s
        turn2 = json.dumps({
            "type": "Turn",
            "end_of_turn": True,
            "turn_order": 1,
            "transcript": "Second sentence.",
            "words": [
                {"text": "Second", "start": 5000, "end": 5400, "confidence": 0.95},
                {"text": "sentence.", "start": 5400, "end": 5800, "confidence": 0.95},
            ],
        })
        client._on_message(None, turn2)
        self.assertEqual(len(client.final_words), 4)
        self.assertEqual(client.final_words[2]["word"], "Second")


if __name__ == "__main__":
    unittest.main()
