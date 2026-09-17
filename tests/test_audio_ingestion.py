import io
import json
import os
import shutil
import tempfile
import time
import unittest

import live_runtime


class TestSessionLogic(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_lesson_")
        self.lesson_dir = os.path.join(self.temp_dir, "lesson_test_123")

    def tearDown(self):
        live_runtime.set_active_session(None)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_active_lesson_session_sequences_and_duplicates(self):
        session = live_runtime.ActiveLessonSession(
            lesson_dir=self.lesson_dir, student="Test Student", capture_mode="audio_worklet"
        )
        live_runtime.set_active_session(session)

        manifest_path = os.path.join(self.lesson_dir, "session_manifest.json")
        self.assertTrue(os.path.exists(manifest_path))

        # 1. First chunk (seq 0)
        pcm = b"\x00\x00" * 4800  # 4800 samples = 0.1s
        status, ok = session.write_chunk("student", 0, pcm, 48000)
        self.assertTrue(ok)
        self.assertEqual(status, "accepted")

        track = session.tracks["student"]
        self.assertEqual(track.samples_received, 4800)
        self.assertEqual(track.last_seq, 0)
        self.assertEqual(track.dropped_chunks, 0)
        self.assertEqual(track.duplicate_chunks, 0)

        # 2. Duplicate chunk (seq 0 again)
        status, ok = session.write_chunk("student", 0, pcm, 48000)
        self.assertTrue(ok)
        self.assertEqual(status, "duplicate")
        # Samples should NOT have doubled
        self.assertEqual(track.samples_received, 4800)
        self.assertEqual(track.duplicate_chunks, 1)

        # 3. Gap detection (seq 3 instead of 1)
        status, ok = session.write_chunk("student", 3, pcm, 48000)
        self.assertTrue(ok)
        self.assertEqual(status, "accepted")
        self.assertEqual(track.samples_received, 9600)
        self.assertEqual(track.dropped_chunks, 2)  # missing 1 and 2

        # 4. Audio clock calculation
        clock = session.get_audio_clock("student")
        self.assertAlmostEqual(clock, 9600 / 48000.0, places=3)

        # 5. Manifest update
        session.write_manifest()
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["tracks"]["student"]["samples_received"], 9600)
        self.assertEqual(manifest["tracks"]["student"]["last_seq"], 3)

        # 6. Close session
        session.close()
        self.assertEqual(session.state, "ended")
        with open(manifest_path, "r", encoding="utf-8") as f:
            final_manifest = json.load(f)
        self.assertEqual(final_manifest["status"], "ended")

    def test_chunk_ignored_without_session(self):
        live_runtime.set_active_session(None)
        handler = live_runtime.AudioIngestionHandler.__new__(live_runtime.AudioIngestionHandler)
        handler.path = "/chunk"
        handler.headers = {"Content-Length": "20", "X-Track-Id": "student", "X-Seq": "1"}
        handler.rfile = io.BytesIO(b"x" * 20)

        sent_responses = []
        handler._send_json = lambda code, payload: sent_responses.append((code, payload))

        handler.do_POST()

        self.assertEqual(len(sent_responses), 1)
        code, payload = sent_responses[0]
        self.assertEqual(code, 200)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["ignored"], "No active recording session")
        # Ensure no zombie session was created!
        self.assertIsNone(live_runtime.get_active_session())


class TestProviderManager(unittest.TestCase):
    def test_provider_manager_lifecycle(self):
        pm = live_runtime.get_provider_manager()
        self.assertIsNotNone(pm)
        self.assertIsNotNone(pm.watchdog_thread)
        self.assertTrue(pm.watchdog_thread.is_alive())

        # Test raw word collection when empty
        dg_words = pm.collect_raw_words("deepgram")
        self.assertEqual(dg_words, [])


if __name__ == "__main__":
    unittest.main()
