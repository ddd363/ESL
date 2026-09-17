import json
import os
import shutil
import tempfile
import time
import unittest

import live_runtime
from deepgram_stream import DeepgramStreamingClient
from gladia_stream import GladiaStreamingClient


class TestLiveRecordingChaos(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="chaos_test_")
        self.lesson_dir = os.path.join(self.temp_dir, "lesson_chaos_001")
        live_runtime.set_active_session(None)

    def tearDown(self):
        live_runtime.set_active_session(None)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_end_to_end_chaos_scenario(self):
        # 1. Start active session
        session = live_runtime.ActiveLessonSession(
            lesson_dir=self.lesson_dir, student="Hannes", capture_mode="audio_worklet"
        )
        live_runtime.set_active_session(session)
        pm = live_runtime.get_provider_manager()

        manifest_path = os.path.join(self.lesson_dir, "session_manifest.json")
        self.assertTrue(os.path.exists(manifest_path))

        # 2. Continuous multi-track ingestion (100 chunks per track)
        # 0.1s chunk @ 48kHz = 4800 samples = 9600 bytes
        chunk_pcm = b"\x01\x00" * 4800
        for seq in range(50):
            status_s, ok_s = session.write_chunk("student", seq, chunk_pcm, 48000)
            status_t, ok_t = session.write_chunk("teacher", seq, chunk_pcm, 48000)
            self.assertTrue(ok_s)
            self.assertTrue(ok_t)
            self.assertEqual(status_s, "accepted")
            self.assertEqual(status_t, "accepted")

        student_track = session.tracks["student"]
        teacher_track = session.tracks["teacher"]
        self.assertEqual(student_track.samples_received, 4800 * 50)
        self.assertEqual(teacher_track.samples_received, 4800 * 50)
        self.assertEqual(student_track.last_seq, 49)
        self.assertEqual(teacher_track.last_seq, 49)

        # 3. Simulate Provider WebSocket Drop and Recovery with Pre-buffer Replay
        mock_dg = DeepgramStreamingClient(api_key="mock_key", sample_rate=48000)
        # Populate mock final words before drop
        mock_dg.final_words = [
            {"word": "Hello", "start": 0.5, "end": 0.8, "confidence": 0.99},
            {"word": "teacher", "start": 0.9, "end": 1.2, "confidence": 0.98},
        ]
        pm.deepgram_streamers["student"] = mock_dg
        # Simulate socket error / disconnect
        mock_dg.error = "Connection reset by peer"
        mock_dg.connected_event.clear()

        # Check that TrackRecorder audio persistence was NEVER affected by the provider error
        self.assertFalse(student_track.recorder.closed)
        self.assertIsNone(student_track.recorder.error)

        # Trigger watchdog recovery
        pm._recover_streamer("deepgram", "student", student_track, "mock_key")
        self.assertEqual(pm.reconnect_counts["deepgram"], 1)

        # Verify previous words were retired with correct offsets
        retired = pm.retired_words_deepgram
        self.assertEqual(len(retired), 2)
        self.assertEqual(retired[0]["word"], "Hello")
        self.assertEqual(retired[1]["word"], "teacher")

        # 4. Network Chaos: Duplicates and Gaps
        # Duplicate seq 40 (already processed)
        status, ok = session.write_chunk("student", 40, chunk_pcm, 48000)
        self.assertTrue(ok)
        self.assertEqual(status, "duplicate")
        self.assertEqual(student_track.duplicate_chunks, 1)
        # Samples should not have increased
        self.assertEqual(student_track.samples_received, 4800 * 50)

        # Gap: Send seq 55 (skipping 50, 51, 52, 53, 54 = 5 dropped)
        status, ok = session.write_chunk("student", 55, chunk_pcm, 48000)
        self.assertTrue(ok)
        self.assertEqual(status, "accepted")
        self.assertEqual(student_track.dropped_chunks, 5)

        # 5. Simulate Streamlit UI Session Reconnection Mid-Lesson
        # When Streamlit re-executes, it accesses get_active_session()
        active = live_runtime.get_active_session()
        self.assertIsNotNone(active)
        self.assertEqual(active.lesson_dir, self.lesson_dir)
        self.assertEqual(active.student, "Hannes")
        self.assertEqual(active.state, "recording")

        # 6. Pause & Resume Semantics
        session.state = "paused"
        session.write_manifest()
        status, ok = session.write_chunk("student", 56, chunk_pcm, 48000)
        self.assertEqual(status, "paused_or_ended")

        session.state = "recording"
        session.write_manifest()
        status, ok = session.write_chunk("student", 56, chunk_pcm, 48000)
        self.assertEqual(status, "accepted")

        # 7. Finalize Lesson
        session.close()
        pm.stop_all()
        self.assertEqual(session.state, "ended")

        # Verify final session_manifest.json on disk
        with open(manifest_path, "r", encoding="utf-8") as f:
            final_manifest = json.load(f)
        self.assertEqual(final_manifest["status"], "ended")
        self.assertEqual(final_manifest["student"], "Hannes")
        self.assertGreater(final_manifest["tracks"]["student"]["samples_received"], 0)

        # Verify on-disk audio recording files exist
        student_ogg = student_track.recorder.path
        teacher_ogg = teacher_track.recorder.path
        self.assertTrue(os.path.exists(student_ogg))
        self.assertTrue(os.path.exists(teacher_ogg))
        self.assertGreater(os.path.getsize(student_ogg), 0)
        self.assertGreater(os.path.getsize(teacher_ogg), 0)


if __name__ == "__main__":
    unittest.main()
