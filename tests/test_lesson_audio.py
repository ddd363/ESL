"""Guards on the continuous lesson recording.

The property that matters is that audio on disk survives things the lesson
does not: a wedged encoder, a dropped Deepgram socket, the app being killed
outright. A transcript can be regenerated from the audio; the audio cannot be
regenerated from anything.
"""

import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import lesson_audio

RATE = 48000


def tone_chunk(index, seconds=0.5, rate=RATE):
    """PCM16 shaped like the chunks the recorder component sends."""
    count = int(rate * seconds)
    return b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * (index * count + i) / rate)))
        for i in range(count)
    )


def probe(path, entries):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", entries, "-of", "default=nw=1", path],
        capture_output=True, text=True,
    )
    return dict(
        line.split("=", 1) for line in out.stdout.strip().splitlines() if "=" in line
    )


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_writes_a_playable_file_with_the_right_duration(self):
        rec = lesson_audio.TrackRecorder(self.root, "student", RATE)
        for i in range(8):                     # 4 seconds
            rec.write(tone_chunk(i))
        rec.close()
        self.assertIsNone(rec.error)
        self.assertTrue(os.path.getsize(rec.path) > 0)
        if lesson_audio.FFMPEG:
            info = probe(rec.path, "format=duration:stream=codec_name")
            self.assertAlmostEqual(float(info["duration"]), 4.0, delta=0.2)

    def test_duration_tracks_the_bytes_the_transcript_was_built_from(self):
        """Audio position and word timestamp both count the same PCM bytes."""
        rec = lesson_audio.TrackRecorder(self.root, "student", RATE)
        for i in range(6):
            rec.write(tone_chunk(i))
        # Reported live, before the writer thread has necessarily caught up.
        self.assertAlmostEqual(rec.seconds_captured, 3.0, places=3)
        rec.close()
        # Nothing may be lost between the queue and the file.
        self.assertEqual(rec.bytes_written, rec.bytes_queued)
        self.assertAlmostEqual(rec.seconds_written, 3.0, places=3)

    def test_writing_after_close_is_ignored_not_an_error(self):
        rec = lesson_audio.TrackRecorder(self.root, "student", RATE)
        rec.write(tone_chunk(0))
        rec.close()
        rec.write(tone_chunk(1))               # a late chunk from the last rerun
        self.assertIsNone(rec.error)

    def test_close_is_idempotent(self):
        rec = lesson_audio.TrackRecorder(self.root, "student", RATE)
        rec.write(tone_chunk(0))
        rec.close()
        rec.close()
        self.assertTrue(rec.closed)

    def test_a_second_take_never_truncates_the_first(self):
        first = lesson_audio.TrackRecorder(self.root, "student", RATE)
        first.write(tone_chunk(0))
        first.close()
        size = os.path.getsize(first.path)
        second = lesson_audio.TrackRecorder(self.root, "student", RATE)
        second.write(tone_chunk(1))
        second.close()
        self.assertNotEqual(first.path, second.path)
        self.assertEqual(os.path.getsize(first.path), size)

    def test_empty_chunks_are_ignored(self):
        rec = lesson_audio.TrackRecorder(self.root, "student", RATE)
        rec.write(b"")
        rec.write(None)
        rec.close()
        self.assertIsNone(rec.error)

    @unittest.skipUnless(lesson_audio.FFMPEG, "needs ffmpeg")
    def test_file_survives_the_app_being_killed_outright(self):
        """The failure that loses a lesson: no close(), no clean shutdown.

        Ogg is chosen for exactly this — a truncated stream still decodes. An
        M4A killed mid-write has no moov atom and will not open at all.
        """
        script = os.path.join(self.root, "kill_me.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(
                "import os, sys\n"
                f"sys.path.insert(0, {ROOT!r})\n"
                f"sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})\n"
                "from test_lesson_audio import tone_chunk, RATE\n"
                "import lesson_audio\n"
                "rec = lesson_audio.TrackRecorder(sys.argv[1], 'student', RATE)\n"
                "for i in range(8):\n"
                "    rec.write(tone_chunk(i))\n"
                "import time; time.sleep(0.6)\n"
                "os._exit(9)\n"
            )
        out_dir = os.path.join(self.root, "lesson")
        result = subprocess.run([sys.executable, script, out_dir], capture_output=True)
        self.assertEqual(result.returncode, 9, result.stderr[:500])
        time.sleep(0.5)
        path = os.path.join(out_dir, "student.ogg")
        self.assertTrue(os.path.exists(path), "no recording left behind")
        self.assertGreater(os.path.getsize(path), 0)
        # It must not merely exist — it must decode from beginning to end.
        decode = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-f", "null", "-"],
            capture_output=True, text=True,
        )
        self.assertEqual(decode.returncode, 0, decode.stderr[:500])
        self.assertAlmostEqual(
            float(probe(path, "format=duration")["duration"]), 4.0, delta=0.3
        )


if __name__ == "__main__":
    unittest.main()
