"""Continuous on-disk recording of a lesson, written as the audio arrives.

The browser hands its own compressed recording over exactly once, inside the
lesson_ended event. Anything that interrupts that single handshake — a closed
tab, a recycled Streamlit session, a browser crash, simply never pressing End —
loses the entire lesson's audio, while the transcript survives because it
autosaves every few seconds.

So the audio is written server-side too, from the PCM already streaming in for
Deepgram. The file exists from the first chunk and grows for the whole lesson.
Ending the lesson cleanly becomes an optimisation rather than the only chance
to save anything.

Ogg/Opus on purpose: a truncated Ogg stream is still playable, so a lesson cut
short by a crash still opens. An M4A killed mid-write has no moov atom and does
not. Opus at 48 kbps mono is about 22 MB per hour per microphone.

The file's timeline is byte-derived from the same PCM stream Deepgram
transcribes, so a position in the audio and a word's timestamp agree exactly —
with or without dropped chunks, since both count the same bytes.
"""

import collections
import os
import shutil
import struct
import subprocess
import threading

FFMPEG = shutil.which("ffmpeg")
OPUS_BITRATE = os.environ.get("ESL_LESSON_AUDIO_BITRATE", "48k")
# A write that has not returned in this long means a wedged encoder; the lesson
# must never be held up by it.
WRITE_STALL_S = 5.0
QUEUE_MAX_CHUNKS = 400


def unique_path(path):
    """A path that does not exist yet, so a restart never truncates a take."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    n = 2
    while os.path.exists(f"{stem}.take{n}{ext}"):
        n += 1
    return f"{stem}.take{n}{ext}"


class TrackRecorder:
    """One microphone's continuous recording.

    PCM is handed over by the script thread and drained by a writer thread, so
    a slow or wedged encoder costs the recording, never the lesson.
    """

    def __init__(self, lesson_dir, source, sample_rate, on_event=None):
        self.source = source
        self.sample_rate = int(sample_rate) if sample_rate else 48000
        self.on_event = on_event or (lambda *a, **k: None)
        self.error = None
        # Queued is what the lesson handed over; written is what has reached the
        # encoder. They differ only by whatever the writer thread has not
        # drained yet, and must be equal once close() returns.
        self.bytes_queued = 0
        self.bytes_written = 0
        self.closed = False
        self.process = None
        self.wav_file = None
        self._lock = threading.Lock()
        self._queue = collections.deque(maxlen=QUEUE_MAX_CHUNKS)
        self._queue_event = threading.Event()
        self._stop = threading.Event()
        self.dropped_chunks = 0

        os.makedirs(lesson_dir, exist_ok=True)
        ext = ".ogg" if FFMPEG else ".wav"
        self.path = unique_path(os.path.join(lesson_dir, f"{source}{ext}"))
        try:
            self._open_sink()
        except Exception as e:
            self.error = str(e)
            self.on_event("lesson_recorder_open_error", source=source, error=self.error)
            return
        self._thread = threading.Thread(
            target=self._writer_loop, daemon=True, name=f"lesson-recorder-{source}"
        )
        self._thread.start()
        self.on_event(
            "lesson_recorder_open",
            source=source, path=self.path, sample_rate=self.sample_rate,
            encoder="ffmpeg-opus" if FFMPEG else "wav",
        )

    # -- sinks ----------------------------------------------------------
    def _open_sink(self):
        if FFMPEG:
            self.process = subprocess.Popen(
                [
                    FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "s16le", "-ar", str(self.sample_rate), "-ac", "1",
                    "-i", "pipe:0",
                    "-c:a", "libopus", "-b:a", OPUS_BITRATE,
                    # Flush often so the file on disk stays current: a crash
                    # should cost seconds of audio, not minutes.
                    "-flush_packets", "1",
                    self.path,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        else:
            # No ffmpeg: a plain WAV, header rewritten as it grows so the file
            # is playable even if the app never gets to close it.
            self.wav_file = open(self.path, "wb")
            self._write_wav_header(0)

    def _write_wav_header(self, data_bytes):
        self.wav_file.seek(0)
        self.wav_file.write(b"RIFF")
        self.wav_file.write(struct.pack("<I", 36 + data_bytes))
        self.wav_file.write(b"WAVEfmt ")
        self.wav_file.write(struct.pack("<IHHIIHH", 16, 1, 1, self.sample_rate,
                                        self.sample_rate * 2, 2, 16))
        self.wav_file.write(b"data")
        self.wav_file.write(struct.pack("<I", data_bytes))
        self.wav_file.seek(0, os.SEEK_END)

    # -- the lesson's side ----------------------------------------------
    def write(self, chunk_bytes):
        """Queue PCM. Never blocks, never raises: this is on the lesson path."""
        if self.closed or self.error or not chunk_bytes:
            return
        with self._lock:
            if len(self._queue) == self._queue.maxlen:
                self.dropped_chunks += 1
                self.bytes_queued -= len(self._queue[0])
            self._queue.append(chunk_bytes)
            self.bytes_queued += len(chunk_bytes)
        self._queue_event.set()

    # -- the writer thread's side ---------------------------------------
    def _writer_loop(self):
        while not self._stop.is_set() or self._queue:
            self._queue_event.wait(0.25)
            self._queue_event.clear()
            while True:
                with self._lock:
                    if not self._queue:
                        break
                    chunk = self._queue.popleft()
                if not self._emit(chunk):
                    return

    def _emit(self, chunk):
        try:
            if self.process is not None:
                self.process.stdin.write(chunk)
            else:
                self.wav_file.write(chunk)
                # Rewriting the header every chunk costs two seeks; cheap, and
                # it keeps a crashed lesson's file valid.
                self._write_wav_header(self.bytes_written + len(chunk))
            self.bytes_written += len(chunk)
            return True
        except Exception as e:
            self.error = str(e)
            self.on_event("lesson_recorder_write_error", source=self.source, error=self.error)
            return False

    def _seconds(self, byte_count):
        return byte_count / float(self.sample_rate * 2)

    @property
    def seconds_written(self):
        """Audio actually in the file."""
        return self._seconds(self.bytes_written)

    @property
    def seconds_captured(self):
        """Audio handed over, including the little still in the queue.

        This is what to show a teacher mid-lesson: the fraction of a second the
        writer thread is behind is not something they should have to think
        about.
        """
        return self._seconds(self.bytes_queued)

    def close(self):
        """Finish the file. Safe to call more than once."""
        if self.closed:
            return
        self.closed = True
        self._stop.set()
        self._queue_event.set()
        thread = getattr(self, "_thread", None)
        if thread is not None:
            thread.join(timeout=WRITE_STALL_S)
        try:
            if self.process is not None:
                try:
                    self.process.stdin.close()
                except Exception:
                    pass
                try:
                    self.process.wait(timeout=WRITE_STALL_S)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            elif self.wav_file is not None:
                self._write_wav_header(self.bytes_written)
                self.wav_file.close()
        except Exception as e:
            self.error = str(e)
        self.on_event(
            "lesson_recorder_close",
            source=self.source, path=self.path,
            bytes=self.bytes_written,
            seconds=round(self.seconds_written, 1),
            bytes_queued=self.bytes_queued,
            dropped_chunks=self.dropped_chunks,
            size_on_disk=os.path.getsize(self.path) if os.path.exists(self.path) else 0,
            error=self.error,
        )
