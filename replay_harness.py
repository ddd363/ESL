"""Dev harness: replay a recorded lesson through the real app, no microphones.

The live pane is otherwise only reachable with two mics and a live Deepgram
socket, which makes the hover UI impossible to check while building it. This
seeds session_state from a saved words.json and reveals the words against the
wall clock, then runs app.py unchanged, so what you see is the real renderer.

    conda run -n project1 python -m streamlit run replay_harness.py

Not imported by the app. Delete it freely.
"""

import json
import os
import runpy
import time

import streamlit as st

LESSON = os.environ.get("REPLAY_LESSON", "audio/lesson_20260820_160055/words.json")
SPEED = float(os.environ.get("REPLAY_SPEED", "12"))    # x real time
START_AT = float(os.environ.get("REPLAY_START", "50"))  # skip the silent lead-in


class ReplayWords(list):
    """Words revealed progressively, so the pane churns like a real lesson."""

    def __init__(self, words, t0):
        super().__init__(words)
        self._all = words
        self._t0 = t0

    def _cutoff(self):
        return START_AT + (time.time() - self._t0) * SPEED

    def __iter__(self):
        cutoff = self._cutoff()
        live = [w for w in self._all if (w.get("start") or 0.0) <= cutoff]
        if os.environ.get("REPLAY_FRESH_DICTS") == "1":
            # Mimic the real app: collect_lesson_words() rebuilds a new dict per
            # live word on every render, so object identity never survives a tick.
            live = [dict(w) for w in live]
        return iter(live)

    def __len__(self):
        return sum(1 for _ in self.__iter__())


if "lesson_words" not in st.session_state:
    with open(LESSON, encoding="utf-8") as f:
        words = json.load(f)["words"]
    st.session_state.lesson_words = ReplayWords(words, time.time())
    st.session_state.lesson_state = "recording"
    st.session_state.lesson_dir = os.environ.get("REPLAY_LESSON_DIR") or None
    st.session_state.live_feedback_on = os.environ.get("REPLAY_FEEDBACK_ON") == "1"
    st.session_state.transcript_filter = "Both"

runpy.run_path("app.py", run_name="__main__")
