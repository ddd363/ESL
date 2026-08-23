"""Guards on turn grouping.

The bug these exist for: both microphones record at once, so the merged word
timeline interleaves simultaneous speech. Breaking a turn on every speaker
change cut a student's sentence into single words each time the teacher said
"Mhmm" — 57% of student turns in a real lesson were one word long.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lesson_turns


def w(source, word, start, end=None):
    return {"source": source, "word": word, "start": start,
            "end": start + 0.3 if end is None else end, "confidence": 0.9}


def texts(turns, source=None):
    return [t["text"] for t in turns if source is None or t["source"] == source]


class InterleavingTests(unittest.TestCase):
    def test_overlapping_speech_does_not_split_a_sentence(self):
        """The real 148-150s stretch: the student said this in 1.6 seconds."""
        words = [
            w("student", "Have", 148.11, 148.43),
            w("teacher", "for", 148.16, 148.40),
            w("teacher", "some", 148.40, 148.64),
            w("student", "you", 148.43, 148.59),
            w("student", "been", 148.59, 148.83),
            w("teacher", "time.", 148.78, 149.10),
            w("student", "to", 148.83, 149.15),
            w("student", "Sri", 149.15, 149.39),
            w("student", "Lanka?", 149.39, 149.71),
        ]
        turns = lesson_turns.group_turns(words)
        self.assertEqual(texts(turns, "student"), ["Have you been to Sri Lanka?"])
        self.assertEqual(texts(turns, "teacher"), ["for some time."])

    def test_a_backchannel_does_not_start_a_new_student_turn(self):
        words = [
            w("student", "I", 10.0),
            w("teacher", "Mhmm.", 10.2),
            w("student", "like", 10.5),
            w("teacher", "Yeah.", 10.8),
            w("student", "it.", 11.0),
        ]
        self.assertEqual(texts(lesson_turns.group_turns(words), "student"), ["I like it."])

    def test_words_are_passed_through_by_reference(self):
        """The renderer maps words to sentences by object identity."""
        first = w("student", "Hello", 1.0)
        turns = lesson_turns.group_turns([first])
        self.assertIs(turns[0]["words"][0], first)


class GapTests(unittest.TestCase):
    def test_a_speakers_own_silence_ends_their_turn(self):
        words = [w("student", "Yes.", 1.0), w("student", "Later.", 20.0)]
        self.assertEqual(texts(lesson_turns.group_turns(words, gap_s=2.0)),
                         ["Yes.", "Later."])

    def test_speech_inside_the_gap_stays_one_turn(self):
        words = [w("student", "Yes", 1.0, 1.3), w("student", "please.", 2.5)]
        self.assertEqual(texts(lesson_turns.group_turns(words, gap_s=2.0)),
                         ["Yes please."])

    def test_the_ceiling_caps_a_monologue(self):
        words = [w("teacher", "word", t / 2.0) for t in range(300)]  # 150s straight
        turns = lesson_turns.group_turns(words, gap_s=2.0, max_s=60.0)
        self.assertGreater(len(turns), 1)
        self.assertTrue(all(t["end"] - t["start"] <= 61.0 for t in turns))


class SlowSpeechTests(unittest.TestCase):
    """A learner hunting for the next word must not lose their sentence."""

    def test_a_long_pause_mid_sentence_keeps_the_turn_open(self):
        words = [
            w("student", "I", 10.0, 10.3),
            w("student", "would", 10.3, 10.6),
            w("student", "usually", 10.6, 11.0),
            w("student", "make", 16.0, 16.4),          # 5s hunting for the word
            w("student", "dinner.", 16.4, 17.0),
        ]
        self.assertEqual(texts(lesson_turns.group_turns(words, gap_s=2.0)),
                         ["I would usually make dinner."])

    def test_the_same_pause_after_a_finished_sentence_does_break(self):
        words = [
            w("student", "I", 10.0, 10.3),
            w("student", "cook.", 10.3, 11.0),
            w("student", "Sometimes.", 16.0, 16.6),
        ]
        self.assertEqual(texts(lesson_turns.group_turns(words, gap_s=2.0)),
                         ["I cook.", "Sometimes."])

    def test_an_abandoned_fragment_does_not_glue_to_the_next_thing(self):
        words = [
            w("student", "It's", 10.0, 10.4),          # trails off, never finished
            w("student", "Actually", 40.0, 40.5),      # 30s later, new thought
            w("student", "yes.", 40.5, 41.0),
        ]
        self.assertEqual(texts(lesson_turns.group_turns(words, gap_s=2.0, hard_gap_s=8.0)),
                         ["It's", "Actually yes."])

    def test_a_question_mark_closes_a_sentence_too(self):
        words = [
            w("student", "Really?", 10.0, 10.5),
            w("student", "Okay.", 16.0, 16.5),
        ]
        self.assertEqual(texts(lesson_turns.group_turns(words, gap_s=2.0)),
                         ["Really?", "Okay."])


class OrderingTests(unittest.TestCase):
    def test_turns_come_back_in_time_order(self):
        words = [
            w("teacher", "First.", 1.0),
            w("student", "Second.", 30.0),
            w("teacher", "Third.", 60.0),
        ]
        turns = lesson_turns.group_turns(words, gap_s=2.0)
        self.assertEqual([t["start"] for t in turns], [1.0, 30.0, 60.0])
        self.assertEqual(texts(turns), ["First.", "Second.", "Third."])

    def test_turn_end_covers_its_longest_word(self):
        words = [w("student", "aaa", 1.0, 9.0), w("student", "b", 1.5, 2.0)]
        self.assertEqual(lesson_turns.group_turns(words)[0]["end"], 9.0)


class DegenerateInputTests(unittest.TestCase):
    def test_no_words(self):
        self.assertEqual(lesson_turns.group_turns([]), [])

    def test_blank_tokens_are_skipped(self):
        words = [w("student", "  ", 1.0), w("student", "Hi.", 1.2)]
        self.assertEqual(texts(lesson_turns.group_turns(words)), ["Hi."])

    def test_missing_timings_do_not_raise(self):
        words = [{"source": "student", "word": "Hi."},
                 {"source": "student", "word": "There.", "start": None, "end": None}]
        self.assertEqual(texts(lesson_turns.group_turns(words)), ["Hi. There."])


if __name__ == "__main__":
    unittest.main()
