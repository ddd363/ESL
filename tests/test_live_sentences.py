import json
import os
import unittest

from live_sentences import (
    ends_sentence,
    in_feedback_window,
    index_words_to_sentences,
    low_confidence_words,
    newest_word_end,
    settled_sentences,
    split_sentences,
    update_feedback_windows,
)

REAL_LESSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "audio", "lesson_20260820_160055", "words.json",
)


def word(token, start, end=None, source="student", confidence=0.95):
    return {
        "source": source,
        "word": token,
        "start": start,
        "end": start + 0.3 if end is None else end,
        "confidence": confidence,
    }


def utterance(tokens, start=0.0, step=0.4, source="student"):
    return [word(t, start + i * step, source=source) for i, t in enumerate(tokens)]


class EndsSentenceTests(unittest.TestCase):
    def test_terminal_punctuation_ends_a_sentence(self):
        for token in ("done.", "really?", "stop!", "well…"):
            self.assertTrue(ends_sentence(token), token)

    def test_plain_word_does_not(self):
        self.assertFalse(ends_sentence("because"))

    def test_abbreviation_does_not(self):
        for token in ("Mr.", "etc.", "e.g.", "vs."):
            self.assertFalse(ends_sentence(token), token)

    def test_bare_initial_does_not(self):
        # "B." while spelling a word out; real in the lesson transcripts.
        self.assertFalse(ends_sentence("B."))
        self.assertFalse(ends_sentence("D."))

    def test_decimal_number_does_not(self):
        self.assertFalse(ends_sentence("3.5"))


class SplitSentencesTests(unittest.TestCase):
    def test_splits_on_terminal_punctuation(self):
        words = utterance(["I", "am", "late.", "You", "are", "early."])
        texts = [s["text"] for s in split_sentences(words, "student")]
        self.assertEqual(texts, ["I am late.", "You are early."])

    def test_splits_on_a_silence_gap(self):
        words = utterance(["I", "went", "there"]) + utterance(["then", "we", "left"], start=20.0)
        texts = [s["text"] for s in split_sentences(words, "student")]
        self.assertEqual(texts, ["I went there", "then we left"])

    def test_does_not_split_mid_sentence_pause(self):
        # A 1s pause is below the 2s threshold and must not cut.
        words = utterance(["I", "went"]) + utterance(["there", "yesterday."], start=1.7)
        self.assertEqual(len(split_sentences(words, "student")), 1)

    def test_bare_initial_does_not_cut_when_speech_continues(self):
        words = utterance(["The", "letter", "B.", "is", "silent."])
        texts = [s["text"] for s in split_sentences(words, "student")]
        self.assertEqual(texts, ["The letter B. is silent."])

    def test_hard_ceiling_cuts_runaway_speech(self):
        words = utterance(["and"] * 95, step=0.35)
        sentences = split_sentences(words, "student")
        self.assertTrue(len(sentences) >= 2)
        self.assertTrue(all(s["word_count"] <= 40 for s in sentences))

    def test_ignores_the_other_source(self):
        words = utterance(["hello."], source="teacher") + utterance(["hi."], start=5.0)
        self.assertEqual([s["text"] for s in split_sentences(words, "student")], ["hi."])

    def test_trailing_unpunctuated_speech_is_kept(self):
        words = utterance(["I", "was", "going", "to"])
        self.assertEqual([s["text"] for s in split_sentences(words, "student")],
                         ["I was going to"])

    def test_ids_are_stable_and_change_with_content(self):
        words = utterance(["I", "am", "late."])
        first = split_sentences(words, "student")[0]["id"]
        self.assertEqual(split_sentences(words, "student")[0]["id"], first)
        grown = utterance(["I", "am", "very", "late."])
        self.assertNotEqual(split_sentences(grown, "student")[0]["id"], first)

    def test_records_reference_the_original_word_dicts(self):
        words = utterance(["I", "am", "late."])
        sentence = split_sentences(words, "student")[0]
        self.assertIs(sentence["words"][0], words[0])


class SettleTests(unittest.TestCase):
    def test_only_sentences_behind_the_boundary_are_settled(self):
        words = utterance(["old", "one", "here."]) + utterance(["new", "one", "here."], start=30.0)
        sentences = split_sentences(words, "student")
        boundary = newest_word_end(words) - 3.0
        settled = settled_sentences(sentences, boundary)
        self.assertEqual([s["text"] for s in settled], ["old one here."])

    def test_newest_word_end_of_empty_list(self):
        self.assertEqual(newest_word_end([]), 0.0)


class IndexAndConfidenceTests(unittest.TestCase):
    def test_index_maps_word_identity_to_sentence_id(self):
        words = utterance(["I", "am", "late.", "Yes."])
        sentences = split_sentences(words, "student")
        index = index_words_to_sentences(sentences)
        self.assertEqual(index[id(words[0])], sentences[0]["id"])
        self.assertEqual(index[id(words[3])], sentences[1]["id"])

    def test_low_confidence_words_are_deduped_and_stripped(self):
        words = [
            word("I", 0.0, confidence=0.99),
            word("studied", 0.4, confidence=0.42),
            word("studied", 0.8, confidence=0.51),
            word("worse.", 1.2, confidence=0.30),
        ]
        sentence = split_sentences(words, "student")[0]
        self.assertEqual(low_confidence_words(sentence, 0.7), ["studied", "worse"])

    def test_missing_confidence_is_not_reported(self):
        words = [word("hello.", 0.0, confidence=None)]
        sentence = split_sentences(words, "student")[0]
        self.assertEqual(low_confidence_words(sentence, 0.7), [])


@unittest.skipUnless(os.path.exists(REAL_LESSON), "recorded lesson not available")
class RealLessonTests(unittest.TestCase):
    """Guards derived from a real 66-minute lesson, not from synthetic input."""

    @classmethod
    def setUpClass(cls):
        with open(REAL_LESSON, encoding="utf-8") as f:
            cls.words = json.load(f)["words"]
        cls.sentences = split_sentences(cls.words, "student")

    def test_no_sentence_contains_an_oversized_gap(self):
        offenders = [
            s for s in self.sentences
            if any((b["start"] - a["end"]) > 2.0 for a, b in zip(s["words"], s["words"][1:]))
        ]
        self.assertEqual(offenders, [])

    def test_ids_are_unique_across_the_lesson(self):
        ids = [s["id"] for s in self.sentences]
        self.assertEqual(len(set(ids)), len(ids))

    def test_call_volume_stays_modest(self):
        eligible = [s for s in self.sentences if s["word_count"] >= 5]
        minutes = max(w["end"] for w in self.words) / 60.0
        # ~2.6/min measured. The guard is against a change that makes this explode.
        self.assertLess(len(eligible) / minutes, 6.0)


def said(start, end):
    """A sentence spoken between two points on the lesson's audio clock."""
    return {"start": start, "end": end}


class ArmedWindowTests(unittest.TestCase):
    """The switch selects a stretch of speech, not a moment of dispatch."""

    def test_switch_on_opens_an_open_ended_window(self):
        self.assertEqual(update_feedback_windows([], True, False, 30.0), [[30.0, None]])

    def test_switch_off_closes_it(self):
        windows = update_feedback_windows([], True, False, 30.0)
        update_feedback_windows(windows, False, True, 90.0)
        self.assertEqual(windows, [[30.0, 90.0]])

    def test_no_edge_changes_nothing(self):
        windows = [[30.0, None]]
        update_feedback_windows(windows, True, True, 55.0)
        update_feedback_windows([], False, False, 55.0)
        self.assertEqual(windows, [[30.0, None]])

    def test_repeated_arming_accumulates_windows(self):
        windows = []
        update_feedback_windows(windows, True, False, 10.0)
        update_feedback_windows(windows, False, True, 20.0)
        update_feedback_windows(windows, True, False, 60.0)
        update_feedback_windows(windows, False, True, 70.0)
        self.assertEqual(windows, [[10.0, 20.0], [60.0, 70.0]])

    def test_speech_before_arming_is_never_analysed(self):
        """The regression: arming the switch must not backfill the lesson."""
        windows = [[100.0, None]]
        self.assertFalse(in_feedback_window(said(5.0, 9.0), windows))
        self.assertFalse(in_feedback_window(said(80.0, 95.0), windows))

    def test_speech_inside_the_window_is_analysed(self):
        windows = [[100.0, 200.0]]
        self.assertTrue(in_feedback_window(said(120.0, 126.0), windows))

    def test_speech_after_disarming_is_never_analysed(self):
        windows = [[100.0, 200.0]]
        self.assertFalse(in_feedback_window(said(201.0, 205.0), windows))
        self.assertFalse(in_feedback_window(said(900.0, 905.0), windows))

    def test_sentence_straddling_the_arm_edge_counts(self):
        windows = [[100.0, 200.0]]
        self.assertTrue(in_feedback_window(said(97.0, 103.0), windows))

    def test_sentence_straddling_the_disarm_edge_counts(self):
        """It began while the switch was on, so it belongs to the window."""
        windows = [[100.0, 200.0]]
        self.assertTrue(in_feedback_window(said(198.0, 204.0), windows))

    def test_open_window_runs_to_the_end_of_the_lesson(self):
        windows = [[100.0, None]]
        self.assertTrue(in_feedback_window(said(100.0, 106.0), windows))
        self.assertTrue(in_feedback_window(said(4000.0, 4006.0), windows))

    def test_gap_between_two_windows_is_excluded(self):
        windows = [[10.0, 20.0], [60.0, 70.0]]
        self.assertTrue(in_feedback_window(said(12.0, 15.0), windows))
        self.assertFalse(in_feedback_window(said(30.0, 40.0), windows))
        self.assertTrue(in_feedback_window(said(62.0, 66.0), windows))

    def test_no_window_means_nothing_is_analysed(self):
        self.assertFalse(in_feedback_window(said(0.0, 500.0), []))

    def test_window_survives_the_switch_going_off(self):
        """A sentence that settles after disarming still belongs to the window."""
        windows = []
        update_feedback_windows(windows, True, False, 100.0)
        spoken = said(148.0, 150.0)          # said while on
        update_feedback_windows(windows, False, True, 152.0)
        # It only settles a few seconds later, with the switch already off.
        self.assertTrue(in_feedback_window(spoken, windows))
