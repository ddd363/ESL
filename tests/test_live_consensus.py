"""Unit tests for deterministic live consensus transcription.

Tests token normalization, timestamp sequence alignment, consensus agreement,
disagreement / uncertainty tagging, fallback to primary-only when Gladia is unavailable,
and feedback exclusion of uncertain words.
"""

import unittest

from live_consensus import (
    align_word_sequences,
    build_consensus_words,
    build_primary_only_word,
    normalize_token,
)
from live_sentences import split_sentences


def make_word(token, start, end=None, source="student", confidence=0.90):
    return {
        "word": token,
        "start": start,
        "end": start + 0.3 if end is None else end,
        "source": source,
        "confidence": confidence,
    }


class TokenNormalizationTests(unittest.TestCase):
    def test_lowercases_and_strips_outer_punctuation(self):
        self.assertEqual(normalize_token("Hello!"), "hello")
        self.assertEqual(normalize_token("...world?"), "world")
        self.assertEqual(normalize_token("“museum”"), "museum")
        self.assertEqual(normalize_token("‘art’"), "art")
        self.assertEqual(normalize_token("test…"), "test")

    def test_preserves_internal_apostrophe(self):
        self.assertEqual(normalize_token("don't"), "don't")
        self.assertEqual(normalize_token("don’t"), "don't")  # curly quote normalized
        self.assertEqual(normalize_token("he's"), "he's")

    def test_empty_and_whitespace(self):
        self.assertEqual(normalize_token(""), "")
        self.assertEqual(normalize_token("   "), "")
        self.assertEqual(normalize_token(None), "")
        self.assertEqual(normalize_token("..."), "")


class SequenceAlignmentTests(unittest.TestCase):
    def test_perfect_agreement_yields_consensus(self):
        dg = [
            make_word("I", 1.0, 1.2),
            make_word("visited", 1.2, 1.6),
            make_word("the", 1.6, 1.8),
            make_word("museum.", 1.8, 2.3),
        ]
        gl = [
            make_word("I", 1.02, 1.22),
            make_word("visited", 1.21, 1.58),
            make_word("the", 1.59, 1.79),
            make_word("museum", 1.81, 2.28),
        ]
        consensus = align_word_sequences(dg, gl, source="student")
        self.assertEqual(len(consensus), 4)
        for item in consensus:
            self.assertEqual(item["status"], "consensus")
            self.assertEqual(item["provider"], "consensus")
            self.assertEqual(item["source"], "student")
            self.assertIn("deepgram", item["evidence"])
            self.assertIn("gladia", item["evidence"])
        self.assertEqual([w["word"] for w in consensus], ["I", "visited", "the", "museum."])

    def test_single_word_substitution_is_marked_uncertain(self):
        dg = [
            make_word("He", 1.0, 1.2),
            make_word("run", 1.2, 1.5),
            make_word("yesterday.", 1.5, 2.0),
        ]
        gl = [
            make_word("He", 1.0, 1.2),
            make_word("ran", 1.2, 1.5),
            make_word("yesterday", 1.5, 2.0),
        ]
        consensus = align_word_sequences(dg, gl, source="student")
        self.assertEqual(len(consensus), 3)
        self.assertEqual(consensus[0]["status"], "consensus")
        # Disputed word
        self.assertEqual(consensus[1]["status"], "uncertain")
        self.assertEqual(consensus[1]["word"], "run")
        self.assertEqual(consensus[1]["evidence"]["deepgram"]["word"], "run")
        self.assertEqual(consensus[1]["evidence"]["gladia"]["word"], "ran")
        self.assertEqual(consensus[2]["status"], "consensus")

    def test_insertion_or_deletion_is_marked_uncertain(self):
        dg = [
            make_word("I", 1.0, 1.2),
            make_word("really", 1.2, 1.5),
            make_word("like", 1.5, 1.8),
            make_word("it.", 1.8, 2.1),
        ]
        gl = [
            make_word("I", 1.0, 1.2),
            make_word("like", 1.5, 1.8),
            make_word("it", 1.8, 2.1),
        ]
        consensus = align_word_sequences(dg, gl, source="student", gladia_settled_boundary=3.0)
        self.assertEqual(len(consensus), 4)
        self.assertEqual(consensus[0]["status"], "consensus")
        # "really" was heard by DG but omitted by GL -> uncertain
        self.assertEqual(consensus[1]["status"], "uncertain")
        self.assertEqual(consensus[1]["word"], "really")
        self.assertEqual(consensus[2]["status"], "consensus")
        self.assertEqual(consensus[3]["status"], "consensus")

    def test_gladia_only_insertion_is_marked_uncertain(self):
        # Deepgram omitted "the", Gladia recognized "the"
        dg = [
            make_word("I", 1.0, 1.2),
            make_word("went", 1.2, 1.5),
            make_word("to", 1.5, 1.7),
            make_word("museum.", 1.7, 2.3),
        ]
        gl = [
            make_word("I", 1.0, 1.2),
            make_word("went", 1.2, 1.5),
            make_word("to", 1.5, 1.7),
            make_word("the", 1.7, 1.9),
            make_word("museum", 1.9, 2.3),
        ]
        consensus = align_word_sequences(dg, gl, source="student", gladia_settled_boundary=3.0)
        self.assertEqual(len(consensus), 5)
        self.assertEqual(consensus[0]["word"], "I")
        self.assertEqual(consensus[0]["status"], "consensus")
        self.assertEqual(consensus[1]["word"], "went")
        self.assertEqual(consensus[1]["status"], "consensus")
        self.assertEqual(consensus[2]["word"], "to")
        self.assertEqual(consensus[2]["status"], "consensus")
        # Gladia inserted "the" -> must be uncertain
        self.assertEqual(consensus[3]["word"], "the")
        self.assertEqual(consensus[3]["status"], "uncertain")
        self.assertIsNone(consensus[3]["evidence"]["deepgram"])
        self.assertEqual(consensus[3]["evidence"]["gladia"]["word"], "the")
        self.assertEqual(consensus[4]["word"], "museum.")
        self.assertEqual(consensus[4]["status"], "consensus")

        # Confirm that a sentence containing this consensus word list is flagged as having uncertain words
        sentences = split_sentences(consensus, "student")
        self.assertEqual(len(sentences), 1)
        self.assertTrue(any(w.get("status") == "uncertain" for w in sentences[0]["words"]))

    def test_gladia_active_with_empty_words_marks_recent_words_pending(self):
        dg = [
            make_word("Old", 1.0, 1.5),
            make_word("speech.", 1.5, 2.0),
            make_word("Recent", 8.0, 8.5),
            make_word("word.", 8.5, 9.0),
        ]
        # Gladia is active, audio clock is 9.0s.
        # "Old" and "speech." ended at 1.5s/2.0s (> 2.5s ago) -> primary_only
        # "Recent" and "word." ended at 8.5s/9.0s (< 2.5s ago) -> pending
        consensus = build_consensus_words(dg, [], gladia_active=True, audio_clock=9.0)
        self.assertEqual(len(consensus), 4)
        self.assertEqual(consensus[0]["status"], "primary_only")
        self.assertEqual(consensus[1]["status"], "primary_only")
        self.assertEqual(consensus[2]["status"], "pending")
        self.assertEqual(consensus[3]["status"], "pending")

    def test_gladia_unavailable_fallback_to_primary_only(self):
        dg = [
            make_word("Hello", 0.0, 0.4, confidence=0.88),
            make_word("there.", 0.4, 0.8, confidence=0.92),
        ]
        # When Gladia is inactive
        res1 = build_consensus_words(dg, None, gladia_active=False)
        self.assertEqual(len(res1), 2)
        for item in res1:
            self.assertEqual(item["status"], "primary_only")
            self.assertEqual(item["provider"], "deepgram")
            self.assertIn("deepgram", item["evidence"])
            self.assertNotIn("gladia", item["evidence"])


class LiveConsensusStreamTests(unittest.TestCase):
    def test_multi_speaker_sources_are_isolated(self):
        dg = [
            make_word("Student", 1.0, 1.3, source="student"),
            make_word("Teacher", 1.1, 1.4, source="teacher"),
            make_word("talking.", 1.3, 1.7, source="student"),
            make_word("reply.", 1.4, 1.8, source="teacher"),
        ]
        gl = [
            make_word("Student", 1.0, 1.3, source="student"),
            make_word("Teacher", 1.1, 1.4, source="teacher"),
            make_word("talking", 1.3, 1.7, source="student"),
            make_word("reply", 1.4, 1.8, source="teacher"),
        ]
        consensus = build_consensus_words(dg, gl, gladia_active=True, audio_clock=2.5)
        self.assertEqual(len(consensus), 4)
        student_words = [w for w in consensus if w["source"] == "student"]
        teacher_words = [w for w in consensus if w["source"] == "teacher"]
        self.assertEqual(len(student_words), 2)
        self.assertEqual(len(teacher_words), 2)
        self.assertTrue(all(w["status"] == "consensus" for w in consensus))


class FeedbackExclusionTests(unittest.TestCase):
    def test_sentence_with_uncertain_word_is_detectable(self):
        # Build consensus words with one uncertain word
        words = [
            {
                "word": "I",
                "start": 1.0,
                "end": 1.2,
                "source": "student",
                "status": "consensus",
                "provider": "consensus",
            },
            {
                "word": "seen",
                "start": 1.2,
                "end": 1.5,
                "source": "student",
                "status": "uncertain",
                "provider": "consensus",
                "evidence": {
                    "deepgram": {"word": "seen", "confidence": 0.65},
                    "gladia": {"word": "saw", "confidence": 0.82},
                },
            },
            {
                "word": "him.",
                "start": 1.5,
                "end": 1.9,
                "source": "student",
                "status": "consensus",
                "provider": "consensus",
            },
        ]
        sentences = split_sentences(words, "student")
        self.assertEqual(len(sentences), 1)
        sent = sentences[0]
        # Verify that checking any(w.get("status") == "uncertain") flags the sentence
        has_uncertain = any(w.get("status") == "uncertain" for w in sent["words"])
        self.assertTrue(has_uncertain)

    def test_clean_consensus_sentence_has_no_uncertain_words(self):
        words = [
            {
                "word": "I",
                "start": 1.0,
                "end": 1.2,
                "source": "student",
                "status": "consensus",
                "provider": "consensus",
            },
            {
                "word": "saw",
                "start": 1.2,
                "end": 1.5,
                "source": "student",
                "status": "consensus",
                "provider": "consensus",
            },
            {
                "word": "him.",
                "start": 1.5,
                "end": 1.9,
                "source": "student",
                "status": "consensus",
                "provider": "consensus",
            },
        ]
        sentences = split_sentences(words, "student")
        self.assertEqual(len(sentences), 1)
        sent = sentences[0]
        has_uncertain = any(w.get("status") == "uncertain" for w in sent["words"])
        self.assertFalse(has_uncertain)


class BoundedWaitAndPunctuationTests(unittest.TestCase):
    def test_words_with_different_punctuation_agree(self):
        dg = [
            make_word("First,", 1.0, 1.4),
            make_word('"second"', 1.4, 1.8),
            make_word("third...", 1.8, 2.2),
        ]
        gl = [
            make_word("first", 1.02, 1.38),
            make_word("second", 1.41, 1.79),
            make_word("third", 1.81, 2.19),
        ]
        consensus = build_consensus_words(dg, gl, gladia_active=True, audio_clock=3.0)
        self.assertEqual(len(consensus), 3)
        self.assertTrue(all(w["status"] == "consensus" for w in consensus))
        self.assertEqual([w["word"] for w in consensus], ["First,", '"second"', "third..."])

    def test_bounded_wait_cutoff_emits_primary_only_for_lagging_gladia(self):
        dg = [
            make_word("Old", 1.0, 1.5),
            make_word("word.", 1.5, 2.0),
            make_word("Recent", 8.0, 8.5),
            make_word("speech.", 8.5, 9.0),
        ]
        gl = [
            make_word("Old", 1.0, 1.5),
            make_word("word", 1.5, 2.0),
        ]
        # At audio_clock = 9.0s (< 2.5s since recent speech):
        # "Recent" (8.0-8.5s) and "speech." (8.5-9.0s) are within bounded wait window -> pending
        consensus_pending = build_consensus_words(dg, gl, gladia_active=True, audio_clock=9.0)
        self.assertEqual(len(consensus_pending), 4)
        self.assertEqual(consensus_pending[0]["status"], "consensus")
        self.assertEqual(consensus_pending[1]["status"], "consensus")
        self.assertEqual(consensus_pending[2]["status"], "pending")
        self.assertEqual(consensus_pending[3]["status"], "pending")

        # At audio_clock = 12.0s (> 2.5s since recent speech):
        # 3.0 seconds have elapsed with no Gladia words past 2.0s -> falls back to primary_only
        consensus_expired = build_consensus_words(dg, gl, gladia_active=True, audio_clock=12.0)
        self.assertEqual(len(consensus_expired), 4)
        self.assertEqual(consensus_expired[0]["status"], "consensus")
        self.assertEqual(consensus_expired[1]["status"], "consensus")
        self.assertEqual(consensus_expired[2]["status"], "primary_only")
        self.assertEqual(consensus_expired[3]["status"], "primary_only")


if __name__ == "__main__":
    unittest.main()
