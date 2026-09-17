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


def make_word(
    token,
    start,
    end=None,
    source="student",
    confidence=0.90,
    is_final=True,
    speech_final=False,
    word_is_final=True,
    turn_order=1,
):
    return {
        "word": token,
        "start": start,
        "end": start + 0.3 if end is None else end,
        "source": source,
        "confidence": confidence,
        "is_final": is_final,
        "speech_final": speech_final,
        "word_is_final": word_is_final,
        "turn_order": turn_order,
    }


class ThreeProviderConsensusTests(unittest.TestCase):
    def test_all_three_providers_agree_yields_consensus(self):
        dg = [make_word("volunteer", 1.0, 1.5, confidence=0.92)]
        gl = [make_word("volunteer", 1.02, 1.48, confidence=0.95)]
        aai = [make_word("volunteer", 1.01, 1.49, confidence=0.94)]

        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=10.0)
        self.assertEqual(len(consensus), 1)
        self.assertEqual(consensus[0]["word"], "volunteer")
        self.assertEqual(consensus[0]["status"], "consensus")
        self.assertIsNotNone(consensus[0]["evidence"]["deepgram"])
        self.assertIsNotNone(consensus[0]["evidence"]["gladia"])
        self.assertIsNotNone(consensus[0]["evidence"]["assemblyai"])

    def test_dg_and_gladia_agree_assemblyai_disagrees(self):
        # 2-of-3 agreement: DG + Gladia agree -> consensus
        dg = [make_word("picnic", 1.0, 1.4, confidence=0.90)]
        gl = [make_word("picnic", 1.02, 1.38, confidence=0.92)]
        aai = [make_word("panic", 1.01, 1.39, confidence=0.75)]

        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=10.0)
        self.assertEqual(len(consensus), 1)
        self.assertEqual(consensus[0]["word"], "picnic")
        self.assertEqual(consensus[0]["status"], "consensus")
        self.assertEqual(consensus[0]["evidence"]["assemblyai"]["word"], "panic")

    def test_dg_and_assemblyai_agree_gladia_disagrees(self):
        # 2-of-3 agreement: DG + AssemblyAI agree -> consensus (symmetric with DG+GL)
        dg = [make_word("volunteer", 1.0, 1.5, confidence=0.92, is_final=True)]
        gl = [make_word("volunteering", 1.02, 1.48, confidence=0.80, is_final=True)]
        aai = [make_word("volunteer", 1.01, 1.49, confidence=0.91, is_final=True)]

        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=10.0)
        self.assertEqual(len(consensus), 1)
        self.assertEqual(consensus[0]["word"], "volunteer")
        self.assertEqual(consensus[0]["status"], "consensus")
        self.assertEqual(consensus[0]["evidence"]["gladia"]["word"], "volunteering")

    def test_gladia_and_assemblyai_agree_dg_disagrees(self):
        # 2-of-3 agreement: Gladia + AssemblyAI agree -> consensus (symmetric with DG+GL)
        dg = [make_word("picnic", 1.0, 1.4, confidence=0.70, is_final=True)]
        gl = [make_word("panic", 1.02, 1.38, confidence=0.95, is_final=True)]
        aai = [make_word("panic", 1.01, 1.39, confidence=0.92, is_final=True)]

        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=10.0)
        self.assertEqual(len(consensus), 1)
        self.assertEqual(consensus[0]["word"], "panic")
        self.assertEqual(consensus[0]["status"], "consensus")
        self.assertEqual(consensus[0]["evidence"]["deepgram"]["word"], "picnic")

    def test_all_three_disagree_evaluated_via_multi_signal_evidence(self):
        dg = [make_word("there", 1.0, 1.3, confidence=0.60, is_final=True)]
        gl = [make_word("their", 1.02, 1.28, confidence=0.95, is_final=True)]
        aai = [make_word("they're", 1.01, 1.29, confidence=0.70, is_final=True)]

        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=10.0)
        self.assertEqual(len(consensus), 1)
        self.assertEqual(consensus[0]["status"], "uncertain")
        # Gladia has highest local confidence + finality bonus
        self.assertEqual(consensus[0]["word"], "their")
        self.assertEqual(consensus[0]["evidence"]["deepgram"]["word"], "there")
        self.assertEqual(consensus[0]["evidence"]["gladia"]["word"], "their")
        self.assertEqual(consensus[0]["evidence"]["assemblyai"]["word"], "they're")

    def test_assemblyai_partial_vs_dg_and_gladia_final(self):
        # AAI is partial while DG and Gladia are final
        dg = [make_word("picnic", 1.0, 1.4, confidence=0.92, is_final=True)]
        gl = [make_word("picnic", 1.02, 1.38, confidence=0.94, is_final=True)]
        aai = [make_word("panic", 1.01, 1.39, confidence=0.85, is_final=False, word_is_final=False)]

        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=10.0)
        self.assertEqual(len(consensus), 1)
        self.assertEqual(consensus[0]["word"], "picnic")
        self.assertEqual(consensus[0]["status"], "consensus")

    def test_assemblyai_final_vs_dg_and_gladia_partial(self):
        # AAI is final with high confidence while DG is partial
        dg = [make_word("picnic", 1.0, 1.4, confidence=0.50, is_final=False)]
        gl = [make_word("panic", 1.02, 1.38, confidence=0.50, is_final=False)]
        aai = [make_word("panic", 1.01, 1.39, confidence=0.95, is_final=True, word_is_final=True)]

        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=10.0)
        self.assertEqual(len(consensus), 1)
        self.assertEqual(consensus[0]["word"], "panic")
        self.assertEqual(consensus[0]["status"], "uncertain")

    def test_assemblyai_unavailable_falls_back_seamlessly_to_two_providers(self):
        dg = [make_word("hello", 1.0, 1.4)]
        gl = [make_word("hello", 1.02, 1.38)]

        consensus = build_consensus_words(dg, gl, aai_words=None, assemblyai_active=False, audio_clock=10.0)
        self.assertEqual(len(consensus), 1)
        self.assertEqual(consensus[0]["word"], "hello")
        self.assertEqual(consensus[0]["status"], "consensus")
        self.assertIsNone(consensus[0]["evidence"].get("assemblyai"))
        self.assertNotIn("assemblyai", consensus[0]["evidence"])

    def test_formatting_differences_such_as_digits_do_not_cause_false_substitutions(self):
        # AssemblyAI returned "8" formatted, while Deepgram returned "eight"
        dg = [make_word("eight", 1.0, 1.4)]
        gl = [make_word("eight", 1.02, 1.38)]
        aai = [make_word("8", 1.01, 1.39)]

        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=10.0)
        self.assertEqual(len(consensus), 1)
        # Reconciles to consensus because normalize_token("8") == normalize_token("eight") == "eight"
        self.assertEqual(consensus[0]["status"], "consensus")
        self.assertEqual(consensus[0]["word"], "eight")

    def test_assemblyai_never_blocks_live_consensus_when_dg_and_gl_settled(self):
        # AAI has no words in this cluster, DG and Gladia are settled
        dg = [make_word("fast", 1.0, 1.4)]
        gl = [make_word("fast", 1.02, 1.38)]
        aai = []

        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=5.0)
        self.assertEqual(len(consensus), 1)
        self.assertEqual(consensus[0]["word"], "fast")
        self.assertEqual(consensus[0]["status"], "consensus")


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
        # Disputed word: Gladia final receives finality bonus over Deepgram on equal confidence
        self.assertEqual(consensus[1]["status"], "uncertain")
        self.assertEqual(consensus[1]["word"], "ran")
        self.assertEqual(consensus[1]["evidence"]["deepgram"]["word"], "run")
        self.assertEqual(consensus[1]["evidence"]["gladia"]["word"], "ran")
        self.assertEqual(consensus[2]["status"], "consensus")

    def test_deepgram_wins_substitution_when_gladia_confidence_is_low(self):
        dg = [make_word("society", 1.0, 1.5, confidence=0.95)]
        gl = [make_word("suicide", 1.05, 1.48, confidence=0.35)]
        consensus = align_word_sequences(dg, gl, source="student")
        self.assertEqual(len(consensus), 1)
        self.assertEqual(consensus[0]["status"], "uncertain")
        # Deepgram high confidence wins over Gladia low confidence
        self.assertEqual(consensus[0]["word"], "society")
        self.assertEqual(consensus[0]["evidence"]["deepgram"]["word"], "society")
        self.assertEqual(consensus[0]["evidence"]["gladia"]["word"], "suicide")

    def test_deepgram_final_wins_when_gladia_is_partial(self):
        dg = [make_word("picnic", 1.0, 1.5, confidence=0.90, is_final=True)]
        gl = [make_word("panic", 1.05, 1.48, confidence=0.90, is_final=False)]
        consensus = align_word_sequences(dg, gl, source="student")
        self.assertEqual(len(consensus), 1)
        self.assertEqual(consensus[0]["status"], "uncertain")
        self.assertEqual(consensus[0]["word"], "picnic")

    def test_time_proximity_gating_prevents_distant_substitutions(self):
        # Two different words separated by > 1.5s must be treated as separate insertion/deletion, not substitution
        dg = [make_word("apple", 1.0, 1.4)]
        gl = [make_word("orange", 3.0, 3.4)]
        consensus = align_word_sequences(dg, gl, source="student", gladia_settled_boundary=4.0)
        self.assertEqual(len(consensus), 2)
        self.assertEqual(consensus[0]["word"], "apple")
        self.assertEqual(consensus[0]["status"], "uncertain")
        self.assertEqual(consensus[1]["word"], "orange")
        self.assertEqual(consensus[1]["status"], "uncertain")

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


    def test_long_monologue_subdivision_drops_no_words(self):
        # Generate 200 words of continuous speech
        dg = []
        gl = []
        for i in range(200):
            t = i * 0.4
            dg.append(make_word(f"word{i}", t, t + 0.35))
            gl.append(make_word(f"word{i}", t + 0.05, t + 0.38))
        consensus = build_consensus_words(dg, gl, gladia_active=True, audio_clock=100.0)
        self.assertEqual(len(consensus), 200)
        self.assertEqual([w["word"] for w in consensus], [f"word{i}" for i in range(200)])
        self.assertTrue(all(w["status"] == "consensus" for w in consensus))

    def test_gladia_delayed_startup_drops_no_words(self):
        # Deepgram started at 0.0s, Gladia only starts at 3.0s
        dg = [
            make_word("First", 0.5, 0.9),
            make_word("sentence", 1.0, 1.4),
            make_word("here.", 1.5, 1.9),
            make_word("Second", 3.5, 3.9),
            make_word("sentence.", 4.0, 4.5),
        ]
        gl = [
            make_word("Second", 3.52, 3.91),
            make_word("sentence", 4.01, 4.49),
        ]
        # At audio_clock = 5.0s, all speech is > 2.5s settled
        consensus = build_consensus_words(dg, gl, gladia_active=True, audio_clock=5.0)
        # 100% of Deepgram words must be present
        self.assertEqual(len(consensus), 5)
        self.assertEqual([w["word"] for w in consensus], ["First", "sentence", "here.", "Second", "sentence."])
        # Words before Gladia started should be primary_only or uncertain, never lost
        self.assertIn(consensus[0]["status"], ("primary_only", "uncertain"))
        self.assertIn(consensus[1]["status"], ("primary_only", "uncertain"))
        self.assertIn(consensus[2]["status"], ("primary_only", "uncertain"))
        # Words with Gladia agreement must be consensus
        self.assertEqual(consensus[3]["status"], "consensus")
        self.assertEqual(consensus[4]["status"], "consensus")


class SymmetricTwoOfThreeConsensusTests(unittest.TestCase):
    def test_dg_gl_agreement_yields_consensus(self):
        dg = [make_word("bicycle", 1.0, 1.4, confidence=0.90, is_final=True)]
        gl = [make_word("bicycle", 1.02, 1.38, confidence=0.92, is_final=True)]
        aai = [make_word("tricycle", 1.01, 1.39, confidence=0.85, is_final=True)]
        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=10.0)
        self.assertEqual(len(consensus), 1)
        self.assertEqual(consensus[0]["word"], "bicycle")
        self.assertEqual(consensus[0]["status"], "consensus")

    def test_gl_aai_agreement_yields_consensus(self):
        dg = [make_word("tricycle", 1.0, 1.4, confidence=0.70, is_final=True)]
        gl = [make_word("bicycle", 1.02, 1.38, confidence=0.95, is_final=True)]
        aai = [make_word("bicycle", 1.01, 1.39, confidence=0.92, is_final=True)]
        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=10.0)
        self.assertEqual(len(consensus), 1)
        self.assertEqual(consensus[0]["word"], "bicycle")
        self.assertEqual(consensus[0]["status"], "consensus")

    def test_dg_aai_agreement_yields_consensus(self):
        dg = [make_word("bicycle", 1.0, 1.4, confidence=0.92, is_final=True)]
        gl = [make_word("tricycle", 1.02, 1.38, confidence=0.80, is_final=True)]
        aai = [make_word("bicycle", 1.01, 1.39, confidence=0.91, is_final=True)]
        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=10.0)
        self.assertEqual(len(consensus), 1)
        self.assertEqual(consensus[0]["word"], "bicycle")
        self.assertEqual(consensus[0]["status"], "consensus")

    def test_unreliable_agreeing_pair_yields_uncertain(self):
        # AAI agrees with GL, but AAI is a provisional partial hypothesis -> marked uncertain
        dg = [make_word("picnic", 1.0, 1.4, confidence=0.85, is_final=True)]
        gl = [make_word("panic", 1.02, 1.38, confidence=0.60, is_final=False)]
        aai = [make_word("panic", 1.01, 1.39, confidence=0.50, is_final=False)]
        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=10.0)
        self.assertEqual(len(consensus), 1)
        self.assertEqual(consensus[0]["status"], "uncertain")


class HeuristicEvidenceScoringTests(unittest.TestCase):
    def test_provider_model_finality_weights(self):
        from live_consensus import _score_candidate
        # Gladia final gets +1.5 finality weight
        gl_score = _score_candidate({"word": "test", "is_final": True, "confidence": None}, provider="gladia")
        # AssemblyAI final gets +1.3 finality weight
        aai_score = _score_candidate({"word": "test", "is_final": True, "confidence": None}, provider="assemblyai")
        # Deepgram final gets +1.0 finality weight
        dg_score = _score_candidate({"word": "test", "is_final": True, "confidence": None}, provider="deepgram")
        self.assertGreater(gl_score, aai_score)
        self.assertGreater(aai_score, dg_score)

    def test_endpointing_flags_boost_score(self):
        from live_consensus import _score_candidate
        base_dg = _score_candidate({"word": "test", "is_final": True, "speech_final": False}, provider="deepgram")
        endpointed_dg = _score_candidate({"word": "test", "is_final": True, "speech_final": True}, provider="deepgram")
        self.assertAlmostEqual(endpointed_dg - base_dg, 0.5)

    def test_provider_local_confidence_tiers(self):
        from live_consensus import _score_candidate
        high_score = _score_candidate({"word": "test", "is_final": True, "confidence": 0.95}, provider="deepgram")
        med_score = _score_candidate({"word": "test", "is_final": True, "confidence": 0.70}, provider="deepgram")
        low_score = _score_candidate({"word": "test", "is_final": True, "confidence": 0.30}, provider="deepgram")
        self.assertGreater(high_score, med_score)
        self.assertGreater(med_score, low_score)


class PunctuationSeparationTests(unittest.TestCase):
    def test_words_with_different_punctuation_reconcile_lexically(self):
        dg = [make_word("finish,", 1.0, 1.4)]
        gl = [make_word("finish", 1.02, 1.38)]
        aai = [make_word("finish.", 1.01, 1.39)]
        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=10.0)
        self.assertEqual(len(consensus), 1)
        self.assertEqual(consensus[0]["status"], "consensus")
        # Word matches regardless of punctuation variations
        self.assertTrue(consensus[0]["word"].startswith("finish"))

    def test_single_provider_terminal_period_preserved_when_others_omit(self):
        # Deepgram places a period, while Gladia and AAI omit punctuation
        dg = [make_word("school.", 1.0, 1.4, is_final=True)]
        gl = [make_word("school", 1.02, 1.38, is_final=True)]
        aai = [make_word("school", 1.01, 1.39, is_final=True)]
        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=10.0)
        self.assertEqual(len(consensus), 1)
        # Reconciled punctuation preserves the single-provider terminal mark
        self.assertEqual(consensus[0]["word"], "school.")

    def test_corroborated_terminal_punctuation_preserved(self):
        # 2 providers have '.'
        dg = [make_word("today.", 1.0, 1.4, is_final=True)]
        gl = [make_word("today.", 1.02, 1.38, is_final=True)]
        aai = [make_word("today", 1.01, 1.39, is_final=True)]
        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=10.0)
        self.assertEqual(len(consensus), 1)
        self.assertEqual(consensus[0]["word"], "today.")


class RevisionSafeConsensusTests(unittest.TestCase):
    def test_word_id_immutable_when_consensus_word_revised_from_picnic_to_panic(self):
        # Step 1: Deepgram emits "picnic" at t=12.30-12.65
        dg = [make_word("picnic", 12.30, 12.65, confidence=0.70)]
        c1 = build_consensus_words(dg, None, gladia_active=False, audio_clock=15.0)
        self.assertEqual(len(c1), 1)
        self.assertEqual(c1[0]["word"], "picnic")
        self.assertEqual(c1[0]["word_id"], "student:12.300")

        # Step 2: Gladia and AssemblyAI arrive later with "panic" at t=12.34-12.71
        gl = [make_word("panic", 12.34, 12.71, confidence=0.95)]
        aai = [make_word("panic", 12.34, 12.71, confidence=0.92)]
        c2 = build_consensus_words(dg, gl, aai_words=aai, audio_clock=15.0)
        self.assertEqual(len(c2), 1)
        # Word content is revised to "panic"
        self.assertEqual(c2[0]["word"], "panic")
        self.assertEqual(c2[0]["status"], "consensus")
        # Logical word identity remains IMMUTABLE
        self.assertEqual(c2[0]["word_id"], "student:12.300")
        self.assertEqual(c2[0]["word_id"], c1[0]["word_id"])
        # Provenance is preserved
        self.assertEqual(c2[0]["evidence"]["deepgram"]["word"], "picnic")
        self.assertEqual(c2[0]["evidence"]["gladia"]["word"], "panic")
        self.assertEqual(c2[0]["evidence"]["assemblyai"]["word"], "panic")

    def test_sentence_id_stable_and_revision_tracked_when_content_revised(self):
        # Step 1: Deepgram generates provisional sentence "I visited picnic."
        dg_words = [
            make_word("I", 1.0, 1.2),
            make_word("visited", 1.2, 1.5),
            make_word("picnic.", 1.5, 1.9),
        ]
        s1 = split_sentences(dg_words, "student")
        self.assertEqual(len(s1), 1)
        sent1 = s1[0]
        self.assertEqual(sent1["text"], "I visited picnic.")
        self.assertEqual(sent1["sentence_id"], "sent:student:1.000")
        rev1 = sent1["revision"]

        # Step 2: Gladia and AAI settle "panic."
        revised_words = [
            make_word("I", 1.0, 1.2),
            make_word("visited", 1.2, 1.5),
            make_word("panic.", 1.52, 1.88),
        ]
        s2 = split_sentences(revised_words, "student")
        self.assertEqual(len(s2), 1)
        sent2 = s2[0]
        self.assertEqual(sent2["text"], "I visited panic.")
        # Sentence ID is strictly identical across revisions
        self.assertEqual(sent2["sentence_id"], sent1["sentence_id"])
        # Revision content digest is distinct
        self.assertNotEqual(sent2["revision"], rev1)
        # Revision dispatch IDs are distinct
        self.assertNotEqual(sent2["id"], sent1["id"])

    def test_late_provider_corroboration_produces_identical_sentence_revision(self):
        # Step 1: Deepgram + Gladia settle words
        dg = [make_word("Hello", 1.0, 1.3), make_word("world.", 1.3, 1.7)]
        gl = [make_word("Hello", 1.02, 1.28), make_word("world.", 1.31, 1.69)]
        c1 = build_consensus_words(dg, gl, audio_clock=10.0)

        # Step 2: Late AssemblyAI arrives with corroboration
        aai = [make_word("Hello", 1.01, 1.29), make_word("world.", 1.32, 1.68)]
        c2 = build_consensus_words(dg, gl, aai_words=aai, audio_clock=10.0)

        s1 = split_sentences(c1, "student")
        s2 = split_sentences(c2, "student")
        self.assertEqual(s1[0]["sentence_id"], s2[0]["sentence_id"])
        self.assertEqual(s1[0]["revision"], s2[0]["revision"])
        self.assertEqual(s1[0]["id"], s2[0]["id"])

    def test_late_arriving_primary_provider_promotes_secondary_insertion_to_consensus(self):
        # Step 1: Gladia and AAI emit word "coffee" first before Deepgram starts
        gl = [make_word("coffee", 5.0, 5.4, confidence=0.92, is_final=True)]
        aai = [make_word("coffee", 5.01, 5.39, confidence=0.94, is_final=True)]
        c1 = build_consensus_words([], gl, aai_words=aai, audio_clock=6.0)
        self.assertEqual(len(c1), 1)
        self.assertEqual(c1[0]["word"], "coffee")
        self.assertEqual(c1[0]["status"], "consensus")
        # Step 2: Deepgram stream catches up and emits "coffee" at the same acoustic slot
        dg = [make_word("coffee", 5.02, 5.42, confidence=0.90, is_final=True)]
        c2 = build_consensus_words(dg, gl, aai_words=aai, audio_clock=6.0)
        self.assertEqual(len(c2), 1)
        # Deepgram is incorporated without duplicate words
        self.assertEqual(c2[0]["word"], "coffee")
        self.assertEqual(c2[0]["status"], "consensus")
        self.assertIsNotNone(c2[0]["evidence"]["deepgram"])
        self.assertIsNotNone(c2[0]["evidence"]["gladia"])
        self.assertIsNotNone(c2[0]["evidence"]["assemblyai"])

    def test_sentence_identity_when_leading_word_is_added_or_prepended(self):
        # Step 1: Sentence initially starts at t=1.00
        w1 = [make_word("I", 1.0, 1.2), make_word("went", 1.2, 1.4), make_word("to", 1.4, 1.6), make_word("school.", 1.6, 2.0)]
        s1 = split_sentences(w1, "student")
        self.assertEqual(len(s1), 1)
        self.assertEqual(s1[0]["sentence_id"], "sent:student:1.000")
        self.assertEqual(s1[0]["text"], "I went to school.")

        # Step 2: Late provider recovers omitted leading word "Actually," at t=0.50
        w2 = [make_word("Actually,", 0.5, 0.9), make_word("I", 1.0, 1.2), make_word("went", 1.2, 1.4), make_word("to", 1.4, 1.6), make_word("school.", 1.6, 2.0)]
        s2 = split_sentences(w2, "student")
        self.assertEqual(len(s2), 1)
        # Adding a leading fronted word establishes a new logical utterance anchor beginning at 0.500
        self.assertEqual(s2[0]["sentence_id"], "sent:student:0.500")
        self.assertEqual(s2[0]["text"], "Actually, I went to school.")
        # Generates a distinct revision ID triggering fresh pedagogical evaluation
        self.assertNotEqual(s2[0]["revision_id"], s1[0]["revision_id"])
        self.assertNotEqual(s2[0]["id"], s1[0]["id"])


class AdversarialAlignmentTests(unittest.TestCase):
    def test_repeated_identical_words_in_sequence(self):
        # "the the"
        dg = [make_word("the", 1.0, 1.2), make_word("the", 1.3, 1.5)]
        gl = [make_word("the", 1.02, 1.22), make_word("the", 1.31, 1.52)]
        aai = [make_word("the", 1.01, 1.21), make_word("the", 1.32, 1.51)]
        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=10.0)
        self.assertEqual(len(consensus), 2)
        self.assertEqual(consensus[0]["word"], "the")
        self.assertEqual(consensus[0]["status"], "consensus")
        self.assertEqual(consensus[1]["word"], "the")
        self.assertEqual(consensus[1]["status"], "consensus")

    def test_repeated_identical_words_one_provider_omits_duplicate(self):
        # DG has "the the", GL has single "the"
        dg = [make_word("the", 1.0, 1.2), make_word("the", 1.3, 1.5)]
        gl = [make_word("the", 1.02, 1.22)]
        aai = [make_word("the", 1.01, 1.21), make_word("the", 1.32, 1.51)]
        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=10.0)
        # DG + AAI agree on second "the" (2-of-3 agreement)
        self.assertEqual(len(consensus), 2)
        self.assertEqual(consensus[0]["word"], "the")
        self.assertEqual(consensus[0]["status"], "consensus")
        self.assertEqual(consensus[1]["word"], "the")
        self.assertEqual(consensus[1]["status"], "consensus")

    def test_repeated_phrase_alignment(self):
        # "I think I think"
        dg = [make_word("I", 1.0, 1.1), make_word("think", 1.1, 1.3),
              make_word("I", 1.4, 1.5), make_word("think.", 1.5, 1.8)]
        gl = [make_word("I", 1.02, 1.12), make_word("think", 1.12, 1.32),
              make_word("I", 1.41, 1.52), make_word("think", 1.52, 1.78)]
        consensus = build_consensus_words(dg, gl, audio_clock=10.0)
        self.assertEqual(len(consensus), 4)
        self.assertEqual([w["word"] for w in consensus], ["I", "think", "I", "think."])
        self.assertTrue(all(w["status"] == "consensus" for w in consensus))

    def test_one_provider_omits_several_words(self):
        dg = [make_word("I", 1.0, 1.2), make_word("went", 1.2, 1.4),
              make_word("to", 1.4, 1.6), make_word("the", 1.6, 1.8),
              make_word("museum.", 1.8, 2.2)]
        gl = [make_word("I", 1.02, 1.22), make_word("museum", 1.82, 2.18)]
        consensus = build_consensus_words(dg, gl, audio_clock=10.0)
        self.assertEqual(len(consensus), 5)
        self.assertEqual([w["word"] for w in consensus], ["I", "went", "to", "the", "museum."])
        self.assertEqual(consensus[0]["status"], "consensus")
        self.assertEqual(consensus[4]["status"], "consensus")

    def test_timestamp_drift_between_providers(self):
        # 1.0 second drift across entire sentence
        dg = [make_word("We", 1.0, 1.3), make_word("love", 1.3, 1.6), make_word("reading.", 1.6, 2.0)]
        gl = [make_word("We", 2.0, 2.3), make_word("love", 2.3, 2.6), make_word("reading", 2.6, 3.0)]
        consensus = build_consensus_words(dg, gl, audio_clock=10.0)
        self.assertEqual(len(consensus), 3)
        self.assertEqual([w["word"] for w in consensus], ["We", "love", "reading."])
        self.assertTrue(all(w["status"] == "consensus" for w in consensus))

    def test_different_speech_rates(self):
        # Deepgram fast, Gladia stretched
        dg = [make_word("Fast", 1.0, 1.2), make_word("speech.", 1.2, 1.4)]
        gl = [make_word("Fast", 1.0, 1.4), make_word("speech", 1.4, 2.0)]
        consensus = build_consensus_words(dg, gl, audio_clock=10.0)
        self.assertEqual(len(consensus), 2)
        self.assertTrue(all(w["status"] == "consensus" for w in consensus))

    def test_extra_phrase_at_end_of_cluster(self):
        dg = [make_word("Hello", 1.0, 1.3)]
        gl = [make_word("Hello", 1.02, 1.31), make_word("everyone", 1.35, 1.7)]
        consensus = build_consensus_words(dg, gl, audio_clock=10.0)
        self.assertEqual(len(consensus), 2)
        self.assertEqual(consensus[0]["word"], "Hello")
        self.assertEqual(consensus[0]["status"], "consensus")
        self.assertEqual(consensus[1]["word"], "everyone")
        self.assertEqual(consensus[1]["status"], "uncertain")


class BoundaryPartitioningGuaranteeTests(unittest.TestCase):
    def test_subdivide_preserves_100_percent_of_input_words(self):
        from live_consensus import _subdivide_cluster_if_needed
        dg = [make_word(f"dg_{i}", i * 0.3, i * 0.3 + 0.25) for i in range(180)]
        gl = [make_word(f"gl_{i}", i * 0.3 + 0.05, i * 0.3 + 0.28) for i in range(180)]
        aai = [make_word(f"aai_{i}", i * 0.3 + 0.02, i * 0.3 + 0.27) for i in range(180)]

        clusters = _subdivide_cluster_if_needed(dg, gl, aai, max_words=100, max_duration_s=30.0)
        total_dg = sum(len(c[0]) for c in clusters)
        total_gl = sum(len(c[1]) for c in clusters)
        total_aai = sum(len(c[2]) for c in clusters)

        self.assertEqual(total_dg, 180)
        self.assertEqual(total_gl, 180)
        self.assertEqual(total_aai, 180)


class EndToEndInvariantsTests(unittest.TestCase):
    def test_invariant_1_stable_word_identity(self):
        dg = [make_word("listen", 1.0, 1.4)]
        consensus = build_consensus_words(dg, None, gladia_active=False, audio_clock=10.0)
        self.assertTrue("word_id" in consensus[0])
        self.assertTrue(consensus[0]["word_id"].startswith("student:"))

    def test_invariant_2_late_provider_cannot_create_duplicate_words(self):
        dg = [make_word("hello", 1.0, 1.3), make_word("world.", 1.3, 1.7)]
        gl = [make_word("hello", 1.02, 1.31)]
        # Late Gladia only has first word
        c1 = build_consensus_words(dg, gl, audio_clock=10.0)
        self.assertEqual(len(c1), 2)
        # When Gladia catches up with second word
        gl_updated = [make_word("hello", 1.02, 1.31), make_word("world", 1.32, 1.68)]
        c2 = build_consensus_words(dg, gl_updated, audio_clock=10.0)
        self.assertEqual(len(c2), 2)
        self.assertEqual([w["word"] for w in c2], ["hello", "world."])

    def test_invariant_3_provenance_preserved_on_revision(self):
        dg = [make_word("learn", 1.0, 1.4, confidence=0.88)]
        gl = [make_word("learn", 1.02, 1.38, confidence=0.94)]
        aai = [make_word("learn", 1.01, 1.39, confidence=0.92)]
        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=10.0)
        self.assertEqual(consensus[0]["evidence"]["deepgram"]["confidence"], 0.88)
        self.assertEqual(consensus[0]["evidence"]["gladia"]["confidence"], 0.94)
        self.assertEqual(consensus[0]["evidence"]["assemblyai"]["confidence"], 0.92)

    def test_invariant_4_sentence_identity_stable_across_revisions(self):
        w = [make_word("Good", 1.0, 1.3), make_word("morning.", 1.3, 1.7)]
        s = split_sentences(w, "student")
        self.assertEqual(s[0]["sentence_id"], "sent:student:1.000")

    def test_invariant_5_feedback_deduplication_across_revision_lifecycle(self):
        # 1. DG produces pending word
        dg = [make_word("I", 1.0, 1.2), make_word("panic.", 1.2, 1.5)]
        # At audio_clock=1.5s, words are pending (waiting for Gladia)
        c_pending = build_consensus_words(dg, None, audio_clock=1.5)
        s_pending = split_sentences(c_pending, "student")
        # In live dispatch, pending status blocks analysis
        statuses_pending = {w.get("status") for w in s_pending[0]["words"]}
        self.assertIn("pending", statuses_pending)

        # 2. Gladia settles words into revision 1 (both agree on panic.)
        gl = [make_word("I", 1.02, 1.22), make_word("panic.", 1.22, 1.48)]
        c_settled = build_consensus_words(dg, gl, audio_clock=10.0)
        s_settled = split_sentences(c_settled, "student")
        sent1 = s_settled[0]
        self.assertEqual(sent1["sentence_id"], "sent:student:1.000")
        rev1_id = sent1["id"]

        # Simulate feedback cache
        feedback_cache = {}
        dispatches = []

        # Dispatch revision 1
        if rev1_id not in feedback_cache:
            dispatches.append(rev1_id)
            feedback_cache[rev1_id] = {"status": "completed", "result": "good grammar"}

        self.assertEqual(len(dispatches), 1)

        # 3. Next render tick with identical text: cache hit, zero new dispatches
        if rev1_id not in feedback_cache:
            dispatches.append(rev1_id)
        self.assertEqual(len(dispatches), 1)

        # 4. Late AssemblyAI arrives with corroboration (text unchanged): zero new dispatches
        aai_corrob = [make_word("I", 1.01, 1.21), make_word("panic.", 1.21, 1.49)]
        c_aai_corrob = build_consensus_words(dg, gl, aai_words=aai_corrob, audio_clock=10.0)
        s_aai_corrob = split_sentences(c_aai_corrob, "student")
        self.assertEqual(s_aai_corrob[0]["id"], rev1_id)
        if s_aai_corrob[0]["id"] not in feedback_cache:
            dispatches.append(s_aai_corrob[0]["id"])
        self.assertEqual(len(dispatches), 1)

        # 5. Gladia/AAI revise word to "picnic" -> produces revision 2
        dg_picnic = [make_word("I", 1.0, 1.2), make_word("picnic.", 1.2, 1.5)]
        gl_picnic = [make_word("I", 1.02, 1.22), make_word("picnic.", 1.22, 1.48)]
        c_revised = build_consensus_words(dg_picnic, gl_picnic, audio_clock=10.0)
        s_revised = split_sentences(c_revised, "student")
        sent2 = s_revised[0]
        # Sentence ID is strictly identical
        self.assertEqual(sent2["sentence_id"], sent1["sentence_id"])
        # New revision ID is generated and dispatched once
        self.assertNotEqual(sent2["id"], rev1_id)
        if sent2["id"] not in feedback_cache:
            dispatches.append(sent2["id"])
            feedback_cache[sent2["id"]] = {"status": "completed", "result": "vocabulary note"}
        self.assertEqual(len(dispatches), 2)

    def test_invariant_6_failed_provider_does_not_block_others(self):
        dg = [make_word("resilience", 1.0, 1.4)]
        # Gladia and AAI are both inactive / empty
        consensus = build_consensus_words(dg, None, aai_words=None, gladia_active=False, assemblyai_active=False, audio_clock=10.0)
        self.assertEqual(len(consensus), 1)
        self.assertEqual(consensus[0]["word"], "resilience")
        self.assertEqual(consensus[0]["status"], "primary_only")

    def test_invariant_7_confidence_scores_not_interchangeable(self):
        from live_consensus import _score_candidate
        # Deepgram 0.90 vs Gladia 0.90 produce different heuristic scores due to model identity
        dg_score = _score_candidate({"word": "x", "is_final": True, "confidence": 0.90}, provider="deepgram")
        gl_score = _score_candidate({"word": "x", "is_final": True, "confidence": 0.90}, provider="gladia")
        self.assertNotEqual(dg_score, gl_score)

    def test_invariant_8_symmetric_two_of_three_status(self):
        # All 3 permutations of 2-of-3 agreement yield consensus
        c_dg_gl = build_consensus_words([make_word("cat", 1.0, 1.3)], [make_word("cat", 1.0, 1.3)], aai_words=[make_word("dog", 1.0, 1.3)], audio_clock=5.0)
        c_gl_aai = build_consensus_words([make_word("dog", 1.0, 1.3)], [make_word("cat", 1.0, 1.3)], aai_words=[make_word("cat", 1.0, 1.3)], audio_clock=5.0)
        c_dg_aai = build_consensus_words([make_word("cat", 1.0, 1.3)], [make_word("dog", 1.0, 1.3)], aai_words=[make_word("cat", 1.0, 1.3)], audio_clock=5.0)
        self.assertEqual(c_dg_gl[0]["status"], "consensus")
        self.assertEqual(c_gl_aai[0]["status"], "consensus")
        self.assertEqual(c_dg_aai[0]["status"], "consensus")

    def test_invariant_9_partitioner_drops_zero_words(self):
        from live_consensus import _subdivide_cluster_if_needed
        words = [make_word(f"w{i}", i * 0.2, i * 0.2 + 0.18) for i in range(160)]
        clusters = _subdivide_cluster_if_needed(words, words, words, max_words=80, max_duration_s=20.0)
        self.assertEqual(sum(len(c[0]) for c in clusters), 160)

    def test_invariant_10_repeated_words_aligned_monotonically(self):
        dg = [make_word("no", 1.0, 1.2), make_word("no", 1.3, 1.5), make_word("no.", 1.6, 1.8)]
        gl = [make_word("no", 1.01, 1.21), make_word("no", 1.31, 1.51), make_word("no", 1.61, 1.81)]
        consensus = build_consensus_words(dg, gl, audio_clock=5.0)
        self.assertEqual(len(consensus), 3)
        self.assertTrue(all(w["status"] == "consensus" for w in consensus))


class SingleProviderWordPreservationTests(unittest.TestCase):
    def test_dg_only_word_preserved(self):
        dg = [make_word("unique_dg", 1.0, 1.4)]
        consensus = build_consensus_words(dg, [], aai_words=[], audio_clock=5.0)
        self.assertEqual(len(consensus), 1)
        self.assertEqual(consensus[0]["word"], "unique_dg")
        self.assertEqual(consensus[0]["evidence"]["deepgram"]["word"], "unique_dg")

    def test_gladia_only_word_preserved(self):
        gl = [make_word("unique_gladia", 1.0, 1.4)]
        consensus = build_consensus_words([], gl, aai_words=[], audio_clock=5.0)
        self.assertEqual(len(consensus), 1)
        self.assertEqual(consensus[0]["word"], "unique_gladia")
        self.assertEqual(consensus[0]["evidence"]["gladia"]["word"], "unique_gladia")

    def test_assemblyai_only_word_preserved(self):
        aai = [make_word("unique_aai", 1.0, 1.4)]
        consensus = build_consensus_words([], [], aai_words=aai, audio_clock=5.0)
        self.assertEqual(len(consensus), 1)
        self.assertEqual(consensus[0]["word"], "unique_aai")
        self.assertEqual(consensus[0]["evidence"]["assemblyai"]["word"], "unique_aai")

    def test_dg_and_aai_agree_when_gladia_omits_word(self):
        dg = [make_word("I", 1.0, 1.2), make_word("really", 1.3, 1.6), make_word("want", 1.6, 1.9), make_word("to", 1.9, 2.1), make_word("go", 2.1, 2.3)]
        gl = [make_word("I", 1.0, 1.2), make_word("want", 1.4, 1.7), make_word("to", 1.7, 1.9), make_word("go", 1.9, 2.1)]
        aai = [make_word("I", 1.0, 1.2), make_word("really", 1.3, 1.6), make_word("want", 1.6, 1.9), make_word("to", 1.9, 2.1), make_word("go", 2.1, 2.3)]
        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=5.0)
        self.assertEqual([w["word"] for w in consensus], ["I", "really", "want", "to", "go"])
        self.assertEqual(consensus[1]["status"], "consensus")

    def test_dg_and_gladia_agree_when_aai_omits_word(self):
        dg = [make_word("I", 1.0, 1.2), make_word("really", 1.3, 1.6), make_word("want", 1.6, 1.9), make_word("to", 1.9, 2.1), make_word("go", 2.1, 2.3)]
        gl = [make_word("I", 1.0, 1.2), make_word("really", 1.3, 1.6), make_word("want", 1.6, 1.9), make_word("to", 1.9, 2.1), make_word("go", 2.1, 2.3)]
        aai = [make_word("I", 1.0, 1.2), make_word("want", 1.6, 1.9), make_word("to", 1.9, 2.1), make_word("go", 2.1, 2.3)]
        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=5.0)
        self.assertEqual([w["word"] for w in consensus], ["I", "really", "want", "to", "go"])
        self.assertEqual(consensus[1]["status"], "consensus")

    def test_gladia_and_aai_agree_when_dg_omits_word(self):
        dg = [make_word("I", 1.0, 1.2), make_word("want", 1.6, 1.9), make_word("to", 1.9, 2.1), make_word("go", 2.1, 2.3)]
        gl = [make_word("I", 1.0, 1.2), make_word("really", 1.3, 1.6), make_word("want", 1.6, 1.9), make_word("to", 1.9, 2.1), make_word("go", 2.1, 2.3)]
        aai = [make_word("I", 1.0, 1.2), make_word("really", 1.3, 1.6), make_word("want", 1.6, 1.9), make_word("to", 1.9, 2.1), make_word("go", 2.1, 2.3)]
        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=5.0)
        self.assertEqual([w["word"] for w in consensus], ["I", "really", "want", "to", "go"])
        self.assertEqual(consensus[1]["status"], "consensus")

    def test_aai_insertion_preserved_when_gladia_is_empty(self):
        dg = [make_word("I", 1.0, 1.2), make_word("want", 1.6, 1.9)]
        aai = [make_word("I", 1.0, 1.2), make_word("really", 1.3, 1.6), make_word("want", 1.6, 1.9)]
        consensus = build_consensus_words(dg, [], aai_words=aai, audio_clock=5.0)
        self.assertEqual([w["word"] for w in consensus], ["I", "really", "want"])

    def test_aai_insertion_preserved_when_dg_is_empty(self):
        gl = [make_word("I", 1.0, 1.2), make_word("want", 1.6, 1.9)]
        aai = [make_word("I", 1.0, 1.2), make_word("really", 1.3, 1.6), make_word("want", 1.6, 1.9)]
        consensus = build_consensus_words([], gl, aai_words=aai, audio_clock=5.0)
        self.assertEqual([w["word"] for w in consensus], ["I", "really", "want"])


class ConsecutiveNonOverlappingWordsTests(unittest.TestCase):
    def test_adjacent_non_overlapping_words_never_merged_into_substitution(self):
        # Deepgram heard "apple" (1.0-1.3), Gladia heard "banana" (1.3-1.6)
        dg = [make_word("apple", 1.0, 1.3)]
        gl = [make_word("banana", 1.3, 1.6)]
        consensus = build_consensus_words(dg, gl, aai_words=None, audio_clock=5.0)
        self.assertEqual(len(consensus), 2)
        self.assertEqual([w["word"] for w in consensus], ["apple", "banana"])

    def test_repeated_adjacent_identical_tokens_retained(self):
        # DG has "really really", GL has single "really", AAI has "really really"
        dg = [make_word("I", 1.0, 1.1), make_word("think", 1.1, 1.3), make_word("I", 1.3, 1.4), make_word("really", 1.4, 1.6), make_word("really", 1.6, 1.8), make_word("like", 1.8, 2.0), make_word("it", 2.0, 2.2)]
        gl = [make_word("I", 1.0, 1.1), make_word("think", 1.1, 1.3), make_word("I", 1.3, 1.4), make_word("really", 1.4, 1.6), make_word("like", 1.8, 2.0), make_word("it", 2.0, 2.2)]
        aai = [make_word("I", 1.0, 1.1), make_word("think", 1.1, 1.3), make_word("I", 1.3, 1.4), make_word("really", 1.4, 1.6), make_word("really", 1.6, 1.8), make_word("like", 1.8, 2.0), make_word("it", 2.0, 2.2)]
        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=5.0)
        self.assertEqual([w["word"] for w in consensus], ["I", "think", "I", "really", "really", "like", "it"])

    def test_short_function_words_and_articles_preserved(self):
        dg = [make_word("a", 1.0, 1.1), make_word("book", 1.1, 1.4)]
        gl = [make_word("the", 1.0, 1.1), make_word("book", 1.1, 1.4)]
        aai = [make_word("a", 1.0, 1.1), make_word("book", 1.1, 1.4)]
        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=5.0)
        self.assertEqual([w["word"] for w in consensus], ["a", "book"])

    def test_fillers_and_hesitations_preserved(self):
        dg = [make_word("um", 1.0, 1.2), make_word("I", 1.2, 1.4), make_word("agree", 1.4, 1.8)]
        gl = [make_word("I", 1.2, 1.4), make_word("agree", 1.4, 1.8)]
        aai = [make_word("um", 1.0, 1.2), make_word("I", 1.2, 1.4), make_word("agree", 1.4, 1.8)]
        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=5.0)
        self.assertEqual([w["word"] for w in consensus], ["um", "I", "agree"])


class WordRetentionHardInvariantTests(unittest.TestCase):
    def test_every_raw_input_word_mapped_to_consensus_evidence_with_zero_loss(self):
        dg = [make_word(f"dg_{i}", i * 0.4, i * 0.4 + 0.35) for i in range(10)]
        gl = [make_word(f"gl_{i}", i * 0.4 + 0.05, i * 0.4 + 0.38) for i in range(10)]
        aai = [make_word(f"aai_{i}", i * 0.4 + 0.02, i * 0.4 + 0.37) for i in range(10)]

        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=10.0)

        # Count total raw inputs
        total_raw_inputs = len(dg) + len(gl) + len(aai)

        # Count total raw words retained in evidence across all consensus items
        dg_in_evidence = sum(1 for w in consensus if w["evidence"].get("deepgram") is not None)
        gl_in_evidence = sum(1 for w in consensus if w["evidence"].get("gladia") is not None)
        aai_in_evidence = sum(1 for w in consensus if w["evidence"].get("assemblyai") is not None)
        total_retained_in_evidence = dg_in_evidence + gl_in_evidence + aai_in_evidence

        self.assertEqual(dg_in_evidence, len(dg))
        self.assertEqual(gl_in_evidence, len(gl))
        self.assertEqual(aai_in_evidence, len(aai))
        self.assertEqual(total_retained_in_evidence, total_raw_inputs)


class NonBlockingSettlementWordRetentionLifecycleTests(unittest.TestCase):
    def test_late_aai_confirm_lifecycle_preserves_word_exactly_once(self):
        # 1. DG emits "really" (1.3-1.6)
        dg = [make_word("I", 1.0, 1.2), make_word("really", 1.3, 1.6), make_word("want", 1.6, 1.9)]
        # 2. GL omits "really"
        gl = [make_word("I", 1.0, 1.2), make_word("want", 1.6, 1.9)]
        # 3. Consensus settles without AAI
        c1 = build_consensus_words(dg, gl, aai_words=[], audio_clock=5.0)
        words1 = [w["word"] for w in c1]
        self.assertEqual(words1, ["I", "really", "want"])
        self.assertEqual(c1[1]["status"], "uncertain")

        # 4. AAI arrives later and confirms "really"
        aai = [make_word("I", 1.0, 1.2), make_word("really", 1.3, 1.6), make_word("want", 1.6, 1.9)]
        c2 = build_consensus_words(dg, gl, aai_words=aai, audio_clock=5.0)
        words2 = [w["word"] for w in c2]
        self.assertEqual(words2, ["I", "really", "want"])
        # Word promoted to consensus, appears exactly once
        self.assertEqual(c2[1]["status"], "consensus")
        self.assertEqual(c2[1]["word_id"], c1[1]["word_id"])

    def test_late_aai_dispute_lifecycle_preserves_word_exactly_once(self):
        # 1. DG emits "really"
        dg = [make_word("I", 1.0, 1.2), make_word("really", 1.3, 1.6), make_word("want", 1.6, 1.9)]
        gl = [make_word("I", 1.0, 1.2), make_word("want", 1.6, 1.9)]
        c1 = build_consensus_words(dg, gl, aai_words=[], audio_clock=5.0)

        # 2. AAI arrives later with disagreement "truly" (1.3-1.6)
        aai = [make_word("I", 1.0, 1.2), make_word("truly", 1.3, 1.6), make_word("want", 1.6, 1.9)]
        c2 = build_consensus_words(dg, gl, aai_words=aai, audio_clock=5.0)
        words2 = [w["word"] for w in c2]
        # Word slot is retained (exactly 3 words), status remains uncertain
        self.assertEqual(len(words2), 3)
        self.assertEqual(c2[1]["status"], "uncertain")
        self.assertEqual(c2[1]["word_id"], c1[1]["word_id"])


class PunctuationConsensusTests(unittest.TestCase):
    def test_1_all_three_providers_produce_period(self):
        dg = [make_word("others.", 1.0, 1.4)]
        gl = [make_word("others.", 1.02, 1.38)]
        aai = [make_word("others.", 1.01, 1.39)]
        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=5.0)
        self.assertEqual(consensus[0]["word"], "others.")

    def test_2_only_dg_produces_period_others_omit(self):
        dg = [make_word("others.", 1.0, 1.4)]
        gl = [make_word("others", 1.02, 1.38)]
        aai = [make_word("others", 1.01, 1.39)]
        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=5.0)
        self.assertEqual(consensus[0]["word"], "others.")

    def test_3_only_gladia_produces_question_mark_others_omit(self):
        dg = [make_word("hobby", 1.0, 1.4)]
        gl = [make_word("hobby?", 1.02, 1.38)]
        aai = [make_word("hobby", 1.01, 1.39)]
        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=5.0)
        self.assertEqual(consensus[0]["word"], "hobby?")

    def test_4_only_aai_produces_exclamation_others_omit(self):
        dg = [make_word("question", 1.0, 1.4)]
        gl = [make_word("question", 1.02, 1.38)]
        aai = [make_word("question!", 1.01, 1.39)]
        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=5.0)
        self.assertEqual(consensus[0]["word"], "question!")

    def test_5_two_providers_produce_period_one_omits(self):
        dg = [make_word("others.", 1.0, 1.4)]
        gl = [make_word("others.", 1.02, 1.38)]
        aai = [make_word("others", 1.01, 1.39)]
        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=5.0)
        self.assertEqual(consensus[0]["word"], "others.")

    def test_6_providers_produce_different_terminal_punctuation(self):
        dg = [make_word("think.", 1.0, 1.4)]
        gl = [make_word("think?", 1.02, 1.38)]
        aai = [make_word("think", 1.01, 1.39)]
        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=5.0)
        self.assertEqual(consensus[0]["word"], "think?")

    def test_7_punctuation_arrives_late_updates_sentence_revision(self):
        # Step 1: DG emits "others" without punctuation
        dg1 = [make_word("others", 1.0, 1.4)]
        gl1 = [make_word("others", 1.0, 1.4)]
        c1 = build_consensus_words(dg1, gl1, audio_clock=5.0)
        s1 = split_sentences(c1, "student")
        self.assertEqual(s1[0]["text"], "others")
        rev1_id = s1[0]["id"]

        # Step 2: Gladia finalizes with late punctuation "others."
        gl2 = [make_word("others.", 1.0, 1.4)]
        c2 = build_consensus_words(dg1, gl2, audio_clock=5.0)
        s2 = split_sentences(c2, "student")
        self.assertEqual(s2[0]["text"], "others.")
        # Sentence ID stable, revision ID updated
        self.assertEqual(s2[0]["sentence_id"], s1[0]["sentence_id"])
        self.assertNotEqual(s2[0]["id"], rev1_id)

    def test_8_punctuation_differs_while_lexical_consensus_is_identical(self):
        dg = [make_word("Hello,", 1.0, 1.4)]
        gl = [make_word("Hello!", 1.02, 1.38)]
        aai = [make_word("hello", 1.01, 1.39)]
        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=5.0)
        self.assertEqual(consensus[0]["status"], "consensus")
        # Terminal exclamation takes precedence over clause comma
        self.assertEqual(consensus[0]["word"], "Hello!")

    def test_9_multiple_consecutive_sentences_split_accurately(self):
        dg = [make_word("First.", 1.0, 1.4), make_word("Second.", 1.6, 2.0)]
        gl = [make_word("First.", 1.0, 1.4), make_word("Second.", 1.6, 2.0)]
        consensus = build_consensus_words(dg, gl, audio_clock=5.0)
        sentences = split_sentences(consensus, "student")
        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0]["text"], "First.")
        self.assertEqual(sentences[1]["text"], "Second.")

    def test_10_long_unpunctuated_transcript_split_on_late_terminal_update(self):
        # Initial: long run without punctuation
        dg_init = [make_word("I", 1.0, 1.2), make_word("want", 1.2, 1.4), make_word("to", 1.4, 1.6), make_word("go", 1.6, 1.8), make_word("now", 1.8, 2.0)]
        gl_init = [make_word("I", 1.0, 1.2), make_word("want", 1.2, 1.4), make_word("to", 1.4, 1.6), make_word("go", 1.6, 1.8), make_word("now", 1.8, 2.0)]
        c_init = build_consensus_words(dg_init, gl_init, audio_clock=5.0)
        s_init = split_sentences(c_init, "student")
        self.assertEqual(len(s_init), 1)

        # Late update: Gladia finalizes sentence boundary at "go."
        gl_late = [make_word("I", 1.0, 1.2), make_word("want", 1.2, 1.4), make_word("to", 1.4, 1.6), make_word("go.", 1.6, 1.8), make_word("now", 1.8, 2.0)]
        c_late = build_consensus_words(dg_init, gl_late, audio_clock=5.0)
        s_late = split_sentences(c_late, "student")
        self.assertEqual(len(s_late), 2)
        self.assertEqual(s_late[0]["text"], "I want to go.")
        self.assertEqual(s_late[1]["text"], "now")

    def test_real_world_transcript_failure_pattern_splits_into_three_sentences(self):
        # User example:
        # 1. "Hope if people if people has a hobby, they have more opportunity to meet meet others."
        # 2. "So so I had a free they had free time for chatting with other people."
        # 3. "What about spending too much time on a hobby?"
        dg_tokens = "Hope if people if people has a hobby, they have more opportunity to meet meet others. So so I had a free they had free time for chatting with other people What about spending too much time on a hobby".split()
        gl_tokens = "Hope if people if people has a hobby they have more opportunity to meet meet others So so I had a free they had free time for chatting with other people. What about spending too much time on a hobby".split()
        aai_tokens = "Hope if people if people has a hobby they have more opportunity to meet meet others So so I had a free they had free time for chatting with other people What about spending too much time on a hobby?".split()

        t = 1.0
        dg_words = []
        for tok in dg_tokens:
            dg_words.append(make_word(tok, t, t + 0.2))
            t += 0.25

        t = 1.0
        gl_words = []
        for tok in gl_tokens:
            gl_words.append(make_word(tok, t, t + 0.2))
            t += 0.25

        t = 1.0
        aai_words = []
        for tok in aai_tokens:
            aai_words.append(make_word(tok, t, t + 0.2))
            t += 0.25

        consensus = build_consensus_words(dg_words, gl_words, aai_words=aai_words, audio_clock=20.0)
        sentences = split_sentences(consensus, "student")
        self.assertEqual(len(sentences), 3)
        self.assertTrue(sentences[0]["text"].endswith("others."))
        self.assertTrue(sentences[1]["text"].endswith("people."))
        self.assertTrue(sentences[2]["text"].endswith("hobby?"))


class EndToEndInvariantsTests(unittest.TestCase):
    """Hard end-to-end invariants governing full-pipeline transcription and sentence extraction."""

    def test_word_conservation_invariant_no_silent_deletion(self):
        """Every raw provider word must map to a visible consensus word, be merged as evidence, or be retained."""
        dg = [
            make_word("We", 1.0, 1.2),
            make_word("really", 1.2, 1.5),
            make_word("want", 1.5, 1.8),
            make_word("to", 1.8, 2.0),
            make_word("go.", 2.0, 2.3),
        ]
        gl = [
            make_word("We", 1.02, 1.22),
            make_word("want", 1.48, 1.78),  # Gladia dropped "really"
            make_word("to", 1.78, 1.98),
            make_word("go.", 1.98, 2.28),
        ]
        aai = [
            make_word("We", 1.01, 1.21),
            make_word("really", 1.21, 1.49),
            make_word("want", 1.49, 1.79),
            make_word("to", 1.79, 1.99),
            make_word("go.", 1.99, 2.29),
        ]

        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=10.0)
        consensus_norms = [normalize_token(w["word"]) for w in consensus]

        # Word "really" was heard by DG and AAI; it must be conserved in consensus
        self.assertIn("really", consensus_norms)
        self.assertEqual(len(consensus), 5)
        self.assertEqual(consensus_norms, ["we", "really", "want", "to", "go"])

        # Word conservation check: all distinct lexical evidence words must be accounted for
        for dg_w in dg:
            dg_norm = normalize_token(dg_w["word"])
            self.assertIn(dg_norm, consensus_norms, f"DG word {dg_w['word']} was silently deleted")

    def test_monotonicity_invariant_later_incomplete_snapshot_does_not_delete_words(self):
        """Once legitimate consensus words are emitted, a later partial or incomplete update cannot delete them."""
        # Initial complete stream
        dg_initial = [
            make_word("I", 1.0, 1.2),
            make_word("went", 1.2, 1.5),
            make_word("to", 1.5, 1.7),
            make_word("Paris.", 1.7, 2.1),
        ]
        c1 = build_consensus_words(dg_initial, audio_clock=5.0)
        self.assertEqual(len(c1), 4)

        # Later tick: Gladia is lagging and only finalized words up to 1.5s ("I", "went")
        gl_partial = [
            make_word("I", 1.02, 1.22),
            make_word("went", 1.22, 1.52),
        ]
        c2 = build_consensus_words(dg_initial, gl_words=gl_partial, audio_clock=5.0)
        self.assertEqual(len(c2), 4)
        c2_norms = [normalize_token(w["word"]) for w in c2]
        self.assertEqual(c2_norms, ["i", "went", "to", "paris"])

    def test_revision_safety_invariant_preserves_logical_identity(self):
        """A later update may revise punctuation or status, but preserves stable word identity and order."""
        dg = [make_word("hello", 1.0, 1.3)]
        c1 = build_consensus_words(dg, audio_clock=3.0)
        self.assertEqual(c1[0]["word_id"], "student:1.000")
        self.assertEqual(c1[0]["word"], "hello")

        # Gladia arrives with terminal period
        gl = [make_word("hello.", 1.02, 1.32)]
        c2 = build_consensus_words(dg, gl_words=gl, audio_clock=5.0)
        self.assertEqual(len(c2), 1)
        self.assertEqual(c2[0]["word_id"], "student:1.000")
        self.assertEqual(c2[0]["word"], "hello.")
        self.assertEqual(c2[0]["status"], "consensus")

    def test_punctuation_monotonicity_invariant(self):
        """Finalized terminal punctuation supported by a provider cannot disappear because another provider omits it."""
        dg = [make_word("Sure,", 1.0, 1.3), make_word("why", 1.3, 1.5), make_word("not?", 1.5, 1.8)]
        gl = [make_word("Sure", 1.0, 1.3), make_word("why", 1.3, 1.5), make_word("not", 1.5, 1.8)]  # Gladia omitted punctuation
        aai = [make_word("Sure", 1.0, 1.3), make_word("why", 1.3, 1.5), make_word("not", 1.5, 1.8)]

        consensus = build_consensus_words(dg, gl, aai_words=aai, audio_clock=5.0)
        self.assertEqual(consensus[0]["word"], "Sure,")
        self.assertEqual(consensus[2]["word"], "not?")

        sentences = split_sentences(consensus, "student")
        self.assertEqual(len(sentences), 1)
        self.assertEqual(sentences[0]["text"], "Sure, why not?")


if __name__ == "__main__":
    unittest.main()


