import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "evaluate_transcription.py"
SPEC = importlib.util.spec_from_file_location("evaluate_transcription", MODULE_PATH)
evaluate_transcription = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate_transcription)


class WordErrorCountsTests(unittest.TestCase):
    def test_counts_substitution_deletion_and_insertion(self):
        self.assertEqual(
            evaluate_transcription.word_error_counts("I went home", "I go home"),
            {
                "reference_words": 3,
                "errors": 1,
                "substitutions": 1,
                "deletions": 0,
                "insertions": 0,
            },
        )
        self.assertEqual(evaluate_transcription.word_error_counts("one two", "one")["deletions"], 1)
        self.assertEqual(evaluate_transcription.word_error_counts("one", "one two")["insertions"], 1)

    def test_summary_uses_only_manually_labeled_asr_substitutions(self):
        manifest = [{"clip_id": "clip-1", "cohort": "esl", "reference_transcript": "I went home"}]
        transcripts = {"clip-1": "I go home"}
        annotations = [
            {
                "clip_id": "clip-1",
                "source": "deepgram",
                "error_type": "asr_substitution",
                "presented_as_learner_error": "true",
            },
            {
                "clip_id": "clip-1",
                "source": "deepgram",
                "error_type": "genuine_learner_error",
                "presented_as_learner_error": "false",
            },
        ]

        summary = evaluate_transcription.summarize_source(
            manifest, transcripts, annotations, "deepgram"
        )

        self.assertEqual(summary["overall"]["wer"], 1 / 3)
        self.assertEqual(summary["false_learner_error_rate"], 1.0)
        self.assertEqual(summary["manually_annotated_asr_substitutions"], 1)


if __name__ == "__main__":
    unittest.main()