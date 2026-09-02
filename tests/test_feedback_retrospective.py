"""Unit tests for feedback_retrospective.py."""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from feedback_retrospective import (
    atomic_save_feedback,
    build_sentence_payload,
    create_retrospective_snapshot,
    format_inspection_json,
    format_inspection_markdown,
    load_lesson_feedback,
    merge_retrospective_result,
    resolve_lesson_words,
)


class TestResolveLessonWords(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_prefers_consensus_when_valid(self):
        consensus_data = {
            "version": 2,
            "words": [
                {"source": "student", "word": "Hello", "start": 1.0, "end": 1.5, "confidence": 0.95},
                {"source": "student", "word": "world.", "start": 1.6, "end": 2.0, "confidence": 0.98},
            ]
        }
        words_data = {
            "version": 2,
            "words": [
                {"source": "student", "word": "Fallback", "start": 1.0, "end": 1.5, "confidence": 0.50},
            ]
        }
        with open(os.path.join(self.test_dir, "words_consensus.json"), "w", encoding="utf-8") as f:
            json.dump(consensus_data, f)
        with open(os.path.join(self.test_dir, "words.json"), "w", encoding="utf-8") as f:
            json.dump(words_data, f)

        resolved = resolve_lesson_words(self.test_dir)
        self.assertEqual(len(resolved), 2)
        self.assertEqual(resolved[0]["word"], "Hello")

    def test_falls_back_to_words_when_consensus_empty(self):
        consensus_data = {"version": 2, "words": []}
        words_data = {
            "version": 2,
            "words": [
                {"source": "student", "word": "Fallback", "start": 1.0, "end": 1.5, "confidence": 0.90},
            ]
        }
        with open(os.path.join(self.test_dir, "words_consensus.json"), "w", encoding="utf-8") as f:
            json.dump(consensus_data, f)
        with open(os.path.join(self.test_dir, "words.json"), "w", encoding="utf-8") as f:
            json.dump(words_data, f)

        resolved = resolve_lesson_words(self.test_dir)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["word"], "Fallback")

    def test_returns_empty_when_both_missing_or_corrupt(self):
        with open(os.path.join(self.test_dir, "words_consensus.json"), "w", encoding="utf-8") as f:
            f.write("{corrupt json")
        with open(os.path.join(self.test_dir, "words.json"), "w", encoding="utf-8") as f:
            f.write("{corrupt json")

        resolved = resolve_lesson_words(self.test_dir)
        self.assertEqual(resolved, [])


class TestCreateRetrospectiveSnapshot(unittest.TestCase):
    def test_snapshot_creation_and_filtering(self):
        words = [
            {"source": "teacher", "word": "How", "start": 0.5, "end": 0.8},
            {"source": "teacher", "word": "are", "start": 0.8, "end": 1.0},
            {"source": "teacher", "word": "you?", "start": 1.0, "end": 1.2},
            {"source": "student", "word": "I", "start": 2.0, "end": 2.2, "confidence": 0.95},
            {"source": "student", "word": "is", "start": 2.2, "end": 2.4, "confidence": 0.95},
            {"source": "student", "word": "fine.", "start": 2.4, "end": 2.8, "confidence": 0.95},
            # Single word utterance (too short -> skipped)
            {"source": "student", "word": "Yeah.", "start": 5.0, "end": 5.3, "confidence": 0.95},
            # Sentence with uncertain ASR word
            {"source": "student", "word": "I", "start": 7.0, "end": 7.2, "confidence": 0.95},
            {"source": "student", "word": "like", "start": 7.2, "end": 7.4, "confidence": 0.95},
            {"source": "student", "word": "apples.", "start": 7.4, "end": 7.8, "confidence": 0.40},
        ]
        turns = [
            {"source": "teacher", "start": 0.5, "end": 1.2, "text": "How are you?"},
            {"source": "student", "start": 2.0, "end": 2.8, "text": "I is fine."},
            {"source": "student", "start": 5.0, "end": 5.3, "text": "Yeah."},
            {"source": "student", "start": 7.0, "end": 7.8, "text": "I like apples."},
        ]

        snapshot = create_retrospective_snapshot(
            words=words,
            turns=turns,
            system_prompt="Test prompt",
            model="deepseek-chat",
            min_words=2,
            low_confidence_threshold=0.70,
        )

        self.assertTrue(snapshot["run_id"].startswith("run_"))
        self.assertEqual(snapshot["total_sentences"], 3)
        self.assertEqual(len(snapshot["items"]), 3)

        item1 = snapshot["items"][0]
        self.assertEqual(item1["text"], "I is fine.")
        self.assertEqual(item1["status"], "pending")
        self.assertEqual(len(item1["payload"]["context"]), 1)
        self.assertEqual(item1["payload"]["context"][0]["text"], "How are you?")

        item2 = snapshot["items"][1]
        self.assertEqual(item2["text"], "Yeah.")
        self.assertEqual(item2["status"], "skipped")
        self.assertEqual(item2["result"]["reason"], "too_short")

        item3 = snapshot["items"][2]
        self.assertEqual(item3["text"], "I like apples.")
        self.assertEqual(item3["status"], "pending")
        self.assertEqual(item3["payload"]["uncertain_words"], ["apples"])

    def test_cap_omission(self):
        words = [
            {"source": "student", "word": "First", "start": 1.0, "end": 1.2},
            {"source": "student", "word": "sentence.", "start": 1.2, "end": 1.5},
            {"source": "student", "word": "Second", "start": 3.0, "end": 3.2},
            {"source": "student", "word": "sentence.", "start": 3.2, "end": 3.5},
            {"source": "student", "word": "Third", "start": 5.0, "end": 5.2},
            {"source": "student", "word": "sentence.", "start": 5.2, "end": 5.5},
        ]
        turns = []
        snapshot = create_retrospective_snapshot(
            words=words,
            turns=turns,
            system_prompt="Test",
            model="deepseek-chat",
            max_calls=2,
        )
        self.assertEqual(snapshot["items"][0]["status"], "pending")
        self.assertEqual(snapshot["items"][1]["status"], "pending")
        self.assertEqual(snapshot["items"][2]["status"], "omitted_cap")


class TestMergeRetrospectiveResult(unittest.TestCase):
    def setUp(self):
        words = [
            {"source": "student", "word": "I", "start": 1.0, "end": 1.2},
            {"source": "student", "word": "goes.", "start": 1.2, "end": 1.5},
        ]
        self.snapshot = create_retrospective_snapshot(
            words=words, turns=[], system_prompt="Prompt", model="model"
        )
        self.run_id = self.snapshot["run_id"]
        self.sentence_id = self.snapshot["items"][0]["sentence_id"]

    def test_rejects_stale_run_id(self):
        result = {"ok": True, "verdict": "ok", "label": "", "explanation": "", "quoted_span": "", "replacement": ""}
        updated, accepted = merge_retrospective_result(
            self.snapshot, self.sentence_id, result, run_id="stale_run_123"
        )
        self.assertFalse(accepted)
        self.assertEqual(updated["items"][0]["status"], "pending")

    def test_accepts_valid_issue_result(self):
        result = {
            "ok": True,
            "verdict": "issue",
            "label": "subject-verb agreement",
            "explanation": "Use 'go' with 'I'.",
            "quoted_span": "goes",
            "replacement": "go",
        }
        updated, accepted = merge_retrospective_result(
            self.snapshot, self.sentence_id, result, run_id=self.run_id
        )
        self.assertTrue(accepted)
        self.assertEqual(updated["items"][0]["status"], "issue")
        self.assertEqual(updated["items"][0]["attempts"], 1)
        self.assertEqual(updated["feedback"][self.sentence_id]["label"], "subject-verb agreement")
        self.assertTrue(updated["is_complete"])

    def test_retry_transitions_and_terminal_failure(self):
        err_res = {"ok": False, "error": "DeepSeek rate limit"}

        # Attempt 1 -> retry
        updated, _ = merge_retrospective_result(
            self.snapshot, self.sentence_id, err_res, run_id=self.run_id, max_attempts=3
        )
        self.assertEqual(updated["items"][0]["status"], "retry")
        self.assertEqual(updated["items"][0]["attempts"], 1)
        self.assertFalse(updated["is_complete"])

        # Attempt 2 -> retry
        updated, _ = merge_retrospective_result(
            updated, self.sentence_id, err_res, run_id=self.run_id, max_attempts=3
        )
        self.assertEqual(updated["items"][0]["status"], "retry")
        self.assertEqual(updated["items"][0]["attempts"], 2)

        # Attempt 3 -> terminal error
        updated, _ = merge_retrospective_result(
            updated, self.sentence_id, err_res, run_id=self.run_id, max_attempts=3
        )
        self.assertEqual(updated["items"][0]["status"], "error")
        self.assertEqual(updated["items"][0]["attempts"], 3)
        self.assertTrue(updated["is_complete"])


class TestAtomicSaveAndExport(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_atomic_save_and_formatters(self):
        words = [
            {"source": "student", "word": "She", "start": 1.0, "end": 1.2},
            {"source": "student", "word": "like", "start": 1.2, "end": 1.4},
            {"source": "student", "word": "tea.", "start": 1.4, "end": 1.8},
        ]
        snapshot = create_retrospective_snapshot(
            words=words, turns=[], system_prompt="Test system prompt", model="deepseek-chat"
        )
        sentence_id = snapshot["items"][0]["sentence_id"]
        result = {
            "ok": True,
            "verdict": "issue",
            "label": "subject-verb agreement",
            "explanation": "Use 'likes' for third person.",
            "quoted_span": "like",
            "replacement": "likes",
        }
        snapshot, _ = merge_retrospective_result(snapshot, sentence_id, result, snapshot["run_id"])

        success = atomic_save_feedback(self.test_dir, snapshot)
        self.assertTrue(success)

        target_file = os.path.join(self.test_dir, "live_feedback.json")
        self.assertTrue(os.path.exists(target_file))

        with open(target_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(saved["version"], 3)
        self.assertEqual(saved["run_id"], snapshot["run_id"])
        self.assertEqual(saved["system_prompt"], "Test system prompt")
        self.assertEqual(len(saved["sentences"]), 1)

        # Test export formatters
        md = format_inspection_markdown(snapshot)
        self.assertIn("DeepSeek Sentence-by-Sentence Analysis Log", md)
        self.assertIn("Test system prompt", md)
        self.assertIn("subject-verb agreement", md)

        js = format_inspection_json(snapshot)
        parsed = json.loads(js)
        self.assertEqual(parsed["run_id"], snapshot["run_id"])


class TestLessonFeedbackPersistenceReload(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_save_and_reload_preserves_provenance_and_payloads(self):
        words = [
            {"source": "teacher", "word": "Where", "start": 0.5, "end": 0.8},
            {"source": "teacher", "word": "did", "start": 0.8, "end": 1.0},
            {"source": "teacher", "word": "you", "start": 1.0, "end": 1.2},
            {"source": "teacher", "word": "go?", "start": 1.2, "end": 1.5},
            {"source": "student", "word": "I", "start": 2.0, "end": 2.2, "confidence": 0.95},
            {"source": "student", "word": "go", "start": 2.2, "end": 2.4, "confidence": 0.95},
            {"source": "student", "word": "to", "start": 2.4, "end": 2.6, "confidence": 0.95},
            {"source": "student", "word": "school", "start": 2.6, "end": 2.8, "confidence": 0.95},
            {"source": "student", "word": "yesterday.", "start": 2.8, "end": 3.2, "confidence": 0.95},
        ]
        turns = [
            {"source": "teacher", "start": 0.5, "end": 1.5, "text": "Where did you go?"},
            {"source": "student", "start": 2.0, "end": 3.2, "text": "I go to school yesterday."},
        ]

        orig_prompt = "You are an ESL reviewer testing a specific custom prompt v2."
        orig_model = "deepseek-chat"

        snapshot = create_retrospective_snapshot(
            words=words,
            turns=turns,
            system_prompt=orig_prompt,
            model=orig_model,
            max_calls=400,
        )
        self.assertEqual(len(snapshot["items"]), 1)
        item = snapshot["items"][0]
        sentence_id = item["sentence_id"]

        result = {
            "ok": True,
            "verdict": "issue",
            "label": "past tense",
            "explanation": "Use past tense 'went' with 'yesterday'.",
            "quoted_span": "go",
            "replacement": "went",
        }
        snapshot, accepted = merge_retrospective_result(
            snapshot, sentence_id, result, snapshot["run_id"]
        )
        self.assertTrue(accepted)

        # Atomically save to test directory
        saved = atomic_save_feedback(self.test_dir, snapshot, windows=[[0.0, None]])
        self.assertTrue(saved)

        # Reload through app's load_lesson_feedback
        reloaded_feedback, reloaded_windows, reloaded_snapshot = load_lesson_feedback(self.test_dir)

        self.assertIsNotNone(reloaded_snapshot)
        self.assertEqual(reloaded_snapshot["run_id"], snapshot["run_id"])
        self.assertEqual(reloaded_snapshot["system_prompt"], orig_prompt)
        self.assertEqual(reloaded_snapshot["model"], orig_model)
        self.assertEqual(len(reloaded_snapshot["items"]), 1)

        reloaded_item = reloaded_snapshot["items"][0]
        self.assertEqual(reloaded_item["text"], "I go to school yesterday.")
        self.assertEqual(reloaded_item["status"], "issue")
        self.assertEqual(reloaded_item["payload"]["context"][0]["text"], "Where did you go?")
        self.assertEqual(reloaded_item["result"]["quoted_span"], "go")
        self.assertEqual(reloaded_item["result"]["replacement"], "went")

        # Verify formatting using reloaded snapshot retains full provenance
        md = format_inspection_markdown(reloaded_snapshot)
        self.assertIn(orig_prompt, md)
        self.assertIn("Where did you go?", md)
        self.assertIn("past tense", md)


class TestFutureCancellationAndStaleDrain(unittest.TestCase):
    def test_future_cancellation(self):
        fut1 = MagicMock()
        fut2 = MagicMock()
        futures = {"sent1": fut1, "sent2": fut2}

        # Simulate cancellation logic in trigger_reanalyse_all
        for fut in list(futures.values()):
            fut.cancel()
        futures.clear()

        fut1.cancel.assert_called_once()
        fut2.cancel.assert_called_once()
        self.assertEqual(len(futures), 0)

    def test_drain_ignores_cancelled_and_stale_futures(self):
        snapshot = create_retrospective_snapshot(
            words=[{"source": "student", "word": "Hello", "start": 1.0, "end": 1.5}],
            turns=[],
            system_prompt="P",
            model="M",
        )
        active_run_id = snapshot["run_id"]

        # Cancelled future
        cancelled_fut = MagicMock()
        cancelled_fut.done.return_value = True
        cancelled_fut.cancelled.return_value = True

        # Stale future from older run
        stale_fut = MagicMock()
        stale_fut.done.return_value = True
        stale_fut.cancelled.return_value = False
        stale_fut.result.return_value = {
            "ok": True,
            "verdict": "ok",
            "run_id": "old_run_999",
        }

        # Valid future
        valid_fut = MagicMock()
        valid_fut.done.return_value = True
        valid_fut.cancelled.return_value = False
        sentence_id = snapshot["items"][0]["sentence_id"]
        valid_fut.result.return_value = {
            "ok": True,
            "verdict": "ok",
            "run_id": active_run_id,
            "label": "",
            "explanation": "",
            "quoted_span": "",
            "replacement": "",
        }

        futures = {
            "sent_canc": cancelled_fut,
            "sent_stale": stale_fut,
            sentence_id: valid_fut,
        }

        # Simulate drain logic
        live_feedback = {}
        for s_id in list(futures.keys()):
            fut = futures.pop(s_id)
            if fut.cancelled():
                continue
            res = fut.result()
            if res.get("run_id") and res.get("run_id") != active_run_id:
                continue
            snapshot, accepted = merge_retrospective_result(
                snapshot, s_id, res, active_run_id
            )
            if accepted:
                live_feedback[s_id] = snapshot["feedback"].get(s_id)

        self.assertEqual(len(futures), 0)
        self.assertIn(sentence_id, live_feedback)
        self.assertNotIn("sent_canc", live_feedback)
        self.assertNotIn("sent_stale", live_feedback)
        self.assertEqual(snapshot["items"][0]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
