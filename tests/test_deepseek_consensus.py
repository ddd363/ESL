"""Unit tests for DeepSeek utterance consensus."""

import unittest
from unittest.mock import MagicMock

from deepseek_consensus import (
    build_utterance_candidates_payload,
    extract_provider_candidates_for_turn,
    make_utterance_key,
    parse_deepseek_utterance_reply,
    run_deepseek_utterance_job,
)


class TestDeepSeekConsensus(unittest.TestCase):
    def test_make_utterance_key(self):
        key1 = make_utterance_key("student", 1.25, 3.50, "hello world")
        key2 = make_utterance_key("student", 1.25, 3.50, "hello world")
        key3 = make_utterance_key("student", 1.25, 3.50, "different text")
        self.assertEqual(key1, key2)
        self.assertNotEqual(key1, key3)
        self.assertTrue(key1.startswith("utt:student:1.25:3.50:"))

    def test_extract_provider_candidates_for_turn(self):
        words = [
            {"word": "Hi", "start": 0.0, "end": 0.4, "confidence": 0.95, "source": "student"},
            {"word": "there", "start": 0.5, "end": 0.9, "confidence": 0.88, "source": "student"},
            {"word": "friend", "start": 1.0, "end": 1.4, "confidence": 0.72, "source": "student"},
            {"word": "teacher_word", "start": 0.5, "end": 0.9, "confidence": 0.99, "source": "teacher"},
            {"word": "way_later", "start": 10.0, "end": 10.5, "confidence": 0.90, "source": "student"},
        ]
        extracted = extract_provider_candidates_for_turn(words, "student", start=0.4, end=1.0, padding=0.2)
        extracted_words = [w["word"] for w in extracted]
        self.assertIn("Hi", extracted_words)
        self.assertIn("there", extracted_words)
        self.assertIn("friend", extracted_words)
        self.assertNotIn("teacher_word", extracted_words)
        self.assertNotIn("way_later", extracted_words)

    def test_build_utterance_candidates_payload(self):
        turn = {
            "source": "student",
            "start": 2.0,
            "end": 4.0,
            "text": "I went to picnic",
        }
        dg_words = [{"word": "I", "start": 2.0, "end": 2.2, "confidence": 0.95, "source": "student"},
                    {"word": "went", "start": 2.3, "end": 2.6, "confidence": 0.90, "source": "student"},
                    {"word": "picnic", "start": 3.0, "end": 3.5, "confidence": 0.65, "source": "student"}]
        gl_words = [{"word": "I", "start": 2.0, "end": 2.2, "confidence": 0.98, "source": "student"},
                    {"word": "went", "start": 2.3, "end": 2.6, "confidence": 0.94, "source": "student"},
                    {"word": "panic", "start": 3.0, "end": 3.5, "confidence": 0.89, "source": "student"}]
        aai_words = [{"word": "I", "start": 2.0, "end": 2.2, "confidence": 0.96, "source": "student"},
                     {"word": "went", "start": 2.3, "end": 2.6, "confidence": 0.92, "source": "student"},
                     {"word": "panic", "start": 3.0, "end": 3.5, "confidence": 0.91, "source": "student"}]

        payload = build_utterance_candidates_payload(turn, dg_words, gl_words, aai_words)
        self.assertEqual(payload["speaker"], "Student")
        self.assertEqual(payload["heuristic_consensus_text"], "I went to picnic")
        self.assertEqual(len(payload["candidates"]["deepgram"]), 3)
        self.assertEqual(len(payload["candidates"]["gladia"]), 3)
        self.assertEqual(len(payload["candidates"]["assemblyai"]), 3)

    def test_parse_deepseek_utterance_reply(self):
        # 1. Clean JSON
        res1 = parse_deepseek_utterance_reply('{"utterance": "I went to panic."}', fallback_text="fallback")
        self.assertEqual(res1, "I went to panic.")

        # 2. Markdown-fenced JSON
        res2 = parse_deepseek_utterance_reply('```json\n{"utterance": "She don\'t know."}\n```', fallback_text="fallback")
        self.assertEqual(res2, "She don't know.")

        # 3. Malformed JSON with plain string fallback
        res3 = parse_deepseek_utterance_reply("Plain text utterance directly returned.", fallback_text="fallback")
        self.assertEqual(res3, "Plain text utterance directly returned.")

        # 4. Empty or invalid
        res4 = parse_deepseek_utterance_reply("", fallback_text="fallback")
        self.assertEqual(res4, "fallback")

    def test_run_deepseek_utterance_job_success(self):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = '{"utterance": "I went to panic."}'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        payload = {"heuristic_consensus_text": "I went picnic"}
        result = run_deepseek_utterance_job("utt:1", mock_client, payload)
        self.assertTrue(result["ok"])
        self.assertEqual(result["utterance"], "I went to panic.")
        self.assertIsNone(result["error"])

    def test_run_deepseek_utterance_job_error_fallback(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API Rate Limit")

        payload = {"heuristic_consensus_text": "Heuristic fallback text"}
        result = run_deepseek_utterance_job("utt:2", mock_client, payload)
        self.assertFalse(result["ok"])
        self.assertEqual(result["utterance"], "Heuristic fallback text")
        self.assertIn("API Rate Limit", result["error"])


if __name__ == "__main__":
    unittest.main()
