import json
import unittest

from app import (
    DEFAULT_LIVE_FEEDBACK_PROMPT,
    derive_span_replacement,
    is_legacy_feedback_prompt,
    parse_live_feedback_reply,
)


class TestLiveFeedbackParser(unittest.TestCase):
    def test_modern_contract_valid_issue(self):
        raw = json.dumps({
            "verdict": "issue",
            "label": "verb form",
            "explanation": "Use 'went' for past tense.",
            "quoted_span": "go",
            "replacement": "went",
        })
        sentence = "Yesterday he go to school."
        res = parse_live_feedback_reply(raw, sentence=sentence)
        self.assertEqual(res["verdict"], "issue")
        self.assertEqual(res["label"], "verb form")
        self.assertEqual(res["quoted_span"], "go")
        self.assertEqual(res["replacement"], "went")
        self.assertEqual(res["explanation"], "Use 'went' for past tense.")

    def test_legacy_contract_single_token_diff_derivation(self):
        raw = json.dumps({
            "verdict": "issue",
            "label": "verb form",
            "explanation": "Use past tense.",
            "rewrites": ["Yesterday he went to school."],
        })
        sentence = "Yesterday he go to school."
        res = parse_live_feedback_reply(raw, sentence=sentence)
        self.assertEqual(res["verdict"], "issue")
        self.assertEqual(res["label"], "verb form")
        self.assertEqual(res["quoted_span"], "go")
        self.assertEqual(res["replacement"], "went")
        self.assertEqual(res["rewrite"], "Yesterday he went to school.")

    def test_legacy_contract_broad_rewrite_preserves_issue(self):
        raw = json.dumps({
            "verdict": "issue",
            "label": "word order",
            "explanation": "The phrase 'about a hobby' is misplaced; it should follow 'discuss'.",
            "rewrites": ["We have been talking about a hobby that you enjoy and I would like to discuss one or two general questions."],
        })
        sentence = "We've been about a hobby that you enjoy to discuss with you one or two more general questions related to this."
        res = parse_live_feedback_reply(raw, sentence=sentence)
        self.assertEqual(res["verdict"], "issue")
        self.assertEqual(res["label"], "word order")
        # Broad rewrite should NOT fabricate a misleading single-span replacement
        self.assertEqual(res["quoted_span"], "")
        self.assertEqual(res["replacement"], "")
        self.assertEqual(res["rewrite"], "We have been talking about a hobby that you enjoy and I would like to discuss one or two general questions.")
        self.assertIn("misplaced", res["explanation"])

    def test_issue_with_explanation_only(self):
        raw = json.dumps({
            "verdict": "issue",
            "label": "missing verb",
            "explanation": "The sentence is missing a main verb.",
            "quoted_span": "",
            "replacement": "",
        })
        sentence = "He a very good student."
        res = parse_live_feedback_reply(raw, sentence=sentence)
        self.assertEqual(res["verdict"], "issue")
        self.assertEqual(res["label"], "missing verb")
        self.assertEqual(res["quoted_span"], "")
        self.assertEqual(res["replacement"], "")
        self.assertEqual(res["explanation"], "The sentence is missing a main verb.")

    def test_contradictory_ok_with_explanation_is_sanitized(self):
        raw = json.dumps({
            "verdict": "ok",
            "label": "word order",
            "explanation": "The phrase 'about a hobby' is misplaced.",
            "quoted_span": "",
            "replacement": "",
        })
        sentence = "We've been about a hobby that you enjoy to discuss with you."
        res = parse_live_feedback_reply(raw, sentence=sentence)
        self.assertEqual(res["verdict"], "ok")
        self.assertEqual(res["label"], "")
        self.assertEqual(res["explanation"], "")
        self.assertEqual(res["quoted_span"], "")
        self.assertEqual(res["replacement"], "")
        self.assertEqual(res["rewrite"], "")
        self.assertEqual(res["rewrites"], [])

    def test_hallucinated_quoted_span_clears_span_but_keeps_issue(self):
        raw = json.dumps({
            "verdict": "issue",
            "label": "vocabulary",
            "explanation": "Incorrect word choice.",
            "quoted_span": "completely_unrelated_word",
            "replacement": "correct_word",
        })
        sentence = "The student spoke clearly today."
        res = parse_live_feedback_reply(raw, sentence=sentence)
        self.assertEqual(res["verdict"], "issue")
        self.assertEqual(res["quoted_span"], "")
        self.assertEqual(res["replacement"], "")
        self.assertEqual(res["explanation"], "Incorrect word choice.")

    def test_empty_issue_payload_downgrades_to_sanitized_ok(self):
        raw = json.dumps({
            "verdict": "issue",
            "label": "",
            "explanation": "",
            "quoted_span": "",
            "replacement": "",
            "rewrites": [],
        })
        sentence = "This is a sentence."
        res = parse_live_feedback_reply(raw, sentence=sentence)
        self.assertEqual(res["verdict"], "ok")
        self.assertEqual(res["label"], "")
        self.assertEqual(res["explanation"], "")

    def test_derive_span_replacement_helpers(self):
        # Single token substitution
        self.assertEqual(
            derive_span_replacement("he go home", "he went home"),
            ("go", "went"),
        )
        # Deletion
        self.assertEqual(
            derive_span_replacement("I bought a new clothes", "I bought new clothes"),
            ("a", ""),
        )
        # Multi-word substitution (<= 5 words)
        self.assertEqual(
            derive_span_replacement("he do volunteer work", "he volunteers"),
            ("do volunteer work", "volunteers"),
        )
        # Broad rewrite / no match (> 5 words)
        self.assertIsNone(
            derive_span_replacement(
                "one two three four five six seven eight",
                "alpha beta gamma delta epsilon zeta eta theta",
            )
        )
        # Empty / None inputs
        self.assertIsNone(derive_span_replacement("", "test"))
        self.assertIsNone(derive_span_replacement("test", ""))

    def test_is_legacy_feedback_prompt(self):
        legacy = '{"verdict": "ok" | "issue", "label": "...", "explanation": "...", "rewrites": ["...", "..."]}'
        modern = DEFAULT_LIVE_FEEDBACK_PROMPT
        self.assertTrue(is_legacy_feedback_prompt(legacy))
        self.assertFalse(is_legacy_feedback_prompt(modern))


if __name__ == "__main__":
    unittest.main()
