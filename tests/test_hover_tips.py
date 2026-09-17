"""Unit tests for hover tips, segment HTML rendering, and tooltip structures."""

import unittest
from app import _segment_html, _turn_html, _format_word_html


class TestHoverTipsRendering(unittest.TestCase):
    def test_flagged_sentence_segment_contains_fb_card_and_hover_classes(self):
        segment = {
            "id": "sent-1",
            "text": "Yesterday he go to school.",
            "html_text": "Yesterday he go to school.",
            "state": "flagged",
            "label": "verb form",
            "explanation": "Use 'went' for past events.",
            "quoted_span": "go",
            "replacement": "went",
            "rewrite": "Yesterday he went to school.",
            "sentence": "Yesterday he go to school.",
        }
        html_out = _segment_html(segment)
        self.assertIn('class="sent flagged"', html_out)
        self.assertIn('data-sent-id="sent-1"', html_out)
        self.assertIn('<span class="sent-text">Yesterday he go to school.</span>', html_out)
        self.assertIn('<span class="fb-card">', html_out)
        self.assertIn('<span class="fb-label">verb form</span>', html_out)
        self.assertIn('<span class="fb-exp">Use &#x27;went&#x27; for past events.</span>', html_out)
        self.assertIn('<span class="fb-fix">went</span>', html_out)

    def test_failed_sentence_segment_contains_error_card(self):
        segment = {
            "id": "sent-2",
            "text": "I like apple.",
            "html_text": "I like apple.",
            "state": "failed",
            "error": "Model timeout.",
        }
        html_out = _segment_html(segment)
        self.assertIn('class="sent failed"', html_out)
        self.assertIn('data-sent-id="sent-2"', html_out)
        self.assertIn('<span class="fb-label failed">not checked</span>', html_out)
        self.assertIn('<span class="fb-exp">Model timeout.</span>', html_out)

    def test_turn_html_wraps_segments_in_speaker_block(self):
        segment = {
            "id": "sent-1",
            "text": "He go home.",
            "html_text": "He go home.",
            "state": "flagged",
            "label": "grammar",
            "explanation": "Use went.",
            "quoted_span": "go",
            "replacement": "went",
        }
        turn_html = _turn_html("student", 12.5, "He go home.", segments=[segment])
        self.assertIn('class="turn student"', turn_html)
        self.assertIn('[00:12]', turn_html)
        self.assertIn('class="speaker">Student:</span>', turn_html)
        self.assertIn('class="sent flagged"', turn_html)

    def test_turn_html_unprocessed_deepseek_star(self):
        turn_html_pending = _turn_html("student", 12.5, "He go home.", unprocessed_deepseek=True)
        self.assertIn('[00:12]<span class="deepseek-unprocessed" title="DeepSeek processing pending">*</span>', turn_html_pending)
        turn_html_done = _turn_html("student", 12.5, "He go home.", unprocessed_deepseek=False)
        self.assertNotIn('*', turn_html_done)

    def test_uncertain_word_tooltip_html(self):
        word_item = {
            "word": "weather",
            "status": "uncertain",
            "evidence": {
                "deepgram": {"word": "weather"},
                "gladia": {"word": "whether"},
            },
        }
        word_html = _format_word_html(word_item)
        self.assertIn('class="uncertain-word"', word_html)
        self.assertIn('title="Deepgram: weather | Gladia: whether"', word_html)
        self.assertIn('>weather</span>', word_html)


if __name__ == "__main__":
    unittest.main()
