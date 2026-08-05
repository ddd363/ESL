import unittest

from feedback_aggregator import aggregate_feedback


def result(turn_id, errors=(), highlights=(), observations=()):
    return {"status": "complete", "turn": {"id": turn_id, "start": turn_id * 10}, "analysis": {"errors": list(errors), "highlights": list(highlights), "observations": list(observations)}}


class FeedbackAggregatorTests(unittest.TestCase):
    def test_merges_near_identical_phrases(self):
        summary = aggregate_feedback([
            result(0, [{"phrase": "I am live action witnessing", "category": "grammar", "suggestion": "I am watching live action."}]),
            result(1, [{"phrase": "I am witnessing live action", "category": "grammar", "suggestion": "I am watching live action."}]),
        ])
        self.assertEqual(summary["priorities"][0]["count"], 2)

    def test_merges_contained_priority_phrases(self):
        summary = aggregate_feedback([
            result(0, [{"phrase": "Of applied science or technology it proceeds to grow", "category": "grammar", "suggestion": "It proceeds to grow."}]),
            result(1, [{"phrase": "it proceeds to grow", "category": "grammar", "suggestion": "It proceeds to grow."}]),
        ])
        self.assertEqual(len(summary["priorities"]), 1)
        self.assertEqual(summary["priorities"][0]["count"], 2)

    def test_severity_promotes_grammar_before_disfluency(self):
        summary = aggregate_feedback([
            result(0, [{"phrase": "one kid's hurt", "category": "grammar", "suggestion": "one kid is hurt"}]),
            result(1, [{"phrase": "um", "category": "disfluency", "suggestion": "pause"}]),
            result(2, [{"phrase": "um", "category": "disfluency", "suggestion": "pause"}]),
        ])
        self.assertEqual(summary["priorities"][0]["category"], "grammar")

    def test_promotion_hysteresis_and_bad_payloads(self):
        data = [result(0, [{"phrase": "we are gonna", "category": "informal register", "suggestion": "we are going to"}]), {"status": "complete", "analysis": "bad"}]
        self.assertEqual(aggregate_feedback(data)["priorities"], [])
        data.append(result(1, [{"phrase": "we are gonna", "category": "informal register", "suggestion": "we are going to"}], ["Clear story sequence."]))
        summary = aggregate_feedback(data, previous_order=["other item"])
        self.assertEqual(summary["priorities"][0]["count"], 2)
        self.assertEqual(len(summary["strengths"]), 1)

    def test_noise_never_promotes_and_non_positive_observations_are_ignored(self):
        summary = aggregate_feedback([
            result(0, [{"phrase": "(And)", "category": "grammar", "suggestion": ""}]),
            result(1, [{"phrase": "(and)", "category": "grammar", "suggestion": ""}]),
            result(2, observations=["Uses the phrase live action."],),
        ])
        self.assertEqual(summary["priorities"], [])
        self.assertEqual(summary["noise_count"], 2)
        self.assertEqual(summary["strengths"], [])

    def test_low_confidence_phrase_is_routed_to_noise(self):
        record = result(0, [{"phrase": "What (network) is he appertaining to?", "category": "word choice", "suggestion": "net worth"}])
        record["turn"]["words"] = [{"text": "network", "confidence": 0.42}]
        summary = aggregate_feedback([record])
        self.assertEqual(summary["priorities"], [])
        self.assertEqual(summary["noise_count"], 1)

    def test_strength_categories_never_use_issue_taxonomy(self):
        summary = aggregate_feedback([
            result(0, highlights=["being in the public eye | category: disfluency | why: This is specific vocabulary."])
        ])
        strength = summary["strengths"][0]
        self.assertEqual(strength["polarity"], "strength")
        self.assertEqual(strength["category"], "fluency")

    def test_priority_carries_exact_word_level_evidence(self):
        record = result(
            0,
            [{"phrase": "the fame", "category": "grammar", "suggestion": "fame"}],
        )
        record["turn"]["words"] = [
            {"text": "the", "start": 37.8, "end": 38.0, "confidence": 0.99},
            {"text": "fame", "start": 38.0, "end": 38.3, "confidence": 0.99},
        ]
        summary = aggregate_feedback([record])
        evidence = summary["priorities"][0]["evidence"][0]
        self.assertEqual(evidence["word_start"], 0)
        self.assertEqual(evidence["word_end"], 2)
        self.assertEqual(evidence["t_start"], 37.8)
        self.assertEqual(evidence["t_end"], 38.3)

    def test_teacher_can_promote_an_emerging_pattern(self):
        data = [
            result(0, [{"phrase": "we are gonna", "category": "informal register", "suggestion": "we are going to"}])
        ]
        self.assertEqual(aggregate_feedback(data)["priorities"], [])
        summary = aggregate_feedback(data, force_priority_phrases=["we are gonna"])
        self.assertEqual(summary["priorities"][0]["phrase"], "we are gonna")

    def test_teacher_rejection_suppresses_a_priority(self):
        data = [
            result(0, [{"phrase": "we are gonna", "category": "grammar", "suggestion": "we are going to"}])
        ]
        self.assertEqual(len(aggregate_feedback(data)["priorities"]), 1)
        self.assertEqual(
            aggregate_feedback(data, exclude_phrases=["we are gonna"])["priorities"],
            [],
        )


if __name__ == "__main__":
    unittest.main()