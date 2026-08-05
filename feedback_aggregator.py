"""Pure aggregation for readable live ESL feedback."""

from __future__ import annotations

import difflib
import math
import re
from collections import Counter

try:
    from rapidfuzz.fuzz import token_set_ratio
except ImportError:  # Keep the live pane usable if an optional dependency is absent.
    token_set_ratio = None


DEFAULT_SEVERITY_WEIGHTS = {
    "grammar": 1.0,
    "coherence": 1.0,
    "collocation": 0.9,
    "word choice": 0.85,
    "informal register": 0.75,
    "disfluency": 0.35,
    "asr noise": 0.15,
}
HIGH_SEVERITY_CATEGORIES = {"grammar", "coherence"}
ISSUE_CATEGORIES = frozenset(DEFAULT_SEVERITY_WEIGHTS)
STRENGTH_CATEGORIES = frozenset({
    "cohesion",
    "collocation",
    "discourse marker",
    "fluency",
    "lexical range",
    "narrative control",
    "register",
})
POSITIVE_OBSERVATION_WORDS = {
    "accurate", "clear", "coherent", "effective", "good", "natural",
    "strong", "varied", "well", "fluent", "appropriate",
}


def _issue_category(value):
    text = str(value or "").strip().lower()
    aliases = {
        "lexical": "word choice",
        "vocabulary": "word choice",
        "fluency": "disfluency",
        "filler": "disfluency",
        "register": "informal register",
    }
    return aliases.get(text, text or "word choice")


def _strength_category(value):
    text = str(value or "").strip().lower()
    aliases = {
        "coherence": "cohesion",
        "disfluency": "fluency",
        "lexical": "lexical range",
        "word choice": "collocation",
    }
    category = aliases.get(text, text or "fluency")
    return category if category in STRENGTH_CATEGORIES else "fluency"


def normalize_phrase(value):
    """Normalise a phrase for clustering without changing the displayed quote."""
    text = str(value or "").lower()
    text = re.sub(r"\((?:[^()]*)\)", " ", text)
    tokens = re.findall(r"[a-z0-9']+", text)
    # A lightweight stem handles the common inflectional near-duplicates without
    # adding a heavyweight NLP runtime to the live app.
    stems = [re.sub(r"(ing|ed|es|s)$", "", token) if len(token) > 4 else token for token in tokens]
    return " ".join(stems)


def _is_noise_phrase(value):
    return bool(re.fullmatch(r"\s*\([^()]+\)\s*", str(value or "")))


def _is_positive_observation(value):
    tokens = set(re.findall(r"[a-z]+", str(value or "").lower()))
    return bool(tokens & POSITIVE_OBSERVATION_WORDS)


def _strength_parts(value):
    text = str(value or "").strip()
    if " | " in text:
        phrase, detail = text.split(" | ", 1)
        category_match = re.search(r"category:\s*([^|]+)", detail, re.I)
        reason_match = re.search(r"why:\s*(.+)", detail, re.I)
        return (
            phrase.strip(),
            (category_match.group(1).strip().lower() if category_match else "fluency"),
            (reason_match.group(1).strip() if reason_match else "This makes the message easy to follow."),
        )
    lower = text.lower()
    if any(token in lower for token in ("then", "because", "after", "before", "story")):
        return text, "narrative control", "This develops the story in a clear sequence."
    if any(token in lower for token in ("however", "although", "therefore", "first", "finally")):
        return text, "cohesion", "This links ideas clearly for the listener."
    if any(token in lower for token in ("idiom", "vivid", "specific", "varied")):
        return text, "lexical range", "This adds precise, varied language."
    return text, "fluency", "This makes the message easy to follow."


def _overlaps_low_confidence_word(phrase, turn, threshold):
    if not isinstance(turn, dict):
        return False
    phrase_tokens = set(re.findall(r"[a-z0-9']+", str(phrase or "").lower()))
    for word in turn.get("words") or []:
        if not isinstance(word, dict):
            continue
        token = str(word.get("text") or word.get("word") or "").lower()
        confidence = word.get("confidence")
        if (
            token
            and token in phrase_tokens
            and isinstance(confidence, (int, float))
            and confidence < threshold
        ):
            return True
    return False


def _evidence_span(turn, phrase):
    """Return the first exact word span for a cited phrase in one turn."""
    if not isinstance(turn, dict):
        return None
    phrase_tokens = re.findall(r"[a-z0-9']+", str(phrase or "").lower())
    words = turn.get("words") or []
    tokens = [
        re.findall(r"[a-z0-9']+", str(word.get("text") or word.get("word") or "").lower())
        if isinstance(word, dict) else []
        for word in words
    ]
    normalized_words = [parts[0] if len(parts) == 1 else "" for parts in tokens]
    for start_index in range(len(normalized_words) - len(phrase_tokens) + 1):
        end_index = start_index + len(phrase_tokens)
        if normalized_words[start_index:end_index] != phrase_tokens:
            continue
        start_word, end_word = words[start_index], words[end_index - 1]
        return {
            "turn_id": turn.get("id", turn.get("index")),
            "word_start": start_index,
            "word_end": end_index,
            "t_start": start_word.get("start"),
            "t_end": end_word.get("end"),
        }
    return None


def _similar(left, right):
    if not left or not right:
        return False
    left_tokens, right_tokens = left.split(), right.split()
    if min(len(left_tokens), len(right_tokens)) >= 3:
        shorter, longer = (left_tokens, right_tokens) if len(left_tokens) <= len(right_tokens) else (right_tokens, left_tokens)
        if any(longer[index:index + len(shorter)] == shorter for index in range(len(longer) - len(shorter) + 1)):
            return True
    if token_set_ratio is not None:
        return token_set_ratio(left, right) >= 85
    if set(left.split()) == set(right.split()):
        return True
    return difflib.SequenceMatcher(None, left, right).ratio() >= 0.85


def _turn_number(item, fallback):
    turn = item.get("turn") if isinstance(item, dict) else None
    if not isinstance(turn, dict):
        return fallback
    return int(turn.get("id", turn.get("index", fallback))) + 1


def _turn_time(item):
    turn = item.get("turn") if isinstance(item, dict) else None
    return turn.get("start") if isinstance(turn, dict) else None


def _cluster_feedback(
    clusters, phrase, category, suggestion, turn_number, timestamp, weights, *, polarity, evidence=None
):
    original_phrase = str(phrase or "")
    phrase = re.sub(r"[()]", "", original_phrase).strip()
    normal = normalize_phrase(phrase)
    category = "asr noise" if _is_noise_phrase(original_phrase) else category
    for cluster in clusters:
        if (
            cluster["polarity"] == polarity
            and cluster["category"] == category
            and _similar(cluster["normalised"], normal)
        ):
            cluster["count"] += 1
            cluster["turn_ids"].append(turn_number)
            cluster["last_seen"] = timestamp
            cluster["suggestions"][suggestion] += 1
            cluster["suggestion"] = cluster["suggestions"].most_common(1)[0][0]
            cluster["score"] = cluster["count"] * cluster["severity"]
            cluster["priority_score"] = cluster["severity"] * math.log1p(cluster["count"])
            if evidence:
                cluster["evidence"].append(evidence)
            return
    severity = float(weights.get(category, 0.65))
    clusters.append({
        "phrase": phrase,
        "normalised": normal,
        "category": category,
        "suggestion": suggestion,
        "suggestions": Counter({suggestion: 1}),
        "count": 1,
        "first_seen": timestamp,
        "last_seen": timestamp,
        "turn_ids": [turn_number],
        "severity": severity,
        "score": severity,
        "priority_score": severity * math.log(2),
        "polarity": polarity,
        "evidence": [evidence] if evidence else [],
    })


def _ordered(clusters, previous_order, hysteresis):
    ranked = sorted(clusters, key=lambda item: item["priority_score"], reverse=True)
    if not previous_order:
        return ranked
    positions = {phrase: index for index, phrase in enumerate(previous_order)}
    for index in range(1, len(ranked)):
        current, prior = ranked[index], ranked[index - 1]
        if positions.get(current["phrase"], index) < positions.get(prior["phrase"], index - 1):
            if current["priority_score"] < prior["priority_score"] * (1 + hysteresis):
                ranked[index - 1], ranked[index] = current, prior
    return ranked


def aggregate_feedback(
    turn_results,
    *,
    severity_weights=None,
    previous_order=(),
    force_priority_phrases=(),
    exclude_phrases=(),
    low_confidence_threshold=0.75,
    hysteresis=0.15,
):
    """Return stable, ranked clusters from worker result records.

    Bad or partial model payloads are ignored individually, never allowed to
    prevent feedback for other completed turns.
    """
    weights = {**DEFAULT_SEVERITY_WEIGHTS, **(severity_weights or {})}
    forced = {normalize_phrase(phrase) for phrase in force_priority_phrases}
    excluded = {normalize_phrase(phrase) for phrase in exclude_phrases}
    errors, strengths = [], []
    records = turn_results.values() if isinstance(turn_results, dict) else (turn_results or [])
    for fallback, item in enumerate(records):
        if not isinstance(item, dict) or item.get("status") != "complete":
            continue
        payload = item.get("analysis")
        if not isinstance(payload, dict):
            continue
        turn_number, timestamp = _turn_number(item, fallback), _turn_time(item)
        for error in payload.get("errors") or []:
            if not isinstance(error, dict):
                continue
            phrase = str(error.get("phrase") or "").strip()
            if phrase:
                category = (
                    "asr noise"
                    if _overlaps_low_confidence_word(phrase, item.get("turn"), low_confidence_threshold)
                    else _issue_category(error.get("category"))
                )
                _cluster_feedback(
                    errors,
                    phrase,
                    category,
                    str(error.get("suggestion") or "").strip(),
                    turn_number,
                    timestamp,
                    weights,
                    polarity="issue",
                    evidence=_evidence_span(item.get("turn"), phrase),
                )
        positive_observations = [
            text for text in payload.get("observations") or []
            if _is_positive_observation(text)
        ]
        for text in list(payload.get("highlights") or []) + positive_observations:
            phrase, category, reason = _strength_parts(text)
            if phrase:
                category = _strength_category(category)
                _cluster_feedback(
                    strengths,
                    phrase,
                    category,
                    reason,
                    turn_number,
                    timestamp,
                    {category: 0.8},
                    polarity="strength",
                    evidence=_evidence_span(item.get("turn"), phrase),
                )

    promoted = [
        item for item in errors
        if item["category"] != "asr noise"
        and (
            item["count"] >= 2
            or item["category"] in HIGH_SEVERITY_CATEGORIES
            or item["normalised"] in forced
        )
        and item["normalised"] not in excluded
    ]
    ranked_priorities = _ordered(promoted, previous_order, hysteresis)
    priorities = ranked_priorities[:3]
    strengths = sorted(strengths, key=lambda item: item["count"], reverse=True)[:3]
    emerging = [
        item for item in _ordered(errors, previous_order, hysteresis)
        if item["category"] != "asr noise" and item not in promoted
    ]
    return {
        "priorities": priorities,
        "priority_total": len(ranked_priorities),
        "priority_overflow": ranked_priorities[3:],
        "emerging": emerging,
        "emerging_count": len(emerging),
        "noise_count": sum(item["count"] for item in errors if item["category"] == "asr noise"),
        "strengths": strengths,
    }