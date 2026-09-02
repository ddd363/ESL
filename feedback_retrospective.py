"""Retrospective sentence-by-sentence DeepSeek feedback processing.

Pure functions for offline/retrospective prompt testing, snapshot queues,
atomic persistence with run provenance, and export formatting.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from live_consensus import normalize_token
from live_sentences import (
    low_confidence_words,
    split_sentences,
)

SCHEMA_VERSION = 3


def _is_valid_word_list(data: Any) -> bool:
    """True if data contains a non-empty list of valid word dictionaries."""
    if not isinstance(data, list) or not data:
        return False
    return any(isinstance(w, dict) and bool(w.get("word") or w.get("text")) for w in data)


def resolve_lesson_words(lesson_dir: str) -> List[Dict[str, Any]]:
    """Resolve words for a lesson folder with safe fallback.

    Checks words_consensus.json first; if missing or empty/invalid, falls back
    to words.json.
    """
    if not lesson_dir or not os.path.isdir(lesson_dir):
        return []

    # 1. Check words_consensus.json
    con_path = os.path.join(lesson_dir, "words_consensus.json")
    if os.path.exists(con_path):
        try:
            with open(con_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            words = payload.get("words", []) if isinstance(payload, dict) else payload
            if _is_valid_word_list(words):
                return [w for w in words if isinstance(w, dict)]
        except Exception:
            pass

    # 2. Fallback to words.json
    words_path = os.path.join(lesson_dir, "words.json")
    if os.path.exists(words_path):
        try:
            with open(words_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            words = payload.get("words", []) if isinstance(payload, dict) else payload
            if _is_valid_word_list(words):
                return [w for w in words if isinstance(w, dict)]
        except Exception:
            pass

    return []


def build_sentence_payload(sentence: Dict[str, Any], turns: List[Dict[str, Any]],
                           source_labels: Optional[Dict[str, str]] = None,
                           low_confidence_threshold: float = 0.70) -> Dict[str, Any]:
    """Build exact live-feedback payload for one sentence.

    Context provides the immediate preceding teacher turn before this sentence.
    """
    labels = source_labels or {"student": "Student", "teacher": "Teacher"}
    sentence_start = sentence.get("start") or 0.0
    context = []
    for turn in reversed(turns):
        turn_start = turn.get("start") or 0.0
        turn_source = turn.get("source")
        if turn_start >= sentence_start or turn_source != "teacher":
            continue
        context.append({
            "speaker": labels.get(turn_source, str(turn_source).title()),
            "text": turn.get("text", ""),
        })
        break

    source = sentence.get("source", "student")
    return {
        "speaker": labels.get(source, str(source).title()),
        "sentence": sentence.get("text", ""),
        "context": context,
        "uncertain_words": low_confidence_words(sentence, low_confidence_threshold),
    }


def create_retrospective_snapshot(
    words: List[Dict[str, Any]],
    turns: List[Dict[str, Any]],
    system_prompt: str,
    model: str,
    max_calls: Optional[int] = None,
    source: str = "student",
    min_words: int = 2,
    low_confidence_threshold: float = 0.70,
    source_labels: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Create an immutable snapshot queue for retrospective analysis at click time."""
    run_id = f"run_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    created_at = datetime.now(timezone.utc).isoformat()
    sentences = split_sentences(words, source)

    items: List[Dict[str, Any]] = []
    eligible_count = 0

    for idx, sentence in enumerate(sentences, 1):
        sentence_id = sentence["id"]
        text = sentence.get("text", "")
        word_count = sentence.get("word_count", len(sentence.get("words", [])))
        start = sentence.get("start", 0.0)
        end = sentence.get("end", start)
        payload = build_sentence_payload(
            sentence, turns, source_labels=source_labels,
            low_confidence_threshold=low_confidence_threshold
        )

        statuses = {w.get("status") for w in sentence.get("words", []) if isinstance(w, dict)}
        
        item: Dict[str, Any] = {
            "index": idx,
            "sentence_id": sentence_id,
            "start": start,
            "end": end,
            "text": text,
            "word_count": word_count,
            "payload": payload,
            "attempts": 0,
            "status": "pending",
            "result": None,
            "error": None,
        }

        if "uncertain" in statuses:
            item["status"] = "skipped"
            item["result"] = {"status": "skipped", "reason": "uncertain_asr"}
        elif word_count < min_words:
            item["status"] = "skipped"
            item["result"] = {"status": "skipped", "reason": "too_short"}
        else:
            eligible_count += 1
            if max_calls is not None and eligible_count > max_calls:
                item["status"] = "omitted_cap"
                item["result"] = {"status": "omitted", "reason": f"exceeded_cap_{max_calls}"}
            else:
                item["status"] = "pending"

        items.append(item)

    return {
        "version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": created_at,
        "system_prompt": system_prompt,
        "model": model,
        "source": source,
        "max_calls": max_calls,
        "total_sentences": len(sentences),
        "eligible_sentences": eligible_count,
        "items": items,
        "feedback": {
            item["sentence_id"]: item["result"]
            for item in items if item["result"] is not None
        },
        "is_complete": all(item["status"] in ("ok", "issue", "skipped", "omitted_cap", "error") for item in items),
    }


def merge_retrospective_result(
    snapshot: Dict[str, Any],
    sentence_id: str,
    job_result: Dict[str, Any],
    run_id: str,
    max_attempts: int = 3,
) -> Tuple[Dict[str, Any], bool]:
    """Merge an async job result into the retrospective snapshot.

    Rejects stale results if run_id does not match the active run.
    Implements retry semantics up to max_attempts.
    Returns (updated_snapshot, was_accepted).
    """
    if not snapshot or snapshot.get("run_id") != run_id:
        return snapshot, False

    items = list(snapshot.get("items", []))
    feedback = dict(snapshot.get("feedback", {}))
    found = False

    for item in items:
        if item.get("sentence_id") != sentence_id:
            continue

        found = True
        attempts = int(item.get("attempts", 0)) + 1
        item["attempts"] = attempts

        if job_result.get("ok"):
            verdict = job_result.get("verdict", "ok")
            item["status"] = verdict  # "ok" or "issue"
            item["error"] = None
            item["result"] = {
                "status": "ok",
                "verdict": verdict,
                "label": job_result.get("label", ""),
                "explanation": job_result.get("explanation", ""),
                "quoted_span": job_result.get("quoted_span", ""),
                "replacement": job_result.get("replacement", ""),
                "attempts": attempts,
            }
            feedback[sentence_id] = item["result"]
        else:
            err_msg = job_result.get("error") or "Analysis failed."
            item["error"] = err_msg
            if attempts < max_attempts:
                item["status"] = "retry"
                item["result"] = {
                    "status": "retry",
                    "error": err_msg,
                    "attempts": attempts,
                }
                feedback[sentence_id] = item["result"]
            else:
                item["status"] = "error"
                item["result"] = {
                    "status": "error",
                    "error": err_msg,
                    "attempts": attempts,
                }
                feedback[sentence_id] = item["result"]
        break

    if not found:
        return snapshot, False

    is_complete = all(
        item.get("status") in ("ok", "issue", "skipped", "omitted_cap", "error")
        for item in items
    )

    updated = dict(snapshot)
    updated["items"] = items
    updated["feedback"] = feedback
    updated["is_complete"] = is_complete
    return updated, True


def atomic_save_feedback(
    lesson_dir: str,
    snapshot_or_feedback: Dict[str, Any],
    windows: Optional[List[List[Any]]] = None,
) -> bool:
    """Atomically write feedback with provenance metadata to live_feedback.json."""
    if not lesson_dir or not os.path.isdir(lesson_dir):
        return False

    if "run_id" in snapshot_or_feedback:
        # Full snapshot format
        raw_items = snapshot_or_feedback.get("items") or snapshot_or_feedback.get("sentences") or []
        items_payload = [
            {
                "index": item.get("index"),
                "sentence_id": item.get("sentence_id"),
                "start": item.get("start"),
                "end": item.get("end"),
                "text": item.get("text"),
                "word_count": item.get("word_count"),
                "payload": item.get("payload"),
                "status": item.get("status"),
                "attempts": item.get("attempts"),
                "result": item.get("result"),
                "error": item.get("error"),
            }
            for item in raw_items
        ]
        payload = {
            "version": SCHEMA_VERSION,
            "run_id": snapshot_or_feedback.get("run_id"),
            "timestamp": snapshot_or_feedback.get("created_at") or datetime.now(timezone.utc).isoformat(),
            "model": snapshot_or_feedback.get("model", ""),
            "system_prompt": snapshot_or_feedback.get("system_prompt", ""),
            "source": snapshot_or_feedback.get("source", "student"),
            "max_calls": snapshot_or_feedback.get("max_calls"),
            "total_sentences": snapshot_or_feedback.get("total_sentences", len(items_payload)),
            "eligible_sentences": snapshot_or_feedback.get("eligible_sentences"),
            "windows": windows if windows is not None else [[0.0, None]],
            "feedback": snapshot_or_feedback.get("feedback", {}),
            "items": items_payload,
            "sentences": items_payload,
            "is_complete": snapshot_or_feedback.get("is_complete", True),
        }
    else:
        # Simple dict fallback
        payload = {
            "version": SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "windows": windows if windows is not None else [[0.0, None]],
            "feedback": snapshot_or_feedback,
        }

    target_path = os.path.join(lesson_dir, "live_feedback.json")
    temp_dir = lesson_dir
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=temp_dir, prefix="live_feedback_", suffix=".tmp", delete=False, encoding="utf-8"
        ) as tmp:
            json.dump(payload, tmp, ensure_ascii=False, indent=2)
            temp_name = tmp.name
        os.replace(temp_name, target_path)
        return True
    except Exception:
        return False


def load_lesson_feedback(lesson_dir: str) -> Tuple[Dict[str, Any], List[List[Any]], Optional[Dict[str, Any]]]:
    """Cached live-feedback results, windows, and optional retrospective snapshot."""
    if not lesson_dir or not os.path.isdir(lesson_dir):
        return {}, [], None
    try:
        with open(os.path.join(lesson_dir, "live_feedback.json"), "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}, [], None
    if not isinstance(payload, dict):
        return {}, [], None
    feedback = payload.get("feedback")
    windows = payload.get("windows")

    snapshot = None
    items = payload.get("items") or payload.get("sentences")
    if isinstance(items, list) and items and payload.get("run_id"):
        snapshot = {
            "version": payload.get("version", SCHEMA_VERSION),
            "run_id": payload.get("run_id"),
            "created_at": payload.get("timestamp") or payload.get("created_at"),
            "system_prompt": payload.get("system_prompt", ""),
            "model": payload.get("model", ""),
            "source": payload.get("source", "student"),
            "max_calls": payload.get("max_calls"),
            "total_sentences": payload.get("total_sentences", len(items)),
            "eligible_sentences": payload.get("eligible_sentences", len(items)),
            "items": items,
            "feedback": feedback if isinstance(feedback, dict) else {},
            "is_complete": payload.get("is_complete", True),
        }
    return (
        feedback if isinstance(feedback, dict) else {},
        [list(w) for w in windows if isinstance(w, (list, tuple))] if isinstance(windows, list) else [],
        snapshot,
    )


def format_inspection_markdown(snapshot: Dict[str, Any]) -> str:
    """Format complete sentence-by-sentence analysis log for agent prompting review."""
    lines: List[str] = []
    run_id = snapshot.get("run_id", "N/A")
    created = snapshot.get("created_at", "N/A")
    model = snapshot.get("model", "N/A")
    prompt = snapshot.get("system_prompt", "").strip()

    lines.append(f"# DeepSeek Sentence-by-Sentence Analysis Log")
    lines.append(f"- **Run ID**: `{run_id}`")
    lines.append(f"- **Timestamp**: `{created}`")
    lines.append(f"- **Model**: `{model}`")
    lines.append(f"- **Total Sentences**: {snapshot.get('total_sentences', 0)}")
    lines.append("")
    lines.append("## System Prompt")
    lines.append("```text")
    lines.append(prompt)
    lines.append("```")
    lines.append("")
    lines.append("## Sentence-by-Sentence Results")
    lines.append("")

    for item in snapshot.get("items", []):
        idx = item.get("index", 0)
        start = item.get("start", 0.0)
        mins = int(start // 60)
        secs = int(start % 60)
        clock = f"{mins:02d}:{secs:02d}"
        text = item.get("text", "")
        status = item.get("status", "unknown")
        payload = item.get("payload") or {}
        context = payload.get("context") or []
        uncertain = payload.get("uncertain_words") or []
        result = item.get("result") or {}

        lines.append(f"### Sentence #{idx} [{clock}] — Status: `{status.upper()}`")
        lines.append(f"- **Student Text**: \"{text}\"")
        if context:
            ctx_text = " | ".join(f"{c.get('speaker')}: {c.get('text')}" for c in context)
            lines.append(f"- **Teacher Context**: {ctx_text}")
        if uncertain:
            lines.append(f"- **Uncertain Words**: `{', '.join(uncertain)}`")

        if status == "issue":
            lines.append(f"- **Verdict**: `ISSUE`")
            lines.append(f"- **Error Label**: `{result.get('label', '')}`")
            lines.append(f"- **Quoted Span**: \"{result.get('quoted_span', '')}\"")
            lines.append(f"- **Replacement**: \"{result.get('replacement', '')}\"")
            lines.append(f"- **Explanation**: {result.get('explanation', '')}")
        elif status == "ok":
            lines.append(f"- **Verdict**: `OK (clean)`")
        elif status == "skipped":
            lines.append(f"- **Reason Skipped**: `{result.get('reason', 'too short / uncertain ASR')}`")
        elif status == "omitted_cap":
            lines.append(f"- **Reason Omitted**: `Exceeded analysis cap`")
        elif status in ("error", "failed"):
            lines.append(f"- **Error**: {item.get('error') or result.get('error', 'API error')}")
        elif status in ("pending", "retry"):
            lines.append(f"- **State**: `In progress / Retrying (attempt {item.get('attempts', 0)})`")

        lines.append("<details><summary>Exact JSON Payload Sent</summary>")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(payload, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("</details>")
        lines.append("")

    return "\n".join(lines)


def format_inspection_json(snapshot: Dict[str, Any]) -> str:
    """Format snapshot as structured JSON for export."""
    return json.dumps(snapshot, ensure_ascii=False, indent=2)
