"""DeepSeek-powered utterance consensus engine.

Reconciles candidate word streams and confidence probabilities from Deepgram,
Gladia, and AssemblyAI into the single most likely verbatim utterance using
DeepSeek's fast model (deepseek-chat).

Independent of DeepSeek language feedback: this module focuses strictly on
acoustic ASR reconciliation while preserving ESL learner errors.
"""

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEEPSEEK_UTTERANCE_MODEL = os.environ.get("DEEPSEEK_UTTERANCE_MODEL", "deepseek-chat")
UTTERANCE_WINDOW_PADDING_S = 0.5

DEEPSEEK_UTTERANCE_SYSTEM_PROMPT = (
    "You are an expert speech recognition reconciler for English as a Second Language (ESL) lessons.\n\n"
    "You are given candidate speech recognition (ASR) words with confidence probabilities (0.0 to 1.0) "
    "from three independent providers (Deepgram, Gladia, and AssemblyAI) for a single speaker turn.\n\n"
    "Your objective: Determine the single most likely verbatim utterance actually spoken by the speaker.\n\n"
    "CRITICAL RULES:\n"
    "1. WEIGH THE EVIDENCE: Compare words across all three providers. Where providers agree or where one provider "
    "has significantly higher confidence and acoustic plausibility in context, select that word. "
    "Notice phonetic confusions (e.g. homophones, syllable mishearings, or truncation) and resolve them based on context.\n"
    "2. PRESERVE ESL LEARNER ERRORS: The student is an English learner. DO NOT grammar-correct the student. "
    "If the student made a grammar, tense, word choice, or agreement mistake (e.g. 'I have went', 'she don't know', 'he go yesterday'), "
    "you MUST preserve their mistake verbatim. Only resolve ASR acoustic mishearings, NEVER learner language errors.\n"
    "3. NATURAL PUNCTUATION & CAPITALIZATION: Output appropriate English capitalization and trailing punctuation "
    "(period, question mark, or comma as indicated by the flow).\n"
    "4. NO EXPLANATIONS: Output ONLY a JSON object with the single key 'utterance' containing the resolved text string.\n"
    'Example JSON output: {"utterance": "I went to the store yesterday."}'
)


def make_utterance_key(source: str, start: float, end: float, text: str) -> str:
    """Generate a stable, unique identifier for an utterance turn."""
    h = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:10]
    return f"utt:{source}:{start:.2f}:{end:.2f}:{h}"


def extract_provider_candidates_for_turn(
    words: List[Dict[str, Any]],
    source: str,
    start: float,
    end: float,
    padding: float = UTTERANCE_WINDOW_PADDING_S,
) -> List[Dict[str, Any]]:
    """Extract and format candidate words from a provider within the turn's timeframe."""
    candidates = []
    window_start = max(0.0, start - padding)
    window_end = end + padding

    for w in words:
        w_src = w.get("source", "student")
        if w_src != source:
            continue
        w_start = w.get("start", 0.0)
        w_end = w.get("end", w_start)
        # Check time overlap
        if w_end >= window_start and w_start <= window_end:
            conf = w.get("confidence")
            candidates.append({
                "word": str(w.get("word", "")),
                "confidence": round(float(conf), 3) if conf is not None else None,
                "start": round(float(w_start), 2),
                "end": round(float(w_end), 2),
            })
    return candidates


def build_utterance_candidates_payload(
    turn: Dict[str, Any],
    dg_words: List[Dict[str, Any]],
    gl_words: Optional[List[Dict[str, Any]]] = None,
    aai_words: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Assemble structured candidates from all 3 providers for DeepSeek."""
    source = turn.get("source", "student")
    start = float(turn.get("start", 0.0))
    end = float(turn.get("end", start))
    text = turn.get("text", "")

    dg_candidates = extract_provider_candidates_for_turn(dg_words, source, start, end)
    gl_candidates = extract_provider_candidates_for_turn(gl_words or [], source, start, end)
    aai_candidates = extract_provider_candidates_for_turn(aai_words or [], source, start, end)

    speaker_label = "Teacher" if source == "teacher" else "Student"

    return {
        "speaker": speaker_label,
        "start": round(start, 2),
        "end": round(end, 2),
        "heuristic_consensus_text": text,
        "candidates": {
            "deepgram": dg_candidates,
            "gladia": gl_candidates,
            "assemblyai": aai_candidates,
        },
    }


def parse_deepseek_utterance_reply(reply_content: str, fallback_text: str = "") -> str:
    """Safely parse the JSON response from DeepSeek into a clean utterance string."""
    if not reply_content or not isinstance(reply_content, str):
        return fallback_text

    text = reply_content.strip()
    # Strip markdown code fences if model enclosed JSON
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "utterance" in data:
            val = str(data["utterance"]).strip()
            return val if val else fallback_text
    except Exception:
        pass

    # Fallback: if JSON decoding failed, see if single-line text was returned
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("{")]
    if lines:
        return lines[0].strip('"\'')
    return fallback_text


def run_deepseek_utterance_job(
    utt_id: str,
    client: Any,
    payload: Dict[str, Any],
    model: str = DEEPSEEK_UTTERANCE_MODEL,
    system_prompt: str = DEEPSEEK_UTTERANCE_SYSTEM_PROMPT,
) -> Dict[str, Any]:
    """Execute the DeepSeek chat completion in a worker thread.

    Returns:
        Dict with keys: {"id", "ok", "utterance", "error"}
    """
    fallback = payload.get("heuristic_consensus_text", "")
    if client is None:
        return {"id": utt_id, "ok": False, "utterance": fallback, "error": "Client not initialized"}

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=400,
            timeout=15.0,
            stream=False,
        )
        raw_content = response.choices[0].message.content or ""
        resolved = parse_deepseek_utterance_reply(raw_content, fallback_text=fallback)
        return {"id": utt_id, "ok": True, "utterance": resolved, "error": None}
    except Exception as e:
        logger.warning("DeepSeek utterance job error for %s: %s", utt_id, e)
        return {"id": utt_id, "ok": False, "utterance": fallback, "error": str(e)}
