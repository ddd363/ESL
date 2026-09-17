#!/usr/bin/env python3
"""Comprehensive live transcription pipeline diagnostic tracing tool.

Traces a real-time utterance through all 7 stages of the live transcription pipeline:
1. Raw Provider Output (Deepgram, Gladia, AssemblyAI)
2. Normalization & Tokenization
3. Utterance Cluster Membership
4. DP Sequence Alignment Operations
5. Consensus Word Records & Provenance
6. Trailing Punctuation Reconciliation
7. Sentence Extraction & UI Turn Generation

Generates per-audio-interval accounting tables:
TIME | DG | GL | AAI | CONSENSUS | SENTENCE/UI

No API keys or sensitive credentials are ever logged or exposed.
"""

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_consensus import (
    align_word_sequences,
    build_consensus_words,
    normalize_token,
    partition_into_utterance_clusters,
    reconcile_punctuation,
    split_word_and_punctuation,
    TERMINAL_PUNCTUATION,
    CLAUSE_PUNCTUATION,
)
from live_sentences import ends_sentence, split_sentences
from lesson_turns import group_turns


class PipelineTracer:
    def __init__(self):
        self.raw_dg_words: List[Dict[str, Any]] = []
        self.raw_gl_words: List[Dict[str, Any]] = []
        self.raw_aai_words: List[Dict[str, Any]] = []
        self.cluster_events: List[Dict[str, Any]] = []
        self.dp_events: List[Dict[str, Any]] = []
        self.consensus_words: List[Dict[str, Any]] = []
        self.sentences: List[Dict[str, Any]] = []
        self.turns: List[Dict[str, Any]] = []
        self.word_trace_log: List[Dict[str, Any]] = []

    def log_word_trace(
        self,
        provider: str,
        word: str,
        start: float,
        end: float,
        confidence: Optional[float] = None,
        is_final: bool = True,
        speech_final: bool = False,
        word_is_final: bool = True,
        turn_order: Optional[int] = None,
        utterance_id: Optional[str] = None,
    ):
        base, punc = split_word_and_punctuation(word)
        norm = normalize_token(base)
        self.word_trace_log.append({
            "stage": "provider_output",
            "provider": provider,
            "word": word,
            "base": base,
            "punc": punc,
            "norm": norm,
            "start": start,
            "end": end,
            "confidence": confidence,
            "is_final": is_final,
            "speech_final": speech_final,
            "word_is_final": word_is_final,
            "turn_order": turn_order,
            "utterance_id": utterance_id,
        })

    def trace_cluster_partitioning(self, dg_words, gl_words, aai_words=None):
        clusters = partition_into_utterance_clusters(dg_words, gl_words, aai_words or [])
        self.cluster_events = []
        for idx, (c_dg, c_gl, c_aai) in enumerate(clusters):
            all_w = c_dg + c_gl + c_aai
            t_start = min((w.get("start", 0.0) for w in all_w), default=0.0)
            t_end = max((w.get("end", w.get("start", 0.0)) for w in all_w), default=t_start)
            self.cluster_events.append({
                "cluster_id": f"cluster_{idx+1}",
                "start": t_start,
                "end": t_end,
                "dg_count": len(c_dg),
                "gl_count": len(c_gl),
                "aai_count": len(c_aai),
                "dg_words": [w.get("word") for w in c_dg],
                "gl_words": [w.get("word") for w in c_gl],
                "aai_words": [w.get("word") for w in c_aai],
            })
        return clusters

    def trace_consensus(
        self,
        dg_words,
        gl_words=None,
        aai_words=None,
        source="student",
        audio_clock=0.0,
    ):
        self.raw_dg_words = list(dg_words or [])
        self.raw_gl_words = list(gl_words or [])
        self.raw_aai_words = list(aai_words or [])

        self.trace_cluster_partitioning(self.raw_dg_words, self.raw_gl_words, self.raw_aai_words)

        self.consensus_words = build_consensus_words(
            self.raw_dg_words,
            self.raw_gl_words,
            aai_words=self.raw_aai_words,
            source=source,
            audio_clock=audio_clock,
        )

        self.sentences = split_sentences(self.consensus_words, source)
        self.turns = group_turns(self.consensus_words)
        return self.consensus_words

    def generate_accounting_table(self, start_time: float, end_time: float) -> str:
        """Produce the per-audio-interval accounting table.
        
        TIME | DG | GL | AAI | CONSENSUS | SENTENCE/UI
        """
        # Collect time intervals in the window
        intervals = []
        for w in self.raw_dg_words:
            if start_time <= (w.get("start", 0.0)) <= end_time:
                intervals.append((w.get("start", 0.0), w.get("end", 0.0), "dg", w))
        for w in self.raw_gl_words:
            if start_time <= (w.get("start", 0.0)) <= end_time:
                intervals.append((w.get("start", 0.0), w.get("end", 0.0), "gl", w))
        for w in self.raw_aai_words:
            if start_time <= (w.get("start", 0.0)) <= end_time:
                intervals.append((w.get("start", 0.0), w.get("end", 0.0), "aai", w))
        for w in self.consensus_words:
            if start_time <= (w.get("start", 0.0)) <= end_time:
                intervals.append((w.get("start", 0.0), w.get("end", 0.0), "con", w))

        # Index words to sentences
        word_to_sent_text = {}
        for s in self.sentences:
            for w in s.get("words", []):
                word_to_sent_text[w.get("word_id")] = s.get("text", "")

        # Cluster by start time proximity
        intervals.sort(key=lambda x: (x[0], x[1]))

        # Group into distinct time bins
        bins = []
        for s, e, src, obj in intervals:
            matched = False
            for b in bins:
                # Bin overlap threshold (within 0.35s)
                if abs(b["mid"] - ((s + e) / 2.0)) <= 0.35 or (min(b["end"], e) - max(b["start"], s)) > -0.15:
                    b["start"] = min(b["start"], s)
                    b["end"] = max(b["end"], e)
                    b["mid"] = (b["start"] + b["end"]) / 2.0
                    b[src].append(obj)
                    matched = True
                    break
            if not matched:
                bins.append({
                    "start": s,
                    "end": e,
                    "mid": (s + e) / 2.0,
                    "dg": [obj] if src == "dg" else [],
                    "gl": [obj] if src == "gl" else [],
                    "aai": [obj] if src == "aai" else [],
                    "con": [obj] if src == "con" else [],
                })

        bins.sort(key=lambda b: b["start"])

        lines = [
            f"{'TIME INTERVAL':<14} | {'DEEPGRAM':<18} | {'GLADIA':<18} | {'ASSEMBLYAI':<18} | {'CONSENSUS':<22} | {'SENTENCE / UI'}",
            f"{'-'*14}-|-{'-'*18}-|-{'-'*18}-|-{'-'*18}-|-{'-'*22}-|-{'-'*30}",
        ]

        for b in bins:
            t_str = f"{b['start']:.2f}–{b['end']:.2f}"
            dg_str = ", ".join(f"'{w.get('word')}'" for w in b["dg"]) or "—"
            gl_str = ", ".join(f"'{w.get('word')}'" for w in b["gl"]) or "—"
            aai_str = ", ".join(f"'{w.get('word')}'" for w in b["aai"]) or "—"
            
            con_parts = []
            sent_parts = []
            for cw in b["con"]:
                w_txt = cw.get("word", "")
                status = cw.get("status", "")
                con_parts.append(f"'{w_txt}' ({status})")
                sent_txt = word_to_sent_text.get(cw.get("word_id"))
                if sent_txt and sent_txt not in sent_parts:
                    # Truncate long sentence text for table display
                    trunc = sent_txt if len(sent_txt) <= 28 else sent_txt[:25] + "..."
                    sent_parts.append(f"\"{trunc}\"")

            con_str = ", ".join(con_parts) or "—"
            sent_str = ", ".join(sent_parts) or "—"

            lines.append(f"{t_str:<14} | {dg_str:<18} | {gl_str:<18} | {aai_str:<18} | {con_str:<22} | {sent_str}")

        return "\n".join(lines)


def run_diagnostics_on_lesson(lesson_dir: str, start_time: float = 0.0, end_time: float = 60.0):
    words_p = os.path.join(lesson_dir, "words.json")
    if not os.path.exists(words_p):
        print(f"words.json not found in {lesson_dir}")
        return

    with open(words_p, "r", encoding="utf-8") as f:
        dg_data = json.load(f)
    dg_words = [w for w in dg_data.get("words", []) if w.get("source") == "student"]

    gl_words = []
    gl_p = os.path.join(lesson_dir, "words_gladia.json")
    if os.path.exists(gl_p):
        with open(gl_p, "r", encoding="utf-8") as f:
            gl_words = json.load(f).get("words", [])

    aai_words = []
    aai_p = os.path.join(lesson_dir, "words_assemblyai.json")
    if os.path.exists(aai_p):
        with open(aai_p, "r", encoding="utf-8") as f:
            aai_words = json.load(f).get("words", [])

    tracer = PipelineTracer()
    tracer.trace_consensus(dg_words, gl_words, aai_words, source="student", audio_clock=end_time + 10.0)

    print(f"\n==================================================================================")
    print(f"DIAGNOSTIC TRACE REPORT: {os.path.basename(lesson_dir)}")
    print(f"Window: {start_time:.2f}s to {end_time:.2f}s")
    print(f"==================================================================================")
    
    table = tracer.generate_accounting_table(start_time, end_time)
    print(table)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Transcription Pipeline Diagnostics Tracer")
    parser.add_argument("--lesson_dir", type=str, default="/Users/spc/PYTHON NOTEBOOKS/ESL app/audio/test_20260902_232248")
    parser.add_argument("--start", type=float, default=50.0)
    parser.add_argument("--end", type=float, default=80.0)
    args = parser.parse_args()

    run_diagnostics_on_lesson(args.lesson_dir, args.start, args.end)
