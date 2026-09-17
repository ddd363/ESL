"""Deterministic live consensus layer for Deepgram and Gladia transcription.

Reconciles two live ASR streams into a single high-confidence consensus word stream:
1. Normalizes tokens conservatively (lowercase, strip punctuation, preserve original display spelling).
2. Partitions speech into utterance clusters by silence gaps to bound DP alignment.
3. Aligns utterances using timestamp overlap and sequence alignment.
4. Tags words deterministically:
   - 'consensus': both providers agree on the normalized word.
   - 'uncertain': providers disagree on word, substitution, or insertion/deletion.
   - 'primary_only': single provider available (Gladia disabled or timed out past bounded wait).
   - 'pending': recent words awaiting Gladia finalization within bounded wait window.
5. Never invents a third word; retains evidence from both providers for provenance.
"""

import string
import unicodedata

PUNCTUATION_TO_STRIP = string.punctuation + "“”‘’…«»—–"
GLADIA_BOUNDED_WAIT_S = 2.5
UTTERANCE_CLUSTER_GAP_S = 1.5
MAX_CLUSTER_WORDS = 150
MAX_CLUSTER_DURATION_S = 40.0


DIGIT_TO_WORD = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
    "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
    "18": "eighteen", "19": "nineteen", "20": "twenty", "30": "thirty",
    "40": "forty", "50": "fifty", "60": "sixty", "70": "seventy",
    "80": "eighty", "90": "ninety", "100": "hundred",
}


def normalize_token(token: str) -> str:
    """Conservative token normalization for comparing ASR outputs.

    Strips outer punctuation and whitespace, lowercases, standardizes apostrophes,
    and maps digit formatting to words so presentation differences do not create
    false lexical disagreements.
    """
    if not token or not isinstance(token, str):
        return ""
    text = unicodedata.normalize("NFKC", token).strip()
    text = text.replace("’", "'").replace("‘", "'")
    text = text.lower()
    text = text.strip(PUNCTUATION_TO_STRIP)
    if text in DIGIT_TO_WORD:
        text = DIGIT_TO_WORD[text]
    return text


def split_word_and_punctuation(token: str):
    """Separate base lexical token from trailing punctuation."""
    if not token or not isinstance(token, str):
        return "", ""
    token_str = token.strip()
    base = token_str.rstrip(PUNCTUATION_TO_STRIP)
    punc = token_str[len(base):]
    return base, punc


TERMINAL_PUNCTUATION = {".", "?", "!", "…"}
CLAUSE_PUNCTUATION = {",", ";", ":", "-", "—"}


def reconcile_punctuation(d_w=None, g_w=None, a_w=None):
    """Deterministically reconcile trailing punctuation across provider tokens.

    Preserves provider-supported punctuation reliably without requiring multi-provider
    voting for every mark, while resolving conflicting marks deterministically.

    Punctuation omission by one provider does NOT constitute contradiction of another
    provider's terminal punctuation.

    Rules:
    1. Multi-provider exact agreement: If 2 or more providers agree on the exact trailing
       punctuation mark, accept it immediately.
    2. Multi-provider terminal conflict: If multiple providers provide different terminal
       marks (e.g. . vs ?), prefer interrogative (?) or exclamatory (!) if present, otherwise
       resolve by provider order.
    3. Single-provider terminal punctuation: If exactly one provider provides a terminal mark
       (., ?, !, …) and other providers omit punctuation, PRESERVE the terminal mark.
    4. Clause vs Terminal: Terminal punctuation takes precedence over clause punctuation (e.g. ,).
    5. Clause punctuation: If only clause punctuation (, ; : -) is present, preserve it.
    6. All omit: return empty string.
    """
    providers = []
    if d_w:
        providers.append(("deepgram", d_w.get("word", ""), d_w.get("is_final", True), d_w.get("confidence")))
    if g_w:
        providers.append(("gladia", g_w.get("word", ""), g_w.get("is_final", True), g_w.get("confidence")))
    if a_w:
        providers.append(("assemblyai", a_w.get("word", ""), a_w.get("is_final", True), a_w.get("confidence")))

    if not providers:
        return ""
    if len(providers) == 1:
        _, punc = split_word_and_punctuation(providers[0][1])
        return punc

    # 1. Multi-provider exact agreement on trailing punctuation
    punc_counts = {}
    for prov, tok, is_fin, conf in providers:
        _, p = split_word_and_punctuation(tok)
        if p:
            punc_counts[p] = punc_counts.get(p, 0) + 1

    for p, count in punc_counts.items():
        if count >= 2:
            return p

    # 2. Terminal punctuation analysis (., ?, !, …)
    terminal_candidates = []
    for prov, tok, is_fin, conf in providers:
        _, p = split_word_and_punctuation(tok)
        if p and any(ch in TERMINAL_PUNCTUATION for ch in p):
            terminal_candidates.append((prov, p, is_fin, conf or 0.0))

    if terminal_candidates:
        if len(terminal_candidates) == 1:
            # Single provider terminal mark (omission by others is NOT contradiction)
            return terminal_candidates[0][1]

        # Multiple differing terminal marks: prefer ? or ! if present
        for prov, p, is_fin, conf in terminal_candidates:
            if "?" in p or "!" in p:
                return p
        return terminal_candidates[0][1]

    # 3. Trailing non-terminal / quote / clause punctuation analysis
    for prov, tok, is_fin, conf in providers:
        _, p = split_word_and_punctuation(tok)
        if p:
            return p

    return ""


def build_evidence_dict(d_w=None, g_w=None, a_w=None, include_none=False, include_aai=False):
    """Build unified evidence structure across Deepgram, Gladia, and AssemblyAI."""
    ev = {}
    if d_w:
        ev["deepgram"] = {
            "word": d_w.get("word", ""),
            "confidence": d_w.get("confidence"),
            "start": d_w.get("start"),
            "end": d_w.get("end"),
            "is_final": d_w.get("is_final", True),
        }
    elif include_none:
        ev["deepgram"] = None

    if g_w:
        ev["gladia"] = {
            "word": g_w.get("word", ""),
            "confidence": g_w.get("confidence"),
            "start": g_w.get("start"),
            "end": g_w.get("end"),
            "is_final": g_w.get("is_final", True),
            "utterance_id": g_w.get("utterance_id"),
        }
    elif include_none:
        ev["gladia"] = None

    if a_w:
        ev["assemblyai"] = {
            "word": a_w.get("word", ""),
            "confidence": a_w.get("confidence"),
            "start": a_w.get("start"),
            "end": a_w.get("end"),
            "is_final": a_w.get("is_final", True),
            "word_is_final": a_w.get("word_is_final", True),
            "turn_order": a_w.get("turn_order"),
        }
    elif include_aai:
        ev["assemblyai"] = None

    return ev


def build_consensus_word_item(
    word_text,
    start,
    end,
    source,
    status,
    evidence,
    confidence=None,
    provider="consensus",
    settled=True,
    revision_number=1,
    revision=None,
    word_id=None,
):
    """Construct a standard consensus word dictionary with immutable identity and mutable content."""
    if not word_id:
        word_id = f"{source}:{start:.3f}"
    rev_num = revision if revision is not None else revision_number
    return {
        "word_id": word_id,
        "word": word_text,
        "start": start,
        "end": end,
        "source": source,
        "provider": provider,
        "status": status,
        "confidence": confidence,
        "settled": settled,
        "revision_number": rev_num,
        "revision": rev_num,
        "evidence": evidence,
    }


def build_primary_only_word(dg_word, source="student"):
    """Create a single-provider word record when secondary providers are unavailable."""
    start = dg_word.get("start", 0.0)
    end = dg_word.get("end", start)
    text = dg_word.get("word", "")
    word_id = f"{source}:{start:.3f}"
    return build_consensus_word_item(
        word_text=text,
        start=start,
        end=end,
        source=source,
        provider="deepgram",
        status="primary_only",
        evidence=build_evidence_dict(d_w=dg_word, include_none=False),
        confidence=dg_word.get("confidence"),
        settled=True,
        word_id=word_id,
    )


def build_pending_word(dg_word, source="student"):
    """Create a pending word record awaiting secondary provider finalization."""
    start = dg_word.get("start", 0.0)
    end = dg_word.get("end", start)
    text = dg_word.get("word", "")
    word_id = f"{source}:{start:.3f}"
    return build_consensus_word_item(
        word_text=text,
        start=start,
        end=end,
        source=source,
        provider="deepgram",
        status="pending",
        evidence=build_evidence_dict(d_w=dg_word, include_none=False),
        confidence=dg_word.get("confidence"),
        settled=False,
        word_id=word_id,
    )


def _score_candidate(word_dict, provider="deepgram"):
    """Evaluate provider-local heuristic evidence for a candidate word on disputed intervals.

    IMPORTANT: Confidence values emitted by Deepgram, Gladia, and AssemblyAI are
    model-internal likelihoods and are NOT calibrated probabilities across providers
    (e.g., 0.90 from Deepgram is not mathematically equivalent to 0.90 from Gladia).
    This scoring function treats confidence as a provider-local ordinal heuristic,
    combined with provider model identity, finality state, and endpointing signals.

    Heuristic Components:
    1. Provider Model Identity & Finality:
       - Gladia Solaria-1 final (Whisper-based ASR) receives +1.5 finality weight.
       - AssemblyAI Universal-3.5 Pro final receives +1.3 finality weight.
       - Deepgram Nova-3 final receives +1.0 standard finality weight.
    2. Endpointing & Finality Flags:
       - Deepgram speech_final: +0.5 (speech endpoint confirmed).
       - AssemblyAI word_is_final: +0.2 (word boundary settled).
    3. Provider-Local Confidence Tiers (binned locally per provider):
       - High (conf >= 0.85): +1.0
       - Medium (conf >= 0.60): +0.5
       - Low (conf < 0.40): -0.5
    """
    if not word_dict or not isinstance(word_dict, dict):
        return 0.0
    score = 0.0
    is_final = word_dict.get("is_final", True)
    if is_final:
        if provider == "gladia":
            score += 1.5
        elif provider == "assemblyai":
            score += 1.3
        else:
            score += 1.0

    if word_dict.get("speech_final", False):
        score += 0.5
    if word_dict.get("word_is_final", False) and provider == "assemblyai":
        score += 0.2

    conf = word_dict.get("confidence")
    if conf is not None:
        if conf >= 0.85:
            score += 1.0
        elif conf >= 0.60:
            score += 0.5
        elif conf < 0.40:
            score -= 0.5
    return score


def _align_cluster_dp(dg_words, gl_words, aai_words=None, source="student", gladia_latest_end=0.0, audio_clock=0.0):
    """Align a single bounded utterance cluster using DP sequence alignment with symmetric 3-provider evidence."""
    aai_words = list(aai_words or [])
    used_aai_indices = set()

    def _find_matching_aai(start_t, end_t, target_norm=None):
        if not aai_words:
            return None
        slot_mid = (start_t + end_t) / 2.0
        best_idx = None
        best_dist = float("inf")
        for idx, a_w in enumerate(aai_words):
            if idx in used_aai_indices:
                continue
            a_start = a_w.get("start", 0.0)
            a_end = a_w.get("end", a_start)
            a_mid = (a_start + a_end) / 2.0
            dist = abs(slot_mid - a_mid)
            overlap = min(end_t, a_end) - max(start_t, a_start)
            a_norm = normalize_token(a_w.get("word", ""))

            is_lexical_match = bool(target_norm and a_norm == target_norm)
            if is_lexical_match:
                if dist <= 2.5 or overlap > -0.5:
                    effective_dist = dist - 0.5
                    if effective_dist < best_dist:
                        best_dist = effective_dist
                        best_idx = idx
            else:
                # Non-lexical matching requires physical acoustic overlap to prevent consuming adjacent distinct words
                if overlap > 0.05 or (dist <= 0.20 and overlap >= -0.05):
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = idx
        if best_idx is not None:
            used_aai_indices.add(best_idx)
            return aai_words[best_idx]
        return None

    has_aai = bool(aai_words)

    if not dg_words and not gl_words:
        results = []
        for a_w in aai_words:
            a_start = a_w.get("start", 0.0)
            a_end = a_w.get("end", a_start)
            word_id = f"{source}:aai:{a_start:.3f}"
            results.append(build_consensus_word_item(
                word_text=a_w.get("word", ""),
                start=a_start,
                end=a_end,
                source=source,
                status="uncertain",
                confidence=a_w.get("confidence"),
                evidence=build_evidence_dict(a_w=a_w, include_none=True, include_aai=has_aai),
                word_id=word_id,
            ))
        return results

    if not dg_words:
        results = []
        for g_w in gl_words:
            g_start = g_w.get("start", 0.0)
            g_end = g_w.get("end", g_start)
            g_base, _ = split_word_and_punctuation(g_w.get("word", ""))
            g_norm = normalize_token(g_base)
            a_w = _find_matching_aai(g_start, g_end, target_norm=g_norm)
            a_base, _ = split_word_and_punctuation(a_w.get("word", "") if a_w else "")
            a_norm = normalize_token(a_base) if a_w else None
            punc = reconcile_punctuation(g_w=g_w, a_w=a_w)
            word_id = f"{source}:gl:{g_start:.3f}"

            # GL + AAI agreement
            if a_norm and g_norm == a_norm:
                gl_final = g_w.get("is_final", True)
                aai_final = a_w.get("is_final", True) if a_w else False
                status = "consensus" if (gl_final and aai_final) else "uncertain"
                display_word = g_base + punc if g_base else g_w.get("word", "")
            else:
                status = "uncertain"
                display_word = g_w.get("word", "")

            results.append(build_consensus_word_item(
                word_text=display_word,
                start=g_start,
                end=g_end,
                source=source,
                status=status,
                confidence=g_w.get("confidence"),
                evidence=build_evidence_dict(g_w=g_w, a_w=a_w, include_none=True, include_aai=has_aai),
                word_id=word_id,
            ))

        # Append any unconsumed AssemblyAI insertions
        for idx, a_w in enumerate(aai_words):
            if idx not in used_aai_indices:
                a_start = a_w.get("start", 0.0)
                a_end = a_w.get("end", a_start)
                word_id = f"{source}:aai:{a_start:.3f}"
                results.append(build_consensus_word_item(
                    word_text=a_w.get("word", ""),
                    start=a_start,
                    end=a_end,
                    source=source,
                    status="uncertain",
                    confidence=a_w.get("confidence"),
                    evidence=build_evidence_dict(a_w=a_w, include_none=True, include_aai=has_aai),
                    word_id=word_id,
                ))
        return results

    if not gl_words:
        results = []
        for d_w in dg_words:
            d_start = d_w.get("start", 0.0)
            d_end = d_w.get("end", d_start)
            d_base, _ = split_word_and_punctuation(d_w.get("word", ""))
            d_norm = normalize_token(d_base)
            a_w = _find_matching_aai(d_start, d_end, target_norm=d_norm)
            a_base, _ = split_word_and_punctuation(a_w.get("word", "") if a_w else "")
            a_norm = normalize_token(a_base) if a_w else None
            punc = reconcile_punctuation(d_w=d_w, a_w=a_w)
            word_id = f"{source}:{d_start:.3f}"

            if a_norm and d_norm == a_norm:
                # DG + AAI agreement
                dg_final = d_w.get("is_final", True)
                aai_final = a_w.get("is_final", True) if a_w else False
                status = "consensus" if (dg_final and aai_final) else "uncertain"
                display_word = d_base + punc if d_base else d_w.get("word", "")
                results.append(build_consensus_word_item(
                    word_text=display_word,
                    start=d_start,
                    end=d_end,
                    source=source,
                    status=status,
                    confidence=d_w.get("confidence"),
                    evidence=build_evidence_dict(d_w=d_w, a_w=a_w, include_none=True, include_aai=has_aai),
                    word_id=word_id,
                ))
            elif d_start <= gladia_latest_end:
                results.append(build_consensus_word_item(
                    word_text=d_w.get("word", ""),
                    start=d_start,
                    end=d_end,
                    source=source,
                    status="uncertain",
                    confidence=d_w.get("confidence"),
                    evidence=build_evidence_dict(d_w=d_w, a_w=a_w, include_none=True, include_aai=has_aai),
                    word_id=word_id,
                ))
            elif audio_clock > 0 and (audio_clock - d_end) < GLADIA_BOUNDED_WAIT_S:
                results.append(build_pending_word(d_w, source=source))
            else:
                results.append(build_primary_only_word(d_w, source=source))

        # Append any unconsumed AssemblyAI insertions
        for idx, a_w in enumerate(aai_words):
            if idx not in used_aai_indices:
                a_start = a_w.get("start", 0.0)
                a_end = a_w.get("end", a_start)
                word_id = f"{source}:aai:{a_start:.3f}"
                results.append(build_consensus_word_item(
                    word_text=a_w.get("word", ""),
                    start=a_start,
                    end=a_end,
                    source=source,
                    status="uncertain",
                    confidence=a_w.get("confidence"),
                    evidence=build_evidence_dict(a_w=a_w, include_none=True, include_aai=has_aai),
                    word_id=word_id,
                ))
        return results

    m = len(dg_words)
    n = len(gl_words)

    dp = [[-float("inf")] * (n + 1) for _ in range(m + 1)]
    backtrack = [[None] * (n + 1) for _ in range(m + 1)]

    dp[0][0] = 0.0
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] - 1.5
        backtrack[i][0] = ("DEL_DG", i - 1, 0)
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] - 1.5
        backtrack[0][j] = ("INS_GL", 0, j - 1)

    for i in range(1, m + 1):
        d_word = dg_words[i - 1]
        d_start = d_word.get("start") or 0.0
        d_end = d_word.get("end") or d_start
        d_mid = (d_start + d_end) / 2.0
        d_norm = normalize_token(d_word.get("word", ""))

        for j in range(1, n + 1):
            g_word = gl_words[j - 1]
            g_start = g_word.get("start") or 0.0
            g_end = g_word.get("end") or g_start
            g_mid = (g_start + g_end) / 2.0
            g_norm = normalize_token(g_word.get("word", ""))

            t_diff = abs(d_mid - g_mid)
            overlap = min(d_end, g_end) - max(d_start, g_start)

            best_score = -float("inf")
            best_op = None
            prev_pos = None

            # Option 1: Match or Substitution with Time-Proximity Gating
            if d_norm and g_norm and d_norm == g_norm:
                if t_diff <= 3.5 or overlap > -0.6:
                    time_penalty = min(t_diff * 0.4, 1.5)
                    score = dp[i - 1][j - 1] + 3.0 - time_penalty
                    if score > best_score:
                        best_score = score
                        best_op = "MATCH"
                        prev_pos = (i - 1, j - 1)
            else:
                # Genuine substitution strictly requires acoustic overlap (competing hypotheses for the SAME spoken word)
                # If words are consecutive non-overlapping tokens, substitution is forbidden to force DEL_DG + INS_GL
                if overlap > 0.05 or (t_diff <= 0.20 and overlap >= -0.05):
                    time_penalty = min(t_diff * 0.7, 1.5)
                    score = dp[i - 1][j - 1] - 1.0 - time_penalty
                    if score > best_score:
                        best_score = score
                        best_op = "SUBST"
                        prev_pos = (i - 1, j - 1)

            # Option 2: Skip Deepgram word (DEL_DG)
            del_score = dp[i - 1][j] - 1.5
            if del_score > best_score:
                best_score = del_score
                best_op = "DEL_DG"
                prev_pos = (i - 1, j)

            # Option 3: Skip Gladia word (INS_GL)
            ins_score = dp[i][j - 1] - 1.5
            if ins_score > best_score:
                best_score = ins_score
                best_op = "INS_GL"
                prev_pos = (i, j - 1)

            dp[i][j] = best_score
            backtrack[i][j] = (best_op, prev_pos[0], prev_pos[1]) if prev_pos else None

    aligned_ops = []
    curr_i, curr_j = m, n
    while curr_i > 0 or curr_j > 0:
        entry = backtrack[curr_i][curr_j]
        if entry is None:
            break
        op, prev_i, prev_j = entry
        aligned_ops.append((op, curr_i - 1 if curr_i > prev_i else None,
                            curr_j - 1 if curr_j > prev_j else None))
        curr_i, curr_j = prev_i, prev_j

    aligned_ops.reverse()

    results = []
    for op, i_idx, j_idx in aligned_ops:
        if op == "MATCH":
            d_w = dg_words[i_idx]
            g_w = gl_words[j_idx]
            slot_start = d_w.get("start", 0.0)
            slot_end = d_w.get("end", slot_start)
            word_id = f"{source}:{slot_start:.3f}"

            d_base, _ = split_word_and_punctuation(d_w.get("word", ""))
            d_norm = normalize_token(d_base)
            a_w = _find_matching_aai(slot_start, slot_end, target_norm=d_norm)

            a_base, _ = split_word_and_punctuation(a_w.get("word", "") if a_w else "")
            a_norm = normalize_token(a_base) if a_w else None
            punc = reconcile_punctuation(d_w, g_w, a_w)

            # DG == GL agree (MATCH) -> consensus
            status = "consensus"
            display_word = d_base + punc if d_base else d_w.get("word", "")
            results.append(build_consensus_word_item(
                word_text=display_word,
                start=slot_start,
                end=slot_end,
                source=source,
                status=status,
                confidence=None,
                evidence=build_evidence_dict(d_w=d_w, g_w=g_w, a_w=a_w, include_none=True, include_aai=has_aai),
                word_id=word_id,
            ))
        elif op == "SUBST":
            d_w = dg_words[i_idx]
            g_w = gl_words[j_idx]
            slot_start = d_w.get("start", 0.0)
            slot_end = d_w.get("end", slot_start)
            word_id = f"{source}:{slot_start:.3f}"

            d_base, _ = split_word_and_punctuation(d_w.get("word", ""))
            g_base, _ = split_word_and_punctuation(g_w.get("word", ""))
            d_norm = normalize_token(d_base)
            g_norm = normalize_token(g_base)

            a_w = _find_matching_aai(slot_start, slot_end)
            a_base, _ = split_word_and_punctuation(a_w.get("word", "") if a_w else "")
            a_norm = normalize_token(a_base) if a_w else None
            punc = reconcile_punctuation(d_w, g_w, a_w)

            # Evaluate provider-local heuristic scores
            dg_ev_score = _score_candidate(d_w, provider="deepgram")
            gl_ev_score = _score_candidate(g_w, provider="gladia")
            aai_ev_score = _score_candidate(a_w, provider="assemblyai") if a_w else 0.0

            # Symmetric 2-of-3 agreement evaluation
            if a_norm and g_norm == a_norm:
                # Gladia + AssemblyAI agree
                gl_ev_score += 1.0
                aai_ev_score += 1.0
                gl_final = g_w.get("is_final", True)
                aai_final = a_w.get("is_final", True) if a_w else False
                gl_conf = g_w.get("confidence")
                aai_conf = a_w.get("confidence") if a_w else None
                # Check reliability of agreeing pair: both final and not very low confidence
                reliable = gl_final and aai_final and (gl_conf is None or gl_conf >= 0.50) and (aai_conf is None or aai_conf >= 0.50)
                status = "consensus" if reliable else "uncertain"
                display_base = g_base or a_base
            elif a_norm and d_norm == a_norm:
                # Deepgram + AssemblyAI agree
                dg_ev_score += 1.0
                aai_ev_score += 1.0
                dg_final = d_w.get("is_final", True)
                aai_final = a_w.get("is_final", True) if a_w else False
                dg_conf = d_w.get("confidence")
                aai_conf = a_w.get("confidence") if a_w else None
                reliable = dg_final and aai_final and (dg_conf is None or dg_conf >= 0.50) and (aai_conf is None or aai_conf >= 0.50)
                status = "consensus" if reliable else "uncertain"
                display_base = d_base or a_base
            else:
                # All 3 disagree
                status = "uncertain"
                scores = [
                    ("gladia", gl_ev_score, g_base),
                    ("deepgram", dg_ev_score, d_base),
                ]
                if a_w:
                    scores.append(("assemblyai", aai_ev_score, a_base))
                scores.sort(key=lambda x: x[1], reverse=True)
                display_base = scores[0][2]

            display_word = display_base + punc if display_base else (g_w.get("word") or d_w.get("word") or "")
            results.append(build_consensus_word_item(
                word_text=display_word,
                start=slot_start,
                end=slot_end,
                source=source,
                status=status,
                confidence=None,
                evidence=build_evidence_dict(d_w=d_w, g_w=g_w, a_w=a_w, include_none=True, include_aai=has_aai),
                word_id=word_id,
            ))
        elif op == "DEL_DG":
            d_w = dg_words[i_idx]
            d_start = d_w.get("start", 0.0)
            d_end = d_w.get("end", d_start)
            word_id = f"{source}:{d_start:.3f}"
            d_base, _ = split_word_and_punctuation(d_w.get("word", ""))
            d_norm = normalize_token(d_base)
            a_w = _find_matching_aai(d_start, d_end, target_norm=d_norm)
            a_base, _ = split_word_and_punctuation(a_w.get("word", "") if a_w else "")
            a_norm = normalize_token(a_base) if a_w else None
            punc = reconcile_punctuation(d_w=d_w, a_w=a_w)

            if a_norm and d_norm == a_norm:
                # DG + AAI agree on this word (Gladia omitted it)
                dg_final = d_w.get("is_final", True)
                aai_final = a_w.get("is_final", True) if a_w else False
                reliable = dg_final and aai_final
                status = "consensus" if reliable else "uncertain"
                display_word = d_base + punc if d_base else d_w.get("word", "")
                results.append(build_consensus_word_item(
                    word_text=display_word,
                    start=d_start,
                    end=d_end,
                    source=source,
                    status=status,
                    confidence=d_w.get("confidence"),
                    evidence=build_evidence_dict(d_w=d_w, a_w=a_w, include_none=True, include_aai=has_aai),
                    word_id=word_id,
                ))
            elif d_start <= gladia_latest_end:
                results.append(build_consensus_word_item(
                    word_text=d_w.get("word", ""),
                    start=d_start,
                    end=d_end,
                    source=source,
                    status="uncertain",
                    confidence=d_w.get("confidence"),
                    evidence=build_evidence_dict(d_w=d_w, a_w=a_w, include_none=True, include_aai=has_aai),
                    word_id=word_id,
                ))
            elif audio_clock > 0 and (audio_clock - d_end) < GLADIA_BOUNDED_WAIT_S:
                results.append(build_pending_word(d_w, source=source))
            else:
                results.append(build_primary_only_word(d_w, source=source))
        elif op == "INS_GL":
            g_w = gl_words[j_idx]
            g_start = g_w.get("start", 0.0)
            g_end = g_w.get("end", g_start)
            word_id = f"{source}:gl:{g_start:.3f}"
            g_base, _ = split_word_and_punctuation(g_w.get("word", ""))
            g_norm = normalize_token(g_base)
            a_w = _find_matching_aai(g_start, g_end, target_norm=g_norm)
            a_base, _ = split_word_and_punctuation(a_w.get("word", "") if a_w else "")
            a_norm = normalize_token(a_base) if a_w else None
            punc = reconcile_punctuation(g_w=g_w, a_w=a_w)

            if a_norm and g_norm == a_norm:
                # GL + AAI agree on this word (Deepgram omitted it)
                gl_final = g_w.get("is_final", True)
                aai_final = a_w.get("is_final", True) if a_w else False
                reliable = gl_final and aai_final
                status = "consensus" if reliable else "uncertain"
                display_word = g_base + punc if g_base else g_w.get("word", "")
            else:
                status = "uncertain"
                display_word = g_w.get("word", "")

            results.append(build_consensus_word_item(
                word_text=display_word,
                start=g_start,
                end=g_end,
                source=source,
                status=status,
                confidence=g_w.get("confidence"),
                evidence=build_evidence_dict(g_w=g_w, a_w=a_w, include_none=True, include_aai=has_aai),
                word_id=word_id,
            ))

    # Append any remaining unconsumed AssemblyAI insertions
    for idx, a_w in enumerate(aai_words):
        if idx not in used_aai_indices:
            a_start = a_w.get("start", 0.0)
            a_end = a_w.get("end", a_start)
            word_id = f"{source}:aai:{a_start:.3f}"
            results.append(build_consensus_word_item(
                word_text=a_w.get("word", ""),
                start=a_start,
                end=a_end,
                source=source,
                status="uncertain",
                confidence=a_w.get("confidence"),
                evidence=build_evidence_dict(a_w=a_w, include_none=True, include_aai=has_aai),
                word_id=word_id,
            ))

    return results


def _subdivide_cluster_if_needed(dg_words, gl_words, aai_words=None, max_words=MAX_CLUSTER_WORDS, max_duration_s=MAX_CLUSTER_DURATION_S):
    """Subdivide an excessively large continuous cluster at the lowest-risk natural boundary."""
    aai_words = list(aai_words or [])
    all_words = dg_words + gl_words + aai_words
    total_words = len(all_words)
    if not all_words:
        return []

    all_starts = [w.get("start", 0.0) for w in all_words]
    all_ends = [w.get("end", w.get("start", 0.0)) for w in all_words]
    duration = (max(all_ends) - min(all_starts)) if all_starts else 0.0

    if total_words <= max_words and duration <= max_duration_s:
        return [(dg_words, gl_words, aai_words)]

    t_min = min(all_starts)
    t_max = max(all_ends)
    t_mid = (t_min + t_max) / 2.0

    best_split_t = t_mid
    best_score = -float("inf")

    # Evaluate candidate boundaries from dg_words
    for i in range(len(dg_words) - 1):
        w1, w2 = dg_words[i], dg_words[i + 1]
        gap = (w2.get("start", 0.0) - w1.get("end", 0.0))
        mid = (w1.get("end", 0.0) + w2.get("start", 0.0)) / 2.0
        dist_from_mid = abs(mid - t_mid)
        score = max(gap, 0.0) * 3.0 - (dist_from_mid / max(duration, 1.0)) * 2.0
        if str(w1.get("word", "")).endswith((".", ",", "?", "!")):
            score += 1.0
        if score > best_score:
            best_score = score
            best_split_t = mid

    # Evaluate candidate boundaries from gl_words
    for i in range(len(gl_words) - 1):
        w1, w2 = gl_words[i], gl_words[i + 1]
        gap = (w2.get("start", 0.0) - w1.get("end", 0.0))
        mid = (w1.get("end", 0.0) + w2.get("start", 0.0)) / 2.0
        dist_from_mid = abs(mid - t_mid)
        score = max(gap, 0.0) * 3.0 - (dist_from_mid / max(duration, 1.0)) * 2.0
        if str(w1.get("word", "")).endswith((".", ",", "?", "!")):
            score += 1.0
        if score > best_score:
            best_score = score
            best_split_t = mid

    dg_left = [w for w in dg_words if ((w.get("start", 0.0) + w.get("end", w.get("start", 0.0))) / 2.0 <= best_split_t)]
    dg_right = [w for w in dg_words if ((w.get("start", 0.0) + w.get("end", w.get("start", 0.0))) / 2.0 > best_split_t)]
    gl_left = [w for w in gl_words if ((w.get("start", 0.0) + w.get("end", w.get("start", 0.0))) / 2.0 <= best_split_t)]
    gl_right = [w for w in gl_words if ((w.get("start", 0.0) + w.get("end", w.get("start", 0.0))) / 2.0 > best_split_t)]
    aai_left = [w for w in aai_words if ((w.get("start", 0.0) + w.get("end", w.get("start", 0.0))) / 2.0 <= best_split_t)]
    aai_right = [w for w in aai_words if ((w.get("start", 0.0) + w.get("end", w.get("start", 0.0))) / 2.0 > best_split_t)]

    # Fallback to index midpoint if split did not partition
    if ((len(dg_left) == len(dg_words) and len(gl_left) == len(gl_words) and len(aai_left) == len(aai_words))
            or (not dg_left and not gl_left and not aai_left)):
        dg_mid_idx = max(1, len(dg_words) // 2) if dg_words else 0
        gl_mid_idx = max(1, len(gl_words) // 2) if gl_words else 0
        aai_mid_idx = max(1, len(aai_words) // 2) if aai_words else 0
        dg_left, dg_right = dg_words[:dg_mid_idx], dg_words[dg_mid_idx:]
        gl_left, gl_right = gl_words[:gl_mid_idx], gl_words[gl_mid_idx:]
        aai_left, aai_right = aai_words[:aai_mid_idx], aai_words[aai_mid_idx:]

    left_clusters = _subdivide_cluster_if_needed(dg_left, gl_left, aai_left, max_words, max_duration_s)
    right_clusters = _subdivide_cluster_if_needed(dg_right, gl_right, aai_right, max_words, max_duration_s)
    return left_clusters + right_clusters


def partition_into_utterance_clusters(dg_words, gl_words, aai_words=None, gap_s=UTTERANCE_CLUSTER_GAP_S):
    """Partition combined words from all 3 providers into time-bounded speech clusters.

    Splits on silence pauses > gap_s across streams and further subdivides long continuous
    monologues at natural boundaries to guarantee bounded alignment execution time.
    """
    aai_words = list(aai_words or [])
    if not dg_words and not gl_words and not aai_words:
        return []

    # Merge all word intervals to find connected speech components
    intervals = []
    for w in dg_words:
        start = w.get("start", 0.0)
        end = w.get("end", start)
        intervals.append((start, end, "dg", w))
    for w in gl_words:
        start = w.get("start", 0.0)
        end = w.get("end", start)
        intervals.append((start, end, "gl", w))
    for w in aai_words:
        start = w.get("start", 0.0)
        end = w.get("end", start)
        intervals.append((start, end, "aai", w))

    intervals.sort(key=lambda x: (x[0], x[1]))

    raw_clusters = []
    curr_dg = []
    curr_gl = []
    curr_aai = []
    cluster_end = -float("inf")

    for start, end, origin, word in intervals:
        if intervals and start - cluster_end > gap_s and (curr_dg or curr_gl or curr_aai):
            raw_clusters.append((curr_dg, curr_gl, curr_aai))
            curr_dg = []
            curr_gl = []
            curr_aai = []
            cluster_end = end
        else:
            cluster_end = max(cluster_end, end)

        if origin == "dg":
            curr_dg.append(word)
        elif origin == "gl":
            curr_gl.append(word)
        else:
            curr_aai.append(word)

    if curr_dg or curr_gl or curr_aai:
        raw_clusters.append((curr_dg, curr_gl, curr_aai))

    # Apply hard size/duration cap to guarantee bounded matrix complexity
    final_clusters = []
    for c_dg, c_gl, c_aai in raw_clusters:
        subdivided = _subdivide_cluster_if_needed(c_dg, c_gl, c_aai)
        final_clusters.extend(subdivided)

    return final_clusters


def align_word_sequences(dg_words, gl_words, aai_words=None, source="student", gladia_settled_boundary=None, audio_clock=0.0):
    """Align word sequences for one source across up to 3 providers partitioned by utterance clusters.

    Args:
        dg_words: Deepgram words list.
        gl_words: Gladia words list.
        aai_words: AssemblyAI words list (optional).
        source: Microphone source label.
        gladia_settled_boundary: Known timestamp Gladia finalized up to.
        audio_clock: Current audio position in seconds.
    """
    gl_words = list(gl_words or [])
    aai_words = list(aai_words or [])
    gladia_latest_end = max(
        (g.get("end") or g.get("start") or 0.0 for g in gl_words), default=0.0
    )

    clusters = partition_into_utterance_clusters(dg_words, gl_words, aai_words)
    aligned_all = []

    for dg_cluster, gl_cluster, aai_cluster in clusters:
        cluster_res = _align_cluster_dp(
            dg_cluster, gl_cluster, aai_words=aai_cluster, source=source,
            gladia_latest_end=gladia_latest_end, audio_clock=audio_clock
        )
        aligned_all.extend(cluster_res)

    aligned_all.sort(key=lambda w: (w.get("start") or 0.0, w.get("end") or 0.0))
    return aligned_all


def build_consensus_words(
    dg_words,
    gl_words=None,
    aai_words=None,
    source="student",
    gladia_active=True,
    assemblyai_active=True,
    audio_clock=0.0,
):
    """Combine Deepgram, Gladia, and AssemblyAI word streams for all sources.

    Non-blocking: If AssemblyAI is unavailable or unconfigured, consensus proceeds
    immediately using Deepgram + Gladia without delay.

    Args:
        dg_words: Deepgram words list.
        gl_words: Gladia words list (or None/empty).
        aai_words: AssemblyAI words list (or None/empty).
        source: Default microphone source label.
        gladia_active: Whether Gladia streaming is enabled / configured.
        assemblyai_active: Whether AssemblyAI streaming is enabled / configured.
        audio_clock: Current lesson audio clock in seconds.

    Returns:
        List of consensus word dicts sorted by start timestamp.
    """
    if not gladia_active and not assemblyai_active:
        return [build_primary_only_word(w, source=w.get("source", source)) for w in dg_words]

    gl_words = gl_words or []
    aai_words = aai_words or []

    # If neither secondary provider has returned any words yet:
    # Deepgram words within the bounded wait window remain pending; older ones become primary_only.
    if not gl_words and not aai_words:
        results = []
        for w in dg_words:
            src = w.get("source", source)
            w_end = w.get("end") or w.get("start") or 0.0
            if audio_clock > 0 and (audio_clock - w_end) < GLADIA_BOUNDED_WAIT_S:
                results.append(build_pending_word(w, source=src))
            else:
                results.append(build_primary_only_word(w, source=src))
        return results

    dg_by_source = {}
    for w in dg_words:
        src = w.get("source", source)
        dg_by_source.setdefault(src, []).append(w)

    gl_by_source = {}
    for w in gl_words:
        src = w.get("source", source)
        gl_by_source.setdefault(src, []).append(w)

    aai_by_source = {}
    for w in aai_words:
        src = w.get("source", source)
        aai_by_source.setdefault(src, []).append(w)

    all_sources = set(dg_by_source.keys()) | set(gl_by_source.keys()) | set(aai_by_source.keys())
    if not all_sources:
        all_sources = {source}

    combined = []
    settled_boundary = max(0.0, audio_clock - GLADIA_BOUNDED_WAIT_S)

    for src in all_sources:
        src_dg = sorted(dg_by_source.get(src, []), key=lambda w: w.get("start", 0.0))
        src_gl = sorted(gl_by_source.get(src, []), key=lambda w: w.get("start", 0.0))
        src_aai = sorted(aai_by_source.get(src, []), key=lambda w: w.get("start", 0.0))
        aligned = align_word_sequences(
            src_dg, src_gl, aai_words=src_aai, source=src,
            gladia_settled_boundary=settled_boundary, audio_clock=audio_clock
        )
        combined.extend(aligned)

    combined.sort(key=lambda w: (w.get("start") or 0.0))
    return combined
