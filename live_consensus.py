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
MAX_CLUSTER_WORDS = 25
MAX_CLUSTER_DURATION_S = 8.0


def normalize_token(token: str) -> str:
    """Conservative token normalization for comparing ASR outputs.

    Strips outer punctuation and whitespace, lowercases, and standardizes apostrophes,
    while leaving internal word characters intact.
    """
    if not token or not isinstance(token, str):
        return ""
    text = unicodedata.normalize("NFKC", token).strip()
    # Normalize curly apostrophes to ASCII apostrophe
    text = text.replace("’", "'").replace("‘", "'")
    text = text.lower()
    text = text.strip(PUNCTUATION_TO_STRIP)
    return text


def build_primary_only_word(dg_word, source="student"):
    """Create a single-provider word record when Gladia is unavailable or timed out."""
    return {
        "word": dg_word.get("word", ""),
        "start": dg_word.get("start", 0.0),
        "end": dg_word.get("end", dg_word.get("start", 0.0)),
        "source": source,
        "provider": "deepgram",
        "status": "primary_only",
        "confidence": dg_word.get("confidence"),
        "evidence": {
            "deepgram": {
                "word": dg_word.get("word", ""),
                "confidence": dg_word.get("confidence"),
            },
        },
    }


def build_pending_word(dg_word, source="student"):
    """Create a pending word record awaiting Gladia finalization."""
    return {
        "word": dg_word.get("word", ""),
        "start": dg_word.get("start", 0.0),
        "end": dg_word.get("end", dg_word.get("start", 0.0)),
        "source": source,
        "provider": "deepgram",
        "status": "pending",
        "confidence": dg_word.get("confidence"),
        "evidence": {
            "deepgram": {
                "word": dg_word.get("word", ""),
                "confidence": dg_word.get("confidence"),
            },
        },
    }


def _align_cluster_dp(dg_words, gl_words, source="student", gladia_latest_end=0.0, audio_clock=0.0):
    """Align a single bounded utterance cluster using DP sequence alignment."""
    if not dg_words and not gl_words:
        return []

    if not dg_words:
        # Gladia words only in this cluster -> disputed insertion
        results = []
        for g_w in gl_words:
            results.append({
                "word": g_w.get("word", ""),
                "start": g_w.get("start", 0.0),
                "end": g_w.get("end", g_w.get("start", 0.0)),
                "source": source,
                "provider": "consensus",
                "status": "uncertain",
                "confidence": g_w.get("confidence"),
                "evidence": {
                    "deepgram": None,
                    "gladia": {"word": g_w.get("word", ""), "confidence": g_w.get("confidence")},
                },
            })
        return results

    if not gl_words:
        results = []
        for d_w in dg_words:
            d_start = d_w.get("start", 0.0)
            d_end = d_w.get("end", d_start)
            # Check bounded wait: if audio is still recent and Gladia has not reached this point
            if audio_clock > 0 and (audio_clock - d_end) < GLADIA_BOUNDED_WAIT_S and d_start > gladia_latest_end:
                results.append(build_pending_word(d_w, source=source))
            else:
                if d_start <= gladia_latest_end:
                    # Gladia processed this time span and omitted the word
                    results.append({
                        "word": d_w.get("word", ""),
                        "start": d_start,
                        "end": d_end,
                        "source": source,
                        "provider": "consensus",
                        "status": "uncertain",
                        "confidence": d_w.get("confidence"),
                        "evidence": {
                            "deepgram": {"word": d_w.get("word", ""), "confidence": d_w.get("confidence")},
                            "gladia": None,
                        },
                    })
                else:
                    results.append(build_primary_only_word(d_w, source=source))
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

            # Option 1: Match or Substitution
            if t_diff <= 3.0 or overlap > -0.5:
                if d_norm and g_norm and d_norm == g_norm:
                    time_penalty = min(t_diff * 0.5, 1.5)
                    score = dp[i - 1][j - 1] + 3.0 - time_penalty
                    if score > best_score:
                        best_score = score
                        best_op = "MATCH"
                        prev_pos = (i - 1, j - 1)
                else:
                    if t_diff <= 1.5 or overlap > -0.3:
                        time_penalty = min(t_diff * 0.8, 2.0)
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
            display_word = d_w.get("word", "")
            results.append({
                "word": display_word,
                "start": d_w.get("start", 0.0),
                "end": d_w.get("end", d_w.get("start", 0.0)),
                "source": source,
                "provider": "consensus",
                "status": "consensus",
                "confidence": None,
                "evidence": {
                    "deepgram": {"word": d_w.get("word", ""), "confidence": d_w.get("confidence")},
                    "gladia": {"word": g_w.get("word", ""), "confidence": g_w.get("confidence")},
                },
            })
        elif op == "SUBST":
            d_w = dg_words[i_idx]
            g_w = gl_words[j_idx]
            display_word = d_w.get("word", "")
            results.append({
                "word": display_word,
                "start": d_w.get("start", 0.0),
                "end": d_w.get("end", d_w.get("start", 0.0)),
                "source": source,
                "provider": "consensus",
                "status": "uncertain",
                "confidence": None,
                "evidence": {
                    "deepgram": {"word": d_w.get("word", ""), "confidence": d_w.get("confidence")},
                    "gladia": {"word": g_w.get("word", ""), "confidence": g_w.get("confidence")},
                },
            })
        elif op == "DEL_DG":
            d_w = dg_words[i_idx]
            d_start = d_w.get("start", 0.0)
            d_end = d_w.get("end", d_start)
            if d_start > gladia_latest_end:
                if audio_clock > 0 and (audio_clock - d_end) < GLADIA_BOUNDED_WAIT_S:
                    results.append(build_pending_word(d_w, source=source))
                else:
                    results.append(build_primary_only_word(d_w, source=source))
            else:
                results.append({
                    "word": d_w.get("word", ""),
                    "start": d_start,
                    "end": d_end,
                    "source": source,
                    "provider": "consensus",
                    "status": "uncertain",
                    "confidence": d_w.get("confidence"),
                    "evidence": {
                        "deepgram": {"word": d_w.get("word", ""), "confidence": d_w.get("confidence")},
                        "gladia": None,
                    },
                })
        elif op == "INS_GL":
            # Gladia recognized a word that Deepgram omitted (disputed insertion)
            g_w = gl_words[j_idx]
            results.append({
                "word": g_w.get("word", ""),
                "start": g_w.get("start", 0.0),
                "end": g_w.get("end", g_w.get("start", 0.0)),
                "source": source,
                "provider": "consensus",
                "status": "uncertain",
                "confidence": g_w.get("confidence"),
                "evidence": {
                    "deepgram": None,
                    "gladia": {"word": g_w.get("word", ""), "confidence": g_w.get("confidence")},
                },
            })

    return results


def _subdivide_cluster_if_needed(dg_words, gl_words, max_words=MAX_CLUSTER_WORDS, max_duration_s=MAX_CLUSTER_DURATION_S):
    """Subdivide an excessively large continuous cluster at the lowest-risk natural boundary."""
    total_words = len(dg_words) + len(gl_words)
    if not dg_words and not gl_words:
        return []

    all_starts = [w.get("start", 0.0) for w in dg_words + gl_words]
    all_ends = [w.get("end", w.get("start", 0.0)) for w in dg_words + gl_words]
    duration = (max(all_ends) - min(all_starts)) if all_starts else 0.0

    if total_words <= max_words and duration <= max_duration_s:
        return [(dg_words, gl_words)]

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

    dg_left = [w for w in dg_words if (w.get("end", w.get("start", 0.0)) <= best_split_t)]
    dg_right = [w for w in dg_words if (w.get("end", w.get("start", 0.0)) > best_split_t)]
    gl_left = [w for w in gl_words if (w.get("end", w.get("start", 0.0)) <= best_split_t)]
    gl_right = [w for w in gl_words if (w.get("end", w.get("start", 0.0)) > best_split_t)]

    # Fallback to index midpoint if split did not partition
    if (len(dg_left) == len(dg_words) and len(gl_left) == len(gl_words)) or (not dg_left and not gl_left):
        dg_mid_idx = max(1, len(dg_words) // 2) if dg_words else 0
        gl_mid_idx = max(1, len(gl_words) // 2) if gl_words else 0
        dg_left, dg_right = dg_words[:dg_mid_idx], dg_words[dg_mid_idx:]
        gl_left, gl_right = gl_words[:gl_mid_idx], gl_words[gl_mid_idx:]

    left_clusters = _subdivide_cluster_if_needed(dg_left, gl_left, max_words, max_duration_s)
    right_clusters = _subdivide_cluster_if_needed(dg_right, gl_right, max_words, max_duration_s)
    return left_clusters + right_clusters


def partition_into_utterance_clusters(dg_words, gl_words, gap_s=UTTERANCE_CLUSTER_GAP_S):
    """Partition combined words into small time-bounded speech clusters.

    Splits on silence pauses > gap_s in both streams and further subdivides long continuous
    monologues at natural boundaries to guarantee bounded alignment execution time.
    """
    if not dg_words and not gl_words:
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

    intervals.sort(key=lambda x: (x[0], x[1]))

    raw_clusters = []
    curr_dg = []
    curr_gl = []
    cluster_end = -float("inf")

    for start, end, origin, word in intervals:
        if intervals and start - cluster_end > gap_s and (curr_dg or curr_gl):
            raw_clusters.append((curr_dg, curr_gl))
            curr_dg = []
            curr_gl = []
            cluster_end = end
        else:
            cluster_end = max(cluster_end, end)

        if origin == "dg":
            curr_dg.append(word)
        else:
            curr_gl.append(word)

    if curr_dg or curr_gl:
        raw_clusters.append((curr_dg, curr_gl))

    # Apply hard size/duration cap to guarantee bounded matrix complexity
    final_clusters = []
    for c_dg, c_gl in raw_clusters:
        subdivided = _subdivide_cluster_if_needed(c_dg, c_gl)
        final_clusters.extend(subdivided)

    return final_clusters


def align_word_sequences(dg_words, gl_words, source="student", gladia_settled_boundary=None, audio_clock=0.0):
    """Align word sequences for one source partitioned by utterance clusters.

    Args:
        dg_words: Deepgram words list.
        gl_words: Gladia words list.
        source: Microphone source label.
        gladia_settled_boundary: Known timestamp Gladia finalized up to.
        audio_clock: Current audio position in seconds.
    """
    gladia_latest_end = max(
        (g.get("end") or g.get("start") or 0.0 for g in gl_words), default=0.0
    )

    clusters = partition_into_utterance_clusters(dg_words, gl_words)
    aligned_all = []

    for dg_cluster, gl_cluster in clusters:
        cluster_res = _align_cluster_dp(
            dg_cluster, gl_cluster, source=source,
            gladia_latest_end=gladia_latest_end, audio_clock=audio_clock
        )
        aligned_all.extend(cluster_res)

    aligned_all.sort(key=lambda w: (w.get("start") or 0.0, w.get("end") or 0.0))
    return aligned_all


def build_consensus_words(dg_words, gl_words=None, source="student", gladia_active=True, audio_clock=0.0):
    """Combine Deepgram and Gladia word streams for all sources with bounded wait.

    Args:
        dg_words: Deepgram words list.
        gl_words: Gladia words list (or None/empty).
        source: Default microphone source label.
        gladia_active: Whether Gladia streaming is enabled / configured.
        audio_clock: Current lesson audio clock in seconds.

    Returns:
        List of consensus word dicts sorted by start timestamp.
    """
    if not gladia_active:
        return [build_primary_only_word(w, source=w.get("source", source)) for w in dg_words]

    gl_words = gl_words or []

    # If Gladia is active but hasn't returned any words yet:
    # Deepgram words within the bounded wait window remain pending; older ones become primary_only.
    if not gl_words:
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

    all_sources = set(dg_by_source.keys()) | set(gl_by_source.keys())
    if not all_sources:
        all_sources = {source}

    combined = []
    settled_boundary = max(0.0, audio_clock - GLADIA_BOUNDED_WAIT_S)

    for src in all_sources:
        src_dg = sorted(dg_by_source.get(src, []), key=lambda w: w.get("start", 0.0))
        src_gl = sorted(gl_by_source.get(src, []), key=lambda w: w.get("start", 0.0))
        aligned = align_word_sequences(
            src_dg, src_gl, source=src,
            gladia_settled_boundary=settled_boundary, audio_clock=audio_clock
        )
        combined.extend(aligned)

    combined.sort(key=lambda w: (w.get("start") or 0.0))
    return combined
