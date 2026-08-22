"""Sentence extraction for the live feedback lane.

Turns are the wrong unit to send to a language model. `group_turns` cuts on
speaker change or an 8s gap, so a turn is routinely mid-thought, and it is
re-derived from the flat word list on every render, so it mutates as words land.
A sentence cut on terminal punctuation is a complete thought and, once it has
settled, never changes again — which is what makes it safe to hang an analysis on.

Everything here is pure: no Streamlit, no network, no globals. That keeps it
testable (see tests/test_live_sentences.py) and safe to call from the 0.5s
render loop.
"""

import hashlib

LIVE_SENTENCE_GAP_S = 2.0        # silence inside one speaker that ends a sentence
LIVE_SENTENCE_MAX_WORDS = 40     # hard cut for runaway unpunctuated speech

SENTENCE_END = (".", "?", "!", "…")

# Tokens ending in "." that do not end the sentence. Bare initials ("B." while
# spelling a word out) are handled separately — they occur in the real transcripts.
SENTENCE_ABBREV = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "e.g.", "i.e.", "etc.", "vs.",
    "no.", "jr.", "sr.",
}


def ends_sentence(token):
    """True when this token closes a sentence rather than just carrying a dot."""
    if not token.endswith(SENTENCE_END):
        return False
    if token.lower() in SENTENCE_ABBREV:
        return False
    if len(token) == 2 and token[0].isalpha() and token.endswith("."):
        return False
    return True


def split_sentences(words, source, gap_s=LIVE_SENTENCE_GAP_S,
                    max_words=LIVE_SENTENCE_MAX_WORDS):
    """Sentence records for one source, derived deterministically from `words`.

    A sentence is cut on a silence gap, on terminal punctuation, or at a hard
    word ceiling. The id carries a hash of the text, so a late-arriving word that
    re-shapes a sentence yields a new id — and therefore a fresh analysis —
    rather than leaving stale feedback attached to text that no longer matches.

    The returned records hold references to the very same word dicts that were
    passed in, which is what lets the renderer map words back to sentences by
    object identity.
    """
    sentences = []
    buffer = []

    def flush():
        if not buffer:
            return
        tokens = [(w.get("word") or "").strip() for w in buffer]
        text = " ".join(t for t in tokens if t)
        if not text:
            buffer.clear()
            return
        start = buffer[0].get("start") or 0.0
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
        sentences.append({
            "id": f"{source}:{start:.2f}:{digest}",
            "source": source,
            "start": start,
            "end": buffer[-1].get("end") or start,
            "text": text,
            "words": list(buffer),
            "word_count": len(buffer),
        })
        buffer.clear()

    for word in words:
        if word.get("source") != source:
            continue
        token = (word.get("word") or "").strip()
        if not token:
            continue
        if buffer:
            gap = (word.get("start") or 0.0) - (buffer[-1].get("end") or 0.0)
            if gap > gap_s:
                flush()
        buffer.append(word)
        if ends_sentence(token) or len(buffer) >= max_words:
            flush()
    flush()
    return sentences


def newest_word_end(words):
    """Latest end time across all sources; the clock the settle rule runs on."""
    return max((w.get("end") or 0.0) for w in words) if words else 0.0


def settled_sentences(sentences, boundary):
    """Sentences far enough in the past that no further word can re-shape them.

    The two Deepgram sockets finalize independently, so a word from one mic can
    arrive after a later word from the other is already final.
    """
    return [s for s in sentences if s["end"] <= boundary]


def index_words_to_sentences(sentences):
    """Map word-dict identity -> sentence id.

    Valid only because the caller derives turns and sentences from one and the
    same word list in a single render, so both hold references to the same dicts.
    """
    index = {}
    for sentence in sentences:
        for word in sentence["words"]:
            index[id(word)] = sentence["id"]
    return index


def low_confidence_words(sentence, threshold):
    """Tokens the recognizer was unsure of, deduped, in order of appearance.

    17% of student words in a real lesson score below 0.70. Handing these to the
    model is the main defence against reporting a mishearing as a learner error.
    """
    seen = set()
    out = []
    for word in sentence["words"]:
        confidence = word.get("confidence")
        if confidence is None or confidence >= threshold:
            continue
        token = (word.get("word") or "").strip().strip(".,?!…")
        if token and token.lower() not in seen:
            seen.add(token.lower())
            out.append(token)
    return out


# ---------------------------------------------------------------------------
# Armed windows
#
# The live-feedback switch selects a stretch of SPEECH, not a moment of
# dispatch. Windows are recorded on the lesson's audio clock, so whether a
# sentence gets analysed depends on when it was spoken - never on where the
# switch happens to be when the sentence finishes settling.
# ---------------------------------------------------------------------------
def update_feedback_windows(windows, switch_on, was_on, now_audio):
    """Open or close the current window on a switch edge. Mutates `windows`."""
    if switch_on and not was_on:
        windows.append([now_audio, None])
    elif was_on and not switch_on:
        if windows and windows[-1][1] is None:
            windows[-1][1] = now_audio
    return windows


def in_feedback_window(sentence, windows):
    """True when the sentence was spoken inside an armed window.

    Overlap, not containment: a sentence already under way when the switch is
    thrown counts at both edges, which is what "said while it was on" means to
    someone flipping a switch mid-conversation. An open window (end None) runs
    to the end of the lesson.
    """
    start = sentence["start"]
    end = sentence["end"]
    for window_start, window_end in windows:
        if end >= window_start and (window_end is None or start <= window_end):
            return True
    return False
