"""Grouping words into speaker turns.

Kept separate from app.py so it can be imported and tested, and separate from
live_sentences.py because it serves a different lane: turns are what gets
displayed, saved and handed to the synthesis model, while sentences are the
unit the live feedback pass analyses.

The rule that matters here is that grouping runs **per speaker**. The two
microphones record simultaneously, so the merged timeline interleaves
overlapping speech word by word:

    148.11  student  Have
    148.16  teacher  for
    148.40  teacher  some
    148.43  student  you
    148.59  student  been
    148.78  teacher  time.
    148.83  student  to
    149.15  student  Sri
    149.39  student  Lanka?

Breaking a turn on every speaker change turns that into three student
fragments — "Have", "you been", "to Sri Lanka?" — even though the student said
one uninterrupted sentence in 1.6 seconds. Grouped per speaker, the teacher's
overlapping words form their own turn and the student's sentence stays whole.
"""

from live_sentences import ends_sentence

TURN_GAP_SECONDS = 2.0       # silence that ends a turn, once a sentence has closed
TURN_HARD_GAP_SECONDS = 8.0  # silence that ends a turn even mid-sentence
TURN_MAX_SECONDS = 60.0      # ceiling, so one speaker's monologue stays readable


def group_turns(words, gap_s=TURN_GAP_SECONDS, max_s=TURN_MAX_SECONDS,
                hard_gap_s=TURN_HARD_GAP_SECONDS):
    """Timestamped turns, ordered by when each began.

    A turn breaks only on its own speaker's silence or the length ceiling —
    never because the other microphone picked something up.

    A silence mid-sentence does not end a turn. A learner searching for the
    next word pauses for seconds in the middle of a thought, and treating that
    as a turn boundary chops the thought into pieces: in one real lesson 102 of
    the student's pauses ran past 2s without the sentence having closed, and
    only 14 ran past 8s. So an ordinary gap ends a turn only once the last word
    closed a sentence, and anything longer than hard_gap_s ends it regardless —
    an abandoned fragment must not glue itself to whatever is said next.

    The word dicts are passed through by reference, which is what lets the
    renderer map words back to sentences by identity.
    """
    by_source = {}
    for word in words:
        if not (word.get("word") or "").strip():
            continue
        by_source.setdefault(word.get("source"), []).append(word)

    turns = []
    for source, source_words in by_source.items():
        source_words.sort(key=lambda w: (w.get("start") or 0.0))
        current = None
        for word in source_words:
            start = word.get("start") or 0.0
            end = word.get("end") or start
            gap = start - current["end"] if current else 0.0
            if (
                current is None
                or gap > hard_gap_s
                or (gap > gap_s and ends_sentence(current["tokens"][-1]))
                or start - current["start"] > max_s
            ):
                current = {
                    "source": source,
                    "start": start,
                    "end": end,
                    "tokens": [],
                    "words": [],
                }
                turns.append(current)
            current["tokens"].append((word.get("word") or "").strip())
            current["words"].append(word)
            # max(), not assignment: within a turn the words are ordered by
            # start, but a long word can still end after the one following it.
            current["end"] = max(current["end"], end)

    turns.sort(key=lambda t: (t["start"], t["source"]))
    for turn in turns:
        turn["text"] = " ".join(turn["tokens"])
    return turns
