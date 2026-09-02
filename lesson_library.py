"""Finding past lessons on disk, and reading Deepgram's pre-recorded replies.

Kept out of app.py so it can be imported and tested: app.py is a Streamlit
script, and importing it runs the whole page.

A lesson is a folder under audio/ holding up to two tracks, one per
microphone. The live recorder writes student.webm / teacher.webm, but a lesson
recorded elsewhere and dropped into audio/ keeps whatever names it arrived
with, so a track is identified by the speaker word anywhere in its filename
("fabiana l3 student.mp3" is the student track).
"""

import json
import os

LESSON_AUDIO_EXTS = {".wav", ".m4a", ".webm", ".ogg", ".mp4", ".mp3"}
SOURCES = ("student", "teacher")
# When a microphone has more than one file, which one the app works from.
# The continuous server-side recording comes first: it is the one that always
# exists, and its timeline is byte-derived from the same stream that produced
# the word timestamps, so audio positions and transcript times agree. The
# browser's own end-of-lesson encode is a complete second copy when it arrives.
TRACK_PREFERENCE = (".ogg", ".wav", ".webm", ".m4a", ".mp3", ".mp4")


def is_audio(filename):
    return os.path.splitext(filename)[1].lower() in LESSON_AUDIO_EXTS


def classify_track(filename):
    """Which microphone a file belongs to, or None when the name doesn't say.

    Derived artefacts (student.replicate.json, transcript.txt) fail the
    extension test first, so only real audio is ever classified.
    """
    if not is_audio(filename):
        return None
    stem = os.path.splitext(os.path.basename(filename))[0].lower()
    for source in SOURCES:
        if source in stem:
            return source
    return None


def lesson_tracks(lesson_dir):
    """{"student": path, "teacher": path} for one lesson folder.

    A folder holding exactly one unlabelled audio file is read as student-only:
    that is the track every downstream step needs, and a single dropped-in
    recording is almost always the student. Two or more unlabelled files are
    left alone rather than guessed at.
    """
    candidates = {}
    unlabelled = []
    try:
        names = sorted(os.listdir(lesson_dir))
    except OSError:
        return {}
    for name in names:
        path = os.path.join(lesson_dir, name)
        if not os.path.isfile(path) or not is_audio(name):
            continue
        source = classify_track(name)
        if source is None:
            unlabelled.append(path)
        else:
            candidates.setdefault(source, []).append(path)

    def rank(path):
        ext = os.path.splitext(path)[1].lower()
        order = TRACK_PREFERENCE.index(ext) if ext in TRACK_PREFERENCE else len(TRACK_PREFERENCE)
        # Among equals prefer the bigger file: a longer take of the same lesson.
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        return (order, -size, path)

    tracks = {source: sorted(paths, key=rank)[0] for source, paths in candidates.items()}
    if not tracks and len(unlabelled) == 1:
        tracks["student"] = unlabelled[0]
    return tracks


def lesson_entry(lesson_dir):
    """One lesson described well enough to pick it out of a list."""
    tracks = lesson_tracks(lesson_dir)
    words_path = os.path.join(lesson_dir, "words.json")
    words_con_path = os.path.join(lesson_dir, "words_consensus.json")
    transcript_path = os.path.join(lesson_dir, "transcript.txt")
    times = [os.path.getmtime(p) for p in tracks.values() if os.path.exists(p)]
    for path in (words_path, words_con_path, transcript_path):
        if os.path.exists(path):
            times.append(os.path.getmtime(path))
    meta = read_lesson_meta(lesson_dir)
    return {
        "dir": lesson_dir,
        "name": os.path.basename(lesson_dir),
        "student": meta.get("student") or "",
        "recorded": meta.get("recorded") or "",
        "tracks": tracks,
        "has_words": os.path.exists(words_path) or os.path.exists(words_con_path),
        "has_transcript": os.path.exists(transcript_path),
        "modified": max(times) if times else 0.0,
    }


def list_lessons(audio_dir):
    """Every folder under audio_dir holding a lesson, newest first.

    Any folder counts, not just the recorder's lesson_* ones — a lesson
    recorded elsewhere is dropped in under whatever name it had. A folder with
    a saved transcript but no audio still counts: it can be read and
    re-synthesised even after the audio has been cleared out.
    """
    if not os.path.isdir(audio_dir):
        return []
    entries = []
    for name in sorted(os.listdir(audio_dir)):
        path = os.path.join(audio_dir, name)
        if not os.path.isdir(path) or name.startswith("."):
            continue
        entry = lesson_entry(path)
        if entry["tracks"] or entry["has_words"] or entry["has_transcript"]:
            entries.append(entry)
    entries.sort(key=lambda e: e["modified"], reverse=True)
    return entries


def describe_lesson(entry):
    """Short summary for the picker, so a lesson is chosen on what it holds.

    Led by the student's name where one was recorded — that is what a teacher
    is actually looking for. Lessons from before names existed, and any folder
    dropped in by hand, fall back to the folder name.
    """
    bits = []
    have = [s for s in SOURCES if s in entry["tracks"]]
    bits.append(" + ".join(have) + " audio" if have else "no audio")
    if entry["has_words"]:
        bits.append("transcribed")
    label = entry["name"]
    if entry.get("student"):
        when = entry.get("recorded") or ""
        label = f"{entry['student']} · {when}" if when else entry["student"]
    return f"{label} — {', '.join(bits)}"


# ---------------------------------------------------------------------------
# Deepgram pre-recorded replies
#
# The batch API returns the same word objects the live socket does, just all at
# once, so they are converted into exactly the records the live path appends to
# lesson_words. Everything downstream — turn grouping, the transcript pane, the
# confidence viewpoint, the synthesis bundle — then cannot tell the two apart.
# ---------------------------------------------------------------------------
def words_from_prerecorded(payload, source, offset=0.0):
    """Deepgram's JSON -> the app's word records for one microphone."""
    results = (payload or {}).get("results") or {}
    channels = results.get("channels") or []
    if not channels or not isinstance(channels[0], dict):
        return []
    alternatives = channels[0].get("alternatives") or []
    if not alternatives or not isinstance(alternatives[0], dict):
        return []
    records = []
    for item in alternatives[0].get("words") or []:
        if not isinstance(item, dict):
            continue
        # Punctuated first, matching the live socket: the transcript keeps its
        # sentence boundaries, which the live feedback pass splits on.
        token = (item.get("punctuated_word") or item.get("word") or "").strip()
        if not token:
            continue
        start = item.get("start")
        end = item.get("end")
        records.append({
            "source": source,
            "word": token,
            "start": (start if isinstance(start, (int, float)) else 0.0) + offset,
            "end": (end if isinstance(end, (int, float)) else 0.0) + offset,
            "confidence": item.get("confidence"),
        })
    return records


def merge_words(*groups):
    """One timeline from the per-microphone word lists."""
    words = [w for group in groups for w in group]
    words.sort(key=lambda w: (w.get("start") or 0.0))
    return words


# ---------------------------------------------------------------------------
# Students
#
# A lesson belongs to someone, and the teacher picks that person before
# pressing record. The roster is kept in one small file so a name typed once
# comes back as a dropdown entry next time, and the chosen name is written into
# the lesson folder as well — the folder name is only a hint, the metadata is
# what the app reads back.
# ---------------------------------------------------------------------------
STUDENTS_PATH = os.path.join("outputs", "students.json")
LESSON_META_NAME = "lesson.json"


def slugify_student(name):
    """Folder-safe form of a name: 'Maria Silva' -> 'maria-silva'."""
    kept = "".join(c if (c.isalnum() or c in " -_") else " " for c in (name or ""))
    return "-".join(kept.split()).strip("-_").lower()


def clean_student_name(name):
    """The name as it will be stored and shown: trimmed, spaces collapsed."""
    return " ".join((name or "").split())


def load_students(path=STUDENTS_PATH):
    """Saved student names, in the order they should appear in the dropdown."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []
    names = payload.get("students") if isinstance(payload, dict) else payload
    if not isinstance(names, list):
        return []
    seen = set()
    ordered = []
    for name in names:
        cleaned = clean_student_name(name if isinstance(name, str) else "")
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            ordered.append(cleaned)
    return ordered


def save_students(names, path=STUDENTS_PATH):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "students": names}, f, ensure_ascii=False, indent=2)


def add_student(name, path=STUDENTS_PATH):
    """Add a name to the roster, returning (roster, cleaned_name).

    Matching is case-insensitive, so "maria" does not become a second Maria.
    An existing entry keeps its original spelling rather than being rewritten.
    """
    cleaned = clean_student_name(name)
    if not cleaned:
        return load_students(path), ""
    roster = load_students(path)
    for existing in roster:
        if existing.lower() == cleaned.lower():
            return roster, existing
    roster.append(cleaned)
    roster.sort(key=str.lower)
    save_students(roster, path)
    return roster, cleaned


def read_lesson_meta(lesson_dir):
    try:
        with open(os.path.join(lesson_dir, LESSON_META_NAME), "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def write_lesson_meta(lesson_dir, student=None, **extra):
    os.makedirs(lesson_dir, exist_ok=True)
    payload = read_lesson_meta(lesson_dir)
    payload.setdefault("version", 1)
    if student:
        payload["student"] = student
    payload.update(extra)
    with open(os.path.join(lesson_dir, LESSON_META_NAME), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload
