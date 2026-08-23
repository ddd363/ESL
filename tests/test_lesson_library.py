"""Guards on finding lessons on disk and reading Deepgram's batch replies.

These are the two places a loaded lesson can go quietly wrong: a lesson that
is invisible in the picker looks like lost work, and a mis-shaped word record
poisons every downstream step without ever raising.
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lesson_library as ll


def touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("x")


class ClassifyTrackTests(unittest.TestCase):
    def test_speaker_word_anywhere_in_the_name(self):
        # Imported lessons keep the names they arrived with.
        self.assertEqual(ll.classify_track("student.webm"), "student")
        self.assertEqual(ll.classify_track("fabiana l3 teacher.mp3"), "teacher")
        self.assertEqual(ll.classify_track("/tmp/a/Student Track.WAV"), "student")

    def test_derived_artefacts_are_not_audio(self):
        # These sit beside the tracks and must never be mistaken for one.
        for name in ("student.replicate.json", "transcript.txt", "words.json",
                     "student.deepgram.json", "live_feedback.json"):
            self.assertIsNone(ll.classify_track(name), name)

    def test_unlabelled_audio_is_left_unclassified(self):
        self.assertIsNone(ll.classify_track("lesson.mp3"))


class LessonTracksTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_labelled_pair(self):
        touch(os.path.join(self.root, "L", "fabiana l3 student.mp3"))
        touch(os.path.join(self.root, "L", "fabiana l3 teacher.mp3"))
        tracks = ll.lesson_tracks(os.path.join(self.root, "L"))
        self.assertEqual(sorted(tracks), ["student", "teacher"])

    def test_lone_unlabelled_file_is_the_student(self):
        """The student track is what every downstream step needs."""
        touch(os.path.join(self.root, "L", "recording.m4a"))
        tracks = ll.lesson_tracks(os.path.join(self.root, "L"))
        self.assertEqual(sorted(tracks), ["student"])
        self.assertTrue(tracks["student"].endswith("recording.m4a"))

    def test_several_unlabelled_files_are_not_guessed_at(self):
        touch(os.path.join(self.root, "L", "part1.m4a"))
        touch(os.path.join(self.root, "L", "part2.m4a"))
        self.assertEqual(ll.lesson_tracks(os.path.join(self.root, "L")), {})

    def test_labelled_file_wins_over_the_unlabelled_fallback(self):
        touch(os.path.join(self.root, "L", "student.webm"))
        touch(os.path.join(self.root, "L", "notes audio.mp3"))
        tracks = ll.lesson_tracks(os.path.join(self.root, "L"))
        self.assertEqual(sorted(tracks), ["student"])
        self.assertTrue(tracks["student"].endswith("student.webm"))

    def test_continuous_recording_wins_over_the_browser_copy(self):
        """Both are real; the .ogg is the one whose timeline matches the words."""
        touch(os.path.join(self.root, "L", "student.webm"))
        touch(os.path.join(self.root, "L", "student.ogg"))
        tracks = ll.lesson_tracks(os.path.join(self.root, "L"))
        self.assertTrue(tracks["student"].endswith("student.ogg"))

    def test_longer_take_wins_among_equals(self):
        a = os.path.join(self.root, "L", "student.ogg")
        b = os.path.join(self.root, "L", "student.take2.ogg")
        touch(a)
        touch(b)
        with open(b, "w", encoding="utf-8") as f:
            f.write("x" * 500)
        self.assertEqual(ll.lesson_tracks(os.path.join(self.root, "L"))["student"], b)

    def test_missing_folder_is_empty_not_an_error(self):
        self.assertEqual(ll.lesson_tracks(os.path.join(self.root, "nope")), {})


class ListLessonsTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_any_folder_counts_not_only_lesson_prefixed_ones(self):
        touch(os.path.join(self.root, "lesson_20260101_101010", "student.webm"))
        touch(os.path.join(self.root, "fabian L2", "fabiana l3 student.mp3"))
        names = {entry["name"] for entry in ll.list_lessons(self.root)}
        self.assertEqual(names, {"lesson_20260101_101010", "fabian L2"})

    def test_transcript_only_lesson_still_listed(self):
        """Audio can be cleared out; the transcript is still worth reopening."""
        touch(os.path.join(self.root, "old", "words.json"))
        entries = ll.list_lessons(self.root)
        self.assertEqual([e["name"] for e in entries], ["old"])
        self.assertTrue(entries[0]["has_words"])
        self.assertEqual(entries[0]["tracks"], {})

    def test_empty_and_hidden_folders_are_skipped(self):
        os.makedirs(os.path.join(self.root, "empty"))
        touch(os.path.join(self.root, ".Trash", "student.webm"))
        self.assertEqual(ll.list_lessons(self.root), [])

    def test_newest_first(self):
        touch(os.path.join(self.root, "older", "student.webm"))
        touch(os.path.join(self.root, "newer", "student.webm"))
        os.utime(os.path.join(self.root, "older", "student.webm"), (1_000_000, 1_000_000))
        os.utime(os.path.join(self.root, "newer", "student.webm"), (2_000_000, 2_000_000))
        self.assertEqual([e["name"] for e in ll.list_lessons(self.root)], ["newer", "older"])

    def test_missing_audio_dir_is_empty_not_an_error(self):
        self.assertEqual(ll.list_lessons(os.path.join(self.root, "nope")), [])


class StudentRosterTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.path = os.path.join(self.root, "students.json")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_a_name_survives_to_the_next_session(self):
        ll.add_student("Maria Silva", self.path)
        self.assertEqual(ll.load_students(self.path), ["Maria Silva"])

    def test_names_are_tidied_before_saving(self):
        _, cleaned = ll.add_student("   Maria    Silva  ", self.path)
        self.assertEqual(cleaned, "Maria Silva")

    def test_the_same_student_is_not_added_twice(self):
        ll.add_student("Maria", self.path)
        roster, cleaned = ll.add_student("maria", self.path)
        self.assertEqual(roster, ["Maria"])
        self.assertEqual(cleaned, "Maria", "existing spelling should win")

    def test_roster_is_alphabetical_regardless_of_case(self):
        for name in ("zoe", "Ana", "maria"):
            ll.add_student(name, self.path)
        self.assertEqual(ll.load_students(self.path), ["Ana", "maria", "zoe"])

    def test_blank_names_are_refused(self):
        roster, cleaned = ll.add_student("   ", self.path)
        self.assertEqual((roster, cleaned), ([], ""))

    def test_a_missing_or_corrupt_roster_reads_as_empty(self):
        self.assertEqual(ll.load_students(self.path), [])
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("not json{{{")
        self.assertEqual(ll.load_students(self.path), [])


class SlugTests(unittest.TestCase):
    def test_names_become_safe_folder_components(self):
        self.assertEqual(ll.slugify_student("Maria Silva"), "maria-silva")
        self.assertEqual(ll.slugify_student("  Ana   B.  "), "ana-b")

    def test_path_separators_cannot_survive_a_slug(self):
        """A typed name must never be able to escape the audio folder."""
        for hostile in ("../../etc", "a/b", "..", "/root"):
            slug = ll.slugify_student(hostile)
            self.assertNotIn("/", slug)
            self.assertNotIn("..", slug)

    def test_a_name_with_nothing_usable_slugs_to_empty(self):
        self.assertEqual(ll.slugify_student("///"), "")
        self.assertEqual(ll.slugify_student(""), "")


class LessonMetaTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_the_student_is_read_back_from_the_folder(self):
        lesson = os.path.join(self.root, "maria-silva_20260822_1830")
        ll.write_lesson_meta(lesson, student="Maria Silva", recorded="2026-08-22 18:30")
        entry = ll.lesson_entry(lesson)
        self.assertEqual(entry["student"], "Maria Silva")
        self.assertIn("Maria Silva", ll.describe_lesson(entry))

    def test_renaming_the_folder_does_not_lose_the_student(self):
        lesson = os.path.join(self.root, "whatever")
        ll.write_lesson_meta(lesson, student="Maria Silva")
        self.assertEqual(ll.lesson_entry(lesson)["student"], "Maria Silva")

    def test_a_lesson_without_metadata_falls_back_to_its_folder_name(self):
        lesson = os.path.join(self.root, "fabian L2")
        touch(os.path.join(lesson, "student.mp3"))
        self.assertIn("fabian L2", ll.describe_lesson(ll.lesson_entry(lesson)))

    def test_metadata_is_not_mistaken_for_audio(self):
        lesson = os.path.join(self.root, "L")
        ll.write_lesson_meta(lesson, student="Maria")
        self.assertEqual(ll.lesson_tracks(lesson), {})


def reply(words):
    return {"results": {"channels": [{"alternatives": [{"words": words}]}]}}


class WordsFromPrerecordedTests(unittest.TestCase):
    def test_records_match_what_the_live_socket_appends(self):
        words = ll.words_from_prerecorded(
            reply([{"word": "hello", "punctuated_word": "Hello,",
                    "start": 1.0, "end": 1.4, "confidence": 0.98}]),
            "student",
        )
        self.assertEqual(words, [{
            "source": "student", "word": "Hello,",
            "start": 1.0, "end": 1.4, "confidence": 0.98,
        }])

    def test_punctuated_form_is_preferred(self):
        """Sentence boundaries are what the live feedback pass splits on."""
        words = ll.words_from_prerecorded(
            reply([{"word": "yes", "punctuated_word": "Yes.", "start": 0, "end": 1}]),
            "teacher",
        )
        self.assertEqual(words[0]["word"], "Yes.")

    def test_blank_and_malformed_entries_are_dropped(self):
        words = ll.words_from_prerecorded(
            reply([{"word": "   ", "start": 0, "end": 1},
                   "not a dict",
                   {"word": "ok", "start": 2, "end": 3}]),
            "student",
        )
        self.assertEqual([w["word"] for w in words], ["ok"])

    def test_missing_timings_become_zero_rather_than_none(self):
        """group_turns() does arithmetic on start/end; None would raise."""
        words = ll.words_from_prerecorded(reply([{"word": "hi"}]), "student")
        self.assertEqual((words[0]["start"], words[0]["end"]), (0.0, 0.0))

    def test_offset_shifts_the_whole_track(self):
        words = ll.words_from_prerecorded(
            reply([{"word": "hi", "start": 2.0, "end": 2.5}]), "student", offset=10.0
        )
        self.assertEqual((words[0]["start"], words[0]["end"]), (12.0, 12.5))

    def test_empty_and_broken_replies_yield_nothing(self):
        for payload in ({}, None, {"results": {}}, {"results": {"channels": []}},
                        {"results": {"channels": [{"alternatives": []}]}},
                        {"results": {"channels": ["nope"]}}):
            self.assertEqual(ll.words_from_prerecorded(payload, "student"), [], payload)


class MergeWordsTests(unittest.TestCase):
    def test_tracks_interleave_into_one_timeline(self):
        student = ll.words_from_prerecorded(
            reply([{"word": "a", "start": 0.0, "end": 0.5},
                   {"word": "c", "start": 4.0, "end": 4.5}]), "student")
        teacher = ll.words_from_prerecorded(
            reply([{"word": "b", "start": 2.0, "end": 2.5}]), "teacher")
        merged = ll.merge_words(student, teacher)
        self.assertEqual([w["word"] for w in merged], ["a", "b", "c"])
        self.assertEqual([w["source"] for w in merged],
                         ["student", "teacher", "student"])


if __name__ == "__main__":
    unittest.main()
