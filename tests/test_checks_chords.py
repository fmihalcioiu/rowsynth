"""Tests for rowsynth.checks.chords."""
from __future__ import annotations

import unittest

from rowsynth.checks import Finding, chords, run, sounds
from rowsynth.errors import RowSynthError
from rowsynth.project import Project

RATE = 44100

INSTRUMENTS = {"reed": {"vol": [12, 9, 6]}, "thud": {"wave": "tri", "vol": [15, 8, 0]}}

# The notes of each: a c e, f a c, g b d f, c e g.
CHORDS = {
    "Am": {"root": 9, "tones": [9, 0, 4]},
    "F": {"root": 5, "tones": [5, 9, 0]},
    "G7": {"root": 7, "tones": [7, 11, 2, 5]},
    "C": {"root": 0, "tones": [0, 4, 7]},
}


def song_of(patterns: dict, bars: dict | None = None, **more: object) -> dict:
    """A song of those patterns for a lead, a bass and a kit of one drum, with those chords if any are given."""
    song = {
        "row_ticks": 6,
        "drums": {"k": {"inst": "thud", "note": "c2"}},
        "channels": {"lead": {"inst": "reed"}, "bass": {"inst": "reed"}, "kit": {"inst": "thud", "drums": True}},
        "patterns": patterns,
        "order": list(patterns),
        **more,
    }
    if bars is not None:
        song["chords"] = bars
    return song


def project_of(**songs: dict) -> Project:
    """A project with the instruments and the chords above and these songs."""
    return Project.from_dict({"instruments": INSTRUMENTS, "chords": CHORDS, "songs": songs})


def tune_of(pattern: dict, bars: list[str], **more: object) -> Project:
    """A project of one song, tune, with one pattern, a: the text of its channels and the chords of its bars."""
    return project_of(tune=song_of({"a": pattern}, {"a": bars}, **more))


def found_in(pattern: dict, bars: list[str], **more: object) -> list[Finding]:
    """The findings on the one pattern of such a project."""
    return chords.check(tune_of(pattern, bars, **more), RATE)


def off(note: str, bar: int, chord: str, channel: str = "lead", place: str = "tune.patterns.a") -> Finding:
    """The finding on a note that starts on a beat of that bar and is not in the chord under it."""
    return Finding("chords", f"songs.{place}",
                   f"channel {channel!r} starts a beat of bar {bar} on {note}, which is not in the chord {chord}")


class ChordsTest(unittest.TestCase):
    def refusal_of(self, project: Project) -> str:
        """What the check refuses a project with."""
        with self.assertRaises(RowSynthError) as caught:
            chords.check(project, RATE)
        return str(caught.exception)

    def test_a_tune_that_keeps_to_its_chords_gives_nothing(self):
        pattern = {"lead": "o4 l4 a > c e c | < f a g b", "bass": "o2 l2 a a | f g", "kit": "l4 k k k k | k k k k"}
        self.assertEqual(found_in(pattern, ["Am", "F G7"]), [])

    def test_a_note_on_a_beat_that_is_not_in_the_chord_is_found_with_its_channel_bar_and_chord(self):
        # The third beat of the second bar is an e flat, and a finding spells every note with a sharp.
        self.assertEqual(found_in({"lead": "o4 l4 a > c e c | < f a e- b"}, ["Am", "F G7"]), [
            Finding("chords", "songs.tune.patterns.a",
                    "channel 'lead' starts a beat of bar 2 on d#, which is not in the chord G7")])

    def test_the_same_note_off_the_beat_is_not_found(self):
        self.assertEqual(found_in({"lead": "o4 l4 a > c e c | < f a g8 e-8 b"}, ["Am", "F G7"]), [])

    def test_two_chords_in_a_bar_have_half_of_it_each(self):
        # An a is in F and not in G7, and a b is in G7 and not in F.
        self.assertEqual(found_in({"lead": "o4 l4 a b a b"}, ["F G7"]), [off("b", 1, "F"), off("a", 1, "G7")])

    def test_three_chords_in_a_bar_have_a_third_of_it_each(self):
        # An e is in C and in neither of the others. The second beat is still in the first third of the bar.
        self.assertEqual(found_in({"lead": "o4 l4 e e e e"}, ["C F G7"]), [off("e", 1, "F"), off("e", 1, "G7")])

    def test_a_drum_channel_is_not_held_to_the_chords(self):
        # The drum is tuned to a c, which G7 does not have. A bass that plays the same c is found.
        pattern = {"lead": "o4 l4 g b > d f", "kit": "l4 k k k k"}
        self.assertEqual(found_in(pattern, ["G7"]), [])
        self.assertEqual(found_in({**pattern, "bass": "o2 l1 c"}, ["G7"]), [off("c", 1, "G7", channel="bass")])

    def test_a_channel_shorter_than_its_pattern_is_not_held_to_the_chords(self):
        lead = "o4 l4 g b > d f | < g b > d f"
        # One bar of a bass that is off the chord, under two bars of lead.
        self.assertEqual(found_in({"lead": lead, "bass": "o2 l1 c"}, ["G7", "G7"]), [])
        # With a bar of rest after it the bass is as long as the pattern, and is found.
        self.assertEqual(found_in({"lead": lead, "bass": "o2 l1 c r"}, ["G7", "G7"]),
                         [off("c", 1, "G7", channel="bass")])

    def test_a_pattern_is_as_long_as_its_longest_channel_even_if_that_plays_drums(self):
        # Two bars of drums over one bar of lead: two chords are the right count, and the lead is the short one.
        self.assertEqual(found_in({"lead": "o4 l4 a a a a", "kit": "l1 k k"}, ["G7", "G7"]), [])

    def test_fewer_chords_than_bars_is_one_finding_and_the_bars_after_them_are_passed_over(self):
        # Every note of the second bar would be off the one chord there is.
        self.assertEqual(found_in({"lead": "o4 l4 c e g c | c+ d+ f+ a"}, ["C"]), [
            Finding("chords", "songs.tune.patterns.a", "its length in bars is 2, and chords are written for 1")])

    def test_more_chords_than_bars_is_one_finding(self):
        self.assertEqual(found_in({"lead": "o4 l4 c e g c"}, ["C", "F", "G7"]), [
            Finding("chords", "songs.tune.patterns.a", "its length in bars is 1, and chords are written for 3")])
        # Three beats are three quarters of a bar, and no count of chords is that.
        self.assertEqual(found_in({"lead": "o4 l4 c e g"}, ["C"]), [
            Finding("chords", "songs.tune.patterns.a", "its length in bars is 0.75, and chords are written for 1")])

    def test_a_song_with_no_chords_and_a_pattern_with_none_are_not_checked(self):
        sharps = {"lead": "o4 l4 c+ d+ f+ g+"}
        project = project_of(tune=song_of({"a": {"lead": "o4 l4 c e g c"}, "b": sharps}, {"a": ["C"]}),
                             other=song_of({"a": sharps}), empty=song_of({"a": sharps}, {}))
        self.assertEqual(chords.check(project, RATE), [])
        self.assertEqual(chords.check(Project(), RATE), [])

    def test_the_beats_and_the_bars_are_counted_in_the_rows_per_beat_of_the_song(self):
        # Eight rows to a beat: the d eighths start on rows 4, 12, 20 and 28, between the beats, and a bar is 32 rows.
        lead = "o4 l8 c d e d g d c d | l4 g b > d e"
        self.assertEqual(found_in({"lead": lead}, ["C", "G7"], rows_per_beat=8), [off("e", 2, "G7")])
        # Three rows to a beat: the fourth beat of the first bar starts on row 9, and the second bar on row 12.
        self.assertEqual(found_in({"lead": "o4 l4 c e g a | g b > d c"}, ["C", "G7"], rows_per_beat=3),
                         [off("a", 1, "C"), off("c", 2, "G7")])

    def test_findings_come_by_song_then_by_pattern_as_chords_has_them_then_by_channel_then_in_time(self):
        wrong = {"lead": "o4 l4 c d e f", "bass": "o2 l2 c b"}
        project = project_of(tune=song_of({"a": wrong, "b": wrong}, {"b": ["C", "C"], "a": ["C"]}),
                             other=song_of({"a": {"lead": "o4 l1 a"}}, {"a": ["C"]}))
        self.assertEqual(chords.check(project, RATE), [
            Finding("chords", "songs.tune.patterns.b", "its length in bars is 1, and chords are written for 2"),
            off("d", 1, "C", place="tune.patterns.b"),
            off("f", 1, "C", place="tune.patterns.b"),
            off("b", 1, "C", channel="bass", place="tune.patterns.b"),
            off("d", 1, "C"),
            off("f", 1, "C"),
            off("b", 1, "C", channel="bass"),
            off("a", 1, "C", place="other.patterns.a")])

    def test_run_gives_the_chord_findings_before_the_sound_findings(self):
        loud = {"def": {"voices": [{"inst": {"gain": 8}, "notes": [[0, "a4", 4]]}]}}
        project = Project.from_dict({"instruments": INSTRUMENTS, "chords": CHORDS, "sounds": {"zap": loud},
                                     "songs": {"tune": song_of({"a": {"lead": "o4 l1 d"}}, {"a": ["C"]})}})
        tune, zap = chords.check(project, RATE), sounds.check(project, RATE)
        self.assertEqual(tune, [off("d", 1, "C")])
        self.assertEqual([finding.rule for finding in zap], ["peak"])
        self.assertEqual(run(project), tune + zap)

    def test_text_that_cannot_be_read_is_refused_with_its_song_pattern_and_channel_in_front(self):
        self.assertEqual(self.refusal_of(tune_of({"lead": "o4 l4 c e g c", "bass": "o2 l2 c z"}, ["C"])),
                         'songs.tune.patterns.a.bass: cannot read "z" in a channel')

    def test_a_bar_that_names_no_chord_is_refused_with_its_place_in_chords(self):
        self.assertEqual(self.refusal_of(tune_of({"lead": "o4 l1 c c"}, ["C", " "])),
                         "songs.tune.chords.a[1]: names no chord")

    def test_a_chord_the_project_does_not_have_is_refused_by_name(self):
        # Loading lets no such chord through, so this one is taken away by hand.
        project = tune_of({"lead": "o4 l1 c c"}, ["C", "F"])
        del project.chords["F"]
        self.assertEqual(self.refusal_of(project), "songs.tune.chords.a[1]: no chord named 'F'")


if __name__ == "__main__":
    unittest.main()
