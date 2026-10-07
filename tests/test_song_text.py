"""Tests for the text a song's channels are written in."""
from __future__ import annotations

import copy
import unittest

from rowsynth.errors import RowSynthError
from rowsynth.song import parse, transpose, up


# What each drum letter plays in these tests.
DRUMS = {"k": {"inst": "thud", "note": "c2"},
         "s": {"inst": "crack", "note": "d3", "rows": 2},
         "h": {"inst": "tick", "note": 80, "vel": 0.5},
         "H": {"inst": "tick", "note": "f#5", "rows": 3, "vel": 0.8},
         "t": {"inst": "tub", "note": "eb3", "rows": 0.5}}

# A short song of two tuned channels and a drum channel.
SONG = {"row_ticks": 6,
        "instruments": {"lead": {"wave": "pulse"}, "bass": {"wave": "tri"}, "thud": {"wave": "noise"}},
        "drums": {"k": {"inst": "thud", "note": "c2"}},
        "channels": {"tune": {"inst": "lead"}, "low": {"inst": "bass", "pan": -0.3},
                     "beat": {"inst": "thud", "drums": True}},
        "patterns": {"a": {"tune": "l8 c e g > c", "low": "o2 c2", "beat": "k4 k4"},
                     "b": {"tune": "o5 c4 < b4", "beat": "l4 k k"}},
        "order": ["a", "b", "a"]}


def event(row: float, rows: float, note: float, inst: str = "lead", vel: float = 1) -> dict:
    """One event as parse gives it."""
    return {"row": row, "rows": rows, "note": note, "inst": inst, "vel": vel}


class ParseTest(unittest.TestCase):
    def refused(self, text: str, drums: dict | None = None) -> str:
        """The message parse refuses `text` with."""
        with self.assertRaises(RowSynthError) as caught:
            parse(text, "lead", drums=drums)
        return str(caught.exception)

    def test_letters_are_the_notes_of_octave_four_a_sixteenth_each(self):
        events, rows = parse("c d e f g a b", "lead")
        self.assertEqual(events, [event(0, 1, 60), event(1, 1, 62), event(2, 1, 64), event(3, 1, 65),
                                  event(4, 1, 67), event(5, 1, 69), event(6, 1, 71)])
        self.assertEqual(rows, 7)

    def test_a_sharp_adds_a_semitone_and_a_flat_takes_one_away(self):
        events, rows = parse("c+ c# d- e- b+ c-", "lead")
        self.assertEqual(events, [event(0, 1, 61), event(1, 1, 61), event(2, 1, 61), event(3, 1, 63),
                                  event(4, 1, 72), event(5, 1, 59)])
        self.assertEqual(rows, 6)

    def test_a_length_is_the_share_of_a_whole_note(self):
        events, rows = parse("c1 c2 c4 c8 c16 c32", "lead")
        self.assertEqual(events, [event(0, 16, 60), event(16, 8, 60), event(24, 4, 60), event(28, 2, 60),
                                  event(30, 1, 60), event(31, 0.5, 60)])
        self.assertEqual(rows, 31.5)

    def test_each_dot_adds_half_of_what_the_one_before_added(self):
        events, rows = parse("c4. c4.. c8...", "lead")
        self.assertEqual(events, [event(0, 6, 60), event(6, 7, 60), event(13, 3.75, 60)])
        self.assertEqual(rows, 16.75)

    def test_a_rest_takes_its_rows_and_gives_no_event(self):
        events, rows = parse("r4 c r8. d r", "lead")
        self.assertEqual(events, [event(4, 1, 60), event(8, 1, 62)])
        self.assertEqual(rows, 10)

    def test_l_sets_the_length_of_what_gives_none(self):
        events, rows = parse("l8 c d l4 e. r l16 f", "lead")
        self.assertEqual(events, [event(0, 2, 60), event(2, 2, 62), event(4, 6, 64), event(14, 1, 65)])
        self.assertEqual(rows, 15)

    def test_o_sets_the_octave(self):
        events, rows = parse("o2 c o5 c o0 c o9 b", "lead")
        self.assertEqual(events, [event(0, 1, 36), event(1, 1, 72), event(2, 1, 12), event(3, 1, 131)])
        self.assertEqual(rows, 4)

    def test_greater_and_less_move_the_octave_one_at_a_time(self):
        events, rows = parse("c >c >>c <c <<<c", "lead")
        self.assertEqual(events, [event(0, 1, 60), event(1, 1, 72), event(2, 1, 96), event(3, 1, 84),
                                  event(4, 1, 48)])
        self.assertEqual(rows, 5)

    def test_the_octave_can_be_moved_past_what_o_can_write(self):
        events, rows = parse("o0 < c << b o9 > c > b+", "lead")
        self.assertEqual(events, [event(0, 1, 0), event(1, 1, -13), event(2, 1, 132), event(3, 1, 156)])
        self.assertEqual(rows, 4)

    def test_bar_lines_and_space_change_nothing(self):
        events, rows = parse("c4 d4 |\n\te4 f4 |", "lead")
        self.assertEqual(events, [event(0, 4, 60), event(4, 4, 62), event(8, 4, 64), event(12, 4, 65)])
        self.assertEqual(rows, 16)
        self.assertEqual(parse("c4d4e4f4", "lead"), (events, rows))

    def test_text_needs_no_space(self):
        events, rows = parse("l8cdefg+4.>c<b-2", "lead")
        self.assertEqual(events, [event(0, 2, 60), event(2, 2, 62), event(4, 2, 64), event(6, 2, 65),
                                  event(8, 6, 68), event(14, 2, 72), event(16, 8, 70)])
        self.assertEqual(rows, 24)

    def test_rows_per_beat_scales_every_length(self):
        events, rows = parse("c4 d8 e16 f4.", "lead", 2)
        self.assertEqual(events, [event(0, 2, 60), event(2, 1, 62), event(3, 0.5, 64), event(3.5, 3, 65)])
        self.assertEqual(rows, 6.5)
        events, rows = parse("c4 d8 e16 f4.", "lead", rows_per_beat=3)
        self.assertEqual(events, [event(0, 3, 60), event(3, 1.5, 62), event(4.5, 0.75, 64), event(5.25, 4.5, 65)])
        self.assertEqual(rows, 9.75)
        events, rows = parse("c4 d8 e16 f4.", "lead", rows_per_beat=8)
        self.assertEqual(events, [event(0, 8, 60), event(8, 4, 62), event(12, 2, 64), event(14, 12, 65)])
        self.assertEqual(rows, 26)

    def test_four_quarter_notes_take_four_beats_at_any_rows_per_beat(self):
        for rows_per_beat in (1, 2, 3, 4, 6, 8):
            with self.subTest(rows_per_beat=rows_per_beat):
                events, rows = parse("l4 c d e f", "lead", rows_per_beat)
                self.assertEqual([found["rows"] for found in events], [rows_per_beat] * 4)
                self.assertEqual(rows, 4 * rows_per_beat)

    def test_a_triplet_length_is_a_fraction_of_a_row_and_is_not_rounded(self):
        events, rows = parse("c12 d12 e12", "lead")
        self.assertEqual(events, [event(0, 1.3333333333333333, 60),
                                  event(1.3333333333333333, 1.3333333333333333, 62),
                                  event(2.6666666666666665, 1.3333333333333333, 64)])
        self.assertEqual(rows, 4)
        events, rows = parse("r12 c12 d12 e12", "lead")
        self.assertEqual(events, [event(1.3333333333333333, 1.3333333333333333, 60),
                                  event(2.6666666666666665, 1.3333333333333333, 62),
                                  event(4, 1.3333333333333333, 64)])
        self.assertEqual(rows, 5.333333333333333)

    def test_empty_text_gives_no_events_and_no_rows(self):
        self.assertEqual(parse("", "lead"), ([], 0))
        self.assertEqual(parse(" \n\t| |", "lead"), ([], 0))

    def test_events_carry_the_instrument_given(self):
        events, rows = parse("c r d", "bass")
        self.assertEqual(events, [event(0, 1, 60, "bass"), event(2, 1, 62, "bass")])
        self.assertEqual(rows, 3)

    def test_at_changes_the_instrument_of_what_follows(self):
        events, rows = parse("c @bass d e @pad_2 f", "lead")
        self.assertEqual(events, [event(0, 1, 60), event(1, 1, 62, "bass"), event(2, 1, 64, "bass"),
                                  event(3, 1, 65, "pad_2")])
        self.assertEqual(rows, 4)

    def test_v_sets_the_volume_of_what_follows_out_of_fifteen(self):
        events, rows = parse("c v0 d v15 e v6 f", "lead")
        self.assertEqual(events, [event(0, 1, 60), event(1, 1, 62, vel=0), event(2, 1, 64), event(3, 1, 65, vel=0.4)])
        self.assertEqual(rows, 4)

    def test_a_tie_makes_one_note_of_two_of_the_same_pitch(self):
        self.assertEqual(parse("c4 & c4", "lead"), ([event(0, 8, 60)], 8))
        self.assertEqual(parse("c2&c2&c1", "lead"), ([event(0, 32, 60)], 32))
        self.assertEqual(parse("c4. & c8.. & c32...", "lead"), ([event(0, 10.4375, 60)], 10.4375))

    def test_a_tie_joins_across_a_bar_line(self):
        self.assertEqual(parse("c2 & | c2 d", "lead"), ([event(0, 16, 60), event(16, 1, 62)], 17))

    def test_a_tie_joins_one_pitch_written_two_ways(self):
        self.assertEqual(parse("c+4 & d-4", "lead"), ([event(0, 8, 61)], 8))
        self.assertEqual(parse("c4 & > < c4", "lead"), ([event(0, 8, 60)], 8))

    def test_a_tie_to_another_pitch_joins_nothing(self):
        self.assertEqual(parse("c4 & d4 c4", "lead"), ([event(0, 4, 60), event(4, 4, 62), event(8, 4, 60)], 12))
        self.assertEqual(parse("c4 & > c4", "lead"), ([event(0, 4, 60), event(4, 4, 72)], 8))

    def test_a_rest_ends_a_tie(self):
        self.assertEqual(parse("c4 & r4 c4", "lead"), ([event(0, 4, 60), event(8, 4, 60)], 12))

    def test_a_tied_note_keeps_the_instrument_and_volume_it_began_with(self):
        events, rows = parse("c4 & @bass v5 c4 c", "lead")
        self.assertEqual(events, [event(0, 8, 60), event(8, 1, 60, "bass", 0.3333333333333333)])
        self.assertEqual(rows, 9)

    def test_a_tie_with_no_note_before_it_does_nothing(self):
        self.assertEqual(parse("& c", "lead"), ([event(0, 1, 60)], 1))

    def test_tied_triplets_add_up_without_rounding(self):
        self.assertEqual(parse("c12&c12&c12", "lead"), ([event(0, 4, 60)], 4))
        events, rows = parse("r12 c12&c12&c12 d", "lead")
        self.assertEqual(events, [event(1.3333333333333333, 4, 60), event(5.333333333333333, 1, 62)])
        self.assertEqual(rows, 6.333333333333333)

    def test_a_drum_letter_plays_what_the_drums_say(self):
        events, rows = parse("k s h H t", "lead", drums=DRUMS)
        self.assertEqual(events, [event(0, 1, 36, "thud"), event(1, 1, 50, "crack"), event(2, 1, 80, "tick", 0.5),
                                  event(3, 1, 78, "tick", 0.8), event(4, 0.5, 51, "tub")])
        self.assertEqual(rows, 5)

    def test_a_drum_sounds_no_longer_than_its_rows_and_still_takes_its_length(self):
        events, rows = parse("k2 s2 H2 t2", "lead", drums=DRUMS)
        self.assertEqual(events, [event(0, 8, 36, "thud"), event(8, 2, 50, "crack"), event(16, 3, 78, "tick", 0.8),
                                  event(24, 0.5, 51, "tub")])
        self.assertEqual(rows, 32)

    def test_a_drum_takes_a_length_dots_and_the_length_l_set(self):
        events, rows = parse("l8 k h. s4.. t", "lead", drums=DRUMS)
        self.assertEqual(events, [event(0, 2, 36, "thud"), event(2, 3, 80, "tick", 0.5), event(5, 2, 50, "crack"),
                                  event(12, 0.5, 51, "tub")])
        self.assertEqual(rows, 14)

    def test_the_volume_scales_the_level_a_drum_has_of_its_own(self):
        events, rows = parse("v6 k h v15 H", "lead", drums=DRUMS)
        self.assertEqual(events, [event(0, 1, 36, "thud", 0.4), event(1, 1, 80, "tick", 0.2),
                                  event(2, 1, 78, "tick", 0.8)])
        self.assertEqual(rows, 3)

    def test_notes_and_drums_can_share_a_channel(self):
        events, rows = parse("c k @bass d s", "lead", drums=DRUMS)
        self.assertEqual(events, [event(0, 1, 60), event(1, 1, 36, "thud"), event(2, 1, 62, "bass"),
                                  event(3, 1, 50, "crack")])
        self.assertEqual(rows, 4)

    def test_text_that_cannot_be_read_is_quoted_from_where_it_starts(self):
        self.assertEqual(self.refused("c d ? e"), 'cannot read "? e" in a channel')

    def test_the_quote_is_twelve_characters_at_most(self):
        self.assertEqual(self.refused("c d e xyz c d e f g a b"), 'cannot read "xyz c d e f " in a channel')

    def test_a_mark_without_what_it_needs_cannot_be_read(self):
        cases = {"C": "C", "7": "7", "c . d": ". d", "c++": "+", "c-- d": "- d", "o c": "o c", "c o": "o",
                 "o10 c": "0 c", "l c": "l c", "c l": "l", "l4. c": ". c"}
        for text, quoted in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self.refused(text), f'cannot read "{quoted}" in a channel')

    def test_a_length_of_zero_cannot_be_read(self):
        cases = {"c0 d": "c0 d", "c d e0 f": "e0 f", "r0.": "r0.", "c+00": "c+00", "l0 c4": "l0 c4"}
        for text, quoted in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self.refused(text), f'cannot read "{quoted}" in a channel')

    def test_an_instrument_needs_a_name_and_a_volume_a_number(self):
        cases = {"@ c": "@ c", "c @-x": "@-x", "c v d": "v d", "c v": "v"}
        for text, quoted in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self.refused(text), f'cannot read "{quoted}" in a channel')

    def test_a_drum_letter_is_refused_where_there_are_no_drums(self):
        for letter in "kshHt":
            with self.subTest(letter=letter):
                self.assertEqual(self.refused(f"c {letter}4"),
                                 f'"{letter}" is a drum, and this is not a drum channel')
        self.assertEqual(self.refused("k", {}), '"k" is a drum, and this is not a drum channel')

    def test_a_drum_letter_the_drums_leave_out_is_refused(self):
        self.assertEqual(self.refused("k s", {"k": DRUMS["k"]}), '"s" is a drum, and this is not a drum channel')

    def test_a_drum_of_length_zero_cannot_be_read(self):
        self.assertEqual(self.refused("k s0", DRUMS), 'cannot read "s0" in a channel')
        self.assertEqual(self.refused("t00.", DRUMS), 'cannot read "t00." in a channel')


class TransposeTest(unittest.TestCase):
    def test_every_note_is_written_with_its_octave(self):
        self.assertEqual(transpose("c d e", 2), "o4 d o4 e o4 g-")

    def test_the_notes_between_the_letters_are_written_as_flats(self):
        self.assertEqual(transpose("c c+ d d+ e f f+ g g+ a a+ b", 0),
                         "o4 c o4 d- o4 d o4 e- o4 e o4 f o4 g- o4 g o4 a- o4 a o4 b- o4 b")
        self.assertEqual(transpose("c# f+ a#", 0), "o4 d- o4 g- o4 b-")
        self.assertEqual(transpose("b+ c-", 0), "o5 c o3 b")

    def test_moving_up_crosses_into_the_octave_above(self):
        self.assertEqual(transpose("o4 a b", 2), "o4 b o5 d-")
        self.assertEqual(transpose("o4 a b", 5), "o5 d o5 e")
        self.assertEqual(transpose("c", 12), "o5 c")
        self.assertEqual(transpose("c", 13), "o5 d-")
        self.assertEqual(transpose("c", 24), "o6 c")

    def test_moving_down_crosses_into_the_octave_below(self):
        self.assertEqual(transpose("o4 c d", -1), "o3 b o4 d-")
        self.assertEqual(transpose("o4 c d", -2), "o3 b- o4 c")
        self.assertEqual(transpose("c", -12), "o3 c")
        self.assertEqual(transpose("c", -13), "o2 b")
        self.assertEqual(transpose("c", -25), "o1 b")

    def test_notes_can_be_moved_to_either_end_of_what_o_can_write(self):
        self.assertEqual(transpose("o0 c+", -1), "o0 c")
        self.assertEqual(transpose("o1 c", -12), "o0 c")
        self.assertEqual(transpose("o9 b-", 1), "o9 b")
        self.assertEqual(transpose("o8 b", 12), "o9 b")

    def test_lengths_and_dots_stay_on_their_notes(self):
        self.assertEqual(transpose("c4. d8.. e16 f", 1), "o4 d-4. o4 e-8.. o4 f16 o4 g-")
        self.assertEqual(transpose("c12 d12 e12 >c6.<b-24", 11), "o4 b12 o5 d-12 o5 e-12 o5 b6. o5 a24")

    def test_octave_marks_go_into_the_notes_after_them(self):
        self.assertEqual(transpose("o3 c > c < < c", 0), "o3 c o4 c o2 c")
        self.assertEqual(transpose("o3 c > c < < c", 2), "o3 d o4 d o2 d")

    def test_everything_that_is_not_a_note_or_an_octave_stays(self):
        self.assertEqual(transpose("@bass v12 l8 c & c r4. | r", 5), "@bass v12 l8 o4 f & o4 f r4. | r")
        self.assertEqual(transpose("l8 o3 @bass v12 e & e g+4. & | g+8 r b & > b < | v9 b2 & b2", -7),
                         "l8 @bass v12 o2 a & o2 a o3 d-4. & | o3 d-8 r o3 e & o4 e | v9 o3 e2 & o3 e2")

    def test_one_space_goes_between_things_and_none_at_the_ends(self):
        self.assertEqual(transpose("  c4\n\td4  |e4", 0), "o4 c4 o4 d4 | o4 e4")
        self.assertEqual(transpose("c4d4&d4", 0), "o4 c4 o4 d4 & o4 d4")
        self.assertEqual(transpose("c4d4&d4", -7), "o3 f4 o3 g4 & o3 g4")

    def test_text_without_notes_comes_back_tidied(self):
        self.assertEqual(transpose("", 5), "")
        self.assertEqual(transpose("   ", 5), "")
        self.assertEqual(transpose("| |", 5), "| |")

    def test_moved_text_reads_as_the_same_events_with_every_note_moved(self):
        texts = ("l8 o3 @bass v12 e & e g+4. & | g+8 r b & > b < | v9 b2 & b2",
                 "c12 d12 e12 >c6.<b-24", "o5 c >> c <<<< c", "c4. & c8.. & c32... r d-")
        for text in texts:
            events, rows = parse(text, "lead")
            for semis in (-13, -12, -1, 0, 1, 7, 12, 13):
                with self.subTest(text=text, semis=semis):
                    moved = [{**found, "note": found["note"] + semis} for found in events]
                    self.assertEqual(parse(transpose(text, semis), "lead"), (moved, rows))

    def test_text_that_cannot_be_read_is_quoted_from_where_it_starts(self):
        with self.assertRaises(RowSynthError) as caught:
            transpose("c d e xyz c d e f g a b", 3)
        self.assertEqual(str(caught.exception), 'cannot read "xyz c d e f " in a channel')

    def test_a_drum_letter_cannot_be_read(self):
        for letter in "kshHt":
            with self.subTest(letter=letter):
                with self.assertRaises(RowSynthError) as caught:
                    transpose(f"c {letter}4 c", 3)
                self.assertEqual(str(caught.exception), f'cannot read "{letter}4 c" in a channel')


class UpTest(unittest.TestCase):
    def test_every_channel_is_moved_but_the_drums(self):
        self.assertEqual(up(SONG, 2)["patterns"],
                         {"a": {"tune": "l8 o4 d o4 g- o4 a o5 d", "low": "o2 d2", "beat": "k4 k4"},
                          "b": {"tune": "o5 d4 o5 d-4", "beat": "l4 k k"}})

    def test_each_channel_holds_what_transpose_makes_of_its_text(self):
        for semis in (-12, -3, 0, 7):
            with self.subTest(semis=semis):
                moved = up(SONG, semis)["patterns"]
                self.assertEqual(moved["a"]["tune"], transpose("l8 c e g > c", semis))
                self.assertEqual(moved["a"]["low"], transpose("o2 c2", semis))
                self.assertEqual(moved["b"]["tune"], transpose("o5 c4 < b4", semis))
                self.assertEqual(moved["a"]["beat"], "k4 k4")
                self.assertEqual(moved["b"]["beat"], "l4 k k")

    def test_the_rest_of_the_song_comes_through_as_it_was(self):
        moved = up(SONG, 2)
        self.assertEqual(set(moved), set(SONG))
        for key in SONG:
            if key != "patterns":
                self.assertEqual(moved[key], SONG[key])

    def test_the_song_passed_in_is_not_changed(self):
        song = copy.deepcopy(SONG)
        moved = up(song, 5)
        self.assertEqual(song, SONG)
        self.assertIsNot(moved, song)
        self.assertIsNot(moved["patterns"], song["patterns"])
        self.assertIsNot(moved["patterns"]["a"], song["patterns"]["a"])

    def test_a_channel_that_says_it_has_no_drums_is_moved(self):
        song = {"channels": {"one": {"inst": "lead", "drums": False}}, "patterns": {"x": {"one": "c e g"}, "y": {}}}
        self.assertEqual(up(song, 12)["patterns"], {"x": {"one": "o5 c o5 e o5 g"}, "y": {}})

    def test_text_that_cannot_be_read_is_refused_by_its_pattern_and_channel(self):
        song = {"channels": {"one": {"inst": "lead"}, "two": {"inst": "lead"}},
                "patterns": {"x": {"one": "c e g"}, "y": {"one": "c", "two": "c e ? g"}}}
        with self.assertRaises(RowSynthError) as caught:
            up(song, 2)
        self.assertEqual(str(caught.exception), "pattern 'y', channel 'two': cannot read \"? g\" in a channel")

    def test_a_pattern_naming_a_channel_the_song_lacks_is_refused(self):
        song = {"channels": {"one": {"inst": "lead"}}, "patterns": {"x": {"one": "c", "two": "e"}}}
        with self.assertRaises(RowSynthError) as caught:
            up(song, 2)
        self.assertEqual(str(caught.exception), "pattern 'x': no channel called 'two'")


if __name__ == "__main__":
    unittest.main()
