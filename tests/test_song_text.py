"""Tests for the text a song's channels are written in."""
from __future__ import annotations

import unittest

from rowsynth.errors import RowSynthError
from rowsynth.song import parse


def event(row: float, rows: float, note: int, inst: str = "lead") -> dict:
    """One event as parse gives it."""
    return {"row": row, "rows": rows, "note": note, "inst": inst, "vel": 1}


class ParseTest(unittest.TestCase):
    def refused(self, text: str) -> str:
        """The message parse refuses `text` with."""
        with self.assertRaises(RowSynthError) as caught:
            parse(text, "lead")
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


if __name__ == "__main__":
    unittest.main()
