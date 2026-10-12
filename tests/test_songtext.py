"""Tests for rowsynth.songtext: a channel's events written back as text that reads as the same events."""
from __future__ import annotations

import random
import unittest
from pathlib import Path
from unittest import mock

from rowsynth.errors import RowSynthError
from rowsynth.notes import midi
from rowsynth.project import Project
from rowsynth.song import parse
from rowsynth.songtext import write

EXAMPLES = Path(__file__).resolve().parent / "data"

# A letter that holds its hits for as long as they are written, one that holds them two rows and one five, one
# at half the level, and one that is silent and half a row long.
DRUMS = {"k": {"inst": "thud", "note": "c2"},
         "s": {"inst": "crack", "note": "d3", "rows": 2},
         "H": {"inst": "tick", "note": "f#5", "rows": 5},
         "h": {"inst": "tick", "note": 80, "vel": 0.5},
         "t": {"inst": "tub", "note": "eb3", "rows": 0.5, "vel": 0}}

TRIED = "whole notes and then up to 4 of the lengths 1 2 3 4 6 8 12 16 24 32 48 64, plain or dotted"


def note(row: float, rows: float, pitch: float = 60, inst: str = "lead", vel: float = 1.0) -> dict:
    """One event of a tuned note, as parse gives it."""
    return {"row": row, "rows": rows, "note": pitch, "inst": inst, "vel": vel}


def hit(row: float, rows: float, letter: str, level: int = 15, **changed: object) -> dict:
    """One event of a drum hit written at a level, as parse gives it, with whatever of it is changed."""
    own = DRUMS[letter]
    return {"row": row, "rows": rows, "note": midi(own["note"]), "inst": own["inst"],
            "vel": level / 15 * own.get("vel", 1), "drum": letter, **changed}


def again(test: unittest.TestCase, text: str, rows_per_beat: int = 4, drums: dict | None = None) -> str:
    """A text read and written again, once it is seen to read as what the first text read as."""
    read = parse(text, "lead", rows_per_beat, drums)
    written = write(*read, "lead", rows_per_beat, drums)
    test.assertEqual(parse(written, "lead", rows_per_beat, drums), read)
    return written


def refusal(test: unittest.TestCase, events: list, rows: float = 8, drums: dict | None = None) -> str:
    """The message the writing of some events is refused with."""
    with test.assertRaises(RowSynthError) as caught:
        write(events, rows, "lead", drums=drums)
    return str(caught.exception)


class Lengths(unittest.TestCase):
    def test_the_plain_length_that_comes_most_often_is_written_once_at_the_start_where_that_is_shorter(self) -> None:
        self.assertEqual(again(self, "c8 d8 e8 f8 g4"), "o4 l8 c d e f g4")
        # Of two that come as often, it is the longer.
        self.assertEqual(again(self, "c4 d4 e4 f4 g8 a8 b8 a8"), "o4 l4 c d e f | g8 a8 b8 a8")
        self.assertEqual(again(self, "c4 d8 e8 f16 g16 a8"), "o4 c4 d8 e8 f16 g16 a8")
        self.assertEqual(again(self, "c1"), "o4 c1")

    def test_a_dotted_note_carries_its_length_whatever_the_start_says(self) -> None:
        self.assertEqual(again(self, "c8 d8 e8 f8 g8. a8."), "o4 l8 c d e f g8. a8.")

    def test_a_note_no_one_length_makes_is_tied_from_the_fewest_the_longest_first(self) -> None:
        self.assertEqual(write([note(0, 5)], 5, "lead"), "o4 c4 & c16")
        self.assertEqual(write([note(0, 7)], 7, "lead"), "o4 c4. & c16")
        self.assertEqual(write([note(0, 10)], 10, "lead"), "o4 c2 & c8")
        # Past a bar the tie goes over the bar line, and past two it begins with whole notes.
        self.assertEqual(write([note(0, 20)], 20, "lead"), "o4 c1 & | c4")
        self.assertEqual(write([note(0, 40)], 40, "lead"), "o4 c1 & | c1 & | c2")

    def test_a_gap_is_the_fewest_rests_the_longest_first(self) -> None:
        self.assertEqual(write([note(0, 1), note(6, 1, 62), note(13, 1, 64)], 14, "lead"), "o4 l16 c r4 r d r4. e")
        self.assertEqual(write([note(60, 4)], 64, "lead"), "o4 r1 | r1 | r1 | r2. c4")

    def test_a_rest_fills_the_rows_after_the_last_note(self) -> None:
        self.assertEqual(write([note(0, 4)], 16, "lead"), "o4 c4 r2.")
        self.assertEqual(parse("o4 c4 r2.", "lead"), ([note(0, 4)], 16))

    def test_a_channel_with_no_events_is_rests_only(self) -> None:
        self.assertEqual(write([], 32, "lead"), "r1 | r1")
        self.assertEqual(write([], 6, "lead"), "r4.")
        self.assertEqual(write([], 0, "lead"), "")

    def test_a_bar_line_stands_where_something_begins_a_bar(self) -> None:
        self.assertEqual(again(self, "c2 d2 e2 f2 r1"), "o4 l2 c d | e f | r1")
        # The second note sounds over the first bar line, so nothing begins on it.
        self.assertEqual(again(self, "c2. d2 e2. r1"), "o4 c2. d2 e2. | r1")

    def test_with_three_rows_to_a_beat_a_row_is_a_triplet(self) -> None:
        events = [note(0, 1), note(1, 2, 62), note(3, 3, 64), note(6, 5, 65), note(11, 1, 67)]
        self.assertEqual(write(events, 12, "lead", 3), "o4 l12 c d6 e4 f3 & f g")
        self.assertEqual(parse("o4 l12 c d6 e4 f3 & f g", "lead", 3), (events, 12))

    def test_with_eight_rows_to_a_beat_a_row_is_a_thirty_second(self) -> None:
        events = [note(0, 1), note(1, 3, 62), note(4, 5, 64), note(9, 7, 65), note(16, 8, 67)]
        self.assertEqual(write(events, 32, "lead", 8), "o4 l32 c d16. e8 & e f8. & f g4 r4")
        self.assertEqual(parse("o4 l32 c d16. e8 & e f8. & f g4 r4", "lead", 8), (events, 32))

    def test_triplets_are_written_and_a_tie_of_them_in_the_lengths_that_read_back(self) -> None:
        self.assertEqual(again(self, "c12 d12 e12 f4 r12 r12 r12"), "o4 l12 c d e f4 r r r")
        # A half tied to a triplet half is as long, and its sum ends a hair off where this text ends.
        self.assertEqual(again(self, "c1 d3 & d3"), "o4 c1 | d3 & d3")
        # A whole note tied to two triplet eighths is as long as well, and the second of them would not tie on.
        self.assertEqual(again(self, "r1 c2 & c3 & c3"), "o4 r1 | c2 & c3 & c3")


class Marks(unittest.TestCase):
    def test_the_octave_moves_by_a_mark_for_one_step_and_by_o_for_a_jump(self) -> None:
        self.assertEqual(again(self, "c > c > c o4 c o7 c o2 c"), "o4 l16 c > c > c o4 c o7 c o2 c")
        self.assertEqual(again(self, "o2 c c > c c < c c"), "o2 l16 c c > c c < c c")

    def test_an_octave_out_of_the_reach_of_o_is_come_to_with_marks(self) -> None:
        events = [note(0, 1, 0), note(1, 1, -13), note(2, 1, 132), note(3, 1, 156), note(4, 1, 131)]
        self.assertEqual(write(events, 5, "lead"), "o0 l16 < c < < b o9 > c > > c o9 b")
        self.assertEqual(parse("o0 l16 < c < < b o9 > c > > c o9 b", "lead"), (events, 5))

    def test_v_stands_only_where_the_level_changes(self) -> None:
        self.assertEqual(again(self, "c v9 d e v15 f v0 g"), "o4 l16 c v9 d e v15 f v0 g")
        self.assertEqual(again(self, "v15 c d e f"), "o4 l16 c d e f")

    def test_an_instrument_is_named_only_where_it_changes(self) -> None:
        self.assertEqual(again(self, "c @bass d e @lead f"), "o4 l16 c @bass d e @lead f")
        self.assertEqual(again(self, "@bass c2"), "o4 @bass c2")

    def test_a_note_between_two_letters_is_spelt_as_a_flat(self) -> None:
        self.assertEqual(again(self, "c+ d+ f# g+ a+ b+ c-"), "o4 l16 d- e- g- a- b- > c o3 b")

    def test_events_out_of_order_are_put_in_order(self) -> None:
        self.assertEqual(write([note(8, 4, 64), note(0, 4), note(4, 4, 62)], 12, "lead"), "o4 c4 d4 e4")


class Drums(unittest.TestCase):
    def test_a_hit_held_for_its_letters_rows_takes_the_gap_after_it_when_one_length_says_it(self) -> None:
        events = [hit(0, 2, "s"), hit(4, 2, "s"), hit(8, 2, "s"), hit(14, 2, "s")]
        self.assertEqual(write(events, 16, "lead", drums=DRUMS), "s4 s4 s4. s8")
        self.assertEqual(parse("s4 s4 s4. s8", "lead", drums=DRUMS), (events, 16))

    def test_a_hit_takes_its_letters_rows_and_rests_where_no_one_length_says_the_gap(self) -> None:
        events = [hit(0, 2, "s"), hit(5, 2, "s"), hit(12, 2, "s")]
        self.assertEqual(write(events, 19, "lead", drums=DRUMS), "s8 r8. s8 r4 r16 s8 r4 r16")
        # Five rows are no one length either, so the hit is written with the longest that fits its room.
        events = [hit(0, 5, "H"), hit(7, 5, "H")]
        self.assertEqual(write(events, 16, "lead", drums=DRUMS), "H4. r16 H2 r16")
        self.assertEqual(parse("H4. r16 H2 r16", "lead", drums=DRUMS), (events, 16))

    def test_a_hit_shorter_than_its_letters_rows_or_with_none_is_written_at_its_own_length(self) -> None:
        self.assertEqual(write([hit(0, 1, "s"), hit(1, 1, "s"), hit(2, 2, "s")], 4, "lead", drums=DRUMS),
                         "s16 s16 s8")
        self.assertEqual(write([hit(0, 4, "k"), hit(4, 2, "k"), hit(8, 6, "k")], 16, "lead", drums=DRUMS),
                         "k4 k8 r8 k4. r8")

    def test_the_level_of_a_hit_has_its_letters_own_taken_off(self) -> None:
        events = [hit(0, 1, "h"), hit(1, 1, "h", 9), hit(2, 1, "k", 9), hit(3, 0.5, "t", 9), hit(4, 1, "h")]
        self.assertEqual(events[1]["vel"], 0.3)
        # The silent letter says nothing of a level, so none is written for it.
        self.assertEqual(write(events, 8, "lead", drums=DRUMS), "l16 h v9 h k t v15 h r8.")
        self.assertEqual(parse("l16 h v9 h k t v15 h r8.", "lead", drums=DRUMS), (events, 8))

    def test_notes_and_hits_share_a_channel(self) -> None:
        self.assertEqual(again(self, "c k @bass d s", drums=DRUMS), "o4 l16 c k @bass d s")

    def test_a_hit_its_letter_cannot_write_is_refused(self) -> None:
        cases = [([hit(0, 1, "k")], None, "row 0: no drum letter 'k' to write the hit with"),
                 ([hit(0, 1, "k", drum="x")], DRUMS, "row 0: no drum letter 'x' to write the hit with"),
                 ([hit(0, 1, "k", note=40.0)], DRUMS, "row 0: the hit is not what the drum letter 'k' plays"),
                 ([hit(2, 1, "k", inst="tick")], DRUMS, "row 2: the hit is not what the drum letter 'k' plays"),
                 ([hit(3, 3, "s")], DRUMS, "row 3: the hit is not what the drum letter 's' plays"),
                 ([hit(1, 5, "k")], DRUMS,
                  "row 1: a hit of 5 rows cannot be written: a hit is one length, and none fits it"),
                 ([hit(3, 5, "H")], DRUMS,
                  "row 3: a hit of 5 rows cannot be written: a hit is one length, and none fits it"),
                 ([hit(1, 1, "h", vel=0.25)], DRUMS, "row 1: a level of 0.25 is no whole v, which counts in "
                                                     "fifteenths of the 0.5 its drum letter has"),
                 ([hit(1, 0.5, "t", vel=0.2)], DRUMS, "row 1: a level of 0.2 is no whole v, which counts in "
                                                      "fifteenths of the 0 its drum letter has")]
        for events, drums, said in cases:
            with self.subTest(said=said):
                self.assertEqual(refusal(self, events, drums=drums), said)


class Refusals(unittest.TestCase):
    def test_events_that_overlap_or_leave_the_rows_are_refused(self) -> None:
        cases = [([note(0, 4), note(2, 4, 62)], "row 0: the event is 4 rows long, and the next begins at row 2: a "
                                                "channel plays one note at a time"),
                 ([note(4, 4), note(4, 4, 62)], "row 4: the event is 4 rows long, and the next begins at row 4: a "
                                                "channel plays one note at a time"),
                 ([note(6, 4)], "row 6: the event is 4 rows long, and the text is to fill 8 rows"),
                 ([note(-1, 2)], "row -1: an event cannot begin before row 0"),
                 ([note(2, 0)], "row 2: an event of 0 rows cannot be written")]
        for events, said in cases:
            with self.subTest(said=said):
                self.assertEqual(refusal(self, events), said)

    def test_rows_that_no_lengths_come_to_are_refused_with_the_lengths_tried(self) -> None:
        said = f"0.3 rows cannot be written: {TRIED}, do not come to exactly that"
        self.assertEqual(refusal(self, [note(2, 0.3)]), f"row 2: {said}")
        self.assertEqual(refusal(self, [note(0.3, 1)]), f"row 0: {said}")
        # The note is written, and it is the rows after it that nothing fills.
        self.assertEqual(refusal(self, [note(0, 1)], 1.3), f"row 1: {said}")

    def test_a_level_a_note_or_an_instrument_the_text_cannot_write_is_refused(self) -> None:
        cases = [(note(3, 1, vel=0.5), "row 3: a level of 0.5 is no whole v, which counts in fifteenths"),
                 (note(3, 1, vel=-0.2), "row 3: a level of -0.2 is no whole v, which counts in fifteenths"),
                 (note(5, 1, 60.5), "row 5: a note of 60.5 is between two notes, and the text has none there"),
                 (note(5, 1, inst="my lead"), "row 5: an instrument called 'my lead' cannot be named in the text"),
                 (note(6, 1, inst=""), "row 6: an instrument called '' cannot be named in the text")]
        for event, said in cases:
            with self.subTest(said=said):
                self.assertEqual(refusal(self, [event]), said)
        with self.assertRaises(RowSynthError) as caught:
            write([], 4, "lead", 0)
        self.assertEqual(str(caught.exception), "rows_per_beat: must be more than 0, not 0")

    def test_text_that_would_not_read_back_as_the_events_is_refused(self) -> None:
        said = "the text written from here on does not read back as the events it was written for"
        events = [note(0, 4), note(4, 4, 62)]
        with mock.patch("rowsynth.songtext.parse", return_value=([note(0, 4), note(4, 2, 62), note(6, 2, 62)], 8)):
            self.assertEqual(refusal(self, events), f"row 4: {said}")
        with mock.patch("rowsynth.songtext.parse", return_value=(events, 7.5)):
            self.assertEqual(refusal(self, events), f"row 8: {said}")


class Proof(unittest.TestCase):
    def test_every_channel_of_the_examples_reads_back_the_same_and_is_written_the_same_again(self) -> None:
        for file in ("arcade.json", "piece.json"):
            for name, song in Project.load(EXAMPLES / file).songs.items():
                per_beat = song.get("rows_per_beat", 4)
                for pattern, texts in song["patterns"].items():
                    for channel, text in texts.items():
                        with self.subTest(file=file, song=name, pattern=pattern, channel=channel):
                            inst = song["channels"][channel]["inst"]
                            drums = song.get("drums") if song["channels"][channel].get("drums") else None
                            read = parse(text, inst, per_beat, drums)
                            written = write(*read, inst, per_beat, drums)
                            self.assertEqual(parse(written, inst, per_beat, drums), read)
                            self.assertEqual(write(*parse(written, inst, per_beat, drums), inst, per_beat, drums),
                                             written)

    def test_two_hundred_random_channels_on_the_grid_read_back_the_same_and_are_written_the_same_again(self) -> None:
        for seed in range(200):
            dice = random.Random(seed)
            events = []
            row = 0.0
            for _ in range(dice.randint(1, 24)):
                # Most notes follow on from the one before, and a few stand after a gap.
                row += dice.choice((0, 0, 0, 1, 2, 3, 5, 9, 14))
                rows = float(dice.randint(1, 20))
                events.append(note(row, rows, dice.randint(36, 95), dice.choice(("lead", "lead", "bass")),
                                   dice.choice((15, 15, 15, 12, 9, 4, 0)) / 15))
                row += rows
            rows = row + dice.choice((0, 0, 1, 6, 23))
            with self.subTest(seed=seed):
                written = write(events, rows, "lead")
                self.assertEqual(parse(written, "lead"), (events, rows))
                self.assertEqual(write(*parse(written, "lead"), "lead"), written)


if __name__ == "__main__":
    unittest.main()
