"""Tests for rowsynth.gui.noteedit: what a hand does to the notes of a channel, and that the result always saves."""
from __future__ import annotations

import random
import unittest

from rowsynth.errors import RowSynthError
from rowsynth.gui.noteedit import (boxed, copied, fitted, hit, move, pasted, place, remove, shifted, snapped, stretch,
                                   unhit, with_inst, with_level)
from rowsynth.gui.songview import Note, pattern, text
from rowsynth.notes import midi

# Two letters that hold a hit for as long as it is written, one of them at half the level, and one that holds it
# for two rows.
DRUMS = {"k": {"inst": "thud", "note": "c2"},
         "h": {"inst": "tick", "note": 90, "vel": 0.5},
         "s": {"inst": "thud", "note": "d4", "rows": 2}}

SONG = {"row_ticks": 3, "instruments": {"lead": {}, "bass": {}, "thud": {}, "tick": {}}, "drums": DRUMS,
        "channels": {"tune": {"inst": "lead"}, "beat": {"inst": "thud", "drums": True}},
        "patterns": {"a": {}}, "order": ["a"]}

# The five octaves the random channels are in, the levels of their notes and the instruments they play with.
LOW, HIGH = 36, 95
LEVELS = (15, 15, 12, 9, 4, 0)
INSTS = ("lead", "lead", "bass")


def note(row: float, rows: float, pitch: int = 60, level: int = 15, inst: str = "lead") -> Note:
    """One note of a tuned channel."""
    return Note(float(row), float(rows), pitch, level, inst)


def drum(row: float, rows: float, letter: str, level: int = 15) -> Note:
    """One hit of a letter, playing what the letter plays."""
    return Note(float(row), float(rows), midi(DRUMS[letter]["note"]), level, DRUMS[letter]["inst"], letter)


def saved(test: unittest.TestCase, notes: tuple[Note, ...], rows: int = 16, channel: str = "tune") -> tuple[Note, ...]:
    """Notes, once they are seen to keep the rule, to be written as text, and to be read from that text the same."""
    test.assertTrue(fitted(notes, rows))
    song = {**SONG, "patterns": {"a": {channel: text(notes, rows, channel, SONG)}}}
    test.assertEqual(next(held.notes for held in pattern(song, "a").channels if held.name == channel), notes)
    return notes


def refusal(test: unittest.TestCase, call: object, *given: object) -> str:
    """The message a call is refused with."""
    with test.assertRaises(RowSynthError) as caught:
        call(*given)
    return str(caught.exception)


class Place(unittest.TestCase):
    def test_a_note_is_placed_among_the_others_in_the_order_of_their_rows(self) -> None:
        notes = (note(0, 2), note(8, 4))
        self.assertEqual(saved(self, place(notes, note(4, 2, 67, 9, "bass"), 16)),
                         (note(0, 2), note(4, 2, 67, 9, "bass"), note(8, 4)))
        self.assertEqual(saved(self, place((), note(12, 4), 16)), (note(12, 4),))

    def test_a_note_still_sounding_where_the_new_one_begins_is_cut_short_there(self) -> None:
        self.assertEqual(saved(self, place((note(0, 8),), note(4, 2, 62), 16)), (note(0, 4), note(4, 2, 62)))
        # One that ends where the new one begins is left as it is.
        self.assertEqual(saved(self, place((note(0, 4),), note(4, 2, 62), 16)), (note(0, 4), note(4, 2, 62)))

    def test_a_note_that_begins_inside_the_new_one_is_removed(self) -> None:
        notes = (note(0, 2), note(4, 1), note(6, 6), note(12, 2))
        self.assertEqual(saved(self, place(notes, note(4, 4, 62), 16)), (note(0, 2), note(4, 4, 62), note(12, 2)))
        # One that begins where the new one ends stays.
        self.assertEqual(saved(self, place(notes, note(2, 2, 62), 16)), (note(0, 2), note(2, 2, 62), *notes[1:]))

    def test_a_placed_note_is_cut_off_at_the_edges_of_the_pattern_and_one_outside_it_is_refused(self) -> None:
        self.assertEqual(saved(self, place((), note(14, 4), 16)), (note(14, 2),))
        self.assertEqual(saved(self, place((note(0, 1),), note(-2, 5), 16)), (note(0, 3),))
        self.assertEqual(refusal(self, place, (), note(16, 2), 16),
                         "row 16: a note of 2 rows there has no part in the 16 rows of the pattern")
        self.assertEqual(refusal(self, place, (), note(4, 0), 16),
                         "row 4: a note of 0 rows there has no part in the 16 rows of the pattern")


class Move(unittest.TestCase):
    def test_a_picked_note_is_moved_by_rows_and_by_semitones(self) -> None:
        found, picked = move((note(0, 2), note(4, 2), note(12, 2)), {1}, 4, -5, 16, LOW, HIGH)
        self.assertEqual(saved(self, found), (note(0, 2), note(8, 2, 55), note(12, 2)))
        self.assertEqual(picked, {1})

    def test_a_note_still_sounding_where_a_moved_one_begins_is_cut_short_there(self) -> None:
        found, picked = move((note(0, 8), note(12, 2)), {1}, -6, 0, 16, LOW, HIGH)
        self.assertEqual(saved(self, found), (note(0, 6), note(6, 2)))
        self.assertEqual(picked, {1})

    def test_notes_that_begin_inside_a_moved_one_are_removed_and_it_is_picked_where_it_now_is(self) -> None:
        found, picked = move((note(0, 2), note(2, 2), note(4, 2), note(8, 2, 64)), {3}, -7, 0, 16, LOW, HIGH)
        self.assertEqual(saved(self, found), (note(0, 1), note(1, 2, 64), note(4, 2)))
        self.assertEqual(picked, {1})

    def test_a_moved_group_stops_together_at_the_edges_and_keeps_its_shape(self) -> None:
        notes = (note(2, 2), note(6, 4, 64), note(15, 1))
        found, picked = move(notes, {0, 1}, -10, -40, 16, 48, 72)
        self.assertEqual(saved(self, found), (note(0, 2, 48), note(4, 4, 52), note(15, 1)))
        self.assertEqual(picked, {0, 1})
        found, picked = move(notes, {0, 1}, 20, 40, 16, 48, 72)
        self.assertEqual(saved(self, found), (note(8, 2, 68), note(12, 4, 72)))
        self.assertEqual(picked, {0, 1})

    def test_moved_notes_do_not_cut_each_other_and_with_none_picked_nothing_moves(self) -> None:
        notes = (note(0, 2), note(2, 2, 62), note(4, 4))
        found, picked = move(notes, {0, 2}, 1, 0, 16, LOW, HIGH)
        self.assertEqual(saved(self, found), (note(1, 2), note(5, 4)))
        self.assertEqual(picked, {0, 1})
        self.assertEqual(move(notes, set(), 3, 3, 16, LOW, HIGH), (notes, frozenset()))


class Stretch(unittest.TestCase):
    def test_picked_notes_are_made_longer_or_shorter_at_their_ends(self) -> None:
        notes = (note(0, 4), note(6, 2), note(10, 4))
        found, picked = stretch(notes, {0, 2}, 2, 16, 1)
        self.assertEqual(saved(self, found), (note(0, 6), note(6, 2), note(10, 6)))
        self.assertEqual(picked, {0, 2})
        self.assertEqual(stretch(notes, {0, 2}, -3, 16, 1), ((note(0, 1), note(6, 2), note(10, 1)), {0, 2}))

    def test_a_stretched_note_stops_where_the_next_picked_one_begins(self) -> None:
        found, picked = stretch((note(0, 2), note(4, 2, 62)), {0, 1}, 6, 16, 1)
        self.assertEqual(saved(self, found), (note(0, 4), note(4, 8, 62)))
        self.assertEqual(picked, {0, 1})

    def test_notes_that_begin_inside_a_stretched_one_are_removed_and_the_picked_are_where_they_now_are(self) -> None:
        found, picked = stretch((note(0, 2), note(3, 1), note(4, 1), note(6, 2, 62), note(10, 2)), {0, 3}, 3, 16, 1)
        self.assertEqual(saved(self, found), (note(0, 5), note(6, 5, 62)))
        self.assertEqual(picked, {0, 1})

    def test_a_note_is_stretched_to_the_least_and_not_under_and_to_the_end_of_the_pattern_and_not_past(self) -> None:
        notes = (note(0, 4), note(12, 2))
        self.assertEqual(stretch(notes, {0, 1}, -10, 16, 2), ((note(0, 2), note(12, 2)), {0, 1}))
        # A note that is under the least already is not made longer by it.
        self.assertEqual(stretch((note(0, 1),), {0}, -1, 16, 2), ((note(0, 1),), {0}))
        found, _ = stretch(notes, {1}, 10, 16, 1)
        self.assertEqual(saved(self, found), (note(0, 4), note(12, 4)))
        self.assertEqual(refusal(self, stretch, notes, {0}, 1, 16, 0),
                         "the least a note is stretched to is more than 0 rows, not 0")


class Picked(unittest.TestCase):
    NOTES = (note(0, 2, 48), note(2, 2, 60, 9), note(4, 4, 72, 4, "bass"))

    def test_the_picked_notes_are_removed_and_nothing_is_picked(self) -> None:
        found, picked = remove(self.NOTES, {0, 2})
        self.assertEqual(saved(self, found), (note(2, 2, 60, 9),))
        self.assertEqual(picked, frozenset())

    def test_the_picked_notes_are_set_to_a_level_from_0_to_15(self) -> None:
        found, picked = with_level(self.NOTES, {1, 2}, 0)
        self.assertEqual(saved(self, found), (note(0, 2, 48), note(2, 2, 60, 0), note(4, 4, 72, 0, "bass")))
        self.assertEqual(picked, {1, 2})
        self.assertEqual(with_level(self.NOTES, {0}, 15)[0], self.NOTES)
        for level in (16, -1, 7.5):
            with self.subTest(level=level):
                self.assertEqual(refusal(self, with_level, self.NOTES, {0}, level),
                                 f"a level is a whole number from 0 to 15, not {level!r}")

    def test_the_picked_notes_are_set_to_an_instrument(self) -> None:
        found, picked = with_inst(self.NOTES, {0, 2}, "bass")
        self.assertEqual(saved(self, found), (note(0, 2, 48, 15, "bass"), self.NOTES[1], self.NOTES[2]))
        self.assertEqual(picked, {0, 2})

    def test_the_picked_notes_are_shifted_by_an_octave(self) -> None:
        found, picked = shifted(self.NOTES, {0, 1}, 12, 48, 72)
        self.assertEqual(saved(self, found), (note(0, 2, 60), note(2, 2, 72, 9), self.NOTES[2]))
        self.assertEqual(picked, {0, 1})
        self.assertEqual(shifted(self.NOTES, {1, 2}, -12, 48, 72)[0],
                         (self.NOTES[0], note(2, 2, 48, 9), note(4, 4, 60, 4, "bass")))

    def test_nothing_is_shifted_where_one_note_would_leave_the_range(self) -> None:
        self.assertEqual(shifted(self.NOTES, {1, 2}, 12, 48, 72), (self.NOTES, {1, 2}))
        self.assertEqual(shifted(self.NOTES, {0, 1}, -12, 48, 72), (self.NOTES, {0, 1}))

    def test_an_index_that_is_no_notes_is_refused(self) -> None:
        self.assertEqual(refusal(self, remove, self.NOTES, {1, 3}), "3 is picked, and that is none of the 3 notes")
        self.assertEqual(refusal(self, copied, self.NOTES, {-1}), "-1 is picked, and that is none of the 3 notes")


class Boxes(unittest.TestCase):
    NOTES = (note(0, 4, 60), note(4, 4, 64), note(8, 4, 67))

    def test_a_box_touches_the_notes_that_sound_in_its_rows_at_its_pitches(self) -> None:
        self.assertEqual(boxed(self.NOTES, 2, 9, 60, 67), {0, 1, 2})
        self.assertEqual(boxed(self.NOTES, 2, 9, 61, 66), {1})
        self.assertEqual(boxed(self.NOTES, 12, 16, 0, 127), frozenset())
        # A box drawn from its other corner is the same box.
        self.assertEqual(boxed(self.NOTES, 9, 2, 66, 61), {1})

    def test_a_note_at_the_border_of_a_box_is_touched_by_pitch_and_not_by_the_row_it_ends_or_begins_at(self) -> None:
        self.assertEqual(boxed(self.NOTES, 4, 8, 64, 64), {1})
        self.assertEqual(boxed(self.NOTES, 4, 8, 0, 127), {1})
        self.assertEqual(boxed(self.NOTES, 3.5, 8.5, 0, 127), {0, 1, 2})
        self.assertEqual(boxed(self.NOTES, 4, 8, 65, 127), frozenset())
        self.assertEqual(boxed(self.NOTES, 4, 8, 0, 63), frozenset())


class Clips(unittest.TestCase):
    NOTES = (note(0, 2), note(4, 2, 62, 9), note(6, 4, 64, 4, "bass"), note(12, 2))

    def test_a_clip_is_the_picked_notes_from_row_0(self) -> None:
        self.assertEqual(copied(self.NOTES, {1, 2}), (note(0, 2, 62, 9), note(2, 4, 64, 4, "bass")))
        self.assertEqual(copied(self.NOTES, {1, 3}), (note(0, 2, 62, 9), note(8, 2)))
        self.assertEqual(copied(self.NOTES, set()), ())

    def test_a_clip_is_pasted_at_a_row_by_the_rule_with_its_notes_picked(self) -> None:
        found, picked = pasted((note(0, 4), note(4, 2), note(6, 2), note(9, 2)), copied(self.NOTES, {1, 2}), 2, 16)
        self.assertEqual(saved(self, found), (note(0, 2), note(2, 2, 62, 9), note(4, 4, 64, 4, "bass"), note(9, 2)))
        self.assertEqual(picked, {1, 2})

    def test_a_clip_pasted_past_the_end_is_cut_off_there(self) -> None:
        clip = copied(self.NOTES, {1, 2, 3})
        found, picked = pasted(self.NOTES, clip, 10, 16)
        self.assertEqual(saved(self, found), (*self.NOTES[:3], note(10, 2, 62, 9), note(12, 4, 64, 4, "bass")))
        self.assertEqual(picked, {3, 4})
        self.assertEqual(pasted(self.NOTES, clip, 16, 16), (self.NOTES, frozenset()))


class Grid(unittest.TestCase):
    def test_a_row_is_snapped_to_the_start_of_the_step_it_is_in(self) -> None:
        self.assertEqual([snapped(row, 4) for row in (0, 3.9, 4, 6.5, -0.5)], [0, 0, 4, 4, -4])
        self.assertEqual([snapped(row, 1) for row in (2.999, 3.0)], [2, 3])
        self.assertEqual(snapped(1.3, 0.5), 1)
        self.assertEqual(refusal(self, snapped, 3, 0), "a grid has steps of more than 0 rows, not 0")

    def test_notes_are_fitted_that_keep_the_rule(self) -> None:
        self.assertTrue(fitted((), 0))
        self.assertTrue(fitted((note(0, 4), note(4, 4), note(12, 4)), 16))
        cases = {"overlapping": (note(0, 5), note(4, 4)), "out of order": (note(4, 4), note(0, 4)),
                 "two at a row": (note(4, 2), note(4, 2, 62)), "before row 0": (note(-1, 2),),
                 "past the rows": (note(14, 4),), "of no rows": (note(4, 0),)}
        for said, notes in cases.items():
            with self.subTest(said=said):
                self.assertFalse(fitted(notes, 16))


class Hits(unittest.TestCase):
    def beat(self, notes: tuple[Note, ...], rows: int = 16) -> tuple[Note, ...]:
        """The notes of a drum channel, once they are seen to save."""
        return saved(self, notes, rows, "beat")

    def test_a_hit_plays_what_its_letter_plays_and_lasts_as_long_as_one_length_says_in_its_room(self) -> None:
        self.assertEqual(self.beat(hit((), 0, "k", 15, DRUMS, 16)), (Note(0, 16, 36, 15, "thud", "k"),))
        # The level a letter has of its own is not in the level of its hit.
        self.assertEqual(self.beat(hit((), 4, "h", 9, DRUMS, 16)), (Note(4, 12, 90, 9, "tick", "h"),))
        # Eleven rows are no one length, so the hit is the eight that are, and five rows are four.
        self.assertEqual(self.beat(hit((), 5, "k", 15, DRUMS, 16)), (drum(5, 8, "k"),))
        self.assertEqual(self.beat(hit((drum(8, 8, "k"),), 3, "h", 15, DRUMS, 16)), (drum(3, 4, "h"), drum(8, 8, "k")))

    def test_a_hit_lasts_no_longer_than_the_rows_its_letter_has_of_its_own(self) -> None:
        self.assertEqual(self.beat(hit((), 0, "s", 15, DRUMS, 16)), (drum(0, 2, "s"),))
        self.assertEqual(self.beat(hit((drum(5, 2, "s"),), 4, "s", 15, DRUMS, 16)), (drum(4, 1, "s"), drum(5, 2, "s")))
        # Five rows of its own in a room of five are four, as no length says five, and in a room of six are five.
        long = {"H": {"inst": "tick", "note": "f#5", "rows": 5}}
        self.assertEqual(hit((), 11, "H", 15, long, 16), (Note(11, 4, 78, 15, "tick", "H"),))
        self.assertEqual(hit((), 10, "H", 15, long, 16), (Note(10, 5, 78, 15, "tick", "H"),))

    def test_a_hit_replaces_what_begins_at_its_row_whatever_its_letter(self) -> None:
        notes = (drum(0, 4, "k"), drum(4, 2, "s", 9), drum(8, 8, "k"))
        self.assertEqual(self.beat(hit(notes, 4, "h", 12, DRUMS, 16)),
                         (drum(0, 4, "k"), drum(4, 4, "h", 12), drum(8, 8, "k")))
        self.assertEqual(self.beat(hit(notes, 4, "s", 3, DRUMS, 16)), (notes[0], drum(4, 2, "s", 3), notes[2]))

    def test_the_hit_before_a_new_one_is_fitted_again_and_a_note_before_it_is_cut_short(self) -> None:
        self.assertEqual(self.beat(hit((drum(0, 16, "k"),), 6, "h", 15, DRUMS, 16)),
                         (drum(0, 6, "k"), drum(6, 8, "h")))
        self.assertEqual(self.beat(hit((drum(0, 16, "k"),), 5, "h", 15, DRUMS, 16)),
                         (drum(0, 4, "k"), drum(5, 8, "h")))
        # A hit that was written shorter than the rule makes it is as long as the rule makes it afterwards.
        self.assertEqual(self.beat(hit((drum(0, 1, "k"), drum(1, 1, "s")), 4, "h", 15, DRUMS, 16)),
                         (drum(0, 1, "k"), drum(1, 2, "s"), drum(4, 12, "h")))
        tuned = Note(0.0, 8.0, 60, 15, "thud")
        self.assertEqual(self.beat(hit((tuned,), 6, "s", 15, DRUMS, 16)),
                         (Note(0.0, 6.0, 60, 15, "thud"), drum(6, 2, "s")))
        self.assertEqual(self.beat(hit((Note(0.0, 2.0, 60, 15, "thud"),), 6, "s", 15, DRUMS, 16))[0].rows, 2)

    def test_a_hit_is_taken_away_and_the_hit_before_it_is_fitted_again(self) -> None:
        notes = (drum(0, 4, "k"), drum(4, 2, "s"), drum(6, 2, "h"), drum(8, 8, "k"))
        self.assertEqual(self.beat(unhit(notes, 6, "h", DRUMS, 16)), (notes[0], notes[1], notes[3]))
        self.assertEqual(self.beat(unhit(notes, 4, "s", DRUMS, 16)), (drum(0, 6, "k"), notes[2], notes[3]))
        self.assertEqual(self.beat(unhit(notes, 8, "k", DRUMS, 16)), (notes[0], notes[1], drum(6, 8, "h")))
        self.assertEqual(self.beat(unhit(notes, 0, "k", DRUMS, 16)), notes[1:])
        self.assertEqual(self.beat(unhit(notes[:1], 0, "k", DRUMS, 16)), ())

    def test_only_a_hit_of_the_letter_at_the_row_is_taken_away(self) -> None:
        notes = (drum(0, 4, "k"), drum(4, 2, "s"))
        self.assertEqual(unhit(notes, 4, "h", DRUMS, 16), notes)
        self.assertEqual(unhit(notes, 2, "k", DRUMS, 16), notes)
        self.assertEqual(unhit(list(notes), 9, "s", DRUMS, 16), notes)

    def test_a_hit_the_pattern_or_the_song_has_no_place_for_is_refused(self) -> None:
        self.assertEqual(refusal(self, hit, (), 0, "t", 15, DRUMS, 16), "no drum letter 't'")
        self.assertEqual(refusal(self, hit, (), 0, "k", 15, None, 16), "no drum letter 'k'")
        for row in (16, -1):
            with self.subTest(row=row):
                self.assertEqual(refusal(self, hit, (), row, "k", 15, DRUMS, 16),
                                 f"row {row}: a hit is put on one of the 16 rows of the pattern")
        self.assertEqual(refusal(self, hit, (), 0, "k", 16, DRUMS, 16),
                         "a level is a whole number from 0 to 15, not 16")


def tuned(dice: random.Random, rows: int) -> tuple[Note, ...]:
    """The notes of a random tuned channel on the grid: most follow on from the one before, a few after a gap."""
    notes: list[Note] = []
    row = 0
    while True:
        row += dice.choice((0, 0, 0, 1, 2, 5))
        length = dice.randint(1, 9)
        if row + length > rows:
            return tuple(notes)
        notes.append(note(row, length, dice.randint(LOW, HIGH), dice.choice(LEVELS), dice.choice(INSTS)))
        row += length


def beaten(dice: random.Random, rows: int) -> tuple[Note, ...]:
    """The hits of a random drum channel on the grid, each of a length that its letter can be written with."""
    starts = sorted(dice.sample(range(rows), dice.randint(0, rows // 2)))
    hits = []
    for row, end in zip(starts, [*starts[1:], rows]):
        letter = dice.choice("khs")
        room = min(end - row, DRUMS[letter].get("rows", rows))
        hits.append(drum(row, dice.choice([rows for rows in (1, 2, 4) if rows <= room]), letter, dice.choice(LEVELS)))
    return tuple(hits)


def some(dice: random.Random, notes: tuple[Note, ...], rows: int) -> frozenset[int]:
    """A few of the notes, picked: those a random box touches, or any few of them."""
    if dice.random() < 0.4:
        row, low = dice.randrange(rows), dice.randint(LOW, HIGH)
        return boxed(notes, row, row + dice.randint(1, 16), low - 12, low + 12)
    return frozenset(dice.sample(range(len(notes)), min(len(notes), dice.randint(1, 4))))


def edited(dice: random.Random, notes: tuple[Note, ...], picked: frozenset[int],
           rows: int) -> tuple[tuple[Note, ...], frozenset[int]]:
    """A tuned channel after one random edit on the grid, and what is picked after it. Most edits change a span."""
    what = dice.choice(("place", "place", "place", "move", "move", "stretch", "stretch", "paste", "paste",
                        "remove", "level", "inst", "shift"))
    if what == "place":
        row = snapped(dice.uniform(0, rows - 0.01), dice.choice((1, 1, 2, 4)))
        new = note(row, dice.randint(1, 12), dice.randint(LOW, HIGH), dice.choice(LEVELS), dice.choice(INSTS))
        return place(notes, new, rows), frozenset()
    if what == "move":
        return move(notes, picked, dice.randint(-24, 24), dice.randint(-30, 30), rows, LOW, HIGH)
    if what == "stretch":
        return stretch(notes, picked, dice.randint(-8, 16), rows, dice.choice((1, 2)))
    if what == "paste":
        return pasted(notes, copied(notes, picked), dice.randint(0, rows), rows)
    if what == "remove":
        return remove(notes, picked)
    if what == "level":
        return with_level(notes, picked, dice.randrange(16))
    if what == "inst":
        return with_inst(notes, picked, dice.choice(INSTS))
    return shifted(notes, picked, dice.choice((-12, 12, 7, -5)), LOW, HIGH)


def struck(dice: random.Random, notes: tuple[Note, ...], picked: frozenset[int],
           rows: int) -> tuple[tuple[Note, ...], frozenset[int]]:
    """A drum channel after one random edit on the grid, and what is picked after it."""
    what = dice.randrange(7)
    if what < 3:
        return hit(notes, dice.randrange(rows), dice.choice("khs"), dice.randrange(16), DRUMS, rows), frozenset()
    if what < 5 and notes:
        taken = dice.choice(notes)
        return unhit(notes, taken.row, taken.drum, DRUMS, rows), frozenset()
    if what == 5:
        return with_level(notes, picked, dice.randrange(16))
    return remove(notes, picked)


class Proof(unittest.TestCase):
    def run_of_edits(self, seed: int, channel: str, made: object, edit: object) -> None:
        """Twelve random edits of a random channel, each keeping the rule, and what they leave saves."""
        dice = random.Random(seed)
        rows = 16 * dice.randint(1, 4)
        notes = made(dice, rows)
        picked = some(dice, notes, rows)
        for _ in range(12):
            if not picked or dice.random() < 0.5:
                picked = some(dice, notes, rows)
            notes, picked = edit(dice, notes, picked, rows)
            self.assertTrue(fitted(notes, rows))
            self.assertTrue(all(index in range(len(notes)) for index in picked))
        saved(self, notes, rows, channel)

    def test_three_hundred_runs_of_random_edits_leave_a_tuned_channel_that_saves(self) -> None:
        for seed in range(300):
            with self.subTest(seed=seed):
                self.run_of_edits(seed, "tune", tuned, edited)

    def test_two_hundred_runs_of_random_edits_leave_a_drum_channel_that_saves(self) -> None:
        for seed in range(200):
            with self.subTest(seed=seed):
                self.run_of_edits(seed, "beat", beaten, struck)


if __name__ == "__main__":
    unittest.main()
