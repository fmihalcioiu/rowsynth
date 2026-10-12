"""Tests for rowsynth.gui.roll: the sums of the piano roll, its view, its gestures, its parts, and what it draws."""
from __future__ import annotations

import math
import os
import unittest
from pathlib import Path

from rowsynth.errors import RowSynthError
from rowsynth.gui import chordview, layout, songview, sprites, theme
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.chordview import Chords
from rowsynth.gui.font import Font
from rowsynth.gui.kit import VELOCITY_HELP, velocity_bar
from rowsynth.gui.roll import (ACROSS_HELP, BOX, CHORD_LIT, CHORDS_H, DARKER, END_W, GHOST, HIGHEST, KEY_H, LEVELS,
                               LIGHTER, LOWEST, MOVE, OFF_GRID, PICK, PLACE, REMOVE, ROW_W_MAX, ROW_W_MIN, SLACK,
                               STRETCH, WHEEL_HELP, Map, Roll, begins, black, mix, name, path)
from rowsynth.gui.songview import Channel, Note, Pattern
from rowsynth.gui.ui import Event, Ui
from rowsynth.project import Project
from rowsynth.song import transpose

# pygame reads these when it is imported: no test opens a window or plays a sound.
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
try:
    import pygame
except ImportError:
    pygame = None

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
NEEDS = "pygame is not installed"

#: A roll whose grid is (50, 30, 260, 192), under a ruler from y 20, with a scroll bar at its right and one under it.
RECT = (10, 20, 305, 207)
#: Where the velocity lane of the roll stands, a pixel under the roll: its bars are two pixels high for each level.
LANE = (10, 228, 305, 34)
#: The surface they are drawn on, which is larger than them on every side, and the colour of what is neither's.
SIZE = (325, 270)
OUTSIDE = theme.GOOD
#: What the help line says the wheel does, at its end.
WHEEL = f"   wheel: {WHEEL_HELP}"

#: The colours of the upper edge and the start of a note in the colour of a song, and of its lower edge and end.
LIGHT = mix(theme.SONG, theme.TEXT, LIGHTER)
DARK = mix(theme.SONG, theme.BACKGROUND, DARKER)


def note(row: float, rows: float, pitch: int, level: int = 15, inst: str = "lead") -> Note:
    """A note that is no drum hit."""
    return Note(float(row), float(rows), pitch, level, inst)


#: Two notes of the same pitch in a row, one that plays with another instrument, and a short low one of no level.
LEAD = (note(0, 4, 64, 12), note(4, 2, 64), note(8, 4, 67, 8, "pluck"), note(14, 1, 60, 0))
#: Four notes for a hand to change, each on a key of its own: e4, f4, g4 and c4. They are shown in the same view.
TUNE = (note(0, 4, 64), note(4, 2, 65), note(8, 4, 67), note(14, 2, 60))


def pattern(lead: tuple[Note, ...] = LEAD, called: str = "p", rows: float = 32.0) -> Pattern:
    """A pattern of four channels: the lead, an arp that lies under its first notes, drums, and one with no text."""
    return Pattern(called, rows, (
        Channel("lead", "lead", False, lead, rows, True),
        Channel("arp", "pluck", False, (note(0, 8, 64), note(8, 8, 76)), rows, True),
        Channel("kit", "kick", True, (Note(2.0, 1.0, 62.0, 15, "hat", "h"),), rows, True),
        Channel("pad", "pad", False, (), 0.0, False)))


def chords(*written: str) -> Chords:
    """The chords of a pattern of two bars called p, read from the strings a song has for its bars."""
    table = {"Am": {"root": 9, "tones": [9, 0, 4]}, "F": {"root": 5, "tones": [5, 9, 0]}}
    return chordview.read(table, {"chords": {"p": list(written)}}, "p", 32.0)


def strength(colour: tuple[int, ...], ground: tuple[int, ...]) -> int:
    """How far a colour is from the one it lies on."""
    return sum(abs(one - other) for one, other in zip(colour, ground))


def rolled(ui: Ui, *given: object, **named: object) -> Roll:
    """A roll on a Ui as its owner has it there: its four parts after it, and its lane where `LANE` is."""
    roll = ui.add(Roll(RECT, *given, **named))
    for part in roll.parts:
        ui.add(part)
    roll.lane.rect = LANE
    return roll


def level_y(level: int) -> int:
    """The y the bar of a level reaches up to in the lane."""
    return velocity_bar(LANE, 0, 1, level / LEVELS)[1]


class Sums(unittest.TestCase):
    def test_the_x_of_a_row_and_the_row_at_an_x_are_opposites_at_several_views(self) -> None:
        self.assertEqual([Map(RECT).x(row) for row in (0, 1, 4.5)], [50, 59, 91])
        for row, row_w in ((0.0, 9), (3.0, 9), (-16.0, 2), (2.5, 7), (10 / 3, 24)):
            seen = Map(RECT, row=row, row_w=row_w)
            for whole in range(-20, 80):
                x = seen.x(whole)
                self.assertEqual((seen.x(whole + 1) - x, seen.row_at(x)), (row_w, whole))
                # Every pixel of a row is on that row.
                self.assertEqual({math.floor(seen.row_at(x + step)) for step in range(row_w)}, {whole})

    def test_the_y_of_a_pitch_and_the_pitch_at_a_y_are_opposites_at_several_views(self) -> None:
        self.assertEqual([Map(RECT).y(pitch) for pitch in (72, 71, 60)], [30, 41, 162])
        # Half a key further up, the key at the upper edge is cut in half.
        self.assertEqual(Map(RECT, top=72.5).y(72), 36)
        for top in (72.0, 60.0, 33.5, 131.0, 64.25):
            seen = Map(RECT, top=top)
            for pitch in range(LOWEST, HIGHEST + 1):
                y = seen.y(pitch)
                self.assertEqual(seen.y(pitch - 1) - y, KEY_H)
                self.assertEqual({seen.pitch_at(y + step) for step in range(KEY_H)}, {pitch})

    def test_five_keys_of_an_octave_are_black(self) -> None:
        self.assertEqual([step for step in range(12) if black(60 + step)], [1, 3, 6, 8, 10])
        self.assertEqual([black(pitch) for pitch in (37, 49, 48, 131)], [True, True, False, False])

    def test_a_pitch_is_named_with_its_octave_and_a_flat_as_moved_text_spells_it(self) -> None:
        self.assertEqual([name(pitch) for pitch in range(58, 64)], ["b-3", "b3", "c4", "d-4", "d4", "e-4"])
        self.assertEqual((name(LOWEST), name(HIGHEST)), ("c0", "b9"))
        for pitch in range(48, 72):
            spelt = name(pitch)
            self.assertEqual(transpose("o4 c", pitch - 60), f"o{spelt[-1]} {spelt[:-1]}")

    def test_the_bar_of_a_note_lies_between_the_lines_of_its_rows_in_the_row_of_its_key(self) -> None:
        seen = Map(RECT)
        self.assertEqual([seen.note(4, 2, 72), seen.note(4, 2, 70.0)], [(87, 30, 17, 10), (87, 52, 17, 10)])
        self.assertEqual(Map(RECT, row=4.0, top=73.0).note(4, 2, 72), (51, 41, 17, 10))
        # Two of the same pitch in a row leave the line between them free, and the shortest is a pixel wide.
        first, second = seen.note(0, 4, 60), seen.note(4, 4, 60)
        self.assertEqual((first[0] + first[2], second[0]), (seen.x(4), seen.x(4) + 1))
        self.assertEqual(Map(RECT, row_w=2).note(3, 0.25, 72), (57, 30, 1, 10))

    def test_the_scroll_bars_are_told_all_there_is_what_is_in_view_and_what_lies_before_it(self) -> None:
        # 120 keys of 11 pixels, 192 of them in view, from the 59th key under the highest. The rows go from a bar
        # before the pattern to a bar after its 32 rows: 64 rows of 9 pixels, 260 of them in view, from the 16th.
        self.assertEqual(Map(RECT).spans(32.0, 16), ((1320, 192, 649), (576, 260, 144)))
        moved = Map(RECT, row=-16.0, top=131.0, row_w=4)
        self.assertEqual(moved.spans(32.0, 16), ((1320, 192, 0), (324, 260, 0)))
        # At either end of each the view is as far as `within` lets it go.
        for seen in (Map(RECT, row=1e9, top=0.0), Map(RECT, row=-1e9, top=1e9)):
            kept = seen.within(32.0, 16)
            (keys, shown, above), (rows, wide, before) = kept.spans(32.0, 16)
            self.assertIn((above, before), ((keys - shown, rows - wide), (0, 0)))

    def test_the_parts_share_the_rectangle_and_in_view_is_what_has_a_pixel_on_the_grid(self) -> None:
        seen = Map(RECT)
        self.assertEqual((seen.keys, seen.ruler, seen.grid, seen.along, seen.across),
                         ((10, 30, 40, 192), (50, 20, 260, 10), (50, 30, 260, 192), (311, 30, 4, 192),
                          (50, 223, 260, 4)))
        # 260 pixels are 28 rows and most of another, and 192 are 17 keys and nearly half of another.
        self.assertEqual((seen.rows(), seen.pitches()), (range(0, 29), range(55, 73)))
        moved = Map(RECT, row=-2.5, top=80.5, row_w=4)
        self.assertEqual((moved.rows(), moved.pitches()), (range(-3, 63), range(64, 82)))
        for found in (seen, moved):
            self.assertEqual((found.x(found.rows()[0]) <= 50, found.x(found.rows()[-1]) < 310), (True, True))
            self.assertEqual((found.y(found.pitches()[-1]) <= 30, found.y(found.pitches()[0]) < 222), (True, True))
        # A rectangle too small for the keys, the ruler and the bars leaves no grid, and nothing is in view. Its
        # parts are cut to it.
        small = Map((0, 0, 30, 12))
        self.assertEqual((small.grid, len(small.rows()), len(small.pitches())), ((40, 10, 0, 0), 0, 0))
        self.assertEqual((small.keys, small.along, small.across), ((0, 10, 30, 0), (30, 10, 0, 0), (30, 11, 0, 1)))

    def test_a_mixed_colour_lies_between_the_two_it_is_mixed_from(self) -> None:
        self.assertEqual([mix((0, 0, 0), (200, 100, 50)), mix((0, 0, 0), (200, 100, 50), 0.25)],
                         [(100, 50, 25), (50, 25, 13)])
        self.assertEqual((mix(theme.SONG, theme.TEXT, 0), mix(theme.SONG, theme.TEXT, 1)), (theme.SONG, theme.TEXT))


class Hands(unittest.TestCase):
    def begun(self, pos: tuple[int, int], button: int = 1, *mods: str, rows: float = 32.0) -> tuple[str, int | None]:
        """What a press at a position begins among the notes of the tune, in the view a roll first has of them."""
        return begins(Map(RECT), TUNE, rows, pos, button, frozenset(mods))

    def test_a_press_on_a_note_moves_it_or_at_its_end_stretches_it_and_with_ctrl_picks_it(self) -> None:
        # e4 is from row 0 for 4 rows: its bar is (51, 118, 35, 10), and the last three pixels of it are from x 83.
        self.assertEqual(END_W, 3)
        self.assertEqual([self.begun((x, 120)) for x in (51, 82, 83, 85)],
                         [(MOVE, 0), (MOVE, 0), (STRETCH, 0), (STRETCH, 0)])
        self.assertEqual([self.begun((85, 120), 1, "ctrl"), self.begun((60, 120), 1, "shift")],
                         [(PICK, 0), (MOVE, 0)])
        # A bar of three pixels is narrower than twice its end: only its last pixel stretches it.
        narrow = Map(RECT, row_w=2)
        self.assertEqual(narrow.note(4, 2, 65), (59, 107, 3, 10))
        self.assertEqual([begins(narrow, TUNE, 32.0, (x, 110)) for x in (59, 60, 61)],
                         [(MOVE, 1), (MOVE, 1), (STRETCH, 1)])

    def test_a_press_where_no_note_is_places_one_in_the_pattern_and_with_shift_draws_a_box(self) -> None:
        self.assertEqual([self.begun((60, 90)), self.begun((60, 90), 1, "shift"), self.begun((60, 90), 1, "ctrl")],
                         [(PLACE, None), (BOX, None), (PLACE, None)])
        # On the line before a note and on the line under one, no note is.
        self.assertEqual([self.begun((50, 120)), self.begun((60, 128))], [(PLACE, None), (PLACE, None)])
        # A pattern of 8 rows ends at x 122. Past it nothing is placed, and a box is still drawn.
        self.assertEqual([self.begun((121, 90), rows=8.0), self.begun((122, 90), rows=8.0),
                          self.begun((122, 90), 1, "shift", rows=8.0)], [(PLACE, None), ("", None), (BOX, None)])
        # Two rows before the pattern are in view: it begins at x 68.
        before = Map(RECT, row=-2.0)
        self.assertEqual([begins(before, TUNE, 32.0, (x, 90)) for x in (67, 68)], [("", None), (PLACE, None)])

    def test_the_right_button_removes_and_any_other_press_begins_nothing(self) -> None:
        self.assertEqual([self.begun((60, 120), 3), self.begun((60, 90), 3), self.begun((60, 120), 3, "ctrl")],
                         [(REMOVE, 0), (REMOVE, None), (REMOVE, 0)])
        # The middle button, the keys, the ruler, the scroll bar under the grid, and the lane under the roll.
        self.assertEqual([self.begun((60, 120), 2), self.begun((20, 120)), self.begun((60, 25)), self.begun((60, 223)),
                          self.begun((60, 240))], [("", None)] * 5)

    def test_a_drag_has_moved_by_the_steps_of_the_grid_and_the_keys_between_its_two_ends(self) -> None:
        seen = Map(RECT)
        # Row 1 is from x 59 to 67, and e4 from y 118 to 128.
        self.assertEqual([seen.moved((60, 120), pos) for pos in ((60, 120), (67, 128), (59, 118))], [(0.0, 0)] * 3)
        self.assertEqual([seen.moved((60, 120), pos) for pos in ((68, 117), (58, 129), (80, 100), (300, 31))],
                         [(1.0, 1), (-1.0, -1), (2.0, 2), (26.0, 8)])
        # With steps of 4 rows it has moved when it comes to another step: the second is from x 86.
        self.assertEqual([seen.moved((60, 120), (x, 120), 4) for x in (85, 86, 150)],
                         [(0.0, 0), (4.0, 0), (8.0, 0)])
        # In another view: rows of 4 pixels from row 2.5, and half a key cut off at the top.
        self.assertEqual(Map(RECT, row=2.5, top=80.5, row_w=4).moved((60, 120), (68, 98)), (2.0, 2))

    def test_the_mouse_passes_every_pixel_on_the_line_between_two_positions(self) -> None:
        self.assertEqual((path((5, 5), (5, 5)), path((5, 5), (8, 5))), ([], [(6, 5), (7, 5), (8, 5)]))
        self.assertEqual(path((0, 0), (4, -2)), [(1, 0), (2, -1), (3, -1), (4, -2)])
        self.assertEqual(path((2, 9), (0, 3)), [(2, 8), (1, 7), (1, 6), (1, 5), (0, 4), (0, 3)])


class View(unittest.TestCase):
    def setUp(self) -> None:
        self.ui = Ui()
        self.roll = rolled(self.ui, "the notes of a channel")
        self.roll.show(pattern(), "lead", 16, theme.SONG)

    def view(self) -> tuple[float, float, int]:
        """The view of the roll: its first row, its top pitch and the width of a row."""
        return self.roll.row, self.roll.top, self.roll.row_w

    def wheel(self, steps: int, *mods: str, pos: tuple[int, int] = (162, 100)) -> bool:
        """Turn the wheel with the mouse on the grid, some keys held. Whether it was used."""
        return self.ui.handle(Event("wheel", pos, wheel=steps, mods=frozenset(mods)))

    def test_a_channel_is_fitted_from_row_0_with_its_notes_in_the_middle_of_the_keys(self) -> None:
        # Its notes are c4 to g4, shown as the octave from b-3 to b-4: five keys over it, and four and a bit under.
        self.assertEqual(self.view(), (0.0, 72.0, 9))
        seen = self.roll.map
        over, under = seen.y(70) - 30, 222 - seen.y(57)
        self.assertEqual((over, under, abs(over - under) < KEY_H), (22, 27, True))
        # A channel with no notes is shown round middle c.
        self.roll.show(pattern(), "pad", 16, theme.SONG)
        self.assertEqual((self.view(), self.roll.notes), ((0.0, 68.0, 9), ()))

    def test_the_wheel_scrolls_the_keys_and_with_shift_the_rows(self) -> None:
        self.assertTrue(self.wheel(1))
        self.assertEqual(self.view(), (0.0, 74.0, 9))
        self.wheel(-3)
        self.assertEqual(self.view(), (0.0, 68.0, 9))
        self.assertTrue(self.wheel(-1, "shift"))
        self.assertEqual(self.view(), (4.0, 68.0, 9))
        self.wheel(2, "shift")
        self.assertEqual(self.view(), (-4.0, 68.0, 9))
        # Off the roll the wheel is not its to use, and neither is a key.
        self.assertEqual((self.wheel(1, pos=(5, 5)), self.ui.handle(Event("key", key="space"))), (False, False))
        self.assertEqual(self.view(), (-4.0, 68.0, 9))

    def test_the_wheel_with_ctrl_zooms_the_rows_between_the_least_and_the_most(self) -> None:
        self.assertTrue(self.wheel(1, "ctrl"))
        self.assertEqual(self.roll.row_w, 10)
        self.wheel(100, "ctrl")
        self.assertEqual(self.roll.row_w, ROW_W_MAX)
        self.wheel(-100, "ctrl", "shift")
        self.assertEqual((self.roll.row_w, self.roll.top), (ROW_W_MIN, 72.0))

    def test_a_zoom_keeps_the_row_under_the_mouse_where_it_was(self) -> None:
        self.roll.show(pattern(rows=128.0), "lead", 16, theme.SONG)
        self.wheel(-5, "shift")
        # The mouse is 4 pixels into row 32.
        self.assertEqual((self.roll.row, self.roll.map.row_at(162)), (20.0, 32 + 4 / 9))
        for step in (1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1):
            before = self.roll.map.row_at(162)
            self.wheel(step, "ctrl")
            seen = self.roll.map
            # It is where it was to the pixel, which is half a pixel either way in rows.
            self.assertLess(abs(seen.row_at(162) - before), 0.5 / seen.row_w + 1e-9)
            self.assertTrue(seen.x(32) <= 162 < seen.x(33))
        self.assertEqual(self.roll.row_w, 5)
        # Over the keys, it is the row at the left edge of the grid that stays.
        self.wheel(3, "ctrl", pos=(20, 100))
        self.assertEqual(self.roll.row_w, 8)
        self.assertLess(abs(self.roll.map.row_at(50) - seen.row_at(50)), 0.5 / 8)

    def test_the_view_leaves_the_pattern_by_no_more_than_a_bar_and_never_the_keys(self) -> None:
        self.wheel(100, "shift")
        self.assertEqual(self.roll.row, -16.0)
        self.wheel(-100, "shift")
        # The bar after the 32 rows ends at the right edge of the grid.
        self.assertEqual((round(self.roll.row, 3), self.roll.map.x(48)), (19.111, 310))
        self.wheel(100)
        self.assertEqual((self.roll.top, self.roll.map.pitches()[-1]), (131.0, HIGHEST))
        self.wheel(-100)
        self.assertEqual((self.roll.map.y(LOWEST) + KEY_H, self.roll.map.pitches()[0]), (222, LOWEST))
        # The same pattern, come back shorter, is still shown within the limits. One that fits with room to spare
        # can be moved back by a bar, and not forward.
        self.roll.show(pattern(rows=16.0), "lead", 16, theme.SONG)
        self.assertEqual(round(self.roll.row, 3), 3.111)
        self.roll.show(pattern(rows=8.0), "lead", 16, theme.SONG)
        self.wheel(-100, "shift")
        self.assertEqual(self.roll.row, 0.0)

    def test_shown_again_the_same_channel_keeps_its_view_and_another_is_fitted_anew(self) -> None:
        self.wheel(3)
        self.wheel(-2, "shift")
        self.wheel(2, "ctrl")
        moved = self.view()
        self.assertNotEqual(moved, (0.0, 72.0, 9))
        # The file changed outside: the same pattern and channel, with other notes.
        self.roll.show(pattern((note(0, 2, 90),)), "lead", 16, theme.SONG)
        self.assertEqual((self.view(), len(self.roll.notes)), (moved, 1))
        # Another channel: the view is set on its notes, at the zoom there was.
        self.roll.show(pattern(), "arp", 16, theme.SOUND)
        self.assertEqual((self.view(), self.roll.colour), ((0.0, 78.0, 11), theme.SOUND))
        # Another pattern of that channel: the view stays where it was left, and is never set on the notes again.
        self.wheel(3)
        self.wheel(-1, "shift")
        left = self.view()
        self.assertEqual(left[1:], (84.0, 11))
        self.roll.show(pattern(called="q"), "arp", 16, theme.SOUND)
        self.assertEqual(self.view(), left)
        self.roll.fit()
        self.assertEqual(self.view(), (0.0, 78.0, 11))

    def test_a_channel_the_pattern_lacks_and_a_bar_that_is_not_four_beats_are_refused(self) -> None:
        for channel, bar_rows, wrong in (("bass", 16, "pattern 'q' has no channel called 'bass'"),
                                         ("lead", 6, "a bar is four beats of whole rows, which 6 rows are not"),
                                         ("lead", 0, "a bar is four beats of whole rows, which 0 rows are not")):
            with self.assertRaises(RowSynthError) as refused:
                self.roll.show(pattern(called="q"), channel, bar_rows, theme.SOUND)
            self.assertEqual(str(refused.exception), wrong)
        # What was shown before is still shown.
        self.assertEqual((self.roll.pattern.name, self.roll.channel, self.roll.bar_rows, self.roll.colour),
                         ("p", "lead", 16, theme.SONG))

    def test_note_at_is_the_note_whose_bar_a_position_is_on(self) -> None:
        # The first note is e4 from row 0 for 4 rows: its bar is (51, 118, 35, 10). The second follows it.
        self.assertEqual([self.roll.note_at(pos) for pos in ((51, 118), (85, 127), (87, 118), (178, 165))],
                         [0, 0, 1, 3])
        # On the line between two notes, on the line under a note, over it, in the keys and in the lane.
        self.assertEqual([self.roll.note_at(pos) for pos in ((86, 120), (60, 128), (60, 117), (20, 120), (60, 240))],
                         [None, None, None, None, None])
        self.wheel(-1, "shift")
        self.assertEqual([self.roll.note_at((51, 118)), self.roll.note_at((67, 120)), self.roll.note_at((69, 120))],
                         [1, 1, None])
        self.assertIsNone(Roll(RECT).note_at((60, 120)))

    def test_the_help_line_says_what_the_left_press_the_right_press_and_the_wheel_do_where_the_mouse_is(self) -> None:
        self.assertEqual((self.roll.help, self.ui.help), ("the notes of a channel", ""))
        rub = "   right press: rub it out"
        for pos, mods, line in (((60, 120), (), f"press: pick e4, a drag moves it{rub}{WHEEL}"),
                                ((84, 120), (), f"press: pick e4, a drag makes it longer or shorter{rub}{WHEEL}"),
                                ((60, 120), ("ctrl",), f"press: pick e4 as well{rub}{WHEEL}"),
                                ((233, 87), (), f"press: put a note at g4, bar 2, row 5{WHEEL}"),
                                ((86, 120), (), f"press: put a note at e4, bar 1, row 5{WHEEL}"),
                                ((233, 87), ("shift",), f"press: drag a box round the notes to pick{WHEEL}"),
                                ((130, 240), (), VELOCITY_HELP),
                                ((20, 90), (), f"press: hear g4{WHEEL}"),
                                ((20, 25), (), "the notes of a channel"),
                                ((60, 224), (), ACROSS_HELP)):
            self.ui.handle(Event("move", pos, mods=frozenset(mods)))
            self.assertEqual(self.ui.help, line)
        self.assertEqual(VELOCITY_HELP, "press: set how hard a note is struck, a drag sets each note it passes   "
                                        "right press: back to the usual velocity")
        self.assertEqual(ACROSS_HELP, f"press: drag the bar to go along the rows, or press beside it to jump{WHEEL}")
        # With steps of 4 rows a note is placed where the step begins that the mouse is in.
        self.roll.grid = 4
        self.ui.handle(Event("move", (260, 87)))
        self.assertEqual((int(self.roll.map.row_at(260)), self.ui.help),
                         (23, f"press: put a note at g4, bar 2, row 5{WHEEL}"))
        # Before the pattern a press places nothing, and with the shift of the wheel still held it draws a box.
        self.wheel(1, "shift", pos=(55, 87))
        self.assertEqual((self.roll.row, self.ui.help), (-4.0, f"press: drag a box round the notes to pick{WHEEL}"))
        self.ui.handle(Event("move", (55, 87)))
        self.assertEqual(self.ui.help, f"wheel: {WHEEL_HELP}")
        # Off the grid the line is the one the roll was made with.
        self.ui.handle(Event("move", (60, 25)))
        self.roll.help = "another line"
        self.assertEqual(self.ui.help, "another line")

    def test_a_drag_of_a_scroll_bar_moves_the_view_and_a_press_beside_its_thumb_jumps_a_view_s_length(self) -> None:
        along, across = self.roll.along, self.roll.across
        self.assertEqual((along.rect, across.rect, along.upright, across.upright),
                         ((311, 30, 4, 192), (50, 223, 260, 4), True, False))
        # The thumb along the keys stands for the 192 pixels in view of 1320: it is 28 long, and starts 94 down.
        self.assertEqual((along.thumb(), self.view()), ((311, 124, 4, 28), (0.0, 72.0, 9)))
        self.ui.handle(Event("down", (312, 130), button=1))
        self.ui.handle(Event("move", (312, 120)))
        # Ten pixels up are 10 of the 164 the thumb can travel, which are 1128 pixels of keys: 69 of them.
        self.assertEqual((along.scroll.offset, along.thumb()[1], self.roll.keys.top), (580, 114, self.roll.top))
        self.assertAlmostEqual(self.roll.top, 72 + 69 / 11)
        self.ui.handle(Event("up", (312, 120), button=1))
        # A press under the thumb goes down by what is in view, which is 192 pixels of keys.
        self.ui.handle(Event("down", (312, 200), button=1))
        self.ui.handle(Event("up", (312, 200), button=1))
        self.assertAlmostEqual(self.roll.top, 72 + (69 - 192) / 11)
        # The bar under the grid goes along the rows: a press after its thumb goes on by what is in view, which
        # here is to where the bar after the pattern ends at the right edge of the grid.
        x, y, w, _ = across.thumb()
        self.assertEqual((x, w), (115, 117))
        self.ui.handle(Event("down", (x + w + 3, y + 1), button=1))
        self.ui.handle(Event("up", (x + w + 3, y + 1), button=1))
        self.assertEqual((round(self.roll.row, 3), self.roll.map.x(48), self.view()[2]), (19.111, 310, 9))
        # The wheel on a bar is the roll's, and so is a key that the bar is handed.
        self.assertTrue(self.wheel(1, pos=(312, 100)))
        self.assertAlmostEqual(self.roll.top, 72 + (69 - 192) / 11 + 2)
        self.roll.show(pattern(TUNE), "lead", 16, theme.SONG)
        self.assertEqual((self.ui.focus, self.ui.handle(Event("key", key="a", mods=frozenset({"ctrl"}))),
                          self.roll.picked), (across, True, {0, 1, 2, 3}))


class Edits(unittest.TestCase):
    def setUp(self) -> None:
        self.ui = Ui()
        self.edits: list[tuple[tuple[Note, ...], str]] = []
        self.heard: list[tuple[float, str]] = []
        self.roll = rolled(self.ui, "the notes of a channel", self.save,
                           lambda pitch, inst: self.heard.append((pitch, inst)))
        self.show(TUNE)

    def show(self, notes: tuple[Note, ...]) -> None:
        """Show the roll a pattern of 32 rows whose lead has some notes."""
        self.roll.show(pattern(notes), "lead", 16, theme.SONG)

    def save(self, notes: tuple[Note, ...], label: str) -> None:
        """What an owner does with an edit: it is kept here, and the roll is shown the notes it gave."""
        self.edits.append((notes, label))
        self.show(notes)

    def at(self, row: int, pitch: int) -> tuple[int, int]:
        """A pixel in the middle of a row and of a key: on the bar of a note that sounds there, short of its end."""
        seen = self.roll.map
        return seen.x(row) + 4, seen.y(pitch) + 4

    def end(self, index: int) -> tuple[int, int]:
        """The last pixel of the bar of one of the notes."""
        found = self.roll.notes[index]
        x, y, w, _ = self.roll.map.note(found.row, found.rows, found.pitch)
        return x + w - 1, y + 4

    def mouse(self, kind: str, pos: tuple[int, int], button: int = 1, *mods: str) -> bool:
        """Hand the Ui what the mouse did, some keys held. Whether it was used."""
        return self.ui.handle(Event(kind, pos, button=0 if kind == "move" else button, mods=frozenset(mods)))

    def drag(self, *stops: tuple[int, int], button: int = 1, mods: tuple[str, ...] = ()) -> None:
        """Press at the first position, move to each of the others in turn, and release at the last."""
        self.mouse("down", stops[0], button, *mods)
        for stop in stops[1:]:
            self.mouse("move", stop, button, *mods)
        self.mouse("up", stops[-1], button, *mods)

    def key(self, key: str, *mods: str) -> bool:
        """Hand the roll a key, as its owner does. Whether it is one of the roll's."""
        return self.roll.key(Event("key", key=key, mods=frozenset(mods)))

    def test_a_click_on_the_empty_grid_places_a_note_of_four_rows_at_the_top_level_and_it_is_heard(self) -> None:
        self.assertEqual((self.roll.row, self.roll.top, self.roll.row_w, self.roll.grid), (0.0, 72.0, 9, 1.0))
        self.assertTrue(self.mouse("down", self.at(20, 62)))
        self.assertEqual((self.roll.notes, self.edits, self.heard), (TUNE, [], []))
        self.mouse("up", self.at(20, 62))
        placed = (*TUNE, note(20, 4, 62))
        self.assertEqual((self.edits, self.heard), ([(placed, "note placed")], [(62, "lead")]))
        self.assertEqual((self.roll.notes, self.roll.picked), (placed, frozenset()))
        # With the rows scrolled on by four, row 30 is in view: what a note has past the end of the pattern is cut off.
        self.ui.handle(Event("wheel", (162, 100), wheel=-1, mods=frozenset({"shift"})))
        self.drag(self.at(30, 62))
        self.assertEqual(self.edits[1:], [((*placed, note(30, 2, 62)), "note placed")])

    def test_a_click_places_a_note_as_long_as_the_last_one_dragged_or_stretched_at_the_last_level_set(self) -> None:
        self.drag(self.at(18, 62), self.at(19, 62))
        self.drag(self.at(22, 62))
        self.assertEqual(self.roll.notes[4:], (note(18, 2, 62), note(22, 2, 62)))
        # The second is stretched from its end in row 23 to row 26, and the first is set to level 9 in the lane.
        self.drag(self.end(5), self.at(26, 62))
        self.drag((self.at(18, 62)[0], level_y(9)))
        self.drag(self.at(28, 70))
        self.assertEqual(self.roll.notes[4:], (note(18, 2, 62, 9), note(22, 5, 62), note(28, 4, 70, 9)))
        self.assertEqual([label for _, label in self.edits],
                         ["note placed", "note placed", "note stretched", "velocity changed", "note placed"])
        self.assertEqual(self.heard, [(62, "lead"), (62, "lead"), (70, "lead")])

    def test_a_drag_places_a_longer_note_that_cuts_the_note_it_begins_in_and_removes_one_that_begins_in_it(
            self) -> None:
        # From row 2 to row 5: e4 sounds until row 4, where f4 begins.
        self.drag(self.at(2, 62), self.at(5, 62))
        self.assertEqual(self.edits, [((note(0, 2, 64), note(2, 4, 62), *TUNE[2:]), "note placed")])
        # Dragged to the left, and to other keys, it is one step of the grid long on the key of the press.
        self.drag(self.at(20, 62), self.at(17, 70))
        self.assertEqual(self.roll.notes[4:], (note(20, 1, 62),))
        # With steps of 4 rows it begins where its step does and ends with the step the mouse is in.
        self.roll.grid = 4
        self.drag(self.at(21, 70), self.at(26, 70))
        self.assertEqual((self.roll.notes[4:], len(self.edits)), ((note(20, 8, 70),), 3))

    def test_a_press_on_a_note_picks_it_alone_and_with_no_drag_changes_nothing(self) -> None:
        self.assertTrue(self.mouse("down", self.at(0, 64)))
        self.assertEqual(self.roll.picked, {0})
        self.mouse("up", self.at(0, 64))
        self.drag(self.at(8, 67))
        self.assertEqual(self.roll.picked, {2})
        self.drag(self.end(3))
        self.assertEqual((self.roll.picked, self.roll.notes, self.edits, self.heard), ({3}, TUNE, [], []))

    def test_a_drag_moves_a_note_by_rows_and_by_keys_and_it_stays_picked_where_it_now_is(self) -> None:
        # c4 goes 11 rows back and 6 keys up: e4 still sounds where it begins, and f4 begins inside it.
        self.mouse("down", self.at(14, 60))
        self.mouse("move", self.at(3, 66))
        self.assertEqual(self.ui.help, f"press: pick c4, a drag moves it   right press: rub it out{WHEEL}")
        self.mouse("up", self.at(3, 66))
        moved = (note(0, 3, 64), note(3, 2, 66), TUNE[2])
        self.assertEqual(self.edits, [(moved, "note moved")])
        self.assertEqual((self.roll.notes, self.roll.picked), (moved, {1}))

    def test_picked_notes_move_together_and_stop_together_at_the_edges_of_the_pattern(self) -> None:
        self.drag(self.at(8, 67), mods=("ctrl",))
        self.drag(self.at(14, 60), mods=("ctrl",))
        self.assertEqual(self.roll.picked, {2, 3})
        # 19 rows on, where the pattern has 16 rows left after them, and 2 keys up.
        self.drag(self.at(9, 67), self.at(28, 69))
        self.assertEqual(self.edits, [((TUNE[0], TUNE[1], note(24, 4, 69), note(30, 2, 62)), "2 notes moved")])
        self.assertEqual(self.roll.picked, {2, 3})
        # Back past the start of the pattern, they stop at row 0 as far apart as they were.
        self.drag(self.at(24, 69), (10, self.at(24, 69)[1]))
        self.assertEqual(self.edits[1:], [((note(0, 4, 69), TUNE[1], note(6, 2, 62)), "2 notes moved")])
        self.assertEqual(self.roll.picked, {0, 2})

    def test_a_drag_from_the_end_of_a_note_stretches_the_picked_notes_and_none_is_shorter_than_a_step(self) -> None:
        self.mouse("move", self.end(0))
        self.assertEqual(self.ui.help,
                         f"press: pick e4, a drag makes it longer or shorter   right press: rub it out{WHEEL}")
        # From row 3, where e4 ends, to row 5: f4 begins in what it is stretched to.
        self.drag(self.end(0), self.at(5, 60))
        self.assertEqual(self.edits, [((note(0, 6, 64), *TUNE[2:]), "note stretched")])
        self.assertEqual(self.roll.picked, {0})
        self.drag(self.end(0), (5, 100))
        self.assertEqual(self.roll.notes, (note(0, 1, 64), *TUNE[2:]))
        # With g4 picked as well, both are a row longer.
        self.drag(self.at(8, 67), mods=("ctrl",))
        self.drag(self.end(1), self.at(12, 67))
        self.assertEqual(self.edits[2:], [((note(0, 2, 64), note(8, 5, 67), TUNE[3]), "2 notes stretched")])
        self.assertEqual(self.roll.picked, {0, 1})

    def test_ctrl_adds_a_note_to_the_picked_ones_or_takes_it_out_and_moves_nothing(self) -> None:
        self.drag(self.at(0, 64))
        self.drag(self.at(8, 67), mods=("ctrl",))
        self.assertEqual(self.roll.picked, {0, 2})
        self.mouse("move", self.at(0, 64), 1, "ctrl")
        self.assertEqual(self.ui.help, f"press: take e4 out of the picked notes   right press: rub it out{WHEEL}")
        self.drag(self.at(0, 64), self.at(3, 70), mods=("ctrl",))
        self.assertEqual((self.roll.picked, self.roll.notes, self.edits), ({2}, TUNE, []))
        # A press without ctrl on a picked note leaves the others picked.
        self.drag(self.at(14, 60), mods=("ctrl",))
        self.drag(self.at(8, 67))
        self.assertEqual(self.roll.picked, {2, 3})

    def test_shift_draws_a_box_that_picks_what_it_touches_and_a_click_with_shift_picks_nothing(self) -> None:
        # From row 3 of a-4 to row 9 of e-4: e4, f4 and g4 sound in it, and c4 does not.
        self.assertTrue(self.mouse("down", self.at(3, 68), 1, "shift"))
        self.mouse("move", self.at(9, 63), 1, "shift")
        self.assertEqual((self.roll.picked, self.roll.drawn, self.ui.help),
                         (frozenset(), TUNE, f"press: drag a box round the notes to pick{WHEEL}"))
        self.mouse("up", self.at(9, 63), 1, "shift")
        self.assertEqual((self.roll.picked, self.edits), ({0, 1, 2}, []))
        # A box drawn from its other corner, over the rows and the keys of f4 alone.
        self.drag(self.at(6, 66), self.at(4, 65), mods=("shift",))
        self.assertEqual(self.roll.picked, {1})
        self.drag(self.at(20, 62), mods=("shift",))
        self.assertEqual((self.roll.picked, self.roll.notes, self.edits), (frozenset(), TUNE, []))

    def test_the_right_button_removes_the_note_under_it(self) -> None:
        self.drag(self.at(8, 67))
        # Where no note is it removes none, and what is picked stays picked.
        self.drag(self.at(20, 62), self.at(22, 64), button=3)
        self.assertEqual((self.edits, self.roll.picked), ([], {2}))
        self.assertTrue(self.mouse("down", self.at(4, 65), 3))
        self.assertEqual((self.roll.drawn, self.edits, self.ui.help),
                         ((TUNE[0], *TUNE[2:]), [],
                          f"press: pick f4, a drag moves it   right press: rub it out{WHEEL}"))
        self.mouse("up", self.at(4, 65), 3)
        self.assertEqual((self.edits, self.roll.picked), ([((TUNE[0], *TUNE[2:]), "note removed")], frozenset()))

    def test_held_and_dragged_the_right_button_removes_every_note_it_passes_over_as_one_edit(self) -> None:
        # From e4 to g4 in one move: the straight line between the two crosses the bar of f4.
        self.mouse("down", self.at(1, 64), 3)
        self.mouse("move", self.at(10, 67), 3)
        self.assertEqual((self.roll.drawn, self.roll.notes, self.edits), (TUNE[3:], TUNE, []))
        self.mouse("up", self.at(10, 67), 3)
        self.assertEqual((self.edits, self.roll.notes), ([(TUNE[3:], "3 notes removed")], TUNE[3:]))

    def test_a_drag_in_the_lane_sets_the_level_of_each_note_it_passes_from_its_height_as_one_edit(self) -> None:
        # The bar of level 5 reaches up to y 250, and that of level 10 to y 240.
        self.assertEqual((level_y(5), level_y(10), self.roll.lane.span), (250, 240, (50, 260)))
        self.assertTrue(self.mouse("down", (60, 250)))
        self.assertEqual((self.roll.drawn[0], self.roll.busy, self.ui.help), (note(0, 4, 64, 5), True, VELOCITY_HELP))
        self.mouse("move", (60, 240))
        self.mouse("move", (95, 240))
        self.assertEqual((self.roll.notes, self.edits), (TUNE, []))
        # The lane shows the notes as the drag would leave them: the first two bars are 20 pixels high.
        self.assertEqual([round(value * LEVELS) for _, _, value, _ in self.roll.lane.items], [10, 10, 15, 15])
        self.mouse("up", (95, 240))
        levelled = (note(0, 4, 64, 10), note(4, 2, 65, 10), *TUNE[2:])
        self.assertEqual((self.edits, self.roll.busy), ([(levelled, "velocity changed")], False))
        # Where no note sounds none is set, and a note that has the level already is not changed.
        self.drag((230, 242))
        self.drag((180, 242), (180, 100))
        self.assertEqual(len(self.edits), 1)
        self.drag((180, 242), (180, 300))
        self.assertEqual(self.edits[1:], [((*levelled[:3], note(14, 2, 60, 0)), "velocity changed")])

    def test_the_right_button_in_the_lane_puts_a_note_back_to_the_highest_level_and_escape_drops_a_drag_there(
            self) -> None:
        self.show((note(0, 4, 64, 6), *TUNE[1:]))
        # Where no bar is the press is not used. On the note it is one edit, made at once.
        self.assertEqual([self.mouse("down", (230, 250), 3), self.mouse("down", (60, 250), 3)], [False, True])
        self.mouse("up", (60, 250), 3)
        self.assertEqual((self.edits, self.roll.busy), ([(TUNE, "velocity changed")], False))
        # A drag that is dropped changes nothing, and neither does what the mouse does before it is let go.
        self.mouse("down", (60, 250))
        self.assertEqual(self.roll.drawn[0], note(0, 4, 64, 5))
        self.assertTrue(self.ui.handle(Event("key", key="escape")))
        self.mouse("move", (95, 240))
        self.mouse("up", (95, 240))
        self.assertEqual((self.roll.drawn, self.roll.busy, len(self.edits)), (TUNE, False, 1))
        # Under the keys, left of where the grid begins, the lane has no bars, and a press there is nobody's.
        self.assertFalse(self.mouse("down", (30, 250)))
        self.assertEqual(self.roll.busy, False)

    def test_delete_and_backspace_remove_the_picked_notes(self) -> None:
        self.assertTrue(self.key("delete"))
        self.assertEqual(self.edits, [])
        self.drag(self.at(0, 64))
        self.drag(self.at(8, 67), mods=("ctrl",))
        self.assertTrue(self.key("delete"))
        self.assertEqual((self.edits, self.roll.picked), ([((TUNE[1], TUNE[3]), "2 notes removed")], frozenset()))
        self.drag(self.at(4, 65))
        self.assertTrue(self.key("backspace"))
        self.assertEqual(self.edits[1:], [((TUNE[3],), "note removed")])

    def test_up_and_down_move_the_picked_notes_a_semitone_and_with_shift_an_octave(self) -> None:
        self.drag(self.at(0, 64))
        self.drag(self.at(14, 60), mods=("ctrl",))
        for key, mods, pitches in (("up", (), (65, 61)), ("down", ("shift",), (53, 49)), ("down", (), (52, 48)),
                                   ("up", ("shift",), (64, 60))):
            self.assertTrue(self.key(key, *mods))
            notes, label = self.edits[-1]
            self.assertEqual(((notes[0].pitch, notes[3].pitch), notes[1:3], label, self.roll.picked),
                             (pitches, TUNE[1:3], "2 notes moved", {0, 3}))
        self.assertEqual((self.roll.notes, len(self.edits)), (TUNE, 4))
        # Where one of them would leave the keys of the roll, none moves.
        self.show((note(0, 4, HIGHEST - 1), note(4, 4, 60)))
        self.key("a", "ctrl")
        self.key("up")
        self.key("up")
        self.assertEqual((self.roll.notes, len(self.edits)), ((note(0, 4, HIGHEST), note(4, 4, 61)), 5))

    def test_left_and_right_move_the_picked_notes_a_step_of_the_grid(self) -> None:
        self.drag(self.at(14, 60))
        self.assertTrue(self.key("right"))
        self.assertEqual(self.edits, [((*TUNE[:3], note(15, 2, 60)), "note moved")])
        # With steps of 4 rows it comes to row 11, where g4 still sounds, and then to row 7, before g4 begins.
        self.roll.grid = 4
        self.assertTrue(self.key("left"))
        self.assertEqual((self.roll.notes, self.roll.picked), ((*TUNE[:2], note(8, 3, 67), note(11, 2, 60)), {3}))
        self.key("left")
        self.assertEqual((self.roll.notes, self.roll.picked), ((*TUNE[:2], note(7, 2, 60)), {2}))
        # It stops at row 0, and from there it does not move.
        self.key("left")
        self.key("left")
        self.key("left")
        self.assertEqual((self.roll.notes, self.roll.picked, len(self.edits)), ((note(0, 2, 60),), {0}, 5))

    def test_ctrl_and_a_picks_all_escape_drops_the_picked_and_any_other_key_is_not_the_rolls(self) -> None:
        self.assertTrue(self.key("a", "ctrl"))
        self.assertEqual(self.roll.picked, {0, 1, 2, 3})
        self.assertTrue(self.key("escape"))
        self.assertEqual(self.roll.picked, frozenset())
        for key, mods in (("a", ()), ("space", ()), ("z", ("ctrl",)), ("delete", ("ctrl",)), ("up", ("ctrl",)),
                          ("left", ("shift",)), ("escape", ("shift",))):
            self.assertFalse(self.key(key, *mods), (key, mods))
        # Without the focus, which no press has given it here, the roll is handed no key by the Ui.
        self.key("a", "ctrl")
        self.assertFalse(self.ui.handle(Event("key", key="delete")))
        self.assertEqual((self.roll.picked, self.roll.notes, self.edits), ({0, 1, 2, 3}, TUNE, []))

    def test_with_the_focus_a_key_from_the_ui_is_the_rolls_and_one_that_is_not_goes_on_to_its_owner(self) -> None:
        typed: list[str] = []
        self.roll.typed = lambda event: typed.append(event.key) or event.key == "z"
        self.drag(self.at(0, 64))
        self.assertIs(self.ui.focus, self.roll)
        self.assertTrue(self.ui.handle(Event("key", key="delete")))
        self.assertEqual((self.edits, typed), ([(TUNE[1:], "note removed")], []))
        # What the owner says of a key that is not the roll's is what the Ui is told. With no owner it is not used.
        self.assertEqual([self.ui.handle(Event("key", key=key)) for key in ("z", "space")], [True, False])
        self.assertEqual(typed, ["z", "space"])
        self.roll.typed = None
        self.assertFalse(self.ui.handle(Event("key", key="z")))

    def test_a_press_on_a_key_of_the_strip_is_heard_gives_the_focus_and_changes_nothing(self) -> None:
        self.drag(self.at(0, 64))
        # A press off the roll takes the focus from it.
        self.mouse("down", (5, 5))
        self.mouse("up", (5, 5))
        self.assertEqual((self.ui.focus, self.roll.picked), (None, {0}))
        # On the keyboard, on the key of g4: the key is held down until the button is let go.
        key = (20, self.at(0, 67)[1])
        self.assertTrue(self.mouse("down", key))
        self.assertEqual((self.ui.focus, self.roll.busy, self.ui.help, self.heard, self.roll.keys.down),
                         (self.roll.keys, False, f"press: hear g4{WHEEL}", [(67, "lead")], 67))
        self.mouse("move", self.at(4, 70))
        self.mouse("up", self.at(4, 70))
        self.assertEqual((self.roll.notes, self.roll.picked, self.edits, self.heard), (TUNE, {0}, [], [(67, "lead")]))
        self.assertIsNone(self.roll.keys.down)
        # The right button on a key is not used, and neither is a press over the keyboard, beside the ruler.
        self.assertEqual([self.mouse("down", key, 3), self.mouse("up", key, 3), self.mouse("down", (20, 25))],
                         [False, False, False])
        self.assertEqual(len(self.heard), 1)
        # With the focus on the keyboard the keys of the computer are the roll's, and then its owner's.
        self.mouse("down", key)
        self.mouse("up", key)
        typed: list[str] = []
        self.roll.typed = lambda event: typed.append(event.key) or True
        self.assertEqual([self.ui.handle(Event("key", key=key)) for key in ("x", "delete")], [True, True])
        self.assertEqual((typed, self.edits), (["x"], [(TUNE[1:], "note removed")]))

    def test_a_placed_note_is_not_heard_where_the_owner_answers_that_it_was_not_saved(self) -> None:
        self.roll.edited = lambda notes, label: self.edits.append((notes, label)) or False
        self.drag(self.at(20, 62))
        self.assertEqual((self.edits, self.heard, self.roll.notes),
                         ([((*TUNE, note(20, 4, 62)), "note placed")], [], TUNE))
        # Any other answer is taken for saved, as no answer is.
        self.roll.edited = lambda notes, label: True
        self.drag(self.at(20, 62))
        self.assertEqual(self.heard, [(62, "lead")])

    def test_the_roll_is_busy_from_the_press_that_begins_a_gesture_until_it_is_released_or_dropped(self) -> None:
        self.assertFalse(self.roll.busy)
        self.mouse("down", self.at(14, 60))
        self.mouse("move", self.at(16, 62))
        self.assertTrue(self.roll.busy)
        self.mouse("up", self.at(16, 62))
        self.assertEqual((self.roll.busy, len(self.edits)), (False, 1))
        self.mouse("down", self.at(20, 70), 1, "shift")
        self.assertTrue(self.roll.busy)
        self.key("escape")
        self.assertFalse(self.roll.busy)

    def test_ctrl_and_c_copies_the_picked_notes_and_ctrl_and_v_pastes_them_at_the_row_under_the_mouse(self) -> None:
        self.drag(self.at(0, 64))
        self.drag(self.at(4, 65), mods=("ctrl",))
        self.assertTrue(self.key("c", "ctrl"))
        self.mouse("move", self.at(20, 70))
        self.assertTrue(self.key("v", "ctrl"))
        pasted = (*TUNE, note(20, 4, 64), note(24, 2, 65))
        self.assertEqual((self.edits, self.roll.picked), ([(pasted, "2 notes pasted")], {4, 5}))
        # With nothing picked, ctrl and c keeps what was copied.
        self.key("escape")
        self.key("c", "ctrl")
        # The mouse is on the keys, off the grid, whose first row in view is row 4: f4 and g4 begin in what is pasted.
        self.ui.handle(Event("wheel", (162, 100), wheel=-1, mods=frozenset({"shift"})))
        self.mouse("move", (20, 100))
        self.key("v", "ctrl")
        self.assertEqual(self.roll.row, 4.0)
        self.assertEqual(self.edits[1:], [((TUNE[0], note(4, 4, 64), note(8, 2, 65), *pasted[3:]), "2 notes pasted")])
        self.assertEqual(self.roll.picked, {1, 2})

    def test_escape_drops_a_gesture_that_is_under_way_and_its_release_changes_nothing(self) -> None:
        self.mouse("down", self.at(14, 60))
        self.mouse("move", self.at(16, 62))
        self.assertEqual(self.roll.drawn, (*TUNE[:3], note(16, 2, 62)))
        # No other key is the roll's while the gesture is under way.
        self.assertEqual((self.key("delete"), self.key("escape")), (False, True))
        self.assertEqual((self.roll.drawn, self.roll.picked), (TUNE, {3}))
        self.mouse("move", self.at(18, 64))
        self.mouse("up", self.at(18, 64))
        self.assertEqual((self.roll.notes, self.roll.picked, self.edits, self.heard), (TUNE, {3}, [], [(62, "lead")]))
        # The next press begins a gesture as any other, and escape with none under way drops what is picked.
        self.drag(self.at(14, 60), self.at(15, 60))
        self.assertEqual(len(self.edits), 1)
        self.key("escape")
        self.assertEqual(self.roll.picked, frozenset())

    def test_a_release_where_the_press_was_changes_nothing_and_nothing_is_called(self) -> None:
        for start in (self.at(14, 60), self.end(3)):
            self.drag(start, self.at(20, 66), start)
        # A drag that stays in the row and on the key of the press is no move.
        x, y = self.at(14, 60)
        self.drag((x, y), (x + 4, y + 5))
        self.assertEqual((self.roll.notes, self.roll.picked, self.edits), (TUNE, {3}, []))

    def test_a_click_in_which_the_mouse_strays_two_pixels_is_no_drag_and_three_pixels_are_one(self) -> None:
        self.assertEqual(SLACK, 2)
        # From the last pixel of row 14 and of c4 into row 15 and on to b3: the note is picked, and not moved.
        x, y = self.at(14, 60)
        self.drag((x + 4, y + 5), (x + 6, y + 7))
        self.assertEqual((self.roll.map.moved((x + 4, y + 5), (x + 6, y + 7)), self.roll.picked, self.edits),
                         ((1.0, -1), {3}, []))
        # A note is placed as long as a click places it. With shift, from the line under it onto its bar, no box
        # is drawn that would touch it: three pixels on, one is.
        x, y = self.at(20, 62)
        self.drag((x, y), (x - 2, y + 2))
        self.drag((x, y + 6), (x + 2, y + 4), mods=("shift",))
        self.assertEqual((self.edits, self.roll.picked), ([((*TUNE, note(20, 4, 62)), "note placed")], frozenset()))
        self.drag((x, y + 6), (x + 3, y + 4), mods=("shift",))
        self.assertEqual(self.roll.picked, {4})
        # Three pixels on in the same row, the note that is placed on e4 is one step of the grid long.
        self.drag((x + 18, y - 22), (x + 21, y - 22))
        self.assertEqual(self.edits[1:], [((*TUNE, note(20, 2, 62), note(22, 1, 64)), "note placed")])

    def test_what_is_drawn_during_a_drag_is_what_its_release_gives(self) -> None:
        # A move, a stretch, a note placed by a drag, and three notes removed.
        for stops, button in (((self.at(14, 60), self.at(3, 66)), 1), ((self.end(0), self.at(5, 60)), 1),
                              ((self.at(2, 62), self.at(5, 62)), 1), ((self.at(1, 64), self.at(10, 67)), 3)):
            with self.subTest(stops=stops):
                self.show(TUNE)
                self.edits.clear()
                self.mouse("down", stops[0], button)
                self.mouse("move", stops[1], button)
                drawn = self.roll.drawn
                self.assertNotEqual(drawn, TUNE)
                self.assertEqual((self.roll.notes, self.edits), (TUNE, []))
                self.mouse("up", stops[1], button)
                self.assertEqual(([notes for notes, _ in self.edits], self.roll.drawn), ([drawn], drawn))

    def test_the_picked_stay_picked_when_the_notes_of_the_edit_are_shown_and_not_when_others_are(self) -> None:
        # An owner that does not show the roll the edit at once: the roll goes on showing what it was given.
        self.roll.edited = lambda notes, label: self.edits.append((notes, label))
        self.drag(self.at(14, 60), self.at(3, 66))
        moved = self.edits[0][0]
        self.assertEqual((moved[1], self.roll.notes, self.roll.picked), (note(3, 2, 66), TUNE, {3}))
        self.show(TUNE)
        self.assertEqual(self.roll.picked, {3})
        for _ in range(2):
            self.show(moved)
            self.assertEqual((self.roll.notes, self.roll.picked), (moved, {1}))
        # The file changed outside: other notes are shown than the next edit gave. By then, what it gave is as
        # any other notes.
        self.drag(self.at(8, 67), self.at(8, 70))
        self.assertEqual((len(self.edits), self.roll.picked), (2, {2}))
        self.show(TUNE)
        self.assertEqual(self.roll.picked, frozenset())
        self.drag(self.at(0, 64))
        self.show(self.edits[1][0])
        self.assertEqual(self.roll.picked, frozenset())
        # Another channel that has the notes that are shown is not the channel they were picked in.
        self.drag(self.at(0, 64))
        self.assertEqual(self.roll.picked, {0})
        self.roll.show(Pattern("p", 32.0, tuple(Channel(called, "lead", False, self.roll.notes, 32.0, True)
                                                for called in ("lead", "bass"))), "bass", 16, theme.SONG)
        self.assertEqual((self.roll.notes, self.roll.picked), (self.edits[1][0], frozenset()))

    def test_a_note_that_is_moved_is_heard_once_at_each_pitch_it_comes_to(self) -> None:
        # e4 and the g4 that plays with pluck are picked, and the press is on g4: that is the note that is heard.
        self.show(LEAD)
        self.drag(self.at(0, 64), mods=("ctrl",))
        self.drag(self.at(8, 67), mods=("ctrl",))
        x, y = self.at(8, 67)
        self.mouse("down", (x, y))
        # g4 is from y 85 to 95: 5 pixels up is a-4, 19 are a4, and 18 pixels on are two rows.
        for right, up in ((2, 0), (9, -1), (9, 5), (10, 6), (18, 8), (18, 19), (18, 8), (18, 0)):
            self.mouse("move", (x + right, y - up))
        self.mouse("up", (x, y))
        self.assertEqual(self.heard, [(68, "pluck"), (69, "pluck"), (68, "pluck"), (67, "pluck")])
        self.assertEqual((self.edits, self.roll.picked), ([], {0, 2}))

    def test_a_pattern_that_is_not_on_the_grid_is_shown_and_not_changed_and_the_help_line_says_why(self) -> None:
        triplets = (note(0, 8 / 3, 62), note(4, 4, 64))
        self.show(triplets)
        for pos, button, mods in ((self.at(4, 64), 1, ()), (self.at(4, 64), 1, ("ctrl",)), (self.at(10, 62), 1, ()),
                                  (self.at(10, 62), 1, ("shift",)), (self.at(4, 64), 3, ()), ((90, 240), 1, ()),
                                  ((90, 240), 3, ())):
            self.assertFalse(self.mouse("down", pos, button, *mods))
            self.mouse("move", self.at(12, 66), button, *mods)
            self.mouse("up", self.at(12, 66), button, *mods)
        self.assertEqual([self.key(key, *mods) for key, mods in (("a", ("ctrl",)), ("delete", ()), ("escape", ()))],
                         [False] * 3)
        self.assertEqual((self.roll.drawn, self.roll.picked, self.edits, self.heard), (triplets, frozenset(), [], []))
        # The keys of the keyboard can be heard all the same, and over the keyboard a press does nothing.
        for pos, line in ((self.at(4, 64), OFF_GRID), (self.at(10, 62), OFF_GRID), ((90, 240), OFF_GRID),
                          ((20, 100), f"press: hear g-4{WHEEL}"), ((20, 25), "the notes of a channel")):
            self.mouse("move", pos)
            self.assertEqual(self.ui.help, line)
        self.assertEqual(OFF_GRID, "this pattern has lengths the grid does not hold: it is shown and not changed")

    def test_the_wheel_is_used_while_no_gesture_is_under_way_and_not_during_one(self) -> None:
        def wheel(*mods: str) -> None:
            self.ui.handle(Event("wheel", (50, 100), wheel=1, mods=frozenset(mods)))

        self.drag(self.at(0, 64))
        for mods in ((), ("ctrl",), ("shift",)):
            wheel(*mods)
        view = (self.roll.row, self.roll.top, self.roll.row_w)
        self.assertEqual((view, self.roll.picked), ((-3.6, 74.0, 10), {0}))
        self.mouse("down", self.at(2, 64))
        for mods in ((), ("ctrl",), ("shift",)):
            wheel(*mods)
        self.assertEqual((self.roll.row, self.roll.top, self.roll.row_w), view)
        self.mouse("up", self.at(2, 64))
        wheel()
        self.assertEqual((self.roll.top, self.edits), (76.0, []))

    def test_other_notes_shown_during_a_gesture_drop_it_and_the_same_notes_do_not(self) -> None:
        self.mouse("down", self.at(14, 60))
        self.mouse("move", self.at(16, 62))
        self.show(TUNE)
        self.assertEqual(self.roll.drawn, (*TUNE[:3], note(16, 2, 62)))
        other = (*TUNE[:3], note(14, 2, 60, 9))
        self.show(other)
        self.assertEqual((self.roll.drawn, self.roll.picked), (other, frozenset()))
        self.mouse("up", self.at(16, 62))
        self.assertEqual((self.roll.notes, self.edits), (other, []))


@unittest.skipUnless(pygame, NEEDS)
class Drawn(unittest.TestCase):
    def setUp(self) -> None:
        self.surface = pygame.Surface(SIZE)
        self.surface.fill(OUTSIDE)
        self.canvas = Canvas(self.surface, Font(sprites.load("font")))
        self.ui = Ui()
        self.roll = rolled(self.ui)
        self.roll.show(pattern(), "lead", 16, theme.SONG)

    def at(self, *pos: int) -> tuple[int, ...]:
        """The colour of one pixel."""
        return tuple(self.surface.get_at(pos))[:3]

    def colours(self, rect: layout.Rect) -> set[tuple[int, ...]]:
        """The colours a rectangle has in it."""
        x, y, w, h = rect
        return {self.at(x + column, y + row) for column in range(w) for row in range(h)}

    def where(self, colour: tuple[int, ...], rect: layout.Rect) -> set[tuple[int, int]]:
        """The pixels of a rectangle that are in a colour."""
        x, y, w, h = rect
        return {(x + column, y + row) for column in range(w) for row in range(h)
                if self.at(x + column, y + row) == colour}

    def draw(self) -> Map:
        """Draw the roll, and give its sums."""
        self.ui.draw(self.canvas)
        return self.roll.map

    def test_the_bar_of_a_note_is_in_the_colour_at_its_place_and_nowhere_else(self) -> None:
        # One note too short for its name: its bar is (87, 118, 8, 10). It is lit along its upper edge and its
        # start, and dark along its lower edge and its end.
        self.roll.show(pattern((note(4, 1, 64),)), "lead", 16, theme.SONG)
        self.roll.ghosts = False
        seen = self.draw()
        inner = {(x, y) for x in range(88, 94) for y in range(119, 127)}
        self.assertEqual(self.where(theme.SONG, seen.grid), inner)
        self.assertEqual(self.where(LIGHT, seen.grid),
                         {(x, 118) for x in range(87, 94)} | {(87, y) for y in range(118, 128)})
        self.assertEqual(self.where(DARK, seen.grid),
                         {(x, 127) for x in range(88, 95)} | {(94, y) for y in range(118, 128)})
        # The line of its row and the line under its key are left as they were.
        self.assertEqual((self.at(86, 120), self.at(90, 128)), (theme.LINE, theme.PANEL))

    def test_two_notes_of_the_same_pitch_in_a_row_are_two_and_a_wide_one_has_its_name_on_it(self) -> None:
        self.draw()
        # The first ends dark at x 85, the line of row 4 is at 86, and the second starts lit at 87.
        self.assertEqual([self.at(x, 120) for x in (84, 85, 86, 87, 88)],
                         [theme.SONG, DARK, theme.LINE, LIGHT, theme.SONG])
        # Both have room for their name, in dark ink from 3 pixels in. The note of one row has not.
        self.assertEqual([self.colours((x, 119, 11, 7)) for x in (54, 90)], [{theme.SONG, theme.BACKGROUND}] * 2)
        self.assertEqual((self.roll.map.note(14, 1, 60), self.colours((178, 163, 6, 8))),
                         ((177, 162, 8, 10), {theme.SONG}))

    def test_the_row_of_a_black_key_is_darker_than_that_of_a_white_one_and_the_grid_ends_with_the_pattern(
            self) -> None:
        seen = self.draw()
        # d-4 and d4, a pixel after the line of row 1, and the line under each of them.
        dark, light = self.at(60, seen.y(61) + 3), self.at(60, seen.y(62) + 3)
        self.assertEqual((dark, light), (theme.BACKGROUND, theme.RAISED))
        self.assertLess(sum(dark), sum(light))
        self.assertEqual({self.at(60, seen.y(pitch) + 10) for pitch in (61, 62)}, {theme.PANEL})
        # Two rows before the pattern are in view: they are bare, up to the line of row 0.
        self.roll.row = -2.0
        self.draw()
        self.assertEqual((self.colours((50, 30, 18, 192)), self.at(68, 33)), ({theme.PANEL}, theme.DIM))

    def test_a_bar_line_is_stronger_than_a_beat_line_and_that_than_the_line_of_a_row(self) -> None:
        seen = self.draw()
        for pitch, ground in ((62, theme.RAISED), (61, theme.BACKGROUND)):
            y = seen.y(pitch) + 3
            row, beat, bar = (strength(self.at(seen.x(at), y), ground) for at in (1, 4, 16))
            self.assertTrue(0 < row < beat < bar)
            self.assertEqual((self.at(seen.x(0), y), self.at(seen.x(8), y)),
                             (self.at(seen.x(16), y), self.at(seen.x(4), y)))
        # The bars are numbered over their lines, from 1, and nothing else is on the ruler.
        inked = {x for x, _ in self.where(theme.DIM, seen.ruler)}
        self.assertEqual((self.colours(seen.ruler), min(inked), max(inked)), ({theme.PANEL, theme.DIM}, 52, 200))
        self.assertEqual({x for x in inked if x > 56}, set(range(196, 201)))
        # Rows too narrow for a line each keep the lines of their beats and bars.
        self.roll.row_w = 4
        seen = self.draw()
        y = seen.y(62) + 3
        self.assertEqual([self.at(seen.x(at), y) for at in (1, 4, 16)], [theme.RAISED, theme.LINE, theme.DIM])

    def test_a_ghost_is_dimmer_than_a_note_lies_behind_the_notes_and_is_gone_with_ghosts_off(self) -> None:
        self.draw()
        # The arp holds e4 for rows 0 to 8, under the two notes of the lead: it shows after them, and on the line
        # between them.
        ghost = mix(theme.SOUND, theme.BACKGROUND, GHOST)
        self.assertEqual([self.at(x, 122) for x in (80, 86, 110)], [theme.SONG, ghost, ghost])
        self.assertEqual((self.at(110, 120), self.at(110, 125)), (theme.RAISED, theme.RAISED))
        self.assertLess(sum(ghost), min(sum(theme.SOUND), sum(theme.SONG)))
        # A drum is no ghost: the hit on row 2 has the number of d4.
        self.assertEqual(self.colours((69, 140, 8, 10)), {theme.RAISED})
        self.roll.ghosts = False
        seen = self.draw()
        self.assertEqual((self.at(110, 122), self.where(ghost, seen.grid)), (theme.RAISED, set()))

    def test_the_lane_has_a_bar_for_each_note_as_high_as_its_level(self) -> None:
        seen = self.draw()
        self.assertEqual((self.roll.lane.rect, self.at(10, 228), self.at(314, 261)), (LANE, theme.LINE, theme.LINE))
        for found in LEAD:
            x = seen.x(found.row) + 1
            self.assertEqual(self.where(theme.SONG, (x, 228, 4, 34)),
                             {(x + column, 259 - row) for column in range(3) for row in range(2 * found.level)})
        self.assertEqual(len(self.where(theme.SONG, LANE)), 3 * 2 * (12 + 15 + 8))
        # Its name is dim at its left end, and from where the grid begins the lane has its bars and nothing else.
        self.assertEqual((self.colours((70, 228, 245, 34)), self.colours((11, 229, 39, 32))),
                         ({theme.PANEL, theme.LINE, theme.SONG}, {theme.PANEL, theme.DIM}))
        # A picked note has its bar in the colour of text.
        for kind in ("down", "up"):
            self.ui.handle(Event(kind, (60, 120), button=1))
        self.draw()
        self.assertEqual((self.roll.picked, self.where(theme.TEXT, LANE)),
                         ({0}, {(51 + column, 259 - row) for column in range(3) for row in range(24)}))
        # With the rows scrolled on by two, that bar would stand under the keys: it is not drawn, and the others
        # are under their notes.
        self.roll.row = 2.0
        seen = self.draw()
        self.assertEqual((seen.x(0), self.where(theme.TEXT, LANE), self.at(seen.x(4) + 1, 259)),
                         (32, set(), theme.SONG))

    def test_the_playhead_is_a_line_down_the_grid_and_the_lane_at_its_row(self) -> None:
        self.roll.playhead = 9.5
        seen = self.draw()
        # The scroll bar under the grid lies over it, and so does the pixel of air between the roll and the lane.
        self.assertEqual((seen.x(9.5), self.where(theme.ACTIVE, (0, 0, *SIZE))),
                         (136, {(136, y) for y in (*range(20, 223), *range(229, 261))}))
        # None is no line, and neither is a row that is not in view.
        for playhead in (None, -1.0, 40.0):
            self.roll.playhead = playhead
            self.draw()
            self.assertEqual(self.where(theme.ACTIVE, (0, 0, *SIZE)), set())

    def test_a_note_with_another_instrument_than_its_channels_has_a_notch_at_its_start(self) -> None:
        seen = self.draw()
        # g4 on row 8 plays with pluck: its bar starts at (123, 85).
        self.assertEqual(self.where(theme.INSTRUMENT, seen.grid),
                         {(123, 85), (124, 85), (125, 85), (123, 86), (124, 86), (123, 87)})
        self.assertEqual((self.at(126, 85), self.at(123, 88), self.at(51, 118)), (LIGHT, LIGHT, LIGHT))

    def test_a_picked_note_has_a_bright_outline_in_place_of_its_light_and_its_dark_edges(self) -> None:
        self.roll.show(pattern((note(4, 1, 64), note(8, 2, 64))), "lead", 16, theme.SONG)
        self.roll.ghosts = False
        for kind in ("down", "up"):
            self.ui.handle(Event(kind, (90, 120), button=1))
        seen = self.draw()
        # The bar of the first note is (87, 118, 8, 10): its outline is the pixels along its four sides.
        bar = {(x, y) for x in range(87, 95) for y in range(118, 128)}
        inner = {(x, y) for x in range(88, 94) for y in range(119, 127)}
        self.assertEqual((self.roll.picked, self.where(theme.TEXT, seen.grid)), ({0}, bar - inner))
        self.assertEqual(self.where(theme.SONG, (87, 118, 8, 10)), inner)
        # The other note, from x 123 to 139, has the edges it had.
        self.assertEqual((self.at(123, 118), self.at(139, 127)), (LIGHT, DARK))

    def test_the_note_that_a_drag_would_place_is_drawn_during_the_drag_and_is_gone_when_it_is_dropped(self) -> None:
        self.roll.show(pattern((note(4, 1, 64),)), "lead", 16, theme.SONG)
        self.roll.ghosts = False
        # Pressed on d4 in row 8 and dragged into row 9: its bar would be (123, 140, 17, 10).
        self.ui.handle(Event("down", (125, 145), button=1))
        self.ui.handle(Event("move", (134, 150)))
        seen = self.draw()
        self.assertEqual((self.roll.notes, self.roll.drawn), ((note(4, 1, 64),), (note(4, 1, 64), note(8, 2, 62))))
        self.assertEqual((seen.note(8, 2, 62), self.at(123, 140), self.at(139, 149), self.at(124, 145)),
                         ((123, 140, 17, 10), LIGHT, DARK, theme.SONG))
        self.assertEqual(self.where(theme.TEXT, seen.grid), set())
        self.roll.key(Event("key", key="escape"))
        self.draw()
        self.assertEqual(self.colours((123, 140, 17, 10)), {theme.RAISED, theme.PANEL})

    def test_a_box_is_drawn_while_it_is_dragged_and_the_notes_it_touches_are_outlined(self) -> None:
        self.roll.ghosts = False
        shift = frozenset({"shift"})
        # From row 2 of g-4 to row 10 of e-4: the two e4 sound in it, and the g4 is over it.
        self.ui.handle(Event("down", (70, 100), button=1, mods=shift))
        self.ui.handle(Event("move", (140, 130), mods=shift))
        self.draw()
        self.assertEqual([self.at(*pos) for pos in ((70, 100), (140, 100), (70, 130), (140, 130), (105, 100),
                                                    (105, 130), (70, 110), (140, 110))], [theme.TEXT] * 8)
        self.assertNotIn(theme.TEXT, self.colours((71, 101, 69, 16)) | self.colours((141, 99, 20, 33)))
        # The outline of the first e4 is bright, and the g4 has its light edge.
        self.assertEqual((self.roll.picked, self.at(51, 118), self.at(85, 127), self.at(126, 85)),
                         (frozenset(), theme.TEXT, theme.TEXT, LIGHT))
        self.ui.handle(Event("up", (140, 130), button=1, mods=shift))
        self.draw()
        self.assertEqual((self.roll.picked, self.at(140, 100), self.at(105, 100), self.at(51, 118)),
                         ({0, 1}, theme.PANEL, theme.BACKGROUND, theme.TEXT))

    def test_the_keys_are_white_and_black_as_on_a_piano_and_each_c_is_named(self) -> None:
        seen = self.draw()
        d, d_flat, c = seen.y(62), seen.y(61), seen.y(60)
        self.assertEqual([self.at(12, d + 3), self.at(12, d_flat + 3), self.at(40, d_flat + 3)],
                         [theme.TEXT, theme.BACKGROUND, theme.TEXT])
        # The name is dark ink at the right end of the key, and no other white key has one.
        self.assertIn(theme.BACKGROUND, self.colours((34, c + 2, 11, 7)))
        self.assertEqual(self.colours((11, d + 1, 37, 9)), {theme.TEXT})
        # The keys end a pixel short of the grid.
        self.assertEqual((self.colours((10, 30, 39, 192)), self.colours((49, 30, 1, 192))),
                         ({theme.TEXT, theme.BACKGROUND}, {theme.PANEL}))
        # A key that is held down is in the colour of what is on, the black part of a black key too.
        for pos, held in (((12, d + 3), 62), ((12, d_flat + 3), 61)):
            self.ui.handle(Event("down", pos, button=1))
            self.draw()
            self.assertEqual((self.roll.keys.down, self.at(12, pos[1]), self.at(40, pos[1])),
                             (held, theme.ACTIVE, theme.ACTIVE))
            self.ui.handle(Event("up", pos, button=1))

    def test_each_scroll_bar_has_a_thumb_for_what_is_in_view_on_a_track_as_dark_as_what_lies_behind(self) -> None:
        seen = self.draw()
        along, across = self.roll.along.thumb(), self.roll.across.thumb()
        self.assertEqual((along, across), ((311, 124, 4, 28), (115, 223, 117, 4)))
        for bar, thumb in ((seen.along, along), (seen.across, across)):
            self.assertEqual((self.colours(bar), self.colours(thumb)), ({theme.BACKGROUND, theme.LINE}, {theme.LINE}))
            self.assertEqual(len(self.where(theme.LINE, bar)), thumb[2] * thumb[3])
        # Under the mouse the thumb is lighter.
        self.ui.handle(Event("move", (312, 130)))
        self.draw()
        self.assertEqual((self.colours(along), self.colours(across)), ({theme.DIM}, {theme.LINE}))

    def test_nothing_is_drawn_outside_the_rectangle_nor_by_a_part_outside_its_own(self) -> None:
        self.roll.top += 0.5
        self.roll.row = -2.5
        self.draw()
        around = ((0, 0, 325, 20), (0, 262, 325, 8), (0, 20, 10, 242), (315, 20, 10, 242), (0, 227, 325, 1))
        self.assertEqual(set().union(*(self.colours(rect) for rect in around)), {OUTSIDE})
        self.assertNotIn(OUTSIDE, self.colours(RECT) | self.colours(LANE))
        # Half a key is cut off at the upper edge of the keys and of the grid: over them is the panel, as it is
        # between the grid and each of its two scroll bars.
        self.assertEqual((self.roll.map.y(72), self.at(40, 30), self.at(40, 29)), (36, theme.TEXT, theme.PANEL))
        self.assertEqual(self.colours((10, 20, 40, 10)) | self.colours((10, 222, 305, 1))
                         | self.colours((310, 20, 1, 207)), {theme.PANEL})
        self.roll.playhead = 9.5
        self.draw()
        self.assertEqual(set().union(*(self.colours(rect) for rect in around)), {OUTSIDE})

    def test_the_lead_of_the_arcade_tune_is_shown_note_for_note_over_the_ghosts_of_its_arp(self) -> None:
        song = Project.load(EXAMPLES / "arcade.json").song("tune")
        surface = pygame.Surface((665, 300))
        roll = Roll((0, 0, 665, 267))
        roll.show(songview.pattern(song, "a"), "lead", songview.bar_rows(song), theme.SONG)
        roll.draw(Canvas(surface, Font(sprites.load("font"))))
        self.surface = surface
        seen = roll.map
        # All of its 4 bars are in view, and its notes from e5 to e6 with room over and under them.
        self.assertEqual((roll.row, roll.top, seen.x(64) < 660, len(roll.notes)), (0.0, 92.0, True, 19))
        bars = [seen.note(found.row, found.rows, found.pitch) for found in roll.notes]
        allowed = {(x + column, y + row) for x, y, w, h in bars for column in range(w) for row in range(h)}
        painted = self.where(theme.SONG, seen.grid)
        self.assertLessEqual(painted, allowed)
        self.assertEqual([(x + 1, y + 1) in painted for x, y, _, _ in bars], [True] * 19)
        # The arp plays c5 on every other row, and the snare, whose number is that of c5, is no ghost on row 4.
        ghost = mix(theme.SOUND, theme.BACKGROUND, GHOST)
        y = seen.y(72) + 4
        self.assertEqual([self.at(seen.x(row) + 2, y) for row in (1, 3, 4, 5)], [ghost, ghost, theme.RAISED, ghost])
        self.assertEqual(self.at(seen.x(64), y), theme.DIM)


class Strip(unittest.TestCase):
    def test_the_strip_of_the_chords_lies_under_the_ruler_and_a_roll_with_no_chords_is_as_it_was(self) -> None:
        seen = Map(RECT, strip=CHORDS_H)
        self.assertEqual((seen.ruler, seen.chords, seen.keys, seen.grid, seen.along, seen.across),
                         ((50, 20, 260, 10), (50, 30, 260, 13), (10, 43, 40, 179), (50, 43, 260, 179),
                          (311, 43, 4, 179), (50, 223, 260, 4)))
        self.assertEqual((Map(RECT).chords, Map(RECT).grid), ((50, 30, 260, 0), (50, 30, 260, 192)))
        roll = Roll(RECT)
        roll.show(pattern(TUNE), "lead", 16, theme.SONG)
        bare = roll.map
        roll.show(pattern(TUNE), "lead", 16, theme.SONG, chords("Am"))
        self.assertEqual((roll.map.strip, roll.map.grid, roll.keys.rect, roll.along.rect),
                         (CHORDS_H, (50, 43, 260, 179), (10, 43, 40, 179), (311, 43, 4, 179)))
        roll.show(pattern(TUNE), "lead", 16, theme.SONG)
        self.assertEqual((roll.chords, roll.map, bare.strip), (None, bare, 0))

    def test_a_press_on_a_bar_of_the_strip_tells_the_owner_of_the_bar_and_begins_nothing(self) -> None:
        ui = Ui()
        asked: list[tuple[int, tuple[int, int]]] = []
        roll = rolled(ui, "the notes of a channel", chord_asked=lambda bar, pos: asked.append((bar, pos)))
        roll.show(pattern(TUNE), "lead", 16, theme.SONG, chords("Am"))
        # The second bar starts at x 194. The ruler over the strip is nobody's, and the right button asks nothing.
        used = []
        for pos, button in (((60, 35), 1), ((200, 42), 1), ((200, 29), 1), ((200, 35), 3)):
            used.append(ui.handle(Event("down", pos, button=button)))
            ui.handle(Event("up", pos, button=button))
        self.assertEqual((used, asked), ([True, True, False, False], [(0, (60, 35)), (1, (200, 42))]))
        self.assertEqual((roll.busy, roll.picked, roll.notes), (False, frozenset(), TUNE))
        ui.handle(Event("move", (200, 35)))
        self.assertEqual(ui.help, f"press: pick the chord of bar 2{WHEEL}")
        # Past the last bar of the pattern the strip has no bar, and a roll with no chords has no strip.
        roll.row = 20.0
        self.assertFalse(ui.handle(Event("down", (200, 35), button=1)))
        roll.row = 0.0
        roll.show(pattern(TUNE), "lead", 16, theme.SONG)
        ui.handle(Event("down", (60, 35), button=1))
        self.assertEqual(len(asked), 2)


@unittest.skipUnless(pygame, NEEDS)
class Chorded(unittest.TestCase):
    def setUp(self) -> None:
        self.surface = pygame.Surface(SIZE)
        self.canvas = Canvas(self.surface, Font(sprites.load("font")))
        self.ui = Ui()
        self.edits: list[str] = []
        self.roll = rolled(self.ui, "the notes of a channel", lambda notes, label: self.edits.append(label))
        self.roll.ghosts = False

    def at(self, *pos: int) -> tuple[int, ...]:
        """The colour of one pixel."""
        return tuple(self.surface.get_at(pos))[:3]

    def where(self, colour: tuple[int, ...], rect: layout.Rect) -> set[tuple[int, int]]:
        """The pixels of a rectangle that are in a colour."""
        x, y, w, h = rect
        return {(x + column, y + row) for column in range(w) for row in range(h)
                if self.at(x + column, y + row) == colour}

    def draw(self) -> Map:
        """Draw the roll on a surface with nothing on it, and give its sums."""
        self.surface.fill(OUTSIDE)
        self.ui.draw(self.canvas)
        return self.roll.map

    def red(self) -> set[tuple[int, int]]:
        """Every pixel that is in the colour of a fault."""
        return self.where(theme.FAULT, RECT)

    def ring(self, found: Note) -> set[tuple[int, int]]:
        """The pixels of the outline round the bar of a note: on the lines of the grid on its four sides."""
        x, y, w, h = self.roll.map.note(found.row, found.rows, found.pitch)
        return {(x - 1 + column, y - 1 + row) for column in range(w + 2) for row in range(h + 2)
                if column in (0, w + 1) or row in (0, h + 1)}

    def test_a_bar_has_its_chord_named_in_the_strip_and_the_rows_of_its_notes_lit_as_far_as_it_goes(self) -> None:
        self.roll.show(pattern(()), "lead", 16, theme.SONG, chords("Am"))
        seen = self.draw()
        # Each bar has a box from its line to a pixel before the next, and a line of air under it. The first has
        # its name 4 pixels in, and the second, which has no chord, is empty.
        named = self.where(theme.ACTIVE, RECT)
        self.assertTrue(named and named <= {(x, y) for x in range(54, 65) for y in range(32, 41)})
        self.assertEqual([self.at(*pos) for pos in ((50, 30), (192, 41), (193, 36), (194, 30), (120, 36), (250, 36),
                                                    (120, 42))],
                         [theme.LINE, theme.LINE, theme.PANEL, theme.LINE, theme.RAISED, theme.RAISED, theme.PANEL])
        self.assertTrue(self.where(theme.DIM, (10, 30, 40, 12)))
        # In the first bar the rows of a, c and e are a shade lighter, in every octave, up to the line of the bar.
        lit = mix(theme.RAISED, theme.TEXT, CHORD_LIT)
        whole = seen.pitches()[1:-1]
        self.assertEqual([pitch for pitch in whole if self.at(seen.x(2) + 3, seen.y(pitch) + 3) == lit],
                         [52, 57, 60, 64])
        self.assertEqual({self.at(seen.x(18) + 3, seen.y(pitch) + 3) for pitch in whole if not black(pitch)},
                         {theme.RAISED})
        y = seen.y(64)
        self.assertEqual([self.at(seen.x(16) + by, y + 3) for by in (-1, 0, 1)], [lit, theme.DIM, theme.RAISED])
        self.assertEqual((self.at(seen.x(2) + 3, y + 10), self.at(seen.x(4), y + 3)), (theme.PANEL, theme.LINE))

    def test_a_note_the_check_would_find_has_a_red_outline_and_the_help_line_says_why(self) -> None:
        self.roll.show(pattern(TUNE), "lead", 16, theme.SONG, chords("Am"))
        seen = self.draw()
        # Under a minor e4 is right, f4 and g4 start beats and are not, and c4 starts between two beats.
        self.assertEqual(self.red(), self.ring(TUNE[1]) | self.ring(TUNE[2]))
        lines = []
        for found in TUNE[:2]:
            self.ui.handle(Event("move", (seen.x(found.row) + 4, seen.y(found.pitch) + 4)))
            lines.append(self.ui.help)
        self.assertEqual(lines, [f"press: pick e4, a drag moves it   right press: rub it out{WHEEL}",
                                 "f4 is not in Am   press: pick f4, a drag moves it   right press: rub it out"])
        # Nothing is red under a string that names no chord, nor in a channel that ends before its pattern does,
        # which the check does not look at, nor in a roll that is shown no chords: that one is as it was.
        short = Pattern("p", 32.0, (Channel("lead", "lead", False, TUNE, 16.0, True),
                                    Channel("arp", "pluck", False, (note(0, 32, 50),), 32.0, True)))
        for shown, written in ((pattern(TUNE), chords("", "Am")), (short, chords("Am")), (pattern(TUNE), None)):
            self.roll.show(shown, "lead", 16, theme.SONG, written)
            self.draw()
            self.assertEqual(self.red(), set())
        self.assertEqual((self.roll.map.grid, self.where(theme.ACTIVE, RECT)), ((50, 30, 260, 192), set()))

    def test_a_note_dragged_off_its_chord_is_red_before_it_is_let_go_and_so_is_one_that_is_being_placed(self) -> None:
        # The lead ends before its pattern, so none of it is red until a gesture would save it, and fill its rows.
        short = Pattern("p", 32.0, (Channel("lead", "lead", False, TUNE, 16.0, True),
                                    Channel("arp", "pluck", False, (note(0, 32, 50),), 32.0, True)))
        self.roll.show(short, "lead", 16, theme.SONG, chords("Am"))
        seen = self.draw()
        x, y = seen.x(1) + 4, seen.y(64) + 4
        self.ui.handle(Event("down", (x, y), button=1))
        self.draw()
        self.assertEqual(self.red(), set())
        # e4 goes up a key to f4, which is not in the chord, and back.
        self.ui.handle(Event("move", (x, y - KEY_H)))
        self.draw()
        self.assertEqual(self.red(), self.ring(note(0, 4, 65)) | self.ring(TUNE[1]) | self.ring(TUNE[2]))
        self.ui.handle(Event("move", (x, y)))
        self.draw()
        self.ui.handle(Event("up", (x, y), button=1))
        self.assertEqual((self.red(), self.edits), (set(), []))
        # A note that a press would place on the fourth beat at d4 is red while the button is held.
        self.roll.show(pattern(TUNE[:1]), "lead", 16, theme.SONG, chords("Am"))
        seen = self.draw()
        self.ui.handle(Event("down", (seen.x(12) + 4, seen.y(62) + 4), button=1))
        self.draw()
        self.assertEqual((self.roll.drawn[-1], self.red()), (note(12, 4, 62), self.ring(note(12, 4, 62))))
        self.ui.handle(Event("key", key="escape"))
        self.draw()
        self.assertEqual((self.red(), self.edits), (set(), []))


if __name__ == "__main__":
    unittest.main()
