"""Tests for rowsynth.gui.roll: the sums of the piano roll, its view under the wheel, and what it draws."""
from __future__ import annotations

import math
import os
import unittest
from pathlib import Path

from rowsynth.errors import RowSynthError
from rowsynth.gui import layout, songview, sprites, theme
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.font import Font
from rowsynth.gui.roll import (DARKER, GHOST, HIGHEST, KEY_H, LIGHTER, LOWEST, ROW_W_MAX, ROW_W_MIN, Map, Roll, black,
                               mix, name)
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

#: A roll whose grid is (50, 30, 260, 192), under a ruler from y 20 and over a lane (50, 226, 260, 34).
RECT = (10, 20, 300, 240)
#: The surface it is drawn on, which is larger than it on every side, and the colour of what is not the roll's.
SIZE = (320, 270)
OUTSIDE = theme.GOOD

#: The colours of the upper edge and the start of a note in the colour of a song, and of its lower edge and end.
LIGHT = mix(theme.SONG, theme.TEXT, LIGHTER)
DARK = mix(theme.SONG, theme.BACKGROUND, DARKER)


def note(row: float, rows: float, pitch: int, level: int = 15, inst: str = "lead") -> Note:
    """A note that is no drum hit."""
    return Note(float(row), float(rows), pitch, level, inst)


#: Two notes of the same pitch in a row, one that plays with another instrument, and a short low one of no level.
LEAD = (note(0, 4, 64, 12), note(4, 2, 64), note(8, 4, 67, 8, "pluck"), note(14, 1, 60, 0))


def pattern(lead: tuple[Note, ...] = LEAD, called: str = "p", rows: float = 32.0) -> Pattern:
    """A pattern of four channels: the lead, an arp that lies under its first notes, drums, and one with no text."""
    return Pattern(called, rows, (
        Channel("lead", "lead", False, lead, rows, True),
        Channel("arp", "pluck", False, (note(0, 8, 64), note(8, 8, 76)), rows, True),
        Channel("kit", "kick", True, (Note(2.0, 1.0, 62.0, 15, "hat", "h"),), rows, True),
        Channel("pad", "pad", False, (), 0.0, False)))


def strength(colour: tuple[int, ...], ground: tuple[int, ...]) -> int:
    """How far a colour is from the one it lies on."""
    return sum(abs(one - other) for one, other in zip(colour, ground))


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

    def test_the_bar_of_a_level_stands_on_the_floor_of_the_lane_two_pixels_high_for_each_level(self) -> None:
        seen = Map(RECT)
        self.assertEqual([seen.level(4, 2, level) for level in (0, 1, 8, 15)],
                         [(87, 258, 4, 0), (87, 256, 4, 2), (87, 242, 4, 16), (87, 228, 4, 30)])
        # It is no wider than the bar of its note.
        self.assertEqual(Map(RECT, row_w=2).level(4, 1, 15), (59, 228, 1, 30))

    def test_the_parts_share_the_rectangle_and_in_view_is_what_has_a_pixel_on_the_grid(self) -> None:
        seen = Map(RECT)
        self.assertEqual((seen.keys, seen.ruler, seen.grid, seen.lane),
                         ((10, 30, 40, 192), (50, 20, 260, 10), (50, 30, 260, 192), (50, 226, 260, 34)))
        # 260 pixels are 28 rows and most of another, and 192 are 17 keys and nearly half of another.
        self.assertEqual((seen.rows(), seen.pitches()), (range(0, 29), range(55, 73)))
        moved = Map(RECT, row=-2.5, top=80.5, row_w=4)
        self.assertEqual((moved.rows(), moved.pitches()), (range(-3, 63), range(64, 82)))
        for found in (seen, moved):
            self.assertEqual((found.x(found.rows()[0]) <= 50, found.x(found.rows()[-1]) < 310), (True, True))
            self.assertEqual((found.y(found.pitches()[-1]) <= 30, found.y(found.pitches()[0]) < 222), (True, True))
        # A rectangle too small for the keys, the ruler and the lane leaves no grid, and nothing is in view.
        small = Map((0, 0, 30, 40))
        self.assertEqual((small.grid, len(small.rows()), len(small.pitches())), ((40, 10, 0, 0), 0, 0))

    def test_a_mixed_colour_lies_between_the_two_it_is_mixed_from(self) -> None:
        self.assertEqual([mix((0, 0, 0), (200, 100, 50)), mix((0, 0, 0), (200, 100, 50), 0.25)],
                         [(100, 50, 25), (50, 25, 13)])
        self.assertEqual((mix(theme.SONG, theme.TEXT, 0), mix(theme.SONG, theme.TEXT, 1)), (theme.SONG, theme.TEXT))


class View(unittest.TestCase):
    def setUp(self) -> None:
        self.ui = Ui()
        self.roll = self.ui.add(Roll(RECT, "the notes of a channel"))
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
        # Another channel, and then another pattern: fitted, at the zoom there was.
        self.roll.show(pattern(), "arp", 16, theme.SOUND)
        self.assertEqual((self.view(), self.roll.colour), ((0.0, 78.0, 11), theme.SOUND))
        self.wheel(3)
        self.roll.show(pattern(called="q"), "arp", 16, theme.SOUND)
        self.assertEqual(self.view(), (0.0, 78.0, 11))
        self.wheel(3)
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

    def test_the_help_line_says_what_note_or_what_place_of_the_grid_the_mouse_is_on(self) -> None:
        self.assertEqual((self.roll.help, self.ui.help), ("the notes of a channel", ""))
        for pos, line in (((60, 120), "e4, 4 rows, level 12"),
                          ((178, 165), "c4, 1 row, level 0"),
                          ((130, 90), "g4, 4 rows, level 8, instrument pluck"),
                          ((233, 87), "g4, bar 2, row 5"),
                          ((86, 120), "e4, bar 1, row 5"),
                          ((20, 90), "the notes of a channel"),
                          ((60, 240), "the notes of a channel")):
            self.ui.handle(Event("move", pos))
            self.assertEqual(self.ui.help, line)
        # A length that is no whole number of rows, and a place before the pattern.
        self.roll.show(pattern((note(0, 8 / 3, 62),)), "lead", 16, theme.SONG)
        self.ui.handle(Event("move", (60, 142)))
        self.assertEqual(self.ui.help, "d4, 2.67 rows, level 15")
        self.wheel(1, "shift", pos=(55, 87))
        self.assertEqual((self.roll.row, self.ui.help), (-4.0, "the notes of a channel"))
        self.roll.help = "another line"
        self.assertEqual(self.ui.help, "another line")


@unittest.skipUnless(pygame, NEEDS)
class Drawn(unittest.TestCase):
    def setUp(self) -> None:
        self.surface = pygame.Surface(SIZE)
        self.surface.fill(OUTSIDE)
        self.canvas = Canvas(self.surface, Font(sprites.load("font")))
        self.ui = Ui()
        self.roll = self.ui.add(Roll(RECT))
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
        self.assertEqual((seen.lane, self.at(50, 226), self.at(309, 259)),
                         ((50, 226, 260, 34), theme.LINE, theme.LINE))
        for found in LEAD:
            x = seen.x(found.row) + 1
            self.assertEqual(self.where(theme.SONG, (x, 226, 4, 34)),
                             {(x + column, 257 - row) for column in range(4) for row in range(2 * found.level)})
        self.assertEqual(len(self.where(theme.SONG, seen.lane)), 4 * 2 * (12 + 15 + 8))
        self.assertEqual(self.colours(seen.lane), {theme.BACKGROUND, theme.LINE, theme.SONG})
        self.assertIn(theme.DIM, self.colours((10, 226, 40, 34)))

    def test_the_playhead_is_a_line_down_the_grid_and_the_lane_at_its_row(self) -> None:
        self.roll.playhead = 9.5
        seen = self.draw()
        self.assertEqual((seen.x(9.5), self.where(theme.ACTIVE, (0, 0, *SIZE))),
                         (136, {(136, y) for y in range(20, 260)}))
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

    def test_nothing_is_drawn_outside_the_rectangle_nor_by_a_part_outside_its_own(self) -> None:
        self.roll.top += 0.5
        self.roll.row = -2.5
        self.draw()
        around = ((0, 0, 320, 20), (0, 260, 320, 10), (0, 20, 10, 240), (310, 20, 10, 240))
        self.assertEqual(set().union(*(self.colours(rect) for rect in around)), {OUTSIDE})
        self.assertNotIn(OUTSIDE, self.colours(RECT))
        # Half a key is cut off at the upper edge of the keys and of the grid: over them is the panel, as it is
        # between the grid and the lane.
        self.assertEqual((self.roll.map.y(72), self.at(40, 30), self.at(40, 29)), (36, theme.TEXT, theme.PANEL))
        self.assertEqual(self.colours((10, 20, 40, 10)) | self.colours((10, 222, 300, 4)), {theme.PANEL})
        self.roll.playhead = 9.5
        self.draw()
        self.assertEqual(set().union(*(self.colours(rect) for rect in around)), {OUTSIDE})

    def test_the_lead_of_the_arcade_tune_is_shown_note_for_note_over_the_ghosts_of_its_arp(self) -> None:
        song = Project.load(EXAMPLES / "arcade.json").song("tune")
        surface = pygame.Surface((660, 300))
        roll = Roll((0, 0, 660, 300))
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


if __name__ == "__main__":
    unittest.main()
