"""Tests for rowsynth.gui.steps: the lanes of a song, the sums that place the steps, the widget, and what it draws."""
from __future__ import annotations

import dataclasses
import os
import unittest
from pathlib import Path

from rowsynth.errors import RowSynthError
from rowsynth.gui import layout, songview, sprites, theme
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.font import Font
from rowsynth.gui.songview import Channel, DrumRow, Note, Pattern
from rowsynth.gui.steps import (BAR_GAP, BARS_H, BEAT_GAP, CHANNEL_GAP, EDGE, EMPTY_FILL, EMPTY_LINE, FAINTEST, GAP,
                                LABEL_EDGE, LABEL_GAP, LABEL_PAD, LABELS_W, LANE_GAP, LANE_H, MARK_W, SQUARE_W, Grid,
                                Lane, Steps, hit_colour, lanes, mix, paint)
from rowsynth.gui.ui import Event, Ui
from rowsynth.project import Project

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
SHIFT = frozenset({"shift"})

# A rectangle that cuts the third lane and the twenty-fifth row, and one high enough for three whole lanes.
RECT = (10, 20, 296, 100)
TALL = (10, 20, 296, 130)
# Where the squares start in both, how wide their room is, and where the rectangles end.
LEFT, TOP, AREA_W = RECT[0] + LABELS_W, RECT[1] + BARS_H, RECT[2] - LABELS_W
RIGHT, BOTTOM = RECT[0] + RECT[2], RECT[1] + RECT[3]
PITCH = SQUARE_W + GAP
LANE = LANE_H + LANE_GAP
# The height of each of the two lines of a label.
LINE_H = (LANE_H - 2 * LABEL_EDGE) // 2
# From the left of a bar of sixteen rows, four to a beat, to the left of the next.
BAR_W = 16 * PITCH + 3 * (BEAT_GAP - GAP) + BAR_GAP - GAP

DRUMS = [DrumRow("k", "kick", 48.0), DrumRow("s", "a very long snare", 72.0), DrumRow("H", "hat", 96.0)]
# The lanes of one drum channel that plays all three, and those of two: one that plays a letter, and one two.
LANES = [Lane("kit", "k", "kick"), Lane("kit", "s", "a very long snare"), Lane("kit", "H", "hat")]
PAIR = [Lane("kick", "k", "kick"), Lane("tops", "k", "kick"), Lane("tops", "H", "hat")]


def hit(letter: str, row: float, level: int = 15) -> Note:
    """A hit of a drum letter that starts at a row."""
    return Note(row, 1.0, 60.0, level, "kick", letter)


def band(rows: float = 32.0, **hits: tuple[Note, ...]) -> Pattern:
    """A pattern with a tuned channel, and a drum channel of each name with its hits. Two bars unless told."""
    return Pattern("a", rows, (Channel("lead", "lead", False, (Note(0.0, 4.0, 60.0, 15, "lead"),), rows, True),
                               *(Channel(name, "kick", True, notes, rows, True) for name, notes in hits.items())))


def kit(*notes: Note, rows: float = 32.0) -> Pattern:
    """A pattern with a tuned channel and one drum channel, which has those hits. It is two bars unless told."""
    return band(rows, kit=notes)


def shown(*notes: Note, rect: layout.Rect = RECT) -> Steps:
    """The steps showing one drum channel with those hits, sixteen rows to a bar and four to a beat."""
    steps = Steps(rect, help="the drum steps")
    steps.show(kit(*notes), LANES, 16, 4)
    return steps


def paired(rect: layout.Rect = TALL) -> Steps:
    """The steps showing two drum channels that play one letter at other rows, and one of them a letter more."""
    steps = Steps(rect, help="the drum steps")
    steps.show(band(kick=(hit("k", 0), hit("k", 9.5)), tops=(hit("k", 4, 9), hit("H", 2))), PAIR, 16, 4)
    return steps


def scroll(steps: Steps, beats: int = 0, drums: int = 0) -> None:
    """Turn the wheel on the steps: so many beats of rows along, and so many drums down."""
    steps.on(Event("wheel", (LEFT, TOP), wheel=-beats, mods=SHIFT), Ui())
    steps.on(Event("wheel", (LEFT, TOP), wheel=-drums), Ui())


def total(colour: tuple[int, ...]) -> int:
    """How bright a colour is: its red, its green and its blue added up."""
    return sum(colour)


def block(rect: layout.Rect) -> set[tuple[int, int]]:
    """Every pixel of a rectangle."""
    x, y, w, h = rect
    return {(x + column, y + row) for column in range(w) for row in range(h)}


class Sums(unittest.TestCase):
    def test_the_x_of_a_row_has_the_gaps_between_beats_and_between_bars_in_it(self) -> None:
        grid = Grid(RECT, 32, 3, 16, 4)
        self.assertEqual(grid.area, (LEFT, TOP, AREA_W, RECT[3] - BARS_H))
        self.assertEqual([grid.x(row) - LEFT for row in (0, 1, 3, 4, 8, 15, 16, 20)],
                         [0, PITCH, 3 * PITCH, 4 * PITCH + BEAT_GAP - GAP, 8 * PITCH + 2 * (BEAT_GAP - GAP),
                          BAR_W - BAR_GAP - SQUARE_W, BAR_W, BAR_W + 4 * PITCH + BEAT_GAP - GAP])
        # Between two rows: that share of the way from one square to the next in its beat.
        self.assertEqual((grid.x(9.5), grid.x(9.75)), (grid.x(9) + PITCH // 2, grid.x(9) + 3 * PITCH // 4))
        # With the view moved, its first row is where row 0 was, and the others as far from it as before.
        moved = dataclasses.replace(grid, first_row=6)
        self.assertEqual((moved.x(6), moved.x(16)), (LEFT, LEFT + grid.x(16) - grid.x(6)))

    def test_the_row_at_an_x_is_the_row_whose_square_it_is_on_and_a_gap_is_no_row(self) -> None:
        for bar_rows, beat_rows, first in ((16, 4, 0), (16, 4, 5), (12, 3, 0), (10, 3, 7), (4, 8, 2)):
            grid = Grid((0, 0, 2000, 60), 40, 1, bar_rows, beat_rows, first)
            with self.subTest(bar_rows=bar_rows, beat_rows=beat_rows, first=first):
                self.assertEqual(grid.rows_in_view(), range(first, 40))
                on = {x: row for row in grid.rows_in_view() for x in range(grid.x(row), grid.x(row) + SQUARE_W)}
                along = range(grid.x(first), grid.x(39) + 60)
                self.assertEqual({x: grid.row_at(x) for x in along}, {x: on.get(x) for x in along})
                # What lies between two squares is the gap of its kind.
                gaps = {grid.x(row + 1) - grid.x(row) - SQUARE_W for row in range(first, 39)}
                self.assertEqual(gaps, {GAP, BAR_GAP} | ({BEAT_GAP} if beat_rows < bar_rows else set()))
        # Left of the squares there is no row, nor right of the rectangle, where the last square in view is cut.
        grid = Grid(RECT, 32, 3, 16, 4)
        self.assertEqual([grid.row_at(x) for x in (LEFT - 1, LEFT, RIGHT - 1, RIGHT)], [None, 0, 24, None])

    def test_the_y_of_a_drum_the_drum_at_a_y_and_the_square_of_a_step(self) -> None:
        grid = Grid(RECT, 32, 3, 16, 4)
        self.assertEqual([grid.y(drum) for drum in range(3)], [TOP, TOP + LANE, TOP + 2 * LANE])
        self.assertEqual([grid.drum_at(y) for y in (TOP - 1, TOP, TOP + LANE_H - 1, TOP + LANE_H, TOP + LANE)],
                         [None, 0, 0, None, 1])
        # The third lane is cut by the rectangle, and under the rectangle there is none.
        self.assertEqual([grid.drum_at(y) for y in (grid.y(2), BOTTOM - 1, BOTTOM)], [2, 2, None])
        self.assertEqual(grid.square(0, 0), (LEFT, TOP, SQUARE_W, LANE_H))
        self.assertEqual(grid.square(2, 17), (LEFT + BAR_W + PITCH, TOP + 2 * LANE, SQUARE_W, LANE_H))
        moved = dataclasses.replace(grid, first_row=16, first_drum=1)
        self.assertEqual(moved.square(1, 17), (LEFT + PITCH, TOP, SQUARE_W, LANE_H))
        self.assertEqual([moved.drum_at(y) for y in (TOP, moved.y(2), moved.y(3))], [1, 2, None])

    def test_the_rows_and_the_drums_in_view_are_those_that_show_at_least_in_part(self) -> None:
        # A bar and nine rows more, of which the last shows only its first columns.
        grid = Grid(RECT, 32, 3, 16, 4)
        self.assertEqual((grid.rows_in_view(), grid.drums_in_view()), (range(0, 25), range(0, 3)))
        self.assertLess(grid.x(24), RIGHT)
        self.assertGreater(grid.x(24) + SQUARE_W, RIGHT)
        moved = dataclasses.replace(grid, first_row=20, first_drum=2)
        self.assertEqual((moved.rows_in_view(), moved.drums_in_view()), (range(20, 32), range(2, 3)))
        short = Grid((0, 0, 400, BARS_H + LANE_H), 5, 2, 16, 4)
        self.assertEqual((short.rows_in_view(), short.drums_in_view()), (range(0, 5), range(0, 1)))
        # A rectangle with no room for a square shows none.
        none = Grid((0, 0, LABELS_W, BARS_H), 32, 3, 16, 4)
        self.assertEqual((none.area, len(none.rows_in_view()), len(none.drums_in_view())),
                         ((LABELS_W, BARS_H, 0, 0), 0, 0))

    def test_the_rows_of_two_bars_take_the_four_colours_by_the_beat_at_4_rows_a_beat_and_at_3(self) -> None:
        for bar_rows, beat_rows in ((16, 4), (12, 3)):
            grid = Grid(RECT, 2 * bar_rows, 1, bar_rows, beat_rows)
            self.assertEqual([grid.colour(row) for row in range(2 * bar_rows)],
                             2 * [colour for colour in theme.STEPS for _ in range(beat_rows)])
        # A bar of more than four beats goes round to the first colour, and each bar starts on it.
        odd = Grid(RECT, 32, 1, 16, 3)
        self.assertEqual([theme.STEPS.index(odd.colour(row)) for row in (0, 2, 3, 9, 11, 12, 14, 15, 16, 19)],
                         [0, 0, 1, 3, 3, 0, 0, 1, 0, 1])

    def test_the_view_is_kept_inside_the_pattern(self) -> None:
        grid = Grid(RECT, 32, 3, 16, 4)
        moved = grid.scrolled(rows=4, drums=1)
        self.assertEqual((moved.first_row, moved.first_drum), (4, 1))
        self.assertEqual(moved.scrolled(rows=-9, drums=-9), grid)
        # The furthest: where the last row, and the last drum, is whole in view and one row less would not be.
        end = grid.scrolled(rows=99, drums=99)
        self.assertEqual((end.first_row, end.first_drum), (8, 1))
        self.assertLessEqual(end.x(31) + SQUARE_W, RIGHT)
        self.assertGreater(dataclasses.replace(end, first_row=7).x(31) + SQUARE_W, RIGHT)
        self.assertLessEqual(end.y(2) + LANE_H, BOTTOM)
        # What fits whole does not move, and neither does a pattern with nothing in it.
        fits = Grid((0, 0, 400, 200), 32, 3, 16, 4)
        self.assertEqual(fits.scrolled(rows=5, drums=2), fits)
        self.assertEqual(Grid(RECT, 0, 0, 16, 4).scrolled(rows=3, drums=3), Grid(RECT, 0, 0, 16, 4))

    def test_the_first_lane_of_another_channel_has_more_air_above_it_which_no_lane_is_in(self) -> None:
        grid = Grid((10, 20, 296, 300), 32, 4, 16, 4, breaks=(1, 3))
        self.assertEqual([grid.y(drum) - TOP for drum in range(4)],
                         [0, LANE + CHANNEL_GAP, 2 * LANE + CHANNEL_GAP, 3 * LANE + 2 * CHANNEL_GAP])
        self.assertEqual([grid.drum_at(y) for y in (grid.y(0) + LANE_H, grid.y(1) - 1, grid.y(1), grid.y(2) - 1,
                                                    grid.y(3) - 1, grid.y(3) + LANE_H - 1, grid.y(3) + LANE_H)],
                         [None, None, 1, None, None, 3, None])
        self.assertEqual(grid.square(3, 0), (LEFT, TOP + 3 * LANE + 2 * CHANNEL_GAP, SQUARE_W, LANE_H))
        # The air above the lane at the top of the view is out of view with the lanes above it.
        moved = dataclasses.replace(grid, first_drum=1)
        self.assertEqual([moved.y(drum) - TOP for drum in (1, 2, 3)], [0, LANE, 2 * LANE + CHANNEL_GAP])
        # The furthest the view goes counts the air: the last three lanes fit this room with it, and not one less.
        room = BARS_H + 2 * LANE + CHANNEL_GAP + LANE_H
        for height, breaks, first in ((room, (1, 3), 1), (room - 1, (1, 3), 2), (room - 1, (), 1)):
            short = Grid((10, 20, 296, height), 32, 4, 16, 4, breaks=breaks)
            self.assertEqual(short.scrolled(drums=99).first_drum, first)

    def test_a_bar_or_a_beat_of_no_rows_is_refused(self) -> None:
        for bar_rows, beat_rows in ((0, 4), (16, 0)):
            with self.assertRaises(RowSynthError) as refused:
                Grid(RECT, 32, 3, bar_rows, beat_rows)
            self.assertEqual(str(refused.exception),
                             f"a bar and a beat are 1 row or more each, not {bar_rows} and {beat_rows}")

    def test_a_hit_is_brighter_the_higher_its_level_and_at_no_level_as_dim_as_an_empty_square(self) -> None:
        self.assertEqual([mix((0, 100, 200), (100, 50, 201), share) for share in (0, 0.5, 1)],
                         [(0, 100, 200), (50, 75, 200), (100, 50, 201)])
        for colour in theme.STEPS:
            with self.subTest(colour=colour):
                fills = [hit_colour(colour, level) for level in (0, 1, 8, 15)]
                self.assertEqual((fills[0], fills[3]), (mix(theme.BACKGROUND, colour, FAINTEST), colour))
                self.assertTrue(all(total(low) < total(high) for low, high in zip(fills, fills[1:])))
                # An empty square is a dim outline round a fill dimmer still, and the faintest hit is above both.
                fill, line = paint(colour, None)
                self.assertEqual((fill, line), (mix(theme.BACKGROUND, colour, EMPTY_FILL),
                                                mix(theme.BACKGROUND, colour, EMPTY_LINE)))
                self.assertTrue(all(low < middle < high for low, middle, high in zip(fill, line, fills[0])))
                # A hit is filled as its level says, inside a lighter outline, and what is lit is lighter.
                self.assertEqual(paint(colour, 8), (fills[2], mix(fills[2], theme.TEXT, EDGE)))
                for level in (None, 8):
                    self.assertTrue(all(total(lit) > total(plain) for lit, plain
                                        in zip(paint(colour, level, True), paint(colour, level))))


class Lanes(unittest.TestCase):
    def test_one_drum_channel_that_plays_three_letters_has_a_lane_for_each_in_the_order_of_the_table(self) -> None:
        first, second = kit(hit("H", 0), hit("k", 4), Note(8.0, 1.0, 60.0, 15, "kick")), kit(hit("s", 2), hit("k", 0))
        self.assertEqual(lanes([first, second], DRUMS), LANES)
        # The patterns can be had one at a time, and a note that is no drum letter has no lane.
        self.assertEqual(lanes(iter([first]), DRUMS), [LANES[0], LANES[2]])

    def test_three_drum_channels_of_one_letter_each_have_a_lane_each_in_the_order_of_the_song(self) -> None:
        found = lanes([band(hats=(hit("H", 0), hit("H", 2)), kick=(hit("k", 0),), snare=(hit("s", 4),))], DRUMS)
        self.assertEqual(found, [Lane("hats", "H", "hat"), Lane("kick", "k", "kick"),
                                 Lane("snare", "s", "a very long snare")])

    def test_a_channel_that_plays_two_letters_has_its_two_lanes_together_among_the_others(self) -> None:
        # What a channel plays in any pattern counts, and two channels can play the same letter.
        found = lanes([band(kick=(hit("k", 0),), tops=(hit("H", 2),), snare=()),
                       band(kick=(), tops=(hit("k", 4),), snare=(hit("s", 4),))], DRUMS)
        self.assertEqual(found, PAIR + [Lane("snare", "s", "a very long snare")])

    def test_a_drum_channel_that_plays_nothing_anywhere_has_a_lane_for_every_letter(self) -> None:
        found = lanes([band(kick=(hit("k", 0),), spare=()), band(kick=(), spare=())], DRUMS)
        self.assertEqual(found, [Lane("kick", "k", "kick"), *(Lane("spare", drum.letter, drum.inst) for drum in DRUMS)])
        # A song with no drum channel has no lanes, and neither has one with no patterns or with no letters.
        tuned = Pattern("a", 16.0, (Channel("lead", "lead", False, (), 16.0, True),))
        self.assertEqual([lanes([tuned], DRUMS), lanes([], DRUMS), lanes([kit()], [])], [[], [], []])


class Pointing(unittest.TestCase):
    def setUp(self) -> None:
        self.ui = Ui()
        self.steps = self.ui.add(shown(hit("k", 0), hit("k", 20, 12), hit("H", 5.5, 9)))
        self.grid = self.steps.grid

    def wheel(self, steps: int, mods: frozenset[str] = frozenset(),
              pos: tuple[int, int] = (LEFT + 5, TOP + 5)) -> bool:
        """Turn the wheel with the mouse on the squares, or where it is told. Whether it was used."""
        return self.ui.handle(Event("wheel", pos, wheel=steps, mods=mods))

    def view(self) -> tuple[int, int]:
        """The first row and the first drum in view."""
        return self.steps.grid.first_row, self.steps.grid.first_drum

    def test_the_wheel_scrolls_the_drums_and_with_shift_the_rows_by_a_beat(self) -> None:
        self.assertEqual(self.view(), (0, 0))
        self.assertTrue(self.wheel(-1))
        self.assertEqual(self.view(), (0, 1))
        self.assertTrue(self.wheel(-1, SHIFT))
        self.assertEqual(self.view(), (4, 1))
        self.assertTrue(self.wheel(1))
        self.assertTrue(self.wheel(1, SHIFT))
        self.assertEqual(self.view(), (0, 0))
        # On the labels as well, and nowhere off the widget.
        self.assertTrue(self.wheel(-1, pos=(RECT[0] + 2, TOP + 5)))
        self.assertFalse(self.wheel(-1, SHIFT, pos=(RECT[0] - 1, TOP + 5)))
        self.assertEqual(self.view(), (0, 1))

    def test_the_view_stops_at_the_ends_of_the_pattern_and_a_step_that_moves_nothing_is_not_used(self) -> None:
        self.assertFalse(self.wheel(1))
        self.assertFalse(self.wheel(1, SHIFT))
        self.assertTrue(self.wheel(-5))
        self.assertTrue(self.wheel(-5, SHIFT))
        self.assertEqual(self.view(), (8, 1))
        self.assertFalse(self.wheel(-1))
        self.assertFalse(self.wheel(-1, SHIFT))
        # A wider and higher rectangle has all of it in view, and the view comes back to the start.
        self.steps.rect = (0, 0, 400, 200)
        self.assertEqual(self.view(), (0, 0))
        self.assertFalse(self.wheel(-1, pos=(LABELS_W + 5, BARS_H + 5)))
        self.assertFalse(self.wheel(-1, SHIFT, pos=(LABELS_W + 5, BARS_H + 5)))

    def test_showing_another_pattern_keeps_the_view_as_far_as_that_one_reaches(self) -> None:
        self.wheel(-1, SHIFT)
        self.steps.show(kit(hit("k", 0)), LANES, 16, 4)
        self.assertEqual(self.view(), (4, 0))
        self.steps.show(kit(hit("k", 0), rows=16.0), LANES[:1], 16, 4)
        self.assertEqual((self.view(), self.steps.grid.rows, self.steps.grid.drums), ((0, 0), 16, 1))
        # A count of rows a hair off a whole one is that whole one, and half a row more is a square more.
        for rows, squares in ((15.9999999, 16), (16.0000001, 16), (16.5, 17)):
            self.steps.show(kit(rows=rows), LANES, 16, 4)
            self.assertEqual(self.steps.grid.rows, squares)

    def test_step_at_is_the_channel_the_letter_and_the_row_on_a_square_and_none_in_a_gap_and_in_the_labels(
            self) -> None:
        x, y, w, h = self.grid.square(1, 5)
        self.assertEqual([self.steps.step_at(pos) for pos in ((x, y), (x + w - 1, y + h - 1), (LEFT, TOP),
                                                             (x, self.grid.y(2)))],
                         [("kit", "s", 5), ("kit", "s", 5), ("kit", "k", 0), ("kit", "H", 5)])
        # Right of the square and above it, in the strip of the bar numbers, on a label and just before the squares.
        self.assertEqual([self.steps.step_at(pos) for pos in ((x + w, y), (x, y - 1), (x, RECT[1] + 2),
                                                             (RECT[0] + 5, y + 5), (LEFT - 1, y))],
                         [None, None, None, None, None])
        # A hit a hair off a whole row is on that row, and after a scroll the same place is another step.
        self.steps.show(kit(hit("s", 4.9999999, 7)), LANES, 16, 4)
        self.ui.handle(Event("move", (x, y)))
        self.assertEqual((self.steps.step_at((x, y)), self.ui.help),
                         (("kit", "s", 5), "s, bar 1, step 6: hit, level 7"))
        self.wheel(-1)
        self.wheel(-1, SHIFT)
        self.assertEqual(self.steps.step_at((LEFT, TOP)), ("kit", "s", 4))

    def test_the_help_line_says_what_is_at_the_square_under_the_mouse(self) -> None:
        self.assertEqual(self.ui.help, "")
        for drum, row, said in ((0, 20, "k, bar 2, step 5: hit, level 12"), (1, 20, "s, bar 2, step 5: empty"),
                                (0, 0, "k, bar 1, step 1: hit, level 15"), (2, 15, "H, bar 1, step 16: empty"),
                                (2, 5, "H, bar 1, step 6: hit off the grid, level 9")):
            x, y, _, _ = self.grid.square(drum, row)
            self.ui.handle(Event("move", (x + 2, y + 9)))
            self.assertEqual(self.ui.help, said)
        # In a gap and on a label it is the widget's own line, which can be set, and a key leaves it be.
        for pos in ((self.grid.x(4) - 1, TOP + 3), (RECT[0] + 5, TOP + 3)):
            self.ui.handle(Event("move", pos))
            self.assertEqual(self.ui.help, "the drum steps")
        self.steps.help = "drums"
        self.assertFalse(self.ui.handle(Event("key", key="space")))
        self.assertEqual(self.ui.help, "drums")
        # After a scroll the mouse is on another step.
        self.ui.handle(Event("move", (LEFT + 2, TOP + 3)))
        self.wheel(-1, SHIFT, pos=(LEFT + 2, TOP + 3))
        self.assertEqual(self.ui.help, "k, bar 1, step 5: empty")

    def test_with_two_drum_channels_step_at_and_the_help_line_say_which_channel_a_square_is_of(self) -> None:
        steps = self.ui.add(paired((0, 200, 296, 130)))
        grid = steps.grid
        self.assertEqual((grid.drums, grid.breaks), (3, (1,)))
        for drum, row, step, said in ((0, 0, ("kick", "k", 0), "k in kick, bar 1, step 1: hit, level 15"),
                                      (1, 0, ("tops", "k", 0), "k in tops, bar 1, step 1: empty"),
                                      (1, 4, ("tops", "k", 4), "k in tops, bar 1, step 5: hit, level 9"),
                                      (0, 9, ("kick", "k", 9), "k in kick, bar 1, step 10: hit off the grid, level 15"),
                                      (1, 9, ("tops", "k", 9), "k in tops, bar 1, step 10: empty"),
                                      (2, 18, ("tops", "H", 18), "H in tops, bar 2, step 3: empty")):
            x, y, _, _ = grid.square(drum, row)
            self.ui.handle(Event("move", (x + 2, y + 9)))
            self.assertEqual((steps.step_at((x + 2, y + 9)), self.ui.help), (step, said))
        # In the air between the two channels there is no step, and the line is the widget's own.
        self.ui.handle(Event("move", (grid.x(0), grid.y(1) - 1)))
        self.assertEqual((steps.step_at((grid.x(0), grid.y(1) - 1)), self.ui.help), (None, "the drum steps"))

    def test_a_lane_of_a_channel_the_pattern_does_not_have_or_of_one_that_is_no_drum_channel_is_refused(self) -> None:
        for channel, said in (("bass", "pattern 'a' has no channel called 'bass'"),
                              ("lead", "channel 'lead' is not a drum channel")):
            with self.assertRaises(RowSynthError) as refused:
                self.steps.show(kit(), [LANES[0], Lane(channel, "k", "kick")], 16, 4)
            self.assertEqual(str(refused.exception), said)
        with self.assertRaises(RowSynthError):
            self.steps.show(kit(), LANES, 16, 0)
        # What it showed before is still there.
        self.ui.handle(Event("move", (LEFT, TOP)))
        self.assertEqual(self.ui.help, "k, bar 1, step 1: hit, level 15")


@unittest.skipUnless(pygame, NEEDS)
class Drawn(unittest.TestCase):
    SIZE = (320, 160)

    def setUp(self) -> None:
        self.surface = pygame.Surface(self.SIZE)
        self.canvas = Canvas(self.surface, Font(sprites.load("font")))

    def draw(self, steps: Steps) -> Grid:
        """Draw the steps on a clean panel, and give their sums."""
        self.surface.fill(theme.PANEL)
        steps.draw(self.canvas)
        return steps.grid

    def at(self, x: int, y: int) -> tuple[int, ...]:
        """The colour of one pixel."""
        return tuple(self.surface.get_at((x, y)))[:3]

    def inked(self, rect: layout.Rect, colour: tuple[int, ...]) -> set[tuple[int, int]]:
        """The pixels of a rectangle that are of a colour."""
        return {pos for pos in block(rect) if self.at(*pos) == colour}

    def written(self, text: str, rect: layout.Rect, colour: tuple[int, ...],
                align: str = "left") -> set[tuple[int, int]]:
        """The pixels of a rectangle that are of a colour when nothing but a text is drawn in it, in that colour."""
        kept, self.surface = self.surface, pygame.Surface(self.SIZE)
        Canvas(self.surface, self.canvas.font).text_in(text, rect, colour, align)
        found, self.surface = self.inked(rect, colour), kept
        return found

    def square(self, rect: layout.Rect) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """What a square is filled with and the colour of its outline, each of which is all of one colour."""
        x, y, w, h = rect
        inside = {self.at(*pos) for pos in block((x + 1, y + 1, w - 2, h - 2))}
        outline = {self.at(*pos) for pos in block(rect) - block((x + 1, y + 1, w - 2, h - 2))}
        self.assertEqual((len(inside), len(outline)), (1, 1))
        return inside.pop(), outline.pop()

    def test_a_hit_is_filled_with_the_colour_of_its_beat_and_an_empty_square_is_not(self) -> None:
        grid = self.draw(shown(hit("k", 0), hit("k", 6), hit("s", 9), hit("H", 14), hit("k", 16), rect=TALL))
        for drum, row, colour in ((0, 0, theme.FAULT), (0, 6, theme.ACTIVE), (1, 9, theme.MUSTARD),
                                  (2, 14, theme.TEXT), (0, 16, theme.FAULT)):
            self.assertEqual(self.square(grid.square(drum, row)), (colour, mix(colour, theme.TEXT, EDGE)))
        for drum, row, colour in ((1, 0, theme.FAULT), (0, 5, theme.ACTIVE), (0, 9, theme.MUSTARD),
                                  (1, 14, theme.TEXT), (2, 16, theme.FAULT)):
            self.assertEqual(self.square(grid.square(drum, row)), paint(colour, None))
            self.assertNotIn(colour, paint(colour, None))
        # Between two squares, two beats, two bars and two lanes there is what was there before.
        self.assertEqual({self.at(grid.x(1) - 1, TOP + 3), self.at(grid.x(4) - 1, TOP + 3),
                          self.at(grid.x(16) - 1, TOP + 3), self.at(LEFT + 2, TOP + LANE_H)}, {theme.PANEL})

    def test_a_louder_hit_is_brighter(self) -> None:
        grid = self.draw(shown(hit("k", 4, 0), hit("k", 5, 3), hit("k", 6, 12), hit("k", 7), rect=TALL))
        fills = [self.square(grid.square(0, row))[0] for row in (4, 5, 6, 7)]
        self.assertEqual(fills, [hit_colour(theme.ACTIVE, level) for level in (0, 3, 12, 15)])
        self.assertTrue(all(total(low) < total(high) for low, high in zip(fills, fills[1:])))
        self.assertGreater(total(fills[0]), total(self.square(grid.square(1, 4))[1]))

    def test_a_hit_that_starts_between_two_rows_is_a_thin_mark_at_its_place(self) -> None:
        steps = shown(hit("s", 9.5), hit("k", 9.5, 0), hit("H", 30.25), rect=TALL)
        grid = self.draw(steps)
        x, y, w, h = grid.square(1, 9)
        self.assertEqual(grid.x(9.5), x + PITCH // 2)
        self.assertEqual(self.inked((x, y, PITCH, h), theme.MUSTARD), block((x + PITCH // 2, y, MARK_W, LANE_H)))
        # The square it is on is empty all the same, and the mark of a quiet hit is dimmer.
        self.assertEqual((self.at(x + 2, y + 5), self.at(x, y)), paint(theme.MUSTARD, None))
        self.assertEqual(self.at(x + PITCH // 2, grid.y(0) + 5), hit_colour(theme.MUSTARD, 0))
        # One in a row that is out of view is not drawn, until its row is scrolled to.
        self.assertEqual(self.inked((LEFT, grid.y(2), AREA_W, LANE_H), theme.TEXT), set())
        scroll(steps, beats=9)
        grid = self.draw(steps)
        self.assertEqual(self.inked((LEFT, grid.y(2), AREA_W, LANE_H), theme.TEXT),
                         block((grid.x(30) + PITCH // 4, grid.y(2), MARK_W, LANE_H)))

    def test_a_label_is_the_letter_of_a_drum_and_under_it_the_name_of_its_instrument_cut_when_too_long(self) -> None:
        grid = self.draw(shown(rect=TALL))
        width = LABELS_W - LABEL_GAP
        for drum, letter, name in ((0, "k", "kick"), (1, "s", "a very long.."), (2, "H", "hat")):
            x, y = RECT[0], grid.y(drum)
            upper = (x + LABEL_PAD, y + LABEL_EDGE, width - 2 * LABEL_PAD, LINE_H)
            lower = (x + LABEL_PAD, y + LABEL_EDGE + LINE_H, width - 2 * LABEL_PAD, LINE_H)
            with self.subTest(letter=letter):
                self.assertEqual(self.inked((x, y, width, LANE_H), theme.TEXT),
                                 self.written(letter, upper, theme.TEXT))
                self.assertEqual(self.inked((x, y, width, LANE_H), theme.INSTRUMENT),
                                 self.written(name, lower, theme.INSTRUMENT))
                # A box with an outline, and free room between it and the first square.
                self.assertEqual((self.at(x, y), self.at(x + width - 1, y + LANE_H - 1), self.at(x + 1, y + 1)),
                                 (theme.LINE, theme.LINE, theme.RAISED))
                self.assertEqual(self.inked((x + width, y, LABEL_GAP, LANE_H), theme.PANEL),
                                 block((x + width, y, LABEL_GAP, LANE_H)))

    def test_with_two_drum_channels_a_label_has_the_name_of_its_channel_too_and_with_one_it_has_not(self) -> None:
        steps = paired()
        steps.show(band(kick=(), tops=(), a_long_channel=()), PAIR + [Lane("a_long_channel", "s", "snare")], 16, 4)
        grid = self.draw(steps)
        width = LABELS_W - LABEL_GAP
        for drum, letter, channel in ((0, "k", "kick"), (1, "k", "tops"), (2, "H", "tops")):
            x, y = TALL[0], grid.y(drum)
            upper = (x + LABEL_PAD, y + LABEL_EDGE, width - 2 * LABEL_PAD, LINE_H)
            beside = (x + 2 * LABEL_PAD + Font.width(letter), y + LABEL_EDGE,
                      width - 3 * LABEL_PAD - Font.width(letter), LINE_H)
            with self.subTest(drum=drum):
                # The letter and the instrument are where they are with one channel, and the name ends at the right.
                self.assertEqual(self.inked((x, y, width, LANE_H), theme.TEXT), self.written(letter, upper, theme.TEXT))
                self.assertEqual(self.inked((x, y, width, LANE_H), theme.DIM),
                                 self.written(channel, beside, theme.DIM, "right"))
                self.assertTrue(self.inked((x, y + LABEL_EDGE + LINE_H, width, LINE_H), theme.INSTRUMENT))
        # A name too long for the room beside the letter is cut, and stays clear of the letter.
        scroll(steps, drums=1)
        grid = self.draw(steps)
        x, y = TALL[0], grid.y(3)
        beside = (x + 2 * LABEL_PAD + Font.width("s"), y + LABEL_EDGE, width - 3 * LABEL_PAD - Font.width("s"), LINE_H)
        self.assertEqual(self.inked((x, y, width, LANE_H), theme.DIM),
                         self.written("a_long_ch..", beside, theme.DIM, "right"))
        # With one drum channel no label has a name in it.
        self.draw(shown(rect=TALL))
        self.assertEqual(self.inked((TALL[0], TOP, width, TALL[3] - BARS_H), theme.DIM), set())

    def test_the_labels_of_the_lanes_of_the_picked_channel_are_lit(self) -> None:
        steps = paired()
        for picked, lit in ((None, ()), ("tops", (1, 2)), ("kick", (0,)), ("lead", ())):
            steps.picked = picked
            grid = self.draw(steps)
            with self.subTest(picked=picked):
                self.assertEqual([self.at(TALL[0] + 1, grid.y(drum) + 1) for drum in range(3)],
                                 [theme.LIT if drum in lit else theme.RAISED for drum in range(3)])
                # The outline of a lit label is that of any label.
                self.assertEqual({self.at(TALL[0], grid.y(drum)) for drum in range(3)}, {theme.LINE})
        # The one drum channel of a song has all the labels.
        one = shown(rect=TALL)
        one.picked = "kit"
        grid = self.draw(one)
        self.assertEqual({self.at(TALL[0] + 1, grid.y(drum) + 1) for drum in range(3)}, {theme.LIT})

    def test_a_faint_line_runs_between_the_lanes_of_one_channel_and_those_of_the_next_and_none_inside_one(
            self) -> None:
        steps = paired(RECT)
        grid = self.draw(steps)
        self.assertEqual((grid.y(1) - grid.y(0) - LANE_H, grid.y(2) - grid.y(1) - LANE_H),
                         (LANE_GAP + CHANNEL_GAP, LANE_GAP))
        # In the middle of the air between the two channels, from side to side of the widget.
        line = (RECT[0], grid.y(0) + LANE_H + (LANE_GAP + CHANNEL_GAP) // 2, RECT[2], 1)
        self.assertEqual(self.inked(RECT, theme.LIT), block(line))
        between = (RECT[0], grid.y(1) + LANE_H, RECT[2], LANE_GAP)
        self.assertEqual(self.inked(between, theme.PANEL), block(between))
        # Scrolled so that the first lane of the second channel is at the top, the line has gone with the air.
        scroll(steps, drums=1)
        self.assertEqual(self.draw(steps).first_drum, 1)
        self.assertEqual(self.inked(RECT, theme.LIT), set())

    def test_a_hit_is_in_the_lane_of_its_own_channel_and_not_in_that_of_another_channel_with_its_letter(self) -> None:
        grid = self.draw(paired())
        # Both channels play k: the one at the first row and between two rows further on, the other at the fifth.
        self.assertEqual(self.square(grid.square(0, 0)), (theme.FAULT, mix(theme.FAULT, theme.TEXT, EDGE)))
        self.assertEqual(self.square(grid.square(1, 0)), paint(theme.FAULT, None))
        self.assertEqual(self.square(grid.square(1, 4))[0], hit_colour(theme.ACTIVE, 9))
        self.assertEqual(self.square(grid.square(0, 4)), paint(theme.ACTIVE, None))
        self.assertEqual(self.square(grid.square(2, 2))[0], theme.FAULT)
        self.assertEqual(self.inked((LEFT, TOP, AREA_W, TALL[3] - BARS_H), theme.MUSTARD),
                         block((grid.x(9.5), grid.y(0), MARK_W, LANE_H)))
        # A channel that has no lane shows nothing, in the lane of its letter that another channel has.
        steps = paired()
        steps.show(band(kick=(hit("k", 0),), tops=(hit("k", 4, 9), hit("H", 2))), PAIR[1:], 16, 4)
        grid = self.draw(steps)
        self.assertEqual([self.square(grid.square(0, row)) for row in (0, 1)], [paint(theme.FAULT, None)] * 2)
        self.assertEqual(self.square(grid.square(0, 4))[0], hit_colour(theme.ACTIVE, 9))

    def test_a_drum_with_no_hits_is_a_lane_of_empty_squares(self) -> None:
        grid = self.draw(shown(hit("k", 0), hit("H", 3), rect=TALL))
        whole = grid.rows_in_view()[:-1]
        self.assertEqual(len(whole), 24)
        self.assertEqual([self.square(grid.square(1, row)) for row in whole],
                         [paint(grid.colour(row), None) for row in whole])
        # A pattern with no rows has the labels and no squares, and no bar is numbered.
        bare = Steps(TALL)
        bare.show(kit(rows=0.0), LANES, 16, 4)
        self.draw(bare)
        room = (LEFT, TALL[1], AREA_W, TALL[3])
        self.assertEqual(self.inked(room, theme.PANEL), block(room))
        self.assertEqual(self.at(RECT[0], TOP), theme.LINE)

    def test_the_playhead_is_a_line_at_its_place_and_lights_the_squares_of_the_row_it_is_in(self) -> None:
        steps = shown(hit("k", 9), hit("H", 2), rect=TALL)
        steps.playhead = 9.5
        grid = self.draw(steps)
        left, top, _, _ = grid.square(0, 9)
        # From the top of the widget, through the bar numbers and between the lanes, to the foot of the last lane.
        self.assertEqual(self.inked(TALL, theme.ACTIVE),
                         block((left + PITCH // 2, TALL[1], 1, BARS_H + 2 * LANE + LANE_H)))
        self.assertEqual((self.at(left + 1, top + 1), self.at(left, top)), paint(theme.MUSTARD, 15, True))
        self.assertEqual((self.at(left + 1, grid.y(1) + 1), self.at(left, grid.y(1))),
                         paint(theme.MUSTARD, None, True))
        self.assertEqual(self.square(grid.square(1, 8)), paint(theme.MUSTARD, None))
        # With none, before the pattern, and in a row scrolled out of view, there is no line and no square is lit.
        for playhead, beats in ((None, 0), (-0.5, 0), (2.5, 1)):
            steps.playhead = playhead
            scroll(steps, beats=beats)
            grid = self.draw(steps)
            with self.subTest(playhead=playhead):
                self.assertEqual(self.inked(TALL, theme.ACTIVE), set())
                self.assertEqual(self.square(grid.square(1, 9)), paint(theme.MUSTARD, None))
        # Nor past the end of a pattern, where one bar leaves room in the view for a second.
        steps.show(kit(rows=16.0), LANES, 16, 4)
        steps.playhead = 16.0
        self.draw(steps)
        self.assertEqual(self.inked(TALL, theme.ACTIVE), set())

    def test_the_bars_are_numbered_from_1_along_the_top_each_over_its_first_square(self) -> None:
        steps = shown(rect=TALL)
        grid = self.draw(steps)
        for bar, text in ((0, "BAR 1"), (1, "BAR 2")):
            strip = (grid.x(16 * bar), TALL[1], 40, BARS_H)
            self.assertEqual(self.inked(strip, theme.DIM), self.written(text, strip, theme.DIM))
            self.assertEqual(min(x for x, _ in self.inked(strip, theme.DIM)), strip[0])
        self.assertEqual(self.inked((TALL[0], TALL[1], TALL[2], BARS_H), theme.DIM),
                         self.inked((LEFT, TALL[1], 40, BARS_H), theme.DIM)
                         | self.inked((LEFT + BAR_W, TALL[1], 40, BARS_H), theme.DIM))
        # Scrolled by a beat, the number of the first bar has gone with its first square, and the second has moved.
        scroll(steps, beats=1)
        grid = self.draw(steps)
        strip = (grid.x(16), TALL[1], 40, BARS_H)
        self.assertEqual((grid.first_row, grid.x(16)), (4, LEFT + BAR_W - 4 * PITCH - BEAT_GAP + GAP))
        self.assertEqual(self.inked((TALL[0], TALL[1], TALL[2], BARS_H), theme.DIM),
                         self.written("BAR 2", strip, theme.DIM))

    def test_nothing_is_drawn_outside_the_rectangle_when_scrolled(self) -> None:
        steps = shown(*(hit(letter, row) for letter in "ksH" for row in range(32)))
        steps.playhead = 31.9
        # In each view something is drawn at an edge: the third lane is cut, then a square, and last a label's box.
        for beats, drums, view, edge in ((0, 0, (0, 0), (LEFT, BOTTOM - 1)), (1, 1, (4, 1), (RIGHT - 1, TOP)),
                                         (9, 0, (8, 1), (RECT[0], TOP))):
            scroll(steps, beats, drums)
            grid = self.draw(steps)
            with self.subTest(view=view):
                self.assertEqual((grid.first_row, grid.first_drum), view)
                self.assertNotEqual(self.at(*edge), theme.PANEL)
                self.assertEqual({self.at(*pos) for pos in block((0, 0, *self.SIZE)) - block(RECT)}, {theme.PANEL})
                # No square reaches into the labels' strip, nor into the strip of the bar numbers.
                self.assertEqual(self.inked((LEFT - LABEL_GAP, RECT[1], LABEL_GAP, RECT[3]), theme.PANEL),
                                 block((LEFT - LABEL_GAP, RECT[1], LABEL_GAP, RECT[3])))
                self.assertEqual(self.inked((LEFT, RECT[1], AREA_W, BARS_H), theme.FAULT), set())

    def test_the_drums_of_the_first_pattern_of_the_arcade_example(self) -> None:
        song = Project.load(EXAMPLES / "arcade.json").song("tune")
        found = songview.pattern(song, "a")
        made = lanes([songview.pattern(song, name) for name in song["order"]], songview.drum_rows(song))
        self.assertEqual(made, [Lane("kit", "k", "kick"), Lane("kit", "s", "snare"), Lane("kit", "h", "hat")])
        steps = Steps((0, 0, 660, 120))
        steps.show(found, made, songview.bar_rows(song), song.get("rows_per_beat", 4))
        self.surface = self.canvas.surface = pygame.Surface((660, 120))
        grid = self.draw(steps)
        self.assertEqual((grid.rows_in_view(), grid.drums_in_view()), (range(0, 64), range(0, 3)))
        hits = {(note.drum, int(note.row)) for channel in found.channels if channel.name == "kit"
                for note in channel.notes}
        self.assertEqual(len(hits), 32)
        filled = {(letter, row) for drum, letter in enumerate("ksh") for row in range(64)
                  if self.square(grid.square(drum, row))[0] == grid.colour(row)}
        self.assertEqual(filled, hits)
        self.assertEqual([self.square(grid.square(drum, row))[0]
                          for drum, row in ((0, 0), (2, 2), (1, 4), (0, 10), (1, 12), (0, 48))],
                         [theme.FAULT, theme.FAULT, theme.ACTIVE, theme.MUSTARD, theme.TEXT, theme.FAULT])
        lower = (LABEL_PAD, grid.y(2) + LABEL_EDGE + LINE_H, 60, LINE_H)
        self.assertEqual(self.inked((0, grid.y(2), LABELS_W, LANE_H), theme.INSTRUMENT),
                         self.written("hat", lower, theme.INSTRUMENT))


if __name__ == "__main__":
    unittest.main()
