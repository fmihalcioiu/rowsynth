"""Tests for rowsynth.gui.steps: the sums that place the drum steps, the widget that shows them, and what it draws."""
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
from rowsynth.gui.steps import (BAR_GAP, BARS_H, BEAT_GAP, EDGE, EMPTY_FILL, EMPTY_LINE, FAINTEST, GAP, LABEL_EDGE,
                                LABEL_GAP, LABEL_PAD, LABELS_W, LANE_GAP, LANE_H, MARK_W, SQUARE_W, Grid, Steps,
                                hit_colour, mix, paint)
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


def hit(letter: str, row: float, level: int = 15) -> Note:
    """A hit of a drum letter that starts at a row."""
    return Note(row, 1.0, 60.0, level, "kick", letter)


def kit(*notes: Note, rows: float = 32.0) -> Pattern:
    """A pattern with a tuned channel and a drum channel, which has those hits. It is two bars unless told."""
    return Pattern("a", rows, (Channel("lead", "lead", False, (Note(0.0, 4.0, 60.0, 15, "lead"),), rows, True),
                               Channel("kit", "kick", True, notes, rows, True)))


def shown(*notes: Note, rect: layout.Rect = RECT) -> Steps:
    """The steps showing a drum channel with those hits, sixteen rows to a bar and four to a beat."""
    steps = Steps(rect, help="the drum steps")
    steps.show(kit(*notes), "kit", DRUMS, 16, 4)
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
        self.steps.show(kit(hit("k", 0)), "kit", DRUMS, 16, 4)
        self.assertEqual(self.view(), (4, 0))
        self.steps.show(kit(hit("k", 0), rows=16.0), "kit", DRUMS[:1], 16, 4)
        self.assertEqual((self.view(), self.steps.grid.rows, self.steps.grid.drums), ((0, 0), 16, 1))
        # A count of rows a hair off a whole one is that whole one, and half a row more is a square more.
        for rows, squares in ((15.9999999, 16), (16.0000001, 16), (16.5, 17)):
            self.steps.show(kit(rows=rows), "kit", DRUMS, 16, 4)
            self.assertEqual(self.steps.grid.rows, squares)

    def test_step_at_is_the_letter_and_the_row_on_a_square_and_none_in_a_gap_and_in_the_labels(self) -> None:
        x, y, w, h = self.grid.square(1, 5)
        self.assertEqual([self.steps.step_at(pos) for pos in ((x, y), (x + w - 1, y + h - 1), (LEFT, TOP),
                                                             (x, self.grid.y(2)))],
                         [("s", 5), ("s", 5), ("k", 0), ("H", 5)])
        # Right of the square and above it, in the strip of the bar numbers, on a label and just before the squares.
        self.assertEqual([self.steps.step_at(pos) for pos in ((x + w, y), (x, y - 1), (x, RECT[1] + 2),
                                                             (RECT[0] + 5, y + 5), (LEFT - 1, y))],
                         [None, None, None, None, None])
        # A hit a hair off a whole row is on that row, and after a scroll the same place is another step.
        self.steps.show(kit(hit("s", 4.9999999, 7)), "kit", DRUMS, 16, 4)
        self.ui.handle(Event("move", (x, y)))
        self.assertEqual((self.steps.step_at((x, y)), self.ui.help), (("s", 5), "s, bar 1, step 6: hit, level 7"))
        self.wheel(-1)
        self.wheel(-1, SHIFT)
        self.assertEqual(self.steps.step_at((LEFT, TOP)), ("s", 4))

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

    def test_a_channel_the_pattern_does_not_have_and_one_that_is_no_drum_channel_are_refused(self) -> None:
        for channel, said in (("bass", "pattern 'a' has no channel called 'bass'"),
                              ("lead", "channel 'lead' is not a drum channel")):
            with self.assertRaises(RowSynthError) as refused:
                self.steps.show(kit(), channel, DRUMS, 16, 4)
            self.assertEqual(str(refused.exception), said)
        with self.assertRaises(RowSynthError):
            self.steps.show(kit(), "kit", DRUMS, 16, 0)
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

    def written(self, text: str, rect: layout.Rect, colour: tuple[int, ...]) -> set[tuple[int, int]]:
        """The pixels of a rectangle that are of a colour when nothing but a text is drawn in it, in that colour."""
        kept, self.surface = self.surface, pygame.Surface(self.SIZE)
        Canvas(self.surface, self.canvas.font).text_in(text, rect, colour)
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

    def test_a_drum_with_no_hits_is_a_lane_of_empty_squares(self) -> None:
        grid = self.draw(shown(hit("k", 0), hit("H", 3), rect=TALL))
        whole = grid.rows_in_view()[:-1]
        self.assertEqual(len(whole), 24)
        self.assertEqual([self.square(grid.square(1, row)) for row in whole],
                         [paint(grid.colour(row), None) for row in whole])
        # A pattern with no rows has the labels and no squares, and no bar is numbered.
        bare = Steps(TALL)
        bare.show(kit(rows=0.0), "kit", DRUMS, 16, 4)
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
        steps.show(kit(rows=16.0), "kit", DRUMS, 16, 4)
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

    def test_the_drum_channel_of_the_first_pattern_of_the_arcade_example(self) -> None:
        song = Project.load(EXAMPLES / "arcade.json").song("tune")
        found = songview.pattern(song, "a")
        steps = Steps((0, 0, 660, 120))
        steps.show(found, "kit", songview.drum_rows(song), songview.bar_rows(song), song.get("rows_per_beat", 4))
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
