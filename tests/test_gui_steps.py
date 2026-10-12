"""Tests for rowsynth.gui.steps: the lanes of a song, the sums that place the steps, the widget, and what it draws."""
from __future__ import annotations

import dataclasses
import os
import unittest
from pathlib import Path

from rowsynth.errors import RowSynthError
from rowsynth.gui import layout, noteedit, songview, sprites, theme
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.font import Font
from rowsynth.gui.kit import VELOCITY_HELP, velocity_bar
from rowsynth.gui.songview import Channel, DrumRow, Note, Pattern
from rowsynth.gui.steps import (ACROSS_HELP, ALONG_HELP, BAR, BAR_GAP, BARS_H, BEAT_GAP, CHANNEL_GAP, DOT, DOT_H,
                                DOT_W, EDGE, EMPTY_FILL, EMPTY_LINE, FAINTEST, GAP, GLOW, LABEL_EDGE, LABEL_GAP,
                                LABEL_PAD, LABELS_W, LANE_GAP, LANE_H, MARK_W, OFF_GRID, PLUS_GAP, PLUS_HELP,
                                SQUARE_W, SQUARE_W_MAX, SQUARE_W_MIN, TAIL, WHEEL_HELP, Gesture, Grid, Lane, Steps,
                                begun, boxed, copied, dot, hit_colour, lanes, lifted, mix, moved, paint, pasted,
                                removed)
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
CTRL = frozenset({"ctrl"})

# A rectangle whose squares' room cuts the third lane and the twenty-fifth row, and one with room for three whole
# lanes and the plus under them. Each has the room of the two scroll bars as well.
RECT = (10, 20, 296 + BAR, 100 + BAR)
TALL = (10, 20, 296 + BAR, 130 + BAR + TAIL)
# Where the squares start in both, how wide their room is and how high in the first, and where that room ends.
LEFT, TOP, AREA_W, AREA_H = RECT[0] + LABELS_W, RECT[1] + BARS_H, RECT[2] - LABELS_W - BAR, RECT[3] - BARS_H - BAR
RIGHT, BOTTOM = LEFT + AREA_W, TOP + AREA_H
# What the help line says of the wheel, after what the presses do.
WHEEL = f"   wheel: {WHEEL_HELP}"
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


# What the three letters play, as a song has it. The snare is held for two rows, whatever its text says.
TABLE = {"k": {"inst": "kick", "note": "c3"}, "s": {"inst": "a very long snare", "note": "c5", "rows": 2},
         "H": {"inst": "hat", "note": "c7"}}


def beat(*hits: tuple[str, int, int], rows: float = 32.0) -> tuple[Note, ...]:
    """The notes of a drum channel with those hits, each a letter, a row and a level, as `noteedit` puts them."""
    notes: tuple[Note, ...] = ()
    for letter, row, level in hits:
        notes = noteedit.hit(notes, row, letter, level, TABLE, rows)
    return notes


# One drum channel on the grid: two kicks, the second of them quiet, a snare between them and a hat after.
KIT = beat(("k", 0, 15), ("s", 4, 9), ("k", 8, 7), ("H", 10, 12))


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
        self.assertEqual(grid.area, (LEFT, TOP, AREA_W, AREA_H))
        # The two scroll bars have the room at the right of the squares and under them, a pixel away.
        self.assertEqual((grid.along, grid.across),
                         ((RIGHT + theme.SCROLL_GAP, TOP, theme.SCROLL_W, AREA_H),
                          (LEFT, BOTTOM + theme.SCROLL_GAP, AREA_W, theme.SCROLL_W)))
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
        # Left of the squares there is no row, nor right of their room, where the last square in view is cut.
        grid = Grid(RECT, 32, 3, 16, 4)
        self.assertEqual([grid.row_at(x) for x in (LEFT - 1, LEFT, RIGHT - 1, RIGHT)], [None, 0, 24, None])

    def test_the_y_of_a_drum_the_drum_at_a_y_and_the_square_of_a_step(self) -> None:
        grid = Grid(RECT, 32, 3, 16, 4)
        self.assertEqual([grid.y(drum) for drum in range(3)], [TOP, TOP + LANE, TOP + 2 * LANE])
        self.assertEqual([grid.drum_at(y) for y in (TOP - 1, TOP, TOP + LANE_H - 1, TOP + LANE_H, TOP + LANE)],
                         [None, 0, 0, None, 1])
        # The third lane is cut by the room of the squares, and under that room there is none.
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
        short = Grid((0, 0, 400, BARS_H + LANE_H + BAR), 5, 2, 16, 4)
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
        # The furthest: where the last row is whole in view and one row less would not be, and the last drum is
        # with the room of the plus under it.
        end = grid.scrolled(rows=99, drums=99)
        self.assertEqual((end.first_row, end.first_drum), (8, 1))
        self.assertLessEqual(end.x(31) + SQUARE_W, RIGHT)
        self.assertGreater(dataclasses.replace(end, first_row=7).x(31) + SQUARE_W, RIGHT)
        self.assertLessEqual(end.y(2) + LANE_H + TAIL, BOTTOM)
        self.assertEqual((grid.plus[2:], end.plus),
                         ((0, 0), (RECT[0], end.y(2) + LANE_H + PLUS_GAP, theme.PLUS, theme.PLUS)))
        # With no lane the plus stands where the label of the first would.
        self.assertEqual(Grid(RECT, 32, 0, 16, 4).plus, (RECT[0], TOP, theme.PLUS, theme.PLUS))

    def test_the_scroll_bars_show_the_view_in_pixels_and_a_bar_that_is_moved_takes_it_to_a_whole_lane_and_row(
            self) -> None:
        grid = Grid(RECT, 32, 3, 16, 4)
        lanes_h, rows_w = 2 * LANE + LANE_H + TAIL, BAR_W + 15 * PITCH + 3 * (BEAT_GAP - GAP) + SQUARE_W
        self.assertEqual(grid.spans(), ((lanes_h, AREA_H, 0), (rows_w, AREA_W, 0)))
        moved = grid.scrolled(rows=5, drums=1)
        self.assertEqual(moved.spans(), ((lanes_h, AREA_H, LANE), (rows_w, AREA_W, 5 * PITCH + BEAT_GAP - GAP)))
        # A bar that was moved takes the view to the lane and to the row that begin nearest to there.
        self.assertEqual([(to.first_drum, to.first_row) for to in (grid.at(lanes=LANE // 2 + 1), grid.at(rows=20),
                                                                   grid.at(lanes=5, rows=3), grid.at(rows=9999))],
                         [(1, 0), (0, 2), (0, 0), (0, 8)])

    def test_the_zoom_makes_the_squares_wider_or_narrower_between_its_limits_round_the_row_under_the_mouse(
            self) -> None:
        grid = Grid(RECT, 32, 3, 16, 4, first_row=4)
        at = grid.x(12) + 2
        for wide in (SQUARE_W + 4, SQUARE_W - 1):
            zoomed = grid.zoomed(wide, at)
            self.assertEqual((zoomed.square_w, zoomed.pitch, zoomed.square(0, 12)[2]), (wide, wide + GAP, wide))
            # Row 12 is still under the mouse, or as near to it as a view that starts at a whole row lets it be.
            self.assertLess(abs(zoomed.x(12) - at), zoomed.pitch)
            self.assertEqual(zoomed.row_at(zoomed.x(12) + wide - 1), 12)
        self.assertEqual([grid.zoomed(wide, at).square_w for wide in (0, 99)], [SQUARE_W_MIN, SQUARE_W_MAX])
        # Zoomed out so far that all the rows fit, the view starts at the first row.
        self.assertEqual(dataclasses.replace(grid, first_row=8).zoomed(SQUARE_W_MIN, at).first_row, 0)
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
        room = BARS_H + 2 * LANE + CHANNEL_GAP + LANE_H + TAIL + BAR
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
        # A letter that no channel plays anywhere has a lane in the picked drum channel, and in no other.
        pair = [band(kick=(hit("k", 0),), tops=(hit("H", 2),))]
        self.assertEqual(lanes(pair, DRUMS), [Lane("kick", "k", "kick"), Lane("tops", "H", "hat")])
        self.assertEqual(lanes(pair, DRUMS, "tops"), [Lane("kick", "k", "kick"), Lane("tops", "s", "a very long snare"),
                                                     Lane("tops", "H", "hat")])
        self.assertEqual(lanes(pair, DRUMS, "lead"), lanes(pair, DRUMS))
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
        self.steps.show(kit(hit("s", 4.9999999, 7)), LANES, 16, 4, TABLE)
        self.ui.handle(Event("move", (x, y)))
        self.assertEqual((self.steps.step_at((x, y)), self.ui.help),
                         (("kit", "s", 5), f"press: pick s at bar 1, step 6, a drag moves it   right press: rub it "
                                           f"out{WHEEL}"))
        self.wheel(-1)
        self.wheel(-1, SHIFT)
        self.assertEqual(self.steps.step_at((LEFT, TOP)), ("kit", "s", 4))

    def test_steps_that_are_only_shown_say_their_own_line_on_every_square_and_a_label_says_what_a_press_does(
            self) -> None:
        self.assertEqual(self.ui.help, "")
        # No drum table was given, so no press changes a hit: on a hit, on an empty square and in a gap.
        for pos in (self.grid.square(0, 20)[:2], self.grid.square(1, 20)[:2], (self.grid.x(4) - 1, TOP + 3)):
            self.ui.handle(Event("move", pos))
            self.assertEqual(self.ui.help, "the drum steps")
        # The line can be set, and a key leaves it be.
        self.steps.help = "drums"
        self.assertFalse(self.ui.handle(Event("key", key="space")))
        self.assertEqual(self.ui.help, "drums")
        # A label says what a press on it does, and of its menu only where there is an owner to open one.
        self.ui.handle(Event("move", (RECT[0] + 5, TOP + 3)))
        self.assertEqual(self.ui.help, f"press: show the velocity of k{WHEEL}")
        self.steps.asked = lambda place, pos: None
        self.assertEqual(self.ui.help, f"press: show the velocity of k   right press: its menu{WHEEL}")
        # What is left of the labels' strip beside the bar numbers is no label.
        self.ui.handle(Event("move", (RECT[0] + 5, RECT[1] + 3)))
        self.assertEqual(self.ui.help, "drums")

    def test_with_two_drum_channels_step_at_and_the_help_line_say_which_channel_a_lane_is_of(self) -> None:
        steps = self.ui.add(paired((0, 200, *TALL[2:])))
        grid = steps.grid
        self.assertEqual((grid.drums, grid.breaks, steps.lanes), (3, (1,), tuple(PAIR)))
        for drum, row, step in ((0, 0, ("kick", "k", 0)), (1, 0, ("tops", "k", 0)), (1, 4, ("tops", "k", 4)),
                                (0, 9, ("kick", "k", 9)), (2, 18, ("tops", "H", 18))):
            x, y, _, _ = grid.square(drum, row)
            self.assertEqual(steps.step_at((x + 2, y + 9)), step)
        # In the air between the two channels there is no step, and the line is the widget's own.
        self.ui.handle(Event("move", (grid.x(0), grid.y(1) - 1)))
        self.assertEqual((steps.step_at((grid.x(0), grid.y(1) - 1)), self.ui.help), (None, "the drum steps"))
        # A label names the channel of its lane as well, and the box for a name is where its letter is written.
        self.ui.handle(Event("move", (2, grid.y(1) + 2)))
        self.assertEqual(self.ui.help, f"press: show the velocity of k in tops{WHEEL}")
        self.assertEqual((grid.label(1), grid.label_at((2, grid.y(1) + 2)), grid.label_at((2, grid.y(1) - 1))),
                         ((0, grid.y(1), LABELS_W - LABEL_GAP, LANE_H), 1, None))
        self.assertEqual(steps.name_rect(1), (LABEL_PAD, grid.y(1) + LABEL_EDGE, LABELS_W - LABEL_GAP - 2 * LABEL_PAD,
                                              LINE_H))

    def test_a_lane_of_a_channel_the_pattern_does_not_have_or_of_one_that_is_no_drum_channel_is_refused(self) -> None:
        for channel, said in (("bass", "pattern 'a' has no channel called 'bass'"),
                              ("lead", "channel 'lead' is not a drum channel")):
            with self.assertRaises(RowSynthError) as refused:
                self.steps.show(kit(), [LANES[0], Lane(channel, "k", "kick")], 16, 4)
            self.assertEqual(str(refused.exception), said)
        with self.assertRaises(RowSynthError):
            self.steps.show(kit(), LANES, 16, 0)
        # What it showed before is still there.
        self.assertEqual((self.steps.step_at((LEFT, TOP)), self.steps.lanes), (("kit", "k", 0), tuple(LANES)))


class Gestures(unittest.TestCase):
    GRID = Grid(TALL, 32, 3, 16, 4)

    def test_a_press_begins_a_put_on_an_empty_square_a_move_on_a_hit_and_a_take_with_the_right_button(self) -> None:
        grid, hits = self.GRID, {(0, 0), (1, 4)}
        x = grid.x(4) + 1
        for drum, button, kind, rows in ((0, 1, "put", []), (1, 1, "move", [4]), (0, 3, "take", []),
                                         (1, 3, "take", [])):
            with self.subTest(drum=drum, button=button):
                self.assertEqual(begun(grid, hits, (x, grid.y(drum) + 3), button, 9),
                                 Gesture(kind, drum, 9, rows, button, x))
        # A hit is put at the top level where no other is told.
        self.assertEqual(begun(grid, hits, (x, grid.y(2)), 1).level, 15)
        # No press begins one in a gap, on a label or among the bar numbers, and the middle button begins none.
        gap = (grid.x(4) - 1, grid.y(1))
        for pos, button in ((gap, 1), ((TALL[0] + 3, grid.y(1)), 1), ((x, TALL[1] + 2), 3), ((x, grid.y(1)), 2)):
            self.assertIsNone(begun(grid, hits, pos, button))
        # With ctrl the left button on a hit picks it, and on an empty square it puts as without.
        self.assertEqual([begun(grid, hits, (x, grid.y(drum) + 3), 1, 9, CTRL) for drum in (1, 0)],
                         [Gesture("pick", 1, 9, [4], 1, x), Gesture("put", 0, 9, button=1, x=x)])
        # With shift it draws a box wherever in the room of the squares no hit is, a gap too, and moves a hit.
        for pos in ((x, grid.y(0) + 3), gap):
            self.assertEqual(begun(grid, hits, pos, 1, 9, SHIFT),
                             Gesture("box", None, 9, button=1, x=pos[0], start=pos, end=pos))
        self.assertEqual(begun(grid, hits, (x, grid.y(1) + 3), 1, 9, SHIFT).kind, "move")
        self.assertEqual([begun(grid, hits, pos, button, 9, SHIFT) for pos, button
                          in (((TALL[0] + 3, grid.y(1)), 1), ((x, TALL[1] + 2), 1), (gap, 3))], [None] * 3)

    def test_the_squares_on_the_way_from_an_x_to_another_are_all_of_them_either_way_round(self) -> None:
        grid = self.GRID
        self.assertEqual(grid.passed(grid.x(2) + 3, grid.x(2) + 3), range(2, 3))
        # Over the gap between two beats and over the one between two bars, however far the mouse went at once.
        for one, other, rows in ((grid.x(2) + 3, grid.x(9), range(2, 10)), (grid.x(14), grid.x(17) + 5, range(14, 18)),
                                 (grid.x(0), grid.x(23), range(0, 24))):
            self.assertEqual((grid.passed(one, other), grid.passed(other, one)), (rows, rows))
        # An end in a gap is on no square: only the squares between the two ends are on the way.
        self.assertEqual(grid.passed(grid.x(3) + SQUARE_W, grid.x(6) - 1), range(4, 6))
        self.assertEqual(grid.passed(grid.x(3) + SQUARE_W, grid.x(4) - 1), range(0))
        # Past the squares in view the way ends at the last of them, on either side.
        self.assertEqual((grid.passed(grid.x(20), RIGHT + 500), grid.passed(-50, grid.x(1))),
                         (range(20, 25), range(0, 2)))
        moved = dataclasses.replace(grid, first_row=8)
        self.assertEqual(moved.passed(0, moved.x(9)), range(8, 10))

    def test_a_drag_takes_in_each_square_once_and_gives_what_the_rules_make_of_its_squares(self) -> None:
        grid = self.GRID
        drag = Gesture("put", 0, 12, x=grid.x(5))
        self.assertEqual([drag.along(grid, grid.x(row)) for row in (5, 2, 6, 3)], [[5], [2, 3, 4], [6], []])
        self.assertEqual((drag.rows, drag.x), ([5, 2, 3, 4, 6], grid.x(3)))
        # A put fills the squares at whose rows the channel has nothing, at its level: the kick that was there
        # stays as it is, and so does the snare.
        base = beat(("k", 4, 3), ("s", 6, 15))
        self.assertEqual([(note.drum, note.row, note.rows, note.level) for note in drag.gives(base, "k", TABLE, 32.0)],
                         [("k", 2.0, 1.0, 12), ("k", 3.0, 1.0, 12), ("k", 4.0, 1.0, 3), ("k", 5.0, 1.0, 12),
                          ("s", 6.0, 2.0, 15)])
        # A take empties the squares that have a hit of its letter.
        drag.kind = "take"
        self.assertEqual(drag.gives(base, "k", TABLE, 32.0), noteedit.unhit(base, 4, "k", TABLE, 32.0))
        self.assertEqual(drag.gives(base, "H", TABLE, 32.0), base)
        # A move passes no square: the hit that was pressed is at the square the mouse is on, or was on last. It
        # is so many rows on, and what that makes of the picked hits is not the doing of one lane.
        move = Gesture("move", 0, 15, [4], 1, grid.x(4))
        self.assertEqual((move.along(grid, grid.x(4) + 2), move.rows, move.by, move.gives(base, "k", TABLE, 32.0)),
                         ([], [4, 4], 0, base))
        self.assertEqual((move.along(grid, grid.x(9)), move.along(grid, grid.x(10) - 1), move.rows, move.by),
                         ([], [], [4, 9], 5))
        self.assertEqual((move.along(grid, grid.x(1)), move.by, move.gives(base, "k", TABLE, 32.0)), ([], -3, base))
        # A box passes none either: it goes from the pixel of its press to the pixel the mouse is on.
        box = Gesture("box", None, 15, button=1, start=(40, 90), end=(30, 95))
        self.assertEqual((box.along(grid, grid.x(3)), box.rows, box.box, move.box), ([], [], (30, 90, 11, 6), None))
        # A hit a hair off its row is taken away, and gives way, at the place it has.
        off = (Note(4.9999999, 1.0, 48, 15, "kick", "k"),)
        self.assertEqual(Gesture("take", 0, 15, [5]).gives(off, "k", TABLE, 32.0), ())
        self.assertEqual([(note.drum, note.row) for note in Gesture("put", 1, 15, [5]).gives(off, "s", TABLE, 32.0)],
                         [("s", 4.9999999)])
        # What a put, a take or a move is called in the undo list goes by how many hits it was.
        self.assertEqual([Gesture(kind, 0, 15).label(count) for kind, count
                          in (("put", 1), ("put", 3), ("take", 1), ("take", 2), ("move", 1), ("move", 2))],
                         ["hit placed", "3 hits placed", "hit removed", "2 hits removed", "hit moved", "2 hits moved"])

    def test_a_put_takes_the_place_of_what_the_channel_plays_only_in_the_square_that_was_pressed(self) -> None:
        def left(notes: tuple[Note, ...], *rows: int) -> list[tuple[str, float]]:
            """The letter and the row of each note that a put of kicks leaves, pressed on the first of some rows."""
            return [(note.drum, note.row) for note in Gesture("put", 0, 15, [*rows]).gives(notes, "k", TABLE, 32.0)]

        base = beat(("k", 4, 3), ("s", 6, 15), ("H", 8, 9))
        # Pressed on the snare's row, the kick takes its place. The hat further on is passed, and stays.
        self.assertEqual(left(base, 6, 7, 8, 9), [("k", 4.0), ("k", 6.0), ("k", 7.0), ("H", 8.0), ("k", 9.0)])
        # Pressed on a free row and dragged over both, nothing gives way, whichever way the drag went and
        # wherever it has come to.
        self.assertEqual(left(base, 5, 6, 7, 8), [("k", 4.0), ("k", 5.0), ("s", 6.0), ("k", 7.0), ("H", 8.0)])
        self.assertEqual(left(base, 9, 8, 7, 6), [("k", 4.0), ("s", 6.0), ("k", 7.0), ("H", 8.0), ("k", 9.0)])
        # A note that is no hit is in the way as a hit is, and gives way as one does.
        tuned = (Note(4.0, 2.0, 60, 15, "kick"),)
        self.assertEqual((left(tuned, 3, 4), left(tuned, 4, 3)), ([("k", 3.0), ("", 4.0)], [("k", 3.0), ("k", 4.0)]))


class Picks(unittest.TestCase):
    # Two drum channels: one of kicks, and one that plays a hat, a kick and a snare.
    NOTES = {"kick": beat(("k", 0, 15), ("k", 1, 9), ("k", 8, 7)),
             "tops": beat(("H", 2, 15), ("k", 4, 9), ("s", 5, 12))}
    FOUR = [Lane("kick", "k", "kick"), Lane("tops", "k", "kick"), Lane("tops", "s", "a very long snare"),
            Lane("tops", "H", "hat")]

    def hits(self, made: dict[str, tuple[Note, ...]]) -> dict[str, list[tuple[str, float, int]]]:
        """The letter, the row and the level of each note of each channel."""
        return {channel: [(note.drum, note.row, note.level) for note in notes] for channel, notes in made.items()}

    def test_the_picked_hits_move_along_their_lanes_together_or_not_at_all(self) -> None:
        made, picks = moved(self.NOTES, {("kick", "k", 0), ("kick", "k", 1), ("tops", "H", 2)}, 1, TABLE, 32.0)
        # A picked hit is not in the way of another, and neither is what another channel plays at the row. Each
        # has its level, and the notes are as putting those hits makes them.
        self.assertEqual(made, {"kick": beat(("k", 1, 15), ("k", 2, 9), ("k", 8, 7)),
                                "tops": beat(("H", 3, 15), ("k", 4, 9), ("s", 5, 12))})
        self.assertEqual(picks, {("kick", "k", 1), ("kick", "k", 2), ("tops", "H", 3)})
        # Nothing moves where one of them would land on what its channel plays, or leave the pattern at an end.
        for chosen, by in (({("tops", "H", 2)}, 2), ({("kick", "k", 8), ("tops", "k", 4)}, 1), ({("kick", "k", 0)}, -1),
                           ({("kick", "k", 8), ("kick", "k", 0)}, 24), ({("tops", "s", 5)}, 0)):
            self.assertEqual(moved(self.NOTES, chosen, by, TABLE, 32.0), (self.NOTES, chosen))
        self.assertEqual(self.hits(moved(self.NOTES, {("kick", "k", 8)}, 23, TABLE, 32.0)[0])["kick"][-1],
                         ("k", 31.0, 7))
        # A hit the notes do not have is not picked, and moves nothing.
        self.assertEqual(moved(self.NOTES, {("kick", "k", 3), ("kick", "s", 0), ("snare", "s", 0)}, 1, TABLE, 32.0),
                         (self.NOTES, frozenset()))

    def test_the_picked_hits_go_to_the_lane_above_or_below_where_that_is_a_drum_of_their_channel(self) -> None:
        made, picks = lifted(self.NOTES, {("tops", "k", 4), ("tops", "s", 5)}, self.FOUR, 1, TABLE, 32.0)
        # Each is a hit of the drum of its new lane, at its row and its level.
        self.assertEqual((self.hits(made)["tops"], made["kick"], picks),
                         ([("H", 2.0, 15), ("s", 4.0, 9), ("H", 5.0, 12)], self.NOTES["kick"],
                          {("tops", "s", 4), ("tops", "H", 5)}))
        self.assertEqual(lifted(self.NOTES, {("tops", "s", 5)}, self.FOUR, -1, TABLE, 32.0)[1], {("tops", "k", 5)})
        # Nothing moves where one of them has no lane there, or a lane of another channel.
        for chosen, by in (({("tops", "k", 4)}, -1), ({("kick", "k", 0)}, 1), ({("kick", "k", 0)}, -1),
                           ({("tops", "s", 5), ("tops", "H", 2)}, 1), ({("tops", "s", 5)}, 0)):
            self.assertEqual(lifted(self.NOTES, chosen, self.FOUR, by, TABLE, 32.0), (self.NOTES, chosen))
        self.assertEqual(lifted(self.NOTES, {("tops", "k", 4)}, self.FOUR[:2], 1, TABLE, 32.0)[0], self.NOTES)

    def test_the_picked_hits_are_removed_and_copied_and_a_clip_is_pasted_where_nothing_is_in_its_way(self) -> None:
        chosen = {("kick", "k", 1), ("tops", "s", 5), ("tops", "H", 2)}
        made, picks = removed(self.NOTES, {*chosen, ("tops", "s", 9)}, TABLE, 32.0)
        self.assertEqual((self.hits(made), picks), ({"kick": [("k", 0.0, 15), ("k", 8.0, 7)], "tops": [("k", 4.0, 9)]},
                                                    frozenset()))
        self.assertEqual(made["kick"], noteedit.unhit(self.NOTES["kick"], 1, "k", TABLE, 32.0))
        # A clip has the hits in the order of their rows, counted from the first, each with its level.
        clip = copied(self.NOTES, chosen)
        self.assertEqual((clip, copied(self.NOTES, [("kick", "k", 5)])),
                         ((("kick", "k", 0, 9), ("tops", "H", 1, 15), ("tops", "s", 4, 12)), ()))
        # Pasted, each is in its own lane, and the hits that were put are the picked ones.
        made, picks = pasted(self.NOTES, clip, 10, TABLE, 32.0)
        self.assertEqual((self.hits(made)["kick"][-1], self.hits(made)["tops"][-2:], picks),
                         (("k", 10.0, 9), [("H", 11.0, 15), ("s", 14.0, 12)],
                          {("kick", "k", 10), ("tops", "H", 11), ("tops", "s", 14)}))
        # One that would land where its channel plays something is left out, and so is one past the end.
        self.assertEqual([pasted(self.NOTES, clip, row, TABLE, 32.0)[1] for row in (3, 30, 1)],
                         [{("kick", "k", 3), ("tops", "s", 7)}, {("kick", "k", 30), ("tops", "H", 31)}, frozenset()])
        self.assertEqual(pasted(self.NOTES, clip, 1, TABLE, 32.0)[0], self.NOTES)
        # So is one whose channel is not shown, and one whose letter the song has no drum for.
        self.assertEqual(pasted({"kick": ()}, clip, 0, TABLE, 32.0), ({"kick": beat(("k", 0, 9))}, {("kick", "k", 0)}))
        self.assertEqual(pasted(self.NOTES, (("kick", "x", 0, 15),), 10, TABLE, 32.0), (self.NOTES, frozenset()))

    def test_a_box_picks_the_hits_whose_squares_it_touches_either_way_round(self) -> None:
        grid = Grid(TALL, 32, 3, 16, 4)
        hits = {(0, 0), (1, 4), (2, 5), (0, 20)}
        # From the air under the first lane to the first pixel of a square of the last.
        one, other = (grid.x(3), grid.y(0) + LANE_H), (grid.x(5), grid.y(2))
        self.assertEqual((boxed(grid, hits, one, other), boxed(grid, hits, other, one)), ({(1, 4), (2, 5)},) * 2)
        self.assertEqual([boxed(grid, hits, pos, pos) for pos in ((grid.x(0), grid.y(0)),
                                                                  (grid.x(0) + SQUARE_W, grid.y(0)))],
                         [{(0, 0)}, frozenset()])
        # A hit out of view is where its square would be: a box that goes on past the squares in view has it.
        self.assertEqual(boxed(dataclasses.replace(grid, first_row=8), hits, (LEFT, TOP), (RIGHT + 200, TOP)),
                         {(0, 20)})


class Editing(unittest.TestCase):
    def setUp(self) -> None:
        self.told: list[tuple[str, tuple[Note, ...], str]] = []
        self.heard: list[tuple[str, str]] = []
        self.added: list[str] = []
        self.ui = Ui()
        self.steps = self.ui.add(Steps(TALL, "the drum steps", edited=lambda *said: self.told.append(said),
                                       sounded=lambda *said: self.heard.append(said),
                                       added=lambda: self.added.append("drum")))
        for part in self.steps.parts:
            self.ui.add(part)
        # The velocity lane stands under the steps, as wide as they are.
        self.steps.lane.rect = (TALL[0], TALL[1] + TALL[3] + 10, TALL[2], 40)
        self.steps.show(kit(*KIT), LANES, 16, 4, TABLE)

    def at(self, drum: int, row: int) -> tuple[int, int]:
        """A place on the square of one step, as the steps are now."""
        x, y, _, _ = self.steps.grid.square(drum, row)
        return x + 2, y + 9

    def mouse(self, kind: str, drum: int, row: int, button: int = 0) -> bool:
        """Do one thing with the mouse on the square of a step. Whether it was used."""
        return self.ui.handle(Event(kind, self.at(drum, row), button=button))

    def click(self, drum: int, row: int, button: int = 1) -> None:
        """Press a button of the mouse on the square of a step, and let it go there."""
        self.mouse("down", drum, row, button)
        self.mouse("up", drum, row, button)

    def wheel(self, drum: int, row: int, steps: int, mods: frozenset[str] = frozenset()) -> bool:
        """Turn the wheel on the square of a step. Whether it was used."""
        return self.ui.handle(Event("wheel", self.at(drum, row), wheel=steps, mods=mods))

    def levels(self) -> dict[tuple[str, float], int]:
        """The level of each hit of the channel as the steps show it, by its letter and its row."""
        return {(note.drum, note.row): note.level for note in self.steps.notes("kit")}

    def view(self) -> tuple[int, int]:
        """The first row and the first drum in view."""
        return self.steps.grid.first_row, self.steps.grid.first_drum

    def ctrl(self, drum: int, row: int) -> None:
        """Press the left button on the square of a step with ctrl held, and let it go there."""
        for kind in ("down", "up"):
            self.ui.handle(Event(kind, self.at(drum, row), button=1, mods=CTRL))

    def key(self, key: str, *mods: str) -> bool:
        """Hand the Ui a key, some others held. Whether it was used."""
        return self.ui.handle(Event("key", key=key, mods=frozenset(mods)))

    def kicks(self) -> list[float]:
        """The rows at which the steps show a hit of the letter k in the channel kit."""
        return [note.row for note in self.steps.notes("kit") if note.drum == "k"]

    def owned(self, lanes: list[Lane] = LANES, **notes: tuple[Note, ...]) -> dict[str, tuple[Note, ...]]:
        """Give the steps an owner that saves: it keeps what they hand on, and shows them what it then has.

        `notes` are the notes it has at first for each drum channel, and with none the one channel of `KIT`.
        The steps are shown them with those lanes. What is handed on for one channel is one more of `told`, and
        what is handed on for several together one more of `as_one`. Given is what the owner has.
        """
        held = dict(notes) or {"kit": KIT}
        self.as_one: list[tuple[dict[str, tuple[Note, ...]], str]] = []

        def show() -> None:
            self.steps.show(band(**held), lanes, 16, 4, TABLE)

        def edited(channel: str, made: tuple[Note, ...], label: str) -> None:
            self.told.append((channel, made, label))
            held[channel] = made
            show()

        def together(made: dict[str, tuple[Note, ...]], label: str) -> None:
            self.as_one.append((dict(made), label))
            held.update(made)
            show()

        self.steps.edited, self.steps.together = edited, together
        show()
        return held

    def test_a_click_on_an_empty_square_puts_a_hit_there_at_level_15_and_tells_of_its_channel_once(self) -> None:
        self.assertTrue(self.mouse("down", 0, 5, 1))
        self.assertEqual(self.told, [])
        self.mouse("up", 0, 5, 1)
        # The hit has the three rows up to the next kick, and the snare before it the one row that is left to it.
        put = (KIT[0], dataclasses.replace(KIT[1], rows=1.0), Note(5.0, 3.0, 48, 15, "kick", "k"), KIT[2], KIT[3])
        self.assertEqual(self.told, [("kit", put, "hit placed")])
        self.assertEqual(put, noteedit.hit(KIT, 5, "k", 15, TABLE, 32.0))
        # Nothing is saved here: the steps go on showing what they were given.
        self.assertEqual(self.steps.notes("kit"), KIT)

    def test_a_click_on_a_hit_lets_it_be_heard_and_changes_nothing_and_the_right_button_rubs_it_out(self) -> None:
        self.click(0, 8)
        self.assertEqual((self.told, self.heard), ([], [("kit", "k")]))
        self.click(0, 8, 3)
        self.assertEqual((self.told, self.heard), ([("kit", (KIT[0], KIT[1], KIT[3]), "hit removed")], [("kit", "k")]))
        self.assertEqual(self.steps.notes("kit"), KIT)

    def test_a_drag_puts_a_hit_in_every_free_square_it_passes_however_far_the_mouse_went_at_once(self) -> None:
        self.mouse("down", 0, 1, 1)
        self.mouse("move", 0, 9)
        self.mouse("move", 0, 6)
        self.mouse("up", 0, 6, 1)
        # Nine squares in one move. The kick that was among them keeps its level, and is no hit that was placed.
        # Neither is one placed at the row of the snare, which stays, cut to the one row it has left.
        notes = tuple(dataclasses.replace(KIT[1], rows=1.0) if row == 4
                      else Note(float(row), 1.0, 48, 7 if row == 8 else 15, "kick", "k") for row in range(10))
        self.assertEqual(self.told, [("kit", (*notes, KIT[3]), "7 hits placed")])

    def test_a_drag_leaves_the_hits_that_other_letters_of_the_channel_have_and_puts_none_at_their_rows(self) -> None:
        # Along the hat's lane, over the row of the snare and over that of the second kick.
        self.mouse("down", 2, 3, 1)
        self.mouse("move", 2, 9)
        seen = self.steps.notes("kit")
        self.assertEqual([(note.drum, note.row, note.level) for note in seen],
                         [("k", 0.0, 15), ("H", 3.0, 15), ("s", 4.0, 9), ("H", 5.0, 15), ("H", 6.0, 15), ("H", 7.0, 15),
                          ("k", 8.0, 7), ("H", 9.0, 15), ("H", 10.0, 12)])
        self.mouse("up", 2, 9, 1)
        self.assertEqual((self.told, self.heard), ([("kit", seen, "5 hits placed")], [("kit", "H")] * 5))

    def test_a_drag_that_began_by_taking_the_place_of_a_hit_leaves_the_others_it_passes(self) -> None:
        # Pressed on the row of the snare, which gives way, and dragged over the row of the second kick.
        self.mouse("down", 2, 4, 1)
        self.mouse("move", 2, 9)
        self.mouse("move", 2, 4)
        seen = self.steps.notes("kit")
        self.assertEqual([(note.drum, note.row) for note in seen], [("k", 0.0), ("H", 4.0), ("H", 5.0), ("H", 6.0),
                                                                    ("H", 7.0), ("k", 8.0), ("H", 9.0), ("H", 10.0)])
        self.mouse("up", 2, 4, 1)
        self.assertEqual(self.told, [("kit", seen, "5 hits placed")])

    def test_a_hit_of_another_channel_at_the_row_is_not_in_the_way_of_a_drag(self) -> None:
        kick, tops = beat(("k", 0, 15), ("k", 2, 15)), beat(("H", 2, 15), ("k", 4, 9))
        self.steps.show(band(kick=kick, tops=tops), PAIR, 16, 4, TABLE)
        # The kick of the other channel is at the first row and at the third, where this one has its hat.
        self.mouse("down", 1, 0, 1)
        self.mouse("move", 1, 3)
        self.mouse("up", 1, 3, 1)
        (channel, notes, label), = self.told
        self.assertEqual((channel, label), ("tops", "3 hits placed"))
        self.assertEqual([(note.drum, note.row) for note in notes],
                         [("k", 0.0), ("k", 1.0), ("H", 2.0), ("k", 3.0), ("k", 4.0)])

    def test_a_hit_that_is_dragged_is_moved_along_its_lane_at_its_level_and_not_onto_what_its_channel_plays(
            self) -> None:
        self.mouse("down", 0, 8, 1)
        self.mouse("move", 0, 12)
        seen = self.steps.notes("kit")
        self.assertEqual([(note.drum, note.row, note.level) for note in seen],
                         [("k", 0.0, 15), ("s", 4.0, 9), ("H", 10.0, 12), ("k", 12.0, 7)])
        self.assertEqual(self.told, [])
        self.mouse("up", 0, 12, 1)
        self.assertEqual((self.told, self.heard), ([("kit", seen, "hit moved")], [("kit", "k")]))
        # Brought back to the square it came from, the hit is where it was, and nothing is told.
        self.mouse("down", 2, 10, 1)
        self.mouse("move", 2, 14)
        self.mouse("move", 2, 10)
        self.mouse("up", 2, 10, 1)
        self.assertEqual(len(self.told), 1)
        # Taken to the row of the snare, which its channel plays, it stays where it is, and nothing is told.
        self.mouse("down", 0, 0, 1)
        self.mouse("move", 0, 4)
        self.assertEqual(self.steps.notes("kit"), KIT)
        self.mouse("up", 0, 4, 1)
        self.assertEqual((len(self.told), self.steps.picks), (1, {("kit", "k", 0)}))

    def test_a_drag_on_a_picked_hit_moves_the_picked_hits_along_their_lanes_or_none_of_them(self) -> None:
        self.owned()
        self.click(0, 0)
        self.ctrl(2, 10)
        self.mouse("down", 2, 10, 1)
        self.mouse("move", 2, 12)
        seen = self.steps.notes("kit")
        self.assertEqual(([(note.drum, note.row, note.level) for note in seen], self.told),
                         ([("k", 2.0, 15), ("s", 4.0, 9), ("k", 8.0, 7), ("H", 12.0, 12)], []))
        self.mouse("up", 2, 12, 1)
        self.assertEqual((self.told, self.steps.picks),
                         ([("kit", seen, "2 hits moved")], {("kit", "k", 2), ("kit", "H", 12)}))
        # Two squares on, the kick would be on the row of the snare, and three back it would leave the pattern:
        # neither of the two moves. One square back both do.
        self.mouse("down", 2, 12, 1)
        for row in (14, 9):
            self.mouse("move", 2, row)
            self.assertEqual(self.steps.notes("kit"), seen)
        self.mouse("move", 2, 11)
        self.mouse("up", 2, 11, 1)
        self.assertEqual(([label for _, _, label in self.told], self.steps.picks),
                         (["2 hits moved"] * 2, {("kit", "k", 1), ("kit", "H", 11)}))

    def test_a_press_on_a_hit_picks_it_alone_and_with_ctrl_it_joins_the_picked_ones_or_leaves_them(self) -> None:
        self.assertEqual(self.steps.picks, frozenset())
        self.click(0, 8)
        self.click(1, 4)
        self.assertEqual((self.steps.picks, self.heard, self.told),
                         ({("kit", "s", 4)}, [("kit", "k"), ("kit", "s")], []))
        # With ctrl a hit joins the picked ones, and is heard. Pressed so again it leaves them, and is not.
        for drum, row, picks in ((0, 0, {("kit", "s", 4), ("kit", "k", 0)}), (1, 4, {("kit", "k", 0)})):
            self.assertTrue(self.ui.handle(Event("down", self.at(drum, row), button=1, mods=CTRL)))
            self.assertEqual((self.steps.picks, self.steps.busy), (picks, False))
            self.ui.handle(Event("up", self.at(drum, row), button=1, mods=CTRL))
        self.assertEqual((self.heard[2:], self.told), ([("kit", "k")], []))
        # The velocity lane marks the bars of the picked hits of its lane: here none of the snare's, and one kick.
        lane, grid = self.steps.lane, self.steps.grid
        self.assertEqual(lane.items, ((grid.x(4), SQUARE_W, 9 / 15, False),))
        self.ui.handle(Event("down", (TALL[0] + 5, grid.y(0) + 5), button=1))
        self.ui.handle(Event("up", (TALL[0] + 5, grid.y(0) + 5), button=1))
        self.assertEqual(lane.items, ((grid.x(0), SQUARE_W, 1.0, True), (grid.x(8), SQUARE_W, 7 / 15, False)))
        # A hit that is put is the picked one in what its owner then shows, and so are all that a drag put.
        self.owned()
        self.click(0, 5)
        self.assertEqual(self.steps.picks, {("kit", "k", 5)})
        self.mouse("down", 2, 12, 1)
        self.mouse("move", 2, 14)
        self.mouse("up", 2, 14, 1)
        self.assertEqual(self.steps.picks, {("kit", "H", 12), ("kit", "H", 13), ("kit", "H", 14)})
        # A right press that rubs nothing out leaves them picked, and one that rubs a hit out leaves none.
        self.click(1, 20, 3)
        self.assertEqual(len(self.steps.picks), 3)
        self.click(2, 13, 3)
        self.assertEqual(self.steps.picks, frozenset())

    def test_shift_and_a_drag_draw_a_box_that_picks_the_hits_it_touches_in_whatever_lanes(self) -> None:
        self.owned(PAIR, kick=beat(("k", 0, 15), ("k", 2, 15)), tops=beat(("H", 2, 15), ("k", 4, 9)))
        self.click(1, 4)
        # From an empty square of the first lane to one of the last: a kick of one channel, a hat of the other.
        self.assertTrue(self.ui.handle(Event("down", self.at(0, 1), button=1, mods=SHIFT)))
        self.ui.handle(Event("move", self.at(2, 3), mods=SHIFT))
        self.assertEqual((self.steps.busy, self.steps.picks), (True, {("tops", "k", 4)}))
        self.ui.handle(Event("up", self.at(2, 3), button=1, mods=SHIFT))
        self.assertEqual((self.steps.busy, self.steps.picks, self.told, self.heard),
                         (False, {("kick", "k", 2), ("tops", "H", 2)}, [], [("tops", "k")]))
        # The help line says where shift would draw a box: on an empty square and in a gap, and not on a hit.
        pick = f"right press: rub it out{WHEEL}"
        for pos in (self.at(0, 1), (self.steps.grid.x(4) - 1, self.at(0, 4)[1])):
            self.ui.handle(Event("move", pos, mods=SHIFT))
            self.assertEqual(self.ui.help, f"press: drag a box round the hits to pick{WHEEL}")
        self.ui.handle(Event("move", self.at(0, 0), mods=SHIFT))
        self.assertEqual(self.ui.help, f"press: pick k at bar 1, step 1, a drag moves it   {pick}")
        # With ctrl it says what the press does to the picked hits.
        for row, said in ((0, "pick k at bar 1, step 1 as well"),
                          (2, "take k at bar 1, step 3 out of the picked hits")):
            self.ui.handle(Event("move", self.at(0, row), mods=CTRL))
            self.assertEqual(self.ui.help, f"press: {said}   {pick}")
        # A press with shift that is not dragged picks nothing.
        self.ui.handle(Event("down", self.at(0, 1), button=1, mods=SHIFT))
        self.ui.handle(Event("up", self.at(0, 1), button=1, mods=SHIFT))
        self.assertEqual(self.steps.picks, frozenset())

    def test_delete_and_backspace_remove_the_picked_hits_and_those_of_two_channels_are_handed_on_as_one(self) -> None:
        self.owned()
        self.click(0, 8)
        self.assertTrue(self.key("delete"))
        self.assertEqual((self.told, self.steps.picks),
                         ([("kit", noteedit.unhit(KIT, 8, "k", TABLE, 32.0), "hit removed")], frozenset()))
        # With nothing picked the key is the steps' all the same, and removes nothing.
        self.assertTrue(self.key("backspace"))
        self.assertEqual(len(self.told), 1)
        # The picked hits of two channels are handed on together, once, for the owner to save as one step.
        kick, tops = beat(("k", 0, 15), ("k", 2, 15)), beat(("H", 2, 15), ("k", 4, 9))
        held = self.owned(PAIR, kick=kick, tops=tops)
        self.click(0, 0)
        self.ctrl(2, 2)
        self.ctrl(1, 4)
        self.assertTrue(self.key("backspace"))
        self.assertEqual((self.told[1:], self.as_one, self.steps.picks),
                         ([], [({"kick": noteedit.unhit(kick, 0, "k", TABLE, 32.0), "tops": ()}, "3 hits removed")],
                          frozenset()))
        self.assertEqual(held["tops"], ())
        # An owner that takes one channel at a time is told of each.
        self.owned(PAIR, kick=kick, tops=tops)
        self.steps.together = None
        self.key("a", "ctrl", "shift")
        self.key("delete")
        self.assertEqual(self.told[1:], [("kick", (), "4 hits removed"), ("tops", (), "4 hits removed")])

    def test_the_arrows_move_the_picked_hits_along_their_lanes_and_to_the_lane_above_or_below(self) -> None:
        self.owned()
        self.click(0, 8)
        self.assertEqual([self.key(key) for key in ("right", "left", "left")], [True] * 3)
        self.assertEqual(([label for _, _, label in self.told], self.steps.picks, self.levels()["k", 7.0]),
                         (["hit moved"] * 3, {("kit", "k", 7)}, 7))
        # It does not land on a row at which its channel plays another drum, nor leave the pattern.
        self.key("left")
        self.key("left")
        self.assertEqual((len(self.told), self.steps.picks), (5, {("kit", "k", 5)}))
        self.assertTrue(self.key("left"))
        self.click(0, 0)
        self.assertTrue(self.key("left"))
        self.assertEqual((len(self.told), self.steps.picks), (5, {("kit", "k", 0)}))
        # Up and down take it to the lane above or below: over the first lane and under the last there is none.
        self.assertEqual([self.key(key) for key in ("up", "down", "down", "down")], [True] * 4)
        self.assertEqual(([label for _, _, label in self.told[5:]], self.steps.picks, self.levels()["H", 0.0]),
                         (["hit moved"] * 2, {("kit", "H", 0)}, 15))
        # With shift an arrow is not the steps', and a lane of another channel is no lane to move to.
        self.assertFalse(self.key("up", "shift"))
        self.owned(PAIR, kick=beat(("k", 0, 15)), tops=beat(("H", 2, 15)))
        self.click(0, 0)
        self.key("down")
        self.assertEqual((len(self.told), self.steps.picks), (7, {("kick", "k", 0)}))
        # Where the hits cannot be changed, no key is the steps'.
        self.steps.show(kit(*KIT, hit("H", 20.5, 9)), LANES, 16, 4, TABLE)
        self.assertEqual([self.key(key) for key in ("delete", "left", "escape")], [False] * 3)

    def test_ctrl_a_picks_the_hits_of_the_chosen_lane_or_of_the_pattern_and_ctrl_c_and_v_copy_and_paste_them(
            self) -> None:
        self.owned()
        self.click(0, 8)
        self.assertTrue(self.key("a", "ctrl", "shift"))
        self.assertEqual(self.steps.picks, {("kit", "k", 0), ("kit", "s", 4), ("kit", "k", 8), ("kit", "H", 10)})
        self.assertTrue(self.key("a", "ctrl"))
        self.assertEqual((self.steps.picks, self.told), ({("kit", "k", 0), ("kit", "k", 8)}, []))
        # The two kicks are pasted in their own lane, whatever lane the mouse is on, the first at its row.
        self.assertTrue(self.key("c", "ctrl"))
        self.mouse("move", 2, 16)
        self.assertTrue(self.key("v", "ctrl"))
        self.assertEqual((self.told[-1][2], self.kicks(), self.steps.picks, self.levels()["k", 24.0]),
                         ("2 hits pasted", [0.0, 8.0, 16.0, 24.0], {("kit", "k", 16), ("kit", "k", 24)}, 7))
        # The second would land on the row of the hat: it is left out, and the first is pasted alone.
        self.mouse("move", 1, 2)
        self.key("v", "ctrl")
        self.assertEqual((self.told[-1][2], self.steps.picks), ("hit pasted", {("kit", "k", 2)}))
        # On the row of a kick, with the second past the end of the pattern, nothing is pasted and nothing told.
        self.mouse("move", 0, 24)
        self.assertTrue(self.key("v", "ctrl"))
        self.assertEqual((len(self.told), self.steps.picks), (2, {("kit", "k", 2)}))
        # With the mouse off the squares it is the first row in view: there the snare is in the way of the first.
        self.wheel(0, 0, -1, SHIFT)
        self.ui.handle(Event("move", (TALL[0] + 5, TALL[1] + 2)))
        self.key("v", "ctrl")
        self.assertEqual((self.steps.grid.first_row, self.told[-1][2], self.steps.picks),
                         (4, "hit pasted", {("kit", "k", 12)}))

    def test_a_drag_stays_in_the_lane_it_began_in(self) -> None:
        self.mouse("down", 0, 1, 1)
        self.mouse("move", 2, 3)
        self.mouse("up", 2, 3, 1)
        (channel, notes, label), = self.told
        self.assertEqual((channel, label), ("kit", "3 hits placed"))
        self.assertEqual([(note.drum, note.row) for note in notes],
                         [("k", 0.0), ("k", 1.0), ("k", 2.0), ("k", 3.0), ("s", 4.0), ("k", 8.0), ("H", 10.0)])

    def test_the_right_button_takes_hits_away_clicked_or_dragged_and_puts_none(self) -> None:
        self.assertTrue(self.mouse("down", 1, 0, 3))
        self.mouse("up", 1, 0, 3)
        self.assertEqual(self.told, [])
        self.click(2, 10, 3)
        # With the hat gone the kick before it has the rows to the end, of which one length says sixteen.
        self.assertEqual(self.told, [("kit", (KIT[0], KIT[1], dataclasses.replace(KIT[2], rows=16.0)), "hit removed")])
        # A drag that began on an empty square takes away as well.
        self.mouse("down", 0, 9, 3)
        self.mouse("move", 0, 0)
        self.mouse("up", 0, 0, 3)
        self.assertEqual(self.told[1:], [("kit", (KIT[1], KIT[3]), "2 hits removed")])
        self.assertEqual(self.heard, [])

    def test_the_velocity_lane_has_a_bar_for_each_hit_of_the_lane_pressed_in_last_and_a_drag_in_it_is_one_edit(
            self) -> None:
        lane, grid = self.steps.lane, self.steps.grid
        # Before any press it is the first lane: its two kicks, each under its square and as high as its level.
        self.assertEqual((self.steps.chosen, lane.span, lane.help, lane.items),
                         (0, (LEFT, grid.area[2]), VELOCITY_HELP,
                          ((grid.x(0), SQUARE_W, 1.0, False), (grid.x(8), SQUARE_W, 7 / 15, False))))
        # A press on the label of the snare makes its lane the one, and so does a press on a square of the hat's.
        label = (TALL[0] + 5, grid.y(1) + 5)
        self.assertTrue(self.ui.handle(Event("down", label, button=1)))
        self.ui.handle(Event("up", label, button=1))
        self.assertEqual((self.steps.chosen, lane.items), (1, ((grid.x(4), SQUARE_W, 9 / 15, False),)))
        self.click(2, 10)
        self.assertEqual((self.steps.chosen, [bar[0] for bar in lane.items], self.told), (2, [grid.x(10)], []))
        # A press in the velocity lane sets the hat to the height of the mouse: it is shown at once, and told
        # at the release, as one edit.
        at = (grid.x(10) + 1, velocity_bar(lane.rect, 0, 1, 5 / 15)[1])
        self.ui.handle(Event("down", at, button=1))
        self.assertEqual((self.levels()["H", 10.0], self.steps.busy, self.told), (5, True, []))
        self.ui.handle(Event("up", at, button=1))
        self.assertEqual((self.told, self.steps.busy, self.steps.level),
                         ([("kit", noteedit.with_level(KIT, [3], 5)[0], "velocity changed")], False, 5))
        # The steps show what they were given again, and the next hit is put at the level that was set last.
        self.assertEqual(self.levels()["H", 10.0], 12)
        self.click(0, 5)
        self.assertEqual(self.told[1], ("kit", noteedit.hit(KIT, 5, "k", 5, TABLE, 32.0), "hit placed"))
        # The right button on a bar is for the usual velocity, which is the top level.
        self.click(1, 4)
        at = (grid.x(4) + 1, lane.rect[1] + 20)
        self.ui.handle(Event("down", at, button=3))
        self.ui.handle(Event("up", at, button=3))
        self.assertEqual(self.told[2:], [("kit", noteedit.with_level(KIT, [1], 15)[0], "velocity changed")])

    def test_the_wheel_goes_along_the_lanes_with_shift_along_the_rows_and_with_ctrl_zooms_and_sets_no_level(
            self) -> None:
        # In this rectangle the third lane is cut, so the lanes scroll.
        self.steps.rect = RECT
        self.assertTrue(self.wheel(1, 4, -1))
        self.assertEqual(self.view(), (0, 1))
        self.assertTrue(self.wheel(1, 4, -1, SHIFT))
        self.assertEqual(self.view(), (4, 1))
        self.assertTrue(self.wheel(1, 4, 1, CTRL))
        self.assertEqual((self.view(), self.steps.grid.square_w), ((4, 1), SQUARE_W + 1))
        # It is so on a hit as on any square: no level is changed, nothing is told, and no gesture is under way.
        self.assertEqual((self.steps.notes("kit"), self.told, self.steps.busy), (KIT, [], False))
        # The zoom stays when another pattern is shown, and the bars of the velocity lane are as wide as a square.
        self.steps.show(kit(*KIT), LANES, 16, 4, TABLE)
        self.assertEqual((self.steps.grid.square_w, {bar[1] for bar in self.steps.lane.items}),
                         (SQUARE_W + 1, {SQUARE_W + 1}))

    def test_the_two_scroll_bars_move_the_view_and_the_plus_stands_under_the_last_lane_when_that_is_in_view(
            self) -> None:
        self.steps.rect = RECT
        grid = self.steps.grid
        along, across, _, plus = self.steps.parts
        self.assertEqual((along.rect, across.rect, along.help, across.help, plus.help),
                         (grid.along, grid.across, ALONG_HELP, ACROSS_HELP, PLUS_HELP))
        self.assertEqual((along.upright, across.upright, plus.visible), (True, False, False))
        # A press under the thumb of the bar along the lanes goes on by what is in view: here to the last lane.
        self.ui.handle(Event("down", (grid.along[0] + 1, BOTTOM - 2), button=1))
        self.ui.handle(Event("up", (grid.along[0] + 1, BOTTOM - 2), button=1))
        self.assertEqual((self.view(), plus.visible, plus.rect), ((0, 1), True, self.steps.grid.plus))
        # The thumb of the bar along the rows, dragged to its end, has the last row in view.
        x, y, _, _ = across.thumb()
        self.ui.handle(Event("down", (x + 2, y + 1), button=1))
        self.ui.handle(Event("move", (x + 2 + AREA_W, y + 1)))
        self.ui.handle(Event("up", (x + 2 + AREA_W, y + 1), button=1))
        self.assertEqual((self.view(), self.told), ((8, 1), []))
        # A press on the plus tells the owner, and taken off the window the steps take their parts with them.
        at = (plus.rect[0] + 3, plus.rect[1] + 3)
        self.ui.handle(Event("down", at, button=1))
        self.ui.handle(Event("up", at, button=1))
        self.assertEqual(self.added, ["drum"])
        self.steps.seen(False)
        self.assertEqual([part.visible for part in (self.steps, *self.steps.parts)], [False] * 5)
        self.steps.seen(True)
        self.assertEqual([part.visible for part in (self.steps, *self.steps.parts)], [True] * 5)

    def test_a_hit_put_by_a_press_where_another_letter_of_its_channel_has_one_takes_its_place_in_both_lanes_at_once(
            self) -> None:
        pick = f"a drag moves it   right press: rub it out{WHEEL}"
        self.mouse("move", 1, 4)
        self.assertEqual(self.ui.help, f"press: pick s at bar 1, step 5, {pick}")
        self.mouse("down", 0, 4, 1)
        seen = self.steps.notes("kit")
        self.assertEqual(seen, (KIT[0], Note(4.0, 4.0, 48, 15, "kick", "k"), KIT[2], KIT[3]))
        # With the button still down, the snare's square is empty and the kick's has the hit.
        self.mouse("move", 1, 4)
        self.assertEqual(self.ui.help, f"press: put s at bar 1, step 5, in place of k{WHEEL}")
        self.mouse("move", 0, 4)
        self.assertEqual(self.ui.help, f"press: pick k at bar 1, step 5, {pick}")
        self.mouse("up", 0, 4, 1)
        self.assertEqual(self.told, [("kit", seen, "hit placed")])

    def test_hits_of_two_channels_at_one_row_do_not_touch_each_other(self) -> None:
        kick, tops = beat(("k", 0, 15)), beat(("k", 4, 9), ("H", 2, 15))
        self.steps.show(band(kick=kick, tops=tops), PAIR, 16, 4, TABLE)
        self.mouse("down", 1, 0, 1)
        self.assertEqual((self.steps.notes("kick"), self.steps.notes("tops")),
                         (kick, noteedit.hit(tops, 0, "k", 15, TABLE, 32.0)))
        self.mouse("up", 1, 0, 1)
        self.click(0, 4)
        self.assertEqual(self.told, [("tops", noteedit.hit(tops, 0, "k", 15, TABLE, 32.0), "hit placed"),
                                     ("kick", noteedit.hit(kick, 4, "k", 15, TABLE, 32.0), "hit placed")])
        self.assertEqual(self.heard, [("tops", "k"), ("kick", "k")])

    def test_what_the_steps_show_during_a_drag_is_what_edited_gets_at_the_release_and_nothing_before(self) -> None:
        self.mouse("down", 0, 7, 1)
        self.mouse("move", 0, 5)
        seen = self.steps.notes("kit")
        self.assertEqual(([note.row for note in seen if note.drum == "k"], self.told),
                         ([0.0, 5.0, 6.0, 7.0, 8.0], []))
        # The release may be anywhere, also off the steps.
        self.ui.handle(Event("up", (0, 0), button=1))
        self.assertEqual(self.told, [("kit", seen, "3 hits placed")])
        self.assertEqual((self.steps.notes("kit"), self.ui.held), (KIT, None))

    def test_the_steps_are_busy_during_a_drag_and_escape_drops_it_then_the_pick_and_another_key_goes_to_the_owner(
            self) -> None:
        self.assertFalse(self.steps.busy)
        self.mouse("down", 0, 5, 1)
        self.assertTrue(self.steps.busy)
        self.mouse("up", 0, 5, 1)
        self.assertEqual((self.steps.busy, len(self.told)), (False, 1))
        # Escape drops a drag that is under way: the steps show what they were given, and its release tells nothing.
        self.click(0, 8)
        self.mouse("down", 0, 6, 1)
        self.mouse("move", 0, 7)
        self.assertTrue(self.key("escape"))
        self.assertEqual((self.steps.busy, self.steps.notes("kit"), self.steps.picks), (False, KIT, {("kit", "k", 8)}))
        self.mouse("up", 0, 7, 1)
        self.assertEqual(len(self.told), 1)
        # With no gesture under way escape drops what is picked. Any other key that is not the steps' goes on to
        # the owner, who may use it.
        keys: list[str] = []
        self.steps.typed = lambda event: keys.append(event.key) or event.key == "a"
        self.assertEqual([self.key(key) for key in ("escape", "a", "f1")], [True, True, False])
        self.assertEqual((self.steps.picks, keys), (frozenset(), ["a", "f1"]))

    def test_a_hit_is_heard_as_it_is_put_whatever_the_owner_answers_of_saving_it(self) -> None:
        self.steps.edited = lambda *said: self.told.append(said) or False
        self.mouse("down", 0, 5, 1)
        self.assertEqual((self.told, self.heard), ([], [("kit", "k")]))
        self.mouse("up", 0, 5, 1)
        # The owner that did not save shows no other notes, and the hit was heard once all the same.
        self.assertEqual((len(self.told), self.heard, self.steps.notes("kit")), (1, [("kit", "k")], KIT))

    def test_a_press_and_release_that_leave_the_notes_as_they_were_call_nothing(self) -> None:
        # The right button on empty squares, clicked and dragged.
        self.click(1, 0, 3)
        self.mouse("down", 1, 0, 3)
        self.mouse("move", 1, 3)
        self.mouse("up", 1, 3, 3)
        # A press in a gap and with the middle button begins nothing, and is not used. Neither is one of the right
        # button on a label, while there is no owner to open the menu of its lane.
        gap = (self.steps.grid.x(4) - 1, self.at(0, 4)[1])
        label = (TALL[0] + 5, self.steps.grid.y(2) + 5)
        for pos, button in ((gap, 1), (self.at(0, 4), 2), (label, 3)):
            self.assertFalse(self.ui.handle(Event("down", pos, button=button)))
            self.assertIsNone(self.ui.held)
            self.ui.handle(Event("up", pos, button=button))
        # With an owner the right button on a label asks for the menu of its lane, at the mouse, and picks nothing.
        asked: list[tuple[int, tuple[int, int]]] = []
        self.steps.asked = lambda place, pos: asked.append((place, pos))
        self.assertTrue(self.ui.handle(Event("down", label, button=3)))
        self.ui.handle(Event("up", label, button=3))
        self.assertEqual((asked, self.steps.chosen), ([(2, label)], 1))
        # While a drag is under way another button and the wheel do nothing to it.
        self.mouse("down", 1, 0, 3)
        self.mouse("down", 0, 0, 1)
        self.mouse("up", 0, 0, 1)
        self.wheel(1, 4, 1)
        self.mouse("up", 1, 0, 3)
        self.assertEqual((self.told, self.heard, self.steps.notes("kit")), ([], [], KIT))

    def test_each_hit_is_told_to_sounded_once_as_it_is_put_and_none_that_is_taken_away(self) -> None:
        self.mouse("down", 0, 1, 1)
        self.assertEqual(self.heard, [("kit", "k")])
        self.mouse("move", 0, 3)
        self.mouse("move", 0, 2)
        self.assertEqual(len(self.heard), 3)
        # Six squares more, of which one had its kick already, and one gets none as the snare has its row.
        self.mouse("move", 0, 9)
        self.mouse("up", 0, 9, 1)
        self.click(0, 8, 3)
        self.assertEqual(self.heard, [("kit", "k")] * 7)

    def test_a_pattern_off_the_grid_is_shown_and_not_changed_and_the_help_line_says_why(self) -> None:
        self.steps.show(kit(*KIT, hit("H", 20.5, 9)), LANES, 16, 4, TABLE)
        for button in (1, 3):
            self.assertFalse(self.mouse("down", 0, 8, button))
            self.mouse("up", 0, 8, button)
        # The wheel goes to the view, which in this rectangle has nowhere to go.
        self.assertFalse(self.wheel(1, 4, 1))
        self.assertEqual((self.told, self.heard, self.levels()["s", 4.0]), ([], [], 9))
        self.assertEqual(self.ui.help, OFF_GRID)
        self.mouse("move", 2, 20)
        self.assertEqual(self.ui.help, "this pattern has lengths the grid does not hold: it is shown and not changed")
        # The velocity lane takes nothing from the mouse either, and says why.
        self.assertEqual((self.steps.lane.enabled, self.steps.lane.help), (False, OFF_GRID))

    def test_without_the_drum_table_nothing_is_changed_and_showing_again_drops_a_drag(self) -> None:
        self.mouse("down", 0, 5, 1)
        self.assertNotEqual(self.steps.notes("kit"), KIT)
        self.steps.show(kit(*KIT), LANES, 16, 4, TABLE)
        self.assertEqual(self.steps.notes("kit"), KIT)
        self.mouse("move", 0, 7)
        self.mouse("up", 0, 7, 1)
        self.assertEqual(self.told, [])
        self.steps.show(kit(*KIT), LANES, 16, 4)
        self.assertFalse(self.mouse("down", 0, 5, 1))
        self.assertFalse(self.wheel(0, 8, 1))
        self.assertEqual((self.told, self.ui.help, self.steps.lane.enabled), ([], "the drum steps", False))
        # A lane of a letter the table does not have is refused, and so are the notes of a channel with no lane.
        with self.assertRaises(RowSynthError) as refused:
            self.steps.show(kit(*KIT), LANES, 16, 4, {"k": TABLE["k"]})
        self.assertEqual(str(refused.exception), "no drum letter 's'")
        with self.assertRaises(RowSynthError) as refused:
            self.steps.notes("lead")
        self.assertEqual(str(refused.exception), "the steps show no channel called 'lead'")

    def test_in_a_song_of_3_rows_to_a_beat_the_hits_that_are_put_are_ones_its_text_can_say(self) -> None:
        song = {"row_ticks": 6, "rows_per_beat": 3, "instruments": {"kick": {}, "hat": {}},
                "drums": {"k": {"inst": "kick", "note": "c3"}, "H": {"inst": "hat", "note": "c7", "rows": 1}},
                "channels": {"kit": {"inst": "kick", "drums": True}}, "patterns": {"a": {"kit": "r1"}},
                "order": ["a"]}

        def view() -> None:
            """Show the pattern as the song has it."""
            self.steps.show(songview.pattern(song, "a"), [Lane("kit", "k", "kick"), Lane("kit", "H", "hat")],
                            songview.bar_rows(song), song["rows_per_beat"], song["drums"])

        def save(channel: str, notes: tuple[Note, ...], label: str) -> None:
            """What an owner does: write the notes as the text of the song, and show what the song then has."""
            song["patterns"]["a"][channel] = songview.text(notes, 12.0, channel, song)
            view()

        self.steps.edited = save
        view()
        self.assertEqual((self.steps.grid.rows, self.steps.grid.bar_rows, self.steps.grid.beat_rows), (12, 12, 3))
        # Nine rows to the end are a dotted half note. Eight are one length only where a beat is four rows.
        self.click(0, 3)
        self.assertEqual([(note.drum, note.row, note.rows) for note in self.steps.notes("kit")], [("k", 3.0, 9.0)])
        self.mouse("down", 1, 6, 1)
        self.mouse("move", 1, 8)
        self.mouse("up", 1, 8, 1)
        self.click(0, 0)
        self.click(1, 7, 3)
        self.click(0, 4)
        self.assertEqual([(note.drum, note.row, note.rows) for note in self.steps.notes("kit")],
                         [("k", 0.0, 3.0), ("k", 3.0, 1.0), ("k", 4.0, 2.0), ("H", 6.0, 1.0), ("H", 8.0, 1.0)])
        self.assertEqual(songview.pattern(song, "a").channels[0].notes, self.steps.notes("kit"))

    def test_the_help_line_names_what_the_presses_and_the_wheel_do_on_the_square_under_the_mouse(self) -> None:
        pick = f"a drag moves it   right press: rub it out{WHEEL}"
        for drum, row, said in ((0, 20, f"press: put k at bar 2, step 5{WHEEL}"),
                                (1, 1, f"press: put s at bar 1, step 2{WHEEL}"),
                                (0, 8, f"press: pick k at bar 1, step 9, {pick}"),
                                (2, 10, f"press: pick H at bar 1, step 11, {pick}")):
            self.mouse("move", drum, row)
            self.assertEqual(self.ui.help, said)
        # In a gap it is the widget's own line.
        self.ui.handle(Event("move", (self.steps.grid.x(4) - 1, TOP + 3)))
        self.assertEqual(self.ui.help, "the drum steps")
        # With two drum channels the channel of a square that a hit would be put in is named as well.
        self.steps.show(band(kick=beat(("k", 0, 15)), tops=beat(("H", 2, 15))), PAIR, 16, 4, TABLE)
        for drum, row, said in ((0, 0, f"press: pick k at bar 1, step 1, {pick}"),
                                (1, 0, f"press: put k in tops at bar 1, step 1{WHEEL}")):
            self.mouse("move", drum, row)
            self.assertEqual(self.ui.help, said)

    def test_the_help_line_says_which_drum_a_hit_would_take_the_place_of(self) -> None:
        for drum, row, said in ((2, 0, "press: put H at bar 1, step 1, in place of k"),
                                (0, 4, "press: put k at bar 1, step 5, in place of s"),
                                (1, 10, "press: put s at bar 1, step 11, in place of H")):
            self.mouse("move", drum, row)
            self.assertEqual(self.ui.help, said + WHEEL)
        # A note of the channel that is no hit is in the way as well, though no lane shows it.
        self.steps.show(kit(*KIT, Note(20.0, 2.0, 60.0, 15, "kick")), LANES, 16, 4, TABLE)
        self.mouse("move", 0, 20)
        self.assertEqual(self.ui.help, f"press: put k at bar 2, step 5, in place of a note{WHEEL}")
        # With two drum channels it is the square's own channel that plays one drum, and the other is not in the way.
        self.steps.show(band(kick=beat(("k", 2, 15)), tops=beat(("H", 2, 15))), PAIR, 16, 4, TABLE)
        for drum, row, said in ((1, 2, f"press: put k in tops at bar 1, step 3, in place of H{WHEEL}"),
                                (0, 2, f"press: pick k at bar 1, step 3, a drag moves it   right press: rub it out"
                                       f"{WHEEL}")):
            self.mouse("move", drum, row)
            self.assertEqual(self.ui.help, said)


@unittest.skipUnless(pygame, NEEDS)
class Drawn(unittest.TestCase):
    SIZE = (330, 180)

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

    def dot_in(self, rect: layout.Rect, colour: tuple[int, ...]) -> bool:
        """Whether an empty square of a beat's colour is drawn with the dot in it. But for the dot it is as any."""
        x, y, w, h = rect
        fill, line = paint(colour, None)
        where, shade = dot(rect, colour)
        inside = block((x + 1, y + 1, w - 2, h - 2))
        found = self.inked(rect, shade)
        self.assertIn(found, (set(), block(where)))
        self.assertEqual((self.inked(rect, line), self.inked(rect, fill)), (block(rect) - inside, inside - found))
        return bool(found)

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

    def test_the_label_of_the_chosen_lane_has_a_strip_along_its_foot_and_what_is_under_the_mouse_is_lit(self) -> None:
        steps = Steps(TALL)
        steps.show(kit(*KIT), LANES, 16, 4, TABLE)
        steps.lane.colour = theme.SOUND
        grid = self.draw(steps)

        def strips() -> list[bool]:
            """Whether the label of each lane has the strip of the chosen one: inside its outline, along its foot."""
            rooms = [(x + 1, y + h - 1 - theme.STRIP, w - 2, theme.STRIP) for x, y, w, h in map(grid.label, range(3))]
            return [self.inked(room, theme.SOUND) == block(room) for room in rooms]

        self.assertEqual((steps.chosen, strips()), (0, [True, False, False]))
        # A press on the label of another lane takes the strip there.
        steps.on(Event("down", (TALL[0] + 3, grid.y(2) + 3), button=1), Ui())
        self.draw(steps)
        self.assertEqual((steps.chosen, strips()), (2, [False, False, True]))
        # Under the mouse an empty square is lit as the playhead lights one, and a label has a lighter outline.
        steps.hot = True
        steps.on(Event("move", (grid.x(1) + 2, grid.y(1) + 5)), Ui())
        self.draw(steps)
        self.assertEqual((self.square(grid.square(1, 1)), self.square(grid.square(1, 2))),
                         (paint(grid.colour(1), None, True), paint(grid.colour(2), None)))
        steps.on(Event("move", (TALL[0] + 3, grid.y(1) + 3)), Ui())
        self.draw(steps)
        self.assertEqual((self.at(TALL[0], grid.y(1)), self.at(TALL[0], grid.y(0)), self.square(grid.square(1, 1))),
                         (theme.DIM, theme.LINE, paint(grid.colour(1), None)))

    def test_a_faint_line_runs_between_the_lanes_of_one_channel_and_those_of_the_next_and_none_inside_one(
            self) -> None:
        steps = paired(RECT)
        grid = self.draw(steps)
        self.assertEqual((grid.y(1) - grid.y(0) - LANE_H, grid.y(2) - grid.y(1) - LANE_H),
                         (LANE_GAP + CHANNEL_GAP, LANE_GAP))
        # In the middle of the air between the two channels, from the side of the labels to the end of the squares.
        line = (RECT[0], grid.y(0) + LANE_H + (LANE_GAP + CHANNEL_GAP) // 2, LABELS_W + AREA_W, 1)
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

    def test_a_square_that_is_painted_is_drawn_filled_during_the_drag_and_as_it_was_given_after_it(self) -> None:
        ui = Ui()
        steps = ui.add(Steps(TALL))
        steps.show(kit(*KIT), LANES, 16, 4, TABLE)
        grid = steps.grid
        ui.handle(Event("down", (grid.x(2) + 2, grid.y(0) + 9), button=1))
        ui.handle(Event("move", (grid.x(5) + 2, grid.y(0) + 9)))
        self.draw(steps)
        # The hits the drag has put are outlined as the picked ones they are to be.
        self.assertEqual([self.square(grid.square(0, row)) for row in (1, 2, 3, 5, 6)],
                         [paint(theme.FAULT, None), (theme.FAULT, theme.TEXT), (theme.FAULT, theme.TEXT),
                          (theme.ACTIVE, theme.TEXT), paint(theme.ACTIVE, None)])
        # The snare keeps the fifth row: it is drawn as before, and the kick's square there is empty, with its dot.
        self.assertEqual(self.square(grid.square(1, 4)), paint(theme.ACTIVE, 9))
        self.assertTrue(self.dot_in(grid.square(0, 4), theme.ACTIVE))
        # The kicks that are painted are in the way in the snare's lane at once.
        self.assertEqual([self.dot_in(grid.square(1, row), grid.colour(row)) for row in (1, 2, 3, 5, 6)],
                         [False, True, True, True, False])
        ui.handle(Event("up", (grid.x(5) + 2, grid.y(0) + 9), button=1))
        self.draw(steps)
        self.assertEqual([self.square(grid.square(0, row)) for row in (2, 5)],
                         [paint(theme.FAULT, None), paint(theme.ACTIVE, None)])
        self.assertFalse(self.dot_in(grid.square(1, 2), theme.FAULT))

    def test_a_picked_hit_has_an_outline_in_the_colour_of_text_and_one_round_it_and_so_has_the_box(self) -> None:
        ui = Ui()
        steps = ui.add(Steps(TALL))
        steps.show(kit(*beat(("k", 1, 15), ("H", 14, 15))), LANES, 16, 4, TABLE)
        grid = steps.grid
        kick, hat = grid.square(0, 1), grid.square(2, 14)

        def ring(square: layout.Rect) -> set[tuple[int, ...]]:
            """The colours of the pixels right round a square."""
            x, y, w, h = square
            return {self.at(*pos) for pos in block((x - 1, y - 1, w + 2, h + 2)) - block(square)}

        def press(pos: tuple[int, int], mods: frozenset[str] = frozenset()) -> None:
            for kind in ("down", "up"):
                ui.handle(Event(kind, pos, button=1, mods=mods))

        self.draw(steps)
        # The hat is in the last beat, whose colour is that of text: its own outline could not tell it as picked.
        self.assertEqual((self.square(kick), ring(kick), self.square(hat), ring(hat)),
                         (paint(theme.FAULT, 15), {theme.PANEL}, (theme.TEXT, theme.TEXT), {theme.PANEL}))
        press((kick[0] + 2, kick[1] + 9))
        self.draw(steps)
        self.assertEqual((self.square(kick), ring(kick), ring(hat)), ((theme.FAULT, theme.TEXT), {theme.TEXT},
                                                                     {theme.PANEL}))
        press((hat[0] + 2, hat[1] + 9), CTRL)
        self.draw(steps)
        self.assertEqual((ring(kick), ring(hat)), ({theme.TEXT}, {theme.TEXT}))
        # A box that is being drawn has an outline of that colour, and the hits are shown as it would pick them.
        start, end = (grid.x(4) + 2, grid.y(1) + 9), (grid.x(6) + 2, grid.y(2) + 3)
        ui.handle(Event("down", start, button=1, mods=SHIFT))
        ui.handle(Event("move", end, mods=SHIFT))
        self.draw(steps)
        x, y, w, h = (*start, end[0] - start[0] + 1, end[1] - start[1] + 1)
        self.assertEqual({self.at(*pos) for pos in block((x, y, w, h)) - block((x + 1, y + 1, w - 2, h - 2))},
                         {theme.TEXT})
        self.assertEqual((ring(kick), ring(hat)), ({theme.PANEL}, {theme.PANEL}))
        ui.handle(Event("up", end, button=1, mods=SHIFT))
        self.draw(steps)
        self.assertEqual((self.at(*start), steps.picks), (paint(theme.ACTIVE, None)[0], frozenset()))

    def test_a_dot_is_in_an_empty_square_whose_row_another_drum_of_its_channel_has_and_in_no_other(self) -> None:
        self.assertEqual(dot((10, 20, SQUARE_W, LANE_H), theme.FAULT),
                         ((12, 34, DOT_W, DOT_H), mix(theme.BACKGROUND, theme.FAULT, DOT)))
        steps = Steps(TALL)
        steps.show(kit(*KIT), LANES, 16, 4, TABLE)
        grid = self.draw(steps)
        # The first kick is in the way in the two other lanes, the snare in the kick's and the hat's, and so on.
        for drum, row in ((1, 0), (2, 0), (0, 4), (2, 4), (1, 8), (0, 10)):
            self.assertTrue(self.dot_in(grid.square(drum, row), grid.colour(row)))
        # No hit of the channel is at the row of these, and a square with a hit has no dot.
        for drum, row in ((0, 1), (1, 1), (2, 1), (0, 20), (2, 23)):
            self.assertFalse(self.dot_in(grid.square(drum, row), grid.colour(row)))
        self.assertEqual([self.square(grid.square(drum, row)) for drum, row in ((0, 0), (1, 4))],
                         [paint(theme.FAULT, 15), paint(theme.ACTIVE, 9)])
        # Where the playhead is the dot is lighter, as its square is.
        steps.playhead = 4.0
        grid = self.draw(steps)
        x, y, _, _ = grid.square(2, 4)
        self.assertEqual((self.at(x + 2, y + 14), self.at(x + 3, y + 15), self.at(x + 2, y + 13)),
                         (*[mix(mix(theme.BACKGROUND, theme.ACTIVE, DOT), theme.TEXT, GLOW)] * 2,
                          paint(theme.ACTIVE, None, True)[0]))

    def test_no_dot_is_in_a_channel_of_one_letter_nor_where_the_hits_cannot_be_changed(self) -> None:
        # Two channels of one letter each, with hits at the same row and at rows of their own.
        steps = Steps(TALL)
        steps.show(band(kick=beat(("k", 0, 15), ("k", 4, 15)), tops=beat(("k", 4, 9), ("k", 6, 9))),
                   [Lane("kick", "k", "kick"), Lane("tops", "k", "kick")], 16, 4, TABLE)
        grid = self.draw(steps)
        for drum, row in ((1, 0), (0, 6), (0, 1), (1, 1)):
            self.assertFalse(self.dot_in(grid.square(drum, row), grid.colour(row)))
        # Without the drum table no hit can be put, and neither can one in a pattern that is off the grid.
        for notes, table in ((KIT, None), ((*KIT, hit("H", 20.5, 9)), TABLE)):
            steps.show(kit(*notes), LANES, 16, 4, table)
            grid = self.draw(steps)
            self.assertFalse(self.dot_in(grid.square(1, 0), theme.FAULT))

    def test_the_drums_of_the_first_pattern_of_the_arcade_example(self) -> None:
        song = Project.load(EXAMPLES / "arcade.json").song("tune")
        found = songview.pattern(song, "a")
        made = lanes([songview.pattern(song, name) for name in song["order"]], songview.drum_rows(song))
        self.assertEqual(made, [Lane("kit", "k", "kick"), Lane("kit", "s", "snare"), Lane("kit", "h", "hat")])
        steps = Steps((0, 0, 660 + BAR, 120 + BAR))
        steps.show(found, made, songview.bar_rows(song), song.get("rows_per_beat", 4))
        self.surface = self.canvas.surface = pygame.Surface((660 + BAR, 120 + BAR))
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
