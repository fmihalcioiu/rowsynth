"""Tests for rowsynth.gui.strips: small notes, the sums of the song strip, its pick, view and order, its drawing."""
from __future__ import annotations

import os
import unittest
from collections.abc import Callable, Sequence
from pathlib import Path

from rowsynth.errors import RowSynthError
from rowsynth.gui import layout, songedit, songview, sprites, theme
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.font import Font
from rowsynth.gui.songview import Block, Note
from rowsynth.gui.strips import (BAR_H, DOUBLE, EDGE_PX, EDGE_W, NAME_H, ROW_W_MIN, SLACK, WHEEL_PX, Map, SongStrip,
                                 places, small)
from rowsynth.gui.ui import Event, Ui, share
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

#: A rectangle for small notes: 64 pixels across and 9 high, so 2 pixels to a row of a pattern of two bars.
SMALL = (10, 20, 64, 9)
#: A strip whose blocks are 54 high, over the line of the picked one at y 74 and the bar at y 77.
RECT = (10, 20, 320, 60)
#: The strip the examples are shown in, and one that the trailer piece does not fit: it is 528 rows long.
WIDE = (10, 20, 660, 60)
NARROW = (10, 20, 200, 60)
#: The surface a strip is drawn on, which is larger than it on every side, and the colour of what is not its.
SIZE = (340, 100)
OUTSIDE = theme.GOOD
#: What the help line says on a block, after the name of its pattern.
PRESS = "ctrl to copy; right button for more"
CTRL = frozenset({"ctrl"})

INSTRUMENTS = {"lead": {"vol": 12}, "bass": {"wave": "tri"}, "thud": {"wave": "noise"}}
DRUMS = {"k": {"inst": "thud", "note": "c2"}, "h": {"inst": "thud", "note": "c6"}}
#: A pattern of two bars, two of one bar, and one of a bar and a half.
PATTERNS = {"a": {"lead": "o4 l4 c e g e | >c< g e c", "low": "o2 l2 c g | c1", "kit": "l4 k h k h | k h k h"},
            "b": {"lead": "o4 g1", "kit": "l4 k k k k"},
            "bridge": {"lead": "o4 g1"},
            "c": {"lead": "o4 l4 c c c c c c"}}

#: The song of two bars, one bar and two bars again as blocks, and one of forty blocks of two bars.
BLOCKS = (Block("a", 0.0, 32.0), Block("b", 32.0, 16.0), Block("a", 48.0, 32.0))
LONG = tuple(Block("a", 32.0 * index, 32.0) for index in range(40))


def song(*order: str, **patterns: dict) -> dict:
    """A song of a lead, a low channel and a drum channel that plays those patterns. Any given replace its own."""
    return {"row_ticks": 6, "instruments": INSTRUMENTS, "drums": DRUMS,
            "channels": {"lead": {"inst": "lead"}, "low": {"inst": "bass"}, "kit": {"inst": "thud", "drums": True}},
            "patterns": {**PATTERNS, **patterns}, "order": list(order)}


#: A song of four patterns of unequal length: two bars, one bar, a bar and a half, and half a bar. Its blocks in a
#: strip 320 wide are 4 pixels to the row, and its gaps are at x 10, 137, 201, 297 and 329.
FOUR = song("a", "b", "c", "d", d={"lead": "o4 l4 c c"})
FOUR_BLOCKS = (Block("a", 0.0, 32.0), Block("b", 32.0, 16.0), Block("c", 48.0, 24.0), Block("d", 72.0, 8.0))
FOUR_GAPS = (10, 137, 201, 297, 329)
#: The place each block of four is at after a move to each of the five gaps, and None where it stays.
TO = ((None, None, 1, 2, 3), (0, None, None, 2, 3), (0, 1, None, None, 3), (0, 1, 2, None, None))


def put(order: Sequence[str], place: int, gap: int, copy: str = "") -> list[str]:
    """The order that a line in a gap promises: the block of a place stands in the gap, and the rest as they were.

    With `copy`, the block stays where it is, and a pattern of that name stands in the gap.
    """
    rest = [(index, name) for index, name in enumerate(order) if copy or index != place]
    return ([name for index, name in rest if index < gap] + [copy or order[place]]
            + [name for index, name in rest if index >= gap])


def note(row: float, rows: float, pitch: float) -> Note:
    """A note that is no drum hit."""
    return Note(float(row), float(rows), pitch, 15, "lead")


def hit(letter: str, row: float, pitch: float, rows: float = 1) -> Note:
    """A hit of a drum letter that plays a pitch."""
    return Note(float(row), float(rows), float(pitch), 15, "thud", letter)


def painted(made: dict, seen: Map, colour: theme.Colour) -> set[tuple[int, int]]:
    """The pixels the notes in one colour take in a strip: of each channel that has it, in every block.

    A channel stands between its lowest and its highest note in the whole song.
    """
    read = {name: songview.pattern(made, name) for name in made["order"]}
    count = len(made["channels"])
    pixels = set()
    for lane in range(count):
        if theme.CHANNELS[lane % len(theme.CHANNELS)] != colour:
            continue
        tuned = [found.pitch for pattern in read.values() for found in pattern.channels[lane].notes if not found.drum]
        low, high = (min(tuned), max(tuned)) if tuned else (None, None)
        for index, block in enumerate(seen.blocks):
            held = read[block.name]
            for x, y, w, h in places(seen.lanes(index, count)[lane], held.channels[lane].notes, held.rows, low, high):
                pixels |= {(x + column, y + row) for column in range(w) for row in range(h)}
    return pixels


class Small(unittest.TestCase):
    def test_a_note_is_a_line_at_the_height_of_its_pitch_from_its_start_to_a_pixel_short_of_its_end(self) -> None:
        notes = (note(0, 4, 60), note(4, 4, 72), note(8, 2, 66), note(10, 1, 63))
        # The lowest is on the lowest line and the highest on the top one, and each leaves a pixel before the next.
        self.assertEqual(places(SMALL, notes, 32), [(10, 28, 7, 1), (18, 20, 7, 1), (26, 24, 3, 1), (30, 26, 1, 1)])
        # The same notes in a pattern twice as long are half as far apart.
        self.assertEqual(places(SMALL, notes[:2], 64), [(10, 28, 3, 1), (14, 20, 3, 1)])
        # A range that is given places them in it. With one end given, the other is found.
        self.assertEqual([y for _, y, _, _ in places(SMALL, notes[:3], 32, 48, 96)], [26, 24, 25])
        self.assertEqual([y for _, y, _, _ in places(SMALL, notes[:3], 32, low=48)], [24, 20, 22])
        # A pitch outside the range stands at its edge.
        self.assertEqual([y for _, y, _, _ in places(SMALL, (note(0, 1, 40), note(1, 1, 100)), 32, 48, 96)], [28, 20])

    def test_a_single_pitch_is_in_the_middle_and_no_room_or_no_rows_gives_no_places(self) -> None:
        self.assertEqual(places(SMALL, (note(0, 4, 60), note(4, 4, 60)), 32), [(10, 24, 7, 1), (18, 24, 7, 1)])
        self.assertEqual(places(SMALL, (note(0, 4, 60),), 32, 60, 60), [(10, 24, 7, 1)])
        for rect, rows in (((10, 20, 0, 9), 32), ((10, 20, 64, 0), 32), (SMALL, 0)):
            self.assertEqual(places(rect, (note(0, 4, 60),), rows), [])
        self.assertEqual(places(SMALL, (), 32), [])

    def test_a_note_shorter_than_a_pixel_is_a_pixel_long(self) -> None:
        # Two rows to the pixel: a note of a row, one that ends in the pixel it starts in, and a quarter of a row.
        short = (note(3, 1, 60), note(4, 1, 60), note(5, 0.25, 60))
        self.assertEqual(places(SMALL, short, 128), [(11, 24, 1, 1), (12, 24, 1, 1), (12, 24, 1, 1)])

    def test_a_drum_hit_is_an_upright_tick_in_the_lane_of_its_letter(self) -> None:
        kit = (hit("k", 0, 36), hit("h", 4, 84), hit("s", 8, 60, rows=8))
        # The letter that plays the lowest note has the lowest lane, and how long a hit sounds is not shown.
        self.assertEqual(places(SMALL, kit, 32), [(10, 27, 1, 2), (18, 20, 1, 2), (26, 23, 1, 2)])
        # One letter alone is in the middle, and two that play the same note have a lane each.
        self.assertEqual(places(SMALL, kit[:1], 32), [(10, 23, 1, 2)])
        self.assertEqual(places(SMALL, (hit("h", 0, 105), hit("H", 2, 105)), 32), [(10, 20, 1, 2), (14, 27, 1, 2)])
        # A range moves no tick, and a tuned note among the hits stands by the tuned notes alone.
        self.assertEqual(places(SMALL, kit, 32, 0, 127), places(SMALL, kit, 32))
        self.assertEqual(places(SMALL, (*kit, note(16, 2, 50)), 32)[3], (42, 24, 3, 1))
        # Where the lanes have not the room, a tick is one pixel high.
        self.assertEqual(places((10, 20, 64, 3), kit, 32), [(10, 22, 1, 1), (18, 20, 1, 1), (26, 21, 1, 1)])


class Sums(unittest.TestCase):
    def test_a_song_that_fits_fills_the_width_and_a_block_is_as_wide_as_its_rows(self) -> None:
        seen = Map(RECT, BLOCKS, 80.0)
        self.assertEqual((seen.width, seen.most, seen.block_h), (320, 0, 54))
        self.assertEqual([seen.x(row) for row in (0, 32, 40.5, 80)], [10, 138, 172, 330])
        # A pixel is left free after each block.
        self.assertEqual([seen.block(index) for index in range(3)],
                         [(10, 20, 127, 54), (138, 20, 63, 54), (202, 20, 127, 54)])
        self.assertEqual((seen.in_view(), [seen.sees(row) for row in (0, 79.5, 80)]),
                         ([0, 1, 2], [True, True, False]))
        # A pattern with no rows is a block with no width, and a strip with no song has every row at its left edge.
        bare = Map(RECT, (Block("a", 0.0, 0.0), Block("b", 0.0, 80.0)), 80.0)
        self.assertEqual((bare.block(0), bare.block(1), bare.in_view()), ((10, 20, 0, 54), (10, 20, 319, 54), [1]))
        self.assertEqual((Map(RECT).x(5), Map(RECT).in_view(), Map(RECT).most), (10, [], 0))

    def test_rows_that_would_be_narrower_than_the_least_keep_it_and_the_view_moves_along_the_song(self) -> None:
        seen = Map(RECT, LONG, 1280.0)
        self.assertEqual((seen.width, seen.most, ROW_W_MIN), (1280, 960, 1))
        self.assertEqual((seen.block(0), seen.block(1), seen.x(1280)), ((10, 20, 31, 54), (42, 20, 31, 54), 1290))
        moved = seen.scrolled(100)
        self.assertEqual((moved.offset, moved.block(3), moved.x(100)), (100, (6, 20, 31, 54), 10))
        # It stops at the start of the song and at its end.
        self.assertEqual((seen.scrolled(-50).offset, moved.scrolled(5000).offset, seen.scrolled(5000).x(1280)),
                         (0, 960, 330))
        # Following a row puts it at the left edge, or as near to it as the end of the song lets it.
        self.assertEqual([moved.following(row).offset for row in (500.5, 1270, 3)], [501, 960, 3])
        # Half a row more than the strip is wide is a song that scrolls by one pixel.
        self.assertEqual(Map(RECT, (Block("a", 0.0, 320.5),), 320.5).most, 1)
        # In view is each block with a pixel in the strip, and each row whose x is in it.
        self.assertEqual((seen.in_view(), seen.scrolled(100).in_view()), (list(range(10)), list(range(3, 14))))
        # The fourth block ends on the pixel before the one that a view from 127 starts on.
        self.assertEqual((seen.scrolled(126).in_view()[0], seen.scrolled(127).in_view()[0]), (3, 4))
        self.assertEqual([seen.scrolled(100).sees(row) for row in (99, 100, 419, 420)], [False, True, True, False])

    def test_the_thumb_of_the_bar_is_to_the_strip_what_the_view_is_to_the_song(self) -> None:
        seen = Map(RECT, LONG, 1280.0)
        self.assertEqual([seen.scrolled(by).thumb() for by in (0, 100, 960)],
                         [(10, 77, 80, 3), (35, 77, 80, 3), (250, 77, 80, 3)])
        self.assertIsNone(Map(RECT, BLOCKS, 80.0).thumb())
        # It is never shorter than the thumb of a list.
        self.assertEqual(Map(RECT, (Block("a", 0.0, 32000.0),), 32000.0).scrolled(16000).thumb(),
                         (168, 77, theme.LIST_THUMB_MIN, BAR_H))

    def test_a_name_is_whole_where_it_fits_cut_where_it_does_not_and_left_out_where_no_letter_would_show(self) -> None:
        blocks = (Block("chorus", 0.0, 45.0), Block("chorus", 45.0, 33.0), Block("chorus", 78.0, 27.0),
                  Block("chorus", 105.0, 25.0), Block("a", 130.0, 15.0), Block("a", 145.0, 13.0),
                  Block("z", 158.0, 242.0))
        seen = Map(RECT, blocks, 400.0)
        self.assertEqual([seen.block(index)[2] for index in range(6)], [44, 32, 26, 24, 14, 12])
        self.assertEqual([seen.name(index) for index in range(6)], ["chorus", "ch..", "c..", "", "a", ""])

    def test_the_channels_share_the_room_under_the_name_of_a_block_as_lanes(self) -> None:
        seen = Map(RECT, BLOCKS, 80.0)
        # A lane of 4 pixels or more leaves its lowest line free, and the notes keep 2 pixels from the sides.
        self.assertEqual(seen.lanes(0, 3), [(12, 32, 123, 12), (12, 45, 123, 13), (12, 59, 123, 12)])
        self.assertEqual(seen.lanes(1, 1), [(140, 32, 59, 39)])
        # Eighteen channels have 2 or 3 pixels each, and those leave none free.
        many = seen.lanes(1, 18)
        self.assertEqual([height for _, _, _, height in many], share(40, [1] * 18))
        self.assertEqual((many[0][1], many[-1][1] + many[-1][3]), (32, 72))
        self.assertTrue(all(below[1] == above[1] + above[3] for above, below in zip(many, many[1:])))
        # No channels are no lanes, and a strip too low for the name has lanes with no height.
        self.assertEqual(seen.lanes(0, 0), [])
        self.assertEqual([height for _, _, _, height in Map((10, 20, 320, 12), BLOCKS, 80.0).lanes(0, 3)], [0, 0, 0])

    def test_a_gap_is_at_the_pixel_left_free_before_its_block_and_the_first_at_the_start_of_the_song(self) -> None:
        seen = Map(RECT, FOUR_BLOCKS, 80.0)
        # Four blocks have five gaps. The last is at the pixel left free after the last block.
        self.assertEqual(tuple(seen.gap_x(gap) for gap in range(5)), FOUR_GAPS)
        self.assertEqual([seen.block(index)[0] for index in range(4)], [10, 138, 202, 298])
        # They go with the view, also out of it.
        moved = Map(RECT, LONG, 1280.0).scrolled(100)
        self.assertEqual([moved.gap_x(gap) for gap in (0, 3, 4, 40)], [-90, 5, 37, 1189])
        # A strip with no song has the one gap, at its left edge.
        self.assertEqual(Map(RECT).gap_x(0), 10)
        # The square of the plus is at the top of the line and in the middle of it, and at an end of the strip
        # beside it.
        self.assertEqual([seen.plus(gap) for gap in range(5)],
                         [(10, 20, 7, 7), (134, 20, 7, 7), (198, 20, 7, 7), (294, 20, 7, 7), (323, 20, 7, 7)])
        self.assertEqual([moved.plus(gap) for gap in (3, 4)], [(10, 20, 7, 7), (34, 20, 7, 7)])

    def test_the_gap_nearest_a_position_is_found_by_its_x_and_of_two_that_are_as_near_it_is_the_earlier(self) -> None:
        seen = Map(RECT, FOUR_BLOCKS, 80.0)
        # The left half of a block is nearest the gap before it, and so is its middle pixel: 73, 169, 249 and 313.
        self.assertEqual([seen.gap_at((x, 40)) for x in (10, 73, 74, 137, 169, 170, 249, 250, 313, 314, 329)],
                         [0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4])
        # How high the position is makes no difference, and one off the strip to a side is taken for its edge.
        self.assertEqual([seen.gap_at(pos) for pos in ((170, -500), (170, 900), (-50, 40), (900, 40))], [2, 2, 0, 4])
        # In a view that has moved, the gaps are at x 5, 37 and so on to 325: none far out of view is found.
        moved = Map(RECT, LONG, 1280.0).scrolled(100)
        self.assertEqual([moved.gap_at((x, 40)) for x in (-500, 10, 21, 22, 329, 900)], [3, 3, 3, 4, 13, 13])
        self.assertEqual(Map(RECT).gap_at((200, 40)), 0)

    def test_to_is_the_place_a_block_is_at_after_it_is_moved_to_a_gap_and_none_where_it_stays(self) -> None:
        self.assertEqual(tuple(tuple(Map.to(place, gap) for gap in range(5)) for place in range(4)), TO)
        # It is the same of a map as of the class, whatever the map holds.
        self.assertEqual((Map(RECT).to(2, 0), Map(RECT, LONG, 1280.0).to(7, 30)), (0, 29))

    def test_the_rectangle_of_a_name_is_the_band_of_its_block_less_what_is_kept_free_at_its_sides(self) -> None:
        seen = Map(RECT, FOUR_BLOCKS, 80.0)
        self.assertEqual([seen.name_rect(index) for index in range(4)],
                         [(14, 20, 119, 11), (142, 20, 55, 11), (206, 20, 87, 11), (302, 20, 23, 11)])
        self.assertEqual(seen.name_rect(1)[3], NAME_H)
        # A block too narrow for what is kept free has a rectangle with no width.
        self.assertEqual(Map(RECT, (Block("a", 0.0, 1.0), Block("b", 1.0, 79.0)), 80.0).name_rect(0), (14, 20, 0, 11))


class Strip(unittest.TestCase):
    def setUp(self) -> None:
        self.picks: list[int] = []
        self.ui = Ui()
        self.strip = self.ui.add(SongStrip(RECT, self.picks.append, "the song"))
        self.strip.show(song("a", "b", "a"))

    def press(self, pos: tuple[int, int], button: int = 1) -> bool:
        """Press a button of the mouse at a position and let it go. Whether the press was used."""
        used = self.ui.handle(Event("down", pos, button=button))
        self.ui.handle(Event("up", pos, button=button))
        return used

    def wheel(self, steps: int, pos: tuple[int, int] = (150, 40)) -> bool:
        """Turn the wheel with the mouse at a position. Whether it was used."""
        return self.ui.handle(Event("wheel", pos, wheel=steps))

    def state(self) -> tuple[int | None, int]:
        """The pick of the strip and its view."""
        return self.strip.picked, self.strip.offset

    def test_a_click_on_a_block_picks_it_and_tells_and_no_key_is_the_strips(self) -> None:
        self.assertEqual((self.strip.picked, self.picks), (0, []))
        self.assertTrue(self.press((150, 40)))
        self.assertEqual((self.strip.picked, self.picks), (1, [1]))
        # It tells also when the block was picked already.
        self.press((150, 70))
        self.assertEqual(self.picks, [1, 1])
        # Between two blocks, under them, and with the middle button, nothing is picked.
        self.assertEqual([self.press((137, 40)), self.press((250, 75)), self.press((250, 40), button=2)], [False] * 3)
        self.assertEqual((self.strip.picked, self.picks), (1, [1, 1]))
        for key in ("left", "right"):
            self.assertFalse(self.ui.handle(Event("key", key=key)))
        self.assertEqual(self.strip.picked, 1)
        # A strip that tells nobody picks all the same.
        quiet = SongStrip(RECT)
        quiet.show(song("a", "b"))
        self.assertEqual((quiet.on(Event("down", (300, 40), button=1), self.ui), quiet.picked), (True, 1))

    def test_the_wheel_moves_only_a_song_that_does_not_fit(self) -> None:
        self.assertEqual((self.wheel(-1), self.wheel(1), self.strip.offset), (False, False, 0))
        self.strip.show(song(*["a"] * 40))
        self.assertEqual((self.wheel(-1), self.strip.offset), (True, WHEEL_PX))
        self.wheel(-2)
        self.wheel(1)
        self.assertEqual(self.strip.offset, 2 * WHEEL_PX)
        # It stops at the end and at the start, and is still the strip's there.
        self.wheel(-100)
        self.assertEqual((self.wheel(-1), self.strip.offset), (True, 960))
        self.wheel(100)
        self.assertEqual(self.strip.offset, 0)
        # Off the strip it is not.
        self.assertEqual((self.wheel(-1, pos=(150, 10)), self.strip.offset), (False, 0))

    def test_the_view_goes_after_a_playhead_that_leaves_it_and_stays_where_the_wheel_put_it(self) -> None:
        self.strip.show(song(*["a"] * 40))
        self.strip.playhead = 10.0
        self.strip.playhead = 319.0
        self.assertEqual(self.strip.offset, 0)
        # Past the right edge: the view starts at the playhead.
        self.strip.playhead = 330.5
        self.assertEqual((self.strip.offset, self.strip.map.x(330.5), self.strip.playhead), (331, 10, 330.5))
        # Round to the start of the song.
        self.strip.playhead = 2.0
        self.assertEqual(self.strip.offset, 2)
        # The wheel moved the view away from it: it is gone after only when it has been in view again.
        self.wheel(-3)
        self.strip.playhead = 3.0
        self.assertEqual(self.strip.offset, 194)
        self.strip.playhead = 200.0
        self.strip.playhead = 520.0
        self.assertEqual(self.strip.offset, 520)
        # The view goes no further than the end of the song.
        self.strip.playhead = 700.0
        self.strip.playhead = 1270.0
        self.assertEqual(self.strip.offset, 960)
        # Stopped, and started somewhere out of view: the playhead did not leave the view.
        self.strip.playhead = None
        self.strip.playhead = 5.0
        self.assertEqual(self.strip.offset, 960)
        # A song that fits has its playhead in view all the time.
        self.strip.show(song("a", "b", "a"))
        self.strip.playhead = 79.0
        self.assertEqual(self.strip.offset, 0)

    def test_block_at_is_the_block_a_position_is_on_and_none_between_two_under_them_and_off_the_strip(self) -> None:
        self.assertEqual([self.strip.block_at(pos) for pos in ((10, 20), (136, 73), (138, 40), (328, 40))],
                         [0, 0, 1, 2])
        off = ((137, 40), (329, 40), (50, 74), (50, 79), (9, 40), (50, 19), (50, 80))
        self.assertEqual([self.strip.block_at(pos) for pos in off], [None] * 7)
        self.strip.show(song(*["a"] * 40))
        self.strip.offset = 100
        self.assertEqual([self.strip.block_at(pos) for pos in ((10, 40), (36, 40), (37, 40), (38, 40), (329, 40))],
                         [3, 3, None, 4, 13])
        self.assertIsNone(SongStrip(RECT).block_at((50, 40)))

    def test_the_help_line_says_what_a_press_on_the_block_under_the_mouse_would_do(self) -> None:
        self.assertEqual((self.strip.help, self.ui.help), ("the song", ""))
        for pos, line in (((20, 30), f"drag to move a, {PRESS}"), ((150, 40), f"drag to move b, {PRESS}"),
                          ((300, 73), "drag to move a, ctrl to copy; right button for more"),
                          ((137, 40), "the song"), ((150, 75), "the song")):
            self.ui.handle(Event("move", pos))
            self.assertEqual(self.ui.help, line)
        # While a press on a block is under way the line is that block's, whatever the mouse is on.
        self.ui.handle(Event("down", (150, 40), button=1))
        for pos in ((300, 40), (137, 40)):
            self.ui.handle(Event("move", pos))
            self.assertEqual(self.ui.help, f"drag to move b, {PRESS}")
        self.ui.handle(Event("up", (137, 40), button=1))
        self.assertEqual(self.ui.help, "the song")
        self.strip.help = "another line"
        self.ui.handle(Event("move", (137, 40)))
        self.assertEqual(self.ui.help, "another line")

    def test_the_same_song_shown_again_keeps_the_pick_and_the_view_and_another_song_starts_anew(self) -> None:
        self.strip.show(song(*["a"] * 40))
        self.assertEqual(self.state(), (0, 0))
        self.wheel(-2)
        self.press((200, 40))
        self.assertEqual(self.state(), (9, 128))
        # The file changed outside: the same order, with other notes.
        self.strip.show(song(*["a"] * 40, a={"lead": "o5 l1 c c"}))
        self.assertEqual(self.state(), (9, 128))
        # Its pattern come back a bar long leaves less to move along: the view is kept inside the song.
        self.wheel(-100)
        self.strip.show(song(*["a"] * 40, a={"lead": "o5 c1"}))
        self.assertEqual((self.state(), self.strip.map.most), ((9, 320), 320))
        # Another order is another song: its first block is picked, and the view is at its start.
        self.strip.show(song(*["a"] * 39, "b"))
        self.assertEqual(self.state(), (0, 0))
        # A song that cannot be read is refused, and the one before is still shown.
        self.press((150, 40))
        with self.assertRaises(RowSynthError) as refused:
            self.strip.show(song("a", "x"))
        self.assertEqual(str(refused.exception), "order: no pattern called 'x'")
        self.assertEqual((self.state(), len(self.strip.map.blocks)), ((4, 0), 40))
        # A song with nothing in its order has no block to pick.
        self.strip.show(song())
        self.assertEqual((self.state(), self.strip.map.blocks), ((None, 0), ()))


class Hand(unittest.TestCase):
    def setUp(self) -> None:
        self.told: list[tuple] = []
        self.now = 0.0
        self.ui = Ui()
        self.strip = self.ui.add(SongStrip(RECT, self.tell("changed"), "the song", self.tell("moved"),
                                           self.tell("copied"), self.tell("asked"), self.tell("named"),
                                           lambda: self.now))
        self.strip.show(FOUR)

    def tell(self, what: str) -> Callable[..., None]:
        """A function for the strip to call, which notes that it was called, and what with."""
        return lambda *more: self.told.append((what, *more))

    def down(self, pos: tuple[int, int], button: int = 1, mods: frozenset[str] = frozenset()) -> bool:
        """Press a button of the mouse at a position. Whether the press was used."""
        return self.ui.handle(Event("down", pos, button=button, mods=mods))

    def move(self, pos: tuple[int, int], mods: frozenset[str] = frozenset()) -> None:
        """Take the mouse to a position."""
        self.ui.handle(Event("move", pos, mods=mods))

    def up(self, pos: tuple[int, int], button: int = 1, mods: frozenset[str] = frozenset()) -> None:
        """Let a button of the mouse go at a position."""
        self.ui.handle(Event("up", pos, button=button, mods=mods))

    def click(self, pos: tuple[int, int], button: int = 1) -> bool:
        """Press a button of the mouse at a position and let it go. Whether the press was used."""
        used = self.down(pos, button)
        self.up(pos, button)
        return used

    def carry(self, start: tuple[int, int], end: tuple[int, int], mods: frozenset[str] = frozenset()) -> list[tuple]:
        """Press the left button at a position, take the mouse to another with some keys held, and let go there.

        What the strip told of it, the pick left out.
        """
        self.told.clear()
        self.down(start)
        self.move(end, mods)
        self.up(end, mods=mods)
        return [told for told in self.told if told[0] != "changed"]

    def piece(self, rect: layout.Rect) -> dict:
        """Show the song of the trailer piece in a strip of a rectangle, and give the song."""
        made = Project.load(EXAMPLES / "piece.json").song("piece")
        self.strip.rect = rect
        self.strip.show(made)
        return made

    def test_a_block_dragged_to_a_gap_is_told_as_moved_to_the_place_that_songedit_puts_it_at(self) -> None:
        order, seen = FOUR["order"], self.strip.map
        for place in range(4):
            x, _, w, _ = seen.block(place)
            for gap, to in enumerate(TO[place]):
                self.told.clear()
                self.down((x + w // 2, 50))
                # How far the mouse is under the strip makes no difference.
                self.move((FOUR_GAPS[gap], 90))
                # The line is in the gap, and there is none where the block would land at the place it has.
                self.assertEqual((self.strip.lifted, self.strip.gap), (place, None if to is None else gap))
                self.up((FOUR_GAPS[gap], 90))
                self.assertEqual(self.told, [("changed", place)] + ([] if to is None else [("moved", place, to)]))
                self.assertEqual((self.strip.busy, self.strip.lifted, self.strip.gap), (False, None, None))
                # What the strip told gives, with `songedit.moved`, the order that the line promised.
                made = order if to is None else songedit.moved(FOUR, *self.told[-1][1:])["order"]
                self.assertEqual(made, put(order, place, gap))

    def test_drop_of_the_trailer_piece_is_carried_to_between_hits_and_grow_and_its_first_block_to_its_end(self) -> None:
        made = self.piece(WIDE)
        order, seen = made["order"], self.strip.map
        self.assertEqual((seen.block(5), seen.gap_x(2), seen.gap_x(9)), ((390, 20, 79, 54), 89, 669))
        self.assertEqual(self.carry((430, 50), (92, 50)), [("moved", 5, 2)])
        self.assertEqual(songedit.moved(made, 5, 2)["order"],
                         ["cold", "hits", "drop", "grow", "tune", "rise", "band", "riff", "end"])
        # The first block to the gap after the last one, and the last block to the gap before the first.
        self.assertEqual((self.carry((20, 50), (669, 50)), self.carry((660, 50), (10, 50))),
                         ([("moved", 0, 8)], [("moved", 8, 0)]))
        for place, to, gap in ((5, 2, 2), (0, 8, 9), (8, 0, 0)):
            self.assertEqual(songedit.moved(made, place, to)["order"], put(order, place, gap))

    def test_a_block_carried_back_to_its_own_place_is_not_told_of(self) -> None:
        self.down((250, 50))
        self.move((40, 50))
        self.assertEqual((self.strip.lifted, self.strip.gap), (2, 0))
        # Back where it was pressed it is still picked up, and no line shows.
        self.move((250, 50))
        self.assertEqual((self.strip.lifted, self.strip.gap), (2, None))
        self.up((250, 50))
        self.assertEqual(self.told, [("changed", 2)])
        # Its own place reaches over the nearer half of the block on each side of it: from x 170 to 313.
        self.assertEqual([self.carry((250, 50), (x, 50)) for x in (169, 170, 313, 314)],
                         [[("moved", 2, 1)], [], [], [("moved", 2, 3)]])

    def test_a_press_that_stays_within_the_slack_is_a_click_and_only_picks(self) -> None:
        self.assertEqual((self.strip.picked, SLACK), (0, 3))
        self.down((250, 50))
        self.move((247, 50))
        self.assertEqual((self.strip.picked, self.strip.busy, self.strip.lifted), (2, True, None))
        self.up((247, 50))
        self.assertEqual(self.told, [("changed", 2)])
        # With ctrl held a block that is picked up is copied wherever it is let go, and one within the slack is not.
        self.assertEqual([self.carry((250, 50), (x, 50), CTRL) for x in (253, 254, 247, 246)],
                         [[], [("copied", 2, 3)], [], [("copied", 2, 2)]])
        # Up and down is not along the strip: however far, it picks nothing up.
        self.assertEqual(self.carry((250, 50), (253, 500), CTRL), [])

    def test_with_ctrl_held_at_the_release_the_block_is_told_as_copied_to_the_gap_and_not_as_moved(self) -> None:
        order = FOUR["order"]
        for gap, x in enumerate(FOUR_GAPS):
            # The gap before the block and the one after it take a copy as any other does.
            self.assertEqual(self.carry((170, 50), (x, 50), CTRL), [("copied", 1, gap)])
            # `songedit.copied` puts the copy right after the block, at place 2, and `songedit.moved` takes it on.
            self.assertEqual(songedit.moved(songedit.copied(FOUR, 1), 2, gap)["order"], put(order, 1, gap, "b2"))
        # Ctrl is asked for at the release. Held on the way and let go, the block is moved.
        self.told.clear()
        self.down((170, 50))
        self.move((297, 50), CTRL)
        self.assertEqual(self.strip.gap, 3)
        self.up((297, 50))
        self.assertEqual(self.told, [("changed", 1), ("moved", 1, 2)])
        # Held only at the release, it is copied.
        self.told.clear()
        self.down((170, 50))
        self.move((297, 50))
        self.up((297, 50), mods=CTRL)
        self.assertEqual(self.told, [("changed", 1), ("copied", 1, 3)])

    def test_escape_drops_the_press_and_its_release_then_tells_nothing(self) -> None:
        self.down((250, 50))
        self.move((40, 50))
        # The strip has the focus from the press, so the Ui hands it the key.
        self.assertTrue(self.ui.handle(Event("key", key="escape")))
        self.assertEqual((self.strip.busy, self.strip.lifted, self.strip.gap), (False, None, None))
        self.move((60, 50))
        self.up((60, 50))
        self.assertEqual(self.told, [("changed", 2)])
        # Handed in by the owner it does the same, also with ctrl held for a copy.
        self.down((250, 50))
        self.move((40, 50), CTRL)
        self.assertTrue(self.strip.key(Event("key", key="escape", mods=CTRL)))
        self.up((40, 50), mods=CTRL)
        self.assertEqual(self.told, [("changed", 2), ("changed", 2)])
        # With no press under way escape is not the strip's, and while a block is carried no other key is.
        self.assertEqual([self.strip.key(Event("key", key="escape")), self.ui.handle(Event("key", key="escape"))],
                         [False, False])
        self.down((250, 50))
        self.move((40, 50))
        self.assertEqual((self.strip.key(Event("key", key="delete")), self.strip.lifted), (False, 2))

    def test_the_right_button_on_a_block_and_past_the_last_block_is_told_for_a_menu_and_picks_nothing(self) -> None:
        self.assertTrue(self.click((250, 50), button=3))
        self.assertEqual((self.told, self.strip.picked, self.strip.busy), ([("asked", 2, (250, 50))], 0, False))
        # Past the last block is the pixel left free after it.
        self.assertTrue(self.click((329, 30), button=3))
        self.assertEqual(self.told[1:], [("asked", None, (329, 30))])
        # Between two blocks, and under a block, the button is not the strip's.
        self.assertEqual([self.click((137, 40), button=3), self.click((250, 75), button=3)], [False, False])
        # In a song with nothing in its order the whole strip is past the last block.
        self.strip.show(song())
        self.assertTrue(self.click((100, 40), button=3))
        self.assertEqual(self.told[1:], [("asked", None, (329, 30)), ("asked", None, (100, 40))])
        # A strip that tells nobody uses the press all the same.
        quiet = SongStrip(RECT)
        quiet.show(FOUR)
        self.assertEqual([quiet.on(Event("down", pos, button=3), self.ui) for pos in ((250, 50), (329, 30), (137, 40))],
                         [True, True, False])

    def test_two_presses_on_the_name_of_a_block_within_the_time_are_a_double_press_and_named_is_told(self) -> None:
        self.assertEqual((NAME_H, DOUBLE), (11, 0.4))
        self.click((250, 20))
        self.now = DOUBLE
        self.down((252, 30))
        # The second press tells of no pick, and begins nothing that a drag would carry.
        self.assertEqual((self.told, self.strip.busy), ([("changed", 2), ("named", 2)], False))
        self.up((252, 30))
        # A third press begins anew, and a fourth is a double press again.
        self.click((250, 25))
        self.click((250, 25))
        self.assertEqual(self.told[2:], [("changed", 2), ("named", 2)])
        # Later than the time, on the names of two blocks, and with either press under the name, it is two presses.
        for first, second, wait, places_told in (((250, 25), (250, 25), DOUBLE + 0.01, (2, 2)),
                                                 ((170, 25), (250, 25), 0.0, (1, 2)),
                                                 ((250, 31), (250, 25), 0.0, (2, 2)),
                                                 ((250, 25), (250, 31), 0.0, (2, 2))):
            self.told.clear()
            self.now += 100
            self.click(first)
            self.now += wait
            self.click(second)
            self.assertEqual(self.told, [("changed", place) for place in places_told])
        # A press in the song that was shown before is no first press in another song, which has c there as well.
        self.told.clear()
        self.now += 100
        self.click((250, 25))
        self.strip.show(song("b", "a", "c", "d", d={"lead": "o4 l4 c c"}))
        self.click((250, 25))
        self.assertEqual(self.told, [("changed", 2), ("changed", 2)])
        # A strip that tells nobody takes the second press for its own all the same, and picks nothing with it.
        quiet = SongStrip(RECT, clock=lambda: self.now)
        quiet.show(FOUR)
        used = [quiet.on(Event("down", pos, button=1), self.ui) for pos in ((250, 25), (250, 25))]
        self.assertEqual((used, quiet.picked, quiet.busy), ([True, True], 2, False))

    def test_name_rect_is_where_the_name_of_a_block_is_drawn_in_the_view_and_no_place_is_refused(self) -> None:
        self.assertEqual([self.strip.name_rect(place) for place in range(4)],
                         [(14, 20, 119, 11), (142, 20, 55, 11), (206, 20, 87, 11), (302, 20, 23, 11)])
        # It goes with the view.
        self.strip.show(song(*["a"] * 40))
        self.strip.offset = 100
        self.assertEqual((self.strip.map.block(3), self.strip.name_rect(3)), ((6, 20, 31, 54), (10, 20, 23, 11)))
        for place in (40, -1, True, 3.0, None):
            with self.assertRaises(RowSynthError) as refused:
                self.strip.name_rect(place)
            self.assertEqual(str(refused.exception),
                             f"{place!r} is no place in the order, which has 40, counted from 0")

    def test_busy_is_from_a_press_of_the_left_button_on_a_block_to_its_release(self) -> None:
        self.assertFalse(self.strip.busy)
        self.down((250, 50))
        self.assertEqual((self.strip.busy, self.strip.lifted), (True, None))
        self.move((40, 50))
        self.assertEqual((self.strip.busy, self.strip.lifted), (True, 2))
        self.up((40, 50))
        self.assertFalse(self.strip.busy)
        # A press between two blocks is none on a block, and the other buttons begin nothing to carry.
        for pos, button in (((137, 40), 1), ((250, 50), 3), ((250, 50), 2)):
            self.down(pos, button)
            self.assertFalse(self.strip.busy)
            self.up(pos, button)

    def test_a_press_of_another_button_while_a_block_is_carried_begins_nothing_and_it_is_carried_on(self) -> None:
        self.down((250, 50))
        self.move((40, 50))
        self.down((40, 50), button=3)
        self.up((40, 50), button=3)
        self.assertEqual((self.told, self.strip.lifted, self.strip.gap), ([("changed", 2)], 2, 0))
        self.up((40, 50))
        self.assertEqual(self.told, [("changed", 2), ("moved", 2, 0)])

    def test_showing_a_song_while_a_block_is_carried_drops_it_and_its_release_tells_nothing(self) -> None:
        self.down((250, 50))
        self.move((40, 50))
        # A song that is refused is not shown, and the block is carried on.
        with self.assertRaises(RowSynthError):
            self.strip.show(song("a", "x"))
        self.assertEqual((self.strip.lifted, self.strip.gap), (2, 0))
        # The same song come back is shown: the pick stays, and the block is carried no more.
        self.strip.show(FOUR)
        self.assertEqual((self.strip.busy, self.strip.lifted, self.strip.gap, self.strip.picked),
                         (False, None, None, 2))
        self.up((40, 50))
        self.assertEqual(self.told, [("changed", 2)])

    def test_a_block_carried_near_an_end_of_a_strip_that_scrolls_moves_the_view_that_way_frame_by_frame(self) -> None:
        made = self.piece(NARROW)
        self.assertEqual((self.strip.map.most, EDGE_W, EDGE_PX), (328, 12, 4))
        # A press near the right end, on tune, carries nothing yet: the view stays.
        self.down((200, 50))
        self.strip.update()
        self.assertEqual(self.strip.offset, 0)
        # Picked up there, it moves the view on by 4 pixels a frame.
        self.move((204, 50))
        self.strip.update()
        self.strip.update()
        self.assertEqual(self.strip.offset, 8)
        # Twelve pixels from the end is near it, and thirteen is not.
        self.move((197, 50))
        self.strip.update()
        self.assertEqual(self.strip.offset, 8)
        self.move((198, 50))
        self.strip.update()
        self.assertEqual(self.strip.offset, 12)
        # Past the end of the strip is near it too. The view stops at the end of the song, and the gap that is
        # nearest the mouse there is the last one, which was out of view.
        self.move((400, 90))
        for _ in range(100):
            self.strip.update()
        self.assertEqual((self.strip.offset, self.strip.gap), (328, 9))
        self.up((400, 90))
        self.assertEqual(self.told, [("changed", 3), ("moved", 3, 8)])
        self.assertEqual(songedit.moved(made, 3, 8)["order"], put(made["order"], 3, 9))
        # Back towards the start, with riff. The gap under the mouse is found in the view as it is in each frame.
        self.told.clear()
        self.down((100, 50))
        self.move((22, 50))
        self.strip.update()
        self.assertEqual((self.strip.offset, self.strip.gap), (328, 6))
        self.move((21, 50))
        self.assertEqual((self.strip.offset, self.strip.gap), (328, 6))
        self.strip.update()
        self.assertEqual((self.strip.offset, self.strip.gap), (324, 5))
        for _ in range(100):
            self.strip.update()
        self.assertEqual((self.strip.offset, self.strip.gap), (0, 0))
        self.up((21, 50))
        self.assertEqual(self.told, [("changed", 7), ("moved", 7, 0)])
        # With no block carried a frame moves nothing, and a song that fits has nowhere to move to.
        self.strip.offset = 100
        self.move((21, 50))
        self.strip.update()
        self.assertEqual(self.strip.offset, 100)
        self.strip.rect = RECT
        self.strip.show(FOUR)
        self.down((250, 50))
        self.move((329, 50))
        self.strip.update()
        self.assertEqual((self.strip.offset, self.strip.gap), (0, 4))

    def test_the_view_does_not_go_after_the_playhead_while_a_press_on_a_block_is_under_way(self) -> None:
        self.piece(NARROW)
        self.strip.playhead = 100.0
        self.down((80, 50))
        self.strip.playhead = 300.0
        self.assertEqual(self.strip.offset, 0)
        self.up((80, 50))
        # After the release it goes after it again, once the playhead has been in view.
        self.strip.playhead = 100.0
        self.strip.playhead = 300.0
        self.assertEqual(self.strip.offset, 300)


@unittest.skipUnless(pygame, NEEDS)
class Drawn(unittest.TestCase):
    def setUp(self) -> None:
        self.ui = Ui()
        self.strip = self.ui.add(SongStrip(RECT))
        self.song = song("a", "b", "a")
        self.strip.show(self.song)
        self.start(SIZE)

    def start(self, size: tuple[int, int]) -> None:
        """Take a new surface of a size, all of it in the colour of what is not the strip's."""
        self.surface = pygame.Surface(size)
        self.surface.fill(OUTSIDE)
        self.canvas = Canvas(self.surface, Font(sprites.load("font")))

    def at(self, *pos: int) -> tuple[int, ...]:
        """The colour of one pixel."""
        return tuple(self.surface.get_at(pos))[:3]

    def colours(self, rect: layout.Rect) -> set[tuple[int, ...]]:
        """The colours a rectangle has in it."""
        x, y, w, h = rect
        return {self.at(x + column, y + row) for column in range(w) for row in range(h)}

    def where(self, colour: tuple[int, ...], rect: layout.Rect | None = None) -> set[tuple[int, int]]:
        """The pixels of a rectangle that are in a colour. With no rectangle, those of the whole surface."""
        x, y, w, h = rect or (0, 0, *self.surface.get_size())
        return {(x + column, y + row) for column in range(w) for row in range(h)
                if self.at(x + column, y + row) == colour}

    def written(self, text: str, rect: layout.Rect) -> set[tuple[int, int]]:
        """The pixels a line of text takes when the canvas writes it at the left of a rectangle."""
        page = pygame.Surface(self.surface.get_size())
        Canvas(page, Font(sprites.load("font"))).text_in(text, rect, theme.TEXT)
        x, y, w, h = rect
        return {(x + column, y + row) for column in range(w) for row in range(h)
                if tuple(page.get_at((x + column, y + row)))[:3] == theme.TEXT}

    def draw(self) -> Map:
        """Draw the strip, and give its sums."""
        self.ui.draw(self.canvas)
        return self.strip.map

    def named(self, seen: Map, index: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        """The pixels of a block in the colour of text, and those its name takes 4 pixels in, in its band."""
        x, y, w, _ = seen.block(index)
        band = (x + theme.PAD, y, max(0, w - 2 * theme.PAD), NAME_H)
        return self.where(theme.TEXT, seen.block(index)), self.written(seen.name(index), band)

    def hold_inside(self) -> None:
        """Hold that the strip has drawn on every pixel of its rectangle, and on none outside it."""
        x, y, w, h = self.strip.rect
        width, height = self.surface.get_size()
        around = ((0, 0, width, y), (0, y + h, width, height - y - h), (0, y, x, h), (x + w, y, width - x - w, h))
        self.assertEqual(set().union(*(self.colours(rect) for rect in around)), {OUTSIDE})
        self.assertNotIn(OUTSIDE, self.colours(self.strip.rect))

    def example(self, file: str, name: str) -> tuple[dict, Map]:
        """Show a song of an example file in a strip 660 wide with its second block picked, and draw it.

        Hold that every block has its name, that every note of every channel is in its colour at its place and
        nothing else is in those colours, and that nothing is drawn outside the strip.
        """
        made = Project.load(EXAMPLES / file).song(name)
        self.strip.rect = WIDE
        self.start((680, 100))
        self.strip.show(made)
        self.strip.picked = 1
        seen = self.draw()
        for index in range(len(seen.blocks)):
            inked, name_of = self.named(seen, index)
            self.assertEqual((inked, bool(inked)), (name_of, bool(seen.name(index))))
        for colour in theme.CHANNELS:
            self.assertEqual(self.where(colour, WIDE), painted(made, seen, colour))
        self.hold_inside()
        return made, seen

    def test_small_draws_each_note_in_the_colour_and_cuts_one_that_runs_out_of_the_rectangle(self) -> None:
        small(self.canvas, SMALL, (note(0, 4, 60), note(28, 8, 72)), 32, theme.SONG)
        self.assertEqual(self.where(theme.SONG), {(x, 28) for x in range(10, 17)} | {(x, 20) for x in range(66, 74)})
        self.assertEqual(self.colours((0, 0, *SIZE)), {OUTSIDE, theme.SONG})

    def test_a_block_has_its_name_at_its_top_left_cut_where_it_is_narrow_and_none_where_narrower_still(self) -> None:
        seen = self.draw()
        for index in range(3):
            inked, name_of = self.named(seen, index)
            self.assertEqual((inked, bool(inked)), (name_of, True))
        # The letter b starts 4 pixels in from the side of its block and 2 under its top.
        inked = self.where(theme.TEXT, seen.block(1))
        self.assertEqual((min(x for x, _ in inked), min(y for _, y in inked)), (142, 22))
        # Blocks of a bar: 79 pixels wide, 39, and 15, which is too narrow for a letter and the two dots.
        for count, width, shown in ((4, 79, "bridge"), (8, 39, "bri.."), (40, 15, "")):
            self.strip.show(song(*["bridge"] * count))
            seen = self.draw()
            inked, name_of = self.named(seen, 1)
            self.assertEqual((seen.block(1)[2], seen.name(1), inked, bool(inked)), (width, shown, name_of, bool(shown)))

    def test_the_notes_of_each_channel_are_lines_in_its_colour_under_the_name_at_their_places(self) -> None:
        seen = self.draw()
        for colour in theme.CHANNELS:
            self.assertEqual(self.where(colour, RECT), painted(self.song, seen, colour))
        self.assertEqual([bool(self.where(colour, RECT)) for colour in theme.CHANNELS], [True, True, True, False])
        # The first note of the lead is c4 for a beat: on the lowest line of its lane, a pixel short of the next.
        self.assertEqual(self.where(theme.SONG, (10, 43, 30, 1)), {(x, 43) for x in range(12, 26)})
        # g4 is as high in the block of one bar, where it is the only note, as in the others.
        self.assertEqual(self.where(theme.SONG, seen.block(1)), {(x, 37) for x in range(140, 198)})
        self.assertEqual(self.at(42, 37), theme.SONG)
        # The hits of the kit are ticks, the kick low in its lane and the hat high.
        self.assertEqual(self.where(theme.SALMON, (10, 59, 32, 12)), {(12, 69), (12, 70), (27, 59), (27, 60)})
        # Two blocks of one pattern look the same, where neither is the picked one.
        self.strip.picked = 1
        self.draw()
        first, third = ([self.at(x + column, 20 + row) for column in range(127) for row in range(54)]
                        for x in (10, 202))
        self.assertEqual(first, third)
        # The song shown again with other notes shows those.
        changed = song("a", "b", "a", a={"lead": "o4 l1 e c"})
        self.strip.show(changed)
        seen = self.draw()
        self.assertEqual(self.where(theme.SONG, seen.block(0)),
                         {(x, 37) for x in range(12, 72)} | {(x, 43) for x in range(73, 134)})
        self.assertEqual((self.where(theme.SONG, RECT), self.where(theme.SOUND, RECT)),
                         (painted(changed, seen, theme.SONG), set()))

    def test_the_picked_block_is_lit_and_has_a_line_of_the_active_colour_under_it(self) -> None:
        self.draw()
        self.assertEqual(self.where(theme.ACTIVE), {(x, y) for x in range(10, 137) for y in (74, 75)})
        # Its face and the band of its name are a shade lighter than those of another block.
        self.assertEqual([self.at(11, 33), self.at(11, 21), self.at(139, 33), self.at(139, 21)],
                         [theme.LIT, theme.LINE, theme.RAISED, theme.LIT])
        # Both have the same outline, and the panel shows between them.
        self.assertEqual([self.at(10, 40), self.at(136, 40), self.at(137, 40), self.at(138, 40)],
                         [theme.LINE, theme.LINE, theme.PANEL, theme.LINE])
        self.ui.handle(Event("down", (150, 40), button=1))
        self.ui.handle(Event("up", (150, 40), button=1))
        self.draw()
        self.assertEqual(self.where(theme.ACTIVE), {(x, y) for x in range(138, 201) for y in (74, 75)})
        self.assertEqual([self.at(11, 33), self.at(139, 33)], [theme.RAISED, theme.LIT])
        self.strip.picked = None
        self.draw()
        self.assertEqual(self.where(theme.ACTIVE), set())

    def test_the_playhead_is_a_line_down_the_blocks_at_its_row(self) -> None:
        self.strip.picked = None
        self.strip.playhead = 40.5
        seen = self.draw()
        self.assertEqual((seen.x(40.5), self.where(theme.ACTIVE)), (172, {(172, y) for y in range(20, 76)}))
        # None is no line, and neither is a row before the song or after it.
        for playhead in (None, -1.0, 80.0):
            self.strip.playhead = playhead
            self.draw()
            self.assertEqual(self.where(theme.ACTIVE), set())

    def test_nothing_is_drawn_outside_the_rectangle_and_the_bar_shows_only_while_the_song_scrolls(self) -> None:
        self.strip.playhead = 40.5
        self.draw()
        self.hold_inside()
        self.assertEqual(self.colours((10, 76, 320, 4)), {theme.PANEL})
        # A song that scrolls, with a block cut at each edge: the picked one at the left, with the playhead in it.
        self.strip.show(song(*["a"] * 40))
        self.strip.offset, self.strip.picked, self.strip.playhead = 100, 3, 100.0
        seen = self.draw()
        self.hold_inside()
        self.assertEqual((seen.block(3), seen.block(13)), ((6, 20, 31, 54), (326, 20, 31, 54)))
        self.assertEqual(self.where(theme.ACTIVE, (0, 74, 340, 2)), {(x, y) for x in range(10, 37) for y in (74, 75)})
        self.assertEqual([self.at(10, 30), self.at(327, 33), self.at(329, 73)],
                         [theme.ACTIVE, theme.RAISED, theme.LINE])
        # The bar is a dark track with the thumb on it, a pixel under the line of the picked block.
        self.assertEqual((seen.thumb(), self.colours((10, 76, 320, 1))), ((35, 77, 80, 3), {theme.PANEL}))
        self.assertEqual(self.colours((10, 77, 320, 3)), {theme.BACKGROUND, theme.LINE})
        self.assertEqual(self.where(theme.LINE, (10, 77, 320, 3)),
                         {(x, y) for x in range(35, 115) for y in range(77, 80)})

    def test_the_arcade_tune_is_two_blocks_of_four_bars_with_every_note_of_its_four_channels(self) -> None:
        _, seen = self.example("arcade.json", "tune")
        self.assertEqual(([seen.block(index) for index in range(2)], [seen.name(index) for index in range(2)]),
                         ([(10, 20, 329, 54), (340, 20, 329, 54)], ["a", "b"]))
        self.assertTrue(all(self.where(colour, WIDE) for colour in theme.CHANNELS))
        # The lead starts on e5 for a beat, the lowest note it has in the song: on the lowest line of its lane.
        self.assertEqual(self.where(theme.SONG, (10, 40, 22, 1)), {(x, 40) for x in range(12, 31)})
        # The second block is picked, and the playhead is in the first.
        self.assertEqual(self.where(theme.ACTIVE), {(x, y) for x in range(340, 669) for y in (74, 75)})
        self.strip.playhead = 20.5
        self.draw()
        self.assertEqual(self.where(theme.ACTIVE, (0, 0, 680, 74)), {(116, y) for y in range(20, 74)})
        self.hold_inside()

    def test_the_trailer_piece_is_nine_blocks_of_eighteen_lanes_and_its_last_is_too_narrow_for_its_name(self) -> None:
        made, seen = self.example("piece.json", "piece")
        self.assertEqual((len(made["channels"]), [seen.name(index) for index in range(9)]),
                         (18, ["cold", "hits", "grow", "tune", "rise", "drop", "band", "riff", ""]))
        self.assertEqual((seen.block(8), self.where(theme.TEXT, seen.block(8))), ((650, 20, 19, 54), set()))
        # The lead starts on d4 for three rows, in a lane two pixels high.
        self.assertEqual((seen.lanes(0, 18)[0], self.where(theme.SONG, (10, 32, 5, 2))),
                         ((12, 32, 35, 2), {(12, 33), (13, 33)}))
        self.assertEqual(self.where(theme.ACTIVE), {(x, y) for x in range(50, 89) for y in (74, 75)})
        # Its fourth block is tune.
        self.ui.handle(Event("move", (200, 40)))
        self.assertEqual((self.strip.block_at((200, 40)), self.ui.help),
                         (3, "drag to move tune, ctrl to copy; right button for more"))
        self.strip.playhead = 20.5
        self.draw()
        self.assertEqual(self.where(theme.ACTIVE, (0, 0, 680, 74)), {(36, y) for y in range(20, 74)})
        self.hold_inside()

    def test_a_block_that_is_picked_up_has_a_bright_outline_and_its_foot_a_pixel_over_its_shadow(self) -> None:
        self.strip.show(FOUR)
        name = self.written("b", (142, 20, 55, 11))
        self.ui.handle(Event("down", (170, 50), button=1))
        seen = self.draw()
        # A press alone picks nothing up: the block has the outline that every block has.
        self.assertEqual((self.where(theme.TEXT, seen.block(1)), self.at(138, 40), self.at(170, 73)),
                         (name, theme.LINE, theme.LINE))
        self.ui.handle(Event("move", (174, 50)))
        self.draw()
        outline = ({(x, y) for x in range(138, 201) for y in (20, 72)}
                   | {(x, y) for x in (138, 200) for y in range(20, 73)})
        self.assertEqual(self.where(theme.TEXT, seen.block(1)), outline | name)
        # Its shadow lies where its foot stood, over the line of the picked block.
        self.assertEqual((self.colours((138, 73, 63, 1)), self.where(theme.ACTIVE)),
                         ({theme.BACKGROUND}, {(x, y) for x in range(138, 201) for y in (74, 75)}))
        # Its face, its band and its notes are as they were, and the other blocks have not changed.
        self.assertEqual([self.at(139, 33), self.at(139, 21), self.at(10, 40), self.at(202, 40), self.at(250, 73)],
                         [theme.LIT, theme.LINE, theme.LINE, theme.LINE, theme.LINE])
        for colour in theme.CHANNELS:
            self.assertEqual(self.where(colour, RECT), painted(FOUR, seen, colour))
        # No line stands beside it while it would land at the place it has.
        self.assertEqual(self.colours((137, 20, 1, 54)) | self.colours((201, 20, 1, 54)), {theme.PANEL})
        self.hold_inside()
        # It is the block that was pressed, also when the owner has picked another meanwhile.
        self.strip.picked = 0
        self.draw()
        self.assertEqual([self.at(10, 40), self.at(138, 40), self.at(170, 73), self.at(170, 74), self.at(70, 74)],
                         [theme.LINE, theme.TEXT, theme.BACKGROUND, theme.PANEL, theme.ACTIVE])
        self.ui.handle(Event("up", (174, 50), button=1))
        self.draw()
        self.assertEqual((self.where(theme.TEXT, seen.block(1)), self.at(138, 40), self.at(170, 73)),
                         (name, theme.LINE, theme.LINE))

    def test_a_bright_line_stands_in_the_gap_that_the_block_would_be_put_in(self) -> None:
        self.strip.show(FOUR)
        self.ui.handle(Event("down", (170, 50), button=1))
        for gap, x in enumerate(FOUR_GAPS):
            self.ui.handle(Event("move", (x, 50)))
            self.draw()
            # The gap before the block and the one after it have no line.
            line = set() if gap in (1, 2) else {(x, y) for y in range(20, 74)}
            self.assertEqual(self.where(theme.TEXT, (x, 20, 1, 60)), line)
            # At the foot of the blocks nothing else is in its colour, so there is no other line.
            self.assertEqual(self.where(theme.TEXT, (10, 73, 320, 1)), line & {(x, 73)})
            self.hold_inside()
        self.ui.handle(Event("up", (329, 50), button=1))
        self.draw()
        self.assertEqual(self.where(theme.TEXT, (10, 73, 320, 1)), set())
        # In a view that has moved the gaps are at x 5, 37 and so on: the one nearest x 150 is at 165.
        self.strip.show(song(*["a"] * 40))
        self.strip.offset = 100
        self.ui.handle(Event("down", (80, 50), button=1))
        self.ui.handle(Event("move", (150, 50)))
        seen = self.draw()
        self.assertEqual((self.strip.lifted, self.strip.gap, seen.gap_x(8)), (5, 8, 165))
        self.assertEqual(self.where(theme.TEXT, (10, 73, 320, 1)), {(165, 73)})
        self.assertEqual(self.where(theme.TEXT, (165, 20, 1, 60)), {(165, y) for y in range(20, 74)})

    def test_the_line_of_a_gap_has_a_plus_at_its_top_while_ctrl_is_held(self) -> None:
        self.strip.show(FOUR)
        self.ui.handle(Event("down", (170, 50), button=1))
        self.ui.handle(Event("move", (297, 50), mods=CTRL))
        self.draw()
        # A bright square of 7 pixels on the line, and in it a dark plus with arms of 2 pixels.
        square = {(x, y) for x in range(294, 301) for y in range(20, 27)}
        plus = {(x, 23) for x in range(295, 300)} | {(297, y) for y in range(21, 26)}
        self.assertEqual((self.where(theme.BACKGROUND, (294, 20, 7, 7)), self.where(theme.TEXT, (294, 20, 7, 7))),
                         (plus, square - plus))
        # The line goes on under it.
        self.assertEqual(self.where(theme.TEXT, (297, 27, 1, 53)), {(297, y) for y in range(27, 74)})
        # With ctrl let go the line has none.
        self.ui.handle(Event("move", (297, 50)))
        self.draw()
        self.assertEqual((self.where(theme.BACKGROUND, (294, 20, 7, 7)), self.where(theme.TEXT, (294, 20, 7, 7))),
                         (set(), {(297, y) for y in range(20, 27)}))
        # With ctrl the gap after the block has a line as well, for its copy.
        self.ui.handle(Event("move", (201, 50), mods=CTRL))
        self.draw()
        self.assertEqual(self.where(theme.BACKGROUND, (198, 20, 7, 7)),
                         {(x, 23) for x in range(199, 204)} | {(201, y) for y in range(21, 26)})
        self.assertEqual(self.where(theme.TEXT, (201, 27, 1, 53)), {(201, y) for y in range(27, 74)})
        # At an end of the strip the square is beside the line, and nothing is drawn outside the strip.
        for x, left in ((10, 10), (329, 323)):
            self.ui.handle(Event("move", (x, 50), mods=CTRL))
            self.draw()
            self.assertEqual(self.where(theme.BACKGROUND, (left, 20, 7, 7)),
                             {(at, 23) for at in range(left + 1, left + 6)} | {(left + 3, y) for y in range(21, 26)})
            self.assertEqual(self.where(theme.TEXT, (x, 27, 1, 53)), {(x, y) for y in range(27, 74)})
            self.hold_inside()

    def test_drop_of_the_trailer_piece_carried_to_between_hits_and_grow_is_lit_and_the_line_is_there(self) -> None:
        made, seen = self.example("piece.json", "piece")
        self.ui.handle(Event("down", (430, 50), button=1))
        self.ui.handle(Event("move", (92, 50)))
        self.draw()
        self.assertEqual((self.strip.lifted, self.strip.gap, seen.gap_x(2)), (5, 2, 89))
        # The line is in the pixel left free between hits and grow, and no other is drawn.
        self.assertEqual(self.where(theme.TEXT, (89, 20, 1, 60)), {(89, y) for y in range(20, 74)})
        self.assertEqual(self.where(theme.TEXT, (10, 73, 660, 1)), {(89, 73)})
        # The outline of drop is bright, and every note of the song is where it was.
        outline = ({(x, y) for x in range(390, 469) for y in (20, 72)}
                   | {(x, y) for x in (390, 468) for y in range(20, 73)})
        self.assertEqual(self.where(theme.TEXT, seen.block(5)), outline | self.written("drop", (394, 20, 71, 11)))
        for colour in theme.CHANNELS:
            self.assertEqual(self.where(colour, WIDE), painted(made, seen, colour))
        self.hold_inside()


if __name__ == "__main__":
    unittest.main()
