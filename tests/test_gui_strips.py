"""Tests for rowsynth.gui.strips: where small notes go, the sums of the song strip, its pick and view, its drawing."""
from __future__ import annotations

import os
import unittest
from pathlib import Path

from rowsynth.errors import RowSynthError
from rowsynth.gui import layout, songview, sprites, theme
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.font import Font
from rowsynth.gui.songview import Block, Note
from rowsynth.gui.strips import BAR_H, NAME_H, ROW_W_MIN, WHEEL_PX, Map, SongStrip, places, small
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
#: The strip the examples are shown in.
WIDE = (10, 20, 660, 60)
#: The surface a strip is drawn on, which is larger than it on every side, and the colour of what is not its.
SIZE = (340, 100)
OUTSIDE = theme.GOOD

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
        # Between two blocks, under them, and with another button, nothing is picked.
        self.assertEqual([self.press((137, 40)), self.press((250, 75)), self.press((250, 40), button=3)], [False] * 3)
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

    def test_the_help_line_says_which_block_the_mouse_is_on_its_bars_and_its_place_in_the_order(self) -> None:
        self.assertEqual((self.strip.help, self.ui.help), ("the song", ""))
        for pos, line in (((20, 30), "a, 2 bars, 1 of 3"), ((150, 40), "b, 1 bar, 2 of 3"),
                          ((300, 73), "a, 2 bars, 3 of 3"), ((137, 40), "the song"), ((150, 75), "the song")):
            self.ui.handle(Event("move", pos))
            self.assertEqual(self.ui.help, line)
        # A pattern of a bar and a half, which ends at x 145.
        self.strip.show(song("c", "a"))
        self.ui.handle(Event("move", (145, 40)))
        self.assertEqual(self.ui.help, "c, 1.5 bars, 1 of 2")
        self.strip.help = "another line"
        self.ui.handle(Event("move", (146, 40)))
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
        # Its third block is four bars of sixteen rows.
        self.ui.handle(Event("move", (100, 40)))
        self.assertEqual((self.strip.block_at((100, 40)), self.ui.help), (2, "grow, 4 bars, 3 of 9"))
        self.strip.playhead = 20.5
        self.draw()
        self.assertEqual(self.where(theme.ACTIVE, (0, 0, 680, 74)), {(36, y) for y in range(20, 74)})
        self.hold_inside()


if __name__ == "__main__":
    unittest.main()
