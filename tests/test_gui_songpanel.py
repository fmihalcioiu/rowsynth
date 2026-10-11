"""Tests for rowsynth.gui.songpanel: where the playing is in a song, the rows of its channels, and the panel itself."""
from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from collections.abc import Hashable
from pathlib import Path
from unittest import mock

from rowsynth.errors import RowSynthError
from rowsynth.gui import layout, songview, sprites, theme
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.font import Font
from rowsynth.gui.lists import List, Row
from rowsynth.gui.model import Model
from rowsynth.gui.songpanel import Order, SongPanel, arrange, channel_rows, colour_of, first_channel, place_at
from rowsynth.gui.songview import Block, Channel, Note
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

#: The middle of the art surface the layout is drawn for, and the least the panel is laid out in.
RECT = (124, 22, 662, 504)
SMALL = (124, 22, 500, 300)

#: The order of the song below as blocks: its patterns are 16 rows and 8, and the first is played again at the end.
BLOCKS = [Block("a", 0.0, 16.0), Block("b", 16.0, 8.0), Block("a", 24.0, 16.0)]
#: At this many ticks a row lasts a quarter of a second, so the 40 rows of the blocks take 10 seconds.
SLOW = 15

INSTRUMENTS = {"tone": {"wave": "tri"}, "thud": {"wave": "noise"}}
#: A song whose rows last a tenth of a second. Its first channel has no text in its first pattern.
WALK = {
    "row_ticks": 6,
    "drums": {"k": {"inst": "thud", "note": "c2"}, "h": {"inst": "thud", "note": "c6"}},
    "channels": {"hush": {"inst": "tone"}, "air": {"inst": "tone"}, "low": {"inst": "tone"},
                 "kit": {"inst": "thud", "drums": True}},
    "patterns": {"a": {"air": "o4 l4 c e g e", "low": "o2 l2 c g", "kit": "l4 k h k h"},
                 "b": {"hush": "o5 l2 c", "air": "o4 l8 g f e d", "kit": "l8 k k h h"}},
    "order": ["a", "b", "a"],
}
BARE = {"row_ticks": 6, "channels": {}, "patterns": {}, "order": []}
#: A song with two drum channels. The first plays one letter. The second plays two, one of them only further on.
PAIR = {
    "row_ticks": 6,
    "drums": {"k": {"inst": "thud", "note": "c2"}, "h": {"inst": "thud", "note": "c6"}},
    "channels": {"air": {"inst": "tone"}, "low": {"inst": "thud", "drums": True},
                 "top": {"inst": "thud", "drums": True}},
    "patterns": {"a": {"air": "o4 l4 c e g e", "low": "l4 k k k k", "top": "l8 r h r h r h r h"},
                 "b": {"air": "o4 l2 g e", "top": "l4 k h k h"}},
    "order": ["a", "b"],
}
PROJECT = {
    "instruments": INSTRUMENTS,
    "sounds": {"ping": {"def": {"voices": [{"inst": "tone", "notes": [[0, "c5", 6]]}]}}},
    "songs": {"walk": WALK, "bare": BARE},
}

HEADINGS = (Row(("TUNED",), "TUNED", head=True), Row(("DRUMS",), "DRUMS", head=True))


class Stand:
    """A stand-in for the session: a real model, and a playing that the test sets."""

    def __init__(self, model: Model) -> None:
        self.model = model
        self.playing = False
        self.at = 0.0

    def position(self) -> float:
        return self.at


def changed(pattern: str, channel: str, text: str) -> dict:
    """The project with other text for one channel of one pattern of its song."""
    project = copy.deepcopy(PROJECT)
    project["songs"]["walk"]["patterns"][pattern][channel] = text
    return project


def click(ui: Ui, pos: tuple[int, int]) -> None:
    """Press the left button at a position and let it go there."""
    ui.handle(Event("down", pos, button=1))
    ui.handle(Event("up", pos, button=1))


def middle(rect: layout.Rect) -> tuple[int, int]:
    """The pixel in the middle of a rectangle."""
    return rect[0] + rect[2] // 2, rect[1] + rect[3] // 2


def spot(rows: List, key: Hashable) -> tuple[int, int]:
    """The top-left corner of the row of a key, as the list is scrolled now."""
    x, y, _, h = rows.rect
    return next((x, down) for down in range(y, y + h) if getattr(rows.at((x, down)), "key", None) == key)


def lies_in(rect: layout.Rect, bounds: layout.Rect) -> bool:
    """Whether a rectangle has no pixel outside another, and neither a width nor a height below 0."""
    x, y, w, h = rect
    return (w >= 0 and h >= 0 and bounds[0] <= x and x + w <= bounds[0] + bounds[2]
            and bounds[1] <= y and y + h <= bounds[1] + bounds[3])


def ink(surface: pygame.Surface, rect: layout.Rect, colour: theme.Colour) -> int:
    """How many pixels of a rectangle are in a colour."""
    x, y, w, h = rect
    return sum(surface.get_at((x + column, y + row))[:3] == colour for column in range(w) for row in range(h))


class Where(unittest.TestCase):
    def test_at_the_start_the_song_is_at_the_first_row_of_its_first_place(self) -> None:
        self.assertEqual(place_at(0.0, SLOW, BLOCKS), (0, 0.0))

    def test_inside_a_pattern_the_row_is_counted_from_where_that_pattern_starts(self) -> None:
        self.assertEqual([place_at(seconds, SLOW, BLOCKS) for seconds in (1.25, 3.875, 5.0)],
                         [(0, 5.0), (0, 15.5), (1, 4.0)])

    def test_the_moment_a_pattern_ends_is_the_first_row_of_the_next(self) -> None:
        self.assertEqual([place_at(4.0, SLOW, BLOCKS), place_at(6.0, SLOW, BLOCKS)], [(1, 0.0), (2, 0.0)])

    def test_a_pattern_that_is_played_again_is_another_place_with_rows_of_its_own(self) -> None:
        self.assertEqual(place_at(7.5, SLOW, BLOCKS), (2, 6.0))

    def test_past_its_end_a_song_has_started_again(self) -> None:
        self.assertEqual([place_at(seconds, SLOW, BLOCKS) for seconds in (10.0, 11.25, 26.0)],
                         [(0, 0.0), (0, 5.0), (2, 0.0)])

    def test_patterns_of_unequal_length_take_their_own_rows_and_one_of_no_rows_takes_no_time(self) -> None:
        self.assertEqual([place_at(seconds, SLOW, BLOCKS)[0] for seconds in (3.75, 4.25, 5.75, 6.25, 9.75)],
                         [0, 1, 1, 2, 2])
        empty = [Block("a", 0.0, 16.0), Block("none", 16.0, 0.0), Block("b", 16.0, 8.0)]
        self.assertEqual(place_at(4.0, SLOW, empty), (2, 0.0))

    def test_a_row_lasts_its_ticks_so_at_half_the_ticks_the_song_is_twice_as_far(self) -> None:
        # Sixty ticks to the second: 1.5 seconds are 15 rows of 6 ticks, and 30 rows of 3.
        self.assertEqual([place_at(1.5, 6, BLOCKS), place_at(1.5, 3, BLOCKS)], [(0, 15.0), (2, 6.0)])

    def test_a_song_with_no_rows_is_nowhere(self) -> None:
        self.assertEqual([place_at(1.0, SLOW, []), place_at(1.0, SLOW, [Block("a", 0.0, 0.0)]),
                          place_at(1.0, 0, BLOCKS)], [None, None, None])


class Listed(unittest.TestCase):
    def test_the_channels_are_rows_under_two_headings_each_in_its_colour_and_dim_where_it_has_no_text(self) -> None:
        channels = songview.pattern(Project.from_dict(PROJECT).song("walk"), "a").channels
        self.assertEqual(channel_rows(channels), [
            HEADINGS[0],
            Row("hush", "hush", tag="tone", square=theme.SONG, dim=True),
            Row("air", "air", tag="tone", square=theme.SOUND),
            Row("low", "low", tag="tone", square=theme.SALMON),
            HEADINGS[1],
            Row("kit", "kit", tag="thud", square=theme.MUSTARD),
        ])
        # A heading stands also over no channel, and the colours start again after the last.
        self.assertEqual(channel_rows(()), list(HEADINGS))
        self.assertEqual([colour_of(4), colour_of(5)], [theme.SONG, theme.SOUND])

    def test_the_first_channel_is_tuned_and_has_notes_or_else_is_the_nearest_thing_to_that(self) -> None:
        note = (Note(0.0, 1.0, 60.0, 15, "tone"),)
        mute, tuned = Channel("mute", "tone", False, (), 0.0, False), Channel("tuned", "tone", False, note, 1.0, True)
        beat, still = Channel("beat", "thud", True, note, 1.0, True), Channel("still", "thud", True, (), 0.0, False)
        self.assertEqual([first_channel(channels) for channels in (
            (still, mute, beat, tuned), (still, mute, beat), (still, mute), (still,), ())],
            ["tuned", "beat", "mute", "still", None])


class Placed(unittest.TestCase):
    def test_the_buttons_of_the_order_are_as_wide_as_their_names_ask_and_share_a_row_too_narrow(self) -> None:
        told: list[int] = []
        row = Order((10, 5, 100, 13), changed=told.append)
        self.assertEqual(row.cells(), [])
        row.show(["a", "long", "b"], 1)
        self.assertEqual(row.cells(), [(10, 5, 13, 13), (23, 5, 31, 13), (54, 5, 13, 13)])
        # Showing tells nothing, and neither does picking the place that is picked.
        row.pick(1)
        self.assertEqual(told, [])
        row.pick(2)
        self.assertEqual((row.picked, told), (2, [2]))
        row.rect = (10, 5, 30, 13)
        self.assertEqual(row.cells(), [(10, 5, 7, 13), (17, 5, 16, 13), (33, 5, 7, 13)])


class Shown(unittest.TestCase):
    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "walk.json"
        self.path.write_text(json.dumps(PROJECT), encoding="utf-8")
        self.stand, self.ui = Stand(Model(self.path)), Ui()
        self.panel = SongPanel(self.stand, self.ui)
        self.panel.place(RECT)
        self.panel.update()

    def outside(self, project: dict) -> None:
        """Have the file changed as by another program, and the model take it up."""
        self.path.write_text(json.dumps(project), encoding="utf-8")
        self.assertEqual(self.stand.model.poll(), {("song", "walk")})

    def play(self, seconds: float) -> None:
        """Have the song be that far into its playing, and the panel take a frame."""
        self.stand.playing, self.stand.at = True, seconds
        self.panel.update()

    def seen(self) -> list[str]:
        """Which of the panel's widgets are seen, by what the panel calls them."""
        return [name for name in ("order", "list", "follow", "ghosts", "roll", "steps", "fault")
                if getattr(self.panel, name).visible]

    def test_at_the_start_the_first_place_and_the_first_tuned_channel_with_notes_are_picked(self) -> None:
        panel = self.panel
        self.assertEqual((panel.order.options, panel.order.picked), (("a", "b", "a"), 0))
        # The first channel of the song has no text in the first pattern.
        self.assertEqual((panel.channel, panel.list.picked), ("air", "air"))
        self.assertEqual((panel.roll.pattern.name, panel.roll.channel, panel.roll.colour), ("a", "air", theme.SOUND))
        self.assertEqual([note.pitch for note in panel.roll.notes], [60, 64, 67, 64])
        self.assertEqual(panel.list.rows, tuple(channel_rows(panel.roll.pattern.channels)))
        self.assertEqual((panel.follow.is_on, panel.ghosts.is_on), (True, True))

    def test_a_click_on_a_button_of_the_order_picks_that_place(self) -> None:
        panel = self.panel
        click(self.ui, middle(panel.order.cells()[1]))
        self.assertEqual((panel.order.picked, panel.roll.pattern.name), (1, "b"))
        self.assertEqual([note.pitch for note in panel.roll.notes], [67, 65, 64, 62])
        # The list is of the channels as that pattern has them.
        self.assertEqual([row.key for row in panel.list.rows if row.dim], ["low"])
        click(self.ui, middle(panel.order.cells()[2]))
        self.assertEqual((panel.order.picked, panel.roll.pattern.name), (2, "a"))

    def test_a_click_on_a_channel_picks_it(self) -> None:
        panel = self.panel
        click(self.ui, spot(panel.list, "low"))
        self.assertEqual((panel.channel, panel.list.picked, panel.roll.channel), ("low", "low", "low"))
        self.assertEqual((panel.roll.colour, [note.pitch for note in panel.roll.notes]), (theme.SALMON, [36, 43]))

    def test_a_drum_channel_is_shown_as_the_steps_and_a_tuned_one_as_the_roll(self) -> None:
        panel = self.panel
        self.assertEqual(self.seen(), ["order", "list", "follow", "ghosts", "roll"])
        click(self.ui, spot(panel.list, "kit"))
        self.assertEqual(self.seen(), ["order", "list", "follow", "steps"])
        # A lane for each letter the drum channel plays, and a square for each row of the pattern.
        self.assertEqual((panel.steps.grid.drums, panel.steps.grid.rows, panel.steps.grid.beat_rows), (2, 16, 4))
        self.assertEqual((panel.steps.rect, panel.steps.picked), (panel.roll.rect, "kit"))
        click(self.ui, spot(panel.list, "air"))
        self.assertEqual(self.seen(), ["order", "list", "follow", "ghosts", "roll"])

    def test_either_of_two_drum_channels_picked_shows_the_drums_of_both_with_its_own_labels_lit(self) -> None:
        self.path.write_text(json.dumps({"instruments": INSTRUMENTS, "songs": {"pair": PAIR}}), encoding="utf-8")
        ui = Ui()
        panel = SongPanel(Stand(Model(self.path)), ui)
        panel.place(RECT)
        panel.update(0.016)
        steps = panel.steps
        self.assertEqual((panel.channel, steps.visible, panel.roll.visible), ("air", False, True))
        for channel in ("top", "low", "top"):
            click(ui, spot(panel.list, channel))
            panel.update(0.016)
            self.assertEqual((panel.channel, steps.visible, panel.roll.visible), (channel, True, False))
            # A lane for the letter of the first, then air, and a lane for each of the two letters of the second.
            self.assertEqual((steps.picked, steps.grid.drums, steps.grid.breaks), (channel, 3, (1,)))
            self.assertEqual([steps.step_at(steps.grid.square(lane, 2)[:2]) for lane in range(3)],
                             [("low", "k", 2), ("top", "k", 2), ("top", "h", 2)])
        # The lanes are those of the whole song: another pattern has the same ones, with its own hits in them.
        ui.handle(Event("move", steps.grid.square(1, 4)[:2]))
        self.assertEqual(ui.help, "k in top, bar 1, step 5: empty")
        click(ui, middle(panel.order.cells()[1]))
        self.assertEqual((steps.picked, steps.grid.drums, steps.grid.rows), ("top", 3, 16))
        ui.handle(Event("move", steps.grid.square(1, 8)[:2]))
        self.assertEqual(ui.help, "k in top, bar 1, step 9: hit, level 15")
        # A tuned channel picked is shown in the roll, as with one drum channel.
        click(ui, spot(panel.list, "air"))
        self.assertEqual((panel.channel, steps.visible, panel.roll.visible, panel.roll.channel),
                         ("air", False, True, "air"))

    def test_the_channel_stays_picked_when_the_pattern_changes(self) -> None:
        panel = self.panel
        click(self.ui, spot(panel.list, "low"))
        click(self.ui, middle(panel.order.cells()[1]))
        # The second pattern has no text for it: it is shown with no notes, and its row is dim.
        self.assertEqual((panel.channel, panel.list.picked), ("low", "low"))
        self.assertEqual((panel.roll.pattern.name, panel.roll.channel, panel.roll.notes), ("b", "low", ()))
        self.assertTrue(next(row for row in panel.list.rows if row.key == "low").dim)
        self.assertEqual(self.seen(), ["order", "list", "follow", "ghosts", "roll"])

    def test_the_playhead_is_shown_only_while_the_pattern_that_plays_is_the_one_on_screen(self) -> None:
        panel = self.panel
        click(self.ui, middle(panel.follow.rect))
        self.assertEqual((panel.roll.playhead, panel.steps.playhead), (None, None))
        shown = []
        # Half a second in is row 5 of the first pattern, and 2 seconds in is the second pattern. At 3 seconds the
        # first is played again, as the third place in the order: it is the pattern on screen all the same.
        for seconds in (0.5, 2.0, 3.0):
            self.play(seconds)
            shown.append((panel.order.picked, panel.roll.playhead, panel.steps.playhead))
        self.assertEqual(shown, [(0, 5.0, 5.0), (0, None, None), (0, 6.0, 6.0)])
        self.stand.playing = False
        panel.update()
        self.assertIsNone(panel.roll.playhead)

    def test_follow_moves_the_picked_place_along_with_the_playing_and_not_when_it_is_off(self) -> None:
        panel = self.panel
        self.play(2.0)
        self.assertEqual((panel.order.picked, panel.roll.pattern.name, panel.roll.playhead), (1, "b", 4.0))
        self.assertEqual(panel.channel, "air")
        self.play(3.0)
        self.assertEqual((panel.order.picked, panel.roll.playhead), (2, 6.0))
        # A song that does not play moves nothing, wherever its position is said to be.
        self.stand.playing, self.stand.at = False, 2.0
        panel.update()
        self.assertEqual(panel.order.picked, 2)
        click(self.ui, middle(panel.follow.rect))
        self.assertFalse(panel.follow.is_on)
        self.play(2.0)
        self.assertEqual((panel.order.picked, panel.roll.playhead), (2, None))

    def test_the_ghosts_toggle_turns_the_ghosts_of_the_roll_off_and_on(self) -> None:
        panel = self.panel
        self.assertTrue(panel.roll.ghosts)
        for on in (False, True):
            click(self.ui, middle(panel.ghosts.rect))
            panel.update()
            self.assertEqual((panel.ghosts.is_on, panel.roll.ghosts), (on, on))

    def test_a_sound_or_an_instrument_picked_shows_nothing_and_the_song_picked_again_starts_anew(self) -> None:
        panel, model = self.panel, self.stand.model
        click(self.ui, spot(panel.list, "low"))
        click(self.ui, middle(panel.order.cells()[1]))
        for thing in (("sound", "ping"), ("instrument", "tone")):
            model.pick(*thing)
            self.play(2.0)
            self.assertEqual((self.seen(), panel.roll.playhead), ([], None))
        model.pick("song", "walk")
        self.stand.playing = False
        panel.update()
        self.assertEqual(self.seen(), ["order", "list", "follow", "ghosts", "roll"])
        self.assertEqual((panel.order.picked, panel.channel), (0, "air"))

    def test_a_song_with_no_order_and_no_channels_shows_the_headings_and_no_view(self) -> None:
        panel = self.panel
        self.stand.model.pick("song", "bare")
        self.play(1.0)
        self.assertEqual(self.seen(), ["order", "list", "follow"])
        self.assertEqual((panel.order.options, panel.list.rows, panel.channel), ((), HEADINGS, None))
        self.assertEqual((panel.roll.playhead, panel.steps.playhead), (None, None))

    def test_the_song_is_read_again_when_the_file_changed_with_the_view_where_it_was_and_at_no_other_time(
            self) -> None:
        panel = self.panel
        click(self.ui, middle(panel.order.cells()[1]))
        fitted = panel.roll.top
        self.ui.handle(Event("wheel", middle(panel.roll.rect), wheel=1))
        view = (panel.roll.row, panel.roll.top, panel.roll.row_w)
        self.assertNotEqual(panel.roll.top, fitted)
        with mock.patch.object(Project, "song", autospec=True, side_effect=Project.song) as read:
            for _ in range(3):
                panel.update()
            self.assertEqual(read.call_count, 0)
            self.outside(changed("b", "air", "o4 l8 g f e d c"))
            for _ in range(3):
                panel.update()
            self.assertEqual(read.call_count, 1)
        self.assertEqual([note.pitch for note in panel.roll.notes], [67, 65, 64, 62, 60])
        self.assertEqual((panel.order.picked, panel.channel), (1, "air"))
        self.assertEqual((panel.roll.row, panel.roll.top, panel.roll.row_w), view)
        # A song that lost the picked place and the picked channel outside is shown from its start.
        short = copy.deepcopy(PROJECT)
        song = short["songs"]["walk"]
        song["order"] = ["a"]
        del song["channels"]["air"], song["patterns"]["a"]["air"], song["patterns"]["b"]["air"]
        self.outside(short)
        panel.update()
        self.assertEqual((panel.order.options, panel.order.picked, panel.channel), (("a",), 0, "low"))

    def test_a_song_the_engine_refuses_shows_the_words_of_the_refusal_until_it_is_mended(self) -> None:
        panel = self.panel
        click(self.ui, middle(panel.order.cells()[1]))
        broken = changed("b", "air", "o4 c ?")
        with self.assertRaises(RowSynthError) as refused:
            songview.blocks(Project.from_dict(broken).song("walk"))
        self.outside(broken)
        self.play(2.0)
        self.assertEqual(self.seen(), ["fault"])
        self.assertEqual((panel.fault.text, panel.fault.colour, panel.fault.rect),
                         (str(refused.exception), theme.FAULT, RECT))
        self.assertEqual(panel.fault.text, "pattern 'b', channel 'air': cannot read \"?\" in a channel")
        self.assertIsNone(panel.roll.playhead)
        self.outside(PROJECT)
        panel.update()
        self.assertEqual((self.seen(), panel.fault.text), (["order", "list", "follow", "ghosts", "roll"], ""))
        self.assertEqual((panel.order.picked, panel.channel), (1, "air"))

    def test_placed_at_two_sizes_every_widget_is_inside_the_panel(self) -> None:
        panel = self.panel
        self.assertEqual(arrange(RECT), {
            "order": (128, 26, 654, 13), "channels": (128, 43, 128, 479), "list": (131, 46, 122, 473),
            "view": (260, 43, 522, 479), "follow": (263, 46, 43, 13), "ghosts": (308, 46, 43, 13),
            "body": (263, 63, 516, 456)})
        self.assertEqual(len(panel.widgets), 7)
        self.assertEqual(self.ui.widgets, panel.widgets)
        for rect in (RECT, SMALL, (0, 0, 30, 5)):
            panel.place(rect)
            self.assertTrue(all(lies_in(widget.rect, rect) for widget in panel.widgets), rect)
            self.assertTrue(all(lies_in(part, rect) for part in arrange(rect).values()), rect)
        # In the least panel the list of the channels is as wide as in any, and the view has what is left.
        panel.place(SMALL)
        self.assertEqual((panel.order.rect, panel.list.rect, panel.follow.rect, panel.ghosts.rect),
                         ((128, 26, 492, 13), (131, 46, 122, 269), (263, 46, 43, 13), (308, 46, 43, 13)))
        self.assertEqual((panel.roll.rect, panel.steps.rect, panel.fault.rect),
                         ((263, 63, 354, 252), (263, 63, 354, 252), SMALL))

    def test_taken_off_the_window_nothing_is_seen_and_brought_back_only_what_was_shown(self) -> None:
        self.panel.show(False)
        self.assertEqual(self.seen(), [])
        self.panel.update()
        self.assertEqual(self.seen(), [])
        self.panel.show(True)
        self.assertEqual(self.seen(), ["order", "list", "follow", "ghosts", "roll"])


@unittest.skipUnless(pygame, NEEDS)
class Drawn(unittest.TestCase):
    def setUp(self) -> None:
        self.surface = pygame.Surface((RECT[2], RECT[3]))
        self.surface.fill(theme.BACKGROUND)
        self.canvas = Canvas(self.surface, Font(sprites.load("font")), sprites.load("icons"))
        self.ui = Ui()

    def test_the_drums_of_the_piece_are_drawn_together_with_the_picked_channel_lit_in_the_list_and_the_steps(
            self) -> None:
        panel = SongPanel(Stand(Model(EXAMPLES / "piece.json")), self.ui)
        panel.place((0, 0, RECT[2], RECT[3]))
        panel.update()
        panel.order.pick(panel.order.options.index("band"))
        click(self.ui, spot(panel.list, "snare"))
        panel.draw(self.canvas)
        self.ui.draw(self.canvas)
        steps, grid = panel.steps, panel.steps.grid
        # Eight drum channels of one letter each: eight lanes, all in view, each parted from the next.
        self.assertEqual((grid.drums, grid.drums_in_view(), grid.breaks), (8, range(0, 8), tuple(range(1, 8))))
        self.assertEqual([steps.step_at(grid.square(lane, 0)[:2])[:2] for lane in range(8)],
                         [("kick", "k"), ("hkick", "k"), ("snare", "s"), ("hsnare", "s"), ("hats", "h"),
                          ("shaker", "h"), ("ohat", "H"), ("crash", "t")])
        x = steps.rect[0]
        self.assertEqual([lane for lane in range(8) if self.surface.get_at((x + 1, grid.y(lane) + 1))[:3] == theme.LIT],
                         [2])
        # The kick is on the first row and the snare on the fifth, each in its own lane and not in the other's.
        hit = [[self.surface.get_at((grid.x(row) + 2, grid.y(lane) + 2))[:3] == grid.colour(row) for row in (0, 4)]
               for lane in (0, 1, 2, 3)]
        self.assertEqual(hit, [[True, False], [False, False], [False, True], [False, False]])
        # In the list the picked channel has its strip, and the square of one with no text here is dimmer.
        snare, crash = (*spot(panel.list, "snare"), 20, 12), (*spot(panel.list, "crash"), 20, 12)
        self.assertEqual((ink(self.surface, snare, theme.SONG), ink(self.surface, crash, theme.SOUND)),
                         (theme.LIST_SQUARE ** 2 + theme.LIST_STRIP * 12, 0))

    def test_the_lead_of_the_arcade_tune_is_drawn_in_its_colour_where_its_first_note_is(self) -> None:
        panel = SongPanel(Stand(Model(EXAMPLES / "arcade.json")), self.ui)
        rect = (0, 0, RECT[2], RECT[3])
        panel.place(rect)
        panel.update()
        panel.draw(self.canvas)
        self.ui.draw(self.canvas)
        self.assertEqual((panel.order.options, panel.channel, panel.roll.colour), (("a", "b"), "lead", theme.SONG))
        first = panel.roll.notes[0]
        bar = panel.roll.map.note(first.row, first.rows, first.pitch)
        self.assertTrue(lies_in(bar, arrange(rect)["body"]))
        self.assertGreater(ink(self.surface, bar, theme.SONG), bar[2] * bar[3] // 2)
        # The key above it has no note there, and the frame of the view is a line round the roll.
        self.assertEqual(ink(self.surface, (bar[0], bar[1] - bar[3] - 1, bar[2], bar[3]), theme.SONG), 0)
        x, y, w, h = arrange(rect)["view"]
        self.assertEqual(ink(self.surface, (x, y, w, 1), theme.LINE), w)


if __name__ == "__main__":
    unittest.main()
