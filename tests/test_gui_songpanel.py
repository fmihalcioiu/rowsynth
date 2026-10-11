"""Tests for rowsynth.gui.songpanel: where the playing is, the rows of the channels, the panel, saving, the mixer."""
from __future__ import annotations

import copy
import dataclasses
import json
import os
import shutil
import tempfile
import threading
import unittest
from collections.abc import Callable, Hashable, Sequence
from pathlib import Path
from unittest import mock

from rowsynth.errors import RowSynthError
from rowsynth.gui import channelbar, chordview, dials, layout, songview, sprites, theme
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.chordview import ChordBox
from rowsynth.gui.font import Font
from rowsynth.gui.lists import List, Row
from rowsynth.gui.mixer import Strip
from rowsynth.gui.model import Model
from rowsynth.gui.popups import Menu
from rowsynth.gui.session import POLL_EVERY, Session
from rowsynth.gui.songpanel import (LAYERS, MIXER, NOTES, SongPanel, arrange, channel_rows, colour_of,
                                    first_channel, lowest_c, place_at, song_parts)
from rowsynth.gui.songview import Block, Channel, Note
from rowsynth.gui.steps import WHEEL_WAIT
from rowsynth.gui.strips import SongStrip
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

#: A project for a hand to change: the song above with an instrument of its own for one channel, and the song with
#: two drum channels.
EDITED = {
    "instruments": {**INSTRUMENTS, "buzz": {"wave": "saw"}},
    "songs": {"walk": {**WALK, "channels": {**WALK["channels"], "low": {"inst": "buzz"}}}, "pair": PAIR},
}
#: A song of two bars for one channel, and a project with two chords in which it has one written for each bar.
#: Under the second, the e that starts the second beat of its bar is wrong.
PAIRED = {"row_ticks": 6, "channels": {"air": {"inst": "tone"}},
          "patterns": {"a": {"air": "o4 l4 c e g e | d e g b"}}, "order": ["a"]}
CHORDED = {
    "instruments": INSTRUMENTS,
    "chords": {"C": {"root": 0, "tones": [0, 4, 7]}, "G": {"root": 7, "tones": [7, 11, 2]}},
    "songs": {"tune": {**PAIRED, "chords": {"a": ["C", "G"]}}},
}
#: The same song with no chords written, in a project that has no chords.
UNCHORDED = {"instruments": INSTRUMENTS, "songs": {"tune": PAIRED}}
#: The rate a project that is changed is rendered at on the threads of its model: a low one, to be quick.
RATE = 8000
#: The time the first write from outside is stamped with, in nanoseconds: none the file gets by being saved.
STAMP = 1_700_000_000 * 10**9

HEADINGS = (Row(("TUNED",), "TUNED", head=True), Row(("DRUMS",), "DRUMS", head=True))


class Stand:
    """A stand-in for the session: a real model, and a playing that the test sets."""

    def __init__(self, model: Model) -> None:
        self.model = model
        self.playing = False
        self.at = 0.0

    def position(self) -> float:
        return self.at


class Speaker:
    """A stand-in for the player: it keeps what it was asked, in order, and plays nothing."""

    def __init__(self) -> None:
        self.asked: list[tuple] = []
        self.playing = False
        # The seconds into what plays, which a test can move.
        self.at = 0.0

    def play(self, sides: Sequence, loop: bool = False) -> None:
        self.asked.append(("play", sides, loop))
        self.playing = True

    def also(self, sides: Sequence) -> None:
        self.asked.append(("also", sides))

    def stop(self, fade: float = 0.0) -> None:
        self.asked.append(("stop",))
        self.playing = False

    def swap(self, sides: Sequence) -> None:
        self.asked.append(("swap", sides))

    def position(self) -> float:
        return self.at

    def level(self) -> tuple[float, float]:
        return 0.0, 0.0


def settle() -> None:
    """Wait for every render and every run of the checks that is still on a thread of a model's."""
    for thread in threading.enumerate():
        if thread.name.startswith("rowsynth-"):
            thread.join()


def differing(one: object, other: object, where: str = "") -> list[str]:
    """The places at which the data of two files differ, each spelt as a change spells its place.

    Two dicts are gone through key by key. Anything else is one place, which differs when it is not written the
    same in both.
    """
    if isinstance(one, dict) and isinstance(other, dict):
        return [place for key in dict.fromkeys([*one, *other])
                for place in (differing(one[key], other[key], f"{where}.{key}".lstrip("."))
                              if key in one and key in other else [f"{where}.{key}".lstrip(".")])]
    return [] if json.dumps(one) == json.dumps(other) else [where]


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


def choose(ui: Ui, panel: SongPanel, option: str) -> None:
    """Click NOTES, MIXER or LAYERS in the choice of a panel."""
    click(ui, middle(panel.choice.cells()[panel.choice.options.index(option)]))


def order(panel: SongPanel) -> list[str]:
    """The names of the blocks of a panel's strip, in the order they stand."""
    return [block.name for block in panel.strip.map.blocks]


def block(panel: SongPanel, place: int) -> tuple[int, int]:
    """The pixel in the middle of the block of a place in the order, in a panel's strip."""
    return middle(panel.strip.map.block(place))


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
        # A channel that is not among those heard is dim too, and with none heard they all are.
        self.assertEqual([[row.key for row in channel_rows(channels, heard) if row.dim]
                          for heard in ({"hush", "air", "kit"}, {"low"}, set())],
                         [["hush", "low"], ["hush", "air", "kit"], ["hush", "air", "low", "kit"]])
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

    def test_the_lowest_c_in_view_is_the_first_c_that_is_no_lower_than_the_lowest_pitch_there(self) -> None:
        self.assertEqual([lowest_c(view) for view in (range(48, 80), range(49, 80), range(59, 62), range(37, 40))],
                         [48, 60, 60, 48])


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
        """Which of the panel's widgets are seen, by what the panel calls them. The two toggles are its bar's."""
        return [name for name in ("strip", "list", "follow", "ghosts", "roll", "steps", "fault")
                if getattr(self.panel.bar if name in ("follow", "ghosts") else self.panel, name).visible]

    def test_at_the_start_the_first_place_and_the_first_tuned_channel_with_notes_are_picked(self) -> None:
        panel = self.panel
        self.assertEqual((order(panel), panel.strip.picked), (["a", "b", "a"], 0))
        # The first channel of the song has no text in the first pattern.
        self.assertEqual((panel.channel, panel.list.picked), ("air", "air"))
        self.assertEqual((panel.roll.pattern.name, panel.roll.channel, panel.roll.colour), ("a", "air", theme.SOUND))
        self.assertEqual([note.pitch for note in panel.roll.notes], [60, 64, 67, 64])
        self.assertEqual(panel.list.rows, tuple(channel_rows(panel.roll.pattern.channels)))
        self.assertEqual((panel.bar.follow.is_on, panel.bar.ghosts.is_on), (True, True))

    def test_a_click_on_a_block_of_the_strip_picks_that_place_and_the_views_show_the_pattern_there(self) -> None:
        panel = self.panel
        click(self.ui, block(panel, 1))
        self.assertEqual((panel.strip.picked, panel.roll.pattern.name), (1, "b"))
        self.assertEqual([note.pitch for note in panel.roll.notes], [67, 65, 64, 62])
        # The list is of the channels as that pattern has them.
        self.assertEqual([row.key for row in panel.list.rows if row.dim], ["low"])
        click(self.ui, block(panel, 2))
        self.assertEqual((panel.strip.picked, panel.roll.pattern.name), (2, "a"))
        # With a drum channel picked, it is the steps that show the pattern of the block.
        click(self.ui, spot(panel.list, "kit"))
        click(self.ui, block(panel, 1))
        self.assertEqual((panel.strip.picked, panel.steps.visible, panel.steps.grid.rows), (1, True, 8))

    def test_a_click_on_a_channel_picks_it(self) -> None:
        panel = self.panel
        click(self.ui, spot(panel.list, "low"))
        self.assertEqual((panel.channel, panel.list.picked, panel.roll.channel), ("low", "low", "low"))
        self.assertEqual((panel.roll.colour, [note.pitch for note in panel.roll.notes]), (theme.SALMON, [36, 43]))

    def test_a_drum_channel_is_shown_as_the_steps_and_a_tuned_one_as_the_roll(self) -> None:
        panel = self.panel
        self.assertEqual(self.seen(), ["strip", "list", "follow", "ghosts", "roll"])
        click(self.ui, spot(panel.list, "kit"))
        self.assertEqual(self.seen(), ["strip", "list", "follow", "steps"])
        # A lane for each letter the drum channel plays, and a square for each row of the pattern.
        self.assertEqual((panel.steps.grid.drums, panel.steps.grid.rows, panel.steps.grid.beat_rows), (2, 16, 4))
        self.assertEqual((panel.steps.rect, panel.steps.picked), (panel.roll.rect, "kit"))
        click(self.ui, spot(panel.list, "air"))
        self.assertEqual(self.seen(), ["strip", "list", "follow", "ghosts", "roll"])

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
        # The lanes are those of the whole song: another pattern has the same ones, with its own hits in them. The
        # steps have the drum table of the song, so the help line says what a press would do.
        ui.handle(Event("move", steps.grid.square(1, 4)[:2]))
        self.assertEqual(ui.help, "put a hit: k in top, bar 1, step 5")
        click(ui, block(panel, 1))
        self.assertEqual((steps.picked, steps.grid.drums, steps.grid.rows), ("top", 3, 16))
        ui.handle(Event("move", steps.grid.square(1, 8)[:2]))
        self.assertEqual(ui.help, "take the hit away: k in top, bar 1, step 9 (level 15)")
        # A tuned channel picked is shown in the roll, as with one drum channel.
        click(ui, spot(panel.list, "air"))
        self.assertEqual((panel.channel, steps.visible, panel.roll.visible, panel.roll.channel),
                         ("air", False, True, "air"))

    def test_the_channel_stays_picked_when_the_pattern_changes(self) -> None:
        panel = self.panel
        click(self.ui, spot(panel.list, "low"))
        click(self.ui, block(panel, 1))
        # The second pattern has no text for it: it is shown with no notes, and its row is dim.
        self.assertEqual((panel.channel, panel.list.picked), ("low", "low"))
        self.assertEqual((panel.roll.pattern.name, panel.roll.channel, panel.roll.notes), ("b", "low", ()))
        self.assertTrue(next(row for row in panel.list.rows if row.key == "low").dim)
        self.assertEqual(self.seen(), ["strip", "list", "follow", "ghosts", "roll"])

    def test_the_playhead_of_the_strip_is_the_row_of_the_song_and_a_view_has_one_only_while_its_pattern_plays(
            self) -> None:
        panel = self.panel
        click(self.ui, middle(panel.bar.follow.rect))
        self.assertEqual((panel.strip.playhead, panel.roll.playhead, panel.steps.playhead), (None, None, None))
        shown = []
        # Half a second in is row 5 of the first pattern. Two seconds in is row 20 of the song, which is row 4 of
        # the second pattern. At 3 seconds the first is played again, as the third place in the order: it is the
        # pattern on screen all the same.
        for seconds in (0.5, 2.0, 3.0):
            self.play(seconds)
            shown.append((panel.strip.picked, panel.strip.playhead, panel.roll.playhead, panel.steps.playhead))
        self.assertEqual(shown, [(0, 5.0, 5.0, 5.0), (0, 20.0, None, None), (0, 30.0, 6.0, 6.0)])
        self.stand.playing = False
        panel.update()
        self.assertEqual((panel.strip.playhead, panel.roll.playhead), (None, None))

    def test_follow_moves_the_picked_block_along_with_the_playing_and_not_when_it_is_off(self) -> None:
        panel = self.panel
        self.play(2.0)
        self.assertEqual((panel.strip.picked, panel.roll.pattern.name, panel.roll.playhead), (1, "b", 4.0))
        self.assertEqual((panel.channel, panel.strip.playhead), ("air", 20.0))
        self.play(3.0)
        self.assertEqual((panel.strip.picked, panel.roll.playhead, panel.strip.playhead), (2, 6.0, 30.0))
        # A song that does not play moves nothing, wherever its position is said to be.
        self.stand.playing, self.stand.at = False, 2.0
        panel.update()
        self.assertEqual((panel.strip.picked, panel.strip.playhead), (2, None))
        click(self.ui, middle(panel.bar.follow.rect))
        self.assertFalse(panel.bar.follow.is_on)
        self.play(2.0)
        self.assertEqual((panel.strip.picked, panel.roll.playhead, panel.strip.playhead), (2, None, 20.0))

    def test_the_bar_shows_the_picked_channel_in_its_colour_and_no_settings_where_no_channel_is_picked(self) -> None:
        panel, bar = self.panel, self.panel.bar
        self.assertEqual((bar.song, bar.channel, bar.name.text, bar.name.colour), ("walk", "air", "air", theme.SOUND))
        self.assertEqual((bar.inst.options[bar.inst.picked], bar.level.visible, bar.word.visible),
                         ("tone", True, False))
        click(self.ui, spot(panel.list, "kit"))
        self.assertEqual((bar.channel, bar.name.colour, bar.inst.options[bar.inst.picked], bar.word.visible),
                         ("kit", theme.MUSTARD, "thud", True))
        # The channel is the song's, so another pattern leaves the bar as it is.
        click(self.ui, block(panel, 1))
        self.assertEqual((bar.channel, bar.level.visible), ("kit", True))
        # A song with no channels has no settings to show, and one that is not shown has no bar.
        self.stand.model.pick("song", "bare")
        panel.update()
        self.assertEqual((bar.song, bar.channel, bar.level.visible, bar.inst.visible, bar.follow.visible),
                         ("bare", None, False, False, True))
        self.stand.model.pick("sound", "ping")
        panel.update()
        self.assertEqual([widget for widget in panel.widgets if widget.visible], [])

    def test_the_ghosts_toggle_turns_the_ghosts_of_the_roll_off_and_on(self) -> None:
        panel = self.panel
        self.assertTrue(panel.roll.ghosts)
        for on in (False, True):
            click(self.ui, middle(panel.bar.ghosts.rect))
            panel.update()
            self.assertEqual((panel.bar.ghosts.is_on, panel.roll.ghosts), (on, on))

    def test_a_sound_or_an_instrument_picked_shows_nothing_and_the_song_picked_again_starts_anew(self) -> None:
        panel, model = self.panel, self.stand.model
        # In a panel this narrow the 40 rows of the song do not fit the strip, and the wheel moves its view.
        panel.place((124, 22, 40, 300))
        self.ui.handle(Event("wheel", middle(panel.strip.rect), wheel=-1))
        click(self.ui, spot(panel.list, "low"))
        click(self.ui, block(panel, 1))
        self.assertEqual((panel.strip.picked, panel.strip.offset, panel.channel), (1, 8, "low"))
        for thing in (("sound", "ping"), ("instrument", "tone")):
            model.pick(*thing)
            self.play(2.0)
            self.assertEqual((self.seen(), panel.roll.playhead, panel.strip.playhead), ([], None, None))
        model.pick("song", "walk")
        self.stand.playing = False
        panel.update()
        self.assertEqual(self.seen(), ["strip", "list", "follow", "ghosts", "roll"])
        self.assertEqual((panel.strip.picked, panel.strip.offset, panel.channel), (0, 0, "air"))

    def test_a_song_with_no_order_and_no_channels_shows_the_headings_and_no_view(self) -> None:
        panel = self.panel
        self.stand.model.pick("song", "bare")
        self.play(1.0)
        self.assertEqual(self.seen(), ["strip", "list", "follow"])
        self.assertEqual((order(panel), panel.strip.picked, panel.list.rows, panel.channel),
                         ([], None, HEADINGS, None))
        self.assertEqual((panel.strip.playhead, panel.roll.playhead, panel.steps.playhead), (None, None, None))

    def test_the_song_is_read_again_when_the_file_changed_with_the_view_where_it_was_and_at_no_other_time(
            self) -> None:
        panel = self.panel
        click(self.ui, block(panel, 1))
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
        self.assertEqual((panel.strip.picked, panel.channel), (1, "air"))
        self.assertEqual((panel.roll.row, panel.roll.top, panel.roll.row_w), view)
        # A song that lost the picked channel outside shows the first one its pattern has notes for, where it was.
        less = copy.deepcopy(PROJECT)
        song = less["songs"]["walk"]
        del song["channels"]["air"], song["patterns"]["a"]["air"], song["patterns"]["b"]["air"]
        self.outside(less)
        panel.update()
        self.assertEqual((order(panel), panel.strip.picked, panel.channel), (["a", "b", "a"], 1, "hush"))

    def test_the_strip_is_shown_the_song_when_it_is_read_and_not_every_frame(self) -> None:
        with mock.patch.object(SongStrip, "show", autospec=True, side_effect=SongStrip.show) as shown:
            panel = SongPanel(self.stand, Ui())
            panel.place(RECT)
            for seconds in (0.0, 0.5, 2.0):
                self.stand.playing, self.stand.at = True, seconds
                panel.update()
            self.assertEqual((shown.call_count, panel.strip.picked), (1, 1))
            self.outside(changed("b", "air", "o4 l8 g f e d c"))
            for _ in range(3):
                panel.update()
            self.assertEqual(shown.call_count, 2)
            # A sound is not the strip's to show, and the song picked again is.
            self.stand.model.pick("sound", "ping")
            panel.update()
            self.stand.model.pick("song", "walk")
            for _ in range(3):
                panel.update()
            self.assertEqual(shown.call_count, 3)
        self.assertEqual((order(panel), [note.pitch for note in panel.roll.notes]),
                         (["a", "b", "a"], [67, 65, 64, 62, 60]))

    def test_an_order_that_came_back_shorter_has_the_picked_place_inside_it(self) -> None:
        panel = self.panel
        click(self.ui, block(panel, 2))
        click(self.ui, spot(panel.list, "low"))
        self.assertEqual((panel.strip.picked, panel.roll.pattern.name), (2, "a"))
        short = copy.deepcopy(PROJECT)
        short["songs"]["walk"]["order"] = ["a", "b"]
        self.outside(short)
        panel.update()
        # An order that is another starts at its first block, and the channel that is still there stays.
        self.assertEqual((order(panel), panel.strip.picked, panel.roll.pattern.name, panel.channel),
                         (["a", "b"], 0, "a", "low"))
        self.assertEqual(self.seen(), ["strip", "list", "follow", "ghosts", "roll"])
        # The playing goes through the order as it is now: 3 seconds in, the song of 24 rows has started again.
        self.play(2.0)
        self.assertEqual((panel.strip.picked, panel.roll.pattern.name, panel.strip.playhead), (1, "b", 20.0))
        self.play(3.0)
        self.assertEqual((panel.strip.picked, panel.strip.playhead, panel.roll.playhead), (0, 6.0, 6.0))

    def test_a_song_the_engine_refuses_shows_the_words_of_the_refusal_until_it_is_mended(self) -> None:
        panel = self.panel
        click(self.ui, block(panel, 1))
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
        self.assertEqual((self.seen(), panel.fault.text), (["strip", "list", "follow", "ghosts", "roll"], ""))
        self.assertEqual((panel.strip.picked, panel.channel), (1, "air"))

    def test_placed_at_two_sizes_every_widget_is_inside_the_panel_and_the_strip_is_above_the_rest(self) -> None:
        panel, bar = self.panel, self.panel.bar
        self.assertEqual(arrange(RECT), {
            "strip": (128, 26, 654, 74), "channels": (128, 104, 128, 418), "choice": (131, 107, 122, 13),
            "song": (131, 122, 122, 28), "list": (131, 152, 122, 367), "beside": (259, 107, 520, 13),
            "view": (260, 104, 522, 418), "bar": (263, 107, 516, 37),
            "body": (263, 146, 516, 373), "mixer": (128, 104, 654, 418), "desk": (131, 122, 648, 397)})
        # The strip, the choice, the list, the eleven widgets of the bar, the roll, the steps and the line of a
        # fault, the six of the song's own settings, the fourteen of the layers, and then those of the mixer: its
        # desk, five of the song's strip, and six for each of four channels.
        self.assertEqual((len(panel.widgets), len(panel.mixer.widgets)), (23 + 14 + 30, 30))
        self.assertEqual((self.ui.widgets, panel.widgets[23:37], panel.widgets[37:]),
                         (panel.widgets, panel.layers.widgets, panel.mixer.widgets))
        self.assertEqual((panel.choice.rect, panel.choice.options, panel.mixing, panel.layering),
                         ((131, 107, 122, 13), (NOTES, MIXER, LAYERS), False, False))
        # The song's own settings are two lines under the choice: a word and then what is set, the tempo over
        # the key. With the mixer they are on the line of the choice, beside it, the tempo first.
        self.assertEqual(song_parts(arrange(RECT)["song"]), {
            "tempo_word": (131, 122, 32, 13), "tempo": (163, 122, 90, 13), "key_word": (131, 137, 32, 13),
            "lower": (163, 137, 13, 13), "shifted": (176, 137, 64, 13), "higher": (240, 137, 13, 13)})
        self.assertEqual(song_parts(arrange(RECT)["beside"]), {
            "tempo_word": (259, 107, 32, 13), "tempo": (291, 107, 90, 13), "key_word": (393, 107, 32, 13),
            "lower": (425, 107, 13, 13), "shifted": (438, 107, 64, 13), "higher": (502, 107, 13, 13)})
        self.assertEqual((panel.tempo.rect, panel.higher.rect, panel.tempo.visible), ((163, 122, 90, 13),
                                                                                      (240, 137, 13, 13), True))
        # The bar is laid out along the top of the view, the two toggles at its right end, and all of it fits.
        parts = channelbar.arrange(arrange(RECT)["bar"])
        self.assertEqual((bar.level.rect, bar.pan.rect, bar.follow.rect, bar.ghosts.rect),
                         (parts["level"], parts["pan"], (691, 119, 43, 13), (736, 119, 43, 13)))
        self.assertEqual((bar.level.visible, bar.pan.visible), (True, True))
        for rect in (RECT, SMALL, (0, 0, 30, 5)):
            panel.place(rect)
            self.assertTrue(all(lies_in(widget.rect, rect) for widget in panel.widgets), rect)
            parts = arrange(rect)
            self.assertTrue(all(lies_in(part, rect) for part in parts.values()), rect)
            # The strip ends over the list of the channels and over the view.
            foot = panel.strip.rect[1] + panel.strip.rect[3]
            self.assertTrue(all(foot <= parts[part][1] for part in ("channels", "view")), rect)
        # In the least panel the strip is as high as in any and the list of the channels as wide, and the view
        # has what is left. The choice is over the list, and the mixer has all that is under the strip.
        panel.place(SMALL)
        self.assertEqual({part: arrange(SMALL)[part] for part in ("choice", "list", "mixer", "desk")},
                         {"choice": (131, 107, 122, 13), "list": (131, 152, 122, 163), "mixer": (128, 104, 492, 214),
                          "desk": (131, 122, 486, 193)})
        self.assertEqual((panel.strip.rect, panel.list.rect), ((128, 26, 492, 74), (131, 152, 122, 163)))
        self.assertEqual((panel.roll.rect, panel.steps.rect, panel.fault.rect),
                         ((263, 146, 354, 169), (263, 146, 354, 169), SMALL))
        # There the bar has no room for the pan, which is not seen. The level stays, and so do the toggles.
        self.assertEqual(arrange(SMALL)["bar"], (263, 107, 354, 37))
        self.assertEqual((bar.level.visible, bar.pan.visible, bar.follow.rect, bar.ghosts.rect),
                         (True, False, (529, 119, 43, 13), (574, 119, 43, 13)))

    def test_taken_off_the_window_nothing_is_seen_and_brought_back_only_what_was_shown(self) -> None:
        self.panel.show(False)
        self.assertEqual(self.seen(), [])
        self.panel.update()
        self.assertEqual(self.seen(), [])
        self.panel.show(True)
        self.assertEqual(self.seen(), ["strip", "list", "follow", "ghosts", "roll"])


class Opened(unittest.TestCase):
    """What the tests of saving share: a project file of the test's own, a real session on it, and the panel."""

    def opened(self, put: Callable[[Path], object]) -> None:
        """Have `put` make the file in a folder of the test's own, and open it with a stand-in player and a clock."""
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        # No thread of the model is left to run into the next test, or into a folder that is gone.
        self.addCleanup(settle)
        self.path = Path(folder.name) / "project.json"
        put(self.path)
        self.now, self.stamp = 100.0, STAMP
        self.speaker = Speaker()
        self.session = Session(Model(self.path, RATE), self.speaker, lambda: self.now)
        self.ui = Ui()
        self.panel = SongPanel(self.session, self.ui)
        self.panel.place(RECT)
        self.panel.update()
        self.roll, self.steps = self.panel.roll, self.panel.steps

    def data(self) -> dict:
        """What the file holds now."""
        return json.loads(self.path.read_text(encoding="utf-8"))

    def text(self, pattern: str, channel: str, song: str = "walk") -> str | None:
        """The text the file has for a channel in a pattern of a song, or None where it has none."""
        return self.data()["songs"][song]["patterns"][pattern].get(channel)

    def read(self, pattern: str, channel: str, song: str = "walk") -> tuple[Note, ...]:
        """The notes of a channel in a pattern of a song, read from the file."""
        found = songview.pattern(Project.load(self.path).song(song), pattern)
        return next(held.notes for held in found.channels if held.name == channel)

    def outside(self, pattern: str, channel: str, text: str) -> None:
        """Have another program write the file with other text for a channel, and the session look at it."""
        project = self.data()
        project["songs"]["walk"]["patterns"][pattern][channel] = text
        self.path.write_text(json.dumps(project), encoding="utf-8")
        # Stamped as no write before it was, so that the look cannot take the file for the one it knows.
        self.stamp += 10**9
        os.utime(self.path, ns=(self.stamp, self.stamp))
        self.now += POLL_EVERY
        self.session.tick()

    def at(self, row: int, pitch: int) -> tuple[int, int]:
        """A pixel in the middle of a row and of a key of the roll, as it is scrolled now."""
        seen = self.roll.map
        return seen.x(row) + 4, seen.y(pitch) + 4

    def square(self, lane: int, row: int) -> tuple[int, int]:
        """A pixel on the square of a step, as the steps are scrolled now."""
        x, y, _, _ = self.steps.grid.square(lane, row)
        return x + 2, y + 9

    def mouse(self, kind: str, pos: tuple[int, int], button: int = 0) -> None:
        """Hand the Ui one thing the mouse did."""
        self.ui.handle(Event(kind, pos, button=button))

    def drag(self, *stops: tuple[int, int], button: int = 1) -> None:
        """Press at the first position, move to each of the others in turn, and release at the last."""
        self.mouse("down", stops[0], button)
        for stop in stops[1:]:
            self.mouse("move", stop)
        self.mouse("up", stops[-1], button)

    def key(self, key: str, *mods: str) -> bool:
        """Hand the Ui a key, some others held. Whether it was used."""
        return self.ui.handle(Event("key", key=key, mods=frozenset(mods)))

    def pitches(self) -> list[float]:
        """The pitches of the notes the roll shows."""
        return [note.pitch for note in self.roll.notes]

    def kicks(self) -> list[float]:
        """The rows at which the steps show a hit of the letter k in the channel kit."""
        return [note.row for note in self.steps.notes("kit") if note.drum == "k"]

    def heard(self) -> list[Sequence]:
        """The sides of each note the player was asked to sound over what plays, in order."""
        return [asked[1] for asked in self.speaker.asked if asked[0] == "also"]

    def tone(self, inst: str, pitch: str | float) -> Sequence:
        """The sides of a note as the model keeps it for the song: the same ones for as long as nothing is loaded."""
        return self.session.model.tone(inst, pitch, song="walk")["sides"]


class Saved(Opened):
    def setUp(self) -> None:
        self.opened(lambda path: path.write_text(json.dumps(EDITED), encoding="utf-8"))

    def test_a_click_on_the_empty_grid_puts_a_note_in_the_text_of_its_channel_and_nothing_else_changes(self) -> None:
        before = self.data()
        self.assertEqual((self.panel.channel, self.roll.pattern.name, self.pitches()), ("air", "a", [60, 64, 67, 64]))
        self.drag(self.at(4, 70))
        # The note is four rows long, and takes the place of the one that began where it does.
        self.assertEqual(self.text("a", "air"), "o4 l4 c b- g e")
        self.assertEqual(differing(before, self.data()), ["songs.walk.patterns.a.air"])
        # The roll shows what the file holds, and the step is called by where it is and by what was done.
        self.assertEqual((self.pitches(), self.roll.notes), ([60, 70, 67, 64], self.read("a", "air")))
        self.assertEqual((self.session.undo_label, self.session.message), ("air of pattern a: note placed", ""))

    def test_undo_takes_the_note_out_of_the_file_and_out_of_the_roll_and_redo_brings_it_back(self) -> None:
        before = self.data()
        self.drag(self.at(4, 70))
        self.assertTrue(self.session.undo())
        self.panel.update()
        self.assertEqual((self.data(), self.pitches()), (before, [60, 64, 67, 64]))
        self.assertEqual((self.session.can_undo, self.session.redo_label), (False, "air of pattern a: note placed"))
        self.assertTrue(self.session.redo())
        self.panel.update()
        self.assertEqual((self.text("a", "air"), self.pitches()), ("o4 l4 c b- g e", [60, 70, 67, 64]))

    def test_a_note_that_is_dragged_and_one_that_is_removed_are_each_a_step_in_the_file(self) -> None:
        self.drag(self.at(12, 64), self.at(12, 66))
        self.assertEqual((self.text("a", "air"), self.session.undo_label),
                         ("o4 l4 c e g g-", "air of pattern a: note moved"))
        self.drag(self.at(0, 60), button=3)
        self.assertEqual((self.text("a", "air"), self.session.undo_label),
                         ("o4 l4 r e g g-", "air of pattern a: note removed"))
        self.assertEqual((self.pitches(), self.roll.notes), ([64, 67, 66], self.read("a", "air")))
        # Each is taken back by itself.
        for text in ("o4 l4 c e g g-", "o4 l4 c e g e"):
            self.session.undo()
            self.assertEqual(self.text("a", "air"), text)
        self.panel.update()
        self.assertEqual(self.pitches(), [60, 64, 67, 64])

    def test_a_hit_put_in_the_steps_is_in_the_text_of_its_own_drum_channel(self) -> None:
        self.session.pick("song", "pair")
        self.panel.update()
        click(self.ui, spot(self.panel.list, "top"))
        before = self.data()
        # The lanes are the k of low, and the k and the h of top. The second has no hit in this pattern.
        self.drag(self.square(1, 0))
        self.assertEqual((self.text("a", "top", "pair"), self.session.undo_label),
                         ("l8 k h r h r h r h", "top of pattern a: hit placed"))
        self.assertEqual(differing(before, self.data()), ["songs.pair.patterns.a.top"])
        self.assertEqual(self.steps.notes("top"), self.read("a", "top", "pair"))
        self.drag(self.square(0, 2))
        self.assertEqual((self.text("a", "low", "pair"), self.session.undo_label),
                         ("k8 k8 k4 k4 k4", "low of pattern a: hit placed"))
        self.assertEqual(differing(before, self.data()), ["songs.pair.patterns.a.low", "songs.pair.patterns.a.top"])
        self.assertEqual(self.steps.notes("low"), self.read("a", "low", "pair"))

    def test_a_channel_the_pattern_has_no_text_for_gets_its_text_and_undo_takes_it_away_again(self) -> None:
        click(self.ui, spot(self.panel.list, "hush"))
        before = self.data()
        self.assertEqual((self.text("a", "hush"), self.roll.notes), (None, ()))
        self.drag(self.at(4, 60))
        self.assertEqual((self.text("a", "hush"), self.session.undo_label),
                         ("o4 r4 c4 r2", "hush of pattern a: note placed"))
        self.assertEqual(differing(before, self.data()), ["songs.walk.patterns.a.hush"])
        self.assertEqual(self.roll.notes, (Note(4.0, 4.0, 60, 15, "tone"),))
        # In the list the channel is no longer one the pattern has no text for.
        self.assertFalse(next(row for row in self.panel.list.rows if row.key == "hush").dim)
        self.session.undo()
        self.panel.update()
        self.assertEqual((self.data(), self.roll.notes), (before, ()))

    def test_a_change_that_is_refused_leaves_the_file_and_the_view_is_said_and_its_note_is_not_heard(self) -> None:
        before, notes = self.path.read_text(encoding="utf-8"), self.roll.notes
        saved, save = [], self.roll.edited
        self.roll.edited = lambda *edit: saved.append(save(*edit)) or saved[-1]
        # Notes that cannot be written as text are not saved, and the session is told where and why.
        with mock.patch.object(songview, "text", side_effect=RowSynthError("row 4: no")):
            self.drag(self.at(4, 70))
        self.assertEqual((self.path.read_text(encoding="utf-8"), self.roll.notes, self.session.can_undo),
                         (before, notes, False))
        self.assertEqual((self.session.message, self.session.messages), ("air of pattern a: row 4: no", 1))
        # Text that names an instrument nobody has is what the check of the project refuses.
        with mock.patch.object(songview, "text", return_value="o4 @nobody c1"):
            self.drag(self.at(4, 70))
        self.panel.update()
        self.assertEqual((self.path.read_text(encoding="utf-8"), self.roll.notes, self.session.can_undo),
                         (before, notes, False))
        self.assertEqual((self.session.message, self.session.messages),
                         ("songs.walk.patterns.a.air: no instrument named 'nobody'", 2))
        # The roll was answered that neither was saved, and neither note was heard.
        self.assertEqual((saved, self.heard()), ([False, False], []))
        # The next change that is made is saved as any other, and its note is heard.
        self.drag(self.at(4, 70))
        self.assertEqual((self.text("a", "air"), self.session.message), ("o4 l4 c b- g e", ""))
        self.assertEqual((saved, len(self.heard())), ([False, False, True], 1))

    def test_a_hit_that_is_not_saved_was_heard_at_its_press_and_the_steps_are_answered_that_it_was_not(self) -> None:
        click(self.ui, spot(self.panel.list, "kit"))
        saved, save = [], self.steps.edited
        self.steps.edited = lambda *edit: saved.append(save(*edit)) or saved[-1]
        with mock.patch.object(songview, "text", side_effect=RowSynthError("row 2: no")):
            self.drag(self.square(0, 2))
        self.assertEqual((saved, len(self.heard()), self.kicks(), self.session.message),
                         ([False], 1, [0.0, 8.0], "kit of pattern a: row 2: no"))
        self.drag(self.square(0, 2))
        self.assertEqual((saved, len(self.heard()), self.kicks(), self.session.message),
                         ([False, True], 2, [0.0, 2.0, 8.0], ""))

    def test_a_placed_note_is_heard_with_the_instrument_of_its_channel_and_a_hit_with_what_its_letter_plays(
            self) -> None:
        click(self.ui, spot(self.panel.list, "low"))
        self.drag(self.at(4, 40))
        self.assertEqual((self.text("a", "low"), len(self.heard())), ("o2 c4 e4 g2", 1))
        self.assertIs(self.heard()[0], self.tone("buzz", 40))
        click(self.ui, spot(self.panel.list, "kit"))
        # The hit is heard as it is put, which is at the press: the lane is the one of the letter h.
        self.mouse("down", self.square(1, 2), 1)
        self.assertEqual(len(self.heard()), 2)
        self.assertIs(self.heard()[1], self.tone("thud", "c6"))
        self.mouse("up", self.square(1, 2), 1)
        self.assertEqual((len(self.heard()), self.session.undo_label), (2, "kit of pattern a: hit placed"))

    def test_a_key_of_the_piano_sounds_its_note_from_the_lowest_c_in_view_and_changes_nothing(self) -> None:
        click(self.ui, spot(self.panel.list, "low"))
        before = self.path.read_text(encoding="utf-8")
        # Without the focus the roll is handed no key.
        self.assertEqual((self.key("x"), self.heard()), (False, []))
        # A press on a key of the strip, here that of e2, is heard with the channel's instrument. It picks nothing,
        # and the roll has the focus. The lowest c among its keys is c2.
        self.drag((self.roll.rect[0] + 5, self.at(0, 40)[1]))
        self.assertEqual((self.ui.focus, self.roll.picked, lowest_c(self.roll.map.pitches())),
                         (self.roll, frozenset(), 36))
        self.assertEqual(len(self.heard()), 1)
        self.assertIs(self.heard()[0], self.tone("buzz", 40))
        # x is the d of the first octave and q the c of the second. With ctrl held, and on f1, no note is played.
        self.assertEqual([self.key("x"), self.key("q", "shift"), self.key("x", "ctrl"), self.key("f1")],
                         [True, True, False, False])
        self.assertEqual(len(self.heard()), 3)
        self.assertIs(self.heard()[1], self.tone("buzz", 38))
        self.assertIs(self.heard()[2], self.tone("buzz", 48))
        # With the keys scrolled up an octave, the piano is an octave higher.
        self.ui.handle(Event("wheel", self.at(0, 36), wheel=6))
        self.assertTrue(self.key("z"))
        self.assertIs(self.heard()[3], self.tone("buzz", 48))
        self.assertEqual((self.path.read_text(encoding="utf-8"), self.session.can_undo), (before, False))

    def test_the_space_bar_and_the_keys_of_undo_and_redo_are_not_used_up_by_the_roll(self) -> None:
        # The keys as the app has them.
        self.ui.keys.update({("space", frozenset()): self.session.toggle,
                             ("z", frozenset({"ctrl"})): self.session.undo,
                             ("y", frozenset({"ctrl"})): self.session.redo})
        self.drag(self.at(4, 70))
        self.assertIs(self.ui.focus, self.roll)
        self.assertTrue(self.key("z", "ctrl"))
        self.panel.update()
        self.assertEqual((self.text("a", "air"), self.pitches()), ("o4 l4 c e g e", [60, 64, 67, 64]))
        self.assertTrue(self.key("y", "ctrl"))
        self.panel.update()
        self.assertEqual((self.text("a", "air"), self.pitches()), ("o4 l4 c b- g e", [60, 70, 67, 64]))
        self.assertFalse(self.session.playing)
        self.assertTrue(self.key("space"))
        self.assertTrue(self.session.playing)
        # None of the three was taken for a key of the piano: only the note that was placed was heard.
        self.assertEqual(len(self.heard()), 1)

    def test_the_notes_an_edit_leaves_picked_are_picked_in_what_its_save_shows(self) -> None:
        self.drag(self.at(12, 64), self.at(12, 66))
        self.assertEqual((self.pitches(), self.roll.picked), ([60, 64, 67, 66], {3}))
        self.panel.update()
        self.assertEqual(self.roll.picked, {3})
        # With the focus the roll has the keys. Two in one frame: the second moves what the first one saved.
        self.assertEqual([self.key("up"), self.key("up", "shift")], [True, True])
        self.assertEqual((self.pitches(), self.roll.picked), ([60, 64, 67, 79], {3}))
        self.assertEqual((self.text("a", "air"), self.session.undo_label),
                         ("o4 l4 c e g > g", "air of pattern a: note moved"))

    def test_a_save_leaves_the_view_where_it_is_and_the_song_playing(self) -> None:
        for steps, mods in ((2, ()), (-1, ("shift",)), (1, ("ctrl",))):
            self.ui.handle(Event("wheel", self.at(8, 64), wheel=steps, mods=frozenset(mods)))
        view = (self.roll.row, self.roll.top, self.roll.row_w)
        self.assertNotEqual(view, (0.0, 79.0, 9))
        self.session.play()
        settle()
        self.session.tick()
        self.assertTrue(self.speaker.playing)
        self.drag(self.at(8, 70))
        self.assertEqual(self.text("a", "air"), "o4 l4 c e b- e")
        settle()
        self.session.tick()
        self.panel.update()
        self.assertEqual(((self.roll.row, self.roll.top, self.roll.row_w), self.panel.strip.picked), (view, 0))
        # The loop was not stopped: it was given the song as it now is, in the place of what it played.
        self.assertEqual([asked[0] for asked in self.speaker.asked], ["play", "also", "swap"])
        self.assertTrue(self.session.playing)

    def test_a_change_from_outside_during_a_gesture_in_the_roll_is_shown_when_the_gesture_has_ended(self) -> None:
        self.mouse("down", self.at(12, 64), 1)
        self.mouse("move", self.at(12, 66))
        self.outside("a", "air", "o4 l4 c e g > c")
        for _ in range(2):
            self.panel.update()
        # The note is where the hand has it, among the notes there were.
        self.assertEqual((self.roll.busy, self.pitches(), [note.pitch for note in self.roll.drawn]),
                         (True, [60, 64, 67, 64], [60, 64, 67, 66]))
        # Let go where it was pressed, the gesture changes nothing. The next frame shows what the file holds.
        self.mouse("move", self.at(12, 64))
        self.mouse("up", self.at(12, 64), 1)
        self.assertEqual((self.roll.busy, self.pitches()), (False, [60, 64, 67, 64]))
        self.panel.update()
        self.assertEqual((self.pitches(), self.text("a", "air")), ([60, 64, 67, 72], "o4 l4 c e g > c"))

    def test_a_drag_in_the_steps_that_ends_after_a_change_from_outside_stands_and_undo_brings_the_other_back(
            self) -> None:
        click(self.ui, spot(self.panel.list, "kit"))
        self.mouse("down", self.square(0, 2), 1)
        self.outside("a", "kit", "l4 k k k k")
        self.panel.update()
        self.assertEqual((self.steps.busy, self.kicks()), (True, [0.0, 2.0, 8.0]))
        # The release saves the hits as the hand left them, which is the later of the two changes, and it is said.
        self.mouse("up", self.square(0, 2), 1)
        self.assertEqual((self.text("a", "kit"), self.kicks()), ("k8 k8 h4 k4 h4", [0.0, 2.0, 8.0]))
        self.assertEqual(self.session.message, "kit of pattern a was changed outside while it was edited here: "
                                               "this edit stands, and undo brings the other back")
        self.session.undo()
        self.panel.update()
        self.assertEqual((self.text("a", "kit"), self.kicks()), ("l4 k k k k", [0.0, 4.0, 8.0, 12.0]))

    def test_an_edit_that_ends_after_another_channel_was_changed_outside_is_saved_beside_it_and_nothing_is_said(
            self) -> None:
        self.mouse("down", self.at(12, 64), 1)
        self.mouse("move", self.at(12, 66))
        self.outside("a", "low", "o2 l2 g c")
        self.panel.update()
        self.assertEqual(self.session.message, "changed outside: walk")
        self.mouse("up", self.at(12, 66), 1)
        self.assertEqual((self.text("a", "air"), self.text("a", "low"), self.session.message),
                         ("o4 l4 c e g g-", "o2 l2 g c", ""))

    def test_the_picked_place_does_not_go_along_with_the_playing_during_a_gesture_and_does_after_it(self) -> None:
        self.mouse("down", self.at(12, 64), 1)
        self.mouse("move", self.at(12, 66))
        # Two seconds in, the song is in its second pattern, which is not the one the hand is in.
        self.speaker.playing, self.speaker.at = True, 2.0
        for _ in range(2):
            self.panel.update()
        self.assertEqual((self.panel.bar.follow.is_on, self.panel.strip.picked, self.roll.pattern.name,
                          self.roll.busy), (True, 0, "a", True))
        self.assertEqual((self.panel.strip.playhead, self.roll.playhead), (20.0, None))
        # The release saves the note in the pattern it was moved in, and the next frame goes along.
        self.mouse("up", self.at(12, 66), 1)
        self.assertEqual((self.text("a", "air"), self.text("b", "air"), self.panel.strip.picked),
                         ("o4 l4 c e g g-", "o4 l8 g f e d", 0))
        self.panel.update()
        self.assertEqual((self.panel.strip.picked, self.roll.pattern.name, self.roll.playhead), (1, "b", 4.0))

    def test_the_steps_have_the_drum_table_and_a_run_of_the_wheel_is_saved_when_its_time_is_up(self) -> None:
        click(self.ui, spot(self.panel.list, "kit"))
        self.steps.clock = lambda: self.now
        self.mouse("move", self.square(0, 0))
        self.assertEqual(self.ui.help, "take the hit away: k, bar 1, step 1 (level 15)")
        for _ in range(3):
            self.ui.handle(Event("wheel", self.square(0, 0), wheel=-1))
        self.panel.update()
        self.assertEqual((self.steps.busy, self.session.can_undo), (True, False))
        # The panel gives the steps their frame, and the frame after the wait ends the run.
        self.now += WHEEL_WAIT
        self.panel.update()
        self.assertEqual((self.steps.busy, self.text("a", "kit"), self.session.undo_label),
                         (False, "l4 v12 k v15 h k h", "kit of pattern a: level changed"))


    def test_a_setting_changed_in_the_bar_is_one_place_in_the_file_and_the_next_frame_shows_it(self) -> None:
        bar, before = self.panel.bar, self.data()
        # The channel plays the first instrument of the project. A click on the right half of the field is the next.
        x, y, w, _ = bar.inst.rect
        click(self.ui, (x + w - 3, y + 3))
        self.assertEqual(differing(before, self.data()), ["songs.walk.channels.air.inst"])
        self.assertEqual((self.data()["songs"]["walk"]["channels"]["air"]["inst"], self.session.undo_label),
                         ("thud", "air: instrument thud"))
        self.assertEqual(len(self.heard()), 1)
        self.assertIs(self.heard()[0], self.tone("thud", "c4"))
        # The frame after it, the list and the notes of the roll have the instrument the file has.
        self.panel.update()
        self.assertEqual(next(row for row in self.panel.list.rows if row.key == "air").tag, "thud")
        self.assertEqual({note.inst for note in self.roll.notes}, {"thud"})

    def test_the_panel_gives_the_bar_its_frame_so_a_run_of_the_wheel_on_a_knob_is_saved_when_its_time_is_up(
            self) -> None:
        level = self.panel.bar.level
        level.clock = lambda: self.now
        for _ in range(3):
            self.ui.handle(Event("wheel", middle(level.rect), wheel=1))
        self.panel.update()
        self.assertEqual((level.busy, level.value, self.session.can_undo), (True, 1.03, False))
        self.now += dials.WHEEL_WAIT
        self.panel.update()
        self.assertEqual((level.busy, self.data()["songs"]["walk"]["channels"]["air"], self.session.undo_label),
                         (False, {"inst": "tone", "gain": 1.03}, "air: level 1.03"))


class Mixed(Opened):
    def setUp(self) -> None:
        self.opened(lambda path: path.write_text(json.dumps(EDITED), encoding="utf-8"))
        self.mixer = self.panel.mixer

    def strip(self, channel: str) -> Strip:
        """The strip the mixer has for a channel."""
        return next(strip for strip in self.mixer.strips if strip.channel == channel)

    def dim(self) -> list[Hashable]:
        """The channels whose rows in the list are dim."""
        return [row.key for row in self.panel.list.rows if row.dim]

    def test_the_choice_puts_the_mixer_in_place_of_the_notes_and_no_widget_of_the_part_taken_off_has_the_mouse(
            self) -> None:
        panel, mixer, before = self.panel, self.mixer, self.data()
        notes = [widget for widget in panel.widgets if widget.visible]
        self.assertEqual((panel.mixing, notes[:3], set(notes) & set(mixer.widgets)),
                         (False, [panel.strip, panel.choice, panel.list], set()))
        grid, fader, mute = self.at(4, 70), middle(self.strip("hush").fader.rect), middle(self.strip("hush").mute.rect)
        choose(self.ui, panel, MIXER)
        # The song's own settings stay: they were under the choice, and are beside it now.
        settings = panel.widgets[17:23]
        self.assertEqual((panel.mixing, [widget for widget in panel.widgets if widget.visible]),
                         (True, [panel.strip, panel.choice, *settings, *mixer.widgets]))
        self.assertEqual((settings[0], panel.tempo.rect, panel.higher.rect),
                         (panel.tempo, (291, 107, 90, 13), (502, 107, 13, 13)))
        # Where the roll was a press places no note, and no key plays one.
        self.drag(grid)
        self.assertEqual((self.data(), self.session.can_undo, self.key("z"), self.heard()), (before, False, False, []))
        # The strip is there with either, and a block picked in it is picked for the notes too.
        click(self.ui, block(panel, 1))
        self.assertEqual((panel.strip.picked, panel.mixing), (1, True))
        choose(self.ui, panel, NOTES)
        self.assertEqual(([widget for widget in panel.widgets if widget.visible], self.roll.pattern.name),
                         (notes, "b"))
        # Where a fader and a toggle of the mixer were, a press sets no level and mutes nothing.
        click(self.ui, fader)
        click(self.ui, mute)
        self.assertEqual((self.data(), self.session.muted), (before, frozenset()))

    def test_layers_in_the_choice_puts_the_layers_and_the_intro_in_place_of_the_notes_and_of_the_mixer(self) -> None:
        panel, layers, before = self.panel, self.panel.layers, self.data()
        notes = [widget for widget in panel.widgets if widget.visible]
        grid = self.at(4, 70)
        choose(self.ui, panel, LAYERS)
        # The song has no layers: the part shows its switch, the line that says what layers are for, and the
        # switch of the intro. The song's own settings are beside the choice, as with the mixer.
        self.assertEqual((panel.layering, panel.mixing, [widget for widget in panel.widgets if widget.visible]),
                         (True, False, [panel.strip, panel.choice, *panel.widgets[17:23], layers.use, layers.about,
                                        layers.intro]))
        self.assertEqual((layers.use.rect, panel.tempo.rect), ((131, 122, 67, 13), (291, 107, 90, 13)))
        # Where the roll was a press places no note.
        self.drag(grid)
        self.assertEqual((self.data(), self.session.can_undo), (before, False))
        # Switched on there, the layers are in the file, and the part shows them when the song is read again.
        click(self.ui, middle(layers.use.rect))
        panel.update()
        self.assertEqual((self.data()["songs"]["walk"]["layers"]["versions"], panel.layering, layers.table.visible),
                         ({"full": ["all"]}, True, True))
        choose(self.ui, panel, MIXER)
        self.assertEqual(([widget for widget in layers.widgets if widget.visible], self.mixer.desk.visible),
                         ([], True))
        choose(self.ui, panel, NOTES)
        self.assertEqual([widget for widget in panel.widgets if widget.visible], notes)

    def test_the_choice_stays_while_the_song_is_picked_and_another_song_starts_with_its_notes(self) -> None:
        panel = self.panel
        choose(self.ui, panel, MIXER)
        # The song read again, after a change from outside, is the same song.
        self.outside("a", "air", "o4 l4 c e g a")
        panel.update()
        self.assertEqual((panel.mixing, self.mixer.desk.visible, self.pitches()), (True, True, [60, 64, 67, 69]))
        self.session.pick("song", "pair")
        panel.update()
        self.assertEqual((panel.mixing, self.mixer.desk.visible, self.roll.visible), (False, False, True))
        self.assertEqual([strip.channel for strip in self.mixer.strips], ["air", "low", "top", None])

    def test_a_press_on_the_name_of_a_channel_in_the_mixer_picks_it_and_shows_its_notes(self) -> None:
        panel = self.panel
        choose(self.ui, panel, MIXER)
        self.assertEqual([strip.channel for strip in self.mixer.strips if strip.name.picked], ["air"])
        click(self.ui, middle(self.strip("low").name.rect))
        self.assertEqual((panel.mixing, panel.channel, panel.list.picked, self.roll.visible, self.roll.channel),
                         (False, "low", "low", True, "low"))
        self.assertEqual([strip.channel for strip in self.mixer.strips if strip.name.picked], ["low"])
        # The name of a drum channel shows the steps.
        choose(self.ui, panel, MIXER)
        click(self.ui, middle(self.strip("kit").name.rect))
        self.assertEqual((panel.mixing, panel.channel, self.steps.visible, self.steps.picked),
                         (False, "kit", True, "kit"))

    def test_a_channel_that_is_not_heard_has_a_dim_row_in_the_list_as_one_with_no_text_has(self) -> None:
        panel, session = self.panel, self.session
        # The first channel has no text in the first pattern.
        self.assertEqual(self.dim(), ["hush"])
        heard = []
        for hear in (lambda: session.mute("air"), lambda: session.solo("low"), session.hear_all):
            hear()
            panel.update()
            heard.append(self.dim())
        self.assertEqual(heard, [["hush", "air"], ["hush", "air", "kit"], ["hush"]])
        # The mixer comes onto the window with what is muted lit and dim, before it has had a frame.
        session.mute("low")
        choose(self.ui, panel, MIXER)
        self.assertEqual([(strip.channel, strip.mute.is_on, strip.dim) for strip in self.mixer.strips[1:3]],
                         [("air", False, False), ("low", True, True)])
        # The panel gives the mixer its frame, in which it takes up what the session was told elsewhere.
        session.mute("low")
        panel.update()
        self.assertEqual((self.strip("low").mute.is_on, self.strip("low").dim), (False, False))
        # A mute made in the mixer is seen in the list when the notes are back, and the pick is where it was.
        click(self.ui, middle(self.strip("kit").mute.rect))
        choose(self.ui, panel, NOTES)
        panel.update()
        self.assertEqual((self.dim(), panel.list.picked, session.can_undo), (["hush", "kit"], "air", False))

    def test_the_knob_of_the_bar_and_the_fader_of_the_mixer_show_the_level_the_other_saved(self) -> None:
        panel, level, fader = self.panel, self.panel.bar.level, self.strip("air").fader
        x, y = middle(level.rect)
        self.drag((x, y), (x, y - 5))
        panel.update()
        self.assertEqual((self.data()["songs"]["walk"]["channels"]["air"], fader.value, self.strip("air").number.text),
                         ({"inst": "tone", "gain": 1.1}, 1.1, "1.1"))
        # The fader travels 350 pixels here: 35 of them are a tenth of its travel, which is 0.2.
        choose(self.ui, panel, MIXER)
        x, y, w, h = fader.handle()
        self.drag((x + w // 2, y + h // 2 - 1), (x + w // 2, y + h // 2 + 34))
        self.assertEqual((self.data()["songs"]["walk"]["channels"]["air"]["gain"], self.session.undo_label),
                         (0.9, "air: level 0.9"))
        panel.update()
        choose(self.ui, panel, NOTES)
        self.assertEqual((level.visible, level.value, level.show(level.value)), (True, 0.9, "0.9"))


class Piece(Opened):
    def setUp(self) -> None:
        self.opened(lambda path: shutil.copyfile(EXAMPLES / "piece.json", path))

    def test_one_note_of_the_piece_changed_is_the_text_of_one_channel_of_one_pattern_changed_in_its_file(self) -> None:
        before, notes = self.data(), self.roll.notes
        self.assertEqual((self.panel.channel, self.roll.pattern.name, len(notes), notes[0].pitch),
                         ("lead", "cold", 11, 62))
        # Its first note goes up a semitone.
        self.drag(self.at(0, 62), self.at(0, 63))
        self.assertEqual(differing(before, self.data()), ["songs.piece.patterns.cold.lead"])
        self.assertEqual(self.session.undo_label, "lead of pattern cold: note moved")
        # The file read again gives the notes the roll shows, which are the ones there were but for that note.
        self.assertEqual(self.roll.notes, (dataclasses.replace(notes[0], pitch=63), *notes[1:]))
        self.assertEqual(self.read("cold", "lead", "piece"), self.roll.notes)


class Chorded(Opened):
    def setUp(self) -> None:
        self.opened(lambda path: path.write_text(json.dumps(CHORDED), encoding="utf-8"))

    def ask(self, bar: int) -> list[str | None]:
        """Press on the place of a bar in the strip of the roll, and give what the menu that opens reads."""
        seen = self.roll.map
        click(self.ui, (seen.x(bar * 16) + 3, seen.chords[1] + 3))
        self.assertIsInstance(self.ui.modal, Menu)
        return [item and item.text for item in self.ui.modal.items]

    def choose(self, bar: int, text: str) -> None:
        """Press on the place of a bar in the strip, and choose the item of its menu that reads a text."""
        at = self.ask(bar).index(text)
        click(self.ui, middle(self.ui.modal.rows[at]))

    def chords(self) -> object:
        """The chords the file has for the song."""
        return self.data()["songs"]["tune"].get("chords")

    def shown(self) -> list[str]:
        """What the strip of the roll reads for each bar."""
        return [bar.text for bar in self.roll.chords.bars]

    def wrong(self) -> list[float]:
        """The rows of the notes of the roll that the chords it is shown have for wrong ones."""
        return [note.row for note in self.roll.notes if chordview.wrong(self.roll.chords, note)]

    def test_a_press_on_the_strip_opens_the_menu_and_a_chord_that_is_chosen_is_saved_as_one_step(self) -> None:
        before = self.data()
        self.assertEqual((self.shown(), self.wrong()), (["C", "G"], [20.0]))
        self.assertEqual(self.ask(1), ["C", "G", None, "none", "New chord..."])
        self.choose(1, "C")
        self.assertEqual((self.ui.modal, self.chords(), self.session.undo_label, self.session.message),
                         (None, {"a": ["C", "C"]}, "bar 2 of pattern a: chord C", ""))
        self.assertEqual(differing(before, self.data()), ["songs.tune.chords.a"])
        # The roll shows what the file holds: under C it is the d and the b of the second bar that are wrong.
        self.assertEqual((self.shown(), self.wrong()), (["C", "C"], [16.0, 28.0]))
        # The chord the bar has already is no step, and the one step is taken back.
        self.choose(1, "C")
        self.assertTrue(self.session.undo())
        self.panel.update()
        self.assertEqual((self.data(), self.shown(), self.wrong(), self.session.can_undo),
                         (before, ["C", "G"], [20.0], False))

    def test_none_clears_a_bar_and_the_last_bar_cleared_takes_the_pattern_out_of_the_chords(self) -> None:
        self.choose(1, "none")
        self.assertEqual((self.chords(), self.shown(), self.wrong(), self.session.undo_label),
                         ({"a": ["C"]}, ["C", ""], [], "bar 2 of pattern a: no chord"))
        self.choose(0, "none")
        self.assertEqual((self.chords(), self.shown(), self.session.undo_label, self.session.message),
                         ({}, ["", ""], "bar 1 of pattern a: no chord", ""))
        # A bar with no chord that is cleared is as it was, which is no step.
        self.choose(1, "none")
        self.assertEqual((self.chords(), self.session.undo_label), ({}, "bar 1 of pattern a: no chord"))
        # A bar that is cleared before one that has a chord keeps an empty string.
        self.assertTrue(self.session.undo() and self.session.undo())
        self.panel.update()
        self.choose(0, "none")
        self.assertEqual((self.chords(), self.shown(), self.wrong()), ({"a": ["", "G"]}, ["", "G"], [20.0]))

    def test_a_new_chord_is_added_to_the_chords_of_the_project_and_set_on_the_bar(self) -> None:
        self.choose(1, "New chord...")
        box = self.ui.modal
        self.assertIsInstance(box, ChordBox)
        # The name of a chord the project has is refused, and the box stays open for the name to be mended.
        self.key("return")
        self.assertEqual((self.ui.modal, self.session.message, self.session.can_undo),
                         (box, "the project has a chord called 'C' already", False))
        left, top, _, high = chordview.keys(chordview.box_parts(box.at, self.ui.size)["keys"])[2]
        click(self.ui, (left + 2, top + high - 2))
        click(self.ui, (box.kinds.cells()[1][0] + 2, box.kinds.rect[1] + 2))
        self.assertEqual(box.name.text, "Dm")
        self.key("return")
        self.assertEqual((self.ui.modal, self.data()["chords"]["Dm"], self.chords(), self.shown(), self.wrong()),
                         (None, {"root": 2, "tones": [2, 5, 9]}, {"a": ["C", "Dm"]}, ["C", "Dm"], [20.0, 24.0, 28.0]))
        # The chord and the bar are one step: taken back, the bar has its chord again and the table no new one.
        self.assertEqual(self.session.undo_label, "new chord Dm on bar 2 of pattern a")
        self.assertTrue(self.session.undo())
        self.assertEqual((self.data(), self.session.can_undo), (CHORDED, False))

    def test_a_new_chord_and_its_bar_taken_back_in_one_leave_the_file_as_it_was_to_the_byte(self) -> None:
        # The file is laid out as a save lays it out, so that a save alone changes none of its bytes.
        for held in (CHORDED, UNCHORDED):
            self.opened(lambda path, held=held: Project.from_dict(held).save(path))
            before = self.path.read_bytes()
            self.choose(1, "New chord...")
            self.ui.modal.name.text = "low"
            self.key("return")
            self.assertEqual((self.ui.modal, self.shown()[1], self.session.undo_label, self.session.message),
                             (None, "low", "new chord low on bar 2 of pattern a", ""))
            self.assertNotEqual(self.path.read_bytes(), before)
            self.assertTrue(self.session.undo())
            self.assertEqual((self.path.read_bytes(), self.session.can_undo), (before, False))
        # A bar the pattern does not have is refused with nothing saved, and the chord is not added alone.
        self.assertFalse(self.panel._add_chord("a", 2, 2, "low", {"root": 0, "tones": [0, 4, 7]}))
        self.assertEqual((self.path.read_bytes(), self.session.can_undo, self.session.message),
                         (before, False, "2 is no bar of the pattern, which has 2, counted from 0"))

    def test_a_song_with_no_chords_has_an_empty_strip_and_nothing_red_and_its_first_chord_is_made_in_the_box(
            self) -> None:
        self.opened(lambda path: path.write_text(json.dumps(UNCHORDED), encoding="utf-8"))
        self.assertEqual((self.shown(), self.wrong(), self.roll.map.chords[3] > 0), (["", ""], [], True))
        if pygame:
            surface = pygame.Surface((RECT[0] + RECT[2], RECT[1] + RECT[3]))
            canvas = Canvas(surface, Font(sprites.load("font")), sprites.load("icons"))
            self.panel.draw(canvas)
            self.ui.draw(canvas)
            x, y, w, h = self.roll.map.chords
            # Each of the two bars has its box, 16 rows less a pixel wide, and there is no name in either.
            self.assertEqual((ink(surface, self.roll.rect, theme.FAULT), ink(surface, (x, y, w, h), theme.ACTIVE),
                              ink(surface, (x, y, w, 1), theme.LINE)), (0, 0, 2 * (16 * self.roll.row_w - 1)))
        # The project has no chord to choose yet. The box makes the first, and the file gets both its tables.
        self.assertEqual(self.ask(0), ["none", "New chord..."])
        self.choose(0, "New chord...")
        self.key("return")
        self.assertEqual((self.ui.modal, self.data()["chords"], self.chords(), self.shown(), self.wrong()),
                         (None, {"C": {"root": 0, "tones": [0, 4, 7]}}, {"a": ["C"]}, ["C", ""], []))
        self.assertEqual(differing(UNCHORDED, self.data()), ["songs.tune.chords", "chords"])


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
        click(self.ui, block(panel, order(panel).index("band")))
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
        self.assertEqual((order(panel), panel.channel, panel.roll.colour), (["a", "b"], "lead", theme.SONG))
        first = panel.roll.notes[0]
        bar = panel.roll.map.note(first.row, first.rows, first.pitch)
        self.assertTrue(lies_in(bar, arrange(rect)["body"]))
        self.assertGreater(ink(self.surface, bar, theme.SONG), bar[2] * bar[3] // 2)
        # The key above it has no note there, and the frame of the view is a line round the roll.
        self.assertEqual(ink(self.surface, (bar[0], bar[1] - bar[3] - 1, bar[2], bar[3]), theme.SONG), 0)
        x, y, w, h = arrange(rect)["view"]
        self.assertEqual(ink(self.surface, (x, y, w, 1), theme.LINE), w)

    def test_a_click_on_the_second_block_of_the_arcade_tune_draws_the_line_of_the_picked_one_under_it(self) -> None:
        panel = SongPanel(Stand(Model(EXAMPLES / "arcade.json")), self.ui)
        panel.place((0, 0, RECT[2], RECT[3]))
        panel.update()
        first, second = ((x, y + h, w, theme.STRIP) for x, y, w, h in map(panel.strip.map.block, range(2)))
        self.assertTrue(lies_in(second, panel.strip.rect))
        for picked, lines in ((0, [first[2] * theme.STRIP, 0]), (1, [0, second[2] * theme.STRIP])):
            if picked:
                click(self.ui, block(panel, picked))
            panel.update()
            panel.draw(self.canvas)
            self.ui.draw(self.canvas)
            self.assertEqual([ink(self.surface, line, theme.ACTIVE) for line in (first, second)], lines)
        self.assertEqual((panel.strip.picked, panel.roll.pattern.name), (1, "b"))

    def test_the_eighteen_names_of_the_piece_are_drawn_in_the_mixer_or_come_into_view_with_the_wheel_none_over_another(
            self) -> None:
        panel = SongPanel(Session(Model(EXAMPLES / "piece.json")), self.ui)
        mixer, views = panel.mixer, {}
        for size in (RECT[2:], SMALL[2:]):
            panel.place((0, 0, *size))
            panel.update()
            if not panel.mixing:
                choose(self.ui, panel, MIXER)
            found = views[size] = []
            first = None
            while mixer.first != first:
                first = mixer.first
                self.surface.fill(theme.BACKGROUND)
                panel.update()
                panel.draw(self.canvas)
                self.ui.draw(self.canvas)
                shown = [strip for strip in mixer.strips if strip.name.visible]
                found.append([strip.channel for strip in shown])
                # Each name in view is drawn in its colour in its own room, and none of it in the room of the next.
                self.assertTrue(all(ink(self.surface, strip.name.rect, strip.name.colour) for strip in shown))
                for one, other in zip(shown, shown[1:]):
                    self.assertLessEqual(one.name.rect[0] + one.name.rect[2], other.name.rect[0])
                    self.assertEqual(ink(self.surface, other.name.rect, one.name.colour), 0)
                self.assertTrue(all(lies_in(strip.name.rect, arrange((0, 0, *size))["desk"]) for strip in shown))
                # The wheel is turned over a meter, which takes nothing from the mouse, until the view stays.
                self.ui.handle(Event("wheel", middle(mixer.meter.rect), wheel=-1))
        names = list(Project.load(EXAMPLES / "piece.json").songs["piece"]["channels"])
        # In the panel the layout is drawn for all eighteen are in view, the ten tuned ones and then the drums.
        self.assertEqual(views[RECT[2:]], [names])
        # In the least panel thirteen are, and three steps of the wheel bring the last into view.
        self.assertEqual([(len(shown), shown[0], shown[-1]) for shown in views[SMALL[2:]]],
                         [(13, "lead", "snare"), (13, "lead_low", "hats"), (13, "stab2", "ohat"),
                          (13, "stab3", "crash")])
        self.assertEqual({name for shown in views[SMALL[2:]] for name in shown}, set(names))


if __name__ == "__main__":
    unittest.main()
