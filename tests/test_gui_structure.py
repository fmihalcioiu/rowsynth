"""Tests for a song changed as a whole in the song panel: its order, its patterns, its channels, its tempo and key."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import unittest
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from rowsynth.gui import layout, popups, songedit, songview, sprites, theme
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.font import Font
from rowsynth.gui.model import Model
from rowsynth.gui.popups import Menu, TextBox
from rowsynth.gui.session import POLL_EVERY, Session
from rowsynth.gui.songpanel import (ELSEWHERE, MIXER, SongPanel, arrange, bars_text, free_name, in_name, shift_text,
                                    song_parts, step_to, tempo_steps, tempo_text)
from rowsynth.gui.strips import EDGE_PX
from rowsynth.gui.ui import Event, Ui
from rowsynth.project import Project
from rowsynth.song import render_loop

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

#: The middle of the art surface the layout is drawn for.
RECT = (124, 22, 662, 504)
#: The rate a project that is changed is rendered at on the threads of its model, and the rate the trailer piece
#: is played at to see that it plays the same: low ones, to be quick.
RATE = 8000
SLOW = 2000
#: The time the first write from outside is stamped with, in nanoseconds: none the file gets by being saved.
STAMP = 1_700_000_000 * 10**9
CTRL = frozenset({"ctrl"})

#: A song of three patterns, of one bar, one bar and two, and of three channels, one of them drums. Its lowest
#: note is the lowest that text can write, so the song cannot be moved down.
TUNE = {
    "row_ticks": 6,
    "drums": {"k": {"inst": "thud", "note": "c2"}, "h": {"inst": "thud", "note": "c6"}},
    "channels": {"lead": {"inst": "tone"}, "bass": {"inst": "buzz"}, "kit": {"inst": "thud", "drums": True}},
    "patterns": {"a": {"lead": "o4 l4 c e g e", "bass": "o0 l2 c g", "kit": "l4 k h k h"},
                 "b": {"lead": "o4 l4 g f e d", "kit": "l8 k k h h k k h h"},
                 "c": {"lead": "o5 l1 c | d", "bass": "o2 l1 c | g"}},
    "order": ["a", "b", "c"],
    "chords": {"a": ["C"], "b": ["C"], "c": ["C", "C"]},
}
PROJECT = {
    "instruments": {"tone": {"wave": "tri"}, "buzz": {"wave": "saw"}, "thud": {"wave": "noise"}},
    "chords": {"C": {"root": 0, "tones": [0, 4, 7]}},
    "songs": {"tune": TUNE},
}

#: What the menu of a block of one bar reads, and the menu of a channel: None is the line between two groups.
BLOCK_MENU = ["Play it again after this", "Copy as a new pattern", "New empty pattern after this", "Rename...",
              "Length (1 bar)...", None, "Remove from the song"]
CHANNEL_MENU = ["Rename...", "Move up", "Move down", "New tuned channel", "New drum channel", None, "Remove"]


class Speaker:
    """A stand-in for the player: it plays nothing, and is where a test says it is."""

    def __init__(self) -> None:
        self.playing = False
        self.at = 0.0

    def play(self, sides: Sequence, loop: bool = False) -> None:
        self.playing = True

    def also(self, sides: Sequence) -> None:
        pass

    def stop(self, fade: float = 0.0) -> None:
        self.playing = False

    def swap(self, sides: Sequence) -> None:
        pass

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
    """The places at which two songs differ, each spelt as a change spells its place. Dicts are gone through by key."""
    if isinstance(one, dict) and isinstance(other, dict):
        return [place for key in dict.fromkeys([*one, *other])
                for place in (differing(one[key], other[key], f"{where}.{key}".lstrip("."))
                              if key in one and key in other else [f"{where}.{key}".lstrip(".")])]
    return [] if json.dumps(one) == json.dumps(other) else [where]


def middle(rect: layout.Rect) -> tuple[int, int]:
    """The pixel in the middle of a rectangle."""
    return rect[0] + rect[2] // 2, rect[1] + rect[3] // 2


def lies_in(rect: layout.Rect, bounds: layout.Rect) -> bool:
    """Whether a rectangle has no pixel outside another."""
    return (bounds[0] <= rect[0] and rect[0] + rect[2] <= bounds[0] + bounds[2]
            and bounds[1] <= rect[1] and rect[1] + rect[3] <= bounds[1] + bounds[3])


class Opened(unittest.TestCase):
    """A project file of the test's own, a real session on it with a stand-in player, and the song panel."""

    name = "tune"

    def opened(self, put: Callable[[Path], object]) -> None:
        """Have `put` make the file in a folder of the test's own, and open it."""
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
        self.panel.strip.clock = lambda: self.now
        self.panel.place(RECT)
        self.panel.update()
        self.strip = self.panel.strip

    def song(self) -> dict:
        """The song as the file holds it now."""
        return json.loads(self.path.read_text(encoding="utf-8"))["songs"][self.name]

    def read(self, pattern: str) -> songview.Pattern:
        """A pattern of the song, read from the file."""
        return songview.pattern(Project.load(self.path).song(self.name), pattern)

    def outside(self, change: Callable[[dict], object]) -> None:
        """Have another program write the file with the song changed, and the session look at it."""
        project = json.loads(self.path.read_text(encoding="utf-8"))
        change(project["songs"][self.name])
        self.path.write_text(json.dumps(project), encoding="utf-8")
        # Stamped as no write before it was, so that the look cannot take the file for the one it knows.
        self.stamp += 10**9
        os.utime(self.path, ns=(self.stamp, self.stamp))
        self.now += POLL_EVERY
        self.session.tick()

    def back(self) -> None:
        """Take the last step back, and have the panel take a frame."""
        self.assertTrue(self.session.undo())
        self.panel.update()

    def order(self) -> list[str]:
        """The names of the blocks of the strip, in the order they stand."""
        return [block.name for block in self.strip.map.blocks]

    def block(self, place: int) -> tuple[int, int]:
        """The pixel in the middle of the block of a place in the order."""
        return middle(self.strip.map.block(place))

    def gap(self, index: int) -> tuple[int, int]:
        """A pixel on the line of a gap of the strip, under the names of the blocks."""
        return self.strip.map.gap_x(index), self.strip.rect[1] + 40

    def row(self, channel: str) -> tuple[int, int]:
        """A pixel on the row of a channel in the list, as the list is scrolled now."""
        x, y, _, h = self.panel.list.rect
        return next((x + 30, down) for down in range(y, y + h)
                    if getattr(self.panel.list.at((x, down)), "key", None) == channel)

    def click(self, pos: tuple[int, int], button: int = 1) -> None:
        """Press a button of the mouse at a position and let it go there."""
        self.ui.handle(Event("down", pos, button=button))
        self.ui.handle(Event("up", pos, button=button))

    def carry(self, place: int, gap: int, mods: frozenset[str] = frozenset()) -> None:
        """Press on the block of a place, take it to a gap with some keys held, and let it go there."""
        self.ui.handle(Event("down", self.block(place), button=1))
        self.ui.handle(Event("move", self.gap(gap), mods=mods))
        self.ui.handle(Event("up", self.gap(gap), button=1, mods=mods))

    def menu(self, pos: tuple[int, int]) -> Menu:
        """Press the right button at a position, and give the menu that this opened."""
        self.click(pos, button=3)
        self.assertIsInstance(self.ui.modal, Menu)
        return self.ui.modal

    def choose(self, pos: tuple[int, int], text: str) -> None:
        """Open the menu at a position with the right button, and click the item whose text starts so."""
        menu = self.menu(pos)
        index = next(index for index, item in enumerate(menu.items) if item is not None and item.text.startswith(text))
        self.click(middle(menu.rows[index]))

    def box(self) -> TextBox:
        """The box that is open to be typed into."""
        self.assertIsInstance(self.ui.modal, TextBox)
        return self.ui.modal

    def type(self, text: str, enter: bool = True) -> None:
        """Type characters, and then press enter unless told not to."""
        self.ui.handle(Event("text", text=text))
        if enter:
            self.key("return")

    def key(self, key: str) -> None:
        """Press a key."""
        self.ui.handle(Event("key", key=key))


class Words(unittest.TestCase):
    def test_the_tempos_are_those_the_engine_can_do_the_slowest_first_with_the_song_s_own_among_them(self) -> None:
        steps = tempo_steps(6)
        self.assertEqual(steps, sorted(songedit.tempos(4), reverse=True))
        self.assertEqual((steps[0], steps[-1], steps.index((6, 150.0))), ((22, 900 / 22), (3, 300.0), 16))
        self.assertEqual([tempo_text(beats) for _, beats in steps[14:18]], ["112.5", "128.6", "150", "180"])
        # A row of two ticks is 450 beats a minute, faster than any that is listed: it stands at the fast end.
        self.assertEqual((tempo_steps(2)[-2:], tempo_steps(2.5)[-1]), ([(3, 300.0), (2, 450.0)], (2.5, 360.0)))
        # With eight rows to the beat a row of six ticks is half as fast, and a row of no ticks has no tempo.
        self.assertEqual((dict(tempo_steps(6, 8))[6], dict(tempo_steps(0))[0]), (75.0, 0.0))

    def test_a_free_name_counts_on_from_the_name_and_a_channel_steps_among_those_of_its_kind(self) -> None:
        self.assertEqual([free_name(taken, name) for taken, name in (
            (("lead",), "lead"), (("lead", "lead2", "lead3"), "lead"), (("pad2",), "pad2"), (("7",), "7"))],
            ["lead2", "lead4", "pad3", "8"])
        # The list has the drum channels apart, so a tuned channel steps over the drum channel between two.
        channels = {"lead": {"inst": "a"}, "kit": {"inst": "b", "drums": True}, "bass": {"inst": "a"},
                    "hats": {"inst": "b", "drums": True}}
        found = {name: [step_to(channels, name, by) for by in (-1, 1)] for name in channels}
        self.assertEqual(found, {"lead": [None, 2], "kit": [None, 3], "bass": [0, None], "hats": [1, None]})
        self.assertEqual([list(songedit.channel_moved({"channels": channels}, name, to)["channels"])
                          for name, to in (("bass", 0), ("lead", 2))],
                         [["bass", "lead", "kit", "hats"], ["kit", "bass", "lead", "hats"]])

    def test_what_is_shown_of_bars_and_of_a_move_and_which_characters_a_name_can_hold(self) -> None:
        self.assertEqual([bars_text(bars) for bars in (1, 4, 0.5, 1.5)], ["1 bar", "4 bars", "0.5 bars", "1.5 bars"])
        self.assertEqual([shift_text(semis) for semis in (0, 2, -3, 12)], ["0", "+2", "-3", "+12"])
        self.assertEqual([char for char in "aZ9 _-+.[]/\\:*?\"<>|\t" if not in_name(char)], list(".[]/\\:*?\"<>|\t"))


class Strip(Opened):
    def setUp(self) -> None:
        self.opened(Project.from_dict(PROJECT).save)

    def test_a_block_dragged_to_a_gap_is_moved_in_the_order_of_the_file_and_is_the_picked_one(self) -> None:
        before = self.song()
        self.carry(0, 3)
        self.assertEqual((self.song()["order"], differing(before, self.song())), (["b", "c", "a"], ["order"]))
        self.assertEqual((self.order(), self.strip.picked, self.panel.roll.pattern.name), (["b", "c", "a"], 2, "a"))
        self.assertEqual((self.session.undo_label, self.session.message), ("tune: pattern a moved", ""))
        # To the front again: the place it is at afterwards is the one that is picked.
        self.carry(2, 1)
        self.assertEqual((self.song()["order"], self.strip.picked, self.panel.roll.pattern.name),
                         (["b", "a", "c"], 1, "a"))
        # A block let go where it stands is not moved, and nothing is saved.
        self.carry(1, 2)
        self.assertEqual((self.song()["order"], self.session.undo_label), (["b", "a", "c"], "tune: pattern a moved"))

    def test_with_ctrl_the_pattern_is_copied_to_the_gap_it_is_let_go_in_beside_its_block_or_away_from_it(self) -> None:
        before = self.song()
        self.carry(1, 2, CTRL)
        made = self.song()
        self.assertEqual((made["order"], list(made["patterns"]), made["patterns"]["b2"], made["chords"]["b2"]),
                         (["a", "b", "b2", "c"], ["a", "b", "c", "b2"], before["patterns"]["b"], ["C"]))
        self.assertEqual((self.strip.picked, self.panel.roll.pattern.name, self.session.undo_label),
                         (2, "b2", "tune: pattern b copied as b2"))
        # Away from its block, at the front and at the end: the copy is where it was let go, and is picked.
        for gap, order in ((0, ["b3", "a", "b", "b2", "c"]), (5, ["b3", "a", "b", "b2", "c", "b4"])):
            self.carry(self.order().index("b"), gap, CTRL)
            self.assertEqual((self.song()["order"], self.strip.picked), (order, gap))
        self.assertEqual((self.panel.roll.pattern.name, self.session.undo_label),
                         ("b4", "tune: pattern b copied as b4"))

    def test_the_view_of_a_strip_that_scrolls_goes_along_under_a_block_that_is_carried_and_stays_when_it_is_moved(
            self) -> None:
        # In a panel this narrow the strip is 48 pixels wide, and the song of 64 rows does not fit it.
        self.panel.place((124, 22, 56, 300))
        x, y, w, _ = self.strip.rect
        # The panel gives the strip its frame: a block carried to the end of the strip has the view move that way.
        self.ui.handle(Event("down", self.block(0), button=1))
        self.ui.handle(Event("move", (x + w - 1, y + 40)))
        for _ in range(2):
            self.panel.update()
        self.assertEqual((self.strip.lifted, self.strip.offset), (0, 2 * EDGE_PX))
        # Escape drops the block, and its release then moves nothing.
        self.key("escape")
        self.ui.handle(Event("up", (x + w - 1, y + 40), button=1))
        self.assertEqual((self.strip.busy, self.session.can_undo), (False, False))
        self.ui.handle(Event("wheel", middle(self.strip.rect), wheel=-1))
        self.assertEqual((self.strip.offset, self.strip.map.in_view()), (16, [1, 2]))
        self.carry(1, 3)
        self.assertEqual((self.song()["order"], self.strip.picked, self.strip.offset), (["a", "c", "b"], 2, 16))

    def test_a_press_on_a_block_is_a_gesture_under_way_and_a_change_from_outside_waits_for_its_release(self) -> None:
        roll = self.panel.roll
        self.ui.handle(Event("down", self.block(0), button=1))
        self.ui.handle(Event("move", self.gap(3)))
        self.outside(lambda song: song["patterns"]["a"].update(lead="o4 l4 c e g > c"))
        # Two seconds in the song plays its second pattern, and FOLLOW is on: the picked place stays all the same.
        self.speaker.playing, self.speaker.at = True, 2.0
        for _ in range(2):
            self.panel.update()
        self.assertEqual((self.strip.busy, self.strip.lifted, self.strip.picked), (True, 0, 0))
        self.assertEqual(([note.pitch for note in roll.notes], self.strip.playhead), ([60, 64, 67, 64], 20.0))
        self.speaker.playing = False
        # The release moves the block in the song as it is by now, so what was changed outside is kept.
        self.ui.handle(Event("up", self.gap(3), button=1))
        self.assertEqual((self.song()["order"], self.song()["patterns"]["a"]["lead"], self.session.message),
                         (["b", "c", "a"], "o4 l4 c e g > c", ""))
        self.assertEqual((self.strip.busy, roll.pattern.name, [note.pitch for note in roll.notes]),
                         (False, "a", [60, 64, 67, 72]))

    def test_the_right_button_on_a_block_opens_its_menu_at_the_mouse_and_picks_nothing(self) -> None:
        before, at = self.path.read_text(encoding="utf-8"), self.block(1)
        menu = self.menu(at)
        self.assertEqual(([item and item.text for item in menu.items], [item.enabled for item in menu.items if item]),
                         (BLOCK_MENU, [True] * 6))
        self.assertEqual((menu.rect[:2], self.strip.picked), (at, 0))
        # The item of the length says how long the pattern is now. Escape closes the menu, and nothing is done.
        self.key("escape")
        self.assertEqual(self.menu(self.block(2)).items[4].text, "Length (2 bars)...")
        self.key("escape")
        self.assertEqual((self.ui.modal, self.path.read_text(encoding="utf-8"), self.session.can_undo),
                         (None, before, False))
        # Past the last block there is no block to do anything to, and no menu.
        x, y, w, _ = self.strip.rect
        self.click((x + w - 1, y + 40), button=3)
        self.assertIsNone(self.ui.modal)

    def test_play_it_again_puts_the_pattern_once_more_after_its_block(self) -> None:
        before = self.song()
        self.choose(self.block(1), "Play it again")
        self.assertEqual((self.song()["order"], differing(before, self.song())), (["a", "b", "b", "c"], ["order"]))
        self.assertEqual((self.strip.picked, self.session.undo_label), (2, "tune: pattern b played again"))
        # The first of the two carried past the second leaves the order as it is: nothing is saved, and the
        # block that was carried is the picked one, at the place it was let go at.
        self.click(self.block(1))
        self.assertEqual(self.strip.picked, 1)
        self.carry(1, 3)
        self.assertEqual((self.song()["order"], self.strip.picked, self.session.undo_label, self.session.message),
                         (["a", "b", "b", "c"], 2, "tune: pattern b played again", ""))

    def test_copy_as_a_new_pattern_makes_one_with_the_same_texts_and_chords_and_plays_it_after_the_block(self) -> None:
        before = self.song()
        self.choose(self.block(2), "Copy")
        made = self.song()
        self.assertEqual(differing(before, made), ["patterns.c2", "order", "chords.c2"])
        self.assertEqual((made["order"], made["patterns"]["c2"], made["chords"]["c2"]),
                         (["a", "b", "c", "c2"], before["patterns"]["c"], ["C", "C"]))
        self.assertEqual((self.strip.picked, self.panel.roll.pattern.name, self.session.undo_label),
                         (3, "c2", "tune: pattern c copied as c2"))

    def test_a_new_empty_pattern_is_as_long_as_the_one_it_comes_after_and_is_the_picked_one(self) -> None:
        before = self.song()
        self.choose(self.block(2), "New empty")
        made = self.song()
        self.assertEqual((differing(before, made), made["order"]), (["patterns.c2", "order"], ["a", "b", "c", "c2"]))
        new = self.read("c2")
        self.assertEqual((new.rows, [held.notes for held in new.channels], list(made["patterns"]["c2"])),
                         (32.0, [(), (), ()], ["lead"]))
        self.assertEqual((self.strip.picked, self.panel.roll.pattern.name, self.session.undo_label),
                         (3, "c2", "tune: new pattern c2 after c"))

    def test_remove_takes_the_block_out_with_its_pattern_and_the_last_block_cannot_be_removed(self) -> None:
        self.click(self.block(1))
        before = self.song()
        self.choose(self.block(0), "Remove")
        made = self.song()
        self.assertEqual((differing(before, made), made["order"]), (["patterns.a", "order", "chords.a"], ["b", "c"]))
        # The block that was picked is picked still, one place further to the front.
        self.assertEqual((self.strip.picked, self.panel.roll.pattern.name, self.session.undo_label),
                         (0, "b", "tune: pattern a taken out of the order"))
        # With the picked block taken out, the one that takes its place is picked.
        self.choose(self.block(0), "Remove")
        self.assertEqual((self.song()["order"], self.strip.picked, self.panel.roll.pattern.name), (["c"], 0, "c"))
        # The last block has its item dim, and a press on it does nothing.
        before = self.path.read_text(encoding="utf-8")
        self.choose(self.block(0), "Remove")
        self.assertEqual((self.ui.modal.items[-1].enabled, self.path.read_text(encoding="utf-8")), (False, before))

    def test_rename_opens_a_box_over_the_name_and_enter_gives_the_pattern_the_new_name_everywhere(self) -> None:
        self.choose(self.block(1), "Play it again")
        before = self.song()
        self.choose(self.block(1), "Rename")
        box = self.box()
        self.assertEqual((box.rect, box.text, box.picked), (popups.over(self.strip.name_rect(1), self.ui.size), "b",
                                                            (0, 1)))
        self.assertEqual(self.ui.help, "the name of pattern b: enter gives it the new one, escape leaves it")
        self.type("mid")
        made = self.song()
        self.assertEqual((made["order"], list(made["patterns"]), list(made["chords"])),
                         (["a", "mid", "mid", "c"], ["a", "mid", "c"], ["a", "mid", "c"]))
        self.assertEqual((made["patterns"]["mid"], made["chords"]["mid"]), (before["patterns"]["b"], ["C"]))
        self.assertEqual((self.ui.modal, self.order(), self.session.undo_label),
                         (None, ["a", "mid", "mid", "c"], "tune: pattern b renamed to mid"))

    def test_a_double_press_on_the_name_of_a_block_opens_the_box_and_the_picked_pattern_is_picked_under_its_new_name(
            self) -> None:
        name = middle(self.strip.name_rect(1))
        self.click(name)
        self.now += 0.2
        self.click(name)
        self.assertEqual((self.box().text, self.strip.picked), ("b", 1))
        self.type("mid")
        self.assertEqual((self.song()["order"], self.order()), (["a", "mid", "c"], ["a", "mid", "c"]))
        self.assertEqual((self.strip.picked, self.panel.roll.pattern.name), (1, "mid"))

    def test_a_name_that_is_refused_leaves_the_box_open_and_is_said_and_only_what_a_name_can_hold_is_typed(
            self) -> None:
        before = self.path.read_text(encoding="utf-8")
        # Enter on the name as it is closes the box, and saves nothing and says nothing.
        self.choose(self.block(1), "Rename")
        self.key("return")
        self.assertEqual((self.ui.modal, self.session.message, self.session.can_undo), (None, "", False))
        self.choose(self.block(1), "Rename")
        self.type("a")
        self.assertEqual((self.box().text, self.session.message), ("a", "the song has a pattern called 'a' already"))
        self.assertEqual((self.path.read_text(encoding="utf-8"), self.session.can_undo), (before, False))
        # A name with nothing in it is none. A dot and a bracket are what a place in the file is spelt with.
        self.key("backspace")
        self.key("return")
        self.assertEqual((self.box().text, self.session.message), ("", "a pattern needs a name, and '' is none"))
        self.type("x.y[2]/", enter=False)
        self.assertEqual(self.box().text, "xy2")
        # Escape leaves the pattern as it was, and the name that was mended is taken.
        self.key("escape")
        self.assertEqual((self.ui.modal, self.path.read_text(encoding="utf-8")), (None, before))
        self.choose(self.block(1), "Rename")
        self.type("a")
        self.type("2")
        self.assertEqual((self.ui.modal, self.song()["order"], self.session.message), (None, ["a", "a2", "c"], ""))

    def test_length_opens_a_box_that_holds_the_bars_and_takes_digits_and_enter_makes_the_pattern_so_long(self) -> None:
        before = self.song()
        self.choose(self.block(2), "Length")
        self.assertEqual((self.box().text, self.ui.help),
                         ("2", "how many bars pattern c is long: enter cuts it or fills it, escape leaves it"))
        self.type("3 x")
        made, longer = self.song(), self.read("c")
        self.assertEqual(differing(before, made), ["patterns.c.lead", "patterns.c.bass", "chords.c"])
        self.assertEqual((longer.rows, [[note.pitch for note in held.notes] for held in longer.channels]),
                         (48.0, [[72, 74], [36, 43], []]))
        self.assertEqual((made["chords"]["c"], self.session.undo_label),
                         (["C", "C", "C"], "tune: pattern c made 3 bars long"))
        # Shorter, the notes past the new end go, and so do the chords written for those bars.
        self.choose(self.block(2), "Length")
        self.assertEqual(self.box().text, "3")
        self.type("1")
        shorter = self.read("c")
        self.assertEqual((shorter.rows, [[note.pitch for note in held.notes] for held in shorter.channels]),
                         (16.0, [[72], [36], []]))
        self.assertEqual((self.song()["chords"]["c"], self.session.undo_label),
                         (["C"], "tune: pattern c made 1 bar long"))
        # No bars are refused: the box stays open, and the pattern is as it was.
        self.choose(self.block(2), "Length")
        self.type("0")
        self.assertEqual((self.box().text, self.session.message, self.read("c").rows),
                         ("0", "the bars of a pattern are a whole number, 1 or more, not 0", 16.0))

    def test_what_a_menu_was_opened_on_is_not_done_to_a_song_whose_order_or_channels_were_changed_outside_meanwhile(
            self) -> None:
        menu = self.menu(self.block(1))
        self.outside(lambda song: song.update(order=["b", "a", "c"]))
        self.panel.update()
        # While the menu is open nothing is read: the strip shows what the menu was opened on.
        self.assertEqual((self.order(), self.session.message), (["a", "b", "c"], "changed outside: tune"))
        self.click(middle(menu.rows[-1]))
        self.assertEqual((self.song()["order"], self.order(), self.session.message),
                         (["b", "a", "c"], ["b", "a", "c"], f"tune {ELSEWHERE}"))
        self.assertEqual(self.session.undo_label, "changed outside: tune")
        # So it is with the menu of a channel, where another channel was put in front of it meanwhile.
        menu = self.menu(self.row("bass"))
        self.outside(lambda song: song.update(channels={"kit": song["channels"].pop("kit"), **song["channels"]}))
        self.panel.update()
        self.click(middle(menu.rows[-1]))
        self.assertEqual((list(self.song()["channels"]), self.session.message, self.session.undo_label),
                         (["kit", "lead", "bass"], f"tune {ELSEWHERE}", "changed outside: tune"))
        # The song was read again, and the drum channel has the colour of the first place in the list.
        self.assertEqual(next(row.square for row in self.panel.list.rows if row.key == "kit"), theme.CHANNELS[0])


class Channels(Opened):
    def setUp(self) -> None:
        self.opened(Project.from_dict(PROJECT).save)

    def test_the_right_button_on_a_channel_opens_its_menu_and_a_channel_does_not_move_past_the_ends_of_its_kind(
            self) -> None:
        at = self.row("lead")
        menu = self.menu(at)
        self.assertEqual(([item and item.text for item in menu.items], menu.rect[:2]), (CHANNEL_MENU, at))
        enabled = {}
        for channel in ("lead", "bass", "kit"):
            self.key("escape")
            enabled[channel] = [item.enabled for item in self.menu(self.row(channel)).items[1:3]]
        # The first tuned channel does not go up, the last not down, and the one drum channel neither way.
        self.assertEqual(enabled, {"lead": [False, True], "bass": [True, False], "kit": [False, False]})
        # A press on an item that is dim does nothing, and the right button picked no channel.
        self.click(middle(self.ui.modal.rows[1]))
        self.assertEqual((self.ui.modal.items[1].text, self.panel.channel, self.session.can_undo),
                         ("Move up", "lead", False))

    def test_move_up_and_move_down_change_the_place_of_a_channel_in_the_song_and_in_the_list(self) -> None:
        before = self.song()
        self.choose(self.row("bass"), "Move up")
        # Nothing that the song holds is another: it is the order of its channels alone.
        self.assertEqual((list(self.song()["channels"]), differing(before, self.song())),
                         (["bass", "lead", "kit"], []))
        self.assertEqual(([row.key for row in self.panel.list.rows if not row.head], self.panel.channel),
                         (["bass", "lead", "kit"], "lead"))
        self.assertEqual(self.session.undo_label, "tune: channel bass moved up")
        self.choose(self.row("bass"), "Move down")
        self.assertEqual((self.song(), self.session.undo_label), (before, "tune: channel bass moved down"))

    def test_a_new_channel_plays_the_instrument_of_the_one_it_was_asked_on_has_a_free_name_and_is_picked(self) -> None:
        panel = self.panel
        self.choose(self.row("bass"), "New tuned")
        self.assertEqual((self.song()["channels"]["bass2"], list(self.song()["channels"])),
                         ({"inst": "buzz"}, ["lead", "bass", "kit", "bass2"]))
        self.assertEqual((panel.channel, panel.list.picked, panel.roll.channel, panel.roll.notes),
                         ("bass2", "bass2", "bass2", ()))
        self.assertEqual(self.session.undo_label, "tune: new tuned channel bass2")
        # The next one asked on the same channel has the next free name.
        self.choose(self.row("bass"), "New tuned")
        self.assertEqual((list(self.song()["channels"])[-1], panel.channel), ("bass3", "bass3"))
        self.choose(self.row("kit"), "New drum")
        self.assertEqual((self.song()["channels"]["kit2"], panel.channel, panel.steps.visible, panel.steps.picked),
                         ({"inst": "thud", "drums": True}, "kit2", True, "kit2"))
        self.assertEqual(self.session.undo_label, "tune: new drum channel kit2")

    def test_rename_opens_a_box_over_the_row_and_the_channel_has_its_new_name_everywhere_and_stays_picked(
            self) -> None:
        panel, before = self.panel, self.song()
        self.choose(self.row("lead"), "Rename")
        box = self.box()
        self.assertEqual((box.rect, box.text), (popups.over(panel.list.name_rect("lead"), self.ui.size), "lead"))
        self.assertEqual(self.ui.help, "the name of channel lead: enter gives it the new one, escape leaves it")
        self.type("top")
        made = self.song()
        self.assertEqual((list(made["channels"]), [list(texts) for texts in made["patterns"].values()]),
                         (["top", "bass", "kit"], [["top", "bass", "kit"], ["top", "kit"], ["top", "bass"]]))
        self.assertEqual((made["patterns"]["a"]["top"], made["channels"]["top"]),
                         (before["patterns"]["a"]["lead"], {"inst": "tone"}))
        self.assertEqual((panel.channel, panel.list.picked, panel.roll.channel, self.session.undo_label),
                         ("top", "top", "top", "tune: channel lead renamed to top"))
        # Another channel than the picked one is named anew, and the picked one stays. A name that is taken is
        # refused: the box stays open, and the reason is said.
        self.choose(self.row("bass"), "Rename")
        self.type("kit")
        self.assertEqual((self.box().text, self.session.message),
                         ("kit", "the song has a channel called 'kit' already"))
        # What is typed next is added to the name that was refused.
        self.type("2")
        self.assertEqual((self.ui.modal, list(self.song()["channels"]), panel.channel),
                         (None, ["top", "kit2", "kit"], "top"))

    def test_remove_takes_a_channel_out_of_the_song_and_one_a_pattern_cannot_do_without_is_refused_and_said(
            self) -> None:
        self.click(self.row("bass"))
        self.choose(self.row("bass"), "Remove")
        made = self.song()
        self.assertEqual((list(made["channels"]), [list(texts) for texts in made["patterns"].values()]),
                         (["lead", "kit"], [["lead", "kit"], ["lead", "kit"], ["lead"]]))
        # The picked channel is gone, so the first one with notes is picked.
        self.assertEqual((self.panel.channel, self.session.undo_label), ("lead", "tune: channel bass removed"))
        before = self.path.read_text(encoding="utf-8")
        self.choose(self.row("lead"), "Remove")
        self.assertEqual(self.session.message,
                         "channel 'lead' has the only text of pattern 'c', which would be left with no rows")
        self.assertEqual((self.path.read_text(encoding="utf-8"), self.session.undo_label),
                         (before, "tune: channel bass removed"))


class Settings(Opened):
    def setUp(self) -> None:
        self.opened(Project.from_dict(PROJECT).save)

    def step(self, way: int) -> None:
        """Click the half of the tempo's stepper that goes one on, or with -1 the one that goes one back."""
        x, y, w, _ = self.panel.tempo.rect
        self.click((x + w - 3 if way > 0 else x + 3, y + 3))

    def shown(self) -> tuple[str, str]:
        """The tempo the stepper shows, and the count between the two buttons of the key."""
        return self.panel.tempo.options[self.panel.tempo.picked], self.panel.shifted.text

    def test_the_tempo_is_shown_in_beats_a_minute_and_a_step_saves_the_ticks_of_a_row(self) -> None:
        before = self.song()
        self.assertEqual((self.shown(), len(self.panel.tempo.options)), (("150", "0"), 20))
        self.step(1)
        self.assertEqual((self.song()["row_ticks"], differing(before, self.song())), (5, ["row_ticks"]))
        self.assertEqual((self.shown()[0], self.session.undo_label), ("180", "tune: tempo 180"))
        for _ in range(2):
            self.step(-1)
        self.assertEqual((self.song()["row_ticks"], self.shown()[0], self.session.undo_label),
                         (7, "128.6", "tune: tempo 128.6"))
        # Each step is one to take back, and the stepper shows what the file holds then.
        self.back()
        self.assertEqual((self.song()["row_ticks"], self.shown()[0]), (6, "150"))
        # A tempo that came from outside and is none of the list is shown at its place among them.
        self.outside(lambda song: song.update(row_ticks=2))
        self.panel.update()
        self.assertEqual((self.shown()[0], self.panel.tempo.picked, len(self.panel.tempo.options)), ("450", 20, 21))

    def test_the_buttons_of_the_key_move_every_tuned_note_a_semitone_and_the_moves_are_counted(self) -> None:
        before = self.song()
        for _ in range(2):
            self.click(middle(self.panel.higher.rect))
        made = self.song()
        self.assertEqual((made, made["patterns"]["a"]["kit"]),
                         (songedit.transposed(before, 2), before["patterns"]["a"]["kit"]))
        self.assertEqual([[note.pitch for note in held.notes] for held in self.read("a").channels[:2]],
                         [[62, 66, 69, 66], [14, 21]])
        self.assertEqual((self.shown()[1], self.session.undo_label), ("+2", "tune: a semitone up"))
        self.click(middle(self.panel.lower.rect))
        self.assertEqual((self.shown()[1], self.session.undo_label), ("+1", "tune: a semitone down"))
        # A move that is taken back is counted back, and one that is made again is counted again.
        self.back()
        self.back()
        self.assertEqual(self.shown()[1], "+1")
        self.assertTrue(self.session.redo())
        self.panel.update()
        self.assertEqual(self.shown()[1], "+2")
        # Another change leaves the count as it is.
        self.step(1)
        self.assertEqual(self.shown(), ("180", "+2"))

    def test_a_move_that_takes_a_note_out_of_what_text_can_write_is_said_and_saves_nothing(self) -> None:
        before = self.path.read_text(encoding="utf-8")
        self.click(middle(self.panel.lower.rect))
        self.assertEqual(self.session.message, "pattern 'a', channel 'bass': -1 semitones up takes a note out of "
                                               "octaves 0 to 9, and the text can write no other")
        self.assertEqual((self.path.read_text(encoding="utf-8"), self.session.can_undo, self.shown()[1]),
                         (before, False, "0"))

    def test_with_the_mixer_shown_the_settings_of_the_song_are_beside_the_choice_and_work_there(self) -> None:
        panel = self.panel
        panel.choice.pick(panel.choice.options.index(MIXER))
        parts = song_parts(arrange(RECT)["beside"])
        self.assertEqual((panel.tempo.rect, panel.lower.rect, panel.tempo.visible, panel.list.visible),
                         (parts["tempo"], parts["lower"], True, False))
        self.step(-1)
        self.click(middle(panel.higher.rect))
        self.assertEqual((self.song()["row_ticks"], self.shown(), panel.mixing), (7, ("128.6", "+1"), True))

    def test_undo_after_each_kind_of_change_puts_the_file_back(self) -> None:
        kinds: dict[str, Callable[[], object]] = {
            "moved": lambda: self.carry(0, 3),
            "copied": lambda: self.carry(0, 3, CTRL),
            "played again": lambda: self.choose(self.block(1), "Play it again"),
            "copied as a new pattern": lambda: self.choose(self.block(1), "Copy"),
            "new pattern": lambda: self.choose(self.block(1), "New empty"),
            "renamed": lambda: (self.choose(self.block(1), "Rename"), self.type("mid")),
            "made longer": lambda: (self.choose(self.block(1), "Length"), self.type("4")),
            "taken out": lambda: self.choose(self.block(1), "Remove"),
            "channel renamed": lambda: (self.choose(self.row("lead"), "Rename"), self.type("top")),
            "channel moved": lambda: self.choose(self.row("lead"), "Move down"),
            "new channel": lambda: self.choose(self.row("lead"), "New tuned"),
            "new drum channel": lambda: self.choose(self.row("lead"), "New drum"),
            "channel removed": lambda: self.choose(self.row("bass"), "Remove"),
            "tempo": lambda: self.step(-1),
            "key": lambda: self.click(middle(self.panel.higher.rect)),
        }
        before = self.path.read_text(encoding="utf-8")
        for kind, change in kinds.items():
            with self.subTest(kind=kind):
                change()
                self.assertNotEqual(self.path.read_text(encoding="utf-8"), before)
                self.assertEqual((self.ui.modal, self.session.message), (None, ""))
                self.back()
                self.assertEqual((self.path.read_text(encoding="utf-8"), self.session.can_undo), (before, False))
                self.assertEqual((self.order(), self.panel.shifted.text, self.shown()[0]),
                                 (["a", "b", "c"], "0", "150"))


class Piece(Opened):
    name = "piece"

    def setUp(self) -> None:
        self.opened(lambda path: shutil.copyfile(EXAMPLES / "piece.json", path))

    def played(self) -> dict:
        """The loop of the song as the file has it."""
        return render_loop(Project.load(self.path).song(self.name), SLOW)

    def test_a_block_of_the_piece_moved_and_moved_back_leaves_the_file_as_it_was_and_the_song_plays_the_same(
            self) -> None:
        before, base, last = self.path.read_text(encoding="utf-8"), self.played(), len(self.order()) - 1
        self.carry(0, last + 1)
        self.assertEqual((self.song()["order"][-1], self.order()[-1], self.strip.picked), ("cold", "cold", last))
        self.assertEqual(differing(json.loads(before)["songs"]["piece"], self.song()), ["order"])
        self.carry(last, 0)
        self.assertEqual((self.path.read_text(encoding="utf-8"), self.order()[0], self.strip.picked),
                         (before, "cold", 0))
        again = self.played()
        self.assertTrue(all(np.array_equal(again[side], base[side]) for side in ("left", "right")))
        self.assertGreater(len(base["left"]), 0)


@unittest.skipUnless(pygame, NEEDS)
class Drawn(Opened):
    def setUp(self) -> None:
        self.opened(Project.from_dict(PROJECT).save)
        self.surface = pygame.Surface(theme.DESIGN)
        self.canvas = Canvas(self.surface, Font(sprites.load("font")), sprites.load("icons"))

    def ink(self, rect: layout.Rect, colour: theme.Colour) -> int:
        """Draw a frame, and count the pixels of a rectangle that are in a colour."""
        self.surface.fill(theme.BACKGROUND)
        self.panel.draw(self.canvas)
        self.ui.draw(self.canvas)
        x, y, w, h = rect
        return sum(self.surface.get_at((x + column, y + row))[:3] == colour for column in range(w) for row in range(h))

    def test_the_settings_of_the_song_the_menu_of_a_block_and_the_box_over_a_name_are_drawn(self) -> None:
        parts = song_parts(arrange(RECT)["song"])
        # The two words are dim, and the count between the two buttons is one figure in the colour of text.
        self.assertEqual([self.ink(parts[part], theme.DIM) > 0 for part in ("tempo_word", "key_word", "shifted")],
                         [True, True, False])
        self.assertGreater(self.ink(parts["shifted"], theme.TEXT), 0)
        # The menu is drawn over the strip and the bar, its outline whole.
        x, y, w, h = self.menu(self.block(1)).rect
        self.assertEqual((self.ink((x, y, w, 1), theme.LINE), self.ink((x, y, 1, h), theme.LINE)), (w, h))
        # The box is drawn over the row of the channel, in the outline of what is being changed.
        self.key("escape")
        self.choose(self.row("bass"), "Rename")
        x, y, w, h = self.box().rect
        self.assertEqual((self.ink((x, y, w, 1), theme.ACTIVE), self.ink((x, y, 1, h), theme.ACTIVE)), (w, h))
        self.assertTrue(lies_in((x, y, w, h), arrange(RECT)["channels"]))


if __name__ == "__main__":
    unittest.main()
