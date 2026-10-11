"""Tests for rowsynth.gui.channelbar: the sums of the bar, what it shows of a channel, and what it saves."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from collections.abc import Callable, Sequence
from pathlib import Path

from rowsynth.gui import layout, sprites, theme
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.channelbar import ChannelBar, arrange, instruments, name_parts, pan_text, tidy
from rowsynth.gui.font import Font
from rowsynth.gui.model import TEST_NOTE, Model
from rowsynth.gui.panels import Panel
from rowsynth.gui.session import POLL_EVERY, Session
from rowsynth.gui.ui import Event, Ui, Widget
from rowsynth.project import Project

# pygame reads these when it is imported: no test opens a window or plays a sound.
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
try:
    import pygame
except ImportError:
    pygame = None

NEEDS = "pygame is not installed"

#: The bar as a song panel in the middle of the art surface has it, and as the least song panel has it.
BAR = (263, 107, 516, 37)
NARROW = (263, 107, 354, 37)

#: A song with two instruments of its own, one of them a name it gives to an instrument of the project. Its first
#: channel says every setting, its second none, and its third is a drum channel.
WALK = {
    "row_ticks": 6,
    "instruments": {"own": {"wave": "pulse"}, "buzz": "tone"},
    "drums": {"k": {"inst": "thud", "note": "c2"}},
    "channels": {"lead": {"inst": "tone", "gain": 0.8, "pan": -0.3, "echo": {"rows": 2, "gain": 0.5}},
                 "low": {"inst": "buzz"},
                 "kit": {"inst": "thud", "gain": 0.6, "drums": True}},
    "patterns": {"a": {"lead": "o4 l4 c e g e", "low": "o2 l2 c g", "kit": "l4 k k k k"}},
    "order": ["a"],
}
PLAIN = {"row_ticks": 6, "channels": {"air": {"inst": "tone"}}, "patterns": {"a": {"air": "o4 l1 c"}}, "order": ["a"]}
PROJECT = {
    "instruments": {"tone": {"wave": "tri"}, "thud": {"wave": "noise"}, "buzz": {"wave": "saw"}},
    "songs": {"walk": WALK, "plain": PLAIN},
}
#: The echo the first channel is written with.
ECHO = {"rows": 2, "gain": 0.5}
#: The rate the project is rendered at on the threads of its model: a low one, to be quick.
RATE = 8000
#: The time the first write from outside is stamped with, in nanoseconds: none the file gets by being saved.
STAMP = 1_700_000_000 * 10**9

#: The widgets of a bar, by what the bar calls them.
WIDGETS = ("name", "word", "heading", "inst", "level", "pan", "echo", "after", "loud", "follow", "ghosts")
#: The parts of a bar that give way when it is narrow, the first first.
GIVING = ("after", "loud", "pan", "level", "echo", "inst", "name")


class Speaker:
    """A stand-in for the player: it keeps what it was asked, in order, and plays nothing."""

    def __init__(self) -> None:
        self.asked: list[tuple] = []
        self.playing = False

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
        return 0.0


def settle() -> None:
    """Wait for every render and every run of the checks that is still on a thread of a model's."""
    for thread in threading.enumerate():
        if thread.name.startswith("rowsynth-"):
            thread.join()


def differing(one: object, other: object, where: str = "") -> list[str]:
    """The places at which the data of two files differ, each spelt as a change spells its place."""
    if isinstance(one, dict) and isinstance(other, dict):
        return [place for key in dict.fromkeys([*one, *other])
                for place in (differing(one[key], other[key], f"{where}.{key}".lstrip("."))
                              if key in one and key in other else [f"{where}.{key}".lstrip(".")])]
    return [] if json.dumps(one) == json.dumps(other) else [where]


def middle(rect: layout.Rect) -> tuple[int, int]:
    """The pixel in the middle of a rectangle."""
    return rect[0] + rect[2] // 2, rect[1] + rect[3] // 2


def click(ui: Ui, pos: tuple[int, int], button: int = 1) -> None:
    """Press a mouse button at a position and let it go there."""
    ui.handle(Event("down", pos, button=button))
    ui.handle(Event("up", pos, button=button))


def lies_in(rect: layout.Rect, bounds: layout.Rect) -> bool:
    """Whether a rectangle has no pixel outside another, and neither a width nor a height below 0."""
    x, y, w, h = rect
    return (w >= 0 and h >= 0 and bounds[0] <= x and x + w <= bounds[0] + bounds[2]
            and bounds[1] <= y and y + h <= bounds[1] + bounds[3])


class Sums(unittest.TestCase):
    def test_a_pan_is_told_as_its_side_and_how_far_and_a_number_is_written_to_two_places(self) -> None:
        self.assertEqual([pan_text(pan) for pan in (-1, -0.3, 0, 0.004, -0.004, 0.05, 0.1 + 0.2, 1)],
                         ["L 100", "L 30", "C", "C", "C", "R 5", "R 30", "R 100"])
        written = [tidy(value) for value in (0.1 + 0.2, 0.8, 1.0, 5.0, -0.25, 0)]
        self.assertEqual(written, [0.3, 0.8, 1, 5, -0.25, 0])
        # A whole number is written without its point.
        self.assertEqual([type(value) for value in written], [float, float, int, int, float, int])

    def test_a_song_can_use_its_own_instruments_which_come_first_and_then_those_of_the_project(self) -> None:
        project = Project.from_dict(PROJECT)
        # The name the song gives to an instrument of the project is there once, among its own.
        self.assertEqual(instruments(project, "walk"), ["own", "buzz", "tone", "thud"])
        self.assertEqual([instruments(project, song) for song in ("plain", None)], [["tone", "thud", "buzz"]] * 2)

    def test_the_parts_stand_from_the_left_with_the_two_toggles_at_the_right_end(self) -> None:
        self.assertEqual(arrange(BAR), {
            "name": (263, 119, 89, 13), "heading": (358, 107, 78, 11), "inst": (358, 119, 78, 13),
            "level": (442, 107, 33, 37), "pan": (481, 107, 33, 37), "echo": (520, 119, 31, 13),
            "after": (557, 107, 33, 37), "loud": (596, 107, 33, 37),
            "follow": (691, 119, 43, 13), "ghosts": (736, 119, 43, 13)})

    def test_a_narrow_bar_drops_the_knobs_of_the_echo_then_the_pan_then_the_level_and_no_two_parts_overlap(
            self) -> None:
        dropped = {}
        for wide in (460, 459, 382, 381, NARROW[2], 342, 303, 266, 182, 40, 0):
            bar = (263, 107, wide, 37)
            parts = arrange(bar)
            dropped[wide] = sum(parts[part][2] == 0 for part in GIVING)
            # What is dropped is dropped in the one order, and what is left is inside the bar, side by side.
            self.assertEqual([part for part in GIVING if parts[part][2] == 0], list(GIVING[:dropped[wide]]), wide)
            self.assertTrue(all(lies_in(part, bar) for part in parts.values()), wide)
            spans = sorted((x, x + w) for name, (x, _, w, _) in parts.items() if w and name != "heading")
            self.assertTrue(all(end <= start for (_, end), (start, _) in zip(spans, spans[1:])), wide)
        self.assertEqual(dropped, {460: 0, 459: 2, 382: 2, 381: 3, 354: 3, 342: 4, 303: 5, 266: 6, 182: 7, 40: 7,
                                   0: 7})
        # In the least song panel the level is there, and the toggles are at the right end as in any.
        self.assertEqual([arrange(NARROW)[part] for part in ("level", "echo", "follow", "ghosts")],
                         [(442, 107, 33, 37), (481, 119, 31, 13), (529, 119, 43, 13), (574, 119, 43, 13)])

    def test_the_word_of_a_drum_channel_stands_after_its_name_and_a_long_name_is_cut_to_leave_it_whole(self) -> None:
        room = (263, 119, 89, 13)
        self.assertEqual([name_parts(room, "lead", False), name_parts(room, "kit", True)],
                         [((263, 119, 23, 13), (286, 119, 0, 13)), ((263, 119, 17, 13), (286, 119, 29, 13))])
        long = "a_long_name_for_a_channel"
        self.assertEqual([name_parts(room, long, False), name_parts(room, long, True)],
                         [((263, 119, 89, 13), (352, 119, 0, 13)), ((263, 119, 54, 13), (323, 119, 29, 13))])
        # A bar that has dropped the name has no room for either.
        self.assertEqual(name_parts((263, 107, 0, 0), "kit", True), ((263, 107, 0, 0), (263, 107, 0, 0)))


class Opened(unittest.TestCase):
    """A project file of the test's own, a real session on it with a stand-in player, and the bar on a panel."""

    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        # No thread of the model is left to run into the next test, or into a folder that is gone.
        self.addCleanup(settle)
        self.path = Path(folder.name) / "project.json"
        self.path.write_text(json.dumps(PROJECT), encoding="utf-8")
        self.now, self.stamp = 100.0, STAMP
        self.speaker = Speaker()
        self.session = Session(Model(self.path, RATE), self.speaker, lambda: self.now)
        self.ui = Ui()
        self.bar = ChannelBar(Panel(self.session, self.ui), lambda: self.now)
        self.bar.place(BAR)
        self.bar.seen(True)
        self.bar.show("walk", "lead", theme.SONG)

    def data(self) -> dict:
        """What the file holds now."""
        return json.loads(self.path.read_text(encoding="utf-8"))

    def channel(self, name: str = "lead") -> dict:
        """What the file holds now for a channel of the song."""
        return self.data()["songs"]["walk"]["channels"][name]

    def outside(self, change: Callable[[dict], object], look: bool = True) -> dict:
        """Have another program change the project and write the file, and the session look at it or not."""
        project = self.data()
        change(project["songs"]["walk"])
        self.path.write_text(json.dumps(project), encoding="utf-8")
        # Stamped as no write before it was, so that the look cannot take the file for the one it knows.
        self.stamp += 10**9
        os.utime(self.path, ns=(self.stamp, self.stamp))
        if look:
            self.now += POLL_EVERY
            self.session.tick()
        return project

    def seen(self) -> list[str]:
        """Which of the bar's widgets are seen, by what the bar calls them."""
        return [name for name in WIDGETS if getattr(self.bar, name).visible]

    def drag(self, widget: Widget, up: int, let_go: bool = True) -> tuple[int, int]:
        """Press the left button in the middle of a widget and move so many pixels up. Where the mouse then is."""
        x, y = middle(widget.rect)
        self.ui.handle(Event("down", (x, y), button=1))
        self.ui.handle(Event("move", (x, y - up)))
        if let_go:
            self.ui.handle(Event("up", (x, y - up), button=1))
        return x, y - up

    def heard(self) -> list[Sequence]:
        """The sides of each note the player was asked to sound over what plays, in order."""
        return [asked[1] for asked in self.speaker.asked if asked[0] == "also"]

    def tone(self, inst: str) -> Sequence:
        """The sides of the test note of an instrument as the model keeps it for the song."""
        return self.session.model.tone(inst, TEST_NOTE, song="walk")["sides"]


class Shown(Opened):
    def test_a_tuned_channel_is_shown_with_its_name_in_its_colour_and_each_setting_as_the_file_has_it(self) -> None:
        bar = self.bar
        self.assertEqual(self.seen(), ["name", "heading", "inst", "level", "pan", "echo", "after", "loud", "follow",
                                       "ghosts"])
        self.assertEqual((bar.name.text, bar.name.colour, bar.name.rect), ("lead", theme.SONG, (263, 119, 23, 13)))
        self.assertEqual((bar.heading.text, bar.inst.options, bar.inst.picked),
                         ("INSTRUMENT", ("own", "buzz", "tone", "thud"), 2))
        knobs = (bar.level, bar.pan, bar.after, bar.loud)
        self.assertEqual([(knob.text, knob.value, knob.low, knob.high, knob.default) for knob in knobs],
                         [("LEVEL", 0.8, 0, 2, 1), ("PAN", -0.3, -1, 1, 0), ("AFTER", 2, 1, 16, 3),
                          ("LOUD", 0.5, 0, 1, 0.3)])
        # A value is shown plainly, and the pan as its side and how far. The knobs are lit in the channel's colour.
        self.assertEqual([knob.show(knob.value) for knob in knobs], ["0.8", "L 30", "2", "0.5"])
        self.assertEqual(({knob.colour for knob in knobs}, bar.echo.text, bar.echo.is_on), ({theme.SONG}, "ECHO", True))

    def test_a_drum_channel_has_the_word_drums_after_its_name_and_no_ghosts_and_what_it_lacks_is_as_the_engine_takes_it(
            self) -> None:
        bar = self.bar
        bar.show("walk", "kit", theme.SALMON)
        self.assertEqual(self.seen(), ["name", "word", "heading", "inst", "level", "pan", "echo", "follow"])
        self.assertEqual((bar.name.text, bar.name.colour, bar.word.text, bar.word.colour, bar.word.rect),
                         ("kit", theme.SALMON, "DRUMS", theme.DIM, (286, 119, 29, 13)))
        self.assertEqual((bar.inst.options[bar.inst.picked], bar.level.value, bar.pan.value, bar.echo.is_on),
                         ("thud", 0.6, 0, False))
        self.assertEqual(bar.pan.show(bar.pan.value), "C")
        # With no channel only FOLLOW is left, and off the window nothing is.
        bar.show("walk", None)
        self.assertEqual(self.seen(), ["follow"])
        bar.show("walk", "kit", theme.SALMON)
        bar.seen(False)
        self.assertEqual(self.seen(), [])
        # In the least song panel the pan has no room, and is not seen.
        bar.seen(True)
        bar.place(NARROW)
        self.assertEqual(self.seen(), ["name", "word", "heading", "inst", "level", "echo", "follow"])


class Saved(Opened):
    def test_a_knob_that_is_dragged_changes_nothing_in_the_file_until_it_is_let_go_and_then_its_one_place(
            self) -> None:
        bar, before = self.bar, self.data()
        mouse = self.drag(bar.level, 5, let_go=False)
        # The knob shows where the hand has it.
        self.assertEqual((bar.level.show(bar.level.value), bar.busy), ("0.9", True))
        self.assertEqual((self.data(), self.session.can_undo), (before, False))
        self.ui.handle(Event("up", mouse, button=1))
        self.assertEqual(differing(before, self.data()), ["songs.walk.channels.lead.gain"])
        self.assertEqual((self.channel()["gain"], bar.level.value, bar.busy), (0.9, 0.9, False))
        # It is one step, called by the channel and by what the knob shows, and taken back the knob shows that.
        self.assertEqual(self.session.undo_label, "lead: level 0.9")
        self.assertTrue(self.session.undo())
        bar.update()
        self.assertEqual((self.data(), bar.level.value, self.session.can_undo), (before, 0.8, False))

    def test_a_level_or_a_pan_put_back_to_what_the_engine_takes_without_it_is_taken_out_of_the_file(self) -> None:
        bar = self.bar
        click(self.ui, middle(bar.level.rect), button=3)
        self.assertEqual((self.channel(), self.session.undo_label, bar.level.value),
                         ({"inst": "tone", "pan": -0.3, "echo": ECHO}, "lead: level 1", 1))
        self.drag(bar.pan, 15)
        self.assertEqual((self.channel(), self.session.undo_label, bar.pan.value),
                         ({"inst": "tone", "echo": ECHO}, "lead: pan C", 0))
        # Taken away from there it is written again.
        self.drag(bar.pan, -20)
        self.assertEqual((self.channel(), self.session.undo_label),
                         ({"inst": "tone", "pan": -0.4, "echo": ECHO}, "lead: pan L 40"))

    def test_the_instrument_is_saved_as_the_stepper_is_stepped_and_one_note_of_the_new_one_is_heard(self) -> None:
        bar, before = self.bar, self.data()
        x, y, w, _ = bar.inst.rect
        click(self.ui, (x + w - 3, y + 3))
        self.assertEqual(differing(before, self.data()), ["songs.walk.channels.lead.inst"])
        self.assertEqual((self.channel()["inst"], bar.inst.picked, self.session.undo_label),
                         ("thud", 3, "lead: instrument thud"))
        self.assertEqual(len(self.heard()), 1)
        self.assertIs(self.heard()[0], self.tone("thud"))
        # Two steps back with the wheel are one change, to the name the song has for an instrument of the project.
        self.ui.handle(Event("wheel", (x + 3, y + 3), wheel=-2))
        self.assertEqual((self.channel()["inst"], bar.inst.picked, self.session.undo_label),
                         ("buzz", 1, "lead: instrument buzz"))
        self.assertEqual(len(self.heard()), 2)
        self.assertIs(self.heard()[1], self.tone("buzz"))

    def test_the_echo_is_switched_on_as_the_usual_one_and_off_and_on_again_it_is_the_echo_it_was(self) -> None:
        bar = self.bar
        bar.show("walk", "low", theme.SOUND)
        self.assertEqual((bar.echo.is_on, bar.after.visible, bar.loud.visible), (False, False, False))
        click(self.ui, middle(bar.echo.rect))
        self.assertEqual((self.channel("low"), self.session.undo_label),
                         ({"inst": "buzz", "echo": {"rows": 3, "gain": 0.3}}, "low: echo on"))
        self.assertEqual((bar.echo.is_on, bar.after.visible, bar.after.value, bar.loud.visible, bar.loud.value),
                         (True, True, 3, True, 0.3))
        # Its two knobs are saved as any knob is, each at its own place in the echo.
        self.drag(bar.after, 14)
        self.assertEqual((self.channel("low")["echo"], self.session.undo_label),
                         ({"rows": 5, "gain": 0.3}, "low: echo after 5"))
        self.drag(bar.loud, 20)
        self.assertEqual((self.channel("low")["echo"], self.session.undo_label),
                         ({"rows": 5, "gain": 0.5}, "low: echo loud 0.5"))
        click(self.ui, middle(bar.echo.rect))
        self.assertEqual((self.channel("low"), self.session.undo_label, bar.echo.is_on, bar.after.visible),
                         ({"inst": "buzz"}, "low: echo off", False, False))
        click(self.ui, middle(bar.echo.rect))
        self.assertEqual((self.channel("low")["echo"], bar.after.value, bar.loud.value),
                         ({"rows": 5, "gain": 0.5}, 5, 0.5))
        # What is kept is that channel's: the echo of another one is the usual one.
        bar.show("walk", "kit", theme.SALMON)
        click(self.ui, middle(bar.echo.rect))
        self.assertEqual(self.channel("kit")["echo"], {"rows": 3, "gain": 0.3})

    def test_an_echo_that_went_from_the_file_outside_is_not_there_to_take_out_and_nothing_is_refused(self) -> None:
        bar = self.bar
        self.outside(lambda song: song["channels"]["lead"].pop("echo"))
        # The bar has not had its frame, and shows the echo it read. Switched off, the file is as asked already.
        self.assertTrue(bar.echo.is_on)
        click(self.ui, middle(bar.echo.rect))
        self.assertEqual((bar.echo.is_on, self.channel(), self.session.message, self.session.undo_label),
                         (False, {"inst": "tone", "gain": 0.8, "pan": -0.3}, "changed outside: walk",
                          "changed outside: walk"))
        # No echo was switched off here, so there is none to bring back.
        click(self.ui, middle(bar.echo.rect))
        self.assertEqual((self.channel()["echo"], self.session.undo_label), ({"rows": 3, "gain": 0.3}, "lead: echo on"))

    def test_a_change_from_outside_is_shown_and_while_a_knob_is_dragged_it_waits_for_the_hand_to_let_go(self) -> None:
        bar = self.bar
        self.outside(lambda song: song["channels"]["lead"].update(gain=0.5))
        bar.update()
        self.assertEqual(bar.level.value, 0.5)
        mouse = self.drag(bar.pan, 10, let_go=False)
        self.outside(lambda song: song["channels"]["lead"].update(gain=0.25, inst="thud"))
        bar.update()
        # Shown its channel again, as a song panel does when the playing moves on, the bar leaves it to the hand.
        bar.show("walk", "lead", theme.SONG)
        self.assertEqual((bar.level.value, bar.inst.picked, bar.pan.show(bar.pan.value), bar.busy),
                         (0.5, 2, "L 10", True))
        # The release saves the pan beside what was written outside, and the bar then shows what the file holds.
        self.ui.handle(Event("up", mouse, button=1))
        self.assertEqual(self.channel(), {"inst": "thud", "gain": 0.25, "pan": -0.1, "echo": ECHO})
        self.assertEqual((bar.level.value, bar.inst.picked, bar.pan.value, self.session.message),
                         (0.25, 3, -0.1, ""))

    def test_another_channel_shown_while_a_knob_is_dragged_takes_the_bar_and_nothing_is_saved(self) -> None:
        bar, before = self.bar, self.data()
        x, y = self.drag(bar.level, 5, let_go=False)
        bar.show("walk", "kit", theme.SALMON)
        self.assertEqual((bar.busy, bar.name.text, bar.level.value), (False, "kit", 0.6))
        self.ui.handle(Event("move", (x, y - 5)))
        self.ui.handle(Event("up", (x, y - 5), button=1))
        self.assertEqual((self.data(), self.session.can_undo, bar.level.value), (before, False, 0.6))

    def test_a_change_that_is_refused_leaves_the_widget_showing_what_the_file_holds_and_the_session_says_why(
            self) -> None:
        bar = self.bar

        def gone(song: dict) -> None:
            """Take an instrument of the song's own out of it, and one of its channels."""
            del song["instruments"]["own"], song["channels"]["low"], song["patterns"]["a"]["low"]

        # Another program has written the file, and the session has not looked at it yet.
        written = self.outside(gone, look=False)
        x, y, _, _ = bar.inst.rect
        self.ui.handle(Event("wheel", (x + 3, y + 3), wheel=-2))
        self.assertEqual((bar.inst.options[bar.inst.picked], self.session.message),
                         ("tone", "songs.walk.channels.lead.inst: no instrument named 'own'"))
        bar.show("walk", "low", theme.SOUND)
        self.drag(bar.level, -10)
        self.assertEqual((bar.level.value, self.session.message), (1, "songs.walk.channels has no key 'low'"))
        click(self.ui, middle(bar.echo.rect))
        self.assertEqual((bar.echo.is_on, bar.after.visible, self.session.message),
                         (False, False, "songs.walk.channels has no key 'low'"))
        # Nothing was saved, no step was kept, and no instrument was heard.
        self.assertEqual((self.data(), self.session.can_undo, self.session.messages, self.heard()),
                         (written, False, 3, []))


@unittest.skipUnless(pygame, NEEDS)
class Drawn(Opened):
    def test_the_lit_travel_of_the_level_knob_grows_when_its_value_is_raised(self) -> None:
        surface = pygame.Surface((960, 540))
        canvas = Canvas(surface, Font(sprites.load("font")), sprites.load("icons"))
        x, y, w, h = self.bar.level.rect
        lit = []
        for up in (0, 30):
            if up:
                self.drag(self.bar.level, up)
            surface.fill(theme.PANEL)
            self.ui.draw(canvas)
            lit.append(sum(surface.get_at((x + column, y + row))[:3] == theme.SONG
                           for column in range(w) for row in range(h)))
        # Eleven marks round the knob for a travel from 0 to 2: at 0.8 the fifth is reached, and at 1.4 the eighth.
        self.assertEqual((lit, self.channel()["gain"]), ([5, 8], 1.4))


if __name__ == "__main__":
    unittest.main()
