"""Tests for rowsynth.gui.soundbar: the sums of the bar, what it shows of a sound, and what is saved of its changes."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from collections.abc import Sequence
from pathlib import Path

from rowsynth.gui import layout, soundview, sprites, theme
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.font import Font
from rowsynth.gui.model import Model
from rowsynth.gui.session import Session
from rowsynth.gui.soundbar import (SEVERAL_HELP, USUAL_ECHO, VOICE_W, VOLUME_W, db_text, hertz_text, knob_room,
                                   seconds_text, ticks_text, voice_parts, voices_room)
from rowsynth.gui.soundpanel import SoundPanel
from rowsynth.gui.ui import Event, Ui, Widget
from rowsynth.gui.wave import places

# pygame reads these when it is imported: no test opens a window or plays a sound.
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
try:
    import pygame
except ImportError:
    pygame = None

NEEDS = "pygame is not installed"

#: The sound panel as the middle of the art surface has it, and the bar in it: a knob of the bar stands from
#: y 25, and a small control from y 37.
PANEL = (0, 0, 662, 480)
BAR = (4, 23, 654, 40)

#: The notes of a sound of two voices: c4 and then e4 in the first, and between them g4 in the second.
FIRST = [[0, "c4", 6], [12, "e4", 6]]
SECOND = [[6, "g4", 6]]
#: An instrument that is written in place in its voice, and the echo one sound is written with.
NOISE = {"wave": "noise", "release": 0}
ECHO = {"time": 0.1, "feedback": 0.5, "mix": 0.4}


def solo(*notes: list, inst: str | dict = "tone") -> dict:
    """The definition of a sound of one voice with some notes."""
    return {"voices": [{"inst": inst, "notes": list(notes)}]}


#: A project for the bar to change: a sound of two voices, one with an echo and a volume, one whose echo has two
#: times, two variants with a volume each, two parts, a sound whose instrument is written in place, and a held one.
PROJECT = {
    "instruments": {"tone": {"wave": "tri", "release": 0}, "buzz": {"wave": "saw", "release": 0},
                    "bell": {"wave": "pulse", "release": 0}},
    "sounds": {
        "two": {"def": {"voices": [{"inst": "tone", "notes": FIRST}, {"inst": "buzz", "notes": SECOND}]}},
        "ring": {"def": {**solo([0, "a4", 6]), "echo": ECHO}, "db": -6},
        "wide": {"def": {**solo([0, "a4", 6]), "echo": {"time": [0.05, 0.11]}}},
        "pair": {"defs": [solo([0, "c4", 6]), solo([0, "c5", 12], inst="buzz")], "db": [0, -12]},
        "both": {"parts": [[0, {"def": solo([0, "c4", 10]), "db": -3}], [0.1, {"def": solo([0, "c5", 6])}]]},
        "own": {"def": solo([0, "c4", 6], inst=NOISE)},
        "hum": {"held": {"note": solo([0, "c4", 24]), "start": 6, "loop": 12, "join": 2, "end": solo([0, "c4", 6]),
                         "let_go": 4}},
    },
}

#: The controls of a bar, by what the bar calls them, from the left.
WIDGETS = ("voices", "add", "inst", "list", "volume", "crossfade", "release", "echo", "time", "amount", "repeats",
           "tone")


class Speaker:
    """A stand-in for the player: it keeps what it was asked, in order, and plays nothing."""

    def __init__(self) -> None:
        self.asked: list[str] = []
        self.playing = False

    def play(self, sides: Sequence, loop: bool = False) -> None:
        self.asked.append("play")
        self.playing = True

    def also(self, sides: Sequence) -> None:
        self.asked.append("also")

    def stop(self, fade: float = 0.0) -> None:
        self.asked.append("stop")
        self.playing = False

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


def lies_in(rect: layout.Rect, bounds: layout.Rect) -> bool:
    """Whether a rectangle has no pixel outside another, and neither a width nor a height below 0."""
    x, y, w, h = rect
    return (w >= 0 and h >= 0 and bounds[0] <= x and x + w <= bounds[0] + bounds[2]
            and bounds[1] <= y and y + h <= bounds[1] + bounds[3])


class Sums(unittest.TestCase):
    def test_the_voices_have_the_room_of_their_tabs_and_the_sign_or_of_the_word_over_them(self) -> None:
        # A tab is 13 wide, and the sign that adds a voice 11, 2 pixels after the last tab.
        self.assertEqual((VOICE_W, voices_room("VOICE", 1), voices_room("VOICE", 2), voices_room("VOICE", 5)),
                         (13, Font.width("VOICE"), 39, 78))
        self.assertEqual(voices_room("VARIANT 10", 1), Font.width("VARIANT 10"))
        self.assertEqual(voice_parts((10, 37, 39, 13), 2), ((10, 37, 26, 13), (38, 38, 11, 11)))
        # A room wider than they need has them at its left end. One too narrow cuts the tabs, and in one of no
        # width neither has any.
        self.assertEqual(voice_parts((10, 37, 59, 13), 2), ((10, 37, 26, 13), (38, 38, 11, 11)))
        self.assertEqual(voice_parts((10, 37, 20, 13), 2), ((10, 37, 7, 13), (19, 38, 11, 11)))
        self.assertEqual(voice_parts((10, 37, 0, 13), 2), ((10, 37, 0, 13), (10, 38, 0, 11)))

    def test_a_knob_has_the_room_of_its_name_or_of_its_widest_value_and_shows_its_value_in_its_unit(self) -> None:
        self.assertEqual((seconds_text(0.09), hertz_text(4000), db_text(-6.5), db_text(0), ticks_text(1),
                          ticks_text(3)), ("0.09 s", "4000 Hz", "-6.5 dB", "0 dB", "1 tick", "3 ticks"))
        self.assertEqual((knob_room("PAN"), knob_room("CROSSFADE"), knob_room("TONE", "12000 Hz"), VOLUME_W),
                         (33, Font.width("CROSSFADE") + 2, Font.width("12000 Hz") + 2, Font.width("-29.5 dB") + 2))


class Opened(unittest.TestCase):
    """A project file of the test's own, a real session on it with a stand-in player, and the bar on a sound panel."""

    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        # No thread of the model is left to run into the next test, or into a folder that is gone.
        self.addCleanup(settle)
        self.path = Path(folder.name) / "project.json"
        self.path.write_text(json.dumps(PROJECT), encoding="utf-8")
        self.session = Session(Model(self.path), Speaker())
        self.session.audition = False
        self.ui = Ui()
        self.panel = SoundPanel(self.session, self.ui)
        self.panel.place(PANEL)
        self.bar, self.view = self.panel.bar, self.panel.view
        self.pick("two")

    def pick(self, name: str) -> None:
        """Pick a sound of the project, and have the panel show it over its render."""
        self.session.pick("sound", name)
        self.session.model.render()
        self.session.tick()
        self.panel.update()

    def data(self) -> dict:
        """What the file holds now."""
        return json.loads(self.path.read_text(encoding="utf-8"))

    def sound(self, name: str = "two") -> dict:
        """What the file holds now for a sound."""
        return self.data()["sounds"][name]

    def seen(self) -> list[str]:
        """Which of the bar's widgets are seen, by what the bar calls them."""
        return [name for name in WIDGETS if getattr(self.bar, name).visible]

    def click(self, pos: tuple[int, int], button: int = 1) -> None:
        """Press a mouse button at a position and let it go there."""
        self.ui.handle(Event("down", pos, button=button))
        self.ui.handle(Event("up", pos, button=button))

    def items(self) -> list[tuple[str, bool]]:
        """The items of the menu that is open: the text of each, and whether it can be chosen."""
        return [(item.text, item.enabled) for item in self.ui.modal.items if item is not None]

    def choose(self, text: str) -> None:
        """Click the item of the open menu that has a text."""
        menu = self.ui.modal
        self.click(middle(menu.rows[[item is not None and item.text == text for item in menu.items].index(True)]))

    def turn(self, widget: Widget, up: int, let_go: bool = True) -> tuple[int, int]:
        """Press the left button in the middle of a widget and move so many pixels up. Where the mouse then is."""
        x, y = middle(widget.rect)
        self.ui.handle(Event("down", (x, y), button=1))
        self.ui.handle(Event("move", (x, y - up)))
        if let_go:
            self.ui.handle(Event("up", (x, y - up), button=1))
        return x, y - up

    def label(self, shape: int) -> tuple[int, int]:
        """A pixel of the view on the label of one of the shapes of the picked entry, counted from 0."""
        model = self.session.model
        entry = model.project.sounds[model.picked[1]]
        shapes = soundview.shapes(entry, model.rate, model.project.instruments)
        seen = self.view.map
        return seen.x(places(shapes, "defs" in entry)[shape]) + 5, seen.names[1] + 2


class Shown(Opened):
    def test_the_bar_has_the_voices_the_instrument_the_volume_and_the_echo_of_the_sound_as_the_file_has_them(
            self) -> None:
        bar = self.bar
        self.assertEqual((bar.name, bar.index, bar.voice, self.view.voice, bar.word), ("two", 0, 0, 0, "VOICE"))
        self.assertEqual(self.seen(), ["voices", "add", "inst", "list", "volume", "echo"])
        # It is the bar every editor has. The tabs of the voices stand at its left end with the sign that adds
        # one after them, then the field of the instrument with the button of its list, and so on to the right.
        self.assertEqual((bar.frame.rect, bar.frame.visible, bar.voices.rect, bar.add.rect),
                         (BAR, True, (10, 37, 26, 13), (38, 38, 11, 11)))
        self.assertEqual((bar.inst.rect, bar.list.rect, bar.volume.rect, bar.echo.rect[:2]),
                         ((55, 37, 78, 13), (139, 37, 13, 13), (158, 25, VOLUME_W, 37), (213, 37)))
        self.assertEqual((bar.voices.options, bar.voices.picked, bar.voices.colour, bar.list.arrow),
                         (("1", "2"), 0, theme.SOUND, True))
        self.assertEqual((bar.inst.options, bar.inst.picked, bar.volume.text, bar.volume.value,
                          bar.volume.show(bar.volume.value), bar.echo.is_on, bar.echo.text),
                         (("tone", "buzz", "bell"), 0, "VOLUME", 0, "0 dB", False, "OFF"))
        # The second voice picked, it is its instrument that is shown, and the view places its notes in it.
        self.click(middle(bar.voices.cells()[1]))
        self.assertEqual((bar.voice, bar.voices.picked, bar.inst.options[bar.inst.picked], self.view.voice),
                         (1, 1, "buzz", 1))
        # A sound with an echo has a knob for each of its settings, the one it leaves out as the engine takes
        # it, and each is called by its word and shown in its unit.
        self.pick("ring")
        knobs = (bar.volume, bar.time, bar.amount, bar.repeats, bar.tone)
        self.assertEqual(self.seen()[-5:], ["echo", "time", "amount", "repeats", "tone"])
        self.assertEqual([(knob.text, knob.value, knob.show(knob.value)) for knob in knobs],
                         [("VOLUME", -6, "-6 dB"), ("TIME", 0.1, "0.1 s"), ("AMOUNT", 0.4, "0.4"),
                          ("REPEATS", 0.5, "0.5"), ("TONE", 3000, "3000 Hz")])
        self.assertEqual((bar.echo.is_on, bar.echo.text, bar.voice, self.view.voice, {knob.colour for knob in knobs}),
                         (True, "ON", 0, 0, {theme.SOUND}))
        # They stand in that order from the left, side by side, and none sticks out of the bar.
        spans = [(getattr(bar, name).rect[0], getattr(bar, name).rect[0] + getattr(bar, name).rect[2])
                 for name in self.seen()]
        self.assertTrue(all(end <= start for (_, end), (start, _) in zip(spans, spans[1:])))
        self.assertTrue(all(lies_in(getattr(bar, name).rect, BAR) for name in self.seen()))
        # An echo of several times has no one number for the knob of its seconds, which says so.
        self.pick("wide")
        self.assertEqual((bar.time.visible, bar.time.enabled, bar.time.help, bar.amount.enabled),
                         (True, False, SEVERAL_HELP, True))
        # An instrument that is written in place stands before those of the project, under the words for one.
        self.pick("own")
        self.assertEqual((bar.inst.options, bar.inst.picked, bar.time.enabled),
                         (("its own", "tone", "buzz", "bell"), 0, True))

    def test_of_several_sounds_the_bar_is_of_the_one_the_hand_is_in_and_says_which_over_its_voices(self) -> None:
        bar = self.bar
        self.pick("pair")
        # The words stand over the voices, which have the room of them: the instrument comes after that room.
        self.assertEqual((bar.word, bar.index, bar.volume.value, bar.inst.options[bar.inst.picked]),
                         ("VARIANT 1", 0, 0, "tone"))
        self.assertEqual((bar.voices.rect, bar.inst.rect[0]), ((10, 37, 13, 13), 16 + Font.width("VARIANT 1")))
        # A press on the label of the second variant has the hand in that one, and the bar by the next frame.
        self.click(self.label(1))
        self.panel.update()
        self.assertEqual((self.view.sound, bar.word, bar.index, bar.volume.value,
                          bar.inst.options[bar.inst.picked]), (1, "VARIANT 2", 1, -12, "buzz"))
        # An entry of parts has no volume of its own, so the bar has no knob for one, and no room for it.
        self.pick("both")
        self.assertEqual((bar.word, bar.index, bar.volume.visible, bar.echo.visible, bar.echo.rect[0]),
                         ("PART 1", 0, False, True, bar.list.rect[0] + bar.list.rect[2] + 6))
        self.click(self.label(1))
        self.panel.update()
        self.assertEqual((bar.word, bar.index), ("PART 2", 1))
        # With an instrument picked, or the panel off the window, nothing of the bar is seen.
        self.session.pick("instrument", "tone")
        self.panel.update()
        self.assertEqual((self.seen(), bar.frame.visible), ([], False))
        self.pick("two")
        self.panel.show(False)
        self.assertEqual((self.seen(), bar.frame.visible), ([], False))
        self.panel.show(True)
        self.assertEqual((self.seen()[:2], bar.frame.visible), (["voices", "add"], True))

    def test_a_held_sound_has_the_knobs_of_its_crossfade_and_its_release_between_the_volume_and_the_echo(self) -> None:
        bar, before = self.bar, self.data()
        self.pick("hum")
        self.assertEqual((bar.word, self.seen()), ("THE NOTE", ["voices", "add", "inst", "list", "volume", "crossfade",
                                                                "release", "echo"]))
        self.assertEqual([(knob.text, knob.value, knob.show(knob.value)) for knob in (bar.crossfade, bar.release)],
                         [("CROSSFADE", 2, "2 ticks"), ("RELEASE", 4, "4 ticks")])
        self.assertTrue(bar.volume.rect[0] < bar.crossfade.rect[0] < bar.release.rect[0] < bar.echo.rect[0])
        # Let go of, such a knob is the length of the held sound it stands for, and one step.
        self.turn(bar.crossfade, 10)
        after = self.data()
        self.assertEqual((differing(before, after), after["sounds"]["hum"]["held"]["join"], self.session.undo_label),
                         (["sounds.hum.held.join"], 5, "hum: crossfade 5 ticks"))
        self.turn(bar.release, 10)
        self.assertEqual((self.sound("hum")["held"]["let_go"], self.session.undo_label),
                         (10, "hum: release 10 ticks"))
        # Any other sound has neither.
        self.pick("two")
        self.assertEqual((bar.crossfade.visible, bar.release.visible), (False, False))


class Saved(Opened):
    def test_a_voice_is_added_with_the_instrument_of_the_picked_one_and_copied_and_removed_by_its_menu(self) -> None:
        bar, before = self.bar, self.data()
        self.click(middle(bar.voices.cells()[1]))
        self.click(middle(bar.add.rect))
        voices = self.sound()["def"]["voices"]
        self.assertEqual((differing(before, self.data()), voices[2:], self.session.undo_label),
                         (["sounds.two.def.voices"], [{"inst": "buzz", "notes": []}], "two: voice added"))
        # The new voice is the picked one, in the bar and in the view, and the bar has made room for its tab.
        self.assertEqual((bar.voices.options, bar.voice, self.view.voice, bar.voices.rect, bar.add.rect),
                         (("1", "2", "3"), 2, 2, (10, 37, 39, 13), (51, 38, 11, 11)))
        # The right button on its tab opens the menu of a voice, in the order every such menu has: a voice
        # has no name. Removed, it is gone with nothing else, and the voice before it is the picked one.
        self.click(middle(bar.voices.cells()[2]), button=3)
        self.assertEqual(self.items(), [("Rename...", False), ("Copy", True), ("Remove", True)])
        self.choose("Remove")
        self.assertEqual((self.data(), self.session.undo_label, bar.voice, self.view.voice),
                         (before, "two: voice 3 removed", 1, 1))
        # Copied, a voice is one more with its instrument and its notes, and the copy is the picked one.
        self.click(middle(bar.voices.cells()[0]), button=3)
        self.choose("Copy")
        self.assertEqual((self.sound()["def"]["voices"][2:], self.session.undo_label, bar.voice, self.view.voice),
                         ([{"inst": "tone", "notes": FIRST}], "two: voice 1 copied", 2, 2))
        # With a voice before it removed, the picked voice is the one it was, a place sooner.
        self.click(middle(bar.voices.cells()[0]), button=3)
        self.choose("Remove")
        self.assertEqual(([voice["inst"] for voice in self.sound()["def"]["voices"]], self.session.undo_label,
                          bar.voice, bar.voices.options), (["buzz", "tone"], "two: voice 1 removed", 1, ("1", "2")))
        # The last voice of a sound is not removed: its menu shows that, dim, and a press on it does nothing.
        self.click(middle(bar.voices.cells()[1]), button=3)
        self.choose("Remove")
        left = self.data()
        self.click(middle(bar.voices.cells()[0]), button=3)
        self.assertEqual((self.items()[-1], self.sound()["def"]["voices"]),
                         (("Remove", False), [{"inst": "buzz", "notes": SECOND}]))
        self.click(middle(self.ui.modal.rows[-1]))
        self.ui.handle(Event("key", key="escape"))
        self.assertEqual((self.data(), self.session.undo_label, self.ui.modal), (left, "two: voice 2 removed", None))

    def test_the_instrument_of_the_picked_voice_is_stepped_and_one_written_in_place_comes_back_by_undo(self) -> None:
        bar, before = self.bar, self.data()
        x, y, w, _ = bar.inst.rect
        self.click((x + w - 3, y + 3))
        self.assertEqual((differing(before, self.data()), self.sound()["def"]["voices"][0]["inst"], bar.inst.picked,
                          self.session.undo_label), (["sounds.two.def.voices"], "buzz", 1, "two: voice 1 plays buzz"))
        # A step of the wheel steps it as well.
        self.ui.handle(Event("wheel", (x + 3, y + 3), wheel=1))
        self.assertEqual((self.sound()["def"]["voices"][0]["inst"], self.session.undo_label),
                         ("bell", "two: voice 1 plays bell"))
        # The button after the field opens the list of the instruments: the one the voice plays cannot be
        # chosen, and neither can what makes a shared instrument, of one that is one already.
        self.click(middle(bar.list.rect))
        self.assertEqual(self.items(), [("tone", True), ("buzz", True), ("bell", False), ("Make shared...", False)])
        self.choose("tone")
        self.assertEqual((self.data(), self.session.undo_label), (before, "two: voice 1 plays tone"))
        # An instrument written in place is replaced only when another is stepped to or chosen, and is then no
        # longer one of those to step to.
        self.pick("own")
        self.assertEqual((bar.inst.options[bar.inst.picked], self.sound("own")["def"]["voices"][0]["inst"]),
                         ("its own", NOISE))
        x, y, w, _ = bar.inst.rect
        self.click((x + w - 3, y + 3))
        self.assertEqual((self.sound("own")["def"]["voices"][0]["inst"], bar.inst.options, bar.inst.picked,
                          self.session.undo_label), ("tone", ("tone", "buzz", "bell"), 0, "own: voice 1 plays tone"))
        # The step taken back brings it back as it was written.
        self.assertTrue(self.session.undo())
        self.panel.update()
        self.assertEqual((self.sound("own")["def"]["voices"][0]["inst"], bar.inst.options[bar.inst.picked]),
                         (NOISE, "its own"))

    def test_the_echo_is_switched_on_as_the_usual_one_and_off_and_on_again_it_is_the_echo_it_was(self) -> None:
        bar, before = self.bar, self.data()
        self.click(middle(bar.echo.rect))
        self.assertEqual((differing(before, self.data()), self.sound()["def"]["echo"], self.session.undo_label),
                         (["sounds.two.def.echo"], USUAL_ECHO, "two: echo on"))
        self.assertEqual((bar.echo.is_on, bar.echo.text, self.seen()[-4:], bar.time.value, bar.tone.value),
                         (True, "ON", ["time", "amount", "repeats", "tone"], 0.09, 4000))
        self.click(middle(bar.echo.rect))
        self.assertEqual((self.data(), self.session.undo_label, bar.echo.is_on, bar.echo.text, bar.time.visible),
                         (before, "two: echo off", False, "OFF", False))
        # The echo of another sound is switched off, and on again it has the settings it had.
        self.pick("ring")
        self.click(middle(bar.echo.rect))
        self.assertEqual(("echo" in self.sound("ring")["def"], self.session.undo_label), (False, "ring: echo off"))
        self.click(middle(bar.echo.rect))
        self.assertEqual((self.sound("ring"), bar.repeats.value), (PROJECT["sounds"]["ring"], 0.5))

    def test_a_knob_of_the_echo_changes_nothing_in_the_file_until_it_is_let_go_and_then_its_one_setting(self) -> None:
        self.pick("ring")
        bar, before = self.bar, self.data()
        mouse = self.turn(bar.time, 5, let_go=False)
        # The knob shows where the hand has it.
        self.assertEqual((bar.time.show(bar.time.value), bar.busy), ("0.15 s", True))
        self.assertEqual((self.data(), self.session.can_undo), (before, False))
        self.ui.handle(Event("up", mouse, button=1))
        self.assertEqual((differing(before, self.data()), self.session.undo_label, bar.time.value, bar.busy),
                         (["sounds.ring.def.echo.time"], "ring: echo time 0.15 s", 0.15, False))
        # Each of the others is saved the same way, as its own setting, and one the file left out is written.
        labels = []
        for knob, up in ((bar.amount, 10), (bar.repeats, 20), (bar.tone, 10)):
            self.turn(knob, up)
            labels.append(self.session.undo_label)
        self.assertEqual(labels, ["ring: echo amount 0.5", "ring: echo repeats 0.69", "ring: echo tone 4200 Hz"])
        self.assertEqual(self.sound("ring"), {"def": {**solo([0, "a4", 6]), "echo": {
            "time": 0.15, "feedback": 0.69, "mix": 0.5, "lp": 4200}}, "db": -6})
        # One step back, the knob shows what the file then holds.
        self.assertTrue(self.session.undo())
        self.panel.update()
        self.assertEqual((bar.tone.value, "lp" in self.sound("ring")["def"]["echo"]), (3000, False))

    def test_the_volume_is_saved_when_its_knob_is_let_go_and_at_0_it_is_taken_out_of_the_file(self) -> None:
        bar, before = self.bar, self.data()
        self.turn(bar.volume, -20)
        self.assertEqual((differing(before, self.data()), self.sound()["db"], self.session.undo_label,
                          bar.volume.value), (["sounds.two.db"], -6.5, "two: volume -6.5 dB", -6.5))
        # A right press puts it back to the 0 that a sound has when it says none, which is not written.
        self.click(middle(bar.volume.rect), button=3)
        self.assertEqual((self.data(), self.session.undo_label, bar.volume.value), (before, "two: volume 0 dB", 0))
        # Where each variant has its own volume, the knob is that of the variant the hand is in.
        self.pick("pair")
        self.click(self.label(1))
        self.panel.update()
        self.turn(bar.volume, 10)
        self.assertEqual((self.sound("pair")["db"], self.session.undo_label, bar.volume.value),
                         ([0, -8.5], "pair: variant 2 volume -8.5 dB", -8.5))
        self.assertEqual(self.sound("pair")["defs"], PROJECT["sounds"]["pair"]["defs"])


@unittest.skipUnless(pygame, NEEDS)
class Drawn(Opened):
    def test_each_voice_has_its_number_in_the_colour_of_its_bars_and_the_picked_one_a_strip_of_it(self) -> None:
        surface = pygame.Surface((662, 480))
        canvas = Canvas(surface, Font(sprites.load("font")))

        def inks(index: int) -> set[tuple[int, ...]]:
            """The colours of the voices that a frame has in the button of one voice, its outline left out."""
            surface.fill(theme.PANEL)
            self.ui.draw(canvas)
            x, y, w, h = self.bar.voices.cells()[index]
            found = {tuple(surface.get_at((x + column, y + row)))[:3]
                     for column in range(1, w - 1) for row in range(1, h - 1)}
            return found & {theme.SOUND, theme.SALMON, theme.ACTIVE}

        def strip(index: int) -> tuple[int, ...]:
            """The colour in the middle of the lower edge of the button of one voice."""
            x, y, w, h = self.bar.voices.cells()[index]
            return tuple(surface.get_at((x + w // 2, y + h - 1)))[:3]

        self.assertEqual((inks(0), strip(0), inks(1), strip(1)), ({theme.SOUND}, theme.SOUND, {theme.SALMON},
                                                                 theme.LINE))
        self.click(middle(self.bar.voices.cells()[1]))
        self.assertEqual((inks(0), strip(0), inks(1), strip(1)), ({theme.SOUND}, theme.LINE, {theme.SALMON},
                                                                 theme.SALMON))


if __name__ == "__main__":
    unittest.main()
