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
from rowsynth.gui.soundbar import DROPPED, SEVERAL_HELP, USUAL_ECHO, arrange, db_text, hertz_text, seconds_text
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

#: The sound panel as the middle of the art surface has it, and the bar in it.
PANEL = (0, 0, 662, 480)
BAR = (8, 29, 646, 37)

#: The notes of a sound of two voices: c4 and then e4 in the first, and between them g4 in the second.
FIRST = [[0, "c4", 6], [12, "e4", 6]]
SECOND = [[6, "g4", 6]]
#: An instrument that is written in place in its voice, and the echo one sound is written with.
NOISE = {"wave": "noise", "release": 0}
ECHO = {"time": 0.1, "feedback": 0.5, "mix": 0.4}


def solo(*notes: list, inst: str | dict = "tone") -> dict:
    """The definition of a sound of one voice with some notes."""
    return {"voices": [{"inst": inst, "notes": list(notes)}]}


#: A project for the bar to change: a sound of two voices, one with an echo and a level, one whose echo has two
#: times, two variants with a level each, two parts, and a sound whose instrument is written in place.
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
    },
}

#: The widgets of a bar, by what the bar calls them, and the parts that give way when it is narrow, the first first.
WIDGETS = ("word", "heading_voice", "voices", "add", "drop", "heading", "inst", "level", "echo", "after", "loud",
           "again", "bright")
GIVING = tuple(part for group in DROPPED for part in group)


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
    def test_the_parts_stand_from_the_left_each_heading_over_what_it_names(self) -> None:
        self.assertEqual(arrange(BAR, 2), {
            "word": (8, 29, 0, 0), "voice": (8, 29, 56, 11), "voices": (8, 41, 26, 13), "add": (36, 41, 13, 13),
            "drop": (51, 41, 13, 13), "heading": (70, 29, 78, 11), "inst": (70, 41, 78, 13),
            "level": (154, 29, 47, 37), "echo": (207, 41, 31, 13), "after": (244, 29, 47, 37),
            "loud": (297, 29, 47, 37), "again": (350, 29, 47, 37), "bright": (403, 29, 47, 37)})
        # For one sound of several the words for it come first, and the rest stand 65 pixels further on. A
        # voice more is 13 pixels more for all that comes after the voices.
        several = arrange(BAR, 1, True)
        self.assertEqual([several[part] for part in ("word", "voices", "voice", "inst", "bright")],
                         [(8, 41, 59, 13), (73, 41, 13, 13), (73, 29, 43, 11), (122, 41, 78, 13), (455, 29, 47, 37)])
        self.assertEqual([arrange(BAR, 5)[part] for part in ("voices", "add", "bright")],
                         [(8, 41, 65, 13), (75, 41, 13, 13), (442, 29, 47, 37)])
        self.assertEqual((seconds_text(0.09), hertz_text(4000), db_text(-6.5), db_text(0)),
                         ("0.09 S", "4000 HZ", "-6.5 DB", "0 DB"))

    def test_a_narrow_bar_drops_the_knobs_of_the_echo_first_and_the_voices_last_and_no_two_parts_overlap(self) -> None:
        dropped = {}
        for wide in (507, 506, 295, 294, 242, 241, 177, 176, 140, 139, 56, 55, 26, 25, 0):
            bar = (8, 29, wide, 37)
            parts = arrange(bar, 2, True)
            dropped[wide] = sum(parts[part][2] == 0 for part in GIVING)
            # What is dropped is dropped in the one order, and what is left is inside the bar, side by side.
            self.assertEqual([part for part in GIVING if parts[part][2] == 0], list(GIVING[:dropped[wide]]), wide)
            self.assertTrue(all(lies_in(part, bar) for part in parts.values()), wide)
            spans = sorted((x, x + w) for name, (x, _, w, _) in parts.items() if w and name in GIVING)
            self.assertTrue(all(end <= start for (_, end), (start, _) in zip(spans, spans[1:])), wide)
            # A heading is there with what it names, and is no wider than that.
            self.assertEqual((parts["voice"][2] > 0, parts["heading"][2] > 0),
                             (parts["voices"][2] > 0, parts["inst"][2] > 0), wide)
        self.assertEqual(dropped, {507: 0, 506: 4, 295: 4, 294: 5, 242: 5, 241: 6, 177: 6, 176: 7, 140: 7, 139: 8,
                                   56: 8, 55: 10, 26: 10, 25: 11, 0: 11})


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
    def test_the_bar_has_the_voices_the_instrument_the_level_and_the_echo_of_the_sound_as_the_file_has_them(
            self) -> None:
        bar = self.bar
        self.assertEqual((bar.name, bar.index, bar.voice, self.view.voice), ("two", 0, 0, 0))
        self.assertEqual(self.seen(), ["heading_voice", "voices", "add", "drop", "heading", "inst", "level", "echo"])
        self.assertEqual((bar.voices.options, bar.voices.picked, bar.voices.colour, bar.voices.rect),
                         (("1", "2"), 0, theme.SOUND, (8, 41, 26, 13)))
        self.assertEqual((bar.heading_voice.text, bar.add.text, bar.drop.text, bar.heading.text),
                         ("VOICE", "+", "-", "INSTRUMENT"))
        self.assertEqual((bar.inst.options, bar.inst.picked, bar.level.value, bar.level.show(bar.level.value),
                          bar.echo.is_on), (("tone", "buzz", "bell"), 0, 0, "0 DB", False))
        # The second voice picked, it is its instrument that is shown, and the view places its notes in it.
        self.click(middle(bar.voices.cells()[1]))
        self.assertEqual((bar.voice, bar.voices.picked, bar.inst.options[bar.inst.picked], self.view.voice),
                         (1, 1, "buzz", 1))
        # A sound with an echo has a knob for each of its settings, the one it leaves out as the engine takes
        # it, and each is shown in its unit.
        self.pick("ring")
        knobs = (bar.level, bar.after, bar.loud, bar.again, bar.bright)
        self.assertEqual(self.seen()[-5:], ["echo", "after", "loud", "again", "bright"])
        self.assertEqual([(knob.text, knob.value, knob.show(knob.value)) for knob in knobs],
                         [("LEVEL", -6, "-6 DB"), ("AFTER", 0.1, "0.1 S"), ("LOUD", 0.4, "0.4"), ("AGAIN", 0.5, "0.5"),
                          ("BRIGHT", 3000, "3000 HZ")])
        self.assertEqual((bar.echo.is_on, bar.voice, self.view.voice, {knob.colour for knob in knobs}),
                         (True, 0, 0, {theme.SOUND}))
        # An echo of several times has no one number for the knob of its seconds, which says so.
        self.pick("wide")
        self.assertEqual((bar.after.visible, bar.after.enabled, bar.after.help, bar.loud.enabled),
                         (True, False, SEVERAL_HELP, True))
        # An instrument that is written in place stands before those of the project, under the word for one.
        self.pick("own")
        self.assertEqual((bar.inst.options, bar.inst.picked, bar.after.enabled),
                         (("own", "tone", "buzz", "bell"), 0, True))

    def test_of_several_sounds_the_bar_is_of_the_one_the_hand_is_in_and_says_which(self) -> None:
        bar = self.bar
        self.pick("pair")
        self.assertEqual((self.seen()[0], bar.word.text, bar.word.rect, bar.index, bar.level.value,
                          bar.inst.options[bar.inst.picked]), ("word", "VARIANT 1", (8, 41, 59, 13), 0, 0, "tone"))
        # A press on the label of the second variant has the hand in that one, and the bar by the next frame.
        self.click(self.label(1))
        self.panel.update()
        self.assertEqual((self.view.sound, bar.word.text, bar.index, bar.level.value,
                          bar.inst.options[bar.inst.picked]), (1, "VARIANT 2", 1, -12, "buzz"))
        # An entry of parts has no level of its own, so the bar has no knob for one.
        self.pick("both")
        self.assertEqual((bar.word.text, bar.index, bar.level.visible, bar.echo.visible), ("PART 1", 0, False, True))
        self.click(self.label(1))
        self.panel.update()
        self.assertEqual((bar.word.text, bar.index), ("PART 2", 1))
        # With an instrument picked, or the panel off the window, nothing of the bar is seen.
        self.session.pick("instrument", "tone")
        self.panel.update()
        self.assertEqual(self.seen(), [])
        self.pick("two")
        self.panel.show(False)
        self.assertEqual(self.seen(), [])
        self.panel.show(True)
        self.assertEqual(self.seen()[1:3], ["voices", "add"])


class Saved(Opened):
    def test_a_voice_is_added_with_the_instrument_of_the_picked_one_and_removed_but_for_the_last(self) -> None:
        bar, before = self.bar, self.data()
        self.click(middle(bar.voices.cells()[1]))
        self.click(middle(bar.add.rect))
        voices = self.sound()["def"]["voices"]
        self.assertEqual((differing(before, self.data()), voices[2:], self.session.undo_label),
                         (["sounds.two.def.voices"], [{"inst": "buzz", "notes": []}], "two: voice added"))
        # The new voice is the picked one, in the bar and in the view, and the bar has made room for it.
        self.assertEqual((bar.voices.options, bar.voice, self.view.voice, bar.voices.rect, bar.add.rect),
                         (("1", "2", "3"), 2, 2, (8, 41, 39, 13), (49, 41, 13, 13)))
        # Removed, it is gone with nothing else, and the voice before it is the picked one.
        self.click(middle(bar.drop.rect))
        self.assertEqual((self.data(), self.session.undo_label, bar.voice, self.view.voice),
                         (before, "two: voice 3 removed", 1, 1))
        self.click(middle(bar.drop.rect))
        left = self.data()
        self.assertEqual((self.sound()["def"]["voices"], self.session.undo_label, bar.voices.options),
                         ([{"inst": "tone", "notes": FIRST}], "two: voice 2 removed", ("1",)))
        # The last voice is not removed: the session says why, and nothing is saved.
        self.click(middle(bar.drop.rect))
        self.assertEqual((self.data(), self.session.undo_label, self.session.message),
                         (left, "two: voice 2 removed", "two: the last voice of a sound is not dropped: a sound with "
                                                        "none has nothing to play"))

    def test_the_instrument_of_the_picked_voice_is_stepped_and_one_written_in_place_comes_back_by_undo(self) -> None:
        bar, before = self.bar, self.data()
        x, y, w, _ = bar.inst.rect
        self.click((x + w - 3, y + 3))
        self.assertEqual((differing(before, self.data()), self.sound()["def"]["voices"][0]["inst"], bar.inst.picked,
                          self.session.undo_label), (["sounds.two.def.voices"], "buzz", 1, "two: voice 1 plays buzz"))
        # Two steps with the wheel are one change.
        self.ui.handle(Event("wheel", (x + 3, y + 3), wheel=1))
        self.assertEqual((self.sound()["def"]["voices"][0]["inst"], self.session.undo_label),
                         ("bell", "two: voice 1 plays bell"))
        # An instrument written in place is replaced only when the stepper is stepped, and is then no longer
        # one of those to step to.
        self.pick("own")
        self.assertEqual((bar.inst.options[bar.inst.picked], self.sound("own")["def"]["voices"][0]["inst"]),
                         ("own", NOISE))
        x, y, w, _ = bar.inst.rect
        self.click((x + w - 3, y + 3))
        self.assertEqual((self.sound("own")["def"]["voices"][0]["inst"], bar.inst.options, bar.inst.picked,
                          self.session.undo_label), ("tone", ("tone", "buzz", "bell"), 0, "own: voice 1 plays tone"))
        # The step taken back brings it back as it was written.
        self.assertTrue(self.session.undo())
        self.panel.update()
        self.assertEqual((self.sound("own")["def"]["voices"][0]["inst"], bar.inst.options[bar.inst.picked]),
                         (NOISE, "own"))

    def test_the_echo_is_switched_on_as_the_usual_one_and_off_and_on_again_it_is_the_echo_it_was(self) -> None:
        bar, before = self.bar, self.data()
        self.click(middle(bar.echo.rect))
        self.assertEqual((differing(before, self.data()), self.sound()["def"]["echo"], self.session.undo_label),
                         (["sounds.two.def.echo"], USUAL_ECHO, "two: echo on"))
        self.assertEqual((bar.echo.is_on, self.seen()[-4:], bar.after.value, bar.bright.value),
                         (True, ["after", "loud", "again", "bright"], 0.09, 4000))
        self.click(middle(bar.echo.rect))
        self.assertEqual((self.data(), self.session.undo_label, bar.echo.is_on, bar.after.visible),
                         (before, "two: echo off", False, False))
        # The echo of another sound is switched off, and on again it has the settings it had.
        self.pick("ring")
        self.click(middle(bar.echo.rect))
        self.assertEqual(("echo" in self.sound("ring")["def"], self.session.undo_label), (False, "ring: echo off"))
        self.click(middle(bar.echo.rect))
        self.assertEqual((self.sound("ring"), bar.again.value), (PROJECT["sounds"]["ring"], 0.5))

    def test_a_knob_of_the_echo_changes_nothing_in_the_file_until_it_is_let_go_and_then_its_one_setting(self) -> None:
        self.pick("ring")
        bar, before = self.bar, self.data()
        mouse = self.turn(bar.after, 5, let_go=False)
        # The knob shows where the hand has it.
        self.assertEqual((bar.after.show(bar.after.value), bar.busy), ("0.15 S", True))
        self.assertEqual((self.data(), self.session.can_undo), (before, False))
        self.ui.handle(Event("up", mouse, button=1))
        self.assertEqual((differing(before, self.data()), self.session.undo_label, bar.after.value, bar.busy),
                         (["sounds.ring.def.echo.time"], "ring: echo after 0.15 S", 0.15, False))
        # Each of the others is saved the same way, as its own setting, and one the file left out is written.
        labels = []
        for knob, up in ((bar.loud, 10), (bar.again, 20), (bar.bright, 10)):
            self.turn(knob, up)
            labels.append(self.session.undo_label)
        self.assertEqual(labels, ["ring: echo loud 0.5", "ring: echo again 0.69", "ring: echo bright 4200 HZ"])
        self.assertEqual(self.sound("ring"), {"def": {**solo([0, "a4", 6]), "echo": {
            "time": 0.15, "feedback": 0.69, "mix": 0.5, "lp": 4200}}, "db": -6})
        # One step back, the knob shows what the file then holds.
        self.assertTrue(self.session.undo())
        self.panel.update()
        self.assertEqual((bar.bright.value, "lp" in self.sound("ring")["def"]["echo"]), (3000, False))

    def test_the_level_is_saved_when_its_knob_is_let_go_and_at_0_it_is_taken_out_of_the_file(self) -> None:
        bar, before = self.bar, self.data()
        self.turn(bar.level, -20)
        self.assertEqual((differing(before, self.data()), self.sound()["db"], self.session.undo_label,
                          bar.level.value), (["sounds.two.db"], -6.5, "two: level -6.5 DB", -6.5))
        # A right click puts it back to the 0 that a sound has when it says none, which is not written.
        self.click(middle(bar.level.rect), button=3)
        self.assertEqual((self.data(), self.session.undo_label, bar.level.value), (before, "two: level 0 DB", 0))
        # Where each variant has its own level, the knob is that of the variant the hand is in.
        self.pick("pair")
        self.click(self.label(1))
        self.panel.update()
        self.turn(bar.level, 10)
        self.assertEqual((self.sound("pair")["db"], self.session.undo_label, bar.level.value),
                         ([0, -8.5], "pair: variant 2 level -8.5 DB", -8.5))
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
