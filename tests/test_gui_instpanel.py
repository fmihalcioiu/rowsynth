"""Tests for rowsynth.gui.instpanel: where an instrument is used, the sums of the panel, and what it shows and saves."""
from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import unittest
from collections.abc import Callable, Sequence
from pathlib import Path
from unittest import mock

from rowsynth import project as validator
from rowsynth.gui import layout, sprites, theme
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.dials import WHEEL_WAIT
from rowsynth.gui.font import Font
from rowsynth.gui.instpanel import (A4, BUZZ_UNUSED, FM, LATER, NEEDED, NUMBERS, TABLES, InstPanel, Waves, arrange,
                                    key_at, keyboard, note_name, slack, used, uses)
from rowsynth.gui.model import Model
from rowsynth.gui.panels import within
from rowsynth.gui.session import POLL_EVERY, Session
from rowsynth.gui.ui import Event, Ui, Widget
from rowsynth.notes import midi
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

#: A pulse with every kind of table and a vibrato, a short noise with a filter, an FM voice, and one with no key.
BELL = {"wave": "pulse", "duty": [1, 2], "vol": [15, 12, 9, 6], "pitch": {"v": [0, -1], "loop": 1},
        "vib": {"delay": 12, "depth": 0.3, "rate": 5}, "release": 3, "gain": 0.8}
HISS = {"wave": "noise", "short": True, "lp": [4000, 2000]}
BRASS = {"wave": "fm", "alg": 5, "fb": 0.25, "ops": [{"mul": 1, "level": 1}, {"mul": 2, "level": 0.5}],
         "vol": [15, 10], "lp": 2800, "gate": 0.7}


def voice(inst: object, note: str = "c4") -> dict:
    """A voice that plays one short note of an instrument."""
    return {"inst": inst, "notes": [[0, note, 4]]}


#: A song with an instrument of its own, a name of its own for the bell, and a noise of its own called as the
#: project's is. Its third channel switches to the bell in one pattern, and by the song's name for it in the
#: other, where the first channel, whose instrument the bell is, switches away from it and back.
WALK = {
    "row_ticks": 6,
    "instruments": {"own": {"wave": "saw"}, "ring": "bell", "hiss": {"wave": "noise"}},
    "drums": {"k": {"inst": "hiss", "note": "c3"}, "s": {"inst": "ring", "note": "c5"}},
    "channels": {"lead": {"inst": "bell"}, "echo": {"inst": "ring"}, "low": {"inst": "own"},
                 "kit": {"inst": "hiss", "drums": True}},
    "patterns": {"a": {"lead": "o4 l4 c e g e", "echo": "o5 l4 c e g e", "low": "o2 l2 c @bell g", "kit": "l4 k s k s"},
                 "b": {"lead": "o4 l2 @own c @bell c", "low": "o2 l2 @ring c @own g"}},
    "order": ["a", "b"],
}
#: Sounds of each form: one sound of two voices, two variants, two parts and a held sound.
SOUNDS = {
    "ping": {"def": {"voices": [voice("bell", "c5"), voice("bell", "e5")]}},
    "pair": {"defs": [{"voices": [voice("bell")]}, {"voices": [voice("hiss")]}]},
    "boom": {"parts": [[0, {"def": {"voices": [voice("hiss")]}}],
                       [0.05, {"def": {"voices": [voice({"wave": "tri"})]}}]]},
    "hum": {"held": {"note": {"voices": [{"inst": "bell", "notes": [[0, "c3", 30]]}]}, "start": 6, "loop": 12,
                     "join": 3, "end": {"voices": [voice("hiss")]}, "let_go": 4}},
}
PROJECT = {"instruments": {"bell": BELL, "hiss": HISS, "brass": BRASS, "plain": {}}, "sounds": SOUNDS,
           "songs": {"walk": WALK}}

#: What each setting of an instrument reads on the window: the words of the widgets that show it.
WORDS = {
    "wave": ["PULSE SAW TRI NOISE FM"], "vol": ["LOUDNESS"], "duty": ["PULSE WIDTH"], "arp": ["JUMP"],
    "pitch": ["SWEEP"], "vib": ["VIBRATO", "VIB WAIT", "VIB RATE"], "slide": ["SLIDE"], "release": ["FADE"],
    "gain": ["LEVEL"], "short": ["BUZZ"], "lp": ["FILTER"], "res": ["RING"], "key_scale": ["KEY TILT"],
    "gate": ["HELD FOR"], "alg": [LATER], "fb": [LATER], "ops": [LATER],
}
#: The rate the project is rendered at on the threads of its model: a low one, to be quick.
RATE = 8000
#: The time the first write from outside is stamped with, in nanoseconds: none the file gets by being saved.
STAMP = 1_700_000_000 * 10**9


class Stand:
    """A stand-in for the session: a real model, of which nothing plays."""

    def __init__(self, model: Model) -> None:
        self.model = model
        self.playing = False


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


def apart(one: layout.Rect, other: layout.Rect) -> bool:
    """Whether two rectangles share no pixel."""
    return (one[0] + one[2] <= other[0] or other[0] + other[2] <= one[0]
            or one[1] + one[3] <= other[1] or other[1] + other[3] <= one[1])


def foot(key: layout.Rect) -> tuple[int, int]:
    """A pixel near the foot of a key of the keyboard, where no black key lies over a white one."""
    return key[0] + key[2] // 2, key[1] + key[3] - 3


class Sums(unittest.TestCase):
    def test_an_instrument_is_counted_where_it_is_used_for_each_kind_of_use(self) -> None:
        held = Project.from_dict(PROJECT)
        # Two channels have the bell, one of them by the song's name for it, and a third switches to it: one that
        # has it and switches back to it is counted once. One drum letter plays it, and voices of three sounds
        # do: two of the first, and the note of the held one.
        self.assertEqual(uses(held, "bell"), {"channels": {"walk": 2}, "switches": {"walk": 1}, "drums": {"walk": 1},
                                              "sounds": {"ping": 2, "pair": 1, "hum": 1}})
        self.assertEqual(used(uses(held, "bell")), "in 3 channels of walk, 1 drum of walk, 3 sounds")
        # The noise the song calls hiss is its own, so the project's is played by sounds alone: a variant, a
        # part and the end of the held one.
        self.assertEqual(uses(held, "hiss"), {"channels": {}, "switches": {}, "drums": {},
                                              "sounds": {"pair": 1, "boom": 1, "hum": 1}})
        self.assertEqual(used(uses(held, "hiss")), "in 3 sounds")
        # The song's own instrument is none of the project's, and nothing plays the one with no key.
        nothing = {"channels": {}, "switches": {}, "drums": {}, "sounds": {}}
        self.assertEqual([uses(held, name) for name in ("own", "plain")], [nothing, nothing])
        self.assertEqual(used(nothing), "used nowhere")
        # One of a thing is not written as many, and songs are counted where there are several.
        self.assertEqual(used({"channels": {"walk": 1}, "switches": {}, "drums": {}, "sounds": {"ping": 2}}),
                         "in 1 channel of walk, 1 sound")
        self.assertEqual(used({"channels": {"walk": 1, "run": 2}, "switches": {"run": 1, "hop": 1},
                               "drums": {"hop": 2}, "sounds": {}}), "in 5 channels of 3 songs, 2 drums of hop")

    def test_a_keyboard_is_two_octaves_of_white_keys_with_the_black_ones_over_them(self) -> None:
        keys = keyboard((50, 100, 196, 34), 48)
        whites, blacks = keys[:14], keys[14:]
        self.assertEqual([pitch for pitch, _ in whites], [48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67, 69, 71])
        self.assertEqual([pitch for pitch, _ in blacks], [49, 51, 54, 56, 58, 61, 63, 66, 68, 70])
        # The white keys fill the rectangle side by side, and a black one lies across the line between two.
        self.assertEqual([key for _, key in whites], [(50 + 14 * place, 100, 14, 34) for place in range(14)])
        self.assertEqual((blacks[0], blacks[2]), ((49, (60, 100, 9, 20)), (54, (102, 100, 9, 20))))
        # Where a black key lies over a white one the position is on the black one.
        at = [key_at((50, 100, 196, 34), 48, pos) for pos in ((50, 100), (59, 110), (60, 110), (68, 119), (68, 120),
                                                              (245, 133), (246, 133), (50, 99))]
        self.assertEqual(at, [48, 48, 49, 49, 50, 71, None, None])
        # A pitch is named as a sound spells its note, which the engine reads back as that pitch.
        self.assertEqual([note_name(pitch) for pitch in (48, 49, 58, 69, 12, 119)],
                         ["c3", "c#3", "a#3", "a4", "c0", "b8"])
        self.assertEqual([midi(note_name(pitch)) for pitch in range(12, 120)], list(range(12, 120)))

    def test_in_the_middle_of_the_window_every_part_fits_with_no_scrolling_and_none_overlaps(self) -> None:
        parts = arrange(RECT)
        self.assertEqual((parts["head"], parts["box"], parts["view"], slack(RECT)),
                         ((128, 26, 654, 17), (128, 47, 654, 475), (132, 51, 646, 467), 0))
        # Five tables one under the other, each as wide as the view, the one of four widths lower than the rest.
        self.assertEqual([parts[table.key] for table in TABLES],
                         [(132, 70, 646, 76), (132, 149, 646, 50), (132, 202, 646, 76), (132, 281, 646, 75),
                          (132, 359, 646, 76)])
        self.assertEqual((parts["later"], parts["gain"], parts["key_scale"]),
                         ((132, 70, 646, 365), (132, 441, 54, 37), (596, 441, 54, 37)))
        # The keyboard is along the foot, with the stepper of its octave before it and the room of a scope after.
        self.assertEqual((parts["octave"], parts["keys"], parts["scope"]),
                         ((132, 494, 36, 13), (174, 484, 196, 34), (376, 484, 402, 34)))
        page = [name for name in parts if name not in ("head", "box", "view", "later")]
        self.assertTrue(all(within(parts[name], parts["view"]) == parts[name] for name in page))
        self.assertTrue(all(apart(parts[one], parts[other]) for one in page for other in page if one != other))
        # The wheel has nothing to scroll: asked for or not, the page stays.
        self.assertEqual(arrange(RECT, 40), parts)

    def test_in_a_low_panel_the_page_keeps_its_height_and_is_scrolled_no_further_than_its_end(self) -> None:
        parts = arrange(SMALL)
        self.assertEqual((parts["view"], NEEDED, slack(SMALL)), ((132, 51, 484, 263), 324, 61))
        # Every table has its least height, and the knobs are as wide as a narrow page leaves them.
        self.assertEqual([parts[table.key][3] for table in TABLES], [45, 30, 45, 45, 45])
        self.assertEqual((parts["gain"], parts["vib.rate"], parts["keys"]),
                         ((132, 298, 50, 37), (456, 298, 50, 37), (174, 341, 196, 34)))
        page = [name for name in parts if name not in ("head", "box", "view", "later")]
        self.assertTrue(all(apart(parts[one], parts[other]) for one in page for other in page if one != other))
        # Scrolled, every part of the page is that much higher, and the boxes stay.
        moved = arrange(SMALL, 24)
        self.assertEqual([(moved[name][0], moved[name][1] + 24, *moved[name][2:]) for name in page],
                         [parts[name] for name in page])
        self.assertEqual([moved[name] for name in ("head", "box", "view")],
                         [parts[name] for name in ("head", "box", "view")])
        # At its end the keyboard stands on the foot of the view, and the page goes no further.
        self.assertEqual((arrange(SMALL, 61)["keys"], arrange(SMALL, 500), arrange(SMALL, -9)),
                         ((174, 280, 196, 34), arrange(SMALL, 61), parts))
        # A panel with no room at all has every part, none with a size below nothing.
        self.assertTrue(all(w >= 0 and h >= 0 for _, _, w, h in arrange((0, 0, 0, 0)).values()))


class Opened(unittest.TestCase):
    """A project file of the test's own, a real session on it with a stand-in player, and the panel."""

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
        # What a key of the piano did before the panel was there.
        self.typed: list[str] = []
        self.ui.keys[("z", frozenset())] = lambda: self.typed.append("z")
        self.panel = InstPanel(self.session, self.ui, lambda: self.now)
        self.panel.place(RECT)
        self.pick("bell")

    def pick(self, name: str, kind: str = "instrument") -> None:
        """Pick a thing of the project, with nothing played, and have the panel take a frame."""
        self.session.model.pick(kind, name)
        self.panel.update()

    def data(self) -> dict:
        """What the file holds now."""
        return json.loads(self.path.read_text(encoding="utf-8"))

    def inst(self, name: str = "bell") -> dict:
        """What the file holds now for an instrument."""
        return self.data()["instruments"][name]

    def outside(self, change: Callable[[dict], object], look: bool = True) -> dict:
        """Have another program change the instruments and write the file, and the session look at it or not."""
        held = self.data()
        change(held["instruments"])
        self.path.write_text(json.dumps(held), encoding="utf-8")
        # Stamped as no write before it was, so that the look cannot take the file for the one it knows.
        self.stamp += 10**9
        os.utime(self.path, ns=(self.stamp, self.stamp))
        if look:
            self.now += POLL_EVERY
            self.session.tick()
        return held

    def drag(self, widget: Widget, up: int, let_go: bool = True) -> tuple[int, int]:
        """Press the left button in the middle of a widget and move so many pixels up. Where the mouse then is."""
        x, y = middle(widget.rect)
        self.ui.handle(Event("down", (x, y), button=1))
        self.ui.handle(Event("move", (x, y - up)))
        if let_go:
            self.ui.handle(Event("up", (x, y - up), button=1))
        return x, y - up

    def bar(self, key: str, step: int, value: float) -> tuple[int, int]:
        """The pixel of a table at which a press draws a step to a value."""
        seen = self.panel.tables[key].map
        return seen.x(step) + 1, seen.y(value)

    def key(self, pitch: int) -> tuple[int, int]:
        """A pixel of the keyboard that is on the key of a pitch and on no other."""
        found = dict(keyboard(self.panel.keys.rect, self.panel.keys.low))[pitch]
        return foot(found) if pitch % 12 in (0, 2, 4, 5, 7, 9, 11) else middle(found)

    def heard(self) -> list[Sequence]:
        """The sides of each note the player was asked to sound over what plays, in order."""
        return [asked[1] for asked in self.speaker.asked if asked[0] == "also"]

    def tone(self, pitch: float, inst: str = "bell") -> Sequence:
        """The sides of a note of an instrument as the model keeps it now."""
        return self.session.model.tone(inst, pitch)["sides"]

    def seen(self) -> list[str]:
        """The parts of the page whose widgets are seen, by what `arrange` calls them."""
        return [part for part, widget in self.panel._placed.items() if widget.visible]


class Shown(Opened):
    def test_every_setting_the_validator_of_an_instrument_knows_has_a_widget_that_is_seen(self) -> None:
        panel = self.panel
        with mock.patch.object(validator, "_record") as record:
            validator._instrument({}, "instruments.any")
        known = list(record.call_args.args[2])
        # A setting the engine gets later is in the validator's table, and fails here until a widget shows it.
        self.assertEqual((len(known), sorted(panel.settings)), (17, sorted(known)))

        def words(widget: Widget) -> str:
            return " ".join(widget.options) if isinstance(widget, Waves) else widget.text

        self.assertEqual({name: [words(widget) for widget in panel.settings[name]] for name in known}, WORDS)
        mine = [widget for name in known if name not in FM for widget in panel.settings[name]]
        # A choice, a toggle, five tables and nine knobs, each with a line that says what it does to the sound.
        self.assertEqual(len(mine), 16)
        self.assertTrue(all(widget in panel.widgets and widget.visible and widget.help for widget in mine))
        # The three of the FM wave have the line that stands for them, which is seen with an FM instrument.
        self.assertEqual(([panel.settings[name] for name in FM], panel.later.visible), ([(panel.later,)] * 3, False))
        self.pick("brass")
        self.assertTrue(panel.later.visible)

    def test_the_widgets_show_what_the_file_holds_and_what_the_engine_takes_for_what_it_does_not(self) -> None:
        panel = self.panel
        self.assertEqual((panel.name, panel.used), ("bell", "in 3 channels of walk, 1 drum of walk, 3 sounds"))
        self.assertEqual((panel.wave.options[panel.wave.picked], panel.wave.waves[panel.wave.picked],
                          panel.short.is_on), ("PULSE", "pulse", False))
        self.assertEqual({key: table.table for key, table in panel.tables.items()},
                         {"vol": [15, 12, 9, 6], "duty": [1, 2], "arp": 0, "pitch": {"v": [0, -1], "loop": 1},
                          "lp": 0})
        self.assertEqual({key: knob.value for key, knob in panel.knobs.items()},
                         {"gain": 0.8, "release": 3, "gate": 0.9, "slide": 0, "vib.depth": 0.3, "vib.delay": 12,
                          "vib.rate": 5, "res": 0, "key_scale": 0})
        # An instrument with no key at all is the engine's own: a square pulse at full level, with nothing done
        # to it, a fade of 2 ticks and a note held for 0.9 of its length.
        self.pick("plain")
        self.assertEqual((panel.name, panel.used, panel.wave.picked, panel.short.is_on),
                         ("plain", "used nowhere", 0, False))
        self.assertEqual([table.table for table in panel.tables.values()], [15, 2, 0, 0, 0])
        self.assertEqual([knob.value for knob in panel.knobs.values()], [1, 2, 0.9, 0, 0, 0, 6, 0, 0])
        # The ranges: a table as the mouse draws in it, and a knob with the value a right click puts it back to.
        self.assertEqual([(table.low, table.high, table.word) for table in panel.tables.values()],
                         [(0, 15, "loudness"), (0, 3, "width"), (-12, 12, "jump"), (-12, 12, "sweep"),
                          (0, 8000, "filter")])
        self.assertEqual([(knob.low, knob.high, knob.steps, knob.default) for knob in panel.knobs.values()],
                         [(0, 2, 200, 1), (0, 60, 60, 2), (0.05, 1, 95, 0.9), (-24, 24, 96, 0), (0, 2, 200, 0),
                          (0, 60, 60, 0), (1, 20, 38, 6), (0, 0.98, 98, 0), (-12, 12, 48, 0)])
        self.assertEqual([knob.colour for knob in panel.knobs.values()] + [table.colour for table in
                                                                          panel.tables.values()],
                         [theme.INSTRUMENT] * 14)

    def test_a_table_or_a_toggle_the_wave_does_not_use_is_dim_takes_no_mouse_and_says_why(self) -> None:
        panel, duty = self.panel, self.panel.tables["duty"]
        # Under the pulse every table is used, and the short noise is not.
        self.assertEqual(([table.enabled for table in panel.tables.values()], panel.short.enabled, panel.short.help),
                         ([True] * 5, False, BUZZ_UNUSED))
        self.assertEqual(duty.help, TABLES[1].help)
        self.pick("hiss")
        self.assertEqual(([table.enabled for table in panel.tables.values()], panel.short.enabled, panel.short.is_on),
                         ([True, False, True, True, True], True, True))
        self.assertEqual((duty.help, TABLES[1].unused),
                         ("only the pulse wave has a width: with another wave this table does nothing",) * 2)
        # The reason is the line the status bar shows with the mouse on the dim table, and a press draws nothing.
        before = self.data()
        at = self.bar("duty", 0, 3)
        click(self.ui, at)
        self.assertEqual((self.ui.help, duty.table, self.data(), self.session.can_undo),
                         (TABLES[1].unused, 2, before, False))
        # The two knobs of the vibrato wait for it to have a depth.
        self.assertEqual([panel.knobs[key].enabled for key in ("vib.depth", "vib.delay", "vib.rate")],
                         [True, False, False])

    def test_an_fm_instrument_has_the_line_in_place_of_the_tables_and_another_wave_picked_loses_nothing(self) -> None:
        panel = self.panel
        self.pick("brass")
        self.assertEqual(self.seen(), ["word", "wave", "short", *(number.key for number in NUMBERS), "octave", "keys"])
        self.assertEqual((panel.later.visible, panel.later.text, panel.later.rect, panel.later.colour),
                         (True, "the wiring and the four operators of the FM wave come in a later step",
                          arrange(RECT)["later"], theme.DIM))
        self.assertEqual((panel.wave.waves[panel.wave.picked], panel.knobs["gate"].value), ("fm", 0.7))
        # Another wave is that one place in the file: the settings of the FM wave stay, to come back to.
        before = self.data()
        click(self.ui, middle(panel.wave.cells()[1]))
        self.assertEqual((differing(before, self.data()), self.inst("brass")),
                         (["instruments.brass.wave"], {**BRASS, "wave": "saw"}))
        self.assertEqual((self.session.undo_label, panel.later.visible, self.seen()[3:8]),
                         ("brass: wave saw", False, ["vol", "duty", "arp", "pitch", "lp"]))
        self.assertEqual((panel.tables["vol"].table, panel.tables["lp"].table), ([15, 10], 2800))
        # Picking FM for an instrument writes its wave and nothing more.
        self.pick("plain")
        click(self.ui, middle(panel.wave.cells()[4]))
        self.assertEqual((self.inst("plain"), panel.later.visible, panel.tables["vol"].visible),
                         ({"wave": "fm"}, True, False))

    def test_with_a_song_a_sound_or_nothing_picked_and_off_the_window_the_panel_shows_nothing(self) -> None:
        panel = self.panel
        shown = [widget for widget in panel.widgets if widget.visible]
        self.assertEqual(shown, [widget for widget in panel.widgets if widget is not panel.later])
        for kind, name in (("song", "walk"), ("sound", "ping")):
            self.pick(name, kind)
            self.assertEqual((panel.name, [widget for widget in panel.widgets if widget.visible]), (None, []))
        self.pick("bell")
        panel.show(False)
        self.assertEqual([widget for widget in panel.widgets if widget.visible], [])
        panel.show(True)
        self.assertEqual([widget for widget in panel.widgets if widget.visible], shown)
        # An instrument that goes from the file outside goes from the panel.
        self.pick("plain")
        self.outside(lambda instruments: instruments.pop("plain"))
        panel.update()
        self.assertEqual((self.session.model.picked, panel.name), (("song", "walk"), None))


class Saved(Opened):
    def test_a_wave_that_is_picked_is_saved_at_once_at_its_one_place(self) -> None:
        panel, before = self.panel, self.data()
        click(self.ui, middle(panel.wave.cells()[2]))
        self.assertEqual((differing(before, self.data()), self.inst()["wave"], panel.wave.picked),
                         (["instruments.bell.wave"], "tri", 2))
        self.assertEqual((self.session.undo_label, panel.tables["duty"].enabled), ("bell: wave tri", False))
        # An instrument that named no wave has the one that was picked written, the pulse too.
        self.pick("plain")
        click(self.ui, middle(panel.wave.cells()[3]))
        click(self.ui, middle(panel.wave.cells()[0]))
        self.assertEqual((self.inst("plain"), self.session.undo_label, panel.short.enabled),
                         ({"wave": "pulse"}, "plain: wave pulse", False))

    def test_a_bar_drawn_in_each_table_saves_that_table_in_the_shape_the_file_takes_and_nothing_else(self) -> None:
        panel = self.panel
        low = self.bar("lp", 0, 0)
        # A list stays a list, one with a loop keeps its loop, and one step of a table the file did not have
        # is written as the number alone.
        drawn = (("vol", "vol", self.bar("vol", 1, 5), [15, 5, 9, 6], "bell: loudness table"),
                 ("duty", "duty", self.bar("duty", 0, 3), [3, 2], "bell: width table"),
                 ("arp", "arp", self.bar("arp", 0, 7), 7, "bell: jump table"),
                 ("pitch", "pitch.v", self.bar("pitch", 0, -4), {"v": [-4, -1], "loop": 1}, "bell: sweep table"),
                 ("lp", "lp", (low[0], low[1] - 30), panel.tables["lp"].map.value_at(low[1] - 30),
                  "bell: filter table"))
        for key, place, at, table, label in drawn:
            before = self.data()
            self.ui.handle(Event("down", at, button=1))
            self.assertEqual((self.data(), panel.tables[key].table), (before, table))
            self.ui.handle(Event("up", at, button=1))
            # Nothing but the table differs: of the one with a loop, its steps alone.
            self.assertEqual((differing(before, self.data()), self.inst()[key]), ([f"instruments.bell.{place}"], table))
            self.assertEqual((panel.tables[key].table, self.session.undo_label), (table, label))
        # Thirty of the 57 rows of the filter, whose top is 8000 Hz, are a whole number of hertz.
        self.assertEqual((self.inst()["lp"], type(self.inst()["lp"])), (4211, int))
        # A drag over two bars sets them both, and is one change to the file when the mouse lets go.
        before = self.data()
        self.ui.handle(Event("down", self.bar("vol", 2, 2), button=1))
        self.ui.handle(Event("move", self.bar("vol", 3, 1)))
        self.assertEqual((self.data(), panel.tables["vol"].table), (before, [15, 5, 2, 1]))
        self.ui.handle(Event("up", self.bar("vol", 3, 1), button=1))
        self.assertEqual((differing(before, self.data()), self.inst()["vol"]),
                         (["instruments.bell.vol"], [15, 5, 2, 1]))
        self.assertTrue(self.session.undo())
        self.assertEqual(self.inst()["vol"], [15, 5, 9, 6])

    def test_a_table_drawn_back_to_the_one_value_the_engine_takes_without_it_is_taken_out_of_the_file(self) -> None:
        click(self.ui, self.bar("arp", 0, 7))
        self.assertEqual(self.inst()["arp"], 7)
        click(self.ui, self.bar("arp", 0, 0))
        self.assertEqual((self.inst(), self.session.undo_label, self.panel.tables["arp"].table),
                         (BELL, "bell: jump table", 0))
        # A table of several steps is written whatever they are, and one of the file's is not taken out for
        # being drawn as it was.
        click(self.ui, self.bar("duty", 0, 2))
        self.assertEqual(self.inst()["duty"], [2, 2])

    def test_a_knob_that_is_dragged_is_saved_when_it_is_let_go_and_not_before(self) -> None:
        panel, before = self.panel, self.data()
        mouse = self.drag(panel.knobs["gain"], 5, let_go=False)
        self.assertEqual((panel.knobs["gain"].value, panel.knobs["gain"].busy), (0.9, True))
        self.assertEqual((self.data(), self.session.can_undo, self.heard()), (before, False, []))
        self.ui.handle(Event("up", mouse, button=1))
        self.assertEqual((differing(before, self.data()), self.inst()["gain"]), (["instruments.bell.gain"], 0.9))
        self.assertEqual((self.session.undo_label, panel.knobs["gain"].value), ("bell: level 0.9", 0.9))

    def test_each_knob_is_saved_at_its_own_place_and_called_in_plain_words(self) -> None:
        panel = self.panel
        saved = {"gain": 0.81, "release": 4, "gate": 0.91, "slide": 0.5, "vib.depth": 0.31, "vib.delay": 13,
                 "vib.rate": 5.5, "res": 0.01, "key_scale": 0.5}
        labels = ["level 0.81", "fade 4", "held for 0.91", "slide 0.5", "vibrato 0.31", "vibrato wait 13",
                  "vibrato rate 5.5", "ring 0.01", "key tilt 0.5"]
        for (key, value), label in zip(saved.items(), labels):
            before = self.data()
            # One step of the wheel is one step of the knob, and it is saved when the run of the wheel has ended.
            self.ui.handle(Event("wheel", middle(panel.knobs[key].rect), wheel=1))
            panel.update()
            self.assertEqual(self.data(), before)
            self.now += WHEEL_WAIT
            panel.update()
            self.assertEqual(differing(before, self.data()), [f"instruments.bell.{key}"])
            held = self.inst()
            for step in key.split("."):
                held = held[step]
            self.assertEqual((held, type(held), self.session.undo_label), (value, type(value), f"bell: {label}"))
        self.assertEqual(self.inst()["vib"], {"delay": 13, "depth": 0.31, "rate": 5.5})

    def test_a_number_put_back_to_what_the_engine_takes_without_it_is_taken_out_of_the_file(self) -> None:
        panel = self.panel
        click(self.ui, middle(panel.knobs["gain"].rect), button=3)
        self.assertEqual(("gain" in self.inst(), self.session.undo_label, panel.knobs["gain"].value),
                         (False, "bell: level 1", 1))
        self.drag(panel.knobs["release"], -2)
        self.assertEqual(("release" in self.inst(), self.session.undo_label, panel.knobs["release"].value),
                         (False, "bell: fade 2", 2))
        # Taken away from there it is written again, and one the file never had is not there to take out.
        self.drag(panel.knobs["release"], 5)
        self.assertEqual((self.inst()["release"], self.session.undo_label), (5, "bell: fade 5"))
        click(self.ui, middle(panel.knobs["slide"].rect), button=3)
        self.assertEqual((self.inst(), self.session.undo_label),
                         ({key: value for key, value in {**BELL, "release": 5}.items() if key != "gain"},
                          "bell: fade 5"))

    def test_a_vibrato_with_no_depth_is_taken_out_and_given_one_again_it_has_what_it_had(self) -> None:
        panel = self.panel
        depth, wait, rate = (panel.knobs[key] for key in ("vib.depth", "vib.delay", "vib.rate"))
        # Its rate put back to the 6 the engine takes is no longer written, and the vibrato stays.
        click(self.ui, middle(rate.rect), button=3)
        self.assertEqual((self.inst()["vib"], self.session.undo_label),
                         ({"delay": 12, "depth": 0.3}, "bell: vibrato rate 6"))
        click(self.ui, middle(depth.rect), button=3)
        self.assertEqual(("vib" in self.inst(), self.session.undo_label), (False, "bell: vibrato 0"))
        self.assertEqual([(knob.value, knob.enabled) for knob in (depth, wait, rate)],
                         [(0, True), (0, False), (6, False)])
        # With no vibrato its two other knobs take no mouse.
        before = self.data()
        self.ui.handle(Event("wheel", middle(wait.rect), wheel=5))
        self.assertEqual((self.data(), wait.value), (before, 0))
        self.drag(depth, 10)
        self.assertEqual((self.inst()["vib"], self.session.undo_label, wait.value, wait.enabled),
                         ({"delay": 12, "depth": 0.2}, "bell: vibrato 0.2", 12, True))
        # One that never had a vibrato gets one of its depth alone.
        self.pick("plain")
        self.drag(depth, 5)
        self.assertEqual(self.inst("plain"), {"vib": {"depth": 0.1}})

    def test_the_short_noise_is_saved_as_it_is_switched_and_off_is_taken_out_of_the_file(self) -> None:
        panel = self.panel
        self.pick("hiss")
        before = self.data()
        click(self.ui, middle(panel.short.rect))
        self.assertEqual((differing(before, self.data()), self.inst("hiss"), self.session.undo_label,
                          panel.short.is_on),
                         (["instruments.hiss.short"], {"wave": "noise", "lp": [4000, 2000]}, "hiss: buzz off", False))
        click(self.ui, middle(panel.short.rect))
        self.assertEqual((self.inst("hiss")["short"], self.session.undo_label, panel.short.is_on),
                         (True, "hiss: buzz on", True))

    def test_a_step_taken_back_puts_the_widget_back(self) -> None:
        panel = self.panel
        self.drag(panel.knobs["gain"], 5)
        click(self.ui, self.bar("vol", 0, 3))
        click(self.ui, middle(panel.wave.cells()[1]))
        self.assertEqual((panel.knobs["gain"].value, panel.tables["vol"].table, panel.wave.picked),
                         (0.9, [3, 12, 9, 6], 1))
        back = []
        for _ in range(3):
            self.assertTrue(self.session.undo())
            panel.update()
            back.append((panel.knobs["gain"].value, panel.tables["vol"].table, panel.wave.picked))
        self.assertEqual(back, [(0.9, [3, 12, 9, 6], 0), (0.9, [15, 12, 9, 6], 0), (0.8, [15, 12, 9, 6], 0)])
        self.assertEqual((self.inst(), self.session.can_undo), (BELL, False))
        self.assertTrue(self.session.redo())
        panel.update()
        self.assertEqual(panel.knobs["gain"].value, 0.9)


class Played(Opened):
    def test_a_key_that_is_pressed_sounds_its_note_in_this_instrument(self) -> None:
        panel = self.panel
        self.assertEqual((panel.keys.low, panel.keys.lit, panel.last, panel.octave.options[panel.octave.picked]),
                         (48, A4, 69, "c3"))
        click(self.ui, self.key(48))
        click(self.ui, self.key(49))
        self.assertEqual((len(self.heard()), panel.keys.lit, panel.last), (2, 49, 49))
        self.assertIs(self.heard()[0], self.tone(48))
        self.assertIs(self.heard()[1], self.tone(49))
        self.assertIsNot(self.tone(49), self.tone(49, "hiss"))
        # The help line names the key under the mouse, and another button plays nothing.
        self.assertEqual(self.ui.help, "c#3: press a key to hear it, and drag along the keys: the keys of the "
                                       "computer play them too")
        click(self.ui, self.key(50), button=3)
        self.assertEqual((len(self.heard()), panel.last), (2, 49))
        # A key that is pressed again is heard again.
        click(self.ui, self.key(49))
        self.assertEqual(len(self.heard()), 3)

    def test_a_drag_along_the_keyboard_plays_each_key_it_comes_onto_once(self) -> None:
        panel = self.panel
        x, y = self.key(48)
        self.ui.handle(Event("down", (x, y), button=1))
        # A move that stays on the key plays nothing more.
        self.ui.handle(Event("move", (x + 2, y - 1)))
        self.assertEqual(len(self.heard()), 1)
        for pitch in (50, 52, 52, 53):
            self.ui.handle(Event("move", self.key(pitch)))
        self.assertEqual((len(self.heard()), panel.keys.lit, panel.last), (4, 53, 53))
        for at, pitch in enumerate((48, 50, 52, 53)):
            self.assertIs(self.heard()[at], self.tone(pitch))
        # Off the keyboard nothing is played, and coming back onto the key it left plays that one again.
        self.ui.handle(Event("move", (x, y - 80)))
        self.ui.handle(Event("move", self.key(53)))
        self.assertEqual(len(self.heard()), 5)
        self.ui.handle(Event("up", self.key(53), button=1))
        # With no button pressed the mouse plays nothing.
        self.ui.handle(Event("move", self.key(55)))
        self.assertEqual((len(self.heard()), panel.last), (5, 53))

    def test_the_stepper_of_the_octave_moves_the_keyboard_up_and_down(self) -> None:
        panel = self.panel
        at = self.key(48)
        x, y, w, _ = panel.octave.rect
        click(self.ui, (x + w - 3, y + 3))
        self.assertEqual((panel.octave.options[panel.octave.picked], panel.keys.low), ("c4", 60))
        click(self.ui, at)
        self.assertIs(self.heard()[-1], self.tone(60))
        self.ui.handle(Event("wheel", (x + 3, y + 3), wheel=-3))
        self.assertEqual((panel.octave.options[panel.octave.picked], panel.keys.low), ("c1", 24))
        click(self.ui, at)
        self.assertIs(self.heard()[-1], self.tone(24))
        # It goes from c0 to c7, where the two octaves end at b8.
        self.assertEqual((panel.octave.options[0], panel.octave.options[-1]), ("c0", "c7"))

    def test_the_keys_of_the_computer_play_as_a_piano_while_a_part_of_the_panel_has_the_focus(self) -> None:
        panel = self.panel
        # With the focus nowhere the key does what it did before the panel was there.
        self.assertTrue(self.ui.handle(Event("key", key="z")))
        self.assertEqual((self.typed, self.heard()), (["z"], []))
        # A press on the ground of the panel gives it the focus: z is the lowest key, and q the c above it.
        click(self.ui, middle(arrange(RECT)["scope"]))
        for key in ("z", "s", "q"):
            self.assertTrue(self.ui.handle(Event("key", key=key)))
        self.assertEqual((self.typed, len(self.heard()), panel.keys.lit, panel.last), (["z"], 3, 60, 60))
        for at, pitch in enumerate((48, 49, 60)):
            self.assertIs(self.heard()[at], self.tone(pitch))
        # So it is with the focus on a knob, and from the octave the keyboard was moved to.
        self.drag(panel.knobs["slide"], 0)
        panel.octave.step(1)
        self.ui.handle(Event("key", key="x"))
        self.assertIs(self.heard()[-1], self.tone(62))
        # With ctrl it is no key of the piano, and a key that plays no note is not used.
        self.assertFalse(self.ui.handle(Event("key", key="z", mods=frozenset({"ctrl"}))))
        self.assertFalse(self.ui.handle(Event("key", key="f1")))
        self.assertEqual((len(self.heard()), self.typed), (4, ["z"]))
        # With another thing picked the panel has nothing to play.
        self.pick("walk", "song")
        self.ui.handle(Event("key", key="z"))
        self.assertEqual((len(self.heard()), self.typed), (4, ["z", "z"]))

    def test_after_a_change_the_note_played_last_is_heard_again_in_the_new_sound_unless_the_session_plays(
            self) -> None:
        panel = self.panel
        was = self.tone(A4)
        # Before any key was played it is a4.
        click(self.ui, middle(panel.knobs["gain"].rect), button=3)
        self.assertEqual(len(self.heard()), 1)
        self.assertIs(self.heard()[0], self.tone(A4))
        self.assertIsNot(self.heard()[0], was)
        click(self.ui, self.key(52))
        was = self.tone(52)
        for change in (lambda: click(self.ui, self.bar("vol", 0, 3)),
                       lambda: click(self.ui, middle(panel.wave.cells()[1])),
                       lambda: self.drag(panel.knobs["release"], 4)):
            change()
            self.assertIs(self.heard()[-1], self.tone(52))
        self.assertEqual((len(self.heard()), self.inst()["release"]), (5, 5))
        self.assertIsNot(self.heard()[-1], was)
        # While the session plays, the change is saved and no note is sounded over what plays.
        self.speaker.playing = True
        self.drag(panel.knobs["release"], 5)
        self.assertEqual((self.inst()["release"], self.session.undo_label, len(self.heard())),
                         (8, "bell: fade 8", 5))


class Outside(Opened):
    def test_a_change_from_outside_is_shown_except_in_the_widget_under_the_hand_which_shows_it_when_let_go(
            self) -> None:
        panel = self.panel
        gain, release, vol = panel.knobs["gain"], panel.knobs["release"], panel.tables["vol"]
        mouse = self.drag(gain, 5, let_go=False)
        self.outside(lambda instruments: instruments["bell"].update(gain=0.5, release=9, vol=[1, 2], wave="saw"))
        panel.update()
        self.assertEqual((gain.value, release.value, vol.table, panel.wave.picked, panel.tables["duty"].enabled),
                         (0.9, 9, [1, 2], 1, False))
        # The release saves the knob beside what was written outside, and the panel shows what the file holds.
        self.ui.handle(Event("up", mouse, button=1))
        self.assertEqual({key: self.inst()[key] for key in ("gain", "release", "vol", "wave")},
                         {"gain": 0.9, "release": 9, "vol": [1, 2], "wave": "saw"})
        self.assertEqual((gain.value, self.session.message), (0.9, ""))
        # A knob that is pressed and let go where it was saves nothing, and then shows what came from outside.
        at = middle(release.rect)
        self.ui.handle(Event("down", at, button=1))
        self.outside(lambda instruments: instruments["bell"].update(release=20, gain=0.4))
        panel.update()
        self.assertEqual((release.value, gain.value), (9, 0.4))
        self.ui.handle(Event("up", at, button=1))
        self.assertEqual((release.value, self.inst()["release"]), (9, 20))
        panel.update()
        self.assertEqual((release.value, self.session.undo_label), (20, "changed outside: bell"))
        # A table keeps the bars the mouse is drawing, and they are what is saved.
        at = self.bar("vol", 0, 9)
        self.ui.handle(Event("down", at, button=1))
        self.outside(lambda instruments: instruments["bell"].update(vol=[3, 3, 3], gain=0.3))
        panel.update()
        self.assertEqual((vol.table, gain.value), ([9, 2], 0.3))
        self.ui.handle(Event("up", at, button=1))
        self.assertEqual((self.inst()["vol"], vol.table, self.session.undo_label), ([9, 2], [9, 2],
                                                                                    "bell: loudness table"))

    def test_a_number_that_went_from_the_file_outside_is_not_there_to_take_out_and_nothing_is_refused(self) -> None:
        gain = self.panel.knobs["gain"]
        # The hand has the level at the 1 the engine takes without it, as another program takes it out of the file.
        mouse = self.drag(gain, 10, let_go=False)
        self.outside(lambda instruments: instruments["bell"].pop("gain"))
        self.panel.update()
        self.assertEqual((gain.value, gain.busy), (1, True))
        self.ui.handle(Event("up", mouse, button=1))
        self.assertEqual((self.inst(), gain.value), ({key: value for key, value in BELL.items() if key != "gain"}, 1))
        # The file is as was asked already: no step is made, and the note is heard in the sound as it is now.
        self.assertEqual((self.session.message, self.session.undo_label, len(self.heard())),
                         ("changed outside: bell", "changed outside: bell", 1))

    def test_a_change_that_is_refused_leaves_the_widgets_showing_the_file_and_the_session_says_why(self) -> None:
        panel = self.panel
        self.pick("plain")
        # Another program has taken the instrument out of the file, and the session has not looked at it yet.
        written = self.outside(lambda instruments: instruments.pop("plain"), look=False)
        refusal = "instruments has no key 'plain'"
        self.drag(panel.knobs["gain"], 5)
        self.assertEqual((panel.knobs["gain"].value, self.session.message), (1, refusal))
        click(self.ui, self.bar("vol", 0, 3))
        self.assertEqual((panel.tables["vol"].table, self.session.message), (15, refusal))
        click(self.ui, middle(panel.wave.cells()[1]))
        self.assertEqual((panel.wave.picked, panel.tables["duty"].enabled, self.session.message), (0, True, refusal))
        self.drag(panel.knobs["vib.depth"], 5)
        self.assertEqual((panel.knobs["vib.depth"].value, panel.knobs["vib.delay"].enabled), (0, False))
        # Nothing was saved, no step was kept, and no note was heard.
        self.assertEqual((self.data(), self.session.can_undo, self.session.messages, self.heard()),
                         (written, False, 4, []))

    def test_another_instrument_picked_while_a_knob_is_dragged_takes_the_panel_and_nothing_is_saved(self) -> None:
        panel, before = self.panel, self.data()
        x, y = self.drag(panel.knobs["gain"], 5, let_go=False)
        self.pick("hiss")
        self.assertEqual((panel.name, panel.knobs["gain"].busy, panel.knobs["gain"].value), ("hiss", False, 1))
        self.ui.handle(Event("move", (x, y - 5)))
        self.ui.handle(Event("up", (x, y - 5), button=1))
        self.assertEqual((self.data(), self.session.can_undo, panel.knobs["gain"].value), (before, False, 1))


class Placed(Opened):
    def test_the_wheel_scrolls_a_page_that_does_not_fit_and_a_part_not_wholly_in_view_is_not_seen(self) -> None:
        panel = self.panel
        every = list(panel._placed)
        # In the middle of the window everything is seen, and the wheel has nothing to do.
        at = middle(arrange(RECT)["scope"])
        self.assertEqual((self.seen(), self.ui.handle(Event("wheel", at, wheel=-1)), panel.scroll), (every, False, 0))
        panel.place(SMALL)
        self.assertEqual((panel.page.rect, panel.scroll), (arrange(SMALL)["box"], 0))
        self.assertEqual(self.seen(), ["word", "wave", "short", "vol", "duty", "arp", "pitch", "lp"])
        # A step of the wheel over a table, which has no use for it, is 24 pixels of the page.
        at = middle(panel.tables["arp"].rect)
        self.assertTrue(self.ui.handle(Event("wheel", at, wheel=-1)))
        self.assertEqual((panel.scroll, panel.tables["arp"].rect, self.seen()),
                         (24, arrange(SMALL, 24)["arp"], ["duty", "arp", "pitch", "lp",
                                                         *(number.key for number in NUMBERS)]))
        self.assertTrue(self.ui.handle(Event("wheel", at, wheel=-3)))
        self.assertEqual((panel.scroll, self.seen()),
                         (61, ["duty", "arp", "pitch", "lp", *(number.key for number in NUMBERS), "octave", "keys"]))
        # At its end the page stays, and the step of the wheel is not used.
        self.assertFalse(self.ui.handle(Event("wheel", at, wheel=-1)))
        self.assertTrue(self.ui.handle(Event("wheel", at, wheel=9)))
        self.assertEqual(panel.scroll, 0)
        # Scrolled to its end and given the room again, the page is where it fits.
        self.ui.handle(Event("wheel", at, wheel=-9))
        panel.place(RECT)
        self.assertEqual((panel.scroll, self.seen()), (0, every))

    def test_at_either_size_every_widget_is_in_view_or_not_seen_no_two_overlap_and_none_is_dropped(self) -> None:
        panel = self.panel
        for name in ("bell", "brass"):
            self.pick(name)
            for rect in (RECT, SMALL):
                panel.place(rect)
                view, ever = arrange(rect)["view"], set()
                for scroll in range(0, slack(rect) + 1):
                    panel.scroll = scroll
                    panel.place(rect)
                    shown = [widget for widget in panel.widgets if widget.visible and widget is not panel.page]
                    ever.update(shown)
                    self.assertTrue(all(within(widget.rect, view) == widget.rect for widget in shown), (rect, scroll))
                    self.assertTrue(all(apart(one.rect, other.rect) for one in shown for other in shown
                                        if one is not other), (rect, scroll))
                # What is not seen at one place of the page is seen at another: the line of an FM instrument
                # stands for the tables, which are not there.
                gone = [panel.later] if name == "bell" else list(panel.tables.values())
                self.assertEqual(ever, set(panel.widgets) - {panel.page, *gone}, (name, rect))


@unittest.skipUnless(pygame, NEEDS)
class Drawn(unittest.TestCase):
    def setUp(self) -> None:
        self.surface = pygame.Surface((960, 540))
        self.surface.fill(theme.BACKGROUND)
        self.canvas = Canvas(self.surface, Font(sprites.load("font")), sprites.load("icons"))
        self.ui = Ui()
        model = Model(EXAMPLES / "arcade.json")
        model.pick("instrument", "chime")
        self.panel = InstPanel(Stand(model), self.ui)
        self.panel.place(RECT)
        self.panel.update()
        self.panel.draw(self.canvas)
        self.ui.draw(self.canvas)

    def ink(self, rect: layout.Rect, colour: theme.Colour) -> int:
        """How many pixels of a rectangle are in a colour."""
        x, y, w, h = rect
        return sum(self.surface.get_at((x + column, y + row))[:3] == colour for column in range(w) for row in range(h))

    def test_the_first_bar_of_the_loudness_of_the_chime_is_as_high_as_the_file_says(self) -> None:
        held = json.loads((EXAMPLES / "arcade.json").read_text(encoding="utf-8"))["instruments"]["chime"]
        self.assertEqual(held["vol"], [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        seen = self.panel.tables["vol"].map
        left, top, _, rows = seen.bars
        # A level of 15 is every row of the field, and each level under it that share of them, to the pixel.
        high = [self.ink((seen.x(step), top, 1, rows), theme.INSTRUMENT) for step in range(15)]
        self.assertEqual((rows, high), (57, [math.floor(level * 57 / 15 + 0.5) for level in held["vol"]]))
        self.assertEqual((high[0], high[7], high[14], seen.x(0)), (57, 30, 4, left))
        # The bar stands on the foot of the field, nine pixels wide, with nothing of its colour beside it.
        self.assertEqual(self.ink((left, top, 10, rows), theme.INSTRUMENT), 9 * 57)

    def test_the_heading_names_the_instrument_in_its_colour_and_the_key_of_the_note_heard_again_is_lit(self) -> None:
        parts = arrange(RECT)
        head = parts["head"]
        wide = Font.width("chime")
        name = (head[0] + 5 + Font.width("INSTRUMENT") + 8, head[1], wide, head[3])
        self.assertEqual((self.panel.name, self.panel.used), ("chime", "in 2 sounds"))
        # All the ink of that colour in the heading is the name, and the words before and after it are dim.
        self.assertGreater(self.ink(name, theme.INSTRUMENT), 0)
        self.assertEqual(self.ink(head, theme.INSTRUMENT), self.ink(name, theme.INSTRUMENT))
        self.assertGreater(self.ink((head[0], head[1], name[0] - head[0], head[3]), theme.DIM), 0)
        self.assertGreater(self.ink((name[0] + wide, head[1], 80, head[3]), theme.DIM), 0)
        # Of the keys only a4 is lit, over its whole key but the line that parts it from the next.
        keys = dict(keyboard(parts["keys"], 48))
        x, y, w, h = keys[A4]
        self.assertEqual(self.ink(parts["keys"], theme.ACTIVE), self.ink((x, y, w - 1, h), theme.ACTIVE))
        self.assertEqual(self.ink((x, y + 20, w - 1, h - 20), theme.ACTIVE), (w - 1) * (h - 20))
        # The room beside the keyboard is left empty for the scope.
        self.assertEqual(self.ink(parts["scope"], theme.PANEL), parts["scope"][2] * parts["scope"][3])


if __name__ == "__main__":
    unittest.main()
