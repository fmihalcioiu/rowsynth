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
from rowsynth.errors import RowSynthError
from rowsynth.gui import fmview, layout, sprites, theme
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.dials import WHEEL_WAIT, Knob
from rowsynth.gui.font import Font
from rowsynth.gui.instpanel import (A4, DIALS, ENVELOPE_HELP, FEEDBACK, FM_TABLES, NO_VIBRATO, NUMBERS, PAGE_HELP,
                                    PLAIN, SCOPE_HELP, SHORT, SHORT_UNUSED, SILENT, TABLES, InstPanel, Row, Waves,
                                    arrange, furthest, heading, heights, key_at, keyboard, note_name, offsets,
                                    reach, role, rows, tops, used, uses)
from rowsynth.gui.kit import Bar, Heading, ScrollBar
from rowsynth.gui.model import Model
from rowsynth.gui.panels import within
from rowsynth.gui.session import POLL_EVERY, Session
from rowsynth.gui.tables import MOST, SIZE_HELP
from rowsynth.gui.ui import Event, Label, Ui, Widget
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

EXAMPLES = Path(__file__).resolve().parent / "data"
NEEDS = "pygame is not installed"

#: The middle of the art surface the layout is drawn for, and the least the panel is laid out in.
RECT = (124, 22, 662, 504)
SMALL = (124, 22, 500, 300)

#: A pulse with every kind of table and a vibrato, a short noise with a filter, an FM voice that names two of its
#: four operators, and one with no key.
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

#: What each setting of an instrument reads on the window: the name of the widget that shows it, which for a
#: table and for the wirings stands beside it in the side column, and for an envelope the kind of picture it is.
#: An operator has three knobs and its envelope.
WORDS = {
    "wave": ["PULSE SAW TRI NOISE FM"], "vol": ["VOLUME SHAPE"], "duty": ["PULSE WIDTH"], "arp": ["ARPEGGIO"],
    "pitch": ["PITCH"], "vib": ["VIBRATO DEPTH", "DELAY", "RATE"], "slide": ["SLIDE"], "release": ["RELEASE"],
    "gain": ["VOLUME"], "short": ["SHORT NOISE"], "lp": ["CUTOFF"], "res": ["RESONANCE"],
    "key_scale": ["KEY TRACKING"], "gate": ["GATE"], "alg": ["WIRING"], "fb": ["FEEDBACK"],
    "ops": ["RATIO", "DETUNE", "VOLUME", "Envelope"] * 4,
}
#: The settings that are the FM wave's. The parts of `arrange` that every editor has, those that stand in the
#: foot, and those that are a frame or a room and have no widget of their own.
FM = ("alg", "fb", "ops")
SKELETON = ("head", "bar", "main", "side", "body", "foot")
FOOT = ("try", "octave", "keys", "scope", "spectrum")
FRAME = (*SKELETON, "view", "ruler", *(f"ops[{op}]" for op in range(4)))
#: The room the status bar has for a line of help in the smallest window, with the longest pick at its right end.
ROOM = theme.MIN_ART[0] - 2 * theme.BAR_MARGIN - Font.width("INSTRUMENT fmdrive") - theme.GROUP_GAP
SHIFT = frozenset({"shift"})
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


def placed(rect: layout.Rect, first: int = 0, fm: bool = False) -> list[str]:
    """The parts of a panel that a widget stands in, by what `arrange` calls them: with `fm`, of an FM instrument."""
    return [name for name in arrange(rect, first, fm) if name not in FRAME]


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

    def test_the_panel_has_the_skeleton_every_editor_has_and_in_the_middle_of_the_window_every_table_fits(self) -> None:
        parts = arrange(RECT)
        self.assertEqual([parts[name] for name in ("head", "bar", "side", "body", "foot")],
                         [(128, 26, 654, 17), (128, 45, 654, 40), (128, 87, 124, 361), (254, 87, 528, 361),
                          (128, 450, 654, 72)])
        self.assertEqual((parts["head"][3], parts["bar"][3], parts["side"][2], parts["foot"][3]),
                         (theme.HEAD_H, theme.BAR_H, theme.SIDE_W, theme.FOOT_H))
        # In the body the ruler is along the top and the scroll bar down the right, and the rows have the rest.
        self.assertEqual((parts["ruler"], parts["view"], parts["scroll"], reach(RECT)),
                         ((254, 87, 523, 10), (254, 100, 523, 348), (778, 100, 4, 348), 0))
        # Five tables one under the other, each as wide as the view, and sharing its height.
        self.assertEqual([parts[table.key] for table in TABLES],
                         [(254, 100, 523, 67), (254, 170, 523, 67), (254, 240, 523, 68), (254, 311, 523, 67),
                          (254, 381, 523, 67)])
        # Beside each, in the side column, is its name, and under that its facts.
        self.assertEqual([parts[f"{table.key}.name"] for table in TABLES],
                         [(133, top, 114, 11) for top in (103, 173, 243, 314, 384)])
        self.assertEqual((parts["vol.facts"], parts["lp.facts"]), ((133, 114, 114, 11), (133, 395, 114, 11)))
        # In the foot the word is over the stepper of the octave. Then come the keyboard, the scope and the
        # spectrum, whose box is as high as the scope and has the names of its marks under it.
        self.assertEqual([parts[name] for name in FOOT],
                         [(134, 458, 36, 11), (134, 475, 36, 13), (176, 458, 196, 56), (378, 458, 157, 47),
                          (541, 458, 235, 56)])
        names = placed(RECT)
        for name in names:
            room = parts["foot" if name in FOOT else "side" if "." in name else "body"]
            self.assertEqual(within(parts[name], room), parts[name], name)
        self.assertTrue(all(apart(parts[one], parts[other]) for one in names for other in names if one != other))
        # Nothing of the FM wave is there.
        self.assertEqual([name for name in names if name in FM or name.startswith("ops")], [])
        # There is no row to go along to: asked for or not, the view stays.
        self.assertEqual(arrange(RECT, 3), parts)

    def test_the_rows_of_an_fm_instrument_are_the_wirings_the_operators_and_the_tables_it_reads(self) -> None:
        self.assertEqual([(row.key, row.least, row.grows) for row in rows(True)],
                         [("alg", 62, False), ("ops[0]", 45, True), ("ops[1]", 45, True), ("ops[2]", 45, True),
                          ("ops[3]", 45, True), ("ruler", 10, False), ("vol", 36, True), ("arp", 36, True),
                          ("pitch", 36, True), ("lp", 36, True)])
        # The width of a pulse is none of the tables an FM voice reads.
        self.assertEqual([table.key for table in FM_TABLES], ["vol", "arp", "pitch", "lp"])
        parts, plain = arrange(RECT, fm=True), arrange(RECT)
        # The skeleton is the same. The ruler is a row here, so the rows have the body from its top.
        self.assertEqual(([parts[name] for name in SKELETON], parts["view"], reach(RECT, True)),
                         ([plain[name] for name in SKELETON], (254, 87, 523, 361), 1))
        # The eight wirings share the first row, with nothing between two.
        self.assertEqual((parts["alg"], [cell[2] for cell in fmview.cells(parts["alg"], 0)]),
                         ((254, 87, 523, 62), [65, 66, 65, 66, 65, 65, 66, 65]))
        self.assertEqual((parts["alg.name"], parts["alg.facts"]), ((133, 90, 114, 11), (133, 101, 114, 11)))
        # Each operator has a row: its three knobs, and after them its envelope. In the side column are its
        # number and under that what it does.
        self.assertEqual([parts[f"ops[{op}]"] for op in range(4)],
                         [(254, top, 523, 45) for top in (152, 200, 248, 296)])
        self.assertEqual([parts[f"ops[3]{part}"] for part in (".name", ".facts", ".mul", ".det", ".level", ".env")],
                         [(133, 299, 114, 11), (133, 310, 114, 11), (254, 300, 37, 37), (295, 300, 37, 37),
                          (336, 300, 37, 37), (379, 296, 398, 45)])
        # Under the operators the ruler stands over the first table. Two tables are in view, and one row
        # further all four are.
        self.assertEqual((parts["ruler"], parts["vol"], parts["arp"], "pitch" in parts, "duty" in parts),
                         ((254, 344, 523, 10), (254, 357, 523, 36), (254, 396, 523, 36), False, False))
        further = arrange(RECT, 1, True)
        self.assertEqual(("alg" in further, further["ops[0]"], further["ruler"],
                          [further[table.key] for table in FM_TABLES]),
                         (False, (254, 87, 523, 45), (254, 279, 523, 10),
                          [(254, top, 523, 36) for top in (292, 331, 370, 409)]))
        self.assertEqual(arrange(RECT, 9, True), further)
        # The foot is where the other waves have it.
        self.assertEqual([parts[name] for name in FOOT], [plain[name] for name in FOOT])
        for first in (0, 1):
            at, names = arrange(RECT, first, True), placed(RECT, first, True)
            # The bar, the wirings or two more tables, each with its two lines, four operators and the foot.
            self.assertEqual((len(names), "duty" in names), ((39, 42)[first], False))
            self.assertTrue(all(apart(at[one], at[other]) for one in names for other in names if one != other))

    def test_what_an_operator_does_in_a_wiring_is_said_as_the_side_column_says_it_under_its_number(self) -> None:
        self.assertEqual([[role(wiring, op) for op in range(4)] for wiring in (1, 4, 5, 7)],
                         [["CHANGES 3", "CHANGES 3", "CHANGES 4", "HEARD"],
                          ["CHANGES 2", "HEARD", "CHANGES 4", "HEARD"],
                          ["CHANGES 2, 3, 4", "HEARD", "HEARD", "HEARD"], ["HEARD"] * 4])
        # The widest of them has its room in the side column.
        widest = max(Font.width(role(wiring, op)) for wiring in range(8) for op in range(4))
        self.assertEqual((widest, arrange(RECT, fm=True)["ops[0].facts"][2]), (89, 114))

    def test_rows_that_do_not_fit_keep_their_least_height_and_the_view_begins_no_further_than_they_go(self) -> None:
        found = (Row("a", 30, False), Row("b", 20), Row("c", 20))
        # With room to spare the rows that grow share it, and with none each is as high as it is at the least.
        self.assertEqual([heights(found, room) for room in (86, 76, 40, 0)],
                         [[30, 25, 25], [30, 20, 20], [30, 20, 20], [30, 20, 20]])
        tall = [30, 20, 20]
        self.assertEqual(offsets(tall), [0, 33, 56])
        # The view begins no further than at the first row from which all the rest are in it: at the last one
        # when not even that one fits.
        self.assertEqual([furthest(tall, room) for room in (76, 75, 43, 42, 20, 5)], [0, 1, 1, 2, 2, 2])
        # A row is in the view when all of it is, from the row the view begins at.
        self.assertEqual([tops(tall, first, 60) for first in (0, 1, 2)],
                         [[0, 33, None], [None, 0, 23], [None, None, 0]])
        self.assertEqual((tops(tall, 0, 40), tops(tall, 0, 29)), ([0, None, None], [None, None, None]))
        # In a low panel three of the five tables are in view, each at its least height.
        parts = arrange(SMALL)
        self.assertEqual((parts["view"], reach(SMALL), reach(SMALL, True)), ((254, 100, 361, 144), 2, 6))
        self.assertEqual([[table.key for table in TABLES if table.key in arrange(SMALL, first)] for first in range(3)],
                         [["vol", "duty", "arp"], ["duty", "arp", "pitch"], ["arp", "pitch", "lp"]])
        last = arrange(SMALL, 2)
        self.assertEqual((last["arp"], last["lp"], last["lp.name"], arrange(SMALL, 9), arrange(SMALL, -4)),
                         ((254, 100, 361, 36), (254, 178, 361, 36), (133, 181, 114, 11), last, parts))
        # The foot is as high as ever, and what stands in it is as wide as a narrow panel leaves it.
        self.assertEqual([parts[name] for name in ("foot", "keys", "scope", "spectrum")],
                         [(128, 246, 492, 72), (176, 254, 196, 56), (378, 254, 92, 47), (476, 254, 138, 56)])
        # The rows of an FM instrument go further: at their end the four tables are in view, and no ruler.
        last = arrange(SMALL, 6, True)
        self.assertEqual(([table.key for table in FM_TABLES if table.key in last], "ruler" in last, last["lp"]),
                         (["vol", "arp", "pitch", "lp"], False, (254, 204, 361, 36)))
        for fm in (False, True):
            for first in range(reach(SMALL, fm) + 1):
                at, names = arrange(SMALL, first, fm), placed(SMALL, first, fm)
                self.assertTrue(all(apart(at[one], at[other]) for one in names for other in names if one != other))
        # A panel with no room at all has no part with a size below nothing.
        self.assertTrue(all(w >= 0 and h >= 0 for fm in (False, True)
                            for _, _, w, h in arrange((0, 0, 0, 0), fm=fm).values()))


class OwnSums(unittest.TestCase):
    def test_an_own_instrument_is_counted_in_its_song_alone_and_the_heading_names_the_song_after_it(self) -> None:
        held = Project.from_dict(PROJECT)
        # One channel has the song's own for its instrument and another switches to it. Its noise is played by
        # a channel and a drum letter, and by no sound: those play the project's of that name.
        self.assertEqual(uses(held, "own", "walk"), {"channels": {"walk": 1}, "switches": {"walk": 1}, "drums": {},
                                                     "sounds": {}})
        self.assertEqual(used(uses(held, "hiss", "walk")), "in 1 channel of walk, 1 drum of walk")
        # The heading has the name, and after it the facts: for an own instrument its song comes first in those,
        # and is not named again in the count.
        self.assertEqual((heading("own", "walk", used(uses(held, "own", "walk"), False)),
                          heading("bell", None, "used nowhere")),
                         (("own", "of walk  in 2 channels"), ("bell", "used nowhere")))


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
        """The parts of the panel whose widgets are seen, by what `arrange` calls them."""
        return [part for part, widget in self.panel._placed.items() if widget.visible]

    def rows(self) -> list[str]:
        """The rows of the main area that are seen, from the top: the wirings, an operator by its envelope, a table."""
        panel = self.panel
        found = {"alg": panel.wiring, **{f"ops[{op}]": column["env"] for op, column in enumerate(panel.operators)},
                 **panel.tables}
        return [key for key, widget in found.items() if widget.visible]

    def cell(self, algorithm: int) -> tuple[int, int]:
        """The pixel in the middle of the picture of an algorithm."""
        return middle(fmview.cells(self.panel.wiring.rect, self.panel.wiring.between)[algorithm])

    def said(self, pos: tuple[int, int]) -> str:
        """What the status bar says with the mouse rested at a position."""
        self.ui.handle(Event("move", pos))
        return self.ui.help


class Shown(Opened):
    def test_every_setting_the_validator_of_an_instrument_knows_has_a_widget_that_is_seen(self) -> None:
        panel = self.panel
        with mock.patch.object(validator, "_record") as record:
            validator._instrument({}, "instruments.any")
        known = list(record.call_args.args[2])
        # A setting the engine gets later is in the validator's table, and fails here until a widget shows it.
        self.assertEqual((len(known), sorted(panel.settings)), (17, sorted(known)))

        def words(name: str, widget: Widget) -> str:
            if isinstance(widget, Waves):
                return " ".join(widget.options)
            if name in panel.titles:
                return panel.titles[name].text
            # The switch reads how it stands, and the settings bar has its name over it.
            return SHORT if widget is panel.short else getattr(widget, "text", type(widget).__name__)

        self.assertEqual({name: [words(name, widget) for widget in panel.settings[name]] for name in known}, WORDS)
        mine = [widget for name in known if name not in FM for widget in panel.settings[name]]
        # A choice, a switch, five tables and nine knobs, each with a line of help.
        self.assertEqual(len(mine), 16)
        self.assertTrue(all(widget in panel.widgets and widget.visible and widget.help for widget in mine))
        # The three of the FM wave have the wirings, a knob, and four operators of three knobs and an envelope
        # each. They are seen with an FM instrument, and not with another.
        theirs = [widget for name in FM for widget in panel.settings[name]]
        self.assertEqual((len(theirs), [widget for widget in theirs if widget.visible]), (18, []))
        self.pick("brass")
        self.assertTrue(all(widget in panel.widgets and widget.visible and widget.help for widget in theirs))

    def test_every_setting_the_validator_of_an_operator_knows_has_a_widget_in_the_column_of_each_operator(
            self) -> None:
        panel = self.panel
        with mock.patch.object(validator, "_record") as record:
            validator._ops([{}], "instruments.any.ops")
            known = record.call_args.args[2]
            known["env"]({}, "instruments.any.ops[0].env")
            parts = list(record.call_args.args[2])
        # A setting an operator gets later is in the validator's table, and fails here until a widget shows it.
        self.assertEqual((list(known), parts), (["mul", "det", "level", "env"], ["a", "d", "s", "sd", "r"]))
        self.assertEqual(([sorted(column) for column in panel.operators], [number.key for number in DIALS]),
                         ([sorted(known)] * 4, ["mul", "det", "level"]))
        # Every part of an envelope is set by a point of its line, to the side or up and down.
        self.assertEqual({key for point in fmview.POINTS for key in (point.time, point.level) if key}, set(parts))
        self.assertEqual({name: [widget for column in panel.operators for widget in column.values()]
                          for name in ("ops",)}, {"ops": list(panel.settings["ops"])})
        # The knobs are called in plain words, and each says what it does to the sound.
        self.assertEqual([(number.text, number.words, bool(number.help)) for number in (FEEDBACK, *DIALS)],
                         [("FEEDBACK", "feedback", True), ("RATIO", "ratio", True),
                          ("DETUNE", "detune", True), ("VOLUME", "volume", True)])

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
                         [(0, 15, "volume"), (0, 3, "pulse width"), (-12, 12, "arpeggio"), (-12, 12, "pitch"),
                          (0, 8000, "cutoff")])
        self.assertEqual([(knob.low, knob.high, knob.steps, knob.default) for knob in panel.knobs.values()],
                         [(0, 2, 200, 1), (0, 60, 60, 2), (0.05, 1, 95, 0.9), (-24, 24, 96, 0), (0, 2, 200, 0),
                          (0, 60, 60, 0), (1, 20, 38, 6), (0, 0.98, 98, 0), (-12, 12, 48, 0)])
        self.assertEqual([knob.colour for knob in panel.knobs.values()] + [table.colour for table in
                                                                          panel.tables.values()],
                         [theme.INSTRUMENT] * 14)

    def test_a_table_or_a_switch_the_wave_does_not_use_is_dim_takes_no_mouse_and_says_why(self) -> None:
        panel, duty = self.panel, self.panel.tables["duty"]
        # Under the pulse every table is used, and the short noise is not.
        self.assertEqual(([table.enabled for table in panel.tables.values()], panel.short.enabled, panel.short.help),
                         ([True] * 5, False, SHORT_UNUSED))
        self.assertEqual((duty.help, panel.titles["duty"].colour),
                         (f"{TABLES[1].help}   wheel: along the tables", theme.TEXT))
        self.pick("hiss")
        self.assertEqual(([table.enabled for table in panel.tables.values()], panel.short.enabled, panel.short.is_on,
                          panel.short.text), ([True, False, True, True, True], True, True, "ON"))
        self.assertEqual((TABLES[1].unused, panel.titles["duty"].colour),
                         ("only the pulse wave has a pulse width: with another wave this table does nothing",
                          theme.DIM))
        # The reason is the line the status bar shows with the mouse on the dim table, and a press draws nothing.
        before = self.data()
        at = self.bar("duty", 0, 3)
        click(self.ui, at)
        self.assertEqual((self.ui.help, duty.table, self.data(), self.session.can_undo),
                         (f"{TABLES[1].unused}   wheel: along the tables", 2, before, False))
        # The two other knobs of the vibrato wait for it to have a depth, and say so.
        self.assertEqual([panel.knobs[key].enabled for key in ("vib.depth", "vib.delay", "vib.rate")],
                         [True, False, False])
        self.assertEqual((self.said(middle(panel.knobs["vib.rate"].rect)), NO_VIBRATO),
                         ("there is no vibrato to set: turn VIBRATO DEPTH up first",) * 2)

    def test_an_fm_instrument_shows_its_wirings_its_feedback_and_a_row_for_each_of_its_four_operators(self) -> None:
        panel = self.panel
        self.pick("brass")
        # All that the rows in view have is seen, and nothing that is not in them.
        self.assertEqual((sorted(self.seen()), panel.tables["duty"].visible, self.rows()),
                         (sorted(placed(RECT, fm=True)), False,
                          ["alg", "ops[0]", "ops[1]", "ops[2]", "ops[3]", "vol", "arp"]))
        self.assertEqual((panel.wave.waves[panel.wave.picked], panel.knobs["gate"].value), ("fm", 0.7))
        self.assertEqual((panel.wiring.algorithm, panel.wiring.rect, panel.feedback.value, panel.feedback.text),
                         (5, arrange(RECT, fm=True)["alg"], 0.25, "FEEDBACK"))
        # The words that stand alone: the names of the rows and their facts in the side column, the operators
        # by the numbers 1 to 4, and the word over the keyboard.
        self.assertEqual([widget.text for widget in panel.widgets if isinstance(widget, Label) and widget.visible],
                         ["VOLUME SHAPE", "ARPEGGIO", "WIRING", "2 steps, 0.03 s", "1 step, 0.02 s", "algorithm 5",
                          "OPERATOR 1", "OPERATOR 2", "OPERATOR 3", "OPERATOR 4",
                          "CHANGES 2, 3, 4", "HEARD", "HEARD", "HEARD", "TRY IT"])
        # Two operators are as the file has them, with the engine's own envelope. The two it leaves out are shown
        # as the engine takes them: at no level, and down at once.
        self.assertEqual([column["env"].op for column in panel.operators], [*BRASS["ops"], SILENT, SILENT])
        self.assertEqual([[column[key].value for key in ("mul", "det", "level")] for column in panel.operators],
                         [[1, 0, 1], [2, 0, 0.5], [1, 0, 0], [1, 0, 0]])
        self.assertEqual((SILENT, fmview.settings(SILENT)["s"]), ({"level": 0, "env": {"s": 0, "d": 0.001}}, 0))
        # The knobs of an operator are named, and so is the one of the feedback. Their ranges, each with the
        # value a right press puts it back to, and their colour.
        dials = [panel.feedback, *panel.operators[2].values()][:4]
        self.assertEqual([dial.text for dial in dials], ["FEEDBACK", "RATIO", "DETUNE", "VOLUME"])
        self.assertEqual([(dial.low, dial.high, dial.steps, dial.default, dial.colour) for dial in dials],
                         [(0, 3, 60, 0, theme.INSTRUMENT), (0, 16, 64, 1, theme.INSTRUMENT),
                          (-10, 10, 200, 0, theme.INSTRUMENT), (0, 8, 160, 1, theme.INSTRUMENT)])
        # The help line says what a press does on the point of an envelope the mouse is on, and on a picture.
        at = panel.operators[0]["env"].map.spots(fmview.settings(BRASS["ops"][0]))[fmview.RELEASE]
        self.assertEqual(self.said(at), "press: drag the release: 0.2 s   wheel: along the operators")
        self.assertEqual(self.said(self.cell(7)), "press: pick algorithm 7: none changes another; 1, 2, 3 and 4 are "
                                                  "heard   wheel: along the operators")

    def test_another_wave_picked_for_an_fm_instrument_and_fm_picked_again_lose_nothing_of_the_file(self) -> None:
        panel = self.panel
        self.pick("brass")
        # Another wave is that one place in the file: the settings of the FM wave stay, to come back to.
        before = self.data()
        click(self.ui, middle(panel.wave.cells()[1]))
        self.assertEqual((differing(before, self.data()), self.inst("brass")),
                         (["instruments.brass.wave"], {**BRASS, "wave": "saw"}))
        self.assertEqual((self.session.undo_label, sorted(self.seen())), ("brass: wave saw", sorted(placed(RECT))))
        self.assertEqual((panel.tables["vol"].table, panel.tables["lp"].table), ([15, 10], 2800))
        click(self.ui, middle(panel.wave.cells()[4]))
        self.assertEqual((self.inst("brass"), self.session.undo_label, sorted(self.seen())),
                         (BRASS, "brass: wave fm", sorted(placed(RECT, fm=True))))
        self.assertEqual((panel.wiring.algorithm, panel.operators[1]["mul"].value), (5, 2))

    def test_the_fm_wave_picked_for_an_instrument_with_no_operators_gives_it_some_so_that_it_is_heard(self) -> None:
        panel = self.panel
        # With no operators an FM voice is silence.
        self.outside(lambda instruments: instruments.update(mute={"wave": "fm"}))
        self.assertFalse(self.tone(A4, "mute")[0].any())
        self.pick("plain")
        click(self.ui, middle(panel.wave.cells()[4]))
        given = {"wave": "fm", "ops": [{}, {}, {"level": 0}, {"level": 0}]}
        self.assertEqual((self.inst("plain"), [dict(op) for op in PLAIN], self.session.undo_label),
                         (given, given["ops"], "plain: wave fm"))
        # It is heard at once, and what is heard is a sound.
        self.assertIs(self.heard()[-1], self.tone(A4, "plain"))
        self.assertGreater(float(abs(self.tone(A4, "plain")[0]).max()), 0.5)
        # The wiring is the 4 the engine takes for an instrument that names none: operator 0 changes operator
        # 1, which is heard, and the other pair is there at no level.
        self.assertEqual((panel.wiring.algorithm, [does.text for does in panel.roles],
                          [column["level"].value for column in panel.operators], "alg" in self.inst("plain")),
                         (4, ["CHANGES 2", "HEARD", "CHANGES 4", "HEARD"], [1, 1, 0, 0], False))
        # The wave and the operators are one step.
        self.assertTrue(self.session.undo())
        self.assertEqual((self.inst("plain"), self.session.undo_label), ({}, "changed outside: mute"))
        self.assertTrue(self.session.redo())
        panel.update()
        # Switched to another wave and back, the operators are in the file all the while.
        click(self.ui, middle(panel.wave.cells()[2]))
        self.assertEqual(self.inst("plain"), {**given, "wave": "tri"})
        before = self.data()
        click(self.ui, middle(panel.wave.cells()[4]))
        self.assertEqual((differing(before, self.data()), self.inst("plain")), (["instruments.plain.wave"], given))

    def test_with_a_song_a_sound_or_nothing_picked_and_off_the_window_the_panel_shows_nothing(self) -> None:
        panel = self.panel
        shown = [widget for widget in panel.widgets if widget.visible]
        # The ground, the heading with the waves, the settings bar with its knobs and its switch, and what has
        # a part of the main area or of the foot.
        self.assertEqual(shown, [widget for widget in panel.widgets
                                 if widget in (panel.page, panel.heading, panel.wave, panel.frame, panel.short,
                                               *panel.knobs.values(),
                                               *(panel._placed[part] for part in placed(RECT)))])
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
        drawn = (("vol", "vol", self.bar("vol", 1, 5), [15, 5, 9, 6], "bell: volume shape table"),
                 ("duty", "duty", self.bar("duty", 0, 3), [3, 2], "bell: pulse width table"),
                 ("arp", "arp", self.bar("arp", 0, 7), 7, "bell: arpeggio table"),
                 ("pitch", "pitch.v", self.bar("pitch", 0, -4), {"v": [-4, -1], "loop": 1}, "bell: pitch table"),
                 ("lp", "lp", (low[0], low[1] - 30), panel.tables["lp"].map.value_at(low[1] - 30),
                  "bell: cutoff table"))
        for key, place, at, table, label in drawn:
            before = self.data()
            self.ui.handle(Event("down", at, button=1))
            self.assertEqual((self.data(), panel.tables[key].table), (before, table))
            self.ui.handle(Event("up", at, button=1))
            # Nothing but the table differs: of the one with a loop, its steps alone.
            self.assertEqual((differing(before, self.data()), self.inst()[key]), ([f"instruments.bell.{place}"], table))
            self.assertEqual((panel.tables[key].table, self.session.undo_label), (table, label))
        # Thirty of the 58 rows of the cutoff, whose top is 8000 Hz, are a whole number of hertz.
        self.assertEqual((panel.tables["lp"].map.bars[3], self.inst()["lp"], type(self.inst()["lp"])),
                         (58, 4138, int))
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
                         (BELL, "bell: arpeggio table", 0))
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
        self.assertEqual((self.session.undo_label, panel.knobs["gain"].value), ("bell: volume 0.9", 0.9))

    def test_each_knob_is_saved_at_its_own_place_and_called_in_plain_words(self) -> None:
        panel = self.panel
        saved = {"gain": 0.81, "release": 4, "gate": 0.91, "slide": 0.5, "vib.depth": 0.31, "vib.delay": 13,
                 "vib.rate": 5.5, "res": 0.01, "key_scale": 0.5}
        labels = ["volume 0.81", "release 4", "gate 0.91", "slide 0.5", "vibrato depth 0.31", "vibrato delay 13",
                  "vibrato rate 5.5", "resonance 0.01", "key tracking 0.5"]
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
                         (False, "bell: volume 1", 1))
        self.drag(panel.knobs["release"], -2)
        self.assertEqual(("release" in self.inst(), self.session.undo_label, panel.knobs["release"].value),
                         (False, "bell: release 2", 2))
        # Taken away from there it is written again, and one the file never had is not there to take out.
        self.drag(panel.knobs["release"], 5)
        self.assertEqual((self.inst()["release"], self.session.undo_label), (5, "bell: release 5"))
        click(self.ui, middle(panel.knobs["slide"].rect), button=3)
        self.assertEqual((self.inst(), self.session.undo_label),
                         ({key: value for key, value in {**BELL, "release": 5}.items() if key != "gain"},
                          "bell: release 5"))

    def test_a_vibrato_with_no_depth_is_taken_out_and_given_one_again_it_has_what_it_had(self) -> None:
        panel = self.panel
        depth, wait, rate = (panel.knobs[key] for key in ("vib.depth", "vib.delay", "vib.rate"))
        # Its rate put back to the 6 the engine takes is no longer written, and the vibrato stays.
        click(self.ui, middle(rate.rect), button=3)
        self.assertEqual((self.inst()["vib"], self.session.undo_label),
                         ({"delay": 12, "depth": 0.3}, "bell: vibrato rate 6"))
        click(self.ui, middle(depth.rect), button=3)
        self.assertEqual(("vib" in self.inst(), self.session.undo_label), (False, "bell: vibrato depth 0"))
        self.assertEqual([(knob.value, knob.enabled) for knob in (depth, wait, rate)],
                         [(0, True), (0, False), (6, False)])
        # With no vibrato its two other knobs take no mouse.
        before = self.data()
        self.ui.handle(Event("wheel", middle(wait.rect), wheel=5))
        self.assertEqual((self.data(), wait.value), (before, 0))
        self.drag(depth, 10)
        self.assertEqual((self.inst()["vib"], self.session.undo_label, wait.value, wait.enabled),
                         ({"delay": 12, "depth": 0.2}, "bell: vibrato depth 0.2", 12, True))
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
                         (["instruments.hiss.short"], {"wave": "noise", "lp": [4000, 2000]},
                          "hiss: short noise off", False))
        # The switch reads how it stands.
        self.assertEqual(panel.short.text, "OFF")
        click(self.ui, middle(panel.short.rect))
        self.assertEqual((self.inst("hiss")["short"], self.session.undo_label, panel.short.is_on, panel.short.text),
                         (True, "hiss: short noise on", True, "ON"))

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
        self.assertEqual(self.ui.help, "press: hear c#3, and a drag plays each key it comes onto")
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

    def test_a_press_anywhere_on_the_editor_gives_it_the_keys_and_a_press_off_it_takes_them_away(self) -> None:
        # On the heading, the settings bar, the side column and the foot, also where no control stands.
        for count, part in enumerate(("head", "bar", "side", "foot"), 1):
            x, y, _, _ = arrange(RECT)[part]
            click(self.ui, (x + 3, y + 3))
            self.ui.handle(Event("key", key="z"))
            click(self.ui, (5, 5))
            self.ui.handle(Event("key", key="z"))
            self.assertEqual((len(self.heard()), len(self.typed)), (count, count), part)
        self.assertIs(self.heard()[-1], self.tone(48))

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
                         (8, "bell: release 8", 5))


class Owned(Opened):
    """The panel with an instrument of the song's own picked: by the name of its song, a dot and its name there."""

    def own(self, name: str = "own") -> dict:
        """What the file holds now for an instrument of the song's own."""
        return self.data()["songs"]["walk"]["instruments"][name]

    def test_an_own_instrument_is_shown_as_one_of_the_project_s_is_with_its_song_beside_its_name(self) -> None:
        panel = self.panel
        self.pick("walk.own")
        self.assertEqual((panel.name, panel.song, panel.used, panel.wave.picked),
                         ("own", "walk", "in 2 channels of walk", 1))
        self.assertTrue(all(widget.visible for widget in (panel.wave, panel.keys, *panel.knobs.values())))
        # The noise the song calls hiss is the one shown, and not the project's of that name, which is a short one.
        self.pick("walk.hiss")
        self.assertEqual((panel.name, panel.song, panel.used, panel.wave.picked, panel.short.is_on),
                         ("hiss", "walk", "in 1 channel of walk, 1 drum of walk", 3, False))
        self.pick("hiss")
        self.assertEqual((panel.name, panel.song, panel.used, panel.short.is_on), ("hiss", None, "in 3 sounds", True))

    def test_a_change_to_an_own_instrument_is_saved_at_its_place_in_the_song_and_it_stays_the_picked_one(
            self) -> None:
        panel = self.panel
        self.pick("walk.own")
        before = self.data()
        click(self.ui, middle(panel.wave.cells()[2]))
        self.assertEqual((differing(before, self.data()), self.own(), self.session.undo_label),
                         (["songs.walk.instruments.own.wave"], {"wave": "tri"}, "own: wave tri"))
        self.assertEqual((self.session.model.picked, panel.name, panel.song, panel.wave.picked),
                         (("instrument", "walk.own"), "own", "walk", 2))
        # A knob that is turned is written, and put back to what the engine takes without it, it is taken out.
        self.drag(panel.knobs["release"], 4)
        self.assertEqual((list(self.own()), self.own()["release"] > 2), (["wave", "release"], True))
        click(self.ui, middle(panel.knobs["release"].rect), button=3)
        self.assertEqual((self.own(), self.session.undo_label), ({"wave": "tri"}, "own: release 2"))
        # The FM wave comes with its operators, written as the whole of the instrument at its place.
        click(self.ui, middle(panel.wave.cells()[4]))
        self.assertEqual((self.own(), self.inst(), panel.wiring.visible),
                         ({"wave": "fm", "ops": [dict(op) for op in PLAIN]}, BELL, True))
        self.assertTrue(self.session.undo())
        self.assertEqual((self.own(), self.session.model.picked), ({"wave": "tri"}, ("instrument", "walk.own")))

    def test_the_keys_and_the_scope_have_the_note_of_an_own_instrument_as_its_song_has_it(self) -> None:
        panel = self.panel
        self.pick("walk.hiss")
        with mock.patch.object(panel.scope, "show", wraps=panel.scope.show) as scope:
            click(self.ui, self.key(48))
            mine = self.session.model.tone("hiss", 48, song="walk")["sides"]
            self.assertIs(self.heard()[0], mine)
            self.assertIs(scope.call_args.args[0], mine[0])
            self.assertIsNot(mine, self.tone(48, "hiss"))
            # After a change the note played last is heard again, in the song's instrument as it is now.
            click(self.ui, middle(panel.short.rect))
            now = self.session.model.tone("hiss", 48, song="walk")["sides"]
            self.assertEqual((self.own("hiss"), len(self.heard())), ({"wave": "noise", "short": True}, 2))
            self.assertIs(self.heard()[1], now)
            self.assertIs(scope.call_args.args[0], now[0])
            self.assertIsNot(now, mine)


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
                                                                                    "bell: volume shape table"))

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


class Wired(Opened):
    """The panel with the FM instrument picked: the file has two of its operators, and leaves two out."""

    def setUp(self) -> None:
        super().setUp()
        self.pick("brass")

    def ops(self) -> list:
        """What the file holds now for the operators of the FM instrument."""
        return self.inst("brass")["ops"]

    def env(self, op: int) -> fmview.Envelope:
        """The envelope of an operator."""
        return self.panel.operators[op]["env"]

    def spot(self, op: int, point: int) -> tuple[int, int]:
        """The pixel a point of an operator's envelope is at."""
        widget = self.env(op)
        return widget.map.spots(fmview.settings(widget.op))[point]

    def grab(self, op: int, point: int, by: tuple[int, int], let_go: bool = True) -> tuple[dict, tuple[int, int]]:
        """Press on a point of an operator's envelope, and move so many pixels right and down.

        The answer is the operator as the sums of the envelope have it after that move, and where the mouse is.
        """
        widget, (x, y) = self.env(op), self.spot(op, point)
        made = fmview.moved(widget.op, point, widget.map, by)
        self.ui.handle(Event("down", (x, y), button=1))
        self.ui.handle(Event("move", (x + by[0], y + by[1])))
        if let_go:
            self.ui.handle(Event("up", (x + by[0], y + by[1]), button=1))
        return made, (x + by[0], y + by[1])

    def inks(self) -> list[theme.Colour | None]:
        """The colour of each column: of its heading, of what it does and of its envelope, None where they differ."""
        panel = self.panel
        return [name.colour if name.colour == does.colour == column["env"].colour else None
                for name, does, column in zip(panel.names, panel.roles, panel.operators)]

    def test_a_click_on_a_wiring_saves_the_algorithm_at_its_one_place_and_nothing_else(self) -> None:
        panel, before = self.panel, self.data()
        click(self.ui, self.cell(2))
        self.assertEqual((differing(before, self.data()), self.inst("brass")["alg"], panel.wiring.algorithm),
                         (["instruments.brass.alg"], 2, 2))
        self.assertEqual((self.session.undo_label, len(self.heard())), ("brass: algorithm 2", 1))
        self.assertIs(self.heard()[0], self.tone(A4, "brass"))
        # The wiring that is picked already is not picked again, and the 4 the engine takes without one is
        # written as any other.
        click(self.ui, self.cell(2))
        click(self.ui, self.cell(4))
        self.assertEqual((self.inst("brass"), self.session.undo_label, len(self.heard())),
                         ({**BRASS, "alg": 4}, "brass: algorithm 4", 2))

    def test_the_colours_of_the_columns_follow_the_wiring_that_is_picked(self) -> None:
        panel = self.panel
        heard, other = theme.ACTIVE, theme.SONG
        # Under wiring 5 operator 0 changes the three others, and those are heard.
        self.assertEqual(([does.text for does in panel.roles], self.inks()),
                         (["CHANGES 2, 3, 4", "HEARD", "HEARD", "HEARD"], [other, heard, heard, heard]))
        click(self.ui, self.cell(2))
        self.assertEqual(([does.text for does in panel.roles], self.inks()),
                         (["CHANGES 4", "CHANGES 3", "CHANGES 4", "HEARD"], [other, other, other, heard]))
        self.assertEqual([name.text for name in panel.names], [f"OPERATOR {op}" for op in range(1, 5)])
        # A wiring that is written outside is followed too, and so is one that is taken back.
        self.outside(lambda instruments: instruments["brass"].update(alg=7))
        panel.update()
        self.assertEqual((panel.wiring.algorithm, {does.text for does in panel.roles}, self.inks()),
                         (7, {"HEARD"}, [heard] * 4))
        self.assertTrue(self.session.undo())
        panel.update()
        self.assertEqual((panel.wiring.algorithm, self.inks()), (2, [other, other, other, heard]))

    def test_an_envelope_that_is_dragged_saves_its_operator_when_the_mouse_lets_go_and_not_before(self) -> None:
        before = self.data()
        made, mouse = self.grab(1, fmview.DECAY, (10, 5), let_go=False)
        # The operator named no envelope, so it had the engine's: its decay is longer now, and falls to a level
        # under the full one.
        self.assertEqual((sorted(made["env"]), made["env"]["d"] > 0.3, made["env"]["s"] < 1, made["mul"]),
                         (["d", "s"], True, True, 2))
        self.assertEqual((self.env(1).op, self.env(1).busy, self.data(), self.session.can_undo, self.heard()),
                         (made, True, before, False, []))
        self.ui.handle(Event("up", mouse, button=1))
        self.assertEqual((differing(before, self.data()), self.ops()),
                         (["instruments.brass.ops"], [BRASS["ops"][0], made]))
        self.assertEqual((self.session.undo_label, self.env(1).op, self.env(1).busy),
                         ("brass: operator 2 decay", made, False))
        self.assertIs(self.heard()[-1], self.tone(A4, "brass"))
        # Each point is called by its own name in the step it makes.
        for point, by, name in ((fmview.RELEASE, (-4, 0), "release"), (fmview.ATTACK, (6, 0), "attack"),
                                (fmview.SUSTAIN, (0, 6), "sustain")):
            made, _ = self.grab(0, point, by)
            self.assertEqual((self.ops()[0], self.session.undo_label), (made, f"brass: operator 1 {name}"))
        self.assertEqual((sorted(self.ops()[0]["env"]), len(self.heard())), (["a", "r", "sd"], 4))

    def test_each_knob_of_an_operator_is_saved_at_its_own_place_and_called_in_plain_words(self) -> None:
        panel = self.panel
        saved = {"mul": 1.25, "det": 0.1, "level": 1.05}
        labels = ["ratio 1.25", "detune 0.1", "volume 1.05"]
        for (key, value), label in zip(saved.items(), labels):
            before = self.data()
            # One step of the wheel is one step of the knob, and it is saved when the run of the wheel has ended.
            self.ui.handle(Event("wheel", middle(panel.operators[0][key].rect), wheel=1))
            panel.update()
            self.assertEqual(self.data(), before)
            self.now += WHEEL_WAIT
            panel.update()
            self.assertEqual((differing(before, self.data()), self.ops()[0][key], self.session.undo_label),
                             (["instruments.brass.ops"], value, f"brass: operator 1 {label}"))
        self.assertEqual(self.ops(), [{"mul": 1.25, "det": 0.1, "level": 1.05}, BRASS["ops"][1]])
        # A knob that is dragged is saved when it is let go, and not before.
        level = panel.operators[1]["level"]
        mouse = self.drag(level, 10, let_go=False)
        self.assertEqual((level.value, level.busy, self.ops()[1]), (1.3, True, BRASS["ops"][1]))
        self.ui.handle(Event("up", mouse, button=1))
        self.assertEqual((self.ops()[1], self.session.undo_label), ({"mul": 2, "level": 1.3},
                                                                   "brass: operator 2 volume 1.3"))

    def test_the_feedback_is_saved_when_its_knob_is_let_go_and_at_none_it_is_taken_out_of_the_file(self) -> None:
        panel, before = self.panel, self.data()
        mouse = self.drag(panel.feedback, 5, let_go=False)
        self.assertEqual((panel.feedback.value, self.data(), self.session.can_undo), (0.4, before, False))
        self.ui.handle(Event("up", mouse, button=1))
        self.assertEqual((differing(before, self.data()), self.inst("brass")["fb"], self.session.undo_label),
                         (["instruments.brass.fb"], 0.4, "brass: feedback 0.4"))
        click(self.ui, middle(panel.feedback.rect), button=3)
        self.assertEqual(("fb" in self.inst("brass"), self.session.undo_label, panel.feedback.value),
                         (False, "brass: feedback 0", 0))

    def test_what_is_put_back_to_what_the_engine_takes_for_an_operator_without_it_is_taken_out_of_the_file(
            self) -> None:
        panel = self.panel
        click(self.ui, middle(panel.operators[1]["mul"].rect), button=3)
        self.assertEqual((self.ops()[1], self.session.undo_label, panel.operators[1]["mul"].value),
                         ({"level": 0.5}, "brass: operator 2 ratio 1", 1))
        # An attack dragged away from none is written, and dragged back to none it is not: an envelope with
        # nothing left to write goes with it.
        made, _ = self.grab(0, fmview.ATTACK, (6, 0))
        self.assertEqual((self.ops()[0], sorted(made["env"]), made["env"]["a"] > 0), (made, ["a"], True))
        self.grab(0, fmview.ATTACK, (-40, 0))
        self.assertEqual((self.ops()[0], self.session.undo_label, self.env(0).op),
                         (BRASS["ops"][0], "brass: operator 1 attack", BRASS["ops"][0]))
        # Of the two parts a decay sets, the level dragged back to the full one goes, and the time stays.
        made, _ = self.grab(1, fmview.DECAY, (10, 5))
        self.grab(1, fmview.DECAY, (0, -300))
        self.assertEqual((self.ops()[1], self.session.undo_label),
                         ({"level": 0.5, "env": {"d": made["env"]["d"]}}, "brass: operator 2 decay"))
        # What the file has at such a value by itself stays in it, where a drag sets another part.
        self.outside(lambda instruments: instruments["brass"]["ops"][0].update(env={"d": 0.3, "r": 0.5}))
        panel.update()
        made, _ = self.grab(0, fmview.RELEASE, (-5, 0))
        self.assertEqual((self.ops()[0]["env"], made["env"]["d"], made["env"]["r"] < 0.5), (made["env"], 0.3, True))

    def test_an_operator_the_file_leaves_out_is_written_after_silent_ones_when_it_is_changed(self) -> None:
        panel, before = self.panel, self.data()
        self.ui.handle(Event("wheel", middle(panel.operators[3]["level"].rect), wheel=2))
        self.now += WHEEL_WAIT
        panel.update()
        # Operator 2 is written as the silent one the engine took for it, so the sound has changed by no more
        # than the level that was turned up.
        self.assertEqual((differing(before, self.data()), self.ops(), self.session.undo_label),
                         (["instruments.brass.ops"], [*BRASS["ops"], SILENT, {**SILENT, "level": 0.1}],
                          "brass: operator 4 volume 0.1"))
        # It is in the file now, and what is changed of it next is written at its own place.
        made, _ = self.grab(2, fmview.DECAY, (20, -30))
        self.assertEqual((self.ops()[2], made["level"], made["env"]["s"] > 0, self.session.undo_label),
                         (made, 0, True, "brass: operator 3 decay"))
        # Its level at the 1 the engine takes for an operator that names none is not written.
        click(self.ui, middle(panel.operators[2]["level"].rect), button=3)
        self.assertEqual((self.ops()[2], self.ops()[3]), ({"env": made["env"]}, {**SILENT, "level": 0.1}))

    def test_a_step_taken_back_puts_the_picture_of_the_fm_voice_back(self) -> None:
        panel = self.panel
        click(self.ui, self.cell(2))
        made, _ = self.grab(1, fmview.DECAY, (10, 5))
        self.drag(panel.operators[0]["level"], 10)

        def state() -> tuple:
            """What the pictures show of the three things that were changed, and of the heading that follows."""
            return panel.wiring.algorithm, panel.roles[0].text, self.env(1).op, panel.operators[0]["level"].value

        self.assertEqual(state(), (2, "CHANGES 4", made, 1.8))
        back = []
        for _ in range(3):
            self.assertTrue(self.session.undo())
            panel.update()
            back.append(state())
        self.assertEqual(back, [(2, "CHANGES 4", made, 1), (2, "CHANGES 4", BRASS["ops"][1], 1),
                                (5, "CHANGES 2, 3, 4", BRASS["ops"][1], 1)])
        self.assertEqual((self.inst("brass"), self.session.can_undo), (BRASS, False))
        self.assertTrue(self.session.redo())
        panel.update()
        self.assertEqual(state(), (2, "CHANGES 4", BRASS["ops"][1], 1))

    def test_the_tables_an_fm_voice_reads_are_shown_and_saved_and_the_width_of_a_pulse_is_not_there(self) -> None:
        panel = self.panel
        # Two of them are in view under the operators, and a step of the wheel brings the other two.
        self.assertEqual([key for key, table in panel.tables.items() if table.visible], ["vol", "arp"])
        self.assertTrue(self.ui.handle(Event("wheel", middle(panel.tables["vol"].rect), wheel=-1)))
        self.assertEqual([(key, table.enabled) for key, table in panel.tables.items() if table.visible],
                         [("vol", True), ("arp", True), ("pitch", True), ("lp", True)])
        self.assertEqual((panel.tables["vol"].table, panel.tables["lp"].table, panel.tables["lp"].rect),
                         ([15, 10], 2800, arrange(RECT, 1, True)["lp"]))
        drawn = (("vol", 3, [3, 10], "volume shape"), ("arp", 7, 7, "arpeggio"), ("pitch", -2, -2, "pitch"))
        for key, value, table, word in drawn:
            before = self.data()
            click(self.ui, self.bar(key, 0, value))
            self.assertEqual((differing(before, self.data()), self.inst("brass")[key], self.session.undo_label),
                             ([f"instruments.brass.{key}"], table, f"brass: {word} table"))
        # A cutoff drawn down to none is the engine's own, and is taken out of the file.
        click(self.ui, self.bar("lp", 0, 0))
        self.assertEqual(("lp" in self.inst("brass"), self.session.undo_label, panel.tables["lp"].table, panel.first),
                         (False, "brass: cutoff table", 0, 1))

    def test_a_change_from_outside_waits_for_an_envelope_that_is_dragged_or_only_pressed(self) -> None:
        panel = self.panel
        made, mouse = self.grab(1, fmview.DECAY, (10, 5), let_go=False)
        self.outside(lambda instruments: (instruments["brass"].update(fb=0.5, alg=2),
                                          instruments["brass"]["ops"][1].update(level=0.9, env={"r": 1})))
        panel.update()
        # All but the envelope under the hand shows what was written, and that one has its colour from it.
        self.assertEqual((self.env(1).op, self.env(1).busy, self.env(1).colour), (made, True, theme.SONG))
        self.assertEqual((panel.operators[1]["level"].value, panel.feedback.value, panel.wiring.algorithm),
                         (0.9, 0.5, 2))
        # The release saves the envelope as the hand has it, in the place of the one written outside and beside
        # the level written outside.
        self.ui.handle(Event("up", mouse, button=1))
        self.assertEqual((self.ops()[1], self.env(1).op, self.session.undo_label, self.session.message),
                         ({"mul": 2, "level": 0.9, "env": made["env"]},) * 2 + ("brass: operator 2 decay", ""))
        # An envelope that is pressed and let go where it was saves nothing, and then shows what came.
        at = self.spot(0, fmview.DECAY)
        self.ui.handle(Event("down", at, button=1))
        self.outside(lambda instruments: instruments["brass"]["ops"][0].update(env={"a": 0.1}))
        panel.update()
        self.assertEqual((self.env(0).op, self.env(0).busy), (BRASS["ops"][0], True))
        self.ui.handle(Event("up", at, button=1))
        self.assertEqual((self.env(0).op, self.session.undo_label), (BRASS["ops"][0], "changed outside: brass"))
        panel.update()
        self.assertEqual(self.env(0).op, {**BRASS["ops"][0], "env": {"a": 0.1}})
        # A knob in a run of the wheel is under the hand as well, though no button holds it.
        level = panel.operators[0]["level"]
        self.ui.handle(Event("wheel", middle(level.rect), wheel=1))
        self.outside(lambda instruments: instruments["brass"]["ops"][0].update(level=0.3, mul=3))
        panel.update()
        self.assertEqual((level.value, level.busy, panel.operators[0]["mul"].value), (1.05, True, 3))
        self.now += WHEEL_WAIT
        panel.update()
        self.assertEqual((self.ops()[0], level.value, self.session.undo_label),
                         ({"mul": 3, "level": 1.05, "env": {"a": 0.1}}, 1.05, "brass: operator 1 volume 1.05"))

    def test_another_instrument_picked_while_an_envelope_is_dragged_takes_the_panel_and_nothing_is_saved(
            self) -> None:
        panel, before = self.panel, self.data()
        _, (x, y) = self.grab(1, fmview.DECAY, (10, 5), let_go=False)
        self.pick("bell")
        self.assertEqual((panel.name, self.env(1).busy, self.env(1).visible), ("bell", False, False))
        self.ui.handle(Event("move", (x, y + 5)))
        self.ui.handle(Event("up", (x, y + 5), button=1))
        self.assertEqual((self.data(), self.session.can_undo), (before, False))

    def test_a_change_to_the_fm_voice_that_is_refused_leaves_the_pictures_showing_the_file(self) -> None:
        panel = self.panel
        # Another program has taken the instrument out of the file, and the session has not looked at it yet.
        written = self.outside(lambda instruments: instruments.pop("brass"), look=False)
        refusal = "instruments has no key 'brass'"
        click(self.ui, self.cell(2))
        self.assertEqual((panel.wiring.algorithm, panel.roles[0].text, self.session.message),
                         (5, "CHANGES 2, 3, 4", refusal))
        self.grab(1, fmview.DECAY, (10, 5))
        self.assertEqual((self.env(1).op, self.session.message), (BRASS["ops"][1], refusal))
        # So it is with an operator the file left out, which would have been written with all the others.
        self.drag(panel.operators[3]["level"], 10)
        self.assertEqual((panel.operators[3]["level"].value, self.session.message), (0, refusal))
        self.assertEqual((self.data(), self.session.can_undo, self.session.messages, self.heard()),
                         (written, False, 3, []))


class Traced(Opened):
    def watch(self) -> tuple[mock.Mock, mock.Mock]:
        """Have what the scope and the spectrum are handed from now on kept, and still shown by them."""
        kept = []
        for widget in (self.panel.scope, self.panel.spectrum):
            patch = mock.patch.object(widget, "show", wraps=widget.show)
            kept.append(patch.start())
            self.addCleanup(patch.stop)
        return kept[0], kept[1]

    def test_the_scope_and_the_spectrum_show_the_note_played_last_which_is_a4_before_any_key_is_pressed(
            self) -> None:
        panel = self.panel
        self.assertEqual((panel.scope.colour, panel.spectrum.colour, panel.scope.visible, panel.spectrum.visible),
                         (theme.INSTRUMENT, theme.INSTRUMENT, True, True))
        self.assertEqual((panel.scope.rect, panel.spectrum.rect, panel.scope.help),
                         (arrange(RECT)["scope"], arrange(RECT)["spectrum"], "3 cycles of a4, 6.8 ms"))
        scope, spectrum = self.watch()
        click(self.ui, self.key(52))
        # The scope is handed the pitch with the note, and the spectrum the note alone.
        self.assertEqual((scope.call_count, spectrum.call_count, scope.call_args.args[1:], spectrum.call_args.args[1:]),
                         (1, 1, (RATE, 52), (RATE,)))
        self.assertIs(scope.call_args.args[0], self.tone(52)[0])
        self.assertIs(spectrum.call_args.args[0], self.tone(52)[0])
        self.assertEqual(panel.scope.help, "3 cycles of e3, 18.1 ms")
        # A key of the computer is a key of the keyboard.
        self.ui.handle(Event("key", key="q"))
        self.assertEqual((scope.call_count, scope.call_args.args[2]), (2, 60))
        self.assertIs(spectrum.call_args.args[0], self.tone(60)[0])

    def test_after_a_change_is_saved_the_two_are_shown_the_sound_as_it_is_now(self) -> None:
        panel = self.panel
        click(self.ui, self.key(52))
        was = self.tone(52)[0]
        scope, spectrum = self.watch()
        click(self.ui, middle(panel.knobs["gain"].rect), button=3)
        self.assertEqual((scope.call_count, scope.call_args.args[1:]), (1, (RATE, 52)))
        self.assertIs(scope.call_args.args[0], self.tone(52)[0])
        self.assertIs(spectrum.call_args.args[0], self.tone(52)[0])
        self.assertIsNot(scope.call_args.args[0], was)
        # So it is after a change to an FM voice.
        self.pick("brass")
        was = self.tone(52, "brass")[0]
        click(self.ui, self.cell(7))
        self.assertIs(scope.call_args.args[0], self.tone(52, "brass")[0])
        self.assertIs(spectrum.call_args.args[0], self.tone(52, "brass")[0])
        self.assertIsNot(spectrum.call_args.args[0], was)

    def test_after_a_change_from_outside_the_two_are_shown_the_sound_as_it_is_now(self) -> None:
        panel = self.panel
        was = self.tone(A4)[0]
        scope, spectrum = self.watch()
        self.outside(lambda instruments: instruments["bell"].update(wave="saw"))
        # They are shown it when the panel takes its frame.
        self.assertEqual(scope.call_count, 0)
        panel.update()
        self.assertEqual((scope.call_count, spectrum.call_count, scope.call_args.args[1:]), (1, 1, (RATE, A4)))
        self.assertIs(scope.call_args.args[0], self.tone(A4)[0])
        self.assertIs(spectrum.call_args.args[0], self.tone(A4)[0])
        self.assertIsNot(scope.call_args.args[0], was)
        # A frame with nothing new shows them nothing again.
        panel.update()
        self.assertEqual(scope.call_count, 1)

    def test_every_instrument_that_is_picked_is_shown_whatever_its_wave_and_a_note_that_is_refused_is_no_sound(
            self) -> None:
        panel = self.panel
        scope, spectrum = self.watch()
        for count, name in enumerate(("plain", "hiss", "brass"), 1):
            self.pick(name)
            self.assertEqual((scope.call_count, spectrum.call_count, scope.call_args.args[1:]),
                             (count, count, (RATE, A4)))
            self.assertIs(scope.call_args.args[0], self.tone(A4, name)[0])
            self.assertIs(spectrum.call_args.args[0], self.tone(A4, name)[0])
            self.assertTrue(panel.scope.visible and panel.spectrum.visible)
        # With a song picked there is no note to show.
        self.pick("walk", "song")
        self.assertEqual((scope.call_count, panel.scope.visible), (3, False))
        # A note the model cannot make is no sound to the two: each has the line it was made with.
        self.assertNotEqual(panel.scope.help, SCOPE_HELP)
        with mock.patch.object(self.session.model, "tone", side_effect=RowSynthError("no such note")):
            self.pick("bell")
        self.assertEqual((panel.scope.help, len(spectrum.call_args.args[0]), panel.spectrum.above),
                         (SCOPE_HELP, 0, 0))


class Placed(Opened):
    def test_the_heading_has_the_waves_at_its_right_end_and_the_bar_every_knob_in_its_order(self) -> None:
        panel, parts = self.panel, arrange(RECT)
        self.assertEqual((type(panel.heading), type(panel.frame), panel.heading.rect, panel.frame.rect),
                         (Heading, Bar, parts["head"], parts["bar"]))
        self.assertEqual((panel.heading.kind, panel.heading.name, panel.heading.facts, panel.heading.colour),
                         ("INSTRUMENT", "bell", "in 3 channels of walk, 1 drum of walk, 3 sounds", theme.INSTRUMENT))
        # The waves are small buttons at the right end of the heading.
        x, y, w, h = panel.wave.rect
        self.assertEqual((panel.wave.options, y, h, x + w), (("PULSE", "SAW", "TRI", "NOISE", "FM"), 28, 13, 780))
        self.assertEqual(within(panel.wave.rect, parts["head"]), panel.wave.rect)
        # Every knob of the instrument stands in the bar, in one row, and after them the switch.
        order = [*(panel.knobs[number.key] for number in NUMBERS), panel.short]
        self.assertEqual([knob.text for knob in order[:-1]],
                         ["VOLUME", "RELEASE", "GATE", "SLIDE", "VIBRATO DEPTH", "DELAY", "RATE", "RESONANCE",
                          "KEY TRACKING"])
        self.assertEqual([widget.rect[0] for widget in order], [134, 177, 226, 265, 304, 389, 428, 467, 528, 607])
        self.assertEqual(({knob.rect[1::2] for knob in order[:-1]}, panel.short.rect, panel.short.text),
                         ({(47, 37)}, (607, 59, 67, 13), "OFF"))
        self.assertTrue(all(widget.visible and within(widget.rect, parts["bar"]) == widget.rect for widget in order))
        self.assertTrue(all(apart(one.rect, other.rect) for one in order for other in order if one is not other))
        # No knob is anywhere else, and the one of the feedback is there for the FM wave alone: it stands last.
        self.assertEqual(([widget for widget in panel.widgets if isinstance(widget, Knob) and widget.visible],
                          panel.feedback.visible), (order[:-1], False))
        self.pick("brass")
        self.assertEqual((panel.feedback.visible, panel.feedback.rect, panel.feedback.text, panel.short.rect),
                         (True, (680, 47, 49, 37), "FEEDBACK", (607, 59, 67, 13)))
        # An instrument of a song's own has its song after its name, and the song is not named again after that.
        self.pick("walk.own")
        self.assertEqual((panel.heading.name, panel.heading.facts), ("own", "of walk  in 2 channels"))
        # In a bar too narrow for all of it the control at its end has no room, and is not seen.
        panel.place(SMALL)
        self.assertEqual([widget.visible for widget in order], [True] * 9 + [False])

    def test_the_side_column_names_each_row_and_counts_the_steps_of_a_table_as_it_is_drawn(self) -> None:
        panel = self.panel
        self.assertEqual({key: (panel.titles[key].text, panel.facts[key].text) for key in panel.tables},
                         {"vol": ("VOLUME SHAPE", "4 steps, 0.07 s"), "duty": ("PULSE WIDTH", "2 steps, 0.03 s"),
                          "arp": ("ARPEGGIO", "1 step, 0.02 s"), "pitch": ("PITCH", "2 steps, 0.03 s"),
                          "lp": ("CUTOFF", "1 step, 0.02 s")})
        self.assertEqual((panel.titles["vol"].rect, panel.facts["vol"].rect, panel.titles["alg"].visible),
                         (arrange(RECT)["vol.name"], arrange(RECT)["vol.facts"], False))
        # While the handle of a table is dragged, the count is the one the release would give.
        seen = panel.tables["vol"].map
        x, y = middle(seen.handle(4))
        self.assertEqual(self.said((x, y)), f"press: {SIZE_HELP}   wheel: along the tables")
        self.ui.handle(Event("down", (x, y), button=1))
        self.ui.handle(Event("move", (x + 2 * seen.pitch, y)))
        panel.update()
        self.assertEqual((panel.facts["vol"].text, self.inst()["vol"]), ("6 steps, 0.10 s", [15, 12, 9, 6]))
        self.ui.handle(Event("up", (x + 2 * seen.pitch, y), button=1))
        self.assertEqual((panel.facts["vol"].text, self.inst()["vol"], self.session.undo_label),
                         ("6 steps, 0.10 s", [15, 12, 9, 6, 6, 6], "bell: volume shape table"))
        # A press under the bars sets the loop, which is told on a line of its own.
        at = (seen.x(1) + 1, middle(seen.strip)[1])
        self.assertEqual(self.said(at), "press: loop the table from step 2   wheel: along the tables")
        click(self.ui, at)
        self.assertEqual(self.inst()["vol"], {"v": [15, 12, 9, 6, 6, 6], "loop": 1})
        # With an FM voice the first row is the wirings, named with the picked one under the name.
        self.pick("brass")
        self.assertEqual((panel.titles["alg"].text, panel.facts["alg"].text, [name.text for name in panel.names]),
                         ("WIRING", "algorithm 5", ["OPERATOR 1", "OPERATOR 2", "OPERATOR 3", "OPERATOR 4"]))
        click(self.ui, self.cell(2))
        self.assertEqual((panel.facts["alg"].text, panel.roles[0].rect, panel.roles[0].text),
                         ("algorithm 2", arrange(RECT, fm=True)["ops[0].facts"], "CHANGES 4"))

    def test_every_table_has_room_for_the_same_steps_under_the_ruler_and_goes_sideways_with_the_others(self) -> None:
        panel = self.panel
        # As many steps as the handle makes a table fit side by side: a step is 8 pixels in every table.
        self.assertEqual(({(table.map.bar, table.map.x(0), table.map.x(MOST)) for table in panel.tables.values()},
                          MOST), ({(7, 256, 768)}, 64))
        # A table of the file that is longer has its room, and every other table room for as many.
        self.outside(lambda instruments: instruments["bell"].update(vol=[9] * 200))
        panel.update()
        self.assertEqual({(table.span, table.map.bar) for table in panel.tables.values()}, {(200, 2)})
        # They do not fit now. Sideways, which is the wheel with shift, one table takes the others along.
        before, at = self.data(), middle(panel.tables["arp"].rect)
        self.assertTrue(self.ui.handle(Event("wheel", at, wheel=-1, mods=SHIFT)))
        self.assertEqual(({table.map.first for table in panel.tables.values()}, panel.first), ({36}, 0))
        self.ui.handle(Event("wheel", at, wheel=-9, mods=SHIFT))
        self.assertEqual({table.map.x(199) for table in panel.tables.values()}, {256 + 519 - 3 - 3})
        # A press is on the step that is drawn there, and no step of the wheel has changed a value.
        self.assertEqual(self.data(), before)
        click(self.ui, (panel.tables["vol"].map.x(199), self.bar("vol", 0, 3)[1]))
        self.assertEqual(self.inst()["vol"], [9] * 199 + [3])

    def test_the_wheel_over_the_main_area_goes_along_the_rows_and_changes_no_value(self) -> None:
        panel, before = self.panel, self.data()
        # In the middle of the window every table is in view: the wheel has nothing to do, and the bar no thumb.
        at = middle(panel.tables["arp"].rect)
        self.assertEqual((self.ui.handle(Event("wheel", at, wheel=-1)), panel.first, panel.along.thumb(), self.rows()),
                         (False, 0, None, ["vol", "duty", "arp", "pitch", "lp"]))
        panel.place(SMALL)
        parts = arrange(SMALL)
        self.assertEqual((panel.along.rect, panel.along.thumb() is None, self.rows()),
                         (parts["scroll"], False, ["vol", "duty", "arp"]))
        # A step of the wheel is one row: over a table, which has no use for it, and over the side column.
        self.assertTrue(self.ui.handle(Event("wheel", middle(panel.tables["arp"].rect), wheel=-1)))
        self.assertEqual((panel.first, self.rows(), panel.tables["duty"].rect, panel.titles["duty"].rect),
                         (1, ["duty", "arp", "pitch"], arrange(SMALL, 1)["duty"], arrange(SMALL, 1)["duty.name"]))
        self.assertTrue(self.ui.handle(Event("wheel", middle(parts["side"]), wheel=-1)))
        self.assertEqual((panel.first, self.rows()), (2, ["arp", "pitch", "lp"]))
        # At their end the rows stay, and the step of the wheel is not used.
        self.assertFalse(self.ui.handle(Event("wheel", middle(parts["side"]), wheel=-1)))
        # Sideways and with ctrl the wheel does nothing there, and it does nothing over the foot or the heading.
        for pos, mods in ((middle(parts["side"]), SHIFT), (middle(parts["side"]), frozenset({"ctrl"})),
                          (middle(parts["foot"]), frozenset()), (middle(parts["head"]), frozenset())):
            self.assertFalse(self.ui.handle(Event("wheel", pos, wheel=1, mods=mods)), (pos, mods))
        self.assertEqual(panel.first, 2)
        self.assertTrue(self.ui.handle(Event("wheel", middle(parts["side"]), wheel=9)))
        self.assertEqual((panel.first, self.data(), self.session.can_undo), (0, before, False))
        # Over a knob the wheel turns the knob, and the rows stay.
        gain = panel.knobs["gain"]
        self.assertTrue(self.ui.handle(Event("wheel", middle(gain.rect), wheel=1)))
        self.assertEqual((gain.value, panel.first), (0.81, 0))
        # Given the room again, the rows are where they all fit.
        self.ui.handle(Event("wheel", middle(parts["side"]), wheel=-2))
        panel.place(RECT)
        self.assertEqual((panel.first, self.rows()), (0, ["vol", "duty", "arp", "pitch", "lp"]))

    def test_the_scroll_bar_shows_where_the_view_is_and_takes_the_rows_along(self) -> None:
        panel, along = self.panel, self.panel.along
        self.pick("brass")
        x, y, w, h = along.rect
        self.assertEqual((type(along), along.rect, along.upright, along.thumb()[1], self.rows()),
                         (ScrollBar, arrange(RECT, fm=True)["scroll"], True, y,
                          ["alg", "ops[0]", "ops[1]", "ops[2]", "ops[3]", "vol", "arp"]))
        self.assertEqual(self.said(middle(along.rect)), "press: drag the bar to go along them, or press beside it to "
                                                        "jump   wheel: along the operators")
        # A press under the thumb takes the view to the end of the rows, and the thumb to the end of the bar.
        click(self.ui, (x + 1, y + h - 2))
        self.assertEqual((panel.first, self.rows(), along.thumb()[1] + along.thumb()[3]),
                         (1, ["ops[0]", "ops[1]", "ops[2]", "ops[3]", "vol", "arp", "pitch", "lp"], y + h))
        # The thumb is dragged, and the view begins at the row that starts nearest to where it was taken.
        at = middle(along.thumb())
        self.ui.handle(Event("down", at, button=1))
        self.ui.handle(Event("move", (at[0], at[1] - 200)))
        self.ui.handle(Event("up", (at[0], at[1] - 200), button=1))
        self.assertEqual((panel.first, self.rows()[0], along.thumb()[1]), (0, "alg", y))
        panel.place(SMALL)
        at = middle(along.thumb())
        self.ui.handle(Event("down", at, button=1))
        moved = []
        for down in (4, 30, 60, 400):
            self.ui.handle(Event("move", (at[0], at[1] + down)))
            moved.append((panel.first, self.rows()[0]))
        self.ui.handle(Event("up", (at[0], at[1] + 400), button=1))
        self.assertEqual((moved[0], moved[-1], [first for first, _ in moved] == sorted(first for first, _ in moved)),
                         ((0, "alg"), (6, "vol"), True))
        # Nothing of all that changed a value.
        self.assertFalse(self.session.can_undo)

    def test_every_line_of_help_is_in_the_form_of_the_sheet_and_fits_the_status_bar(self) -> None:
        panel, lines = self.panel, set()
        x, y, w, h = RECT
        for name, first in (("bell", 0), ("hiss", 0), ("brass", 0), ("brass", 1)):
            self.pick(name)
            panel.first = first
            panel.place(RECT)
            lines.update(self.said((left, top)) for top in range(y, y + h, 4) for left in range(x, x + w, 4))
        # Wherever the mouse rests on the panel there is a line. None begins with a capital or ends with a full
        # stop, and each has its room in the smallest window.
        self.assertEqual(sorted(line for line in lines if not line or line[0].isupper() or line.endswith(".")
                                or Font.width(line) > ROOM), [])
        self.assertGreater(len(lines), 80)
        # What the left press does comes first, then the right press, then the wheel.
        marks = ("press: ", "   right press: ", "   wheel: ")
        for line in lines:
            found = [line.index(mark) for mark in marks if mark in line]
            self.assertEqual(found, sorted(found), line)
        table = f"{TABLES[0].help}   wheel: along the tables"
        for line in (f"press: {PAGE_HELP}", f"press: {PAGE_HELP}   wheel: along the tables",
                     f"press: {PAGE_HELP}   wheel: along the operators", table, NUMBERS[0].help, DIALS[0].help,
                     f"{ENVELOPE_HELP}   wheel: along the operators", SHORT_UNUSED,
                     "press: set step 1 of 4: volume 15, at 0.00 s   right press: back to zero   wheel: along the "
                     "tables", "press: hear c3, and a drag plays each key it comes onto"):
            self.assertIn(line, lines)
        # A knob: a drag, with shift slower, a right press back to its usual value, and the wheel steps it.
        self.assertEqual((NUMBERS[0].help, FEEDBACK.help),
                         ("press: drag up or down for how loud the instrument is beside the others, slower with "
                          "shift   right press: back to 1   wheel: steps it",
                          "press: drag up or down for how far operator 1 bends its own wave, slower with shift   "
                          "right press: back to 0   wheel: steps it"))

    def test_at_either_size_every_widget_is_in_view_or_not_seen_no_two_overlap_and_no_row_is_dropped(self) -> None:
        panel = self.panel
        grounds = (panel.page, panel.heading, panel.frame)
        for name in ("bell", "brass"):
            self.pick(name)
            fm = name == "brass"
            for rect in (RECT, SMALL):
                ever = set()
                for first in range(reach(rect, fm) + 1):
                    panel.first = first
                    panel.place(rect)
                    shown = [widget for widget in panel.widgets if widget.visible and widget not in grounds]
                    ever.update(shown)
                    self.assertTrue(all(within(widget.rect, rect) == widget.rect for widget in shown), (rect, first))
                    self.assertTrue(all(apart(one.rect, other.rect) for one in shown for other in shown
                                        if one is not other), (rect, first))
                # What is not seen with the view at one row is seen with it at another: every row the wave has.
                parts = {part for first in range(reach(rect, fm) + 1) for part in placed(rect, first, fm)}
                self.assertLessEqual({panel._placed[part] for part in parts}, ever, (name, rect))
                self.assertEqual(sorted(part for part in parts if "." not in part and part not in FOOT),
                                 sorted(["scroll", *(["alg"] if fm else []),
                                         *(table.key for table in (FM_TABLES if fm else TABLES))]), (name, rect))
            self.assertEqual((reach(RECT, fm), reach(SMALL, fm)), ((1, 6) if fm else (0, 2)))


@unittest.skipUnless(pygame, NEEDS)
class Drawn(unittest.TestCase):
    def setUp(self) -> None:
        self.open("arcade.json", "chime")

    def open(self, file: str, name: str) -> None:
        """Draw the panel with an instrument of an example picked, on a surface of its own."""
        self.surface = pygame.Surface((960, 540))
        self.surface.fill(theme.BACKGROUND)
        self.canvas = Canvas(self.surface, Font(sprites.load("font")), sprites.load("icons"))
        self.ui = Ui()
        model = Model(EXAMPLES / file)
        model.pick("instrument", name)
        self.panel = InstPanel(Stand(model), self.ui)
        self.panel.place(RECT)
        self.panel.update()
        self.panel.draw(self.canvas)
        self.ui.draw(self.canvas)

    def ink(self, rect: layout.Rect, colour: theme.Colour) -> int:
        """How many pixels of a rectangle are in a colour."""
        x, y, w, h = rect
        return sum(self.surface.get_at((x + column, y + row))[:3] == colour for column in range(w) for row in range(h))

    def test_the_first_bar_of_the_volume_shape_of_the_chime_is_as_high_as_the_file_says(self) -> None:
        held = json.loads((EXAMPLES / "arcade.json").read_text(encoding="utf-8"))["instruments"]["chime"]
        self.assertEqual(held["vol"], [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        seen = self.panel.tables["vol"].map
        left, top, _, rows = seen.bars
        # A level of 15 is every row of the field, and each level under it that share of them, to the pixel.
        high = [self.ink((seen.x(step), top, 1, rows), theme.INSTRUMENT) for step in range(15)]
        self.assertEqual((rows, high), (58, [math.floor(level * 58 / 15 + 0.5) for level in held["vol"]]))
        self.assertEqual((high[0], high[7], high[14], seen.x(0)), (58, 31, 4, left))
        # The bar stands on the foot of the field, seven pixels wide, with nothing of its colour beside it.
        self.assertEqual(self.ink((left, top, 8, rows), theme.INSTRUMENT), 7 * 58)

    def test_the_side_column_has_the_names_of_the_tables_and_the_ruler_a_number_at_every_eighth_step(self) -> None:
        panel, parts = self.panel, arrange(RECT)
        # The side column and the foot are framed panels.
        for part in ("side", "foot"):
            x, y, w, h = parts[part]
            self.assertEqual([self.surface.get_at(pos)[:3] for pos in ((x, y), (x + w - 1, y + h - 1), (x + 1, y + 1))],
                             [theme.LINE, theme.LINE, theme.PANEL], part)
        # Each table has its name beside it in the colour of text, and under that the count of its steps, dim.
        for key in panel.tables:
            self.assertGreater(self.ink(parts[f"{key}.name"], theme.TEXT), 0, key)
            self.assertEqual(self.ink(parts[f"{key}.name"], theme.DIM), 0, key)
            self.assertGreater(self.ink(parts[f"{key}.facts"], theme.DIM), 0, key)
        self.assertEqual((panel.titles["vol"].text, panel.facts["vol"].text), ("VOLUME SHAPE", "15 steps, 0.25 s"))
        # The ruler has a number where every eighth step begins, counted from 1, and none between two.
        seen = panel.tables["vol"].map
        x, y, w, h = parts["ruler"]
        self.assertEqual([self.ink((seen.x(step), y, 6, h), theme.DIM) > 0 for step in (0, 4, 8, 12, 56)],
                         [True, False, True, False, True])
        self.assertEqual((self.surface.get_at((x + w - 1, y))[:3], self.ink(parts["ruler"], theme.TEXT)),
                         (theme.PANEL, 0))
        # The word over the stepper of the octave is in the foot.
        self.assertGreater(self.ink(parts["try"], theme.DIM), 0)

    def test_the_heading_names_the_instrument_in_its_colour_and_the_key_of_the_note_heard_again_is_lit(self) -> None:
        parts = arrange(RECT)
        head = parts["head"]
        wide = Font.width("chime")
        name = (head[0] + theme.BAR_MARGIN + Font.width("INSTRUMENT") + theme.HEAD_SPACE, head[1], wide, head[3])
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
        self.assertEqual(self.ink((x, y + 34, w - 1, h - 34), theme.ACTIVE), (w - 1) * (h - 34))

    def test_the_scope_has_the_wave_of_the_chime_and_the_spectrum_its_bars_in_the_colour_of_an_instrument(
            self) -> None:
        scope, spectrum = self.panel.scope, self.panel.spectrum
        self.assertEqual((scope.rect, spectrum.rect, scope.help),
                         (arrange(RECT)["scope"], arrange(RECT)["spectrum"], "3 cycles of a4, 6.8 ms"))
        # The wave is one line from side to side of the scope: a pixel at least in each column inside its outline.
        x, y, w, h = scope.rect
        self.assertTrue(all(self.ink((x + 1 + column, y, 1, h), theme.INSTRUMENT) for column in range(w - 2)))
        # A pulse has a bar in most columns of the spectrum, and the loudest is as high as the box is inside.
        x, y, w, h = spectrum.bars
        high = [self.ink((x + column, y, 1, h), theme.INSTRUMENT) for column in range(w)]
        self.assertEqual((h, max(high), sum(bar > 0 for bar in high) > w * 3 // 4), (45, 45, True))
        # The line where harsh begins stands in it, and the names of the marks under its box.
        self.assertGreater(self.ink(spectrum.bars, theme.FAULT), h - 1)
        self.assertGreater(self.ink((x, y + h + 1, w, 9), theme.DIM), 0)

    def test_the_picked_wiring_of_an_fm_instrument_is_lit_and_its_columns_are_in_the_colours_of_the_wiring(
            self) -> None:
        self.open("piece.json", "fmbrass")
        panel = self.panel
        cells = fmview.cells(panel.wiring.rect, panel.wiring.between)
        self.assertEqual((panel.name, panel.wiring.algorithm, len(cells)), ("fmbrass", 4, 8))
        # The picked picture has the lit ground and an outline in the colour of what is on. The others have neither.
        corners = [self.surface.get_at((x, y))[:3] for x, y, _, _ in cells]
        grounds = [self.surface.get_at((x + 2, y + h - 3))[:3] for x, y, _, h in cells]
        self.assertEqual((corners, grounds), ([theme.ACTIVE if wiring == 4 else theme.LINE for wiring in range(8)],
                                              [theme.LIT if wiring == 4 else theme.RAISED for wiring in range(8)]))
        # In wiring 4 the second and the fourth operator are heard: their names in the side column and the lines
        # of their envelopes are in that colour, and those of the two that change them in the other.
        for op, (mine, other) in enumerate(((theme.SONG, theme.ACTIVE), (theme.ACTIVE, theme.SONG)) * 2):
            for rect in (panel.names[op].rect, panel.roles[op].rect, panel.operators[op]["env"].rect):
                self.assertEqual((self.ink(rect, mine) > 0, self.ink(rect, other)), (True, 0), (op, rect))
        # The spectrum of the brass has bars, and the scope its wave.
        self.assertGreater(self.ink(panel.spectrum.bars, theme.INSTRUMENT), 1000)
        self.assertGreater(self.ink(panel.scope.rect, theme.INSTRUMENT), panel.scope.rect[2] - 3)
        # The rows do not all fit: the scroll bar down the side of the body has a thumb, at its upper end.
        thumb = panel.along.thumb()
        self.assertEqual((thumb[:3], self.ink(panel.along.rect, theme.LINE)),
                         (panel.along.rect[:3], thumb[2] * thumb[3]))
        self.assertLess(thumb[3], panel.along.rect[3])


if __name__ == "__main__":
    unittest.main()
