"""Tests for rowsynth.gui.soundpanel: what the middle of the window shows of a picked sound, and what it saves."""
from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import threading
import unittest
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from rowsynth.bank import entry_kind
from rowsynth.errors import RowSynthError
from rowsynth.gui import layout, soundview, sprites, theme
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.font import Font
from rowsynth.gui.model import Model
from rowsynth.gui.session import HOLD, POLL_EVERY, Session
from rowsynth.gui.kit import told
from rowsynth.gui.soundpanel import HELP, HOLD_HELP, MEANWHILE, SoundPanel, arrange, facts, length, words, zone
from rowsynth.gui.ui import Event, Ui
from rowsynth.gui.wave import PLUS_GAP, VELOCITY_DONE, WHEEL_HELP, on_tick, places, spans
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
RATE = 44100

#: The panel in a window, with room round it on the surface it is drawn on: its heading is (24, 14, 392, 17), its
#: bar (24, 33, 392, 40), its view (24, 75, 392, 127), and at its foot its lane (24, 204, 392, 24) and the box of
#: its wave (24, 230, 392, 46).
RECT = (20, 10, 400, 270)
#: The panel as the middle of the art surface has it: its bar is (4, 23, 654, 40) and its view (4, 65, 654, 337).
WIDE = (0, 0, 662, 480)
SIZE = (440, 290)
OUTSIDE = theme.GOOD

#: A sound that names an instrument its project does not have, and what the engine says of it.
BROKEN = {"def": {"voices": [{"inst": "nope", "notes": [[0, "c4", 4]]}]}}
REFUSAL = "def: voices[0].inst: no instrument named 'nope'"

#: A held sound: a start of a tenth of a second, a loop of half a second, and an end of a tenth.
HUM = {"held": {"note": {"voices": [{"inst": "bass", "notes": [[0, "c3", 60]]}]}, "start": 6, "loop": 30, "join": 3,
                "end": {"voices": [{"inst": "bass", "notes": [[0, "c3", 2]]}]}, "let_go": 4}, "db": -6}
START, LOOP, END = 0.1, 0.5, 0.1

#: The notes of a sound of two voices: c4 and then e4 in the first, and between them g4 in the second.
FIRST = [[0, "c4", 6], [12, "e4", 6]]
SECOND = [[6, "g4", 6]]


def duo(first: list, second: list) -> dict:
    """The definition of a sound of two voices with some notes in each: the first plays tone and the second buzz."""
    return {"voices": [{"inst": "tone", "notes": first}, {"inst": "buzz", "notes": second}]}


def solo(*notes: list) -> dict:
    """The definition of a sound of one voice, which plays tone, with some notes."""
    return {"voices": [{"inst": "tone", "notes": list(notes)}]}


#: A project for a hand to change. Its instruments do not fade, so a sound ends two ticks after its notes. It has
#: a plain sound of two voices, one of two variants with a level each, and a held one.
TWO = {"def": duo(FIRST, SECOND)}
PAIR = {"defs": [solo([0, "c4", 6]), solo([0, "c5", 12])], "db": [0, -12]}
DRONE = {"held": {"note": solo([0, "c3", 60]), "start": 6, "loop": 30, "join": 3, "end": solo([0, "c3", 2]),
                  "let_go": 4}, "db": -6}
OWN = {"instruments": {"tone": {"wave": "tri", "release": 0}, "buzz": {"wave": "saw", "release": 0}},
       "sounds": {"two": TWO, "pair": PAIR, "drone": DRONE}}
#: A project of one sound of each kind: a plain one of 8 ticks, three variants with a level each, two parts of
#: which the second starts a tenth of a second in, and the held one, whose note has room for 62 ticks.
TRIO = {"defs": [solo([0, "c4", 6]), solo([0, "e4", 6]), solo([0, "g4", 6])], "db": [0, -3, -6]}
BOTH = {"parts": [[0, {"def": solo([0, "c4", 10]), "db": -3}], [0.1, {"def": solo([0, "c5", 6])}]]}
MIXED = {"instruments": OWN["instruments"],
         "sounds": {"one": {"def": solo([0, "c4", 6])}, "trio": TRIO, "both": BOTH, "drone": DRONE}}
#: A project with a sound whose one voice has its instrument written in place. Kept with the sound are two notes
#: of its maker's own, of which one is no text, and it is expected to play one note.
WRITTEN = {"wave": "saw", "release": 0}
KEPT = {"def": {"voices": [{"inst": WRITTEN, "notes": [[0, "c4", 6]]}]}, "meta": {"by": "hand", "take": 3},
        "check": [{"at": 0.05, "note": "c4"}]}
EXTRA = {"instruments": OWN["instruments"], "sounds": {"own": KEPT, "two": TWO}}
#: The time the first write from outside is stamped with, in nanoseconds: none the file gets by being saved.
STAMP = 1_700_000_000 * 10**9


def with_hum() -> dict:
    """What the file of the arcade example holds, with the held sound among its sounds."""
    data = json.loads((EXAMPLES / "arcade.json").read_text(encoding="utf-8"))
    data["sounds"]["hum"] = HUM
    return data


def render(*buffers: np.ndarray) -> dict:
    """The render of a sound or an instrument, as a model gives one: the first of its buffers is its one side."""
    return {"sides": list(buffers[:1]), "rate": RATE, "loop": False, "seconds": len(buffers[0]) / RATE,
            "buffers": list(buffers)}


def silence(*seconds: float) -> dict:
    """A render of silence: a buffer for each of some lengths."""
    return render(*(np.zeros(round(length * RATE), dtype=np.float32) for length in seconds))


class Held:
    """A stand-in for the model: the project of the arcade example, what is picked, and a render the test hands in.

    `made` is the render of the picked thing, or None while it is not ready. `refused` is what the engine says
    against rendering it, and "" while it says nothing.
    """

    def __init__(self) -> None:
        self.rate = RATE
        self.project = Project.load(EXAMPLES / "arcade.json")
        self.picked: tuple[str, str] | None = ("sound", "coin")
        self.made: dict | None = None
        self.refused = ""

    def render_ready(self) -> bool:
        return self.made is not None or bool(self.refused)

    def render(self) -> dict:
        if self.refused:
            raise RowSynthError(self.refused)
        # A render that is not ready is waited for, which a panel never does.
        assert self.made is not None, "the panel asked for a render that is not ready"
        return self.made


class Stand:
    """A stand-in for the session: a model, whether the picked thing plays and how far into it, and what it was asked.

    `lets_go` is whether a release stops what plays, as the session stops a held sound that it lets go.
    """

    def __init__(self) -> None:
        self.model = Held()
        self.playing = False
        self.at = 0.0
        self.asked: list[tuple] = []
        self.lets_go = True

    def position(self) -> float:
        return self.at

    def play(self, which: int | None = None) -> None:
        self.asked.append(("play", which))
        self.playing = True

    def hold(self) -> None:
        self.asked.append(("hold",))
        self.playing = True

    def release(self) -> None:
        self.asked.append(("release",))
        self.playing = self.playing and not self.lets_go


class Speaker:
    """A stand-in for the player: it sounds nothing, and is as far into what it was given as a test says."""

    def __init__(self) -> None:
        self.playing = False
        self.at = 0.0
        self.asked: list[str] = []
        # The sides of each thing it was asked to play, and of each note it was asked to sound over that.
        self.played: list[Sequence] = []
        self.notes: list[Sequence] = []

    def play(self, sides: Sequence, loop: bool = False) -> None:
        self.asked.append("play")
        self.played.append(sides)
        self.playing = True

    def also(self, sides: Sequence) -> None:
        self.asked.append("also")
        self.notes.append(sides)

    def stop(self, fade: float = 0.0) -> None:
        self.asked.append("stop")
        self.playing = False

    def position(self) -> float:
        return self.at if self.playing else 0.0


def panel_on(rect: layout.Rect = RECT, picked: tuple[str, str] | None = ("sound", "coin"),
             made: dict | None = None) -> tuple[Stand, Ui, SoundPanel]:
    """A stand-in session, a Ui, and the panel made with the two, placed in a rectangle and brought up to date.

    The project is the arcade example with the held sound added. `picked` is what is picked in it, and `made`
    the render of that.
    """
    stand, ui = Stand(), Ui()
    stand.model.project = Project.from_dict(with_hum())
    stand.model.picked, stand.model.made = picked, made
    panel = SoundPanel(stand, ui)
    panel.place(rect)
    panel.update()
    return stand, ui, panel


def press(ui: Ui, pos: tuple[int, int], button: int = 1) -> bool:
    """Press a mouse button at a position and let it go there. Whether the press was used."""
    used = ui.handle(Event("down", pos, button=button))
    ui.handle(Event("up", pos, button=button))
    return used


def centre(rect: layout.Rect) -> tuple[int, int]:
    """The pixel in the middle of a rectangle."""
    return rect[0] + rect[2] // 2, rect[1] + rect[3] // 2


def choose(ui: Ui, text: str) -> None:
    """Click the item of the menu that is open whose text begins with some words."""
    menu = ui.modal
    press(ui, centre(menu.rows[[item is not None and item.text.startswith(text) for item in menu.items].index(True)]))


def lies_in(rect: layout.Rect, bounds: layout.Rect) -> bool:
    """Whether a rectangle has no pixel outside another, and neither a width nor a height below 0."""
    x, y, w, h = rect
    return (w >= 0 and h >= 0 and bounds[0] <= x and x + w <= bounds[0] + bounds[2]
            and bounds[1] <= y and y + h <= bounds[1] + bounds[3])


class Sums(unittest.TestCase):
    def test_a_place_in_the_row_of_a_held_sound_is_a_zone_and_a_place_in_that_zone(self) -> None:
        lengths = (100, 400, 50)
        # In the start, and exactly where the loop begins.
        self.assertEqual([zone(at, lengths) for at in (0, 60, 99.5, 100)], [(0, 0), (0, 60), (0, 99.5), (1, 0)])
        # Played through once, the loop goes round once: where it ends the end begins, which is the tail.
        self.assertEqual([zone(at, lengths) for at in (300, 499, 500, 530)], [(1, 200), (1, 399), (2, 0), (2, 30)])
        # Held for three rounds: in the first, in the later ones, exactly where one comes round, and in the tail.
        self.assertEqual([zone(at, lengths, 3) for at in (300, 499, 500, 700, 900, 1299, 1300, 1320)],
                         [(1, 200), (1, 399), (1, 0), (1, 200), (1, 0), (1, 399), (2, 0), (2, 20)])
        # In seconds as in samples.
        self.assertEqual((zone(0.75, (0.25, 0.5, 0.25), 2), zone(1.5, (0.25, 0.5, 0.25), 2)), ((1, 0.0), (2, 0.25)))

    def test_the_words_of_the_heading_are_what_the_entry_of_a_sound_is_and_how_many_it_holds(self) -> None:
        self.assertEqual([words(kind, count) for kind, count in (("defs", 3), ("defs", 1), ("parts", 2),
                                                                 ("parts", 1), ("held", 3), ("def", 1))],
                         ["3 variants", "1 variant", "2 parts", "1 part", "held", "single"])
        stand, _, panel = panel_on()
        sounds, instruments = stand.model.project.sounds, stand.model.project.instruments
        # After them the heading says how long the sound is: of variants the longest, and of anything else
        # where the last of its sounds ends.
        for picked, kind, said, long in ((("sound", "blip"), "defs", "3 variants", "0.1 s"),
                                         (("sound", "explosion"), "parts", "2 parts", "0.65 s"),
                                         (("sound", "hum"), "held", "held", "0.7 s"),
                                         (("sound", "coin"), "def", "single", "0.38 s")):
            stand.model.picked = picked
            panel.update()
            shapes = soundview.shapes(sounds[picked[1]], RATE, instruments)
            self.assertEqual((panel.kind, panel.words, panel.head.kind, panel.head.name, panel.head.facts),
                             (kind, said, "SOUND", picked[1], f"{said}, {long}"), picked)
            self.assertEqual((facts(kind, shapes), length(kind, shapes) > 0), (panel.head.facts, True))
        self.assertEqual(length("def", []), 0.0)
        # What is no sound has no such words.
        for picked in (("instrument", "chime"), ("song", "tune"), None):
            stand.model.picked = picked
            panel.update()
            self.assertEqual((panel.kind, panel.words), ("def", ""), picked)
        self.assertEqual((panel.head.kind, panel.head.facts), ("INSTRUMENT", ""))
        # A sound the engine refuses has no words but the refusal.
        stand.model.project = SimpleNamespace(sounds={"coin": BROKEN}, instruments={})
        stand.model.picked = ("sound", "coin")
        panel.update()
        self.assertEqual((panel.kind, panel.words, panel.refusal), ("def", "", REFUSAL))


class Shown(unittest.TestCase):
    def test_the_parts_are_laid_as_every_editor_has_them_and_are_inside_the_panel_at_any_size(self) -> None:
        self.assertEqual(arrange(WIDE), {"head": (4, 4, 654, 17), "bar": (4, 23, 654, 40), "view": (4, 65, 654, 337),
                                         "lane": (4, 404, 654, 24), "wave": (4, 430, 654, 46)})
        _, ui, panel = panel_on()
        view, places = panel.view, arrange(RECT)
        # The heading and its buttons are the panel's first widgets, the view and its parts the last, and those
        # of the bar stand between them.
        self.assertEqual((ui.widgets, panel.widgets[:5], panel.widgets[-6:]),
                         (panel.widgets, [panel.head, panel.kinds, panel.limits, panel.notes, panel.hold],
                          [view, *view.parts]))
        self.assertEqual((panel.head.rect, view.rect, view.foot, view.lane.rect, view.help),
                         (places["head"], places["view"], places["wave"], places["lane"], HELP))
        self.assertEqual((HELP, places["view"], places["lane"], places["wave"]),
                         (told(wheel=WHEEL_HELP), (24, 75, 392, 127), (24, 204, 392, 24), (24, 230, 392, 46)))
        # The buttons that act on the whole sound stand at the right end of the heading, each as wide as its
        # word, and the three that open a list have the arrow of one. A sound that is not held has no HOLD.
        buttons = (panel.kinds, panel.limits, panel.notes)
        self.assertEqual([(button.text, button.arrow, button.visible, bool(button.help)) for button in buttons],
                         [("KIND", True, True, True), ("LIMITS", True, True, True), ("NOTES", True, True, True)])
        self.assertEqual((panel.hold.visible, panel.hold.text, panel.hold.help), (False, "HOLD", HOLD_HELP))
        rects = [button.rect for button in buttons]
        self.assertEqual(([rect[1::2] for rect in rects], rects[2][0] + rects[2][2]), ([(16, 13)] * 3, 24 + 392 - 2))
        self.assertTrue(rects[0][0] + rects[0][2] < rects[1][0] and rects[1][0] + rects[1][2] < rects[2][0])
        # The keyboard, the scroll bars and the lane of the view are seen with it, and the bar with its voices.
        self.assertEqual(([part.visible for part in view.parts[:4]], view.keys.rect[:3], view.lane.span[0]),
                         ([True] * 4, (24, 85, 40), 64))
        self.assertEqual((panel.bar.voices in panel.widgets, panel.bar.voices.visible), (True, True))
        self.assertTrue(lies_in(panel.bar.echo.rect, places["bar"]))
        for rect in (RECT, (40, 10, 60, 30), (0, 22, 3, 2)):
            panel.place(rect)
            self.assertEqual((panel.view.rect, panel.head.rect), (arrange(rect)["view"], arrange(rect)["head"]))
            self.assertTrue(all(lies_in(place, rect) for place in arrange(rect).values()), rect)
            self.assertTrue(all(lies_in(widget.rect, rect) for widget in panel.widgets if widget.visible), rect)

    def test_a_sound_is_its_bars_over_an_empty_wave_until_its_render_is_ready(self) -> None:
        stand, _, panel = panel_on()
        view = panel.view
        shape = soundview.shapes(stand.model.project.sounds["coin"], RATE, stand.model.project.instruments)[0][2]
        self.assertEqual((panel.heading, panel.refusal, panel.colour, view.visible),
                         (("SOUND", "coin"), "", theme.SOUND, True))
        # The two notes of coin are there, on a clock that has a second of room after the sound, and there is no
        # wave.
        seen = view.map
        x, y, _, _ = seen.bar(shape.bars[1].start, shape.bars[1].seconds, shape.bars[1].pitch)
        self.assertEqual((view.seconds, seen.notes, view.bar_at((x + 1, y + 1)), view.columns()),
                         (shape.seconds + 1, True, shape.bars[1], []))
        # The render comes: the wave is its buffer, in the colour of a sound.
        stand.model.made = render(np.full(8820, 0.25, dtype=np.float32))
        panel.update()
        (left, columns), = view.columns()
        self.assertEqual((left, len(columns), view.colour, view.bar_at((x + 1, y + 1))),
                         (seen.x(0.0), seen.x(0.2) - seen.x(0.0), theme.SOUND, shape.bars[1]))
        self.assertTrue(np.all(columns == 0.25))
        # A render the engine refuses leaves the bars over an empty wave.
        stand.model.made, stand.model.refused = None, "the sound is too long"
        panel.update()
        self.assertEqual((view.columns(), view.map.notes, panel.refusal), ([], True, ""))

    def test_an_instrument_is_the_wave_of_its_test_note_with_no_bars(self) -> None:
        stand, _, panel = panel_on()
        stand.model.picked = ("instrument", "chime")
        panel.update()
        view = panel.view
        self.assertEqual((panel.heading, panel.colour, view.visible, view.map.notes, view.seconds),
                         (("INSTRUMENT", "chime"), theme.INSTRUMENT, True, False, 0.0))
        stand.model.made = render(np.full(4410, 0.5, dtype=np.float32))
        panel.update()
        (left, columns), = view.columns()
        self.assertEqual((view.seconds, view.colour, view.map.notes, left, len(columns)),
                         (0.1, theme.INSTRUMENT, False, view.map.wave[0], view.map.wave[2]))
        self.assertTrue(np.all(columns == 0.5))

    def test_a_song_or_nothing_shows_nothing_and_a_panel_taken_off_keeps_its_view_off(self) -> None:
        stand, ui, panel = panel_on()
        stand.playing = True
        for picked in (("song", "tune"), None):
            stand.model.picked = picked
            panel.update()
            self.assertEqual((panel.heading, panel.showing, panel.view.visible, panel.view.playhead),
                             (None, False, False, None))
            # The view takes no mouse, and brought back the panel still has nothing to show.
            self.assertFalse(ui.handle(Event("wheel", (200, 100), wheel=1, mods=frozenset({"ctrl"}))))
            panel.show(True)
            self.assertFalse(panel.view.visible)
        stand.model.picked = ("sound", "coin")
        panel.update()
        self.assertTrue(panel.view.visible)
        # Taken off the window, the view stays off whatever is picked, and comes back with the panel.
        panel.show(False)
        panel.update()
        self.assertEqual((panel.heading, panel.view.visible), (("SOUND", "coin"), False))
        panel.show(True)
        self.assertTrue(panel.view.visible)

    def test_shapes_are_made_again_only_for_another_pick_or_another_project(self) -> None:
        with mock.patch.object(soundview, "shapes", wraps=soundview.shapes) as made:
            stand, ui, panel = panel_on()
            view = panel.view
            # Frames go by, the render comes, the sound plays and the view is zoomed in: nothing is made again.
            panel.update()
            stand.model.made = render(np.zeros(4410, dtype=np.float32))
            panel.update()
            stand.playing, stand.at = True, 0.05
            ui.handle(Event("wheel", (200, 100), wheel=3, mods=frozenset({"ctrl"})))
            panel.update()
            self.assertEqual((made.call_count, view.zoom), (1, 3))
            # The file is loaded again: the same sound is made anew, and the view stays. Until the render of it
            # has come the wave is the one of before, and stale.
            stand.model.project, stand.model.made = Project.load(EXAMPLES / "arcade.json"), None
            panel.update()
            panel.update()
            self.assertEqual((made.call_count, view.zoom, view.stale, len(view.columns())), (2, 3, True, 1))
            stand.model.made = render(np.zeros(4410, dtype=np.float32))
            panel.update()
            self.assertEqual((made.call_count, view.zoom, view.stale, len(view.columns())), (2, 3, False, 1))
            # Another sound is made, and is shown whole.
            stand.model.picked = ("sound", "blip")
            panel.update()
            panel.update()
            self.assertEqual((made.call_count, view.zoom, panel.heading, view.map.labels),
                             (3, 0, ("SOUND", "blip"), True))
            made.assert_called_with(stand.model.project.sounds["blip"], RATE, stand.model.project.instruments)
            # An instrument has no shapes to make.
            stand.model.picked = ("instrument", "tick")
            panel.update()
            self.assertEqual(made.call_count, 3)

    def test_the_playhead_follows_the_session_while_it_plays_and_is_gone_when_it_stops(self) -> None:
        stand, _, panel = panel_on()
        self.assertIsNone(panel.view.playhead)
        stand.playing = True
        for at in (0.0, 0.2, 0.31):
            stand.at = at
            panel.update()
            self.assertEqual(panel.view.playhead, at)
        # A frame that is told the seconds since the one before goes by the position of the session all the same.
        stand.at = 0.4
        self.assertEqual((panel.update(0.016), panel.view.playhead), (None, 0.4))
        stand.playing = False
        panel.update()
        self.assertIsNone(panel.view.playhead)

    def test_a_sound_the_engine_refuses_shows_what_it_says_in_place_of_the_view(self) -> None:
        stand, _, panel = panel_on()
        instruments = stand.model.project.instruments
        stand.model.project = SimpleNamespace(sounds={"coin": BROKEN}, instruments=instruments)
        stand.playing = True
        panel.update()
        self.assertEqual((panel.heading, panel.refusal, panel.showing, panel.view.visible, panel.view.playhead),
                         (("SOUND", "coin"), REFUSAL, False, False, None))
        # The file is put right outside: the view is back.
        stand.model.project = Project.load(EXAMPLES / "arcade.json")
        panel.update()
        self.assertEqual((panel.refusal, panel.view.visible, panel.view.map.notes), ("", True, True))


class Heard(unittest.TestCase):
    def test_a_variant_is_picked_by_a_press_and_played_from_its_menu_with_the_playhead_in_its_stretch(self) -> None:
        # Three variants of a tenth of a second, the second from 0.12 s and the third from 0.24 s.
        stand, ui, panel = panel_on(picked=("sound", "blip"), made=silence(0.1, 0.1, 0.1))
        view = panel.view
        seen = view.map
        second = (seen.x(0.12) + 5, seen.names[1] + 2)

        def play(label: tuple[int, int]) -> None:
            """Open the menu of the variant a label is of with the right button, and choose what plays it."""
            press(ui, label, 3)
            choose(ui, "Play")

        self.assertEqual((view.kind, view.played, len(view.columns())), ("defs", None, 3))
        # A press on the label of the second picks it: the hand is in it, the bar is of it, and nothing plays.
        # So it is with a press in the stretch of the third's wave.
        self.assertTrue(press(ui, second))
        panel.update()
        self.assertEqual((stand.asked, view.sound, panel.bar.index, view.played), ([], 1, 1, None))
        self.assertTrue(press(ui, (seen.x(0.24) + 30, seen.wave[1] + 10)))
        panel.update()
        self.assertEqual((stand.asked, view.sound, panel.bar.word, stand.playing), ([], 2, "VARIANT 3", False))
        # Its menu plays the second, and it is the one that is marked.
        play(second)
        self.assertEqual(stand.asked, [("play", 1)])
        for at in (0.0, 0.04, 0.1):
            stand.at = at
            panel.update()
            self.assertEqual(view.played, 1)
            self.assertAlmostEqual(view.playhead, 0.12 + at)
        # Its playhead never leaves its stretch, and its mark stays when it has ended.
        stand.at = 0.11
        panel.update()
        self.assertEqual((view.played, view.playhead), (1, None))
        stand.playing = False
        panel.update()
        self.assertEqual((view.played, view.playhead), (1, None))
        play((seen.x(0.24) + 5, seen.names[1] + 2))
        stand.at = 0.02
        panel.update()
        self.assertEqual((stand.asked, view.played), ([("play", 1), ("play", 2)], 2))
        self.assertAlmostEqual(view.playhead, 0.26)
        # A play the panel did not ask for is of a variant the session does not name: nothing is marked. That
        # is so when it begins after the chosen one has ended, and when it begins while that still sounds.
        for playing in (False, True):
            play(second)
            stand.at = 0.05
            panel.update()
            self.assertEqual(view.played, 1)
            stand.playing = playing
            panel.update()
            stand.playing, stand.at = True, 0.01
            panel.update()
            self.assertEqual((view.played, view.playhead), (None, None), playing)
        # Another sound is picked, and then this one again: nothing of it was played yet.
        play(second)
        for picked, made, played in ((("sound", "blip"), stand.model.made, 1), (("sound", "coin"), None, None),
                                     (("sound", "blip"), stand.model.made, None)):
            stand.model.picked, stand.model.made = picked, made
            panel.update()
            self.assertEqual(view.played, played, picked)

    def test_the_hold_button_is_there_for_a_held_sound_alone(self) -> None:
        stand, ui, panel = panel_on()
        for picked, there in ((("sound", "coin"), False), (("sound", "blip"), False), (("sound", "explosion"), False),
                              (("instrument", "bass"), False), (("song", "tune"), False), (None, False),
                              (("sound", "hum"), True)):
            stand.model.picked = picked
            panel.update()
            self.assertEqual(panel.hold.visible, there, picked)
        # It is the last of the buttons of the heading, after the three that every sound has.
        on = centre(panel.hold.rect)
        self.assertEqual((panel.hold.rect[0] > panel.notes.rect[0], press(ui, on), stand.asked),
                         (True, True, [("hold",), ("release",)]))
        # Off the window the button is off as the view is, and it comes back with the panel.
        panel.show(False)
        panel.update()
        self.assertEqual((panel.hold.visible, panel.view.visible, press(ui, on)), (False, False, False))
        panel.show(True)
        self.assertEqual((panel.hold.visible, panel.view.visible), (True, True))

    def test_the_hold_button_holds_from_its_press_to_its_release_also_with_the_mouse_off_it(self) -> None:
        stand, ui, panel = panel_on(picked=("sound", "hum"))
        on = centre(panel.hold.rect)
        ui.handle(Event("move", on))
        self.assertEqual((ui.help, stand.asked), (HOLD_HELP, []))
        # The press holds the sound at once. The mouse leaves the button, and the release lets go all the same.
        self.assertTrue(ui.handle(Event("down", on, button=1)))
        self.assertEqual(stand.asked, [("hold",)])
        ui.handle(Event("move", (100, 200)))
        self.assertEqual((stand.asked, ui.held), ([("hold",)], panel.hold))
        ui.handle(Event("up", (100, 200), button=1))
        self.assertEqual((stand.asked, ui.held), ([("hold",), ("release",)], None))
        # Released on the button it is the same, and the other buttons of the mouse do nothing.
        self.assertTrue(press(ui, on))
        self.assertEqual((press(ui, on, 2), press(ui, on, 3)), (False, False))
        self.assertEqual(stand.asked, [("hold",), ("release",)] * 2)

    def test_a_held_sound_that_is_played_through_once_goes_from_its_start_to_its_loop_to_its_end(self) -> None:
        stand, ui, panel = panel_on(picked=("sound", "hum"), made=silence(START, LOOP, END))
        view = panel.view
        # The clock has a second of room after the 0.7 s of the sound.
        self.assertEqual((view.kind, len(view.columns()), round(view.seconds, 9)), ("held", 3, 1.7))
        # Something else plays it, as the space bar does: the row is its start, its loop once, and its end.
        stand.playing = True
        for at, second in ((0.0, 0.0), (0.06, 0.06), (START + 0.3, START + 0.3), (START + LOOP + 0.04, 0.64)):
            stand.at = at
            panel.update()
            self.assertAlmostEqual(view.playhead, second)
        stand.playing = False
        panel.update(0.016)
        self.assertIsNone(view.playhead)
        # A hold that is let go before its render has come is played through once as well.
        stand.model.made, stand.lets_go = None, False
        panel.update()
        ui.handle(Event("down", centre(panel.hold.rect), button=1))
        panel.update()
        self.assertEqual((stand.asked, stand.playing, view.playhead), ([("hold",)], True, None))
        ui.handle(Event("up", centre(panel.hold.rect), button=1))
        stand.model.made, stand.at = silence(START, LOOP, END), START + LOOP + 0.04
        panel.update()
        self.assertAlmostEqual(view.playhead, 0.64)


@unittest.skipUnless(pygame, NEEDS)
class Drawn(unittest.TestCase):
    def setUp(self) -> None:
        self.surface = pygame.Surface(SIZE)
        self.surface.fill(OUTSIDE)
        self.canvas = Canvas(self.surface, Font(sprites.load("font")))
        self.stand, self.ui, self.panel = panel_on()

    def colours(self, rect: layout.Rect) -> set[tuple[int, ...]]:
        """The colours a rectangle has in it."""
        x, y, w, h = rect
        return {tuple(self.surface.get_at((x + column, y + row)))[:3] for column in range(w) for row in range(h)}

    def written(self, colour: tuple[int, ...], rect: layout.Rect, text: str, left: int) -> bool:
        """Whether what a rectangle has in a colour is a text that starts at an x: all of it, and nothing else.

        The first letter and the last can leave up to three columns of their own room free.
        """
        x, y, w, h = rect
        inked = [x + column for column in range(w)
                 if any(tuple(self.surface.get_at((x + column, y + row)))[:3] == colour for row in range(h))]
        right = left + Font.width(text)
        return bool(inked) and left <= inked[0] <= left + 3 and right - 3 <= inked[-1] + 1 <= right

    def draw(self) -> None:
        """Bring the panel up to date and draw a frame of it: the panel, and over it its view."""
        self.surface.fill(OUTSIDE)
        self.panel.update()
        self.panel.draw(self.canvas)
        self.ui.draw(self.canvas)

    def test_the_heading_is_the_kind_and_the_name_in_its_colour_and_a_refusal_stands_in_the_middle(self) -> None:
        places = arrange(RECT)
        x, y, w, h = places["head"]
        for picked, colour in ((("sound", "coin"), theme.SOUND), (("instrument", "chime"), theme.INSTRUMENT)):
            self.stand.model.picked = picked
            self.draw()
            kind, name = picked[0].upper(), picked[1]
            # The kind starts 6 pixels in, and the name 6 pixels after it. The heading has its outline.
            self.assertTrue(self.written(theme.DIM, (30, y, Font.width(kind), h), kind, 30))
            self.assertTrue(self.written(colour, places["head"], name, 30 + Font.width(kind) + 6))
            # A sound has the three buttons that open a list there as well, raised.
            buttons = {theme.RAISED, theme.LIT, theme.TEXT} if picked[0] == "sound" else set()
            self.assertEqual(self.colours(places["head"]), {theme.PANEL, theme.LINE, theme.DIM, colour} | buttons)
            self.assertEqual(self.colours((x, y, w, 1)) | self.colours((x, y + 1, 1, h - 2)), {theme.LINE})
            # The view is drawn in its room, and nothing is outside the panel.
            self.assertIn(theme.BACKGROUND, self.colours(places["view"]))
            around = ((0, 0, 440, 10), (0, 280, 440, 10), (0, 10, 20, 270), (420, 10, 20, 270))
            self.assertEqual(set().union(*(self.colours(rect) for rect in around)), {OUTSIDE})
        # A sound has its settings bar, its lane and the box of its wave under one another, each with its
        # outline, and the keyboard of every view of notes down the left of its grid.
        self.stand.model.picked = ("sound", "coin")
        self.draw()
        for part in ("bar", "lane", "wave"):
            left, top, wide, _ = places[part]
            self.assertEqual(self.colours((left, top, wide, 1)), {theme.LINE}, part)
        self.assertEqual(self.colours(self.panel.view.keys.rect) - {theme.PANEL}, {theme.TEXT, theme.BACKGROUND})
        # A song draws nothing at all.
        self.stand.model.picked = ("song", "tune")
        self.draw()
        self.assertEqual(self.colours((0, 0, *SIZE)), {OUTSIDE})
        # What the engine says against a sound is one line in the colour of a fault, as far from the left of
        # the view's room as from its right, and half way down it.
        self.stand.model.picked = ("sound", "coin")
        self.stand.model.project = SimpleNamespace(sounds={"coin": BROKEN}, instruments={})
        self.draw()
        x, y, w, h = arrange(RECT)["view"]
        self.assertTrue(self.written(theme.FAULT, (x, y, w, h), REFUSAL, x + (w - Font.width(REFUSAL)) // 2))
        self.assertEqual(self.colours((x, y, w, h // 2 - 8)) | self.colours((x, y + h // 2 + 8, w, h // 2 - 8)),
                         {theme.PANEL})
        self.assertEqual(self.colours((x, y, w, h)), {theme.PANEL, theme.FAULT})

    def test_the_heading_says_in_dim_words_what_a_sound_is_and_how_long_and_a_held_one_has_its_button(self) -> None:
        head = arrange(RECT)["head"]
        for name, said in (("blip", "3 variants, 0.1 s"), ("explosion", "2 parts, 0.65 s"), ("hum", "held, 0.7 s")):
            self.stand.model.picked = ("sound", name)
            self.draw()
            # The name stands 6 pixels after the kind, and the facts 12 pixels after the name.
            at = 30 + Font.width("SOUND") + 6
            after = at + Font.width(name) + 12
            self.assertTrue(self.written(theme.SOUND, head, name, at))
            self.assertTrue(self.written(theme.DIM, (after, head[1], Font.width(said), head[3]), said, after), name)
        # The button of the held sound is the last of the heading: raised, with its word 4 pixels in.
        hold = x, y, w, h = self.panel.hold.rect
        self.assertEqual((x + w, y, h), (head[0] + head[2] - 2, head[1] + 2, 13))
        self.assertEqual(self.colours(hold), {theme.RAISED, theme.LIT, theme.LINE, theme.TEXT})
        self.assertTrue(self.written(theme.TEXT, hold, "HOLD", x + 4))
        # Kept pressed it is sunk and its word is lit, also with the mouse off it.
        self.ui.handle(Event("down", centre(hold), button=1))
        self.ui.handle(Event("move", (100, 100)))
        self.draw()
        self.assertEqual(self.colours(hold), {theme.BACKGROUND, theme.LINE, theme.ACTIVE})
        self.ui.handle(Event("up", (100, 100), button=1))
        self.draw()
        self.assertEqual(self.colours(hold), {theme.RAISED, theme.LIT, theme.LINE, theme.TEXT})
        # In a narrow panel the words end before the first button, and do not run on under it: here the facts
        # have no room, and the name has.
        self.panel.place((20, 10, 260, 270))
        self.draw()
        first = self.panel.kinds.rect[0]
        self.assertEqual((first, self.colours((first - 12, y - 1, 12, h + 2))), (102, {theme.PANEL}))
        self.assertTrue(self.written(theme.SOUND, (24, head[1], first - 24, head[3]), "hum",
                                     30 + Font.width("SOUND") + 6))
        self.panel.place(RECT)
        # A sound that is not held has no such button: the three that open a list stand at the right end.
        self.stand.model.picked = ("sound", "blip")
        self.draw()
        notes = self.panel.notes.rect
        self.assertEqual((self.panel.hold.visible, notes[0] + notes[2]), (False, x + w))


class OnAModel(unittest.TestCase):
    def setUp(self) -> None:
        def settle() -> None:
            """Waits for every render that is still on a thread of the model's."""
            for thread in threading.enumerate():
                if thread.name.startswith("rowsynth-"):
                    thread.join()

        self.addCleanup(settle)

    def test_on_a_real_model_and_session_coin_is_shown_over_the_render_the_model_made(self) -> None:
        model = Model(EXAMPLES / "arcade.json")
        session = Session(model)
        panel = SoundPanel(session, Ui())
        panel.place(RECT)
        panel.update()
        self.assertEqual((model.picked, panel.heading, panel.view.visible), (("song", "tune"), None, False))
        # The pick starts the render on a thread: the panel is brought up to date before and after it has come.
        session.pick("sound", "coin")
        panel.update()
        made = model.render()
        panel.update()
        view = panel.view
        (left, columns), = view.columns()
        # The clock has a second of room after the sound, and the wave ends where the sound does.
        seen = view.map
        self.assertEqual((panel.heading, view.seconds, view.playhead, left, len(columns)),
                         (("SOUND", "coin"), made["seconds"] + 1, None, seen.wave[0],
                          seen.x(made["seconds"]) - seen.wave[0]))
        self.assertLess(len(columns), seen.wave[2])
        self.assertTrue(np.array_equal(columns, soundview.columns(made["buffers"][0], len(columns))))
        self.assertGreater(float(np.abs(columns).max()), 0.5)

    def test_blip_and_explosion_of_the_arcade_example_have_as_many_waves_as_their_renders_have_buffers(self) -> None:
        model = Model(EXAMPLES / "arcade.json")
        session = Session(model)
        panel = SoundPanel(session, Ui())
        panel.place(RECT)
        view = panel.view
        for name, said, count, lefts in (("blip", "3 variants", 3, (0.0, 0.12, 0.24)),
                                         ("explosion", "2 parts", 1, (0.0,))):
            session.pick("sound", name)
            buffers = model.render()["buffers"]
            panel.update()
            waves = view.columns()
            self.assertEqual((panel.words, len(buffers), len(waves), view.visible), (said, count, count, True))
            # Each wave is its own buffer, from where its variant starts: the one of the parts from second 0.
            self.assertEqual([left for left, _ in waves], [view.map.x(second) for second in lefts])
            for (_, columns), buffer in zip(waves, buffers):
                self.assertTrue(np.array_equal(columns, soundview.columns(buffer, len(columns))))
                self.assertGreater(float(np.abs(columns).max()), 0.1)

    def test_a_held_sound_of_a_project_goes_along_its_start_round_its_loop_and_let_go_through_its_end(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "held.json"
        path.write_text(json.dumps(with_hum()), encoding="utf-8")
        speaker, ui = Speaker(), Ui()
        session = Session(Model(path), speaker)
        session.audition = False
        panel = SoundPanel(session, ui)
        panel.place(RECT)
        session.pick("sound", "hum")
        self.assertEqual([len(row) / RATE for row in session.model.render()["buffers"]], [START, LOOP, END])
        session.tick()
        panel.update()
        view, on = panel.view, centre(panel.hold.rect)
        self.assertEqual((panel.words, panel.hold.visible, view.kind, len(view.columns())), ("held", True, "held", 3))

        def at(second: float, passed: float = 0.0) -> float | None:
            """The playhead of a frame in which the player is some seconds into what it plays."""
            speaker.at = second
            panel.update(passed)
            return view.playhead

        # Held: along the start, and then round the loop for as long as the button is kept pressed.
        ui.handle(Event("down", on, button=1))
        rounds = math.ceil(HOLD / LOOP)
        self.assertEqual((speaker.asked, session.playing, rounds), (["stop", "play"], True, 20))
        for second, shown in ((0.04, 0.04), (START, START), (START + 0.2, START + 0.2), (START + 7 * LOOP + 0.3, 0.4),
                              (START + (rounds - 1) * LOOP + 0.45, 0.55)):
            self.assertAlmostEqual(at(second), shown)
        # Let go, with the mouse off the button: the session plays the end over what it lets go and tells no
        # more of it, so the end is followed by the seconds of the frames, for as long as it is.
        ui.handle(Event("up", (0, 0), button=1))
        self.assertEqual((speaker.asked[2:], session.playing), (["stop", "also"], False))
        for passed, shown in ((0.0, 0.6), (0.03, 0.63), (0.05, 0.68)):
            self.assertAlmostEqual(at(0.0, passed), shown)
        self.assertEqual((at(0.0, 0.05), at(0.0, 0.05)), (None, None))
        # Held and let go again, and then played through by something else while its end still sounds: that
        # play is followed, and when it is over nothing more of the end is.
        ui.handle(Event("down", on, button=1))
        self.assertAlmostEqual(at(START + 0.2), START + 0.2)
        ui.handle(Event("up", on, button=1))
        self.assertAlmostEqual(at(0.0, 0.02), 0.62)
        session.play()
        self.assertAlmostEqual(at(START + LOOP + 0.05), 0.65)
        speaker.playing = False
        self.assertIsNone(at(0.0, 0.01))
        # Kept pressed for longer than the session holds a sound, it is let go by itself: its end follows the
        # last round of its loop, and the release that comes then changes nothing.
        ui.handle(Event("down", on, button=1))
        self.assertAlmostEqual(at(START + rounds * LOOP - 0.01), START + LOOP - 0.01)
        self.assertAlmostEqual(at(START + rounds * LOOP + 0.03), 0.63)
        ui.handle(Event("up", on, button=1))
        self.assertEqual((speaker.asked[4:], session.playing), (["play", "stop", "also", "play", "play"], True))
        self.assertAlmostEqual(at(START + rounds * LOOP + 0.06), 0.66)
        # Played through once after that, by something else, it is followed as that and no longer as a hold.
        session.play()
        self.assertAlmostEqual(at(START + LOOP + 0.05), 0.65)


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


class OnAFile(unittest.TestCase):
    """A project file of the test's own, a real session on it with a stand-in player, and the panel on that."""

    #: What the file holds at the start, and the sound of it that is picked.
    DATA, FIRST = OWN, "two"

    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        # No thread of the model is left to run into the next test, or into a folder that is gone.
        self.addCleanup(settle)
        self.folder = Path(folder.name)
        self.open(self.DATA)
        self.pick(self.FIRST)

    def open(self, data: dict | Path) -> None:
        """Open a session and the panel on a file of the folder: one that holds some data, or a copy of a file."""
        self.path = self.folder / "project.json"
        if isinstance(data, Path):
            shutil.copy(data, self.path)
        else:
            self.path.write_text(json.dumps(data), encoding="utf-8")
        self.now, self.stamp = 100.0, STAMP
        self.speaker, self.ui = Speaker(), Ui()
        self.session = Session(Model(self.path), self.speaker, lambda: self.now)
        # Nothing plays by itself, but where a test has it so.
        self.session.audition = False
        self.panel = SoundPanel(self.session, self.ui)
        self.panel.place(WIDE)
        self.view = self.panel.view

    def pick(self, name: str) -> None:
        """Pick a sound of the project, and have the panel show it over its render."""
        self.session.pick("sound", name)
        self.rendered()

    def rendered(self) -> dict:
        """Wait for the render of the picked sound, have the session and the panel take it, and give it."""
        made = self.session.model.render()
        self.session.tick()
        self.panel.update()
        return made

    def data(self) -> dict:
        """What the file holds now."""
        return json.loads(self.path.read_text(encoding="utf-8"))

    def outside(self, change: Callable[[dict], object]) -> dict:
        """Have another program change the sounds of the project and write the file, and the session look at it."""
        project = self.data()
        change(project["sounds"])
        self.path.write_text(json.dumps(project), encoding="utf-8")
        # Stamped as no write before it was, so that the look cannot take the file for the one it knows.
        self.stamp += 10**9
        os.utime(self.path, ns=(self.stamp, self.stamp))
        self.now += POLL_EVERY
        self.session.tick()
        return project

    def at(self, tick: float, pitch: int, sound: int = 0) -> tuple[int, int]:
        """A pixel of the view in the middle of a tick of a sound of the picked entry, and of a key."""
        model = self.session.model
        entry = model.project.sounds[model.picked[1]]
        shapes = soundview.shapes(entry, model.rate, model.project.instruments)
        laid = [(label, place, shape)
                for (label, _, shape), place in zip(shapes, places(shapes, entry_kind(entry) == "defs"))]
        seen = self.view.map
        return seen.x(spans(entry, laid)[sound].at + (tick + 0.5) / 60), seen.y(pitch) + 4

    def mouse(self, kind: str, pos: tuple[int, int], button: int = 1, *mods: str) -> bool:
        """Hand the Ui what the mouse did, some keys held. Whether it was used."""
        return self.ui.handle(Event(kind, pos, button=0 if kind == "move" else button, mods=frozenset(mods)))

    def drag(self, *stops: tuple[int, int], button: int = 1, mods: tuple[str, ...] = ()) -> None:
        """Press at the first position, move to each of the others in turn, and release at the last."""
        self.mouse("down", stops[0], button, *mods)
        for stop in stops[1:]:
            self.mouse("move", stop, button, *mods)
        self.mouse("up", stops[-1], button, *mods)


class Saved(OnAFile):
    def test_a_note_placed_is_in_the_picked_voice_and_in_the_file_and_is_one_step_to_take_back(self) -> None:
        before = self.data()
        self.assertEqual((self.view.drawn, self.view.kind, self.panel.bar.voices.options),
                         (TWO["def"], "def", ("1", "2")))
        # The second voice is picked in the bar, and a click on the grid places a note in it.
        self.drag(centre(self.panel.bar.voices.cells()[1]))
        self.drag(self.at(2, 65))
        after = self.data()
        placed = duo(FIRST, [*SECOND, [2, "f4", 6]])
        # In the file it is that voice's notes that are others, and nothing else.
        self.assertEqual((differing(before, after), after["sounds"]["two"], self.view.drawn),
                         (["sounds.two.def.voices"], {"def": placed}, placed))
        self.assertEqual((self.session.undo_label, self.session.message), ("two: note placed", ""))
        # It was heard, one note of the instrument of its voice, and nothing was played.
        self.assertEqual((len(self.speaker.notes), self.speaker.played), (1, []))
        self.assertIs(self.speaker.notes[0], self.session.model.tone("buzz", 65)["sides"])
        # The step is taken back and made again, and the view shows what the file holds each time.
        self.assertTrue(self.session.undo())
        self.panel.update()
        self.assertEqual((self.data(), self.view.drawn, self.session.can_undo), (before, TWO["def"], False))
        self.assertTrue(self.session.redo())
        self.panel.update()
        self.assertEqual((self.data(), self.view.drawn, self.session.can_redo), (after, placed, False))

    def test_a_note_placed_in_a_voice_whose_instrument_is_written_in_place_is_heard_in_that_instrument(self) -> None:
        written = {"wave": "saw", "release": 0}
        settle()
        self.open({**OWN, "sounds": {"own": {"def": {"voices": [{"inst": written, "notes": [[0, "c4", 6]]}]}}}})
        self.pick("own")
        self.drag(self.at(2, 65))
        self.assertEqual((self.data()["sounds"]["own"]["def"]["voices"][0], self.session.message),
                         ({"inst": written, "notes": [[0, "c4", 6], [2, "f4", 6]]}, ""))
        self.assertEqual((len(self.speaker.notes), self.speaker.played), (1, []))
        self.assertIs(self.speaker.notes[0], self.session.model.tone(written, 65)["sides"])

    def test_a_note_placed_in_the_second_variant_changes_that_variant_alone_and_it_is_the_one_played(self) -> None:
        self.pick("pair")
        self.session.audition = True
        before = self.data()
        self.drag(self.at(2, 74, 1))
        after = self.data()
        second = solo([0, "c5", 12], [2, "d5", 6])
        self.assertEqual((differing(before, after), after["sounds"]["pair"]),
                         (["sounds.pair.defs"], {"defs": [PAIR["defs"][0], second], "db": [0, -12]}))
        self.assertEqual((self.session.undo_label, self.view.sound, self.view.drawn), ("pair: note placed", 1, second))
        # When its render has come the saved sound is played, once: its second variant, which the view marks.
        made = self.rendered()
        self.assertEqual((len(self.speaker.played), len(self.speaker.played[0]), self.view.played), (1, 1, 1))
        self.assertIs(self.speaker.played[0][0], made["buffers"][1])
        # With the audition off a save plays nothing.
        self.session.audition = False
        self.drag(self.at(9, 70, 1))
        self.rendered()
        self.assertEqual((len(self.data()["sounds"]["pair"]["defs"][1]["voices"][0]["notes"]),
                          len(self.speaker.played)), (3, 1))

    def test_the_picked_notes_and_the_zoom_stay_through_a_save_and_through_the_render_of_it(self) -> None:
        self.ui.handle(Event("wheel", self.at(10, 64), wheel=1, mods=frozenset({"ctrl"})))
        seen = (self.view.zoom, self.view.start)
        self.assertEqual((seen[0], seen[1] > 0), (1, True))
        self.drag(self.at(4, 60), mods=("ctrl",))
        self.drag(self.at(8, 67), mods=("ctrl",))
        # The two picked notes go 2 ticks on and 2 keys up, as one step.
        self.drag(self.at(8, 67), self.at(10, 69))
        moved = duo([[2, "d4", 6], FIRST[1]], [[8, "a4", 6]])
        self.assertEqual((self.data()["sounds"]["two"]["def"], self.session.undo_label),
                         (moved, "two: 2 notes moved"))
        self.assertEqual((self.view.picked, (self.view.zoom, self.view.start)), ({(0, 0), (1, 0)}, seen))
        self.rendered()
        self.assertEqual((self.view.picked, (self.view.zoom, self.view.start)), ({(0, 0), (1, 0)}, seen))
        # A change of the bar leaves the notes as they are, and so what is picked.
        self.drag(centre(self.panel.bar.echo.rect))
        self.assertEqual((self.session.undo_label, "echo" in self.view.drawn, self.view.picked, self.view.zoom),
                         ("two: echo on", True, {(0, 0), (1, 0)}, 1))

    def test_the_wave_of_before_stays_under_the_saved_notes_and_is_stale_until_their_render_has_come(self) -> None:
        before = self.session.model.render()
        self.assertEqual((self.view.stale, self.view.seconds), (False, before["seconds"] + 1))
        scale = self.view.map.scale
        # A note that ends 4 ticks after the others makes the sound longer.
        self.drag(self.at(16, 70))
        seen = self.view.map
        (left, columns), = self.view.columns()
        self.assertEqual((self.view.stale, self.view.drawn["voices"][0]["notes"][2:]), (True, [[16, "a#4", 6]]))
        self.assertEqual((left, len(columns), self.view.seconds > before["seconds"]),
                         (seen.x(0.0), seen.x(before["seconds"]) - seen.x(0.0), True))
        made = self.rendered()
        (left, columns), = self.view.columns()
        # The clock has its second of room after the new end, and a second is as wide as it was: the wave ends
        # where the longer sound does.
        seen = self.view.map
        self.assertEqual((self.view.stale, len(columns), self.view.seconds, seen.scale),
                         (False, seen.x(made["seconds"]) - seen.x(0.0), made["seconds"] + 1, scale))
        self.assertGreater(made["seconds"], before["seconds"])
        # Another sound that is picked has no wave of before to show.
        self.session.pick("sound", "pair")
        self.panel.update()
        self.assertFalse(self.view.stale)

    def test_a_change_from_outside_waits_for_a_gesture_and_the_edit_of_the_hand_then_stands_over_it(self) -> None:
        self.mouse("down", self.at(1, 60))
        self.mouse("move", self.at(3, 60))
        written = self.outside(lambda sounds: sounds["two"]["def"]["voices"][1]["notes"].append([0, "a4", 3]))
        self.panel.update()
        moved = duo([[2, "c4", 6], FIRST[1]], SECOND)
        # The view goes on drawing the notes it was shown, as the release would leave them.
        self.assertEqual((self.view.busy, self.view.drawn, self.session.message),
                         (True, moved, "changed outside: two"))
        self.mouse("up", self.at(3, 60))
        self.assertEqual((self.data()["sounds"]["two"]["def"], self.view.drawn, self.session.message),
                         (moved, moved, f"two {MEANWHILE}"))
        # The step taken back, the sound is as it was written outside.
        self.assertTrue(self.session.undo())
        self.panel.update()
        self.assertEqual((self.data(), self.view.drawn), (written, written["sounds"]["two"]["def"]))
        # With no gesture under way, what is written outside is shown by the next frame.
        self.outside(lambda sounds: sounds["two"]["def"]["voices"][1].update(notes=[]))
        self.panel.update()
        self.assertEqual((self.view.drawn, self.view.busy), (duo(FIRST, []), False))
        # A knob of the bar that is under the hand holds it back as well: let go, its volume is saved beside
        # what was written outside.
        level = self.panel.bar.volume
        x, y = centre(level.rect)
        self.mouse("down", (x, y))
        self.mouse("move", (x, y + 10))
        self.outside(lambda sounds: sounds["two"]["def"]["voices"][1].update(notes=SECOND))
        self.panel.update()
        self.assertEqual((self.panel.bar.busy, level.value, self.view.drawn), (True, -3.5, duo(FIRST, [])))
        self.mouse("up", (x, y + 10))
        self.assertEqual((self.data()["sounds"]["two"], self.view.drawn, self.session.undo_label),
                         ({"def": duo(FIRST, SECOND), "db": -3.5}, duo(FIRST, SECOND), "two: volume -3.5 dB"))

    def test_a_gesture_that_ends_after_its_sound_was_removed_outside_saves_nothing_and_says_so(self) -> None:
        start, stop = self.at(1, 60), self.at(3, 60)
        self.mouse("down", start)
        self.mouse("move", stop)
        written = self.outside(lambda sounds: sounds.pop("two"))
        self.panel.update()
        self.assertEqual((self.view.busy, "two" in self.session.model.project.sounds), (True, False))
        steps = self.session.undo_label
        self.mouse("up", stop)
        self.assertEqual((self.data(), self.session.message, self.session.undo_label, self.view.busy),
                         (written, "two: the project has no such sound any more", steps, False))

    def test_a_volume_let_go_of_after_its_sound_was_removed_outside_saves_nothing_and_says_so(self) -> None:
        x, y = centre(self.panel.bar.volume.rect)
        self.mouse("down", (x, y))
        self.mouse("move", (x, y + 10))
        written = self.outside(lambda sounds: sounds.pop("two"))
        self.panel.update()
        steps = self.session.undo_label
        self.mouse("up", (x, y + 10))
        self.assertEqual((self.data(), self.session.message, self.session.undo_label),
                         (written, "two: the project has no such sound any more", steps))

    def test_a_change_that_leaves_the_note_of_a_held_sound_too_short_is_refused_and_said_and_saves_nothing(
            self) -> None:
        self.pick("drone")
        before = self.data()
        self.assertEqual((self.view.kind, self.panel.hold.visible), ("held", True))
        # The right button on its one note would leave the start and the loop nothing to be made of.
        self.drag(self.at(1, 48), button=3)
        self.assertEqual(self.session.message, "drone: the note of a held sound must be long enough for its start and "
                                               "its loop: it is 2 ticks long, and they take 36")
        self.assertEqual((self.data(), self.session.can_undo, self.view.drawn, self.view.stale),
                         (before, False, DRONE["held"]["note"], False))
        # A note placed in its end is saved as any is: the end is the second sound of the entry.
        self.drag(self.at(1, 50, 1))
        self.assertEqual((self.data()["sounds"]["drone"]["held"]["end"], self.session.undo_label, self.view.sound),
                         (solo([0, "c3", 2], [1, "d3", 6]), "drone: note placed", 1))
        self.assertEqual(differing(before, self.data()), ["sounds.drone.held.end.voices"])

    def test_a_note_of_coin_moved_and_moved_back_leaves_the_data_of_the_arcade_example_as_it_was(self) -> None:
        settle()
        self.open(EXAMPLES / "arcade.json")
        self.pick("coin")
        before = self.data()
        # b5 goes 3 ticks on and 3 keys up, and comes back the same way: its name is spelt as it was.
        self.drag(self.at(1, 83), self.at(4, 86))
        self.assertEqual(self.data()["sounds"]["coin"]["def"]["voices"][0]["notes"], [[3, "d6", 4], [4, "e6", 14]])
        self.assertEqual(differing(before, self.data()), ["sounds.coin.def.voices"])
        self.drag(self.at(4, 86), self.at(1, 83))
        self.assertEqual((self.data(), self.session.undo_label), (before, "coin: note moved"))
        # By the keys as by the mouse: an octave up and a step on, and back.
        for key, mods in (("up", {"shift"}), ("right", set()), ("left", set()), ("down", {"shift"})):
            self.assertTrue(self.ui.handle(Event("key", key=key, mods=frozenset(mods))))
        self.assertEqual(self.data(), before)


class Menus(OnAFile):
    """A panel on a file whose menus and boxes are worked: what a test does to them, and what it reads of them."""

    def sound(self, name: str) -> dict:
        """What the file holds now for a sound."""
        return self.data()["sounds"][name]

    def items(self) -> list[tuple[str, bool]]:
        """The items of the menu that is open: the text of each, and whether it can be chosen."""
        return [(item.text, item.enabled) for item in self.ui.modal.items if item is not None]

    def choose(self, text: str) -> None:
        """Click the item of the open menu whose text begins with some words."""
        choose(self.ui, text)

    def type(self, text: str) -> None:
        """Type a text in the box that is open, in place of what it holds, and press enter."""
        self.ui.handle(Event("text", text=text) if text else Event("key", key="backspace"))
        self.ui.handle(Event("key", key="return"))

    def label(self, which: int) -> tuple[int, int]:
        """A pixel on the label of one of the shapes of the picked sound, counted from 0."""
        return centre(self.view.tab(which))

    def plus(self) -> tuple[int, int]:
        """A pixel on the sign after the last label."""
        entry = self.sound(self.session.model.picked[1])
        tabs = [self.view.tab(which) for which in range(len(entry.get("defs") or entry["parts"]))]
        return max(x + w for x, _, w, _ in tabs) + PLUS_GAP + 2, tabs[0][1] + 2

    def ruler(self, second: float) -> tuple[int, int]:
        """A pixel of the ruler at a second of the clock."""
        return self.view.map.x(second), self.view.map.ruler[1] + 2


class Kinds(Menus):
    """The kind of a sound and what belongs to the kind, changed in the window: each change is one save."""

    DATA, FIRST = MIXED, "one"

    def test_each_kind_is_made_any_other_from_the_menu_of_the_heading_and_undo_brings_the_old_kind_back(self) -> None:
        said = {"one": ["SINGLE", "VARIANTS", "PARTS", "HELD"],
                "trio": ["SINGLE (keeps variant 1 only)", "VARIANTS", "PARTS", "HELD (keeps variants 1 and 2 only)"],
                "both": ["SINGLE (keeps part 1 only)", "VARIANTS (drops where the parts start)", "PARTS",
                         "HELD (drops where the parts start)"],
                "drone": ["SINGLE (keeps the note only)", "VARIANTS (drops the loop)", "PARTS (drops the loop)",
                          "HELD"]}
        made = {"def": "made a single sound", "defs": "made variants", "parts": "made parts",
                "held": "made a held sound"}
        for name, kind in zip(said, made):
            self.pick(name)
            before = self.data()
            # The button says what it is for, and the heading what the sound is.
            self.assertEqual((self.panel.kind, self.panel.kinds.visible, self.panel.kinds.text,
                              self.panel.words.split()[-1].upper() in said["one"][list(made).index(kind)]),
                             (kind, True, "KIND", True))
            for to, text in zip(made, said[name]):
                # The button of the heading opens the menu: what the sound is cannot be chosen, and an item
                # says what the sound would let go of.
                self.drag(centre(self.panel.kinds.rect))
                self.assertEqual(self.items(), [(item, item != said[name][list(made).index(kind)])
                                                for item in said[name]])
                if to == kind:
                    self.ui.handle(Event("key", key="escape"))
                    continue
                self.choose(text)
                # The file was saved through the project's check, so it holds an entry the project accepts.
                self.assertEqual((entry_kind(Project.load(self.path).sounds[name]), self.session.undo_label,
                                  self.panel.kind, self.session.message), (to, f"{name}: {made[to]}", to, ""))
                self.assertTrue(self.session.undo())
                self.panel.update()
                self.assertEqual((self.data(), self.panel.kind), (before, kind))

    def test_a_variant_is_added_as_a_copy_of_the_one_the_hand_is_in_and_moved_and_removed_by_its_menu(self) -> None:
        self.pick("trio")
        first, second, third = TRIO["defs"]
        self.drag(self.plus())
        self.assertEqual((self.sound("trio"), self.session.undo_label, self.view.sound),
                         ({"defs": [first, second, third, first], "db": [0, -3, -6, 0]}, "trio: variant 4 added", 3))
        # The right button on its label opens the menu of a named thing, in the order every such menu has: a
        # variant has no name, and the last variant is not moved further right.
        self.drag(self.label(3), button=3)
        self.assertEqual(self.items(), [("Rename...", False), ("Copy", True), ("Play", True), ("Move left", True),
                                        ("Move right", False), ("Remove", True)])
        self.choose("Move left")
        self.assertEqual((self.sound("trio"), self.session.undo_label, self.view.sound),
                         ({"defs": [first, second, first, third], "db": [0, -3, 0, -6]},
                          "trio: variant 4 moved left", 2))
        self.drag(self.label(0), button=3)
        self.assertEqual(self.items()[3:], [("Move left", False), ("Move right", True), ("Remove", True)])
        self.choose("Remove")
        self.assertEqual((self.sound("trio"), self.session.undo_label),
                         ({"defs": [second, first, third], "db": [-3, 0, -6]}, "trio: variant 1 removed"))
        # Copied, a variant is one more at the end, with its level, and the hand is in the copy.
        self.drag(self.label(0), button=3)
        self.choose("Copy")
        self.assertEqual((self.sound("trio"), self.session.undo_label, self.view.sound),
                         ({"defs": [second, first, third, second], "db": [-3, 0, -6, -3]},
                          "trio: variant 1 copied as variant 4", 3))
        # Nothing of all this was played.
        self.assertEqual(self.speaker.played, [])

    def test_the_last_variant_and_the_last_part_cannot_be_removed(self) -> None:
        self.pick("both")
        self.drag(self.label(1), button=3)
        self.assertEqual(self.items()[-1], ("Remove", True))
        self.choose("Remove")
        self.assertEqual((self.sound("both"), self.session.undo_label),
                         ({"parts": BOTH["parts"][:1]}, "both: part 2 removed"))
        self.drag(self.label(0), button=3)
        self.assertEqual(self.items()[-1], ("Remove", False))
        # What cannot be done is shown, dim. A press on it does nothing, and the menu stays open.
        self.drag(centre(self.ui.modal.rows[-1]))
        self.assertEqual((self.sound("both"), self.ui.modal is None), ({"parts": BOTH["parts"][:1]}, False))
        self.ui.handle(Event("key", key="escape"))
        # One sound made variants is one variant, which is not removed and has nowhere to move to.
        self.pick("one")
        self.drag(centre(self.panel.kinds.rect))
        self.choose("VARIANTS")
        self.drag(self.label(0), button=3)
        self.assertEqual(self.items(), [("Rename...", False), ("Copy", True), ("Play", True), ("Move left", False),
                                        ("Move right", False), ("Remove", False)])

    def test_a_part_dragged_by_its_label_saves_its_new_start_and_nothing_else(self) -> None:
        self.pick("both")
        before, scale = self.data(), self.view.map.scale
        x, y = self.label(1)
        by = round(0.1 * scale)
        start = on_tick(0.1 + by / scale)
        self.drag((x, y), (x + by // 2, y), (x + by, y + 30))
        self.assertEqual((start, self.sound("both"), differing(before, self.data()), self.session.undo_label),
                         (0.2, {"parts": [BOTH["parts"][0], [0.2, BOTH["parts"][1][1]]]}, ["sounds.both.parts"],
                          "both: part 2 moved to 0.2 s"))
        # It is the part the hand is in. A press on a label that is let go where it was saves nothing.
        self.assertEqual((self.view.sound, self.view.busy), (1, False))
        self.drag(self.label(0))
        self.assertEqual((self.session.undo_label, self.view.sound), ("both: part 2 moved to 0.2 s", 0))
        # Dragged past the start of the clock, a part starts at second 0.
        x, y = self.label(1)
        self.drag((x, y), (x - 2000, y))
        self.assertEqual((self.sound("both")["parts"][1][0], self.session.undo_label),
                         (0, "both: part 2 moved to 0 s"))

    def test_a_part_is_added_at_the_end_or_where_the_ruler_is_clicked_with_ctrl_and_copied_by_its_menu(self) -> None:
        self.pick("both")
        low, high = (part[1] for part in BOTH["parts"])
        # The sign adds a copy of the part the hand is in, where the sound ends: 14 ticks in.
        self.drag(self.plus())
        self.assertEqual((self.sound("both")["parts"][2:], self.session.undo_label, self.view.sound),
                         ([[on_tick(14 / 60), low]], "both: part 3 added at 0.233333 s", 2))
        self.drag(self.label(1))
        at = self.ruler(0.05)
        self.drag(at, mods=("ctrl",))
        self.assertEqual((self.sound("both")["parts"][3:], self.session.undo_label, self.view.sound),
                         ([[0.05, high]], "both: part 4 added at 0.05 s", 3))
        # Without ctrl a click in the ruler adds nothing.
        self.drag(at)
        self.assertEqual(len(self.sound("both")["parts"]), 4)
        self.drag(self.label(0), button=3)
        self.assertEqual(self.items(), [("Rename...", False), ("Copy", True), ("Volume (-3 dB)...", True),
                                        ("Remove", True)])
        self.choose("Copy")
        self.assertEqual((self.sound("both")["parts"][4:], self.session.undo_label, self.view.sound),
                         ([[0, low]], "both: part 1 copied as part 5", 4))

    def test_a_part_copied_has_its_tab_beside_that_of_the_part_it_starts_with_and_is_dragged_by_it(self) -> None:
        self.pick("both")
        self.drag(self.label(0), button=3)
        self.choose("Copy")
        first, copy = self.view.tab(0), self.view.tab(2)
        self.assertEqual((self.sound("both")["parts"][2][0], copy[0]), (0, first[0] + first[2]))
        # Dragged by its own tab to a tenth of a second, it starts with the second part: its tab is beside that one.
        x, y = self.label(2)
        self.drag((x, y), (x + round(0.1 * self.view.map.scale), y))
        second, copy = self.view.tab(1), self.view.tab(2)
        self.assertEqual(([at for at, _ in self.sound("both")["parts"]], self.session.undo_label, copy[0]),
                         ([0, 0.1, 0.1], "both: part 3 moved to 0.1 s", second[0] + second[2]))
        # The first is still taken by its tab, and so is the second.
        for which in (0, 1):
            self.drag(self.label(which))
            self.assertEqual(self.view.sound, which)

    def test_the_volume_of_a_part_is_typed_in_a_box_over_its_label(self) -> None:
        self.pick("both")
        self.drag(self.label(1), button=3)
        self.choose("Volume (0 dB)")
        self.assertEqual(self.ui.modal.text, "")
        self.type("-6")
        self.assertEqual((self.sound("both")["parts"][1][1], self.session.undo_label),
                         ({**BOTH["parts"][1][1], "db": -6}, "both: part 2 volume -6 dB"))
        # What is no number is said and saves nothing, and an empty box takes the volume out of the file.
        for typed, saved, label, message in (("-", -6, "both: part 2 volume -6 dB", "both: '-' is no number"),
                                             ("", None, "both: part 2 volume taken out", "")):
            self.drag(self.label(1), button=3)
            self.choose("Volume (-6 dB)")
            self.type(typed)
            self.assertEqual((self.sound("both")["parts"][1][1].get("db"), self.session.undo_label,
                              self.session.message), (saved, label, message))

    def test_a_border_of_a_held_sound_is_dragged_to_its_start_or_its_loop_and_stops_where_the_note_ends(self) -> None:
        self.pick("drone")
        before = self.data()

        def pull(ticks: float, by: float) -> None:
            """Drag the border that stands so many ticks into the clock by so many ticks."""
            seen = self.view.map
            x, y = seen.x(ticks / 60), seen.names[1] + 2
            self.drag((x, y), (x + round(by * seen.scale / 60), y))

        pull(6, 2)
        self.assertEqual((self.sound("drone")["held"]["start"], differing(before, self.data()),
                          self.session.undo_label), (8, ["sounds.drone.held.start"], "drone: loop starts at tick 8"))
        pull(38, -10)
        self.assertEqual((self.sound("drone")["held"]["loop"], self.session.undo_label),
                         (20, "drone: loop 20 ticks long"))
        # The note has room for 62 ticks: after a start of 8, the loop is no longer than 54.
        pull(28, 500)
        self.assertEqual((self.sound("drone")["held"], self.session.undo_label, self.session.message),
                         ({**DRONE["held"], "start": 8, "loop": 54}, "drone: loop 54 ticks long", ""))
        # And the start is dragged no further than leaves the loop in the note.
        pull(8, 500)
        self.assertEqual(self.sound("drone")["held"]["start"], 8)

    def test_the_crossfade_and_the_release_of_a_held_sound_are_two_knobs_of_the_bar(self) -> None:
        bar = self.panel.bar
        self.assertEqual((bar.crossfade.visible, bar.release.visible), (False, False))
        self.pick("drone")
        self.assertEqual([(knob.visible, knob.text, knob.value, knob.show(knob.value), bool(knob.help))
                          for knob in (bar.crossfade, bar.release)],
                         [(True, "CROSSFADE", 3, "3 ticks", True), (True, "RELEASE", 4, "4 ticks", True)])
        labels = []
        for knob, up in ((bar.crossfade, 10), (bar.release, 10), (bar.crossfade, 60)):
            x, y = centre(knob.rect)
            self.drag((x, y), (x, y - up))
            labels.append(self.session.undo_label)
        # The crossfade is kept no longer than the start, which is 6 ticks: turned further, nothing more is saved.
        self.assertEqual(labels, ["drone: crossfade 6 ticks", "drone: release 10 ticks", "drone: release 10 ticks"])
        self.assertEqual((self.sound("drone")["held"], bar.crossfade.value),
                         ({**DRONE["held"], "join": 6, "let_go": 10}, 6))

    def test_the_longest_time_is_set_in_the_ruler_dragged_by_its_line_and_taken_away(self) -> None:
        before = self.data()
        self.drag(self.ruler(0.1), button=3)
        self.assertEqual(self.items(), [("Set a longest time here", True), ("No longest time", False),
                                        ("Expect a note here...", True)])
        self.choose("Set")
        self.assertEqual((self.sound("one"), self.session.undo_label, self.view.limit),
                         ({**MIXED["sounds"]["one"], "max_seconds": 0.1}, "one: longest time 0.1 s", 0.1))
        # The line stands at that second, and is dragged by its grip in the ruler to another.
        x, y = self.ruler(0.1)
        self.drag((x, y), (x + round(0.05 * self.view.map.scale), y))
        self.assertEqual((self.sound("one")["max_seconds"], self.session.undo_label, self.view.limit),
                         (0.15, "one: longest time 0.15 s", 0.15))
        # The line stands past the end of the sound, which is 8 ticks long: in the second of room that the clock
        # has after it.
        self.assertAlmostEqual(self.view.seconds, 8 / 60 + 1)
        self.drag(self.ruler(0.02), button=3)
        self.assertEqual(self.items()[1], ("No longest time", True))
        self.choose("No longest time")
        self.assertEqual((self.data(), self.session.undo_label, self.view.limit),
                         (before, "one: no longest time", None))

    def test_the_other_limits_are_numbers_typed_from_the_menu_of_the_heading(self) -> None:
        self.drag(centre(self.panel.limits.rect))
        self.assertEqual([text for text, _ in self.items()],
                         ["Longest time (none)...", "Furthest off zero (0.002, the usual)...",
                          "Brightest (0.3, the usual)..."])
        self.choose("Brightest")
        self.type("0.5")
        self.drag(centre(self.panel.limits.rect))
        self.choose("Furthest")
        self.type("0.01")
        self.assertEqual((self.sound("one"), self.session.undo_label),
                         ({**MIXED["sounds"]["one"], "max_bright": 0.5, "max_dc": 0.01}, "one: furthest off zero 0.01"))
        self.drag(centre(self.panel.limits.rect))
        self.assertEqual(self.items()[2], ("Brightest (0.5)...", True))
        self.choose("Brightest")
        self.type("")
        self.assertEqual((self.sound("one"), self.session.undo_label),
                         ({**MIXED["sounds"]["one"], "max_dc": 0.01}, "one: brightest as usual"))

    def test_a_change_from_outside_waits_while_a_handle_is_dragged_and_the_wave_is_stale_until_its_render(self) -> None:
        self.pick("both")
        x, y = self.label(1)
        by = round(0.1 * self.view.map.scale)
        self.mouse("down", (x, y))
        self.mouse("move", (x + by, y))
        self.outside(lambda sounds: sounds["both"]["parts"][0][1].update(db=-9))
        self.panel.update()
        # The view goes on with the entry it was shown: the level in the menu of part 1 would still be -3.
        self.assertEqual((self.view.busy, self.session.message, self.session.model.project.sounds["both"]["parts"][0]
                          [1]["db"]), (True, "changed outside: both", -9))
        self.mouse("up", (x + by, y))
        self.assertEqual((self.sound("both")["parts"], self.session.message),
                         ([[0, {**BOTH["parts"][0][1], "db": -9}], [0.2, BOTH["parts"][1][1]]], f"both {MEANWHILE}"))
        self.assertEqual((self.view.busy, self.view.stale), (False, True))
        self.rendered()
        self.assertFalse(self.view.stale)


class Sheet(Menus):
    """What the sheet of the window gave the sound editor: the things of a sound that had no place in it before."""

    DATA, FIRST = EXTRA, "own"

    def test_the_notes_kept_with_a_sound_are_listed_added_changed_and_removed_and_no_text_is_left_alone(self) -> None:
        before, button = self.data(), centre(self.panel.notes.rect)
        # The button opens the keys and the texts as a list, with what adds one after the last. A value that
        # is no text is dim.
        self.drag(button)
        self.assertEqual(self.items(), [("by: hand", True), ("take: 3", False), ("+", True)])
        # A box takes the key, and then one the text.
        self.choose("+")
        self.type("mood")
        self.type("calm")
        self.assertEqual((self.sound("own")["meta"], self.session.undo_label, differing(before, self.data())),
                         ({"by": "hand", "take": 3, "mood": "calm"}, "own: mood added to NOTES",
                          ["sounds.own.meta.mood"]))
        # A key the sound has is said before any text is asked for.
        self.drag(button)
        self.choose("+")
        self.type("by")
        self.assertEqual((self.ui.modal, self.session.message), (None, "own: NOTES has 'by' already"))
        # An item opens the menu of its note: its text is changed in a box that holds it, or it is removed.
        self.drag(button)
        self.choose("by:")
        self.assertEqual(self.items(), [("Change...", True), ("Remove", True)])
        self.choose("Change")
        self.assertEqual(self.ui.modal.text, "hand")
        self.type("ear")
        self.assertEqual((self.sound("own")["meta"], self.session.undo_label),
                         ({"by": "ear", "take": 3, "mood": "calm"}, "own: by of NOTES changed"))
        self.drag(button)
        self.choose("mood")
        self.choose("Remove")
        self.assertEqual((self.sound("own")["meta"], self.session.undo_label),
                         ({"by": "ear", "take": 3}, "own: mood removed from NOTES"))
        # Each was one step: three taken back, the file is as it was.
        for _ in range(3):
            self.assertTrue(self.session.undo())
        self.assertEqual(self.data(), before)

    def test_a_note_the_sound_is_expected_to_play_is_marked_in_the_ruler_dragged_changed_and_removed(self) -> None:
        first = KEPT["check"][0]
        # The menu of the ruler marks one at the second under the mouse: a box takes the name of the note.
        self.drag(self.ruler(0.1), button=3)
        self.choose("Expect")
        self.assertEqual(self.ui.modal.text, "")
        self.type("E4")
        self.assertEqual((self.sound("own")["check"], self.session.undo_label),
                         ([first, {"at": 0.1, "note": "e4"}], "own: expects e4 at 0.1 s"))
        # What is no note is said, and nothing is saved.
        self.drag(self.ruler(0.02), button=3)
        self.choose("Expect")
        self.type("h9")
        self.assertEqual((len(self.sound("own")["check"]), self.session.message),
                         (2, "own: a note is a number or a name such as 'c#4', not 'h9'"))
        # Its flag is dragged along the ruler to another second.
        x, y = self.ruler(0.1)
        self.drag((x, y), (x + round(0.03 * self.view.map.scale), y))
        self.assertEqual((self.sound("own")["check"][1], self.session.undo_label),
                         ({"at": 0.13, "note": "e4"}, "own: expected note moved to 0.13 s"))
        # The right button on the flag opens its menu: its note is changed in a box, or it is removed.
        self.drag(self.ruler(0.13), button=3)
        self.assertEqual(self.items(), [("Change...", True), ("Remove", True)])
        self.choose("Change")
        self.assertEqual(self.ui.modal.text, "e4")
        self.type("g4")
        self.assertEqual((self.sound("own")["check"][1], self.session.undo_label),
                         ({"at": 0.13, "note": "g4"}, "own: expects g4 at 0.13 s"))
        self.drag(self.ruler(0.13), button=3)
        self.choose("Remove")
        self.assertEqual((self.sound("own")["check"], self.session.undo_label),
                         ([first], "own: expected note g4 removed"))
        # Without its last one the sound has no such notes in the file.
        self.drag(self.ruler(0.05), button=3)
        self.choose("Remove")
        self.assertEqual(("check" in self.sound("own"), self.session.message), (False, ""))

    def test_an_instrument_written_in_place_is_made_a_shared_one_from_the_list_of_the_instruments(self) -> None:
        bar, before = self.panel.bar, self.data()
        self.assertEqual((bar.inst.options[bar.inst.picked], bar.list.visible), ("its own", True))
        # The button after the field opens the list of the instruments, and after them what makes this one
        # an instrument of the project. A name the project has is refused, and said.
        self.drag(centre(bar.list.rect))
        self.assertEqual(self.items(), [("tone", True), ("buzz", True), ("Make shared...", True)])
        self.choose("Make shared")
        self.type("tone")
        self.assertEqual((self.data(), self.session.message),
                         (before, "own: the project has an instrument called 'tone' already"))
        # Under a new name the instrument moves to those of the project, and the voice names it.
        self.drag(centre(bar.list.rect))
        self.choose("Make shared")
        self.type("rasp")
        after = self.data()
        self.assertEqual((after["instruments"]["rasp"], after["sounds"]["own"]["def"]["voices"][0]["inst"],
                          differing(before, after)), (WRITTEN, "rasp", ["instruments.rasp", "sounds.own.def.voices"]))
        self.assertEqual((self.session.undo_label, bar.inst.options, bar.inst.options[bar.inst.picked]),
                         ("own: voice 1 plays rasp, now a shared instrument", ("tone", "buzz", "rasp"), "rasp"))
        # The two were one step.
        self.assertTrue(self.session.undo())
        self.panel.update()
        self.assertEqual((self.data(), bar.inst.options[bar.inst.picked]), (before, "its own"))
        # A voice that names its instrument has nothing to make shared: the item is dim. An instrument of the
        # list that is chosen plays the voice.
        self.pick("two")
        self.drag(centre(bar.list.rect))
        self.assertEqual(self.items(), [("tone", False), ("buzz", True), ("Make shared...", False)])
        self.choose("buzz")
        self.assertEqual((self.sound("two")["def"]["voices"][0]["inst"], self.session.undo_label),
                         ("buzz", "two: voice 1 plays buzz"))

    def test_a_key_of_the_keyboard_and_the_piano_keys_of_the_computer_play_the_instrument_of_the_picked_voice(
            self) -> None:
        self.pick("two")
        seen = self.view.map
        on = (self.view.keys.rect[0] + 5, seen.y(60) + 4)
        # A press on c4 of the keyboard is heard in the instrument of the first voice. Nothing plays, nothing
        # is changed, and the keyboard has the focus.
        self.drag(on)
        self.assertEqual((len(self.speaker.notes), self.speaker.played, self.session.can_undo, self.ui.focus),
                         (1, [], False, self.view.keys))
        self.assertIs(self.speaker.notes[0], self.session.model.tone("tone", 60)["sides"])
        # With the second voice picked it is that one's. The keys of the computer play it as a piano from the
        # lowest c in view: z is that c, and q the one an octave over it.
        self.drag(centre(self.panel.bar.voices.cells()[1]))
        self.drag(on)
        low = -(-seen.pitches().start // 12) * 12
        self.assertEqual((low in seen.pitches(), self.ui.handle(Event("key", key="z")),
                          self.ui.handle(Event("key", key="q"))), (True, True, True))
        for note, pitch in zip(self.speaker.notes[1:], (60, low, low + 12)):
            self.assertIs(note, self.session.model.tone("buzz", pitch)["sides"])
        # With ctrl held it is no key of the piano, and a key that plays no note is not the view's either.
        self.assertEqual((self.ui.handle(Event("key", key="z", mods=frozenset({"ctrl"}))),
                          self.ui.handle(Event("key", key="f1")), len(self.speaker.notes), self.speaker.played),
                         (False, False, 4, []))

    def test_a_drag_in_the_velocity_lane_is_one_step_and_the_usual_velocity_is_written_as_none(self) -> None:
        self.pick("two")
        before = self.data()
        left, top, _, high = self.view.lane.rect
        self.assertEqual(((left, top, high), self.view.lane.span), ((4, 404, 24), self.view.map.grid[0::2]))
        # Half way up the lane is a velocity of a half: it is the fourth number of the note, in the file.
        x = self.at(1, 60)[0]
        self.drag((x, top + high - 2 - 10))
        self.assertEqual((self.sound("two")["def"]["voices"][0]["notes"][0], self.session.undo_label,
                          differing(before, self.data()), self.speaker.notes),
                         ([0, "c4", 6, 0.5], f"two: {VELOCITY_DONE}", ["sounds.two.def.voices"], []))
        # The right button on its bar gives the note the usual velocity again, which the file does not write.
        self.drag((x, top + 12), button=3)
        self.assertEqual((self.data(), self.session.undo_label), (before, f"two: {VELOCITY_DONE}"))
        self.assertTrue(self.session.undo())
        self.panel.update()
        self.assertEqual(self.view.drawn["voices"][0]["notes"][0], [0, "c4", 6, 0.5])

    def test_the_help_line_on_the_wave_says_the_pitch_the_render_has_there(self) -> None:
        seen = self.view.map
        self.assertEqual(self.view.foot, (4, 430, 654, 46))
        # The sound is a saw at c4 for a tenth of a second.
        self.mouse("move", (seen.x(0.03), seen.wave[1] + 10))
        self.assertRegex(self.ui.help, r"^0\.03 s, -[0-9.]+ dB, 26[0-9.]+ Hz, c4$")
        # Beside the stretch of the wave, in its box under the keyboard, there is nothing to read.
        self.mouse("move", (self.view.foot[0] + 10, seen.wave[1] + 10))
        self.assertEqual(self.ui.help, HELP)


if __name__ == "__main__":
    unittest.main()
