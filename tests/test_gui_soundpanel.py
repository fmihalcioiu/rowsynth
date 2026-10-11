"""Tests for rowsynth.gui.soundpanel: what the middle of the window shows of a picked sound or instrument."""
from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import unittest
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from rowsynth.errors import RowSynthError
from rowsynth.gui import layout, soundview, sprites, theme
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.font import Font
from rowsynth.gui.model import Model
from rowsynth.gui.session import HOLD, Session
from rowsynth.gui.soundpanel import HELP, HOLD_HELP, SoundPanel, arrange, words, zone
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
RATE = 44100

#: The panel in a window, with room round it on the surface it is drawn on: its boxes are (24, 14, 392, 17) and
#: (24, 35, 392, 241), its view is (28, 39, 384, 233), and its button (385, 14, 31, 17).
RECT = (20, 10, 400, 270)
SIZE = (440, 290)
OUTSIDE = theme.GOOD

#: A sound that names an instrument its project does not have, and what the engine says of it.
BROKEN = {"def": {"voices": [{"inst": "nope", "notes": [[0, "c4", 4]]}]}}
REFUSAL = "def: voices[0].inst: no instrument named 'nope'"

#: A held sound: a start of a tenth of a second, a loop of half a second, and an end of a tenth.
HUM = {"held": {"note": {"voices": [{"inst": "bass", "notes": [[0, "c3", 60]]}]}, "start": 6, "loop": 30, "join": 3,
                "end": {"voices": [{"inst": "bass", "notes": [[0, "c3", 2]]}]}, "let_go": 4}, "db": -6}
START, LOOP, END = 0.1, 0.5, 0.1


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

    def play(self, sides: Sequence, loop: bool = False) -> None:
        self.asked.append("play")
        self.playing = True

    def also(self, sides: Sequence) -> None:
        self.asked.append("also")

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
                         ["3 variants", "1 variant", "2 parts", "1 part", "held", ""])
        stand, _, panel = panel_on()
        for picked, kind, said in ((("sound", "blip"), "defs", "3 variants"),
                                   (("sound", "explosion"), "parts", "2 parts"), (("sound", "hum"), "held", "held"),
                                   (("sound", "coin"), "def", ""), (("instrument", "chime"), "def", ""),
                                   (("song", "tune"), "def", ""), (None, "def", "")):
            stand.model.picked = picked
            panel.update()
            self.assertEqual((panel.kind, panel.words), (kind, said), picked)
        # A sound the engine refuses has no words but the refusal.
        stand.model.project = SimpleNamespace(sounds={"coin": BROKEN}, instruments={})
        stand.model.picked = ("sound", "coin")
        panel.update()
        self.assertEqual((panel.kind, panel.words, panel.refusal), ("def", "", REFUSAL))


class Shown(unittest.TestCase):
    def test_the_boxes_the_button_and_the_view_are_inside_the_panel_at_any_size(self) -> None:
        self.assertEqual(arrange((0, 0, 662, 480)), {"head": (4, 4, 654, 17), "hold": (627, 4, 31, 17),
                                                     "box": (4, 25, 654, 451), "view": (8, 29, 646, 443)})
        _, ui, panel = panel_on()
        self.assertEqual((ui.widgets, panel.widgets, panel.view.rect, panel.view.help),
                         ([panel.view, panel.hold], [panel.view, panel.hold], (28, 39, 384, 233), HELP))
        self.assertEqual((panel.hold.rect, panel.hold.text, panel.hold.help), ((385, 14, 31, 17), "HOLD", HOLD_HELP))
        for rect in (RECT, (40, 10, 60, 30), (0, 22, 3, 2)):
            panel.place(rect)
            self.assertEqual((panel.view.rect, panel.hold.rect), (arrange(rect)["view"], arrange(rect)["hold"]))
            self.assertTrue(all(lies_in(place, rect) for place in arrange(rect).values()), rect)

    def test_a_sound_is_its_bars_over_an_empty_wave_until_its_render_is_ready(self) -> None:
        stand, _, panel = panel_on()
        view = panel.view
        shape = soundview.shapes(stand.model.project.sounds["coin"], RATE, stand.model.project.instruments)[0][2]
        self.assertEqual((panel.heading, panel.refusal, panel.colour, view.visible),
                         (("SOUND", "coin"), "", theme.SOUND, True))
        # The two notes of coin are there, on a clock as long as the sound, and there is no wave.
        seen = view.map
        x, y, _, _ = seen.bar(shape.bars[1].start, shape.bars[1].seconds, shape.bars[1].pitch)
        self.assertEqual((view.seconds, seen.notes, view.bar_at((x + 1, y + 1)), view.columns()),
                         (shape.seconds, True, shape.bars[1], []))
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
            # The file is loaded again: the same sound is made anew, with an empty wave, and the view stays.
            stand.model.project, stand.model.made = Project.load(EXAMPLES / "arcade.json"), None
            panel.update()
            panel.update()
            self.assertEqual((made.call_count, view.zoom, view.columns()), (2, 3, []))
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
    def test_a_variant_chosen_in_the_view_is_played_and_marked_with_the_playhead_in_its_stretch(self) -> None:
        # Three variants of a tenth of a second, the second from 0.12 s and the third from 0.24 s.
        stand, ui, panel = panel_on(picked=("sound", "blip"), made=silence(0.1, 0.1, 0.1))
        view = panel.view
        seen = view.map
        self.assertEqual((view.kind, view.played, len(view.columns())), ("defs", None, 3))
        # A press on the label of the second plays that one, and it is the one that is marked.
        self.assertTrue(press(ui, (seen.x(0.12) + 5, seen.names[1] + 2)))
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
        # A press in the stretch of the third's wave plays the third.
        self.assertTrue(press(ui, (seen.x(0.24) + 30, seen.wave[1] + 10)))
        stand.at = 0.02
        panel.update()
        self.assertEqual((stand.asked, view.played), ([("play", 1), ("play", 2)], 2))
        self.assertAlmostEqual(view.playhead, 0.26)
        # A play the panel did not ask for is of a variant the session does not name: nothing is marked. That
        # is so when it begins after the chosen one has ended, and when it begins while that still sounds.
        for playing in (False, True):
            press(ui, (seen.x(0.12) + 5, seen.names[1] + 2))
            stand.at = 0.05
            panel.update()
            self.assertEqual(view.played, 1)
            stand.playing = playing
            panel.update()
            stand.playing, stand.at = True, 0.01
            panel.update()
            self.assertEqual((view.played, view.playhead), (None, None), playing)
        # Another sound is picked, and then this one again: nothing of it was played yet.
        press(ui, (seen.x(0.12) + 5, seen.names[1] + 2))
        for picked, made, played in ((("sound", "blip"), stand.model.made, 1), (("sound", "coin"), None, None),
                                     (("sound", "blip"), stand.model.made, None)):
            stand.model.picked, stand.model.made = picked, made
            panel.update()
            self.assertEqual(view.played, played, picked)

    def test_the_hold_button_is_there_for_a_held_sound_alone(self) -> None:
        stand, ui, panel = panel_on()
        on = centre(panel.hold.rect)
        for picked, there in ((("sound", "coin"), False), (("sound", "blip"), False), (("sound", "explosion"), False),
                              (("instrument", "bass"), False), (("song", "tune"), False), (None, False),
                              (("sound", "hum"), True)):
            stand.model.picked = picked
            panel.update()
            self.assertEqual((panel.hold.visible, press(ui, on)), (there, there), picked)
        self.assertEqual(stand.asked, [("hold",), ("release",)])
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
        self.assertEqual((view.kind, len(view.columns()), round(view.seconds, 9)), ("held", 3, 0.7))
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
        for picked, colour in ((("sound", "coin"), theme.SOUND), (("instrument", "chime"), theme.INSTRUMENT)):
            self.stand.model.picked = picked
            self.draw()
            kind, name = picked[0].upper(), picked[1]
            # The kind starts 5 pixels in, and the name 8 pixels after it. Both boxes have their outline.
            self.assertTrue(self.written(theme.DIM, places["head"], kind, 29))
            self.assertTrue(self.written(colour, places["head"], name, 29 + Font.width(kind) + 8))
            self.assertEqual(self.colours(places["head"]), {theme.PANEL, theme.LINE, theme.DIM, colour})
            self.assertEqual(self.colours((24, 35, 392, 1)) | self.colours((24, 36, 1, 239)), {theme.LINE})
            # The view is drawn in its box, and nothing is outside the panel.
            self.assertIn(theme.BACKGROUND, self.colours(places["view"]))
            around = ((0, 0, 440, 10), (0, 280, 440, 10), (0, 10, 20, 270), (420, 10, 20, 270))
            self.assertEqual(set().union(*(self.colours(rect) for rect in around)), {OUTSIDE})
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

    def test_the_heading_says_in_dim_words_what_a_sound_of_several_is_and_a_held_one_has_its_button(self) -> None:
        head, hold = arrange(RECT)["head"], arrange(RECT)["hold"]
        for name, said in (("blip", "3 variants"), ("explosion", "2 parts"), ("hum", "held")):
            self.stand.model.picked = ("sound", name)
            self.draw()
            # The words stand 8 pixels after the name, which stands 8 pixels after the kind.
            after = 29 + Font.width("SOUND") + 8 + Font.width(name) + 8
            self.assertTrue(self.written(theme.DIM, (after, head[1], head[0] + head[2] - after, head[3]), said, after))
            self.assertTrue(self.written(theme.SOUND, head, name, 29 + Font.width("SOUND") + 8))
        # The button of the held sound is the right end of the heading: raised, with its word 4 pixels in.
        x, y, w, h = hold
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
        # In a narrow panel the words are cut 8 pixels before the button, and do not run on under it.
        self.panel.place((20, 10, 120, 270))
        self.draw()
        x, y, w, h = arrange((20, 10, 120, 270))["hold"]
        self.assertIn(theme.DIM, self.colours((x - 14, y + 1, 6, h - 2)))
        self.assertEqual(self.colours((x - 8, y + 1, 8, h - 2)), {theme.PANEL})
        self.panel.place(RECT)
        x, y, w, h = hold
        # A sound that is not held has the heading's own ground there.
        self.stand.model.picked = ("sound", "blip")
        self.draw()
        self.assertEqual(self.colours((x, y + 1, w - 1, h - 2)), {theme.PANEL})


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
        self.assertEqual((panel.heading, view.seconds, view.playhead, left, len(columns)),
                         (("SOUND", "coin"), made["seconds"], None, view.map.wave[0], view.map.wave[2]))
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


if __name__ == "__main__":
    unittest.main()
