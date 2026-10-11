"""Tests for rowsynth.gui.soundpanel: what the middle of the window shows of a picked sound or instrument."""
from __future__ import annotations

import os
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from rowsynth.commands import mono
from rowsynth.errors import RowSynthError
from rowsynth.gui import layout, soundview, sprites, theme
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.font import Font
from rowsynth.gui.model import Model
from rowsynth.gui.session import Session
from rowsynth.gui.soundpanel import HELP, SoundPanel, arrange
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
#: (24, 35, 392, 241), and its view is (28, 39, 384, 233).
RECT = (20, 10, 400, 270)
SIZE = (440, 290)
OUTSIDE = theme.GOOD

#: A sound that names an instrument its project does not have, and what the engine says of it.
BROKEN = {"def": {"voices": [{"inst": "nope", "notes": [[0, "c4", 4]]}]}}
REFUSAL = "def: voices[0].inst: no instrument named 'nope'"


def render(*sides: np.ndarray) -> dict:
    """The render of a sound or an instrument, as a model gives one."""
    return {"sides": list(sides), "rate": RATE, "loop": False, "seconds": len(sides[0]) / RATE}


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
    """A stand-in for the session: a model, whether the picked thing plays, and how far into it."""

    def __init__(self) -> None:
        self.model = Held()
        self.playing = False
        self.at = 0.0

    def position(self) -> float:
        return self.at


def panel_on(rect: layout.Rect = RECT) -> tuple[Stand, Ui, SoundPanel]:
    """A stand-in session, a Ui, and the panel made with the two, placed in a rectangle and brought up to date."""
    stand, ui = Stand(), Ui()
    panel = SoundPanel(stand, ui)
    panel.place(rect)
    panel.update()
    return stand, ui, panel


def lies_in(rect: layout.Rect, bounds: layout.Rect) -> bool:
    """Whether a rectangle has no pixel outside another, and neither a width nor a height below 0."""
    x, y, w, h = rect
    return (w >= 0 and h >= 0 and bounds[0] <= x and x + w <= bounds[0] + bounds[2]
            and bounds[1] <= y and y + h <= bounds[1] + bounds[3])


class Shown(unittest.TestCase):
    def test_the_boxes_and_the_view_are_inside_the_panel_at_any_size(self) -> None:
        self.assertEqual(arrange((0, 0, 662, 480)),
                         {"head": (4, 4, 654, 17), "box": (4, 25, 654, 451), "view": (8, 29, 646, 443)})
        _, ui, panel = panel_on()
        self.assertEqual((ui.widgets, panel.widgets, panel.view.rect, panel.view.help),
                         ([panel.view], [panel.view], (28, 39, 384, 233), HELP))
        for rect in (RECT, (40, 10, 60, 30), (0, 22, 3, 2)):
            panel.place(rect)
            self.assertEqual(panel.view.rect, arrange(rect)["view"])
            self.assertTrue(all(lies_in(place, rect) for place in arrange(rect).values()), rect)

    def test_a_sound_is_its_bars_over_an_empty_wave_until_its_render_is_ready(self) -> None:
        stand, _, panel = panel_on()
        view = panel.view
        shape = soundview.shapes(stand.model.project.sounds["coin"], RATE, stand.model.project.instruments)[0][2]
        self.assertEqual((panel.heading, panel.refusal, panel.colour, view.visible),
                         (("SOUND", "coin"), "", theme.SOUND, True))
        # The two notes of coin are there, on a clock as long as the sound, and the wave has no columns.
        seen = view.map
        x, y, _, _ = seen.bar(shape.bars[1].start, shape.bars[1].seconds, shape.bars[1].pitch)
        self.assertEqual((view.seconds, seen.notes, view.bar_at((x + 1, y + 1)), len(view.columns()[1])),
                         (shape.seconds, True, shape.bars[1], 0))
        # The render comes: the wave is the mean of its sides, in the colour of a sound.
        stand.model.made = render(np.full(8820, 0.5, dtype=np.float32), np.zeros(8820, dtype=np.float32))
        panel.update()
        left, columns = view.columns()
        self.assertEqual((left, len(columns), view.colour, view.bar_at((x + 1, y + 1))),
                         (seen.x(0.0), seen.x(0.2) - seen.x(0.0), theme.SOUND, shape.bars[1]))
        self.assertTrue(np.all(columns == 0.25))
        # A render the engine refuses leaves the bars over an empty wave.
        stand.model.made, stand.model.refused = None, "the sound is too long"
        panel.update()
        self.assertEqual((len(view.columns()[1]), view.map.notes, panel.refusal), (0, True, ""))

    def test_an_instrument_is_the_wave_of_its_test_note_with_no_bars(self) -> None:
        stand, _, panel = panel_on()
        stand.model.picked = ("instrument", "chime")
        panel.update()
        view = panel.view
        self.assertEqual((panel.heading, panel.colour, view.visible, view.map.notes, view.seconds),
                         (("INSTRUMENT", "chime"), theme.INSTRUMENT, True, False, 0.0))
        stand.model.made = render(np.full(4410, 0.5, dtype=np.float32))
        panel.update()
        left, columns = view.columns()
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
            self.assertEqual((made.call_count, view.zoom, len(view.columns()[1])), (2, 3, 0))
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


class OnAModel(unittest.TestCase):
    def test_on_a_real_model_and_session_coin_is_shown_over_the_render_the_model_made(self) -> None:
        def settle() -> None:
            """Waits for every render that is still on a thread of the model's."""
            for thread in threading.enumerate():
                if thread.name.startswith("rowsynth-"):
                    thread.join()

        self.addCleanup(settle)
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
        left, columns = view.columns()
        self.assertEqual((panel.heading, view.seconds, view.playhead, left, len(columns)),
                         (("SOUND", "coin"), made["seconds"], None, view.map.wave[0], view.map.wave[2]))
        self.assertTrue(np.array_equal(columns, soundview.columns(mono(made["sides"]), len(columns))))
        self.assertGreater(float(np.abs(columns).max()), 0.5)


if __name__ == "__main__":
    unittest.main()
