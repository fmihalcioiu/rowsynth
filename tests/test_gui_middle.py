"""Tests for rowsynth.gui.middle: which panel fits what is picked, and the middle that shows that one."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rowsynth.gui import layout, sprites, theme
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.font import Font
from rowsynth.gui.middle import EMPTY, SONG, SOUND, Middle, fitting
from rowsynth.gui.model import Model
from rowsynth.gui.ui import Event, Ui

# pygame reads these when it is imported: no test opens a window or plays a sound.
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
try:
    import pygame
except ImportError:
    pygame = None

NEEDS = "pygame is not installed"
#: The middle of the art surface the layout is drawn for, and the pixel in the middle of that.
RECT = (124, 22, 662, 504)
CENTRE = (455, 274)

#: A project with one thing of each kind.
PROJECT = {
    "instruments": {"tone": {"wave": "tri"}},
    "sounds": {"ping": {"def": {"voices": [{"inst": "tone", "notes": [[0, "c5", 6]]}]}}},
    "songs": {"walk": {"row_ticks": 6, "channels": {"air": {"inst": "tone"}},
                       "patterns": {"a": {"air": "o4 l4 c e g e"}}, "order": ["a"]}},
}


class Stand:
    """A stand-in for the session: a real model, of which nothing plays."""

    def __init__(self, model: Model) -> None:
        self.model = model
        self.playing = False

    def position(self) -> float:
        return 0.0


class Fitting(unittest.TestCase):
    def test_a_song_fits_the_song_panel_a_sound_or_an_instrument_the_sound_panel_and_nothing_neither(self) -> None:
        self.assertEqual([fitting(picked) for picked in (("song", "walk"), ("sound", "ping"), ("instrument", "tone"),
                                                         None)], [SONG, SOUND, SOUND, None])


class Shown(unittest.TestCase):
    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "walk.json"
        self.path.write_text(json.dumps(PROJECT), encoding="utf-8")
        self.model, self.ui = Model(self.path), Ui()
        self.middle = Middle(Stand(self.model), self.ui)
        self.middle.place(RECT)
        self.middle.update()
        self.song, self.sound = self.middle.panels[SONG], self.middle.panels[SOUND]

    def pick(self, kind: str, name: str) -> None:
        """Pick a thing of the project, and have the middle take a frame."""
        self.model.pick(kind, name)
        self.middle.update()

    def seen(self) -> list[str]:
        """Which of the two panels have a widget that is seen, by what they are called."""
        return [name for name, panel in self.middle.panels.items() if any(widget.visible for widget in panel.widgets)]

    def test_a_song_picked_is_shown_in_the_song_panel_with_the_sound_panel_off_the_window(self) -> None:
        self.assertEqual((self.model.picked, self.middle.fits, self.seen()), (("song", "walk"), SONG, [SONG]))
        self.assertEqual((self.song.shown, self.sound.shown), (True, False))
        names = tuple(block.name for block in self.song.strip.map.blocks)
        self.assertEqual((names, self.song.channel, self.sound.heading), (("a",), "air", None))
        # Both have the whole rectangle, the one that is off the window too.
        self.assertEqual((self.middle.rect, self.song.rect, self.sound.rect), (RECT, RECT, RECT))

    def test_a_sound_or_an_instrument_picked_is_shown_in_the_sound_panel_with_the_song_panel_off(self) -> None:
        for kind, name in (("sound", "ping"), ("instrument", "tone")):
            self.pick(kind, name)
            self.assertEqual((self.middle.fits, self.seen(), self.song.shown, self.sound.shown),
                             (SOUND, [SOUND], False, True))
            self.assertEqual(self.sound.heading, (kind.upper(), name))

    def test_with_nothing_picked_neither_panel_is_on_the_window(self) -> None:
        # The file is emptied outside, and with it goes what was picked.
        self.path.write_text("{}", encoding="utf-8")
        self.model.poll()
        self.middle.update()
        self.assertEqual((self.model.picked, self.middle.fits, self.seen()), (None, None, []))
        self.assertEqual((self.song.shown, self.sound.shown), (False, False))

    def test_picked_back_and_forth_no_widget_of_the_panel_that_is_off_takes_the_mouse(self) -> None:
        self.ui.handle(Event("move", CENTRE))
        for kind, name in (("sound", "ping"), ("song", "walk"), ("instrument", "tone"), ("song", "walk"),
                           ("sound", "ping")):
            self.pick(kind, name)
            on, off = (self.song, self.sound) if kind == "song" else (self.sound, self.song)
            self.assertEqual([widget for widget in off.widgets if widget.visible], [])
            # The mouse rests where the roll of the one and the view of the other are: the wheel zooms one of them.
            self.assertIn(self.ui.hot, on.widgets)
            before = (self.song.roll.row_w, self.sound.view.zoom)
            self.assertTrue(self.ui.handle(Event("wheel", CENTRE, wheel=1, mods=frozenset({"ctrl"}))))
            zoomed = (self.song.roll.row_w - before[0], self.sound.view.zoom - before[1])
            self.assertEqual(zoomed, (1, 0) if kind == "song" else (0, 1))

    def test_taken_off_the_window_neither_panel_is_on_it_and_brought_back_the_one_that_fits_is(self) -> None:
        self.middle.show(False)
        self.assertEqual((self.seen(), self.song.shown, self.sound.shown), ([], False, False))
        self.pick("sound", "ping")
        self.assertEqual((self.middle.fits, self.seen(), self.sound.heading), (SOUND, [], None))
        self.middle.show(True)
        self.middle.update()
        self.assertEqual((self.seen(), self.sound.heading), ([SOUND], ("SOUND", "ping")))

    def test_the_seconds_of_a_frame_reach_the_panel_that_is_shown_and_the_other_takes_no_frame(self) -> None:
        self.model.pick("sound", "ping")
        with mock.patch.object(self.sound, "update") as told, mock.patch.object(self.song, "update") as other:
            self.middle.update(0.016)
        told.assert_called_once_with(0.016)
        other.assert_not_called()
        # A frame of the song panel has the roll go by the toggle over it.
        self.model.pick("song", "walk")
        self.song.bar.ghosts.is_on = False
        with mock.patch.object(self.sound, "update") as other:
            self.middle.update(0.016)
        other.assert_not_called()
        self.assertFalse(self.song.roll.ghosts)


@unittest.skipUnless(pygame, NEEDS)
class Drawn(unittest.TestCase):
    def test_with_nothing_picked_one_dim_line_in_the_middle_says_the_project_is_empty(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "empty.json"
        path.write_text("{}", encoding="utf-8")
        ui = Ui()
        middle = Middle(Stand(Model(path)), ui)
        middle.place(RECT)
        middle.update()
        surface = pygame.Surface((960, 540))
        surface.fill(theme.BACKGROUND)
        canvas = Canvas(surface, Font(sprites.load("font")))
        middle.draw(canvas)
        ui.draw(canvas)

        def ink(rect: layout.Rect) -> int:
            """How many pixels of a rectangle are dim."""
            x, y, w, h = rect
            return sum(surface.get_at((x + column, y + row))[:3] == theme.DIM
                       for column in range(w) for row in range(h))

        # The line is as far from the left of the rectangle as from its right, and half way down it.
        wide = Font.width(EMPTY)
        line = (RECT[0] + (RECT[2] - wide) // 2, CENTRE[1] - Font.LINE, wide, 2 * Font.LINE)
        self.assertEqual((EMPTY, ink(line) > 0, ink((0, 0, 960, 540)) - ink(line)), ("the project is empty", True, 0))
        self.assertGreater(min(ink((line[0], line[1], 3, line[3])), ink((line[0] + wide - 3, line[1], 3, line[3]))), 0)


if __name__ == "__main__":
    unittest.main()
