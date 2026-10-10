"""Tests for the GUI's window: its scale, its art surface, and the app run in it for a few frames."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# No test opens a window that can be seen, or a sound device. This is said before pygame is first imported.
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from rowsynth.errors import RowSynthError
from rowsynth.gui import run, start, theme

try:
    import pygame
except ImportError:
    pygame = None
else:
    from rowsynth.gui.screen import Screen, fit

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
NEEDS = "pygame is not installed"


@unittest.skipUnless(pygame, NEEDS)
class Fit(unittest.TestCase):
    def test_a_full_hd_window_shows_the_smallest_art_size_at_twice_its_size(self) -> None:
        self.assertEqual(theme.MIN_ART, (960, 540))
        self.assertEqual(fit((1920, 1080)), (2, (960, 540)))

    def test_a_window_with_no_room_for_twice_keeps_every_pixel_of_its_own_at_scale_1(self) -> None:
        # The last two are smaller than the smallest art size, and are not scaled down to hold it.
        for size in ((1280, 720), (960, 540), (1919, 1080), (1920, 1079), (800, 600), (640, 360)):
            with self.subTest(size=size):
                self.assertEqual(fit(size), (1, size))

    def test_a_larger_window_takes_the_largest_scale_that_leaves_the_smallest_art_size(self) -> None:
        self.assertEqual(fit((2560, 1440)), (2, (1280, 720)))
        self.assertEqual(fit((2880, 1620)), (3, (960, 540)))
        self.assertEqual(fit((3840, 2160)), (4, (960, 540)))
        # The side with the least room decides, and what does not divide is rounded down.
        self.assertEqual(fit((3841, 1100)), (2, (1920, 550)))

    def test_a_scale_that_is_given_is_kept_whatever_the_window(self) -> None:
        self.assertEqual(fit((1920, 1080), 1), (1, (1920, 1080)))
        self.assertEqual(fit((1920, 1080), 3), (3, (640, 360)))
        self.assertEqual(fit((1280, 720), 2), (2, (640, 360)))
        self.assertEqual(fit((1921, 1081), 2), (2, (960, 540)))

    def test_a_scale_under_1_is_refused(self) -> None:
        with self.assertRaises(RowSynthError) as refused:
            fit((1920, 1080), 0)
        self.assertEqual(str(refused.exception),
                         "a scale of 0 cannot be drawn: the scale is a whole number, 1 or more")


@unittest.skipUnless(pygame, NEEDS)
class Window(unittest.TestCase):
    def opened(self, *args: object) -> Screen:
        """A screen that is closed again when the test is over."""
        screen = Screen(*args)
        self.addCleanup(screen.close)
        return screen

    def test_it_opens_a_window_that_can_be_resized_with_an_art_surface_of_its_size_over_the_scale(self) -> None:
        screen = self.opened()
        self.assertEqual(pygame.display.get_surface().get_size(), theme.WINDOW)
        self.assertTrue(pygame.display.get_surface().get_flags() & pygame.RESIZABLE)
        self.assertEqual((screen.scale, screen.begin().get_size()), (2, (960, 540)))

    def test_to_art_turns_a_position_in_the_window_into_art_pixels(self) -> None:
        screen = self.opened((1920, 1080))
        self.assertEqual([screen.to_art(pos) for pos in ((0, 0), (1, 1), (2, 5), (1919, 1079))],
                         [(0, 0), (0, 0), (1, 2), (959, 539)])
        self.assertEqual(self.opened((1920, 1080), 3).to_art((640, 361)), (213, 120))

    def test_present_shows_an_art_pixel_as_a_square_of_whole_pixels_with_no_smoothing(self) -> None:
        screen = self.opened((1920, 1080))
        art = screen.begin()
        art.fill(theme.BACKGROUND)
        art.set_at((5, 7), theme.ACTIVE)
        screen.present()
        window = pygame.display.get_surface()
        for pos in ((10, 14), (11, 14), (10, 15), (11, 15)):
            self.assertEqual(window.get_at(pos)[:3], theme.ACTIVE, pos)
        for pos in ((9, 14), (12, 14), (10, 13), (10, 16)):
            self.assertEqual(window.get_at(pos)[:3], theme.BACKGROUND, pos)

    def test_the_art_surface_follows_the_window_to_a_new_size(self) -> None:
        screen = self.opened((1920, 1080))
        # What a resize leaves behind it: the window's own surface at another size.
        pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
        self.assertEqual((screen.begin().get_size(), screen.scale), ((1280, 720), 1))
        self.assertEqual(screen.to_art((700, 30)), (700, 30))
        # A scale that was given stays, and a window that does not divide by it has a strip left over.
        screen = self.opened((1920, 1080), 2)
        pygame.display.set_mode((1281, 721), pygame.RESIZABLE)
        art = screen.begin()
        self.assertEqual((art.get_size(), screen.scale), ((640, 360), 2))
        art.fill(theme.TEXT)
        screen.present()
        window = pygame.display.get_surface()
        self.assertEqual((window.get_at((1279, 719))[:3], window.get_at((1280, 720))[:3]),
                         (theme.TEXT, theme.BACKGROUND))

    def test_a_snapshot_is_a_copy_of_the_art_surface(self) -> None:
        screen = self.opened((960, 540))
        screen.begin().fill(theme.GOOD)
        taken = screen.snapshot()
        screen.begin().fill(theme.FAULT)
        self.assertEqual((taken.get_size(), taken.get_at((3, 3))[:3]), ((960, 540), theme.GOOD))


@unittest.skipUnless(pygame, NEEDS)
class Run(unittest.TestCase):
    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)

    def test_run_for_3_frames_returns_0_and_its_shot_is_a_png_of_the_empty_frame(self) -> None:
        # The folder of the shot is not there yet, and its name does not end as that of a PNG does.
        shot = self.folder / "shots" / "frame"
        self.assertEqual(run(EXAMPLES / "arcade.json", frames=3, shot=shot), 0)
        self.assertIsNone(pygame.display.get_surface())
        self.assertEqual(shot.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        with open(shot, "rb") as file:
            drawn = pygame.image.load(file, ".png")
        self.assertEqual(drawn.get_size(), (960, 540))
        # The top bar and its line, the left panel's line, the middle, the right panel's line and the status bar's.
        self.assertEqual([drawn.get_at(pos)[:3] for pos in ((0, 0), (0, 21), (123, 22), (124, 22), (786, 300),
                                                            (787, 300), (500, 526), (500, 539))],
                         [theme.PANEL, theme.LINE, theme.LINE, theme.BACKGROUND, theme.LINE, theme.PANEL,
                          theme.LINE, theme.PANEL])

    def test_a_window_that_is_closed_ends_the_run_before_its_frames_are_up(self) -> None:
        app = start(EXAMPLES / "arcade.json")
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        self.assertEqual(app.run(frames=5), 0)
        self.assertIsNone(pygame.display.get_surface())

    def test_a_missing_file_is_refused_before_any_window_is_opened(self) -> None:
        path = self.folder / "nowhere.json"
        with mock.patch("pygame.display.set_mode") as opened, self.assertRaises(RowSynthError) as refused:
            run(path, frames=1)
        self.assertEqual(str(refused.exception), f"{path}: no such file")
        opened.assert_not_called()


if __name__ == "__main__":
    unittest.main()
