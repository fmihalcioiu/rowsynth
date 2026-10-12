"""Tests for the GUI's window: its scale, its art surface, and the app run in it for a few frames."""
from __future__ import annotations

import os
import sys
import tempfile
import threading
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
    from rowsynth.gui import screen as gui_screen
    from rowsynth.gui.screen import Screen, dpi_aware, fit, opening, work_area

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
NEEDS = "pygame is not installed"


def settle() -> None:
    """Waits for every render and every run of the checks that is still on a thread of a model's."""
    for thread in threading.enumerate():
        if thread.name.startswith("rowsynth-"):
            thread.join()


@unittest.skipUnless(pygame, NEEDS)
class Fit(unittest.TestCase):
    def test_a_full_hd_window_shows_the_size_the_layout_is_drawn_for_at_twice_its_size(self) -> None:
        self.assertEqual(theme.DESIGN, (960, 540))
        self.assertEqual(fit((1920, 1080)), (2, (960, 540)))

    def test_a_window_that_fills_a_full_hd_desktop_under_its_title_bar_is_still_at_twice(self) -> None:
        self.assertEqual(fit((1920, 1009)), (2, (960, 504)))
        # The least a scale of 2 is chosen at.
        self.assertEqual(fit((1920, 960)), (2, theme.MIN_ART))

    def test_a_window_with_no_room_for_twice_keeps_every_pixel_of_its_own_at_scale_1(self) -> None:
        # The last two are smaller than the least art size, and are not scaled down to hold it.
        for size in ((1280, 720), (960, 540), (1919, 1080), (1920, 959), (800, 600), (640, 360)):
            with self.subTest(size=size):
                self.assertEqual(fit(size), (1, size))

    def test_a_larger_window_takes_the_largest_scale_that_leaves_the_least_art_size(self) -> None:
        self.assertEqual(fit((2560, 1440)), (2, (1280, 720)))
        self.assertEqual(fit((2880, 1440)), (3, (960, 480)))
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
class Opening(unittest.TestCase):
    def test_a_window_opens_at_the_size_the_layout_is_drawn_for_times_the_largest_scale_with_room(self) -> None:
        self.assertEqual(opening((2560, 1440)), (1920, 1080))
        self.assertEqual(opening((1366, 768)), (960, 540))

    def test_it_is_cut_down_to_the_desktop_less_what_its_bars_take_and_keeps_its_scale(self) -> None:
        self.assertEqual(theme.DESKTOP_TAKEN_H, 96)
        for desktop, size, scale in (((1920, 1080), (1920, 984), 2), ((3840, 2160), (3840, 2064), 4)):
            with self.subTest(desktop=desktop):
                self.assertEqual(opening(desktop), size)
                self.assertEqual(fit(size)[0], scale)

    def test_on_a_desktop_too_small_for_scale_1_it_opens_at_what_fits(self) -> None:
        self.assertEqual(opening((800, 600)), (800, 504))

    def test_a_scale_that_is_given_takes_the_place_of_the_largest(self) -> None:
        self.assertEqual(opening((2560, 1440), 1), (960, 540))
        self.assertEqual(opening((3840, 2160), 2), (1920, 1080))
        # One too large for the desktop is kept, and the window is cut down.
        self.assertEqual(opening((1920, 1080), 3), (1920, 984))


@unittest.skipUnless(pygame, NEEDS)
class DpiAware(unittest.TestCase):
    def told(self, platform: str, **fails: object) -> mock.Mock:
        """What was asked of ctypes by `dpi_aware` on a system of that name. `fails` is how the asking goes wrong."""
        with mock.patch.object(sys, "platform", platform), mock.patch("rowsynth.gui.screen.ctypes") as ctypes:
            ctypes.windll.user32.SetProcessDPIAware.configure_mock(**fails)
            dpi_aware()
        return ctypes

    def test_on_windows_the_system_is_told_once(self) -> None:
        self.assertEqual(self.told("win32").mock_calls, [mock.call.windll.user32.SetProcessDPIAware()])

    def test_on_another_system_nothing_is_asked(self) -> None:
        for platform in ("linux", "darwin"):
            with self.subTest(platform=platform):
                self.assertEqual(self.told(platform).mock_calls, [])

    def test_a_failure_to_tell_it_is_passed_over(self) -> None:
        asked = self.told("win32", side_effect=OSError("no such thing"))
        asked.windll.user32.SetProcessDPIAware.assert_called_once_with()


@unittest.skipUnless(pygame, NEEDS)
class Window(unittest.TestCase):
    def opened(self, *args: object, **named: object) -> Screen:
        """A screen that is closed again when the test is over."""
        screen = Screen(*args, **named)
        self.addCleanup(screen.close)
        return screen

    def test_with_no_size_given_it_opens_at_the_size_that_fits_the_first_desktop(self) -> None:
        started = []

        def told() -> None:
            started.append(pygame.display.get_init())

        pygame.display.quit()
        with mock.patch("pygame.display.get_desktop_sizes", return_value=[(2560, 1440), (1080, 1920)]), \
                mock.patch("rowsynth.gui.screen.dpi_aware", side_effect=told):
            screen = self.opened()
            self.assertEqual(pygame.display.get_surface().get_size(), (1920, 1080))
            self.assertTrue(pygame.display.get_surface().get_flags() & pygame.RESIZABLE)
            self.assertEqual((screen.scale, screen.begin().get_size()), (2, (960, 540)))
            # A scale that is given is the one the size is worked out at.
            screen.close()
            screen = self.opened(scale=1)
            self.assertEqual((pygame.display.get_surface().get_size(), screen.scale), ((960, 540), 1))
        # Each time, the system was told whose the pixels are before the display was started.
        self.assertEqual(started, [False, False])

    def test_it_opens_with_no_frame_of_the_systems_and_is_moved_sized_and_made_to_fill_the_desktop(self) -> None:
        screen = self.opened((960, 540))
        flags = pygame.display.get_surface().get_flags()
        self.assertEqual((screen.frameless, screen.big, bool(flags & pygame.NOFRAME), bool(flags & pygame.RESIZABLE)),
                         (True, False, True, True))
        screen.put((10, 20, 800, 500))
        self.assertEqual((screen.stand(), screen.begin().get_size()), ((10, 20, 800, 500), (800, 500)))
        # pygame points what it tells of the window at the handle it gave out last, so that one is held on to.
        told = [event.window for event in pygame.event.get() if getattr(event, "window", None) is not None]
        self.assertTrue(told and all(window is gui_screen._HANDLES[0] for window in told))
        self.assertEqual(len(gui_screen._HANDLES), 1)
        # The window fills what the task bar leaves of the display its middle is on, and is put back from there.
        work = (0, 0, 1920, 1032)
        with mock.patch("pygame.display.get_desktop_sizes", return_value=[(1920, 1080)]), \
                mock.patch.object(gui_screen, "work_area", return_value=work) as asked:
            screen.fill()
            self.assertEqual((screen.stand(), screen.big, screen.kept), (work, True, (10, 20, 800, 500)))
            screen.fill()
        # It was asked with the middle of the window as it stood.
        self.assertEqual(asked.call_args_list[0], mock.call((410, 270), (1920, 1080)))
        self.assertEqual((screen.stand(), screen.big), ((10, 20, 800, 500), False))
        # Taken off the screen it stands where it stood. The dummy display has no pointers, which is passed over.
        screen.minimise()
        for shape in ("ew", "nwse", ""):
            screen.pointer(shape)
        self.assertEqual(screen.stand(), (10, 20, 800, 500))

    def test_where_pygame_cannot_stand_in_for_the_frame_the_window_keeps_the_systems_and_stays_where_it_is(
            self) -> None:
        # A pygame with no handle of the window at all, and one whose handle cannot be reached after all.
        for lacking in (mock.patch.object(gui_screen, "own_frame", return_value=None),
                        mock.patch.object(Screen, "_reach", side_effect=RuntimeError("no such window"))):
            with lacking:
                screen = self.opened((960, 540))
            flags = pygame.display.get_surface().get_flags()
            self.assertEqual((screen.frameless, bool(flags & pygame.NOFRAME), bool(flags & pygame.RESIZABLE)),
                             (False, False, True))
            screen.put((10, 20, 800, 500))
            screen.fill()
            screen.minimise()
            self.assertEqual((screen.stand(), screen.big, pygame.display.get_surface().get_size()),
                             (None, False, (960, 540)))
            screen.close()

    def test_the_part_of_the_desktop_a_window_may_fill_is_all_of_it_where_windows_cannot_be_asked(self) -> None:
        with mock.patch.object(sys, "platform", "linux"):
            self.assertEqual(work_area((100, 100), (1920, 1080)), (0, 0, 1920, 1080))
        refused = mock.patch.object(gui_screen.ctypes, "WinDLL", create=True, side_effect=OSError("no such thing"))
        with mock.patch.object(sys, "platform", "win32"), refused:
            self.assertEqual(work_area((100, 100), (1366, 768)), (0, 0, 1366, 768))

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

    def test_the_least_size_of_the_window_is_the_least_art_size_at_the_scale_it_opened_with(self) -> None:
        self.assertEqual(theme.LEAST_ART, (640, 400))
        self.assertEqual((self.opened((1920, 1080)).least, self.opened((960, 540)).least, self.opened(scale=3).least),
                         ((1280, 800), (640, 400), (1920, 1200)))

    def test_the_art_surface_follows_the_window_to_a_new_size(self) -> None:
        screen = self.opened((1920, 1080))
        # What a resize leaves behind it: the window's own surface at another size.
        pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
        # The scale the window opened with stays: a smaller window shows less, and nothing smaller.
        self.assertEqual((screen.begin().get_size(), screen.scale), ((640, 360), 2))
        self.assertEqual(screen.to_art((700, 30)), (350, 15))
        # Nor does a larger window show anything larger than the window did as it opened.
        screen = self.opened((1280, 720))
        pygame.display.set_mode((2560, 1440), pygame.RESIZABLE)
        self.assertEqual((screen.begin().get_size(), screen.scale), ((2560, 1440), 1))
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
        # A run leaves its render and its checks on their threads: none is left to run into the next test.
        self.addCleanup(settle)
        # The desktop is a full HD one, whatever the machine the tests are run on has.
        desktop = mock.patch("pygame.display.get_desktop_sizes", return_value=[(1920, 1080)])
        desktop.start()
        self.addCleanup(desktop.stop)

    def test_run_for_3_frames_returns_0_and_its_shot_is_a_png_of_the_frame(self) -> None:
        # The folder of the shot is not there yet, and its name does not end as that of a PNG does.
        shot = self.folder / "shots" / "frame"
        self.assertEqual(run(EXAMPLES / "arcade.json", frames=3, shot=shot), 0)
        self.assertIsNone(pygame.display.get_surface())
        self.assertEqual(shot.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        with open(shot, "rb") as file:
            drawn = pygame.image.load(file, ".png")
        # The window was 1920 by 984, to fit the desktop, and so at twice.
        self.assertEqual(drawn.get_size(), (960, 492))
        # The title row and its line, the top bar and its line, the left panel's line, the middle, the right
        # panel's line and the status bar's.
        self.assertEqual([drawn.get_at(pos)[:3] for pos in ((0, 0), (0, 13), (0, 14), (0, 35), (123, 36), (124, 36),
                                                            (786, 300), (787, 300), (500, 478), (500, 491))],
                         [theme.BACKGROUND, theme.LINE, theme.PANEL, theme.LINE, theme.LINE, theme.BACKGROUND,
                          theme.LINE, theme.PANEL, theme.LINE, theme.PANEL])

    def test_a_run_for_no_frames_is_refused_before_any_window_is_opened(self) -> None:
        for frames in (0, -2):
            with self.subTest(frames=frames), mock.patch("pygame.display.set_mode") as opened:
                with self.assertRaises(RowSynthError) as refused:
                    run(EXAMPLES / "arcade.json", frames=frames)
                self.assertEqual(str(refused.exception), f"a run of {frames} frames shows nothing: the count of "
                                                         "frames is a whole number, 1 or more")
                opened.assert_not_called()

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
