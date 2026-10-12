"""Tests for what the standalone program runs: the command line as it is, and the window with no console."""
from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rowsynth import program
from rowsynth.gui import sprites

ROOT = Path(__file__).resolve().parents[1]

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


def builder():
    """The script that builds the standalone program, read as a module: it is no part of the package."""
    spec = importlib.util.spec_from_file_location("standalone_build", ROOT / "standalone" / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CommandLine(unittest.TestCase):
    def test_it_is_the_command_line_of_the_package_with_the_same_words_after_it(self) -> None:
        with mock.patch.object(program, "main", return_value=7) as ran:
            self.assertEqual(program.command_line(["check", "audio.json"]), 7)
        ran.assert_called_once_with(["check", "audio.json"])

    def test_with_no_words_it_is_the_command_line_still_unless_a_double_click_started_the_program(self) -> None:
        with mock.patch.object(program, "main", return_value=2) as ran, \
                mock.patch.object(program, "window") as opened, \
                mock.patch.object(program, "alone_in_console", return_value=True):
            # Run by Python, a console of its own is nothing to go by: it is some shell's that has ended.
            self.assertEqual(program.command_line([]), 2)
        ran.assert_called_once_with([])
        opened.assert_not_called()
        # The standalone program in a shell, which shares its console.
        with mock.patch.object(program, "main", return_value=2) as ran, \
                mock.patch.object(program, "frozen", return_value=True), \
                mock.patch.object(program, "alone_in_console", return_value=False):
            self.assertEqual(program.command_line([]), 2)
        ran.assert_called_once_with([])

    def test_started_by_a_double_click_it_lets_go_of_its_console_and_opens_the_window(self) -> None:
        streams = (sys.stdout, sys.stderr)
        self.addCleanup(lambda: setattr(sys, "stdout", streams[0]))
        self.addCleanup(lambda: setattr(sys, "stderr", streams[1]))
        with mock.patch.object(program, "main") as ran, mock.patch.object(program, "window", return_value=0) as opened, \
                mock.patch.object(program, "frozen", return_value=True), \
                mock.patch.object(program, "alone_in_console", return_value=True), \
                mock.patch.object(program, "ctypes") as system:
            self.assertEqual(program.command_line([]), 0)
            # What it printed on is gone with the console: it prints on nothing from then on.
            self.assertTrue(sys.stdout is sys.stderr and sys.stdout is not streams[0])
            sys.stdout.close()
        system.windll.kernel32.FreeConsole.assert_called_once_with()
        opened.assert_called_once_with([])
        ran.assert_not_called()

    def test_whether_it_is_the_standalone_program_goes_by_what_the_program_says_of_itself(self) -> None:
        self.assertFalse(program.frozen())
        with mock.patch.object(sys, "frozen", True, create=True):
            self.assertTrue(program.frozen())
        self.assertIsInstance(program.alone_in_console(), bool)
        self.assertEqual((program.COMMAND_LINE, program.WINDOW), ("rowsynth", "rowsynth-window"))


class Window(unittest.TestCase):
    def test_it_opens_the_window_as_the_gui_command_does_with_the_words_after_its_name(self) -> None:
        with mock.patch.object(program, "main", return_value=0) as ran, mock.patch.object(program, "say") as said:
            self.assertEqual(program.window(["audio.json", "--scale", "2"]), 0)
            self.assertEqual(program.window([]), 0)
        self.assertEqual(ran.call_args_list, [mock.call(["gui", "audio.json", "--scale", "2"]), mock.call(["gui"])])
        said.assert_not_called()

    def test_what_the_command_prints_as_an_error_is_said_in_a_box_since_there_is_no_console(self) -> None:
        def refuses(argv: list[str]) -> int:
            print("error: gone.json: no such file", file=sys.stderr)
            return 2

        before = sys.stderr
        with mock.patch.object(program, "main", side_effect=refuses), mock.patch.object(program, "say") as said:
            self.assertEqual(program.window(["gone.json"]), 2)
        said.assert_called_once_with("error: gone.json: no such file")
        # The process has the streams it had.
        self.assertIs(sys.stderr, before)

    def test_words_the_command_does_not_take_stop_its_reader_and_are_said_as_well(self) -> None:
        def stops(argv: list[str]) -> int:
            sys.stderr.write("usage: rowsynth gui [FILE]\nrowsynth gui: error: unrecognized arguments: --loud\n")
            raise SystemExit(2)

        with mock.patch.object(program, "main", side_effect=stops), mock.patch.object(program, "say") as said:
            self.assertEqual(program.window(["--loud"]), 2)
        self.assertIn("unrecognized arguments: --loud", said.call_args.args[0])

    def test_a_window_that_closed_as_it_should_says_nothing_and_a_program_with_no_output_has_one_to_print_on(
            self) -> None:
        seen = []

        def runs(argv: list[str]) -> int:
            # A program with no console has no standard output at all: what is printed must go somewhere.
            seen.append(sys.stdout)
            print("{}")
            return 0

        before = sys.stdout
        self.addCleanup(lambda: setattr(sys, "stdout", before))
        sys.stdout = None
        with mock.patch.object(program, "main", side_effect=runs), mock.patch.object(program, "say") as said:
            self.assertEqual(program.window([]), 0)
        self.assertEqual((sys.stdout, seen[0] is not None), (None, True))
        said.assert_not_called()

    def test_the_real_command_refuses_a_file_that_is_not_there_before_any_window_opens(self) -> None:
        with mock.patch.object(program, "say") as said, mock.patch.object(sys, "stdout", io.StringIO()):
            self.assertEqual(program.window(["no-such-folder/gone.json"]), 2)
        self.assertEqual(said.call_args.args[0], "error: no-such-folder/gone.json: no such file")


class Icon(unittest.TestCase):
    def test_the_program_s_picture_is_drawn_at_32_pixels_and_anew_at_16_in_the_one_palette(self) -> None:
        found = sprites.load("app")
        sizes = {name: (len(sprite.rows[0]), len(sprite.rows)) for name, sprite in found.sprites.items()}
        self.assertEqual(sizes, {"icon": (32, 32), "small": (16, 16)})
        # Its corners are left empty, so that the picture is a rounded one on whatever lies behind it.
        for sprite in found.sprites.values():
            last = len(sprite.rows) - 1
            self.assertEqual({sprite.rows[y][x] for x in (0, last) for y in (0, last)}, {sprites.EMPTY})

    def test_each_size_of_the_icon_is_a_whole_number_of_times_the_picture_it_is_made_of(self) -> None:
        build = builder()
        found = sprites.load("app").sprites
        self.assertEqual([side for side, _ in build.SIDES], [16, 32, 48, 64, 128, 256])
        for side, name in build.SIDES:
            self.assertEqual(side % len(found[name].rows), 0, side)
        self.assertRegex(build.version(), r"^\d+\.\d+\.\d+$")

    @unittest.skipUnless(Image, "Pillow is not installed")
    def test_the_icon_file_holds_every_size_with_no_pixel_of_it_smoothed(self) -> None:
        build = builder()
        with tempfile.TemporaryDirectory() as folder:
            path = build.icon(Path(folder) / "made" / "rowsynth.ico")
            with Image.open(path) as made:
                self.assertEqual(sorted(made.info["sizes"]), [(side, side) for side, _ in build.SIDES])
                made.size = (64, 64)
                made.load()
                picture = made.convert("RGBA")
        drawn = sprites.load("app").sprites["icon"]
        # A pixel of the picture is a square of two by two of the icon at 64, in its own colour and no other.
        for x, y in ((0, 0), (5, 6), (10, 7), (4, 20), (6, 20), (16, 16)):
            ink = drawn.rows[y][x]
            wanted = (0, 0, 0, 0) if ink == sprites.EMPTY else (*drawn.palette[ink], 255)
            self.assertEqual({picture.getpixel((2 * x + dx, 2 * y + dy)) for dx in (0, 1) for dy in (0, 1)}, {wanted})


if __name__ == "__main__":
    unittest.main()
