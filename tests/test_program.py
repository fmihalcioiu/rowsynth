"""Tests for what the standalone programs run: the window with no console, and the command line as it is."""
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

    def test_with_no_words_it_is_the_command_line_still_and_opens_no_window(self) -> None:
        # A double click on it is no way to the window: that is the other program's.
        with mock.patch.object(program, "main", return_value=2) as ran, \
                mock.patch.object(program, "window") as opened, \
                mock.patch.object(sys, "frozen", True, create=True):
            self.assertEqual(program.command_line([]), 2)
        ran.assert_called_once_with([])
        opened.assert_not_called()

    def test_with_nothing_given_it_takes_the_words_after_the_name_of_the_program(self) -> None:
        with mock.patch.object(program, "main", return_value=0) as ran, \
                mock.patch.object(sys, "argv", ["rowsynth-cli.exe", "info", "audio.json"]):
            self.assertEqual(program.command_line(), 0)
        ran.assert_called_once_with(["info", "audio.json"])

    def test_the_window_has_the_plain_name_and_the_command_line_is_in_the_folder_of_the_files(self) -> None:
        self.assertEqual((program.WINDOW, program.COMMAND_LINE, program.HELD), ("rowsynth", "rowsynth-cli", "bin"))


class Folder(unittest.TestCase):
    def test_the_command_line_is_moved_into_the_folder_of_the_files_and_the_window_stays(self) -> None:
        build = builder()
        ending = ".exe" if sys.platform == "win32" else ""
        with tempfile.TemporaryDirectory() as made:
            folder = Path(made)
            (folder / "bin").mkdir()
            for name in ("rowsynth", "rowsynth-cli"):
                (folder / (name + ending)).write_bytes(name.encode())
            window, line = build.settle(folder)
            self.assertEqual((window, line),
                             (folder / ("rowsynth" + ending), folder / "bin" / ("rowsynth-cli" + ending)))
            self.assertEqual((window.read_bytes(), line.read_bytes()), (b"rowsynth", b"rowsynth-cli"))
            # The window is the one program to be seen in the folder.
            self.assertEqual(sorted(path.name for path in folder.iterdir()), ["bin", "rowsynth" + ending])


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
