"""Tests for rowsynth.gui.exports: the files a write will make, named beforehand, and the window that writes them."""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from rowsynth.bank import render_entry
from rowsynth.cli import main
from rowsynth.errors import RowSynthError
from rowsynth.gui import exports, sprites, theme
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.exports import Export, File, open_export
from rowsynth.gui.font import Font
from rowsynth.gui.layout import Rect
from rowsynth.gui.model import Model
from rowsynth.gui.session import Session
from rowsynth.gui.ui import Event, Ui
from rowsynth.project import Project
from rowsynth.song import render_song
from rowsynth.wav import load_wav

# pygame reads these when it is imported: no test opens a window or plays a sound.
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
try:
    import pygame
except ImportError:
    pygame = None

NEEDS = "pygame is not installed"
# With three ticks to a row, a row is 400 samples at this rate.
RATE = 8000


def sound(note: str, inst: str = "beep", ticks: int = 4) -> dict:
    """A def of one voice playing one note from the start."""
    return {"voices": [{"inst": inst, "notes": [[0, note, ticks]]}]}


def held(start: int, loop: int, ticks: int) -> dict:
    """A held sound whose note lasts `ticks`, cut into a start and a loop of so many ticks."""
    return {"held": {"note": sound("a3", "hum", ticks), "start": start, "loop": loop, "join": 0,
                     "end": sound("a3", "hum"), "let_go": 3}}


# One bar of two channels: its 16 rows are 6400 samples, and it is the song that is picked as the file is opened.
TUNE = {"row_ticks": 3,
        "channels": {"lead": {"inst": "beep", "pan": -0.5}, "bass": {"inst": "hum", "pan": 0.5}},
        "patterns": {"a": {"lead": "l4 c e g > c", "bass": "o3 c2 g2"}}, "order": ["a"]}
# The same bar with drums, in layers: two groups that have a file a key, one of drums that has one file, a
# version mixed ahead, and an intro.
BAND = {"row_ticks": 3,
        "drums": {"k": {"inst": "boom", "note": "d2"}},
        "channels": {"lead": {"inst": "beep"}, "bass": {"inst": "hum"}, "kit": {"inst": "boom", "drums": True}},
        "patterns": {"a": {"lead": "l4 c e g > c", "bass": "o3 c2 g2", "kit": "l4 k k k k"}}, "order": ["a"],
        "layers": {"groups": {"top": ["lead"], "low": ["bass"], "beat": ["kit"]},
                   "versions": {"calm": ["low"], "full": ["top", "low", "beat"]}, "keys": [0, 2],
                   "premix": [{"version": "full", "key": 2}]},
        "intro": {"order": ["a"], "tail": 0.2}}
PROJECT = {
    "instruments": {"beep": {"vol": [15, 12, 9, 6]}, "hum": {"wave": "tri", "vol": 12},
                    "boom": {"wave": "noise", "vol": [15, 9, 4], "release": 0}},
    "sounds": {"zap": {"def": sound("a4")}, "trio": {"defs": [sound("c5"), sound("e5")]}, "drone": held(2, 6, 12)},
    "songs": {"tune": TUNE, "band": BAND},
}
# A project with a sound the package refuses to render: its note ends long before its start and its loop do.
BROKEN = {**PROJECT, "sounds": {"short": held(30, 30, 2)}}


def centre(rect: Rect) -> tuple[int, int]:
    """The pixel in the middle of a rectangle."""
    return rect[0] + rect[2] // 2, rect[1] + rect[3] // 2


def click(ui: Ui, rect: Rect) -> None:
    """Press the left button in the middle of a rectangle and let it go there."""
    ui.handle(Event("down", centre(rect), button=1))
    ui.handle(Event("up", centre(rect), button=1))


def type_in(ui: Ui, text: str) -> None:
    """Type characters as the window tells them: each as the key that went down and then as its character."""
    for char in text:
        ui.handle(Event("key", key=char))
        ui.handle(Event("text", text=char))


def names(files: tuple[File, ...] | list[File]) -> list[str]:
    """The names of files, in order."""
    return sorted(file.name for file in files)


def found(folder: Path) -> dict[str, int]:
    """Every file under a folder with its size, by its place in the folder with a "/" after each folder."""
    return {path.relative_to(folder).as_posix(): path.stat().st_size for path in folder.rglob("*") if path.is_file()}


def blank() -> types.SimpleNamespace:
    """A stand-in for a canvas that nothing is drawn on: the Ui asks it for the size of its surface only."""
    return types.SimpleNamespace(surface=types.SimpleNamespace(get_size=lambda: theme.DESIGN))


class Case(unittest.TestCase):
    """A project file in a folder of its own, and the window that writes files open on it."""

    def setUp(self) -> None:
        self.folder = self.empty()
        self.path = self.folder / "project.json"
        # Where the window puts its files until it is told another folder.
        self.out = self.folder / "out"
        self.project = Project.from_dict(PROJECT)
        self.project.save(self.path)

    def empty(self) -> Path:
        """A new folder with nothing in it, gone again after the test, once no write is under way."""
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.addCleanup(self.settle)
        return Path(folder.name)

    @staticmethod
    def settle() -> None:
        """Waits for every write that is still on a thread of a window's."""
        for thread in threading.enumerate():
            if thread.name == "rowsynth-export":
                thread.join()

    def opened(self, start: str = "render", data: dict = PROJECT,
               picked: tuple[str, str] | None = None) -> tuple[Ui, Export]:
        """A Ui with the window open on it at a tab, on a project rendered at RATE, with a thing picked."""
        Project.from_dict(data).save(self.path)
        model = Model(self.path, RATE)
        if picked is not None:
            model.pick(*picked)
        self.session = Session(model)
        ui = Ui()
        return ui, open_export(ui, self.session, start)

    def command(self, *argv: str) -> None:
        """Runs a command of the command line, which is to go well, and lets go of what it prints."""
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(list(argv)), 0)

    def assert_written(self, files: list[File], folder: Path) -> None:
        """The files that were named are the files a folder holds, and each size that was known is the true one."""
        written = found(folder)
        self.assertEqual(names(files), sorted(written))
        for file in files:
            if file.size is not None:
                self.assertEqual((file.name, file.size), (file.name, written[file.name]))

    def waiting(self) -> threading.Event:
        """Has every write wait before it begins, until the event that is given back is set."""
        go, real = threading.Event(), exports.write
        self.addCleanup(go.set)

        def write(project: Project, made: exports.Plan) -> int:
            go.wait()
            return real(project, made)

        patched = mock.patch.object(exports, "write", write)
        patched.start()
        self.addCleanup(patched.stop)
        return go


class Named(Case):
    def test_the_files_of_the_first_tab_are_those_the_render_command_writes(self) -> None:
        # A song is 16 rows and half a second more in stereo, a loop of it 16 rows in mono, and a sound unknown.
        cases = ((("song", "tune"), False, [41644]), (("song", "tune"), True, [12844, 12844]),
                 (("sound", "zap"), False, [None]), (("sound", "drone"), False, [None]))
        for picked, stems, sizes in cases:
            with self.subTest(picked=picked, stems=stems):
                files = exports.render_files(self.project, *picked, RATE, stems)
                self.assertEqual([file.size for file in files], sizes)
                folder = self.empty()
                out = folder if stems else folder / files[0].name
                self.command("render", str(self.path), picked[1], "--out", str(out), "--rate", str(RATE),
                             *(["--stems"] if stems else []))
                self.assert_written(files, folder)

    def test_a_variant_and_a_part_are_named_as_the_page_names_them_and_written_as_the_command_writes_them(self) -> None:
        page = names(exports.page_files(self.project, RATE))
        cases = ((("sound", "trio"), {"variant": 1}, "trio-1.wav"),
                 (("sound", "drone"), {"part": "loop"}, "drone-loop.wav"))
        for picked, which, name in cases:
            with self.subTest(name=name):
                made = exports.plan(self.project, "render", self.out, RATE, picked=picked, **which)
                self.assertEqual((names(made.files), made.refusal), ([name], None))
                self.assertIn(f"audio/{name}", page)
                self.assertEqual(exports.write(self.project, made), 1)
                told = self.folder / "told.wav"
                flag, value = next(iter(which.items()))
                self.command("render", str(self.path), picked[1], "--out", str(told), "--rate", str(RATE),
                             f"--{flag}", str(value))
                self.assertEqual((self.out / name).read_bytes(), told.read_bytes())
        # Left out, they are the first variant, and the file has the name alone.
        self.assertEqual(names(exports.render_files(self.project, "sound", "trio", RATE)), ["trio.wav"])

    def test_the_files_of_a_build_are_those_the_build_command_writes(self) -> None:
        files = exports.build_files(self.project, RATE)
        folder = self.empty()
        self.command("build", str(self.path), "--out", str(folder), "--rate", str(RATE))
        self.assert_written(files, folder)
        # A file a group and key, one for the drums, the version mixed ahead and the intro, and the plain song.
        self.assertEqual((len(files), files[-1]), (9, File("music.json")))
        self.assertEqual([file.size for file in files if file.name in ("tune.wav", "band.intro.wav")], [25644, None])
        self.assertEqual(names(exports.build_files(self.project, RATE, "opus", ["tune"])), ["music.json", "tune.opus"])

    def test_the_files_of_the_page_are_those_the_page_command_writes(self) -> None:
        files = exports.page_files(self.project, RATE)
        folder = self.empty()
        self.command("page", str(self.path), "--out", str(folder), "--rate", str(RATE))
        self.assert_written(files, folder)
        self.assertEqual((len(files), files[-1]), (14, File("index.html")))
        # The start of a held sound is cut to a length that is known, and one of no samples has no file.
        sizes = {file.name: file.size for file in files}
        self.assertEqual((sizes["audio/drone-start.wav"], sizes["audio/zap.wav"]), (44 + 2 * 267, None))
        gone = Project.from_dict({**PROJECT, "sounds": {"drone": held(0, 6, 12)}})
        self.assertNotIn("audio/drone-start.wav", names(exports.page_files(gone, RATE)))

    def test_a_file_that_is_there_already_is_marked_as_replaced(self) -> None:
        _, window = self.opened()
        self.assertEqual([(row.text, row.tag) for row in window.files.rows], [("tune.wav", "41 KB")])
        self.assertEqual(window.said(), ("1 file to write", theme.DIM))
        self.out.mkdir()
        (self.out / "tune.wav").write_bytes(b"old")
        window.refresh()
        self.assertEqual(window.plan.files, (File("tune.wav", 41644, True),))
        self.assertEqual([(row.text, row.tag) for row in window.files.rows], [("tune.wav", "replaces 41 KB")])
        self.assertEqual(window.said(), ("1 file to write, 1 in place of what is there", theme.DIM))


class Refused(Case):
    def test_what_cannot_be_done_is_said_before_it_is_tried_and_write_is_not_enabled(self) -> None:
        ui, window = self.opened("page")
        with mock.patch.object(shutil, "which", return_value=None):
            window.page_format.pick(1)
        said = ("ffmpeg is needed to write mp3, and none is found on the path", theme.FAULT)
        self.assertEqual((window.said(), window.write.enabled, window.write.help), (said, False, said[0]))
        # The files are listed all the same, and a press of the button writes none of them.
        self.assertIn("audio/zap.mp3", names(window.plan.files))
        click(ui, window.write.rect)
        self.assertEqual((window.worker, self.out.exists()), (None, False))
        # A file where the folder is to be made is in the way of every format.
        window.page_format.pick(0)
        self.assertTrue(window.write.enabled)
        window.folder = self.path / "out"
        window.refresh()
        said = (f"{self.path}: a file of this name is in the way of the folder", theme.FAULT)
        self.assertEqual((window.said(), window.write.enabled), (said, False))
        # Nothing picked that the first tab could write, and Opus at a rate it does not take.
        self.session.model.pick("instrument", "beep")
        window.tabs.pick(0)
        self.assertEqual(window.said(), (exports.NO_PICK, theme.FAULT))
        self.assertIn("opus takes a rate", exports.plan(self.project, "build", self.out, 44100, "opus").refusal)

    def test_a_refusal_of_the_package_is_shown_and_nothing_is_left_behind(self) -> None:
        ui, window = self.opened(data=BROKEN, picked=("sound", "short"))
        self.assertEqual((names(window.plan.files), window.plan.refusal), (["short-start.wav"], None))
        click(ui, window.write.rect)
        window.worker.join()
        words, colour = window.said()
        self.assertTrue(words.startswith("held: note: must be long enough for the start and the loop"), words)
        self.assertEqual((colour, window.write.enabled, self.out.exists()), (theme.FAULT, True, False))
        with self.assertRaisesRegex(RowSynthError, "opens at one of render, build, page, not at 'zip'"):
            open_export(ui, self.session, "zip")

    def test_a_variant_or_a_part_the_picked_thing_does_not_have_is_why_nothing_can_be_written(self) -> None:
        cases = ((("sound", "trio"), {"variant": 2}, "'trio' has no variant 2: it has 2, counted from 0"),
                 (("sound", "zap"), {"part": "loop"}, "--part is for a held sound, and 'zap' is not one"),
                 (("sound", "drone"), {"variant": 0}, "--variant is for a sound that is not held"),
                 (("sound", "drone"), {"part": "middle"}, "a held sound has the parts start, loop, end, and no part"),
                 (("song", "tune"), {"variant": 0}, "--variant is for a sound, and 'tune' is a song"),
                 (("song", "tune"), {"part": "end"}, "--part is for a held sound, and 'tune' is a song"))
        for picked, which, why in cases:
            with self.subTest(picked=picked, which=which):
                made = exports.plan(self.project, "render", self.out, RATE, picked=picked, **which)
                self.assertEqual((made.files, made.refusal[:len(why)]), ((), why))
                with self.assertRaises(RowSynthError) as refused:
                    exports.write(self.project, made)
                self.assertEqual(str(refused.exception), made.refusal)
        self.assertFalse(self.out.exists())


class Chosen(Case):
    def test_a_sound_of_several_variants_has_a_stepper_from_the_default_to_each_by_its_number(self) -> None:
        ui, window = self.opened(picked=("sound", "trio"))
        row = exports.parts(ui.size)["row1"]
        self.assertEqual((window.variant.visible, window.part.visible, window.stems.visible), (True, False, False))
        self.assertEqual((window.variant.options, window.variant.rect, window.words()),
                         (("default", 0, 1), row, ("PICKED", "VARIANT", "")))
        self.assertEqual((window.plan.variant, names(window.plan.files)), (None, ["trio.wav"]))
        # Each press on its right half goes one on, no further than the last, and the list of files follows.
        x, y, w, h = row
        seen = []
        for _ in range(3):
            click(ui, (x + w // 2, y, w // 2, h))
            seen.append((window.plan.variant, [line.text for line in window.files.rows]))
        self.assertEqual(seen, [(0, ["trio-0.wav"]), (1, ["trio-1.wav"]), (1, ["trio-1.wav"])])
        click(ui, window.write.rect)
        window.worker.join()
        self.assertEqual((window.said(), sorted(found(self.out))),
                         ((f"wrote 1 file to {self.out}", theme.GOOD), ["trio-1.wav"]))
        np.testing.assert_allclose(load_wav(self.out / "trio-1.wav")[0][0],
                                   render_entry(self.project.sound("trio"), RATE)[1], atol=1 / 32767)

    def test_a_held_sound_has_a_choice_of_its_three_parts_and_each_is_a_file_of_its_own(self) -> None:
        ui, window = self.opened(picked=("sound", "drone"))
        self.assertEqual((window.part.visible, window.variant.visible, window.stems.visible), (True, False, False))
        self.assertEqual((window.part.options, window.part.rect, window.words()),
                         (("START", "LOOP", "END"), exports.parts(ui.size)["row1"], ("PICKED", "PART", "")))
        self.assertEqual((window.plan.part, names(window.plan.files)), ("start", ["drone-start.wav"]))
        ui.handle(Event("move", centre(window.part.rect)))
        self.assertEqual(window.help, exports.HELP["part"])
        click(ui, window.part.cells()[1])
        self.assertEqual((window.plan.part, [line.text for line in window.files.rows]), ("loop", ["drone-loop.wav"]))
        click(ui, window.write.rect)
        window.worker.join()
        # The loop is three times as long as the start, so it is not the start that was written.
        _, loop, _ = render_entry(self.project.sound("drone"), RATE)
        np.testing.assert_allclose(load_wav(self.out / "drone-loop.wav")[0][0], loop, atol=1 / 32767)

    def test_neither_is_shown_for_a_song_or_for_a_sound_that_has_no_use_for_it(self) -> None:
        _, window = self.opened(picked=("sound", "trio"))
        window.variant.step(2)
        self.assertEqual((window.plan.variant, window.variant.help), (1, exports.HELP["variant"]))
        # The switch of a song, the variants and the parts, as each thing is picked in turn.
        cases = ((("song", "tune"), (True, False, False), ""), (("sound", "zap"), (False, False, False), ""),
                 (("sound", "drone"), (False, False, True), "PART"),
                 (("sound", "trio"), (False, True, False), "VARIANT"))
        for picked, shown, word in cases:
            with self.subTest(picked=picked):
                self.session.model.pick(*picked)
                window.said()
                self.assertEqual((window.stems.visible, window.variant.visible, window.part.visible), shown)
                self.assertEqual((window.words()[1], window.plan.refusal), (word, None))
        # The variants were other ones in between, so the sound is at its default again.
        self.assertEqual((window.variant.picked, window.plan.variant, names(window.plan.files)),
                         (0, None, ["trio.wav"]))
        window.tabs.pick(1)
        self.assertEqual((window.variant.visible, window.part.visible, window.words()),
                         (False, False, exports.WORDS["build"]))


class Written(Case):
    def test_write_on_the_first_tab_makes_the_wav_file_the_package_renders_of_the_picked_thing(self) -> None:
        ui, window = self.opened()
        click(ui, window.write.rect)
        window.worker.join()
        self.assertEqual(window.said(), (f"wrote 1 file to {self.out}", theme.GOOD))
        sides, rate = load_wav(self.out / "tune.wav")
        want = render_song(self.project.song("tune"), rate=RATE)
        self.assertEqual((rate, len(sides)), (RATE, 2))
        for side, wanted in zip(sides, (want["left"], want["right"])):
            np.testing.assert_allclose(side, wanted, atol=1 / 32767)

    def test_one_loop_per_channel_makes_one_mono_file_a_channel(self) -> None:
        ui, window = self.opened()
        click(ui, window.stems.rect)
        self.assertEqual(names(window.plan.files), ["bass.wav", "lead.wav"])
        click(ui, window.write.rect)
        window.worker.join()
        self.assertEqual(window.said(), (f"wrote 2 files to {self.out}", theme.GOOD))
        self.assertEqual(sorted(found(self.out)), ["bass.wav", "lead.wav"])
        self.assertEqual([len(load_wav(self.out / name)[0]) for name in ("bass.wav", "lead.wav")], [1, 1])
        # A sound has no channels: the switch is not there for it, and the sound is the one file.
        self.session.model.pick("sound", "zap")
        self.assertEqual((window.said()[0], window.stems.visible, names(window.plan.files)),
                         ("1 file to write", False, ["zap.wav"]))

    def test_a_build_as_wav_writes_the_manifest_and_the_files_it_names(self) -> None:
        ui, window = self.opened("build")
        window.rate.step(-len(exports.RATES))
        window.songs.step(1)
        self.assertEqual((window.plan.rate, names(window.plan.files)), (RATE, ["music.json", "tune.wav"]))
        window.songs.step(-1)
        listed = names(window.plan.files)
        click(ui, window.write.rect)
        window.worker.join()
        self.assertEqual(window.said(), (f"wrote 9 files to {self.out}", theme.GOOD))
        manifest = json.loads((self.out / "music.json").read_text(encoding="utf-8"))
        self.assertEqual((manifest["rate"], manifest["format"], list(manifest["songs"])),
                         (RATE, "wav", ["tune", "band"]))
        self.assertEqual((manifest["songs"]["tune"]["file"], sorted(found(self.out))), ("tune.wav", listed))

    def test_the_line_says_what_is_being_written_and_then_how_many_files_and_where(self) -> None:
        go = self.waiting()
        ui, window = self.opened()
        click(ui, window.write.rect)
        worker = window.worker
        self.assertEqual(window.said(), (f"writing tune: 1 file to {self.out}", theme.ACTIVE))
        self.assertEqual((window.busy, window.write.enabled), (True, False))
        go.set()
        worker.join()
        self.assertEqual(window.said(), (f"wrote 1 file to {self.out}", theme.GOOD))
        # The list is of the folder as it is now, and another write can begin.
        self.assertEqual((window.busy, window.write.enabled, window.plan.files[0].replaces), (False, True, True))


class Closed(Case):
    def test_closed_while_it_writes_the_write_goes_on_and_the_session_says_how_it_ended(self) -> None:
        go = self.waiting()
        ui, window = self.opened()
        click(ui, window.write.rect)
        worker = window.worker
        click(ui, window.shut.rect)
        # It is no window any more: it only waits, and takes nothing from the mouse.
        self.assertEqual((ui.modal, ui.widgets, window.hit(centre(window.rect))), (None, [window], False))
        ui.draw(blank())
        self.assertEqual(self.session.message, "")
        go.set()
        worker.join()
        ui.draw(blank())
        self.assertEqual((self.session.message, ui.widgets), (f"wrote 1 file to {self.out}", []))
        self.assertEqual(sorted(found(self.out)), ["tune.wav"])

    def test_the_folder_is_typed_into_a_box_and_escape_closes_the_box_and_then_the_window(self) -> None:
        ui, window = self.opened("page")
        self.assertEqual((ui.modal, window.tabs.picked, window.folder, window.takes_text), (window, 2, self.out, False))
        click(ui, window.change.rect)
        self.assertEqual((window.takes_text, window.help), (True, exports.HELP["box"]))
        # The box opens with the folder picked, so what is typed takes its place.
        type_in(ui, "spare")
        ui.handle(Event("key", key="return"))
        self.assertEqual((window.folder, window.plan.folder, window.takes_text), (Path("spare"), Path("spare"), False))
        click(ui, window.change.rect)
        type_in(ui, "x")
        ui.handle(Event("key", key="escape"))
        self.assertEqual((window.folder, window.takes_text, ui.modal), (Path("spare"), False, window))
        ui.handle(Event("key", key="escape"))
        self.assertEqual((ui.modal, ui.widgets), (None, []))


@unittest.skipUnless(pygame, NEEDS)
class Drawn(Case):
    def inked(self, surface: pygame.Surface, rect: Rect, colour: theme.Colour) -> bool:
        """Whether any pixel of a rectangle has a colour."""
        x, y, w, h = rect
        return any(tuple(surface.get_at((x + right, y + down)))[:3] == colour
                   for right in range(w) for down in range(h))

    def test_each_tab_is_drawn_and_what_cannot_be_done_is_in_the_colour_of_a_fault(self) -> None:
        surface = pygame.Surface(theme.DESIGN)
        canvas = Canvas(surface, Font(sprites.load("font")), sprites.load("icons"))
        ui, window = self.opened()
        at = exports.parts(theme.DESIGN)
        for tab in range(len(exports.TABS)):
            window.tabs.pick(tab)
            ui.draw(canvas)
            self.assertEqual(tuple(surface.get_at(at["box"][:2]))[:3], theme.LINE)
            self.assertEqual((self.inked(surface, at["said"], theme.DIM), self.inked(surface, at["said"], theme.FAULT)),
                             (True, False))
        with mock.patch.object(shutil, "which", return_value=None):
            window.page_format.pick(2)
            ui.draw(canvas)
        self.assertTrue(self.inked(surface, at["said"], theme.FAULT))

    def test_the_second_row_of_a_held_sound_is_drawn_as_its_parts_with_their_word_before_them(self) -> None:
        surface = pygame.Surface(theme.DESIGN)
        canvas = Canvas(surface, Font(sprites.load("font")), sprites.load("icons"))
        ui, window = self.opened(picked=("sound", "drone"))
        at = exports.parts(theme.DESIGN)
        ui.draw(canvas)
        # The word is dim, and the part that is on has the colour of what is active.
        self.assertEqual((self.inked(surface, at["word1"], theme.DIM), self.inked(surface, at["row1"], theme.ACTIVE)),
                         (True, True))
        # A sound of one def has nothing in that row.
        self.session.model.pick("sound", "zap")
        ui.draw(canvas)
        self.assertEqual([self.inked(surface, at[part], colour) for part, colour in
                          (("word1", theme.DIM), ("row1", theme.ACTIVE), ("row1", theme.TEXT))], [False] * 3)


if __name__ == "__main__":
    unittest.main()
