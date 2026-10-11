"""Tests for the files of the window: what a folder holds, the chooser, the first screen, the files opened last."""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import threading
import unittest
from collections.abc import Hashable
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

# No test opens a window that can be seen, or a sound device. This is said before pygame is first imported.
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from rowsynth import checks, cli
from rowsynth.bank import render_entry
from rowsynth.errors import RowSynthError
from rowsynth.gui import files, layout, start, theme
from rowsynth.gui import model as gui_model
from rowsynth.gui.exports import Export
from rowsynth.gui.lists import List
from rowsynth.gui.popups import Menu
from rowsynth.gui.ui import Event, Ui
from rowsynth.project import Project

try:
    import pygame
except ImportError:
    pygame = None

if TYPE_CHECKING:
    from rowsynth.gui.app import App

NEEDS = "pygame is not installed"
CTRL = frozenset({"ctrl"})
#: A small project: one instrument, and one song of one bar on one channel.
SMALL = {
    "instruments": {"beep": {"vol": [15, 12, 9, 6]}},
    "songs": {"tune": {"row_ticks": 3, "channels": {"tune": {"inst": "beep"}},
                       "patterns": {"a": {"tune": "l4 c e g > c"}}, "order": ["a"]}},
}


def click(ui: Ui, pos: tuple[int, int]) -> None:
    """Press the left button at a position and let it go there."""
    ui.handle(Event("down", pos, button=1))
    ui.handle(Event("up", pos, button=1))


def press(ui: Ui, key: str, mods: frozenset[str] = frozenset()) -> None:
    """Press a key that types nothing."""
    ui.handle(Event("key", key=key, mods=mods))


def write(ui: Ui, text: str) -> None:
    """Type characters, as the window tells them."""
    for char in text:
        ui.handle(Event("text", text=char))


def middle(rect: layout.Rect) -> tuple[int, int]:
    """The pixel in the middle of a rectangle."""
    return rect[0] + rect[2] // 2, rect[1] + rect[3] // 2


def spot(rows: List, key: Hashable) -> tuple[int, int]:
    """The top-left corner of the row of a key, as the list is scrolled now."""
    x, y, _, h = rows.rect
    return next((x, down) for down in range(y, y + h) if getattr(rows.at((x, down)), "key", None) == key)


def ink(surface: pygame.Surface, rect: layout.Rect, colour: theme.Colour) -> int:
    """How many pixels of a rectangle are in a colour."""
    x, y, w, h = rect
    return sum(surface.get_at((x + column, y + row))[:3] == colour for column in range(w) for row in range(h))


def settle() -> None:
    """Waits for every render and every run of the checks that is still on a thread of a model's."""
    for thread in threading.enumerate():
        if thread.name.startswith("rowsynth-"):
            thread.join()


class InAFolder(unittest.TestCase):
    """A folder of its own for each test: two sub-folders, two project files, and a file that is no project."""

    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(os.path.abspath(folder.name))
        for name in ("sub", "Beta"):
            (self.folder / name).mkdir()
        for name in ("a.json", "B.JSON", "notes.txt"):
            (self.folder / name).write_text("{}\n", encoding="utf-8")
        (self.folder / "sub" / "deep.json").write_text("{}\n", encoding="utf-8")
        #: What the clock of a chooser says, which a test moves on.
        self.now = 0.0

    def chooser(self, **more: object) -> tuple[Ui, files.Chooser, list[Path]]:
        """A Ui with a chooser open in the folder, and the list the files it takes are put in."""
        ui, taken = Ui(), []
        box = files.Chooser(self.folder, taken.append, clock=lambda: self.now, **more)
        ui.open(box)
        return ui, box, taken


class Folders(InAFolder):
    def test_a_folder_holds_its_sub_folders_and_its_project_files_and_one_that_cannot_be_read_says_so(self) -> None:
        self.assertEqual(files.holds(self.folder), files.Held(("Beta", "sub"), ("a.json", "B.JSON")))
        missing = self.folder / "nowhere"
        held = files.holds(missing)
        self.assertEqual((held.folders, held.files), ((), ()))
        self.assertTrue(held.problem.startswith(f"{missing} cannot be read: "), held.problem)
        # The examples are those of a checkout: a folder that is none has no example, whatever it holds.
        (self.folder / "examples").mkdir()
        (self.folder / "examples" / "tune.json").write_text("{}\n", encoding="utf-8")
        self.assertEqual(files.examples(self.folder), [])
        (self.folder / "pyproject.toml").write_text("", encoding="utf-8")
        self.assertEqual(files.examples(self.folder), [self.folder / "examples" / "tune.json"])

    def test_the_files_opened_last_are_read_and_written_and_are_eight_at_the_most(self) -> None:
        settings = self.folder / "settings" / "rowsynth"
        self.assertEqual(files.read_recent(settings), [])
        names = [str(self.folder / f"{number}.json") for number in range(10)]
        recent: list[str] = []
        for name in names:
            recent = files.remembered(recent, name)
        # The latest is first, and the two opened longest ago have gone.
        self.assertEqual(recent, names[:1:-1])
        files.write_recent(settings, recent)
        self.assertEqual(files.read_recent(settings), recent)
        # A file that is opened again is first, and is there once.
        self.assertEqual(files.remembered(recent, names[5]), [names[5], *recent[:4], *recent[5:]])
        self.assertEqual(files.forgotten(recent, names[9]), recent[1:])
        # A list that is not what was written counts as none.
        (settings / files.RECENT_FILE).write_text("o4 l8 c e g\n", encoding="utf-8")
        self.assertEqual(files.read_recent(settings), [])

    def test_the_settings_folder_is_the_users_own_on_each_system(self) -> None:
        home = Path("home") / "me"
        with mock.patch.object(Path, "home", return_value=home):
            with mock.patch.object(files.sys, "platform", "win32"), \
                    mock.patch.dict(os.environ, {"APPDATA": str(home / "Roaming")}):
                self.assertEqual(files.settings_folder(), home / "Roaming" / "rowsynth")
            with mock.patch.object(files.sys, "platform", "darwin"):
                self.assertEqual(files.settings_folder(), home / "Library" / "Application Support" / "rowsynth")
            with mock.patch.object(files.sys, "platform", "linux"), mock.patch.dict(os.environ) as set_to:
                set_to.pop("XDG_CONFIG_HOME", None)
                self.assertEqual(files.settings_folder(), home / ".config" / "rowsynth")
                set_to["XDG_CONFIG_HOME"] = str(home / "settings")
                self.assertEqual(files.settings_folder(), home / "settings" / "rowsynth")
        # Where it really is, it is not in the checkout the package is in.
        checkout = Path(files.__file__).resolve().parents[2]
        self.assertNotIn(checkout, [files.settings_folder().resolve(), *files.settings_folder().resolve().parents])


class Choosing(InAFolder):
    def test_enter_takes_the_name_that_is_typed_and_escape_closes_with_no_file_taken(self) -> None:
        ui, box, taken = self.chooser()
        self.assertEqual((ui.modal, box.folder, [row.text for row in box.list.rows]),
                         (box, self.folder, ["..", "Beta", "sub", "a.json", "B.JSON"]))
        # A name with no ending finds the file that has one.
        write(ui, "a")
        press(ui, "return")
        self.assertEqual((taken, ui.modal), ([self.folder / "a.json"], None))
        # A name that no file has is refused in the chooser's own line, and the chooser stays open.
        ui, box, taken = self.chooser()
        write(ui, "zz")
        press(ui, "return")
        self.assertEqual((taken, ui.modal, box.words), ([], box, "zz: no such file"))
        closed = []
        box.closed = lambda: closed.append(True)
        press(ui, "escape")
        self.assertEqual((taken, ui.modal, closed), ([], None, [True]))

    def test_a_double_press_goes_into_a_folder_and_takes_a_file_and_the_row_of_two_dots_goes_up(self) -> None:
        ui, box, taken = self.chooser()
        sub = spot(box.list, (files.FOLDER, "sub"))
        # Two presses too far apart are no double press: the folder is picked, and that is all.
        click(ui, sub)
        self.now += theme.DOUBLE_PRESS + 0.1
        click(ui, sub)
        self.assertEqual((box.folder, box.list.picked), (self.folder, (files.FOLDER, "sub")))
        self.now += theme.DOUBLE_PRESS
        click(ui, sub)
        self.assertEqual((box.folder, [row.text for row in box.list.rows]), (self.folder / "sub", ["..", "deep.json"]))
        up = spot(box.list, (files.FOLDER, files.UP))
        click(ui, up)
        click(ui, up)
        self.assertEqual(box.folder, self.folder)
        # One press on a file puts its name in the box, and the second takes it.
        file = spot(box.list, (files.FILE, "B.JSON"))
        click(ui, file)
        self.assertEqual((box.box.text, taken, ui.modal), ("B.JSON", [], box))
        click(ui, file)
        self.assertEqual((taken, ui.modal), ([self.folder / "B.JSON"], None))

    def test_a_name_that_is_there_already_is_refused_for_a_new_file_in_the_choosers_own_line(self) -> None:
        self.assertEqual([files.free_name(self.folder, name) for name in ("a.json", "new.json")],
                         ["a-2.json", "new.json"])
        ui, box, taken = self.chooser(new=True, name="a.json")
        press(ui, "return")
        self.assertEqual((taken, ui.modal, box.words), ([], box, "a.json is there already"))
        # The name without its ending means the same file. Typing takes the place of the name that was offered.
        write(ui, "a")
        self.assertEqual((box.box.text, box.words), ("a", ""))
        press(ui, "return")
        self.assertEqual((taken, ui.modal, box.words), ([], box, "a.json is there already"))
        # The second button takes the name as enter does: a name of its own is given the ending of a project file.
        press(ui, "a", CTRL)
        write(ui, "fresh")
        click(ui, middle(files.arrange(ui.size)["take"]))
        self.assertEqual((taken, ui.modal), ([self.folder / "fresh.json"], None))


class Drives(InAFolder):
    def test_the_drives_are_those_of_windows_and_there_are_none_on_any_other_system(self) -> None:
        with mock.patch.object(files.sys, "platform", "linux"):
            self.assertEqual(files.drives(), [])
        with mock.patch.object(files.sys, "platform", "win32"):
            with mock.patch.object(os, "listdrives", lambda: ["C:\\", "E:\\"], create=True):
                self.assertEqual(files.drives(), ["C:\\", "E:\\"])
            # A Python that cannot list them is asked for each letter whose root is there.
            with mock.patch.object(os, "listdrives", None, create=True), \
                    mock.patch.object(os.path, "exists", lambda root: root in ("C:\\", "Q:\\")):
                self.assertEqual(files.drives(), ["C:\\", "Q:\\"])

    def test_up_from_the_root_of_a_drive_is_the_list_of_the_drives_and_one_gone_into_shows_its_root(self) -> None:
        # Where the system has no drives to list, a root has no row that goes up.
        _, box, _ = self.chooser(drives=lambda: [])
        box.go(self.folder.anchor)
        self.assertNotIn((files.FOLDER, files.UP), [row.key for row in box.list.rows])
        # Two folders of the test stand in for two drives.
        made_up = [str(self.folder / "sub"), str(self.folder / "Beta")]
        ui, box, _ = self.chooser(drives=lambda: made_up)
        box.go(self.folder.anchor)
        self.assertEqual((box.list.rows[0].key, box.at_drives, box.where()),
                         ((files.FOLDER, files.UP), False, self.folder.anchor))
        up = spot(box.list, (files.FOLDER, files.UP))
        click(ui, up)
        click(ui, up)
        # The list of the drives has no row that goes up, and the folder line says what it is.
        self.assertEqual((box.at_drives, box.where(), [row.key for row in box.list.rows], box.words),
                         (True, "drives", [(files.FOLDER, name) for name in made_up], ""))
        drive = spot(box.list, (files.FOLDER, made_up[0]))
        click(ui, drive)
        click(ui, drive)
        self.assertEqual((box.at_drives, box.folder, box.where(), [row.text for row in box.list.rows]),
                         (False, self.folder / "sub", made_up[0], ["..", "deep.json"]))

    def test_in_the_list_of_the_drives_a_name_is_refused_and_a_whole_path_is_gone_to(self) -> None:
        ui, box, taken = self.chooser(drives=lambda: [str(self.folder)])
        box.show_drives()
        write(ui, "a")
        press(ui, "return")
        self.assertEqual((taken, ui.modal, box.words, box.at_drives), ([], box, files.NO_DRIVE, True))
        # A whole path is gone to as anywhere else: a folder is shown, and a file is taken.
        press(ui, "a", CTRL)
        write(ui, str(self.folder / "sub"))
        press(ui, "return")
        self.assertEqual((box.at_drives, box.folder, box.words, box.box.text), (False, self.folder / "sub", "", ""))
        box.show_drives()
        write(ui, str(self.folder / "a.json"))
        press(ui, "return")
        self.assertEqual((taken, ui.modal), ([self.folder / "a.json"], None))


class Gone(unittest.TestCase):
    """Two files opened before, of which the second is gone: what says so is handed in, and no disk is asked."""

    recent = [os.path.join("music", "kept.json"), os.path.join("music", "gone.json")]

    @staticmethod
    def there(name: str) -> bool:
        """Whether a file is there: the one that is called kept is."""
        return "kept" in name

    def test_a_file_opened_before_that_is_gone_is_not_enabled_in_the_menu_and_cannot_be_chosen(self) -> None:
        kept, gone = self.recent
        items = files.menu(self.recent, self.there)
        self.assertEqual([(item.value, item.enabled) for item in items if item is not None and not item.enabled],
                         [(("recent", gone), False)])
        ui, chosen = Ui(), []
        menu = ui.open(Menu(items, (0, 0), chosen.append))
        rows = {item.value: row for item, row in zip(menu.items, menu.rows) if item is not None}
        click(ui, middle(rows["recent", gone]))
        self.assertEqual((chosen, ui.modal), ([], menu))
        click(ui, middle(rows["recent", kept]))
        self.assertEqual((chosen, ui.modal), ([("recent", kept)], None))

    def test_a_press_on_a_dim_row_of_the_first_screen_opens_nothing_and_says_the_file_is_no_longer_there(self) -> None:
        kept, gone = self.recent
        ui, pressed, said = Ui(), [], []
        first = files.First(ui, None, None, pressed.append, (), self.recent, said.append, self.there)
        first.place((0, 0, *theme.DESIGN))
        self.assertEqual([(row.text, row.dim) for row in first.list.rows],
                         [("RECENT", False), ("kept.json", False), ("gone.json", True)])
        click(ui, spot(first.list, ("recent", gone)))
        self.assertEqual((pressed, said, first.list.picked), ([], [f"{gone} is no longer there"], None))
        click(ui, spot(first.list, ("recent", kept)))
        self.assertEqual((pressed, len(said)), ([("recent", kept)], 1))

    def test_the_keys_that_move_the_pick_step_over_a_dim_row_of_the_first_screen(self) -> None:
        kept, gone = self.recent
        last = os.path.join("music", "kept as well.json")
        ui, pressed, said = Ui(), [], []
        first = files.First(ui, None, None, pressed.append, (), [kept, gone, last], said.append, self.there)
        first.place((0, 0, *theme.DESIGN))
        click(ui, spot(first.list, ("recent", kept)))
        press(ui, "down")
        self.assertEqual((pressed, said, first.list.picked),
                         ([("recent", kept), ("recent", last)], [], ("recent", last)))
        press(ui, "up")
        self.assertEqual((pressed[2:], said, first.list.picked), ([("recent", kept)], [], ("recent", kept)))
        # Any other list stops at a dim row, which is a row like the rest there.
        plain = Ui().add(List((0, 0, 100, 100), first.list.rows, ("recent", kept)))
        plain.on(Event("key", key="down"), ui)
        self.assertEqual(plain.picked, ("recent", gone))


@unittest.skipUnless(pygame, NEEDS)
class InTheWindow(unittest.TestCase):
    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(os.path.abspath(folder.name))
        #: The settings folder of the test, which is not there until something is remembered in it.
        self.settings = self.folder / "settings"
        # No thread of one test is left to run into the next, and neither is a window or a sound output.
        self.addCleanup(settle)
        self.addCleanup(pygame.mixer.quit)
        self.addCleanup(pygame.display.quit)
        # The desktop is a full HD one, whatever the machine the tests are run on has.
        desktop = mock.patch("pygame.display.get_desktop_sizes", return_value=[(1920, 1080)])
        desktop.start()
        self.addCleanup(desktop.stop)
        # Every render and the checks wait, and are refused when let go, so that none costs its time.
        go = threading.Event()

        def refused(*args: object) -> None:
            go.wait(60)
            raise RowSynthError("not made in this test")

        for owner, name in ((gui_model, "render_stems"), (gui_model, "render_entry"), (checks, "run")):
            patch = mock.patch.object(owner, name, side_effect=refused)
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(go.set)

    def saved(self, name: str) -> Path:
        """The small project in a file of the test's folder."""
        path = self.folder / name
        path.write_text(json.dumps(SMALL), encoding="utf-8", newline="\n")
        return path

    def first(self) -> App:
        """The app with no file, on its first screen. A chooser of its starts in the test's folder."""
        app = start(settings=self.settings)
        app.started = self.folder
        return app

    def test_with_no_file_the_window_opens_on_the_first_screen_and_its_three_things(self) -> None:
        kept, gone = self.saved("kept.json"), self.folder / "gone.json"
        files.write_recent(self.settings, [str(kept), str(gone)])
        app = self.first()
        first = app.first
        self.assertEqual((app.session, app.model, app.panels), (None, None, {}))
        self.assertEqual((first.new.text, first.open.text), ("NEW PROJECT", "OPEN..."))
        # A row for each example, and the files opened last: the one that is gone is dim.
        self.assertEqual([(row.text, row.head, row.dim) for row in first.list.rows],
                         [("EXAMPLES", True, False), ("arcade", False, False), ("piece", False, False),
                          ("RECENT", True, False), ("kept.json", False, False), ("gone.json", False, True)])
        art = app.screen.begin()
        app.update()
        app.draw(art)
        parts = layout.frame(art.get_size())
        # The program's name is drawn, and of the frame only the status bar: where the top bar is, nothing.
        self.assertGreater(ink(art, files.first_places(first.rect)["name"], theme.TEXT), 0)
        self.assertEqual(ink(art, parts["top"], theme.BACKGROUND), parts["top"][2] * parts["top"][3])
        self.assertEqual(art.get_at(middle(parts["status"]))[:3], theme.PANEL)
        # Each button opens a chooser: one for a file that is not there yet, and one for a file that is.
        click(app.ui, middle(first.new.rect))
        self.assertEqual((type(app.ui.modal), app.ui.modal.new), (files.Chooser, True))
        press(app.ui, "escape")
        click(app.ui, middle(first.open.rect))
        self.assertEqual((type(app.ui.modal), app.ui.modal.new), (files.Chooser, False))

    def test_a_new_project_is_saved_to_its_file_at_once_and_loads_and_is_not_silent(self) -> None:
        app = self.first()
        app.new()
        chooser = app.ui.modal
        self.assertEqual((chooser.folder, chooser.box.text), (self.folder, "project.json"))
        write(app.ui, "first")
        press(app.ui, "return")
        path = self.folder / "first.json"
        self.assertEqual((app.first, app.ui.modal, app.model.path, app.model.picked),
                         (None, None, path, ("song", "song")))
        project = Project.load(path)
        self.assertEqual(project, files.new_project())
        self.assertEqual([len(table) for table in (project.instruments, project.sounds, project.songs)], [1, 1, 1])
        self.assertGreater(float(abs(render_entry(project.sound("blip"), 8000)[0]).max()), 0.1)
        # The file is the one opened last, and that is remembered for the next run.
        self.assertEqual(files.read_recent(self.settings), [str(path)])

    def test_another_file_opened_puts_a_new_session_in_place_of_the_old_and_a_broken_file_leaves_it(self) -> None:
        one, two = self.saved("one.json"), self.saved("two.json")
        broken = self.folder / "broken.json"
        broken.write_text("o4 l8 c e g\n", encoding="utf-8")
        app = start(one, settings=self.settings)
        old = app.session
        self.assertTrue(old.change("songs.tune.patterns.a.tune", "l2 c e", "tune of pattern a: note removed"))
        self.assertTrue(old.can_undo)
        with mock.patch.object(old.player, "close", wraps=old.player.close) as closed:
            self.assertTrue(app.open(two))
        # The old session's player is closed, and the new session has one of its own and no step to take back.
        closed.assert_called_once_with()
        top = app.panels["top"]
        self.assertEqual((app.session is old, app.session.model is app.model, app.model.path, app.session.can_undo),
                         (False, True, two, False))
        self.assertEqual((top.session is app.session, top.name.text, pygame.mixer.get_init() is None),
                         (True, "two.json", False))
        # A file that is no project, and one that is not there, are refused in the status bar: the open one stays.
        session = app.session
        for path, why in ((broken, "line 1, column 1: Expecting value"), (self.folder / "nowhere.json",
                                                                         "no such file")):
            self.assertFalse(app.open(path))
            self.assertEqual((app.session is session, app.model.path, app.session.message),
                             (True, two, f"{path}: {why}"))
        self.assertEqual(app.recent, [str(two), str(one)])
        # Close goes back to the first screen, with no session and no sound output.
        app.do(("close",))
        self.assertEqual((app.first is None, app.session, app.panels, pygame.mixer.get_init()),
                         (False, None, {}, None))

    def test_an_example_is_opened_as_a_copy_of_ones_own_and_the_bytes_of_the_example_are_untouched(self) -> None:
        example = files.examples()[0]
        before = example.read_bytes()
        app = self.first()
        click(app.ui, spot(app.first.list, ("example", str(example))))
        chooser = app.ui.modal
        self.assertEqual((chooser.new, chooser.folder, chooser.box.text), (True, self.folder, "arcade.json"))
        press(app.ui, "return")
        copy = self.folder / "arcade.json"
        self.assertEqual((app.model.path, copy.read_bytes()), (copy, before))
        # What is changed in the window is changed in the copy.
        self.assertTrue(app.session.change("songs.tune.patterns.a.lead", "l1 c", "lead of pattern a"))
        self.assertNotEqual(copy.read_bytes(), before)
        self.assertEqual(example.read_bytes(), before)
        # A second copy is offered a name that the first does not have.
        app.leave()
        click(app.ui, spot(app.first.list, ("example", str(example))))
        self.assertEqual(app.ui.modal.box.text, "arcade-2.json")

    def test_a_copy_is_saved_to_another_file_and_the_open_file_stays_open(self) -> None:
        one = self.saved("one.json")
        app = start(one, settings=self.settings)
        app.do(("copy",))
        chooser = app.ui.modal
        self.assertEqual((chooser.new, chooser.folder, chooser.box.text), (True, self.folder, "one-2.json"))
        press(app.ui, "return")
        copy = self.folder / "one-2.json"
        self.assertEqual((Project.load(copy), app.model.path, app.session.message, app.recent),
                         (app.model.project, one, f"a copy is kept in {copy}", [str(one)]))

    def test_a_dim_file_opened_before_only_says_it_is_gone_and_one_that_goes_later_is_dropped(self) -> None:
        kept, late, gone = self.saved("kept.json"), self.saved("late.json"), self.folder / "gone.json"
        files.write_recent(self.settings, [str(gone), str(late), str(kept)])
        app = self.first()
        # The file that was gone as the list was made has a dim row. A press on it opens nothing and drops nothing.
        click(app.ui, spot(app.first.list, ("recent", str(gone))))
        self.assertEqual((app.session, app.ui.modal, len(app.recent), app.first.list.picked), (None, None, 3, None))
        # The status bar says that it is no longer there.
        art = app.screen.begin()
        app.update()
        app.draw(art)
        self.assertGreater(ink(art, layout.frame(art.get_size())["status"], theme.TEXT), 0)
        # One that goes between the listing and the press is found gone as it is opened, and is off the list.
        late.unlink()
        click(app.ui, spot(app.first.list, ("recent", str(late))))
        left = [str(gone), str(kept)]
        self.assertEqual((app.session, app.recent, files.read_recent(self.settings)), (None, left, left))
        self.assertEqual([(row.text, row.dim) for row in app.first.list.rows if not row.head][-2:],
                         [("gone.json", True), ("kept.json", False)])
        click(app.ui, spot(app.first.list, ("recent", str(kept))))
        self.assertEqual((app.first, app.model.path), (None, kept))

    def test_the_gui_command_with_no_file_runs_for_three_frames_and_returns_0(self) -> None:
        shot = self.folder / "first.png"
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed), \
                mock.patch.object(files, "settings_folder", return_value=self.settings):
            self.assertEqual(cli.main(["gui", "--frames", "3", "--shot", str(shot)]), 0)
            # With a file as well. A run for a count of frames remembers nothing in the settings folder.
            self.assertEqual(cli.main(["gui", str(self.saved("one.json")), "--frames", "2"]), 0)
        self.assertEqual(printed.getvalue().count('"frames"'), 2)
        self.assertEqual(json.loads(printed.getvalue().split("\n{")[0]),
                         {"file": None, "frames": 3, "shot": str(shot)})
        self.assertEqual((shot.read_bytes()[:8], self.settings.exists()), (b"\x89PNG\r\n\x1a\n", False))

    def test_the_file_button_opens_the_menu_under_it_and_each_thing_of_the_menu_reaches_its_action(self) -> None:
        one, two = self.saved("one.json"), self.saved("two.json")
        files.write_recent(self.settings, [str(two)])
        app = start(one, settings=self.settings)
        asked = mock.Mock()
        app.actions = {word: getattr(asked, word) for word in app.actions}
        x, y, _, h = app.panels["top"].file.rect
        click(app.ui, middle(app.panels["top"].file.rect))
        menu = app.ui.modal
        self.assertEqual((type(menu), menu.rect[:2]), (Menu, (x, y + h)))
        here = self.folder.name + os.sep
        self.assertEqual([item and item.text for item in menu.items],
                         ["New project...", "Open...", "Save a copy as...", "Export...", None,
                          here + "one.json", here + "two.json", None, "Close"])
        for item in menu.items:
            if item is not None:
                menu.chosen(item.value)
        self.assertEqual(asked.mock_calls, [mock.call.new(), mock.call.open(), mock.call.copy(),
                                            mock.call.export(), mock.call.recent(str(one)),
                                            mock.call.recent(str(two)), mock.call.close()])

    def test_export_opens_the_window_that_writes_files_over_the_open_project_and_does_nothing_with_none(self) -> None:
        app = start(self.saved("one.json"), settings=self.settings)
        app.do(("export",))
        self.assertIsInstance(app.ui.modal, Export)
        app.ui.close(app.ui.modal)
        app.leave()
        app.do(("export",))
        self.assertNotIsInstance(app.ui.modal, Export)


if __name__ == "__main__":
    unittest.main()
