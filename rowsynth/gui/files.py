"""Files, from the window: what a folder holds, the files opened last, a new project, the chooser, the first screen."""
from __future__ import annotations

import json
import os
import string
import sys
import time
from collections.abc import Callable, Hashable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from ..project import Project
from . import theme
from .font import ADVANCE, CUT, GLYPH_H, GLYPH_W, Font
from .kit import told
from .layout import Rect
from .lists import List, Row
from .panels import within
from .popups import ENTER, Item, TextBox
from .ui import NOWHERE, Button, Event, Pos, Ui, Widget, draw_button, inside

if TYPE_CHECKING:
    from .canvas import Canvas

#: How the name of a project file ends, and how that of a file of sound does.
ENDING = ".json"
WAV = ".wav"
#: The row of a chooser that goes up, to the folder the one shown is in.
UP = ".."
#: What a row of a chooser is: the first part of its key. The second is its name.
FOLDER, FILE = "folder", "file"
#: How many of the files opened last are remembered, and the file of the settings folder they are kept in.
MOST_RECENT = 8
RECENT_FILE = "recent.json"
#: The most characters a name typed into a chooser has: room for a whole path.
MOST = 200

#: The program's name, as the first screen has it, and the words of its buttons and its two headings.
NAME = "rowsynth"
NEW_PROJECT = "NEW PROJECT"
OPEN_FILE = "OPEN..."
EXAMPLES = "EXAMPLES"
RECENT = "RECENT"
#: The words on the buttons of a chooser: the one that closes it, and the one that takes the name, by what for.
CANCEL = "CANCEL"
OPEN = "OPEN"
CREATE = "CREATE"
SAVE = "SAVE"
#: What a chooser says of a name that is no name.
NO_NAME = "type a name, or press a file"
#: What the folder line of a chooser reads while its list is of the drives, and what it says of a name typed there.
DRIVES = "drives"
NO_DRIVE = "go into a drive first, or type a whole path"
#: The line of help of a chooser, and what its parts say of themselves with the mouse on one: the row that goes
#: up, a folder, a file where one is opened and where a new one is named, the box of the name and the two buttons.
ROWS = "go through the rows"
CHOOSER_HELP = "a file is taken by two presses on it or by its name and enter, and escape closes"
UP_HELP = told("pick the way up, twice to go up to the folder this one is in", wheel=ROWS)
FOLDER_HELP = told("pick this folder, twice to go into it", wheel=ROWS)
FILE_HELP = told("put this name in the box, twice to take the file", wheel=ROWS)
OLD_FILE_HELP = told("put this name in the box: the new file needs a name that no file here has yet", wheel=ROWS)
PART_HELP = {
    "name": told("put the caret here, to type the name of the file or a whole path"),
    "cancel": told("close with no file taken, as escape does"),
    "take": told("take the name in the box, as enter does: with no name there, go into the picked folder"),
}

#: The things of the FILE menu that stand above the files opened last, each as its text and what the app is
#: asked for. A thing that is added here needs the app to know it by that word.
MENU = (("New project...", "new"), ("Open...", "open"), ("Save a copy as...", "copy"), ("Export...", "export"),
        ("Measure a WAV file...", "measure"))
#: The last thing of the FILE menu, under the files opened last.
CLOSE = ("Close", "close")

#: The smallest project that can be heard: one instrument, one sound of one note, and one song of one empty bar
#: on one channel.
SMALLEST = {
    "instruments": {"tone": {"wave": "pulse", "vol": [15, 13, 11, 9, 7, 5, 3, 1]}},
    "sounds": {"blip": {"def": {"voices": [{"inst": "tone", "notes": [[0, "c5", 8]]}]}}},
    "songs": {"song": {"row_ticks": 6, "channels": {"lead": {"inst": "tone"}},
                       "patterns": {"a": {"lead": "r1"}}, "order": ["a"]}},
}


class Held(NamedTuple):
    """What a folder holds: the names of its sub-folders and of its project files, each list in order.

    `problem` is what kept the folder from being read, in words, and "" for a folder that was read.
    """

    folders: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    problem: str = ""


def holds(folder: str | Path, ending: str = ENDING) -> Held:
    """What a folder holds that a chooser shows: its sub-folders, and its files whose names end in `ending`.

    That is the ending of a project file unless another is given. Both are in the order of the alphabet,
    capitals and small letters together. A folder that is not there, or cannot be read, holds nothing and has
    the reason as its `problem`: nothing is raised.
    """
    try:
        with os.scandir(folder) as found:
            entries = [(entry.name, entry.is_dir()) for entry in found]
    except OSError as error:
        return Held(problem=f"{folder} cannot be read: {error.strerror or error}")

    def ordered(names: list[str]) -> tuple[str, ...]:
        return tuple(sorted(names, key=lambda name: (name.lower(), name)))

    return Held(ordered([name for name, is_folder in entries if is_folder]),
                ordered([name for name, is_folder in entries if not is_folder and name.lower().endswith(ending)]))


def rows_of(held: Held, up: bool) -> list[Row]:
    """What a folder holds as the rows of a chooser's list: the sub-folders and then the files.

    Where `up` is set, the first row is the one that goes up. A row is known by what it is and its name.
    """
    folders = ([UP] if up else []) + list(held.folders)
    return ([Row((FOLDER, name), name, square=theme.FOLDER) for name in folders]
            + [Row((FILE, name), name, square=theme.PROJECT_FILE) for name in held.files])


def chosen(folder: str | Path, typed: str, new: bool, ending: str = ENDING) -> tuple[Path | None, str]:
    """The file a typed name means in a folder, or why the name is refused: one of the two is given.

    A name with a path in it is counted from the folder, and one that starts at a root is taken as it is. For
    a `new` file the name gets `ending` where it has none, which is the ending of a project file unless another
    is given, and a file that is there already is refused. Otherwise the file has to be there: a name with no
    ending also finds the file that has one.
    """
    path = Path(os.path.abspath(Path(folder) / typed))
    ended = path if path.suffix.lower() == ending else path.with_name(path.name + ending)
    if new:
        if os.path.lexists(ended):
            return None, f"{ended.name} is there already"
        return ended, ""
    for found in (path, ended):
        if found.is_file():
            return found, ""
    return None, f"{path.name}: no such file"


def free_name(folder: str | Path, name: str) -> str:
    """A name for a new file in a folder, as like `name` as can be.

    It is `name`, or `name` with -2, -3 and so on before its ending: the first of them that nothing in the
    folder has.
    """
    stem, ending = os.path.splitext(name)
    free, number = name, 1
    while os.path.lexists(Path(folder) / free):
        number += 1
        free = f"{stem}-{number}{ending}"
    return free


def drives() -> list[str]:
    """The drives of a Windows system, each as its root, as in "C:\\". None on any other system.

    They are what `os.listdrives` gives where Python has it, and if not the letters A to Z whose root is there.
    """
    if sys.platform != "win32":
        return []
    listed = getattr(os, "listdrives", None)
    if listed is not None:
        return list(listed())
    return [root for root in (f"{letter}:\\" for letter in string.ascii_uppercase) if os.path.exists(root)]


def tail(text: str, width: int) -> str:
    """`text` as it is shown in `width` pixels with its end kept: whole when it fits, and if not starting in '..'."""
    room = max(0, (width + 1) // ADVANCE)
    if len(text) <= room:
        return text
    return (CUT + text[len(text) - max(0, room - len(CUT)):])[:room]


def settings_folder() -> Path:
    """The folder the program keeps its own settings in, which is the user's and is never in the package.

    On Windows it is `rowsynth` in the roaming application data (`%APPDATA%`). On macOS it is `rowsynth` in
    `~/Library/Application Support`. Anywhere else it is `rowsynth` in `$XDG_CONFIG_HOME`, or in `~/.config`
    where that is not set. The folder is not made here.
    """
    home = Path.home()
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA") or home / "AppData" / "Roaming") / NAME
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / NAME
    return Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config") / NAME


def _same(path: str | Path) -> str:
    """A path as it is compared with another: whole, and in one case where the system does not tell them apart."""
    return os.path.normcase(os.path.abspath(path))


def remembered(files: Sequence[str], path: str | Path) -> list[str]:
    """The files opened last with one more opened: that one first, as a whole path, and once only.

    The list holds `MOST_RECENT` files at the most, and the one opened longest ago is the one that goes.
    """
    return [os.path.abspath(path), *forgotten(files, path)][:MOST_RECENT]


def forgotten(files: Sequence[str], path: str | Path) -> list[str]:
    """The files opened last with one of them taken off the list."""
    return [name for name in files if _same(name) != _same(path)]


def read_recent(settings: str | Path) -> list[str]:
    """The files opened last as a settings folder has them, the latest first. None where it has no such list.

    A list that cannot be read, or is not what `write_recent` writes, counts as none.
    """
    try:
        data = json.loads((Path(settings) / RECENT_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    found = data.get("recent") if isinstance(data, dict) else None
    return [name for name in found if isinstance(name, str)][:MOST_RECENT] if isinstance(found, list) else []


def write_recent(settings: str | Path, files: Sequence[str]) -> None:
    """Write the files opened last to a settings folder, which is made where it is missing."""
    folder = Path(settings)
    folder.mkdir(parents=True, exist_ok=True)
    text = json.dumps({"recent": list(files)[:MOST_RECENT]}, indent=2, ensure_ascii=False) + "\n"
    (folder / RECENT_FILE).write_text(text, encoding="utf-8", newline="\n")


def new_project() -> Project:
    """A new project: the smallest the package accepts that can be heard."""
    return Project.from_dict(SMALLEST)


def examples(root: str | Path | None = None) -> list[Path]:
    """The example projects the package can find: the files of the `examples` folder beside it, in a checkout.

    `root` is the folder the package is in, and with None it is found. Where that is no checkout, as when the
    package is installed, or has no such folder, there are none.
    """
    root = Path(__file__).resolve().parents[2] if root is None else Path(root)
    if not (root / "pyproject.toml").is_file():
        return []
    return [root / "examples" / name for name in holds(root / "examples").files]


def brief(path: str | Path) -> str:
    """A file in a few words: the folder it is in, and its name."""
    path = Path(path)
    return f"{path.parent.name}{os.sep}{path.name}" if path.parent.name else path.name


def gone(path: str | Path) -> str:
    """What is said of a file opened before that is pressed while it is listed as gone."""
    return f"{path} is no longer there"


def menu(recent: Sequence[str], there: Callable[[str], bool] = os.path.isfile) -> list[Item | None]:
    """The things of the FILE menu: those of `MENU`, a line, the files opened last, a line, and `CLOSE`.

    The value of an item is what the app is asked for: a word, and after it what goes with the word. A file
    opened before is ("recent", path), and one that `there` says is gone is not enabled: it is dim, and cannot
    be chosen. With no such files there is one line and not two.
    """
    items: list[Item | None] = [Item(text, (what,)) for text, what in MENU]
    if recent:
        items += [None, *(Item(brief(name), ("recent", name), enabled=there(name)) for name in recent)]
    return [*items, None, Item(CLOSE[0], (CLOSE[1],))]


def first_rows(found: Sequence[str | Path], recent: Sequence[str],
               there: Callable[[str], bool] = os.path.isfile) -> list[Row]:
    """The rows of the first screen's list: the examples under their heading, and the files opened last under theirs.

    An example is known by ("example", path) and reads as its name without the ending. A file opened before is
    known by ("recent", path), reads as its name, and has the folder it is in at the right. One that `there`
    says is gone is dim. A heading with nothing under it is left out.
    """
    rows = [Row("examples", EXAMPLES, head=True, colour=theme.SONG)] if found else []
    rows += [Row(("example", str(path)), Path(path).stem) for path in found]
    rows += [Row("recent", RECENT, head=True, colour=theme.INSTRUMENT)] if recent else []
    rows += [Row(("recent", name), Path(name).name, tag=Path(name).parent.name, dim=not there(name))
             for name in recent]
    return rows


def arrange(size: tuple[int, int]) -> dict[str, Rect]:
    """Where the parts of a chooser go on an art surface of a size, each by what it is called.

    "box" is the whole of it, in the middle of the surface and never larger than it. From the top stand
    "title", "folder", "list" and "words", each the whole width, and at the foot "name", "cancel" and "take"
    side by side. The list has the height the others leave. Every part is cut to the box.
    """
    wide, high = min(theme.CHOOSER[0], size[0]), min(theme.CHOOSER[1], size[1])
    x, y = (size[0] - wide) // 2, (size[1] - high) // 2
    pad, line, gap, button = theme.CHOOSER_PAD, theme.CHOOSER_LINE_H, theme.CHOOSER_GAP, theme.CHOOSER_BUTTON_W
    left, inner = x + pad, wide - 2 * pad
    foot = y + high - pad - theme.CHOOSER_FOOT_H
    words = foot - gap - line
    rows = y + pad + 2 * line + gap
    named = inner - 2 * (button + gap)
    placed = {
        "box": (x, y, wide, high),
        "title": (left, y + pad, inner, line),
        "folder": (left, y + pad + line, inner, line),
        "list": (left, rows, inner, words - gap - rows),
        "words": (left, words, inner, line),
        "name": (left, foot + (theme.CHOOSER_FOOT_H - theme.TEXT_BOX_H) // 2, named, theme.TEXT_BOX_H),
        "cancel": (left + named + gap, foot, button, theme.CHOOSER_FOOT_H),
        "take": (left + named + 2 * gap + button, foot, button, theme.CHOOSER_FOOT_H),
    }
    return {part: within(place, placed["box"]) for part, place in placed.items()}


def _inner(rect: Rect) -> Rect:
    """What of a rectangle lies inside its outline."""
    x, y, w, h = rect
    return x + 1, y + 1, max(0, w - 2), max(0, h - 2)


class Chooser(Widget):
    """A box a file is chosen in, opened as the modal widget: `ui.open(Chooser(folder, taken))`.

    At its top stand `title` and the folder it is in. Under them is `list`, with the sub-folders of that folder
    and its project files: the row `..` is the folder it is in. At its foot is `box`, where a name is typed,
    and two buttons. `words` is the one line it has to say: why a folder could not be read, or a name not taken.
    With an `ending` the files it lists and takes are those whose names end so, in place of project files.

    `drives` gives the drives of the system, and where it gives any the root of a drive has the row `..` too.
    It leads to the list of the drives, each a folder row: `at_drives` is set while that is shown. That list
    has no row `..`, the folder line reads `DRIVES`, and a drive that is gone into shows its root. A name typed
    there is refused in `words`, since it is in no folder, and a whole path is gone to as anywhere else.

    A press on a file puts its name in the box. A double press on a folder goes into it, and on a file takes
    it. Enter takes what is typed, and so does the second button, which reads as `word`. A name that is a
    folder goes into it, and with no name typed the picked folder is gone into. Up and down move the pick.
    Escape closes the chooser, and so does the first button: `closed` is then called.

    With `new` the name is for a file that is not there yet: one that is there is refused in `words`, and the
    chooser stays open. Otherwise the file has to be there. A file that is taken closes the chooser, and
    `taken` is called with its path. `clock` gives the time in seconds, which two presses are told apart by.

    `help` is the line of the part the mouse is on: a row of the list, the box of the name or a button. Anywhere
    else on the chooser it is the chooser's own line.
    """

    takes_text = True

    def __init__(self, folder: str | Path, taken: Callable[[Path], object], new: bool = False, title: str = "",
                 name: str = "", word: str = "", closed: Callable[[], object] | None = None,
                 clock: Callable[[], float] = time.perf_counter,
                 drives: Callable[[], list[str]] = drives, ending: str = ENDING) -> None:
        nowhere = (0, 0, 0, 0)
        super().__init__(nowhere, CHOOSER_HELP)
        self.taken = taken
        self.new = new
        self.ending = ending
        self.title = title
        self.word = word or (SAVE if new else OPEN)
        self.closed = closed
        self.words = ""
        self.list = List(nowhere, changed=self._picked)
        self.box = TextBox(nowhere, name, most=MOST)
        self._clock = clock
        self._drives = drives
        #: Whether the list is of the drives, and not of what `folder` holds.
        self.at_drives = False
        self._mouse: Pos = NOWHERE
        # The button the left press went down on whose release has not come: "cancel", "take" or None.
        self._down: str | None = None
        # The row that was pressed last and the moment of the press, while a second press can make it a double one.
        self._last: tuple[Hashable, float] | None = None
        self._places = arrange(theme.DESIGN)
        self.go(folder)
        self.keep_in(theme.DESIGN)

    @property
    def help(self) -> str:
        """The line of the part the mouse is on: a row of the list, the box of the name or a button.

        Anywhere else it is the chooser's own line.
        """
        row = self.list.at(self._mouse)
        if row is not None:
            if row.key[0] == FILE:
                return OLD_FILE_HELP if self.new else FILE_HELP
            return UP_HELP if row.key[1] == UP and not self.at_drives else FOLDER_HELP
        return next((line for part, line in PART_HELP.items() if inside(self._places[part], self._mouse)),
                    self._help)

    @help.setter
    def help(self, line: str) -> None:
        self._help = line

    def keep_in(self, size: tuple[int, int]) -> None:
        """Take the size of the art surface, and lay the chooser out in the middle of it."""
        self._places = arrange(size)
        self.rect = self._places["box"]
        self.list.rect = _inner(self._places["list"])
        self.box.rect = self._places["name"]

    def go(self, folder: str | Path) -> None:
        """Show another folder. One that cannot be read is shown as holding nothing, and `words` says why.

        The root of a drive has the row that goes up as well, where the system has drives to list.
        """
        self.folder = Path(os.path.abspath(folder))
        self.at_drives = False
        held = holds(self.folder, self.ending)
        self._show(rows_of(held, self.folder.parent != self.folder or bool(self._drives())), held.problem)

    def show_drives(self) -> None:
        """Show the drives in place of a folder, each as a folder row by its root. No row goes up from there."""
        self.at_drives = True
        self._show(rows_of(Held(tuple(self._drives())), False))

    def _show(self, rows: list[Row], words: str = "") -> None:
        """Put other rows in the list, from their top and with none picked, and say what there is to say of them."""
        self.list.rows = rows
        self.list.picked = None
        self.list.scroll.to(0)
        self.words = words
        self._last = None

    def where(self) -> str:
        """What the folder line reads: the folder that is shown, or `DRIVES` while the list is of the drives."""
        return DRIVES if self.at_drives else str(self.folder)

    def enter(self, name: str) -> None:
        """Go into a sub-folder of the folder shown, by its name, or up with the name of the row that goes up.

        Up from the root of a drive is the list of the drives, and from that list a drive is gone into by its root.
        """
        if self.at_drives:
            self.go(name)
        elif name != UP:
            self.go(self.folder / name)
        elif self.folder.parent != self.folder:
            self.go(self.folder.parent)
        else:
            self.show_drives()

    def take(self, ui: Ui) -> None:
        """Take the name that is typed: go into the folder it names, or close with the file it means.

        A name that is refused is said in `words`, and the chooser stays open. With no name typed, the picked
        folder is gone into. While the list is of the drives, a name that is no whole path is refused.
        """
        typed = self.box.text.strip()
        picked = self.list.picked
        if not typed:
            if isinstance(picked, tuple) and picked[0] == FOLDER:
                self.enter(picked[1])
            else:
                self.words = NO_NAME
            return
        if self.at_drives and not Path(typed).is_absolute():
            self.words = NO_DRIVE
            return
        if (self.folder / typed).is_dir():
            self.go(self.folder / typed)
            self._name("")
            return
        path, self.words = chosen(self.folder, typed, self.new, self.ending)
        if path is not None:
            ui.close(self)
            self.taken(path)

    def close(self, ui: Ui) -> None:
        """Close the chooser with no file taken, and tell."""
        ui.close(self)
        if self.closed is not None:
            self.closed()

    def _name(self, text: str) -> None:
        """Put a name in the box, all of it picked, so that typing takes its place."""
        self.box.text = text
        self.box.anchor, self.box.caret = 0, len(text)

    def _picked(self, key: Hashable) -> None:
        """Take a row that was picked in the list: the name of a file goes into the box."""
        if isinstance(key, tuple) and key[0] == FILE:
            self._name(key[1])
            self.words = ""

    def on(self, event: Event, ui: Ui) -> bool:
        """Use everything: what is typed, the keys, and what the mouse does on the list, the box and the buttons."""
        if event.kind == "text":
            self.words = ""
            self.box.on(event, ui)
        elif event.kind == "key":
            self._key(event, ui)
        else:
            self._pointed(event, ui)
        return True

    def _key(self, event: Event, ui: Ui) -> None:
        """Close on escape, take the name on enter, move the pick up or down, and leave the rest to the box."""
        if event.key == "escape":
            self.close(ui)
        elif event.key in ENTER:
            self.take(ui)
        elif event.key in ("up", "down"):
            self.list.on(event, ui)
        else:
            self.box.on(event, ui)

    def _pointed(self, event: Event, ui: Ui) -> None:
        """Hand what the mouse does to the part it is on. A button is pressed on the release that is still on it."""
        self._mouse = event.pos
        over = next((part for part in ("cancel", "take") if inside(self._places[part], event.pos)), None)
        if event.kind == "down":
            self._down = over if event.button == 1 else None
            if inside(self.box.rect, event.pos):
                self.box.on(event, ui)
            elif inside(self.list.rect, event.pos):
                # Asked before the list has the press, which may scroll the row away from under the mouse.
                row = self.list.at(event.pos) if event.button == 1 else None
                self.list.on(event, ui)
                if row is not None:
                    self._pressed(row.key, ui)
            return
        # The wheel scrolls the list wherever the mouse is, and the thumb of its bar follows the mouse off it.
        self.list.on(event, ui)
        if event.kind == "up" and event.button == 1:
            pressed, self._down = self._down, None
            if pressed is not None and pressed == over:
                if pressed == "cancel":
                    self.close(ui)
                else:
                    self.take(ui)

    def _pressed(self, key: Hashable, ui: Ui) -> None:
        """Count a press on a row. The second of two in a row goes into the folder, or takes the file."""
        now = self._clock()
        twice = self._last is not None and self._last[0] == key and now - self._last[1] <= theme.DOUBLE_PRESS
        self._last = None if twice else (key, now)
        if not twice or not isinstance(key, tuple):
            return
        if key[0] == FOLDER:
            self.enter(key[1])
        else:
            self.take(ui)

    def draw(self, canvas: Canvas) -> None:
        """Draw the box, its title, the folder with its end kept, the list in its outline, and the foot.

        The line of words is in the colour of what is wrong.
        """
        places = self._places
        with canvas.clip(self.rect):
            canvas.box(self.rect, theme.RAISED)
            canvas.frame(self.rect, theme.LINE)
            canvas.text_in(self.title, places["title"], theme.TEXT)
            canvas.text_in(tail(self.where(), places["folder"][2]), places["folder"], theme.DIM)
            canvas.frame(places["list"], theme.LINE)
            self.list.hot = inside(self.list.rect, self._mouse)
            self.list.draw(canvas)
            canvas.text_in(self.words, places["words"], theme.FAULT)
            self.box.draw(canvas)
            for part, text in (("cancel", CANCEL), ("take", self.word)):
                hot = inside(places[part], self._mouse)
                draw_button(canvas, places[part], text, hot=hot, pressed=hot and self._down == part)


def first_places(rect: Rect) -> dict[str, Rect]:
    """Where the things of the first screen go in its rectangle: "name", "new", "open" and "list".

    They stand one under the other in the middle, the two buttons side by side. The list is as high as
    `theme.FIRST_LIST_H`, or as what the rectangle leaves it. Every thing is cut to the rectangle.
    """
    x, y, w, h = rect
    gap, tall = theme.FIRST_GAP, theme.FIRST_BUTTON_H
    name_w, name_h = Font.width(NAME) * theme.FIRST_NAME_SCALE, GLYPH_H * theme.FIRST_NAME_SCALE
    wide = min(theme.FIRST_W, w)
    left = x + (w - wide) // 2
    above = name_h + 2 * gap + tall
    high = max(0, min(theme.FIRST_LIST_H, h - above))
    top = y + max(0, (h - above - high) // 2)
    button = (wide - theme.GROUP_GAP) // 2
    placed = {
        "name": (x + (w - name_w) // 2, top, name_w, name_h),
        "new": (left, top + name_h + gap, button, tall),
        "open": (left + wide - button, top + name_h + gap, button, tall),
        "list": (left, top + above, wide, high),
    }
    return {thing: within(place, rect) for thing, place in placed.items()}


class First:
    """The first screen, which stands in place of the panels while no project is open.

    It has the program's name, the button `new` for a new project and `open` for a file, and `list`: the
    examples under one heading and the files opened last under another, the gone ones dim. A press on a row
    calls `pressed` with the key of the row: ("example", path) or ("recent", path). A press on a dim row opens
    nothing: `say` is called with the words of `gone` for its file, and the row is not picked. The keys that
    move the pick step over a dim row.

    `there` says whether a file opened before is still on the disk, as the list is made.
    """

    def __init__(self, ui: Ui, new: Callable[[], object], open: Callable[[], object],
                 pressed: Callable[[Hashable], object], found: Sequence[str | Path] = (),
                 recent: Sequence[str] = (), say: Callable[[str], object] | None = None,
                 there: Callable[[str], bool] = os.path.isfile) -> None:
        nowhere = (0, 0, 0, 0)
        self.ui = ui
        self.pressed = pressed
        self.say = say
        self.there = there
        self.rect: Rect = nowhere
        self.new = ui.add(Button(nowhere, NEW_PROJECT, action=new,
                                 help=told("make a new project in a file of its own: one instrument, one sound, "
                                           "one song")))
        self.open = ui.add(Button(nowhere, OPEN_FILE, action=open, help=told("open a project file")))
        self.list = ui.add(List(nowhere, changed=self._pressed,
                                help=told("open an example as a copy of your own, or a file opened before as it is",
                                          wheel=ROWS)))
        self.list.skips_dim = True
        self._places = first_places(nowhere)
        self.show(found, recent)

    def show(self, found: Sequence[str | Path], recent: Sequence[str]) -> None:
        """Take the examples and the files opened last, and list them anew with none of them picked."""
        self.list.rows = first_rows(found, recent, self.there)
        self.list.picked = None

    def _pressed(self, key: Hashable) -> None:
        """Take a row that was pressed: one of a file that is gone is only said to be, and any other is told on."""
        if any(row.dim for row in self.list.rows if row.key == key):
            self.list.picked = None
            if self.say is not None:
                self.say(gone(key[1]))
        else:
            self.pressed(key)

    def place(self, rect: Rect) -> None:
        """Take the rectangle of the screen, and lay the buttons and the list out in it."""
        self.rect = rect
        self._places = first_places(rect)
        self.new.rect, self.open.rect = self._places["new"], self._places["open"]
        self.list.rect = _inner(self._places["list"])

    def draw(self, canvas: Canvas) -> None:
        """Draw the program's name, each pixel of its letters as a square, and the outline of the list."""
        x, y, _, _ = self._places["name"]
        scale = theme.FIRST_NAME_SCALE
        with canvas.clip(self.rect):
            for index, char in enumerate(NAME):
                glyph = canvas.font.glyph(char, theme.TEXT)
                for row in range(GLYPH_H):
                    for column in range(GLYPH_W):
                        if glyph.get_at((column, row))[3]:
                            canvas.box((x + (index * ADVANCE + column) * scale, y + row * scale, scale, scale),
                                       theme.TEXT)
            canvas.frame(self._places["list"], theme.LINE)
