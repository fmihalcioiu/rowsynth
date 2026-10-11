"""The main loop: a window on a project file or on the first screen, drawn sixty times a second, and its files."""
from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path

import pygame

from ..errors import RowSynthError
from ..project import Project
from . import files, layout, sprites, theme
from .audio import Player
from .browser import Browser
from .canvas import Canvas
from .exports import open_export
from .font import Font
from .inspector import Inspector
from .layout import Rect
from .middle import Middle
from .model import Model
from .panels import Panel
from .popups import Menu
from .screen import Screen
from .session import Session
from .theme import Colour
from .topbar import TopBar
from .ui import Pos, Ui, translate

FPS = 60
# The seconds a message that was just said has the status bar, whatever the mouse is on.
MESSAGE_TIME = 3.0

# The side of each part of the frame that its line runs along: the one that faces the middle.
_LINES = {"top": "bottom", "left": "right", "right": "left", "status": "top"}

# The key that plays what is picked, and stops it.
_PLAY = ("space", frozenset())
# The key that takes the last step back, and the two that make it again.
_UNDO = ("z", frozenset({"ctrl"}))
_REDO = (("y", frozenset({"ctrl"})), ("z", frozenset({"ctrl", "shift"})))

# What stands at the top of a chooser, by what it was opened for.
_OPEN = "OPEN A PROJECT"
_NEW = "A FILE FOR THE NEW PROJECT"
_COPY = "A FILE FOR THE COPY"
_EXAMPLE = "A FILE FOR YOUR COPY OF THE EXAMPLE"
# What the file of a new project is called until it is given another name.
_NEW_NAME = "project" + files.ENDING


def picked_words(picked: tuple[str, str] | None) -> str:
    """What is picked in a few words: its kind in capitals and then its name, and nothing for nothing picked."""
    return "" if picked is None else f"{picked[0].upper()} {picked[1]}"


def status_line(help: str, message: str, age: float) -> tuple[str, Colour]:
    """The one line the status bar shows, and the colour it is in.

    `help` is the help line of the widget under the mouse, "" with no such line. `message` is what the session
    has to say, and `age` the seconds since it was said. A message has the bar for its first `MESSAGE_TIME`
    seconds, whatever the mouse is on. After that the help line has the bar, and the message only where there
    is no help line. A message is in the colour of text, and a help line is dim.
    """
    if message and (age < MESSAGE_TIME or not help):
        return message, theme.TEXT
    return help, theme.DIM


def _model(path: str | Path) -> Model:
    """The model of the project in a file. A file that is not there, or holds no project, is refused."""
    try:
        return Model(path)
    except FileNotFoundError:
        raise RowSynthError(f"{path}: no such file") from None
    except RowSynthError:
        # The model names the file as a path spells it. The refusal names it as it was given.
        Project.load(path)
        raise


def _why(error: OSError) -> str:
    """What kept a file from being written, in the system's words."""
    return error.strerror or str(error)


class App:
    """The window, the project of a file as a session in it, and the loop that keeps the window drawn.

    `panels` is from the name of a part of the frame to the panel that fills it. The parts with no panel are
    drawn empty. With no project open there is no session, no model and no panel: `first` is then the first
    screen, which stands in their place, and it is None while a project is open.

    The window works on one file at a time, and another can be opened in it: `open` puts a new session in the
    place of the one there was. `actions` is from a word of the FILE menu to what does it. `recent` is the
    files opened last, the latest first, and `started` the folder the program was started in.
    """

    def __init__(self, path: str | Path | None = None, scale: int | None = None,
                 settings: str | Path | None = None) -> None:
        """Read the project in the file, then open the window. A file that holds no project is refused first.

        With no file the window opens on the first screen. `settings` is the folder the files opened last are
        remembered in from one run to the next: with None they are remembered for this run only.

        The sound output is opened at the rate the project is rendered at. Where it cannot be opened the session
        has none: everything else works, and the session says so from the first frame.
        """
        model = None if path is None else _model(path)
        self.path = path
        self.settings = settings
        self.recent: list[str] = [] if settings is None else files.read_recent(settings)
        self.started = Path.cwd()
        self.screen = Screen(scale=scale)
        # What is typed comes as text of its own, apart from the keys. Classic pygame and pygame-ce both send it
        # from the start, and both can be asked to: that changes nothing where it is on, and makes sure of it.
        pygame.key.start_text_input()
        self.closed = False
        self.canvas = Canvas(self.screen.art, Font(sprites.load("font")), sprites.load("icons"))
        self.actions: dict[str, Callable[..., object]] = {
            "new": self.new, "open": self.browse, "copy": self.keep_copy, "recent": self.again,
            "example": self.example, "export": self.export, "close": self.leave}
        self.model: Model | None = None
        self.session: Session | None = None
        self.first: files.First | None = None
        self.panels: dict[str, Panel] = {}
        self.ui = Ui()
        # What the app has to say while no session is open to say it, and how many times it had something.
        self._own = ""
        self._owns = 0
        # How many things there were to say by the last frame, and the seconds since the last of them was said.
        self._said = 0
        self._age = MESSAGE_TIME
        self._size = self.screen.art.get_size()
        if model is None:
            self.leave()
        else:
            self._enter(model, path)

    def _fresh(self) -> None:
        """Start the widgets anew, with nothing said: a Ui that has none of them, on the surface as it is."""
        self.ui = Ui()
        self.ui.size = self._size
        self.panels = {}
        self.first = None
        self._own, self._owns = "", 0
        self._said, self._age = 0, MESSAGE_TIME

    def _shut(self) -> None:
        """Close the session that is open: its sound stops and its player is closed. With none, nothing."""
        if self.session is not None:
            self.session.close()

    def _enter(self, model: Model, path: str | Path) -> None:
        """Put a model in the place of what is open: a player and a session of its own, and the panels anew.

        The session there was is closed first. The new one has no step to take back, and the file is the first
        of those opened last.
        """
        self._shut()
        try:
            player = Player(model.rate)
        except RowSynthError:
            player = None
        self._fresh()
        self.model, self.path = model, path
        self.session = Session(model, player)
        self._said = self.session.messages
        self.ui.keys[_PLAY] = self.session.toggle
        self.ui.keys[_UNDO] = self.session.undo
        for keys in _REDO:
            self.ui.keys[keys] = self.session.redo
        self.panels = {"top": TopBar(self.session, self.ui, self.file_menu),
                       "left": Browser(self.session, self.ui),
                       "middle": Middle(self.session, self.ui),
                       "right": Inspector(self.session, self.ui)}
        self.place(self._size)
        self._keep(files.remembered(self.recent, path))

    def leave(self) -> None:
        """Close the project that is open, and show the first screen in place of the panels."""
        self._shut()
        self._fresh()
        self.model = self.session = self.path = None
        self.first = files.First(self.ui, self.new, self.browse, self.do, files.examples(), self.recent, self.say)
        self.place(self._size)

    def open(self, path: str | Path) -> bool:
        """Open the project in a file in the place of what is open, and say whether it was opened.

        A file that is not there, or holds no project, is refused in the status bar, and what was open stays.
        """
        try:
            model = _model(path)
        except (RowSynthError, OSError) as error:
            self.say(str(error))
            return False
        self._enter(model, path)
        return True

    def again(self, name: str) -> None:
        """Open a file of those opened last. One that is gone is taken off the list, and the status bar says so."""
        if os.path.isfile(name):
            self.open(name)
            return
        self._keep(files.forgotten(self.recent, name))
        self.say(f"{name} is gone, and is off the list")
        if self.first is not None:
            self.first.show(files.examples(), self.recent)

    def new(self) -> None:
        """Ask for a file that is not there yet, make the smallest project in it, and open that."""
        folder = self._folder()
        self.ui.open(files.Chooser(folder, self._make, True, _NEW, files.free_name(folder, _NEW_NAME),
                                   files.CREATE))

    def browse(self) -> None:
        """Ask for a project file, and open it."""
        self.ui.open(files.Chooser(self._folder(), self.open, title=_OPEN))

    def keep_copy(self) -> None:
        """Ask for a file that is not there yet, and write the open project to it. The open file stays open."""
        if self.model is not None:
            folder = self._folder()
            self.ui.open(files.Chooser(folder, self._copy, True, _COPY,
                                       files.free_name(folder, self.model.path.name), files.SAVE))

    def example(self, path: str | Path) -> None:
        """Ask where to keep a copy of an example, and open the copy: the example itself is never opened.

        The chooser starts in the folder the program was started in, with a name that nothing there has.
        """
        self.ui.open(files.Chooser(self.started, lambda to: self._take_example(Path(path), to), True, _EXAMPLE,
                                   files.free_name(self.started, Path(path).name), files.CREATE))

    def file_menu(self, at: Pos) -> None:
        """Open the FILE menu at a place: what is done with files, the files opened last among it."""
        self.ui.open(Menu(files.menu(self.recent), at, self.do))

    def do(self, what: object) -> None:
        """Do a thing of the FILE menu or of the first screen: `what` is its word, and what goes with the word."""
        word, *more = what
        self.actions[word](*more)

    def export(self) -> None:
        """Open the window that writes files, for the project that is open. With none open there is nothing to write."""
        if self.session is not None:
            open_export(self.ui, self.session)

    def say(self, words: str) -> None:
        """Put words in the status bar: through the session where a project is open, and as the app's own if not."""
        if self.session is not None:
            self.session.say(words)
        else:
            self._own = words
            self._owns += bool(words)

    def _folder(self) -> Path:
        """The folder a chooser starts in: that of the open file, or else that of the file opened last.

        With neither, or with that folder gone, it is the folder the program was started in.
        """
        if self.model is not None:
            return Path(os.path.abspath(self.model.path)).parent
        folders = [Path(name).parent for name in self.recent[:1]]
        return next((folder for folder in folders if folder.is_dir()), self.started)

    def _make(self, path: Path) -> None:
        """Save a new project to a file, and open it. A file that cannot be written is said in the status bar."""
        try:
            files.new_project().save(path)
        except OSError as error:
            self.say(f"{path} cannot be written: {_why(error)}")
            return
        self.open(path)

    def _take_example(self, example: Path, to: Path) -> None:
        """Copy an example to a file, and open the copy. A copy that cannot be made is said in the status bar."""
        try:
            to.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(example, to)
        except OSError as error:
            self.say(f"{to} cannot be written: {_why(error)}")
            return
        self.open(to)

    def _copy(self, path: Path) -> None:
        """Write the open project to another file, and say that it is there, or why it is not."""
        try:
            self.model.project.save(path)
        except OSError as error:
            self.say(f"{path} cannot be written: {_why(error)}")
            return
        self.say(f"a copy is kept in {path}")

    def _keep(self, recent: list[str]) -> None:
        """Take the files opened last as they are now, and write them to the settings folder where there is one.

        A folder that cannot be written is no fault: the list is then kept for this run only.
        """
        self.recent = recent
        if self.settings is not None:
            try:
                files.write_recent(self.settings, recent)
            except OSError:
                pass

    def _words(self) -> tuple[str, int]:
        """What there is to say, and how many times there was something: the session's, or with none the app's.

        The session's is its status line: its message, or a notice of what another program did while that has
        its time.
        """
        if self.session is not None:
            return self.session.status, self.session.messages
        return self._own, self._owns

    def place(self, size: tuple[int, int]) -> None:
        """Give each panel its part of the frame on an art surface of a size.

        With no project open, the first screen gets all of the surface but the status bar.
        """
        self._size = size
        parts = layout.frame(size)
        for name, panel in self.panels.items():
            panel.place(parts[name])
        if self.first is not None:
            self.first.place((0, 0, size[0], parts["status"][1]))

    def handle(self, event: pygame.event.Event) -> None:
        """Take in one thing that happened to the window. What the mouse and the keys did, and what was typed,
        goes on to the widgets.

        The window's size is followed by the screen, and needs nothing here.
        """
        if event.type == pygame.QUIT:
            self.closed = True
            return
        happened = translate(event, self.screen.to_art)
        if happened is not None:
            self.ui.handle(happened)

    def update(self, passed: float = 0.0) -> None:
        """A frame's share of what is not drawing: the session's waiting work, then what each panel shows.

        `passed` is the seconds since the frame before, which each panel is told. They are also what the age of
        the message is counted in: it is 0 in the frame that finds something said since the frame before, and
        older by `passed` in each frame after that.
        """
        if self.session is not None:
            self.session.tick()
        for panel in self.panels.values():
            panel.update(passed)
        _, said = self._words()
        self._age = self._age + passed if said == self._said else 0.0
        self._said = said

    def draw(self, art: pygame.Surface) -> None:
        """Draw one frame on the art surface: the parts of the frame as flat panels, each with its line.

        The status bar is drawn on its part, then what each panel draws, and over it all the widgets. On a
        surface of another size than the last the panels are placed anew first. With no project open only the
        status bar is drawn of the frame, and the first screen above it.
        """
        canvas = self.canvas
        canvas.surface = art
        size = art.get_size()
        if size != self._size:
            self.place(size)
        canvas.box((0, 0, *size), theme.BACKGROUND)
        parts = layout.frame(size)
        for name, side in _LINES.items():
            if self.first is None or name == "status":
                canvas.box(parts[name], theme.PANEL)
                canvas.box(layout.edge(parts[name], side), theme.LINE)
        self._status(parts["status"])
        if self.first is not None:
            self.first.draw(canvas)
        for panel in self.panels.values():
            if panel.shown:
                panel.draw(canvas)
        self.ui.draw(canvas)

    def _status(self, rect: Rect) -> None:
        """Draw what the status bar says: at its right end what is picked, and one line in the room before that.

        The line is the one `status_line` picks of the help of the widget the mouse is on and the message
        there is. What was said since the last frame was counted is new.
        """
        x, y, w, h = rect
        # Under the bar's line, and in from its two ends.
        left, top, room, high = x + theme.BAR_MARGIN, y + 1, max(0, w - 2 * theme.BAR_MARGIN), h - 1
        picked = picked_words(None if self.model is None else self.model.picked)
        self.canvas.text_in(picked, (left, top, room, high), theme.DIM, "right")
        if picked:
            room = max(0, room - Font.width(picked) - theme.GROUP_GAP)
        message, said = self._words()
        line, colour = status_line(self.ui.help, message, self._age if said == self._said else 0.0)
        self.canvas.text_in(line, (left, top, room, high), colour)

    def run(self, frames: int | None = None, shot: str | Path | None = None) -> int:
        """Keep the window drawn until it is closed, then close it. Return how many frames were drawn.

        With `frames` it stops by itself after that many. With `shot` the art surface as it was last drawn is
        written to that path as a PNG, in a folder that is made when it is missing.
        """
        clock = pygame.time.Clock()
        drawn = 0
        # The seconds the frame before took, its wait for the next one counted.
        passed = 0.0
        try:
            while frames is None or drawn < frames:
                for event in pygame.event.get():
                    self.handle(event)
                if self.closed:
                    break
                self.update(passed)
                self.draw(self.screen.begin())
                self.screen.present()
                drawn += 1
                passed = clock.tick(FPS) / 1000
            if shot is not None:
                Path(shot).parent.mkdir(parents=True, exist_ok=True)
                # Given a file and not a path, pygame goes by the hint for the format and not by the path's ending.
                with open(shot, "wb") as file:
                    pygame.image.save(self.screen.snapshot(), file, ".png")
        finally:
            self.close()
        return drawn

    def close(self) -> None:
        """Close the session and then the window.

        A render or a run of the checks that is still on its worker is not waited for: its thread does not keep
        the process from ending.
        """
        try:
            self._shut()
        finally:
            self.screen.close()
