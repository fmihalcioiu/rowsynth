"""The main loop: a project file open in a window, listed, played, and drawn sixty times a second."""
from __future__ import annotations

from pathlib import Path

import pygame

from ..errors import RowSynthError
from ..project import Project
from . import layout, sprites, theme
from .audio import Player
from .browser import Browser
from .canvas import Canvas
from .font import Font
from .layout import Rect
from .model import Model
from .panels import Panel
from .screen import Screen
from .session import Session
from .topbar import TopBar
from .ui import Ui, translate

FPS = 60

# The side of each part of the frame that its line runs along: the one that faces the middle.
_LINES = {"top": "bottom", "left": "right", "right": "left", "status": "top"}

# The key that plays what is picked, and stops it.
_PLAY = ("space", frozenset())


def picked_words(picked: tuple[str, str] | None) -> str:
    """What is picked in a few words: its kind in capitals and then its name, and nothing for nothing picked."""
    return "" if picked is None else f"{picked[0].upper()} {picked[1]}"


class App:
    """The project of a file as a session, the window it is shown in, and the loop that keeps the window drawn.

    `panels` is from the name of a part of the frame to the panel that fills it. The parts with no panel are
    drawn empty.
    """

    def __init__(self, path: str | Path, scale: int | None = None) -> None:
        """Read the project in the file, then open the window. A file that holds no project is refused first.

        The sound output is opened at the rate the project is rendered at. Where it cannot be opened the session
        has none: everything else works, and the session says so when something is to play.
        """
        try:
            self.model = Model(path)
        except FileNotFoundError:
            raise RowSynthError(f"{path}: no such file") from None
        except RowSynthError:
            # The model names the file as a path spells it. The refusal names it as it was given.
            Project.load(path)
            raise
        self.path = path
        self.screen = Screen(scale=scale)
        self.closed = False
        self.canvas = Canvas(self.screen.art, Font(sprites.load("font")), sprites.load("icons"))
        try:
            player = Player(self.model.rate)
        except RowSynthError:
            player = None
        self.session = Session(self.model, player)
        self.ui = Ui()
        self.ui.keys[_PLAY] = self.session.toggle
        self.panels: dict[str, Panel] = {"top": TopBar(self.session, self.ui),
                                         "left": Browser(self.session, self.ui)}
        self.place(self.screen.art.get_size())

    def place(self, size: tuple[int, int]) -> None:
        """Give each panel its part of the frame on an art surface of a size."""
        self._size = size
        parts = layout.frame(size)
        for name, panel in self.panels.items():
            panel.place(parts[name])

    def handle(self, event: pygame.event.Event) -> None:
        """Take in one thing that happened to the window. What the mouse and the keys did goes on to the widgets.

        The window's size is followed by the screen, and needs nothing here.
        """
        if event.type == pygame.QUIT:
            self.closed = True
            return
        happened = translate(event, self.screen.to_art)
        if happened is not None:
            self.ui.handle(happened)

    def update(self) -> None:
        """A frame's share of what is not drawing: the session's waiting work, then what each panel shows."""
        self.session.tick()
        for panel in self.panels.values():
            panel.update()

    def draw(self, art: pygame.Surface) -> None:
        """Draw one frame on the art surface: the parts of the frame as flat panels, each with its line.

        The status bar is drawn on its part, then what each panel draws, and over it all the widgets. On a
        surface of another size than the last the panels are placed anew first.
        """
        canvas = self.canvas
        canvas.surface = art
        size = art.get_size()
        if size != self._size:
            self.place(size)
        canvas.box((0, 0, *size), theme.BACKGROUND)
        parts = layout.frame(size)
        for name, side in _LINES.items():
            canvas.box(parts[name], theme.PANEL)
            canvas.box(layout.edge(parts[name], side), theme.LINE)
        self._status(parts["status"])
        for panel in self.panels.values():
            if panel.shown:
                panel.draw(canvas)
        self.ui.draw(canvas)

    def _status(self, rect: Rect) -> None:
        """Draw what the status bar says: at its right end what is picked, and one line in the room before that.

        The line is the help of the widget the mouse is on. With no such help it is the session's message.
        """
        x, y, w, h = rect
        # Under the bar's line, and in from its two ends.
        left, top, room, high = x + theme.BAR_MARGIN, y + 1, max(0, w - 2 * theme.BAR_MARGIN), h - 1
        picked = picked_words(self.model.picked)
        self.canvas.text_in(picked, (left, top, room, high), theme.DIM, "right")
        if picked:
            room = max(0, room - Font.width(picked) - theme.GROUP_GAP)
        line = self.ui.help
        self.canvas.text_in(line or self.session.message, (left, top, room, high), theme.DIM if line else theme.TEXT)

    def run(self, frames: int | None = None, shot: str | Path | None = None) -> int:
        """Keep the window drawn until it is closed, then close it. Return how many frames were drawn.

        With `frames` it stops by itself after that many. With `shot` the art surface as it was last drawn is
        written to that path as a PNG, in a folder that is made when it is missing.
        """
        clock = pygame.time.Clock()
        drawn = 0
        try:
            while frames is None or drawn < frames:
                for event in pygame.event.get():
                    self.handle(event)
                if self.closed:
                    break
                self.update()
                self.draw(self.screen.begin())
                self.screen.present()
                drawn += 1
                clock.tick(FPS)
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
            self.session.close()
        finally:
            self.screen.close()
