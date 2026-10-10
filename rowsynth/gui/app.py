"""The main loop: a project file open in a window, drawn sixty times a second."""
from __future__ import annotations

from pathlib import Path

import pygame

from ..errors import RowSynthError
from ..project import Project
from . import layout, sprites, theme
from .canvas import Canvas
from .font import Font
from .screen import Screen
from .ui import Toggle, Ui, translate

FPS = 60

# The side of each part of the frame that its line runs along: the one that faces the middle.
_LINES = {"top": "bottom", "left": "right", "right": "left", "status": "top"}

# Where the one switch of the top bar stands, until the bar has its own parts.
_SWITCH = (6, 4, 25, 13)
# From each end of the status bar to its line of help.
_MARGIN = 6


class App:
    """The project of a file, the window it is shown in, and the loop that keeps the window drawn."""

    def __init__(self, path: str | Path, scale: int | None = None) -> None:
        """Read the project in the file, then open the window. A file that holds no project is refused first."""
        try:
            self.project = Project.load(path)
        except FileNotFoundError:
            raise RowSynthError(f"{path}: no such file") from None
        self.path = path
        self.screen = Screen(scale=scale)
        self.closed = False
        self.canvas = Canvas(self.screen.art, Font(sprites.load("font")))
        self.ui = Ui()
        self.ui.add(Toggle(_SWITCH, "KIT", help="a switch that does nothing yet"))

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

    def draw(self, art: pygame.Surface) -> None:
        """Draw one frame on the art surface: the parts of the frame as flat panels, each with its line.

        The status bar shows the help line of the widget the mouse is on. The widgets are drawn over the frame.
        """
        canvas = self.canvas
        canvas.surface = art
        size = art.get_size()
        canvas.box((0, 0, *size), theme.BACKGROUND)
        parts = layout.frame(size)
        for name, side in _LINES.items():
            canvas.box(parts[name], theme.PANEL)
            canvas.box(layout.edge(parts[name], side), theme.LINE)
        x, y, w, h = parts["status"]
        # Under the bar's line, and in from its two ends.
        canvas.text_in(self.ui.help, (x + _MARGIN, y + 1, w - 2 * _MARGIN, h - 1), theme.DIM)
        self.ui.draw(canvas)

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
            self.screen.close()
        return drawn
