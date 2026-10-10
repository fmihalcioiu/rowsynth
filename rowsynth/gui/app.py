"""The main loop: a project file open in a window, drawn sixty times a second."""
from __future__ import annotations

from pathlib import Path

import pygame

from ..errors import RowSynthError
from ..project import Project
from . import layout, theme
from .screen import Screen

FPS = 60

# The side of each part of the frame that its line runs along: the one that faces the middle.
_LINES = {"top": "bottom", "left": "right", "right": "left", "status": "top"}


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

    def handle(self, event: pygame.event.Event) -> None:
        """Take in one thing that happened to the window. Its size is followed by the screen, and needs nothing here."""
        if event.type == pygame.QUIT:
            self.closed = True

    def draw(self, art: pygame.Surface) -> None:
        """Draw one frame on the art surface: the parts of the frame as flat panels, each with its line."""
        art.fill(theme.BACKGROUND)
        parts = layout.frame(art.get_size())
        for name, side in _LINES.items():
            art.fill(theme.PANEL, parts[name])
            art.fill(theme.LINE, layout.edge(parts[name], side))

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
