"""The window, and the small surface that is drawn on and scaled up whole to fill it."""
from __future__ import annotations

import ctypes
import sys

import pygame

from ..errors import RowSynthError
from . import theme


def fit(window: tuple[int, int], scale: int | None = None) -> tuple[int, tuple[int, int]]:
    """The scale and the size of the art surface for a window of a size. The scale is a whole number.

    With no scale given it is the largest at which the art surface is no smaller than `theme.MIN_ART`, and 1 in
    a window too small for that. The art surface is the window's size divided by the scale, rounded down, so a
    larger window shows more and shows nothing larger. A scale under 1 is refused.
    """
    width, height = window
    if scale is None:
        scale = max(1, min(width // theme.MIN_ART[0], height // theme.MIN_ART[1]))
    elif scale < 1:
        raise RowSynthError(f"a scale of {scale} cannot be drawn: the scale is a whole number, 1 or more")
    return scale, (max(1, width // scale), max(1, height // scale))


def opening(desktop: tuple[int, int], scale: int | None = None) -> tuple[int, int]:
    """The size a window opens at on a desktop of a size, so that it fits there.

    The desktop's height is first less by what the title bar and the task bar take. The window is the size the
    layout is drawn for times a scale, cut down to what the desktop has left. The scale is the one given, or
    with none the largest that `fit` would still choose in what is left. A desktop too small for the layout at
    scale 1 gets a window of what it has left.
    """
    width, height = max(1, desktop[0]), max(1, desktop[1] - theme.DESKTOP_TAKEN_H)
    scale, _ = fit((width, height), scale)
    return min(theme.DESIGN[0] * scale, width), min(theme.DESIGN[1] * scale, height)


def dpi_aware() -> None:
    """On Windows, tell the system that this process draws its own pixels. Elsewhere there is nothing to tell.

    Untold, Windows stretches the whole window on a display that is scaled above 100%, which smooths the pixels.
    It has to be told before the display is started. Where it cannot be told, the window is opened all the same.
    """
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


class Screen:
    """The window and the art surface. Everything is drawn on the art surface, in art pixels."""

    def __init__(self, size: tuple[int, int] | None = None, scale: int | None = None) -> None:
        """Open a window that can be resized. `scale` is kept at every size when one is given.

        A `size` that is given is the window's, as it is. With none the window opens at the size that fits the
        first desktop, which is the main one.
        """
        self._given = scale
        # A scale that cannot be drawn is refused here, before the display is started.
        fit(theme.DESIGN, scale)
        dpi_aware()
        pygame.display.init()
        if size is None:
            size = opening(pygame.display.get_desktop_sizes()[0], scale)
        pygame.display.set_caption("rowsynth")
        self.window = pygame.display.set_mode(size, pygame.RESIZABLE)
        # The window may not have come out at the size asked for, so its own size is the one that counts.
        self.scale, art = fit(self.window.get_size(), scale)
        self.art = pygame.Surface(art)

    def begin(self) -> pygame.Surface:
        """The art surface, to draw the next frame on. It follows the window: after a resize it has a new size."""
        self.window = pygame.display.get_surface()
        self.scale, art = fit(self.window.get_size(), self._given)
        if art != self.art.get_size():
            self.art = pygame.Surface(art)
        return self.art

    def present(self) -> None:
        """Show the art surface in the window: each of its pixels as a square of whole pixels, with no smoothing."""
        width, height = self.art.get_size()
        shown = (width * self.scale, height * self.scale)
        if shown != self.window.get_size():
            # The window is not a whole number of art pixels across or down, and a strip is left over at an edge.
            self.window.fill(theme.BACKGROUND)
        self.window.blit(self.art if self.scale == 1 else pygame.transform.scale(self.art, shown), (0, 0))
        pygame.display.flip()

    def to_art(self, pos: tuple[int, int]) -> tuple[int, int]:
        """A position in the window, as the mouse gives it, in pixels of the art surface."""
        return pos[0] // self.scale, pos[1] // self.scale

    def snapshot(self) -> pygame.Surface:
        """A copy of the art surface as it is now."""
        return self.art.copy()

    def close(self) -> None:
        """Close the window."""
        pygame.display.quit()
