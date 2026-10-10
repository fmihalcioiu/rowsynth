"""The window, and the small surface that is drawn on and scaled up whole to fill it."""
from __future__ import annotations

import pygame

from ..errors import RowSynthError
from . import theme


def fit(window: tuple[int, int], scale: int | None = None) -> tuple[int, tuple[int, int]]:
    """The scale and the size of the art surface for a window of a size. The scale is a whole number.

    With no scale given it is the largest at which the art surface is still at least the smallest the layout is
    made for, and 1 in a window too small for that. The art surface is the window's size divided by the scale,
    rounded down, so a larger window shows more and shows nothing larger. A scale under 1 is refused.
    """
    width, height = window
    if scale is None:
        scale = max(1, min(width // theme.MIN_ART[0], height // theme.MIN_ART[1]))
    elif scale < 1:
        raise RowSynthError(f"a scale of {scale} cannot be drawn: the scale is a whole number, 1 or more")
    return scale, (max(1, width // scale), max(1, height // scale))


class Screen:
    """The window and the art surface. Everything is drawn on the art surface, in art pixels."""

    def __init__(self, size: tuple[int, int] = theme.WINDOW, scale: int | None = None) -> None:
        """Open a window of `size` that can be resized. `scale` is kept at every size when one is given."""
        self._given = scale
        # A scale that cannot be drawn is refused here, before there is a window.
        fit(size, scale)
        pygame.display.init()
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
