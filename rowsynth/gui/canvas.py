"""The one thing widgets draw with: boxes, lines, text and icons on the art surface, in art pixels."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from ..errors import RowSynthError
from . import layout, theme
from .font import BODY_H, Font, Stamps, fit
from .layout import Rect
from .sprites import SpriteSet
from .theme import Colour

if TYPE_CHECKING:
    import pygame

#: Where a line of text can be put in a rectangle, from side to side.
ALIGNS = ("left", "centre", "right")


class Canvas:
    """A surface to draw on, the font its text is in, and the icons it can draw.

    A rectangle is (x, y, w, h), as the layout gives them. `surface` can be set to another surface when the window
    gives a new one: what has been made of the glyphs and the icons is kept.
    """

    def __init__(self, surface: pygame.Surface, font: Font, icons: SpriteSet | None = None) -> None:
        """`icons` is a set of sprites of one ink. Without it every icon is drawn as the box of a missing letter."""
        self.surface = surface
        self.font = font
        self._icons = Stamps({name: sprite.rows for name, sprite in icons.sprites.items()} if icons else {})

    def box(self, rect: Rect, colour: Colour) -> None:
        """Fill a rectangle. What of it lies off the surface is left out.

        One that starts before the left edge or over the upper one is cut there first: pygame would move it
        onto the surface at its whole size, and so fill pixels after its end.
        """
        x, y, w, h = rect
        if x < 0:
            x, w = 0, w + x
        if y < 0:
            y, h = 0, h + y
        if w > 0 and h > 0:
            self.surface.fill(colour, (x, y, w, h))

    def frame(self, rect: Rect, colour: Colour) -> None:
        """Draw the outline of a rectangle, one pixel thick and just inside it."""
        if rect[2] > 0 and rect[3] > 0:
            for side in ("top", "bottom", "left", "right"):
                self.box(layout.edge(rect, side), colour)

    def hline(self, x: int, y: int, w: int, colour: Colour) -> None:
        """Draw a line `w` pixels long from (x, y) to the right."""
        self.box((x, y, w, 1), colour)

    def vline(self, x: int, y: int, h: int, colour: Colour) -> None:
        """Draw a line `h` pixels long from (x, y) down."""
        self.box((x, y, 1, h), colour)

    def text(self, text: str, x: int, y: int, colour: Colour = theme.TEXT) -> int:
        """Draw `text` with its top-left corner at (x, y), and return the x just after it."""
        return self.font.draw(self.surface, text, x, y, colour)

    def text_in(self, text: str, rect: Rect, colour: Colour = theme.TEXT, align: str = "left") -> None:
        """Draw one line of text in a rectangle: centred up and down, and at the left, the centre or the right.

        Nothing is drawn outside the rectangle. Text too long for it is cut, and ends in '..'.
        """
        if align not in ALIGNS:
            raise RowSynthError(f"text is put at the left, the centre or the right of a rectangle, not at {align!r}")
        x, y, w, h = rect
        shown = fit(text, w)
        spare = w - self.font.width(shown)
        x += {"left": 0, "centre": spare // 2, "right": spare}[align]
        with self.clip(rect):
            # The body of the letters is what is centred: the tails hang below it.
            self.text(shown, x, y + (h - BODY_H) // 2, colour)

    def icon_size(self, name: str) -> tuple[int, int]:
        """How wide and how high the icon of a name is drawn: for one that is missing, the size of the box."""
        return self._icons.size(name)

    def icon(self, name: str, x: int, y: int, colour: Colour) -> None:
        """Draw the icon of a name in a colour, its top-left corner at (x, y). A missing one is drawn as the box."""
        self.surface.blit(self._icons.get(name, colour), (x, y))

    @contextmanager
    def clip(self, rect: Rect) -> Iterator[None]:
        """Inside the `with`, nothing is drawn outside a rectangle, nor outside the clip there was before.

        The clip there was before comes back after.
        """
        before = self.surface.get_clip()
        self.surface.set_clip(before.clip(rect))
        try:
            yield
        finally:
            self.surface.set_clip(before)
