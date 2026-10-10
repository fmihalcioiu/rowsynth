"""The GUI's pixel font: text measured, cut to a width and drawn in art pixels. Its glyphs are stamps of one ink."""
from __future__ import annotations

import re
from types import ModuleType
from typing import TYPE_CHECKING

from ..errors import RowSynthError
from .sprites import EMPTY, Colour, SpriteSet

if TYPE_CHECKING:
    import pygame

#: The size of every glyph: seven rows of body, and two below the line for tails.
GLYPH_W = 5
GLYPH_H = 9
BODY_H = 7

#: From the left edge of one glyph to the left edge of the next: the glyph and a gap of 1.
ADVANCE = GLYPH_W + 1

#: What stands for a character the font has no glyph for, so that the gap is seen.
BOX = ("#####", "#...#", "#...#", "#...#", "#...#", "#...#", "#####", ".....", ".....")

#: What text that was cut ends in.
CUT = ".."

_GLYPH_NAME = re.compile(r"u[0-9a-f]{4}")


def sprite_name(char: str) -> str:
    """What a character's sprite is called in the font's set file: 'u' and its code as four hex digits."""
    return f"u{ord(char):04x}"


def fit(text: str, width: int) -> str:
    """`text` as it is shown in `width` pixels: whole when it fits, and if not cut to what fits, ending in '..'.

    Where not even the two dots fit, it is as many of them as do.
    """
    room = max(0, (width + 1) // ADVANCE)
    if len(text) <= room:
        return text
    return (text[:max(0, room - len(CUT))] + CUT)[:room]


class Stamps:
    """
    Pictures of one ink by name, each a tuple of rows. A glyph is one, and so is an icon.

    Every painted pixel is drawn in the colour asked for. A picture in a colour is made when it is first asked
    for, and kept. A name there is no picture for gives the box.
    """

    def __init__(self, rows: dict[str, tuple[str, ...]]) -> None:
        self._rows = dict(rows)
        self._made: dict[tuple[str | None, tuple[int, ...]], pygame.Surface] = {}

    def size(self, name: str) -> tuple[int, int]:
        """How wide and how high the picture of a name is drawn."""
        rows = self._rows.get(name, BOX)
        return len(rows[0]), len(rows)

    def get(self, name: str, colour: Colour | str | pygame.Color) -> pygame.Surface:
        """The picture of a name in one colour, clear where it has no ink."""
        return self.inked(name, _ink(colour))

    def inked(self, name: str, ink: tuple[int, ...]) -> pygame.Surface:
        """As `get`, for a colour that is already (red, green, blue, alpha)."""
        key = (name if name in self._rows else None, ink)
        made = self._made.get(key)
        if made is None:
            made = self._made[key] = _pygame().Surface(self.size(name), _pygame().SRCALPHA)
            for y, row in enumerate(self._rows.get(name, BOX)):
                for x, cell in enumerate(row):
                    if cell != EMPTY:
                        made.set_at((x, y), ink)
        return made


class Font:
    """
    Text in the glyphs of a set file whose sprites are named as `sprite_name` names them.

    Every painted pixel of a glyph is drawn in the colour asked for: the palette
    of the set is only what the file is shown in.
    """

    #: The height of a line of text: the nine rows of a glyph and two of air.
    LINE = GLYPH_H + 2

    def __init__(self, glyphs: SpriteSet) -> None:
        rows: dict[str, tuple[str, ...]] = {}
        for name, sprite in glyphs.sprites.items():
            if not _GLYPH_NAME.fullmatch(name):
                continue
            width, height = len(sprite.rows[0]), len(sprite.rows)
            if (width, height) != (GLYPH_W, GLYPH_H):
                raise RowSynthError(f"glyph {name!r} is {width} by {height}, a glyph is {GLYPH_W} by {GLYPH_H}")
            rows[chr(int(name[1:], 16))] = sprite.rows
        self._glyphs = Stamps(rows)

    @staticmethod
    def width(text: str) -> int:
        """How wide `text` is drawn. Glyphs sit 6 apart, and nothing is counted after the last one."""
        return len(text) * ADVANCE - 1 if text else 0

    def glyph(self, char: str, colour: Colour | str | pygame.Color) -> pygame.Surface:
        """One character in one colour, clear where it has no ink. It is made once and kept."""
        return self._glyphs.get(char, colour)

    def draw(self, surface: pygame.Surface, text: str, x: int, y: int, colour: Colour | str | pygame.Color) -> int:
        """
        Draws `text` with its top-left corner at (x, y) and returns the x just past the last glyph.

        A character with no glyph is drawn as a box.
        """
        ink = _ink(colour)
        for index, char in enumerate(text):
            surface.blit(self._glyphs.inked(char, ink), (x + index * ADVANCE, y))
        return x + self.width(text)


def _pygame() -> ModuleType:
    """pygame, imported when text is first drawn and not at the top, so that measuring text needs none of it."""
    import pygame

    return pygame


def _ink(colour: Colour | str | pygame.Color) -> tuple[int, ...]:
    """A colour given in any way pygame takes one, as (red, green, blue, alpha)."""
    return tuple(_pygame().Color(colour))
