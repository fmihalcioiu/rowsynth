"""The GUI's pixel font: text measured and drawn in art pixels."""
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

#: From the left edge of one glyph to the left edge of the next: the glyph and a gap of 1.
ADVANCE = GLYPH_W + 1

#: What stands for a character the font has no glyph for, so that the gap is seen.
BOX = ("#####", "#...#", "#...#", "#...#", "#...#", "#...#", "#####", ".....", ".....")

_GLYPH_NAME = re.compile(r"u[0-9a-f]{4}")


def sprite_name(char: str) -> str:
    """What a character's sprite is called in the font's set file: 'u' and its code as four hex digits."""
    return f"u{ord(char):04x}"


class Font:
    """
    Text in the glyphs of a set file whose sprites are named as `sprite_name` names them.

    Every painted pixel of a glyph is drawn in the colour asked for: the palette
    of the set is only what the file is shown in.
    """

    #: The height of a line of text: the nine rows of a glyph and two of air.
    LINE = GLYPH_H + 2

    def __init__(self, glyphs: SpriteSet) -> None:
        self._rows: dict[str, tuple[str, ...]] = {}
        self._made: dict[tuple[str, tuple[int, ...]], pygame.Surface] = {}
        for name, sprite in glyphs.sprites.items():
            if not _GLYPH_NAME.fullmatch(name):
                continue
            width, height = len(sprite.rows[0]), len(sprite.rows)
            if (width, height) != (GLYPH_W, GLYPH_H):
                raise RowSynthError(f"glyph {name!r} is {width} by {height}, a glyph is {GLYPH_W} by {GLYPH_H}")
            self._rows[chr(int(name[1:], 16))] = sprite.rows

    @staticmethod
    def width(text: str) -> int:
        """How wide `text` is drawn. Glyphs sit 6 apart, and nothing is counted after the last one."""
        return len(text) * ADVANCE - 1 if text else 0

    def glyph(self, char: str, colour: Colour | str | pygame.Color) -> pygame.Surface:
        """One character in one colour, clear where it has no ink. It is made once and kept."""
        return self._glyph(char, _ink(colour))

    def draw(self, surface: pygame.Surface, text: str, x: int, y: int, colour: Colour | str | pygame.Color) -> int:
        """
        Draws `text` with its top-left corner at (x, y) and returns the x just past the last glyph.

        A character with no glyph is drawn as a box.
        """
        ink = _ink(colour)
        for index, char in enumerate(text):
            surface.blit(self._glyph(char, ink), (x + index * ADVANCE, y))
        return x + self.width(text)

    def _glyph(self, char: str, ink: tuple[int, ...]) -> pygame.Surface:
        """The kept glyph of `char` in `ink`, or the box where the font has none. Made when first asked for."""
        key = (char if char in self._rows else "", ink)
        made = self._made.get(key)
        if made is None:
            made = self._made[key] = _pygame().Surface((GLYPH_W, GLYPH_H), _pygame().SRCALPHA)
            for y, row in enumerate(self._rows.get(char, BOX)):
                for x, cell in enumerate(row):
                    if cell != EMPTY:
                        made.set_at((x, y), ink)
        return made


def _pygame() -> ModuleType:
    """pygame, imported when text is first drawn and not at the top, so that measuring text needs none of it."""
    import pygame

    return pygame


def _ink(colour: Colour | str | pygame.Color) -> tuple[int, ...]:
    """A colour given in any way pygame takes one, as (red, green, blue, alpha)."""
    return tuple(_pygame().Color(colour))
