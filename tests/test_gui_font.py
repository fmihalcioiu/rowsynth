"""Tests for rowsynth.gui.font, and for the glyphs in the font's set file."""
from __future__ import annotations

import os
import unittest
from itertools import combinations

from rowsynth.errors import RowSynthError
from rowsynth.gui import sprites
from rowsynth.gui.font import ADVANCE, BOX, Font, sprite_name

# pygame reads these when it is imported: no test opens a window or plays a sound.
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
try:
    import pygame
except ImportError:
    pygame = None

PRINTABLE = [chr(code) for code in range(32, 127)]

#: Characters the user reads notes and numbers with, grouped where one could be taken for another.
LOOKALIKES = ("l1I", "0O", "5S", "8B", "g9", "hH")

#: A letter the font has no glyph for: an e with an acute accent.
UNKNOWN = chr(0xE9)

PANEL = (38, 36, 34, 255)
ORANGE = (240, 140, 46, 255)


def cells(rows: tuple[str, ...], x: int = 0, y: int = 0) -> set[tuple[int, int]]:
    """Where a glyph has ink when its top-left corner is at (x, y)."""
    return {(x + column, y + row) for row, line in enumerate(rows) for column, cell in enumerate(line) if cell != "."}


class Glyphs(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.glyphs = sprites.load("font").sprites

    def rows(self, char: str) -> tuple[str, ...]:
        """The rows of one character's glyph."""
        return self.glyphs[sprite_name(char)].rows

    def test_every_printable_ascii_character_has_a_glyph_5_by_9(self) -> None:
        for char in PRINTABLE:
            with self.subTest(char=char):
                self.assertEqual([len(row) for row in self.rows(char)], [5] * 9)

    def test_no_two_characters_share_a_shape(self) -> None:
        by_shape: dict[tuple[str, ...], str] = {}
        for char in PRINTABLE:
            by_shape[self.rows(char)] = by_shape.get(self.rows(char), "") + char
        self.assertEqual([chars for chars in by_shape.values() if len(chars) > 1], [])

    def test_characters_that_look_alike_differ_in_at_least_three_pixels(self) -> None:
        for group in LOOKALIKES:
            for one, other in combinations(group, 2):
                with self.subTest(one=one, other=other):
                    self.assertGreaterEqual(len(cells(self.rows(one)) ^ cells(self.rows(other))), 3)

    def test_only_the_tails_and_the_bar_go_below_the_line(self) -> None:
        below = {char for char in PRINTABLE if cells(self.rows(char)[7:])}
        self.assertEqual(below, set("gjpqy|"))


class Measuring(unittest.TestCase):
    def test_glyphs_sit_six_apart_and_nothing_is_counted_after_the_last(self) -> None:
        self.assertEqual([Font.width(text) for text in ("", "a", "ab", "rowsynth")], [0, 5, 11, 47])
        self.assertEqual(Font.LINE, 11)

    def test_a_glyph_of_another_size_is_refused_by_name_and_other_sprites_are_left_out(self) -> None:
        mark = sprites.Sprite(("##",), {})
        Font(sprites.SpriteSet({}, {"mark": mark}))
        with self.assertRaises(RowSynthError) as caught:
            Font(sprites.SpriteSet({}, {"mark": mark, "u0041": mark}))
        self.assertIn("glyph 'u0041' is 2 by 1", str(caught.exception))


@unittest.skipUnless(pygame, "pygame is not installed")
class Drawing(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.set = sprites.load("font")
        cls.glyphs = cls.set.sprites

    def setUp(self) -> None:
        self.font = Font(self.set)
        self.surface = pygame.Surface((40, 16))
        self.surface.fill(PANEL)

    def painted(self) -> dict[tuple[int, int], tuple[int, ...]]:
        """Every pixel of the surface that is no longer the panel's colour, and the colour it has."""
        width, height = self.surface.get_size()
        found = {(x, y): tuple(self.surface.get_at((x, y))) for x in range(width) for y in range(height)}
        return {where: colour for where, colour in found.items() if colour != PANEL}

    def test_text_puts_ink_where_its_glyphs_have_it_and_nowhere_else_in_the_colour_asked_for(self) -> None:
        self.font.draw(self.surface, "Ag |", 3, 2, ORANGE)
        expected: set[tuple[int, int]] = set()
        for index, char in enumerate("Ag |"):
            expected |= cells(self.glyphs[sprite_name(char)].rows, 3 + index * ADVANCE, 2)
        painted = self.painted()
        self.assertEqual(set(painted), expected)
        self.assertEqual(set(painted.values()), {ORANGE})

    def test_draw_returns_the_x_just_past_the_last_glyph(self) -> None:
        self.assertEqual(self.font.draw(self.surface, "abc", 3, 2, ORANGE), 20)
        self.assertEqual(self.font.draw(self.surface, "", 3, 2, ORANGE), 3)

    def test_a_character_with_no_glyph_is_drawn_as_a_box_no_glyph_looks_like(self) -> None:
        self.font.draw(self.surface, UNKNOWN, 3, 2, "#f08c2e")
        painted = self.painted()
        self.assertEqual(set(painted), cells(BOX, 3, 2))
        self.assertEqual(set(painted.values()), {ORANGE})
        self.assertNotIn(BOX, [sprite.rows for sprite in self.glyphs.values()])

    def test_a_glyph_in_a_colour_is_made_once_and_kept(self) -> None:
        first = self.font.glyph("A", ORANGE)
        self.font.draw(self.surface, "AAA", 0, 0, (240, 140, 46))
        self.assertIs(self.font.glyph("A", "#f08c2e"), first)
        self.assertIsNot(self.font.glyph("A", "#efe6d5"), first)
        self.assertIs(self.font.glyph(UNKNOWN, ORANGE), self.font.glyph("\n", ORANGE))
