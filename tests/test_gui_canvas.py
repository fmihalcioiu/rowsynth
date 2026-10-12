"""Tests for rowsynth.gui.canvas, and for how the font cuts text to a width."""
from __future__ import annotations

import os
import unittest

from rowsynth.errors import RowSynthError
from rowsynth.gui import sprites, theme
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.font import ADVANCE, BOX, Font, fit, sprite_name

# pygame reads these when it is imported: no test opens a window or plays a sound.
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
try:
    import pygame
except ImportError:
    pygame = None

SIZE = (60, 30)

#: An icon as the icon set has them: 9 by 9, in one ink.
PLAY = (".........", ".##......", ".####....", ".######..", ".#######.", ".######..", ".####....", ".##......",
        ".........")


def cells(rows: tuple[str, ...], x: int = 0, y: int = 0) -> set[tuple[int, int]]:
    """Where a picture has ink when its top-left corner is at (x, y)."""
    return {(x + column, y + row) for row, line in enumerate(rows) for column, cell in enumerate(line) if cell != "."}


def block(x: int, y: int, w: int, h: int) -> set[tuple[int, int]]:
    """Every pixel of a rectangle."""
    return {(x + column, y + row) for column in range(w) for row in range(h)}


class Fit(unittest.TestCase):
    def test_text_that_fits_is_kept_whole_and_longer_text_is_cut_and_ends_in_two_dots(self) -> None:
        self.assertEqual([fit("lead", 23), fit("", 0)], ["lead", ""])
        self.assertEqual([fit("lead", 22), fit("a long name", 41), fit("a long name", 46)],
                         ["l..", "a lon..", "a lon.."])
        # Where not even the dots fit, as many of them as do.
        self.assertEqual([fit("lead", width) for width in (11, 10, 5, 4, 0, -3)], ["..", ".", ".", "", "", ""])
        for width in range(80):
            self.assertLessEqual(Font.width(fit("a long name", width)), width)


@unittest.skipUnless(pygame, "pygame is not installed")
class Drawing(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.set = sprites.load("font")

    def setUp(self) -> None:
        self.surface = pygame.Surface(SIZE)
        self.surface.fill(theme.BACKGROUND)
        self.canvas = Canvas(self.surface, Font(self.set))

    def painted(self) -> dict[tuple[int, int], tuple[int, ...]]:
        """Every pixel of the surface that is no longer the background's colour, and the colour it has."""
        found = {where: tuple(self.surface.get_at(where))[:3] for where in block(0, 0, *SIZE)}
        return {where: colour for where, colour in found.items() if colour != theme.BACKGROUND}

    def written(self, text: str, x: int, y: int) -> set[tuple[int, int]]:
        """Where `text` has ink when its top-left corner is at (x, y)."""
        glyphs = [self.set.sprites[sprite_name(char)].rows for char in text]
        return set().union(*(cells(rows, x + index * ADVANCE, y) for index, rows in enumerate(glyphs)))

    def test_a_box_fills_its_rectangle_and_no_more(self) -> None:
        self.canvas.box((3, 4, 5, 2), theme.ACTIVE)
        self.assertEqual(self.painted(), dict.fromkeys(block(3, 4, 5, 2), theme.ACTIVE))

    def test_a_box_that_starts_off_the_surface_is_cut_there_and_ends_where_it_ends(self) -> None:
        # Six pixels of it lie before the left edge and three over the upper one: what is left is 4 by 2.
        self.canvas.box((-6, -3, 10, 5), theme.ACTIVE)
        self.assertEqual(self.painted(), dict.fromkeys(block(0, 0, 4, 2), theme.ACTIVE))
        # One that ends before the surface begins fills nothing, and neither does one with no width.
        self.surface.fill(theme.BACKGROUND)
        for rect in ((-9, 2, 9, 3), (2, -4, 3, 4), (-20, -20, 5, 5), (3, 3, 0, 4), (3, 3, 4, -1)):
            self.canvas.box(rect, theme.ACTIVE)
        self.assertEqual(self.painted(), {})
        # Inside a clip it is the same: the box ends where it ends, and not at the end of the clip.
        with self.canvas.clip((2, 0, SIZE[0] - 2, 4)):
            self.canvas.box((-6, 1, 10, 2), theme.GOOD)
        self.assertEqual(self.painted(), dict.fromkeys(block(2, 1, 2, 2), theme.GOOD))

    def test_a_frame_is_the_outline_just_inside_its_rectangle_and_lines_run_right_and_down(self) -> None:
        self.canvas.frame((3, 4, 6, 5), theme.LINE)
        # A rectangle of no size has no outline.
        self.canvas.frame((30, 4, 0, 5), theme.LINE)
        self.canvas.hline(3, 12, 7, theme.GOOD)
        self.canvas.vline(20, 5, 6, theme.FAULT)
        self.assertEqual(self.painted(), {**dict.fromkeys(block(3, 4, 6, 5) - block(4, 5, 4, 3), theme.LINE),
                                          **dict.fromkeys(block(3, 12, 7, 1), theme.GOOD),
                                          **dict.fromkeys(block(20, 5, 1, 6), theme.FAULT)})

    def test_text_is_in_the_colour_of_text_unless_told_and_returns_the_x_after_it(self) -> None:
        self.assertEqual(self.canvas.text("Hg", 3, 2), 14)
        self.assertEqual(self.painted(), dict.fromkeys(self.written("Hg", 3, 2), theme.TEXT))
        self.assertEqual(self.canvas.text("", 3, 2, theme.DIM), 3)

    def test_text_in_a_rectangle_is_centred_up_and_down_at_the_left_the_centre_or_the_right(self) -> None:
        # Two letters are 11 wide, so 29 pixels are left over. The seven rows of their body leave 3 above.
        for align, x in (("left", 4), ("centre", 18), ("right", 33)):
            with self.subTest(align=align):
                self.surface.fill(theme.BACKGROUND)
                self.canvas.text_in("Hg", (4, 6, 40, 13), theme.SONG, align)
                self.assertEqual(self.painted(), dict.fromkeys(self.written("Hg", x, 9), theme.SONG))
        with self.assertRaises(RowSynthError) as refused:
            self.canvas.text_in("Hg", (4, 6, 40, 13), align="middle")
        self.assertIn("not at 'middle'", str(refused.exception))

    def test_text_too_long_for_its_rectangle_is_cut_and_none_of_it_is_drawn_outside(self) -> None:
        self.canvas.text_in("a name too long", (4, 6, 40, 13))
        self.assertEqual(set(self.painted()), self.written("a na..", 4, 9))
        # A rectangle lower than the letters shows the rows of them that are in it.
        self.surface.fill(theme.BACKGROUND)
        self.canvas.text_in("HH", (4, 6, 40, 3))
        self.assertEqual(set(self.painted()), self.written("HH", 4, 4) & block(4, 6, 40, 3))

    def test_inside_a_clip_nothing_is_drawn_outside_its_rectangle_and_after_it_all_is_as_before(self) -> None:
        with self.canvas.clip((10, 8, 5, 4)):
            self.canvas.box((0, 0, *SIZE), theme.ACTIVE)
            self.canvas.text("HHHHHHHHHH", 0, 0, theme.ACTIVE)
        self.assertEqual(set(self.painted()), block(10, 8, 5, 4))
        self.canvas.box((0, 0, *SIZE), theme.GOOD)
        self.assertEqual(set(self.painted()), block(0, 0, *SIZE))

    def test_a_clip_in_a_clip_is_what_the_two_share_and_each_gives_back_the_one_before(self) -> None:
        with self.canvas.clip((10, 10, 10, 10)):
            with self.canvas.clip((15, 5, 10, 10)):
                self.canvas.box((0, 0, *SIZE), theme.ACTIVE)
            with self.canvas.clip((40, 0, 5, 5)):
                self.canvas.box((0, 0, *SIZE), theme.FAULT)
            self.assertEqual(self.painted(), dict.fromkeys(block(15, 10, 5, 5), theme.ACTIVE))
            # A clip that is left by an error is given back too.
            with self.assertRaises(ZeroDivisionError), self.canvas.clip((0, 0, 1, 1)):
                1 / 0
            self.canvas.box((0, 0, *SIZE), theme.GOOD)
            self.assertEqual(self.painted(), dict.fromkeys(block(10, 10, 10, 10), theme.GOOD))
        self.assertEqual(tuple(self.surface.get_clip()), (0, 0, *SIZE))

    def test_an_icon_is_in_the_colour_asked_for_and_one_there_is_none_of_is_the_box_of_a_missing_letter(self) -> None:
        # The palette of the set is not what an icon is drawn in.
        icons = sprites.SpriteSet({}, {"play": sprites.Sprite(PLAY, {"#": theme.FAULT})})
        with_icons = Canvas(self.surface, Font(self.set), icons)
        with_icons.icon("play", 5, 6, theme.GOOD)
        self.assertEqual(self.painted(), dict.fromkeys(cells(PLAY, 5, 6), theme.GOOD))
        self.assertEqual(with_icons.icon_size("play"), (9, 9))
        # A name the set lacks, and any name on a canvas that was given no set.
        for canvas in (with_icons, self.canvas):
            self.surface.fill(theme.BACKGROUND)
            canvas.icon("stop", 5, 6, theme.MUSTARD)
            self.assertEqual(self.painted(), dict.fromkeys(cells(BOX, 5, 6), theme.MUSTARD))
            self.assertEqual(canvas.icon_size("stop"), (5, 9))


if __name__ == "__main__":
    unittest.main()
