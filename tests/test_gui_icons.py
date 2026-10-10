"""Tests for the icons in the GUI's set file."""
from __future__ import annotations

import unittest

from rowsynth.gui import sprites

#: Every icon the GUI has, in the groups the screen uses them in.
NAMES = (
    "new", "open", "save", "export", "build", "page",
    "undo", "redo",
    "play", "stop", "loop", "pause",
    "song", "sound", "instrument",
    "pulse", "saw", "tri", "noise", "fm",
    "pencil", "eraser", "select", "zoom_in", "zoom_out",
    "mute", "solo", "speaker", "echo", "drum", "lock",
    "plus", "minus", "close", "tick", "warn", "up", "down", "left", "right", "grip", "copy", "bin", "link",
)

SIZE = 9


class Icons(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.set = sprites.load("icons")
        cls.icons = cls.set.sprites

    def rows(self, name: str) -> tuple[str, ...]:
        """The rows of one icon."""
        return self.icons[name].rows

    def test_the_set_is_found_in_the_package_and_holds_every_icon_and_no_other(self) -> None:
        self.assertEqual(sorted(self.icons), sorted(NAMES))

    def test_every_icon_is_9_by_9(self) -> None:
        for name in NAMES:
            with self.subTest(name=name):
                self.assertEqual([len(row) for row in self.rows(name)], [SIZE] * SIZE)

    def test_every_icon_is_drawn_in_the_one_ink_of_the_set(self) -> None:
        self.assertEqual(len(self.set.palette), 1)
        for name in NAMES:
            with self.subTest(name=name):
                self.assertEqual(self.icons[name].palette, self.set.palette)
                self.assertLessEqual(set("".join(self.rows(name))), {sprites.EMPTY, *self.set.palette})

    def test_no_icon_is_empty(self) -> None:
        for name in NAMES:
            with self.subTest(name=name):
                self.assertNotEqual(set("".join(self.rows(name))), {sprites.EMPTY})

    def test_no_two_icons_share_a_shape(self) -> None:
        by_shape: dict[tuple[str, ...], list[str]] = {}
        for name in NAMES:
            by_shape.setdefault(self.rows(name), []).append(name)
        self.assertEqual([names for names in by_shape.values() if len(names) > 1], [])

    def test_icons_that_point_opposite_ways_are_mirrors_of_each_other(self) -> None:
        for one, other in (("undo", "redo"), ("left", "right")):
            with self.subTest(one=one, other=other):
                self.assertEqual(self.rows(one), tuple(row[::-1] for row in self.rows(other)))
        self.assertEqual(self.rows("up"), self.rows("down")[::-1])
