"""Tests for rowsynth.gui.sprites."""
from __future__ import annotations

import json
import unittest

from rowsynth.errors import RowSynthError
from rowsynth.gui import sprites

DARK, GREEN = (27, 27, 38), (79, 174, 91)


def written(entry: object = None, **top: object) -> str:
    """The text of a set file with a palette of 'o' and 'g', and one sprite called 'blob' when `entry` is given."""
    data: dict = {"palette": {"o": "#1b1b26", "g": "#4FAE5B"}, "sprites": {}, "anims": {}}
    if entry is not None:
        data["sprites"]["blob"] = entry
    data.update(top)
    return json.dumps(data)


class Reading(unittest.TestCase):
    def refused(self, text: str, *reasons: str) -> None:
        """Reading `text` raises the package's error, and the message names the file and gives each reason."""
        with self.assertRaises(RowSynthError) as caught:
            sprites.parse(text, "set.json")
        for reason in ("set.json", *reasons):
            self.assertIn(reason, str(caught.exception))

    def test_the_font_s_file_is_found_in_the_package_and_read(self) -> None:
        found = sprites.load("font")
        self.assertEqual(found.palette, {"#": (239, 230, 213)})
        self.assertEqual(len(found.sprites), 95)
        letter = (".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#", ".....", ".....")
        self.assertEqual(found.sprites["u0041"], sprites.Sprite(letter, found.palette))

    def test_a_set_is_read_into_its_palette_and_the_rows_of_each_sprite(self) -> None:
        text = written({"rows": ["..oo..", ".oggo."]}, anims={"idle": {"frames": ["blob"], "fps": 3}})
        found = sprites.parse(text, "set.json")
        self.assertEqual(found.palette, {"o": DARK, "g": GREEN})
        self.assertEqual(found.sprites, {"blob": sprites.Sprite(("..oo..", ".oggo."), {"o": DARK, "g": GREEN})})

    def test_a_sprite_s_own_palette_is_laid_over_the_set_s_for_that_sprite_only(self) -> None:
        found = sprites.parse(written({"rows": ["ogw"], "palette": {"g": "#8ed97a", "w": "#ffffff"}}), "set.json")
        self.assertEqual(found.sprites["blob"].palette, {"o": DARK, "g": (142, 217, 122), "w": (255, 255, 255)})
        self.assertEqual(found.palette, {"o": DARK, "g": GREEN})

    def test_a_file_the_package_does_not_have_is_refused_by_name(self) -> None:
        with self.assertRaises(RowSynthError) as caught:
            sprites.load("no-such-set")
        self.assertIn("no-such-set.json", str(caught.exception))

    def test_text_that_is_not_json_is_refused(self) -> None:
        self.refused('{"palette": ', "not JSON at line 1")

    def test_a_wrong_shape_at_the_top_is_refused(self) -> None:
        for text, reason in (
            ("[]", "a set file is an object"),
            (written(sprite={}), "a set file is an object"),
            (written(sprites=[]), "'sprites' must be an object"),
        ):
            with self.subTest(text=text):
                self.refused(text, reason)

    def test_a_bad_palette_is_refused(self) -> None:
        for palette, reason in (
            (["o"], "the palette must be an object"),
            ({"og": "#1b1b26"}, "a palette key is one character"),
            ({".": "#1b1b26"}, "a palette key is one character"),
            ({" ": "#1b1b26"}, "a palette key is one character"),
            ({"o": "#1b1b2"}, "the colour of 'o' must be written '#rrggbb'"),
            ({"o": [27, 27, 38]}, "the colour of 'o' must be written '#rrggbb'"),
        ):
            with self.subTest(palette=palette):
                self.refused(written(palette=palette), reason)

    def test_a_bad_sprite_is_refused_by_name(self) -> None:
        for entry, reason in (
            (["oo"], "must be an object with 'rows'"),
            ({"rows": ["oo"], "size": 2}, "must be an object with 'rows'"),
            ({}, "'rows' must be a list of strings"),
            ({"rows": "oo"}, "'rows' must be a list of strings"),
            ({"rows": []}, "'rows' must be a list of strings"),
            ({"rows": ["oo", ""]}, "'rows' must be a list of strings"),
            ({"rows": ["oo", "ogo"]}, "its rows are not all the same length"),
            ({"rows": ["oo", "ox"]}, "row 1 has 'x' at column 1, which is not in the palette"),
            ({"rows": ["oo"], "palette": {"g": "green"}}, "the colour of 'g' must be written '#rrggbb'"),
        ):
            with self.subTest(entry=entry):
                self.refused(written(entry), "sprite 'blob'", reason)
