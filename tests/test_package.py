"""Tests for the package as a whole: the names it gives, its version and its command."""
from __future__ import annotations

import re
import unittest
from importlib import import_module
from pathlib import Path

import rowsynth

SETTINGS = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")


class Package(unittest.TestCase):
    def test_every_name_it_says_it_gives_is_there(self) -> None:
        for name in rowsynth.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(rowsynth, name))

    def test_its_names_are_listed_in_order_and_once(self) -> None:
        self.assertEqual(rowsynth.__all__, sorted(set(rowsynth.__all__)))

    def test_a_sound_can_be_made_from_the_top_of_the_package_alone(self) -> None:
        project = rowsynth.Project.from_dict({
            "instruments": {"beep": {"vol": [12, 8, 4]}},
            "sounds": {"ping": {"def": {"voices": [{"inst": "beep", "notes": [[0, "a5", 3]]}]}}},
        })
        (buffer,) = rowsynth.render_entry(project.sound("ping"), 8000)
        self.assertEqual(rowsynth.measure(buffer, 8000)["peak_db"], -3.0)

    def test_its_version_is_the_one_in_the_package_settings(self) -> None:
        found = re.search(r'^version = "([^"]+)"$', SETTINGS, re.MULTILINE)
        self.assertEqual(rowsynth.__version__, found[1])

    def test_the_command_in_the_package_settings_is_the_command_line_s_main(self) -> None:
        found = re.search(r'^rowsynth = "([\w.]+):(\w+)"$', SETTINGS, re.MULTILINE)
        self.assertIs(getattr(import_module(found[1]), found[2]), import_module("rowsynth.cli").main)
