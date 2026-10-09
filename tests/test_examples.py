"""Tests for the example projects: each one loads, passes every check and renders."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from rowsynth.bank import PEAK_DB, render_bank
from rowsynth.checks import run
from rowsynth.project import Project

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class Examples(unittest.TestCase):
    def test_there_are_examples_to_test(self) -> None:
        self.assertTrue(sorted(EXAMPLES.glob("*.json")))

    def test_every_example_loads_and_passes_the_checks(self) -> None:
        for path in sorted(EXAMPLES.glob("*.json")):
            with self.subTest(example=path.name):
                self.assertEqual(run(Project.load(path)), [])


class Arcade(unittest.TestCase):
    def setUp(self) -> None:
        self.project = Project.load(EXAMPLES / "arcade.json")

    def test_it_has_its_six_sounds_one_of_them_in_variants_and_one_in_parts(self) -> None:
        self.assertEqual(list(self.project.sounds), ["coin", "jump", "laser", "explosion", "blip", "powerup"])
        self.assertEqual(len(self.project.sounds["blip"]["defs"]), 3)
        self.assertEqual(len(self.project.sounds["explosion"]["parts"]), 2)

    def test_every_sound_renders_and_none_is_louder_than_the_bank_s_peak(self) -> None:
        sounds = {name: self.project.sound(name) for name in self.project.sounds}
        top = 10 ** (PEAK_DB / 20)
        for name, rendered in render_bank(sounds).items():
            for i, buffer in enumerate(rendered["buffers"]):
                with self.subTest(sound=name, buffer=i):
                    self.assertGreater(len(buffer), 0)
                    self.assertLessEqual(float(np.abs(buffer).max()), top + 1e-6)
