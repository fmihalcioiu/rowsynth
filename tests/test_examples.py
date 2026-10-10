"""Tests for the example projects: each one loads, passes every check and renders."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rowsynth.bank import PEAK_DB, render_bank
from rowsynth.build import build
from rowsynth.checks import run
from rowsynth.project import Project
from rowsynth.song import render_song

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

    def test_its_tune_is_eight_bars_with_a_chord_for_each(self) -> None:
        tune = self.project.songs["tune"]
        self.assertEqual(tune["order"], ["a", "b"])
        self.assertEqual({name: len(bars) for name, bars in tune["chords"].items()}, {"a": 4, "b": 4})

    def test_its_tune_plays_for_its_eight_bars_and_stays_under_full_scale(self) -> None:
        played = render_song(self.project.song("tune"), rate=8000)
        self.assertEqual((played["rows"], played["seconds"]), (128, 12.8))
        loudest = max(float(np.abs(played[side]).max()) for side in ("left", "right"))
        self.assertGreater(loudest, 0.25)
        self.assertLess(loudest, 1.0)

    def test_its_tune_is_layered_and_builds_as_one_file_a_group_and_key(self) -> None:
        layers = self.project.songs["tune"]["layers"]
        self.assertEqual(list(layers["groups"]), ["rhythm", "low", "middle", "top"])
        self.assertEqual(list(layers["versions"]), ["beat", "thin", "most", "full"])
        self.assertEqual(layers["keys"], [0, 2])
        with tempfile.TemporaryDirectory() as folder:
            entry = build(self.project, folder, rate=8000)["songs"]["tune"]
            written = sorted(found.name for found in Path(folder).iterdir())
        # The drums are the same in both keys, so they are one file. The other three groups are two each.
        self.assertEqual(entry["groups"]["rhythm"], {"file": "tune-rhythm.wav"})
        self.assertEqual(entry["groups"]["top"], {"files": {"0": "tune-top-0.wav", "2": "tune-top-2.wav"}})
        self.assertEqual(len(written), 8)
        self.assertIn("music.json", written)
        self.assertEqual(entry["bar_samples"] * 8, entry["loop_samples"])


class Piece(unittest.TestCase):
    def setUp(self) -> None:
        self.project = Project.load(EXAMPLES / "piece.json")

    def test_it_is_one_song_of_nine_patterns_played_by_eighteen_channels(self) -> None:
        self.assertEqual(list(self.project.songs), ["piece"])
        piece = self.project.songs["piece"]
        self.assertEqual(piece["order"], ["cold", "hits", "grow", "tune", "rise", "drop", "band", "riff", "end"])
        self.assertEqual(len(piece["channels"]), 18)

    def test_it_plays_for_thirty_three_bars_and_stays_under_full_scale(self) -> None:
        played = render_song(self.project.song("piece"), rate=8000)
        self.assertEqual((played["rows"], played["seconds"]), (33 * 16, 52.8))
        loudest = max(float(np.abs(played[side]).max()) for side in ("left", "right"))
        self.assertGreater(loudest, 0.5)
        self.assertLess(loudest, 1.0)
