"""Tests for the example project: it loads, passes every check and renders, and its songs are the ones it says."""
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
# The two projects the demo was put together from, which the tests of the rest still read.
DATA = Path(__file__).resolve().parent / "data"


class Examples(unittest.TestCase):
    def test_there_is_the_one_example_to_test(self) -> None:
        self.assertEqual([path.name for path in sorted(EXAMPLES.glob("*.json"))], ["demo.json"])

    def test_every_example_loads_and_passes_the_checks(self) -> None:
        for path in sorted(EXAMPLES.glob("*.json")):
            with self.subTest(example=path.name):
                self.assertEqual(run(Project.load(path)), [])


class Demo(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project = Project.load(EXAMPLES / "demo.json")

    def test_it_has_three_songs_and_the_trailer_song_plays_instruments_of_its_own(self) -> None:
        self.assertEqual(list(self.project.songs), ["tune", "korobeiniki", "piece"])
        self.assertEqual(list(self.project.instruments),
                         ["chime", "hop", "zap", "rumble", "thump", "tick", "lead", "pluck", "bass", "kick", "snare",
                          "hat"])
        # Two of the twelve have names the project has for other instruments: the song plays its own.
        own = self.project.songs["piece"]["instruments"]
        self.assertEqual(list(own), ["hat", "ohat", "punch", "snare", "fmpad", "fmdrive", "fmbrass", "fmstab",
                                     "fmarp", "crash", "sub", "stamp"])
        self.assertTrue(all(own[name] != self.project.instruments[name] for name in ("hat", "snare")))
        self.assertTrue(all("instruments" not in self.project.songs[name] for name in ("tune", "korobeiniki")))

    def test_its_first_song_and_its_last_sound_as_the_projects_they_came_from(self) -> None:
        for name, file in (("tune", "arcade.json"), ("piece", "piece.json")):
            with self.subTest(song=name):
                before = render_song(Project.load(DATA / file).song(name), rate=8000)
                played = render_song(self.project.song(name), rate=8000)
                for side in ("left", "right"):
                    np.testing.assert_array_equal(played[side], before[side])

    def test_its_second_song_is_thirty_five_bars_of_six_channels_and_stays_under_full_scale(self) -> None:
        song = self.project.songs["korobeiniki"]
        self.assertEqual(song["order"], ["intro", "first", "second", "middle", "third", "close"])
        self.assertEqual(list(song["channels"]), ["lead", "under", "bell", "arp", "bass", "kit"])
        played = render_song(self.project.song("korobeiniki"), rate=8000)
        self.assertEqual((played["rows"], played["seconds"]), (35 * 16, 56.0))
        loudest = max(float(np.abs(played[side]).max()) for side in ("left", "right"))
        self.assertGreater(loudest, 0.25)
        self.assertLess(loudest, 1.0)

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
        # Seven files of the tune, one loop for each of the other two songs, and the manifest.
        self.assertEqual((len([name for name in written if name.startswith("tune-")]), len(written)), (7, 10))
        self.assertTrue({"music.json", "korobeiniki.wav", "piece.wav"} <= set(written))
        self.assertEqual(entry["bar_samples"] * 8, entry["loop_samples"])

    def test_its_trailer_song_is_nine_patterns_played_by_eighteen_channels(self) -> None:
        piece = self.project.songs["piece"]
        self.assertEqual(piece["order"], ["cold", "hits", "grow", "tune", "rise", "drop", "band", "riff", "end"])
        self.assertEqual(len(piece["channels"]), 18)

    def test_its_trailer_song_plays_for_thirty_three_bars_and_stays_under_full_scale(self) -> None:
        played = render_song(self.project.song("piece"), rate=8000)
        self.assertEqual((played["rows"], played["seconds"]), (33 * 16, 52.8))
        loudest = max(float(np.abs(played[side]).max()) for side in ("left", "right"))
        self.assertGreater(loudest, 0.5)
        self.assertLess(loudest, 1.0)
