"""Tests for a layered song cut into groups, each mixed to stereo in every key."""
from __future__ import annotations

import copy
import math
import unittest
from unittest import mock

import numpy as np

from rowsynth.errors import RowSynthError
from rowsynth.layers import bar_rows, keyless, pan_gains, render_groups
from rowsynth.song import render_stems, up

# With three ticks to a row, a row is 400 samples at 8000 Hz.
RATE = 8000

# Two groups of tuned channels and one of drums alone.
GROUPS = {"top": ["air", "mid"], "base": ["low"], "beat": ["kick", "hat"]}

# Three tuned channels and two drum channels, heard as written and two semitones up. The low channel has no pan,
# and the second pattern has no text for the middle one.
SONG = {
    "row_ticks": 3,
    "instruments": {
        "reed": {"vol": [14, 11, 9], "duty": 1, "gate": 0.6, "release": 1},
        "buzz": {"wave": "saw", "vol": 9, "lp": 1800},
        "sub": {"wave": "tri", "vol": 12},
        "thump": {"wave": "noise", "vol": [15, 8, 3], "release": 0},
        "tap": {"wave": "noise", "short": True, "vol": [9, 2], "release": 0},
    },
    "drums": {"k": {"inst": "thump", "note": "c2"}, "h": {"inst": "tap", "note": 88, "vel": 0.6}},
    "channels": {
        "air": {"inst": "reed", "pan": -0.6, "echo": {"rows": 1.5, "gain": 0.4}},
        "mid": {"inst": "buzz", "gain": 0.7, "pan": 0.7},
        "low": {"inst": "sub"},
        "kick": {"inst": "thump", "drums": True, "pan": -0.2},
        "hat": {"inst": "tap", "drums": True, "pan": 0.5},
    },
    "patterns": {
        "a": {"air": "l8 e g > c < b", "mid": "o3 g2", "low": "o2 c4 g4", "kick": "k4 k4", "hat": "l8 r h r h"},
        "b": {"air": "o5 d4 c4", "low": "o2 f2", "kick": "k2", "hat": "l8 h h r h"},
    },
    "order": ["a", "b"],
    "layers": {"groups": GROUPS, "keys": [0, 2]},
}


def layered(**layers: object) -> dict:
    """The song with other layers."""
    return {**SONG, "layers": layers}


def by_hand(song: dict, channels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """The two sides of a group of a song's channels, made from its stems one after the other.

    Each stem goes in at the cosine and the sine of the angle its pan comes to. A sum is made in doubles and
    rounded to float32 before the next stem is added to it.
    """
    played = render_stems(song, RATE)
    stems = {found["name"]: found for found in played["stems"]}
    left = np.zeros(played["loop_samples"], dtype=np.float32)
    right = np.zeros(played["loop_samples"], dtype=np.float32)
    for name in channels:
        angle = (stems[name]["pan"] + 1) * math.pi / 4
        wide = stems[name]["samples"].astype(np.float64)
        left = (left.astype(np.float64) + wide * math.cos(angle)).astype(np.float32)
        right = (right.astype(np.float64) + wide * math.sin(angle)).astype(np.float32)
    return left, right


def same(test: unittest.TestCase, got: tuple, wanted: tuple, about: object = None) -> None:
    """Holds that the two sides of a mix are the wanted ones to the last bit, and that neither is silence."""
    test.assertEqual(len(got), 2, about)
    for side, hand in zip(got, wanted):
        test.assertTrue(np.array_equal(side, hand), about)
        test.assertGreater(float(np.abs(hand).max()), 0.01, about)


class PanGainsTest(unittest.TestCase):
    def test_ends_are_all_on_one_side_and_the_middle_has_each_side_3_decibels_down(self):
        self.assertEqual(pan_gains(-1), (1.0, 0.0))
        left, right = pan_gains(1)
        self.assertLess(abs(left), 1e-15)
        self.assertEqual(right, 1.0)
        for gain in pan_gains(0):
            self.assertAlmostEqual(gain, math.sqrt(0.5), 15)
            self.assertAlmostEqual(20 * math.log10(gain), -3.0103, 4)

    def test_gains_are_the_cosine_and_the_sine_of_one_angle_and_keep_the_power(self):
        for pan, left, right in ((-0.6, 0.951057, 0.309017), (-0.2, 0.809017, 0.587785), (0.5, 0.382683, 0.923880),
                                 (0.7, 0.233445, 0.972370)):
            got = pan_gains(pan)
            self.assertAlmostEqual(got[0], left, 6, pan)
            self.assertAlmostEqual(got[1], right, 6, pan)
            self.assertAlmostEqual(got[0] ** 2 + got[1] ** 2, 1.0, 15, pan)


class BarRowsTest(unittest.TestCase):
    def test_bar_is_four_beats_of_the_rows_the_song_has_to_a_beat(self):
        self.assertEqual(bar_rows(SONG), 16)
        self.assertEqual(bar_rows({**SONG, "rows_per_beat": 4}), 16)
        self.assertEqual(bar_rows({**SONG, "rows_per_beat": 3}), 12)
        self.assertEqual(bar_rows({**SONG, "rows_per_beat": 6}), 24)


class KeylessTest(unittest.TestCase):
    def test_group_of_drum_channels_alone_is_keyless(self):
        self.assertIs(keyless(SONG, ["kick", "hat"]), True)
        self.assertIs(keyless(SONG, ["hat"]), True)

    def test_group_with_a_channel_that_is_no_drum_channel_is_not(self):
        self.assertIs(keyless(SONG, ["low"]), False)
        self.assertIs(keyless(SONG, ["kick", "low", "hat"]), False)
        song = copy.deepcopy(SONG)
        song["channels"]["hat"]["drums"] = False
        self.assertIs(keyless(song, ["kick", "hat"]), False)
        # A channel the song does not have is none of its drum channels.
        self.assertIs(keyless(SONG, ["kick", "gong"]), False)


class RenderGroupsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = render_groups(SONG, RATE)

    def mixes(self, name: str) -> dict:
        """The mixes of one group of the song, by key."""
        return self.out["groups"][name]["mixes"]

    def test_gives_the_loop_samples_the_rows_and_every_group_in_the_order_written(self):
        self.assertEqual(sorted(self.out), ["groups", "loop_samples", "rows"])
        self.assertEqual(list(self.out["groups"]), ["top", "base", "beat"])
        for name, group in self.out["groups"].items():
            self.assertEqual(sorted(group), ["keyless", "mixes"], name)
            for key, sides in group["mixes"].items():
                self.assertIsInstance(sides, tuple, (name, key))
                self.assertEqual([(side.dtype, side.shape) for side in sides], [(np.float32, (6400,))] * 2, name)
        # The loop and the rows are those of the song's stems.
        stems = render_stems(SONG, RATE)
        self.assertEqual((self.out["loop_samples"], self.out["rows"]), (6400, 16))
        self.assertEqual((self.out["loop_samples"], self.out["rows"]), (stems["loop_samples"], stems["rows"]))

    def test_mix_is_the_stems_of_its_group_each_shared_between_the_sides_by_its_pan(self):
        for name, channels in GROUPS.items():
            same(self, self.mixes(name)[0], by_hand(SONG, channels), name)
        # The channel with no pan is in the middle.
        left, right = self.mixes("base")[0]
        self.assertTrue(np.array_equal(left, right))

    def test_stems_go_in_one_at_a_time_in_the_order_the_group_names_them(self):
        turned = render_groups(layered(groups={"top": ["mid", "air"]}), RATE)["groups"]["top"]["mixes"][0]
        same(self, turned, by_hand(SONG, ["mid", "air"]))
        # A side is rounded to float32 after each stem, so the other order ends a last digit away here and there.
        for side, other in zip(turned, self.mixes("top")[0]):
            self.assertFalse(np.array_equal(side, other))
            np.testing.assert_allclose(side.astype(np.float64), other, rtol=0, atol=1e-7)

    def test_tuned_group_has_a_mix_in_every_key_made_from_the_song_moved_up(self):
        higher = up(SONG, 2)
        for name in ("top", "base"):
            self.assertIs(self.out["groups"][name]["keyless"], False, name)
            self.assertEqual(list(self.mixes(name)), [0, 2], name)
            same(self, self.mixes(name)[2], by_hand(higher, GROUPS[name]), name)
            self.assertFalse(np.array_equal(self.mixes(name)[2][0], self.mixes(name)[0][0]), name)

    def test_group_of_drums_alone_is_mixed_once_and_sits_under_the_first_key(self):
        self.assertIs(self.out["groups"]["beat"]["keyless"], True)
        self.assertEqual(list(self.mixes("beat")), [0])
        other = render_groups(layered(groups=GROUPS, keys=[2, 0]), RATE)["groups"]
        self.assertEqual(list(other["beat"]["mixes"]), [2])
        same(self, other["beat"]["mixes"][2], self.mixes("beat")[0])
        # The tuned groups have their keys in the order of the list, and each key holds what it held.
        self.assertEqual(list(other["top"]["mixes"]), [2, 0])
        for key in (0, 2):
            same(self, other["top"]["mixes"][key], self.mixes("top")[key], key)

    def test_group_of_a_drum_channel_and_a_tuned_one_is_mixed_in_every_key(self):
        drive = render_groups(layered(groups={"drive": ["kick", "low"]}, keys=[0, 2]), RATE)["groups"]["drive"]
        self.assertIs(drive["keyless"], False)
        self.assertEqual(list(drive["mixes"]), [0, 2])
        same(self, drive["mixes"][0], by_hand(SONG, ["kick", "low"]))
        same(self, drive["mixes"][2], by_hand(up(SONG, 2), ["kick", "low"]))

    def test_keys_left_out_are_the_one_key_the_song_is_written_in(self):
        out = render_groups(layered(groups=GROUPS), RATE)
        for name, group in out["groups"].items():
            self.assertEqual(list(group["mixes"]), [0], name)
            same(self, group["mixes"][0], self.mixes(name)[0], name)

    def test_song_need_not_be_heard_in_the_key_it_is_written_in(self):
        out = render_groups(layered(groups=GROUPS, keys=[-3]), RATE)
        self.assertEqual((out["loop_samples"], out["rows"]), (6400, 16))
        self.assertEqual([list(group["mixes"]) for group in out["groups"].values()], [[-3], [-3], [-3]])
        same(self, out["groups"]["top"]["mixes"][-3], by_hand(up(SONG, -3), GROUPS["top"]))
        same(self, out["groups"]["beat"]["mixes"][-3], self.mixes("beat")[0])

    def test_rate_left_out_is_48000(self):
        # Half a row of one tick is 400 samples at 48000 Hz, and 367.5 at 44100.
        song = {"row_ticks": 1, "instruments": {"hiss": {"wave": "noise", "release": 0}},
                "channels": {"voice": {"inst": "hiss", "pan": 0.4}}, "patterns": {"p": {"voice": "c32"}},
                "order": ["p"], "layers": {"groups": {"all": ["voice"]}}}
        out = render_groups(song)
        self.assertEqual(out["loop_samples"], 400)
        same(self, out["groups"]["all"]["mixes"][0], render_groups(song, 48000)["groups"]["all"]["mixes"][0])
        self.assertEqual(render_groups(song, 44100)["loop_samples"], 368)

    def test_song_is_left_as_it_was(self):
        song = copy.deepcopy(SONG)
        render_groups(song, RATE)
        self.assertEqual(song, SONG)


class KnownSamplesTest(unittest.TestCase):
    def check(self, out: dict, spots: tuple, known: dict) -> None:
        """Holds the samples at those spots of each side of each mix: known is {(group, key): (left, right)}."""
        self.assertEqual([(name, key) for name, group in out["groups"].items() for key in group["mixes"]], list(known))
        for (name, key), sides in known.items():
            for got, wanted in zip(out["groups"][name]["mixes"][key], sides):
                np.testing.assert_allclose(got[list(spots)], wanted, rtol=0, atol=1e-6, err_msg=f"{name} {key}")

    def test_three_groups_in_two_keys(self):
        # The second pattern starts at sample 3200. A loop opens on what was still sounding at its end.
        out = render_groups(SONG, RATE)
        self.assertEqual(out["loop_samples"], 6400)
        base = (0.016981, 0.028226, 0.038457, -0.051002, -0.004377, 0.077114, -0.036349, -0.021517, -0.188227,
                0.053048)
        higher = (-0.003161, 0.008559, 0.019253, -0.142098, 0.002165, 0.178885, 0.130151, 0.134848, -0.074423,
                  0.006017)
        self.check(out, (0, 1, 2, 150, 1000, 2401, 3200, 3201, 4801, 6399), {
            ("top", 0): ((-0.011750, 0.033208, 0.055619, 0.390254, -0.150720, 0.220170, -0.043101, 0.004866,
                          0.035748, -0.022050),
                         (-0.003818, 0.002056, 0.000077, 0.174012, -0.042831, 0.148411, -0.010873, 0.004639,
                          0.011615, -0.007165)),
            ("top", 2): ((-0.011478, 0.033488, 0.055930, -0.139067, 0.096422, -0.007256, -0.036186, 0.011618,
                          0.035831, -0.021772),
                         (-0.003730, 0.002202, 0.000375, -0.152911, 0.035939, -0.070689, -0.016867, -0.001214,
                          0.011642, -0.007074)),
            ("base", 0): (base, base),
            ("base", 2): (higher, higher),
            ("beat", 0): ((0.017399, 0.033644, 0.048434, -0.059184, -0.003893, 0.005796, 0.020809, 0.039810,
                           0.002242, -0.000412),
                          (0.012107, 0.023923, 0.034681, -0.043014, -0.009399, 0.013993, 0.019701, 0.038184,
                           0.001108, -0.000846)),
        })

    def test_first_pattern_at_44100_and_48000_hz(self):
        song = {**SONG, "order": ["a"]}
        out = render_groups(song, 44100)
        self.assertEqual(out["loop_samples"], 17640)
        base = (-0.048970, 0.096836, -0.179763, 0.135474, -0.052032)
        higher = (0.124980, 0.009096, 0.001429, 0.177897, 0.123933)
        self.check(out, (0, 100, 1000, 6000, 17639), {
            ("top", 0): ((-0.051615, -0.149602, -0.176396, 0.441778, -0.056809),
                         (-0.013650, -0.052903, -0.069732, 0.186010, -0.027254)),
            ("top", 2): ((-0.044651, 0.051786, -0.152942, -0.109154, -0.036163),
                         (-0.019510, 0.015311, 0.055424, 0.059658, 0.023725)),
            ("base", 0): (base, base),
            ("base", 2): (higher, higher),
            ("beat", 0): ((0.003404, 0.154768, -0.051150, 0.002995, 0.000021),
                          (0.001960, 0.112111, -0.037170, 0.007230, -0.000501)),
        })
        out = render_groups(song)
        self.assertEqual(out["loop_samples"], 19200)
        base = (-0.049253, 0.086334, -0.128901, -0.003873, -0.052068)
        higher = (0.124898, 0.025505, -0.154365, 0.002540, 0.123935)
        self.check(out, (0, 100, 1000, 6000, 19199), {
            ("top", 0): ((-0.051657, -0.151474, -0.178271, -0.153317, -0.056682),
                         (-0.013664, -0.059536, -0.178699, -0.047581, -0.027212)),
            ("top", 2): ((-0.044711, 0.056338, -0.156774, 0.098605, -0.036057),
                         (-0.019525, 0.009648, -0.029662, 0.032229, 0.023769)),
            ("base", 0): (base, base),
            ("base", 2): (higher, higher),
            ("beat", 0): ((0.003155, 0.151508, -0.062164, -0.003246, 0.000045),
                          (0.001815, 0.109754, -0.045174, -0.007836, -0.000448)),
        })


class ErrorTest(unittest.TestCase):
    def refused(self, song: dict, rate: object = RATE) -> str:
        """What the error says when the groups of a song are refused, which is before any note is played."""
        with mock.patch("rowsynth.song.render_note") as played:
            with self.assertRaises(RowSynthError, msg=repr((song.get("layers"), rate))) as caught:
                render_groups(song, rate)
        played.assert_not_called()
        return str(caught.exception)

    def test_refuses_a_song_with_no_layers(self):
        song = {key: value for key, value in SONG.items() if key != "layers"}
        self.assertEqual(self.refused(song), "the song has no layers, so it has no groups to mix")

    def test_refuses_layers_with_no_groups(self):
        for song in (layered(), layered(groups={}), layered(keys=[0, 2])):
            self.assertEqual(self.refused(song), "layers: no groups, so there is nothing to mix")

    def test_refuses_a_group_that_names_a_channel_the_song_lacks(self):
        self.assertEqual(self.refused(layered(groups={**GROUPS, "base": ["low", "pad"]})),
                         "group 'base': no channel called 'pad'")

    def test_refuses_a_group_whose_channel_plays_in_no_pattern_and_so_has_no_stem(self):
        self.assertEqual(self.refused({**SONG, "order": ["b", "b"]}),
                         "group 'top': channel 'mid' plays in no pattern of the order, so it has no stem")

    def test_refuses_a_rate_that_makes_no_sense(self):
        for rate in (0, -8000, 8000.5, "8000", None, True, float("inf")):
            self.assertTrue(self.refused(SONG, rate).startswith("rate: "), rate)


if __name__ == "__main__":
    unittest.main()
