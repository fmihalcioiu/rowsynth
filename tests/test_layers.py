"""Tests for a song's stems mixed to stereo: the groups of a layered song in every key, its versions, and an intro."""
from __future__ import annotations

import copy
import math
import unittest
from unittest import mock

import numpy as np

from rowsynth.errors import RowSynthError
from rowsynth.layers import (bar_rows, group_mix, keyless, mix_version, pan_gains, render_groups, render_intro,
                             shared_scale)
from rowsynth.song import render_stems, up

# With three ticks to a row, a row is 400 samples at 8000 Hz.
RATE = 8000

# Two groups of tuned channels and one of drums alone.
GROUPS = {"top": ["air", "mid"], "base": ["low"], "beat": ["kick", "hat"]}

# Every group at once, and a version without the top one that names the drums first.
VERSIONS = {"full": ["top", "base", "beat"], "thin": ["beat", "base"]}

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
    "layers": {"groups": GROUPS, "versions": VERSIONS, "keys": [0, 2]},
}


def layered(**layers: object) -> dict:
    """The song with other layers."""
    return {**SONG, "layers": layers}


def mix(left: list, right: list) -> tuple[np.ndarray, np.ndarray]:
    """The two sides of a mix written by hand."""
    return np.array(left, dtype=np.float32), np.array(right, dtype=np.float32)


def rendered(keys: list, groups: dict) -> dict:
    """What `render_groups` gives, for groups written by hand with two samples a side."""
    return {"loop_samples": 2, "rows": 1.0, "keys": keys, "groups": groups}


def quiet() -> dict:
    """A tuned group in two keys and a keyless one, written by hand, with no sample over a quarter."""
    return rendered([0, 2], {
        "tune": {"keyless": False, "mixes": {0: mix([0.25, -0.125], [0.0, 0.125]),
                                             2: mix([-0.25, 0.0], [0.125, 0.25])}},
        "drum": {"keyless": True, "mixes": {0: mix([0.125, 0.25], [-0.25, 0.125])}},
    })


def versions(**listed: list) -> dict:
    """As much of a song as its versions take: {version: [groups]}."""
    return {"layers": {"versions": listed}}


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

    def test_gives_the_loop_samples_the_rows_the_keys_and_every_group_in_the_order_written(self):
        self.assertEqual(sorted(self.out), ["groups", "keys", "loop_samples", "rows"])
        self.assertEqual(self.out["keys"], [0, 2])
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
        turned = render_groups(layered(groups=GROUPS, keys=[2, 0]), RATE)
        self.assertEqual(turned["keys"], [2, 0])
        other = turned["groups"]
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
        self.assertEqual(out["keys"], [0])
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
        out = render_groups(song, RATE)
        # The keys that come back are a list of their own.
        out["keys"].append(5)
        self.assertEqual(song, SONG)


class GroupMixTest(unittest.TestCase):
    def test_tuned_group_gives_its_mix_in_the_key(self):
        tune = quiet()["groups"]["tune"]
        for key in (0, 2):
            self.assertIs(group_mix(tune, key), tune["mixes"][key])

    def test_keyless_group_gives_its_one_mix_whatever_the_key(self):
        drum = quiet()["groups"]["drum"]
        for key in (0, 2, 7):
            self.assertIs(group_mix(drum, key), drum["mixes"][0])
        # The one mix is found under whichever key it sits.
        other = {"keyless": True, "mixes": {2: mix([0.5, 0.0], [0.0, 0.5])}}
        self.assertIs(group_mix(other, 0), other["mixes"][2])

    def test_refuses_a_key_a_tuned_group_has_no_mix_in(self):
        with self.assertRaises(RowSynthError) as caught:
            group_mix(quiet()["groups"]["tune"], 7)
        self.assertEqual(str(caught.exception), "key 7: the group has no mix in it, only in [0, 2]")


class MixVersionTest(unittest.TestCase):
    def refused(self, out: dict, song: dict, version: str, key: int) -> str:
        """What the error says when a version of a song in a key is refused."""
        with self.assertRaises(RowSynthError, msg=repr((version, key))) as caught:
            mix_version(out, song, version, key)
        return str(caught.exception)

    def test_version_is_the_sum_of_its_groups_in_the_key(self):
        out, song = quiet(), versions(both=["tune", "drum"], bare=["drum"])
        for key, wanted in ((0, mix([0.375, 0.125], [-0.25, 0.25])), (2, mix([-0.125, 0.25], [-0.125, 0.375]))):
            got = mix_version(out, song, "both", key)
            self.assertIsInstance(got, tuple, key)
            self.assertEqual([side.dtype for side in got], [np.float32] * 2, key)
            same(self, got, wanted, key)

    def test_keyless_group_goes_in_the_same_in_every_key_and_stays_as_it_was(self):
        out, song = quiet(), versions(both=["tune", "drum"], bare=["drum"])
        drum = out["groups"]["drum"]["mixes"][0]
        for key in (0, 2):
            got = mix_version(out, song, "bare", key)
            same(self, got, drum, key)
            # The sides are buffers of their own: what is done to them is not done to the group.
            for side, kept in zip(got, drum):
                self.assertFalse(np.shares_memory(side, kept), key)
                side *= 2
        same(self, drum, mix([0.125, 0.25], [-0.25, 0.125]))

    def test_groups_go_in_one_at_a_time_in_the_order_the_version_names_them(self):
        # Half the last digit a float32 has at 1 is lost when it is added to 1. Two halves added first make a
        # whole digit, which is kept.
        half = 2.0 ** -24
        out = rendered([0], {name: {"keyless": False, "mixes": {0: mix([value, 0.0], [-value, 0.0])}}
                             for name, value in (("one", 1.0), ("speck", half), ("grain", half))})
        song = versions(down=["one", "speck", "grain"], back=["grain", "speck", "one"])
        left, right = mix_version(out, song, "down", 0)
        self.assertEqual((float(left[0]), float(right[0])), (1.0, -1.0))
        left, right = mix_version(out, song, "back", 0)
        self.assertEqual((float(left[0]), float(right[0])), (1.0 + 2 * half, -1.0 - 2 * half))

    def test_refuses_a_version_the_song_does_not_have(self):
        out = quiet()
        for song in (versions(both=["tune", "drum"]), versions(), {"layers": {}}, {}):
            self.assertEqual(self.refused(out, song, "bare", 0), "layers.versions: no version called 'bare'")

    def test_refuses_a_version_that_names_a_group_that_was_not_rendered(self):
        self.assertEqual(self.refused(quiet(), versions(wide=["tune", "pad", "drum"]), "wide", 0),
                         "version 'wide': group 'pad' was not rendered")

    def test_refuses_a_key_that_is_not_one_of_the_rendered_keys(self):
        # A version of keyless groups alone has no more keys than the others.
        song = versions(both=["tune", "drum"], bare=["drum"])
        for version in ("both", "bare"):
            self.assertEqual(self.refused(quiet(), song, version, 7),
                             f"version '{version}': key 7 is not one of the rendered keys, [0, 2]")


class SharedScaleTest(unittest.TestCase):
    def test_scale_brings_the_loudest_sample_of_all_the_mixes_to_1_decibel_under_full_scale(self):
        # A quarter is the loudest of them: 0.891251 is 1 dB under full scale, and four times that is the scale.
        self.assertAlmostEqual(shared_scale(quiet()), 3.565004, 6)
        # The loudest sample is looked for on both sides of every mix, and counts whichever way it points.
        for name, key, side in (("tune", 0, 0), ("tune", 2, 1), ("drum", 0, 0), ("drum", 0, 1)):
            out = quiet()
            out["groups"][name]["mixes"][key][side][1] = -0.5
            scale = shared_scale(out)
            self.assertAlmostEqual(scale, 1.782502, 6, (name, key, side))
            self.assertAlmostEqual(20 * math.log10(scale * 0.5), -1.0, 12, (name, key, side))

    def test_peak_db_is_the_level_the_loudest_sample_is_brought_to(self):
        self.assertEqual(shared_scale(quiet(), 0), 4.0)
        self.assertAlmostEqual(shared_scale(quiet(), -20.0), 0.4, 15)
        self.assertAlmostEqual(shared_scale(quiet(), -3.0), 2.831783, 6)

    def test_groups_that_are_all_silence_have_a_scale_of_1(self):
        silence = {"keyless": False, "mixes": {0: mix([0.0, 0.0], [0.0, 0.0]), 2: mix([0.0, 0.0], [0.0, 0.0])}}
        self.assertEqual(shared_scale(rendered([0, 2], {"hush": silence})), 1.0)
        self.assertEqual(shared_scale(rendered([0, 2], {"hush": silence}), -6.0), 1.0)


def group_mixes(out: dict) -> dict:
    """Every mix of the groups of a song: {(group, key): (left, right)}."""
    return {(name, key): sides for name, group in out["groups"].items() for key, sides in group["mixes"].items()}


def version_mixes(out: dict, song: dict) -> dict:
    """Every version of a song in every one of its keys: {(version, key): (left, right)}."""
    return {(version, key): mix_version(out, song, version, key)
            for version in song["layers"]["versions"] for key in out["keys"]}


class KnownSamplesTest(unittest.TestCase):
    def check(self, mixes: dict, spots: tuple, known: dict) -> None:
        """Holds the samples at those spots of each side of each mix: known is {(name, key): (left, right)}."""
        self.assertEqual(list(mixes), list(known))
        for about, sides in known.items():
            for got, wanted in zip(mixes[about], sides):
                np.testing.assert_allclose(got[list(spots)], wanted, rtol=0, atol=1e-6, err_msg=repr(about))

    def test_three_groups_and_two_versions_in_two_keys(self):
        # The second pattern starts at sample 3200. A loop opens on what was still sounding at its end.
        out = render_groups(SONG, RATE)
        self.assertEqual(out["loop_samples"], 6400)
        spots = (0, 1, 2, 150, 1000, 2401, 3200, 3201, 4801, 6399)
        base = (0.016981, 0.028226, 0.038457, -0.051002, -0.004377, 0.077114, -0.036349, -0.021517, -0.188227,
                0.053048)
        higher = (-0.003161, 0.008559, 0.019253, -0.142098, 0.002165, 0.178885, 0.130151, 0.134848, -0.074423,
                  0.006017)
        self.check(group_mixes(out), spots, {
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
        self.check(version_mixes(out, SONG), spots, {
            ("full", 0): ((0.022629, 0.095078, 0.142510, 0.280069, -0.158990, 0.303080, -0.058640, 0.023159,
                           -0.150237, 0.030586),
                          (0.025270, 0.054205, 0.073214, 0.079996, -0.056607, 0.239518, -0.027521, 0.021306,
                           -0.175504, 0.045037)),
            ("full", 2): ((0.002759, 0.075691, 0.123617, -0.340349, 0.094695, 0.177425, 0.114775, 0.186276,
                           -0.036350, -0.016166),
                          (0.005217, 0.034684, 0.054309, -0.338023, 0.028705, 0.122188, 0.132986, 0.171819,
                           -0.061673, -0.001903)),
            ("thin", 0): ((0.034380, 0.061870, 0.086891, -0.110185, -0.008270, 0.082910, -0.015539, 0.018293,
                           -0.185985, 0.052636),
                          (0.029088, 0.052149, 0.073138, -0.094016, -0.013776, 0.091107, -0.016648, 0.016667,
                           -0.187119, 0.052202)),
            ("thin", 2): ((0.014238, 0.042203, 0.067687, -0.201282, -0.001728, 0.184681, 0.150961, 0.174658,
                           -0.072181, 0.005605),
                          (0.008946, 0.032482, 0.053934, -0.185113, -0.007233, 0.192878, 0.149852, 0.173033,
                           -0.073315, 0.005172)),
        })
        self.assertAlmostEqual(shared_scale(out), 1.456895, 6)

    def test_first_pattern_at_44100_and_48000_hz(self):
        song = {**SONG, "order": ["a"]}
        out = render_groups(song, 44100)
        self.assertEqual(out["loop_samples"], 17640)
        spots = (0, 100, 1000, 6000, 17639)
        base = (-0.048970, 0.096836, -0.179763, 0.135474, -0.052032)
        higher = (0.124980, 0.009096, 0.001429, 0.177897, 0.123933)
        self.check(group_mixes(out), spots, {
            ("top", 0): ((-0.051615, -0.149602, -0.176396, 0.441778, -0.056809),
                         (-0.013650, -0.052903, -0.069732, 0.186010, -0.027254)),
            ("top", 2): ((-0.044651, 0.051786, -0.152942, -0.109154, -0.036163),
                         (-0.019510, 0.015311, 0.055424, 0.059658, 0.023725)),
            ("base", 0): (base, base),
            ("base", 2): (higher, higher),
            ("beat", 0): ((0.003404, 0.154768, -0.051150, 0.002995, 0.000021),
                          (0.001960, 0.112111, -0.037170, 0.007230, -0.000501)),
        })
        self.check(version_mixes(out, song), spots, {
            ("full", 0): ((-0.097181, 0.102002, -0.407310, 0.580247, -0.108820),
                          (-0.060661, 0.156043, -0.286666, 0.328714, -0.079786)),
            ("full", 2): ((0.083733, 0.215649, -0.202663, 0.071737, 0.087790),
                          (0.107430, 0.136517, 0.019683, 0.244785, 0.147157)),
            ("thin", 0): ((-0.045567, 0.251604, -0.230914, 0.138469, -0.052011),
                          (-0.047011, 0.208947, -0.216933, 0.142704, -0.052532)),
            ("thin", 2): ((0.128384, 0.163863, -0.049722, 0.180892, 0.123953),
                          (0.126940, 0.121206, -0.035741, 0.185127, 0.123432)),
        })
        self.assertAlmostEqual(shared_scale(out), 1.388617, 6)
        out = render_groups(song)
        self.assertEqual(out["loop_samples"], 19200)
        spots = (0, 100, 1000, 6000, 19199)
        base = (-0.049253, 0.086334, -0.128901, -0.003873, -0.052068)
        higher = (0.124898, 0.025505, -0.154365, 0.002540, 0.123935)
        self.check(group_mixes(out), spots, {
            ("top", 0): ((-0.051657, -0.151474, -0.178271, -0.153317, -0.056682),
                         (-0.013664, -0.059536, -0.178699, -0.047581, -0.027212)),
            ("top", 2): ((-0.044711, 0.056338, -0.156774, 0.098605, -0.036057),
                         (-0.019525, 0.009648, -0.029662, 0.032229, 0.023769)),
            ("base", 0): (base, base),
            ("base", 2): (higher, higher),
            ("beat", 0): ((0.003155, 0.151508, -0.062164, -0.003246, 0.000045),
                          (0.001815, 0.109754, -0.045174, -0.007836, -0.000448)),
        })
        self.check(version_mixes(out, song), spots, {
            ("full", 0): ((-0.097755, 0.086368, -0.369337, -0.160436, -0.108705),
                          (-0.061103, 0.136552, -0.352774, -0.059290, -0.079727)),
            ("full", 2): ((0.083342, 0.233351, -0.373304, 0.097899, 0.087922),
                          (0.107188, 0.144908, -0.229202, 0.026932, 0.147256)),
            ("thin", 0): ((-0.046098, 0.237842, -0.191065, -0.007119, -0.052024),
                          (-0.047439, 0.196088, -0.174075, -0.011709, -0.052516)),
            ("thin", 2): ((0.128053, 0.177013, -0.216529, -0.000706, 0.123979),
                          (0.126713, 0.135260, -0.199539, -0.005296, 0.123487)),
        })
        self.assertAlmostEqual(shared_scale(out), 1.383735, 6)


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

    def test_refuses_layers_with_an_empty_list_of_keys(self):
        self.assertEqual(self.refused(layered(groups=GROUPS, keys=[])),
                         "layers.keys: no keys, so there is no key to mix the groups in")

    def test_refuses_a_group_that_names_a_channel_the_song_lacks(self):
        self.assertEqual(self.refused(layered(groups={**GROUPS, "base": ["low", "pad"]})),
                         "group 'base': no channel called 'pad'")

    def test_refuses_a_group_whose_channel_plays_in_no_pattern_and_so_has_no_stem(self):
        self.assertEqual(self.refused({**SONG, "order": ["b", "b"]}),
                         "group 'top': channel 'mid' plays in no pattern of the order, so it has no stem")

    def test_refuses_a_rate_that_makes_no_sense(self):
        for rate in (0, -8000, 8000.5, "8000", None, True, float("inf")):
            self.assertTrue(self.refused(SONG, rate).startswith("rate: "), rate)


# A song with an intro of two patterns in front of its loop of one. The intro gives neither a tail nor a fade.
# With six ticks to a row, a row is 800 samples at 8000 Hz and a bar is 12800, which is 1.6 seconds. The intro
# opens on a rest of one row. It ends on a bell that is held to the last row and takes 96 ticks more to die away,
# which is the length of a bar. The bell is to the left and the reed to the right, and the drum channel has no
# text in the intro.
LED_IN = {
    "row_ticks": 6,
    "instruments": {**SONG["instruments"], "bell": {"wave": "tri", "vol": [15, 12, 10], "gate": 1, "release": 96}},
    "drums": SONG["drums"],
    "channels": {
        "chime": {"inst": "bell", "pan": -0.5},
        "tune": {"inst": "reed", "gain": 0.8, "pan": 0.6, "echo": {"rows": 1.5, "gain": 0.4}},
        "kick": {"inst": "thump", "drums": True},
    },
    "patterns": {
        "rise": {"tune": "r16 l8 e g b r16"},
        "land": {"chime": "o5 c4 g4", "tune": "o3 c2"},
        "round": {"tune": "l8 c e g e", "kick": "k4 k4"},
    },
    "order": ["round"],
    "intro": {"order": ["rise", "land"]},
}


def led_in(**intro: object) -> dict:
    """The song with other settings for its intro, which keeps its two patterns unless it is given others."""
    return {**LED_IN, "intro": {"order": ["rise", "land"], **intro}}


def rung_out() -> tuple[np.ndarray, np.ndarray]:
    """The two sides of the song's intro before they are cut and faded, made by hand from stems.

    The two patterns are played with a bar of rests on every channel after them. The stems go in in the order
    the channels first have text in: the reed, the bell, and the drum, which has its rest and nothing else.
    """
    rests = {name: "r1" for name in LED_IN["channels"]}
    played = {**LED_IN, "patterns": {**LED_IN["patterns"], "rests": rests}, "order": ["rise", "land", "rests"]}
    return by_hand(played, ["tune", "chime", "kick"])


def sides(out: dict) -> tuple[np.ndarray, np.ndarray]:
    """The left side and the right of what `render_intro` gives."""
    return out["left"], out["right"]


def ends(out: dict) -> tuple[int, int]:
    """Where the loop is to begin and where the sides end, in what `render_intro` gives."""
    return out["walk_samples"], out["in_samples"]


class RenderIntroTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.song = copy.deepcopy(LED_IN)
        cls.out = render_intro(cls.song, RATE)
        cls.whole = rung_out()

    def test_gives_the_two_sides_and_where_the_loop_begins_and_where_the_sides_end(self):
        self.assertEqual(sorted(self.out), ["in_samples", "left", "right", "walk_samples"])
        # The two patterns have 16 rows, and the tail that was left out is a second and a half.
        self.assertEqual(ends(self.out), (12800, 24800))
        self.assertEqual([(side.dtype, side.shape) for side in sides(self.out)], [(np.float32, (24800,))] * 2)
        self.assertEqual(self.song, LED_IN)

    def test_sides_are_the_stems_of_the_patterns_and_a_bar_of_rests_each_shared_by_its_pan(self):
        # As far as the fade, which was left out and is the last twentieth of a second: 400 samples.
        same(self, [side[:24400] for side in sides(self.out)], [hand[:24400] for hand in self.whole])

    def test_last_notes_ring_out_into_the_bar_of_rests_and_are_not_added_back_on_at_the_start(self):
        for side in sides(self.out):
            # The bell is heard after the patterns end. The row of rest the intro opens on has none of it: all
            # there is in that row is what the high-pass of a stem still holds from the stem's end.
            self.assertGreater(float(np.abs(side[12800:24400]).max()), 0.1)
            self.assertLess(float(np.abs(side[:800]).max()), 1e-4)
        # Stems of the two patterns alone are a loop of them, and there the bell dies away over the start.
        for side in by_hand({**LED_IN, "order": ["rise", "land"]}, ["tune", "chime"]):
            self.assertGreater(float(np.abs(side[:800]).max()), 0.1)

    def test_end_of_the_tail_falls_away_evenly_to_nothing(self):
        # The last sample keeps nothing of what it had, the one before it 1 part in 400, and the first 399.
        kept = np.arange(399, -1, -1) / 400
        for side, hand in zip(sides(self.out), self.whole):
            self.assertGreater(float(np.abs(hand[24400:24800]).max()), 0.005)
            self.assertTrue(np.array_equal(side[24400:], (hand[24400:24800] * kept).astype(np.float32)))
            self.assertEqual(float(side[-1]), 0.0)

    def test_tail_and_fade_are_seconds_that_come_to_whole_samples_with_a_half_rounded_up(self):
        # A 128th of a second is 62.5 samples. A fade may be as long as the tail.
        out = render_intro(led_in(tail=0.0078125, fade=0.0078125), RATE)
        self.assertEqual(ends(out), (12800, 12863))
        kept = np.arange(62, -1, -1) / 63
        for side, hand in zip(sides(out), self.whole):
            self.assertTrue(np.array_equal(side[:12800], hand[:12800]))
            self.assertGreater(abs(float(hand[12800])), 0.05)
            self.assertTrue(np.array_equal(side[12800:], (hand[12800:12863] * kept).astype(np.float32)))
        # With no tail the sides end where the loop is to begin, and with no fade they end as loud as they are.
        out = render_intro(led_in(tail=0, fade=0), RATE)
        self.assertEqual(ends(out), (12800, 12800))
        same(self, sides(out), [hand[:12800] for hand in self.whole])

    def test_loop_begins_at_the_share_of_the_stems_that_the_patterns_have_of_the_rows(self):
        # With one tick to a row the 32 rows of the stems are 4266.67 samples, which makes 4267. The patterns
        # have half of the rows, and half of 4267 is 2133.5, which makes 2134. Their 16 rows alone are 2133.
        self.assertEqual(ends(render_intro({**led_in(tail=0.25), "row_ticks": 1}, RATE)), (2134, 4134))
        # With two ticks it is half of 8533. A half goes up to an odd number as it does to an even one.
        self.assertEqual(ends(render_intro({**led_in(tail=0.25), "row_ticks": 2}, RATE)), (4267, 6267))

    def test_bar_of_rests_is_a_pattern_with_a_name_that_no_pattern_of_the_song_has(self):
        def bar_of(song: dict) -> tuple[str, dict]:
            """The name the bar of rests was given in the song the stems were made from, and the intro."""
            with mock.patch("rowsynth.layers.render_stems", wraps=render_stems) as stems:
                out = render_intro(song, RATE)
            played = stems.call_args[0][0]
            (name,) = set(played["patterns"]) - set(song["patterns"])
            self.assertEqual(played["order"], [*song["intro"]["order"], name])
            self.assertEqual(played["patterns"][name], {"chime": "r1", "tune": "r1", "kick": "r1"})
            return name, out

        first, _ = bar_of(LED_IN)
        # A song that has a pattern of that name keeps it, and the bar of rests takes another name.
        patterns = {**LED_IN["patterns"], first: LED_IN["patterns"]["rise"]}
        second, out = bar_of({**LED_IN, "patterns": patterns, "intro": {"order": [first, "land"]}})
        self.assertNotEqual(second, first)
        same(self, sides(out), sides(self.out))

    def test_known_samples_at_8000_44100_and_48000_hz(self):
        def check(out: dict, where: tuple, spots: tuple, left: tuple, right: tuple) -> None:
            self.assertEqual(ends(out), where)
            for side, wanted in zip(sides(out), (left, right)):
                self.assertEqual(len(side), where[1])
                np.testing.assert_allclose(side[list(spots)], wanted, rtol=0, atol=1e-6, err_msg=repr(where))

        # The first note starts a row in. The last 400 samples at 8000 Hz are the fade, 2205 at 44100, 2400 at 48000.
        check(self.out, (12800, 24800), (0, 800, 801, 5000, 12799, 12800, 20000, 24399, 24400, 24600, 24798, 24799),
              (0.000029, 0.002538, 0.014681, -0.012318, -0.349850, -0.214999, -0.025072, -0.005183, -0.013623,
               0.008783, -0.000024, 0.0),
              (0.000012, 0.007813, 0.045182, -0.037911, -0.144913, -0.089056, -0.010385, -0.002147, -0.005643,
               0.003638, -0.000010, 0.0))
        check(render_intro(LED_IN, 44100), (70560, 136710),
              (4410, 30000, 70560, 100000, 134504, 134505, 135600, 136708, 136709),
              (0.000482, -0.011122, -0.216458, 0.164440, -0.010899, -0.013781, 0.004389, -0.000006, 0.0),
              (0.001485, -0.034229, -0.089660, 0.068113, -0.004515, -0.005708, 0.001818, -0.000002, 0.0))
        # The rate left out is 48000.
        check(render_intro(LED_IN), (76800, 148800),
              (4800, 30000, 76800, 100000, 146399, 146400, 147600, 148798, 148799),
              (0.000444, -0.012452, -0.216454, -0.246805, -0.010903, -0.013792, 0.008907, -0.000006, 0.0),
              (0.001365, -0.038323, -0.089658, -0.102230, -0.004516, -0.005713, 0.003689, -0.000002, 0.0))


class IntroErrorTest(unittest.TestCase):
    def refused(self, song: dict) -> str:
        """What the error says when the intro of a song is refused, which is before any note is played."""
        with mock.patch("rowsynth.song.render_note") as played:
            with self.assertRaises(RowSynthError, msg=repr(song.get("intro"))) as caught:
                render_intro(song, RATE)
        played.assert_not_called()
        return str(caught.exception)

    def test_refuses_a_song_with_no_intro(self):
        self.assertEqual(self.refused(SONG), "the song has no intro, so there is nothing to play before its loop")

    def test_refuses_an_intro_with_no_patterns_in_its_order(self):
        for intro in ({}, {"order": []}, {"tail": 1.5, "fade": 0.05}):
            self.assertEqual(self.refused({**LED_IN, "intro": intro}),
                             "intro.order: no patterns, so the intro has nothing to play")

    def test_refuses_an_intro_that_names_a_pattern_the_song_lacks(self):
        self.assertEqual(self.refused(led_in(order=["rise", "fall", "land"])), "intro.order: no pattern called 'fall'")

    def test_refuses_a_tail_or_a_fade_that_is_below_0_or_is_no_number(self):
        for key in ("tail", "fade"):
            self.assertEqual(self.refused(led_in(**{key: -0.01})), f"intro.{key}: must be 0 or more, not -0.01")
            for value in ("1", None, True, float("inf")):
                self.assertEqual(self.refused(led_in(**{key: value})), f"intro.{key}: must be a number, not {value!r}")

    def test_refuses_a_fade_longer_than_the_tail(self):
        self.assertEqual(self.refused(led_in(tail=0.25, fade=0.3)),
                         "intro.fade: 0.3 seconds is longer than the tail, which is 0.25")
        # Either of them may be the one that was left out.
        self.assertEqual(self.refused(led_in(fade=2)), "intro.fade: 2 seconds is longer than the tail, which is 1.5")
        self.assertEqual(self.refused(led_in(tail=0.01)),
                         "intro.fade: 0.05 seconds is longer than the tail, which is 0.01")

    def test_refuses_a_tail_longer_than_the_bar_of_rests_and_takes_one_that_fills_it(self):
        out = render_intro(led_in(tail=1.6), RATE)
        self.assertEqual(ends(out), (12800, 25600))
        self.assertEqual([len(side) for side in sides(out)], [25600, 25600])
        # The stems have to be made before it is known how many samples the bar has, so notes are played here.
        for song, said in ((led_in(tail=1.6001), "1.6001 seconds is 12801 samples at a rate of 8000, and the silent "
                                                 "bar after the intro has 12800"),
                           ({**LED_IN, "row_ticks": 3}, "1.5 seconds is 12000 samples at a rate of 8000, and the "
                                                        "silent bar after the intro has 6400")):
            with self.assertRaises(RowSynthError) as caught:
                render_intro(song, RATE)
            self.assertEqual(str(caught.exception), f"intro.tail: {said}")


if __name__ == "__main__":
    unittest.main()
