"""Tests for a build: the songs of a project written as files, and the manifest that names them."""
from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from rowsynth.build import build, build_song, manifest_path, source_hash, stale
from rowsynth.errors import RowSynthError
from rowsynth.layers import group_mix, mix_version, render_groups, render_intro, shared_scale
from rowsynth.project import Project
from rowsynth.song import render_loop
from rowsynth.wav import load_wav

# With three ticks to a row, a row is 400 samples at 8000 Hz. Opus takes this rate.
RATE = 8000

# The tests that run ffmpeg are left out where there is none.
with_ffmpeg = unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is not on the path")

# One step of a 16-bit file. A sample read back from a file is within half a step of the sample that was written.
STEP = 1 / 32767

# Where the loudest sample of a file is brought to: 1 dB under full scale.
PEAK = 10 ** (-1 / 20)

INSTRUMENTS = {
    "pipe": {"vol": [13, 11, 9], "duty": 1, "gate": 0.7, "release": 3},
    "sub": {"wave": "tri", "vol": 12},
    "boom": {"wave": "noise", "vol": [15, 9, 4], "release": 0},
    "tsk": {"wave": "noise", "short": True, "vol": [9, 3], "release": 0},
}

# A tune to the left with an echo, a bass to the right, and drums in the middle. The song has a bass of its own
# over the project's. Its 16 rows are 6400 samples at 8000 Hz.
TUNE = {
    "row_ticks": 3,
    "gain": 0.4,
    "instruments": {"sub": {"wave": "saw", "vol": 10, "lp": 1500}},
    "drums": {"k": {"inst": "boom", "note": "d2"}, "h": {"inst": "tsk", "note": 92, "vel": 0.5}},
    "channels": {
        "lead": {"inst": "pipe", "pan": -0.7, "echo": {"rows": 1.5, "gain": 0.4}},
        "bass": {"inst": "sub", "gain": 0.9, "pan": 0.6},
        "kit": {"inst": "boom", "drums": True},
    },
    "patterns": {"a": {"lead": "l8 e g b > d", "bass": "o2 e4 b4", "kit": "k8 h8 k8 h8"},
                 "b": {"lead": "o5 c4 < a4", "bass": "o2 a2", "kit": "l8 k h h k"}},
    "order": ["a", "b"],
}

# One channel in the middle.
DRONE = {
    "row_ticks": 3,
    "instruments": {"hum": {"wave": "tri", "vol": 11, "release": 9}},
    "channels": {"low": {"inst": "hum"}},
    "patterns": {"p": {"low": "o3 c4 g4"}},
    "order": ["p"],
}

# Every channel rests, for four rows.
HUSH = {
    "row_ticks": 3,
    "channels": {"lead": {"inst": "pipe", "pan": -0.5}, "bass": {"inst": "sub", "pan": 0.5}},
    "patterns": {"p": {"lead": "r4", "bass": "r8 r8"}},
    "order": ["p"],
}

# Three channels of noise at full level, all to the left: their sum is close to three times full scale.
LOUD = {
    "row_ticks": 3,
    "gain": 1.5,
    "instruments": {"wall": {"wave": "noise", "gate": 1, "release": 0}},
    "channels": {name: {"inst": "wall", "pan": -1} for name in ("one", "two", "three")},
    "patterns": {"p": {"one": "c4", "two": "e4", "three": "g4"}},
    "order": ["p"],
}

PLAIN = {"tune": TUNE, "drone": DRONE, "hush": HUSH}

# The tune with a second voice over it and a second channel of drums, cut into three groups, the drums alone in
# one. It is heard as written and two semitones up, in two versions, and the fuller of them is mixed ahead in
# the higher key.
MARCH = {
    **TUNE,
    "channels": {**TUNE["channels"], "alto": {"inst": "pipe", "gain": 0.6, "pan": 0.3},
                 "hat": {"inst": "tsk", "drums": True, "pan": 0.4}},
    "patterns": {"a": {**TUNE["patterns"]["a"], "alto": "o4 b4 g4", "hat": "l8 r h r h"},
                 "b": {**TUNE["patterns"]["b"], "alto": "o4 e2", "hat": "l16 h h h h r4"}},
    "layers": {"groups": {"top": ["lead", "alto"], "low": ["bass"], "beat": ["kit", "hat"]},
               "versions": {"full": ["top", "low", "beat"], "thin": ["beat", "low"]},
               "keys": [0, 2],
               "premix": [{"version": "full", "key": 2}]},
}

# The files of the march's groups, and the one of its version that is mixed ahead.
GROUP_FILES = ["march-beat.wav", "march-low-0.wav", "march-low-2.wav", "march-top-0.wav", "march-top-2.wav"]
AHEAD = "march.full.2.wav"

# What only an intro plays: the lead, the bass and the first of the drums together, four times over. It is louder
# than the loop of the tune. With the alto on the note of the lead it is louder than any group of the march, and
# than its version that is mixed ahead.
STAMP = {"lead": "o4 l8 a a a a", "bass": "o2 l8 a a a a", "kit": "l8 k k k k"}
STAMP_OF_FOUR = {**STAMP, "alto": "o4 l8 a a a a"}


def project_of(**songs: dict) -> Project:
    """A project of those songs, in that order, with the instruments above."""
    return Project.from_dict({"instruments": INSTRUMENTS, "songs": songs})


def march(*left_out: str, **layers: object) -> dict:
    """The march with other layers: its own without the keys named first, and with those given by name over them."""
    kept = {key: value for key, value in MARCH["layers"].items() if key not in left_out}
    return {**MARCH, "layers": {**kept, **layers}}


def led_in(song: dict, stamp: dict | None = None) -> dict:
    """The tune or the march with an intro: one pattern of 8 rows, and half a second of tail after it.

    A bar of either is 0.8 seconds, and the 1.5 seconds of a tail that is left out do not fit in one. With
    `stamp` the intro is that pattern, which the loop does not play. With none it is the pattern "b" of the song
    itself, and is quieter than the loop and than the loudest of its groups.
    """
    if stamp is None:
        return {**song, "intro": {"order": ["b"], "tail": 0.5}}
    return {**song, "patterns": {**song["patterns"], "in": stamp}, "intro": {"order": ["in"], "tail": 0.5}}


def loudest(sides: list) -> float:
    """The loudest sample of the two sides of a loop or of a file."""
    return max(float(np.abs(side).max()) for side in sides)


class BuildCase(unittest.TestCase):
    """What the tests below share: a project of three plain songs, and a folder to build into that is not there yet."""

    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.out = Path(folder.name, "built")
        self.project = project_of(**PLAIN)

    def files(self, folder: Path | None = None) -> list[str]:
        """The names of the files in a folder, which is the one built into when none is given."""
        return sorted(path.name for path in (folder or self.out).iterdir())

    def built(self, song: dict = MARCH, rate: int = RATE, folder: Path | None = None, fmt: str = "wav") -> dict:
        """The entry of a song built under the name of the march, into the folder of the test when none is given."""
        return build_song(project_of(march=song), "march", folder or self.out, rate, fmt)

    def refused(self, call, *args: object, **keys: object) -> str:
        """What the error says when a call is refused, which is before any note is played."""
        with mock.patch("rowsynth.song.render_note") as played:
            with self.assertRaises(RowSynthError, msg=repr((args, keys))) as caught:
                call(*args, **keys)
        played.assert_not_called()
        return str(caught.exception)


class BuildSongTest(BuildCase):
    def test_file_is_the_loop_in_stereo_with_its_loudest_sample_1_decibel_under_full_scale(self):
        build_song(self.project, "tune", self.out, RATE)
        self.assertEqual(self.files(), ["tune.wav"])
        sides, rate = load_wav(self.out / "tune.wav")
        self.assertEqual((len(sides), rate), (2, RATE))
        loop = render_loop(self.project.song("tune"), RATE)
        whole = [loop["left"], loop["right"]]
        scale = PEAK / loudest(whole)
        for side, wanted in zip(sides, whole):
            self.assertEqual(len(side), 6400)
            np.testing.assert_allclose(side, wanted.astype(np.float64) * scale, rtol=0, atol=0.51 * STEP)
        self.assertAlmostEqual(loudest(sides), PEAK, delta=STEP)
        # The tune is to the left, and both sides are brought up by the same amount.
        self.assertLess(float(np.abs(sides[1]).max()), 0.75 * float(np.abs(sides[0]).max()))

    def test_entry_gives_the_loop_samples_the_gain_the_source_and_the_name_of_the_file(self):
        entry = build_song(self.project, "tune", self.out, RATE)
        self.assertEqual(list(entry), ["loop_samples", "gain", "source", "file"])
        self.assertEqual((entry["loop_samples"], entry["file"]), (6400, "tune.wav"))
        self.assertEqual(entry["source"], source_hash(self.project, "tune", RATE))
        # The gain has six decimal places, and the file times the gain is the song at its own level.
        self.assertEqual(entry["gain"], 0.993813)
        loop = render_loop(self.project.song("tune"), RATE)
        for side, wanted in zip(load_wav(self.out / "tune.wav")[0], (loop["left"], loop["right"])):
            np.testing.assert_allclose(side.astype(np.float64) * entry["gain"], wanted, rtol=0, atol=STEP)

    def test_song_that_goes_past_full_scale_is_brought_down_to_the_same_level(self):
        entry = build_song(project_of(loud=LOUD), "loud", self.out, RATE)
        self.assertEqual(entry["gain"], 3.33238)
        sides, _ = load_wav(self.out / "loud.wav")
        self.assertAlmostEqual(float(np.abs(sides[0]).max()), PEAK, delta=STEP)
        self.assertFalse(sides[1].any())

    def test_silent_song_is_written_as_it_is_with_a_gain_of_1(self):
        entry = build_song(self.project, "hush", self.out, RATE)
        self.assertEqual((entry["loop_samples"], entry["gain"], entry["file"]), (1600, 1.0, "hush.wav"))
        sides, _ = load_wav(self.out / "hush.wav")
        self.assertEqual([len(side) for side in sides], [1600, 1600])
        self.assertFalse(sides[0].any() or sides[1].any())

    def test_known_16_bit_values_at_44100_and_48000_hz(self):
        # The rate left out is 48000.
        cases = (({"rate": 44100}, 44100, 35280, 1.020406, (0, 1, 500, 4410, 17640, 17641, 30000, 35279),
                  (-641, 4265, -2963, 2526, -2818, -2363, -4908, -811),
                  (-693, 563, 4550, -2525, 1136, 1328, 2800, 1329), 13859),
                 ({}, 48000, 38400, 1.021952, (0, 1, 500, 4800, 19200, 19201, 30000, 38399),
                  (1584, 4951, -3969, 2502, -2824, -2406, 5185, -806),
                  (-164, 717, 4167, -2532, 1125, 1303, 4376, 1328), 13886))
        for keys, rate, loop_samples, gain, spots, left, right, right_peak in cases:
            entry = build_song(self.project, "tune", self.out, **keys)
            self.assertEqual((entry["loop_samples"], entry["gain"]), (loop_samples, gain), rate)
            sides, read_rate = load_wav(self.out / "tune.wav")
            self.assertEqual(read_rate, rate)
            values = [np.rint(side.astype(np.float64) * 32767).astype(int) for side in sides]
            self.assertEqual([len(side) for side in values], [loop_samples] * 2, rate)
            self.assertEqual(values[0][list(spots)].tolist(), list(left), rate)
            self.assertEqual(values[1][list(spots)].tolist(), list(right), rate)
            # The loudest sample is on the left: 29204 steps of 32767 are 1 dB under full scale.
            self.assertEqual([int(np.abs(side).max()) for side in values], [29204, right_peak], rate)

    def test_refuses_a_name_that_is_no_song_of_the_project(self):
        self.assertEqual(self.refused(build_song, self.project, "march", self.out, RATE), "no song named 'march'")
        self.assertFalse(self.out.exists())

    def test_refuses_a_rate_that_makes_no_sense_and_takes_a_whole_one_written_as_a_float(self):
        for rate in (0, -8000, 8000.5, "8000", None, True, float("inf")):
            self.assertTrue(self.refused(build_song, self.project, "hush", self.out, rate).startswith("rate: "), rate)
        self.assertFalse(self.out.exists())
        self.assertEqual(build_song(self.project, "hush", self.out, 8000.0),
                         build_song(self.project, "hush", self.out / "again", 8000))
        self.assertEqual(load_wav(self.out / "hush.wav")[1], 8000)

    def test_song_that_cannot_be_rendered_is_refused_with_its_name_in_front(self):
        unread = copy.deepcopy(TUNE)
        unread["patterns"]["b"]["lead"] = "c ? d"
        for song, said in ((unread, "pattern 'b', channel 'lead': cannot read \"? d\" in a channel"),
                           ({**TUNE, "order": []}, "the song has no rows, so there is nothing to loop")):
            self.assertEqual(self.refused(build_song, project_of(march=song), "march", self.out, RATE),
                             f"songs.march: {said}")
        self.assertFalse(self.out.exists())


class BuildLayeredSongTest(BuildCase):
    def read(self, file: str) -> list[np.ndarray]:
        """The two sides of a built file, once it is known to be in stereo, a loop long, at the rate of the tests."""
        sides, rate = load_wav(self.out / file)
        self.assertEqual(([len(side) for side in sides], rate), ([6400, 6400], RATE), file)
        return sides

    def rendering(self) -> tuple[dict, dict, float]:
        """The march as the project gives it, its groups rendered, and the one scale that is shared by their files."""
        song = project_of(march=MARCH).song("march")
        rendered = render_groups(song, RATE)
        return song, rendered, shared_scale(rendered)

    def test_layered_song_is_one_file_a_group_and_key_and_one_in_all_for_a_group_of_drums_alone(self):
        self.built()
        self.assertEqual(self.files(), [*GROUP_FILES, AHEAD])
        for file in self.files():
            self.assertTrue(all(side.any() for side in self.read(file)), file)

    def test_files_of_the_groups_are_their_mixes_all_multiplied_by_the_one_shared_scale(self):
        entry = self.built()
        _, rendered, scale = self.rendering()
        peaks = {}
        for group, named in entry["groups"].items():
            for key in entry["keys"]:
                file = named["file"] if "file" in named else named["files"][str(key)]
                sides = self.read(file)
                for side, wanted in zip(sides, group_mix(rendered["groups"][group], key)):
                    np.testing.assert_allclose(side, wanted.astype(np.float64) * scale, rtol=0, atol=0.51 * STEP)
                peaks[file] = loudest(sides)
        self.assertEqual(sorted(peaks), GROUP_FILES)
        # Only the loudest of them is 1 dB under full scale. The others keep their level beside it.
        self.assertAlmostEqual(peaks.pop("march-top-2.wav"), PEAK, delta=STEP)
        self.assertLess(max(peaks.values()), 0.85 * PEAK)
        self.assertLess(min(peaks.values()), 0.3 * PEAK)

    def test_files_of_the_groups_of_a_version_add_up_to_the_version_at_the_shared_scale(self):
        self.built()
        song, rendered, scale = self.rendering()
        for version, key, files in (("full", 0, ("march-top-0.wav", "march-low-0.wav", "march-beat.wav")),
                                    ("thin", 2, ("march-beat.wav", "march-low-2.wav"))):
            added = [sum(side.astype(np.float64) for side in pair) for pair in zip(*map(self.read, files))]
            for side, wanted in zip(added, mix_version(rendered, song, version, key)):
                # A file is within half a step of 16 bits of its samples, and a sum is within that for each file.
                np.testing.assert_allclose(side, wanted.astype(np.float64) * scale, rtol=0,
                                           atol=0.51 * STEP * len(files), err_msg=version)

    def test_entry_names_the_file_of_each_group_and_key_and_gives_the_keys_and_versions_of_the_song(self):
        entry = self.built()
        self.assertEqual(list(entry), ["loop_samples", "bar_samples", "gain", "source", "keys", "groups", "versions",
                                       "premixed"])
        # The 16 rows of the march are one bar.
        self.assertEqual((entry["loop_samples"], entry["bar_samples"]), (6400, 6400))
        self.assertEqual(entry["source"], source_hash(project_of(march=MARCH), "march", RATE))
        self.assertEqual(entry["keys"], [0, 2])
        # Where a key names a file it is a string, which is what it is once the manifest has been through JSON.
        self.assertEqual(entry["groups"], {"top": {"files": {"0": "march-top-0.wav", "2": "march-top-2.wav"}},
                                           "low": {"files": {"0": "march-low-0.wav", "2": "march-low-2.wav"}},
                                           "beat": {"file": "march-beat.wav"}})
        self.assertEqual(list(entry["groups"]), ["top", "low", "beat"])
        self.assertEqual(entry["versions"], MARCH["layers"]["versions"])
        self.assertEqual(json.loads(json.dumps(entry)), entry)

    def test_gain_has_six_decimal_places_and_brings_a_file_back_to_the_level_of_the_song(self):
        entry = self.built()
        self.assertEqual(entry["gain"], 1.016429)
        _, rendered, _ = self.rendering()
        for file, group, key in (("march-top-2.wav", "top", 2), ("march-low-0.wav", "low", 0),
                                 ("march-beat.wav", "beat", 2)):
            for side, wanted in zip(self.read(file), group_mix(rendered["groups"][group], key)):
                np.testing.assert_allclose(side.astype(np.float64) * entry["gain"], wanted, rtol=0, atol=STEP)

    def test_version_mixed_ahead_is_at_a_scale_of_its_own_and_its_scale_brings_it_to_the_level_of_the_groups(self):
        entry = self.built()
        self.assertEqual(entry["premixed"], [{"id": "full/2", "file": AHEAD, "scale": 1.041125}])
        ahead = self.read(AHEAD)
        # The groups added up are louder than the loudest of them, and the file is 1 dB under full scale all the same.
        self.assertAlmostEqual(loudest(ahead), PEAK, delta=STEP)
        song, rendered, scale = self.rendering()
        for side, wanted in zip(ahead, mix_version(rendered, song, "full", 2)):
            np.testing.assert_allclose(side.astype(np.float64) * 1.041125, wanted.astype(np.float64) * scale,
                                       rtol=0, atol=STEP)

    def test_song_that_asks_for_no_premix_has_none_in_its_entry_and_the_same_files_for_its_groups(self):
        with_one = self.built()
        bare = self.out.parent / "bare"
        entry = self.built(march("premix"), folder=bare)
        self.assertEqual(list(entry), list(with_one)[:-1])
        self.assertEqual(self.files(bare), GROUP_FILES)
        for key in ("loop_samples", "bar_samples", "gain", "keys", "groups", "versions"):
            self.assertEqual(entry[key], with_one[key], key)
        for file in GROUP_FILES:
            self.assertEqual((bare / file).read_bytes(), (self.out / file).read_bytes(), file)
        # A list of none is written as it is.
        self.assertEqual(self.built(march(premix=[]), folder=bare)["premixed"], [])

    def test_files_are_named_by_the_keys_of_the_song_and_keys_left_out_are_the_key_it_is_written_in(self):
        entry = self.built(march(keys=[5, -3], premix=[{"version": "thin", "key": -3}]))
        self.assertEqual(entry["keys"], [5, -3])
        self.assertEqual(entry["groups"]["low"], {"files": {"5": "march-low-5.wav", "-3": "march-low--3.wav"}})
        self.assertEqual(entry["premixed"][0]["id"], "thin/-3")
        self.assertEqual(self.files(), ["march-beat.wav", "march-low--3.wav", "march-low-5.wav", "march-top--3.wav",
                                        "march-top-5.wav", "march.thin.-3.wav"])
        written = self.out.parent / "written"
        entry = self.built(march("keys", "premix"), folder=written)
        self.assertEqual(entry["keys"], [0])
        self.assertEqual(entry["groups"], {"top": {"files": {"0": "march-top-0.wav"}},
                                           "low": {"files": {"0": "march-low-0.wav"}},
                                           "beat": {"file": "march-beat.wav"}})
        self.assertEqual(self.files(written), ["march-beat.wav", "march-low-0.wav", "march-top-0.wav"])

    def test_bar_samples_are_the_share_a_bar_has_of_the_loop_with_a_half_rounded_up(self):
        # 32 rows are two bars. At 8003 Hz they are 12805 samples, and half of them is 6402 and a half.
        entry = self.built({**MARCH, "order": ["a", "b", "a", "b"]}, 8003)
        self.assertEqual((entry["loop_samples"], entry["bar_samples"]), (12805, 6403))
        # 48 rows are three bars: a third of 19202 samples is 6400 and two thirds.
        entry = self.built({**MARCH, "order": ["a", "b"] * 3}, 8001)
        self.assertEqual((entry["loop_samples"], entry["bar_samples"]), (19202, 6401))

    def test_known_16_bit_values_of_a_layered_song_at_44100_and_48000_hz(self):
        # For each rate: the loop, the bar, the gain and the scale of the version mixed ahead; where the samples
        # below are taken; the left and the right there of a group in the higher key, of the drums and of the
        # version mixed ahead; and the loudest sample of each file.
        cases = ((44100, (35280, 35280, 1.048494, 1.110007), (0, 1, 500, 17640, 35279),
                  {"march-top-2.wav": [(-1308, -889, 11935, -3424, -1397), (-286, -45, 238, -793, -335)],
                   "march-beat.wav": [(5, 109, 1555, 84, -101), (5, 109, 1555, 88, -101)],
                   AHEAD: [(-1091, -622, 9880, -2427, -1459), (2, 306, -5381, 1156, -728)]},
                  [9711, 8544, 8148, 25345, 29204, 29204]),
                 (48000, (38400, 38400, 1.05075, 1.111993), (0, 1, 500, 19200, 38399),
                  {"march-top-2.wav": [(-1309, -925, -7268, -3421, -1391), (-288, -68, -3960, -795, -334)],
                   "march-beat.wav": [(-4, 92, 1833, 76, -101), (-4, 92, 1833, 79, -101)],
                   AHEAD: [(-1098, -668, -3520, -2429, -1450), (-9, 271, 2295, 1140, -725)]},
                  [9589, 8527, 8133, 25381, 29204, 29204]))
        for rate, numbers, spots, held, peaks in cases:
            folder = self.out / str(rate)
            entry = self.built(rate=rate, folder=folder)
            self.assertEqual((entry["loop_samples"], entry["bar_samples"], entry["gain"],
                              entry["premixed"][0]["scale"]), numbers, rate)
            values = {}
            for file in (*GROUP_FILES, AHEAD):
                sides, read_rate = load_wav(folder / file)
                self.assertEqual(read_rate, rate, file)
                values[file] = [np.rint(side.astype(np.float64) * 32767).astype(int) for side in sides]
                self.assertEqual([len(side) for side in values[file]], [numbers[0]] * 2, (rate, file))
            for file, wanted in held.items():
                self.assertEqual([tuple(side[list(spots)].tolist()) for side in values[file]], wanted, (rate, file))
            # 29204 steps of 32767 are 1 dB under full scale. One file of the groups is there, and the version
            # mixed ahead is there by a scale of its own.
            self.assertEqual([max(int(np.abs(side).max()) for side in sides) for sides in values.values()], peaks,
                             rate)

    def test_silent_layered_song_is_written_as_it_is_with_a_gain_and_a_scale_of_1(self):
        layers = {"groups": {"one": ["lead"], "two": ["bass"]}, "versions": {"both": ["one", "two"]},
                  "premix": [{"version": "both", "key": 0}]}
        entry = self.built({**HUSH, "layers": layers})
        self.assertEqual((entry["loop_samples"], entry["gain"]), (1600, 1.0))
        self.assertEqual(entry["premixed"], [{"id": "both/0", "file": "march.both.0.wav", "scale": 1.0}])
        self.assertEqual(self.files(), ["march-one-0.wav", "march-two-0.wav", "march.both.0.wav"])
        for file in self.files():
            sides, _ = load_wav(self.out / file)
            self.assertEqual([len(side) for side in sides], [1600, 1600], file)
            self.assertFalse(sides[0].any() or sides[1].any(), file)

    def test_refuses_a_channel_that_is_in_no_group_or_is_in_a_group_twice(self):
        groups = MARCH["layers"]["groups"]
        for changed, said in (({**groups, "top": ["lead"]}, "channel 'alto' is in no group"),
                              ({**groups, "low": ["bass", "alto"]}, "channel 'alto' is in 'top' and again in 'low'"),
                              ({**groups, "low": ["bass", "bass"]}, "channel 'bass' is in 'low' and again in 'low'")):
            self.assertEqual(self.refused(self.built, march(groups=changed)),
                             f"songs.march: layers.groups: {said}, so the files would not add up to the song")
        self.assertFalse(self.out.exists())

    def test_refuses_a_premix_of_a_version_or_in_a_key_the_song_does_not_have(self):
        premix = [{"version": "thin", "key": 0}, {"version": "full", "key": 1}]
        self.assertEqual(self.refused(self.built, march(premix=premix)),
                         "songs.march: layers.premix[1]: key 1 is not one of the song's keys, [0, 2]")
        self.assertEqual(self.refused(self.built, march("keys")),
                         "songs.march: layers.premix[0]: key 2 is not one of the song's keys, [0]")
        # A project is not loaded with a premix of a version it does not have, so this one is put in after.
        project = project_of(march=MARCH)
        project.songs["march"]["layers"]["premix"] = [{"version": "loud", "key": 0}]
        self.assertEqual(self.refused(build_song, project, "march", self.out, RATE),
                         "songs.march: layers.premix[0]: no version named 'loud'")
        self.assertFalse(self.out.exists())

    def test_refuses_two_files_that_would_have_one_name_once_case_is_left_out_of_it(self):
        cased = {"groups": {"Top": ["lead"], "top": ["alto"], "low": ["bass"], "beat": ["kit", "hat"]},
                 "versions": {"full": ["Top", "top", "low", "beat"]}}
        # A group of drums alone has no key in the name of its file, and this one has a name that ends as one does.
        keyed = {"groups": {"top": ["lead", "alto", "bass"], "top-2": ["kit", "hat"]},
                 "versions": {"full": ["top", "top-2"]}}
        asked = {"version": "full", "key": 2}
        twice = {"premix": [asked, {"version": "thin", "key": 0}, asked]}
        for layers, first, second in ((cased, "march-Top-0.wav", "march-top-0.wav"),
                                      (keyed, "march-top-2.wav", "march-top-2.wav"),
                                      (twice, AHEAD, AHEAD)):
            self.assertEqual(self.refused(self.built, march(**layers)),
                             f"songs.march: two files would be called {first!r} and {second!r}, which is one name "
                             "where case does not count")
        self.assertFalse(self.out.exists())

    def test_what_the_groups_or_a_version_cannot_be_mixed_for_is_refused_with_the_name_of_the_song_in_front(self):
        patterns = copy.deepcopy(MARCH["patterns"])
        del patterns["a"]["alto"]
        self.assertEqual(self.refused(self.built, {**MARCH, "patterns": patterns, "order": ["a"]}),
                         "songs.march: group 'top': channel 'alto' plays in no pattern of the order, "
                         "so it has no stem")
        self.assertFalse(self.out.exists())
        # A project is not loaded with a version of a group it does not have, so this one is put in after. It is
        # found once the groups are rendered and the version is to be mixed.
        project = project_of(march=MARCH)
        project.songs["march"]["layers"]["versions"]["full"].append("gone")
        with self.assertRaises(RowSynthError) as caught:
            build_song(project, "march", self.out, RATE)
        self.assertEqual(str(caught.exception), "songs.march: version 'full': group 'gone' was not rendered")


class BuildIntroTest(BuildCase):
    def rendered(self, data: dict) -> tuple[dict, tuple[np.ndarray, np.ndarray]]:
        """A song as the project gives it, and the two sides of its intro as `render_intro` gives them."""
        song = project_of(march=data).song("march")
        intro = render_intro(song, RATE)
        return song, (intro["left"], intro["right"])

    def loop(self, song: dict) -> tuple[np.ndarray, np.ndarray]:
        """The two sides of the loop of a song without layers, as it is rendered."""
        loop = render_loop(song, RATE)
        return loop["left"], loop["right"]

    def written(self, file: str, wanted: tuple, scale: float) -> list[np.ndarray]:
        """The two sides of a built file, once they are known to be those given times a scale, in stereo."""
        sides, rate = load_wav(self.out / file)
        self.assertEqual((len(sides), rate), (2, RATE), file)
        for side, was in zip(sides, wanted):
            self.assertEqual(len(side), len(was), file)
            np.testing.assert_allclose(side, was.astype(np.float64) * scale, rtol=0, atol=0.51 * STEP, err_msg=file)
        return sides

    def test_song_with_an_intro_has_one_file_more_which_is_the_intro_as_it_is_rendered(self):
        data = led_in(TUNE, STAMP)
        self.built(data)
        self.assertEqual(self.files(), ["march.intro.wav", "march.wav"])
        song, intro = self.rendered(data)
        sides = self.written("march.intro.wav", intro, PEAK / loudest([*intro, *self.loop(song)]))
        # Its pattern is 8 rows of 400 samples, and it runs on for half a second after them, which is 4000 more.
        self.assertEqual([len(side) for side in sides], [7200, 7200])
        # What the pattern is still sounding when it ends is in the file, after it.
        self.assertTrue(all(side[3200:].any() for side in sides))

    def test_intro_and_loop_are_written_at_the_one_scale_of_the_loudest_sample_of_the_two(self):
        data = led_in(TUNE, STAMP)
        entry = self.built(data)
        song, intro = self.rendered(data)
        loop = self.loop(song)
        # The intro is the louder of the two. What has it 1 dB under full scale is the scale of the loop as well,
        # which is as far under the intro in its file as it was rendered.
        self.assertGreater(loudest(intro), 1.02 * loudest(loop))
        scale = PEAK / loudest(intro)
        self.assertAlmostEqual(loudest(self.written("march.intro.wav", intro, scale)), PEAK, delta=STEP)
        self.assertAlmostEqual(loudest(self.written("march.wav", loop, scale)), scale * loudest(loop), delta=STEP)
        # The gain is 1 over that scale, and either file times the gain is at the level of the song.
        self.assertEqual(entry["gain"], 1.017109)
        for file, wanted in (("march.intro.wav", intro), ("march.wav", loop)):
            for side, was in zip(load_wav(self.out / file)[0], wanted):
                np.testing.assert_allclose(side.astype(np.float64) * entry["gain"], was, rtol=0, atol=STEP)

    def test_intro_that_is_quieter_than_the_loop_leaves_the_loop_and_the_gain_as_they_are_without_it(self):
        data = led_in(TUNE)
        entry = self.built(data)
        apart = self.out.parent / "apart"
        self.assertEqual(entry["gain"], self.built(TUNE, folder=apart)["gain"])
        self.assertEqual((self.out / "march.wav").read_bytes(), (apart / "march.wav").read_bytes())
        # The scale is the one the loop has on its own, and the intro is under the loop by as much as it was.
        song, intro = self.rendered(data)
        sides = self.written("march.intro.wav", intro, PEAK / loudest(self.loop(song)))
        self.assertLess(loudest(sides), 0.85 * PEAK)

    def test_entry_has_the_intro_last_with_its_file_its_samples_and_the_sample_the_loop_begins_at(self):
        entry = self.built(led_in(TUNE, STAMP))
        self.assertEqual(list(entry), ["loop_samples", "gain", "source", "file", "intro"])
        self.assertEqual((entry["loop_samples"], entry["file"]), (6400, "march.wav"))
        self.assertEqual(entry["source"], source_hash(project_of(march=led_in(TUNE, STAMP)), "march", RATE))
        self.assertEqual(list(entry["intro"]), ["file", "in_samples", "walk_samples"])
        self.assertEqual(entry["intro"], {"file": "march.intro.wav", "in_samples": 7200, "walk_samples": 3200})
        self.assertEqual(json.loads(json.dumps(entry)), entry)

    def test_known_16_bit_values_of_an_intro_and_its_loop_at_44100_and_48000_hz(self):
        # For each rate: the gain, the samples of the intro and the sample of it the loop begins at; then for the
        # intro and for the loop, where the samples below are taken, the left and the right there, and the loudest
        # sample of each side.
        cases = ((44100, 1.047096, (39690, 17640),
                  ((0, 1, 500, 17639, 17640, 20000, 39689), (169, 583, -3998, -2078, 2389, -1126, 0),
                   (121, 294, -2931, 2538, 3433, -270, 0), [29204, 13712]),
                  ((0, 1, 500, 4410, 17640, 17641, 30000, 35279),
                   (-625, 4156, -2887, 2461, -2746, -2302, -4783, -790),
                   (-675, 549, 4434, -2460, 1107, 1294, 2728, 1295), [28459, 13506])),
                 (48000, 1.049289, (43200, 19200),
                  ((0, 1, 500, 19199, 19200, 21000, 43199), (155, 536, -5262, -2070, 5065, -1527, 0),
                   (111, 270, -5059, 2534, 4094, -369, 0), [29204, 13722]),
                  ((0, 1, 500, 4800, 19200, 19201, 30000, 38399),
                   (1543, 4822, -3866, 2436, -2750, -2343, 5050, -785),
                   (-160, 698, 4058, -2466, 1096, 1269, 4262, 1293), [28443, 13524])))
        for rate, gain, ends, *held in cases:
            folder = self.out / str(rate)
            entry = self.built(led_in(TUNE, STAMP), rate, folder)
            self.assertEqual((entry["gain"], entry["intro"]["in_samples"], entry["intro"]["walk_samples"]),
                             (gain, *ends), rate)
            lengths = (ends[0], entry["loop_samples"])
            for file, length, (spots, left, right, peaks) in zip(("march.intro.wav", "march.wav"), lengths, held):
                sides, read_rate = load_wav(folder / file)
                self.assertEqual(read_rate, rate, file)
                values = [np.rint(side.astype(np.float64) * 32767).astype(int) for side in sides]
                self.assertEqual([len(side) for side in values], [length] * 2, (rate, file))
                self.assertEqual([tuple(side[list(spots)].tolist()) for side in values], [left, right], (rate, file))
                # 29204 steps of 32767 are 1 dB under full scale. The intro is there, and the loop is under it.
                self.assertEqual([int(np.abs(side).max()) for side in values], peaks, (rate, file))

    def test_silence_is_written_as_it_is_and_the_scale_goes_by_what_sounds_beside_it(self):
        # Nothing sounds, in the loop or before it: four rows, and half a second after them in the intro.
        entry = self.built({**HUSH, "intro": {"order": ["p"], "tail": 0.5}})
        self.assertEqual((entry["gain"], entry["intro"]["in_samples"], entry["intro"]["walk_samples"]),
                         (1.0, 5600, 1600))
        for file, length in (("march.wav", 1600), ("march.intro.wav", 5600)):
            sides, _ = load_wav(self.out / file)
            self.assertEqual([len(side) for side in sides], [length] * 2, file)
            self.assertFalse(sides[0].any() or sides[1].any(), file)
        # Only the intro sounds. It is 1 dB under full scale, and the loop is silence still.
        woken = {**HUSH, "patterns": {**HUSH["patterns"], "in": {"lead": "o4 c8 e8", "bass": "o2 c4"}},
                 "intro": {"order": ["in"], "tail": 0.3, "fade": 0}}
        self.assertEqual(self.built(woken)["gain"], 0.584231)
        self.assertAlmostEqual(loudest(load_wav(self.out / "march.intro.wav")[0]), PEAK, delta=STEP)
        self.assertFalse(any(side.any() for side in load_wav(self.out / "march.wav")[0]))

    def test_layered_song_with_an_intro_has_its_file_beside_those_of_its_groups_and_its_entry_last(self):
        entry = self.built(led_in(MARCH, STAMP_OF_FOUR))
        self.assertEqual(self.files(), [*GROUP_FILES, AHEAD, "march.intro.wav"])
        self.assertEqual(list(entry), ["loop_samples", "bar_samples", "gain", "source", "keys", "groups", "versions",
                                       "premixed", "intro"])
        self.assertEqual(entry["intro"], {"file": "march.intro.wav", "in_samples": 7200, "walk_samples": 3200})
        # It comes last with no version mixed ahead as well.
        bare = self.out.parent / "bare"
        entry = self.built(led_in(march("premix"), STAMP_OF_FOUR), folder=bare)
        self.assertEqual(list(entry)[-2:], ["versions", "intro"])
        self.assertEqual(self.files(bare), [*GROUP_FILES, "march.intro.wav"])

    def test_files_of_the_groups_and_the_intro_are_at_the_one_scale_of_the_loudest_sample_among_them(self):
        data = led_in(MARCH, STAMP_OF_FOUR)
        entry = self.built(data)
        song, intro = self.rendered(data)
        mixed = render_groups(song, RATE)
        # The intro is louder than the loudest of the groups. What has it 1 dB under full scale is the scale of
        # every file of the groups, and none of them is as loud as that now.
        among_groups = PEAK / shared_scale(mixed)
        self.assertGreater(loudest(intro), 1.15 * among_groups)
        scale = PEAK / loudest(intro)
        self.assertAlmostEqual(loudest(self.written("march.intro.wav", intro, scale)), PEAK, delta=STEP)
        peaks = {}
        for group, named in entry["groups"].items():
            for key in entry["keys"]:
                file = named["file"] if "file" in named else named["files"][str(key)]
                peaks[file] = loudest(self.written(file, group_mix(mixed["groups"][group], key), scale))
        self.assertEqual(sorted(peaks), GROUP_FILES)
        self.assertAlmostEqual(max(peaks.values()), scale * among_groups, delta=STEP)
        self.assertLess(max(peaks.values()), 0.85 * PEAK)
        self.assertEqual(entry["gain"], 1.229206)

    def test_version_mixed_ahead_keeps_its_own_scale_and_is_brought_to_the_level_that_is_shared_with_the_intro(self):
        data = led_in(MARCH, STAMP_OF_FOUR)
        entry = self.built(data)
        apart = self.out.parent / "apart"
        without = self.built(MARCH, folder=apart)
        # Its file is the one the march has without an intro. The groups are lower beside this intro than they
        # were, so what brings the file to their level is less.
        self.assertEqual((self.out / AHEAD).read_bytes(), (apart / AHEAD).read_bytes())
        self.assertEqual(without["premixed"], [{"id": "full/2", "file": AHEAD, "scale": 1.041125}])
        self.assertEqual(entry["premixed"], [{"id": "full/2", "file": AHEAD, "scale": 0.860905}])
        song, intro = self.rendered(data)
        scale = PEAK / loudest(intro)
        for side, wanted in zip(load_wav(self.out / AHEAD)[0], mix_version(render_groups(song, RATE), song, "full", 2)):
            np.testing.assert_allclose(side.astype(np.float64) * 0.860905, wanted.astype(np.float64) * scale,
                                       rtol=0, atol=STEP)

    def test_intro_that_is_quieter_than_the_groups_leaves_their_files_and_their_entry_as_they_are_without_it(self):
        data = led_in(MARCH)
        entry = self.built(data)
        apart = self.out.parent / "apart"
        without = self.built(MARCH, folder=apart)
        for file in (*GROUP_FILES, AHEAD):
            self.assertEqual((self.out / file).read_bytes(), (apart / file).read_bytes(), file)
        # The source is that of another song. The rest of the entry is what it was, with the intro after it.
        self.assertEqual(list(entry), [*without, "intro"])
        for key in without:
            if key != "source":
                self.assertEqual(entry[key], without[key], key)
        # The scale is the one the groups share on their own, and the intro is under the loudest of them.
        song, intro = self.rendered(data)
        sides = self.written("march.intro.wav", intro, shared_scale(render_groups(song, RATE)))
        self.assertLess(loudest(sides), 0.9 * PEAK)

    def test_what_an_intro_is_refused_for_comes_with_the_name_of_the_song_in_front_and_nothing_is_written(self):
        for song in (TUNE, MARCH):
            layered = "layers" in song
            # A fade that is longer than its tail is refused before a note is played.
            self.assertEqual(self.refused(self.built, {**song, "intro": {"order": ["b"], "tail": 0.5, "fade": 0.6}}),
                             "songs.march: intro.fade: 0.6 seconds is longer than the tail, which is 0.5", layered)
            # A tail that is left out is 1.5 seconds, and a bar of the song is 0.8. How many samples the silent
            # bar comes to is known once the intro has been played.
            with self.assertRaises(RowSynthError) as caught:
                self.built({**song, "intro": {"order": ["b"]}})
            self.assertEqual(str(caught.exception), "songs.march: intro.tail: 1.5 seconds is 12000 samples at a "
                             "rate of 8000, and the silent bar after the intro has 6400", layered)
            self.assertFalse(self.out.exists(), layered)


class BuildTest(BuildCase):
    def test_builds_every_song_in_the_order_of_the_project_and_gives_the_manifest(self):
        manifest = build(self.project, self.out, RATE)
        self.assertEqual(list(manifest), ["rate", "format", "songs"])
        # The format left out is WAV.
        self.assertEqual((manifest["rate"], manifest["format"]), (RATE, "wav"))
        self.assertEqual(list(manifest["songs"]), ["tune", "drone", "hush"])
        self.assertEqual(self.files(), ["drone.wav", "hush.wav", "music.json", "tune.wav"])
        # Each song is written as it is on its own, and its entry is the one it has on its own.
        apart = self.out.parent / "apart"
        for name, entry in manifest["songs"].items():
            self.assertEqual(entry, build_song(self.project, name, apart, RATE), name)
            self.assertEqual((self.out / entry["file"]).read_bytes(), (apart / entry["file"]).read_bytes(), name)

    def test_layered_song_is_built_beside_a_plain_one_and_its_entry_reads_back_from_the_manifest_as_it_was(self):
        project = project_of(hush=HUSH, march=MARCH)
        manifest = build(project, self.out, RATE)
        self.assertEqual(self.files(), ["hush.wav", *GROUP_FILES, AHEAD, "music.json"])
        self.assertEqual(manifest["songs"]["march"], build_song(project, "march", self.out.parent / "apart", RATE))
        # The keys that name the files of a group are strings before the manifest is written, as they are after.
        self.assertEqual(json.loads((self.out / "music.json").read_text(encoding="utf-8")), manifest)
        self.assertEqual(stale(project, manifest), [])

    def test_song_with_an_intro_is_built_with_it_and_its_entry_reads_back_from_the_manifest_as_it_was(self):
        project = project_of(hush=HUSH, tune=led_in(TUNE, STAMP))
        manifest = build(project, self.out, RATE)
        self.assertEqual(self.files(), ["hush.wav", "music.json", "tune.intro.wav", "tune.wav"])
        self.assertEqual(manifest["songs"]["tune"]["intro"],
                         {"file": "tune.intro.wav", "in_samples": 7200, "walk_samples": 3200})
        self.assertNotIn("intro", manifest["songs"]["hush"])
        self.assertEqual(json.loads((self.out / "music.json").read_text(encoding="utf-8")), manifest)
        self.assertEqual(stale(project, manifest), [])

    def test_manifest_is_written_beside_the_files_as_json_with_an_indent_of_2_and_a_final_newline(self):
        # The rate left out is 48000, at which four rows of three ticks are 9600 samples.
        manifest = build(self.project, self.out, songs=["hush"])
        source = source_hash(self.project, "hush", 48000)
        self.assertEqual(manifest, {"rate": 48000, "format": "wav",
                                    "songs": {"hush": {"loop_samples": 9600, "gain": 1.0, "source": source,
                                                       "file": "hush.wav"}}})
        self.assertEqual((self.out / "music.json").read_bytes().decode("utf-8"),
                         '{\n  "rate": 48000,\n  "format": "wav",\n  "songs": {\n    "hush": {\n'
                         f'      "loop_samples": 9600,\n      "gain": 1.0,\n      "source": "{source}",\n'
                         '      "file": "hush.wav"\n    }\n  }\n}\n')

    def test_manifest_path_is_where_a_build_into_the_folder_writes_its_manifest(self):
        for folder in (self.out, str(self.out)):
            self.assertEqual(manifest_path(folder), self.out / "music.json")
        self.assertFalse(manifest_path(self.out).exists())
        manifest = build(self.project, self.out, RATE, songs=["hush"])
        self.assertEqual(json.loads(manifest_path(self.out).read_text(encoding="utf-8")), manifest)

    def test_songs_names_the_songs_to_build_in_the_order_they_are_named_in(self):
        manifest = build(self.project, self.out, RATE, songs=["hush", "tune"])
        self.assertEqual(list(manifest["songs"]), ["hush", "tune"])
        self.assertEqual(self.files(), ["hush.wav", "music.json", "tune.wav"])
        # With none named, the manifest is all there is to write.
        self.assertEqual(build(self.project, self.out / "none", RATE, songs=[]),
                         {"rate": RATE, "format": "wav", "songs": {}})
        self.assertEqual(self.files(self.out / "none"), ["music.json"])

    def test_makes_the_folder_when_it_is_missing_and_leaves_alone_what_it_did_not_write(self):
        deep = self.out / "music" / "loops"
        build(self.project, deep, RATE, songs=["hush"])
        self.assertEqual(self.files(deep), ["hush.wav", "music.json"])
        # What is in the folder already stays, a file of an earlier build with it.
        kept = {"notes.txt": b"kept\r\n", "march.wav": b"no file of this build",
                "hush.wav": (deep / "hush.wav").read_bytes()}
        for name, data in kept.items():
            (deep / name).write_bytes(data)
        manifest = build(self.project, deep, RATE, songs=["drone"])
        self.assertEqual(self.files(deep), ["drone.wav", "hush.wav", "march.wav", "music.json", "notes.txt"])
        for name, data in kept.items():
            self.assertEqual((deep / name).read_bytes(), data, name)
        self.assertEqual(json.loads((deep / "music.json").read_text(encoding="utf-8")), manifest)
        self.assertEqual(list(manifest["songs"]), ["drone"])

    def test_refused_build_writes_no_manifest(self):
        self.assertTrue(self.refused(build, self.project, self.out, 0).startswith("rate: "))
        self.assertFalse(self.out.exists())
        with self.assertRaises(RowSynthError) as caught:
            build(self.project, self.out, RATE, songs=["hush", "march", "drone"])
        self.assertEqual(str(caught.exception), "no song named 'march'")
        self.assertEqual(self.files(), ["hush.wav"])


class BuildFormatTest(BuildCase):
    def ogg(self, file: str, rate: int = RATE) -> None:
        """Holds a built file to being an Ogg file of Opus, in stereo, made from samples at a rate."""
        raw = (self.out / file).read_bytes()
        # The first page of an Ogg file is 27 bytes and a table of one entry, and then the head of the Opus stream.
        # Its tenth byte is how many sides it has, and the four from its thirteenth are the rate, low byte first.
        self.assertEqual((raw[:4], raw[28:36], raw[37], int.from_bytes(raw[40:44], "little")),
                         (b"OggS", b"OpusHead", 2, rate), file)

    @with_ffmpeg
    def test_plain_song_as_opus_is_one_ogg_file_with_that_ending_in_the_folder_and_in_its_entry(self):
        entry = build_song(self.project, "drone", self.out, RATE, "opus")
        self.assertEqual(self.files(), ["drone.opus"])
        self.ogg("drone.opus")
        # The rest of the entry is what it is for a WAV file: the format changes no sample that goes into a file.
        apart = self.out.parent / "apart"
        self.assertEqual(entry, {**build_song(self.project, "drone", apart, RATE), "file": "drone.opus"})
        self.assertEqual(self.files(apart), ["drone.wav"])

    @with_ffmpeg
    def test_layered_song_as_opus_has_that_ending_on_the_files_of_its_groups_and_of_its_version_mixed_ahead(self):
        entry = self.built(fmt="opus")
        self.assertEqual(self.files(), ["march-beat.opus", "march-low-0.opus", "march-low-2.opus", "march-top-0.opus",
                                        "march-top-2.opus", "march.full.2.opus"])
        for file in self.files():
            self.ogg(file)
        self.assertEqual(entry["groups"], {"top": {"files": {"0": "march-top-0.opus", "2": "march-top-2.opus"}},
                                           "low": {"files": {"0": "march-low-0.opus", "2": "march-low-2.opus"}},
                                           "beat": {"file": "march-beat.opus"}})
        self.assertEqual(entry["premixed"], [{"id": "full/2", "file": "march.full.2.opus", "scale": 1.041125}])

    @with_ffmpeg
    def test_song_with_an_intro_as_opus_has_that_ending_on_its_intro_and_on_its_loop(self):
        entry = self.built(led_in(TUNE), fmt="opus")
        self.assertEqual(self.files(), ["march.intro.opus", "march.opus"])
        for file in self.files():
            self.ogg(file)
        self.assertEqual((entry["file"], entry["intro"]["file"]), ("march.opus", "march.intro.opus"))

    @with_ffmpeg
    def test_building_twice_as_opus_gives_the_same_bytes_and_other_bytes_for_another_song(self):
        again = self.out.parent / "again"
        self.assertEqual(build(self.project, self.out, RATE, "opus"), build(self.project, again, RATE, "opus"))
        self.assertEqual(self.files(), ["drone.opus", "hush.opus", "music.json", "tune.opus"])
        for file in self.files():
            self.assertEqual((self.out / file).read_bytes(), (again / file).read_bytes(), file)
        self.assertNotEqual((self.out / "drone.opus").read_bytes(), (self.out / "tune.opus").read_bytes())

    @with_ffmpeg
    def test_manifest_says_the_format_and_names_the_files_by_it_and_opus_takes_the_rate_that_is_left_out(self):
        manifest = build(self.project, self.out, fmt="opus", songs=["hush", "drone"])
        self.assertEqual(list(manifest), ["rate", "format", "songs"])
        self.assertEqual((manifest["rate"], manifest["format"]), (48000, "opus"))
        self.assertEqual([entry["file"] for entry in manifest["songs"].values()], ["hush.opus", "drone.opus"])
        self.assertEqual(self.files(), ["drone.opus", "hush.opus", "music.json"])
        for file in ("drone.opus", "hush.opus"):
            self.ogg(file, 48000)
        self.assertEqual(json.loads((self.out / "music.json").read_text(encoding="utf-8")), manifest)
        # The format is no part of what a song is made from: the songs are those of the project still.
        self.assertEqual(stale(self.project, manifest), [])

    def test_wav_by_name_is_the_build_with_the_format_left_out_and_looks_for_no_ffmpeg(self):
        named = self.out.parent / "named"
        with mock.patch("shutil.which", return_value=None) as which:
            self.assertEqual(build(self.project, named, RATE, "wav"), build(self.project, self.out, RATE))
        which.assert_not_called()
        self.assertEqual(self.files(named), ["drone.wav", "hush.wav", "music.json", "tune.wav"])
        for file in self.files(named):
            self.assertEqual((named / file).read_bytes(), (self.out / file).read_bytes(), file)

    def test_refuses_a_format_a_build_is_not_written_in_before_a_note_is_played_or_the_folder_is_made(self):
        # MP3 is a format of the listen page, and is none of a build.
        for fmt in ("mp3", "flac", "WAV", "", None):
            said = f"format: a build is written as wav or opus, not {fmt!r}"
            self.assertEqual(self.refused(build, self.project, self.out, RATE, fmt), said)
            self.assertEqual(self.refused(build_song, self.project, "tune", self.out, RATE, fmt), said)
        self.assertFalse(self.out.exists())

    def test_refuses_opus_at_a_rate_it_does_not_take_before_a_note_is_played_or_the_folder_is_made(self):
        # No ffmpeg is needed to know it.
        with mock.patch("shutil.which", return_value=None):
            for rate in (44100, 11025.0):
                said = f"opus takes a rate of 8000, 12000, 16000, 24000 or 48000, got {int(rate)}"
                self.assertEqual(self.refused(build, self.project, self.out, rate, "opus"), said)
                self.assertEqual(self.refused(build_song, self.project, "tune", self.out, rate, "opus"), said)
        self.assertFalse(self.out.exists())

    def test_opus_with_no_ffmpeg_on_the_path_is_refused_as_the_first_file_is_to_be_written_and_none_is(self):
        with mock.patch("shutil.which", return_value=None):
            with self.assertRaises(RowSynthError) as caught:
                build(self.project, self.out, RATE, "opus")
        # What `save_audio` says comes through, behind the song that was to be written.
        self.assertEqual(str(caught.exception),
                         "songs.tune: ffmpeg is needed to write opus, and none is found on the path")
        # The folder is made by then, and has no file of a song in it and no manifest.
        self.assertEqual(self.files(), [])


class SourceHashTest(BuildCase):
    def test_hash_is_16_hex_digits_and_the_same_for_the_same_song_at_the_same_rate(self):
        digits = source_hash(self.project, "tune", RATE)
        self.assertRegex(digits, r"\A[0-9a-f]{16}\Z")
        self.assertEqual(source_hash(project_of(**copy.deepcopy(PLAIN)), "tune", RATE), digits)
        self.assertEqual(source_hash(self.project, "tune", 8000.0), digits)
        # The other songs of the project are no part of it, and have digits of their own.
        self.assertEqual(source_hash(project_of(tune=TUNE), "tune", RATE), digits)
        self.assertNotEqual(source_hash(self.project, "drone", RATE), digits)

    def test_hash_changes_with_the_song_its_instruments_the_rate_and_the_version(self):
        digits = source_hash(self.project, "tune", RATE)
        retuned = copy.deepcopy(TUNE)
        retuned["patterns"]["a"]["lead"] = "l8 e g b > e"
        self.assertNotEqual(source_hash(project_of(tune=retuned), "tune", RATE), digits)
        # An instrument of the project is part of the song as it is rendered.
        softer = Project.from_dict({"instruments": {**INSTRUMENTS, "pipe": {**INSTRUMENTS["pipe"], "duty": 2}},
                                    "songs": {"tune": TUNE}})
        self.assertNotEqual(source_hash(softer, "tune", RATE), digits)
        self.assertNotEqual(source_hash(self.project, "tune", 44100), digits)
        with mock.patch("rowsynth.build.__version__", "0.0.0"):
            self.assertNotEqual(source_hash(self.project, "tune", RATE), digits)

    def test_hash_changes_with_the_order_of_the_channels_as_the_samples_do(self):
        patterns = {name: dict(reversed(pattern.items())) for name, pattern in TUNE["patterns"].items()}
        turned = {**TUNE, "patterns": patterns}
        # The song is the same data, and is another song once it is played: its stems are added the other way round.
        self.assertEqual(turned, TUNE)
        project = project_of(tune=turned)
        self.assertNotEqual(source_hash(project, "tune", RATE), source_hash(self.project, "tune", RATE))
        self.assertFalse(np.array_equal(render_loop(project.song("tune"), RATE)["left"],
                                        render_loop(self.project.song("tune"), RATE)["left"]))

    def test_refuses_a_name_that_is_no_song_and_a_rate_that_makes_no_sense(self):
        self.assertEqual(self.refused(source_hash, self.project, "march", RATE), "no song named 'march'")
        for rate in (0, 8000.5, "8000", None):
            self.assertTrue(self.refused(source_hash, self.project, "tune", rate).startswith("rate: "), rate)


class StaleTest(BuildCase):
    def test_manifest_of_a_build_of_the_project_as_it_is_has_no_stale_songs(self):
        # The songs are held against the project at the rate of the manifest, which is not the one a build has
        # when it is given none.
        manifest = build(self.project, self.out, RATE)
        self.assertEqual(stale(self.project, manifest), [])
        self.assertEqual(stale(self.project, json.loads((self.out / "music.json").read_text(encoding="utf-8"))), [])
        self.assertEqual(stale(self.project, {**manifest, "rate": 44100}), ["tune", "drone", "hush"])

    def test_names_the_songs_that_have_changed_and_the_songs_the_project_no_longer_has(self):
        manifest = build(self.project, self.out, RATE)
        retuned = copy.deepcopy(DRONE)
        retuned["patterns"]["p"]["low"] = "o3 c4 a4"
        self.assertEqual(stale(project_of(tune=TUNE, drone=retuned, hush=HUSH), manifest), ["drone"])
        self.assertEqual(stale(project_of(hush=HUSH, drone=retuned), manifest), ["tune", "drone"])
        self.assertEqual(stale(project_of(), manifest), ["tune", "drone", "hush"])
        # A song the manifest does not have is not one of its songs, stale or not.
        self.assertEqual(stale(project_of(**PLAIN, loud=LOUD), manifest), [])


if __name__ == "__main__":
    unittest.main()
