"""Tests for a build: the songs of a project written as files, and the manifest that names them."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from rowsynth.build import build, build_song, manifest_path, source_hash, stale
from rowsynth.errors import RowSynthError
from rowsynth.project import Project
from rowsynth.song import render_loop
from rowsynth.wav import load_wav

# With three ticks to a row, a row is 400 samples at 8000 Hz.
RATE = 8000

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


def project_of(**songs: dict) -> Project:
    """A project of those songs, in that order, with the instruments above."""
    return Project.from_dict({"instruments": INSTRUMENTS, "songs": songs})


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

    def test_refuses_a_song_with_layers_or_an_intro(self):
        layers = {"groups": {"all": ["lead", "bass", "kit"]}, "versions": {"full": ["all"]}}
        intro = {"order": ["a"]}
        for more in ({"layers": layers}, {"intro": intro}, {"layers": layers, "intro": intro}):
            project = project_of(hush=HUSH, march={**TUNE, **more})
            self.assertEqual(self.refused(build_song, project, "march", self.out, RATE),
                             "songs.march: a song with layers or an intro cannot be built", list(more))
        self.assertFalse(self.out.exists())

    def test_song_that_cannot_be_rendered_is_refused_with_its_name_in_front(self):
        unread = copy.deepcopy(TUNE)
        unread["patterns"]["b"]["lead"] = "c ? d"
        for song, said in ((unread, "pattern 'b', channel 'lead': cannot read \"? d\" in a channel"),
                           ({**TUNE, "order": []}, "the song has no rows, so there is nothing to loop")):
            self.assertEqual(self.refused(build_song, project_of(march=song), "march", self.out, RATE),
                             f"songs.march: {said}")
        self.assertFalse(self.out.exists())


class BuildTest(BuildCase):
    def test_builds_every_song_in_the_order_of_the_project_and_gives_the_manifest(self):
        manifest = build(self.project, self.out, RATE)
        self.assertEqual(list(manifest), ["rate", "songs"])
        self.assertEqual(manifest["rate"], RATE)
        self.assertEqual(list(manifest["songs"]), ["tune", "drone", "hush"])
        self.assertEqual(self.files(), ["drone.wav", "hush.wav", "music.json", "tune.wav"])
        # Each song is written as it is on its own, and its entry is the one it has on its own.
        apart = self.out.parent / "apart"
        for name, entry in manifest["songs"].items():
            self.assertEqual(entry, build_song(self.project, name, apart, RATE), name)
            self.assertEqual((self.out / entry["file"]).read_bytes(), (apart / entry["file"]).read_bytes(), name)

    def test_manifest_is_written_beside_the_files_as_json_with_an_indent_of_2_and_a_final_newline(self):
        # The rate left out is 48000, at which four rows of three ticks are 9600 samples.
        manifest = build(self.project, self.out, songs=["hush"])
        source = source_hash(self.project, "hush", 48000)
        self.assertEqual(manifest, {"rate": 48000, "songs": {"hush": {"loop_samples": 9600, "gain": 1.0,
                                                                      "source": source, "file": "hush.wav"}}})
        self.assertEqual((self.out / "music.json").read_bytes().decode("utf-8"),
                         '{\n  "rate": 48000,\n  "songs": {\n    "hush": {\n      "loop_samples": 9600,\n'
                         f'      "gain": 1.0,\n      "source": "{source}",\n      "file": "hush.wav"\n'
                         '    }\n  }\n}\n')

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
        self.assertEqual(build(self.project, self.out / "none", RATE, songs=[]), {"rate": RATE, "songs": {}})
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
