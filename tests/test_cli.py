"""Tests for the command line and its commands."""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import rowsynth
from rowsynth import checks
from rowsynth.bank import render_entry
from rowsynth.build import build
from rowsynth.cli import main
from rowsynth.commands import emit, load_file, load_project, mono
from rowsynth.errors import RowSynthError
from rowsynth.measure import brightness, measure
from rowsynth.project import Project
from rowsynth.song import render_song, render_stems
from rowsynth.wav import load_wav, save_wav


def blip(inst: str, note: object, ticks: float = 3) -> dict:
    """A sound of one note: a short one, unless it is told how many ticks to last."""
    return {"voices": [{"inst": inst, "notes": [[0, note, ticks]]}]}


# A project with a sound of each form and two songs, one of them layered. No table of it is in name order.
PROJECT = {
    "chords": {"Dm": {"root": 2, "tones": [2, 5, 9]}, "A7": {"root": 9, "tones": [9, 1, 4, 7]}},
    "instruments": {
        "snap": {"wave": "noise", "vol": [14, 6], "release": 0},
        "pipe": {"wave": "tri", "vol": 12},
    },
    "sounds": {
        "tap": {"def": blip("snap", 72)},
        "drip": {"defs": [blip("pipe", "e5"), blip("pipe", "g5"), blip("pipe", "a5")], "db": [-2, -3, -4]},
        "knock": {"parts": [[0, {"def": blip("snap", 50)}], [0.1, {"def": blip("snap", 45)}]]},
        # The note lasts the 30 ticks its start and its loop are cut from: a tenth of a second, and four tenths.
        "drone": {
            "held": {"note": blip("pipe", "d2", 30), "start": 6, "loop": 24, "join": 3, "end": blip("pipe", "d2"),
                     "let_go": 2},
        },
    },
    "songs": {
        # The verse is 16 rows and the turn 8, so the order fills 40. At 6 ticks a row that is 4 seconds.
        "round": {
            "row_ticks": 6,
            "instruments": {"lead": {"vol": [15, 11], "duty": 1}},
            "drums": {"k": {"inst": "snap", "note": "c2"}},
            "channels": {"tune": {"inst": "lead"}, "low": {"inst": "pipe"}, "kit": {"inst": "snap", "drums": True}},
            "patterns": {
                "verse": {"tune": "o4 l4 d f a f", "low": "o2 l2 d a", "kit": "l4 k k k k"},
                "turn": {"tune": "o4 l8 a g f e", "low": "o2 l2 a"},
            },
            "order": ["verse", "turn", "verse"],
            "chords": {"verse": ["Dm"], "turn": ["A7"]},
            "layers": {
                "groups": {"hits": ["kit"], "lines": ["tune", "low"]},
                "versions": {"whole": ["hits", "lines"], "bare": ["hits"]},
                "keys": [0, 3],
            },
        },
        # Two eighths and a dotted quarter are 10 rows, and so are a half and an eighth. At 3 ticks a row that is
        # half a second. The first channel sits to the left, and the second in the middle, having no pan.
        "sting": {
            "row_ticks": 3,
            "channels": {"one": {"inst": "pipe", "pan": -0.5}, "under": {"inst": "pipe"}},
            "patterns": {"x": {"one": "l8 c e g4.", "under": "o2 l2 c g8"}},
            "order": ["x"],
        },
    },
}

INFO = {
    "instruments": ["snap", "pipe"],
    "sounds": {
        "tap": {"kind": "sound"},
        "drip": {"kind": "variants", "variants": 3},
        "knock": {"kind": "parts"},
        "drone": {"kind": "held"},
    },
    "songs": {
        "round": {
            "channels": ["tune", "low", "kit"],
            "patterns": ["verse", "turn"],
            "order": ["verse", "turn", "verse"],
            "rows": 40.0,
            "seconds": 4.0,
            "layers": {"groups": ["hits", "lines"], "versions": ["whole", "bare"], "keys": [0, 3]},
        },
        "sting": {"channels": ["one", "under"], "patterns": ["x"], "order": ["x"], "rows": 10.0, "seconds": 0.5},
    },
    "chords": ["Dm", "A7"],
}

# A project of two songs that are one loop each, and not in name order. The first is the half second it is in
# PROJECT. The second is 4 rows at 3 ticks a row, a fifth of a second.
LOOPS = {
    "instruments": PROJECT["instruments"],
    "songs": {
        "sting": PROJECT["songs"]["sting"],
        "hum": {
            "row_ticks": 3,
            "channels": {"low": {"inst": "pipe"}},
            "patterns": {"p": {"low": "o2 l4 d"}},
            "order": ["p"],
        },
    },
}


def tone(hz: float, level: float, rate: int) -> np.ndarray:
    """Half a second of a sine that starts at zero."""
    return level * np.sin(2 * np.pi * hz * np.arange(rate // 2) / rate)


# One step of a 16-bit file. A sample read back from a file is nearer than this to the sample that was written.
STEP = 1 / 32767


class CommandLineCase(unittest.TestCase):
    """What the tests below share: a folder for their files, and main run with what it prints caught."""

    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.out = io.StringIO()
        self.err = io.StringIO()

    def saved(self, data: dict) -> Path:
        """The path of a project file holding `data`."""
        path = self.folder / "project.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def run_main(self, *argv: str) -> int:
        """What main returns for these arguments. What it prints is left in `out` and `err`."""
        with contextlib.redirect_stdout(self.out), contextlib.redirect_stderr(self.err):
            return main(list(argv))

    def printed(self) -> tuple[str, str]:
        return self.out.getvalue(), self.err.getvalue()


class Shared(CommandLineCase):
    def test_emit_prints_json_indented_by_2_in_the_order_given_with_non_ascii_as_it_is(self) -> None:
        with contextlib.redirect_stdout(self.out):
            emit({"b": 1, "a": ["fl\xfbte"]})
        self.assertEqual(self.out.getvalue(), '{\n  "b": 1,\n  "a": [\n    "fl\xfbte"\n  ]\n}\n')

    def test_emit_prints_null_for_a_number_json_cannot_write(self) -> None:
        with contextlib.redirect_stdout(self.out):
            emit({"peak_db": -math.inf, "levels": [math.nan, 1.5, (math.inf,)]})
        self.assertNotIn("Infinity", self.out.getvalue())
        self.assertNotIn("NaN", self.out.getvalue())
        self.assertEqual(json.loads(self.out.getvalue()), {"peak_db": None, "levels": [None, 1.5, [None]]})

    def test_load_project_gives_the_project_in_the_file(self) -> None:
        self.assertEqual(load_project(self.saved(PROJECT)), Project.from_dict(PROJECT))

    def test_load_file_gives_what_the_reader_makes_of_a_file_and_refuses_a_missing_one_by_its_path(self) -> None:
        path = self.folder / "words.txt"
        with self.assertRaises(RowSynthError) as refused:
            load_file(Path.read_text, path)
        self.assertEqual(str(refused.exception), f"{path}: no such file")
        path.write_text("here\n", encoding="utf-8")
        self.assertEqual(load_file(Path.read_text, path), "here\n")

    def test_mono_gives_one_side_as_it_is_and_two_as_their_mean_worked_out_in_doubles(self) -> None:
        left, right = np.float32([0.1, -0.5]), np.float32([0.2, 0.25])
        self.assertEqual(mono([left]).tolist(), left.astype(np.float64).tolist())
        mean = mono([left, right])
        self.assertEqual(mean.dtype, np.float64)
        self.assertEqual(mean.tolist(), ((left.astype(np.float64) + right) / 2).tolist())
        # Worked out in single precision, the first of them comes out a little off.
        self.assertNotEqual(mean[0], (left[0] + right[0]) / np.float32(2))


class Build(CommandLineCase):
    def build(self, out: str | Path, *typed: str, data: dict = LOOPS) -> int:
        """What main returns for building a project file holding `data` into the folder `out`.

        `typed` goes straight after the file: the names of songs, and then flags.
        """
        return self.run_main("build", str(self.saved(data)), *typed, "--out", str(out))

    def test_build_prints_the_folder_as_it_was_typed_the_manifest_the_rate_and_every_song_of_the_project(self) -> None:
        # The folder is typed with a separator after it, which the path of the manifest inside it has no use for.
        typed = f"{self.folder / 'built'}{os.sep}"
        self.assertEqual(self.build(typed), 0)
        manifest = build(Project.from_dict(LOOPS), self.folder / "apart")
        told = {"folder": typed, "manifest": str(self.folder / "built" / "music.json"), **manifest}
        self.assertEqual(self.printed(), (json.dumps(told, indent=2, ensure_ascii=False) + "\n", ""))
        self.assertEqual(list(told), ["folder", "manifest", "rate", "songs"])
        # With no rate typed it is 48000. With no name typed it is every song, in the order of the project.
        self.assertEqual(told["rate"], 48000)
        self.assertEqual([(name, entry["loop_samples"], entry["file"]) for name, entry in told["songs"].items()],
                         [("sting", 24000, "sting.wav"), ("hum", 9600, "hum.wav")])

    def test_the_files_are_in_a_folder_the_command_makes_with_a_manifest_that_holds_what_was_printed(self) -> None:
        out = self.folder / "made" / "built"
        self.assertEqual(self.build(out, "--rate", "8000"), 0)
        self.assertEqual(sorted(found.name for found in out.iterdir()), ["hum.wav", "music.json", "sting.wav"])
        told = json.loads(self.out.getvalue())
        self.assertEqual(Path(told["manifest"]), out / "music.json")
        self.assertEqual(json.loads(Path(told["manifest"]).read_text(encoding="utf-8")),
                         {"rate": told["rate"], "songs": told["songs"]})
        self.assertEqual([entry["file"] for entry in told["songs"].values()], ["sting.wav", "hum.wav"])

    def test_names_after_the_file_are_the_only_songs_built_in_the_order_they_are_named_in(self) -> None:
        one, both = self.folder / "one", self.folder / "both"
        self.assertEqual(self.build(one, "hum", "--rate", "8000"), 0)
        self.assertEqual(sorted(found.name for found in one.iterdir()), ["hum.wav", "music.json"])
        self.assertEqual(list(json.loads(self.out.getvalue())["songs"]), ["hum"])
        self.assertEqual(list(json.loads((one / "music.json").read_text(encoding="utf-8"))["songs"]), ["hum"])
        # Each document is read on its own.
        self.out = io.StringIO()
        self.assertEqual(self.build(both, "hum", "sting", "--rate", "8000"), 0)
        self.assertEqual(list(json.loads(self.out.getvalue())["songs"]), ["hum", "sting"])

    def test_a_name_may_come_after_a_flag(self) -> None:
        out = self.folder / "late"
        self.assertEqual(self.run_main("build", str(self.saved(LOOPS)), "--out", str(out), "--rate", "8000", "hum"), 0)
        self.assertEqual(sorted(found.name for found in out.iterdir()), ["hum.wav", "music.json"])

    def test_rate_is_the_samples_a_second_the_songs_are_built_at(self) -> None:
        out, apart = self.folder / "built", self.folder / "apart"
        self.assertEqual(self.build(out, "--rate", "8000"), 0)
        told = json.loads(self.out.getvalue())
        self.assertEqual((told["rate"], told["songs"]), (8000, build(Project.from_dict(LOOPS), apart, 8000)["songs"]))
        # Half a second and a fifth of one, at 8000 samples a second.
        self.assertEqual([entry["loop_samples"] for entry in told["songs"].values()], [4000, 1600])
        for name, entry in told["songs"].items():
            sides, rate = load_wav(out / entry["file"])
            self.assertEqual((rate, [len(side) for side in sides]), (8000, [entry["loop_samples"]] * 2), name)
            self.assertEqual((out / entry["file"]).read_bytes(), (apart / entry["file"]).read_bytes(), name)

    def test_build_of_a_missing_file_is_an_error_naming_it_with_code_2(self) -> None:
        path, out = self.folder / "nowhere.json", self.folder / "built"
        self.assertEqual(self.run_main("build", str(path), "--out", str(out)), 2)
        self.assertEqual(self.printed(), ("", f"error: {path}: no such file\n"))
        self.assertFalse(out.exists())

    def test_a_name_that_is_no_song_of_the_project_is_refused_with_code_2_and_no_manifest_is_written(self) -> None:
        out = self.folder / "built"
        self.assertEqual(self.build(out, "march"), 2)
        self.assertEqual(self.printed(), ("", "error: no song named 'march'\n"))
        self.assertFalse((out / "music.json").exists())

    def test_a_song_that_cannot_be_built_is_refused_by_its_name_with_code_2_and_no_manifest_is_written(self) -> None:
        spoilt = json.loads(json.dumps(LOOPS))
        spoilt["songs"]["hum"]["patterns"]["p"]["low"] = "o2 l4 ? d"
        out = self.folder / "built"
        self.assertEqual(self.build(out, "--rate", "8000", data=spoilt), 2)
        printed, err = self.printed()
        self.assertEqual(printed, "")
        self.assertTrue(err.startswith("error: songs.hum: pattern 'p', channel 'low': "), err)
        # The song before it in the project was written, and is all there is.
        self.assertEqual([found.name for found in out.iterdir()], ["sting.wav"])

    def test_build_without_out_is_refused_with_code_2(self) -> None:
        with self.assertRaises(SystemExit) as left:
            self.run_main("build", str(self.saved(LOOPS)))
        self.assertEqual(left.exception.code, 2)
        out, err = self.printed()
        self.assertEqual(out, "")
        self.assertIn("the following arguments are required: --out", err)


class Check(CommandLineCase):
    def project(self, **limits: float) -> dict:
        """A project of one sound of PROJECT, which keeps every rule at 44100 Hz, with these limits set on it."""
        return {"instruments": PROJECT["instruments"], "sounds": {"tap": {**PROJECT["sounds"]["tap"], **limits}}}

    def test_check_of_a_project_that_keeps_the_rules_prints_no_findings_with_code_0(self) -> None:
        self.assertEqual(self.run_main("check", str(self.saved(self.project()))), 0)
        self.assertEqual(self.printed(), ('{\n  "findings": []\n}\n', ""))

    def test_check_prints_the_finding_on_a_sound_that_breaks_a_rule_with_code_1(self) -> None:
        # The sound lasts longer than the hundredth of a second it is allowed here.
        data = self.project(max_seconds=0.01)
        self.assertEqual(self.run_main("check", str(self.saved(data))), 1)
        found = checks.run(Project.from_dict(data))
        told = {"findings": [{"rule": one.rule, "target": one.target, "message": one.message} for one in found]}
        self.assertEqual(self.printed(), (json.dumps(told, indent=2) + "\n", ""))
        self.assertEqual([(one.rule, one.target) for one in found], [("length", "sounds.tap")])

    def test_rate_is_the_samples_a_second_the_checks_render_at(self) -> None:
        data = self.project()
        self.assertEqual(self.run_main("check", str(self.saved(data)), "--rate", "8000"), 1)
        project = Project.from_dict(data)
        found = checks.run(project, 8000)
        told = {"findings": [one.to_dict() for one in found]}
        self.assertEqual(self.printed(), (json.dumps(told, indent=2) + "\n", ""))
        # The sound that keeps every rule at 44100 samples a second breaks one of them at 8000.
        self.assertEqual([(one.rule, one.target) for one in found], [("ends", "sounds.tap")])
        self.assertEqual(checks.run(project), [])

    def test_check_of_a_missing_file_is_an_error_naming_it_with_code_2(self) -> None:
        path = self.folder / "nowhere.json"
        self.assertEqual(self.run_main("check", str(path)), 2)
        self.assertEqual(self.printed(), ("", f"error: {path}: no such file\n"))


class Info(CommandLineCase):
    def test_info_prints_what_a_project_holds_by_name_in_the_order_of_the_file(self) -> None:
        self.assertEqual(self.run_main("info", str(self.saved(PROJECT))), 0)
        self.assertEqual(self.printed(), (json.dumps(INFO, indent=2) + "\n", ""))

    def test_info_of_a_layered_song_with_its_keys_left_out_gives_the_one_key_it_is_written_in(self) -> None:
        keyless = json.loads(json.dumps(PROJECT))
        del keyless["songs"]["round"]["layers"]["keys"]
        self.assertEqual(self.run_main("info", str(self.saved(keyless))), 0)
        self.assertEqual(json.loads(self.out.getvalue())["songs"]["round"]["layers"]["keys"], [0])

    def test_info_of_a_project_with_nothing_in_it_prints_every_table_empty(self) -> None:
        self.assertEqual(self.run_main("info", str(self.saved({}))), 0)
        self.assertEqual(json.loads(self.out.getvalue()), {"instruments": [], "sounds": {}, "songs": {}, "chords": []})

    def test_a_song_whose_text_cannot_be_read_is_an_error_naming_the_song(self) -> None:
        spoilt = json.loads(json.dumps(PROJECT))
        spoilt["songs"]["sting"]["patterns"]["x"]["one"] = "l8 c e ? g"
        self.assertEqual(self.run_main("info", str(self.saved(spoilt))), 2)
        out, err = self.printed()
        self.assertEqual(out, "")
        self.assertTrue(err.startswith("error: songs.sting: pattern 'x', channel 'one': "), err)


class Measure(CommandLineCase):
    def wav(self, samples: object, rate: int) -> Path:
        """The path of a WAV file holding `samples`."""
        path = self.folder / "sound.wav"
        save_wav(path, samples, rate)
        return path

    def named(self, name: str, *flags: str) -> int:
        """What main returns for measuring `name`, out of a project file holding PROJECT."""
        return self.run_main("measure", str(self.saved(PROJECT)), name, *flags)

    def test_measure_prints_the_numbers_of_a_mono_file(self) -> None:
        path = self.wav(tone(440, 0.5, 8000), 8000)
        self.assertEqual(self.run_main("measure", str(path)), 0)
        (row,), _ = load_wav(path)
        told = {"rate": 8000, "channels": 1, "seconds": 0.5,
                "measure": measure(row, 8000), "brightness": brightness(row, rate=8000)}
        self.assertEqual(self.printed(), (json.dumps(told, indent=2) + "\n", ""))
        # Half of full scale is 6 dB down, and a sine has all its energy at its own pitch.
        self.assertEqual((told["measure"]["peak_db"], told["brightness"]["centroid_hz"]), (-6.0, 440))

    def test_a_stereo_file_is_measured_as_the_mean_of_its_two_sides(self) -> None:
        path = self.wav([tone(440, 0.5, 16000), tone(1320, 0.25, 16000)], 16000)
        self.assertEqual(self.run_main("measure", str(path)), 0)
        (left, right), _ = load_wav(path)
        mean = (left.astype(np.float64) + right) / 2
        told = {"rate": 16000, "channels": 2, "seconds": 0.5,
                "measure": measure(mean, 16000), "brightness": brightness(mean, rate=16000)}
        self.assertEqual(self.printed(), (json.dumps(told, indent=2) + "\n", ""))
        for side in (left, right):
            self.assertNotEqual(measure(side, 16000), told["measure"])

    def test_a_silent_file_has_null_for_its_levels(self) -> None:
        self.assertEqual(self.run_main("measure", str(self.wav(np.zeros(800, dtype=np.float32), 8000))), 0)
        told = json.loads(self.out.getvalue())
        self.assertEqual((told["measure"]["peak_db"], told["measure"]["rms_db"]), (None, None))
        self.assertEqual(told["measure"]["seconds"], 0.1)

    def test_at_adds_the_pitch_sounding_at_that_time(self) -> None:
        # One note for half a second and then another.
        path = self.wav(np.concatenate((tone(440, 0.5, 8000), tone(660, 0.5, 8000))), 8000)
        self.assertEqual(self.run_main("measure", str(path), "--at", "0.6"), 0)
        told = json.loads(self.out.getvalue())
        self.assertEqual(list(told), ["rate", "channels", "seconds", "measure", "brightness", "pitch"])
        self.assertEqual(list(told["pitch"]), ["at", "hz"])
        self.assertEqual(told["pitch"]["at"], 0.6)
        self.assertAlmostEqual(told["pitch"]["hz"], 660, delta=1)

    def test_at_0_reads_the_pitch_the_file_starts_on(self) -> None:
        path = self.wav(np.concatenate((tone(440, 0.5, 8000), tone(660, 0.5, 8000))), 8000)
        self.assertEqual(self.run_main("measure", str(path), "--at", "0"), 0)
        self.assertAlmostEqual(json.loads(self.out.getvalue())["pitch"]["hz"], 440, delta=1)

    def test_at_below_0_is_refused_with_code_2(self) -> None:
        path = self.wav(tone(440, 0.5, 8000), 8000)
        self.assertEqual(self.run_main("measure", str(path), "--at", "-0.1"), 2)
        self.assertEqual(self.printed(),
                         ("", "error: --at must be 0 or more and under the file's 0.5 seconds, not -0.1\n"))

    def test_at_the_end_of_the_file_or_past_it_is_refused_with_code_2(self) -> None:
        path = self.wav(tone(440, 0.5, 8000), 8000)
        for at in ("0.5", "2"):
            self.assertEqual(self.run_main("measure", str(path), "--at", at), 2)
        self.assertEqual(self.printed(),
                         ("", "error: --at must be 0 or more and under the file's 0.5 seconds, not 0.5\n"
                              "error: --at must be 0 or more and under the file's 0.5 seconds, not 2.0\n"))

    def test_rate_variant_and_part_are_refused_with_a_wav_even_at_0_as_being_for_a_sound_or_a_song(self) -> None:
        path = self.wav(tone(440, 0.5, 8000), 8000)
        for flag, value in (("--rate", "0"), ("--variant", "0"), ("--part", "start")):
            self.assertEqual(self.run_main("measure", str(path), flag, value), 2)
        refused = f"is for a project's sound or song, and {path} with no name after it is measured as a WAV file"
        self.assertEqual(self.printed(),
                         ("", f"error: --rate {refused}\nerror: --variant {refused}\nerror: --part {refused}\n"))

    def test_a_sound_of_a_project_is_measured_at_the_level_the_bank_gives_it_and_no_file_is_written(self) -> None:
        self.assertEqual(self.named("tap"), 0)
        (buffer,) = render_entry(Project.from_dict(PROJECT).sound("tap"))
        told = {"name": "tap", "kind": "sound", "rate": 44100, "channels": 1, "seconds": len(buffer) / 44100,
                "measure": measure(buffer, 44100), "brightness": brightness(buffer, rate=44100)}
        self.assertEqual(self.printed(), (json.dumps(told, indent=2) + "\n", ""))
        # A bank brings the loudest sample of every sound to 3 dB under full scale. The sound is longer than a
        # frame of the brightness, so that is read from it and not left at zero.
        self.assertEqual(told["measure"]["peak_db"], -3.0)
        self.assertGreater(told["brightness"]["centroid_hz"], 0)
        self.assertEqual([found.name for found in self.folder.iterdir()], ["project.json"])

    def test_variant_picks_one_of_a_sound_counted_from_0_and_the_first_is_measured_without_it(self) -> None:
        self.assertEqual(self.named("drip"), 0)
        self.assertEqual(self.named("drip", "--variant", "2"), 0)
        buffers = render_entry(Project.from_dict(PROJECT).sound("drip"))
        told = [{"name": "drip", "kind": "sound", "rate": 44100, "channels": 1, "seconds": len(buffer) / 44100,
                 "measure": measure(buffer, 44100), "brightness": brightness(buffer, rate=44100)}
                for buffer in (buffers[0], buffers[2])]
        self.assertEqual(self.printed(), ("".join(json.dumps(one, indent=2) + "\n" for one in told), ""))
        # Each variant has a level of its own under the bank's peak of -3 dB: the first is 2 dB down, the third 4.
        self.assertEqual([one["measure"]["peak_db"] for one in told], [-5.0, -7.0])

    def test_a_held_sound_is_measured_as_its_start_when_no_part_is_asked_for(self) -> None:
        self.assertEqual(self.named("drone"), 0)
        start, _, _ = render_entry(Project.from_dict(PROJECT).sound("drone"))
        # The start is 6 ticks, a tenth of a second.
        told = {"name": "drone", "kind": "sound", "part": "start", "rate": 44100, "channels": 1, "seconds": 0.1,
                "measure": measure(start, 44100), "brightness": brightness(start, rate=44100)}
        self.assertEqual(self.printed(), (json.dumps(told, indent=2) + "\n", ""))

    def test_part_picks_the_start_the_loop_or_the_end_of_a_held_sound_to_measure_by_name(self) -> None:
        parts = ("start", "loop", "end")
        for part in parts:
            self.assertEqual(self.named("drone", "--rate", "8000", "--part", part), 0)
        buffers = render_entry(Project.from_dict(PROJECT).sound("drone"), 8000)
        told = [{"name": "drone", "kind": "sound", "part": part, "rate": 8000, "channels": 1,
                 "seconds": len(buffer) / 8000, "measure": measure(buffer, 8000),
                 "brightness": brightness(buffer, rate=8000)}
                for part, buffer in zip(parts, buffers)]
        self.assertEqual(self.printed(), ("".join(json.dumps(one, indent=2) + "\n" for one in told), ""))
        # The start is 6 ticks and the loop 24. The end is a note of its own, and as long as neither.
        self.assertEqual([one["seconds"] for one in told[:2]], [0.1, 0.4])
        self.assertNotIn(told[2]["seconds"], (0.1, 0.4))

    def test_part_is_printed_for_a_held_sound_and_for_nothing_else_that_is_measured(self) -> None:
        has = {}
        for name in ("drone", "tap", "drip", "knock", "sting"):
            # Each document is read on its own.
            self.out = io.StringIO()
            self.assertEqual(self.named(name, "--rate", "8000"), 0)
            has[name] = "part" in json.loads(self.out.getvalue())
        self.assertEqual(has, {"drone": True, "tap": False, "drip": False, "knock": False, "sting": False})

    def test_a_song_of_a_project_is_measured_as_the_mean_of_its_two_sides(self) -> None:
        self.assertEqual(self.named("sting", "--rate", "8000"), 0)
        mix = render_song(Project.from_dict(PROJECT).song("sting"), rate=8000)
        mean = (mix["left"].astype(np.float64) + mix["right"]) / 2
        # The half second of the song's rows and the half second it runs on for after them.
        told = {"name": "sting", "kind": "song", "rate": 8000, "channels": 2, "seconds": 1.0,
                "measure": measure(mean, 8000), "brightness": brightness(mean, rate=8000)}
        self.assertEqual(self.printed(), (json.dumps(told, indent=2) + "\n", ""))
        for side in (mix["left"], mix["right"]):
            self.assertNotEqual(measure(side, 8000), told["measure"])

    def test_rate_is_the_samples_a_second_a_sound_is_rendered_at_to_be_measured(self) -> None:
        self.assertEqual(self.named("tap", "--rate", "8000"), 0)
        (buffer,) = render_entry(Project.from_dict(PROJECT).sound("tap"), 8000)
        told = json.loads(self.out.getvalue())
        self.assertEqual((told["rate"], told["seconds"], told["measure"]),
                         (8000, len(buffer) / 8000, measure(buffer, 8000)))

    def test_at_adds_the_pitch_of_a_sound_at_that_time(self) -> None:
        self.assertEqual(self.named("drip", "--at", "0.01"), 0)
        told = json.loads(self.out.getvalue())
        self.assertEqual(list(told), ["name", "kind", "rate", "channels", "seconds", "measure", "brightness", "pitch"])
        self.assertEqual(told["pitch"]["at"], 0.01)
        # The first variant plays e5, which is 7 semitones above the 440 Hz of a4.
        self.assertAlmostEqual(told["pitch"]["hz"], 440 * 2 ** (7 / 12), delta=1)

    def test_at_the_end_of_a_sound_or_of_a_song_is_refused_by_that_word_with_code_2(self) -> None:
        (buffer,) = render_entry(Project.from_dict(PROJECT).sound("tap"), 8000)
        seconds = len(buffer) / 8000
        self.assertEqual(self.named("tap", "--rate", "8000", "--at", str(seconds)), 2)
        self.assertEqual(self.named("sting", "--rate", "8000", "--at", "1"), 2)
        self.assertEqual(self.printed(),
                         ("", f"error: --at must be 0 or more and under the sound's {seconds} seconds, not {seconds}\n"
                              "error: --at must be 0 or more and under the song's 1.0 seconds, not 1.0\n"))

    def test_a_name_that_is_no_sound_and_no_song_of_the_project_is_not_measured(self) -> None:
        self.assertEqual(self.named("pipe"), 2)
        self.assertEqual(self.printed(),
                         ("", f"error: 'pipe' is no sound and no song of {self.folder / 'project.json'}\n"))

    def test_a_name_after_the_file_makes_it_a_project_file_whatever_the_file_is_called(self) -> None:
        path = self.saved(PROJECT).rename(self.folder / "sounds.wav")
        self.assertEqual(self.run_main("measure", str(path), "tap", "--rate", "8000"), 0)
        self.assertEqual(json.loads(self.out.getvalue())["kind"], "sound")
        # Alone, the same file is read as a WAV, which it is not.
        self.assertEqual(self.run_main("measure", str(path)), 2)
        self.assertEqual(self.err.getvalue(), f"error: {path}: not a RIFF/WAVE file\n")

    def test_the_name_may_come_after_a_flag(self) -> None:
        path = str(self.saved(PROJECT))
        self.assertEqual(self.run_main("measure", path, "--rate", "8000", "tap"), 0)
        after = json.loads(self.out.getvalue())
        # Each document is read on its own.
        self.out = io.StringIO()
        self.assertEqual(self.run_main("measure", "--rate", "8000", path, "tap"), 0)
        self.assertEqual(json.loads(self.out.getvalue()), after)
        self.assertEqual((after["name"], after["rate"]), ("tap", 8000))

    def test_three_positionals_are_refused_with_code_2(self) -> None:
        with self.assertRaises(SystemExit) as left:
            self.named("tap", "drip")
        self.assertEqual(left.exception.code, 2)
        out, err = self.printed()
        self.assertEqual(out, "")
        self.assertIn("unrecognized arguments: drip", err)

    def test_no_positional_is_refused_with_code_2_showing_the_two_forms_of_the_command(self) -> None:
        with self.assertRaises(SystemExit) as left:
            self.run_main("measure")
        self.assertEqual(left.exception.code, 2)
        out, err = self.printed()
        self.assertEqual(out, "")
        self.assertTrue(err.startswith(
            "usage: rowsynth measure [-h] WAV [--at SECONDS]\n"
            "       rowsynth measure [-h] FILE NAME [--rate N] [--variant N] [--part {start,loop,end}] "
            "[--at SECONDS]\n"), err)


class Page(CommandLineCase):
    # The audio files of PROJECT's page: one for a sound of one def and one for a sound of parts, one a variant,
    # and the start, the loop and the end of a held sound.
    FILES = ["tap.wav", "drip-0.wav", "drip-1.wav", "drip-2.wav", "knock.wav",
             "drone-start.wav", "drone-loop.wav", "drone-end.wav"]
    # Those of its songs, in a folder of their own inside that of the sounds: the loop of each, and each
    # version of the layered one in each of its keys.
    SONG_FILES = ["round.wav", "round.whole.0.wav", "round.whole.3.wav", "round.bare.0.wav", "round.bare.3.wav",
                  "sting.wav"]

    def page(self, out: str | Path, *flags: str, data: dict = PROJECT) -> int:
        """What main returns for writing the listen page of a project file holding `data` into the folder `out`."""
        return self.run_main("page", str(self.saved(data)), "--out", str(out), *flags)

    def test_page_prints_the_page_the_folder_as_it_was_typed_the_rate_and_how_much_is_on_the_page(self) -> None:
        # The folder is typed with a separator after it, which the path of the page inside it has no use for.
        typed = f"{self.folder / 'site'}{os.sep}"
        self.assertEqual(self.page(typed), 0)
        told = {"page": str(self.folder / "site" / "index.html"), "folder": typed, "rate": 48000, "sounds": 4,
                "songs": 2, "files": len(self.FILES) + len(self.SONG_FILES)}
        self.assertEqual(self.printed(), (json.dumps(told, indent=2, ensure_ascii=False) + "\n", ""))

    def test_the_page_and_its_audio_files_are_in_a_folder_the_command_makes(self) -> None:
        out = self.folder / "made" / "site"
        self.assertEqual(self.page(out, "--rate", "8000"), 0)
        self.assertEqual(sorted(found.name for found in out.iterdir()), ["audio", "index.html"])
        self.assertEqual(sorted(found.name for found in (out / "audio").iterdir()), sorted([*self.FILES, "songs"]))
        self.assertEqual(sorted(found.name for found in (out / "audio" / "songs").iterdir()), sorted(self.SONG_FILES))
        told = json.loads(self.out.getvalue())
        self.assertEqual(Path(told["page"]), out / "index.html")
        self.assertEqual(told["files"], len(self.FILES) + len(self.SONG_FILES))

    def test_rate_is_the_samples_a_second_the_audio_files_are_rendered_at(self) -> None:
        out = self.folder / "site"
        self.assertEqual(self.page(out, "--rate", "8000"), 0)
        (buffer,) = render_entry(Project.from_dict(PROJECT).sound("tap"), 8000)
        (row,), rate = load_wav(out / "audio" / "tap.wav")
        self.assertEqual((rate, len(row)), (8000, len(buffer)))
        self.assertLess(float(np.abs(row - buffer).max()), STEP)
        self.assertEqual(json.loads(self.out.getvalue())["rate"], 8000)

    def test_a_project_with_no_sounds_gets_a_page_and_no_audio_files(self) -> None:
        out = self.folder / "site"
        self.assertEqual(self.page(out, data={}), 0)
        told = {"page": str(out / "index.html"), "folder": str(out), "rate": 48000, "sounds": 0, "songs": 0,
                "files": 0}
        self.assertEqual(self.printed(), (json.dumps(told, indent=2, ensure_ascii=False) + "\n", ""))
        self.assertEqual([found.name for found in out.iterdir()], ["index.html"])

    def test_page_of_a_missing_file_is_an_error_naming_it_with_code_2(self) -> None:
        path, out = self.folder / "nowhere.json", self.folder / "site"
        self.assertEqual(self.run_main("page", str(path), "--out", str(out)), 2)
        self.assertEqual(self.printed(), ("", f"error: {path}: no such file\n"))
        self.assertFalse(out.exists())

    def test_a_sound_that_cannot_be_rendered_is_refused_by_its_name_and_no_page_is_written(self) -> None:
        # A held sound whose note is too short for its start and its loop. The file loads all the same: loading
        # holds a sound to its shape, and does not render it.
        held = {**PROJECT["sounds"]["drone"]["held"], "note": blip("pipe", "d2")}
        data = {"instruments": PROJECT["instruments"],
                "sounds": {"tap": PROJECT["sounds"]["tap"], "drone": {"held": held}}}
        out = self.folder / "site"
        self.assertEqual(self.page(out, "--rate", "8000", data=data), 2)
        printed, err = self.printed()
        self.assertEqual(printed, "")
        self.assertTrue(err.startswith("error: drone: held: note: must be long enough for the start and the loop: "),
                        err)
        self.assertFalse(out.exists())

    def test_page_without_out_is_refused_with_code_2(self) -> None:
        with self.assertRaises(SystemExit) as left:
            self.run_main("page", str(self.saved(PROJECT)))
        self.assertEqual(left.exception.code, 2)
        out, err = self.printed()
        self.assertEqual(out, "")
        self.assertIn("the following arguments are required: --out", err)


class Render(CommandLineCase):
    def render(self, name: str, out: Path, *flags: str, data: dict = PROJECT) -> int:
        """What main returns for rendering `name`, out of a project file holding `data`, to `out`."""
        return self.run_main("render", str(self.saved(data)), name, "--out", str(out), *flags)

    def refusal(self, name: str, *flags: str, data: dict = PROJECT) -> str:
        """The errors printed so far, after a rendering that is refused with code 2 and writes nothing."""
        out = self.folder / "refused.wav"
        self.assertEqual(self.render(name, out, *flags, data=data), 2)
        self.assertEqual(self.out.getvalue(), "")
        self.assertFalse(out.exists())
        return self.err.getvalue()

    def test_a_sound_is_written_as_a_mono_file_at_the_level_the_bank_gives_it(self) -> None:
        path = self.folder / "made" / "tap.wav"
        self.assertEqual(self.render("tap", path), 0)
        (buffer,) = render_entry(Project.from_dict(PROJECT).sound("tap"))
        (row,), rate = load_wav(path)
        self.assertEqual((rate, len(row)), (44100, len(buffer)))
        self.assertLess(float(np.abs(row - buffer).max()), STEP)
        told = {"file": str(path), "rate": 44100, "channels": 1, "seconds": len(buffer) / 44100,
                "measure": measure(buffer, 44100)}
        self.assertEqual(self.printed(), (json.dumps(told, indent=2, ensure_ascii=False) + "\n", ""))
        # A bank brings the loudest sample of every sound to 3 dB under full scale.
        self.assertEqual(told["measure"]["peak_db"], -3.0)

    def test_variant_picks_one_of_a_sound_counted_from_0_and_the_first_is_written_without_it(self) -> None:
        buffers = render_entry(Project.from_dict(PROJECT).sound("drip"), 8000)
        first, third = self.folder / "first.wav", self.folder / "third.wav"
        self.assertEqual(self.render("drip", first, "--rate", "8000"), 0)
        self.assertEqual(self.render("drip", third, "--rate", "8000", "--variant", "2"), 0)
        # Each variant has a level of its own under the bank's peak of -3 dB: the first is 2 dB down, the third 4.
        for path, buffer, peak in ((first, buffers[0], -5.0), (third, buffers[2], -7.0)):
            (row,), _ = load_wav(path)
            self.assertEqual(len(row), len(buffer))
            self.assertLess(float(np.abs(row - buffer).max()), STEP)
            self.assertEqual(measure(row, 8000)["peak_db"], peak)

    def test_a_held_sound_is_written_as_its_start_when_no_part_is_asked_for(self) -> None:
        path = self.folder / "drone.wav"
        self.assertEqual(self.render("drone", path), 0)
        start, _, _ = render_entry(Project.from_dict(PROJECT).sound("drone"))
        (row,), rate = load_wav(path)
        self.assertEqual((rate, len(row)), (44100, len(start)))
        self.assertLess(float(np.abs(row - start).max()), STEP)
        # The start is 6 ticks, a tenth of a second.
        told = {"file": str(path), "part": "start", "rate": 44100, "channels": 1, "seconds": 0.1,
                "measure": measure(start, 44100)}
        self.assertEqual(self.printed(), (json.dumps(told, indent=2, ensure_ascii=False) + "\n", ""))

    def test_part_picks_the_start_the_loop_or_the_end_of_a_held_sound_to_write_by_name(self) -> None:
        buffers = render_entry(Project.from_dict(PROJECT).sound("drone"), 8000)
        # No two of the three are of one length, so a file that held another part than was asked for would not pass.
        self.assertEqual(len({len(buffer) for buffer in buffers}), 3)
        told = []
        for part, buffer in zip(("start", "loop", "end"), buffers):
            path = self.folder / f"{part}.wav"
            self.assertEqual(self.render("drone", path, "--rate", "8000", "--part", part), 0)
            (row,), _ = load_wav(path)
            self.assertEqual(len(row), len(buffer))
            self.assertLess(float(np.abs(row - buffer).max()), STEP)
            told.append({"file": str(path), "part": part, "rate": 8000, "channels": 1,
                         "seconds": len(buffer) / 8000, "measure": measure(buffer, 8000)})
        self.assertEqual(self.printed(),
                         ("".join(json.dumps(one, indent=2, ensure_ascii=False) + "\n" for one in told), ""))

    def test_part_is_printed_for_a_held_sound_and_for_nothing_else_that_is_written(self) -> None:
        has = {}
        for name in ("drone", "tap", "drip", "knock", "sting"):
            # Each document is read on its own.
            self.out = io.StringIO()
            self.assertEqual(self.render(name, self.folder / f"{name}.wav", "--rate", "8000"), 0)
            has[name] = "part" in json.loads(self.out.getvalue())
        self.assertEqual(has, {"drone": True, "tap": False, "drip": False, "knock": False, "sting": False})

    def test_rate_is_the_samples_a_second_a_file_is_rendered_at(self) -> None:
        path = self.folder / "tap.wav"
        self.assertEqual(self.render("tap", path, "--rate", "8000"), 0)
        (buffer,) = render_entry(Project.from_dict(PROJECT).sound("tap"), 8000)
        (row,), rate = load_wav(path)
        self.assertEqual((rate, len(row)), (8000, len(buffer)))
        self.assertLess(float(np.abs(row - buffer).max()), STEP)
        told = json.loads(self.out.getvalue())
        self.assertEqual((told["rate"], told["seconds"], told["measure"]),
                         (8000, len(buffer) / 8000, measure(buffer, 8000)))

    def test_a_song_is_written_as_a_stereo_file_and_measured_as_the_mean_of_its_two_sides(self) -> None:
        path = self.folder / "sting.wav"
        self.assertEqual(self.render("sting", path, "--rate", "8000"), 0)
        mix = render_song(Project.from_dict(PROJECT).song("sting"), rate=8000)
        # The sides are not alike, so a file that had them crossed, or one of them twice, would not pass.
        self.assertGreater(float(np.abs(mix["left"] - mix["right"]).max()), 0.01)
        sides, rate = load_wav(path)
        self.assertEqual((rate, len(sides)), (8000, 2))
        for side, rendered in zip(sides, (mix["left"], mix["right"])):
            self.assertEqual(len(side), len(rendered))
            self.assertLess(float(np.abs(side - rendered).max()), STEP)
        mean = (mix["left"].astype(np.float64) + mix["right"]) / 2
        # The file is the half second of the song's rows and the half second it runs on for after them.
        told = {"file": str(path), "rate": 8000, "channels": 2, "seconds": 1.0, "measure": measure(mean, 8000)}
        self.assertEqual(self.printed(), (json.dumps(told, indent=2, ensure_ascii=False) + "\n", ""))

    def test_stems_writes_a_song_as_one_mono_loop_a_channel_into_a_folder_it_makes(self) -> None:
        folder = self.folder / "loops" / "sting"
        self.assertEqual(self.render("sting", folder, "--stems", "--rate", "8000"), 0)
        self.assertEqual(sorted(found.name for found in folder.iterdir()), ["one.wav", "under.wav"])
        files = []
        for stem in render_stems(Project.from_dict(PROJECT).song("sting"), 8000)["stems"]:
            file = f"{stem['name']}.wav"
            (row,), rate = load_wav(folder / file)
            self.assertEqual((rate, len(row)), (8000, 4000))
            self.assertLess(float(np.abs(row - stem["samples"]).max()), STEP)
            files.append({"channel": stem["name"], "file": file, "pan": stem["pan"],
                          "measure": measure(stem["samples"], 8000)})
        told = {"folder": str(folder), "rate": 8000, "loop_samples": 4000, "seconds": 0.5, "files": files}
        self.assertEqual(self.printed(), (json.dumps(told, indent=2, ensure_ascii=False) + "\n", ""))
        self.assertEqual([(file["channel"], file["pan"]) for file in files], [("one", -0.5), ("under", 0)])

    def test_a_name_that_is_both_a_sound_and_a_song_is_refused_with_code_2(self) -> None:
        both = {**PROJECT, "sounds": {**PROJECT["sounds"], "sting": PROJECT["sounds"]["tap"]}}
        self.assertEqual(self.refusal("sting", data=both),
                         f"error: 'sting' is both a sound and a song of {self.folder / 'project.json'}\n")

    def test_a_name_that_is_neither_a_sound_nor_a_song_is_refused_with_code_2(self) -> None:
        self.assertEqual(self.refusal("pipe"),
                         f"error: 'pipe' is no sound and no song of {self.folder / 'project.json'}\n")

    def test_a_variant_the_sound_does_not_have_is_refused_saying_how_many_it_has(self) -> None:
        self.refusal("drip", "--rate", "8000", "--variant", "3")
        self.assertEqual(self.refusal("drip", "--rate", "8000", "--variant", "-1"),
                         "error: 'drip' has no variant 3: it has 3, counted from 0\n"
                         "error: 'drip' has no variant -1: it has 3, counted from 0\n")

    def test_variant_with_a_song_is_refused_with_code_2_even_when_it_is_0(self) -> None:
        self.assertEqual(self.refusal("sting", "--variant", "0"),
                         "error: --variant is for a sound, and 'sting' is a song\n")

    def test_part_with_a_sound_that_is_not_held_is_refused_with_code_2_whatever_form_it_has(self) -> None:
        for name in ("tap", "drip"):
            self.refusal(name, "--part", "start")
        self.assertEqual(self.refusal("knock", "--part", "loop"),
                         "error: --part is for a held sound, and 'tap' is not one\n"
                         "error: --part is for a held sound, and 'drip' is not one\n"
                         "error: --part is for a held sound, and 'knock' is not one\n")

    def test_part_with_a_song_is_refused_with_code_2(self) -> None:
        self.assertEqual(self.refusal("sting", "--part", "loop"),
                         "error: --part is for a held sound, and 'sting' is a song\n")

    def test_variant_with_a_held_sound_is_refused_with_code_2_saying_to_use_part_even_when_it_is_0(self) -> None:
        self.assertEqual(self.refusal("drone", "--variant", "0"),
                         "error: --variant is for a sound that is not held, and 'drone' is a held sound: "
                         "use --part\n")

    def test_a_part_that_is_none_of_the_three_is_refused_with_code_2(self) -> None:
        with self.assertRaises(SystemExit) as left:
            self.render("drone", self.folder / "middle.wav", "--part", "middle")
        self.assertEqual(left.exception.code, 2)
        out, err = self.printed()
        self.assertEqual(out, "")
        self.assertIn("invalid choice: 'middle'", err)

    def test_stems_with_a_sound_is_refused_with_code_2(self) -> None:
        self.assertEqual(self.refusal("tap", "--stems"), "error: --stems is for a song, and 'tap' is a sound\n")


class Refusals(CommandLineCase):
    def test_a_missing_file_is_an_error_naming_it_with_code_2(self) -> None:
        path = self.folder / "nowhere.json"
        self.assertEqual(self.run_main("info", str(path)), 2)
        self.assertEqual(self.printed(), ("", f"error: {path}: no such file\n"))

    def test_a_file_that_is_not_json_is_an_error_naming_it_with_code_2(self) -> None:
        path = self.folder / "notes.json"
        path.write_text('{"instruments": {"pipe": }}\n', encoding="utf-8")
        self.assertEqual(self.run_main("info", str(path)), 2)
        out, err = self.printed()
        self.assertEqual(out, "")
        self.assertTrue(err.startswith(f"error: {path}: line 1, column 26: "), err)

    def test_a_file_that_is_not_a_project_is_an_error_saying_where_with_code_2(self) -> None:
        path = self.saved({"instruments": {"pipe": {"wave": "tri", "loud": 1}}})
        self.assertEqual(self.run_main("info", str(path)), 2)
        self.assertEqual(self.printed(), ("", "error: instruments.pipe: unknown key 'loud'\n"))

    def test_a_file_that_cannot_be_opened_is_an_error_with_code_2(self) -> None:
        self.assertEqual(self.run_main("info", str(self.folder)), 2)
        out, err = self.printed()
        self.assertEqual(out, "")
        self.assertTrue(err.startswith("error: ") and err.endswith("\n"), err)

    def test_a_missing_wav_is_an_error_naming_it_with_code_2(self) -> None:
        path = self.folder / "nowhere.wav"
        self.assertEqual(self.run_main("measure", str(path)), 2)
        self.assertEqual(self.printed(), ("", f"error: {path}: no such file\n"))

    def test_a_file_that_is_not_a_wav_is_an_error_naming_it_with_code_2(self) -> None:
        path = self.folder / "notes.wav"
        path.write_text("o4 l8 c e g\n", encoding="utf-8")
        self.assertEqual(self.run_main("measure", str(path)), 2)
        self.assertEqual(self.printed(), ("", f"error: {path}: not a RIFF/WAVE file\n"))

    def test_an_unknown_command_is_refused_with_code_2(self) -> None:
        with self.assertRaises(SystemExit) as left:
            self.run_main("play", "project.json")
        self.assertEqual(left.exception.code, 2)
        out, err = self.printed()
        self.assertEqual(out, "")
        self.assertIn("invalid choice: 'play'", err)

    def test_no_command_prints_the_help_to_standard_error_with_code_2(self) -> None:
        self.assertEqual(self.run_main(), 2)
        out, err = self.printed()
        self.assertEqual(out, "")
        self.assertTrue(err.startswith("usage: rowsynth "), err)

    def test_the_help_lists_the_commands_in_name_order(self) -> None:
        self.run_main()
        # A command's line starts four spaces in. Where its help runs on to a second line, that one starts further in.
        self.assertEqual(re.findall(r"(?m)^ {4}(\w+) ", self.err.getvalue()),
                         ["build", "check", "info", "measure", "page", "render"])


class Process(CommandLineCase):
    def run_process(self, *argv: str, encoding: str) -> subprocess.CompletedProcess:
        """`python -m rowsynth` run as a process of its own, whose output Python would write in `encoding`."""
        above = Path(rowsynth.__file__).resolve().parents[1]
        return subprocess.run([sys.executable, "-m", "rowsynth", *argv], cwd=above, capture_output=True,
                              env={**os.environ, "PYTHONIOENCODING": encoding})

    def test_python_m_rowsynth_info_prints_the_same_json_as_a_process(self) -> None:
        done = self.run_process("info", str(self.saved(PROJECT)), encoding="utf-8")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.dumps(json.loads(done.stdout)), json.dumps(INFO))

    def test_a_process_prints_utf_8_whatever_encoding_its_output_was_given(self) -> None:
        named = {"instruments": {"kő": {"wave": "tri", "vol": 12}}}
        done = self.run_process("info", str(self.saved(named)), encoding="ascii")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(done.stdout.decode("utf-8"))["instruments"], ["kő"])

    def test_a_process_names_a_missing_file_in_utf_8_too(self) -> None:
        path = self.folder / "kő.json"
        done = self.run_process("info", str(path), encoding="ascii")
        self.assertEqual(done.returncode, 2)
        self.assertEqual(done.stderr.decode("utf-8").splitlines(), [f"error: {path}: no such file"])
