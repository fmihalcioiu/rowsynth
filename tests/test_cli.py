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
from rowsynth.cli import main
from rowsynth.commands import emit, load_file, load_project
from rowsynth.errors import RowSynthError
from rowsynth.measure import brightness, measure
from rowsynth.project import Project
from rowsynth.wav import load_wav, save_wav


def blip(inst: str, note: object) -> dict:
    """A sound of one short note."""
    return {"voices": [{"inst": inst, "notes": [[0, note, 3]]}]}


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
        "drone": {
            "held": {"note": blip("pipe", "d2"), "start": 6, "loop": 24, "join": 3, "end": blip("pipe", "d2"),
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
        # Two eighths and a dotted quarter are 10 rows. At 3 ticks a row that is half a second.
        "sting": {
            "row_ticks": 3,
            "channels": {"one": {"inst": "pipe"}},
            "patterns": {"x": {"one": "l8 c e g4."}},
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
        "sting": {"channels": ["one"], "patterns": ["x"], "order": ["x"], "rows": 10.0, "seconds": 0.5},
    },
    "chords": ["Dm", "A7"],
}


def tone(hz: float, level: float, rate: int) -> np.ndarray:
    """Half a second of a sine that starts at zero."""
    return level * np.sin(2 * np.pi * hz * np.arange(rate // 2) / rate)


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


class Info(CommandLineCase):
    def test_info_prints_what_a_project_holds_by_name_in_the_order_of_the_file(self) -> None:
        self.assertEqual(self.run_main("info", str(self.saved(PROJECT))), 0)
        self.assertEqual(self.printed(), (json.dumps(INFO, indent=2) + "\n", ""))

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
        self.assertEqual(re.findall(r"(?m)^ {4}(\w+) ", self.err.getvalue()), ["info", "measure"])


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
