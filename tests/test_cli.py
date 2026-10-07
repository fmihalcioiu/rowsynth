"""Tests for the command line and its info command."""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import rowsynth
from rowsynth.cli import main
from rowsynth.commands import emit, load_project
from rowsynth.project import Project


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


class CommandLineCase(unittest.TestCase):
    """What the tests below share: a folder for project files, and main run with what it prints caught."""

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

    def test_load_project_gives_the_project_in_the_file(self) -> None:
        self.assertEqual(load_project(self.saved(PROJECT)), Project.from_dict(PROJECT))


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
        self.assertRegex(err, r"(?m)^\s+info\s")


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
