"""Tests for the project file."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from rowsynth.errors import RowSynthError
from rowsynth.project import Project

INSTRUMENTS = {
    "reed": {
        "wave": "pulse",
        "vol": [15, 12, 10, 8],
        "duty": {"v": [2, 1, 0, 1], "loop": 1},
        "arp": {"v": [0, 4, 7], "loop": 0},
        "pitch": [0, -0.5],
        "vib": {"delay": 8, "depth": 0.3, "rate": 5.5},
        "slide": -2,
        "release": 4,
        "gain": 0.8,
        "lp": [4000, 2500, 1200],
        "res": 0.4,
        "key_scale": -1.5,
        "gate": 0.75,
    },
    "deep": {"wave": "tri", "vol": 15, "duty": 3},
    "buzz": {"wave": "saw", "lp": 1800, "vib": {"depth": 0.1}},
    "hiss": {"wave": "noise", "short": True, "vol": [12, 6, 2], "release": 1},
    "bell": {
        "wave": "fm",
        "alg": 5,
        "fb": 1.2,
        "ops": [
            {"mul": 1, "det": 0, "level": 1, "env": {"a": 0.001, "d": 0.3, "s": 0.2, "sd": 1.5, "r": 0.4}},
            {"mul": 3.5, "det": 2, "level": 0.6, "env": {"a": 0.001, "d": 0.1}},
            {},
        ],
    },
    "plain": {},
}
PROJECT = {"instruments": INSTRUMENTS}


class ProjectCase(unittest.TestCase):
    """What the tests below share: a refusal is an error with exactly this message."""

    def refused(self, data: object, message: str) -> None:
        with self.assertRaises(RowSynthError) as caught:
            Project.from_dict(data)
        self.assertEqual(str(caught.exception), message)

    def instrument_refused(self, inst: object, message: str) -> None:
        self.refused({"instruments": {"reed": inst}}, "instruments.reed" + message)


class Loading(ProjectCase):
    def test_a_project_using_every_instrument_key_loads(self) -> None:
        project = Project.from_dict(PROJECT)
        self.assertEqual(project.instruments, INSTRUMENTS)

    def test_a_project_with_nothing_in_it_loads(self) -> None:
        self.assertEqual(Project.from_dict({}).instruments, {})
        self.assertEqual(Project.from_dict({}).to_dict(), {})
        self.assertEqual(Project().to_dict(), {})

    def test_to_dict_gives_back_what_was_loaded_with_its_key_order(self) -> None:
        self.assertEqual(json.dumps(Project.from_dict(PROJECT).to_dict()), json.dumps(PROJECT))

    def test_a_project_made_from_its_own_dict_is_equal(self) -> None:
        project = Project.from_dict(PROJECT)
        self.assertEqual(Project.from_dict(project.to_dict()), project)

    def test_projects_holding_different_things_are_not_equal(self) -> None:
        other = copy.deepcopy(PROJECT)
        other["instruments"]["deep"]["vol"] = 14
        self.assertNotEqual(Project.from_dict(other), Project.from_dict(PROJECT))
        self.assertNotEqual(Project.from_dict(PROJECT), PROJECT)

    def test_an_instrument_added_after_loading_is_in_to_dict(self) -> None:
        project = Project.from_dict({})
        project.instruments["snap"] = {"wave": "noise"}
        self.assertEqual(project.to_dict(), {"instruments": {"snap": {"wave": "noise"}}})

    def test_the_project_shares_nothing_with_what_went_in_or_came_out(self) -> None:
        data = copy.deepcopy(PROJECT)
        project = Project.from_dict(data)
        data["instruments"]["reed"]["vol"].append(0)
        project.to_dict()["instruments"]["reed"]["wave"] = "saw"
        self.assertEqual(project.instruments, INSTRUMENTS)


class RefusedInstruments(ProjectCase):
    def test_a_project_that_is_not_a_dict_is_refused(self) -> None:
        self.refused([PROJECT], "project: must be a dict")

    def test_an_unknown_table_is_refused_by_name(self) -> None:
        self.refused({"instruments": {}, "tunes": {}}, "project: unknown key 'tunes'")

    def test_a_table_that_is_not_a_dict_is_refused(self) -> None:
        self.refused({"instruments": [{"wave": "saw"}]}, "instruments: must be a dict")

    def test_an_instrument_that_is_not_a_dict_is_refused(self) -> None:
        self.instrument_refused("saw", ": must be a dict")

    def test_an_unknown_instrument_key_is_refused_by_name(self) -> None:
        self.instrument_refused({"wave": "saw", "volume": 12}, ": unknown key 'volume'")
        self.instrument_refused({"keyScale": 1}, ": unknown key 'keyScale'")

    def test_a_wave_outside_the_five_is_refused(self) -> None:
        self.instrument_refused(
            {"wave": "square"}, ".wave: must be one of 'pulse', 'saw', 'tri', 'noise', 'fm', not 'square'")
        self.instrument_refused({"wave": 2}, ".wave: must be one of 'pulse', 'saw', 'tri', 'noise', 'fm', not 2")

    def test_a_step_table_value_that_is_not_a_number_is_refused_in_each_form(self) -> None:
        for key in ("vol", "arp", "pitch", "lp"):
            with self.subTest(key=key):
                self.instrument_refused({key: "loud"}, f".{key}: must be a number")
                self.instrument_refused({key: [1, None, 3]}, f".{key}[1]: must be a number")
                self.instrument_refused({key: {"v": [1, 2, True], "loop": 0}}, f".{key}.v[2]: must be a number")

    def test_a_looped_step_table_needs_its_list_and_its_loop(self) -> None:
        self.instrument_refused({"vol": {"v": [15, 8]}}, ".vol: missing key 'loop'")
        self.instrument_refused({"vol": {"loop": 0}}, ".vol: missing key 'v'")
        self.instrument_refused({"vol": {"v": [15, 8], "loop": 0, "hold": 1}}, ".vol: unknown key 'hold'")
        self.instrument_refused({"vol": {"v": 15, "loop": 0}}, ".vol.v: must be a list")
        self.instrument_refused({"vol": {"v": [15, 8], "loop": 0.5}}, ".vol.loop: must be a whole number")

    def test_a_duty_that_is_not_0_to_3_is_refused_in_each_form(self) -> None:
        self.instrument_refused({"duty": 4}, ".duty: must be a whole number from 0 to 3")
        self.instrument_refused({"duty": [0, 1.5]}, ".duty[1]: must be a whole number from 0 to 3")
        self.instrument_refused({"duty": {"v": [2, -1], "loop": 0}}, ".duty.v[1]: must be a whole number from 0 to 3")

    def test_a_plain_value_that_is_not_a_number_is_refused(self) -> None:
        for key in ("slide", "release", "gain", "res", "key_scale", "gate", "fb"):
            with self.subTest(key=key):
                self.instrument_refused({key: "1"}, f".{key}: must be a number")
                self.instrument_refused({key: [1]}, f".{key}: must be a number")
                self.instrument_refused({key: True}, f".{key}: must be a number")

    def test_short_that_is_not_true_or_false_is_refused(self) -> None:
        self.instrument_refused({"wave": "noise", "short": 1}, ".short: must be true or false")

    def test_a_vib_of_the_wrong_shape_is_refused(self) -> None:
        self.instrument_refused({"vib": 0.3}, ".vib: must be a dict")
        self.instrument_refused({"vib": {"delay": 4, "rate": 6}}, ".vib: missing key 'depth'")
        self.instrument_refused({"vib": {"depth": 0.3, "speed": 6}}, ".vib: unknown key 'speed'")
        for key in ("delay", "depth", "rate"):
            with self.subTest(key=key):
                self.instrument_refused({"vib": {"depth": 0.3, key: "slow"}}, f".vib.{key}: must be a number")

    def test_an_alg_that_is_not_0_to_7_is_refused(self) -> None:
        for alg in (8, -1, 2.0, "4", True):
            with self.subTest(alg=alg):
                self.instrument_refused({"wave": "fm", "alg": alg}, ".alg: must be a whole number from 0 to 7")

    def test_operators_of_the_wrong_shape_are_refused(self) -> None:
        self.instrument_refused({"ops": {"mul": 1}}, ".ops: must be a list")
        self.instrument_refused({"ops": [{}, {}, {}, {}, {}]}, ".ops: must hold four operators at most, not 5")
        self.instrument_refused({"ops": [{}, 2]}, ".ops[1]: must be a dict")
        self.instrument_refused({"ops": [{"mult": 2}]}, ".ops[0]: unknown key 'mult'")
        for key in ("mul", "det", "level"):
            with self.subTest(key=key):
                self.instrument_refused({"ops": [{}, {key: "2"}]}, f".ops[1].{key}: must be a number")

    def test_an_operator_envelope_of_the_wrong_shape_is_refused(self) -> None:
        self.instrument_refused({"ops": [{"env": [0.1, 0.2]}]}, ".ops[0].env: must be a dict")
        self.instrument_refused({"ops": [{"env": {"attack": 0.1}}]}, ".ops[0].env: unknown key 'attack'")
        for key in ("a", "d", "s", "sd", "r"):
            with self.subTest(key=key):
                self.instrument_refused({"ops": [{"env": {key: None}}]}, f".ops[0].env.{key}: must be a number")


class Files(ProjectCase):
    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)

    def test_a_saved_project_loads_back_equal(self) -> None:
        project = Project.from_dict(PROJECT)
        project.save(self.folder / "project.json")
        loaded = Project.load(self.folder / "project.json")
        self.assertEqual(loaded, project)
        self.assertEqual(json.dumps(loaded.to_dict()), json.dumps(PROJECT))

    def test_save_writes_an_indent_of_2_with_lf_ends_and_a_final_newline(self) -> None:
        Project.from_dict({"instruments": {"deep": {"wave": "tri", "vol": [15, 9]}}}).save(self.folder / "p.json")
        expected = (
            '{\n  "instruments": {\n    "deep": {\n      "wave": "tri",\n      "vol": [\n        15,\n        9\n'
            "      ]\n    }\n  }\n}\n"
        )
        self.assertEqual((self.folder / "p.json").read_bytes(), expected.encode("utf-8"))

    def test_save_makes_the_folders_above_the_file(self) -> None:
        path = self.folder / "songs" / "drafts" / "project.json"
        Project.from_dict(PROJECT).save(str(path))
        self.assertEqual(Project.load(str(path)), Project.from_dict(PROJECT))

    def test_load_reads_utf8_with_or_without_a_byte_order_mark(self) -> None:
        text = json.dumps({"instruments": {"fl\xfbte": {"wave": "tri"}}}, ensure_ascii=False).encode("utf-8")
        for name, mark in (("plain.json", b""), ("marked.json", b"\xef\xbb\xbf")):
            with self.subTest(name=name):
                (self.folder / name).write_bytes(mark + text)
                self.assertEqual(list(Project.load(self.folder / name).instruments), ["fl\xfbte"])

    def test_save_writes_utf8_without_a_byte_order_mark(self) -> None:
        Project.from_dict({"instruments": {"fl\xfbte": {}}}).save(self.folder / "p.json")
        written = (self.folder / "p.json").read_bytes()
        self.assertTrue(written.startswith(b"{"))
        self.assertIn("fl\xfbte".encode("utf-8"), written)

    def test_json_that_does_not_parse_is_refused_with_its_path_line_and_column(self) -> None:
        path = self.folder / "broken.json"
        path.write_bytes(b'{\n  "instruments": {\n    "deep": {"wave": }\n  }\n}\n')
        with self.assertRaises(RowSynthError) as caught:
            Project.load(path)
        self.assertTrue(str(caught.exception).startswith(f"{path}: line 3, column 22: "), str(caught.exception))

    def test_a_file_that_is_not_utf8_is_refused_with_its_path(self) -> None:
        path = self.folder / "latin.json"
        path.write_bytes(b'{"instruments": {"fl\xfbte": {}}}')
        with self.assertRaises(RowSynthError) as caught:
            Project.load(path)
        self.assertEqual(str(caught.exception), f"{path}: not UTF-8 text")

    def test_a_loaded_file_is_checked_like_any_other_data(self) -> None:
        path = self.folder / "p.json"
        path.write_text('{"instruments": {"deep": {"wave": "sine"}}}', encoding="utf-8")
        with self.assertRaises(RowSynthError) as caught:
            Project.load(path)
        self.assertEqual(
            str(caught.exception), "instruments.deep.wave: must be one of 'pulse', 'saw', 'tri', 'noise', 'fm', not 'sine'")
