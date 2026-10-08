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
SOUNDS = {
    "coin": {
        "def": {
            "voices": [{"inst": "reed", "notes": [[0, "b5", 4], [4, "e6", 10, 0.8]]}],
            "gain": 0.4,
            "echo": {"time": 0.12, "feedback": 0.25, "mix": 0.2, "lp": 2500},
        },
        "db": -3,
        "meta": {"family": "pickups", "tags": ["short", "bright"]},
        "check": [{"at": 0.02, "note": "b5"}, {"at": 0.1, "note": 88}],
        "max_seconds": 0.6,
        "max_dc": 0.004,
        "max_bright": 0.5,
    },
    "step": {
        "defs": [
            {"voices": [{"inst": "hiss", "notes": [[0, 60, 2]]}]},
            {
                "voices": [{"inst": {"wave": "noise", "vol": [10, 4]}, "notes": [[0, 64.5, 2]]}],
                "echo": {"time": [0.05, 0.08]},
            },
        ],
        "db": [-6, -8],
    },
    "door": {
        "parts": [
            [0, {"def": {"voices": [{"inst": "buzz", "notes": [[0, "Eb2", 12]]}]}, "db": -2}],
            [0.25, {"def": {"voices": [{"inst": "bell", "notes": [[0, "c#5", 20]]}]}}],
        ],
    },
    "hum": {
        "held": {
            "note": {"voices": [{"inst": "deep", "notes": [[0, "a2", 90]]}]},
            "start": 12,
            "loop": 48,
            "join": 6,
            "end": {"voices": [{"inst": "deep", "notes": [[0, "a2", 8]]}, {"inst": "hiss", "notes": []}]},
            "let_go": 4,
        },
        "db": -9,
    },
}
CHORDS = {
    "Am": {"root": 9, "tones": [9, 0, 4]},
    "F": {"root": 5, "tones": [5, 9, 0]},
    "G7": {"root": 7, "tones": [7, 11, 2, 5]},
    "C": {"root": 0, "tones": [0, 4, 7]},
}
SONGS = {
    "tune": {
        "row_ticks": 6,
        "rows_per_beat": 4,
        "gain": 0.3,
        "instruments": {
            "organ": {"wave": "saw", "vol": [9, 8, 7], "lp": 1500, "gate": 1},
            "deep": {"wave": "pulse", "duty": 1, "vol": 13},
            "glass": "bell",
            "thud": {"wave": "tri", "pitch": [0, -6, -12], "vol": [15, 10, 0]},
        },
        "drums": {
            "k": {"inst": "thud", "note": "c2"},
            "s": {"inst": "hiss", "note": 62, "rows": 2, "vel": 0.8},
            "h": {"inst": "hiss", "note": "a6", "rows": 1},
        },
        "channels": {
            "melody": {"inst": "reed", "gain": 0.9, "pan": 0.2, "echo": {"rows": 3, "gain": 0.4}},
            "under": {"inst": "deep", "gain": 1, "pan": -0.1},
            "wash": {"inst": "organ"},
            "taps": {"inst": "thud", "gain": 0.8, "drums": True},
        },
        "patterns": {
            "a": {
                "melody": "o4 l8 a > c e c < b > c e4 | @glass f4 a4 @reed g2",
                "under": "o2 l4 a a a a | f f g g",
                "wash": "o3 l1 a | f2 g2",
                "taps": "l8 k h s h k h s h | k h s h k k s4",
            },
            "b": {
                "melody": "o5 l4 c e g e | d < b > d2",
                "under": "o2 l2 c c | g g",
                "wash": "o3 l1 c | g",
                "taps": "l4 k s k s | k s k8 k8 s",
            },
            "lead_in": {"melody": "o4 l16 e g a > c e4 r2", "taps": "l4 r r r s8 s8"},
        },
        "order": ["a", "b", "a"],
        "chords": {"a": ["Am", "F G7"], "b": ["C", "G7"]},
        "layers": {
            "groups": {"kit": ["taps"], "roots": ["under"], "upper": ["melody", "wash"]},
            "versions": {"sparse": ["kit", "roots"], "dense": ["kit", "roots", "upper"]},
            "keys": [0, 2],
            "premix": [{"version": "sparse", "key": 0}, {"version": "dense", "key": 2}],
        },
        "intro": {"order": ["lead_in"], "tail": 1.5, "fade": 0.05},
    },
    "jingle": {
        "row_ticks": 5,
        "channels": {"one": {"inst": "reed"}},
        "patterns": {"x": {"one": "l8 c e g > c4."}},
        "order": ["x"],
    },
}
PROJECT = {"instruments": INSTRUMENTS, "chords": CHORDS, "sounds": SOUNDS, "songs": SONGS}


def sound_of(inst: object = "reed", notes: object = None, **more: object) -> dict:
    """A sound of one voice, for the tests that spoil one part of it."""
    return {"voices": [{"inst": inst, "notes": [[0, "c4", 6]] if notes is None else notes}], **more}


def song_of(**changes: object) -> dict:
    """A song of two channels and two patterns, for the tests that spoil one part of it."""
    song = {
        "row_ticks": 6,
        "drums": {"k": {"inst": "hiss", "note": "c2"}, "s": {"inst": "hiss", "note": "c4"}},
        "channels": {"one": {"inst": "reed"}, "kit": {"inst": "hiss", "drums": True}},
        "patterns": {"x": {"one": "o4 l4 c e g e", "kit": "l4 k s k s"}, "y": {"one": "o4 l1 c"}},
        "order": ["x", "y", "x"],
    }
    return {**song, **changes}


class ProjectCase(unittest.TestCase):
    """What the tests below share: a refusal is an error with exactly this message."""

    def refused(self, data: object, message: str) -> None:
        with self.assertRaises(RowSynthError) as caught:
            Project.from_dict(data)
        self.assertEqual(str(caught.exception), message)

    def instrument_refused(self, inst: object, message: str) -> None:
        self.refused({"instruments": {"reed": inst}}, "instruments.reed" + message)

    def sound_refused(self, entry: object, message: str) -> None:
        self.refused({"instruments": INSTRUMENTS, "sounds": {"coin": entry}}, "sounds.coin" + message)

    def song_refused(self, song: object, message: str) -> None:
        self.refused({"instruments": INSTRUMENTS, "chords": CHORDS, "songs": {"tune": song}}, "songs.tune" + message)


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

    def test_to_dict_keeps_the_tables_in_the_order_they_came_in(self) -> None:
        project = Project.from_dict({"sounds": SOUNDS, "instruments": INSTRUMENTS})
        self.assertEqual(list(project.to_dict()), ["sounds", "instruments"])

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


class Sounds(ProjectCase):
    def test_a_project_with_a_sound_of_each_form_loads(self) -> None:
        project = Project.from_dict(PROJECT)
        self.assertEqual(project.sounds, SOUNDS)
        self.assertEqual(list(project.sounds), ["coin", "step", "door", "hum"])

    def test_sound_writes_out_the_instrument_each_voice_names(self) -> None:
        project = Project.from_dict(PROJECT)
        self.assertEqual(project.sound("coin")["def"]["voices"][0]["inst"], INSTRUMENTS["reed"])
        self.assertEqual(project.sound("step")["defs"][0]["voices"][0]["inst"], INSTRUMENTS["hiss"])
        door = project.sound("door")
        self.assertEqual(door["parts"][0][1]["def"]["voices"][0]["inst"], INSTRUMENTS["buzz"])
        self.assertEqual(door["parts"][1][1]["def"]["voices"][0]["inst"], INSTRUMENTS["bell"])
        held = project.sound("hum")["held"]
        self.assertEqual(held["note"]["voices"][0]["inst"], INSTRUMENTS["deep"])
        self.assertEqual([voice["inst"] for voice in held["end"]["voices"]], [INSTRUMENTS["deep"], INSTRUMENTS["hiss"]])

    def test_sound_leaves_an_instrument_given_whole_as_it_is(self) -> None:
        step = Project.from_dict(PROJECT).sound("step")
        self.assertEqual(step["defs"][1]["voices"][0]["inst"], {"wave": "noise", "vol": [10, 4]})

    def test_sound_changes_nothing_but_the_instrument_names(self) -> None:
        coin = Project.from_dict(PROJECT).sound("coin")
        coin["def"]["voices"][0]["inst"] = "reed"
        self.assertEqual(json.dumps(coin), json.dumps(SOUNDS["coin"]))

    def test_changing_what_sound_returns_never_changes_the_project(self) -> None:
        project = Project.from_dict(PROJECT)
        coin = project.sound("coin")
        coin["def"]["voices"][0]["inst"]["vol"].append(0)
        coin["def"]["voices"][0]["notes"].clear()
        coin["meta"]["tags"].append("spoilt")
        step = project.sound("step")
        step["defs"][1]["voices"][0]["inst"]["wave"] = "saw"
        step["db"][0] = 0
        self.assertEqual(project.to_dict(), PROJECT)
        self.assertEqual(project.sound("coin")["def"]["voices"][0]["inst"], INSTRUMENTS["reed"])

    def test_two_voices_naming_one_instrument_get_a_dict_each(self) -> None:
        held = Project.from_dict(PROJECT).sound("hum")["held"]
        self.assertIsNot(held["note"]["voices"][0]["inst"], held["end"]["voices"][0]["inst"])

    def test_asking_for_a_sound_the_project_does_not_hold_is_refused(self) -> None:
        with self.assertRaises(RowSynthError) as caught:
            Project.from_dict(PROJECT).sound("splash")
        self.assertEqual(str(caught.exception), "no sound named 'splash'")

    def test_note_names_load_in_either_case_with_a_sharp_or_a_flat(self) -> None:
        notes = [[0, note, 4] for note in ("a4", "c#5", "eb3", "Bb0", "G9", 69, 60.5, 0, -3)]
        project = Project.from_dict({"instruments": INSTRUMENTS, "sounds": {"run": {"def": sound_of(notes=notes)}}})
        self.assertEqual(project.sounds["run"]["def"]["voices"][0]["notes"], notes)

    def test_anything_may_sit_inside_meta(self) -> None:
        meta = {"wave": "sine", "inst": "blip", "deep": [{"note": "h9"}, None]}
        project = Project.from_dict({"sounds": {"ping": {"def": {"voices": []}, "meta": meta}}})
        self.assertEqual(project.sound("ping")["meta"], meta)


class RefusedSounds(ProjectCase):
    def test_a_sounds_table_that_is_not_a_dict_is_refused(self) -> None:
        self.refused({"sounds": [{"def": sound_of()}]}, "sounds: must be a dict")

    def test_an_entry_that_is_not_a_dict_is_refused(self) -> None:
        self.sound_refused([sound_of()], ": must be a dict")

    def test_an_unknown_entry_key_is_refused_by_name(self) -> None:
        self.sound_refused({"def": sound_of(), "priority": 2}, ": unknown key 'priority'")
        self.sound_refused({"def": sound_of(), "maxSeconds": 2}, ": unknown key 'maxSeconds'")

    def test_an_entry_without_one_form_and_only_one_is_refused(self) -> None:
        message = ": must hold one of 'def', 'defs', 'parts' and 'held', and only one"
        self.sound_refused({"db": -3}, message)
        self.sound_refused({"def": sound_of(), "defs": [sound_of()]}, message)

    def test_a_voice_naming_an_instrument_the_project_does_not_hold_is_refused(self) -> None:
        self.sound_refused({"def": sound_of("blip")}, ".def.voices[0].inst: no instrument named 'blip'")

    def test_an_instrument_given_whole_in_a_voice_is_checked_like_any_other(self) -> None:
        self.sound_refused(
            {"def": sound_of({"wave": "fm", "alg": 9})}, ".def.voices[0].inst.alg: must be a whole number from 0 to 7")
        self.sound_refused({"def": sound_of({"vol": {"v": [15]}})}, ".def.voices[0].inst.vol: missing key 'loop'")
        self.sound_refused({"def": sound_of(7)}, ".def.voices[0].inst: must be a dict")

    def test_a_sound_of_the_wrong_shape_is_refused(self) -> None:
        self.sound_refused({"def": "beep"}, ".def: must be a dict")
        self.sound_refused({"def": {"gain": 0.5}}, ".def: missing key 'voices'")
        self.sound_refused({"def": sound_of(volume=1)}, ".def: unknown key 'volume'")
        self.sound_refused({"def": sound_of(gain="loud")}, ".def.gain: must be a number")
        self.sound_refused({"def": {"voices": {}}}, ".def.voices: must be a list")

    def test_a_voice_of_the_wrong_shape_is_refused(self) -> None:
        self.sound_refused({"def": {"voices": ["reed"]}}, ".def.voices[0]: must be a dict")
        self.sound_refused({"def": {"voices": [{"notes": []}]}}, ".def.voices[0]: missing key 'inst'")
        self.sound_refused({"def": {"voices": [{"inst": "reed"}]}}, ".def.voices[0]: missing key 'notes'")
        self.sound_refused(
            {"def": {"voices": [{"inst": "reed", "notes": [], "pan": 0}]}}, ".def.voices[0]: unknown key 'pan'")
        self.sound_refused({"def": sound_of(notes="c4")}, ".def.voices[0].notes: must be a list")

    def test_a_played_note_of_the_wrong_shape_is_refused(self) -> None:
        at = ".def.voices[0].notes[1]"
        shape = ": must be [tick, note, ticks] or [tick, note, ticks, velocity]"
        for note, message in (
            ("c4", ": must be a list"),
            ([0, 60], shape),
            ([0, 60, 4, 1, 0], shape),
            (["0", 60, 4], "[0]: must be a number"),
            ([0, 60, None], "[2]: must be a number"),
            ([0, 60, 4, "f"], "[3]: must be a number"),
        ):
            with self.subTest(note=note):
                self.sound_refused({"def": sound_of(notes=[[0, 60, 4], note])}, at + message)

    def test_a_note_that_is_neither_a_number_nor_a_name_is_refused(self) -> None:
        for note in ("h4", "c", "c44", "#4", "c--1", "cb#4", "4c", "", None, True, [60]):
            with self.subTest(note=note):
                self.sound_refused(
                    {"def": sound_of(notes=[[0, note, 4]])},
                    f".def.voices[0].notes[0][1]: must be a number or a note name such as 'c#4', not {note!r}")

    def test_a_name_the_note_reader_takes_is_a_note_here_too(self) -> None:
        for note in ("c-1", "A4", "eb3", "g#9", 60.5):
            with self.subTest(note=note):
                project = Project.from_dict(
                    {"instruments": INSTRUMENTS, "sounds": {"coin": {"def": sound_of(notes=[[0, note, 4]])}}})
                self.assertEqual(project.sound("coin")["def"]["voices"][0]["notes"][0][1], note)

    def test_an_echo_of_the_wrong_shape_is_refused(self) -> None:
        self.sound_refused({"def": sound_of(echo=0.2)}, ".def.echo: must be a dict")
        self.sound_refused({"def": sound_of(echo={"mix": 0.2})}, ".def.echo: missing key 'time'")
        self.sound_refused({"def": sound_of(echo={"time": 0.1, "wet": 1})}, ".def.echo: unknown key 'wet'")
        self.sound_refused({"def": sound_of(echo={"time": "0.1"})}, ".def.echo.time: must be a number")
        self.sound_refused({"def": sound_of(echo={"time": [0.1, None]})}, ".def.echo.time[1]: must be a number")
        for key in ("feedback", "mix", "lp"):
            with self.subTest(key=key):
                self.sound_refused(
                    {"def": sound_of(echo={"time": 0.1, key: [0.3]})}, f".def.echo.{key}: must be a number")

    def test_variants_of_the_wrong_shape_are_refused(self) -> None:
        self.sound_refused({"defs": sound_of()}, ".defs: must be a list")
        self.sound_refused({"defs": []}, ".defs: must hold at least one")
        self.sound_refused(
            {"defs": [sound_of(), sound_of("blip")]}, ".defs[1].voices[0].inst: no instrument named 'blip'")

    def test_a_level_of_the_wrong_shape_is_refused(self) -> None:
        self.sound_refused({"def": sound_of(), "db": "-3"}, ".db: must be a number")
        self.sound_refused({"defs": [sound_of(), sound_of()], "db": [-3, None]}, ".db[1]: must be a number")

    def test_a_list_of_levels_without_one_for_each_variant_is_refused(self) -> None:
        message = ".db: a list of levels must hold one for each of 'defs'"
        self.sound_refused({"defs": [sound_of(), sound_of()], "db": [-3]}, message)
        self.sound_refused({"def": sound_of(), "db": [-3]}, message)

    def test_parts_of_the_wrong_shape_are_refused(self) -> None:
        part = {"def": sound_of()}
        self.sound_refused({"parts": {"0": part}}, ".parts: must be a list")
        self.sound_refused({"parts": []}, ".parts: must hold at least one")
        self.sound_refused({"parts": [part]}, ".parts[0]: must be a list")
        self.sound_refused({"parts": [[0, part], [0.5]]}, ".parts[1]: must be [seconds, entry]")
        self.sound_refused({"parts": [[0, part, part]]}, ".parts[0]: must be [seconds, entry]")
        self.sound_refused({"parts": [["0", part]]}, ".parts[0][0]: must be a number")
        self.sound_refused({"parts": [[0, part], [-0.5, part]]}, ".parts[1][0]: must be 0 or more")

    def test_an_entry_inside_parts_is_checked_like_any_other(self) -> None:
        part = {"def": sound_of()}
        self.sound_refused(
            {"parts": [[0, part], [0.5, {"def": sound_of("blip")}]]},
            ".parts[1][1].def.voices[0].inst: no instrument named 'blip'")
        self.sound_refused(
            {"parts": [[0, {"db": 1}]]},
            ".parts[0][1]: must hold one of 'def', 'defs', 'parts' and 'held', and only one")

    def test_a_part_that_is_not_one_sound_is_refused(self) -> None:
        part = {"def": sound_of()}
        message = ".parts[1][1]: a part is one sound: it must hold 'def'"
        self.sound_refused({"parts": [[0, part], [0.5, {"defs": [sound_of(), sound_of()]}]]}, message)
        self.sound_refused({"parts": [[0, part], [0.5, {"parts": [[0, part]]}]]}, message)

    def test_a_level_on_an_entry_of_parts_is_refused(self) -> None:
        part = {"def": sound_of(), "db": -2}
        self.sound_refused(
            {"parts": [[0, part]], "db": -6}, ".db: an entry of parts has no level of its own: each part has its 'db'")

    def test_a_held_sound_of_the_wrong_shape_is_refused(self) -> None:
        held = {"note": sound_of(), "start": 10, "loop": 40, "join": 4, "end": sound_of(), "let_go": 3}
        self.sound_refused({"held": [held]}, ".held: must be a dict")
        self.sound_refused({"held": {**held, "fade": 2}}, ".held: unknown key 'fade'")
        for key in held:
            with self.subTest(missing=key):
                rest = {name: value for name, value in held.items() if name != key}
                self.sound_refused({"held": rest}, f".held: missing key {key!r}")
        for key in ("start", "loop", "join", "let_go"):
            with self.subTest(key=key):
                self.sound_refused({"held": {**held, key: "4"}}, f".held.{key}: must be a number")

    def test_the_sounds_of_a_held_sound_are_checked_like_any_other(self) -> None:
        held = {"note": sound_of(), "start": 10, "loop": 40, "join": 4, "end": sound_of(), "let_go": 3}
        self.sound_refused(
            {"held": {**held, "note": sound_of("blip")}}, ".held.note.voices[0].inst: no instrument named 'blip'")
        self.sound_refused({"held": {**held, "end": {"voices": 1}}}, ".held.end.voices: must be a list")

    def test_meta_that_is_not_a_dict_is_refused(self) -> None:
        self.sound_refused({"def": sound_of(), "meta": "pickup"}, ".meta: must be a dict")

    def test_check_marks_of_the_wrong_shape_are_refused(self) -> None:
        for check, message in (
            ({"at": 0, "note": 60}, ": must be a list"),
            ([[0, 60]], "[0]: must be a dict"),
            ([{"at": 0, "note": 60}, {"note": 60}], "[1]: missing key 'at'"),
            ([{"at": 0}], "[0]: missing key 'note'"),
            ([{"at": 0, "note": 60, "cents": 5}], "[0]: unknown key 'cents'"),
            ([{"at": "0", "note": 60}], "[0].at: must be a number"),
            ([{"at": 0, "note": "do"}], "[0].note: must be a number or a note name such as 'c#4', not 'do'"),
        ):
            with self.subTest(check=check):
                self.sound_refused({"def": sound_of(), "check": check}, ".check" + message)

    def test_a_limit_that_is_not_a_number_is_refused(self) -> None:
        for key in ("max_seconds", "max_dc", "max_bright"):
            with self.subTest(key=key):
                self.sound_refused({"def": sound_of(), key: "1"}, f".{key}: must be a number")


class Songs(ProjectCase):
    def test_a_project_with_chords_and_songs_loads(self) -> None:
        project = Project.from_dict(PROJECT)
        self.assertEqual(project.chords, CHORDS)
        self.assertEqual(project.songs, SONGS)

    def test_song_lays_its_own_instruments_over_the_projects(self) -> None:
        table = Project.from_dict(PROJECT).song("tune")["instruments"]
        self.assertEqual(list(table), ["reed", "deep", "buzz", "hiss", "bell", "plain", "organ", "glass", "thud"])
        self.assertEqual(table["reed"], INSTRUMENTS["reed"])
        self.assertEqual(table["deep"], SONGS["tune"]["instruments"]["deep"])
        self.assertEqual(table["organ"], SONGS["tune"]["instruments"]["organ"])

    def test_song_writes_out_an_instrument_of_its_own_that_names_one_of_the_projects(self) -> None:
        table = Project.from_dict(PROJECT).song("tune")["instruments"]
        self.assertEqual(table["glass"], INSTRUMENTS["bell"])
        self.assertIsNot(table["glass"], table["bell"])

    def test_song_changes_nothing_but_its_instruments(self) -> None:
        tune = Project.from_dict(PROJECT).song("tune")
        tune["instruments"] = SONGS["tune"]["instruments"]
        self.assertEqual(json.dumps(tune), json.dumps(SONGS["tune"]))

    def test_every_instrument_a_song_names_is_a_dict_in_its_table(self) -> None:
        tune = Project.from_dict(PROJECT).song("tune")
        named = [channel["inst"] for channel in tune["channels"].values()]
        named += [drum["inst"] for drum in tune["drums"].values()]
        named += ["glass", "reed"]
        self.assertEqual(sorted(set(named)), ["deep", "glass", "hiss", "organ", "reed", "thud"])
        for name in named:
            self.assertIsInstance(tune["instruments"][name], dict)

    def test_a_song_with_no_instruments_of_its_own_gets_the_projects(self) -> None:
        jingle = Project.from_dict(PROJECT).song("jingle")
        self.assertEqual(jingle["instruments"], INSTRUMENTS)
        self.assertEqual(jingle["order"], ["x"])

    def test_changing_what_song_returns_never_changes_the_project(self) -> None:
        project = Project.from_dict(PROJECT)
        tune = project.song("tune")
        tune["instruments"]["reed"]["vol"].append(0)
        tune["instruments"]["glass"]["alg"] = 0
        tune["instruments"]["organ"]["wave"] = "tri"
        del tune["instruments"]["deep"]
        tune["patterns"]["a"]["under"] = "r1"
        tune["order"].append("b")
        tune["layers"]["keys"].append(5)
        self.assertEqual(project.to_dict(), PROJECT)
        self.assertEqual(project.song("tune")["instruments"]["glass"], INSTRUMENTS["bell"])

    def test_layers_may_leave_their_keys_out(self) -> None:
        layers = {"groups": {"each": ["one", "kit"]}, "versions": {"entire": ["each"]}}
        project = Project.from_dict({"instruments": INSTRUMENTS, "songs": {"tune": song_of(layers=layers)}})
        self.assertEqual(project.song("tune")["layers"], layers)

    def test_asking_for_a_song_the_project_does_not_hold_is_refused(self) -> None:
        with self.assertRaises(RowSynthError) as caught:
            Project.from_dict(PROJECT).song("march")
        self.assertEqual(str(caught.exception), "no song named 'march'")


class RefusedSongs(ProjectCase):
    def channel_refused(self, channel: object, message: str) -> None:
        channels = {"one": channel, "kit": {"inst": "hiss", "drums": True}}
        self.song_refused(song_of(channels=channels), ".channels.one" + message)

    def layers_refused(self, changes: dict, message: str) -> None:
        layers = {
            "groups": {"hits": ["kit"], "lines": ["one"]},
            "versions": {"bare": ["hits"], "both": ["hits", "lines"]},
            "keys": [0, 3],
        }
        self.song_refused(song_of(layers={**layers, **changes}), ".layers" + message)

    def test_a_songs_table_that_is_not_a_dict_is_refused(self) -> None:
        self.refused({"songs": [song_of()]}, "songs: must be a dict")

    def test_a_song_that_is_not_a_dict_is_refused(self) -> None:
        self.song_refused("c d e", ": must be a dict")

    def test_an_unknown_song_key_is_refused_by_name(self) -> None:
        self.song_refused(song_of(tempo=120), ": unknown key 'tempo'")
        self.song_refused(song_of(rowsPerBeat=4), ": unknown key 'rowsPerBeat'")

    def test_a_song_without_its_ticks_channels_patterns_or_order_is_refused(self) -> None:
        for key in ("row_ticks", "channels", "patterns", "order"):
            with self.subTest(key=key):
                rest = {name: value for name, value in song_of().items() if name != key}
                self.song_refused(rest, f": missing key {key!r}")

    def test_a_song_value_that_is_not_a_number_is_refused(self) -> None:
        for key in ("row_ticks", "rows_per_beat", "gain"):
            with self.subTest(key=key):
                self.song_refused(song_of(**{key: "6"}), f".{key}: must be a number")

    def test_a_songs_own_instrument_is_checked_like_any_other(self) -> None:
        self.song_refused(song_of(instruments=["organ"]), ".instruments: must be a dict")
        self.song_refused(song_of(instruments={"organ": 3}), ".instruments.organ: must be a dict")
        self.song_refused(
            song_of(instruments={"organ": {"wave": "sine"}}),
            ".instruments.organ.wave: must be one of 'pulse', 'saw', 'tri', 'noise', 'fm', not 'sine'")

    def test_a_songs_own_instrument_naming_one_the_project_does_not_hold_is_refused(self) -> None:
        self.song_refused(song_of(instruments={"glass": "gong"}), ".instruments.glass: no instrument named 'gong'")
        own = {"organ": {"wave": "saw"}, "glass": "organ"}
        self.song_refused(song_of(instruments=own), ".instruments.glass: no instrument named 'organ'")

    def test_drums_of_the_wrong_shape_are_refused(self) -> None:
        drum = {"inst": "hiss", "note": "c2"}
        for drums, message in (
            ([drum], ": must be a dict"),
            ({"k": drum, "x": drum}, ": unknown key 'x'"),
            ({"k": "hiss"}, ".k: must be a dict"),
            ({"s": {"note": "c2"}}, ".s: missing key 'inst'"),
            ({"h": {"inst": "hiss"}}, ".h: missing key 'note'"),
            ({"H": {**drum, "pan": 0}}, ".H: unknown key 'pan'"),
            ({"t": {**drum, "note": "x2"}}, ".t.note: must be a number or a note name such as 'c#4', not 'x2'"),
            ({"k": {**drum, "rows": "2"}}, ".k.rows: must be a number"),
            ({"k": {**drum, "vel": [1]}}, ".k.vel: must be a number"),
        ):
            with self.subTest(drums=drums):
                self.song_refused(song_of(drums=drums), ".drums" + message)

    def test_a_drum_naming_an_instrument_the_song_cannot_reach_is_refused(self) -> None:
        self.song_refused(
            song_of(drums={"k": {"inst": "thud", "note": "c2"}}), ".drums.k.inst: no instrument named 'thud'")
        self.song_refused(
            song_of(drums={"k": {"inst": {"wave": "tri"}, "note": "c2"}}), ".drums.k.inst: must be a string")

    def test_channels_of_the_wrong_shape_are_refused(self) -> None:
        self.song_refused(song_of(channels=["one", "kit"]), ".channels: must be a dict")
        self.channel_refused("reed", ": must be a dict")
        self.channel_refused({"gain": 1}, ": missing key 'inst'")
        self.channel_refused({"inst": "reed", "volume": 1}, ": unknown key 'volume'")
        self.channel_refused({"inst": "reed", "gain": "1"}, ".gain: must be a number")
        self.channel_refused({"inst": "reed", "pan": "left"}, ".pan: must be a number")
        self.channel_refused({"inst": "reed", "drums": "yes"}, ".drums: must be true or false")

    def test_a_channel_naming_an_instrument_the_song_cannot_reach_is_refused(self) -> None:
        self.channel_refused({"inst": "organ"}, ".inst: no instrument named 'organ'")
        self.channel_refused({"inst": {"wave": "saw"}}, ".inst: must be a string")

    def test_an_instrument_of_another_song_is_out_of_reach(self) -> None:
        first = song_of(instruments={"organ": {"wave": "saw"}})
        second = song_of(channels={"one": {"inst": "organ"}, "kit": {"inst": "hiss", "drums": True}})
        self.refused(
            {"instruments": INSTRUMENTS, "songs": {"first": first, "second": second}},
            "songs.second.channels.one.inst: no instrument named 'organ'")

    def test_a_channel_echo_of_the_wrong_shape_is_refused(self) -> None:
        for echo, message in (
            (3, ": must be a dict"),
            ({"gain": 0.4}, ": missing key 'rows'"),
            ({"rows": 3}, ": missing key 'gain'"),
            ({"rows": 3, "gain": 0.4, "lp": 900}, ": unknown key 'lp'"),
            ({"rows": "3", "gain": 0.4}, ".rows: must be a number"),
            ({"rows": 3, "gain": None}, ".gain: must be a number"),
        ):
            with self.subTest(echo=echo):
                self.channel_refused({"inst": "reed", "echo": echo}, ".echo" + message)

    def test_patterns_of_the_wrong_shape_are_refused(self) -> None:
        self.song_refused(song_of(patterns=["x", "y"]), ".patterns: must be a dict")
        self.song_refused(song_of(patterns={"x": "c d e", "y": {}}), ".patterns.x: must be a dict")
        self.song_refused(song_of(patterns={"x": {}, "y": {"one": ["c", "d"]}}), ".patterns.y.one: must be a string")

    def test_a_pattern_naming_a_channel_the_song_does_not_hold_is_refused(self) -> None:
        self.song_refused(
            song_of(patterns={"x": {"one": "c"}, "y": {"one": "c", "two": "e"}}),
            ".patterns.y: no channel named 'two'")

    def test_an_instrument_named_in_pattern_text_that_the_song_cannot_reach_is_refused(self) -> None:
        self.song_refused(
            song_of(patterns={"x": {"one": "c d @bell e @gong f"}, "y": {}}),
            ".patterns.x.one: no instrument named 'gong'")

    def test_an_order_of_the_wrong_shape_is_refused(self) -> None:
        self.song_refused(song_of(order="x y"), ".order: must be a list")
        self.song_refused(song_of(order=["x", 2]), ".order[1]: must be a string")

    def test_an_order_naming_a_pattern_the_song_does_not_hold_is_refused(self) -> None:
        self.song_refused(song_of(order=["x", "y", "c"]), ".order[2]: no pattern named 'c'")

    def test_a_songs_chords_of_the_wrong_shape_are_refused(self) -> None:
        self.song_refused(song_of(chords=["Am"]), ".chords: must be a dict")
        self.song_refused(song_of(chords={"x": "Am F"}), ".chords.x: must be a list")
        self.song_refused(song_of(chords={"x": ["Am", 7]}), ".chords.x[1]: must be a string")

    def test_chords_for_a_pattern_the_song_does_not_hold_are_refused(self) -> None:
        self.song_refused(song_of(chords={"x": ["Am"], "z": ["F"]}), ".chords: no pattern named 'z'")

    def test_a_bar_naming_a_chord_the_project_does_not_hold_is_refused(self) -> None:
        self.song_refused(song_of(chords={"x": ["Am", "F  Dm7"]}), ".chords.x[1]: no chord named 'Dm7'")

    def test_layers_of_the_wrong_shape_are_refused(self) -> None:
        self.song_refused(song_of(layers=["one", "kit"]), ".layers: must be a dict")
        self.layers_refused({"mode": "stack"}, ": unknown key 'mode'")
        self.layers_refused({"groups": ["kit", "one"]}, ".groups: must be a dict")
        self.layers_refused({"groups": {"hits": "kit", "lines": ["one"]}}, ".groups.hits: must be a list")
        self.layers_refused({"versions": ["bare"]}, ".versions: must be a dict")
        self.layers_refused({"versions": {"bare": "hits"}}, ".versions.bare: must be a list")
        self.layers_refused({"keys": 2}, ".keys: must be a list")

    def test_layers_without_groups_or_versions_are_refused(self) -> None:
        layers = {"groups": {"each": ["one", "kit"]}, "versions": {"entire": ["each"]}}
        for key in layers:
            with self.subTest(key=key):
                rest = {name: value for name, value in layers.items() if name != key}
                self.song_refused(song_of(layers=rest), f".layers: missing key {key!r}")

    def test_a_group_naming_a_channel_the_song_does_not_hold_is_refused(self) -> None:
        self.layers_refused(
            {"groups": {"hits": ["kit"], "lines": ["one", "two"]}}, ".groups.lines[1]: no channel named 'two'")
        self.layers_refused({"groups": {"hits": ["kit"], "lines": [1]}}, ".groups.lines[0]: must be a string")

    def test_a_version_naming_a_group_the_song_does_not_hold_is_refused(self) -> None:
        self.layers_refused({"versions": {"both": ["hits", "strings"]}}, ".versions.both[1]: no group named 'strings'")

    def test_an_empty_list_of_keys_is_refused(self) -> None:
        self.layers_refused({"keys": []}, ".keys: must hold at least one key")

    def test_a_key_that_is_not_a_whole_number_is_refused(self) -> None:
        self.layers_refused({"keys": [0, 2.5]}, ".keys[1]: must be a whole number")
        self.layers_refused({"keys": ["0"]}, ".keys[0]: must be a whole number")

    def test_a_premix_of_the_wrong_shape_is_refused(self) -> None:
        for premix, message in (
            ({"version": "both", "key": 0}, ": must be a list"),
            (["both"], "[0]: must be a dict"),
            ([{"key": 0}], "[0]: missing key 'version'"),
            ([{"version": "both"}], "[0]: missing key 'key'"),
            ([{"version": "both", "key": 0, "gain": 1}], "[0]: unknown key 'gain'"),
            ([{"version": "both", "key": "0"}], "[0].key: must be a whole number"),
            ([{"version": ["both"], "key": 0}], "[0].version: must be a string"),
        ):
            with self.subTest(premix=premix):
                self.layers_refused({"premix": premix}, ".premix" + message)

    def test_a_premix_naming_a_version_the_song_does_not_hold_is_refused(self) -> None:
        self.layers_refused(
            {"premix": [{"version": "both", "key": 0}, {"version": "loud", "key": 0}]},
            ".premix[1].version: no version named 'loud'")

    def test_an_intro_of_the_wrong_shape_is_refused(self) -> None:
        self.song_refused(song_of(intro=["y"]), ".intro: must be a dict")
        self.song_refused(song_of(intro={"tail": 1}), ".intro: missing key 'order'")
        self.song_refused(song_of(intro={"order": ["y"], "bars": 1}), ".intro: unknown key 'bars'")
        self.song_refused(song_of(intro={"order": "y"}), ".intro.order: must be a list")
        self.song_refused(song_of(intro={"order": ["y"], "tail": "1.5"}), ".intro.tail: must be a number")
        self.song_refused(song_of(intro={"order": ["y"], "fade": True}), ".intro.fade: must be a number")

    def test_an_intro_naming_a_pattern_the_song_does_not_hold_is_refused(self) -> None:
        self.song_refused(song_of(intro={"order": ["y", "start"]}), ".intro.order[1]: no pattern named 'start'")


class RefusedNames(ProjectCase):
    """Sounds, songs, channels, groups and versions give files their names."""

    def test_a_sound_or_a_song_named_with_a_path_is_refused(self) -> None:
        self.refused({"instruments": INSTRUMENTS, "sounds": {"up/coin": {"def": sound_of()}}},
                     "sounds: name 'up/coin' cannot hold '/', since files are called by it")
        self.refused({"instruments": INSTRUMENTS, "songs": {"..": song_of()}},
                     "songs: name '..' means a folder, and files are called by it")

    def test_a_channel_named_with_a_path_is_refused(self) -> None:
        name = "up\\one"
        channels = {"one": {"inst": "reed"}, "kit": {"inst": "hiss", "drums": True}, name: {"inst": "reed"}}
        self.song_refused(song_of(channels=channels),
                          f".channels: name {name!r} cannot hold {name[2]!r}, since files are called by it")

    def test_a_group_or_a_version_that_cannot_name_a_file_is_refused(self) -> None:
        layers = {"groups": {"hits": ["kit"], "li:nes": ["one"]}, "versions": {"bare": ["hits"]}}
        self.song_refused(song_of(layers=layers),
                          ".layers.groups: name 'li:nes' cannot hold ':', since files are called by it")
        layers = {"groups": {"hits": ["kit"], "lines": ["one"]}, "versions": {"": ["hits"]}}
        self.song_refused(song_of(layers=layers), ".layers.versions: a name cannot be empty")

    def test_a_name_with_a_control_character_is_refused(self) -> None:
        self.refused({"instruments": INSTRUMENTS, "sounds": {"co\tin": {"def": sound_of()}}},
                     "sounds: name 'co\\tin' cannot hold '\\t', since files are called by it")

    def test_names_in_any_script_and_with_spaces_or_dots_inside_are_taken(self) -> None:
        sounds = {"k\u0151 2": {"def": sound_of()}, "coin.low": {"def": sound_of()}}
        self.assertEqual(list(Project.from_dict({"instruments": INSTRUMENTS, "sounds": sounds}).sounds), list(sounds))


class RefusedChords(ProjectCase):
    def chord_refused(self, chord: object, message: str) -> None:
        self.refused({"chords": {"Am": chord}}, "chords.Am" + message)

    def test_a_chords_table_that_is_not_a_dict_is_refused(self) -> None:
        self.refused({"chords": ["Am"]}, "chords: must be a dict")

    def test_a_chord_of_the_wrong_shape_is_refused(self) -> None:
        self.chord_refused([9, 0, 4], ": must be a dict")
        self.chord_refused({"tones": [9, 0, 4]}, ": missing key 'root'")
        self.chord_refused({"root": 9}, ": missing key 'tones'")
        self.chord_refused({"root": 9, "tones": [9, 0, 4], "fifth": 4}, ": unknown key 'fifth'")
        self.chord_refused({"root": 9, "tones": 9}, ".tones: must be a list")

    def test_a_root_or_a_tone_that_is_not_a_pitch_class_is_refused(self) -> None:
        for root in (12, -1, 9.0, "a"):
            with self.subTest(root=root):
                self.chord_refused({"root": root, "tones": [0]}, ".root: must be a whole number from 0 to 11")
        self.chord_refused({"root": 9, "tones": [9, 12]}, ".tones[1]: must be a whole number from 0 to 11")
        self.chord_refused({"root": 9, "tones": ["a"]}, ".tones[0]: must be a whole number from 0 to 11")


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
