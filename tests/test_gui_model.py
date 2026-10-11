"""Tests for rowsynth.gui.model."""
from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
import traceback
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import numpy as np

from rowsynth import checks
from rowsynth.bank import render_entry
from rowsynth.edit import NOTHING
from rowsynth.errors import RowSynthError
from rowsynth.gui import model as gui_model
from rowsynth.gui.model import SAVE_WAITS, Model
from rowsynth.measure import brightness, measure
from rowsynth.project import Project
from rowsynth.song import render_loop

# With three ticks to a row, a row is 400 samples at 8000 Hz, and a song of one bar is 6400: room for five of the
# frames its brightness is read from.
RATE = 8000

# The time the first write of a test is stamped with, in nanoseconds. Every later write is stamped a second on.
START = 1_700_000_000 * 10**9


def sound(note: str, ticks: float = 4, inst: str = "beep") -> dict:
    """A def of one voice playing one note from the start."""
    return {"voices": [{"inst": inst, "notes": [[0, note, ticks]]}]}


def song(text: str) -> dict:
    """A song of one bar: a tune left of the middle, and one long note under it right of the middle."""
    return {"row_ticks": 3,
            "channels": {"tune": {"inst": "beep", "pan": -0.5}, "low": {"inst": "hum", "pan": 0.5}},
            "patterns": {"a": {"tune": text, "low": "o3 c1"}}, "order": ["a"]}


# The tables come in another order than the listing shows them in, and no table has its names in the order of
# the alphabet. The sounds are one of each form. The checks find three things in it.
PROJECT = {
    "instruments": {"hum": {"wave": "tri", "vol": 12}, "beep": {"vol": [15, 12, 9, 6]}},
    "sounds": {
        "ping": {"def": sound("a4")},
        "blip": {"defs": [sound("c5"), sound("e5"), sound("g5")]},
        "crash": {"parts": [[0, {"def": sound("c3")}], [0.05, {"def": sound("c4"), "db": -6}]]},
        "drone": {"held": {"note": sound("a3", 12, "hum"), "start": 2, "loop": 6, "join": 1,
                           "end": sound("a3", 1, "hum"), "let_go": 2}},
    },
    "songs": {"tune": song("l4 c e g > c"), "air": song("l2 e g")},
}

# A song the engine refuses to loop.
NO_ROWS = {"songs": {"still": {"row_ticks": 3, "channels": {}, "patterns": {}, "order": []}}}
NO_ROWS_SAID = "the song has no rows, so there is nothing to loop"

# One fault in a song, where "air" comes round, and one in a sound, the peak of the second variant of "zaps". The
# sound "zap" has no fault, and the name of the one that has begins with its name.
FAULTS = {
    "instruments": PROJECT["instruments"],
    "sounds": {"zap": {"def": sound("a4")},
               "zaps": {"defs": [sound("c5"), {"voices": [{"inst": {"gain": 8}, "notes": [[0, "a4", 4]]}]}]}},
    "songs": {"tune": song("l4 c e g > c"), "air": song("l2 e g")},
}

# A project that loads and that the checks refuse: its song has text they cannot read.
UNREAD = {"instruments": PROJECT["instruments"], "songs": {"odd": song("l4 c e x9")}}
UNREAD_SAID = "songs.odd: pattern 'a', channel 'tune': cannot read \"x9\" in a channel"

# The checks themselves, for a test that has put something else in their place.
RUN = checks.run


def found_in(data: dict) -> list[checks.Finding]:
    """What the checks find in a project."""
    return RUN(Project.from_dict(data), RATE)


def changed(**tables: dict) -> dict:
    """PROJECT with the names given for each table set to what they come with. A name that comes with None is gone."""
    data = copy.deepcopy(PROJECT)
    for table, entries in tables.items():
        for name, entry in entries.items():
            if entry is None:
                del data[table][name]
            else:
                data[table][name] = entry
    return data


class ModelCase(unittest.TestCase):
    """What the tests below share: a folder for one project file, and the file stamped with a time of the test's own."""

    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "project.json"
        self.stamp = START

    def write(self, data: dict | str, later: bool = True) -> None:
        """Writes the file: a project, or text as it is. It is stamped a second after the write before, if `later`."""
        text = data if isinstance(data, str) else json.dumps(data)
        self.path.write_text(text, encoding="utf-8", newline="\n")
        if later:
            self.stamp += 10**9
        os.utime(self.path, ns=(self.stamp, self.stamp))

    def opened(self, data: dict = PROJECT) -> Model:
        """A model of the file, holding `data`."""
        self.write(data)
        return Model(self.path, RATE)

    def assert_render(self, made: dict, sides: list[np.ndarray], loop: bool) -> None:
        """Holds a render to its sides, sample for sample, and to what goes with them."""
        self.assertEqual([made["rate"], made["loop"], made["seconds"]], [RATE, loop, len(sides[0]) / RATE])
        self.assertEqual([side.dtype for side in made["sides"]], [np.float32] * len(sides))
        self.assertEqual(len(made["sides"]), len(sides))
        for side, wanted in zip(made["sides"], sides):
            np.testing.assert_array_equal(side, wanted)

    def assert_loop(self, made: dict, data: dict, name: str) -> None:
        """Holds a render to the loop the engine makes of a song of `data`."""
        loop = render_loop(Project.from_dict(data).song(name), RATE)
        self.assertEqual(made["seconds"], loop["seconds"])
        self.assert_render(made, [loop["left"], loop["right"]], True)


class OpeningTest(ModelCase):
    def test_the_listing_is_always_songs_sounds_and_instruments_with_their_names_as_the_file_has_them(self):
        self.assertEqual(self.opened().listing(), [
            ("songs", [("tune", ""), ("air", "")]),
            ("sounds", [("ping", ""), ("blip", "3"), ("crash", "parts"), ("drone", "held")]),
            ("instruments", [("hum", ""), ("beep", "")]),
        ])
        self.assertEqual(self.opened({}).listing(), [("songs", []), ("sounds", []), ("instruments", [])])

    def test_the_first_song_is_picked_at_the_start_or_with_no_song_the_first_sound_or_nothing(self):
        no_songs = {table: PROJECT[table] for table in ("instruments", "sounds")}
        no_sounds = {"instruments": PROJECT["instruments"]}
        for data, picked in ((PROJECT, ("song", "tune")), (no_songs, ("sound", "ping")), (no_sounds, None)):
            with self.subTest(picked=picked):
                self.assertEqual(self.opened(data).picked, picked)

    def test_a_file_that_cannot_be_read_or_is_no_project_is_refused_as_loading_it_is(self):
        for text in (None, "{", '{"songs": 3}'):
            with self.subTest(text=text):
                if text is not None:
                    self.write(text)
                with self.assertRaises((OSError, RowSynthError)) as loading:
                    Project.load(self.path)
                with self.assertRaises(type(loading.exception)) as opening:
                    Model(self.path, RATE)
                self.assertEqual(str(opening.exception), str(loading.exception))


class PickingTest(ModelCase):
    def test_picking_makes_a_song_a_sound_or_an_instrument_the_picked_thing(self):
        model = self.opened()
        for picked in (("sound", "crash"), ("instrument", "beep"), ("song", "air")):
            model.pick(*picked)
            self.assertEqual(model.picked, picked)

    def test_picking_what_the_project_does_not_have_is_refused_and_the_picked_thing_stays(self):
        model = self.opened()
        for kind, name, said in (
            ("sound", "tune", "no sound named 'tune'"),
            ("songs", "tune", "a thing of a project is a song, a sound or an instrument, not 'songs'"),
        ):
            with self.subTest(kind=kind):
                with self.assertRaises(RowSynthError) as refused:
                    model.pick(kind, name)
                self.assertEqual(str(refused.exception), said)
        self.assertEqual(model.picked, ("song", "tune"))


class RenderTest(ModelCase):
    def test_a_song_is_rendered_as_its_loop_in_two_sides(self):
        made = self.opened().render()
        self.assert_loop(made, PROJECT, "tune")
        self.assertFalse(np.array_equal(*made["sides"]))

    def test_a_sound_is_rendered_as_the_first_buffer_of_its_entry(self):
        model = self.opened()
        model.pick("sound", "blip")
        first = render_entry(Project.from_dict(PROJECT).sound("blip"), RATE)[0]
        self.assert_render(model.render(), [first], False)

    def test_an_instrument_is_rendered_as_one_c4_of_twelve_ticks_through_the_bank(self):
        model = self.opened()
        model.pick("instrument", "hum")
        note = {"def": {"voices": [{"inst": PROJECT["instruments"]["hum"], "notes": [[0, "c4", 12]]}]}}
        self.assert_render(model.render(), render_entry(note, RATE), False)

    def test_a_render_is_made_once_for_as_long_as_its_thing_stays_picked(self):
        model = self.opened()
        with mock.patch.object(gui_model, "render_loop", wraps=render_loop) as engine:
            made = model.render()
            model.numbers()
            model.pick("song", "tune")
            self.assertIs(model.render(), made)
            self.assertEqual(engine.call_count, 1)
            model.pick("song", "air")
            self.assert_loop(model.render(), PROJECT, "air")
            self.assertEqual(engine.call_count, 2)

    def test_with_nothing_picked_there_is_nothing_to_render_or_to_measure(self):
        model = self.opened({})
        for ask in (model.render, model.numbers):
            with self.assertRaises(RowSynthError) as refused:
                ask()
            self.assertEqual(str(refused.exception), "nothing is picked")

    def test_a_render_the_engine_refuses_is_tried_once_and_refused_each_time_it_is_asked_for(self):
        model = self.opened(NO_ROWS)
        with mock.patch.object(gui_model, "render_loop", wraps=render_loop) as engine:
            for ask in (model.render, model.numbers, model.render):
                with self.assertRaises(RowSynthError) as refused:
                    ask()
                self.assertEqual(str(refused.exception), NO_ROWS_SAID)
        self.assertEqual(engine.call_count, 1)
        self.assertTrue(model.render_ready())

    def test_the_numbers_are_those_of_the_mean_of_the_sides(self):
        model = self.opened()
        for picked in (("song", "tune"), ("sound", "crash"), ("instrument", "beep")):
            with self.subTest(picked=picked):
                model.pick(*picked)
                row = np.mean(model.render()["sides"], axis=0, dtype=np.float64)
                numbers = model.numbers()
                self.assertEqual(numbers, {**measure(row, RATE), **brightness(row, rate=RATE)})
                self.assertEqual(list(numbers), ["seconds", "peak_db", "rms_db", "dc", "clipped", "head_step",
                                                 "tail_step", "above_4k", "centroid_hz"])
        # The song is long enough for its brightness to be read, so that number is not a zero for want of samples.
        model.pick("song", "tune")
        self.assertGreater(model.numbers()["centroid_hz"], 0)


class FindingsTest(ModelCase):
    def test_the_findings_are_those_of_the_checks_at_the_model_s_rate_and_are_made_once(self):
        model = self.opened()
        with mock.patch.object(checks, "run", wraps=checks.run) as run:
            found = model.findings()
            self.assertIs(model.findings(), found)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(found, found_in(PROJECT))
        self.assertEqual([finding.target for finding in found], ["songs.air", "sounds.drone.start", "sounds.drone.end"])

    def test_a_project_the_checks_refuse_is_tried_once_and_refused_each_time_its_findings_are_asked_for(self):
        model = self.opened(UNREAD)
        with mock.patch.object(checks, "run", wraps=checks.run) as run:
            for _ in range(2):
                with self.assertRaises(RowSynthError) as refused:
                    model.findings()
                self.assertEqual(str(refused.exception), UNREAD_SAID)
        self.assertEqual(run.call_count, 1)
        self.assertTrue(model.findings_ready())
        self.assertIsNone(model.count())
        self.assertEqual(model.findings_for("song", "odd"), [])

    def test_the_count_is_how_many_findings_there_are_once_they_are_ready(self):
        for data, count in ((PROJECT, 3), (FAULTS, 2), ({"instruments": PROJECT["instruments"]}, 0)):
            with self.subTest(count=count):
                model = self.opened(data)
                self.assertIsNone(model.count())
                model.findings()
                self.assertEqual(model.count(), count)

    def test_the_findings_of_one_thing_are_those_whose_place_is_that_thing_or_lies_in_it(self):
        model = self.opened(FAULTS)
        self.assertEqual(model.findings_for("song", "air"), [])
        join, peak = model.findings()
        self.assertEqual([join.target, peak.target], ["songs.air", "sounds.zaps[1]"])
        for kind, name, found in (("song", "air", [join]), ("sound", "zaps", [peak]), ("song", "tune", []),
                                  ("sound", "zap", []), ("instrument", "hum", []), ("sound", "gone", [])):
            with self.subTest(kind=kind, name=name):
                self.assertEqual(model.findings_for(kind, name), found)
        with self.assertRaises(RowSynthError) as refused:
            model.findings_for("sounds", "zaps")
        self.assertEqual(str(refused.exception),
                         "a thing of a project is a song, a sound or an instrument, not 'sounds'")
        # The parts of a held sound are places in it.
        model = self.opened()
        _, start, end = model.findings()
        self.assertEqual(model.findings_for("sound", "drone"), [start, end])


class PollTest(ModelCase):
    def test_a_look_at_a_file_that_was_not_written_again_finds_nothing_and_keeps_the_render(self):
        model = self.opened()
        made = model.render()
        self.assertEqual(model.poll(), set())
        self.assertIsNone(model.problem)
        self.assertIs(model.render(), made)

    def test_a_look_after_the_file_was_written_again_gives_what_was_added_removed_or_changed(self):
        model = self.opened()
        self.write(changed(sounds={"ping": {"def": sound("b4")}, "pong": {"def": sound("d4")}, "crash": None},
                           instruments={"hum": {"wave": "tri", "vol": 9}}, songs={"air": None}))
        self.assertEqual(model.poll(), {("sound", "ping"), ("sound", "pong"), ("sound", "crash"),
                                        ("instrument", "hum"), ("song", "air")})
        self.assertEqual(model.listing(), [
            ("songs", [("tune", "")]),
            ("sounds", [("ping", ""), ("blip", "3"), ("drone", "held"), ("pong", "")]),
            ("instruments", [("hum", ""), ("beep", "")]),
        ])
        self.assertEqual(model.poll(), set())

    def test_a_change_to_the_chords_alone_is_given_as_each_chord_added_removed_or_changed(self):
        chord = {"root": 0, "tones": [0, 4, 7]}
        model = self.opened({**PROJECT, "chords": {"C": chord, "F": {"root": 5, "tones": [5, 9, 0]}, "Am": chord}})
        listing = model.listing()
        model.render()
        model.findings()
        self.write({**PROJECT, "chords": {"C": {"root": 0, "tones": [0, 4, 7, 11]}, "G": chord, "Am": chord}})
        self.assertEqual(model.poll(), {("chord", "C"), ("chord", "F"), ("chord", "G")})
        self.assertEqual([model.listing(), model.picked], [listing, ("song", "tune")])
        # The chords are part of what the checks go by, so what was kept is of a project that is no more.
        self.assertEqual([model.render_ready(), model.findings_ready()], [False, False])

    def test_a_file_written_again_with_nothing_in_it_changed_gives_nothing_and_keeps_what_was_kept(self):
        model = self.opened()
        made, found = model.render(), model.findings()
        # The same things, each table with its names the other way round.
        self.write({table: dict(reversed(things.items())) for table, things in PROJECT.items()})
        self.assertEqual(model.poll(), set())
        self.assertEqual([model.problem, model.picked], [None, ("song", "tune")])
        self.assertIs(model.render(), made)
        self.assertIs(model.findings(), found)
        self.assertEqual(model.listing(), [
            ("songs", [("air", ""), ("tune", "")]),
            ("sounds", [("drone", "held"), ("crash", "parts"), ("blip", "3"), ("ping", "")]),
            ("instruments", [("beep", ""), ("hum", "")]),
        ])
        self.assertEqual(model.poll(), set())

    def test_the_project_is_the_one_object_until_a_load_puts_another_in_its_place(self):
        model = self.opened()
        before = model.project
        self.assertEqual(model.poll(), set())
        self.assertIs(model.project, before)
        after = changed(songs={"tune": song("l4 c e g a"), "air": None})
        self.write(after)
        model.poll()
        self.assertIsNot(model.project, before)
        self.assertEqual(model.project, Project.from_dict(after))

    def test_after_a_load_the_picked_thing_is_still_picked_and_what_was_kept_of_it_is_made_again(self):
        model = self.opened()
        model.render()
        numbers, found = model.numbers(), model.findings()
        after = changed(songs={"tune": song("l4 c e g a"), "air": None})
        self.write(after)
        self.assertEqual(model.poll(), {("song", "tune"), ("song", "air")})
        self.assertEqual(model.picked, ("song", "tune"))
        self.assertFalse(model.render_ready())
        self.assert_loop(model.render(), after, "tune")
        self.assertNotEqual(model.numbers(), numbers)
        self.assertNotEqual(model.findings(), found)
        self.assertEqual(model.findings(), checks.run(Project.from_dict(after), RATE))

    def test_after_a_load_a_picked_thing_that_is_gone_is_picked_no_longer(self):
        model = self.opened()
        model.pick("sound", "crash")
        self.write(changed(sounds={"crash": None}))
        model.poll()
        self.assertIsNone(model.picked)

    def test_a_file_that_is_half_written_leaves_the_model_as_it_was_and_is_its_problem_until_it_is_whole(self):
        model = self.opened()
        made, listing = model.render(), model.listing()
        whole = json.dumps(changed(sounds={"ping": None}))
        self.write(whole[:len(whole) // 2])
        with self.assertRaises(RowSynthError) as loading:
            Project.load(self.path)
        for _ in range(2):
            self.assertEqual(model.poll(), set())
            self.assertEqual(model.problem, str(loading.exception))
        self.assertEqual([model.listing(), model.picked], [listing, ("song", "tune")])
        self.assertIs(model.render(), made)
        self.write(whole)
        self.assertEqual(model.poll(), {("sound", "ping")})
        self.assertIsNone(model.problem)

    def test_a_file_that_is_gone_is_the_model_s_problem_until_it_is_back(self):
        model = self.opened()
        made = model.render()
        self.path.unlink()
        self.assertEqual(model.poll(), set())
        self.assertEqual(model.problem, f"{self.path}: No such file or directory")
        self.assertEqual(model.picked, ("song", "tune"))
        # Back as it was, to the time it was written at: there is nothing to load again.
        self.write(PROJECT, later=False)
        self.assertEqual(model.poll(), set())
        self.assertIsNone(model.problem)
        self.assertIs(model.render(), made)

    def test_the_changes_of_a_look_are_each_whole_thing_that_was_added_removed_or_changed(self):
        chord = {"root": 0, "tones": [0, 4, 7]}
        before = {**copy.deepcopy(PROJECT), "chords": {"C": chord, "Am": chord}}
        before["instruments"]["spare"] = {"wave": "saw"}
        model = self.opened(before)
        self.assertEqual(model.changes, [])
        after = copy.deepcopy(before)
        del after["instruments"]["spare"], after["sounds"]["crash"], after["songs"]["air"], after["chords"]["C"]
        after["instruments"].update(beep={"vol": 9}, bell={"wave": "tri"})
        after["sounds"].update(ping={"def": sound("b4", inst="bell")}, pong={"def": sound("d4")})
        after["chords"]["G"] = {"root": 7, "tones": [7, 11, 2]}
        self.write(after)
        self.assertEqual(len(model.poll()), 9)
        # What a song or a sound may name comes before them when it is new, and after them when it is gone.
        self.assertEqual(model.changes, [
            ("instruments.beep", {"vol": [15, 12, 9, 6]}, {"vol": 9}),
            ("instruments.bell", NOTHING, {"wave": "tri"}),
            ("chords.G", NOTHING, {"root": 7, "tones": [7, 11, 2]}),
            ("songs.air", PROJECT["songs"]["air"], NOTHING),
            ("sounds.ping", PROJECT["sounds"]["ping"], {"def": sound("b4", inst="bell")}),
            ("sounds.crash", PROJECT["sounds"]["crash"], NOTHING),
            ("sounds.pong", NOTHING, {"def": sound("d4")}),
            ("instruments.spare", {"wave": "saw"}, NOTHING),
            ("chords.C", chord, NOTHING),
        ])
        self.assertEqual([model.poll(), model.changes], [set(), []])
        # A look that cannot read the file has no changes either.
        self.write("{")
        self.assertEqual([model.poll(), model.changes], [set(), []])


class ChangeTest(ModelCase):
    def held(self) -> dict:
        """What the file holds now."""
        return json.loads(self.path.read_text(encoding="utf-8"))

    def blocked(self, times: int) -> mock.Mock:
        """Has the file refuse to be put in place `times` times, as it does while another program has it open."""
        replace, left = os.replace, [times]

        def blocked(source: str, target: Path) -> None:
            if left[0]:
                left[0] -= 1
                raise PermissionError(13, "Permission denied")
            replace(source, target)

        patch = mock.patch.object(os, "replace", side_effect=blocked)
        self.addCleanup(patch.stop)
        return patch.start()

    def test_a_change_is_in_the_file_and_in_the_project_and_the_next_look_finds_nothing_in_it(self):
        model = self.opened()
        before = model.project
        model.render()
        model.findings()
        where, after = "songs.tune.patterns.a.tune", changed(songs={"tune": song("l4 c e g a")})
        self.assertEqual(model.change(where, "l4 c e g a"),
                         {"file": str(self.path), "where": where, "was": "l4 c e g > c", "now": "l4 c e g a"})
        self.assertEqual(self.held(), after)
        self.assertIsNot(model.project, before)
        self.assertEqual(model.project, Project.from_dict(after))
        # What was kept is of a project that is no more, as after any load.
        self.assertEqual([model.render_ready(), model.findings_ready()], [False, False])
        made = model.render()
        self.assert_loop(made, after, "tune")
        self.assertEqual([model.poll(), model.changes, model.problem], [set(), [], None])
        self.assertIs(model.render(), made)
        # What is there already is not written, and the model stays as it is.
        project = model.project
        self.assertTrue(model.change(where, "l4 c e g a")["same"])
        self.assertIs(model.project, project)
        self.assertIs(model.render(), made)

    def test_a_delete_takes_a_thing_out_of_the_file_and_a_picked_thing_that_is_gone_is_picked_no_longer(self):
        model = self.opened()
        model.pick("sound", "crash")
        self.assertEqual(model.change("sounds.crash", delete=True),
                         {"file": str(self.path), "where": "sounds.crash", "was": PROJECT["sounds"]["crash"]})
        self.assertEqual(self.held(), changed(sounds={"crash": None}))
        self.assertIsNone(model.picked)
        self.assertEqual(model.poll(), set())

    def test_a_change_that_is_refused_leaves_the_file_and_the_model_as_they_were(self):
        model = self.opened()
        project, made, written = model.project, model.render(), self.path.read_bytes()
        with self.assertRaises(RowSynthError) as refused:
            model.change("sounds.ping.def.voices[0].inst", "bell")
        self.assertEqual(str(refused.exception), "sounds.ping.def.voices[0].inst: no instrument named 'bell'")
        self.assertEqual(self.path.read_bytes(), written)
        self.assertIs(model.project, project)
        self.assertIs(model.render(), made)
        self.assertEqual(model.poll(), set())
        # A file that is gone is refused in plain words, and not waited for.
        self.path.unlink()
        with self.assertRaises(RowSynthError) as refused:
            model.change("songs.tune.gain", 0.3)
        self.assertEqual(str(refused.exception), f"{self.path}: No such file or directory")
        self.assertIs(model.project, project)

    def test_a_save_another_program_stands_in_the_way_of_is_tried_again_after_a_wait(self):
        self.write(PROJECT)
        waits: list[float] = []
        model = Model(self.path, RATE, wait=waits.append)
        replace = self.blocked(2)
        self.assertEqual(model.change("songs.tune.gain", 0.3)["now"], 0.3)
        self.assertEqual([replace.call_count, waits], [3, list(SAVE_WAITS[:2])])
        self.assertEqual([self.held()["songs"]["tune"]["gain"], model.project.songs["tune"]["gain"]], [0.3, 0.3])
        # No try has left a file of its own beside the project.
        self.assertEqual(os.listdir(self.path.parent), [self.path.name])
        self.assertEqual(model.poll(), set())

    def test_a_save_that_fails_at_the_last_try_too_is_refused_and_nothing_is_changed(self):
        self.write(PROJECT)
        waits: list[float] = []
        model = Model(self.path, RATE, wait=waits.append)
        project, written = model.project, self.path.read_bytes()
        replace = self.blocked(len(SAVE_WAITS) + 1)
        with self.assertRaises(RowSynthError) as refused:
            model.change("songs.tune.gain", 0.3)
        self.assertEqual(str(refused.exception),
                         f"{self.path}: another program has the file open, so the change was not saved")
        self.assertEqual([replace.call_count, waits], [len(SAVE_WAITS) + 1, list(SAVE_WAITS)])
        # The waits are about a fifth of a second in all.
        self.assertAlmostEqual(sum(SAVE_WAITS), 0.2)
        self.assertEqual(self.path.read_bytes(), written)
        self.assertIs(model.project, project)
        self.assertEqual(model.poll(), set())


class WorkerTest(ModelCase):
    """A render and the checks on threads of their own. An event holds them back, so each test says when they end."""

    def held(self, error: Exception | None = None) -> threading.Event:
        """Has the engine wait before it renders a loop, and the checks before they run. Gives what lets them go on.

        With `error`, each raises it once it is let go, and does nothing else.
        """
        go = threading.Event()

        def waiting(real: Callable) -> Callable:
            def wait(*args: object) -> object:
                go.wait(60)
                if error is not None:
                    raise error
                return real(*args)
            return wait

        for owner, name, kept_as in ((gui_model, "render_loop", "engine"), (checks, "run", "checker")):
            patch = mock.patch.object(owner, name, side_effect=waiting(getattr(owner, name)))
            setattr(self, kept_as, patch.start())
            self.addCleanup(patch.stop)
        # A test that fails before it lets them go leaves no thread waiting.
        self.addCleanup(go.set)
        return go

    def started(self, model: Model) -> threading.Thread:
        """The thread that a render of the picked thing is begun on."""
        model.start_render()
        (worker,) = model._rendered._workers.values()
        return worker

    def checking(self, model: Model) -> threading.Thread:
        """The thread that the checks are begun on."""
        model.start_findings()
        (worker,) = model._found._workers.values()
        return worker

    def test_a_render_begun_on_its_own_thread_is_ready_when_the_thread_is_through(self):
        model = self.opened()
        go = self.held()
        worker = self.started(model)
        model.start_render()
        self.assertEqual(list(model._rendered._workers.values()), [worker])
        self.assertFalse(model.render_ready())
        go.set()
        worker.join()
        self.assertTrue(model.render_ready())
        self.assertEqual(model._rendered._workers, {})
        self.assert_loop(model.render(), PROJECT, "tune")
        model.start_render()
        self.assertEqual(model._rendered._workers, {})
        self.assertEqual(self.engine.call_count, 1)

    def test_a_pick_made_while_a_render_runs_is_not_given_that_render(self):
        model = self.opened()
        go = self.held()
        worker = self.started(model)
        model.pick("sound", "ping")
        go.set()
        worker.join()
        self.assertFalse(model.render_ready())
        ping = render_entry(Project.from_dict(PROJECT).sound("ping"), RATE)
        self.assert_render(model.render(), ping, False)

    def test_a_load_made_while_a_render_runs_is_not_given_that_render(self):
        model = self.opened()
        go = self.held()
        worker = self.started(model)
        after = changed(songs={"tune": song("l4 c e g a")})
        self.write(after)
        model.poll()
        go.set()
        worker.join()
        self.assertEqual(model.picked, ("song", "tune"))
        self.assertFalse(model.render_ready())
        self.assert_loop(model.render(), after, "tune")

    def test_asking_for_a_render_that_is_on_its_way_waits_for_it(self):
        model = self.opened()
        go = self.held()
        worker = self.started(model)
        join = worker.join
        # The engine is let go at the moment the model starts to wait for the thread, and not before.
        with mock.patch.object(worker, "join", side_effect=lambda: (go.set(), join())) as waited:
            made = model.render()
        self.assertEqual(waited.call_count, 1)
        self.assertEqual(self.engine.call_count, 1)
        self.assert_loop(made, PROJECT, "tune")

    def test_a_render_that_breaks_on_the_thread_is_told_once_and_kept_as_a_refusal_is(self):
        model = self.opened()
        go = self.held(ZeroDivisionError("broken"))
        with mock.patch.object(threading, "excepthook") as told:
            worker = self.started(model)
            go.set()
            worker.join()
        # The thread ended in the error, which is how its traceback comes to be written, the once.
        self.assertEqual([told.call_count, told.call_args[0][0].exc_type], [1, ZeroDivisionError])
        self.assertTrue(model.render_ready())
        self.assertEqual(model._rendered._workers, {})
        depths = set()
        for ask in (model.render, model.numbers, model.render):
            with self.assertRaises(ZeroDivisionError) as broke:
                ask()
            self.assertEqual(str(broke.exception), "broken")
            depths.add(len(traceback.extract_tb(broke.exception.__traceback__)))
        # Raised again and again, the error carries no more traceback than it did.
        self.assertEqual(len(depths), 1)
        model.start_render()
        self.assertEqual([model._rendered._workers, self.engine.call_count], [{}, 1])
        # It is what one thing of one load broke with, and is not taken for what another would do.
        model.pick("song", "air")
        self.assertFalse(model.render_ready())

    def test_checks_that_break_on_the_thread_are_told_once_and_kept_as_a_refusal_is(self):
        model = self.opened()
        go = self.held(ZeroDivisionError("broken"))
        with mock.patch.object(threading, "excepthook") as told:
            worker = self.checking(model)
            go.set()
            worker.join()
        self.assertEqual(told.call_count, 1)
        self.assertEqual([model.findings_ready(), model.count(), model.findings_for("song", "air")],
                         [True, None, []])
        for _ in range(2):
            with self.assertRaises(ZeroDivisionError):
                model.findings()
        self.assertEqual(self.checker.call_count, 1)
        self.write(changed(songs={"air": None}))
        model.poll()
        self.assertFalse(model.findings_ready())

    def test_findings_begun_on_their_own_thread_are_ready_when_the_thread_is_through(self):
        model = self.opened()
        go = self.held()
        worker = self.checking(model)
        model.start_findings()
        self.assertEqual(list(model._found._workers.values()), [worker])
        self.assertEqual([model.findings_ready(), model.count(), model.findings_for("song", "air")],
                         [False, None, []])
        go.set()
        worker.join()
        self.assertTrue(model.findings_ready())
        self.assertEqual(model._found._workers, {})
        self.assertEqual(model.count(), 3)
        self.assertEqual(model.findings(), found_in(PROJECT))
        self.assertEqual(self.checker.call_count, 1)

    def test_a_pick_made_while_the_checks_run_does_not_lose_their_findings(self):
        model = self.opened()
        go = self.held()
        worker = self.checking(model)
        model.pick("sound", "ping")
        go.set()
        worker.join()
        self.assertTrue(model.findings_ready())
        self.assertEqual(model.findings(), found_in(PROJECT))

    def test_a_load_made_while_the_checks_run_is_not_given_their_findings(self):
        model = self.opened()
        go = self.held()
        worker = self.checking(model)
        after = changed(songs={"air": None})
        self.write(after)
        model.poll()
        go.set()
        worker.join()
        self.assertEqual([model.findings_ready(), model.count()], [False, None])
        self.assertEqual(model.findings(), found_in(after))
        self.assertEqual(model.count(), 2)

    def test_a_render_and_the_checks_run_at_the_same_time_and_both_come_out_right(self):
        model = self.opened()
        go = self.held()
        workers = [self.started(model), self.checking(model)]
        self.assertEqual([worker.is_alive() for worker in workers], [True, True])
        self.assertEqual([model.render_ready(), model.findings_ready()], [False, False])
        go.set()
        for worker in workers:
            worker.join()
        self.assertEqual([model.render_ready(), model.findings_ready()], [True, True])
        self.assert_loop(model.render(), PROJECT, "tune")
        self.assertEqual(model.findings(), found_in(PROJECT))
        self.assertEqual([self.engine.call_count, self.checker.call_count], [1, 1])


if __name__ == "__main__":
    unittest.main()
