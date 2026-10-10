"""Tests for rowsynth.gui.model."""
from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from rowsynth import checks
from rowsynth.bank import render_entry
from rowsynth.errors import RowSynthError
from rowsynth.gui import model as gui_model
from rowsynth.gui.model import Model
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

    def test_the_findings_are_those_of_the_checks_at_the_model_s_rate_and_are_made_once(self):
        model = self.opened()
        with mock.patch.object(checks, "run", wraps=checks.run) as run:
            found = model.findings()
            self.assertIs(model.findings(), found)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(found, checks.run(Project.from_dict(PROJECT), RATE))
        self.assertEqual([finding.target for finding in found], ["songs.air", "sounds.drone.start", "sounds.drone.end"])


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


class WorkerTest(ModelCase):
    """A render on its own thread. The engine is held back by an event, so each test says when the render ends."""

    def held(self, error: Exception | None = None) -> threading.Event:
        """Has the engine wait before it renders a loop, or raises `error`. Gives the event that lets it go on."""
        go = threading.Event()

        def wait(song: dict, rate: int) -> dict:
            go.wait(60)
            if error is not None:
                raise error
            return render_loop(song, rate)

        patch = mock.patch.object(gui_model, "render_loop", side_effect=wait)
        self.engine = patch.start()
        self.addCleanup(patch.stop)
        # A test that fails before it lets the engine go leaves no thread waiting.
        self.addCleanup(go.set)
        return go

    def started(self, model: Model) -> threading.Thread:
        """The thread that a render of the picked thing is begun on."""
        model.start_render()
        (worker,) = model._workers.values()
        return worker

    def test_a_render_begun_on_its_own_thread_is_ready_when_the_thread_is_through(self):
        model = self.opened()
        go = self.held()
        worker = self.started(model)
        model.start_render()
        self.assertEqual(list(model._workers.values()), [worker])
        self.assertFalse(model.render_ready())
        go.set()
        worker.join()
        self.assertTrue(model.render_ready())
        self.assertEqual(model._workers, {})
        self.assert_loop(model.render(), PROJECT, "tune")
        model.start_render()
        self.assertEqual(model._workers, {})
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

    def test_a_render_that_breaks_on_the_thread_is_not_kept_and_breaks_again_where_it_is_asked_for(self):
        model = self.opened()
        go = self.held(ZeroDivisionError("broken"))
        with mock.patch.object(threading, "excepthook") as told:
            worker = self.started(model)
            go.set()
            worker.join()
        self.assertEqual(told.call_count, 1)
        self.assertFalse(model.render_ready())
        self.assertEqual(model._workers, {})
        with self.assertRaises(ZeroDivisionError):
            model.render()
        self.assertEqual(self.engine.call_count, 2)


if __name__ == "__main__":
    unittest.main()
