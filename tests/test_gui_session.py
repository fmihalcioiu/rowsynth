"""Tests for rowsynth.gui.session."""
from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
import unittest
from collections.abc import Callable, Sequence
from pathlib import Path
from unittest import mock

from rowsynth import checks
from rowsynth.errors import RowSynthError
from rowsynth.gui import model as gui_model
from rowsynth.gui.model import KINDS, Model
from rowsynth.gui.session import GLOW_TIME, POLL_EVERY, Session
from rowsynth.project import Project

RATE = 8000

# The time by the clock a test begins at, and the time its first write of the file is stamped with, in nanoseconds.
START = 100.0
STAMP = 1_700_000_000 * 10**9


def sound(note: str) -> dict:
    """A def of one voice playing one note from the start."""
    return {"voices": [{"inst": "beep", "notes": [[0, note, 4]]}]}


def song(text: str) -> dict:
    """A song of one bar on one channel."""
    return {"row_ticks": 3, "channels": {"tune": {"inst": "beep"}}, "patterns": {"a": {"tune": text}}, "order": ["a"]}


# The checks find one thing in it: where "air" comes round.
PROJECT = {
    "instruments": {"beep": {"vol": [15, 12, 9, 6]}},
    "sounds": {"zap": {"def": sound("a4")}, "pop": {"def": sound("c5")}},
    "songs": {"tune": song("l4 c e g > c"), "air": song("l2 e g")},
}

# A song the engine refuses to loop.
STILL = {"row_ticks": 3, "channels": {}, "patterns": {}, "order": []}
STILL_SAID = "the song has no rows, so there is nothing to loop"


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


def settle() -> None:
    """Waits for every render and every run of the checks that is still on a thread of the model's."""
    for thread in threading.enumerate():
        if thread.name.startswith("rowsynth-"):
            thread.join()


class Clock:
    """A clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> float:
        return self.now


class Speaker:
    """A stand-in for the player: it keeps what it was asked, in order, and plays nothing."""

    def __init__(self) -> None:
        self.asked: list[tuple] = []
        self.playing = False

    def play(self, sides: Sequence, loop: bool = False) -> None:
        self.asked.append(("play", sides, loop))
        self.playing = True

    def stop(self) -> None:
        self.asked.append(("stop",))
        self.playing = False

    def swap(self, sides: Sequence) -> None:
        self.asked.append(("swap", sides))

    def loop(self, on: bool) -> None:
        self.asked.append(("loop", on))

    def position(self) -> float:
        return 0.125 if self.playing else 0.0

    def level(self) -> tuple[float, float]:
        return (0.5, 0.25) if self.playing else (0.0, 0.0)

    def close(self) -> None:
        self.asked.append(("close",))
        self.playing = False

    def did(self, what: str) -> list[tuple]:
        """What came with each time it was asked to do one thing, in order."""
        return [asked[1:] for asked in self.asked if asked[0] == what]


class SessionCase(unittest.TestCase):
    """What the tests below share: one project file stamped by the test, a clock moved by hand, a stand-in player."""

    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        # No thread of one test is left to run into the next.
        self.addCleanup(settle)
        self.path = Path(folder.name) / "project.json"
        self.stamp = STAMP
        self.clock = Clock()
        self.speaker = Speaker()

    def write(self, data: dict | str, later: bool = True) -> None:
        """Writes the file: a project, or text as it is. It is stamped a second after the write before, if `later`."""
        text = data if isinstance(data, str) else json.dumps(data)
        self.path.write_text(text, encoding="utf-8", newline="\n")
        if later:
            self.stamp += 10**9
        os.utime(self.path, ns=(self.stamp, self.stamp))

    def opened(self, data: dict = PROJECT, player: bool = True) -> Session:
        """A session on the file, holding `data`, with the stand-in player or with none."""
        self.write(data)
        return Session(Model(self.path, RATE), self.speaker if player else None, self.clock)

    def later(self, seconds: float) -> None:
        """Moves the clock on."""
        self.clock.now += seconds

    def looked(self, session: Session) -> None:
        """A tick at which the file is looked at again."""
        self.later(POLL_EVERY)
        session.tick()

    def held(self) -> threading.Event:
        """Has every render and the checks wait before they begin. Gives what lets them go on."""
        go = threading.Event()

        def waiting(real: Callable) -> Callable:
            def wait(*args: object) -> object:
                go.wait(60)
                return real(*args)
            return wait

        for owner, name in ((gui_model, "render_loop"), (gui_model, "render_entry"), (checks, "run")):
            patch = mock.patch.object(owner, name, side_effect=waiting(getattr(owner, name)))
            patch.start()
            self.addCleanup(patch.stop)
        # A test that fails before it lets them go leaves no thread waiting.
        self.addCleanup(go.set)
        return go

    def assert_played(self, *wanted: tuple[dict, bool]) -> None:
        """Holds the player to the renders it was asked to play, in order, each with whether it was as a loop."""
        played = self.speaker.did("play")
        self.assertEqual([loop for _, loop in played], [loop for _, loop in wanted])
        for (sides, _), (made, _) in zip(played, wanted):
            self.assertIs(sides, made["sides"])

    def assert_swapped(self, *wanted: dict) -> None:
        """Holds the player to the renders it was asked to put in place of a loop, in order."""
        swapped = self.speaker.did("swap")
        self.assertEqual(len(swapped), len(wanted))
        for (sides,), made in zip(swapped, wanted):
            self.assertIs(sides, made["sides"])


class PlayTest(SessionCase):
    def test_a_play_asked_for_before_the_render_is_ready_starts_when_it_is(self):
        session = self.opened()
        go = self.held()
        session.tick()
        session.play()
        session.tick()
        self.assertTrue(session.playing)
        self.assertEqual(self.speaker.asked, [])
        # Neither the play nor a tick waited for the render: it is still held.
        self.assertFalse(session.model.render_ready())
        go.set()
        made = session.model.render()
        session.tick()
        session.tick()
        self.assert_played((made, True))

    def test_a_play_asked_for_when_the_render_is_ready_starts_at_once(self):
        session = self.opened()
        self.assertEqual([session.playing, session.position(), session.seconds, session.level()],
                         [False, 0.0, 0.0, (0.0, 0.0)])
        made = session.model.render()
        session.play()
        self.assert_played((made, True))
        self.assertEqual([session.playing, session.position(), session.seconds, session.level()],
                         [True, 0.125, made["seconds"], (0.5, 0.25)])
        # From its start again, with the render taken up already.
        session.tick()
        session.play()
        self.assert_played((made, True), (made, True))
        self.assert_swapped()

    def test_stop_ends_the_sound_and_gives_up_a_play_that_waits_for_its_render(self):
        session = self.opened()
        go = self.held()
        session.play()
        session.stop()
        self.assertFalse(session.playing)
        go.set()
        session.model.render()
        session.tick()
        self.assertEqual(self.speaker.did("play"), [])
        session.play()
        session.stop()
        self.assertEqual([session.playing, self.speaker.playing, session.position()], [False, False, 0.0])

    def test_toggle_plays_when_nothing_does_and_stops_what_plays(self):
        session = self.opened()
        made = session.model.render()
        session.toggle()
        self.assertTrue(session.playing)
        session.toggle()
        self.assertFalse(session.playing)
        self.assert_played((made, True))
        self.assertEqual(self.speaker.did("stop"), [()])

    def test_a_song_goes_round_while_loop_is_on_and_a_sound_or_an_instrument_plays_once(self):
        session = self.opened()
        session.audition = False
        self.assertTrue(session.loop)
        played = []
        for loop in (True, False):
            session.loop = loop
            for kind, name in (("song", "air"), ("sound", "zap"), ("instrument", "beep")):
                session.pick(kind, name)
                made = session.model.render()
                session.play()
                played.append((made, loop and kind == "song"))
        self.assertEqual([loop for _, loop in played], [True, False, False, False, False, False])
        self.assert_played(*played)

    def test_a_change_of_loop_is_taken_where_it_is_by_a_song_that_plays_and_by_nothing_else(self):
        session = self.opened()
        tune = session.model.render()
        session.loop = False
        self.assertEqual(self.speaker.asked, [])
        session.play()
        session.loop = True
        session.loop = True
        session.loop = False
        # The player is asked to change the song it plays, and never to play it again.
        self.assertEqual(self.speaker.did("loop"), [(True,), (False,)])
        self.assert_played((tune, False))
        session.pick("sound", "zap")
        zap = session.model.render()
        session.tick()
        session.loop = True
        self.assertEqual(self.speaker.did("loop"), [(True,), (False,)])
        self.assert_played((tune, False), (zap, False))


class PickTest(SessionCase):
    def test_picking_a_sound_or_an_instrument_plays_it_once_as_soon_as_its_render_is_ready(self):
        session = self.opened()
        go = self.held()
        session.pick("sound", "zap")
        session.tick()
        self.assertEqual([session.model.picked, session.playing], [("sound", "zap"), True])
        self.assertEqual(self.speaker.did("play"), [])
        self.assertFalse(session.model.render_ready())
        go.set()
        zap = session.model.render()
        session.tick()
        session.pick("instrument", "beep")
        beep = session.model.render()
        session.tick()
        session.tick()
        self.assert_played((zap, False), (beep, False))

    def test_picking_a_song_stops_what_plays_and_plays_nothing(self):
        session = self.opened()
        tune = session.model.render()
        session.play()
        session.pick("song", "air")
        self.assertEqual([session.model.picked, session.playing, session.seconds], [("song", "air"), False, 0.0])
        air = session.model.render()
        session.tick()
        self.assert_played((tune, True))
        self.assertEqual(self.speaker.did("stop"), [()])
        self.assertEqual(session.seconds, air["seconds"])

    def test_with_audition_off_picking_a_sound_plays_nothing(self):
        session = self.opened()
        session.audition = False
        session.pick("sound", "zap")
        self.assertFalse(session.playing)
        session.model.render()
        session.tick()
        self.assertEqual(self.speaker.did("play"), [])

    def test_picking_what_the_project_does_not_have_is_refused_and_what_plays_goes_on(self):
        session = self.opened()
        session.model.render()
        session.play()
        with self.assertRaises(RowSynthError) as refused:
            session.pick("sound", "tune")
        self.assertEqual(str(refused.exception), "no sound named 'tune'")
        self.assertEqual([session.model.picked, session.playing], [("song", "tune"), True])
        self.assertEqual(self.speaker.did("stop"), [])


class OutsideTest(SessionCase):
    def test_a_thing_that_changed_outside_glows_and_the_glow_fades_with_the_clock(self):
        session = self.opened()
        session.tick()
        self.assertEqual(session.glows(), {})
        self.write(changed(songs={"tune": song("l4 c e g a")}))
        self.looked(session)
        self.assertEqual(session.glows(), {("song", "tune"): 1.0})
        self.later(GLOW_TIME / 2)
        session.tick()
        self.assertEqual(session.glows(), {("song", "tune"): 0.5})
        self.assertEqual([session.glow("song", "tune"), session.glow("song", "air")], [0.5, 0.0])
        self.later(GLOW_TIME / 2)
        self.assertEqual([session.glow("song", "tune"), session.glows()], [0.0, {}])
        session.tick()
        self.assertEqual([session.glow("song", "tune"), session.glows()], [0.0, {}])

    def test_a_change_outside_is_named_in_the_message_and_the_picked_thing_is_rendered_again(self):
        session = self.opened()
        made = session.model.render()
        session.tick()
        self.assertEqual(session.message, "")
        go = self.held()
        self.write(changed(songs={"tune": song("l2 c e")}, sounds={"zap": {"def": sound("b4")}}))
        self.looked(session)
        self.assertEqual(session.message, "changed outside: tune, zap")
        # What is told of the picked thing is what the model has: nothing, until the render is there again.
        self.assertIsNone(session.numbers())
        go.set()
        again = session.model.render()
        session.tick()
        self.assertIsNot(again, made)
        self.assertEqual(session.numbers(), session.model.numbers())
        self.assertEqual(self.speaker.asked, [])

    def test_a_loop_that_plays_is_given_the_new_render_and_is_not_begun_again(self):
        session = self.opened()
        made = session.model.render()
        session.play()
        go = self.held()
        self.write(changed(songs={"tune": song("l4 c e g a")}))
        self.looked(session)
        session.tick()
        self.assertTrue(session.playing)
        self.assert_swapped()
        go.set()
        again = session.model.render()
        session.tick()
        session.tick()
        self.assert_swapped(again)
        self.assert_played((made, True))
        self.assertEqual(self.speaker.did("stop"), [])

    def test_a_song_that_plays_once_is_left_to_end_as_it_was(self):
        session = self.opened()
        session.loop = False
        made = session.model.render()
        session.play()
        self.write(changed(songs={"tune": song("l4 c e g a")}))
        self.looked(session)
        session.model.render()
        session.tick()
        self.assert_swapped()
        self.assert_played((made, False))

    def test_a_change_to_a_thing_that_is_not_picked_glows_as_well(self):
        session = self.opened()
        session.model.render()
        session.tick()
        self.write(changed(sounds={"pop": {"def": sound("d5")}}))
        self.looked(session)
        self.assertEqual(session.glows(), {("sound", "pop"): 1.0})
        self.assertEqual([session.message, session.model.picked], ["changed outside: pop", ("song", "tune")])

    def test_a_change_to_the_chords_alone_is_named_glows_no_thing_of_the_listing_and_has_the_checks_run_again(self):
        chord = {"root": 0, "tones": [0, 4, 7]}
        session = self.opened({**PROJECT, "chords": {"C": chord}})
        with mock.patch.object(checks, "run", wraps=checks.run) as run:
            session.tick()
            settle()
            session.tick()
            self.assertEqual([run.call_count, session.finding_count()], [1, 1])
            self.write({**PROJECT, "chords": {"C": chord, "G": {"root": 7, "tones": [7, 11, 2]}}})
            self.looked(session)
            settle()
            session.tick()
            self.assertEqual([run.call_count, session.finding_count()], [2, 1])
        self.assertEqual([session.message, session.glows()], ["changed outside: G", {("chord", "G"): 1.0}])
        listed = [(KINDS[section], name) for section, things in session.model.listing() for name, _ in things]
        self.assertEqual([session.glow(*thing) for thing in listed], [0.0] * 5)

    def test_a_file_written_again_with_nothing_in_it_changed_starts_no_render_and_no_checks(self):
        session = self.opened()
        with mock.patch.object(gui_model, "render_loop", wraps=gui_model.render_loop) as engine, \
                mock.patch.object(checks, "run", wraps=checks.run) as run:
            session.tick()
            made = session.model.render()
            session.play()
            self.write(PROJECT)
            for _ in range(2):
                self.looked(session)
            settle()
            session.tick()
            self.assertEqual([engine.call_count, run.call_count], [1, 1])
        self.assertIs(session.model.render(), made)
        self.assertEqual([session.glows(), session.message], [{}, ""])
        self.assertEqual([session.numbers(), session.finding_count()], [session.model.numbers(), 1])
        self.assert_swapped()
        self.assert_played((made, True))

    def test_when_the_picked_thing_is_removed_outside_the_first_thing_of_the_listing_takes_its_place(self):
        session = self.opened()
        session.audition = False
        session.pick("sound", "pop")
        pop = session.model.render()
        session.play()
        self.write(changed(sounds={"pop": None}))
        self.looked(session)
        self.assertEqual([session.model.picked, session.playing], [("song", "tune"), False])
        self.assertEqual(session.message, "changed outside: pop")
        tune = session.model.render()
        session.tick()
        self.assertEqual(session.seconds, tune["seconds"])
        self.assert_played((pop, False))
        # With no song and no sound left it is the first instrument, and with nothing left it is nothing.
        self.write({"instruments": PROJECT["instruments"]})
        self.looked(session)
        self.assertEqual(session.model.picked, ("instrument", "beep"))
        self.write({})
        self.looked(session)
        self.assertEqual([session.model.picked, session.seconds, session.numbers()], [None, 0.0, None])

    def test_more_than_three_changes_at_once_are_named_as_the_first_three_and_a_count(self):
        session = self.opened()
        session.tick()
        self.write(changed(songs={"tune": song("l2 c e"), "air": None},
                           sounds={"zap": {"def": sound("b4")}, "pop": None, "new": {"def": sound("e5")}}))
        self.looked(session)
        self.assertEqual(session.message, "changed outside: air, new, pop and 2 more")
        self.assertEqual(len(session.glows()), 5)

    def test_what_the_engine_refuses_is_said_stops_what_plays_and_is_not_tried_at_each_tick(self):
        session = self.opened()
        tune = session.model.render()
        session.play()
        with mock.patch.object(gui_model, "render_loop", wraps=gui_model.render_loop) as engine:
            self.write(changed(songs={"tune": STILL}))
            self.looked(session)
            # The song is refused by the engine, and the project with it by the checks.
            for ask in (session.model.render, session.model.findings):
                with self.assertRaises(RowSynthError):
                    ask()
            for _ in range(4):
                self.looked(session)
            self.assertEqual(engine.call_count, 1)
            # The checks were taken up after the render, so theirs are the words that stand.
            self.assertEqual([session.message, session.playing], [f"songs.tune: {STILL_SAID}", False])
            self.assertEqual([session.seconds, session.numbers()], [0.0, None])
            self.assertEqual(self.speaker.did("stop"), [()])
            # Asked to play, it is refused again as it was, with no new try.
            session.play()
            self.assertEqual([session.message, session.playing, engine.call_count], [STILL_SAID, False, 1])
            # It is tried again when the file has changed.
            self.write(PROJECT)
            self.looked(session)
            again = session.model.render()
            session.tick()
            self.assertEqual([session.message, engine.call_count], ["changed outside: tune", 2])
            self.assertEqual(session.seconds, again["seconds"])
        self.assert_played((tune, True))

    def test_a_render_that_breaks_is_tried_once_said_with_the_name_of_its_error_and_tried_again_after_a_change(self):
        session = self.opened()
        with mock.patch.object(gui_model, "render_loop", side_effect=ZeroDivisionError("broken")) as engine, \
                mock.patch.object(threading, "excepthook") as told:
            session.tick()
            settle()
            for _ in range(4):
                self.looked(session)
            # One try and one traceback, however many looks there were.
            self.assertEqual([engine.call_count, told.call_count], [1, 1])
            self.assertEqual(session.message, "ZeroDivisionError: broken")
            self.assertEqual([session.playing, session.seconds, session.numbers()], [False, 0.0, None])
            session.play()
            self.assertEqual([session.playing, engine.call_count, told.call_count], [False, 1, 1])
            self.write(changed(songs={"tune": song("l2 c e")}))
            self.looked(session)
            self.assertEqual(session.glows(), {("song", "tune"): 1.0})
            settle()
            session.tick()
            self.assertEqual([engine.call_count, told.call_count], [2, 2])
            self.assertEqual(session.message, "ZeroDivisionError: broken")
        self.assertEqual(self.speaker.did("play"), [])

    def test_a_file_that_cannot_be_read_just_now_is_the_message_for_as_long_as_that_lasts(self):
        session = self.opened()
        session.tick()
        whole = json.dumps(PROJECT)
        self.write(whole[:len(whole) // 2])
        with self.assertRaises(RowSynthError) as loading:
            Project.load(self.path)
        for _ in range(2):
            self.looked(session)
            self.assertEqual(session.message, str(loading.exception))
        self.write(whole, later=False)
        self.looked(session)
        self.assertEqual(session.message, "")


class StateTest(SessionCase):
    def test_with_no_player_everything_works_and_nothing_sounds(self):
        session = self.opened(player=False)
        tune = session.model.render()
        session.tick()
        session.play()
        self.assertFalse(session.playing)
        session.toggle()
        session.stop()
        session.loop = False
        self.assertEqual([session.position(), session.level(), session.seconds], [0.0, (0.0, 0.0), tune["seconds"]])
        session.pick("sound", "zap")
        self.assertEqual([session.message, session.playing], ["", False])
        zap = session.model.render()
        session.tick()
        self.assertEqual([session.playing, session.seconds], [False, zap["seconds"]])
        session.close()

    def test_with_no_player_the_message_says_so_from_the_first_tick_until_something_else_is_said(self):
        session = self.opened(player=False)
        for _ in range(2):
            session.tick()
            self.assertEqual(session.message, "no sound output")
        self.write(changed(sounds={"pop": None}))
        self.looked(session)
        self.assertEqual(session.message, "changed outside: pop")
        # It is said again each time something is to play.
        session.play()
        self.assertEqual(session.message, "no sound output")
        # A session that has a player says nothing of the kind.
        heard = self.opened()
        heard.tick()
        self.assertEqual(heard.message, "")

    def test_checks_that_refused_the_project_are_told_from_checks_that_still_run(self):
        session = self.opened(changed(songs={"air": STILL}))
        go = self.held()
        session.tick()
        self.assertEqual([session.finding_count(), session.findings_refused], [None, False])
        go.set()
        session.model.render()
        with self.assertRaises(RowSynthError) as refused:
            session.model.findings()
        self.assertEqual(str(refused.exception), f"songs.air: {STILL_SAID}")
        session.tick()
        self.assertEqual([session.finding_count(), session.findings_refused], [0, True])
        self.assertEqual(session.message, f"songs.air: {STILL_SAID}")
        # Put right outside, the project has its findings counted again.
        self.write(PROJECT)
        self.looked(session)
        self.assertEqual(session.message, "changed outside: air")
        session.model.findings()
        session.tick()
        self.assertEqual([session.finding_count(), session.findings_refused], [1, False])

    def test_with_nothing_picked_there_is_nothing_to_play(self):
        session = self.opened({})
        session.tick()
        session.play()
        session.toggle()
        self.assertEqual([session.playing, session.message, session.numbers(), session.seconds],
                         [False, "", None, 0.0])
        self.assertEqual(self.speaker.did("play"), [])

    def test_the_numbers_and_the_count_of_findings_are_none_until_the_workers_are_through(self):
        session = self.opened()
        go = self.held()
        session.tick()
        self.assertEqual([session.numbers(), session.finding_count()], [None, None])
        go.set()
        session.model.render()
        found = session.model.findings()
        self.assertEqual(session.numbers(), session.model.numbers())
        self.assertEqual([session.finding_count(), [finding.target for finding in found]], [1, ["songs.air"]])

    def test_the_file_is_looked_at_no_more_often_than_the_time_between_two_looks(self):
        session = self.opened()
        with mock.patch.object(session.model, "poll", wraps=session.model.poll) as poll:
            session.tick()
            session.tick()
            self.later(POLL_EVERY / 2)
            session.tick()
            self.assertEqual(poll.call_count, 1)
            self.later(POLL_EVERY / 2)
            session.tick()
            session.tick()
            self.assertEqual(poll.call_count, 2)

    def test_close_stops_the_sound_and_closes_the_player(self):
        session = self.opened()
        session.model.render()
        session.play()
        session.close()
        self.assertEqual([asked[0] for asked in self.speaker.asked], ["play", "stop", "close"])
        self.assertFalse(session.playing)


if __name__ == "__main__":
    unittest.main()
