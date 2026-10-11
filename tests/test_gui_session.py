"""Tests for rowsynth.gui.session."""
from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import threading
import unittest
from collections.abc import Callable, Sequence
from pathlib import Path
from unittest import mock

import numpy as np

from rowsynth import checks
from rowsynth.edit import NOTHING
from rowsynth.errors import RowSynthError
from rowsynth.gui import model as gui_model
from rowsynth.gui.model import KINDS, Model
from rowsynth.gui.session import BOTH, GLOW_TIME, HOLD, NOTICE_TIME, POLL_EVERY, Session, joined, piano
from rowsynth.project import Project
from rowsynth.song import mix_stems, render_stems

RATE = 8000

# The time by the clock a test begins at, and the time its first write of the file is stamped with, in nanoseconds.
START = 100.0
STAMP = 1_700_000_000 * 10**9


def sound(note: str, inst: str = "beep") -> dict:
    """A def of one voice playing one note from the start."""
    return {"voices": [{"inst": inst, "notes": [[0, note, 4]]}]}


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

# The ticks a player takes to let the loop of the held sound below go, and the samples those come to.
LET_GO = 3
LET_GO_SAMPLES = 400

# PROJECT with more to hear in it: a sound of three variants, a held sound, and a song with a drum and with an
# instrument of its own under a name the project has too.
HEARD = {
    "instruments": {**PROJECT["instruments"], "hum": {"wave": "tri", "vol": 12}},
    "sounds": {**PROJECT["sounds"],
               "trio": {"defs": [sound("c5"), sound("e5"), sound("g5")]},
               "drone": {"held": {"note": {"voices": [{"inst": "hum", "notes": [[0, "a3", 12]]}]}, "start": 2,
                                  "loop": 6, "join": 1, "end": sound("a3", "hum"), "let_go": LET_GO}}},
    "songs": {**PROJECT["songs"],
              "beat": {**song("l4 c e"), "instruments": {"beep": {"wave": "saw"}},
                       "drums": {"k": {"inst": "beep", "note": "c2"}}}},
}

# A song of one bar with four channels that all play, to mute and to solo. In MIXED it is the first song of
# PROJECT, which is the one that is picked as the file is opened.
BAND = {"row_ticks": 3,
        "channels": {"lead": {"inst": "beep", "pan": -0.5}, "bass": {"inst": "beep", "pan": 0.5},
                     "pad": {"inst": "beep"}, "top": {"inst": "beep", "gain": 0.5}},
        "patterns": {"a": {"lead": "l4 c e g > c", "bass": "o3 c2 g2", "pad": "o4 e1",
                           "top": "o5 l8 c e g e c e g e"}},
        "order": ["a"]}
MIXED = {**PROJECT, "songs": {"band": BAND, **PROJECT["songs"]}}
# The samples its loop has at RATE.
BAND_SAMPLES = 6400

# The place of the text the song "tune" plays, and what a change to it is called.
TUNE = "songs.tune.patterns.a.tune"
TUNE_LABEL = "tune of pattern a"


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


def band_mix(data: dict, *names: str) -> tuple[np.ndarray, np.ndarray]:
    """The two sides the engine mixes of the stems of the band in `data`, those of the channels named alone."""
    stems = render_stems(Project.from_dict(data).song("band"), RATE)["stems"]
    return mix_stems([stem for stem in stems if stem["name"] in names], BAND_SAMPLES)


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
        # The seconds into what plays, which a test can move.
        self.at = 0.125

    def play(self, sides: Sequence, loop: bool = False) -> None:
        self.asked.append(("play", sides, loop))
        self.playing = True

    def also(self, sides: Sequence) -> None:
        self.asked.append(("also", sides))

    def stop(self, fade: float = 0.0) -> None:
        self.asked.append(("stop", fade) if fade else ("stop",))
        self.playing = False

    def swap(self, sides: Sequence) -> None:
        self.asked.append(("swap", sides))

    def loop(self, on: bool) -> None:
        self.asked.append(("loop", on))

    def position(self) -> float:
        return self.at if self.playing else 0.0

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

    def data(self) -> dict:
        """What the file holds now."""
        return json.loads(self.path.read_text(encoding="utf-8"))

    def later(self, seconds: float) -> None:
        """Moves the clock on."""
        self.clock.now += seconds

    def looked(self, session: Session) -> None:
        """A tick at which the file is looked at again."""
        self.later(POLL_EVERY)
        session.tick()

    def through(self, session: Session) -> None:
        """A tick by which a notice that was the message has had its time. The file is looked at again as well."""
        self.later(NOTICE_TIME)
        session.tick()

    def held(self) -> threading.Event:
        """Has every render and the checks wait before they begin. Gives what lets them go on."""
        go = threading.Event()

        def waiting(real: Callable) -> Callable:
            def wait(*args: object) -> object:
                go.wait(60)
                return real(*args)
            return wait

        for owner, name in ((gui_model, "render_stems"), (gui_model, "render_entry"), (checks, "run")):
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


class HearTest(SessionCase):
    def test_a_note_sounds_over_what_plays_in_the_instrument_its_song_has_by_that_name(self):
        session = self.opened(HEARD)
        tune = session.model.render()
        session.play()
        session.sound("beep", 64)
        session.sound("beep", "e4", song="beat")
        shared, own = session.model.tone("beep", 64), session.model.tone("beep", 64, song="beat")
        self.assertIsNot(shared, own)
        (first,), (second,) = self.speaker.did("also")
        self.assertIs(first, shared["sides"])
        self.assertIs(second, own["sides"])
        # What plays goes on.
        self.assert_played((tune, True))
        self.assertEqual([self.speaker.did("stop"), session.playing, session.message], [[], True, ""])

    def test_a_drum_letter_sounds_the_instrument_and_the_note_its_song_gives_it(self):
        session = self.opened(HEARD)
        session.drum("k", "beat")
        ((sides,),) = self.speaker.did("also")
        self.assertIs(sides, session.model.tone("beep", "c2", song="beat")["sides"])
        for letter, name in (("s", "beat"), ("k", "tune"), ("k", "nowhere")):
            session.drum(letter, name)
            self.assertEqual(session.message, f"no drum {letter!r} in the song {name!r}")
        self.assertEqual(len(self.speaker.did("also")), 1)

    def test_with_no_player_a_note_and_a_drum_make_nothing_and_say_nothing(self):
        session = self.opened(HEARD, player=False)
        with mock.patch.object(session.model, "tone", wraps=session.model.tone) as tone:
            session.sound("beep", 60)
            session.drum("k", "beat")
        self.assertEqual([tone.call_count, session.message], [0, ""])

    def test_a_note_that_is_refused_does_not_sound_and_the_refusal_is_the_message(self):
        session = self.opened(HEARD)
        for said, asked in (("no instrument named 'bell'", ("bell", 60)), ("not a note: 'x9'", ("beep", "x9"))):
            session.sound(*asked)
            self.assertEqual(session.message, said)
        self.assertEqual(self.speaker.asked, [])

    def test_a_sound_with_variants_plays_them_in_turn_or_the_one_that_is_asked_for(self):
        session = self.opened(HEARD)
        session.audition = False
        session.pick("sound", "trio")
        made = session.model.render()
        for which in (None, None, None, None, 2, None, 0, 4):
            session.play(which)
        # Picked again, it is played from its first variant again.
        session.audition = True
        session.pick("sound", "trio")
        session.tick()
        session.play()
        played = self.speaker.did("play")
        self.assertEqual([loop for _, loop in played], [False] * 10)
        for (sides, _), at in zip(played, (0, 1, 2, 0, 2, 0, 0, 1, 0, 1)):
            self.assertEqual(len(sides), 1)
            self.assertIs(sides[0], made["buffers"][at])

    def test_a_held_sound_is_played_through_once_or_held_until_it_is_let_go(self):
        session = self.opened(HEARD)
        session.audition = False
        session.pick("sound", "drone")
        start, loop, end = session.model.render()["buffers"]
        session.play()
        # Nothing holds it, so there is nothing to let go.
        session.release()
        session.hold()
        self.assertTrue(session.playing)
        self.assertEqual([asked[0] for asked in self.speaker.asked], ["stop", "play", "play"])
        rounds = math.ceil(HOLD * RATE / len(loop))
        for (sides, looped), held in zip(self.speaker.did("play"), (1, rounds)):
            self.assertFalse(looped)
            self.assertEqual(len(sides), 1)
            np.testing.assert_array_equal(sides[0], joined(start, loop, end, LET_GO_SAMPLES, held))
        self.assertGreaterEqual(rounds * len(loop), HOLD * RATE)
        # Let go, its loop dies away over the ticks of its "let_go" while its end is played over it.
        session.release()
        self.assertEqual([asked[0] for asked in self.speaker.asked[3:]], ["stop", "also"])
        self.assertEqual(self.speaker.asked[3][1], LET_GO / 60)
        (ended,) = self.speaker.asked[4][1]
        self.assertIs(ended, end)
        self.assertFalse(session.playing)
        # It is let go once. A hold that was stopped is not let go at all, and neither is one that went on until
        # the sound was let go by itself.
        session.release()
        session.hold()
        session.stop()
        session.release()
        session.hold()
        self.speaker.at = HOLD + 1
        session.release()
        self.assertEqual([asked[0] for asked in self.speaker.asked[5:]], ["play", "stop", "play"])

    def test_a_hold_that_is_let_go_before_its_render_is_there_plays_the_sound_through_once(self):
        session = self.opened(HEARD)
        session.audition = False
        go = self.held()
        session.pick("sound", "drone")
        session.hold()
        session.release()
        self.assertTrue(session.playing)
        go.set()
        buffers = session.model.render()["buffers"]
        session.tick()
        ((sides, looped),) = self.speaker.did("play")
        np.testing.assert_array_equal(sides[0], joined(*buffers, LET_GO_SAMPLES))
        session.release()
        self.assertEqual([looped, self.speaker.did("also")], [False, []])


class JoinedTest(unittest.TestCase):
    def test_the_end_is_laid_over_the_loop_as_it_dies_away_after_the_rounds_it_was_held_for(self):
        start, loop, end = (np.array(row, dtype=np.float32) for row in ([1, 1], [2, 4], [8, 8, 8, 8, 8]))
        self.assertEqual(joined(start, loop, end, 4).tolist(), [1, 1, 2, 4, 10, 11, 9, 9, 8])
        self.assertEqual(joined(start, loop, end, 4, rounds=3).tolist(), [1, 1, 2, 4, 2, 4, 2, 4, 10, 11, 9, 9, 8])
        # A letting go that is longer than the end runs on past it, and one of no length leaves the end as it is.
        self.assertEqual(joined(start, loop, end[:1], 4).tolist(), [1, 1, 2, 4, 10, 3, 1, 1])
        self.assertEqual(joined(start, loop, end, 0).tolist(), [1, 1, 2, 4, 8, 8, 8, 8, 8])
        self.assertEqual(joined(start, loop, end, 4).dtype, np.float32)


class PianoTest(unittest.TestCase):
    def test_the_two_rows_of_keys_are_the_notes_from_c_up_and_the_upper_row_runs_on_past_its_octave(self):
        self.assertEqual([piano(key) for key in "zsxdcvgbhnjm"], list(range(12)))
        self.assertEqual([piano(key) for key in "q2w3er5t6y7u"], list(range(12, 24)))
        self.assertEqual([piano(key) for key in "i9o0p"], list(range(24, 29)))

    def test_a_key_that_plays_no_note_is_none(self):
        for key in ("a", "1", "space", "", "Z", "zs", "[2]"):
            with self.subTest(key=key):
                self.assertIsNone(piano(key))


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
        with mock.patch.object(gui_model, "render_stems", wraps=gui_model.render_stems) as engine, \
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
        with mock.patch.object(gui_model, "render_stems", wraps=gui_model.render_stems) as engine:
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
        with mock.patch.object(gui_model, "render_stems", side_effect=ZeroDivisionError("broken")) as engine, \
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


class ChangeTest(SessionCase):
    def test_a_change_is_in_the_file_at_once_and_can_be_taken_back_and_made_again(self):
        session = self.opened()
        session.tick()
        self.assertEqual([session.can_undo, session.can_redo, session.undo_label, session.redo_label],
                         [False, False, "", ""])
        self.assertEqual([session.undo(), session.redo()], [False, False])
        after = changed(songs={"tune": song("l2 c e")})
        self.assertTrue(session.change(TUNE, "l2 c e", TUNE_LABEL))
        self.assertEqual(self.data(), after)
        self.assertEqual(session.model.project, Project.from_dict(after))
        self.assertEqual([session.can_undo, session.can_redo, session.undo_label, session.message],
                         [True, False, TUNE_LABEL, ""])
        self.assertTrue(session.undo())
        self.assertEqual(self.data(), PROJECT)
        self.assertEqual([session.can_undo, session.can_redo, session.redo_label], [False, True, TUNE_LABEL])
        self.assertTrue(session.redo())
        self.assertEqual(self.data(), after)
        self.assertEqual([session.can_undo, session.can_redo, session.message], [True, False, ""])
        # A value that is there already is no step.
        self.assertTrue(session.change("songs.air.row_ticks", 3, "tempo of air"))
        self.assertEqual(session.undo_label, TUNE_LABEL)

    def test_what_is_removed_is_put_back_by_undo_and_an_item_of_a_list_is_put_back_where_it_was(self):
        session = self.opened()
        session.audition = False
        session.pick("sound", "pop")
        self.assertTrue(session.remove("sounds.pop", "pop"))
        self.assertEqual(self.data(), changed(sounds={"pop": None}))
        # The picked thing is gone, as when it is removed outside.
        self.assertEqual(session.model.picked, ("song", "tune"))
        self.assertTrue(session.undo())
        self.assertEqual(self.data(), PROJECT)
        vol = "instruments.beep.vol"
        self.assertTrue(session.remove(f"{vol}[1]", "volume of beep"))
        self.assertEqual(self.data()["instruments"]["beep"]["vol"], [15, 9, 6])
        self.assertTrue(session.undo())
        self.assertEqual(self.data(), PROJECT)
        self.assertFalse(session.remove(f"{vol}[4]", "volume of beep"))
        self.assertEqual(session.message, "instruments.beep.vol has no item [4]: it holds 4, counted from 0")

    def test_a_change_that_is_refused_is_not_made_adds_no_step_and_is_the_message(self):
        session = self.opened()
        session.tick()
        written = self.path.read_bytes()
        self.assertFalse(session.change("songs.tune.channels.tune.inst", "bell", "instrument of tune"))
        self.assertEqual(session.message, "songs.tune.channels.tune.inst: no instrument named 'bell'")
        self.assertEqual([session.can_undo, self.path.read_bytes()], [False, written])
        # A change that is made has nothing to say.
        self.assertTrue(session.change(TUNE, "l2 c e", TUNE_LABEL))
        self.assertEqual(session.message, "")

    def test_a_change_from_outside_is_a_step_and_taking_it_back_puts_the_old_things_back_in_the_file(self):
        before = changed(instruments={"hum": {"wave": "tri"}}, sounds={"pop": {"def": sound("c5", "hum")}})
        session = self.opened(before)
        session.tick()
        # Outside, pop goes over to a new instrument and the one it had is removed, and more beside.
        after = changed(instruments={"bell": {"wave": "saw"}}, songs={"tune": song("l2 c e")},
                        sounds={"pop": {"def": sound("c5", "bell")}, "zap": None})
        self.write(after)
        self.looked(session)
        label = "changed outside: bell, hum, pop and 2 more"
        self.assertEqual([session.message, session.undo_label, session.can_redo], [label, label, False])
        self.assertTrue(session.undo())
        self.assertEqual(self.data(), before)
        self.assertEqual([session.can_undo, session.redo_label, session.message], [False, label, ""])
        self.assertTrue(session.redo())
        self.assertEqual(self.data(), after)
        # Both were made here, so a look at the file finds nothing in them.
        self.looked(session)
        self.assertEqual([session.undo_label, session.can_redo, session.message], [label, False, ""])

    def test_a_change_made_just_after_one_from_outside_to_the_same_place_stands_and_undo_brings_the_other_back(self):
        session = self.opened()
        session.tick()
        outside = changed(songs={"tune": song("l8 c d e f")})
        self.write(outside)
        self.assertTrue(session.change(TUNE, "l2 c e", TUNE_LABEL))
        self.assertEqual(self.data(), changed(songs={"tune": song("l2 c e")}))
        self.assertEqual(session.message, "tune of pattern a was changed outside at the same moment: this change "
                                          "stands, and undo brings the other back")
        self.looked(session)
        self.assertEqual(session.glows(), {})
        self.assertTrue(session.undo())
        self.assertEqual(self.data(), outside)
        self.assertEqual([session.message, session.can_undo], ["", False])

    def test_another_thing_changed_outside_just_before_a_change_made_here_glows_and_is_a_step_before_it(self):
        session = self.opened()
        session.tick()
        outside = changed(sounds={"pop": {"def": sound("d5")}})
        self.write(outside)
        self.assertTrue(session.change(TUNE, "l2 c e", TUNE_LABEL))
        self.assertEqual(self.data(), changed(sounds=outside["sounds"], songs={"tune": song("l2 c e")}))
        # The other program's thing glows and is named, and the thing changed here does neither.
        self.assertEqual([session.glows(), session.message], [{("sound", "pop"): 1.0}, "changed outside: pop"])
        # It was taken up with the change, so the next look finds nothing more in it.
        self.looked(session)
        self.assertEqual([set(session.glows()), session.undo_label], [{("sound", "pop")}, TUNE_LABEL])
        self.assertTrue(session.undo())
        self.assertEqual([self.data(), session.undo_label], [outside, "changed outside: pop"])
        self.assertTrue(session.undo())
        self.assertEqual([self.data(), session.can_undo], [PROJECT, False])

    def test_other_things_changed_outside_just_before_several_changes_made_as_one_are_a_step_before_theirs(self):
        session = self.opened()
        session.tick()
        # Outside, pop goes over to an instrument that is new.
        outside = changed(instruments={"bell": {"wave": "saw"}}, sounds={"pop": {"def": sound("c5", "bell")}})
        self.write(outside)
        air = "songs.air.patterns.a.tune"
        self.assertTrue(session.change_many([(TUNE, "l2 c e"), (air, "l2 g e")], "two tunes"))
        after = copy.deepcopy(outside)
        after["songs"] = {"tune": song("l2 c e"), "air": song("l2 g e")}
        self.assertEqual(self.data(), after)
        self.assertEqual(session.glows(), {("instrument", "bell"): 1.0, ("sound", "pop"): 1.0})
        self.assertEqual([session.message, session.undo_label], ["changed outside: bell, pop", "two tunes"])
        self.assertTrue(session.undo())
        self.assertEqual([self.data(), session.undo_label], [outside, "changed outside: bell, pop"])
        self.assertTrue(session.undo())
        self.assertEqual([self.data(), session.can_undo], [PROJECT, False])

    def test_another_place_of_the_same_thing_changed_outside_is_put_back_by_taking_its_step_back(self):
        session = self.opened()
        session.tick()
        outside = copy.deepcopy(PROJECT)
        outside["songs"]["tune"]["channels"]["tune"]["gain"] = 0.5
        self.write(outside)
        self.assertTrue(session.change(TUNE, "l2 c e", TUNE_LABEL))
        after = copy.deepcopy(outside)
        after["songs"]["tune"]["patterns"]["a"]["tune"] = "l2 c e"
        self.assertEqual(self.data(), after)
        # The place changed here held what the project here had, so there is nothing to say of it.
        self.assertEqual([session.glows(), session.message], [{("song", "tune"): 1.0}, "changed outside: tune"])
        # Each step is taken back and made again alone: the one made here first, and the other program's after it.
        for step, wanted in ((session.undo, outside), (session.undo, PROJECT), (session.redo, outside),
                             (session.redo, after)):
            self.assertTrue(step())
            self.assertEqual(self.data(), wanted)

    def test_what_changed_outside_just_before_a_step_is_taken_back_is_a_step_on_top_of_it(self):
        session = self.opened()
        session.tick()
        self.assertTrue(session.change(TUNE, "l2 c e", TUNE_LABEL))
        outside = changed(songs={"tune": song("l2 c e")})
        outside["songs"]["tune"]["channels"]["tune"]["gain"] = 0.5
        self.write(outside)
        self.assertTrue(session.undo())
        back = copy.deepcopy(PROJECT)
        back["songs"]["tune"]["channels"]["tune"]["gain"] = 0.5
        self.assertEqual(self.data(), back)
        self.assertEqual([session.glows(), session.message], [{("song", "tune"): 1.0}, "changed outside: tune"])
        self.assertEqual([session.undo_label, session.can_redo], ["changed outside: tune", False])
        # Taking the other program's step back leaves the step that was taken back before it as it is.
        self.assertTrue(session.undo())
        self.assertEqual([self.data(), session.can_undo], [PROJECT, False])

    def test_an_undo_that_is_refused_writes_nothing_and_keeps_its_step(self):
        hum = {"hum": {"wave": "tri"}}
        session = self.opened(changed(instruments=hum, sounds={"zap": {"def": sound("a4", "hum")}}))
        session.tick()
        # Outside, zap goes over to the other instrument and pop gets another note: one step of two changes.
        after = changed(instruments=hum, sounds={"zap": {"def": sound("b4")}, "pop": {"def": sound("d5")}})
        self.write(after)
        self.looked(session)
        # Then the instrument zap had is removed outside, which no look has found as the step is taken back.
        del after["instruments"]["hum"]
        self.write(after)
        written = self.path.read_bytes()
        # The two changes are made as one, so pop is not put back alone.
        self.assertFalse(session.undo())
        self.assertEqual(self.path.read_bytes(), written)
        self.assertEqual(session.message, "sounds.zap.def.voices[0].inst: no instrument named 'hum'")
        self.assertEqual([session.undo_label, session.can_redo], ["changed outside: pop, zap", False])

    def test_several_places_changed_as_one_are_one_step_that_is_taken_back_and_made_again_in_one_write(self):
        session = self.opened()
        session.tick()
        # The instrument under another name, in its table and in every sound and song: none of these alone holds.
        after = json.loads(json.dumps(PROJECT).replace('"beep"', '"bell"'))
        changes = [("instruments", after["instruments"]),
                   *((f"{table}.{name}", after[table][name]) for table in ("sounds", "songs") for name in after[table])]
        label = "instrument beep renamed to bell"
        self.assertTrue(session.change_many(changes, label))
        self.assertEqual(self.data(), after)
        self.assertEqual(session.model.project, Project.from_dict(after))
        self.assertEqual([session.undo_label, session.can_redo, session.message], [label, False, ""])
        with mock.patch.object(os, "replace", wraps=os.replace) as replaced:
            self.assertTrue(session.undo())
            self.assertEqual([replaced.call_count, self.data(), session.redo_label], [1, PROJECT, label])
            self.assertTrue(session.redo())
            self.assertEqual([replaced.call_count, self.data(), session.can_undo], [2, after, True])
        # What is refused is made nowhere and is no step, and neither are values that are there already.
        written = self.path.read_bytes()
        self.assertFalse(session.change_many([("songs.tune.gain", 0.3), ("instruments.bell", NOTHING)], "bell gone"))
        self.assertEqual(session.message, "sounds.zap.def.voices[0].inst: no instrument named 'bell'")
        self.assertTrue(session.change_many([("songs.tune.row_ticks", 3), ("songs.air.row_ticks", 3)], "tempos"))
        self.assertEqual([self.path.read_bytes(), session.undo_label, session.message], [written, label, ""])

    def test_a_change_made_here_does_not_glow_and_has_the_render_and_the_checks_made_again(self):
        session = self.opened()
        made = session.model.render()
        session.play()
        session.tick()
        settle()
        session.tick()
        self.assertEqual(session.finding_count(), 1)
        go = self.held()
        self.assertTrue(session.change(TUNE, "l2 c e", TUNE_LABEL))
        session.tick()
        self.assertEqual([session.glows(), session.numbers(), session.finding_count()], [{}, None, None])
        self.assertTrue(session.playing)
        self.assert_swapped()
        go.set()
        again = session.model.render()
        session.model.findings()
        session.tick()
        self.assertIsNot(again, made)
        # The checks were run on the project as it is now, in which they find one thing more.
        self.assertEqual([session.numbers(), session.finding_count()], [session.model.numbers(), 2])
        self.assert_swapped(again)
        # A step taken back is a change made here as well.
        self.assertTrue(session.undo())
        back = session.model.render()
        session.tick()
        self.assert_swapped(again, back)
        self.assertEqual(session.glows(), {})
        self.assert_played((made, True))
        self.assertEqual(self.speaker.did("stop"), [])

    def test_changes_with_the_same_key_are_taken_back_in_one_and_changes_with_none_one_by_one(self):
        session = self.opened()
        level = "songs.tune.channels.tune.gain"
        for value in (0.9, 0.8, 0.7):
            self.assertTrue(session.change(level, value, "level of tune", key="level"))
        self.assertTrue(session.change(TUNE, "l2 c e", TUNE_LABEL))
        self.assertTrue(session.undo())
        self.assertEqual(session.undo_label, "level of tune")
        self.assertTrue(session.undo())
        self.assertEqual([self.data(), session.can_undo], [PROJECT, False])
        self.assertTrue(session.redo())
        self.assertEqual(self.data()["songs"]["tune"]["channels"]["tune"], {"inst": "beep", "gain": 0.7})
        for value in (0.6, 0.5):
            self.assertTrue(session.change(level, value, "level of tune"))
        self.assertTrue(session.undo())
        self.assertEqual(self.data()["songs"]["tune"]["channels"]["tune"]["gain"], 0.6)

    def test_a_step_keeps_the_value_it_was_made_with_whatever_becomes_of_the_one_handed_in(self):
        session = self.opened()
        vol = [15, 10, 5]
        self.assertTrue(session.change("instruments.beep.vol", vol, "volume of beep"))
        vol.append(0)
        self.assertTrue(session.undo())
        self.assertTrue(session.redo())
        self.assertEqual(self.data()["instruments"]["beep"]["vol"], [15, 10, 5])


class MuteTest(SessionCase):
    def playing(self) -> tuple[Session, dict]:
        """A session on MIXED whose band plays as a loop, and the render of the whole band."""
        session = self.opened(MIXED)
        whole = session.model.render()
        session.play()
        return session, whole

    def mixed(self, session: Session) -> dict:
        """The render the model has once a mix that is on its way is through, taken up by a tick."""
        settle()
        session.tick()
        return session.model.render()

    def assert_listening(self, session: Session, muted: set[str], soloed: set[str], heard: set[str] | None) -> None:
        """Holds a session to what is muted and what is soloed, and its model to the channels that are heard."""
        self.assertEqual([session.muted, session.soloed, session.model.heard], [muted, soloed, heard])

    def test_a_mute_has_all_the_other_channels_heard_and_a_loop_that_plays_goes_on_with_the_mix(self):
        session, whole = self.playing()
        self.assert_listening(session, set(), set(), None)
        self.assertTrue(session.mute("bass"))
        self.assert_listening(session, {"bass"}, set(), {"lead", "pad", "top"})
        mix = self.mixed(session)
        session.tick()
        # The loop was given the mix to go on with, the once: it was not stopped, and not played from its start.
        self.assert_swapped(mix)
        self.assert_played((whole, True))
        self.assertEqual([self.speaker.did("stop"), session.playing, session.message], [[], True, ""])
        # The mix is the band without its bass, to within rounding, and the rest of the render is the band's.
        for side, wanted in zip(mix["sides"], band_mix(MIXED, "lead", "pad", "top"), strict=True):
            np.testing.assert_allclose(side, wanted, rtol=0, atol=1e-6)
        self.assertGreater(float(np.abs(mix["sides"][1] - whole["sides"][1]).max()), 0.01)
        self.assertEqual(session.seconds, whole["seconds"])

    def test_a_second_press_unmutes_and_the_loop_goes_on_with_the_whole_render_at_once(self):
        session, whole = self.playing()
        session.mute("bass")
        mix = self.mixed(session)
        self.assertTrue(session.mute("bass"))
        self.assert_listening(session, set(), set(), None)
        # The render of the whole song is kept, so it is there with no tick and nothing mixed.
        self.assert_swapped(mix, whole)
        self.assertIs(session.model.render(), whole)
        self.assert_played((whole, True))

    def test_a_solo_wins_over_the_mutes_and_a_mute_pressed_under_it_gives_the_loop_nothing_new(self):
        session, _ = self.playing()
        session.mute("bass")
        session.mute("pad")
        self.assertTrue(session.solo("bass"))
        self.assert_listening(session, {"bass", "pad"}, {"bass"}, {"bass"})
        mix = self.mixed(session)
        swaps = self.speaker.did("swap")
        self.assertIs(swaps[-1][0], mix["sides"])
        self.assertTrue(session.mute("top"))
        self.assert_listening(session, {"bass", "pad", "top"}, {"bass"}, {"bass"})
        self.assertIs(self.mixed(session), mix)
        self.assertEqual(len(self.speaker.did("swap")), len(swaps))

    def test_two_solos_are_both_heard_and_no_other_channel(self):
        session = self.opened(MIXED)
        session.solo("lead")
        session.solo("top")
        self.assert_listening(session, set(), {"lead", "top"}, {"lead", "top"})
        for side, wanted in zip(session.model.render()["sides"], band_mix(MIXED, "lead", "top"), strict=True):
            np.testing.assert_array_equal(side, wanted)

    def test_a_solo_taken_off_gives_the_mutes_back(self):
        session = self.opened(MIXED)
        session.mute("pad")
        session.solo("lead")
        self.assert_listening(session, {"pad"}, {"lead"}, {"lead"})
        self.assertTrue(session.solo("lead"))
        self.assert_listening(session, {"pad"}, set(), {"lead", "bass", "top"})

    def test_hear_all_takes_every_mute_and_every_solo_off(self):
        session, whole = self.playing()
        session.mute("pad")
        session.solo("lead")
        self.mixed(session)
        session.hear_all()
        self.assert_listening(session, set(), set(), None)
        swaps = self.speaker.did("swap")
        self.assertIs(swaps[-1][0], whole["sides"])
        # With nothing muted and nothing soloed there is nothing to take off, and the loop is given nothing.
        session.hear_all()
        self.assertEqual(len(self.speaker.did("swap")), len(swaps))

    def test_a_channel_that_is_gone_from_the_file_is_muted_and_soloed_no_longer(self):
        session, whole = self.playing()
        session.tick()
        session.mute("bass")
        session.mute("pad")
        session.solo("top")
        self.assert_listening(session, {"bass", "pad"}, {"top"}, {"top"})
        after = copy.deepcopy(MIXED)
        for name in ("pad", "top"):
            del after["songs"]["band"]["channels"][name], after["songs"]["band"]["patterns"]["a"][name]
        self.write(after)
        self.looked(session)
        # The mute that is left counts, now that nothing is soloed.
        self.assert_listening(session, {"bass"}, set(), {"lead"})
        mix = self.mixed(session)
        self.assertIs(self.speaker.did("swap")[-1][0], mix["sides"])
        for side, wanted in zip(mix["sides"], band_mix(after, "lead"), strict=True):
            np.testing.assert_array_equal(side, wanted)
        self.assert_played((whole, True))

    def test_another_thing_picked_has_nothing_muted_or_soloed_and_the_same_thing_picked_again_keeps_them(self):
        session = self.opened(MIXED)
        session.mute("bass")
        session.solo("lead")
        session.pick("song", "band")
        self.assert_listening(session, {"bass"}, {"lead"}, {"lead"})
        session.pick("song", "tune")
        self.assert_listening(session, set(), set(), None)
        # The song that takes the place of one removed outside has a channel of the name that was muted.
        self.assertTrue(session.mute("tune"))
        self.write(changed(songs={"band": BAND, "tune": None}))
        self.looked(session)
        self.assertEqual(session.model.picked, ("song", "air"))
        self.assert_listening(session, set(), set(), None)

    def test_a_press_while_nothing_plays_is_what_the_next_play_sounds_like(self):
        session = self.opened(MIXED)
        self.assertTrue(session.solo("bass"))
        settle()
        self.assertEqual(self.speaker.asked, [])
        session.play()
        mix = session.model.render()
        self.assert_played((mix, True))
        for side, wanted in zip(mix["sides"], band_mix(MIXED, "bass"), strict=True):
            np.testing.assert_array_equal(side, wanted)

    def test_what_is_no_channel_of_the_picked_song_is_refused_in_the_message_and_changes_nothing(self):
        session, _ = self.playing()
        session.mute("bass")
        mix = self.mixed(session)
        for said, (press, channel) in enumerate(((session.mute, "drums"), (session.solo, "tune"),
                                                 (session.solo, ["bass"])), 1):
            self.assertFalse(press(channel))
            self.assertEqual([session.message, session.messages],
                             [f"the song 'band' has no channel named {channel!r}", said])
        self.assert_listening(session, {"bass"}, set(), {"lead", "pad", "top"})
        self.assertIs(self.mixed(session), mix)
        self.assert_swapped(mix)
        # What is no song has no channels at all.
        session.pick("sound", "zap")
        self.assertFalse(session.mute("bass"))
        self.assertEqual(session.message, "no song is picked, so there are no channels to mute or solo")
        self.assert_listening(session, set(), set(), None)

    def test_the_levels_are_read_at_the_place_the_song_plays_at_and_a_muted_channel_reads_0_at_once(self):
        session = self.opened(MIXED)
        session.model.render()
        self.assertEqual(session.levels(), {})
        session.play()
        early = session.levels()
        self.assertEqual(early, session.model.levels(0.125))
        self.assertEqual(list(early), list(BAND["channels"]))
        self.assertGreater(min(early.values()), 0.01)
        self.speaker.at = 0.5
        late = session.levels()
        self.assertEqual(late, session.model.levels(0.5))
        self.assertNotEqual(late, early)
        # A channel reads 0 from the press that mutes it, before any mix is taken up, and the others as they did.
        session.mute("bass")
        self.assertEqual(session.levels(), {**late, "bass": 0.0})
        session.stop()
        self.assertEqual(session.levels(), {})

    def test_the_numbers_stay_those_of_the_whole_song_while_a_mix_is_made_and_after(self):
        session, _ = self.playing()
        session.tick()
        numbers = session.numbers()
        self.assertEqual(numbers, session.model.numbers())
        go = threading.Event()
        # A test that fails before it lets the mix go leaves no thread waiting.
        self.addCleanup(go.set)

        def held(*args: object) -> tuple:
            go.wait(60)
            return mix_stems(*args)

        with mock.patch.object(gui_model, "mix_stems", side_effect=held):
            self.assertTrue(session.solo("bass"))
            session.tick()
            self.assertFalse(session.model.render_ready())
            self.assertIs(session.numbers(), numbers)
            # The loop goes on as it was until the mix is there.
            self.assert_swapped()
            self.assertTrue(session.playing)
            go.set()
            settle()
        session.tick()
        self.assert_swapped(session.model.render())
        self.assertIs(session.numbers(), numbers)


class NoticeTest(SessionCase):
    def test_a_pick_does_not_wipe_a_notice_from_the_status_line_and_the_next_notice_has_its_turn_after_it(self):
        session = self.opened()
        session.tick()
        self.write(changed(songs={"tune": song("l8 c d e f")}))
        self.assertTrue(session.change(TUNE, "l2 c e", TUNE_LABEL))
        notice = f"{TUNE_LABEL} {BOTH}"
        self.assertEqual([session.message, session.status], [notice, notice])
        # Another thing changes outside while that notice has its time, and then a thing is picked.
        self.write(changed(songs={"tune": song("l2 c e")}, sounds={"pop": None}))
        self.looked(session)
        said = session.messages
        session.pick("song", "air")
        # The pick clears the message as it did before, and the status line keeps the notice.
        self.assertEqual([session.message, session.status, session.messages], ["", notice, said])
        self.later(NOTICE_TIME - POLL_EVERY)
        session.tick()
        self.assertEqual([session.status, session.messages], ["changed outside: pop", said + 1])
        # With the notices through, the line is the message, which the pick cleared.
        self.through(session)
        self.assertEqual([session.status, session.messages], ["", said + 1])

    def test_notices_in_a_row_have_the_status_line_in_turn_and_those_that_wait_name_what_changed_as_one(self):
        session = self.opened()
        session.tick()
        # Outside, a thing more is removed before each of three looks: pop, then zap, then air.
        for sounds, songs in (({"pop": None}, {}), ({"pop": None, "zap": None}, {}),
                              ({"pop": None, "zap": None}, {"air": None})):
            self.write(changed(sounds=sounds, songs=songs))
            self.looked(session)
        # Each is the message as it comes, and the first has the status line for its time.
        self.assertEqual([session.message, session.status, session.messages],
                         ["changed outside: air", "changed outside: pop", 3])
        self.through(session)
        self.assertEqual([session.status, session.messages], ["changed outside: air, zap", 4])
        # Each look is a step of its own all the same. With the notices through, the line is the message.
        self.through(session)
        self.assertEqual([session.status, session.messages, session.undo_label],
                         ["changed outside: air", 4, "changed outside: air"])

    def test_what_is_said_while_a_notice_has_the_status_line_has_it_after_the_notice_and_else_at_once(self):
        session = self.opened()
        session.tick()
        session.say("one")
        session.say("two")
        self.assertEqual([session.message, session.status, session.messages], ["two", "two", 2])
        self.write(changed(sounds={"pop": None}))
        self.looked(session)
        session.say("three")
        session.say("four")
        self.assertEqual([session.message, session.status, session.messages], ["four", "changed outside: pop", 5])
        # The line is then what was said last, which is new to it: it is counted once more.
        self.through(session)
        self.assertEqual([session.message, session.status, session.messages], ["four", "four", 6])
        session.say("five")
        self.assertEqual([session.status, session.messages], ["five", 7])


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

    def test_what_say_is_given_is_the_message_until_something_else_is_said_or_a_change_is_made(self):
        session = self.opened()
        self.assertEqual([session.message, session.messages], ["", 0])
        session.say("the notes were not saved")
        self.assertEqual([session.message, session.messages], ["the notes were not saved", 1])
        # A change that is made has nothing to say, and nothing said is not one more message.
        self.assertTrue(session.change(TUNE, "l2 c e", TUNE_LABEL))
        self.assertEqual([session.message, session.messages], ["", 1])
        session.say("once more")
        session.say("")
        self.assertEqual([session.message, session.messages], ["", 2])

    def test_each_thing_said_is_counted_also_the_same_words_again_and_an_unread_file_once_while_it_lasts(self):
        session = self.opened()
        session.tick()
        refusal = "songs.tune.channels.tune.inst: no instrument named 'bell'"
        for count in (1, 2):
            self.assertFalse(session.change("songs.tune.channels.tune.inst", "bell", "instrument of tune"))
            self.assertEqual([session.message, session.messages], [refusal, count])
        # What changed outside is said by the session itself, and counted as what `say` is given.
        self.write(changed(sounds={"pop": None}))
        self.looked(session)
        self.assertEqual([session.message, session.messages], ["changed outside: pop", 3])
        # A file that cannot be read is the message through the model: it is one more when it begins.
        whole = json.dumps(PROJECT)
        self.write(whole[:len(whole) // 2])
        for _ in range(2):
            self.looked(session)
            self.assertEqual([session.message == session.model.problem, session.messages], [True, 4])
        self.write(whole)
        self.looked(session)
        self.assertEqual([session.message, session.messages], ["changed outside: pop", 5])

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
