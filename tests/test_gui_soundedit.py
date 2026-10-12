"""Tests for rowsynth.gui.soundedit: a sound effect changed, and that what comes back still loads and renders."""
from __future__ import annotations

import json
import random
import time
import unittest
from collections.abc import Callable
from pathlib import Path

import numpy as np

from rowsynth.bank import RENDER_GAIN, render_entry
from rowsynth.errors import RowSynthError
from rowsynth.gui.soundedit import (HIGHEST, LOWEST, as_held, as_one, as_parts, as_variants, boxed, copied, held_with,
                                    kind_of, mark_added, mark_dropped, mark_set, meta_dropped, meta_set, moved,
                                    part_added, part_at, part_dropped, part_level, pasted, placed, removed, shared,
                                    snapped, sound_in, stretched, variant_added, variant_dropped, variant_moved,
                                    voice_added, voice_copied, voice_dropped, voice_inst, with_echo, with_setting,
                                    with_sound, with_velocity)
from rowsynth.notes import midi
from rowsynth.project import Project
from rowsynth.sound import render_sound

EXAMPLE = Path(__file__).resolve().parent / "data" / "arcade.json"

# A tick is a hundred samples at 6000 Hz, so a length in whole ticks is a whole number of samples. The random runs
# are rendered at a lower rate still.
RATE = 6000
QUICK = 2000

# `beep` fades for one tick and `soft` for three.
INSTRUMENTS = {"beep": {"vol": [15, 10, 5], "release": 1}, "soft": {"wave": "tri", "release": 3}}
ECHO = {"time": 0.05, "feedback": 0.2, "mix": 0.4, "lp": 2000}

# The held sound the README adds to the sounds of the example.
HUM = {"held": {"note": {"voices": [{"inst": "bass", "notes": [[0, "c3", 60]]}]}, "start": 6, "loop": 30, "join": 3,
                "end": {"voices": [{"inst": "bass", "notes": [[0, "c3", 2]]}]}, "let_go": 4}, "db": -6}

# What the random sounds are made of: notes as names and as numbers, lengths, velocities and instruments.
NOTES = ("c3", "eb3", "g#4", "a4", "bb5", "c6", 36, 60, 72.5, 95)
LENGTHS = (0, 1, 2, 4, 8, 2.5)
VELOCITIES = (None, None, 0, 0.5, 1)
INSTS = ("beep", "soft", {"wave": "saw", "vol": [12, 8, 4]})


def sound(*voices: list, inst: object = "beep", **more: object) -> dict:
    """A definition with a voice for each list of notes, every voice on one instrument."""
    return {"voices": [{"inst": inst, "notes": notes} for notes in voices], **more}


def held(**lengths: float) -> dict:
    """The held sound of the README on the instruments here, with any of its lengths as given."""
    return {"held": {"note": sound([[0, "c3", 60]]), "start": 6, "loop": 30, "join": 3,
                     "end": sound([[0, "c3", 2]]), "let_go": 4, **lengths}, "db": -6}


def lengths(entry: dict) -> tuple:
    """The start, the loop, the join and the letting go of a held entry."""
    return tuple(entry["held"][key] for key in ("start", "loop", "join", "let_go"))


def entry_of(made: dict) -> dict:
    """An entry as it is, and a definition as the one sound of an entry."""
    return {"def": made} if "voices" in made else made


def loaded(made: dict) -> dict:
    """An entry or a definition as the project hands it out to be rendered, once it has passed the project's check."""
    return Project.from_dict({"instruments": INSTRUMENTS, "sounds": {"s": entry_of(made)}}).sound("s")


def buffers(made: dict, rate: int = RATE) -> list[np.ndarray]:
    """What the bank renders of an entry or of a definition, once it has passed the project's check."""
    return render_entry(loaded(made), rate)


def raw(defn: dict) -> np.ndarray:
    """A definition as the bank renders it before it is levelled, so that two renders can be added up."""
    return render_sound(loaded(defn)["def"], RATE, RENDER_GAIN)


def changed(test: unittest.TestCase, call: Callable[..., object], given: dict, *more: object) -> object:
    """What a function makes of a sound, once what was handed in is seen to be as it was and the new one to render."""
    before = json.dumps(given)
    made = call(given, *more)
    test.assertEqual(json.dumps(given), before)
    rendered = buffers(made[0] if isinstance(made, tuple) else made)
    test.assertTrue(any(len(samples) for samples in rendered))
    test.assertTrue(all(np.isfinite(samples).all() for samples in rendered))
    return made


def refusal(test: unittest.TestCase, call: Callable[..., object], *given: object) -> str:
    """The message a call is refused with."""
    with test.assertRaises(RowSynthError) as caught:
        call(*given)
    return str(caught.exception)


def same(test: unittest.TestCase, one: list[np.ndarray], other: list[np.ndarray]) -> None:
    """Two renders of an entry are sample for sample the same."""
    test.assertEqual(len(one), len(other))
    for samples, others in zip(one, other):
        test.assertTrue(np.array_equal(samples, others))


def scribble(value: object) -> None:
    """Empty every dict and every list that a value holds, however deep."""
    if isinstance(value, (dict, list, tuple, frozenset)):
        inside = list(value.values()) if isinstance(value, dict) else list(value)
        if isinstance(value, (dict, list)):
            value.clear()
        for part in inside:
            scribble(part)


class Notes(unittest.TestCase):
    def test_a_placed_note_is_the_last_of_its_voice_and_is_written_as_it_is_given(self) -> None:
        given = sound([[0, "c5", 4]], [])
        self.assertEqual(changed(self, placed, given, 0, 6, "eb5", 2), sound([[0, "c5", 4], [6, "eb5", 2]], []))
        self.assertEqual(changed(self, placed, given, 1, 2.5, 67, 0, 0.5), sound([[0, "c5", 4]], [[2.5, 67, 0, 0.5]]))
        # A whole number is written with nothing after a point.
        self.assertEqual(json.dumps(placed(given, 1, 4.0, 60.0, 2.0, 1.0)["voices"][1]["notes"]), "[[4, 60, 2, 1]]")

    def test_a_note_that_cannot_be_placed_is_refused(self) -> None:
        given = sound([[0, "c5", 4]])
        for more, said in (
            ((1, 0, "c5", 4), "1 is no voice of the sound, of which there are 1, counted from 0"),
            ((True, 0, "c5", 4), "True is no voice of the sound, of which there are 1, counted from 0"),
            ((0, -1, "c5", 4), "the tick a note starts at is a number, 0 or more, not -1"),
            ((0, 0, "h4", 4), "a note is a number or a name such as 'c#4', not 'h4'"),
            ((0, 0, float("nan"), 4), "a note is a number, not nan"),
            ((0, 0, 132, 4), "a note is placed from c-9 to b9, which are -96 and 131, and 132 is outside them"),
            ((0, 0, "c5", -0.5), "the ticks a note is held for are a number, 0 or more, not -0.5"),
            ((0, 0, "c5", 4, "loud"), "the velocity of a note is a number, not 'loud'"),
        ):
            with self.subTest(more=more):
                self.assertEqual(refusal(self, placed, given, *more), said)
        self.assertEqual((midi("c-9"), midi("b9")), (LOWEST, HIGHEST))

    def test_picked_notes_are_moved_by_ticks_and_by_semitones_and_the_others_stay(self) -> None:
        given = sound([[0, "c5", 4], [4, "e5", 4, 0.5]], [[2, 60, 3]])
        self.assertEqual(changed(self, moved, given, {(0, 1), (1, 0)}, 3, -2),
                         sound([[0, "c5", 4], [7, "d5", 4, 0.5]], [[5, 58, 3]]))
        self.assertEqual(moved(given, set(), 3, 3), given)

    def test_a_name_stays_a_name_in_sharps_and_a_number_a_number_each_with_its_velocity_or_none(self) -> None:
        given = sound([[0, "eb3", 2], [2, "a4", 2, 0.5], [4, 60, 2], [6, 60.5, 2, 1]])
        every = {(0, 0), (0, 1), (0, 2), (0, 3)}
        self.assertEqual(changed(self, moved, given, every, 0, 1),
                         sound([[0, "e3", 2], [2, "a#4", 2, 0.5], [4, 61, 2], [6, 61.5, 2, 1]]))
        self.assertEqual(moved(given, every, 0, -4),
                         sound([[0, "b2", 2], [2, "f4", 2, 0.5], [4, 56, 2], [6, 56.5, 2, 1]]))
        # A note that is moved in time alone keeps the spelling it has, flat and all.
        self.assertEqual(moved(given, every, 2, 0)["voices"][0]["notes"][0], [2, "eb3", 2])
        for name in ("c4", "eb3", "g#5", "B-1"):
            for by in range(-30, 31):
                with self.subTest(name=name, by=by):
                    new = moved(sound([[0, name, 1]]), {(0, 0)}, 0, by)["voices"][0]["notes"][0][1]
                    self.assertIsInstance(new, str)
                    self.assertEqual(midi(new), midi(name) + by)

    def test_a_moved_group_stops_together_at_tick_0_and_at_the_lowest_and_the_highest_note(self) -> None:
        given = sound([[2, "c4", 2], [6, 64, 4]], [[1, "c0", 1]])
        group = {(0, 0), (0, 1)}
        self.assertEqual(changed(self, moved, given, group, -10, 0), sound([[0, "c4", 2], [4, 64, 4]], [[1, "c0", 1]]))
        self.assertEqual(changed(self, moved, given, group, 0, -200),
                         sound([[2, "c-9", 2], [6, -92, 4]], [[1, "c0", 1]]))
        self.assertEqual(changed(self, moved, given, group, 0, 200), sound([[2, "g9", 2], [6, 131, 4]], [[1, "c0", 1]]))
        # A note that is past an end already is not moved back, and goes no further out.
        self.assertEqual(moved(sound([[0, 140, 1]]), {(0, 0)}, 0, 5), sound([[0, 140, 1]]))
        self.assertEqual(moved(sound([[0, 140, 1]]), {(0, 0)}, 0, -5), sound([[0, 135, 1]]))

    def test_a_name_is_not_moved_by_part_of_a_semitone_and_a_number_is(self) -> None:
        given = sound([[0, "c4", 1], [1, 60, 1]])
        self.assertEqual(refusal(self, moved, given, {(0, 0), (0, 1)}, 0, 0.5),
                         "a note written as a name is moved by whole semitones, not by 0.5")
        self.assertEqual(changed(self, moved, given, {(0, 1)}, 0.5, 0.5), sound([[0, "c4", 1], [1.5, 60.5, 1]]))
        # Beside a number that lies between two notes, a name stops a whole step short of the end.
        high = sound([[0, "c4", 1], [1, 130.5, 1]])
        self.assertEqual(moved(high, {(0, 0), (0, 1)}, 0, 9), high)

    def test_picked_notes_are_stretched_and_none_is_made_shorter_than_the_least(self) -> None:
        given = sound([[0, "c4", 4], [4, "e4", 0.5, 0.7]], [[0, 60, 8]])
        self.assertEqual(changed(self, stretched, given, {(0, 0), (0, 1)}, 3),
                         sound([[0, "c4", 7], [4, "e4", 3.5, 0.7]], [[0, 60, 8]]))
        # The least is a tick unless it is given, and a note that is shorter already is not made longer by it.
        self.assertEqual(changed(self, stretched, given, {(0, 0), (0, 1), (1, 0)}, -6),
                         sound([[0, "c4", 1], [4, "e4", 0.5, 0.7]], [[0, 60, 2]]))
        self.assertEqual(stretched(given, {(1, 0)}, -20, 3)["voices"][1]["notes"], [[0, 60, 3]])
        # The engine takes a note of no length, so the least can be none.
        self.assertEqual(changed(self, stretched, given, {(1, 0)}, -20, 0)["voices"][1]["notes"], [[0, 60, 0]])
        self.assertEqual(refusal(self, stretched, given, {(1, 0)}, 1, -1),
                         "the least a note is stretched to is a number, 0 or more, not -1")

    def test_removed_notes_are_gone_and_the_others_keep_their_order(self) -> None:
        given = sound([[0, "c4", 1], [1, "d4", 1], [2, "e4", 1]], [[0, 60, 1]])
        self.assertEqual(changed(self, removed, given, {(0, 0), (0, 2), (1, 0)}), sound([[1, "d4", 1]], []))
        self.assertEqual(removed(given, set()), given)

    def test_a_velocity_is_given_changed_and_taken_away(self) -> None:
        given = sound([[0, "c4", 1], [1, "d4", 1, 0.5]])
        self.assertEqual(changed(self, with_velocity, given, {(0, 0), (0, 1)}, 0.25),
                         sound([[0, "c4", 1, 0.25], [1, "d4", 1, 0.25]]))
        self.assertEqual(changed(self, with_velocity, given, {(0, 1)}, None), sound([[0, "c4", 1], [1, "d4", 1]]))
        self.assertEqual(refusal(self, with_velocity, given, {(0, 0)}, "loud"),
                         "the velocity of a note is a number, not 'loud'")

    def test_a_pick_that_is_no_note_of_the_sound_is_refused_by_all_of_them(self) -> None:
        given = sound([[0, "c4", 1]], [])
        calls = ((moved, (1, 1)), (stretched, (1,)), (removed, ()), (with_velocity, (1,)), (copied, ()))
        for pick in ((0, 1), (1, 0), (2, 0), (0, True), (0.0, 0), (0, 0, 0), "00", 0):
            for call, more in calls:
                with self.subTest(pick=pick, call=call.__name__):
                    self.assertEqual(refusal(self, call, given, [pick], *more),
                                     f"{pick!r} is picked, and that is no note of the sound: a note is (voice, index)")

    def test_a_box_picks_the_notes_it_touches_in_every_voice(self) -> None:
        given = sound([[0, "c4", 4], [4, "e4", 4], [8, 72, 0]], [[2, "g4", 4]])
        # A note that begins where the box ends is not touched, and neither is one that ends where it starts.
        self.assertEqual(boxed(given, 0, 4, 0, 127), {(0, 0), (1, 0)})
        self.assertEqual(boxed(given, 4, 6, 60, 67), {(0, 1), (1, 0)})
        # Either pair can be given either way round.
        self.assertEqual(boxed(given, 6, 4, 64, 60), {(0, 1)})
        # A note of no length touches the box that holds the tick it starts at.
        self.assertEqual(boxed(given, 8, 9, 72, 72), {(0, 2)})
        self.assertEqual(boxed(given, 7, 8, 0, 127), {(0, 1)})
        self.assertEqual(boxed(given, 20, 30, 0, 127), frozenset())

    def test_a_clip_begins_at_tick_0_and_is_pasted_into_the_voices_it_came_from(self) -> None:
        given = sound([[0, "c4", 2], [4, "e4", 2, 0.5]], [[6, 60, 3]])
        clip = copied(given, {(0, 1), (1, 0)})
        self.assertEqual(clip, ((0, [0, "e4", 2, 0.5]), (1, [2, 60, 3])))
        made, picked = changed(self, pasted, given, clip, 10)
        self.assertEqual(made, sound([[0, "c4", 2], [4, "e4", 2, 0.5], [10, "e4", 2, 0.5]], [[6, 60, 3], [12, 60, 3]]))
        self.assertEqual(picked, {(0, 2), (1, 1)})
        self.assertEqual((copied(given, set()), pasted(given, (), 3)), ((), (given, frozenset())))
        self.assertEqual(refusal(self, pasted, given, clip, -1),
                         "the tick a clip is put in at is a number, 0 or more, not -1")
        self.assertEqual(refusal(self, pasted, sound([]), clip, 0),
                         "1 is no voice of the sound, of which there are 1, counted from 0")

    def test_a_tick_is_snapped_to_where_the_step_of_the_grid_it_is_in_begins(self) -> None:
        self.assertEqual([snapped(6.5, 4), snapped(4, 4), snapped(3.99, 4), snapped(0.9, 0.25), snapped(7.2, 1)],
                         [4, 4, 0, 0.75, 7])
        self.assertEqual(json.dumps(snapped(6.5, 4.0)), "4")
        for grid in (0, -1, "4", True):
            with self.subTest(grid=grid):
                self.assertEqual(refusal(self, snapped, 3, grid),
                                 f"a grid has steps of more than 0 ticks, not {grid!r}")


class Overlap(unittest.TestCase):
    def test_two_notes_of_one_voice_sound_at_once_and_what_is_heard_is_their_sum(self) -> None:
        # The second note begins while the first is held, in the same voice.
        both = sound([[0, "c4", 10], [4, "g4", 10]])
        heard = raw(both)
        # In a voice each they give the very same samples: the engine does not read which voice a note is in.
        self.assertTrue(np.array_equal(heard, raw(sound([[0, "c4", 10]], [[4, "g4", 10]]))))
        # Each beside a silent copy of the other, the two add up to what is heard.
        first, second = raw(sound([[0, "c4", 10], [4, "g4", 10, 0]])), raw(sound([[0, "c4", 10, 0], [4, "g4", 10]]))
        self.assertGreater(float(np.abs(first[500:1000]).max()), 0.01)
        self.assertTrue(np.allclose(heard, first + second, atol=1e-6))
        # With the first cut short where the second begins, as a channel of a song would have it, it is another sound.
        cut = raw(sound([[0, "c4", 4], [4, "g4", 10]]))
        self.assertEqual(len(cut), len(heard))
        self.assertGreater(float(np.abs(heard - cut).max()), 0.01)

    def test_no_edit_cuts_a_note_short_or_removes_one_to_make_room(self) -> None:
        given = sound([[0, "c4", 8], [8, "e4", 4], [12, "g4", 4]])
        self.assertEqual(changed(self, placed, given, 0, 2, "a4", 20)["voices"][0]["notes"][:3],
                         given["voices"][0]["notes"])
        self.assertEqual(changed(self, moved, given, {(0, 1)}, -6, 0),
                         sound([[0, "c4", 8], [2, "e4", 4], [12, "g4", 4]]))
        self.assertEqual(changed(self, stretched, given, {(0, 0)}, 10),
                         sound([[0, "c4", 18], [8, "e4", 4], [12, "g4", 4]]))
        made, picked = changed(self, pasted, given, copied(given, {(0, 0)}), 9)
        self.assertEqual((made["voices"][0]["notes"], picked), ([*given["voices"][0]["notes"], [9, "c4", 8]], {(0, 3)}))


class Voices(unittest.TestCase):
    def test_an_added_voice_is_the_last_and_has_no_notes_yet(self) -> None:
        given = sound([[0, "c4", 2]])
        self.assertEqual(changed(self, voice_added, given, "soft"),
                         {"voices": [{"inst": "beep", "notes": [[0, "c4", 2]]}, {"inst": "soft", "notes": []}]})
        own = {"wave": "saw", "vol": [12, 8]}
        written = changed(self, voice_added, given, own)
        self.assertEqual(written["voices"][1], {"inst": own, "notes": []})
        self.assertIsNot(written["voices"][1]["inst"], own)

    def test_a_dropped_voice_takes_its_notes_with_it_and_the_last_voice_stays(self) -> None:
        given = sound([[0, "c4", 2]], [[1, "e4", 2]])
        self.assertEqual(changed(self, voice_dropped, given, 0), sound([[1, "e4", 2]]))
        self.assertEqual(refusal(self, voice_dropped, sound([]), 0),
                         "the last voice of a sound is not dropped: a sound with none has nothing to play")
        self.assertEqual(refusal(self, voice_dropped, given, 2),
                         "2 is no voice of the sound, of which there are 2, counted from 0")

    def test_a_voice_is_played_by_a_name_or_by_an_instrument_written_in_place(self) -> None:
        given = sound([[0, "c4", 2]], [[1, "e4", 2]])
        named = changed(self, voice_inst, given, 1, "soft")
        self.assertEqual([voice["inst"] for voice in named["voices"]], ["beep", "soft"])
        own = {"wave": "tri", "release": 0}
        written = changed(self, voice_inst, given, 0, own)
        self.assertEqual([voice["inst"] for voice in written["voices"]], [own, "beep"])
        self.assertIsNot(written["voices"][0]["inst"], own)
        self.assertEqual([voice["notes"] for voice in written["voices"]], [[[0, "c4", 2]], [[1, "e4", 2]]])
        # A string is a name even where nothing has it, and it is the project's check that says so.
        self.assertEqual(refusal(self, loaded, voice_inst(given, 0, "nothing")),
                         "sounds.s.def.voices[0].inst: no instrument named 'nothing'")

    def test_what_is_no_instrument_or_one_that_a_check_refuses_is_refused(self) -> None:
        given = sound([[0, "c4", 2]])
        for inst, said in ((7, "an instrument is a name, or a dict of its settings written in place, not 7"),
                           ({"volume": 3}, "inst: unknown key 'volume'"),
                           ({"duty": 5}, "inst.duty: must be a whole number from 0 to 3"),
                           ({"release": -1}, "inst.release: must be 0 or more, not -1")):
            with self.subTest(inst=inst):
                self.assertEqual(refusal(self, voice_added, given, inst), said)
                self.assertEqual(refusal(self, voice_inst, given, 0, inst), said)
        self.assertEqual(refusal(self, voice_inst, given, 1, "soft"),
                         "1 is no voice of the sound, of which there are 1, counted from 0")


class Rest(unittest.TestCase):
    def test_an_echo_is_put_on_changed_and_taken_off(self) -> None:
        given = sound([[0, "c4", 2]])
        made = changed(self, with_echo, given, ECHO)
        self.assertEqual(made, sound([[0, "c4", 2]], echo=ECHO))
        self.assertIsNot(made["echo"], ECHO)
        self.assertGreater(len(buffers(made)[0]), len(buffers(given)[0]))
        self.assertEqual(changed(self, with_echo, made, {"time": [0.03, 0.07]})["echo"], {"time": [0.03, 0.07]})
        self.assertEqual((changed(self, with_echo, made, None), with_echo(given, None)), (given, given))

    def test_an_echo_that_the_engine_or_the_project_refuses_is_refused(self) -> None:
        for echo, said in ((7, "echo: must be a dict, not 7"),
                           ({"mix": 0.3}, "echo: missing key 'time'"),
                           ({"time": 0.1, "size": 2}, "echo: unknown key 'size'"),
                           ({"time": 0.1, "lp": -1}, "echo.lp: must be 0 or more, not -1"),
                           ({"time": "long"}, "echo.time: must be a number or a list of numbers, not 'long'"),
                           ({"time": (0.1, 0.2)}, "sound.echo.time: must be a number")):
            with self.subTest(echo=echo):
                self.assertEqual(refusal(self, with_echo, sound([[0, "c4", 2]]), echo), said)

    def test_a_setting_is_set_and_removed_in_the_place_the_file_keeps_it(self) -> None:
        defn = sound([[0, "c4", 2]])
        self.assertEqual(changed(self, with_setting, defn, "gain", 0.8), sound([[0, "c4", 2]], gain=0.8))
        self.assertEqual(with_setting(sound([[0, "c4", 2]], gain=0.8), "gain", None), defn)
        entry = {"def": defn}
        for key, value in (("db", -6), ("max_seconds", 0.5), ("max_dc", 0.01), ("max_bright", 0.4),
                           ("check", [{"at": 0.01, "note": "c4"}]), ("meta", {"kind": "ui"})):
            with self.subTest(key=key):
                made = changed(self, with_setting, entry, key, value)
                self.assertEqual(made, {"def": defn, key: value})
                self.assertEqual((with_setting(made, key, None), with_setting(entry, key, None)), (entry, entry))
        # Variants have one level for all, or one each. At 6 decibels down the samples are about half the size.
        self.assertEqual(changed(self, with_setting, {"defs": [defn, defn]}, "db", [-3, -9])["db"], [-3, -9])
        peaks = [float(np.abs(buffers(made)[0]).max()) for made in (entry, with_setting(entry, "db", -6))]
        self.assertAlmostEqual(peaks[1] / peaks[0], 10 ** (-6 / 20), places=5)

    def test_a_setting_that_the_file_does_not_allow_there_is_refused(self) -> None:
        defn = sound([[0, "c4", 2]])
        entry = {"def": defn}
        of_entry = "an entry with 'def', which has 'db', 'meta', 'check', 'max_seconds', 'max_dc', 'max_bright'"
        for place, key, value, said in (
            (defn, "db", -6, "'db' is no setting of a sound definition, which has 'gain'"),
            (defn, "echo", None, "'echo' is no setting of a sound definition, which has 'gain'"),
            (entry, "gain", 1, f"'gain' is no setting of {of_entry}"),
            (entry, "def", None, f"'def' is no setting of {of_entry}"),
            ({"parts": [[0, entry]]}, "db", -6,
             "an entry of parts has no level of its own: each part has its own, which `part_level` sets"),
            (entry, "db", [-6], "entry.db: a list of levels must hold one for each of 'defs'"),
            ({"defs": [defn, defn]}, "db", [-6], "entry.db: a list of levels must hold one for each of 'defs'"),
            (entry, "max_seconds", "long", "entry.max_seconds: must be a number"),
            (entry, "db", float("inf"), "'db' is not set to inf: a file cannot write that number"),
            (entry, "check", [{"at": 0}], "entry.check[0]: missing key 'note'"),
            (defn, "gain", True, "sound.gain: must be a number"),
        ):
            with self.subTest(key=key, value=value):
                self.assertEqual(refusal(self, with_setting, place, key, value), said)


class Kinds(unittest.TestCase):
    def test_the_kind_of_an_entry_is_in_the_words_of_the_engine(self) -> None:
        defn = sound([[0, "c4", 2]])
        self.assertEqual([kind_of(entry) for entry in ({"def": defn}, {"defs": [defn]}, {"parts": [[0, {"def": defn}]]},
                                                        held())], ["def", "defs", "parts", "held"])
        self.assertEqual(refusal(self, kind_of, {"def": defn, "defs": [defn]}),
                         "entry: must have one of 'def', 'defs', 'parts' and 'held', and has 'def' and 'defs'")

    def test_one_sound_is_made_any_other_kind_and_comes_back_as_it_was(self) -> None:
        defn = sound([[0, "c4", 60]], echo=ECHO)
        more = {"meta": {"kind": "ui"}, "max_seconds": 3, "check": [{"at": 0.2, "note": "c4"}]}
        for level in ({}, {"db": -6}):
            given = {"def": defn, **level, **more}
            base = buffers(given)
            variants, parts = changed(self, as_variants, given), changed(self, as_parts, given)
            kept = changed(self, as_held, given, RATE)
            with self.subTest(level=level):
                self.assertEqual(variants, {"defs": [defn], **level, **more})
                self.assertEqual(parts, {"parts": [[0, {"def": defn, **level}]], **more})
                # The lengths are the shares the held sound of the README has, and the end has no notes.
                self.assertEqual(kept, {"held": {"note": defn, "start": 6, "loop": 30, "join": 3, "end": sound([]),
                                                 "let_go": 4}, **level, **more})
                self.assertEqual([as_one(made) for made in (given, variants, parts, kept)], [given] * 4)
                same(self, buffers(variants), base)
                same(self, buffers(parts), base)

    def test_variants_are_made_parts_at_second_0_with_their_levels_and_come_back(self) -> None:
        one, two, three = sound([[0, "c4", 3]]), sound([[0, "e4", 3]]), sound([[0, "g4", 3]])
        given = {"defs": [one, two, three], "db": [-3, -6, -9], "max_dc": 0.01}
        parts = changed(self, as_parts, given)
        self.assertEqual(parts, {"parts": [[0, {"def": one, "db": -3}], [0, {"def": two, "db": -6}],
                                           [0, {"def": three, "db": -9}]], "max_dc": 0.01})
        self.assertEqual((as_variants(parts), as_variants(given)), (given, given))
        # One level for all goes to every part and comes back as one, and no level stays none.
        for level in ({}, {"db": -4}):
            with self.subTest(level=level):
                shared = {"defs": [one, two], **level}
                self.assertEqual(as_parts(shared), {"parts": [[0, {"def": one, **level}], [0, {"def": two, **level}]]})
                self.assertEqual(as_variants(as_parts(shared)), shared)
        # A part with no level, among parts that have one, is at 0: that is what none means.
        mixed = {"parts": [[0, {"def": one, "db": -3}], [0.5, {"def": two}]]}
        self.assertEqual(changed(self, as_variants, mixed), {"defs": [one, two], "db": [-3, 0]})
        self.assertEqual(as_parts(mixed), mixed)

    def test_the_first_sound_is_kept_with_its_own_level_where_the_new_kind_holds_one(self) -> None:
        one, two = sound([[0, "c4", 30]]), sound([[0, "e4", 3]])
        variants = {"defs": [one, two], "db": [-3, -6], "max_dc": 0.01}
        self.assertEqual(changed(self, as_one, variants), {"def": one, "db": -3, "max_dc": 0.01})
        # A part is an entry of its own: what it holds comes with it, ahead of what the whole entry holds.
        parts = {"parts": [[0.5, {"def": two, "db": -2, "max_dc": 0.5}], [0, {"def": one}]], "max_dc": 0.01,
                 "meta": {"kind": "ui"}}
        self.assertEqual(changed(self, as_one, parts), {"def": two, "db": -2, "max_dc": 0.5, "meta": {"kind": "ui"}})
        self.assertEqual(as_variants(parts),
                         {"defs": [two, one], "db": [-2, 0], "max_dc": 0.01, "meta": {"kind": "ui"}})
        # A held sound takes the first for its note and the second for its end.
        kept = changed(self, as_held, variants, RATE)
        self.assertEqual((kept["held"]["note"], kept["held"]["end"], kept["db"], kept["max_dc"]), (one, two, -3, 0.01))
        self.assertEqual(lengths(kept), (3, 15, 1.5, 4))

    def test_a_held_sound_is_made_its_note_and_its_end_and_the_one_of_the_readme_comes_back(self) -> None:
        given = held()
        note, end = given["held"]["note"], given["held"]["end"]
        self.assertEqual(changed(self, as_variants, given), {"defs": [note, end], "db": -6})
        # The end starts where the loop ended, once round: 36 ticks are 0.6 seconds.
        self.assertEqual(changed(self, as_parts, given),
                         {"parts": [[0, {"def": note, "db": -6}], [0.6, {"def": end, "db": -6}]]})
        self.assertEqual(changed(self, as_one, given), {"def": note, "db": -6})
        # Its lengths are the shares a new held sound is given, so nothing of it is lost on the way.
        self.assertEqual((as_held(as_variants(given), RATE), as_held(as_parts(given), RATE), as_held(given)),
                         (given, given, given))

    def test_a_sound_is_held_from_where_its_first_note_begins(self) -> None:
        # The notes are held from tick 10 to tick 50. The 10 ticks before them are in the start.
        late = changed(self, as_held, {"def": sound([[10, "c4", 20]], [[20, "e4", 30]])}, RATE)
        self.assertEqual(lengths(late), (14, 20, 2, 4))
        # A join that would be under two samples is none, since the engine fades over two samples or none.
        short = changed(self, as_held, {"def": sound([[0, "c4", 0.2]])}, RATE)
        self.assertEqual(lengths(short), (0.02, 0.1, 0, 4))

    def test_a_sound_with_no_note_held_long_enough_is_not_made_a_held_one(self) -> None:
        needs = "a held sound needs a note that is held long enough for a start and a loop, and "
        self.assertEqual(refusal(self, as_held, {"def": sound([])}, RATE), needs + "this sound has no note")
        self.assertEqual(refusal(self, as_held, {"defs": [sound([[4, "c4", 0]]), sound([[0, "c4", 9]])]}, RATE),
                         needs + "the notes of this sound are held for 0 ticks")
        # A loop of half the 0.008 ticks would be under half a sample at 6000 Hz, and is three samples at 44100.
        tiny = {"def": sound([[0, "c4", 0.008]])}
        self.assertEqual(refusal(self, as_held, tiny, RATE), needs + "the notes of this sound are held for 0.008 ticks")
        self.assertEqual(as_held(tiny)["held"]["loop"], 0.004)

    def test_silence_is_what_the_rules_cannot_see_and_the_engine_refuses_of_a_held_sound(self) -> None:
        # The engine levels a held sound by its loudest sample. One sound of silence renders, and held it does not.
        silent = {"def": sound([[0, "c4", 4, 0]])}
        self.assertEqual(float(np.abs(buffers(silent)[0]).max()), 0.0)
        self.assertEqual(refusal(self, buffers, as_held(silent, RATE)),
                         "held: the start, the loop and the end are all silent, and there is no loudest sample to "
                         "level them by")


class Variants(unittest.TestCase):
    def setUp(self) -> None:
        self.one, self.two, self.three = sound([[0, "c4", 3]]), sound([[0, "e4", 3]]), sound([[0, "g4", 3]])

    def test_a_variant_is_added_as_a_copy_dropped_and_moved_each_with_its_level(self) -> None:
        one, two, three = self.one, self.two, self.three
        given = {"defs": [one, two, three], "db": [-3, -6, -9]}
        added = changed(self, variant_added, given)
        self.assertEqual(added, {"defs": [one, two, three, three], "db": [-3, -6, -9, -9]})
        self.assertIsNot(added["defs"][3], added["defs"][2])
        self.assertEqual(changed(self, variant_added, given, 0),
                         {"defs": [one, two, three, one], "db": [-3, -6, -9, -3]})
        self.assertEqual(changed(self, variant_dropped, given, 1), {"defs": [one, three], "db": [-3, -9]})
        self.assertEqual(changed(self, variant_moved, given, 0, 2), {"defs": [two, three, one], "db": [-6, -9, -3]})
        self.assertEqual(variant_moved(given, 2, 0), {"defs": [three, one, two], "db": [-9, -3, -6]})
        # One level for all stays one.
        shared = {"defs": [one, two], "db": -4}
        self.assertEqual([variant_added(shared), variant_dropped(shared, 0), variant_moved(shared, 1, 0)],
                         [{"defs": [one, two, two], "db": -4}, {"defs": [two], "db": -4},
                          {"defs": [two, one], "db": -4}])

    def test_the_last_variant_stays_and_only_an_entry_of_variants_has_them(self) -> None:
        self.assertEqual(refusal(self, variant_dropped, {"defs": [self.one]}, 0),
                         "the last variant is not dropped: an entry of variants holds one at least")
        for call, more, wrong in ((variant_added, (2,), 2), (variant_dropped, (-1,), -1), (variant_moved, (2, 0), 2),
                                  (variant_moved, (0, True), True)):
            with self.subTest(call=call.__name__, more=more):
                self.assertEqual(refusal(self, call, {"defs": [self.one, self.two]}, *more),
                                 f"{wrong!r} is no variant, of which there are 2, counted from 0")
                self.assertEqual(refusal(self, call, {"def": self.one}, *more),
                                 "only an entry with 'defs' has variants, not one with 'def'")


class Parts(unittest.TestCase):
    def setUp(self) -> None:
        self.one, self.two = sound([[0, "c4", 3]]), sound([[0, "e4", 3]])
        self.given = {"parts": [[0, {"def": self.one, "db": -5}], [0.02, {"def": self.two}]]}

    def test_a_part_is_added_as_a_copy_dropped_set_at_another_second_and_at_another_level(self) -> None:
        one, two, given = self.one, self.two, self.given
        parts = given["parts"]
        self.assertEqual(changed(self, part_added, given, 0.5), {"parts": [*parts, [0.5, {"def": two}]]})
        self.assertEqual(changed(self, part_added, given, 0.25, 0), {"parts": [*parts, [0.25, {"def": one, "db": -5}]]})
        self.assertEqual(changed(self, part_dropped, given, 0), {"parts": [[0.02, {"def": two}]]})
        later = changed(self, part_at, given, 1, 0.3)
        self.assertEqual(later, {"parts": [parts[0], [0.3, {"def": two}]]})
        # 0.28 seconds later are 1680 samples more.
        self.assertEqual(len(buffers(later)[0]) - len(buffers(given)[0]), 1680)
        self.assertEqual(changed(self, part_level, given, 1, -9), {"parts": [parts[0], [0.02, {"def": two, "db": -9}]]})
        self.assertEqual(part_level(given, 0, None), {"parts": [[0, {"def": one}], parts[1]]})

    def test_the_last_part_stays_and_a_second_under_0_or_a_level_that_is_none_is_refused(self) -> None:
        given = self.given
        self.assertEqual(refusal(self, part_dropped, {"parts": given["parts"][:1]}, 0),
                         "the last part is not dropped: an entry of parts holds one at least")
        for call, more in ((part_added, (-1,)), (part_at, (0, -1))):
            self.assertEqual(refusal(self, call, given, *more),
                             "the second a part starts at is a number, 0 or more, not -1")
        self.assertEqual(refusal(self, part_level, given, 0, "loud"), "the level of a part is a number, not 'loud'")
        for call, more in ((part_added, (0, 2)), (part_dropped, (2,)), (part_at, (2, 0)), (part_level, (2, 0))):
            with self.subTest(call=call.__name__):
                self.assertEqual(refusal(self, call, given, *more),
                                 "2 is no part, of which there are 2, counted from 0")
                self.assertEqual(refusal(self, call, {"defs": [self.one]}, *more),
                                 "only an entry with 'parts' has parts, not one with 'defs'")


class Held(unittest.TestCase):
    def test_a_length_is_kept_inside_what_the_note_has_room_for_by_the_sums_of_the_engine(self) -> None:
        given = held()
        # The note is held for 60 ticks, fades for the one tick of `beep` and runs on for two: 6300 samples.
        self.assertEqual(len(raw(given["held"]["note"])), 6300)

        def kept(key: str, value: float, entry: dict = given) -> tuple:
            return lengths(changed(self, held_with, entry, key, value, INSTRUMENTS, RATE))

        self.assertEqual(kept("start", 10), (10, 30, 3, 4))
        # The start goes no further than where the loop still fits after it, and no lower than the join.
        self.assertEqual((kept("start", 99), kept("start", 1)), ((33, 30, 3, 4), (3, 30, 3, 4)))
        # The loop goes no further than the end of the note, and is no shorter than the join.
        self.assertEqual((kept("loop", 99), kept("loop", 0)), ((6, 57, 3, 4), (6, 3, 3, 4)))
        self.assertEqual(kept("loop", 0, held(join=0)), (6, 0.01, 0, 4))
        # The join is no longer than the start and no longer than the loop, and one of a single sample is none.
        self.assertEqual((kept("join", 99), kept("join", 99, held(loop=4)), kept("join", 0)),
                         ((6, 30, 6, 4), (6, 4, 4, 4), (6, 30, 0, 4)))
        self.assertEqual((kept("join", 0.01), kept("join", 0.02)), ((6, 30, 0, 4), (6, 30, 0.02, 4)))
        self.assertEqual(kept("let_go", 9.5), (6, 30, 3, 9.5))
        # The room is the engine's, with the fade of the instrument the note names: `soft` fades for three ticks.
        self.assertEqual(lengths(held_with(given, "loop", 99, {"beep": INSTRUMENTS["soft"]}, RATE)), (6, 59, 3, 4))
        # A name the instruments do not have is taken to fade for no ticks, the shortest its note can be.
        self.assertEqual(lengths(held_with(given, "loop", 99, None, RATE)), (6, 56, 3, 4))
        # At the rate the engine renders at by default a tick is 735 samples, and the sums come to the same ticks.
        self.assertEqual(lengths(held_with(given, "start", 99, INSTRUMENTS)), (33, 30, 3, 4))

    def test_what_is_no_length_of_a_held_sound_and_an_entry_the_engine_refuses_are_refused(self) -> None:
        self.assertEqual(refusal(self, held_with, {"def": sound([[0, "c4", 2]])}, "start", 1),
                         "only an entry with 'held' has a start, a loop, a join and a letting go, not one with 'def'")
        self.assertEqual(refusal(self, held_with, held(), "end", 1),
                         "'end' is no length of a held sound, which has 'start', 'loop', 'join', 'let_go'")
        self.assertEqual(refusal(self, held_with, held(), "loop", -1),
                         "the loop of a held sound is a number, 0 or more, not -1")
        # An entry that was wrong before stays refused, in the engine's words where they are the engine's.
        self.assertEqual(refusal(self, held_with, held(loop=90), "join", 2, INSTRUMENTS, RATE),
                         "the note of a held sound must be long enough for its start and its loop: it is 63 ticks "
                         "long, and they take 96")
        self.assertEqual(refusal(self, held_with, held(start=2, join=4), "loop", 10, INSTRUMENTS, RATE),
                         "held: join: must not be longer than the start: 400 samples against 200 at 6000 Hz")
        self.assertEqual(refusal(self, buffers, held(loop=90)),
                         "held: note: must be long enough for the start and the loop: it is 6300 samples at 6000 "
                         "Hz, and they take 9600")


class Sounds(unittest.TestCase):
    def test_a_sound_is_taken_out_of_an_entry_of_any_kind_and_put_back(self) -> None:
        one, two = sound([[0, "c4", 3]]), sound([[0, "e4", 3]], echo=ECHO)
        new = sound([[0, "g4", 40]], [[2, "b4", 4]], inst="soft")
        kept = held()
        for given, found in (({"def": one, "db": -3}, [one]), ({"defs": [one, two], "db": [-1, -2]}, [one, two]),
                             ({"parts": [[0, {"def": one}], [0.1, {"def": two, "db": -2}]]}, [one, two]),
                             (kept, [kept["held"]["note"], kept["held"]["end"]])):
            for index, defn in enumerate(found):
                with self.subTest(kind=kind_of(given), index=index):
                    self.assertEqual(sound_in(given, index), defn)
                    made = changed(self, with_sound, given, index, new, INSTRUMENTS, RATE)
                    self.assertEqual([sound_in(made, other) for other in range(len(found))],
                                     [*found[:index], new, *found[index + 1:]])
                    self.assertEqual(with_sound(made, index, defn, INSTRUMENTS, RATE), given)
            for call, more in ((sound_in, ()), (with_sound, (new,))):
                self.assertEqual(refusal(self, call, given, len(found), *more),
                                 f"{len(found)} is no sound of the entry, of which there are {len(found)}, "
                                 "counted from 0")

    def test_a_definition_that_the_engine_or_the_project_refuses_is_not_put_back(self) -> None:
        given = {"def": sound([[0, "c4", 3]])}
        for defn, said in ((7, "sound: must be a dict, not 7"),
                           (sound([[-1, "c4", 2]]), "voices[0].notes[0][0]: must be 0 or more, not -1"),
                           (sound([(0, "c4", 2)]), "entry.def.voices[0].notes[0]: must be a list")):
            with self.subTest(defn=defn):
                self.assertEqual(refusal(self, with_sound, given, 0, defn), said)

    def test_the_note_of_a_held_sound_stays_long_enough_for_its_start_and_its_loop(self) -> None:
        short = "the note of a held sound must be long enough for its start and its loop: it is {} ticks long, and " \
                "they take 36"
        self.assertEqual(refusal(self, with_sound, held(), 0, sound([[0, "c3", 30]]), INSTRUMENTS, RATE),
                         short.format(33))
        # 33 ticks, the one tick `beep` fades for and the two the buffer runs on are the 36 that are needed.
        fits = changed(self, with_sound, held(), 0, sound([[0, "c3", 33]]), INSTRUMENTS, RATE)
        self.assertEqual(len(buffers(fits)[0]) + len(buffers(fits)[1]), len(raw(sound([[0, "c3", 33]]))))
        # With the instruments not at hand the fade is taken to be none, and nothing passes that might not render.
        self.assertEqual(refusal(self, with_sound, held(), 0, sound([[0, "c3", 33]]), None, RATE), short.format(35))
        self.assertEqual(sound_in(with_sound(held(), 0, sound([[0, "c3", 34]]), None, RATE), 0), sound([[0, "c3", 34]]))
        # The end is any sound.
        self.assertEqual(sound_in(changed(self, with_sound, held(), 1, sound([]), INSTRUMENTS, RATE), 1), sound([]))


class Kept(unittest.TestCase):
    """What a sound has beside its notes: one more voice like another, the notes it is expected to play, the notes
    of its maker's own, and an instrument of its own made one of the project's."""

    ONE = {"def": sound([[0, "c4", 4]], [[2, "e4", 2, 0.5]]), "db": -3}

    def test_a_voice_is_copied_with_its_instrument_and_its_notes_as_the_last_voice(self) -> None:
        own = {"wave": "saw", "vol": [12, 8, 4]}
        defn = {"voices": [{"inst": own, "notes": [[0, "c4", 4], [2, 64, 2, 0.5]]}, {"inst": "beep", "notes": []}]}
        made = changed(self, voice_copied, defn, 0)
        self.assertEqual(made["voices"], [*defn["voices"], defn["voices"][0]])
        # The copy is its own: what is done to it is not done to the voice it was made of.
        made["voices"][2]["notes"].clear()
        made["voices"][2]["inst"].clear()
        self.assertEqual((made["voices"][0], defn["voices"][0]["inst"]), (defn["voices"][0], own))
        self.assertEqual(changed(self, voice_copied, defn, 1)["voices"][2], {"inst": "beep", "notes": []})
        for voice in (2, -1):
            self.assertEqual(refusal(self, voice_copied, defn, voice),
                             f"{voice} is no voice of the sound, of which there are 2, counted from 0")

    def test_a_note_a_sound_is_expected_to_play_is_added_last_set_and_dropped_and_the_last_takes_the_key(self) -> None:
        first = changed(self, mark_added, self.ONE, 0.05, "c4")
        self.assertEqual(first, {**self.ONE, "check": [{"at": 0.05, "note": "c4"}]})
        # A note is a name as it was written, or a number, which is written whole where it is whole.
        both = changed(self, mark_added, first, 0.2, 64.0)
        self.assertEqual((both["check"], type(both["check"][1]["note"])),
                         ([{"at": 0.05, "note": "c4"}, {"at": 0.2, "note": 64}], int))
        self.assertEqual(mark_added(first, 0.2, 60.5)["check"][1], {"at": 0.2, "note": 60.5})
        # Set, one has another second, or another note, and what is not named stays.
        self.assertEqual(changed(self, mark_set, both, 1, 0.3)["check"],
                         [{"at": 0.05, "note": "c4"}, {"at": 0.3, "note": 64}])
        self.assertEqual(changed(self, mark_set, both, 0, None, "d4")["check"],
                         [{"at": 0.05, "note": "d4"}, {"at": 0.2, "note": 64}])
        self.assertEqual(mark_set(both, 0, 0, 72)["check"][0], {"at": 0, "note": 72})
        # Dropped, it is gone alone, and an entry without its last one is as it was before the first.
        left = changed(self, mark_dropped, both, 0)
        self.assertEqual((left["check"], mark_dropped(left, 0)), ([{"at": 0.2, "note": 64}], self.ONE))
        # They are kept for any kind of entry, and each passes the check of the project.
        for entry in ({"defs": [sound([[0, "c4", 3]]), sound([[0, "e4", 3]])]}, held()):
            with self.subTest(kind=kind_of(entry)):
                self.assertEqual(changed(self, mark_added, entry, 0.1, "g3")["check"], [{"at": 0.1, "note": "g3"}])

    def test_a_second_under_0_what_is_no_note_and_a_note_that_is_not_there_are_refused(self) -> None:
        both = mark_added(mark_added(self.ONE, 0.05, "c4"), 0.2, 64)
        for call, given, said in (
                (mark_added, (self.ONE, -0.1, "c4"),
                 "the second a note is expected at is a number, 0 or more, not -0.1"),
                (mark_added, (self.ONE, "soon", "c4"),
                 "the second a note is expected at is a number, 0 or more, not 'soon'"),
                (mark_added, (self.ONE, 0.1, "h9"), "a note is a number or a name such as 'c#4', not 'h9'"),
                (mark_added, (self.ONE, 0.1, None), "a note is a number or a name such as 'c#4', not None"),
                (mark_set, (both, 0, -1), "the second a note is expected at is a number, 0 or more, not -1"),
                (mark_set, (both, 1, None, "x"), "a note is a number or a name such as 'c#4', not 'x'"),
                (mark_set, (both, 2, 0.3),
                 "2 is no note the sound is expected to play, of which there are 2, counted from 0"),
                (mark_dropped, (both, -1),
                 "-1 is no note the sound is expected to play, of which there are 2, counted from 0"),
                (mark_dropped, (self.ONE, 0),
                 "0 is no note the sound is expected to play, of which there are 0, counted from 0")):
            with self.subTest(call=call.__name__, given=given[1:]):
                self.assertEqual(refusal(self, call, *given), said)

    def test_a_text_is_kept_under_a_key_of_the_notes_of_a_sound_changed_and_dropped_with_the_last_key(self) -> None:
        first = changed(self, meta_set, self.ONE, "mood", "calm", True)
        self.assertEqual(first, {**self.ONE, "meta": {"mood": "calm"}})
        # A new key is the last, and one that is changed stays where it stands. A text can be empty.
        kept = {**self.ONE, "meta": {"by": "hand", "take": 3}}
        more = changed(self, meta_set, kept, "mood", "calm", True)
        self.assertEqual(list(more["meta"].items()), [("by", "hand"), ("take", 3), ("mood", "calm")])
        self.assertEqual(list(changed(self, meta_set, more, "by", "ear")["meta"].items()),
                         [("by", "ear"), ("take", 3), ("mood", "calm")])
        self.assertEqual(meta_set(kept, "by", "")["meta"], {"by": "", "take": 3})
        # Without `fresh` a key the entry does not have yet is set as well.
        self.assertEqual(meta_set(kept, "mood", "calm")["meta"], more["meta"])
        # Dropped, a key is gone alone: what is no text stays, and without its last key the entry has no notes.
        self.assertEqual(changed(self, meta_dropped, more, "by")["meta"], {"take": 3, "mood": "calm"})
        self.assertEqual(meta_dropped(first, "mood"), self.ONE)

    def test_a_key_that_is_none_one_taken_already_and_a_value_that_is_no_text_are_refused(self) -> None:
        kept = {**self.ONE, "meta": {"by": "hand", "take": 3}}
        for call, given, said in (
                (meta_set, (kept, "by", "ear", True), "NOTES has 'by' already"),
                (meta_set, (kept, "", "x"), "a key of NOTES is text with something in it, not ''"),
                (meta_set, (kept, 3, "x"), "a key of NOTES is text with something in it, not 3"),
                (meta_set, (kept, "mood", 3), "what stands under a key of NOTES here is text, not 3"),
                (meta_set, (kept, "take", "4"), "'take' of NOTES holds no text, and is left as the file has it"),
                (meta_dropped, (kept, "take"), "'take' of NOTES holds no text, and is left as the file has it"),
                (meta_dropped, (kept, "mood"), "NOTES has no 'mood'"),
                (meta_dropped, (self.ONE, "by"), "NOTES has no 'by'"),
                (meta_dropped, (kept, ""), "a key of NOTES is text with something in it, not ''")):
            with self.subTest(call=call.__name__, given=given[1:]):
                self.assertEqual(refusal(self, call, *given), said)

    def test_an_instrument_written_in_place_is_made_shared_by_two_changes_the_instrument_first(self) -> None:
        own = {"wave": "saw", "release": 0}
        data = {"instruments": INSTRUMENTS, "sounds": {
            "a": {"def": sound([[0, "c4", 4]], [[0, "e4", 4]], inst=own)},
            "b": {"defs": [sound([[0, "c4", 4]]), sound([[0, "c4", 4]], inst=own)], "db": [0, -3]}}}
        before = json.dumps(data)
        # Only the one voice names the instrument: another that has one like it written in place keeps its own.
        self.assertEqual(shared(data, "a", 0, 1, "rasp"), [("instruments.rasp", own), ("sounds.a", {"def": {
            "voices": [{"inst": own, "notes": [[0, "c4", 4]]}, {"inst": "rasp", "notes": [[0, "e4", 4]]}]}})])
        # The sound is counted in its entry as `sound_in` counts, and the rest of the entry is as it was.
        places = shared(data, "b", 1, 0, "rasp")
        self.assertEqual(places, [("instruments.rasp", own), ("sounds.b", {
            "defs": [sound([[0, "c4", 4]]), sound([[0, "c4", 4]], inst="rasp")], "db": [0, -3]})])
        # What comes back shares nothing with the project. With the two changes made the project loads, and
        # hands the sound out with the instrument it had.
        scribble([value for _, value in shared(data, "a", 0, 0, "rasp")])
        self.assertEqual(json.dumps(data), before)
        made = {"instruments": {**INSTRUMENTS, "rasp": places[0][1]}, "sounds": {**data["sounds"], "b": places[1][1]}}
        self.assertEqual(Project.from_dict(made).sound("b")["defs"][1]["voices"][0]["inst"], own)
        # A project with no instruments yet gets its table with the one in it.
        bare = {"sounds": {"a": {"def": sound([[0, "c4", 4]], inst=own)}}}
        self.assertEqual(shared(bare, "a", 0, 0, "rasp"),
                         [("instruments", {"rasp": own}), ("sounds.a", {"def": sound([[0, "c4", 4]], inst="rasp")})])

    def test_a_voice_that_names_its_instrument_and_a_name_that_cannot_be_given_are_refused(self) -> None:
        own = {"wave": "saw", "release": 0}
        data = {"instruments": INSTRUMENTS, "sounds": {
            "a": {"def": sound([[0, "c4", 4]], inst=own)}, "b": {"def": sound([[0, "c4", 4]])}}}
        for given, said in (
                (("b", 0, 0, "rasp"),
                 "voice 1 names its instrument, 'beep': only one written in place is made shared"),
                (("c", 0, 0, "rasp"), "the project has no sound called 'c'"),
                (("a", 1, 0, "rasp"), "1 is no sound of the entry, of which there are 1, counted from 0"),
                (("a", 0, 1, "rasp"), "1 is no voice of the sound, of which there are 1, counted from 0"),
                (("a", 0, 0, "beep"), "the project has an instrument called 'beep' already"),
                (("a", 0, 0, ""), "an instrument needs a name, and '' is none"),
                (("a", 0, 0, "ra.sp"), "an instrument cannot be called 'ra.sp': a name cannot hold '.'")):
            with self.subTest(given=given):
                self.assertEqual(refusal(self, shared, data, *given), said)


class Results(unittest.TestCase):
    def test_what_comes_back_shares_nothing_with_what_was_handed_in(self) -> None:
        own, echo = {"wave": "tri", "vol": [12, 6]}, dict(ECHO)
        defn = {"voices": [{"inst": own, "notes": [[0, "c4", 40], [2, 64, 4, 0.5]]},
                           {"inst": "beep", "notes": [[1, "g4", 2]]}], "echo": echo, "gain": 0.4}
        other = sound([[0, "e4", 3]])
        every = {(0, 0), (0, 1), (1, 0)}
        kinds = {"def": {"def": defn, "db": -3, "meta": {"tags": ["ui"]}},
                 "defs": {"defs": [defn, other], "db": [-1, -2]},
                 "parts": {"parts": [[0, {"def": defn, "db": -1, "meta": {"tags": ["ui"]}}], [0.1, {"def": other}]]},
                 "held": {**held(), "meta": {"tags": ["ui"]}}}
        made = {
            "placed": placed(defn, 0, 1, "c4", 2), "moved": moved(defn, every, 1, 1),
            "stretched": stretched(defn, every, 1), "removed": removed(defn, {(0, 0)}),
            "with_velocity": with_velocity(defn, every, 0.5), "copied": copied(defn, every),
            "pasted": pasted(defn, copied(defn, every), 3), "voice_added": voice_added(defn, own),
            "voice_dropped": voice_dropped(defn, 1), "voice_inst": voice_inst(defn, 1, own),
            "with_echo": with_echo(defn, echo), "gain": with_setting(defn, "gain", 0.2),
            "meta": with_setting(kinds["def"], "meta", kinds["def"]["meta"]),
            "variant_added": variant_added(kinds["defs"]), "variant_dropped": variant_dropped(kinds["defs"], 1),
            "variant_moved": variant_moved(kinds["defs"], 0, 1), "part_added": part_added(kinds["parts"], 1, 0),
            "part_dropped": part_dropped(kinds["parts"], 1), "part_at": part_at(kinds["parts"], 0, 1),
            "part_level": part_level(kinds["parts"], 0, -2), "held_with": held_with(kinds["held"], "start", 9),
            "let_go": held_with(kinds["held"], "let_go", 9), "sound_in": sound_in(kinds["parts"], 0),
            "with_sound": with_sound(kinds["defs"], 1, defn), "voice_copied": voice_copied(defn, 0),
            "mark_added": mark_added(kinds["def"], 0.1, "c4"), "meta_set": meta_set(kinds["defs"], "by", "hand"),
        }
        for kind, entry in kinds.items():
            for call in (as_one, as_variants, as_parts, as_held):
                made[f"{call.__name__} of {kind}"] = call(entry)
        given = [own, echo, defn, other, kinds]
        before = json.dumps(given)
        for label, new in made.items():
            with self.subTest(label=label):
                scribble(new)
                self.assertEqual(json.dumps(given), before)


def count(entry: dict) -> int:
    """How many sounds an entry holds: its variants, its parts, the note and the end of a held one, or one."""
    kind = kind_of(entry)
    return {"def": 1, "held": 2}.get(kind) or len(entry[kind])


def edits(entry: dict, instruments: dict) -> tuple[dict[str, dict], dict[str, dict]]:
    """Every function an entry allows, once, on the last of its sounds: the new entries by what was done.

    The second lot are the changes that are taken back again, or that change only how the sound is kept. They
    should leave the entry as it was, or sounding as it did.
    """
    at = count(entry) - 1
    kind, defn = kind_of(entry), sound_in(entry, at)
    every = boxed(defn, 0, 10000, LOWEST, HIGHEST)
    notes, voices = len(defn["voices"][0]["notes"]), len(defn["voices"])

    def put(new: dict) -> dict:
        return with_sound(entry, at, new, instruments, RATE)

    there, longer = moved(defn, every, 3, 2), stretched(defn, every, 2)
    more, wider, echoing = placed(defn, 0, 2, "d5", 3, 0.5), voice_added(defn, "tick"), with_echo(defn, ECHO)
    limited = with_setting(entry, "max_seconds", 9)
    made = {
        "placed": put(more), "moved": put(there), "stretched": put(longer), "removed": put(removed(defn, {(0, 0)})),
        "with_velocity": put(with_velocity(defn, every, 0.5)),
        "pasted": put(pasted(defn, copied(defn, every), 5)[0]),
        "voice_added": put(wider), "voice_dropped": put(voice_dropped(voice_added(defn, {"wave": "saw"}), 0)),
        "voice_inst by name": put(voice_inst(defn, 0, "tick")),
        "voice_inst in place": put(voice_inst(defn, 0, {"wave": "tri", "release": 1})),
        "echo on": put(echoing), "echo off": put(with_echo(defn, None)), "gain": put(with_setting(defn, "gain", 0.3)),
        "max_seconds": limited,
        "as_one": as_one(entry), "as_variants": as_variants(entry), "as_parts": as_parts(entry),
        "as_held": as_held(entry, RATE),
    }
    quiet = {
        "moved there and back": put(moved(there, every, -3, -2)),
        "stretched and back": put(stretched(longer, every, -2)),
        "placed and removed": put(removed(more, {(0, notes)})),
        "voice added and dropped": put(voice_dropped(wider, voices)),
        "echo changed and back": put(with_echo(echoing, defn.get("echo"))),
        "taken out and put back": put(defn),
        "limit set and removed": with_setting(limited, "max_seconds", None),
    }
    if kind != "parts":
        made["db"] = with_setting(entry, "db", -4)
        quiet["level set and back"] = with_setting(made["db"], "db", entry.get("db"))
    if kind == "def":
        quiet.update({"variants and back": as_one(as_variants(entry)), "parts and back": as_one(as_parts(entry)),
                      "held and back": as_one(as_held(entry, RATE))})
    elif kind == "defs":
        added = variant_added(entry, 0)
        made.update({"variant_added": added, "variant_dropped": variant_dropped(entry, 0),
                     "variant_moved": variant_moved(entry, 0, at)})
        quiet.update({"variant added and dropped": variant_dropped(added, at + 1),
                      "variant moved and back": variant_moved(made["variant_moved"], at, 0),
                      "parts and back": as_variants(as_parts(entry))})
    elif kind == "parts":
        added, later = part_added(entry, 0.1, 0), part_at(entry, at, 0.3)
        made.update({"part_added": added, "part_dropped": part_dropped(entry, 0), "part_at": later,
                     "part_level": part_level(entry, at, -9)})
        quiet.update({"part added and dropped": part_dropped(added, at + 1),
                      "part moved and back": part_at(later, at, entry["parts"][at][0]),
                      "part levelled and back": part_level(made["part_level"], at, entry["parts"][at][1].get("db"))})
    else:
        for key, value in (("start", 10), ("loop", 99), ("join", 1), ("let_go", 0)):
            made[f"held_with {key}"] = held_with(entry, key, value, instruments, RATE)
            quiet[f"{key} changed and back"] = held_with(made[f"held_with {key}"], key, entry["held"][key],
                                                         instruments, RATE)
        quiet.update({"variants and back": as_held(as_variants(entry), RATE),
                      "parts and back": as_held(as_parts(entry), RATE)})
    return made, quiet


class Examples(unittest.TestCase):
    def setUp(self) -> None:
        self.data = Project.load(EXAMPLE).to_dict()
        self.data["sounds"]["hum"] = HUM

    def render(self, name: str, entry: dict) -> list[np.ndarray]:
        """An entry rendered in the place of a sound of the example, once the project's check has passed it there."""
        project = Project.from_dict({**self.data, "sounds": {**self.data["sounds"], name: entry}})
        return render_entry(project.sound(name), RATE)

    def test_every_change_to_a_sound_of_the_example_passes_the_project_s_check_and_renders(self) -> None:
        kinds = set()
        for name, entry in self.data["sounds"].items():
            before = json.dumps(entry)
            made, quiet = edits(entry, self.data["instruments"])
            self.assertEqual(json.dumps(entry), before)
            kinds.add(kind_of(entry))
            for label, new in {**made, **quiet}.items():
                with self.subTest(name=name, label=label):
                    rendered = self.render(name, new)
                    self.assertEqual(len(rendered), {"defs": count(new), "held": 3}.get(kind_of(new), 1))
                    self.assertTrue(all(len(samples) and np.isfinite(samples).all() for samples in rendered))
        self.assertEqual(kinds, {"def", "defs", "parts", "held"})

    def test_a_change_that_is_taken_back_leaves_a_sound_of_the_example_sample_for_sample_as_it_was(self) -> None:
        for name, entry in self.data["sounds"].items():
            base = self.render(name, entry)
            for label, new in edits(entry, self.data["instruments"])[1].items():
                with self.subTest(name=name, label=label):
                    self.assertEqual(new, entry)
                    same(self, self.render(name, new), base)


def made_up(dice: random.Random) -> dict:
    """A random definition: one to three voices of up to five notes, many sounding at once, and an echo or none."""
    def note() -> list:
        velocity = dice.choice(VELOCITIES)
        new = [dice.randint(0, 40), dice.choice(NOTES), dice.choice(LENGTHS)]
        return new if velocity is None else [*new, velocity]

    voices = [{"inst": dice.choice(INSTS), "notes": [note() for _ in range(dice.randint(0, 5))]}
              for _ in range(dice.randint(1, 3))]
    return {"voices": voices, **dice.choice(({}, {}, {"echo": ECHO}))}


def edited(dice: random.Random, defn: dict) -> dict:
    """A definition after one random edit of a few of its notes, of its voices, of its echo or of its gain."""
    voices = len(defn["voices"])
    notes = [(voice, index) for voice in range(voices) for index in range(len(defn["voices"][voice]["notes"]))]
    picked = set(dice.sample(notes, min(len(notes), dice.randint(1, 3))))
    if dice.random() < 0.3:
        low = dice.randint(30, 90)
        picked = boxed(defn, dice.randint(0, 40), dice.randint(0, 60), low, low + 24)
    what = dice.choice(("placed", "placed", "moved", "moved", "stretched", "removed", "with_velocity", "pasted",
                        "voice_added", "voice_dropped", "voice_inst", "with_echo", "with_setting"))
    if what == "placed":
        return placed(defn, dice.randrange(voices), snapped(dice.uniform(0, 40), dice.choice((0.5, 1, 4))),
                      dice.choice(NOTES), dice.choice(LENGTHS), dice.choice(VELOCITIES))
    if what == "moved":
        return moved(defn, picked, dice.choice((-12, -3, 0.5, 4, 20)), dice.randint(-60, 60))
    if what == "stretched":
        return stretched(defn, picked, dice.choice((-6, -1, 0.5, 3, 10)), dice.choice((0, 1, 2)))
    if what == "removed":
        return removed(defn, picked)
    if what == "with_velocity":
        return with_velocity(defn, picked, dice.choice(VELOCITIES))
    if what == "pasted":
        return pasted(defn, copied(defn, picked), dice.randint(0, 30))[0]
    if what == "voice_added" or (what == "voice_dropped" and voices == 1):
        return voice_added(defn, dice.choice(INSTS))
    if what == "voice_dropped":
        return voice_dropped(defn, dice.randrange(voices))
    if what == "voice_inst":
        return voice_inst(defn, dice.randrange(voices), dice.choice(INSTS))
    if what == "with_echo":
        return with_echo(defn, dice.choice((None, ECHO, {"time": [0.02, 0.05]})))
    return with_setting(defn, "gain", dice.choice((None, 0.3, 0.9)))


class Proof(unittest.TestCase):
    def test_two_hundred_runs_of_ten_random_edits_each_leave_a_sound_that_loads_and_renders(self) -> None:
        # The first render of a process has the engine's loops to make ready, and that is not what is timed here.
        buffers(sound([[0, "c4", 1]]), QUICK)
        began = time.perf_counter()
        for seed in range(200):
            with self.subTest(seed=seed):
                dice = random.Random(seed)
                defn = made_up(dice)
                for _ in range(10):
                    before = json.dumps(defn)
                    made = edited(dice, defn)
                    self.assertEqual(json.dumps(defn), before)
                    defn = made
                samples = buffers(defn, QUICK)[0]
                self.assertTrue(len(samples) and np.isfinite(samples).all())
        self.assertLess(time.perf_counter() - began, 3.0)


if __name__ == "__main__":
    unittest.main()
