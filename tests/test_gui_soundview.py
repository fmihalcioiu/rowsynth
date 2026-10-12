"""Tests for rowsynth.gui.soundview."""
from __future__ import annotations

import copy
import math
import time
import unittest
from collections.abc import Callable
from pathlib import Path

import numpy as np

from rowsynth.bank import render_entry
from rowsynth.errors import RowSynthError
from rowsynth.gui.soundview import Bar, Shape, columns, shape, shapes, ticks_to_seconds
from rowsynth.project import Project
from rowsynth.sound import render_sound

# A tick is a hundred samples at 6000 Hz, so a time in ticks is as many sixtieths of a second to the sample.
RATE = 6000
# A square wave that is cut the moment it is let go.
SQUARE = {"release": 0}
# A triangle that fades for three ticks and a half.
SOFT = {"wave": "tri", "release": 3.5}
INSTRUMENTS = {"beep": {"vol": [15, 10, 5], "release": 1}, "soft": SOFT}
BAD_NOTE = "voices[0].notes[0][1]: not a note: 'h4'"

EXAMPLE = Path(__file__).resolve().parent / "data" / "arcade.json"
# The held sound the README adds to the sounds of the example.
HUM = {"held": {"note": {"voices": [{"inst": "bass", "notes": [[0, "c3", 60]]}]}, "start": 6, "loop": 30, "join": 3,
                "end": {"voices": [{"inst": "bass", "notes": [[0, "c3", 2]]}]}, "let_go": 4}, "db": -6}


def sound(*voices: list, inst: object = SQUARE, **more: object) -> dict:
    """A def with a voice for each list of notes, every voice on one instrument."""
    return {"voices": [{"inst": inst, "notes": notes} for notes in voices], **more}


class Refusals(unittest.TestCase):
    """What the tests below share: holding calls to what they are refused with."""

    def assert_refused(self, cases: list[tuple[str, Callable[[], object]]]) -> None:
        """Holds each call to the refusal it comes with."""
        for said, call in cases:
            with self.subTest(said=said):
                with self.assertRaises(RowSynthError) as caught:
                    call()
                self.assertEqual(str(caught.exception), said)


class ShapeTest(Refusals):
    def test_one_note_is_one_bar_from_its_tick_for_as_long_as_it_is_held(self) -> None:
        self.assertEqual(shape(sound([[6, "a4", 9]]), RATE).bars, (Bar(6 / 60, 9 / 60, 69.0, "", 0),))

    def test_two_voices_are_bars_in_order_of_start_each_with_the_voice_it_is_in(self) -> None:
        found = shape(sound([[4, "c4", 2], [0, "e4", 2]], [[2, "g4", 6]]), RATE)
        self.assertEqual(found.bars, (Bar(0.0, 2 / 60, 64.0, "", 0), Bar(2 / 60, 6 / 60, 67.0, "", 1),
                                      Bar(4 / 60, 2 / 60, 60.0, "", 0)))

    def test_a_rest_is_the_gap_between_one_bar_and_the_next(self) -> None:
        first, second = shape(sound([[0, "c4", 3], [8, "c4", 2]]), RATE).bars
        self.assertEqual((first.start + first.seconds, second.start), (3 / 60, 8 / 60))

    def test_a_note_is_the_same_bar_written_as_a_number_or_as_a_name(self) -> None:
        named = shape(sound([[0, "c#5", 2]]), RATE)
        self.assertEqual(named.bars[0].pitch, 73.0)
        self.assertEqual(shape(sound([[0, 73, 2]]), RATE), named)

    def test_a_bar_has_the_name_of_its_instrument_and_no_name_for_one_written_in_place(self) -> None:
        defn = {"voices": [{"inst": "beep", "notes": [[0, "c4", 2]]},
                           {"inst": {"wave": "saw"}, "notes": [[1, "c5", 2]]}]}
        before = copy.deepcopy((defn, INSTRUMENTS))
        self.assertEqual([(bar.inst, bar.voice) for bar in shape(defn, RATE, INSTRUMENTS).bars], [("beep", 0), ("", 1)])
        self.assertEqual((defn, INSTRUMENTS), before)

    def test_a_bar_starts_and_ends_where_the_rendered_wave_of_its_note_does(self) -> None:
        # At 8000 Hz a tick is 133 samples and a third, and the engine puts a time on the nearest sample.
        defn = sound([[5, "a4", 3]])
        (bar,) = shape(defn, 8000).bars
        self.assertEqual((bar.start, bar.seconds), (667 / 8000, 400 / 8000))
        wave = render_sound(defn, 8000)
        first = int(np.flatnonzero(wave)[0])
        # Once the note is cut, the high-pass of the output stage rings on faintly to the end of the buffer. So
        # the note ends where the wave is last over half its loudest sample.
        last = int(np.flatnonzero(np.abs(wave) > np.abs(wave).max() / 2)[-1])
        self.assertLess(abs(first / 8000 - bar.start), 1 / 60)
        self.assertLess(abs((last + 1) / 8000 - (bar.start + bar.seconds)), 1 / 60)

    def test_seconds_is_the_length_of_the_render_with_its_releases_its_tail_and_its_echo(self) -> None:
        notes = [[0, "c4", 2], [1.5, "e4", 4.25]]
        cases = [
            (sound([[6, "a4", 9]]), sound([[6, "a4", 9]])),
            (sound(notes, [[3, 40, 1]], inst="soft"), sound(notes, [[3, 40, 1]], inst=SOFT)),
            (sound(notes, echo={"time": 0.07}), sound(notes, echo={"time": 0.07})),
            (sound(notes, echo={"time": [0.03, 0.05], "feedback": 0.8}),
             sound(notes, echo={"time": [0.03, 0.05], "feedback": 0.8})),
            (sound(notes, echo={"time": 0}), sound(notes, echo={"time": 0})),
            (sound(), sound()),
        ]
        for i, (defn, full) in enumerate(cases):
            for rate in (RATE, 8000, 44100):
                with self.subTest(case=i, rate=rate):
                    self.assertEqual(shape(defn, rate, INSTRUMENTS).seconds, len(render_sound(full, rate)) / rate)

    def test_an_echo_gives_each_repeat_its_time_and_its_strength_while_that_is_over_a_hundredth(self) -> None:
        found = shape(sound([[0, "c4", 2]], echo={"time": 0.09, "feedback": 0.35, "mix": 0.3}), RATE).echo
        self.assertEqual([at for at, _ in found], [0.09, 0.18, 0.27, 0.36])
        np.testing.assert_allclose([strength for _, strength in found], [0.3, 0.105, 0.03675, 0.0128625])
        # Two delay lines share the mix, and their repeats come in order of time.
        found = shape(sound([[0, "c4", 2]], echo={"time": [0.08, 0.05], "feedback": 0.1, "mix": 0.4}), RATE).echo
        self.assertEqual([at for at, _ in found], [0.05, 0.08, 0.1, 0.16])
        np.testing.assert_allclose([strength for _, strength in found], [0.2, 0.2, 0.02, 0.02])
        # A strength has no sign and goes no higher than 1, and with no feedback there is one repeat.
        self.assertEqual(shape(sound([[0, "c4", 2]], echo={"time": 0.05, "feedback": 0, "mix": -3}), RATE).echo,
                         ((0.05, 1.0),))

    def test_a_sound_with_no_echo_or_with_an_echo_of_no_time_has_no_repeats(self) -> None:
        self.assertEqual(shape(sound([[0, "c4", 2]]), RATE).echo, ())
        self.assertEqual(shape(sound([[0, "c4", 2]], echo={"time": 0}), RATE).echo, ())

    def test_ticks_are_seconds_to_the_nearest_sample_of_the_rate(self) -> None:
        self.assertEqual([ticks_to_seconds(6), ticks_to_seconds(2.5, RATE), ticks_to_seconds(1, 8000)],
                         [0.1, 2.5 / 60, 133 / 8000])

    def test_what_is_wrong_with_a_definition_its_instruments_or_the_rate_is_refused(self) -> None:
        named = sound([], inst="bell")
        self.assert_refused([
            ("sound: must be a dict, not None", lambda: shape(None, RATE)),
            (BAD_NOTE, lambda: shape(sound([[0, "h4", 1]]), RATE)),
            ("voices[0].inst: no instrument named 'bell'", lambda: shape(named, RATE)),
            ("voices[0].inst: no instrument named 'bell'", lambda: shape(named, RATE, INSTRUMENTS)),
            ("instruments: must be a dict, not ['bell']", lambda: shape(named, RATE, ["bell"])),
            ("rate: must be a whole number, 1 or more, not 0", lambda: shape(sound(), 0)),
            ("tick: must be 0 or more, not -1", lambda: ticks_to_seconds(-1)),
        ])


class ShapesTest(Refusals):
    def setUp(self) -> None:
        self.project = Project.load(EXAMPLE)
        self.project.sounds["hum"] = HUM

    def of(self, name: str) -> list[tuple[str, float, Shape]]:
        """The shapes of a sound of the example, at the rate the engine renders at unless it is told another."""
        return shapes(self.project.sounds[name], instruments=self.project.instruments)

    def rendered(self, name: str) -> list[float]:
        """The seconds of each buffer the engine renders of a sound of the example."""
        return [len(buffer) / 44100 for buffer in render_entry(self.project.sound(name))]

    def test_one_sound_is_one_shape_with_no_label_at_the_start_of_the_clock(self) -> None:
        coin = (Bar(0.0, 4 / 60, 83.0, "chime", 0), Bar(4 / 60, 14 / 60, 88.0, "chime", 0))
        self.assertEqual(self.of("coin"), [("", 0.0, Shape(coin, 23 / 60, ()))])
        ((label, at, powerup),) = self.of("powerup")
        self.assertEqual((label, at, [bar.pitch for bar in powerup.bars]),
                         ("", 0.0, [72.0, 76.0, 79.0, 84.0, 88.0, 91.0]))
        self.assertEqual([seconds for seconds, _ in powerup.echo], [0.09, 0.18, 0.27, 0.36])
        self.assertEqual([powerup.seconds], self.rendered("powerup"))

    def test_variants_are_a_shape_each_numbered_from_one_and_all_at_the_start(self) -> None:
        found = self.of("blip")
        self.assertEqual([(label, at, [bar.pitch for bar in made.bars]) for label, at, made in found],
                         [("1", 0.0, [81.0]), ("2", 0.0, [84.0]), ("3", 0.0, [88.0])])
        self.assertEqual([made.seconds for _, _, made in found], self.rendered("blip"))

    def test_parts_are_a_shape_each_numbered_from_one_at_the_second_the_part_starts(self) -> None:
        found = self.of("explosion")
        self.assertEqual([(label, at, [(bar.start, bar.inst) for bar in made.bars]) for label, at, made in found],
                         [("1", 0.0, [(0.0, "thump")]), ("2", 0.02, [(0.0, "rumble")])])
        # The one buffer the engine adds the parts up into ends with the part that ends last.
        self.assertAlmostEqual(max(at + made.seconds for _, at, made in found), self.rendered("explosion")[0])
        # A part starts on the sample nearest to its seconds, as the engine puts it.
        ((_, at, _),) = shapes({"parts": [[0.0101, {"def": sound([[0, "c4", 1]])}]]}, RATE)
        self.assertEqual(at, 61 / RATE)

    def test_a_held_sound_is_its_start_its_loop_and_its_end_laid_end_to_end(self) -> None:
        def bass(ticks: int) -> tuple[Bar, ...]:
            return (Bar(0.0, ticks / 60, 48.0, "bass", 0),)

        found = self.of("hum")
        self.assertEqual(found, [("start", 0.0, Shape(bass(6), 6 / 60, ())),
                                 ("loop", 6 / 60, Shape(bass(30), 30 / 60, ())),
                                 ("end", 36 / 60, Shape(bass(2), 6 / 60, ()))])
        self.assertEqual([made.seconds for _, _, made in found], self.rendered("hum"))

    def test_a_zone_of_a_held_sound_has_what_the_note_holds_in_it_cut_at_its_borders(self) -> None:
        # The start is the note's ticks 0 to 4 and the loop its ticks 4 to 10. The last note lies past both.
        note = sound([[0, "c4", 2], [3, "e4", 2], [5, "g4", 0], [9, "b4", 4], [12, "d5", 2]], [[4, "c3", 6]],
                     echo={"time": 0.05, "feedback": 0})
        entry = {"held": {"note": note, "start": 4, "loop": 6, "join": 0, "end": sound([[0, "a4", 1]]), "let_go": 2}}
        (_, _, start), (_, at, loop), (_, after, end) = shapes(entry, RATE)
        self.assertEqual(start, Shape((Bar(0.0, 2 / 60, 60.0, "", 0), Bar(3 / 60, 1 / 60, 64.0, "", 0)), 4 / 60,
                                      ((0.05, 0.3),)))
        self.assertEqual(loop, Shape((Bar(0.0, 1 / 60, 64.0, "", 0), Bar(0.0, 6 / 60, 48.0, "", 1),
                                      Bar(1 / 60, 0.0, 67.0, "", 0), Bar(5 / 60, 1 / 60, 71.0, "", 0)), 6 / 60,
                                     ((0.05, 0.3),)))
        self.assertEqual((at, after, end), (4 / 60, 10 / 60, shape(sound([[0, "a4", 1]]), RATE)))

    def test_what_is_wrong_with_an_entry_is_refused_with_its_place_in_front(self) -> None:
        good, bad = sound([[0, "c4", 1]]), sound([[0, "h4", 1]])
        held = {"note": good, "start": 1, "loop": 1, "join": 0, "end": good, "let_go": 0}
        self.assert_refused([(said, lambda entry=entry: shapes(entry, RATE)) for said, entry in (
            ("entry: must be a dict, not []", []),
            (f"def: {BAD_NOTE}", {"def": bad}),
            ("defs: must be a list, not 5", {"defs": 5}),
            (f"defs[1]: {BAD_NOTE}", {"defs": [good, bad]}),
            ("parts: must be a list, not 5", {"parts": 5}),
            ("parts[1]: must be [seconds, entry], not [0]", {"parts": [[0, {"def": good}], [0]]}),
            (f"parts[0]: def: {BAD_NOTE}", {"parts": [[0, {"def": bad}]]}),
            ("held: loop: must be at least one sample long, and 0 ticks are none at 6000 Hz",
             {"held": {**held, "loop": 0}}),
            (f"held: note: {BAD_NOTE}", {"held": {**held, "note": bad}}),
            (f"held: end: {BAD_NOTE}", {"held": {**held, "end": bad}}),
            ("held: note: must be long enough for the start and the loop: it is 300 samples at 6000 Hz, and they "
             "take 400", {"held": {**held, "loop": 3}}),
        )])
        # A rate that is none is no fault of any place in the entry.
        self.assert_refused([("rate: must be a whole number, 1 or more, not 0",
                              lambda: shapes({"parts": [[0, {"def": good}]]}, 0))])


class ColumnsTest(Refusals):
    def test_each_column_of_a_ramp_has_the_lowest_and_the_highest_sample_of_its_stretch(self) -> None:
        found = columns(np.arange(12, dtype=np.float32), 4)
        self.assertEqual((found.dtype, found.shape), (np.float32, (4, 2)))
        np.testing.assert_array_equal(found, [[0, 2], [3, 5], [6, 8], [9, 11]])
        # Ten samples do not go into four columns evenly: the stretches are of 2, 3, 2 and 3.
        np.testing.assert_array_equal(columns(-np.arange(10, dtype=np.float32), 4),
                                      [[-1, 0], [-4, -2], [-6, -5], [-9, -7]])

    def test_with_fewer_samples_than_columns_an_empty_stretch_takes_the_sample_it_falls_on(self) -> None:
        found = columns(np.array([1, -2, 3], dtype=np.float32), 7)
        np.testing.assert_array_equal(found[:, 0], [1, 1, 1, -2, -2, 3, 3])
        np.testing.assert_array_equal(found[:, 1], found[:, 0])

    def test_start_and_stop_pick_the_samples_as_a_slice_does(self) -> None:
        ramp = np.arange(12, dtype=np.float32)
        np.testing.assert_array_equal(columns(ramp, 3, 2, 8), [[2, 3], [4, 5], [6, 7]])
        np.testing.assert_array_equal(columns(ramp, 2, 8), [[8, 9], [10, 11]])
        np.testing.assert_array_equal(columns(ramp, 2, stop=4), [[0, 1], [2, 3]])
        np.testing.assert_array_equal(columns(ramp, 2, 10, 99), [[10, 10], [11, 11]])

    def test_a_stretch_of_no_samples_is_columns_of_nothing_and_no_columns_are_an_empty_array(self) -> None:
        ramp = np.arange(12, dtype=np.float32) + 1
        for start, stop in ((5, 5), (8, 3), (12, None), (40, 50)):
            with self.subTest(start=start, stop=stop):
                np.testing.assert_array_equal(columns(ramp, 3, start, stop), np.zeros((3, 2)))
        np.testing.assert_array_equal(columns(np.zeros(0, dtype=np.float32), 3), np.zeros((3, 2)))
        self.assertEqual(columns(ramp, 0).shape, (0, 2))

    def test_what_is_no_wave_and_what_is_no_count_are_refused(self) -> None:
        wave = np.zeros(4, dtype=np.float32)
        self.assert_refused([
            ("samples: must be an array of one dimension", lambda: columns([0.0, 1.0], 2)),
            ("samples: must be an array of one dimension", lambda: columns(np.zeros((2, 4), dtype=np.float32), 2)),
            ("width: must be a whole number, 0 or more, not 2.5", lambda: columns(wave, 2.5)),
            ("start: must be a whole number, 0 or more, not -1", lambda: columns(wave, 2, -1)),
            ("stop: must be a whole number, 0 or more, not True", lambda: columns(wave, 2, 0, True)),
        ])

    def test_a_minute_of_noise_is_read_into_800_columns_in_under_a_twentieth_of_a_second(self) -> None:
        noise = np.random.default_rng(1).standard_normal(60 * 44100, dtype=np.float32)
        best = math.inf
        for _ in range(3):
            began = time.perf_counter()
            found = columns(noise, 800)
            best = min(best, time.perf_counter() - began)
        self.assertLess(best, 0.05)
        np.testing.assert_array_equal(found[0], [noise[:3307].min(), noise[:3307].max()])
        np.testing.assert_array_equal(found[-1], [noise[-3308:].min(), noise[-3308:].max()])


if __name__ == "__main__":
    unittest.main()
