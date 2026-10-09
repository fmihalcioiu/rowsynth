"""Tests for rowsynth.bank."""
from __future__ import annotations

import copy
import math
import unittest

import numpy as np

from rowsynth.bank import PEAK_DB, RENDER_GAIN, held_parts, normalise, render_bank, render_entry
from rowsynth.errors import RowSynthError
from rowsynth.sound import render_sound

# A tick is 133 and a third samples at 8000 Hz.
RATE = 8000
# The size of a sample at 3 decibels under full scale.
PEAK = 10 ** (-3 / 20)
# Samples that jump about between -0.25 and 0.44. The loudest of them is above 0.
WAVE = (((np.arange(300) * 29) % 83 - 30) / 120).astype(np.float32)
PING = {"vol": [15, 11, 7], "duty": 1, "release": 1}
BUZZ = {"wave": "saw", "vol": 13, "lp": [3000, 1800, 900], "res": 0.4, "release": 1}
HISS = {"wave": "noise", "vol": [15, 8], "release": 0}
BELL = {"wave": "fm", "alg": 5, "fb": 0.3, "key_scale": -2, "release": 2,
        "ops": [{"mul": 2, "level": 0.9, "env": {"a": 0.002, "d": 0.05, "s": 0.5, "r": 0.03}},
                {"level": 0.8, "env": {"d": 0.08, "s": 0.6, "r": 0.04}}]}
CHIRP = {"voices": [{"inst": PING, "notes": [[0, "e5", 2], [2, "a5", 2, 0.7]]}], "gain": 0.9}
CRASH = {"voices": [{"inst": BUZZ, "notes": [[0, "c3", 3]]}, {"inst": HISS, "notes": [[0, 84, 2, 0.6]]}],
         "echo": {"time": 0.011, "feedback": 0.4, "mix": 0.5}}
# At its own gain this one is bent by the knee of the output stage.
CHIME = {"voices": [{"inst": BELL, "notes": [[0, "g4", 3], [1, 74.5, 2, 0.8]]}], "gain": 4}
STEPS = [{"voices": [{"inst": PING, "notes": [[0, note, 2]]}]} for note in ("c5", "d5", "f5")]
QUIET = {"voices": []}
# Two sounds on one timeline. The second starts half way between samples 62 and 63, and is over before the first.
BOTH = {"parts": [[0, {"def": CRASH, "db": -6}], [0.0078125, {"def": CHIRP, "db": -2.5}]]}
# A note that sounds through all of its six ticks, and is 1200 samples with its release and its tail.
DRONE = {"voices": [{"inst": BUZZ, "notes": [[0, "a2", 6]]}]}
# A sound far quieter than the others when all are rendered at one gain.
TAP = {"voices": [{"inst": {**PING, "gain": 0.25}, "notes": [[0, "e4", 1]]}]}
# A held sound: 267 samples of start, 400 of loop with the last 133 a join, and an end louder than both.
HELD = {"held": {"note": DRONE, "start": 2, "loop": 3, "join": 1, "end": CHIRP, "let_go": 4}, "db": -4}


def held(**change: object) -> dict:
    """HELD with some of what it is made of changed."""
    return {**HELD, "held": {**HELD["held"], **change}}


def peak(samples: np.ndarray) -> float:
    """The size of the largest sample."""
    return float(np.abs(samples).max())


def same(one: list, other: list) -> bool:
    """True when two lists of buffers are as long as each other and hold the same samples."""
    return len(one) == len(other) and all(np.array_equal(a, b) for a, b in zip(one, other))


class NormaliseTest(unittest.TestCase):
    def test_loudest_sample_is_brought_to_the_peak_and_the_others_with_it(self):
        for peak_db in (-3.0, -12.5, 0, 6):
            for samples in (WAVE, -WAVE):
                out = normalise(samples.copy(), peak_db)
                factor = 10 ** (peak_db / 20) / peak(samples)
                self.assertAlmostEqual(peak(out), 10 ** (peak_db / 20), places=6, msg=peak_db)
                np.testing.assert_allclose(out, samples.astype(np.float64) * factor, rtol=0, atol=1e-6)

    def test_works_in_place_and_hands_back_its_buffer(self):
        samples = WAVE.copy()
        self.assertIs(normalise(samples), samples)
        self.assertFalse(np.array_equal(samples, WAVE))
        self.assertEqual(samples.dtype, np.float32)

    def test_silence_and_an_empty_buffer_are_left_as_they_are(self):
        for count in (200, 1, 0):
            samples = np.zeros(count, dtype=np.float32)
            self.assertIs(normalise(samples, 6), samples)
            self.assertEqual(len(samples), count)
            self.assertFalse(samples.any())

    def test_known_values_and_a_peak_left_out_is_3_decibels_under_full_scale(self):
        samples = np.array([0.1, -0.9, 0.3, 0.0, -0.45], dtype=np.float32)
        known = ((-1, (0.099028, -0.891251, 0.297084, 0.0, -0.445625)),
                 (6, (0.221696, -1.995262, 0.665087, 0.0, -0.997631)))
        for peak_db, values in known:
            np.testing.assert_allclose(normalise(samples.copy(), peak_db), values, rtol=0, atol=1e-6)
        left_out = normalise(samples.copy())
        np.testing.assert_allclose(left_out, (0.078661, -0.707946, 0.235982, 0.0, -0.353973), rtol=0, atol=1e-6)
        self.assertEqual(PEAK_DB, -3.0)
        self.assertTrue(np.array_equal(left_out, normalise(samples.copy(), -3.0)))

    def test_refuses_a_buffer_that_is_not_float32_samples_and_a_peak_that_is_not_a_number(self):
        wrong = [("samples", (samples, -3.0)) for samples in ([0.5] * 9, np.full(9, 0.5), np.zeros((3, 3), np.float32))]
        wrong += [("peak_db", (WAVE.copy(), peak_db)) for peak_db in ("-3", None, True, float("nan"))]
        for name, given in wrong:
            with self.assertRaises(RowSynthError, msg=repr(given)) as caught:
                normalise(*given)
            self.assertTrue(str(caught.exception).startswith(name + ": "), caught.exception)


class EntryTest(unittest.TestCase):
    def test_def_is_one_buffer_with_its_loudest_sample_at_the_peak(self):
        for defn in (CHIRP, CRASH, CHIME):
            buffers = render_entry({"def": defn}, RATE)
            self.assertEqual(len(buffers), 1)
            self.assertIs(type(buffers[0]), np.ndarray)
            self.assertEqual(buffers[0].dtype, np.float32)
            self.assertEqual(len(buffers[0]), len(render_sound(defn, RATE)))
            self.assertAlmostEqual(peak(buffers[0]), PEAK, places=6)

    def test_def_is_rendered_at_the_render_gain_whatever_its_own(self):
        self.assertEqual(RENDER_GAIN, 0.12)
        got = render_entry({"def": CHIME}, RATE)[0]
        self.assertTrue(np.array_equal(got, normalise(render_sound(CHIME, RATE, 0.12))))
        self.assertFalse(np.array_equal(got, normalise(render_sound(CHIME, RATE))))
        for gain in (0.01, 0.12, 40):
            self.assertTrue(np.array_equal(got, render_entry({"def": {**CHIME, "gain": gain}}, RATE)[0]), gain)

    def test_defs_are_one_buffer_a_variant(self):
        buffers = render_entry({"defs": STEPS}, RATE)
        self.assertEqual(len(buffers), 3)
        for defn, samples in zip(STEPS, buffers):
            self.assertTrue(np.array_equal(samples, render_entry({"def": defn}, RATE)[0]))
            self.assertAlmostEqual(peak(samples), PEAK, places=6)
        self.assertFalse(np.array_equal(buffers[0], buffers[1]))
        self.assertTrue(same(render_entry({"defs": tuple(STEPS)}, RATE), buffers))

    def test_db_sets_a_sound_against_the_peak_and_left_out_is_0(self):
        left_out = render_entry({"def": CRASH}, RATE)[0]
        self.assertTrue(np.array_equal(left_out, render_entry({"def": CRASH, "db": 0}, RATE)[0]))
        for db in (-6, -0.5, 2.5, -40):
            got = render_entry({"def": CRASH, "db": db}, RATE)[0]
            self.assertAlmostEqual(peak(got), 10 ** ((db - 3) / 20), places=6, msg=db)
            np.testing.assert_allclose(got, left_out.astype(np.float64) * 10 ** (db / 20), rtol=0, atol=1e-7)

    def test_db_is_one_level_for_every_variant_or_a_list_with_one_for_each(self):
        alone = [render_entry({"def": defn, "db": db}, RATE)[0] for defn, db in zip(STEPS, (-1, -3.5, 0))]
        self.assertTrue(same(render_entry({"defs": STEPS, "db": [-1, -3.5, 0]}, RATE), alone))
        self.assertTrue(same(render_entry({"defs": STEPS, "db": (-1, -3.5, 0)}, RATE), alone))
        self.assertTrue(same(render_entry({"defs": STEPS, "db": -2}, RATE),
                             render_entry({"defs": STEPS, "db": [-2, -2, -2]}, RATE)))
        self.assertTrue(same(render_entry({"def": CHIRP, "db": [-6]}, RATE),
                             render_entry({"def": CHIRP, "db": -6}, RATE)))

    def test_silent_sound_stays_silent_and_its_length_goes_by_the_rate(self):
        for rate, count in ((8000, 267), (11025, 368), (48000, 1600)):
            samples = render_entry({"def": QUIET, "db": -4}, rate)[0]
            self.assertEqual(len(samples), count, rate)
            self.assertFalse(samples.any(), rate)
        self.assertEqual(len(render_entry({"def": QUIET})[0]), 1470)

    def test_keys_that_are_not_read_change_nothing_and_the_entry_is_left_as_it_was(self):
        entry = {"defs": copy.deepcopy(STEPS), "db": [-1, -3.5, 0], "meta": {"tags": ["ui"]},
                 "check": [{"at": 0.01, "note": "c5"}], "max_seconds": 0.2, "max_dc": 0.01, "max_bright": 0.5}
        before = copy.deepcopy(entry)
        self.assertTrue(same(render_entry(entry, RATE), render_entry({"defs": STEPS, "db": [-1, -3.5, 0]}, RATE)))
        self.assertEqual(entry, before)


class PartsTest(unittest.TestCase):
    def test_parts_are_one_buffer_with_each_part_at_its_own_level_added_in_from_its_start(self):
        crash, chirp = (render_entry(part, RATE)[0] for seconds, part in BOTH["parts"])
        buffers = render_entry(BOTH, RATE)
        self.assertEqual(len(buffers), 1)
        self.assertIs(type(buffers[0]), np.ndarray)
        self.assertEqual(buffers[0].dtype, np.float32)
        added = crash.copy()
        added[63:63 + len(chirp)] += chirp
        self.assertTrue(np.array_equal(buffers[0], added))
        self.assertFalse(np.array_equal(buffers[0][63:], crash[63:]))
        self.assertTrue(same(render_entry({"parts": tuple(tuple(part) for part in BOTH["parts"])}, RATE), buffers))

    def test_one_part_from_the_start_is_its_sound_and_two_over_each_other_are_not_brought_back_to_the_peak(self):
        alone = render_entry({"def": CHIRP, "db": -6}, RATE)
        self.assertTrue(same(render_entry({"parts": [[0, {"def": CHIRP, "db": -6}]]}, RATE), alone))
        twice = render_entry({"parts": [[0, {"def": CHIRP}], [0, {"def": CHIRP}]]}, RATE)[0]
        self.assertTrue(np.array_equal(twice, render_entry({"def": CHIRP}, RATE)[0] * np.float32(2)))
        self.assertAlmostEqual(peak(twice), 2 * PEAK, places=6)

    def test_buffer_ends_with_the_part_that_ends_last_and_is_silent_where_no_part_plays(self):
        first, second = render_entry({"def": STEPS[0]}, RATE)[0], render_entry({"def": STEPS[1], "db": -4}, RATE)[0]
        parts = [[0.1, {"def": STEPS[0]}], [0, {"def": STEPS[1], "db": -4}]]
        out = render_entry({"parts": parts}, RATE)[0]
        self.assertEqual(len(out), 800 + 667)
        self.assertTrue(np.array_equal(out[:667], second))
        self.assertFalse(out[667:800].any())
        self.assertTrue(np.array_equal(out[800:], first))
        self.assertTrue(np.array_equal(render_entry({"parts": parts[::-1]}, RATE)[0], out))

    def test_part_starts_at_the_sample_nearest_to_its_seconds_and_half_way_goes_up(self):
        known = ((0, 8000, 0), (0.0123, 8000, 98), (0.01234, 8000, 99), (0.0078125, 8000, 63), (0.125, 44100, 5513),
                 (0.00390625, 48000, 188))
        for seconds, rate, start in known:
            out = render_entry({"parts": [[seconds, {"def": QUIET}]]}, rate)[0]
            self.assertEqual(len(out), start + len(render_sound(QUIET, rate)), (seconds, rate))


class HeldTest(unittest.TestCase):
    def test_start_and_loop_are_cut_from_one_render_of_the_note_and_the_end_is_rendered_on_its_own(self):
        note = render_sound(DRONE, RATE, RENDER_GAIN)
        parts = held_parts(HELD, RATE)
        self.assertIs(type(parts), tuple)
        for part in parts:
            self.assertIs(type(part), np.ndarray)
            self.assertEqual(part.dtype, np.float32)
        start, loop, end = parts
        self.assertTrue(np.array_equal(start, note[:267]))
        self.assertEqual(len(loop), 400)
        self.assertTrue(np.array_equal(loop[:267], note[267:534]))
        self.assertTrue(np.array_equal(end, render_sound(CHIRP, RATE, RENDER_GAIN)))
        self.assertTrue(same(held_parts({"held": HELD["held"]}, RATE), parts))
        self.assertTrue(same(held_parts(held(note={**DRONE, "gain": 3}, end={**CHIRP, "gain": 0.01}), RATE), parts))

    def test_last_of_the_loop_is_faded_into_what_came_before_the_loop_and_ends_on_the_sample_before_its_first(self):
        note = render_sound(DRONE, RATE, RENDER_GAIN)
        loop = held_parts(HELD, RATE)[1]
        angle = np.arange(133) * (math.pi / 2 / 132)
        faded = note[534:667].astype(np.float64) * np.cos(angle) + note[134:267].astype(np.float64) * np.sin(angle)
        np.testing.assert_allclose(loop[267:], faded, rtol=0, atol=1e-7)
        self.assertGreater(float(np.abs(loop[267:] - note[534:667]).max()), 0.01)
        self.assertEqual(loop[267], note[534])
        self.assertEqual(loop[399], note[266])
        self.assertNotEqual(note[666], note[266])

    def test_join_runs_from_two_samples_to_the_whole_of_the_loop_and_of_the_start(self):
        note = render_sound(DRONE, RATE, RENDER_GAIN)
        loop = held_parts(held(join=0.015), RATE)[1]
        self.assertTrue(np.array_equal(loop[:399], note[267:666]))
        self.assertEqual(loop[399], note[266])
        start, loop, end = held_parts(held(start=2, loop=2, join=2), RATE)
        self.assertEqual((len(start), len(loop)), (267, 267))
        self.assertEqual(loop[0], note[267])
        self.assertFalse(np.array_equal(loop[1:], note[268:534]))
        self.assertEqual(loop[266], note[266])

    def test_join_of_0_leaves_the_loop_as_it_was_cut_and_a_start_of_0_is_an_empty_buffer(self):
        note = render_sound(DRONE, RATE, RENDER_GAIN)
        self.assertTrue(np.array_equal(held_parts(held(join=0), RATE)[1], note[267:667]))
        start, loop, end = held_parts(held(start=0, join=0), RATE)
        self.assertEqual((len(start), start.dtype), (0, np.float32))
        self.assertTrue(np.array_equal(loop, note[:400]))
        buffers = render_entry(held(start=0, join=0), RATE)
        self.assertEqual([len(samples) for samples in buffers], [0, 400, 934])

    def test_ticks_are_taken_to_the_nearest_sample_and_half_way_goes_up(self):
        known = ((8000, 2, 3, 267, 400), (8000, 0.5, 2.5, 67, 333), (11025, 1.25, 0.01, 230, 2),
                 (44100, 0.5, 1.5, 368, 1103), (48000, 0.015625, 3.015625, 13, 2413))
        for rate, start, loop, first, second in known:
            parts = held_parts(held(start=start, loop=loop, join=0), rate)
            self.assertEqual((len(parts[0]), len(parts[1])), (first, second), (rate, start, loop))
        # Half a tick of join at 44100 Hz is 367 samples and a half. It fades 368: all of the loop's 735 but 367.
        note = render_sound(DRONE, 44100, RENDER_GAIN)
        loop = held_parts(held(start=1, loop=1, join=0.5), 44100)[1]
        self.assertTrue(np.array_equal(loop[:368], note[735:1103]))
        self.assertNotEqual(loop[368], note[1103])

    def test_three_buffers_are_levelled_by_one_factor_from_the_loudest_sample_of_them_all(self):
        # The loudest sample is in the end, and with a quiet end in the start. The last two have no db, and one over 0.
        known = ((HELD, 2), (held(end=TAP), 0), ({"held": HELD["held"]}, 2), ({**HELD, "db": 2.5}, 2))
        for entry, loudest in known:
            parts = held_parts(entry, RATE)
            buffers = render_entry(entry, RATE)
            self.assertEqual(len(buffers), 3)
            level = 10 ** ((-3 + entry.get("db", 0)) / 20)
            factor = level / max(peak(part) for part in parts)
            for part, samples in zip(parts, buffers):
                self.assertEqual(samples.dtype, np.float32)
                self.assertTrue(np.array_equal(samples, (part.astype(np.float64) * factor).astype(np.float32)))
            peaks = [peak(samples) for samples in buffers]
            self.assertAlmostEqual(peaks[loudest], level, places=6)
            self.assertEqual([size > 0.95 * level for size in peaks], [i == loudest for i in range(3)], peaks)


class KnownSamplesTest(unittest.TestCase):
    def check(self, out: np.ndarray, count: int, known: tuple) -> None:
        self.assertEqual(len(out), count)
        spots = [spot for spot, sample in known]
        np.testing.assert_allclose(out[spots], [sample for spot, sample in known], rtol=0, atol=1e-6)

    def test_pulse_voice_at_the_peak_and_two_voices_with_an_echo_6_decibels_under_it(self):
        self.check(render_entry({"def": CHIRP}, RATE)[0], 934,
                   ((0, 0.014906), (1, 0.086204), (2, 0.123317), (98, 0.707946), (133, -0.086795), (311, -0.208812),
                    (467, 0.128471), (933, 0.0)))
        self.check(render_entry({"def": CRASH, "db": -6}, RATE)[0], 1504,
                   ((0, 0.007742), (1, 0.002255), (2, -0.014430), (129, -0.354813), (214, 0.035471), (501, -0.026425),
                    (752, 0.003755), (1503, 0.0)))

    def test_fm_voice_over_the_peak_at_44100_and_48000_hz(self):
        self.check(render_entry({"def": CHIME, "db": 2.5})[0], 5145,
                   ((0, 0.0), (1, 0.000841), (2, 0.002549), (735, -0.345224), (1034, 0.944061), (1715, 0.759718),
                    (2572, -0.000883), (5144, 0.0)))
        self.check(render_entry({"def": CHIME, "db": 2.5}, 48000)[0], 5600,
                   ((0, 0.0), (1, 0.000709), (2, 0.002147), (800, -0.346454), (1125, 0.944061), (1866, 0.778294),
                    (2800, 0.000985), (5599, 0.0)))

    def test_variants_at_levels_of_their_own(self):
        known = (((0, 0.013130), (1, 0.075934), (95, 0.516966), (108, 0.630957), (222, -0.156483), (333, -0.046313)),
                 ((0, 0.009899), (1, 0.057249), (95, -0.007109), (110, 0.473151), (222, -0.099570), (333, -0.039101)),
                 ((0, 0.014885), (1, 0.086081), (95, -0.139521), (127, 0.707946), (222, -0.182928), (333, 0.166883)))
        buffers = render_entry({"defs": STEPS, "db": [-1, -3.5, 0]}, RATE)
        self.assertEqual(len(buffers), 3)
        for samples, spots in zip(buffers, known):
            self.check(samples, 667, spots)

    def test_two_parts_on_one_timeline_at_8000_and_48000_hz(self):
        self.check(render_entry(BOTH, RATE)[0], 1504,
                   ((0, 0.007742), (1, 0.002255), (62, -0.287504), (63, -0.130614), (64, -0.255230), (174, 0.740766),
                    (501, -0.113821), (1503, 0.0)))
        self.check(render_entry(BOTH, 48000)[0], 9024,
                   ((0, 0.001225), (1, 0.002323), (374, -0.111910), (375, -0.345819), (376, -0.343443), (670, 0.711590),
                    (3008, -0.111258), (9023, 0.0)))

    def test_held_sound_as_it_is_cut_and_as_it_is_levelled_at_8000_hz(self):
        start, loop, end = held_parts(HELD, RATE)
        self.check(start, 267, ((0, 0.0), (1, -0.007376), (133, 0.058374), (221, -0.144566), (266, 0.037340)))
        self.check(loop, 400, ((0, 0.038528), (29, -0.132364), (200, -0.006436), (267, -0.024714), (334, 0.004515),
                               (398, 0.034412), (399, 0.037340)))
        self.check(end, 934, ((0, 0.003772), (98, 0.179152), (311, -0.052842), (933, 0.0)))
        start, loop, end = render_entry(HELD, RATE)
        self.check(start, 267, ((0, 0.0), (1, -0.018390), (133, 0.145544), (221, -0.360450), (266, 0.093101)))
        self.check(loop, 400, ((0, 0.096063), (29, -0.330027), (200, -0.016047), (267, -0.061620), (334, 0.011257),
                               (398, 0.085801), (399, 0.093101)))
        self.check(end, 934, ((0, 0.009405), (98, 0.446684), (311, -0.131751), (933, 0.0)))

    def test_held_sound_at_48000_hz_and_one_whose_loudest_sample_is_in_its_start_at_44100(self):
        start, loop, end = render_entry(HELD, 48000)
        self.check(start, 1600, ((0, 0.0), (1, -0.000169), (446, -0.348865), (800, 0.144565), (1599, 0.092932)))
        self.check(loop, 2400, ((0, 0.093429), (176, -0.320689), (1200, -0.018994), (1600, -0.066889),
                                (2000, -0.001520), (2398, 0.091780), (2399, 0.092932)))
        self.check(end, 5600, ((0, 0.001595), (657, 0.446684), (1866, -0.128398), (5599, 0.0)))
        start, loop, end = render_entry(held(end=TAP))
        self.check(start, 1470, ((0, 0.0), (1, -0.000274), (410, -0.446684), (735, 0.185114), (1469, 0.118931)))
        self.check(loop, 2205, ((0, 0.119624), (162, -0.410573), (1102, -0.025169), (1470, -0.085650),
                                (1838, -0.000534), (2203, 0.117324), (2204, 0.118931)))
        self.check(end, 2940, ((0, 0.000556), (670, 0.147987), (980, -0.021434), (2939, 0.0)))


class ErrorTest(unittest.TestCase):
    def refused(self, entry: object, rate: object = RATE) -> str:
        """What the error says when an entry is refused."""
        with self.assertRaises(RowSynthError, msg=repr((entry, rate))) as caught:
            render_entry(entry, rate)
        return str(caught.exception)

    def at(self, place: str, said: str, about: object = None) -> None:
        """Holds that an error opens with the place of what is wrong."""
        self.assertTrue(said.startswith(place + ": "), (said, about))

    def test_refuses_what_is_not_an_entry_and_a_key_an_entry_does_not_have(self):
        for entry in (None, [], "def", 3, [{"def": CHIRP}]):
            self.at("entry", self.refused(entry), entry)
        for key in ("dB", "gain", "Def", "variants", "", 3):
            said = self.refused({"def": CHIRP, key: 1})
            self.at("entry", said, key)
            self.assertIn(repr(key), said)

    def test_refuses_an_entry_that_is_made_of_nothing_or_of_more_than_one_thing(self):
        for entry in ({}, {"db": -3}, {"meta": {}, "check": []}):
            said = self.refused(entry)
            self.at("entry", said, entry)
            self.assertIn("none", said)
        for kinds in (("def", "defs"), ("def", "parts"), ("defs", "held"), ("parts", "held"), ("def", "defs", "held")):
            said = self.refused({kind: [CHIRP] if kind == "defs" else CHIRP for kind in kinds})
            self.at("entry", said, kinds)
            for kind in kinds:
                self.assertIn(f"and has {kind!r}" if kind == kinds[0] else f"and {kind!r}", said)

    def test_refuses_held_that_is_not_a_dict_or_lacks_a_key_or_has_one_too_many(self):
        for value in (None, [], "held", 3, [HELD["held"]]):
            self.assertIn("must be a dict", self.refused({"held": value}))
            self.at("held", self.refused({"held": value}), value)
        for key in HELD["held"]:
            said = self.refused({"held": {name: value for name, value in HELD["held"].items() if name != key}})
            self.at("held", said, key)
            self.assertIn(f"missing key {key!r}", said)
        for key in ("letGo", "gain", "db", "Start"):
            said = self.refused(held(**{key: 1}))
            self.at("held", said, key)
            self.assertIn(f"unknown key {key!r}", said)

    def test_refuses_held_lengths_that_are_not_numbers_from_0_on(self):
        for key in ("start", "loop", "join", "let_go"):
            for value in ("2", None, True, [2], float("nan"), float("inf"), -0.001, -2):
                self.at(f"held: {key}", self.refused(held(**{key: value})), (key, value))
        self.assertEqual(len(render_entry(held(let_go=0), RATE)), 3)

    def test_refuses_a_loop_of_no_samples_and_a_join_of_one(self):
        # A sample is 0.0075 ticks at 8000 Hz.
        for loop in (0, 0.003):
            said = self.refused(held(loop=loop, join=0))
            self.at("held: loop", said, loop)
            self.assertIn(f"{loop!r} ticks are none at 8000 Hz", said)
        self.assertEqual(len(held_parts(held(loop=0.004, join=0), RATE)[1]), 1)
        for join in (0.0075, 0.004, 0.011):
            said = self.refused(held(join=join))
            self.at("held: join", said, join)
            self.assertIn(f"{join!r} ticks are one at 8000 Hz", said)
        # The same join is six samples at 48000 Hz.
        self.assertEqual(len(render_entry(held(join=0.0075), 48000)), 3)

    def test_refuses_a_join_longer_than_the_loop_or_than_the_start(self):
        wrong = (({"start": 3, "loop": 1, "join": 2}, "the loop: 267 samples against 133"),
                 ({"start": 1, "loop": 3, "join": 2}, "the start: 267 samples against 133"),
                 ({"start": 2, "loop": 2, "join": 2.0075}, "the loop: 268 samples against 267"),
                 ({"start": 0, "loop": 3, "join": 1}, "the start: 133 samples against 0"))
        for change, why in wrong:
            said = self.refused(held(**change))
            self.at("held: join", said, change)
            self.assertIn("must not be longer than " + why, said)

    def test_refuses_a_note_too_short_for_the_start_and_the_loop(self):
        # The note is 1200 samples at 8000 Hz. Four ticks and five are 533 and 667.
        self.assertEqual([len(part) for part in held_parts(held(start=4, loop=5), RATE)], [533, 667, 934])
        for change in ({"start": 4, "loop": 5.0075}, {"start": 5, "loop": 5}, {"start": 1e9, "loop": 1}):
            said = self.refused(held(**change))
            self.at("held: note", said, change)
            self.assertIn("it is 1200 samples", said)
        self.assertIn("they take 1201", self.refused(held(start=4, loop=5.0075)))

    def test_refuses_a_held_sound_whose_three_parts_are_all_silent(self):
        hush = held(note=QUIET, start=1, loop=1, join=0, end=QUIET)
        said = self.refused(hush)
        self.at("held", said)
        self.assertIn("silent", said)
        for part, count in zip(held_parts(hush, RATE), (133, 133, 267)):
            self.assertEqual(len(part), count)
            self.assertFalse(part.any())
        for change in ({"note": DRONE}, {"end": CHIRP}):
            self.assertEqual(len(render_entry({**hush, "held": {**hush["held"], **change}}, RATE)), 3)

    def test_refuses_a_db_beside_held_that_is_a_list_or_not_a_number(self):
        for db in ([-4], [-4, -4, -4], (-4, -3, -2), [], "-4", None, True, float("nan")):
            said = self.refused({**HELD, "db": db})
            self.at("db", said, db)
            self.assertIn("must be a number", said)

    def test_names_the_note_or_the_end_of_a_held_sound_that_is_wrong(self):
        bad = {"voices": [{"inst": PING, "notes": [[0, "h5", 2]]}]}
        for key in ("note", "end"):
            self.at(f"held: {key}: voices[0].notes[0][1]", self.refused(held(**{key: bad})), key)
            for defn in (None, 3, "drone", [DRONE]):
                self.at(f"held: {key}: sound", self.refused(held(**{key: defn})), (key, defn))

    def test_held_parts_refuses_what_render_entry_does_and_an_entry_of_another_kind(self):
        wrong = ((held(join=-1), RATE, "held: join: "), (held(start=5, loop=5), RATE, "held: note: "),
                 (held(end=None), RATE, "held: end: sound: "), ({"held": None}, RATE, "held: "),
                 (None, RATE, "entry: "), ({**HELD, "defs": STEPS}, RATE, "entry: "), (HELD, 8000.5, "rate: "),
                 (HELD, 0, "rate: "))
        for entry, rate, place in wrong:
            with self.assertRaises(RowSynthError, msg=repr((entry, rate))) as caught:
                held_parts(entry, rate)
            self.assertTrue(str(caught.exception).startswith(place), caught.exception)
        for kind, entry in (("def", {"def": CHIRP}), ("defs", {"defs": STEPS}), ("parts", BOTH)):
            with self.assertRaises(RowSynthError, msg=kind) as caught:
                held_parts(entry, RATE)
            self.at("entry", str(caught.exception), kind)
            self.assertIn(f"not one with {kind!r}", str(caught.exception))

    def test_refuses_defs_that_are_not_a_list(self):
        for defs in (None, {}, "defs", 3, CHIRP):
            self.at("defs", self.refused({"defs": defs}), defs)

    def test_refuses_parts_that_are_not_a_list_or_are_none_and_a_db_beside_them(self):
        for parts in (None, {}, "parts", 3, CHIRP):
            self.at("parts", self.refused({"parts": parts}), parts)
        for parts in ([], ()):
            said = self.refused({"parts": parts})
            self.at("parts", said, parts)
            self.assertIn("at least one", said)
        for db in (-3, 0, [0, 0], "-3"):
            said = self.refused({**BOTH, "db": db})
            self.at("db", said, db)
            self.assertIn("'parts'", said)

    def test_refuses_a_part_that_is_not_a_pair_and_seconds_that_are_not_a_time_from_0_on(self):
        for part in (None, 3, "at", {"def": CHIRP}, [], [0], [0, {"def": CHIRP}, -6]):
            said = self.refused({"parts": [[0, {"def": CHIRP}], part]})
            self.at("parts[1]", said, part)
            self.assertIn("[seconds, entry]", said)
        for seconds in ("0", None, True, [0], float("nan"), float("inf"), -0.001, -2):
            self.at("parts[1]: seconds", self.refused({"parts": [[0, {"def": CHIRP}], [seconds, {"def": CHIRP}]]}),
                    seconds)

    def test_refuses_a_part_with_defs_parts_or_held_and_says_that_a_part_is_one_sound(self):
        for kind, value in (("defs", STEPS), ("parts", [[0, {"def": CHIRP}]]), ("held", {"note": CHIRP})):
            said = self.refused({"parts": [[0, {"def": CHIRP}], [0.01, {kind: value, "db": -3}]]})
            self.at("parts[1]: entry", said, kind)
            self.assertIn("a part is one sound", said)
            self.assertIn(f"not {kind!r}", said)

    def test_names_the_part_whose_entry_is_wrong(self):
        bad = {"voices": [{"inst": PING, "notes": [[0, "h5", 2]]}]}
        wrong = (({"def": bad}, "def: voices[0].notes[0][1]"), ({"def": None}, "def: sound"),
                 ({"def": CHIRP, "db": "-6"}, "db"), ({"def": CHIRP, "db": [0, 0]}, "db"),
                 ({"def": CHIRP, "gain": 1}, "entry"), ({"db": -3}, "entry"), ({"def": CHIRP, "defs": STEPS}, "entry"),
                 (None, "entry"), ([{"def": CHIRP}], "entry"))
        for entry, place in wrong:
            said = self.refused({"parts": [[0, {"def": CHIRP}], [0.01, {"def": CRASH}], [0.02, entry]]})
            self.at(f"parts[2]: {place}", said, entry)

    def test_refuses_a_db_that_is_not_a_number_or_a_list_with_one_number_a_variant(self):
        for db in ("-6", None, True, {}, float("nan"), float("inf")):
            self.at("db", self.refused({"def": CHIRP, "db": db}), db)
        for db, count in (([-1, -2], 3), ([-1, -2, -3, -4], 3), ([], 3), ((-1,), 3), ([-1, -2], 1), ([], 1)):
            said = self.refused({"defs": STEPS, "db": db} if count == 3 else {"def": CHIRP, "db": db})
            self.at("db", said, db)
            self.assertIn(f"{count}, not {len(db)}", said)
        for db in ([-1, "-2", -3], [-1, None, -3], [-1, True, "-3"], [-1, [-2], -3]):
            self.at("db[1]", self.refused({"defs": STEPS, "db": db}), db)

    def test_names_the_def_that_is_wrong(self):
        bad = {"voices": [{"inst": PING, "notes": [[0, "h5", 2]]}]}
        self.at("def: voices[0].notes[0][1]", self.refused({"def": bad}))
        self.at("defs[1]: voices[0].notes[0][1]", self.refused({"defs": [CHIRP, bad, CRASH]}))
        self.at("defs[2]: gain", self.refused({"defs": [CHIRP, CRASH, {"voices": [], "gain": "loud"}]}))
        for defn in (None, 3, "chirp", [CHIRP]):
            self.at("def: sound", self.refused({"def": defn}), defn)
            self.at("defs[0]: sound", self.refused({"defs": [defn]}), defn)

    def test_refuses_a_rate_that_makes_no_sense(self):
        for rate in (0, -8000, 8000.5, "8000", None, True, float("inf")):
            self.at("rate", self.refused({"def": CHIRP}, rate), rate)


class BankTest(unittest.TestCase):
    def test_every_sound_has_its_buffers_its_meta_and_its_held_which_only_a_held_sound_has(self):
        tags = {"priority": 1, "tags": ["ui"]}
        sounds = {"chirp": {"def": CHIRP, "meta": tags}, "steps": {"defs": STEPS, "db": [-1, -3.5, 0]},
                  "crash": {"def": CRASH, "db": -6, "meta": {}}, "both": {**BOTH, "meta": {"tags": ["hit"]}},
                  "drone": {**HELD, "meta": {"tags": ["loop"]}}}
        bank = render_bank(sounds, RATE)
        self.assertEqual(list(bank), ["chirp", "steps", "crash", "both", "drone"])
        for name, entry in sounds.items():
            self.assertEqual(sorted(bank[name]), ["buffers", "held", "meta"], name)
            self.assertTrue(same(bank[name]["buffers"], render_entry(entry, RATE)), name)
            if name != "drone":
                self.assertIsNone(bank[name]["held"], name)
        self.assertEqual(len(bank["drone"]["buffers"]), 3)
        self.assertEqual(bank["drone"]["held"], {"let_go": 4 / 60})
        self.assertIs(bank["chirp"]["meta"], tags)
        self.assertEqual(tags, {"priority": 1, "tags": ["ui"]})
        self.assertEqual(bank["steps"]["meta"], {})
        self.assertEqual(bank["crash"]["meta"], {})
        self.assertEqual(bank["both"]["meta"], {"tags": ["hit"]})
        self.assertEqual(bank["drone"]["meta"], {"tags": ["loop"]})

    def test_held_says_in_seconds_how_long_the_sound_takes_to_be_let_go_whatever_the_rate(self):
        for let_go, seconds in ((4, 4 / 60), (0, 0.0), (1.5, 0.025), (90, 1.5)):
            for rate in (RATE, 11025):
                self.assertEqual(render_bank({"drone": held(let_go=let_go)}, rate)["drone"]["held"],
                                 {"let_go": seconds}, (let_go, rate))

    def test_rate_left_out_is_44100_and_no_sounds_are_an_empty_bank(self):
        self.assertEqual(len(render_bank({"quiet": {"def": QUIET}})["quiet"]["buffers"][0]), 1470)
        self.assertEqual(len(render_bank({"quiet": {"def": QUIET}}, 48000)["quiet"]["buffers"][0]), 1600)
        self.assertEqual(render_bank({}, RATE), {})

    def test_puts_the_name_of_the_sound_in_front_of_what_is_refused(self):
        bad = {"voices": [{"inst": PING, "notes": [[0, "h5", 2]]}]}
        wrong = (({"defs": [CHIRP, bad]}, "defs[1]: voices[0].notes[0][1]: "), ({"def": CHIRP, "db": [0, 0]}, "db: "),
                 ({"def": CHIRP, "defs": STEPS}, "entry: "), ({"held": {}}, "held: "), (None, "entry: "),
                 (held(join=4), "held: join: "), (held(end=bad), "held: end: voices[0].notes[0][1]: "),
                 ({"parts": [[0, {"def": CHIRP}], [0.01, {"def": bad}]]}, "parts[1]: def: voices[0].notes[0][1]: "))
        for entry, place in wrong:
            with self.assertRaises(RowSynthError, msg=repr(entry)) as caught:
                render_bank({"chirp": {"def": CHIRP}, "second one": entry}, RATE)
            self.assertTrue(str(caught.exception).startswith("second one: " + place), caught.exception)

    def test_refuses_sounds_that_are_not_a_dict_and_a_rate_that_makes_no_sense(self):
        for sounds in (None, [], "chirp", [{"def": CHIRP}], 3):
            with self.assertRaises(RowSynthError, msg=repr(sounds)) as caught:
                render_bank(sounds, RATE)
            self.assertTrue(str(caught.exception).startswith("sounds: "), caught.exception)
        for sounds in ({"chirp": {"def": CHIRP}}, {}):
            for rate in (0, -8000, 8000.5, "8000", None, True):
                with self.assertRaises(RowSynthError, msg=repr(rate)) as caught:
                    render_bank(sounds, rate)
                self.assertTrue(str(caught.exception).startswith("rate: "), caught.exception)
