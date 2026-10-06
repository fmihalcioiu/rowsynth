"""Tests for rowsynth.note."""
from __future__ import annotations

import math
import unittest

import numpy as np

from rowsynth.errors import RowSynthError
from rowsynth.note import DEFAULT_RATE, TICK_HZ, render_note, step
from rowsynth.notes import hz
from rowsynth.osc import tone

# A tick is 133 and a third samples at 8000 Hz: ticks start at samples 0, 133, 267, 400, 533, 667, 800.
RATE = 8000
HISS = {"wave": "noise"}


def play(inst: dict, note: float = 69.0, ticks: float = 4, velocity: float = 1.0, rate: int = RATE) -> np.ndarray:
    """One note in a silent buffer that ends with the note's release, in doubles."""
    length = math.floor((ticks + inst.get("release", 2)) * rate / TICK_HZ + 0.5)
    out = np.zeros(length, dtype=np.float32)
    render_note(out, 0, inst, note, ticks, velocity, rate)
    return out.astype(np.float64)


def tick_of(samples: np.ndarray, tick: int) -> np.ndarray:
    """The samples of one tick of a note that starts with the buffer."""
    return samples[math.floor(tick * RATE / TICK_HZ + 0.5):math.floor((tick + 1) * RATE / TICK_HZ + 0.5)]


def levels(inst: dict, ticks: float = 4, velocity: float = 1.0) -> list[float]:
    """The level a note has come to by the last sample of each tick. Noise is 1 or -1, so its size is the level."""
    out = np.abs(play({**inst, "wave": "noise"}, 69.0, ticks, velocity))
    return [float(tick_of(out, tick)[-1]) for tick in range(math.ceil(ticks + inst.get("release", 2)))]


def pitch(samples: np.ndarray) -> float:
    """The pitch of samples in hertz, from how often they climb through zero. It is right to within `fine`."""
    climbs = np.count_nonzero((samples[:-1] < 0) & (samples[1:] >= 0))
    return climbs * RATE / (len(samples) - 1)


def fine(samples: np.ndarray) -> float:
    """How far off `pitch` can be, in hertz: one climb in the length of the samples."""
    return RATE / (len(samples) - 1)


class ConstantsTest(unittest.TestCase):
    def test_sixty_ticks_a_second(self):
        self.assertEqual(TICK_HZ, 60)

    def test_default_rate_is_44100(self):
        self.assertEqual(DEFAULT_RATE, 44100)


class StepTest(unittest.TestCase):
    def test_number_is_itself_at_every_tick(self):
        for tick in (0, 1, 7, 1000):
            self.assertEqual(step(9, tick, 15), 9.0, tick)
        self.assertEqual(step(0, 3, 15), 0.0)
        self.assertEqual(step(-2.5, 3, 15), -2.5)

    def test_list_gives_a_value_a_tick_and_holds_its_last(self):
        self.assertEqual([step([15, 9, 4], tick, 0) for tick in range(6)], [15.0, 9.0, 4.0, 4.0, 4.0, 4.0])
        self.assertEqual(step((1, 2), 1, 0), 2.0)
        self.assertEqual(step([7], 500, 0), 7.0)

    def test_loop_goes_round_again_from_its_index(self):
        table = {"v": [1, 2, 3, 4], "loop": 1}
        self.assertEqual([step(table, tick, 0) for tick in range(11)],
                         [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 2.0])

    def test_loop_from_the_start_repeats_the_whole_list(self):
        table = {"v": [5, 6, 7], "loop": 0}
        self.assertEqual([step(table, tick, 0) for tick in range(8)], [5.0, 6.0, 7.0, 5.0, 6.0, 7.0, 5.0, 6.0])

    def test_loop_on_the_last_index_holds_it(self):
        table = {"v": [5, 6, 7], "loop": 2}
        self.assertEqual([step(table, tick, 0) for tick in range(6)], [5.0, 6.0, 7.0, 7.0, 7.0, 7.0])

    def test_loop_outside_the_list_holds_the_last_value(self):
        for loop in (3, 40, -1, -7):
            table = {"v": [5, 6, 7], "loop": loop}
            self.assertEqual([step(table, tick, 0) for tick in range(6)], [5.0, 6.0, 7.0, 7.0, 7.0, 7.0], loop)

    def test_loop_left_out_holds_the_last_value(self):
        self.assertEqual([step({"v": [5, 6]}, tick, 0) for tick in range(4)], [5.0, 6.0, 6.0, 6.0])

    def test_nothing_to_read_gives_the_fallback(self):
        for seq in (None, [], (), {"v": []}, {"v": [], "loop": 0}, {}):
            self.assertEqual(step(seq, 0, 15), 15.0, repr(seq))
            self.assertEqual(step(seq, 9, 2), 2.0, repr(seq))

    def test_gives_a_float(self):
        for seq in (3, [3], {"v": [3], "loop": 0}, None):
            self.assertIs(type(step(seq, 0, 15)), float, repr(seq))

    def test_refuses_what_is_not_a_step_table(self):
        for seq in ("15", True, [15, "9"], [None], {"v": 15}, {"v": [15], "loop": 0.5}, {"v": [15], "loop": True},
                    float("nan"), [float("inf")]):
            with self.assertRaises(RowSynthError, msg=repr(seq)):
                step(seq, 1, 0)


class LevelTest(unittest.TestCase):
    def test_level_comes_to_vol_over_15(self):
        for vol, level in ((15, 1.0), (7.5, 0.5), (6, 0.4), (0, 0.0)):
            self.assertAlmostEqual(levels({"vol": vol, "release": 0}, ticks=2)[-1], level, places=3, msg=vol)

    def test_vol_left_out_is_15(self):
        self.assertTrue(np.array_equal(play(HISS), play({"wave": "noise", "vol": 15})))

    def test_vol_0_is_silence(self):
        self.assertFalse(play({"vol": 0}).any())

    def test_vol_list_sets_the_level_of_each_tick_and_holds_its_last(self):
        got = levels({"vol": [15, 3, 9, 12], "release": 0}, ticks=6)
        np.testing.assert_allclose(got, [1.0, 0.2, 0.6, 0.8, 0.8, 0.8], rtol=0, atol=5e-4)

    def test_vol_loop_goes_round(self):
        got = levels({"vol": {"v": [15, 3, 9], "loop": 1}, "release": 0}, ticks=7)
        np.testing.assert_allclose(got, [1.0, 0.2, 0.6, 0.2, 0.6, 0.2, 0.6], rtol=0, atol=5e-4)

    def test_gain_and_velocity_multiply_the_level(self):
        self.assertAlmostEqual(levels({"vol": 12, "gain": 0.5, "release": 0}, ticks=2)[-1], 0.4, places=3)
        self.assertAlmostEqual(levels({"vol": 12, "release": 0}, ticks=2, velocity=0.25)[-1], 0.2, places=3)
        self.assertAlmostEqual(levels({"vol": 12, "gain": 3, "release": 0}, ticks=2, velocity=0.5)[-1], 1.2, places=3)

    def test_gain_and_velocity_scale_every_sample(self):
        inst = {"wave": "saw", "vol": [15, 4, 9]}
        plain = play(inst)
        np.testing.assert_allclose(play({**inst, "gain": 0.3}), 0.3 * plain, rtol=0, atol=1e-6)
        np.testing.assert_allclose(play(inst, velocity=0.7), 0.7 * plain, rtol=0, atol=1e-6)

    def test_level_starts_from_nothing_and_covers_a_share_of_the_way_each_sample(self):
        level = np.abs(play(HISS, ticks=1))
        share = 1 - 0.988 ** (44100 / RATE)
        self.assertAlmostEqual(float(level[0]), share, places=6)
        left = 1.0 - level[:40]
        np.testing.assert_allclose(left[1:] / left[:-1], 1 - share, rtol=0, atol=1e-5)

    def test_share_is_0_012_at_44100_hz(self):
        level = np.abs(play({"wave": "noise", "release": 0}, ticks=1, rate=44100))
        self.assertAlmostEqual(float(level[0]), 0.012, places=6)
        self.assertAlmostEqual(float(level[1]), 0.012 + 0.988 * 0.012, places=6)

    def test_level_takes_the_same_time_at_any_rate(self):
        def half_way(rate: int) -> float:
            level = np.abs(play({"wave": "noise", "release": 0}, ticks=1, rate=rate))
            return (int(np.argmax(level >= 0.5)) + 1) / rate

        self.assertAlmostEqual(half_way(8000), half_way(32000), delta=1 / 8000)
        self.assertTrue(0.001 < half_way(8000) < 0.002, half_way(8000))

    def test_level_moves_to_a_new_step_without_a_jump(self):
        level = np.abs(play({"wave": "noise", "vol": [15, 3], "release": 0}, ticks=2))
        self.assertLess(float(np.abs(np.diff(level)).max()), 0.07)


class ReleaseTest(unittest.TestCase):
    def test_release_steps_the_level_down_evenly(self):
        np.testing.assert_allclose(levels({"release": 3}, ticks=2), [1.0, 1.0, 0.75, 0.5, 0.25], rtol=0, atol=5e-4)

    def test_release_left_out_is_two_ticks(self):
        np.testing.assert_allclose(levels({}, ticks=1), [1.0, 2 / 3, 1 / 3], rtol=0, atol=5e-4)
        self.assertTrue(np.array_equal(play(HISS), play({"wave": "noise", "release": 2})))

    def test_release_fades_what_the_vol_table_says(self):
        np.testing.assert_allclose(levels({"vol": [15, 6], "release": 1}, ticks=2), [1.0, 0.4, 0.2], rtol=0, atol=5e-4)
        got = levels({"vol": [6, 15, 15], "release": 2}, ticks=1)
        np.testing.assert_allclose(got, [0.4, 2 / 3, 1 / 3], rtol=0, atol=5e-4)

    def test_long_release_fades_to_almost_nothing(self):
        got = levels({"release": 19}, ticks=1)
        self.assertEqual(len(got), 20)
        self.assertTrue(all(later < sooner for sooner, later in zip(got, got[1:])))
        self.assertAlmostEqual(got[-1], 0.05, places=3)

    def test_release_0_ends_the_note_at_its_level(self):
        np.testing.assert_allclose(levels({"release": 0}, ticks=2), [1.0, 1.0], rtol=0, atol=5e-4)

    def test_nothing_is_written_after_the_release(self):
        out = np.zeros(1000, dtype=np.float32)
        end = render_note(out, 0, {"wave": "saw", "release": 1}, 69.0, 2, 1.0, RATE)
        self.assertEqual(end, 400)
        self.assertNotEqual(float(out[399]), 0.0)
        self.assertFalse(out[400:].any())


class WaveTest(unittest.TestCase):
    def test_every_wave_is_its_plain_voice_under_the_level(self):
        shape = {"vol": [15, 5, 10], "release": 1}
        level = np.abs(play({**shape, "wave": "noise"}, ticks=3))
        for wave in ("pulse", "saw", "tri"):
            got = play({**shape, "wave": wave}, 60.5, ticks=3)
            plain = tone(wave, hz(60.5), len(got) / RATE, RATE)
            np.testing.assert_allclose(got, plain * level, rtol=0, atol=1e-6, err_msg=wave)

    def test_wave_left_out_is_the_pulse(self):
        self.assertTrue(np.array_equal(play({}), play({"wave": "pulse"})))
        self.assertFalse(np.array_equal(play({}), play({"wave": "saw"})))

    def test_pitch_is_the_notes(self):
        for wave in ("pulse", "saw", "tri"):
            for note in (45, 57, 69, 76.5, 81):
                # Half a second of it.
                got = play({"wave": wave, "release": 0}, note, ticks=30)
                self.assertAlmostEqual(pitch(got), hz(note), delta=fine(got), msg=(wave, note))

    def test_noise_is_clocked_4000_times_a_second_at_a4_and_twice_that_six_semitones_up(self):
        for note, clock in ((63, 2000.0), (69, 4000.0), (75, 8000.0), (81, 16000.0)):
            got = play(HISS, note, ticks=3)
            plain = tone("noise", clock, len(got) / RATE, RATE)
            self.assertTrue(np.array_equal(np.sign(got), plain), note)

    def test_short_picks_the_noise_that_repeats(self):
        got = play({"wave": "noise", "short": True}, 75, ticks=3)
        self.assertTrue(np.array_equal(np.sign(got), tone("noise", 8000.0, len(got) / RATE, RATE, short=True)))
        self.assertFalse(np.array_equal(got, play(HISS, 75, ticks=3)))
        self.assertTrue(np.array_equal(play({"wave": "noise", "short": False}, 75), play(HISS, 75)))

    def test_short_changes_no_other_wave(self):
        for wave in ("pulse", "saw", "tri"):
            self.assertTrue(np.array_equal(play({"wave": wave, "short": True}), play({"wave": wave})), wave)

    def test_gate_changes_nothing(self):
        for wave in ("pulse", "noise"):
            self.assertTrue(np.array_equal(play({"wave": wave, "gate": 0.5}), play({"wave": wave})), wave)


class BufferTest(unittest.TestCase):
    def test_adds_to_what_is_there(self):
        inst = {"wave": "tri", "vol": [15, 8]}
        alone = play(inst, ticks=2)
        out = np.full(len(alone), 0.25, dtype=np.float32)
        render_note(out, 0, inst, 69.0, 2, 1.0, RATE)
        np.testing.assert_allclose(out, 0.25 + alone, rtol=0, atol=1e-6)
        render_note(out, 0, inst, 69.0, 2, 1.0, RATE)
        np.testing.assert_allclose(out, 0.25 + 2 * alone, rtol=0, atol=1e-6)

    def test_at_moves_the_whole_note(self):
        inst = {"wave": "saw", "vol": [15, 8]}
        alone = play(inst, ticks=2)
        out = np.zeros(len(alone) + 150, dtype=np.float32)
        render_note(out, 100, inst, 69.0, 2, 1.0, RATE)
        self.assertFalse(out[:100].any())
        self.assertTrue(np.array_equal(out[100:100 + len(alone)], alone))
        self.assertFalse(out[100 + len(alone):].any())

    def test_returns_the_sample_after_the_release(self):
        out = np.zeros(2000, dtype=np.float32)
        self.assertEqual(render_note(out, 0, {}, 69.0, 4, 1.0, RATE), 800)
        self.assertEqual(render_note(out, 0, {"release": 0}, 69.0, 1, 1.0, RATE), 133)
        self.assertEqual(render_note(out, 0, {"release": 5}, 69.0, 3, 1.0, RATE), 1067)
        self.assertEqual(render_note(out, 250, {"release": 5}, 69.0, 3, 1.0, RATE), 1317)
        self.assertEqual(render_note(out, 0, {"release": 0}, 69.0, 0, 1.0, RATE), 0)
        self.assertIs(type(render_note(out, 0, {}, 69.0, 4, 1.0, RATE)), int)

    def test_index_rounds_a_half_up(self):
        # Six ticks at 11025 Hz are 1102.5 samples.
        out = np.zeros(1200, dtype=np.float32)
        self.assertEqual(render_note(out, 0, {"release": 0}, 69.0, 6, 1.0, 11025), 1103)
        self.assertFalse(out[1103:].any())
        self.assertNotEqual(float(out[1102]), 0.0)

    def test_part_of_a_tick_counts_in_the_index(self):
        out = np.zeros(1000, dtype=np.float32)
        self.assertEqual(render_note(out, 0, {"release": 0}, 69.0, 2.5, 1.0, RATE), 333)
        self.assertEqual(render_note(out, 0, {"release": 1.5}, 69.0, 1, 1.0, RATE), 333)

    def test_rate_and_velocity_left_out_are_44100_and_1(self):
        left_out = np.zeros(900, dtype=np.float32)
        given = np.zeros(900, dtype=np.float32)
        self.assertEqual(render_note(left_out, 0, {}, 69.0, 1), 2205)
        self.assertEqual(render_note(given, 0, {}, 69.0, 1, 1.0, 44100), 2205)
        self.assertTrue(np.array_equal(left_out, given))

    def test_buffer_too_short_keeps_the_start_of_the_note(self):
        inst = {"wave": "saw", "vol": [15, 8, 11]}
        whole = play(inst, ticks=4)
        for length in (300, 133, 1, 0):
            out = np.zeros(length, dtype=np.float32)
            self.assertEqual(render_note(out, 0, inst, 69.0, 4, 1.0, RATE), 800, length)
            self.assertTrue(np.array_equal(out, whole[:length]), length)

    def test_buffer_that_ends_before_the_note_is_left_alone(self):
        out = np.full(100, 0.5, dtype=np.float32)
        self.assertEqual(render_note(out, 500, {}, 69.0, 4, 1.0, RATE), 1300)
        self.assertTrue(np.array_equal(out, np.full(100, 0.5, dtype=np.float32)))


class KnownSamplesTest(unittest.TestCase):
    def check(self, out: np.ndarray, known: tuple) -> None:
        spots = [spot for spot, sample in known]
        np.testing.assert_allclose(out[spots], [sample for spot, sample in known], rtol=0, atol=1e-6)

    def test_saw_under_a_vol_list(self):
        out = np.zeros(560, dtype=np.float32)
        end = render_note(out, 5, {"wave": "saw", "vol": [15, 6, 11], "gain": 0.8}, 57.5, 2, 0.9, RATE)
        self.assertEqual(end, 538)
        self.check(out, ((4, 0.0), (5, 0.0), (6, -0.084649), (7, -0.115554), (8, -0.139696), (60, 0.079851),
                         (137, 0.340320), (138, 0.366355), (139, 0.390290), (300, -0.102669), (420, 0.116879),
                         (537, -0.155363), (538, 0.0)))

    def test_pulse_under_a_vol_loop_at_11025_hz(self):
        out = np.zeros(740, dtype=np.float32)
        end = render_note(out, 0, {"vol": {"v": [12, 15, 4], "loop": 1}, "release": 1}, 72.0, 3, 1.0, 11025)
        self.assertEqual(end, 735)
        self.check(out, ((0, 0.0), (1, 0.073651), (2, 0.107893), (3, 0.140521), (100, -0.793906), (183, -0.799889),
                         (184, -0.809323), (185, -0.818312), (400, -0.231067), (551, 0.277768), (552, 0.288245),
                         (700, 0.499833), (734, -0.499968), (735, 0.0)))

    def test_triangle_at_half_velocity(self):
        out = np.zeros(270, dtype=np.float32)
        end = render_note(out, 0, {"wave": "tri", "vol": 10, "release": 0}, 40.25, 2, 0.5, RATE)
        self.assertEqual(end, 267)
        self.check(out, ((0, 0.021461), (1, 0.041541), (2, 0.060328), (50, -0.322143), (133, -0.199973),
                         (200, 0.244444), (266, 0.022222), (267, 0.0)))

    def test_noise_through_a_release(self):
        out = np.zeros(540, dtype=np.float32)
        end = render_note(out, 2, {"wave": "noise", "release": 3}, 81.0, 1, 1.0, RATE)
        self.assertEqual(end, 535)
        self.check(out, ((1, 0.0), (2, 0.064384), (3, 0.124623), (4, 0.180983), (5, 0.233714), (6, 0.283051),
                         (130, 0.999813), (136, -0.968719), (300, 0.529725), (450, -0.259589), (534, -0.250036),
                         (535, 0.0)))

    def test_short_noise_above_full_level(self):
        out = np.zeros(540, dtype=np.float32)
        end = render_note(out, 0, {"wave": "noise", "short": True, "vol": [15, 8], "gain": 1.5}, 99.0, 2, 1.0, RATE)
        self.assertEqual(end, 533)
        self.check(out, ((0, 0.096576), (1, 0.186934), (2, -0.271474), (3, 0.350572), (4, 0.424576), (5, 0.493816),
                         (6, 0.558599), (200, -0.807579), (400, 0.516200), (532, 0.266705), (533, 0.0)))


class ErrorTest(unittest.TestCase):
    def refused(self, inst: object = None, out: object = None, at: object = 0, note: object = 69.0,
                ticks: object = 2, velocity: object = 1.0, rate: object = RATE) -> str:
        """What the error says when a note is refused."""
        if out is None:
            out = np.zeros(400, dtype=np.float32)
        with self.assertRaises(RowSynthError, msg=repr((inst, at, note, ticks, velocity, rate))) as caught:
            render_note(out, at, {} if inst is None else inst, note, ticks, velocity, rate)
        return str(caught.exception)

    def test_refuses_a_key_an_instrument_does_not_have(self):
        for key in ("volume", "echo", "Wave", "", 3):
            self.assertIn(repr(key), self.refused({"vol": 12, key: 1}))

    def test_refuses_what_is_not_an_instrument(self):
        for inst in ("pulse", ["pulse"], 3, True):
            self.assertIn("instrument", self.refused(inst))

    def test_refuses_a_wave_it_does_not_know(self):
        for wave in ("square", "fm", "PULSE", "", None, 0, ["pulse"]):
            said = self.refused({"wave": wave})
            self.assertIn("wave", said)
            self.assertIn(repr(wave), said)

    def test_refuses_a_vol_that_is_not_a_step_table(self):
        for vol in ("15", None, True, {"v": 15}, {"loop": 1}, {"v": None}, float("nan"), ("15",)):
            self.assertIn("vol", self.refused({"vol": vol}), repr(vol))

    def test_names_the_part_of_a_step_table_that_is_wrong(self):
        self.assertIn("vol[1]", self.refused({"vol": [15, "9", 4]}))
        self.assertIn("vol[2]", self.refused({"vol": [15, 9, None]}))
        self.assertIn("vol[0]", self.refused({"vol": [float("inf")]}))
        self.assertIn("vol.v[1]", self.refused({"vol": {"v": [15, True], "loop": 0}}))
        self.assertIn("vol.v", self.refused({"vol": {"v": "15", "loop": 0}}))
        self.assertIn("vol.loop", self.refused({"vol": {"v": [15, 9], "loop": 0.5}}))
        self.assertIn("vol.loop", self.refused({"vol": {"v": [15, 9], "loop": None}}))
        self.assertIn("'again'", self.refused({"vol": {"v": [15, 9], "again": 0}}))

    def test_refuses_a_release_that_is_not_a_length_of_time(self):
        for release in ("2", None, True, [2], -1, -0.5, float("nan"), float("inf")):
            self.assertIn("release", self.refused({"release": release}), repr(release))

    def test_refuses_a_gain_that_is_not_a_number(self):
        for gain in ("1", None, True, [1], float("nan")):
            self.assertIn("gain", self.refused({"gain": gain}), repr(gain))

    def test_refuses_a_short_that_is_not_true_or_false(self):
        for short in (1, 0, "yes", None, [True]):
            self.assertIn("short", self.refused({"wave": "noise", "short": short}), repr(short))

    def test_refuses_a_note_that_is_not_a_number(self):
        for note in ("a4", None, True, [69], float("nan")):
            self.assertIn("note", self.refused(note=note), repr(note))

    def test_refuses_ticks_that_are_not_a_length_of_time(self):
        for ticks in ("2", None, True, -1, -0.5, float("inf")):
            self.assertIn("ticks", self.refused(ticks=ticks), repr(ticks))

    def test_refuses_a_velocity_that_is_not_a_number(self):
        for velocity in ("1", None, True, float("nan")):
            self.assertIn("velocity", self.refused(velocity=velocity), repr(velocity))

    def test_refuses_a_rate_that_makes_no_sense(self):
        for rate in (0, -8000, 8000.5, "8000", None, True, float("inf")):
            self.assertIn("rate", self.refused(rate=rate), repr(rate))

    def test_refuses_a_start_that_is_not_a_sample_of_the_buffer_or_after_it(self):
        for at in (-1, 1.5, "0", None, True, float("nan")):
            self.assertIn("at", self.refused(at=at), repr(at))

    def test_refuses_a_buffer_that_is_not_float32_samples(self):
        for out in ([0.0] * 400, np.zeros(400), np.zeros(400, dtype=np.int16), np.zeros((2, 200), dtype=np.float32)):
            self.assertIn("out", self.refused(out=out))

    def test_refusing_leaves_the_buffer_as_it_was(self):
        out = np.full(400, 0.5, dtype=np.float32)
        self.refused({"vol": [15, 9, "4"]}, out=out)
        self.refused({"release": 1}, out=out, rate=0)
        self.assertTrue(np.array_equal(out, np.full(400, 0.5, dtype=np.float32)))
