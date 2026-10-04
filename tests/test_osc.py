"""Tests for rowsynth.osc."""
from __future__ import annotations

import unittest

import numpy as np

from rowsynth.errors import RowSynthError
from rowsynth.osc import DUTY, FM, NOISE, PULSE, SAW, TRI, WAVES, _k_blep, _k_pulse, new_state, tone

# 125 Hz at 8000 Hz moves the phase 1/64 a sample, which is exact in binary: a period is 64 samples to the bit.
RATE = 8000
FREQ = 125.0
PERIOD = 64


def one_period(wave: str, **more) -> np.ndarray:
    """The first period of a tone that fits the rate, in doubles."""
    return tone(wave, FREQ, PERIOD / RATE, RATE, **more).astype(np.float64)


def repeats_after(samples: np.ndarray) -> int:
    """The period in samples: the smallest shift, up to half the array, that lays it on itself. 0 if none does."""
    for shift in range(1, len(samples) // 2 + 1):
        if np.array_equal(samples[shift:], samples[:-shift]):
            return shift
    return 0


class TableTest(unittest.TestCase):
    def test_waves_have_their_numbers(self):
        self.assertEqual((PULSE, SAW, TRI, NOISE, FM), (0, 1, 2, 3, 4))
        self.assertEqual(WAVES, {"pulse": PULSE, "saw": SAW, "tri": TRI, "noise": NOISE, "fm": FM})

    def test_duty_shares(self):
        self.assertEqual(DUTY, (0.125, 0.25, 0.5, 0.75))


class StateTest(unittest.TestCase):
    def test_new_state_is_float64_with_the_phase_at_zero(self):
        state = new_state()
        self.assertEqual(state.dtype, np.float64)
        self.assertEqual(state[0], 0.0)

    def test_each_new_state_is_its_own(self):
        state = new_state()
        state[0] = 0.5
        self.assertEqual(new_state()[0], 0.0)


class BlepTest(unittest.TestCase):
    def test_nothing_away_from_the_jump(self):
        for t in (0.1, 0.25, 0.5, 0.75, 0.9):
            self.assertEqual(_k_blep(t, 0.1), 0.0, t)

    def test_a_whole_step_at_the_jump_fading_to_nothing_a_sample_away(self):
        self.assertEqual(_k_blep(0.0, 0.1), -1.0)
        self.assertEqual(_k_blep(0.05, 0.1), -0.25)
        self.assertAlmostEqual(_k_blep(0.0999999, 0.1), 0.0, places=6)
        self.assertAlmostEqual(_k_blep(0.9000001, 0.1), 0.0, places=6)
        self.assertAlmostEqual(_k_blep(0.95, 0.1), 0.25, places=12)
        self.assertAlmostEqual(_k_blep(0.9999999, 0.1), 1.0, places=5)


class PulseTest(unittest.TestCase):
    def test_repeats_every_period_and_no_sooner(self):
        for duty in range(4):
            self.assertEqual(repeats_after(tone("pulse", FREQ, 4 * PERIOD / RATE, RATE, duty=duty)), PERIOD, duty)

    def test_kernel_moves_the_phase_and_wraps_it(self):
        state = new_state()
        _k_pulse(state, FREQ, float(RATE), 0.5)
        self.assertEqual(state[0], 1.0 / PERIOD)
        for _ in range(PERIOD - 1):
            _k_pulse(state, FREQ, float(RATE), 0.5)
        self.assertEqual(state[0], 0.0)

    def test_stays_high_for_its_share_of_the_period(self):
        for duty, share in enumerate(DUTY):
            wave = one_period("pulse", duty=duty)
            top, bottom = 2.0 - 2.0 * share, -2.0 * share
            # One sample of each edge sits half way up; the rest are flat.
            self.assertEqual(int((wave == top).sum()), int(PERIOD * share) - 1, duty)
            self.assertEqual(int((wave == bottom).sum()), int(PERIOD * (1.0 - share)) - 1, duty)
            self.assertEqual(int((wave == top - 1.0).sum()), 2, duty)

    def test_range_is_two_wide_and_shifted_by_the_duty(self):
        for duty, share in enumerate(DUTY):
            for freq, rate in ((FREQ, RATE), (440.0, 11025)):
                wave = tone("pulse", freq, 0.05, rate, duty=duty)
                self.assertEqual(float(wave.max()), 2.0 - 2.0 * share, (duty, freq))
                self.assertEqual(float(wave.min()), -2.0 * share, (duty, freq))

    def test_smoothed_edges_never_pass_the_flat_levels(self):
        for duty, share in enumerate(DUTY):
            for freq in (1000.0, 3000.0, 5000.0):
                wave = tone("pulse", freq, 0.05, 11025, duty=duty)
                self.assertLessEqual(float(wave.max()), 2.0 - 2.0 * share, (duty, freq))
                self.assertGreaterEqual(float(wave.min()), -2.0 * share, (duty, freq))

    def test_no_offset_at_any_duty(self):
        for duty in range(4):
            whole = tone("pulse", FREQ, 5 * PERIOD / RATE, RATE, duty=duty).astype(np.float64)
            self.assertAlmostEqual(float(whole.mean()), 0.0, places=9, msg=duty)
            # 190 periods of a tone that does not fit the rate.
            loose = tone("pulse", 440.0, 4761 / 11025, 11025, duty=duty).astype(np.float64)
            self.assertAlmostEqual(float(loose.mean()), 0.0, places=3, msg=duty)

    def test_first_samples_of_a_3520_hz_pulse_at_each_duty(self):
        known = (
            (0.75, 1.561688, 0.070414, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.027602),
            (0.5, 1.5, 1.5, 0.746754, -0.482549, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.277602),
            (0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.458605, -0.930196, -1.0, -1.0, -1.0, -1.0, -0.777602),
            (-0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.135555, -1.342941, -1.5, -1.277602),
        )
        for duty, samples in enumerate(known):
            got = tone("pulse", 3520.0, 0.001, 44100, duty=duty)[:len(samples)]
            np.testing.assert_allclose(got, samples, rtol=0, atol=1e-6, err_msg=f"duty {duty}")


class SawTest(unittest.TestCase):
    def test_repeats_every_period_and_no_sooner(self):
        self.assertEqual(repeats_after(tone("saw", FREQ, 4 * PERIOD / RATE, RATE)), PERIOD)

    def test_climbs_in_a_straight_line_and_drops_through_the_middle(self):
        wave = one_period("saw")
        self.assertEqual(wave[0], 0.0)
        self.assertTrue(np.array_equal(wave[1:], np.arange(1, PERIOD) / (PERIOD / 2) - 1.0))

    def test_range_stays_inside_one_each_way(self):
        for freq in (20.0, 440.0, 1000.0, 3000.0, 5000.0):
            wave = tone("saw", freq, 0.2, 11025)
            self.assertLess(float(np.abs(wave).max()), 1.0, freq)
        low = tone("saw", 20.0, 0.2, 11025)
        self.assertGreater(float(low.max()), 0.99)
        self.assertLess(float(low.min()), -0.99)

    def test_no_offset(self):
        whole = tone("saw", FREQ, 5 * PERIOD / RATE, RATE).astype(np.float64)
        self.assertAlmostEqual(float(whole.mean()), 0.0, places=9)

    def test_first_samples_of_a_3520_hz_saw(self):
        known = (0.0, -0.840363, -0.680726, -0.521088, -0.361451, -0.201814, -0.042177, 0.11746,
                 0.277097, 0.436735, 0.596372, 0.756009, 0.693248, -0.6455, -0.765079, -0.605442)
        got = tone("saw", 3520.0, 0.001, 44100)[:len(known)]
        np.testing.assert_allclose(got, known, rtol=0, atol=1e-6)


class TriTest(unittest.TestCase):
    def test_repeats_every_period_and_no_sooner(self):
        self.assertEqual(repeats_after(tone("tri", FREQ, 4 * PERIOD / RATE, RATE)), PERIOD)

    def test_one_period_is_32_steps_down_then_up(self):
        # Twice the frequency: a step a sample.
        wave = tone("tri", 2 * FREQ, 32 / RATE, RATE)
        levels = np.array(list(range(15, -1, -1)) + list(range(16)))
        np.testing.assert_allclose(wave, levels / 7.5 - 1.0, rtol=0, atol=1e-6)

    def test_sixteen_levels_evenly_spaced(self):
        for freq in (FREQ, 440.0, 1234.5):
            levels = np.unique(tone("tri", freq, 0.1, RATE))
            self.assertEqual(len(levels), 16, freq)
            np.testing.assert_allclose(levels, np.linspace(-1.0, 1.0, 16), rtol=0, atol=1e-6, err_msg=str(freq))

    def test_range_is_one_each_way(self):
        for freq in (20.0, 440.0, 3000.0):
            wave = tone("tri", freq, 0.2, 11025)
            self.assertEqual(float(wave.max()), 1.0, freq)
            self.assertEqual(float(wave.min()), -1.0, freq)

    def test_every_level_lasts_as_long_as_any_other(self):
        levels, counts = np.unique(one_period("tri"), return_counts=True)
        self.assertEqual(list(counts), [4] * 16)
        self.assertAlmostEqual(float(one_period("tri").mean()), 0.0, places=9)

    def test_first_samples_of_a_3520_hz_triangle(self):
        known = (1.0, 0.733333, 0.333333, 0.066667, -0.333333, -0.6, -1.0, -0.866667,
                 -0.466667, -0.2, 0.2, 0.6, 0.866667, 0.866667, 0.6, 0.2)
        got = tone("tri", 3520.0, 0.001, 44100)[:len(known)]
        np.testing.assert_allclose(got, known, rtol=0, atol=1e-6)


class ToneTest(unittest.TestCase):
    def test_gives_float32_of_the_length_asked_for(self):
        wave = tone("pulse", 440.0, 0.01, RATE)
        self.assertEqual(wave.dtype, np.float32)
        self.assertEqual(len(wave), 80)

    def test_length_rounds_a_half_up(self):
        self.assertEqual(len(tone("pulse", 1.0, 0.5, 5)), 3)

    def test_no_length_gives_no_samples(self):
        self.assertEqual(len(tone("pulse", 440.0, 0, RATE)), 0)

    def test_same_samples_every_time(self):
        self.assertTrue(np.array_equal(tone("pulse", 440.0, 0.01, RATE), tone("pulse", 440.0, 0.01, RATE)))

    def test_only_the_share_of_the_rate_matters(self):
        self.assertTrue(np.array_equal(tone("pulse", FREQ, 0.02, RATE), tone("pulse", 2 * FREQ, 0.01, 2 * RATE)))

    def test_duty_left_out_is_the_square(self):
        self.assertTrue(np.array_equal(tone("pulse", 440.0, 0.01, RATE), tone("pulse", 440.0, 0.01, RATE, duty=2)))

    def test_duty_shapes_the_pulse_only(self):
        for wave in ("saw", "tri"):
            plain = tone(wave, 440.0, 0.01, RATE)
            self.assertTrue(np.array_equal(plain, tone(wave, 440.0, 0.01, RATE, duty=0)), wave)
            self.assertFalse(np.array_equal(plain, tone("pulse", 440.0, 0.01, RATE)), wave)

    def test_refuses_a_wave_it_cannot_play(self):
        for wave in ("square", "PULSE", "", "fm", None, 0):
            with self.assertRaises(RowSynthError, msg=repr(wave)):
                tone(wave, 440.0, 0.01, RATE)

    def test_refuses_a_duty_outside_0_to_3(self):
        for duty in (-1, 4, 1.5, "2", None, True):
            with self.assertRaises(RowSynthError, msg=repr(duty)):
                tone("pulse", 440.0, 0.01, RATE, duty=duty)

    def test_refuses_a_rate_that_makes_no_sense(self):
        for rate in (0, -8000, 8000.5, float("nan"), float("inf"), "8000", None, True):
            with self.assertRaises(RowSynthError, msg=repr(rate)):
                tone("pulse", 440.0, 0.01, rate)

    def test_refuses_a_length_that_makes_no_sense(self):
        for seconds in (-0.01, float("nan"), float("inf"), "1", None, True):
            with self.assertRaises(RowSynthError, msg=repr(seconds)):
                tone("pulse", 440.0, seconds, RATE)

    def test_refuses_a_frequency_that_makes_no_sense(self):
        for freq in (-440.0, float("nan"), float("inf"), "440", None, True):
            with self.assertRaises(RowSynthError, msg=repr(freq)):
                tone("pulse", freq, 0.01, RATE)

    def test_error_says_where_and_what_was_given(self):
        with self.assertRaises(RowSynthError) as caught:
            tone("square", 440.0, 0.01, RATE)
        self.assertIn("tone", str(caught.exception))
        self.assertIn("'square'", str(caught.exception))
        with self.assertRaises(RowSynthError) as caught:
            tone("pulse", 440.0, 0.01, RATE, duty=7)
        self.assertIn("duty", str(caught.exception))
        self.assertIn("7", str(caught.exception))
