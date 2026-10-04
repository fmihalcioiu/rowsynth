"""Tests for rowsynth.measure."""

from __future__ import annotations

import math
import unittest

import numpy as np

from rowsynth.errors import RowSynthError
from rowsynth.measure import measure


class MeasureTest(unittest.TestCase):
    def test_result_has_the_seven_figures(self):
        self.assertEqual(
            list(measure([0.5, -0.5])),
            ["seconds", "peak_db", "rms_db", "dc", "clipped", "head_step", "tail_step"],
        )

    def test_seconds_is_the_length_over_the_rate(self):
        self.assertEqual(measure(np.zeros(1000), 8000)["seconds"], 0.125)
        self.assertEqual(measure(np.zeros(4410))["seconds"], 0.1)

    def test_seconds_is_held_to_three_places(self):
        # 100 / 48000 is 0.0020833...
        self.assertEqual(measure(np.zeros(100), 48000)["seconds"], 0.002)
        # 500 / 8000 is 0.0625 exactly, and the half goes up.
        self.assertEqual(measure(np.zeros(500), 8000)["seconds"], 0.063)

    def test_peak_is_the_largest_sample_either_side_of_zero(self):
        # 20 * log10(0.5) is -6.0206.
        self.assertEqual(measure([0.125, -0.5, 0.25])["peak_db"], -6.0)
        self.assertEqual(measure([0.5, -1.0, 0.0])["peak_db"], 0.0)
        self.assertEqual(measure([0.0, 0.001])["peak_db"], -60.0)

    def test_loudness_is_the_root_of_the_mean_square(self):
        # Every square is 0.25, so the root is 0.5.
        self.assertEqual(measure([0.5, -0.5, 0.5, -0.5])["rms_db"], -6.0)
        # One square of 1 among four samples: a mean of 0.25, a root of 0.5.
        self.assertEqual(measure([1.0, 0.0, 0.0, 0.0])["rms_db"], -6.0)
        # 20 * log10(0.1) is -20.
        self.assertEqual(measure([0.1] * 50)["rms_db"], -20.0)

    def test_a_full_scale_sine_peaks_at_zero_and_is_three_decibels_under_in_loudness(self):
        # Over whole periods the squares of a sine average 0.5, and 10 * log10(0.5) is -3.0103.
        got = measure(np.sin(2 * np.pi * np.arange(256) / 64))
        self.assertEqual(got["peak_db"], 0.0)
        self.assertEqual(got["rms_db"], -3.0)
        self.assertEqual(got["dc"], 0.0)

    def test_dc_is_the_mean_sample(self):
        self.assertEqual(measure([0.5, 0.5, -0.25, 0.25])["dc"], 0.25)
        self.assertEqual(measure([0.25, -0.25])["dc"], 0.0)
        self.assertEqual(measure([-0.5, 0.25])["dc"], -0.125)

    def test_dc_is_held_to_four_places_with_a_half_going_away_from_zero(self):
        # A mean of 1/32 is 0.03125 exactly.
        self.assertEqual(measure([0.0625, 0.0])["dc"], 0.0313)
        self.assertEqual(measure([-0.0625, 0.0])["dc"], -0.0313)
        # 0.75 / 8192 is 0.0000916, which is nearer 0.0001 than nothing.
        self.assertEqual(measure([0.75] + [0.0] * 8191)["dc"], 0.0001)

    def test_a_dc_too_small_to_show_is_a_zero_without_a_minus_sign(self):
        got = measure([-0.25] + [0.0] * 8191)["dc"]
        self.assertEqual(got, 0.0)
        self.assertEqual(math.copysign(1.0, got), 1.0)

    def test_clipped_counts_samples_at_0_999_or_beyond(self):
        self.assertEqual(measure([0.999, -0.999, 1.0, -1.25, 0.998, -0.998, 0.0])["clipped"], 4)
        self.assertEqual(measure([0.5, -0.9])["clipped"], 0)

    def test_steps_are_the_sizes_of_the_first_and_last_samples(self):
        got = measure([-0.25, 0.875, 0.5])
        self.assertEqual(got["head_step"], 0.25)
        self.assertEqual(got["tail_step"], 0.5)
        got = measure([0.0, 0.875, 0.0])
        self.assertEqual(got["head_step"], 0.0)
        self.assertEqual(got["tail_step"], 0.0)

    def test_steps_are_held_to_three_places(self):
        # 1/16 is 0.0625 exactly, and the half goes up. 1/3 does not end.
        got = measure([0.0625, 0.0, -1 / 3])
        self.assertEqual(got["head_step"], 0.063)
        self.assertEqual(got["tail_step"], 0.333)

    def test_one_sample_is_measured(self):
        self.assertEqual(
            measure([-0.5], 1000),
            {"seconds": 0.001, "peak_db": -6.0, "rms_db": -6.0, "dc": -0.5,
             "clipped": 0, "head_step": 0.5, "tail_step": 0.5},
        )

    def test_silence_has_a_peak_and_a_loudness_of_minus_infinity(self):
        self.assertEqual(
            measure(np.zeros(800), 8000),
            {"seconds": 0.1, "peak_db": float("-inf"), "rms_db": float("-inf"), "dc": 0.0,
             "clipped": 0, "head_step": 0.0, "tail_step": 0.0},
        )

    def test_a_list_and_an_array_measure_the_same(self):
        values = [0.5, -0.25, 0.125, 1.0, -0.75]
        self.assertEqual(measure(values), measure(np.array(values, dtype=np.float32)))
        self.assertEqual(measure(values), measure(np.array(values, dtype=np.float64)))

    def test_no_samples_is_an_error(self):
        with self.assertRaisesRegex(RowSynthError, "empty"):
            measure([])
        with self.assertRaisesRegex(RowSynthError, "empty"):
            measure(np.zeros(0, dtype=np.float32))

    def test_more_than_one_row_is_an_error(self):
        with self.assertRaisesRegex(RowSynthError, "one row"):
            measure(np.zeros((2, 100)))


if __name__ == "__main__":
    unittest.main()
