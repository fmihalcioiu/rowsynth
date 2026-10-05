"""Tests for the FM operator's envelope."""
from __future__ import annotations

import math
import unittest

import numpy as np

from rowsynth.errors import RowSynthError
from rowsynth.fm import envelope

RATE = 8000


def level(db: float) -> float:
    """Decibels as an amplitude."""
    return 10 ** (db / 20)


class EnvelopeTest(unittest.TestCase):
    def test_is_float32_and_seconds_long(self):
        out = envelope({}, 0.01, 0.05, RATE)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.shape, (400,))

    def test_attack_is_a_straight_rise_to_full_after_a_seconds(self):
        # 0.01 s at 8000 Hz is 80 samples, an eightieth of the way each.
        out = envelope({"a": 0.01}, 1.0, 0.02, RATE)
        self.assertAlmostEqual(float(out[0]), 1 / 80, places=6)
        self.assertAlmostEqual(float(out[19]), 0.25, places=6)
        self.assertAlmostEqual(float(out[39]), 0.5, places=6)
        self.assertAlmostEqual(float(out[79]), 1.0, places=6)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_no_attack_is_full_at_the_first_sample(self):
        out = envelope({}, 1.0, 0.01, RATE)
        self.assertEqual(float(out[0]), 1.0)

    def test_default_envelope_stays_full_while_held(self):
        out = envelope({}, 1.0, 0.1, RATE)
        self.assertTrue(np.all(out == 1.0))

    def test_decay_is_a_straight_line_in_decibels_reaching_the_sustain_level_after_d_seconds(self):
        # Full at sample 0, then 80 samples down to a half: 6.02 dB, so 3.01 dB at the middle.
        out = envelope({"d": 0.01, "s": 0.5}, 1.0, 0.03, RATE)
        half_db = 20 * math.log10(0.5)
        self.assertAlmostEqual(float(out[20]), level(half_db / 4), places=6)
        self.assertAlmostEqual(float(out[40]), level(half_db / 2), places=6)
        self.assertAlmostEqual(float(out[40]), math.sqrt(0.5), places=6)
        self.assertAlmostEqual(float(out[80]), 0.5, places=6)
        # And there it stays.
        self.assertTrue(np.all(out[81:] == 0.5))

    def test_decay_of_zero_is_at_the_sustain_level_one_sample_after_full(self):
        out = envelope({"d": 0, "s": 0.25}, 1.0, 0.01, RATE)
        self.assertEqual(float(out[0]), 1.0)
        self.assertTrue(np.all(out[1:] == 0.25))

    def test_sustain_of_zero_rests_80_decibels_down(self):
        out = envelope({"d": 0, "s": 0}, 1.0, 0.01, RATE)
        self.assertAlmostEqual(float(out[-1]), 0.0001, places=9)

    def test_falling_sustain_loses_sd_decibels_a_second(self):
        # At the sustain level of a half from sample 1 on, then 200 dB a second: 20 dB in 800 samples.
        out = envelope({"d": 0, "s": 0.5, "sd": 200}, 1.0, 0.11, RATE)
        self.assertAlmostEqual(float(out[1]), 0.5, places=6)
        self.assertAlmostEqual(float(out[401]), 0.5 * level(-10), places=6)
        self.assertAlmostEqual(float(out[801]), 0.05, places=6)

    def test_release_is_60_decibels_down_after_r_seconds(self):
        # Let go at sample 400. 0.1 s is 800 samples, 0.075 dB each, the first of them at sample 400.
        out = envelope({"r": 0.1}, 0.05, 0.16, RATE)
        self.assertEqual(float(out[399]), 1.0)
        self.assertAlmostEqual(float(out[400]), level(-0.075), places=6)
        self.assertAlmostEqual(float(out[799]), level(-30), places=6)
        self.assertAlmostEqual(float(out[1199]), 0.001, places=6)

    def test_release_starts_from_the_sustain_level(self):
        out = envelope({"d": 0, "s": 0.5, "r": 0.1}, 0.05, 0.11, RATE)
        self.assertEqual(float(out[399]), 0.5)
        self.assertAlmostEqual(float(out[799]), 0.5 * level(-30), places=6)

    def test_release_is_never_shorter_than_a_millisecond(self):
        # A millisecond is 8 samples at 8000 Hz: 7.5 dB each.
        out = envelope({"r": 0}, 0.05, 0.06, RATE)
        self.assertAlmostEqual(float(out[400]), level(-7.5), places=6)
        self.assertAlmostEqual(float(out[407]), 0.001, places=6)

    def test_envelope_is_done_below_minus_96_decibels(self):
        # 0.075 dB a sample from sample 400: 95.25 dB down at sample 1669, 96 at sample 1679.
        out = envelope({"r": 0.1}, 0.05, 0.3, RATE)
        self.assertAlmostEqual(float(out[1669]), level(-95.25), places=8)
        self.assertGreater(float(out[1669]), 0.0)
        self.assertFalse(out[1681:].any())

    def test_release_from_the_attack_falls_from_the_level_reached(self):
        # An attack of 800 samples let go half way up, at a half; then 0.075 dB a sample.
        out = envelope({"a": 0.1, "r": 0.1}, 0.05, 0.2, RATE)
        self.assertAlmostEqual(float(out[399]), 0.5, places=6)
        self.assertAlmostEqual(float(out[400]), 0.5 * level(-0.075), places=6)
        self.assertAlmostEqual(float(out[799]), 0.5 * level(-30), places=6)
        self.assertAlmostEqual(float(out[1199]), 0.0005, places=6)
        self.assertLessEqual(float(out.max()), 0.5 + 1e-6)

    def test_never_let_go_when_held_to_the_end(self):
        out = envelope({"r": 0.001}, 0.05, 0.05, RATE)
        self.assertTrue(np.all(out == 1.0))

    def test_value_that_is_not_a_number_is_an_error_naming_the_key(self):
        for key in ("a", "d", "s", "sd", "r"):
            for bad in ("0.1", None, True, [0.1]):
                with self.assertRaisesRegex(RowSynthError, rf"env\.{key}: not a number"):
                    envelope({key: bad}, 0.01, 0.02, RATE)

    def test_envelope_that_is_not_a_table_is_an_error(self):
        with self.assertRaisesRegex(RowSynthError, "env: not an envelope"):
            envelope([0.1, 0.2], 0.01, 0.02, RATE)


if __name__ == "__main__":
    unittest.main()
