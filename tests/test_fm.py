"""Tests for the FM voice and its envelope."""
from __future__ import annotations

import math
import unittest

import numpy as np

from rowsynth.errors import RowSynthError
from rowsynth.fm import CARRIER, INTO, _k_gate_off, envelope, fm_tone, fm_voice

RATE = 8000

# For each algorithm: which operators bend each of the four, and which are heard.
WIRING = [
    ([[], [0], [1], [2]], [3]),
    ([[], [], [0, 1], [2]], [3]),
    ([[], [], [1], [0, 2]], [3]),
    ([[], [0], [], [1, 2]], [3]),
    ([[], [0], [], [2]], [1, 3]),
    ([[], [0], [0], [0]], [1, 2, 3]),
    ([[], [0], [], []], [1, 2, 3]),
    ([[], [], [], []], [0, 1, 2, 3]),
]

# A voice with something of everything: both pairs of algorithm 4, feedback, detune, four envelopes.
VOICE = {
    "alg": 4,
    "fb": 0.5,
    "ops": [
        {"mul": 3, "det": 0.7, "level": 1.6, "env": {"a": 0.002, "d": 0.03, "s": 0.4, "sd": 12, "r": 0.02}},
        {"mul": 1, "level": 0.8, "env": {"d": 0.05, "s": 0.6, "r": 0.03}},
        {"mul": 2, "det": -1.2, "level": 1.1, "env": {"a": 0.004, "d": 0, "s": 0.7, "r": 0.02}},
        {"mul": 1, "det": 0.3, "level": 0.6, "env": {"a": 0.001, "d": 0.02, "s": 0.5, "sd": 30, "r": 0.04}},
    ],
}


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


class AlgorithmTest(unittest.TestCase):
    def test_tables_have_a_row_an_algorithm_and_hold_only_0_and_1(self):
        self.assertEqual(INTO.shape, (8, 4, 4))
        self.assertEqual(CARRIER.shape, (8, 4))
        self.assertTrue(np.all((INTO == 0) | (INTO == 1)))
        self.assertTrue(np.all((CARRIER == 0) | (CARRIER == 1)))

    def test_each_algorithm_is_wired_as_its_chart_says(self):
        for alg, (bends, heard) in enumerate(WIRING):
            for i in range(4):
                with self.subTest(alg=alg, operator=i):
                    self.assertEqual([j for j in range(4) if INTO[alg, i, j] == 1], bends[i])
            with self.subTest(alg=alg):
                self.assertEqual([i for i in range(4) if CARRIER[alg, i] == 1], heard)

    def test_operator_is_bent_only_by_operators_before_it(self):
        for alg in range(8):
            for i in range(4):
                self.assertFalse(INTO[alg, i, i:].any(), (alg, i))

    def test_operator_that_is_heard_bends_nothing(self):
        for alg in range(8):
            for j in range(4):
                if CARRIER[alg, j] == 1:
                    self.assertFalse(INTO[alg, :, j].any(), (alg, j))


class VoiceTest(unittest.TestCase):
    def test_params_and_state_are_four_rows_of_float64(self):
        params, state = fm_voice(VOICE, RATE)
        self.assertEqual((params.shape, params.dtype), ((4, 8), np.float64))
        self.assertEqual((state.shape, state.dtype), ((4, 6), np.float64))
        self.assertFalse(state.any())

    def test_row_of_params_is_mul_det_level_and_the_envelope_as_steps_a_sample(self):
        op = {"mul": 2, "det": 1.5, "level": 0.7, "env": {"a": 0.01, "d": 0.02, "s": 0.5, "sd": 40, "r": 0.1}}
        params, _ = fm_voice({"ops": [op]}, RATE)
        # 80 samples of attack; 6.02 dB over 160 samples of decay; 40 dB a second; 60 dB over 800 samples.
        half_db = 20 * math.log10(0.5)
        np.testing.assert_allclose(params[0], [2, 1.5, 0.7, 1 / 80, half_db, -half_db / 160, 40 / 8000, 60 / 800], rtol=1e-12)

    def test_operator_with_nothing_set_is_a_sine_at_full_level_with_no_attack(self):
        params, _ = fm_voice({"ops": [{}]}, RATE)
        # A decay of 0.3 s to a sustain of 1 is no fall at all; a release of 0.2 s is 60 dB over 1600 samples.
        np.testing.assert_allclose(params[0], [1, 0, 1, 1, 0, 0, 0, 60 / 1600], rtol=1e-12)

    def test_decay_of_zero_is_an_endless_step(self):
        params, _ = fm_voice({"ops": [{"env": {"d": 0, "s": 0.5}}]}, RATE)
        self.assertEqual(params[0, 5], math.inf)

    def test_operators_left_out_are_silent_and_80_decibels_down_in_a_millisecond(self):
        params, _ = fm_voice({"ops": [{}, {}]}, RATE)
        for row in (2, 3):
            np.testing.assert_allclose(params[row], [1, 0, 0, 1, -80, 80 / 8, 0, 60 / 1600], rtol=1e-12)
        params, _ = fm_voice({}, RATE)
        self.assertFalse(params[:, 2].any())

    def test_steps_follow_the_sample_rate(self):
        slow, _ = fm_voice(VOICE, 22050)
        fast, _ = fm_voice(VOICE, 44100)
        # Half the samples in every second, so every step is twice as large; levels stay as they are.
        np.testing.assert_allclose(slow[:, 5:], 2 * fast[:, 5:], rtol=1e-12)
        np.testing.assert_allclose(slow[0, 3], 2 * fast[0, 3], rtol=1e-12)
        np.testing.assert_array_equal(slow[:, :3], fast[:, :3])
        np.testing.assert_array_equal(slow[:, 4], fast[:, 4])

    def test_gate_off_lets_every_operator_go(self):
        params, state = fm_voice(VOICE, RATE)
        _k_gate_off(params, state)
        self.assertTrue(np.all(state[:, 3] == 3.0))

    def test_algorithm_outside_0_to_7_is_an_error(self):
        for alg in (-1, 8, 2.5, 100):
            with self.assertRaisesRegex(RowSynthError, "^alg: not an algorithm"):
                fm_voice({"alg": alg}, RATE)

    def test_every_algorithm_from_0_to_7_is_taken(self):
        for alg in range(8):
            fm_voice({"alg": alg}, RATE)

    def test_more_than_four_operators_is_an_error(self):
        with self.assertRaisesRegex(RowSynthError, "^ops: 5 operators"):
            fm_voice({"ops": [{}, {}, {}, {}, {}]}, RATE)

    def test_value_that_is_not_a_number_is_an_error_naming_the_key(self):
        cases = [
            ({"alg": "4"}, r"^alg: not a number"),
            ({"fb": "0.5"}, r"^fb: not a number"),
            ({"fb": None}, r"^fb: not a number"),
            ({"ops": [{}, {"mul": "2"}]}, r"^ops\[1\]\.mul: not a number"),
            ({"ops": [{"det": [1]}]}, r"^ops\[0\]\.det: not a number"),
            ({"ops": [{}, {}, {"level": True}]}, r"^ops\[2\]\.level: not a number"),
            ({"ops": [{}, {}, {}, {"env": {"d": "x"}}]}, r"^ops\[3\]\.env\.d: not a number"),
        ]
        for inst, message in cases:
            with self.assertRaisesRegex(RowSynthError, message):
                fm_voice(inst, RATE)

    def test_operators_of_the_wrong_shape_are_an_error_naming_the_place(self):
        cases = [
            ({"ops": {"mul": 1}}, r"^ops: not a list"),
            ({"ops": [{}, 3]}, r"^ops\[1\]: not an operator"),
            ({"ops": [{"env": 0.5}]}, r"^ops\[0\]\.env: not an envelope"),
        ]
        for inst, message in cases:
            with self.assertRaisesRegex(RowSynthError, message):
                fm_voice(inst, RATE)


class ToneTest(unittest.TestCase):
    def test_is_float32_and_seconds_long(self):
        out = fm_tone(VOICE, 220.0, 0.01, 0.02, RATE)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.shape, (160,))

    def test_algorithm_7_with_one_operator_is_a_sine_at_its_level(self):
        out = fm_tone({"alg": 7, "ops": [{"level": 0.5}]}, 440.0, 1.0, 0.05, RATE)
        n = np.arange(400)
        np.testing.assert_allclose(out, 0.5 * np.sin(2 * math.pi * 440 * n / RATE), atol=1e-6)

    def test_mul_and_det_place_an_operator_at_freq_times_mul_plus_det(self):
        out = fm_tone({"alg": 7, "ops": [{"mul": 2, "det": 5}]}, 300.0, 1.0, 0.05, RATE)
        n = np.arange(400)
        np.testing.assert_allclose(out, np.sin(2 * math.pi * 605 * n / RATE), atol=1e-6)

    def test_algorithm_7_adds_its_four_operators_up(self):
        ops = [{"level": 0.4}, {"mul": 2, "level": 0.3}, {"mul": 3, "level": 0.2}, {"mul": 5, "level": 0.1}]
        out = fm_tone({"alg": 7, "ops": ops}, 200.0, 1.0, 0.05, RATE)
        turn = 2 * math.pi * 200 * np.arange(400) / RATE
        expected = 0.4 * np.sin(turn) + 0.3 * np.sin(2 * turn) + 0.2 * np.sin(3 * turn) + 0.1 * np.sin(5 * turn)
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_modulator_bends_its_carrier_within_the_same_sample(self):
        # Algorithm 4, first pair only: the carrier's phase is moved by the modulator's level in radians.
        out = fm_tone({"alg": 4, "ops": [{"level": 1.5}, {"level": 0.8}]}, 250.0, 1.0, 0.05, RATE)
        turn = 2 * math.pi * 250 * np.arange(400) / RATE
        np.testing.assert_allclose(out, 0.8 * np.sin(turn + 1.5 * np.sin(turn)), atol=1e-6)

    def test_modulator_is_not_heard(self):
        out = fm_tone({"alg": 0, "ops": [{"level": 1}]}, 250.0, 1.0, 0.02, RATE)
        self.assertFalse(out.any())

    def test_feedback_bends_the_first_operator_by_its_own_last_sample(self):
        out = fm_tone({"alg": 7, "fb": 0.8, "ops": [{"level": 0.5}]}, 500.0, 1.0, 0.01, RATE)
        turn = 2 * math.pi * 500 / RATE
        last = 0.0
        for n in range(80):
            last = math.sin(n * turn + 0.8 * last)
            self.assertAlmostEqual(float(out[n]), 0.5 * last, places=6)

    def test_feedback_changes_the_wave(self):
        plain = fm_tone({"alg": 7, "fb": 0, "ops": [{}]}, 500.0, 1.0, 0.01, RATE)
        bent = fm_tone({"alg": 7, "fb": 0.8, "ops": [{}]}, 500.0, 1.0, 0.01, RATE)
        # The first two samples have no last sample to speak of; after that the two part.
        np.testing.assert_array_equal(plain[:2], bent[:2])
        self.assertGreater(float(np.abs(plain - bent).max()), 0.1)

    def test_feedback_reaches_only_the_first_operator(self):
        plain = fm_tone({"alg": 7, "fb": 0, "ops": [{"level": 0}, {}]}, 500.0, 1.0, 0.01, RATE)
        bent = fm_tone({"alg": 7, "fb": 0.8, "ops": [{"level": 0}, {}]}, 500.0, 1.0, 0.01, RATE)
        np.testing.assert_array_equal(plain, bent)

    def test_heard_operator_is_a_sine_under_its_envelope(self):
        env = {"a": 0.005, "d": 0.01, "s": 0.5, "sd": 20, "r": 0.01}
        out = fm_tone({"alg": 7, "ops": [{"level": 0.7, "env": env}]}, 440.0, 0.03, 0.06, RATE)
        shape = envelope(env, 0.03, 0.06, RATE)
        n = np.arange(480)
        np.testing.assert_allclose(out, 0.7 * np.sin(2 * math.pi * 440 * n / RATE) * shape, atol=1e-6)

    def test_voice_is_let_go_after_held_seconds_and_ends_in_silence(self):
        # A release of 0.01 s is 0.75 dB a sample: 96 dB down 128 samples after sample 400.
        out = fm_tone({"alg": 7, "ops": [{"env": {"r": 0.01}}]}, 440.0, 0.05, 0.08, RATE)
        self.assertGreater(float(np.abs(out[380:400]).max()), 0.9)
        self.assertLess(float(np.abs(out[480:]).max()), 0.001)
        self.assertFalse(out[530:].any())

    def test_known_voice_has_these_samples(self):
        out = fm_tone(VOICE, 220.0, 0.05, 0.08, RATE)
        expected = {1: 0.243882, 10: 1.066261, 60: -0.544598, 149: 0.963841, 200: 0.111779,
                    399: -0.360567, 403: 0.572821, 431: -0.275031, 519: 0.034697, 634: 0.000656}
        self.assertEqual(float(out[0]), 0.0)
        for index, value in expected.items():
            self.assertAlmostEqual(float(out[index]), value, places=6, msg=f"sample {index}")

    def test_instrument_that_is_wrong_is_an_error(self):
        with self.assertRaisesRegex(RowSynthError, "^alg: "):
            fm_tone({"alg": 9}, 440.0, 0.01, 0.02, RATE)
        with self.assertRaisesRegex(RowSynthError, r"^ops\[0\]\.env\.a: "):
            fm_tone({"ops": [{"env": {"a": "slow"}}]}, 440.0, 0.01, 0.02, RATE)


if __name__ == "__main__":
    unittest.main()
