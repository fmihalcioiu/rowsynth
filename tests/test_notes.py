"""Tests for rowsynth.notes."""
from __future__ import annotations

import unittest

from rowsynth.errors import RowSynthError
from rowsynth.notes import hz, midi


class MidiTest(unittest.TestCase):
    def test_names_give_their_numbers(self):
        for name, number in (("a4", 69.0), ("c4", 60.0), ("c#5", 73.0), ("eb3", 51.0), ("g9", 127.0), ("b0", 23.0)):
            self.assertEqual(midi(name), number, name)

    def test_every_letter_sits_on_its_semitone(self):
        self.assertEqual([midi(letter + "4") for letter in "cdefgab"], [60.0, 62.0, 64.0, 65.0, 67.0, 69.0, 71.0])

    def test_sharp_adds_one_and_flat_takes_one(self):
        self.assertEqual(midi("f#3"), midi("f3") + 1)
        self.assertEqual(midi("fb3"), midi("f3") - 1)
        self.assertEqual(midi("bb2"), 46.0)
        self.assertEqual(midi("b#2"), midi("c3"))

    def test_octave_may_be_negative(self):
        self.assertEqual(midi("c-1"), 0.0)
        self.assertEqual(midi("a-2"), -3.0)
        self.assertEqual(midi("bb-1"), 10.0)

    def test_letter_case_does_not_matter(self):
        self.assertEqual(midi("A4"), 69.0)
        self.assertEqual(midi("C#5"), 73.0)
        self.assertEqual(midi("EB3"), 51.0)

    def test_name_gives_a_float(self):
        self.assertIs(type(midi("a4")), float)

    def test_number_passes_through_as_a_float(self):
        self.assertEqual(midi(69), 69.0)
        self.assertIs(type(midi(69)), float)
        self.assertEqual(midi(60.5), 60.5)
        self.assertEqual(midi(-7), -7.0)

    def test_bad_name_is_refused(self):
        for name in ("h4", "a", "4", "", "a10", "a##4", "a#b4", "a 4", " a4", "a4 ", "a4\n", "a--1", "a4.5", "69"):
            with self.assertRaises(RowSynthError, msg=repr(name)):
                midi(name)

    def test_what_is_neither_name_nor_number_is_refused(self):
        for name in (None, True, False, ["a4"], ("a", 4), b"a4", {"note": "a4"}):
            with self.assertRaises(RowSynthError, msg=repr(name)):
                midi(name)

    def test_error_names_what_was_given(self):
        with self.assertRaises(RowSynthError) as caught:
            midi("x9")
        self.assertIn("'x9'", str(caught.exception))
        with self.assertRaises(RowSynthError) as caught:
            midi(None)
        self.assertIn("None", str(caught.exception))


class HzTest(unittest.TestCase):
    def test_a4_is_440(self):
        self.assertEqual(hz(69), 440.0)

    def test_twelve_semitones_double_the_frequency(self):
        self.assertEqual(hz(81), 880.0)
        self.assertEqual(hz(57), 220.0)
        self.assertEqual(hz(33), 55.0)
        self.assertEqual(hz(105), 3520.0)

    def test_known_notes(self):
        self.assertAlmostEqual(hz(60), 261.625565, places=6)
        self.assertAlmostEqual(hz(0), 8.175799, places=6)
        self.assertAlmostEqual(hz(127), 12543.853951, places=6)

    def test_fractions_fall_between_their_neighbours(self):
        self.assertAlmostEqual(hz(72.5), 538.583559, places=6)
        self.assertLess(hz(72), hz(72.5))
        self.assertLess(hz(72.5), hz(73))

    def test_a_semitone_is_the_twelfth_root_of_two(self):
        self.assertAlmostEqual(hz(70) / hz(69), 2.0 ** (1.0 / 12.0), places=12)

    def test_takes_what_midi_gives(self):
        self.assertEqual(hz(midi("a3")), 220.0)
