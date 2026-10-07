"""Tests for rowsynth.bank."""
from __future__ import annotations

import copy
import unittest

import numpy as np

from rowsynth.bank import PEAK_DB, RENDER_GAIN, normalise, render_bank, render_entry
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

    def test_refuses_an_entry_of_parts_and_a_held_one_as_kinds_that_cannot_be_rendered(self):
        for kind, value in (("parts", [[0, {"def": CHIRP}]]), ("held", {"note": CHIRP}), ("parts", None)):
            said = self.refused({kind: value, "db": -3})
            self.at("entry", said, kind)
            self.assertIn(repr(kind), said)
            self.assertIn("cannot be rendered", said)

    def test_refuses_defs_that_are_not_a_list(self):
        for defs in (None, {}, "defs", 3, CHIRP):
            self.at("defs", self.refused({"defs": defs}), defs)

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
    def test_every_sound_has_its_buffers_its_meta_and_no_held(self):
        tags = {"priority": 1, "tags": ["ui"]}
        sounds = {"chirp": {"def": CHIRP, "meta": tags}, "steps": {"defs": STEPS, "db": [-1, -3.5, 0]},
                  "crash": {"def": CRASH, "db": -6, "meta": {}}}
        bank = render_bank(sounds, RATE)
        self.assertEqual(list(bank), ["chirp", "steps", "crash"])
        for name, entry in sounds.items():
            self.assertEqual(sorted(bank[name]), ["buffers", "held", "meta"], name)
            self.assertTrue(same(bank[name]["buffers"], render_entry(entry, RATE)), name)
            self.assertIsNone(bank[name]["held"], name)
        self.assertIs(bank["chirp"]["meta"], tags)
        self.assertEqual(tags, {"priority": 1, "tags": ["ui"]})
        self.assertEqual(bank["steps"]["meta"], {})
        self.assertEqual(bank["crash"]["meta"], {})

    def test_rate_left_out_is_44100_and_no_sounds_are_an_empty_bank(self):
        self.assertEqual(len(render_bank({"quiet": {"def": QUIET}})["quiet"]["buffers"][0]), 1470)
        self.assertEqual(len(render_bank({"quiet": {"def": QUIET}}, 48000)["quiet"]["buffers"][0]), 1600)
        self.assertEqual(render_bank({}, RATE), {})

    def test_puts_the_name_of_the_sound_in_front_of_what_is_refused(self):
        bad = {"voices": [{"inst": PING, "notes": [[0, "h5", 2]]}]}
        wrong = (({"defs": [CHIRP, bad]}, "defs[1]: voices[0].notes[0][1]: "), ({"def": CHIRP, "db": [0, 0]}, "db: "),
                 ({"def": CHIRP, "defs": STEPS}, "entry: "), ({"held": {}}, "entry: "), (None, "entry: "))
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
