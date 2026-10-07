"""Tests for rowsynth.sound."""
from __future__ import annotations

import copy
import math
import unittest
from unittest import mock

import numpy as np

from rowsynth.errors import RowSynthError
from rowsynth.note import render_note
from rowsynth.sound import finish, render_sound

# A tick is 133 and a third samples at 8000 Hz: ticks start at samples 0, 133, 267, 400, 533, 667, 800.
RATE = 8000
# The share of itself the high-pass keeps from one sample to the next at 8000 Hz.
KEPT = 1 - 2 * math.pi * 30 / RATE
# Noise at full level from its first tick to its last. Its first sample is never 0.
HISS = {"wave": "noise", "release": 0}
# A triangle that stays well under the knee.
HUM = {"wave": "tri", "vol": 12}
BLIP = {"vol": [15, 11, 8], "duty": 1, "release": 1}
GLASS = {
    "wave": "fm",
    "alg": 4,
    "fb": 0.3,
    "release": 3,
    "ops": [
        {"mul": 3, "det": 0.5, "level": 1.1, "env": {"a": 0.002, "d": 0.04, "s": 0.4, "sd": 10, "r": 0.03}},
        {"level": 0.8, "env": {"d": 0.06, "s": 0.6, "r": 0.04}},
        {"mul": 2, "level": 0.9, "env": {"a": 0.003, "d": 0.03, "s": 0.5, "r": 0.02}},
        {"det": -0.4, "level": 0.5, "env": {"a": 0.001, "d": 0.05, "s": 0.7, "r": 0.05}},
    ],
}
# Samples that jump about between -0.6 and 0.6, none of them 0.
WASH = ((np.arange(700) * 37) % 101 - 50.5) / 85


def sound(notes: list, inst: dict = HISS, rate: int = RATE, **keys: object) -> np.ndarray:
    """One voice playing notes, in doubles."""
    return render_sound({"voices": [{"inst": inst, "notes": notes}], **keys}, rate).astype(np.float64)


def stage(samples: object, gain: float = 1.0, rate: int = RATE, fade: bool = True) -> np.ndarray:
    """Samples through the output stage, in doubles. What was handed in is left as it was."""
    return finish(np.array(samples, dtype=np.float32), gain, rate, fade).astype(np.float64)


def peak(samples: np.ndarray) -> float:
    """The size of the largest sample."""
    return float(np.abs(samples).max())


class LengthTest(unittest.TestCase):
    def test_buffer_ends_two_ticks_after_the_last_release(self):
        # Held for four ticks, let go for the two an instrument has when it says nothing, and two more: eight ticks.
        self.assertEqual(len(sound([[0, "a4", 4]], {})), 1067)
        self.assertEqual(len(sound([[3, "a4", 4]], {"release": 5})), 1867)
        self.assertEqual(len(sound([[0, "a4", 1]], {"release": 0})), 400)

    def test_note_that_ends_last_sets_the_length(self):
        self.assertEqual(len(sound([[0, "a4", 4], [1, "c5", 1]], {})), 1067)
        self.assertEqual(len(sound([[1, "c5", 1], [0, "a4", 4]], {})), 1067)

    def test_voice_that_ends_last_sets_the_length(self):
        voices = [{"inst": {"release": 0}, "notes": [[0, 69, 3]]}, {"inst": {"release": 6}, "notes": [[0, 57, 1]]},
                  {"inst": {}, "notes": [[1, 64, 1]]}]
        self.assertEqual(len(render_sound({"voices": voices}, RATE)), 1200)

    def test_part_of_a_tick_rounds_the_length_up(self):
        self.assertEqual(len(sound([[0, "a4", 1.5]], {"release": 0})), 467)
        self.assertEqual(len(sound([[0.25, "a4", 1]], {"release": 0})), 434)
        self.assertEqual(len(sound([[0, "a4", 1]], {"release": 0.7})), 494)

    def test_nothing_to_play_is_two_ticks_of_silence(self):
        for voices in ([], [{"inst": {}, "notes": []}], [{"inst": {"release": 0}, "notes": [[0, "a4", 0]]}]):
            out = render_sound({"voices": voices}, RATE)
            self.assertEqual(len(out), 267, voices)
            self.assertFalse(out.any(), voices)

    def test_length_goes_by_the_rate(self):
        self.assertEqual(len(render_sound({"voices": []}, 11025)), 368)
        self.assertEqual(len(render_sound({"voices": []}, 48000)), 1600)

    def test_rate_left_out_is_44100(self):
        defn = {"voices": [{"inst": HISS, "notes": [[0, 81, 1]]}]}
        self.assertEqual(len(render_sound(defn)), 2205)
        self.assertTrue(np.array_equal(render_sound(defn), render_sound(defn, 44100)))


class NotesTest(unittest.TestCase):
    def test_note_lands_on_its_tick(self):
        for tick, start in ((0, 0), (1, 133), (2, 267), (3, 400), (4.5, 600)):
            out = sound([[tick, 81, 1]])
            self.assertFalse(out[:start].any(), tick)
            self.assertNotEqual(float(out[start]), 0.0, tick)

    def test_start_rounds_a_half_up(self):
        # Six ticks at 11025 Hz are 1102.5 samples.
        out = sound([[6, 81, 1]], rate=11025)
        self.assertFalse(out[:1103].any())
        self.assertNotEqual(float(out[1103]), 0.0)

    def test_sound_is_its_notes_added_into_one_buffer_and_finished(self):
        defn = {"voices": [{"inst": BLIP, "notes": [[0, "c4", 3], [2, "e4", 3, 0.5]]},
                           {"inst": GLASS, "notes": [[1, 55, 4, 0.8]]}], "gain": 0.4}
        got = render_sound(defn, RATE)
        out = np.zeros(len(got), dtype=np.float32)
        render_note(out, 0, BLIP, 60.0, 3, 1.0, RATE)
        render_note(out, 267, BLIP, 64.0, 3, 0.5, RATE)
        render_note(out, 133, GLASS, 55.0, 4, 0.8, RATE)
        self.assertTrue(np.array_equal(got, finish(out, 0.4, RATE)))

    def test_name_is_its_number(self):
        for name, number in (("a4", 69), ("c4", 60), ("c#5", 73), ("eb3", 51), ("A4", 69)):
            self.assertTrue(np.array_equal(sound([[0, name, 2]], HUM), sound([[0, number, 2]], HUM)), name)
        self.assertFalse(np.array_equal(sound([[0, "a4", 2]], HUM), sound([[0, 70, 2]], HUM)))

    def test_number_can_lie_between_two_names(self):
        between = sound([[0, 69.5, 2]], HUM)
        self.assertFalse(np.array_equal(between, sound([[0, 69, 2]], HUM)))
        self.assertFalse(np.array_equal(between, sound([[0, 70, 2]], HUM)))

    def test_velocity_left_out_is_1(self):
        self.assertTrue(np.array_equal(sound([[0, 69, 2]], HUM), sound([[0, 69, 2, 1.0]], HUM)))
        self.assertFalse(np.array_equal(sound([[0, 69, 2]], HUM), sound([[0, 69, 2, 0.9]], HUM)))

    def test_velocity_scales_its_note(self):
        np.testing.assert_allclose(sound([[0, 69, 2, 0.5]], HUM), 0.5 * sound([[0, 69, 2]], HUM), rtol=0, atol=1e-6)

    def test_notes_on_one_tick_add_up(self):
        twice = sound([[0, 69, 2], [0, 69, 2]], HUM, gain=0.25)
        np.testing.assert_allclose(twice, sound([[0, 69, 2]], HUM, gain=0.5), rtol=0, atol=1e-6)

    def test_voices_add_up(self):
        one = {"voices": [{"inst": HUM, "notes": [[0, 57, 3], [1, 64, 2]]}]}
        two = {"voices": [{"inst": HUM, "notes": [[0, 57, 3]]}, {"inst": HUM, "notes": [[1, 64, 2]]}]}
        self.assertTrue(np.array_equal(render_sound(one, RATE), render_sound(two, RATE)))

    def test_gain_left_out_is_a_half(self):
        self.assertTrue(np.array_equal(sound([[0, 69, 2]], HUM), sound([[0, 69, 2]], HUM, gain=0.5)))
        self.assertFalse(np.array_equal(sound([[0, 69, 2]], HUM), sound([[0, 69, 2]], HUM, gain=0.4)))

    def test_gain_scales_a_sound_that_stays_under_the_knee(self):
        quiet = sound([[0, 69, 3]], HUM, gain=0.2)
        np.testing.assert_allclose(quiet, 0.4 * sound([[0, 69, 3]], HUM, gain=0.5), rtol=0, atol=1e-6)

    def test_sound_ends_on_silence(self):
        # The note is over at sample 533, two ticks before the end, and the high-pass is still settling from it.
        out = sound([[0, "a2", 4]], {"wave": "saw", "release": 0})
        self.assertEqual(len(out), 800)
        self.assertTrue(out[540:760].all())
        self.assertEqual(float(out[-1]), 0.0)

    def test_gives_float32_samples(self):
        out = render_sound({"voices": [{"inst": HUM, "notes": [[0, 69, 1]]}]}, RATE)
        self.assertIs(type(out), np.ndarray)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.ndim, 1)

    def test_tuples_do_for_lists(self):
        listed = {"voices": [{"inst": HUM, "notes": [[0, "a3", 2], [1, 60, 2, 0.5]]}]}
        tupled = {"voices": ({"inst": HUM, "notes": ((0, "a3", 2), (1, 60, 2, 0.5))},)}
        self.assertTrue(np.array_equal(render_sound(listed, RATE), render_sound(tupled, RATE)))

    def test_definition_is_left_as_it_was(self):
        defn = {"voices": [{"inst": dict(BLIP), "notes": [[0, "c4", 2], [1, 64, 2, 0.5]]},
                           {"inst": copy.deepcopy(GLASS), "notes": [[0, 55, 2]]}], "gain": 0.4}
        before = copy.deepcopy(defn)
        render_sound(defn, RATE)
        self.assertEqual(defn, before)


class KneeTest(unittest.TestCase):
    def test_keeps_a_loud_sound_under_full_scale(self):
        quiet = sound([[0, "a3", 4]], {"wave": "saw"}, gain=0.5)
        loud = sound([[0, "a3", 4]], {"wave": "saw"}, gain=50)
        self.assertLess(peak(quiet), 0.7)
        self.assertGreater(peak(loud), 0.98)
        self.assertLess(peak(loud), 1.0)

    def test_leaves_what_is_under_0_7_as_it_is(self):
        # The first sample of the buffer comes out of the high-pass at the share it keeps.
        for level in (0.1, 0.5, 0.7, -0.6):
            self.assertAlmostEqual(stage(np.full(100, level))[0], KEPT * level, places=6, msg=level)

    def test_rounds_off_what_is_over_0_7(self):
        for level in (0.72, 0.8, 1.5, 4.0, -1.5):
            over = KEPT * abs(level) - 0.7
            self.assertGreater(over, 0.0)
            bent = math.copysign(0.7 + 0.29 * math.tanh(over / 0.29), level)
            self.assertAlmostEqual(stage(np.full(100, level))[0], bent, places=6, msg=level)

    def test_louder_in_is_louder_out_and_nothing_passes_0_99(self):
        got = [stage(np.full(100, level))[0] for level in (0.5, 0.7, 0.75, 0.9, 1.2, 2.0, 3.0)]
        self.assertTrue(all(more > less for less, more in zip(got, got[1:])), got)
        self.assertLess(got[-1], 0.99)
        self.assertAlmostEqual(stage(np.full(100, 1000.0))[0], 0.99, places=6)
        self.assertAlmostEqual(stage(np.full(100, -1000.0))[0], -0.99, places=6)

    def test_bends_both_ways_alike(self):
        samples = 5 * WASH[:300]
        self.assertGreater(peak(stage(samples)), 0.9)
        self.assertTrue(np.array_equal(stage(-samples), -stage(samples)))

    def test_gain_comes_before_the_knee(self):
        np.testing.assert_allclose(stage(WASH[:300], gain=4), stage(4 * WASH[:300]), rtol=0, atol=1e-6)
        self.assertGreater(peak(stage(WASH[:300], gain=4)), 0.9)


class HighPassTest(unittest.TestCase):
    def test_removes_an_offset(self):
        # 400 Hz is twenty samples a turn, so 400 samples are whole turns and their mean is their offset.
        turns = 0.2 * np.sin(2 * np.pi * 400 * np.arange(1200) / RATE)
        out = stage(0.4 + turns)
        self.assertAlmostEqual(float(out[600:1000].mean()), 0.0, places=4)
        self.assertAlmostEqual(peak(out[600:1000]), 0.2, delta=0.005)

    def test_offset_dies_away_by_the_same_share_every_sample(self):
        out = stage(np.full(300, 0.5))
        self.assertAlmostEqual(out[0], 0.5 * KEPT, places=6)
        np.testing.assert_allclose(out[1:200] / out[:199], KEPT, rtol=0, atol=1e-4)

    def test_share_it_keeps_goes_by_the_rate(self):
        for rate in (8000, 11025, 44100, 48000):
            out = stage(np.full(400, 0.5), rate=rate)
            self.assertAlmostEqual(out[0], 0.5 * (1 - 2 * math.pi * 30 / rate), places=6, msg=rate)

    def test_takes_3_decibels_off_a_tone_at_30_hz_and_little_off_one_at_300_hz(self):
        # A second of each tone is whole turns of it. The fade comes after that second.
        lost = []
        for hertz in (3, 30, 300):
            tone = 0.5 * np.sin(2 * np.pi * hertz * np.arange(8100) / RATE)
            out = stage(tone)
            lost.append(10 * math.log10(float(np.mean(tone[:8000] ** 2)) / float(np.mean(out[:8000] ** 2))))
        self.assertAlmostEqual(lost[0], 20.0, delta=0.5)
        self.assertAlmostEqual(lost[1], 3.0, delta=0.3)
        self.assertLess(lost[2], 0.2)


class FadeTest(unittest.TestCase):
    def test_end_fades_to_nothing_over_its_last_5_milliseconds(self):
        # Forty samples at 8000 Hz. Before them the buffer is as it would be with more to come.
        short = stage(WASH[:300])
        long = stage(WASH)
        self.assertTrue(np.array_equal(short[:260], long[:260]))
        np.testing.assert_allclose(short[260:], long[260:300] * np.arange(39, -1, -1) / 40, rtol=0, atol=1e-7)
        self.assertEqual(short[-1], 0.0)
        self.assertNotEqual(short[-2], 0.0)

    def test_fade_takes_as_long_at_any_rate(self):
        for rate, count in ((8000, 40), (11025, 55), (44100, 220), (48000, 239)):
            changed = np.flatnonzero(stage(WASH[:400], rate=rate) != stage(WASH, rate=rate)[:400])
            self.assertEqual(int(changed[0]), 400 - count, rate)

    def test_buffer_shorter_than_the_fade_fades_from_its_first_sample(self):
        short = stage(WASH[:10])
        np.testing.assert_allclose(short, stage(WASH)[:10] * np.arange(9, -1, -1) / 10, rtol=0, atol=1e-7)
        self.assertNotEqual(short[0], 0.0)

    def test_one_sample_fades_to_nothing_and_an_empty_buffer_stays_empty(self):
        self.assertEqual(stage([0.5]).tolist(), [0.0])
        for fade in (True, False):
            self.assertEqual(len(stage([], fade=fade)), 0)

    def test_fade_false_leaves_the_end_as_loud_as_it_was(self):
        self.assertLess(peak(stage(WASH[:300])[-10:]), 0.2)
        self.assertGreater(peak(stage(WASH[:300], fade=False)[-10:]), 0.5)

    def test_fade_false_finishes_a_loop_as_its_second_time_round(self):
        # The high-pass starts from what the end of the loop leaves in it, and the end is left as it is.
        loop = WASH[:300]
        got = stage(loop, fade=False)
        self.assertTrue(np.array_equal(got, stage(np.tile(loop, 3))[300:600]))
        self.assertFalse(np.array_equal(got[:20], stage(loop)[:20]))

    def test_fade_false_keeps_a_loop_with_an_offset_from_stepping_where_it_joins(self):
        steady = np.full(400, 0.4)
        self.assertGreater(stage(steady)[0], 0.39)
        self.assertLess(peak(stage(steady, fade=False)), 1e-4)

    def test_fade_false_still_bends_what_is_over_the_knee(self):
        got = stage(5 * WASH[:300], fade=False)
        self.assertGreater(peak(got), 0.9)
        self.assertLess(peak(got), 1.0)


class BufferTest(unittest.TestCase):
    def test_works_in_place_and_hands_back_its_buffer(self):
        out = WASH[:300].astype(np.float32)
        before = out.copy()
        self.assertIs(finish(out, 1.0, RATE), out)
        self.assertFalse(np.array_equal(out, before))
        self.assertEqual(out.dtype, np.float32)

    def test_gain_rate_and_fade_left_out_are_a_half_44100_and_a_fade(self):
        samples = (3 * WASH[:300]).astype(np.float32)
        left_out = finish(samples.copy())
        self.assertTrue(np.array_equal(left_out, finish(samples.copy(), 0.5, 44100, True)))
        for other in ((0.6, 44100, True), (0.5, 48000, True), (0.5, 44100, False)):
            self.assertFalse(np.array_equal(left_out, finish(samples.copy(), *other)), other)

    def test_gain_of_0_and_silence_come_out_as_silence(self):
        for fade in (True, False):
            self.assertFalse(stage(WASH[:300], gain=0, fade=fade).any())
            self.assertFalse(stage(np.zeros(300), fade=fade).any())


class KnownSamplesTest(unittest.TestCase):
    def check(self, out: np.ndarray, known: tuple) -> None:
        spots = [spot for spot, sample in known]
        np.testing.assert_allclose(out[spots], [sample for spot, sample in known], rtol=0, atol=1e-6)

    def test_pulse_voice_with_two_notes(self):
        out = render_sound({"voices": [{"inst": BLIP, "notes": [[0, "a4", 2], [2, "c#5", 2, 0.6]]}]}, RATE)
        self.assertEqual(len(out), 934)
        self.check(out, ((0, 0.015717), (1, 0.090894), (2, 0.130027), (3, 0.165580), (100, -0.258479),
                         (266, -0.179509), (267, -0.158515), (268, -0.102588), (269, -0.068985), (400, 0.355976),
                         (532, -0.118019), (533, -0.115238), (534, -0.108122), (600, 0.122980), (666, -0.039000),
                         (667, 0.000987), (700, 0.000449), (933, 0.0)))

    def test_chip_and_fm_voices_into_the_knee_at_11025_hz(self):
        defn = {"voices": [{"inst": HUM, "notes": [[0, "a2", 4]]},
                           {"inst": GLASS, "notes": [[0, 57, 2], [1.5, "e4", 2, 0.7]]},
                           {"inst": HISS, "notes": [[1, 87, 1, 0.5]]}], "gain": 1.6}
        out = render_sound(defn, 11025)
        self.assertEqual(len(out), 1562)
        self.check(out, ((0, 0.059311), (1, 0.135301), (2, 0.233344), (3, 0.352366), (50, -0.919455),
                         (183, -0.856838), (184, -0.963969), (185, -0.933514), (186, -0.857378), (250, -0.705120),
                         (275, -0.731616), (276, 0.882124), (277, 0.811864), (300, 0.816459), (367, -0.781211),
                         (368, -0.060194), (369, -0.382793), (500, -0.069744), (643, -0.984739), (644, -0.860699),
                         (700, 0.853307), (900, 0.804854), (1100, 0.412408), (1194, -0.005665), (1300, -0.000922),
                         (1561, 0.0)))

    def test_output_stage_on_samples_that_go_over_the_knee(self):
        samples = (3.4 * WASH[:300]).astype(np.float32)
        self.check(finish(samples.copy(), 0.8, RATE),
                   ((0, -0.988642), (1, -0.384642), (2, 0.778516), (3, -0.976111), (4, -0.052349), (50, -0.590015),
                    (150, 0.986560), (258, 0.077975), (259, 0.975600), (260, -0.773296), (261, 0.359416),
                    (280, 0.126677), (298, -0.023556), (299, 0.0)))
        self.check(finish(samples.copy(), 0.8, RATE, False),
                   ((0, -0.988454), (1, -0.366269), (2, 0.794847), (3, -0.974375), (4, -0.035244), (50, -0.584303),
                    (150, 0.986572), (258, 0.078015), (259, 0.975603), (260, -0.793090), (261, 0.378369),
                    (280, 0.266712), (298, -0.942229), (299, 0.131286)))
        self.check(finish(samples.copy(), 0.8, 48000),
                   ((0, -0.988908), (1, -0.423982), (2, 0.756309), (30, 0.988706), (60, 0.988374), (61, -0.492237),
                    (62, 0.681238), (100, 0.362952), (200, -0.307939), (298, -0.003954), (299, 0.0)))


class ErrorTest(unittest.TestCase):
    GOOD = {"inst": {}, "notes": [[0, "a4", 1]]}

    def refused(self, defn: object, rate: object = RATE) -> str:
        """What the error says when a sound is refused."""
        with self.assertRaises(RowSynthError, msg=repr((defn, rate))) as caught:
            render_sound(defn, rate)
        return str(caught.exception)

    def refused_note(self, note: object) -> str:
        """What the error says when the second note of the first voice is refused."""
        return self.refused({"voices": [{"inst": {}, "notes": [[0, "a4", 1], note]}, self.GOOD]})

    def refused_inst(self, inst: object) -> str:
        """What the error says when the instrument of the second voice is refused."""
        return self.refused({"voices": [self.GOOD, {"inst": inst, "notes": [[0, "a4", 1]]}]})

    def at(self, place: str, said: str, about: object = None) -> None:
        """Holds that an error opens with the place of what is wrong."""
        self.assertTrue(said.startswith(place + ": "), (said, about))

    def test_refuses_what_is_not_a_definition(self):
        for defn in (None, [], "sound", 3, [{"voices": []}]):
            self.at("sound", self.refused(defn), defn)

    def test_refuses_a_key_a_definition_does_not_have(self):
        for key in ("volume", "echo", "Voices", "inst", "", 3):
            said = self.refused({"voices": [], key: 1})
            self.at("sound", said, key)
            self.assertIn(repr(key), said)

    def test_refuses_a_definition_without_voices(self):
        for defn in ({}, {"gain": 0.5}):
            said = self.refused(defn)
            self.at("sound", said, defn)
            self.assertIn("'voices'", said)

    def test_refuses_voices_that_are_not_a_list(self):
        for voices in (None, {}, "voices", 3, self.GOOD):
            self.at("voices", self.refused({"voices": voices}), voices)

    def test_refuses_a_gain_that_is_not_a_number(self):
        for gain in ("0.5", None, True, [0.5], float("nan"), float("inf")):
            self.at("gain", self.refused({"voices": [self.GOOD], "gain": gain}), gain)

    def test_refuses_a_voice_that_is_not_an_instrument_and_its_notes(self):
        for voice in (None, [], "voice", 3, [{}, []]):
            self.at("voices[1]", self.refused({"voices": [self.GOOD, voice]}), voice)
        for voice, key in (({"notes": []}, "inst"), ({"inst": {}}, "notes"), ({}, "inst"),
                           ({"inst": {}, "notes": [], "gain": 1}, "gain"), ({"inst": {}, "notes": [], "": 1}, "")):
            said = self.refused({"voices": [voice]})
            self.at("voices[0]", said, voice)
            self.assertIn(repr(key), said)

    def test_refuses_an_instrument_that_is_not_a_dict(self):
        for inst in ("lead", None, ["pulse"], 3, True):
            self.at("voices[1].inst", self.refused_inst(inst), inst)

    def test_names_the_place_of_what_is_wrong_in_an_instrument(self):
        wrong = (
            ({"volume": 12}, "voices[1].inst", "'volume'"),
            ({"wave": "square"}, "voices[1].inst.wave", "'square'"),
            ({"vol": [15, "9"]}, "voices[1].inst.vol[1]", "'9'"),
            ({"vol": {"v": [15], "loop": 0.5}}, "voices[1].inst.vol.loop", "0.5"),
            ({"arp": {"v": [0, None]}}, "voices[1].inst.arp.v[1]", "None"),
            ({"duty": [0, 4]}, "voices[1].inst.duty", "4"),
            ({"vib": {"rate": 6}}, "voices[1].inst.vib", "'depth'"),
            ({"vib": {"depth": "1"}}, "voices[1].inst.vib.depth", "'1'"),
            ({"release": -1}, "voices[1].inst.release", "-1"),
            ({"short": 1}, "voices[1].inst.short", "1"),
            ({"lp": 300, "res": None}, "voices[1].inst.res", "None"),
            ({"wave": "fm", "alg": 8}, "voices[1].inst.alg", "8"),
            ({"wave": "fm", "fb": "0.4"}, "voices[1].inst.fb", "'0.4'"),
            ({"wave": "fm", "ops": [{}] * 5}, "voices[1].inst.ops", "5"),
            ({"wave": "fm", "ops": [{}, 3]}, "voices[1].inst.ops[1]", "3"),
            ({"wave": "fm", "ops": [{}, {"env": {"a": "0.1"}}]}, "voices[1].inst.ops[1].env.a", "'0.1'"),
        )
        for inst, place, what in wrong:
            said = self.refused_inst(inst)
            self.at(place, said, inst)
            self.assertIn(what, said[len(place):])

    def test_refuses_notes_that_are_not_a_list(self):
        for notes in (None, {}, "a4", 3):
            self.at("voices[1].notes", self.refused({"voices": [self.GOOD, {"inst": {}, "notes": notes}]}), notes)

    def test_refuses_a_note_that_is_not_three_or_four_values(self):
        for note in (None, "a4", 69, {}, [], [0], [0, "a4"], [0, "a4", 2, 1.0, 5]):
            self.at("voices[0].notes[1]", self.refused_note(note), note)

    def test_refuses_a_tick_that_is_not_a_time_from_0_on(self):
        for tick in ("0", None, True, [0], -1, -0.5, float("nan"), float("inf")):
            self.at("voices[0].notes[1][0]", self.refused_note([tick, "a4", 1]), tick)

    def test_refuses_a_note_that_is_neither_a_name_nor_a_number(self):
        for note in ("h4", "a", "a10", "", None, True, [69], float("nan"), float("inf")):
            said = self.refused_note([0, note, 1, 0.5])
            self.at("voices[0].notes[1][1]", said, note)
            self.assertIn(repr(note), said)

    def test_refuses_ticks_that_are_not_a_length_of_time(self):
        for ticks in ("2", None, True, [2], -1, -0.5, float("nan"), float("inf")):
            self.at("voices[0].notes[1][2]", self.refused_note([0, "a4", ticks]), ticks)

    def test_refuses_a_velocity_that_is_not_a_number(self):
        for velocity in ("1", None, True, [1], float("nan")):
            self.at("voices[0].notes[1][3]", self.refused_note([0, "a4", 1, velocity]), velocity)

    def test_refuses_a_rate_that_makes_no_sense(self):
        for rate in (0, -8000, 8000.5, "8000", None, True, float("inf")):
            self.at("rate", self.refused({"voices": [self.GOOD]}, rate), rate)

    def test_nothing_is_played_when_a_sound_is_refused(self):
        late = {"voices": [self.GOOD, {"inst": {}, "notes": [[0, "a4", 1], [1, "h4", 1]]}]}
        cases = ((late, RATE), ({"voices": [self.GOOD], "gain": None}, RATE), ({"voices": [self.GOOD]}, 0),
                 ({"voices": [self.GOOD, {"inst": {"wave": "fm", "ops": [3]}, "notes": []}]}, RATE))
        for defn, rate in cases:
            with mock.patch("rowsynth.sound.render_note") as played:
                self.refused(defn, rate)
            played.assert_not_called()

    def test_output_stage_refuses_a_buffer_that_is_not_float32_samples(self):
        for out in ([0.0] * 300, np.zeros(300), np.zeros(300, dtype=np.int16), np.zeros((2, 150), dtype=np.float32)):
            with self.assertRaises(RowSynthError) as caught:
                finish(out, 0.5, RATE)
            self.at("out", str(caught.exception))

    def test_output_stage_refuses_a_gain_a_rate_or_a_fade_of_the_wrong_kind(self):
        wrong = [("gain", (gain, RATE, True)) for gain in ("0.5", None, True, float("nan"))]
        wrong += [("rate", (0.5, rate, True)) for rate in (0, -8000, 8000.5, "8000", None, True)]
        wrong += [("fade", (0.5, RATE, fade)) for fade in (1, 0, "yes", None)]
        out = np.full(300, 0.5, dtype=np.float32)
        for name, given in wrong:
            with self.assertRaises(RowSynthError, msg=repr(given)) as caught:
                finish(out, *given)
            self.at(name, str(caught.exception), given)
        self.assertTrue(np.array_equal(out, np.full(300, 0.5, dtype=np.float32)))
