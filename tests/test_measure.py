"""Tests for rowsynth.measure."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from rowsynth import notes
from rowsynth.errors import RowSynthError
from rowsynth.measure import brightness, measure, pitch_at, spectrum
from rowsynth.project import Project
from rowsynth.sound import render_sound

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def tone(hz: float, count: int, rate: int, level: float = 1.0) -> np.ndarray:
    """`count` samples of a sine that starts at zero."""
    return level * np.sin(2 * np.pi * hz * np.arange(count) / rate)


def hiss(count: int) -> np.ndarray:
    """White noise at half of full scale, the same every time it is made."""
    state, found = 1, []
    for _ in range(count):
        state = (state * 1103515245 + 12345) % 2 ** 31
        found.append(state / 2 ** 31 - 0.5)
    return np.array(found)


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


class SpectrumTest(unittest.TestCase):
    def setUp(self):
        # At 8000 Hz with frames of 256: bin 20 for the first 1024 samples, bin 50 for the next 1024.
        self.both = np.concatenate((tone(625, 1024, 8000), tone(1562.5, 1024, 8000)))

    def test_there_are_half_as_many_bins_as_samples_in_a_frame(self):
        mags, hz_per_bin = spectrum(np.zeros(8000), size=256, rate=8000)
        self.assertEqual(mags.shape, (128,))
        self.assertEqual(hz_per_bin, 31.25)

    def test_the_defaults_are_a_frame_of_4096_at_44100(self):
        mags, hz_per_bin = spectrum(np.zeros(100))
        self.assertEqual(mags.shape, (2048,))
        self.assertEqual(hz_per_bin, 44100 / 4096)

    def test_a_full_scale_sine_reads_close_to_one_in_its_own_bin(self):
        # 1250 Hz is bin 40 of 256 at 8000 Hz. The window takes the reading down to (size - 1) / size.
        mags, _ = spectrum(tone(1250, 256, 8000), size=256, rate=8000)
        self.assertEqual(int(mags.argmax()), 40)
        self.assertAlmostEqual(mags[40], 255 / 256, places=5)

    def test_a_sine_at_half_level_reads_half(self):
        # A hundred periods in the frame: bin 100.
        mags, _ = spectrum(0.5 * np.sin(2 * np.pi * 100 * np.arange(4096) / 4096))
        self.assertEqual(int(mags.argmax()), 100)
        self.assertAlmostEqual(mags[100], 0.499878, places=6)

    def test_the_window_spreads_a_sine_into_the_bins_beside_it_and_little_further(self):
        mags, _ = spectrum(tone(1250, 256, 8000), size=256, rate=8000)
        self.assertAlmostEqual(mags[39], 0.500966, places=6)
        self.assertAlmostEqual(mags[41], 0.500966, places=6)
        self.assertLess(np.delete(mags, [38, 39, 40, 41, 42]).max(), 1e-3)

    def test_the_bins_follow_the_rate(self):
        # 3000 Hz at 48000 Hz with frames of 1024: 46.875 Hz a bin, so bin 64.
        mags, hz_per_bin = spectrum(tone(3000, 1024, 48000, 0.25), size=1024, rate=48000)
        self.assertEqual(hz_per_bin, 46.875)
        self.assertEqual(int(mags.argmax()), 64)
        self.assertAlmostEqual(mags[64], 0.249756, places=6)

    def test_the_frame_starts_at_the_time_asked(self):
        self.assertEqual(int(spectrum(self.both, 0.0, 256, 8000)[0].argmax()), 20)
        self.assertEqual(int(spectrum(self.both, 0.128, 256, 8000)[0].argmax()), 50)

    def test_a_frame_past_the_end_is_pulled_back_to_the_last_whole_one(self):
        late, _ = spectrum(self.both, 99.0, 256, 8000)
        last, _ = spectrum(self.both[-256:], 0.0, 256, 8000)
        np.testing.assert_array_equal(late, last)

    def test_a_time_before_the_start_reads_the_first_frame(self):
        early, _ = spectrum(self.both, -3.0, 256, 8000)
        first, _ = spectrum(self.both, 0.0, 256, 8000)
        np.testing.assert_array_equal(early, first)

    def test_a_buffer_shorter_than_a_frame_is_padded_with_silence(self):
        short = tone(1000, 100, 8000, 0.5)
        mags, _ = spectrum(short, size=256, rate=8000)
        padded, _ = spectrum(np.concatenate((short, np.zeros(156))), size=256, rate=8000)
        np.testing.assert_array_equal(mags, padded)
        self.assertEqual(int(mags.argmax()), 32)
        self.assertAlmostEqual(mags[32], 0.145616, places=6)

    def test_silence_has_no_magnitude_anywhere(self):
        mags, _ = spectrum(np.zeros(4096))
        self.assertEqual(mags.max(), 0.0)

    def test_a_list_is_read_like_an_array(self):
        values = tone(1250, 256, 8000)
        np.testing.assert_array_equal(
            spectrum(values.tolist(), size=256, rate=8000)[0], spectrum(values, size=256, rate=8000)[0])

    def test_a_frame_size_that_is_not_a_power_of_two_is_an_error(self):
        for size in (0, 1, 3, 1000, -256):
            with self.subTest(size=size):
                with self.assertRaisesRegex(RowSynthError, "power of two"):
                    spectrum(np.zeros(4096), size=size)

    def test_no_samples_is_an_error(self):
        with self.assertRaisesRegex(RowSynthError, "empty"):
            spectrum([])


class BrightnessTest(unittest.TestCase):
    def test_a_low_tone_has_its_centre_at_its_pitch_and_nothing_above_4k(self):
        self.assertEqual(brightness(tone(200, 11025, 44100, 0.5)), {"centroid_hz": 200, "above_4k": 0.0})

    def test_a_high_tone_has_its_centre_at_its_pitch_and_everything_above_4k(self):
        self.assertEqual(brightness(tone(8000, 11025, 44100, 0.5)), {"centroid_hz": 8000, "above_4k": 1.0})

    def test_the_line_is_drawn_at_4000_hz(self):
        self.assertEqual(brightness(tone(3900, 11025, 44100, 0.4)), {"centroid_hz": 3900, "above_4k": 0.0})
        self.assertEqual(brightness(tone(4100, 11025, 44100, 0.4)), {"centroid_hz": 4100, "above_4k": 1.0})

    def test_two_tones_of_one_level_share_the_energy_and_centre_halfway(self):
        both = tone(1000, 11025, 44100, 0.4) + tone(9000, 11025, 44100, 0.4)
        self.assertEqual(brightness(both), {"centroid_hz": 5000, "above_4k": 0.5})

    def test_a_loud_stretch_weighs_more_than_a_quiet_one(self):
        # Ten times the level is a hundred times the energy, so the quiet high tone holds about 1/101 of it.
        both = np.concatenate((tone(500, 8192, 44100, 0.5), tone(8000, 8192, 44100, 0.05)))
        self.assertEqual(brightness(both), {"centroid_hz": 576, "above_4k": 0.01})

    def test_silence_gives_zeros(self):
        self.assertEqual(brightness(np.zeros(8192)), {"centroid_hz": 0, "above_4k": 0.0})

    def test_a_sound_too_quiet_to_weigh_gives_zeros(self):
        self.assertEqual(brightness(tone(1000, 8192, 44100, 1e-5)), {"centroid_hz": 0, "above_4k": 0.0})

    def test_a_buffer_shorter_than_one_frame_gives_zeros(self):
        self.assertEqual(brightness(tone(1000, 2047, 44100)), {"centroid_hz": 0, "above_4k": 0.0})
        self.assertEqual(brightness(tone(1000, 2048, 44100)), {"centroid_hz": 1000, "above_4k": 0.0})

    def test_the_frame_size_and_the_rate_can_be_set(self):
        self.assertEqual(brightness(tone(3000, 600, 8000, 0.5), size=256, rate=8000),
                         {"centroid_hz": 3000, "above_4k": 0.0})
        self.assertEqual(brightness(tone(200, 12000, 48000, 0.5), rate=48000), {"centroid_hz": 200, "above_4k": 0.0})
        self.assertEqual(brightness(tone(8000, 12000, 48000, 0.5), rate=48000), {"centroid_hz": 8000, "above_4k": 1.0})

    def test_each_frame_is_taken_from_where_spectrum_puts_its_time(self):
        # A sawtooth of 294 samples a period. Some frames start a sample before their count, which is
        # where spectrum() puts their time. Had each started on its count, the centre would read 1244.
        saw = (np.arange(13230) % 294) / 147 - 1
        self.assertEqual(brightness(saw, size=64), {"centroid_hz": 1243, "above_4k": 0.05})

    def test_the_centre_is_a_whole_number_of_hz(self):
        self.assertIs(type(brightness(tone(200, 4096, 44100))["centroid_hz"]), int)
        self.assertIs(type(brightness(np.zeros(4096))["centroid_hz"]), int)

    def test_a_list_is_read_like_an_array(self):
        values = tone(1000, 600, 8000, 0.5)
        self.assertEqual(brightness(values.tolist(), 256, 8000), brightness(values, 256, 8000))

    def test_a_frame_size_that_is_not_a_power_of_two_is_an_error(self):
        for size in (0, 1, 3, 1000, -256):
            with self.subTest(size=size):
                with self.assertRaisesRegex(RowSynthError, "power of two"):
                    brightness(np.zeros(100), size=size)

    def test_no_samples_is_an_error(self):
        with self.assertRaisesRegex(RowSynthError, "empty"):
            brightness([])


class PitchAtTest(unittest.TestCase):
    def test_a_sine_reads_as_its_own_note_across_the_range(self):
        for rate in (44100, 48000):
            for hz in (55, 110, 220, 440, 880, 1760, 3520):
                with self.subTest(rate=rate, hz=hz):
                    got = pitch_at(tone(hz, rate // 5, rate), 0.0, rate=rate)
                    self.assertAlmostEqual(got, hz, delta=hz * 0.01)

    def test_a_reading_has_one_decimal_place(self):
        self.assertEqual(pitch_at(tone(55, 8820, 44100), 0.0), 54.9)
        self.assertEqual(pitch_at(tone(440, 8820, 44100), 0.0), 439.5)
        self.assertEqual(pitch_at(tone(3520, 8820, 44100), 0.0), 3519.7)
        self.assertEqual(pitch_at(tone(3520, 9600, 48000), 0.0, rate=48000), 3519.2)

    def test_clicks_read_as_the_gap_between_them_and_two_alone_as_no_note(self):
        # The only lag at which anything lines up is 50 samples, and nothing either side pulls it off 50.
        clicks = np.zeros(400)
        clicks[::50] = 0.75
        self.assertEqual(pitch_at(clicks, 0.0), 882.0)
        self.assertEqual(pitch_at(clicks, 0.0, rate=8000), 160.0)
        # Two clicks are a gap and no wave that repeats: moved by the gap, one of the two meets nothing.
        clicks[100:] = 0.0
        self.assertEqual(pitch_at(clicks, 0.0), 0.0)

    def test_noise_reads_as_no_note(self):
        white = hiss(8820)
        for t in (0.0, 0.05, 0.1, 0.15):
            self.assertEqual(pitch_at(white, t), 0.0, t)
        # The engine's noise holds each value for a while, the longer the lower its note, and repeats no more.
        for pitch in ("c4", "c5", "c6", "c7", "c8"):
            noise = render_sound({"voices": [{"inst": {"wave": "noise"}, "notes": [[0, pitch, 30]]}]})
            for t in (0.05, 0.2, 0.35):
                self.assertEqual(pitch_at(noise, t), 0.0, (pitch, t))

    def test_each_note_of_a_fast_run_with_an_echo_reads_as_itself_in_its_middle(self):
        # Six notes on one voice, the first five 4 ticks each, with the echo of the notes before under each.
        defn = Project.load(EXAMPLES / "arcade.json").sound("powerup")["def"]
        (voice,) = defn["voices"]
        self.assertEqual((len(voice["notes"]), defn["echo"]["time"]), (6, 0.09))
        samples = render_sound(defn)
        for tick, note, ticks in voice["notes"]:
            with self.subTest(note=note):
                got = pitch_at(samples, (tick + ticks / 2) / 60)
                self.assertGreater(got, 0.0)
                # A quarter tone is half a semitone.
                self.assertLess(abs(12 * math.log2(got / notes.hz(notes.midi(note)))), 0.5)
        # At 0.15 s the third note has the end of the second and the echo of the first two under it.
        self.assertEqual(pitch_at(samples, 0.15), 783.0)

    def test_a_wave_with_overtones_reads_as_its_lowest_note(self):
        # A square wave of 200 samples a period: 220.5 Hz at 44100 Hz and 240 Hz at 48000 Hz.
        square = np.where((np.arange(8820) // 100) % 2 == 0, 0.5, -0.5)
        self.assertEqual(pitch_at(square, 0.0), 220.5)
        self.assertEqual(pitch_at(square, 0.0, rate=48000), 240.0)

    def test_overtones_alone_read_as_the_note_they_belong_to(self):
        overtones = sum(tone(200 * k, 8820, 44100, 0.2) for k in (2, 3, 4))
        self.assertEqual(pitch_at(overtones, 0.0), 199.9)

    def test_the_time_picks_the_stretch_that_is_read(self):
        both = np.concatenate((tone(220, 8820, 44100), tone(660, 8820, 44100)))
        self.assertEqual(pitch_at(both, 0.1), 219.9)
        self.assertEqual(pitch_at(both, 0.3), 659.6)

    def test_a_time_before_the_start_reads_from_the_start(self):
        values = tone(2000, 8820, 44100)
        self.assertEqual(pitch_at(values, 0.0), 1999.4)
        self.assertEqual(pitch_at(values, -1.0), 1999.4)

    def test_fewer_than_64_samples_read_as_no_note(self):
        values = tone(2000, 8820, 44100)
        self.assertEqual(pitch_at(values[:63], 0.0), 0.0)
        self.assertEqual(pitch_at(values[:64], 0.0), 1986.0)

    def test_a_time_too_near_the_end_or_past_it_reads_as_no_note(self):
        # 800 samples at 8000 Hz: half of them are left at 0.05 s, 60 at 0.0925 s and none from 0.1 s on.
        values = tone(250, 800, 8000)
        self.assertEqual(pitch_at(values, 0.05, rate=8000), 249.3)
        self.assertEqual(pitch_at(values, 0.0925, rate=8000), 0.0)
        self.assertEqual(pitch_at(values, 0.1, rate=8000), 0.0)
        self.assertEqual(pitch_at(values, 5.0, rate=8000), 0.0)

    def test_silence_reads_as_no_note(self):
        self.assertEqual(pitch_at(np.zeros(4410), 0.0), 0.0)

    def test_next_to_no_sound_reads_as_no_note(self):
        # The squares over the stretch of 1764 samples have to reach a millionth in all. A sine of level
        # 3e-5 comes to 0.8 millionths and one of 4e-5 to 1.4.
        self.assertEqual(pitch_at(tone(440, 4410, 44100, 3e-5), 0.0), 0.0)
        self.assertEqual(pitch_at(tone(440, 4410, 44100, 4e-5), 0.0), 439.5)

    def test_lo_is_the_lowest_note_looked_for(self):
        values = tone(100, 8820, 44100)
        self.assertEqual(pitch_at(values, 0.0), 99.2)
        self.assertEqual(pitch_at(values, 0.0, lo=150), 0.0)

    def test_hi_is_the_highest_note_looked_for(self):
        # One period of 5000 Hz is a shorter lag than any tried for 4000 Hz. Two periods are found instead.
        values = tone(5000, 4410, 44100)
        self.assertEqual(pitch_at(values, 0.0), 2498.8)
        self.assertEqual(pitch_at(values, 0.0, hi=6000), 4995.7)

    def test_window_is_how_long_a_stretch_is_read(self):
        # One period of 25 Hz is 1764 samples, the whole of the usual stretch, so it takes a longer one.
        values = tone(25, 22050, 44100)
        self.assertEqual(pitch_at(values, 0.0, lo=20), 0.0)
        self.assertEqual(pitch_at(values, 0.0, lo=20, window=0.1), 24.6)

    def test_a_window_of_fewer_than_64_samples_reads_as_no_note(self):
        # 0.001 s at 44100 Hz is 44 samples.
        self.assertEqual(pitch_at(tone(440, 4410, 44100), 0.0, window=0.001), 0.0)

    def test_a_list_is_read_like_an_array(self):
        values = tone(250, 800, 8000)
        self.assertEqual(pitch_at(values.tolist(), 0.0, rate=8000), 249.3)
        self.assertEqual(pitch_at(values.astype(np.float32), 0.0, rate=8000), 249.3)

    def test_no_samples_is_an_error(self):
        with self.assertRaisesRegex(RowSynthError, "empty"):
            pitch_at([], 0.0)

    def test_more_than_one_row_is_an_error(self):
        with self.assertRaisesRegex(RowSynthError, "one row"):
            pitch_at(np.zeros((2, 4410)), 0.0)


if __name__ == "__main__":
    unittest.main()
