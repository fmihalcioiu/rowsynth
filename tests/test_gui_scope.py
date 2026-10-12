"""Tests for rowsynth.gui.scope: the stretch a scope shows, the bars of a spectrum, their help and what they draw."""
from __future__ import annotations

import math
import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from rowsynth.bank import render_entry
from rowsynth.checks import sounds
from rowsynth.errors import RowSynthError
from rowsynth.gui import layout, sprites, theme
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.font import Font
from rowsynth.gui.model import TEST_NOTE, TEST_TICKS
from rowsynth.gui.roll import mix
from rowsynth.gui.scope import (CYCLES, FLOOR_DB, FRAME, HARSH, HARSH_HZ, INTO, NO_SOUND, STRETCH, TINT, Scope,
                                Spectrum, column, cycles_told, heard, heights, hz_at, levels, place_told, span,
                                summed)
from rowsynth.gui.ui import Event, Ui
from rowsynth.measure import brightness, pitch_at
from rowsynth.notes import midi
from rowsynth.project import Project
from rowsynth.sound import render_sound

# pygame reads these when it is imported: no test opens a window or plays a sound.
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
try:
    import pygame
except ImportError:
    pygame = None

EXAMPLES = Path(__file__).resolve().parent / "data"
NEEDS = "pygame is not installed"

RATE = 44100
#: A frequency that is on a bin of a frame at that rate, a little over 1 kHz: its bar is in column 128 of 258.
TONE = 48 * RATE / FRAME
#: A scope whose wave has the columns from x 11 to x 208. Zero is at y 55, and full scale 31 rows from it.
SCOPE = (10, 20, 200, 70)
#: A spectrum whose bars are in (11, 21, 258, 59), over a strip from y 81 in which the marks are named. At 44100 Hz
#: the harsh line is at x 196, and its label and the words under it are in the two rectangles.
SPECTRUM = (10, 20, 260, 70)
BARS = (11, 21, 258, 59)
LABEL = (199, 22, 47, 9)
WORDS = (199, 31, 47, 9)
#: The surface they are drawn on, which is larger on every side, and the colour of what is not theirs.
SIZE = (280, 100)
OUTSIDE = theme.GOOD
#: The ground right of the harsh line.
TINTED = mix(theme.BACKGROUND, theme.FAULT, TINT)


def sine(freq: float, rate: int = RATE, level: float = 0.6) -> np.ndarray:
    """A quarter of a second of a sine that starts at zero, going up."""
    return (level * np.sin(2 * np.pi * freq * np.arange(rate // 4) / rate)).astype(np.float32)


def hiss(count: int = RATE // 4) -> np.ndarray:
    """Noise at half of full scale, the same every time it is made."""
    state, found = 1, []
    for _ in range(count):
        state = (state * 1103515245 + 12345) % 2 ** 31
        found.append(state / 2 ** 31 - 0.5)
    return np.array(found, dtype=np.float32)


def played(wave: str, pitch: str) -> np.ndarray:
    """One short note of a plain instrument of a wave, as the engine renders it."""
    return render_sound({"voices": [{"inst": {"wave": wave}, "notes": [[0, pitch, 6]]}]}, RATE)


def runs(flags: list[bool]) -> list[int]:
    """How long each unbroken run of true flags is."""
    found, run = [], 0
    for flag in [*flags, False]:
        if flag:
            run += 1
        elif run:
            found.append(run)
            run = 0
    return found


def cells(*rects: layout.Rect) -> set[tuple[int, int]]:
    """Every pixel of some rectangles."""
    return {(x + column, y + row) for x, y, w, h in rects for column in range(w) for row in range(h)}


class Stretch(unittest.TestCase):
    def test_the_stretch_of_a_sine_is_three_cycles_from_where_it_rises_through_zero_at_two_rates(self) -> None:
        for rate in (44100, 22050):
            wave = sine(440.0, rate)
            first, last = span(wave, rate, 69)
            self.assertEqual(last - first, math.floor(CYCLES * rate / 440))
            self.assertTrue(wave[first - 1] < 0 <= wave[first])
            # It is the first such place after a tenth of the sound, so no more than a cycle further on.
            self.assertTrue(0 < first - int(INTO * len(wave)) <= rate / 440 + 1)
            self.assertEqual(span(wave, rate, "a4"), (first, last))
        self.assertEqual(span(sine(440.0), RATE, 69), (1103, 1403))

    def test_with_no_pitch_given_the_note_is_read_from_the_sound(self) -> None:
        for freq in (55.0, 440.0, 1760.0):
            self.assertAlmostEqual(heard(sine(freq), RATE), 69 + 12 * math.log2(freq / 440), delta=0.1)
        wave = sine(440.0)
        # It is what `measure.pitch_at` reads a tenth of the way in, as a note number.
        self.assertEqual(heard(wave, RATE), 69 + 12 * math.log2(pitch_at(wave, int(INTO * len(wave)) / RATE) / 440))
        first, last = span(wave, RATE)
        self.assertEqual(first, 1103)
        self.assertAlmostEqual(last - first, CYCLES * RATE / 440, delta=1.5)

    def test_noise_has_no_note_and_is_shown_for_a_fixed_stretch_as_it_is(self) -> None:
        noise = hiss()
        self.assertIsNone(heard(noise, RATE))
        self.assertEqual(span(noise, RATE), (1102, 1102 + round(STRETCH * RATE)))
        # The noise of the engine holds each value for a while, the longer the lower its note: it has none either.
        for pitch in ("c4", "c6", "c8"):
            self.assertIsNone(heard(played("noise", pitch), RATE))
        self.assertIsNone(heard(np.zeros(0, dtype=np.float32), RATE))
        # With the pitch of its note given, noise is cut as a note is: from where it rises through zero.
        first, last = span(noise, RATE, 69)
        self.assertEqual(last - first, 300)
        self.assertTrue(noise[first - 1] < 0 <= noise[first])

    def test_a_sound_shorter_than_its_stretch_is_shown_whole_and_one_that_ends_soon_from_earlier(self) -> None:
        wave = sine(440.0)
        self.assertEqual((span(wave[:100], RATE, 69), span(hiss(300), RATE)), ((0, 100), (0, 300)))
        self.assertEqual(span(np.zeros(0, dtype=np.float32), RATE), (0, 0))
        # A tenth into 320 samples is sample 32, and three cycles from there would end past the end.
        self.assertEqual(span(wave[:320], RATE, 69), (20, 320))
        # A wave that never rises through zero is shown from a tenth of the way in.
        self.assertEqual(span(np.abs(wave) + 0.1, RATE, 69), (1102, 1402))

    def test_the_stretch_in_plain_words(self) -> None:
        self.assertEqual(cycles_told(1103, 1403, RATE, 69), "3 cycles of a4, 6.8 ms")
        self.assertEqual(cycles_told(0, 441, RATE, None), "no note, 10.0 ms")
        # A sound too short for three cycles has as many as there are, and a note read from it has the nearest name.
        self.assertEqual(cycles_told(0, 100, RATE, 69.0), "1 cycle of a4, 2.3 ms")
        self.assertEqual(cycles_told(0, 150, RATE, 68.98), "1.5 cycles of a4, 3.4 ms")
        self.assertEqual(cycles_told(0, 2022, RATE, midi("c2")), "3 cycles of c2, 45.9 ms")

    def test_what_is_no_row_of_samples_no_rate_or_no_note_is_refused_and_then_nothing_changes(self) -> None:
        wave = sine(440.0)
        wrong = ((np.zeros((2, 8), dtype=np.float32), RATE, 69), ([0.0, 0.5], RATE, 69), (wave, 0, 69),
                 (wave, 44100.5, 69), (wave, RATE, "h9"), (wave, RATE, True))
        scope, spectrum = Scope(SCOPE), Spectrum(SPECTRUM)
        scope.show(wave, RATE, 69)
        spectrum.show(sine(2 * HARSH_HZ), RATE)
        for samples, rate, pitch in wrong:
            with self.assertRaises(RowSynthError, msg=repr((rate, pitch))):
                span(samples, rate, pitch)
            with self.assertRaises(RowSynthError, msg=repr((rate, pitch))):
                scope.show(samples, rate, pitch)
        for samples, rate, _ in wrong[:4]:
            with self.assertRaises(RowSynthError, msg=repr(rate)):
                spectrum.show(samples, rate)
            with self.assertRaises(RowSynthError, msg=repr(rate)):
                summed(samples, rate)
        # A spectrum starts at 50 Hz, so a rate that holds nothing above that has none.
        with self.assertRaises(RowSynthError) as refused:
            spectrum.show(wave, 100)
        self.assertEqual(str(refused.exception),
                         "rate: a spectrum starts at 50 Hz, and at 100 Hz there is no sound above that")
        self.assertEqual((scope.help, spectrum.above), ("3 cycles of a4, 6.8 ms", 1.0))


class Bars(unittest.TestCase):
    def test_the_scale_is_even_in_octaves_from_50_hz_to_half_the_rate(self) -> None:
        self.assertEqual((column(HARSH_HZ, 258, 44100), column(HARSH_HZ, 258, 22050)), (185, 209))
        self.assertEqual([column(freq, 258, 44100) for freq in (50, 100, 1000, 10000, 22049)], [0, 29, 126, 224, 257])
        self.assertEqual([column(freq, 258, 20000) for freq in (50, 100, 1000, 4000, 9999)], [0, 33, 145, 213, 257])
        # An octave is as many columns wide wherever it lies, and the frequency of a column is in that column.
        for freq in (60, 440, 9000):
            self.assertAlmostEqual(column(2 * freq, 2580, 44100) - column(freq, 2580, 44100),
                                   2580 / math.log2(441), delta=1)
        for rate in (44100, 22050):
            self.assertEqual([column(hz_at(at + 0.5, 258, rate), 258, rate) for at in range(258)], list(range(258)))
            self.assertAlmostEqual(hz_at(0, 258, rate), 50)
            self.assertAlmostEqual(hz_at(258, 258, rate), rate / 2)

    def test_a_sine_is_one_peak_in_the_column_of_its_frequency_and_the_rest_is_far_below(self) -> None:
        for rate in (44100, 22050):
            freq = 48 * rate / FRAME
            tall = heights(*summed(sine(freq, rate), rate), 258, 59)
            peak = column(freq, 258, rate)
            self.assertEqual((tall[peak], tall.count(59), tall.index(59)), (59, 1, peak))
            # The bin on each side of it is 6 dB under it, in the column on each side, and that is all of it.
            self.assertEqual(tall[peak - 2:peak + 3], [0, 54, 59, 54, 0])
            self.assertEqual(sum(tall), 59 + 2 * 54)
        self.assertEqual(column(TONE, 258, RATE), 128)

    def test_two_sines_an_octave_apart_are_two_peaks_an_octave_of_columns_apart(self) -> None:
        tall = heights(*summed(sine(TONE) + sine(2 * TONE), RATE), 258, 59)
        peaks = [at for at, high in enumerate(tall) if high == 59]
        self.assertEqual(peaks, [column(TONE, 258, RATE), column(2 * TONE, 258, RATE)])
        # From 50 Hz to 22050 Hz there are 8.78 octaves, which is 29.4 columns to each.
        self.assertEqual(peaks[1] - peaks[0], 29)

    def test_silence_and_no_samples_have_no_height(self) -> None:
        for samples in (np.zeros(RATE // 4, dtype=np.float32), np.zeros(0, dtype=np.float32)):
            mags, per_bin = summed(samples, RATE)
            self.assertEqual((len(mags), bool(mags.any()), per_bin), (FRAME // 2, False, RATE / FRAME))
            self.assertEqual(heights(mags, per_bin, 258, 59), [0] * 258)
        self.assertEqual(levels(np.zeros(8), 50.0, 4), [FLOOR_DB] * 4)
        self.assertEqual((heights(np.ones(8), 50.0, 0, 59), levels(np.ones(8), 50.0, -3)), ([], []))

    def test_a_column_has_its_loudest_bin_and_one_with_no_bin_is_on_the_line_between_its_neighbours(self) -> None:
        # Eight bins 50 Hz apart, under a top of 400 Hz: three octaves. In three columns the first has the bin at
        # 50 Hz, the second the two from 100 Hz and the third the four from 200 Hz. The bin at 0 Hz is on none.
        found = levels(np.array([9, 1, 0.5, 0.1, 0.01, 0.04, 0.03, 0.02]), 50.0, 3)
        for level, db in zip(found, (0, 20 * math.log10(0.5), 20 * math.log10(0.04))):
            self.assertAlmostEqual(level, db)
        # In thirty columns the bins at 50 and at 100 Hz are ten apart, and the columns between are on their line.
        mags = np.array([0, 1, 0.1, 0, 0, 0, 0, 0])
        for level, db in zip(levels(mags, 50.0, 30)[:12], [-2.0 * at for at in range(11)] + [-30.4]):
            self.assertAlmostEqual(level, db)
        self.assertEqual(heights(mags, 50.0, 30, 36)[:11], list(range(36, 25, -1)))
        # Nothing is under the floor, and a bar there has no height.
        self.assertEqual((levels(mags, 50.0, 30)[15:], heights(mags, 50.0, 30, 36)[15:]), ([FLOOR_DB] * 15, [0] * 15))

    def test_a_whole_sound_is_framed_as_its_brightness_is(self) -> None:
        # A dull tone two frames long, and a short bright one in the middle of it. There are three frames, half a
        # frame apart: the bright tone is in the middle of the second, and at the ends of the other two.
        sound = sine(TONE, level=0.5)[:2 * FRAME].copy()
        sound[FRAME - 256:FRAME + 256] += sine(8 * TONE, level=0.5)[:512]
        mags, per_bin = summed(sound, RATE)
        freqs = np.arange(len(mags)) * per_bin
        power = mags[1:] ** 2
        above = brightness(sound, FRAME, RATE)["above_4k"]
        self.assertAlmostEqual(float(power[freqs[1:] > HARSH_HZ].sum() / power.sum()), above, delta=0.0051)
        self.assertEqual(above, 0.17)
        # A sound shorter than a frame is one frame, though its brightness reads nothing of it.
        short = sine(TONE)[:FRAME // 2]
        self.assertEqual((int(np.argmax(summed(short, RATE)[0])), brightness(short, FRAME, RATE)["above_4k"]),
                         (48, 0.0))
        # A frame with next to no energy counts for nothing, so a sound of such frames alone is silence.
        self.assertFalse(summed(sine(TONE, level=1e-5), RATE)[0].any())

    def test_a_place_in_plain_words(self) -> None:
        self.assertEqual(place_told(880, -12.2), "880 Hz, a5, -12 dB")
        self.assertEqual(place_told(440.4, 0.0), "440 Hz, a4, 0 dB")
        self.assertEqual(place_told(10000, FLOOR_DB), "10000 Hz, e-9, under -72 dB")


class Handled(unittest.TestCase):
    def setUp(self) -> None:
        self.ui = Ui()

    def test_the_help_line_of_a_scope_says_how_long_the_stretch_is_and_of_what_note(self) -> None:
        shown = self.ui.add(Scope(SCOPE, help="the shape of the wave"))
        self.ui.handle(Event("move", (50, 50)))
        self.assertEqual(self.ui.help, "the shape of the wave")
        said = []
        for samples, pitch in ((sine(440.0), 69), (sine(440.0), None), (hiss(), None), (sine(440.0)[:100], "a4"),
                               (np.zeros(800, dtype=np.float32), 69), (np.zeros(0, dtype=np.float32), None)):
            shown.show(samples, RATE, pitch)
            said.append(self.ui.help)
        self.assertEqual(said, ["3 cycles of a4, 6.8 ms", "3 cycles of a4, 6.8 ms", "no note, 10.0 ms",
                                "1 cycle of a4, 2.3 ms", "the shape of the wave", "the shape of the wave"])
        # It takes nothing from the mouse.
        self.assertFalse(self.ui.handle(Event("down", (50, 50), button=1)))

    def test_the_help_line_of_a_spectrum_says_the_frequency_the_note_and_the_level_under_the_mouse(self) -> None:
        shown = self.ui.add(Spectrum(SPECTRUM, help="where the energy sits"))
        self.ui.handle(Event("move", (139, 40)))
        self.assertEqual(self.ui.help, "where the energy sits")
        shown.show(sine(TONE), RATE)
        said = []
        # On the peak and beside it, far from it, on the outline, and on the strip of the marks.
        for place in ((139, 40), (139, 79), (138, 21), (71, 40), (10, 40), (139, 85)):
            self.assertFalse(self.ui.handle(Event("move", place)))
            said.append(self.ui.help)
        self.assertEqual(said, ["1038 Hz, c6, 0 dB", "1038 Hz, c6, 0 dB", "1013 Hz, b5, -6 dB",
                                "208 Hz, a-3, under -72 dB", "where the energy sits", "where the energy sits"])
        self.ui.handle(Event("move", (139, 40)))
        shown.show(np.zeros(RATE // 4, dtype=np.float32), RATE)
        self.assertEqual(self.ui.help, "where the energy sits")


@unittest.skipUnless(pygame, NEEDS)
class Drawn(unittest.TestCase):
    def setUp(self) -> None:
        self.surface = pygame.Surface(SIZE)
        self.canvas = Canvas(self.surface, Font(sprites.load("font")))
        self.ui = Ui()

    def scope(self, rect: layout.Rect = SCOPE) -> Scope:
        """A scope on the Ui, in place of what was there."""
        self.ui.widgets.clear()
        return self.ui.add(Scope(rect, theme.SOUND))

    def spectrum(self, rect: layout.Rect = SPECTRUM) -> Spectrum:
        """A spectrum on the Ui, in place of what was there."""
        self.ui.widgets.clear()
        return self.ui.add(Spectrum(rect, theme.SOUND))

    def draw(self) -> None:
        """Draw the widgets on a surface of the colour that is not theirs."""
        self.surface.fill(OUTSIDE)
        self.ui.draw(self.canvas)

    def at(self, *pos: int) -> tuple[int, ...]:
        """The colour of one pixel."""
        return tuple(self.surface.get_at(pos))[:3]

    def ink(self, rect: layout.Rect, colour: theme.Colour) -> set[tuple[int, int]]:
        """The pixels of a rectangle that are of a colour."""
        return {pos for pos in cells(rect) if self.at(*pos) == colour}

    def written(self, text: str, rect: layout.Rect, align: str = "left") -> set[tuple[int, int]]:
        """The pixels a line of text takes in a rectangle."""
        page = pygame.Surface(SIZE)
        Canvas(page, self.canvas.font).text_in(text, rect, theme.DIM, align)
        return {pos for pos in cells(rect) if page.get_at(pos)[:3] == theme.DIM}

    def wave(self) -> list[list[int]]:
        """The rows the wave of the scope is on in each of its columns, from the left, the highest first."""
        inked = self.ink(SCOPE, theme.SOUND)
        return [sorted(y for x, y in inked if x == column) for column in range(11, 209)]

    def test_the_scope_of_a_sine_rises_through_zero_at_its_left_edge_and_has_three_crests(self) -> None:
        self.scope().show(sine(440.0), RATE, 69)
        self.draw()
        columns = self.wave()
        tops, feet = [rows[0] for rows in columns], [rows[-1] for rows in columns]
        # It starts a row over the line at zero and climbs from there. A crest is 19 rows over the line.
        self.assertEqual((columns[0], tops[:12]), ([54], sorted(tops[:12], reverse=True)))
        self.assertEqual((min(tops), max(feet)), (36, 74))
        self.assertEqual((len(runs([top == 36 for top in tops])), len(runs([foot == 74 for foot in feet]))), (3, 3))
        # The columns meet, so it is one line from side to side.
        self.assertTrue(all(before[0] - 1 <= rows[-1] and rows[0] <= before[-1] + 1
                            for before, rows in zip(columns, columns[1:])))
        # Under it are the line at zero and fainter ones at full scale, in a dark box with an outline.
        self.assertEqual(self.ink((11, 55, 198, 1), theme.LINE) | self.ink((11, 55, 198, 1), theme.SOUND),
                         cells((11, 55, 198, 1)))
        self.assertEqual(self.ink(SCOPE, theme.RAISED), cells((11, 24, 198, 1), (11, 86, 198, 1)))
        self.assertEqual([self.at(*pos) for pos in ((10, 20), (209, 89), (11, 21), (208, 88))],
                         [theme.LINE, theme.LINE, theme.BACKGROUND, theme.BACKGROUND])

    def test_the_scope_of_a_square_has_flat_tops_and_what_is_past_full_scale_is_drawn_at_full_scale(self) -> None:
        square = np.where(sine(440.0) >= 0, 0.6, -0.6).astype(np.float32)
        shown = self.scope()
        shown.show(square, RATE, 69)
        self.draw()
        columns = self.wave()
        # Three cycles: three stretches of one pixel a column at the top, and three at the foot.
        for row in (36, 74):
            flat = runs([rows == [row] for rows in columns])
            self.assertEqual((len(flat), min(flat) >= 30), (3, True), flat)
        self.assertEqual(columns[0], [36])
        shown.show(2.5 * square, RATE, 69)
        self.draw()
        columns = self.wave()
        self.assertEqual((min(rows[0] for rows in columns), max(rows[-1] for rows in columns)), (24, 86))
        self.assertEqual(len(runs([rows == [24] for rows in columns])), 3)

    def test_the_spectrum_of_a_sine_has_its_peak_column_the_harsh_line_and_the_tinted_ground(self) -> None:
        self.spectrum().show(sine(TONE), RATE)
        self.draw()
        # The bar of the peak is as high as the bars go, and the one on each side of it 54 rows.
        self.assertEqual(self.ink(BARS, theme.SOUND), cells((138, 26, 1, 54), (139, 21, 1, 59), (140, 26, 1, 54)))
        # The harsh line stands at 4 kHz from top to foot, and all the ground right of it is tinted.
        self.assertEqual(self.ink(SPECTRUM, theme.FAULT), cells((196, 21, 1, 59)))
        right = cells((197, 21, 72, 59))
        self.assertEqual(self.ink(BARS, TINTED) - right, set())
        # But for the line of the mark at 10 kHz, and the label and the words with their dark edge.
        self.assertTrue(right - self.ink(BARS, TINTED) <= cells((235, 21, 1, 59), (198, 22, 49, 19)))
        self.assertEqual((self.at(197, 21), self.at(268, 79), self.at(195, 21)),
                         (TINTED, TINTED, theme.BACKGROUND))
        self.assertEqual(self.ink(BARS, theme.DIM), self.written(HARSH, LABEL))
        self.assertEqual(self.ink(BARS, theme.TEXT), self.written("0% above", WORDS))
        self.assertEqual([self.at(*pos) for pos in ((10, 20), (269, 80), (11, 21), (40, 21), (137, 21), (235, 21))],
                         [theme.LINE, theme.LINE, theme.BACKGROUND, theme.RAISED, theme.RAISED, theme.RAISED])

    def test_the_words_are_plain_for_a_dull_sound_and_in_the_fault_colour_for_a_bright_one(self) -> None:
        shown = self.spectrum()
        dull, bright = played("tri", "c2"), played("noise", "c8")
        shown.show(dull, RATE)
        self.draw()
        self.assertEqual((shown.above, brightness(dull, rate=RATE)["above_4k"]), (0.0, 0.0))
        self.assertEqual(self.ink(BARS, theme.TEXT), self.written("0% above", WORDS))
        self.assertEqual(self.ink(BARS, theme.FAULT), cells((196, 21, 1, 59)))
        shown.show(bright, RATE)
        self.draw()
        self.assertEqual(shown.above, brightness(bright, rate=RATE)["above_4k"])
        self.assertGreater(shown.above, sounds._BRIGHT)
        words = f"{round(shown.above * 100)}% above"
        self.assertEqual(self.ink(BARS, theme.FAULT),
                         cells((196, 21, 1, 59)) | self.written(words, (199, 31, Font.width(words), 9)))
        # The label and the words are read over the bars: each of their pixels has a dark one on every side.
        self.assertEqual((self.ink(BARS, theme.TEXT), self.ink(BARS, theme.DIM)), (set(), self.written(HARSH, LABEL)))
        self.assertEqual((self.at(199, 22), self.at(198, 30), self.at(205, 45)),
                         (theme.BACKGROUND, theme.BACKGROUND, theme.SOUND))
        # The limit is the one the checks hold a sound to.
        with mock.patch.object(sounds, "_BRIGHT", shown.above):
            self.draw()
        self.assertEqual((self.ink(BARS, theme.FAULT), self.ink(BARS, theme.TEXT)),
                         (cells((196, 21, 1, 59)), self.written(words, (199, 31, Font.width(words), 9))))

    def test_the_marks_under_half_the_rate_are_named_under_the_box_each_centred_on_its_line(self) -> None:
        shown = self.spectrum()
        shown.show(sine(TONE), RATE)
        self.draw()
        strip = (10, 81, 260, 9)
        named = (self.written("100", (32, 81, 17, 9)) | self.written("1k", (132, 81, 11, 9))
                 | self.written("10k", (227, 81, 17, 9)))
        self.assertEqual((self.ink(strip, theme.DIM), len(self.ink(strip, OUTSIDE))), (named, 260 * 9 - len(named)))
        # At 20000 Hz the scale ends at 10 kHz, so that is no mark on it. The harsh line is nearer the right
        # edge, where the label and the words have no room after it and stand before it.
        shown.show(sine(625.0, 20000), 20000)
        self.draw()
        self.assertEqual(self.ink(strip, theme.DIM),
                         self.written("100", (36, 81, 17, 9)) | self.written("1k", (151, 81, 11, 9)))
        self.assertEqual((self.ink(SPECTRUM, theme.FAULT), self.ink(BARS, theme.RAISED)),
                         (cells((224, 21, 1, 59)), cells((44, 21, 1, 59), (156, 21, 1, 59))))
        self.assertEqual(self.ink(BARS, theme.DIM), self.written(HARSH, (175, 22, 47, 9), "right"))
        self.assertEqual(self.ink(BARS, theme.TEXT), self.written("0% above", (175, 31, 47, 9), "right"))

    def test_with_no_sound_there_is_the_box_and_a_dim_word(self) -> None:
        nothing = (np.zeros(RATE // 4, dtype=np.float32), np.zeros(0, dtype=np.float32))
        shown = self.scope()
        for samples in (None, *nothing):
            if samples is not None:
                shown.show(sine(440.0), RATE, 69)
                shown.show(samples, RATE, 69)
            self.draw()
            self.assertEqual(self.ink(SCOPE, theme.DIM), self.written(NO_SOUND, SCOPE, "centre"))
            self.assertEqual(self.ink(SCOPE, theme.LINE), cells(SCOPE) - cells((11, 21, 198, 68)))
            self.assertEqual(len(self.ink(SCOPE, theme.BACKGROUND)), 198 * 68 - len(self.written(NO_SOUND, SCOPE)))
        shown, bright = self.spectrum(), played("noise", "c8")
        for samples in (None, *nothing):
            if samples is not None:
                shown.show(bright, RATE)
                shown.show(samples, RATE)
            self.draw()
            self.assertEqual(self.ink(SPECTRUM, theme.DIM), self.written(NO_SOUND, BARS, "centre"))
            self.assertEqual(self.ink(SPECTRUM, theme.LINE), cells((10, 20, 260, 61)) - cells(BARS))
            self.assertEqual(len(self.ink(SPECTRUM, theme.BACKGROUND)), 258 * 59 - len(self.written(NO_SOUND, BARS)))
            # The strip under the box is left as it was.
            self.assertEqual(self.ink((10, 81, 260, 9), OUTSIDE), cells((10, 81, 260, 9)))

    def test_nothing_is_drawn_outside_the_rectangle(self) -> None:
        loud, silent = 2.5 * played("noise", "c8"), np.zeros(0, dtype=np.float32)
        for rect in (SCOPE, SPECTRUM, (10, 20, 30, 12), (10, 20, 3, 40), (10, 20, 0, 0)):
            around = cells((0, 0, *SIZE)) - cells(rect)
            for samples in (loud, silent):
                self.scope(rect).show(samples, RATE)
                self.ui.add(Spectrum(rect)).show(samples, RATE)
                self.draw()
                self.assertTrue(all(self.at(*pos) == OUTSIDE for pos in around), rect)

    def test_a_test_note_of_every_instrument_of_the_examples_shows_in_both(self) -> None:
        seen = 0
        for name in ("arcade.json", "piece.json"):
            for inst in Project.load(EXAMPLES / name).instruments.values():
                voice = {"inst": inst, "notes": [[0, TEST_NOTE, TEST_TICKS]]}
                samples = render_entry({"def": {"voices": [voice]}}, RATE)[0]
                scope = self.scope()
                scope.show(samples, RATE, midi(TEST_NOTE))
                self.draw()
                self.assertEqual(scope.help, "3 cycles of c4, 11.5 ms")
                self.assertTrue(all(self.wave()))
                spectrum = self.spectrum()
                spectrum.show(samples, RATE)
                self.draw()
                # The share the spectrum states is the one the right panel has, to the last place.
                self.assertEqual(spectrum.above, brightness(samples, rate=RATE)["above_4k"])
                words = f"{round(spectrum.above * 100)}% above"
                ink = theme.FAULT if spectrum.above > sounds._BRIGHT else theme.TEXT
                self.assertEqual(self.ink(WORDS, ink), self.written(words, WORDS))
                self.assertEqual(self.at(196, 79), theme.FAULT)
                seen += 1
        self.assertEqual(seen, 24)


if __name__ == "__main__":
    unittest.main()
