"""Tests for rowsynth.gui.wave: the sums of the view of a sound, its clock under the wheel, and what it draws."""
from __future__ import annotations

import os
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from rowsynth.bank import entry_kind, render_entry
from rowsynth.errors import RowSynthError
from rowsynth.gui import layout, soundview, sprites, theme
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.font import Font
from rowsynth.gui.roll import DARKER, LIGHTER, mix
from rowsynth.gui.soundview import Bar, Shape
from rowsynth.gui.ui import Event, Ui
from rowsynth.gui.wave import (VARIANT_GAP, ZONES, ZOOM_MOST, Map, SoundView, echo_colour, joined, keys, places,
                               ruler_step, stretch, tint, voice_colour)
from rowsynth.project import Project

# pygame reads these when it is imported: no test opens a window or plays a sound.
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
try:
    import pygame
except ImportError:
    pygame = None

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
NEEDS = "pygame is not installed"

#: A view whose grid is (58, 30, 400, 205), under a ruler from y 20 and over a wave (58, 239, 400, 81).
RECT = (10, 20, 448, 300)
#: The surface it is drawn on, which is larger than it on every side, and the colour of what is not the view's.
SIZE = (470, 330)
OUTSIDE = theme.GOOD
#: The samples in a second of the waves made here: on a clock of a second that is two of them to a pixel.
RATE = 800


def bar(start: float, seconds: float, pitch: int, inst: str = "zap", voice: int = 0) -> Bar:
    """A note of a sound."""
    return Bar(start, seconds, float(pitch), inst, voice)


def one(*bars: Bar, echo: tuple[tuple[float, float], ...] = ()) -> list[tuple[str, float, Shape]]:
    """One sound a second long, as `soundview.shapes` gives one."""
    return [("", 0.0, Shape(bars, 1.0, echo))]


#: g4 held for 0.15 s, and in a second voice c5 for a quarter of a second, with an instrument written in place.
#: The echo brings both back once, a quarter of a second later and a quarter as strong. The keys are fitted so
#: that g4 is the row from y 151 and c5 the row from y 96.
ONE = one(bar(0.25, 0.15, 67), bar(0.5, 0.25, 72, "", 1), echo=((0.25, 0.25),))


#: Two variants, of 0.4 and of 0.5 s. A tenth of a second is left free between them, so the clock is a second
#: long: the second variant starts at 0.5 s, which is x 258, and its label stands in the strip from y 30.
PAIR = [("1", 0.0, Shape((bar(0.0, 0.2, 60),), 0.4, ())), ("2", 0.0, Shape((bar(0.0, 0.3, 64),), 0.5, ()))]
#: The three zones of a held sound, a second long together: they meet at x 158 and at x 358.
HELD = [("start", 0.0, Shape((bar(0.0, 0.25, 60),), 0.25, ())), ("loop", 0.25, Shape((bar(0.0, 0.5, 60),), 0.5, ())),
        ("end", 0.75, Shape((), 0.25, ()))]

#: A project of one sound of two variants that differ plainly: the second is an octave higher, twice as long and
#: a quarter as loud.
OWN = {"instruments": {"tone": {"wave": "tri"}},
       "sounds": {"pair": {"defs": [{"voices": [{"inst": "tone", "notes": [[0, "c4", 6]]}]},
                                    {"voices": [{"inst": "tone", "notes": [[0, "c5", 12]]}]}], "db": [0, -12]}}}


def rows(*seconds: float) -> list[np.ndarray]:
    """A row of silence for each of some lengths."""
    return [np.zeros(round(length * RATE), dtype=np.float32) for length in seconds]


def wave() -> np.ndarray:
    """A second of samples: a quarter of it swings by a half, a quarter stays at a quarter, then a click at full
    scale and silence, and the last quarter swings from 0 to just short of what counts as clipped."""
    samples = np.zeros(RATE, dtype=np.float32)
    samples[0:200:2], samples[1:200:2] = 0.5, -0.5
    samples[200:400] = 0.25
    samples[400:402] = 1.0, -1.0
    samples[600:800:2] = 0.998
    return samples


def strength(colour: tuple[int, ...], ground: tuple[int, ...]) -> int:
    """How far a colour is from the one it lies on."""
    return sum(abs(one - other) for one, other in zip(colour, ground))


class Sums(unittest.TestCase):
    def test_the_x_of_a_second_and_the_second_at_an_x_are_opposites_at_two_zooms(self) -> None:
        whole = Map(RECT)
        self.assertEqual((whole.scale, [whole.x(second) for second in (0, 0.25, 0.5, 1)]),
                         (400.0, [58, 158, 258, 458]))
        # Four steps in, a second is 1.25 times wider for each of them.
        near = Map(RECT, start=0.5, zoom=4)
        self.assertEqual((near.scale, [near.x(second) for second in (0.5, 0.75)]), (976.5625, [58, 302]))
        for seen in (whole, near):
            for x in range(40, 470):
                self.assertEqual(seen.x(seen.second_at(x)), x)
            for second in (0.0, 0.3, 0.55, 0.9):
                # A second is at its pixel, which is half a pixel either way.
                self.assertLessEqual(abs(seen.second_at(seen.x(second)) - second), 0.5 / seen.scale)
        # A clock of no length is counted as a second long, and one of 2 seconds has half the scale.
        self.assertEqual((Map(RECT, seconds=0.0).scale, Map(RECT, seconds=2.0).scale), (400.0, 200.0))

    def test_the_y_of_a_pitch_and_the_pitch_at_a_y_are_opposites_at_two_heights_of_a_key(self) -> None:
        self.assertEqual([Map(RECT).y(pitch) for pitch in (72, 71, 60)], [30, 41, 162])
        self.assertEqual(Map(RECT, top=80, key_h=5).y(60), 130)
        for seen in (Map(RECT), Map(RECT, top=80, key_h=5)):
            for pitch in seen.pitches():
                y = seen.y(pitch)
                self.assertEqual(seen.y(pitch - 1) - y, seen.key_h)
                self.assertEqual({seen.pitch_at(y + step) for step in range(seen.key_h)}, {pitch})
        # 205 pixels are 18 keys of 11 and most of another, or 41 keys of 5.
        self.assertEqual((Map(RECT).pitches(), Map(RECT, top=80, key_h=5).pitches()), (range(54, 73), range(40, 81)))
        # The keys are fitted round the notes: g4 to c5 are 6 of the 18, with 6 keys over them. Where the notes
        # need the room a key is lower, down to a pixel.
        self.assertEqual((keys(205, 67, 72), keys(205, 60, 60)), ((78, 11), (68, 11)))
        self.assertEqual((keys(205, 40, 80), keys(205, 0, 127), keys(100, 0, 127), keys(0, 60, 60)),
                         ((80, 5), (165, 1), (127, 1), (60, 1)))

    def test_the_bar_of_a_note_is_in_the_row_of_its_key_from_just_after_its_second(self) -> None:
        seen = Map(RECT)
        self.assertEqual([seen.bar(0.5, 0.25, 72), seen.bar(0.5, 0.25, 70.0)], [(259, 30, 99, 10), (259, 52, 99, 10)])
        self.assertEqual(Map(RECT, start=0.5, zoom=4, top=80, key_h=5).bar(0.75, 0.25, 67), (303, 95, 243, 4))
        # Two in a row leave a pixel free between them, the shortest is a pixel wide, and the lowest a pixel high.
        first, second = seen.bar(0.25, 0.25, 60), seen.bar(0.5, 0.25, 60)
        self.assertEqual((first[0] + first[2], second[0]), (seen.x(0.5), seen.x(0.5) + 1))
        self.assertEqual((seen.bar(0.5, 0.001, 72), replace(seen, key_h=1).bar(0.5, 0.25, 72)),
                         ((259, 30, 1, 10), (259, 30, 99, 1)))

    def test_columns_with_a_gap_between_them_are_stretched_to_meet_and_the_others_are_left(self) -> None:
        # The second lies under the first with a gap, the third over the second, the fourth overlaps the third.
        self.assertEqual(joined([(10, 12), (20, 22), (5, 6), (6, 9)]), [(10, 12), (13, 22), (5, 19), (6, 9)])
        # A neighbour that already touches stays as it is, the one after it is brought down to meet it.
        self.assertEqual(joined([(10, 12), (13, 14), (9, 9)]), [(10, 12), (13, 14), (9, 12)])
        self.assertEqual((joined([(4, 4)]), joined([])), ([(4, 4)], []))

    def test_a_column_of_the_wave_ends_where_its_highest_and_its_lowest_sample_fall(self) -> None:
        seen = Map(RECT)
        # The box is (58, 239, 400, 81): zero is its middle line, and full scale is 2 pixels short of its outline.
        self.assertEqual((seen.wave, seen.zero, seen.column(-1.0, 1.0), seen.column(0.0, 0.0)),
                         ((58, 239, 400, 81), 279, (242, 316), (279, 279)))
        self.assertEqual((seen.column(-0.5, 0.5), seen.column(0.25, 0.5), seen.column(-3.0, 2.0)),
                         ((260, 297), (260, 270), (242, 316)))
        # With no notes the box is all there is under the ruler, and a column is as much longer.
        bare = Map(RECT, notes=False)
        self.assertEqual((bare.grid, bare.wave, bare.zero, bare.column(-1.0, 1.0)),
                         ((58, 30, 400, 0), (58, 30, 400, 290), 175, (34, 316)))

    def test_the_ruler_goes_in_tenths_of_a_second_or_in_seconds_by_what_its_numbers_have_room_for(self) -> None:
        # A number in tenths takes 24 pixels with the air round it, and a number of seconds 12.
        self.assertEqual([ruler_step(scale) for scale in (240, 239, 12, 11.9, 1.8, 1.7)], [-1, 0, 0, 1, 1, 2])
        # Zoomed far in it goes on to hundredths and to thousandths, and no further.
        self.assertEqual([ruler_step(scale) for scale in (2999, 3000, 35999, 36000, 1e9)], [-1, -2, -2, -3, -3])
        # By the width: a second on 400 pixels is marked in tenths, and on 160 pixels at its start alone. Four
        # seconds on 400 pixels are marked in seconds.
        self.assertEqual((Map(RECT).marks(), Map((10, 20, 208, 300)).marks(), Map(RECT, seconds=4.0).marks()),
                         (list(range(10)), [0], [0, 1, 2, 3]))
        # Only the marks whose line is on the grid are given: here those of 0.5 to 0.9 s.
        near = Map(RECT, start=0.5, zoom=4)
        self.assertEqual((near.marks(), near.x(0.9), near.x(1.0)), ([5, 6, 7, 8, 9], 449, 546))

    def test_shapes_are_laid_at_their_seconds_and_variants_one_after_the_other_with_a_gap(self) -> None:
        short, long = Shape((), 0.25, ()), Shape((), 0.5, ())
        self.assertEqual((places([]), places([], True), places([("", 0.0, long)])), ([], [], [0.0]))
        # Parts, and the zones of a held sound, are where they start: two parts can start together.
        self.assertEqual(places([("1", 0.0, short), ("2", 0.02, long)]), [0.0, 0.02])
        self.assertEqual(places([("1", 0.0, short), ("2", 0.0, long)]), [0.0, 0.0])
        self.assertEqual(places([("start", 0.0, short), ("loop", 0.25, long), ("end", 0.75, short)]),
                         [0.0, 0.25, 0.75])
        # Variants all start at 0: each is laid after the one before, a share of the longest further on.
        gap = VARIANT_GAP * 0.5
        self.assertEqual(places([("1", 0.0, short), ("2", 0.0, long), ("3", 0.0, short)], True),
                         [0.0, 0.25 + gap, 0.75 + 2 * gap])

    def test_a_place_is_in_the_stretch_that_starts_last_before_it_or_at_it(self) -> None:
        self.assertEqual([stretch([58, 258], at) for at in (57, 58, 257, 258, 1000)], [None, 0, 0, 1, 1])
        self.assertEqual((stretch([], 5), stretch([0.0, 0.5, 0.5], 0.5)), (None, 2))


class View(unittest.TestCase):
    def setUp(self) -> None:
        self.ui = Ui()
        self.view = self.ui.add(SoundView(RECT, "the notes of a sound"))
        self.view.show(ONE, [wave()], RATE, theme.SOUND)

    def seen(self) -> tuple[float, int]:
        """The view: the second at the left edge of its grid, to the millionth, and its zoom."""
        return round(self.view.start, 6), self.view.zoom

    def wheel(self, steps: int, *mods: str, pos: tuple[int, int] = (258, 100)) -> bool:
        """Turn the wheel with the mouse on the grid, at half a second, some keys held. Whether it was used."""
        return self.ui.handle(Event("wheel", pos, wheel=steps, mods=frozenset(mods)))

    def test_ctrl_and_the_wheel_zoom_about_the_mouse_between_the_whole_and_the_most(self) -> None:
        self.assertTrue(self.wheel(1, "ctrl"))
        self.assertEqual((self.seen(), self.view.map.scale, self.view.map.second_at(258)), ((0.1, 1), 500.0, 0.5))
        # In or out, the second under the mouse stays under it, here and at another place.
        for step in (1, 1, 1, -1, 1, -1, -1):
            self.wheel(step, "ctrl")
            self.assertAlmostEqual(self.view.map.second_at(258), 0.5, places=9)
        before = self.view.map.second_at(390)
        self.wheel(3, "ctrl", pos=(390, 250))
        self.assertAlmostEqual(self.view.map.second_at(390), before, places=9)
        self.assertEqual(self.view.zoom, 5)
        self.wheel(100, "ctrl")
        self.assertEqual(self.view.zoom, ZOOM_MOST)
        self.wheel(-100, "ctrl", "shift")
        self.assertEqual(self.seen(), (0.0, 0))
        # Over the keys, it is the second at the left edge of the grid that stays.
        self.wheel(2, "ctrl")
        before = self.view.map.second_at(58)
        self.wheel(2, "ctrl", pos=(20, 100))
        self.assertEqual((self.view.zoom, self.view.map.second_at(58)), (4, before))

    def test_the_wheel_alone_scrolls_the_time_only_when_the_clock_is_zoomed_in(self) -> None:
        # Not zoomed in, the wheel is not the view's to use, and neither is a key or a wheel off the view.
        self.assertEqual((self.wheel(-1), self.wheel(1), self.ui.handle(Event("key", key="space"))),
                         (False, False, False))
        self.assertEqual((self.wheel(1, "ctrl", pos=(5, 5)), self.seen()), (False, (0.0, 0)))
        self.wheel(1, "ctrl")
        # A step is 36 pixels, at 500 to the second, and the view goes no further than the clock does.
        self.assertTrue(self.wheel(-1))
        self.assertEqual(self.seen(), (0.172, 1))
        self.wheel(-1)
        self.assertEqual((self.seen(), self.view.map.x(1.0)), ((0.2, 1), 458))
        self.wheel(100)
        self.assertEqual(self.seen(), (0.0, 1))

    def test_at_the_start_the_whole_fits_and_shown_again_the_view_stays_until_it_is_fitted(self) -> None:
        seen = self.view.map
        self.assertEqual((self.view.seconds, self.seen(), seen.x(0.0), seen.x(1.0)), (1.0, (0.0, 0), 58, 458))
        self.assertEqual((seen.top, seen.key_h, seen.labels, seen.notes), (78, 11, False, True))
        self.wheel(4, "ctrl")
        self.wheel(-3)
        moved = self.seen()
        self.assertNotEqual(moved, (0.0, 0))
        # The render came, or the file changed outside: other notes on the same clock, fitted to their keys.
        self.view.show(one(bar(0.0, 0.5, 60)), [wave()], RATE, theme.SONG)
        self.assertEqual((self.seen(), self.view.colour, self.view.map.top), (moved, theme.SONG, 68))
        # The clock is as long as the longer of the shapes and the render, and a shorter one ends sooner.
        self.view.show(ONE, [np.zeros(2 * RATE)], RATE, theme.SOUND)
        self.assertEqual((self.view.seconds, self.seen()), (2.0, moved))
        self.view.show([("", 0.0, Shape((), 0.5, ()))], [], RATE, theme.SOUND)
        self.assertEqual((self.view.seconds, self.view.zoom, self.view.map.x(0.5)), (0.5, 4, 458))
        self.view.fit()
        self.assertEqual((self.seen(), self.view.map.x(0.5)), ((0.0, 0), 458))

    def test_the_help_says_what_note_or_what_moment_of_the_wave_the_mouse_is_on(self) -> None:
        self.assertEqual((self.view.help, self.ui.help), ("the notes of a sound", ""))
        for pos, line in (((170, 155), "g4, 0.15 s, instrument zap"),
                          ((300, 100), "c5, 0.25 s"),
                          # The copy an echo brings of g4 is no note, and neither are the keys or the ruler.
                          ((270, 155), "the notes of a sound"),
                          ((20, 155), "the notes of a sound"),
                          ((170, 25), "the notes of a sound"),
                          ((118, 250), "0.15 s, -6.0 dB"),
                          ((178, 300), "0.30 s, -12.0 dB"),
                          ((318, 279), "0.65 s, silent"),
                          ((20, 279), "the notes of a sound")):
            self.ui.handle(Event("move", pos))
            self.assertEqual(self.ui.help, line, pos)
        self.assertEqual((self.view.bar_at((170, 155)), self.view.bar_at((270, 155))), (ONE[0][2].bars[0], None))
        # Past the end of the samples there is no level to tell.
        self.view.show(ONE, [wave()[:200]], RATE, theme.SOUND)
        self.assertEqual(self.ui.help, "the notes of a sound")
        self.view.help = "another line"
        self.assertEqual(self.ui.help, "another line")

    def test_what_is_no_kind_no_rows_no_wave_or_no_rate_is_refused_and_what_was_shown_stays(self) -> None:
        one_row = "samples: must be an array of one dimension"
        for shapes, samples, rate, kind, wrong in (
                ([], [wave()], RATE, "variants", "kind: must be one of def, defs, parts, held, not 'variants'"),
                ([], wave(), RATE, "def", "samples: must be a list of rows, each an array of one dimension"),
                ([], [[0.0, 0.5]], RATE, "def", one_row),
                ([], [np.zeros((2, 4))], RATE, "def", one_row),
                # Two rows are for two shapes, and three shapes have one row or three.
                ([], rows(0.5, 0.5), RATE, "def", "samples: must be one row, or a row for each of the 0 shapes, "
                                                  "not 2 rows"),
                (HELD, rows(0.25, 0.5), RATE, "held", "samples: must be one row, or a row for each of the 3 "
                                                      "shapes, not 2 rows"),
                ([], [wave()], 0, "def", "rate: must be a whole number, 1 or more, not 0")):
            with self.assertRaises(RowSynthError) as refused:
                self.view.show(shapes, samples, rate, theme.SONG, kind)
            self.assertEqual(str(refused.exception), wrong)
        self.assertEqual((self.view.colour, self.view.kind, self.view.seconds, self.view.map.notes),
                         (theme.SOUND, "def", 1.0, True))

    def test_a_press_on_the_label_of_a_variant_or_in_its_stretch_of_the_wave_chooses_it(self) -> None:
        chosen: list[int] = []
        self.view.chosen = chosen.append
        # The wave of the first is at a half all along, and that of the second at a quarter.
        self.view.show(PAIR, [np.full(320, 0.5, dtype=np.float32), np.full(400, 0.25, dtype=np.float32)], RATE,
                       theme.SOUND, "defs")
        seen = self.view.map
        self.assertEqual((seen.names, seen.wave, seen.x(0.5), self.view.seconds),
                         ((58, 30, 400, 10), (58, 239, 400, 81), 258, 1.0))
        # The second variant: on its label, at its first pixel, and in its stretch of the wave. The first: its
        # stretch goes on through what is left free after it, up to where the second starts.
        for pos, which in (((300, 35), 1), ((258, 30), 1), ((400, 280), 1), ((457, 319), 1),
                           ((60, 35), 0), ((58, 239), 0), ((257, 280), 0)):
            self.assertEqual((self.view.choice_at(pos), self.ui.handle(Event("down", pos, button=1)), chosen[-1]),
                             (which, True, which), pos)
            # The release that follows chooses nothing more.
            self.ui.handle(Event("up", pos, button=1))
        self.assertEqual(chosen, [1, 1, 1, 1, 0, 0, 0])
        # On its label the help says so. On its wave it is the moment and the level of that wave, as for any
        # sound, and there is no level where no wave is.
        for pos, line in (((300, 35), "click to hear variant 2"), ((60, 35), "click to hear variant 1"),
                          ((300, 280), "0.60 s, -12.0 dB"), ((100, 280), "0.10 s, -6.0 dB"),
                          ((238, 280), "the notes of a sound")):
            self.ui.handle(Event("move", pos))
            self.assertEqual(self.ui.help, line)

    def test_a_press_anywhere_else_or_on_a_sound_with_no_variants_chooses_nothing(self) -> None:
        chosen: list[int] = []
        self.view.chosen = chosen.append
        self.view.show(PAIR, rows(0.4, 0.5), RATE, theme.SOUND, "defs")
        # The grid, the ruler, and the keys beside the wave are no variant's, and neither button but the left
        # chooses.
        for pos, button in (((300, 100), 1), ((300, 25), 1), ((20, 280), 1), ((300, 280), 3), ((300, 35), 2)):
            self.assertFalse(self.ui.handle(Event("down", pos, button=button)), pos)
            self.ui.handle(Event("up", pos, button=button))
        self.assertEqual((self.view.choice_at((300, 100)), self.view.choice_at((20, 280))), (None, None))
        # A plain sound, the parts of a sound and the zones of a held one are not to be chosen from.
        for shapes, samples, kind in ((ONE, [wave()], "def"), (PAIR, rows(1.0), "parts"),
                                      (HELD, rows(0.25, 0.5, 0.25), "held")):
            self.view.show(shapes, samples, RATE, theme.SOUND, kind)
            for pos in ((300, 35), (300, 280), (100, 280)):
                self.assertEqual((self.view.choice_at(pos), self.ui.handle(Event("down", pos, button=1))),
                                 (None, False))
                self.ui.handle(Event("up", pos, button=1))
        self.assertEqual(chosen, [])
        # With no one to tell, a press on a variant is not used, and the help does not ask for one.
        self.view.chosen = None
        self.view.show(PAIR, rows(0.4, 0.5), RATE, theme.SOUND, "defs")
        self.assertEqual((self.view.choice_at((300, 35)), self.ui.handle(Event("down", (300, 35), button=1)),
                          self.ui.help), (1, False, "the notes of a sound"))

    def test_the_second_for_a_place_in_a_shape_is_none_outside_the_stretch_of_that_shape(self) -> None:
        self.view.show(PAIR, rows(0.4, 0.5), RATE, theme.SOUND, "defs")
        # The second variant starts at half a second, and is half a second long.
        self.assertEqual([self.view.clock(1, into) for into in (0.0, 0.25, 0.5)], [0.5, 0.75, 1.0])
        self.assertEqual([self.view.clock(1, into) for into in (-0.01, 0.51, 3.0)], [None, None, None])
        # The first ends at 0.4 s: what is left free after it is not in it. There are two shapes, and no more.
        self.assertEqual([self.view.clock(0, into) for into in (0.0, 0.4, 0.45)], [0.0, 0.4, None])
        self.assertEqual((self.view.clock(2, 0.0), self.view.clock(-1, 0.0)), (None, None))
        # The zones of a held sound are where they start, and a plain sound is at 0.
        self.view.show(HELD, rows(0.25, 0.5, 0.25), RATE, theme.SOUND, "held")
        self.assertEqual([self.view.clock(zone, 0.125) for zone in range(3)], [0.125, 0.375, 0.875])
        self.assertEqual((self.view.clock(0, 0.26), self.view.clock(2, 0.25)), (None, 1.0))
        self.view.show(ONE, [wave()], RATE, theme.SOUND)
        self.assertEqual((self.view.clock(0, 0.3), self.view.clock(0, 1.1)), (0.3, None))


@unittest.skipUnless(pygame, NEEDS)
class Drawn(unittest.TestCase):
    def setUp(self) -> None:
        self.surface = pygame.Surface(SIZE)
        self.surface.fill(OUTSIDE)
        self.canvas = Canvas(self.surface, Font(sprites.load("font")))
        self.ui = Ui()
        self.view = self.ui.add(SoundView(RECT))
        self.view.show(ONE, [wave()], RATE, theme.SOUND)

    def at(self, *pos: int) -> tuple[int, ...]:
        """The colour of one pixel."""
        return tuple(self.surface.get_at(pos))[:3]

    def colours(self, rect: layout.Rect) -> set[tuple[int, ...]]:
        """The colours a rectangle has in it."""
        x, y, w, h = rect
        return {self.at(x + column, y + row) for column in range(w) for row in range(h)}

    def where(self, colour: tuple[int, ...], rect: layout.Rect) -> set[tuple[int, int]]:
        """The pixels of a rectangle that are in a colour."""
        x, y, w, h = rect
        return {(x + column, y + row) for column in range(w) for row in range(h)
                if self.at(x + column, y + row) == colour}

    def draw(self) -> Map:
        """Draw the view, and give its sums."""
        self.ui.draw(self.canvas)
        return self.view.map

    def sound(self, project: Project,
              name: str) -> tuple[SoundView, Map, list[tuple[str, float, Shape]], list[np.ndarray]]:
        """A sound of a project drawn 600 by 400 over its real render: the view, its sums, the shapes and the
        buffers. The surface that is looked at from here on is the one it is drawn on."""
        shapes = soundview.shapes(project.sounds[name], instruments=project.instruments)
        buffers = render_entry(project.sound(name))
        view = SoundView((0, 0, 600, 400))
        view.show(shapes, buffers, 44100, theme.SOUND, entry_kind(project.sounds[name]))
        self.surface = pygame.Surface((600, 400))
        view.draw(Canvas(self.surface, Font(sprites.load("font"))))
        return view, view.map, shapes, buffers

    def arcade(self, name: str) -> tuple[SoundView, Map, list[tuple[str, float, Shape]], list[np.ndarray]]:
        """A sound of the arcade example, drawn as `sound` draws one."""
        return self.sound(Project.load(EXAMPLES / "arcade.json"), name)

    def wave_of(self, seen: Map, left: int, columns: np.ndarray) -> set[tuple[int, int]]:
        """The pixels that some columns of a wave take, made to meet, but for the two that lie under the outline
        of the box."""
        ends = joined([seen.column(*column) for column in columns.tolist()])
        return {(left + index, y) for index, (upper, lower) in enumerate(ends) for y in range(upper, lower + 1)
                if seen.wave[0] < left + index < seen.wave[0] + seen.wave[2] - 1}

    def test_a_bar_is_at_its_place_in_the_colour_of_its_voice_and_a_wide_one_has_its_name_on_it(self) -> None:
        # The view's own colour, and then the channels' colours from the one after it, round and round.
        self.assertEqual([voice_colour(theme.SOUND, voice) for voice in range(5)],
                         [theme.SOUND, theme.SALMON, theme.MUSTARD, theme.SONG, theme.SOUND])
        self.assertEqual([voice_colour(theme.INSTRUMENT, voice) for voice in range(3)],
                         [theme.INSTRUMENT, theme.SONG, theme.SOUND])
        # Two notes too short for their names: their bars are (159, 151, 9, 10) and (259, 96, 9, 10).
        self.view.show(one(bar(0.25, 0.025, 67), bar(0.5, 0.025, 72, voice=1)), [wave()], RATE, theme.SOUND)
        seen = self.draw()
        self.assertEqual(self.where(theme.SOUND, seen.grid),
                         {(x, y) for x in range(160, 167) for y in range(152, 160)})
        self.assertEqual(self.where(theme.SALMON, seen.grid),
                         {(x, y) for x in range(260, 267) for y in range(97, 105)})
        # A bar is lit along its upper edge and its start, and dark along its lower edge and its end.
        light, dark = mix(theme.SOUND, theme.TEXT, LIGHTER), mix(theme.SOUND, theme.BACKGROUND, DARKER)
        self.assertEqual(self.where(light, seen.grid),
                         {(x, 151) for x in range(159, 167)} | {(159, y) for y in range(151, 161)})
        self.assertEqual(self.where(dark, seen.grid),
                         {(x, 160) for x in range(160, 168)} | {(167, y) for y in range(151, 161)})
        # The pixel at its second, and the line under its key, are left as they were.
        self.assertEqual((self.at(158, 155), self.at(163, 161)), (theme.RAISED, theme.PANEL))
        self.view.show(ONE, [wave()], RATE, theme.SOUND)
        self.draw()
        self.assertEqual(self.colours((162, 152, 11, 7)), {theme.SOUND, theme.BACKGROUND})

    def test_the_copy_an_echo_brings_of_a_bar_is_later_by_its_time_and_fainter_by_its_strength(self) -> None:
        seen = self.draw()
        # g4 is the bar (159, 151, 59, 10). A quarter of a second is 100 pixels, and the copy is 2 pixels lower
        # at the top and at the foot.
        faint = echo_colour(theme.SOUND, 0.25)
        self.assertEqual(self.where(faint, seen.grid), {(x, y) for x in range(259, 318) for y in range(153, 159)})
        # The copy of the second voice's bar is in that voice's colour, and ends with the grid.
        self.assertEqual(self.where(echo_colour(theme.SALMON, 0.25), seen.grid),
                         {(x, y) for x in range(359, 458) for y in range(98, 104)})
        # The weaker the repeat, the nearer its copy is to the background.
        fainter = [echo_colour(theme.SOUND, strong) for strong in (1.0, 0.25, 0.04, 0.0)]
        told = [strength(colour, theme.BACKGROUND) for colour in fainter]
        self.assertEqual((fainter[1], fainter[3]), (mix(theme.BACKGROUND, theme.SOUND, 0.4), theme.BACKGROUND))
        self.assertTrue(strength(theme.SOUND, theme.BACKGROUND) > told[0] > told[1] > told[2] > told[3] == 0)
        # One note alone is in the row from y 118: a second repeat is later and fainter still.
        self.view.show(one(bar(0.25, 0.15, 67), echo=((0.25, 0.25), (0.5, 0.04))), [wave()], RATE, theme.SOUND)
        self.draw()
        self.assertEqual((self.at(270, 122), self.at(370, 122), self.at(270, 119)), (faint, fainter[2], theme.RAISED))
        # A sound with no echo has no copies.
        self.view.show(one(bar(0.25, 0.15, 67)), [wave()], RATE, theme.SOUND)
        self.draw()
        self.assertEqual(self.colours((259, 119, 199, 8)), {theme.RAISED, theme.LINE})

    def test_the_wave_has_a_column_for_each_pixel_from_its_lowest_sample_to_its_highest(self) -> None:
        seen = self.draw()
        (left, columns), = self.view.columns()
        self.assertEqual((left, len(columns)), (58, 400))
        self.assertTrue(np.array_equal(columns, soundview.columns(wave(), 400)))
        # A swing of a half, a level of a quarter, silence, and a swing from 0 to just under full scale.
        for x, ends in ((108, (260, 297)), (178, (270, 270)), (308, (279, 279)), (408, (242, 279))):
            self.assertEqual(self.where(theme.SOUND, (x, 239, 1, 81)), {(x, y) for y in range(ends[0], ends[1] + 1)})
        # Column for column it is what the samples say, with the line at zero behind it.
        self.assertEqual(self.where(theme.SOUND, seen.wave) | self.where(theme.FAULT, seen.wave),
                         self.wave_of(seen, left, columns))
        self.assertEqual(self.colours(seen.wave), {theme.BACKGROUND, theme.LINE, theme.SOUND, theme.FAULT})
        self.assertIn(theme.DIM, self.colours((10, 239, 48, 81)))
        # With no rows the box holds the line at zero and nothing else, and the bars are there as before.
        self.view.show(ONE, [], RATE, theme.SOUND)
        self.draw()
        self.assertEqual(self.view.columns(), [])
        self.assertEqual(self.where(theme.LINE, (59, 240, 398, 79)), {(x, 279) for x in range(59, 457)})
        self.assertEqual((self.colours((59, 240, 398, 79)), self.at(170, 152)),
                         ({theme.BACKGROUND, theme.LINE}, theme.SOUND))
        # With no shapes the wave is all there is under the ruler: no keys stand beside it, only its name, and
        # its columns are as much longer.
        self.view.show([], [wave()], RATE, theme.INSTRUMENT)
        seen = self.draw()
        self.assertEqual((seen.notes, seen.wave, self.view.seconds), (False, (58, 30, 400, 290), 1.0))
        self.assertEqual(self.colours((10, 30, 48, 290)), {theme.PANEL, theme.DIM})
        self.assertEqual((seen.column(-0.5, 0.5), self.where(theme.INSTRUMENT, (108, 30, 1, 290))),
                         ((104, 245), {(108, y) for y in range(104, 246)}))

    def test_a_column_whose_samples_reach_full_scale_is_in_the_colour_of_a_fault(self) -> None:
        seen = self.draw()
        # The click at half a second is one column, from full scale above to full scale below.
        self.assertEqual(self.where(theme.FAULT, (0, 0, *SIZE)), {(258, y) for y in range(242, 317)})
        # A wave past full scale is drawn at full scale, and stays in its box.
        self.view.show(ONE, [np.full(RATE, 1.5, dtype=np.float32)], RATE, theme.SOUND)
        self.draw()
        self.assertEqual(self.where(theme.FAULT, seen.wave), {(x, 242) for x in range(59, 457)})
        self.assertEqual(self.where(theme.SOUND, seen.wave), set())

    def test_the_ruler_numbers_its_marks_over_their_lines_and_the_keys_are_as_on_a_piano(self) -> None:
        seen = self.draw()
        # A tenth is 40 pixels. Its line is faint on the grid, and that of every tenth mark is stronger.
        self.assertEqual([self.at(seen.x(tenth / 10), 33) for tenth in (0, 1, 5, 9)],
                         [theme.DIM, theme.LINE, theme.LINE, theme.LINE])
        # Each number starts 2 pixels after its line, three letters wide, and the word stands before the ruler.
        inked = {x for x, _ in self.where(theme.DIM, seen.ruler)}
        self.assertEqual([edge(x for x in inked if 98 <= x < 138) for edge in (min, max)], [100, 116])
        self.assertEqual((self.colours(seen.ruler), theme.DIM in self.colours((10, 20, 48, 10))),
                         ({theme.PANEL, theme.DIM}, True))
        # c5 is a white key with its name at its right end, and d-5 a black one for most of its length.
        c, d_flat = seen.y(72), seen.y(73)
        self.assertEqual([self.at(12, c + 3), self.at(12, d_flat + 3), self.at(45, d_flat + 3)],
                         [theme.TEXT, theme.BACKGROUND, theme.TEXT])
        self.assertIn(theme.BACKGROUND, self.colours((42, c + 2, 11, 7)))
        # The row of a black key is darker than that of a white one, and the keys end a pixel short of the grid.
        self.assertEqual((self.at(70, c + 3), self.at(70, d_flat + 3)), (theme.RAISED, theme.BACKGROUND))
        self.assertEqual(self.colours((57, 30, 1, 205)), {theme.PANEL})

    def test_the_playhead_is_a_line_at_its_second_and_nothing_is_drawn_outside_the_rectangle(self) -> None:
        self.view.playhead = 0.5
        self.draw()
        self.assertEqual(self.where(theme.ACTIVE, (0, 0, *SIZE)), {(258, y) for y in range(20, 320)})
        # None is no line, and neither is a second that is not in view.
        for playhead in (None, -0.1, 1.5):
            self.view.playhead = playhead
            self.draw()
            self.assertEqual(self.where(theme.ACTIVE, (0, 0, *SIZE)), set())
        around = ((0, 0, 470, 20), (0, 320, 470, 10), (0, 20, 10, 300), (458, 20, 12, 300))
        self.view.playhead = 0.3
        for steps, mod in ((0, ""), (3, "ctrl"), (-2, ""), (20, "ctrl")):
            self.ui.handle(Event("wheel", (300, 100), wheel=steps, mods=frozenset({mod})))
            self.draw()
            self.assertEqual(set().union(*(self.colours(rect) for rect in around)), {OUTSIDE})
            self.assertNotIn(OUTSIDE, self.colours(RECT))
        # A view too small for its parts keeps to its rectangle as well.
        self.surface.fill(OUTSIDE)
        self.view.rect = (100, 100, 70, 60)
        self.draw()
        self.assertEqual(len(self.where(OUTSIDE, (0, 0, *SIZE))), 470 * 330 - 70 * 60)

    def test_each_variant_has_the_wave_of_its_own_buffer_under_its_own_bars(self) -> None:
        view, seen, shapes, buffers = self.sound(Project.from_dict(OWN), "pair")
        waves = view.columns()
        laid = places(shapes, True)
        starts = [seen.x(place) for place in laid]
        self.assertEqual(([label for label, _, _ in shapes], len(buffers), len(waves), view.kind),
                         (["1", "2"], 2, 2, "defs"))
        for (left, columns), place, (_, _, shape), buffer in zip(waves, laid, shapes, buffers):
            # A wave starts at the pixel its variant does, as the bar of the note there, and is its own buffer
            # from the first sample to the last, a column a pixel.
            self.assertEqual((left, left + len(columns)), (seen.x(place), seen.x(place + len(buffer) / 44100)))
            self.assertTrue(np.array_equal(columns, soundview.columns(buffer, len(columns))))
            self.assertEqual(self.at(left + 3, seen.y(round(shape.bars[0].pitch)) + 4), theme.SOUND)
        # What is drawn of the two is those columns, and nothing in what is left free between them.
        painted = self.where(theme.SOUND, seen.wave)
        self.assertEqual(painted, set().union(*(self.wave_of(seen, left, columns) for left, columns in waves)))
        self.assertEqual({x for x, _ in painted if waves[0][0] + len(waves[0][1]) <= x < starts[1]}, set())
        # The second is a quarter as loud as the first, and twice as long.
        loudest = [float(np.abs(columns).max()) for _, columns in waves]
        self.assertTrue(loudest[1] < 0.2 < 0.7 < loudest[0])
        self.assertGreater(len(waves[1][1]), 1.5 * len(waves[0][1]))
        # A line runs down the box of the waves where the second starts, over and under its wave.
        x, y, _, h = seen.wave
        self.assertEqual({self.at(starts[1], y + 3), self.at(starts[1], y + h - 4)}, {theme.LINE})

    def test_the_label_of_the_variant_that_was_played_is_lit_and_the_playhead_is_in_its_stretch(self) -> None:
        self.view.show(PAIR, rows(0.4, 0.5), RATE, theme.SOUND, "defs")
        labels = (range(61, 66), range(261, 266))
        for played in (1, 0, None):
            self.view.played = played
            seen = self.draw()
            lit = {x for x, _ in self.where(theme.ACTIVE, seen.names)}
            plain = {x for x, _ in self.where(theme.TEXT, seen.names)} - {258}
            for index, label in enumerate(labels):
                # A label is in one of the two colours, all of it.
                self.assertEqual((bool(lit & set(label)), bool(plain & set(label))),
                                 (index == played, index != played), (played, index))
            self.assertTrue(lit | plain <= set(labels[0]) | set(labels[1]))
        # The playhead of the second variant, half way into it, is a line at 0.75 s and nowhere else.
        self.view.played, self.view.playhead = 1, self.view.clock(1, 0.25)
        seen = self.draw()
        self.assertEqual({(x, y) for x, y in self.where(theme.ACTIVE, (0, 0, *SIZE)) if x not in labels[1]},
                         {(358, y) for y in range(20, 320)})
        self.assertTrue(seen.x(0.5) <= 358 < seen.x(1.0))

    def test_the_zones_of_a_held_sound_are_tinted_each_in_its_colour_under_its_label(self) -> None:
        self.view.show(HELD, rows(0.25, 0.5, 0.25), RATE, theme.SOUND, "held")
        seen = self.draw()
        # The three meet at x 158 and x 358, where a line runs down the labels and the grid. d4 is a white key,
        # in the row from y 106, and the key under it a black one.
        white, black = seen.y(62) + 3, seen.y(61) + 3
        self.assertEqual((seen.grid, seen.x(0.25), seen.x(0.75), white, black),
                         ((58, 40, 400, 195), 158, 358, 109, 120))
        # From its first pixel to its last, a zone has its tint on the rows of the grid and in the box of the wave.
        for zone, ends in enumerate(((59, 157), (159, 357), (359, 456))):
            for x in ends:
                self.assertEqual((self.at(x, white), self.at(x, black), self.at(x, 245)),
                                 (tint(theme.RAISED, zone), tint(theme.BACKGROUND, zone),
                                  tint(theme.BACKGROUND, zone)), (zone, x))
        self.assertEqual({self.at(x, white) for x in (158, 358)}, {theme.TEXT})
        # Each tint is its ground a little of the way to the colour of its zone, and the three are told apart.
        tints = [tint(theme.RAISED, zone) for zone in range(3)]
        self.assertEqual((tints[0], len({theme.RAISED, *tints})), (mix(theme.RAISED, ZONES[0], 0.14), 4))
        self.assertEqual(ZONES, (theme.GOOD, theme.MUSTARD, theme.SALMON))
        # START, LOOP and END are written in those colours, each from where its zone starts.
        for zone, (start, word) in enumerate(((58, "START"), (158, "LOOP"), (358, "END"))):
            inked = {x for x, _ in self.where(ZONES[zone], seen.names)}
            self.assertTrue(start + 3 <= min(inked) <= start + 6, (word, sorted(inked)))
            self.assertTrue(start + Font.width(word) <= max(inked) + 1 <= start + 3 + Font.width(word), word)
        self.assertEqual({x for x, _ in self.where(theme.TEXT, seen.names)}, {158, 358})
        # A plain sound has no tint anywhere.
        every = {tint(ground, zone) for ground in (theme.RAISED, theme.BACKGROUND) for zone in range(3)}
        self.view.show(ONE, [wave()], RATE, theme.SOUND)
        self.draw()
        self.assertEqual(self.colours(RECT) & every, set())

    def test_coin_of_the_arcade_example_is_its_two_bars_over_the_wave_of_its_render(self) -> None:
        view, seen, shapes, (samples,) = self.arcade("coin")
        # b5 and then e6, in one voice with the instrument chime, and no labels.
        (label, _, shape), = shapes
        self.assertEqual((label, [found.pitch for found in shape.bars], seen.labels, view.seconds),
                         ("", [83, 88], False, shape.seconds))
        bars = [seen.bar(found.start, found.seconds, found.pitch) for found in shape.bars]
        allowed = {(x + column, y + row) for x, y, w, h in bars for column in range(w) for row in range(h)}
        painted = self.where(theme.SOUND, seen.grid)
        self.assertLessEqual(painted, allowed)
        self.assertEqual([(x + 1, y + 1) in painted for x, y, _, _ in bars], [True, True])
        self.assertIs(view.bar_at((bars[1][0] + 1, bars[1][1] + 1)), shape.bars[1])
        # The wave is the render's own, a column a pixel, and nothing of it clips.
        (left, columns), = view.columns()
        self.assertEqual((left, len(columns)), (48, 552))
        self.assertTrue(np.array_equal(columns, soundview.columns(samples, 552)))
        self.assertEqual((self.where(theme.SOUND, seen.wave), self.where(theme.FAULT, seen.wave)),
                         (self.wave_of(seen, left, columns), set()))

    def test_the_parts_of_explosion_and_the_variants_of_blip_are_labelled_with_a_line_between_them(self) -> None:
        view, seen, shapes, buffers = self.arcade("explosion")
        grid = seen.grid
        # Its two parts are added up into one buffer, which is the one wave, from the start of the clock.
        self.assertEqual((len(shapes), len(buffers), [left for left, _ in view.columns()], view.kind),
                         (2, 1, [48], "parts"))
        # The second part starts a fiftieth of a second after the first: its line runs down the labels and the
        # grid, and each label stands just after where its part starts.
        self.assertEqual((seen.labels, seen.names, grid[:3], seen.x(0.02)),
                         (True, (48, 10, 552, 10), (48, 20, 552), 65))
        self.assertEqual(self.where(theme.TEXT, grid), {(65, y) for y in range(20, 20 + grid[3])})
        inked = {x for x, _ in self.where(theme.TEXT, seen.names)}
        self.assertEqual((min(inked), 65 in inked, max(inked)), (51, True, 72))
        # The thump is c3 from the start, and the rumble c5 from the line on.
        self.assertEqual((self.at(51, seen.y(48) + 4), self.at(68, seen.y(72) + 4)), (theme.SOUND, theme.SOUND))

        view, seen, _, buffers = self.arcade("blip")
        grid = seen.grid
        # Three variants of a tenth of a second, with a fifth of that between two: the clock is 0.34 s long.
        starts = [seen.x(second) for second in (0.0, 0.12, 0.24)]
        self.assertEqual((round(view.seconds, 9), starts), (0.34, [48, 243, 438]))
        self.assertEqual({x for x, _ in self.where(theme.TEXT, grid)}, {243, 438})
        inked = {x for x, _ in self.where(theme.TEXT, seen.names)}
        self.assertEqual([bool(inked & set(range(start + 3, start + 8))) for start in starts], [True, True, True])
        self.assertEqual(inked - {243, 438} - {x for start in starts for x in range(start + 3, start + 8)}, set())
        # Each has its one note where its own clock starts, and under it the wave of its own buffer.
        self.assertEqual([self.at(start + 3, seen.y(pitch) + 4) for start, pitch in zip(starts, (81, 84, 88))],
                         [theme.SOUND] * 3)
        waves = view.columns()
        self.assertEqual(([left for left, _ in waves], [len(buffer) for buffer in buffers]), (starts, [4410] * 3))
        self.assertEqual([left + len(columns) for left, columns in waves],
                         [seen.x(second) for second in (0.1, 0.22, 0.34)])


if __name__ == "__main__":
    unittest.main()
