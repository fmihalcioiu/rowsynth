"""Tests for rowsynth.gui.wave: the sums of the view of a sound, its clock, its gestures, and what it draws."""
from __future__ import annotations

import copy
import math
import os
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from rowsynth.bank import entry_kind, render_entry
from rowsynth.errors import RowSynthError
from rowsynth.gui import layout, soundedit, soundview, sprites, theme
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.font import Font
from rowsynth.gui.kit import VELOCITY_HELP, told
from rowsynth.gui.roll import BOX, DARKER, LIGHTER, MOVE, PICK, PLACE, REMOVE, STRETCH, mix
from rowsynth.gui.soundview import Bar, Shape
from rowsynth.gui.ui import Event, Ui
from rowsynth.gui.wave import (ACROSS_HELP, ALONG_HELP, FLAG, FLAG_W, GRIP, LABEL, LIMIT_AIR, MARK, NO_PITCH,
                               OPEN_LEAST, PART, PLUS_GAP, ROOM_LEAST, RULER, RULER_HELP, STALE, VARIANT_GAP,
                               VELOCITY_DONE, WHEEL_HELP, ZONES,
                               ZOOM_MOST, Map, SoundView, Span, bars, begins, beside, called, echo_colour, joined,
                               limit_lines, mark_lines, on_tick, opened, places, plus, ruler_step, span_at, spans,
                               spelt, stretch, tabs, tint, voice_colour)
from rowsynth.project import Project

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

#: A view whose grid is (58, 30, 400, 204), under a ruler from y 20, beside a keyboard from x 18, with a scroll
#: bar down its right from x 459 and one under it from y 235. The box of its wave is put right under it, as wide
#: as the grid: (58, 239, 400, 81).
RECT = (18, 20, 445, 219)
FOOT = (58, 239, 400, 81)
#: The surface it is drawn on, which is larger than it on every side, and the colour of what is not the view's.
SIZE = (470, 330)
OUTSIDE = theme.GOOD
#: The pixels of the scroll bar under the grid that are in the column of half a second: a line that marks that
#: second runs down the view and the box of the wave, and is under the bar there.
UNDER_BAR = {(258, y) for y in range(235, 239)}
#: What the view says where a press would do nothing.
PLAIN = "the notes of a sound"
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


#: The instruments of the sounds that a hand changes here. Neither fades, so a sound ends two ticks after its notes.
INSTRUMENTS = {"tone": {"wave": "tri", "release": 0}, "buzz": {"wave": "saw", "release": 0}}
#: The notes of a sound of two voices: c4 and then e4 in the first, and between them g4 in the second, each held
#: for 6 ticks. The sound is 20 ticks long, so on the grid of the view a tick is 20 pixels. With three keys kept
#: free over the notes and under them, c4 is the row from y 162, e4 that from y 118 and g4 that from y 85.
FIRST = [[0, "c4", 6], [12, "e4", 6]]
SECOND = [[6, "g4", 6]]
C4, E4, G4 = (0, 0), (0, 1), (1, 0)
#: What the help line says on the bar of c4, away from its end.
MOVE_C4 = told("pick c4, a drag moves it", "rub it out", WHEEL_HELP)


def duo(first: list, second: list) -> dict:
    """The definition of a sound of two voices with some notes in each: the first plays tone and the second buzz."""
    return {"voices": [{"inst": "tone", "notes": first}, {"inst": "buzz", "notes": second}]}


def solo(*notes: list, inst: str = "tone") -> dict:
    """The definition of a sound of one voice with some notes."""
    return {"voices": [{"inst": inst, "notes": list(notes)}]}


DUO = {"def": duo(FIRST, SECOND)}
SOLO = {"def": solo([0, "c4", 6])}
#: Two variants, of 8 ticks and of 14: the second starts 10.8 ticks after the first, which is at 0.18 s.
TWINS = {"defs": [solo([0, "c4", 6]), solo([0, "c5", 12])], "db": [0, -12]}
#: Two parts that lie over one another: the second starts a tenth of a second after the first, which is 6 ticks.
LAYERS = {"parts": [[0, {"def": solo([0, "c4", 10]), "db": -3}], [0.1, {"def": solo([0, "c5", 6], inst="buzz")}]]}
#: A held sound: its note is held for 60 ticks, of which the start and the loop take 36, and then comes its end.
HUM = {"held": {"note": solo([0, "c3", 60]), "start": 6, "loop": 30, "join": 3, "end": solo([0, "c3", 2]),
                "let_go": 4}, "db": -6}


# Every test of this file but those of `Room` opens a sound with its length as wide as the grid, however short it
# is, and none of the room after it in view: the places they hold a view to are counted from that. What a sound
# is opened with after its end is held to in `Room`.
_NO_AIR = mock.patch.multiple("rowsynth.gui.wave", OPEN_AIR=0.0, OPEN_LEAST=0.0)


def setUpModule() -> None:
    _NO_AIR.start()


def tearDownModule() -> None:
    _NO_AIR.stop()


def view_on(ui: Ui, help: str = "") -> SoundView:
    """A view in `RECT` with the box of its wave at `FOOT`, added to a Ui with its parts after it, as an owner
    adds them."""
    view = ui.add(SoundView(RECT, help))
    view.foot = FOOT
    for part in view.parts:
        ui.add(part)
    return view


def laid_out(entry: dict) -> tuple[Map, tuple[Span, ...]]:
    """An entry as a view in `RECT` lays it out when it is opened on it: the sums of that view, and the sounds of
    the entry as spans."""
    kind = entry_kind(entry)
    shapes = soundview.shapes(entry, instruments=INSTRUMENTS)
    view = SoundView(RECT)
    view.foot = FOOT
    view.show(shapes, [], 44100, theme.SOUND, kind, entry)
    laid = [(label, place, shape) for (label, _, shape), place in zip(shapes, places(shapes, kind == "defs"))]
    return view.map, spans(entry, laid)


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

    def test_the_y_of_a_pitch_and_the_pitch_at_a_y_are_opposites_wherever_the_keys_are_scrolled_to(self) -> None:
        self.assertEqual([Map(RECT).y(pitch) for pitch in (72, 71, 60)], [30, 41, 162])
        # Scrolled by half a key, the key at the upper edge is cut there: 5 of its 11 lines are over the grid.
        part = Map(RECT, top=80.5)
        self.assertEqual((part.y(81), part.y(80)), (25, 36))
        for seen in (Map(RECT), part):
            for pitch in seen.pitches():
                y = seen.y(pitch)
                self.assertEqual(seen.y(pitch - 1) - y, 11)
                self.assertEqual({seen.pitch_at(y + step) for step in range(11)}, {pitch})
        # 204 pixels are 18 keys of 11 and half of another.
        self.assertEqual((Map(RECT).pitches(), part.pitches()), (range(54, 73), range(63, 82)))

    def test_a_view_is_opened_with_its_notes_in_the_middle_of_the_keys_and_goes_no_further_than_the_keys_do(
            self) -> None:
        # As many keys over the notes as under them: g4 to c5 are 6 of the 18, with 6 keys over them.
        self.assertEqual((opened(204, 67, 72), opened(204, 60, 60)), (78.0, 68.0))
        # Notes that reach further than the grid does have their middle in the middle of it, and a grid of no
        # height has the lowest of them just under its edge.
        self.assertEqual((opened(204, 40, 80), opened(0, 60, 60)), (68.0, 59.0))
        # No key is shown over the highest that a note can be placed at, or under the lowest.
        high, low = Map(RECT, top=500.0).within(), Map(RECT, top=-500.0).within()
        self.assertEqual((high.top, high.pitches().stop - 1, low.pitches().start), (131.0, 131, -96))
        self.assertEqual(low.y(-96) + 11, 30 + 204)
        # The bar along the keys has all 228 of them, 11 pixels each, and the grid 59 keys down from the top.
        # The bar along the time has the whole clock in view until it is zoomed in.
        self.assertEqual(Map(RECT).scrolls(), ((2508, 204, 649), (400, 400, 0)))
        self.assertEqual(Map(RECT, start=0.5, zoom=4).scrolls()[1], (977, 400, 488))

    def test_the_bar_of_a_note_is_in_the_row_of_its_key_from_just_after_its_second(self) -> None:
        seen = Map(RECT)
        self.assertEqual([seen.bar(0.5, 0.25, 72), seen.bar(0.5, 0.25, 70.0)], [(259, 30, 99, 10), (259, 52, 99, 10)])
        self.assertEqual(Map(RECT, start=0.5, zoom=4, top=80.0).bar(0.75, 0.25, 67), (303, 173, 244, 10))
        # Two in a row leave a pixel free between them, and the shortest is a pixel wide.
        first, second = seen.bar(0.25, 0.25, 60), seen.bar(0.5, 0.25, 60)
        self.assertEqual((first[0] + first[2], second[0]), (seen.x(0.5), seen.x(0.5) + 1))
        self.assertEqual(seen.bar(0.5, 0.001, 72), (259, 30, 1, 10))

    def test_columns_with_a_gap_between_them_are_stretched_to_meet_and_the_others_are_left(self) -> None:
        # The second lies under the first with a gap, the third over the second, the fourth overlaps the third.
        self.assertEqual(joined([(10, 12), (20, 22), (5, 6), (6, 9)]), [(10, 12), (13, 22), (5, 19), (6, 9)])
        # A neighbour that already touches stays as it is, the one after it is brought down to meet it.
        self.assertEqual(joined([(10, 12), (13, 14), (9, 9)]), [(10, 12), (13, 14), (9, 12)])
        self.assertEqual((joined([(4, 4)]), joined([])), ([(4, 4)], []))

    def test_a_column_of_the_wave_ends_where_its_highest_and_its_lowest_sample_fall(self) -> None:
        seen = Map(RECT, foot=FOOT)
        # The box is (58, 239, 400, 81): zero is its middle line, and full scale is 2 pixels short of its outline.
        self.assertEqual((seen.wave, seen.box, seen.zero, seen.column(-1.0, 1.0), seen.column(0.0, 0.0)),
                         (FOOT, FOOT, 279, (242, 316), (279, 279)))
        self.assertEqual((seen.column(-0.5, 0.5), seen.column(0.25, 0.5), seen.column(-3.0, 2.0)),
                         ((260, 297), (260, 270), (242, 316)))
        # A box that is wider than the grid has the wave in the stretch of it that lies under the grid.
        wide = Map(RECT, foot=(18, 239, 445, 46))
        self.assertEqual((wide.wave, wide.box, wide.zero), ((58, 239, 400, 46), (18, 239, 445, 46), 262))
        # With no notes there is no such box: the wave has all there is under the ruler, and a column is as
        # much longer.
        bare = Map(RECT, notes=False, foot=FOOT)
        self.assertEqual((bare.grid, bare.wave, bare.box, bare.zero, bare.column(-1.0, 1.0)),
                         ((58, 30, 400, 0), (58, 30, 400, 209), (58, 30, 400, 209), 134, (33, 235)))

    def test_the_ruler_goes_in_tenths_of_a_second_or_in_seconds_by_what_its_numbers_have_room_for(self) -> None:
        # A number in tenths takes 24 pixels with the air round it, and a number of seconds 12.
        self.assertEqual([ruler_step(scale) for scale in (240, 239, 12, 11.9, 1.8, 1.7)], [-1, 0, 0, 1, 1, 2])
        # Zoomed far in it goes on to hundredths and to thousandths, and no further.
        self.assertEqual([ruler_step(scale) for scale in (2999, 3000, 35999, 36000, 1e9)], [-1, -2, -2, -3, -3])
        # By the width: a second on 400 pixels is marked in tenths, and on 160 pixels at its start alone. Four
        # seconds on 400 pixels are marked in seconds.
        self.assertEqual((Map(RECT).marks(), Map((18, 20, 205, 219)).marks(), Map(RECT, seconds=4.0).marks()),
                         (list(range(10)), [0], [0, 1, 2, 3]))
        # Only the marks whose line is on the grid are given: here those of 0.5 to 0.9 s.
        near = Map(RECT, start=0.5, zoom=4)
        self.assertEqual((near.marks(), near.x(0.9), near.x(1.0)), ([5, 6, 7, 8, 9], 449, 547))

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

    def test_a_part_starts_on_a_tick_and_variants_have_a_line_of_the_longest_time_each(self) -> None:
        # A tick is a sixtieth of a second: the nearest one, written to the millionth, and none before 0.
        self.assertEqual([on_tick(second) for second in (0.1, 0.11, 0.0249, 0.026, -1.0)],
                         [0.1, 0.116667, 0.016667, 0.033333, 0.0])
        laid = [(label, place, shape) for (label, _, shape), place in zip(PAIR, places(PAIR, True))]
        # A variant has the line where its own clock comes to the limit, if that is before the next one starts.
        self.assertEqual((limit_lines(laid, "defs", 0.25), limit_lines(laid, "defs", 0.75)),
                         ([(0, 0.25), (1, 0.75)], [(1, 1.25)]))
        self.assertEqual((limit_lines(laid, "parts", 0.75), limit_lines(ONE, "def", 2), limit_lines(ONE, "def", None)),
                         ([(0, 0.75)], [(0, 2.0)], []))
        # A note the sound is expected to play has its flag the same way: one for each variant that is as long.
        marks = [{"at": 0.25, "note": "e6"}, {"at": 0.75, "note": 64}]
        self.assertEqual(mark_lines(laid, "defs", marks), [(0, 0.25), (0, 0.75), (1, 1.25)])
        self.assertEqual((mark_lines(ONE, "def", marks), mark_lines(ONE, "def", [])), ([(0, 0.25), (1, 0.75)], []))

    def test_a_place_is_in_the_stretch_that_starts_last_before_it_or_at_it(self) -> None:
        self.assertEqual([stretch([58, 258], at) for at in (57, 58, 257, 258, 1000)], [None, 0, 0, 1, 1])
        self.assertEqual((stretch([], 5), stretch([0.0, 0.5, 0.5], 0.5)), (None, 2))

    def test_tabs_that_would_lie_on_one_another_are_laid_side_by_side_and_the_others_stay(self) -> None:
        # Two that start together: the first has its place, and the second stands just after it.
        self.assertEqual(beside([58, 58], [11, 11]), [58, 69])
        # Three, in the order they are given, each as wide as it is.
        self.assertEqual(beside([58, 58, 58], [11, 17, 11]), [58, 69, 86])
        # Two that are near and clear of one another stay where they are, and so does one alone.
        self.assertEqual((beside([58, 69], [11, 11]), beside([80, 58], [11, 11]), beside([58], [11]), beside([], [])),
                         ([58, 69], [80, 58], [58], []))
        # One that starts under where another was laid stands after that one. The one that starts sooner is
        # laid first, whichever of them it is.
        self.assertEqual((beside([58, 58, 75], [11, 11, 11]), beside([60, 58], [11, 11])), ([58, 69, 80], [69, 58]))
        # In the strip of the names it is the tabs of parts that are laid so, with the sign after the last.
        seen = Map(RECT, labels=True)
        laid = [(label, 0.25, Shape((), 0.5, ())) for label in ("1", "2", "3")]
        self.assertEqual((tabs(seen, laid, True), plus(seen, laid, True)),
                         ([(158, 30, 11, 10), (169, 30, 11, 10), (180, 30, 11, 10)], (193, 30, 11, 10)))
        self.assertEqual((tabs(seen, laid), plus(seen, laid)), ([(158, 30, 11, 10)] * 3, (171, 30, 11, 10)))


class Hands(unittest.TestCase):
    def test_the_sounds_of_an_entry_lie_where_their_shapes_do_each_with_room_up_to_the_next(self) -> None:
        _, (one,) = laid_out(DUO)
        self.assertEqual((one.at, one.until, one.defn, one.echo), (0.0, math.inf, DUO["def"], ()))
        # A variant has room up to where the next starts, and the last has no end to it.
        _, twins = laid_out(TWINS)
        self.assertEqual([(round(span.at, 9), round(span.until, 9)) for span in twins], [(0.0, 0.18), (0.18, math.inf)])
        self.assertEqual([span.defn for span in twins], TWINS["defs"])
        # Parts lie over one another. The note of a held sound lies under its start and its loop, and its end
        # under the third shape.
        _, layers = laid_out(LAYERS)
        _, hum = laid_out(HUM)
        self.assertEqual([(span.at, span.until) for span in layers], [(0.0, math.inf), (0.1, math.inf)])
        self.assertEqual([(span.at, span.until, span.defn) for span in hum],
                         [(0.0, 0.6, HUM["held"]["note"]), (0.6, math.inf, HUM["held"]["end"])])
        # A span has the repeats of its sound's echo, and what it holds is a copy: the entry is not the hand's.
        ring = {"def": {**solo([0, "a4", 6]), "echo": {"time": 0.1}}}
        _, (echoed,) = laid_out(ring)
        self.assertEqual((echoed.echo, echoed.defn is ring["def"]),
                         (soundview.shapes(ring, instruments=INSTRUMENTS)[0][2].echo, False))
        self.assertGreater(len(echoed.echo), 1)
        with self.assertRaises(RowSynthError) as refused:
            spans(TWINS, [])
        self.assertEqual(str(refused.exception),
                         "entry: must be the entry the shapes were made of: it has 2, and there are 0")
        # A second is in the room of one variant. Of the parts that have it, it is in the one the hand was in
        # last, and else in the one that starts last.
        self.assertEqual([span_at(twins, second) for second in (0.0, 0.17, 0.18, 0.4)], [0, 0, 1, 1])
        self.assertEqual([span_at(layers, second, last) for second, last in ((0.05, 1), (0.15, 0), (0.15, 1),
                                                                             (0.15, 5))], [0, 0, 1, 1])
        self.assertIsNone(span_at((), 0.1))
        self.assertEqual([called(kind, index) for kind, index in (("def", 0), ("defs", 1), ("parts", 0), ("held", 0),
                                                                  ("held", 1))],
                         ["", "variant 2", "part 1", "the note", "the end"])
        self.assertEqual([spelt(pitch) for pitch in (60, 61, 70, 59, -96, 131)],
                         ["c4", "c#4", "a#4", "b3", "c-9", "b9"])

    def test_the_bar_of_a_note_is_at_its_ticks_and_is_cut_where_the_room_of_its_sound_ends(self) -> None:
        seen, (one,) = laid_out(DUO)
        self.assertEqual((seen.scale, seen.top), (1200.0, 72.0))
        self.assertEqual(bars(seen, one), [(C4, (59, 162, 119, 10), True), (E4, (299, 118, 119, 10), True),
                                           (G4, (179, 85, 119, 10), True)])
        # The tick of an x is counted from where the sound starts, and no x is before its tick 0.
        self.assertEqual([round(seen.tick_at(x), 9) for x in (58, 68, 258)], [0.0, 0.5, 10.0])
        self.assertEqual([round(seen.tick_at(x, 0.1), 9) for x in (58, 178, 198)], [0.0, 0.0, 1.0])
        # The note of the held sound goes on past the 36 ticks of its start and its loop, and its bar does not.
        seen, (note, end) = laid_out(HUM)
        self.assertEqual((bars(seen, note), bars(seen, end)),
                         ([((0, 0), (59, 128, 359, 10), False)], [((0, 0), (419, 128, 19, 10), True)]))
        # A note that starts past the room of its sound has no bar.
        late = replace(note, defn=solo([40, "c3", 6]))
        self.assertEqual(bars(seen, late), [])

    def test_a_press_on_a_bar_moves_stretches_or_picks_its_note_and_where_no_bar_is_places_or_boxes(self) -> None:
        seen, laid = laid_out(DUO)
        # The bar of c4 is 119 pixels from x 59: its last three are for stretching.
        self.assertEqual([begins(seen, laid, (x, 166)) for x in (59, 174, 175, 177)],
                         [(MOVE, 0, C4), (MOVE, 0, C4), (STRETCH, 0, C4), (STRETCH, 0, C4)])
        ctrl, shift = frozenset({"ctrl"}), frozenset({"shift"})
        self.assertEqual((begins(seen, laid, (100, 166), 1, ctrl), begins(seen, laid, (177, 166), 3)),
                         ((PICK, 0, C4), (REMOVE, 0, C4)))
        # Where no bar is: also on the line of pixels a bar leaves free under it, and just after its end.
        for pos in ((100, 100), (100, 172), (178, 166)):
            self.assertEqual([begins(seen, laid, pos, button, mods) for button, mods in ((1, frozenset()), (1, shift),
                                                                                         (1, ctrl | shift), (3, ctrl))],
                             [(PLACE, 0, None), (BOX, 0, None), (BOX, 0, None), (REMOVE, 0, None)], pos)
        # The middle button begins nothing, and neither does a press on the keys, the ruler or the wave.
        nothing = ("", None, None)
        self.assertEqual([begins(seen, laid, (100, 166), 2), begins(seen, laid, (20, 166)),
                          begins(seen, laid, (100, 25)), begins(seen, laid, (100, 280))], [nothing] * 4)
        # The bar of the held note is cut at x 418: its last pixel moves it. The end of the sound is a sound
        # of its own, whose bar is stretched at its own end.
        seen, laid = laid_out(HUM)
        self.assertEqual([begins(seen, laid, (x, 132)) for x in (417, 419, 437)],
                         [(MOVE, 0, (0, 0)), (MOVE, 1, (0, 0)), (STRETCH, 1, (0, 0))])
        self.assertEqual([begins(seen, laid, (x, 100)) for x in (416, 420)], [(PLACE, 0, None), (PLACE, 1, None)])
        # Of two parts that lie over one another, a press where no bar is is in the one the hand was in last.
        seen, laid = laid_out(LAYERS)
        at = seen.x(0.15)
        self.assertEqual([begins(seen, laid, (at, 100), last=last) for last in (0, 1)],
                         [(PLACE, 0, None), (PLACE, 1, None)])
        self.assertEqual(begins(seen, laid, (seen.x(0.05), 100), last=1), (PLACE, 0, None))


class View(unittest.TestCase):
    def setUp(self) -> None:
        self.ui = Ui()
        self.view = view_on(self.ui, PLAIN)
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
        # In or out, the second under the mouse stays under it, to the pixel, here and at another place: in the
        # box of the wave, where the wheel is the view's as well.
        for step in (1, 1, 1, -1, 1, -1, -1):
            self.wheel(step, "ctrl")
            self.assertLess(abs(self.view.map.second_at(258) - 0.5), 1 / self.view.map.scale)
        before = self.view.map.second_at(390)
        self.wheel(3, "ctrl", pos=(390, 250))
        self.assertLess(abs(self.view.map.second_at(390) - before), 1 / self.view.map.scale)
        self.assertEqual(self.view.zoom, 5)
        self.wheel(100, "ctrl")
        self.assertEqual(self.view.zoom, ZOOM_MOST)
        # Ctrl is read before shift.
        self.wheel(-100, "ctrl", "shift")
        self.assertEqual(self.seen(), (0.0, 0))
        # Over the keys, it is the second at the left edge of the grid that stays.
        self.wheel(2, "ctrl")
        before = self.view.map.second_at(58)
        self.wheel(2, "ctrl", pos=(20, 100))
        self.assertEqual(self.view.zoom, 4)
        self.assertLess(abs(self.view.map.second_at(58) - before), 1 / self.view.map.scale)

    def test_the_wheel_goes_along_the_keys_and_with_shift_along_the_time_and_stops_where_either_ends(self) -> None:
        # The view was opened on its notes, g4 to c5. Alone the wheel goes two keys a step, and up for a step up.
        self.assertEqual((self.view.top, self.view.keys.top), (78.0, 78.0))
        self.assertTrue(self.wheel(1))
        self.assertEqual((self.view.top, self.view.keys.top, self.seen()), (80.0, 80.0, (0.0, 0)))
        self.assertTrue(self.wheel(-3))
        self.assertEqual(self.view.top, 74.0)
        # Over the keyboard, either scroll bar and the wave it is the view's as well: they leave it to the view.
        for pos in ((20, 100), (460, 100), (300, 237), (300, 280)):
            self.assertTrue(self.wheel(1, pos=pos), pos)
        self.assertEqual(self.view.top, 82.0)
        # It goes no further than the keys do, either way.
        self.wheel(100)
        self.assertEqual((self.view.top, self.view.map.pitches().stop), (131.0, 132))
        self.wheel(-200)
        self.assertEqual(self.view.map.pitches().start, -96)
        # With shift it goes along the time: nowhere while the whole clock is in view.
        self.assertTrue(self.wheel(-1, "shift"))
        self.assertEqual(self.seen(), (0.0, 0))
        self.wheel(1, "ctrl")
        # A step is 36 pixels, at 500 to the second, and the view goes no further than the clock does.
        self.assertTrue(self.wheel(-1, "shift"))
        self.assertEqual(self.seen(), (0.172, 1))
        self.wheel(-1, "shift")
        self.assertEqual((self.seen(), self.view.map.x(1.0)), ((0.2, 1), 458))
        self.wheel(100, "shift")
        self.assertEqual(self.seen(), (0.0, 1))
        # A key is not the view's while nothing can be changed, and a wheel off the view and its wave is nobody's.
        self.assertEqual((self.ui.handle(Event("key", key="space")), self.wheel(1, "ctrl", pos=(5, 5)), self.seen()),
                         (False, False, (0.0, 1)))

    def test_the_scroll_bars_show_what_is_in_view_and_a_drag_of_one_moves_the_view(self) -> None:
        along, across, keys = self.view.along, self.view.across, self.view.keys
        self.assertEqual((keys.rect, along.rect, across.rect),
                         ((18, 30, 40, 204), (459, 30, 4, 204), (58, 235, 400, 4)))
        self.assertEqual((along.help, across.help), (ALONG_HELP, ACROSS_HELP))
        # The keys are far longer than the grid, so the bar along them has a thumb. The whole clock is in view,
        # so the bar along the time has none, and a press on it does nothing.
        x, y, _, high = along.thumb()
        self.assertEqual((across.thumb(), self.ui.handle(Event("down", (300, 237), button=1))), (None, False))
        # The thumb dragged down takes the view down the keys, and the keyboard with it.
        self.ui.handle(Event("down", (x + 1, y + 2), button=1))
        self.ui.handle(Event("move", (x + 1, y + 22)))
        self.ui.handle(Event("up", (x + 1, y + 22), button=1))
        self.assertTrue(self.view.top < 78.0 and along.thumb()[1] == y + 20)
        self.assertEqual(keys.top, self.view.top)
        # Zoomed in, the bar along the time has a thumb as well, and a press after it goes on by what is in view.
        self.wheel(3, "ctrl", pos=(58, 100))
        left, _, wide, _ = across.thumb()
        self.assertEqual((self.view.start, left), (0.0, 58))
        self.assertTrue(self.ui.handle(Event("down", (456, 237), button=1)))
        self.ui.handle(Event("up", (456, 237), button=1))
        seen = self.view.map
        self.assertEqual((seen.x(1.0), across.thumb()[0] + wide), (458, 458))

    def test_at_the_start_the_whole_fits_and_shown_again_the_view_stays_until_it_is_fitted(self) -> None:
        seen = self.view.map
        self.assertEqual((self.view.seconds, self.seen(), seen.x(0.0), seen.x(1.0)), (1.0, (0.0, 0), 58, 458))
        self.assertEqual((seen.top, seen.labels, seen.notes), (78.0, False, True))
        self.wheel(4, "ctrl")
        self.wheel(-3, "shift")
        self.wheel(-2)
        moved = self.seen()
        self.assertNotEqual((moved, self.view.top), ((0.0, 0), 74.0))
        self.assertEqual(self.view.top, 74.0)
        # The render came, or the file changed outside: other notes on the same clock. The view stays where it
        # was left, and so do the keys: they are not fitted to the notes.
        self.view.show(one(bar(0.0, 0.5, 60)), [wave()], RATE, theme.SONG)
        self.assertEqual((self.seen(), self.view.colour, self.view.map.top), (moved, theme.SONG, 74.0))
        # The clock is as long as the longer of the shapes and the render, and a shorter one ends sooner.
        self.view.show(ONE, [np.zeros(2 * RATE)], RATE, theme.SOUND)
        self.assertEqual((self.view.seconds, self.seen()), (2.0, moved))
        self.view.show([("", 0.0, Shape((), 0.5, ()))], [], RATE, theme.SOUND)
        self.assertEqual((self.view.seconds, self.view.zoom, self.view.map.x(0.5)), (0.5, 4, 458))
        # Fitted, the whole clock is in view at once, and the sound that is shown next is opened on its notes:
        # c4 alone has 8 keys over it. After that the keys stay again.
        self.view.fit()
        self.assertEqual((self.seen(), self.view.map.x(0.5), self.view.top), ((0.0, 0), 458, 74.0))
        self.view.show(one(bar(0.0, 0.5, 60)), [], RATE, theme.SOUND)
        self.assertEqual((self.view.top, self.view.map.y(60)), (68.0, 118))
        self.view.show(ONE, [], RATE, theme.SOUND)
        self.assertEqual(self.view.top, 68.0)

    def test_the_help_says_what_note_or_what_moment_of_the_wave_the_mouse_is_on(self) -> None:
        self.assertEqual((self.view.help, self.ui.help), (PLAIN, ""))
        for pos, line in (((170, 155), "g4, 0.15 s, instrument zap"),
                          ((300, 100), "c5, 0.25 s"),
                          # The copy an echo brings of g4 is no note, and neither is the ruler.
                          ((270, 155), PLAIN),
                          ((170, 25), PLAIN),
                          # On the keyboard it is the key that speaks, and beside the box of the wave nothing.
                          ((20, 155), told("hear g4", wheel=WHEEL_HELP)),
                          ((20, 279), ""),
                          # The wave has its second, its level, and its pitch: these waves are too coarse for one.
                          ((118, 250), f"0.15 s, -6.0 dB, {NO_PITCH}"),
                          ((178, 300), f"0.30 s, -12.0 dB, {NO_PITCH}"),
                          ((318, 279), f"0.65 s, silent, {NO_PITCH}")):
            self.ui.handle(Event("move", pos))
            self.assertEqual(self.ui.help, line, pos)
        self.assertEqual((self.view.bar_at((170, 155)), self.view.bar_at((270, 155))), (ONE[0][2].bars[0], None))
        # Past the end of the samples there is no level to tell.
        self.view.show(ONE, [wave()[:200]], RATE, theme.SOUND)
        self.assertEqual(self.ui.help, PLAIN)
        self.view.help = "another line"
        self.assertEqual(self.ui.help, "another line")

    def test_the_help_on_the_wave_says_the_pitch_there_with_the_name_of_its_note(self) -> None:
        rate = 44100
        at = np.arange(rate // 2) / rate
        # Half a second of the a of 440 Hz, as `measure.pitch_at` reads it, and after it silence.
        tone = np.concatenate((0.5 * np.sin(2 * np.pi * 440 * at), np.zeros(rate // 4))).astype(np.float32)
        self.view.show(one(bar(0.0, 0.5, 69)), [tone], rate, theme.SOUND)
        for pos, line in (((158, 280), "0.25 s, -6.0 dB, 439.5 Hz, a4"), ((318, 250), f"0.65 s, silent, {NO_PITCH}")):
            self.ui.handle(Event("move", pos))
            self.assertEqual(self.ui.help, line)
        # An octave higher it is the a of the next octave.
        high = (0.5 * np.sin(2 * np.pi * 880 * at)).astype(np.float32)
        self.view.show(one(bar(0.0, 0.5, 81)), [high], rate, theme.SOUND)
        self.ui.handle(Event("move", (158, 280)))
        self.assertTrue(self.ui.help.endswith(" Hz, a5"), self.ui.help)

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
        # On its label the help says so. On its wave it says the moment and the level of that wave first, as for
        # any sound, and there is no level where no wave is.
        for pos, line in (((300, 35), "press: pick variant 2"), ((60, 35), "press: pick variant 1"),
                          ((300, 280), f"0.60 s, -12.0 dB, {NO_PITCH}   press: pick variant 2"),
                          ((100, 280), f"0.10 s, -6.0 dB, {NO_PITCH}   press: pick variant 1"),
                          ((238, 280), "press: pick variant 1")):
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
                          self.ui.help), (1, False, PLAIN))

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


class Edits(unittest.TestCase):
    def setUp(self) -> None:
        self.ui = Ui()
        self.view = view_on(self.ui, PLAIN)
        self.edits: list[tuple[int, dict, str]] = []
        self.heard: list[tuple[object, float]] = []
        self.told: list[str] = []
        # Whether the owner saves what it is handed. One that does not answers false.
        self.saves = True
        self.view.edited = self.save
        self.view.sounded = lambda inst, pitch: self.heard.append((inst, pitch))
        self.view.refused = self.told.append
        self.show(DUO)

    def show(self, entry: dict | None = None, other: bool = True) -> None:
        """Show the view an entry, or again the one it has, with no wave under it.

        An entry that is given is another sound, which the view is opened on as an owner opens it: but where it
        is not `other`, and is the sound there was as a change left it.
        """
        if entry is not None and other:
            self.view.fit()
        self.entry = self.entry if entry is None else copy.deepcopy(entry)
        shapes = soundview.shapes(self.entry, instruments=INSTRUMENTS)
        self.view.show(shapes, [], 44100, theme.SOUND, entry_kind(self.entry), self.entry)

    def save(self, index: int, defn: dict, label: str) -> bool:
        """What an owner does with an edit: it is kept here, put in the entry, and the view is shown the entry."""
        self.edits.append((index, defn, label))
        if self.saves:
            self.show(soundedit.with_sound(self.entry, index, defn, INSTRUMENTS), other=False)
        return self.saves

    def at(self, tick: float, pitch: int, sound: int = 0) -> tuple[int, int]:
        """A pixel of the view in the middle of a tick of a sound of the entry, and of a key."""
        seen, (_, laid) = self.view.map, laid_out(self.entry)
        return seen.x(laid[sound].at + (tick + 0.5) / 60), seen.y(pitch) + 4

    def end(self, pick: tuple[int, int], sound: int = 0) -> tuple[int, int]:
        """The last pixel of the bar of a note of a sound of the entry, as the view has it."""
        seen, (_, laid) = self.view.map, laid_out(self.entry)
        x, y, w, _ = next(rect for found, rect, _ in bars(seen, laid[sound]) if found == pick)
        return x + w - 1, y + 4

    def mouse(self, kind: str, pos: tuple[int, int], button: int = 1, *mods: str) -> bool:
        """Hand the Ui what the mouse did, some keys held. Whether it was used."""
        return self.ui.handle(Event(kind, pos, button=0 if kind == "move" else button, mods=frozenset(mods)))

    def drag(self, *stops: tuple[int, int], button: int = 1, mods: tuple[str, ...] = ()) -> None:
        """Press at the first position, move to each of the others in turn, and release at the last."""
        self.mouse("down", stops[0], button, *mods)
        for stop in stops[1:]:
            self.mouse("move", stop, button, *mods)
        self.mouse("up", stops[-1], button, *mods)

    def key(self, key: str, *mods: str) -> bool:
        """Hand the view a key, as the Ui does while the view has the focus. Whether it is one of the view's."""
        return self.view.key(Event("key", key=key, mods=frozenset(mods)))

    def notes(self, voice: int = 0) -> list:
        """The notes of a voice of the sound the hand is in, as the view draws them."""
        return self.view.drawn["voices"][voice]["notes"]

    def test_a_click_places_a_note_of_six_ticks_in_the_picked_voice_and_a_drag_one_as_long_as_it_goes(self) -> None:
        self.assertEqual((self.view.grid, self.view.voice, self.view.sound, self.view.drawn), (1, 0, 0, DUO["def"]))
        self.assertTrue(self.mouse("down", self.at(8, 62)))
        self.assertEqual((self.edits, self.heard, self.view.busy), ([], [], True))
        self.mouse("up", self.at(8, 62))
        placed = duo([*FIRST, [8, "d4", 6]], SECOND)
        self.assertEqual((self.edits, self.heard), ([(0, placed, "note placed")], [("tone", 62)]))
        self.assertEqual((self.view.drawn, self.view.picked, self.view.busy), (placed, frozenset(), False))
        # In the second voice, dragged from tick 2 into tick 4, it is 3 ticks long: a black key is spelt sharp.
        self.view.voice = 1
        self.drag(self.at(2, 70), self.at(4, 65))
        self.assertEqual((self.notes(1), self.heard[1:]), ([*SECOND, [2, "a#4", 3]], [("buzz", 70)]))
        # A click then places a note of that length. Dragged back before its start, a note is one step long.
        self.drag(self.at(15, 60))
        self.drag(self.at(10, 58), self.at(3, 58))
        self.assertEqual(self.notes(1)[2:], [[15, "c4", 3], [10, "a#3", 1]])
        # With steps of 4 ticks it begins where its step does, and ends with the step the mouse is in.
        self.view.grid = 4
        self.drag(self.at(9, 65), self.at(14, 65))
        self.assertEqual((self.notes(1)[4:], self.notes(0)), ([[8, "f4", 8]], placed["voices"][0]["notes"]))
        self.assertEqual([(index, label) for index, _, label in self.edits], [(0, "note placed")] * 5)
        # A sound with fewer voices than the picked one takes the note in its last.
        self.show(SOLO)
        self.drag(self.at(4, 62))
        self.assertEqual((self.notes(0), self.heard[-1]), ([[0, "c4", 6], [4, "d4", 8]], ("tone", 62)))

    def test_a_drag_moves_the_picked_notes_and_two_of_one_voice_that_then_overlap_are_both_kept(self) -> None:
        # c4 goes 12 ticks on and 4 keys up, onto the e4 of its own voice.
        self.mouse("down", self.at(1, 60))
        self.mouse("move", self.at(13, 64))
        moved = duo([[12, "e4", 6], [12, "e4", 6]], SECOND)
        # While it is dragged the view draws what the release would give, and nothing is handed on.
        self.assertEqual((self.view.drawn, self.view.picked, self.ui.help, self.edits), (moved, {C4}, MOVE_C4, []))
        self.mouse("up", self.at(13, 64))
        self.assertEqual((self.edits, self.view.drawn, self.view.picked), ([(0, moved, "note moved")], moved, {C4}))
        # Picked notes of two voices move together, and stop together at tick 0: g4 is 6 ticks from it.
        self.show(DUO)
        self.drag(self.at(13, 64), mods=("ctrl",))
        self.drag(self.at(7, 67), mods=("ctrl",))
        self.drag(self.at(7, 67), (60, self.at(7, 69)[1]))
        both = duo([FIRST[0], [6, "f#4", 6]], [[0, "a4", 6]])
        self.assertEqual((self.edits[1:], self.view.picked), ([(0, both, "2 notes moved")], {E4, G4}))

    def test_a_drag_from_the_end_of_a_bar_stretches_the_picked_notes_and_none_is_shorter_than_a_step(self) -> None:
        self.mouse("move", self.end(C4))
        self.assertEqual(self.ui.help, told("pick c4, a drag makes it longer or shorter", "rub it out", WHEEL_HELP))
        # From tick 5, where c4 ends, into tick 8.
        self.drag(self.end(C4), self.at(8, 60))
        self.assertEqual(self.edits, [(0, duo([[0, "c4", 9], FIRST[1]], SECOND), "note stretched")])
        self.assertEqual(self.view.picked, {C4})
        self.drag(self.end(C4), (60, 166))
        self.assertEqual(self.notes(0), [[0, "c4", 1], FIRST[1]])
        # With e4 picked as well, both are 2 ticks longer, and a click then places a note as long as the one
        # that was under the hand.
        self.drag(self.at(13, 64), mods=("ctrl",))
        self.drag(self.end(E4), self.at(19, 64))
        self.assertEqual((self.notes(0), self.edits[-1][2], self.view.picked),
                         ([[0, "c4", 3], [12, "e4", 8]], "2 notes stretched", {C4, E4}))
        self.drag(self.at(3, 70))
        self.assertEqual(self.notes(0)[2:], [[3, "a#4", 8]])

    def test_the_right_button_removes_the_note_under_it_and_dragged_every_note_it_passes(self) -> None:
        self.drag(self.at(7, 67), button=3)
        self.assertEqual((self.edits, self.view.picked), ([(0, duo(FIRST, []), "note removed")], frozenset()))
        # From c4 in a straight line to e4, which does not pass g4: the two are one edit.
        self.show(DUO)
        self.drag(self.at(1, 60), self.at(13, 64), button=3)
        self.assertEqual(self.edits[1:], [(0, duo([], SECOND), "2 notes removed")])
        # Where no bar is, the right button removes nothing.
        self.drag(self.at(10, 70), self.at(12, 70), button=3)
        self.assertEqual((len(self.edits), self.heard), (2, []))

    def test_ctrl_adds_to_the_picked_notes_and_a_box_and_ctrl_a_take_those_of_the_picked_voice(self) -> None:
        for pos, picked in ((self.at(1, 60), {C4}), (self.at(7, 67), {C4, G4}), (self.at(1, 60), {G4})):
            self.drag(pos, self.at(10, 70), mods=("ctrl",))
            self.assertEqual(self.view.picked, picked)
        # A press without ctrl on a note that is not picked picks it alone.
        self.drag(self.at(13, 64))
        self.assertEqual(self.view.picked, {E4})
        # A box from tick 0 of a4 to tick 19 of a#3 touches all three notes: it picks those of the picked voice.
        self.assertTrue(self.mouse("down", self.at(0, 69), 1, "shift"))
        self.mouse("move", self.at(19, 58), 1, "shift")
        self.assertEqual((self.view.picked, self.ui.help),
                         ({E4}, told("drag a box round the notes of voice 1 to pick", wheel=WHEEL_HELP)))
        self.mouse("up", self.at(19, 58), 1, "shift")
        self.assertEqual(self.view.picked, {C4, E4})
        self.view.voice = 1
        self.drag(self.at(0, 69), self.at(19, 58), mods=("shift",))
        self.assertEqual(self.view.picked, {G4})
        # With ctrl held as well it picks those of every voice. A box that ends in tick 5 does not touch g4.
        self.drag(self.at(0, 69), self.at(19, 58), mods=("shift", "ctrl"))
        self.assertEqual(self.view.picked, {C4, E4, G4})
        self.drag(self.at(0, 69), self.at(5, 58), mods=("shift", "ctrl"))
        self.assertEqual(self.view.picked, {C4})
        # A click with shift picks nothing. Ctrl and a picks the notes of the picked voice, with shift all.
        self.drag(self.at(0, 69), mods=("shift",))
        self.assertEqual(self.view.picked, frozenset())
        self.assertTrue(self.key("a", "ctrl"))
        self.assertEqual(self.view.picked, {G4})
        self.view.voice = 0
        self.assertEqual((self.key("a", "ctrl"), self.view.picked), (True, {C4, E4}))
        self.assertEqual((self.key("a", "ctrl", "shift"), self.view.picked), (True, {C4, E4, G4}))
        self.assertEqual((self.edits, self.heard), ([], []))

    def test_a_release_where_the_press_was_calls_nothing_and_escape_drops_a_gesture(self) -> None:
        x, y = self.at(1, 60)
        # A press on a note picks it. Released there, after straying two pixels, or back where it began, it
        # has moved nothing.
        self.drag((x, y))
        self.drag((x, y), (x + 2, y - 2))
        self.drag((x, y), self.at(9, 65), (x, y))
        self.drag(self.end(E4))
        self.assertEqual((self.view.picked, self.view.drawn, self.edits, self.heard), ({E4}, DUO["def"], [], [(
            "tone", 65), ("tone", 60)]))
        # A place whose mouse strays two pixels is a click, and three pixels are a drag: of one step here.
        x, y = self.at(8, 70)
        self.drag((x, y), (x + 2, y))
        x, y = self.at(8, 58)
        self.drag((x, y), (x + 3, y))
        self.assertEqual(self.notes(0)[2:], [[8, "a#4", 6], [8, "a#3", 1]])
        # Escape drops the gesture that is under way: the notes are drawn as they are, and its release does
        # nothing. With none under way it drops what is picked.
        self.mouse("down", self.at(1, 60))
        self.mouse("move", self.at(5, 62))
        self.assertEqual((self.view.busy, self.notes(0)[0], self.key("delete")), (True, [4, "d4", 6], False))
        self.assertEqual((self.key("escape"), self.view.busy, self.notes(0)[0]), (True, False, [0, "c4", 6]))
        self.mouse("up", self.at(5, 62))
        self.assertEqual((len(self.edits), self.view.picked), (2, {C4}))
        self.assertEqual((self.key("escape"), self.view.picked), (True, frozenset()))

    def test_the_keys_move_remove_copy_and_paste_the_picked_notes(self) -> None:
        self.drag(self.at(1, 60))
        # With the focus the Ui hands the view its keys. Up is a semitone, and with shift down an octave.
        self.assertTrue(self.ui.handle(Event("key", key="up")))
        self.assertEqual(self.edits, [(0, duo([[0, "c#4", 6], FIRST[1]], SECOND), "note moved")])
        self.assertTrue(self.key("down", "shift"))
        self.assertTrue(self.key("right"))
        self.assertEqual((self.notes(0)[0], self.view.picked), ([1, "c#3", 6], {C4}))
        # Left is a step of the grid back, and at tick 0 it moves nothing.
        self.assertEqual((self.key("left"), self.key("left"), self.notes(0)[0], len(self.edits)),
                         (True, True, [0, "c#3", 6], 4))
        # Copied, the note is pasted at the tick under the mouse, in its own voice, and is the picked one.
        self.assertTrue(self.key("c", "ctrl"))
        self.mouse("move", self.at(14, 70))
        self.assertTrue(self.key("v", "ctrl"))
        pasted = duo([[0, "c#3", 6], FIRST[1], [14, "c#3", 6]], SECOND)
        self.assertEqual((self.edits[-1], self.view.picked), ((0, pasted, "note pasted"), {(0, 2)}))
        # With the mouse off the grid it is pasted at the first tick in view.
        self.mouse("move", (20, 25))
        self.key("v", "ctrl")
        self.assertEqual((self.notes(0)[3:], self.view.picked), ([[0, "c#3", 6]], {(0, 3)}))
        self.assertTrue(self.key("delete"))
        self.assertEqual((self.edits[-1], self.view.picked), ((0, pasted, "note removed"), frozenset()))
        # With nothing picked backspace removes nothing, and other keys are not the view's.
        self.assertEqual((self.key("backspace"), self.key("space"), self.key("a"), self.key("up", "ctrl"),
                          len(self.edits)), (True, False, False, False, 7))
        # A sound with one voice refuses a note that was copied from a second voice, and the owner is told.
        self.drag(self.at(7, 67))
        self.key("c", "ctrl")
        self.show(SOLO)
        self.assertTrue(self.key("v", "ctrl"))
        self.assertEqual((self.told, len(self.edits), self.view.drawn),
                         (["1 is no voice of the sound, of which there are 1, counted from 0"], 7, SOLO["def"]))
        # Shown shapes alone, the view has no keys.
        self.view.show(ONE, [], RATE, theme.SOUND)
        self.assertEqual((self.key("delete"), self.key("escape"), self.view.drawn), (False, False, None))

    def test_a_note_is_heard_once_at_each_pitch_it_is_moved_to_and_when_it_is_placed_and_saved(self) -> None:
        self.mouse("down", self.at(1, 60))
        for stop in (self.at(1, 61), self.at(3, 61), self.at(3, 63), self.at(4, 63), self.at(4, 61)):
            self.mouse("move", stop)
        self.assertEqual(self.heard, [("tone", 61), ("tone", 63), ("tone", 61)])
        self.mouse("up", self.at(4, 61))
        # A note of the second voice is heard on the instrument of that voice. A stretch is not heard.
        self.drag(self.at(7, 67), self.at(7, 68))
        self.drag(self.end(G4), self.at(14, 68))
        self.assertEqual((self.heard[3:], len(self.edits)), ([("buzz", 68)], 3))
        # A note that is placed is not heard where the owner answers that it was not saved.
        self.saves = False
        self.drag(self.at(16, 70))
        self.assertEqual((len(self.edits), len(self.heard), self.view.drawn), (4, 4, self.entry["def"]))

    def test_the_help_line_says_what_a_press_where_the_mouse_is_would_do(self) -> None:
        self.assertEqual((self.view.help, self.ui.help), (PLAIN, ""))
        box = "drag a box round the notes"
        # The left press, the right press where it does something, and the wheel, in that order.
        for pos, mods, press, right in (
                (self.at(8, 62), (), "put a note at d4, tick 8, voice 1", ""),
                (self.at(1, 60), (), "pick c4, a drag moves it", "rub it out"),
                (self.end(G4), (), "pick g4, a drag makes it longer or shorter", "rub it out"),
                (self.at(1, 60), ("ctrl",), "pick c4 as well", "rub it out"),
                (self.at(8, 62), ("shift",), f"{box} of voice 1 to pick", ""),
                (self.at(8, 62), ("shift", "ctrl"), f"{box} to pick", ""),
                ((20, 166), (), "hear c4", "")):
            self.mouse("move", pos, 0, *mods)
            self.assertEqual(self.ui.help, told(press, right, WHEEL_HELP), (pos, mods))
        # In the ruler a press does nothing while nobody opens its menu.
        self.mouse("move", (100, 25))
        self.assertEqual(self.ui.help, PLAIN)
        self.view.asked = lambda *what: None
        self.assertEqual(self.ui.help, told(right=RULER_HELP, wheel=WHEEL_HELP))
        # A picked note is left out by ctrl, and the place of a note is told in steps of the grid.
        self.drag(self.at(1, 60))
        self.mouse("move", self.at(1, 60), 0, "ctrl")
        self.assertEqual(self.ui.help, told("take c4 out of the picked notes", "rub it out", WHEEL_HELP))
        self.view.grid, self.view.voice = 4, 1
        self.mouse("move", self.at(11, 70))
        self.assertEqual(self.ui.help, told("put a note at a#4, tick 8, voice 2", wheel=WHEEL_HELP))
        # A sound of one voice names no voice. In one of several sounds the place is told with its sound.
        self.show(SOLO)
        self.mouse("move", self.at(3, 62), 0, "shift")
        self.assertEqual(self.ui.help, told(f"{box} to pick", wheel=WHEEL_HELP))
        self.view.grid = 1
        self.show(TWINS)
        self.mouse("move", self.at(2, 74, 1))
        self.assertEqual(self.ui.help, told("put a note at d5, tick 2, variant 2", wheel=WHEEL_HELP))

    def test_the_hand_is_in_the_sound_it_pressed_in_last_on_the_grid_or_on_a_label(self) -> None:
        self.show(TWINS)
        # A click in the room of the second variant places a note in that one, counted from its own start.
        self.drag(self.at(2, 74, 1))
        second = solo([0, "c5", 12], [2, "d5", 6])
        self.assertEqual((self.edits, self.view.sound, self.view.drawn), ([(1, second, "note placed")], 1, second))
        self.assertEqual(self.entry["defs"], [TWINS["defs"][0], second])
        # A box in its room picks of its notes. A press on a note of the first variant has the hand in the
        # first, with nothing of the second picked.
        self.drag(self.at(8, 74, 1), self.at(1, 73, 1), mods=("shift",))
        self.assertEqual(self.view.picked, {(0, 1)})
        self.drag(self.at(1, 60))
        self.assertEqual((self.view.sound, self.view.picked, self.view.drawn), (0, {(0, 0)}, TWINS["defs"][0]))
        # A press on the label of a variant has the hand in it, and with someone to tell they are told as well.
        seen = self.view.map
        label = (seen.x(0.18) + 5, seen.names[1] + 2)
        self.mouse("move", label)
        self.assertEqual(self.ui.help, told("pick variant 2", wheel=WHEEL_HELP))
        self.assertEqual((self.mouse("down", label), self.view.sound, self.view.picked, self.view.busy),
                         (True, 1, frozenset(), False))
        self.mouse("up", label)
        chosen: list[int] = []
        self.view.chosen = chosen.append
        self.assertEqual((self.mouse("down", (label[0] - 100, label[1])), self.view.sound, chosen), (True, 0, [0]))
        self.mouse("up", label)
        # So does a press in its stretch of the wave, which has the hand in it too.
        self.assertEqual((self.mouse("down", (label[0], 280)), self.view.sound, chosen), (True, 1, [0, 1]))
        self.mouse("up", (label[0], 280))
        self.view.chosen = None
        self.assertEqual((self.mouse("down", (label[0] - 100, 280)), self.view.sound, chosen), (True, 0, [0, 1]))
        self.mouse("up", (label[0] - 100, 280))
        # Of two parts that lie over one another, a click where no bar is goes to the part the hand is in.
        self.show(LAYERS)
        self.drag(self.at(8, 70))
        self.drag(self.at(1, 72, 1))
        self.drag(self.at(4, 68, 1))
        self.assertEqual([(index, defn["voices"][0]["notes"][-1]) for index, defn, _ in self.edits[1:]],
                         [(0, [8, "a#4", 6]), (1, [4, "g#4", 6])])
        # The start and the loop of a held sound are both its note, and its end is a sound of its own.
        self.show(HUM)
        seen = self.view.map
        for x, sound, line in ((seen.x(0.65), 1, "pick the end"), (seen.x(0.3), 0, "pick the note")):
            self.mouse("move", (x, seen.names[1] + 2))
            self.assertEqual((self.ui.help, self.mouse("down", (x, seen.names[1] + 2)), self.view.sound),
                             (told(line, wheel=WHEEL_HELP), True, sound))
            self.mouse("up", (x, seen.names[1] + 2))
        self.assertEqual(len(self.edits), 3)

    def test_what_is_picked_stays_picked_while_its_sound_has_the_notes_it_had(self) -> None:
        self.drag(self.at(1, 60))
        self.drag(self.at(7, 67), mods=("ctrl",))
        # The same entry shown again, as when its render has come, and one with another echo: the notes are
        # the ones there were.
        self.show()
        self.show({"def": {**DUO["def"], "echo": {"time": 0.1}}})
        self.assertEqual(self.view.picked, {C4, G4})
        # A gesture goes on when the entry it began in is shown again, and is dropped by another.
        self.mouse("down", self.at(1, 60))
        self.mouse("move", self.at(3, 60))
        self.show()
        self.assertEqual((self.view.busy, self.notes(0)[0]), (True, [2, "c4", 6]))
        self.show(DUO)
        self.assertEqual((self.view.busy, self.notes(0)[0], self.view.picked), (False, [0, "c4", 6], {C4, G4}))
        self.mouse("up", self.at(3, 60))
        self.assertEqual(self.edits, [])
        # Other notes shown, as after a step taken back, are shown with nothing picked.
        self.show({"def": duo(FIRST, [])})
        self.assertEqual(self.view.picked, frozenset())
        # What an edit leaves picked is picked when its sound is shown, and not when the owner did not save.
        self.drag(self.at(1, 60), self.at(2, 60))
        self.assertEqual((self.view.picked, self.notes(0)[0]), ({C4}, [1, "c4", 6]))
        self.saves = False
        self.drag(self.at(13, 64), button=3)
        self.assertEqual((self.view.picked, self.notes(0), len(self.edits)), ({C4}, [[1, "c4", 6], FIRST[1]], 2))

    def test_the_tabs_of_parts_that_start_together_stand_side_by_side_and_each_is_taken_by_itself(self) -> None:
        said: list[tuple] = []
        self.view.handled = lambda *what: said.append(what)
        self.view.added = lambda at: said.append(("added", at))
        self.view.asked = lambda what, which, second, pos: said.append((what, which))
        low, high = (part[1] for part in LAYERS["parts"])
        self.show({"parts": [[0.1, low], [0.1, high], [0.1, low]]})
        seen = self.view.map
        x, (_, y, _, h) = seen.x(0.1), seen.names
        # The first stands where the three start, and the others after it, each as wide as its label makes it.
        self.assertEqual([self.view.tab(which) for which in range(3)],
                         [(x, y, 11, h), (x + 11, y, 11, h), (x + 22, y, 11, h)])
        for which in range(3):
            on = (x + 11 * which + 5, y + 4)
            self.mouse("move", on)
            self.assertEqual((self.view.handle_at(on), self.ui.help),
                             ((PART, which), told(f"pick part {which + 1}, a drag moves where it starts", "its menu",
                                                  WHEEL_HELP)))
        # The second is dragged by its own tab, a twentieth of a second on, and the hand is in it.
        self.drag((x + 16, y + 4), (x + 16 + round(0.05 * seen.scale), y + 4))
        self.assertEqual((said, self.view.sound), ([(PART, 1, 0.15)], 1))
        # The right button on the third asks for the menu of that one. The sign comes after the last tab: it is
        # the button that every view has for one more of something, and says what it adds.
        self.drag((x + 27, y + 4), button=3)
        more = self.view.more
        self.mouse("move", (x + 33 + PLUS_GAP + 2, y + 4))
        self.assertEqual((more.rect, more.visible, self.ui.help),
                         ((x + 33 + PLUS_GAP, y, 11, h), True, told("add a part at the end, a copy of part 2")))
        self.drag((x + 33 + PLUS_GAP + 2, y + 4))
        self.assertEqual(said[1:], [(LABEL, 2), ("added", None)])
        # A sound that has nothing to add to has no such sign.
        self.show(DUO)
        self.assertFalse(more.visible)

    def test_a_key_of_the_keyboard_is_heard_in_the_instrument_of_the_picked_voice_and_changes_nothing(self) -> None:
        keys = self.view.keys
        self.assertEqual((keys.rect, keys.top, keys.key_h, keys.low, keys.high),
                         ((18, 30, 40, 204), 72.0, 11, -96, 131))
        # c4 is the key from y 162. A press on it is heard, and the key is lit until the button is let go.
        self.assertTrue(self.mouse("down", (30, 166)))
        self.assertEqual((self.heard, keys.down, self.edits, self.view.picked, self.view.busy),
                         ([("tone", 60)], 60, [], frozenset(), False))
        self.mouse("up", (30, 166))
        self.view.voice = 1
        self.drag((30, self.view.map.y(67) + 4))
        self.assertEqual((self.heard[1:], keys.down), ([("buzz", 67)], None))
        # An owner has any pitch heard the same way, as for the piano keys of the computer: those it is handed
        # as keys the view has no use for, also while the keyboard has the focus.
        self.view.hear(61)
        typed: list[str] = []
        self.view.typed = lambda event: typed.append(event.key) or event.key == "q"
        self.assertEqual((self.heard[2:], self.ui.focus, self.ui.handle(Event("key", key="q")),
                          self.ui.handle(Event("key", key="f1")), typed),
                         ([("buzz", 61)], keys, True, False, ["q", "f1"]))
        # The view's own keys are the view's there too: ctrl and a picks the notes of the picked voice.
        self.assertEqual((self.ui.handle(Event("key", key="a", mods=frozenset({"ctrl"}))), self.view.picked, typed),
                         (True, {G4}, ["q", "f1"]))
        # Shown shapes alone, there is no voice to hear a key in.
        self.view.show(ONE, [], RATE, theme.SOUND)
        self.drag((30, 166))
        self.assertEqual(len(self.heard), 3)

    def test_the_keys_stay_where_they_are_left_when_the_notes_change_and_open_on_another_sound(self) -> None:
        self.assertEqual(self.view.top, 72.0)
        self.ui.handle(Event("wheel", self.at(8, 62), wheel=2))
        self.assertEqual(self.view.top, 76.0)
        # A note placed far over the others leaves the keys where they are, and so does its render when it comes.
        self.drag(self.at(8, 74))
        self.assertEqual((self.notes(0)[2:], self.view.top), ([[8, "d5", 6]], 76.0))
        self.show()
        self.assertEqual(self.view.top, 76.0)
        # Another sound is opened on its notes: its one c4 has 8 keys over it.
        self.show(SOLO)
        self.assertEqual((self.view.top, self.view.map.y(60)), (68.0, 118))

    def test_the_lane_has_a_bar_for_each_note_of_the_picked_voice_and_a_drag_in_it_is_one_edit(self) -> None:
        lane = self.view.lane
        # The lane stands where its owner puts it: here under the box of the wave, 24 pixels high, so that a
        # velocity of a half is at y 412.
        lane.rect = (18, 400, 445, 24)
        self.show()
        self.assertEqual((lane.span, lane.items, lane.colour, lane.help, lane.enabled),
                         ((58, 400), ((59, 119, 1.0, False), (299, 119, 1.0, False)), theme.SOUND, VELOCITY_HELP, True))
        # A press sets the note under it, and the view draws it so at once: nothing is handed on before the
        # release. Dragged on over e4, each has the velocity of where the mouse passed it.
        self.assertTrue(self.mouse("down", (100, 412)))
        self.assertEqual((self.notes(0)[0], self.view.busy, self.edits), ([0, "c4", 6, 0.5], True, []))
        self.mouse("move", (340, 417))
        self.mouse("up", (340, 417))
        weighed = duo([[0, "c4", 6, 0.4], [12, "e4", 6, 0.25]], SECOND)
        self.assertEqual((self.edits, self.view.busy, self.heard), ([(0, weighed, VELOCITY_DONE)], False, []))
        self.assertEqual(lane.items, ((59, 119, 0.4, False), (299, 119, 0.25, False)))
        # The right button gives a note the usual velocity, which is written as none, and so is one of 1.
        self.drag((100, 412), button=3)
        self.drag((340, 402))
        self.assertEqual([defn["voices"][0]["notes"] for _, defn, _ in self.edits[1:]],
                         [[FIRST[0], [12, "e4", 6, 0.25]], FIRST])
        self.assertEqual((self.view.drawn, len(self.edits)), (DUO["def"], 3))
        # The lane has the notes of the picked voice, in its colour, and the picked ones are told to it.
        self.view.voice = 1
        self.drag(self.at(7, 67))
        self.assertEqual((lane.items, lane.colour), (((179, 119, 1.0, True),), theme.SALMON))
        # Shown shapes alone, the lane takes nothing from the mouse.
        self.view.show(ONE, [], RATE, theme.SOUND)
        self.assertEqual((lane.enabled, lane.items), (False, ()))

    def test_the_flag_of_an_expected_note_is_dragged_along_the_ruler_and_the_right_button_asks_for_its_menu(
            self) -> None:
        said: list[tuple] = []
        self.view.handled = lambda *what: said.append(what)
        self.view.asked = lambda what, which, second, pos: said.append((what, which, round(second, 6)))
        self.show({**DUO, "check": [{"at": 0.1, "note": "e4"}]}, other=False)
        seen = self.view.map
        # A tenth of a second is 120 pixels on this clock. The flag stands before its line.
        x, y = seen.x(0.1), seen.ruler[1] + 2
        self.assertEqual((x, [self.view.handle_at((at, y)) for at in (x - FLAG_W - 1, x - FLAG_W, x + GRIP,
                                                                    x + GRIP + 1)]),
                         (178, [None, (MARK, 0), (MARK, 0), None]))
        self.mouse("move", (x, y))
        self.assertEqual(self.ui.help, told("drag where e4 is expected", "change it or remove it", WHEEL_HELP))
        # Dragged a twentieth of a second on, it is told of at its release, to the hundredth of a second.
        self.mouse("down", (x, y))
        self.mouse("move", (x + 62, y + 40))
        self.assertEqual((self.view.busy, said), (True, []))
        self.mouse("up", (x + 62, y + 40))
        self.assertEqual((said, self.view.busy, self.edits), ([(MARK, 0, 0.15)], False, []))
        # It goes to no second before 0. The right button on it asks for its menu, with the second it is at,
        # and anywhere else in the ruler for the menu of the ruler.
        self.drag((x, y), (0, y))
        self.drag((x - 2, y), button=3)
        self.drag((x + 100, y), button=3)
        self.assertEqual(said[1:], [(MARK, 0, 0.0), (MARK, 0, 0.1), (RULER, 0, round(seen.second_at(x + 100), 6))])
        # With nobody to tell there is no flag to take.
        self.view.handled = None
        self.assertIsNone(self.view.handle_at((x, y)))


@unittest.skipUnless(pygame, NEEDS)
class Drawn(unittest.TestCase):
    def setUp(self) -> None:
        self.surface = pygame.Surface(SIZE)
        self.surface.fill(OUTSIDE)
        self.canvas = Canvas(self.surface, Font(sprites.load("font")))
        self.ui = Ui()
        self.view = view_on(self.ui)
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
        """Draw the view and its parts, and give its sums."""
        self.ui.draw(self.canvas)
        return self.view.map

    def open(self, *shown: object) -> None:
        """Show the view another sound, which it is opened on as an owner opens it: with the keys on its notes."""
        self.view.fit()
        self.view.show(*shown)

    def sound(self, project: Project,
              name: str) -> tuple[SoundView, Map, list[tuple[str, float, Shape]], list[np.ndarray]]:
        """A sound of a project drawn over its real render, on a grid 552 wide from x 48 with the box of the wave
        under it: the view, its sums, the shapes and the buffers. The surface that is looked at from here on is
        the one it is drawn on."""
        shapes = soundview.shapes(project.sounds[name], instruments=project.instruments)
        buffers = render_entry(project.sound(name))
        view = SoundView((8, 0, 597, 300))
        view.foot = (48, 300, 552, 100)
        view.show(shapes, buffers, 44100, theme.SOUND, entry_kind(project.sounds[name]))
        self.surface = pygame.Surface((600, 400))
        view.draw(Canvas(self.surface, Font(sprites.load("font"))))
        return view, view.map, shapes, buffers

    def arcade(self, name: str) -> tuple[SoundView, Map, list[tuple[str, float, Shape]], list[np.ndarray]]:
        """A sound of the arcade example, drawn as `sound` draws one."""
        return self.sound(Project.load(EXAMPLES / "arcade.json"), name)

    def column_at(self, seen: Map, columns: np.ndarray, index: int) -> set[tuple[int, int]]:
        """The pixels that one column of a wave takes, made to meet its neighbours, inside the outline of the box."""
        ends = joined([seen.column(*column) for column in columns.tolist()])
        return {(seen.wave[0] + index, y) for y in range(ends[index][0], ends[index][1] + 1)}

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
        # A view opened on one note alone has it in the row from y 118: a second repeat is later and fainter
        # still.
        self.open(one(bar(0.25, 0.15, 67), echo=((0.25, 0.25), (0.5, 0.04))), [wave()], RATE, theme.SOUND)
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
        # Column for column it is what the samples say, with the line at zero behind it. The name of the box
        # is behind the wave too: this one is loud where the name stands.
        self.assertEqual(self.where(theme.SOUND, seen.wave) | self.where(theme.FAULT, seen.wave),
                         self.wave_of(seen, left, columns))
        self.assertEqual(self.colours(seen.wave), {theme.BACKGROUND, theme.LINE, theme.SOUND, theme.FAULT})
        # With no rows the box holds the line at zero and its name, at the right of its upper edge, and the
        # bars are there as before.
        self.view.show(ONE, [], RATE, theme.SOUND)
        self.draw()
        self.assertEqual(self.view.columns(), [])
        named = self.where(theme.DIM, seen.wave)
        self.assertTrue(named and all(428 <= x < 452 and 241 <= y < 252 for x, y in named), sorted(named)[:3])
        self.assertEqual(self.where(theme.LINE, (59, 240, 398, 79)), {(x, 279) for x in range(59, 457)})
        self.assertEqual((self.colours((59, 252, 398, 67)), self.at(170, 152)),
                         ({theme.BACKGROUND, theme.LINE}, theme.SOUND))
        # A box that is wider than the grid has the wave in the stretch of it that is under the grid: the rest
        # is the ground of a panel, inside the outline of the box.
        self.view.foot = (18, 239, 445, 81)
        self.view.show(ONE, [wave()], RATE, theme.SOUND)
        seen = self.draw()
        self.assertEqual((seen.wave, self.colours((19, 240, 39, 79)), self.colours((18, 239, 445, 1))),
                         (FOOT, {theme.PANEL}, {theme.LINE}))
        self.assertEqual(self.where(theme.SOUND, FOOT) | self.where(theme.FAULT, FOOT),
                         self.wave_of(seen, left, columns) | self.column_at(seen, columns, 0)
                         | self.column_at(seen, columns, 399))
        # With no shapes the wave is all there is under the ruler, in a box of that size: no keys stand beside
        # it, its columns are as much longer, and nothing is drawn in the box at the foot.
        self.surface.fill(OUTSIDE)
        self.view.show([], [wave()], RATE, theme.INSTRUMENT)
        seen = self.draw()
        self.assertEqual((seen.notes, seen.wave, seen.box, self.view.seconds),
                         (False, (58, 30, 400, 209), (58, 30, 400, 209), 1.0))
        self.assertEqual(([part.visible for part in self.view.parts], self.colours((18, 30, 40, 209)),
                          self.colours((18, 239, 445, 81))), ([False] * 5, {theme.PANEL}, {OUTSIDE}))
        self.assertEqual((seen.column(-0.5, 0.5), self.where(theme.INSTRUMENT, (108, 30, 1, 209))),
                         ((83, 184), {(108, y) for y in range(83, 185)}))

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
        # Each number starts 2 pixels after its line, three letters wide, and nothing stands before the ruler.
        inked = {x for x, _ in self.where(theme.DIM, seen.ruler)}
        self.assertEqual([edge(x for x in inked if 98 <= x < 138) for edge in (min, max)], [100, 116])
        self.assertEqual((self.colours(seen.ruler), self.colours((18, 20, 40, 10))),
                         ({theme.PANEL, theme.DIM}, {theme.PANEL}))
        # The keyboard is the one every view of notes has, beside the rows it names. c5 is a white key with its
        # name at its right end, and d-5 a black one for most of its length.
        c, d_flat = seen.y(72), seen.y(73)
        self.assertEqual([self.at(20, c + 3), self.at(20, d_flat + 3), self.at(50, d_flat + 3)],
                         [theme.TEXT, theme.BACKGROUND, theme.TEXT])
        self.assertIn(theme.BACKGROUND, self.colours((42, c + 2, 11, 7)))
        # The row of a black key is darker than that of a white one, and the keys end a pixel short of the grid.
        self.assertEqual((self.at(70, c + 3), self.at(70, d_flat + 3)), (theme.RAISED, theme.BACKGROUND))
        self.assertEqual(self.colours((57, 30, 1, 204)), {theme.PANEL})

    def test_the_keys_stay_as_high_as_they_are_for_notes_far_apart_and_stand_beside_their_rows_as_they_scroll(
            self) -> None:
        # Three octaves do not make a key lower: the view is opened on the middle of them, and is scrolled.
        self.open(one(bar(0.0, 0.5, 60), bar(0.5, 0.5, 96)), [], RATE, theme.SOUND)
        tops = []
        for steps in (0, 2, -3):
            self.ui.handle(Event("wheel", (300, 100), wheel=steps))
            seen = self.draw()
            tops.append(seen.top)
            # c6 is a white key with its name on it, under a black one, each beside its row of the grid.
            y = seen.y(84)
            self.assertEqual((self.view.keys.at((30, y + 3)), seen.y(83) - y, self.view.keys.key_h), (84, 11, 11))
            self.assertEqual((self.at(20, y + 3), self.at(70, y + 3)), (theme.TEXT, theme.RAISED), steps)
            self.assertIn(theme.BACKGROUND, self.colours((42, y + 2, 11, 7)))
            if y > 40:
                self.assertEqual((self.at(20, y - 8), self.at(70, y - 8)), (theme.BACKGROUND, theme.BACKGROUND))
        self.assertEqual(tops, [86.0, 90.0, 84.0])
        # The bars of the two notes are out of view, one under the grid and one over it, until the view is
        # taken to them.
        self.assertEqual((seen.y(60) >= 30 + 204, seen.y(96) + 11 <= 30, self.where(theme.SOUND, seen.grid)),
                         (True, True, set()))
        self.ui.handle(Event("wheel", (300, 100), wheel=6))
        seen = self.draw()
        self.assertEqual((seen.top, self.at(300, seen.y(96) + 4)), (96.0, theme.SOUND))

    def test_the_playhead_is_a_line_at_its_second_and_nothing_is_drawn_outside_the_rectangle(self) -> None:
        self.view.playhead = 0.5
        self.draw()
        # It runs down the view and the box of the wave, under the scroll bar that lies between the two.
        self.assertEqual(self.where(theme.ACTIVE, (0, 0, *SIZE)), {(258, y) for y in range(20, 320)} - UNDER_BAR)
        # None is no line, and neither is a second that is not in view.
        for playhead in (None, -0.1, 1.5):
            self.view.playhead = playhead
            self.draw()
            self.assertEqual(self.where(theme.ACTIVE, (0, 0, *SIZE)), set())
        # Nothing is drawn outside the rectangle of the view and the box of the wave.
        around = ((0, 0, 470, 20), (0, 320, 470, 10), (0, 20, 18, 300), (463, 20, 7, 219), (18, 239, 40, 81),
                  (458, 239, 12, 81))
        self.view.playhead = 0.3
        for steps, mod in ((0, ""), (3, "ctrl"), (-2, "shift"), (5, ""), (20, "ctrl")):
            self.ui.handle(Event("wheel", (300, 100), wheel=steps, mods=frozenset({mod})))
            self.draw()
            self.assertEqual(set().union(*(self.colours(rect) for rect in around)), {OUTSIDE})
            self.assertNotIn(OUTSIDE, self.colours(RECT) | self.colours(FOOT))
        # A view too small for its parts keeps to its rectangle as well, and so does a box too small for a wave.
        self.surface.fill(OUTSIDE)
        self.view.rect, self.view.foot = (100, 100, 70, 60), (100, 162, 70, 5)
        self.draw()
        self.assertEqual(len(self.where(OUTSIDE, (0, 0, *SIZE))), 470 * 330 - 70 * 65)

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
                         {(358, y) for y in range(20, 320) if not 235 <= y < 239})
        self.assertTrue(seen.x(0.5) <= 358 < seen.x(1.0))

    def test_the_zones_of_a_held_sound_are_tinted_each_in_its_colour_under_its_label(self) -> None:
        self.open(HELD, rows(0.25, 0.5, 0.25), RATE, theme.SOUND, "held")
        seen = self.draw()
        # The three meet at x 158 and x 358, where a line runs down the labels and the grid. d4 is a white key,
        # in the row from y 106, and the key under it a black one.
        white, black = seen.y(62) + 3, seen.y(61) + 3
        self.assertEqual((seen.grid, seen.x(0.25), seen.x(0.75), white, black),
                         ((58, 40, 400, 194), 158, 358, 109, 120))
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
        self.assertEqual((self.colours(RECT) | self.colours(FOOT)) & every, set())

    def test_a_picked_note_is_outlined_a_gesture_is_drawn_as_its_release_would_leave_it_and_a_stale_wave_is_dim(
            self) -> None:
        shapes = soundview.shapes(DUO, instruments=INSTRUMENTS)
        self.open(shapes, [wave()[:266]], RATE, theme.SOUND, "def", DUO)
        seen = self.draw()
        # The bars are those of the entry, each in the colour of its voice, and nothing is outlined.
        self.assertEqual((self.at(100, 166), self.at(340, 122), self.at(220, 89), self.where(theme.TEXT, seen.grid)),
                         (theme.SOUND, theme.SOUND, theme.SALMON, set()))
        # A click on c4 picks it: its bar (59, 162, 119, 10) has a bright outline in place of its edges.
        for kind in ("down", "up"):
            self.ui.handle(Event(kind, (100, 166), button=1))
        self.draw()
        outline = {(x, y) for x in range(59, 178) for y in range(162, 172) if x in (59, 177) or y in (162, 171)}
        self.assertEqual((self.where(theme.TEXT, seen.grid), self.at(100, 166)), (outline, theme.SOUND))
        # Dragged 4 ticks on and 2 keys up, it is drawn where the release would leave it, and not where it is.
        self.ui.handle(Event("down", (100, 166), button=1))
        self.ui.handle(Event("move", (180, 144)))
        self.draw()
        moved = {(x + 80, y - 22) for x, y in outline}
        self.assertEqual((self.where(theme.TEXT, seen.grid), self.at(180, 144), self.at(100, 166)),
                         (moved, theme.SOUND, theme.RAISED))
        # With nobody to save it, the release leaves the notes as they were shown.
        self.ui.handle(Event("up", (180, 144), button=1))
        self.draw()
        self.assertEqual(self.where(theme.TEXT, seen.grid), outline)
        # A box is outlined while it is drawn, from the pixel of the press to the pixel of the mouse.
        self.ui.handle(Event("down", (100, 100), button=1, mods=frozenset({"shift"})))
        self.ui.handle(Event("move", (160, 150), mods=frozenset({"shift"})))
        self.draw()
        frame = {(x, y) for x in range(100, 161) for y in range(100, 151) if x in (100, 160) or y in (100, 150)}
        self.assertEqual(self.where(theme.TEXT, seen.grid), frame)
        self.ui.handle(Event("up", (160, 150), button=1, mods=frozenset({"shift"})))
        # A stale wave has less than half of its colour, and a fresh one all of it.
        dim = mix(theme.BACKGROUND, theme.SOUND, STALE)
        for stale, inks in ((True, (True, False)), (False, (False, True))):
            self.view.show(shapes, [wave()[:266]], RATE, theme.SOUND, "def", DUO, stale)
            self.draw()
            found = self.colours(seen.wave)
            self.assertEqual((self.view.stale, dim in found, theme.SOUND in found), (stale, *inks))
        self.assertLess(strength(dim, theme.BACKGROUND), strength(theme.SOUND, theme.BACKGROUND) / 2)

    def test_the_line_of_the_longest_time_stands_at_its_second_down_the_whole_of_the_view(self) -> None:
        shapes = soundview.shapes(DUO, instruments=INSTRUMENTS)
        long = shapes[0][2].seconds

        def lines(limit: float) -> set[tuple[int, int]]:
            """The pixels in the colour of a fault, with the sound held to a longest time over a silent wave."""
            self.view.show(shapes, rows(0.3), 44100, theme.SOUND, "def", {**DUO, "max_seconds": limit})
            self.draw()
            return self.where(theme.FAULT, (0, 0, *SIZE))

        # The sound is a third of a second long, and its clock has a second of room after it. A limit inside the
        # sound leaves the clock as it is. Its line runs down the view and the box of the wave, under the scroll
        # bar that lies between the two.
        down = [y for y in range(20, 320) if not 235 <= y < 239]
        line = {(358, y) for y in down}
        self.assertEqual((lines(0.25), self.view.limit, self.view.seconds, self.view.map.x(0.25)),
                         (line, 0.25, long + 1, 358))
        # A limit that is past the sound is in the room after it: out of view as the sound was opened, and there
        # once the view has gone along the clock to it.
        self.assertEqual((lines(0.5), self.view.seconds), (set(), long + 1))
        self.view.start = 0.3
        found, seen = lines(0.5), self.view.map
        self.assertEqual((seen.start, found), (0.3, {(seen.x(0.5), y) for y in down}))
        self.view.start = 0.0
        # The clock goes on to a limit that is past its room, and a little further. One that is more than twice
        # the clock away is off it.
        self.assertEqual((lines(2), self.view.seconds, self.view.limit), (set(), 2 * (1 + LIMIT_AIR), 2))
        self.assertEqual((lines(3), self.view.seconds, self.view.limit), (set(), long + 1, 3))
        # Where it can be dragged, it has a grip in the ruler.
        self.view.handled = lambda *told: None
        self.assertEqual(lines(0.25), line | {(x, y) for x in range(358 - GRIP, 358 + GRIP + 1) for y in range(20, 30)})

    def test_a_part_dragged_by_its_label_is_drawn_where_its_release_would_leave_it_over_the_dim_wave(self) -> None:
        told: list[tuple] = []
        self.view.handled = lambda *what: told.append(what)
        # By itself the part that is dragged is a tenth of a second of a steady half.
        self.view.alone = lambda which: np.full(RATE // 10, 0.5, dtype=np.float32)
        self.view.show(soundview.shapes(LAYERS, instruments=INSTRUMENTS), [wave()[:160]], RATE, theme.SOUND, "parts",
                       LAYERS, instruments=INSTRUMENTS)
        seen = self.draw()
        x, y, w, h = self.view.tab(1)
        by = round(0.05 * seen.scale)
        to = seen.x(0.15)
        self.assertEqual((x, theme.SOUND in self.colours(seen.wave)), (seen.x(0.1), True))
        self.ui.handle(Event("down", (x + 4, y + 4), button=1))
        self.ui.handle(Event("move", (x + 4 + by, y + 40)))
        self.draw()
        # Its tab is lit at the tick the mouse is at, and its note c5 starts there: no longer where it was.
        note = seen.y(72) + 4
        self.assertEqual((self.view.busy, self.view.sound, self.at(to + 1, y + 1), self.at(to + 3, note),
                          self.at(x + 3, note)), (True, 1, theme.LIT, theme.SOUND, theme.RAISED))
        # The wave of the whole is dim, and the part's own is drawn whole from where the part would start.
        dim = mix(theme.BACKGROUND, theme.SOUND, STALE)
        whole = {at for at, _ in self.where(theme.SOUND, seen.wave)}
        self.assertEqual((dim in self.colours(seen.wave), min(whole), max(whole) > to + 20), (True, to, True))
        self.assertEqual(told, [])
        self.ui.handle(Event("up", (x + 4 + by, y + 40), button=1))
        self.assertEqual((told, self.view.busy), ([(PART, 1, 0.15)], False))
        # Escape drops a handle that is pulled, and its release then tells of nothing.
        self.ui.handle(Event("down", (x + 4, y + 4), button=1))
        self.ui.handle(Event("move", (x + 4 + by, y + 4)))
        self.assertTrue(self.ui.handle(Event("key", key="escape")))
        self.ui.handle(Event("up", (x + 4 + by, y + 4), button=1))
        self.assertEqual((len(told), self.view.busy), (1, False))

    def test_tabs_laid_side_by_side_have_each_its_label_and_the_line_of_a_part_stays_at_its_second(self) -> None:
        self.view.handled = lambda *what: None
        long, short = {"def": solo([0, "c4", 40])}, {"def": solo([0, "c5", 6])}
        # Two parts start a tenth of a second in, and a third a tick after them: on a clock of 0.8 s that is 8
        # pixels on, under the tab of the first.
        entry = {"parts": [[0.1, long], [0.1, short], [on_tick(0.1 + 1 / 60), short]]}
        self.view.show(soundview.shapes(entry, instruments=INSTRUMENTS), [], 44100, theme.SOUND, "parts", entry)
        seen = self.draw()
        (_, y, _, h), lefts = seen.names, [self.view.tab(which)[0] for which in range(3)]
        self.assertEqual(([seen.x(at) for at, _ in entry["parts"]], lefts), ([108, 108, 116], [108, 119, 130]))
        # Each label is in its own tab. The first tab stands where its part starts, and has that line for its
        # edge: the others have their own outline.
        inked = {at for at, _ in self.where(theme.TEXT, seen.names)}
        labels = [set(range(left + 3, left + 8)) for left in lefts]
        self.assertEqual(([bool(inked & label) for label in labels], inked - set().union(*labels)),
                         ([True] * 3, {108}))
        # The tab of the part the hand is in is lit, as what is picked is everywhere.
        self.assertEqual([(self.at(left, y + 1), self.at(left + 1, y + 1)) for left in lefts],
                         [(theme.TEXT, theme.LIT), (theme.LINE, theme.RAISED), (theme.LINE, theme.RAISED)])
        # The line of the third runs down the grid at its second, and under the tab of the first on its way.
        self.assertEqual({at for at, _ in self.where(theme.TEXT, seen.grid)}, {108, 116})
        self.assertEqual(self.at(116, y + h), theme.TEXT)

    def test_the_flag_of_an_expected_note_is_a_line_down_the_ruler_with_a_small_flag_before_it(self) -> None:
        entry = {**DUO, "check": [{"at": 0.1, "note": "e4"}, {"at": 0.25, "note": "g4"}]}
        self.view.handled = lambda *what: None
        self.open(soundview.shapes(entry, instruments=INSTRUMENTS), [], 44100, theme.SOUND, "def", entry)
        seen = self.draw()
        self.assertEqual(([seen.x(mark["at"]) for mark in entry["check"]], seen.ruler), ([178, 358], (58, 20, 400, 10)))
        flags = {(x, y) for x in (178, 358) for y in range(20, 30)}
        flags |= {(x - step, y) for x in (178, 358) for step in range(1, FLAG_W + 1) for y in range(20, 24)}
        self.assertEqual(self.where(FLAG, (0, 0, *SIZE)), flags)
        # Dragged, a flag is drawn where its release would leave it.
        self.ui.handle(Event("down", (178, 25), button=1))
        self.ui.handle(Event("move", (238, 25)))
        self.draw()
        self.assertEqual({x for x, _ in self.where(FLAG, seen.ruler) if x < 300}, set(range(238 - FLAG_W, 239)))
        self.ui.handle(Event("up", (238, 25), button=1))
        # A sound with none has no flag.
        self.view.show(soundview.shapes(DUO, instruments=INSTRUMENTS), [], 44100, theme.SOUND, "def", DUO)
        self.draw()
        self.assertEqual(self.where(FLAG, (0, 0, *SIZE)), set())

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


#: A sound of one note that is held for 48 ticks: with the two ticks it takes to end it is 50 long.
LONG = {"def": solo([0, "c4", 48])}


@unittest.skipUnless(pygame, NEEDS)
class Room(unittest.TestCase):
    """What the clock of a sound that can be changed has after the end of the sound, opened as a sound is."""

    def setUp(self) -> None:
        # The air a sound is opened with is the one there is, here.
        _NO_AIR.stop()
        self.addCleanup(_NO_AIR.start)
        self.ui = Ui()
        self.view = view_on(self.ui, PLAIN)
        self.edits: list[tuple[int, dict, str]] = []
        self.view.edited = self.save
        self.show(LONG)

    def show(self, entry: dict | None = None) -> None:
        """Show the view an entry, or again the one it has as a change left it, with no wave under it."""
        if entry is not None:
            self.view.fit()
            self.entry = copy.deepcopy(entry)
        shapes = soundview.shapes(self.entry, instruments=INSTRUMENTS)
        self.long = max(place + shape.seconds for _, place, shape in shapes)
        self.view.show(shapes, [], 44100, theme.SOUND, entry_kind(self.entry), self.entry)

    def save(self, index: int, defn: dict, label: str) -> bool:
        """What an owner does with an edit: it is kept here, put in the entry, and the view is shown the entry."""
        self.edits.append((index, defn, label))
        self.entry = soundedit.with_sound(self.entry, index, defn, INSTRUMENTS)
        self.show()
        return True

    def wheel(self, steps: int, *mods: str) -> None:
        """Turn the wheel with the mouse on the grid, some keys held."""
        self.ui.handle(Event("wheel", (258, 100), wheel=steps, mods=frozenset(mods)))

    def test_the_sums_of_a_map_that_shows_less_than_its_clock_go_by_what_it_shows(self) -> None:
        # Two seconds of clock, of which half a second is as wide as the grid: 800 pixels to the second.
        seen = Map(RECT, seconds=2.0, shown=0.5)
        self.assertEqual((seen.scale, seen.fitted, seen.x(0.5), seen.scrolls()[1]), (800.0, 0.5, 458, (1600, 400, 0)))
        # The view goes along the clock until the end of the clock is at the right edge of the grid.
        far = replace(seen, start=9.0).within()
        self.assertEqual((far.start, far.x(2.0), far.scrolls()[1]), (1.5, 458, (1600, 400, 1200)))
        # It is zoomed out as far as the first zoom that has the whole clock in view, and no further: 1.25 to
        # the power of 7 is the first that is more than 4. The clock then starts at the left edge.
        out = replace(seen, start=1.0).zoomed(-100, 258).within()
        self.assertEqual((seen.least, out.zoom, out.start), (-7, -7, 0.0))
        self.assertTrue(out.x(2.0) <= 458 < replace(out, zoom=-6).x(2.0))
        self.assertEqual(seen.zoomed(100, 258).zoom, ZOOM_MOST)
        # A zoom under the least, which a clock that has become shorter leaves, is brought up to it.
        self.assertEqual(replace(seen, zoom=-7, seconds=1.0).within().zoom, -4)
        # A map that shows its whole clock is as it was: no zoom under 0, and a second as wide as the clock makes it.
        whole = Map(RECT, seconds=2.0)
        self.assertEqual((whole.least, whole.fitted, whole.scale, whole.zoomed(-3, 258).zoom), (0, 2.0, 200.0, 0))
        self.assertEqual(Map(RECT, seconds=0.2, shown=0.5).least, 0)

    def test_a_sound_is_opened_with_a_quarter_of_its_length_after_it_in_view_and_a_second_of_room(self) -> None:
        view, long = self.view, self.long
        seen = view.map
        self.assertEqual((ROOM_LEAST, OPEN_LEAST, long, view.shown, view.seconds, view.zoom, view.start),
                         (1.0, 0.5, 50 / 60, long * 1.25, long + 1.0, 0, 0.0))
        self.assertEqual((seen.x(0.0), seen.x(long * 1.25)), (58, 458))
        # There is more of the clock than is in view, so the bar along the time has a thumb from the first.
        left, _, wide, _ = view.across.thumb()
        self.assertEqual((left, wide < 400), (58, True))
        # With shift the wheel goes along the clock into the room, as far as the end of the clock.
        self.wheel(-1, "shift")
        self.assertAlmostEqual(view.start, 36 / seen.scale)
        self.wheel(-1000, "shift")
        self.assertAlmostEqual(view.start, long + 1.0 - long * 1.25)
        self.assertLessEqual(abs(view.map.x(view.seconds) - 458), 1)
        # A sound of two seconds has as long again after it.
        self.show({"def": solo([0, "c4", 120])})
        self.assertEqual((view.shown, view.seconds, view.start), (self.long * 1.25, 2 * self.long, 0.0))
        self.assertGreaterEqual(self.long, 2.0)
        # A sound of a few ticks is opened with half a second in view, and has a second of room as any has.
        self.show({"def": solo([0, "c4", 3])})
        self.assertEqual((self.long, view.shown, view.seconds, view.map.x(0.5)), (5 / 60, 0.5, 5 / 60 + 1.0, 458))
        self.assertIsNotNone(view.across.thumb())

    def test_what_is_shown_and_not_changed_has_no_room_and_its_whole_clock_in_view(self) -> None:
        self.view.fit()
        self.view.show(ONE, [wave()], RATE, theme.SOUND)
        self.assertEqual((self.view.shown, self.view.seconds, self.view.map.x(1.0), self.view.across.thumb()),
                         (0.0, 1.0, 458, None))
        self.wheel(-3, "ctrl")
        self.assertEqual(self.view.zoom, 0)

    def test_a_note_put_in_the_room_makes_the_sound_longer_and_a_second_stays_as_wide_as_it_was(self) -> None:
        view, long = self.view, self.long
        scale = view.map.scale
        # The view goes along the clock to where a second and a half is in it, which is after the end of the sound.
        view.start = 0.7
        seen = view.map
        self.assertTrue(seen.start == 0.7 and long < 1.5 and 58 < seen.x(1.5) < 458)
        pos = (seen.x(1.5) + 1, seen.y(64) + 4)
        self.ui.handle(Event("down", pos, button=1))
        self.ui.handle(Event("up", pos, button=1))
        (index, defn, label), = self.edits
        self.assertEqual((index, defn["voices"][0]["notes"][0], defn["voices"][0]["notes"][1][:2], label),
                         (0, [0, "c4", 48], [90, "e4"], "note placed"))
        # The sound ends later now, and the room is after its new end: as long again as the sound, which is
        # more than a second long now. The view is where it was, and a second is as wide as it was.
        after = self.long
        self.assertGreater(after, 1.5)
        self.assertEqual((view.seconds, view.shown, view.start, view.zoom, view.map.scale),
                         (2 * after, long * 1.25, 0.7, 0, scale))
        # A note is made longer into the room in the same way: its end is dragged on by half a second.
        ticks = defn["voices"][0]["notes"][1][2]
        seen = view.map
        x, y, w, _ = seen.bar(1.5, ticks / 60, 64)
        self.ui.handle(Event("down", (x + w - 1, y + 4), button=1))
        self.ui.handle(Event("move", (x + w - 1 + round(0.5 * scale), y + 4)))
        self.ui.handle(Event("up", (x + w - 1 + round(0.5 * scale), y + 4), button=1))
        self.assertEqual((len(self.edits), self.edits[-1][1]["voices"][0]["notes"][1][:3], view.map.scale),
                         (2, [90, "e4", ticks + 30], scale))
        self.assertEqual((self.long > after, view.seconds), (True, 2 * self.long))

    def test_ctrl_and_the_wheel_zoom_out_until_the_whole_clock_with_its_room_is_in_view(self) -> None:
        view = self.view
        self.wheel(-100, "ctrl")
        seen = view.map
        self.assertEqual((view.zoom, view.start), (seen.least, 0.0))
        self.assertLess(view.zoom, 0)
        self.assertEqual((seen.x(0.0), seen.x(view.seconds) <= 458, view.across.thumb()), (58, True, None))
        # In again, a second is as wide as the sound was opened with.
        self.wheel(-view.zoom, "ctrl")
        self.assertEqual(view.zoom, 0)
        self.assertAlmostEqual(view.map.scale, 400 / (self.long * 1.25))

    def test_another_sound_is_opened_on_its_own_length_and_the_same_one_stays_as_it_was_left(self) -> None:
        view, short = self.view, self.long
        self.wheel(2, "ctrl")
        self.show()
        self.assertEqual((view.shown, view.zoom), (short * 1.25, 2))
        self.show({"def": solo([0, "c4", 60])})
        self.assertEqual((view.shown, view.zoom, view.start, view.map.x(self.long * 1.25)),
                         (self.long * 1.25, 0, 0.0, 458))


if __name__ == "__main__":
    unittest.main()
