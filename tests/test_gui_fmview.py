"""Tests for rowsynth.gui.fmview: the wirings against the engine, their pictures, and the envelope with its points."""
from __future__ import annotations

import itertools
import json
import os
import unittest
from pathlib import Path

import numpy as np

from rowsynth import fm
from rowsynth.errors import RowSynthError
from rowsynth.gui import layout, sprites, theme
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.fmview import (ATTACK, DECAY, HOLD, POINTS, RANGES, RELEASE, SUSTAIN, WIRING_H, Envelope, Map, Wires,
                                 Wiring, cells, falling, length, moments, moved, nearest, picture, places, said,
                                 settings, told, wires)
from rowsynth.gui.font import Font
from rowsynth.gui.ui import Event, Ui

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
OUTSIDE = theme.GOOD

#: A low rate and a short note keep the proof quick: 160 samples a render.
RATE = 8000
NOTE = (220.0, 0.015, 0.02)
#: A voice in which every operator has a level and a pitch of its own, so that none can stand in for another.
OPS = [{"mul": mul, "level": level} for mul, level in ((1, 0.9), (2, 1.1), (3, 0.7), (5, 1.3))]

#: The picture of one algorithm: its boxes are 16 apart, the base row at y 56 and the base line at y 73.
CELL = (10, 20, 74, 62)
#: A row whose pictures are 74 wide and 80 apart, the first at x 3.
ROW = (3, 5, 634, WIRING_H)
#: An envelope whose line runs in (14, 24, 101, 51). The note of `OP` is a second long, so a pixel is a hundredth
#: of a second from side to side, and a fiftieth of the level up and down.
RECT = (10, 20, 109, 59)
OP = {"mul": 2, "det": 0.5, "level": 1.5, "env": {"a": 0.2, "d": 0.3, "s": 0.5, "r": 0.2}}
#: Where the points of `OP` are: the top of the attack, the end of the decay, let go, and the end of the release.
SPOTS = [(34, 24), (64, 49), (94, 49), (114, 74)]


def pixels(*rects: layout.Rect) -> set[tuple[int, int]]:
    """Every pixel of some rectangles."""
    return {(x + column, y + row) for x, y, w, h in rects for column in range(w) for row in range(h)}


def reaches(wired: Wires, op: int, quiet: frozenset[int]) -> bool:
    """Whether an operator is heard, or changes one that is, by way of operators that are not quiet."""
    if op in quiet:
        return False
    return op in wired.heard or any(reaches(wired, to, quiet) for source, to in wired.changes if source == op)


def tone(algorithm: int, quiet: frozenset[int], fb: float = 0.0) -> np.ndarray:
    """One short note of the voice in an algorithm, with the level of the quiet operators set to nothing."""
    ops = [op | {"level": 0} if index in quiet else op for index, op in enumerate(OPS)]
    return fm.fm_tone({"alg": algorithm, "fb": fb, "ops": ops}, *NOTE, RATE)


def env(op: dict = OP, **new: float) -> dict:
    """An operator with some settings of its envelope set to other values."""
    return op | {"env": op.get("env", {}) | new}


class Wirings(unittest.TestCase):
    def test_the_wirings_are_the_tables_the_engine_plays_from_and_what_is_no_algorithm_is_refused(self) -> None:
        for algorithm in range(8):
            wired = wires(algorithm)
            self.assertEqual(wired.heard, tuple(np.flatnonzero(fm.CARRIER[algorithm]).tolist()))
            self.assertEqual(sorted(wired.changes), sorted((int(source), int(to)) for to, source
                                                           in zip(*np.nonzero(fm.INTO[algorithm]))))
            self.assertEqual(wired.feedback, fm.FEEDBACK_OPERATOR)
        self.assertEqual(wires(2), Wires((3,), ((1, 2), (0, 3), (2, 3)), 0))
        self.assertEqual(wires(5), Wires((1, 2, 3), ((0, 1), (0, 2), (0, 3)), 0))
        for wrong in (8, -1, 2.0, True, "4", None):
            for ask in (wires, told, places, lambda algorithm: picture(CELL, algorithm),
                        lambda algorithm: Wiring(ROW, algorithm), Wiring(ROW).pick):
                with self.assertRaises(RowSynthError, msg=repr(wrong)) as refused:
                    ask(wrong)
                self.assertEqual(str(refused.exception), f"an algorithm is a whole number from 0 to 7, not {wrong!r}")

    def test_the_sound_changes_exactly_when_the_wiring_says_an_operator_reaches_the_output(self) -> None:
        # The proof that the pictures tell the truth. A note is rendered twice, once as it is and once with an
        # operator's level set to nothing: in every algorithm, and with every choice of the others quiet already.
        counts = []
        for algorithm in range(8):
            wired = wires(algorithm)
            counts.append(0)
            for count in range(4):
                for quiet in map(frozenset, itertools.combinations(range(4), count)):
                    as_it_is = tone(algorithm, quiet)
                    for op in sorted(set(range(4)) - quiet):
                        differs = not np.array_equal(as_it_is, tone(algorithm, quiet | {op}))
                        self.assertEqual(differs, reaches(wired, op, quiet), (algorithm, sorted(quiet), op))
                        counts[-1] += differs
            # With none quiet, each of the four is heard or changes one that is.
            self.assertEqual([reaches(wired, op, frozenset()) for op in range(4)], [True] * 4)
        # Of the 32 choices in an algorithm, in so many the sound changes: the fewer, the longer its chains.
        self.assertEqual(counts, [15, 16, 18, 18, 24, 31, 28, 32])

    def test_the_feedback_is_heard_exactly_when_the_operator_it_is_on_reaches_the_output(self) -> None:
        for algorithm in range(8):
            wired = wires(algorithm)
            for count in range(5):
                for quiet in map(frozenset, itertools.combinations(range(4), count)):
                    differs = not np.array_equal(tone(algorithm, quiet), tone(algorithm, quiet, 0.9))
                    self.assertEqual(differs, reaches(wired, wired.feedback, quiet), (algorithm, sorted(quiet)))

    def test_a_wiring_in_plain_words(self) -> None:
        self.assertEqual([told(algorithm) for algorithm in range(8)], [
            "0 changes 1, 1 changes 2, 2 changes 3; 3 is heard",
            "0 and 1 change 2, 2 changes 3; 3 is heard",
            "1 changes 2, 0 and 2 change 3; 3 is heard",
            "0 changes 1, 1 and 2 change 3; 3 is heard",
            "0 changes 1, 2 changes 3; 1 and 3 are heard",
            "0 changes 1, 2 and 3; 1, 2 and 3 are heard",
            "0 changes 1; 1, 2 and 3 are heard",
            "none changes another; 0, 1, 2 and 3 are heard",
        ])


class Laid(unittest.TestCase):
    def test_every_operator_has_a_place_by_how_far_it_is_from_the_output_and_none_is_on_another(self) -> None:
        in_a_row = ((0, 0), (1, 0), (2, 0), (3, 0))
        self.assertEqual([places(algorithm) for algorithm in (0, 4, 5, 6, 7)], [in_a_row] * 5)
        # Two that are as far from the output share a column, the later a row up, and what changes one of them
        # stands in its row.
        self.assertEqual([places(algorithm) for algorithm in (1, 2, 3)],
                         [((0, 0), (0, 1), (1, 0), (2, 0)), ((1, 0), (0, 1), (1, 1), (2, 0)),
                          ((0, 0), (1, 0), (1, 1), (2, 0))])
        for algorithm in range(8):
            seen = picture(CELL, algorithm)
            self.assertEqual(len(seen.boxes), 4)
            taken = [pixels(box) for box in seen.boxes]
            self.assertEqual((len(set().union(*taken)), {len(box) for box in taken}), (4 * 121, {121}))
            # All of the picture is in its rectangle, clear of the outline.
            parts = pixels(*seen.boxes, *seen.stems, seen.base, seen.loop, *itertools.chain(*seen.lines))
            self.assertLessEqual(parts, pixels((11, 21, 72, 60)), algorithm)
        # The boxes are in the middle from side to side, whether they take four columns or three.
        self.assertEqual([[box[:2] for box in picture(CELL, algorithm).boxes] for algorithm in (4, 1)],
                         [[(17, 56), (33, 56), (49, 56), (65, 56)], [(25, 56), (25, 40), (41, 56), (57, 56)]])

    def test_the_heard_ones_stand_on_the_base_line(self) -> None:
        for algorithm in range(8):
            wired, seen = wires(algorithm), picture(CELL, algorithm)
            self.assertEqual({places(algorithm)[op][1] for op in wired.heard}, {0}, algorithm)
            self.assertEqual(len(seen.stems), len(wired.heard))
            for op, stem in zip(wired.heard, seen.stems):
                # From the middle of the foot of its box down to the line.
                self.assertEqual(stem, (seen.boxes[op][0] + 5, 67, 1, 6))
                self.assertLess(pixels((stem[0], 73, 1, 1)), pixels(seen.base))
        # The line runs a little past the outermost of them, so that one alone stands on something too.
        self.assertEqual([picture(CELL, algorithm).base for algorithm in (0, 4, 7)],
                         [(67, 73, 7, 1), (35, 73, 39, 1), (19, 73, 55, 1)])

    def test_a_changing_one_is_left_of_or_above_what_it_changes_and_its_line_leads_there(self) -> None:
        for algorithm in range(8):
            wired, spots, seen = wires(algorithm), places(algorithm), picture(CELL, algorithm)
            self.assertEqual(len(seen.lines), len(wired.changes))
            for (source, to), strips in zip(wired.changes, seen.lines):
                self.assertTrue(spots[source][0] < spots[to][0] or spots[source][1] > spots[to][1], (algorithm, source))
                # The line touches the two boxes and runs through none.
                line = pixels(*strips)
                self.assertFalse(line & pixels(*seen.boxes), (algorithm, source, to))
                for x, y, w, h in (seen.boxes[source], seen.boxes[to]):
                    self.assertTrue(line & pixels((x - 1, y, w + 2, h), (x, y - 1, w, h + 2)), (algorithm, source, to))
            # The loop of the feedback is on a corner of its operator, clear of every line and every other box.
            x, y, _, _ = seen.boxes[wired.feedback]
            self.assertEqual(seen.loop, (x - 3, y - 3, 6, 6))
            others = [box for op, box in enumerate(seen.boxes) if op != wired.feedback]
            self.assertFalse(pixels(seen.loop) & pixels(*others, *seen.stems, *itertools.chain(*seen.lines)), algorithm)
        # Straight to the next box of the row, with a head of two pixels at the changed one.
        self.assertEqual(picture(CELL, 1).lines[0], ((36, 61, 5, 1), (39, 60, 1, 1), (39, 62, 1, 1)))
        # From a row above: out to the right, and down into the top.
        self.assertEqual(picture(CELL, 1).lines[1], ((36, 45, 11, 1), (46, 46, 1, 10), (45, 54, 1, 1), (47, 54, 1, 1)))
        # Over one box, and higher over two.
        self.assertEqual(picture(CELL, 5).lines[1:],
                         (((22, 52, 1, 4), (22, 52, 33, 1), (54, 52, 1, 4), (53, 54, 1, 1), (55, 54, 1, 1)),
                          ((22, 48, 1, 8), (22, 48, 49, 1), (70, 48, 1, 8), (69, 54, 1, 1), (71, 54, 1, 1))))

    def test_the_pictures_share_the_row_with_a_gap_between_two(self) -> None:
        self.assertEqual(cells(ROW), [(3 + 80 * index, 5, 74, 62) for index in range(8)])
        # Pixels that do not go round are given to the pictures one at a time.
        self.assertEqual([cell[:3:2] for cell in cells((0, 0, 640, 40))],
                         [(0, 75), (81, 75), (162, 74), (242, 75), (323, 75), (404, 75), (485, 74), (565, 75)])
        self.assertEqual([cell[2] for cell in cells((0, 0, 30, 40))], [0] * 8)


class Picked(unittest.TestCase):
    def setUp(self) -> None:
        self.ui = Ui()
        self.told: list[int] = []
        self.shown = self.ui.add(Wiring(ROW, 4, changed=self.told.append, help="how the four operators are wired"))

    def click(self, down: tuple[int, int], up: tuple[int, int] | None = None, button: int = 1) -> bool:
        """Press a mouse button at a position and let it go, there or at another. Whether the press was used."""
        used = self.ui.handle(Event("down", down, button=button))
        self.ui.handle(Event("up", up or down, button=button))
        return used

    def test_a_click_picks_another_algorithm_and_tells_and_the_picked_one_clicked_again_tells_nothing(self) -> None:
        self.assertTrue(self.click((3 + 2 * 80 + 30, 40)))
        self.assertEqual((self.shown.algorithm, self.told), (2, [2]))
        # The first pixel of a picture and the last are its own.
        self.click((3 + 7 * 80, 5))
        self.click((3 + 80 + 73, 66))
        self.assertEqual((self.shown.algorithm, self.told), (1, [2, 7, 1]))
        # The picture the release is on is the one that is picked.
        self.assertTrue(self.click((3 + 5 * 80 + 30, 40), (3 + 6 * 80 + 30, 40)))
        self.assertEqual((self.shown.algorithm, self.told), (6, [2, 7, 1, 6]))
        self.told.clear()
        self.assertTrue(self.click((3 + 6 * 80 + 30, 40)))
        self.assertEqual((self.shown.algorithm, self.told), (6, []))
        # Nor does a release between two pictures or off the row, the right button, or a press between two.
        self.assertTrue(self.click((3 + 5 * 80 + 30, 40), (3 + 5 * 80 + 76, 40)))
        self.assertTrue(self.click((3 + 5 * 80 + 30, 40), (3 + 5 * 80 + 30, 90)))
        self.assertFalse(self.click((3 + 5 * 80 + 30, 40), button=3))
        self.assertFalse(self.click((3 + 5 * 80 + 76, 40)))
        self.assertFalse(self.shown.on(Event("key", key="a"), self.ui))
        self.assertEqual((self.shown.algorithm, self.told), (6, []))
        # Picked from outside, it tells as it does for a click.
        self.shown.pick(6)
        self.shown.pick(0)
        self.assertEqual((self.shown.algorithm, self.told), (0, [0]))

    def test_the_help_line_says_the_wiring_under_the_mouse_in_plain_words(self) -> None:
        said_here = []
        for place in ((3 + 80 + 30, 40), (3 + 4 * 80 + 30, 40), (3 + 4 * 80 + 76, 40)):
            self.ui.handle(Event("move", place))
            said_here.append(self.ui.help)
        self.assertEqual(said_here, ["0 and 1 change 2, 2 changes 3; 3 is heard",
                                     "0 changes 1, 2 changes 3; 1 and 3 are heard", "how the four operators are wired"])


class Sums(unittest.TestCase):
    def test_the_settings_are_the_operators_own_and_the_engines_where_it_names_none(self) -> None:
        self.assertEqual(settings(OP), {"a": 0.2, "d": 0.3, "s": 0.5, "sd": 0.0, "r": 0.2})
        self.assertEqual(settings({"mul": 1, "level": 0}), fm.ENVELOPE)
        self.assertEqual(fm.ENVELOPE, {"a": 0.0, "d": 0.3, "s": 1.0, "sd": 0.0, "r": 0.2})
        # They are the engine's: an envelope that names nothing and one that names them all are the same samples.
        np.testing.assert_array_equal(fm.envelope({}, 0.05, 0.4, RATE), fm.envelope(dict(fm.ENVELOPE), 0.05, 0.4, RATE))
        for wrong, message in (("loud", "an operator is a table of its settings, not 'loud'"),
                               ({"env": [0.1]}, "the envelope of an operator is a table of its settings, not [0.1]"),
                               ({"env": {"a": "0.1"}}, "the 'a' of an envelope is a number, not '0.1'"),
                               ({"env": {"s": True}}, "the 's' of an envelope is a number, not True")):
            for ask in (settings, length, lambda op: Envelope(RECT, op),
                        lambda op: setattr(Envelope(RECT, OP), "op", op)):
                with self.assertRaises(RowSynthError) as refused:
                    ask(wrong)
                self.assertEqual(str(refused.exception), message)

    def test_the_note_is_held_on_after_its_decay_and_a_pixel_is_a_time_and_a_level(self) -> None:
        self.assertEqual(HOLD, 0.3)
        for moment, wanted in zip(moments(settings(OP)), (0.2, 0.5, 0.8, 1.0)):
            self.assertAlmostEqual(moment, wanted)
        self.assertAlmostEqual(length(OP), 1.0)
        self.assertAlmostEqual(length({}), 0.8)
        # A time under 0 is none to the engine, and none here.
        self.assertEqual(moments({"a": -1, "d": -2, "s": 1, "sd": 0, "r": -3}), (0, 0, HOLD, HOLD))
        seen = Map(RECT, 1.0)
        self.assertEqual((seen.plot, seen.second, seen.rows), ((14, 24, 101, 51), 0.01, 50))
        self.assertEqual([seen.y(level) for level in (0, 0.5, 1, 0.009, 0.01, -3, 7)], [74, 49, 24, 74, 73, 74, 24])
        self.assertEqual(Map((10, 20, 3, 3), 1.0).plot, (14, 24, 1, 1))
        for wrong in (0, -0.5):
            for ask in (lambda seconds: Map(RECT, seconds), lambda seconds: Envelope(RECT, OP, seconds=seconds),
                        lambda seconds: setattr(Envelope(RECT, OP), "seconds", seconds)):
                with self.assertRaises(RowSynthError) as refused:
                    ask(wrong)
                self.assertEqual(str(refused.exception),
                                 f"the picture of an envelope shows more than 0 seconds, not {wrong!r}")

    def test_the_points_are_on_the_line_which_is_the_engines_envelope(self) -> None:
        self.assertEqual(Map(RECT, 1.0).spots(settings(OP)), SPOTS)
        ops = (OP, {}, env(a=0, d=0, s=0.25, sd=30, r=0.01), env(a=0.5, d=1, s=0, r=0.6), env(s=1, sd=96))
        for op, seconds in itertools.product(ops, (None, 0.4, 3)):
            seen = Map(RECT, seconds or length(op))
            values = settings(op)
            curve = seen.curve(values)
            self.assertEqual((len(curve.low), len(curve.high)), (101, 101))
            # The line begins at the foot, and goes on from column to column with no gap.
            self.assertEqual(curve.low[0], 0)
            for column in range(1, 101):
                self.assertLessEqual(curve.low[column], curve.high[column - 1])
                self.assertGreaterEqual(curve.high[column], curve.low[column - 1])
            # What is heard: the engine's own samples at the rate of a song, at the middle of every column.
            heard = fm.envelope(values, moments(values)[SUSTAIN], seen.seconds + 0.01, 44100)
            for column in range(1, 101):
                level = float(heard[round(column * seen.second * 44100) - 1])
                self.assertTrue(curve.low[column] - 0.02 <= level <= curve.high[column] + 0.02, (op, seconds, column))
            # The attack ends at the top, and the release 60 decibels under where the note was let go. From the
            # end of the decay the level falls as the sustain has it. A decay to the full level is over at once,
            # so there it has been falling since the attack.
            full, fallen = max(values["s"], 0.0001), falling(values)
            self.assertAlmostEqual(fallen, 0.6 if values["s"] == 1 else HOLD)
            heights = (1, full * 10 ** (-values["sd"] * (fallen - HOLD) / 20),
                       full * 10 ** (-values["sd"] * fallen / 20), 0)
            # Each point is at its time, and where it is in the picture the line passes through it.
            shown = 0
            for (column, level), moment, height in zip(curve.spots, moments(values), heights):
                self.assertLessEqual(abs(column - moment / seen.second), 1, (op, seconds))
                if column <= 100:
                    self.assertTrue(curve.low[column] <= level <= curve.high[column], (op, seconds))
                    self.assertAlmostEqual(level, height, places=2)
                    shown += 1
            self.assertEqual(shown, sum(moment <= seen.seconds + 1e-9 for moment in moments(values)))

    def test_a_point_moves_only_the_ways_it_has_a_setting_for(self) -> None:
        seen = Map(RECT, 1.0)
        self.assertEqual([(point.name, point.time, point.level) for point in POINTS],
                         [("attack", "a", None), ("decay", "d", "s"), ("sustain", None, "sd"), ("release", "r", None)])
        self.assertEqual([moved(OP, point, seen, (10, -5)) for point in range(4)],
                         [env(a=0.3), env(d=0.4, s=0.6), OP, env(r=0.3)])
        self.assertEqual([moved(OP, point, seen, (-10, 5)) for point in range(4)],
                         [env(a=0.1), env(d=0.2, s=0.4), env(sd=6.461), env(r=0.1)])
        # The sustain falls as fast as it takes to be at the height of the mouse when the note is let go.
        self.assertAlmostEqual(0.5 * 10 ** (-6.461 * HOLD / 20), 0.4, places=4)
        self.assertEqual(moved(env(sd=6.461), SUSTAIN, seen, (0, 5)), env(sd=14.79))
        self.assertEqual(moved(env(sd=6.461), SUSTAIN, seen, (0, -5)), env(sd=0))
        # From the full level it has the time of the decay to fall in as well. With nothing to fall from, it
        # does not move.
        self.assertEqual(moved(env(s=1), SUSTAIN, seen, (0, 5)), env(s=1, sd=1.525))
        self.assertAlmostEqual(10 ** (-1.525 * (0.3 + HOLD) / 20), 0.9, places=4)
        self.assertEqual(moved(env(s=0), SUSTAIN, seen, (0, 5)), env(s=0))
        # A point that is not moved leaves the operator as it was, and so does a setting that comes out as it was:
        # one the operator did not name is not written.
        for op in (OP, {"mul": 1, "level": 0}, {"env": {"a": 9, "s": 3, "sd": -2, "r": 0.2}}):
            self.assertEqual([moved(op, point, seen, (0, 0)) for point in range(4)], [op] * 4)
        self.assertEqual(moved({"level": 0}, ATTACK, seen, (-10, 0)), {"level": 0})
        self.assertEqual(moved({"level": 0}, DECAY, seen, (10, -5)), {"level": 0, "env": {"d": 0.4}})
        # The operator that is handed in is not changed.
        self.assertEqual(OP["env"], {"a": 0.2, "d": 0.3, "s": 0.5, "r": 0.2})

    def test_a_press_takes_the_nearest_point_within_reach_and_a_point_is_said_in_plain_words(self) -> None:
        self.assertEqual([nearest(SPOTS, pos) for pos in ((34, 24), (37, 21), (61, 52), (38, 24), (64, 45), (0, 0))],
                         [ATTACK, ATTACK, DECAY, None, None, None])
        # Of two as near it takes the later.
        on_each_other = [(14, 24), (14, 24), (44, 24), (47, 24)]
        self.assertEqual([nearest(on_each_other, (x, 24)) for x in (14, 45, 46)], [DECAY, SUSTAIN, RELEASE])
        values = settings(env(d=0.25, s=0.6, sd=4))
        self.assertEqual([said(point, values) for point in range(4)],
                         ["attack: 0.2 s", "decay: 0.25 s, level 0.6", "sustain: falls 4 dB a second",
                          "release: 0.2 s"])


class Handled(unittest.TestCase):
    def setUp(self) -> None:
        self.ui = Ui()
        self.changed: list[dict] = []
        self.settled: list[dict] = []
        self.shown = self.ui.add(Envelope(RECT, OP, self.changed.append, self.settled.append,
                                          help="how the level of the operator rises and falls"))

    def drag(self, *places: tuple[int, int], button: int = 1) -> bool:
        """Press at the first place, go by way of the others, and let go at the last. Whether the press was used."""
        used = self.ui.handle(Event("down", places[0], button=button))
        for place in places[1:]:
            self.ui.handle(Event("move", place))
        self.ui.handle(Event("up", places[-1], button=button))
        return used

    def test_the_attack_and_the_release_are_dragged_along_time_and_held_at_the_ends_of_their_ranges(self) -> None:
        self.assertTrue(self.drag((34, 24), (44, 9), (5000, 24), (-900, 24)))
        self.assertEqual(self.changed, [env(a=0.3), env(a=4), env(a=0)])
        self.assertEqual((self.settled, self.shown.op, RANGES["a"]), ([env(a=0)], env(a=0), (0, 4)))
        # A whole value is a whole number in the file.
        self.assertEqual([type(op["env"]["a"]) for op in self.changed], [float, int, int])
        self.shown.op = OP
        self.assertTrue(self.drag((114, 74), (124, 60), (5000, 74), (-900, 74)))
        self.assertEqual(self.changed[3:], [env(r=0.3), env(r=4), env(r=0)])
        self.assertEqual((self.settled[1:], RANGES["r"]), ([env(r=0)], (0, 4)))

    def test_the_decay_is_dragged_along_time_and_along_level(self) -> None:
        self.assertTrue(self.drag((64, 49), (74, 39), (74, 49), (5000, -5000), (-5000, 5000)))
        self.assertEqual(self.changed, [env(d=0.4, s=0.7), env(d=0.4), env(d=4, s=1), env(d=0, s=0)])
        self.assertEqual((self.settled, RANGES["d"], RANGES["s"]), ([env(d=0, s=0)], (0, 4), (0, 1)))

    def test_the_sustain_is_dragged_along_level_and_sets_how_fast_the_level_falls(self) -> None:
        self.assertTrue(self.drag((94, 49), (104, 54), (94, 5000), (60, 59)))
        self.assertEqual(self.changed, [env(sd=6.461), env(sd=96), env(sd=14.79)])
        self.assertEqual((self.settled, RANGES["sd"]), ([env(sd=14.79)], (0, 96)))
        # It rises no higher than the level the decay falls to: there the level does not fall at all.
        spot = self.shown.map.spots(settings(self.shown.op))[SUSTAIN]
        self.assertEqual(spot, (94, 59))
        self.assertTrue(self.drag(spot, (94, -5000)))
        self.assertEqual((self.changed[3:], self.settled[1:]), ([env(sd=0)], [env(sd=0)]))

    def test_changed_is_told_on_each_move_and_settled_once_when_the_hand_lets_go(self) -> None:
        self.ui.handle(Event("down", (64, 49), button=1))
        self.assertEqual((self.changed, self.settled), ([], []))
        for place in ((65, 49), (66, 48), (66, 48), (70, 44)):
            self.ui.handle(Event("move", place))
        # A move that leaves it as it is tells nothing. The operator is what a release now would give.
        self.assertEqual(self.changed, [env(d=0.31), env(d=0.32, s=0.52), env(d=0.36, s=0.6)])
        self.assertEqual((self.shown.op, self.settled, self.shown.busy), (env(d=0.36, s=0.6), [], True))
        # The picture keeps its clock until the release, so that the point stays under the mouse.
        self.assertEqual((self.shown.map.seconds, self.shown.map.spots(settings(self.shown.op))[DECAY]),
                         (1.0, (70, 44)))
        self.ui.handle(Event("up", (71, 44), button=1))
        self.assertEqual((self.changed[3:], self.settled), ([env(d=0.37, s=0.6)], [env(d=0.37, s=0.6)]))
        self.assertFalse(self.shown.busy)
        self.assertAlmostEqual(self.shown.map.seconds, 1.07)
        # What is handed out is a copy: the widget's own operator is not changed through it.
        self.settled[0]["env"]["s"] = 0
        self.shown.op["env"]["s"] = 0
        self.assertEqual(self.shown.op, env(d=0.37, s=0.6))

    def test_nothing_is_settled_when_the_hand_lets_go_where_it_began(self) -> None:
        self.assertTrue(self.drag((64, 49)))
        self.assertEqual((self.changed, self.settled, self.shown.op), ([], [], OP))
        # Out and back: each way is told, and the operator is the one it began with, key for key.
        self.assertTrue(self.drag((65, 50), (90, 30), (65, 50)))
        self.assertEqual((self.changed, self.settled, self.shown.op), ([env(d=0.55, s=0.9), OP], [], OP))
        self.assertEqual(list(self.shown.op["env"]), ["a", "d", "s", "r"])
        # One that names no envelope names none after it either.
        bare = {"mul": 1, "level": 0}
        self.shown.op = bare
        spot = self.shown.map.spots(settings(bare))[RELEASE]
        self.assertTrue(self.drag(spot, (spot[0] - 20, spot[1]), spot))
        self.assertEqual((self.changed[2:], self.settled), ([bare | {"env": {"r": 0.04}}, bare], []))
        self.assertEqual((self.shown.op, bare), (bare, {"mul": 1, "level": 0}))

    def test_a_press_that_is_on_no_point_or_not_of_the_left_button_is_not_used(self) -> None:
        self.assertEqual([self.drag((50, 40), (70, 70)), self.drag((64, 49), (80, 30), button=3),
                          self.drag((68, 49), (80, 30))], [False] * 3)
        # Another button while a point is dragged changes nothing, and the wheel and the keys are not used.
        self.ui.handle(Event("down", (64, 49), button=1))
        events = (Event("down", (64, 49), button=3), Event("up", (64, 49), button=3),
                  Event("wheel", (64, 49), wheel=1), Event("key", key="a"))
        self.assertEqual([self.shown.on(event, self.ui) for event in events], [False] * 4)
        self.assertEqual((self.changed, self.settled), ([], []))
        self.ui.handle(Event("move", (74, 49)))
        self.ui.handle(Event("up", (74, 49), button=1))
        self.assertEqual((self.changed, self.settled), ([env(d=0.4)], [env(d=0.4)]))
        # One that is not enabled takes nothing, and says its own line.
        self.shown.enabled = False
        self.assertFalse(self.drag((74, 49), (80, 30)))
        self.assertEqual((len(self.changed), self.ui.help), (1, "how the level of the operator rises and falls"))

    def test_another_operator_is_shown_when_it_is_set_and_two_points_on_each_other_are_pulled_apart(self) -> None:
        # A drag that is under way is dropped: the operator is another one now.
        self.ui.handle(Event("down", (64, 49), button=1))
        self.ui.handle(Event("move", (74, 39)))
        self.assertTrue(self.shown.busy)
        self.shown.op = {"mul": 1, "level": 0}
        self.assertFalse(self.shown.busy)
        self.ui.handle(Event("move", (84, 29)))
        self.ui.handle(Event("up", (84, 29), button=1))
        self.assertEqual((self.changed, self.settled, self.shown.op), ([env(d=0.4, s=0.7)], [], {"mul": 1, "level": 0}))
        # Its picture is as long as its note, unless it is told how long to be.
        self.assertAlmostEqual(self.shown.map.seconds, 0.8)
        self.shown.seconds = 1.0
        self.assertEqual((self.shown.map.seconds, self.shown.map.second), (1.0, 0.01))
        # With no attack and no decay the two points are on each other. The later is taken, and frees the other.
        self.shown.op = env(a=0, d=0, s=1)
        spots = self.shown.map.spots(settings(self.shown.op))
        self.assertEqual((spots[ATTACK], spots[DECAY], self.shown.at((14, 24))), ((14, 24), (14, 24), DECAY))
        self.drag((14, 24), (34, 24))
        self.assertEqual(self.settled, [env(a=0, d=0.2, s=1)])
        self.assertEqual((self.shown.at((14, 24)), self.shown.at((34, 24))), (ATTACK, DECAY))

    def test_the_help_line_names_the_point_under_the_mouse_and_what_it_is_set_to(self) -> None:
        said_here = []
        for place in (*SPOTS, (36, 22), (50, 40)):
            self.ui.handle(Event("move", place))
            said_here.append(self.ui.help)
        self.assertEqual(said_here, ["attack: 0.2 s", "decay: 0.3 s, level 0.5", "sustain: falls 0 dB a second",
                                     "release: 0.2 s", "attack: 0.2 s",
                                     "how the level of the operator rises and falls"])
        # While a point is dragged the line is its own, with what a release now would give.
        self.ui.handle(Event("down", (64, 49), button=1))
        self.ui.handle(Event("move", (59, 34)))
        self.assertEqual(self.ui.help, "decay: 0.25 s, level 0.8")
        self.ui.handle(Event("move", (34, 24)))
        self.assertEqual(self.ui.help, "decay: 0 s, level 1")


@unittest.skipUnless(pygame, NEEDS)
class Drawn(unittest.TestCase):
    def setUp(self) -> None:
        self.surface = pygame.Surface((640, 90))
        self.canvas = Canvas(self.surface, Font(sprites.load("font")))
        self.ui = Ui()

    def draw(self) -> None:
        """Draw the widgets on a surface of the colour that is not theirs."""
        self.surface.fill(OUTSIDE)
        self.ui.draw(self.canvas)

    def at(self, *pos: int) -> tuple[int, ...]:
        """The colour of one pixel."""
        return tuple(self.surface.get_at(pos))[:3]

    def ink(self, rect: layout.Rect, colour: theme.Colour) -> set[tuple[int, int]]:
        """The pixels of a rectangle that are of a colour."""
        return {pos for pos in pixels(rect) if self.at(*pos) == colour}

    def clean(self, *rects: layout.Rect) -> bool:
        """Whether all of some rectangles was drawn on, and nothing outside them."""
        self.draw()
        return self.ink((0, 0, 640, 90), OUTSIDE) == pixels((0, 0, 640, 90)) - pixels(*rects)

    def written(self, text: str, rect: layout.Rect) -> set[tuple[int, int]]:
        """The pixels a number takes in the middle of a rectangle."""
        page = pygame.Surface((640, 90))
        Canvas(page, self.canvas.font).text_in(text, rect, theme.TEXT, "centre")
        return {pos for pos in pixels(rect) if page.get_at(pos)[:3] == theme.TEXT}

    def envelope(self, op: dict = OP, **more: object) -> Envelope:
        """An envelope on the Ui, in place of what was there."""
        self.ui.widgets.clear()
        return self.ui.add(Envelope(RECT, op, **more))

    def line(self, shown: Envelope) -> set[tuple[int, int]]:
        """The pixels of the line of an envelope, as the engine's envelope has it column by column."""
        seen = shown.map
        curve = seen.curve(settings(shown.op))
        return {(14 + column, y) for column, (low, high) in enumerate(zip(curve.low, curve.high))
                for y in range(seen.y(high), seen.y(low) + 1)} & pixels((11, 21, 107, 57))

    def test_the_eight_wirings_are_eight_pictures_and_the_picked_one_is_lit(self) -> None:
        shown = self.ui.add(Wiring(ROW, 4))
        self.draw()
        for algorithm, (x, y, w, h) in enumerate(cells(ROW)):
            picked = algorithm == 4
            # Its outline, its face, and its number in the corner.
            self.assertEqual({self.at(*pos) for pos in ((x, y), (x + w - 1, y + h - 1), (x + 30, y), (x, y + 30))},
                             {theme.ACTIVE if picked else theme.LINE})
            self.assertEqual({self.at(x + 1, y + 1), self.at(x + w - 2, y + h - 2)},
                             {theme.LIT if picked else theme.RAISED})
            self.assertEqual(self.ink((x + 4, y + 3, 5, 9), theme.ACTIVE if picked else theme.DIM),
                             self.written(str(algorithm), (x + 4, y + 3, 5, 9)))
        # Between two pictures nothing is drawn, nor anywhere off the row.
        self.assertTrue(self.clean(*cells(ROW)))
        # The picture under the mouse is lit too, and keeps its outline.
        self.ui.handle(Event("move", (3 + 80 + 30, 40)))
        shown.pick(6)
        self.draw()
        self.assertEqual([self.at(3 + 80 * algorithm + 1, 6) for algorithm in (1, 4, 6)],
                         [theme.LIT, theme.RAISED, theme.LIT])
        self.assertEqual([self.at(3 + 80 * algorithm, 5) for algorithm in (1, 4, 6)],
                         [theme.LINE, theme.LINE, theme.ACTIVE])

    def test_the_heard_operators_are_in_the_colour_of_what_is_on_and_the_others_in_that_of_a_song(self) -> None:
        self.ui.add(Wiring(ROW, 4))
        self.draw()
        for algorithm, cell in enumerate(cells(ROW)):
            wired, seen = wires(algorithm), picture(cell, algorithm)
            for op, (x, y, w, h) in enumerate(seen.boxes):
                heard = op in wired.heard
                # A heard one is a full box with its number cut out, another an outline with its number in it.
                self.assertEqual({self.at(*pos) for pos in ((x, y), (x + w - 1, y + h - 1), (x + 5, y), (x, y + 5))},
                                 {theme.ACTIVE if heard else theme.SONG})
                self.assertEqual(self.at(x + 1, y + 1), theme.ACTIVE if heard else theme.BACKGROUND)
                self.assertEqual(self.ink((x + 1, y + 1, 9, 9), theme.BACKGROUND if heard else theme.SONG),
                                 self.written(str(op), (x, y, w, h)))
            # The base line and what stands on it, and each line from a changing one to the one it changes.
            self.assertLessEqual(pixels(seen.base, *seen.stems), self.ink(cell, theme.ACTIVE))
            self.assertLessEqual(pixels(*itertools.chain(*seen.lines)), self.ink(cell, theme.SONG))
            # Nothing else is in those colours but the number of the algorithm and the loop of the feedback.
            x, y, w, h = cell
            inside = (x + 1, y + 1, w - 2, h - 2)
            drawn = (self.ink(inside, theme.ACTIVE) | self.ink(inside, theme.SONG)) - pixels((x + 4, y + 4, 5, 9))
            loop = pixels(seen.loop) - pixels((seen.loop[0] + 1, seen.loop[1] + 1, 4, 4))
            self.assertEqual(drawn - pixels(*seen.boxes),
                             (pixels(seen.base, *seen.stems, *itertools.chain(*seen.lines)) | loop)
                             - pixels(*seen.boxes), algorithm)

    def test_the_feedback_is_a_small_loop_on_the_operator_it_is_on(self) -> None:
        self.ui.add(Wiring(ROW, 0))
        self.draw()
        for algorithm, cell in enumerate(cells(ROW)):
            wired, seen = wires(algorithm), picture(cell, algorithm)
            face = theme.LIT if algorithm == 0 else theme.RAISED
            for op, (x, y, _, _) in enumerate(seen.boxes):
                # Out of the top of the box, round the corner, and back in at its side.
                loop = pixels((x - 3, y - 3, 6, 1), (x - 3, y - 3, 1, 6), (x - 3, y + 2, 3, 1), (x + 2, y - 3, 1, 3))
                ink = theme.ACTIVE if op in wired.heard else theme.SONG
                self.assertEqual({self.at(*pos) for pos in loop}, {ink if op == wired.feedback else face},
                                 (algorithm, op))
                self.assertEqual(self.at(x - 2, y - 2), face)
        # It is on the first operator in every algorithm, in the colour that one has there.
        self.assertEqual({wires(algorithm).feedback for algorithm in range(8)}, {0})
        self.assertEqual((self.at(3 + 4 * 80 + 7 - 3, 5 + 36 - 3), self.at(3 + 7 * 80 + 7 - 3, 5 + 36 - 3)),
                         (theme.SONG, theme.ACTIVE))

    def test_the_line_passes_through_its_points_is_the_engines_and_shows_where_the_note_is_let_go(self) -> None:
        shown = self.envelope(colour=theme.SONG)
        self.draw()
        squares = pixels(*((x - 2, y - 2, 5, 5) for x, y in SPOTS))
        self.assertEqual(self.ink(RECT, theme.SONG), self.line(shown) - squares)
        # A point is a square with a dark rim, its middle on the line.
        for x, y in SPOTS:
            self.assertIn((x, y), self.line(shown))
            self.assertEqual(self.ink((x - 2, y - 2, 5, 5), theme.TEXT), pixels((x - 1, y - 1, 3, 3)))
            self.assertEqual(self.ink((x - 2, y - 2, 5, 5), theme.BACKGROUND),
                             pixels((x - 2, y - 2, 5, 5)) - pixels((x - 1, y - 1, 3, 3)))
        # At three times, in the attack, the decay and the release, it is at the level the engine's samples have.
        heard = fm.envelope(OP["env"], 0.8, 1.0, 44100)
        levels = [float(heard[round(time * 44100) - 1]) for time in (0.1, 0.35, 0.9)]
        for level, wanted in zip(levels, (0.5, 0.7071, 0.0158)):
            self.assertAlmostEqual(level, wanted, places=3)
        self.assertEqual([shown.map.y(level) for level in levels], [49, 39, 73])
        self.assertEqual([self.at(*pos) for pos in ((24, 49), (49, 39), (104, 73))], [theme.SONG] * 3)
        self.assertEqual([self.at(*pos) for pos in ((24, 44), (24, 54), (49, 36), (49, 42), (104, 71), (104, 74))],
                         [theme.BACKGROUND] * 6)
        # The field is dark in its outline.
        self.assertEqual((self.at(10, 20), self.at(118, 78), self.at(11, 21)),
                         (theme.LINE, theme.LINE, theme.BACKGROUND))
        # A faint upright line stands where the note is let go: from the top of the field to its foot, behind
        # the line and the point.
        upright = pixels((94, 21, 1, 57)) - self.line(shown) - pixels((92, 47, 5, 5))
        self.assertEqual(self.ink(RECT, theme.RAISED), upright)
        self.assertEqual(len(upright), 57 - 7)
        # It goes with the note: a longer decay is let go later.
        shown.seconds = 1.0
        shown.op = env(d=0.4)
        self.draw()
        self.assertEqual({x for x, _ in self.ink(RECT, theme.RAISED)}, {104})

    def test_a_drag_shows_what_the_release_would_give_and_its_point_is_lit(self) -> None:
        shown = self.envelope()
        self.ui.handle(Event("move", (64, 49)))
        self.draw()
        # The point under the mouse is a full square.
        self.assertEqual((self.ink((62, 47, 5, 5), theme.TEXT), self.at(32, 22)),
                         (pixels((62, 47, 5, 5)), theme.BACKGROUND))
        self.ui.handle(Event("down", (64, 49), button=1))
        self.ui.handle(Event("move", (74, 39)))
        self.draw()
        self.assertEqual(shown.op, env(d=0.4, s=0.7))
        self.assertEqual(self.ink((72, 37, 5, 5), theme.TEXT), pixels((72, 37, 5, 5)))
        self.assertEqual(self.ink(RECT, theme.ACTIVE),
                         self.line(shown) - pixels(*((x - 2, y - 2, 5, 5) for x, y in ((34, 24), (74, 39), (104, 39)))))
        self.assertEqual({x for x, _ in self.ink(RECT, theme.RAISED)}, {104})
        # The end of the release is past the picture until the hand lets go, and then all of the note is in it.
        self.assertEqual(shown.map.spots(settings(shown.op))[RELEASE], (124, 74))
        self.ui.handle(Event("up", (74, 39), button=1))
        self.assertEqual(shown.map.spots(settings(shown.op)), [(32, 24), (69, 39), (96, 39), (114, 74)])

    def test_an_envelope_that_is_not_enabled_is_dim_and_nothing_is_drawn_outside_the_rectangle_of_either(self) -> None:
        shown = self.envelope()
        shown.enabled = False
        self.ui.handle(Event("move", (64, 49)))
        self.draw()
        self.assertEqual((self.ink(RECT, theme.ACTIVE), self.ink(RECT, theme.TEXT)), (set(), set()))
        squares = pixels(*((x - 2, y - 2, 5, 5) for x, y in SPOTS))
        self.assertEqual(self.ink(RECT, theme.DIM), self.line(shown) - squares)
        self.assertEqual([self.at(*pos) for pos in ((64, 49), (62, 47), (10, 20))],
                         [theme.LINE, theme.BACKGROUND, theme.LINE])
        # A picture that ends before the note does, a drag far off, and rectangles too small for anything.
        shown = self.envelope(env(a=3, d=4, r=4, sd=9))
        self.assertTrue(self.clean(RECT))
        shown.seconds = 0.5
        self.assertTrue(self.clean(RECT))
        shown.seconds = 4
        self.ui.handle(Event("down", shown.map.spots(settings(shown.op))[ATTACK], button=1))
        self.ui.handle(Event("move", (-700, 900)))
        self.assertEqual(shown.op, env(a=0, d=4, r=4, sd=9))
        self.assertTrue(self.clean(RECT))
        for small in ((10, 20, 30, 7), (10, 20, 2, 40), (10, 20, 0, 0)):
            self.ui.widgets.clear()
            self.ui.add(Envelope(small, OP))
            self.assertTrue(self.clean(small))
        # A row too low and too narrow for its pictures shows of each what its rectangle has room for.
        for small in ((10, 20, 300, 30), (10, 20, 30, 7), (10, 20, 0, 0)):
            self.ui.widgets.clear()
            self.ui.add(Wiring(small, 3))
            self.assertTrue(self.clean(*cells(small)))

    def test_every_fm_instrument_of_the_piece_shows_in_both_and_a_drag_out_and_back_leaves_it_as_it_was(self) -> None:
        instruments = json.loads((EXAMPLES / "piece.json").read_text(encoding="utf-8"))["instruments"]
        voices = {name: inst for name, inst in instruments.items() if inst.get("wave") == "fm"}
        self.assertEqual(sorted(voices), ["fmarp", "fmbrass", "fmdrive", "fmpad", "fmstab", "stamp", "sub"])
        for name, inst in voices.items():
            self.ui.widgets.clear()
            row = self.ui.add(Wiring(ROW, inst["alg"]))
            self.draw()
            self.assertEqual((self.at(3 + 80 * inst["alg"], 5), row.algorithm), (theme.ACTIVE, inst["alg"]))
            heard = wires(inst["alg"]).heard
            self.assertEqual(len(inst["ops"]), 4)
            for index, op in enumerate(inst["ops"]):
                changed: list[dict] = []
                settled: list[dict] = []
                shown = self.envelope(op, changed=changed.append, settled=settled.append,
                                      colour=theme.ACTIVE if index in heard else theme.SONG)
                self.draw()
                self.assertEqual((self.at(10, 20), shown.op), (theme.LINE, op))
                # The picture is as long as the note, so all four points are in it, each to be taken.
                spots = shown.map.spots(settings(op))
                self.assertLessEqual(set(spots), pixels(shown.map.plot), (name, index))
                for point, (x, y) in enumerate(spots):
                    self.assertEqual(shown.at((x, y)), point, (name, index))
                    self.ui.handle(Event("down", (x, y), button=1))
                    self.ui.handle(Event("move", (x + 7, y + 5)))
                    self.draw()
                    self.assertNotEqual(shown.op, op, (name, index, point))
                    self.ui.handle(Event("move", (x, y)))
                    self.ui.handle(Event("up", (x, y), button=1))
                    self.assertEqual((shown.op, changed[-1]), (op, op), (name, index, point))
                self.assertEqual((len(changed), settled), (8, []))


if __name__ == "__main__":
    unittest.main()
