"""Tests for rowsynth.gui.tables: the shapes of a step table, the sums of its bars, its gestures and what it draws."""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from rowsynth import note
from rowsynth.errors import RowSynthError
from rowsynth.gui import layout, sprites, theme
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.font import Font
from rowsynth.gui.tables import (DRAW, FAINT, GAP, LOOP, LOOP_HELP, MOST, SIZE, SIZE_HELP, ZERO, Map, StepTable, at,
                                 begins, counted, fit, passed, read, sized, told, write)
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
SHIFT = frozenset({"shift"})

#: A table whose bars are in (12, 32, 116, 45): three rows of pixels to a level from 0 to 15, the foot at y 77.
RECT = (10, 30, 120, 54)
#: One whose bars are in (12, 32, 116, 49): from -12 to 12 that is two rows to a semitone round a zero line at y 56.
TALL = (10, 30, 120, 58)
#: The surface they are drawn on, which is larger on every side, and the colour of what is not theirs.
SIZE_PX = (140, 110)
OUTSIDE = theme.GOOD

#: Eight levels, eight sweeps and four widths: at eight steps or fewer a bar is 9 wide, and a step 10.
VOL = [15, 14, 12, 10, 8, 6, 4, 2]
PITCH = [0, -4, -4, -4, -3, -3, -3, -3]
DUTY = [0, 1, 2, 1]
#: The ranges of the three, and what the engine takes for one that an instrument does not have.
RANGES = {"vol": (0, 15, 15), "pitch": (-12, 12, 0), "duty": (0, 3, 2)}


def cells(*rects: layout.Rect) -> set[tuple[int, int]]:
    """Every pixel of some rectangles."""
    return {(x + column, y + row) for x, y, w, h in rects for column in range(w) for row in range(h)}


class Shapes(unittest.TestCase):
    def test_every_shape_the_file_allows_is_read_into_steps_and_a_loop(self) -> None:
        self.assertEqual((read(15), read(0.5)), (((15,), None), ((0.5,), None)))
        self.assertEqual((read([0, 3, 3, 2]), read([])), (((0, 3, 3, 2), None), ((), None)))
        self.assertEqual(read({"v": [0, -1], "loop": 1}), ((0, -1), 1))
        # A loop that is no index of the list is none to the engine: the table holds its last step.
        for loop in (-1, 2, 99):
            self.assertEqual(read({"v": [0, -1], "loop": loop}), ((0, -1), None))
        self.assertEqual(read({"v": [], "loop": 0}), ((), None))

    def test_a_table_is_written_back_as_it_was_read_and_in_the_shortest_shape_that_says_the_same(self) -> None:
        for table in (15, 0.5, [0, 3, 3, 2], [1.5, 2], [], {"v": [0, -1], "loop": 1}, {"v": [2], "loop": 0}):
            self.assertEqual(write(*read(table)), table)
        # One step is the number alone, and a loop that is none leaves a list.
        longer = ([7], {"v": [1, 2], "loop": -1}, {"v": [4], "loop": 9}, {"v": [], "loop": 3})
        self.assertEqual([write(*read(table)) for table in longer], [7, [1, 2], 4, []])
        for table in longer:
            self.assertEqual([note.step(write(*read(table)), tick, 9) for tick in range(6)],
                             [note.step(table, tick, 9) for tick in range(6)])
        self.assertEqual((write((3, 4)), write((3, 4), 0)), ([3, 4], {"v": [3, 4], "loop": 0}))

    def test_what_is_no_table_is_refused(self) -> None:
        wrong = ("loud", None, True, [1, "2"], [1, False], {"v": [1]}, {"v": [1], "loop": 0, "x": 1},
                 {"v": 1, "loop": 0}, {"v": [1], "loop": 0.0}, {"v": [1], "loop": True})
        for table in wrong:
            with self.assertRaises(RowSynthError, msg=repr(table)):
                read(table)
        with self.assertRaises(RowSynthError) as refused:
            write((1, 2), 2)
        self.assertEqual(str(refused.exception), "a table of 2 steps cannot go round from step 3")
        with self.assertRaises(RowSynthError) as refused:
            StepTable(RECT, VOL, 1, 15)
        self.assertEqual(str(refused.exception),
                         "the values of a step table go from 0 or less up to more than 0, not from 1 to 15")
        with self.assertRaises(RowSynthError):
            Map(RECT, -12, 0)


class Sums(unittest.TestCase):
    def test_the_parts_of_the_rectangle(self) -> None:
        seen = Map(RECT, 0, 15)
        # The field begins at the top: the name of a table and the count of its steps are written by its owner.
        self.assertEqual((seen.field, seen.bars), ((10, 30, 120, 49), (12, 32, 116, 45)))
        self.assertEqual((seen.zone, seen.strip), ((12, 30, 116, 49), (12, 79, 116, 5)))
        # After eight bars: the handle, and the bars of the ticks past the end. Under them the bracket of a loop.
        self.assertEqual((seen.handle(8), seen.after(8, 0), seen.after(8, 2)), ((92, 30, 3, 49), 96, 116))
        self.assertEqual((seen.bracket(3, 8), seen.bracket(7, 8)), ((42, 80, 49, 4), (82, 80, 9, 4)))

    def test_the_x_of_a_step_and_the_step_at_an_x_are_each_others_opposite(self) -> None:
        seen = Map(RECT, 0, 15)
        self.assertEqual([seen.x(step) for step in (0, 1, 7)], [12, 22, 82])
        self.assertEqual([seen.step_at(x) for x in (2, 11, 12, 21, 22, 91, 92)], [-1, -1, 0, 0, 1, 7, 8])
        for bar, first in ((9, 0), (2, 0), (2, 40), (5, 13)):
            seen = Map(RECT, 0, 15, bar, first)
            self.assertEqual([seen.step_at(seen.x(step)) for step in range(MOST)], list(range(MOST)))
            # Every pixel of a bar, and of the gap after it, is its step's.
            self.assertEqual({seen.step_at(seen.x(5) + pixel) for pixel in range(bar + GAP)}, {5})
        self.assertEqual((Map(RECT, 0, 15, 2, 40).x(0), Map(RECT, 0, 15, 2, 40).step_at(12)), (-28, 13))

    def test_in_a_range_from_0_the_bars_stand_on_the_foot(self) -> None:
        seen = Map(RECT, 0, 15)
        self.assertEqual((seen.line, seen.up, seen.down), (77, 45, 0))
        self.assertEqual([seen.y(value) for value in (0, 1, 8, 15)], [77, 74, 53, 32])
        self.assertEqual([seen.value_at(seen.y(value)) for value in range(16)], list(range(16)))
        # A value out of the range is its nearer end, and so is a height off the bars.
        self.assertEqual((seen.y(99), seen.y(-3), seen.value_at(-40), seen.value_at(500)), (32, 77, 15, 0))
        # A bar of no height is one pixel in the lowest row, so that the step is seen.
        self.assertEqual((seen.block(0, 15), seen.block(2, 4), seen.block(1, 0)),
                         ((12, 32, 9, 45), (32, 65, 9, 12), (22, 76, 9, 1)))

    def test_in_a_range_both_ways_the_bars_stand_on_a_zero_line_and_hang_from_it(self) -> None:
        seen = Map(TALL, -12, 12)
        self.assertEqual((seen.line, seen.up, seen.down), (56, 24, 24))
        self.assertEqual([seen.y(value) for value in (12, 1, 0, -1, -12)], [32, 54, 56, 58, 80])
        self.assertEqual([seen.value_at(seen.y(value)) for value in range(-12, 13)], list(range(-12, 13)))
        self.assertEqual((seen.value_at(0), seen.value_at(500), seen.value_at(56)), (12, -12, 0))
        # Above the line a bar ends just over it, under the line it begins just under it, and 0 is on the line.
        self.assertEqual((seen.block(0, 3), seen.block(1, -4), seen.block(2, 0)),
                         ((12, 50, 9, 6), (22, 57, 9, 8), (32, 56, 9, 1)))
        # A range that goes further one way has its line where the two parts meet.
        uneven = Map(TALL, -6, 18)
        self.assertEqual((uneven.line, uneven.up, uneven.down), (68, 36, 12))

    def test_values_that_are_not_whole_are_set_to_a_few_places_and_every_row_is_one(self) -> None:
        seen = Map(RECT, 0, 1, whole=False)
        self.assertEqual([seen.y(seen.value_at(y)) for y in range(32, 78)], list(range(32, 78)))
        self.assertEqual((seen.value_at(47), seen.value_at(62), seen.value_at(32)), (0.667, 0.333, 1))
        both = Map(TALL, -1, 1, whole=False)
        self.assertEqual([both.y(both.value_at(y)) for y in range(32, 81)], list(range(32, 81)))
        self.assertEqual((both.value_at(44), both.value_at(68), both.value_at(56)), (0.5, -0.5, 0))
        # A whole value is a whole number in the file, in a table of either kind.
        self.assertEqual([type(value) for value in (seen.value_at(32), both.value_at(56), Map(RECT, 0, 15).kept(3.6))],
                         [int, int, int])

    def test_the_bars_get_narrower_down_to_2_pixels_and_after_that_the_table_scrolls(self) -> None:
        self.assertEqual([fit(116, count) for count in (0, 1, 8, 11, 12, 20, 37, 38, MOST)],
                         [9, 9, 9, 9, 8, 4, 2, 2, 2])
        narrow = Map(RECT, 0, 15, 2)
        self.assertEqual((narrow.reach(37), narrow.reach(MOST)), (114, 195))
        # It scrolls no further than to the handle at the right edge, and not at all when all of it fits.
        self.assertEqual([Map(RECT, 0, 15, 2, first).within(MOST).first for first in (-5, 0, 40, 79, 500)],
                         [0, 0, 40, 79, 79])
        self.assertEqual(Map(RECT, 0, 15, 2, 30).within(37).first, 0)

    def test_a_fast_drag_leaves_no_step_behind(self) -> None:
        seen = Map(RECT, 0, 15)
        # While the mouse stays on a step, that step has the height the mouse is at now.
        self.assertEqual(passed(seen, 8, (15, 41), (17, 50)), [(0, 9)])
        self.assertEqual(passed(seen, 8, (15, 41), (15, 41)), [(0, 12)])
        # From the first step to the fifth in one move, falling from 15 to 3: each between has the height on the way.
        self.assertEqual(passed(seen, 8, (15, 32), (55, 68)), [(1, 11), (2, 8), (3, 5), (4, 3)])
        self.assertEqual(passed(seen, 8, (55, 68), (15, 32)), [(3, 5), (2, 8), (1, 11), (0, 15)])
        # Only the steps the table has are given, however far the mouse went.
        self.assertEqual(passed(seen, 3, (15, 32), (55, 68)), [(1, 11), (2, 8)])
        self.assertEqual(passed(seen, 8, (15, 41), (900, 41)), [(step, 12) for step in range(1, 8)])
        self.assertEqual(passed(seen, 8, (25, 41), (-300, 41)), [(0, 12)])

    def test_past_its_end_a_table_is_what_the_engine_reads_there(self) -> None:
        self.assertEqual([at(DUTY, None, tick) for tick in range(8)], [0, 1, 2, 1, 1, 1, 1, 1])
        self.assertEqual([at(DUTY, 0, tick) for tick in range(9)], [0, 1, 2, 1, 0, 1, 2, 1, 0])
        self.assertEqual([at(DUTY, 2, tick) for tick in range(4, 9)], [2, 1, 2, 1, 2])
        self.assertEqual((at((5,), None, 40), at((), None, 3)), (5, 0))
        for table in (7, VOL, {"v": PITCH, "loop": 4}, {"v": DUTY, "loop": 3}):
            self.assertEqual([at(*read(table), tick) for tick in range(30)],
                             [note.step(table, tick, 0) for tick in range(30)])

    def test_a_table_is_made_longer_by_its_last_step_and_never_shorter_than_one_step(self) -> None:
        self.assertEqual(sized((3, 5), None, 4), ((3, 5, 5, 5), None))
        self.assertEqual((sized((3, 5, 7), 1, 2), sized((3, 5, 7), 2, 2)), (((3, 5), 1), ((3, 5), None)))
        self.assertEqual((sized((3, 5), 0, 0), sized((3, 5), None, -4)), (((3,), 0), ((3,), None)))
        self.assertEqual((sized((3,), None, 500), sized((), None, 2)), (((3,) * MOST, None), ((0, 0), None)))
        # A table of the file that is longer than the handle makes one is not cut by it.
        long = tuple(range(80))
        self.assertEqual([len(sized(long, None, count)[0]) for count in (90, 80, 70)], [80, 80, 70])

    def test_a_step_and_the_count_of_the_steps_in_plain_words(self) -> None:
        self.assertEqual(told(2, [15, 14, 12, 10, 8, 6, 4, 2, 1], "level"), "step 3 of 9: level 12, at 0.03 s")
        self.assertEqual(told(0, [-0.25], "semitones"), "step 1 of 1: semitones -0.25, at 0.00 s")
        self.assertEqual(told(60, [2] * 61), "step 61 of 61: value 2, at 1.00 s")
        self.assertEqual([counted(count) for count in (9, 1, 0, MOST)],
                         ["9 steps, 0.15 s", "1 step, 0.02 s", "0 steps, 0.00 s", "64 steps, 1.07 s"])

    def test_what_a_press_begins_depends_on_where_it_is_and_on_the_button(self) -> None:
        seen = Map(RECT, 0, 15)
        self.assertEqual([begins(seen, 8, None, (15, 40), button) for button in (1, 2, 3)], [DRAW, "", ZERO])
        # The outline of the field is the field's, so that the lowest and the highest value can be pressed.
        self.assertEqual([begins(seen, 8, None, pos) for pos in ((15, 30), (15, 78), (11, 40), (15, 25))],
                         [DRAW, DRAW, "", ""])
        self.assertEqual([begins(seen, 8, None, pos) for pos in ((92, 40), (94, 78), (95, 40), (120, 40))],
                         [SIZE, SIZE, "", ""])
        self.assertEqual(begins(seen, 8, None, (93, 40), 3), "")
        # Under the bars: anywhere on a step while there is no bracket, and else up to the step it begins at.
        self.assertEqual([begins(seen, 8, None, pos) for pos in ((15, 79), (85, 83), (93, 81))], [LOOP, LOOP, ""])
        self.assertEqual([begins(seen, 8, 3, (x, 81)) for x in (15, 35, 45, 55, 85)], [LOOP, LOOP, LOOP, "", ""])
        self.assertEqual((begins(seen, 8, 3, (45, 81), 3), begins(seen, 0, None, (15, 40))), ("", ""))


class Handled(unittest.TestCase):
    def setUp(self) -> None:
        self.ui = Ui()
        self.told: list[object] = []

    def table(self, table: object = VOL, low: float = 0, high: float = 15, rect: layout.Rect = RECT,
              **more: object) -> StepTable:
        """A table on the Ui that tells `told` of its changes."""
        return self.ui.add(StepTable(rect, table, low, high, changed=self.told.append, **more))

    def drag(self, *places: tuple[int, int], button: int = 1) -> bool:
        """Press at the first place, go by way of the others, and let go at the last. Whether the press was used."""
        used = self.ui.handle(Event("down", places[0], button=button))
        for place in places[1:]:
            self.ui.handle(Event("move", place))
        self.ui.handle(Event("up", places[-1], button=button))
        return used

    def test_a_press_sets_one_step_to_the_height_of_the_mouse_and_tells_on_the_release(self) -> None:
        shown = self.table()
        self.assertTrue(self.ui.handle(Event("down", (35, 62), button=1)))
        self.assertEqual((shown.steps, self.told), ((15, 14, 5, 10, 8, 6, 4, 2), []))
        self.ui.handle(Event("up", (35, 62), button=1))
        self.assertEqual(self.told, [[15, 14, 5, 10, 8, 6, 4, 2]])
        self.assertEqual((shown.table, shown.steps, shown.loop), (self.told[0], (15, 14, 5, 10, 8, 6, 4, 2), None))
        # A press that is not on a step, or with the middle button, is left for what lies below.
        self.assertEqual([self.drag((35, 25)), self.drag((120, 50)), self.drag((35, 62), button=2)], [False] * 3)
        self.assertEqual(len(self.told), 1)

    def test_a_drag_sets_each_step_it_passes_and_those_it_went_over_too_fast_to_be_on(self) -> None:
        shown = self.table()
        self.drag((15, 32), (55, 68), (65, -50), (75, 400))
        # Off the top a step is as high as the range goes, and under the foot it is 0.
        self.assertEqual(self.told, [[15, 11, 8, 5, 3, 15, 0, 2]])
        # Whole numbers when the table is one of those, and else the height to a few places.
        fine = self.table([0.5, 0.5], 0, 1, whole=False)
        self.ui.widgets.remove(shown)
        self.drag((15, 47), (25, 62))
        self.assertEqual((self.told[1], fine.steps), ([0.667, 0.333], (0.667, 0.333)))

    def test_the_right_button_sets_the_steps_it_passes_to_zero(self) -> None:
        self.table()
        self.assertTrue(self.drag((25, 40), (55, 70), button=3))
        self.assertEqual(self.told, [[15, 0, 0, 0, 0, 6, 4, 2]])
        self.ui.widgets.clear()
        sweep = self.table(PITCH, -12, 12, TALL)
        self.drag((45, 35), (75, 80), button=3)
        self.assertEqual((self.told[1], sweep.steps), ([0, -4, -4, 0, 0, 0, 0, -3], (0, -4, -4, 0, 0, 0, 0, -3)))

    def test_nothing_is_told_when_the_release_leaves_the_table_as_it_was(self) -> None:
        shown = self.table({"v": VOL, "loop": 2})
        # The height the step has, a drag that comes back to it, the handle and the loop not moved.
        self.assertTrue(self.drag((15, 32)))
        self.assertTrue(self.drag((35, 62), (35, 41)))
        self.assertTrue(self.drag((93, 50), (123, 50), (93, 50)))
        self.assertTrue(self.drag((35, 81), (65, 81), (35, 81)))
        self.assertEqual((self.told, shown.table), ([], {"v": VOL, "loop": 2}))
        self.ui.widgets.clear()
        self.table(PITCH, -12, 12, TALL)
        self.assertTrue(self.drag((15, 70), button=3))
        self.assertEqual(self.told, [])

    def test_the_handle_after_the_last_bar_adds_steps_and_takes_them_away(self) -> None:
        shown = self.table()
        self.assertEqual(shown.map.handle(8), (92, 30, 3, 49))
        self.drag((93, 50), (113, 50))
        # Each new step takes the value of the last one.
        self.assertEqual(self.told, [VOL + [2, 2]])
        self.drag((113, 50), (53, 50))
        self.assertEqual(self.told[1], [15, 14, 12, 10])
        # Never under one step, which the file has as the number alone. Never over the most.
        self.drag((53, 50), (-200, 50))
        self.assertEqual((self.told[2], shown.steps), (15, (15,)))
        self.drag((23, 50), (5000, 50))
        self.assertEqual((self.told[3], len(self.told)), ([15] * MOST, 4))
        # A list of no steps, which the file allows, has no last step: the new ones are 0.
        self.ui.widgets.clear()
        self.table([])
        self.drag((13, 50), (33, 50))
        self.assertEqual(self.told[4], [0, 0])

    def test_the_handle_takes_the_loop_away_with_the_step_it_began_at(self) -> None:
        shown = self.table({"v": DUTY, "loop": 2}, 0, 3)
        self.ui.handle(Event("down", (53, 50), button=1))
        self.ui.handle(Event("move", (33, 50)))
        self.assertEqual((shown.steps, shown.loop), ((0, 1), None))
        # Back again in the same drag, the steps and the loop are the ones there were.
        self.ui.handle(Event("move", (43, 50)))
        self.assertEqual((shown.steps, shown.loop), ((0, 1, 2), 2))
        self.ui.handle(Event("up", (33, 50), button=1))
        self.assertEqual(self.told, [[0, 1]])

    def test_the_start_of_the_loop_is_dragged_along_the_steps(self) -> None:
        shown = self.table({"v": DUTY, "loop": 1}, 0, 3)
        self.ui.handle(Event("down", (25, 81), button=1))
        self.ui.handle(Event("move", (45, 200)))
        self.assertEqual((shown.loop, self.told), (3, []))
        self.ui.handle(Event("move", (15, 81)))
        self.assertEqual(shown.loop, 0)
        # Further right than the last step it stays on the last.
        self.ui.handle(Event("move", (300, 81)))
        self.ui.handle(Event("up", (300, 81), button=1))
        self.assertEqual((self.told, shown.loop), ([{"v": DUTY, "loop": 3}], 3))

    def test_the_loop_dragged_off_the_left_end_is_taken_away(self) -> None:
        shown = self.table({"v": DUTY, "loop": 1}, 0, 3)
        self.drag((25, 81), (5, 81))
        self.assertEqual((self.told, shown.loop, shown.table), ([DUTY], None, DUTY))

    def test_a_click_under_the_bars_where_no_bracket_is_sets_a_loop_from_that_step(self) -> None:
        shown = self.table(DUTY, 0, 3)
        self.assertTrue(self.drag((35, 81)))
        self.assertEqual((self.told, shown.loop), ([{"v": DUTY, "loop": 2}], 2))
        # Left of the bracket there is none either. On the bracket, past its start, a press is not used.
        self.assertFalse(self.drag((45, 81)))
        self.assertTrue(self.drag((15, 81)))
        self.assertEqual(self.told[1:], [{"v": DUTY, "loop": 0}])
        # A table of one step can go round as well, and is then no number alone.
        self.ui.widgets.clear()
        self.table(2, 0, 3)
        self.drag((15, 81))
        self.assertEqual(self.told[2], {"v": [2], "loop": 0})

    def test_while_a_drag_is_under_way_the_table_is_what_its_release_gives(self) -> None:
        shown = self.table()
        self.ui.handle(Event("down", (93, 50), button=1))
        self.ui.handle(Event("move", (393, 50)))
        ahead = shown.table
        self.assertEqual((ahead, len(shown.steps), self.told), (VOL + [2] * 30, 38, []))
        # The bars stay as wide as they were at the press, so that the handle does not run from the mouse.
        self.assertEqual(shown.map.bar, 9)
        self.ui.handle(Event("up", (393, 50), button=1))
        self.assertEqual((self.told, shown.table, shown.map.bar), ([ahead], ahead, 2))
        # Drawing on the narrow bars: the first step, the one the mouse went over, and the one it is on.
        self.ui.handle(Event("down", (13, 32), button=1))
        self.ui.handle(Event("move", (19, 71)))
        ahead = shown.table
        self.ui.handle(Event("up", (19, 71), button=1))
        self.assertEqual((self.told[1], ahead[:4]), (ahead, [15, 8, 2, 10]))

    def test_a_table_that_is_not_enabled_takes_nothing_from_the_mouse(self) -> None:
        shown = self.table(help="the level at each tick", word="level")
        shown.enabled = False
        self.assertEqual([self.drag((35, 62)), self.drag((93, 50), (113, 50)), self.drag((35, 81))], [False] * 3)
        self.assertEqual((self.told, shown.table), ([], VOL))
        self.ui.handle(Event("move", (35, 50)))
        self.assertEqual(self.ui.help, "the level at each tick")

    def said(self, *places: tuple[int, int]) -> list[str]:
        """What the help line says with the mouse rested at each of some places."""
        said = []
        for place in places:
            self.ui.handle(Event("move", place))
            said.append(self.ui.help)
        return said

    def test_the_help_line_says_what_the_presses_do_on_the_step_under_the_mouse_which_it_names(self) -> None:
        shown = self.table(help="the level at each tick", word="level")
        # The left press, then the right one. Above the table there is no line, and right of the handle it is
        # the line the table was made with.
        self.assertEqual(self.said((35, 50), (85, 31), (35, 29), (120, 50)),
                         ["press: set step 3 of 8: level 12, at 0.03 s   right press: back to zero",
                          "press: set step 8 of 8: level 2, at 0.12 s   right press: back to zero",
                          "", "the level at each tick"])
        # While a drag is under way the value is the one its release would give.
        self.ui.handle(Event("down", (35, 62), button=1))
        self.ui.handle(Event("move", (45, 74)))
        self.assertEqual(self.ui.help, "press: set step 4 of 8: level 1, at 0.05 s   right press: back to zero")
        self.ui.handle(Event("up", (45, 74), button=1))
        # What the wheel does there comes last. The owner says it, since the wheel is left to the owner.
        shown.wheel_help = "along the tables"
        self.assertEqual(self.said((85, 50)), ["press: set step 8 of 8: level 2, at 0.12 s   right press: back to "
                                              "zero   wheel: along the tables"])

    def test_the_handle_and_the_loop_under_the_bars_each_have_a_line_of_their_own(self) -> None:
        shown = self.table(help="the level at each tick", wheel_help="along the tables")
        self.assertEqual(self.said((93, 50), (35, 81)),
                         [f"press: {SIZE_HELP}   wheel: along the tables",
                          "press: loop the table from step 3   wheel: along the tables"])
        self.assertEqual(SIZE_HELP, "drag right for more steps, left for fewer")
        # With a loop the press under the bars takes hold of its start, up to the step it begins at. On the
        # bracket further along a press does nothing, and the line is the one the table was made with.
        shown.table = {"v": VOL, "loop": 3}
        self.assertEqual(self.said((15, 81), (45, 81), (55, 81)),
                         [f"press: {LOOP_HELP}   wheel: along the tables"] * 2 + ["the level at each tick"])
        self.assertEqual(LOOP_HELP, "drag the start of the loop, off the left end to take the loop away")
        # While the handle is dragged the line stays that of the handle, wherever the mouse has got to.
        self.ui.handle(Event("down", (93, 50), button=1))
        self.assertEqual(self.said((60, 40)), [f"press: {SIZE_HELP}   wheel: along the tables"])

    def test_only_the_wheel_that_goes_sideways_is_used_and_only_by_a_table_whose_steps_do_not_fit(self) -> None:
        shown = self.table()
        for mods in (frozenset(), SHIFT, frozenset({"ctrl"})):
            self.assertFalse(self.ui.handle(Event("wheel", (35, 50), wheel=-1, mods=mods)))
        self.ui.widgets.clear()
        shown = self.table([1] * MOST)
        moved: list[int] = []
        shown.moved = moved.append
        # Up and down, and with ctrl, the wheel is left to what lies under the table: it changes no value.
        for mods in (frozenset(), frozenset({"ctrl"})):
            self.assertFalse(self.ui.handle(Event("wheel", (35, 50), wheel=-1, mods=mods)))
        self.assertTrue(self.ui.handle(Event("wheel", (35, 50), wheel=-1, mods=SHIFT)))
        self.assertEqual((shown.first, shown.map.first, shown.map.bar, shown.map.step_at(12), moved),
                         (36, 36, 2, 12, [36]))
        # A press is on the step that is drawn there.
        self.drag((12, 32))
        self.assertEqual((self.told, shown.table), ([[1] * 12 + [15] + [1] * 51], [1] * 12 + [15] + [1] * 51))
        self.ui.handle(Event("wheel", (35, 50), wheel=-9, mods=SHIFT))
        self.assertEqual(shown.map.first, 79)
        self.ui.handle(Event("wheel", (35, 50), wheel=99, mods=SHIFT))
        self.assertEqual((shown.map.first, moved), (0, [36, 79, 0]))

    def test_tables_under_one_ruler_have_room_for_as_many_steps_and_go_sideways_as_one(self) -> None:
        short, long = self.table(), self.table([1] * 100, rect=(10, 90, 120, 54))
        self.assertEqual((short.map.bar, long.map.bar), (9, 2))
        # Given the count of the longest, a step is as wide in the short one, and at the same place.
        short.span = long.span = 100
        self.assertEqual((short.map.bar, [short.map.x(step) for step in (0, 7)]),
                         (2, [long.map.x(step) for step in (0, 7)]))
        # Its owner has it go along with the one the wheel was on: it goes as far, though its own steps fit.
        long.moved = lambda first: setattr(short, "first", first)
        self.assertTrue(self.ui.handle(Event("wheel", (35, 110), wheel=-2, mods=SHIFT)))
        self.assertEqual((long.map.first, short.map.first, short.map.x(24), long.map.x(24)), (72, 72, 12, 12))

    def test_another_table_is_shown_when_it_is_set_and_one_that_was_not_changed_keeps_its_shape(self) -> None:
        shown = self.table([15])
        self.assertEqual((shown.table, shown.steps), ([15], (15,)))
        # A drag that is under way is dropped: the table is another one now.
        self.ui.handle(Event("down", (15, 62), button=1))
        self.assertEqual(shown.table, 5)
        shown.table = {"v": [3, 4], "loop": 9}
        self.ui.handle(Event("up", (15, 62), button=1))
        self.assertEqual((shown.table, shown.steps, shown.loop), ({"v": [3, 4], "loop": 9}, (3, 4), None))
        self.assertEqual(self.told, [])
        # What is handed out is a copy: the widget's own table is not changed through it.
        shown.table["v"].append(7)
        self.assertEqual(shown.table, {"v": [3, 4], "loop": 9})


@unittest.skipUnless(pygame, NEEDS)
class Drawn(unittest.TestCase):
    def setUp(self) -> None:
        self.surface = pygame.Surface(SIZE_PX)
        self.canvas = Canvas(self.surface, Font(sprites.load("font")))
        self.ui = Ui()

    def table(self, table: object = VOL, low: float = 0, high: float = 15, rect: layout.Rect = RECT,
              **more: object) -> StepTable:
        """A table on the Ui, in place of the one before."""
        self.ui.widgets.clear()
        return self.ui.add(StepTable(rect, table, low, high, **more))

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

    def written(self, text: str, rect: layout.Rect, align: str) -> set[tuple[int, int]]:
        """The pixels a line of dim text takes in a rectangle."""
        page = pygame.Surface(SIZE_PX)
        Canvas(page, self.canvas.font).text_in(text, rect, theme.DIM, align)
        return {pos for pos in cells(rect) if page.get_at(pos)[:3] == theme.DIM}

    def test_each_step_is_a_bar_as_high_as_its_value_in_the_colour(self) -> None:
        self.table()
        self.draw()
        bars = cells(*((12 + 10 * step, 77 - 3 * level, 9, 3 * level) for step, level in enumerate(VOL)))
        self.assertEqual(self.ink(RECT, theme.ACTIVE), bars)
        # The field is dark inside its outline, and a range from 0 has no line: the bars stand on the foot.
        self.assertEqual([self.at(*pos) for pos in ((10, 30), (129, 78), (11, 31), (21, 50), (30, 77))],
                         [theme.LINE, theme.LINE, theme.BACKGROUND, theme.BACKGROUND, theme.BACKGROUND])
        # A step of 0 is one pixel high, and the handle stands after the last bar.
        self.table([0, 15])
        self.draw()
        self.assertEqual(self.ink(RECT, theme.ACTIVE), cells((12, 76, 9, 1), (22, 32, 9, 45)))
        self.assertEqual(self.ink((12, 32, 116, 45), theme.TEXT), cells((32, 50, 3, 9)))
        # A list of no steps is the handle alone, with nothing after it.
        self.table([])
        self.draw()
        self.assertEqual(self.ink(RECT, theme.TEXT) | self.ink(RECT, theme.ACTIVE), cells((12, 50, 3, 9)))
        self.assertEqual(len(self.ink((11, 31, 118, 47), theme.BACKGROUND)), 118 * 47 - 3 * 9)

    def test_a_range_both_ways_has_a_zero_line_and_the_bars_under_it_hang_from_it(self) -> None:
        self.table(PITCH, -12, 12, TALL, colour=theme.SONG)
        self.draw()
        hung = [(12 + 10 * step, 57, 9, -2 * value) for step, value in enumerate(PITCH) if value]
        self.assertEqual(self.ink(TALL, theme.SONG), cells((12, 56, 9, 1), *hung))
        # The line runs from side to side: it shows between two bars and above the ones that hang, and the
        # handle stands across it.
        self.assertEqual(self.ink((12, 56, 116, 1), theme.LINE), cells((21, 56, 71, 1), (95, 56, 33, 1)))
        self.assertEqual(self.ink(TALL, theme.TEXT), cells((92, 52, 3, 9)))
        self.assertEqual([self.at(22, y) for y in (55, 56, 57, 64, 65)],
                         [theme.BACKGROUND, theme.LINE, theme.SONG, theme.SONG, theme.BACKGROUND])
        self.table([5, -12], -12, 12, TALL, colour=theme.SONG)
        self.draw()
        self.assertEqual(self.ink((12, 32, 20, 49), theme.SONG), cells((12, 46, 9, 10), (22, 57, 9, 24)))

    def test_the_loop_is_a_bracket_under_the_bars_from_the_step_it_goes_back_to(self) -> None:
        self.table({"v": DUTY, "loop": 1}, 0, 3, colour=theme.SOUND)
        self.draw()
        # A block under the step it begins at, a line along the foot, and a tick up at its end.
        self.assertEqual(self.ink((10, 79, 120, 5), theme.TEXT), cells((22, 80, 9, 4), (22, 83, 29, 1), (50, 80, 1, 4)))
        self.table({"v": DUTY, "loop": 3}, 0, 3)
        self.draw()
        self.assertEqual(self.ink((10, 79, 120, 5), theme.TEXT), cells((42, 80, 9, 4)))
        self.table(DUTY, 0, 3)
        self.draw()
        self.assertEqual(self.ink((10, 79, 120, 5), theme.TEXT), set())

    def test_after_the_last_bar_the_steps_that_repeat_are_drawn_again_fainter(self) -> None:
        faint = theme.shade(theme.SOUND, FAINT)
        tops = {1: 62, 2: 47}
        self.table({"v": DUTY, "loop": 1}, 0, 3, colour=theme.SOUND)
        self.draw()
        # After the handle, as far as there is room: the last one is cut where the bars end.
        again = [(56 + 10 * turn, tops[value], 9, 77 - tops[value]) for turn, value in enumerate((1, 2, 1, 1, 2, 1, 1))]
        self.assertEqual(self.ink(RECT, faint), cells(*again, (126, 47, 2, 30)))
        self.assertEqual(self.ink(RECT, theme.SOUND), cells((12, 76, 9, 1), (22, 62, 9, 15), (32, 47, 9, 30),
                                                             (42, 62, 9, 15)))
        # A table with no loop holds its last step.
        self.table(DUTY, 0, 3, colour=theme.SOUND)
        self.draw()
        self.assertEqual(self.ink(RECT, faint), cells(*((56 + 10 * turn, 62, 9, 15) for turn in range(7)),
                                                      (126, 62, 2, 15)))

    def test_no_word_is_drawn_and_a_step_the_handle_adds_is_drawn_while_it_is_dragged(self) -> None:
        wide = (10, 30, 122, 54)
        shown = self.table(rect=wide)
        self.draw()
        # The name and the count are written beside the table by its owner: nothing in it is in the dim colour.
        self.assertEqual(self.ink(wide, theme.DIM), set())
        # While the handle is dragged, the steps are the ones the release would give.
        self.ui.handle(Event("down", (93, 50), button=1))
        self.ui.handle(Event("move", (103, 50)))
        self.draw()
        self.assertEqual(len(self.ink(wide, theme.ACTIVE) - self.ink((12, 32, 80, 45), theme.ACTIVE)), 9 * 6)
        self.assertEqual((shown.steps[-1], counted(len(shown.steps))), (2, "9 steps, 0.15 s"))

    def test_a_table_that_is_not_enabled_is_dim(self) -> None:
        shown = self.table({"v": VOL, "loop": 2})
        shown.enabled = False
        self.draw()
        self.assertEqual((self.ink(RECT, theme.ACTIVE), self.ink(RECT, theme.TEXT)), (set(), set()))
        self.assertEqual(self.ink((12, 32, 80, 45), theme.DIM),
                         cells(*((12 + 10 * step, 77 - 3 * level, 9, 3 * level) for step, level in enumerate(VOL))))
        self.assertEqual((self.at(93, 54), self.at(33, 81)), (theme.LINE, theme.LINE))

    def test_nothing_is_drawn_outside_the_rectangle_however_many_steps_there_are(self) -> None:
        around = cells((0, 0, *SIZE_PX)) - cells(RECT)

        def clean() -> bool:
            self.draw()
            return all(self.at(*pos) == OUTSIDE for pos in around)

        shown = self.table({"v": [15] * MOST, "loop": 0})
        self.assertTrue(clean())
        self.ui.handle(Event("wheel", (35, 50), wheel=-9, mods=SHIFT))
        self.assertTrue(clean())
        # Scrolled to its end, the last bar and the handle are at the right edge of the bars.
        self.assertEqual((shown.map.x(MOST - 1), shown.map.handle(MOST)), (122, (125, 30, 3, 49)))
        self.assertEqual([self.at(*pos) for pos in ((12, 40), (13, 40), (123, 40), (124, 40), (126, 54))],
                         [theme.ACTIVE, theme.BACKGROUND, theme.ACTIVE, theme.BACKGROUND, theme.TEXT])
        # Steps that the handle adds past the edge wait there until the release.
        self.table()
        self.ui.handle(Event("down", (93, 50), button=1))
        self.ui.handle(Event("move", (700, -90)))
        self.assertTrue(clean())
        for small in ((10, 30, 30, 12), (10, 30, 3, 40), (10, 30, 0, 0)):
            self.table({"v": PITCH, "loop": 1}, -12, 12, small)
            self.assertTrue(clean())

    def test_the_tables_of_every_instrument_of_the_examples_are_read_drawn_and_written_back(self) -> None:
        seen, lists_of_one = 0, 0
        for name in ("arcade.json", "piece.json"):
            for inst in json.loads((EXAMPLES / name).read_text(encoding="utf-8"))["instruments"].values():
                for key, (low, high, fallback) in RANGES.items():
                    table = inst.get(key, fallback)
                    steps, loop = read(table)
                    shown = self.table(table, low, high)
                    self.draw()
                    self.assertEqual(self.at(11, 31), theme.BACKGROUND)
                    # The widget hands back what the file has, and the engine reads the steps as it reads the file.
                    self.assertEqual(shown.table, table)
                    self.assertEqual([at(steps, loop, tick) for tick in range(len(steps) + 9)],
                                     [note.step(table, tick, fallback) for tick in range(len(steps) + 9)])
                    # Written anew it is the same, but for a list of one step: that is the number alone.
                    one = isinstance(table, list) and len(table) == 1
                    self.assertEqual(write(steps, loop), table[0] if one else table)
                    seen, lists_of_one = seen + 1, lists_of_one + one
        self.assertEqual((seen, lists_of_one), (72, 7))


if __name__ == "__main__":
    unittest.main()
