"""Tests for rowsynth.gui.dials: the sums of a dial, and the knob, the slider, the meter and the stepper."""
from __future__ import annotations

import os
import unittest

from rowsynth.errors import RowSynthError
from rowsynth.gui import layout, sprites, theme
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.dials import (Knob, Meter, Slider, Stepper, around, bars, core, disc, dragged, handle_at, kept,
                                knob_parts, lit_marks, meter_height, plain, stretch, value_at, wheeled, zones)
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

NEEDS = "pygame is not installed"
SHIFT = frozenset({"shift"})
#: A knob 13 across: its rim, its face, and what is not of it.
ROUND = ("....#####....", "..##ooooo##..", ".#ooooooooo#.", ".#ooooooooo#.", "#ooooooooooo#", "#ooooooooooo#",
         "#ooooooooooo#", "#ooooooooooo#", "#ooooooooooo#", ".#ooooooooo#.", ".#ooooooooo#.", "..##ooooo##..",
         "....#####....")


def level(db: float) -> float:
    """The level, from 0 to 1, that is so many decibels under the top."""
    return 10 ** (db / 20)


def click(ui: Ui, pos: tuple[int, int], button: int = 1) -> bool:
    """Press a mouse button at a position and let it go there. Whether the press was used."""
    used = ui.handle(Event("down", pos, button=button))
    ui.handle(Event("up", pos, button=button))
    return used


class Sums(unittest.TestCase):
    def test_a_drag_of_the_whole_travel_is_the_whole_range_and_the_value_stays_inside_it(self) -> None:
        self.assertEqual((dragged(0, 100, 0, 1, 100), dragged(-12, 25, -12, 12, 50)), (1, 0))
        self.assertAlmostEqual(dragged(0.5, 20, 0, 1, 100), 0.7)
        self.assertAlmostEqual(dragged(0.5, -20, 0, 1, 100), 0.3)
        self.assertEqual((dragged(0.5, 500, 0, 1, 100), dragged(0.5, -500, 0, 1, 100)), (1, 0))
        # The travel is the theme's when none is given.
        self.assertEqual(dragged(0, theme.DRAG_TRAVEL / 2, 0, 8), 4)

    def test_a_fine_drag_goes_at_a_tenth_of_the_speed(self) -> None:
        self.assertAlmostEqual(dragged(0.5, 20, 0, 1, 100, fine=True), 0.52)
        self.assertAlmostEqual(dragged(0.5, -150, 0, 1, 100, fine=True), 0.35)
        self.assertEqual(dragged(0.5, 9000, 0, 1, 100, fine=True), 1)

    def test_a_dial_with_steps_is_put_on_the_nearest_whole_step(self) -> None:
        self.assertEqual([kept(value, 0, 10, 10) for value in (3.4, 3.5, -2, 99)], [3, 4, 0, 10])
        self.assertEqual((kept(0.26, 0, 1, 4), kept(0.26, 0, 1), kept(-7.4, -12, 12, 24)), (0.25, 0.26, -7))
        self.assertEqual((dragged(0, 34, 0, 10, 100, steps=10), dragged(0, 36, 0, 10, 100, steps=10)), (3, 4))
        self.assertEqual(dragged(2, 30, 0, 10, 100, fine=True, steps=10), 2)

    def test_the_two_sums_of_a_slider_are_each_others_opposite(self) -> None:
        for low, high, run in ((0, 1, 50), (-1, 1, 104), (20, 20000, 7)):
            there = [handle_at(value_at(place, low, high, run), low, high, run) for place in range(run + 1)]
            self.assertEqual(there, list(range(run + 1)))
        self.assertEqual((value_at(25, 0, 1, 50), value_at(0, -1, 1, 104), value_at(104, -1, 1, 104)), (0.5, -1, 1))
        # A place off the track is the end it is nearer to, and so is a value out of the range.
        self.assertEqual((value_at(-9, 0, 1, 50), value_at(99, 0, 1, 50)), (0, 1))
        self.assertEqual((handle_at(-3, 0, 1, 50), handle_at(7, 0, 1, 50)), (0, 50))

    def test_a_meter_is_even_in_decibels_from_48_under_at_its_foot_to_0_at_its_top(self) -> None:
        self.assertEqual((meter_height(0), meter_height(1)), (0, 1))
        for db, height in ((-48, 0), (-6, 0.875), (-24, 0.5), (-12, 0.75)):
            self.assertAlmostEqual(meter_height(level(db)), height)
        self.assertEqual((meter_height(level(-60)), meter_height(1.5), meter_height(-0.2)), (0, 1, 0))
        # A bar is green up to 12 under, mustard up to 3 under, and red above. Two bars share the inside.
        self.assertEqual(zones(96), [(theme.GOOD, 0, 72), (theme.MUSTARD, 72, 90), (theme.FAULT, 90, 96)])
        self.assertEqual((bars((10, 20, 6, 50), 1), bars((10, 20, 11, 50), 2)),
                         ([(11, 21, 4, 48)], [(11, 21, 4, 48), (16, 21, 4, 48)]))

    def test_the_bars_of_a_meter_that_lies_are_one_above_the_other_and_are_counted_from_the_left(self) -> None:
        self.assertEqual((bars((10, 20, 50, 6), 1, lying=True), bars((10, 20, 50, 11), 2, lying=True)),
                         ([(11, 21, 48, 4)], [(11, 21, 48, 4), (11, 26, 48, 4)]))
        # The first twelve pixels of a bar and its last twelve: up from the foot of one that stands.
        self.assertEqual((stretch((11, 21, 4, 48), 0, 12), stretch((11, 21, 4, 48), 36, 48)),
                         ((11, 57, 4, 12), (11, 21, 4, 12)))
        self.assertEqual((stretch((11, 21, 48, 4), 0, 12, lying=True), stretch((11, 21, 48, 4), 36, 48, lying=True)),
                         ((11, 21, 12, 4), (47, 21, 12, 4)))

    def test_the_marks_of_a_knob_that_its_value_has_reached_are_lit(self) -> None:
        self.assertEqual([lit_marks(value, 0, 1) for value in (0, 0.09, 0.1, 0.3, 0.5, 0.99, 1)],
                         [1, 1, 2, 4, 6, 10, 11])
        self.assertEqual([lit_marks(value, -12, 12, 5) for value in (-40, -12, -6.1, -6, 0, 12, 40)],
                         [1, 1, 1, 2, 3, 5, 5])

    def test_a_knob_is_round_and_turns_from_seven_o_clock_over_the_top_to_five(self) -> None:
        drawn = [["."] * 13 for _ in range(13)]
        for mark, rows in (("#", disc(13)), ("o", core(13))):
            for row, (first, width) in enumerate(rows):
                drawn[row][first:first + width] = mark * width
        self.assertEqual(tuple("".join(row) for row in drawn), ROUND)
        self.assertEqual([around(turn, 4) for turn in (0, 0.5, 1)], [(-3, 3), (0, -4), (3, 3)])
        self.assertEqual([around(turn, 8) for turn in (0.1, 0.4, 0.9)], [(-8, 2), (-4, -7), (8, 2)])
        # The knob is in the middle of its rectangle, between the line of its label and the line of its value.
        self.assertEqual(knob_parts((10, 5, 50, 37)), ((10, 5, 50, 11), (35, 23), (10, 31, 50, 11)))

    def test_a_number_is_shown_with_no_more_places_than_it_needs(self) -> None:
        self.assertEqual([plain(value) for value in (3.0, 0.35, 0.1 + 0.2, 4000, -0.0001, -2.5, 1 / 3)],
                         ["3", "0.35", "0.3", "4000", "0", "-2.5", "0.333"])


class Handled(unittest.TestCase):
    def setUp(self) -> None:
        self.ui = Ui()
        self.told: list[float] = []

    def knob(self, value: float = 0.5, **more: object) -> Knob:
        """A knob from 0 to 1 whose middle is at (25, 218), on the Ui, that tells `told` of its changes."""
        return self.ui.add(Knob((0, 200, 50, 37), value, 0, 1, changed=self.told.append, **more))

    def test_a_knob_dragged_up_rises_and_dragged_down_falls(self) -> None:
        knob = self.knob()
        self.assertTrue(self.ui.handle(Event("down", (25, 218), button=1)))
        self.ui.handle(Event("move", (25, 208)))
        self.assertAlmostEqual(knob.value, 0.6)
        # The drag goes on with the mouse off the knob, and how far it is to the side counts for nothing.
        self.ui.handle(Event("move", (90, 248)))
        self.assertAlmostEqual(knob.value, 0.2)
        self.ui.handle(Event("up", (90, 248), button=1))
        self.ui.handle(Event("move", (25, 210)))
        self.assertAlmostEqual(knob.value, 0.2)

    def test_a_knob_is_held_at_the_top_of_its_range_and_at_its_foot(self) -> None:
        knob = self.knob()
        self.ui.handle(Event("down", (25, 218), button=1))
        for y in (138, 128):
            self.ui.handle(Event("move", (25, y)))
            self.assertEqual((knob.value, self.told), (1, [1]))
        # The mouse has to come back the way it went before the value follows it down.
        self.ui.handle(Event("move", (25, 178)))
        self.assertAlmostEqual(knob.value, 0.9)
        self.ui.handle(Event("move", (25, 618)))
        self.assertEqual(knob.value, 0)

    def test_with_shift_held_a_drag_is_fine_and_the_value_does_not_jump_when_shift_goes_down_or_up(self) -> None:
        knob = self.knob()
        self.ui.handle(Event("down", (25, 218), button=1, mods=SHIFT))
        self.ui.handle(Event("move", (25, 198), mods=SHIFT))
        self.assertAlmostEqual(knob.value, 0.52)
        self.ui.handle(Event("move", (25, 178)))
        self.assertAlmostEqual(knob.value, 0.72)
        self.ui.handle(Event("move", (25, 168), mods=SHIFT))
        self.assertAlmostEqual(knob.value, 0.73)

    def test_the_wheel_moves_a_dial_a_step_or_a_hundredth_of_its_range_when_it_has_no_steps(self) -> None:
        knob = self.knob()
        self.assertTrue(self.ui.handle(Event("wheel", (25, 218), wheel=1)))
        self.assertAlmostEqual(knob.value, 0.51)
        self.ui.handle(Event("wheel", (25, 218), wheel=-3))
        self.assertAlmostEqual(knob.value, 0.48)
        octave = self.ui.add(Knob((60, 200, 50, 37), 2, 0, 8, steps=8, changed=self.told.append))
        self.ui.handle(Event("wheel", (85, 218), wheel=1))
        self.assertEqual(octave.value, 3)
        for _ in range(2):
            self.ui.handle(Event("wheel", (85, 218), wheel=-9))
        self.assertEqual((octave.value, self.told[2:], wheeled(8, 1, 0, 8, 8)), (0, [3, 0], 8))
        self.assertFalse(self.ui.handle(Event("wheel", (150, 218), wheel=1)))

    def test_a_right_click_puts_a_dial_back_to_its_default_when_it_has_one(self) -> None:
        knob = self.knob(0.9, default=0.25)
        self.assertTrue(click(self.ui, (25, 218), button=3))
        self.assertEqual((knob.value, self.told), (0.25, [0.25]))
        # A second one changes nothing and tells nothing. One with no default leaves the click to what lies below.
        click(self.ui, (25, 218), button=3)
        self.assertEqual(self.told, [0.25])
        bare = self.ui.add(Slider((60, 200, 54, 8), 0.9, 0, 1, changed=self.told.append))
        self.assertFalse(click(self.ui, (70, 204), button=3))
        self.assertEqual((bare.value, self.told), (0.9, [0.25]))

    def test_changed_is_called_once_for_each_change_and_not_at_all_when_nothing_changed(self) -> None:
        self.knob(steps=10)
        self.ui.handle(Event("down", (25, 218), button=1))
        self.assertEqual(self.told, [])
        for y in (216, 214, 212, 208, 206, 198, 198):
            self.ui.handle(Event("move", (25, y)))
        self.ui.handle(Event("up", (25, 198), button=1))
        self.assertEqual(self.told, [0.6, 0.7])

    def test_a_press_on_the_track_of_a_lying_slider_puts_the_handle_there_and_the_drag_goes_on_from_it(self) -> None:
        slider = self.ui.add(Slider((10, 10, 54, 8), 0, 0, 1, changed=self.told.append))
        self.assertEqual((slider.upright, slider.handle()), (False, (10, 10, 4, 8)))
        self.ui.handle(Event("down", (37, 14), button=1))
        self.assertEqual((slider.value, slider.handle(), self.told), (0.5, (35, 10, 4, 8), [0.5]))
        self.ui.handle(Event("move", (47, 30)))
        self.assertAlmostEqual(slider.value, 0.7)
        self.assertEqual(slider.handle(), (45, 10, 4, 8))
        # Past the end of the track the handle waits at the end, and it is under the mouse again on the way back.
        self.ui.handle(Event("move", (200, 14)))
        self.assertEqual((slider.value, slider.handle()), (1, (60, 10, 4, 8)))
        self.ui.handle(Event("move", (12, 14)))
        self.assertEqual((slider.value, slider.handle()), (0, (10, 10, 4, 8)))

    def test_an_upright_slider_has_its_low_end_at_its_foot(self) -> None:
        slider = self.ui.add(Slider((10, 10, 12, 56), 0, 0, 1, changed=self.told.append))
        self.assertEqual((slider.upright, slider.handle()), (True, (10, 60, 12, 6)))
        self.ui.handle(Event("down", (15, 37), button=1))
        self.assertEqual((slider.value, slider.handle()), (0.5, (10, 35, 12, 6)))
        self.ui.handle(Event("move", (15, 27)))
        self.assertAlmostEqual(slider.value, 0.7)
        self.assertEqual(slider.handle(), (10, 25, 12, 6))

    def test_a_stepper_goes_one_back_on_its_left_half_and_one_on_on_its_right_and_stops_at_the_ends(self) -> None:
        stepper = self.ui.add(Stepper((10, 10, 40, 13), (100, 112, 128), changed=self.told.append))
        click(self.ui, (29, 15))
        self.assertEqual((stepper.picked, self.told), (0, []))
        for x in (30, 49, 49, 29):
            click(self.ui, (x, 15))
        self.assertEqual((stepper.picked, self.told), (1, [1, 2, 1]))
        for wheel in (-1, -1, 5):
            self.assertTrue(self.ui.handle(Event("wheel", (20, 15), wheel=wheel)))
        self.assertEqual((stepper.picked, self.told), (2, [1, 2, 1, 0, 2]))
        # Let go off it, and pressed with the right button: neither steps.
        self.ui.handle(Event("down", (12, 15), button=1))
        self.ui.handle(Event("up", (12, 40), button=1))
        self.assertFalse(click(self.ui, (12, 15), button=3))
        self.assertEqual(stepper.picked, 2)

    def test_what_is_not_enabled_takes_nothing_from_the_mouse(self) -> None:
        knob = self.knob(default=0.1)
        slider = self.ui.add(Slider((60, 200, 54, 8), 0.5, 0, 1, changed=self.told.append, default=0.1))
        stepper = self.ui.add(Stepper((60, 220, 40, 13), (2, 3, 4), 1, changed=self.told.append))
        for widget, pos in ((knob, (25, 218)), (slider, (70, 204)), (stepper, (95, 226))):
            widget.enabled = False
            self.assertFalse(self.ui.handle(Event("down", pos, button=1)))
            self.assertFalse(self.ui.handle(Event("move", (pos[0] + 9, pos[1] - 9))))
            self.assertFalse(self.ui.handle(Event("up", pos, button=1)))
            self.assertFalse(click(self.ui, pos, button=3))
            self.assertFalse(self.ui.handle(Event("wheel", pos, wheel=1)))
        self.assertEqual((knob.value, slider.value, stepper.picked, self.told), (0.5, 0.5, 1, []))

    def test_the_line_kept_at_the_top_of_a_bar_falls_slowly_and_never_under_the_bar(self) -> None:
        meter = self.ui.add(Meter((0, 0, 6, 98)))
        self.assertEqual((meter.levels, meter.tops), ((0.0,), [0.0]))
        meter.levels = (1.0,)
        meter.levels = (level(-24),)
        self.assertEqual(meter.tops, [1])
        # It falls 12 decibels a second, a quarter of the meter.
        meter.fall(1)
        self.assertAlmostEqual(meter.tops[0], 0.75)
        for seconds in (1.5, 60):
            meter.fall(seconds)
            self.assertAlmostEqual(meter.tops[0], 0.5)
        # A bar that rises takes its line up with it, and each of two bars has its own. A meter takes no mouse.
        meter.levels = (1.0, 0.0)
        self.assertEqual(meter.tops, [1, 0])
        self.assertFalse(click(self.ui, (2, 50)))
        with self.assertRaises(RowSynthError) as refused:
            meter.levels = (0.1, 0.2, 0.3)
        self.assertEqual(str(refused.exception), "a meter has one bar or two, not 3")

    def test_a_dial_wants_a_range_that_rises_and_whole_steps_and_a_stepper_an_option_it_has(self) -> None:
        rect = (0, 0, 50, 37)
        for make, wrong in (
                (lambda: Knob(rect, 1, 2, 2), "a dial goes from a low value to a higher one, not from 2 to 2"),
                (lambda: Slider(rect, 0, 0, 1, steps=0), "the steps of a dial are a whole number of 1 or more, not 0"),
                (lambda: Stepper(rect, ()), "a stepper needs at least one option"),
                (lambda: Stepper(rect, (1, 2), 2), "a stepper of 2 options has no option 2")):
            with self.assertRaises(RowSynthError) as refused:
                make()
            self.assertEqual(str(refused.exception), wrong)


@unittest.skipUnless(pygame, NEEDS)
class Drawn(unittest.TestCase):
    def setUp(self) -> None:
        self.surface = pygame.Surface((120, 104))
        self.canvas = Canvas(self.surface, Font(sprites.load("font")))
        self.ui = Ui()

    def draw(self) -> None:
        """Draw the widgets on a clean panel."""
        self.surface.fill(theme.PANEL)
        self.ui.draw(self.canvas)

    def at(self, *pos: int) -> tuple[int, ...]:
        """The colour of one pixel."""
        return tuple(self.surface.get_at(pos))[:3]

    def ink(self, rect: layout.Rect, colour: theme.Colour) -> set[tuple[int, int]]:
        """The pixels of a rectangle that are of a colour."""
        x, y, w, h = rect
        return {(x + column, y + row) for column in range(w) for row in range(h)
                if self.at(x + column, y + row) == colour}

    def written(self, text: str, rect: layout.Rect, colour: theme.Colour) -> set[tuple[int, int]]:
        """The pixels a line of text takes when it is centred in a rectangle."""
        page = pygame.Surface(self.surface.get_size())
        Canvas(page, self.canvas.font).text_in(text, rect, colour, "centre")
        x, y, w, h = rect
        return {(x + column, y + row) for column in range(w) for row in range(h)
                if page.get_at((x + column, y + row))[:3] == colour}

    def test_the_travel_of_a_knob_is_lit_in_its_colour_up_to_its_value(self) -> None:
        knob = self.ui.add(Knob((10, 5, 50, 37), 0, 0, 1, colour=theme.SOUND, text="tone"))
        lit = []
        for value in (0, 0.5, 1):
            knob.value = value
            self.draw()
            lit.append(self.ink(knob.rect, theme.SOUND))
        self.assertEqual([len(marks) for marks in lit], [1, 6, 11])
        self.assertTrue(lit[0] < lit[1] < lit[2])
        self.assertEqual((lit[0], (35, 15) in lit[1], (41, 29) in lit[2]), ({(29, 29)}, True, True))
        # At the half the pointer is straight up on the face, inside the rim. A mark not reached is a faint one.
        knob.value = 0.5
        self.draw()
        self.assertEqual([self.at(35, y) for y in (23, 21, 20, 19, 18, 17, 16)],
                         [theme.LIT, theme.TEXT, theme.TEXT, theme.TEXT, theme.LIT, theme.LINE, theme.PANEL])
        self.assertEqual(self.at(41, 29), theme.LINE)
        # The label is above in capitals, and the value below with no more places than it needs.
        label, _, under = knob_parts(knob.rect)
        self.assertEqual(self.ink(label, theme.DIM), self.written("TONE", label, theme.DIM))
        self.assertEqual(self.ink(under, theme.TEXT), self.written("0.5", under, theme.TEXT))
        knob.show = lambda value: f"{value * 100:.0f} HZ"
        knob.enabled = False
        self.draw()
        self.assertEqual((self.ink(knob.rect, theme.SOUND), self.ink(knob.rect, theme.TEXT)), (set(), set()))
        self.assertEqual(self.ink(under, theme.DIM), self.written("50 HZ", under, theme.DIM))

    def test_the_handle_of_a_slider_is_drawn_where_the_sum_says(self) -> None:
        lying = self.ui.add(Slider((10, 10, 54, 8), 0.5, 0, 1, colour=theme.SONG))
        upright = self.ui.add(Slider((80, 2, 12, 56), 0.25, 0, 1, colour=theme.SALMON))
        self.draw()
        self.assertEqual((lying.handle(), upright.handle()),
                         ((10 + handle_at(0.5, 0, 1, 50), 10, 4, 8), (80, 52 - handle_at(0.25, 0, 1, 50), 12, 6)))
        for slider in (lying, upright):
            # The handle is in the colour of the slider, with a dark outline.
            x, y, w, h = slider.handle()
            inside = {(x + column, y + row) for column in range(1, w - 1) for row in range(1, h - 1)}
            self.assertEqual(self.ink(slider.rect, slider.colour), inside)
            self.assertEqual(len(self.ink(slider.handle(), theme.BACKGROUND)), w * h - len(inside))
        # The track of the lying one is lit, darker than the handle, from its low end to the handle.
        dark = theme.shade(theme.SONG, theme.TRAVEL)
        self.assertEqual(self.ink(lying.rect, dark), {(x, y) for x in range(10, 35) for y in (13, 14)})
        self.assertEqual([self.at(x, 13) for x in (39, 63)], [theme.BACKGROUND] * 2)
        self.assertEqual([self.at(x, y) for x in (85, 86) for y in (2, 30, 57)], [theme.BACKGROUND] * 6)
        lying.enabled = upright.enabled = False
        self.draw()
        self.assertEqual(self.ink(lying.rect, theme.SONG) | self.ink(upright.rect, theme.SALMON), set())
        self.assertEqual(len(self.ink(lying.rect, theme.DIM)), 2 * 6)

    def test_a_bar_is_green_then_mustard_then_red_at_the_very_top_and_a_limit_is_a_line_across(self) -> None:
        meter = self.ui.add(Meter((10, 2, 11, 98), limits=(level(-24),)))
        meter.levels = (1.0, level(-24))
        self.draw()
        # Inside the outline there are 96 rows, from y 98 up to y 3: the first bar is lit to the top.
        self.assertEqual([self.at(12, 98 - row) for row in (0, 71, 72, 89, 90, 95)],
                         [theme.GOOD, theme.GOOD, theme.MUSTARD, theme.MUSTARD, theme.FAULT, theme.FAULT])
        self.assertEqual((self.at(10, 50), self.at(15, 50), self.at(12, 2)), (theme.LINE, theme.BACKGROUND, theme.LINE))
        # The second is lit to the half, where the limit is drawn across both.
        self.assertEqual(self.ink((10, 51, 11, 1), theme.TEXT), {(x, 51) for x in range(11, 20)})
        self.assertEqual([self.at(17, y) for y in (98, 52, 50, 3)],
                         [theme.GOOD, theme.GOOD, theme.BACKGROUND, theme.BACKGROUND])
        # When the bars drop, a thin line stays where each was and falls: here a quarter of the meter in a second.
        meter.levels = (level(-24), 0.0)
        meter.fall(1)
        self.draw()
        self.assertEqual([self.at(12, y) for y in (26, 27, 28, 50, 52)],
                         [theme.BACKGROUND, theme.GOOD, theme.BACKGROUND, theme.BACKGROUND, theme.GOOD])
        self.assertEqual([self.at(17, y) for y in (74, 75, 76, 98)],
                         [theme.BACKGROUND, theme.GOOD, theme.BACKGROUND, theme.BACKGROUND])

    def lit(self, meter: Meter) -> set[tuple[int, int]]:
        """The pixels of a meter that are in one of the colours of a bar."""
        return set().union(*(self.ink(meter.rect, colour) for colour in (theme.GOOD, theme.MUSTARD, theme.FAULT)))

    def test_a_bar_that_lies_is_as_long_as_one_that_stands_is_high(self) -> None:
        standing = self.ui.add(Meter((2, 2, 6, 98)))
        lying = self.ui.add(Meter((12, 2, 98, 6)))
        self.assertEqual((standing.upright, lying.upright), (True, False))
        for db, long in ((-48, 0), (-12, 72), (0, 96)):
            standing.levels = lying.levels = (level(db),)
            self.draw()
            # The one that stands is lit up from its foot, at y 98, and the one that lies from its left, at x 13.
            self.assertEqual(self.lit(standing), {(x, 98 - row) for x in range(3, 7) for row in range(long)})
            self.assertEqual(self.lit(lying), {(13 + column, y) for column in range(long) for y in range(3, 7)})

    def test_a_bar_that_lies_is_green_then_mustard_then_red_at_its_right_end(self) -> None:
        meter = self.ui.add(Meter((10, 20, 98, 11)))
        meter.levels = (1.0, level(-24))
        self.draw()
        # Inside the outline there are 96 columns, from x 11: the upper bar is the first, and is lit to the end.
        self.assertEqual([self.at(11 + column, 22) for column in (0, 71, 72, 89, 90, 95)],
                         [theme.GOOD, theme.GOOD, theme.MUSTARD, theme.MUSTARD, theme.FAULT, theme.FAULT])
        self.assertEqual((self.at(50, 20), self.at(50, 25), self.at(107, 22)),
                         (theme.LINE, theme.BACKGROUND, theme.LINE))
        # The lower one is lit to the half.
        self.assertEqual([self.at(x, 27) for x in (11, 58, 59, 106)],
                         [theme.GOOD, theme.GOOD, theme.BACKGROUND, theme.BACKGROUND])

    def test_the_line_kept_at_the_end_of_a_bar_that_lies_is_upright_and_falls_to_the_left(self) -> None:
        meter = self.ui.add(Meter((10, 20, 98, 11)))
        meter.levels = (1.0, level(-24))
        meter.levels = (level(-24), 0.0)
        meter.fall(1)
        self.draw()
        # A quarter of the meter in a second: the line of the upper bar is at 72 of 96, that of the lower at 24.
        self.assertEqual(self.lit(meter) - {(x, y) for x in range(11, 59) for y in range(21, 25)},
                         {(82, y) for y in range(21, 25)} | {(34, y) for y in range(26, 30)})
        # It stops at the end of its bar, and with no bar it falls to nothing.
        meter.fall(60)
        self.draw()
        self.assertEqual(self.lit(meter), {(x, y) for x in range(11, 59) for y in range(21, 25)})

    def test_a_limit_of_a_meter_that_lies_is_an_upright_line_through_its_bars(self) -> None:
        meter = self.ui.add(Meter((10, 20, 98, 11), limits=(level(-24), 0.0)))
        meter.levels = (0.0, 0.0)
        self.draw()
        # At the half, and one at the floor is the first column inside the outline.
        self.assertEqual(self.ink(meter.rect, theme.TEXT), {(x, y) for x in (58, 11) for y in range(21, 30)})

    def test_a_stepper_shows_its_option_between_two_arrows_and_the_arrow_at_an_end_is_dim(self) -> None:
        stepper = self.ui.add(Stepper((10, 10, 40, 13), (100, 128, 150), 1))
        left = {(14, 16), (15, 15), (15, 16), (15, 17), (16, 14), (16, 15), (16, 16), (16, 17), (16, 18)}
        right = {(59 - x, y) for x, y in left}
        between = (20, 10, 20, 13)
        for picked, shown, dim in ((1, "128", set()), (0, "100", left), (2, "150", right)):
            stepper.picked = picked
            self.draw()
            self.assertEqual(self.ink(stepper.rect, theme.TEXT),
                             ((left | right) - dim) | self.written(shown, between, theme.TEXT))
            self.assertEqual(self.ink(stepper.rect, theme.DIM), dim)
        self.assertEqual((self.at(10, 10), self.at(12, 12)), (theme.LINE, theme.BACKGROUND))
        stepper.enabled = False
        self.draw()
        self.assertEqual(self.ink(stepper.rect, theme.TEXT), set())
        self.assertEqual(self.ink(stepper.rect, theme.DIM), left | right | self.written("150", between, theme.DIM))


if __name__ == "__main__":
    unittest.main()
