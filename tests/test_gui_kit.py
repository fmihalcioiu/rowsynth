"""Tests for rowsynth.gui.kit: the wheel, the heading, the settings bar, the keys, a scroll bar, the lane, the menu."""
from __future__ import annotations

import os
import unittest

from rowsynth.gui import layout, sprites, theme
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.dials import Knob, Stepper
from rowsynth.gui.font import Font
from rowsynth.gui.kit import (ACROSS, ALONG, HIGHEST, LOWEST, VELOCITY_HELP, WHEEL_KEYS, WHEEL_PX, ZOOM, Bar, Heading,
                              Keys, Plus, ScrollBar, Slot, Velocity, bar_cells, heading_parts, key_rect, moment_at,
                              name_room, passed, pitch_at, right_cells, thing_menu, told, velocity_at, velocity_bar,
                              wheel, wide, x_at, zoomed)
from rowsynth.gui.popups import Item
from rowsynth.gui.ui import Button, Choice, Event, Label, Toggle, Ui

# pygame reads these when it is imported: no test opens a window or plays a sound.
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
try:
    import pygame
except ImportError:
    pygame = None

NEEDS = "pygame is not installed"
#: A heading and a settings bar as wide as the middle of the window, a keyboard, and a lane with a note at each of
#: three places: the second is picked, and the third lies under the second for half its width.
HEAD = (128, 26, 654, theme.HEAD_H)
BAR = (128, 121, 654, theme.BAR_H)
KEYS = (10, 30, 40, 110)
LANE = (10, 150, 300, 34)
ITEMS = ((60, 36, 0.5, False), (100, 18, 1.0, True), (109, 18, 0.2, False))


def click(ui: Ui, pos: tuple[int, int], button: int = 1) -> bool:
    """Press a mouse button at a position and let it go there. Whether the press was used."""
    used = ui.handle(Event("down", pos, button=button))
    ui.handle(Event("up", pos, button=button))
    return used


class Wheel(unittest.TestCase):
    def test_a_step_of_the_wheel_zooms_with_ctrl_goes_across_with_shift_and_along_alone(self) -> None:
        def turned(steps: int, *mods: str) -> tuple[str, int]:
            return wheel(Event("wheel", (5, 5), wheel=steps, mods=frozenset(mods)))

        self.assertEqual([turned(1), turned(-3, "shift"), turned(2, "ctrl"), turned(-1, "ctrl", "shift"),
                          turned(1, "alt")], [(ALONG, 1), (ACROSS, -3), (ZOOM, 2), (ZOOM, -1), (ALONG, 1)])
        self.assertEqual((WHEEL_KEYS, WHEEL_PX, LOWEST, HIGHEST), (2, 36, 12, 131))

    def test_a_help_line_names_the_left_press_the_right_press_and_the_wheel_in_that_order(self) -> None:
        self.assertEqual(told("pick it", "its menu", "along the rows"),
                         "press: pick it   right press: its menu   wheel: along the rows")
        # What does nothing is left out.
        self.assertEqual([told("pick it"), told(right="its menu"), told("pick it", wheel="steps it"), told()],
                         ["press: pick it", "right press: its menu", "press: pick it   wheel: steps it", ""])

    def test_the_sums_of_a_clock_and_of_the_keys_are_opposites_and_a_zoom_keeps_a_moment_where_it_is(self) -> None:
        self.assertEqual([x_at(50, 0.0, 9, moment) for moment in (0, 1, 4.5)], [50, 59, 91])
        for start, scale in ((0.0, 9), (2.5, 7), (-16.0, 2), (0.31, 123.4)):
            for whole in range(-5, 30):
                x = x_at(50, start, scale, whole)
                wide = round(scale * (whole + 1)) - round(scale * whole)
                self.assertEqual(x_at(50, start, scale, whole + 1) - x, wide)
                self.assertAlmostEqual(moment_at(50, start, scale, x), whole, delta=0.5 / scale + 1e-9)
            # The moment at x 162 is still there at a scale of 20.
            fresh = zoomed(50, start, scale, 20, 162)
            self.assertAlmostEqual(moment_at(50, fresh, 20, 162), moment_at(50, start, scale, 162), delta=0.5 / 20)
        # The key of the top pitch has its upper edge at the upper edge, and half a key further up it is cut.
        self.assertEqual([key_rect(KEYS, 72.0, 11, pitch) for pitch in (72, 71)], [(10, 30, 40, 11), (10, 41, 40, 11)])
        self.assertEqual(key_rect(KEYS, 72.5, 11, 72)[1], 36)
        for top, key_h in ((72.0, 11), (64.25, 11), (90.0, 3), (40.5, 1)):
            for pitch in range(LOWEST, HIGHEST + 1):
                y = key_rect(KEYS, top, key_h, pitch)[1]
                self.assertEqual({pitch_at(KEYS, top, key_h, y + step) for step in range(key_h)}, {pitch})


class Laid(unittest.TestCase):
    def test_a_heading_has_its_kind_its_name_and_its_facts_from_the_left_and_its_buttons_from_the_right(self) -> None:
        self.assertEqual(heading_parts(HEAD, "SONG", "piece", "9 patterns, 18 channels"),
                         ((134, 26, 23, 17), (163, 26, 29, 17), (204, 26, 137, 17)))
        # A button is as wide as its text needs, with the room of its arrow, and a choice as its options need.
        choice = Choice((0, 0, 0, 0), ("NOTES", "MIXER", "LAYERS"))
        chords, hold = Button((0, 0, 0, 0), "CHORDS", arrow=True), Toggle((0, 0, 0, 0), "HOLD")
        self.assertEqual([wide(widget) for widget in (choice, chords, hold, Label((1, 2, 30, 4), "x"))],
                         [3 * 8 + 29 + 29 + 35, 35 + 8 + 8, 23 + 8, 30])
        # They stand two pixels from the top and from the right end, two apart, the last at the end.
        self.assertEqual(right_cells(HEAD, [117, 51, 31]),
                         [(577, 28, 117, 13), (696, 28, 51, 13), (749, 28, 31, 13)])
        heading = Heading(HEAD, "SONG", theme.SONG)
        heading.show("piece", "9 patterns, 18 channels")
        heading.right([choice, chords, hold])
        self.assertEqual((choice.rect, chords.rect, hold.rect, heading.hit((140, 30)), heading.help),
                         ((577, 28, 117, 13), (696, 28, 51, 13), (749, 28, 31, 13), False, ""))
        # A heading too narrow for all of it cuts the texts before the buttons, and the buttons to itself.
        narrow = (128, 26, 200, 17)
        heading.rect = narrow
        heading.right([choice, chords, hold])
        cut = heading_parts(narrow, "SONG", "piece", "9 patterns", choice.rect[0])
        self.assertEqual((choice.rect, [part[2] for part in cut]), ((128, 28, 112, 13), [0, 0, 0]))
        self.assertEqual(heading_parts(narrow, "SONG", "piece", "9 patterns", 200)[1:],
                         ((163, 26, 29, 17), (204, 26, 0, 17)))

    def test_the_controls_of_a_bar_stand_from_the_left_and_a_second_group_up_to_the_right_end(self) -> None:
        self.assertEqual((theme.BAR_H, theme.BAR_TOP + theme.BAR_ROW_H + 1), (40, 40))
        left, right = bar_cells(BAR, [89, 78, 37], [56, 43])
        self.assertEqual((left, right), ([(134, 123, 89, 37), (229, 123, 78, 37), (313, 123, 37, 37)],
                                         [(671, 123, 56, 37), (733, 123, 43, 37)]))
        # In a bar too narrow the right group has its room first, and what has none has no width.
        left, right = bar_cells((0, 0, 300, 40), [89, 78, 37], [56, 43])
        self.assertEqual(([cell[2] for cell in left], right), ([89, 78, 0], [(189, 2, 56, 37), (251, 2, 43, 37)]))
        left, right = bar_cells((0, 0, 60, 40), [89], [56, 43])
        self.assertEqual(([cell[2] for cell in left], [cell[2] for cell in right]), ([0], [0, 43]))
        self.assertTrue(all(cell[2] == 0 for group in bar_cells((0, 0, 8, 3), [10], [10]) for cell in group))

    def test_a_bar_gives_a_knob_the_height_of_its_row_and_any_other_control_that_of_a_button_under_its_name(
            self) -> None:
        bar = Bar(BAR)
        inst, volume = Stepper((0, 0, 0, 0), ("lead",)), Knob((0, 0, 0, 0), 1.0, 0.0, 2.0)
        follow, tempo = Toggle((0, 0, 0, 0), "FOLLOW"), Stepper((0, 0, 0, 0), ("150",))
        rects = bar.lay([Slot("CHANNEL", None, 89, "left"), Slot("INSTRUMENT", inst, 78), Slot("VOLUME", volume, 37)],
                        [Slot("TEMPO", tempo, 56), Slot("", follow, 43, tall=True)])
        self.assertEqual(rects, [(134, 135, 89, 13), (229, 135, 78, 13), (313, 123, 37, 37), (671, 135, 56, 13),
                                 (733, 123, 43, 37)])
        self.assertEqual((inst.rect, volume.rect, tempo.rect, follow.rect), (rects[1], rects[2], rects[3], rects[4]))
        # The knob is told its name, which it draws itself. The bar has the names of the others, over them.
        self.assertEqual((volume.text, bar.hit((140, 130)), bar._names),
                         ("VOLUME", False, [("CHANNEL", (134, 123, 89, 11), "left", None),
                                            ("INSTRUMENT", (229, 123, 78, 11), "centre", inst),
                                            ("TEMPO", (671, 123, 56, 11), "centre", tempo)]))


class Pressed(unittest.TestCase):
    def test_a_press_on_a_key_is_heard_and_the_key_is_down_until_the_button_is_let_go(self) -> None:
        ui, heard, typed = Ui(), [], []
        keys = ui.add(Keys(KEYS, heard.append, "the keys", "along the keys"))
        keys.typed = lambda event: typed.append(event.key) or event.key == "z"
        # Ten keys from c5 down to d-4, each 11 high: the pixel at y 85 is on g4.
        self.assertEqual((keys.top, keys.key_h, keys.pitches(), keys.at((20, 85)), keys.at((60, 85))),
                         (72.0, 11, range(63, 73), 67, None))
        ui.handle(Event("move", (20, 85)))
        self.assertEqual(ui.help, "press: hear g4   wheel: along the keys")
        self.assertTrue(ui.handle(Event("down", (20, 85), button=1)))
        self.assertEqual((heard, keys.down), ([67], 67))
        ui.handle(Event("up", (20, 20), button=1))
        self.assertEqual((heard, keys.down), ([67], None))
        # A key it is handed goes on to its editor, and the right button and the wheel are not the keyboard's.
        self.assertEqual([ui.handle(Event("key", key=key)) for key in ("z", "x")], [True, False])
        self.assertEqual((ui.focus, typed, heard), (keys, ["z", "x"], [67]))
        self.assertEqual([click(ui, (20, 85), 3), ui.handle(Event("wheel", (20, 85), wheel=1))], [False, False])
        # It stands where its owner has it, and has no key under the lowest or over the highest.
        keys.top, keys.key_h = 132.0, 3
        self.assertEqual((keys.pitches(), keys.at((20, 30)), keys.at((20, 33)), click(ui, (20, 30))),
                         (range(96, 132), None, 131, False))
        ui.handle(Event("move", (20, 30)))
        self.assertEqual(ui.help, "the keys")

    def test_a_scroll_bar_lies_or_stands_by_its_shape_and_has_no_thumb_while_everything_is_in_view(self) -> None:
        ui, moves = Ui(), []
        lying = ui.add(ScrollBar((50, 100, 200, 4), moves.append, "the rows"))
        lying.show(800, 200, 300)
        self.assertEqual((lying.upright, ScrollBar((0, 0, 4, 50)).upright, lying.thumb(), lying.busy),
                         (False, True, (125, 100, 50, 4), False))
        # The thumb travels 150 pixels for the 600 that are out of view: 15 pixels on are 60.
        ui.handle(Event("down", (130, 101), button=1))
        ui.handle(Event("move", (145, 180)))
        self.assertEqual((moves, lying.scroll.offset, lying.thumb(), lying.busy), ([360], 360, (140, 100, 50, 4), True))
        ui.handle(Event("up", (145, 180), button=1))
        # A press beside the thumb goes a view's length that way, and stops at the end.
        self.assertEqual([click(ui, (60, 101)), click(ui, (60, 101)), click(ui, (60, 101)), lying.busy],
                         [True, True, True, False])
        self.assertEqual((moves, lying.scroll.offset), ([360, 160, 0], 0))
        # With all of it in view there is no thumb, and neither a press nor the wheel is the bar's.
        lying.show(150, 200, 0)
        self.assertEqual((lying.thumb(), click(ui, (60, 101)), ui.handle(Event("wheel", (60, 101), wheel=1)), moves),
                         (None, False, False, [360, 160, 0]))

    def test_the_bar_of_a_velocity_stands_on_the_floor_of_the_lane_and_a_height_gives_its_value_back(self) -> None:
        # The lane leaves two pixels free at its floor and at its top: 30 of its 34 are for the bars.
        self.assertEqual([velocity_bar(LANE, 60, 36, value) for value in (0.0, 0.5, 1.0, 2.0)],
                         [(60, 182, 3, 0), (60, 167, 3, 15), (60, 152, 3, 30), (60, 152, 3, 30)])
        self.assertEqual(velocity_bar(LANE, 60, 2, 1.0)[2], 2)
        for step in range(16):
            top = velocity_bar(LANE, 60, 36, step / 15)[1]
            self.assertAlmostEqual(velocity_at(LANE, top), step / 15)
        self.assertEqual([velocity_at(LANE, y) for y in (0, 152, 182, 183, 400)], [1.0, 1.0, 0.0, 0.0, 0.0])
        # A drag passes the bars whose notes it is over, in the order it comes to them, each with the y it had
        # where it was last over that note. Where one of them is picked, only the picked ones are passed.
        plain = tuple((x, w, value, False) for x, w, value, _ in ITEMS)
        self.assertEqual([passed(plain, (70, 160), (70, 160)), passed(plain, (97, 160), (97, 160)),
                          passed(plain, (110, 170), (110, 170))], [[(0, 160)], [], [(1, 170), (2, 170)]])
        self.assertEqual(passed(plain, (50, 150), (130, 190)), [(0, 173), (1, 184), (2, 188)])
        self.assertEqual(passed(plain, (130, 190), (50, 150)), [(2, 180), (1, 175), (0, 155)])
        self.assertEqual((passed(ITEMS, (50, 150), (130, 190)), passed(ITEMS, (70, 160), (70, 160))),
                         ([(1, 184)], [(0, 160)]))

    def test_a_press_and_a_drag_in_the_lane_set_values_and_a_right_press_asks_for_the_usual_one(self) -> None:
        ui, told_of = Ui(), []
        lane = ui.add(Velocity(LANE, lambda index, value: told_of.append((index, value)),
                               lambda: told_of.append("settled")))
        lane.show(ITEMS)
        self.assertEqual((lane.items, lane.help, lane.room, lane.busy),
                         (ITEMS, VELOCITY_HELP, (11, 151, 298, 32), False))
        # A press on the first note, a move up on it, and a move on to the picked one, which is then set alone.
        self.assertTrue(ui.handle(Event("down", (70, 167), button=1)))
        ui.handle(Event("move", (70, 152)))
        ui.handle(Event("move", (104, 152)))
        self.assertEqual((told_of, lane.busy), ([(0, 0.5), (0, 1.0), (1, 1.0)], True))
        ui.handle(Event("up", (104, 152), button=1))
        self.assertEqual((told_of[3:], lane.busy), ([(1, 1.0), "settled"], False))
        # The right button on a note asks for its usual value and settles at once. Off the notes it is not used.
        told_of.clear()
        self.assertEqual([click(ui, (70, 160), 3), click(ui, (97, 160), 3), told_of],
                         [True, False, [(0, None), "settled"]])
        # Outside the stretch of the bars no press is used, and a drag that leaves it goes on at its edge: the
        # first note, which ends before the stretch begins, is not passed.
        lane.show([(x, w, value, False) for x, w, value, _ in ITEMS])
        lane.span = (98, 100)
        told_of.clear()
        self.assertEqual((lane.room, click(ui, (70, 160)), ui.handle(Event("wheel", (104, 160), wheel=1))),
                         ((98, 151, 100, 32), False, False))
        ui.handle(Event("down", (120, 167), button=1))
        ui.handle(Event("move", (30, 167)))
        self.assertEqual(told_of, [(2, 0.5), (2, 0.5), (1, 0.5)])
        # A drag that is dropped tells nothing more, and a key the lane is handed goes on to its editor.
        lane.drop()
        ui.handle(Event("up", (30, 167), button=1))
        lane.typed = lambda event: event.key == "delete"
        self.assertEqual((len(told_of), [ui.handle(Event("key", key=key)) for key in ("delete", "a")]),
                         (3, [True, False]))

    def test_the_menu_of_a_named_thing_has_rename_copy_what_is_special_and_remove_and_dims_what_cannot_be_done(
            self) -> None:
        did = []
        special = [Item("Move up", lambda: did.append("up")), Item("Make it a drum channel", None, enabled=False)]
        menu = thing_menu(lambda: did.append("rename"), None, special, lambda: did.append("remove"))
        self.assertEqual([item and (item.text, item.enabled) for item in menu],
                         [("Rename...", True), ("Copy", False), ("Move up", True), ("Make it a drum channel", False),
                          None, ("Remove", True)])
        for item in menu:
            if item is not None and item.enabled:
                item.value()
        self.assertEqual(did, ["rename", "up", "remove"])
        # With nothing that can be done, the three are all there, and all dim.
        self.assertEqual([item and (item.text, item.enabled) for item in thing_menu()],
                         [("Rename...", False), ("Copy", False), None, ("Remove", False)])


@unittest.skipUnless(pygame, NEEDS)
class Drawn(unittest.TestCase):
    def setUp(self) -> None:
        self.surface = pygame.Surface((800, 200))
        self.surface.fill(theme.GOOD)
        self.canvas = Canvas(self.surface, Font(sprites.load("font")))
        self.ui = Ui()

    def at(self, *pos: int) -> tuple[int, ...]:
        """The colour of one pixel."""
        return tuple(self.surface.get_at(pos))[:3]

    def colours(self, rect: layout.Rect) -> set[tuple[int, ...]]:
        """The colours a rectangle has in it."""
        x, y, w, h = rect
        return {self.at(x + column, y + row) for column in range(w) for row in range(h)}

    def test_a_heading_and_a_bar_are_outlined_panels_with_dim_words_and_the_name_in_its_colour(self) -> None:
        heading = self.ui.add(Heading(HEAD, "SONG", theme.SONG))
        heading.show("piece", "9 patterns, 18 channels")
        bar = self.ui.add(Bar(BAR))
        echo, hidden = self.ui.add(Toggle((0, 0, 0, 0), "ON")), self.ui.add(Stepper((0, 0, 0, 0), ("x",)))
        knob = self.ui.add(Knob((0, 0, 0, 0), 1.0, 0.0, 2.0))
        bar.lay([Slot("ECHO", echo, 31), Slot("GONE", hidden, 40), Slot("VOLUME", knob, 37)])
        hidden.visible = False
        self.ui.draw(self.canvas)
        kind, name, facts = ((x, y + 1, w, h - 2)
                             for x, y, w, h in heading_parts(HEAD, "SONG", "piece", "9 patterns, 18 channels"))
        self.assertEqual([self.colours(part) for part in (kind, name, facts)],
                         [{theme.PANEL, theme.DIM}, {theme.PANEL, theme.SONG}, {theme.PANEL, theme.DIM}])
        self.assertEqual((self.colours((128, 26, 654, 1)), self.colours((128, 121, 1, 40)), self.at(127, 30)),
                         ({theme.LINE}, {theme.LINE}, theme.GOOD))
        # The name of a control that is seen is over it, and that of one that is not is not drawn. The knob has
        # its own name on the same line.
        self.assertEqual([theme.DIM in self.colours(room) for room in ((134, 123, 31, 11), (171, 123, 40, 37),
                                                                      (217, 123, 37, 11))], [True, False, True])
        self.assertEqual((echo.rect, knob.rect), ((134, 135, 31, 13), (217, 123, 37, 37)))

    def test_every_c_of_the_keyboard_is_named_whole_also_where_a_key_is_lower_than_its_name(self) -> None:
        keys = self.ui.add(Keys(KEYS))
        keys.top, keys.key_h = 76.0, 3
        self.ui.draw(self.canvas)
        # 110 pixels are 36 keys and most of another: c5 and c4 and c3 are among them. Each name has its room
        # at the right end of its key, cleared to the light of the key, and all of its letters are in it.
        self.assertEqual(keys.pitches(), range(40, 77))
        for pitch in (72, 60, 48):
            room = name_room(key_rect(KEYS, 76.0, 3, pitch), pitch)
            self.assertEqual((room[2:], self.colours(room)), ((13, 9), {theme.TEXT, theme.BACKGROUND}))
            inked = [(x, y) for x in range(room[0], room[0] + 13) for y in range(room[1], room[1] + 9)
                     if self.at(x, y) == theme.BACKGROUND]
            across, down = {x for x, _ in inked}, {y for _, y in inked}
            self.assertEqual((min(across), max(across), min(down), max(down)),
                             (room[0] + 1, room[0] + 11, room[1] + 1, room[1] + 7))
        # A black key is dark for three fifths of the width, and the keyboard ends a pixel short of its rectangle.
        e_flat = key_rect(KEYS, 76.0, 3, 63)[1]
        self.assertEqual([self.at(x, e_flat + 1) for x in (10, 33, 34, 48, 49)],
                         [theme.BACKGROUND, theme.BACKGROUND, theme.TEXT, theme.TEXT, theme.PANEL])
        self.assertEqual(self.colours((0, 20, 60, 10)) | self.colours((0, 140, 60, 10)), {theme.GOOD})

    def test_the_lane_is_named_at_the_end_it_is_told_and_a_plus_is_a_small_button_with_two_strokes(self) -> None:
        lane = self.ui.add(Velocity(LANE, colour=theme.SOUND, align="right"))
        lane.show(ITEMS)
        lane.playhead = 200
        fired: list[str] = []
        plus = self.ui.add(Plus((400, 150, theme.PLUS, theme.PLUS), lambda: fired.append("more"), "one more voice"))
        self.ui.draw(self.canvas)
        # The picked bar is in the colour of text and lies over the one it shares its place with.
        self.assertEqual([self.at(*pos) for pos in ((61, 170), (61, 166), (101, 153), (110, 180), (200, 160),
                                                    (10, 150), (309, 183))],
                         [theme.SOUND, theme.PANEL, theme.TEXT, theme.SOUND, theme.ACTIVE, theme.LINE, theme.LINE])
        self.assertEqual((theme.DIM in self.colours((250, 152, 54, 11)), self.colours((11, 151, 40, 32))),
                         (True, {theme.PANEL}))
        inked = {(x, y) for x in range(400, 411) for y in range(150, 161) if self.at(x, y) == theme.TEXT}
        self.assertEqual(inked, {(x, 155) for x in range(403, 408)} | {(405, y) for y in range(153, 158)})
        click(self.ui, (405, 155))
        self.ui.handle(Event("move", (405, 155)))
        self.assertEqual((fired, self.ui.help, isinstance(plus, Button)), (["more"], "one more voice", True))


if __name__ == "__main__":
    unittest.main()
