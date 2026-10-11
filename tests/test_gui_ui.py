"""Tests for rowsynth.gui.ui: events, who they go to, who has the focus, and the first widgets."""
from __future__ import annotations

import dataclasses
import os
import unittest
from unittest import mock

from rowsynth.errors import RowSynthError
from rowsynth.gui import layout, sprites, theme
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.font import Font
from rowsynth.gui.lists import List, Row
from rowsynth.gui.ui import Button, Choice, Event, Label, Toggle, Ui, Widget, inside, share, translate

# pygame reads these when it is imported: no test opens a window or plays a sound.
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
try:
    import pygame
except ImportError:
    pygame = None

NEEDS = "pygame is not installed"
CTRL = frozenset({"ctrl"})
#: Five rows of a list, known by their numbers. In a list at the top of the screen, row 2 is 2 * ROW down.
ROWS = tuple(Row(number, f"row {number}") for number in range(5))
ROW = theme.LIST_ROW_H


class Probe(Widget):
    """A widget that notes the kinds of the events it is offered, and uses those of the kinds it is given."""

    def __init__(self, rect: layout.Rect, uses: tuple[str, ...] = (), help: str = "") -> None:
        super().__init__(rect, help)
        self.uses = uses
        self.seen: list[str] = []

    def on(self, event: Event, ui: Ui) -> bool:
        self.seen.append(event.kind)
        return event.kind in self.uses


def click(ui: Ui, pos: tuple[int, int], button: int = 1) -> bool:
    """Press a mouse button at a position and let it go there. Whether the press was used."""
    used = ui.handle(Event("down", pos, button=button))
    ui.handle(Event("up", pos, button=button))
    return used


class Events(unittest.TestCase):
    def test_an_event_cannot_be_changed_and_what_it_is_not_told_is_empty(self) -> None:
        event = Event("move", (3, 4))
        self.assertEqual((event.button, event.wheel, event.key, event.mods), (0, 0, "", frozenset()))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            event.pos = (5, 6)
        with self.assertRaises(RowSynthError) as refused:
            Event("drag")
        self.assertEqual(str(refused.exception), "an event is one of down, up, move, wheel, key, not 'drag'")


class Handing(unittest.TestCase):
    def test_a_rectangle_is_hit_on_its_own_pixels_and_a_label_never(self) -> None:
        rect = (10, 20, 4, 3)
        self.assertEqual([inside(rect, pos) for pos in ((10, 20), (13, 22), (9, 20), (14, 20), (10, 19), (10, 23))],
                         [True, True, False, False, False, False])
        self.assertEqual([Widget(rect).hit((13, 22)), Widget(rect).hit((14, 22))], [True, False])
        self.assertFalse(Label(rect, "name").hit((11, 21)))

    def test_the_topmost_visible_enabled_widget_under_the_mouse_is_hot_and_its_help_is_the_line(self) -> None:
        ui = Ui()
        low, top = ui.add(Probe((0, 0, 20, 20), help="low")), ui.add(Probe((10, 10, 20, 20), help="top"))
        self.assertEqual((ui.hot, ui.help), (None, ""))
        for pos, hot, line in (((15, 15), top, "top"), ((5, 5), low, "low"), ((40, 40), None, "")):
            ui.handle(Event("move", pos))
            self.assertEqual((ui.hot, ui.help), (hot, line))
        ui.handle(Event("move", (15, 15)))
        top.visible = False
        self.assertIs(ui.hot, low)
        top.visible, top.enabled = True, False
        self.assertIs(ui.hot, low)

    def test_what_the_topmost_widget_does_not_use_is_offered_to_the_one_below(self) -> None:
        ui = Ui()
        low, top = ui.add(Probe((0, 0, 20, 20), ("wheel",))), ui.add(Probe((10, 10, 20, 20)))
        self.assertTrue(ui.handle(Event("wheel", (15, 15), wheel=1)))
        self.assertEqual((top.seen, low.seen), (["wheel"], ["wheel"]))
        # A press that no widget used holds none of them.
        self.assertFalse(ui.handle(Event("down", (15, 15), button=1)))
        self.assertIsNone(ui.held)
        self.assertFalse(ui.handle(Event("wheel", (40, 40), wheel=1)))

    def test_a_widget_that_used_a_down_gets_the_moves_and_the_up_also_when_the_mouse_has_left_it(self) -> None:
        ui = Ui()
        grip, other = ui.add(Probe((0, 0, 10, 10), ("down",))), ui.add(Probe((20, 0, 10, 10), ("down",)))
        self.assertTrue(ui.handle(Event("down", (5, 5), button=1)))
        self.assertIs(ui.held, grip)
        self.assertTrue(ui.handle(Event("move", (25, 5))))
        # No other widget is hot meanwhile, and the release of another button does not end it.
        self.assertIsNone(ui.hot)
        self.assertTrue(ui.handle(Event("up", (25, 5), button=3)))
        self.assertIs(ui.held, grip)
        self.assertTrue(ui.handle(Event("up", (25, 5), button=1)))
        self.assertIsNone(ui.held)
        self.assertEqual((grip.seen, other.seen), (["down", "move", "up", "up"], []))
        self.assertFalse(ui.handle(Event("move", (25, 5))))
        self.assertEqual((ui.hot, other.seen), (other, ["move"]))

    def test_a_key_is_offered_to_the_widget_that_has_the_focus_and_to_no_other_and_then_to_keys(self) -> None:
        ui, called = Ui(), []
        ui.keys[("space", frozenset())] = lambda: called.append("play")
        ui.keys[("z", CTRL)] = lambda: called.append("undo")
        deaf, keen = ui.add(Probe((0, 0, 5, 5), ("down",))), ui.add(Probe((10, 0, 5, 5), ("down", "key")))
        # Before any press no widget has the focus, and a key is offered to none: not to one that would use it.
        self.assertEqual([ui.handle(Event("key", key="space")), ui.handle(Event("key", key="z")),
                          ui.handle(Event("key", key="z", mods=CTRL))], [True, False, True])
        self.assertEqual((ui.focus, called, deaf.seen, keen.seen), (None, ["play", "undo"], [], []))
        # The widget that used a press has the focus. What it does not use of the keys goes on to `keys`.
        click(ui, (2, 2))
        self.assertTrue(ui.handle(Event("key", key="space")))
        self.assertEqual((ui.focus, called, deaf.seen, keen.seen),
                         (deaf, ["play", "undo", "play"], ["down", "up", "key"], []))
        # A press that another widget uses takes the focus there, and a key that one uses goes no further.
        click(ui, (12, 2))
        self.assertTrue(ui.handle(Event("key", key="space")))
        self.assertEqual((ui.focus, called, deaf.seen, keen.seen),
                         (keen, ["play", "undo", "play"], ["down", "up", "key"], ["down", "up", "key"]))


class Focus(unittest.TestCase):
    def test_with_two_lists_the_arrow_keys_move_only_the_one_that_was_clicked(self) -> None:
        ui = Ui()
        left, right = ui.add(List((0, 0, 60, 100), ROWS)), ui.add(List((70, 0, 60, 100), ROWS))
        click(ui, (5, ROW))
        for _ in range(2):
            self.assertTrue(ui.handle(Event("key", key="down")))
        self.assertEqual((ui.focus, left.picked, right.picked), (left, 3, None))
        # The list that was added last has no more claim to the keys than the other.
        click(ui, (75, 0))
        self.assertTrue(ui.handle(Event("key", key="end")))
        self.assertEqual((ui.focus, left.picked, right.picked), (right, 3, 4))

    def test_a_click_on_empty_ground_takes_the_focus_away_and_the_keys_then_reach_keys(self) -> None:
        ui, called = Ui(), []
        ui.keys[("down", frozenset())] = lambda: called.append("down")
        rows = ui.add(List((0, 0, 60, 100), ROWS))
        click(ui, (5, 0))
        self.assertTrue(ui.handle(Event("key", key="down")))
        self.assertEqual((ui.focus, rows.picked, called), (rows, 1, []))
        self.assertFalse(click(ui, (80, 5)))
        self.assertTrue(ui.handle(Event("key", key="down")))
        self.assertEqual((ui.focus, rows.picked, called), (None, 1, ["down"]))

    def test_a_widget_that_is_hidden_or_not_enabled_loses_the_focus_and_does_not_have_it_when_it_is_back(self) -> None:
        for gone in ("visible", "enabled"):
            ui = Ui()
            probe = ui.add(Probe((0, 0, 10, 10), ("down", "key")))
            click(ui, (5, 5))
            self.assertIs(ui.focus, probe)
            setattr(probe, gone, False)
            self.assertIsNone(ui.focus)
            setattr(probe, gone, True)
            self.assertFalse(ui.handle(Event("key", key="up")))
            self.assertEqual((ui.focus, probe.seen), (None, ["down", "up"]))
        # Nobody has to ask in between: a frame drawn without the widget is enough. A probe draws on no canvas.
        click(ui, (5, 5))
        probe.visible = False
        ui.draw(None)
        probe.visible = True
        self.assertIsNone(ui.focus)

    def test_the_space_bar_reaches_keys_while_a_list_has_the_focus(self) -> None:
        ui, called = Ui(), []
        ui.keys[("space", frozenset())] = lambda: called.append("play")
        rows = ui.add(List((0, 0, 60, 100), ROWS))
        click(ui, (5, 0))
        self.assertTrue(ui.handle(Event("key", key="space")))
        self.assertEqual((ui.focus, rows.picked, called), (rows, 0, ["play"]))


class Widgets(unittest.TestCase):
    def test_a_button_fires_on_a_release_that_is_still_on_it_and_never_when_it_is_not_enabled(self) -> None:
        ui, fired = Ui(), []
        button = ui.add(Button((10, 10, 30, 13), "GO", action=lambda: fired.append("go"), help="go there"))
        self.assertTrue(ui.handle(Event("down", (12, 12), button=1)))
        self.assertEqual(fired, [])
        ui.handle(Event("up", (39, 22), button=1))
        self.assertEqual(fired, ["go"])
        # Let go off the button, pressed off the button, and pressed with the right button: none of them fires.
        ui.handle(Event("down", (12, 12), button=1))
        ui.handle(Event("move", (50, 50)))
        ui.handle(Event("up", (50, 50), button=1))
        ui.handle(Event("down", (50, 50), button=1))
        ui.handle(Event("up", (12, 12), button=1))
        self.assertFalse(click(ui, (12, 12), button=3))
        self.assertEqual((fired, ui.hot, ui.help), (["go"], button, "go there"))
        button.enabled = False
        self.assertFalse(click(ui, (12, 12)))
        self.assertEqual((fired, ui.hot, ui.help), (["go"], None, ""))
        # One with no action, on top of it, can be pressed, and nothing comes of it.
        ui.add(Button((10, 10, 30, 13)))
        self.assertTrue(click(ui, (12, 12)))
        self.assertEqual(fired, ["go"])

    def test_a_toggle_flips_at_each_press_and_tells_its_new_state(self) -> None:
        ui, told = Ui(), []
        toggle = ui.add(Toggle((0, 0, 30, 13), "LOOP", changed=told.append))
        self.assertFalse(toggle.is_on)
        click(ui, (5, 5))
        self.assertTrue(toggle.is_on)
        click(ui, (5, 5))
        self.assertEqual((toggle.is_on, told), (False, [True, False]))
        self.assertTrue(Toggle((0, 0, 30, 13), on=True).is_on)

    def test_a_choice_keeps_exactly_one_on_and_tells_when_another_is_picked(self) -> None:
        ui, told = Ui(), []
        choice = ui.add(Choice((10, 0, 78, 13), ("PATTERN", "SONG"), changed=told.append))
        self.assertEqual((choice.picked, choice.at((57, 5)), choice.at((58, 5)), choice.at((88, 5))), (0, 0, 1, None))
        click(ui, (60, 5))
        self.assertEqual((choice.picked, told), (1, [1]))
        # The one that is on stays on when it is pressed, and nothing is told.
        click(ui, (60, 5))
        self.assertEqual((choice.picked, told), (1, [1]))
        click(ui, (12, 5))
        self.assertEqual((choice.picked, told), (0, [1, 0]))
        for options, picked, wrong in (((), 0, "a choice needs at least one option"),
                                       (("A", "B"), 2, "a choice of 2 options has no option 2")):
            with self.assertRaises(RowSynthError) as refused:
                Choice((0, 0, 40, 13), options, picked)
            self.assertEqual(str(refused.exception), wrong)

    def test_a_choice_shares_its_rectangle_out_by_the_width_of_its_texts(self) -> None:
        # Each text with the room a button leaves at its two sides: 49 and 31 of 80, here of 78 and then of 160.
        self.assertEqual([share(10, (1, 1, 1)), share(0, (2, 1)), share(7, (5,))], [[3, 4, 3], [0, 0], [7]])
        self.assertEqual(Choice((10, 2, 78, 13), ("PATTERN", "SONG")).cells(), [(10, 2, 48, 13), (58, 2, 30, 13)])
        self.assertEqual(Choice((10, 2, 160, 13), ("PATTERN", "SONG")).cells(), [(10, 2, 98, 13), (108, 2, 62, 13)])


@unittest.skipUnless(pygame, NEEDS)
class Drawn(unittest.TestCase):
    def setUp(self) -> None:
        self.surface = pygame.Surface((100, 30))
        self.surface.fill(theme.PANEL)
        self.canvas = Canvas(self.surface, Font(sprites.load("font")))
        self.ui = Ui()

    def at(self, *pos: int) -> tuple[int, ...]:
        """The colour of one pixel."""
        return tuple(self.surface.get_at(pos))[:3]

    def colours(self, rect: layout.Rect) -> set[tuple[int, ...]]:
        """The colours a rectangle has in it."""
        x, y, w, h = rect
        return {self.at(x + column, y + row) for column in range(w) for row in range(h)}

    def face(self) -> tuple[int, ...]:
        """The colour of a button at (10, 5), read between its outline and what is on it, after it is drawn."""
        self.ui.draw(self.canvas)
        return self.at(12, 11)

    def test_the_face_of_a_button_is_raised_when_idle_lit_when_hot_and_sunk_while_held(self) -> None:
        button = self.ui.add(Button((10, 5, 40, 13), "GO"))
        self.assertEqual(self.face(), theme.RAISED)
        self.assertEqual((self.at(10, 5), self.at(49, 17), self.at(12, 6)), (theme.LINE, theme.LINE, theme.LIT))
        self.assertEqual(self.colours(button.rect), {theme.RAISED, theme.LIT, theme.LINE, theme.TEXT})
        self.ui.handle(Event("move", (12, 11)))
        self.assertEqual(self.face(), theme.LIT)
        self.ui.handle(Event("down", (12, 11), button=1))
        self.assertEqual(self.face(), theme.BACKGROUND)
        # Held with the mouse off it, it is as when idle. Not enabled, its text is dim.
        self.ui.handle(Event("move", (60, 11)))
        self.assertEqual(self.face(), theme.RAISED)
        self.ui.handle(Event("up", (60, 11), button=1))
        button.enabled = False
        self.ui.draw(self.canvas)
        self.assertEqual(self.colours(button.rect), {theme.RAISED, theme.LIT, theme.LINE, theme.DIM})

    def test_a_button_shows_its_icon_then_its_text_a_label_its_text_and_what_is_not_visible_nothing(self) -> None:
        # With no icon set the icon is the box, 5 wide: with the 3 between and the 11 of the text, 19 of the 40.
        button = self.ui.add(Button((10, 5, 40, 13), "GO", "play"))
        label = self.ui.add(Label((60, 5, 23, 13), "a long name", theme.INSTRUMENT, "right"))
        self.ui.draw(self.canvas)
        inked = sorted(x for x in range(10, 50) if theme.TEXT in self.colours((x, 5, 1, 13)))
        self.assertEqual((inked[0], inked[4], inked[5], inked[-1]), (20, 24, 28, 38))
        # The label is cut to its rectangle, and has nothing of its own behind its text.
        self.assertEqual(self.colours(label.rect), {theme.PANEL, theme.INSTRUMENT})
        self.assertEqual(self.colours((50, 0, 10, 30)) | self.colours((83, 0, 17, 30)), {theme.PANEL})
        self.surface.fill(theme.PANEL)
        button.visible = label.visible = False
        self.ui.draw(self.canvas)
        self.assertEqual(self.colours((0, 0, 100, 30)), {theme.PANEL})

    def test_what_is_on_has_its_text_and_a_strip_along_its_lower_edge_in_its_colour(self) -> None:
        toggle = self.ui.add(Toggle((10, 5, 40, 13), "LOOP", colour=theme.SONG))
        self.ui.draw(self.canvas)
        self.assertEqual(self.colours(toggle.rect), {theme.RAISED, theme.LIT, theme.LINE, theme.TEXT})
        click(self.ui, (12, 11))
        self.ui.handle(Event("move", (60, 11)))
        self.ui.draw(self.canvas)
        self.assertEqual(self.colours(toggle.rect), {theme.RAISED, theme.LIT, theme.LINE, theme.SONG})
        self.assertEqual((self.colours((10, 16, 40, 2)), self.at(10, 15)), ({theme.SONG}, theme.LINE))
        # A choice is joined toggles. Along its lower edge: the outline of the first option, then the strip of the
        # second, which is on, from the line the two share.
        toggle.visible = False
        self.surface.fill(theme.PANEL)
        self.ui.add(Choice((10, 5, 78, 13), ("PATTERN", "SONG"), picked=1))
        self.ui.draw(self.canvas)
        self.assertEqual((self.colours((10, 17, 48, 1)), self.colours((58, 16, 30, 2))),
                         ({theme.LINE}, {theme.ACTIVE}))
        self.assertEqual([self.at(x, 11) for x in (57, 58, 59)], [theme.RAISED, theme.LINE, theme.RAISED])
        self.assertIn(theme.TEXT, self.colours((11, 6, 46, 10)))
        self.assertEqual(self.colours((59, 6, 28, 10)), {theme.RAISED, theme.LIT, theme.ACTIVE})


def halved(pos: tuple[int, int]) -> tuple[int, int]:
    """A position in a window that shows an art pixel at twice its size, in art pixels."""
    return pos[0] // 2, pos[1] // 2


@unittest.skipUnless(pygame, NEEDS)
class Translating(unittest.TestCase):
    def setUp(self) -> None:
        pygame.display.init()
        self.addCleanup(pygame.display.quit)

    def test_a_mouse_press_at_a_scale_of_2_is_a_down_in_art_pixels_and_what_is_of_no_use_is_none(self) -> None:
        press = pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(21, 11), button=3)
        self.assertEqual(translate(press, halved), Event("down", (10, 5), button=3))
        release = pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(21, 11), button=1)
        self.assertEqual(translate(release, halved), Event("up", (10, 5), button=1))
        moved = pygame.event.Event(pygame.MOUSEMOTION, pos=(8, 9), rel=(1, 0), buttons=(0, 0, 0))
        self.assertEqual(translate(moved, halved), Event("move", (4, 4)))
        # A key let go, the wheel told a second time as a button, and the window gaining the keyboard.
        for event in (pygame.event.Event(pygame.KEYUP, key=pygame.K_z, mod=0),
                      pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(1, 1), button=4),
                      pygame.event.Event(pygame.ACTIVEEVENT, gain=1, state=1)):
            self.assertIsNone(translate(event, halved))

    def test_a_step_of_the_wheel_is_told_where_the_mouse_is_with_the_keys_that_are_held(self) -> None:
        step = pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-1)
        with mock.patch("pygame.mouse.get_pos", return_value=(21, 11)), \
                mock.patch("pygame.key.get_mods", return_value=pygame.KMOD_LSHIFT | pygame.KMOD_NUM):
            self.assertEqual(translate(step, halved), Event("wheel", (10, 5), wheel=-1, mods=frozenset({"shift"})))

    def test_a_key_going_down_has_pygames_name_for_it_and_ctrl_when_that_is_held(self) -> None:
        pressed = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_z, mod=pygame.KMOD_RCTRL)
        self.assertEqual(translate(pressed, halved), Event("key", key="z", mods=CTRL))
        space = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE, mod=pygame.KMOD_LALT | pygame.KMOD_LSHIFT)
        self.assertEqual(translate(space, halved), Event("key", key="space", mods=frozenset({"alt", "shift"})))


if __name__ == "__main__":
    unittest.main()
