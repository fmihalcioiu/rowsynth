"""Tests for rowsynth.gui.ui: events, who they go to, who has the focus, and the first widgets."""
from __future__ import annotations

import dataclasses
import os
import types
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
    """A widget that notes the kinds of the events it is offered, and uses those of the kinds it is given.

    It notes the sizes of the art surface it is told as well. `press` is called with the Ui when a button goes
    down on it.
    """

    def __init__(self, rect: layout.Rect, uses: tuple[str, ...] = (), help: str = "", press=None) -> None:
        super().__init__(rect, help)
        self.uses = uses
        self.press = press
        self.seen: list[str] = []
        self.sizes: list[tuple[int, int]] = []

    def on(self, event: Event, ui: Ui) -> bool:
        self.seen.append(event.kind)
        if event.kind == "down" and self.press is not None:
            self.press(ui)
        return event.kind in self.uses

    def keep_in(self, size: tuple[int, int]) -> None:
        self.sizes.append(size)


class Field(Probe):
    """A probe that is typed into."""

    takes_text = True


def blank(size: tuple[int, int] = (100, 30)) -> types.SimpleNamespace:
    """A stand-in for a canvas that nothing is drawn on: the Ui asks it for the size of its surface only."""
    return types.SimpleNamespace(surface=types.SimpleNamespace(get_size=lambda: size))


def click(ui: Ui, pos: tuple[int, int], button: int = 1) -> bool:
    """Press a mouse button at a position and let it go there. Whether the press was used."""
    used = ui.handle(Event("down", pos, button=button))
    ui.handle(Event("up", pos, button=button))
    return used


class Events(unittest.TestCase):
    def test_an_event_cannot_be_changed_and_what_it_is_not_told_is_empty(self) -> None:
        event = Event("move", (3, 4))
        self.assertEqual((event.button, event.wheel, event.key, event.mods, event.text), (0, 0, "", frozenset(), ""))
        self.assertFalse(event.double)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            event.pos = (5, 6)
        with self.assertRaises(RowSynthError) as refused:
            Event("drag")
        self.assertEqual(str(refused.exception), "an event is one of down, up, move, wheel, key, text, not 'drag'")


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

    def test_a_widget_that_is_not_enabled_still_gives_its_help_line_and_one_that_is_hidden_does_not(self) -> None:
        ui = Ui()
        low, top = ui.add(Probe((0, 0, 20, 20), help="low")), ui.add(Probe((10, 10, 20, 20), help="nothing to do"))
        top.enabled = False
        ui.handle(Event("move", (15, 15)))
        self.assertEqual((ui.hot, ui.help), (low, "nothing to do"))
        top.visible = False
        self.assertEqual((ui.hot, ui.help), (low, "low"))
        # While a widget is held the line is its own while the mouse is on it, and none when it has left.
        top.visible, top.enabled = True, True
        held = ui.add(Probe((30, 0, 10, 10), ("down",), help="held"))
        ui.handle(Event("down", (35, 5), button=1))
        self.assertEqual((ui.held, ui.help), (held, "held"))
        ui.handle(Event("move", (15, 15)))
        self.assertEqual(ui.help, "")
        ui.handle(Event("up", (15, 15), button=1))
        self.assertEqual(ui.help, "nothing to do")

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


class Twice(unittest.TestCase):
    def setUp(self) -> None:
        self.now = 10.0
        self.ui = Ui(lambda: self.now)
        self.doubles: list[tuple[str, bool]] = []

    def taker(self, rect: layout.Rect, called: str, uses: bool = True) -> Widget:
        """A widget on the Ui that notes each press it is offered, with whether it was a double one."""
        outer = self

        class Taker(Widget):
            def on(self, event: Event, ui: Ui) -> bool:
                if event.kind == "down":
                    outer.doubles.append((called, event.double))
                return uses and event.kind == "down"

        return self.ui.add(Taker(rect))

    def press(self, pos: tuple[int, int], after: float = 0.1, button: int = 1) -> bool:
        """Press a button at a position so many seconds after the press before, and let it go. Was it double."""
        self.now += after
        click(self.ui, pos, button)
        return self.doubles[-1][1]

    def test_a_press_that_follows_another_on_the_same_widget_soon_and_near_is_a_double_press(self) -> None:
        self.taker((0, 0, 40, 40), "a")
        self.taker((50, 0, 40, 40), "b")
        # The second of two is double, and the one after it is a first one again.
        self.assertEqual([self.press((10, 10)), self.press((13, 7)), self.press((13, 7)), self.press((13, 7))],
                         [False, True, False, True])
        self.assertEqual((theme.DOUBLE_PRESS, theme.DOUBLE_REACH), (0.4, 3))
        # Later than the time, further than the reach either way, with another button, and on another widget.
        self.press((10, 10))
        self.assertEqual([self.press((10, 10), 0.41), self.press((14, 10)), self.press((14, 14)),
                          self.press((14, 14), button=3), self.press((60, 14))], [False] * 5)
        self.assertEqual([self.press((60, 14), 0.3), self.press((10, 10))], [True, False])

    def test_a_press_no_widget_used_is_no_first_press_and_one_a_widget_passed_on_does_not_stand_between_two(
            self) -> None:
        self.taker((0, 0, 40, 40), "under")
        self.taker((0, 0, 20, 40), "over", uses=False)
        # Both are offered each press, the one on top first, and it is the one that uses them that has the two.
        self.press((10, 10))
        self.press((10, 10))
        self.assertEqual(self.doubles, [("over", False), ("under", False), ("over", False), ("under", True)])
        # A press on empty ground between two is the end of the first.
        self.assertEqual([self.press((30, 10)), click(self.ui, (90, 90)), self.press((30, 10))], [False, False, False])
        # A widget that is open over the others is handed its own second press as a double one.
        top = Probe((0, 0, 99, 99))
        seen: list[tuple[str, bool]] = []
        top.on = lambda event, ui: seen.append((event.kind, event.double)) or True
        self.ui.open(top)
        for _ in range(2):
            click(self.ui, (30, 10))
        self.assertEqual(seen, [("down", False), ("up", False), ("down", True), ("up", False)])


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
        # Nobody has to ask in between: a frame drawn without the widget is enough. A probe draws nothing.
        click(ui, (5, 5))
        probe.visible = False
        ui.draw(blank())
        probe.visible = True
        self.assertIsNone(ui.focus)

    def test_the_space_bar_reaches_keys_while_a_list_has_the_focus(self) -> None:
        ui, called = Ui(), []
        ui.keys[("space", frozenset())] = lambda: called.append("play")
        rows = ui.add(List((0, 0, 60, 100), ROWS))
        click(ui, (5, 0))
        self.assertTrue(ui.handle(Event("key", key="space")))
        self.assertEqual((ui.focus, rows.picked, called), (rows, 0, ["play"]))


class Typing(unittest.TestCase):
    def test_typed_characters_reach_the_widget_with_the_focus_when_it_takes_text_and_nothing_else(self) -> None:
        ui = Ui()
        plain, field = ui.add(Probe((0, 0, 5, 5), ("down",))), ui.add(Field((10, 0, 5, 5), ("down", "text")))
        typed = Event("text", text="a")
        # With the focus on no widget, and then on one that is not typed into, they are handed to none.
        self.assertFalse(ui.handle(typed))
        click(ui, (2, 2))
        self.assertFalse(ui.handle(typed))
        self.assertEqual((ui.focus, plain.seen, field.seen, ui.mouse), (plain, ["down", "up"], [], (2, 2)))
        click(ui, (12, 2))
        self.assertTrue(ui.handle(typed))
        self.assertEqual((ui.focus, plain.seen, field.seen), (field, ["down", "up"], ["down", "up", "text"]))

    def test_the_keys_of_a_widget_that_takes_text_are_its_own_and_none_of_them_reaches_keys(self) -> None:
        ui, called = Ui(), []
        ui.keys[("space", frozenset())] = lambda: called.append("play")
        field = ui.add(Field((10, 0, 5, 5), ("down",)))
        click(ui, (12, 2))
        # The space bar is a key the field does not use, and after it the character it types.
        self.assertEqual([ui.handle(Event("key", key="space")), ui.handle(Event("text", text=" "))], [False, False])
        self.assertEqual((called, field.seen), ([], ["down", "up", "key", "text"]))
        # With the focus gone from the field the same key is a key, and its character is nothing.
        click(ui, (40, 2))
        self.assertEqual([ui.handle(Event("key", key="space")), ui.handle(Event("text", text=" "))], [True, False])
        self.assertEqual((called, field.seen), (["play"], ["down", "up", "key", "text"]))

    def test_the_character_of_a_key_that_opened_a_widget_to_type_into_is_not_typed_into_it(self) -> None:
        ui = Ui()
        field = Field((10, 0, 5, 5))
        ui.keys[("r", frozenset())] = lambda: ui.open(field)
        self.assertTrue(ui.handle(Event("key", key="r")))
        self.assertEqual((ui.modal, ui.focus), (field, field))
        # The character that comes right after its key says the same key again. What is typed after that goes in.
        for typed in (Event("text", text="r"), Event("key", key="e"), Event("text", text="e"), Event("text", text="f")):
            self.assertTrue(ui.handle(typed))
        self.assertEqual(field.seen, ["key", "text", "text"])


class Modal(unittest.TestCase):
    def test_while_one_is_open_every_event_goes_to_it_alone_and_the_widgets_under_it_are_not_hot(self) -> None:
        ui, called = Ui(), []
        ui.keys[("space", frozenset())] = lambda: called.append("play")
        under = ui.add(Probe((0, 0, 20, 20), ("down", "wheel", "key"), help="under"))
        click(ui, (5, 5))
        top = Probe((50, 50, 10, 10), help="top")
        self.assertIs(ui.open(top), top)
        self.assertEqual((ui.modal, ui.focus, ui.widgets, top.sizes), (top, top, [under], [theme.DESIGN]))
        # It uses none of them, and they are used all the same: nothing goes on to what is under it or to `keys`.
        happened = (Event("move", (5, 5)), Event("down", (5, 5), button=1), Event("up", (5, 5), button=1),
                    Event("wheel", (5, 5), wheel=1), Event("key", key="space"), Event("text", text=" "))
        self.assertEqual([ui.handle(event) for event in happened], [True] * 6)
        self.assertEqual((top.seen, under.seen, called), (["move", "down", "up", "wheel", "key"], ["down", "up"], []))
        # The line of help is its own wherever the mouse is, and it is hot when the mouse is on it.
        self.assertEqual((ui.hot, ui.help), (None, "top"))
        ui.handle(Event("move", (55, 55)))
        self.assertEqual((ui.hot, ui.help), (top, "top"))

    def test_closing_gives_the_focus_back_and_the_release_of_the_press_that_closed_it_reaches_nothing(self) -> None:
        ui = Ui()
        under = ui.add(Probe((0, 0, 20, 20), ("down", "key")))
        click(ui, (5, 5))
        first, top = Probe((50, 50, 10, 10)), Probe((50, 50, 10, 10), press=lambda ui: ui.close(top))
        # Only one is open at a time, and closing one that is not open changes nothing.
        ui.open(first)
        ui.open(top)
        ui.close(first)
        self.assertEqual((ui.modal, ui.focus), (top, top))
        ui.handle(Event("down", (5, 5), button=1))
        self.assertEqual((ui.modal, ui.focus, ui.held, ui.hot), (None, under, top, None))
        ui.handle(Event("up", (5, 5), button=1))
        self.assertEqual((ui.held, ui.hot, top.seen, under.seen), (None, under, ["down", "up"], ["down", "up"]))
        self.assertTrue(ui.handle(Event("key", key="up")))
        self.assertEqual(under.seen, ["down", "up", "key"])

    def test_a_widget_that_opens_one_on_a_press_gets_its_release_and_has_the_focus_when_it_closes(self) -> None:
        ui = Ui()
        top = Probe((50, 50, 10, 10))
        opener = ui.add(Probe((0, 0, 20, 20), ("down",), press=lambda ui: ui.open(top)))
        ui.handle(Event("down", (5, 5), button=3))
        self.assertEqual((ui.modal, ui.focus, ui.held), (top, top, opener))
        ui.handle(Event("up", (5, 5), button=3))
        ui.handle(Event("move", (6, 5)))
        self.assertEqual((opener.seen, top.seen, ui.held), (["down", "up"], ["move"], None))
        ui.close(top)
        self.assertIs(ui.focus, opener)


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
        # It is not hot, and it still says what it is for: that is where a button tells why it does nothing.
        self.assertEqual((fired, ui.hot, ui.help), (["go"], None, "go there"))
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

    def test_a_button_that_opens_a_list_has_a_small_dim_arrow_after_its_text_that_points_down(self) -> None:
        # The 23 pixels of the text, 3 of air and the 5 of the arrow stand in the middle of the 50, from x 19.
        self.ui.add(Button((10, 5, 50, 13), "KIND", arrow=True))
        plain = self.ui.add(Button((10, 18, 50, 12), "KIND"))
        self.ui.draw(self.canvas)
        dim = {(x, y) for x in range(100) for y in range(30) if self.at(x, y) == theme.DIM}
        self.assertEqual(dim, {(45 + step + column, 10 + step) for step in range(3) for column in range(5 - 2 * step)})
        inked = sorted(x for x in range(10, 60) if theme.TEXT in self.colours((x, 5, 1, 13)))
        self.assertEqual((inked[0], inked[-1] < 45, plain.arrow), (19, True, False))
        # Sunk, the arrow sits a pixel lower with the text.
        self.ui.handle(Event("down", (12, 11), button=1))
        self.ui.draw(self.canvas)
        self.assertEqual((self.at(47, 13), self.at(47, 10)), (theme.DIM, theme.BACKGROUND))

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

    def test_the_modal_widget_is_drawn_over_the_others_and_told_the_size_of_the_surface_when_it_changes(self) -> None:
        class Patch(Probe):
            """A probe that fills its rectangle with the colour of what is wrong."""

            def draw(self, canvas: Canvas) -> None:
                canvas.box(self.rect, theme.FAULT)

        top = self.ui.open(Patch((20, 10, 10, 5)))
        # A widget that is added while it is open still lies under it.
        self.ui.add(Button((10, 5, 40, 13), "GO"))
        self.ui.handle(Event("move", (22, 12)))
        self.ui.draw(self.canvas)
        self.assertEqual((self.colours(top.rect), self.at(19, 12), top.hot), ({theme.FAULT}, theme.RAISED, True))
        self.ui.draw(self.canvas)
        self.assertEqual((self.ui.size, top.sizes), ((100, 30), [theme.DESIGN, (100, 30)]))


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

    def test_what_is_typed_is_a_text_with_its_characters_and_the_key_that_goes_with_it_is_a_key(self) -> None:
        # pygame tells a key that types twice: the key that went down, with its character, and then the text.
        pressed = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a, mod=pygame.KMOD_LSHIFT, unicode="A")
        self.assertEqual(translate(pressed, halved), Event("key", key="a", mods=frozenset({"shift"})))
        self.assertEqual(translate(pygame.event.Event(pygame.TEXTINPUT, text="A"), halved), Event("text", text="A"))
        # Text that is still being put together, as some keyboards do it, is not typed yet.
        self.assertIsNone(translate(pygame.event.Event(pygame.TEXTEDITING, text="a", start=0, length=1), halved))


if __name__ == "__main__":
    unittest.main()
