"""Tests for rowsynth.gui.panels: a panel's own widgets, taking it off the window, and a border that is dragged."""
from __future__ import annotations

import unittest

from rowsynth.gui.panels import Border, Panel
from rowsynth.gui.ui import Event, Ui, Widget


class Probe(Widget):
    """A widget that counts the presses it was handed."""

    def __init__(self, rect: tuple[int, int, int, int]) -> None:
        super().__init__(rect)
        self.presses = 0

    def on(self, event: Event, ui: Ui) -> bool:
        if event.kind == "down":
            self.presses += 1
            return True
        return False


class Shown(unittest.TestCase):
    def setUp(self) -> None:
        self.ui = Ui()
        self.panel = Panel(None, self.ui)

    def test_a_widget_added_through_the_panel_is_in_the_ui_and_is_one_of_the_panels_own(self) -> None:
        probe = self.panel.add(Probe((0, 0, 10, 10)))
        self.assertIs(self.ui.widgets[-1], probe)
        self.assertEqual(self.panel.widgets, [probe])
        self.assertTrue(self.panel.shown)

    def test_a_panel_taken_off_hides_its_own_widgets_and_they_take_no_mouse(self) -> None:
        probe = self.panel.add(Probe((0, 0, 10, 10)))
        other = self.ui.add(Probe((20, 0, 10, 10)))
        self.panel.show(False)
        self.assertEqual((self.panel.shown, probe.visible, other.visible), (False, False, True))
        self.ui.handle(Event("down", (5, 5), button=1))
        self.ui.handle(Event("up", (5, 5), button=1))
        self.assertEqual(probe.presses, 0)
        self.panel.show(True)
        self.ui.handle(Event("down", (5, 5), button=1))
        self.assertEqual(probe.presses, 1)

    def test_a_widget_added_while_the_panel_is_off_starts_hidden(self) -> None:
        self.panel.show(False)
        probe = self.panel.add(Probe((0, 0, 10, 10)))
        self.assertFalse(probe.visible)
        self.panel.show(True)
        self.assertTrue(probe.visible)

    def test_a_frame_of_a_panel_can_be_told_the_seconds_since_the_one_before_or_nothing(self) -> None:
        self.assertEqual((self.panel.update(), self.panel.update(0.25)), (None, None))


class Dragged(unittest.TestCase):
    def test_a_border_tells_where_its_line_is_dragged_to_while_the_left_button_is_held_on_it(self) -> None:
        ui, told = Ui(), []
        border = ui.add(Border((98, 10, 5, 80), told.append, "press: drag"))
        self.assertEqual((border.line(), border.help), (100, "press: drag"))
        # The mouse over it, and a press of the right button, move nothing.
        ui.handle(Event("move", (101, 20)))
        self.assertFalse(ui.handle(Event("down", (101, 20), button=3)))
        ui.handle(Event("up", (101, 20), button=3))
        # The line keeps as far from the mouse as it was pressed, also when the mouse has left the strip.
        ui.handle(Event("down", (101, 20), button=1))
        for x in (131, 60):
            ui.handle(Event("move", (x, 300)))
        ui.handle(Event("up", (60, 300), button=1))
        ui.handle(Event("move", (100, 20)))
        self.assertEqual(told, [130, 59])


if __name__ == "__main__":
    unittest.main()
