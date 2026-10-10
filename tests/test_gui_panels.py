"""Tests for rowsynth.gui.panels: a panel's own widgets, and taking it off the window and bringing it back."""
from __future__ import annotations

import unittest

from rowsynth.gui.panels import Panel
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


if __name__ == "__main__":
    unittest.main()
