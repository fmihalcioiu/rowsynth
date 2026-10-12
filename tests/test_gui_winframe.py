"""What Windows is told of a window that draws its own title row: which part moves it, which sizes it."""
from __future__ import annotations

import ctypes
import sys
import unittest
from unittest import mock

from rowsynth.gui import layout
from rowsynth.gui.winframe import CAPTION, CLIENT, EDGES, Frame, _Sizes, _signed, zone

SIZE = (1920, 984)


class Zones(unittest.TestCase):
    def test_the_title_row_moves_the_window_and_the_rest_of_it_is_the_window_s_own(self) -> None:
        self.assertEqual((zone(SIZE, (300, 14), "caption", False), zone(SIZE, (300, 400), "", False)),
                         (CAPTION, CLIENT))

    def test_an_edge_or_a_corner_sizes_the_window_also_where_the_title_row_is(self) -> None:
        reach = layout.EDGE_REACH
        found = [zone(SIZE, pos, what, False) for pos, what in (
            ((0, 400), ""), ((SIZE[0] - 1, 400), ""), ((300, SIZE[1] - 1), ""), ((300, reach - 1), "caption"),
            ((0, 0), "caption"), ((SIZE[0] - 1, SIZE[1] - 1), ""))]
        self.assertEqual(found, [EDGES["w"], EDGES["e"], EDGES["s"], EDGES["n"], EDGES["nw"], EDGES["se"]])
        # Just inside the reach of the upper edge, the title row has the mouse.
        self.assertEqual(zone(SIZE, (300, reach), "caption", False), CAPTION)

    def test_a_button_of_the_row_is_the_window_s_own_even_in_the_corner_it_stands_in(self) -> None:
        self.assertEqual(zone(SIZE, (SIZE[0] - 2, 2), "button", False), CLIENT)

    def test_a_window_that_fills_its_display_has_no_edges_and_is_still_moved_by_its_row(self) -> None:
        self.assertEqual((zone(SIZE, (0, 400), "", True), zone(SIZE, (300, 2), "caption", True)), (CLIENT, CAPTION))

    def test_a_place_of_the_desktop_is_read_out_of_a_message_with_its_sign(self) -> None:
        self.assertEqual((_signed(0x0010), _signed(0xFFF0), _signed(0x12348000)), (16, -16, -32768))

    def test_the_system_is_told_the_least_size_of_the_window_after_the_window_as_it_was_has_answered(self) -> None:
        sizes = _Sizes()
        sizes.least.x, sizes.least.y, sizes.most.x = 136, 39, 4000
        # A hold with no window behind it: what the window was before stands in for the system.
        frame = Frame.__new__(Frame)
        frame._user, frame._before, frame.least = mock.Mock(), 7, (1280, 800)
        frame._user.CallWindowProcW.return_value = 0
        at = ctypes.addressof(sizes)
        self.assertEqual(frame._told(5, 0x0024, 0, at), 0)
        frame._user.CallWindowProcW.assert_called_once_with(7, 5, 0x0024, 0, at)
        self.assertEqual(((sizes.least.x, sizes.least.y), sizes.most.x), ((1280, 800), 4000))
        # A least size the window had that is larger stays.
        frame.least = (640, 900)
        frame._told(5, 0x0024, 0, at)
        self.assertEqual((sizes.least.x, sizes.least.y), (1280, 900))

    @unittest.skipIf(sys.platform == "win32", "only where the system is not Windows")
    def test_anywhere_but_on_windows_there_is_no_such_hold_on_a_window(self) -> None:
        with self.assertRaises(OSError):
            Frame(0, lambda pos: "")


if __name__ == "__main__":
    unittest.main()
