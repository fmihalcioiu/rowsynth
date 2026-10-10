"""Tests for the layout of the GUI's frame, which is worked out with no window."""
from __future__ import annotations

import unittest

import numpy as np

from rowsynth.gui import theme
from rowsynth.gui.layout import edge, frame


class Frame(unittest.TestCase):
    def test_at_the_art_size_it_is_drawn_for_the_parts_are_where_they_were_drawn(self) -> None:
        self.assertEqual(frame(theme.DESIGN), {
            "top": (0, 0, 960, 22),
            "left": (0, 22, 124, 504),
            "middle": (124, 22, 662, 504),
            "right": (786, 22, 174, 504),
            "status": (0, 526, 960, 14),
        })

    def test_at_the_least_art_size_only_the_height_between_the_bars_is_less(self) -> None:
        self.assertEqual(theme.MIN_ART, (960, 480))
        self.assertEqual(frame(theme.MIN_ART), {
            "top": (0, 0, 960, 22),
            "left": (0, 22, 124, 444),
            "middle": (124, 22, 662, 444),
            "right": (786, 22, 174, 444),
            "status": (0, 466, 960, 14),
        })

    def test_the_parts_cover_the_art_surface_with_no_gap_and_no_overlap(self) -> None:
        # The last four are too small for the bars and the panels to be whole.
        for width, height in ((960, 540), (960, 480), (1280, 720), (1920, 1080), (961, 547), (640, 360), (200, 30),
                              (90, 400), (1, 1)):
            with self.subTest(size=(width, height)):
                covered = np.zeros((height, width), dtype=int)
                for x, y, w, h in frame((width, height)).values():
                    self.assertTrue(w >= 0 and h >= 0, (w, h))
                    covered[y:y + h, x:x + w] += 1
                self.assertTrue((covered == 1).all())

    def test_a_larger_art_surface_gives_its_room_to_the_middle(self) -> None:
        parts = frame((1280, 720))
        self.assertEqual((parts["top"][3], parts["status"][3]), (theme.TOP_H, theme.STATUS_H))
        self.assertEqual((parts["left"][2], parts["right"][2]), (theme.LEFT_W, theme.RIGHT_W))
        self.assertEqual(parts["middle"], (124, 22, 1280 - 124 - 174, 720 - 22 - 14))
        self.assertEqual(parts["right"][0], 1280 - 174)

    def test_an_edge_is_the_strip_one_pixel_thick_inside_a_side(self) -> None:
        rect = (10, 20, 30, 40)
        self.assertEqual([edge(rect, side) for side in ("top", "bottom", "left", "right")],
                         [(10, 20, 30, 1), (10, 59, 30, 1), (10, 20, 1, 40), (39, 20, 1, 40)])


if __name__ == "__main__":
    unittest.main()
