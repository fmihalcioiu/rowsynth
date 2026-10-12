"""Tests for the layout of the GUI's frame and the sums of a window that is dragged, worked out with no window."""
from __future__ import annotations

import unittest

import numpy as np

from rowsynth.gui import theme
from rowsynth.gui.layout import (BORDER_REACH, CORNER_REACH, EDGE_REACH, LEAST_MIDDLE, LEAST_SIDE, LEAST_WINDOW, MOVE,
                                 POINTERS, TITLE_H, borders, dragged, edge, filled, frame, grip, pulled)


class Frame(unittest.TestCase):
    def test_at_the_art_size_it_is_drawn_for_the_parts_are_where_they_were_drawn(self) -> None:
        self.assertEqual(frame(theme.DESIGN), {
            "title": (0, 0, 960, 0),
            "top": (0, 0, 960, 22),
            "left": (0, 22, 124, 504),
            "middle": (124, 22, 662, 504),
            "right": (786, 22, 174, 504),
            "status": (0, 526, 960, 14),
        })

    def test_at_the_least_art_size_only_the_height_between_the_bars_is_less(self) -> None:
        self.assertEqual(theme.MIN_ART, (960, 480))
        self.assertEqual(frame(theme.MIN_ART), {
            "title": (0, 0, 960, 0),
            "top": (0, 0, 960, 22),
            "left": (0, 22, 124, 444),
            "middle": (124, 22, 662, 444),
            "right": (786, 22, 174, 444),
            "status": (0, 466, 960, 14),
        })

    def test_a_title_row_stands_over_the_top_bar_and_takes_its_height_from_the_columns(self) -> None:
        self.assertEqual(TITLE_H, 14)
        self.assertEqual(frame(theme.DESIGN, TITLE_H), {
            "title": (0, 0, 960, 14),
            "top": (0, 14, 960, 22),
            "left": (0, 36, 124, 490),
            "middle": (124, 36, 662, 490),
            "right": (786, 36, 174, 490),
            "status": (0, 526, 960, 14),
        })

    def test_the_side_panels_are_as_wide_as_they_are_given_and_the_middle_has_the_rest(self) -> None:
        parts = frame(theme.DESIGN, TITLE_H, 200, 100)
        self.assertEqual([parts[part] for part in ("left", "middle", "right")],
                         [(0, 36, 200, 490), (200, 36, 660, 490), (860, 36, 100, 490)])
        # One that is given as None is as wide as the theme has it.
        self.assertEqual(frame(theme.DESIGN, 0, None, 100)["left"][2], theme.LEFT_W)

    def test_the_parts_cover_the_art_surface_with_no_gap_and_no_overlap(self) -> None:
        # The last four are too small for the bars and the panels to be whole.
        for width, height in ((960, 540), (960, 480), (1280, 720), (1920, 1080), (961, 547), (640, 360), (200, 30),
                              (90, 400), (1, 1)):
            for more in ((), (TITLE_H,), (TITLE_H, 300, 400)):
                with self.subTest(size=(width, height), more=more):
                    covered = np.zeros((height, width), dtype=int)
                    for x, y, w, h in frame((width, height), *more).values():
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


class Borders(unittest.TestCase):
    def test_a_border_is_taken_by_the_strip_around_its_line_and_dragged_it_sets_the_width_of_its_panel(self) -> None:
        self.assertEqual(BORDER_REACH, 2)
        # The line of the left panel is its last pixel, 123, and that of the right panel its first, 786.
        self.assertEqual(borders(frame(theme.DESIGN, TITLE_H)), {"left": (121, 36, 5, 490), "right": (784, 36, 5, 490)})
        self.assertEqual(pulled(960, 124, 174, "left", 199), (200, 174))
        self.assertEqual(pulled(960, 124, 174, "right", 700), (124, 260))
        # Put where it is, a border changes nothing.
        self.assertEqual((pulled(960, 124, 174, "left", 123), pulled(960, 124, 174, "right", 786)),
                         ((124, 174), (124, 174)))

    def test_each_column_keeps_a_least_width(self) -> None:
        self.assertEqual((LEAST_SIDE, LEAST_MIDDLE), (72, 320))
        # A side panel is no narrower than its least, and no wider than what leaves the middle its least.
        self.assertEqual([pulled(960, 124, 174, "left", x) for x in (-50, 10, 900)],
                         [(72, 174), (72, 174), (960 - 174 - 320, 174)])
        self.assertEqual([pulled(960, 124, 174, "right", x) for x in (2000, 950, 0)],
                         [(124, 72), (124, 72), (124, 960 - 124 - 320)])
        # On a surface with no room for both, the panel keeps its own least width.
        self.assertEqual(pulled(400, 124, 174, "left", 200), (72, 174))


class Dragging(unittest.TestCase):
    SIZE = (800, 600)

    def test_an_edge_takes_the_mouse_a_few_pixels_in_and_a_corner_further_along_its_two_edges(self) -> None:
        self.assertEqual((EDGE_REACH, CORNER_REACH), (4, 12))
        self.assertEqual([grip(self.SIZE, pos) for pos in ((400, 0), (400, 3), (400, 596), (0, 300), (3, 300),
                                                           (799, 300), (796, 300))],
                         ["n", "n", "s", "w", "w", "e", "e"])
        self.assertEqual([grip(self.SIZE, pos) for pos in ((0, 0), (11, 3), (3, 11), (799, 0), (788, 2), (0, 599),
                                                           (2, 588), (799, 599), (797, 590))],
                         ["nw", "nw", "nw", "ne", "ne", "sw", "sw", "se", "se"])
        # Past the reach of the corner it is the edge again.
        self.assertEqual([grip(self.SIZE, pos) for pos in ((12, 3), (3, 12), (787, 2))], ["n", "w", "n"])
        # In from the edges it is nothing, and so it is off the window.
        self.assertEqual([grip(self.SIZE, pos) for pos in ((400, 4), (4, 300), (795, 300), (400, 595), (11, 11),
                                                           (-1, 300), (800, 300), (400, -1), (400, 600))],
                         [""] * 9)
        # Each has the pointer that shows which way it goes.
        self.assertEqual(POINTERS, {"n": "ns", "s": "ns", "w": "ew", "e": "ew", "nw": "nwse", "se": "nwse",
                                    "ne": "nesw", "sw": "nesw"})

    def test_a_move_takes_the_whole_window_and_an_edge_goes_with_the_mouse_while_the_others_stay(self) -> None:
        rect = (100, 50, 800, 600)
        self.assertEqual([dragged(rect, MOVE, (30, -20)), dragged(rect, MOVE, (-500, 0))],
                         [(130, 30, 800, 600), (-400, 50, 800, 600)])
        self.assertEqual([dragged(rect, edge, (30, -20)) for edge in ("e", "w", "s", "n")],
                         [(100, 50, 830, 600), (130, 50, 770, 600), (100, 50, 800, 580), (100, 30, 800, 620)])
        self.assertEqual([dragged(rect, corner, (30, -20)) for corner in ("se", "nw", "ne", "sw")],
                         [(100, 50, 830, 580), (130, 30, 770, 620), (100, 30, 830, 620), (130, 50, 770, 580)])

    def test_a_window_is_never_dragged_smaller_than_the_least_and_the_edge_stops_where_it_would_be(self) -> None:
        self.assertEqual(LEAST_WINDOW, (640, 360))
        rect = (100, 50, 800, 600)
        self.assertEqual([dragged(rect, "se", (-500, -500)), dragged(rect, "nw", (500, 500))],
                         [(100, 50, 640, 360), (260, 290, 640, 360)])
        # The least can be another.
        self.assertEqual(dragged(rect, "w", (790, 0), (20, 20)), (880, 50, 20, 600))

    def test_a_window_fills_the_desktop_and_is_put_back_where_it_stood(self) -> None:
        stood, work = (100, 50, 800, 600), (0, 0, 1920, 1032)
        self.assertEqual(filled(stood, None, work), (work, stood))
        self.assertEqual(filled(work, stood, work), (stood, None))


if __name__ == "__main__":
    unittest.main()
