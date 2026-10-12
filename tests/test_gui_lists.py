"""Tests for rowsynth.gui.lists: how far a list is scrolled, where its rows go, and the widget that shows them."""
from __future__ import annotations

import dataclasses
import os
import unittest
from collections.abc import Hashable, Iterable

from rowsynth.gui import layout, sprites, theme
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.font import Font, fit
from rowsynth.gui.lists import DIMMED, List, Row, Scroll, lay, mix
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
ROW, GAP, BAR, PAD = theme.LIST_ROW_H, theme.LIST_GAP, theme.LIST_BAR_W, theme.LIST_PAD

#: Three headings with the rows under them, as the contents of a project are listed.
TREE = (
    Row("SONGS", "SONGS", head=True, colour=theme.SONG),
    Row("tune", "tune"),
    Row("SOUNDS", "SOUNDS", head=True, colour=theme.SOUND, tag="2"),
    Row("coin", "coin"),
    Row("jump", "jump"),
    Row("INSTRUMENTS", "INSTRUMENTS", head=True, colour=theme.INSTRUMENT),
    Row("lead", "lead"),
)
#: Ten rows with no heading, known by their numbers.
FLAT = tuple(Row(number, f"row {number}") for number in range(10))
#: A rectangle in which every row of TREE fits, and one that shows three rows.
TALL = (10, 20, 80, 200)
SHORT = (10, 20, 80, 3 * ROW)


def made(rect: layout.Rect, rows: Iterable[Row], picked: Hashable = None) -> tuple[Ui, List, list[Hashable]]:
    """A Ui with one list in it, the list, and what the list has told so far."""
    ui, told = Ui(), []
    return ui, ui.add(List(rect, rows, picked, told.append)), told


def click(ui: Ui, pos: tuple[int, int], button: int = 1) -> bool:
    """Press a mouse button at a position and let it go there. Whether the press was used."""
    used = ui.handle(Event("down", pos, button=button))
    ui.handle(Event("up", pos, button=button))
    return used


def seen(rows: List) -> list[Hashable]:
    """The keys of the rows that are in view, from the top down."""
    x, y, _, h = rows.rect
    keys: list[Hashable] = []
    for down in range(y, y + h):
        row = rows.at((x, down))
        if row is not None and row.key not in keys:
            keys.append(row.key)
    return keys


def spot(rows: List, key: Hashable) -> tuple[int, int]:
    """The top-left corner of the row of a key, as the list is scrolled now."""
    x, y, _, h = rows.rect
    return next((x, down) for down in range(y, y + h) if getattr(rows.at((x, down)), "key", None) == key)


def press(ui: Ui, key: str) -> bool:
    """Press a key. Whether it was used."""
    return ui.handle(Event("key", key=key))


class Scrolling(unittest.TestCase):
    def test_scrolling_to_an_offset_stops_at_both_ends_and_content_shorter_than_the_view_stays_at_0(self) -> None:
        scroll = Scroll(content=100, view=40)
        self.assertEqual((scroll.offset, scroll.most), (0, 60))
        for offset, kept in ((25, 25), (-5, 0), (60, 60), (61, 60)):
            scroll.to(offset)
            self.assertEqual(scroll.offset, kept)
        # Less content than there was: the same offset, asked for again, is brought back inside.
        scroll.content = 70
        scroll.to(scroll.offset)
        self.assertEqual(scroll.offset, 30)
        short = Scroll(content=30, view=40)
        short.to(10)
        self.assertEqual((short.offset, short.most), (0, 0))

    def test_scrolling_by_rows_steps_a_row_at_a_time_and_stops_at_the_ends(self) -> None:
        scroll = Scroll(content=100, view=40, row=10)
        for rows, offset in ((3, 30), (-1, 20), (9, 60), (-20, 0)):
            scroll.by(rows)
            self.assertEqual(scroll.offset, offset)
        self.assertEqual(Scroll().row, ROW)
        short = Scroll(content=30, view=40, row=10)
        short.by(1)
        self.assertEqual(short.offset, 0)

    def test_showing_a_stretch_moves_the_least_that_brings_it_into_view(self) -> None:
        # The view shows the content from 30 to just before 70.
        scroll = Scroll(content=100, view=40, offset=30)
        for top, bottom, offset in ((40, 50, 30), (30, 70, 30), (20, 30, 20), (75, 85, 45), (95, 120, 60)):
            scroll.show(top, bottom)
            self.assertEqual(scroll.offset, offset)
        # A stretch longer than the view is shown from its top, from below it and from above it.
        scroll.show(0, 90)
        self.assertEqual(scroll.offset, 0)
        scroll.show(30, 90)
        self.assertEqual(scroll.offset, 30)
        short = Scroll(content=30, view=40)
        short.show(20, 30)
        self.assertEqual(short.offset, 0)

    def test_the_thumb_is_to_the_track_what_the_view_is_to_the_content_and_there_is_none_when_all_fits(self) -> None:
        scroll = Scroll(content=200, view=50)
        self.assertEqual(scroll.thumb(100), (0, 25))
        for offset, start in ((60, 30), (150, 75)):
            scroll.to(offset)
            self.assertEqual(scroll.thumb(100), (start, 25))
        self.assertEqual(scroll.thumb(40), (30, 10))
        self.assertEqual([Scroll(content=size, view=50).thumb(100) for size in (50, 0)], [None, None])

    def test_the_thumb_is_never_shorter_than_its_least_length_and_still_reaches_both_ends(self) -> None:
        least = theme.LIST_THUMB_MIN
        scroll = Scroll(content=5000, view=50)
        self.assertEqual(scroll.thumb(100), (0, least))
        scroll.to(2475)
        self.assertEqual(scroll.thumb(100), ((100 - least) // 2, least))
        scroll.to(4950)
        self.assertEqual(scroll.thumb(100), (100 - least, least))
        # On a track with less room than that, it is the whole track.
        self.assertEqual(scroll.thumb(least - 3), (0, least - 3))

    def test_a_drag_of_the_thumb_scrolls_by_what_its_way_along_the_track_is_of_the_content(self) -> None:
        # A thumb of 25 on a track of 100 has 75 to go for 150 of content: 2 for each pixel.
        scroll = Scroll(content=200, view=50)
        for pixels, offset in ((30, 60), (-10, 40), (500, 150), (-500, 0)):
            scroll.drag(pixels, 100)
            self.assertEqual(scroll.offset, offset)
        # Nothing moves when all fits, nor when the thumb is the whole track.
        fits = Scroll(content=50, view=50)
        fits.drag(10, 100)
        scroll.drag(3, theme.LIST_THUMB_MIN)
        self.assertEqual((fits.offset, scroll.offset), (0, 0))


class Rows(unittest.TestCase):
    def test_a_row_cannot_be_changed_and_what_it_is_not_told_is_plain(self) -> None:
        row = Row("key", "name")
        self.assertEqual((row.head, row.colour, row.tag, row.indent), (False, theme.DIM, "", 0))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            row.text = "other"

    def test_rows_are_laid_from_the_top_with_air_above_a_later_heading_and_folded_ones_left_out(self) -> None:
        def laid(*folded: str) -> list[tuple[Hashable, int, theme.Colour | None]]:
            return [(place.row.key, place.top, place.colour) for place in lay(TREE, folded)]

        song, sound, instrument = theme.SONG, theme.SOUND, theme.INSTRUMENT
        self.assertEqual((ROW, GAP), (12, 8))
        self.assertEqual(laid(), [("SONGS", 0, song), ("tune", 12, song), ("SOUNDS", 32, sound), ("coin", 44, sound),
                                  ("jump", 56, sound), ("INSTRUMENTS", 76, instrument), ("lead", 88, instrument)])
        self.assertEqual(laid("SOUNDS", "SONGS"), [("SONGS", 0, song), ("SOUNDS", 20, sound),
                                                   ("INSTRUMENTS", 40, instrument), ("lead", 52, instrument)])
        # Rows with no heading above them have no colour of a heading.
        self.assertEqual([tuple(place) for place in lay(FLAT[:2])], [(FLAT[0], 0, None), (FLAT[1], 12, None)])
        self.assertEqual(lay(()), [])

    def test_one_colour_is_mixed_into_another_by_a_strength_that_is_held_between_0_and_1(self) -> None:
        under, over = (0, 100, 200), (100, 0, 200)
        self.assertEqual([mix(under, over, strength) for strength in (0, 0.25, 0.5, 1, -1, 3)],
                         [under, (25, 75, 200), (50, 50, 200), over, under, over])


class Picking(unittest.TestCase):
    def test_a_click_on_a_row_picks_it_and_tells_its_key(self) -> None:
        ui, rows, told = made(TALL, TREE)
        self.assertEqual((rows.picked, rows.rows), (None, TREE))
        self.assertTrue(click(ui, spot(rows, "coin")))
        self.assertEqual((rows.picked, told), ("coin", ["coin"]))
        click(ui, spot(rows, "lead"))
        self.assertEqual((rows.picked, told), ("lead", ["coin", "lead"]))
        # A press where no row is is used and picks nothing. One of the right button is not used.
        self.assertTrue(click(ui, (10, 219)))
        self.assertFalse(click(ui, spot(rows, "tune"), button=3))
        self.assertEqual((rows.picked, told), ("lead", ["coin", "lead"]))
        # A list is made with a row picked, when there is such a row, and can be made with no one to tell.
        self.assertEqual([List(TALL, TREE, key).picked for key in ("jump", "SOUNDS", "gone")], ["jump", None, None])
        mute = ui.add(List(TALL, TREE))
        click(ui, spot(mute, "jump"))
        self.assertEqual((mute.picked, told), ("jump", ["coin", "lead"]))

    def test_the_picked_row_clicked_again_tells_again(self) -> None:
        ui, rows, told = made(TALL, TREE, "coin")
        click(ui, spot(rows, "coin"))
        click(ui, spot(rows, "coin"))
        self.assertEqual((rows.picked, told), ("coin", ["coin", "coin"]))

    def test_a_heading_folds_its_rows_away_where_no_click_and_no_key_reaches_them_and_brings_them_back(self) -> None:
        ui, rows, told = made(TALL, TREE, "coin")
        self.assertEqual(seen(rows), ["SONGS", "tune", "SOUNDS", "coin", "jump", "INSTRUMENTS", "lead"])
        self.assertTrue(click(ui, spot(rows, "SOUNDS")))
        self.assertEqual(seen(rows), ["SONGS", "tune", "SOUNDS", "INSTRUMENTS", "lead"])
        # The picked row is folded away and stays picked. Nothing is told, and a heading is never picked.
        self.assertEqual((rows.picked, told), ("coin", []))
        # The keys go from where the picked row would stand, past the rows that are folded away.
        for key, picked in (("down", "lead"), ("up", "tune"), ("down", "lead")):
            self.assertTrue(press(ui, key))
            self.assertEqual(rows.picked, picked)
        click(ui, spot(rows, "SOUNDS"))
        self.assertEqual(seen(rows), ["SONGS", "tune", "SOUNDS", "coin", "jump", "INSTRUMENTS", "lead"])
        press(ui, "up")
        self.assertEqual((rows.picked, told), ("jump", ["lead", "tune", "lead", "jump"]))

    def test_the_wheel_scrolls_three_rows_a_step_and_stops_at_the_ends(self) -> None:
        ui, rows, told = made(SHORT, FLAT)
        for wheel, offset in ((-1, 3 * ROW), (-1, 6 * ROW), (-1, 7 * ROW), (1, 4 * ROW), (2, 0), (1, 0)):
            self.assertTrue(ui.handle(Event("wheel", (15, 25), wheel=wheel)))
            self.assertEqual(rows.scroll.offset, offset)
        self.assertFalse(ui.handle(Event("wheel", (5, 25), wheel=-1)))
        self.assertEqual((rows.scroll.offset, rows.picked, told), (0, None, []))

    def test_a_list_goes_one_way_so_the_wheel_with_shift_or_ctrl_is_not_used_and_scrolls_nothing(self) -> None:
        ui, rows, _ = made(SHORT, FLAT)
        for mods in ({"shift"}, {"ctrl"}, {"ctrl", "shift"}):
            self.assertFalse(ui.handle(Event("wheel", (15, 25), wheel=-1, mods=frozenset(mods))))
        self.assertEqual(rows.scroll.offset, 0)
        # Alt is none of the keys the wheel is read by.
        self.assertTrue(ui.handle(Event("wheel", (15, 25), wheel=-1, mods=frozenset({"alt"}))))
        self.assertEqual(rows.scroll.offset, 3 * ROW)

    def test_a_row_that_is_a_plus_is_told_of_when_it_is_pressed_and_is_picked_by_no_press_and_no_key(self) -> None:
        plus = Row(("new", "SOUNDS"), "new", plus=True)
        ui, rows, told = made(TALL, (*TREE[:5], plus, *TREE[5:]), "jump")
        added: list[Hashable] = []
        rows.added = added.append
        self.assertTrue(click(ui, spot(rows, plus.key)))
        self.assertEqual((added, rows.picked, told), ([plus.key], "jump", []))
        # The keys go from the row before it to the row after it, and back.
        for key, picked in (("down", "lead"), ("up", "jump"), ("end", "lead")):
            press(ui, key)
            self.assertEqual(rows.picked, picked)
        # The right button on it is not used, and it cannot be the picked row: it is folded away with the others.
        asked: list[tuple] = []
        rows.asked = lambda *what: asked.append(what)
        self.assertFalse(click(ui, spot(rows, plus.key), button=3))
        rows.picked = plus.key
        rows.rows = rows.rows
        self.assertEqual((asked, rows.picked, added), ([], None, [plus.key]))
        click(ui, spot(rows, "SOUNDS"))
        self.assertEqual((seen(rows), rows.row_rect(plus.key)),
                         (["SONGS", "tune", "SOUNDS", "INSTRUMENTS", "lead"], None))
        # With no owner to tell, a press on it does nothing.
        other, bare, said = made(TALL, (plus,))
        self.assertTrue(click(other, (15, 25)))
        self.assertEqual((bare.picked, said), (None, []))

    def test_the_rectangle_of_a_row_is_all_of_its_row_as_the_list_is_scrolled_in_view_or_not(self) -> None:
        ui, rows, _ = made(SHORT, FLAT)
        # The bar takes its room from the rows.
        self.assertEqual([rows.row_rect(key) for key in (0, 2, 9, 10)],
                         [(10, 20, 80 - BAR, ROW), (10, 20 + 2 * ROW, 80 - BAR, ROW), (10, 20 + 9 * ROW, 80 - BAR, ROW),
                          None])
        rows.scroll.to(ROW // 2)
        self.assertEqual(rows.row_rect(0), (10, 20 - ROW // 2, 80 - BAR, ROW))
        # The name of a row that is a plus stands after the room of a square, also under no heading.
        rows.rows = (Row("more", "new", plus=True),)
        self.assertEqual((rows.row_rect("more"), rows.name_rect("more")),
                         ((10, 20, 80, ROW), (10 + PAD + theme.LIST_SQUARE + theme.LIST_SPACE, 20,
                                              80 - 2 * PAD - theme.LIST_SQUARE - theme.LIST_SPACE, ROW)))

    def test_the_keys_move_the_pick_past_a_heading_and_stop_at_both_ends(self) -> None:
        ui, rows, told = made(TALL, TREE)
        # A press under its last row gives the list the focus, and picks nothing.
        click(ui, (15, 215))
        for key, picked in (("down", "tune"), ("down", "coin"), ("up", "tune"), ("up", "tune"), ("end", "lead"),
                            ("down", "lead"), ("end", "lead"), ("home", "tune"), ("home", "tune")):
            self.assertTrue(press(ui, key))
            self.assertEqual(rows.picked, picked)
        # Only a move is told.
        self.assertEqual(told, ["tune", "coin", "tune", "lead", "tune"])
        # With nothing picked, up goes to the last row.
        rows.picked = None
        press(ui, "up")
        self.assertEqual(rows.picked, "lead")
        # Other keys are not used, nor these with ctrl held, nor any by a list with no row to pick.
        self.assertFalse(press(ui, "space"))
        self.assertFalse(ui.handle(Event("key", key="up", mods=frozenset({"ctrl"}))))
        rows.rows = TREE[:1]
        self.assertFalse(press(ui, "down"))
        self.assertEqual((rows.picked, told), (None, ["tune", "coin", "tune", "lead", "tune", "lead"]))

    def test_a_row_picked_by_key_is_scrolled_into_view_by_the_least_that_shows_it(self) -> None:
        ui, rows, _ = made(SHORT, FLAT)
        # The keys go to the list that was last pressed.
        click(ui, spot(rows, 0))
        for key, picked, offset in (("end", 9, 7 * ROW), ("home", 0, 0), ("down", 1, 0), ("down", 2, 0),
                                    ("down", 3, ROW), ("end", 9, 7 * ROW), ("up", 8, 7 * ROW), ("up", 7, 7 * ROW),
                                    ("up", 6, 6 * ROW)):
            press(ui, key)
            self.assertEqual((rows.picked, rows.scroll.offset), (picked, offset))
            self.assertIn(picked, seen(rows))

    def test_the_thumb_of_the_bar_is_dragged_from_where_it_was_taken_hold_of(self) -> None:
        # Ten rows where five fit: the thumb is half of the track, and its whole way is the other half.
        ui, rows, told = made((10, 20, 80, 5 * ROW), FLAT)
        start, length = rows.scroll.thumb(5 * ROW)
        way, most = 5 * ROW - length, rows.scroll.most
        self.assertEqual((start, length, most), (0, 30, 5 * ROW))
        self.assertTrue(ui.handle(Event("down", (89, 22), button=1)))
        for pos, offset in (((89, 22 + way // 2), most // 2), ((0, 22 + way), most), ((0, 500), most),
                            ((300, 22 + way // 2), most // 2), ((89, 0), 0)):
            ui.handle(Event("move", pos))
            self.assertEqual(rows.scroll.offset, offset)
        # Once it is let go, the mouse moves nothing. No row was picked on the way.
        ui.handle(Event("move", (89, 22 + way // 2)))
        ui.handle(Event("up", (89, 22 + way // 2), button=1))
        ui.handle(Event("move", (89, 22 + way)))
        self.assertEqual((rows.scroll.offset, rows.picked, told), (most // 2, None, []))

    def test_a_press_on_the_track_above_or_below_the_thumb_scrolls_by_what_the_view_shows(self) -> None:
        ui, rows, told = made((10, 20, 80, 5 * ROW), FLAT + tuple(Row(number, "more") for number in range(10, 20)))
        top, bottom = (86, 20), (89, 20 + 5 * ROW - 1)
        for pos, offset in ((bottom, 5 * ROW), (bottom, 10 * ROW), (top, 5 * ROW)):
            self.assertTrue(click(ui, pos))
            self.assertEqual(rows.scroll.offset, offset)
        # Just left of the bar is a row.
        click(ui, (85, 20))
        self.assertEqual((rows.scroll.offset, rows.picked, told), (5 * ROW, 5, [5]))

    def test_rows_that_are_replaced_keep_the_pick_while_its_key_is_there_and_headings_stay_folded(self) -> None:
        ui, rows, told = made(TALL, TREE, "jump")
        click(ui, spot(rows, "INSTRUMENTS"))
        rows.rows = TREE[:3] + (Row("blip", "blip"),) + TREE[3:] + (Row("bass", "bass"),)
        self.assertEqual((rows.picked, seen(rows)),
                         ("jump", ["SONGS", "tune", "SOUNDS", "blip", "coin", "jump", "INSTRUMENTS"]))
        # A row that is gone is not picked any more, and nothing is told of it. A heading is not a row to pick.
        rows.rows = [row for row in TREE if row.key != "jump"] + [Row("jump", "JUMP", head=True)]
        self.assertEqual((rows.picked, told), (None, []))
        self.assertEqual(seen(rows), ["SONGS", "tune", "SOUNDS", "coin", "INSTRUMENTS", "jump"])

    def test_rows_that_are_replaced_by_fewer_than_the_offset_leave_it_inside_what_there_is(self) -> None:
        _, rows, _ = made(SHORT, FLAT)
        rows.scroll.to(1000)
        self.assertEqual((rows.scroll.offset, seen(rows)), (7 * ROW, [7, 8, 9]))
        rows.rows = FLAT[:4]
        self.assertEqual((rows.scroll.offset, seen(rows)), (ROW, [1, 2, 3]))
        rows.rows = FLAT[:2]
        self.assertEqual((rows.scroll.offset, seen(rows)), (0, [0, 1]))
        rows.rows = ()
        self.assertEqual((rows.scroll.offset, seen(rows)), (0, []))
        # A rectangle that is made higher, so that less is left to scroll, does the same.
        rows.rows = FLAT[:4]
        rows.rect = (10, 20, 80, 2 * ROW)
        self.assertEqual(seen(rows), [0, 1])
        rows.scroll.to(1000)
        self.assertEqual((rows.scroll.offset, seen(rows)), (2 * ROW, [2, 3]))
        rows.rect = SHORT
        self.assertEqual((seen(rows), rows.scroll.offset), ([1, 2, 3], ROW))

    def test_which_row_is_under_a_position_also_when_the_list_is_scrolled(self) -> None:
        def keys(rows: List, *places: tuple[int, int]) -> list[Hashable]:
            return [getattr(rows.at(pos), "key", None) for pos in places]

        # A row is its own pixels across the whole list. The air above a later heading is no row.
        rows = List(TALL, TREE)
        self.assertEqual(keys(rows, (10, 20), (89, 31), (10, 32), (10, 44), (10, 51), (10, 52), (10, 119), (10, 120)),
                         ["SONGS", "SONGS", "tune", None, None, "SOUNDS", "lead", None])
        self.assertEqual(keys(rows, (9, 20), (90, 20), (10, 19), (10, 220)), [None, None, None, None])
        # Scrolled by half a row, with the bar at the right, which is no row either.
        rows = List(SHORT, FLAT)
        rows.scroll.to(ROW // 2)
        self.assertEqual(keys(rows, (10, 20), (10, 25), (10, 26), (10, 55), (85, 25), (86, 25)),
                         [0, 0, 1, 3, 0, None])

    def test_a_press_of_the_right_button_on_a_row_tells_its_key_and_where_and_picks_nothing(self) -> None:
        ui, rows, told = made(TALL, TREE, "coin")
        asked: list[tuple] = []
        rows.asked = lambda *what: asked.append(what)
        at = spot(rows, "jump")
        self.assertTrue(click(ui, at, button=3))
        self.assertEqual((asked, rows.picked, told), ([("jump", at)], "coin", []))
        # A heading is not asked about, and neither is a place with no row: the press is not used there.
        self.assertEqual([click(ui, spot(rows, "SOUNDS"), button=3), click(ui, (10, 219), button=3)], [False, False])
        self.assertEqual((len(asked), seen(rows)), (1, [row.key for row in TREE]))

    def test_the_rectangle_of_a_name_is_where_the_text_of_its_row_is_drawn_as_the_list_is_scrolled(self) -> None:
        # After the pad and the square of its heading's colour, to the pad at the other side or to the tag.
        ui, rows, _ = made(TALL, (*TREE, Row("deep", "deep", indent=2, square=theme.SONG, tag="kick")))
        self.assertEqual([rows.name_rect(key) for key in ("tune", "SOUNDS", "deep")],
                         [(27, 32, 57, 12), (27, 52, 47, 12), (43, 120, 13, 12)])
        # A row that is folded away has none, and neither has a key no row has.
        click(ui, spot(rows, "SOUNDS"))
        self.assertEqual([rows.name_rect("coin"), rows.name_rect("gone")], [None, None])
        # A row with no heading over it and no colour of its own has no square. The bar takes its room.
        _, flat, _ = made(SHORT, FLAT)
        self.assertEqual(flat.name_rect(1), (16, 32, 64, 12))
        flat.scroll.by(1)
        self.assertEqual([flat.name_rect(1), flat.name_rect(0)], [(16, 20, 64, 12), (16, 8, 64, 12)])


@unittest.skipUnless(pygame, NEEDS)
class Drawn(unittest.TestCase):
    #: What the surface is filled with before a list is drawn: a colour the theme has not got.
    OUT = (1, 2, 3)
    #: Where a list that TREE fits in exactly is drawn.
    RECT = (10, 10, 100, 100)

    def setUp(self) -> None:
        self.surface = pygame.Surface((120, 120))
        self.canvas = Canvas(self.surface, Font(sprites.load("font")))
        self.ui = Ui()

    def draw(self) -> None:
        """Draw the widgets on a surface that has nothing on it."""
        self.surface.fill(self.OUT)
        self.ui.draw(self.canvas)

    def at(self, *pos: int) -> tuple[int, ...]:
        """The colour of one pixel."""
        return tuple(self.surface.get_at(pos))[:3]

    def colours(self, rect: layout.Rect) -> set[tuple[int, ...]]:
        """The colours a rectangle has in it."""
        x, y, w, h = rect
        return {self.at(x + column, y + row) for column in range(w) for row in range(h)}

    def ink(self, rect: layout.Rect, colour: theme.Colour) -> set[tuple[int, int]]:
        """Where a rectangle has a colour in it."""
        x, y, w, h = rect
        return {(x + column, y + row) for column in range(w) for row in range(h)
                if self.at(x + column, y + row) == colour}

    def band(self, rows: List, key: Hashable) -> layout.Rect:
        """The rectangle of the row of a key in a list with no bar."""
        return *spot(rows, key), rows.rect[2], ROW

    def test_a_row_that_is_a_plus_has_its_text_dim_and_nothing_where_its_square_would_be(self) -> None:
        rows = self.ui.add(List(self.RECT, (*TREE[:2], Row("more", "new", plus=True)), "more"))
        self.draw()
        # It is not the picked row, whatever the list was told, and the room before its text is left to a button.
        self.assertEqual((rows.picked, self.colours(self.band(rows, "more")), self.colours(self.band(rows, "tune"))),
                         (None, {theme.PANEL, theme.DIM}, {theme.PANEL, theme.SONG, theme.TEXT}))
        self.assertEqual(self.ink((10, 34, PAD + theme.LIST_SQUARE + theme.LIST_SPACE, ROW), theme.DIM), set())

    def test_the_picked_row_is_on_a_lit_background_and_the_row_under_the_mouse_on_one_lit_less(self) -> None:
        rows = self.ui.add(List(self.RECT, TREE, "tune"))
        self.draw()
        tune, coin = self.band(rows, "tune"), self.band(rows, "coin")
        self.assertEqual((tune, coin), ((10, 22, 100, 12), (10, 54, 100, 12)))
        self.assertEqual(self.colours(tune), {theme.LIT, theme.SONG, theme.TEXT})
        self.assertEqual(self.colours(coin), {theme.PANEL, theme.SOUND, theme.TEXT})
        # The picked row has a strip of its heading's colour down its left side.
        self.assertEqual((self.colours((10, 22, 2, 12)), self.at(12, 22), self.at(109, 33)),
                         ({theme.SONG}, theme.LIT, theme.LIT))
        self.ui.handle(Event("move", (50, 56)))
        self.draw()
        self.assertEqual(self.colours(coin), {theme.RAISED, theme.SOUND, theme.TEXT})
        self.assertEqual(self.colours(tune), {theme.LIT, theme.SONG, theme.TEXT})
        # The picked row stays as lit with the mouse on it, and a row the mouse has left is as it was.
        self.ui.handle(Event("move", (50, 24)))
        self.draw()
        self.assertEqual((self.at(109, 33), self.colours(coin)), (theme.LIT, {theme.PANEL, theme.SOUND, theme.TEXT}))

    def test_a_heading_is_in_its_colour_with_a_mark_that_shows_whether_it_is_folded_and_a_square_before_its_rows(
            self) -> None:
        def mark() -> tuple[int, int, int]:
            """How many pixels the mark before the first heading has, and how wide and how high it is."""
            self.draw()
            inked = self.ink((10 + PAD, 10, theme.LIST_SQUARE, ROW), theme.SONG)
            across, down = {x for x, _ in inked}, {y for _, y in inked}
            return len(inked), len(across), len(down)

        rows = self.ui.add(List(self.RECT, TREE))
        self.assertEqual(mark(), (9, 5, 3))
        self.assertEqual(self.colours(self.band(rows, "SONGS")), {theme.PANEL, theme.SONG})
        self.assertEqual(self.colours(self.band(rows, "INSTRUMENTS")), {theme.PANEL, theme.INSTRUMENT})
        # The square is as far in as the mark, and the two texts start at the same place.
        self.assertEqual(self.ink(self.band(rows, "coin"), theme.SOUND),
                         {(10 + PAD + column, 54 + 3 + row) for column in range(6) for row in range(6)})
        self.assertEqual([min(x for x, _ in self.ink(self.band(rows, key), colour))
                          for key, colour in (("SONGS", theme.SONG), ("tune", theme.TEXT))], [16, 27])
        self.assertEqual(self.colours(self.band(rows, "tune")), {theme.PANEL, theme.SONG, theme.TEXT})
        click(self.ui, spot(rows, "SONGS"))
        self.ui.handle(Event("move", (0, 0)))
        self.assertEqual(mark(), (9, 3, 5))
        # Rows with no heading above them have no square, and their text starts where the square would.
        rows.rows = FLAT[:2]
        self.draw()
        self.assertEqual(self.colours(self.band(rows, 0)), {theme.PANEL, theme.TEXT})
        self.assertEqual(min(x for x, _ in self.ink(self.band(rows, 0), theme.TEXT)), 16)

    def test_a_row_with_a_colour_of_its_own_has_its_square_and_its_strip_when_picked_in_it(self) -> None:
        rows = self.ui.add(List(self.RECT, [TREE[0], Row("own", "own", square=theme.SALMON),
                                            Row("other", "other", square=theme.MUSTARD)], "own"))
        self.draw()
        own, other = self.band(rows, "own"), self.band(rows, "other")
        self.assertEqual(self.colours(own), {theme.LIT, theme.SALMON, theme.TEXT})
        self.assertEqual(self.colours(other), {theme.PANEL, theme.MUSTARD, theme.TEXT})
        # The square is where any row has it, and the strip is down the left side of the picked row.
        self.assertEqual(self.ink(other, theme.MUSTARD),
                         {(10 + PAD + column, other[1] + 3 + row) for column in range(6) for row in range(6)})
        self.assertEqual(len(self.ink(own, theme.SALMON)), 6 * 6 + theme.LIST_STRIP * ROW)
        self.assertEqual(self.colours((10, own[1], theme.LIST_STRIP, ROW)), {theme.SALMON})
        # With no heading above it, it has its square all the same, and its text starts after it.
        rows.rows = [Row("own", "own", square=theme.SALMON)]
        self.draw()
        self.assertEqual(self.colours(self.band(rows, "own")), {theme.LIT, theme.SALMON, theme.TEXT})
        self.assertEqual(min(x for x, _ in self.ink(self.band(rows, "own"), theme.TEXT)), 27)

    def test_a_row_with_no_colour_of_its_own_has_its_headings_beside_one_that_has(self) -> None:
        self.assertEqual((Row("key", "name").square, Row("key", "name").dim), (None, False))
        rows = self.ui.add(List(self.RECT, [TREE[0], Row("own", "own", square=theme.SALMON), Row("plain", "plain"),
                                            TREE[2], Row("coin", "coin")], "plain"))
        self.draw()
        self.assertEqual(self.colours(self.band(rows, "plain")), {theme.LIT, theme.SONG, theme.TEXT})
        self.assertEqual(self.colours((10, self.band(rows, "plain")[1], theme.LIST_STRIP, ROW)), {theme.SONG})
        self.assertEqual(self.colours(self.band(rows, "own")), {theme.PANEL, theme.SALMON, theme.TEXT})
        self.assertEqual(self.colours(self.band(rows, "coin")), {theme.PANEL, theme.SOUND, theme.TEXT})
        # A heading has no square: one that is given a colour for it is in its own colour as before.
        rows.rows = [dataclasses.replace(TREE[0], square=theme.SALMON, dim=True), Row("tune", "tune")]
        self.draw()
        self.assertEqual(self.colours(self.band(rows, "SONGS")), {theme.PANEL, theme.SONG})
        self.assertEqual(self.colours(self.band(rows, "tune")), {theme.PANEL, theme.SONG, theme.TEXT})

    def test_a_dim_row_has_its_text_dim_its_tag_dimmer_and_its_square_mixed_towards_the_panel(self) -> None:
        rows = self.ui.add(List(self.RECT, [TREE[0], Row("own", "own", tag="12", square=theme.SALMON, dim=True),
                                            Row("plain", "plain", tag="12", dim=True),
                                            Row("clear", "clear", tag="12")], "own"))
        self.draw()
        own, plain, clear = (self.band(rows, key) for key in ("own", "plain", "clear"))
        faded = [mix(theme.PANEL, colour, DIMMED) for colour in (theme.SALMON, theme.SONG)]
        self.assertEqual(self.colours(own), {theme.LIT, faded[0], theme.DIM, theme.LINE})
        self.assertEqual(self.colours(plain), {theme.PANEL, faded[1], theme.DIM, theme.LINE})
        self.assertEqual(self.colours(clear), {theme.PANEL, theme.SONG, theme.TEXT, theme.DIM})
        # The square keeps its size, and the strip of a picked dim row is as dim as its square.
        self.assertEqual((len(self.ink(plain, faded[1])), len(self.ink(own, faded[0]))),
                         (6 * 6, 6 * 6 + theme.LIST_STRIP * ROW))
        # The name is where the name of any row is, and the tag where the tag of any row is.
        for band in (own, plain):
            name, tag = self.ink(band, theme.DIM), self.ink(band, theme.LINE)
            self.assertEqual(min(x for x, _ in name), 27)
            self.assertLess(max(x for x, _ in name), min(x for x, _ in tag))
            self.assertEqual({x for x, _ in tag}, {x for x, _ in self.ink(clear, theme.DIM)})

    def test_the_tag_is_at_the_right_end_and_a_name_too_long_for_the_room_before_it_is_cut(self) -> None:
        name = "a very long name of a thing"
        # 60 wide: 48 between the two sides, and 32 before a tag of two letters.
        rows = self.ui.add(List((10, 10, 60, 100), [Row("bare", name), Row("tagged", name, tag="12"),
                                                    Row("deep", "a", indent=2)]))
        self.draw()
        bare, tagged, deep = (self.band(rows, key) for key in ("bare", "tagged", "deep"))
        whole, before = self.ink(bare, theme.TEXT), self.ink(tagged, theme.TEXT)
        tag = {x for x, _ in self.ink(tagged, theme.DIM)}
        self.assertEqual((min(tag), max(tag), self.colours(bare)), (53, 63, {theme.PANEL, theme.TEXT}))
        self.assertLess(max(x for x, _ in whole), 70 - PAD)
        self.assertLess(max(x for x, _ in before), min(tag) - theme.LIST_SPACE)
        # A row that stands two steps in starts that much further right.
        self.assertEqual(min(x for x, _ in self.ink(deep, theme.TEXT)),
                         min(x for x, _ in whole) + 2 * theme.LIST_INDENT)
        # What is drawn of the name is what the font cuts it to.
        self.assertEqual([fit(name, 48), fit(name, 32)], ["a very..", "a v.."])
        rows.rows = [Row("bare", "a very.."), Row("tagged", "a v..", tag="12")]
        self.draw()
        self.assertEqual((self.ink(bare, theme.TEXT), self.ink(tagged, theme.TEXT)), (whole, before))

    def test_a_glowing_row_is_on_the_active_colour_at_a_strength_of_1_and_half_way_to_it_at_a_half(self) -> None:
        rows = self.ui.add(List(self.RECT, FLAT[:4], 3))
        rows.glow = {1: 1.0, 2: 0.5, 3: 0.5, "gone": 1.0}
        self.draw()
        self.assertEqual([self.at(109, 10 + ROW * row) for row in range(3)],
                         [theme.PANEL, theme.ACTIVE, (139, 88, 40)])
        self.assertEqual(self.colours(self.band(rows, 1)), {theme.ACTIVE, theme.TEXT})
        # The picked row glows from the background it has.
        self.assertEqual(self.at(109, 10 + ROW * 3), mix(theme.LIT, theme.ACTIVE, 0.5))
        rows.glow = {}
        self.draw()
        self.assertEqual([self.at(109, 10 + ROW * row) for row in range(4)], 3 * [theme.PANEL] + [theme.LIT])

    def test_nothing_is_drawn_outside_the_rectangle_of_a_list_that_is_scrolled_by_half_a_row(self) -> None:
        rect = (10, 10, 100, 33)
        rows = self.ui.add(List(rect, FLAT, 0))
        rows.glow = {3: 1.0}
        rows.scroll.to(ROW // 2)
        self.draw()
        x, y, w, h = rect
        outside = {self.at(column, row) for column in range(120) for row in range(120)
                   if not (x <= column < x + w and y <= row < y + h)}
        self.assertEqual(outside, {self.OUT})
        self.assertNotIn(self.OUT, self.colours(rect))
        # The lower half of the picked first row is at the top, and the top of the glowing fourth at the bottom.
        self.assertEqual([self.at(10, row) for row in (10, 15, 16, 39, 40, 42)],
                         2 * [theme.LIT] + 2 * [theme.PANEL] + 2 * [theme.ACTIVE])

    def test_the_bar_is_there_only_while_the_rows_do_not_fit_and_its_thumb_is_where_the_list_is_scrolled_to(
            self) -> None:
        rows = self.ui.add(List((10, 10, 100, 3 * ROW), FLAT[:3]))
        self.draw()
        track = (110 - BAR, 10, BAR, 3 * ROW)
        self.assertEqual(self.colours(track), {theme.PANEL})
        rows.rows = FLAT
        self.draw()
        start, length = rows.scroll.thumb(3 * ROW)
        self.assertEqual((start, length, self.colours(track)), (0, 11, {theme.BACKGROUND, theme.LINE}))
        self.assertEqual(self.ink(track, theme.LINE), self.ink((110 - BAR, 10, BAR, length), theme.LINE))
        self.assertEqual(len(self.ink(track, theme.LINE)), BAR * length)
        # The rows end where the bar begins.
        self.assertEqual(self.colours((109 - BAR, 10, 1, 3 * ROW)), {theme.PANEL})
        rows.scroll.to(rows.scroll.most)
        self.draw()
        self.assertEqual(self.ink(track, theme.LINE), self.ink((110 - BAR, 10 + 3 * ROW - length, BAR, length),
                                                                 theme.LINE))
        self.assertEqual(len(self.ink(track, theme.LINE)), BAR * length)


if __name__ == "__main__":
    unittest.main()
