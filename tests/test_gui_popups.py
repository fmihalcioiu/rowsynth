"""Tests for rowsynth.gui.popups: the menu that opens at a place, and the box a line of text is typed into."""
from __future__ import annotations

import os
import unittest

from rowsynth.errors import RowSynthError
from rowsynth.gui import layout, sprites, theme
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.font import Font
from rowsynth.gui.popups import Item, Menu, TextBox, ask, caret_x, gap_at, in_view, over, place
from rowsynth.gui.ui import Event, Ui, Widget, inside

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
#: Two groups of things to do with a line between them. The second has an icon, and the fourth cannot be chosen.
ITEMS = (Item("RENAME", "rename"), Item("COPY", "copy", icon="copy"), None,
         Item("REMOVE", "remove", enabled=False), Item("CLEAR", "clear"))
#: Opened at (10, 20), the menu of those is this box. A pixel on each of its rows, the line among them.
MENU = (10, 20, 59, 57)
RENAME, COPY, LINE, REMOVE, CLEAR = (20, 28), (20, 40), (20, 48), (20, 57), (20, 69)
#: A text box with room for ten characters and the caret after them. The caret before the first is at x 13.
BOX = (10, 10, 72, 13)
LONG = "abcdefghijklmnop"


class Keys(Widget):
    """A widget that uses every press and every key, and notes the keys."""

    def __init__(self, rect: layout.Rect) -> None:
        super().__init__(rect)
        self.keys: list[str] = []

    def on(self, event: Event, ui: Ui) -> bool:
        if event.kind == "key":
            self.keys.append(event.key)
        return event.kind in ("down", "key")


def click(ui: Ui, pos: tuple[int, int], button: int = 1) -> None:
    """Press a mouse button at a position and let it go there."""
    ui.handle(Event("down", pos, button=button))
    ui.handle(Event("up", pos, button=button))


def press(ui: Ui, *keys: str, mods: frozenset[str] = frozenset()) -> None:
    """Press keys that type nothing, one after the other."""
    for key in keys:
        ui.handle(Event("key", key=key, mods=mods))


def type_in(ui: Ui, text: str) -> None:
    """Type characters as the window tells them: each as the key that went down and then as its character."""
    for char in text:
        ui.handle(Event("key", key="space" if char == " " else char))
        ui.handle(Event("text", text=char))


def menu_on(items: tuple[Item | None, ...] = ITEMS, at: tuple[int, int] = MENU[:2]) -> tuple[Ui, Menu, list]:
    """A Ui with a menu open on it, and the list that is told what was chosen, or "closed"."""
    ui, told = Ui(), []
    menu = Menu(items, at, told.append, lambda: told.append("closed"))
    ui.open(menu)
    return ui, menu, told


def box_on(text: str = "lead", **more: object) -> tuple[Ui, TextBox, list]:
    """A Ui with a text box open on it, and the list that is told the text it ended with, or "cancelled"."""
    ui, told = Ui(), []
    box = TextBox(BOX, text, told.append, lambda: told.append("cancelled"), **more)
    ui.open(box)
    return ui, box, told


class MenuSums(unittest.TestCase):
    def test_the_box_is_as_wide_as_its_widest_text_and_the_rows_stand_under_each_other_in_it(self) -> None:
        # The outline, the air and the room for an icon before a text of six letters: 6 + 12 + 35 + 6.
        box, rows = place(ITEMS, (10, 20))
        self.assertEqual((box, rows[0], rows[1], rows[4]),
                         (MENU, (11, 22, 57, 12), (11, 34, 57, 12), (11, 63, 57, 12)))
        # The line between the two groups has a thin strip of its own.
        self.assertEqual(rows[2:4], [(11, 46, 57, 5), (11, 51, 57, 12)])
        # With no icon there is no room for one, and a short text leaves the box at its least width.
        self.assertEqual(place((Item("A"), Item("RENAME IT")), (0, 0)),
                         ((0, 0, 65, 28), [(1, 2, 63, 12), (1, 14, 63, 12)]))
        self.assertEqual(place((Item("A"),), (3, 4))[0], (3, 4, theme.MENU_MIN_W, 16))

    def test_opened_at_any_corner_of_the_surface_the_box_is_moved_inside_it(self) -> None:
        size = (200, 100)
        for at, corner in (((0, 0), (0, 0)), ((199, 0), (141, 0)), ((0, 99), (0, 43)), ((199, 99), (141, 43)),
                           ((50, 20), (50, 20)), ((141, 43), (141, 43)), ((-5, -5), (0, 0))):
            box, rows = place(ITEMS, at, size)
            self.assertEqual((box, rows[0]), ((*corner, 59, 57), (corner[0] + 1, corner[1] + 2, 57, 12)), at)
        # With no surface told it stays where it was opened, and one too large for the surface starts at its corner.
        self.assertEqual((place(ITEMS, (199, 99))[0], place(ITEMS, (30, 30), (40, 30))[0]),
                         ((199, 99, 59, 57), (0, 0, 59, 57)))


class Menus(unittest.TestCase):
    def test_a_press_on_a_row_closes_the_menu_and_tells_what_the_row_stands_for(self) -> None:
        ui, menu, told = menu_on()
        self.assertEqual((ui.modal, ui.focus, menu.rect, menu.lit), (menu, menu, MENU, None))
        ui.handle(Event("move", COPY))
        self.assertEqual(menu.lit, 1)
        ui.handle(Event("down", COPY, button=1))
        self.assertEqual((told, ui.modal), (["copy"], None))
        ui.handle(Event("up", COPY, button=1))
        self.assertEqual(told, ["copy"])

    def test_a_row_that_is_not_enabled_the_line_and_the_edge_of_the_box_do_nothing_and_are_not_lit(self) -> None:
        ui, menu, told = menu_on()
        for pos in (REMOVE, LINE, MENU[:2], (68, 76)):
            click(ui, pos)
            self.assertEqual((told, ui.modal, menu.lit), ([], menu, None), pos)
        # Neither does another button than the left on a row that is lit.
        for button in (2, 3):
            click(ui, CLEAR, button)
        self.assertEqual((told, ui.modal, menu.lit), ([], menu, 4))

    def test_a_press_off_the_menu_and_escape_close_it_and_tell_that_nothing_was_chosen(self) -> None:
        for leave in (lambda ui: click(ui, (9, 28)), lambda ui: click(ui, (69, 28), 3),
                      lambda ui: press(ui, "escape")):
            ui, menu, told = menu_on()
            leave(ui)
            self.assertEqual((told, ui.modal), (["closed"], None))
        # A menu that tells nobody closes all the same.
        ui = Ui()
        ui.open(Menu(ITEMS, (10, 20)))
        click(ui, RENAME)
        ui.open(Menu(ITEMS, (10, 20)))
        press(ui, "escape")
        self.assertIsNone(ui.modal)

    def test_up_and_down_move_the_lit_row_over_the_enabled_rows_and_round_and_enter_chooses_it(self) -> None:
        ui, menu, told = menu_on()
        # With no row lit, enter chooses nothing.
        press(ui, "return")
        self.assertEqual((told, ui.modal), ([], menu))
        lit = []
        for key in ("down", "down", "down", "down", "up", "up"):
            press(ui, key)
            lit.append(menu.lit)
        self.assertEqual(lit, [0, 1, 4, 0, 4, 1])
        press(ui, "enter")
        self.assertEqual((told, ui.modal), (["copy"], None))
        # Up goes to the last row from none. The other keys are the menu's too, and do nothing.
        ui, menu, told = menu_on()
        press(ui, "up", "space", "z")
        self.assertEqual((menu.lit, told, ui.modal), (4, [], menu))
        ui.handle(Event("move", (9, 28)))
        self.assertIsNone(menu.lit)
        # Where no row is enabled, none is lit.
        ui, menu, told = menu_on((Item("A", enabled=False),))
        press(ui, "down", "up", "return")
        self.assertEqual((menu.lit, told), (None, []))

    def test_opened_on_a_ui_near_the_edge_of_its_surface_the_menu_and_its_rows_are_inside(self) -> None:
        ui, told = Ui(), []
        ui.size = (200, 100)
        menu = ui.open(Menu(ITEMS, (199, 99), told.append))
        self.assertEqual((menu.rect, menu.rows[4], menu.at), ((141, 43, 59, 57), (142, 86, 57, 12), (199, 99)))
        click(ui, (150, 90))
        self.assertEqual(told, ["clear"])
        # On a surface of another size it is put anew, from the place it was opened at.
        menu.keep_in((100, 60))
        self.assertEqual(menu.rect, (41, 3, 59, 57))

    def test_a_menu_with_nothing_to_do_is_refused(self) -> None:
        for items in ((), (None, None)):
            with self.assertRaises(RowSynthError) as refused:
                Menu(items, (0, 0))
            self.assertEqual(str(refused.exception), "a menu needs at least one thing to do")

    def test_what_is_chosen_can_open_a_text_box_and_the_release_of_that_press_is_still_the_menus(self) -> None:
        ui, named = Ui(), []
        menu = ui.open(Menu(ITEMS, (10, 20), lambda value: ask(ui, (200, 200, 30, 12), "lead", named.append)))
        ui.handle(Event("down", RENAME, button=1))
        box = ui.modal
        self.assertEqual((type(box), box.text, box.picked, ui.held), (TextBox, "lead", (0, 4), menu))
        ui.handle(Event("up", RENAME, button=1))
        self.assertEqual((ui.modal, ui.held, named), (box, None, []))
        type_in(ui, "bass")
        press(ui, "return")
        self.assertEqual((named, ui.modal), (["bass"], None))


class TextSums(unittest.TestCase):
    def test_the_caret_stands_in_the_air_between_two_characters_and_a_gap_is_the_nearest_to_an_x(self) -> None:
        self.assertEqual([caret_x(gap) for gap in (0, 1, 5)], [0, 6, 30])
        # Half the way between two gaps goes to the right one. Off the text it is the first gap or the last.
        self.assertEqual([gap_at(x, 4) for x in (-9, 0, 2, 3, 6, 8, 9, 24, 99)], [0, 0, 0, 1, 1, 1, 2, 4, 4])
        self.assertEqual([gap_at(x, 0) for x in (-9, 0, 9)], [0, 0, 0])

    def test_a_text_is_scrolled_the_least_that_has_its_caret_in_view_and_no_further_than_its_end(self) -> None:
        # A view of 31 pixels shows five characters and the caret after them.
        self.assertEqual([in_view(0, caret, 5, 31) for caret in (0, 3, 5)], [0, 0, 0])
        # Ten characters are 61 pixels with the caret after the last: at most 30 of them lie left of the view.
        self.assertEqual([in_view(0, 5, 10, 31), in_view(0, 6, 10, 31), in_view(0, 10, 10, 31)], [0, 6, 30])
        self.assertEqual([in_view(30, caret, 10, 31) for caret in (10, 7, 5, 4, 0)], [30, 30, 30, 24, 0])
        # A text that has become shorter is drawn back, so that the view is filled.
        self.assertEqual([in_view(30, 6, 6, 31), in_view(30, 2, 3, 31)], [6, 0])

    def test_the_box_over_a_name_has_its_text_where_the_name_stood_and_is_kept_on_the_surface(self) -> None:
        size = (960, 540)
        # The name of the file in the top bar: its letters start at x 6, and are 7 down in a bar 22 high.
        self.assertEqual(over((6, 0, 66, 22), size), (2, 4, 74, 13))
        # A short name gets the least width. One at an edge of the surface is moved in.
        self.assertEqual(over((100, 50, 20, 12), size), (96, 49, theme.TEXT_BOX_MIN_W, theme.TEXT_BOX_H))
        self.assertEqual([over((180, 92, 18, 12), (200, 100)), over((0, 0, 10, 12), (200, 100))],
                         [(128, 87, 72, 13), (0, 0, 72, 13)])
        # It is never wider than the surface.
        self.assertEqual(over((6, 0, 400, 22), (50, 40)), (0, 4, 50, 13))


class TextBoxes(unittest.TestCase):
    def test_it_opens_with_all_of_its_text_picked_and_typing_takes_its_place(self) -> None:
        ui, box, told = box_on()
        self.assertEqual((ui.modal, ui.focus, box.text, box.picked, box.caret), (box, box, "lead", (0, 4), 4))
        type_in(ui, "ab")
        self.assertEqual((box.text, box.picked, told), ("ab", (2, 2), []))

    def test_what_is_typed_goes_in_at_the_caret(self) -> None:
        ui, box, _ = box_on()
        press(ui, "home")
        type_in(ui, "x")
        press(ui, "right", "right")
        type_in(ui, "y ")
        self.assertEqual((box.text, box.caret), ("xley ad", 5))
        # Characters that come together go in together. A key alone types nothing, and neither does what
        # cannot be shown.
        ui.handle(Event("text", text="12"))
        ui.handle(Event("key", key="q"))
        ui.handle(Event("text", text="\t"))
        self.assertEqual((box.text, box.picked), ("xley 12ad", (7, 7)))

    def test_only_what_is_allowed_goes_in_and_the_text_grows_no_longer_than_the_most(self) -> None:
        ui, box, _ = box_on(allowed=str.isalnum, most=6)
        # Where nothing of what was typed goes in, the picked text stays, and stays picked.
        type_in(ui, " _")
        self.assertEqual((box.text, box.picked), ("lead", (0, 4)))
        press(ui, "end")
        type_in(ui, "a_b")
        type_in(ui, "c")
        self.assertEqual((box.text, box.caret), ("leadab", 6))
        # What is picked makes room: typed over three characters of a text that is full, three go in.
        box.anchor, box.caret = 1, 4
        ui.handle(Event("text", text="12-34567"))
        self.assertEqual((box.text, box.caret), ("l123ab", 4))

    def test_backspace_and_delete_take_out_the_picked_text_or_the_character_before_or_after_the_caret(self) -> None:
        ui, box, _ = box_on()
        press(ui, "backspace", "backspace", "delete")
        self.assertEqual((box.text, box.caret), ("", 0))
        ui, box, _ = box_on()
        press(ui, "end", "delete", "backspace")
        self.assertEqual((box.text, box.caret), ("lea", 3))
        press(ui, "home", "backspace", "delete")
        self.assertEqual((box.text, box.caret), ("ea", 0))
        ui, box, _ = box_on()
        box.anchor, box.caret = 3, 1
        press(ui, "delete")
        self.assertEqual((box.text, box.picked), ("ld", (1, 1)))

    def test_left_right_home_and_end_move_the_caret_and_leave_nothing_picked(self) -> None:
        ui, box, _ = box_on()
        at = []
        for key in ("left", "left", "right", "end", "right", "left", "home"):
            press(ui, key)
            at.append((box.caret, box.picked))
        # From picked text, left goes to its start. At an end of the text the caret stays.
        self.assertEqual(at, [(gap, (gap, gap)) for gap in (0, 0, 1, 4, 4, 3, 0)])
        box.anchor, box.caret = 3, 1
        press(ui, "right")
        self.assertEqual((box.caret, box.picked), (3, (3, 3)))

    def test_a_press_on_the_box_puts_the_caret_in_the_nearest_gap(self) -> None:
        ui, box, told = box_on()
        at = []
        for x in (10, 13, 15, 16, 27, 28, 37, 80):
            click(ui, (x, 15))
            at.append(box.caret)
        self.assertEqual((at, box.picked, told), ([0, 0, 0, 1, 2, 3, 4, 4], (4, 4), []))
        # Another button than the left moves nothing, and ends nothing.
        click(ui, (13, 15), 3)
        self.assertEqual((box.caret, ui.modal), (4, box))

    def test_ctrl_and_a_picks_everything(self) -> None:
        ui, box, _ = box_on()
        press(ui, "home")
        press(ui, "a")
        self.assertEqual(box.picked, (0, 0))
        press(ui, "a", mods=CTRL)
        self.assertEqual((box.picked, box.caret, box.text), ((0, 4), 4, "lead"))

    def test_enter_closes_the_box_and_tells_its_text_and_escape_or_a_press_off_it_tells_that_it_was_left(self) -> None:
        for key in ("return", "enter"):
            ui, box, told = box_on()
            type_in(ui, "bass")
            press(ui, key)
            self.assertEqual((told, ui.modal), (["bass"], None))
        for leave in (lambda ui: press(ui, "escape"), lambda ui: click(ui, (9, 15)), lambda ui: click(ui, (40, 23), 3)):
            ui, box, told = box_on()
            type_in(ui, "bass")
            leave(ui)
            self.assertEqual((told, ui.modal), (["cancelled"], None))
        # A box that tells nobody closes all the same.
        for key in ("return", "escape"):
            ui = Ui()
            ui.open(TextBox(BOX, "lead"))
            press(ui, key)
            self.assertIsNone(ui.modal)

    def test_a_text_too_long_for_the_box_is_scrolled_so_that_the_caret_is_in_view(self) -> None:
        # The box shows 66 pixels of the text, and sixteen characters are 97 with the caret after the last.
        ui, box, _ = box_on(LONG)
        self.assertEqual((box.caret, box.scroll), (16, 31))
        press(ui, "home")
        self.assertEqual((box.caret, box.scroll), (0, 0))
        press(ui, "end")
        # The pixel the box shows first is 31 of the text: the nearest gap is the one after five characters.
        click(ui, (13, 15))
        self.assertEqual((box.caret, box.scroll), (5, 30))
        press(ui, "end", "backspace", "backspace", "backspace")
        self.assertEqual((box.text, box.scroll), (LONG[:13], 13))
        # A text that fits is not scrolled at all.
        self.assertEqual(TextBox(BOX, LONG[:10]).scroll, 0)

    def test_with_a_box_open_the_space_bar_and_a_key_of_the_piano_reach_neither_keys_nor_a_widget_under(self) -> None:
        ui, called, named = Ui(), [], []
        ui.keys[("space", frozenset())] = lambda: called.append("play")
        under = ui.add(Keys((0, 0, 20, 20)))
        click(ui, (5, 5))
        box = ask(ui, (30, 30, 24, 12), "lead", named.append)
        type_in(ui, "z x")
        self.assertEqual((box.text, called, under.keys, ui.focus), ("z x", [], [], box))
        press(ui, "return")
        # With the box closed the widget under it has the focus again, and the keys are keys.
        type_in(ui, "z ")
        self.assertEqual((named, called, under.keys, ui.focus), (["z x"], [], ["z", "space"], under))
        ui.handle(Event("down", (50, 50), button=1))
        type_in(ui, " ")
        self.assertEqual(called, ["play"])

    def test_asking_opens_a_box_over_the_name_on_the_surface_the_ui_was_drawn_on(self) -> None:
        ui, told = Ui(), []
        ui.size = (200, 100)
        box = ask(ui, (180, 92, 18, 12), "kit", told.append, lambda: told.append("cancelled"), str.isalpha, 4,
                  "the name of the channel")
        self.assertEqual((ui.modal, box.rect, box.text, box.picked), (box, (128, 87, 72, 13), "kit", (0, 3)))
        self.assertEqual((ui.help, box.most), ("the name of the channel", 4))
        type_in(ui, "a1bcde")
        press(ui, "escape")
        self.assertEqual((box.text, told), ("abcd", ["cancelled"]))


@unittest.skipUnless(pygame, NEEDS)
class Drawn(unittest.TestCase):
    def setUp(self) -> None:
        self.surface = pygame.Surface((120, 90))
        self.surface.fill(theme.PANEL)
        self.canvas = Canvas(self.surface, Font(sprites.load("font")), sprites.load("icons"))
        self.ui = Ui()

    def at(self, *pos: int) -> tuple[int, ...]:
        """The colour of one pixel."""
        return tuple(self.surface.get_at(pos))[:3]

    def colours(self, rect: layout.Rect) -> set[tuple[int, ...]]:
        """The colours a rectangle has in it."""
        x, y, w, h = rect
        return {self.at(x + column, y + row) for column in range(w) for row in range(h)}

    def off(self, rect: layout.Rect) -> set[tuple[int, ...]]:
        """The colours of the surface outside a rectangle."""
        width, height = self.surface.get_size()
        return {self.at(x, y) for x in range(width) for y in range(height) if not inside(rect, (x, y))}

    def test_the_menu_has_its_lit_row_lit_its_disabled_row_dim_and_a_line_between_its_groups(self) -> None:
        menu = self.ui.open(Menu(ITEMS, (10, 20)))
        self.ui.handle(Event("move", COPY))
        self.ui.draw(self.canvas)
        self.assertEqual((menu.rect, self.at(10, 20), self.at(68, 76), self.at(11, 21)),
                         (MENU, theme.LINE, theme.LINE, theme.RAISED))
        rename, copy, line, remove, clear = menu.rows
        self.assertEqual([self.colours(row) for row in (rename, copy, remove, clear)],
                         [{theme.RAISED, theme.TEXT}, {theme.LIT, theme.TEXT}, {theme.RAISED, theme.DIM},
                          {theme.RAISED, theme.TEXT}])
        # The icon stands before the text of its row, and the other texts leave its room free.
        self.assertIn(theme.TEXT, self.colours((16, 35, 9, 9)))
        self.assertEqual((self.colours((11, 22, 17, 12)), self.colours((28, 24, 1, 7))), ({theme.RAISED}, {theme.TEXT}))
        # The line is one pixel thick in the middle of its strip, and stops short of the outline.
        self.assertEqual((self.colours((16, 48, 47, 1)), self.colours((11, 46, 57, 2)) | self.colours((11, 49, 57, 2)),
                          self.at(15, 48), self.at(63, 48)),
                         ({theme.LINE}, {theme.RAISED}, theme.RAISED, theme.RAISED))
        self.assertEqual(self.off(menu.rect), {theme.PANEL})

    def test_the_caret_is_a_line_where_the_sum_says_and_the_picked_text_has_a_ground_of_its_own(self) -> None:
        box = self.ui.open(TextBox(BOX, "lead bass"))
        box.anchor, box.caret = 2, 6
        self.ui.draw(self.canvas)
        self.assertEqual((self.at(10, 10), self.at(81, 22), self.at(11, 11)),
                         (theme.ACTIVE, theme.ACTIVE, theme.BACKGROUND))
        # The caret is nine pixels long in the gap after six characters, and stops short of the outline.
        x = BOX[0] + theme.TEXT_PAD
        self.assertEqual((self.colours((x + caret_x(6), 12, 1, 9)), self.colours((x + caret_x(6) + 1, 11, 29, 11))),
                         ({theme.ACTIVE}, {theme.BACKGROUND, theme.TEXT}))
        # The ground of the picked text runs from the gap it starts in to the gap the caret is in.
        self.assertEqual((self.colours((x + caret_x(2), 11, 24, 11)), self.at(x + caret_x(6), 11),
                          self.at(x + caret_x(6), 21), self.colours((x, 11, 12, 11))),
                         ({theme.LIT, theme.TEXT}, theme.LIT, theme.LIT, {theme.BACKGROUND, theme.TEXT}))
        self.assertEqual(self.off(box.rect), {theme.PANEL})
        # With nothing picked there is no such ground, and the caret alone stands before the first character.
        box.anchor = box.caret = 0
        self.ui.draw(self.canvas)
        self.assertEqual((self.colours((x, 12, 1, 9)), self.at(x, 11), self.at(x, 21)),
                         ({theme.ACTIVE}, theme.BACKGROUND, theme.BACKGROUND))
        self.assertEqual(self.colours((x + 1, 11, 65, 11)), {theme.BACKGROUND, theme.TEXT})

    def test_a_text_that_is_scrolled_is_cut_to_the_inside_of_the_box_and_nothing_is_drawn_outside(self) -> None:
        box = self.ui.open(TextBox(BOX, LONG))
        self.ui.draw(self.canvas)
        # The caret is after the last character, in the last column the text has in the box.
        x = BOX[0] + theme.TEXT_PAD + caret_x(16) - box.scroll
        self.assertEqual((x, self.colours((x, 12, 1, 9))), (78, {theme.ACTIVE}))
        self.assertEqual(self.colours((13, 11, 65, 11)), {theme.LIT, theme.TEXT})
        self.assertEqual(self.colours((11, 11, 2, 11)) | self.colours((79, 11, 2, 11)), {theme.BACKGROUND})
        self.assertEqual(self.off(box.rect), {theme.PANEL})


if __name__ == "__main__":
    unittest.main()
