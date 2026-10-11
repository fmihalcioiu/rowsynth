"""What opens over everything else: a menu of things to do at a place, and a box that one line of text is typed into."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..errors import RowSynthError
from . import theme
from .font import ADVANCE, BODY_H, GLYPH_H, Font
from .layout import Rect
from .ui import Event, Pos, Ui, Widget, inside

if TYPE_CHECKING:
    from .canvas import Canvas

Size = tuple[int, int]

#: The two keys that say yes: the one beside the letters and the one of the number pad.
ENTER = ("return", "enter")


@dataclass(frozen=True)
class Item:
    """One thing to do in a menu: its text, what it stands for, the name of its icon, and whether it can be chosen.

    `value` is what the owner of the menu is told when the item is chosen, and can be anything.
    """

    text: str
    value: object = None
    icon: str = ""
    enabled: bool = True


def place(items: Sequence[Item | None], at: Pos, size: Size | None = None) -> tuple[Rect, list[Rect]]:
    """The box of a menu that is opened at a place, and the rectangle of each of its items, from the top down.

    A None among the items is a line between two groups, and has the strip its line is drawn in. The box is as
    wide as its widest text needs, with the room for an icon where an item has one, and never narrower than
    `theme.MENU_MIN_W`. Its top-left corner is at `at`. With the `size` of the art surface given, the box is
    moved the least that puts all of it on the surface. One too large for the surface starts at its left or at
    its top.
    """
    icons = theme.MENU_ICON + theme.GAP if any(item is not None and item.icon for item in items) else 0
    widest = max((Font.width(item.text) for item in items if item is not None), default=0)
    wide = max(theme.MENU_MIN_W, 2 * theme.MENU_PAD + icons + widest)
    highs = [theme.MENU_LINE_H if item is None else theme.MENU_ROW_H for item in items]
    high = 2 * theme.MENU_EDGE + sum(highs)
    x, y = at
    if size is not None:
        x, y = max(0, min(x, size[0] - wide)), max(0, min(y, size[1] - high))
    rows, top = [], y + theme.MENU_EDGE
    for tall in highs:
        rows.append((x + 1, top, wide - 2, tall))
        top += tall
    return (x, y, wide, high), rows


def caret_x(index: int) -> int:
    """Where the caret stands in gap `index` of a text, in pixels from where it stands before the first character.

    Gap 0 is before the first character, and each further one is after one more. The caret is one pixel wide,
    and so is the air between two characters: that is where it stands.
    """
    return index * ADVANCE


def gap_at(x: int, count: int) -> int:
    """The gap of a text of `count` characters that is nearest to an x, counted as `caret_x` counts.

    Half the way between two gaps goes to the one on the right. Left of the text it is 0, and right of it
    `count`.
    """
    return max(0, min(count, (x + ADVANCE // 2) // ADVANCE))


def in_view(scroll: int, caret: int, count: int, width: int) -> int:
    """How far a text of `count` characters is scrolled in a view `width` pixels wide, so that its caret is seen.

    `scroll` is how far it was scrolled before: the pixels of the text that lie left of the view. It is changed
    the least that brings the caret into the view. Then it is kept to what there is of the text: no part of the
    view is left empty at the right while some of the text lies left of it.
    """
    x = caret_x(caret)
    scroll = min(x, max(scroll, x - width + 1))
    return max(0, min(scroll, caret_x(count) + 1 - width))


def over(rect: Rect, size: Size) -> Rect:
    """The box that is typed into over a rectangle a name is drawn in, on an art surface of a size.

    The name is drawn at the left of its rectangle, and the text in the box stands where the name stood. The
    box is `theme.TEXT_BOX_H` high. It is wide enough for a name that fills the rectangle, never narrower than
    `theme.TEXT_BOX_MIN_W` and never wider than the surface, and is moved the least that puts all of it on the
    surface.
    """
    x, y, w, h = rect
    wide = min(size[0], max(theme.TEXT_BOX_MIN_W, w + 2 * theme.TEXT_PAD + 2))
    left = x - theme.TEXT_PAD - 1
    top = y + (h - BODY_H) // 2 - (theme.TEXT_BOX_H - BODY_H) // 2
    return max(0, min(left, size[0] - wide)), max(0, min(top, size[1] - theme.TEXT_BOX_H)), wide, theme.TEXT_BOX_H


def _end(ui: Ui, widget: Widget, tell: Callable[..., object] | None, *what: object) -> None:
    """Close a widget that is open, and then tell its owner, who may open another one."""
    ui.close(widget)
    if tell is not None:
        tell(*what)


class Menu(Widget):
    """A short list of things to do, opened at a place as the modal widget: `ui.open(Menu(items, at, chosen))`.

    Each of `items` is an `Item`, or None for a thin line between two groups. The menu opens with its top-left
    corner at `at`, which is where the mouse is, and is moved onto the art surface when the Ui opens it. The
    row under the mouse is lit: `lit` is its index among the items, or None. A press of the left button on a
    row that is enabled closes the menu and calls `chosen` with the value of the item. A press of any button
    off the menu closes it and calls `closed`, and so does escape. A press anywhere else on the menu does
    nothing. Up and down move the lit row over the rows that are enabled, round from the last to the first, and
    enter chooses the lit one.
    """

    def __init__(self, items: Sequence[Item | None], at: Pos, chosen: Callable[[object], object] | None = None,
                 closed: Callable[[], object] | None = None) -> None:
        self.items = tuple(items)
        if all(item is None for item in self.items):
            raise RowSynthError("a menu needs at least one thing to do")
        self.at = at
        self.chosen = chosen
        self.closed = closed
        self.lit: int | None = None
        rect, self.rows = place(self.items, at)
        super().__init__(rect)

    def keep_in(self, size: Size) -> None:
        """Take the size of the art surface, and move the menu the least that puts all of it there."""
        self.rect, self.rows = place(self.items, self.at, size)

    def row_at(self, pos: Pos) -> int | None:
        """The index of the item whose row a position is on, when it can be chosen. None on anything else."""
        for index, (item, row) in enumerate(zip(self.items, self.rows)):
            if inside(row, pos):
                return index if item is not None and item.enabled else None
        return None

    def on(self, event: Event, ui: Ui) -> bool:
        """Use everything: light the row under the mouse, and act on a press and on the menu's keys."""
        if event.kind == "key":
            self._key(event, ui)
            return True
        if event.kind == "text":
            return True
        self.lit = self.row_at(event.pos)
        if event.kind == "down" and not self.hit(event.pos):
            _end(ui, self, self.closed)
        elif event.kind == "down" and event.button == 1 and self.lit is not None:
            _end(ui, self, self.chosen, self.items[self.lit].value)
        return True

    def _key(self, event: Event, ui: Ui) -> None:
        """Close on escape, choose the lit row on enter, and move the lit row up or down."""
        can = [index for index, item in enumerate(self.items) if item is not None and item.enabled]
        if event.key == "escape":
            _end(ui, self, self.closed)
        elif event.key in ENTER and self.lit is not None:
            _end(ui, self, self.chosen, self.items[self.lit].value)
        elif event.key in ("up", "down") and can:
            step = 1 if event.key == "down" else -1
            # With no row lit, down goes to the first and up to the last.
            here = can.index(self.lit) if self.lit in can else -(step > 0)
            self.lit = can[(here + step) % len(can)]

    def draw(self, canvas: Canvas) -> None:
        """Draw the box, the lit row's ground, and each item: its icon and its text, dim when it is not enabled."""
        x, _, w, _ = self.rect
        left = x + theme.MENU_PAD
        icons = theme.MENU_ICON + theme.GAP if any(item is not None and item.icon for item in self.items) else 0
        with canvas.clip(self.rect):
            canvas.box(self.rect, theme.RAISED)
            canvas.frame(self.rect, theme.LINE)
            for index, (item, row) in enumerate(zip(self.items, self.rows)):
                _, top, _, high = row
                if item is None:
                    canvas.hline(left, top + high // 2, w - 2 * theme.MENU_PAD, theme.LINE)
                    continue
                if index == self.lit:
                    canvas.box(row, theme.LIT)
                ink = theme.TEXT if item.enabled else theme.DIM
                if item.icon:
                    canvas.icon(item.icon, left, top + (high - theme.MENU_ICON) // 2, ink)
                canvas.text_in(item.text, (left + icons, top, w - 2 * theme.MENU_PAD - icons, high), ink)


class TextBox(Widget):
    """One line of text that is typed, opened as the modal widget: `ask` opens one over a name.

    `caret` is the gap the caret stands in, from 0 before the first character. `anchor` is the gap the picked
    text runs from, up to the caret or back to it: where the two are the same, nothing is picked. The box opens
    with all of its text picked, so that typing takes the place of what was there.

    Typed characters take the place of the picked text, or go in at the caret. Only those that `allowed` says
    yes to go in, when it is given, and the text never grows longer than `most` characters. Backspace and
    delete take the picked text out, or the character before or after the caret. Left, right, home and end move
    the caret, and nothing is picked after: with text picked, left goes to its start and right to its end. A
    press of the left button puts the caret in the nearest gap. Ctrl and a picks everything.

    Enter closes the box and calls `done` with the text. Escape, or a press off the box, closes it and calls
    `cancelled`. A text too wide for the box is scrolled so that the caret is seen: `scroll` is how far.
    """

    takes_text = True

    def __init__(self, rect: Rect, text: str = "", done: Callable[[str], object] | None = None,
                 cancelled: Callable[[], object] | None = None, allowed: Callable[[str], object] | None = None,
                 most: int = 32, help: str = "") -> None:
        super().__init__(rect, help)
        self.text = text
        self.done = done
        self.cancelled = cancelled
        self.allowed = allowed
        self.most = most
        self.anchor, self.caret = 0, len(text)
        self.scroll = 0
        self._show()

    @property
    def picked(self) -> tuple[int, int]:
        """The gap the picked text starts at and the gap it ends at. They are the same when nothing is picked."""
        return min(self.anchor, self.caret), max(self.anchor, self.caret)

    def _show(self) -> None:
        """Scroll the text the least that has the caret in the box."""
        self.scroll = in_view(self.scroll, self.caret, len(self.text), max(1, self.rect[2] - 2 * theme.TEXT_PAD))

    def _put(self, start: int, end: int, text: str = "") -> None:
        """Put a text in the place of what is between two gaps, and the caret after it with nothing picked."""
        self.text = self.text[:start] + text + self.text[end:]
        self.anchor = self.caret = start + len(text)

    def on(self, event: Event, ui: Ui) -> bool:
        """Use everything: what is typed, the box's keys, and a press on the box or off it."""
        if event.kind == "text":
            self._type(event.text)
        elif event.kind == "key":
            self._key(event, ui)
        elif event.kind == "down" and not self.hit(event.pos):
            _end(ui, self, self.cancelled)
        elif event.kind == "down" and event.button == 1:
            x = event.pos[0] - self.rect[0] - theme.TEXT_PAD + self.scroll
            self.anchor = self.caret = gap_at(x, len(self.text))
        self._show()
        return True

    def _type(self, typed: str) -> None:
        """Put typed characters in: those that can be shown and are allowed, and no more than there is room for.

        Where none of them goes in, nothing changes, and what is picked stays picked.
        """
        start, end = self.picked
        room = max(0, self.most - len(self.text) + end - start)
        fit = [char for char in typed if char.isprintable() and (self.allowed is None or self.allowed(char))]
        if fit[:room]:
            self._put(start, end, "".join(fit[:room]))

    def _key(self, event: Event, ui: Ui) -> None:
        """Act on a key of the box. A key that types is none of them: its character comes as text."""
        key, count = event.key, len(self.text)
        start, end = self.picked
        if key in ENTER:
            _end(ui, self, self.done, self.text)
        elif key == "escape":
            _end(ui, self, self.cancelled)
        elif key == "a" and event.mods == {"ctrl"}:
            self.anchor, self.caret = 0, count
        elif key in ("backspace", "delete"):
            if start == end:
                start, end = (max(0, start - 1), end) if key == "backspace" else (start, min(count, end + 1))
            self._put(start, end)
        elif key in ("left", "right", "home", "end"):
            some = start != end
            self.anchor = self.caret = {"left": start if some else max(0, start - 1),
                                        "right": end if some else min(count, end + 1),
                                        "home": 0, "end": count}[key]

    def draw(self, canvas: Canvas) -> None:
        """Draw the box, the ground of the picked text, the text, and the caret: nothing outside the box.

        The caret is a line one pixel wide in the gap it stands in. It reaches one pixel above the letters and
        one below the line they stand on, and does not blink.
        """
        self._show()
        x, y, w, h = self.rect
        start, end = self.picked
        # Where the caret stands before the first character, as the text is scrolled now, and the top of a letter.
        left, top = x + theme.TEXT_PAD - self.scroll, y + (h - BODY_H) // 2
        # The ground of the picked text reaches as far above the letters as their tails hang below them.
        tails = GLYPH_H - BODY_H
        with canvas.clip(self.rect):
            canvas.box(self.rect, theme.BACKGROUND)
            canvas.frame(self.rect, theme.ACTIVE)
            with canvas.clip((x + theme.TEXT_PAD, y + 1, w - 2 * theme.TEXT_PAD, h - 2)):
                if start != end:
                    canvas.box((left + caret_x(start), top - tails, caret_x(end) - caret_x(start) + 1,
                                GLYPH_H + tails), theme.LIT)
                canvas.text(self.text, left + 1, top, theme.TEXT)
                canvas.vline(left + caret_x(self.caret), top - 1, BODY_H + 2, theme.ACTIVE)


def ask(ui: Ui, rect: Rect, text: str, done: Callable[[str], object],
        cancelled: Callable[[], object] | None = None, allowed: Callable[[str], object] | None = None,
        most: int = 32, help: str = "") -> TextBox:
    """Open a text box over the rectangle a name is drawn in, as the modal widget, and give it back.

    The box is where `over` puts it on the art surface the Ui was last drawn on, and holds `text`, all of it
    picked. `done` is called with the new text on enter, and `cancelled` when the box is left without one.
    """
    box = TextBox(over(rect, ui.size), text, done, cancelled, allowed, most, help)
    ui.open(box)
    return box
