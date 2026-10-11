"""A list that scrolls: how far it is scrolled as plain numbers, its rows, and the widget that shows them."""
from __future__ import annotations

from collections.abc import Callable, Collection, Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from . import theme
from .font import Font
from .layout import Rect
from .theme import Colour
from .ui import NOWHERE, Event, Pos, Ui, Widget, inside

if TYPE_CHECKING:
    from .canvas import Canvas

#: The keys that move the pick of a list.
KEYS = ("up", "down", "home", "end")
#: How much of its colour the square of a dim row keeps. The rest is the colour of the panel.
DIMMED = 0.4


def _nearest(above: int, below: int) -> int:
    """`above` over `below`, as the nearest whole number. `below` is above 0."""
    return (2 * above + below) // (2 * below)


@dataclass
class Scroll:
    """How far a content is scrolled in a view that may be too short to show it whole: plain numbers, in pixels.

    `content` and `view` are heights, and `offset` is how much of the content lies above the view. `row` is the
    height of one row, which `by` steps in. When `content` or `view` has been set, `to(offset)` brings the offset
    back inside what there is.
    """

    content: int = 0
    view: int = 0
    offset: int = 0
    row: int = theme.LIST_ROW_H

    @property
    def most(self) -> int:
        """The offset at which the end of the content is at the end of the view. It is 0 when everything fits."""
        return max(0, self.content - self.view)

    def to(self, offset: int) -> None:
        """Scroll to an offset, or as near to it as there is content."""
        self.offset = max(0, min(self.most, offset))

    def by(self, rows: int) -> None:
        """Scroll down by a number of rows, or up by a number below 0, and stop at the ends."""
        self.to(self.offset + rows * self.row)

    def show(self, top: int, bottom: int) -> None:
        """Scroll the least that brings the stretch of the content from `top` to just before `bottom` into view.

        A stretch longer than the view is shown from its top.
        """
        if top < self.offset:
            self.to(top)
        elif bottom > self.offset + self.view:
            self.to(min(top, bottom - self.view))

    def thumb(self, track: int) -> tuple[int, int] | None:
        """Where the thumb of a bar starts on a track of `track` pixels, and how long it is. None when all fits.

        The thumb is to the track what the view is to the content, but never shorter than `theme.LIST_THUMB_MIN`
        on a track that has the room.
        """
        if not self.most:
            return None
        length = min(track, max(theme.LIST_THUMB_MIN, _nearest(track * self.view, self.content)))
        return _nearest((track - length) * self.offset, self.most), length

    def drag(self, pixels: int, track: int) -> None:
        """Scroll as far as the thumb takes it when it is moved `pixels` down a track of `track` pixels."""
        thumb = self.thumb(track)
        if thumb is not None and thumb[1] < track:
            self.to(self.offset + _nearest(pixels * self.most, track - thumb[1]))


@dataclass(frozen=True)
class Row:
    """One line of a list.

    `key` is what the owner knows the row by, and no two rows of a list have the same. A row that is a `head` is a
    heading: it cannot be picked, and a click on it folds the rows under it away. `colour` is of a heading: its
    text is in it, and so is the small square before each row under it. `tag` is a short text at the right end,
    such as a count. `indent` is how many steps the row stands in from the left.

    `square` and `dim` are of a row that is no heading. `square` is a colour of its own for its square, and for
    its strip when it is picked: with None they have the heading's. A `dim` row has its text in `theme.DIM`, its
    tag dimmer still, and its square mixed towards the panel.
    """

    key: Hashable
    text: str
    head: bool = False
    colour: Colour = theme.DIM
    tag: str = ""
    indent: int = 0
    square: Colour | None = None
    dim: bool = False


class Place(NamedTuple):
    """A row that is shown, how far its top is below the top of the content, and the colour of its heading.

    The colour of a heading is its own. A row with no heading above it has none.
    """

    row: Row
    top: int
    colour: Colour | None


def lay(rows: Sequence[Row], folded: Collection[Hashable] = ()) -> list[Place]:
    """Where the rows go, from the top down. Those under a heading whose key is in `folded` are left out.

    A row is `theme.LIST_ROW_H` high. A heading that is not the first thing has `theme.LIST_GAP` of air above it.
    """
    placed: list[Place] = []
    top, hidden, colour = 0, False, None
    for row in rows:
        if row.head:
            hidden, colour = row.key in folded, row.colour
            top += theme.LIST_GAP if placed else 0
        elif hidden:
            continue
        placed.append(Place(row, top, colour))
        top += theme.LIST_ROW_H
    return placed


def mix(under: Colour, over: Colour, strength: float) -> Colour:
    """`over` mixed into `under`: at a strength of 0 it is `under`, at 1 it is `over`, and between them in between.

    A strength below 0 counts as 0, and one above 1 as 1.
    """
    strength = max(0.0, min(1.0, strength))
    red, green, blue = (round(low + (high - low) * strength) for low, high in zip(under, over))
    return red, green, blue


class List(Widget):
    """Rows one under the other, which scroll when they do not fit. One can be picked: `picked` is its key, or None.

    A click on a row picks it, and `changed` is called with its key, also when it was picked already. A click on a
    heading folds the rows under it away, and another brings them back: a picked row that is folded away stays
    picked. The up, down, home and end keys move the pick, past headings and folded rows. The wheel scrolls, and
    so does the bar that stands at the right edge while the rows do not fit.

    `rows` can be set at any time: the pick stays when its key is still there, and headings stay folded by their
    key. `glow` is from a key to a strength from 0 to 1, for the owner to set: the background of such a row has
    `theme.ACTIVE` mixed into it by that strength.
    """

    def __init__(self, rect: Rect, rows: Iterable[Row] = (), picked: Hashable | None = None,
                 changed: Callable[[Hashable], object] | None = None, help: str = "") -> None:
        super().__init__(rect, help)
        self.picked: Hashable | None = picked
        self.changed = changed
        self.glow: dict[Hashable, float] = {}
        self.scroll = Scroll()
        self._folded: set[Hashable] = set()
        self._placed: list[Place] = []
        self._mouse: Pos = NOWHERE
        #: While the thumb is dragged: how far down the mouse took hold of it, and the offset there was then.
        self._grab: tuple[int, int] | None = None
        self.rows = rows

    @property
    def rows(self) -> tuple[Row, ...]:
        """The rows, from the top down, the folded ones too."""
        return self._rows

    @rows.setter
    def rows(self, rows: Iterable[Row]) -> None:
        self._rows = tuple(rows)
        if all(row.head or row.key != self.picked for row in self._rows):
            self.picked = None
        self._settle()

    def _settle(self) -> None:
        """Lay the rows out as they are folded now, and keep the offset inside what there is of them."""
        self._placed = lay(self._rows, self._folded)
        self.scroll.content = self._placed[-1].top + theme.LIST_ROW_H if self._placed else 0
        self.scroll.view = self.rect[3]
        self.scroll.to(self.scroll.offset)

    def _bar(self) -> tuple[Rect, Rect] | None:
        """The track of the bar and the thumb on it. None while the rows fit: there is no bar then."""
        x, y, w, h = self.rect
        thumb = self.scroll.thumb(h)
        if thumb is None:
            return None
        left = x + w - theme.LIST_BAR_W
        return (left, y, theme.LIST_BAR_W, h), (left, y + thumb[0], theme.LIST_BAR_W, thumb[1])

    def _across(self) -> Rect:
        """The part of the rectangle the rows are in: all of it but the bar."""
        x, y, w, h = self.rect
        return x, y, w - (theme.LIST_BAR_W if self._bar() else 0), h

    def at(self, pos: Pos) -> Row | None:
        """The row a position is on as the list is scrolled now, or None when it is on none."""
        self._settle()
        if not inside(self._across(), pos):
            return None
        down = pos[1] - self.rect[1] + self.scroll.offset
        return next((place.row for place in self._placed if place.top <= down < place.top + theme.LIST_ROW_H), None)

    def _pick(self, key: Hashable) -> None:
        """Pick the row of a key, scroll it into view, and tell."""
        self.picked = key
        top = next(place.top for place in self._placed if place.row.key == key)
        self.scroll.show(top, top + theme.LIST_ROW_H)
        if self.changed is not None:
            self.changed(key)

    def on(self, event: Event, ui: Ui) -> bool:
        """Use the keys that move the pick, the wheel, and a press of the left button."""
        self._settle()
        if event.kind == "key":
            return self._key(event)
        self._mouse = event.pos
        if event.kind == "wheel":
            self.scroll.by(-theme.LIST_WHEEL * event.wheel)
            return True
        if event.kind == "move" and self._grab is not None:
            held_at, offset = self._grab
            self.scroll.to(offset)
            self.scroll.drag(event.pos[1] - held_at, self.rect[3])
        if event.button != 1:
            return False
        if event.kind == "up":
            self._grab = None
            return False
        self._press(event.pos)
        return True

    def _press(self, pos: Pos) -> None:
        """Take hold of the thumb, scroll a view's worth from the track beside it, fold a heading or pick a row."""
        bar = self._bar()
        if bar is not None and inside(bar[0], pos):
            _, top, _, length = bar[1]
            if top <= pos[1] < top + length:
                self._grab = (pos[1], self.scroll.offset)
            else:
                self.scroll.to(self.scroll.offset + (self.scroll.view if pos[1] >= top else -self.scroll.view))
            return
        row = self.at(pos)
        if row is not None and row.head:
            self._folded ^= {row.key}
            self._settle()
        elif row is not None:
            self._pick(row.key)

    def _key(self, event: Event) -> bool:
        """Move the pick to the row before or after, or to the first or the last, among the rows that are shown.

        A pick that is folded away moves from where it would stand. With nothing picked, down goes to the first row
        and up to the last. The keys are not used while there is no row to pick.
        """
        keys = [row.key for row in self._rows if not row.head]
        shown = {place.row.key for place in self._placed if not place.row.head}
        if event.mods or event.key not in KEYS or not shown:
            return False
        here = keys.index(self.picked) if self.picked in keys else None
        if event.key == "home" or (here is None and event.key == "down"):
            ahead = keys
        elif event.key == "end" or here is None:
            ahead = keys[::-1]
        elif event.key == "down":
            ahead = keys[here + 1:]
        else:
            ahead = keys[:here][::-1]
        key = next((key for key in ahead if key in shown), self.picked)
        if key != self.picked:
            self._pick(key)
        return True

    def draw(self, canvas: Canvas) -> None:
        """Draw the panel, the rows in view, and the bar while they do not fit: nothing outside the rectangle."""
        self._settle()
        x, y, w, h = self._across()
        lit = self.at(self._mouse) if self.hot and self._grab is None else None
        bar = self._bar()
        with canvas.clip(self.rect):
            canvas.box(self.rect, theme.PANEL)
            for place in self._placed:
                top = y + place.top - self.scroll.offset
                if y - theme.LIST_ROW_H < top < y + h:
                    self._draw_row(canvas, place, (x, top, w, theme.LIST_ROW_H), place.row == lit)
            if bar is not None:
                canvas.box(bar[0], theme.BACKGROUND)
                canvas.box(bar[1], theme.LINE)

    def _draw_row(self, canvas: Canvas, place: Place, rect: Rect, lit: bool) -> None:
        """Draw one row: its background, what stands before its text, its tag at the right, and its text between.

        The square of a row, and its strip when it is picked, are in the row's own colour where it has one, and
        in its heading's where it has none. A dim row is drawn dim.
        """
        row, _, colour = place
        x, y, w, h = rect
        picked = not row.head and row.key == self.picked
        dim = row.dim and not row.head
        ground = theme.LIT if picked else theme.RAISED if lit else theme.PANEL
        canvas.box(rect, mix(ground, theme.ACTIVE, self.glow.get(row.key, 0)))
        if not row.head and row.square is not None:
            colour = row.square
        if dim and colour is not None:
            colour = mix(theme.PANEL, colour, DIMMED)
        if picked and colour is not None:
            canvas.box((x, y, theme.LIST_STRIP, h), colour)
        left = x + theme.LIST_PAD + row.indent * theme.LIST_INDENT
        right = x + w - theme.LIST_PAD
        if row.head:
            _mark(canvas, left, y, row.key in self._folded, row.colour)
        elif colour is not None:
            canvas.box((left, y + (h - theme.LIST_SQUARE) // 2, theme.LIST_SQUARE, theme.LIST_SQUARE), colour)
        if colour is not None:
            left += theme.LIST_SQUARE + theme.LIST_SPACE
        if row.tag:
            width = Font.width(row.tag)
            canvas.text_in(row.tag, (right - width, y, width, h), theme.LINE if dim else theme.DIM)
            right -= width + theme.LIST_SPACE
        # A text too long for the room between is cut there, as `font.fit` cuts it.
        ink = row.colour if row.head else theme.DIM if dim else theme.TEXT
        canvas.text_in(row.text, (left, y, max(0, right - left), h), ink)


def _mark(canvas: Canvas, x: int, y: int, folded: bool, colour: Colour) -> None:
    """Draw the mark before a heading, in the room of a square at x and in a row whose top is y.

    It is an arrowhead that points down while the rows under the heading show, and to the right while they are
    folded away.
    """
    long, deep = theme.LIST_MARK, (theme.LIST_MARK + 1) // 2
    for step in range(deep):
        if folded:
            canvas.vline(x + (theme.LIST_SQUARE - deep) // 2 + step, y + (theme.LIST_ROW_H - long) // 2 + step,
                         long - 2 * step, colour)
        else:
            canvas.hline(x + (theme.LIST_SQUARE - long) // 2 + step, y + (theme.LIST_ROW_H - deep) // 2 + step,
                         long - 2 * step, colour)
