"""The parts every editor shares: the wheel, a heading, a settings bar, keys, scroll bars, a velocity lane, a menu."""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..song import _NAMES
from . import theme
from .dials import Knob
from .font import BODY_H, Font
from .layout import Rect
from .lists import Scroll
from .panels import within
from .popups import Item
from .theme import Colour
from .ui import NOWHERE, Button, Choice, Event, Pos, Ui, Widget, draw_button, inside

if TYPE_CHECKING:
    from .canvas import Canvas

#: What one step of the wheel moves a view by: this many keys or lanes, or this many pixels of time.
WHEEL_KEYS = 2
WHEEL_PX = 36
#: What a step of the wheel asks of a view, as `wheel` calls it: to zoom its time, to go sideways, or up and down.
ZOOM, ACROSS, ALONG = "zoom", "across", "along"
#: The keys a keyboard has, from the c of octave 0 to the b of octave 9: what a channel's text can name with `o`.
LOWEST = 12
HIGHEST = 131
#: What the velocity lane is called.
VELOCITY = "VELOCITY"
#: What the items of the menu of a named thing read: the first two, and the last.
RENAME, COPY, REMOVE = "Rename...", "Copy", "Remove"

#: One bar of a velocity lane: its x, how wide its note is, its value from 0 to 1, and whether its note is picked.
Bar4 = tuple[int, int, float, bool]


def _px(value: float) -> int:
    """The whole pixel nearest to a place. A half goes up, so that places a whole number apart stay that far apart."""
    return math.floor(value + 0.5)


def wheel(event: Event) -> tuple[str, int]:
    """What a step of the wheel asks of a view, and how many steps, up counted above 0: the one place it is read.

    With ctrl it is `ZOOM`: the time is zoomed round the mouse. With shift it is `ACROSS`: the view goes
    sideways, along its time or its columns. Alone it is `ALONG`: the view goes up and down, along its keys,
    its lanes or its rows. A view that only goes one way goes that way with `ALONG`.
    """
    if "ctrl" in event.mods:
        return ZOOM, event.wheel
    return (ACROSS if "shift" in event.mods else ALONG), event.wheel


def told(press: str = "", right: str = "", wheel: str = "") -> str:
    """One line of help in the form every view says it: what the left press, the right press and the wheel do.

    They stand in that order, each after its name, and one that does nothing is left out.
    """
    parts = (("press", press), ("right press", right), ("wheel", wheel))
    return "   ".join(f"{name}: {words}" for name, words in parts if words)


#: What the velocity lane says of itself.
VELOCITY_HELP = told("set how hard a note is struck, a drag sets each note it passes", "back to the usual velocity")


def black(pitch: int) -> bool:
    """Whether the key of a pitch is a black one: those are the notes that are spelt with a flat."""
    return "-" in _NAMES[pitch % 12]


def name(pitch: int) -> str:
    """A pitch as a channel's text spells it, flats as `transpose` spells them, with its octave: 60 is c4, 63 e-4."""
    return f"{_NAMES[pitch % 12]}{pitch // 12 - 1}"


def x_at(left: int, start: float, scale: float, moment: float) -> int:
    """The x of a moment on a clock that is drawn from `left`: `start` is the moment there, `scale` its pixels a unit.

    The clock counts in rows or in seconds, as its owner has it. Moments a whole number of pixels apart stay
    that far apart however the clock is scrolled.
    """
    return left + _px(moment * scale) - _px(start * scale)


def moment_at(left: int, start: float, scale: float, x: int) -> float:
    """The moment at the left edge of the pixels at an x: the opposite of `x_at`."""
    return (x - left + _px(start * scale)) / scale


def zoomed(left: int, start: float, scale: float, fresh: float, x: int) -> float:
    """The moment a clock starts at when its scale becomes `fresh`, so that the moment at an x stays at that x."""
    return moment_at(left, start, scale, x) - (x - left) / fresh


def key_rect(rect: Rect, top: float, key_h: int, pitch: int) -> Rect:
    """The rectangle of the key of a pitch on a keyboard that stands in a rectangle, and of its row of a grid there.

    `top` is the pitch whose key has its upper edge at the upper edge of the rectangle, and `key_h` how high a
    key is. `top` can be a fraction: then part of a key is cut off at that edge.
    """
    x, y, w, _ = rect
    return x, y + _px(top * key_h) - pitch * key_h, w, key_h


def pitch_at(rect: Rect, top: float, key_h: int, y: int) -> int:
    """The pitch whose key a y is on, on a keyboard as `key_rect` has it."""
    return -((y - rect[1] - _px(top * key_h)) // key_h)


def name_room(key: Rect, pitch: int) -> Rect:
    """The room of the name of a key, at the right end of the key: its letters, and a pixel free all round them.

    The letters stand in the middle of the key's height. On a key lower than its room the room reaches over
    the keys above and under it, which is what keeps the letters whole.
    """
    x, y, w, h = key
    across = Font.width(name(pitch)) + 2
    return x + w - 4 - across, y + (h - BODY_H) // 2 - 1, across, BODY_H + 2


def wide(widget: Widget) -> int:
    """How wide a button or a choice is that is as wide as what it shows needs. Any other widget is as wide as it is."""
    if isinstance(widget, Choice):
        return sum(Font.width(text) + 2 * theme.PAD for text in widget.options)
    if not isinstance(widget, Button):
        return widget.rect[2]
    icon = theme.MENU_ICON + (theme.GAP if widget.text else 0) if widget.icon else 0
    return Font.width(widget.text) + 2 * theme.PAD + icon + (theme.GAP + theme.DOWN_ARROW if widget.arrow else 0)


def heading_parts(rect: Rect, kind: str, name: str, facts: str, end: int | None = None) -> tuple[Rect, Rect, Rect]:
    """Where the kind, the name and the facts of a heading stand, from the left, each as wide as it is written.

    `end` is the x at which what stands at the right end of the heading begins, or None where nothing does. A
    text that would reach it, or the right end, is given the room there is before it, which can be none.
    """
    x, y, w, h = rect
    limit = (x + w if end is None else end) - theme.BAR_MARGIN
    at, parts = x + theme.BAR_MARGIN, []
    for text, gap in ((kind, theme.HEAD_SPACE), (name, theme.HEAD_FACTS), (facts, 0)):
        room = max(0, min(Font.width(text), limit - at))
        parts.append(within((at, y, room, h), rect))
        at += room + gap
    return parts[0], parts[1], parts[2]


def right_cells(rect: Rect, widths: Sequence[int]) -> list[Rect]:
    """The rectangle of each of some buttons that stand at the right end of a heading, in the order given.

    Each is as wide as asked and as high as a small button, with `theme.BUTTON_GAP` between two. The last ends
    as far from the right end as the buttons are from the top. One the heading has no room for is cut to it.
    """
    x, y, w, h = rect
    inset = max(0, (h - theme.BUTTON_H) // 2)
    at, cells = x + w - inset, []
    for width in reversed(widths):
        at -= width
        cells.append(within((at, y + inset, width, theme.BUTTON_H), rect))
        at -= theme.BUTTON_GAP
    return cells[::-1]


class Heading(Widget):
    """The one-line heading of an editor: at the left what kind of thing is edited, its name and a few facts.

    The kind is in capitals and dim, the name in `colour`, the colour of its kind, and the facts dim. `show`
    gives the name and the facts. `right` lays the buttons that act on the whole thing out from the right end:
    they are widgets of their owner, which adds them to the Ui after the heading. The heading is
    `theme.HEAD_H` high, and takes nothing from the mouse.
    """

    def __init__(self, rect: Rect, kind: str, colour: Colour) -> None:
        super().__init__(rect)
        self.kind = kind
        self.colour = colour
        self.name = ""
        self.facts = ""
        # The x at which the buttons at the right end begin. None while there are none.
        self._end: int | None = None

    def hit(self, pos: Pos) -> bool:
        """Never: a heading takes nothing from the mouse."""
        return False

    def show(self, name: str, facts: str = "") -> None:
        """Show the name of the thing that is edited, and a few facts about it."""
        self.name, self.facts = name, facts

    def right(self, widgets: Sequence[Widget]) -> None:
        """Lay some buttons out from the right end, in the order given, each as `right_cells` has it.

        A button, and a choice, is as wide as what it shows needs. It is called again when the heading has
        another rectangle. The texts at the left end before the first of the buttons.
        """
        cells = right_cells(self.rect, [wide(widget) for widget in widgets])
        for widget, cell in zip(widgets, cells):
            widget.rect = cell
        self._end = min((cell[0] for cell in cells), default=None)

    def draw(self, canvas: Canvas) -> None:
        """Draw the panel of the heading with its outline, and the three texts."""
        canvas.box(self.rect, theme.PANEL)
        canvas.frame(self.rect, theme.LINE)
        texts = ((self.kind, theme.DIM), (self.name, self.colour), (self.facts, theme.DIM))
        for (text, colour), room in zip(texts, heading_parts(self.rect, self.kind, self.name, self.facts, self._end)):
            canvas.text_in(text, room, colour)


@dataclass(frozen=True)
class Slot:
    """One control of a settings bar: the name over it, its widget, and how wide its room is.

    `widget` is the control, or None for room that its owner fills itself, or leaves free. The name is drawn
    while the widget is seen, and with no widget always. `align` is where the name stands over the control. A
    `tall` control has the whole height of the row of the bar, and one that is not is as high as a small
    button. A knob is always tall, and draws its name itself.
    """

    name: str
    widget: Widget | None
    width: int
    align: str = "centre"
    tall: bool = False


def bar_cells(rect: Rect, left: Sequence[int], right: Sequence[int] = ()) -> tuple[list[Rect], list[Rect]]:
    """The room of each control of a settings bar that stands in a rectangle, by the widths the controls ask for.

    Those of `left` stand in a row from the left end and those of `right` in a row up to the right end, both in
    the order given, `theme.BAR_GAP` apart. A room is as high as the row of the bar. Where the bar is too
    narrow, the controls on the right have their room first, the last of them before the others, and then those
    on the left, the first of them before the others. One there is no room for has no width, and no two overlap.
    """
    x, y, w, _ = rect
    top, high = y + theme.BAR_TOP, theme.BAR_ROW_H
    first, at = x + theme.BAR_MARGIN, x + w - theme.BAR_MARGIN
    ends, fits = [], True
    for width in reversed(right):
        fits = fits and at - width >= first
        ends.append(within((at - width, top, width, high) if fits else (at, top, 0, high), rect))
        at -= width + theme.BAR_GAP if fits else 0
    starts, here, fits = [], first, True
    for width in left:
        fits = fits and here + width <= at
        starts.append(within((here, top, width, high) if fits else (here, top, 0, high), rect))
        here += width + theme.BAR_GAP if fits else 0
    return starts, ends[::-1]


class Bar(Widget):
    """The settings bar of an editor: the controls of what is picked in a row, each with its name over it.

    `lay` gives each control its rectangle, a group from the left end and a second group from the right end.
    The bar draws its outline and the names. The controls are widgets of their owner, which adds them to the
    Ui after the bar. A knob shows its name and its value itself, so the name of a knob is the knob's to draw.
    The bar is `theme.BAR_H` high, and takes nothing from the mouse.
    """

    def __init__(self, rect: Rect) -> None:
        super().__init__(rect)
        # The names the bar draws: each with where it stands, how, and the widget it is the name of.
        self._names: list[tuple[str, Rect, str, Widget | None]] = []

    def hit(self, pos: Pos) -> bool:
        """Never: a bar takes nothing from the mouse."""
        return False

    def lay(self, left: Sequence[Slot], right: Sequence[Slot] = ()) -> list[Rect]:
        """Give each control its rectangle in the bar as it stands now, and give them back: `left`, then `right`.

        The rooms are those of `bar_cells`. A tall control has all of its room. Any other is as high as a small
        button and stands in the middle of the height, with its name over it on the line a knob has its name
        on. A knob is told its name. A control with no room has a rectangle of no width: its owner hides it.
        """
        rooms = bar_cells(self.rect, [slot.width for slot in left], [slot.width for slot in right])
        self._names = []
        rects = []
        for slot, room in zip((*left, *right), (*rooms[0], *rooms[1])):
            x, y, w, h = room
            knob = isinstance(slot.widget, Knob)
            if not (knob or slot.tall):
                room = within((x, y + (theme.BAR_ROW_H - theme.BUTTON_H) // 2, w, theme.BUTTON_H), self.rect)
            if slot.widget is not None:
                slot.widget.rect = room
            if knob:
                slot.widget.text = slot.name
            elif slot.name and w > 0:
                self._names.append((slot.name, within((x, y, w, Font.LINE), self.rect), slot.align, slot.widget))
            rects.append(room)
        return rects

    def draw(self, canvas: Canvas) -> None:
        """Draw the panel of the bar with its outline, and the name of each control that is seen."""
        canvas.box(self.rect, theme.PANEL)
        canvas.frame(self.rect, theme.LINE)
        for text, room, align, widget in self._names:
            if widget is None or widget.visible:
                canvas.text_in(text, room, theme.DIM, align)


def _typed(widget: Keys | ScrollBar | Velocity, event: Event) -> bool:
    """Hand a key a part of an editor has the focus for on to the editor: whether the editor used it."""
    return widget.typed is not None and bool(widget.typed(event))


class Keys(Widget):
    """The upright keyboard beside a grid: a key for each pitch, the higher the further up, and every c named.

    `top` is the pitch whose key has its upper edge at the upper edge of the rectangle, and `key_h` how high a
    key is: its owner sets them to those of its grid, so the keys stand beside their rows. The keyboard never
    fits itself to notes. A press of the left button on a key calls `heard` with its pitch, and the key is lit
    until the button is let go. `low` and `high` are the lowest and the highest key there is.

    The help line on a key says that a press lets it be heard, and after that `wheel_help`, which is what the
    wheel does there: the keyboard leaves the wheel to what lies under it. A key of the computer that it is
    handed, which it is while it has the focus, goes on to `typed`, which is its editor's.
    """

    def __init__(self, rect: Rect, heard: Callable[[int], object] | None = None, help: str = "",
                 wheel_help: str = "") -> None:
        super().__init__(rect, help)
        self.heard = heard
        self.wheel_help = wheel_help
        self.typed: Callable[[Event], object] | None = None
        self.top = 72.0
        self.key_h = theme.KEY_H
        self.low, self.high = LOWEST, HIGHEST
        #: The pitch of the key that is held down by the mouse, or None.
        self.down: int | None = None
        self._mouse: Pos = NOWHERE

    @property
    def help(self) -> str:
        """What a press on the key under the mouse does, or off the keys the line the keyboard was made with."""
        pitch = self.at(self._mouse)
        return self._help if pitch is None else told(f"hear {name(pitch)}", wheel=self.wheel_help)

    @help.setter
    def help(self, line: str) -> None:
        self._help = line

    def at(self, pos: Pos) -> int | None:
        """The pitch of the key a position is on, or None where it is on no key."""
        pitch = pitch_at(self.rect, self.top, self.key_h, pos[1])
        return pitch if inside(self.rect, pos) and self.low <= pitch <= self.high else None

    def pitches(self) -> range:
        """The pitches of the keys of which at least a line of pixels is in the rectangle, from the lowest."""
        _, y, _, h = self.rect
        if h <= 0:
            return range(0)
        return range(max(self.low, pitch_at(self.rect, self.top, self.key_h, y + h - 1)),
                     min(self.high, pitch_at(self.rect, self.top, self.key_h, y)) + 1)

    def on(self, event: Event, ui: Ui) -> bool:
        """Use a press of the left button on a key, which is heard, and hand a key of the computer on."""
        if event.kind == "key":
            return _typed(self, event)
        if event.kind == "wheel":
            return False
        self._mouse = event.pos
        if event.kind == "up" and event.button == 1:
            self.down = None
        if event.kind != "down" or event.button != 1 or ui.held is not None:
            return False
        self.down = self.at(event.pos)
        if self.down is not None and self.heard is not None:
            self.heard(self.down)
        return self.down is not None

    def draw(self, canvas: Canvas) -> None:
        """Draw the keys: each a light block, a black one dark for most of its length, and each c named whole.

        The key that is held is in the colour of what is on, and the one under the mouse is a shade apart.
        The names are drawn over the keys, each in a room cleared to the light of its key.
        """
        x, _, w, _ = self.rect
        over = self.at(self._mouse) if self.hot else None
        dark = int(w * theme.KEY_BLACK)
        with canvas.clip(self.rect):
            canvas.box(self.rect, theme.PANEL)
            faces = {}
            for pitch in self.pitches():
                key = (x, key_rect(self.rect, self.top, self.key_h, pitch)[1], w - 1, self.key_h)
                held, lit = pitch == self.down, pitch == over
                faces[pitch] = theme.ACTIVE if held else theme.shade(theme.TEXT, theme.KEY_HOT) if lit else theme.TEXT
                canvas.box(key, faces[pitch])
                if self.key_h >= theme.KEY_LINES_FROM:
                    canvas.frame(key, theme.BACKGROUND)
                if black(pitch):
                    canvas.box((x, key[1], dark, self.key_h),
                               theme.ACTIVE if held else theme.LIT if lit else theme.BACKGROUND)
            for pitch, face in faces.items():
                if pitch % 12 == 0:
                    room = name_room((x, key_rect(self.rect, self.top, self.key_h, pitch)[1], w, self.key_h), pitch)
                    canvas.box(room, face)
                    canvas.text_in(name(pitch), room, theme.BACKGROUND, "centre")


class ScrollBar(Widget):
    """The thin bar along one side of a grid, which shows what part of the grid is in view, and can be dragged.

    It stands upright when its rectangle is taller than wide, and lies otherwise. `scroll` has the numbers, in
    pixels: its owner gives them with `show`, as they are along the side the bar is on. A press of the left
    button on the thumb takes hold of it, and a drag moves it. A press beside the thumb moves the view by its
    own length that way. `moved` is called with the new offset each time the hand changed it, and the owner
    moves its view there. While everything is in view there is no thumb, and a press does nothing.

    The bar leaves the wheel to what lies under it. A key of the computer that it is handed, which it is while
    it has the focus, goes on to `typed`, which is its editor's.
    """

    def __init__(self, rect: Rect, moved: Callable[[int], object] | None = None, help: str = "") -> None:
        super().__init__(rect, help)
        self.moved = moved
        self.typed: Callable[[Event], object] | None = None
        self.scroll = Scroll()
        # While the thumb is dragged: how far along the mouse took hold of it, and the offset there was then.
        self._grab: tuple[int, int] | None = None

    @property
    def upright(self) -> bool:
        """Whether it stands: its rectangle is taller than wide."""
        return self.rect[3] > self.rect[2]

    @property
    def busy(self) -> bool:
        """Whether the thumb is being dragged."""
        return self._grab is not None

    def show(self, content: int, view: int, offset: int) -> None:
        """Take how long the whole is, how much of it is in view, and how much of it lies before the view."""
        self.scroll.content, self.scroll.view = content, view
        self.scroll.to(offset)

    def _track(self) -> tuple[int, int]:
        """Where the track of the bar starts along the side it is on, and how long it is."""
        x, y, w, h = self.rect
        return (y, h) if self.upright else (x, w)

    def thumb(self) -> Rect | None:
        """The rectangle of the thumb: the part of the bar that stands for what is in view. None when all is."""
        x, y, w, h = self.rect
        found = self.scroll.thumb(self._track()[1])
        if found is None:
            return None
        return (x, y + found[0], w, found[1]) if self.upright else (x + found[0], y, found[1], h)

    def on(self, event: Event, ui: Ui) -> bool:
        """Use a press of the left button while there is a thumb, carry a drag of the thumb, and hand a key on."""
        if event.kind == "key":
            return _typed(self, event)
        if event.kind == "wheel":
            return False
        start, length = self._track()
        place = event.pos[1 if self.upright else 0]
        before = self.scroll.offset
        if event.kind == "down":
            found = self.scroll.thumb(length)
            if event.button != 1 or found is None or ui.held is not None:
                return False
            if found[0] <= place - start < found[0] + found[1]:
                self._grab = (place, before)
            else:
                self.scroll.to(before + (self.scroll.view if place - start >= found[0] else -self.scroll.view))
        elif event.kind == "move" and self._grab is not None:
            self.scroll.to(self._grab[1])
            self.scroll.drag(place - self._grab[0], length)
        elif event.kind == "up" and event.button == 1:
            self._grab = None
        if self.scroll.offset != before and self.moved is not None:
            self.moved(self.scroll.offset)
        return event.kind == "down"

    def draw(self, canvas: Canvas) -> None:
        """Draw the track, and on it the thumb, which is lighter under the mouse and while it is dragged."""
        canvas.box(self.rect, theme.BACKGROUND)
        thumb = self.thumb()
        if thumb is not None:
            canvas.box(thumb, theme.DIM if self.hot or self.busy else theme.LINE)


def velocity_bar(rect: Rect, x: int, width: int, value: float) -> Rect:
    """The rectangle of a bar in a velocity lane that stands in a rectangle: nothing high at 0, and the most at 1.

    It starts at `x`, where its note does, and stands on the floor of the lane. It is `theme.VELOCITY_W` wide
    where its note is not narrower. `theme.VELOCITY_PAD` is left free at the floor and at the top.
    """
    _, y, _, h = rect
    reach = _px(min(max(value, 0.0), 1.0) * max(0, h - 2 * theme.VELOCITY_PAD))
    return x, y + h - theme.VELOCITY_PAD - reach, max(1, min(width, theme.VELOCITY_W)), reach


def velocity_at(rect: Rect, y: int) -> float:
    """The value whose bar in a lane reaches up to a y: 0 at the floor and under it, and 1 at the top and over it."""
    _, top, _, h = rect
    room = h - 2 * theme.VELOCITY_PAD
    return min(max((top + h - theme.VELOCITY_PAD - y) / room, 0.0), 1.0) if room > 0 else 0.0


def passed(items: Sequence[Bar4], start: Pos, end: Pos) -> list[tuple[int, int]]:
    """The bars of a lane that the mouse passes on its way from a position to another, each with a y for its value.

    A bar is passed where the mouse is over its note, from the x of the bar for the width of the note. They
    come in the order the mouse comes to them, each as its index among the items and the y the mouse had where
    it was last over that note. Where one of them is picked, only the picked ones are passed.
    """
    (x0, y0), (x1, y1) = start, end
    low, high = min(x0, x1), max(x0, x1)
    over = [(index, min(max(x1, x), x + max(w, 1) - 1)) for index, (x, w, _, _) in enumerate(items)
            if x <= high and low < x + max(w, 1)]
    if any(items[index][3] for index, _ in over):
        over = [found for found in over if items[found[0]][3]]
    over.sort(key=lambda found: abs(found[1] - x0))
    return [(index, y1 if x1 == x0 else _px(y0 + (y1 - y0) * (near - x0) / (x1 - x0))) for index, near in over]


class Velocity(Widget):
    """The lane at the foot of an editor: a bar for each note, as high as the note is struck hard.

    `show` gives the bars, each as its x, the width of its note, its value from 0 to 1 and whether its note is
    picked. A press of the left button sets the value of the bar under it from its height, and a drag sets
    that of each bar it passes: `changed` is called with the index of the bar and the value, and the owner
    shows the lane what it made of it. `settled` is called when the button is let go, for an owner that makes
    one step of a drag. A press of the right button on a bar calls `changed` with its index and None, which is
    for the usual value, and then `settled`. Where a picked note is under the mouse, only picked ones are set.

    `colour` is that of the bars, and a picked one is in the colour of text. `align` is the end of the lane its
    name stands at. `span` is the stretch of the lane the bars stand in, as its x and its width: that of the
    grid over it, which the owner sets. Outside it no bar is drawn and no press is used. With None it is all
    of the lane. `playhead` is the x of what plays now, or None: its owner sets it. The lane leaves the wheel
    alone. A key of the computer that it is handed, which it is while it has the focus, goes on to `typed`,
    which is its editor's.
    """

    def __init__(self, rect: Rect, changed: Callable[[int, float | None], object] | None = None,
                 settled: Callable[[], object] | None = None, colour: Colour = theme.SONG,
                 help: str = VELOCITY_HELP, align: str = "left") -> None:
        super().__init__(rect, help)
        self.changed = changed
        self.settled = settled
        self.colour = colour
        self.align = align
        self.typed: Callable[[Event], object] | None = None
        self.items: tuple[Bar4, ...] = ()
        self.span: tuple[int, int] | None = None
        self.playhead: int | None = None
        # Where the mouse was last in the drag that is under way.
        self._at: Pos | None = None

    @property
    def busy(self) -> bool:
        """Whether a drag is under way: from a press of the left button to its release, or until it is dropped."""
        return self._at is not None

    @property
    def room(self) -> Rect:
        """Where the bars stand: inside the outline of the lane, and no further to the sides than `span` goes."""
        x, y, w, h = self.rect
        inner = (x + 1, y + 1, max(0, w - 2), max(0, h - 2))
        return inner if self.span is None else within((self.span[0], y + 1, self.span[1], h - 2), inner)

    def show(self, items: Sequence[Bar4]) -> None:
        """Take the bars of the lane: each its x, the width of its note, its value and whether it is picked."""
        self.items = tuple(items)

    def drop(self) -> None:
        """Let go of the drag that is under way, with nothing told of it."""
        self._at = None

    def _set(self, start: Pos, end: Pos) -> None:
        """Tell of each bar the mouse passed between two positions, with the value its height gives."""
        for index, y in passed(self.items, start, end):
            if self.changed is not None:
                self.changed(index, velocity_at(self.rect, y))

    def on(self, event: Event, ui: Ui) -> bool:
        """Use a press of the left button and the drag after it, and one of the right button on a bar."""
        if event.kind == "key":
            return _typed(self, event)
        if event.kind == "wheel":
            return False
        left, _, across, _ = self.room
        # A drag that leaves the stretch of the bars goes on at its edge.
        pos = (min(max(event.pos[0], left), left + across - 1), event.pos[1])
        if event.kind == "down":
            if ui.held is not None or event.button not in (1, 3) or not inside(self.room, event.pos):
                return False
            if event.button == 1:
                self._at = pos
                self._set(pos, pos)
                return True
            found = passed(self.items, pos, pos)
            for index, _ in found:
                if self.changed is not None:
                    self.changed(index, None)
            if found and self.settled is not None:
                self.settled()
            return bool(found)
        if self._at is not None:
            self._set(self._at, pos)
            self._at = pos
            if event.kind == "up" and event.button == 1:
                self._at = None
                if self.settled is not None:
                    self.settled()
        return False

    def draw(self, canvas: Canvas) -> None:
        """Draw the panel of the lane with its outline and its name, the bars, the picked ones last, the playhead."""
        x, y, w, h = self.rect
        canvas.box(self.rect, theme.PANEL)
        with canvas.clip(self.room):
            for at, width, value, picked in sorted(self.items, key=lambda item: item[3]):
                ink = theme.DIM if not self.enabled else theme.TEXT if picked else self.colour
                canvas.box(velocity_bar(self.rect, at, width, value), ink)
            if self.playhead is not None:
                canvas.vline(self.playhead, y, h, theme.ACTIVE)
        canvas.frame(self.rect, theme.LINE)
        canvas.text_in(VELOCITY, within((x + theme.BAR_MARGIN, y + theme.BAR_TOP, w - 2 * theme.BAR_MARGIN, Font.LINE),
                                        self.rect), theme.DIM, self.align)


class Plus(Button):
    """The small button that stands after the last of anything that can grow: a press adds one more.

    It is `theme.PLUS` wide and high where its owner lays it so, and says the line of help it is given.
    """

    def __init__(self, rect: Rect, action: Callable[[], object] | None = None, help: str = "") -> None:
        super().__init__(rect, action=action, help=help)

    def draw(self, canvas: Canvas) -> None:
        """Draw the button, and on it a plus of two strokes. Dim when not enabled."""
        x, y, w, h = self.rect
        pressed = self.hot and self.held
        draw_button(canvas, self.rect, hot=self.hot, pressed=pressed)
        ink, long = theme.TEXT if self.enabled else theme.DIM, theme.PLUS_STROKE
        cx, cy = x + w // 2, y + h // 2 + int(pressed)
        with canvas.clip(self.rect):
            canvas.hline(cx - long // 2, cy, long, ink)
            canvas.vline(cx, cy - long // 2, long, ink)


def thing_menu(rename: object = None, copy: object = None, special: Sequence[Item | None] = (),
               remove: object = None) -> list[Item | None]:
    """The menu of a thing with a name, in the order every such menu has: to rename it, to copy it, and to remove it.

    `rename`, `copy` and `remove` are what the three items stand for: what the owner of the menu is told when
    one is chosen. One that is None cannot be done, and its item is shown dim: it is never left out. `special`
    is the items that only this kind of thing has, which stand after `Copy`. A line stands before `Remove`.
    """
    return [Item(RENAME, rename, enabled=rename is not None), Item(COPY, copy, enabled=copy is not None), *special,
            None, Item(REMOVE, remove, enabled=remove is not None)]
