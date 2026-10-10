"""The widgets that hold a number: a knob, a slider, a meter and a stepper, and the sums they stand on."""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

from ..errors import RowSynthError
from . import theme
from .font import BODY_H, Font
from .layout import Rect
from .theme import Colour
from .ui import Event, Pos, Ui, Widget, share

if TYPE_CHECKING:
    from .canvas import Canvas

#: The most places after the point a number is shown with when its dial is given no way to show it.
PLACES = 3


def _whole(number: float) -> int:
    """The nearest whole number, a half going up."""
    return math.floor(number + 0.5)


def fraction(value: float, low: float, high: float) -> float:
    """How far a value is from `low` to `high`, from 0 to 1. One outside the range counts as its nearer end."""
    return min(1.0, max(0.0, (value - low) / (high - low)))


def kept(value: float, low: float, high: float, steps: int | None = None) -> float:
    """A value kept inside its range and, when the range is cut into `steps` equal steps, put on the nearest."""
    value = min(high, max(low, value))
    if steps is None:
        return value
    return low + _whole((value - low) * steps / (high - low)) * (high - low) / steps


def dragged(start: float, pixels: float, low: float, high: float, travel: int = theme.DRAG_TRAVEL,
            fine: bool = False, steps: int | None = None) -> float:
    """The value a drag of `pixels` leaves a dial at that was at `start`: `travel` pixels are the whole range.

    With `fine` the drag goes at a tenth of that speed. The value is kept in the range and on the steps.
    """
    speed = theme.FINE if fine else 1
    return kept(start + pixels * speed * (high - low) / travel, low, high, steps)


def wheeled(value: float, clicks: int, low: float, high: float, steps: int | None = None) -> float:
    """The value `clicks` steps of the wheel leave a dial at: a step each, or a hundredth of the range without."""
    return kept(value + clicks * (high - low) / (steps or theme.WHEEL_STEPS), low, high, steps)


def plain(value: float) -> str:
    """A number as text with no more places than it needs: three at most, and no noughts at its end."""
    text = f"{value:.{PLACES}f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def handle_at(value: float, low: float, high: float, run: int) -> int:
    """Where a slider's handle sits for a value: how many pixels from the low end of its track, `run` at most."""
    return _whole(fraction(value, low, high) * run)


def value_at(place: int, low: float, high: float, run: int) -> float:
    """The value at which a slider's handle sits `place` pixels from the low end: the opposite of `handle_at`."""
    return low + min(1.0, max(0.0, place / run)) * (high - low)


def db_height(db: float) -> float:
    """How far up a meter a level in decibels is, from 0 at the floor to 1 at 0 dB."""
    return min(1.0, max(0.0, 1 - db / theme.METER_FLOOR_DB))


def meter_height(level: float) -> float:
    """How far up a meter a level from 0 to 1 is, from 0 to 1. The meter is even in decibels."""
    return db_height(20 * math.log10(level)) if level > 0 else 0.0


def zones(rows: int) -> list[tuple[Colour, int, int]]:
    """The colours of a bar `rows` pixels high, from its foot: each with its first row and the one after its last."""
    warm, hot = (_whole(db_height(db) * rows) for db in (theme.METER_WARM_DB, theme.METER_HOT_DB))
    return [(theme.GOOD, 0, warm), (theme.MUSTARD, warm, hot), (theme.FAULT, hot, rows)]


def bars(rect: Rect, count: int) -> list[Rect]:
    """The rectangle of each of a meter's bars: side by side inside its outline, with a gap between two."""
    x, y, w, h = rect
    cells, left = [], x + 1
    for width in share(w - 2 - (count - 1) * theme.METER_GAP, [1] * count):
        cells.append((left, y + 1, width, h - 2))
        left += width + theme.METER_GAP
    return cells


def lit_marks(value: float, low: float, high: float, marks: int = theme.KNOB_MARKS) -> int:
    """How many of a knob's marks are lit, counted from the low end: those the value has reached.

    The marks are spread evenly over the travel, the first at its low end, so the first is always lit.
    """
    return math.floor(fraction(value, low, high) * (marks - 1) + 1e-9) + 1


def around(turn: float, reach: float) -> Pos:
    """Where a point `reach` pixels from the centre of a knob is, as (right, down) of it, at a turn from 0 to 1.

    At 0 the point is at seven o'clock. It goes over the top, and at 1 it is at five.
    """
    angle = math.radians(theme.KNOB_START - turn * theme.KNOB_SWEEP)
    return _whole(math.cos(angle) * reach), -_whole(math.sin(angle) * reach)


@lru_cache(maxsize=None)
def disc(size: int) -> tuple[tuple[int, int], ...]:
    """The rows of a disc `size` pixels across, from its top: for each, its first pixel and how many it has."""
    middle = (size - 1) / 2
    rows = []
    for row in range(size):
        inside = [column for column in range(size)
                  if (column - middle) ** 2 + (row - middle) ** 2 <= (size / 2) ** 2]
        rows.append((inside[0], len(inside)))
    return tuple(rows)


@lru_cache(maxsize=None)
def core(size: int) -> tuple[tuple[int, int], ...]:
    """As `disc`, for the disc without its rim: the pixels that have the disc on all four sides of them."""
    rows = disc(size)
    spans = [(0, 0)]
    for above, row, below in zip(rows, rows[1:], rows[2:]):
        first = max(row[0] + 1, above[0], below[0])
        end = min(row[0] + row[1] - 1, above[0] + above[1], below[0] + below[1])
        spans.append((first, end - first))
    return (*spans, (0, 0))


def knob_parts(rect: Rect) -> tuple[Rect, Pos, Rect]:
    """Where a knob's parts go in its rectangle: the line of its label, its centre, and the line of its value.

    The knob is in the middle. The two lines are as wide as the rectangle, the label just clear of the marks
    above and the value as far below: 33 pixels from the top of the one to the foot of the other.
    """
    x, y, w, h = rect
    cx, cy = x + w // 2, y + h // 2
    clear = theme.KNOB_MARKS_AT + theme.KNOB_GAP
    air = (Font.LINE - BODY_H) // 2
    return (x, cy - clear - BODY_H - air, w, Font.LINE), (cx, cy), (x, cy + clear + 1 - air, w, Font.LINE)


class Dial(Widget):
    """What a knob and a slider share: a `value` from `low` to `high`, and the ways the mouse changes it.

    A drag with the left button changes it, at a tenth of the speed while shift is held. A step of the wheel
    moves it a step. A right click puts it back to `default`, when it has one. `steps` is how many equal steps
    the range is cut into: the value is then always on one of them. `changed` is called with the new value
    each time the value changes, and not when it stays as it was.
    """

    def __init__(self, rect: Rect, value: float, low: float, high: float,
                 changed: Callable[[float], object] | None = None, default: float | None = None,
                 steps: int | None = None, colour: Colour = theme.ACTIVE, help: str = "") -> None:
        super().__init__(rect, help)
        if not low < high:
            raise RowSynthError(f"a dial goes from a low value to a higher one, not from {low:g} to {high:g}")
        if steps is not None and (not isinstance(steps, int) or steps < 1):
            raise RowSynthError(f"the steps of a dial are a whole number of 1 or more, not {steps!r}")
        self.value = value
        self.low = low
        self.high = high
        self.changed = changed
        self.default = default
        self.steps = steps
        self.colour = colour
        #: Where the drag that is going on started: the value, the place, and whether it was fine.
        self._drag: tuple[float, int, bool] | None = None
        self._at = 0

    def set(self, value: float) -> None:
        """Give it a value, kept in its range and on its steps, and tell if that is not the one it had."""
        value = kept(value, self.low, self.high, self.steps)
        if value != self.value:
            self.value = value
            if self.changed is not None:
                self.changed(value)

    def place(self, pos: Pos) -> int:
        """How far along the drag a position is, in pixels: further is a higher value. Here, further up."""
        return -pos[1]

    def travel(self) -> int:
        """How many pixels of drag take it through its whole range."""
        return theme.DRAG_TRAVEL

    def press(self, pos: Pos) -> float:
        """Take the press that starts a drag, and give the value the drag goes on from."""
        return self.value

    def on(self, event: Event, ui: Ui) -> bool:
        """Use the wheel, a press of the left button and the drag after it, and a right click if it has a default."""
        if event.kind == "wheel":
            self.set(wheeled(self.value, event.wheel, self.low, self.high, self.steps))
            return True
        if event.kind == "down" and event.button == 3:
            if self.default is not None and self._drag is None:
                self.set(self.default)
            return self.default is not None
        if event.kind == "down" and event.button == 1:
            self._drag = (self.press(event.pos), self.place(event.pos), "shift" in event.mods)
            self._at = self.place(event.pos)
            return True
        if event.kind == "up" and event.button == 1:
            self._drag = None
        elif event.kind == "move" and self._drag is not None:
            start, place, fine = self._drag
            if ("shift" in event.mods) != fine:
                # Shift went down or up: the drag goes on from where it has come to, so the value does not jump.
                start, place, fine = self.value, self._at, not fine
                self._drag = (start, place, fine)
            self._at = self.place(event.pos)
            self.set(dragged(start, self._at - place, self.low, self.high, self.travel(), fine, self.steps))
        return False


class Knob(Dial):
    """A round knob with a pointer, and round it the marks of its travel, lit in `colour` up to its value.

    Dragging up raises it and dragging down lowers it. `text` is its label, drawn above it in capitals. Its
    value is drawn below it as `show(value)` gives it: without `show`, the number with no more places than it
    needs. The rectangle is the whole of it, with the knob in the middle.
    """

    def __init__(self, rect: Rect, value: float, low: float, high: float,
                 changed: Callable[[float], object] | None = None, default: float | None = None,
                 steps: int | None = None, colour: Colour = theme.ACTIVE, text: str = "",
                 show: Callable[[float], str] | None = None, help: str = "") -> None:
        super().__init__(rect, value, low, high, changed, default, steps, colour, help)
        self.text = text
        self.show = show

    def draw(self, canvas: Canvas) -> None:
        """Draw the knob, its marks, its pointer, its label and its value. Dim when not enabled."""
        label, (cx, cy), under = knob_parts(self.rect)
        lit, ink = (self.colour, theme.TEXT) if self.enabled else (theme.DIM, theme.DIM)
        left, top = cx - theme.KNOB // 2, cy - theme.KNOB // 2
        for colour, rows in ((theme.LINE, disc(theme.KNOB)), (theme.LIT, core(theme.KNOB))):
            for row, (first, width) in enumerate(rows):
                canvas.hline(left + first, top + row, width, colour)
        reached = lit_marks(self.value, self.low, self.high)
        for mark in range(theme.KNOB_MARKS):
            dx, dy = around(mark / (theme.KNOB_MARKS - 1), theme.KNOB_MARKS_AT)
            canvas.box((cx + dx, cy + dy, 1, 1), lit if mark < reached else theme.LINE)
        for reach in theme.KNOB_POINTER:
            dx, dy = around(fraction(self.value, self.low, self.high), reach)
            canvas.box((cx + dx, cy + dy, 1, 1), ink)
        canvas.text_in(self.text.upper(), label, theme.DIM, "centre")
        canvas.text_in((self.show or plain)(self.value), under, ink, "centre")


class Slider(Dial):
    """A track with a handle on it: upright when the rectangle is taller than wide, and lying otherwise.

    The low end is at the foot of an upright one and at the left of a lying one. A press on the track puts the
    handle there, and the drag goes on from it. Lying, the track is lit from its low end to the handle.
    """

    @property
    def upright(self) -> bool:
        """Whether it stands: its rectangle is taller than wide."""
        return self.rect[3] > self.rect[2]

    def _size(self) -> int:
        """How long the handle is along the track."""
        return theme.HANDLE_UPRIGHT if self.upright else theme.HANDLE_LYING

    def travel(self) -> int:
        """How many pixels the handle can move: the length of the track less the handle, and 1 at least."""
        return max(1, (self.rect[3] if self.upright else self.rect[2]) - self._size())

    def place(self, pos: Pos) -> int:
        """How many pixels a position is from the low end of the track."""
        x, y, _, h = self.rect
        return y + h - 1 - pos[1] if self.upright else pos[0] - x

    def press(self, pos: Pos) -> float:
        """Put the middle of the handle where the press is, and give the value that is exactly there."""
        value = value_at(self.place(pos) - self._size() // 2, self.low, self.high, self.travel())
        self.set(value)
        return value

    def handle(self) -> Rect:
        """The rectangle of the handle for the value it has now."""
        x, y, w, h = self.rect
        size, at = self._size(), handle_at(self.value, self.low, self.high, self.travel())
        return (x, y + h - size - at, w, size) if self.upright else (x + at, y, size, h)

    def draw(self, canvas: Canvas) -> None:
        """Draw the track, and on it the handle in the slider's colour. Dim when not enabled."""
        x, y, w, h = self.rect
        ink = self.colour if self.enabled else theme.DIM
        handle = self.handle()
        if self.upright:
            canvas.box((x + (w - theme.TRACK) // 2, y, theme.TRACK, h), theme.BACKGROUND)
        else:
            track = y + (h - theme.TRACK) // 2
            canvas.box((x, track, w, theme.TRACK), theme.BACKGROUND)
            canvas.box((x, track, handle[0] - x, theme.TRACK), theme.shade(ink, theme.TRAVEL))
        canvas.box(handle, ink)
        canvas.frame(handle, theme.BACKGROUND)


class Meter(Widget):
    """One bar or two side by side that show a level each, even in decibels. It takes nothing from the mouse.

    `levels` is one number or two from 0 to 1, set by whoever owns the meter, each frame. The highest a bar has
    been is kept as a thin line that falls back slowly: the owner calls `fall` with the time that has passed.
    A bar is green, mustard in its upper part and red at the very top. `limits` are levels drawn as lines
    across the meter.
    """

    def __init__(self, rect: Rect, limits: Sequence[float] = ()) -> None:
        super().__init__(rect)
        self.limits = tuple(limits)
        self._levels: tuple[float, ...] = (0.0,)
        #: How far up the meter the kept line of each bar is, from 0 to 1.
        self.tops = [0.0]

    @property
    def levels(self) -> tuple[float, ...]:
        """The level of each bar."""
        return self._levels

    @levels.setter
    def levels(self, levels: Sequence[float]) -> None:
        levels = tuple(levels)
        if len(levels) not in (1, 2):
            raise RowSynthError(f"a meter has one bar or two, not {len(levels)}")
        if len(levels) != len(self._levels):
            self.tops = [0.0] * len(levels)
        self._levels = levels
        self.fall(0)

    def fall(self, seconds: float) -> None:
        """Let the kept lines fall for the time that has passed. None goes under the bar it belongs to."""
        drop = seconds * theme.METER_FALL_DB / -theme.METER_FLOOR_DB
        self.tops = [max(meter_height(level), top - drop) for level, top in zip(self._levels, self.tops)]

    def hit(self, pos: Pos) -> bool:
        """Never: a meter takes nothing from the mouse."""
        return False

    def draw(self, canvas: Canvas) -> None:
        """Draw the bars in their colours, the kept line of each, and the limits across them all."""
        x, y, w, h = self.rect
        canvas.box(self.rect, theme.BACKGROUND)
        canvas.frame(self.rect, theme.LINE)
        for (left, top, width, rows), level, kept_at in zip(bars(self.rect, len(self._levels)), self._levels,
                                                            self.tops):
            lit, line = _whole(meter_height(level) * rows), _whole(kept_at * rows)
            for colour, first, end in zones(rows):
                if lit > first:
                    canvas.box((left, top + rows - min(lit, end), width, min(lit, end) - first), colour)
                if first < line <= end:
                    canvas.hline(left, top + rows - line, width, colour)
        for limit in self.limits:
            canvas.hline(x + 1, y + h - 1 - max(1, _whole(meter_height(limit) * (h - 2))), w - 2, theme.TEXT)


class Stepper(Widget):
    """One of a list of values, shown as text between two small arrows: `picked` is its index.

    A click on the left half goes one back and one on the right half goes one on, and so do steps of the wheel,
    down and up. It stops at the ends, where the arrow that leads no further is dim. `changed` is called with
    the index of the new one when another is picked.
    """

    def __init__(self, rect: Rect, options: Sequence[object], picked: int = 0,
                 changed: Callable[[int], object] | None = None, help: str = "") -> None:
        super().__init__(rect, help)
        self.options = tuple(options)
        if not self.options:
            raise RowSynthError("a stepper needs at least one option")
        if not 0 <= picked < len(self.options):
            raise RowSynthError(f"a stepper of {len(self.options)} options has no option {picked}")
        self.picked = picked
        self.changed = changed

    def step(self, by: int) -> None:
        """Go `by` options on, or back when `by` is below 0, no further than the ends."""
        index = min(len(self.options) - 1, max(0, self.picked + by))
        if index != self.picked:
            self.picked = index
            if self.changed is not None:
                self.changed(index)

    def on(self, event: Event, ui: Ui) -> bool:
        """Use the wheel, and a press of the left button: its release steps by the half it is on."""
        if event.kind == "wheel":
            self.step(event.wheel)
            return True
        if event.kind == "down":
            return event.button == 1
        if event.kind == "up" and event.button == 1 and ui.held is self and self.hit(event.pos):
            self.step(-1 if event.pos[0] < self.rect[0] + self.rect[2] // 2 else 1)
        return False

    def draw(self, canvas: Canvas) -> None:
        """Draw the field, an arrow at each end and the option between them. Dim when not enabled."""
        x, y, w, h = self.rect
        canvas.box(self.rect, theme.BACKGROUND)
        canvas.frame(self.rect, theme.LINE)
        middle = y + (h - BODY_H) // 2 + BODY_H // 2
        ends = ((self.picked == 0, x + theme.PAD, 1),
                (self.picked == len(self.options) - 1, x + w - theme.PAD - 1, -1))
        for at_end, tip, back in ends:
            ink = theme.DIM if at_end or not self.enabled else theme.TEXT
            for column in range(theme.ARROW):
                canvas.vline(tip + back * column, middle - column, 2 * column + 1, ink)
        side = theme.PAD + theme.ARROW + theme.GAP
        canvas.text_in(str(self.options[self.picked]), (x + side, y, w - 2 * side, h),
                       theme.TEXT if self.enabled else theme.DIM, "centre")
