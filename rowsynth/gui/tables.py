"""A step table of an instrument as a row of bars drawn with the mouse: its shapes in the file, its sums, its widget."""
from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from .. import note
from ..errors import RowSynthError
from . import kit, theme
from .dials import plain
from .layout import Rect
from .theme import Colour
from .ui import NOWHERE, Event, Pos, Ui, Widget, inside

if TYPE_CHECKING:
    from .canvas import Canvas

#: How high the strip under the field is, in which the bracket of the loop lies.
LOOP_H = 5
#: From the outside of the field to the bars: its outline, and a pixel of air.
INSET = 2
#: How wide a bar is at the most and at the least, and how far two bars are apart.
BAR_MAX = 9
BAR_MIN = 2
GAP = 1
#: The handle after the last bar: how wide it is, and how high the part of it that is drawn.
HANDLE_W = 3
HANDLE_H = 9
#: The most steps the handle makes a table.
MOST = 64
#: How dark the steps that repeat are drawn after the last bar, as a share of the colour of the bars.
FAINT = 0.4
#: The most places after the point of a value that the mouse sets in a table of values that are not whole.
PLACES = 3

#: What a press can begin, as `begins` calls it.
DRAW, ZERO, SIZE, LOOP = "draw", "zero", "size", "loop"
#: What the help line says the right press does on a step, and the left press on the handle after the last bar.
ZERO_HELP = "back to zero"
SIZE_HELP = "drag right for more steps, left for fewer"
#: What it says the left press does under the bars: on a table with a loop, and before the step of one with none.
LOOP_HELP = "drag the start of the loop, off the left end to take the loop away"
LOOP_FROM = "loop the table from"

Steps = tuple[float, ...]
#: A table taken apart: its steps, and the index it goes round from, or None when it holds its last step.
Parts = tuple[Steps, int | None]


def _px(value: float) -> int:
    """The nearest whole number, a half going up."""
    return math.floor(value + 0.5)


def _is_number(value: object) -> bool:
    """Whether a value is a number of a file: true and false are none."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _ranged(low: float, high: float) -> None:
    """Refuses a range of values that neither starts at 0 nor goes both ways round it."""
    if not low <= 0 < high:
        raise RowSynthError(f"the values of a step table go from 0 or less up to more than 0, "
                            f"not from {low:g} to {high:g}")


def read(table: object) -> Parts:
    """A table as the file has it, as its steps and the index it goes round from.

    The file has a number, which is one step, a list of numbers, or {"v": list, "loop": index}. A loop that is no
    index of the list is none, as the engine has it: the table holds its last step, and the index is None.
    """
    values, loop = table, None
    if isinstance(table, dict):
        values, loop = table.get("v"), table.get("loop")
        whole = isinstance(loop, int) and not isinstance(loop, bool)
        if set(table) != {"v", "loop"} or not isinstance(values, (list, tuple)) or not whole:
            raise RowSynthError(f"a table with a loop is {{'v': list, 'loop': index}}, not {table!r}")
    elif not isinstance(table, (list, tuple)):
        values = (table,)
    for value in values:
        if not _is_number(value):
            raise RowSynthError(f"a step table is a number, a list of numbers or {{'v': list, 'loop': index}}, "
                                f"not {table!r}")
    return tuple(values), loop if loop is not None and 0 <= loop < len(values) else None


def write(steps: Sequence[float], loop: int | None = None) -> object:
    """Steps and the index they go round from as the file has a table, in the shortest shape that says it.

    A table with a loop is {"v": list, "loop": index}. One without is the number alone when it has one step, and
    a list when it has any other count of them.
    """
    steps = list(steps)
    if loop is not None:
        if not 0 <= loop < len(steps):
            raise RowSynthError(f"a table of {len(steps)} steps cannot go round from step {loop + 1}")
        return {"v": steps, "loop": loop}
    return steps[0] if len(steps) == 1 else steps


def at(steps: Sequence[float], loop: int | None, tick: int) -> float:
    """What a table is at a tick, as the engine reads it: past its end it goes round, or holds its last step.

    A table of no steps is 0 at every tick.
    """
    return note.step(write(steps, loop), tick, 0.0)


def sized(steps: Sequence[float], loop: int | None, count: int, most: int = MOST) -> Parts:
    """A table made `count` steps long: no fewer than one, and no more than `most` unless it was longer already.

    A step that is added takes the value of the last one, and steps are taken away from the end. A loop that
    began in a step that is gone is gone with it.
    """
    count = min(max(count, 1), max(most, len(steps)))
    made = (tuple(steps) + (steps[-1] if steps else 0,) * count)[:count]
    return made, loop if loop is not None and loop < count else None


def fit(width: int, count: int) -> int:
    """How wide a bar is when `count` steps share `width` pixels: the widest at which they and the handle fit.

    It is never wider than `BAR_MAX`. Nor is it narrower than `BAR_MIN`: steps that do not fit then are scrolled.
    """
    return min(BAR_MAX, max(BAR_MIN, (width - HANDLE_W) // max(count, 1) - GAP))


def counted(count: int) -> str:
    """How many steps a table has and how long they last, as it is shown: `9 steps, 0.15 s`."""
    return f"{count} step{'' if count == 1 else 's'}, {count / note.TICK_HZ:.2f} s"


def told(step: int, steps: Sequence[float], word: str = "value") -> str:
    """One step of a table in plain words: `step 3 of 9: level 12, at 0.03 s`. `word` is what its value is called."""
    return f"step {step + 1} of {len(steps)}: {word} {plain(steps[step])}, at {step / note.TICK_HZ:.2f} s"


@dataclass(frozen=True)
class Map:
    """Where the steps and the values of a table fall in the rectangle of its widget, and back.

    low and high are the range of a value: it starts at 0, or goes both ways round a zero line. bar is how wide a
    bar is, and first how many pixels of the bars are scrolled away to the left. With whole, the mouse sets whole
    numbers.
    """

    rect: Rect
    low: float
    high: float
    bar: int = BAR_MAX
    first: int = 0
    whole: bool = True

    def __post_init__(self) -> None:
        _ranged(self.low, self.high)

    @property
    def field(self) -> Rect:
        """The field with its outline: what the strip under it leaves of the rectangle."""
        x, y, w, h = self.rect
        return x, y, w, max(0, h - LOOP_H)

    @property
    def bars(self) -> Rect:
        """Where the bars are: the field inside its outline and a pixel of air."""
        x, y, w, h = self.field
        return x + INSET, y + INSET, max(0, w - 2 * INSET), max(0, h - 2 * INSET)

    @property
    def zone(self) -> Rect:
        """Where a press draws: the columns of the bars, over the whole height of the field."""
        return self.bars[0], self.field[1], self.bars[2], self.field[3]

    @property
    def strip(self) -> Rect:
        """The strip under the field in which the bracket of the loop lies, as wide as the bars."""
        _, y, _, h = self.rect
        return self.bars[0], y + h - LOOP_H, self.bars[2], LOOP_H

    @property
    def pitch(self) -> int:
        """From the left edge of one bar to the left edge of the next."""
        return self.bar + GAP

    def x(self, step: int) -> int:
        """The x of the left edge of the bar of a step."""
        return self.bars[0] + step * self.pitch - self.first

    def step_at(self, x: int) -> int:
        """The step whose bar, or the gap after it, an x is on. It is under 0 left of the first bar."""
        return (x - self.bars[0] + self.first) // self.pitch

    @property
    def up(self) -> int:
        """How many rows of pixels the bars have above the zero line: all there are when the range starts at 0."""
        rows = self.bars[3]
        return rows if self.low == 0 else max(0, _px((rows - 1) * self.high / (self.high - self.low)))

    @property
    def down(self) -> int:
        """How many rows of pixels the bars have under the zero line: none when the range starts at 0."""
        return 0 if self.low == 0 else max(0, self.bars[3] - 1 - self.up)

    @property
    def line(self) -> int:
        """The y of the zero line. When the range starts at 0 that is the row just under the bars."""
        return self.bars[1] + self.up

    def kept(self, value: float) -> float:
        """A value kept in the range, and as the mouse sets it: whole, or with no more than `PLACES` places."""
        value = min(self.high, max(self.low, value))
        if self.whole:
            return _px(value)
        value = round(value, PLACES)
        return int(value) if value == int(value) else value

    def y(self, value: float) -> int:
        """The row the bar of a value reaches to: up from the zero line, down from it, or the line itself for 0.

        A value out of the range counts as the end of it that is nearer.
        """
        value = min(self.high, max(self.low, value))
        if value > 0:
            return self.line - _px(value * self.up / self.high)
        if value < 0:
            return self.line + _px(value * self.down / self.low)
        return self.line

    def value_at(self, y: int) -> float:
        """The value whose bar reaches to a y: the opposite of `y`, kept in the range and as the mouse sets it."""
        rows = self.line - y
        if rows > 0 and self.up:
            return self.kept(rows * self.high / self.up)
        if rows < 0 and self.down:
            return self.kept(-rows * self.low / self.down)
        return self.kept(0)

    def block(self, step: int, value: float) -> Rect:
        """The rectangle of the bar of a step: from the zero line to the y of its value.

        A bar of no height is drawn one pixel high, so that the step is seen: on the zero line, or in the lowest
        row when the range starts at 0.
        """
        reach, line, left = self.y(value), self.line, self.x(step)
        if reach < line:
            return left, reach, self.bar, line - reach
        if reach > line:
            return left, line + 1, self.bar, reach - line
        return left, line if self.down else line - 1, self.bar, 1

    def handle(self, count: int) -> Rect:
        """Where the handle after the last of `count` bars is taken: over the whole height of the field."""
        _, y, _, h = self.zone
        return self.x(count), y, HANDLE_W, h

    def after(self, count: int, turn: int) -> int:
        """The x of the bar that is drawn for the tick `turn` ticks past the end of `count` steps, after the handle."""
        return self.x(count + turn) + HANDLE_W + GAP

    def bracket(self, loop: int, count: int) -> Rect:
        """The rectangle of the bracket of a loop: under the bars from the step it goes back to up to the last."""
        _, y, _, h = self.strip
        left = self.x(loop)
        return left, y + 1, self.x(count) - GAP - left, h - 1

    def reach(self, count: int) -> int:
        """How many pixels `count` bars and the handle after them take, side by side."""
        return count * self.pitch + HANDLE_W

    def within(self, count: int) -> Map:
        """This map, scrolled no further than `count` steps let it: the handle goes no further left than the edge."""
        return replace(self, first=min(max(self.first, 0), max(0, self.reach(count) - self.bars[2])))


def passed(seen: Map, count: int, start: Pos, end: Pos) -> list[tuple[int, float]]:
    """The steps the mouse passed on its way from a position to another, each with the value of its height there.

    The step it is on now has the height it is at now. One it went over has the height it had over the middle of
    that step, so that a fast drag leaves none behind. The step it set out from is left out, unless it is still
    on it. Only steps of a table of `count` steps are given.
    """
    (x0, y0), (x1, y1) = start, end
    first, last = seen.step_at(x0), seen.step_at(x1)
    way = 1 if last > first else -1
    found = []
    for step in range(first + way, last, way) if first != last else ():
        middle = seen.x(step) + seen.pitch // 2
        found.append((step, y0 + _px((middle - x0) * (y1 - y0) / (x1 - x0))))
    found.append((last, y1))
    return [(step, seen.value_at(y)) for step, y in found if 0 <= step < count]


def begins(seen: Map, count: int, loop: int | None, pos: Pos, button: int = 1) -> str:
    """Which gesture a press at a position begins on a table of `count` steps that goes round from `loop`.

    On a step in the field the left button draws and the right one sets to zero. The left one sizes the table on
    the handle after the last bar. Under the bars it moves the loop when it is on the step the bracket begins at,
    and sets one where no bracket is: on the bracket further along it begins nothing. Any other press begins
    nothing either: "".
    """
    step = seen.step_at(pos[0])
    if inside(seen.zone, pos):
        if 0 <= step < count:
            return {1: DRAW, 3: ZERO}.get(button, "")
        return SIZE if button == 1 and inside(seen.handle(count), pos) else ""
    if button == 1 and inside(seen.strip, pos) and 0 <= step < count and (loop is None or step <= loop):
        return LOOP
    return ""


@dataclass
class Gesture:
    """A drag that is under way: what its press began, on which table, and where the mouse has been since.

    kind is what `begins` called it and button the button that was pressed. seen is the sums of the widget and
    steps and loop its table, all as they were at the press: the bars stay where they are until the release.
    start is where the press was, pos where the mouse is now, and drawn the steps it passed, each with the value
    it was given last.
    """

    kind: str
    button: int
    seen: Map
    steps: Steps
    loop: int | None
    start: Pos
    pos: Pos = field(init=False)
    drawn: dict[int, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.pos = self.start
        self.to(self.start)

    def to(self, pos: Pos) -> None:
        """Take the mouse to another position, by way of every step between this one and that."""
        if self.kind in (DRAW, ZERO):
            for step, value in passed(self.seen, len(self.steps), self.pos, pos):
                self.drawn[step] = value if self.kind == DRAW else self.seen.kept(0)
        self.pos = pos

    def gives(self) -> Parts:
        """What a release where the mouse is now gives: the steps, and the index they go round from.

        The handle makes the table as many steps longer or shorter as the mouse is from where it was pressed. The
        loop begins at the step the mouse is over, at the last step when the mouse is further right, and is gone
        when the mouse is left of the first.
        """
        seen, steps = self.seen, self.steps
        if self.kind == SIZE:
            return sized(steps, self.loop, len(steps) + seen.step_at(self.pos[0]) - seen.step_at(self.start[0]))
        if self.kind == LOOP:
            step = seen.step_at(self.pos[0])
            return steps, None if step < 0 else min(step, len(steps) - 1)
        return tuple(self.drawn.get(step, value) for step, value in enumerate(steps)), self.loop


class StepTable(Widget):
    """One step table as a row of bars: a bar for each step, as high as its value, drawn with the mouse.

    `table` is the table as the file has it: a number, a list, or a list with its loop. `low` and `high` are the
    range of a value: from 0 up, or both ways round a zero line that the bars then stand on and hang from. The
    widget saves nothing. When a drag ends and the table is another than it was, `changed` is called once with
    the new one in the shape the file takes, and the widget shows that one from then on: its owner saves, and
    sets `table` when the file has something else.

    A press or a drag with the left button sets each step it passes to the height of the mouse, in whole numbers
    when `whole`. The right button sets them to zero. The handle after the last bar is dragged right for more
    steps and left for fewer. Under the bars is the bracket of the loop: the step it begins at is dragged, off
    the left end to take the loop away, and a click where no bracket is sets one. After the handle the steps are
    drawn again, fainter, as the table goes on past its end.

    The widget has no name on it: its owner writes its name and the count of its steps beside it. An owner with
    several tables under one ruler gives each the same `span`, the count of the steps the bars make room for,
    so that the bars of all are equally wide: with 0 a table makes room for its own. `first` is how many pixels
    of the bars are scrolled away to the left. The wheel is read by `kit.wheel`: sideways, which is with shift,
    it scrolls a table whose steps do not fit, and `moved` is told the new `first`, for an owner that has its
    other tables go along. Any other step of the wheel is left to what lies under the table.

    The help line says what the presses do where the mouse is: on a step, which it names with its value as
    `word` calls it, on the handle, and under the bars. After that it says `wheel_help`, which is what the wheel
    does there. Anywhere else, and on a table that is not enabled, the line is `help`.
    """

    def __init__(self, rect: Rect, table: object, low: float, high: float,
                 changed: Callable[[object], object] | None = None, colour: Colour = theme.ACTIVE,
                 whole: bool = True, help: str = "", word: str = "value", wheel_help: str = "") -> None:
        super().__init__(rect, help)
        _ranged(low, high)
        self.low = low
        self.high = high
        self.changed = changed
        self.colour = colour
        self.whole = whole
        self.word = word
        self.wheel_help = wheel_help
        self.span = 0
        self.first = 0
        self.moved: Callable[[int], object] | None = None
        self._mouse: Pos = NOWHERE
        self._drag: Gesture | None = None
        self.table = table

    @property
    def table(self) -> object:
        """The table in the shape the file takes. While a drag is under way, it is what a release now would give.

        A table that says what the given one said is the given one, in the shape it came in.
        """
        now = self._now()
        return copy.deepcopy(self._given) if now == (self._steps, self._loop) else write(*now)

    @table.setter
    def table(self, table: object) -> None:
        self._steps, self._loop = read(table)
        self._given = copy.deepcopy(table)
        self._drag = None

    @property
    def steps(self) -> Steps:
        """The steps as they are drawn: while a drag is under way, as its release would leave them."""
        return self._now()[0]

    @property
    def loop(self) -> int | None:
        """The index the table goes round from as it is drawn, or None when it holds its last step."""
        return self._now()[1]

    @property
    def map(self) -> Map:
        """The sums of the widget as it is now: while a drag is under way, as they were at its press."""
        if self._drag is not None:
            return self._drag.seen
        count = max(len(self._steps), self.span)
        width = max(0, self.rect[2] - 2 * INSET)
        return Map(self.rect, self.low, self.high, fit(width, count), self.first, self.whole).within(count)

    @property
    def help(self) -> str:
        """What the presses and the wheel do where the mouse is, or the line the table was made with.

        It goes by what a press of the left button would begin there, and while a drag is under way by what
        that drag is. A step is named with the value it has, or would have on a release now.
        """
        if not self.enabled:
            return self._help
        seen, steps = self.map, self.steps
        kind = self._drag.kind if self._drag is not None else begins(seen, len(steps), self.loop, self._mouse)
        step = min(max(seen.step_at(self._mouse[0]), 0), len(steps) - 1)
        if kind in (DRAW, ZERO):
            return kit.told(f"set {told(step, steps, self.word)}", ZERO_HELP, self.wheel_help)
        if kind == SIZE:
            return kit.told(SIZE_HELP, wheel=self.wheel_help)
        if kind == LOOP:
            return kit.told(LOOP_HELP if self.loop is not None else f"{LOOP_FROM} step {step + 1}",
                            wheel=self.wheel_help)
        return self._help

    @help.setter
    def help(self, line: str) -> None:
        self._help = line

    def _now(self) -> Parts:
        """The steps and the loop to draw: what the drag under way gives, or what there is."""
        return self._drag.gives() if self._drag is not None else (self._steps, self._loop)

    def on(self, event: Event, ui: Ui) -> bool:
        """Note where the mouse is, begin a drag at a press and carry it on to its release, and scroll.

        A press is used when it begins a drag. The wheel is used when it goes sideways on a table whose steps
        do not fit.
        """
        if event.kind == "key":
            return False
        self._mouse = event.pos
        if event.kind == "wheel":
            return self._drag is None and self._scroll(event)
        if event.kind == "down":
            # A press while another button is held begins nothing.
            return ui.held is None and self._press(event)
        drag = self._drag
        if drag is not None:
            drag.to(event.pos)
            if event.kind == "up" and event.button == drag.button:
                self._release()
        return False

    def _scroll(self, event: Event) -> bool:
        """Scroll the bars by a step of the wheel that goes sideways, and say whether there was anything to scroll."""
        way, steps = kit.wheel(event)
        seen, count = self.map, max(len(self._steps), self.span)
        if way != kit.ACROSS or seen.reach(count) <= seen.bars[2]:
            return False
        self.first = replace(seen, first=seen.first - steps * kit.WHEEL_PX).within(count).first
        if self.moved is not None:
            self.moved(self.first)
        return True

    def _press(self, event: Event) -> bool:
        """Begin the drag that a press begins, and say whether it began one."""
        seen = self.map
        kind = begins(seen, len(self._steps), self._loop, event.pos, event.button)
        if kind:
            self._drag = Gesture(kind, event.button, seen, self._steps, self._loop, event.pos)
        return bool(kind)

    def _release(self) -> None:
        """End the drag: a table that is another than it was is taken, and the owner is told of it."""
        drag, self._drag = self._drag, None
        now = drag.gives()
        if now != (self._steps, self._loop):
            self.table = write(*now)
            if self.changed is not None:
                self.changed(self.table)

    def draw(self, canvas: Canvas) -> None:
        """Draw the bars in the field, the repeat after them and the bracket of the loop.

        All of it is in the rectangle. While a drag is under way the bars are as its release would leave them.
        It is dim when not enabled.
        """
        seen = self.map
        steps, loop = self._now()
        count = len(steps)
        ink, mark = (self.colour, theme.TEXT) if self.enabled else (theme.DIM, theme.LINE)
        left, top, wide, high = seen.bars
        with canvas.clip(self.rect):
            canvas.box(seen.field, theme.BACKGROUND)
            canvas.frame(seen.field, theme.LINE)
            with canvas.clip(seen.bars):
                if seen.down:
                    canvas.hline(left, seen.line, wide, theme.LINE)
                for step, value in enumerate(steps):
                    canvas.box(seen.block(step, value), ink)
                turn = 0
                while count and seen.after(count, turn) < left + wide:
                    _, y, w, h = seen.block(0, at(steps, loop, count + turn))
                    canvas.box((seen.after(count, turn), y, w, h), theme.shade(ink, FAINT))
                    turn += 1
                grip = min(HANDLE_H, high)
                canvas.box((seen.x(count), top + (high - grip) // 2, HANDLE_W, grip), mark)
            if loop is not None:
                with canvas.clip(seen.strip):
                    x, y, w, h = seen.bracket(loop, count)
                    canvas.box((x, y, seen.bar, h), mark)
                    canvas.hline(x, y + h - 1, w, mark)
                    canvas.vline(x + w - 1, y, h, mark)
