"""An FM voice in pictures: its eight wirings to pick from, and an operator's envelope as a line with points to drag."""
from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from .. import fm
from ..errors import RowSynthError
from . import theme
from .dials import plain
from .layout import Rect
from .theme import Colour
from .ui import NOWHERE, Event, Pos, Ui, Widget, inside, share

if TYPE_CHECKING:
    from .canvas import Canvas

#: The number the first operator goes by, in a picture and in the help line: the file counts them from 0.
FIRST = 0

#: The height a row of wirings is drawn for, and how far two pictures of it are apart.
WIRING_H = 62
BETWEEN = 6
#: The side of an operator's box. From the left of one column of boxes to the left of the next, and from the top
#: of one row to the top of the next.
BOX = 11
STEP_X = 16
STEP_Y = 16
#: From the foot of a heard box down to the base line, how far the base line runs on past the outermost of them,
#: and from the base line to the foot of the picture.
STEM = 6
BASE_OVER = 3
FOOT = 8
#: A line that goes over boxes runs this far above them for each column it goes over. The head of a line is a
#: pixel on either side of it, this far before the box it leads to.
OVER = 4
HEAD = 2
#: How far the loop of the feedback stands out from the corner of its box.
LOOP = 3
#: From the corner of a picture to the number of its algorithm.
LABEL = 4

#: From the outside of an envelope's field to where its line runs: the outline, and room for half a point.
INSET = 4
#: The side of a point of the line, and how far from its middle a press still takes it.
POINT = 5
GRAB = 3
#: How long the note of the picture is held on after its decay is over, in seconds, before it is let go.
HOLD = 0.3
#: How many samples of the engine's envelope a pixel of the picture has from side to side: the rate the envelope
#: is asked for at follows from it, so a long note takes no more sums than a short one.
SAMPLES = 32
#: The most places after the point of a value that the mouse sets.
PLACES = 3
#: The range the mouse takes each setting of an envelope through: seconds, a level, and decibels a second. The
#: file has no tops for the times and the fall. Those here are far enough for any note.
RANGES = {"a": (0, 4), "d": (0, 4), "s": (0, 1), "sd": (0, 96), "r": (0, 4)}


def _px(value: float) -> int:
    """The nearest whole number, a half going up."""
    return math.floor(value + 0.5)


@dataclass(frozen=True)
class Wires:
    """One algorithm as the engine's tables have it, the operators counted from 0.

    heard are the operators that go to the output. changes holds a pair for each operator that changes another:
    the one that changes, and the one it changes. feedback is the operator that the feedback is on.
    """

    heard: tuple[int, ...]
    changes: tuple[tuple[int, int], ...]
    feedback: int


def _whole(algorithm: object) -> int:
    """An algorithm, refused when it is none of the engine's."""
    count = len(fm.CARRIER)
    if isinstance(algorithm, bool) or not isinstance(algorithm, int) or not 0 <= algorithm < count:
        raise RowSynthError(f"an algorithm is a whole number from 0 to {count - 1}, not {algorithm!r}")
    return algorithm


def wires(algorithm: int) -> Wires:
    """Which operators of an algorithm are heard, which changes which, and where the feedback is.

    It is read from the tables the engine plays from, so it is what is heard.
    """
    return _wires(_whole(algorithm))


@lru_cache(maxsize=None)
def _wires(algorithm: int) -> Wires:
    """As `wires`, for what is known to be an algorithm: read once and kept."""
    heard, into = fm.CARRIER[algorithm], fm.INTO[algorithm]
    ops = range(len(heard))
    return Wires(tuple(op for op in ops if heard[op]),
                 tuple((source, op) for op in ops for source in ops if into[op, source]), fm.FEEDBACK_OPERATOR)


def _listed(ops: Sequence[int]) -> str:
    """Some operators by their numbers: `1`, `1 and 3`, `1, 2 and 3`."""
    names = [str(op + FIRST) for op in ops]
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def told(algorithm: int) -> str:
    """A wiring in plain words: `0 and 1 change 2, 2 changes 3; 3 is heard`."""
    wired = wires(algorithm)
    changed: dict[tuple[int, ...], list[int]] = {}
    for op in sorted({op for _, op in wired.changes}):
        changed.setdefault(tuple(source for source, to in wired.changes if to == op), []).append(op)
    parts = [f"{_listed(sources)} {'changes' if len(sources) == 1 else 'change'} {_listed(ops)}"
             for sources, ops in changed.items()]
    return (f"{', '.join(parts) or 'none changes another'}; "
            f"{_listed(wired.heard)} {'is' if len(wired.heard) == 1 else 'are'} heard")


def places(algorithm: int) -> tuple[tuple[int, int], ...]:
    """Where each operator stands in the picture of an algorithm, as (column, row).

    Columns count from the left, and rows up from the base row, in which the heard ones stand. Each heard
    operator has a column of its own, in the order of their numbers, and what changes it stands to its left: as
    many columns from it as it is steps from the output. An operator stands in the row of what it changes, and
    of two that would stand in one place the later goes a row up.
    """
    return _places(_whole(algorithm))


@lru_cache(maxsize=None)
def _places(algorithm: int) -> tuple[tuple[int, int], ...]:
    """As `places`, for what is known to be an algorithm: worked out once and kept."""
    wired = _wires(algorithm)
    count = fm.CARRIER.shape[1]
    to = [[changed for source, changed in wired.changes if source == op] for op in range(count)]
    # An operator is changed only by ones before it, so from the last one down each finds what it changes done.
    far, lead = [0] * count, list(range(count))
    for op in reversed(range(count)):
        if to[op]:
            far[op], lead[op] = 1 + max(far[changed] for changed in to[op]), lead[to[op][0]]
    right, column = {}, 0
    for op in sorted(set(lead)):
        column += 1 + max(far[other] for other in range(count) if lead[other] == op)
        right[op] = column - 1
    found: dict[int, tuple[int, int]] = {}
    for op in sorted(range(count), key=lambda op: (far[op], op)):
        spot = (right[lead[op]] - far[op], found[to[op][0]][1] if to[op] else 0)
        while spot in found.values():
            spot = (spot[0], spot[1] + 1)
        found[op] = spot
    return tuple(found[op] for op in range(count))


@dataclass(frozen=True)
class Picture:
    """The picture of one algorithm in a rectangle, as the rectangles it is drawn with.

    boxes has the box of each operator. stems has a line for each heard one, from its box down to base, the line
    they stand on. lines has, for each pair of `Wires.changes`, the strips of the line from the one to the
    other, its head at the changed one. loop is the outline of the loop of the feedback, which its box covers
    a corner of.
    """

    boxes: tuple[Rect, ...]
    stems: tuple[Rect, ...]
    base: Rect
    lines: tuple[tuple[Rect, ...], ...]
    loop: Rect


def _line(source: Rect, to: Rect, columns: int) -> tuple[Rect, ...]:
    """The strips of the line from one box to another that is `columns` columns to its right, and its head.

    To the next box of its row it runs straight. From a row above it runs out to the right and down into the
    top of the other. Any other runs over the boxes between: up, along, and down into the top.
    """
    (sx, sy, _, _), (tx, ty, _, _), half = source, to, BOX // 2
    if columns == 1 and sy == ty:
        back = tx - HEAD
        return (sx + BOX, sy + half, tx - sx - BOX, 1), (back, sy + half - 1, 1, 1), (back, sy + half + 1, 1, 1)
    into = tx + half
    head = ((into - 1, ty - HEAD, 1, 1), (into + 1, ty - HEAD, 1, 1))
    if columns == 1 and sy < ty:
        return (sx + BOX, sy + half, into - sx - BOX + 1, 1), (into, sy + half + 1, 1, ty - sy - half - 1), *head
    rail, out = min(sy, ty) - OVER * max(1, columns - 1), sx + half
    return (out, rail, 1, sy - rail), (out, rail, into - out + 1, 1), (into, rail, 1, ty - rail), *head


def picture(cell: Rect, algorithm: int) -> Picture:
    """The picture of an algorithm in a rectangle: its boxes in the middle from side to side, near the foot."""
    wired, spots = wires(algorithm), places(algorithm)
    x, y, w, h = cell
    left = x + (w - max(column for column, _ in spots) * STEP_X - BOX) // 2
    base = y + h - 1 - FOOT
    top = base - STEM - BOX
    boxes = tuple((left + column * STEP_X, top - row * STEP_Y, BOX, BOX) for column, row in spots)
    middles = [boxes[op][0] + BOX // 2 for op in wired.heard]
    return Picture(
        boxes,
        tuple((middle, top + BOX, 1, STEM) for middle in middles),
        (min(middles) - BASE_OVER, base, max(middles) - min(middles) + 1 + 2 * BASE_OVER, 1),
        tuple(_line(boxes[source], boxes[to], spots[to][0] - spots[source][0]) for source, to in wired.changes),
        (boxes[wired.feedback][0] - LOOP, boxes[wired.feedback][1] - LOOP, 2 * LOOP, 2 * LOOP))


def cells(rect: Rect) -> list[Rect]:
    """The rectangle of each algorithm's picture in a row of them: side by side from the left, a gap between two."""
    x, y, w, h = rect
    count = len(fm.CARRIER)
    found = []
    for width in share(max(0, w - (count - 1) * BETWEEN), [1] * count):
        found.append((x, y, width, h))
        x += width + BETWEEN
    return found


class Wiring(Widget):
    """The algorithms of the FM voice as a row of pictures, of which one is picked: `algorithm`.

    A picture shows the operators as numbered boxes. The heard ones are in the colour of what is on and stand on
    a base line. The others are in the colour of a song, each with a line to the one it changes. The feedback is
    a small loop on the corner of its operator. A click on a picture picks it, and `changed` is called with the
    algorithm when that is another one. Over a picture the help line says its wiring in plain words, and
    anywhere else it is `help`.
    """

    def __init__(self, rect: Rect, algorithm: int = 0, changed: Callable[[int], object] | None = None,
                 help: str = "") -> None:
        super().__init__(rect, help)
        self.algorithm = _whole(algorithm)
        self.changed = changed
        self._mouse: Pos = NOWHERE

    @property
    def help(self) -> str:
        """The wiring of the picture under the mouse in plain words, or the line the widget was made with."""
        over = self.at(self._mouse)
        return self._help if over is None else told(over)

    @help.setter
    def help(self, line: str) -> None:
        self._help = line

    def at(self, pos: Pos) -> int | None:
        """The algorithm whose picture a position is on, or None when it is on none."""
        return next((index for index, cell in enumerate(cells(self.rect)) if inside(cell, pos)), None)

    def pick(self, algorithm: int) -> None:
        """Pick an algorithm, and tell. Picking the one that is picked changes nothing."""
        if _whole(algorithm) != self.algorithm:
            self.algorithm = algorithm
            if self.changed is not None:
                self.changed(algorithm)

    def on(self, event: Event, ui: Ui) -> bool:
        """Use a press of the left button on a picture, and pick the picture its release is on."""
        if event.kind == "key":
            return False
        self._mouse = event.pos
        over = self.at(event.pos)
        if event.kind == "down":
            return event.button == 1 and over is not None
        if event.kind == "up" and event.button == 1 and ui.held is self and over is not None:
            self.pick(over)
        return False

    def draw(self, canvas: Canvas) -> None:
        """Draw the eight pictures. The picked one is lit and outlined in the colour of what is on."""
        over = self.at(self._mouse) if self.hot else None
        for algorithm, cell in enumerate(cells(self.rect)):
            picked = algorithm == self.algorithm
            wired, seen = wires(algorithm), picture(cell, algorithm)
            inks = [theme.ACTIVE if op in wired.heard else theme.SONG for op in range(len(seen.boxes))]
            with canvas.clip(cell):
                canvas.box(cell, theme.LIT if picked or algorithm == over else theme.RAISED)
                canvas.frame(cell, theme.ACTIVE if picked else theme.LINE)
                canvas.text(str(algorithm), cell[0] + LABEL, cell[1] + LABEL, theme.ACTIVE if picked else theme.DIM)
                for strip in (seen.base, *seen.stems):
                    canvas.box(strip, theme.ACTIVE)
                canvas.frame(seen.loop, inks[wired.feedback])
                for (source, _), strips in zip(wired.changes, seen.lines):
                    for strip in strips:
                        canvas.box(strip, inks[source])
                for op, box in enumerate(seen.boxes):
                    heard = op in wired.heard
                    canvas.box(box, inks[op] if heard else theme.BACKGROUND)
                    canvas.frame(box, inks[op])
                    canvas.text_in(str(op + FIRST), box, theme.BACKGROUND if heard else inks[op], "centre")


@dataclass(frozen=True)
class Point:
    """One point of an envelope's line. It stands where a stage of the envelope ends, and is named after it.

    time is the setting it sets when it is moved from side to side, and level the one it sets when it is moved
    up and down. One that it has none of is None: that way the point does not move.
    """

    name: str
    time: str | None
    level: str | None


#: The points, in the order of their stages: the top of the attack, the end of the decay, where the note is let
#: go after it was held, and the end of the release.
POINTS = (Point("attack", "a", None), Point("decay", "d", "s"), Point("sustain", None, "sd"),
          Point("release", "r", None))
ATTACK, DECAY, SUSTAIN, RELEASE = range(len(POINTS))


def settings(op: object) -> dict[str, float]:
    """The settings of an operator's envelope: its own, and the engine's where it names none."""
    if not isinstance(op, dict):
        raise RowSynthError(f"an operator is a table of its settings, not {op!r}")
    env = op.get("env", {})
    if not isinstance(env, dict):
        raise RowSynthError(f"the envelope of an operator is a table of its settings, not {env!r}")
    values = dict(fm.ENVELOPE)
    for key in values:
        value = env.get(key, values[key])
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RowSynthError(f"the {key!r} of an envelope is a number, not {value!r}")
        values[key] = value
    return values


def moments(values: dict[str, float]) -> tuple[float, float, float, float]:
    """When each point is, in seconds from the start of the note, for the settings of an envelope.

    The note is held on for `HOLD` seconds after its decay is over, so that every stage has its stretch.
    """
    top = max(values["a"], 0)
    fallen = top + max(values["d"], 0)
    return top, fallen, fallen + HOLD, fallen + HOLD + max(values["r"], 0)


def length(op: dict) -> float:
    """How long the note of an operator's picture is, from its start to the end of its release, in seconds."""
    return moments(settings(op))[RELEASE]


def falling(values: dict[str, float]) -> float:
    """For how many seconds of the picture's note the level falls as the sustain has it, before the note is let go.

    That is the time the note is held on for, `HOLD`. To the engine a decay to the full level is over at once,
    so with a sustain level of 1 or more the time of the decay is in it too.
    """
    return HOLD + (max(values["d"], 0) if values["s"] >= 1 else 0)


@dataclass(frozen=True)
class Curve:
    """The engine's envelope as a picture so many columns wide has it.

    low and high are the lowest and the highest level in each column. spots has, for each point, the column it
    is in and the level the envelope has there. A point the picture ends before has a column past the last one,
    and one that is too far past it to be taken has no level but 0.
    """

    low: tuple[float, ...]
    high: tuple[float, ...]
    spots: tuple[tuple[int, float], ...]


@lru_cache(maxsize=64)
def _curve(a: float, d: float, s: float, sd: float, r: float, seconds: float, columns: int) -> Curve:
    """The envelope of five settings over `seconds`, in `columns` columns: the first is its start, the last its end."""
    values = dict(zip(fm.ENVELOPE, (a, d, s, sd, r)))
    top, fallen, let_go, gone = moments(values)
    last = max(columns - 1, 1)
    rate = math.ceil(SAMPLES * last / seconds)
    # Past the picture the envelope is asked for only as far as a point can be taken from inside the field.
    out = fm.envelope(values, let_go, (last + INSET + GRAB + 1) * seconds / last, rate)
    # The level after each sample, and before the first one: `levels[k]` is at k samples into the note.
    levels = np.concatenate((np.zeros(1, dtype=out.dtype), out))
    per = last / (seconds * rate)
    column = np.floor(np.arange(len(levels)) * per + 0.5).astype(np.int64)
    low, high = [], []
    for start, end in zip(np.searchsorted(column, range(columns)), np.searchsorted(column, range(columns), "right")):
        # With the last sample before the column, so that the line goes on from where it was.
        part = levels[max(start - 1, 0):max(end, 1)]
        low.append(float(part.min()))
        high.append(float(part.max()))
    # The attack is over with the first sample that is full, and the decay no sooner than the sample after it.
    full = max(1, math.ceil(top * rate - 1e-9))
    samples = (full, max(full + 1, _px(fallen * rate)), _px(let_go * rate), _px(gone * rate))
    return Curve(tuple(low), tuple(high),
                 tuple((_px(k * per), float(levels[k]) if k < len(levels) else 0.0) for k in samples))


@dataclass(frozen=True)
class Map:
    """Where the times and the levels of an envelope fall in the rectangle of its widget.

    seconds is the time the picture shows from its left to its right. A level of 0 is at its foot and one of 1
    at its top.
    """

    rect: Rect
    seconds: float

    def __post_init__(self) -> None:
        if not self.seconds > 0:
            raise RowSynthError(f"the picture of an envelope shows more than 0 seconds, not {self.seconds!r}")

    @property
    def plot(self) -> Rect:
        """Where the line runs: the field inside its outline and the room for half a point."""
        x, y, w, h = self.rect
        return x + INSET, y + INSET, max(1, w - 2 * INSET), max(1, h - 2 * INSET)

    @property
    def second(self) -> float:
        """How many seconds a pixel is, from side to side."""
        return self.seconds / max(self.plot[2] - 1, 1)

    @property
    def rows(self) -> int:
        """How many rows of pixels a level of 1 stands above the foot."""
        return max(self.plot[3] - 1, 1)

    def y(self, level: float) -> int:
        """The row of a level. One out of the picture counts as the foot or the top."""
        return self.plot[1] + self.plot[3] - 1 - _px(min(1.0, max(0.0, level)) * self.rows)

    def curve(self, values: dict[str, float]) -> Curve:
        """The engine's envelope for some settings, column by column of the picture."""
        return _curve(*(values[key] for key in fm.ENVELOPE), self.seconds, self.plot[2])

    def spots(self, values: dict[str, float]) -> list[Pos]:
        """The middle of each point for some settings: on the line, at the time of the point."""
        return [(self.plot[0] + column, self.y(level)) for column, level in self.curve(values).spots]


def nearest(spots: Sequence[Pos], pos: Pos) -> int | None:
    """The point a press at a position takes: the nearest one that is no more than `GRAB` pixels off, or None.

    Of two as near the later is taken, so that two points on each other can be pulled apart.
    """
    best: tuple[int, int] | None = None
    for index, (x, y) in enumerate(spots):
        off = max(abs(pos[0] - x), abs(pos[1] - y))
        if off <= GRAB and (best is None or off <= best[0]):
            best = (off, index)
    return None if best is None else best[1]


def kept(value: float, key: str) -> float:
    """A value of a setting kept in its range, and as the mouse sets it: with no more than `PLACES` places."""
    low, high = RANGES[key]
    value = round(min(high, max(low, value)), PLACES)
    return int(value) if value == int(value) else value


def moved(op: dict, point: int, seen: Map, by: Pos) -> dict:
    """The operator as the file would hold it after a point was dragged so many pixels right and down.

    A point moves only the ways it has a setting for. To the side it sets a time. Up and down the decay sets
    the level it falls to, and the sustain how fast the level falls while the note is held: as fast as it takes
    to be at that height when the note is let go. A setting that comes out as it was is left as the operator
    had it, also when it had none.
    """
    values, which, (right, down) = settings(op), POINTS[point], by
    new: dict[str, float] = {}
    if which.time and right:
        new[which.time] = kept(values[which.time] + right * seen.second, which.time)
    if which.level == "s" and down:
        new["s"] = kept(values["s"] - down / seen.rows, "s")
    if which.level == "sd" and down and values["s"] > 0:
        level = values["s"] * 10 ** (-values["sd"] * falling(values) / 20) - down / seen.rows
        fall = RANGES["sd"][1] if level <= 0 else 20 * math.log10(values["s"] / level) / falling(values)
        new["sd"] = kept(fall, "sd")
    new = {key: value for key, value in new.items() if value != values[key]}
    made = copy.deepcopy(op)
    if new:
        made["env"] = {**made.get("env", {}), **new}
    return made


def said(point: int, values: dict[str, float]) -> str:
    """A point and what it is set to, in plain words: `decay: 0.25 s, level 0.6`."""
    which = POINTS[point]
    parts = [f"{plain(values[which.time])} s"] if which.time else []
    if which.level:
        parts.append(f"level {plain(values['s'])}" if which.level == "s"
                     else f"falls {plain(values['sd'])} dB a second")
    return f"{which.name}: {', '.join(parts)}"


@dataclass
class _Drag:
    """A drag that is under way: the point it has, where it was pressed, and the sums and the operator then."""

    point: int
    start: Pos
    seen: Map
    began: dict


class Envelope(Widget):
    """One operator's level over time, as a line with a point at the end of each stage to drag.

    `op` is the operator as the file has it. The line is the engine's own envelope for a note that is held for
    its attack, its decay and `HOLD` seconds more, and then let go: a faint upright line stands where it is let
    go. `seconds` is the time the picture shows from side to side. Without it the picture is as long as the
    note, and is fitted again when a drag has ended.

    The points are those of `POINTS`. The attack and the release move to the side and set a time, the decay
    moves both ways and sets a time and a level, and the sustain moves up and down and sets how fast the level
    falls while the note is held. The widget saves nothing. `changed` is called with the operator each time a
    move of a drag makes it another, and `settled` once when the drag ends and the operator is another than it
    was at the press: both get it as the file would hold it. Setting `op` drops a drag that is under way, so an
    owner that reads the file each frame leaves it alone while the widget is `busy`. Over a point the help line
    names it and what it is set to, and anywhere else it is `help`.
    """

    def __init__(self, rect: Rect, op: dict, changed: Callable[[dict], object] | None = None,
                 settled: Callable[[dict], object] | None = None, colour: Colour = theme.ACTIVE,
                 seconds: float | None = None, help: str = "") -> None:
        super().__init__(rect, help)
        self.changed = changed
        self.settled = settled
        self.colour = colour
        self.seconds = seconds
        self._mouse: Pos = NOWHERE
        self._drag: _Drag | None = None
        self.op = op

    @property
    def seconds(self) -> float | None:
        """The time the picture shows from side to side, or None for one that is as long as its note."""
        return self._seconds

    @seconds.setter
    def seconds(self, seconds: float | None) -> None:
        self._seconds = None if seconds is None else Map(self.rect, seconds).seconds

    @property
    def op(self) -> dict:
        """The operator as the file would hold it. While a drag is under way, it is what a release now would give."""
        return copy.deepcopy(self._op)

    @op.setter
    def op(self, op: dict) -> None:
        settings(op)
        self._op = copy.deepcopy(op)
        self._drag = None

    @property
    def busy(self) -> bool:
        """Whether a drag is under way: from the press that began it to its release, or until it is dropped."""
        return self._drag is not None

    @property
    def map(self) -> Map:
        """The sums of the widget as it is now: while a drag is under way, as they were at its press."""
        if self._drag is not None:
            return self._drag.seen
        return Map(self.rect, length(self._op) if self.seconds is None else self.seconds)

    @property
    def help(self) -> str:
        """The point under the mouse, or the one that is dragged, with what it is set to. Else the widget's line."""
        point = self._live()
        return self._help if point is None else said(point, settings(self._op))

    @help.setter
    def help(self, line: str) -> None:
        self._help = line

    def at(self, pos: Pos) -> int | None:
        """The point a press at a position takes, as its index in `POINTS`, or None when it takes none."""
        return nearest(self.map.spots(settings(self._op)), pos)

    def _live(self) -> int | None:
        """The point that is dragged or, on a widget that is enabled, the one under the mouse."""
        if self._drag is not None:
            return self._drag.point
        return self.at(self._mouse) if self.enabled else None

    def on(self, event: Event, ui: Ui) -> bool:
        """Note where the mouse is, begin a drag at a press of the left button on a point, and carry it on."""
        if event.kind in ("key", "wheel"):
            return False
        self._mouse = event.pos
        if event.kind == "down":
            point = self.at(event.pos) if event.button == 1 and ui.held is None else None
            if point is not None:
                self._drag = _Drag(point, event.pos, self.map, self._op)
            return point is not None
        drag = self._drag
        if drag is not None:
            op = moved(drag.began, drag.point, drag.seen, (event.pos[0] - drag.start[0], event.pos[1] - drag.start[1]))
            if op != self._op:
                self._op = op
                if self.changed is not None:
                    self.changed(self.op)
            if event.kind == "up" and event.button == 1:
                self._drag = None
                if op != drag.began and self.settled is not None:
                    self.settled(self.op)
        return False

    def draw(self, canvas: Canvas) -> None:
        """Draw the field, the line where the note is let go, the envelope and its points. Dim when not enabled."""
        seen, values = self.map, settings(self._op)
        curve, live = seen.curve(values), self._live() if self.hot or self._drag is not None else None
        ink, knob = (self.colour, theme.TEXT) if self.enabled else (theme.DIM, theme.LINE)
        x, y, w, h = self.rect
        left, half = seen.plot[0], POINT // 2
        canvas.box(self.rect, theme.BACKGROUND)
        canvas.frame(self.rect, theme.LINE)
        with canvas.clip((x + 1, y + 1, max(0, w - 2), max(0, h - 2))):
            canvas.vline(left + curve.spots[SUSTAIN][0], y + 1, max(0, h - 2), theme.RAISED)
            for column, (low, high) in enumerate(zip(curve.low, curve.high)):
                canvas.vline(left + column, seen.y(high), seen.y(low) - seen.y(high) + 1, ink)
            for point, (px, py) in enumerate(seen.spots(values)):
                canvas.box((px - half, py - half, POINT, POINT), knob if point == live else theme.BACKGROUND)
                canvas.box((px - half + 1, py - half + 1, POINT - 2, POINT - 2), knob)
