"""The view of a sound: its notes as bars on a clock of seconds, its wave on that clock, its sums, its gestures."""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from ..errors import RowSynthError
from ..measure import pitch_at
from ..note import DEFAULT_RATE, TICK_HZ
from ..notes import midi
from . import soundedit, soundview, theme
from .font import Font
from .kit import (ACROSS, WHEEL_KEYS, WHEEL_PX, ZOOM, Keys, Plus, ScrollBar, Velocity, black, key_rect, moment_at, name,
                  told, wheel, x_at, zoomed)
from .kit import pitch_at as key_at
from .layout import Rect
from .panels import within
from .roll import BOX, DARKER, END_W, LIGHTER, MOVE, PICK, PLACE, REMOVE, SLACK, STRETCH, counted, mix, path
from .soundedit import HIGHEST, LOWEST, Clip, Pick, Picked
from .soundview import Bar, Shape
from .theme import Colour
from .ui import NOWHERE, Event, Pos, Ui, Widget, inside

if TYPE_CHECKING:
    from .canvas import Canvas

#: How high a key is, and with it its row of the grid, and how wide the keyboard down the left is: as in every
#: view of notes.
KEY_H = theme.KEY_H
KEYS_W = theme.KEYS_W
#: The room a scroll bar takes beside the grid and under it: the bar, and the air between it and the grid.
BAR = theme.SCROLL_W + theme.SCROLL_GAP
#: The key round which a sound with no notes is shown.
MIDDLE_C = 60
#: How high the strip along the top is in which the seconds are numbered, and the strip under it that names the
#: shapes where there are several.
RULER_H = 10
LABELS_H = 10
#: The marks of the ruler are a power of ten of seconds apart: from thousandths at the finest to hundreds.
RULER_FINEST = -3
RULER_COARSEST = 2
#: What a number of the ruler has free round it: 2 pixels before it, and the rest after.
RULER_AIR = 7
#: What the box of the wave is called, and how far a wave at full scale keeps from the outline of its box.
WAVE = "WAVE"
WAVE_AIR = 2
#: The share of full scale from which a sample counts as clipped, as `measure` counts them.
FULL = 0.999
#: What one step of the zoom makes the clock wider by, and how many steps the zoom goes.
ZOOM_STEP = 1.25
ZOOM_MOST = 16
#: The room the clock of a sound that can be changed has after the end of the sound, for notes that make it
#: longer: as long again as the sound, and no less than this many seconds.
ROOM_LEAST = 1.0
#: How much of that room is in view as a sound is opened, as a share of the sound's length, and the least there
#: is in view then, in seconds: a sound of a few ticks is not opened with each tick as wide as a hand.
OPEN_AIR = 0.25
OPEN_LEAST = 0.5
#: What is left free between two variants: this share of the longest of them.
VARIANT_GAP = 0.2
#: How much of its bar's colour the copy has that a repeat of the echo at full strength is drawn as, and how far a
#: copy keeps from the upper and the lower edge of its bar.
ECHO_SHARE = 0.8
ECHO_INSET = 2
#: What an entry of a project's sounds can be, as `bank.entry_kind` names it: one sound, variants, parts, held.
KINDS = ("def", "defs", "parts", "held")
#: The colour of each zone of a held sound, its start, its loop and its end, which its label is in. The ground
#: of a zone goes this far towards it.
ZONES = (theme.GOOD, theme.MUSTARD, theme.SALMON)
TINT = 0.14
#: How many ticks long the note is that a click places, until one has been placed by a drag or stretched.
LENGTH = 6
#: How much of its colour a wave keeps while it is stale: its sound was changed, and the render of that has not come.
STALE = 0.4
#: What a handle can be: the label of a part, the two borders of a held sound, the line of the longest time, and
#: the flag of a note the sound is expected to play.
PART, START, LOOP, LIMIT, MARK = "part", "start", "loop", "limit", "mark"
#: What a right press asks a menu for: that of the label of a variant or a part, or that of the ruler. That of a
#: flag is asked for as `MARK`.
LABEL, RULER = "label", "ruler"
#: How far to each side of its line an upright handle is drawn, and is taken by the mouse.
GRIP = 2
#: The flag of a note the sound is expected to play: how wide and how high it is before its line, where the ruler
#: has no number, and its colour.
FLAG_W = 4
FLAG_H = 4
FLAG = theme.MUSTARD
#: From a side of the tab of a label to its text, and from the last tab to the sign that adds one.
TAB_PAD = 3
PLUS_GAP = 2
#: The line of the longest time is dragged in steps of so many seconds, and to no less than one of them.
LIMIT_STEP = 0.01
#: The clock goes on to a longest time that is no more than so many times as long as the sound, and then by this
#: share of it further, so that the line stands clear of the edge.
LIMIT_REACH = 2.0
LIMIT_AIR = 0.05
#: What one of the sounds is called that the sign adds to: an entry of any other kind has none.
ADDED = {"defs": "variant", "parts": "part"}

#: What the wheel does anywhere on the view, as the help line says it, and what the two scroll bars say.
WHEEL_HELP = "along the keys, shift along the time, ctrl zooms"
ALONG_HELP = told("drag the bar to go along the keys, or press beside it to jump", wheel=WHEEL_HELP)
ACROSS_HELP = told("drag the bar to go along the time, or press beside it to jump", wheel=WHEEL_HELP)
#: What a right press does in the ruler, as the help line says it, and what is said of a wave with no note to read.
RULER_HELP = "a longest time, or a note to expect"
NO_PITCH = "no clear pitch"
#: What an edit in the velocity lane is called in the list of what can be undone.
VELOCITY_DONE = "velocity changed"

# What was done to the notes by the gestures that are listed with a count of notes.
_DONE = {PLACE: "placed", MOVE: "moved", STRETCH: "stretched", REMOVE: "removed"}
_NO_MODS: frozenset[str] = frozenset()
_CTRL = frozenset({"ctrl"})


def _px(value: float) -> int:
    """The whole pixel nearest to a place. A half goes up."""
    return math.floor(value + 0.5)


def places(shapes: Sequence[tuple[str, float, Shape]], apart: bool = False) -> list[float]:
    """The second on the view's clock at which the clock of each shape starts.

    That is the second the shape is at. But shapes that are `apart`, as variants are, are laid one after the
    other, with `VARIANT_GAP` of the longest of them left free between two.
    """
    if not apart or not shapes:
        return [float(at) for _, at, _ in shapes]
    gap = VARIANT_GAP * max(shape.seconds for _, _, shape in shapes)
    found, at = [], 0.0
    for _, _, shape in shapes:
        found.append(at)
        at += shape.seconds + gap
    return found


def stretch(starts: Sequence[float], at: float) -> int | None:
    """Which of the stretches of a line a place is in, counted from 0: each goes from its start to the next start.

    The starts are in order, and the last stretch has no end. None for a place before the first of them.
    """
    found = None
    for index, start in enumerate(starts):
        if at >= start:
            found = index
    return found


def on_tick(second: float) -> float:
    """The second at which the tick nearest to a second begins, to the millionth. None is before second 0."""
    return round(max(0, math.floor(second * TICK_HZ + 0.5)) / TICK_HZ, 6)


def limit_lines(laid: Sequence[tuple[str, float, Shape]], kind: str, limit: float | None) -> list[tuple[int, float]]:
    """Where the line of a longest time stands among some shapes as they are laid: (shape, second of the clock).

    Variants have a clock each, so each has its line, where that falls before the next variant starts. An
    entry of any other kind has the one clock and the one line. With no limit there is none.
    """
    if limit is None:
        return []
    if kind != "defs":
        return [(0, float(limit))]
    starts = [place for _, place, _ in laid]
    return [(index, start + limit) for index, (start, end) in enumerate(zip(starts, [*starts[1:], math.inf]))
            if start + limit < end]


def mark_lines(laid: Sequence[tuple[str, float, Shape]], kind: str,
               marks: Sequence[dict]) -> list[tuple[int, float]]:
    """Where the flags of the notes a sound is expected to play stand among some shapes: (note, second of the clock).

    A note is one of an entry's `check`, counted from 0, and is expected at a second of the sound's own clock.
    Variants have a clock each, so each has a flag for it, as each has a line of the longest time.
    """
    return [(index, second) for index, mark in enumerate(marks)
            for _, second in limit_lines(laid, kind, mark["at"])]


def tint(ground: Colour, zone: int) -> Colour:
    """The colour a ground has in a zone of a held sound, counted from 0: `TINT` of the way to the zone's colour."""
    return mix(ground, ZONES[zone], TINT)


def opened(height: int, low: int, high: int) -> float:
    """The pitch of the key at the upper edge of a grid of a height that is opened on notes from low to high.

    As many keys stand over the notes as under them. Notes that reach further than the grid does have their
    middle in the middle of it.
    """
    return float(high + (height // KEY_H - (high - low + 1)) // 2)


def ruler_step(scale: float) -> int:
    """How far apart the marks of the ruler are on a clock of `scale` pixels a second, as a power of ten of seconds.

    It is the finest at which the numbers of the marks have room: -1 is tenths of a second and 0 is seconds. A
    clock zoomed far in goes on to hundredths and thousandths, and a long one to tens and hundreds.
    """
    for power in range(RULER_FINEST, RULER_COARSEST):
        # A number is a figure, and then a point and a figure for each power under 0, or a nought for each over it.
        if 10.0 ** power * scale >= Font.width("0" * (1 + abs(power) + (power < 0))) + RULER_AIR:
            return power
    return RULER_COARSEST


def voice_colour(colour: Colour, voice: int) -> Colour:
    """The colour of the bars of a voice, counted from 0.

    The first voice has the view's own colour. Each one after it has the next of `theme.CHANNELS`, counted on from
    the view's colour where that is one of them, and round to the first after the last.
    """
    if voice == 0:
        return colour
    first = theme.CHANNELS.index(colour) if colour in theme.CHANNELS else -1
    return theme.CHANNELS[(first + voice) % len(theme.CHANNELS)]


def echo_colour(colour: Colour, strength: float) -> Colour:
    """The colour of the copy of a bar that a repeat of the echo is drawn as: the weaker the repeat, the fainter.

    It goes by the square root of the strength, from 0 to 1, so that a repeat a tenth as strong is still seen.
    """
    return mix(theme.BACKGROUND, colour, ECHO_SHARE * math.sqrt(min(max(strength, 0.0), 1.0)))


def joined(ends: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """The columns of a wave made to meet: each is stretched to touch the one before, where a gap lay between.

    Each is the y of a column's upper end and of its lower end. Zoomed in far, a column holds few samples, and the
    next can begin well above or below where it ended: left as they are, the wave is strokes and not a line.
    """
    met: list[tuple[int, int]] = []
    for at, (upper, lower) in enumerate(ends):
        if at:
            before_upper, before_lower = ends[at - 1]
            upper = min(upper, before_lower + 1)
            lower = max(lower, before_upper - 1)
        met.append((upper, lower))
    return met


@dataclass(frozen=True)
class Map:
    """Where the seconds and the pitches of a sound fall in the rectangle of its view, and back.

    rect is the view's rectangle: the keyboard, the ruler, the grid and the two scroll bars are in it. seconds is
    how long the clock is, start the second at the left edge of the grid, and zoom how many steps the clock is
    zoomed in. shown is how many seconds are as wide as the grid at zoom 0: with 0 it is the whole clock, and
    with fewer the rest of the clock is reached by going along it. top is the pitch of the key at the upper edge of the
    grid, which can be a fraction: then part of a key is cut off at that edge. A key is `KEY_H` high. labels is
    whether the shapes are named, in a strip of their own over the grid. notes is whether there is a grid: with
    none the wave has the room under the ruler. foot is the box of the wave, which stands where the owner of the
    view puts it, at the foot of the editor. The sums of the clock and of the keys are those of `kit`, which
    every view of notes has.
    """

    rect: Rect
    seconds: float = 1.0
    start: float = 0.0
    zoom: int = 0
    top: float = 72.0
    labels: bool = False
    notes: bool = True
    foot: Rect = (0, 0, 0, 0)
    shown: float = 0.0

    @property
    def ruler(self) -> Rect:
        """The strip along the top in which the seconds are numbered, as wide as the grid."""
        x, y, w, h = self.rect
        return within((x + KEYS_W, y, w - KEYS_W - BAR, RULER_H), self.rect)

    @property
    def names(self) -> Rect:
        """The strip under the ruler in which the shapes are named. It is nothing high where they have no names."""
        x, y, w, h = self.ruler
        return within((x, y + h, w, LABELS_H if self.labels else 0), self.rect)

    @property
    def grid(self) -> Rect:
        """Where the bars are: what the keys, the strips and the two scroll bars leave of the rectangle."""
        x, y, w, h = self.names
        foot = self.rect[1] + self.rect[3]
        return x, y + h, w, max(0, foot - y - h - BAR) if self.notes else 0

    @property
    def keys(self) -> Rect:
        """The keyboard, down the left of the grid."""
        _, y, _, h = self.grid
        return within((self.rect[0], y, KEYS_W, h), self.rect)

    @property
    def along(self) -> Rect:
        """The upright scroll bar, down the right of the grid: the one that goes along the keys."""
        x, y, w, h = self.grid
        return within((x + w + theme.SCROLL_GAP, y, theme.SCROLL_W, h), self.rect)

    @property
    def across(self) -> Rect:
        """The lying scroll bar, under the grid: the one that goes along the time."""
        x, y, w, h = self.grid
        return within((x, y + h + theme.SCROLL_GAP, w, theme.SCROLL_W if h else 0), self.rect)

    @property
    def wave(self) -> Rect:
        """Where the wave is drawn: the stretch of its box that lies under the grid, on the same clock.

        With no grid there is no such box, and the wave has all there is under the ruler.
        """
        x, y, w, h = self.grid
        if self.notes:
            return within((x, self.foot[1], w, self.foot[3]), self.foot)
        return x, y, w, max(0, self.rect[1] + self.rect[3] - y)

    @property
    def box(self) -> Rect:
        """The box the wave is drawn in, with its outline: the one at the foot, or with no grid all of `wave`."""
        return self.foot if self.notes else self.wave

    @property
    def scale(self) -> float:
        """How many pixels wide a second is."""
        return ZOOM_STEP ** self.zoom * max(1, self.grid[2]) / self.fitted

    @property
    def fitted(self) -> float:
        """How many seconds are as wide as the grid at zoom 0: `shown`, or with none of it the whole clock."""
        return self.shown if self.shown > 0 else self.seconds if self.seconds > 0 else 1.0

    @property
    def least(self) -> int:
        """The least zoom there is: 0, or under it the first at which the whole clock is in view."""
        if self.shown <= 0 or self.seconds <= self.shown:
            return 0
        return math.floor(math.log(self.shown / self.seconds, ZOOM_STEP))

    @property
    def zero(self) -> int:
        """The y of the line of the wave at which a sample is 0."""
        return self.wave[1] + self.wave[3] // 2

    def x(self, second: float) -> int:
        """The x of a second of the clock."""
        return x_at(self.grid[0], self.start, self.scale, second)

    def second_at(self, x: int) -> float:
        """The second of the clock at the left edge of the pixels at an x."""
        return moment_at(self.grid[0], self.start, self.scale, x)

    def tick_at(self, x: int, at: float = 0.0) -> float:
        """The tick at an x on the clock of a sound that starts at second `at` of the view's: sixty to a second.

        Its whole part is the tick that the left edge of the pixels at x is in. No x is before tick 0.
        """
        return max(0.0, (self.second_at(x) - at) * TICK_HZ)

    def y(self, pitch: int) -> int:
        """The y of the upper edge of the key of a pitch, and of its row of the grid."""
        return key_rect(self.grid, self.top, KEY_H, pitch)[1]

    def pitch_at(self, y: int) -> int:
        """The pitch whose key, and whose row of the grid, a y is on."""
        return key_at(self.grid, self.top, KEY_H, y)

    def bar(self, start: float, seconds: float, pitch: float) -> Rect:
        """The rectangle of the bar of a note that starts at a second and is held for some: in the row of its key.

        It starts a pixel after its second, so that two notes in a row are two bars, and is a pixel wide at the
        least. The lowest line of pixels of its key is left free.
        """
        left = self.x(start)
        return left + 1, self.y(round(pitch)), max(1, self.x(start + seconds) - left - 1), KEY_H - 1

    def column(self, lowest: float, highest: float) -> tuple[int, int]:
        """The y of the upper end and of the lower end of a column of the wave, both of them in the column.

        They are where the highest and the lowest sample of its stretch fall. Full scale is `WAVE_AIR` short of
        the outline of the box, and what is past full scale is drawn there. Silence is the one pixel at zero.
        """
        reach = max(0, (self.wave[3] - 1) // 2 - 1 - WAVE_AIR)
        upper, lower = (self.zero - _px(min(max(sample, -1.0), 1.0) * reach) for sample in (highest, lowest))
        return upper, lower

    def pitches(self) -> range:
        """The pitches of which at least a line of pixels is on the grid, from the lowest."""
        _, y, _, h = self.grid
        return range(self.pitch_at(y + h - 1), self.pitch_at(y) + 1) if h else range(0)

    def marks(self) -> list[int]:
        """The marks of the ruler whose line is on the grid, each as how many steps of the ruler after second 0."""
        x, _, w, _ = self.grid
        step = 10.0 ** ruler_step(self.scale)
        first = max(0, math.floor(self.second_at(x) / step))
        last = math.ceil(self.second_at(x + w) / step)
        return [mark for mark in range(first, last + 1) if x <= self.x(mark * step) < x + w]

    def zoomed(self, zoom: int, x: int) -> Map:
        """This map at another zoom, no less than `least` and no more than `ZOOM_MOST`. The second at `x` stays
        at that x."""
        fresh = replace(self, zoom=min(max(zoom, self.least), ZOOM_MOST))
        return replace(fresh, start=zoomed(self.grid[0], self.start, self.scale, fresh.scale, x))

    def within(self) -> Map:
        """This map, moved no further than its limits let it.

        The clock starts no sooner than second 0, and ends no later than it does. The zoom is no less than
        `least`. No key is shown under `soundedit.LOWEST` or over `soundedit.HIGHEST`, which are the ends of
        what a note can be placed at.
        """
        zoom = min(max(self.zoom, self.least), ZOOM_MOST) if self.shown > 0 else self.zoom
        if self.shown > 0:
            last = max(0.0, self.seconds - self.shown * ZOOM_STEP ** -zoom)
        else:
            last = max(0.0, self.seconds) * (1 - ZOOM_STEP ** -zoom)
        top = min(float(HIGHEST), max(self.top, LOWEST - 1 + self.grid[3] / KEY_H))
        return replace(self, start=min(max(self.start, 0.0), last), zoom=zoom, top=top)

    def scrolls(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """What the two scroll bars show, the one along the keys first.

        Each is three lengths in pixels: all there is to go along, how much of it the grid has in view, and how
        much of it lies before the view. All there is are the keys from the highest down to the lowest, and
        the clock from its start to its end.
        """
        _, _, w, h = self.grid
        return (((HIGHEST - LOWEST + 1) * KEY_H, h, _px((HIGHEST - self.top) * KEY_H)),
                (_px(max(0.0, self.seconds) * self.scale), w, _px(self.start * self.scale)))


def beside(lefts: Sequence[int], widths: Sequence[int]) -> list[int]:
    """The x at which each of some tabs is laid, by the x each starts at and how wide each is: none lies on another.

    They are taken from the left, and those that start at one x in the order they are given. A tab that would
    lie on the one laid before it stands just after that one. Any other stays where it starts, and so does the
    first of those that start together.
    """
    found, end = list(lefts), None
    # The order is that of where they start: it is settled before any of them is laid elsewhere.
    for index in sorted(range(len(found)), key=found.__getitem__):
        if end is not None and found[index] < end:
            found[index] = end
        end = found[index] + widths[index]
    return found


def tabs(seen: Map, laid: Sequence[tuple[str, float, Shape]], parts: bool = False) -> list[Rect]:
    """The tab of each label in the strip of the names: from where its shape starts, `TAB_PAD` round its text.

    The shapes of `parts` can start together. Their tabs are laid as `beside` lays them, so that each can be
    taken by the mouse: one that would lie on another stands after it, and not where its part starts.
    """
    _, y, _, h = seen.names
    lefts = [seen.x(place) for _, place, _ in laid]
    widths = [Font.width(label) + 2 * TAB_PAD for label, _, _ in laid]
    return [(left, y, wide, h) for left, wide in zip(beside(lefts, widths) if parts else lefts, widths)]


def plus(seen: Map, laid: Sequence[tuple[str, float, Shape]], parts: bool = False) -> Rect:
    """The box of the sign that adds a sound, in the strip of the names: `PLUS_GAP` after the tab that ends
    furthest on, as `tabs` lays them, and no further on than the strip goes. It is `theme.PLUS` wide."""
    x, y, w, h = seen.names
    after = max((left + width for left, _, width, _ in tabs(seen, laid, parts)), default=x) + PLUS_GAP
    return max(x, min(after, x + w - theme.PLUS)), y, theme.PLUS, h


def called(kind: str, index: int) -> str:
    """What one sound of an entry is called, by the kind of the entry and its place in it, counted from 0.

    "variant 2", "part 1", and "the note" or "the end" of a held sound. The one sound of an entry is called
    nothing.
    """
    if kind == "held":
        return ("the note", "the end")[index]
    return {"defs": f"variant {index + 1}", "parts": f"part {index + 1}"}.get(kind, "")


@lru_cache(maxsize=None)
def spelt(pitch: int) -> str:
    """A pitch as the note of a sound names it, with sharps as `soundedit.moved` spells a note: 61 is c#4."""
    middle = {"voices": [{"inst": "", "notes": [[0, name(MIDDLE_C), 0]]}]}
    return soundedit.moved(middle, {(0, 0)}, 0, pitch - MIDDLE_C)["voices"][0]["notes"][0][1]


@dataclass(frozen=True)
class Span:
    """One sound of an entry where it lies on the clock of the view, for a hand to change.

    at is the second of the view's clock at which the sound's own clock starts, and until the second at which
    its room ends: what a bar has past that is not drawn, and is not the hand's. defn is the definition of the
    sound. echo has the repeats its echo brings, as a `Shape` has them.
    """

    at: float
    until: float
    defn: dict
    echo: tuple[tuple[float, float], ...] = ()


def spans(entry: dict, laid: Sequence[tuple[str, float, Shape]]) -> tuple[Span, ...]:
    """The sounds of an entry as spans, in the order `soundedit.sound_in` counts them, by its shapes as they are laid.

    One sound, a variant and a part lie where their shape does. The note of a held sound lies under its start
    and its loop, and its end under the third of its shapes. A variant has room up to where the next starts,
    and the note of a held sound up to where its end does. Parts lie over one another, so the room of a part
    has no end, and neither has that of the last sound of any entry. Shapes that are not those of the entry
    are refused.
    """
    kind = soundedit.kind_of(entry)
    count = {"def": 1, "held": 3}.get(kind) or len(entry[kind])
    if len(laid) != count:
        raise RowSynthError(f"entry: must be the entry the shapes were made of: it has {count}, and there are "
                            f"{len(laid)}")
    firsts = (0, 2) if kind == "held" else range(count)
    starts = [laid[first][1] for first in firsts]
    ends = [math.inf] * len(starts) if kind == "parts" else [*starts[1:], math.inf]
    return tuple(Span(start, end, soundedit.sound_in(entry, index), laid[first][2].echo)
                 for index, (first, start, end) in enumerate(zip(firsts, starts, ends)))


def _notes(span: Span) -> list[list]:
    """The notes of a span, voice by voice."""
    return [voice["notes"] for voice in span.defn["voices"]]


def bars(seen: Map, span: Span) -> list[tuple[Pick, Rect, bool]]:
    """The bars of the notes of a span as a map has them, in the order they are drawn: (note, rectangle, whole).

    A note is (voice, index). A bar is cut where the room of its span ends, and a note that starts past there
    has none. `whole` is whether the end of the bar is that of its note, and not such a cut.
    """
    edge = seen.x(span.until) if span.until < math.inf else None
    found = []
    for voice, held in enumerate(span.defn["voices"]):
        for index, (tick, note, ticks, *_) in enumerate(held["notes"]):
            x, y, w, h = seen.bar(span.at + tick / TICK_HZ, ticks / TICK_HZ, midi(note))
            wide = w if edge is None else min(w, edge - x)
            if wide > 0:
                found.append(((voice, index), (x, y, wide, h), wide == w))
    return found


def note_at(seen: Map, laid: Sequence[Span], pos: Pos) -> tuple[int, Pick, Rect, bool] | None:
    """The note whose bar is under a position on the grid: which of some spans it is in, and what `bars` has of it.

    None where no bar is.
    """
    if inside(seen.grid, pos):
        # The last is drawn on top of the others.
        for which in reversed(range(len(laid))):
            for pick, rect, whole in reversed(bars(seen, laid[which])):
                if inside(rect, pos):
                    return which, pick, rect, whole
    return None


def span_at(laid: Sequence[Span], second: float, last: int = 0) -> int | None:
    """Which of some spans has a second of the view's clock in its room, or None where none has.

    Parts lie over one another, so several can have it. Then it is the span `last`, which is the one the hand
    was in before, if that is one of them, and else the one of them that starts last.
    """
    rooms = [which for which, span in enumerate(laid) if span.at <= second < span.until]
    return last if last in rooms else (rooms[-1] if rooms else None)


def begins(seen: Map, laid: Sequence[Span], pos: Pos, button: int = 1, mods: frozenset[str] = _NO_MODS,
           last: int = 0) -> tuple[str, int | None, Pick | None]:
    """Which gesture a press at a position begins among the sounds of an entry: what, in which sound, on which note.

    On the grid the right button removes. The left one on a bar picks its note with ctrl, and else moves it,
    or stretches it on the last `END_W` pixels of the bar: a bar narrower than twice that is stretched at its
    last pixel only, and one that is cut where the room of its sound ends is not stretched at all. Where no
    bar is, it draws a box with shift, and else places a note where the key is one from `soundedit.LOWEST` to
    `soundedit.HIGHEST`. The sound is that of the bar, and with no bar the one `span_at` gives for `last`.
    Any other press begins nothing: "", None and None.
    """
    if button not in (1, 3) or not inside(seen.grid, pos):
        return "", None, None
    hit = note_at(seen, laid, pos)
    if hit is not None:
        which, pick, (x, _, w, _), whole = hit
        if button == 3:
            return REMOVE, which, pick
        if "ctrl" in mods:
            return PICK, which, pick
        end = whole and pos[0] >= x + w - (END_W if w >= 2 * END_W else 1)
        return (STRETCH if end else MOVE), which, pick
    which = span_at(laid, seen.second_at(pos[0]), last)
    if which is None:
        return "", None, None
    if button == 3:
        return REMOVE, which, None
    if "shift" in mods:
        return BOX, which, None
    if soundedit.LOWEST <= seen.pitch_at(pos[1]) <= soundedit.HIGHEST:
        return PLACE, which, None
    return "", None, None


@dataclass
class Gesture:
    """A gesture that is under way: what a press began, in which sound, and where the mouse has been since.

    kind is what `begins` called it, button the button that was pressed, which the sound it is in, and pick
    the note it was pressed on, or None. seen, span and grid are the sums of the view, that sound and the
    ticks in a step of the grid, all as they were at the press. picked is the picked notes as the press left
    them. new is the note that a click here places, as (voice, tick, note, ticks). every is whether a box takes
    the notes of every voice, and not those of the voice of `new` alone. start is where the press was, and
    help what the help line said it does. pos is where the mouse is now, and dragged whether it has been more
    than `SLACK` pixels from the press: until then nothing is moved, stretched, made longer or boxed. passed
    has the notes a removal passed over. heard is the pitch at which the note under the press was heard last.
    """

    kind: str
    button: int
    which: int
    pick: Pick | None
    seen: Map
    span: Span
    grid: float
    picked: Picked
    new: tuple[int, float, str, float]
    every: bool
    start: Pos
    help: str = ""
    pos: Pos = field(init=False)
    dragged: bool = field(default=False, init=False)
    passed: dict[Pick, None] = field(default_factory=dict, init=False)
    heard: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.pos = self.start
        self.heard = None if self.pick is None else midi(self.grabbed()[1])
        self._pass(self.start)

    def to(self, pos: Pos) -> None:
        """Take the mouse to another position, by way of every pixel between this one and that."""
        if self.kind == REMOVE:
            for step in path(self.pos, pos):
                self._pass(step)
        self.pos = pos
        self.dragged = self.dragged or max(abs(pos[0] - self.start[0]), abs(pos[1] - self.start[1])) > SLACK

    def _pass(self, pos: Pos) -> None:
        """Take note of the note of the sound that a removal passes at a position."""
        hit = note_at(self.seen, (self.span,), pos) if self.kind == REMOVE else None
        if hit is not None:
            self.passed[hit[1]] = None

    @property
    def box(self) -> Rect | None:
        """The box that is being drawn: from the pixel of the press to the pixel the mouse is on. None for no box."""
        if self.kind != BOX or not self.dragged:
            return None
        (x, y), (now_x, now_y) = self.start, self.pos
        return min(x, now_x), min(y, now_y), abs(now_x - x) + 1, abs(now_y - y) + 1

    def moved(self) -> tuple[float, int]:
        """How many ticks and how many keys the mouse is from where it was pressed, and neither until it has dragged.

        The ticks are from the step of the grid that the press was in to the step the mouse is in now.
        """
        if not self.dragged:
            return 0, 0
        seen, at = self.seen, self.span.at
        return (soundedit.snapped(seen.tick_at(self.pos[0], at), self.grid)
                - soundedit.snapped(seen.tick_at(self.start[0], at), self.grid),
                seen.pitch_at(self.pos[1]) - seen.pitch_at(self.start[1]))

    def gives(self) -> tuple[dict, Picked]:
        """What a release where the mouse is now gives: the sound, and the notes of it that are picked.

        Each is what the rule of `soundedit` makes of it, so no note is cut for another. A move or a stretch
        that has not dragged, or is back in the step and on the key it began in, gives the sound there was.
        """
        defn, seen, at = self.span.defn, self.seen, self.span.at
        by_ticks, by_pitch = self.moved()
        if self.kind == PLACE:
            voice, tick, note, ticks = self.new
            # Dragged, the note ends with the step of the grid the mouse is in, and is no shorter than a step.
            ticks = max(by_ticks, 0) + self.grid if self.dragged else ticks
            return soundedit.placed(defn, voice, tick, note, ticks), frozenset()
        if self.kind == MOVE and (by_ticks or by_pitch):
            return soundedit.moved(defn, self.picked, by_ticks, by_pitch), self.picked
        if self.kind == STRETCH and by_ticks:
            return soundedit.stretched(defn, self.picked, by_ticks, self.grid), self.picked
        if self.kind == BOX:
            if not self.dragged:
                return defn, frozenset()
            touched = soundedit.boxed(defn, seen.tick_at(self.start[0], at), seen.tick_at(self.pos[0], at),
                                      seen.pitch_at(self.start[1]), seen.pitch_at(self.pos[1]))
            return defn, frozenset(pick for pick in touched if self.every or pick[0] == self.new[0])
        if self.kind == REMOVE and self.passed:
            return soundedit.removed(defn, self.passed), frozenset()
        return defn, self.picked

    def grabbed(self) -> list:
        """The note under the press of a move or a stretch, as a release where the mouse is now leaves it."""
        voice, index = self.pick
        return self.gives()[0]["voices"][voice]["notes"][index]

    @property
    def label(self) -> str:
        """The few words for what the gesture did to the notes, as a list of what can be undone has them."""
        count = {PLACE: 1, REMOVE: len(self.passed)}.get(self.kind, len(self.picked))
        return counted(count, _DONE[self.kind])


@dataclass
class Pull:
    """A handle that is pulled along the clock: which one, where the press was, and what it stands at.

    kind is what the handle is, and which the part it is of, the variant whose line it is, or the expected note
    whose flag it is. seen is the sums of the view at the press, start where the press was, and was what the
    handle stood at then: a second for a part, for a longest time and for an expected note, and ticks for a
    border of a held sound. help is what the help line said of it.
    pos is where the mouse is now, dragged whether it has been more than `SLACK` pixels to a side of the press,
    and value what a release where the mouse is leaves the handle at. wave is that of a part by itself, where
    the owner of the view gave one.
    """

    kind: str
    which: int
    seen: Map
    start: Pos
    was: float
    help: str = ""
    pos: Pos = field(init=False)
    dragged: bool = field(default=False, init=False)
    value: float = field(init=False)
    wave: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.pos, self.value = self.start, self.was


class SoundView(Widget):
    """The view of a sound: its notes as bars on a grid of seconds and keys, beside a keyboard, and its wave.

    It is laid out as the piano roll of a song is, on a clock of seconds. In its rectangle are the keyboard, the
    ruler with the strip of the names under it, the grid and the two scroll bars. The wave has a box of its
    own, `foot`, which stands where the owner puts it, at the foot of the editor: it is on the clock of the grid,
    and the mouse is the view's there too.

    The view has five parts that are widgets of their own. Four are the parts of `kit` that every view of notes
    has: `keys` is the keyboard down its left, `along` the scroll bar down its right, which goes along the keys,
    `across` the one under the grid, which goes along the time, and `lane` the velocity lane, which stands
    where its owner puts it. The fifth is `more`, the sign after the last label of variants and of parts.
    `parts` has the five. Their owner adds them to the Ui after the view. They are seen while the view is, the
    sign only where there is something to add to: the view sees to that itself. A key that one of the parts is
    handed, which it is after a press on it, is the view's as if the view had the focus.

    The keyboard does not fit itself to the notes. `top` is the pitch of the key at the upper edge of the grid,
    and a key is `KEY_H` high. The first sound that is shown is opened on its notes, and so is the first after
    `fit`: after that the view stays where it is left. The wheel is read by `kit.wheel`: it goes along the keys,
    with shift along the time, and with ctrl it zooms the time round the mouse. A press on a key of the keyboard
    is heard in the instrument of the picked voice: `hear` does that for any pitch.

    `playhead` is the second of its clock that is playing, or None: the owner sets it each frame, and `clock`
    gives the second for a place in one of the shapes. `colour` is that of the kind of thing shown, `kind` what
    its entry is, and `seconds` how long its clock is. `start` and `zoom` are the view of the time, as a `Map`
    has them. `stale` is whether the wave is that of the sound as it was before a change: it is drawn dim.

    Shown shapes alone, it shows and changes nothing, and `help` says what the mouse is on: a note, a moment of
    the wave, the label of a variant. Shown the entry of the shapes as well, the mouse and the keys place, move,
    stretch and remove the notes of its sounds on a grid of ticks, and it saves nothing. When a gesture ends
    and the sound it was in is another than it was, `edited` is called with which sound of the entry that is, as
    `soundedit.sound_in` counts them, its new definition, and a few words for the list of what can be undone.
    The view goes on showing what `show` gave it last: its owner saves, and shows it the result. An owner that
    answers false has not saved: any other answer, and none, is taken for saved. `sounded` is called with the
    instrument of a voice and a pitch for the owner to let be heard: when a note is placed and saved, each
    time the note that the mouse moves comes to another pitch, and for a key of the keyboard. `refused` is
    called with the words of a change that the rules of `soundedit` refuse. `help` then says what the left
    press, the right press and the wheel do where the mouse is.

    `sound` is the sound of the entry that the hand is in: the one it pressed in last, on the grid or on the
    label of a sound. Its label is lit where it has a tab. `voice` is the voice a new note goes into, and the
    owner sets it. `grid` is the ticks in a step of the grid: a placed note begins on a step, and notes are
    moved and stretched by whole steps. `picked` is the picked notes of that sound, each as (voice, index), and
    `drawn` the sound as a gesture that is under way would leave it: `busy` is whether one is under way. A box
    picks the notes of the voice that is `voice`, and so do ctrl and a: with ctrl held as well, the box picks
    those of every voice, and so do ctrl, shift and a. A key that the Ui hands the view, which it does while
    the view has the focus, goes to `key`, and one that is not the view's to `typed`, which is its owner's.

    The lane has a bar for each note of the picked voice of that sound: how hard the note is struck, which is
    the fourth number of a note, and 1 where it has none. A drag in it is one edit, made when the button is let
    go and handed to `edited`. A velocity of 1, and the usual one a right press gives, is written as none.

    A press on the label of a variant, or in its stretch of the wave, picks that variant: the hand is in it,
    and with `chosen` set that is called with its number, from 0. Nothing is played by it. `played` is the
    shape whose label is lit as the one that sounds, or None: the owner sets it.

    What belongs to the kind of the entry has handles, which are there once the owner has set `handled`. The
    label of a variant and of a part is a tab. That of a part is dragged along the clock to where the part is
    to start, on a tick and never before second 0: its bars follow, and so does its own wave where `alone`
    gives the wave of a part by itself. The tabs of parts that start together stand side by side, so that each
    is taken by itself. The two borders of a held sound are dragged in the strip of the labels, and stop where
    `soundedit.held_with` keeps the start and the loop, by the instruments that `show` was given. Where the
    entry has a longest time, its `max_seconds`, a line stands at that second in the colour of a fault, and is
    dragged in the ruler. The clock goes on to a line that is no more than `LIMIT_REACH` times the sound away.
    Each note the entry is expected to play, one of its `check`, has a flag in the ruler at its second, which
    is dragged along the ruler. While a handle is dragged the view is `busy`, and at its release `handled` is
    called with what the handle is, which part, variant or note it is of, and what it stands at: the second of
    a part, of the longest time or of an expected note, or the ticks of the `start` or the `loop`. The view
    changes nothing itself.

    With `added` set, variants and parts have the sign `more` after their last label: a press on it calls
    `added` with None, and a click in the ruler of an entry of parts with ctrl held calls it with the second
    there, on a tick. With `asked` set, the right button calls it for a menu: with `LABEL`, the variant or the
    part, the second and the position on a label, with `MARK`, the note, its second and the position on a
    flag, and with `RULER`, the variant, the second of its clock and the position anywhere else in the ruler.
    """

    def __init__(self, rect: Rect, help: str = "") -> None:
        self.colour: Colour = theme.SOUND
        self.kind = KINDS[0]
        self.playhead: float | None = None
        self.played: int | None = None
        self.chosen: Callable[[int], object] | None = None
        self.edited: Callable[[int, dict, str], object] | None = None
        self.sounded: Callable[[str | dict, float], object] | None = None
        self.refused: Callable[[str], object] | None = None
        self.added: Callable[[float | None], object] | None = None
        self.asked: Callable[[str, int, float, Pos], object] | None = None
        self.handled: Callable[[str, int, float], object] | None = None
        self.alone: Callable[[int], np.ndarray | None] | None = None
        self.typed: Callable[[Event], object] | None = None
        self.seconds = 0.0
        self.start, self.zoom, self.top = 0.0, 0, float(MIDDLE_C)
        # How many seconds are as wide as the grid at zoom 0: 0 for the whole clock.
        self.shown = 0.0
        self.foot: Rect = (0, 0, 0, 0)
        self.stale = False
        self.grid = 1
        self.voice = 0
        # The shapes that are shown, each as (label, the second of the view's clock its own starts at, shape).
        self._laid: tuple[tuple[str, float, Shape], ...] = ()
        # The waves under them, each as (the second of the view's clock it starts at, its samples).
        self._waves: tuple[tuple[float, np.ndarray], ...] = ()
        self._rate = DEFAULT_RATE
        self._mouse: Pos = NOWHERE
        self._mods = _NO_MODS
        # The sounds of the entry that is shown, for the hand to change: none where only shapes are shown.
        self._spans: tuple[Span, ...] = ()
        self._sound = 0
        self._picked: Picked = frozenset()
        self._gesture: Gesture | None = None
        # What the last edit gave, until an entry is shown: which sound, its definition, and the picked notes.
        self._gave: tuple[int, dict, Picked] | None = None
        self._clip: Clip = ()
        # How long the last note was that was placed by a drag or stretched.
        self._length = LENGTH
        # The entry that is shown and the instruments of its project, its longest time, the notes it is
        # expected to play, and the handle that is pulled.
        self._entry: dict | None = None
        self._instruments: dict | None = None
        self._limit: float | None = None
        self._marks: tuple[dict, ...] = ()
        self._pull: Pull | None = None
        # Whether the keys were opened on the notes of a sound since the view was made or fitted.
        self._opened = False
        # The velocities a drag in the lane has given so far, by the note, and the note of each bar of the lane.
        self._levels: dict[Pick, int | float | None] = {}
        self._lane: list[Pick] = []
        # What was read of the wave last: where, and the words for its pitch there.
        self._read: tuple[object, str] = (None, "")
        nowhere = (0, 0, 0, 0)
        self.keys = Keys(nowhere, self.hear, told(wheel=WHEEL_HELP), WHEEL_HELP)
        self.keys.low, self.keys.high = LOWEST, HIGHEST
        self.along = ScrollBar(nowhere, self._go_along, ALONG_HELP)
        self.across = ScrollBar(nowhere, self._go_across, ACROSS_HELP)
        self.lane = Velocity(nowhere, self._weigh, self._weighed, align="right")
        self.more = Plus(nowhere, self._add)
        for part in (self.keys, self.along, self.across, self.lane):
            part.typed = self._keyed
        self._visible = True
        super().__init__(rect, help)

    @property
    def parts(self) -> tuple[Widget, ...]:
        """The five parts that are widgets of their own: the keyboard, the two scroll bars, the lane and the sign."""
        return self.keys, self.along, self.across, self.lane, self.more

    @property
    def rect(self) -> Rect:
        """The rectangle of the view: the keyboard, the ruler, the strip of the names, the grid and the scroll bars."""
        return self._rect

    @rect.setter
    def rect(self, rect: Rect) -> None:
        self._rect = rect
        self._sync()

    @property
    def visible(self) -> bool:
        """Whether the view is seen. Its parts are seen with it: the four of the notes while it shows notes, and
        the sign where there is something to add to."""
        return self._visible

    @visible.setter
    def visible(self, on: bool) -> None:
        self._visible = on
        self._seen()

    def _seen(self) -> None:
        """Have each part seen or not, by whether the view is and what it shows."""
        notes = self._visible and bool(self._laid)
        for part in (self.keys, self.along, self.across, self.lane):
            part.visible = notes
        self.more.visible = self._visible and self.adds

    @property
    def adds(self) -> bool:
        """Whether there is something to add to: an entry of variants or of parts, and somebody who is told."""
        return bool(self._spans) and self.added is not None and self.kind in ADDED

    def hit(self, pos: Pos) -> bool:
        """Whether a position is on the view: in its rectangle, or in the box of the wave under a grid."""
        return inside(self.rect, pos) or (bool(self._laid) and inside(self.foot, pos))

    @property
    def help(self) -> str:
        """What the presses and the wheel do where the mouse is, or the line the view was made with.

        While a gesture is under way, or a handle is pulled, it is what was said at the press that began it.
        """
        under_way = self._gesture or self._pull
        said = under_way.help if under_way is not None else self._tell(self._mouse, self._mods)
        return said or self._help

    @help.setter
    def help(self, line: str) -> None:
        self._help = line

    @property
    def map(self) -> Map:
        """The sums of the view as it is now, kept within their limits."""
        return Map(self.rect, self.seconds, self.start, self.zoom, self.top,
                   any(label for label, _, _ in self._laid), bool(self._laid), self.foot, self.shown).within()

    @property
    def sound(self) -> int:
        """Which sound of the entry the hand is in, as `soundedit.sound_in` counts them.

        The owner sets it to 0 when it is about to show another entry: then nothing is picked, and a gesture
        that is under way is dropped.
        """
        return self._sound

    @sound.setter
    def sound(self, which: int) -> None:
        self._sound, self._picked = which, frozenset()
        self._drop()

    @property
    def picked(self) -> Picked:
        """Which notes of the sound the hand is in are picked, each as (voice, index)."""
        return self._picked

    @property
    def drawn(self) -> dict | None:
        """The sound the hand is in as it is drawn: while a gesture is under way, what a release where the mouse is
        gives. None where nothing can be changed."""
        laid = self._now()[0]
        return laid[self._sound].defn if laid else None

    @property
    def busy(self) -> bool:
        """Whether a gesture is under way, on the grid or in the lane, or a handle is pulled: from the press that
        began it to its release, or until it is dropped."""
        return self._gesture is not None or self._pull is not None or self.lane.busy or bool(self._levels)

    @property
    def limit(self) -> float | None:
        """The longest time of the entry that is shown, in seconds, or None where it has none."""
        return self._limit

    def _drop(self) -> None:
        """Let go of the gesture that is under way and of the handle that is pulled, with nothing told of either."""
        self._gesture, self._pull, self._levels = None, None, {}
        self.lane.drop()

    def _moving(self) -> bool:
        """Whether a handle is dragged that takes sounds along with it: the label of a part, or a border."""
        pull = self._pull
        return pull is not None and pull.dragged and pull.kind in (PART, START, LOOP)

    def _placed(self) -> tuple[tuple[str, float, Shape], ...]:
        """The shapes where they lie: while a part or a border is dragged, where its release would leave them.

        A part lies at the second its label is at. The loop of a held sound starts where its start ends, and
        its end where the loop does.
        """
        pull, laid = self._pull, self._laid
        if not self._moving():
            return laid
        if pull.kind == PART:
            label, _, shape = laid[pull.which]
            return (*laid[:pull.which], (label, pull.value, shape), *laid[pull.which + 1:])
        lengths = {START: laid[1][1], LOOP: laid[2][1] - laid[1][1]}
        lengths[pull.kind] = soundview.ticks_to_seconds(pull.value, self._rate)
        (first, _, start), (second, _, loop), (third, _, end) = laid
        return ((first, 0.0, start), (second, lengths[START], loop), (third, lengths[START] + lengths[LOOP], end))

    def _now(self) -> tuple[tuple[Span, ...], Picked]:
        """The sounds to draw and the picked notes: with a gesture under way what it gives, and else what there is.

        While a part or a border is dragged, the sounds lie where its release would leave them. While the lane
        is dragged, the notes have the velocities it has given so far.
        """
        gesture, at = self._gesture, self._sound
        if gesture is None:
            if self._moving():
                return spans(self._entry, self._placed()), self._picked
            if not self._levels:
                return self._spans, self._picked
            span, defn = self._spans[at], self._spans[at].defn
            for pick, velocity in self._levels.items():
                defn = soundedit.with_velocity(defn, {pick}, velocity)
            return (*self._spans[:at], replace(span, defn=defn), *self._spans[at + 1:]), self._picked
        defn, picked = gesture.gives()
        return (*self._spans[:gesture.which], replace(gesture.span, defn=defn),
                *self._spans[gesture.which + 1:]), picked

    def show(self, shapes: Sequence[tuple[str, float, Shape]], samples: Sequence[np.ndarray], rate: int,
             colour: Colour, kind: str = "def", entry: dict | None = None, stale: bool = False,
             instruments: dict | None = None) -> None:
        """Show some shapes, as `soundview.shapes` gives them, over the samples of their render at a rate.

        `kind` is what the entry of the shapes is, as `bank.entry_kind` names it. The shapes of "defs" are
        variants, laid one after the other, and those of "held" are its zones, each tinted in its colour.

        The samples are rows, as `render_entry` gives its buffers. Several rows are one for each shape: each is
        the wave of its shape, and starts where that starts. One row alone starts where the clock does: it is
        the wave of one sound or of parts added up, and with no shapes a wave alone, with no grid over it. No
        rows are an empty wave. `stale` is for rows of the sound as it was before a change. `colour` is that of
        the kind of thing shown. The view stays where it was, as far as the new clock lets it. The keys are
        set on the notes only for the first sound there is to show, and for the first after `fit`.

        The clock ends where the last shape or wave does. With an `entry`, whose notes can be changed, it goes
        on past there: by as long again as the sound, and by no less than `ROOM_LEAST` seconds, so that there
        is room to put a later note and to make a note longer. The sound then gets longer, and the room is
        after its new end. Such a sound is opened with its whole length in view and `OPEN_AIR` of that after
        it, or as far as the line of its longest time where that is near, and with no less than `OPEN_LEAST`
        seconds. From then on a second is as wide as the zoom makes it, however long the sound gets.

        `entry` is the entry of a project's sounds that the shapes were made of: with it the notes can be
        changed by hand, and with None they are shown and no more. What is picked stays picked while the sound
        it is in has the notes it had. The sound that the last edit gave is shown with what that edit left
        picked. With any other notes nothing is picked, and a gesture that is under way is dropped when any
        sound of the entry is another than it was. A handle that is pulled is dropped by any other entry.
        `instruments` are those of the project the entry is in, by which the borders of a held sound are kept
        inside its note.

        What is no kind, no rows, no wave or no rate is refused, and so are rows that are neither one nor one
        for each shape, and an entry with other shapes than these. Then nothing changes.
        """
        if kind not in KINDS:
            raise RowSynthError(f"kind: must be one of {', '.join(KINDS)}, not {kind!r}")
        if not isinstance(samples, (list, tuple)):
            raise RowSynthError("samples: must be a list of rows, each an array of one dimension")
        for row in samples:
            soundview.columns(row, 0)
        if len(samples) not in (0, 1, len(shapes)):
            raise RowSynthError(f"samples: must be one row, or a row for each of the {len(shapes)} shapes, "
                                f"not {len(samples)} rows")
        soundview.ticks_to_seconds(0, rate)
        laid = tuple((label, place, shape)
                     for (label, _, shape), place in zip(shapes, places(shapes, kind == "defs")))
        fresh = () if entry is None else spans(entry, laid)
        each = len(samples) == len(laid) > 1
        waves = tuple((laid[index][1] if each else 0.0, row) for index, row in enumerate(samples))
        self._laid, self._waves, self._rate, self.colour, self.kind = laid, waves, rate, colour, kind
        self.stale = stale
        if entry != self._entry:
            self._pull = None
        limit = None if entry is None else entry.get("max_seconds")
        numbered = isinstance(limit, (int, float)) and not isinstance(limit, bool) and limit > 0
        self._entry, self._instruments, self._limit = entry, instruments, limit if numbered else None
        self._marks = () if entry is None else tuple(entry.get("check", ()))
        self._take(fresh)
        end = max([0.0] + [place + shape.seconds for _, place, shape in laid]
                  + [place + len(row) / rate for place, row in waves])

        def reach(length: float) -> float:
            """A length of the clock, or where the longest time is near after it, as far as that and a little on."""
            near = numbered and kind != "defs" and length / (1 + LIMIT_AIR) < limit <= LIMIT_REACH * length
            return limit * (1 + LIMIT_AIR) if near else length

        roomy = entry is not None and bool(laid)
        self.seconds = reach(end + max(end, ROOM_LEAST) if roomy else end)
        if not roomy:
            self.shown = 0.0
        elif self.shown <= 0:
            self.shown = max(reach(end), end * (1 + OPEN_AIR), OPEN_LEAST)
        if laid and not self._opened:
            pitches = [round(bar.pitch) for _, _, shape in laid for bar in shape.bars] or [MIDDLE_C]
            self._opened, self.top = True, opened(self.map.grid[3], min(pitches), max(pitches))
        self._see(self.map)

    def _take(self, fresh: tuple[Span, ...]) -> None:
        """Take the sounds of the entry that is shown, and keep of what the hand had what still fits them."""
        before, self._spans = self._spans, fresh
        gave, self._gave = self._gave, None
        if [span.defn for span in fresh] != [span.defn for span in before]:
            self._gesture, self._levels = None, {}
            self.lane.drop()
        at = self._sound
        if gave is not None and gave[0] < len(fresh) and fresh[gave[0]].defn == gave[1]:
            self._sound, self._picked = gave[0], gave[2]
        elif not (at < len(fresh) and at < len(before) and _notes(fresh[at]) == _notes(before[at])):
            self._sound, self._picked = min(at, max(0, len(fresh) - 1)), frozenset()

    def fit(self) -> None:
        """Set the view to the start of the clock, at the zoom a sound is opened with.

        The sound that is shown next has the keys opened on its notes, as the first one had, and its length
        in view.
        """
        self.start, self.zoom, self._opened, self.shown = 0.0, 0, False, 0.0

    def _see(self, seen: Map) -> None:
        """Take the view of a map, brought within its limits, and have the parts show it."""
        seen = seen.within()
        self.start, self.zoom, self.top = seen.start, seen.zoom, seen.top
        self._sync()

    def _sync(self) -> None:
        """Have the parts show what the view shows now: the keys beside their rows, the view, and the velocities.

        The keyboard and the two scroll bars are laid out in the rectangle, and the sign after the last label.
        The lane is told the stretch its bars stand in, which is as wide as the grid, a bar for each note of
        the picked voice of the sound the hand is in, as it is drawn, and where the playhead is. Where nothing
        can be changed the lane takes nothing from the mouse.
        """
        seen = self.map
        left, _, wide, _ = seen.grid
        self.keys.rect, self.along.rect, self.across.rect = seen.keys, seen.along, seen.across
        self.keys.top, self.keys.key_h = seen.top, KEY_H
        keys, clock = seen.scrolls()
        self.along.show(*keys)
        self.across.show(*clock)
        laid, picked = self._now()
        items, self._lane = [], []
        if self._sound < len(laid):
            span = laid[self._sound]
            own = self._voice(span)
            for (voice, index), (x, _, w, _), _ in bars(seen, span):
                if voice == own:
                    note = span.defn["voices"][voice]["notes"][index]
                    items.append((x, w, float(note[3]) if len(note) > 3 else 1.0, (voice, index) in picked))
                    self._lane.append((voice, index))
            self.lane.colour = voice_colour(self.colour, max(own, 0))
        self.lane.span = (left, wide)
        self.lane.show(items)
        self.lane.playhead = None if self.playhead is None else seen.x(self.playhead)
        self.lane.enabled = bool(self._spans)
        self.more.rect = within(plus(seen, self._placed(), self.kind == "parts"), seen.names)
        if self.adds:
            end = " at the end" if self.kind == "parts" else ""
            self.more.help = told(f"add a {ADDED[self.kind]}{end}, a copy of {called(self.kind, self._sound)}")
        self._seen()

    def bar_at(self, pos: Pos) -> Bar | None:
        """The note whose bar is under a position, or None for none. The copy of a bar that an echo brings is none."""
        seen = self.map
        if inside(seen.grid, pos):
            # The last is drawn on top of the others.
            for _, place, shape in reversed(self._laid):
                for bar in reversed(shape.bars):
                    if inside(seen.bar(place + bar.start, bar.seconds, bar.pitch), pos):
                        return bar
        return None

    def clock(self, index: int, into: float) -> float | None:
        """The second of the view's clock that is `into` seconds into one of the shapes, counted from 0.

        None where the shape is not: before its start, past its end, and for a shape there is none of. So a
        playhead that goes by this stays in the stretch of its shape.
        """
        if not 0 <= index < len(self._laid):
            return None
        _, place, shape = self._laid[index]
        return place + into if 0 <= into <= shape.seconds else None

    def choice_at(self, pos: Pos) -> int | None:
        """The variant a position picks, counted from 0: the one whose label or whose stretch of the wave it is on.

        A stretch goes from where its variant starts to where the next one does. None anywhere else, and for a
        sound that has no variants.
        """
        seen = self.map
        if self.kind != "defs" or not (inside(seen.names, pos) or inside(seen.wave, pos)):
            return None
        return stretch([seen.x(place) for _, place, _ in self._laid], pos[0])

    def columns(self) -> list[tuple[int, np.ndarray]]:
        """The waves as they are in view: of each the x of its first column, and its columns as `soundview.columns`
        gives them.

        A wave has a column for each pixel of the box that its samples reach, and none where they do not. With
        its start in view its first column begins at its first sample, and with its end in view its last column
        ends at its last.
        """
        return self._columns(self.map, self._waves)

    def _columns(self, seen: Map, waves: Sequence[tuple[float, np.ndarray]]) -> list[tuple[int, np.ndarray]]:
        """Some waves as a map has them in view, each as (the second it starts at, its samples): as `columns`."""
        x, _, w, _ = seen.wave
        found = []
        for place, samples in waves:
            begins, ends = seen.x(place), seen.x(place + len(samples) / self._rate)
            left = max(x, begins)
            right = max(left, min(x + w, ends))
            first = 0 if left == begins else max(0, round((seen.second_at(left) - place) * self._rate))
            last = len(samples) if right == ends else max(0, round((seen.second_at(right) - place) * self._rate))
            found.append((left, soundview.columns(samples, right - left, first, last)))
        return found

    def sound_at(self, pos: Pos) -> int | None:
        """The sound of the entry whose label a position is on, as `soundedit.sound_in` counts them.

        The start and the loop of a held sound are both its note. None anywhere else, and where nothing can be
        changed.
        """
        seen = self.map
        if not self._spans or not inside(seen.names, pos):
            return None
        shape = stretch([seen.x(place) for _, place, _ in self._laid], pos[0])
        return shape if shape is None or self.kind != "held" else shape // 2

    def tab(self, which: int) -> Rect:
        """The tab of the label of one of the shapes, counted from 0, in the strip of the names."""
        return tabs(self.map, self._laid, self.kind == "parts")[which]

    def _flags(self) -> list[tuple[int, float]]:
        """Where the flags of the expected notes stand, as `mark_lines` has them: one that is dragged where its
        release would leave it."""
        marks, pull = list(self._marks), self._pull
        if pull is not None and pull.kind == MARK:
            marks[pull.which] = {**marks[pull.which], "at": pull.value}
        return mark_lines(self._laid, self.kind, marks)

    def handle_at(self, pos: Pos) -> tuple[str, int] | None:
        """The handle a position is on: what it is, and which part it is of, which variant's line or which note.

        In the ruler the flag of an expected note is taken from where the flag starts to `GRIP` pixels after
        its line, and the line of a longest time `GRIP` pixels to each side of it: a flag before a line. In the
        strip of the names a border of a held sound is taken as near, the nearer of the two and of two as near
        the loop's, and a part by the tab of its label, the one drawn last first. None anywhere else, and
        while nobody is told of a handle.
        """
        if not self._spans or self.handled is None:
            return None
        seen = self.map
        if inside(seen.ruler, pos):
            for index, second in self._flags():
                if -FLAG_W <= pos[0] - seen.x(second) <= GRIP:
                    return MARK, index
            for index, second in limit_lines(self._laid, self.kind, self._limit):
                if abs(pos[0] - seen.x(second)) <= GRIP:
                    return LIMIT, index
        elif inside(seen.names, pos) and self.kind == "held":
            far, key = min((abs(pos[0] - seen.x(place)), key)
                           for key, (_, place, _) in zip((LOOP, START), self._laid[:0:-1]))
            return (key, 0) if far <= GRIP else None
        elif self.kind == "parts":
            which = self._tab_at(seen, pos)
            return None if which is None else (PART, which)
        return None

    def _tab_at(self, seen: Map, pos: Pos) -> int | None:
        """The shape whose tab a position is on, the one drawn last first, or None for none."""
        boxes = tabs(seen, self._laid, self.kind == "parts")
        return next((which for which in reversed(range(len(boxes))) if inside(boxes[which], pos)), None)

    def _own(self, second: float) -> tuple[int, float]:
        """A second of the view's clock on the clock a longest time is counted on: which variant, and its second.

        Variants have a clock each, and a second is on that of the variant whose stretch it is in. Anything
        else has the view's clock. No second is before 0.
        """
        if self.kind != "defs":
            return 0, max(0.0, second)
        starts = [place for _, place, _ in self._laid]
        which = stretch(starts, second) or 0
        return which, max(0.0, second - starts[which])

    def _voice(self, span: Span) -> int:
        """The voice of a sound that a new note goes into: the picked one, or the last where it has fewer."""
        return min(self.voice, len(span.defn["voices"]) - 1)

    def _does(self, seen: Map, pos: Pos, mods: frozenset[str]) -> str:
        """What the presses and the wheel do at a position on the grid, as `kit.told` words it. No words where a
        press does nothing."""
        kind, which, pick = begins(seen, self._spans, pos, 1, mods, self._sound)
        if not kind:
            return ""
        span = self._spans[which]
        voices = len(span.defn["voices"])
        voice = f"voice {self._voice(span) + 1}"
        if kind == PLACE:
            tick = soundedit.snapped(seen.tick_at(pos[0], span.at), self.grid)
            said = [f"put a note at {spelt(seen.pitch_at(pos[1]))}", f"tick {tick:g}", voice * (voices > 1),
                    called(self.kind, which)]
            return told(", ".join(word for word in said if word), wheel=WHEEL_HELP)
        if kind == BOX:
            of = "" if voices == 1 or "ctrl" in mods else f" of {voice}"
            return told(f"drag a box round the notes{of} to pick", wheel=WHEEL_HELP)
        named = spelt(round(midi(span.defn["voices"][pick[0]]["notes"][pick[1]][1])))
        if kind == PICK:
            out = which == self._sound and pick in self._picked
            press = f"take {named} out of the picked notes" if out else f"pick {named} as well"
        else:
            press = f"pick {named}, a drag {'moves it' if kind == MOVE else 'makes it longer or shorter'}"
        return told(press, "rub it out", WHEEL_HELP)

    def _offers(self, seen: Map, pos: Pos) -> str:
        """What the presses and the wheel do at a position on a handle or in the ruler. No words anywhere else,
        and none for what nobody is told of."""
        hit = self.handle_at(pos)
        menus = self.asked is not None
        if hit is not None:
            kind, which = hit
            if kind == PART:
                return told(f"pick part {which + 1}, a drag moves where it starts", "its menu" * menus, WHEEL_HELP)
            if kind == MARK:
                return told(f"drag where {self._marks[which]['note']} is expected", "change it or remove it" * menus,
                            WHEEL_HELP)
            if kind == LIMIT:
                return told("drag the longest time", RULER_HELP * menus, WHEEL_HELP)
            return told(f"drag where the loop {'starts' if kind == START else 'ends'}", wheel=WHEEL_HELP)
        if self._spans and menus and inside(seen.ruler, pos):
            return told("ctrl adds a part here" * (self.adds and self.kind == "parts"), RULER_HELP, WHEEL_HELP)
        return ""

    def _pitch(self, samples: np.ndarray, first: int) -> str:
        """The pitch a wave has from one of its samples on, in words: its hertz and the name of the nearest note.

        It is as `measure.pitch_at` reads it, and where that reads none the words say so. What was read last
        is kept, so that a mouse that rests is read once.
        """
        where = (id(samples), first, self._rate)
        if self._read[0] != where:
            found = pitch_at(samples, first / self._rate, rate=self._rate)
            said = f"{found:g} Hz, {spelt(round(69 + 12 * math.log2(found / 440)))}" if found else NO_PITCH
            self._read = (where, said)
        return self._read[1]

    def _reads(self, seen: Map, x: int) -> str:
        """What the wave has at an x: its second, the level there and the pitch from there on. No words where no
        wave is."""
        for place, samples in self._waves:
            first, last = (round((seen.second_at(x + step) - place) * self._rate) for step in (0, 1))
            heard = samples[max(first, 0):max(last, first + 1, 0)]
            if len(heard):
                peak = float(np.abs(heard).max())
                level = f"{20 * math.log10(peak):.1f} dB" if peak > 0 else "silent"
                return f"{seen.second_at(x):.2f} s, {level}, {self._pitch(samples, max(first, 0))}"
        return ""

    def _tell(self, pos: Pos, mods: frozenset[str] = _NO_MODS) -> str:
        """What is at a position, in plain words: what the presses and the wheel do there, or a note that is
        shown and no more, or a moment of the wave with its level and its pitch. Else no words."""
        seen = self.map
        if self._spans and inside(seen.grid, pos):
            return self._does(seen, pos, mods)
        offered = self._offers(seen, pos)
        if offered:
            return offered
        bar = self.bar_at(pos)
        if bar is not None:
            line = f"{spelt(round(bar.pitch))}, {bar.seconds:.2f} s"
            return f"{line}, instrument {bar.inst}" if bar.inst else line
        which = self.sound_at(pos)
        if which is not None:
            menu = "its menu" * (self.asked is not None and self.kind in ADDED)
            return told(f"pick {called(self.kind, which)}", menu, WHEEL_HELP)
        which = self.choice_at(pos)
        picks = which is not None and (self.chosen is not None or bool(self._spans))
        pick = told(f"pick variant {which + 1}") if picks else ""
        if inside(seen.wave, pos):
            return "   ".join(words for words in (self._reads(seen, pos[0]), pick) if words)
        return pick

    def on(self, event: Event, ui: Ui) -> bool:
        """Note where the mouse is, take a press and carry the gesture it begins on to its release, and use the wheel.

        A press on a handle begins a pull, and one in the ruler, or with the right button on a label or a flag,
        tells the owner: see `_grab`. A press of the left button on a label, or in a variant's stretch of the
        wave, picks: it begins no gesture. The wheel goes along the keys, with shift along the time, and with
        ctrl it zooms: not while a gesture is under way or a handle is pulled. A press is used when it does
        any of these, and the owner is told of no edit of the notes before its release. A key goes to `key`,
        and one that is not the view's to `typed`.
        """
        if event.kind == "key":
            return self._keyed(event)
        self._mouse, self._mods = event.pos, event.mods
        gesture, used = self._gesture, False
        if event.kind == "down":
            # A press while another button is held begins nothing.
            used = ui.held is None and (self._grab(event) or self._choose(event) or self._press(event))
        elif event.kind == "wheel":
            used = not self.busy and self._wheel(event)
        elif self._pull is not None:
            self._haul(event.pos)
            if event.kind == "up" and event.button == 1:
                self._let_go()
        elif gesture is not None:
            self._drag(event.pos)
            if event.kind == "up" and event.button == gesture.button:
                self._release()
        self._sync()
        return used

    def _keyed(self, event: Event) -> bool:
        """Take a key of the computer: it goes to `key`, and one that is not the view's to `typed`. Was it used."""
        used = self.key(event) or (self.typed is not None and bool(self.typed(event)))
        self._sync()
        return used

    def _wheel(self, event: Event) -> bool:
        """Go along the keys by a step of the wheel, with shift along the time, with ctrl zoom. It is always used."""
        seen = self.map
        way, steps = wheel(event)
        if way == ZOOM:
            seen = seen.zoomed(seen.zoom + steps, max(event.pos[0], seen.grid[0]))
        elif way == ACROSS:
            seen = replace(seen, start=seen.start - steps * WHEEL_PX / seen.scale)
        else:
            seen = replace(seen, top=seen.top + steps * WHEEL_KEYS)
        self._see(seen)
        return True

    def _go_along(self, offset: int) -> None:
        """Take the view to where the scroll bar along the keys was moved: so many pixels down from the top key."""
        self._see(replace(self.map, top=HIGHEST - offset / KEY_H))

    def _go_across(self, offset: int) -> None:
        """Take the view to where the scroll bar along the time was moved: so many pixels after second 0."""
        seen = self.map
        self._see(replace(seen, start=offset / seen.scale))

    def hear(self, pitch: float) -> None:
        """Let a pitch be heard in the instrument of the picked voice of the sound the hand is in.

        It is what a press on a key of the keyboard does, and it changes nothing and picks nothing. Where
        nothing can be changed there is no voice to hear it in.
        """
        if self._spans:
            span = self._spans[self._sound]
            self._hear(span, self._voice(span), pitch)

    def _weigh(self, index: int, value: float | None) -> None:
        """Take the velocity the lane gives a note, from 0 to 1: it is kept until the lane has settled.

        None is for the usual velocity, which is written as none, and so is 1. The note is that of a bar the
        lane was shown.
        """
        if self._spans and self._gesture is None and self._pull is None and index < len(self._lane):
            pick = self._lane[index]
            velocity = None if value is None else round(min(max(value, 0.0), 1.0), 2)
            self._levels.pop(pick, None)
            self._levels[pick] = None if velocity in (None, 1.0) else velocity
        self._sync()

    def _weighed(self) -> None:
        """End what the lane did: the sound with its new velocities is handed on as one edit, if it is another."""
        defn = self.drawn
        levels, self._levels = self._levels, {}
        if levels and self._spans and defn != self._spans[self._sound].defn:
            self._give(self._sound, defn, self._picked, VELOCITY_DONE)
        self._sync()

    def _add(self) -> None:
        """Tell the owner that the sign after the last label was pressed."""
        if self.adds:
            self.added(None)

    def _grab(self, event: Event) -> bool:
        """Take a press that is for what belongs to the kind of the entry, and say whether it was one.

        The right button asks for the menu of a flag, of the ruler, or of the label of a variant or a part.
        The left one begins the pull of a handle, with the hand in the part whose label it is, or adds a part:
        in the ruler of an entry of parts with ctrl held.
        """
        if not self._spans or event.button not in (1, 3):
            return False
        seen, pos = self.map, event.pos
        second = seen.second_at(pos[0])
        hit = self.handle_at(pos)
        if event.button == 3:
            label = self.sound_at(pos) if self.kind == "defs" else self._tab_at(seen, pos)
            if self.asked is None or not (inside(seen.ruler, pos) or label is not None and self.kind in ADDED):
                return False
            if hit is not None and hit[0] == MARK:
                self.asked(MARK, hit[1], self._marks[hit[1]]["at"], pos)
            elif label is None:
                self.asked(RULER, *self._own(second), pos)
            else:
                self.asked(LABEL, label, second, pos)
            return True
        if hit is not None:
            kind, which = hit
            if kind == PART:
                was = self._entry["parts"][which][0]
            elif kind == MARK:
                was = self._marks[which]["at"]
            else:
                was = self._limit if kind == LIMIT else self._entry["held"][kind]
            self._gesture, self._pull = None, Pull(kind, which, seen, pos, was, self._offers(seen, pos))
            if kind == PART:
                self._into(which)
            return True
        if self.adds and self.kind == "parts" and "ctrl" in event.mods and inside(seen.ruler, pos):
            self.added(on_tick(second))
            return True
        return False

    def _haul(self, pos: Pos) -> None:
        """Take the handle that is pulled to where the mouse is: it stands at what a release there leaves it at.

        A part stands on a tick, and at no second before 0. A longest time stands on a step of `LIMIT_STEP`,
        and an expected note on a hundredth of a second, at none before 0. A border stands on a tick, where
        `soundedit.held_with` keeps the start or the loop. The wave of a part by itself is asked for when its
        label is first dragged.
        """
        pull = self._pull
        pull.pos = pos
        if not pull.dragged and abs(pos[0] - pull.start[0]) > SLACK:
            pull.dragged = True
            if pull.kind == PART and self.alone is not None:
                pull.wave = self.alone(pull.which)
        if not pull.dragged:
            return
        by = (pos[0] - pull.start[0]) / pull.seen.scale
        if pull.kind == PART:
            pull.value = on_tick(pull.was + by)
        elif pull.kind == LIMIT:
            pull.value = round(max(LIMIT_STEP, pull.was + by), 2)
        elif pull.kind == MARK:
            pull.value = round(max(0.0, pull.was + by), 2)
        else:
            ticks = max(0, math.floor(pull.was + by * TICK_HZ + 0.5))
            try:
                pull.value = soundedit.held_with(self._entry, pull.kind, ticks, self._instruments,
                                                 self._rate)["held"][pull.kind]
            except RowSynthError:
                pull.value = pull.was

    def _let_go(self) -> None:
        """End the pull: a handle that stands at another place than it did is told of, for the owner to save."""
        pull, self._pull = self._pull, None
        if pull.dragged and pull.value != pull.was and self.handled is not None:
            self.handled(pull.kind, pull.which, pull.value)

    def _into(self, which: int) -> None:
        """Have the hand in a sound of the entry. In another one than before, nothing is picked."""
        if which != self._sound:
            self._sound, self._picked = which, frozenset()

    def _choose(self, event: Event) -> bool:
        """Take a press of the left button that picks, and say whether it did.

        On the label of a variant or in its stretch of the wave, the owner is told of the variant, where
        somebody is. On the label of any sound that can be changed, and in the stretch of a variant that can,
        the hand is in that sound from then on.
        """
        if event.button != 1:
            return False
        which = self.choice_at(event.pos)
        tells = which is not None and self.chosen is not None
        if tells:
            self.chosen(which)
        sound = which if which is not None and self._spans else self.sound_at(event.pos)
        if sound is not None:
            self._into(sound)
        return tells or sound is not None

    def _press(self, event: Event) -> bool:
        """Begin the gesture that a press begins, and say whether it began one. A press on a note picks first.

        The note is picked alone unless it was picked already. With ctrl it joins the picked ones, or leaves them.
        """
        self._gesture = None
        seen, pos = self.map, event.pos
        kind, which, pick = begins(seen, self._spans, pos, event.button, event.mods, self._sound)
        if not kind:
            return False
        said = self._tell(pos, event.mods)
        self._into(which)
        if kind == PICK:
            self._picked ^= {pick}
        elif kind in (MOVE, STRETCH) and pick not in self._picked:
            self._picked = frozenset({pick})
        span = self._spans[which]
        new = (self._voice(span), soundedit.snapped(seen.tick_at(pos[0], span.at), self.grid),
               spelt(seen.pitch_at(pos[1])), self._length)
        self._gesture = Gesture(kind, event.button, which, pick, seen, span, self.grid, self._picked, new,
                                "ctrl" in event.mods, pos, said)
        return True

    def _drag(self, pos: Pos) -> None:
        """Take the gesture to where the mouse is, and tell of the note that is being moved at each new pitch."""
        gesture = self._gesture
        gesture.to(pos)
        if gesture.kind == MOVE:
            pitch = midi(gesture.grabbed()[1])
            if pitch != gesture.heard:
                gesture.heard = pitch
                self._hear(gesture.span, gesture.pick[0], pitch)

    def _release(self) -> None:
        """End the gesture: another sound than there was is handed on, and the same one is picked as it leaves it.

        The length of a note that was placed by a drag or stretched is kept for the next click. A note that was
        placed is told of, to be heard, unless the owner answers that it was not saved.
        """
        gesture, self._gesture = self._gesture, None
        defn, picked = gesture.gives()
        if defn == gesture.span.defn:
            self._picked = picked
            return
        voice, _, note, _ = gesture.new
        if gesture.kind == PLACE and gesture.dragged:
            self._length = defn["voices"][voice]["notes"][-1][2]
        elif gesture.kind == STRETCH:
            self._length = gesture.grabbed()[2]
        if self._give(gesture.which, defn, picked, gesture.label) and gesture.kind == PLACE:
            self._hear(gesture.span, voice, midi(note))

    def _give(self, which: int, defn: dict, picked: Picked, label: str) -> bool:
        """Hand an edit on to the owner, and keep what it leaves picked for when its sound is shown.

        The answer is whether the edit was saved: it was not where the owner answers false. With no owner, and
        with one that answers anything else or nothing, it is taken for saved.
        """
        self._gave = (which, defn, picked)
        return self.edited is None or self.edited(which, defn, label) is not False

    def _hear(self, span: Span, voice: int, pitch: float) -> None:
        """Tell the owner of a note of a voice of a sound to let be heard."""
        if self.sounded is not None:
            self.sounded(span.defn["voices"][voice]["inst"], pitch)

    def key(self, event: Event) -> bool:
        """Take a key, and say whether it is one of the view's.

        Delete and backspace remove the picked notes. Up and down move them a semitone, with shift an octave, and
        left and right a step of the grid. Ctrl and a picks the notes of the picked voice, and with shift those
        of every voice. Ctrl and c copies the picked notes, and ctrl and v pastes them at the tick under the
        mouse, in the sound whose room the mouse is in, or at the first tick in view when the mouse is off the
        grid. Escape drops the gesture that is under way or the handle that is pulled, or else what is picked.
        While either is under way no other key is the view's, and where nothing can be changed none is.
        """
        if not self._spans:
            return False
        pressed, mods = event.key, event.mods
        if pressed == "escape" and not mods:
            if not self.busy:
                self._picked = frozenset()
            self._drop()
            return True
        if self.busy:
            return False
        span, picked = self._spans[self._sound], self._picked
        if pressed in ("delete", "backspace") and not mods:
            self._made(soundedit.removed(span.defn, picked), frozenset(), counted(len(picked), "removed"))
        elif pressed in ("up", "down") and mods <= {"shift"}:
            semis = (12 if mods else 1) * (1 if pressed == "up" else -1)
            self._made(soundedit.moved(span.defn, picked, 0, semis), picked, counted(len(picked), "moved"))
        elif pressed in ("left", "right") and not mods:
            by_ticks = self.grid if pressed == "right" else -self.grid
            self._made(soundedit.moved(span.defn, picked, by_ticks, 0), picked, counted(len(picked), "moved"))
        elif pressed == "a" and _CTRL <= mods <= {"ctrl", "shift"}:
            own = None if "shift" in mods else self._voice(span)
            self._picked = frozenset((voice, index) for voice, held in enumerate(span.defn["voices"])
                                     for index in range(len(held["notes"])) if own in (None, voice))
        elif pressed == "c" and mods == _CTRL:
            self._clip = soundedit.copied(span.defn, picked) if picked else self._clip
        elif pressed == "v" and mods == _CTRL:
            self._paste()
        else:
            return False
        return True

    def _paste(self) -> None:
        """Put the copied notes in at the tick under the mouse, each in the voice it was copied from.

        A sound with fewer voices than the notes were copied from refuses them, and the owner is told why.
        """
        seen, which, x = self.map, self._sound, self.map.grid[0]
        if inside(seen.grid, self._mouse):
            x = self._mouse[0]
            under = span_at(self._spans, seen.second_at(x), which)
            which = which if under is None else under
        if not self._clip:
            return
        span = self._spans[which]
        try:
            defn, picked = soundedit.pasted(span.defn, self._clip,
                                            soundedit.snapped(seen.tick_at(x, span.at), self.grid))
        except RowSynthError as error:
            if self.refused is not None:
                self.refused(str(error))
            return
        self._into(which)
        self._made(defn, picked, counted(len(picked), "pasted"))

    def _made(self, defn: dict, picked: Picked, label: str) -> None:
        """Take what a key made of the sound the hand is in: another sound is handed on, and the same one is picked
        as the key leaves it."""
        if defn == self._spans[self._sound].defn:
            self._picked = picked
        else:
            self._give(self._sound, defn, picked, label)

    def _stretches(self, seen: Map) -> list[Rect]:
        """Where a line that marks a second runs down: the rectangle of the view, and the box of the wave at the
        foot."""
        return [self.rect, seen.foot] if seen.notes else [self.rect]

    def draw(self, canvas: Canvas) -> None:
        """Draw the ruler with its flags, the grid with its bars, the strip of the names, the box of the wave, the
        line of the longest time and the playhead: all of it in the rectangle and in the box of the wave.

        The parts draw themselves, after the view: they are shown here what the view shows in this frame.
        """
        self._sync()
        seen = self.map
        left, _, wide, _ = seen.grid
        with canvas.clip(self.rect):
            canvas.box(self.rect, theme.PANEL)
            self._draw_ruler(canvas, seen)
            if self._laid:
                self._draw_grid(canvas, seen)
                self._draw_bars(canvas, seen)
                self._draw_names(canvas, seen)
        self._draw_wave(canvas, seen)
        self._draw_limit(canvas, seen)
        with canvas.clip(within(seen.ruler, self.rect)):
            self._draw_flags(canvas, seen)
        if self.playhead is not None:
            for box in self._stretches(seen):
                with canvas.clip(within((left, box[1], wide, box[3]), box)):
                    canvas.vline(seen.x(self.playhead), box[1], box[3], theme.ACTIVE)

    def _draw_ruler(self, canvas: Canvas, seen: Map) -> None:
        """The number of each mark of the ruler, just after where its line is."""
        _, y, _, h = seen.ruler
        power = ruler_step(seen.scale)
        step = 10.0 ** power
        with canvas.clip(seen.ruler):
            for mark in seen.marks():
                number = (seen.x(mark * step) + 2, y, max(0, _px(step * seen.scale) - 3), h)
                canvas.text_in(f"{mark * step:.{max(0, -power)}f}", number, theme.DIM)

    def _draw_flags(self, canvas: Canvas, seen: Map) -> None:
        """The flag of each note the sound is expected to play: a line down the ruler at its second, and a small
        flag before it at the top."""
        _, y, _, h = seen.ruler
        for _, second in self._flags():
            x = seen.x(second)
            canvas.vline(x, y, h, FLAG)
            canvas.box((x - FLAG_W, y, FLAG_W, FLAG_H), FLAG)

    def _zones(self, seen: Map) -> list[tuple[int, int]]:
        """The zones of a held sound from side to side, each as (x, width): one ends where the next starts, and the
        last where its end does. Anything else has none.
        """
        if self.kind != "held":
            return []
        laid = self._placed()
        edges = [seen.x(start) for _, start, _ in laid] + [seen.x(laid[-1][1] + laid[-1][2].seconds)]
        return [(left, right - left) for left, right in zip(edges, edges[1:])][:len(ZONES)]

    def _draw_grid(self, canvas: Canvas, seen: Map) -> None:
        """The rows of the keys, tinted in the zones of a held sound, and the lines of the ruler's marks down them:
        every tenth of them is stronger."""
        x, y, w, h = seen.grid
        step = 10.0 ** ruler_step(seen.scale)
        zones = self._zones(seen)
        with canvas.clip(seen.grid):
            for pitch in seen.pitches():
                top = seen.y(pitch)
                ground = theme.BACKGROUND if black(pitch) else theme.RAISED
                canvas.box((x, top, w, KEY_H), ground)
                for zone, (left, wide) in enumerate(zones):
                    canvas.box((left, top, wide, KEY_H), tint(ground, zone))
                canvas.hline(x, top + KEY_H - 1, w, theme.PANEL)
            for mark in seen.marks():
                canvas.vline(seen.x(mark * step), y, h, theme.LINE if mark % 10 else theme.DIM)

    def _rooms(self) -> list[tuple[float, tuple, list[tuple[float, float, float, int, bool]]]]:
        """What is drawn on the grid, sound by sound: the second at which the room of each ends, the repeats of
        its echo, and its notes as (second, seconds, pitch, voice, whether it is picked).

        Where the notes can be changed they are those of the entry, as a gesture that is under way would leave
        them. Else they are those of the shapes, each shape a sound with no end to its room.
        """
        if not self._spans:
            return [(math.inf, shape.echo, [(place + bar.start, bar.seconds, bar.pitch, bar.voice, False)
                                            for bar in shape.bars]) for _, place, shape in self._laid]
        laid, picked = self._now()
        return [(span.until, span.echo,
                 [(span.at + tick / TICK_HZ, ticks / TICK_HZ, midi(note), voice,
                   which == self._sound and (voice, index) in picked)
                  for voice, held in enumerate(span.defn["voices"])
                  for index, (tick, note, ticks, *_) in enumerate(held["notes"])])
                for which, span in enumerate(laid)]

    def _draw_bars(self, canvas: Canvas, seen: Map) -> None:
        """The copies that the echoes bring, each later by its repeat's time, and over them the bars of the notes.

        Nothing of a sound is drawn past where its room ends. A picked note has a bright outline, and so has the
        box that a gesture is drawing.
        """
        left, top, wide, high = seen.grid
        rooms = self._rooms()
        with canvas.clip(seen.grid):
            for copies in (True, False):
                for until, echo, notes in rooms:
                    room = (left, top, max(0, seen.x(until) - left) if until < math.inf else wide, high)
                    with canvas.clip(room):
                        if copies:
                            self._draw_copies(canvas, seen, echo, notes)
                        else:
                            self._draw_notes(canvas, seen, notes)
            box = self._gesture.box if self._gesture is not None else None
            if box is not None:
                canvas.frame(box, theme.TEXT)

    def _draw_copies(self, canvas: Canvas, seen: Map, echo: tuple, notes: Sequence[tuple]) -> None:
        """The copies of the bars of some notes that the repeats of an echo bring."""
        for delay, strength in echo:
            for start, seconds, pitch, voice, _ in notes:
                x, y, w, h = seen.bar(start + delay, seconds, pitch)
                canvas.box((x, y + ECHO_INSET, w, h - 2 * ECHO_INSET),
                           echo_colour(voice_colour(self.colour, voice), strength))

    def _draw_notes(self, canvas: Canvas, seen: Map, notes: Sequence[tuple]) -> None:
        """The bars of some notes, each in the colour of its voice, with its name on it where that has room."""
        for start, seconds, pitch, voice, picked in notes:
            colour = voice_colour(self.colour, voice)
            rect = x, y, w, h = seen.bar(start, seconds, pitch)
            canvas.box(rect, colour)
            if picked:
                canvas.frame(rect, theme.TEXT)
            elif w > 2:
                light, dark = mix(colour, theme.TEXT, LIGHTER), mix(colour, theme.BACKGROUND, DARKER)
                canvas.hline(x, y, w, light)
                canvas.hline(x, y + h - 1, w, dark)
                canvas.vline(x + w - 1, y, h, dark)
                canvas.vline(x, y, h, light)
            label = spelt(round(pitch))
            if w >= Font.width(label) + 5:
                canvas.text_in(label, (x + 3, y, w - 5, h), theme.BACKGROUND)

    def _ink(self, index: int) -> Colour:
        """The colour of the label of a shape: lit for the one that was played, and in a held sound that of its zone."""
        if index == self.played:
            return theme.ACTIVE
        return ZONES[index] if self.kind == "held" and index < len(ZONES) else theme.TEXT

    def _draw_names(self, canvas: Canvas, seen: Map) -> None:
        """A line down the grid where each shape after the first starts, and over the grid the label of each shape.

        Where there are handles, the label of a variant and of a part is on a tab, lit for the sound the hand
        is in, with a line down the grid from every one, and each border of a held sound has a grip in the
        strip of the labels. The sign that adds a variant or a part is a part of its own, and draws itself.

        The label of a part stands where `tabs` lays its tab, which is beside the tab it would lie on. Its line
        stays at the second the part starts at: it is the edge of a tab that stands there, and runs under any
        other tab.
        """
        x, y, w, h = seen.names
        deep = h + seen.grid[3]
        laid = self._placed()
        starts = [seen.x(place) for _, place, _ in laid]
        handles = bool(self._spans) and self.handled is not None
        every = tabs(seen, laid, self.kind == "parts")
        lefts = [left for left, _, _, _ in every]
        boxes = every if handles and self.kind in ADDED else []
        grips = handles and self.kind == "held"
        with canvas.clip((x, y, w, deep)):
            for index, start in enumerate(starts):
                if index or boxes:
                    canvas.vline(start, y, deep, theme.TEXT)
            for index, (label, _, _) in enumerate(laid):
                # A label that starts at a grip stands clear of it.
                left = lefts[index] + (GRIP if grips and index else 0)
                if boxes:
                    canvas.box(boxes[index], theme.LIT if index == self._sound else theme.RAISED)
                    canvas.frame(boxes[index], theme.LINE)
                    # The lines were drawn first, so that none runs over a tab: this one is the tab's own.
                    if left == starts[index]:
                        canvas.vline(left, y, h, theme.TEXT)
                # A label has the room up to the next label there is after its own.
                room = min((start for start in lefts if start > left), default=x + w) - left - 4
                canvas.text_in(label.upper(), (left + 3, y, max(0, room), h), self._ink(index))
            if grips:
                for left in starts[1:]:
                    canvas.box((left - GRIP, y, 2 * GRIP + 1, h), theme.TEXT)

    def _draw_limit(self, canvas: Canvas, seen: Map) -> None:
        """The line of the longest time, down the view and the box of the wave in the colour of a fault, where it
        is on the clock: with a grip in the ruler where it can be dragged."""
        pull = self._pull
        limit = pull.value if pull is not None and pull.kind == LIMIT else self._limit
        left, _, wide, _ = seen.grid
        for _, second in limit_lines(self._laid, self.kind, limit):
            x = seen.x(second)
            for box in self._stretches(seen):
                with canvas.clip(within((left, box[1], wide, box[3]), box)):
                    canvas.vline(x, box[1], box[3], theme.FAULT)
            if self.handled is not None:
                with canvas.clip(within(seen.ruler, self.rect)):
                    canvas.box((x - GRIP, seen.ruler[1], 2 * GRIP + 1, seen.ruler[3]), theme.FAULT)

    def _draw_wave(self, canvas: Canvas, seen: Map) -> None:
        """The box of the waves with its name in it, the line at zero, and of each wave a column for each pixel
        with samples.

        The stretch of the box that is on the clock is dark, tinted in the zones of a held sound, and has a
        line where each variant after the first starts. The columns of a wave are made to meet, so it is one
        line at any zoom. A stale wave is dim: it has `STALE` of its colours. So have the waves while a part
        or a border is dragged, and over them the wave of that part by itself is drawn whole, from where its
        release would leave the part.
        """
        x, y, w, h = seen.wave
        box = left, top, wide, high = seen.box
        pull, moving = self._pull, self._moving()
        whole = (self.colour, theme.FAULT)
        ink, fault = ((mix(theme.BACKGROUND, colour, STALE) if self.stale or moving else colour) for colour in whole)
        with canvas.clip(box):
            canvas.box(box, theme.PANEL)
            canvas.box(seen.wave, theme.BACKGROUND)
            name = (left + theme.BAR_MARGIN, top + theme.BAR_TOP, wide - 2 * theme.BAR_MARGIN, Font.LINE)
            canvas.text_in(WAVE, within(name, box), theme.DIM, "right")
        with canvas.clip(within(seen.wave, (left + 1, top + 1, wide - 2, high - 2))):
            for zone, (start, across) in enumerate(self._zones(seen)):
                canvas.box((start, y, across, h), tint(theme.BACKGROUND, zone))
            canvas.hline(x, seen.zero, w, theme.LINE)
            if self.kind == "defs":
                for _, place, _ in self._laid[1:]:
                    canvas.vline(seen.x(place), y, h, theme.LINE)
            own = [(pull.value, pull.wave)] if moving and pull.wave is not None else []
            for waves, inks in ((self._waves, (ink, fault)), (own, whole)):
                for first, columns in self._columns(seen, waves):
                    spans = columns.tolist()
                    ends = joined([seen.column(lowest, highest) for lowest, highest in spans])
                    for column, ((lowest, highest), (upper, lower)) in enumerate(zip(spans, ends)):
                        clipped = max(highest, -lowest) >= FULL
                        canvas.vline(first + column, upper, lower - upper + 1, inks[clipped])
        canvas.frame(box, theme.LINE)
