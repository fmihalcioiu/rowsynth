"""The view of a sound: its notes as bars over its wave on one clock of seconds, its sums, and its gestures."""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from ..errors import RowSynthError
from ..note import DEFAULT_RATE, TICK_HZ
from ..notes import midi
from . import soundedit, soundview, theme
from .font import BODY_H, Font
from .layout import Rect
from .roll import (BOX, DARKER, END_W, LIGHTER, MOVE, PICK, PLACE, REMOVE, SLACK, STRETCH, black, counted, mix, name,
                   path)
from .soundedit import Clip, Pick, Picked
from .soundview import Bar, Shape
from .theme import Colour
from .ui import NOWHERE, Event, Pos, Ui, Widget, inside

if TYPE_CHECKING:
    from .canvas import Canvas

#: How high a key is, and with it its row of the grid, where the notes fit so. Where they do not it is lower.
KEY_H = 11
#: A key lower than this has no outline, and its row of the grid no line under it.
KEY_LINES_FROM = 4
#: How wide the strip of keys down the left is, which has room for the word beside the ruler, and how much of it a
#: black key takes.
KEYS_W = 48
BLACK_W = 28
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
#: How high the box of the wave is under the grid, and how far under the grid it starts.
WAVE_H = 81
GAP = 4
#: How far a wave at full scale keeps from the outline of its box.
WAVE_AIR = 2
#: The share of full scale from which a sample counts as clipped, as `measure` counts them.
FULL = 0.999
#: What one step of the zoom makes the clock wider by, and how many steps the zoom goes.
ZOOM_STEP = 1.25
ZOOM_MOST = 16
#: How many pixels one step of the wheel scrolls the time.
WHEEL_PX = 36
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
#: How many keys are kept free over the highest note and under the lowest of a sound that the hand can change,
#: where the grid has the room: a note can be placed or dragged that much further.
SPARE = 3
#: How much of its colour a wave keeps while it is stale: its sound was changed, and the render of that has not come.
STALE = 0.4

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


def tint(ground: Colour, zone: int) -> Colour:
    """The colour a ground has in a zone of a held sound, counted from 0: `TINT` of the way to the zone's colour."""
    return mix(ground, ZONES[zone], TINT)


def keys(height: int, low: int, high: int) -> tuple[int, int]:
    """The pitch of the key at the upper edge of a grid of a height, and how high a key is, for notes from low to high.

    A key is `KEY_H` high where the notes fit so, with as many keys over them as under them. Where they do not, it
    is as high as lets them all in, and one pixel at the least.
    """
    span = high - low + 1
    key_h = max(1, min(KEY_H, height // span))
    return high + max(0, height // key_h - span) // 2, key_h


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

    rect is the view's rectangle. seconds is how long the clock is, start the second at the left edge of the grid,
    and zoom how many steps the clock is zoomed in: at 0 the whole of it is as wide as the grid. top is the pitch
    of the key at the upper edge of the grid, and key_h how high a key is. labels is whether the shapes are named,
    in a strip of their own over the grid. notes is whether there is a grid: with none the wave has its room.
    """

    rect: Rect
    seconds: float = 1.0
    start: float = 0.0
    zoom: int = 0
    top: int = 72
    key_h: int = KEY_H
    labels: bool = False
    notes: bool = True

    @property
    def ruler(self) -> Rect:
        """The strip along the top in which the seconds are numbered."""
        x, y, w, _ = self.rect
        return x + KEYS_W, y, max(0, w - KEYS_W), RULER_H

    @property
    def names(self) -> Rect:
        """The strip under the ruler in which the shapes are named. It is nothing high where they have no names."""
        x, y, w, h = self.ruler
        return x, y + h, w, LABELS_H if self.labels else 0

    @property
    def grid(self) -> Rect:
        """Where the bars are: what the keys, the strips and the wave leave of the rectangle."""
        x, y, w, h = self.names
        foot = self.rect[1] + self.rect[3]
        return x, y + h, w, max(0, foot - y - h - GAP - WAVE_H) if self.notes else 0

    @property
    def keys(self) -> Rect:
        """The strip of keys, down the left of the grid."""
        _, y, _, h = self.grid
        return self.rect[0], y, KEYS_W, h

    @property
    def wave(self) -> Rect:
        """The box of the wave, under the grid and as wide. With no grid it is all there is under the ruler."""
        x, y, w, h = self.grid
        if self.notes:
            return x, y + h + GAP, w, WAVE_H
        return x, y, w, max(0, self.rect[1] + self.rect[3] - y)

    @property
    def scale(self) -> float:
        """How many pixels wide a second is."""
        return ZOOM_STEP ** self.zoom * max(1, self.grid[2]) / (self.seconds if self.seconds > 0 else 1.0)

    @property
    def zero(self) -> int:
        """The y of the line of the wave at which a sample is 0."""
        return self.wave[1] + self.wave[3] // 2

    def x(self, second: float) -> int:
        """The x of a second of the clock."""
        return self.grid[0] + _px((second - self.start) * self.scale)

    def second_at(self, x: int) -> float:
        """The second of the clock at the left edge of the pixels at an x."""
        return self.start + (x - self.grid[0]) / self.scale

    def tick_at(self, x: int, at: float = 0.0) -> float:
        """The tick at an x on the clock of a sound that starts at second `at` of the view's: sixty to a second.

        Its whole part is the tick that the left edge of the pixels at x is in. No x is before tick 0.
        """
        return max(0.0, (self.second_at(x) - at) * TICK_HZ)

    def y(self, pitch: int) -> int:
        """The y of the upper edge of the key of a pitch, and of its row of the grid."""
        return self.grid[1] + (self.top - pitch) * self.key_h

    def pitch_at(self, y: int) -> int:
        """The pitch whose key, and whose row of the grid, a y is on."""
        return self.top - (y - self.grid[1]) // self.key_h

    def bar(self, start: float, seconds: float, pitch: float) -> Rect:
        """The rectangle of the bar of a note that starts at a second and is held for some: in the row of its key.

        It starts a pixel after its second, so that two notes in a row are two bars, and is a pixel wide at the
        least. The lowest line of pixels of its key is left free, where the key is more than one high.
        """
        left = self.x(start)
        return left + 1, self.y(round(pitch)), max(1, self.x(start + seconds) - left - 1), max(1, self.key_h - 1)

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
        """This map at another zoom, no less than 0 and no more than `ZOOM_MOST`. The second at `x` stays at that x."""
        fresh = replace(self, zoom=min(max(zoom, 0), ZOOM_MOST))
        return replace(fresh, start=self.second_at(x) - (x - fresh.grid[0]) / fresh.scale)

    def within(self) -> Map:
        """This map, moved no further than the clock goes: it starts no sooner than second 0, and ends no later."""
        last = max(0.0, self.seconds) * (1 - ZOOM_STEP ** -self.zoom)
        return replace(self, start=min(max(self.start, 0.0), last))


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


class SoundView(Widget):
    """The view of a sound: its notes as bars on a grid of seconds and keys, and under them its wave on the same clock.

    `playhead` is the second of its clock that is playing, or None: the owner sets it each frame, and `clock`
    gives the second for a place in one of the shapes. `colour` is that of the kind of thing shown, `kind` what
    its entry is, and `seconds` how long its clock is. `start` and `zoom` are the view, as a `Map` has them.
    `stale` is whether the wave is that of the sound as it was before a change: it is drawn dim.

    Shown shapes alone, it shows and changes nothing, and `help` says what the mouse is on: a note, a moment of
    the wave, the label of a variant. Shown the entry of the shapes as well, the mouse and the keys place, move,
    stretch and remove the notes of its sounds on a grid of ticks, and it saves nothing. When a gesture ends
    and the sound it was in is another than it was, `edited` is called with which sound of the entry that is, as
    `soundedit.sound_in` counts them, its new definition, and a few words for the list of what can be undone.
    The view goes on showing what `show` gave it last: its owner saves, and shows it the result. An owner that
    answers false has not saved: any other answer, and none, is taken for saved. `sounded` is called with the
    instrument of a voice and a pitch for the owner to let be heard: when a note is placed and saved, and each
    time the note that the mouse moves comes to another pitch. `refused` is called with the words of a change
    that the rules of `soundedit` refuse. `help` then says what a press where the mouse is would do.

    `sound` is the sound of the entry that the hand is in: the one it pressed in last, on the grid or on the
    label of a sound. `voice` is the voice a new note goes into, and the owner sets it. `grid` is the ticks in a
    step of the grid: a placed note begins on a step, and notes are moved and stretched by whole steps.
    `picked` is the picked notes of that sound, each as (voice, index), and `drawn` the sound as a gesture that
    is under way would leave it: `busy` is whether one is under way. A box picks the notes of the voice that is
    `voice`, and so do ctrl and a: with ctrl held as well, the box picks those of every voice, and so do ctrl,
    shift and a. A key that the Ui hands the view, which it does while the view has the focus, goes to `key`.

    A variant can be chosen: `chosen` is called with its number, from 0, as the left button goes down on its
    label or in its stretch of the wave. `played` is the shape whose label is lit, or None: the owner sets it to
    the variant that was played last.
    """

    def __init__(self, rect: Rect, help: str = "") -> None:
        super().__init__(rect, help)
        self.colour: Colour = theme.SOUND
        self.kind = KINDS[0]
        self.playhead: float | None = None
        self.played: int | None = None
        self.chosen: Callable[[int], object] | None = None
        self.edited: Callable[[int, dict, str], object] | None = None
        self.sounded: Callable[[str | dict, float], object] | None = None
        self.refused: Callable[[str], object] | None = None
        self.seconds = 0.0
        self.start, self.zoom = 0.0, 0
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

    @property
    def help(self) -> str:
        """What the mouse is on or what a press there would do, in plain words, or the line the view was made with.

        While a gesture is under way it is what the press did that began it.
        """
        told = self._gesture.help if self._gesture is not None else self._tell(self._mouse, self._mods)
        return told or self._help

    @help.setter
    def help(self, line: str) -> None:
        self._help = line

    @property
    def map(self) -> Map:
        """The sums of the view as it is now: the keys fitted to the notes, and the clock kept within its limits.

        Where the notes can be changed, `SPARE` keys more are fitted over them and under them.
        """
        seen = Map(self.rect, self.seconds, self.start, self.zoom,
                   labels=any(label for label, _, _ in self._laid), notes=bool(self._laid))
        pitches = [round(bar.pitch) for _, _, shape in self._laid for bar in shape.bars] or [MIDDLE_C]
        spare = SPARE if self._spans else 0
        top, key_h = keys(seen.grid[3], min(pitches) - spare, max(pitches) + spare)
        return replace(seen, top=top, key_h=key_h).within()

    @property
    def sound(self) -> int:
        """Which sound of the entry the hand is in, as `soundedit.sound_in` counts them.

        The owner sets it to 0 when it is about to show another entry: then nothing is picked, and a gesture
        that is under way is dropped.
        """
        return self._sound

    @sound.setter
    def sound(self, which: int) -> None:
        self._sound, self._picked, self._gesture = which, frozenset(), None

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
        """Whether a gesture is under way: from the press that began it to its release, or until it is dropped."""
        return self._gesture is not None

    def _now(self) -> tuple[tuple[Span, ...], Picked]:
        """The sounds to draw and the picked notes: with a gesture under way what it gives, and else what there is."""
        gesture = self._gesture
        if gesture is None:
            return self._spans, self._picked
        defn, picked = gesture.gives()
        return (*self._spans[:gesture.which], replace(gesture.span, defn=defn),
                *self._spans[gesture.which + 1:]), picked

    def show(self, shapes: Sequence[tuple[str, float, Shape]], samples: Sequence[np.ndarray], rate: int,
             colour: Colour, kind: str = "def", entry: dict | None = None, stale: bool = False) -> None:
        """Show some shapes, as `soundview.shapes` gives them, over the samples of their render at a rate.

        `kind` is what the entry of the shapes is, as `bank.entry_kind` names it. The shapes of "defs" are
        variants, laid one after the other, and those of "held" are its zones, each tinted in its colour.

        The samples are rows, as `render_entry` gives its buffers. Several rows are one for each shape: each is
        the wave of its shape, and starts where that starts. One row alone starts where the clock does: it is
        the wave of one sound or of parts added up, and with no shapes a wave alone, with no grid over it. No
        rows are an empty wave. `stale` is for rows of the sound as it was before a change. `colour` is that of
        the kind of thing shown. The view stays where it was, as far as the new clock lets it: `fit` shows the
        whole again.

        `entry` is the entry of a project's sounds that the shapes were made of: with it the notes can be
        changed by hand, and with None they are shown and no more. What is picked stays picked while the sound
        it is in has the notes it had. The sound that the last edit gave is shown with what that edit left
        picked. With any other notes nothing is picked, and a gesture that is under way is dropped when any
        sound of the entry is another than it was.

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
        self._take(fresh)
        self.seconds = max([0.0] + [place + shape.seconds for _, place, shape in laid]
                           + [place + len(row) / rate for place, row in waves])
        self._see(self.map)

    def _take(self, fresh: tuple[Span, ...]) -> None:
        """Take the sounds of the entry that is shown, and keep of what the hand had what still fits them."""
        before, self._spans = self._spans, fresh
        gave, self._gave = self._gave, None
        if [span.defn for span in fresh] != [span.defn for span in before]:
            self._gesture = None
        at = self._sound
        if gave is not None and gave[0] < len(fresh) and fresh[gave[0]].defn == gave[1]:
            self._sound, self._picked = gave[0], gave[2]
        elif not (at < len(fresh) and at < len(before) and _notes(fresh[at]) == _notes(before[at])):
            self._sound, self._picked = min(at, max(0, len(fresh) - 1)), frozenset()

    def fit(self) -> None:
        """Set the view to the whole of the clock, which then is as wide as the grid."""
        self.start, self.zoom = 0.0, 0

    def _see(self, seen: Map) -> None:
        """Take the view of a map, brought within its limits."""
        seen = seen.within()
        self.start, self.zoom = seen.start, seen.zoom

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
        """The variant a position chooses, counted from 0: the one whose label or whose stretch of the wave it is on.

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
        seen = self.map
        x, _, w, _ = seen.wave
        found = []
        for place, samples in self._waves:
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

    def _voice(self, span: Span) -> int:
        """The voice of a sound that a new note goes into: the picked one, or the last where it has fewer."""
        return min(self.voice, len(span.defn["voices"]) - 1)

    def _does(self, seen: Map, pos: Pos, mods: frozenset[str], button: int) -> str:
        """What a press at a position on the grid would do to the notes, in plain words. No words for nothing."""
        kind, which, pick = begins(seen, self._spans, pos, button, mods, self._sound)
        if not kind:
            return ""
        span = self._spans[which]
        voices = len(span.defn["voices"])
        voice = f"voice {self._voice(span) + 1}"
        if kind == PLACE:
            tick = soundedit.snapped(seen.tick_at(pos[0], span.at), self.grid)
            said = [f"place a note at {spelt(seen.pitch_at(pos[1]))}", f"tick {tick:g}", voice * (voices > 1),
                    called(self.kind, which)]
            return ", ".join(word for word in said if word)
        if kind == BOX:
            return f"pick the notes{'' if voices == 1 or 'ctrl' in mods else ' of ' + voice} that a box touches"
        if pick is None:
            return ""
        named = spelt(round(midi(span.defn["voices"][pick[0]]["notes"][pick[1]][1])))
        if kind == PICK:
            out = which == self._sound and pick in self._picked
            return f"leave {named} out of the picked notes" if out else f"pick {named} as well"
        return f"{kind} {named}"

    def _tell(self, pos: Pos, mods: frozenset[str] = _NO_MODS, button: int = 1) -> str:
        """What is at a position, in plain words: a note, a label there is to press, or a moment of the wave and
        its level. Where the notes can be changed it is what a press on the grid would do. Else no words."""
        seen = self.map
        if self._spans and inside(seen.grid, pos):
            return self._does(seen, pos, mods, button)
        bar = self.bar_at(pos)
        if bar is not None:
            line = f"{spelt(round(bar.pitch))}, {bar.seconds:.2f} s"
            return f"{line}, instrument {bar.inst}" if bar.inst else line
        which = self.choice_at(pos)
        if which is not None and self.chosen is not None and inside(seen.names, pos):
            return f"click to hear variant {which + 1}"
        which = self.sound_at(pos)
        if which is not None:
            return f"click to work on {called(self.kind, which)}"
        if not inside(seen.wave, pos):
            return ""
        for place, samples in self._waves:
            first, last = (round((seen.second_at(pos[0] + step) - place) * self._rate) for step in (0, 1))
            heard = samples[max(first, 0):max(last, first + 1, 0)]
            if len(heard):
                peak = float(np.abs(heard).max())
                level = f"{20 * math.log10(peak):.1f} dB" if peak > 0 else "silent"
                return f"{seen.second_at(pos[0]):.2f} s, {level}"
        return ""

    def on(self, event: Event, ui: Ui) -> bool:
        """Note where the mouse is, take a press and carry the gesture it begins on to its release, and use the wheel.

        A press of the left button on a label, or in a variant's stretch of the wave, chooses: it begins no
        gesture. With ctrl the wheel zooms, and alone it scrolls a clock zoomed in: not while a gesture is under
        way. A press is used when it chooses or begins a gesture, and the owner is told of no edit before its
        release. A key goes to `key`.
        """
        if event.kind == "key":
            return self.key(event)
        self._mouse, self._mods = event.pos, event.mods
        if event.kind == "down":
            # A press while another button is held begins nothing.
            return ui.held is None and (self._choose(event) or self._press(event))
        gesture = self._gesture
        if event.kind == "wheel":
            return gesture is None and self._wheel(event)
        if gesture is not None:
            self._drag(event.pos)
            if event.kind == "up" and event.button == gesture.button:
                self._release()
        return False

    def _wheel(self, event: Event) -> bool:
        """Zoom by a step of the wheel with ctrl, and scroll a clock zoomed in by one alone. Whether it was used."""
        seen = self.map
        if "ctrl" in event.mods:
            seen = seen.zoomed(seen.zoom + event.wheel, max(event.pos[0], seen.grid[0]))
        elif seen.zoom:
            seen = replace(seen, start=seen.start - event.wheel * WHEEL_PX / seen.scale)
        else:
            return False
        self._see(seen)
        return True

    def _into(self, which: int) -> None:
        """Have the hand in a sound of the entry. In another one than before, nothing is picked."""
        if which != self._sound:
            self._sound, self._picked = which, frozenset()

    def _choose(self, event: Event) -> bool:
        """Take a press of the left button that chooses, and say whether it did.

        On the label of a variant or in its stretch of the wave, the owner is told of the variant. On the label
        of any sound that can be changed, the hand is in that sound from then on.
        """
        if event.button != 1:
            return False
        which = self.choice_at(event.pos) if self.chosen is not None else None
        if which is not None:
            self.chosen(which)
        sound = self.sound_at(event.pos) if which is None or not self._spans else which
        if sound is not None:
            self._into(sound)
        return which is not None or sound is not None

    def _press(self, event: Event) -> bool:
        """Begin the gesture that a press begins, and say whether it began one. A press on a note picks first.

        The note is picked alone unless it was picked already. With ctrl it joins the picked ones, or leaves them.
        """
        self._gesture = None
        seen, pos = self.map, event.pos
        kind, which, pick = begins(seen, self._spans, pos, event.button, event.mods, self._sound)
        if not kind:
            return False
        told = self._tell(pos, event.mods, event.button)
        self._into(which)
        if kind == PICK:
            self._picked ^= {pick}
        elif kind in (MOVE, STRETCH) and pick not in self._picked:
            self._picked = frozenset({pick})
        span = self._spans[which]
        new = (self._voice(span), soundedit.snapped(seen.tick_at(pos[0], span.at), self.grid),
               spelt(seen.pitch_at(pos[1])), self._length)
        self._gesture = Gesture(kind, event.button, which, pick, seen, span, self.grid, self._picked, new,
                                "ctrl" in event.mods, pos, told)
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
        grid. Escape drops the gesture that is under way, or else what is picked. While a gesture is under way
        no other key is the view's, and where nothing can be changed none is.
        """
        if not self._spans:
            return False
        pressed, mods = event.key, event.mods
        if pressed == "escape" and not mods:
            if self._gesture is None:
                self._picked = frozenset()
            self._gesture = None
            return True
        if self._gesture is not None:
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

    def draw(self, canvas: Canvas) -> None:
        """Draw the ruler, the keys, the grid with its bars, the wave and the playhead: all of it in the rectangle."""
        seen = self.map
        with canvas.clip(self.rect):
            canvas.box(self.rect, theme.PANEL)
            self._draw_ruler(canvas, seen)
            if self._laid:
                self._draw_keys(canvas, seen)
                self._draw_grid(canvas, seen)
                self._draw_bars(canvas, seen)
                self._draw_names(canvas, seen)
            self._draw_wave(canvas, seen)
            if self.playhead is not None:
                _, y, _, h = self.rect
                with canvas.clip((seen.grid[0], y, seen.grid[2], h)):
                    canvas.vline(seen.x(self.playhead), y, h, theme.ACTIVE)

    def _draw_ruler(self, canvas: Canvas, seen: Map) -> None:
        """The word for what the ruler counts, and the number of each of its marks, just after where its line is."""
        _, y, _, h = seen.ruler
        canvas.text_in("SECONDS", (self.rect[0], y, KEYS_W - GAP, h), theme.DIM, "right")
        power = ruler_step(seen.scale)
        step = 10.0 ** power
        with canvas.clip(seen.ruler):
            for mark in seen.marks():
                number = (seen.x(mark * step) + 2, y, max(0, _px(step * seen.scale) - 3), h)
                canvas.text_in(f"{mark * step:.{max(0, -power)}f}", number, theme.DIM)

    def _draw_keys(self, canvas: Canvas, seen: Map) -> None:
        """The keys: each a light block, a black one dark for most of its length, and each c named if it has room."""
        x, _, w, _ = seen.keys
        with canvas.clip(seen.keys):
            for pitch in seen.pitches():
                key = (x, seen.y(pitch), w - 1, seen.key_h)
                canvas.box(key, theme.TEXT)
                if seen.key_h >= KEY_LINES_FROM:
                    canvas.frame(key, theme.BACKGROUND)
                if black(pitch):
                    canvas.box((x, key[1], BLACK_W, seen.key_h), theme.BACKGROUND)
                elif pitch % 12 == 0 and seen.key_h >= BODY_H + 2:
                    canvas.text_in(name(pitch), (x, key[1], w - 5, seen.key_h), theme.BACKGROUND, "right")

    def _zones(self, seen: Map) -> list[tuple[int, int]]:
        """The zones of a held sound from side to side, each as (x, width): one ends where the next starts, and the
        last where the clock does. Anything else has none.
        """
        if self.kind != "held":
            return []
        edges = [seen.x(start) for _, start, _ in self._laid] + [seen.x(self.seconds)]
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
                canvas.box((x, top, w, seen.key_h), ground)
                for zone, (left, wide) in enumerate(zones):
                    canvas.box((left, top, wide, seen.key_h), tint(ground, zone))
                if seen.key_h >= KEY_LINES_FROM:
                    canvas.hline(x, top + seen.key_h - 1, w, theme.PANEL)
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
                inset = ECHO_INSET if h > 2 * ECHO_INSET else 0
                canvas.box((x, y + inset, w, h - 2 * inset), echo_colour(voice_colour(self.colour, voice), strength))

    def _draw_notes(self, canvas: Canvas, seen: Map, notes: Sequence[tuple]) -> None:
        """The bars of some notes, each in the colour of its voice, with its name on it where that has room."""
        for start, seconds, pitch, voice, picked in notes:
            colour = voice_colour(self.colour, voice)
            rect = x, y, w, h = seen.bar(start, seconds, pitch)
            canvas.box(rect, colour)
            if picked:
                canvas.frame(rect, theme.TEXT)
            elif w > 2 and h > 2:
                light, dark = mix(colour, theme.TEXT, LIGHTER), mix(colour, theme.BACKGROUND, DARKER)
                canvas.hline(x, y, w, light)
                canvas.hline(x, y + h - 1, w, dark)
                canvas.vline(x + w - 1, y, h, dark)
                canvas.vline(x, y, h, light)
            label = spelt(round(pitch))
            if w >= Font.width(label) + 5 and h >= BODY_H + 2:
                canvas.text_in(label, (x + 3, y, w - 5, h), theme.BACKGROUND)

    def _ink(self, index: int) -> Colour:
        """The colour of the label of a shape: lit for the one that was played, and in a held sound that of its zone."""
        if index == self.played:
            return theme.ACTIVE
        return ZONES[index] if self.kind == "held" and index < len(ZONES) else theme.TEXT

    def _draw_names(self, canvas: Canvas, seen: Map) -> None:
        """A line down the grid where each shape after the first starts, and over the grid the label of each shape."""
        x, y, w, h = seen.names
        deep = h + seen.grid[3]
        starts = [seen.x(place) for _, place, _ in self._laid]
        with canvas.clip((x, y, w, deep)):
            for index, (label, _, _) in enumerate(self._laid):
                left = starts[index]
                if index:
                    canvas.vline(left, y, deep, theme.TEXT)
                # A label has the room up to the next start there is after its own.
                room = min((start for start in starts if start > left), default=x + w) - left - 4
                canvas.text_in(label.upper(), (left + 3, y, max(0, room), h), self._ink(index))

    def _draw_wave(self, canvas: Canvas, seen: Map) -> None:
        """The box of the waves with its name beside it, the line at zero, and of each wave a column for each pixel
        with samples.

        The box is tinted in the zones of a held sound, and has a line where each variant after the first starts.
        The columns of a wave are made to meet, so it is one line at any zoom. A stale wave is dim: it has
        `STALE` of its colours.
        """
        x, y, w, h = seen.wave
        ink, fault = ((mix(theme.BACKGROUND, colour, STALE) if self.stale else colour)
                      for colour in (self.colour, theme.FAULT))
        canvas.text_in("WAVE", (self.rect[0], y, KEYS_W - GAP, h), theme.DIM, "right")
        canvas.box(seen.wave, theme.BACKGROUND)
        with canvas.clip((x + 1, y + 1, w - 2, h - 2)):
            for zone, (left, wide) in enumerate(self._zones(seen)):
                canvas.box((left, y, wide, h), tint(theme.BACKGROUND, zone))
            canvas.hline(x, seen.zero, w, theme.LINE)
            if self.kind == "defs":
                for _, place, _ in self._laid[1:]:
                    canvas.vline(seen.x(place), y, h, theme.LINE)
            for left, columns in self.columns():
                spans = columns.tolist()
                ends = joined([seen.column(lowest, highest) for lowest, highest in spans])
                for column, ((lowest, highest), (upper, lower)) in enumerate(zip(spans, ends)):
                    clipped = max(highest, -lowest) >= FULL
                    canvas.vline(left + column, upper, lower - upper + 1, fault if clipped else ink)
        canvas.frame(seen.wave, theme.LINE)
