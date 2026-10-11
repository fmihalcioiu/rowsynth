"""The view of a sound: its notes as bars over its wave on one clock of seconds, and the sums that place them."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from ..note import DEFAULT_RATE
from . import soundview, theme
from .font import BODY_H, Font
from .layout import Rect
from .roll import DARKER, LIGHTER, black, mix, name
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


def _px(value: float) -> int:
    """The whole pixel nearest to a place. A half goes up."""
    return math.floor(value + 0.5)


def places(shapes: Sequence[tuple[str, float, Shape]]) -> list[float]:
    """The second on the view's clock at which the clock of each shape starts.

    That is the second the shape is at. But several shapes that are all at 0, as variants are, are laid one after
    the other, with `VARIANT_GAP` of the longest of them left free between two.
    """
    if len(shapes) < 2 or any(at for _, at, _ in shapes):
        return [float(at) for _, at, _ in shapes]
    gap = VARIANT_GAP * max(shape.seconds for _, _, shape in shapes)
    found, at = [], 0.0
    for _, _, shape in shapes:
        found.append(at)
        at += shape.seconds + gap
    return found


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


class SoundView(Widget):
    """The view of a sound: its notes as bars on a grid of seconds and keys, and under them its wave on the same clock.

    It shows, and changes nothing. `playhead` is the second that is playing, or None: the owner sets it each frame.
    `colour` is that of the kind of thing shown, and `seconds` how long its clock is. `start` and `zoom` are the
    view, as a `Map` has them. `help`, while the mouse is on a bar or on the wave, says what is there; anywhere
    else it is the line the view was made with.
    """

    def __init__(self, rect: Rect, help: str = "") -> None:
        super().__init__(rect, help)
        self.colour: Colour = theme.SOUND
        self.playhead: float | None = None
        self.seconds = 0.0
        self.start, self.zoom = 0.0, 0
        # The shapes that are shown, each as (label, the second of the view's clock its own starts at, shape).
        self._laid: tuple[tuple[str, float, Shape], ...] = ()
        self._samples = np.zeros(0, dtype=np.float32)
        self._rate = DEFAULT_RATE
        self._mouse: Pos = NOWHERE

    @property
    def help(self) -> str:
        """What the mouse is on, in plain words, or the line the view was made with."""
        return self._tell(self._mouse) or self._help

    @help.setter
    def help(self, line: str) -> None:
        self._help = line

    @property
    def map(self) -> Map:
        """The sums of the view as it is now: the keys fitted to the notes, and the clock kept within its limits."""
        seen = Map(self.rect, self.seconds, self.start, self.zoom,
                   labels=any(label for label, _, _ in self._laid), notes=bool(self._laid))
        pitches = [round(bar.pitch) for _, _, shape in self._laid for bar in shape.bars] or [MIDDLE_C]
        top, key_h = keys(seen.grid[3], min(pitches), max(pitches))
        return replace(seen, top=top, key_h=key_h).within()

    def show(self, shapes: Sequence[tuple[str, float, Shape]], samples: np.ndarray, rate: int,
             colour: Colour) -> None:
        """Show some shapes, as `soundview.shapes` gives them, over the samples of their render at a rate.

        The samples are one row, for a sound in two sides their mean, and start where the clock does. No shapes
        are a wave alone, with no grid over it, and no samples are an empty wave. `colour` is that of the kind of
        thing shown. The view stays where it was, as far as the new clock lets it: `fit` shows the whole again.
        What is no wave and what is no rate are refused, and then nothing changes.
        """
        soundview.columns(samples, 0)
        soundview.ticks_to_seconds(0, rate)
        laid = tuple((label, place, shape) for (label, _, shape), place in zip(shapes, places(shapes)))
        self._laid, self._samples, self._rate, self.colour = laid, samples, rate, colour
        self.seconds = max([place + shape.seconds for _, place, shape in laid] + [len(samples) / rate])
        self._see(self.map)

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

    def columns(self) -> tuple[int, np.ndarray]:
        """The wave as it is in view: the x of its first column, and the columns as `soundview.columns` gives them.

        There is a column for each pixel of the box that the samples reach, and none where they do not.
        """
        seen = self.map
        x, _, w, _ = seen.wave
        left = max(x, seen.x(0.0))
        right = max(left, min(x + w, seen.x(len(self._samples) / self._rate)))
        first, last = (max(0, round(seen.second_at(edge) * self._rate)) for edge in (left, right))
        return left, soundview.columns(self._samples, right - left, first, last)

    def _tell(self, pos: Pos) -> str:
        """What is at a position, in plain words: a note, or a moment of the wave and its level. Else no words."""
        bar = self.bar_at(pos)
        if bar is not None:
            line = f"{name(round(bar.pitch))}, {bar.seconds:.2f} s"
            return f"{line}, instrument {bar.inst}" if bar.inst else line
        seen = self.map
        if not inside(seen.wave, pos):
            return ""
        first, last = (round(seen.second_at(pos[0] + step) * self._rate) for step in (0, 1))
        heard = self._samples[first:max(last, first + 1)]
        if not len(heard):
            return ""
        peak = float(np.abs(heard).max())
        level = f"{20 * math.log10(peak):.1f} dB" if peak > 0 else "silent"
        return f"{seen.second_at(pos[0]):.2f} s, {level}"

    def on(self, event: Event, ui: Ui) -> bool:
        """Note where the mouse is, and use the wheel: with ctrl it zooms, and alone it scrolls a clock zoomed in."""
        if event.kind == "key":
            return False
        self._mouse = event.pos
        if event.kind != "wheel":
            return False
        seen = self.map
        if "ctrl" in event.mods:
            seen = seen.zoomed(seen.zoom + event.wheel, max(event.pos[0], seen.grid[0]))
        elif seen.zoom:
            seen = replace(seen, start=seen.start - event.wheel * WHEEL_PX / seen.scale)
        else:
            return False
        self._see(seen)
        return True

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

    def _draw_grid(self, canvas: Canvas, seen: Map) -> None:
        """The rows of the keys, and the lines of the ruler's marks down them: every tenth of them is stronger."""
        x, y, w, h = seen.grid
        step = 10.0 ** ruler_step(seen.scale)
        with canvas.clip(seen.grid):
            for pitch in seen.pitches():
                top = seen.y(pitch)
                canvas.box((x, top, w, seen.key_h), theme.BACKGROUND if black(pitch) else theme.RAISED)
                if seen.key_h >= KEY_LINES_FROM:
                    canvas.hline(x, top + seen.key_h - 1, w, theme.PANEL)
            for mark in seen.marks():
                canvas.vline(seen.x(mark * step), y, h, theme.LINE if mark % 10 else theme.DIM)

    def _draw_bars(self, canvas: Canvas, seen: Map) -> None:
        """The copies that the echoes bring, each later by its repeat's time, and over them the bars of the notes."""
        with canvas.clip(seen.grid):
            for _, place, shape in self._laid:
                for delay, strength in shape.echo:
                    for bar in shape.bars:
                        x, y, w, h = seen.bar(place + bar.start + delay, bar.seconds, bar.pitch)
                        inset = ECHO_INSET if h > 2 * ECHO_INSET else 0
                        canvas.box((x, y + inset, w, h - 2 * inset),
                                   echo_colour(voice_colour(self.colour, bar.voice), strength))
            for _, place, shape in self._laid:
                for bar in shape.bars:
                    colour = voice_colour(self.colour, bar.voice)
                    rect = x, y, w, h = seen.bar(place + bar.start, bar.seconds, bar.pitch)
                    canvas.box(rect, colour)
                    if w > 2 and h > 2:
                        light, dark = mix(colour, theme.TEXT, LIGHTER), mix(colour, theme.BACKGROUND, DARKER)
                        canvas.hline(x, y, w, light)
                        canvas.hline(x, y + h - 1, w, dark)
                        canvas.vline(x + w - 1, y, h, dark)
                        canvas.vline(x, y, h, light)
                    label = name(round(bar.pitch))
                    if w >= Font.width(label) + 5 and h >= BODY_H + 2:
                        canvas.text_in(label, (x + 3, y, w - 5, h), theme.BACKGROUND)

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
                canvas.text_in(label.upper(), (left + 3, y, max(0, room), h), theme.TEXT)

    def _draw_wave(self, canvas: Canvas, seen: Map) -> None:
        """The box of the wave with its name beside it, the line at zero, and a column for each pixel with samples.

        The columns are made to meet, so the wave is one line at any zoom.
        """
        x, y, w, h = seen.wave
        canvas.text_in("WAVE", (self.rect[0], y, KEYS_W - GAP, h), theme.DIM, "right")
        canvas.box(seen.wave, theme.BACKGROUND)
        left, columns = self.columns()
        with canvas.clip((x + 1, y + 1, w - 2, h - 2)):
            canvas.hline(x, seen.zero, w, theme.LINE)
            spans = columns.tolist()
            ends = joined([seen.column(lowest, highest) for lowest, highest in spans])
            for column, ((lowest, highest), (upper, lower)) in enumerate(zip(spans, ends)):
                clipped = max(highest, -lowest) >= FULL
                canvas.vline(left + column, upper, lower - upper + 1, theme.FAULT if clipped else self.colour)
        canvas.frame(seen.wave, theme.LINE)
