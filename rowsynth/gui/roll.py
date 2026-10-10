"""The piano roll: one channel of one pattern as bars on a grid of rows and keys, and the sums that place them."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ..errors import RowSynthError
from ..song import _NAMES
from . import songview, theme
from .font import Font
from .layout import Rect
from .songview import Channel, Note, Pattern
from .theme import Colour
from .ui import NOWHERE, Event, Pos, Ui, Widget, inside

if TYPE_CHECKING:
    from .canvas import Canvas

#: How wide a row of the grid is at the start, and the least and the most the zoom makes it.
ROW_W = 9
ROW_W_MIN = 2
ROW_W_MAX = 24
#: A row narrower than this has no line of its own: only the beats and the bars have one.
ROW_LINES_FROM = 5
#: How high a key is, and with it its row of the grid.
KEY_H = 11
#: How wide the strip of keys down the left is, and how much of it a black key takes.
KEYS_W = 40
BLACK_W = 24
#: How high the strip along the top is in which the bars are numbered.
RULER_H = 10
#: How high the lane of the levels is, and how far under the grid it starts.
LANE_H = 34
GAP = 4
#: How wide the bar of a level is, where its note is not narrower, and the highest level a note has.
LEVEL_W = 4
LEVELS = 15
#: How far the bar of a ghost keeps from the upper and the lower edge of the bar a note has there.
GHOST_INSET = 3
#: The side of the notch at the start of a note that plays with another instrument than its channel's own.
NOTCH = 3
#: The keys of the roll, from the c of octave 0 to the b of octave 9: what a channel's text can name with `o`.
LOWEST = 12
HIGHEST = 131
#: What one step of the wheel scrolls: this many keys, or with shift this many pixels of rows.
WHEEL_KEYS = 2
WHEEL_PX = 36
#: How far the upper edge and the start of a note go towards the colour of the text, how far its lower edge and its
#: end go towards the background, and how far a ghost goes towards it.
LIGHTER = 0.45
DARKER = 0.4
GHOST = 0.6


def _px(value: float) -> int:
    """The whole pixel nearest to a place. A half goes up, so that places a whole number apart stay that far apart."""
    return math.floor(value + 0.5)


def mix(one: Colour, other: Colour, part: float = 0.5) -> Colour:
    """A colour between two: `part` of the way from the first to the second."""
    red, green, blue = (_px(a + (b - a) * part) for a, b in zip(one, other))
    return red, green, blue


# The lines of the grid, from the faintest: at every row, at every beat and at every bar. The faintest is the
# colour between the two the rows of the keys have, so it shows as little on the one as on the other.
_ROW_LINE, _BEAT_LINE, _BAR_LINE = theme.PANEL, theme.LINE, theme.DIM


def black(pitch: int) -> bool:
    """Whether the key of a pitch is a black one: those are the notes that are spelt with a flat."""
    return "-" in _NAMES[pitch % 12]


def name(pitch: int) -> str:
    """A pitch as a channel's text spells it, flats as `transpose` spells them, with its octave: 60 is c4, 63 e-4."""
    return f"{_NAMES[pitch % 12]}{pitch // 12 - 1}"


@dataclass(frozen=True)
class Map:
    """Where the rows and the pitches of a pattern fall in the rectangle of a roll, and back.

    rect is the roll's rectangle. row is the row at the left edge of its grid, top the pitch of the key at the
    upper edge of it, and row_w how wide a row is. row and top can be fractions: then part of a row, or of a key,
    is cut off at that edge.
    """

    rect: Rect
    row: float = 0.0
    top: float = 72.0
    row_w: int = ROW_W

    @property
    def grid(self) -> Rect:
        """Where the notes are: what the keys, the ruler and the lane leave of the rectangle."""
        x, y, w, h = self.rect
        return x + KEYS_W, y + RULER_H, max(0, w - KEYS_W), max(0, h - RULER_H - GAP - LANE_H)

    @property
    def keys(self) -> Rect:
        """The strip of keys, down the left of the grid."""
        x, y, _, _ = self.rect
        return x, y + RULER_H, KEYS_W, self.grid[3]

    @property
    def ruler(self) -> Rect:
        """The strip over the grid in which the bars are numbered."""
        x, _, w, _ = self.grid
        return x, self.rect[1], w, RULER_H

    @property
    def lane(self) -> Rect:
        """The lane of the levels, under the grid and as wide."""
        x, y, w, h = self.grid
        return x, y + h + GAP, w, LANE_H

    def x(self, row: float) -> int:
        """The x of the left edge of a row, which is where the line of the row is drawn."""
        return self.grid[0] + _px(row * self.row_w) - _px(self.row * self.row_w)

    def row_at(self, x: int) -> float:
        """The row at an x, counted in rows from the start of the pattern: its whole part is the row x is on."""
        return (x - self.grid[0] + _px(self.row * self.row_w)) / self.row_w

    def y(self, pitch: int) -> int:
        """The y of the upper edge of the key of a pitch, and of its row of the grid."""
        return self.grid[1] + _px(self.top * KEY_H) - pitch * KEY_H

    def pitch_at(self, y: int) -> int:
        """The pitch whose key, and whose row of the grid, a y is on."""
        return -((y - self.grid[1] - _px(self.top * KEY_H)) // KEY_H)

    def note(self, row: float, rows: float, pitch: float) -> Rect:
        """The rectangle of the bar of a note: from just after the line of its row to just before the line it ends at.

        It is at least one pixel wide. The lowest line of pixels of its key is left free, as that of the grid is.
        """
        left = self.x(row)
        return left + 1, self.y(round(pitch)), max(1, self.x(row + rows) - left - 1), KEY_H - 1

    def level(self, row: float, rows: float, level: int) -> Rect:
        """The rectangle of the bar of a note's level in the lane: nothing high at 0, and the most at `LEVELS`.

        It starts where the bar of the note does. The frame of the lane and a pixel of air inside it are left free.
        """
        x, _, w, _ = self.note(row, rows, 0)
        _, y, _, h = self.lane
        reach = level * (h - 4) // LEVELS
        return x, y + h - 2 - reach, min(w, LEVEL_W), reach

    def rows(self) -> range:
        """The whole rows of which at least a pixel is on the grid."""
        x, _, w, _ = self.grid
        return range(math.floor(self.row_at(x)), math.floor(self.row_at(x + w - 1)) + 1) if w else range(0)

    def pitches(self) -> range:
        """The pitches of which at least a line of pixels is on the grid, from the lowest."""
        _, y, _, h = self.grid
        return range(self.pitch_at(y + h - 1), self.pitch_at(y) + 1) if h else range(0)

    def zoomed(self, row_w: int, x: int) -> Map:
        """This map with rows of another width, no less than `ROW_W_MIN` and no more than `ROW_W_MAX`.

        The row at `x` stays at that x.
        """
        row_w = min(max(row_w, ROW_W_MIN), ROW_W_MAX)
        return replace(self, row_w=row_w, row=self.row_at(x) - (x - self.grid[0]) / row_w)

    def within(self, rows: float, bar_rows: int) -> Map:
        """This map, moved no further than its limits let it, on a pattern of `rows` rows with `bar_rows` to the bar.

        No more than a bar is shown before the pattern. No more than a bar is shown after it either, but by a grid
        too wide for that, which starts no later than row 0. No key is shown under `LOWEST` or over `HIGHEST`.
        """
        _, _, w, h = self.grid
        last = max(0.0, rows + bar_rows - w / self.row_w)
        top = min(float(HIGHEST), max(self.top, LOWEST - 1 + h / KEY_H))
        return replace(self, row=min(max(self.row, -float(bar_rows)), last), top=top)


class Roll(Widget):
    """The piano roll: the notes of one channel in one pattern as bars, between a strip of keys and a lane of levels.

    It shows, and does not change a note. `ghosts` is whether the notes of the other channels that are not drums
    are drawn dim behind. `playhead` is the row that is playing, which can be a fraction, or None: the owner sets it
    each frame. `row`, `top` and `row_w` are the view, as a `Map` has them. `help`, while the mouse is on a note or
    on the grid of the pattern, says what is there; anywhere else it is the line the roll was made with.
    """

    def __init__(self, rect: Rect, help: str = "") -> None:
        super().__init__(rect, help)
        self.pattern: Pattern | None = None
        self.channel = ""
        self.bar_rows = 16
        self.colour: Colour = theme.SONG
        self.ghosts = True
        self.playhead: float | None = None
        self.row, self.top, self.row_w = 0.0, 72.0, ROW_W
        self._own: Channel | None = None
        self._mouse: Pos = NOWHERE

    @property
    def help(self) -> str:
        """What the mouse is on, in plain words, or the line the roll was made with."""
        return self._tell(self._mouse) or self._help

    @help.setter
    def help(self, line: str) -> None:
        self._help = line

    @property
    def map(self) -> Map:
        """The sums of the roll as it is now."""
        return Map(self.rect, self.row, self.top, self.row_w)

    @property
    def notes(self) -> tuple[Note, ...]:
        """The notes of the channel that is shown, in the order they are written in."""
        return self._own.notes if self._own else ()

    def _rows(self) -> float:
        """The rows of the pattern that is shown."""
        return self.pattern.rows if self.pattern else 0.0

    def _see(self, seen: Map) -> None:
        """Take the view of a map, brought within its limits on the pattern that is shown."""
        seen = seen.within(self._rows(), self.bar_rows)
        self.row, self.top, self.row_w = seen.row, seen.top, seen.row_w

    def show(self, pattern: Pattern, channel: str, bar_rows: int, colour: Colour) -> None:
        """Show one channel of a pattern, with `bar_rows` rows to the bar, its notes in `colour`.

        The view is fitted when the pattern or the channel is another than before, by their names. When the same
        ones are shown again, with other notes, the view stays. A channel the pattern does not have is refused,
        and so is a bar that is not four beats of whole rows.
        """
        own = next((found for found in pattern.channels if found.name == channel), None)
        if own is None:
            raise RowSynthError(f"pattern {pattern.name!r} has no channel called {channel!r}")
        if bar_rows < 4 or bar_rows % 4:
            raise RowSynthError(f"a bar is four beats of whole rows, which {bar_rows} rows are not")
        another = self.pattern is None or (self.pattern.name, self.channel) != (pattern.name, channel)
        self.pattern, self.channel, self.bar_rows, self.colour, self._own = pattern, channel, bar_rows, colour, own
        if another:
            self.fit()
        else:
            self._see(self.map)

    def fit(self) -> None:
        """Set the view to start at row 0, with the channel's notes in the middle of the keys. The zoom stays."""
        low, high = songview.span(self.notes)
        spare = self.map.grid[3] / KEY_H - (high - low + 1)
        self._see(replace(self.map, row=0.0, top=float(high + math.floor(spare / 2))))

    def note_at(self, pos: Pos) -> int | None:
        """Which of the channel's notes has its bar under a position: its index among them, or None for none."""
        seen = self.map
        if inside(seen.grid, pos):
            notes = self.notes
            # The last is drawn on top of the others.
            for index in reversed(range(len(notes))):
                if inside(seen.note(notes[index].row, notes[index].rows, notes[index].pitch), pos):
                    return index
        return None

    def _tell(self, pos: Pos) -> str:
        """What is at a position, in plain words: a note, or a place on the grid of the pattern. Else no words."""
        index = self.note_at(pos)
        if index is not None:
            note = self.notes[index]
            rows = round(note.rows, 2)
            line = f"{name(round(note.pitch))}, {rows:g} {'row' if rows == 1 else 'rows'}, level {note.level}"
            return line if note.inst == self._own.inst else f"{line}, instrument {note.inst}"
        seen = self.map
        row = math.floor(seen.row_at(pos[0]))
        if not inside(seen.grid, pos) or not 0 <= row < self._rows():
            return ""
        return f"{name(seen.pitch_at(pos[1]))}, bar {row // self.bar_rows + 1}, row {row % self.bar_rows + 1}"

    def on(self, event: Event, ui: Ui) -> bool:
        """Note where the mouse is, and use the wheel: it scrolls the keys, with shift the rows, with ctrl it zooms."""
        if event.kind == "key":
            return False
        self._mouse = event.pos
        if event.kind != "wheel":
            return False
        seen = self.map
        if "ctrl" in event.mods:
            seen = seen.zoomed(seen.row_w + event.wheel, max(event.pos[0], seen.grid[0]))
        elif "shift" in event.mods:
            seen = replace(seen, row=seen.row - event.wheel * WHEEL_PX / seen.row_w)
        else:
            seen = replace(seen, top=seen.top + event.wheel * WHEEL_KEYS)
        self._see(seen)
        return True

    def draw(self, canvas: Canvas) -> None:
        """Draw the keys, the grid, the notes, the lane of the levels and the playhead: all of it in the rectangle."""
        seen = self.map
        with canvas.clip(self.rect):
            canvas.box(self.rect, theme.PANEL)
            self._draw_keys(canvas, seen)
            self._draw_grid(canvas, seen)
            self._draw_notes(canvas, seen)
            self._draw_lane(canvas, seen)
            if self.playhead is not None:
                _, y, _, h = self.rect
                with canvas.clip((seen.grid[0], y, seen.grid[2], h)):
                    canvas.vline(seen.x(self.playhead), y, h, theme.ACTIVE)

    def _draw_keys(self, canvas: Canvas, seen: Map) -> None:
        """The keys: each a light block, a black one dark for most of its length, and each c named."""
        x, _, w, _ = seen.keys
        with canvas.clip(seen.keys):
            for pitch in seen.pitches():
                key = (x, seen.y(pitch), w - 1, KEY_H)
                canvas.box(key, theme.TEXT)
                canvas.frame(key, theme.BACKGROUND)
                if black(pitch):
                    canvas.box((x, key[1], BLACK_W, KEY_H), theme.BACKGROUND)
                elif pitch % 12 == 0:
                    canvas.text_in(name(pitch), (x, key[1], w - 5, KEY_H), theme.BACKGROUND, "right")

    def _draw_grid(self, canvas: Canvas, seen: Map) -> None:
        """The rows of the keys as far as the pattern goes, the lines of its rows, and the numbers of its bars."""
        _, y, _, h = seen.grid
        rows = self._rows()
        left, right = seen.x(0), seen.x(rows)
        with canvas.clip(seen.grid):
            for pitch in seen.pitches():
                top = seen.y(pitch)
                canvas.box((left, top, right - left, KEY_H), theme.BACKGROUND if black(pitch) else theme.RAISED)
                canvas.hline(left, top + KEY_H - 1, right - left, theme.PANEL)
            for row in seen.rows():
                if not 0 <= row <= rows:
                    continue
                if row % self.bar_rows == 0:
                    canvas.vline(seen.x(row), y, h, _BAR_LINE)
                elif row % (self.bar_rows // 4) == 0:
                    canvas.vline(seen.x(row), y, h, _BEAT_LINE)
                elif seen.row_w >= ROW_LINES_FROM:
                    canvas.vline(seen.x(row), y, h, _ROW_LINE)
        with canvas.clip(seen.ruler):
            for bar in range(math.ceil(rows / self.bar_rows)):
                number = (seen.x(bar * self.bar_rows) + 2, seen.ruler[1], self.bar_rows * seen.row_w - 3, RULER_H)
                canvas.text_in(str(bar + 1), number, theme.DIM)

    def _draw_notes(self, canvas: Canvas, seen: Map) -> None:
        """The ghosts, each in the colour its channel takes by its place in the song, and over them the notes."""
        with canvas.clip(seen.grid):
            for index, channel in enumerate(self.pattern.channels if self.pattern and self.ghosts else ()):
                if channel is self._own or channel.drums:
                    continue
                colour = mix(theme.CHANNELS[index % len(theme.CHANNELS)], theme.BACKGROUND, GHOST)
                for note in channel.notes:
                    x, y, w, h = seen.note(note.row, note.rows, note.pitch)
                    canvas.box((x, y + GHOST_INSET, w, h - 2 * GHOST_INSET), colour)
            light, dark = mix(self.colour, theme.TEXT, LIGHTER), mix(self.colour, theme.BACKGROUND, DARKER)
            for note in self.notes:
                bar = x, y, w, h = seen.note(note.row, note.rows, note.pitch)
                canvas.box(bar, self.colour)
                if w > 2:
                    canvas.hline(x, y, w, light)
                    canvas.hline(x, y + h - 1, w, dark)
                    canvas.vline(x + w - 1, y, h, dark)
                    canvas.vline(x, y, h, light)
                if note.inst != self._own.inst:
                    for step in range(NOTCH):
                        canvas.hline(x, y + step, min(w, NOTCH - step), theme.INSTRUMENT)
                label = name(round(note.pitch))
                if w >= Font.width(label) + 5:
                    canvas.text_in(label, (x + 3, y, w - 5, h), theme.BACKGROUND)

    def _draw_lane(self, canvas: Canvas, seen: Map) -> None:
        """The lane under the grid, with its name beside it and in it a bar for the level of each note."""
        x, y, w, h = seen.lane
        canvas.text_in("LEVEL", (self.rect[0], y, KEYS_W - 5, h), theme.DIM, "right")
        canvas.box(seen.lane, theme.BACKGROUND)
        with canvas.clip((x + 1, y + 1, w - 2, h - 2)):
            for note in self.notes:
                canvas.box(seen.level(note.row, note.rows, note.level), self.colour)
        canvas.frame(seen.lane, theme.LINE)
