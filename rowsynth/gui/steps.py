"""The drum steps: the drums of a pattern as lanes of squares, one for each letter of a drum channel, and the sums."""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ..checks import _whole_rows
from ..errors import RowSynthError
from . import theme
from .font import Font
from .layout import Rect
from .theme import Colour
from .ui import NOWHERE, Event, Pos, Ui, Widget

if TYPE_CHECKING:
    from .canvas import Canvas
    from .songview import DrumRow, Pattern

#: A square: how wide it is, and how high, which is the height of a drum's lane.
SQUARE_W = 6
LANE_H = 30
#: What is left free between two squares of one beat, between two beats, and between two bars.
GAP = 2
BEAT_GAP = 3
BAR_GAP = 6
#: What is left free between two lanes, and how much more between the lanes of one channel and those of the next.
LANE_GAP = 8
CHANNEL_GAP = 7
#: The strip down the left that the labels stand in, and how much of it is left free before the first square.
LABELS_W = 92
LABEL_GAP = 4
#: From the sides of a label's box to its text, and from its top and its bottom to its two lines.
LABEL_PAD = 5
LABEL_EDGE = 2
#: The strip along the top that the bars are numbered in.
BARS_H = 13
#: How wide the mark of a hit that starts between two rows.
MARK_W = 2

#: How much of its beat's colour an empty square has in its fill and in its outline: the rest is the background.
EMPTY_FILL = 0.2
EMPTY_LINE = 0.38
#: How much of it a hit at level 0 has. A hit at the top level has all of it.
FAINTEST = 0.5
TOP_LEVEL = 15
#: How much of the colour of text is in the outline of a hit, and in a square that the playhead is in.
EDGE = 0.3
GLOW = 0.4

# From the left of one square to the left of the next in its beat.
_PITCH = SQUARE_W + GAP


def mix(one: Colour, other: Colour, share: float) -> Colour:
    """A colour between two: the first at a share of 0, the other at a share of 1."""
    red, green, blue = (round(a + (b - a) * share) for a, b in zip(one, other))
    return red, green, blue


def hit_colour(colour: Colour, level: int) -> Colour:
    """What a hit is filled with: its beat's colour at the top level, and dimmer the lower its level.

    At level 0 it is still brighter than the outline of an empty square.
    """
    return mix(theme.BACKGROUND, colour, FAINTEST + (1 - FAINTEST) * level / TOP_LEVEL)


def paint(colour: Colour, level: int | None, lit: bool = False) -> tuple[Colour, Colour]:
    """What a square of a beat's colour is filled with, and the colour of its outline.

    `level` is that of its hit, and None for an empty square, which is a dim outline round a fill dimmer still.
    A hit has a lighter outline. `lit`, as where the playhead is, both are lighter.
    """
    if level is None:
        fill, line = mix(theme.BACKGROUND, colour, EMPTY_FILL), mix(theme.BACKGROUND, colour, EMPTY_LINE)
    else:
        fill = hit_colour(colour, level)
        line = mix(fill, theme.TEXT, EDGE)
    return (mix(fill, theme.TEXT, GLOW), mix(line, theme.TEXT, GLOW)) if lit else (fill, line)


@dataclass(frozen=True)
class Lane:
    """One lane of the steps: a drum channel, one drum letter it plays, and the instrument that letter plays."""

    channel: str
    letter: str
    inst: str


def lanes(patterns: Iterable[Pattern], drums: Sequence[DrumRow]) -> list[Lane]:
    """The lanes of a song: for each of its drum channels, in the song's order, a lane for each letter it plays.

    `patterns` are the patterns the song plays, and `drums` its drum letters. The letters of a channel are those
    it plays in any of the patterns, in the order `drums` has them. A drum channel that plays none anywhere has a
    lane for every letter, so that it is seen and can be filled.
    """
    played: dict[str, set[str]] = {}
    for pattern in patterns:
        for channel in pattern.channels:
            if channel.drums:
                played.setdefault(channel.name, set()).update(note.drum for note in channel.notes if note.drum)
    return [Lane(channel, drum.letter, drum.inst) for channel, letters in played.items()
            for drum in drums if drum.letter in letters or not letters]


@dataclass(frozen=True)
class Grid:
    """Where the steps of a pattern are in a rectangle: the sums, with nothing drawn.

    rect is the widget's rectangle. rows is the whole rows of the pattern and drums how many lanes it has, each
    the lane of one drum, counted from 0 at the top. bar_rows and beat_rows are the rows in a bar and in a beat.
    The view is first_row and first_drum: the row whose squares are the first from the left, and the drum whose
    lane is at the top. breaks are the drums whose lane is the first of another channel than the lane above it:
    each has `CHANNEL_GAP` more air above it.
    """

    rect: Rect
    rows: int
    drums: int
    bar_rows: int
    beat_rows: int
    first_row: int = 0
    first_drum: int = 0
    breaks: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.bar_rows < 1 or self.beat_rows < 1:
            raise RowSynthError(f"a bar and a beat are 1 row or more each, not {self.bar_rows} and {self.beat_rows}")

    @property
    def area(self) -> Rect:
        """Where the squares are: the rectangle less the labels' strip and the strip of the bar numbers."""
        x, y, w, h = self.rect
        return x + LABELS_W, y + BARS_H, max(0, w - LABELS_W), max(0, h - BARS_H)

    @property
    def _bar_w(self) -> int:
        """From the left of one bar to the left of the next: its squares, and the gaps in it and after it."""
        beat_gaps = (self.bar_rows - 1) // self.beat_rows
        return self.bar_rows * _PITCH + beat_gaps * (BEAT_GAP - GAP) + BAR_GAP - GAP

    def _along(self, row: int) -> int:
        """From the left of the pattern's first square to the left of the square of a row."""
        bars, within = divmod(row, self.bar_rows)
        return bars * self._bar_w + within * _PITCH + within // self.beat_rows * (BEAT_GAP - GAP)

    def x(self, row: float) -> int:
        """The x of the left of a row's squares.

        A place between two rows is further on by its share of a square and the gap after it.
        """
        whole = math.floor(row)
        return self.area[0] + self._along(whole) - self._along(self.first_row) + int((row - whole) * _PITCH)

    def row_at(self, x: int) -> int | None:
        """The row whose squares an x is on. None in a gap, off the squares' room, and past the last row."""
        left, _, width, _ = self.area
        if not left <= x < left + width:
            return None
        bars, along = divmod(x - left + self._along(self.first_row), self._bar_w)
        beats, along = divmod(along, self.beat_rows * _PITCH + BEAT_GAP - GAP)
        step, along = divmod(along, _PITCH)
        within = beats * self.beat_rows + step
        row = bars * self.bar_rows + within
        # Past a square, past the last square of a beat, past the last of a bar, and past the last of the pattern.
        if along >= SQUARE_W or step >= self.beat_rows or within >= self.bar_rows or row >= self.rows:
            return None
        return row

    def _down(self, drum: int) -> int:
        """From the top of the first lane to the top of a drum's lane: the lanes above it and the air between."""
        return drum * (LANE_H + LANE_GAP) + CHANNEL_GAP * sum(first <= drum for first in self.breaks)

    def y(self, drum: int) -> int:
        """The y of the top of a drum's lane. The air above the lane at the top of the view is not in view."""
        return self.area[1] + self._down(drum) - self._down(self.first_drum)

    def drum_at(self, y: int) -> int | None:
        """The drum whose lane a y is on. None between two lanes, off the squares' room, and under the last lane."""
        _, top, _, height = self.area
        if not top <= y < top + height:
            return None
        return next((drum for drum in self.drums_in_view() if self.y(drum) <= y < self.y(drum) + LANE_H), None)

    def square(self, drum: int, row: int) -> Rect:
        """The square of one step: one row of one drum."""
        return self.x(row), self.y(drum), SQUARE_W, LANE_H

    def rows_in_view(self) -> range:
        """The rows of the pattern whose squares are in view, or partly so."""
        left, _, width, _ = self.area
        stop = self.first_row
        while stop < self.rows and self.x(stop) < left + width:
            stop += 1
        return range(self.first_row, stop)

    def drums_in_view(self) -> range:
        """The drums whose lanes are in view, or partly so."""
        _, top, _, height = self.area
        stop = self.first_drum
        while stop < self.drums and self.y(stop) < top + height:
            stop += 1
        return range(self.first_drum, stop)

    def colour(self, row: int) -> Colour:
        """The colour a row's squares take: the first of the four in the first beat of a bar, and so on round."""
        return theme.STEPS[row % self.bar_rows // self.beat_rows % len(theme.STEPS)]

    def scrolled(self, rows: int = 0, drums: int = 0) -> Grid:
        """The same steps with the view moved by so many rows and so many drums, and kept inside the pattern.

        The furthest it goes is where the last row, and the last drum, is whole in view.
        """
        _, _, width, height = self.area
        over = self._along(self.rows - 1) + SQUARE_W - width
        last_row = next((row for row in range(self.rows) if self._along(row) >= over), max(0, self.rows - 1))
        under = self._down(self.drums - 1) + LANE_H - height
        last_drum = next((drum for drum in range(self.drums) if self._down(drum) >= under), max(0, self.drums - 1))
        return replace(self, first_row=min(max(0, self.first_row + rows), last_row),
                       first_drum=min(max(0, self.first_drum + drums), last_drum))


class Steps(Widget):
    """The drums of one pattern, from all its drum channels together: a lane of squares for each `Lane`, a square a row.

    The lanes of one channel stand together, with more air and a faint line before those of the next. `picked` is
    the name of a channel, or None: the labels of its lanes are lit. `playhead` is the row that is playing, which
    may be between two rows, or None. Their owner sets both. The wheel scrolls the lanes, and with shift the rows.
    `help` is the widget's own line, and while the mouse is on a square it says what is there.
    """

    def __init__(self, rect: Rect, help: str = "") -> None:
        super().__init__(rect, help)
        self.playhead: float | None = None
        self.picked: str | None = None
        self._grid = Grid(rect, 0, 0, 1, 1)
        self._lanes: tuple[Lane, ...] = ()
        #: Whether the lanes are of more than one channel: a lane is then told by its channel as well.
        self._several = False
        #: The level of each hit that starts on a whole row, by the place of its lane and its row.
        self._hits: dict[tuple[int, int], int] = {}
        #: The place of the lane, the row and the level of each hit that starts between two rows.
        self._marks: list[tuple[int, float, int]] = []
        self._mouse: Pos = NOWHERE

    def show(self, pattern: Pattern, lanes: Sequence[Lane], bar_rows: int, beat_rows: int) -> None:
        """Show the drums of a pattern. Each lane is one letter of one of its drum channels, as `lanes` makes them.

        The view stays where it is, as far as the pattern reaches. A lane of a channel the pattern does not have
        is refused, and so is one of a channel that is not a drum channel.
        """
        lanes = tuple(lanes)
        held = {channel.name: channel for channel in pattern.channels}
        for lane in lanes:
            if lane.channel not in held:
                raise RowSynthError(f"pattern {pattern.name!r} has no channel called {lane.channel!r}")
            if not held[lane.channel].drums:
                raise RowSynthError(f"channel {lane.channel!r} is not a drum channel")
        breaks = tuple(place for place in range(1, len(lanes)) if lanes[place].channel != lanes[place - 1].channel)
        self._grid = Grid(self.rect, math.ceil(_whole_rows(pattern.rows)), len(lanes), bar_rows, beat_rows,
                          self._grid.first_row, self._grid.first_drum, breaks)
        self._lanes = lanes
        places = {(lane.channel, lane.letter): place for place, lane in enumerate(lanes)}
        channels = dict.fromkeys(lane.channel for lane in lanes)
        self._several = len(channels) > 1
        self._hits, self._marks = {}, []
        for channel in channels:
            for note in held[channel].notes:
                place = places.get((channel, note.drum))
                # A row that is a hair off a whole one, as adding up triplets leaves it, is that whole row.
                row = _whole_rows(note.row)
                if place is not None and float(row).is_integer():
                    self._hits[place, int(row)] = note.level
                elif place is not None:
                    self._marks.append((place, note.row, note.level))

    @property
    def grid(self) -> Grid:
        """The sums for the rectangle as it is now, the view kept inside the pattern."""
        return replace(self._grid, rect=self.rect).scrolled()

    @property
    def help(self) -> str:
        """What is at the square the mouse is on, in plain words. Off the squares, the widget's own line.

        The drum is told by its letter, and where the lanes are of more than one channel by its channel as well.
        """
        found = self._at(self._mouse)
        if found is None:
            return self._help
        lane, row = self._lanes[found[0]], found[1]
        bar, step = divmod(row, self._grid.bar_rows)
        between = [level for place, at, level in self._marks if (place, math.floor(at)) == found]
        if found in self._hits:
            what = f"hit, level {self._hits[found]}"
        elif between:
            what = f"hit off the grid, level {between[0]}"
        else:
            what = "empty"
        drum = f"{lane.letter} in {lane.channel}" if self._several else lane.letter
        return f"{drum}, bar {bar + 1}, step {step + 1}: {what}"

    @help.setter
    def help(self, text: str) -> None:
        self._help = text

    def _at(self, pos: Pos) -> tuple[int, int] | None:
        """The place of the lane and the row of the square a position is on, or None when it is on no square."""
        grid = self.grid
        row, drum = grid.row_at(pos[0]), grid.drum_at(pos[1])
        return None if row is None or drum is None else (drum, row)

    def step_at(self, pos: Pos) -> tuple[str, str, int] | None:
        """The channel, the drum letter and the row of the square a position is on, or None when it is on none."""
        found = self._at(pos)
        if found is None:
            return None
        lane = self._lanes[found[0]]
        return lane.channel, lane.letter, found[1]

    def on(self, event: Event, ui: Ui) -> bool:
        """Follow the mouse, and scroll at the wheel: the lanes by one, and with shift the rows by a beat.

        A step of the wheel is used when it moved the view.
        """
        if event.kind == "key":
            return False
        self._mouse = event.pos
        if event.kind != "wheel":
            return False
        grid = self.grid
        if "shift" in event.mods:
            self._grid = grid.scrolled(rows=-event.wheel * grid.beat_rows)
        else:
            self._grid = grid.scrolled(drums=-event.wheel)
        return self._grid != grid

    def draw(self, canvas: Canvas) -> None:
        """Draw the labels, the lines between channels, the bar numbers, the squares, the hits between rows and
        the playhead.

        Nothing is drawn outside the rectangle, and no square where the labels are.
        """
        grid = self.grid
        rows, drums = grid.rows_in_view(), grid.drums_in_view()
        x, y, w, h = self.rect
        left, _, width, _ = grid.area
        playing = self.playhead if self.playhead is not None and 0 <= self.playhead < grid.rows else None
        with canvas.clip(self.rect):
            for drum in drums:
                # The line is in the middle of the air above the first lane of a channel, where that air is in view.
                if drum in grid.breaks and drum > drums.start:
                    canvas.hline(x, grid.y(drum) - (LANE_GAP + CHANNEL_GAP + 1) // 2, w, theme.LIT)
                self._label(canvas, (x, grid.y(drum), LABELS_W - LABEL_GAP, LANE_H), self._lanes[drum])
            with canvas.clip((left, y, width, h)):
                for bar in range(rows.start // grid.bar_rows, -(-rows.stop // grid.bar_rows)):
                    start = bar * grid.bar_rows
                    bar_w = grid.x(start + grid.bar_rows) - BAR_GAP - grid.x(start)
                    canvas.text_in(f"BAR {bar + 1}", (grid.x(start), y, bar_w, BARS_H), theme.DIM)
                for drum in drums:
                    for row in rows:
                        lit = playing is not None and math.floor(playing) == row
                        fill, line = paint(grid.colour(row), self._hits.get((drum, row)), lit)
                        square = grid.square(drum, row)
                        canvas.box(square, fill)
                        canvas.frame(square, line)
                for drum, at, level in self._marks:
                    if drum in drums and math.floor(at) in rows:
                        canvas.box((grid.x(at), grid.y(drum), MARK_W, LANE_H),
                                   hit_colour(grid.colour(math.floor(at)), level))
                if playing is not None and drums:
                    canvas.vline(grid.x(playing), y, grid.y(drums[-1]) + LANE_H - y, theme.ACTIVE)

    def _label(self, canvas: Canvas, box: Rect, lane: Lane) -> None:
        """Draw the label of a lane in its box: its letter, and under it the name of the instrument it plays.

        Where the lanes are of more than one channel, the name of its channel stands at the right of the letter.
        The box of a lane of the picked channel is lit.
        """
        x, y, w, h = box
        canvas.box(box, theme.LIT if lane.channel == self.picked else theme.RAISED)
        canvas.frame(box, theme.LINE)
        line_h = (h - 2 * LABEL_EDGE) // 2
        left, top, wide = x + LABEL_PAD, y + LABEL_EDGE, w - 2 * LABEL_PAD
        canvas.text_in(lane.letter, (left, top, wide, line_h), theme.TEXT)
        if self._several:
            taken = Font.width(lane.letter) + LABEL_PAD
            canvas.text_in(lane.channel, (left + taken, top, max(0, wide - taken), line_h), theme.DIM, "right")
        canvas.text_in(lane.inst, (left, top + line_h, wide, line_h), theme.INSTRUMENT)
