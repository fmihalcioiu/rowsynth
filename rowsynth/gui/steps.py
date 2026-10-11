"""The drum steps: the drums of a pattern as lanes of squares, one for each letter of a drum channel, and the sums."""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Container, Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from ..checks import _whole_rows
from ..errors import RowSynthError
from . import noteedit, songview, theme
from .font import Font
from .layout import Rect
from .theme import Colour
from .ui import NOWHERE, Event, Pos, Ui, Widget

if TYPE_CHECKING:
    from .canvas import Canvas
    from .noteedit import Notes
    from .songview import DrumRow, Note, Pattern

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
#: The dot in an empty square whose row another drum of its channel has: how wide it is, and how high.
DOT_W = 2
DOT_H = 2

#: How much of its beat's colour an empty square has in its fill and in its outline: the rest is the background.
EMPTY_FILL = 0.2
EMPTY_LINE = 0.38
#: How much of it a hit at level 0 has. A hit at the top level has all of it.
FAINTEST = 0.5
TOP_LEVEL = 15
#: How much of the colour of text is in the outline of a hit, and in a square that the playhead is in.
EDGE = 0.3
GLOW = 0.4
#: How much of its beat's colour the dot has in a square whose row another drum of its channel has.
DOT = 0.5

#: The lowest level the wheel brings a hit to: at 0 it would not be heard.
WHEEL_LEAST = 1
#: How many seconds after the last step of the wheel on a hit its run of steps ends, and is one edit.
WHEEL_WAIT = 0.5
#: What the help line has after what is at a square, where the pattern is not changed: why it is not.
OFF_GRID = "; the pattern has notes off the grid, so it is not changed here"

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


def dot(square: Rect, colour: Colour, lit: bool = False) -> tuple[Rect, Colour]:
    """Where the dot is in a square of a beat's colour, and what colour it is: in the middle, and dim.

    The dot is in an empty square whose row another drum of its channel has. `lit`, it is lighter.
    """
    x, y, w, h = square
    shade = mix(theme.BACKGROUND, colour, DOT)
    return (x + (w - DOT_W) // 2, y + (h - DOT_H) // 2, DOT_W, DOT_H), mix(shade, theme.TEXT, GLOW) if lit else shade


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

    def passed(self, one: int, other: int) -> range:
        """The rows in view whose squares are on the way from an x to another, the squares at the two ends with them.

        It is every such row, however far apart the two are, and the same rows either way round. An x in a gap
        is on no square, and one past the squares in view is past the last of them.
        """
        low, high = sorted((one, other))
        on = [row for row in self.rows_in_view() if self.x(row) <= high and low < self.x(row) + SQUARE_W]
        return range(on[0], on[-1] + 1) if on else range(0)

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


@dataclass
class Gesture:
    """What the mouse is in the middle of doing to the hits of one lane, and what it comes to.

    kind is "put" for a drag that puts a hit in each empty square it passes, "take" for a drag that takes away
    each hit it passes, and "level" for a run of steps of the wheel on one hit. drum is the place of the lane,
    and rows the rows of the squares the gesture has passed. level is the level a put hit gets, and the one the
    wheel has brought its hit to. button is the mouse button of a drag, and x where the mouse was last in it.
    until is the time on the clock at which a run of the wheel ends.
    """

    kind: str
    drum: int
    level: int
    rows: list[int] = field(default_factory=list)
    button: int = 0
    x: int = 0
    until: float = 0.0

    def along(self, grid: Grid, x: int) -> list[int]:
        """Take a drag on to an x. Given are the rows of the squares on the way that it had not passed before."""
        fresh = [row for row in grid.passed(self.x, x) if row not in self.rows]
        self.x = x
        self.rows += fresh
        return fresh

    def gives(self, notes: Sequence[Note], letter: str, drums: dict | None, rows: float,
              rows_per_beat: int = 4) -> Notes:
        """The notes of the lane's channel as the gesture leaves them, by the rules of `noteedit`.

        `notes` are those of the channel before the gesture, and `letter` is the lane's. The drum table of the
        song, the rows of the pattern and the song's rows to a beat are as `noteedit.hit` takes them. A put
        fills the squares it passed at whose rows the channel played nothing. What the channel plays at a row
        stays, and no hit is put there: only in the first square, the one that was pressed, does the hit take
        the place of a hit of another letter. A take empties the squares that had a hit of the letter, and the
        wheel sets the level of its one hit.
        """
        notes = tuple(notes)
        held = {_whole_rows(note.row): note for note in notes}
        for row in sorted(self.rows):
            there = held.get(row)
            mine = there is not None and there.drum == letter
            # A note that is a hair off its row is replaced, and taken away, at the place it has.
            at = float(row) if there is None else there.row
            if self.kind == "put" and not mine and (there is None or row == self.rows[0]):
                notes = noteedit.hit(notes, at, letter, self.level, drums, rows, rows_per_beat)
            elif self.kind == "take" and mine:
                notes = noteedit.unhit(notes, at, letter, drums, rows, rows_per_beat)
            elif self.kind == "level" and mine:
                notes, _ = noteedit.with_level(notes, [notes.index(there)], self.level)
        return notes

    def label(self, count: int) -> str:
        """A few plain words for the undo list, for a gesture that put or took away so many hits."""
        if self.kind == "level":
            return "level changed"
        done = "placed" if self.kind == "put" else "removed"
        return f"hit {done}" if count == 1 else f"{count} hits {done}"


def begun(grid: Grid, hits: Container[tuple[int, int]], pos: Pos, button: int,
          level: int = TOP_LEVEL) -> Gesture | None:
    """The drag that a press of a mouse button begins at a position, and None where it begins none.

    `hits` has the squares with a hit in them, each as the place of its lane and its row. The left button puts
    hits when it is pressed on an empty square, at `level`, and takes them away when it is pressed on a hit. The
    right button takes them away wherever it is pressed. A press on no square begins nothing, and neither does
    any other button. The drag has passed no square yet: `along` takes it to where the press is.
    """
    row, drum = grid.row_at(pos[0]), grid.drum_at(pos[1])
    if row is None or drum is None or button not in (1, 3):
        return None
    return Gesture("take" if button == 3 or (drum, row) in hits else "put", drum, level, button=button, x=pos[0])


class Steps(Widget):
    """The drums of one pattern, from all its drum channels together: a lane of squares for each `Lane`, a square a row.

    The lanes of one channel stand together, with more air and a faint line before those of the next. `picked` is
    the name of a channel, or None: the labels of its lanes are lit. `playhead` is the row that is playing, which
    may be between two rows, or None. Their owner sets both. The wheel scrolls the lanes, and with shift the rows.

    The hits are switched on and off with the mouse, where `show` was given the drum table and the pattern is on
    the grid. The left button on an empty square puts a hit at `level`, and dragged it puts one in every empty
    square it passes. On a hit it takes the hit away, and dragged every hit it passes. The right button always
    takes away. A drag stays in the lane it began in. On a hit, with no key held, the wheel does not scroll: it
    changes the level of the hit by one a step, and `level` is the one it set last.

    A channel plays one drum at a time. So a hit that is put by a press takes the place of a hit that another
    letter of its channel has at that row, and a drag puts none in the squares it passes after that where
    another letter has one. Such a square is drawn empty with a dot in it, where the hits can be changed.

    Nothing is saved here. While a gesture is under way `busy` is true, and the steps show what it would give.
    When it ends, and a channel's notes are other than `show` gave them, `edited` is called with the name of the
    channel, its new notes and a few words for the undo list, and the steps show what `show` gave again: their
    owner saves, and shows the result. A drag ends at the release. A run of the wheel ends when the mouse leaves
    its square, or `WHEEL_WAIT` seconds after its last step by `clock`: for that their owner calls `update` every
    frame. `sounded` is called with the channel and the letter of each hit as it is put, for it to be heard. That
    is before the gesture has ended, so the owner's answer to `edited`, whether it saved, is not waited for.

    `help` is the widget's own line. While the mouse is on a square it says what a press there would do, with
    the level of a hit and the letter a hit would take the place of. Where nothing can be changed it says what
    is there.
    """

    def __init__(self, rect: Rect, help: str = "", edited: Callable[[str, Notes, str], object] | None = None,
                 sounded: Callable[[str, str], object] | None = None,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        super().__init__(rect, help)
        self.playhead: float | None = None
        self.picked: str | None = None
        self.edited = edited
        self.sounded = sounded
        self.clock = clock
        self.level = TOP_LEVEL
        self._grid = Grid(rect, 0, 0, 1, 1)
        self._lanes: tuple[Lane, ...] = ()
        #: Whether the lanes are of more than one channel: a lane is then told by its channel as well.
        self._several = False
        #: The notes of each channel that has a lane, as `show` gave them, and the rows of the pattern.
        self._notes: dict[str, Notes] = {}
        self._rows = 0.0
        #: The drum table of the song, whether the hits can be changed, and what the help line says where not.
        self._drums: dict | None = None
        self._open = False
        self._why = ""
        #: The gesture under way, and the notes it would give, by the name of their channel.
        self._gesture: Gesture | None = None
        self._given: dict[str, Notes] = {}
        #: The level of each hit that starts on a whole row, by the place of its lane and its row.
        self._hits: dict[tuple[int, int], int] = {}
        #: The place of the lane, the row and the level of each hit that starts between two rows.
        self._marks: list[tuple[int, float, int]] = []
        #: The empty squares whose row another drum of their channel has, each with the letter of that drum.
        self._taken: dict[tuple[int, int], str] = {}
        self._mouse: Pos = NOWHERE

    def show(self, pattern: Pattern, lanes: Sequence[Lane], bar_rows: int, beat_rows: int,
             drums: dict | None = None) -> None:
        """Show the drums of a pattern. Each lane is one letter of one of its drum channels, as `lanes` makes them.

        `beat_rows` is the song's rows to a beat. `drums` is the drum table of the song, as `noteedit.hit` takes
        it: with it the hits can be changed, where every note of the pattern is on the grid. Without it they are
        shown and no more. The view stays where it is, as far as the pattern reaches. A gesture under way is
        dropped, and nothing is called for it. A lane of a channel the pattern does not have is refused, and so
        are one of a channel that is not a drum channel and one of a letter the drum table does not have.
        """
        lanes = tuple(lanes)
        held = {channel.name: channel for channel in pattern.channels}
        for lane in lanes:
            if lane.channel not in held:
                raise RowSynthError(f"pattern {pattern.name!r} has no channel called {lane.channel!r}")
            if not held[lane.channel].drums:
                raise RowSynthError(f"channel {lane.channel!r} is not a drum channel")
            if drums is not None and lane.letter not in drums:
                raise RowSynthError(f"no drum letter {lane.letter!r}")
        breaks = tuple(place for place in range(1, len(lanes)) if lanes[place].channel != lanes[place - 1].channel)
        self._grid = Grid(self.rect, math.ceil(_whole_rows(pattern.rows)), len(lanes), bar_rows, beat_rows,
                          self._grid.first_row, self._grid.first_drum, breaks)
        self._lanes = lanes
        self._notes = {channel: held[channel].notes for channel in dict.fromkeys(lane.channel for lane in lanes)}
        self._rows = pattern.rows
        self._several = len(self._notes) > 1
        on_grid = songview.on_grid(pattern)
        self._drums, self._open = drums, drums is not None and on_grid
        self._why = OFF_GRID if drums is not None and not on_grid else ""
        self._gesture, self._given = None, {}
        self._look()

    def notes(self, channel: str) -> Notes:
        """The notes of a drum channel as the steps show them: with a gesture under way, as it would leave them.

        A channel that has no lane is refused.
        """
        if channel not in self._notes:
            raise RowSynthError(f"the steps show no channel called {channel!r}")
        return self._given.get(channel, self._notes[channel])

    @property
    def busy(self) -> bool:
        """Whether a gesture is under way: a drag that has not been released, or a run of the wheel not yet ended."""
        return self._gesture is not None

    def _look(self) -> None:
        """Work out which squares have a hit, the hits between two rows, and the squares whose row is taken.

        It is worked out from the notes as they are shown. A row is taken in the other lanes of a channel by
        what the channel plays there, which is a hit of another letter, or a note that is no hit.
        """
        self._hits, self._marks, self._taken = {}, [], {}
        for place, lane in enumerate(self._lanes):
            for note in self.notes(lane.channel):
                # A row that is a hair off a whole one, as adding up triplets leaves it, is that whole row.
                row = _whole_rows(note.row)
                if not float(row).is_integer():
                    if note.drum == lane.letter:
                        self._marks.append((place, note.row, note.level))
                elif note.drum == lane.letter:
                    self._hits[place, int(row)] = note.level
                else:
                    self._taken[place, int(row)] = note.drum

    @property
    def grid(self) -> Grid:
        """The sums for the rectangle as it is now, the view kept inside the pattern."""
        return replace(self._grid, rect=self.rect).scrolled()

    @property
    def help(self) -> str:
        """What a press on the square under the mouse would do, in plain words. Off the squares, the widget's own line.

        On a hit its level comes after, and on a square whose row another drum of the channel has, that the hit
        would take its place. Where the hits cannot be changed it is what is at the square, and for a pattern off
        the grid why it is not changed. The drum is told by its letter, and where the lanes are of more than one
        channel by its channel as well.
        """
        found = self._at(self._mouse)
        if found is None:
            return self._help
        lane, row = self._lanes[found[0]], found[1]
        bar, step = divmod(row, self._grid.bar_rows)
        drum = f"{lane.letter} in {lane.channel}" if self._several else lane.letter
        where = f"{drum}, bar {bar + 1}, step {step + 1}"
        if self._open and found in self._hits:
            return f"take the hit away: {where} (level {self._hits[found]})"
        if self._open:
            other = self._taken.get(found)
            why = "" if other is None else (f" (takes the place of {other or 'a note'}: "
                                            f"{lane.channel} plays one drum at a time)")
            return f"put a hit: {where}{why}"
        between = [level for place, at, level in self._marks if (place, math.floor(at)) == found]
        if found in self._hits:
            what = f"hit, level {self._hits[found]}"
        elif between:
            what = f"hit off the grid, level {between[0]}"
        else:
            what = "empty"
        return f"{where}: {what}{self._why}"

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
        """Follow the mouse. Begin a drag at a press on a square, take it along, and end it at the release.

        The wheel changes the level of the hit it is on, when no key is held. Anywhere else it scrolls: the lanes
        by one, and with shift the rows by a beat. A press is used when it began a drag, and so is all that the
        mouse does up to the release. A step of the wheel is used on a hit, and when it moved the view.
        """
        if event.kind == "key":
            return False
        self.update()
        self._mouse = event.pos
        doing = self._gesture
        if doing is not None and doing.kind != "level":
            if event.kind == "move":
                self._drag(event.pos[0])
            elif event.kind == "up" and event.button == doing.button:
                self._end()
            return True
        if doing is not None and self._at(event.pos) != (doing.drum, doing.rows[0]):
            self._end()
        if event.kind == "wheel":
            return self._turn(event)
        return event.kind == "down" and self._press(event)

    def update(self) -> None:
        """End a run of the wheel whose time is up. The owner calls this every frame."""
        doing = self._gesture
        if doing is not None and doing.kind == "level" and self.clock() >= doing.until:
            self._end()

    def _press(self, event: Event) -> bool:
        """Begin the drag that a press begins, and say whether it began one. A run of the wheel ends first."""
        self._end()
        if not self._open:
            return False
        self._gesture = begun(self.grid, self._hits, event.pos, event.button, self.level)
        if self._gesture is None:
            return False
        self._drag(event.pos[0])
        return True

    def _drag(self, x: int) -> None:
        """Take the drag under way on to an x, and show what it would give. Each hit it newly puts is sounded."""
        doing = self._gesture
        lane = self._lanes[doing.drum]
        fresh, had = doing.along(self.grid, x), self._hits
        self._try()
        for row in fresh:
            # Not every square on the way gets a hit: one may have had its hit, or another drum of the channel.
            if self.sounded is not None and (doing.drum, row) in self._hits and (doing.drum, row) not in had:
                self.sounded(lane.channel, lane.letter)

    def _turn(self, event: Event) -> bool:
        """Take a step of the wheel: to the level of the hit under it when no key is held, and else to the view."""
        found = self._at(event.pos)
        level = self._hits.get(found) if self._open and not event.mods else None
        if level is None:
            self._end()
            grid = self.grid
            if "shift" in event.mods:
                self._grid = grid.scrolled(rows=-event.wheel * grid.beat_rows)
            else:
                self._grid = grid.scrolled(drums=-event.wheel)
            return self._grid != grid
        if self._gesture is None:
            self._gesture = Gesture("level", found[0], level, [found[1]])
        # A hit that was written at a level under the least is not brought up to it by a step down.
        self.level = self._gesture.level = min(max(level + event.wheel, min(level, WHEEL_LEAST)), TOP_LEVEL)
        self._gesture.until = self.clock() + WHEEL_WAIT
        self._try()
        return True

    def _try(self) -> None:
        """Work out what the gesture under way would give, and show that."""
        doing = self._gesture
        lane = self._lanes[doing.drum]
        self._given = {lane.channel: doing.gives(self._notes[lane.channel], lane.letter, self._drums, self._rows,
                                                 self._grid.beat_rows)}
        self._look()

    def _end(self) -> None:
        """End the gesture under way, where there is one, and show what `show` gave again.

        `edited` is called for each channel whose notes the gesture has made other than they were.
        """
        doing, given, before = self._gesture, self._given, self._notes
        if doing is None:
            return
        self._gesture, self._given = None, {}
        self._look()
        letter = self._lanes[doing.drum].letter
        for channel, notes in given.items():
            if notes != before[channel] and self.edited is not None:
                # The hits of the lane's letter are more, or fewer, by as many as the drag put or took away.
                now, then = (sum(note.drum == letter for note in held) for held in (notes, before[channel]))
                self.edited(channel, notes, doing.label(abs(now - then)))

    def draw(self, canvas: Canvas) -> None:
        """Draw the labels, the lines between channels, the bar numbers, the squares, the hits between rows and
        the playhead.

        Where the hits can be changed, a dot is in each empty square whose row another drum of its channel has.
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
                        if self._open and (drum, row) in self._taken:
                            canvas.box(*dot(square, grid.colour(row), lit))
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
