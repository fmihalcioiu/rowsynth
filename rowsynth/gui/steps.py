"""The drum steps: the drums of a pattern as lanes of squares, one for each letter of a drum channel, and the sums."""
from __future__ import annotations

import math
from collections.abc import Callable, Container, Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from ..checks import _whole_rows
from ..errors import RowSynthError
from . import noteedit, songview, theme
from .font import Font
from .kit import ACROSS, VELOCITY_HELP, ZOOM, Plus, ScrollBar, Velocity, told, wheel
from .layout import Rect
from .panels import within
from .roll import OFF_GRID
from .theme import Colour
from .ui import NOWHERE, Event, Pos, Ui, Widget, inside

if TYPE_CHECKING:
    from .canvas import Canvas
    from .noteedit import Notes
    from .songview import DrumRow, Note, Pattern

#: A square: how wide it is at the start, and the least and the most the zoom makes it. And how high it is, which
#: is the height of a drum's lane.
SQUARE_W = 6
SQUARE_W_MIN = 3
SQUARE_W_MAX = 16
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
#: The room a scroll bar takes beside the squares and under them: the bar, and the air between it and them.
BAR = theme.SCROLL_W + theme.SCROLL_GAP
#: From the last lane down to the plus that adds a drum, and the room the two take under that lane.
PLUS_GAP = 3
TAIL = PLUS_GAP + theme.PLUS
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

#: What a press on a square can begin, as `begun` calls it.
PUT, TAKE, MOVE = "put", "take", "move"
#: What the wheel does anywhere on the steps, as the help line says it, and what the two scroll bars say.
WHEEL_HELP = "along the lanes, shift along the rows, ctrl zooms"
ALONG_HELP = told("drag the bar to go along the lanes, or press beside it to jump", wheel=WHEEL_HELP)
ACROSS_HELP = told("drag the bar to go along the rows, or press beside it to jump", wheel=WHEEL_HELP)
#: What the plus under the last lane says.
PLUS_HELP = told("add a drum to the song: its letter is asked for")
#: What an edit in the velocity lane is called in the list of what can be undone.
VELOCITY_DONE = "velocity changed"


def _px(value: float) -> int:
    """The whole number nearest to a value. A half goes up."""
    return math.floor(value + 0.5)


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


def lanes(patterns: Iterable[Pattern], drums: Sequence[DrumRow], spare: str | None = None) -> list[Lane]:
    """The lanes of a song: for each of its drum channels, in the song's order, a lane for each letter it plays.

    `patterns` are the patterns the song plays, and `drums` its drum letters. The letters of a channel are those
    it plays in any of the patterns, in the order `drums` has them. A drum channel that plays none anywhere has a
    lane for every letter, so that it is seen and can be filled. `spare` is the name of a drum channel, the one
    that is picked: it has a lane as well for each letter that no channel plays anywhere, which is where a drum
    that was just added is first played.
    """
    played: dict[str, set[str]] = {}
    for pattern in patterns:
        for channel in pattern.channels:
            if channel.drums:
                played.setdefault(channel.name, set()).update(note.drum for note in channel.notes if note.drum)
    unplayed = {drum.letter for drum in drums}.difference(*played.values())
    return [Lane(channel, drum.letter, drum.inst) for channel, letters in played.items() for drum in drums
            if drum.letter in letters or not letters or (channel == spare and drum.letter in unplayed)]


@dataclass(frozen=True)
class Grid:
    """Where the steps of a pattern are in a rectangle: the sums, with nothing drawn.

    rect is the widget's rectangle. rows is the whole rows of the pattern and drums how many lanes it has, each
    the lane of one drum, counted from 0 at the top. bar_rows and beat_rows are the rows in a bar and in a beat.
    The view is first_row and first_drum: the row whose squares are the first from the left, and the drum whose
    lane is at the top. breaks are the drums whose lane is the first of another channel than the lane above it:
    each has `CHANNEL_GAP` more air above it. square_w is how wide a square is, which is the zoom.
    """

    rect: Rect
    rows: int
    drums: int
    bar_rows: int
    beat_rows: int
    first_row: int = 0
    first_drum: int = 0
    breaks: tuple[int, ...] = ()
    square_w: int = SQUARE_W

    def __post_init__(self) -> None:
        if self.bar_rows < 1 or self.beat_rows < 1:
            raise RowSynthError(f"a bar and a beat are 1 row or more each, not {self.bar_rows} and {self.beat_rows}")

    @property
    def area(self) -> Rect:
        """Where the squares are: what the labels' strip, the strip of the bar numbers and the two scroll bars leave."""
        x, y, w, h = self.rect
        return x + LABELS_W, y + BARS_H, max(0, w - LABELS_W - BAR), max(0, h - BARS_H - BAR)

    @property
    def names(self) -> Rect:
        """The column down the left of the squares that the labels of the lanes stand in, and the plus under them."""
        _, top, _, high = self.area
        return within((self.rect[0], top, LABELS_W, high), self.rect)

    @property
    def along(self) -> Rect:
        """The upright scroll bar, down the right of the squares: the one that goes along the lanes."""
        x, y, w, h = self.area
        return within((x + w + theme.SCROLL_GAP, y, theme.SCROLL_W, h), self.rect)

    @property
    def across(self) -> Rect:
        """The lying scroll bar, under the squares: the one that goes along the rows."""
        x, y, w, h = self.area
        return within((x, y + h + theme.SCROLL_GAP, w, theme.SCROLL_W), self.rect)

    @property
    def pitch(self) -> int:
        """From the left of one square to the left of the next in its beat."""
        return self.square_w + GAP

    @property
    def _bar_w(self) -> int:
        """From the left of one bar to the left of the next: its squares, and the gaps in it and after it."""
        beat_gaps = (self.bar_rows - 1) // self.beat_rows
        return self.bar_rows * self.pitch + beat_gaps * (BEAT_GAP - GAP) + BAR_GAP - GAP

    def _along(self, row: int) -> int:
        """From the left of the pattern's first square to the left of the square of a row."""
        bars, within = divmod(row, self.bar_rows)
        return bars * self._bar_w + within * self.pitch + within // self.beat_rows * (BEAT_GAP - GAP)

    def x(self, row: float) -> int:
        """The x of the left of a row's squares.

        A place between two rows is further on by its share of a square and the gap after it.
        """
        whole = math.floor(row)
        return self.area[0] + self._along(whole) - self._along(self.first_row) + int((row - whole) * self.pitch)

    def row_at(self, x: int) -> int | None:
        """The row whose squares an x is on. None in a gap, off the squares' room, and past the last row."""
        left, _, width, _ = self.area
        if not left <= x < left + width:
            return None
        bars, along = divmod(x - left + self._along(self.first_row), self._bar_w)
        beats, along = divmod(along, self.beat_rows * self.pitch + BEAT_GAP - GAP)
        step, along = divmod(along, self.pitch)
        within = beats * self.beat_rows + step
        row = bars * self.bar_rows + within
        # Past a square, past the last square of a beat, past the last of a bar, and past the last of the pattern.
        if along >= self.square_w or step >= self.beat_rows or within >= self.bar_rows or row >= self.rows:
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
        return self.x(row), self.y(drum), self.square_w, LANE_H

    def label(self, drum: int) -> Rect:
        """The box of the label of a drum's lane, beside its squares. What of it is in `names` is in view."""
        return self.rect[0], self.y(drum), LABELS_W - LABEL_GAP, LANE_H

    def label_at(self, pos: Pos) -> int | None:
        """The drum whose label a position is on, or None where it is on none."""
        if not inside(self.names, pos):
            return None
        return next((drum for drum in self.drums_in_view() if inside(self.label(drum), pos)), None)

    @property
    def plus(self) -> Rect:
        """The square of the plus that adds a drum: under the label of the last lane, or where the first would be.

        It has no width while it is not whole in view: the view has to be moved to its end to see it.
        """
        x, top, _, high = self.names
        under = self._down(self.drums - 1) + LANE_H + PLUS_GAP if self.drums else 0
        y = top + under - self._down(self.first_drum)
        whole = top <= y and y + theme.PLUS <= top + high
        return within((x, y, theme.PLUS, theme.PLUS), self.rect) if whole else (x, top, 0, 0)

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
        on = [row for row in self.rows_in_view() if self.x(row) <= high and low < self.x(row) + self.square_w]
        return range(on[0], on[-1] + 1) if on else range(0)

    def colour(self, row: int) -> Colour:
        """The colour a row's squares take: the first of the four in the first beat of a bar, and so on round."""
        return theme.STEPS[row % self.bar_rows // self.beat_rows % len(theme.STEPS)]

    def scrolled(self, rows: int = 0, drums: int = 0) -> Grid:
        """The same steps with the view moved by so many rows and so many drums, and kept inside the pattern.

        The furthest it goes is where the last row is whole in view, and the last drum with the room of the
        plus under it.
        """
        _, _, width, height = self.area
        over = self._along(self.rows - 1) + self.square_w - width
        last_row = next((row for row in range(self.rows) if self._along(row) >= over), max(0, self.rows - 1))
        under = self._down(self.drums - 1) + LANE_H + TAIL - height
        last_drum = next((drum for drum in range(self.drums) if self._down(drum) >= under), max(0, self.drums - 1))
        return replace(self, first_row=min(max(0, self.first_row + rows), last_row),
                       first_drum=min(max(0, self.first_drum + drums), last_drum))

    def spans(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """What the two scroll bars show, the one along the lanes first.

        Each is three lengths in pixels: all there is to go along, how much of it the squares' room has in
        view, and how much of it lies before the view. All there is are the lanes with the room of the plus
        under the last, and the rows of the pattern.
        """
        _, _, width, height = self.area
        lanes = (self._down(self.drums - 1) + LANE_H if self.drums else 0) + TAIL
        rows = self._along(self.rows - 1) + self.square_w if self.rows else 0
        return (lanes, height, self._down(self.first_drum)), (rows, width, self._along(self.first_row))

    def at(self, lanes: int | None = None, rows: int | None = None) -> Grid:
        """The same steps with the view where a scroll bar was moved to: so many pixels of the lanes, or of the rows.

        The view starts at a whole lane and at a whole row: the ones nearest to there, of those `scrolled` lets
        it start at. The last of them counts as where the bar has its end, so a bar at its end has the end of
        the pattern in view.
        """
        _, _, width, height = self.area
        (all_lanes, _, _), (all_rows, _, _) = self.spans()
        end = self.scrolled(self.rows, self.drums)
        first_drum, first_row = self.first_drum, self.first_row
        if lanes is not None:
            first_drum = min(range(end.first_drum + 1),
                             key=lambda drum: abs(min(self._down(drum), all_lanes - height) - lanes))
        if rows is not None:
            first_row = min(range(end.first_row + 1),
                            key=lambda row: abs(min(self._along(row), all_rows - width) - rows))
        return replace(self, first_drum=first_drum, first_row=first_row)

    def zoomed(self, square_w: int, x: int) -> Grid:
        """The same steps with squares of another width, no less than `SQUARE_W_MIN` and no more than `SQUARE_W_MAX`.

        The row at `x` stays at that x, as nearly as a view that starts at a whole row lets it.
        """
        reach = max(0, x - self.area[0])
        here = self._along(self.first_row) + reach
        row = max((row for row in range(self.rows) if self._along(row) <= here), default=0)
        fresh = replace(self, square_w=min(max(square_w, SQUARE_W_MIN), SQUARE_W_MAX))
        return fresh.at(rows=fresh._along(row) - reach)


@dataclass
class Gesture:
    """What the mouse is in the middle of doing to the hits of one lane, and what it comes to.

    kind is `PUT` for a drag that puts a hit in each empty square it passes, `TAKE` for a drag that takes away
    each hit it passes, and `MOVE` for a drag that takes one hit along the lane. drum is the place of the lane,
    and rows the rows of the squares the gesture has passed: of a move they are the row its hit came from, and
    after it the row the hit is at now. level is the level a put hit gets. button is the mouse button of the
    drag, and x where the mouse was last in it.
    """

    kind: str
    drum: int
    level: int
    rows: list[int] = field(default_factory=list)
    button: int = 0
    x: int = 0

    def along(self, grid: Grid, x: int) -> list[int]:
        """Take a drag on to an x. Given are the rows of the squares on the way that it had not passed before.

        A move passes none: its hit is at the square the mouse is on, or was on last.
        """
        if self.kind == MOVE:
            on = grid.passed(x, x)
            self.x = x
            self.rows[1:] = [on[0]] if on else self.rows[1:]
            return []
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
        the place of a hit of another letter. A take empties the squares that had a hit of the letter. A move
        takes the hit of the letter from the row it was pressed at to the row it is at now, at the level it
        has, and it takes the place of what the channel plays there.
        """
        notes = tuple(notes)
        held = {_whole_rows(note.row): note for note in notes}
        if self.kind == MOVE:
            there, landing = held.get(self.rows[0]), held.get(self.rows[-1])
            if self.rows[-1] == self.rows[0] or there is None or there.drum != letter:
                return notes
            notes = noteedit.unhit(notes, there.row, letter, drums, rows, rows_per_beat)
            return noteedit.hit(notes, float(self.rows[-1]) if landing is None else landing.row, letter,
                                there.level, drums, rows, rows_per_beat)
        for row in sorted(self.rows):
            there = held.get(row)
            mine = there is not None and there.drum == letter
            # A note that is a hair off its row is replaced, and taken away, at the place it has.
            at = float(row) if there is None else there.row
            if self.kind == PUT and not mine and (there is None or row == self.rows[0]):
                notes = noteedit.hit(notes, at, letter, self.level, drums, rows, rows_per_beat)
            elif self.kind == TAKE and mine:
                notes = noteedit.unhit(notes, at, letter, drums, rows, rows_per_beat)
        return notes

    def label(self, count: int) -> str:
        """A few plain words for the undo list, for a gesture that put or took away so many hits."""
        if self.kind == MOVE:
            return "hit moved"
        done = "placed" if self.kind == PUT else "removed"
        return f"hit {done}" if count == 1 else f"{count} hits {done}"


def begun(grid: Grid, hits: Container[tuple[int, int]], pos: Pos, button: int,
          level: int = TOP_LEVEL) -> Gesture | None:
    """The drag that a press of a mouse button begins at a position, and None where it begins none.

    `hits` has the squares with a hit in them, each as the place of its lane and its row. The left button puts
    hits when it is pressed on an empty square, at `level`, and moves the hit it is pressed on. The right button
    takes hits away wherever it is pressed. A press on no square begins nothing, and neither does any other
    button. A put and a take have passed no square yet: `along` takes them to where the press is.
    """
    row, drum = grid.row_at(pos[0]), grid.drum_at(pos[1])
    if row is None or drum is None or button not in (1, 3):
        return None
    if button == 1 and (drum, row) in hits:
        return Gesture(MOVE, drum, level, [row], button, pos[0])
    return Gesture(TAKE if button == 3 else PUT, drum, level, button=button, x=pos[0])


class Steps(Widget):
    """The drums of one pattern, from all its drum channels together: a lane of squares for each `Lane`, a square a row.

    The lanes of one channel stand together, with more air and a faint line before those of the next. Each lane
    has its name at the side: a label with the letter of its drum and the instrument the letter plays. `picked`
    is the name of a channel, or None: the labels of its lanes are lit. `playhead` is the row that is playing,
    which may be between two rows, or None. Their owner sets both.

    The steps have four parts that are widgets of their own, the parts of `kit`: `along` is the scroll bar down
    the right, which goes along the lanes, `across` the one under the squares, which goes along the rows, `lane`
    the velocity lane, and `plus` the button under the label of the last lane, which adds a drum: a press on it
    calls `added`. `parts` has the four. Their owner adds them to the Ui after the steps, says with `seen`
    whether they are on the window, and puts the lane where it stands, at the foot of the editor. The wheel is
    read by `kit.wheel`: it goes along the lanes, with shift along the rows by a beat, and with ctrl it zooms
    the rows round the mouse.

    The hits are changed with the mouse, where `show` was given the drum table and the pattern is on the grid.
    The left button on an empty square puts a hit at `level`, and dragged it puts one in every empty square it
    passes. On a hit it lets the hit be heard, and dragged it moves the hit along its lane. The right button
    rubs a hit out, and dragged every hit it passes. A drag stays in the lane it began in.

    The velocity lane has a bar for each hit of one lane: `chosen` is the place of that lane, which is the one
    that was pressed in last, on a square or on its label. Its label has a strip along its foot. A drag in the
    velocity lane is one edit, made when the button is let go, and `level` is the level that was set there last.
    A press of the right button on a label calls `asked` with the place of its lane and the position, for the
    owner to open the menu of that drum at: `lanes` has the lane at each place.

    A channel plays one drum at a time. So a hit that is put by a press, or moved, takes the place of a hit that
    another letter of its channel has at that row, and a drag puts none in the squares it passes after that where
    another letter has one. Such a square is drawn empty with a dot in it, where the hits can be changed.

    Nothing is saved here. While a gesture is under way `busy` is true, and the steps show what it would give.
    When it ends, and a channel's notes are other than `show` gave them, `edited` is called with the name of the
    channel, its new notes and a few words for the undo list, and the steps show what `show` gave again: their
    owner saves, and shows the result. `sounded` is called with the channel and the letter of each hit as it is
    put, and of a hit that is pressed, for it to be heard. That is before the gesture has ended, so the owner's
    answer to `edited`, whether it saved, is not waited for. Escape drops a gesture that is under way. Any other
    key the steps are handed, which they are while they or one of their parts have the focus, goes on to `typed`.

    `help` says what the left press, the right press and the wheel do on the square or the label under the
    mouse. Off those it is the widget's own line, and on the squares of a pattern off the grid it says why
    nothing is changed.
    """

    def __init__(self, rect: Rect, help: str = "", edited: Callable[[str, Notes, str], object] | None = None,
                 sounded: Callable[[str, str], object] | None = None,
                 typed: Callable[[Event], object] | None = None,
                 asked: Callable[[int, Pos], object] | None = None,
                 added: Callable[[], object] | None = None) -> None:
        self.playhead: float | None = None
        self.edited = edited
        self.sounded = sounded
        self.typed = typed
        self.asked = asked
        self.level = TOP_LEVEL
        self._picked: str | None = None
        self._help = help
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
        #: The levels a drag in the velocity lane has given so far, by the row of the hit, the last one last.
        self._levels: dict[int, int] = {}
        #: The level of each hit that starts on a whole row, by the place of its lane and its row.
        self._hits: dict[tuple[int, int], int] = {}
        #: The place of the lane, the row and the level of each hit that starts between two rows.
        self._marks: list[tuple[int, float, int]] = []
        #: The empty squares whose row another drum of their channel has, each with the letter of that drum.
        self._taken: dict[tuple[int, int], str] = {}
        self._mouse: Pos = NOWHERE
        #: The channel and the letter of the lane that was pressed in last. None before any was.
        self._last: tuple[str, str] | None = None
        #: The row of the hit of each bar of the velocity lane, in the order the lane has its bars.
        self._bars: list[int] = []
        #: Whether the steps are on the window. The plus is seen with them where it has its room.
        self._on = True
        nowhere = (0, 0, 0, 0)
        self.along = ScrollBar(nowhere, self._go_along, ALONG_HELP)
        self.across = ScrollBar(nowhere, self._go_across, ACROSS_HELP)
        self.lane = Velocity(nowhere, self._weigh, self._weighed)
        self.plus = Plus(nowhere, added, PLUS_HELP)
        for part in (self.along, self.across, self.lane):
            part.typed = self._keyed
        super().__init__(rect, help)

    @property
    def parts(self) -> tuple[Widget, ...]:
        """The four parts that are widgets of their own: the two scroll bars, the velocity lane and the plus."""
        return self.along, self.across, self.lane, self.plus

    @property
    def rect(self) -> Rect:
        """The rectangle of the steps: the labels, the bar numbers, the squares and the two scroll bars are in it."""
        return self._rect

    @rect.setter
    def rect(self, rect: Rect) -> None:
        self._rect = rect
        self._sync()

    @property
    def picked(self) -> str | None:
        """The name of the picked channel, whose labels are lit, or None."""
        return self._picked

    @picked.setter
    def picked(self, channel: str | None) -> None:
        self._picked = channel
        self._sync()

    @property
    def lanes(self) -> tuple[Lane, ...]:
        """The lanes that are shown, from the top."""
        return self._lanes

    @property
    def chosen(self) -> int | None:
        """The place of the lane whose hits the velocity lane has: the one that was pressed in last.

        Before any was, and where the lanes no longer have that one, it is the first lane of the picked
        channel, or the first of all. None while there is no lane.
        """
        keys = [(lane.channel, lane.letter) for lane in self._lanes]
        if self._last in keys:
            return keys.index(self._last)
        return next((place for place, key in enumerate(keys) if key[0] == self._picked), 0 if keys else None)

    def seen(self, on: bool) -> None:
        """Bring the steps onto the window with their parts, or take them off. The plus is seen where it has room."""
        self._on = self.visible = on
        for part in (self.along, self.across, self.lane):
            part.visible = on
        self._sync()

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
                          self._grid.first_row, self._grid.first_drum, breaks, self._grid.square_w)
        self._lanes = lanes
        self._notes = {channel: held[channel].notes for channel in dict.fromkeys(lane.channel for lane in lanes)}
        self._rows = pattern.rows
        self._several = len(self._notes) > 1
        on_grid = songview.on_grid(pattern)
        self._drums, self._open = drums, drums is not None and on_grid
        self._why = OFF_GRID if drums is not None and not on_grid else ""
        self._drop()
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
        """Whether a gesture is under way, on the squares or in the velocity lane: from its press to its release."""
        return self._gesture is not None or self.lane.busy or bool(self._levels)

    def _drop(self) -> None:
        """Let go of the gesture that is under way, on the squares or in the lane, with nothing told of it."""
        self._gesture, self._given, self._levels = None, {}, {}
        self.lane.drop()

    def _look(self) -> None:
        """Work out which squares have a hit, the hits between two rows, and the squares whose row is taken.

        It is worked out from the notes as they are shown. A row is taken in the other lanes of a channel by
        what the channel plays there, which is a hit of another letter, or a note that is no hit. The parts
        are then shown what the steps show.
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
        self._sync()

    def _sync(self) -> None:
        """Have the parts show what the steps show now: the view, the hits of the chosen lane, and the playhead.

        The two scroll bars are laid out in the rectangle, and the plus under the last lane. The velocity lane
        is told the stretch its bars stand in, which is as wide as the squares' room, and a bar for each hit of
        the chosen lane, under its square. Where the hits cannot be changed the lane takes nothing from the
        mouse, and on a pattern off the grid it says why.
        """
        grid = self.grid
        self.along.rect, self.across.rect = grid.along, grid.across
        lanes, rows = grid.spans()
        self.along.show(*lanes)
        self.across.show(*rows)
        left, _, wide, _ = grid.area
        place = self.chosen
        self._bars = sorted(row for drum, row in self._hits if drum == place)
        self.lane.span = (left, wide)
        self.lane.show([(grid.x(row), grid.square_w, self._hits[place, row] / TOP_LEVEL, False)
                        for row in self._bars])
        self.lane.playhead = None if self.playhead is None else grid.x(self.playhead)
        self.lane.enabled = self._open
        self.lane.help = self._why or VELOCITY_HELP
        self.plus.rect = grid.plus
        self.plus.visible = self._on and self.plus.rect[2] > 0

    @property
    def grid(self) -> Grid:
        """The sums for the rectangle as it is now, the view kept inside the pattern."""
        return replace(self._grid, rect=self.rect).scrolled()

    @property
    def help(self) -> str:
        """What the left press, the right press and the wheel do where the mouse is, as `kit.told` words it.

        On a label it is what they do to its lane, and on a square what they do there: on a square whose row
        another drum of the channel has, the hit would take its place. Where the pattern is off the grid it
        says that on every square. Anywhere else, and where no drum table was given, it is the widget's own
        line. The drum is told by its letter, and where the lanes are of more than one channel by its channel
        as well.
        """
        grid = self.grid
        label, found = grid.label_at(self._mouse), self._at(self._mouse)
        if label is not None:
            return told(f"show the velocity of {self._drum(label)}", "its menu" if self.asked else "", WHEEL_HELP)
        if found is None or not self._open:
            return self._why if found is not None and self._why else self._help
        lane, row = self._lanes[found[0]], found[1]
        bar, step = divmod(row, grid.bar_rows)
        place = f"bar {bar + 1}, step {step + 1}"
        if found in self._hits:
            return told(f"pick {lane.letter} at {place}, a drag moves it", "rub it out", WHEEL_HELP)
        other = self._taken.get(found)
        why = "" if other is None else f", in place of {other or 'a note'}"
        return told(f"put {self._drum(found[0])} at {place}{why}", wheel=WHEEL_HELP)

    @help.setter
    def help(self, text: str) -> None:
        self._help = text

    def _drum(self, place: int) -> str:
        """The drum of the lane at a place as the help line names it: its letter, and with several channels its own."""
        lane = self._lanes[place]
        return f"{lane.letter} in {lane.channel}" if self._several else lane.letter

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

    def name_rect(self, place: int) -> Rect:
        """Where the letter of the lane at a place is written in its label, as the view has it now.

        It is for a box that is typed into to be put over.
        """
        x, y, w, h = self.grid.label(place)
        return x + LABEL_PAD, y + LABEL_EDGE, w - 2 * LABEL_PAD, (h - 2 * LABEL_EDGE) // 2

    def on(self, event: Event, ui: Ui) -> bool:
        """Follow the mouse. Begin a drag at a press on a square, take it along, and end it at the release.

        The wheel goes along the lanes, with shift along the rows by a beat, and with ctrl it zooms: it is used
        when it changed the view. A press is used when it began a drag, and so is all that the mouse does up to
        the release. So is a press on a label that picked its lane, or asked for its menu. A key goes to
        `_keyed`.
        """
        if event.kind == "key":
            return self._keyed(event)
        self._mouse = event.pos
        doing = self._gesture
        if doing is not None:
            if event.kind == "move":
                self._drag(event.pos[0])
            elif event.kind == "up" and event.button == doing.button:
                self._end()
            return True
        if event.kind == "wheel":
            return self._turn(event)
        # A press while another button is held begins nothing.
        return event.kind == "down" and ui.held is None and self._press(event)

    def _keyed(self, event: Event) -> bool:
        """Take a key of the computer: escape drops a gesture that is under way, and any other goes to `typed`."""
        if event.key == "escape" and self.busy:
            self._drop()
            self._look()
            return True
        return self.typed is not None and bool(self.typed(event))

    def _choose(self, place: int) -> None:
        """Make the lane at a place the one whose hits the velocity lane has."""
        self._last = (self._lanes[place].channel, self._lanes[place].letter)
        self._sync()

    def _press(self, event: Event) -> bool:
        """Take a press: on a label it is that lane's, and on a square it begins the drag that `begun` gives.

        The left button on a label picks its lane, and the right one asks the owner for its menu. A press on
        a square picks the lane of the square as well, and a hit that is pressed with the left button is told
        of, to be heard. The answer is whether the press was used.
        """
        grid, pos = self.grid, event.pos
        label = grid.label_at(pos)
        if label is not None:
            if event.button == 1:
                self._choose(label)
            elif event.button == 3 and self.asked is not None:
                self.asked(label, pos)
            return event.button == 1 or (event.button == 3 and self.asked is not None)
        if not self._open:
            return False
        self._gesture = begun(grid, self._hits, pos, event.button, self.level)
        if self._gesture is None:
            return False
        self._choose(self._gesture.drum)
        lane = self._lanes[self._gesture.drum]
        if self._gesture.kind == MOVE and self.sounded is not None:
            self.sounded(lane.channel, lane.letter)
        self._drag(pos[0])
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
        """Take a step of the wheel to the view: along the lanes, with shift along the rows, with ctrl the zoom.

        The answer is whether the view is another than before.
        """
        grid = self.grid
        way, steps = wheel(event)
        if way == ZOOM:
            self._grid = grid.zoomed(grid.square_w + steps, event.pos[0])
        elif way == ACROSS:
            self._grid = grid.scrolled(rows=-steps * grid.beat_rows)
        else:
            self._grid = grid.scrolled(drums=-steps)
        self._sync()
        return self._grid != grid

    def _go_along(self, offset: int) -> None:
        """Take the view to where the scroll bar along the lanes was moved: so many pixels down the lanes."""
        self._grid = self.grid.at(lanes=offset)
        self._sync()

    def _go_across(self, offset: int) -> None:
        """Take the view to where the scroll bar along the rows was moved: so many pixels into the rows."""
        self._grid = self.grid.at(rows=offset)
        self._sync()

    def _weigh(self, index: int, value: float | None) -> None:
        """Take the velocity the lane gives a hit of the chosen lane, from 0 to 1: it is kept until the lane settles.

        None is for the usual velocity, which is the top level. The steps show the hit at its new level.
        """
        if not self._open or self._gesture is not None or index >= len(self._bars):
            return
        lane = self._lanes[self.chosen]
        self._levels.pop(self._bars[index], None)
        self._levels[self._bars[index]] = (TOP_LEVEL if value is None
                                           else min(max(_px(value * TOP_LEVEL), 0), TOP_LEVEL))
        notes = self._notes[lane.channel]
        for row, level in self._levels.items():
            found = [at for at, note in enumerate(notes) if note.drum == lane.letter and _whole_rows(note.row) == row]
            notes, _ = noteedit.with_level(notes, found, level)
        self._given = {lane.channel: notes}
        self._look()

    def _weighed(self) -> None:
        """End what the velocity lane did: the notes with their new levels are handed on as one edit, if any is another.

        The level that was set last is kept for the next hit that is put.
        """
        if self._gesture is not None:
            return
        given, levels, before = self._given, self._levels, self._notes
        self._given, self._levels = {}, {}
        if levels:
            self.level = list(levels.values())[-1]
        self._look()
        for channel, notes in given.items():
            if notes != before[channel] and self.edited is not None:
                self.edited(channel, notes, VELOCITY_DONE)

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

        Where the hits can be changed, a dot is in each empty square whose row another drum of its channel has,
        and the square under the mouse is lit. The parts draw themselves, after the steps: they are shown here
        what the steps show in this frame. Nothing is drawn outside the rectangle, and no square where the
        labels are.
        """
        self._sync()
        grid = self.grid
        rows, drums = grid.rows_in_view(), grid.drums_in_view()
        x, y, _, _ = self.rect
        left, top, width, height = grid.area
        chosen = self.chosen
        over = self._at(self._mouse) if self.hot and self._open else None
        named = grid.label_at(self._mouse) if self.hot else None
        playing = self.playhead if self.playhead is not None and 0 <= self.playhead < grid.rows else None
        with canvas.clip((x, y, LABELS_W + width, BARS_H + height)):
            for drum in drums:
                # The line is in the middle of the air above the first lane of a channel, where that air is in view.
                if drum in grid.breaks and drum > drums.start:
                    canvas.hline(x, grid.y(drum) - (LANE_GAP + CHANNEL_GAP + 1) // 2, LABELS_W + width, theme.LIT)
            with canvas.clip(grid.names):
                for drum in drums:
                    self._label(canvas, grid.label(drum), self._lanes[drum], drum == chosen, drum == named)
            with canvas.clip((left, y, width, BARS_H + height)):
                for bar in range(rows.start // grid.bar_rows, -(-rows.stop // grid.bar_rows)):
                    start = bar * grid.bar_rows
                    bar_w = grid.x(start + grid.bar_rows) - BAR_GAP - grid.x(start)
                    canvas.text_in(f"BAR {bar + 1}", (grid.x(start), y, bar_w, BARS_H), theme.DIM)
                for drum in drums:
                    for row in rows:
                        lit = (playing is not None and math.floor(playing) == row) or (drum, row) == over
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

    def _label(self, canvas: Canvas, box: Rect, lane: Lane, chosen: bool = False, hot: bool = False) -> None:
        """Draw the label of a lane in its box: its letter, and under it the name of the instrument it plays.

        Where the lanes are of more than one channel, the name of its channel stands at the right of the letter.
        The box of a lane of the picked channel is lit. The lane the velocity lane shows has a strip along its
        foot, inside its outline, in the colour of the bars there. The label under the mouse has a lighter
        outline.
        """
        x, y, w, h = box
        canvas.box(box, theme.LIT if lane.channel == self._picked else theme.RAISED)
        canvas.frame(box, theme.DIM if hot else theme.LINE)
        if chosen:
            canvas.box((x + 1, y + h - 1 - theme.STRIP, w - 2, theme.STRIP), self.lane.colour)
        line_h = (h - 2 * LABEL_EDGE) // 2
        left, top, wide = x + LABEL_PAD, y + LABEL_EDGE, w - 2 * LABEL_PAD
        canvas.text_in(lane.letter, (left, top, wide, line_h), theme.TEXT)
        if self._several:
            taken = Font.width(lane.letter) + LABEL_PAD
            canvas.text_in(lane.channel, (left + taken, top, max(0, wide - taken), line_h), theme.DIM, "right")
        canvas.text_in(lane.inst, (left, top + line_h, wide, line_h), theme.INSTRUMENT)
