"""The piano roll: one channel of one pattern as bars on a grid of rows and keys, its sums, and its gestures."""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from ..errors import RowSynthError
from ..song import _NAMES
from . import chordview, noteedit, songview, theme
from .chordview import Chords
from .font import Font
from .layout import Rect
from .noteedit import Picked
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
#: How high the strip is in which the chords of the bars are named, between the ruler and the grid, a line of air
#: under the names counted. And how far a name stands from the start of its bar.
CHORDS_H = 13
CHORD_PAD = 4
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
#: How far the row of a key goes towards the colour of the text across a bar whose chord has the note of the key.
CHORD_LIT = 0.1
#: How many pixels at the right end of the bar of a note are for stretching it. A press further left moves it.
END_W = 3
#: How many pixels the mouse may stray from where it was pressed, to any side, and not have dragged: a click wobbles.
SLACK = 2
#: How many rows long the note is that a click places, until one has been placed by a drag or stretched.
LENGTH = 4.0

#: What a press can begin, as `begins` calls it.
PLACE, MOVE, STRETCH, PICK, BOX, REMOVE, LEVEL = "place", "move", "stretch", "pick", "box", "remove", "level"
#: What the help line says anywhere on a pattern that has a note the rows cannot say.
OFF_GRID = "this pattern has lengths the grid does not hold: it is shown and not changed"

# What was done to the notes by the gestures that are listed with a count of notes.
_DONE = {PLACE: "placed", MOVE: "moved", STRETCH: "stretched", REMOVE: "removed"}
_NO_MODS: frozenset[str] = frozenset()
_CTRL = frozenset({"ctrl"})


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


def counted(count: int, done: str) -> str:
    """The few words that what was done to so many notes is listed with: `note moved`, `2 notes moved`."""
    return f"note {done}" if count == 1 else f"{count} notes {done}"


def path(start: Pos, end: Pos) -> list[Pos]:
    """The pixels on the straight line from a position to another, the first left out: what the mouse passed."""
    wide, high = end[0] - start[0], end[1] - start[1]
    steps = max(abs(wide), abs(high))
    return [(start[0] + _px(wide * step / steps), start[1] + _px(high * step / steps))
            for step in range(1, steps + 1)]


def _sounding(notes: Sequence[Note], row: float) -> int | None:
    """Which of some notes sounds at a row: its index among them, or None where none does."""
    return next((index for index, note in enumerate(notes) if note.row <= row < note.row + note.rows), None)


@dataclass(frozen=True)
class Map:
    """Where the rows and the pitches of a pattern fall in the rectangle of a roll, and back.

    rect is the roll's rectangle. row is the row at the left edge of its grid, top the pitch of the key at the
    upper edge of it, and row_w how wide a row is. row and top can be fractions: then part of a row, or of a key,
    is cut off at that edge. strip is how high the strip of the chords is, between the ruler and the grid: 0 for
    a roll that shows no chords, which has no such strip.
    """

    rect: Rect
    row: float = 0.0
    top: float = 72.0
    row_w: int = ROW_W
    strip: int = 0

    @property
    def grid(self) -> Rect:
        """Where the notes are: what the keys, the ruler, the strip of the chords and the lane leave of it."""
        x, y, w, h = self.rect
        head = RULER_H + self.strip
        return x + KEYS_W, y + head, max(0, w - KEYS_W), max(0, h - head - GAP - LANE_H)

    @property
    def keys(self) -> Rect:
        """The strip of keys, down the left of the grid."""
        _, y, _, h = self.grid
        return self.rect[0], y, KEYS_W, h

    @property
    def ruler(self) -> Rect:
        """The strip along the top in which the bars are numbered, as wide as the grid."""
        x, _, w, _ = self.grid
        return x, self.rect[1], w, RULER_H

    @property
    def chords(self) -> Rect:
        """The strip in which the chords of the bars are named, under the ruler and as wide. It is `strip` high."""
        x, y, w, _ = self.ruler
        return x, y + RULER_H, w, self.strip

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

    def level_at(self, y: int) -> int:
        """The level whose bar in the lane reaches up to a y and no higher: 0 at its foot, and `LEVELS` at its top."""
        _, top, _, h = self.lane
        return min(max(math.ceil((top + h - 2 - y) * LEVELS / (h - 4)), 0), LEVELS)

    def note_at(self, notes: Sequence[Note], pos: Pos) -> int | None:
        """Which of some notes has its bar under a position on the grid: its index among them, or None for none."""
        if inside(self.grid, pos):
            # The last is drawn on top of the others.
            for index in reversed(range(len(notes))):
                if inside(self.note(notes[index].row, notes[index].rows, notes[index].pitch), pos):
                    return index
        return None

    def moved(self, start: Pos, pos: Pos, grid: float = 1.0) -> tuple[float, int]:
        """How many rows and how many pitches a drag from a position to another has moved by.

        The rows are from the step of a grid that the drag began in to the step it is in now: whole steps of it.
        """
        return (noteedit.snapped(self.row_at(pos[0]), grid) - noteedit.snapped(self.row_at(start[0]), grid),
                self.pitch_at(pos[1]) - self.pitch_at(start[1]))

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


def begins(seen: Map, notes: Sequence[Note], rows: float, pos: Pos, button: int = 1,
           mods: frozenset[str] = _NO_MODS) -> tuple[str, int | None]:
    """Which gesture a press at a position begins among some notes, in a pattern of `rows` rows, and on which note.

    The left button in the lane sets levels: the note is the one that sounds at that row. On the grid the right
    button removes. The left one on a note picks it with ctrl, and else moves it, or stretches it on the last
    `END_W` pixels of its bar: a bar narrower than twice that is stretched at its last pixel only. Where no note
    is, it draws a box with shift, and else places a note where the row is one of the pattern's and the key one of
    the roll's. Any other press begins nothing: "" and None.
    """
    if button == 1 and inside(seen.lane, pos):
        return LEVEL, _sounding(notes, seen.row_at(pos[0]))
    if button not in (1, 3) or not inside(seen.grid, pos):
        return "", None
    index = seen.note_at(notes, pos)
    if button == 3:
        return REMOVE, index
    if index is None:
        if "shift" in mods:
            return BOX, None
        free = 0 <= seen.row_at(pos[0]) < rows and LOWEST <= seen.pitch_at(pos[1]) <= HIGHEST
        return (PLACE if free else ""), None
    if "ctrl" in mods:
        return PICK, index
    x, _, w, _ = seen.note(notes[index].row, notes[index].rows, notes[index].pitch)
    return (STRETCH if pos[0] >= x + w - (END_W if w >= 2 * END_W else 1) else MOVE), index


@dataclass
class Gesture:
    """A gesture that is under way: what a press began, among which notes, and where the mouse has been since.

    kind is what `begins` called it, button the button that was pressed, and index the note it was pressed on, or
    None. seen, notes, rows and grid are the sums of the roll, its notes, the rows of its pattern and the rows in
    a step of its grid, all as they were at the press. picked is the picked notes as the press left them. new is
    the note that a click here places. start is where the press was, and help what the help line said it does.
    pos is where the mouse is now, and dragged whether it has been more than `SLACK` pixels from the press: until
    then nothing is moved, stretched, made longer or boxed. passed has the notes it passed: for a removal those
    under it, and for the levels those at its rows, each with the level its height gave last. heard is the pitch
    at which the note under the press was heard last.
    """

    kind: str
    button: int
    index: int | None
    seen: Map
    notes: tuple[Note, ...]
    rows: float
    grid: float
    picked: frozenset[int]
    new: Note
    start: Pos
    help: str = ""
    pos: Pos = field(init=False)
    dragged: bool = field(default=False, init=False)
    passed: dict[int, int] = field(default_factory=dict, init=False)
    heard: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.pos = self.start
        self.heard = self.notes[self.index].pitch if self.index is not None else None
        self._pass(self.start)

    def to(self, pos: Pos) -> None:
        """Take the mouse to another position, by way of every pixel between this one and that."""
        for step in path(self.pos, pos):
            self._pass(step)
        self.pos = pos
        self.dragged = self.dragged or max(abs(pos[0] - self.start[0]), abs(pos[1] - self.start[1])) > SLACK

    def _pass(self, pos: Pos) -> None:
        """Take note of what the mouse passes at a position: a note to remove, or a note and a level for it."""
        left, _, wide, _ = self.seen.lane
        index = None
        if self.kind == REMOVE:
            index = self.seen.note_at(self.notes, pos)
        elif self.kind == LEVEL and left <= pos[0] < left + wide:
            index = _sounding(self.notes, self.seen.row_at(pos[0]))
        if index is not None:
            # The note that was passed last is the last of them.
            self.passed.pop(index, None)
            self.passed[index] = self.seen.level_at(pos[1])

    @property
    def box(self) -> Rect | None:
        """The box that is being drawn: from the pixel of the press to the pixel the mouse is on. None for no box."""
        if self.kind != BOX or not self.dragged:
            return None
        (x, y), (now_x, now_y) = self.start, self.pos
        return min(x, now_x), min(y, now_y), abs(now_x - x) + 1, abs(now_y - y) + 1

    def gives(self) -> Picked:
        """What a release where the mouse is now gives: the notes, and the indexes among them of the picked ones.

        Each is what the rule of `noteedit` makes of it. A move or a stretch that has not dragged, or is back in
        the step and on the key it began in, gives the notes there were.
        """
        seen, notes = self.seen, self.notes
        by_rows, by_pitch = seen.moved(self.start, self.pos, self.grid) if self.dragged else (0.0, 0)
        if self.kind == PLACE:
            # Dragged, the note ends with the step of the grid the mouse is in, and is no shorter than a step.
            rows = max(by_rows, 0.0) + self.grid if self.dragged else self.new.rows
            return noteedit.place(notes, replace(self.new, rows=rows), self.rows), frozenset()
        if self.kind == MOVE and (by_rows or by_pitch):
            return noteedit.move(notes, self.picked, by_rows, by_pitch, self.rows, LOWEST, HIGHEST)
        if self.kind == STRETCH and by_rows:
            return noteedit.stretch(notes, self.picked, by_rows, self.rows, self.grid)
        if self.kind == BOX:
            if not self.dragged:
                return notes, frozenset()
            return notes, noteedit.boxed(notes, seen.row_at(self.start[0]), seen.row_at(self.pos[0]),
                                         seen.pitch_at(self.start[1]), seen.pitch_at(self.pos[1]))
        if self.kind == REMOVE and self.passed:
            return noteedit.remove(notes, self.passed)
        if self.kind == LEVEL:
            for index, level in self.passed.items():
                notes, _ = noteedit.with_level(notes, {index}, level)
        return notes, self.picked

    def grabbed(self) -> Note:
        """The note under the press of a move or a stretch, as a release where the mouse is now leaves it."""
        notes, picked = self.gives()
        # The picked notes keep their order, so it has the place among them that it had.
        return notes[sorted(picked)[sorted(self.picked).index(self.index)]]

    @property
    def label(self) -> str:
        """The few words for what the gesture did to the notes, as a list of what can be undone has them."""
        if self.kind == LEVEL:
            return "level changed"
        count = {PLACE: 1, REMOVE: len(self.passed)}.get(self.kind, len(self.picked))
        return counted(count, _DONE[self.kind])


class Roll(Widget):
    """The piano roll: the notes of one channel in one pattern as bars, between a strip of keys and a lane of levels.

    The mouse and the keys place, move, stretch and remove its notes, and it saves nothing. When a gesture ends
    and the notes are others than they were, `edited` is called with the channel's new notes and a few words for
    the list of what can be undone, and the roll goes on showing what `show` gave it last: its owner saves, and
    shows it the result. An owner that answers false has not saved: any other answer, and none, is taken for
    saved. `sounded` is called with a pitch and an instrument for the owner to let be heard: when a note is
    placed and saved, each time the note that the mouse moves comes to another pitch, and when the left button
    is pressed on a key of the strip, which changes nothing and picks nothing. `grid` is the rows in a
    step of the grid, a whole number of them: a placed note begins on a step, and notes are moved and stretched
    by whole steps. `picked` is the picked notes, and `drawn` the notes as a gesture that is under way would
    leave them: `busy` is whether one is under way. A key that the Ui hands the roll, which it does while the
    roll has the focus, goes to `key`. One that is not the roll's is handed on to `typed`, which says whether
    the owner used it.

    `chords` is the chords of the pattern as `chordview.read` gives them, or None for a roll that shows none. With
    them a strip under the ruler names the chord of each bar, and across a bar the rows of the notes of its
    chord are lit, in every octave. A note that the chord check would find is outlined in the colour of a fault,
    and so is one that a gesture under way would leave so: the help line on such a note says why. A press on a
    bar of the strip calls `chord_asked` with the bar, counted from 0, and the position, for the owner to open
    a menu at: the roll changes no chord.

    `ghosts` is whether the notes of the other channels that are not drums are drawn dim behind. `playhead` is the
    row that is playing, which can be a fraction, or None: the owner sets it each frame. `row`, `top` and `row_w`
    are the view, as a `Map` has them. `help`, while the mouse is where a press does something, says what; anywhere
    else it is the line the roll was made with. A pattern that is not on the grid is shown and not changed.
    """

    def __init__(self, rect: Rect, help: str = "", edited: Callable[[tuple[Note, ...], str], object] | None = None,
                 sounded: Callable[[float, str], object] | None = None,
                 typed: Callable[[Event], object] | None = None,
                 chord_asked: Callable[[int, Pos], object] | None = None) -> None:
        super().__init__(rect, help)
        self.edited = edited
        self.sounded = sounded
        self.typed = typed
        self.chord_asked = chord_asked
        self.chords: Chords | None = None
        self.grid = 1.0
        self.pattern: Pattern | None = None
        self.channel = ""
        self.bar_rows = 16
        self.colour: Colour = theme.SONG
        self.ghosts = True
        self.playhead: float | None = None
        self.row, self.top, self.row_w = 0.0, 72.0, ROW_W
        self._own: Channel | None = None
        self._mouse: Pos = NOWHERE
        self._mods = _NO_MODS
        # Whether the pattern is on the grid, so that its notes can be changed.
        self._open = False
        self._picked: frozenset[int] = frozenset()
        self._gesture: Gesture | None = None
        # What the last edit gave, until other notes are shown: its notes, and the picked ones among them.
        self._gave: Picked | None = None
        self._clip: tuple[Note, ...] = ()
        # How long the last note was that was placed by a drag or stretched, and the last level that was set.
        self._length = LENGTH
        self._level = LEVELS

    @property
    def help(self) -> str:
        """What a press where the mouse is would do, in plain words, or the line the roll was made with.

        While a gesture is under way it is what the press did that began it. On a note that is outlined as a
        wrong one it is why the note is wrong.
        """
        if self._gesture is not None:
            return self._gesture.help or self._help
        return self._why(self._mouse) or self._tell(self._mouse, self._mods) or self._help

    @help.setter
    def help(self, line: str) -> None:
        self._help = line

    @property
    def map(self) -> Map:
        """The sums of the roll as it is now."""
        return Map(self.rect, self.row, self.top, self.row_w, 0 if self.chords is None else CHORDS_H)

    @property
    def notes(self) -> tuple[Note, ...]:
        """The notes of the channel that is shown, in the order they are written in."""
        return self._own.notes if self._own else ()

    @property
    def picked(self) -> frozenset[int]:
        """Which of the notes are picked: their indexes among `notes`."""
        return self._picked

    @property
    def drawn(self) -> tuple[Note, ...]:
        """The notes as they are drawn: while a gesture is under way, what a release where the mouse is gives."""
        return self._now()[0]

    @property
    def busy(self) -> bool:
        """Whether a gesture is under way: from the press that began it to its release, or until it is dropped."""
        return self._gesture is not None

    def _now(self) -> Picked:
        """The notes to draw and the picked ones among them: what the gesture under way gives, or what there is."""
        return self._gesture.gives() if self._gesture is not None else (self.notes, self._picked)

    def _rows(self) -> float:
        """The rows of the pattern that is shown."""
        return self.pattern.rows if self.pattern else 0.0

    def _see(self, seen: Map) -> None:
        """Take the view of a map, brought within its limits on the pattern that is shown."""
        seen = seen.within(self._rows(), self.bar_rows)
        self.row, self.top, self.row_w = seen.row, seen.top, seen.row_w

    def show(self, pattern: Pattern, channel: str, bar_rows: int, colour: Colour,
             chords: Chords | None = None) -> None:
        """Show one channel of a pattern, with `bar_rows` rows to the bar, its notes in `colour`.

        `chords` is the chords of the pattern, for the strip that names them. With none given the roll has no
        such strip, lights no rows and outlines no note as a wrong one.

        The view is fitted when the pattern or the channel is another than before, by their names. When the same
        ones are shown again, with other notes, the view stays. A channel the pattern does not have is refused,
        and so is a bar that is not four beats of whole rows.

        What is picked stays picked while the notes are the ones there were. The notes that the last edit gave
        are shown with what that edit left picked. With any other notes nothing is picked, and a gesture that is
        under way is dropped.
        """
        own = next((found for found in pattern.channels if found.name == channel), None)
        if own is None:
            raise RowSynthError(f"pattern {pattern.name!r} has no channel called {channel!r}")
        if bar_rows < 4 or bar_rows % 4:
            raise RowSynthError(f"a bar is four beats of whole rows, which {bar_rows} rows are not")
        another = self.pattern is None or (self.pattern.name, self.channel) != (pattern.name, channel)
        same = not another and own.notes == self.notes
        if not same or pattern.rows != self._rows():
            self._gesture = None
        if not same:
            gave = None if another else self._gave
            self._picked = gave[1] if gave is not None and gave[0] == own.notes else frozenset()
            self._gave = None
        self.pattern, self.channel, self.bar_rows, self.colour, self._own = pattern, channel, bar_rows, colour, own
        self.chords = chords
        self._open = songview.on_grid(pattern)
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
        return self.map.note_at(self.notes, pos)

    def _faults(self, notes: Sequence[Note]) -> list[tuple[int, str, float] | None]:
        """What the chord check would find on each of some notes of the channel, as `chordview.wrong` gives it.

        None for a note it passes. The notes that are shown are held to the chords where the check looks at
        their channel. Other notes, which a gesture under way would leave, are held to them whatever the channel
        was: saved, its text fills the pattern, and the check looks at it then. With no chords shown nothing is
        found.
        """
        chords = self.chords
        if chords is None or self._own is None:
            return [None] * len(notes)
        saved = self._gesture is not None and tuple(notes) != self.notes
        looked = not self._own.drums if saved else chordview.looked(self.pattern, self._own)
        return [chordview.wrong(chords, note) if looked else None for note in notes]

    def _why(self, pos: Pos) -> str:
        """Why the note under a position is outlined as a wrong one, in plain words. No words where it is not."""
        index = self.note_at(pos)
        if index is None:
            return ""
        found = self._faults([self.notes[index]])[0]
        return "" if found is None else f"{name(round(self.notes[index].pitch))} is not in {found[1]}"

    def _bar_at(self, pos: Pos) -> int | None:
        """The bar whose place in the strip of the chords a position is on, counted from 0, or None for none."""
        seen = self.map
        if self.chords is None or not inside(seen.chords, pos):
            return None
        bar = math.floor(seen.row_at(pos[0]) / self.bar_rows)
        return bar if 0 <= bar < len(self.chords.bars) else None

    def _tell(self, pos: Pos, mods: frozenset[str] = _NO_MODS, button: int = 1) -> str:
        """What a press at a position would do, in plain words. No words where it would do nothing."""
        if self.pattern is None:
            return ""
        seen = self.map
        key = self._key_at(pos) if button == 1 else None
        if key is not None:
            return f"hear {name(key)}"
        bar = self._bar_at(pos) if button == 1 else None
        if bar is not None:
            return f"pick the chord of bar {bar + 1}"
        kind, index = begins(seen, self.notes, self._rows(), pos, button, mods)
        if not kind or not self._open:
            return OFF_GRID if kind else ""
        named = name(round(self.notes[index].pitch)) if index is not None else ""
        if kind == PLACE:
            row = noteedit.snapped(seen.row_at(pos[0]), self.grid)
            return (f"place a note at {name(seen.pitch_at(pos[1]))}, bar {int(row // self.bar_rows) + 1}, "
                    f"row {row % self.bar_rows + 1:g}")
        if kind == BOX:
            return "pick the notes that a box touches"
        if kind == PICK:
            return f"leave {named} out of the picked notes" if index in self._picked else f"pick {named} as well"
        if kind == LEVEL:
            return f"set the level of {named} to {seen.level_at(pos[1])}" if named else ""
        return f"{kind} {named}" if named else ""

    def on(self, event: Event, ui: Ui) -> bool:
        """Note where the mouse is, begin a gesture at a press and carry it on to its release, and use the wheel.

        The wheel scrolls the keys, with shift the rows, and with ctrl it zooms: not while a gesture is under way.
        A press is used when it begins a gesture, and the owner is told of no edit before its release. A press of
        the left button on a key of the strip is used as well: it begins no gesture, and its key is heard. So is
        one on a bar of the strip of the chords, of which the owner is told. A key of the computer goes to
        `key`, and one that is not the roll's to `typed`: it is used when either used it.
        """
        if event.kind == "key":
            return self.key(event) or (self.typed is not None and bool(self.typed(event)))
        self._mouse, self._mods = event.pos, event.mods
        if event.kind == "down":
            # A press while another button is held begins nothing.
            return ui.held is None and (self._press(event) or self._strike(event) or self._ask(event))
        gesture = self._gesture
        if event.kind == "wheel":
            return gesture is None and self._wheel(event)
        if gesture is not None:
            self._drag(event.pos)
            if event.kind == "up" and event.button == gesture.button:
                self._release()
        return False

    def _key_at(self, pos: Pos) -> int | None:
        """The pitch of the key of the strip that a position is on, or None where it is on no key of a channel."""
        seen = self.map
        pitch = seen.pitch_at(pos[1])
        on_key = self._own is not None and inside(seen.keys, pos) and LOWEST <= pitch <= HIGHEST
        return pitch if on_key else None

    def _strike(self, event: Event) -> bool:
        """Tell of the key of the strip that the left button is pressed on, to be heard, and say whether it was on one.

        The note is the key's, with the channel's own instrument. Nothing is changed and nothing is picked.
        """
        key = self._key_at(event.pos) if event.button == 1 else None
        if key is not None and self.sounded is not None:
            self.sounded(key, self._own.inst)
        return key is not None

    def _ask(self, event: Event) -> bool:
        """Tell the owner of the bar of the strip of the chords that the left button is pressed on.

        The answer is whether it was on one. Nothing is changed and nothing is picked: the owner sets the chord.
        """
        bar = self._bar_at(event.pos) if event.button == 1 else None
        if bar is not None and self.chord_asked is not None:
            self.chord_asked(bar, event.pos)
        return bar is not None

    def _wheel(self, event: Event) -> bool:
        """Scroll the keys by a step of the wheel, with shift the rows, and with ctrl zoom. It is always used."""
        seen = self.map
        if "ctrl" in event.mods:
            seen = seen.zoomed(seen.row_w + event.wheel, max(event.pos[0], seen.grid[0]))
        elif "shift" in event.mods:
            seen = replace(seen, row=seen.row - event.wheel * WHEEL_PX / seen.row_w)
        else:
            seen = replace(seen, top=seen.top + event.wheel * WHEEL_KEYS)
        self._see(seen)
        return True

    def _press(self, event: Event) -> bool:
        """Begin the gesture that a press begins, and say whether it began one. A press on a note picks first.

        The note is picked alone unless it was picked already. With ctrl it joins the picked ones, or leaves them.
        """
        self._gesture = None
        if not self._open:
            return False
        seen, pos = self.map, event.pos
        kind, index = begins(seen, self.notes, self._rows(), pos, event.button, event.mods)
        if not kind:
            return False
        told = self._tell(pos, event.mods, event.button)
        if kind == PICK:
            self._picked ^= {index}
        elif kind in (MOVE, STRETCH) and index not in self._picked:
            self._picked = frozenset({index})
        new = Note(noteedit.snapped(seen.row_at(pos[0]), self.grid), self._length, seen.pitch_at(pos[1]),
                   self._level, self._own.inst)
        self._gesture = Gesture(kind, event.button, index, seen, self.notes, self._rows(), self.grid, self._picked,
                                new, pos, told)
        return True

    def _drag(self, pos: Pos) -> None:
        """Take the gesture to where the mouse is, and tell of the note that is being moved at each new pitch."""
        gesture = self._gesture
        gesture.to(pos)
        if gesture.kind == MOVE:
            note = gesture.grabbed()
            if note.pitch != gesture.heard:
                gesture.heard = note.pitch
                self._sound(note)

    def _release(self) -> None:
        """End the gesture: other notes than there were are handed on, and the same notes are picked as it leaves them.

        The length of a note that was placed by a drag or stretched is kept for the next click, and the level that
        was set last for the next note. A note that was placed is told of, to be heard, unless the owner answers
        that it was not saved.
        """
        gesture, self._gesture = self._gesture, None
        notes, picked = gesture.gives()
        if gesture.kind == LEVEL and gesture.passed:
            self._level = list(gesture.passed.values())[-1]
        if notes == gesture.notes:
            self._picked = picked
            return
        placed = next(note for note in notes if note.row == gesture.new.row) if gesture.kind == PLACE else None
        if placed is not None and gesture.dragged:
            self._length = placed.rows
        elif gesture.kind == STRETCH:
            self._length = gesture.grabbed().rows
        if self._give(notes, picked, gesture.label) and placed is not None:
            self._sound(placed)

    def _give(self, notes: tuple[Note, ...], picked: frozenset[int], label: str) -> bool:
        """Hand an edit on to the owner, and keep what it leaves picked for when its notes are shown.

        The answer is whether the edit was saved: it was not where the owner answers false. With no owner, and
        with one that answers anything else or nothing, it is taken for saved.
        """
        self._gave = (notes, picked)
        return self.edited is None or self.edited(notes, label) is not False

    def _sound(self, note: Note) -> None:
        """Tell the owner of a note to let be heard."""
        if self.sounded is not None:
            self.sounded(note.pitch, note.inst)

    def key(self, event: Event) -> bool:
        """Take a key, and say whether it is one of the roll's.

        Delete and backspace remove the picked notes. Up and down move them a semitone, with shift an octave, and
        left and right a step of the grid. Ctrl and a picks all, ctrl and c copies the picked ones, and ctrl and
        v pastes them at the row under the mouse, or at the first row in view when the mouse is off the grid.
        Escape drops the gesture that is under way, or else what is picked. While a gesture is under way no
        other key is the roll's, and on a pattern that is not on the grid none is.
        """
        if not self._open:
            return False
        pressed, mods = event.key, event.mods
        if pressed == "escape" and not mods:
            if self._gesture is None:
                self._picked = frozenset()
            self._gesture = None
            return True
        if self._gesture is not None:
            return False
        notes, picked, rows = self.notes, self._picked, self._rows()
        if pressed in ("delete", "backspace") and not mods:
            self._made(noteedit.remove(notes, picked), counted(len(picked), "removed"))
        elif pressed in ("up", "down") and mods <= {"shift"}:
            semis = (12 if mods else 1) * (1 if pressed == "up" else -1)
            self._made(noteedit.shifted(notes, picked, semis, LOWEST, HIGHEST), counted(len(picked), "moved"))
        elif pressed in ("left", "right") and not mods:
            by_rows = self.grid if pressed == "right" else -self.grid
            self._made(noteedit.move(notes, picked, by_rows, 0, rows, LOWEST, HIGHEST),
                       counted(len(picked), "moved"))
        elif pressed == "a" and mods == _CTRL:
            self._picked = frozenset(range(len(notes)))
        elif pressed == "c" and mods == _CTRL:
            self._clip = noteedit.copied(notes, picked) if picked else self._clip
        elif pressed == "v" and mods == _CTRL:
            seen = self.map
            at_row = seen.row_at(self._mouse[0] if inside(seen.grid, self._mouse) else seen.grid[0])
            if self._clip:
                made = noteedit.pasted(notes, self._clip, noteedit.snapped(max(at_row, 0.0), self.grid), rows)
                self._made(made, counted(len(made[1]), "pasted"))
        else:
            return False
        return True

    def _made(self, made: Picked, label: str) -> None:
        """Take what a key made of the notes: other notes are handed on, and the same ones picked as it leaves them."""
        notes, picked = made
        if notes == self.notes:
            self._picked = picked
        else:
            self._give(notes, picked, label)

    def draw(self, canvas: Canvas) -> None:
        """Draw the keys, the grid, the notes, the lane of the levels and the playhead: all of it in the rectangle.

        While a gesture is under way, the notes are drawn as its release would leave them. With chords shown,
        the strip that names them is drawn as well.
        """
        seen = self.map
        notes, picked = self._now()
        with canvas.clip(self.rect):
            canvas.box(self.rect, theme.PANEL)
            self._draw_keys(canvas, seen)
            self._draw_grid(canvas, seen)
            self._draw_chords(canvas, seen)
            self._draw_notes(canvas, seen, notes, picked)
            self._draw_lane(canvas, seen, notes)
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
        """The rows of the keys as far as the pattern goes, the lines of its rows, and the numbers of its bars.

        Where a chord lies, the rows of its notes are a shade lighter, as far as the chord goes.
        """
        _, y, _, h = seen.grid
        rows = self._rows()
        left, right = seen.x(0), seen.x(rows)
        spans = [span for bar in (self.chords.bars if self.chords else ()) for span in bar.spans]
        with canvas.clip(seen.grid):
            for pitch in seen.pitches():
                top = seen.y(pitch)
                ground = theme.BACKGROUND if black(pitch) else theme.RAISED
                lit = mix(ground, theme.TEXT, CHORD_LIT)
                canvas.box((left, top, right - left, KEY_H), ground)
                for span in spans:
                    if pitch % 12 in span.tones:
                        start, end = seen.x(span.row), seen.x(min(span.row + span.rows, rows))
                        canvas.box((start, top, end - start, KEY_H), lit)
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

    def _draw_chords(self, canvas: Canvas, seen: Map) -> None:
        """The strip of the chords, with its name beside it: a box for each bar of the pattern, and its chord in it.

        The box of a bar with no chord is empty. With no chords shown there is no strip, and nothing is drawn.
        """
        if self.chords is None:
            return
        _, y, _, h = seen.chords
        canvas.text_in("CHORDS", (self.rect[0], y, KEYS_W - 5, h - 1), theme.DIM, "right")
        rows = self._rows()
        with canvas.clip(seen.chords):
            for index, bar in enumerate(self.chords.bars):
                left, right = seen.x(index * self.bar_rows), seen.x(min((index + 1) * self.bar_rows, rows))
                box = (left, y, right - left - 1, h - 1)
                canvas.box(box, theme.RAISED)
                canvas.frame(box, theme.LINE)
                room = max(0, right - left - 2 * CHORD_PAD)
                canvas.text_in(bar.text, (left + CHORD_PAD, y, room, h - 1), theme.ACTIVE)

    def _draw_notes(self, canvas: Canvas, seen: Map, notes: Sequence[Note], picked: frozenset[int]) -> None:
        """The ghosts, each in the colour its channel takes by its place in the song, and over them the notes.

        A picked note has a bright outline, and so has the box that a gesture is drawing. A note the chord
        check would find has an outline in the colour of a fault round its bar, on the lines of the grid, so
        that it is seen on a picked note as well.
        """
        with canvas.clip(seen.grid):
            for index, channel in enumerate(self.pattern.channels if self.pattern and self.ghosts else ()):
                if channel is self._own or channel.drums:
                    continue
                colour = mix(theme.CHANNELS[index % len(theme.CHANNELS)], theme.BACKGROUND, GHOST)
                for note in channel.notes:
                    x, y, w, h = seen.note(note.row, note.rows, note.pitch)
                    canvas.box((x, y + GHOST_INSET, w, h - 2 * GHOST_INSET), colour)
            light, dark = mix(self.colour, theme.TEXT, LIGHTER), mix(self.colour, theme.BACKGROUND, DARKER)
            for index, note in enumerate(notes):
                bar = x, y, w, h = seen.note(note.row, note.rows, note.pitch)
                canvas.box(bar, self.colour)
                if index in picked:
                    canvas.frame(bar, theme.TEXT)
                elif w > 2:
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
            for note, found in zip(notes, self._faults(notes)):
                if found is not None:
                    x, y, w, h = seen.note(note.row, note.rows, note.pitch)
                    canvas.frame((x - 1, y - 1, w + 2, h + 2), theme.FAULT)
            box = self._gesture.box if self._gesture is not None else None
            if box is not None:
                canvas.frame(box, theme.TEXT)

    def _draw_lane(self, canvas: Canvas, seen: Map, notes: Sequence[Note]) -> None:
        """The lane under the grid, with its name beside it and in it a bar for the level of each note."""
        x, y, w, h = seen.lane
        canvas.text_in("LEVEL", (self.rect[0], y, KEYS_W - 5, h), theme.DIM, "right")
        canvas.box(seen.lane, theme.BACKGROUND)
        with canvas.clip((x + 1, y + 1, w - 2, h - 2)):
            for note in notes:
                canvas.box(seen.level(note.row, note.rows, note.level), self.colour)
        canvas.frame(seen.lane, theme.LINE)
