"""The song strip: a song's patterns as blocks in the order they play, each with its notes in small, and the sums."""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from . import songview, theme
from .font import CUT, fit
from .layout import Rect
from .lists import Scroll
from .songview import Block, Note, Pattern
from .theme import Colour
from .ui import NOWHERE, Event, Pos, Ui, Widget, inside, share

if TYPE_CHECKING:
    from .canvas import Canvas

#: The least width of a row of the song. A song whose rows would be narrower in the strip keeps it, and scrolls.
ROW_W_MIN = 1
#: What is left free after a block, before the next one.
BLOCK_GAP = 1
#: How high the band along the top of a block is that its name stands in.
NAME_H = 11
#: From a side of a block and from its foot to the notes in it, its outline counted. Under the band of the name,
#: which has no outline, they start a pixel nearer.
INSET = 2
#: A channel's lane in a block leaves its lowest line of pixels free when it is at least this high.
AIR_FROM = 4
#: How high the tick of a drum hit is, where its lane has the room.
TICK_H = 2
#: How high the bar under the blocks is that shows where the view is, and what is left free over it.
BAR_H = 3
BAR_GAP = 1
#: How many pixels one step of the wheel moves a strip that scrolls.
WHEEL_PX = 64


def _px(value: float) -> int:
    """The whole pixel nearest to a place. A half goes up."""
    return math.floor(value + 0.5)


def _down(room: int, part: float, whole: float) -> int:
    """How far under the top of `room` pixels a thing is that stands `part` of `whole` of the way up from the foot.

    It is no lower than the foot and no higher than the top. With nothing to measure against, it is in the middle.
    """
    if whole <= 0:
        return room // 2
    return room - _px(min(max(part / whole, 0.0), 1.0) * room)


def places(rect: Rect, notes: Iterable[Note], rows: float, low: float | None = None,
           high: float | None = None) -> list[Rect]:
    """Where each of a channel's notes goes when they are drawn small in a rectangle that stands for `rows` rows.

    Time goes across and pitch up. A note is a line one pixel high from its start to a pixel short of its end, so
    that two in a row are seen as two, and it is never shorter than a pixel. It stands by its pitch between `low`
    at the foot and `high` at the top, which are the lowest and the highest of these notes unless they are given.
    A single pitch is in the middle. A drum hit is a tick one pixel wide where it starts, in the lane of its
    letter: the lanes share the height, and the letter that plays the lowest note has the lowest of them. A
    rectangle with no room in it, and one that stands for no rows, has no places.
    """
    x, y, w, h = rect
    notes = list(notes)
    if w <= 0 or h <= 0 or rows <= 0:
        return []
    tuned = [note.pitch for note in notes if not note.drum]
    low = min(tuned, default=0.0) if low is None else low
    high = max(tuned, default=0.0) if high is None else high
    plays = {note.drum: note.pitch for note in notes if note.drum}
    lanes = sorted(plays, key=lambda letter: (plays[letter], letter))
    tick = min(TICK_H, max(1, h // max(1, len(lanes))))
    found = []
    for note in notes:
        left = x + math.floor(note.row * w / rows)
        if note.drum:
            found.append((left, y + _down(h - tick, lanes.index(note.drum), len(lanes) - 1), 1, tick))
        else:
            right = x + math.floor((note.row + note.rows) * w / rows)
            found.append((left, y + _down(h - 1, note.pitch - low, high - low), max(1, right - left - 1), 1))
    return found


def small(canvas: Canvas, rect: Rect, notes: Iterable[Note], rows: float, colour: Colour,
          low: float | None = None, high: float | None = None) -> None:
    """Draw a channel's notes small in a rectangle, in one colour: each where `places` puts it, none outside."""
    with canvas.clip(rect):
        for place in places(rect, notes, rows, low, high):
            canvas.box(place, colour)


@dataclass(frozen=True)
class Map:
    """Where the blocks of a song fall in the rectangle of a strip, and back: the sums, with nothing drawn.

    rect is the strip's rectangle. blocks are the places in the order of the song, and rows the rows of the whole
    song. offset is the view: how many pixels of the song lie before the left edge of the strip.
    """

    rect: Rect
    blocks: tuple[Block, ...] = ()
    rows: float = 0.0
    offset: int = 0

    @property
    def width(self) -> int:
        """How wide the whole song is: as wide as the strip, or wider where its rows would be under `ROW_W_MIN`."""
        return max(self.rect[2], math.ceil(self.rows * ROW_W_MIN))

    @property
    def most(self) -> int:
        """The offset at which the end of the song is at the right edge of the strip. It is 0 when the song fits."""
        return self.width - self.rect[2]

    @property
    def block_h(self) -> int:
        """How high a block is: what the line under the picked one, and the bar under that, leave of the strip."""
        return max(0, self.rect[3] - theme.STRIP - BAR_GAP - BAR_H)

    def x(self, row: float) -> int:
        """The x of a row of the song, which can be a fraction."""
        along = _px(row * self.width / self.rows) if self.rows else 0
        return self.rect[0] + along - self.offset

    def sees(self, row: float) -> bool:
        """Whether a row of the song is in view."""
        return 0 <= self.x(row) - self.rect[0] < self.rect[2]

    def block(self, index: int) -> Rect:
        """The rectangle of a block: as wide as its rows, less what is left free after it."""
        found = self.blocks[index]
        left = self.x(found.row)
        return left, self.rect[1], max(0, self.x(found.row + found.rows) - left - BLOCK_GAP), self.block_h

    def in_view(self) -> list[int]:
        """The blocks of which at least a pixel is in view, by their index."""
        x, _, w, _ = self.rect
        found = []
        for index in range(len(self.blocks)):
            left, _, width, _ = self.block(index)
            if width and left < x + w and left + width > x:
                found.append(index)
        return found

    def block_at(self, pos: Pos) -> int | None:
        """The index of the block a position is on. None between two blocks, under them, and off the strip."""
        if not inside(self.rect, pos):
            return None
        return next((index for index in self.in_view() if inside(self.block(index), pos)), None)

    def name(self, index: int) -> str:
        """The name of a block as its width lets it show: whole, cut short, or nothing when no letter of it fits."""
        name = self.blocks[index].name
        shown = fit(name, self.block(index)[2] - 2 * theme.PAD)
        return shown if shown == name or len(shown) > len(CUT) else ""

    def lanes(self, index: int, count: int) -> list[Rect]:
        """Where the notes of each channel go in a block: `count` lanes, which share the room under its name.

        The first lane is at the top. One that is `AIR_FROM` high or more leaves its lowest line of pixels free.
        """
        x, y, w, h = self.block(index)
        found, top = [], y + NAME_H + INSET - 1
        for height in share(max(0, h - NAME_H - 2 * INSET + 1), [1] * count):
            found.append((x + INSET, top, max(0, w - 2 * INSET), height - (height >= AIR_FROM)))
            top += height
        return found

    def thumb(self) -> Rect | None:
        """The thumb of the bar under the blocks: it is to the strip what the view is to the song. None when it fits."""
        x, y, w, h = self.rect
        found = Scroll(self.width, w, self.offset).thumb(w)
        return None if found is None else (x + found[0], y + h - BAR_H, found[1], BAR_H)

    def scrolled(self, pixels: int = 0) -> Map:
        """This map with the view moved on by so many pixels, and kept between the start and the end of the song."""
        return replace(self, offset=min(max(0, self.offset + pixels), self.most))

    def following(self, row: float) -> Map:
        """This map with the view moved so that a row is at its left edge, or as near to it as the song lets it."""
        return self.scrolled(self.x(row) - self.rect[0])


class SongStrip(Widget):
    """A song at a glance: a block for each place in its order, as wide as its rows, with its notes in small.

    `picked` is the index of the picked block, and None while there is no block. A click on a block picks it, and
    `changed` is called with its index, also when it was picked already. `playhead` is the row of the song that is
    playing, which can be a fraction, or None: the owner sets it each frame, and the view goes after it when it
    leaves the view. `offset` is the view, as a `Map` has it. The whole song fits the strip unless that makes its
    rows narrower than `ROW_W_MIN`: then the wheel moves it. `help`, while the mouse is on a block, says which
    block it is; anywhere else it is the line the strip was made with.
    """

    def __init__(self, rect: Rect, changed: Callable[[int], object] | None = None, help: str = "") -> None:
        super().__init__(rect, help)
        self.changed = changed
        self.picked: int | None = None
        self.offset = 0
        self._blocks: tuple[Block, ...] = ()
        self._rows = 0.0
        self._bar_rows = 16
        self._patterns: dict[str, Pattern] = {}
        #: The lowest and the highest pitch of each channel in the whole song, so that a pitch is as high in one
        #: block as in another. None for a channel with no tuned note.
        self._spans: list[tuple[float | None, float | None]] = []
        self._playhead: float | None = None
        self._mouse: Pos = NOWHERE

    @property
    def help(self) -> str:
        """The block the mouse is on, its bars and its place in the order, or the line the strip was made with."""
        index = self.block_at(self._mouse)
        if index is None:
            return self._help
        block = self._blocks[index]
        bars = round(block.rows / self._bar_rows, 2)
        return f"{block.name}, {bars:g} {'bar' if bars == 1 else 'bars'}, {index + 1} of {len(self._blocks)}"

    @help.setter
    def help(self, line: str) -> None:
        self._help = line

    @property
    def map(self) -> Map:
        """The sums of the strip as it is now, the view kept inside the song."""
        return Map(self.rect, self._blocks, self._rows, self.offset).scrolled()

    @property
    def playhead(self) -> float | None:
        """The row of the song that is playing, or None."""
        return self._playhead

    @playhead.setter
    def playhead(self, row: float | None) -> None:
        before, self._playhead = self._playhead, row
        seen = self.map
        # Only a playhead that was in view is gone after: a view the wheel moved away from it stays where it is.
        if row is not None and before is not None and seen.sees(before) and not seen.sees(row):
            self.offset = seen.following(row).offset

    def show(self, song: dict) -> None:
        """Show a song, as `Project.song` gives it: a block for each place in its order, each channel in its colour.

        A song with the same order as the one before is taken for the same song, come back with other notes: the
        pick and the view stay. Any other starts with its first block picked and the view at its start. What
        `songview.blocks` refuses is refused, and what was shown before is then still shown.
        """
        found, rows = songview.blocks(song)
        patterns = {name: songview.pattern(song, name) for name in dict.fromkeys(block.name for block in found)}
        spans = []
        for index in range(len(song["channels"])):
            pitches = [note.pitch for pattern in patterns.values() for note in pattern.channels[index].notes
                       if not note.drum]
            spans.append((min(pitches, default=None), max(pitches, default=None)))
        another = [block.name for block in found] != [block.name for block in self._blocks]
        self._blocks, self._rows, self._bar_rows = tuple(found), rows, songview.bar_rows(song)
        self._patterns, self._spans = patterns, spans
        if another:
            self.picked, self.offset = (0 if found else None), 0
        self.offset = self.map.offset

    def block_at(self, pos: Pos) -> int | None:
        """The index of the block a position is on, or None when it is on none."""
        return self.map.block_at(pos)

    def on(self, event: Event, ui: Ui) -> bool:
        """Note where the mouse is, pick the block the left button goes down on, and move the view at the wheel.

        The wheel is used only by a song that does not fit the strip. No key is used.
        """
        if event.kind == "key":
            return False
        self._mouse = event.pos
        seen = self.map
        if event.kind == "wheel":
            self.offset = seen.scrolled(-event.wheel * WHEEL_PX).offset
            return seen.most > 0
        index = seen.block_at(event.pos)
        if event.kind != "down" or event.button != 1 or index is None:
            return False
        self.picked = index
        if self.changed is not None:
            self.changed(index)
        return True

    def draw(self, canvas: Canvas) -> None:
        """Draw the blocks in view, the line under the picked one, the bar while the song scrolls, and the playhead.

        Nothing is drawn outside the rectangle.
        """
        seen = self.map
        x, y, w, _ = self.rect
        with canvas.clip(self.rect):
            canvas.box(self.rect, theme.PANEL)
            for index in seen.in_view():
                self._draw_block(canvas, seen, index)
            thumb = seen.thumb()
            if thumb is not None:
                canvas.box((x, thumb[1], w, BAR_H), theme.BACKGROUND)
                canvas.box(thumb, theme.LINE)
            if self._playhead is not None:
                canvas.vline(seen.x(self._playhead), y, seen.block_h + theme.STRIP, theme.ACTIVE)

    def _draw_block(self, canvas: Canvas, seen: Map, index: int) -> None:
        """One block: its face, the band of its name, its outline, its name, and the notes of its channels."""
        rect = x, y, w, h = seen.block(index)
        picked = index == self.picked
        pattern = self._patterns[seen.blocks[index].name]
        with canvas.clip(rect):
            canvas.box(rect, theme.LIT if picked else theme.RAISED)
            canvas.box((x, y, w, NAME_H), theme.LINE if picked else theme.LIT)
            canvas.frame(rect, theme.LINE)
            name = seen.name(index)
            if name:
                canvas.text_in(name, (x + theme.PAD, y, w - 2 * theme.PAD, NAME_H), theme.TEXT)
            for lane, room in enumerate(seen.lanes(index, len(pattern.channels))):
                small(canvas, room, pattern.channels[lane].notes, pattern.rows,
                      theme.CHANNELS[lane % len(theme.CHANNELS)], *self._spans[lane])
        if picked:
            canvas.box((x, y + h, w, theme.STRIP), theme.ACTIVE)
