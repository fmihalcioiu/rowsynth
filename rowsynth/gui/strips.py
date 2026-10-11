"""The song strip: a song's patterns as blocks in the order they play, each with its notes in small, and the sums."""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ..errors import RowSynthError
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
#: How many pixels the mouse may go along the strip from where it was pressed on a block and not have picked the
#: block up: a click wobbles.
SLACK = 3
#: How many pixels the foot of a block that is picked up stands over where it stood. Its shadow lies there.
LIFT = 1
#: How long an arm of the plus is that the line of a gap carries while ctrl is held, its middle pixel not counted.
PLUS = 2
#: While a block is carried this many pixels from an end of the strip or nearer, or past the end, the view moves
#: that way, and by this many pixels a frame.
EDGE_W = 12
EDGE_PX = 4
#: Two presses on the name of one block are a double press when no more than this many seconds lie between them.
DOUBLE = 0.4


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

    def name_rect(self, index: int) -> Rect:
        """The rectangle the name of a block is drawn in: the band along its top, less what is free at its sides."""
        x, y, w, _ = self.block(index)
        return x + theme.PAD, y, max(0, w - 2 * theme.PAD), NAME_H

    def gap_x(self, index: int) -> int:
        """The x of a gap, where a block can be put: gap 0 is before the first block, and the last after the last.

        So a song of four blocks has five, and gap 2 is between its second block and its third. The x is that of
        the pixel left free before the block the gap is before, and of the one after the last block. Before the
        first block none is free, and it is the first pixel of the song.
        """
        row = self.blocks[index].row if index < len(self.blocks) else self.rows
        return max(self.x(row) - BLOCK_GAP, self.x(0))

    def gap_at(self, pos: Pos) -> int:
        """The gap nearest a position, by its x alone. Of two that are as near, it is the earlier.

        A position off the strip to one side counts as on the edge of the strip there, so that no gap far out of
        view is found.
        """
        x, _, w, _ = self.rect
        at = min(max(pos[0], x), x + w - 1)
        return min(range(len(self.blocks) + 1), key=lambda index: abs(self.gap_x(index) - at))

    @staticmethod
    def to(place: int, gap: int) -> int | None:
        """The place in the order a block is at after it is moved from a place to a gap: the `to` of `songedit.moved`.

        None for the gap before the block and for the one after it, where it stays at the place it has. A copy
        takes nothing out of the order, so the place it is at afterwards is the number of its gap.
        """
        if gap in (place, place + 1):
            return None
        return gap if gap < place else gap - 1

    def plus(self, gap: int) -> Rect:
        """The square the plus of a gap stands in: at the top of the line of the gap, in the middle of it.

        It is kept inside the strip, so at an end of the strip it is beside the line.
        """
        x, y, w, _ = self.rect
        side = 2 * PLUS + 3
        return min(max(self.gap_x(gap) - PLUS - 1, x), max(x, x + w - side)), y, side, side

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


@dataclass
class Drag:
    """A press of the left button on a block, which a drag along the strip makes into carrying the block.

    place is the place of the block in the order, and start where the press was. dragged is whether the mouse has
    been more than `SLACK` pixels along the strip from the press. From then on the block is picked up, and it
    stays so when the mouse comes back.
    """

    place: int
    start: Pos
    dragged: bool = False

    def along(self, pos: Pos) -> None:
        """Take the mouse on to another position."""
        self.dragged = self.dragged or abs(pos[0] - self.start[0]) > SLACK

    def gap(self, seen: Map, pos: Pos, copy: bool) -> int | None:
        """The gap in which a release at a position puts the block, or with `copy` a copy of it. None for neither.

        A block that is not picked up is put nowhere. One that is moved is put nowhere either by the gap before
        it and the one after it, where it would land at the place it has.
        """
        if not self.dragged:
            return None
        gap = seen.gap_at(pos)
        return gap if copy or seen.to(self.place, gap) is not None else None


class SongStrip(Widget):
    """A song at a glance: a block for each place in its order, as wide as its rows, with its notes in small.

    `picked` is the index of the picked block, and None while there is no block. A press of the left button on a
    block picks it, and `changed` is called with its index, also when it was picked already. `playhead` is the row
    of the song that is playing, which can be a fraction, or None: the owner sets it each frame, and the view goes
    after it when it leaves the view. `offset` is the view, as a `Map` has it. The whole song fits the strip unless
    that makes its rows narrower than `ROW_W_MIN`: then the wheel moves it.

    The order is changed with the mouse, and the strip saves nothing: it tells its owner what the hand asked for.
    A block that is dragged more than `SLACK` pixels along the strip is picked up, and a bright line shows the gap
    it would be put in, which is the one nearest the mouse. The release calls `moved` with the place the block had
    and the place it is at afterwards, as `songedit.moved` takes them. Where the block would land at the place it
    has, nothing is called, and no line is shown. With ctrl held at the release the block stays and `copied` is
    called, with its place and the place its copy is at afterwards: the line has a plus while ctrl is held.
    `busy` is whether a press on a block is under way, from the press to the release. `lifted` is the place of the
    block that is picked up, and `gap` the gap the line shows. Escape, which the owner hands to `key`, drops what
    is under way, and so does `show`. While a block is carried near an end of the strip the view moves that way:
    for that the owner calls `update` every frame.

    The right button on a block calls `asked` with the place of the block and the position, and past the last block
    with None and the position: the owner can open a menu there. Two presses on the name of one block that are no
    more than `DOUBLE` seconds apart by `clock` call `named` with its place, and `name_rect` is where that name is
    drawn: the owner can put a box for text over it. The strip opens nothing itself.

    `help`, while the mouse is on a block, says what a press there would do; anywhere else it is the line the strip
    was made with.
    """

    def __init__(self, rect: Rect, changed: Callable[[int], object] | None = None, help: str = "",
                 moved: Callable[[int, int], object] | None = None,
                 copied: Callable[[int, int], object] | None = None,
                 asked: Callable[[int | None, Pos], object] | None = None,
                 named: Callable[[int], object] | None = None,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        super().__init__(rect, help)
        self.changed = changed
        self.moved = moved
        self.copied = copied
        self.asked = asked
        self.named = named
        self.clock = clock
        self.picked: int | None = None
        self.offset = 0
        self._blocks: tuple[Block, ...] = ()
        self._rows = 0.0
        self._patterns: dict[str, Pattern] = {}
        #: The lowest and the highest pitch of each channel in the whole song, so that a pitch is as high in one
        #: block as in another. None for a channel with no tuned note.
        self._spans: list[tuple[float | None, float | None]] = []
        self._playhead: float | None = None
        self._mouse: Pos = NOWHERE
        self._mods: frozenset[str] = frozenset()
        #: The press on a block that is under way, or None.
        self._drag: Drag | None = None
        #: The last press on the name of a block, where a second one would make a double press: its place, and
        #: what the clock said.
        self._last: tuple[int, float] | None = None

    @property
    def help(self) -> str:
        """What a press on the block the mouse is on would do, or the line the strip was made with.

        While a press on a block is under way, it is the line of that block.
        """
        index = self._drag.place if self._drag is not None else self.block_at(self._mouse)
        if index is None:
            return self._help
        return f"drag to move {self._blocks[index].name}, ctrl to copy; right button for more"

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
        # While a press on a block is under way the view stays as well, so that no gap moves under the hand.
        if (row is not None and before is not None and seen.sees(before) and not seen.sees(row)
                and self._drag is None):
            self.offset = seen.following(row).offset

    @property
    def busy(self) -> bool:
        """Whether a press on a block is under way: from the press to its release, or until it is dropped."""
        return self._drag is not None

    @property
    def lifted(self) -> int | None:
        """The place of the block that is picked up, or None while none is."""
        return self._drag.place if self._drag is not None and self._drag.dragged else None

    @property
    def gap(self) -> int | None:
        """The gap that a release where the mouse is would put the block in, or its copy while ctrl is held.

        None where a release would do nothing. Gaps are counted as `Map.gap_x` counts them.
        """
        return None if self._drag is None else self._drag.gap(self.map, self._mouse, "ctrl" in self._mods)

    def show(self, song: dict) -> None:
        """Show a song, as `Project.song` gives it: a block for each place in its order, each channel in its colour.

        A song with the same order as the one before is taken for the same song, come back with other notes: the
        pick and the view stay. Any other starts with its first block picked and the view at its start. A press
        on a block that is under way is dropped, and nothing is called for it. What `songview.blocks` refuses is
        refused, and what was shown before is then still shown.
        """
        found, rows = songview.blocks(song)
        patterns = {name: songview.pattern(song, name) for name in dict.fromkeys(block.name for block in found)}
        spans = []
        for index in range(len(song["channels"])):
            pitches = [note.pitch for pattern in patterns.values() for note in pattern.channels[index].notes
                       if not note.drum]
            spans.append((min(pitches, default=None), max(pitches, default=None)))
        another = [block.name for block in found] != [block.name for block in self._blocks]
        self._blocks, self._rows = tuple(found), rows
        self._patterns, self._spans = patterns, spans
        self._drag = None
        if another:
            self.picked, self.offset, self._last = (0 if found else None), 0, None
        self.offset = self.map.offset

    def block_at(self, pos: Pos) -> int | None:
        """The index of the block a position is on, or None when it is on none."""
        return self.map.block_at(pos)

    def name_rect(self, place: int) -> Rect:
        """The rectangle the name of the block at a place is drawn in, as the view has it now.

        A place the order does not have is refused.
        """
        count = len(self._blocks)
        if isinstance(place, bool) or not isinstance(place, int) or not 0 <= place < count:
            raise RowSynthError(f"{place!r} is no place in the order, which has {count}, counted from 0")
        return self.map.name_rect(place)

    def on(self, event: Event, ui: Ui) -> bool:
        """Note where the mouse is and what is held, take a press, carry a block on to its release, and use the wheel.

        A press is used when it is the strip's, as `_press` says. The wheel moves the view, and is used only by
        a song that does not fit the strip. A key goes to `key`.
        """
        if event.kind == "key":
            return self.key(event)
        self._mouse, self._mods = event.pos, event.mods
        seen = self.map
        if event.kind == "wheel":
            self.offset = seen.scrolled(-event.wheel * WHEEL_PX).offset
            return seen.most > 0
        if event.kind == "down":
            # A press while another button is held begins nothing.
            return ui.held is None and self._press(event, seen)
        if self._drag is not None:
            self._drag.along(event.pos)
            if event.kind == "up" and event.button == 1:
                self._release()
        return False

    def _press(self, event: Event, seen: Map) -> bool:
        """Take a press of a button, and say whether it was the strip's.

        The right button is the strip's on a block and past the last block, and `asked` is told of it. The left
        button is the strip's on a block. As the second press of a double press on a name it is told to `named`,
        and does no more. Any other picks the block, tells `changed`, and begins what a drag makes into carrying.
        """
        self._drag = None
        index, pos = seen.block_at(event.pos), event.pos
        if event.button == 3:
            if index is None and not (inside(seen.rect, pos) and pos[0] >= seen.gap_x(len(seen.blocks))):
                return False
            if self.asked is not None:
                self.asked(index, pos)
            return True
        if event.button != 1 or index is None:
            return False
        before, now = self._last, self.clock()
        self._last = (index, now) if pos[1] < seen.rect[1] + NAME_H else None
        if self._last is not None and before is not None and before[0] == index and now - before[1] <= DOUBLE:
            # A third press begins anew.
            self._last = None
            if self.named is not None:
                self.named(index)
            return True
        self.picked = index
        self._drag = Drag(index, pos)
        if self.changed is not None:
            self.changed(index)
        return True

    def _release(self) -> None:
        """End the press on a block: one that was carried to a gap is told of, as moved, or with ctrl as copied."""
        drag, gap = self._drag, self.gap
        self._drag = None
        if gap is None:
            return
        if "ctrl" in self._mods:
            tell, to = self.copied, gap
        else:
            tell, to = self.moved, Map.to(drag.place, gap)
        if tell is not None:
            tell(drag.place, to)

    def key(self, event: Event) -> bool:
        """Take a key, and say whether it was the strip's.

        Escape is the strip's while a press on a block is under way. It drops the press, whether a block is
        picked up or not, and the release then does nothing. No other key is the strip's.
        """
        if event.key != "escape" or self._drag is None:
            return False
        self._drag = None
        return True

    def update(self) -> None:
        """Move the view while a block is carried near an end of the strip. The owner calls this every frame.

        The view moves `EDGE_PX` pixels towards the end that the mouse is within `EDGE_W` pixels of, or past.
        It moves only in a strip that scrolls, and no further than the start and the end of the song.
        """
        if self.lifted is None:
            return
        x, _, w, _ = self.rect
        at = self._mouse[0]
        way = -1 if at < x + EDGE_W else 1 if at >= x + w - EDGE_W else 0
        self.offset = self.map.scrolled(way * EDGE_PX).offset

    def draw(self, canvas: Canvas) -> None:
        """Draw the blocks in view, the line under the picked one, the bar while the song scrolls, and the playhead.

        While a block is carried, it is drawn picked up, and a bright line stands in the gap its release would
        put it in. The line has a plus at its top while ctrl is held. Nothing is drawn outside the rectangle.
        """
        seen = self.map
        x, y, w, _ = self.rect
        lifted, gap = self.lifted, self.gap
        with canvas.clip(self.rect):
            canvas.box(self.rect, theme.PANEL)
            for index in seen.in_view():
                self._draw_block(canvas, seen, index, index == lifted)
            thumb = seen.thumb()
            if thumb is not None:
                canvas.box((x, thumb[1], w, BAR_H), theme.BACKGROUND)
                canvas.box(thumb, theme.LINE)
            if self._playhead is not None:
                canvas.vline(seen.x(self._playhead), y, seen.block_h + theme.STRIP, theme.ACTIVE)
            if gap is not None:
                canvas.vline(seen.gap_x(gap), y, seen.block_h, theme.TEXT)
                if "ctrl" in self._mods:
                    left, top, side, _ = seen.plus(gap)
                    canvas.box((left, top, side, side), theme.TEXT)
                    canvas.hline(left + 1, top + PLUS + 1, side - 2, theme.BACKGROUND)
                    canvas.vline(left + PLUS + 1, top + 1, side - 2, theme.BACKGROUND)

    def _draw_block(self, canvas: Canvas, seen: Map, index: int, lifted: bool = False) -> None:
        """One block: its face, the band of its name, its outline, its name, and the notes of its channels.

        One that is picked up has a bright outline, and its foot stands `LIFT` pixels over its shadow.
        """
        x, y, w, h = seen.block(index)
        rect = (x, y, w, max(0, h - LIFT)) if lifted else (x, y, w, h)
        picked = index == self.picked
        pattern = self._patterns[seen.blocks[index].name]
        if lifted:
            canvas.box((x, y, w, h), theme.BACKGROUND)
        with canvas.clip(rect):
            canvas.box(rect, theme.LIT if picked else theme.RAISED)
            canvas.box((x, y, w, NAME_H), theme.LINE if picked else theme.LIT)
            canvas.frame(rect, theme.TEXT if lifted else theme.LINE)
            name = seen.name(index)
            if name:
                canvas.text_in(name, seen.name_rect(index), theme.TEXT)
            for lane, room in enumerate(seen.lanes(index, len(pattern.channels))):
                small(canvas, room, pattern.channels[lane].notes, pattern.rows,
                      theme.CHANNELS[lane % len(theme.CHANNELS)], *self._spans[lane])
        if picked:
            canvas.box((x, y + h, w, theme.STRIP), theme.ACTIVE)
