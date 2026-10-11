"""The layers of a song as a table of its groups against its versions, each version to be heard, and its intro."""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from itertools import accumulate
from typing import TYPE_CHECKING, NamedTuple

from ..errors import RowSynthError
from ..layers import _FADE, _TAIL
from ..note import TICK_HZ
from . import layeredit, songview, theme
from .channelbar import BUTTON_H, HEIGHT, SPACE, tidy
from .dials import Knob, Stepper, knob_parts, plain
from .font import Font
from .layout import Rect
from .panels import within
from .popups import Item, Menu, ask
from .theme import Colour
from .ui import Button, Event, Label, Pos, Toggle, Ui, Widget, draw_button, inside

if TYPE_CHECKING:
    from ..project import Project
    from .canvas import Canvas
    from .panels import Panel

#: What the switches and the words of the part read.
USE = "USE LAYERS"
ALL = "ALL"
KEYS = "KEYS"
INTRO = "INTRO"
PLAYS = "PLAYS"
TAIL = "TAIL"
FADE = "FADE"
#: What is said of layers to a song that has none.
ABOUT = "layers let a program fade between a calm and a full version of the same song"

#: How high a row of the table is: its headings, each group, and the button that adds a group.
ROW_H = BUTTON_H + 1
#: The least width of the room the groups are named in. It is as wide as the longest of them needs.
LEFT_LEAST = 96
#: How wide the column of a version is: as wide as fits, from the most down to the least. Columns that do not
#: fit at the least width are not all in view, and neither are groups for which the table is not high enough:
#: the table then shows them from a first one on, and a small arrow marks each way in which there are more.
COL_MOST = 64
COL_LEAST = 30
#: How wide the field of the keys is, that of the patterns an intro plays, and a knob of the intro with its two
#: lines: the line of its value has room for hundredths of a second.
KEYS_W = 54
ORDER_W = 78
LENGTH_W = Font.width("0.00 s") + 2
#: What is left free over the intro, the most its picture is wide, and what is left free above and below it.
APART = 8
PICTURE_MOST = 240
PICTURE_AIR = 3
#: The travel of the fade's knob: its lowest value, its highest, and the steps between them. An intro may hold
#: any fade up to its tail, so the travel is chosen here: in hundredths of a second up to half of one. The tail
#: goes in hundredths up to a bar of its song.
FADES = (0.0, 0.5, 50)

#: The rules that add, rename and remove a group or a version.
_ADDED = {"group": layeredit.group_added, "version": layeredit.version_added}
_RENAMED = {"group": layeredit.group_renamed, "version": layeredit.version_renamed}
_REMOVED = {"group": layeredit.group_removed, "version": layeredit.version_removed}


def key_text(key: int) -> str:
    """A key as it is shown: the semitones the song is moved up with their sign, and "0" for the song as written."""
    return f"{key:+d}" if key else "0"


def seconds_text(seconds: float) -> str:
    """A length of time as it is shown: the number with no more places than it needs, and an s after it."""
    return f"{plain(seconds)} s"


class Group(NamedTuple):
    """One group as the table shows it: its name, and each of its channels with the colour it has in its song."""

    name: str
    channels: tuple[tuple[str, Colour], ...]


class Version(NamedTuple):
    """One version as the table shows it: its name, the names of the groups in it, and the keys it is premixed in."""

    name: str
    groups: frozenset[str]
    premix: tuple[int, ...] = ()


class Laid(NamedTuple):
    """Where the parts of a table stand, as `lay` has them.

    `left` is how wide the room is the groups are named in, and `wide` the column of a version. `heads` has the
    heading of each version in view and `rows` the room each group in view is named in. `add_version` is the
    button after the last heading, and `add_group` the one under the last group. `first_group` and
    `first_version` are the places of the first group and of the first version in view, among all there are.
    """

    left: int
    wide: int
    heads: tuple[Rect, ...]
    rows: tuple[Rect, ...]
    add_version: Rect
    add_group: Rect
    first_group: int = 0
    first_version: int = 0


def kept(first: int, count: int, fit: int) -> int:
    """The first of `count` things in a view that `fit` of them fit in, kept in range.

    It is no less than 0, and no more than the first at which the last of them is in view. Where they all fit,
    and where none does, it is 0.
    """
    return max(0, min(first, count - fit)) if fit > 0 else 0


def lay(rect: Rect, groups: int, versions: int, label: int = 0, first_group: int = 0, first_version: int = 0) -> Laid:
    """Where a table of so many groups against so many versions stands in its rectangle.

    The headings of the versions run along the top, and under them is a row for each group. A row begins with
    the room the group is named in, `label` wide where the table has that much, and never under `LEFT_LEAST`.
    After it the versions have a column each, as wide as fits from `COL_MOST` down to `COL_LEAST`. A small
    button stands after the last heading and another under the last row. A rectangle too small for a heading
    with one column and a row under it has no table in it.

    Where not every group or not every version fits, those in view are the ones from `first_group` and from
    `first_version` on, each kept in range as `kept` keeps it. The rows and the headings stand where they
    stand for a table that shows its first ones.
    """
    x, y, w, h = rect
    room = w - LEFT_LEAST - BUTTON_H
    if room < COL_LEAST or h < 2 * ROW_H:
        return Laid(0, 0, (), (), (x, y, 0, 0), (x, y, 0, 0))
    wide = max(COL_LEAST, min(COL_MOST, room // max(1, versions)))
    seen = min(versions, room // wide)
    left = max(LEFT_LEAST, min(label, w - BUTTON_H - seen * wide))
    shown = min(groups, h // ROW_H - 2)
    return Laid(left, wide,
                tuple((x + left + at * wide, y, wide, ROW_H) for at in range(seen)),
                tuple((x, y + (at + 1) * ROW_H, left, ROW_H) for at in range(shown)),
                (x + left + seen * wide, y, BUTTON_H, BUTTON_H),
                (x, y + (shown + 1) * ROW_H, BUTTON_H, BUTTON_H),
                kept(first_group, groups, shown), kept(first_version, versions, seen))


def head_parts(head: Rect) -> tuple[Rect, Rect]:
    """Where the button that plays a version goes in its heading, and the name of the version after it."""
    x, y, w, _ = head
    return (x + 1, y, BUTTON_H, BUTTON_H), (x + BUTTON_H + theme.GAP, y, max(0, w - BUTTON_H - theme.GAP - 1), BUTTON_H)


def cell(laid: Laid, group: int, version: int) -> Rect:
    """The switch that says whether a group is in a version, both in view, by their places among all there are."""
    return (laid.heads[version - laid.first_version][0] + 1, laid.rows[group - laid.first_group][1], laid.wide - 2,
            ROW_H - 1)


def hit(laid: Laid, pos: Pos) -> tuple | None:
    """What of a table a position is on, or None for nothing.

    ("play", version) is the button that plays a version and ("version", version) the rest of its heading,
    ("group", group) the room a group is named in and ("cell", group, version) the switch of the two, each by
    its place among all the groups or the versions there are, in view or not. ("add", "version") and
    ("add", "group") are the two buttons that add one.
    """
    for at, head in enumerate(laid.heads, laid.first_version):
        if inside(head, pos):
            return ("play" if inside(head_parts(head)[0], pos) else "version", at)
    for at, row in enumerate(laid.rows, laid.first_group):
        if inside(row, pos):
            return "group", at
        for to, head in enumerate(laid.heads, laid.first_version):
            if inside((head[0], row[1], head[2], row[3]), pos):
                return "cell", at, to
    for kind in ("version", "group"):
        if inside(laid.add_version if kind == "version" else laid.add_group, pos):
            return "add", kind
    return None


def more(laid: Laid, groups: int, versions: int) -> dict[str, Pos]:
    """Where the marks go that say a table of so many groups and versions has some out of view, each by its way.

    A mark is a small arrow that points the way there are more, and what is given for it is its tip. "left" is
    for versions before the first in view: it stands before the first heading, in the corner over the groups.
    "right" is for versions after the last in view, and stands after the last switch of the first row. "up" is
    for groups above the first in view, in that corner as well, and "down" for groups under the last, beside
    the button that adds one. A way in which nothing is out of view has no mark, and neither has a rectangle
    with no table in it.
    """
    if not laid.wide:
        return {}
    (right, top), (x, foot) = laid.add_version[:2], laid.add_group[:2]
    middle, air = BUTTON_H // 2, (BUTTON_H - theme.ARROW) // 2
    tips = {
        "left": (laid.first_version > 0, (x + laid.left - theme.PAD - theme.ARROW, top + middle)),
        "right": (laid.first_version + len(laid.heads) < versions,
                  (right + theme.PAD + theme.ARROW - 1, top + ROW_H + middle)),
        "up": (laid.first_group > 0, (x + theme.PAD + theme.ARROW - 1, top + air)),
        "down": (laid.first_group + len(laid.rows) < groups,
                 (x + BUTTON_H + theme.PAD + theme.ARROW - 1, foot + air + theme.ARROW - 1)),
    }
    return {way: tip for way, (out, tip) in tips.items() if out}


def picture(rect: Rect, intro: Sequence[float], first: float, tail: float) -> tuple[list[Rect], Rect, Rect]:
    """Where an intro and the start of the loop are drawn in a rectangle, along one line of time.

    `intro` has the rows of each pattern the intro plays, `first` those of the block the loop begins with, and
    `tail` the rows the intro runs on for. Given are the block of each pattern of the intro, the block of the
    loop after them, and the stretch of the tail, which begins where the loop does: the whole is as wide as
    the rectangle. With no rows there is nothing to draw, and every part has no width.
    """
    x, y, w, h = rect
    before = sum(intro)
    total = before + max(first, tail)
    if w <= 0 or h <= 0 or total <= 0:
        return [], (x, y, 0, 0), (x, y, 0, 0)

    def at(rows: float) -> int:
        return x + math.floor(w * rows / total + 0.5)

    edges = [at(rows) for rows in accumulate(intro, initial=0.0)]
    start = edges[-1]
    return ([(left, y, right - left, h) for left, right in zip(edges, edges[1:])],
            (start, y, at(before + first) - start, h), (start, y, at(before + tail) - start, h))


class Table(Widget):
    """The groups of a song's layers down the left and its versions along the top, with a switch where they meet.

    `groups` and `versions` are what `show` gave it, and `heard` the names of the versions that are heard alone
    just now, which the owner sets: their buttons are lit. A group is named with its channels after its name,
    each a small square in the channel's colour and the channel's name. The table saves nothing and plays
    nothing: it tells its owner what the hand asked for, each group and version by its name. A click on a
    switch calls `flipped(group, version)`, one on the button of a heading `played(version)`, and one on a
    button that adds `added(kind)`, the kind "group" or "version". The right button on a group or on a heading
    calls `asked(kind, name, pos)`. The help line says what is under the mouse.

    A table too small for all its groups or all its versions shows them from `first_group` and from
    `first_version` on, which are places among all there are. The wheel over the headings moves the versions
    by a column a step, and anywhere else on the table the groups by a row. A small arrow marks each way in
    which there are more, as `more` has them. What is not in view cannot be pressed.
    """

    def __init__(self, rect: Rect, flipped: Callable[[str, str], object] | None = None,
                 played: Callable[[str], object] | None = None, added: Callable[[str], object] | None = None,
                 asked: Callable[[str, str, Pos], object] | None = None) -> None:
        super().__init__(rect)
        self.flipped, self.played, self.added, self.asked = flipped, played, added, asked
        self.groups: tuple[Group, ...] = ()
        self.versions: tuple[Version, ...] = ()
        self.heard: frozenset[str] = frozenset()
        self.first_group = self.first_version = 0
        # How wide the widest name of a group is, and the widest group with its channels after it.
        self._name_w = self._label = 0
        # What the mouse is on, and what the left button went down on.
        self._over: tuple | None = None
        self._pressed: tuple | None = None

    @property
    def laid(self) -> Laid:
        """Where the parts of the table stand now, as `lay` has them."""
        return lay(self.rect, len(self.groups), len(self.versions), self._label, self.first_group, self.first_version)

    def show(self, groups: Sequence[Group], versions: Sequence[Version]) -> None:
        """Take the groups and the versions to show, in the order the song has them.

        The first group and the first version in view stay, or go back as far as it takes to be in range.
        """
        self.groups, self.versions = tuple(groups), tuple(versions)
        self._name_w = max((Font.width(group.name) for group in self.groups), default=0)
        after = max((sum(theme.LIST_SPACE + theme.LIST_SQUARE + theme.GAP + Font.width(name)
                         for name, _ in group.channels) for group in self.groups), default=0)
        self._label = 2 * theme.PAD + self._name_w + after
        self._settle()

    def _settle(self) -> None:
        """Keep the first group and the first version in view to what the table has room for, as `lay` keeps them."""
        laid = self.laid
        self.first_group, self.first_version = laid.first_group, laid.first_version

    def _place(self, kind: str, name: str) -> int | None:
        """The place a group or a version in view has among all there are, by its name. None for one not in view."""
        laid = self.laid
        held, first, seen = ((self.groups, laid.first_group, len(laid.rows)) if kind == "group"
                             else (self.versions, laid.first_version, len(laid.heads)))
        return next((at for at, found in enumerate(held[first:first + seen], first) if found.name == name), None)

    def cell(self, group: str, version: str) -> Rect:
        """The switch of a group and a version, by their names. One that is not in view has no width."""
        row, column = self._place("group", group), self._place("version", version)
        return (self.rect[0], self.rect[1], 0, 0) if row is None or column is None else cell(self.laid, row, column)

    def play(self, version: str) -> Rect:
        """The button that plays a version, by its name. One that is not in view has no width."""
        at, laid = self._place("version", version), self.laid
        return (self.rect[0], self.rect[1], 0, 0) if at is None else head_parts(laid.heads[at - laid.first_version])[0]

    def name_rect(self, kind: str, name: str) -> Rect:
        """Where the name of a group or of a version is drawn, for a box to go over. The table for one not in view."""
        at, laid = self._place(kind, name), self.laid
        if at is None:
            return self.rect
        if kind == "version":
            return head_parts(laid.heads[at - laid.first_version])[1]
        x, y, _, _ = laid.rows[at - laid.first_group]
        return x + theme.PAD, y, self._name_w, ROW_H - 1

    def _words(self, found: tuple | None) -> str:
        """The help line for what the mouse is on."""
        if found is None:
            return "the groups of channels down the side, the versions along the top: a switch is a group in a version"
        if found[0] == "add":
            return f"a new {found[1]}"
        if found[0] == "group":
            return f"group {self.groups[found[1]].name}: the right button renames it, removes it or picks its channels"
        version, _, ahead = self.versions[found[-1]]
        if found[0] == "play":
            return f"hear version {version} alone: the channels of its groups and no others"
        if found[0] == "version":
            mixed = f", premixed in {', '.join(map(key_text, ahead))}" if ahead else ""
            return f"version {version}{mixed}: the right button renames or removes it, or premixes it in a key"
        return f"group {self.groups[found[1]].name} in version {version}: a click puts it in or takes it out"

    def _wheel(self, event: Event) -> bool:
        """Move what is in view by the steps of the wheel, up being towards the first. Whether there was a use for it.

        Over the headings it is the versions that move, by a column a step, and anywhere else the groups, by a
        row. Where all of them are in view the wheel is not used, and is left to what lies under the table.
        """
        laid = self.laid
        x, y, w, _ = self.rect
        across = inside((x, y, w, ROW_H), event.pos)
        count, seen = (len(self.versions), len(laid.heads)) if across else (len(self.groups), len(laid.rows))
        if seen >= count:
            return False
        if across:
            self.first_version = laid.first_version - event.wheel
        else:
            self.first_group = laid.first_group - event.wheel
        self._settle()
        return True

    def on(self, event: Event, ui: Ui) -> bool:
        """Use the wheel, a press of the left button on a switch or a button, and of the right one on a name.

        The name is that of a group, or the heading of a version.
        """
        if event.kind in ("key", "text"):
            return False
        moved = event.kind == "wheel" and self._wheel(event)
        found = self._over = hit(self.laid, event.pos)
        self.help = self._words(found)
        if event.kind == "wheel":
            return moved
        if event.kind == "down" and event.button == 3:
            named = found is not None and found[0] in ("group", "version", "play")
            if named and self.asked is not None:
                kind = "group" if found[0] == "group" else "version"
                self.asked(kind, (self.groups if kind == "group" else self.versions)[found[1]].name, event.pos)
            return named
        if event.kind == "down":
            self._pressed = found if event.button == 1 else None
            return self._pressed is not None
        if event.kind == "up" and event.button == 1:
            pressed, self._pressed = self._pressed, None
            if ui.held is self and found is not None and found == pressed:
                self._act(found)
        return False

    def _act(self, found: tuple) -> None:
        """Tell the owner what a click was on."""
        told = {"cell": self.flipped, "play": self.played, "add": self.added}.get(found[0])
        if told is None:
            return
        if found[0] == "cell":
            told(self.groups[found[1]].name, self.versions[found[2]].name)
        else:
            told(found[1] if found[0] == "add" else self.versions[found[1]].name)

    def draw(self, canvas: Canvas) -> None:
        """Draw the headings with their buttons, the groups with their channels, the switches and the two buttons.

        They are those in view. Each way in which there are more has its small arrow.
        """
        laid = self.laid
        over = self._over if self.hot else None
        down = over if self.held and over == self._pressed else None
        groups = self.groups[laid.first_group:laid.first_group + len(laid.rows)]
        versions = self.versions[laid.first_version:laid.first_version + len(laid.heads)]
        with canvas.clip(self.rect):
            for at, (head, version) in enumerate(zip(laid.heads, versions), laid.first_version):
                button, name = head_parts(head)
                lit = version.name in self.heard
                draw_button(canvas, button, icon="play", ink=theme.ACTIVE if lit else theme.TEXT,
                            hot=over == ("play", at), pressed=down == ("play", at),
                            strip=theme.ACTIVE if lit else None)
                canvas.text_in(version.name, name, theme.TEXT)
            for at, (row, group) in enumerate(zip(laid.rows, groups), laid.first_group):
                x, y, w, h = row
                canvas.box((x, y, w - 1, h - 1), theme.LIT if over == ("group", at) else theme.RAISED)
                canvas.text_in(group.name, (x + theme.PAD, y, self._name_w, h - 1), theme.TEXT)
                left = x + theme.PAD + self._name_w
                with canvas.clip((x, y, w - 1 - theme.PAD, h - 1)):
                    for name, colour in group.channels:
                        left += theme.LIST_SPACE
                        canvas.box((left, y + (h - 1 - theme.LIST_SQUARE) // 2, theme.LIST_SQUARE,
                                    theme.LIST_SQUARE), colour)
                        left += theme.LIST_SQUARE + theme.GAP
                        canvas.text_in(name, (left, y, Font.width(name), h - 1), theme.DIM)
                        left += Font.width(name)
                for to, version in enumerate(versions, laid.first_version):
                    box = cell(laid, at, to)
                    canvas.box(box, theme.ACTIVE if group.name in version.groups else theme.BACKGROUND)
                    canvas.frame(box, theme.TEXT if over == ("cell", at, to) else theme.LINE)
            for kind, button in (("version", laid.add_version), ("group", laid.add_group)):
                if button[2] > 0:
                    draw_button(canvas, button, icon="plus", hot=over == ("add", kind), pressed=down == ("add", kind))
            for way, (x, y) in more(laid, len(self.groups), len(self.versions)).items():
                # The arrow widens from its tip, against the way it points.
                back = -1 if way in ("right", "down") else 1
                for step in range(theme.ARROW):
                    if way in ("left", "right"):
                        canvas.vline(x + back * step, y - step, 2 * step + 1, theme.DIM)
                    else:
                        canvas.hline(x - step, y + back * step, 2 * step + 1, theme.DIM)


class Picture(Widget):
    """A small picture of an intro: its patterns as blocks before the block the loop begins with.

    The tail is drawn as a slope that falls from where the loop begins, over as much of the loop as the intro
    runs on for. `names` and `rows` are the patterns the intro plays and how long each is, `first` and
    `first_rows` the pattern the loop begins with, and `tail` the rows of the tail: the owner sets them. It
    takes nothing from the mouse.
    """

    def __init__(self, rect: Rect) -> None:
        super().__init__(rect)
        self.names: tuple[str, ...] = ()
        self.rows: tuple[float, ...] = ()
        self.first = ""
        self.first_rows = 0.0
        self.tail = 0.0

    def hit(self, pos: Pos) -> bool:
        """Never: the picture takes nothing from the mouse."""
        return False

    def draw(self, canvas: Canvas) -> None:
        """Draw the blocks of the intro, the block of the loop, the slope of the tail over it, and the names."""
        blocks, loop, tail = picture(self.rect, self.rows, self.first_rows, self.tail)
        with canvas.clip(self.rect):
            canvas.box(loop, theme.BACKGROUND)
            x, y, w, h = tail
            for column in range(w):
                high = math.floor((h - 2) * (1 - column / w) + 0.5)
                canvas.vline(x + column, y + h - 1 - high, high, theme.shade(theme.ACTIVE, theme.TRAVEL))
                canvas.box((x + column, y + h - 1 - high, 1, 1), theme.ACTIVE)
            canvas.frame(loop, theme.LINE)
            canvas.text_in(self.first, (loop[0] + theme.PAD, loop[1] + 1, loop[2] - 2 * theme.PAD, Font.LINE),
                           theme.DIM, "right")
            for name, block in zip(self.names, blocks):
                x, y, w, h = block
                canvas.box(block, theme.RAISED)
                canvas.box((x, y + h - theme.STRIP, w, theme.STRIP), theme.ACTIVE)
                canvas.frame(block, theme.LINE)
                canvas.text_in(name, (x + theme.PAD, y + 1, w - 2 * theme.PAD, Font.LINE), theme.TEXT)


def arrange(rect: Rect, groups: int | None) -> dict[str, Rect]:
    """Where the parts of the layers go in their rectangle, each by what it is called, and each cut to it.

    `groups` is how many groups the song's layers have, and None for a song with none. Along the top stand
    "use", the switch of the layers, then "all", and after a gap "keys_word", "keys", "key_add" and "key_drop".
    Under the switch "about" is the line a song with no layers has, and "table" is the table of one that has
    them, as high as its groups need and no higher than leaves the intro its room. "intro" is the room of the
    intro, under whichever of the two there is.
    """
    x, y, w, h = rect
    use, every, word = (Font.width(text) + 2 * theme.PAD for text in (USE, ALL, KEYS))
    at = x + use + SPACE + every + theme.GROUP_GAP
    top = y + BUTTON_H + SPACE
    high = Font.LINE if groups is None else max(0, min((groups + 2) * ROW_H, y + h - top - APART - HEIGHT))
    placed = {
        "use": (x, y, use, BUTTON_H),
        "all": (x + use + SPACE, y, every, BUTTON_H),
        "keys_word": (at, y, word, BUTTON_H),
        "keys": (at + word, y, KEYS_W, BUTTON_H),
        "key_add": (at + word + KEYS_W + theme.BUTTON_GAP, y, BUTTON_H, BUTTON_H),
        "key_drop": (at + word + KEYS_W + 2 * theme.BUTTON_GAP + BUTTON_H, y, BUTTON_H, BUTTON_H),
        "about": (x, top, w, Font.LINE),
        "table": (x, top, w, high),
        "intro": (x, top + high + APART, w, HEIGHT),
    }
    return {part: within(place, rect) for part, place in placed.items()}


def intro_parts(room: Rect) -> dict[str, Rect]:
    """Where the parts of the intro go in the room it has, each by what it is called, and each cut to the room.

    From the left stand "switch", then "order" with "heading" over it, the knobs "tail" and "fade", and
    "picture", which takes what is left up to `PICTURE_MOST`. A part the room is too narrow for has no width.
    """
    x, y, w, h = room
    top = y + (h - BUTTON_H) // 2
    switch = Font.width(INTRO) + 2 * theme.PAD
    at = x + switch + SPACE
    placed = {"switch": (x, top, switch, BUTTON_H), "order": (at, top, ORDER_W, BUTTON_H),
              "heading": knob_parts((at, y, ORDER_W, h))[0]}
    at += ORDER_W + SPACE
    for part in ("tail", "fade"):
        placed[part] = (at, y, LENGTH_W, h)
        at += LENGTH_W + SPACE
    placed["picture"] = (at, y + PICTURE_AIR, min(PICTURE_MOST, x + w - at), h - 2 * PICTURE_AIR)
    return {part: within(place, room) for part, place in placed.items()}


class Layers:
    """The layers and the intro of a song, seen, heard and changed: what stands under the strip when LAYERS is chosen.

    It is no panel of the frame. A song panel owns it, adds its widgets through itself, gives it a rectangle
    with `place`, a song with `show` and a frame with `update`, and says with `seen` whether it is on the
    window. `use` is the switch of the layers. A song with none has it off, and `about` under it says what
    layers are for: switched on, the song has one group with all its channels and one version of that group.
    Switched off, all of the layers go.

    A song with layers has `table`: its groups against its versions, with a switch for each pair. The buttons
    after the last heading and under the last group add a version and a group. The right button on either
    opens a menu to name it anew or remove it, and for a group to pick its channels from those of the song: a
    channel is in one group, so one that is picked leaves the group it was in. The menu of a version has each
    key of the song as well, to have the version premixed in it: one more file at the build. Groups and
    versions the table has no room for are brought into view by the wheel.

    The button in the heading of a version has that version heard alone: the channels of its groups are the
    soloed channels of the session, so it is the sound the solos of the mixer give, with nothing rendered
    again, and the song plays if it did not. The button of another version changes over as the song plays, and
    `all` has every channel heard again. `keys` shows the keys the song is built in, `key_add` adds one that is
    typed, and `key_drop` removes the one that is shown. The song is heard as it is written whichever is shown.

    `intro` is the switch of the intro, with or without layers. While it is on, `order` shows the patterns it
    plays and opens a menu to add one or take the last out, `tail` and `fade` are its two lengths, and
    `picture` draws it before the loop.

    Every change is the whole song saved as `layeredit` leaves it: one change to the file, and one step that
    can be taken back, called by the name of the song and what was done. What `layeredit` or the save refuses
    changes nothing, and the session says why. After each change, and whenever the model has loaded the file
    again, the widgets show what the file holds. While a knob is under the hand nothing is read, and what
    changed outside meanwhile shows when the hand has let go.
    """

    def __init__(self, panel: Panel, clock: Callable[[], float] = time.perf_counter) -> None:
        """`panel` is the panel the widgets are added through. `clock` times a run of the wheel on a knob."""
        self.session = panel.session
        self._ui = panel.ui
        add, nowhere = panel.add, (0, 0, 0, 0)
        self.use = add(Toggle(nowhere, USE, changed=self._use,
                              help="the channels sorted into groups, and versions that are each a choice of groups"))
        self.about = add(Label(nowhere, ABOUT))
        self.all = add(Button(nowhere, ALL, action=self._all, help="hear every channel again, and no version alone"))
        self.keys_word = add(Label(nowhere, KEYS))
        self.keys = add(Stepper(nowhere, ("",), help="the keys the song is built in, as semitones up: it is heard "
                                                       "here as it is written"))
        self.key_add = add(Button(nowhere, icon="plus", action=self._ask_key,
                                  help="one more key: type how many semitones up the song is built as well"))
        self.key_drop = add(Button(nowhere, icon="minus", action=self._drop_key))
        self.table = add(Table(nowhere, self._flip, self._play, self._add, self._ask))
        self.intro = add(Toggle(nowhere, INTRO, changed=self._switch,
                                help="patterns that are played once before the song goes round"))
        self.heading = add(Label(nowhere, PLAYS, align="centre"))
        self.order = add(Button(nowhere, action=self._ask_order,
                                help="the patterns the intro plays, in turn: a click adds one or takes the last out"))
        self.tail = add(Knob(nowhere, _TAIL, 0.0, _TAIL, default=_TAIL, text=TAIL, show=seconds_text, clock=clock,
                             help="how long the intro rings on past its last pattern: a bar of the song at the most",
                             settled=lambda value: self._put(layeredit.intro_tail, "tail", value)))
        low, high, steps = FADES
        self.fade = add(Knob(nowhere, _FADE, low, high, default=_FADE, steps=steps, text=FADE, show=seconds_text,
                             clock=clock, help="how long the very end of the intro takes to fall away to nothing",
                             settled=lambda value: self._put(layeredit.intro_fade, "fade", value)))
        self.picture = add(Picture(nowhere))
        #: The name of the song that is shown.
        self.song: str | None = None
        # The project the widgets were set from last, and what it holds for the song: None for no such song.
        self._project: Project | None = None
        self._song: dict | None = None
        # The keys of the song, and the channels each version is heard by, by its name.
        self._keys: list[int] = [0]
        self._channels: dict[str, frozenset[str]] = {}
        self._rect: Rect = nowhere
        self._on = False
        self._lay()

    @property
    def widgets(self) -> list[Widget]:
        """Every widget of the part, in the order they were added."""
        return [self.use, self.about, self.all, self.keys_word, self.keys, self.key_add, self.key_drop, self.table,
                self.intro, self.heading, self.order, self.tail, self.fade, self.picture]

    @property
    def busy(self) -> bool:
        """Whether a knob is under the hand: dragged, or in a run of the wheel that has not ended."""
        return self.tail.busy or self.fade.busy

    def place(self, rect: Rect) -> None:
        """Take the part's rectangle, and lay the widgets out in it as `arrange` and `intro_parts` have them."""
        self._rect = rect
        self._lay()

    def seen(self, on: bool) -> None:
        """Bring the part onto the window, or take it off. On it, a widget is seen by what there is to show."""
        self._on = on
        self._lay()

    def show(self, song: str | None) -> None:
        """Show the layers and the intro of a song of the project, by its name. None, or no such song, shows none.

        The song that is shown already is read again only where the model has loaded its file again, and not
        while a knob is under the hand. Another song takes the part at once, and what the hand was doing to a
        knob is dropped, with nothing saved.
        """
        if song != self.song:
            self.tail.drop()
            self.fade.drop()
            self.song = song
            self._read()
        elif not self.busy and self.session.model.project is not self._project:
            self._read()

    def update(self) -> None:
        """Give the knobs their frame, read the song again where the file was loaded again, and light what is heard.

        The frame of a knob ends a run of the wheel whose time is up, which saves it. While a knob is under the
        hand nothing is read. The button of a version is lit while the channels that are heard are its own.
        """
        self.tail.update()
        self.fade.update()
        if not self.busy and self.session.model.project is not self._project:
            self._read()
        heard = self.session.model.heard
        self.table.heard = frozenset(name for name, channels in self._channels.items()
                                     if channels and channels == heard)

    def _read(self) -> None:
        """Set every widget from what the project holds for the song now, and lay them out."""
        project = self._project = self.session.model.project
        song = self._song = project.songs.get(self.song)
        layers = (song or {}).get("layers")
        intro = (song or {}).get("intro")
        self.use.is_on, self.intro.is_on = layers is not None, intro is not None
        shown = self._keys[self.keys.picked] if self.keys.picked < len(self._keys) else None
        groups: list[Group] = []
        versions: list[Version] = []
        self._keys, self._channels = [0], {}
        if layers is not None:
            colours = {name: theme.CHANNELS[place % len(theme.CHANNELS)] for place, name in enumerate(song["channels"])}
            groups = [Group(name, tuple((channel, colours[channel]) for channel in members))
                      for name, members in layers["groups"].items()]
            ahead = layers.get("premix", [])
            versions = [Version(name, frozenset(named), tuple(mix["key"] for mix in ahead if mix["version"] == name))
                        for name, named in layers["versions"].items()]
            self._channels = {name: frozenset(channel for group in named for channel in layers["groups"][group])
                              for name, named in layers["versions"].items()}
            self._keys = layeredit.keys(song)
        self.table.show(groups, versions)
        self.keys.options = tuple(key_text(key) for key in self._keys)
        self.keys.picked = self._keys.index(shown) if shown in self._keys else 0
        self.key_drop.enabled = len(self._keys) > 1
        self.key_drop.help = ("the song is no longer built in the key that is shown" if self.key_drop.enabled
                              else "the only key of a song stays")
        if intro is not None:
            self._read_intro(song, intro)
        self._lay()

    def _read_intro(self, song: dict, intro: dict) -> None:
        """Set the widgets of the intro from what the song holds for it, and its picture from the rows it plays."""
        self.order.text = " ".join(intro["order"])
        bar = max(layeredit.bar_seconds(song), 0.01)
        tail = intro.get("tail", _TAIL)
        self.tail.high, self.tail.steps, self.tail.default = bar, max(1, round(bar * 100)), min(_TAIL, bar)
        self.tail.value, self.fade.value = tail, intro.get("fade", _FADE)
        first = song["order"][0] if song["order"] else ""
        try:
            read = self._project.song(self.song)
            rows = {name: songview.pattern(read, name).rows for name in {*intro["order"], first} if name}
        except RowSynthError:
            # A song the engine refuses has no rows to draw. The song panel says what is wrong with it.
            rows = {}
        drawn = self.picture
        drawn.names = tuple(intro["order"]) if rows else ()
        drawn.rows = tuple(rows[name] for name in drawn.names)
        drawn.first, drawn.first_rows = first, rows.get(first, 0.0)
        drawn.tail = tail * TICK_HZ / song["row_ticks"] if rows and song["row_ticks"] > 0 else 0.0

    def _lay(self) -> None:
        """Give each widget its rectangle, and have it seen by whether the part is on the window and what there is.

        The switches are seen with any song. A song with no layers has the line that says what they are for,
        and one with layers the table and what stands beside the switch. The settings of the intro and its
        picture are seen while the intro is on. A widget the part has no room for is not seen.
        """
        song = self._on and self._song is not None
        layered, intro = song and self.use.is_on, song and self.intro.is_on
        parts = arrange(self._rect, len(self.table.groups) if self.use.is_on else None)
        inner = intro_parts(parts["intro"])
        placed = [(self.use, parts["use"], song), (self.about, parts["about"], song and not layered),
                  (self.intro, inner["switch"], song)]
        placed += [(widget, parts[part], layered) for part, widget in (
            ("all", self.all), ("keys_word", self.keys_word), ("keys", self.keys), ("key_add", self.key_add),
            ("key_drop", self.key_drop), ("table", self.table))]
        placed += [(widget, inner[part], intro) for part, widget in (
            ("heading", self.heading), ("order", self.order), ("tail", self.tail), ("fade", self.fade),
            ("picture", self.picture))]
        for widget, rect, on in placed:
            widget.rect, widget.visible = rect, bool(on) and rect[2] > 0 and rect[3] > 0

    def _change(self, make: Callable[[dict], dict], label: str | Callable[[dict], str]) -> bool:
        """Save the song as a rule of `layeredit` leaves it, as one step, and show what the file then holds.

        `make` is given the song as the project holds it now and gives the new one. `label` is what the step is
        called after the name of the song: words, or a function that makes them of the new song. The answer is
        whether it was saved. What `make` refuses is not saved, and the session is told the refusal. A save the
        project refuses is told by the session itself.
        """
        song = self.session.model.project.songs.get(self.song)
        saved = False
        if song is not None:
            try:
                made = make(song)
            except RowSynthError as error:
                self.session.say(str(error))
            else:
                words = label if isinstance(label, str) else label(made)
                saved = self.session.change(f"songs.{self.song}", made, f"{self.song}: {words}")
        self._read()
        return saved

    def _use(self, on: bool) -> None:
        """Save the song with layers, the least there can be, or with none."""
        if on:
            self._change(layeredit.layered, "layers on")
        else:
            self._change(layeredit.unlayered, "layers off")

    def _switch(self, on: bool) -> None:
        """Save the song with an intro, as `layeredit.intro_on` starts one, or with none."""
        if on:
            self._change(layeredit.intro_on, "intro on")
        else:
            self._change(layeredit.intro_off, "intro off")

    def _flip(self, group: str, version: str) -> None:
        """Save the song with a group in a version where the table shows it out of it, and out where it shows it in."""
        on = not any(held.name == version and group in held.groups for held in self.table.versions)
        self._change(lambda song: layeredit.switched(song, group, version, on),
                     f"group {group} {'in' if on else 'out of'} version {version}")

    def _add(self, kind: str) -> None:
        """Save the song with one more group or version, which has the next free name."""
        self._change(_ADDED[kind], lambda made: f"new {kind} {list(made['layers'][f'{kind}s'])[-1]}")

    def _play(self, version: str) -> None:
        """Have a version heard alone, and the song play if it does not go round already.

        The channels of the version's groups become the soloed channels of the session, one at a time: those
        to come first and then those to go, so that at no moment every channel is heard. A version with no
        channel has nothing to hear, which is said.
        """
        session, channels = self.session, self._channels.get(version, frozenset())
        if not channels:
            session.say(f"version {version} has no channel in its groups, so there is nothing to hear")
            return
        soloed = session.soloed
        for channel in (*sorted(channels - soloed), *sorted(soloed - channels)):
            session.solo(channel)
        if not (session.playing and session.loop):
            session.play()

    def _all(self) -> None:
        """Have every channel heard again."""
        self.session.hear_all()

    def _menu(self, items: Sequence[Item | None], at: Pos) -> None:
        """Open a menu at a place, over everything else. Each item stands for what is done when it is chosen."""
        self._ui.open(Menu(items, at, chosen=lambda act: act()))

    def _ask(self, kind: str, name: str, at: Pos) -> None:
        """Open the menu of a group or of a version, at the mouse. The only one of its kind cannot be removed.

        The menu of a version has an item for each key of the song after that, with a tick before it where
        the version is premixed in that key: the build then writes one more file, that version in that key
        already mixed. One that is chosen is put in where it was out, and taken out where it was in.
        """
        several = len(self.table.groups if kind == "group" else self.table.versions) > 1
        items: list[Item | None] = [Item("Rename...", lambda: self._rename(kind, name))]
        if kind == "group":
            items.append(Item("Channels...", lambda: self._pick(name, at)))
        items += [None, Item("Remove", lambda: self._change(lambda song: _REMOVED[kind](song, name),
                                                            f"{kind} {name} removed"), enabled=several)]
        if kind == "version":
            ahead = next((version.premix for version in self.table.versions if version.name == name), ())
            items += [None, *(Item(f"premix in {key_text(key)}", lambda key=key: self._premix(name, key),
                                   "tick" if key in ahead else "") for key in self._keys)]
        self._menu(items, at)

    def _premix(self, version: str, key: int) -> None:
        """Save the song with a version premixed in a key where the file has it not, and not where it has it."""
        def words(made: dict) -> str:
            on = {"version": version, "key": key} in made["layers"].get("premix", [])
            return f"premix of {version} in {key_text(key)} {'on' if on else 'off'}"

        self._change(lambda song: layeredit.premixed(song, version, key), words)

    def _rename(self, kind: str, old: str) -> None:
        """Open a box over the name of a group or of a version: enter gives it that name everywhere."""
        def done(new: str) -> None:
            if new != old:
                self._change(lambda song: _RENAMED[kind](song, old, new), f"{kind} {old} renamed to {new}")

        ask(self._ui, self.table.name_rect(kind, old), old, done, allowed=layeredit.in_name,
            help=f"the name of {kind} {old}: enter gives it the new one, escape leaves it")

    def _pick(self, group: str, at: Pos) -> None:
        """Open the list of the song's channels at a place, with a tick before each that is in a group.

        A channel that is chosen is put in the group, or taken out of it where it is in it. One that is in
        another group has that group's name after its own, and leaves it. The list opens again after each
        choice, as the file then has it, until a press beside it or escape.
        """
        song = self.session.model.project.songs.get(self.song) or {}
        groups = song.get("layers", {}).get("groups", {})
        if group not in groups:
            return
        owner = {channel: name for name, members in groups.items() for channel in members}

        def flip(channel: str) -> None:
            if owner.get(channel) == group:
                self._change(lambda song: layeredit.channel_taken(song, channel), f"{channel} out of group {group}")
            else:
                self._change(lambda song: layeredit.channel_put(song, channel, group), f"{channel} in group {group}")
            self._pick(group, at)

        items = [Item(channel if owner.get(channel) in (group, None) else f"{channel} ({owner[channel]})",
                      lambda channel=channel: flip(channel), "tick" if owner.get(channel) == group else "")
                 for channel in song["channels"]]
        if items:
            self._menu(items, at)

    def _ask_key(self) -> None:
        """Open a box over the keys to type a number of semitones into: enter adds it as a key, which is then shown."""
        def done(typed: str) -> None:
            try:
                key = int(typed)
            except ValueError:
                self.session.say(f"a key is a whole number of semitones, not {typed!r}")
                return
            if self._change(lambda song: layeredit.key_added(song, key), f"key {key_text(key)} added"):
                self.keys.picked = self._keys.index(key)

        ask(self._ui, self.keys.rect, "", done, allowed="+-0123456789".__contains__, most=3,
            help="how many semitones up the song is built as well, or down with a minus: enter adds the key")

    def _drop_key(self) -> None:
        """Save the song without the key that is shown."""
        key = self._keys[self.keys.picked]
        self._change(lambda song: layeredit.key_removed(song, key), f"key {key_text(key)} removed")

    def _ask_order(self) -> None:
        """Open the menu of the intro's patterns under their field: one more at the end, or the last taken out."""
        song = self.session.model.project.songs.get(self.song) or {}

        def step(text: str, patterns: Callable[[list[str]], list[str]], words: str, enabled: bool = True) -> Item:
            return Item(text, lambda: self._change(
                lambda song: layeredit.intro_order(song, patterns(song.get("intro", {}).get("order", []))), words),
                enabled=enabled)

        items: list[Item | None] = [step(f"Add {name} at the end", lambda held, name=name: [*held, name],
                                         f"{name} added to the intro") for name in song.get("patterns", {})]
        if items:
            x, y, _, h = self.order.rect
            self._menu([*items, None, step("Take the last out", lambda held: held[:-1],
                                           "last pattern taken out of the intro",
                                           len(song.get("intro", {}).get("order", [])) > 1)], (x, y + h))

    def _put(self, make: Callable[[dict, float], dict], word: str, value: float) -> None:
        """Save the length a knob of the intro was left at, written to two places and called by what it is."""
        value = tidy(value)
        self._change(lambda song: make(song, value), f"intro {word} {seconds_text(value)}")
