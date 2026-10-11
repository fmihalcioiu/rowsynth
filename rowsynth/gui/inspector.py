"""The right panel: what the picked thing measures, in plain words, and what the checks found in the project."""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, NamedTuple

from ..checks import Finding, songs, sounds
from ..edit import steps
from ..errors import RowSynthError
from . import theme
from .font import Font, wrap
from .layout import Rect
from .lists import Scroll
from .model import KINDS
from .panels import Panel, within
from .theme import Colour
from .ui import NOWHERE, Event, Pos, Ui, Widget, inside

if TYPE_CHECKING:
    from .canvas import Canvas
    from .session import Session

#: From a side of the panel to what it shows. The panel's line runs down its left side, and is this thick.
PAD = 6
EDGE = 1
#: A heading: from the top of its half of the panel to its line, and from there to the first line under it.
HEAD_TOP = 4
HEAD_H = 14
#: From the last line of the numbers to the line that parts them from the findings.
FOOT = 8
#: The least there is between a label and its value.
SPACE = 6
#: The air above the first line of a finding and below its last, and on each side of the line that parts the
#: findings of the picked thing from the rest.
AIR = 3
RULE_AIR = 3
#: How wide the icon before a finding is. Every icon is 9 by 9.
ICON = 9

#: The two headings.
MEASURED = "MEASURED"
FOUND = "FOUND"
#: What stands for a value that is not ready yet, and for findings the checks have not come to.
DOTS = "..."
#: A level with no sound in it, an offset or a step of no size, and no clipped sample.
SILENT = "silent"
NONE = "none"
NO = "no"
#: What is said where the checks found nothing, and where they could not read the project at all.
NOTHING = "nothing found"
REFUSED = "the checks could not run"
#: What a variant of a sound is called, which a target counts from 0 and the panel from 1.
VARIANT = "variant"

#: The numbers of a render in the order they are shown, each with its label.
LABELS = {"seconds": "LENGTH", "peak_db": "PEAK", "rms_db": "LOUDNESS", "dc": "OFFSET", "clipped": "CLIPPED",
          "head_step": "START", "tail_step": "END", "above_4k": "HIGHS", "centroid_hz": "BRIGHTNESS"}

#: The numbers the checks hold a thing of a kind to, each with what tells a value that breaks the rule. The
#: limits are the checks' own. A song goes round, so its first and its last sample are no steps out of silence.
#: A sound is brought to one peak before it is measured, so its peak says nothing. An instrument is held to
#: nothing until a sound or a song plays it.
_FAULTS: dict[str, dict[str, Callable[[float], bool]]] = {
    "song": {"peak_db": lambda value: value >= songs._PEAK_DB, "clipped": lambda value: value > 0},
    "sound": {"dc": lambda value: abs(value) >= sounds._DC,
              "head_step": lambda value: value >= sounds._STEP,
              "tail_step": lambda value: value >= sounds._STEP,
              "above_4k": lambda value: value > sounds._BRIGHT},
}

#: The colour of each kind of thing: the place of a finding is written in the colour of the thing it is in.
_INKS = {"song": theme.SONG, "sound": theme.SOUND, "instrument": theme.INSTRUMENT}

#: What the step after one of these is called: "parts[0]" is part 1, and "patterns.a" is pattern a.
_SAID = {"parts": "part", "patterns": "pattern"}


def worded(key: str, value: float) -> str:
    """One number of a render in plain words. `key` is what `Model.numbers` has it under.

    A level with no sound in it is "silent". An offset or a step of no size is "none", and no clipped sample is
    "no". The share above 4 kHz is in hundredths. A key that is none of the numbers is refused.
    """
    if key == "seconds":
        return f"{value:.2f} s"
    if key in ("peak_db", "rms_db"):
        return SILENT if value == -math.inf else f"{value:.1f} dB"
    if key == "dc":
        return f"{value:+.4f}" if value else NONE
    if key == "clipped":
        return f"{value} sample{'' if value == 1 else 's'}" if value else NO
    if key in ("head_step", "tail_step"):
        return f"{value:.3f}" if value else NONE
    if key == "above_4k":
        return f"{round(value * 100)}%"
    if key == "centroid_hz":
        return f"{value} Hz"
    raise RowSynthError(f"a render has no number called {key!r}")


def faulty(kind: str, key: str, value: float) -> bool:
    """Whether the checks would find fault with one number of a thing of a kind: "song", "sound" or "instrument".

    A number the checks have no limit for, in a thing of that kind, is never at fault.
    """
    breaks = _FAULTS.get(kind, {}).get(key)
    return breaks is not None and breaks(value)


def table(kind: str, numbers: Mapping[str, float] | None) -> list[tuple[str, str, Colour]]:
    """The numbers of a thing of a kind as they are shown: for each its label, its value in words, and a colour.

    The colour is of the value: that of what is wrong for one the checks would find fault with, and that of text
    for any other. None is for numbers that are not ready, and then every value is dim dots.
    """
    if numbers is None:
        return [(label, DOTS, theme.DIM) for label in LABELS.values()]
    return [(label, worded(key, numbers[key]), theme.FAULT if faulty(kind, key, numbers[key]) else theme.TEXT)
            for key, label in LABELS.items()]


def _path(target: str) -> list[str | int]:
    """The steps of a target, or none of them for one that is not written as a place is."""
    try:
        return steps(target)
    except RowSynthError:
        return []


def belongs(target: str) -> tuple[str, str] | None:
    """The thing the target of a finding is in, as (kind, name): "sounds.zaps[1]" is in the sound zaps.

    None for a target that is in no song, no sound and no instrument.
    """
    path = _path(target)
    if len(path) < 2 or path[0] not in KINDS or not isinstance(path[1], str):
        return None
    return KINDS[path[0]], path[1]


def where(target: str) -> str:
    """The place of a target in words: the name of its thing, and then each step into the thing, with commas.

    "sounds.zaps[1]" is "zaps, variant 2", "sounds.boom.parts[0]" is "boom, part 1" and
    "songs.tune.patterns.a.lead" is "tune, pattern a, lead". What a target counts from 0 is counted from 1. A
    target that is in no thing is said as it is written.
    """
    thing = belongs(target)
    if thing is None:
        return target

    def counted(step: str | int) -> str:
        return str(step + 1) if isinstance(step, int) else step

    words = [thing[1]]
    rest = iter(_path(target)[2:])
    for step in rest:
        held = next(rest, None) if step in _SAID else None
        if held is not None:
            words.append(f"{_SAID[step]} {counted(held)}")
        else:
            words.append(f"{VARIANT} {counted(step)}" if isinstance(step, int) else step)
    return ", ".join(words)


class Block(NamedTuple):
    """One thing in the list of findings: a finding, or the line that says the picked thing has none.

    `top` is how far its first line is below the top of all there is to show. `icon` stands before it. `head` is
    the lines beside the icon, the place of a finding in words, and `body` the lines under those, its message.
    `thing` is what a click on it picks, as (kind, name), or None for a block that picks nothing.
    """

    top: int
    icon: str
    head: tuple[str, ...]
    body: tuple[str, ...]
    thing: tuple[str, str] | None

    @property
    def bottom(self) -> int:
        """How far the end of its last line is below the top of all there is to show."""
        return self.top + (len(self.head) + len(self.body)) * Font.LINE


class Laid(NamedTuple):
    """Findings laid out at a width: the blocks from the top down, the line between them, and the height of it all.

    `rule` is how far the line that parts the findings of the picked thing from the rest is below the top. It is
    None where there is no rest, and so no line.
    """

    blocks: tuple[Block, ...]
    rule: int | None
    height: int


def lay(found: Sequence[Finding], picked: tuple[str, str] | None, width: int) -> Laid:
    """Where findings go in a column `width` pixels wide, their words broken into lines that fit it.

    The findings of the picked thing come first, in the order the checks give them. Where it has none, a line
    that says so stands in their place. The rest of the project's come after, under a line. Every block has
    `AIR` above its first line and below its last, and the line has `RULE_AIR` on each side of it.
    """
    own = [finding for finding in found if picked is not None and belongs(finding.target) == picked]
    rest = [finding for finding in found if finding not in own]

    def told(findings: list[Finding]) -> list[tuple[str, str, str, tuple[str, str] | None]]:
        return [("warn", where(finding.target), finding.message, belongs(finding.target)) for finding in findings]

    first, after = told(own) or [("tick", NOTHING, "", None)], told(rest)
    blocks: list[Block] = []
    rule, top = None, 0
    for index, (icon, place, message, thing) in enumerate(first + after):
        if after and index == len(first):
            rule = top + RULE_AIR
            top = rule + 1 + RULE_AIR
        block = Block(top + AIR, icon, tuple(wrap(place, width - ICON - theme.GAP)), tuple(wrap(message, width)),
                      thing)
        blocks.append(block)
        top = block.bottom + AIR
    return Laid(tuple(blocks), rule, top)


def arrange(rect: Rect) -> dict[str, Rect]:
    """Where the parts of the panel go in its rectangle, each by what it is called, and each cut to the panel.

    From the top: "measured" is the line of the upper heading, and "table" the lines of the numbers, which take
    what they need. "rule" is the line that parts the two halves, "found" the line of the lower heading, and
    "list" all that is left under it, for the findings.
    """
    x, y, w, h = rect
    left, wide = x + PAD, w - 2 * PAD
    table_top = y + HEAD_TOP + HEAD_H
    rule = table_top + len(LABELS) * Font.LINE + FOOT
    list_top = rule + HEAD_TOP + HEAD_H
    placed = {"measured": (left, y + HEAD_TOP, wide, Font.LINE),
              "table": (left, table_top, wide, len(LABELS) * Font.LINE),
              "rule": (x, rule, w, 1),
              "found": (left, rule + HEAD_TOP, wide, Font.LINE),
              "list": (x, list_top, w, y + h - list_top)}
    return {part: within(place, rect) for part, place in placed.items()}


class Findings(Widget):
    """What the checks found, as blocks of text one under the other. The wheel scrolls them when they do not fit.

    `found` is the findings to show, or None while the checks run: dim dots stand for them then. `picked` is the
    thing whose findings come first. Both are for the owner to set. A press of the left button on a finding
    calls `changed` with the thing the finding is in, as (kind, name). The rectangle is as wide as the panel.
    """

    def __init__(self, rect: Rect, changed: Callable[[tuple[str, str]], object] | None = None,
                 help: str = "") -> None:
        super().__init__(rect, help)
        self.found: tuple[Finding, ...] | None = None
        #: Whether there are no findings because the checks could not read the project, and not because they run.
        self.refused = False
        self.picked: tuple[str, str] | None = None
        self.changed = changed
        self.scroll = Scroll(row=Font.LINE)
        self._laid = Laid((), None, 0)
        #: What the blocks were last laid out of: the findings, the picked thing and the width.
        self._made_of: tuple | None = None
        self._mouse: Pos = NOWHERE

    @property
    def blocks(self) -> tuple[Block, ...]:
        """The blocks as they are laid out now, from the top down. There are none while the checks run."""
        self._settle()
        return self._laid.blocks

    def _column(self) -> tuple[int, int]:
        """Where the text begins from side to side, and how wide it can be."""
        x, _, w, _ = self.rect
        return x + PAD, max(0, w - 2 * PAD)

    def _settle(self) -> None:
        """Lay the blocks out anew, from their top, when what they were made of is no longer what there is.

        The offset is kept inside what there is to show.
        """
        made_of = (self.found, self.picked, self.rect[2])
        if made_of != self._made_of:
            self._made_of = made_of
            none = self.found is None
            self._laid = Laid((), None, 0) if none else lay(self.found, self.picked, self._column()[1])
            self.scroll.offset = 0
        self.scroll.content = self._laid.height
        self.scroll.view = self.rect[3]
        self.scroll.to(self.scroll.offset)

    def at(self, pos: Pos) -> Block | None:
        """The finding a position is on as they are scrolled now, or None when it is on none."""
        self._settle()
        if not inside(self.rect, pos):
            return None
        down = pos[1] - self.rect[1] + self.scroll.offset
        return next((block for block in self._laid.blocks
                     if block.thing is not None and block.top - AIR <= down < block.bottom + AIR), None)

    def on(self, event: Event, ui: Ui) -> bool:
        """Use the wheel, and a press of the left button on a finding."""
        if event.kind == "key":
            return False
        self._settle()
        self._mouse = event.pos
        if event.kind == "wheel":
            self.scroll.by(-theme.LIST_WHEEL * event.wheel)
            return True
        block = self.at(event.pos)
        if event.kind != "down" or event.button != 1 or block is None:
            return False
        if self.changed is not None:
            self.changed(block.thing)
        return True

    def draw(self, canvas: Canvas) -> None:
        """Draw the dots, or the blocks in view and the line between them: nothing outside the rectangle.

        The finding under the mouse is on a lit ground, which stays clear of the panel's line.
        """
        self._settle()
        x, y, w, h = self.rect
        left, room = self._column()
        lit = self.at(self._mouse) if self.hot else None
        with canvas.clip(self.rect):
            if self.found is None:
                said, colour = (REFUSED, theme.FAULT) if self.refused else (DOTS, theme.DIM)
                canvas.text_in(said, (left, y + AIR, room, Font.LINE), colour)
            for block in self._laid.blocks:
                top = y + block.top - self.scroll.offset
                high = block.bottom - block.top
                if top + high + AIR <= y or top - AIR >= y + h:
                    continue
                if block is lit:
                    canvas.box((x + EDGE, top - AIR, w - EDGE, high + 2 * AIR), theme.RAISED)
                self._draw_block(canvas, block, left, top, room)
            if self._laid.rule is not None:
                canvas.hline(left, y + self._laid.rule - self.scroll.offset, room, theme.LINE)

    def _draw_block(self, canvas: Canvas, block: Block, left: int, top: int, room: int) -> None:
        """Draw one block with its first line at `top`: its icon, its head beside the icon, its body under them."""
        good = block.icon == "tick"
        _, high = canvas.icon_size(block.icon)
        canvas.icon(block.icon, left, top + (Font.LINE - high) // 2, theme.GOOD if good else theme.FAULT)
        kind = block.thing[0] if block.thing else ""
        ink = theme.GOOD if good else _INKS.get(kind, theme.TEXT)
        indent = ICON + theme.GAP
        for line in block.head:
            canvas.text_in(line, (left + indent, top, max(0, room - indent), Font.LINE), ink)
            top += Font.LINE
        for line in block.body:
            canvas.text_in(line, (left, top, room, Font.LINE), theme.TEXT)
            top += Font.LINE


class Inspector(Panel):
    """The panel at the right: the numbers of the picked thing, and under them what the checks found.

    `values` is the table as it is shown now, as `table` gives it. `list` shows the findings: a click on one
    picks the thing it is in, in the session.
    """

    def __init__(self, session: Session, ui: Ui) -> None:
        super().__init__(session, ui)
        self.values = table("", None)
        self.list = self.add(Findings(self.rect, changed=lambda thing: session.pick(*thing),
                                      help="click a finding to go to the thing it is in"))
        self._places = arrange(self.rect)

    def place(self, rect: Rect) -> None:
        """Take the panel's rectangle: the numbers take what they need of it, and the findings the rest."""
        super().place(rect)
        self._places = arrange(rect)
        self.list.rect = self._places["list"]

    def update(self, passed: float = 0.0) -> None:
        """Show the numbers of the picked thing and the findings as the session has them now."""
        session = self.session
        picked = session.model.picked
        self.values = table(picked[0] if picked else "", session.numbers())
        # Findings that are not there yet are not asked for: the asking would wait for them. Nor are those of
        # checks that could not read the project: there are none, and the asking would raise their refusal.
        refused = session.findings_refused
        ready = session.finding_count() is not None and not refused
        self.list.found = tuple(session.model.findings()) if ready else None
        self.list.refused = refused
        self.list.picked = picked

    def draw(self, canvas: Canvas) -> None:
        """Draw the two headings, the table of the numbers, and the line between the two halves."""
        places = self._places
        x, y, w, _ = places["table"]
        canvas.text_in(MEASURED, places["measured"], theme.DIM)
        with canvas.clip(places["table"]):
            for row, (label, value, colour) in enumerate(self.values):
                top = y + row * Font.LINE
                canvas.text_in(value, (x, top, w, Font.LINE), colour, "right")
                # A label too long for the room before its value is cut there.
                canvas.text_in(label, (x, top, max(0, w - Font.width(value) - SPACE), Font.LINE), theme.DIM)
        canvas.box(places["rule"], theme.LINE)
        canvas.text_in(FOUND, places["found"], theme.DIM)
