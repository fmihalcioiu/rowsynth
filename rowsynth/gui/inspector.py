"""The right panel: what the picked thing measures, in plain words, and what the checks found in the project."""
from __future__ import annotations

import math
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from ..checks import Finding, songs, sounds
from ..commands import load_file
from ..commands.measure import _numbers
from ..edit import steps
from ..errors import RowSynthError
from ..wav import load_wav
from . import fixes, theme
from .fixes import Fix
from .font import Font, wrap
from .kit import ALONG, ScrollBar, told, wheel
from .layout import Rect
from .lists import Scroll
from .model import KINDS
from .panels import Panel, within
from .popups import ENTER
from .theme import Colour
from .ui import NOWHERE, Button, Event, Pos, Ui, Widget, draw_button, inside

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
#: What stands on the button of a finding that has a safe repair, and on the button that makes every such repair.
FIX = "FIX"
FIX_ALL = "FIX ALL"
#: How wide the two buttons are, and how high. That of a finding stands at the right of the first line of its
#: place and begins a pixel above it. The other stands at the right of the lower heading.
FIX_W = Font.width(FIX) + 2 * theme.PAD
FIX_ALL_W = Font.width(FIX_ALL) + 2 * theme.PAD
FIX_H = Font.LINE + 2
#: What the thread is called on which the repairs are worked out.
WORKER = "rowsynth-fixes"
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

#: What the findings say of themselves, what the bar beside them says, and what the button says that makes every
#: safe repair.
SCROLLS = "go through what was found"
LIST_HELP = told("go to the thing a finding is in", wheel=SCROLLS)
BAR_HELP = told("drag the bar to go through what was found, or press beside it to jump", wheel=SCROLLS)
ALL_HELP = told("make every repair that is safe, each a step of its own")

#: What the row of the join of a loop is called. What it says of a loop that comes round clean, and of one the
#: rule finds a step in where the rule's own words name no number.
JOIN = "LOOP JOIN"
CLEAN = "clean"
CLICKS = "clicks"
#: The step in what a "join" finding says: the number after the word it is told by.
_STEP = re.compile(r" by ([0-9][0-9.e+-]*[0-9]|[0-9])")

#: The two rows that stand over the numbers of a WAV file: its samples a second, and its one side or its two.
RATE = "RATE"
SIDES = "CHANNELS"
MONO, STEREO = "mono", "stereo"
#: The window the numbers of a WAV file are shown in: how wide it is, from a side of it to what it shows, and
#: between its parts. Its one button, how wide that is and how high, and what the window and the button say.
WAV_W = 190
WAV_PAD = 8
WAV_GAP = 6
CLOSE = "CLOSE"
CLOSE_W = Font.width(CLOSE) + 2 * theme.PAD + 8
CLOSE_H = 16
WAV_HELP = "the numbers of a WAV file, in the words of the numbers of a picked thing"
CLOSE_HELP = told("close the window, as escape and enter do")

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


def join_row(found: Sequence[Finding] | None) -> tuple[str, str, Colour]:
    """The row of the join of a loop as it is shown: its label, the step in words, and the colour of those.

    `found` is what the "join" rule of the checks finds where the loop goes round, as `Model.join` gives it.
    None is for a render that is not ready: dim dots. Nothing found is `CLEAN`. Anything found is the step
    the rule holds the loop to, as the rule's own words give the number, the largest of them where it found
    several, in the colour of what is wrong. Where its words name no number it is `CLICKS`.
    """
    if found is None:
        return JOIN, DOTS, theme.DIM
    if not found:
        return JOIN, CLEAN, theme.TEXT
    steps_found = [step[1] for step in (_STEP.search(finding.message) for finding in found) if step]
    return JOIN, max(steps_found, key=float, default=CLICKS), theme.FAULT


def measured(path: str | Path) -> list[tuple[str, str, Colour]]:
    """The numbers of a WAV file as they are shown: for each its label, its value in words, and a colour.

    They are the ones the `measure` command gives for the file, worked out as the command works them out: its
    rate, whether it has one side or two, and then the rows of `table`, measured on the mean of its sides. A
    file is held to no rule, so no value is in the colour of what is wrong. A file that is not there, cannot
    be read or is no WAV file the package reads is refused.
    """
    try:
        sides, rate = load_file(load_wav, path)
    except OSError as error:
        raise RowSynthError(f"{path}: {error.strerror or error}") from None
    given = _numbers(sides, rate, None, "file")
    return [(RATE, f"{given['rate']} Hz", theme.TEXT),
            (SIDES, STEREO if given["channels"] == 2 else MONO, theme.TEXT),
            *table("", {**given["measure"], **given["brightness"]})]


def wav_parts(size: tuple[int, int], rows: int) -> dict[str, Rect]:
    """Where the parts of the window of a WAV file go on an art surface of a size, for so many rows of numbers.

    "box" is the whole of it, in the middle of the surface and never larger than it. From the top stand
    "title", "name" and "table", each the whole width, and at the foot "close", at the right. Every part is
    cut to the box.
    """
    high = min(size[1], 2 * WAV_PAD + (rows + 2) * Font.LINE + 2 * WAV_GAP + CLOSE_H)
    wide = min(size[0], WAV_W)
    x, y = (size[0] - wide) // 2, (size[1] - high) // 2
    left, inner = x + WAV_PAD, wide - 2 * WAV_PAD
    top = y + WAV_PAD + 2 * Font.LINE + WAV_GAP
    placed = {"box": (x, y, wide, high),
              "title": (left, y + WAV_PAD, inner, Font.LINE),
              "name": (left, y + WAV_PAD + Font.LINE, inner, Font.LINE),
              "table": (left, top, inner, rows * Font.LINE),
              "close": (left + inner - CLOSE_W, top + rows * Font.LINE + WAV_GAP, CLOSE_W, CLOSE_H)}
    return {part: within(place, placed["box"]) for part, place in placed.items()}


class Measured(Widget):
    """The numbers of a WAV file in a small window of their own, opened as the modal widget.

    `name` is the name of the file, and `rows` its numbers as `measured` gives them. The one button closes
    the window, and so do escape, enter and a press off the window.
    """

    def __init__(self, name: str, rows: Sequence[tuple[str, str, Colour]]) -> None:
        super().__init__((0, 0, 0, 0), WAV_HELP)
        self.name = name
        self.rows = list(rows)
        self._mouse: Pos = NOWHERE
        # Whether the left press went down on the button, and its release has not come.
        self._down = False
        self._places = wav_parts(theme.DESIGN, len(self.rows))
        self.keep_in(theme.DESIGN)

    @property
    def help(self) -> str:
        """The line of the button with the mouse on it, and anywhere else the window's own."""
        return CLOSE_HELP if inside(self._places["close"], self._mouse) else self._help

    @help.setter
    def help(self, line: str) -> None:
        self._help = line

    def keep_in(self, size: tuple[int, int]) -> None:
        """Take the size of the art surface, and lay the window out in the middle of it."""
        self._places = wav_parts(size, len(self.rows))
        self.rect = self._places["box"]

    def on(self, event: Event, ui: Ui) -> bool:
        """Use everything: close on escape, on enter, on a press off the window and on a press of the button."""
        if event.kind == "key":
            if event.key == "escape" or event.key in ENTER:
                ui.close(self)
        elif event.kind != "text":
            self._mouse = event.pos
            over = inside(self._places["close"], event.pos)
            if event.kind == "down":
                self._down = over and event.button == 1
                if not self.hit(event.pos):
                    ui.close(self)
            elif event.kind == "up" and event.button == 1:
                pressed, self._down = self._down, False
                if pressed and over:
                    ui.close(self)
        return True

    def draw(self, canvas: Canvas) -> None:
        """Draw the box, the heading, the name of the file, its numbers as the panel draws them, and the button."""
        places = self._places
        x, y, w, _ = places["table"]
        with canvas.clip(self.rect):
            canvas.box(self.rect, theme.RAISED)
            canvas.frame(self.rect, theme.LINE)
            canvas.text_in(MEASURED, places["title"], theme.DIM)
            canvas.text_in(self.name, places["name"], theme.TEXT)
            with canvas.clip(places["table"]):
                for row, (label, value, colour) in enumerate(self.rows):
                    top = y + row * Font.LINE
                    canvas.text_in(value, (x, top, w, Font.LINE), colour, "right")
                    canvas.text_in(label, (x, top, max(0, w - Font.width(value) - SPACE), Font.LINE), theme.DIM)
            hot = inside(places["close"], self._mouse)
            draw_button(canvas, places["close"], CLOSE, hot=hot, pressed=hot and self._down)


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
    `thing` is what a click on it picks, as (kind, name), or None for a block that picks nothing. `fix` is the
    safe repair of its finding, which its button makes, or None for a block that has no button.
    """

    top: int
    icon: str
    head: tuple[str, ...]
    body: tuple[str, ...]
    thing: tuple[str, str] | None
    fix: Fix | None = None

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


def lay(found: Sequence[Finding], picked: tuple[str, str] | None, width: int,
        mends: Mapping[Finding, Fix] | None = None) -> Laid:
    """Where findings go in a column `width` pixels wide, their words broken into lines that fit it.

    The findings of the picked thing come first, in the order the checks give them. Where it has none, a line
    that says so stands in their place. The rest of the project's come after, under a line. Every block has
    `AIR` above its first line and below its last, and the line has `RULE_AIR` on each side of it. `mends` has
    the safe repair of each finding that has one: the place of such a finding leaves room for its button.
    """
    own = [finding for finding in found if picked is not None and belongs(finding.target) == picked]
    rest = [finding for finding in found if finding not in own]

    def told(findings: list[Finding]) -> list[tuple[str, str, str, tuple[str, str] | None, Fix | None]]:
        return [("warn", where(finding.target), finding.message, belongs(finding.target), (mends or {}).get(finding))
                for finding in findings]

    first, after = told(own) or [("tick", NOTHING, "", None, None)], told(rest)
    blocks: list[Block] = []
    rule, top = None, 0
    for index, (icon, place, message, thing, fix) in enumerate(first + after):
        if after and index == len(first):
            rule = top + RULE_AIR
            top = rule + 1 + RULE_AIR
        room = width - ICON - theme.GAP - (FIX_W + theme.GAP if fix else 0)
        block = Block(top + AIR, icon, tuple(wrap(place, room)), tuple(wrap(message, width)), thing, fix)
        blocks.append(block)
        top = block.bottom + AIR
    return Laid(tuple(blocks), rule, top)


def arrange(rect: Rect) -> dict[str, Rect]:
    """Where the parts of the panel go in its rectangle, each by what it is called, and each cut to the panel.

    From the top: "measured" is the line of the upper heading, and "table" the lines of the numbers, which take
    what they need: a line for each number and one more, for the join of a thing that has a loop. "rule" is
    the line that parts the two halves, "found" the line of the lower heading, and "list" all that is left
    under it, for the findings. "all" is the button at the right of the lower heading, which makes every safe
    repair, and "bar" the thin bar down the right side of the list, which shows how far it is scrolled.
    """
    x, y, w, h = rect
    left, wide = x + PAD, w - 2 * PAD
    table_top = y + HEAD_TOP + HEAD_H
    lines = len(LABELS) + 1
    rule = table_top + lines * Font.LINE + FOOT
    list_top = rule + HEAD_TOP + HEAD_H
    placed = {"measured": (left, y + HEAD_TOP, wide, Font.LINE),
              "table": (left, table_top, wide, lines * Font.LINE),
              "rule": (x, rule, w, 1),
              "found": (left, rule + HEAD_TOP, wide, Font.LINE),
              "all": (left + wide - FIX_ALL_W, rule + HEAD_TOP - 1, FIX_ALL_W, FIX_H),
              "list": (x, list_top, w, y + h - list_top),
              "bar": (x + w - theme.SCROLL_W, list_top, theme.SCROLL_W, y + h - list_top)}
    return {part: within(place, rect) for part, place in placed.items()}


class Findings(Widget):
    """What the checks found, as blocks of text one under the other. The wheel scrolls them when they do not fit:
    it is read by `kit.wheel`, and goes up and down with nothing held.

    `found` is the findings to show, or None while the checks run: dim dots stand for them then. `picked` is the
    thing whose findings come first. `mends` has the safe repair of each finding that has one, and such a
    finding has a button. All three are for the owner to set. A press of the left button on a finding calls
    `changed` with the thing the finding is in, as (kind, name). A press of a button, let go on it, calls `mend`
    with the repair. The rectangle is as wide as the panel.

    The help line is the one it was made with, and over a button what the repair of that button does.
    """

    def __init__(self, rect: Rect, changed: Callable[[tuple[str, str]], object] | None = None,
                 help: str = "", mend: Callable[[Fix], object] | None = None) -> None:
        super().__init__(rect, help)
        self.found: tuple[Finding, ...] | None = None
        #: Whether there are no findings because the checks could not read the project, and not because they run.
        self.refused = False
        self.picked: tuple[str, str] | None = None
        self.mends: Mapping[Finding, Fix] = {}
        self.changed = changed
        self.mend = mend
        self.scroll = Scroll(row=Font.LINE)
        self._laid = Laid((), None, 0)
        #: What the blocks were last laid out of: the findings, the picked thing, the width and the repairs.
        self._made_of: tuple | None = None
        self._mouse: Pos = NOWHERE
        #: The repair whose button is pressed and not let go of yet.
        self._pressed: Fix | None = None

    @property
    def help(self) -> str:
        """The one line of help: over the button of a repair, what the repair does."""
        fix = self.over(self._mouse)
        return self._help if fix is None else told(f"fix {fix.label}")

    @help.setter
    def help(self, line: str) -> None:
        self._help = line

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
        made_of = (self.found, self.picked, self.rect[2], self.mends)
        if made_of != self._made_of:
            self._made_of = made_of
            none = self.found is None
            self._laid = Laid((), None, 0) if none else lay(self.found, self.picked, self._column()[1], self.mends)
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

    def button(self, block: Block) -> Rect | None:
        """Where the button of a finding stands as the findings are scrolled now. None for one that has no repair."""
        if block.fix is None:
            return None
        left, room = self._column()
        return left + room - FIX_W, self.rect[1] + block.top - self.scroll.offset - 1, FIX_W, FIX_H

    def over(self, pos: Pos) -> Fix | None:
        """The repair whose button a position is on as the findings are scrolled now, or None when it is on none."""
        self._settle()
        if not inside(self.rect, pos):
            return None
        return next((block.fix for block in self._laid.blocks
                     if block.fix is not None and inside(self.button(block), pos)), None)

    def on(self, event: Event, ui: Ui) -> bool:
        """Use the wheel, a press of the left button on a finding, and one on a button that is let go there."""
        if event.kind == "key":
            return False
        self._settle()
        self._mouse = event.pos
        if event.kind == "wheel":
            way, steps_up = wheel(event)
            if way == ALONG:
                self.scroll.by(-theme.LIST_WHEEL * steps_up)
            return True
        if event.kind == "up" and event.button == 1:
            pressed, self._pressed = self._pressed, None
            if pressed is not None and self.over(event.pos) == pressed and self.mend is not None:
                self.mend(pressed)
            return False
        if event.kind != "down" or event.button != 1:
            return False
        self._pressed = self.over(event.pos)
        if self._pressed is not None:
            return True
        block = self.at(event.pos)
        if block is None:
            return False
        if self.changed is not None:
            self.changed(block.thing)
        return True

    def draw(self, canvas: Canvas) -> None:
        """Draw the dots, or the blocks in view and the line between them: nothing outside the rectangle.

        The finding under the mouse is on a lit ground, which stays clear of the panel's line. The button of a
        finding is lit under the mouse, and sunk while it is pressed there.
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
        button = self.button(block)
        if button is not None:
            hot = self.hot and inside(button, self._mouse)
            draw_button(canvas, button, FIX, hot=hot, pressed=hot and self.held and block.fix == self._pressed)


class Inspector(Panel):
    """The panel at the right: the numbers of the picked thing, and under them what the checks found.

    `values` is the table as it is shown now, as `table` gives it, and for a song and a held sound one row
    more, as `join_row` gives it: the join of its loop. `list` shows the findings: a click on one picks the
    thing it is in, in the session. `bar` is the thin bar beside them, which is there while they do not all
    fit and can be dragged.

    A finding that has a safe repair has a button that makes it, and `all` is the button that makes every one
    of them: it is there while two findings or more have a repair. The repairs are worked out once for the
    findings of a project, on a thread of their own, since one may take a song played again: the buttons are
    there when that is through.
    """

    def __init__(self, session: Session, ui: Ui) -> None:
        super().__init__(session, ui)
        self.values = table("", None)
        self.list = self.add(Findings(self.rect, changed=lambda thing: session.pick(*thing), help=LIST_HELP,
                                      mend=self.mend))
        self.all = self.add(Button(self.rect, FIX_ALL, action=self.mend_all, help=ALL_HELP))
        self.all.visible = False
        self.bar = self.add(ScrollBar(self.rect, lambda offset: self.list.scroll.to(offset), BAR_HELP))
        self.bar.visible = False
        self._places = arrange(self.rect)
        # Held where the two below are read and changed: what the repairs were last asked for, the project and
        # its findings, and the repairs that were worked out, with what they were asked for.
        self._lock = threading.Lock()
        self._asked: tuple | None = None
        self._worked: tuple[tuple, dict[Finding, Fix]] | None = None

    def place(self, rect: Rect) -> None:
        """Take the panel's rectangle: the numbers take what they need of it, and the findings the rest."""
        super().place(rect)
        self._places = arrange(rect)
        self.list.rect = self._places["list"]
        self.all.rect = self._places["all"]
        self.bar.rect = self._places["bar"]

    def mend(self, fix: Fix) -> bool:
        """Make a repair, as one step that can be taken back, and say what was done. False where it is refused.

        A refusal is the session's to say, and nothing is changed then.
        """
        if not self.session.change_many(fix.changes, fix.label):
            return False
        # What the session says of the change itself stands: that it took the place of one made outside.
        if not self.session.message:
            self.session.say(fix.label)
        return True

    def mend_all(self) -> None:
        """Make every repair the findings have, in the order they are shown, each as a step of its own.

        A repair that several findings share is made once. The first that is refused is the last that is tried.
        """
        made: list[Fix] = []
        for fix in [block.fix for block in self.list.blocks if block.fix is not None]:
            if fix in made:
                continue
            if not self.mend(fix):
                return
            made.append(fix)
        if len(made) > 1:
            self.session.say(f"{len(made)} fixes made")

    def _mends(self, found: tuple[Finding, ...] | None) -> Mapping[Finding, Fix]:
        """The repairs of the findings that are shown, as far as they are worked out: none until they all are.

        They are asked for once, when findings arrive that are not those of the project they were last asked for.
        """
        if not found:
            return {}
        model = self.session.model
        with self._lock:
            asked = self._asked
            if asked is None or asked[0] is not model.project or asked[1] != found:
                asked = self._asked = (model.project, found)
                threading.Thread(target=self._work, args=(asked, model.rate), name=WORKER, daemon=True).start()
            worked = self._worked
        return worked[1] if worked is not None and worked[0] is asked else {}

    def _work(self, asked: tuple, rate: int) -> None:
        """Work out the repairs of the findings of a project, and keep them if they are still the ones asked for."""
        project, found = asked
        data, shared, worked = project.to_dict(), {}, {}
        for finding in found:
            if self._asked is not asked:
                return
            fix = fixes.fix(data, finding, rate, shared)
            if fix is not None:
                worked[finding] = fix
        with self._lock:
            if self._asked is asked:
                self._worked = (asked, worked)

    def update(self, passed: float = 0.0) -> None:
        """Show the numbers of the picked thing and the findings as the session has them now."""
        session = self.session
        picked = session.model.picked
        self.values = table(picked[0] if picked else "", session.numbers())
        if session.model.loops():
            self.values.append(join_row(session.join()))
        # Findings that are not there yet are not asked for: the asking would wait for them. Nor are those of
        # checks that could not read the project: there are none, and the asking would raise their refusal.
        refused = session.findings_refused
        ready = session.finding_count() is not None and not refused
        self.list.found = tuple(session.model.findings()) if ready else None
        self.list.refused = refused
        self.list.picked = picked
        self.list.mends = self._mends(self.list.found)
        self.all.visible = self.shown and len(self.list.mends) > 1
        # The findings are laid out as they are asked for their blocks, and the bar shows how far they are scrolled.
        scroll = self.list.scroll if self.list.blocks else Scroll()
        self.bar.show(scroll.content, scroll.view, scroll.offset)
        self.bar.visible = self.shown and scroll.most > 0

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
