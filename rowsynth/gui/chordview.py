"""The chords of a pattern as the chord check reads them, a bar's chord changed, and the box a new chord is made in."""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ..checks import _whole_rows
from ..checks.chords import _NOTES, breach, looked_at, under
from ..edit import NOTHING
from ..errors import RowSynthError
from . import theme
from .font import Font
from .layout import Rect
from .popups import ENTER, TextBox
from .songview import Channel, Note, Pattern, bar_rows
from .ui import Button, Choice, Event, Pos, Ui, Widget, inside

if TYPE_CHECKING:
    from .canvas import Canvas

# What is read here is read with the check's own rule, `under` and `breach` of `rowsynth.checks.chords`, and
# not with a copy of it: a note is wrong here when the check finds it, and a chord lies where the check has it.

#: The kinds of chord that can be made here, by what each is called: the semitones over its root that belong to
#: it, the root first, and what the name of such a chord ends on after its root.
KINDS: dict[str, tuple[tuple[int, ...], str]] = {
    "major": ((0, 4, 7), ""),
    "minor": ((0, 3, 7), "m"),
    "dim": ((0, 3, 6), "dim"),
    "aug": ((0, 4, 8), "aug"),
    "sus2": ((0, 2, 7), "sus2"),
    "sus4": ((0, 5, 7), "sus4"),
    "7": ((0, 4, 7, 10), "7"),
    "maj7": ((0, 4, 7, 11), "maj7"),
    "min7": ((0, 3, 7, 10), "m7"),
}
#: What the name of a chord cannot hold besides a space: a place in a project is spelt with these.
_NOT_IN_NAMES = frozenset(".[]")
#: The pitch classes of the white keys of an octave, from c up.
_WHITES = (0, 2, 4, 5, 7, 9, 11)

#: The box a new chord is made in. From its frame to what stands in it, and from one part of it to the next.
BOX_PAD = 6
BOX_GAP = 4
#: How high its heading is, how wide and how high its keyboard, and how high a button and the row of the kinds.
HEAD_H = 11
KEYS_SIZE = (98, 38)
BUTTON_H = 13
#: From the keyboard to what stands beside it, and how wide the two buttons are.
BESIDE = 10
OK_W = 28
CANCEL_W = 46
#: What the box is headed, what stands before the name, and what the two buttons read.
HEADING = "NEW CHORD"
NAME = "NAME"
OK = "OK"
CANCEL = "CANCEL"
#: The one line of help the box has.
BOX_HELP = "press a key for the root, pick the kind and type the name: enter adds the chord, escape leaves it"


@dataclass(frozen=True)
class Span:
    """A stretch of a pattern that lies under one chord.

    row is the row it starts at and rows how many it has. name is what the chord is called in the project's
    table, and tones are the pitch classes that belong to it, from 0 for c to 11 for b.
    """

    row: float
    rows: float
    name: str
    tones: frozenset[int]


@dataclass(frozen=True)
class Bar:
    """One bar of a pattern as the chord check reads it.

    text is what the song has written for it, its names a space apart, and "" for a bar with nothing written.
    spans are the stretches of the bar that lie under a chord, in order: one for a bar with one chord, more
    where several share it, and none for a bar with no chord.
    """

    text: str
    spans: tuple[Span, ...]


@dataclass(frozen=True)
class Chords:
    """The chords of one pattern of a song, and what the check holds the notes of the pattern to them by.

    table is the project's chords by name. written has the string the song has for each bar, per_beat is the
    rows of a beat and per_bar those of a bar. bars has every bar of the pattern, also those the song has no
    string for.
    """

    table: dict
    written: tuple[str, ...]
    per_beat: float
    per_bar: float
    bars: tuple[Bar, ...]


def read(table: dict, song: dict, pattern: str, rows: float) -> Chords:
    """The chords a song has written for a pattern of `rows` rows, read as the chord check reads them.

    `table` is the project's chords. A bar is four beats of the song, and the pattern has as many bars as its
    rows begin. Each whole row of a bar is asked of the check's own `under`, so a bar that several chords share
    is shared as the check shares it. A bar after the last string has no chord, and neither has one whose
    string names none. A bar whose string the check refuses, since it names a chord the table does not have,
    is shown with none as well.
    """
    written = tuple(song.get("chords", {}).get(pattern, ()))
    per_bar = bar_rows(song)
    bars = []
    for bar in range(math.ceil(_whole_rows(rows) / per_bar)):
        spans: list[Span] = []
        for step in range(math.ceil(per_bar)):
            row = float(bar * per_bar + step)
            try:
                held = under(table, written, row, per_bar, "chords")
            except RowSynthError:
                held = None
            if held is None:
                continue
            if spans and spans[-1].name == held[1] and spans[-1].row + spans[-1].rows == row:
                spans[-1] = replace(spans[-1], rows=spans[-1].rows + 1)
            else:
                spans.append(Span(row, 1.0, held[1], frozenset(table[held[1]]["tones"])))
        bars.append(Bar(" ".join(written[bar].split()) if bar < len(written) else "", tuple(spans)))
    return Chords(table, written, song.get("rows_per_beat", 4), per_bar, tuple(bars))


def looked(pattern: Pattern, channel: Channel) -> bool:
    """Whether the chord check looks at a channel of a pattern, as its own `looked_at` says.

    It looks at no drum channel, and at none that ends before its pattern does.
    """
    return looked_at(channel.drums, channel.rows, pattern.rows)


def wrong(chords: Chords, note: Note) -> tuple[int, str, float] | None:
    """What the chord check finds on a note of a channel it looks at, as its own `breach` says. None for no finding.

    A finding is the bar the note starts in, counted from 0, the name of the chord under it, and the pitch
    class of the note. A note under a string the check refuses is not found: the check says that of the
    string, and of no note.
    """
    try:
        return breach(chords.table, chords.written, note.row, note.pitch, chords.per_beat, chords.per_bar, "chords")
    except RowSynthError:
        return None


def with_chord(table: dict, written: Sequence[str], bars: int, bar: int, name: str | None) -> list[str]:
    """The strings a song has for the bars of a pattern, with one bar under another chord, or under none.

    `written` is the strings there are, `bars` how many bars the pattern has, and `bar` the one that is
    changed, counted from 0. `name` is one of the chords of `table`, the project's, or None for no chord. A
    bar before it that has no string yet gets an empty one, as a bar that is cleared does: the project
    takes it, and to the chord check a bar whose string names no chord is a bar with none, whose notes it
    passes over. Strings at the end that name no chord are left out, so a pattern with no chord left has no
    strings.

    Refused are a bar the pattern does not have, and a chord the table does not have.
    """
    if isinstance(bar, bool) or not isinstance(bar, int) or not 0 <= bar < bars:
        raise RowSynthError(f"{bar!r} is no bar of the pattern, which has {bars}, counted from 0")
    if name is not None and name not in table:
        raise RowSynthError(f"the project has no chord called {name!r}")
    made = [*written, *[""] * (bar + 1 - len(written))]
    made[bar] = name or ""
    while made and not made[-1].split():
        made.pop()
    return made


def saving(song_name: str, song: dict, pattern: str, written: Sequence[str]) -> tuple[str, object]:
    """Where the strings for the bars of a pattern are saved, and what is put there, as `Session.change` takes them.

    `song` is the song as the project holds it. A song that has chords gets the strings of the pattern among
    them. With no string left the pattern is taken out of them: the value is NOTHING. A song that has no
    chords at all gets them, with the pattern alone.
    """
    if "chords" not in song:
        return f"songs.{song_name}.chords", {pattern: list(written)}
    return f"songs.{song_name}.chords.{pattern}", list(written) or NOTHING


def chord(root: int, kind: str) -> dict:
    """A chord for the project's table: a root and the pitch classes a chord of that kind has on it, the root first.

    The root is a pitch class, from 0 for c to 11 for b, and the kind one of `KINDS`. Any other is refused.
    """
    if isinstance(root, bool) or not isinstance(root, int) or not 0 <= root <= 11:
        raise RowSynthError(f"the root of a chord is a whole number from 0 for c to 11 for b, not {root!r}")
    if kind not in KINDS:
        raise RowSynthError(f"a chord is one of {', '.join(KINDS)}, not {kind!r}")
    return {"root": root, "tones": [(root + step) % 12 for step in KINDS[kind][0]]}


def chord_name(root: int, kind: str) -> str:
    """What a chord of a root and a kind is called until it is given another name: `C`, `Am`, `F#dim`."""
    return _NOTES[chord(root, kind)["root"]].upper() + KINDS[kind][1]


def in_name(char: str) -> bool:
    """Whether the name of a chord can hold a character.

    It holds no space, since the chords of a bar are written a space apart, and no dot or square bracket,
    which the places of a project are spelt with.
    """
    return not (char.isspace() or char in _NOT_IN_NAMES or ord(char) < 32)


def named(table: dict, name: object) -> str:
    """A name for a new chord, once it is seen to be one that can be given.

    A name is text with something in it, and with nothing in it that `in_name` says no to. No chord of `table`,
    the project's, has it. Any other name is refused, and the refusal says which of these it broke.
    """
    if not isinstance(name, str) or not name:
        raise RowSynthError(f"a chord needs a name, and {name!r} is none")
    for char in name:
        if not in_name(char):
            raise RowSynthError(f"a chord cannot be called {name!r}: a name cannot hold {char!r}")
    if name in table:
        raise RowSynthError(f"the project has a chord called {name!r} already")
    return name


def adding(table: dict | None, name: str, made: dict) -> tuple[str, object]:
    """Where a new chord is saved in a project, and what is put there, as `Session.change` takes them.

    `table` is the chords the file of the project has, or None for a file with no table of them: that one
    gets a table, with the chord alone. The name is held to what `named` allows.
    """
    named(table or {}, name)
    return ("chords", {name: made}) if table is None else (f"chords.{name}", made)


def keys(rect: Rect) -> list[Rect]:
    """The twelve keys of a keyboard of one octave in a rectangle, by their pitch classes: from c up.

    The seven white keys stand side by side, each a seventh of the rectangle wide and as high as it. A black
    key is two thirds as wide and three fifths as high, and lies over the line between two white ones.
    """
    x, y, w, h = rect
    wide = w // 7
    found: dict[int, Rect] = {tone: (x + place * wide, y, wide, h) for place, tone in enumerate(_WHITES)}
    for tone in range(12):
        if tone not in found:
            narrow = wide * 2 // 3
            found[tone] = (found[tone - 1][0] + wide - narrow // 2, y, narrow, h * 3 // 5)
    return [found[tone] for tone in range(12)]


def key_at(rect: Rect, pos: Pos) -> int | None:
    """The pitch class of the key of such a keyboard that a position is on, or None where it is on none.

    A black key lies over the white ones, so it is asked first.
    """
    found = keys(rect)
    for tone in sorted(range(12), key=_WHITES.__contains__):
        if inside(found[tone], pos):
            return tone
    return None


def box_parts(at: Pos, size: tuple[int, int] | None = None) -> dict[str, Rect]:
    """Where the parts of the box a new chord is made in go, each by what it is called, the box opened at a place.

    "box" is all of it, as wide as the row of the kinds needs. Its top-left corner is at `at`, and with the
    `size` of the art surface given it is moved the least that puts all of it on the surface. "heading" runs
    along the top. Under it "keys" is the keyboard, and beside that "word" and "name", the name with the
    word before it, and at the foot of the keyboard "ok" and "cancel". "kinds" is the row under them all.
    """
    labels = [Font.width(kind.upper()) + 2 * theme.PAD for kind in KINDS]
    inner = sum(labels)
    wide, high = inner + 2 * BOX_PAD, 2 * BOX_PAD + HEAD_H + KEYS_SIZE[1] + BUTTON_H + 2 * BOX_GAP
    x, y = at
    if size is not None:
        x, y = max(0, min(x, size[0] - wide)), max(0, min(y, size[1] - high))
    left, top = x + BOX_PAD, y + BOX_PAD + HEAD_H + BOX_GAP
    beside, word = left + KEYS_SIZE[0] + BESIDE, Font.width(NAME) + theme.GAP
    right, foot = left + inner, top + KEYS_SIZE[1] - BUTTON_H
    return {
        "box": (x, y, wide, high),
        "heading": (left, y + BOX_PAD, inner, HEAD_H),
        "keys": (left, top, *KEYS_SIZE),
        "word": (beside, top, word, theme.TEXT_BOX_H),
        "name": (beside + word, top, right - beside - word, theme.TEXT_BOX_H),
        "ok": (right - CANCEL_W - theme.BUTTON_GAP - OK_W, foot, OK_W, BUTTON_H),
        "cancel": (right - CANCEL_W, foot, CANCEL_W, BUTTON_H),
        "kinds": (left, top + KEYS_SIZE[1] + BOX_GAP, inner, BUTTON_H),
    }


class ChordBox(Widget):
    """The box a new chord is made in, opened as the modal widget: `ui.open(ChordBox(at, done))`.

    It shows the keyboard of one octave. A press on a key makes it the root: `root` is its pitch class. `kinds`
    is the choice of the kind, and `chord` the chord the two make. Its notes are lit on the keyboard, the root
    in a colour of its own. `name` is the box the name is typed in, which has the keys and what is typed: only
    what `in_name` says yes to goes in. It holds the name `chord_name` gives for as long as that is what stands
    in it: a name that was typed stays.

    OK and enter call `done` with the name and the chord, and close the box unless `done` answers false: then
    the box stays open, so that the name can be mended. CANCEL, escape and a press off the box close it with
    nothing done.
    """

    takes_text = True

    def __init__(self, at: Pos, done: Callable[[str, dict], object] | None = None) -> None:
        super().__init__(box_parts(at)["box"], BOX_HELP)
        self.at = at
        self.done = done
        self.root = 0
        nowhere = (0, 0, 0, 0)
        self.kinds = Choice(nowhere, [kind.upper() for kind in KINDS])
        self.name = TextBox(nowhere, chord_name(self.root, self.kind), allowed=in_name)
        self.ok = Button(nowhere, OK)
        self.cancel = Button(nowhere, CANCEL)
        self._parts: dict[str, Rect] = {}
        # The button the left button of the mouse went down on, until it is let go.
        self._pressed: Button | None = None
        self._mouse: Pos = (-1, -1)
        self._lay(None)

    @property
    def kind(self) -> str:
        """The kind that is chosen, as `KINDS` calls it."""
        return tuple(KINDS)[self.kinds.picked]

    @property
    def chord(self) -> dict:
        """The chord of the root and the kind that are chosen, as the project's table holds one."""
        return chord(self.root, self.kind)

    def keep_in(self, size: tuple[int, int]) -> None:
        """Take the size of the art surface, and move the box the least that puts all of it there."""
        self._lay(size)

    def _lay(self, size: tuple[int, int] | None) -> None:
        """Lay the parts out as `box_parts` has them."""
        self._parts = parts = box_parts(self.at, size)
        self.rect = parts["box"]
        for part, widget in (("kinds", self.kinds), ("name", self.name), ("ok", self.ok), ("cancel", self.cancel)):
            widget.rect = parts[part]

    def choose(self, root: int, kind: str) -> None:
        """Take another root and another kind. The name goes along while it is the one the two before it gave."""
        follows = self.name.text == chord_name(self.root, self.kind)
        self.root, self.kinds.picked = chord(root, kind)["root"], list(KINDS).index(kind)
        if follows:
            self.name.text = chord_name(self.root, self.kind)
            self.name.anchor, self.name.caret = 0, len(self.name.text)

    def on(self, event: Event, ui: Ui) -> bool:
        """Use everything: a press on a key, a kind or a button, the keys of the box, and what is typed."""
        if event.kind == "text":
            self.name.on(event, ui)
        elif event.kind == "key":
            if event.key in ENTER:
                self._end(ui)
            elif event.key == "escape":
                ui.close(self)
            else:
                self.name.on(event, ui)
        else:
            self._mouse = event.pos
            if event.kind == "down":
                self._down(event, ui)
            elif event.kind == "up" and event.button == 1:
                pressed, self._pressed = self._pressed, None
                if pressed is not None and pressed.hit(event.pos):
                    if pressed is self.ok:
                        self._end(ui)
                    else:
                        ui.close(self)
        return True

    def _down(self, event: Event, ui: Ui) -> None:
        """Take a press: off the box it closes it, and the left button on a part of the box is that part's."""
        pos = event.pos
        if not self.hit(pos):
            ui.close(self)
        elif event.button == 1:
            tone, kind = key_at(self._parts["keys"], pos), self.kinds.at(pos)
            if tone is not None:
                self.choose(tone, self.kind)
            elif kind is not None:
                self.choose(self.root, tuple(KINDS)[kind])
            elif self.name.hit(pos):
                self.name.on(event, ui)
            else:
                self._pressed = next((button for button in (self.ok, self.cancel) if button.hit(pos)), None)

    def _end(self, ui: Ui) -> None:
        """Tell the owner of the name and the chord, and close the box unless the owner answers false."""
        if self.done is None or self.done(self.name.text, self.chord) is not False:
            ui.close(self)

    def draw(self, canvas: Canvas) -> None:
        """Draw the box: its heading, the keyboard with the notes of the chord lit, the name, the buttons, the kinds."""
        parts = self._parts
        with canvas.clip(self.rect):
            canvas.box(self.rect, theme.PANEL)
            canvas.frame(self.rect, theme.LINE)
            canvas.text_in(HEADING, parts["heading"], theme.DIM)
            canvas.text_in(NAME, parts["word"], theme.DIM)
            tones, found = self.chord["tones"], keys(parts["keys"])
            # The black keys lie over the white ones, so they are drawn after them.
            for tone in sorted(range(12), key=lambda tone: tone not in _WHITES):
                lit = theme.ACTIVE if tone == self.root else theme.MUSTARD
                plain = theme.TEXT if tone in _WHITES else theme.BACKGROUND
                canvas.box(found[tone], lit if tone in tones else plain)
                canvas.frame(found[tone], theme.BACKGROUND)
            self.kinds.draw(canvas)
            self.name.draw(canvas)
            for button in (self.ok, self.cancel):
                button.hot, button.held = button.hit(self._mouse), button is self._pressed
                button.draw(canvas)
