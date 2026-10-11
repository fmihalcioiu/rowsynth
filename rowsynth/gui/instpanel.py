"""The middle of the window for an instrument: its wave, tables and knobs, its FM voice, and a keyboard to try it on."""
from __future__ import annotations

import copy
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from ..edit import NOTHING, steps
from ..errors import RowSynthError
from ..fm import CARRIER, ENVELOPE, _SILENT, _algorithm
from ..note import DEFAULT_RATE
from ..osc import WAVES
from . import theme
from .channelbar import tidy
from .dials import Knob, Stepper, plain
from .fmview import FIRST, POINTS, WIRING_H, Envelope, Wiring, settings, wires
from .font import Font
from .layout import Rect
from .panels import Panel, within
from .scope import TEXT_H, Scope, Spectrum
from .session import piano
from .tables import StepTable
from .ui import NOWHERE, Choice, Event, Label, Pos, Toggle, Ui, Widget, draw_button, inside, share

if TYPE_CHECKING:
    from ..project import Project
    from .canvas import Canvas
    from .session import Session

#: From the sides of the panel to its two boxes, and between the two.
MARGIN = 4
#: How high the box of the heading is, from its sides to its words, and between two things it says. The heading
#: of an operator has that much between its two things as well.
HEAD_H = 17
HEAD_PAD = 5
HEAD_SPACE = 8
#: From the outline of the lower box to the page in it.
PAD = 4
#: From one part of the page to the next, and from one table to the next.
SPACE = 6
TABLE_GAP = 3
#: How high the line of the wave is: its buttons are small ones.
ROW_H = 13
#: How wide an icon of a wave is, as the set file has them.
ICON = 9
#: How high a table is at the least, for each share it has of the height of the tables.
TABLE_UNIT = 15
#: How high the row of the knobs is, how wide a knob with its two lines is at the most, and between two knobs.
KNOB_H = 37
KNOB_W = 54
KNOB_GAP = 4
#: The keyboard: how high, how wide at the most, how many octaves, and how wide the stepper of its octave. It is
#: as high as the scope and the spectrum beside it need to be read, and those two share the room by these shares.
KEYS_H = 56
KEYS_W = 196
OCTAVES = 2
OCTAVE_W = 36
BESIDE = (2, 3)
#: How high the envelope of an operator is at the least, and the shares by which the envelopes and the tables
#: under them take the height that the page of an FM instrument has to spare.
ENV_MIN = 45
SPARE = (3, 2)
#: How many pixels one step of the wheel scrolls a page that does not fit, and how wide the thumb that shows it.
WHEEL_PX = 24
THUMB_W = 2
#: The rectangle of a widget that has no room: one for which the page that is laid out has no part.
OFF = (0, 0, 0, 0)

#: What the heading and the line of the wave read, and the toggle of the short noise.
INSTRUMENT = "INSTRUMENT"
WAVE = "WAVE"
BUZZ = "BUZZ"
#: What stands over the column of an operator before its number, and after it what the operator does in the wiring.
OPERATOR = "OPERATOR"
HEARD = "HEARD"
CHANGES = "CHANGES"
#: What is said of an instrument that nothing plays.
NOWHERE_USED = "used nowhere"

#: The waves the engine has, in its order, and the one it takes for an instrument that names none.
NAMES = tuple(WAVES)
USUAL = "pulse"
#: The operators of an FM voice, by their numbers in the file, and the wiring the engine takes for an instrument
#: that names none.
OPERATORS = range(CARRIER.shape[1])
WIRING = _algorithm({})
#: What the engine takes for an operator that an instrument leaves out: one that is silent.
SILENT = _SILENT
#: The operators an instrument with none is given when the FM wave is first picked for it, so that it is heard:
#: operator 0 bends operator 1, both as the engine takes an operator that names nothing, and that one sine bent
#: by another peaks where a pulse does. The other pair is the same at no level, ready to be turned up: all four
#: at the level the engine takes would peak at twice that.
PLAIN = ({}, {}, {"level": 0}, {"level": 0})
#: What a drag of an envelope is called in a step when no point of it came to another place.
SHAPE = "envelope"

#: The semitones above c of the white keys of an octave, and those of them a black key comes after.
WHITE = (0, 2, 4, 5, 7, 9, 11)
BLACK_AFTER = (0, 2, 5, 7, 9)
#: The notes of an octave as a sound spells them, sharps and all.
SPELT = ("c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b")
#: The octaves the keyboard can start at, the one it starts at first, and the note heard before any was played: a4.
LOWEST = tuple(range(8))
FIRST_OCTAVE = 3
A4 = 69
#: The keys of the computer that play a note, as `session.piano` lays them out.
PIANO = tuple(key for key in "abcdefghijklmnopqrstuvwxyz0123456789" if piano(key) is not None)

_AT = re.compile(r"@(\w+)", re.ASCII)


@dataclass(frozen=True)
class Table:
    """One step table an instrument can have: its key in the file, and how it is shown.

    text is its label and word what one of its values is called. low and high are the range the mouse draws in,
    and absent what the engine takes for every tick where the instrument has no such table. share is its share
    of the height of the tables. wave is the one wave that uses it, None for all of them, and unused what its
    help line says under any other wave.
    """

    key: str
    text: str
    word: str
    low: float
    high: float
    absent: float
    share: int
    help: str
    wave: str | None = None
    unused: str = ""


@dataclass(frozen=True)
class Number:
    """One number of an instrument that a knob holds: its place in the instrument, and how it is shown.

    text is the knob's label and words what the setting is called in a step that can be taken back. The knob
    goes from low to high in so many steps. absent is what the engine takes where the instrument has no such
    number: a knob put there is taken out of the file.
    """

    key: str
    text: str
    words: str
    low: float
    high: float
    steps: int
    absent: float
    help: str


#: The step tables, from the top. A file may hold any number in most of them, so the ranges are chosen here:
#: the loudness and the width are the engine's own, a jump and a sweep go an octave each way, and the filter
#: goes up to 8000 Hz, with 0 for none.
TABLES = (
    Table("vol", "LOUDNESS", "loudness", 0, 15, 15, 3,
          "how loud a note is tick by tick, 0 to 15: drag to draw it, and the handle at its end makes it longer"),
    Table("duty", "PULSE WIDTH", "width", 0, 3, 2, 2,
          "the shape of the pulse tick by tick: 0 is thin and sharp, 1 fuller, 2 square and hollow, 3 sounds like 1",
          "pulse", "only the pulse wave has a width: with another wave this table does nothing"),
    Table("arp", "JUMP", "jump", -12, 12, 0, 3,
          "semitones above or below its note that the sound jumps to at each tick: a fast round is heard as a chord"),
    Table("pitch", "SWEEP", "sweep", -12, 12, 0, 3,
          "semitones the note moves by at each tick and keeps: under the line it falls, as a kick or a zap does"),
    Table("lp", "FILTER", "filter", 0, 8000, 0, 3,
          "how much of the high end is left tick by tick: all above this many hertz is dulled, and 0 dulls nothing"),
)
#: The numbers, from the left. The three of the vibrato are parts of one setting, which is there with its depth.
NUMBERS = (
    Number("gain", "LEVEL", "level", 0, 2, 200, 1,
           "how loud the instrument is beside the others, as a factor on its loudness: a right click is 1 again"),
    Number("release", "FADE", "fade", 0, 60, 60, 2,
           "how many ticks a note takes to fade away once it is let go: a right click is 2 again"),
    Number("gate", "HELD FOR", "held for", 0.05, 1, 95, 0.9,
           "the share of its length a note of a song is held for before it is let go: the keys here do not go by it"),
    Number("slide", "SLIDE", "slide", -24, 24, 96, 0,
           "semitones a second the note keeps sliding by for as long as it sounds, up or down"),
    Number("vib.depth", "VIBRATO", "vibrato", 0, 2, 200, 0,
           "how far the note wavers up and down, in semitones: at 0 it does not waver"),
    Number("vib.delay", "VIB WAIT", "vibrato wait", 0, 60, 60, 0,
           "how many ticks into a note the wavering starts: it needs VIBRATO above 0"),
    Number("vib.rate", "VIB RATE", "vibrato rate", 1, 20, 38, 6,
           "how many times a second the note wavers: it needs VIBRATO above 0"),
    Number("res", "RING", "ring", 0, 0.98, 98, 0,
           "how strongly the filter rings at the pitch where it starts to dull: it needs the FILTER table"),
    Number("key_scale", "KEY TILT", "key tilt", -12, 12, 48, 0,
           "decibels louder for each octave above middle c and quieter for each below: under 0 high notes are quiet"),
)
#: The feedback of the FM wave, and the numbers of an operator beside its envelope, from the left. A file may
#: hold any number in them, so the ranges are chosen here: they reach past everything the examples use.
FEEDBACK = Number("fb", "FEEDBACK", "feedback", 0, 3, 60, 0,
                  "how far operator 0 bends its own wave, in radians: a little makes it brighter, and a lot is noise")
DIALS = (
    Number("mul", "PITCH X", "pitch times", 0, 16, 64, 1,
           "the pitch of the operator as so many times the pitch of the note: 2 is an octave above it, 0.5 one below"),
    Number("det", "DETUNE", "detune", -10, 10, 200, 0,
           "hertz added to the pitch of the operator: a little sets it beating slowly against the others"),
    Number("level", "LEVEL", "level", 0, 8, 160, 1,
           "how loud the operator is where it is heard, and where it is not, how far it bends the ones it changes"),
)
#: What the toggle of the short noise says of itself, under the noise wave and under any other.
BUZZ_HELP = "noise in its short register, which is a buzz with a pitch to it"
BUZZ_UNUSED = "only the noise wave has a short register: with another wave this does nothing"
#: What the choice of the wave, the keyboard and the stepper of its octave say of themselves.
WAVE_HELP = "what the sound is made of: PULSE is hollow, SAW bright, TRI soft, NOISE a hiss, FM four sines"
KEYS_HELP = "press a key to hear it, and drag along the keys: the keys of the computer play them too"
OCTAVE_HELP = "the octave the keyboard starts at: its lowest key is this c"
#: What the wirings and the envelope of an operator say of themselves, off their pictures and off their points, and
#: the scope and the spectrum while they have no sound to show.
WIRING_HELP = "which operators are heard and which change which: click a wiring to pick it"
ENVELOPE_HELP = ("how loud the operator is over a note: drag a point for its attack, its decay and the level it "
                 "falls to, its fall while held, or its release")
SCOPE_HELP = "a few cycles of the note played last, as the instrument sounds now"
SPECTRUM_HELP = "what is in the note played last from low to high, and where it begins to be harsh"

#: How high the tables are together at the least, with the gaps between them, and the page with all its parts.
TABLES_MIN = TABLE_UNIT * sum(table.share for table in TABLES) + TABLE_GAP * (len(TABLES) - 1)
NEEDED = ROW_H + SPACE + TABLES_MIN + SPACE + KNOB_H + SPACE + KEYS_H
#: The tables an FM voice reads, which stand side by side on its page, how high they are at the least, and that
#: page with all its parts: the wirings, the heading, the envelope and the knobs of an operator, and the rest.
FM_TABLES = tuple(table for table in TABLES if table.wave in (None, "fm"))
FM_TABLE_MIN = TABLE_UNIT * max(table.share for table in FM_TABLES)
FM_NEEDED = (ROW_H + SPACE + WIRING_H + SPACE + Font.LINE + ENV_MIN + TABLE_GAP + KNOB_H + SPACE + FM_TABLE_MIN
             + SPACE + KNOB_H + SPACE + KEYS_H)
#: What the scope and the spectrum are handed for a note that could not be made: no sound.
NO_SOUND = np.zeros(0, dtype=np.float32)


def note_name(pitch: int) -> str:
    """A pitch as a sound spells its note, with a sharp where it has one: 60 is c4, and 61 c#4."""
    return f"{SPELT[pitch % 12]}{pitch // 12 - 1}"


def _held(held: object, key: str, absent: object) -> object:
    """What an instrument holds at a place in it, as "vib.depth" or "ops[1].mul", or `absent` where it holds none."""
    for step in steps(key):
        if not (step < len(held) if isinstance(step, int) and isinstance(held, list)
                else isinstance(held, dict) and step in held):
            return absent
        held = held[step]
    return held


def role(algorithm: int, op: int) -> str:
    """What an operator does in a wiring, as the heading of its column says it: `HEARD`, or `CHANGES 1, 2`."""
    wired = wires(algorithm)
    changed = ", ".join(str(to + FIRST) for source, to in wired.changes if source == op)
    return HEARD if op in wired.heard else f"{CHANGES} {changed}"


def _voices(entry: dict) -> list[dict]:
    """Every voice of the entry of a sound, whichever form the entry has."""
    found = [voice for defn in ([entry["def"]] if "def" in entry else entry.get("defs", []))
             for voice in defn["voices"]]
    for _, part in entry.get("parts", []):
        found += _voices(part)
    if "held" in entry:
        found += entry["held"]["note"]["voices"] + entry["held"]["end"]["voices"]
    return found


def uses(project: Project, name: str) -> dict[str, dict[str, int]]:
    """Where an instrument of a project is used, counted for each kind of use.

    "channels" is from a song to how many of its channels have the instrument as their own, and "switches" to
    how many more of its channels switch to it somewhere in their text. "drums" is from a song to how many of
    its drum letters play it, and "sounds" from a sound to how many of its voices do. Only what uses it is in
    a count. A name that a song gives to an instrument of the project stands for that one, and a name of an
    instrument that is the song's own is none of the project's.
    """
    found: dict[str, dict[str, int]] = {"channels": {}, "switches": {}, "drums": {}, "sounds": {}}
    for title, song in project.songs.items():
        own = song.get("instruments", {})

        def meant(named: str, own: dict = own) -> bool:
            """Whether a name in the song stands for the instrument."""
            return own.get(named, named) == name

        played = [channel for channel, held in song["channels"].items() if meant(held["inst"])]
        switched = {channel for pattern in song["patterns"].values() for channel, text in pattern.items()
                    if channel not in played and any(meant(named) for named in _AT.findall(text))}
        drums = sum(meant(drum["inst"]) for drum in song.get("drums", {}).values())
        for kind, count in (("channels", len(played)), ("switches", len(switched)), ("drums", drums)):
            if count:
                found[kind][title] = count
    for title, entry in project.sounds.items():
        count = sum(voice["inst"] == name for voice in _voices(entry))
        if count:
            found["sounds"][title] = count
    return found


def used(found: dict[str, dict[str, int]]) -> str:
    """What `uses` found in a few words, as the heading says it: `in 3 channels of tune, 2 sounds`.

    A channel that switches to the instrument is a channel of it. Used by one song, that song is named, and by
    more their count is. An instrument nothing plays is `used nowhere`.
    """
    def some(count: int, word: str) -> str:
        return f"{count} {word}{'' if count == 1 else 's'}"

    def of(counts: dict[str, int], word: str) -> str:
        where = next(iter(counts)) if len(counts) == 1 else f"{len(counts)} songs"
        return f"{some(sum(counts.values()), word)} of {where}"

    channels = {song: found["channels"].get(song, 0) + found["switches"].get(song, 0)
                for song in (*found["channels"], *found["switches"])}
    parts = [of(counts, word) for counts, word in ((channels, "channel"), (found["drums"], "drum")) if counts]
    if found["sounds"]:
        parts.append(some(len(found["sounds"]), "sound"))
    return "in " + ", ".join(parts) if parts else NOWHERE_USED


def keyboard(rect: Rect, low: int) -> list[tuple[int, Rect]]:
    """The keys of a keyboard of `OCTAVES` octaves in a rectangle, each as its pitch and its rectangle.

    `low` is the pitch of its lowest key, a c. The white keys come first, side by side over the whole height.
    The black ones come after them: each lies over the two whites it is between, in the upper part.
    """
    x, y, _, h = rect
    whites, blacks = [], []
    for place, wide in enumerate(share(rect[2], [1] * (len(WHITE) * OCTAVES))):
        octave, step = divmod(place, len(WHITE))
        pitch = low + 12 * octave + WHITE[step]
        whites.append((pitch, (x, y, wide, h)))
        if WHITE[step] in BLACK_AFTER:
            narrow = wide * 2 // 3
            blacks.append((pitch + 1, (x + wide - narrow // 2, y, narrow, h * 3 // 5)))
        x += wide
    return whites + blacks


def key_at(rect: Rect, low: int, pos: Pos) -> int | None:
    """The pitch of the key a position is on, a black one where it is on both. None where it is on no key."""
    return next((pitch for pitch, key in reversed(keyboard(rect, low)) if inside(key, pos)), None)


def slack(rect: Rect, fm: bool = False) -> int:
    """How many pixels the page of a panel can be scrolled: by how much it is higher than its room. 0 if it fits.

    `fm` is whether the page is that of an FM instrument.
    """
    return max(0, (FM_NEEDED if fm else NEEDED) - arrange(rect)["view"][3])


def arrange(rect: Rect, scroll: int = 0, fm: bool = False) -> dict[str, Rect]:
    """Where the parts of the panel go in its rectangle, each by what it is called.

    "head" is the box of the heading, along the top. "box" is the box under it, down to the foot, and "view"
    what is inside that box, clear of its outline: those three are cut to the panel. The other parts are the
    page, which is as wide as the view and as high, or `NEEDED` high where the view is lower: it is then
    scrolled `scroll` pixels up, no further than to its end, and a part of it can lie outside the view.

    Down the page come the line of the wave ("word", "wave" and "short"), the tables one under the other, each
    by its key, sharing what height is left by their shares, the row of the knobs, each by its key, and at the
    foot "octave", "keys" and beside the keyboard "scope" and "spectrum". The spectrum is as high as the
    keyboard, and the scope as high as the box of the spectrum, which has the names of its marks under it.

    With `fm` the page is that of an FM instrument, `FM_NEEDED` high at the least. In the place of the tables
    it has the wirings ("alg") with the knob of the feedback ("fb") after them, and under those the operators
    side by side. Each is a column: its heading ("ops[0]", and "ops[0].does" for what it does in the wiring),
    its envelope ("ops[0].env") and under that its knobs ("ops[0].mul" and so on). Under the columns the
    tables of `FM_TABLES` stand side by side. The envelopes and those tables take the height that is left by
    the shares of `SPARE`. A part that only the other page has is not there.
    """
    x, y, w, h = rect
    box = within((x + MARGIN, y + 2 * MARGIN + HEAD_H, w - 2 * MARGIN, h - 3 * MARGIN - HEAD_H), rect)
    view = within((box[0] + PAD, box[1] + PAD, box[2] - 2 * PAD, box[3] - 2 * PAD), box)
    placed = {"head": within((x + MARGIN, y + MARGIN, w - 2 * MARGIN, HEAD_H), rect), "box": box, "view": view}
    left, top, wide, high = view
    needed = FM_NEEDED if fm else NEEDED
    page = max(high, needed)
    at = top - min(max(scroll, 0), page - high)

    word, short = Font.width(WAVE), Font.width(BUZZ) + 2 * theme.PAD
    waves = sum(Font.width(name) + ICON + theme.GAP + 2 * theme.PAD for name in NAMES)
    placed["word"] = (left, at, word, ROW_H)
    placed["wave"] = (left + word + SPACE, at, waves, ROW_H)
    placed["short"] = (left + word + waves + 2 * SPACE, at, short, ROW_H)
    at += ROW_H + SPACE

    if fm:
        knob = min(KNOB_W, wide)
        placed["alg"] = (left, at, max(0, wide - knob - SPACE), WIRING_H)
        placed[FEEDBACK.key] = (left + wide - knob, at + (WIRING_H - KNOB_H) // 2, knob, KNOB_H)
        at += WIRING_H + SPACE
        more, taller = share(page - needed, SPARE)
        tall, column = ENV_MIN + more, left
        for op, across in enumerate(share(max(0, wide - SPACE * (len(OPERATORS) - 1)), [1] * len(OPERATORS))):
            name = f"ops[{op}]"
            named = min(across, Font.width(f"{OPERATOR} {op + FIRST}"))
            said = min(across, named + HEAD_SPACE)
            placed[name] = (column, at, named, Font.LINE)
            placed[f"{name}.does"] = (column + said, at, across - said, Font.LINE)
            placed[f"{name}.env"] = (column, at + Font.LINE, across, tall)
            cell = max(0, min(KNOB_W, (across - KNOB_GAP * (len(DIALS) - 1)) // len(DIALS)))
            for place, number in enumerate(DIALS):
                placed[f"{name}.{number.key}"] = (column + place * (cell + KNOB_GAP),
                                                  at + Font.LINE + tall + TABLE_GAP, cell, KNOB_H)
            column += across + SPACE
        at += Font.LINE + tall + TABLE_GAP + KNOB_H + SPACE
        tall, column = FM_TABLE_MIN + taller, left
        for table, across in zip(FM_TABLES, share(max(0, wide - SPACE * (len(FM_TABLES) - 1)), [1] * len(FM_TABLES))):
            placed[table.key] = (column, at, across, tall)
            column += across + SPACE
        at += tall + SPACE
    else:
        room = page - needed + TABLES_MIN - TABLE_GAP * (len(TABLES) - 1)
        for table, tall in zip(TABLES, share(room, [table.share for table in TABLES])):
            placed[table.key] = (left, at, wide, tall)
            at += tall + TABLE_GAP
        at += SPACE - TABLE_GAP

    cell = max(0, min(KNOB_W, (wide - KNOB_GAP * (len(NUMBERS) - 1)) // len(NUMBERS)))
    for place, number in enumerate(NUMBERS):
        placed[number.key] = (left + place * (cell + KNOB_GAP), at, cell, KNOB_H)
    at += KNOB_H + SPACE

    keys = max(0, min(KEYS_W, wide - OCTAVE_W - SPACE))
    beside = left + OCTAVE_W + keys + 2 * SPACE
    scope, spectrum = share(max(0, left + wide - beside - SPACE), BESIDE)
    placed["octave"] = (left, at + (KEYS_H - ROW_H) // 2, OCTAVE_W, ROW_H)
    placed["keys"] = (left + OCTAVE_W + SPACE, at, keys, KEYS_H)
    placed["scope"] = (beside, at, scope, KEYS_H - TEXT_H)
    placed["spectrum"] = (beside + scope + SPACE, at, spectrum, KEYS_H)
    return placed


class Page(Widget):
    """The ground the parts of a panel stand on. It takes the press and the wheel that no part of the panel used.

    A press gives it the focus, so that the keys go to the panel. `scrolled` is called with the steps of the
    wheel, and says whether the page moved.
    """

    def __init__(self, rect: Rect, scrolled: Callable[[int], bool]) -> None:
        super().__init__(rect)
        self.scrolled = scrolled

    def on(self, event: Event, ui: Ui) -> bool:
        """Use a press, and a step of the wheel that moves the page."""
        if event.kind == "wheel":
            return self.scrolled(event.wheel)
        return event.kind == "down"


class Waves(Choice):
    """A choice of waves by their names, each with its icon before its name in capitals."""

    def __init__(self, rect: Rect, waves: tuple[str, ...], changed: Callable[[int], object] | None = None,
                 help: str = "") -> None:
        super().__init__(rect, [wave.upper() for wave in waves], 0, changed, help)
        self.waves = tuple(waves)

    def cells(self) -> list[Rect]:
        """The rectangle of each wave, side by side from the left, shared out by what each shows."""
        x, y, _, h = self.rect
        cells = []
        for width in share(self.rect[2], [Font.width(text) + ICON + theme.GAP + 2 * theme.PAD
                                          for text in self.options]):
            cells.append((x, y, width, h))
            x += width
        return cells

    def draw(self, canvas: Canvas) -> None:
        """Draw the waves as joined toggles with their icons. The one that is on is drawn last, whole."""
        cells = self.cells()
        for index in sorted(range(len(cells)), key=lambda index: index == self.picked):
            x, y, w, h = cells[index]
            is_on = index == self.picked
            lit = self.hot and index == self._over
            ink = theme.DIM if not self.enabled else theme.ACTIVE if is_on else theme.TEXT
            # All but the last reach one pixel into the next, so that the two outlines are one line.
            draw_button(canvas, (x, y, w + (index < len(cells) - 1), h), self.options[index], self.waves[index],
                        ink, lit, lit and self.held, theme.ACTIVE if is_on else None)


class Keys(Widget):
    """A piano keyboard of `OCTAVES` octaves from `low`, the pitch of its lowest key, which is a c.

    A press of the left button on a key calls `played` with its pitch, and so does each other key the mouse
    comes onto while the button is kept pressed: a key is played once for as long as the mouse stays on it.
    `lit` is the pitch of the key that is drawn lit, None for none: the one played last, here or by its owner.
    The help line names the key under the mouse.
    """

    def __init__(self, rect: Rect, low: int, played: Callable[[int], object] | None = None, help: str = "") -> None:
        super().__init__(rect, help)
        self.low = low
        self.played = played
        self.lit: int | None = None
        self._mouse: Pos = NOWHERE
        #: The key the mouse is on while the button is kept pressed.
        self._on: int | None = None

    @property
    def help(self) -> str:
        """The name of the key under the mouse and then the line it was made with, or that line alone."""
        pitch = key_at(self.rect, self.low, self._mouse)
        return self._help if pitch is None else f"{note_name(pitch)}: {self._help}"

    @help.setter
    def help(self, line: str) -> None:
        self._help = line

    def on(self, event: Event, ui: Ui) -> bool:
        """Play the key a press of the left button is on, and each key the mouse comes onto until its release."""
        if event.kind == "key":
            return False
        self._mouse = event.pos
        if event.kind == "down":
            if event.button != 1:
                return False
            self._strike(event.pos)
            return True
        if event.kind == "move" and ui.held is self:
            self._strike(event.pos)
        elif event.kind == "up":
            self._on = None
        return False

    def _strike(self, pos: Pos) -> None:
        """Play the key at a position, unless it is the one the mouse was on already."""
        pitch = key_at(self.rect, self.low, pos)
        if pitch is not None and pitch != self._on:
            self.lit = pitch
            if self.played is not None:
                self.played(pitch)
        self._on = pitch

    def draw(self, canvas: Canvas) -> None:
        """Draw the white keys with a dark line between two, the black ones over them, and the lit one lit."""
        with canvas.clip(self.rect):
            canvas.box(self.rect, theme.BACKGROUND)
            for pitch, (x, y, w, h) in keyboard(self.rect, self.low):
                black = pitch % 12 not in WHITE
                ink = theme.ACTIVE if pitch == self.lit else theme.BACKGROUND if black else theme.TEXT
                canvas.box((x, y, w if black else w - 1, h), ink)


class InstPanel(Panel):
    """The middle of the window while an instrument is picked: every setting of it to see, to change and to hear.

    The heading is the name of the instrument, and in dim words where it is used. Under it `wave` is a choice
    of the waves the engine has, and `short` the toggle of the short noise. `tables` are the step tables an
    instrument can have, each by its key, one under the other, and `knobs` its numbers, each by its place in
    the instrument. `keys` is a keyboard that plays the instrument, with `octave` to move it, and the keys of
    the computer play it as a piano while a part of the panel has the focus. Beside the keyboard `scope` and
    `spectrum` show the note played last, `last`, as the instrument sounds now: they are handed it when it is
    played, and again after each change to the file, made here or outside. `settings` is from each key an
    instrument can have in the file to the widgets that show it. With a song, a sound or nothing picked the
    panel shows nothing.

    A table that the picked wave does not use is dim and takes no mouse, and so is the toggle.

    While the wave is FM the page is another one. `wiring` is the wirings of the four operators as pictures, of
    which a click picks one, and `feedback` the knob of the feedback. Under them each operator has a column:
    its number and what it does in the picked wiring (`names` and `roles`), in the colour the pictures have for
    one that is heard or for one that changes another, its envelope in that colour with points to drag, and a
    knob for each number it has beside the envelope. `operators` has, for each operator, the widget of each key
    an operator can have in the file. An operator the file leaves out is shown as the engine takes it, silent.
    Under the columns are the tables an FM voice reads, side by side: the one it does not read is not there.

    Each change is one change to the file at the place of the instrument, and one step that can be taken back:
    the wave, the toggle and the wiring as they are picked, a table and an envelope when the mouse lets go of
    them, a knob when the hand has. A number put back to what the engine takes without it is taken out of the
    file, and so is a table of that one value, a vibrato with no depth, and a part of an envelope that a drag
    put there. An operator the file leaves out is written when it is changed, after silent ones for those
    before it that are left out too. The FM wave picked for an instrument with no operators gives it those of
    `PLAIN` in the same step, so that it is heard, and another wave picked leaves all of the FM wave in the
    file. After each change the widgets show what the file holds: one that was refused is not in it, and the
    session says why. The note played last is then heard again in the sound as it is now, unless the session
    plays.

    When the model has loaded its file again, the widgets show what it holds now. One that is under the hand
    just then is left as the hand has it, and shows the file when it is let go.

    What does not fit in a low panel is not dropped: the page is scrolled with the wheel, `scroll` pixels, and
    a part that is not wholly in view is not seen.
    """

    def __init__(self, session: Session, ui: Ui, clock: Callable[[], float] = time.perf_counter) -> None:
        """`clock` times a run of the wheel on a knob."""
        super().__init__(session, ui)

        def knob(number: Number, settled: Callable[[float], object]) -> Knob:
            """A knob of the panel for a number: `settled` is told what it was left at."""
            return self.add(Knob(OFF, number.absent, number.low, number.high, default=number.absent,
                                 steps=number.steps, colour=theme.INSTRUMENT, text=number.text, help=number.help,
                                 settled=settled, clock=clock))

        self.page = self.add(Page(OFF, self._wheel))
        self.word = self.add(Label(OFF, WAVE))
        self.wave = self.add(Waves(OFF, NAMES, self._waved, WAVE_HELP))
        self.short = self.add(Toggle(OFF, BUZZ, changed=self._buzzed, colour=theme.INSTRUMENT, help=BUZZ_HELP))
        self.tables: dict[str, StepTable] = {
            table.key: self.add(StepTable(OFF, table.absent, table.low, table.high, partial(self._drawn, table),
                                          theme.INSTRUMENT, table.text, help=table.help, word=table.word))
            for table in TABLES}
        self.knobs: dict[str, Knob] = {number.key: knob(number, partial(self._turned, number)) for number in NUMBERS}
        self.wiring = self.add(Wiring(OFF, WIRING, self._wired, WIRING_HELP))
        self.feedback = knob(FEEDBACK, partial(self._turned, FEEDBACK))
        self.names = [self.add(Label(OFF, f"{OPERATOR} {op + FIRST}")) for op in OPERATORS]
        self.roles = [self.add(Label(OFF, "")) for _ in OPERATORS]
        self.operators: list[dict[str, Widget]] = [
            {**{number.key: knob(number, partial(self._tuned, op, number)) for number in DIALS},
             "env": self.add(Envelope(OFF, {}, settled=partial(self._shaped, op), help=ENVELOPE_HELP))}
            for op in OPERATORS]
        self.octave = self.add(Stepper(OFF, [f"c{octave}" for octave in LOWEST], FIRST_OCTAVE, self._stepped,
                                       OCTAVE_HELP))
        self.keys = self.add(Keys(OFF, 12 * (LOWEST[FIRST_OCTAVE] + 1), self._play, KEYS_HELP))
        self.scope = self.add(Scope(OFF, theme.INSTRUMENT, SCOPE_HELP))
        self.spectrum = self.add(Spectrum(OFF, theme.INSTRUMENT, SPECTRUM_HELP))
        shown: dict[str, list[Widget]] = {
            "wave": [self.wave], "short": [self.short], **{key: [table] for key, table in self.tables.items()},
            "alg": [self.wiring], FEEDBACK.key: [self.feedback],
            "ops": [widget for column in self.operators for widget in column.values()]}
        for key, dial in self.knobs.items():
            shown.setdefault(key.split(".")[0], []).append(dial)
        self.settings: dict[str, tuple[Widget, ...]] = {key: tuple(widgets) for key, widgets in shown.items()}
        #: The name of the instrument that is shown, None for none, and what the heading says of its use.
        self.name: str | None = None
        self.used = ""
        #: The note that is heard again after a change: the one played last on the keyboard, whose key is lit.
        self.last = self.keys.lit = A4
        self.scroll = 0
        #: The widgets that have a part of a page each, by what `arrange` calls the part.
        self._placed: dict[str, Widget] = {"word": self.word, "wave": self.wave, "short": self.short, **self.tables,
                                           **self.knobs, "alg": self.wiring, FEEDBACK.key: self.feedback,
                                           "octave": self.octave, "keys": self.keys, "scope": self.scope,
                                           "spectrum": self.spectrum}
        for op, column in zip(OPERATORS, self.operators):
            self._placed.update({f"ops[{op}]": self.names[op], f"ops[{op}].does": self.roles[op],
                                 **{f"ops[{op}].{key}": widget for key, widget in column.items()}})
        # Every knob there is: each has a frame of its own to take.
        self._dials = [widget for widget in self.widgets if isinstance(widget, Knob)]
        # What was read last: the picked thing, the project it was read from, and the wave of the instrument.
        self._picked: tuple[str, str] | None = None
        self._project: Project | None = None
        self._wave = USUAL
        # Whether a widget under the hand was left as it was by the last read, and has still to show the file.
        self._waits = False
        # What each vibrato that was taken out here had beside its depth, by the name of its instrument.
        self._vibs: dict[str, dict] = {}
        self._parts = arrange(OFF)
        for key in PIANO:
            spot = (key, frozenset())
            ui.keys[spot] = partial(self._typed, key, ui.keys.get(spot))
        self._see()

    def show(self, on: bool) -> None:
        """Bring the panel onto the window, or take it off. Brought back, it shows what there is to show."""
        super().show(on)
        self._see()

    def place(self, rect: Rect) -> None:
        """Take the panel's rectangle, and lay the widgets out in it as `arrange` has them."""
        super().place(rect)
        self._lay()

    def update(self, passed: float = 0.0) -> None:
        """Give the knobs their frame, and read the instrument where it or the project is another than before.

        The frame of a knob ends a run of the wheel whose time is up, which saves it. Another thing picked
        takes every widget at once, and what the hand was doing is dropped with nothing saved. A project
        loaded again leaves the widget under the hand as it is, until the hand has let go.
        """
        model = self.session.model
        fresh = model.picked != self._picked
        for dial in self._dials:
            if fresh:
                dial.drop()
            dial.update()
        waited = self._waits and not any(self._hand(widget) for widget in self._placed.values())
        if fresh or waited or model.project is not self._project:
            self._read(fresh)

    def draw(self, canvas: Canvas) -> None:
        """Draw the two boxes, the heading in the upper one, and beside a page that is scrolled its thumb."""
        if self.name is None:
            return
        parts = self._parts
        with canvas.clip(self.rect):
            for box in (parts["head"], parts["box"]):
                canvas.box(box, theme.PANEL)
                canvas.frame(box, theme.LINE)
            x, y, w, h = parts["head"]
            at, last = x + HEAD_PAD, x + w - HEAD_PAD
            for text, colour in ((INSTRUMENT, theme.DIM), (self.name, theme.INSTRUMENT), (self.used, theme.DIM)):
                canvas.text_in(text, (at, y, max(0, last - at), h), colour)
                at += Font.width(text) + HEAD_SPACE
            reach = slack(self.rect, self._wave == "fm")
            if reach:
                x, y, w, h = parts["view"]
                long = max(theme.LIST_THUMB_MIN, h * h // (h + reach))
                canvas.box((x + w + 1, y + (h - long) * self.scroll // reach, THUMB_W, long), theme.LINE)

    def _hand(self, widget: Widget) -> bool:
        """Whether a widget is under the hand: pressed, a knob in a run of the wheel, or an envelope in a drag."""
        return self.ui.held is widget or getattr(widget, "busy", False)

    def _show(self, widget: Widget, what: str, value: object, forced: bool) -> None:
        """Have a widget show a value of the file, unless it is under the hand and the read is not `forced`."""
        if forced or not self._hand(widget):
            setattr(widget, what, value)
        else:
            self._waits = True

    def _read(self, forced: bool = False) -> None:
        """Set the widgets from what the project holds now for the picked instrument, and the heading with them.

        With `forced` every widget is set. Without, one that is under the hand is left to the hand. The page
        is laid out again, as its wave has it, and the scope and the spectrum are handed the sound as it is now.
        """
        model = self.session.model
        self._picked, self._project = model.picked, model.project
        kind, name = self._picked or (None, None)
        held = self._project.instruments.get(name) if kind == "instrument" else None
        self.name = None if held is None else name
        self._waits = False
        if held is not None:
            self.used = used(uses(self._project, name))
            self._wave = wave = held.get("wave", USUAL)
            self._show(self.wave, "picked", NAMES.index(wave), forced)
            self._show(self.short, "is_on", held.get("short", False), forced)
            self.short.enabled = wave == "noise"
            self.short.help = BUZZ_HELP if self.short.enabled else BUZZ_UNUSED
            for table in TABLES:
                widget = self.tables[table.key]
                self._show(widget, "table", held.get(table.key, table.absent), forced)
                widget.enabled = table.wave in (None, wave)
                widget.help = table.help if widget.enabled else table.unused
            for number in NUMBERS:
                self._show(self.knobs[number.key], "value", _held(held, number.key, number.absent), forced)
            for key in ("vib.delay", "vib.rate"):
                self.knobs[key].enabled = "vib" in held
            wired = held.get("alg", WIRING)
            self._show(self.wiring, "algorithm", wired, forced)
            self._show(self.feedback, "value", held.get(FEEDBACK.key, FEEDBACK.absent), forced)
            for op, column in zip(OPERATORS, self.operators):
                has = _held(held, f"ops[{op}]", SILENT)
                self.roles[op].text = role(wired, op)
                colour = theme.ACTIVE if op in wires(wired).heard else theme.SONG
                self.names[op].colour = self.roles[op].colour = column["env"].colour = colour
                self._show(column["env"], "op", has, forced)
                for number in DIALS:
                    self._show(column[number.key], "value", has.get(number.key, number.absent), forced)
        self._lay()
        self._trace()

    def _lay(self) -> None:
        """Give each widget its rectangle: its part of the page, scrolled no further than the page goes.

        The page is the one the wave of the instrument has, and a widget that page has no part for has no room.
        """
        fm = self._wave == "fm"
        self.scroll = min(max(self.scroll, 0), slack(self.rect, fm))
        self._parts = parts = arrange(self.rect, self.scroll, fm)
        for part, widget in self._placed.items():
            widget.rect = parts.get(part, OFF)
        self.page.rect = parts["box"]
        self._see()

    def _see(self) -> None:
        """Have each widget seen or not: by whether the panel is on the window, what there is to show, and the view.

        A part of the page is seen only when all of it is in view, and a widget with no room is not seen.
        """
        on = self.shown and self.name is not None
        view = self._parts["view"]
        for widget in self._placed.values():
            _, _, w, h = widget.rect
            widget.visible = on and w > 0 and h > 0 and within(widget.rect, view) == widget.rect
        self.page.visible = on

    def _trace(self) -> None:
        """Hand the scope and the spectrum the note played last, as the model makes it of the instrument now.

        The scope is told its pitch. A note that cannot be made is no sound to the two of them.
        """
        if self.name is None:
            return
        try:
            tone = self.session.model.tone(self.name, self.last)
            self.scope.show(tone["sides"][0], tone["rate"], self.last)
            self.spectrum.show(tone["sides"][0], tone["rate"])
        except RowSynthError:
            self.scope.show(NO_SOUND, DEFAULT_RATE)
            self.spectrum.show(NO_SOUND, DEFAULT_RATE)

    def _wheel(self, steps: int) -> bool:
        """Scroll the page by steps of the wheel, up for up, and say whether it moved."""
        before = self.scroll
        self.scroll -= steps * WHEEL_PX
        self._lay()
        return self.scroll != before

    def _change(self, key: str, value: object, words: str) -> bool:
        """Put a value at a place of the instrument that is shown, as one step, and show what the file then holds.

        `key` is the place in the instrument, "" for the whole of it, and `words` say what was done there. With
        NOTHING for the value what is there is taken out of the file. The answer is whether the file is as was
        asked. A change that is refused is not made, and the session's message is the refusal. After one that
        was made, the note played last is heard again, unless the session plays.
        """
        if self.name is None:
            return False
        where, label = f"instruments.{self.name}{'.' if key else ''}{key}", f"{self.name}: {words}"
        if value is not NOTHING:
            saved = self.session.change(where, value, label)
        else:
            # What the file does not have is not there to take out: it is as was asked already.
            held = self.session.model.project.instruments.get(self.name, {})
            saved = _held(held, key, NOTHING) is NOTHING or self.session.remove(where, label)
        self._read(forced=True)
        if saved and self.name is not None and not self.session.playing:
            self.session.sound(self.name, self.last)
        return saved

    def _waved(self, index: int) -> None:
        """Save the wave that was picked. Nothing else of the instrument is changed, whatever the wave.

        But an instrument with no operators would not be heard as FM: it is given those of `PLAIN` with that
        wave, and is written whole, so that the two are one step.
        """
        wave = NAMES[index]
        held = self.session.model.project.instruments.get(self.name, {})
        if wave == "fm" and not held.get("ops"):
            self._change("", {**held, "wave": wave, "ops": [dict(op) for op in PLAIN]}, f"wave {wave}")
        else:
            self._change("wave", wave, f"wave {wave}")

    def _buzzed(self, on: bool) -> None:
        """Save the short noise as it was switched: off is what the engine takes without it."""
        self._change("short", True if on else NOTHING, f"buzz {'on' if on else 'off'}")

    def _drawn(self, table: Table, steps: object) -> None:
        """Save a table as the mouse left it, in the shape the file takes."""
        alone = not isinstance(steps, (list, dict))
        self._change(table.key, NOTHING if alone and steps == table.absent else steps, f"{table.word} table")

    def _turned(self, number: Number, value: float) -> None:
        """Save the value a knob was left at, written as `tidy` has it and called by what the knob shows."""
        value = tidy(value)
        words = f"{number.words} {plain(value)}"
        if number.key == "vib.depth":
            self._wavered(value, words)
        else:
            self._change(number.key, NOTHING if value == number.absent else value, words)

    def _wavered(self, depth: float, words: str) -> None:
        """Save the depth of the vibrato: with none the vibrato is taken out, and with one again it is put back.

        What a vibrato had beside its depth when it was taken out here is kept, and is its again when it is
        given a depth.
        """
        held = self.session.model.project.instruments.get(self.name, {}).get("vib")
        if held is None and depth:
            self._change("vib", {**self._vibs.get(self.name, {}), "depth": depth}, words)
        elif depth:
            self._change("vib.depth", depth, words)
        elif self._change("vib", NOTHING, words) and held is not None:
            self._vibs[self.name] = {key: part for key, part in held.items() if key != "depth"}

    def _wired(self, algorithm: int) -> None:
        """Save the wiring that was picked, at its one place."""
        self._change("alg", algorithm, f"wiring {algorithm}")

    def _tuned(self, op: int, number: Number, value: float) -> None:
        """Save the value a knob of an operator was left at, as `_turned` saves one of the instrument."""
        value = tidy(value)
        self._operate(op, number.key, NOTHING if value == number.absent else value, f"{number.words} {plain(value)}")

    def _shaped(self, op: int, has: dict) -> None:
        """Save the envelope of an operator as a drag left it, called by the point that was dragged.

        `has` is the operator as the envelope has it. What the drag set to the value the engine takes without
        it is not written, and an envelope with nothing left to write is taken out.
        """
        held = self.session.model.project.instruments.get(self.name, {})
        was, now = settings(_held(held, f"ops[{op}]", SILENT)), settings(has)
        moved = {key for key in now if now[key] != was[key]}
        point = next((point.name for point in POINTS if moved & {point.time, point.level}), SHAPE)
        env = {key: value for key, value in has.get("env", {}).items()
               if key not in moved or value != ENVELOPE[key]}
        self._operate(op, "env", env or NOTHING, point)

    def _operate(self, op: int, key: str, value: object, words: str) -> None:
        """Put a value at a place of an operator, or take what is there out with NOTHING, as one step.

        An operator the file leaves out is silent to the engine. It is written as that silent one with the
        value in it, after silent ones for those before it that the file leaves out as well.
        """
        words = f"operator {op + FIRST} {words}"
        ops = self.session.model.project.instruments.get(self.name, {}).get("ops", [])
        if op < len(ops):
            self._change(f"ops[{op}].{key}", value, words)
        else:
            made = {name: part for name, part in {**SILENT, key: value}.items() if part is not NOTHING}
            self._change("ops", copy.deepcopy([*ops, *([SILENT] * (op - len(ops))), made]), words)

    def _stepped(self, index: int) -> None:
        """Move the keyboard to the octave the stepper was put on."""
        self.keys.low = 12 * (LOWEST[index] + 1)

    def _play(self, pitch: int) -> None:
        """Let a note of the instrument be heard, over whatever plays: it is the one played last, and is shown."""
        self.last = self.keys.lit = pitch
        if self.name is not None:
            self.session.sound(self.name, pitch)
            self._trace()

    def _typed(self, key: str, before: Callable[[], object] | None = None) -> None:
        """Play the note of a key of the computer, counted from the lowest key, while the panel has the focus.

        `before` is what the key did before the panel took it, and does where the panel has no focus.
        """
        if self.name is not None and self.ui.focus in self.widgets:
            self._play(self.keys.low + piano(key))
        elif before is not None:
            before()
