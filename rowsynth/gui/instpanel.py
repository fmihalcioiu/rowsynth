"""The middle of the window for an instrument: its wave, its settings, its tables or FM voice, and keys to try it."""
from __future__ import annotations

import copy
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from ..edit import NOTHING, steps
from ..errors import RowSynthError
from ..fm import CARRIER, ENVELOPE, FEEDBACK_OPERATOR, _SILENT, _algorithm
from ..note import DEFAULT_RATE
from ..osc import WAVES
from . import theme
from .channelbar import tidy
from .dials import Knob, Stepper, plain
from .fmview import FIRST, POINTS, WIRING_H, Envelope, Wiring, settings, wires
from .font import Font
from .kit import ALONG, Bar, Heading, ScrollBar, Slot, told, wheel
from .layout import Rect
from .model import inst_of, inst_place, own
from .panels import Panel, within
from .scope import TEXT_H, Scope, Spectrum
from .session import piano
from .tables import INSET, MOST, Map, StepTable, counted, fit
from .ui import NOWHERE, Choice, Event, Label, Pos, Toggle, Ui, Widget, inside, share

if TYPE_CHECKING:
    from ..project import Project
    from .canvas import Canvas
    from .session import Session

#: From the sides of the panel to its parts, and from one part to the next: as the panel of a song has them.
MARGIN = 4
GAP = 2
#: From one row of the main area to the next, and from the ruler to the first row under it.
ROW_GAP = 3
#: How high the ruler of the steps is, and every how many steps it has a number.
RULER_H = 10
RULER_EVERY = 8
#: In the side column: from its frame to the words of a row, and from the top of a row to its name.
SIDE_PAD = 5
SIDE_TOP = 3
#: How high the row of a table is at the least, and that of an operator: its envelope is no lower.
TABLE_MIN = 36
ENV_MIN = 45
#: A knob: how high it is with its two lines, and how wide at the least in the settings bar. A knob there whose
#: name is wider has the room of its name, and a pixel free at each end.
KNOB_H = theme.BAR_ROW_H
KNOB_W = 33
#: The knobs of an operator, which stand in its row before its envelope: how wide each is, and between two.
DIAL_W = 37
DIAL_GAP = 4
#: From the knobs of an operator to its envelope, and between two parts of the foot.
SPACE = 6
#: How far apart the pictures of the wirings are: they share the width of a row, which is just enough for them.
WIRING_GAP = 0
#: In the foot: from its frame to what stands in it, from side to side. The keyboard: how high, how wide at the
#: most, how many octaves, and how wide the stepper of its octave. It is as high as the scope and the spectrum
#: beside it need to be read, and those two share the room by these shares.
FOOT_PAD = 6
KEYS_H = 56
KEYS_W = 196
OCTAVES = 2
OCTAVE_W = 36
BESIDE = (2, 3)
#: The rectangle of a widget that has no room: one for which the panel as it is laid out has no part.
OFF = (0, 0, 0, 0)

#: What the heading reads before the name, the switch of the short noise with its two states, and the word over
#: the keyboard.
INSTRUMENT = "INSTRUMENT"
SHORT = "SHORT NOISE"
IS_ON, IS_OFF = "ON", "OFF"
TRY = "TRY IT"
#: What the side column calls the row of the wirings, and what the picked one is called: under its name, and in
#: a step that can be taken back.
WIRING = "WIRING"
ALGORITHM = "algorithm"
#: What the side column calls an operator before its number, and under that what the operator does in the wiring.
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
USUAL_WIRING = _algorithm({})
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


def turned(what: str, back: str) -> str:
    """The line of help of a knob: what a drag of it is for, what a right press puts it back to, and the wheel."""
    return told(f"drag up or down {what}, slower with shift", f"back to {back}", "steps it")


@dataclass(frozen=True)
class Table:
    """One step table an instrument can have: its key in the file, and how it is shown.

    text is its name in the side column, and in small letters what a change to it is called in a step that can
    be taken back. word is what one of its values is called. low and high are the range the mouse draws in, and
    absent what the engine takes for every tick where the instrument has no such table. help is what it does to
    the sound. wave is the one wave that uses it, None for all of them, and unused what its help line says
    under any other wave.
    """

    key: str
    text: str
    word: str
    low: float
    high: float
    absent: float
    help: str
    wave: str | None = None
    unused: str = ""


@dataclass(frozen=True)
class Number:
    """One number of an instrument that a knob holds: its place in the instrument, and how it is shown.

    text is the knob's name and words what the setting is called in a step that can be taken back. The knob
    goes from low to high in so many steps. absent is what the engine takes where the instrument has no such
    number: a knob put there is taken out of the file. needs is what the help line says while the knob takes
    no mouse, for one that has such times.
    """

    key: str
    text: str
    words: str
    low: float
    high: float
    steps: int
    absent: float
    help: str
    needs: str = ""


#: The step tables, from the top. A file may hold any number in most of them, so the ranges are chosen here:
#: the volume and the width are the engine's own, an arpeggio and a pitch go an octave each way, and the cutoff
#: goes up to 8000 Hz, with 0 for none.
TABLES = (
    Table("vol", "VOLUME SHAPE", "volume", 0, 15, 15, "how loud a note is tick by tick, from 0 to 15"),
    Table("duty", "PULSE WIDTH", "pulse width", 0, 3, 2,
          "the shape of the pulse tick by tick: 0 is thin and sharp, 1 fuller, 2 square and hollow, 3 sounds like 1",
          "pulse", "only the pulse wave has a pulse width: with another wave this table does nothing"),
    Table("arp", "ARPEGGIO", "arpeggio", -12, 12, 0,
          "semitones above or below its note that the sound jumps to at each tick: a fast round is heard as a chord"),
    Table("pitch", "PITCH", "pitch", -12, 12, 0,
          "semitones the note moves by at each tick and keeps: under the line it falls, as a kick or a zap does"),
    Table("lp", "CUTOFF", "cutoff", 0, 8000, 0,
          "how much of the high end is left tick by tick: all above this many hertz is dulled, and 0 dulls nothing"),
)
#: What the two other knobs of the vibrato say while it has no depth.
NO_VIBRATO = "there is no vibrato to set: turn VIBRATO DEPTH up first"
#: The numbers, in the order they stand in the settings bar. The three of the vibrato are parts of one setting,
#: which is there with its depth.
NUMBERS = (
    Number("gain", "VOLUME", "volume", 0, 2, 200, 1,
           turned("for how loud the instrument is beside the others", "1")),
    Number("release", "RELEASE", "release", 0, 60, 60, 2,
           turned("for the ticks a note takes to fade once let go", "2")),
    Number("gate", "GATE", "gate", 0.05, 1, 95, 0.9,
           turned("for the share of its length a song holds a note", "0.9")),
    Number("slide", "SLIDE", "slide", -24, 24, 96, 0,
           turned("for the semitones a second a note keeps sliding", "0")),
    Number("vib.depth", "VIBRATO DEPTH", "vibrato depth", 0, 2, 200, 0,
           turned("for how far the note wavers, in semitones", "0")),
    Number("vib.delay", "DELAY", "vibrato delay", 0, 60, 60, 0,
           turned("for the ticks before a note starts to waver", "0"), NO_VIBRATO),
    Number("vib.rate", "RATE", "vibrato rate", 1, 20, 38, 6,
           turned("for how many times a second the note wavers", "6"), NO_VIBRATO),
    Number("res", "RESONANCE", "resonance", 0, 0.98, 98, 0,
           turned("for how the cutoff rings, heard with a CUTOFF table", "0")),
    Number("key_scale", "KEY TRACKING", "key tracking", -12, 12, 48, 0,
           turned("for the decibels louder each higher octave is", "0")),
)
#: The feedback of the FM wave, which stands last in the settings bar, and the numbers of an operator before its
#: envelope, from the left. A file may hold any number in them, so the ranges are chosen here: they reach past
#: everything the examples use.
FEEDBACK = Number("fb", "FEEDBACK", "feedback", 0, 3, 60, 0,
                  turned(f"for how far operator {FEEDBACK_OPERATOR + FIRST} bends its own wave", "0"))
DIALS = (
    Number("mul", "RATIO", "ratio", 0, 16, 64, 1, turned("for its pitch as so many times that of the note", "1")),
    Number("det", "DETUNE", "detune", -10, 10, 200, 0, turned("for the hertz added to its pitch", "0")),
    Number("level", "VOLUME", "volume", 0, 8, 160, 1,
           turned("for how loud it is, or how far it bends another", "1")),
)
#: What the switch of the short noise says of itself, under the noise wave and under any other.
SHORT_HELP = told("switch the short noise on or off: it is a buzz with a pitch to it")
SHORT_UNUSED = "only the noise wave has a short noise: with another wave this does nothing"
#: What the choice of the wave and the stepper of the octave say of themselves, and the keyboard after the key.
WAVE_HELP = told("pick what the sound is made of: PULSE is hollow, SAW bright, TRI soft, NOISE a hiss, FM four sines")
OCTAVE_HELP = told("an arrow moves the keyboard an octave: its lowest key is this c", wheel="steps it")
KEYS_HELP = "and a drag plays each key it comes onto"
#: What the wirings and the envelope of an operator say of themselves, off their pictures and off their points, and
#: the scope and the spectrum while they have no sound to show.
WIRING_HELP = "which operators are heard and which change which"
ENVELOPE_HELP = "how loud the operator is over a note: the points set its attack, decay, fall while held and release"
SCOPE_HELP = "a few cycles of the note played last, as the instrument sounds now"
SPECTRUM_HELP = "what is in the note played last from low to high, and where it begins to be harsh"
#: What a press does on the ground of the panel, where the mouse is on none of its parts, and on the scroll bar.
PAGE_HELP = "the two rows of piano keys of the computer play the instrument"
BAR_HELP = "drag the bar to go along them, or press beside it to jump"
#: What the wheel does over the main area, as the help lines say it: with the tables in it, and with an FM voice.
WHEEL_TABLES = "along the tables"
WHEEL_VOICE = "along the operators"

#: The tables an FM voice reads: all but the width of a pulse.
FM_TABLES = tuple(table for table in TABLES if table.wave in (None, "fm"))
#: What the scope and the spectrum are handed for a note that could not be made: no sound.
NO_SOUND = np.zeros(0, dtype=np.float32)


def slot_w(name: str) -> int:
    """How wide the room of a control is in the settings bar: the least a knob has, or what its name needs."""
    return max(KNOB_W, Font.width(name) + 2)


def beside(line: str, wheeled: str) -> str:
    """A line of help that says something of a place, and after it what the wheel does there."""
    return f"{line}   {told(wheel=wheeled)}"


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
    """What an operator does in a wiring, as the side column says it under its name: `HEARD`, or `CHANGES 2, 3`."""
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


def uses(project: Project, name: str, of: str | None = None) -> dict[str, dict[str, int]]:
    """Where an instrument of a project is used, counted for each kind of use.

    "channels" is from a song to how many of its channels have the instrument as their own, and "switches" to
    how many more of its channels switch to it somewhere in their text. "drums" is from a song to how many of
    its drum letters play it, and "sounds" from a sound to how many of its voices do. Only what uses it is in
    a count. A name that a song gives to an instrument of the project stands for that one, and a name of an
    instrument that is the song's own is none of the project's.

    With `of` the instrument is one of that song's own: only the song uses it, by the name it has there.
    """
    found: dict[str, dict[str, int]] = {"channels": {}, "switches": {}, "drums": {}, "sounds": {}}
    for title, song in project.songs.items():
        if of not in (None, title):
            continue
        own = {} if of else song.get("instruments", {})

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
    for title, entry in ({} if of else project.sounds).items():
        count = sum(voice["inst"] == name for voice in _voices(entry))
        if count:
            found["sounds"][title] = count
    return found


def used(found: dict[str, dict[str, int]], named: bool = True) -> str:
    """What `uses` found in a few words, as the heading says it: `in 3 channels of tune, 2 sounds`.

    A channel that switches to the instrument is a channel of it. Used by one song, that song is named, and by
    more their count is. An instrument nothing plays is `used nowhere`. Without `named` no song is named, as in
    `in 3 channels`: that is for an instrument of a song's own, whose heading has named its song already.
    """
    def some(count: int, word: str) -> str:
        return f"{count} {word}{'' if count == 1 else 's'}"

    def of(counts: dict[str, int], word: str) -> str:
        where = next(iter(counts)) if len(counts) == 1 else f"{len(counts)} songs"
        return some(sum(counts.values()), word) + (f" of {where}" if named else "")

    channels = {song: found["channels"].get(song, 0) + found["switches"].get(song, 0)
                for song in (*found["channels"], *found["switches"])}
    parts = [of(counts, word) for counts, word in ((channels, "channel"), (found["drums"], "drum")) if counts]
    if found["sounds"]:
        parts.append(some(len(found["sounds"]), "sound"))
    return "in " + ", ".join(parts) if parts else NOWHERE_USED


def heading(name: str, song: str | None, said: str) -> tuple[str, str]:
    """What the heading reads after `INSTRUMENT`: the name, and the facts, which are what is said of its use.

    An instrument of a song's own has its song before those, as in `INSTRUMENT lead  of tune  in 2 channels`.
    """
    return name, said if song is None else f"of {song}  {said}"


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


@dataclass(frozen=True)
class Row:
    """One row of the main area: what it is called, how high it is at the least, and whether it grows.

    A row that grows has a share of the height the view has to spare when all the rows are in it.
    """

    key: str
    least: int
    grows: bool = True


def rows(fm: bool = False) -> tuple[Row, ...]:
    """The rows of the main area from the top: the tables, each by its key.

    With `fm` they are those of an FM instrument: the wirings ("alg"), the operators ("ops[0]" and so on), the
    ruler of the steps ("ruler"), and under it the tables an FM voice reads.
    """
    if not fm:
        return tuple(Row(table.key, TABLE_MIN) for table in TABLES)
    return (Row("alg", WIRING_H, False), *(Row(f"ops[{op}]", ENV_MIN) for op in OPERATORS),
            Row("ruler", RULER_H, False), *(Row(table.key, TABLE_MIN) for table in FM_TABLES))


def heights(found: Sequence[Row], room: int) -> list[int]:
    """How high each row is in a view `room` pixels high: its least, and a share of what is spare where all fit."""
    spare = max(0, room - sum(row.least for row in found) - ROW_GAP * (len(found) - 1))
    more = iter(share(spare, [1] * sum(row.grows for row in found)))
    return [row.least + (next(more) if row.grows else 0) for row in found]


def offsets(tall: Sequence[int]) -> list[int]:
    """How many pixels of the rows lie over each of them, by their heights, the gap after each row counted."""
    found, over = [], 0
    for high in tall:
        found.append(over)
        over += high + ROW_GAP
    return found


def furthest(tall: Sequence[int], room: int) -> int:
    """The last row a view `room` pixels high can begin at: the first from which every row after it is in view.

    That is the first row when all of them fit, and the last one when not even that one fits.
    """
    starts = offsets(tall)
    end = starts[-1] + tall[-1] if tall else 0
    return next((row for row, start in enumerate(starts) if end - start <= room), max(0, len(tall) - 1))


def tops(tall: Sequence[int], first: int, room: int) -> list[int | None]:
    """How far under the top of a view each row begins, when the view begins at the row `first`.

    None is for a row that is not wholly in the view: one before the first, the one the view ends in, and every
    row after that one.
    """
    starts = offsets(tall)
    found: list[int | None] = []
    for row, (start, high) in enumerate(zip(starts, tall)):
        at = start - starts[first]
        found.append(at if row >= first and at + high <= room else None)
    return found


def skeleton(rect: Rect) -> dict[str, Rect]:
    """The parts every editor has, in the rectangle of the panel, each by what it is called.

    "head" is the heading, `theme.HEAD_H` high, and under it "bar" the settings bar, `theme.BAR_H` high. "foot"
    is at the lower end, `theme.FOOT_H` high. "main" is what is left between the bar and the foot: "side" is
    the column down its left, `theme.SIDE_W` wide, and "body" the rest of it.
    """
    x, y, w, h = rect
    left, wide, end = x + MARGIN, w - 2 * MARGIN, y + h - MARGIN
    bar = y + MARGIN + theme.HEAD_H + GAP
    top = bar + theme.BAR_H + GAP
    foot = max(top, end - theme.FOOT_H)
    high = max(0, foot - GAP - top)
    return {"head": (left, y + MARGIN, wide, theme.HEAD_H), "bar": (left, bar, wide, theme.BAR_H),
            "main": (left, top, wide, high), "side": (left, top, theme.SIDE_W, high),
            "body": (left + theme.SIDE_W + GAP, top, max(0, wide - theme.SIDE_W - GAP), high),
            "foot": (left, foot, wide, theme.FOOT_H)}


def view(rect: Rect, fm: bool = False) -> Rect:
    """Where the rows stand in the body of a panel: beside the scroll bar, and under the ruler along its top.

    With `fm` the ruler is no part of the body: it is a row then, and stands over the first table.
    """
    x, y, w, h = skeleton(rect)["body"]
    ruled = 0 if fm else RULER_H + ROW_GAP
    return x, y + ruled, max(0, w - theme.SCROLL_W - theme.SCROLL_GAP), max(0, h - ruled)


def reach(rect: Rect, fm: bool = False) -> int:
    """The last row the view of a panel can begin at: how many steps of the wheel its rows go. 0 where all fit."""
    room = view(rect, fm)[3]
    return furthest(heights(rows(fm), room), room)


def _row(key: str, row: Rect, side: int) -> dict[str, Rect]:
    """The parts of one row that stands in a rectangle, with its two lines in the side column, which begins at `side`.

    The ruler is the row itself, and has no lines. Any other row has its name and under it its facts. The row of
    an operator has its knobs side by side from the left, in the middle of its height, and after them its envelope.
    """
    x, y, w, h = row
    if key == "ruler":
        return {key: row}
    name = (side + SIDE_PAD, y + SIDE_TOP, theme.SIDE_W - 2 * SIDE_PAD, Font.LINE)
    parts = {key: row, f"{key}.name": name, f"{key}.facts": (name[0], name[1] + Font.LINE, name[2], Font.LINE)}
    if key.startswith("ops"):
        for place, number in enumerate(DIALS):
            parts[f"{key}.{number.key}"] = (x + place * (DIAL_W + DIAL_GAP), y + (h - KNOB_H) // 2, DIAL_W, KNOB_H)
        taken = len(DIALS) * (DIAL_W + DIAL_GAP) - DIAL_GAP + SPACE
        parts[f"{key}.env"] = (x + taken, y, max(0, w - taken), h)
    return parts


def arrange(rect: Rect, first: int = 0, fm: bool = False) -> dict[str, Rect]:
    """Where the parts of the panel go in its rectangle, each by what it is called, and each cut to the panel.

    The parts of `skeleton` are there. In the body stand "view", where the rows are, as `view` has it, "scroll",
    the bar down its right side, and "ruler", the ruler of the steps along its top.

    The rows are those of `rows`, as high as `heights` has them in the view. They stand one under the other from
    the row `first`, which is no further than `furthest` lets it be, and only a row that is wholly in the view is
    there. A row is there by its key: the rectangle of a table, of the wirings ("alg"), and of all of an
    operator ("ops[0]"). Beside it in the side column are its name ("vol.name") and under that its facts
    ("vol.facts"). An operator has its knobs ("ops[0].mul" and so on) and after them its envelope ("ops[0].env").
    With `fm` the ruler is one of the rows, and is there while it is in view.

    In the foot stand "try", the word over "octave", then "keys", and beside the keyboard "scope" and "spectrum".
    The spectrum is as high as the keyboard, and the scope as high as the box of the spectrum, which has the
    names of its marks under it.
    """
    placed = skeleton(rect)
    left, wide = placed["main"][0], placed["main"][2]
    body_x, body_y, body_w, _ = placed["body"]
    x, y, across, room = placed["view"] = view(rect, fm)
    placed["scroll"] = (body_x + body_w - theme.SCROLL_W, y, theme.SCROLL_W, room)
    if not fm:
        placed["ruler"] = (x, body_y, across, RULER_H)
    found = rows(fm)
    tall = heights(found, room)
    first = min(max(first, 0), furthest(tall, room))
    for row, at, high in zip(found, tops(tall, first, room), tall):
        if at is not None:
            placed.update(_row(row.key, (x, y + at, across, high), left))

    at, top = left + FOOT_PAD, placed["foot"][1] + (theme.FOOT_H - KEYS_H) // 2
    end = left + wide - FOOT_PAD
    keys = max(0, min(KEYS_W, end - at - OCTAVE_W - SPACE))
    after = at + OCTAVE_W + keys + 2 * SPACE
    scope, spectrum = share(max(0, end - after - SPACE), BESIDE)
    placed["try"] = (at, top, OCTAVE_W, Font.LINE)
    placed["octave"] = (at, top + Font.LINE + SPACE, OCTAVE_W, theme.BUTTON_H)
    placed["keys"] = (at + OCTAVE_W + SPACE, top, keys, KEYS_H)
    placed["scope"] = (after, top, scope, KEYS_H - TEXT_H)
    placed["spectrum"] = (after + scope + SPACE, top, spectrum, KEYS_H)
    return {part: within(place, rect) for part, place in placed.items()}


class Page(Widget):
    """The ground the parts of a panel stand on. It takes the press and the wheel that no part of the panel used.

    A press gives it the focus, so that the keys go to the panel. `wheeled` is handed a step of the wheel, and
    says whether it was used. `area` is where the wheel does something: there the help line says `wheel_help`
    after what a press does, which is the line the ground was made with.
    """

    def __init__(self, rect: Rect, wheeled: Callable[[Event], bool], help: str = "") -> None:
        super().__init__(rect, help)
        self.wheeled = wheeled
        self.area: Rect = (0, 0, 0, 0)
        self.wheel_help = ""
        self._mouse: Pos = NOWHERE

    @property
    def help(self) -> str:
        """What a press does on the ground, and over `area` what the wheel does as well."""
        return told(self._help, wheel=self.wheel_help if inside(self.area, self._mouse) else "")

    @help.setter
    def help(self, line: str) -> None:
        self._help = line

    def on(self, event: Event, ui: Ui) -> bool:
        """Note where the mouse is, use a press, and hand a step of the wheel on."""
        if event.kind == "key":
            return False
        self._mouse = event.pos
        if event.kind == "wheel":
            return self.wheeled(event)
        return event.kind == "down"


class Waves(Choice):
    """A choice of waves, each by its name in capitals: `waves` has the names as the engine knows them."""

    def __init__(self, rect: Rect, waves: tuple[str, ...], changed: Callable[[int], object] | None = None,
                 help: str = "") -> None:
        super().__init__(rect, [wave.upper() for wave in waves], 0, changed, help)
        self.waves = tuple(waves)


class Keys(Widget):
    """A piano keyboard of `OCTAVES` octaves from `low`, the pitch of its lowest key, which is a c.

    A press of the left button on a key calls `played` with its pitch, and so does each other key the mouse
    comes onto while the button is kept pressed: a key is played once for as long as the mouse stays on it.
    `lit` is the pitch of the key that is drawn lit, None for none: the one played last, here or by its owner.
    The help line says that a press lets the key under the mouse be heard, by its name, and after that the
    words the keyboard was made with.
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
        """What a press does on the key under the mouse, which is named, or on a key that is not."""
        pitch = key_at(self.rect, self.low, self._mouse)
        return told(f"hear {'a key' if pitch is None else note_name(pitch)}, {self._help}")

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

    It is laid out as every editor is. `heading` says what instrument it is, and in dim words where it is
    used. An instrument of a song's own is shown and changed as one of the project's is: `song` is then its
    song, which the heading names after it, and it is changed at its place in the song and heard as the song
    has it. The heading names that song once: where the instrument is used is then said with no song named.
    At the right end of the heading `wave` is a choice of the waves the engine has.

    `frame` is the settings bar: `knobs` are the numbers of the instrument, each by its place in it, and after
    them stands `short`, the switch of the short noise. The main area has a side column and a body. `tables`
    are the step tables an instrument can have, each by its key, one under the other in the body as rows of
    bars, under a ruler that numbers their steps. The side column has for each its name and under that the
    count of its steps, which are `titles` and `facts`. Every table has the same room for its steps, so a step
    is as wide in one as in another, and a table that is scrolled sideways takes the others and the ruler
    along. In the foot `keys` is a keyboard that plays the instrument, with `octave` to move it, and the keys
    of the computer play it as a piano while a part of the panel has the focus: a press anywhere on the panel
    gives it that. Beside the keyboard `scope` and `spectrum` show the note played last, `last`, as the
    instrument sounds now: they are handed it when it is played, and again after each change to the file, made
    here or outside. `settings` is from each key an instrument can have in the file to the widgets that show
    it. With a song, a sound or nothing picked the panel shows nothing.

    A table that the picked wave does not use is dim and takes no mouse, and so is the switch. The two other
    knobs of a vibrato that has no depth are dim as well.

    While the wave is FM the main area has other rows, and the settings bar has `feedback`, the knob of the
    feedback, after the switch. The first row is `wiring`, the wirings of the four operators as pictures, of
    which a click picks one: the side column names the row, and under that the picked one. Then each operator
    has a row: its knobs, one for each number it has beside the envelope, and its envelope with points to
    drag. The side column has its number and what it does in the picked wiring (`names` and `roles`), in the
    colour the pictures have for one that is heard or for one that changes another, and the envelope is in
    that colour too. `operators` has, for each operator, the widget of each key an operator can have in the
    file. An operator the file leaves out is shown as the engine takes it, silent. Under the operators stands
    the ruler, and under that the tables an FM voice reads: the one it does not read is not there.

    Rows that do not fit are not dropped. `first` is the row the view begins at, and only rows that are wholly
    in view are seen. The wheel over the main area goes along the rows, as `kit.wheel` reads it, and changes no
    value there: over a knob it turns the knob. `along` is the scroll bar down the side of the body, which
    shows where the view is and can be dragged.

    Each change is one change to the file at the place of the instrument, and one step that can be taken back:
    the wave, the switch and the wiring as they are picked, a table and an envelope when the mouse lets go of
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
    """

    def __init__(self, session: Session, ui: Ui, clock: Callable[[], float] = time.perf_counter) -> None:
        """`clock` times a run of the wheel on a knob."""
        super().__init__(session, ui)

        def knob(number: Number, settled: Callable[[float], object]) -> Knob:
            """A knob of the panel for a number: `settled` is told what it was left at."""
            return self.add(Knob(OFF, number.absent, number.low, number.high, default=number.absent,
                                 steps=number.steps, colour=theme.INSTRUMENT, text=number.text, help=number.help,
                                 settled=settled, clock=clock))

        # The ground is under everything, and the heading and the bar under what stands on them.
        self.page = self.add(Page(OFF, self._wheel, PAGE_HELP))
        self.heading = self.add(Heading(OFF, INSTRUMENT, theme.INSTRUMENT))
        self.wave = self.add(Waves(OFF, NAMES, self._waved, WAVE_HELP))
        self.frame = self.add(Bar(OFF))
        self.knobs: dict[str, Knob] = {number.key: knob(number, partial(self._turned, number)) for number in NUMBERS}
        self.short = self.add(Toggle(OFF, IS_OFF, changed=self._switched, colour=theme.INSTRUMENT, help=SHORT_HELP))
        self.feedback = knob(FEEDBACK, partial(self._turned, FEEDBACK))
        self.tables: dict[str, StepTable] = {
            table.key: self.add(StepTable(OFF, table.absent, table.low, table.high, partial(self._drawn, table),
                                          theme.INSTRUMENT, help=table.help, word=table.word))
            for table in TABLES}
        for table in self.tables.values():
            table.moved = self._went
        self.wiring = self.add(Wiring(OFF, USUAL_WIRING, self._wired, WIRING_HELP, WIRING_GAP))
        self.titles: dict[str, Label] = {key: self.add(Label(OFF, text, theme.TEXT))
                                         for key, text in ((*((table.key, table.text) for table in TABLES),
                                                            ("alg", WIRING)))}
        self.facts: dict[str, Label] = {key: self.add(Label(OFF, "")) for key in self.titles}
        self.names = [self.add(Label(OFF, f"{OPERATOR} {op + FIRST}")) for op in OPERATORS]
        self.roles = [self.add(Label(OFF, "")) for _ in OPERATORS]
        self.operators: list[dict[str, Widget]] = [
            {**{number.key: knob(number, partial(self._tuned, op, number)) for number in DIALS},
             "env": self.add(Envelope(OFF, {}, settled=partial(self._shaped, op), help=ENVELOPE_HELP))}
            for op in OPERATORS]
        self.along = self.add(ScrollBar(OFF, self._to))
        self.word = self.add(Label(OFF, TRY))
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
        #: The name of the instrument that is shown, None for none, the song whose own it is, None for one of
        #: the project's, and where it is used, in words.
        self.name: str | None = None
        self.song: str | None = None
        self.used = ""
        #: The note that is heard again after a change: the one played last on the keyboard, whose key is lit.
        self.last = self.keys.lit = A4
        #: The row the view of the main area begins at.
        self.first = 0
        #: The widgets that `arrange` has a part for, each by what the part is called.
        self._placed: dict[str, Widget] = {**self.tables, "alg": self.wiring, "scroll": self.along,
                                           "try": self.word, "octave": self.octave, "keys": self.keys,
                                           "scope": self.scope, "spectrum": self.spectrum}
        for key in self.titles:
            self._placed.update({f"{key}.name": self.titles[key], f"{key}.facts": self.facts[key]})
        for op, column in zip(OPERATORS, self.operators):
            self._placed.update({f"ops[{op}].name": self.names[op], f"ops[{op}].facts": self.roles[op],
                                 **{f"ops[{op}].{key}": widget for key, widget in column.items()}})
        # What stands in the settings bar, which lays it out, and every knob there is: each has a frame to take.
        self._barred: tuple[Widget, ...] = (*self.knobs.values(), self.short, self.feedback)
        self._dials = [widget for widget in self.widgets if isinstance(widget, Knob)]
        # What was read last: the picked thing, the project it was read from, and the wave of the instrument.
        self._picked: tuple[str, str] | None = None
        self._project: Project | None = None
        self._wave = USUAL
        # Whether a widget under the hand was left as it was by the last read, and has still to show the file.
        self._waits = False
        # What each vibrato that was taken out here had beside its depth, by the place of its instrument.
        self._vibs: dict[str, dict] = {}
        # How many steps the tables make room for, and how many pixels of their bars are scrolled away.
        self._span = MOST
        self._across = 0
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
        loaded again leaves the widget under the hand as it is, until the hand has let go. The side column
        counts the steps of each table as it is drawn, also while its handle is dragged.
        """
        model = self.session.model
        fresh = model.picked != self._picked
        for dial in self._dials:
            if fresh:
                dial.drop()
            dial.update()
        waited = self._waits and not any(self._hand(widget) for widget in self.widgets)
        if fresh or waited or model.project is not self._project:
            self._read(fresh)
        self._count()

    def draw(self, canvas: Canvas) -> None:
        """Draw the frames of the side column and of the foot, and the ruler with the numbers of its steps."""
        if self.name is None:
            return
        parts = self._parts
        with canvas.clip(self.rect):
            for part in ("side", "foot"):
                canvas.box(parts[part], theme.PANEL)
                canvas.frame(parts[part], theme.LINE)
            ruler = parts.get("ruler", OFF)
            seen = self._steps()
            canvas.box(ruler, theme.PANEL)
            with canvas.clip(ruler):
                for step in range(0, self._span, RULER_EVERY):
                    canvas.text_in(str(step + 1), (seen.x(step), ruler[1], RULER_EVERY * seen.pitch, ruler[3]),
                                   theme.DIM)

    def _steps(self) -> Map:
        """Where the steps are from side to side of the view: as every table has them, and the ruler with them."""
        seen = self._parts["view"]
        return Map(seen, 0, 1, fit(max(0, seen[2] - 2 * INSET), self._span), self._across).within(self._span)

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

        With `forced` every widget is set. Without, one that is under the hand is left to the hand. The panel
        is laid out again, as its wave has it, and the scope and the spectrum are handed the sound as it is now.
        """
        model = self.session.model
        self._picked, self._project = model.picked, model.project
        kind, name = self._picked or (None, None)
        song, name = own(name, self._project.instruments) if kind == "instrument" else (None, None)
        held = inst_of(self._project, song, name)
        self.song, self.name = (None, None) if held is None else (song, name)
        self._waits = False
        if held is not None:
            found = uses(self._project, name, song)
            self.used = used(found)
            self.heading.show(*heading(name, song, used(found, song is None)))
            self._wave = wave = held.get("wave", USUAL)
            wheeled = WHEEL_VOICE if wave == "fm" else WHEEL_TABLES
            self.page.wheel_help = self.wiring.wheel_help = wheeled
            self.along.help = told(BAR_HELP, wheel=wheeled)
            self.wiring.help = beside(WIRING_HELP, wheeled)
            self._show(self.wave, "picked", NAMES.index(wave), forced)
            self._show(self.short, "is_on", held.get("short", False), forced)
            self.short.text = IS_ON if self.short.is_on else IS_OFF
            self.short.enabled = wave == "noise"
            self.short.help = SHORT_HELP if self.short.enabled else SHORT_UNUSED
            for table in TABLES:
                widget = self.tables[table.key]
                self._show(widget, "table", held.get(table.key, table.absent), forced)
                widget.enabled = table.wave in (None, wave)
                widget.wheel_help = wheeled
                widget.help = beside(table.help if widget.enabled else table.unused, wheeled)
                self.titles[table.key].colour = theme.TEXT if widget.enabled else theme.DIM
            for number in NUMBERS:
                dial = self.knobs[number.key]
                self._show(dial, "value", _held(held, number.key, number.absent), forced)
                dial.enabled = not number.needs or "vib" in held
                dial.help = number.help if dial.enabled else number.needs
            wired = held.get("alg", USUAL_WIRING)
            self._show(self.wiring, "algorithm", wired, forced)
            self._show(self.feedback, "value", held.get(FEEDBACK.key, FEEDBACK.absent), forced)
            for op, column in zip(OPERATORS, self.operators):
                has = _held(held, f"ops[{op}]", SILENT)
                self.roles[op].text = role(wired, op)
                colour = theme.ACTIVE if op in wires(wired).heard else theme.SONG
                self.names[op].colour = self.roles[op].colour = column["env"].colour = colour
                column["env"].wheel_help = wheeled
                column["env"].help = beside(ENVELOPE_HELP, wheeled)
                self._show(column["env"], "op", has, forced)
                for number in DIALS:
                    self._show(column[number.key], "value", has.get(number.key, number.absent), forced)
        self._lay()
        self._count()
        self._trace()

    def _count(self) -> None:
        """Have the side column say the picked wiring, and how many steps each table has as it is drawn."""
        self.facts["alg"].text = f"{ALGORITHM} {self.wiring.algorithm}"
        for key, table in self.tables.items():
            self.facts[key].text = counted(len(table.steps))

    def _lay(self) -> None:
        """Give each widget its rectangle: its part of the panel, the rows from no further than they go.

        The rows are those the wave of the instrument has, and a widget that has no part has no room. The
        heading lays the choice of the wave out at its right end, and the settings bar its controls from its
        left, the feedback last and only for an FM voice. Every table is given the room of the longest of them,
        and of no fewer steps than the handle makes a table. The scroll bar is shown where the view is.
        """
        fm = self._wave == "fm"
        room = view(self.rect, fm)[3]
        tall = heights(rows(fm), room)
        far = furthest(tall, room)
        self.first = min(max(self.first, 0), far)
        self._parts = parts = arrange(self.rect, self.first, fm)
        self.page.rect, self.page.area = self.rect, parts["main"]
        self.heading.rect, self.frame.rect = parts["head"], parts["bar"]
        self.heading.right([self.wave])
        slots = [Slot(number.text, self.knobs[number.key], slot_w(number.text)) for number in NUMBERS]
        slots.append(Slot(SHORT, self.short, slot_w(SHORT)))
        self.feedback.rect = OFF
        self.frame.lay([*slots, Slot(FEEDBACK.text, self.feedback, slot_w(FEEDBACK.text))] if fm else slots)
        for part, widget in self._placed.items():
            widget.rect = parts.get(part, OFF)
        starts = offsets(tall)
        self.along.show(starts[far] + room, room, starts[self.first])
        self._span = max(MOST, *(len(table.steps) for table in self.tables.values()))
        self._across = self._steps().first
        for table in self.tables.values():
            table.span, table.first = self._span, self._across
        self._see()

    def _see(self) -> None:
        """Have each widget seen or not: by whether the panel is on the window, what there is to show, and its room.

        A widget with no room is not seen: a row that is not wholly in view, a part of the other wave, and a
        control the settings bar is too narrow for.
        """
        on = self.shown and self.name is not None
        for widget in (*self._placed.values(), *self._barred):
            widget.visible = on and widget.rect[2] > 0 and widget.rect[3] > 0
        for widget in (self.page, self.heading, self.wave, self.frame):
            widget.visible = on

    def _trace(self) -> None:
        """Hand the scope and the spectrum the note played last, as the model makes it of the instrument now.

        The scope is told its pitch. A note that cannot be made is no sound to the two of them.
        """
        if self.name is None:
            return
        try:
            tone = self.session.model.tone(self.name, self.last, song=self.song)
            self.scope.show(tone["sides"][0], tone["rate"], self.last)
            self.spectrum.show(tone["sides"][0], tone["rate"])
        except RowSynthError:
            self.scope.show(NO_SOUND, DEFAULT_RATE)
            self.spectrum.show(NO_SOUND, DEFAULT_RATE)

    def _wheel(self, event: Event) -> bool:
        """Go along the rows by a step of the wheel over the main area, up for up, and say whether they moved.

        Only a step that goes up and down is used, and it changes no value.
        """
        way, count = wheel(event)
        if way != ALONG or not inside(self._parts["main"], event.pos):
            return False
        before = self.first
        self.first -= count
        self._lay()
        return self.first != before

    def _to(self, offset: int) -> None:
        """Have the view begin at the row that starts nearest to where the scroll bar was taken, in pixels."""
        fm = self._wave == "fm"
        room = view(self.rect, fm)[3]
        tall = heights(rows(fm), room)
        starts = offsets(tall)
        self.first = min(range(furthest(tall, room) + 1), key=lambda row: abs(starts[row] - offset))
        self._lay()

    def _went(self, first: int) -> None:
        """Have every table, and the ruler with them, scrolled sideways as far as the one the wheel was on."""
        self._across = first
        for table in self.tables.values():
            table.first = first

    def _where(self, key: str = "") -> str:
        """The place of the instrument that is shown, or of a place in it: among the project's, or in its song."""
        return f"{inst_place(self.song, self.name)}{'.' if key else ''}{key}"

    def _inst(self) -> dict:
        """The instrument that is shown, as the project holds it now: one that is gone holds nothing."""
        return inst_of(self.session.model.project, self.song, self.name) or {}

    def _change(self, key: str, value: object, words: str) -> bool:
        """Put a value at a place of the instrument that is shown, as one step, and show what the file then holds.

        `key` is the place in the instrument, "" for the whole of it, and `words` say what was done there. With
        NOTHING for the value what is there is taken out of the file. The answer is whether the file is as was
        asked. A change that is refused is not made, and the session's message is the refusal. After one that
        was made, the note played last is heard again, unless the session plays.
        """
        if self.name is None:
            return False
        where, label = self._where(key), f"{self.name}: {words}"
        if value is not NOTHING:
            saved = self.session.change(where, value, label)
        else:
            # What the file does not have is not there to take out: it is as was asked already.
            saved = _held(self._inst(), key, NOTHING) is NOTHING or self.session.remove(where, label)
        self._read(forced=True)
        if saved and self.name is not None and not self.session.playing:
            self.session.sound(self.name, self.last, self.song)
        return saved

    def _waved(self, index: int) -> None:
        """Save the wave that was picked. Nothing else of the instrument is changed, whatever the wave.

        But an instrument with no operators would not be heard as FM: it is given those of `PLAIN` with that
        wave, and is written whole, so that the two are one step.
        """
        wave = NAMES[index]
        held = self._inst()
        if wave == "fm" and not held.get("ops"):
            self._change("", {**held, "wave": wave, "ops": [dict(op) for op in PLAIN]}, f"wave {wave}")
        else:
            self._change("wave", wave, f"wave {wave}")

    def _switched(self, on: bool) -> None:
        """Save the short noise as it was switched: off is what the engine takes without it."""
        self._change("short", True if on else NOTHING, f"{SHORT.lower()} {'on' if on else 'off'}")

    def _drawn(self, table: Table, steps: object) -> None:
        """Save a table as the mouse left it, in the shape the file takes."""
        alone = not isinstance(steps, (list, dict))
        self._change(table.key, NOTHING if alone and steps == table.absent else steps, f"{table.text.lower()} table")

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
        held, where = self._inst().get("vib"), self._where()
        if held is None and depth:
            self._change("vib", {**self._vibs.get(where, {}), "depth": depth}, words)
        elif depth:
            self._change("vib.depth", depth, words)
        elif self._change("vib", NOTHING, words) and held is not None:
            self._vibs[where] = {key: part for key, part in held.items() if key != "depth"}

    def _wired(self, algorithm: int) -> None:
        """Save the wiring that was picked, at its one place."""
        self._change("alg", algorithm, f"{ALGORITHM} {algorithm}")

    def _tuned(self, op: int, number: Number, value: float) -> None:
        """Save the value a knob of an operator was left at, as `_turned` saves one of the instrument."""
        value = tidy(value)
        self._operate(op, number.key, NOTHING if value == number.absent else value, f"{number.words} {plain(value)}")

    def _shaped(self, op: int, has: dict) -> None:
        """Save the envelope of an operator as a drag left it, called by the point that was dragged.

        `has` is the operator as the envelope has it. What the drag set to the value the engine takes without
        it is not written, and an envelope with nothing left to write is taken out.
        """
        was, now = settings(_held(self._inst(), f"ops[{op}]", SILENT)), settings(has)
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
        ops = self._inst().get("ops", [])
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
            self.session.sound(self.name, pitch, self.song)
            self._trace()

    def _typed(self, key: str, before: Callable[[], object] | None = None) -> None:
        """Play the note of a key of the computer, counted from the lowest key, while the panel has the focus.

        `before` is what the key did before the panel took it, and does where the panel has no focus.
        """
        if self.name is not None and self.ui.focus in self.widgets:
            self._play(self.keys.low + piano(key))
        elif before is not None:
            before()
