"""The small language a song's channels are written in."""
from __future__ import annotations

import re

from .errors import RowSynthError
from .notes import midi

# Everything a channel's text can hold, one kind to a line.
_TOKEN = re.compile("|".join((
    r"\s+",                                                           # space, which only keeps things apart
    r"\|",                                                            # a bar line, which changes nothing
    r"@(?P<inst>\w+)",                                                # the instrument from here on: @lead
    r"(?P<name>[a-gr])(?P<sign>[+#-]?)(?P<length>\d*)(?P<dots>\.*)",  # a note, or r for a rest: c f+8 b-4. r2
    r"(?P<drum>[kshHt])(?P<drum_length>\d*)(?P<drum_dots>\.*)",       # a drum hit, timed as a note is: k s8 H4.
    r"l(?P<default>\d+)",                                             # the length for notes written without one: l8
    r"o(?P<octave>\d)",                                               # the octave: o4
    r"v(?P<vol>\d+)",                                                 # the volume from here on, 0 to 15: v12
    r"(?P<mark>[<>&])",                                               # an octave down, an octave up, a tie
)), re.ASCII)

# Semitones from c up to each letter, and what a sharp or a flat adds.
_STEP = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
_SIGN = {"": 0, "+": 1, "#": 1, "-": -1}


def _unread(text: str, at: int) -> RowSynthError:
    """The error for text that cannot be read from `at` on, quoting its next twelve characters."""
    return RowSynthError(f'cannot read "{text[at:at + 12]}" in a channel')


def _number(octave: int, name: str, sign: str) -> int:
    """The number of a note: the c of octave 4 is 60."""
    return 12 * (octave + 1) + _STEP[name] + _SIGN[sign]


def _rows(length: int, dots: int, rows_per_beat: int) -> float:
    """How many rows a length comes to: 4 is a quarter note, and each dot adds half of what the one before added."""
    rows = rows_per_beat * 4 / length
    add = rows
    for _ in range(dots):
        add /= 2
        rows += add
    return rows


def parse(text: str, inst: str, rows_per_beat: int = 4, drums: dict | None = None) -> tuple[list[dict], float]:
    """Reads a channel's text. Gives its events, each {"row", "rows", "note", "inst", "vel"}, and the rows it fills.

    `drums` is what each drum letter plays: {letter: {"inst", "note", "rows", "vel"}}, the last two optional.
    """
    events = []
    row = 0.0
    octave = 4
    default = 16
    vel = 1.0
    tie = False
    at = 0
    while at < len(text):
        found = _TOKEN.match(text, at)
        if not found:
            raise _unread(text, at)
        if found["inst"]:
            inst = found["inst"]
        elif found["name"]:
            length = int(found["length"] or default)
            if not length:
                raise _unread(text, at)
            rows = _rows(length, len(found["dots"]), rows_per_beat)
            if found["name"] != "r":
                note = _number(octave, found["name"], found["sign"])
                # A tie makes the note before longer, when this is the same note and starts where that one ends.
                last = events[-1] if events else None
                if tie and last and last["note"] == note and last["row"] + last["rows"] == row:
                    last["rows"] += rows
                else:
                    events.append({"row": row, "rows": rows, "note": note, "inst": inst, "vel": vel})
            tie = False
            row += rows
        elif found["drum"]:
            hit = drums.get(found["drum"]) if drums else None
            if not hit:
                raise RowSynthError(f'"{found["drum"]}" is a drum, and this is not a drum channel')
            length = int(found["drum_length"] or default)
            if not length:
                raise _unread(text, at)
            rows = _rows(length, len(found["drum_dots"]), rows_per_beat)
            events.append({"row": row, "rows": min(rows, hit.get("rows", rows)), "note": midi(hit["note"]),
                           "inst": hit["inst"], "vel": vel * hit.get("vel", 1)})
            row += rows
        elif found["default"]:
            default = int(found["default"])
            if not default:
                raise _unread(text, at)
        elif found["octave"]:
            octave = int(found["octave"])
        elif found["vol"]:
            vel = int(found["vol"]) / 15
        elif found["mark"] == ">":
            octave += 1
        elif found["mark"] == "<":
            octave -= 1
        elif found["mark"] == "&":
            tie = True
        at = found.end()
    return events, row


# The twelve notes of an octave from c up, as moved text spells them.
_NAMES = ("c", "d-", "d", "e-", "e", "f", "g-", "g", "a-", "a", "b-", "b")


def transpose(text: str, semis: int) -> str:
    """Rewrites a channel's text `semis` semitones higher, spelling each note with its octave in front."""
    out = []
    octave = 4
    at = 0
    while at < len(text):
        found = _TOKEN.match(text, at)
        if not found or found["drum"]:
            raise _unread(text, at)
        if found["name"] and found["name"] != "r":
            note = _number(octave, found["name"], found["sign"]) + semis
            out.append(f"o{note // 12 - 1} {_NAMES[note % 12]}{found['length']}{found['dots']}")
        elif found["octave"]:
            octave = int(found["octave"])
        elif found["mark"] == ">":
            octave += 1
        elif found["mark"] == "<":
            octave -= 1
        elif not found[0].isspace():
            out.append(found[0])
        at = found.end()
    return " ".join(out)


def up(song: dict, semis: int) -> dict:
    """A copy of a song with its tuned channels `semis` semitones higher. Drum channels keep their text."""
    channels = song["channels"]
    patterns = {}
    for name, pattern in song["patterns"].items():
        moved = {}
        for channel, text in pattern.items():
            if channel not in channels:
                raise RowSynthError(f"pattern {name!r}: no channel called {channel!r}")
            if channels[channel].get("drums"):
                moved[channel] = text
                continue
            try:
                moved[channel] = transpose(text, semis)
            except RowSynthError as error:
                # A song has many texts: say which one could not be read.
                raise RowSynthError(f"pattern {name!r}, channel {channel!r}: {error}") from None
        patterns[name] = moved
    return {**song, "patterns": patterns}
