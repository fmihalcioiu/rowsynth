"""The small language a song's channels are written in."""
from __future__ import annotations

import re

from .errors import RowSynthError

# Everything a channel's text can hold, one kind to a line.
_TOKEN = re.compile("|".join((
    r"\s+",                                                           # space, which only keeps things apart
    r"\|",                                                            # a bar line, which changes nothing
    r"(?P<name>[a-gr])(?P<sign>[+#-]?)(?P<length>\d*)(?P<dots>\.*)",  # a note, or r for a rest: c f+8 b-4. r2
    r"l(?P<default>\d+)",                                             # the length for notes written without one: l8
    r"o(?P<octave>\d)",                                               # the octave: o4
    r"(?P<mark>[<>])",                                                # an octave down, an octave up
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


def parse(text: str, inst: str, rows_per_beat: int = 4) -> tuple[list[dict], float]:
    """Reads a channel's text. Gives its events, each {"row", "rows", "note", "inst", "vel"}, and the rows it fills."""
    events = []
    row = 0.0
    octave = 4
    default = 16
    at = 0
    while at < len(text):
        found = _TOKEN.match(text, at)
        if not found:
            raise _unread(text, at)
        if found["name"]:
            length = int(found["length"] or default)
            if not length:
                raise _unread(text, at)
            rows = _rows(length, len(found["dots"]), rows_per_beat)
            if found["name"] != "r":
                note = _number(octave, found["name"], found["sign"])
                events.append({"row": row, "rows": rows, "note": note, "inst": inst, "vel": 1.0})
            row += rows
        elif found["default"]:
            default = int(found["default"])
            if not default:
                raise _unread(text, at)
        elif found["octave"]:
            octave = int(found["octave"])
        elif found["mark"] == ">":
            octave += 1
        elif found["mark"] == "<":
            octave -= 1
        at = found.end()
    return events, row
