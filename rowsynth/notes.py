"""Note names as numbers, and note numbers as frequencies."""
from __future__ import annotations

import re
from numbers import Real

from .errors import RowSynthError

# Semitones above C for each letter, and what a sharp or a flat adds.
_LETTER = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
_MARK = {"": 0, "#": 1, "b": -1}
_NAME = re.compile(r"([a-g])([#b]?)(-?[0-9])")


def midi(name: str | float) -> float:
    """The number of a note: 'a4' is 69.0, 'c#5' is 73.0, 'eb3' is 51.0, middle C is 'c4'. A number passes through."""
    if isinstance(name, Real) and not isinstance(name, bool):
        return float(name)
    found = _NAME.fullmatch(name.lower()) if isinstance(name, str) else None
    if found is None:
        raise RowSynthError(f"not a note: {name!r}")
    letter, mark, octave = found.groups()
    return float(12 * (int(octave) + 1) + _LETTER[letter] + _MARK[mark])


def hz(note: float) -> float:
    """The frequency of a note number in hertz: 69 is 440.0, and twelve more doubles it."""
    return 440.0 * 2.0 ** ((note - 69) / 12)
