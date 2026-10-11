"""Note names as numbers, and note numbers as frequencies."""
from __future__ import annotations

import re
from numbers import Real

from .errors import RowSynthError

# Semitones above C for each letter, and what a sharp or a flat adds.
_LETTER = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
_MARK = {"": 0, "#": 1, "b": -1}
_NAME = re.compile(r"([a-g])([#b]?)(-?[0-9])")

# The lowest note and the highest that the package plays. Going down nothing gives out, the wave only gets
# slower, so the lowest is the lowest that a name can say: the c flat of octave -9. Going up, a pitch above half
# the samples a second cannot be told from a lower one. At the usual 44100 samples a second the last whole note
# under that half is 136, which is 21096 Hz, and 137 is 22351 Hz. The noise's register is clocked 208 times a
# sample there, and twice as often every six semitones further up: with no highest note it would never be done.
LOWEST = -97
HIGHEST = 136


def midi(name: str | float) -> float:
    """The number of a note: 'a4' is 69.0, 'c#5' is 73.0, 'eb3' is 51.0, middle C is 'c4'. A number passes through."""
    if isinstance(name, Real) and not isinstance(name, bool):
        return float(name)
    found = _NAME.fullmatch(name.lower()) if isinstance(name, str) else None
    if found is None:
        raise RowSynthError(f"not a note: {name!r}")
    letter, mark, octave = found.groups()
    return float(12 * (int(octave) + 1) + _LETTER[letter] + _MARK[mark])


def in_range(note: float) -> bool:
    """True for a note number that can be played: one from LOWEST to HIGHEST. What is no number at all is not one."""
    return LOWEST <= note <= HIGHEST


def nearest(note: float) -> float:
    """The note that can be played nearest to a note number: the number itself, or the end of the range it is past."""
    # In this order what is no number at all, which is neither above nor below anything, comes out as the lowest.
    return min(max(LOWEST, note), HIGHEST)


def hz(note: float) -> float:
    """The frequency of a note number in hertz: 69 is 440.0, and twelve more doubles it."""
    return 440.0 * 2.0 ** ((note - 69) / 12)
