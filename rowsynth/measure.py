"""Measuring a buffer of samples: the figures a sound is checked by."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from .errors import RowSynthError


def _row(samples) -> np.ndarray:
    """The samples as one row of doubles. Any other shape, or no samples at all, is refused."""
    row = np.asarray(samples, dtype=np.float64)
    if row.ndim != 1:
        raise RowSynthError(f"samples must be one row of numbers, not an array of shape {row.shape}")
    if row.size == 0:
        raise RowSynthError("samples are empty: there is nothing to measure")
    return row


def _fixed(value: float, places: int) -> float:
    """`value` to a number of decimal places. An exact half goes away from zero, and a zero has no minus sign."""
    if not math.isfinite(value):
        return value
    # As a fraction the value is exact, so one that only looks like a half (1.0005 is a hair under) stays down.
    scale = 10 ** places
    whole = math.floor(abs(Fraction(value)) * scale + Fraction(1, 2))
    return (whole if value > 0 else -whole) / scale


def _db(level: float) -> float:
    """A level in decibels against full scale. No level at all is minus infinity."""
    return 20 * math.log10(level) if level > 0 else float("-inf")


def measure(samples, rate: int = 44100) -> dict:
    """The plain figures of a buffer: how long, how loud, how far off centre, and how it starts and ends.

    `seconds` is the length. `peak_db` and `rms_db` are decibels against full scale, minus infinity
    for silence. `dc` is the mean sample. `clipped` counts the samples at 0.999 of full scale or
    beyond. `head_step` and `tail_step` are the sizes of the first and the last sample: the jump out
    of silence and back into it, which clicks when it is large.
    """
    row = _row(samples)
    level = np.abs(row)
    # A running total adds the samples in order, one at a time. np.sum pairs them up, and the last digit can differ.
    total = float(np.cumsum(row)[-1])
    squares = float(np.cumsum(row * row)[-1])
    return {
        "seconds": _fixed(len(row) / rate, 3),
        "peak_db": _fixed(_db(float(level.max())), 1),
        "rms_db": _fixed(_db(math.sqrt(squares / len(row))), 1),
        "dc": _fixed(total / len(row), 4),
        "clipped": int(np.count_nonzero(level >= 0.999)),
        "head_step": _fixed(float(level[0]), 3),
        "tail_step": _fixed(float(level[-1]), 3),
    }
