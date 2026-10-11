"""Measuring a buffer of samples: the figures a sound is checked by."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from ._jit import kernel
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


def _frame_size(size: int) -> int:
    """A frame size as a plain int. It has to be a power of two, so that it halves into whole bins."""
    if not isinstance(size, (int, np.integer)) or size < 2 or size & (size - 1):
        raise RowSynthError(f"a frame size must be a power of two, 2 or more, not {size!r}")
    return int(size)


def spectrum(samples, t: float = 0.0, size: int = 4096, rate: int = 44100) -> tuple[np.ndarray, float]:
    """The magnitudes of one frame of `size` samples starting at time `t`, and the width of a bin in Hz.

    A frame that would run past the end is pulled back to fit, and a buffer shorter than a frame is
    padded with silence. The frame is shaped by a Hann window and scaled so that a full-scale sine
    reads close to 1.0 in its bin. There are `size / 2` bins, from 0 Hz to just under half the rate.
    """
    row = _row(samples)
    size = _frame_size(size)
    start = max(0, min(len(row) - size, math.floor(t * rate)))
    part = row[start:start + size]
    frame = np.zeros(size)
    frame[:len(part)] = part
    frame *= 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(size) / (size - 1))
    mags = np.abs(np.fft.rfft(frame))[:size // 2] / (size / 4)
    return mags, rate / size


def brightness(samples, size: int = 2048, rate: int = 44100) -> dict:
    """Where the energy of a buffer sits: `centroid_hz`, its centre in Hz, and `above_4k`, the share above 4 kHz.

    Frames of `size` samples, half a frame apart, each count by their energy, so a loud stretch
    weighs more than a quiet one and a silent one not at all. A buffer shorter than one frame, or
    silent throughout, gives zeros.
    """
    row = _row(samples)
    size = _frame_size(size)
    # Bin 0 is the offset and not a pitch, so it is left out.
    hz = np.arange(1, size // 2) * (rate / size)
    energy = centre = high = 0.0
    for at in range(0, len(row) - size + 1, size // 2):
        # The frame is asked for by its time and starts where spectrum() puts that time: now and then a sample early.
        mags, _ = spectrum(row, at / rate, size, rate)
        power = mags[1:] ** 2
        total = float(power.sum())
        if total < 1e-7:
            continue
        energy += total
        centre += float(np.dot(power, hz))
        high += float(power[hz > 4000].sum())
    if not energy:
        return {"centroid_hz": 0, "above_4k": 0.0}
    return {"centroid_hz": math.floor(centre / energy + 0.5), "above_4k": _fixed(high / energy, 2)}


# How like itself, one period on, the first half of a stretch has to be for a note to be read there, from 0 to 1.
# A note that holds is all but 1 and white noise is under 0.2. The line lies between two that come close: a note
# with the echo of other notes under it, which is to be read, and a wave with nine tenths of its energy an
# octave up, which half a period on is 0.8 like itself and is not to be read an octave high.
_ALIKE = 0.8


@kernel
def _k_period(corr, alike, low, high) -> float:
    """One period as a lag from `low` to `high`, with a fraction, or 0.0 when the wave repeats at none of them.

    corr is the match of the whole stretch with itself at each lag. alike is how like itself the first half of
    the stretch is at each lag, from -1 to 1.
    """
    # At small lags a wave still overlaps itself, so the search waits until the match has fallen to zero or below.
    lag = 1
    while lag <= high and corr[lag] > 0.0:
        lag += 1
    # The wave is itself again two or three periods along as well, and where other notes sound under it, it is
    # most itself at the lag that fits them all. So the earliest peak that is alike enough is the period.
    for i in range(max(lag, low), high):
        if alike[i] >= _ALIKE and corr[i] >= corr[i - 1] and corr[i] >= corr[i + 1]:
            # A parabola through the peak and the lag on each side puts the top between whole lags.
            a = corr[i - 1]
            b = corr[i]
            c = corr[i + 1]
            bend = 2.0 * (a - 2.0 * b + c)
            if bend == 0.0:
                bend = 1.0
            return i + (a - c) / bend
    return 0.0


def pitch_at(samples, t: float, lo: float = 50.0, hi: float = 4000.0, window: float = 0.04, rate: int = 44100) -> float:
    """The note sounding at time `t`, in Hz to one decimal place, or 0.0 when there is none to read.

    A stretch of `window` seconds from `t` is set against itself moved along, and the shortest move that
    lines it up again is one period. The move counts when the first half of the stretch is like what lies
    that far on. So a wave that does not repeat reads as no note, as noise does, and a note is still read
    when the next one starts in the second half of the stretch. Noise so slow that the stretch holds only a
    few of its steps can still line up by chance. Only notes from `lo` to `hi` Hz are looked for. A stretch
    of fewer than 64 samples, or one with next to no sound in it, reads as no note.
    """
    row = _row(samples)
    start = max(0, math.floor(t * rate))
    count = min(math.floor(window * rate), len(row) - start)
    if count < 64:
        return 0.0
    low = math.floor(rate / hi)
    high = min(math.floor(rate / lo), count - 1)
    part = row[start:start + count]
    if float(np.dot(part, part)) < 1e-6:
        return 0.0
    # Lag by lag, the mean product of the stretch and itself moved along. Zeros stand in past its end.
    padded = np.concatenate((part, np.zeros(high)))
    lags = np.arange(high + 1)
    corr = np.correlate(padded, part, "valid") / (count - lags)
    # Lag by lag, the product of the first half and what lies that far on, over the size of the two: 1 when
    # they are one shape, whatever the level of each. Past the end of the stretch there is less of both.
    head = count // 2
    energy = np.concatenate(((0.0,), np.cumsum(part * part)))
    size = np.sqrt(energy[np.minimum(head, count - lags)] * (energy[np.minimum(lags + head, count)] - energy[lags]))
    alike = np.zeros(high + 1)
    np.divide(np.correlate(padded, part[:head], "valid")[:high + 1], size, out=alike, where=size > 0)
    period = float(_k_period(corr, alike, low, high))
    return _fixed(rate / period, 1) if period else 0.0
