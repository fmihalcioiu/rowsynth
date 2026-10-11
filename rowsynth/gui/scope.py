"""A scope and a spectrum: a few cycles of a wave standing still, and where the energy of a sound sits, low to high."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from ..checks import sounds
from ..errors import RowSynthError
from ..measure import brightness, pitch_at, spectrum
from ..note import DEFAULT_RATE
from ..notes import hz, midi
from . import theme
from .font import BODY_H, Font
from .layout import Rect
from .roll import mix, name
from .soundview import columns, ticks_to_seconds
from .theme import Colour
from .ui import NOWHERE, Event, Pos, Ui, Widget, inside
from .wave import joined

if TYPE_CHECKING:
    from .canvas import Canvas

#: How many cycles of the wave a scope shows.
CYCLES = 3
#: How far into a sound a scope looks, as a share of its length: past the attack, where the wave has settled.
INTO = 0.1
#: How many seconds a scope shows of a sound that has no note to count cycles by.
STRETCH = 0.01
#: How many seconds of a sound a note is read from. The note counts when over that stretch the wave is this like
#: itself one period on, from 0 to 1: a held note is all but 1 there and noise under a half, and
#: `measure.pitch_at` reads a low note out of noise as well.
HEARD = 0.04
ALIKE = 0.75
#: How far a wave at full scale keeps from the outline of a scope.
AIR = 2

#: The frequency at the left edge of a spectrum, in Hz. At its right edge is half the rate.
LOW_HZ = 50.0
#: The level at the foot of a spectrum, in decibels under its loudest bar.
FLOOR_DB = -72.0
#: The samples in each frame a spectrum is read from: those the checks read the brightness from.
FRAME = sounds._FRAME
#: From where `measure.brightness` counts a sound as bright, in Hz, and the energy under which it leaves a frame
#: out. It has no names for the two.
HARSH_HZ = 4000.0
QUIET = 1e-7
#: How far the ground right of the harsh line goes towards `theme.FAULT`.
TINT = 0.14
#: The marks along the bottom of a spectrum, each as its frequency in Hz and what it is called.
MARKS = ((100.0, "100"), (1000.0, "1k"), (10000.0, "10k"))
#: How high a line of text is here: the body of its letters, and a pixel over it and under it.
TEXT_H = BODY_H + 2
#: How far the label of the harsh line, and the words under it, stand from the line.
BESIDE = 3
#: The label of the harsh line, and what stands in a scope or a spectrum that has no sound to show.
HARSH = "HARSH"
NO_SOUND = "no sound"


def _px(value: float) -> int:
    """The nearest whole number, a half going up."""
    return math.floor(value + 0.5)


def _note(freq: float) -> float:
    """The note number of a frequency in Hz, which need not be whole: the opposite of `notes.hz`."""
    return 69 + 12 * math.log2(freq / 440.0)


def _checked(samples: np.ndarray, rate: int) -> np.ndarray:
    """The samples as they are. What is not one row of them is refused, and so is a rate that is none."""
    columns(samples, 0)
    ticks_to_seconds(0, rate)
    return samples


def heard(samples: np.ndarray, rate: int) -> float | None:
    """The note of a sound that came with none, read `INTO` of the way into it: a note number, not always whole.

    None when `measure.pitch_at` reads no note there, and when the wave is less than `ALIKE` like itself one
    period on, as noise is.
    """
    row = np.asarray(_checked(samples, rate), dtype=np.float64)
    place = int(INTO * len(row))
    found = pitch_at(row, place / rate, window=HEARD, rate=rate) if len(row) else 0.0
    if not found:
        return None
    lag = _px(rate / found)
    count = min(int(HEARD * rate), len(row) - place - lag)
    here, there = row[place:place + count], row[place + lag:place + lag + count]
    size = math.sqrt(float(np.dot(here, here)) * float(np.dot(there, there)))
    if not size or float(np.dot(here, there)) / size < ALIKE:
        return None
    return _note(found)


def _cut(row: np.ndarray, rate: int, pitch: float | None) -> tuple[int, int]:
    """The stretch a scope shows of a row whose note is known: `pitch`, or None for a sound that has none."""
    count = len(row)
    # Whole cycles are cut short of their last sample and never past it, where the next cycle would begin to show.
    length = max(2, _px(STRETCH * rate) if pitch is None else math.floor(CYCLES * rate / hz(pitch)))
    if length >= count:
        return 0, count
    first = min(int(INTO * count), count - length)
    if pitch is not None:
        # The wave rises through zero where a sample under it is followed by one that is not.
        rising = np.flatnonzero((row[first:count - length] < 0) & (row[first + 1:count - length + 1] >= 0))
        if len(rising):
            first += int(rising[0]) + 1
    return first, first + length


def span(samples: np.ndarray, rate: int, pitch: float | str | None = None) -> tuple[int, int]:
    """The stretch of a sound that a scope shows, as a slice counts it: its first sample, and the one after its last.

    With the pitch of its note, as a number or a name, that is `CYCLES` cycles from the first place after
    `INTO` of the sound where the wave rises through zero, so that two renders of nearly the same sound show
    nearly the same picture. With none the note is read from the sound, as `heard` reads it. A sound with no note
    is shown for `STRETCH` seconds from `INTO` of the way in, as it is. A sound shorter than its stretch is shown
    whole, and one that ends too soon after `INTO` is shown from earlier.
    """
    row = _checked(samples, rate)
    return _cut(row, rate, heard(row, rate) if pitch is None else midi(pitch))


def cycles_told(first: int, last: int, rate: int, pitch: float | None) -> str:
    """The stretch a scope shows in plain words: `3 cycles of a4, 6.8 ms`, or `no note, 10.0 ms` with no pitch."""
    long = f"{(last - first) * 1000 / rate:.1f} ms"
    if pitch is None:
        return f"no note, {long}"
    count = f"{round((last - first) * hz(pitch) / rate, 1):g}"
    return f"{count} cycle{'' if count == '1' else 's'} of {name(_px(pitch))}, {long}"


def _octaves(rate: float) -> float:
    """How many octaves a spectrum spans at a rate, from `LOW_HZ` to half the rate. A rate with none is refused."""
    if rate / 2 <= LOW_HZ:
        raise RowSynthError(f"rate: a spectrum starts at {LOW_HZ:g} Hz, and at {rate:g} Hz there is no sound above "
                            f"that")
    return math.log2(rate / 2 / LOW_HZ)


def column(freq: float, width: int, rate: int) -> int:
    """The column of a spectrum `width` pixels wide that a frequency falls in at a rate, the leftmost being 0.

    The scale is even in octaves: `LOW_HZ` is at the left edge, and half the rate at the right.
    """
    return math.floor(width * math.log2(freq / LOW_HZ) / _octaves(rate))


def hz_at(place: float, width: int, rate: int) -> float:
    """The frequency at the left edge of a column of a spectrum: the opposite of `column`."""
    return LOW_HZ * 2 ** (place * _octaves(rate) / max(width, 1))


def summed(samples: np.ndarray, rate: int) -> tuple[np.ndarray, float]:
    """The spectrum of a whole sound, as `measure.spectrum` gives that of a frame: magnitudes, and a bin's width in Hz.

    The sound is framed as `measure.brightness` frames it: frames of `FRAME` samples half a frame apart, each
    counting by its energy, and one with next to none not at all. The magnitudes are the root of what the
    frames add up to, so only how they stand to each other says anything. A sound shorter than a frame is one
    frame with silence after it, which `brightness` does not read. No samples and silence give zeros.
    """
    row = np.asarray(_checked(samples, rate), dtype=np.float64)
    energy = np.zeros(FRAME // 2)
    if len(row):
        for at in range(0, len(row) - FRAME + 1, FRAME // 2) or (0,):
            power = spectrum(row, at / rate, FRAME, rate)[0] ** 2
            # Bin 0 is the offset and not a pitch: it is left out of what a frame holds, and is on no column.
            if float(power[1:].sum()) >= QUIET:
                energy += power
    return np.sqrt(energy), rate / FRAME


def levels(mags: np.ndarray, per_bin: float, width: int) -> list[float]:
    """A spectrum as one level for each of `width` columns, in decibels under the loudest of them.

    A column has the loudest bin whose frequency falls in it. Low down, where the bins lie further apart than
    the columns, a column with no bin is on the line between its neighbours that have one, and before the first
    of those it is as that one. No level is under `FLOOR_DB`, and with no sound at all every one is there.
    """
    mags = np.asarray(mags, dtype=np.float64)
    freqs = np.arange(len(mags)) * per_bin
    on = freqs >= LOW_HZ
    if width <= 0 or not mags[on].any():
        return [FLOOR_DB] * max(width, 0)
    # Half the rate is where the bins end: there are as many of them as fit under it.
    at = np.floor(width * np.log2(freqs[on] / LOW_HZ) / _octaves(2 * per_bin * len(mags))).astype(int)
    loudest, has = np.zeros(width), np.zeros(width, dtype=bool)
    np.maximum.at(loudest, at, mags[on])
    has[at] = True
    least = 10 ** (FLOOR_DB / 20)
    found = 20 * np.log10(np.maximum(loudest[has] / loudest.max(), least))
    return np.interp(np.arange(width), np.flatnonzero(has), found).tolist()


def heights(mags: np.ndarray, per_bin: float, width: int, rows: int) -> list[int]:
    """A spectrum as the height of a bar for each of `width` columns, in pixels: `rows` for the loudest.

    The heights are even in decibels, as `levels` gives them: a bar at `FLOOR_DB` or under has no height.
    """
    return [_px(rows * (1 - level / FLOOR_DB)) for level in levels(mags, per_bin, width)]


def _over(canvas: Canvas, text: str, rect: Rect, colour: Colour, align: str) -> None:
    """Draw a line of text that is read over the bars as well: each of its pixels has a dark one on every side."""
    x, y, w, h = rect
    for dx, dy in ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)):
        canvas.text_in(text, (x + dx, y + dy, w, h), theme.BACKGROUND, align)
    canvas.text_in(text, rect, colour, align)


def place_told(freq: float, level: float) -> str:
    """A place of a spectrum in plain words, its frequency, the nearest note and its level: `880 Hz, a5, -12 dB`."""
    loud = f"{_px(level)} dB" if level > FLOOR_DB else f"under {FLOOR_DB:g} dB"
    return f"{_px(freq)} Hz, {name(_px(_note(freq)))}, {loud}"


class Scope(Widget):
    """A few cycles of a wave, standing still: the shape a setting gave it.

    It renders nothing. Its owner hands it one row of samples with `show`, and again whenever the sound changed.
    The wave is drawn in `colour` on a faint line at zero, between faint lines at full scale. `help` says how
    long the stretch is and of what note, and with no sound to show it is the line the scope was made with.
    """

    def __init__(self, rect: Rect, colour: Colour = theme.ACTIVE, help: str = "") -> None:
        super().__init__(rect, help)
        self.colour = colour
        self._shown = np.zeros(0, dtype=np.float32)
        self._said = ""

    @property
    def help(self) -> str:
        """The stretch that is shown in plain words, or the line the scope was made with."""
        return self._said or self._help

    @help.setter
    def help(self, line: str) -> None:
        self._help = line

    def show(self, samples: np.ndarray, rate: int, pitch: float | str | None = None) -> None:
        """Show the stretch that `span` picks of some samples at a rate. `pitch` is the note they play, if known.

        No samples and silence are no sound. What is not one row of samples, no rate or no note is refused,
        and then nothing changes.
        """
        row = _checked(samples, rate)
        given = None if pitch is None else midi(pitch)
        if not row.any():
            self._shown, self._said = row[:0], ""
            return
        note = heard(row, rate) if given is None else given
        first, last = _cut(row, rate, note)
        self._shown, self._said = row[first:last].copy(), cycles_told(first, last, rate, note)

    def draw(self, canvas: Canvas) -> None:
        """Draw the box, and in it the lines at zero and at full scale under the wave, a column for each pixel.

        The columns are made to meet, so the wave is one line. With no sound there is the box and a dim word.
        """
        x, y, w, h = self.rect
        zero = y + h // 2
        reach = max(0, (h - 1) // 2 - 1 - AIR)
        with canvas.clip(self.rect):
            canvas.box(self.rect, theme.BACKGROUND)
            if len(self._shown):
                for line, colour in ((zero - reach, theme.RAISED), (zero + reach, theme.RAISED), (zero, theme.LINE)):
                    canvas.hline(x, line, w, colour)
                spans = np.clip(columns(self._shown, max(0, w - 2)), -1.0, 1.0).tolist()
                ends = joined([(zero - _px(highest * reach), zero - _px(lowest * reach)) for lowest, highest in spans])
                for at, (upper, lower) in enumerate(ends):
                    canvas.vline(x + 1 + at, upper, lower - upper + 1, self.colour)
            else:
                canvas.text_in(NO_SOUND, self.rect, theme.DIM, "centre")
            canvas.frame(self.rect, theme.LINE)


class Spectrum(Widget):
    """Where the energy of a sound sits from low to high, as upright bars, and where it begins to be harsh.

    It renders nothing. Its owner hands it one row of samples with `show`, and again whenever the sound changed.
    The scale is even in octaves from `LOW_HZ` to half the rate, and even in decibels down to `FLOOR_DB` under
    the loudest bar. An upright line stands at `HARSH_HZ`, from where `measure.brightness` counts a sound as
    bright, and the ground right of it is tinted. By the line is its label, and under that the share of the
    energy above the line, as `brightness` gives it: in `theme.FAULT` when it is more than the checks allow a
    sound. Under the box the marks are named. `help`, while the mouse is on the bars, says the frequency there,
    the nearest note and the level under the loudest bar. Anywhere else, and with no sound, it is the line the
    spectrum was made with.
    """

    def __init__(self, rect: Rect, colour: Colour = theme.ACTIVE, help: str = "") -> None:
        super().__init__(rect, help)
        self.colour = colour
        self._mags = np.zeros(0)
        self._per_bin = 0.0
        self._rate = DEFAULT_RATE
        self._above = 0.0
        self._mouse: Pos = NOWHERE

    @property
    def box(self) -> Rect:
        """The box with its outline: all of the rectangle but the strip under it, in which the marks are named."""
        x, y, w, h = self.rect
        return x, y, w, max(0, h - TEXT_H)

    @property
    def bars(self) -> Rect:
        """Where the bars are: the box inside its outline."""
        x, y, w, h = self.box
        return x + 1, y + 1, max(0, w - 2), max(0, h - 2)

    @property
    def above(self) -> float:
        """The share of the energy above `HARSH_HZ`, from 0 to 1, as `measure.brightness` gives it."""
        return self._above

    @property
    def help(self) -> str:
        """The place under the mouse in plain words, or the line the spectrum was made with."""
        x, _, w, _ = self.bars
        if self._mags.any() and inside(self.bars, self._mouse):
            at = self._mouse[0] - x
            return place_told(hz_at(at + 0.5, w, self._rate), levels(self._mags, self._per_bin, w)[at])
        return self._help

    @help.setter
    def help(self, line: str) -> None:
        self._help = line

    def show(self, samples: np.ndarray, rate: int) -> None:
        """Show the spectrum of the whole of some samples at a rate, as `summed` gives it.

        No samples and silence are no sound. What is not one row of samples, no rate, or a rate too low for
        the scale is refused, and then nothing changes.
        """
        row = _checked(samples, rate)
        _octaves(rate)
        above = brightness(row, FRAME, rate)["above_4k"] if len(row) else 0.0
        self._mags, self._per_bin = summed(row, rate)
        self._rate, self._above = rate, above

    def on(self, event: Event, ui: Ui) -> bool:
        """Note where the mouse is, and use nothing."""
        if event.kind != "key":
            self._mouse = event.pos
        return False

    def draw(self, canvas: Canvas) -> None:
        """Draw the box, and in it the tinted ground, the lines of the marks, the bars, the harsh line and its words.

        Under the box each mark is named, its name centred on its line. With no sound there is the box and a
        dim word. All of it is in the rectangle.
        """
        x, y, w, h = self.bars
        with canvas.clip(self.rect):
            canvas.box(self.box, theme.BACKGROUND)
            canvas.frame(self.box, theme.LINE)
            if not self._mags.any():
                canvas.text_in(NO_SOUND, self.bars, theme.DIM, "centre")
                return
            line = x + column(HARSH_HZ, w, self._rate)
            marks = [(x + column(freq, w, self._rate), text) for freq, text in MARKS if freq < self._rate / 2]
            words = f"{_px(self._above * 100)}% above"
            wide = max(Font.width(HARSH), Font.width(words))
            # The words stand right of the line, and left of it where they have no room there.
            fits = line + BESIDE + wide < x + w
            left, side = (line + BESIDE, "left") if fits else (line - BESIDE - wide + 1, "right")
            with canvas.clip(self.bars):
                canvas.box((line + 1, y, max(0, x + w - line - 1), h), mix(theme.BACKGROUND, theme.FAULT, TINT))
                for at, _ in marks:
                    canvas.vline(at, y, h, theme.RAISED)
                for at, high in enumerate(heights(self._mags, self._per_bin, w, h)):
                    canvas.vline(x + at, y + h - high, high, self.colour)
                canvas.vline(line, y, h, theme.FAULT)
                _over(canvas, HARSH, (left, y + 1, wide, TEXT_H), theme.DIM, side)
                _over(canvas, words, (left, y + 1 + TEXT_H, wide, TEXT_H),
                      theme.FAULT if self._above > sounds._BRIGHT else theme.TEXT, side)
            for at, text in marks:
                wide = Font.width(text)
                canvas.text_in(text, (at - wide // 2, y + h + 1, wide, TEXT_H), theme.DIM)
