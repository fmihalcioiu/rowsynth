"""One note of an instrument, its settings stepped sixty times a second, added to a buffer of samples."""
from __future__ import annotations

import math
from numbers import Real

import numpy as np

from .errors import RowSynthError
from .notes import hz
from .osc import DUTY, PULSE, SAW, TRI, WAVES, _k_noise, _k_pulse, _k_saw, _k_tri, new_state

# A tick is a sixtieth of a second at any sample rate. An instrument's step tables move on once a tick.
TICK_HZ = 60
DEFAULT_RATE = 44100

# The waves a note can be played with.
_WAVES = ("pulse", "saw", "tri", "noise")

# The share of the way to its target that the level covers in one sample at 44100 Hz.
_SLEW = 0.012


def step(seq, tick: int, fallback: float) -> float:
    """The value of a step table at a tick.

    A number is itself at every tick. A list gives one value a tick, and past its end its last value holds.
    {"v": list, "loop": n} goes round again from index n past its end. None or an empty list gives the fallback.
    """
    values, loop = seq, -1
    if isinstance(seq, dict):
        values, loop = seq.get("v", ()), seq.get("loop", -1)
    elif seq is None:
        values = ()
    elif not isinstance(seq, (list, tuple)):
        values = (seq,)
    if not isinstance(values, (list, tuple)) or isinstance(loop, bool) or not isinstance(loop, int):
        raise RowSynthError(f"not a step table: {seq!r}")
    size = len(values)
    if size == 0:
        return float(fallback)
    if tick < size:
        value = values[tick]
    elif loop < 0 or loop >= size:
        value = values[-1]
    else:
        value = values[loop + (tick - size) % (size - loop)]
    if not _is_number(value):
        raise RowSynthError(f"not a step table: {seq!r}")
    return float(value)


def _k_note(out, edges, level, slew, state, wave, rate, hertz, short):
    """Adds a note to out: tick t fills the samples from edges[t] up to edges[t + 1], as far as out reaches.

    level is the loudness each tick heads for. Every sample the loudness covers the share slew of what is left
    to go. state, wave, rate, hertz and short are the oscillator's.
    """
    size = out.shape[0]
    last = 0.0
    for tick in range(level.shape[0]):
        target = float(level[tick])
        for i in range(edges[tick], min(edges[tick + 1], size)):
            # A loudness that jumped would click.
            last += (target - last) * slew
            if wave == PULSE:
                v = _k_pulse(state, hertz, rate, DUTY[2])
            elif wave == SAW:
                v = _k_saw(state, hertz, rate)
            elif wave == TRI:
                v = _k_tri(state, hertz, rate)
            else:
                v = _k_noise(state, hertz, rate, short)
            out[i] = float(out[i]) + v * last


def _is_number(value: object) -> bool:
    """True for an int or a float that is finite. True and False do not count."""
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _number(value: object, where: str) -> None:
    """Refuses what is not a number."""
    if not _is_number(value):
        raise RowSynthError(f"{where}: must be a number, not {value!r}")


def _length(value: object, where: str) -> None:
    """Refuses a length of time that is not a number, or is less than none."""
    _number(value, where)
    if value < 0:
        raise RowSynthError(f"{where}: must be 0 or more, not {value!r}")


def _count(value: object, where: str, least: int) -> None:
    """Refuses what is not a whole number of at least `least`."""
    _number(value, where)
    if value != int(value) or value < least:
        raise RowSynthError(f"{where}: must be a whole number, {least} or more, not {value!r}")


def _flag(value: object, where: str) -> None:
    """Refuses what is neither True nor False."""
    if not isinstance(value, bool):
        raise RowSynthError(f"{where}: must be true or false, not {value!r}")


def _wave(value: object, where: str) -> None:
    """Refuses the name of a wave there is no playing."""
    if not isinstance(value, str) or value not in _WAVES:
        raise RowSynthError(f"{where}: unknown wave {value!r}")


def _steps(table: object, where: str) -> None:
    """Refuses what is not a step table, naming the part of it that is wrong."""
    if isinstance(table, dict):
        for key in table:
            if key not in ("v", "loop"):
                raise RowSynthError(f"{where}: unknown key {key!r}")
        loop = table.get("loop", -1)
        if isinstance(loop, bool) or not isinstance(loop, int):
            raise RowSynthError(f"{where}.loop: must be a whole number, not {loop!r}")
        table, where = table.get("v"), f"{where}.v"
        if not isinstance(table, (list, tuple)):
            raise RowSynthError(f"{where}: must be a list, not {table!r}")
    if isinstance(table, (list, tuple)):
        for i, value in enumerate(table):
            _number(value, f"{where}[{i}]")
    elif not _is_number(table):
        raise RowSynthError(f"{where}: must be a number, a list or {{'v': list, 'loop': index}}, not {table!r}")


def _unread(value: object, where: str) -> None:
    """Lets through a key that is not read here."""


# What each key of an instrument must be. "gate" says how long a note is held, which whoever asks for the
# note has settled already.
_KEYS = {
    "wave": _wave,
    "vol": _steps,
    "release": _length,
    "gain": _number,
    "short": _flag,
    "gate": _unread,
}


def _instrument(inst: object) -> None:
    """Refuses an instrument that has a key it should not have, or a value of the wrong kind."""
    if not isinstance(inst, dict):
        raise RowSynthError(f"instrument: must be a dict, not {inst!r}")
    for key, value in inst.items():
        if key not in _KEYS:
            raise RowSynthError(f"instrument: unknown key {key!r}")
        _KEYS[key](value, key)


def _noise_rate(note: float) -> float:
    """How often the noise's register is clocked for a note, in hertz: 4000 at a4, doubling every six semitones."""
    return 4000.0 * 2.0 ** ((note - 69) / 6)


def render_note(out: np.ndarray, at: int, inst: dict, note: float, ticks: int,
                velocity: float = 1.0, rate: int = DEFAULT_RATE) -> int:
    """Adds one note of an instrument to out, starting at sample `at`, and returns the sample after its release.

    out is a float32 array. What is in it stays, and whatever of the note falls past its end is dropped.
    note is a number (69 is a4), ticks is how long it is held, and velocity scales its level.
    """
    if not isinstance(out, np.ndarray) or out.dtype != np.float32 or out.ndim != 1:
        raise RowSynthError("out: must be a float32 array of one dimension")
    _count(at, "at", 0)
    _instrument(inst)
    _number(note, "note")
    _length(ticks, "ticks")
    _number(velocity, "velocity")
    _count(rate, "rate", 1)
    at = int(at)
    wave = inst.get("wave", "pulse")
    release = inst.get("release", 2)
    total = ticks + release
    count = math.ceil(total)
    tick_samples = rate / TICK_HZ
    gain = inst.get("gain", 1) * velocity
    freq = _noise_rate(note) if wave == "noise" else hz(note)
    level = np.zeros(count)
    for tick in range(count):
        vol = step(inst.get("vol"), tick, 15) / 15
        if tick >= ticks:
            vol *= 1 - (tick - ticks + 1) / (release + 1)
        level[tick] = vol * gain
    edges = np.floor(at + np.arange(count + 1) * tick_samples + 0.5).astype(np.int64)
    # The level takes the same length of time to arrive at any rate.
    slew = 1 - (1 - _SLEW) ** (DEFAULT_RATE / rate)
    short = 1 if inst.get("short", False) else 0
    _k_note(out, edges, level, slew, new_state(), WAVES[wave], float(rate), float(freq), short)
    return at + math.floor(total * tick_samples + 0.5)
