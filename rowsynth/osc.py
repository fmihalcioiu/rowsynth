"""The oscillators: each call gives the next sample of one wave."""
from __future__ import annotations

import math
from numbers import Real

import numpy as np

from ._jit import kernel
from .errors import RowSynthError

# Waves by number, for the loops over samples, which cannot take a name.
PULSE, SAW, TRI, NOISE, FM = 0, 1, 2, 3, 4
WAVES = {"pulse": PULSE, "saw": SAW, "tri": TRI, "noise": NOISE, "fm": FM}

# The share of each period a pulse stays high, by duty number.
DUTY = (0.125, 0.25, 0.5, 0.75)


def new_state() -> np.ndarray:
    """The memory of an oscillator that has not played yet: float64 [phase, register, accumulator, output].

    It starts [0, 1, 0, 1]. The last three are the noise's: its shift register sits in its cell as a whole number.
    """
    return np.array([0.0, 1.0, 0.0, 1.0])


@kernel
def _k_blep(t: float, dt: float) -> float:
    """What smooths a jump at phase 0 over the sample on each side of it. dt is the phase one sample covers."""
    if t < dt:
        x = t / dt
        return x + x - x * x - 1.0
    if t > 1.0 - dt:
        x = (t - 1.0) / dt
        return x * x + x + x + 1.0
    return 0.0


@kernel
def _k_pulse(state: np.ndarray, freq: float, rate: float, duty: float) -> float:
    """The next sample of a pulse that is high for the share duty of each period, both edges smoothed."""
    dt = freq / rate
    t = float(state[0])
    v = 1.0 if t < duty else -1.0
    v += _k_blep(t, dt)
    v -= _k_blep((t - duty + 1.0) % 1.0, dt)
    state[0] = (t + dt) % 1.0
    # High and low last unequal times, so the wave sits off centre: move it back to rest on zero.
    return v - (2.0 * duty - 1.0)


@kernel
def _k_saw(state: np.ndarray, freq: float, rate: float) -> float:
    """The next sample of a sawtooth: a straight climb through each period, the drop back smoothed."""
    dt = freq / rate
    t = float(state[0])
    v = 2.0 * t - 1.0 - _k_blep(t, dt)
    state[0] = (t + dt) % 1.0
    return v


@kernel
def _k_tri(state: np.ndarray, freq: float, rate: float) -> float:
    """The next sample of the stepped triangle: 32 steps a period over 16 levels, down then up, not smoothed."""
    t = float(state[0])
    step = int(t * 32.0)
    level = 15 - step if step < 16 else step - 16
    state[0] = (t + freq / rate) % 1.0
    return level / 7.5 - 1.0


@kernel
def _k_noise(state: np.ndarray, freq: float, rate: float, short: int) -> float:
    """The next sample of noise: the low bit of a 15-bit shift register that is clocked freq times a second.

    Each clock shifts the register down a bit and puts bit 0 xor bit 1 in at the top. With short at 1 it is
    bit 0 xor bit 6, and the register comes round after 93 clocks, not 32767: a buzz with a pitch to it.
    """
    acc = float(state[2]) + freq / rate
    reg = int(state[1])
    out = float(state[3])
    tap = 6 if short == 1 else 1
    while acc >= 1.0:
        acc -= 1.0
        bit = (reg ^ (reg >> tap)) & 1
        reg = (reg >> 1) | (bit << 14)
        out = -1.0 if (reg & 1) == 1 else 1.0
    state[1] = float(reg)
    state[2] = acc
    state[3] = out
    return out


@kernel
def _k_tone(out: np.ndarray, state: np.ndarray, wave: int, freq: float, rate: float, duty: float, short: int) -> None:
    """Fills out with one wave held at one frequency."""
    for i in range(len(out)):
        if wave == PULSE:
            out[i] = _k_pulse(state, freq, rate, duty)
        elif wave == SAW:
            out[i] = _k_saw(state, freq, rate)
        elif wave == TRI:
            out[i] = _k_tri(state, freq, rate)
        elif wave == NOISE:
            out[i] = _k_noise(state, freq, rate, short)


def _is_number(value: object) -> bool:
    """True for an int or a float that is finite. True and False do not count."""
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def tone(wave: str, freq: float, seconds: float, rate: int = 44100, duty: int = 2, short: bool = False) -> np.ndarray:
    """The plain voice with nothing on it: one wave at one frequency, as float32 samples.

    wave is "pulse", "saw", "tri" or "noise"; duty picks the pulse's width from DUTY.
    For noise, freq is how often the register is clocked, and short picks the short register.
    """
    if wave not in ("pulse", "saw", "tri", "noise"):
        raise RowSynthError(f"tone: cannot play the wave {wave!r}")
    if not _is_number(freq) or freq < 0:
        raise RowSynthError(f"tone: freq must be a number, 0 or more, not {freq!r}")
    if not _is_number(seconds) or seconds < 0:
        raise RowSynthError(f"tone: seconds must be a number, 0 or more, not {seconds!r}")
    if not _is_number(rate) or rate != int(rate) or rate <= 0:
        raise RowSynthError(f"tone: rate must be a whole number above 0, not {rate!r}")
    if not _is_number(duty) or duty not in range(len(DUTY)):
        raise RowSynthError(f"tone: duty must be a whole number from 0 to 3, not {duty!r}")
    out = np.zeros(math.floor(seconds * rate + 0.5), dtype=np.float32)
    _k_tone(out, new_state(), WAVES[wave], float(freq), float(rate), DUTY[int(duty)], 1 if short else 0)
    return out
