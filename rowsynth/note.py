"""One note of an instrument, its settings stepped sixty times a second, added to a buffer of samples."""
from __future__ import annotations

import math
from numbers import Real

import numpy as np

from ._jit import kernel
from .errors import RowSynthError
from .fm import _k_fm, _k_gate_off, fm_voice
from .notes import hz
from .osc import DUTY, FM, NOISE, PULSE, SAW, TRI, WAVES, _k_noise, _k_pulse, _k_saw, _k_tri, new_state

# A tick is a sixtieth of a second at any sample rate. An instrument's step tables move on once a tick.
TICK_HZ = 60
DEFAULT_RATE = 44100

# The share of the way to its target that the level covers in one sample at 44100 Hz.
_SLEW = 0.012

# The highest cutoff the low-pass is given, in hertz.
_TOP_CUTOFF = 18000

# The ticks a note fades for once it is let go, when its instrument does not say.
_RELEASE = 2


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


@kernel
def _k_note(out, edges, level, slew, state, wave, rate, freq, short, duty, fm_params, fm_state, gate,
            lp, resonant, svf):
    """Adds a note to out: tick t fills the samples from edges[t] up to edges[t + 1], as far as out reaches.

    level is the loudness each tick heads for. Every sample the loudness covers the share slew of what is left
    to go. state, wave, rate and short are the oscillator's, and freq and duty its frequency and the width of
    its pulse at each tick. The FM wave has its voice in fm_params and fm_state, and lets it go as tick gate
    starts: a gate that is no tick of the note's leaves it held. lp is the low-pass at each tick: the share of
    the way to its input that each of its two stages covers in a sample, which is 1 when there is nothing to
    take off. With resonant at 1 the low-pass is a state-variable one instead, and svf holds its three
    coefficients at each tick, all 0 when there is nothing to take off.
    """
    size = out.shape[0]
    last = 0.0
    low1 = 0.0
    low2 = 0.0
    ic1 = 0.0
    ic2 = 0.0
    for tick in range(level.shape[0]):
        if wave == FM and tick == gate:
            _k_gate_off(fm_params, fm_state)
        target = float(level[tick])
        hertz = float(freq[tick])
        width = float(duty[tick])
        ease = float(lp[tick])
        a1 = float(svf[tick, 0])
        a2 = float(svf[tick, 1])
        a3 = float(svf[tick, 2])
        for i in range(edges[tick], min(edges[tick + 1], size)):
            # A loudness that jumped would click.
            last += (target - last) * slew
            if wave == PULSE:
                v = _k_pulse(state, hertz, rate, width)
            elif wave == SAW:
                v = _k_saw(state, hertz, rate)
            elif wave == TRI:
                v = _k_tri(state, hertz, rate)
            elif wave == NOISE:
                v = _k_noise(state, hertz, rate, short)
            else:
                v = _k_fm(fm_params, fm_state, hertz, rate)
            if resonant == 1:
                if a1 != 0.0:
                    # One sample through the filter: ic1 and ic2 are what its two integrators hold, v2 its low-pass.
                    v3 = v - ic2
                    v1 = a1 * ic1 + a2 * v3
                    v2 = ic2 + a2 * ic1 + a3 * v3
                    ic1 = 2.0 * v1 - ic1
                    ic2 = 2.0 * v2 - ic2
                    v = v2
            elif ease < 1.0:
                # Two poles, one feeding the other: 12 dB off for each octave above the cutoff.
                low1 += (v - low1) * ease
                low2 += (low1 - low2) * ease
                v = low2
            out[i] = float(out[i]) + v * last


def _is_number(value: object) -> bool:
    """True for an int or a float that is finite. True and False do not count."""
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _number(value: object, where: str) -> None:
    """Refuses what is not a number."""
    if not _is_number(value):
        raise RowSynthError(f"{where}: must be a number, not {value!r}")


def _length(value: object, where: str) -> None:
    """Refuses a length of time that is not a number or is below 0."""
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


def _buffer(out: object, where: str) -> None:
    """Refuses what is not a buffer of samples."""
    if not isinstance(out, np.ndarray) or out.dtype != np.float32 or out.ndim != 1:
        raise RowSynthError(f"{where}: must be a float32 array of one dimension")


def _wave(value: object, where: str) -> None:
    """Refuses what is not the name of a wave."""
    if not isinstance(value, str) or value not in WAVES:
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


def _duty(table: object, where: str) -> None:
    """Refuses a step table that asks for a pulse width there is not: they are numbered 0 to 3."""
    _steps(table, where)
    values = table.get("v") if isinstance(table, dict) else table
    for value in values if isinstance(values, (list, tuple)) else (values,):
        if value not in range(len(DUTY)):
            raise RowSynthError(f"{where}: must hold whole numbers from 0 to 3, not {value!r}")


def _vib(vib: object, where: str) -> None:
    """Refuses what is not a vibrato: {"delay": ticks, "depth": semitones, "rate": Hz}, the depth at least."""
    if not isinstance(vib, dict):
        raise RowSynthError(f"{where}: must be a dict, not {vib!r}")
    for key, value in vib.items():
        if key not in ("delay", "depth", "rate"):
            raise RowSynthError(f"{where}: unknown key {key!r}")
        _number(value, f"{where}.{key}")
    if "depth" not in vib:
        raise RowSynthError(f"{where}: missing key 'depth'")


def _unread(value: object, where: str) -> None:
    """Lets through a key that is not read here."""


# What each key of an instrument must be. "gate" says how long a note is held, which whoever asks for the
# note has settled already. "alg", "fb" and "ops" are the FM voice's, which checks them as it is built: only
# an instrument with that wave has them read.
_KEYS = {
    "wave": _wave,
    "vol": _steps,
    "duty": _duty,
    "arp": _steps,
    "pitch": _steps,
    "vib": _vib,
    "slide": _number,
    "release": _length,
    "gain": _number,
    "short": _flag,
    "lp": _steps,
    "res": _number,
    "key_scale": _number,
    "gate": _unread,
    "alg": _unread,
    "fb": _unread,
    "ops": _unread,
}


def _instrument(inst: object, where: str = "") -> None:
    """Refuses an instrument that has a key it should not have, or a value of the wrong kind.

    where is the place the instrument sits at in the data it comes from, and the error opens with it. An
    instrument that sits nowhere has its keys named on their own.
    """
    whole = where or "instrument"
    under = f"{where}." if where else ""
    if not isinstance(inst, dict):
        raise RowSynthError(f"{whole}: must be a dict, not {inst!r}")
    for key, value in inst.items():
        if key not in _KEYS:
            raise RowSynthError(f"{whole}: unknown key {key!r}")
        _KEYS[key](value, under + key)
    if inst.get("wave") == "fm":
        try:
            # Whatever is wrong with a voice is wrong with it at any rate.
            fm_voice(inst, DEFAULT_RATE)
        except RowSynthError as error:
            raise RowSynthError(f"{under}{error}") from None


def _noise_rate(note: float) -> float:
    """How often the noise's register is clocked for a note, in hertz: 4000 at a4, doubling every six semitones."""
    return 4000.0 * 2.0 ** ((note - 69) / 6)


def render_note(out: np.ndarray, at: int, inst: dict, note: float, ticks: int,
                velocity: float = 1.0, rate: int = DEFAULT_RATE) -> int:
    """Adds one note of an instrument to out, starting at sample `at`, and returns the sample after its release.

    out is a float32 array. What is in it stays, and whatever of the note falls past its end is dropped.
    note is a number (69 is a4), ticks is how long it is held, and velocity scales its level.
    """
    _buffer(out, "out")
    _count(at, "at", 0)
    _instrument(inst)
    _number(note, "note")
    _length(ticks, "ticks")
    _number(velocity, "velocity")
    _count(rate, "rate", 1)
    at = int(at)
    wave = inst.get("wave", "pulse")
    # Only the FM wave has a voice. The kernel takes one whatever the wave, so the others hand it an empty one.
    fm_params = fm_state = np.zeros((0, 0))
    if wave == "fm":
        fm_params, fm_state = fm_voice(inst, rate)
    release = inst.get("release", _RELEASE)
    total = ticks + release
    count = math.ceil(total)
    tick_samples = rate / TICK_HZ
    # Every octave above middle C adds key_scale decibels to the level, and every octave below takes them off.
    gain = inst.get("gain", 1) * velocity * 10 ** ((inst.get("key_scale", 0) * (note - 60)) / 240)
    vib = inst.get("vib")
    res = inst.get("res")
    resonant = 0 if res is None else 1
    level = np.zeros(count)
    freq = np.zeros(count)
    duty = np.zeros(count)
    lp = np.ones(count)
    svf = np.zeros((count, 3))
    sweep = 0.0
    for tick in range(count):
        vol = step(inst.get("vol"), tick, 15) / 15
        if tick >= ticks:
            vol *= 1 - (tick - ticks + 1) / (release + 1)
        level[tick] = vol * gain
        sweep += step(inst.get("pitch"), tick, 0)
        wobble = 0.0
        if vib is not None and tick >= vib.get("delay", 0):
            # A rate of 0 is taken as a rate left out.
            angle = (2 * math.pi * (vib.get("rate") or 6) * (tick - vib.get("delay", 0))) / TICK_HZ
            wobble = vib["depth"] * math.sin(angle)
        semis = note + step(inst.get("arp"), tick, 0) + sweep + wobble + inst.get("slide", 0) * (tick / TICK_HZ)
        freq[tick] = _noise_rate(semis) if wave == "noise" else hz(semis)
        duty[tick] = DUTY[int(step(inst.get("duty"), tick, 2))]
        cutoff = step(inst.get("lp"), tick, 0)
        if cutoff > 0:
            lp[tick] = 1 - math.exp((-2 * math.pi * min(cutoff, _TOP_CUTOFF)) / rate)
            if res is not None:
                # The tangent runs away as the cutoff nears half the rate, so the cutoff is kept short of that.
                g = math.tan((math.pi * min(cutoff, _TOP_CUTOFF, 0.45 * rate)) / rate)
                # Resonance thins the damping, from 2 down to 0.04 at the most: with none the filter would ring on.
                k = 2 - 2 * min(max(res, 0), 0.98)
                a1 = 1 / (1 + g * (g + k))
                a2 = g * a1
                svf[tick, 0] = a1
                svf[tick, 1] = a2
                svf[tick, 2] = g * a2
    edges = np.floor(at + np.arange(count + 1) * tick_samples + 0.5).astype(np.int64)
    # The level takes the same length of time to arrive at any rate.
    slew = 1 - (1 - _SLEW) ** (DEFAULT_RATE / rate)
    short = 1 if inst.get("short", False) else 0
    # The voice is let go as the first tick of the release starts. A note that ends part of the way through a
    # tick has no tick starting there, and its voice stays held.
    gate = int(ticks) if ticks == int(ticks) else -1
    _k_note(out, edges, level, slew, new_state(), WAVES[wave], float(rate), freq, short, duty, fm_params, fm_state,
            gate, lp, resonant, svf)
    return at + math.floor(total * tick_samples + 0.5)
