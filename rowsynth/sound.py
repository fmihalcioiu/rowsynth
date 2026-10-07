"""A sound: voices playing notes into one buffer, and the output stage that buffer leaves through."""
from __future__ import annotations

import math

import numpy as np

from .errors import RowSynthError
from .note import DEFAULT_RATE, TICK_HZ, _RELEASE, _buffer, _count, _flag, _instrument, _length, _number, render_note
from .notes import midi

# The ticks a sound's buffer runs on for after its last release is over.
_TAIL = 2

# The high-pass takes off what lies under about this many hertz.
_LOW_HZ = 30

# Up to the knee a sample passes as it is. Past it the samples are bent over, and the furthest they get is
# the knee and the room above it together: 0.99.
_KNEE = 0.7
_ROOM = 0.29

# The samples the fade at the end takes at 44100 Hz: 5 milliseconds.
_FADE = 220


def _k_output(out, gain, keep, memory):
    """Puts out through its gain, the high-pass and the knee, in place.

    keep is the share of itself the high-pass keeps from one sample to the next. memory is what it holds
    between samples: the last one that went in and the last one that came out.
    """
    last_in = memory[0]
    last_out = memory[1]
    for i in range(out.shape[0]):
        x = float(out[i]) * gain
        y = keep * (last_out + x - last_in)
        last_in = x
        last_out = y
        size = abs(y)
        if size <= _KNEE:
            out[i] = y
        else:
            out[i] = math.copysign(_KNEE + _ROOM * math.tanh((size - _KNEE) / _ROOM), y)
    memory[0] = last_in
    memory[1] = last_out


def _k_fade(out, count):
    """Fades the last count samples of out: the last one to nothing, each one before it a step less far."""
    last = out.shape[0] - 1
    for i in range(count):
        out[last - i] = float(out[last - i]) * (i / count)


def finish(out: np.ndarray, gain: float = 0.5, rate: int = DEFAULT_RATE, fade: bool = True) -> np.ndarray:
    """The output stage, worked on out in place: a gain, a high-pass, a soft knee and a fade at the end.

    out is a float32 array, and the same array comes back. The high-pass takes off what lies under about 30 Hz,
    an offset with it. The knee leaves what is under 0.7 as it is and bends what is over, so that nothing
    reaches full scale. The fade brings the last 5 milliseconds down to silence. fade=False is for a buffer
    that is played round and round: its end is left as it is, and the high-pass starts as if the buffer had
    been round once already.
    """
    _buffer(out, "out")
    _number(gain, "gain")
    _count(rate, "rate", 1)
    _flag(fade, "fade")
    keep = 1 - (2 * math.pi * _LOW_HZ) / rate
    memory = np.zeros(2)
    if not fade:
        # Played round and round, the buffer's first sample follows its last. A first run over a copy is kept
        # for nothing but what the high-pass holds at the end of it, and the true run starts from that.
        _k_output(out.copy(), float(gain), keep, memory)
    _k_output(out, float(gain), keep, memory)
    if fade:
        _k_fade(out, min(out.shape[0], math.floor((_FADE * rate) / DEFAULT_RATE + 0.5)))
    return out


def _keys(value: object, where: str, needed: tuple, optional: tuple = ()) -> None:
    """Refuses what is not a dict, a key that is neither needed nor optional, and a needed key left out."""
    if not isinstance(value, dict):
        raise RowSynthError(f"{where}: must be a dict, not {value!r}")
    for key in value:
        if key not in needed + optional:
            raise RowSynthError(f"{where}: unknown key {key!r}")
    for key in needed:
        if key not in value:
            raise RowSynthError(f"{where}: missing key {key!r}")


def _list(value: object, where: str) -> None:
    """Refuses what is not a list. A tuple does for one."""
    if not isinstance(value, (list, tuple)):
        raise RowSynthError(f"{where}: must be a list, not {value!r}")


def _read(defn: object) -> tuple[list[tuple], float]:
    """Checks a sound definition from end to end, and returns its notes and its gain.

    A note comes back as (instrument, tick, note number, ticks, velocity). Whatever is wrong is refused with
    the place it sits at: "voices[1].notes[0][2]" is the third value of the first note of the second voice.
    """
    _keys(defn, "sound", ("voices",), ("gain",))
    gain = defn.get("gain", 0.5)
    _number(gain, "gain")
    _list(defn["voices"], "voices")
    notes = []
    for i, voice in enumerate(defn["voices"]):
        where = f"voices[{i}]"
        _keys(voice, where, ("inst", "notes"))
        _instrument(voice["inst"], f"{where}.inst")
        _list(voice["notes"], f"{where}.notes")
        for j, note in enumerate(voice["notes"]):
            spot = f"{where}.notes[{j}]"
            if not isinstance(note, (list, tuple)) or len(note) not in (3, 4):
                raise RowSynthError(f"{spot}: must be [tick, note, ticks] or [tick, note, ticks, velocity], "
                                    f"not {note!r}")
            tick, pitch, ticks = note[:3]
            velocity = note[3] if len(note) == 4 else 1.0
            _length(tick, f"{spot}[0]")
            try:
                number = midi(pitch)
            except RowSynthError as error:
                raise RowSynthError(f"{spot}[1]: {error}") from None
            _number(number, f"{spot}[1]")
            _length(ticks, f"{spot}[2]")
            _number(velocity, f"{spot}[3]")
            notes.append((voice["inst"], tick, number, ticks, velocity))
    return notes, gain


def render_sound(defn: dict, rate: int = DEFAULT_RATE) -> np.ndarray:
    """A sound as samples: every note of every voice added into one buffer, and that put through `finish`.

    defn is {"voices": [{"inst": instrument, "notes": [[tick, note, ticks, velocity], ...]}, ...], "gain": level}.
    A note starts at its tick and is held for its ticks. It is a name such as "c#4" or a number, and its
    velocity can be left out. So can the gain, which is then 0.5. The buffer runs until two ticks after the
    last release is over.
    """
    notes, gain = _read(defn)
    _count(rate, "rate", 1)
    tick_samples = rate / TICK_HZ
    end = max((tick + ticks + inst.get("release", _RELEASE) for inst, tick, _, ticks, _ in notes), default=0)
    out = np.zeros(math.ceil((end + _TAIL) * tick_samples), dtype=np.float32)
    for inst, tick, note, ticks, velocity in notes:
        render_note(out, math.floor(tick * tick_samples + 0.5), inst, note, ticks, velocity, rate)
    return finish(out, gain, rate)
