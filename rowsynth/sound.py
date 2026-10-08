"""A sound: voices playing notes into one buffer, the echo it can be given, and the output stage it leaves through."""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from ._jit import kernel
from .errors import RowSynthError
from .note import (DEFAULT_RATE, TICK_HZ, _RELEASE, _buffer, _count, _flag, _instrument, _is_number, _length, _number,
                   render_note)
from .notes import midi

# The ticks a sound's buffer runs on for after its last release is over.
_TAIL = 2

# The most of what comes back that a delay line takes in again. With all of it the returns would never die away.
_TOP_FEEDBACK = 0.95

# A buffer with an echo is made long enough for every return that is less than this many powers of ten under the
# first. Three powers of ten are sixty decibels.
_HEARD = 3

# The high-pass takes off what lies under about this many hertz.
_LOW_HZ = 30

# Up to the knee a sample passes as it is. Past it the samples are bent over, and the furthest they get is
# the knee and the room above it together: 0.99.
_KNEE = 0.7
_ROOM = 0.29

# The samples the fade at the end takes at 44100 Hz: 5 milliseconds.
_FADE = 220


@kernel
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


@kernel
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


def _times(value: object, where: str) -> None:
    """Refuses what is neither a number nor a list of numbers. A tuple does for a list."""
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _number(item, f"{where}[{i}]")
    elif not _is_number(value):
        raise RowSynthError(f"{where}: must be a number or a list of numbers, not {value!r}")


# What each setting of an echo must be. Under 0 the cutoff would have the low-pass run away, so it is held to
# what a length of time is held to: 0 or more.
_ECHO = {"time": _times, "feedback": _number, "mix": _number, "lp": _length}


def _settings(settings: object, where: str = "") -> None:
    """Refuses settings of an echo that lack their time, have a key they should not have, or a value of the wrong kind.

    where is the place the settings sit at in a sound definition, and the error opens with it. Settings that
    sit nowhere have their keys named on their own.
    """
    _keys(settings, where or "echo", ("time",), tuple(_ECHO))
    for key, value in settings.items():
        _ECHO[key](value, f"{where}.{key}" if where else key)


@kernel
def _k_line(out, dry, line, delay, feedback, ease, level):
    """Adds to out what one delay line gives back of dry.

    line is the delay line, as long as out. Each of its cells takes the sample of dry at its place, where dry
    has one, and feedback of what is coming back there. What comes back at a sample is the cell delay samples
    before it, through a low-pass that covers the share ease of the way to its input every sample, and it is
    added to out at level. Every cell is written before it is read, so what line holds on the way in does not
    matter.
    """
    size = dry.shape[0]
    back = 0.0
    for i in range(out.shape[0]):
        sent = 0.0
        if i >= delay:
            sent = float(line[i - delay])
        back += (sent - back) * ease
        fresh = 0.0
        if i < size:
            fresh = float(dry[i])
        line[i] = fresh + feedback * back
        out[i] = float(out[i]) + level * back


def echo(dry: np.ndarray, time: float | list[float], feedback: float = 0.3, mix: float = 0.3, lp: float = 3000.0,
         rate: int = DEFAULT_RATE) -> np.ndarray:
    """dry with its echo added, in a new and longer buffer. dry is left as it was.

    The sound comes back `time` seconds after itself, then again that much later at `feedback` of the level it
    came back at, and so on. Every return has been through a low-pass at `lp` Hz once more than the one before
    it, so each is darker than the last. time can be a list: every time in it is a delay line of its own, and
    the lines share `mix` between them. With one line the first return is at `mix` of the sound, with two each
    line's is at half of that. A time that is not above 0 is no line, and with no line at all what comes back
    is dry as it is. feedback is held between 0 and 0.95. The buffer is longer than dry by the longest time,
    once for every return that is less than sixty decibels under the first.
    """
    _buffer(dry, "dry")
    _settings({"time": time, "feedback": feedback, "mix": mix, "lp": lp})
    _count(rate, "rate", 1)
    times = [seconds for seconds in (time if isinstance(time, (list, tuple)) else (time,)) if seconds > 0]
    if not times:
        return dry.copy()
    kept = min(max(feedback, 0), _TOP_FEEDBACK)
    returns = math.ceil(-_HEARD / math.log10(kept)) if kept > 0 else 1
    out = np.zeros(len(dry) + math.ceil(returns * max(times) * rate), dtype=np.float32)
    out[:len(dry)] = dry
    ease = 1 - math.exp((-2 * math.pi * lp) / rate)
    line = np.zeros(len(out), dtype=np.float32)
    for seconds in times:
        # A time under half a sample would be no delay at all, and a line cannot read the cell it is writing.
        delay = max(1, math.floor(seconds * rate + 0.5))
        _k_line(out, dry, line, delay, float(kept), ease, mix / len(times))
    return out


class _Sound(NamedTuple):
    """What a sound definition says, once it has been checked."""

    notes: list[tuple]
    gain: float
    echo: dict | None


def _read(defn: object) -> _Sound:
    """Checks a sound definition from end to end, and returns what it says: its notes, its gain and its echo.

    A note comes back as (instrument, tick, note number, ticks, velocity). The echo comes back as its settings,
    or as None from a sound that has none. Whatever is wrong is refused with the place it sits at:
    "voices[1].notes[0][2]" is the third value of the first note of the second voice.
    """
    _keys(defn, "sound", ("voices",), ("gain", "echo"))
    gain = defn.get("gain", 0.5)
    _number(gain, "gain")
    if "echo" in defn:
        _settings(defn["echo"], "echo")
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
    return _Sound(notes, gain, defn.get("echo"))


def render_sound(defn: dict, rate: int = DEFAULT_RATE, gain: float | None = None) -> np.ndarray:
    """A sound as samples: every note of every voice added into one buffer, and that put through `finish`.

    defn is {"voices": [{"inst": instrument, "notes": [[tick, note, ticks, velocity], ...]}, ...], "gain": level,
    "echo": {"time": seconds, "feedback": 0.3, "mix": 0.3, "lp": 3000}}.
    A note starts at its tick and is held for its ticks. It is a name such as "c#4" or a number, and its
    velocity can be left out. So can the gain, which is then 0.5. The buffer runs until two ticks after the
    last release is over. An echo is added by `echo` before the output stage, and makes the buffer longer by
    its returns. Of its settings only the time is needed, and that can be a list of times.
    A gain handed in beside the definition is the one the sound goes through the output stage at, in place of
    the definition's own.
    """
    sound = _read(defn)
    _count(rate, "rate", 1)
    if gain is None:
        gain = sound.gain
    _number(gain, "gain")
    tick_samples = rate / TICK_HZ
    end = max((tick + ticks + inst.get("release", _RELEASE) for inst, tick, _, ticks, _ in sound.notes), default=0)
    out = np.zeros(math.ceil((end + _TAIL) * tick_samples), dtype=np.float32)
    for inst, tick, note, ticks, velocity in sound.notes:
        render_note(out, math.floor(tick * tick_samples + 0.5), inst, note, ticks, velocity, rate)
    if sound.echo is not None:
        out = echo(out, **sound.echo, rate=rate)
    return finish(out, gain, rate)
