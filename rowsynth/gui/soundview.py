"""A sound effect as its view shows it: its notes as bars on a clock of seconds, and a wave as columns."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bank import entry_kind, held_spans, part_start
from ..errors import RowSynthError
from ..note import DEFAULT_RATE
from ..sound import echo_lines, read_sound, sound_samples, tick_sample

# A repeat of an echo is listed while it is stronger than this share of the sound it repeats.
_FAINTEST = 0.01


@dataclass(frozen=True)
class Bar:
    """One note of a sound: where it is held on a clock of seconds, and what plays it.

    start is when it sounds and seconds for how long it is held, which leaves out the fade once it is let go.
    pitch is its note number. inst is the name of its instrument, or "" for an instrument written in place.
    voice is the voice of the sound it is in, counted from 0.
    """

    start: float
    seconds: float
    pitch: float
    inst: str
    voice: int


@dataclass(frozen=True)
class Shape:
    """One sound as bars.

    bars are its notes in order of start. seconds is the length of its render, the tail and the echo with it.
    echo has a (seconds, strength) for each repeat of the whole sound that its echo brings, the soonest first:
    how long after the sound the repeat comes, and how strong it is from 0 to 1. With no echo it is empty.
    """

    bars: tuple[Bar, ...]
    seconds: float
    echo: tuple[tuple[float, float], ...]


def ticks_to_seconds(ticks: float, rate: int = DEFAULT_RATE) -> float:
    """A time in ticks as seconds, taken to the nearest sample at a rate as the engine takes it."""
    return tick_sample(ticks, rate) / rate


def _written_out(defn: object, instruments: dict | None) -> object:
    """A sound definition with every instrument it names written out from a table, as the engine reads one.

    The definition is left as it was. What does not have the shape of one is handed back for the engine to refuse.
    """
    if not isinstance(defn, dict) or not isinstance(defn.get("voices"), (list, tuple)):
        return defn
    voices = []
    for i, voice in enumerate(defn["voices"]):
        if isinstance(voice, dict) and isinstance(voice.get("inst"), str):
            if voice["inst"] not in (instruments or {}):
                raise RowSynthError(f"voices[{i}].inst: no instrument named {voice['inst']!r}")
            voice = {**voice, "inst": instruments[voice["inst"]]}
        voices.append(voice)
    return {**defn, "voices": voices}


def _repeats(settings: dict, rate: int) -> tuple[tuple[float, float], ...]:
    """The repeats an echo brings of a whole sound, each as (seconds after the sound, strength), the soonest first.

    Every delay line brings the sound back again and again, each time at its feedback of the time before.
    """
    lines = echo_lines(**settings, rate=rate)
    found = []
    for delay in lines.delays:
        at, strength = delay, abs(lines.level)
        while strength > _FAINTEST:
            found.append((at / rate, min(strength, 1.0)))
            at, strength = at + delay, strength * lines.feedback
    return tuple(sorted(found))


def _read(place: str, defn: object, rate: int, instruments: dict | None) -> tuple[list[tuple], int, tuple]:
    """A sound definition as the engine times it: its notes, the samples of its render, and its echo's repeats.

    A note is (the sample it starts at, the samples it is held for, note number, instrument name, voice), and
    the notes are in order of start. place is where the definition sits in its entry, and what is wrong with it
    is refused with that in front. A definition that sits nowhere has "" for its place.
    """
    if instruments is not None and not isinstance(instruments, dict):
        raise RowSynthError(f"instruments: must be a dict, not {instruments!r}")
    try:
        full = _written_out(defn, instruments)
        sound = read_sound(full)
    except RowSynthError as error:
        raise RowSynthError(f"{place}: {error}" if place else str(error)) from None
    # The definition has been read, so all that can be refused from here on is the rate.
    total = sound_samples(full, rate)
    notes = []
    for _, tick, pitch, ticks, _, voice in sound.notes:
        named = defn["voices"][voice]["inst"]
        notes.append((tick_sample(tick, rate), tick_sample(ticks, rate), pitch,
                      named if isinstance(named, str) else "", voice))
    notes.sort(key=lambda note: note[0])
    return notes, total, () if sound.echo is None else _repeats(sound.echo, rate)


def _bars(notes: list[tuple], rate: int, first: int, last: int) -> tuple[Bar, ...]:
    """The notes that are held between two samples, as bars on a clock that starts at the first of the two.

    A note that is held across either of them is cut there. A note of no length is kept where it starts.
    """
    bars = []
    for at, length, pitch, inst, voice in notes:
        if at < last and max(at + length, at + 1) > first:
            low, high = max(at, first), min(at + length, last)
            bars.append(Bar((low - first) / rate, (high - low) / rate, pitch, inst, voice))
    return tuple(bars)


def _shape(place: str, defn: object, rate: int, instruments: dict | None) -> Shape:
    """The shape of the whole of a sound definition that sits at a place in its entry."""
    notes, total, echo = _read(place, defn, rate, instruments)
    return Shape(_bars(notes, rate, 0, total), total / rate, echo)


def shape(defn: dict, rate: int = DEFAULT_RATE, instruments: dict | None = None) -> Shape:
    """One sound definition as bars, each where `render_sound` puts its note at a rate.

    A voice has its instrument written in place, or names it. A name is looked up in instruments, the table of
    the project the sound is in, and one that is not there is refused. Nothing is rendered.
    """
    return _shape("", defn, rate, instruments)


def _listed(value: object, where: str) -> list:
    """value, which must be a list. A tuple does for one."""
    if not isinstance(value, (list, tuple)):
        raise RowSynthError(f"{where}: must be a list, not {value!r}")
    return value


def shapes(entry: dict, rate: int = DEFAULT_RATE,
           instruments: dict | None = None) -> list[tuple[str, float, Shape]]:
    """An entry of a project's sounds as the sounds it holds, each as (label, at, shape).

    at is the second on the entry's clock at which the shape's own clock starts. One sound is labelled "" and
    is at 0. Variants are labelled "1", "2" and so on, each at 0. Parts are labelled the same way, by their
    place, and each is at the second it starts on the timeline. A held sound is its three zones, "start",
    "loop" and "end", each at the second it begins when they are laid end to end. A zone is as long as the
    buffer `render_entry` gives for it. The start and the loop hold the bars of the note that fall in them, a
    bar that crosses a border cut there, and both carry the note's echo. The join and the letting go are
    lengths in ticks, which `ticks_to_seconds` puts on the same clock. instruments is as `shape` takes it.
    What is wrong with a sound of the entry is refused with its place in front: "defs[1]: ...".
    """
    kind = entry_kind(entry)
    # A rate that is none is refused once, before any place of the entry is looked at.
    tick_sample(0, rate)
    if kind == "def":
        return [("", 0.0, _shape("def", entry["def"], rate, instruments))]
    if kind == "defs":
        return [(str(i + 1), 0.0, _shape(f"defs[{i}]", defn, rate, instruments))
                for i, defn in enumerate(_listed(entry["defs"], "defs"))]
    if kind == "parts":
        found = []
        for i, part in enumerate(_listed(entry["parts"], "parts")):
            try:
                at, inner = part_start(part, rate)
            except RowSynthError as error:
                raise RowSynthError(f"parts[{i}]: {error}") from None
            found.append((str(i + 1), at / rate, _shape(f"parts[{i}]: def", inner["def"], rate, instruments)))
        return found
    start, loop, _ = held_spans(entry, rate)
    held = entry["held"]
    notes, total, echo = _read("held: note", held["note"], rate, instruments)
    if total < start + loop:
        raise RowSynthError(f"held: note: must be long enough for the start and the loop: it is {total} samples "
                            f"at {rate} Hz, and they take {start + loop}")
    end = _shape("held: end", held["end"], rate, instruments)
    return [("start", 0.0, Shape(_bars(notes, rate, 0, start), start / rate, echo)),
            ("loop", start / rate, Shape(_bars(notes, rate, start, start + loop), loop / rate, echo)),
            ("end", (start + loop) / rate, end)]


def _whole(value: object, where: str) -> None:
    """Refuses what is not a whole number of 0 or more."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise RowSynthError(f"{where}: must be a whole number, 0 or more, not {value!r}")


def columns(samples: np.ndarray, width: int, start: int = 0, stop: int | None = None) -> np.ndarray:
    """A wave for drawing, a column a pixel: the lowest and the highest sample of each of `width` equal stretches.

    The stretches are of samples[start:stop], counted as a slice counts them, so what lies past the end is left
    out. The result is a float32 array with a row a column, [lowest, highest]. With more columns than samples a
    stretch can have no sample in it, and it takes the value of the sample it falls on. With no samples at all
    every column is 0. For a sound in two sides, hand in their mean.
    """
    if not isinstance(samples, np.ndarray) or samples.ndim != 1:
        raise RowSynthError("samples: must be an array of one dimension")
    for value, where in ((width, "width"), (start, "start"), (0 if stop is None else stop, "stop")):
        _whole(value, where)
    wave = samples[start:stop]
    out = np.zeros((width, 2), dtype=np.float32)
    if width and len(wave):
        # Each stretch starts at the sample its share of the way along falls on. Where two start at the same
        # sample the first of them is empty, and `reduceat` gives an empty stretch the sample it starts at.
        firsts = (np.arange(width, dtype=np.int64) * len(wave)) // width
        out[:, 0] = np.minimum.reduceat(wave, firsts)
        out[:, 1] = np.maximum.reduceat(wave, firsts)
    return out
