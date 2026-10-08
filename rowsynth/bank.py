"""A sound bank: every sound of a set rendered and brought to one level, with its variants."""
from __future__ import annotations

import math

import numpy as np

from .errors import RowSynthError
from .note import DEFAULT_RATE, _buffer, _count, _length, _number
from .sound import _keys, _list, render_sound

# The gain every sound of a bank is rendered at, whatever gain it has of its own: low enough for a sound to
# stay under the knee of the output stage and come out with its shape unbent.
RENDER_GAIN = 0.12

# Where the loudest sample of every sound is brought to, in decibels against full scale, before the sound is
# set at its own level.
PEAK_DB = -3.0

# What an entry can be made of. It has one of these keys and no more.
_KINDS = ("def", "defs", "parts", "held")

# The other keys an entry can have. Those after "meta" are for whoever checks a sound, and are not read here.
_REST = ("db", "meta", "check", "max_seconds", "max_dc", "max_bright")


def _scale(samples: np.ndarray, factor: float) -> None:
    """Multiplies samples by factor, in place. Each product is worked out as a double and stored as a float32."""
    samples[:] = samples.astype(np.float64) * factor


def normalise(samples: np.ndarray, peak_db: float = PEAK_DB) -> np.ndarray:
    """Scales samples in place so that the loudest of them sits at peak_db, and hands the same array back.

    samples is a float32 array, and peak_db is in decibels against full scale. Silence has no loudest sample
    to go by, and is left as it is.
    """
    _buffer(samples, "samples")
    _number(peak_db, "peak_db")
    peak = float(np.abs(samples).max(initial=0.0))
    if peak > 0:
        _scale(samples, 10 ** (peak_db / 20) / peak)
    return samples


def _levels(db: object, count: int) -> list:
    """The level of each of count variants, from the "db" of their entry: one number for all, or a list of them."""
    if not isinstance(db, (list, tuple)):
        _number(db, "db")
        return [db] * count
    if len(db) != count:
        raise RowSynthError(f"db: a list must have one level a variant: {count}, not {len(db)}")
    for i, level in enumerate(db):
        _number(level, f"db[{i}]")
    return list(db)


def _kind(entry: object) -> str:
    """The kind of an entry: the one key of _KINDS it has. Refuses what is not an entry, or is of no kind or of two."""
    _keys(entry, "entry", (), _KINDS + _REST)
    kinds = [kind for kind in _KINDS if kind in entry]
    if len(kinds) != 1:
        has = " and ".join(repr(kind) for kind in kinds) or "none of them"
        raise RowSynthError(f"entry: must have one of 'def', 'defs', 'parts' and 'held', and has {has}")
    return kinds[0]


def _levelled(defn: object, place: str, db: float, rate: int) -> np.ndarray:
    """One def as a buffer at its level: rendered at RENDER_GAIN, brought to PEAK_DB, and set db decibels from there.

    place is where the def sits in its entry. What is wrong with the def is refused with that in front.
    """
    try:
        samples = render_sound(defn, rate, RENDER_GAIN)
    except RowSynthError as error:
        raise RowSynthError(f"{place}: {error}") from None
    # Two steps, each stored before the next: brought to the peak, and from there to its own level.
    _scale(normalise(samples), 10 ** (db / 20))
    return samples


def _one(entry: dict, rate: int) -> np.ndarray:
    """The buffer of {"def": def}, at the entry's level."""
    return _levelled(entry["def"], "def", _levels(entry.get("db", 0), 1)[0], rate)


def _variants(entry: dict, rate: int) -> list[np.ndarray]:
    """The buffers of {"defs": [def, ...]}: one a variant, each at its level."""
    defs = entry["defs"]
    _list(defs, "defs")
    levels = _levels(entry.get("db", 0), len(defs))
    return [_levelled(defn, f"defs[{i}]", level, rate) for i, (defn, level) in enumerate(zip(defs, levels))]


def _placed(part: object, rate: int) -> tuple[int, np.ndarray]:
    """One part of a timeline, [seconds, entry]: the sample it starts at, and its buffer at its own level."""
    if not isinstance(part, (list, tuple)) or len(part) != 2:
        raise RowSynthError(f"must be [seconds, entry], not {part!r}")
    seconds, entry = part
    _length(seconds, "seconds")
    kind = _kind(entry)
    if kind != "def":
        # A part gives the timeline one buffer. Of variants, all but the first would be lost without a word.
        raise RowSynthError(f"entry: a part is one sound: it must have 'def', not {kind!r}")
    return math.floor(seconds * rate + 0.5), _one(entry, rate)


def _timeline(entry: dict, rate: int) -> np.ndarray:
    """The buffer of {"parts": [[seconds, entry], ...]}: every part from its own start, all of them added up.

    The buffer ends with the part that ends last, which need not be the last of the list. What is wrong with
    a part is refused with the part in front: "parts[1]: ...".
    """
    if "db" in entry:
        raise RowSynthError("db: an entry with 'parts' has no level of its own: each part has its 'db'")
    _list(entry["parts"], "parts")
    if not entry["parts"]:
        raise RowSynthError("parts: must have at least one part")
    placed = []
    for i, part in enumerate(entry["parts"]):
        try:
            placed.append(_placed(part, rate))
        except RowSynthError as error:
            raise RowSynthError(f"parts[{i}]: {error}") from None
    out = np.zeros(max(at + len(samples) for at, samples in placed), dtype=np.float32)
    for at, samples in placed:
        out[at:at + len(samples)] += samples
    return out


def render_entry(entry: dict, rate: int = DEFAULT_RATE) -> list[np.ndarray]:
    """The buffers of a bank entry: one for a def, one a variant for defs, and one for parts on a timeline.

    Every def is rendered at RENDER_GAIN, its own gain set aside, and its loudest sample is brought to PEAK_DB.
    Then it is set at the entry's "db", its level against that peak in decibels: at -6 its samples are about
    half the size. "db" is one number for the whole entry, or a list with a number for each variant, and 0
    when it is left out.
    {"parts": [[seconds, entry], ...]} is several sounds heard as one. Each part is an entry of one def with
    a "db" of its own, and starts at the sample nearest to its seconds. The parts are added up into one
    buffer, so the entry has no "db" itself. A held entry is refused.
    """
    kind = _kind(entry)
    _count(rate, "rate", 1)
    if kind == "def":
        return [_one(entry, rate)]
    if kind == "defs":
        return _variants(entry, rate)
    if kind == "parts":
        return [_timeline(entry, rate)]
    raise RowSynthError("entry: an entry with 'held' cannot be rendered")


def render_bank(sounds: dict, rate: int = DEFAULT_RATE) -> dict:
    """Every sound of a set, rendered: {name: {"buffers": [...], "meta": {...}, "held": None}}.

    sounds is {name: entry}. "buffers" are what `render_entry` makes of the entry. "meta" is the entry's own,
    carried through as it is, and an empty dict for an entry that has none. "held" is None for an entry of
    these kinds. An entry that is refused is refused with the name of its sound in front.
    """
    if not isinstance(sounds, dict):
        raise RowSynthError(f"sounds: must be a dict, not {sounds!r}")
    _count(rate, "rate", 1)
    bank = {}
    for name, entry in sounds.items():
        try:
            buffers = render_entry(entry, rate)
        except RowSynthError as error:
            raise RowSynthError(f"{name}: {error}") from None
        bank[name] = {"buffers": buffers, "meta": entry.get("meta", {}), "held": None}
    return bank
