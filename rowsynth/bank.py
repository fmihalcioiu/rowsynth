"""A sound bank: every sound of a set rendered and brought to one level, with its variants."""
from __future__ import annotations

import math

import numpy as np

from .errors import RowSynthError
from .note import DEFAULT_RATE, TICK_HZ, _buffer, _count, _length, _number
from .sound import _keys, _list, render_sound, tick_sample

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

# The keys of a held sound that are lengths of time in ticks, sixty to a second. A length need not be whole.
_HELD_TICKS = ("start", "loop", "join", "let_go")


def _scale(samples: np.ndarray, factor: float) -> None:
    """Multiplies samples by factor, in place. Each product is worked out as a double and stored as a float32."""
    samples[:] = samples.astype(np.float64) * factor


def _to_peak(buffers: list[np.ndarray], peak_db: float) -> bool:
    """Scales buffers in place, all by one factor, so that the loudest sample among them sits at peak_db.

    False when all of them are silent: there is then no loudest sample to go by, and they are left as they are.
    """
    peak = max(float(np.abs(samples).max(initial=0.0)) for samples in buffers)
    if peak > 0:
        factor = 10 ** (peak_db / 20) / peak
        for samples in buffers:
            _scale(samples, factor)
    return peak > 0


def normalise(samples: np.ndarray, peak_db: float = PEAK_DB) -> np.ndarray:
    """Scales samples in place so that the loudest of them sits at peak_db, and hands the same array back.

    samples is a float32 array, and peak_db is in decibels against full scale. Silence has no loudest sample
    to go by, and is left as it is.
    """
    _buffer(samples, "samples")
    _number(peak_db, "peak_db")
    _to_peak([samples], peak_db)
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


def entry_kind(entry: object) -> str:
    """The kind of an entry: the one key of 'def', 'defs', 'parts' and 'held' it has.

    Refuses what is not an entry, or is of no kind or of two.
    """
    _keys(entry, "entry", (), _KINDS + _REST)
    kinds = [kind for kind in _KINDS if kind in entry]
    if len(kinds) != 1:
        has = " and ".join(repr(kind) for kind in kinds) or "none of them"
        raise RowSynthError(f"entry: must have one of 'def', 'defs', 'parts' and 'held', and has {has}")
    return kinds[0]


def _raw(defn: object, place: str, rate: int) -> np.ndarray:
    """One def as a buffer before it is levelled: rendered at RENDER_GAIN, whatever gain it has of its own.

    place is where the def sits in its entry. What is wrong with the def is refused with that in front.
    """
    try:
        return render_sound(defn, rate, RENDER_GAIN)
    except RowSynthError as error:
        raise RowSynthError(f"{place}: {error}") from None


def _levelled(defn: object, place: str, db: float, rate: int) -> np.ndarray:
    """One def as a buffer at its level: rendered at RENDER_GAIN, brought to PEAK_DB, and set db decibels from there."""
    samples = _raw(defn, place, rate)
    # Two steps, each stored before the next: brought to the peak, and from there to its own level. A held
    # sound gets to its level in one step (see `_held`), and the two ways do not give the same samples: here
    # every sample is rounded to a float32 twice, and the two factors multiplied are not that one factor.
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


def part_start(part: object, rate: int = DEFAULT_RATE) -> tuple[int, dict]:
    """One part of a timeline, [seconds, entry]: the sample it starts at, and its entry, which is of one def.

    The sample is the one nearest to its seconds. Nothing is rendered.
    """
    _count(rate, "rate", 1)
    if not isinstance(part, (list, tuple)) or len(part) != 2:
        raise RowSynthError(f"must be [seconds, entry], not {part!r}")
    seconds, entry = part
    _length(seconds, "seconds")
    kind = entry_kind(entry)
    if kind != "def":
        # A part gives the timeline one buffer. Of variants, all but the first would be lost without a word.
        raise RowSynthError(f"entry: a part is one sound: it must have 'def', not {kind!r}")
    return math.floor(seconds * rate + 0.5), entry


def _placed(part: object, rate: int) -> tuple[int, np.ndarray]:
    """One part of a timeline, [seconds, entry]: the sample it starts at, and its buffer at its own level."""
    at, entry = part_start(part, rate)
    return at, _one(entry, rate)


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


def _joined(note: np.ndarray, start: int, loop: int, join: int) -> np.ndarray:
    """The loop of a held note: `loop` samples from `start` on, the last `join` of them faded into what led up to it.

    Over those last samples the loop's own are weighted by a cosine that falls from 1 to 0, and the `join`
    samples just before `start` by a sine that rises from 0 to 1. The loop then ends on the sample that came
    before its first, so its end runs into its beginning. A join of 0 fades nothing.
    """
    out = note[start:start + loop].copy()
    angle = (math.pi / 2) * (np.arange(join) / (join - 1))
    own = note[start + loop - join:start + loop].astype(np.float64)
    before = note[start - join:start].astype(np.float64)
    out[loop - join:] = own * np.cos(angle) + before * np.sin(angle)
    return out


def _spans(held: object, rate: int) -> tuple[int, int, int]:
    """The samples the start, the loop and the join of a held sound take, once its keys and lengths are checked.

    What is wrong with the held sound is refused with "held" and its key in front: "held: join: ...".
    """
    _keys(held, "held", ("note", "end") + _HELD_TICKS)
    for key in _HELD_TICKS:
        _length(held[key], f"held: {key}")
    start, loop, join = (tick_sample(held[key], rate) for key in ("start", "loop", "join"))
    if loop == 0:
        raise RowSynthError(f"held: loop: must be at least one sample long, and {held['loop']!r} ticks "
                            f"are none at {rate} Hz")
    if join == 1:
        # The weights of the fade are spread over join - 1 steps. One sample has no step to spread them over.
        raise RowSynthError(f"held: join: must be 0 or at least two samples long, and {held['join']!r} ticks "
                            f"are one at {rate} Hz")
    for key, count in (("loop", loop), ("start", start)):
        if join > count:
            # The fade is laid on the end of the loop, and reads as many samples from the end of the start.
            raise RowSynthError(f"held: join: must not be longer than the {key}: {join} samples "
                                f"against {count} at {rate} Hz")
    return start, loop, join


def _cut(entry: dict, rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The start, the loop and the end of {"held": {...}}, as they come before they are levelled.

    What is wrong with the held sound is refused with "held" and its key in front: "held: join: ...".
    """
    held = entry["held"]
    start, loop, join = _spans(held, rate)
    note = _raw(held["note"], "held: note", rate)
    if len(note) < start + loop:
        raise RowSynthError(f"held: note: must be long enough for the start and the loop: it is {len(note)} "
                            f"samples at {rate} Hz, and they take {start + loop}")
    return note[:start].copy(), _joined(note, start, loop, join), _raw(held["end"], "held: end", rate)


def _only_held(entry: object, rate: object) -> None:
    """Refuses an entry of another kind than a held one, and a rate that is none."""
    kind = entry_kind(entry)
    if kind != "held":
        raise RowSynthError(f"entry: only an entry with 'held' has a start, a loop and an end, not one with {kind!r}")
    _count(rate, "rate", 1)


def held_spans(entry: dict, rate: int = DEFAULT_RATE) -> tuple[int, int, int]:
    """The samples the start, the loop and the join of a held entry take at a rate, with nothing rendered.

    They are what `held_parts` cuts and fades by: each length in ticks taken to the nearest sample. What it
    refuses of the entry's keys and lengths is refused here. Whether the note is long enough for its start and
    its loop is not looked at, since that takes the note.
    """
    _only_held(entry, rate)
    return _spans(entry["held"], rate)


def held_parts(entry: dict, rate: int = DEFAULT_RATE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The start, the loop and the end of a held entry, as they come before `render_entry` levels them.

    entry is {"held": {"note": def, "start": ticks, "loop": ticks, "join": ticks, "end": def, "let_go": ticks}}.
    A held sound goes on for as long as something holds it: its start is played once, its loop goes round
    after that, and its end is played when it is let go. The start and the loop are cut from one render of the
    note, at RENDER_GAIN: the start is its first `start` ticks, and the loop the `loop` ticks that follow. A
    length in ticks is taken to the nearest sample. The last `join` ticks of the loop are faded, at equal
    power, into the `join` ticks before the loop began, so that the loop comes round to its first sample
    without a click. A join of 0 leaves the loop as it was cut. The end is its def rendered on its own, at
    RENDER_GAIN. "let_go" is checked here and not used.
    """
    _only_held(entry, rate)
    return _cut(entry, rate)


def _held(entry: dict, rate: int) -> list[np.ndarray]:
    """The buffers of {"held": {...}}: its start, its loop and its end, levelled as one sound."""
    db = entry.get("db", 0)
    # The three are one sound heard in turn, not variants: they have one level, and a list is none.
    _number(db, "db")
    parts = list(_cut(entry, rate))
    # One factor for all three, from the loudest sample of them all. Each brought to the peak by its own
    # loudest sample, the loop would not go on at the level the start left off at.
    if not _to_peak(parts, PEAK_DB + db):
        raise RowSynthError("held: the start, the loop and the end are all silent, and there is no loudest sample "
                            "to level them by")
    return parts


def render_entry(entry: dict, rate: int = DEFAULT_RATE) -> list[np.ndarray]:
    """The buffers of a bank entry: one for a def, one a variant for defs, one for parts, and three for a held sound.

    Every def is rendered at RENDER_GAIN, its own gain set aside, and its loudest sample is brought to PEAK_DB.
    Then it is set at the entry's "db", its level against that peak in decibels: at -6 its samples are about
    half the size. "db" is one number for the whole entry, or a list with a number for each variant, and 0
    when it is left out.
    {"parts": [[seconds, entry], ...]} is several sounds heard as one. Each part is an entry of one def with
    a "db" of its own, and starts at the sample nearest to its seconds. The parts are added up into one
    buffer, so the entry has no "db" itself.
    {"held": {...}} is the start, the loop and the end that `held_parts` makes, in that order. The three are
    levelled as one: a single factor brings the loudest sample of them all to PEAK_DB and sets it at the
    entry's "db", which is one number.
    """
    kind = entry_kind(entry)
    _count(rate, "rate", 1)
    if kind == "def":
        return [_one(entry, rate)]
    if kind == "defs":
        return _variants(entry, rate)
    if kind == "parts":
        return [_timeline(entry, rate)]
    return _held(entry, rate)


def render_bank(sounds: dict, rate: int = DEFAULT_RATE) -> dict:
    """Every sound of a set, rendered: {name: {"buffers": [...], "meta": {...}, "held": {"let_go": seconds} or None}}.

    sounds is {name: entry}. "buffers" are what `render_entry` makes of the entry. "meta" is the entry's own,
    carried through as it is, and an empty dict for an entry that has none. "held" is None but for a held
    entry. There it says how long the sound takes to be let go: the entry's "let_go" ticks, in seconds.
    An entry that is refused is refused with the name of its sound in front.
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
        held = {"let_go": entry["held"]["let_go"] / TICK_HZ} if "held" in entry else None
        bank[name] = {"buffers": buffers, "meta": entry.get("meta", {}), "held": held}
    return bank
