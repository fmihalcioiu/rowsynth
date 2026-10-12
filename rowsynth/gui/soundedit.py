"""A sound effect changed: its notes, its voices, its echo and the kind of its entry. Data in, data out. No window."""
from __future__ import annotations

import copy
import math
from collections.abc import Iterable

from ..bank import _HELD_TICKS, _KINDS, _REST, entry_kind, held_spans
from ..checks.chords import _NOTES
from ..errors import RowSynthError
from ..note import DEFAULT_RATE, TICK_HZ, _is_number
from ..note import _instrument as _playable
from ..notes import midi
from ..project import _entry, _instrument, _sound, _sounds_in
from ..sound import _settings, read_sound, sound_samples, tick_sample
from .noteedit import _stopped
from .projectedit import allowed

# A sound definition goes in, or an entry of a project's `sounds`, as the project holds it. A new one comes out, to
# be saved whole in its place, so that one change is one step to take back. What is handed in is never changed, and
# what comes back shares nothing with it. A change that cannot be made is refused, with the reason.
#
# A note is `[tick, note, ticks]` or `[tick, note, ticks, velocity]`. It is known by its voice and its index in that
# voice, both counted from 0, and the picked notes are a set of such pairs. Only `removed` and `voice_dropped` take
# a note to another index, so what is picked before any other change is picked after it.
#
# Two notes of one voice may sound at once. `render_sound` adds every note of every voice into the one buffer, each
# from an oscillator of its own, and does not read which voice a note is in. So no edit cuts a note short or removes
# one to make room for another, as the edits of a song's channel do. The notes of a voice keep the order they are
# written in: the engine adds them up in that order, and another order would round the samples another way.
#
# A voice has its instrument by name or written in place. A string is a name and is kept as it is. Whether the
# project has an instrument of that name cannot be seen from a sound, and the project's check says so when the sound
# is saved. A dict is an instrument written in place: it is held to the project's check and to the engine's as it
# is taken, and a copy of it is kept.
#
# The start and the loop of a held sound have to fit in its note, and the engine counts that in samples at a rate,
# with the fade of the instruments the note plays. So what works on a held sound takes the project's instruments and
# a rate. A name that the instruments do not have is taken to fade for no ticks, which is the shortest its notes can
# be, so no length passes here that the engine would refuse. What cannot be seen here is silence: the engine levels a
# held sound by its loudest sample and refuses one that has none, and only a render shows that.

#: A note of a sound as (voice, index), the notes that are picked, and copied notes as (voice, note) pairs.
Pick = tuple[int, int]
Picked = frozenset[Pick]
Clip = tuple[tuple[int, list], ...]

#: The lowest note and the highest that a note is placed at or moved to: the c of octave -9 and the b of octave 9,
#: the ends of what a name can say. They lie inside the engine's own, the `LOWEST` and `HIGHEST` of `rowsynth.notes`,
#: outside which the project refuses a note.
LOWEST, HIGHEST = -96, 131

# What a sound made into a held one begins with. The start, the loop and the join are the ticks its notes are held
# for over these, and the letting go is so many ticks: the lengths of the held sound in the README.
_START, _LOOP, _JOIN, _LET_GO = 10, 2, 20, 4

# The instrument a name stands for where the project's instruments are not at hand: one that fades for no ticks.
_SHORTEST = {"release": 0}


def _plain(value: float) -> int | float:
    """A number as a file writes it: a whole one as a whole number, with nothing after a point."""
    return int(value) if value == int(value) else float(value)


def _amount(value: object, what: str, least: float | None = None) -> int | float:
    """A number as a file writes it, `least` or more where one is given. Any other is refused, in words `what` opens."""
    if not _is_number(value) or (least is not None and value < least):
        most = "" if least is None else f", {least:g} or more"
        raise RowSynthError(f"{what} a number{most}, not {value!r}")
    return _plain(value)


def _is_index(value: object, count: int) -> bool:
    """Whether a value is the index of one of `count` things, counted from 0. True and False are none."""
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < count


def _index(value: object, count: int, what: str) -> int:
    """The index of one of `count` things, once it is seen to be one. `what` says what they are, for the refusal."""
    if not _is_index(value, count):
        raise RowSynthError(f"{value!r} is no {what}, of which there are {count}, counted from 0")
    return value


def _chosen(defn: dict, picked: Iterable[Pick]) -> list[Pick]:
    """The picked notes in order of voice and index. A pick that is no note of the sound is refused."""
    voices = defn["voices"]
    chosen = set()
    for pick in picked:
        if not (isinstance(pick, (tuple, list)) and len(pick) == 2 and _is_index(pick[0], len(voices))
                and _is_index(pick[1], len(voices[pick[0]]["notes"]))):
            raise RowSynthError(f"{pick!r} is picked, and that is no note of the sound: a note is (voice, index)")
        chosen.add((pick[0], pick[1]))
    return sorted(chosen)


def _fresh(tick: object, note: object, ticks: object, velocity: object = None) -> list:
    """A note as a file writes it, once its values are seen to be what the engine takes. No velocity writes none."""
    try:
        midi(note)
    except RowSynthError:
        raise RowSynthError(f"a note is a number or a name such as 'c#4', not {note!r}") from None
    made = [_amount(tick, "the tick a note starts at is", 0),
            note if isinstance(note, str) else _amount(note, "a note is"),
            _amount(ticks, "the ticks a note is held for are", 0)]
    return made if velocity is None else [*made, _amount(velocity, "the velocity of a note is")]


def _spelt(note: str | float, by: float) -> str | float:
    """A note so many semitones higher, written as it was: a name as a name, a number as a number.

    A name is spelt as a finding of the checks spells one, with sharps and no flats: `eb3` two higher is `f3`, and
    one higher `e3`. A note that is not moved keeps the spelling it has.
    """
    if not by:
        return note
    number = midi(note) + by
    return f"{_NOTES[int(number) % 12]}{int(number) // 12 - 1}" if isinstance(note, str) else _plain(number)


def placed(defn: dict, voice: int, tick: float, note: str | float, ticks: float,
           velocity: float | None = None) -> dict:
    """A sound with one more note in a voice, the last of that voice's notes. Every other note stays as it is.

    The note is written as it is given: a name as a name, a number as a number, and a velocity only where one is
    given. Refused are a voice the sound does not have, a tick or a length under 0, a note that is none or lies
    outside `LOWEST` to `HIGHEST`, and a velocity that is no number.
    """
    at = _index(voice, len(defn["voices"]), "voice of the sound")
    new = _fresh(tick, note, ticks, velocity)
    if not LOWEST <= midi(note) <= HIGHEST:
        raise RowSynthError(f"a note is placed from c-9 to b9, which are {LOWEST} and {HIGHEST}, and {note!r} is "
                            "outside them")
    made = copy.deepcopy(defn)
    made["voices"][at]["notes"].append(new)
    return made


def moved(defn: dict, picked: Iterable[Pick], by_ticks: float, by_pitch: float) -> dict:
    """A sound with the picked notes moved by so many ticks and so many semitones. The others stay as they are.

    The picked notes keep their shape. They stop together at tick 0, and at `LOWEST` and `HIGHEST`. A note written
    as a number stays a number. One written as a name stays a name, spelt with sharps as a finding of the checks
    spells a note, and one that is moved in time alone keeps the spelling it has. Each keeps its velocity or its
    lack of one. With a name among the picked, a move by part of a semitone is refused, since no name says such
    a note.
    """
    chosen = _chosen(defn, picked)
    by_ticks = _amount(by_ticks, "the ticks notes are moved by are")
    by_pitch = _amount(by_pitch, "the semitones notes are moved by are")
    made = copy.deepcopy(defn)
    group = [made["voices"][voice]["notes"][index] for voice, index in chosen]
    if not group:
        return made
    pitches = [midi(note[1]) for note in group]
    down, up = LOWEST - min(pitches), HIGHEST - max(pitches)
    if any(isinstance(note[1], str) for note in group):
        if by_pitch != int(by_pitch):
            raise RowSynthError(f"a note written as a name is moved by whole semitones, not by {by_pitch!r}")
        # Beside a number that lies between two notes, the names stop a whole step short of the end.
        down, up = math.ceil(down), math.floor(up)
    by_ticks = _stopped(by_ticks, -min(note[0] for note in group), math.inf)
    by_pitch = _stopped(by_pitch, down, up)
    for note in group:
        note[0] = _plain(note[0] + by_ticks)
        note[1] = _spelt(note[1], by_pitch)
    return made


def stretched(defn: dict, picked: Iterable[Pick], by_ticks: float, least: float = 1) -> dict:
    """A sound with the picked notes held for so many ticks longer, or shorter by a number under 0.

    None is made shorter than `least` ticks, and one that is shorter already is not made longer by that. A sound
    has no end for a note to stop at. A least under 0 is refused: the engine takes a note of no length and none
    of less.
    """
    chosen = _chosen(defn, picked)
    by_ticks = _amount(by_ticks, "the ticks notes are stretched by are")
    least = _amount(least, "the least a note is stretched to is", 0)
    made = copy.deepcopy(defn)
    for voice, index in chosen:
        note = made["voices"][voice]["notes"][index]
        note[2] = _plain(max(note[2] + by_ticks, min(least, note[2])))
    return made


def removed(defn: dict, picked: Iterable[Pick]) -> dict:
    """A sound without the picked notes. The notes after one that is gone are an index lower, so nothing is picked."""
    chosen = set(_chosen(defn, picked))
    made = copy.deepcopy(defn)
    for number, voice in enumerate(made["voices"]):
        voice["notes"] = [note for index, note in enumerate(voice["notes"]) if (number, index) not in chosen]
    return made


def with_velocity(defn: dict, picked: Iterable[Pick], velocity: float | None) -> dict:
    """A sound with the picked notes at another velocity, a factor on their level. None writes them with none."""
    chosen = _chosen(defn, picked)
    level = [] if velocity is None else [_amount(velocity, "the velocity of a note is")]
    made = copy.deepcopy(defn)
    for voice, index in chosen:
        notes = made["voices"][voice]["notes"]
        notes[index] = notes[index][:3] + level
    return made


def boxed(defn: dict, tick: float, end: float, low: float, high: float) -> Picked:
    """The notes of every voice that a box touches, which goes from a tick to another and from a note to another.

    A note touches it that is held between the two ticks and is no lower than the lower note and no higher than
    the higher. One that ends where the box starts, or begins where it ends, does not. A note of no length
    touches the box that holds the tick it starts at. Either pair can be given either way round.
    """
    tick, end = sorted((tick, end))
    low, high = sorted((low, high))
    return frozenset((number, index) for number, voice in enumerate(defn["voices"])
                     for index, note in enumerate(voice["notes"])
                     if note[0] < end and (tick < note[0] + note[2] or tick <= note[0])
                     and low <= midi(note[1]) <= high)


def copied(defn: dict, picked: Iterable[Pick]) -> Clip:
    """The picked notes as a clip: each as (voice, note) in order, moved so that the first of them begins at tick 0."""
    group = [(voice, defn["voices"][voice]["notes"][index]) for voice, index in _chosen(defn, picked)]
    start = min((note[0] for _, note in group), default=0)
    return tuple((voice, [_plain(note[0] - start), *note[1:]]) for voice, note in group)


def pasted(defn: dict, clip: Iterable[tuple[int, list]], at_tick: float) -> tuple[dict, Picked]:
    """A sound with a clip put in at a tick, and the notes of the clip, which are the picked ones after it.

    Each note goes to the voice it was copied from, as the last of that voice's notes, and nothing is cut to make
    room for it. A tick under 0 is refused, and so is a clip with a note for a voice the sound does not have.
    """
    at = _amount(at_tick, "the tick a clip is put in at is", 0)
    made = copy.deepcopy(defn)
    voices = made["voices"]
    picked = set()
    for voice, note in clip:
        notes = voices[_index(voice, len(voices), "voice of the sound")]["notes"]
        picked.add((voice, len(notes)))
        notes.append(_fresh(note[0] + at, *note[1:]))
    return made, frozenset(picked)


def snapped(tick: float, grid: float) -> int | float:
    """The tick at which the step of a grid begins that a tick is in: with steps of 4 ticks, 6.5 is in the one at 4.

    A grid whose steps have no ticks is refused.
    """
    if not _is_number(grid) or not grid > 0:
        raise RowSynthError(f"a grid has steps of more than 0 ticks, not {grid!r}")
    return _plain(math.floor(tick / grid) * grid)


def _inst(inst: object) -> str | dict:
    """An instrument for a voice, as it is kept: a name as it is, or a copy of one that is written in place.

    One written in place is held to the project's check and to the engine's, and what they refuse is refused.
    """
    if isinstance(inst, str):
        return inst
    if not isinstance(inst, dict):
        raise RowSynthError(f"an instrument is a name, or a dict of its settings written in place, not {inst!r}")
    _instrument(inst, "inst")
    _playable(inst, "inst")
    return copy.deepcopy(inst)


def voice_added(defn: dict, inst: str | dict) -> dict:
    """A sound with one more voice, the last of them, which plays `inst` and has no notes yet."""
    made = copy.deepcopy(defn)
    made["voices"].append({"inst": _inst(inst), "notes": []})
    return made


def voice_dropped(defn: dict, voice: int) -> dict:
    """A sound without one of its voices and the notes of it. The last voice of a sound is not dropped."""
    at = _index(voice, len(defn["voices"]), "voice of the sound")
    if len(defn["voices"]) == 1:
        raise RowSynthError("the last voice of a sound is not dropped: a sound with none has nothing to play")
    made = copy.deepcopy(defn)
    del made["voices"][at]
    return made


def voice_copied(defn: dict, voice: int) -> dict:
    """A sound with a copy of one of its voices as one more voice, the last: its instrument and its notes."""
    at = _index(voice, len(defn["voices"]), "voice of the sound")
    made = copy.deepcopy(defn)
    made["voices"].append(copy.deepcopy(made["voices"][at]))
    return made


def voice_inst(defn: dict, voice: int, inst: str | dict) -> dict:
    """A sound with one of its voices played by another instrument: a name, or one written in place."""
    at = _index(voice, len(defn["voices"]), "voice of the sound")
    made = copy.deepcopy(defn)
    made["voices"][at]["inst"] = _inst(inst)
    return made


def _named(sounds: Iterable[dict]) -> dict:
    """A table with an instrument under every name that sounds play, for a check that looks the names up."""
    return {voice["inst"]: {} for defn in sounds for voice in defn["voices"] if isinstance(voice["inst"], str)}


def _finite(value: object) -> bool:
    """Whether every number that a value holds, however deep, is one a file can write."""
    if isinstance(value, (dict, list)):
        return all(map(_finite, value.values() if isinstance(value, dict) else value))
    return not isinstance(value, float) or math.isfinite(value)


def with_echo(defn: dict, echo: dict | None) -> dict:
    """A sound with an echo of those settings in place of any it had, or with no echo for None.

    The settings are `time`, `feedback`, `mix` and `lp`, and only the time is needed. They are held to the
    engine's check and to the project's, and what those refuse is refused.
    """
    made = copy.deepcopy(defn)
    if echo is None:
        made.pop("echo", None)
        return made
    _settings(echo, "echo")
    made["echo"] = copy.deepcopy(echo)
    _sound(made, "sound", _named([made]))
    return made


def with_setting(place: dict, key: str, value: object) -> dict:
    """A sound definition or an entry with one of its settings at another value, or without it for None.

    The file keeps the level and the limits of a sound in its entry, so `place` is either. A definition has
    `gain`. An entry has `db`, its level, the limits `max_seconds`, `max_dc` and `max_bright`, and `check` and
    `meta`. With variants `db` can be a list with a level for each. A key that the file does not allow there is
    refused, and so is the `db` of an entry of parts, where each part has its own. The new entry is held to the
    project's check, and what that refuses is refused.
    """
    made = copy.deepcopy(place)
    if "voices" in place:
        allowed, what = ("gain",), "a sound definition"
    else:
        allowed, what = _REST, f"an entry with {entry_kind(place)!r}"
    if key not in allowed:
        raise RowSynthError(f"{key!r} is no setting of {what}, which has {', '.join(map(repr, allowed))}")
    if key == "db" and "parts" in place:
        raise RowSynthError("an entry of parts has no level of its own: each part has its own, which `part_level` "
                            "sets")
    if value is None:
        made.pop(key, None)
        return made
    if not _finite(value):
        raise RowSynthError(f"{key!r} is not set to {value!r}: a file cannot write that number")
    made[key] = copy.deepcopy(value)
    if "voices" in place:
        _sound(made, "sound", _named([made]))
    else:
        _entry(made, "entry", _named(_sounds_in(made)))
    return made


def _mark(at: object, note: object) -> dict:
    """One note a sound is expected to play, as a file writes it, once its second and its note are seen to be so."""
    try:
        midi(note)
    except RowSynthError:
        raise RowSynthError(f"a note is a number or a name such as 'c#4', not {note!r}") from None
    return {"at": _amount(at, "the second a note is expected at is", 0),
            "note": note if isinstance(note, str) else _plain(note)}


def mark_added(entry: dict, at: float, note: str | float) -> dict:
    """An entry with one more note that it is expected to play, the last of its `check`: a note at a second.

    The second is on the clock of the sound, and of each variant where it has variants: the `pitch` rule of the
    checks reads the note back there. A second under 0 is refused, and so is what is no note.
    """
    return with_setting(entry, "check", [*entry.get("check", []), _mark(at, note)])


def mark_set(entry: dict, index: int, at: float | None = None, note: str | float | None = None) -> dict:
    """An entry with one of the notes it is expected to play at another second, or as another note.

    The note is counted in its `check` from 0. With None its second, or its note, stays as it is.
    """
    marks = list(entry.get("check", []))
    was = marks[_index(index, len(marks), "note the sound is expected to play")]
    marks[index] = _mark(was["at"] if at is None else at, was["note"] if note is None else note)
    return with_setting(entry, "check", marks)


def mark_dropped(entry: dict, index: int) -> dict:
    """An entry without one of the notes it is expected to play. Without the last of them it has no `check`."""
    marks = list(entry.get("check", []))
    del marks[_index(index, len(marks), "note the sound is expected to play")]
    return with_setting(entry, "check", marks or None)


def _noted(entry: dict, key: object, there: bool) -> dict:
    """The `meta` of an entry, once a key is seen to be one whose text can be set: `there` is for one it must have.

    A key is text with something in it. A value that is no text is left as the file has it, so its key is refused.
    """
    meta = entry.get("meta", {})
    if not isinstance(key, str) or not key:
        raise RowSynthError(f"a key of NOTES is text with something in it, not {key!r}")
    if there and key not in meta:
        raise RowSynthError(f"NOTES has no {key!r}")
    if key in meta and not isinstance(meta[key], str):
        raise RowSynthError(f"{key!r} of NOTES holds no text, and is left as the file has it")
    return meta


def meta_set(entry: dict, key: str, text: str, fresh: bool = False) -> dict:
    """An entry with a text under a key of its `meta`, which the window calls NOTES. A new key is the last.

    With `fresh` the key is one more, and one the entry has already is refused. Refused as well are a key that
    is empty or no text, a text that is none, and a key whose value is no text: that is left alone.
    """
    meta = _noted(entry, key, False)
    if fresh and key in meta:
        raise RowSynthError(f"NOTES has {key!r} already")
    if not isinstance(text, str):
        raise RowSynthError(f"what stands under a key of NOTES here is text, not {text!r}")
    return with_setting(entry, "meta", {**meta, key: text})


def meta_dropped(entry: dict, key: str) -> dict:
    """An entry without one key of its `meta`. Without the last of them it has no `meta`.

    A key the entry does not have is refused, and so is one whose value is no text: that is left alone.
    """
    meta = _noted(entry, key, True)
    return with_setting(entry, "meta", {name: value for name, value in meta.items() if name != key} or None)


def kind_of(entry: dict) -> str:
    """The kind of an entry in the engine's words: 'def' is one sound, 'defs' variants, 'parts' and 'held'."""
    return entry_kind(entry)


def _sounds(entry: dict) -> list[tuple[dict, int | float | None]]:
    """Copies of the sounds of an entry, each with its level, in order. None is a level that is not written.

    A held sound has its note and then its end, both at the level of the entry.
    """
    kind, level = entry_kind(entry), entry.get("db")
    if kind == "def":
        found = [(entry["def"], level)]
    elif kind == "defs":
        found = list(zip(entry["defs"], level if isinstance(level, list) else [level] * len(entry["defs"])))
    elif kind == "parts":
        found = [(part["def"], part.get("db")) for _, part in entry["parts"]]
    else:
        found = [(entry["held"]["note"], level), (entry["held"]["end"], level)]
    return copy.deepcopy(found)


def _built(kind: str, sounds: object, level: object, entry: dict) -> dict:
    """An entry of a kind: its sounds, its level where it has one, and what else another entry holds.

    Whatever an entry holds besides its sounds and its level, every kind can hold as it is.
    """
    rest = {key: copy.deepcopy(value) for key, value in entry.items() if key not in _KINDS and key != "db"}
    return {kind: sounds, **({} if level is None else {"db": level}), **rest}


def _levelled(defn: dict, level: int | float | None) -> dict:
    """An entry of one sound, with its level where one is written. A part of a timeline is such an entry."""
    return {"def": defn, **({} if level is None else {"db": level})}


def as_one(entry: dict) -> dict:
    """An entry as one sound: its first variant or its first part, each with its own level, or the note of a held sound.

    The other sounds of the entry are let go, and so are the seconds of a part and the lengths of a held sound. A
    part is an entry of its own, and what it holds besides its sound comes with it, ahead of what the whole holds.
    """
    kind = entry_kind(entry)
    if kind == "def":
        return copy.deepcopy(entry)
    defn, level = _sounds(entry)[0]
    made = _built("def", defn, level, entry)
    if kind == "parts":
        made.update(copy.deepcopy(entry["parts"][0][1]))
    return made


def as_variants(entry: dict) -> dict:
    """An entry as variants: one sound as the only one, each part as one, or the note and the end of a held sound.

    The levels stay: one number where the sounds share a level or none is written, and otherwise a list with a
    level for each, 0 for a part that has none. The seconds of the parts and the lengths of a held sound are let go.
    """
    if entry_kind(entry) == "defs":
        return copy.deepcopy(entry)
    found = _sounds(entry)
    levels = [level for _, level in found]
    if all(level == levels[0] for level in levels):
        shared = levels[0]
    else:
        shared = [0 if level is None else level for level in levels]
    return _built("defs", [defn for defn, _ in found], shared, entry)


def as_parts(entry: dict) -> dict:
    """An entry as parts on a timeline, each with the level its sound had: every sound of it at second 0.

    The end of a held sound is the exception. It starts at the second the loop of the held sound ended, once
    round, where the view of the held sound draws it. The join and the letting go are let go.
    """
    kind = entry_kind(entry)
    if kind == "parts":
        return copy.deepcopy(entry)
    found = _sounds(entry)
    starts = [0] * len(found)
    if kind == "held":
        starts[1] = _plain((entry["held"]["start"] + entry["held"]["loop"]) / TICK_HZ)
    return _built("parts", [[at, _levelled(defn, level)] for at, (defn, level) in zip(starts, found)], None, entry)


def as_held(entry: dict, rate: int = DEFAULT_RATE) -> dict:
    """An entry as a held sound: its first sound is the note, at its own level, and its second the end.

    An entry with one sound is given an end with no notes, so that letting go only fades the loop out. Any
    further sounds are let go. The lengths are shares of the ticks from where the first note of the note begins to
    where the last is held to: the loop is half of them, the join a twentieth, and the start a tenth, after the
    ticks before the first note. The letting go is 4 ticks. A join that would be under two samples at `rate` is 0.

    A held sound needs a note that is held long enough for a start and a loop. A sound with no note is refused,
    and so is one whose notes are held too short a time for a loop of one sample at `rate`. Whether the start and
    the loop have any sound in them cannot be seen without a render, and the engine refuses to level a held sound
    that is all silence.
    """
    if entry_kind(entry) == "held":
        return copy.deepcopy(entry)
    found = _sounds(entry)
    note, level = found[0]
    spans = [(each[0], each[0] + each[2]) for voice in note["voices"] for each in voice["notes"]]
    needs = "a held sound needs a note that is held long enough for a start and a loop, and "
    if not spans:
        raise RowSynthError(needs + "this sound has no note")
    first = min(start for start, _ in spans)
    ticks = max(end for _, end in spans) - first
    if tick_sample(ticks / _LOOP, rate) < 1:
        raise RowSynthError(needs + f"the notes of this sound are held for {ticks:g} ticks")
    if len(found) > 1:
        end = found[1][0]
    else:
        end = {"voices": [{"inst": copy.deepcopy(note["voices"][0]["inst"]), "notes": []}]}
    join = ticks / _JOIN if tick_sample(ticks / _JOIN, rate) >= 2 else 0
    held = {"note": note, "start": _plain(first + ticks / _START), "loop": _plain(ticks / _LOOP),
            "join": _plain(join), "end": end, "let_go": _LET_GO}
    return _built("held", held, level, entry)


def _of(entry: dict, kind: str, what: str) -> None:
    """Refuse an entry of another kind than the one a function works on. `what` is what only that kind has."""
    found = entry_kind(entry)
    if found != kind:
        raise RowSynthError(f"only an entry with {kind!r} has {what}, not one with {found!r}")


def variant_added(entry: dict, like: int | None = None) -> dict:
    """An entry of variants with one more, the last: a copy of the variant `like`, or of the last with none named.

    Where the variants have a level each, the copy has the level of the one it copies.
    """
    _of(entry, "defs", "variants")
    made = copy.deepcopy(entry)
    defs = made["defs"]
    at = len(defs) - 1 if like is None else _index(like, len(defs), "variant")
    defs.append(copy.deepcopy(defs[at]))
    if isinstance(made.get("db"), list):
        made["db"].append(made["db"][at])
    return made


def variant_dropped(entry: dict, index: int) -> dict:
    """An entry of variants without one of them and without its level. The last variant is not dropped."""
    _of(entry, "defs", "variants")
    made = copy.deepcopy(entry)
    at = _index(index, len(made["defs"]), "variant")
    if len(made["defs"]) == 1:
        raise RowSynthError("the last variant is not dropped: an entry of variants holds one at least")
    del made["defs"][at]
    if isinstance(made.get("db"), list):
        del made["db"][at]
    return made


def variant_moved(entry: dict, index: int, to: int) -> dict:
    """An entry of variants with one of them at another place among them: the place it is at afterwards.

    Where the variants have a level each, its level goes with it.
    """
    _of(entry, "defs", "variants")
    made = copy.deepcopy(entry)
    count = len(made["defs"])
    at, to = _index(index, count, "variant"), _index(to, count, "variant")
    for listed in (made["defs"], made.get("db")):
        if isinstance(listed, list):
            listed.insert(to, listed.pop(at))
    return made


def part_added(entry: dict, at: float, like: int | None = None) -> dict:
    """An entry of parts with one more, the last, which starts at second `at` of the timeline.

    It is a copy of the part `like`, or of the last with none named: its sound, its level and what else it holds.
    A second under 0 is refused.
    """
    _of(entry, "parts", "parts")
    seconds = _amount(at, "the second a part starts at is", 0)
    made = copy.deepcopy(entry)
    parts = made["parts"]
    source = parts[len(parts) - 1 if like is None else _index(like, len(parts), "part")]
    parts.append([seconds, copy.deepcopy(source[1])])
    return made


def part_dropped(entry: dict, index: int) -> dict:
    """An entry of parts without one of them. The last part is not dropped."""
    _of(entry, "parts", "parts")
    made = copy.deepcopy(entry)
    at = _index(index, len(made["parts"]), "part")
    if len(made["parts"]) == 1:
        raise RowSynthError("the last part is not dropped: an entry of parts holds one at least")
    del made["parts"][at]
    return made


def part_at(entry: dict, index: int, at: float) -> dict:
    """An entry of parts with one of them starting at another second of the timeline. One under 0 is refused."""
    _of(entry, "parts", "parts")
    made = copy.deepcopy(entry)
    made["parts"][_index(index, len(made["parts"]), "part")][0] = _amount(at, "the second a part starts at is", 0)
    return made


def part_level(entry: dict, index: int, db: float | None) -> dict:
    """An entry of parts with one of them at another level, in decibels from the peak, or with none written for None."""
    _of(entry, "parts", "parts")
    made = copy.deepcopy(entry)
    part = made["parts"][_index(index, len(made["parts"]), "part")][1]
    if db is None:
        part.pop("db", None)
    else:
        part["db"] = _amount(db, "the level of a part is")
    return made


def _written(defn: object, instruments: dict | None) -> object:
    """A sound definition with every instrument it names written out, as the engine reads one.

    A name that `instruments` does not have stands for an instrument that fades for no ticks. What does not have
    the shape of a definition is handed back for the engine to refuse.
    """
    if not isinstance(defn, dict) or not isinstance(defn.get("voices"), list):
        return defn
    table = instruments or {}
    return {**defn, "voices": [{**voice, "inst": table.get(voice["inst"], _SHORTEST)}
                               if isinstance(voice, dict) and isinstance(voice.get("inst"), str) else voice
                               for voice in defn["voices"]]}


def _fitted(entry: dict, instruments: dict | None, rate: int) -> dict:
    """A held entry, once the engine's own sums take it at a rate: `held_spans`, and the note against them.

    What `held_spans` refuses of the lengths is refused in its words. A note that is too short for the start and
    the loop is refused as the engine refuses it when it renders, with the lengths in ticks.
    """
    start, loop, _ = held_spans(entry, rate)
    room = sound_samples(_written(entry["held"]["note"], instruments), rate)
    if room < start + loop:
        per = rate / TICK_HZ
        raise RowSynthError("the note of a held sound must be long enough for its start and its loop: it is "
                            f"{room / per:g} ticks long, and they take {(start + loop) / per:g}")
    return entry


def held_with(entry: dict, key: str, value: float, instruments: dict | None = None,
              rate: int = DEFAULT_RATE) -> dict:
    """A held entry with another `start`, `loop`, `join` or `let_go`, in ticks, kept inside what its note has room for.

    The room is counted as the engine counts it, in samples at `rate`: `sound_samples` of the note, with its
    instruments written out from `instruments`, against `tick_sample` of each length. The start is kept from the
    join up to where the loop still fits in the note after it. The loop is kept from the join, and one sample, up
    to the end of the note. The join is kept no longer than the start and the loop, and one that comes to a
    single sample is 0, since the engine fades over two samples or none. A length that is kept back is written as
    the ticks of the samples it is kept to. The letting go is any length of 0 or more.

    Refused are an entry of another kind, a key that is none of the four, a length under 0, and an entry that
    the engine's sums refuse even so, which is one that was wrong before.
    """
    _of(entry, "held", "a start, a loop, a join and a letting go")
    if key not in _HELD_TICKS:
        raise RowSynthError(f"{key!r} is no length of a held sound, which has {', '.join(map(repr, _HELD_TICKS))}")
    ticks = _amount(value, f"the {key} of a held sound is", 0)
    made = copy.deepcopy(entry)
    held = made["held"]
    if key == "let_go":
        held[key] = ticks
        return made
    start, loop, join = (tick_sample(held[name], rate) for name in ("start", "loop", "join"))
    room = sound_samples(_written(held["note"], instruments), rate)
    kept = {"start": (join, room - loop), "loop": (max(join, 1), room - start), "join": (0, min(start, loop))}
    least, most = kept[key]
    asked = tick_sample(ticks, rate)
    samples = min(max(asked, least), most)
    if key == "join" and samples == 1:
        samples = 0
    held[key] = ticks if samples == asked else _plain(samples * TICK_HZ / rate)
    return _fitted(made, instruments, rate)


def _spot(entry: dict, index: object) -> tuple[dict | list, str | int]:
    """Where an entry keeps the sound at an index: what holds it, and its key or its index there.

    One sound is at index 0. Variants and parts are counted in their order. A held sound has its note at 0 and
    its end at 1. An index that is none of them is refused.
    """
    kind = entry_kind(entry)
    if kind == "def":
        spots = [(entry, "def")]
    elif kind == "defs":
        spots = [(entry["defs"], number) for number in range(len(entry["defs"]))]
    elif kind == "parts":
        spots = [(part, "def") for _, part in entry["parts"]]
    else:
        spots = [(entry["held"], "note"), (entry["held"], "end")]
    return spots[_index(index, len(spots), "sound of the entry")]


def sound_in(entry: dict, index: int) -> dict:
    """A copy of the definition of one sound of an entry, for the functions that work on a definition.

    The one sound of an entry is at index 0, a variant or a part at its place, the note of a held sound at 0 and
    its end at 1.
    """
    holder, key = _spot(entry, index)
    return copy.deepcopy(holder[key])


def with_sound(entry: dict, index: int, defn: dict, instruments: dict | None = None,
               rate: int = DEFAULT_RATE) -> dict:
    """An entry with a definition put in the place of one of its sounds, which `sound_in` counts.

    The definition is held to the engine's reading of it and the new entry to the project's check, and what they
    refuse is refused. The note of a held sound has to stay long enough for the start and the loop, by the
    engine's sums at `rate` with the instruments written out from `instruments`: one that is too short is
    refused, and the start or the loop is made shorter first with `held_with`.
    """
    read_sound(_written(defn, instruments))
    made = copy.deepcopy(entry)
    holder, key = _spot(made, index)
    holder[key] = copy.deepcopy(defn)
    _entry(made, "entry", _named(_sounds_in(made)))
    return _fitted(made, instruments, rate) if "held" in made else made


def shared(data: dict, sound: str, index: int, voice: int, name: str,
           rate: int = DEFAULT_RATE) -> list[tuple[str, object]]:
    """The changes that make a shared instrument of one that is written in place in a voice of a sound.

    `data` is the data of a project, as `Project.to_dict` gives it, `sound` the name of one of its sounds,
    `index` which sound of that entry, as `sound_in` counts them, and `voice` the voice. The instrument moves
    to the project's `instruments` under `name`, after the others, and the voice names it. The changes are
    (place, value) pairs for `Session.change_many`, which makes them as one step: the instrument first, since
    the sound names it.

    Refused are a sound the project does not have, a voice whose instrument is a name already, and a name
    that `projectedit.allowed` refuses, as one that an instrument of the project has.
    """
    entry = data.get("sounds", {}).get(sound)
    if entry is None:
        raise RowSynthError(f"the project has no sound called {sound!r}")
    defn = sound_in(entry, index)
    at = _index(voice, len(defn["voices"]), "voice of the sound")
    inst = defn["voices"][at]["inst"]
    if not isinstance(inst, dict):
        raise RowSynthError(f"voice {at + 1} names its instrument, {inst!r}: only one written in place is made "
                            "shared")
    name = allowed(data, "instrument", name)
    table = data.get("instruments")
    made = with_sound(entry, index, voice_inst(defn, at, name), {**(table or {}), name: inst}, rate)
    place = ("instruments", {name: inst}) if table is None else (f"instruments.{name}", inst)
    return [place, (f"sounds.{sound}", made)]
