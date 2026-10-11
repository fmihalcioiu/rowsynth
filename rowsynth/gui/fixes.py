"""One-click repairs: for a finding of the checks, the changes to a project that repair it, where a repair is safe.

A repair is safe when it changes only what the finding is about, makes nothing sound different beyond what the
finding names, and leaves the rule kept. Each is worked out by asking the rule itself about the project as it
would be after the repair, and one that the rule would still find fault with is not given.

The rules that have a repair:

- "rows": the short channel is given rests up to the rows of its pattern. It was silent there already, so the
  song sounds as it did.
- "level": the gain of the song is lowered to the first number that reads cleanly at which no mix the rule holds
  is too loud. Every "level" finding of a song has this one repair.
- "peak": every note of the sound is set at one share of its level, the most in whole hundredths at which the
  output stage bends the sound no longer. The bank brings every sound to one peak after that, so the sound is as
  loud as it was and only the bend is gone.
- "layers", where a version names a group twice: the version names each of its groups once.

The rules that have none, and why:

- "layers", for all else it finds: which group a channel goes in, and which version a group, is a choice.
- "ends": the jump is how the first note of the sound sets in, and any repair would change how it begins.
- "dc" and "bright": the offset and the brightness are in the wave the instrument makes. Only another sound would
  pass, or a wider limit, and the limit is what was asked for.
- "length": the notes may be too long or the limit too short, and which of the two cannot be known.
- "pitch" and "chords": a note or a chord would have to be guessed.
- "join", of a held sound and of a song: a loop that comes round without a click is found by ear, with other
  lengths or other notes.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from typing import NamedTuple

from ..checks import Finding, _whole_rows, songs, sounds
from ..edit import get_in, steps
from ..errors import RowSynthError
from ..note import DEFAULT_RATE
from ..project import Project
from ..song import _GAIN, _channels, parse
from ..songtext import _rest, _sizes, _word

# How many times the gain of a song is lowered, and the song played again, before its repair is given up. A song
# whose channels are under the knee of the output stage is right at the first.
_TRIES = 4

# The steps of a place: names, and whole numbers for the items of lists.
Path = list[str | int]


class Fix(NamedTuple):
    """A repair: what it does, in a few plain words, and the changes that make it.

    Each change is (where, value), as `Session.change_many` takes them, to be made together as one step.
    """

    label: str
    changes: tuple[tuple[str, object], ...]


def _at(*path: str | int) -> str:
    """A place as it is written, from its steps. A name that a place cannot spell, as one with a dot, is refused."""
    where = "".join(f"[{step}]" if isinstance(step, int) else f".{step}" for step in path).lstrip(".")
    if steps(where) != list(path):
        raise RowSynthError(f"{where!r} is not the place of {path!r}")
    return where


def _under(most: float) -> float:
    """The largest number below `most` that reads cleanly: with two figures that count, and two decimals at least."""
    places = max(2, 1 - math.floor(math.log10(most)))
    return round((math.ceil(most * 10 ** places) - 1) / 10 ** places, places)


def _filled(data: dict, path: Path, finding: Finding, rate: int, shared: dict) -> Fix | None:
    """The repair of a "rows" finding: rests after the text of the short channel, up to the rows of its pattern.

    A finding names its channel only in its words. So each channel of the pattern is filled in turn, and the one
    after whose filling the rule no longer has this finding is the one.
    """
    if len(path) != 4 or path[2] != "patterns":
        return None
    name, pattern = path[1], path[3]
    song = get_in(data, _at("songs", name))
    texts = get_in(song, _at("patterns", pattern))
    if finding not in songs._rows(_at("songs", name), {**song, "patterns": {pattern: texts}}):
        return None
    per_beat = song.get("rows_per_beat", 4)
    read = {channel: parse(text, settings["inst"], per_beat, song.get("drums") if settings.get("drums") else None)[1]
            for channel, settings, text in _channels(song, pattern)}
    # Rows that triplets left a hair off a whole number are filled from that number, which the rule takes them for.
    longest = _whole_rows(max(read.values(), default=0.0))
    for channel, rows in read.items():
        rests: list = []
        try:
            where = _at(*path, channel)
            _rest(rests, _whole_rows(rows), longest, _sizes(per_beat))
        except RowSynthError:
            continue
        text = " ".join(filter(None, [texts[channel].rstrip(), *(_word(rest, None) for rest in rests)]))
        mended = {**song, "patterns": {pattern: {**texts, channel: text}}}
        if finding not in songs._rows(_at("songs", name), mended):
            return Fix(f"{name}: {channel} of pattern {pattern} filled to {songs._said(longest)} rows",
                       ((where, text),))
    return None


def _quieter(song: dict, rate: int) -> float | None:
    """The gain at which the "level" rule finds nothing too loud in a song, or None where none is found.

    The gain is lowered by as much as the loudest sample is over the limit, to a number that reads cleanly, and
    the song is played again to see. A channel that the output stage bends comes down by less than its gain
    does, so that can take more than one try.
    """
    gain = song.get("gain", _GAIN)
    loud = songs._loudest(song, rate)
    for _ in range(_TRIES):
        if not (gain > 0 and math.isfinite(loud) and loud >= songs._PEAK):
            return None
        gain = _under(gain * songs._PEAK / loud)
        loud = songs._loudest({**song, "gain": gain}, rate)
        if loud < songs._PEAK:
            return gain
    return None


def _lowered(data: dict, path: Path, finding: Finding, rate: int, shared: dict) -> Fix | None:
    """The repair of a "level" finding: the gain of its song, lowered. It is worked out once for a song."""
    if len(path) != 2:
        return None
    name, where = path[1], _at(*path, "gain")
    if where not in shared:
        gain = _quieter(Project.from_dict(data).song(name), rate)
        shared[where] = None if gain is None else Fix(f"{name}: gain lowered to {gain:g}", ((where, gain),))
    return shared[where]


def _once(data: dict, path: Path, finding: Finding, rate: int, shared: dict) -> Fix | None:
    """The repair of a "layers" finding on a version that names a group twice: the version with each group once.

    The version is found as `_filled` finds its channel. Nothing else that the rule finds has a repair.
    """
    if len(path) != 2:
        return None
    song = get_in(data, finding.target)
    versions = get_in(song, "layers.versions")
    if finding not in songs._layers(finding.target, song):
        return None
    for version, named in versions.items():
        once = list(dict.fromkeys(named))
        try:
            where = _at(*path, "layers", "versions", version)
        except RowSynthError:
            continue
        mended = {**song, "layers": {**song["layers"], "versions": {**versions, version: once}}}
        if len(once) < len(named) and finding not in songs._layers(finding.target, mended):
            return Fix(f"{path[1]}: version {version} names each group once", ((where, once),))
    return None


def _sound_of(path: Path) -> tuple[Path, str, tuple[str, ...]] | None:
    """The sound that a target of the sound checks is a render of, or None for a target that is of none.

    It is given as the steps to it inside its entry, what it is called in words, and every target that the
    "peak" findings on its renders go by: the note of a held sound is rendered into its start and its loop.
    """
    name, rest = path[1], path[2:]
    target = _at(*path)
    if not rest:
        return ["def"], name, (target,)
    if len(rest) == 1 and isinstance(rest[0], int):
        return ["defs", rest[0]], f"{name}, variant {rest[0] + 1}", (target,)
    if len(rest) == 2 and rest[0] == "parts" and isinstance(rest[1], int):
        return ["parts", rest[1], 1, "def"], f"{name}, part {rest[1] + 1}", (target,)
    if rest in (["start"], ["loop"]):
        return ["held", "note"], f"{name}, held note", (_at(*path[:2], "start"), _at(*path[:2], "loop"))
    if rest == ["end"]:
        return ["held", "end"], f"{name}, end", (target,)
    return None


def _softer(entry: dict, target: str, inside: Path, targets: tuple[str, ...], rate: int) -> tuple[int, list] | None:
    """The most hundredths of their level at which the notes of one sound of an entry are bent no longer.

    `entry` has its instruments written out, and its notes are changed here. With the hundredths come the
    notes of each voice at that level. None where the sound is not bent, where no level above nothing passes,
    and where the checks find something in the entry afterwards that they did not find before.
    """
    def kinds(found: list[Finding]) -> Counter:
        return Counter((one.rule, one.target) for one in found)

    def bent(found: list[Finding]) -> bool:
        return any(one.rule == "peak" and one.target in targets for one in found)

    voices = get_in(entry, _at(*inside))["voices"]
    notes = [voice["notes"] for voice in voices]

    def at(share: int) -> list[list]:
        return [[[*note[:3], round((note[3] if len(note) > 3 else 1) * share / 100, 4)] for note in voice]
                for voice in notes]

    before = list(sounds._findings(target, entry, rate))
    if not bent(before):
        return None
    # The top of a sound falls as its notes do, so the most that passes is found by halving what is left to try.
    low, high, after = 0, 100, None
    while high - low > 1:
        share = (low + high) // 2
        for voice, made in zip(voices, at(share)):
            voice["notes"] = made
        found = list(sounds._findings(target, entry, rate))
        if bent(found):
            high = share
        else:
            low, after = share, found
    if after is None or kinds(after) - kinds(before):
        return None
    return low, at(low)


def _unbent(data: dict, path: Path, finding: Finding, rate: int, shared: dict) -> Fix | None:
    """The repair of a "peak" finding: every note of its sound at one share of its level. It is worked out once."""
    found = _sound_of(path)
    if found is None:
        return None
    inside, said, targets = found
    name = path[1]
    where = _at("sounds", name, *inside)
    if where not in shared:
        made = _softer(Project.from_dict(data).sound(name), _at("sounds", name), inside, targets, rate)
        shared[where] = None if made is None else Fix(
            f"{said}: every note at {made[0] / 100:g} of its level",
            tuple((_at("sounds", name, *inside, "voices", index, "notes"), notes)
                  for index, notes in enumerate(made[1]) if notes))
    return shared[where]


# The repair of each rule that has one, by the table its findings are in and the name of the rule.
_REPAIRS: dict[tuple[str | int, str], Callable[[dict, Path, Finding, int, dict], Fix | None]] = {
    ("songs", "rows"): _filled,
    ("songs", "level"): _lowered,
    ("songs", "layers"): _once,
    ("sounds", "peak"): _unbent,
}


def fix(data: dict, finding: Finding, rate: int = DEFAULT_RATE, shared: dict | None = None) -> Fix | None:
    """The safe repair of a finding in a project, as (label, changes), or None where there is no safe one.

    `data` is the project as plain data, as `Project.to_dict` gives it, and `finding` one that the checks gave
    for it at `rate`. The label says what the repair does, with the new number where it sets one. Each change
    is (where, value), to be made together. Nothing is changed here.

    `shared` is for whoever asks for the repairs of many findings of one project: a dict of their own, empty at
    first. A repair that several findings have, as the gain of a song that is too loud on both sides, is worked
    out once and kept there. A finding that is not of this project has no repair.
    """
    try:
        path = steps(finding.target)
        repair = _REPAIRS.get((path[0], finding.rule))
        return None if repair is None else repair(data, path, finding, rate, {} if shared is None else shared)
    except RowSynthError:
        return None
