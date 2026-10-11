"""The rule a song's tune is held to: a note that starts on a beat is one of the tones of the chord under it."""
from __future__ import annotations

import math
from collections.abc import Sequence

from ..errors import RowSynthError
from ..layers import bar_rows
from ..project import Project
from ..song import parse
from . import Finding, _at, _same_rows, _whole_rows

# The twelve notes of an octave from c up, as a finding spells them.
_NOTES = ("c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b")


def _read(song: dict, pattern: str, target: str) -> list[tuple[str, bool, list[dict], float]]:
    """A pattern's channels as they are read: the name of each, whether it plays drums, its events and its rows.

    Text that cannot be read is refused with the target and the channel in front.
    """
    read = []
    for channel, text in song["patterns"][pattern].items():
        settings = song["channels"][channel]
        drums = bool(settings.get("drums"))
        with _at(f"{target}.{channel}"):
            events, rows = parse(text, settings["inst"], song.get("rows_per_beat", 4),
                                 song.get("drums") if drums else None)
        read.append((channel, drums, events, rows))
    return read


def looked_at(drums: bool, rows: float, longest: float) -> bool:
    """Whether the notes of a channel are held to the chords: it plays no drums and is as long as its pattern.

    `rows` is the rows of the channel, and `longest` those of the longest channel of its pattern. Drums have no
    tune, and a channel that ends before its pattern does is not looked at.
    """
    return not drums and _same_rows(rows, _whole_rows(longest))


def under(chords: dict, written: Sequence[str], row: float, per_bar: float, where: str) -> tuple[int, str] | None:
    """The bar a row is in, counted from 0, and the name of the chord under the row. None in a bar with no chord.

    `chords` is the project's table of them, and `written` the strings a song has for the bars of a pattern,
    of `per_bar` rows each. A bar has no chord where it lies after the last string, and where its string names
    none: an empty one, or one of spaces only. Chords written for one bar have an even share of it each, in
    the order they are written in. `where` is the place of the strings: a string that names a chord the table
    does not have is refused with the place of the string in front.
    """
    bar = math.floor(row / per_bar)
    names = written[bar].split() if bar < len(written) else []
    if not names:
        return None
    chord = names[min(len(names) - 1, math.floor((row % per_bar) * len(names) / per_bar))]
    if chord not in chords:
        raise RowSynthError(f"{where}[{bar}]: no chord named {chord!r}")
    return bar, chord


def breach(chords: dict, written: Sequence[str], row: float, note: float, per_beat: float, per_bar: float,
           where: str) -> tuple[int, str, float] | None:
    """What a note that starts at a row breaks of the rule: its bar, the chord under it, and its pitch class.

    None for a note that breaks nothing. Only a note that starts on a beat is held to a chord, and only in a
    bar that has one written. Beats and bars begin on whole rows, of which a beat has `per_beat` and a bar
    `per_bar`. A row that triplets have left a hair off a whole number is taken for that number, so that a
    note has the beat that is heard. The rest is as `under` has it, and is refused as it refuses.
    """
    row = _whole_rows(row)
    if row % per_beat:
        return None
    held = under(chords, written, row, per_bar, where)
    if held is None:
        return None
    bar, chord = held
    tone = note % 12
    return None if tone in chords[chord]["tones"] else (bar, chord, tone)


def _breaches(chords: dict, name: str, song: dict, pattern: str) -> list[Finding]:
    """What one pattern of a song breaks of the rule, held to the chords the song has written for its bars.

    `chords` is the project's table of them. The count of the bars comes first, then the notes, a channel at a time.
    Beats and bars begin on whole rows. Rows that triplets have left a hair off a whole number are taken for
    that number, so that a pattern has the bars and a note the beat that are heard.
    """
    target = f"songs.{name}.patterns.{pattern}"
    written = song["chords"][pattern]
    per_beat = song.get("rows_per_beat", 4)
    per_bar = bar_rows(song)
    read = _read(song, pattern, target)
    # A pattern lasts as long as the longest of its channels.
    length = _whole_rows(max([0.0, *(rows for *_, rows in read)]))
    bars = length / per_bar
    found = []
    if len(written) != bars:
        found.append(Finding("chords", target, f"its length in bars is {bars:g}, and chords are written for "
                                               f"{len(written)}"))
    for channel, drums, events, rows in read:
        if not looked_at(drums, rows, length):
            continue
        for event in events:
            broken = breach(chords, written, event["row"], event["note"], per_beat, per_bar,
                            f"songs.{name}.chords.{pattern}")
            if broken is not None:
                bar, chord, tone = broken
                found.append(Finding("chords", target, f"channel {channel!r} starts a beat of bar {bar + 1} on "
                                                       f"{_NOTES[tone]}, which is not in the chord {chord}"))
    return found


def check(project: Project, rate: int) -> list[Finding]:
    """The findings on every pattern that a song of the project has chords for. Nothing is rendered, at any rate.

    A song's "chords" has one string for each bar of a pattern, a bar being four beats. A string names one chord
    or several with spaces between them. A pattern with more or fewer strings than bars gives one finding. So
    does every note that starts on a beat and is none of the tones of the chord under it, on each channel that
    is as long as its pattern and plays no drums. Notes in a bar with no chord are passed over: a bar after the
    last string, and one whose string names none, which is empty or has spaces only.

    The findings come in the order of the songs in the project, then of the patterns in "chords", then of the
    channels in the pattern, then of the notes in time. Text that cannot be read is refused with its song,
    pattern and channel in front. A string that names a chord the project does not have is refused with its
    place in "chords" in front, once a note is held to it.
    """
    found = []
    for name, song in project.songs.items():
        for pattern in song.get("chords", {}):
            found += _breaches(project.chords, name, song, pattern)
    return found
