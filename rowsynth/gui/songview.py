"""A song as the song view shows it: notes as bars on a grid of rows, a channel and a pattern at a time. No window."""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from .. import layers
from ..checks import _whole_rows
from ..errors import RowSynthError
from ..notes import midi
from ..song import _channels, _within, parse

# The number of middle c, and how far to each side of it the octave goes that is shown where no note is.
_MIDDLE = 60
_HALF = 6


@dataclass(frozen=True)
class Note:
    """One note of a channel in a pattern. A drum hit is a note as well.

    row is where it starts in its pattern and rows how long it is, both counted in rows as `parse` gives them.
    pitch is its note number, 60 for middle c. level is the `v` it was written with, 0 to 15: what a drum letter
    has for a level of its own is not in it. inst is the instrument it plays with, which is not its channel's
    after a switch. drum is the letter of a drum hit, and "" for any other note.
    """

    row: float
    rows: float
    pitch: float
    level: int
    inst: str
    drum: str = ""


@dataclass(frozen=True)
class Channel:
    """One channel of a song as one pattern has it.

    inst is the channel's own instrument, and drums whether it is a drum channel. notes are in the order they are
    written in. rows is what the channel's text fills, its rests with it. written is whether the pattern has text
    for the channel at all: one it has none for has no notes and no rows.
    """

    name: str
    inst: str
    drums: bool
    notes: tuple[Note, ...]
    rows: float
    written: bool


@dataclass(frozen=True)
class Pattern:
    """One pattern of a song: its rows, which are those of its longest channel, and every channel of the song."""

    name: str
    rows: float
    channels: tuple[Channel, ...]


@dataclass(frozen=True)
class Block:
    """One place in the order of a song: the pattern played there, the row of the song it starts at, and its rows."""

    name: str
    row: float
    rows: float


@dataclass(frozen=True)
class DrumRow:
    """One drum letter of a song: the instrument it plays, and the number of the note it plays on it."""

    letter: str
    inst: str
    pitch: float


def pattern(song: dict, name: str) -> Pattern:
    """One pattern of a song, read as `lay` reads it. The song is as `Project.song` gives it, its instruments with it.

    The channels are all those of the song, in the song's order, and the ones the pattern has no text for are
    among them. A pattern the song does not have is refused. So is what `lay` refuses in a pattern, in the same
    words: text for a channel the song does not have, text that cannot be read, and a note on an instrument the
    song does not have.
    """
    if name not in song["patterns"]:
        raise RowSynthError(f"no pattern called {name!r}")
    rows_per_beat = song.get("rows_per_beat", 4)
    instruments = song.get("instruments", {})
    # A letter's own level is kept from `parse`, so that a hit comes back at the level its text wrote.
    letters = {letter: {key: value for key, value in hit.items() if key != "vel"}
               for letter, hit in (song.get("drums") or {}).items()}
    read: dict[str, tuple[tuple[Note, ...], float]] = {}
    for channel, settings, text in _channels(song, name):
        with _within(name, channel):
            events, rows = parse(text, settings["inst"], rows_per_beat, letters if settings.get("drums") else None)
            for found in events:
                if found["inst"] not in instruments:
                    raise RowSynthError(f"no instrument called {found['inst']!r}")
        read[channel] = (tuple(Note(found["row"], found["rows"], found["note"], round(found["vel"] * 15),
                                    found["inst"], found.get("drum", "")) for found in events), rows)
    channels = []
    for channel, settings in song["channels"].items():
        notes, rows = read.get(channel, ((), 0.0))
        channels.append(Channel(channel, settings["inst"], bool(settings.get("drums")), notes, rows, channel in read))
    return Pattern(name, max([0.0, *(rows for _, rows in read.values())]), tuple(channels))


def blocks(song: dict) -> tuple[list[Block], float]:
    """The order of a song as blocks, and the rows of the whole song.

    Every place in the order is a block of its own, so a pattern that comes twice is two blocks. A block starts
    where the one before it ends. A name in the order that is no pattern is refused as `lay` refuses it, and so
    is whatever `pattern` refuses.
    """
    lengths: dict[str, float] = {}
    found = []
    start = 0.0
    for name in song["order"]:
        if name not in song["patterns"]:
            raise RowSynthError(f"order: no pattern called {name!r}")
        if name not in lengths:
            lengths[name] = pattern(song, name).rows
        found.append(Block(name, start, lengths[name]))
        start += lengths[name]
    return found, start


def _whole(rows: float) -> bool:
    """Whether a place counted in rows is a whole row, but for the hair that adding up triplets leaves."""
    return float(_whole_rows(rows)).is_integer()


def on_grid(pattern: Pattern) -> bool:
    """Whether every note of a pattern starts on a whole row and ends on one."""
    return all(_whole(note.row) and _whole(note.row + note.rows)
               for channel in pattern.channels for note in channel.notes)


def span(notes: Iterable[Note], least: int = 12) -> tuple[int, int]:
    """The lowest and the highest note number a piano roll has to show for some notes.

    Both are whole, and they are at least `least` apart. Where the notes take up less than that, the room left
    over is shared between the two ends, so the notes are in the middle. With no notes it is the octave round
    middle c.
    """
    pitches = [note.pitch for note in notes]
    low = math.floor(min(pitches)) if pitches else _MIDDLE - _HALF
    high = math.ceil(max(pitches)) if pitches else _MIDDLE + _HALF
    low -= max(0, least - (high - low)) // 2
    return low, max(high, low + least)


def drum_rows(song: dict) -> list[DrumRow]:
    """The drum letters of a song in the order its file has them, each with the instrument and the note it plays."""
    return [DrumRow(letter, hit["inst"], midi(hit["note"])) for letter, hit in (song.get("drums") or {}).items()]


def bar_rows(song: dict) -> int:
    """The rows in a bar of a song, counted as the chord check counts them: four of its beats."""
    return layers.bar_rows(song)
