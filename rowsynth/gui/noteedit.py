"""What a hand does to the notes of one channel of one pattern: notes in, notes out, and nothing drawn. No window."""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace

from .. import songtext
from ..errors import RowSynthError
from ..notes import midi
from .songview import Note

# The rule every edit keeps. A channel plays one note at a time, because that is all its text can say. So the note
# that is placed, moved or stretched last gets the span it asks for. A note that began earlier and is still
# sounding where that span begins is cut short there, and a note that begins inside the span is removed. Nothing
# is left overlapping, out of order, before row 0 or past the rows of the pattern.
#
# A drum hit is put with `hit` and taken away with `unhit`, which give it a length its text can say. What the
# other edits make of the span, the pitch or the instrument of a hit, the text may refuse.

#: The notes of a channel in the order of their rows, and what an edit gives: such notes, and the indexes among
#: them of the notes that are picked after it.
Notes = tuple[Note, ...]
Picked = tuple[Notes, frozenset[int]]

# The rows that one length of the text says where a beat is four rows. A drum hit is written as one length.
_ONE = tuple(rows for rows, _, _ in songtext._sizes(4))


def _chosen(notes: Sequence[Note], picked: Iterable[int]) -> frozenset[int]:
    """The picked indexes. One that is the index of no note is refused."""
    chosen = frozenset(picked)
    for index in chosen:
        if index not in range(len(notes)):
            raise RowSynthError(f"{index!r} is picked, and that is none of the {len(notes)} notes")
    return chosen


def _level(level: int) -> int:
    """A level, once it is seen to be a whole number from 0 to 15."""
    if level not in range(16):
        raise RowSynthError(f"a level is a whole number from 0 to 15, not {level!r}")
    return int(level)


def _inside(note: Note, rows: float) -> Note | None:
    """What is left of a note that is cut off at row 0 and at the end of a pattern of `rows` rows. None for nothing."""
    front = max(0.0, -note.row)
    start = note.row + front
    length = min(note.rows - front, rows - start)
    return replace(note, row=start, rows=length) if length > 0 else None


def _cut(note: Note, starts: Iterable[float]) -> Note:
    """A note, cut short at the first of some rows that comes after its start, where it is still sounding there."""
    room = min((start - note.row for start in starts if start > note.row), default=note.rows)
    return replace(note, rows=room) if room < note.rows else note


def _lay(notes: Iterable[Note], fresh: Iterable[Note]) -> Picked:
    """Notes with fresh ones put among them by the rule: all of them in order, and the indexes of the fresh ones.

    A fresh note gets the span it asks for, but that it is cut short where the next fresh one begins.
    """
    fresh = sorted(fresh, key=lambda note: note.row)
    starts = [note.row for note in fresh]
    laid = [(_cut(note, starts), True) for note in fresh]
    spans = [(note.row, note.row + note.rows) for note, _ in laid]
    laid += [(_cut(note, starts), False) for note in notes
             if not any(start <= note.row < end for start, end in spans)]
    laid.sort(key=lambda found: found[0].row)
    return tuple(note for note, _ in laid), frozenset(index for index, (_, new) in enumerate(laid) if new)


def _each(notes: Sequence[Note], chosen: frozenset[int], change: Callable[[Note], Note]) -> Picked:
    """The notes with each chosen one changed, and the same ones picked."""
    return tuple(change(note) if index in chosen else note for index, note in enumerate(notes)), chosen


def _stopped(by: float, down: float, up: float) -> float:
    """A move of `by`, held to no less than `down` and no more than `up`. What is past one already is not moved back."""
    return min(max(by, min(down, 0)), max(up, 0))


def place(notes: Sequence[Note], note: Note, rows: float) -> Notes:
    """The notes with one more, in a pattern of `rows` rows, and the others as the rule leaves them.

    What the new note has before row 0 or past the end of the pattern is cut off. One with nothing left is refused.
    """
    cut = _inside(note, rows)
    if cut is None:
        raise RowSynthError(f"row {note.row:g}: a note of {note.rows:g} rows there has no part in the {rows:g} "
                            "rows of the pattern")
    return _lay(notes, [cut])[0]


def move(notes: Sequence[Note], picked: Iterable[int], by_rows: float, by_pitch: float, rows: float,
         low: float, high: float) -> Picked:
    """The picked notes moved by so many rows and so many semitones, and the others as the rule leaves them.

    The picked notes keep their shape and do not cut each other. They stop together at row 0, at the end of a
    pattern of `rows` rows, and at the pitches `low` and `high`.
    """
    chosen = _chosen(notes, picked)
    group = [note for index, note in enumerate(notes) if index in chosen]
    if not group:
        return tuple(notes), chosen
    by_rows = _stopped(by_rows, -min(note.row for note in group),
                       rows - max(note.row + note.rows for note in group))
    by_pitch = _stopped(by_pitch, low - min(note.pitch for note in group), high - max(note.pitch for note in group))
    return _lay((note for index, note in enumerate(notes) if index not in chosen),
                [replace(note, row=note.row + by_rows, pitch=note.pitch + by_pitch) for note in group])


def stretch(notes: Sequence[Note], picked: Iterable[int], by_rows: float, rows: float, least: float) -> Picked:
    """The picked notes made longer at their ends by so many rows, or shorter by a number under 0.

    None is made shorter than `least` rows, and none goes past the end of a pattern of `rows` rows. A picked note
    stops where the next picked one begins, and the others are as the rule leaves them. A least of no rows is
    refused.
    """
    if not least > 0:
        raise RowSynthError(f"the least a note is stretched to is more than 0 rows, not {least!r}")
    chosen = _chosen(notes, picked)
    return _lay((note for index, note in enumerate(notes) if index not in chosen),
                [replace(note, rows=min(max(note.rows + by_rows, min(least, note.rows)), rows - note.row))
                 for index, note in enumerate(notes) if index in chosen])


def remove(notes: Sequence[Note], picked: Iterable[int]) -> Picked:
    """The notes without the picked ones, and nothing picked."""
    chosen = _chosen(notes, picked)
    return tuple(note for index, note in enumerate(notes) if index not in chosen), frozenset()


def with_level(notes: Sequence[Note], picked: Iterable[int], level: int) -> Picked:
    """The picked notes at another level, which is a whole number from 0 to 15. Any other is refused."""
    level = _level(level)
    return _each(notes, _chosen(notes, picked), lambda note: replace(note, level=level))


def with_inst(notes: Sequence[Note], picked: Iterable[int], inst: str) -> Picked:
    """The picked notes played with another instrument."""
    return _each(notes, _chosen(notes, picked), lambda note: replace(note, inst=inst))


def shifted(notes: Sequence[Note], picked: Iterable[int], semis: int, low: float, high: float) -> Picked:
    """The picked notes so many semitones higher: an octave is 12.

    Nothing moves where one of them would be under the pitch `low` or over the pitch `high`.
    """
    chosen = _chosen(notes, picked)
    if any(not low <= notes[index].pitch + semis <= high for index in chosen):
        return tuple(notes), chosen
    return _each(notes, chosen, lambda note: replace(note, pitch=note.pitch + semis))


def boxed(notes: Sequence[Note], row: float, end: float, low: float, high: float) -> frozenset[int]:
    """The indexes of the notes that a box touches, which goes from a row to another and from a pitch to another.

    A note touches it that sounds between the two rows and is no lower than the lower pitch and no higher than
    the higher. One that ends where the box starts, or begins where it ends, does not. Either pair can be given
    either way round.
    """
    row, end = sorted((row, end))
    low, high = sorted((low, high))
    return frozenset(index for index, note in enumerate(notes)
                     if note.row < end and row < note.row + note.rows and low <= note.pitch <= high)


def copied(notes: Sequence[Note], picked: Iterable[int]) -> Notes:
    """The picked notes as a clip: the same notes in order, moved so that the first of them begins at row 0."""
    chosen = _chosen(notes, picked)
    group = [note for index, note in enumerate(notes) if index in chosen]
    start = min((note.row for note in group), default=0.0)
    return tuple(replace(note, row=note.row - start) for note in group)


def pasted(notes: Sequence[Note], clip: Iterable[Note], at_row: float, rows: float) -> Picked:
    """The notes with a clip put in at a row, in a pattern of `rows` rows, and the notes of the clip picked.

    What the clip has past the end of the pattern is cut off. The others are as the rule leaves them.
    """
    moved = (_inside(replace(note, row=note.row + at_row), rows) for note in clip)
    return _lay(notes, [note for note in moved if note is not None])


def snapped(row: float, grid: float) -> float:
    """The row at which the step of a grid begins that a row is in: with steps of 4 rows, 6.5 is in the one at 4.

    A grid whose steps have no rows is refused.
    """
    if not grid > 0:
        raise RowSynthError(f"a grid has steps of more than 0 rows, not {grid!r}")
    return float(math.floor(row / grid) * grid)


def fitted(notes: Iterable[Note], rows: float) -> bool:
    """Whether notes keep the rule in a pattern of `rows` rows.

    Each has rows of its own, none begins before row 0 or before the one before it has ended, and the last ends
    no later than the pattern does.
    """
    end = 0.0
    for note in notes:
        if note.row < end or not note.rows > 0:
            return False
        end = note.row + note.rows
    return end <= rows


def _letter(drums: dict | None, letter: str) -> dict:
    """What a drum letter plays, from the drum table of a song. A letter the table does not have is refused."""
    if letter not in (drums or {}):
        raise RowSynthError(f"no drum letter {letter!r}")
    return drums[letter]


def _held(room: float, own: float | None) -> float:
    """How long a hit is that has `room` rows to the next note or to the end, and whose letter has `own` rows.

    It is what one length of the text makes it. A letter with rows of its own holds the hit that long, where a
    length says as much without leaving the room. Any other hit is the longest that one length says in its room,
    and that is no longer than its letter's rows.
    """
    if own is not None and any(own <= rows <= room for rows in _ONE):
        return float(own)
    most = room if own is None else min(room, own)
    return max((rows for rows in _ONE if rows <= most), default=float(most))


def _refit(notes: Notes, index: int, drums: dict | None, rows: float) -> Notes:
    """The notes with the hit at an index as long as `_held` makes it, up to the note after it or to the end.

    Nothing changes where the index is before the first note, or the note there is no hit.
    """
    if index < 0 or not notes[index].drum:
        return notes
    note = notes[index]
    end = notes[index + 1].row if index + 1 < len(notes) else rows
    held = _held(end - note.row, _letter(drums, note.drum).get("rows"))
    return (*notes[:index], replace(note, rows=held), *notes[index + 1:])


def hit(notes: Sequence[Note], row: float, letter: str, level: int, drums: dict | None, rows: float) -> Notes:
    """The notes of a drum channel with a hit of a letter put at a row, in a pattern of `rows` rows.

    `drums` is the drum table of the song: the hit plays the instrument and the note its letter has there. A hit
    of any letter that begins at that row is replaced, and so is a note that does. The hit is as long as its text
    will make it: up to the next note or the end of the pattern, no longer than the rows its letter has of its
    own, and no longer than one length says. The hit before it is made that long again by the same rule, and a
    note before it that is no hit is cut short where it begins. Refused are a letter the table does not have, a
    row outside the pattern, and a level that is no whole number from 0 to 15.
    """
    own = _letter(drums, letter)
    if not 0 <= row < rows:
        raise RowSynthError(f"row {row:g}: a hit is put on one of the {rows:g} rows of the pattern")
    kept = [note for note in notes if note.row != row]
    room = min((note.row for note in kept if note.row > row), default=rows) - row
    new = Note(float(row), _held(room, own.get("rows")), midi(own["note"]), _level(level), own["inst"], letter)
    laid, (at,) = _lay(kept, [new])
    return _refit(laid, at - 1, drums, rows)


def unhit(notes: Sequence[Note], row: float, letter: str, drums: dict | None, rows: float) -> Notes:
    """The notes of a drum channel without the hit of a letter at a row. With no such hit they stay as they are.

    The hit before it is made as long again as `hit` makes one, now that it has the room. For that the drum
    table of the song and the rows of the pattern are given, as they are to `hit`.
    """
    found = [index for index, note in enumerate(notes) if note.row == row and note.drum == letter]
    if not found:
        return tuple(notes)
    kept = tuple(note for index, note in enumerate(notes) if index != found[0])
    return _refit(kept, found[0] - 1, drums, rows)
