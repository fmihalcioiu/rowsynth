"""A channel's events written back as the text of a song: what `parse` in `song` reads, the other way round."""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterator

from .errors import RowSynthError
from .notes import midi
from .song import _NAMES, _TOKEN, _rows, parse

# The lengths the text is written with: those that halve a whole note, and the triplets of those.
_LENGTHS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)

# The most lengths that are tied into one note or that fill one gap, once whole notes have brought it under two.
_MOST = 4

# The level a channel's text starts at, and the lowest and the highest octave `o` can write.
_LEVEL = 15
_OCTAVES = (0, 9)

# What an event has that the text must give back.
_KEYS = ("row", "rows", "note", "inst", "vel", "drum")

# A length as (rows, length, dots), and what takes time in the text as (name, length, tied to the next).
Size = tuple[float, int, int]
Timed = tuple[str, Size, bool]


def _refused(row: float, said: str) -> RowSynthError:
    """The error for what cannot be written, with the row it is at in front."""
    return RowSynthError(f"row {row:g}: {said}")


def _past(row: float, end: float) -> bool:
    """Whether a row is past another by more than the hair that the same lengths leave when added up another way."""
    return row - end > 1e-9 * max(1.0, abs(end))


def _sizes(rows_per_beat: int) -> list[Size]:
    """Every length the text is written with, the longest first.

    Each is there plain and with one dot, but for the whole note, so that none is longer than a bar. Of two
    that come to the same rows the plain one is kept.
    """
    found: dict[float, Size] = {}
    for dots in (0, 1):
        for length in _LENGTHS:
            if dots and length == 1:
                continue
            rows = _rows(length, dots, rows_per_beat)
            found.setdefault(rows, (rows, length, dots))
    return sorted(found.values(), reverse=True)


def _chains(start: float, end: float, sizes: list[Size]) -> Iterator[list[Size]]:
    """Every run of lengths that takes `start` to `end` when they are added to it one after the other, the best first.

    The sums are made as `parse` makes them and are held to `end` exactly. The best run has the fewest lengths.
    Of those it has the fewest dots, and of those it begins with the longest. To keep the search small, whatever
    is more than two whole notes off begins with whole notes, and no more than `_MOST` lengths come after them.
    """
    whole = sizes[0]
    lead = []
    while end - start > 2 * whole[0]:
        lead.append(whole)
        start += whole[0]

    def reach(at: float, first: int, left: int) -> Iterator[list[Size]]:
        """Every run of `left` lengths from `at` to the end, none of them longer than the one before it."""
        if not left:
            if at == end:
                yield []
            return
        for index in range(first, len(sizes)):
            rows = sizes[index][0]
            # Even this many of it fall short, and so do the shorter ones that follow.
            if _past(end, at + rows * left):
                break
            if not _past(at + rows, end):
                for rest in reach(at + rows, index, left - 1):
                    yield [sizes[index], *rest]

    for count in range(_MOST + 1):
        for found in sorted(reach(start, 0, count), key=lambda run: sum(dots for _, _, dots in run)):
            yield lead + found


def _chain(row: float, start: float, end: float, sizes: list[Size], fits: Callable | None = None) -> list[Size]:
    """The best run of lengths from `start` to `end`: the best that `fits`, or the best there is where none does.

    Where no run comes to it, the refusal names `row`.
    """
    best = None
    for found in _chains(start, end, sizes):
        if fits is None or fits(found):
            return found
        best = found if best is None else best
    if best is None:
        tried = " ".join(str(length) for length in _LENGTHS)
        raise _refused(row, f"{end - start:g} rows cannot be written: whole notes and then up to {_MOST} of the "
                            f"lengths {tried}, plain or dotted, do not come to exactly that")
    return best


def _lands(run: list[Size], start: float, end: float, sizes: list[Size]) -> bool:
    """Whether a note written at `start` as a run of tied lengths reads back as one note, with room for what follows.

    `parse` ties a length on when the note so far ends on the row the text has come to. Those are two sums of
    the same lengths, and with triplets they can differ by a hair. For the same reason the row after the note
    has to be one from which rests come to `end`, where the next thing begins.
    """
    at, rows = start, 0.0
    for size in run:
        if start + rows != at:
            return False
        rows += size[0]
        at += size[0]
    return next(_chains(at, end, sizes), None) is not None


def _bar(items: list[str | Timed], row: float, bar: float) -> None:
    """Adds a bar line where what is written next begins a bar that is not the first."""
    if row > 0 and not row % bar:
        items.append("|")


def _rest(items: list[str | Timed], row: float, end: float, sizes: list[Size]) -> float:
    """Adds the rests that take a channel from `row` to `end`, and gives the row it is then at."""
    for size in _chain(row, row, end, sizes):
        _bar(items, row, sizes[0][0])
        items.append(("r", size, False))
        row += size[0]
    return row


def _level(event: dict, now: int, own: float = 1) -> int:
    """The `v` an event was written at, once the level its drum letter has of its own is taken off.

    A letter whose own level is nothing silences every `v`, so the one in force, `now`, is as good as any.
    """
    asked = math.floor(event["vel"] / own * 15 + 0.5) if own else now
    if asked < 0 or asked / 15 * own != event["vel"]:
        raise _refused(event["row"], f"a level of {event['vel']:g} is no whole v, which counts in fifteenths"
                                     + (f" of the {own:g} its drum letter has" if own != 1 else ""))
    return asked


def _hit(event: dict, end: float, drums: dict | None, sizes: list[Size]) -> tuple[dict, Size]:
    """What the letter of a drum hit plays, and the length the hit is written with. `end` is where its room ends.

    `parse` holds a hit for the lesser of its written length and the rows of its letter. A hit that is shorter
    than its letter's rows, or whose letter has none, is written at its own length. One held for all of its
    letter's rows can be written longer: with the length that fills its room when there is one, or else with
    those rows, or else with the longest length between the two.
    """
    letter, start, rows = event["drum"], event["row"], event["rows"]
    hit = (drums or {}).get(letter)
    if not hit:
        raise _refused(start, f"no drum letter {letter!r} to write the hit with")
    most = hit.get("rows", rows)
    if event["note"] != midi(hit["note"]) or event["inst"] != hit["inst"] or rows > most:
        raise _refused(start, f"the hit is not what the drum letter {letter!r} plays")
    if rows < most or "rows" not in hit:
        fits = [size for size in sizes if size[0] == rows]
    else:
        fits = sorted((size for size in sizes if most <= size[0] and start + size[0] <= end),
                      key=lambda size: (start + size[0] != end, size[0] != most))
    if not fits:
        raise _refused(start, f"a hit of {rows:g} rows cannot be written: a hit is one length, and none fits it")
    return hit, fits[0]


def _octave(head: list[str], items: list[str | Timed], now: int | None, to: int) -> None:
    """Adds what moves the octave from `now` to `to`: a mark for one step, and `o` for more or to begin with.

    The `o` a text begins with goes to its head. An octave out of the reach of `o` is come to with marks, from
    the nearest that `o` can write or from where the octave is, whichever takes fewer.
    """
    near = min(_OCTAVES[1], max(_OCTAVES[0], to))
    if now is None or (abs(to - now) > 1 and abs(to - near) < abs(to - now)):
        (head if now is None else items).append(f"o{near}")
        now = near
    items.extend([">" if to > now else "<"] * abs(to - now))


def _word(item: str | Timed, default: int | None) -> str:
    """One item as it stands in the text. A plain length that is the default is left off."""
    if isinstance(item, str):
        return item
    name, (_, length, dots), tied = item
    return name + ("" if length == default and not dots else str(length)) + "." * dots + (" &" if tied else "")


def write(events: list[dict], rows: float, inst: str, rows_per_beat: int = 4, drums: dict | None = None) -> str:
    """Writes a channel's events as text that `parse` reads back as the same events, filling `rows` rows.

    The events are as `parse` gives them, with their rows counted from the start of the pattern, and `inst`,
    `rows_per_beat` and `drums` are what `parse` is to be given. Events out of order are put in order.

    The text is written as a person would write it. It begins with the octave of its first note, and after
    that `>` and `<` move it a step and `o` moves it further. `l` follows with the plain length that comes most
    often, where that makes the text shorter, and those notes carry no length. `v` and `@name` stand only where
    the level and the instrument change. A note that no one length makes is tied from the lengths `_chains`
    puts first, and a gap is rests found the same way. A note between two letters is spelt as a flat. A bar
    line stands wherever something begins on one, and one space is between any two things.

    Refused, each with its row: an event before row 0, one of no rows, one that is still sounding where the
    next begins or where the rows end, rows that no lengths come to, a level that is no whole `v`, a note that
    is no whole number, an instrument the text cannot name, and a drum hit its letter does not play or cannot
    time. Text that would not read back as the events is refused as well, so none is ever given.
    """
    if not rows_per_beat > 0:
        raise RowSynthError(f"rows_per_beat: must be more than 0, not {rows_per_beat!r}")
    sizes = _sizes(rows_per_beat)
    bar = sizes[0][0]
    ordered = sorted(events, key=lambda event: event["row"])
    head: list[str] = []
    items: list[str | Timed] = []
    row = 0.0
    octave = None
    level = _LEVEL
    playing = inst
    for index, event in enumerate(ordered):
        start, length = event["row"], event["rows"]
        last = index + 1 == len(ordered)
        end = rows if last else ordered[index + 1]["row"]
        if start < 0:
            raise _refused(start, "an event cannot begin before row 0")
        if not length > 0:
            raise _refused(start, f"an event of {length:g} rows cannot be written")
        if _past(start + length, end):
            raise _refused(start, f"the event is {length:g} rows long, and " + (
                f"the text is to fill {rows:g} rows" if last else
                f"the next begins at row {end:g}: a channel plays one note at a time"))
        row = _rest(items, row, start, sizes)
        _bar(items, row, bar)
        if "drum" in event:
            hit, size = _hit(event, end, drums, sizes)
            name, chain, to = event["drum"], [size], None
            asked = _level(event, level, hit.get("vel", 1))
        else:
            note = event["note"]
            if note != math.floor(note):
                raise _refused(start, f"a note of {note:g} is between two notes, and the text has none there")
            name, to = _NAMES[int(note) % 12], int(note) // 12 - 1
            chain = _chain(start, 0.0, length, sizes, lambda run: _lands(run, start, end, sizes))
            asked = _level(event, level)
            if event["inst"] != playing:
                playing = event["inst"]
                found = _TOKEN.fullmatch(f"@{playing}")
                if not found or found["inst"] != playing:
                    raise _refused(start, f"an instrument called {playing!r} cannot be named in the text")
                items.append(f"@{playing}")
        if asked != level:
            level = asked
            items.append(f"v{level}")
        if to is not None:
            _octave(head, items, octave, to)
            octave = to
        for count, size in enumerate(chain):
            if count:
                _bar(items, row, bar)
            items.append((name, size, count + 1 < len(chain)))
            row += size[0]
    _rest(items, row, rows, sizes)
    plain = Counter(item[1][1] for item in items if not isinstance(item, str) and not item[1][2])
    default = None
    if plain:
        common, count = max(plain.items(), key=lambda found: (found[1], -found[0]))
        digits = len(str(common))
        # A length written once at the front has to save more than it costs.
        if count * digits > digits + 2:
            default = common
            head.append(f"l{common}")
    text = " ".join(head + [_word(item, default) for item in items])
    # The one thing the text is held to, whatever went on above.
    read, filled = parse(text, inst, rows_per_beat, drums)
    given = [tuple(event.get(key) for key in _KEYS) for event in ordered]
    back = [tuple(event.get(key) for key in _KEYS) for event in read]
    if (back, filled) != (given, rows):
        differ = next((index for index, (one, other) in enumerate(zip(given, back)) if one != other),
                      min(len(given), len(back)))
        raise _refused(given[differ][0] if differ < len(given) else rows,
                       "the text written from here on does not read back as the events it was written for")
    return text
