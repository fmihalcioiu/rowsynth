"""A song: the small language its channels are written in, its patterns laid end to end, and the whole of it played."""
from __future__ import annotations

import math
import re
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

from .errors import RowSynthError
from .note import DEFAULT_RATE, TICK_HZ, _count, _flag, render_note
from .notes import midi
from .sound import finish

# Everything a channel's text can hold, one kind to a line.
_TOKEN = re.compile("|".join((
    r"\s+",                                                           # space, which only keeps things apart
    r"\|",                                                            # a bar line, which changes nothing
    r"@(?P<inst>\w+)",                                                # the instrument from here on: @lead
    r"(?P<name>[a-gr])(?P<sign>[+#-]?)(?P<length>\d*)(?P<dots>\.*)",  # a note, or r for a rest: c f+8 b-4. r2
    r"(?P<drum>[kshHt])(?P<drum_length>\d*)(?P<drum_dots>\.*)",       # a drum hit, timed as a note is: k s8 H4.
    r"l(?P<default>\d+)",                                             # the length for notes written without one: l8
    r"o(?P<octave>\d)",                                               # the octave: o4
    r"v(?P<vol>\d+)",                                                 # the volume from here on, 0 to 15: v12
    r"(?P<mark>[<>&])",                                               # an octave down, an octave up, a tie
)), re.ASCII)

# Semitones from c up to each letter, and what a sharp or a flat adds.
_STEP = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
_SIGN = {"": 0, "+": 1, "#": 1, "-": -1}


def _unread(text: str, at: int) -> RowSynthError:
    """The error for text that cannot be read from `at` on, quoting its next twelve characters."""
    return RowSynthError(f'cannot read "{text[at:at + 12]}" in a channel')


def _number(octave: int, name: str, sign: str) -> int:
    """The number of a note: the c of octave 4 is 60."""
    return 12 * (octave + 1) + _STEP[name] + _SIGN[sign]


def _rows(length: int, dots: int, rows_per_beat: int) -> float:
    """How many rows a length comes to: 4 is a quarter note, and each dot adds half of what the one before added."""
    rows = rows_per_beat * 4 / length
    add = rows
    for _ in range(dots):
        add /= 2
        rows += add
    return rows


def parse(text: str, inst: str, rows_per_beat: int = 4, drums: dict | None = None) -> tuple[list[dict], float]:
    """Reads a channel's text. Gives its events, each {"row", "rows", "note", "inst", "vel"}, and the rows it fills.

    `drums` is what each drum letter plays: {letter: {"inst", "note", "rows", "vel"}}, the last two optional.
    """
    events = []
    row = 0.0
    octave = 4
    default = 16
    vel = 1.0
    tie = False
    at = 0
    while at < len(text):
        found = _TOKEN.match(text, at)
        if not found:
            raise _unread(text, at)
        if found["inst"]:
            inst = found["inst"]
        elif found["name"]:
            length = int(found["length"] or default)
            if not length:
                raise _unread(text, at)
            rows = _rows(length, len(found["dots"]), rows_per_beat)
            if found["name"] != "r":
                note = _number(octave, found["name"], found["sign"])
                # A tie makes the note before longer, when this is the same note and starts where that one ends.
                last = events[-1] if events else None
                if tie and last and last["note"] == note and last["row"] + last["rows"] == row:
                    last["rows"] += rows
                else:
                    events.append({"row": row, "rows": rows, "note": note, "inst": inst, "vel": vel})
            tie = False
            row += rows
        elif found["drum"]:
            hit = drums.get(found["drum"]) if drums else None
            if not hit:
                raise RowSynthError(f'"{found["drum"]}" is a drum, and this is not a drum channel')
            length = int(found["drum_length"] or default)
            if not length:
                raise _unread(text, at)
            rows = _rows(length, len(found["drum_dots"]), rows_per_beat)
            events.append({"row": row, "rows": min(rows, hit.get("rows", rows)), "note": midi(hit["note"]),
                           "inst": hit["inst"], "vel": vel * hit.get("vel", 1)})
            row += rows
        elif found["default"]:
            default = int(found["default"])
            if not default:
                raise _unread(text, at)
        elif found["octave"]:
            octave = int(found["octave"])
        elif found["vol"]:
            vel = int(found["vol"]) / 15
        elif found["mark"] == ">":
            octave += 1
        elif found["mark"] == "<":
            octave -= 1
        elif found["mark"] == "&":
            tie = True
        at = found.end()
    return events, row


# The twelve notes of an octave from c up, as moved text spells them.
_NAMES = ("c", "d-", "d", "e-", "e", "f", "g-", "g", "a-", "a", "b-", "b")


def transpose(text: str, semis: int) -> str:
    """Rewrites a channel's text `semis` semitones higher, spelling each note with its octave in front."""
    out = []
    octave = 4
    at = 0
    while at < len(text):
        found = _TOKEN.match(text, at)
        if not found or found["drum"]:
            raise _unread(text, at)
        if found["name"] and found["name"] != "r":
            note = _number(octave, found["name"], found["sign"]) + semis
            out.append(f"o{note // 12 - 1} {_NAMES[note % 12]}{found['length']}{found['dots']}")
        elif found["octave"]:
            octave = int(found["octave"])
        elif found["mark"] == ">":
            octave += 1
        elif found["mark"] == "<":
            octave -= 1
        elif not found[0].isspace():
            out.append(found[0])
        at = found.end()
    return " ".join(out)


def _channels(song: dict, name: str) -> Iterator[tuple[str, dict, str]]:
    """What a pattern holds: for each channel it has text for, the channel's name, its settings and that text."""
    channels = song["channels"]
    for channel, text in song["patterns"][name].items():
        if channel not in channels:
            raise RowSynthError(f"pattern {name!r}: no channel called {channel!r}")
        yield channel, channels[channel], text


@contextmanager
def _within(pattern: str, channel: str) -> Iterator[None]:
    """Has whatever is refused inside it say which of a song's many texts it was refused in."""
    try:
        yield
    except RowSynthError as error:
        raise RowSynthError(f"pattern {pattern!r}, channel {channel!r}: {error}") from None


def up(song: dict, semis: int) -> dict:
    """A copy of a song with its tuned channels `semis` semitones higher. Drum channels keep their text."""
    patterns = {}
    for name in song["patterns"]:
        moved = {}
        for channel, settings, text in _channels(song, name):
            if settings.get("drums"):
                moved[channel] = text
            else:
                with _within(name, channel):
                    moved[channel] = transpose(text, semis)
        patterns[name] = moved
    return {**song, "patterns": patterns}


def lay(song: dict) -> tuple[dict[str, list[dict]], float]:
    """A song's patterns laid end to end in its order. Gives each channel's events, and the rows the song fills.

    An event's row is counted from the start of the song. A pattern lasts as long as the longest of its
    channels, and the next one starts where it ends. A channel is left out when no pattern in the order has
    text for it.
    """
    rows_per_beat = song.get("rows_per_beat", 4)
    instruments = song.get("instruments", {})
    laid: dict[str, list[dict]] = {}
    start = 0.0
    for name in song["order"]:
        if name not in song["patterns"]:
            raise RowSynthError(f"order: no pattern called {name!r}")
        longest = 0.0
        for channel, settings, text in _channels(song, name):
            with _within(name, channel):
                events, rows = parse(text, settings["inst"], rows_per_beat,
                                     song.get("drums") if settings.get("drums") else None)
                for found in events:
                    if found["inst"] not in instruments:
                        raise RowSynthError(f"no instrument called {found['inst']!r}")
                    found["row"] += start
            longest = max(longest, rows)
            laid.setdefault(channel, []).extend(events)
        start += longest
    return laid, start


# The share of its rows a note is held for, when its instrument does not say.
_GATE = 0.9

# The gain a song leaves through, when it does not say.
_GAIN = 0.35

# The seconds the two sides run on for after the last row, for what is still sounding there.
_TAIL = 0.5


def _held(song: dict, inst: dict, event: dict) -> int:
    """The ticks a note is held for: the share of its rows that its instrument's gate keeps, and one at the least."""
    return max(1, math.floor(event["rows"] * song["row_ticks"] * inst.get("gate", _GATE) + 0.5))


def render_song(song: dict, only: list[str] | None = None, dry: bool = False, rate: int = DEFAULT_RATE) -> dict:
    """A song as samples: {"left", "right", "rows", "seconds"}, the two sides and how long its rows take.

    Every channel plays its notes into a buffer of its own, which goes to both sides at the channel's gain. Its
    pan shares it between them: -1 is all left, 1 all right, and at 0, where a pan left out puts it, each side
    has it 3 dB down. A channel with an echo plays every note again `rows` later at `gain` of its level, and
    dry=True leaves those out. `only` names the channels to play, and the rest stay silent. The two sides
    leave through `finish` at the song's gain, 0.35 if it has none. They run half a second past the last row
    for what is still sounding there, which "seconds" does not count.
    """
    _flag(dry, "dry")
    _count(rate, "rate", 1)
    laid, rows = lay(song)
    row_samples = song["row_ticks"] * rate / TICK_HZ
    total = math.ceil(rows * row_samples + rate * _TAIL)
    left = np.zeros(total, dtype=np.float32)
    right = np.zeros(total, dtype=np.float32)
    for name, events in laid.items():
        if only is not None and name not in only:
            continue
        channel = song["channels"][name]
        echo = None if dry else channel.get("echo")
        mono = np.zeros(total, dtype=np.float32)
        for event in events:
            inst = song["instruments"][event["inst"]]
            ticks = _held(song, inst, event)
            render_note(mono, math.floor(event["row"] * row_samples + 0.5), inst, event["note"], ticks,
                        event["vel"], rate)
            if echo:
                # This echo is no delay line. The note itself is played a second time.
                render_note(mono, math.floor((event["row"] + echo["rows"]) * row_samples + 0.5), inst,
                            event["note"], ticks, event["vel"] * echo["gain"], rate)
        level = channel.get("gain", 1)
        angle = (channel.get("pan", 0) + 1) * math.pi / 4
        # The sums are worked out in doubles, and only what a side then holds is rounded to float32.
        wide = mono.astype(np.float64)
        left += wide * (level * math.cos(angle))
        right += wide * (level * math.sin(angle))
    gain = song.get("gain", _GAIN)
    return {"left": finish(left, gain, rate), "right": finish(right, gain, rate), "rows": rows,
            "seconds": rows * row_samples / rate}
