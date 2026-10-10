"""A song: the small language of its channels, its patterns laid end to end, and it played as a mix, stems or a loop."""
from __future__ import annotations

import math
import re
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

from .errors import RowSynthError
from .note import DEFAULT_RATE, TICK_HZ, _RELEASE, _count, _flag, render_note
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
    An event made from a drum letter has one key more, "drum", which is that letter.
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
                           "inst": hit["inst"], "vel": vel * hit.get("vel", 1), "drum": found["drum"]})
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

# The gain of a song that does not say.
_GAIN = 0.35

# The seconds the two sides run on for after the last row, for what is still sounding there.
_TAIL = 0.5


def _held(song: dict, inst: dict, event: dict) -> int:
    """The ticks a note is held for: the share of its rows that its instrument's gate keeps, and one at the least."""
    return max(1, math.floor(event["rows"] * song["row_ticks"] * inst.get("gate", _GATE) + 0.5))


def _row_samples(song: dict, rate: int) -> float:
    """The samples a row of a song takes, which need not be a whole number of them."""
    return song["row_ticks"] * rate / TICK_HZ


def _notes(song: dict, events: list[dict], echo: dict | None, rate: int) -> list[tuple]:
    """What a channel plays, in the order it plays it: each note as (sample, instrument, note, ticks, velocity).

    A note starts at the sample nearest to its row. With an echo, every note is followed by itself again,
    `rows` later and at `gain` of its velocity.
    """
    row_samples = _row_samples(song, rate)
    notes = []
    for event in events:
        inst = song["instruments"][event["inst"]]
        ticks = _held(song, inst, event)
        starts = [(event["row"], event["vel"])]
        if echo:
            # This echo is no delay line. The note itself is played a second time.
            starts.append((event["row"] + echo["rows"], event["vel"] * echo["gain"]))
        for row, velocity in starts:
            notes.append((math.floor(row * row_samples + 0.5), inst, event["note"], ticks, velocity))
    return notes


def _play(notes: list[tuple], size: int, rate: int) -> np.ndarray:
    """A buffer of `size` samples with those notes added into it. Whatever falls past its end is dropped."""
    out = np.zeros(size, dtype=np.float32)
    for at, inst, note, ticks, velocity in notes:
        render_note(out, at, inst, note, ticks, velocity, rate)
    return out


def pan_gains(pan: float) -> tuple[float, float]:
    """The gains a pan gives the left side and the right: -1 is all left, 1 all right, and 0 has each 3 dB down.

    They are the cosine and the sine of one angle, so their squares add up to 1, and a channel is as loud
    wherever its pan puts it.
    """
    angle = (pan + 1) * math.pi / 4
    return math.cos(angle), math.sin(angle)


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
    row_samples = _row_samples(song, rate)
    total = math.ceil(rows * row_samples + rate * _TAIL)
    left = np.zeros(total, dtype=np.float32)
    right = np.zeros(total, dtype=np.float32)
    for name, events in laid.items():
        if only is not None and name not in only:
            continue
        channel = song["channels"][name]
        mono = _play(_notes(song, events, None if dry else channel.get("echo"), rate), total, rate)
        level = channel.get("gain", 1)
        to_left, to_right = pan_gains(channel.get("pan", 0))
        # The sums are worked out in doubles, and only what a side then holds is rounded to float32.
        wide = mono.astype(np.float64)
        left += wide * (level * to_left)
        right += wide * (level * to_right)
    gain = song.get("gain", _GAIN)
    return {"left": finish(left, gain, rate), "right": finish(right, gain, rate), "rows": rows,
            "seconds": rows * row_samples / rate}


def render_stems(song: dict, rate: int = DEFAULT_RATE) -> dict:
    """A song a channel at a time: {"stems": [{"name", "pan", "samples"}], "rows", "loop_samples", "seconds"}.

    A stem is one channel in mono, `loop_samples` long, made to be played round and round. What the channel is
    still sounding when the rows end, a release or an echo, is added back on at the start of its stem, which
    is where it is heard once the loop has come round. For the same reason a stem leaves through `finish`
    with no fade at its end. It has the channel's echo, and its level is the channel's gain times the song's,
    0.35 if the song has none. It is not panned: the channel's pan comes back beside it, 0 if it has none, for
    whoever mixes the stems. They come in the order `lay` gives the channels in. "seconds" is how long the
    loop lasts. A song with no rows has nothing to loop, and is refused.
    """
    _count(rate, "rate", 1)
    laid, rows = lay(song)
    # The rows are turned into ticks first and into samples after. Multiplied by `_row_samples` instead, the last
    # digit can differ, and a length that sits on a half would round the other way.
    loop_samples = math.floor(rows * song["row_ticks"] * rate / TICK_HZ + 0.5)
    if not loop_samples:
        if rows:
            raise RowSynthError(f"the song's rows come to no samples at a rate of {rate}, so there is nothing to loop")
        raise RowSynthError("the song has no rows, so there is nothing to loop")
    tick_samples = rate / TICK_HZ
    gain = song.get("gain", _GAIN)
    stems = []
    for name, events in laid.items():
        channel = song["channels"][name]
        notes = _notes(song, events, channel.get("echo"), rate)
        # The channel is played into a buffer that runs on past the loop, as far as its last release does.
        ends = [at + math.ceil((ticks + inst.get("release", _RELEASE)) * tick_samples)
                for at, inst, _, ticks, _ in notes]
        wide = _play(notes, max([loop_samples, *ends]), rate).astype(np.float64)
        level = channel.get("gain", 1) * gain
        samples = np.zeros(loop_samples, dtype=np.float32)
        # The buffer goes into the loop a lap at a time, and a tail longer than the loop goes round more than
        # once. Each lap is added in doubles, and what the loop then holds is rounded to float32 before the next.
        for start in range(0, len(wide), loop_samples):
            lap = wide[start:start + loop_samples]
            samples[:len(lap)] += lap * level
        stems.append({"name": name, "pan": channel.get("pan", 0), "samples": finish(samples, 1, rate, fade=False)})
    return {"stems": stems, "rows": rows, "loop_samples": loop_samples, "seconds": loop_samples / rate}


def _mix_stems(stems: list[dict], size: int) -> tuple[np.ndarray, np.ndarray]:
    """Stems mixed to a left side and a right one of `size` samples, each stem going where its pan puts it.

    A stem that is longer goes in as far as `size` and is cut there. The stems are added one at a time in the
    order given. Each sum is worked out in doubles, and what a side then holds is rounded to float32 before the
    next stem goes in.
    """
    left = np.zeros(size, dtype=np.float32)
    right = np.zeros(size, dtype=np.float32)
    for stem in stems:
        to_left, to_right = pan_gains(stem["pan"])
        wide = stem["samples"][:size].astype(np.float64)
        left += wide * to_left
        right += wide * to_right
    return left, right


def render_loop(song: dict, rate: int = DEFAULT_RATE) -> dict:
    """A song as one loop in stereo: {"left", "right", "rows", "loop_samples", "seconds"}.

    The two sides are the song's stems mixed: each stem is shared between them by its channel's pan, and the
    stems are added in the order `render_stems` gives them in. A side is `loop_samples` long and made to be
    played round and round, as a stem is: it opens on what the song is still sounding when its rows end, and its
    end is not faded. "rows" and "seconds" are those of the stems. Each stem has been through the output stage
    on its own and their sum has been through nothing, so the sides of a loud song go past full scale, where
    those of `render_song` are bent over to stay under it. A song with no rows has nothing to loop, and is
    refused.
    """
    played = render_stems(song, rate)
    left, right = _mix_stems(played["stems"], played["loop_samples"])
    return {"left": left, "right": right, "rows": played["rows"], "loop_samples": played["loop_samples"],
            "seconds": played["seconds"]}
