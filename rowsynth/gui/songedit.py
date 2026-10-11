"""A song changed as a whole: its order, patterns, channels, key and tempo. A song in, a new song out. No window."""
from __future__ import annotations

import copy
import re
from collections.abc import Iterable
from dataclasses import replace

from ..errors import RowSynthError
from ..note import TICK_HZ
from ..project import _AT, _NOT_IN_FILE_NAMES
from ..song import _within, up
from . import songview
from .noteedit import _held
from .songview import Note

# A song goes in and a new song comes out. The song is as the project holds it, `project.songs[name]`, and what
# comes back is saved whole in its place, so that one change is one step to take back. The song handed in is
# never changed, and what comes back shares nothing with it.
#
# A change that cannot be made is refused, with the reason. After any other, a song that loaded and played before
# still does: every name in it points at something, no pattern that had rows is left with none, and no text is
# left that cannot be read. The one thing that cannot be seen from a song is whether the project has the
# instrument that a new channel is given.
#
# A place is an index into the order of the song, counted from 0.

# What a new name cannot hold: what the project refuses in a name that files are called by, as a channel's is,
# and what a place in a project is spelt with. `edit.steps` reads a dot and a bracket as the end of a name, so
# the text of a channel or a pattern with one in its name could not be saved on its own.
_NOT_IN_NAMES = _NOT_IN_FILE_NAMES | frozenset(".[]")

# A name as what it starts with and the number it ends on, which may be none.
_COUNTED = re.compile(r"(.*?)(\d*)", re.DOTALL)

# The lowest note and the highest that the text of a channel can write: the c of octave 0 and the b of octave 9.
_LOWEST, _HIGHEST = 12, 131

# The ticks in a minute, and the slowest and the fastest tempo that `tempos` lists, in beats a minute.
_MINUTE = 60 * TICK_HZ
_SLOWEST, _FASTEST = 40, 300


def allowed(song: dict, kind: str, name: object) -> str:
    """A new name for a pattern or a channel of a song, once it is seen to be one that can be given.

    `kind` is "pattern" or "channel". A name is text with something in it. It holds nothing that the project
    refuses in a name files are called by, and no dot or square bracket, which the places of a project are spelt
    with. No other pattern or channel of the song has it. Any other name is refused, and the refusal says which
    of these it broke. Every function here that takes a new name holds it to this.
    """
    if not isinstance(name, str) or not name:
        raise RowSynthError(f"a {kind} needs a name, and {name!r} is none")
    for char in name:
        if char in _NOT_IN_NAMES or ord(char) < 32:
            raise RowSynthError(f"a {kind} cannot be called {name!r}: a name cannot hold {char!r}")
    if name in song[f"{kind}s"]:
        raise RowSynthError(f"the song has a {kind} called {name!r} already")
    return name


def _known(song: dict, kind: str, name: object) -> None:
    """Refuse a pattern or a channel that the song does not have. `kind` is "pattern" or "channel"."""
    if not isinstance(name, str) or name not in song[f"{kind}s"]:
        raise RowSynthError(f"the song has no {kind} called {name!r}")


def _whole(value: object, what: str, least: int | None = None) -> int:
    """A whole number, `least` or more where one is given. Any other is refused, in words that `what` begins."""
    if isinstance(value, bool) or not isinstance(value, int) or (least is not None and value < least):
        most = "" if least is None else f", {least} or more"
        raise RowSynthError(f"{what} a whole number{most}, not {value!r}")
    return value


def _place(song: dict, place: object) -> int:
    """A place in the order of a song, once it is seen to be one. Any other is refused."""
    count = len(song["order"])
    if isinstance(place, bool) or not isinstance(place, int) or not 0 <= place < count:
        raise RowSynthError(f"{place!r} is no place in the order, which has {count}, counted from 0")
    return place


def _rekeyed(table: dict, old: str, new: str) -> dict:
    """A table with one of its keys under another name, in the place it was in."""
    return {new if key == old else key: value for key, value in table.items()}


def _swapped(names: Iterable[str], old: str, new: str) -> list[str]:
    """A list of names with one of them under another name wherever it stands."""
    return [new if name == old else name for name in names]


def _next(song: dict, name: str) -> str:
    """The next free name after a pattern's: `a` gives `a2`, or `a3` where the song has an `a2`, and `tune2` `tune3`."""
    stem, count = _COUNTED.fullmatch(name).groups()
    number = int(count or 1) + 1
    while f"{stem}{number}" in song["patterns"]:
        number += 1
    return f"{stem}{number}"


def _readable(song: dict) -> dict:
    """A song with an instrument under every name it plays, which `songview.pattern` wants to find in it.

    A song as the project holds it has only the instruments of its own, and the rest are the project's. Nothing
    of an instrument is read here but that it is there.
    """
    named = {settings["inst"] for settings in song["channels"].values()}
    named.update(hit["inst"] for hit in song.get("drums", {}).values())
    named.update(name for texts in song["patterns"].values() for text in texts.values() for name in _AT.findall(text))
    return {**song, "instruments": {**dict.fromkeys(named, {}), **song.get("instruments", {})}}


def _rests(song: dict, rows: float) -> dict:
    """What a pattern with no notes holds to be `rows` rows long: rests for the first channel of the song.

    A pattern is as long as the longest of its channels, so one of them has to count its rows out. A song with
    no channel is refused.
    """
    if not song["channels"]:
        raise RowSynthError("the song has no channel, and a pattern needs one to count its rows out in rests")
    first = next(iter(song["channels"]))
    return {first: songview.text((), rows, first, song)}


def _cut(notes: Iterable[Note], rows: float, song: dict) -> list[Note]:
    """The notes of a channel as a pattern of `rows` rows has room for them.

    A note that begins at the end or after it goes, and one that crosses the end is cut short there. A drum hit
    is written as one length, so one that crosses the end is cut to the longest length that its room has.
    """
    kept = []
    for note in notes:
        room = rows - note.row
        if room <= 0:
            continue
        if note.rows > room:
            if note.drum:
                room = _held(room, song["drums"][note.drum].get("rows"), song.get("rows_per_beat", 4))
            note = replace(note, rows=room)
        kept.append(note)
    return kept


def moved(song: dict, place: int, to: int) -> dict:
    """A song with the block at one place of its order moved to another, which is the place it is at afterwards."""
    made = copy.deepcopy(song)
    order = made["order"]
    order.insert(_place(song, to), order.pop(_place(song, place)))
    return made


def repeated(song: dict, place: int) -> dict:
    """A song that plays the pattern at a place of its order once more, right after it."""
    at = _place(song, place)
    made = copy.deepcopy(song)
    made["order"].insert(at + 1, made["order"][at])
    return made


def copied(song: dict, place: int, name: str | None = None) -> dict:
    """A song with a new pattern that is a copy of the one at a place of its order, played right after it.

    The copy has the texts of the pattern and the chords written for its bars. Its name is `name`, held to what
    `allowed` allows. With none given it is the next free one: `a` gives `a2`, or `a3` where the song has an
    `a2`, and `tune2` gives `tune3`.
    """
    old = song["order"][_place(song, place)]
    name = allowed(song, "pattern", _next(song, old) if name is None else name)
    made = copy.deepcopy(song)
    made["patterns"][name] = dict(made["patterns"][old])
    if old in made.get("chords", {}):
        made["chords"][name] = list(made["chords"][old])
    made["order"].insert(place + 1, name)
    return made


def added(song: dict, after: int, bars: int, name: str | None = None) -> dict:
    """A song with a new pattern of `bars` bars and no notes, played right after a place of its order.

    The pattern holds rests for the first channel of the song, which is what makes it that long. Its name is
    `name`, held to what `allowed` allows. With none given it is the next free one after the name of the pattern
    it is put after, found as `copied` finds one. Bars that are no whole number, or fewer than 1, are refused,
    and so is a song with no channel.
    """
    at = _place(song, after)
    name = allowed(song, "pattern", _next(song, song["order"][at]) if name is None else name)
    rows = float(_whole(bars, "the bars of a pattern are", 1) * songview.bar_rows(song))
    made = copy.deepcopy(song)
    made["patterns"][name] = _rests(song, rows)
    made["order"].insert(at + 1, name)
    return made


def dropped(song: dict, place: int) -> dict:
    """A song without one place of its order.

    The pattern that was played there goes with it, and so do the chords written for its bars, when no place of
    the order plays it any more and the intro does not either. Nothing else in a song names a pattern. The last
    place of an order is not dropped: a song with none has nothing to play.
    """
    made = copy.deepcopy(song)
    order = made["order"]
    name = order[_place(song, place)]
    if len(order) == 1:
        raise RowSynthError(f"pattern {name!r} is all the order has, and a song with none has nothing to play")
    del order[place]
    if name not in order and name not in made.get("intro", {}).get("order", []):
        del made["patterns"][name]
        made.get("chords", {}).pop(name, None)
    return made


def renamed(song: dict, old: str, new: str) -> dict:
    """A song with one of its patterns under another name, everywhere the song names it.

    Those are its patterns, its order, the chords written for its bars and the order of its intro. The pattern
    keeps its place in each. The new name is held to what `allowed` allows, and a pattern the song does not have
    is refused.
    """
    _known(song, "pattern", old)
    allowed(song, "pattern", new)
    made = copy.deepcopy(song)
    made["patterns"] = _rekeyed(made["patterns"], old, new)
    made["order"] = _swapped(made["order"], old, new)
    if "chords" in made:
        made["chords"] = _rekeyed(made["chords"], old, new)
    if "intro" in made:
        made["intro"]["order"] = _swapped(made["intro"]["order"], old, new)
    return made


def resized(song: dict, name: str, bars: int) -> dict:
    """A song with one of its patterns made `bars` bars long, where it was longer or shorter.

    Every channel the pattern has text for is cut off at the new end or filled with rests up to it. A note that
    crosses the end is cut short there, and one that begins at it or after it goes. A drum hit that crosses it is
    cut to the longest length its text can say in the room it has left. The texts are read with
    `songview.pattern` and written with `songview.text`, as the editor writes them, and a channel that is as long
    as that already keeps its text as it is. A pattern with no text at all is given rests for the first channel
    of the song.

    The chords written for the bars of the pattern are cut off as well. Where bars are added, each has the chord
    of the last bar that was written, so that the notes there are held to a chord as those before them are.

    Refused are a pattern the song does not have, bars that are no whole number or fewer than 1, a pattern with
    a note that is off the grid of rows, and whatever `songview.text` refuses to write.
    """
    _known(song, "pattern", name)
    rows = float(_whole(bars, "the bars of a pattern are", 1) * songview.bar_rows(song))
    found = songview.pattern(_readable(song), name)
    if not songview.on_grid(found):
        raise RowSynthError(f"pattern {name!r} has notes off the grid of rows, and such a pattern is not made "
                            "longer or shorter")
    made = copy.deepcopy(song)
    texts = made["patterns"][name]
    for channel in found.channels:
        if channel.written and channel.rows != rows:
            with _within(name, channel.name):
                texts[channel.name] = songview.text(_cut(channel.notes, rows, song), rows, channel.name, song)
    if not texts:
        texts.update(_rests(song, rows))
    chords = made.get("chords", {})
    if name in chords:
        chords[name] = chords[name][:bars] + chords[name][-1:] * (bars - len(chords[name]))
    return made


def channel_added(song: dict, name: str, inst: str, drums: bool = False) -> dict:
    """A song with one more channel, the last of them, which plays `inst` and has no text in any pattern.

    `drums` makes it a drum channel. The name is held to what `allowed` allows. Whether the project has the
    instrument cannot be seen from a song, and the project's check says so when the song is saved. In a song
    with layers the new channel is in no group yet.
    """
    made = copy.deepcopy(song)
    made["channels"][allowed(song, "channel", name)] = {"inst": inst, **({"drums": True} if drums else {})}
    return made


def channel_dropped(song: dict, name: str) -> dict:
    """A song without one of its channels: its settings, its text in every pattern, and its place in the layers.

    Those are all the places a song names a channel in. It is refused where something cannot do without the
    channel, and the refusal says what: a pattern that has text for no other channel, which would be left with
    no rows, and a group of the layers that has no other channel, which would be left empty. A channel the song
    does not have is refused.
    """
    _known(song, "channel", name)
    made = copy.deepcopy(song)
    del made["channels"][name]
    for pattern, texts in made["patterns"].items():
        if list(texts) == [name]:
            raise RowSynthError(f"channel {name!r} has the only text of pattern {pattern!r}, which would be left "
                                "with no rows")
        texts.pop(name, None)
    for group, members in made.get("layers", {}).get("groups", {}).items():
        if name in members:
            members[:] = [member for member in members if member != name]
            if not members:
                raise RowSynthError(f"channel {name!r} is all that group {group!r} of the layers has, and the "
                                    "group would be left empty")
    return made


def channel_renamed(song: dict, old: str, new: str) -> dict:
    """A song with one of its channels under another name, everywhere the song names it.

    Those are its channels, the texts of its patterns and the groups of its layers. The channel keeps its place
    in each, so the song is played as it was. The new name is held to what `allowed` allows, and a channel the
    song does not have is refused.
    """
    _known(song, "channel", old)
    allowed(song, "channel", new)
    made = copy.deepcopy(song)
    made["channels"] = _rekeyed(made["channels"], old, new)
    made["patterns"] = {pattern: _rekeyed(texts, old, new) for pattern, texts in made["patterns"].items()}
    for members in made.get("layers", {}).get("groups", {}).values():
        members[:] = _swapped(members, old, new)
    return made


def channel_moved(song: dict, name: str, to: int) -> dict:
    """A song with one of its channels at another place among them, counted from 0: the place it is at afterwards.

    That is the order the channels are listed in, and it changes nothing that is played. The engine adds the
    channels up in the order in which the texts of the patterns first come to them, which stays as it was, so
    the samples are the same ones. A channel the song does not have is refused, and so is a place that is none.
    """
    _known(song, "channel", name)
    names = [channel for channel in song["channels"] if channel != name]
    if isinstance(to, bool) or not isinstance(to, int) or not 0 <= to <= len(names):
        raise RowSynthError(f"{to!r} is no place among the channels, of which there are {len(names) + 1}, "
                            "counted from 0")
    names.insert(to, name)
    made = copy.deepcopy(song)
    made["channels"] = {channel: made["channels"][channel] for channel in names}
    return made


def transposed(song: dict, semis: int) -> dict:
    """A song with its tuned channels `semis` semitones higher, or lower by a number under 0.

    It is `rowsynth.song.up` that moves it, and what that refuses is refused. Drum channels keep their text, and
    every moved note is spelt with its octave in front. The chords written for the bars stay as they are: they
    are named from the project's table, which a song cannot change. Semitones that are no whole number are
    refused, and so is a move that takes a note out of octaves 0 to 9, which the text could not write.
    """
    _whole(semis, "the semitones a song is moved by are")
    readable = _readable(song)
    for name in song["patterns"]:
        for channel in songview.pattern(readable, name).channels:
            if not channel.drums and any(not _LOWEST <= note.pitch + semis <= _HIGHEST for note in channel.notes):
                raise RowSynthError(f"pattern {name!r}, channel {channel.name!r}: {semis} semitones up takes a "
                                    "note out of octaves 0 to 9, and the text can write no other")
    return copy.deepcopy(up(song, semis))


def tempos(rows_per_beat: float) -> list[tuple[int, float]]:
    """The tempos the engine can do from 40 to 300 beats a minute, the fastest first, each as (row_ticks, beats).

    A row lasts a whole number of ticks, sixty to the second, and a beat is `rows_per_beat` rows. So a minute
    has 3600 ticks, and the beats in it are 3600 over the ticks in a beat: with four rows to a beat, 900 over
    the ticks in a row. A beat of no rows is refused.
    """
    if isinstance(rows_per_beat, bool) or not isinstance(rows_per_beat, (int, float)) or not rows_per_beat > 0:
        raise RowSynthError(f"a beat has more than 0 rows, not {rows_per_beat!r}")
    found = []
    row_ticks = 1
    while (beats := _MINUTE / (rows_per_beat * row_ticks)) >= _SLOWEST:
        if beats <= _FASTEST:
            found.append((row_ticks, beats))
        row_ticks += 1
    return found


def with_tempo(song: dict, row_ticks: int) -> dict:
    """A song whose rows last `row_ticks` ticks each: a whole number of them, 1 or more. Any other is refused."""
    made = copy.deepcopy(song)
    made["row_ticks"] = _whole(row_ticks, "the ticks of a row are", 1)
    return made
