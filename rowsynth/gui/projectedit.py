"""The things of a project added, renamed, copied and removed: its data in, a list of changes out. No window."""
from __future__ import annotations

import copy
import re
from collections.abc import Iterator

from ..edit import NOTHING
from ..errors import RowSynthError
from ..project import _AT, _NOT_IN_FILE_NAMES, _sounds_in
from ..song import _TOKEN, _unread
from .model import TEST_NOTE, TEST_TICKS, _section, inst_place

# The data of a project goes in, as `Project.to_dict` gives it, and a list of changes comes out, which
# `Session.change_many` makes as one step. The data handed in is never changed, and what comes back shares
# nothing with it. A thing is a song, a sound or an instrument, and `kind` says which, in that word.
#
# A change is to a whole thing, or to the whole table of a kind: a thing keeps its place among the others when
# its name changes, and only the table can say so. An instrument of a song's own is put in and taken out at its
# place in the song. What cannot be done is refused, with the reason. After any other, a project that loaded before
# still does, once all the changes are made: no one of them alone need leave it so.

#: One change: a place as `edit.change_file_many` reads it, and the value put there, or NOTHING for what is
#: there taken out.
Change = tuple[str, object]

#: How many of its uses a refusal names before it counts the rest.
NAMED = 3

#: The smallest instrument that is heard, the channel and the pattern a new song has, the ticks its rows last
#: and the text of its one bar, written as the editor writes it.
INSTRUMENT = {"wave": "pulse"}
CHANNEL = "lead"
PATTERN = "a"
ROW_TICKS = 6
TEXT = "o4 c4 r2."


def _fm(level: float, attack: float, decay: float, sustain: float, release: float, mul: int = 1,
        det: float | None = None) -> dict:
    """One operator of an FM instrument as the file holds it: how loud it is, its envelope, and its pitch."""
    pitch = {"mul": mul} if det is None else {"mul": mul, "det": det}
    return {**pitch, "level": level, "env": {"a": attack, "d": decay, "s": sustain, "r": release}}


#: The ready instruments a new one can be started from, by name, in the order they are offered: each as the
#: example projects hold it. The first six are the arcade example's, the last three the trailer piece's.
STARTERS: dict[str, dict] = {
    "pluck": {"wave": "pulse", "duty": 1, "vol": [12, 10, 8, 6, 5, 4, 3], "release": 2},
    "bass": {"wave": "tri", "vol": [15, 15, 14, 13, 12], "release": 2, "gate": 0.8},
    "kick": {"wave": "tri", "vol": [15, 14, 12, 9, 6, 3], "pitch": [0, -6, -5, -4, -3, -2], "release": 1},
    "snare": {"wave": "noise", "vol": [14, 11, 8, 6, 4, 2, 1], "release": 2, "lp": 7000},
    "hat": {"wave": "noise", "vol": [8, 4, 1], "release": 1},
    "chime": {"wave": "pulse", "duty": 2, "vol": [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], "release": 3},
    "fmbrass": {"wave": "fm", "alg": 4, "fb": 0.25,
                "ops": [_fm(2, 0.03, 0.3, 0.6, 0.12), _fm(0.8, 0.01, 0.3, 0.85, 0.12),
                        _fm(1.2, 0.02, 0.2, 0.5, 0.12), _fm(0.5, 0.01, 0.3, 0.85, 0.12, det=1.2)],
                "vol": [15], "vib": {"delay": 20, "depth": 0.12, "rate": 6}, "release": 6, "gate": 0.9, "lp": 2800,
                "key_scale": -4},
    "fmpad": {"wave": "fm", "alg": 5,
              "ops": [_fm(1, 0.3, 1, 0.8, 0.6), _fm(0.5, 0.25, 0.5, 1, 0.6), _fm(0.22, 0.35, 0.5, 1, 0.6, 2, 1),
                      _fm(0.5, 0.3, 0.5, 1, 0.6, det=-1.3)],
              "vol": [15], "release": 40, "gate": 0.97, "lp": 1800, "key_scale": -4},
    "sub": {"wave": "fm", "alg": 4,
            "ops": [_fm(0.5, 0, 0.08, 0.1, 0.03), _fm(1, 0, 0.2, 0.7, 0.05), {"mul": 1, "level": 0},
                    _fm(0.3, 0, 0.2, 0.6, 0.05, 2)],
            "vol": [15], "release": 2, "gate": 0.55, "lp": 500},
}

# What a new name cannot hold: what the project refuses in a name that files are called by, as a song's and a
# sound's are, and what a place in a project is spelt with.
_NOT_IN_NAMES = _NOT_IN_FILE_NAMES | frozenset(".[]")

# A name as what it starts with and the number it ends on, which may be none.
_COUNTED = re.compile(r"(.*?)(\d*)", re.DOTALL)

# A name the text of a channel can switch to with `@`.
_WORD = re.compile(r"\w+", re.ASCII)


def _a(kind: str) -> str:
    """A kind with the word that goes before it: "a song", "an instrument"."""
    return f"{'an' if kind[0] in 'aeiou' else 'a'} {kind}"


def _table(data: dict, kind: str) -> dict:
    """The things of one kind that a project holds, by name: none where its data has no table of them."""
    return data.get(_section(kind), {})


def _known(data: dict, kind: str, name: object) -> dict:
    """The thing of a kind that has a name, as the project holds it. One the project does not have is refused."""
    table = _table(data, kind)
    if not isinstance(name, str) or name not in table:
        raise RowSynthError(f"the project has no {kind} called {name!r}")
    return table[name]


def _own(data: dict, song: object, name: object) -> dict:
    """The instrument of a song's own that has a name, as the project holds it.

    A song the project does not have is refused, and so is a name the song has no instrument of its own under:
    one it only gives to an instrument of the project is none.
    """
    table = _known(data, "song", song).get("instruments", {})
    if not isinstance(name, str) or not isinstance(table.get(name), dict):
        raise RowSynthError(f"the song {song!r} has no instrument of its own called {name!r}")
    return table[name]


def _put(data: dict, kind: str, name: str, thing: dict) -> Change:
    """The change that puts a thing in the table of its kind, after the others: the table itself where there is none."""
    section = _section(kind)
    return (f"{section}.{name}", thing) if section in data else (section, {name: thing})


def allowed(data: dict, kind: str, name: object, song: str | None = None) -> str:
    """A new name for a song, a sound or an instrument, once it is seen to be one that can be given.

    A name is text with something in it. It holds nothing that the project refuses in a name files are called
    by, and no dot or square bracket, which the places of a project are spelt with. No other thing of its kind
    has it. Any other name is refused, and the refusal says which of these it broke. Every function here that
    gives a thing a name holds it to this.

    With `song` the name is for an instrument of that song's own. It is held to the same, but for who may have
    it already: no instrument of the song's own has it, and no name the song gives to one of the project's.
    """
    table = _table(data, kind) if song is None else _known(data, "song", song).get("instruments", {})
    if not isinstance(name, str) or not name:
        raise RowSynthError(f"{_a(kind)} needs a name, and {name!r} is none")
    for char in name:
        if char in _NOT_IN_NAMES or ord(char) < 32:
            raise RowSynthError(f"{_a(kind)} cannot be called {name!r}: a name cannot hold {char!r}")
    if name in table:
        owner = "the project" if song is None else f"the song {song!r}"
        raise RowSynthError(f"{owner} has {_a(kind)} called {name!r} already")
    return name


def free(data: dict, kind: str, like: str | None = None) -> str:
    """A name that no thing of a kind has yet, for a new one.

    With `like` it is the next free one after that name: `lead` gives `lead2`, or `lead3` where there is a
    `lead2`, and `pad2` gives `pad3`. With none it is the kind itself, as in `song`, and where that is taken the
    next free one after it.
    """
    taken = _table(data, kind)
    if like is None and kind not in taken:
        return kind
    stem, count = _COUNTED.fullmatch(kind if like is None else like).groups()
    number = int(count or 1) + 1
    while f"{stem}{number}" in taken:
        number += 1
    return f"{stem}{number}"


def fresh(data: dict, kind: str, name: str) -> str:
    """The name for a new thing of a kind that is to be called `name`: that one, or where it is taken the next free."""
    return free(data, kind, name) if name in _table(data, kind) else name


def added(data: dict, kind: str, name: str | None = None, like: str | None = None) -> list[Change]:
    """The changes that give a project a new song, sound or instrument, the last of its kind.

    Its name is `name`, held to what `allowed` allows, and with none given the one `free` finds. With `like`
    it is a copy of the thing of its kind that has that name, which the project must have. With none it is the
    smallest thing of its kind that the project accepts and that can be heard: an instrument of one plain
    wave, a sound of one voice that plays one note, a song of one channel and one pattern of a bar with one
    note in it. The sound and the song play the first instrument of the project. Where the project has none, a
    new one comes along with them, and its change is the first.
    """
    name = allowed(data, kind, free(data, kind, like) if name is None else name)
    if like is not None:
        return [_put(data, kind, name, copy.deepcopy(_known(data, kind, like)))]
    if kind == "instrument":
        return [_put(data, kind, name, dict(INSTRUMENT))]
    changes = []
    inst = next(iter(_table(data, "instrument")), None)
    if inst is None:
        inst = free(data, "instrument")
        changes.append(_put(data, "instrument", inst, dict(INSTRUMENT)))
    if kind == "sound":
        thing = {"def": {"voices": [{"inst": inst, "notes": [[0, TEST_NOTE, TEST_TICKS]]}]}}
    else:
        thing = {"row_ticks": ROW_TICKS, "channels": {CHANNEL: {"inst": inst}},
                 "patterns": {PATTERN: {CHANNEL: TEXT}}, "order": [PATTERN]}
    return [*changes, _put(data, kind, name, thing)]


def started(data: dict, starter: str, name: str | None = None) -> list[Change]:
    """The change that gives a project a new instrument that is a copy of one of `STARTERS`, the last of its kind.

    Its name is `name`, held to what `allowed` allows. With none given it is the name of the starter, and
    where the project has an instrument of that name the next free one after it, as `fresh` finds it: `bass2`.
    A starter there is none of is refused.
    """
    if starter not in STARTERS:
        raise RowSynthError(f"there is no ready instrument called {starter!r}")
    name = allowed(data, "instrument", fresh(data, "instrument", starter) if name is None else name)
    return [_put(data, "instrument", name, copy.deepcopy(STARTERS[starter]))]


def _users(data: dict) -> Iterator[tuple[str, str, dict]]:
    """Every song and every sound of a project, the songs first: its table, its name and what it holds."""
    for section in ("songs", "sounds"):
        for title, thing in data.get(section, {}).items():
            yield section, title, thing


def _named(section: str, title: str, thing: dict, name: str) -> Iterator[tuple[str, dict, str, bool]]:
    """Where a song or a sound names an instrument of the project, each as (words, holder, key, text).

    What is at `key` of `holder` is the name itself, or with `text` the text of a channel that switches to it
    somewhere. The words say where that is, for a person to find it. A name the song gives to the instrument
    stands for it. A song with an instrument of its own under the name plays that one, and names the
    project's nowhere else.
    """
    if section == "sounds":
        for defn in _sounds_in(thing):
            for voice in defn["voices"]:
                if voice["inst"] == name:
                    yield f"the sound {title}", voice, "inst", False
        return
    own = thing.get("instruments", {})
    for key, inst in own.items():
        if inst == name:
            yield f"the instrument {key} of {title}", own, key, False
    if name in own:
        return
    for channel, held in thing["channels"].items():
        if held["inst"] == name:
            yield f"the channel {channel} of {title}", held, "inst", False
    for letter, drum in thing.get("drums", {}).items():
        if drum["inst"] == name:
            yield f"the drum {letter} of {title}", drum, "inst", False
    for pattern, texts in thing["patterns"].items():
        for channel, text in texts.items():
            if name in _AT.findall(text):
                yield f"the channel {channel} in the pattern {pattern} of {title}", texts, channel, True


def uses(data: dict, kind: str, name: str, song: str | None = None) -> list[str]:
    """Where a thing of a project is named by another, each place once, in words a person can find it by.

    Only an instrument is named: by a channel of a song, by one of its drums, by a text that switches to it
    with `@`, by a name the song gives it, and by a voice of a sound. That reads as "the channel lead of tune",
    "the drum k of tune", "the channel lead in the pattern a of tune", "the instrument soft of tune" and "the
    sound zap". The songs come first, and the sounds after them. Nothing names a song or a sound, so those
    have no uses. A thing the project does not have is refused.

    With `song` the thing is an instrument of that song's own. Only the song names it: by its channels, its
    drums and its texts, as it would name one of the project's if it had none of its own.
    """
    if song is not None:
        _own(data, song, name)
        bare = {**data["songs"][song], "instruments": {}}
        return list(dict.fromkeys(words for words, _, _, _ in _named("songs", song, bare, name)))
    _known(data, kind, name)
    if kind != "instrument":
        return []
    return list(dict.fromkeys(words for section, title, thing in _users(data)
                              for words, _, _, _ in _named(section, title, thing, name)))


def removed(data: dict, kind: str, name: str, song: str | None = None) -> list[Change]:
    """The change that takes a thing out of a project, or with `song` an instrument of its own out of that song.

    An instrument that is still named is refused, and the refusal says where, as `uses` words it: `NAMED` of
    the places, and how many more there are. A thing the project does not have is refused.
    """
    found = uses(data, kind, name, song)
    if found:
        more = len(found) - NAMED
        raise RowSynthError(f"{kind} {name!r} is still used by " + ", ".join(found[:NAMED])
                            + (f" and {more} more" if more > 0 else ""))
    return [(f"{_section(kind)}.{name}" if song is None else inst_place(song, name), NOTHING)]


def shared(data: dict, song: str, name: str) -> list[Change]:
    """The changes that make an instrument of a song's own one of the project's, under the name it has.

    The project gets it as the last of its instruments, and the song has it of its own no longer: so the song
    goes on playing it by that name. Where it was all the song's table of instruments held, the table goes with
    it, so that no empty one is left in the song. The name is held to what `allowed` allows, so a project that
    has an instrument of that name already is refused. So is a song with no instrument of its own under the name.
    """
    inst = copy.deepcopy(_own(data, song, name))
    alone = len(data["songs"][song]["instruments"]) == 1
    gone = f"songs.{song}.instruments" if alone else inst_place(song, name)
    return [_put(data, "instrument", allowed(data, "instrument", name), inst), (gone, NOTHING)]


def own_copied(data: dict, song: str, name: str) -> list[Change]:
    """The change that gives a song an instrument of the project as one of its own, under the name it has.

    The song then plays its own copy wherever it played the project's by that name, and the project's is as
    it was for every other song and sound. A song with no instruments of its own gets the table with it.
    Refused are an instrument and a song the project does not have, and a song that has an instrument of its
    own of that name already, or gives the name to one of the project's.
    """
    inst = copy.deepcopy(_known(data, "instrument", name))
    allowed(data, "instrument", name, song)
    if "instruments" in data["songs"][song]:
        return [(inst_place(song, name), inst)]
    return [(f"songs.{song}.instruments", {name: inst})]


def _switched(text: str, old: str, new: str, words: str) -> str:
    """The text of a channel with every switch to one instrument made a switch to another, and all else as it was.

    The text is read as the engine reads it, a token at a time, so `@lead` is not taken for the start of
    `@lead2`, and nothing but the name is another. `words` say where the text is. A text the
    engine cannot read is refused, and so is a new name the text cannot write after `@`.
    """
    if not _WORD.fullmatch(new):
        raise RowSynthError(f"{words} switches to the instrument as @{old}, and a name written so holds only "
                            f"letters, digits and _, which {new!r} does not")
    out, at = [], 0
    while at < len(text):
        found = _TOKEN.match(text, at)
        if not found:
            raise RowSynthError(f"{words}: {_unread(text, at)}")
        out.append(f"@{new}" if found["inst"] == old else found[0])
        at = found.end()
    return "".join(out)


def renamed(data: dict, kind: str, old: str, new: str) -> list[Change]:
    """The changes that give a thing of a project another name, everywhere the project names it.

    The first is the table of its kind, in which the thing keeps its place. A song or a sound is named nowhere
    else. An instrument is named where `uses` finds it, and each song and each sound that names it is one more
    change, the whole of it with the new name in every such place: its channels, its drums, the voices of a
    sound, a name the song gives the instrument, and every `@` of the name in the text of a channel, changed
    as `_switched` changes it.

    The new name is held to what `allowed` allows, and a thing the project does not have is refused. So is an
    instrument whose new name a song that plays it has for an instrument of its own, since the song would
    then play that one, and whatever `_switched` refuses.
    """
    _known(data, kind, old)
    allowed(data, kind, new)
    section = _section(kind)
    changes: list[Change] = [(section, {new if name == old else name: copy.deepcopy(thing)
                                        for name, thing in data[section].items()})]
    if kind != "instrument":
        return changes
    for other, title, thing in _users(data):
        made = copy.deepcopy(thing)
        places = list(_named(other, title, made, old))
        own = made.get("instruments", {}) if other == "songs" else {}
        if new in own and any(holder is not own for _, holder, _, _ in places):
            raise RowSynthError(f"the song {title!r} has an instrument of its own called {new!r}, and would play "
                                "that one in place of this")
        for words, holder, key, text in places:
            holder[key] = _switched(holder[key], old, new, words) if text else new
        if places:
            changes.append((f"{other}.{title}", made))
    return changes


def own_renamed(data: dict, song: str, old: str, new: str) -> list[Change]:
    """The change that gives an instrument of a song's own another name, in that song and nowhere else.

    It is the whole song. The instrument keeps its place in the song's own table, and has the new name in
    each channel and each drum of the song that plays it, and at every `@` of the old one in the text of a
    channel, changed as `_switched` changes it. No other song and no sound can name it, so nothing else is
    changed.

    The new name is held to what `allowed` allows for an instrument of the song's own. Refused as well are a
    song with no instrument of its own under the old name, a new name by which the song plays an instrument
    of the project, since it would then play this one in its place, and whatever `_switched` refuses.
    """
    _own(data, song, old)
    allowed(data, "instrument", new, song)
    made = copy.deepcopy(data["songs"][song])
    # The song with no instruments of its own names both by its channels, its drums and its texts alone.
    bare = {**made, "instruments": {}}
    taken = next(_named("songs", song, bare, new), None)
    if taken is not None:
        raise RowSynthError(f"{taken[0]} plays the instrument {new!r} of the project, and would play this one "
                            "in place of that")
    for words, holder, key, text in list(_named("songs", song, bare, old)):
        holder[key] = _switched(holder[key], old, new, words) if text else new
    made["instruments"] = {new if name == old else name: inst for name, inst in made["instruments"].items()}
    return [(f"songs.{song}", made)]
