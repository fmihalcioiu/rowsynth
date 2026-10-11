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
# its name changes, and only the table can say so. An instrument of a song's own is taken out at its place in
# the song. What cannot be done is refused, with the reason. After any other, a project that loaded before
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


def allowed(data: dict, kind: str, name: object) -> str:
    """A new name for a song, a sound or an instrument, once it is seen to be one that can be given.

    A name is text with something in it. It holds nothing that the project refuses in a name files are called
    by, and no dot or square bracket, which the places of a project are spelt with. No other thing of its kind
    has it. Any other name is refused, and the refusal says which of these it broke. Every function here that
    gives a thing a name holds it to this.
    """
    table = _table(data, kind)
    if not isinstance(name, str) or not name:
        raise RowSynthError(f"{_a(kind)} needs a name, and {name!r} is none")
    for char in name:
        if char in _NOT_IN_NAMES or ord(char) < 32:
            raise RowSynthError(f"{_a(kind)} cannot be called {name!r}: a name cannot hold {char!r}")
    if name in table:
        raise RowSynthError(f"the project has {_a(kind)} called {name!r} already")
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
    goes on playing it by that name. The name is held to what `allowed` allows, so a project that has an
    instrument of that name already is refused. So is a song with no instrument of its own under the name.
    """
    inst = copy.deepcopy(_own(data, song, name))
    return [_put(data, "instrument", allowed(data, "instrument", name), inst), (inst_place(song, name), NOTHING)]


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
