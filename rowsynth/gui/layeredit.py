"""The layers and the intro of a song changed: groups, versions, keys, and what plays before the loop. No window."""
from __future__ import annotations

import copy
import math
from collections.abc import Collection

from ..errors import RowSynthError
from ..layers import _FADE, _TAIL, bar_rows
from ..note import TICK_HZ
from . import songedit
from .songedit import _NOT_IN_NAMES, _rekeyed, _swapped

# A song goes in and a new song comes out, as in `songedit`: the song is as the project holds it, the one handed
# in is never changed, and what comes back shares nothing with it and passes the project's check. A change that
# cannot be made is refused, with the reason.
#
# The project's check wants the groups to name channels of the song and the versions to name groups, and no
# more. The checks and the build want more of layers that are finished: every channel in one group and in it
# once, no group without channels or in no version. Here a channel is in one group or in none, and a group may
# be empty and a version too, since that is how they are while they are being made. The checks say what is
# still missing.

#: What the one group and the one version are called that a song gets as its layers are switched on.
GROUP = "all"
VERSION = "full"
#: What a new group and a new version are called, with a number after it where the name is taken.
NEW = {"group": "group", "version": "version"}


def in_name(char: str) -> bool:
    """Whether the name of a group or of a version can hold a character."""
    return char not in _NOT_IN_NAMES and ord(char) >= 32


def allowed(song: dict, kind: str, name: object) -> str:
    """A new name for a group or a version of a song's layers, once it is seen to be one that can be given.

    `kind` is "group" or "version". Files are called by both, so a name holds nothing that the project refuses
    in such a name, and no dot or square bracket, which the places of a project are spelt with. It is text
    with something in it, and no other group or version of the song has it.
    """
    if not isinstance(name, str) or not name:
        raise RowSynthError(f"a {kind} needs a name, and {name!r} is none")
    for char in name:
        if not in_name(char):
            raise RowSynthError(f"a {kind} cannot be called {name!r}: a name cannot hold {char!r}")
    if name in _layers(song)[f"{kind}s"]:
        raise RowSynthError(f"the layers have a {kind} called {name!r} already")
    return name


def bar_seconds(song: dict) -> float:
    """How long a bar of a song lasts, in seconds: the longest the tail of its intro can be."""
    return bar_rows(song) * song["row_ticks"] / TICK_HZ


def _layers(song: dict) -> dict:
    """The layers of a song. A song that has none is refused."""
    if "layers" not in song:
        raise RowSynthError("the song has no layers")
    return song["layers"]


def _intro(song: dict) -> dict:
    """The intro of a song. A song that has none is refused."""
    if "intro" not in song:
        raise RowSynthError("the song has no intro")
    return song["intro"]


def _known(song: dict, kind: str, name: object) -> None:
    """Refuse a group or a version that the layers of a song do not have. `kind` is "group" or "version"."""
    if not isinstance(name, str) or name not in _layers(song)[f"{kind}s"]:
        raise RowSynthError(f"the layers have no {kind} called {name!r}")


def _channel(song: dict, name: object) -> None:
    """Refuse what is no channel of a song."""
    if not isinstance(name, str) or name not in song["channels"]:
        raise RowSynthError(f"the song has no channel called {name!r}")


def _free(taken: Collection[str], name: str) -> str:
    """A name that is not taken: the one given, or with the first number from 2 after it that makes it free."""
    number, found = 1, name
    while found in taken:
        number += 1
        found = f"{name}{number}"
    return found


def _seconds(value: object, what: str) -> float:
    """A length of time in seconds, 0 or more. Any other is refused, in words that `what` begins."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise RowSynthError(f"{what} is a number of seconds, 0 or more, not {value!r}")
    return value


def layered(song: dict) -> dict:
    """A song with layers switched on: one group that holds every channel, and one version that holds that group.

    That is the song as it sounded before, said in layers. A song that has layers already is refused.
    """
    if "layers" in song:
        raise RowSynthError("the song has layers already")
    made = copy.deepcopy(song)
    made["layers"] = {"groups": {GROUP: list(song["channels"])}, "versions": {VERSION: [GROUP]}}
    return made


def unlayered(song: dict) -> dict:
    """A song with its layers switched off: the groups, the versions, the keys and what was mixed ahead all go."""
    _layers(song)
    made = copy.deepcopy(song)
    del made["layers"]
    return made


def group_added(song: dict, name: str | None = None) -> dict:
    """A song with one more group, the last of them, which has no channels yet and is in no version.

    Its name is `name`, held to what `allowed` allows. With none given it is "group", or "group2" and so on
    where that is taken.
    """
    name = allowed(song, "group", _free(_layers(song)["groups"], NEW["group"]) if name is None else name)
    made = copy.deepcopy(song)
    made["layers"]["groups"][name] = []
    return made


def group_renamed(song: dict, old: str, new: str) -> dict:
    """A song with one of its groups under another name, in its place among the groups and in every version."""
    _known(song, "group", old)
    allowed(song, "group", new)
    made = copy.deepcopy(song)
    layers = made["layers"]
    layers["groups"] = _rekeyed(layers["groups"], old, new)
    layers["versions"] = {version: _swapped(named, old, new) for version, named in layers["versions"].items()}
    return made


def group_removed(song: dict, name: str) -> dict:
    """A song without one of its groups: it goes from every version too, and its channels are then in no group.

    The only group there is cannot go, since layers with none have nothing to mix: the layers are switched off
    instead.
    """
    _known(song, "group", name)
    made = copy.deepcopy(song)
    layers = made["layers"]
    if len(layers["groups"]) == 1:
        raise RowSynthError(f"group {name!r} is the only one: layers need a group, and are switched off to have none")
    del layers["groups"][name]
    for named in layers["versions"].values():
        named[:] = [group for group in named if group != name]
    return made


def channel_put(song: dict, channel: str, group: str) -> dict:
    """A song with one of its channels in a group, the last of that group, and in no other group.

    A channel is heard with one group, so what other group it was in lets go of it.
    """
    _known(song, "group", group)
    _channel(song, channel)
    made = copy.deepcopy(song)
    groups = made["layers"]["groups"]
    if channel not in groups[group]:
        groups[group].append(channel)
    for name, members in groups.items():
        if name != group:
            members[:] = [member for member in members if member != channel]
    return made


def channel_taken(song: dict, channel: str) -> dict:
    """A song with one of its channels taken out of the group it is in, so that it is in none.

    A channel in no group is heard in no version, which the checks say. One that is in none already is refused.
    """
    _channel(song, channel)
    made = copy.deepcopy(song)
    groups = _layers(made)["groups"]
    if not any(channel in members for members in groups.values()):
        raise RowSynthError(f"channel {channel!r} is in no group, so there is none to take it out of")
    for members in groups.values():
        members[:] = [member for member in members if member != channel]
    return made


def version_added(song: dict, name: str | None = None) -> dict:
    """A song with one more version, the last of them, which has no group yet.

    Its name is `name`, held to what `allowed` allows. With none given it is "version", or "version2" and so on
    where that is taken.
    """
    name = allowed(song, "version", _free(_layers(song)["versions"], NEW["version"]) if name is None else name)
    made = copy.deepcopy(song)
    made["layers"]["versions"][name] = []
    return made


def version_copied(song: dict, name: str) -> dict:
    """A song with one more version, the last of them, which has the groups one of its versions has.

    It is called as that one is, with the first number from 2 after it that no version has yet: `full2`. What
    was to be mixed ahead of the one is not mixed ahead of its copy.
    """
    _known(song, "version", name)
    made = copy.deepcopy(song)
    versions = made["layers"]["versions"]
    versions[allowed(song, "version", _free(versions, name))] = list(versions[name])
    return made


def version_renamed(song: dict, old: str, new: str) -> dict:
    """A song with one of its versions under another name, in its place among them and where it is mixed ahead."""
    _known(song, "version", old)
    allowed(song, "version", new)
    made = copy.deepcopy(song)
    layers = made["layers"]
    layers["versions"] = _rekeyed(layers["versions"], old, new)
    for mix in layers.get("premix", []):
        if mix["version"] == old:
            mix["version"] = new
    return made


def version_removed(song: dict, name: str) -> dict:
    """A song without one of its versions, and without what was to be mixed ahead of it.

    The only version there is cannot go: a layered song is heard as one of its versions.
    """
    _known(song, "version", name)
    made = copy.deepcopy(song)
    layers = made["layers"]
    if len(layers["versions"]) == 1:
        raise RowSynthError(f"version {name!r} is the only one, and a layered song is heard as one of its versions")
    del layers["versions"][name]
    if "premix" in layers:
        layers["premix"] = [mix for mix in layers["premix"] if mix["version"] != name]
    return made


def switched(song: dict, group: str, version: str, on: bool | None = None) -> dict:
    """A song with a group in a version, or out of it: `on` says which, and None is for the other than it is now.

    A group that is put in stands among the groups of the version where the order of the groups has it, so
    one that is taken out and put in again is where it was.
    """
    _known(song, "group", group)
    _known(song, "version", version)
    made = copy.deepcopy(song)
    order = list(made["layers"]["groups"])
    named = made["layers"]["versions"][version]
    if on is None:
        on = group not in named
    if not on:
        named[:] = [held for held in named if held != group]
    elif group not in named:
        after = [at for at, held in enumerate(named) if order.index(held) > order.index(group)]
        named.insert(after[0] if after else len(named), group)
    return made


def keys(song: dict) -> list[int]:
    """The keys a layered song is heard in, each as semitones up: the one it is written in where it says none."""
    return list(_layers(song).get("keys", [0]))


def key_added(song: dict, key: int) -> dict:
    """A song that is heard in one more key, the last of its keys: a whole number of semitones up, or down below 0.

    Refused are a key the song has already, and one that takes a note out of what the text can write, which
    `songedit.transposed` finds.
    """
    held = keys(song)
    if isinstance(key, bool) or not isinstance(key, int):
        raise RowSynthError(f"a key is a whole number of semitones, not {key!r}")
    if key in held:
        raise RowSynthError(f"the song is heard in key {key} already")
    songedit.transposed(song, key)
    made = copy.deepcopy(song)
    made["layers"]["keys"] = [*held, key]
    return made


def key_removed(song: dict, key: int) -> dict:
    """A song that is no longer heard in one of its keys, and without what was to be mixed ahead in it.

    The only key there is cannot go. Where the one key left is the one the song is written in, the keys are
    not written, which is what the engine takes for a song that says none.
    """
    held = keys(song)
    if key not in held:
        raise RowSynthError(f"the song is not heard in key {key!r}")
    if len(held) == 1:
        raise RowSynthError(f"key {key} is the only one, and a song is heard in one key at the least")
    made = copy.deepcopy(song)
    layers = made["layers"]
    layers["keys"] = [other for other in held if other != key]
    if layers["keys"] == [0]:
        del layers["keys"]
    if "premix" in layers:
        layers["premix"] = [mix for mix in layers["premix"] if mix["key"] != key]
    return made


def premixed(song: dict, version: str, key: int, on: bool | None = None) -> dict:
    """A song with a version mixed ahead in a key, or no longer: `on` says which, and None is for the other than now.

    What is mixed ahead is one more file at the build, that version in that key. One that is put in is the
    last of them, and one that is as `on` says already leaves the song as it is. Where the last of them is
    taken out, the layers say nothing of mixing ahead. Refused are a version the layers do not have, and a
    key that is none of those `keys` gives for the song.
    """
    _known(song, "version", version)
    if isinstance(key, bool) or not isinstance(key, int) or key not in keys(song):
        raise RowSynthError(f"the song is not heard in key {key!r}")
    made = copy.deepcopy(song)
    layers = made["layers"]
    pair = {"version": version, "key": key}
    held = layers.get("premix", [])
    if on is None:
        on = pair not in held
    if on and pair not in held:
        layers["premix"] = [*held, pair]
    elif not on and pair in held:
        layers["premix"] = [mix for mix in held if mix != pair]
        if not layers["premix"]:
            del layers["premix"]
    return made


def intro_on(song: dict) -> dict:
    """A song with an intro: the first pattern of its order, played once before the song goes round.

    The tail and the fade are left out, which is 1.5 seconds and 0.05 to the engine. Where a bar of the song
    is shorter than that tail, which the engine would refuse, the tail is written as long as a bar. A song
    that has an intro already is refused, and so is one with nothing in its order.
    """
    if "intro" in song:
        raise RowSynthError("the song has an intro already")
    if not song["order"]:
        raise RowSynthError("the song has no pattern in its order, so an intro has nothing to play")
    made = copy.deepcopy(song)
    made["intro"] = {"order": [song["order"][0]]}
    bar = math.floor(bar_seconds(song) * 100) / 100
    if bar < _TAIL:
        made["intro"]["tail"] = bar
    return made


def intro_off(song: dict) -> dict:
    """A song without its intro. Its patterns stay."""
    _intro(song)
    made = copy.deepcopy(song)
    del made["intro"]
    return made


def intro_order(song: dict, order: list[str]) -> dict:
    """A song whose intro plays other patterns: those `order` names, in turn. It names one at the least."""
    _intro(song)
    if not isinstance(order, list) or not order:
        raise RowSynthError("an intro plays one pattern at the least")
    for name in order:
        if not isinstance(name, str) or name not in song["patterns"]:
            raise RowSynthError(f"the song has no pattern called {name!r}")
    made = copy.deepcopy(song)
    made["intro"]["order"] = list(order)
    return made


def _timed(song: dict, key: str, usual: float, seconds: float) -> dict:
    """A song with a length of its intro set, or taken out of the file where it is what the engine takes for none."""
    made = copy.deepcopy(song)
    if seconds == usual:
        made["intro"].pop(key, None)
    else:
        made["intro"][key] = seconds
    return made


def intro_tail(song: dict, seconds: float) -> dict:
    """A song whose intro runs on for so many seconds past its patterns, for what is still sounding.

    The tail is no longer than a bar of the song, which is all the engine plays of it, and no shorter than the
    fade at its end.
    """
    fade = _intro(song).get("fade", _FADE)
    _seconds(seconds, "the tail of an intro")
    if seconds > bar_seconds(song):
        raise RowSynthError(f"a tail of {seconds} seconds is longer than a bar of the song, which lasts "
                            f"{bar_seconds(song):g}")
    if seconds < fade:
        raise RowSynthError(f"a tail of {seconds} seconds is shorter than the fade at its end, which is {fade}")
    return _timed(song, "tail", _TAIL, seconds)


def intro_fade(song: dict, seconds: float) -> dict:
    """A song whose intro falls away to nothing over its last so many seconds. That is no longer than its tail."""
    tail = _intro(song).get("tail", _TAIL)
    _seconds(seconds, "the fade of an intro")
    if seconds > tail:
        raise RowSynthError(f"a fade of {seconds} seconds is longer than the tail it ends, which is {tail}")
    return _timed(song, "fade", _FADE, seconds)
