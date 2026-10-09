"""A song's stems mixed to stereo: the groups of a layered song in every key, its versions, and a song's intro."""
from __future__ import annotations

import math

import numpy as np

from .errors import RowSynthError
from .note import _length
from .song import lay, pan_gains, render_stems, up


def bar_rows(song: dict) -> int:
    """The rows in a bar of a song, which is four of its beats."""
    return song.get("rows_per_beat", 4) * 4


def keyless(song: dict, channels: list[str]) -> bool:
    """Whether a group of a song's channels sounds the same in every key: it does when all of them are drum channels.

    A drum channel keeps its text when the song is moved up, so it plays what it played before. A channel the
    song does not have is no drum channel.
    """
    return all(song["channels"].get(name, {}).get("drums", False) for name in channels)


def _groups(song: dict) -> dict[str, list[str]]:
    """The groups of a layered song, {group: [channels]}, once every channel they name is known to have a stem."""
    layers = song.get("layers")
    if layers is None:
        raise RowSynthError("the song has no layers, so it has no groups to mix")
    groups = layers.get("groups")
    if not groups:
        raise RowSynthError("layers: no groups, so there is nothing to mix")
    # A channel has a stem when the song laid end to end has it, and that is known before a note is played.
    laid, _ = lay(song)
    for group, channels in groups.items():
        for name in channels:
            if name not in song["channels"]:
                raise RowSynthError(f"group {group!r}: no channel called {name!r}")
            if name not in laid:
                raise RowSynthError(f"group {group!r}: channel {name!r} plays in no pattern of the order, "
                                    "so it has no stem")
    return groups


def _mix(stems: list[dict], size: int) -> tuple[np.ndarray, np.ndarray]:
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


def render_groups(song: dict, rate: int = 48000) -> dict:
    """The groups of a layered song as samples: {"loop_samples", "rows", "keys", "groups"}.

    `song["layers"]["groups"]` is {group: [channels]}. `song["layers"]["keys"]` lists the keys the song is heard
    in, each a number of semitones it is moved up: 0 is the song as written, and a list left out is [0]. "keys"
    is that list. "groups" is {group: {"keyless", "mixes"}}, and "mixes" is {key: (left, right)}, the group's
    stems in that key: each is shared between the two sides by its channel's pan, and they are added in the
    order the group names its channels in. A side is `loop_samples` long and made to be played round and round,
    as a stem is. A group that is keyless, drum channels alone, is mixed once, so its "mixes" hold one mix and
    not one a key: `group_mix` finds the mix of either kind of group in a key. "rows" are the rows of the song.
    """
    groups = _groups(song)
    keys = list(song["layers"].get("keys", [0]))
    if not keys:
        raise RowSynthError("layers.keys: no keys, so there is no key to mix the groups in")
    renders = {key: render_stems(up(song, key) if key else song, rate) for key in keys}
    stems = {key: {stem["name"]: stem for stem in played["stems"]} for key, played in renders.items()}
    # A song moved up has other notes and the same lengths: the first key's loop and rows are those of every key.
    first = renders[keys[0]]
    mixed = {}
    for group, channels in groups.items():
        still = keyless(song, channels)
        mixes = {}
        for key in keys[:1] if still else keys:
            mixes[key] = _mix([stems[key][name] for name in channels], first["loop_samples"])
        mixed[group] = {"keyless": still, "mixes": mixes}
    return {"loop_samples": first["loop_samples"], "rows": first["rows"], "keys": keys, "groups": mixed}


def group_mix(group: dict, key: int) -> tuple[np.ndarray, np.ndarray]:
    """The left side and the right of one group of `render_groups` in a key.

    A keyless group has one mix, kept under the first of the keys, and gives it whatever the key.
    """
    mixes = group["mixes"]
    if group["keyless"]:
        return next(iter(mixes.values()))
    if key not in mixes:
        raise RowSynthError(f"key {key!r}: the group has no mix in it, only in {list(mixes)}")
    return mixes[key]


def mix_version(rendered: dict, song: dict, version: str, key: int) -> tuple[np.ndarray, np.ndarray]:
    """One version of a layered song in one key, as a left side and a right one: the sum of its groups.

    `rendered` is what `render_groups` gave for the song, and `key` is one of its keys. The groups of the
    version are those `song["layers"]["versions"]` lists under its name. Their mixes are added as they are,
    one at a time in that order, and what a side holds is rounded to float32 before the next goes in. The
    sides are new buffers, and the mixes stay as they were.
    """
    versions = (song.get("layers") or {}).get("versions") or {}
    if version not in versions:
        raise RowSynthError(f"layers.versions: no version called {version!r}")
    if key not in rendered["keys"]:
        raise RowSynthError(f"version {version!r}: key {key!r} is not one of the rendered keys, {rendered['keys']}")
    left = np.zeros(rendered["loop_samples"], dtype=np.float32)
    right = np.zeros(rendered["loop_samples"], dtype=np.float32)
    for name in versions[version]:
        if name not in rendered["groups"]:
            raise RowSynthError(f"version {version!r}: group {name!r} was not rendered")
        group_left, group_right = group_mix(rendered["groups"][name], key)
        left += group_left.astype(np.float64)
        right += group_right.astype(np.float64)
    return left, right


def shared_scale(rendered: dict, peak_db: float = -1.0) -> float:
    """The one factor that brings the loudest sample of all the group mixes of `render_groups` to `peak_db`.

    `peak_db` is in decibels against full scale. Every mix multiplied by this factor keeps its level beside
    the others, so that files made from them can be added up into a version. Groups that are all silence have
    no loudest sample to go by, and the factor is 1.
    """
    peak = 0.0
    for group in rendered["groups"].values():
        for sides in group["mixes"].values():
            for side in sides:
                peak = max(peak, float(np.abs(side).max(initial=0.0)))
    return 10 ** (peak_db / 20) / peak if peak > 0 else 1.0


# The seconds an intro runs on for past the last of its patterns, and how many of them at its very end are
# faded to nothing, when the intro does not say.
_TAIL = 1.5
_FADE = 0.05


def _intro(song: dict) -> tuple[list[str], float, float]:
    """The intro of a song once it is known to make sense: its order, and the seconds of its tail and of its fade."""
    intro = song.get("intro")
    if intro is None:
        raise RowSynthError("the song has no intro, so there is nothing to play before its loop")
    order = intro.get("order")
    if not order:
        raise RowSynthError("intro.order: no patterns, so the intro has nothing to play")
    for name in order:
        if name not in song["patterns"]:
            raise RowSynthError(f"intro.order: no pattern called {name!r}")
    tail = intro.get("tail", _TAIL)
    fade = intro.get("fade", _FADE)
    _length(tail, "intro.tail")
    _length(fade, "intro.fade")
    if fade > tail:
        raise RowSynthError(f"intro.fade: {fade!r} seconds is longer than the tail, which is {tail!r}")
    return list(order), tail, fade


def _unused(patterns: dict) -> str:
    """A name that none of a song's patterns has."""
    name = "silence"
    while name in patterns:
        name += "_"
    return name


def render_intro(song: dict, rate: int = 48000) -> dict:
    """The intro of a song as samples: {"left", "right", "walk_samples", "in_samples"}.

    `song["intro"]` is {"order", "tail", "fade"}: the patterns that are played once before the song's loop, and
    two lengths in seconds, which are 1.5 and 0.05 when left out. The patterns are played as stems with one bar
    of rests after them, so that what their last notes are still sounding rings out into that bar and is not
    added back on at the start. The stems are mixed as a group's are, each shared between the two sides by its
    channel's pan. `walk_samples` is where the patterns end, which is where the loop is to begin. The sides
    run on past it for the tail and are `in_samples` long, and over the last `fade` seconds of them they fall
    away evenly to nothing. The tail is cut from the silent bar, and one that is longer than the bar is refused.
    """
    order, tail, fade = _intro(song)
    bar = _unused(song["patterns"])
    # Every channel of the song rests for a whole note, which is one bar.
    rests = {name: "r1" for name in song["channels"]}
    played = render_stems({**song, "patterns": {**song["patterns"], bar: rests}, "order": [*order, bar]}, rate)
    loop_samples, rows = played["loop_samples"], played["rows"]
    # The patterns have the rows of the stems less the bar's, and the same share of the samples.
    walk_samples = math.floor(loop_samples * (rows - bar_rows(song)) / rows + 0.5)
    tail_samples = math.floor(tail * rate + 0.5)
    # The stems end where the silent bar does. How many samples the bar came to is known only now they are made.
    if tail_samples > loop_samples - walk_samples:
        raise RowSynthError(f"intro.tail: {tail!r} seconds is {tail_samples} samples at a rate of {rate}, and the "
                            f"silent bar after the intro has {loop_samples - walk_samples}")
    in_samples = walk_samples + tail_samples
    fade_samples = math.floor(fade * rate + 0.5)
    left, right = _mix(played["stems"], in_samples)
    # The last sample is brought to nothing, and each one before it keeps one more share in `fade_samples` of
    # what it had. The products are worked out in doubles and rounded to float32 as they go back into a side.
    # A fade of no samples has no slope, and nothing is changed.
    slope = np.arange(fade_samples - 1, -1, -1) / fade_samples
    for side in (left, right):
        side[in_samples - fade_samples:] *= slope
    return {"left": left, "right": right, "walk_samples": walk_samples, "in_samples": in_samples}
