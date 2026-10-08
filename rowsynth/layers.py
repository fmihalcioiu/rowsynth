"""A layered song: its channels cut into groups that are always heard together, each mixed to stereo in every key."""
from __future__ import annotations

import numpy as np

from .errors import RowSynthError
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

    The stems are added one at a time in the order given. Each sum is worked out in doubles, and what a side
    then holds is rounded to float32 before the next stem goes in.
    """
    left = np.zeros(size, dtype=np.float32)
    right = np.zeros(size, dtype=np.float32)
    for stem in stems:
        to_left, to_right = pan_gains(stem["pan"])
        wide = stem["samples"].astype(np.float64)
        left += wide * to_left
        right += wide * to_right
    return left, right


def render_groups(song: dict, rate: int = 48000) -> dict:
    """The groups of a layered song as samples: {"loop_samples", "rows", "groups": {name: {"keyless", "mixes"}}}.

    `song["layers"]["groups"]` is {group: [channels]}. `song["layers"]["keys"]` lists the keys the song is heard
    in, each a number of semitones it is moved up: 0 is the song as written, and a list left out is [0].
    "mixes" is {key: (left, right)}, the group's stems in that key: each is shared between the two sides by its
    channel's pan, and they are added in the order the group names its channels in. A side is `loop_samples`
    long and made to be played round and round, as a stem is. A group that is keyless, drum channels alone, is
    mixed once, and its one mix sits under the first key of the list. "rows" are the rows of the song.
    """
    groups = _groups(song)
    keys = song["layers"].get("keys", [0])
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
    return {"loop_samples": first["loop_samples"], "rows": first["rows"], "groups": mixed}
