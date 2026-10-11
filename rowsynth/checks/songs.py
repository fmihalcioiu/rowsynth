"""The rules a song is held to: the rows of its patterns, the joins of its loops, the level of its mixes, its layers."""
from __future__ import annotations

import math

import numpy as np

from ..layers import group_mix, mix_version, render_groups, shared_scale
from ..project import Project
from ..song import _channels, _mix_stems, _within, parse, render_stems
from . import Finding, _at, _same_rows, _whole_rows

# A loop clicks each time it comes round when its last sample and its first are this far apart, or more.
_JOIN = 0.02

# The loudest sample of a mix is to stay under this level, in decibels against full scale and as a sample.
_PEAK_DB = -1.0
_PEAK = 10 ** (_PEAK_DB / 20)

# A sample of this size or more is counted as one at full scale.
_FULL = 0.999


def _said(rows: float) -> str:
    """A count of rows as a message says it: a whole one, or one a hair off it, with no point, and any other in full."""
    rows = _whole_rows(rows)
    return str(int(rows)) if rows == int(rows) else str(rows)


def _rows(target: str, song: dict) -> list[Finding]:
    """What the patterns of a song break of the "rows" rule, which wants the same rows of every channel of a pattern.

    A pattern runs for the rows of its longest channel. A channel with fewer was miscounted somewhere, and from
    there on its notes do not fall on the rows they were written for. A channel of triplets is not such a one:
    its rows add up to a hair off those of the channels beside it, and `_same_rows` takes them for the same.
    Every pattern of the song is read, whether the order plays it or not.
    """
    rows_per_beat = song.get("rows_per_beat", 4)
    found = []
    for name in song["patterns"]:
        filled = {}
        for channel, settings, text in _channels(song, name):
            with _within(name, channel):
                _, filled[channel] = parse(text, settings["inst"], rows_per_beat,
                                           song.get("drums") if settings.get("drums") else None)
        longest = max(filled.values(), default=0.0)
        for channel, rows in filled.items():
            if not _same_rows(rows, longest):
                found.append(Finding("rows", f"{target}.patterns.{name}",
                                     f"channel {channel!r} has {_said(rows)} rows, and the longest channel of the "
                                     f"pattern has {_said(longest)}"))
    return found


def _step(side: np.ndarray) -> float:
    """How far the last sample of a loop lies from its first, which is the sample that follows it."""
    return abs(float(side[-1]) - float(side[0]))


def _join(target: str, what: str, step: float) -> list[Finding]:
    """What a loop breaks of the "join" rule with a step of that size where it comes round. `what` names the loop."""
    if step < _JOIN:
        return []
    return [Finding("join", target, f"{what} steps by {step:.3f} from its last sample round to its first, and "
                                    f"{_JOIN} or more clicks")]


def _level(target: str, what: str, *sides: np.ndarray) -> list[Finding]:
    """What the sides of a mix break of the "level" rule, taken as one. `what` names them.

    The samples are compared as doubles, so that one a float32 holds just under a limit is not rounded up to it.
    """
    sizes = np.abs(np.concatenate(sides).astype(np.float64))
    found = []
    peak = float(sizes.max(initial=0.0))
    if peak >= _PEAK:
        found.append(Finding("level", target, f"{what} peaks at {20 * math.log10(peak):.2f} dBFS, and "
                                              f"{_PEAK_DB:g} or over is too loud"))
    full = int(np.count_nonzero(sizes >= _FULL))
    if full:
        found.append(Finding("level", target, f"{what} is at full scale, {_FULL} or over, for {full} of its "
                                              f"samples"))
    return found


def _played(target: str, song: dict, rate: int) -> list[Finding]:
    """What a song breaks of the "join" and the "level" rule as its channels play it: a stem each, and their mix.

    Every stem is a loop of its own. A song with layers is never heard with all of its channels at once, so
    only one without them has its stems mixed and the two sides of that mix held to the level, one by one.
    """
    played = render_stems(song, rate)
    found = []
    for stem in played["stems"]:
        found += _join(target, f"the stem of channel {stem['name']!r}", _step(stem["samples"]))
    if "layers" not in song:
        # The stems go to the two sides as `render_loop` sends them, without playing the song a second time.
        left, right = _mix_stems(played["stems"], played["loop_samples"])
        found += _level(target, "the left side of its mix", left)
        found += _level(target, "the right side of its mix", right)
    return found


def _layers(target: str, song: dict) -> list[Finding]:
    """What the layers of a song break of the "layers" rule, which wants groups that can be added up into versions.

    Every channel is to be in one group, and in it once: a channel in none is never heard, and one that is
    named twice is heard at twice its level wherever both of its places are played. A group is to have
    channels, and to be in a version. A version is to name a group once. Two groups that are in exactly the
    same versions are always heard together, and are one group cut in two. A name in a group that is no
    channel of the song is passed over: it is for the renderer to refuse.
    """
    groups, versions = song["layers"]["groups"], song["layers"]["versions"]
    found = []

    def add(message: str) -> None:
        found.append(Finding("layers", target, message))

    owner: dict[str, str] = {}
    for group, channels in groups.items():
        if not channels:
            add(f"group {group!r} has no channels")
        for name in channels:
            if name not in song["channels"]:
                continue
            if name in owner:
                add(f"channel {name!r} is in group {owner[name]!r} and again in group {group!r}")
            else:
                owner[name] = group
    for name in song["channels"]:
        if name not in owner:
            add(f"channel {name!r} is in no group")
    for version, named in versions.items():
        if len(set(named)) != len(named):
            add(f"version {version!r} names a group twice")
    # The versions a group is in, and the last group that was found in just those.
    heard: dict[tuple, str] = {}
    for group in groups:
        where = tuple(version for version, named in versions.items() if group in named)
        if not where:
            add(f"group {group!r} is in no version")
        if where in heard:
            add(f"groups {heard[where]!r} and {group!r} are in exactly the same versions, so they would be one "
                f"group")
        heard[where] = group
    return found


def _versions(target: str, song: dict, rate: int) -> list[Finding]:
    """What a layered song breaks of the "level" and the "join" rule as it is heard: a version at a time, in a key.

    The groups are mixed once in every key. A version is the sum of its groups, and its two sides are held to
    the level together. A group is played round as a loop of its own, at the one scale that has the loudest
    sample of all the groups at the level limit, and the wider of the steps of its two sides is held to the
    join at that scale. A group of drums alone is one loop whatever the key.
    """
    rendered = render_groups(song, rate)
    keys = rendered["keys"]
    found = []
    for version in song["layers"]["versions"]:
        for key in keys:
            found += _level(target, f"version {version!r} in key {key}", *mix_version(rendered, song, version, key))
    scale = shared_scale(rendered, _PEAK_DB)
    for name, group in rendered["groups"].items():
        for key in keys[:1] if group["keyless"] else keys:
            what = f"group {name!r}" if group["keyless"] else f"group {name!r} in key {key}"
            step = max(_step(side) for side in group_mix(group, key)) * scale
            found += _join(target, f"{what}, at the scale the groups share,", step)
    return found


def _loudest(song: dict, rate: int) -> float:
    """The loudest sample of the mixes that the "level" rule holds a song to, for whoever asks how far over it is.

    They are the two sides of the mix of its stems, and in a song with layers the two sides of every version in
    every key. `check` finds nothing by this.
    """
    if "layers" in song:
        rendered = render_groups(song, rate)
        mixes = [mix_version(rendered, song, version, key)
                 for version in song["layers"]["versions"] for key in rendered["keys"]]
    else:
        played = render_stems(song, rate)
        mixes = [_mix_stems(played["stems"], played["loop_samples"])]
    return max((float(np.abs(side.astype(np.float64)).max(initial=0.0)) for sides in mixes for side in sides),
               default=0.0)


def _findings(target: str, song: dict, rate: int) -> list[Finding]:
    """The findings on one song, in the order `check` gives them in."""
    found = _rows(target, song)
    # A song with a channel short of its pattern would be played out of step, and its samples say nothing.
    in_step = not found
    if in_step:
        found += _played(target, song, rate)
    if "layers" in song:
        broken = _layers(target, song)
        found += broken
        if in_step and not broken:
            found += _versions(target, song, rate)
    return found


def check(project: Project, rate: int) -> list[Finding]:
    """The findings on every song of a project, rendered at `rate`.

    They come in the order of the songs in the project, and for a song in the order of the rules: "rows" (one
    pattern after the other, and its channels in the order it has them in), "join" (the stem of one channel
    after the other) and "level" (the left side of the mix before the right, and its peak before its samples
    at full scale). A song with layers has no mix of all its channels. After the "join" of its stems come its
    "layers" (the groups with no channels or with a channel some group already has, then the channels in no
    group, the versions that name a group twice, and the groups in no version or in the versions of another),
    its "level" (one version after the other, each in every key) and the "join" of its groups (one group after
    the other, each in every key, and a group of drums alone once). A "rows" finding is on its pattern,
    "songs.NAME.patterns.PATTERN", and every other one on the song, "songs.NAME".

    A song with a "rows" finding is not played, so it has no "join" and no "level". One with a "layers"
    finding has its stems played and its groups left unmixed. The intro of a song is not played: its patterns
    are held to "rows" as every pattern is, and that is all. A name that points at nothing, a channel, an
    instrument or a group, is no finding. Loading refuses it, and in a project put together by hand the song is
    refused where reading or playing it comes on the name, as a song that cannot be read or played is: in the
    renderer's words, with "songs.NAME" in front.
    """
    found = []
    for name in project.songs:
        target = f"songs.{name}"
        with _at(target):
            found += _findings(target, project.song(name), rate)
    return found
