"""A build: the songs of a project written as the files a player loads, and a manifest that names them."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import __version__
from .encode import save_audio, unwritable
from .errors import RowSynthError
from .layers import bar_rows, group_mix, keyless, mix_version, render_groups, render_intro
from .measure import _fixed
from .note import _count
from .project import Project
from .song import render_loop

# The samples a second a build is rendered at, when no rate is asked for.
BUILD_RATE = 48000

# The formats the files of a build are written in. MP3 is not one of them: an MP3 file has silence added at
# its ends, and a loop of one cannot go round without a gap.
BUILD_FORMATS = ("wav", "opus")

# Where the loudest sample of a song's files is brought to, in decibels against full scale.
_PEAK_DB = -1.0

# The file the manifest is written to, beside the files it names.
_MANIFEST = "music.json"

# A file to write: its name, and the left side and the right of what goes into it.
_File = tuple[str, tuple[np.ndarray, np.ndarray]]


def _rate(rate: object) -> int:
    """A rate as a plain int, once it is known to be a whole number of 1 or more."""
    _count(rate, "rate", 1)
    return int(rate)


def _writable(fmt: object, rate: int) -> None:
    """Refuses a format that a build is not written in, and one that cannot be written at the rate."""
    if fmt not in BUILD_FORMATS:
        raise RowSynthError(f"format: a build is written as {' or '.join(BUILD_FORMATS)}, not {fmt!r}")
    why = unwritable(fmt, rate)
    if why is not None:
        raise RowSynthError(why)


def _file(stem: str, fmt: str) -> str:
    """The name of a file of a build: what it is called, and the format it is written in as its ending."""
    return f"{stem}.{fmt}"


def source_hash(project: Project, name: str, rate: int) -> str:
    """What the files of a song are made from, as 16 hex digits: the song, the rate and this package's version.

    The song is taken as it would be rendered, the project's instruments under its own. A change to any of the
    three gives other digits, so digits that were kept when the files were built say whether they are still
    the files a build would write.
    """
    song = project.song(name)
    # The keys stay in the order they come in. Channels are added up in that order, and each sum is rounded
    # before the next, so the same song with two of its channels changed round is not the same samples.
    text = json.dumps([song, _rate(rate), __version__])
    return hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def _factor(value: float) -> float:
    """What a player is to multiply samples by, as the manifest holds it: to six decimal places."""
    return _fixed(value, 6)


def _write_level(out_dir: Path, files: list[_File], rate: int, fmt: str) -> float:
    """Writes files in stereo into a folder in the format `fmt`, all at one scale, and gives the scale.

    The scale is what has the loudest sample of all the files at `_PEAK_DB`. Every sample of every file is
    multiplied by it, so the files are as loud beside each other as their sides were. Silence has no loudest
    sample to go by: it is written as it is, and the scale is 1. The sides stay as they were.
    """
    peak = max(float(np.abs(side).max(initial=0.0)) for _, sides in files for side in sides)
    scale = 10 ** (_PEAK_DB / 20) / peak if peak > 0 else 1.0
    for file, sides in files:
        # Each product is worked out as a double and rounded to float32, and the file is made from the float32.
        scaled = [(side.astype(np.float64) * scale).astype(np.float32) for side in sides]
        save_audio(out_dir / file, scaled, rate, fmt)
    return scale


def _build_loop(song: dict, name: str, out_dir: Path, rate: int, fmt: str, source: str,
                beside: list[_File]) -> dict:
    """Writes a song without layers as one loop and gives its entry in the manifest.

    `beside` are the files of the song that are not its loop. They are written with it, at the one scale.
    """
    loop = render_loop(song, rate)
    file = _file(name, fmt)
    scale = _write_level(out_dir, [(file, (loop["left"], loop["right"])), *beside], rate, fmt)
    return {"loop_samples": loop["loop_samples"], "gain": _factor(1 / scale), "source": source, "file": file}


def _one_group_each(song: dict) -> None:
    """Refuses a layered song with a channel that is in no group, or that its groups name more than once.

    The files of the groups are there to be added up into the song. A channel in no group would be missing
    from the sum, and one that is named twice, in two groups or in one, would be in it twice.
    """
    owner: dict[str, str] = {}
    for group, channels in song["layers"]["groups"].items():
        for channel in channels:
            if channel in owner:
                raise RowSynthError(f"layers.groups: channel {channel!r} is in {owner[channel]!r} and again in "
                                    f"{group!r}, so the files would not add up to the song")
            owner[channel] = group
    for channel in song["channels"]:
        if channel not in owner:
            raise RowSynthError(f"layers.groups: channel {channel!r} is in no group, so the files would not add "
                                "up to the song")


def _distinct(files: list[str]) -> None:
    """Refuses file names of which two are the same once case is left out of it, as some systems leave it out."""
    seen: dict[str, str] = {}
    for file in files:
        folded = file.casefold()
        if folded in seen:
            raise RowSynthError(f"two files would be called {seen[folded]!r} and {file!r}, which is one name "
                                "where case does not count")
        seen[folded] = file


def _layer_files(song: dict, name: str, fmt: str,
                 taken: list[str]) -> tuple[dict, list[tuple[str, str, int]], list[tuple[str, str, int]]]:
    """The files the layers of a song are written as, found before a note of them is played: groups, mixes, premixes.

    `groups` is what the manifest says of them: {group: {"file": file}} for a keyless group, which is the same in
    every key and has one file, and {group: {"files": {key: file}}} for any other, the key as a string, which is
    what it is in JSON. `mixes` has (file, group, key) for every one of those files, and `premixes` has (file,
    version, key) for every version the song asks to have mixed ahead. Every name ends in the format `fmt`.
    `taken` are the names of the files the song has besides. A song whose files would not add up to it, a
    premix of a version or in a key the song does not have, and two files with one name, those in `taken`
    among them, are refused.
    """
    layers = song["layers"]
    keys = layers.get("keys", [0])
    _one_group_each(song)
    groups: dict[str, dict] = {}
    mixes = []
    for group, channels in layers["groups"].items():
        if keyless(song, channels):
            groups[group] = {"file": _file(f"{name}-{group}", fmt)}
            # The one mix of a keyless group is its mix in every key, and the first key finds it.
            mixes.append((groups[group]["file"], group, keys[0]))
        else:
            files = [(_file(f"{name}-{group}-{key}", fmt), group, key) for key in keys]
            groups[group] = {"files": {str(key): file for file, _, key in files}}
            mixes += files
    premixes = []
    for at, mix in enumerate(layers.get("premix", [])):
        version, key = mix["version"], mix["key"]
        if version not in layers["versions"]:
            raise RowSynthError(f"layers.premix[{at}]: no version named {version!r}")
        if key not in keys:
            raise RowSynthError(f"layers.premix[{at}]: key {key!r} is not one of the song's keys, {keys}")
        premixes.append((_file(f"{name}.{version}.{key}", fmt), version, key))
    _distinct([*(file for file, _, _ in mixes + premixes), *taken])
    return groups, mixes, premixes


def _build_layers(song: dict, name: str, out_dir: Path, rate: int, fmt: str, source: str,
                  beside: list[_File]) -> dict:
    """Writes a layered song as a file a group and key, all at one scale, and gives its entry in the manifest.

    `beside` are the files of the song that are not of its groups. They are written with those, at the one
    scale. A version mixed ahead is written at a scale of its own, and its entry has what brings it to theirs.
    """
    groups, mixes, premixes = _layer_files(song, name, fmt, [file for file, _ in beside])
    rendered = render_groups(song, rate)
    loop_samples = rendered["loop_samples"]
    mixed = [(file, group_mix(rendered["groups"][group], key)) for file, group, key in mixes]
    scale = _write_level(out_dir, [*mixed, *beside], rate, fmt)
    premixed = []
    for file, version, key in premixes:
        # The groups are added up as they were rendered. Their sum can go past the loudest of them, so it has
        # a scale of its own, and the two scales say how far its file is from theirs.
        own = _write_level(out_dir, [(file, mix_version(rendered, song, version, key))], rate, fmt)
        premixed.append({"id": f"{version}/{key}", "file": file, "scale": _factor(scale / own)})
    entry = {"loop_samples": loop_samples,
             # A bar's share of the rows is its share of the samples, to the nearest sample.
             "bar_samples": math.floor(loop_samples * bar_rows(song) / rendered["rows"] + 0.5),
             "gain": _factor(1 / scale), "source": source, "keys": rendered["keys"], "groups": groups,
             "versions": song["layers"]["versions"]}
    if "premix" in song["layers"]:
        entry["premixed"] = premixed
    return entry


def build_song(project: Project, name: str, out_dir: str | Path, rate: int = BUILD_RATE, fmt: str = "wav") -> dict:
    """Writes one song of a project into a folder and gives its entry in the manifest.

    Every file is written by `save_audio` in the format `fmt`, "wav" or "opus", and has that as its ending, in
    the entry as in the folder: the names below are those of a build of WAV files. Opus is made by ffmpeg.

    A song without layers is one loop in stereo, NAME.wav, with its loudest sample at 1 dB under full scale. Its
    entry is {"loop_samples", "gain", "source", "file"}: the samples in the loop, what the samples of the file
    are to be multiplied by to have the song at its own level again, the digits of `source_hash`, and the name
    of the file without its folder. A silent song has no loudest sample to go by: it is written as it is, with
    a gain of 1.

    A song with layers is one stereo file a group and key, NAME-GROUP-KEY.wav, and NAME-GROUP.wav for a keyless
    group, which is the same in every key. All of them are multiplied by the one number that has the loudest
    sample of them all at 1 dB under full scale, so that a player can add them up into a version. Its entry is
    {"loop_samples", "bar_samples", "gain", "source", "keys", "groups", "versions"}. "bar_samples" is how many
    samples apart the bar lines of the loop are, and "gain" is what it is for a song without layers. "groups"
    is {group: {"file": file}} for a keyless group and {group: {"files": {key: file}}} for any other, the key
    as a string. "keys" and "versions" are the song's own. With `layers["premix"]`, a list of {"version",
    "key"}, the entry has "premixed" as well, a list of {"id", "file", "scale"}: each is that version in that
    key mixed ahead, NAME.VERSION.KEY.wav, with its own loudest sample at 1 dB under full scale. Its "id" is
    VERSION/KEY, and its "scale" is what its samples are to be multiplied by to be at the level of the groups'.

    A song with an intro, with layers or without, has one stereo file more, NAME.intro.wav: what `render_intro`
    gives, to be played once before the loop. It is at the level of the loop or of the groups: the one number
    they are multiplied by is the intro's as well, and it has the loudest sample of them all, the intro's with
    them, at 1 dB under full scale. A version mixed ahead keeps a scale of its own. The entry has "intro" last,
    {"file", "in_samples", "walk_samples"}: the name of the file, the samples in it, and the sample of it at
    which the loop is to begin.

    A layered song is refused when a channel of it is in no group or in a group more than once, when a premix
    names a version or a key it does not have, and when two of its files would have one name where case does
    not count. An intro is refused when `render_intro` refuses it. What is wrong with a song is refused with
    its name in front.

    A format that is neither of the two, and Opus at a rate it does not take, are refused before a note is
    played. What `save_audio` refuses besides, which is Opus when no ffmpeg is found or when ffmpeg ends in an
    error, is refused as a file is to be written, and has the name of the song in front as well.
    """
    song = project.song(name)
    rate = _rate(rate)
    _writable(fmt, rate)
    source = source_hash(project, name, rate)
    out_dir = Path(out_dir)
    write = _build_layers if "layers" in song else _build_loop
    try:
        if "intro" not in song:
            return write(song, name, out_dir, rate, fmt, source, [])
        intro = render_intro(song, rate)
        file = _file(f"{name}.intro", fmt)
        entry = write(song, name, out_dir, rate, fmt, source, [(file, (intro["left"], intro["right"]))])
        entry["intro"] = {"file": file, "in_samples": intro["in_samples"], "walk_samples": intro["walk_samples"]}
        return entry
    except RowSynthError as error:
        raise RowSynthError(f"songs.{name}: {error}") from None


def manifest_path(out_dir: str | Path) -> Path:
    """Where a build into a folder writes its manifest: music.json, beside the files it names."""
    return Path(out_dir, _MANIFEST)


def build(project: Project, out_dir: str | Path, rate: int = BUILD_RATE, fmt: str = "wav",
          songs: list[str] | None = None) -> dict:
    """Writes the songs of a project into a folder with their manifest, and gives the manifest.

    `songs` names the songs to build, and left out it is every song of the project, in the project's order.
    The folder is made when it is missing. Each song is written as `build_song` writes it, in the format `fmt`.
    The manifest is {"rate": rate, "format": fmt, "songs": {name: entry}}, and it is written to the folder's
    `manifest_path` once every song is there. Whatever else the folder holds is left alone. A rate that makes
    no sense, a format a build is not written in and Opus at a rate it does not take are refused before the
    folder is made or a song is rendered.
    """
    rate = _rate(rate)
    _writable(fmt, rate)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = project.songs if songs is None else songs
    entries = {name: build_song(project, name, out_dir, rate, fmt) for name in names}
    manifest = {"rate": rate, "format": fmt, "songs": entries}
    text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    manifest_path(out_dir).write_text(text, encoding="utf-8", newline="\n")
    return manifest


def stale(project: Project, manifest: dict) -> list[str]:
    """The songs of a manifest whose files are no longer what a build of the project would write.

    A song is stale when the project no longer has it, or when its "source" in the manifest is not the
    `source_hash` the project gives it at the manifest's rate. The names come in the manifest's order.
    """
    rate = manifest["rate"]
    return [name for name, entry in manifest["songs"].items()
            if name not in project.songs or entry["source"] != source_hash(project, name, rate)]
