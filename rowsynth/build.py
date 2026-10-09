"""A build: the songs of a project written as the files a player loads, and a manifest that names them."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import __version__
from .errors import RowSynthError
from .measure import _fixed
from .note import _count
from .project import Project
from .song import render_loop
from .wav import save_wav

# Where the loudest sample of a song's file is brought to, in decibels against full scale.
_PEAK_DB = -1.0

# The file the manifest is written to, beside the files it names.
_MANIFEST = "music.json"


def _rate(rate: object) -> int:
    """A rate as a plain int, once it is known to be a whole number of 1 or more."""
    _count(rate, "rate", 1)
    return int(rate)


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


def build_song(project: Project, name: str, out_dir: str | Path, rate: int = 48000) -> dict:
    """Writes one song of a project into a folder and gives its entry in the manifest.

    The song is one loop in stereo, NAME.wav, with its loudest sample at 1 dB under full scale. The entry is
    {"loop_samples", "gain", "source", "file"}: the samples in the loop, what the samples of the file are to be
    multiplied by to have the song at its own level again, the digits of `source_hash`, and the name of the
    file without its folder. A silent song has no loudest sample to go by: it is written as it is, with a gain
    of 1. A song with layers or an intro is more than one loop, and is refused. What is wrong with the song is
    refused with its name in front.
    """
    song = project.song(name)
    rate = _rate(rate)
    if "layers" in song or "intro" in song:
        raise RowSynthError(f"songs.{name}: a song with layers or an intro cannot be built")
    try:
        loop = render_loop(song, rate)
    except RowSynthError as error:
        raise RowSynthError(f"songs.{name}: {error}") from None
    sides = [loop["left"], loop["right"]]
    peak = max(float(np.abs(side).max()) for side in sides)
    scale = 10 ** (_PEAK_DB / 20) / peak if peak > 0 else 1.0
    for side in sides:
        # Each product is worked out as a double and stored as a float32.
        side[:] = side.astype(np.float64) * scale
    file = f"{name}.wav"
    save_wav(Path(out_dir, file), sides, rate)
    return {"loop_samples": loop["loop_samples"], "gain": _fixed(1 / scale, 6),
            "source": source_hash(project, name, rate), "file": file}


def build(project: Project, out_dir: str | Path, rate: int = 48000, songs: list[str] | None = None) -> dict:
    """Writes the songs of a project into a folder with their manifest, and gives the manifest.

    `songs` names the songs to build, and left out it is every song of the project, in the project's order.
    The folder is made when it is missing. Each song is written as `build_song` writes it. The manifest is
    {"rate": rate, "songs": {name: entry}}, and it is written to music.json in the folder once every song is
    there. Whatever else the folder holds is left alone.
    """
    rate = _rate(rate)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = project.songs if songs is None else songs
    manifest = {"rate": rate, "songs": {name: build_song(project, name, out_dir, rate) for name in names}}
    text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    (out_dir / _MANIFEST).write_text(text, encoding="utf-8", newline="\n")
    return manifest


def stale(project: Project, manifest: dict) -> list[str]:
    """The songs of a manifest whose files are no longer what a build of the project would write.

    A song is stale when the project no longer has it, or when its "source" in the manifest is not the
    `source_hash` the project gives it at the manifest's rate. The names come in the manifest's order.
    """
    rate = manifest["rate"]
    return [name for name, entry in manifest["songs"].items()
            if name not in project.songs or entry["source"] != source_hash(project, name, rate)]
