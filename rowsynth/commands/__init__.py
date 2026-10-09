"""What the commands share. Every other module of this package is one command of the command line."""
from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

import numpy as np

from ..bank import render_entry
from ..errors import RowSynthError
from ..project import Project
from ..song import render_song

Read = TypeVar("Read")


def _writable(data: object) -> object:
    """`data` with None in place of each number JSON cannot write: an infinity, or what is not a number at all."""
    if isinstance(data, float) and not math.isfinite(data):
        return None
    if isinstance(data, dict):
        return {key: _writable(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [_writable(value) for value in data]
    return data


def emit(data: object) -> None:
    """Print `data` to standard output as one JSON document: an indent of 2, keys in the order they come in.

    A level of minus infinity, which is what silence measures as, is printed as null. Python would write
    -Infinity there, and a reader that keeps to JSON refuses the whole document for it.
    """
    print(json.dumps(_writable(data), indent=2, ensure_ascii=False, allow_nan=False))


def load_file(read: Callable[[str | Path], Read], path: str | Path) -> Read:
    """What `read` makes of the file at `path`. A file that is not there is refused by its path."""
    try:
        return read(path)
    except FileNotFoundError:
        raise RowSynthError(f"{path}: no such file") from None


def load_project(path: str | Path) -> Project:
    """The project a file holds. A file that is not there is refused by its path, as one that is not JSON is."""
    return load_file(Project.load, path)


def kind_of(project: Project, file: str | Path, name: str, variant: int | None) -> str:
    """What a name is in the project of a file: "sound" or "song".

    A name that is both is refused, and so is one that is neither. A variant asked for of a song is refused
    too, since only a sound has variants.
    """
    sound, song = name in project.sounds, name in project.songs
    if sound and song:
        raise RowSynthError(f"{name!r} is both a sound and a song of {file}")
    if not sound and not song:
        raise RowSynthError(f"{name!r} is no sound and no song of {file}")
    if song and variant is not None:
        raise RowSynthError(f"--variant is for a sound, and {name!r} is a song")
    return "sound" if sound else "song"


def render_sides(project: Project, kind: str, name: str, rate: int, variant: int | None) -> list[np.ndarray]:
    """A sound or a song of a project as its sides: the one side of a sound, or the left and the right of a song.

    `kind` is what `kind_of` says of the name. A sound is at the level the bank gives it, and is the variant
    asked for, counted from 0: the first when `variant` is None. One the sound does not have is refused.
    """
    if kind == "song":
        mix = render_song(project.song(name), rate=rate)
        return [mix["left"], mix["right"]]
    buffers = render_entry(project.sound(name), rate)
    picked = variant or 0
    if not 0 <= picked < len(buffers):
        raise RowSynthError(f"{name!r} has no variant {picked}: it has {len(buffers)}, counted from 0")
    return [buffers[picked]]


def mono(sides: Sequence[np.ndarray]) -> np.ndarray:
    """The sides of a sound as one row of doubles: their mean, which for a sound of one side is that side."""
    return np.mean(sides, axis=0, dtype=np.float64)
