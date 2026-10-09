"""What the commands share. Every other module of this package is one command of the command line."""
from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np

from ..bank import render_entry
from ..errors import RowSynthError
from ..project import Project
from ..song import render_song

Read = TypeVar("Read")

# The parts of a held sound, in the order `render_entry` gives their buffers. The first is the one a command
# takes when it is not told which.
PARTS = ("start", "loop", "end")


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


@dataclass(frozen=True)
class Pick:
    """What a command is to render out of a project: a song, or one buffer of a sound.

    kind is "sound" or "song". buffer is which of the buffers `render_entry` makes of the sound: the variant,
    or in a held sound the part. part is the name of that part, and None in anything but a held sound.
    """

    name: str
    kind: str
    buffer: int = 0
    part: str | None = None

    def told(self) -> dict:
        """What a printed document says of which buffer this is: the part of a held sound, and no more."""
        return {} if self.part is None else {"part": self.part}


def pick(project: Project, file: str | Path, name: str, *, variant: int | None, part: str | None) -> Pick:
    """What a name picks out of the project of a file, with the two flags that can go with it.

    `variant` is what --variant was given and `part` what --part was, each None where the flag was left out.
    With neither, a sound is its first variant and a held sound its start. Whatever does not fit is refused
    here and nowhere else: a name that is both a sound and a song, or neither; either flag with a song;
    --part with a sound that is not held, and --variant with one that is; a variant the sound does not have.
    """
    sound, song = name in project.sounds, name in project.songs
    if sound and song:
        raise RowSynthError(f"{name!r} is both a sound and a song of {file}")
    if not sound and not song:
        raise RowSynthError(f"{name!r} is no sound and no song of {file}")
    if song:
        if variant is not None:
            raise RowSynthError(f"--variant is for a sound, and {name!r} is a song")
        if part is not None:
            raise RowSynthError(f"--part is for a held sound, and {name!r} is a song")
        return Pick(name, "song")
    entry = project.sounds[name]
    if "held" in entry:
        if variant is not None:
            raise RowSynthError(f"--variant is for a sound that is not held, and {name!r} is a held sound: "
                                "use --part")
        part = part or PARTS[0]
        return Pick(name, "sound", PARTS.index(part), part)
    if part is not None:
        raise RowSynthError(f"--part is for a held sound, and {name!r} is not one")
    # Variants are a list of defs. A sound of any other form is one buffer, and so has the one variant.
    count = len(entry["defs"]) if "defs" in entry else 1
    picked = variant or 0
    if not 0 <= picked < count:
        raise RowSynthError(f"{name!r} has no variant {picked}: it has {count}, counted from 0")
    return Pick(name, "sound", picked)


def render_sides(project: Project, picked: Pick, rate: int) -> list[np.ndarray]:
    """What was picked out of a project, as its sides: the one side of a sound, or the left and the right of a song.

    A sound is at the level the bank gives it, and is the one of its buffers that was picked.
    """
    if picked.kind == "song":
        mix = render_song(project.song(picked.name), rate=rate)
        return [mix["left"], mix["right"]]
    return [render_entry(project.sound(picked.name), rate)[picked.buffer]]


def mono(sides: Sequence[np.ndarray]) -> np.ndarray:
    """The sides of a sound as one row of doubles: their mean, which for a sound of one side is that side."""
    return np.mean(sides, axis=0, dtype=np.float64)
