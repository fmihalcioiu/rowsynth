"""What the commands share. Every other module of this package is one command of the command line."""
from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from ..errors import RowSynthError
from ..project import Project

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
