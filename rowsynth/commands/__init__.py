"""What the commands share. Every other module of this package is one command of the command line."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from ..errors import RowSynthError
from ..project import Project

Read = TypeVar("Read")


def emit(data: object) -> None:
    """Print `data` to standard output as one JSON document: an indent of 2, keys in the order they come in."""
    print(json.dumps(data, indent=2, ensure_ascii=False))


def load_file(read: Callable[[str | Path], Read], path: str | Path) -> Read:
    """What `read` makes of the file at `path`. A file that is not there is refused by its path."""
    try:
        return read(path)
    except FileNotFoundError:
        raise RowSynthError(f"{path}: no such file") from None


def load_project(path: str | Path) -> Project:
    """The project a file holds. A file that is not there is refused by its path, as one that is not JSON is."""
    return load_file(Project.load, path)
