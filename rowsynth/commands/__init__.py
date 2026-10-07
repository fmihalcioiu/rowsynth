"""What the commands share. Every other module of this package is one command of the command line."""
from __future__ import annotations

import json
from pathlib import Path

from ..errors import RowSynthError
from ..project import Project


def emit(data: object) -> None:
    """Print `data` to standard output as one JSON document: an indent of 2, keys in the order they come in."""
    print(json.dumps(data, indent=2, ensure_ascii=False))


def load_project(path: str | Path) -> Project:
    """The project a file holds. A file that is not there is refused by its path, as one that is not JSON is."""
    try:
        return Project.load(path)
    except FileNotFoundError:
        raise RowSynthError(f"{path}: no such file") from None
