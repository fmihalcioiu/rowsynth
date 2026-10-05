"""A project: one JSON file of named data, checked as it is loaded."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import NoReturn

from .errors import RowSynthError

_TABLES = ("instruments",)
_WAVES = ("pulse", "saw", "tri", "noise", "fm")


class Project:
    """The instruments of a project, by name."""

    def __init__(self) -> None:
        self.instruments: dict = {}
        self._keys: list[str] = []

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Project):
            return NotImplemented
        return all(getattr(self, key) == getattr(other, key) for key in _TABLES)

    def to_dict(self) -> dict:
        """The project as plain data: the tables it was loaded with, in their order, then any filled since."""
        keys = self._keys + [key for key in _TABLES if key not in self._keys and getattr(self, key)]
        return {key: copy.deepcopy(getattr(self, key)) for key in keys}

    @classmethod
    def from_dict(cls, data: dict) -> Project:
        """A project from plain data. Anything of the wrong shape is refused, with the place it sits at."""
        _project(data)
        project = cls()
        project._keys = list(data)
        for key in data:
            setattr(project, key, copy.deepcopy(data[key]))
        return project

    @classmethod
    def load(cls, path: str | Path) -> Project:
        """A project read from a JSON file in UTF-8, with or without a byte order mark."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        except UnicodeDecodeError as error:
            raise RowSynthError(f"{path}: not UTF-8 text") from error
        except json.JSONDecodeError as error:
            raise RowSynthError(f"{path}: line {error.lineno}, column {error.colno}: {error.msg}") from error
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        """Write the project as JSON, making the folders above the file when they are missing."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"
        path.write_text(text, encoding="utf-8", newline="\n")


def _fail(where: str, what: str) -> NoReturn:
    """Refuse the data: `where` is the path to the place, `what` is wrong with it."""
    raise RowSynthError(f"{where}: {what}")


def _dict(value: object, where: str) -> None:
    if not isinstance(value, dict):
        _fail(where, "must be a dict")


def _list(value: object, where: str) -> None:
    if not isinstance(value, list):
        _fail(where, "must be a list")


def _number(value: object, where: str) -> None:
    """Python counts true and false as numbers. A project does not."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(where, "must be a number")


def _whole(value: object, where: str, low: int | None = None, high: int | None = None) -> None:
    """A whole number, from `low` to `high` when those are given."""
    span = "" if low is None else f" from {low} to {high}"
    if isinstance(value, bool) or not isinstance(value, int) or (low is not None and not low <= value <= high):
        _fail(where, f"must be a whole number{span}")


def _flag(value: object, where: str) -> None:
    if not isinstance(value, bool):
        _fail(where, "must be true or false")


def _record(value: object, where: str, shape: dict, required: tuple = ()) -> None:
    """A dict with every key of `required` and no key outside `shape`, which maps each key to its check."""
    _dict(value, where)
    for key in required:
        if key not in value:
            _fail(where, f"missing key {key!r}")
    for key, part in value.items():
        if key not in shape:
            _fail(where, f"unknown key {key!r}")
        shape[key](part, f"{where}.{key}")


def _project(data: object) -> None:
    """The whole file: its tables, then everything in them."""
    _dict(data, "project")
    for key, table in data.items():
        if key not in _TABLES:
            _fail("project", f"unknown key {key!r}")
        _dict(table, key)
    instruments = data.get("instruments", {})
    for name, inst in instruments.items():
        _instrument(inst, f"instruments.{name}")


def _steps(table: object, where: str, low: int | None = None, high: int | None = None) -> None:
    """A step table: a number, a list of numbers, or {"v": list, "loop": index}.

    With `low` and `high`, every value is a whole number between them.
    """
    if isinstance(table, dict):
        _record(table, where, {"v": _list, "loop": _whole}, ("v", "loop"))
        table, where = table["v"], f"{where}.v"
    values = [(table, where)]
    if isinstance(table, list):
        values = [(value, f"{where}[{i}]") for i, value in enumerate(table)]
    for value, at in values:
        if low is None:
            _number(value, at)
        else:
            _whole(value, at, low, high)


def _wave(value: object, where: str) -> None:
    if value not in _WAVES:
        _fail(where, f"must be one of {', '.join(map(repr, _WAVES))}, not {value!r}")


def _vib(value: object, where: str) -> None:
    """A vibrato is nothing without its depth. The delay and the rate have values to fall back on."""
    _record(value, where, {"delay": _number, "depth": _number, "rate": _number}, ("depth",))


def _ops(value: object, where: str) -> None:
    """The operators of an FM voice: four at most, each with its envelope."""
    _list(value, where)
    if len(value) > 4:
        _fail(where, f"must hold four operators at most, not {len(value)}")
    env = {key: _number for key in ("a", "d", "s", "sd", "r")}
    for i, op in enumerate(value):
        _record(op, f"{where}[{i}]", {
            "mul": _number,
            "det": _number,
            "level": _number,
            "env": lambda part, at: _record(part, at, env),
        })


def _instrument(inst: object, where: str) -> None:
    """One instrument: only the keys an instrument has, each of its own kind."""
    _record(inst, where, {
        "wave": _wave,
        "vol": _steps,
        "duty": lambda table, at: _steps(table, at, 0, 3),
        "arp": _steps,
        "pitch": _steps,
        "vib": _vib,
        "slide": _number,
        "release": _number,
        "gain": _number,
        "short": _flag,
        "lp": _steps,
        "res": _number,
        "key_scale": _number,
        "gate": _number,
        "alg": lambda alg, at: _whole(alg, at, 0, 7),
        "fb": _number,
        "ops": _ops,
    })
