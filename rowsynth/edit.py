"""A change to a project: a place in its data read, set or deleted, and one or several of them done to its file."""
from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import NoReturn

from .errors import RowSynthError
from .project import Project

_NAME = re.compile(r"[^.\[\]]+")
_NUMBER = re.compile(r"[0-9]+")

# What a value can be, in the words the project's checks have for each. To Python true and false are whole
# numbers, so they are asked after before the numbers are.
_KINDS = ((dict, "a dict"), (list, "a list"), (str, "a string"), (bool, "true or false"), ((int, float), "a number"))


class _Nothing:
    """What stands for no value at all. None cannot: it is a value, the null of a file."""

    def __repr__(self) -> str:
        return "NOTHING"


NOTHING = _Nothing()


def steps(where: str) -> list[str | int]:
    """A place read into its steps: a string for each name, and a whole number for each item of a list.

    A place is written as the checks write a target: names joined by dots, and `[n]` straight after what holds
    a list for its item n, counted from 0. "sounds.zaps[1].notes" is ["sounds", "zaps", 1, "notes"]. A name
    cannot hold a dot or a bracket. What is not written so is refused, with the column the reading stops at,
    counted from 1.
    """
    def stop(at: int, expected: str) -> NoReturn:
        raise RowSynthError(f"{where!r}, column {at + 1}: {expected}")

    found: list[str | int] = []
    at = 0
    while True:
        name = _NAME.match(where, at)
        if not name:
            stop(at, "a name is expected here")
        found.append(name[0])
        at = name.end()
        while where.startswith("[", at):
            number = _NUMBER.match(where, at + 1)
            if not number:
                stop(at + 1, "a whole number, 0 or more, is expected after '['")
            if not where.startswith("]", number.end()):
                stop(number.end(), "']' is expected after the number")
            found.append(int(number[0]))
            at = number.end() + 1
        if at == len(where):
            return found
        if where[at] != ".":
            stop(at, "a dot, '[' or the end of the place is expected here")
        at += 1


def _written(path: list[str | int]) -> str:
    """Steps written back as the place they were read from."""
    return "".join(f"[{step}]" if isinstance(step, int) else f".{step}" for step in path).lstrip(".")


def _take(held: object, path: list[str | int], at: int) -> object:
    """What step `at` of a path leads to out of `held`, which is what the steps before it led to.

    A step that leads nowhere is refused: a key the dict does not have, an item the list does not have, a key
    asked of a list, an item asked of a dict, or anything asked of a number, a string or the like. The message
    names the part of the place that is there, and the key or the item that is not.
    """
    step = path[at]
    item = isinstance(step, int)
    fits = isinstance(held, list if item else dict)
    if fits and (step < len(held) if item else step in held):
        return held[step]
    there = _written(path[:at]) or "the data"
    asked = f"item [{step}]" if item else f"key {step!r}"
    if not fits:
        kind = next((word for kinds, word in _KINDS if isinstance(held, kinds)), "null")
        raise RowSynthError(f"{there} is {kind}, and has no {asked}")
    count = f": it holds {len(held)}, counted from 0" if item else ""
    raise RowSynthError(f"{there} has no {asked}{count}")


def _holder(data: object, path: list[str | int]) -> object:
    """What holds the last step of a path: where the steps before it lead, taken one after the other."""
    held = data
    for at in range(len(path) - 1):
        held = _take(held, path, at)
    return held


def get_in(data: object, where: str) -> object:
    """What is at a place in `data`. A place that is not there is refused."""
    path = steps(where)
    return _take(_holder(data, path), path, len(path) - 1)


def set_in(data: object, where: str, value: object) -> object:
    """Put `value` at a place in `data`, and give what was there before, or NOTHING when nothing was.

    A key its dict does not have yet is added at the end, and so is the item `[n]` of a list that holds n.
    An item past that is refused, and so is a place whose earlier steps lead nowhere. `data` is as it was
    after a refusal.
    """
    path = steps(where)
    held, step = _holder(data, path), path[-1]
    if isinstance(step, int) and isinstance(held, list) and step == len(held):
        held.append(value)
        return NOTHING
    if isinstance(step, str) and isinstance(held, dict) and step not in held:
        held[step] = value
        return NOTHING
    was = _take(held, path, len(path) - 1)
    held[step] = value
    return was


def delete_in(data: object, where: str) -> object:
    """Take what is at a place out of `data`, and give it. A place that is not there is refused."""
    path = steps(where)
    held = _holder(data, path)
    was = _take(held, path, len(path) - 1)
    del held[path[-1]]
    return was


def _same(one: object, other: object) -> bool:
    """Whether two values are written as the same JSON.

    Python calls 1 and true equal, and two dicts whose keys come in another order. In a file they differ.
    """
    return json.dumps(one) == json.dumps(other)


def change_file_many(path: str | Path, changes: Iterable[tuple[str, object]]) -> list[dict]:
    """Make several changes to the project file at `path` as one, and save the file when the project then holds.

    Each change is (where, value): the value put at the place, as `set_in` puts it, or with NOTHING for the
    value what is there taken out. The file is read once, the changes are made in the order given, and the
    project is checked once, whole, after the last of them. So things that name one another can be changed
    together, where each change alone would leave the project wrong: an instrument under another name, and the
    song that plays it. The file is saved once. Where a place leads nowhere or the check refuses the project,
    that is refused in its own words and nothing is written.

    The answer has for each change what `change_file` tells of one: {"file", "where", "was", "now"}, with "was"
    what was there as that change was made. A value that is what was there already has "same": true, and
    where every change is so, the file is not written.
    """
    data = Project.load(path).to_dict()
    told = []
    for where, value in changes:
        one = {"file": str(path), "where": where}
        if value is NOTHING:
            one["was"] = delete_in(data, where)
        else:
            # The data has a value of its own, so that a later change to a place in it leaves this one as it is.
            was = set_in(data, where, copy.deepcopy(value))
            if was is not NOTHING:
                one["was"] = was
            one["now"] = value
            if was is not NOTHING and _same(was, value):
                one["same"] = True
        told.append(one)
    if not all(one.get("same") for one in told):
        Project.from_dict(data).save(path)
    return told


def change_file(path: str | Path, where: str, value: object = NOTHING, delete: bool = False) -> dict:
    """Make one change to the project file at `path`, and save the file when the project still holds.

    The change is `value` put at the place `where`, as `set_in` puts it, or with `delete` what is there taken
    out. The file is read, changed, checked and saved within this one call, so what another program wrote to
    it a moment before is kept. The project is checked whole, as it is when it is loaded, before anything is
    written: a change that would leave it wrong is refused with the check's own message, and the file stays
    as it was. It is `change_file_many` with the one change.

    The answer is {"file", "where", "was", "now"}. "was" is left out when nothing was there, and "now" after
    a delete. A value that is what the file holds already is not written, and the answer has "same": true.
    """
    if (value is NOTHING) != delete:
        raise RowSynthError(f"{where}: a change is a value to put there or a delete, and only one of the two")
    return change_file_many(path, [(where, value)])[0]
