"""The steps that were made to a project, kept in order so that each can be taken back and made again."""
from __future__ import annotations

import json
from collections.abc import Iterable
from typing import NamedTuple

from ..edit import NOTHING

# How many steps are kept. With one more, the oldest is let go.
MAX_STEPS = 200

# One change to a project: the place, what was there and what is there now. NOTHING is for what is not there.
Change = tuple[str, object, object]


def same(one: object, other: object) -> bool:
    """Whether two values of a place are the same: neither is there, or both are written as the same JSON.

    Python calls 1 and true equal, and two dicts whose keys come in another order. In a file they differ.
    """
    if one is NOTHING or other is NOTHING:
        return one is other
    return json.dumps(one) == json.dumps(other)


def _follows(earlier: tuple[Change, ...], later: tuple[Change, ...]) -> bool:
    """Whether a step goes on where the one before it ended: the same places, each begun at what it was left at."""
    return len(earlier) == len(later) and all(
        one[0] == other[0] and same(one[2], other[1]) for one, other in zip(earlier, later))


class _Step(NamedTuple):
    """One step: its changes in the order they were made, what it is called, and what it is joined by."""

    changes: tuple[Change, ...]
    label: str
    key: object


class History:
    """The steps made to a project: those that can be taken back, and those taken back that can be made again.

    Nothing here changes a project or a file. `undo` and `redo` give the changes there are to make, and whoever
    asks makes them.
    """

    def __init__(self, limit: int = MAX_STEPS) -> None:
        """`limit` is how many steps are kept."""
        self.limit = limit
        self._done: list[_Step] = []
        self._undone: list[_Step] = []
        # Whether the last step that was made is the last thing that happened here, so that one more can join it.
        self._open = False

    @property
    def can_undo(self) -> bool:
        """Whether there is a step to take back."""
        return bool(self._done)

    @property
    def can_redo(self) -> bool:
        """Whether there is a step that was taken back, to make again."""
        return bool(self._undone)

    @property
    def undo_label(self) -> str:
        """What the step `undo` would take back is called, "" when there is none."""
        return self._done[-1].label if self._done else ""

    @property
    def redo_label(self) -> str:
        """What the step `redo` would make again is called, "" when there is none."""
        return self._undone[-1].label if self._undone else ""

    def add(self, changes: Iterable[Change], label: str, key: object = None) -> None:
        """Keeps a step that was made: its changes as (where, was, now), in the order they were made.

        `label` says in a few plain words what the step changed. A step added straight after one with the same
        `key`, unless the key is None, becomes one step with it, from the first `was` to the last `now` and
        under the later label: a knob that is dragged is one step, however many changes the drag was. That is
        for a step that goes on where the one before it ended. One that changes other places, or finds a place
        not as that one left it, is a step of its own.

        A step whose every change ends where it began is not kept, and neither is one that has come to that by
        joining. A step that is kept empties what could be made again, and with more steps than `limit` the
        oldest are let go.
        """
        changes = tuple(changes)
        last = self._done[-1] if self._open else None
        if last is not None and key is not None and last.key == key and _follows(last.changes, changes):
            changes = tuple((where, was, now) for (where, was, _), (_, _, now) in zip(last.changes, changes))
            self._done.pop()
            self._open = False
        if all(same(was, now) for _, was, now in changes):
            return
        self._done.append(_Step(changes, label, key))
        del self._done[:max(0, len(self._done) - self.limit)]
        self._undone.clear()
        self._open = True

    def undo(self) -> tuple[Change, ...] | None:
        """Takes the last step back: gives the changes to make for it, or None when there is no step.

        They are the opposite of the step's own, each from its `now` to its `was`, the last of them first.
        """
        if not self._done:
            return None
        step = self._done.pop()
        self._undone.append(step)
        self._open = False
        return tuple((where, now, was) for where, was, now in reversed(step.changes))

    def redo(self) -> tuple[Change, ...] | None:
        """Makes again the step taken back last: gives its changes as they were made, or None when there is none."""
        if not self._undone:
            return None
        step = self._undone.pop()
        self._done.append(step)
        self._open = False
        return step.changes
