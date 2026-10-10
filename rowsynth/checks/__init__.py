"""Checks: the rules a project is held to. Every other module of this package holds one set of them."""
from __future__ import annotations

import math
import pkgutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib import import_module

from ..errors import RowSynthError
from ..project import Project

# The rows of a channel are its lengths added up one after the other, as floats. A length that is no even share
# of a row, a triplet's, leaves the sum a hair off: six triplet quarters come to 15.999999999999998 rows and not
# to 16. Rows closer together than this are the same rows. The adding is out by far less, a millionth of this
# after three hundred triplets, and no length that is written comes near to being this short.
_ROW_SLACK = 1e-6


@dataclass(frozen=True)
class Finding:
    """One breach of a rule: the rule, what broke it, and what was found there.

    target is the place in the project. "sounds.coin" is a sound, "sounds.coin[1]" the second variant of one,
    and "sounds.coin.parts[0]" the first part of one made of parts. A held sound has three parts,
    "sounds.hum.start", "sounds.hum.loop" and "sounds.hum.end", and what is found between them goes by the
    sound itself, "sounds.hum".
    message says what was found, with the number.
    """

    rule: str
    target: str
    message: str

    def to_dict(self) -> dict:
        """The finding as plain data, ready for JSON."""
        return asdict(self)


@contextmanager
def _at(target: str) -> Iterator[None]:
    """What is refused inside is refused with the target in front."""
    try:
        yield
    except RowSynthError as error:
        raise RowSynthError(f"{target}: {error}") from None


def _same_rows(one: float, other: float) -> bool:
    """Whether two counts of rows, or two places counted in rows, are the same but for what adding them up left."""
    return abs(one - other) < _ROW_SLACK


def _whole_rows(rows: float) -> float:
    """A count of rows, or a place counted in them, as the whole number it is the same as. Any other stays as it is."""
    near = float(math.floor(rows + 0.5))
    return near if _same_rows(rows, near) else rows


def run(project: Project, rate: int = 44100) -> list[Finding]:
    """Every finding of every check on a project, rendered at `rate`. A project that keeps the rules gives none.

    Each other module of this package has a `check(project, rate)` that gives its own findings. They are
    called in the order of their names, and the findings come in that order.
    """
    found: list[Finding] = []
    for name in sorted(module.name for module in pkgutil.iter_modules(__path__)):
        found += import_module(f"{__name__}.{name}").check(project, rate)
    return found
