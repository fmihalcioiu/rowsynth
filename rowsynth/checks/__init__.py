"""Checks: the rules a project is held to. Every other module of this package holds one set of them."""
from __future__ import annotations

import pkgutil
from dataclasses import asdict, dataclass
from importlib import import_module

from ..project import Project


@dataclass(frozen=True)
class Finding:
    """One breach of a rule: the rule, what broke it, and what was found there.

    target is the place in the project: "sounds.coin", or "sounds.coin[1]" for the second variant of a sound.
    message says what was found, with the number.
    """

    rule: str
    target: str
    message: str

    def to_dict(self) -> dict:
        """The finding as plain data, ready for JSON."""
        return asdict(self)


def run(project: Project, rate: int = 44100) -> list[Finding]:
    """Every finding of every check on a project, rendered at `rate`. A project that keeps the rules gives none.

    Each other module of this package has a `check(project, rate)` that gives its own findings. They are
    called in the order of their names, and the findings come in that order.
    """
    found: list[Finding] = []
    for name in sorted(module.name for module in pkgutil.iter_modules(__path__)):
        found += import_module(f"{__name__}.{name}").check(project, rate)
    return found
