"""The check command: what the checks find in a project."""
from __future__ import annotations

import argparse

from .. import checks
from ..note import DEFAULT_RATE
from . import emit, load_project

NAME = "check"
HELP = "hold a project file to the rules and give what breaks them: the exit code is 1 when anything does"


def configure(parser: argparse.ArgumentParser) -> None:
    """Give the parser the project to check and the rate to render it at."""
    parser.add_argument("file", metavar="FILE", help="the project file")
    parser.add_argument("--rate", metavar="N", type=int, default=DEFAULT_RATE,
                        help="the samples a second to render at (default %(default)s)")


def run(args: argparse.Namespace) -> int:
    """Print every finding of the checks on the project in the file. Return 1 when there are any, and 0 when none."""
    found = checks.run(load_project(args.file), args.rate)
    emit({"findings": [finding.to_dict() for finding in found]})
    return 1 if found else 0
