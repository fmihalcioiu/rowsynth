"""The set command: one thing in a project file changed, and the rest of the file kept."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NoReturn

from ..edit import NOTHING, change_file
from ..errors import RowSynthError
from . import emit, load_file

NAME = "set"
HELP = ("change one thing in a project file and keep the rest as it is: the file is read, changed, checked and "
        "written in one go, so a change another program made to it a moment before is not lost")
_EXAMPLE = 'example: %(prog)s tune.json songs.tune.patterns.a.lead "o4 l8 c e g e"'


def configure(parser: argparse.ArgumentParser) -> None:
    """Give the parser the file, the place in it and the three ways to say what goes there, with an example."""
    parser.epilog = _EXAMPLE
    parser.add_argument("file", metavar="FILE", help="the project file")
    parser.add_argument("where", metavar="WHERE",
                        help="the place to change, written as the checks write a target: names joined by dots, "
                             "and an item of a list as [n] counted from 0, as in sounds.coin.def.voices[0].inst")
    parser.add_argument("value", metavar="VALUE", nargs="?",
                        help="what to put there: read as JSON when it is JSON, and as text when it is not. "
                             "A key or the next item of a list that is not there yet is added")
    parser.add_argument("--text", action="store_true", help="take the value as text whatever it looks like")
    parser.add_argument("--from", dest="source", metavar="PATH",
                        help="read the value from this file, or from standard input when PATH is -")
    parser.add_argument("--delete", action="store_true", help="remove what is there")


def _not_json(name: str) -> NoReturn:
    """Python reads NaN and Infinity as numbers. JSON has no such numbers, so here they are text like any other."""
    raise ValueError(name)


def _read(path: str) -> str:
    """The text of the file at `path`, or of standard input when it is "-": UTF-8, with or without a byte order mark."""
    try:
        if path == "-":
            # Read as UTF-8 whatever the console is set to, as what is printed is written. A stream that is
            # not the process's own, as in a test, is read as it is.
            reconfigure = getattr(sys.stdin, "reconfigure", None)
            if reconfigure:
                reconfigure(encoding="utf-8-sig")
            return sys.stdin.read()
        return load_file(lambda file: Path(file).read_text(encoding="utf-8-sig"), path)
    except UnicodeDecodeError:
        raise RowSynthError(f"{'standard input' if path == '-' else path}: not UTF-8 text") from None


def _value(args: argparse.Namespace) -> object:
    """The value the command was given: typed after the place, or read from where --from says.

    It is what it reads as when it is JSON, and with --text or when it is not JSON it is the text itself.
    """
    typed = args.value
    if typed is None:
        # The line end a file closes with is no part of the text in it.
        typed = _read(args.source).rstrip("\r\n")
    if args.text:
        return typed
    try:
        return json.loads(typed, parse_constant=_not_json)
    except ValueError:
        return typed


def run(args: argparse.Namespace) -> int:
    """Make the one change to the file, print what was there and what is there now, and return 0."""
    if (args.value is not None) + (args.source is not None) + args.delete != 1:
        raise RowSynthError("set takes one of VALUE, --from and --delete, and only one")
    if args.delete and args.text:
        raise RowSynthError("--text is for a value, and --delete has none")
    value = NOTHING if args.delete else _value(args)
    emit(load_file(lambda file: change_file(file, args.where, value, args.delete), args.file))
    return 0
