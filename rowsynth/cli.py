"""The command line: python -m rowsynth <command> ..."""
from __future__ import annotations

import argparse
import pkgutil
import sys
from collections.abc import Sequence
from importlib import import_module
from types import ModuleType

from . import commands
from .errors import RowSynthError


def _command_modules() -> list[ModuleType]:
    """The modules of rowsynth.commands, in name order. Each is one command: NAME, HELP, configure and run."""
    names = sorted(found.name for found in pkgutil.iter_modules(commands.__path__))
    return [import_module(f"{commands.__name__}.{name}") for name in names]


def _parsers() -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    """The parser of the whole command line, and the parser of each command by its name.

    There is a sub-command for every command module, set up by the module itself.
    """
    parser = argparse.ArgumentParser(prog="rowsynth", description="A chip synth driven by text.")
    subs = parser.add_subparsers(title="commands", metavar="COMMAND")
    named = {}
    for module in _command_modules():
        sub = subs.add_parser(module.NAME, help=module.HELP, description=module.HELP)
        module.configure(sub)
        sub.set_defaults(run=module.run)
        named[module.NAME] = sub
    return parser, named


def _utf8(stream: object) -> None:
    """Has a stream of the process write UTF-8. One that is not the process's own, as in a test, is left alone."""
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure:
        reconfigure(encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command that `argv` names and give its exit code. With None, `argv` is the process's own.

    A command that stops on a RowSynthError or an OSError has the message printed to standard error after
    "error: ", and the code is 2. With no command named the help goes to standard error, and the code is 2 again.
    What is printed is UTF-8 whatever the console is set to, so a name in any script comes through a pipe whole.
    After the command its flags and its other words may come in any order.
    """
    _utf8(sys.stdout)
    _utf8(sys.stderr)
    parser, named = _parsers()
    argv = list(sys.argv[1:] if argv is None else argv)
    command = named.get(argv[0]) if argv else None
    if command is None:
        # No command, a call for help or a word that is no command: the whole parser has the answer to each.
        parser.parse_args(argv)
        parser.print_help(sys.stderr)
        return 2
    # Read by the whole parser, a word after a flag is lost to a command that takes a word or two: the parser
    # hands out the words it has seen when it meets the first flag. The command's own parser can be told to
    # read the flags first and the words after.
    args = command.parse_intermixed_args(argv[1:])
    try:
        return args.run(args)
    except (RowSynthError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
