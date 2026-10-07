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


def _parser() -> argparse.ArgumentParser:
    """The parser of the whole command line: a sub-command for every command module, set up by the module itself."""
    parser = argparse.ArgumentParser(prog="rowsynth", description="A chip synth driven by text.")
    subs = parser.add_subparsers(title="commands", metavar="COMMAND")
    for module in _command_modules():
        sub = subs.add_parser(module.NAME, help=module.HELP, description=module.HELP)
        module.configure(sub)
        sub.set_defaults(run=module.run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command that `argv` names and give its exit code. With None, `argv` is the process's own.

    A command that stops on a RowSynthError or an OSError has the message printed to standard error after
    "error: ", and the code is 2. With no command named the help goes to standard error, and the code is 2 again.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "run"):
        parser.print_help(sys.stderr)
        return 2
    try:
        return args.run(args)
    except (RowSynthError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
