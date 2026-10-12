"""What the standalone program runs: the command line as it is, and the window with no console behind it."""
from __future__ import annotations

import ctypes
import io
import os
import sys
from collections.abc import Sequence

from .cli import main

#: What the two programs of the standalone folder are called, without the ending the system gives them.
COMMAND_LINE = "rowsynth"
WINDOW = "rowsynth-window"
#: What the box of the system is headed that says why the window did not open.
HEADING = "rowsynth"
_STOP_SIGN = 0x10


def frozen() -> bool:
    """Whether this is the standalone program, which has Python and the package inside it."""
    return bool(getattr(sys, "frozen", False))


def alone_in_console() -> bool:
    """Whether the process has a console that no other process shares: a double click started it, and no shell."""
    if sys.platform != "win32":
        return False
    try:
        listed = (ctypes.c_uint * 2)()
        return ctypes.windll.kernel32.GetConsoleProcessList(listed, 2) == 1
    except (AttributeError, OSError):
        return False


def say(words: str) -> None:
    """Tell the person something where there is no console to print it on: in a small box of the system's.

    Anywhere but on Windows, and where the box cannot be opened, nothing is said.
    """
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.user32.MessageBoxW(None, words, HEADING, _STOP_SIGN)
    except (AttributeError, OSError):
        pass


def command_line(argv: Sequence[str] | None = None) -> int:
    """The command line of the standalone program: what `python -m rowsynth` runs, with the same words after it.

    Started by a double click with nothing after its name, it has a console of its own and nobody to read it:
    it lets go of the console and opens the window.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv and frozen() and alone_in_console():
        ctypes.windll.kernel32.FreeConsole()
        # What the process printed on is gone with the console.
        sys.stdout = sys.stderr = open(os.devnull, "w", encoding="utf-8")
        return window([])
    return main(argv)


def window(argv: Sequence[str] | None = None) -> int:
    """Open the window, as `rowsynth gui` does with the same words after it: a file to open, or nothing.

    This is what the program with no console runs. What `main` prints as an error is shown in a box of the
    system's, since nobody could read it anywhere else. The answer is the exit code of the command.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    before, heard = (sys.stdout, sys.stderr), io.StringIO()
    # A program with no console has no standard output either: what the command prints goes nowhere.
    nowhere = open(os.devnull, "w", encoding="utf-8") if sys.stdout is None else None
    if nowhere is not None:
        sys.stdout = nowhere
    sys.stderr = heard
    try:
        code = main(["gui", *argv])
    except SystemExit as stopped:
        # The words after the name were not ones the command takes: its reader has said so, and stops.
        code = stopped.code if isinstance(stopped.code, int) else 2
    finally:
        sys.stdout, sys.stderr = before
        if nowhere is not None:
            nowhere.close()
    words = heard.getvalue().strip()
    if code and words:
        say(words)
    return code
