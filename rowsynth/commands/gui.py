"""The gui command: a window on a project file, or on the first screen where one is made or opened."""
from __future__ import annotations

import argparse

from ..gui import check_frames, start
from ..gui.theme import MIN_ART
from . import emit

NAME = "gui"
HELP = ("open a window on a project file, or with no file on the first screen: this needs pygame, which "
        "pip install rowsynth[gui] gets")


def configure(parser: argparse.ArgumentParser) -> None:
    """Give the parser the project to open, and what a run that nobody watches is told: how long, and what to keep."""
    parser.add_argument("file", metavar="FILE", nargs="?",
                        help="the project file (default: none, and the window opens on its first screen, where a "
                             "project is made or opened)")
    parser.add_argument("--scale", metavar="N", type=int,
                        help="how many pixels of the window one drawn pixel takes each way, a whole number "
                             f"(default: the largest that leaves {MIN_ART[0]} by {MIN_ART[1]} drawn pixels or more)")
    parser.add_argument("--frames", metavar="N", type=int,
                        help="close the window after this many frames, 1 or more (default: it stays until it is "
                             "closed)")
    parser.add_argument("--shot", metavar="PATH", help="write the last frame to this path as a PNG")


def run(args: argparse.Namespace) -> int:
    """Open the window, on the project in the file where one is given. When it has closed, print what was shown.

    The files opened in the window are remembered in the user's settings folder. A run for a count of frames
    is watched by nobody: it reads no such list and remembers nothing.
    """
    from ..gui.files import settings_folder

    # A count of frames that cannot be run is refused here, before the window opens.
    check_frames(args.frames)
    settings = settings_folder() if args.frames is None else None
    drawn = start(args.file, args.scale, settings).run(args.frames, args.shot)
    told = {"file": args.file, "frames": drawn}
    if args.shot is not None:
        told["shot"] = args.shot
    emit(told)
    return 0
