"""The gui command: a project file open in a window."""
from __future__ import annotations

import argparse

from ..gui import start
from ..gui.theme import MIN_ART
from . import emit

NAME = "gui"
HELP = "open a project file in a window: this needs pygame, which pip install rowsynth[gui] gets"


def configure(parser: argparse.ArgumentParser) -> None:
    """Give the parser the project to open, and what a run that nobody watches is told: how long, and what to keep."""
    parser.add_argument("file", metavar="FILE", help="the project file")
    parser.add_argument("--scale", metavar="N", type=int,
                        help="how many pixels of the window one drawn pixel takes each way, a whole number "
                             f"(default: the largest that leaves {MIN_ART[0]} by {MIN_ART[1]} drawn pixels or more)")
    parser.add_argument("--frames", metavar="N", type=int,
                        help="close the window after this many frames (default: it stays until it is closed)")
    parser.add_argument("--shot", metavar="PATH", help="write the last frame to this path as a PNG")


def run(args: argparse.Namespace) -> int:
    """Open the project in the file in a window. When the window has closed, print what was shown, and return 0."""
    drawn = start(args.file, args.scale).run(args.frames, args.shot)
    told = {"file": args.file, "frames": drawn}
    if args.shot is not None:
        told["shot"] = args.shot
    emit(told)
    return 0
