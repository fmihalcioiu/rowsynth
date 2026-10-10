"""The page command: the listen page of a project, written into a folder."""
from __future__ import annotations

import argparse

from ..page import PAGE_RATE, write_page
from . import emit, load_project

NAME = "page"
HELP = ("write the listen page of a project file into a folder: one page with a player for each of its sounds "
        "and its songs")


def configure(parser: argparse.ArgumentParser) -> None:
    """Give the parser the project, the folder its page goes in and the rate to render at."""
    parser.add_argument("file", metavar="FILE", help="the project file")
    parser.add_argument("--out", metavar="DIR", required=True,
                        help="the folder to write into: index.html, and its audio files in a folder audio beside it")
    parser.add_argument("--rate", metavar="N", type=int, default=PAGE_RATE,
                        help="the samples a second to render at (default %(default)s)")


def run(args: argparse.Namespace) -> int:
    """Write the listen page of the project in the file, print what was written, and return 0."""
    project = load_project(args.file)
    written = write_page(project, args.out, args.rate)
    # Every sound and every song of the project is on its page: no sound has buffers that are all empty, and a
    # song is at the least its loop.
    emit({"page": written["page"], "folder": args.out, "rate": args.rate, "sounds": len(project.sounds),
          "songs": len(project.songs), "files": len(written["files"])})
    return 0
