"""The build command: songs of a project, written into a folder with the manifest that names their files."""
from __future__ import annotations

import argparse

from ..build import BUILD_RATE, build, manifest_path
from . import emit, load_project

NAME = "build"
HELP = "write songs of a project file into a folder as the files a player loads, with a manifest that names them"


def configure(parser: argparse.ArgumentParser) -> None:
    """Give the parser the project, the songs of it to build, the folder they go in and the rate to render at."""
    parser.add_argument("file", metavar="FILE", help="the project file")
    parser.add_argument("names", metavar="NAME", nargs="*",
                        help="a song to build (default: every song of the project)")
    parser.add_argument("--out", metavar="DIR", required=True,
                        help="the folder to write into: the files of each song, and the manifest beside them")
    parser.add_argument("--rate", metavar="N", type=int, default=BUILD_RATE,
                        help="the samples a second to render at (default %(default)s)")


def run(args: argparse.Namespace) -> int:
    """Build the songs that are named, or every song of the project in the file, print what was built, and return 0."""
    project = load_project(args.file)
    # With no name typed the list is empty, and to `build` an empty list is no songs at all.
    manifest = build(project, args.out, args.rate, args.names or None)
    emit({"folder": args.out, "manifest": str(manifest_path(args.out)), "rate": manifest["rate"],
          "songs": manifest["songs"]})
    return 0
