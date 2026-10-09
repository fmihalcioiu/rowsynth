"""The measure command: the numbers of a WAV file, or of a sound or a song of a project."""
from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np

from ..errors import RowSynthError
from ..measure import brightness, measure, pitch_at
from ..note import DEFAULT_RATE
from ..wav import load_wav
from . import emit, kind_of, load_file, load_project, mono, render_sides

NAME = "measure"
HELP = ("give the numbers of a WAV file, or of a sound or a song of a project file: its length, its level, "
        "its brightness and, when asked, its pitch")


def configure(parser: argparse.ArgumentParser) -> None:
    """Give the parser the two forms of the command: a WAV file alone, or a project file and a name in it."""
    parser.usage = ("%(prog)s [-h] WAV [--at SECONDS]\n"
                    "       %(prog)s [-h] FILE NAME [--rate N] [--variant N] [--at SECONDS]")
    parser.add_argument("file", metavar="FILE",
                        help="alone, a WAV file, mono or stereo. With NAME after it, a project file. "
                             "What has two sides is measured as their mean")
    parser.add_argument("name", metavar="NAME", nargs="?",
                        help="the sound or the song of the project file to measure. No file is written")
    parser.add_argument("--rate", metavar="N", type=int,
                        help=f"the samples a second to render the sound or the song at (default {DEFAULT_RATE})")
    parser.add_argument("--variant", metavar="N", type=int,
                        help="which variant of a sound to measure, counted from 0 (default 0)")
    parser.add_argument("--at", metavar="SECONDS", type=float, help="also give the pitch sounding at this time")


def _numbers(sides: Sequence[np.ndarray], rate: int, at: float | None, what: str) -> dict:
    """The numbers of what has these sides, with its pitch at the time `at` when that is one.

    `what` is the word for it where a time outside it is refused: "file", "sound" or "song".
    """
    row = mono(sides)
    seconds = len(row) / rate
    told = {
        "rate": rate,
        "channels": len(sides),
        "seconds": seconds,
        "measure": measure(row, rate),
        "brightness": brightness(row, rate=rate),
    }
    if at is not None:
        if not 0 <= at < seconds:
            raise RowSynthError(f"--at must be 0 or more and under the {what}'s {seconds} seconds, not {at}")
        told["pitch"] = {"at": at, "hz": pitch_at(row, at, rate=rate)}
    return told


def _wav(args: argparse.Namespace) -> dict:
    """The numbers of a WAV file. A rate or a variant is refused: a file is measured as it is."""
    for flag in ("rate", "variant"):
        if getattr(args, flag) is not None:
            raise RowSynthError(f"--{flag} is for a project's sound or song, and {args.file} with no name "
                                "after it is measured as a WAV file")
    channels, rate = load_file(load_wav, args.file)
    return _numbers(channels, rate, args.at, "file")


def _named(args: argparse.Namespace) -> dict:
    """The numbers of a sound or a song of a project file, rendered to be measured and written nowhere."""
    project = load_project(args.file)
    name = args.name
    kind = kind_of(project, args.file, name, args.variant)
    rate = DEFAULT_RATE if args.rate is None else args.rate
    sides = render_sides(project, kind, name, rate, args.variant)
    return {"name": name, "kind": kind, **_numbers(sides, rate, args.at, kind)}


def run(args: argparse.Namespace) -> int:
    """Print the numbers of the file, or of the sound or the song named after it, and return 0."""
    emit(_wav(args) if args.name is None else _named(args))
    return 0
