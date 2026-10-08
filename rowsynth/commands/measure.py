"""The measure command: the numbers of a WAV file."""
from __future__ import annotations

import argparse

import numpy as np

from ..errors import RowSynthError
from ..measure import brightness, measure, pitch_at
from ..wav import load_wav
from . import emit, load_file

NAME = "measure"
HELP = "give the numbers of a WAV file: its length, its level, its brightness and, when asked, its pitch"


def configure(parser: argparse.ArgumentParser) -> None:
    """Give the parser the file to measure and the time to read a pitch at."""
    parser.add_argument("wav", metavar="WAV",
                        help="the file, mono or stereo. A stereo file is measured as the mean of its two sides")
    parser.add_argument("--at", metavar="SECONDS", type=float, help="also give the pitch sounding at this time")


def run(args: argparse.Namespace) -> int:
    """Print the numbers of the file, with its pitch at the time asked for when one is, and return 0."""
    channels, rate = load_file(load_wav, args.wav)
    # One row is measured: the mean of the sides, which for a mono file is the one side it has.
    row = np.mean(channels, axis=0, dtype=np.float64)
    seconds = len(row) / rate
    told = {
        "rate": rate,
        "channels": len(channels),
        "seconds": seconds,
        "measure": measure(row, rate),
        "brightness": brightness(row, rate=rate),
    }
    if args.at is not None:
        if not 0 <= args.at < seconds:
            raise RowSynthError(f"--at must be 0 or more and under the file's {seconds} seconds, not {args.at}")
        told["pitch"] = {"at": args.at, "hz": pitch_at(row, args.at, rate=rate)}
    emit(told)
    return 0
