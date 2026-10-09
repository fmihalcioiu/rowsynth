"""The render command: a sound or a song of a project, written as a WAV file."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..errors import RowSynthError
from ..measure import measure
from ..note import DEFAULT_RATE
from ..song import render_stems
from ..wav import save_wav
from . import emit, kind_of, load_project, mono, render_sides

NAME = "render"
HELP = "write a sound or a song of a project file as a WAV file, and give its numbers"


def configure(parser: argparse.ArgumentParser) -> None:
    """Give the parser what to render, where to write it and how."""
    parser.add_argument("file", metavar="FILE", help="the project file")
    parser.add_argument("name", metavar="NAME", help="the sound or the song to render")
    parser.add_argument("--out", metavar="PATH", required=True,
                        help="the WAV file to write: mono for a sound, stereo for a song. With --stems, a folder")
    parser.add_argument("--rate", metavar="N", type=int, default=DEFAULT_RATE,
                        help="the samples a second to render at (default %(default)s)")
    parser.add_argument("--variant", metavar="N", type=int,
                        help="which variant of a sound to write, counted from 0 (default 0)")
    parser.add_argument("--stems", action="store_true",
                        help="write a song as one mono loop a channel, each CHANNEL.wav in the folder PATH")


def _file(path: str, sides: list[np.ndarray], rate: int) -> dict:
    """Write a sound as one WAV file, one side as mono and two as stereo, and say what the file holds."""
    save_wav(path, sides if len(sides) == 2 else sides[0], rate)
    row = mono(sides)
    return {"file": path, "rate": rate, "channels": len(sides), "seconds": len(row) / rate,
            "measure": measure(row, rate)}


def _stems(song: dict, folder: str, rate: int) -> dict:
    """Write a song as one mono loop a channel, each CHANNEL.wav in the folder, and say what the files hold."""
    loops = render_stems(song, rate)
    files = []
    for stem in loops["stems"]:
        file = f"{stem['name']}.wav"
        # A folder that is missing is made by the writer, which makes whatever is above the file it writes.
        save_wav(Path(folder, file), stem["samples"], rate)
        files.append({"channel": stem["name"], "file": file, "pan": stem["pan"],
                      "measure": measure(stem["samples"], rate)})
    return {"folder": folder, "rate": rate, "loop_samples": loops["loop_samples"], "seconds": loops["seconds"],
            "files": files}


def run(args: argparse.Namespace) -> int:
    """Write the sound or the song that is named, print what was written with its numbers, and return 0.

    A sound is one variant in a mono file, at the level the bank gives it. A song is its two sides in one
    file, or with --stems its channels in a folder.
    """
    project = load_project(args.file)
    name = args.name
    kind = kind_of(project, args.file, name, args.variant)
    if args.stems:
        if kind == "sound":
            raise RowSynthError(f"--stems is for a song, and {name!r} is a sound")
        told = _stems(project.song(name), args.out, args.rate)
    else:
        told = _file(args.out, render_sides(project, kind, name, args.rate, args.variant), args.rate)
    emit(told)
    return 0
