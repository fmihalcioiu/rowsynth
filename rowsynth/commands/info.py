"""The info command: what a project file holds, by name."""
from __future__ import annotations

import argparse

from ..errors import RowSynthError
from ..note import TICK_HZ
from ..project import Project
from ..song import lay
from . import emit, load_project

NAME = "info"
HELP = "say what a project file holds: its instruments, sounds, songs and chords"

# The key that gives a bank entry its form, and what this command calls an entry of that form.
_KINDS = {"def": "sound", "defs": "variants", "parts": "parts", "held": "held"}


def configure(parser: argparse.ArgumentParser) -> None:
    """Give the parser the one argument the command takes."""
    parser.add_argument("file", metavar="FILE", help="the project file")


def _sound(entry: dict) -> dict:
    """One bank entry: which of the four forms it has, and for variants how many there are."""
    key = next(key for key in _KINDS if key in entry)
    told = {"kind": _KINDS[key]}
    if key == "defs":
        told["variants"] = len(entry["defs"])
    return told


def _song(project: Project, name: str) -> dict:
    """One song: its channels and patterns by name, its order, how long it runs, and its layers when it has them."""
    song = project.song(name)
    try:
        _, rows = lay(song)
    except RowSynthError as error:
        raise RowSynthError(f"songs.{name}: {error}") from None
    told = {
        "channels": list(song["channels"]),
        "patterns": list(song["patterns"]),
        "order": song["order"],
        "rows": rows,
        "seconds": rows * song["row_ticks"] / TICK_HZ,
    }
    if "layers" in song:
        layers = song["layers"]
        told["layers"] = {
            "groups": list(layers["groups"]),
            "versions": list(layers["versions"]),
            "keys": layers["keys"],
        }
    return told


def run(args: argparse.Namespace) -> int:
    """Print what the project in the file holds, every name in the order the file has it, and return 0."""
    project = load_project(args.file)
    emit({
        "instruments": list(project.instruments),
        "sounds": {name: _sound(entry) for name, entry in project.sounds.items()},
        "songs": {name: _song(project, name) for name in project.songs},
        "chords": list(project.chords),
    })
    return 0
