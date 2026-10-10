"""Reads the GUI's art: pixelrows set files, kept as JSON inside the package, into plain data."""
from __future__ import annotations

import json
import re
from importlib import resources
from typing import NamedTuple

from ..errors import RowSynthError

#: A colour as (red, green, blue), each from 0 to 255.
Colour = tuple[int, int, int]

#: The character of an empty pixel. It is in no palette.
EMPTY = "."

_SET_KEYS = ("palette", "sprites", "anims")
_SPRITE_KEYS = ("rows", "palette")
_COLOUR = re.compile(r"#[0-9a-fA-F]{6}")


class Sprite(NamedTuple):
    """One picture: its rows from the top, one character a pixel, and the colours of its characters."""

    rows: tuple[str, ...]
    palette: dict[str, Colour]


class SpriteSet(NamedTuple):
    """What a set file holds: its palette, and its sprites by name. Its anims are not read."""

    palette: dict[str, Colour]
    sprites: dict[str, Sprite]


def load(name: str) -> SpriteSet:
    """The set in `rowsynth/gui/assets/NAME.json`. RowSynthError if there is no such file or it is not a set."""
    file = f"{name}.json"
    found = resources.files("rowsynth.gui").joinpath("assets").joinpath(file)
    if not found.is_file():
        raise RowSynthError(f"{file}: the GUI has no such set file")
    return parse(found.read_text(encoding="utf-8"), file)


def parse(text: str, file: str) -> SpriteSet:
    """The set written in `text`. A RowSynthError names `file` and the sprite that is wrong."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise RowSynthError(f"{file}: not JSON at line {error.lineno}, column {error.colno}: {error.msg}") from None
    if not isinstance(data, dict) or not set(data) <= set(_SET_KEYS):
        raise RowSynthError(f"{file}: a set file is an object with 'palette', 'sprites' and 'anims'")
    palette = _palette(data.get("palette", {}), file)
    entries = data.get("sprites", {})
    if not isinstance(entries, dict):
        raise RowSynthError(f"{file}: 'sprites' must be an object")
    sprites = {name: _sprite(entry, palette, f"{file}: sprite {name!r}") for name, entry in entries.items()}
    return SpriteSet(palette, sprites)


def _palette(entries: object, where: str) -> dict[str, Colour]:
    """A palette of a set file as characters to colours. `where` starts a refusal."""
    if not isinstance(entries, dict):
        raise RowSynthError(f"{where}: the palette must be an object")
    found: dict[str, Colour] = {}
    for key, colour in entries.items():
        if len(key) != 1 or key == EMPTY or key.isspace():
            raise RowSynthError(f"{where}: a palette key is one character that is not '.' or a space, got {key!r}")
        if not isinstance(colour, str) or not _COLOUR.fullmatch(colour):
            raise RowSynthError(f"{where}: the colour of {key!r} must be written '#rrggbb', got {colour!r}")
        found[key] = (int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16))
    return found


def _sprite(entry: object, palette: dict[str, Colour], where: str) -> Sprite:
    """One sprite of a set file, read against the set's palette with its own entries laid over it."""
    if not isinstance(entry, dict) or not set(entry) <= set(_SPRITE_KEYS):
        raise RowSynthError(f"{where}: must be an object with 'rows', and a 'palette' if it has its own")
    rows = entry.get("rows")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, str) and row for row in rows):
        raise RowSynthError(f"{where}: 'rows' must be a list of strings, one for each row of pixels")
    if len({len(row) for row in rows}) > 1:
        raise RowSynthError(f"{where}: its rows are not all the same length")
    if "palette" in entry:
        palette = {**palette, **_palette(entry["palette"], where)}
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch != EMPTY and ch not in palette:
                raise RowSynthError(f"{where}: row {y} has {ch!r} at column {x}, which is not in the palette")
    return Sprite(tuple(rows), palette)
