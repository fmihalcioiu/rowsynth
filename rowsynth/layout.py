"""The layout of a project file: JSON on one line where it fits, opened out where a file is read and changed."""
from __future__ import annotations

import json

WIDTH = 110

# The parts of a project that are opened out whatever their size. A key names a part of the dict it sits in, "*"
# names every part with no entry of its own, and what a key leads to says the same of that part's own parts.
# The top tables are here so that each entry starts a line, and a song's tables so that changing one channel of
# one pattern changes one line of the file.
_SONG = {"instruments": {}, "drums": {}, "channels": {}, "patterns": {"*": {}}, "layers": {}}
_OPEN = {"*": {}, "songs": {"*": _SONG}}


def dumps(data: dict) -> str:
    """The text of a project file for `data`, a project as plain data. `json.loads` of it gives `data` back.

    The top tables are opened out, one entry a line or more. So is every song, with its instruments, drums,
    channels, patterns and layers, and so is each pattern, one channel's text a line. Anything else is on one
    line when it ends by column `WIDTH`, not counting a comma after it, and is opened out one key or item a line
    when it does not. A text is never broken. Numbers and texts are as `json.dumps` writes them with non-ASCII
    text left as it is, keys keep their order, and the last line ends with a newline.
    """
    return _value(data, 0, 0, _OPEN) + "\n"


def _flat(value: object) -> str:
    """A value on one line."""
    return json.dumps(value, ensure_ascii=False)


def _key(key: object) -> str:
    """A dict's key as `json.dumps` writes it: it makes a text of a key that is a number, true, false or null."""
    return _flat({key: None})[1:-len(": null}")]


def _value(value: object, indent: int, lead: int, opened: dict | None) -> str:
    """JSON for `value`, which starts at column `lead` of a line indented by `indent` spaces.

    `opened` is what `_OPEN` holds for the value: None when it is opened out only if it does not fit.
    """
    if not isinstance(value, (dict, list)) or not value:
        return _flat(value)
    if opened is None:
        flat = _flat(value)
        if lead + len(flat) <= WIDTH:
            return flat
    pad = " " * (indent + 2)
    if isinstance(value, dict):
        below = opened or {}
        rows = []
        for key, part in value.items():
            name = _key(key) + ": "
            rows.append(name + _value(part, indent + 2, len(pad) + len(name), below.get(key, below.get("*"))))
        first, last = "{", "}"
    else:
        rows = [_value(item, indent + 2, len(pad), None) for item in value]
        first, last = "[", "]"
    return first + "\n" + ",\n".join(pad + row for row in rows) + "\n" + " " * indent + last
