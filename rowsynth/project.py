"""A project: one JSON file of named data, checked as it is loaded."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import NoReturn

from .errors import RowSynthError
from .notes import midi

_TABLES = ("instruments", "chords", "sounds", "songs")
_WAVES = ("pulse", "saw", "tri", "noise", "fm")
_AT = re.compile(r"@(\w+)", re.ASCII)


class Project:
    """The instruments, chords, sounds and songs of a project, by name."""

    def __init__(self) -> None:
        self.instruments: dict = {}
        self.sounds: dict = {}
        self.songs: dict = {}
        self.chords: dict = {}
        self._keys: list[str] = []

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Project):
            return NotImplemented
        return all(getattr(self, key) == getattr(other, key) for key in _TABLES)

    def sound(self, name: str) -> dict:
        """A copy of one sound with every instrument it names written out in full, ready to render."""
        if name not in self.sounds:
            raise RowSynthError(f"no sound named {name!r}")
        entry = copy.deepcopy(self.sounds[name])
        for defn in _sounds_in(entry):
            for voice in defn["voices"]:
                if isinstance(voice["inst"], str):
                    voice["inst"] = copy.deepcopy(self.instruments[voice["inst"]])
        return entry

    def song(self, name: str) -> dict:
        """A copy of one song, ready to render: its instruments are the project's with its own laid over."""
        if name not in self.songs:
            raise RowSynthError(f"no song named {name!r}")
        song = copy.deepcopy(self.songs[name])
        instruments = copy.deepcopy(self.instruments)
        for key, inst in song.get("instruments", {}).items():
            instruments[key] = copy.deepcopy(self.instruments[inst]) if isinstance(inst, str) else inst
        song["instruments"] = instruments
        return song

    def to_dict(self) -> dict:
        """The project as plain data: the tables it was loaded with, in their order, then any filled since."""
        keys = self._keys + [key for key in _TABLES if key not in self._keys and getattr(self, key)]
        return {key: copy.deepcopy(getattr(self, key)) for key in keys}

    @classmethod
    def from_dict(cls, data: dict) -> Project:
        """A project from plain data. Anything of the wrong shape is refused, with the place it sits at."""
        _project(data)
        project = cls()
        project._keys = list(data)
        for key in data:
            setattr(project, key, copy.deepcopy(data[key]))
        return project

    @classmethod
    def load(cls, path: str | Path) -> Project:
        """A project read from a JSON file in UTF-8, with or without a byte order mark."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        except UnicodeDecodeError as error:
            raise RowSynthError(f"{path}: not UTF-8 text") from error
        except json.JSONDecodeError as error:
            raise RowSynthError(f"{path}: line {error.lineno}, column {error.colno}: {error.msg}") from error
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        """Write the project as JSON, making the folders above the file when they are missing."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"
        path.write_text(text, encoding="utf-8", newline="\n")


def _fail(where: str, what: str) -> NoReturn:
    """Refuse the data: `where` is the path to the place, `what` is wrong with it."""
    raise RowSynthError(f"{where}: {what}")


def _dict(value: object, where: str) -> None:
    if not isinstance(value, dict):
        _fail(where, "must be a dict")


def _list(value: object, where: str) -> None:
    if not isinstance(value, list):
        _fail(where, "must be a list")


def _number(value: object, where: str) -> None:
    """Python counts true and false as numbers. A project does not."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(where, "must be a number")


def _numbers(value: object, where: str) -> None:
    """One number, or a list of them."""
    if isinstance(value, list):
        for i, item in enumerate(value):
            _number(item, f"{where}[{i}]")
    else:
        _number(value, where)


def _whole(value: object, where: str, low: int | None = None, high: int | None = None) -> None:
    """A whole number, from `low` to `high` when those are given."""
    span = "" if low is None else f" from {low} to {high}"
    if isinstance(value, bool) or not isinstance(value, int) or (low is not None and not low <= value <= high):
        _fail(where, f"must be a whole number{span}")


def _flag(value: object, where: str) -> None:
    if not isinstance(value, bool):
        _fail(where, "must be true or false")


def _text(value: object, where: str) -> None:
    if not isinstance(value, str):
        _fail(where, "must be a string")


def _record(value: object, where: str, shape: dict, required: tuple = ()) -> None:
    """A dict with every key of `required` and no key outside `shape`, which maps each key to its check."""
    _dict(value, where)
    for key in required:
        if key not in value:
            _fail(where, f"missing key {key!r}")
    for key, part in value.items():
        if key not in shape:
            _fail(where, f"unknown key {key!r}")
        shape[key](part, f"{where}.{key}")


def _project(data: object) -> None:
    """The whole file: its tables, then everything in them."""
    _dict(data, "project")
    for key, table in data.items():
        if key not in _TABLES:
            _fail("project", f"unknown key {key!r}")
        _dict(table, key)
    instruments = data.get("instruments", {})
    for name, inst in instruments.items():
        _instrument(inst, f"instruments.{name}")
    for name, entry in data.get("sounds", {}).items():
        _entry(entry, f"sounds.{name}", instruments)
    chords = data.get("chords", {})
    for name, chord in chords.items():
        _chord(chord, f"chords.{name}")
    for name, song in data.get("songs", {}).items():
        _song(song, f"songs.{name}", instruments, chords)


def _steps(table: object, where: str, low: int | None = None, high: int | None = None) -> None:
    """A step table: a number, a list of numbers, or {"v": list, "loop": index}.

    With `low` and `high`, every value is a whole number between them.
    """
    if isinstance(table, dict):
        _record(table, where, {"v": _list, "loop": _whole}, ("v", "loop"))
        table, where = table["v"], f"{where}.v"
    values = [(table, where)]
    if isinstance(table, list):
        values = [(value, f"{where}[{i}]") for i, value in enumerate(table)]
    for value, at in values:
        if low is None:
            _number(value, at)
        else:
            _whole(value, at, low, high)


def _wave(value: object, where: str) -> None:
    if value not in _WAVES:
        _fail(where, f"must be one of {', '.join(map(repr, _WAVES))}, not {value!r}")


def _vib(value: object, where: str) -> None:
    """A vibrato is nothing without its depth. The delay and the rate have values to fall back on."""
    _record(value, where, {"delay": _number, "depth": _number, "rate": _number}, ("depth",))


def _ops(value: object, where: str) -> None:
    """The operators of an FM voice: four at most, each with its envelope."""
    _list(value, where)
    if len(value) > 4:
        _fail(where, f"must hold four operators at most, not {len(value)}")
    env = {key: _number for key in ("a", "d", "s", "sd", "r")}
    for i, op in enumerate(value):
        _record(op, f"{where}[{i}]", {
            "mul": _number,
            "det": _number,
            "level": _number,
            "env": lambda part, at: _record(part, at, env),
        })


def _instrument(inst: object, where: str) -> None:
    """One instrument: only the keys an instrument has, each of its own kind."""
    _record(inst, where, {
        "wave": _wave,
        "vol": _steps,
        "duty": lambda table, at: _steps(table, at, 0, 3),
        "arp": _steps,
        "pitch": _steps,
        "vib": _vib,
        "slide": _number,
        "release": _number,
        "gain": _number,
        "short": _flag,
        "lp": _steps,
        "res": _number,
        "key_scale": _number,
        "gate": _number,
        "alg": lambda alg, at: _whole(alg, at, 0, 7),
        "fb": _number,
        "ops": _ops,
    })


def _inst(value: object, where: str, instruments: dict) -> None:
    """An instrument where one is used: a dict of its own, or the name of one of the project's."""
    if not isinstance(value, str):
        _instrument(value, where)
    elif value not in instruments:
        _fail(where, f"no instrument named {value!r}")


def _note(value: object, where: str) -> None:
    """A note: a number, or a name. What counts as a name is settled in one place, by the note reader."""
    try:
        midi(value)
    except RowSynthError:
        _fail(where, f"must be a number or a note name such as 'c#4', not {value!r}")


def _sound(defn: object, where: str, instruments: dict) -> None:
    """One sound: voices playing notes, its level and its echo."""
    _record(defn, where, {
        "voices": _list,
        "gain": _number,
        "echo": lambda echo, at: _record(
            echo, at, {"time": _numbers, "feedback": _number, "mix": _number, "lp": _number}, ("time",)),
    }, ("voices",))
    for i, voice in enumerate(defn["voices"]):
        at = f"{where}.voices[{i}]"
        _record(voice, at, {"inst": lambda inst, at: _inst(inst, at, instruments), "notes": _list}, ("inst", "notes"))
        for j, note in enumerate(voice["notes"]):
            spot = f"{at}.notes[{j}]"
            _list(note, spot)
            if len(note) not in (3, 4):
                _fail(spot, "must be [tick, note, ticks] or [tick, note, ticks, velocity]")
            for k, check in enumerate((_number, _note, _number, _number)[:len(note)]):
                check(note[k], f"{spot}[{k}]")


def _entry(entry: object, where: str, instruments: dict) -> None:
    """A bank entry: one sound, variants of one, parts on a timeline or a held sound, and what goes with it."""
    def sound(defn: object, at: str) -> None:
        _sound(defn, at, instruments)

    held = {"note": sound, "start": _number, "loop": _number, "join": _number, "end": sound, "let_go": _number}
    _record(entry, where, {
        "def": sound,
        "defs": _list,
        "parts": _list,
        "held": lambda value, at: _record(value, at, held, tuple(held)),
        "db": _numbers,
        "meta": _dict,
        "check": _list,
        "max_seconds": _number,
        "max_dc": _number,
        "max_bright": _number,
    })
    if sum(key in entry for key in ("def", "defs", "parts", "held")) != 1:
        _fail(where, "must hold one of 'def', 'defs', 'parts' and 'held', and only one")
    for key in ("defs", "parts"):
        if key in entry and not entry[key]:
            _fail(f"{where}.{key}", "must hold at least one")
    for i, defn in enumerate(entry.get("defs", [])):
        sound(defn, f"{where}.defs[{i}]")
    for i, part in enumerate(entry.get("parts", [])):
        at = f"{where}.parts[{i}]"
        _list(part, at)
        if len(part) != 2:
            _fail(at, "must be [seconds, entry]")
        _number(part[0], f"{at}[0]")
        _entry(part[1], f"{at}[1]", instruments)
    levels = entry.get("db")
    if isinstance(levels, list) and ("defs" not in entry or len(levels) != len(entry["defs"])):
        _fail(f"{where}.db", "a list of levels must hold one for each of 'defs'")
    for i, mark in enumerate(entry.get("check", [])):
        _record(mark, f"{where}.check[{i}]", {"at": _number, "note": _note}, ("at", "note"))


def _sounds_in(entry: dict) -> list[dict]:
    """Every sound a bank entry holds, whichever form it has."""
    found = [entry["def"]] if "def" in entry else list(entry.get("defs", []))
    for _, part in entry.get("parts", []):
        found += _sounds_in(part)
    if "held" in entry:
        found += [entry["held"]["note"], entry["held"]["end"]]
    return found


def _known(name: object, where: str, names: dict, kind: str) -> None:
    """A name that must point at one of `names`. `kind` says what those are, for the message."""
    _text(name, where)
    if name not in names:
        _fail(where, f"no {kind} named {name!r}")


def _names(value: object, where: str, names: dict, kind: str) -> None:
    """A list of names, each pointing at one of `names`."""
    _list(value, where)
    for i, name in enumerate(value):
        _known(name, f"{where}[{i}]", names, kind)


def _chord(chord: object, where: str) -> None:
    """A chord: the pitch class of its root, and the pitch classes that belong to it."""
    _record(chord, where, {"root": lambda root, at: _whole(root, at, 0, 11), "tones": _list}, ("root", "tones"))
    for i, tone in enumerate(chord["tones"]):
        _whole(tone, f"{where}.tones[{i}]", 0, 11)


def _song(song: object, where: str, shared: dict, chords: dict) -> None:
    """One song: only the keys a song has, and every name in it pointing at something."""
    _record(song, where, {
        "row_ticks": _number,
        "rows_per_beat": _number,
        "gain": _number,
        "instruments": _dict,
        "drums": lambda drums, at: _record(drums, at, {letter: _dict for letter in "kshHt"}),
        "channels": _dict,
        "patterns": _dict,
        "order": _list,
        "chords": _dict,
        "layers": _dict,
        "intro": _dict,
    }, ("row_ticks", "channels", "patterns", "order"))
    for name, own in song.get("instruments", {}).items():
        _inst(own, f"{where}.instruments.{name}", shared)
    instruments = {**shared, **song.get("instruments", {})}
    channels, patterns = song["channels"], song["patterns"]

    def inst(name: object, at: str) -> None:
        _known(name, at, instruments, "instrument")

    for letter, drum in song.get("drums", {}).items():
        _record(drum, f"{where}.drums.{letter}", {
            "inst": inst,
            "note": _note,
            "rows": _number,
            "vel": _number,
        }, ("inst", "note"))
    for name, channel in channels.items():
        _record(channel, f"{where}.channels.{name}", {
            "inst": inst,
            "gain": _number,
            "pan": _number,
            "drums": _flag,
            "echo": lambda echo, at: _record(echo, at, {"rows": _number, "gain": _number}, ("rows", "gain")),
        }, ("inst",))
    for name, pattern in patterns.items():
        at = f"{where}.patterns.{name}"
        _dict(pattern, at)
        for channel, text in pattern.items():
            if channel not in channels:
                _fail(at, f"no channel named {channel!r}")
            _text(text, f"{at}.{channel}")
            for named in _AT.findall(text):
                inst(named, f"{at}.{channel}")
    _names(song["order"], f"{where}.order", patterns, "pattern")
    for name, bars in song.get("chords", {}).items():
        if name not in patterns:
            _fail(f"{where}.chords", f"no pattern named {name!r}")
        _list(bars, f"{where}.chords.{name}")
        for i, bar in enumerate(bars):
            at = f"{where}.chords.{name}[{i}]"
            _text(bar, at)
            for chord in bar.split():
                _known(chord, at, chords, "chord")
    if "layers" in song:
        _layers(song["layers"], f"{where}.layers", channels)
    if "intro" in song:
        _record(song["intro"], f"{where}.intro", {
            "order": lambda order, at: _names(order, at, patterns, "pattern"),
            "tail": _number,
            "fade": _number,
        }, ("order",))


def _layers(layers: object, where: str, channels: dict) -> None:
    """A song's layers: groups of channels, versions made of groups, the keys, and the versions mixed ahead."""
    _record(layers, where, {
        "groups": _dict,
        "versions": _dict,
        "keys": _list,
        "premix": _list,
    }, ("groups", "versions", "keys"))
    groups, versions = layers["groups"], layers["versions"]
    for name, members in groups.items():
        _names(members, f"{where}.groups.{name}", channels, "channel")
    for name, members in versions.items():
        _names(members, f"{where}.versions.{name}", groups, "group")
    for i, key in enumerate(layers["keys"]):
        _whole(key, f"{where}.keys[{i}]")
    for i, mix in enumerate(layers.get("premix", [])):
        _record(mix, f"{where}.premix[{i}]", {
            "version": lambda version, at: _known(version, at, versions, "version"),
            "key": _whole,
        }, ("version", "key"))
