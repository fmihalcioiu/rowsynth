"""What the window knows, with no window in it: the open project, what is picked, its render and the findings."""
from __future__ import annotations

import copy
import os
import threading
import time
from collections.abc import Callable
from numbers import Real
from pathlib import Path

import numpy as np

from .. import checks
from ..bank import render_entry
from ..commands import mono
from ..edit import NOTHING, change_file
from ..errors import RowSynthError
from ..measure import brightness, measure
from ..note import DEFAULT_RATE
from ..notes import midi
from ..project import Project
from ..song import render_loop

# The sections of the listing in the order they are shown, and the kind of the things each one holds.
KINDS = {"songs": "song", "sounds": "sound", "instruments": "instrument"}

# Every table of a project, with the kind of the things in it. The chords are in no section of the listing.
_TABLES = {**KINDS, "chords": "chord"}
# The tables whose things are named by the things of the other two.
_NAMED = ("instruments", "chords")

# The seconds waited before each new try of a save that another program stood in the way of: a fifth of a second
# in all, and then the change is refused.
SAVE_WAITS = (0.02, 0.04, 0.06, 0.08)

# The note an instrument is heard by when it is the picked thing, and the ticks that note is held for.
TEST_NOTE = "c4"
TEST_TICKS = 12

# How many of the notes `Model.tone` made last are kept as sound.
TONES = 32

# The numbers of a render, in the order they are shown.
NUMBERS = ("seconds", "peak_db", "rms_db", "dc", "clipped", "head_step", "tail_step", "above_4k", "centroid_hz")


def _section(kind: str) -> str:
    """The section that holds the things of one kind. A kind that is none of the three is refused."""
    for section, held in KINDS.items():
        if held == kind:
            return section
    raise RowSynthError(f"a thing of a project is a song, a sound or an instrument, not {kind!r}")


def _table(project: Project, kind: str) -> dict:
    """The table of a project that holds the things of one kind."""
    return getattr(project, _section(kind))


def _tag(entry: dict) -> str:
    """What a sound is listed with when its entry is more than one sound: how many variants, "parts" or "held"."""
    if "defs" in entry:
        return str(len(entry["defs"]))
    return next((form for form in ("parts", "held") if form in entry), "")


def _first(project: Project) -> tuple[str, str] | None:
    """What is picked as a project is opened: its first song, or with no song its first sound, or nothing."""
    for kind in ("song", "sound"):
        for name in _table(project, kind):
            return kind, name
    return None


def _differ(old: Project, new: Project) -> list[tuple[str, str, object, object]]:
    """What two projects do not share, as the changes that make the first of them into the second.

    Each is (table, name, was, now) for one whole thing: a song, a sound, an instrument or a chord that only one
    of the two has, or that changed. NOTHING is for a thing that is not there. With the chords every table of a
    project is looked at, so two projects that differ in nothing here are the same project.

    The changes come in an order they can be made in one at a time, the project holding after each, and taken
    back in from the last to the first: the instruments and chords that are new or changed, then the songs and
    sounds, then the instruments and chords that are gone. A song or a sound names instruments and chords, and
    none of those names a song or a sound.
    """
    found = []
    for table in _TABLES:
        before, after = getattr(old, table), getattr(new, table)
        for name in [*before, *(name for name in after if name not in before)]:
            was, now = before.get(name, NOTHING), after.get(name, NOTHING)
            if was != now:
                found.append((table, name, was, now))
    return sorted(found, key=lambda change: (2 if change[3] is NOTHING else 0) if change[0] in _NAMED else 1)


def _look(path: Path) -> tuple[int, int] | None:
    """The time a file was last written at and its size, or None for a file that cannot be looked at."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _note(inst: dict, pitch: str | float, ticks: float, rate: int) -> np.ndarray:
    """One note of an instrument as a buffer, at the level the bank gives a sound: a sound of that one note."""
    voice = {"inst": copy.deepcopy(inst), "notes": [[0, pitch, ticks]]}
    return render_entry({"def": {"voices": [voice]}}, rate)[0]


def _instrument(project: Project, name: object, song: str | None) -> tuple[tuple[str | None, str], dict]:
    """The instrument a name stands for, in a song or with no such song in the project: where it is, and it.

    Where it is, is (song, name) for one of the song's own and (None, name) for one of the project's. A name
    the song gives to an instrument of the project stands for that one. A name nobody has is refused.
    """
    own = project.songs.get(song, {}).get("instruments", {})
    found = own.get(name, name) if isinstance(name, str) else None
    if isinstance(found, dict):
        return (song, name), found
    if not isinstance(found, str) or found not in project.instruments:
        raise RowSynthError(f"no instrument named {name!r}")
    return (None, found), project.instruments[found]


def _make(project: Project, picked: tuple[str, str] | None, rate: int) -> dict:
    """One thing of a project as sound, and the numbers of that sound: {"render", "numbers"}."""
    if picked is None:
        raise RowSynthError("nothing is picked")
    kind, name = picked
    buffers = None
    if kind == "song":
        played = render_loop(project.song(name), rate)
        sides = [played["left"], played["right"]]
    elif kind == "sound":
        buffers = render_entry(project.sound(name), rate)
        sides = buffers[:1]
    else:
        sides = [_note(project.instruments[name], TEST_NOTE, TEST_TICKS, rate)]
    row = mono(sides)
    told = {**measure(row, rate), **brightness(row, rate=rate)}
    return {"render": {"sides": sides, "rate": rate, "loop": kind == "song", "seconds": len(row) / rate,
                       "buffers": [row] if buffers is None else buffers},
            "numbers": {key: told[key] for key in NUMBERS}}


def _check(project: Project, rate: int) -> list[checks.Finding]:
    """What the checks find in a project, rendered at `rate`."""
    return checks.run(project, rate)


def _within(target: str, place: str) -> bool:
    """Whether the target of a finding is a place in a project or lies in it, as "sounds.coin[1]" in "sounds.coin"."""
    return target == place or target.startswith((f"{place}.", f"{place}["))


class _Kept:
    """One thing a model makes and keeps: the result, what it was made for, the thread it is made on.

    `job` gives what a result made now would be for, its key, and what `make` is to make it from. It is called
    with `lock` held, and what is kept is read and changed with that lock held and at no other time. A result
    is kept only while its key is still the key of now. One that comes too late is dropped, so it is never taken
    for the result of another key. Where the making went wrong, what went wrong is kept in the result's place:
    it is tried once for a key, and not again.
    """

    def __init__(self, name: str, lock: threading.Lock, make: Callable[..., object],
                 job: Callable[[], tuple[tuple, tuple]]) -> None:
        """`name` is what the threads are called. `make` is called with what `job` gives beside the key."""
        self._name = name
        self._lock = lock
        self._make = make
        self._job = job
        # The last result that was made: (its key, the result, what went wrong). One of the last two is None.
        # What went wrong is the error with the traceback it had as it was caught.
        self._kept: tuple | None = None
        self._workers: dict[tuple, threading.Thread] = {}

    def start(self) -> None:
        """Begins the making on a thread of its own, unless the result is kept or already on its way."""
        with self._lock:
            key, parts = self._job()
            if key in self._workers or self._has(key):
                return
            worker = threading.Thread(target=self._work, args=(key, parts), name=self._name, daemon=True)
            worker.start()
            self._workers[key] = worker

    def ready(self) -> bool:
        """Whether `take` would now answer at once, with the result or with what went wrong."""
        with self._lock:
            return self._has(self._job()[0])

    def peek(self) -> object | None:
        """The result if it is kept. None while it is not, and where the making went wrong. Never waits."""
        with self._lock:
            return self._kept[1] if self._has(self._job()[0]) else None

    def take(self) -> object:
        """The result. Unless it is kept, it is waited for when it is on its way, and made now when it is not.

        Where the making went wrong, the error it went wrong with is raised again each time this is asked: a
        RowSynthError for what was refused, and whatever else it was for what broke.
        """
        with self._lock:
            key, parts = self._job()
            worker = self._workers.get(key)
        if worker is not None:
            worker.join()
        if not self.ready():
            self._work(key, parts)
        _, made, wrong = self._kept
        if wrong is not None:
            # Raised as it is, the error would carry one more stretch of traceback each time.
            error, trace = wrong
            raise error.with_traceback(trace)
        return made

    def drop(self) -> None:
        """Lets go of what is kept. The lock is held by whoever calls this."""
        self._kept = None

    def _has(self, key: tuple) -> bool:
        """Whether what is kept is the result of `key`."""
        return self._kept is not None and self._kept[0] == key

    def _work(self, key: tuple, parts: tuple) -> None:
        """Makes the result of `key`, and keeps it if that is still the key of now.

        An error the making ends in is kept in the result's place. A RowSynthError is a refusal, and that is all
        that is done with it. Anything else is something broken, and is raised here as well, this once: on a
        thread of its own that writes its traceback to standard error.
        """
        kept = None
        try:
            kept = (key, self._make(*parts), None)
        except Exception as error:
            kept = (key, None, (error, error.__traceback__))
            if not isinstance(error, RowSynthError):
                raise
        finally:
            with self._lock:
                self._workers.pop(key, None)
                if kept is not None and key == self._job()[0]:
                    self._kept = kept


class Model:
    """A project file that is open: what it holds, what is picked, what that renders to and what the checks find.

    One thread owns a model and calls it. Only `start_render` and `start_findings` put work on other threads. A
    render made there is kept only if the thing it was made of is still the picked one, and findings only if
    they are of the project as it is loaded now.
    """

    def __init__(self, path: str | Path, rate: int = DEFAULT_RATE,
                 wait: Callable[[float], None] = time.sleep) -> None:
        """Opens the project file at `path`, to be rendered at `rate`. What `Project.load` refuses is refused.

        `wait` is what waits a number of seconds, between two tries of a save another program stands in the way of.
        """
        self.path = Path(path)
        self.rate = rate
        # What kept the last look at the file from going well, or None when it went well.
        self.problem: str | None = None
        # What the last look found changed outside, each whole thing as (where, was, now), in an order the changes
        # can be made in one at a time. NOTHING is for a thing that is not there.
        self.changes: list[tuple[str, object, object]] = []
        self._wait = wait
        # The file is looked at before it is read, so that a change made while it is read shows at the next look.
        # A file that cannot be looked at cannot be read either, and the reading says what is wrong with it.
        self._seen = _look(self.path)
        self._project = Project.load(self.path)
        self._picked = _first(self._project)
        # How many times the file has been loaded again with something in it changed.
        self._loads = 0
        # Held wherever the picked thing, the project or anything kept of them is read or changed.
        self._lock = threading.Lock()
        # The notes `tone` made last, each by where its instrument is, its pitch and its ticks: the one asked
        # for longest ago first. Only the thread that owns the model reads or changes them.
        self._tones: dict[tuple, dict] = {}
        self._rendered = _Kept("rowsynth-render", self._lock, _make, self._render_job)
        self._found = _Kept("rowsynth-checks", self._lock, _check, self._findings_job)

    @property
    def project(self) -> Project:
        """The project as the file last gave it. A load puts another object here, so `is` tells a load from none."""
        return self._project

    @property
    def picked(self) -> tuple[str, str] | None:
        """The one thing that is selected: (kind, name), the kind "song", "sound" or "instrument". None for nothing."""
        return self._picked

    def listing(self) -> list[tuple[str, list[tuple[str, str]]]]:
        """What the project holds: [(section, [(name, tag), ...]), ...], the names in the order the file has them.

        The sections are "songs", "sounds" and "instruments", always the three and in that order. A sound whose
        entry is more than one sound has a tag: how many variants it has, as in "3", or "parts", or "held".
        Every other tag is "".
        """
        return [(section, [(name, _tag(entry) if section == "sounds" else "")
                           for name, entry in getattr(self._project, section).items()])
                for section in KINDS]

    def pick(self, kind: str, name: str) -> None:
        """Selects a song, a sound or an instrument of the project. One the project does not have is refused."""
        if name not in _table(self._project, kind):
            raise RowSynthError(f"no {kind} named {name!r}")
        with self._lock:
            self._picked = (kind, name)

    def render(self) -> dict:
        """The picked thing as sound: {"sides", "rate", "loop", "seconds", "buffers"}. Each side is a float32 array.

        A song is its seamless loop, in two sides, and "loop" is true. A sound is the first buffer of its entry.
        An instrument is one note of it, c4 held for 12 ticks, at the level the bank gives a sound. Those two
        have one side, and "loop" is false. "seconds" is how long the sides are.

        "buffers" are all the buffers `render_entry` makes of a sound, in its order: one for one sound and for
        one of parts, one a variant, and for a held sound its start, its loop and its end. The first of them is
        the one side. For a song and for an instrument it is one row, the mean of the sides.

        The render is kept for as long as its thing stays picked and the project stays as it is, so asking twice
        costs one render. One that is on its way is waited for. With nothing picked there is nothing to render,
        which is refused. A render that went wrong is tried once and no more: what the engine refused is refused
        again each time it is asked for, and an error the engine broke with is raised again as the error it was.
        """
        return self._rendered.take()["render"]

    def numbers(self) -> dict:
        """The numbers of the picked thing's render, measured on the mean of its sides.

        {"seconds", "peak_db", "rms_db", "dc", "clipped", "head_step", "tail_step", "above_4k", "centroid_hz"}, as
        `measure` and `brightness` give them. They are made with the render and kept with it.
        """
        return self._rendered.take()["numbers"]

    def tone(self, inst: str, pitch: str | float, ticks: float = TEST_TICKS, song: str | None = None) -> dict:
        """One note of an instrument as sound: {"sides", "rate"}, one side, at the level the bank gives a sound.

        `inst` is the name of an instrument: one of the song's own where `song` names a song of the project
        that has it, and else one of the project's. `pitch` is a note, a name such as "c#4" or a number, and
        it is held for `ticks`. With the test note and its ticks, the side is the render of the instrument.

        The note is made at once, on the thread that asks. The last `TONES` asked for are kept, by instrument,
        pitch and ticks, until the file is loaded again with something in it changed: one of them asked for
        again is the same sound, and costs nothing. A name no instrument has is refused, and so is a pitch
        that is no note, and whatever the engine refuses of the note.
        """
        where, found = _instrument(self._project, inst, song)
        if isinstance(ticks, bool) or not isinstance(ticks, Real):
            raise RowSynthError(f"the ticks of a note must be a number, not {ticks!r}")
        key = (where, midi(pitch), ticks)
        made = self._tones.pop(key, None)
        if made is None:
            made = {"sides": [_note(found, key[1], ticks, self.rate)], "rate": self.rate}
        # Put in again at the end, the one asked for last is the last to be let go of.
        self._tones[key] = made
        if len(self._tones) > TONES:
            del self._tones[next(iter(self._tones))]
        return made

    def start_render(self) -> None:
        """Begins the render of the picked thing on a thread of its own, unless it is kept or already on its way."""
        self._rendered.start()

    def render_ready(self) -> bool:
        """Whether `render` and `numbers` would now answer at once, with the render or with what went wrong."""
        return self._rendered.ready()

    def findings(self) -> list[checks.Finding]:
        """What the checks find in the project, rendered at the model's rate, in the order the checks give it.

        The findings are kept until the project changes, whatever is picked. Those that are on their way are
        waited for. Checks that went wrong are run once and no more: a project they refuse is refused again each
        time its findings are asked for, and an error they broke with is raised again as the error it was.
        """
        return self._found.take()

    def start_findings(self) -> None:
        """Begins the checks on a thread of their own, unless the findings are kept or already on their way."""
        self._found.start()

    def findings_ready(self) -> bool:
        """Whether `findings` would now answer at once, with the findings or with what went wrong."""
        return self._found.ready()

    def count(self) -> int | None:
        """How many findings there are, or None while they are not ready. Never waits.

        The checks give their findings no levels: each is a breach of a rule, so there is the one count. A
        project the checks refuse or break on has none: it is None with the findings ready.
        """
        found = self._found.peek()
        return None if found is None else len(found)

    def findings_for(self, kind: str, name: str) -> list[checks.Finding]:
        """The findings that are in one song, sound or instrument, in the order the checks give them. Never waits.

        A finding says where it is with its target: "sounds.coin" is in the sound coin, and so are
        "sounds.coin[1]" and "sounds.coin.parts[0]". While the findings are not ready the list is empty, as it is
        for a project the checks refuse or break on.
        """
        place = f"{_section(kind)}.{name}"
        return [finding for finding in self._found.peek() or [] if _within(finding.target, place)]

    def poll(self) -> set[tuple[str, str]]:
        """Looks at the file, loads it again when its time or its size has changed, and gives what differs.

        Each thing that differs is (kind, name): a song, a sound, an instrument or a chord that was added,
        removed or changed, the kind of a chord being "chord". With no change the set is empty. A file that
        cannot be read just now, as when another program is part-way through saving it, is no error: the model
        keeps what it had, the set is empty, and `problem` holds the message until a later look goes well.
        After a load in which something differs, the picked thing stays picked if the project still has it, and
        nothing is picked if it does not, and the kept render, numbers, findings and notes are dropped. A file
        written again with nothing in it changed drops none of them: only the order of its names is taken from it.

        `changes` then holds the same things as changes to the file: (where, was, now) for each whole thing, as
        in ("sounds.zap", what the sound was, what it is now), with NOTHING for a thing that is not there. They
        are in an order they can be made in one at a time, and taken back in from the last to the first.
        """
        self.changes = []
        seen = _look(self.path)
        if seen is not None and seen == self._seen:
            self.problem = None
            return set()
        found = self._load(seen)
        self.changes = [(f"{table}.{name}", was, now) for table, name, was, now in found]
        return {(_TABLES[table], name) for table, name, _, _ in found}

    def change(self, where: str, value: object = NOTHING, delete: bool = False) -> dict:
        """Makes one change to the file, as `change_file` makes it, and takes the file in as it then is.

        The answer is the one `change_file` gives. The project is then the one the file holds, what was kept of
        the project before is dropped as after any load, and the next `poll` finds nothing in this write. A
        change that is refused is refused here, and then the file and the model are as they were. A file that
        another program has open cannot be written: the change is tried again after each wait of `SAVE_WAITS`,
        and refused when the last try fails too.
        """
        for pause in (*SAVE_WAITS, None):
            try:
                told = change_file(self.path, where, value, delete)
                break
            except PermissionError as error:
                if pause is None:
                    raise RowSynthError(f"{self.path}: another program has the file open, so the change was not "
                                        "saved") from error
                self._wait(pause)
            except OSError as error:
                raise RowSynthError(f"{self.path}: {error.strerror}") from error
        if not told.get("same"):
            self._load(_look(self.path))
        return told

    def _load(self, seen: tuple[int, int] | None) -> list[tuple[str, str, object, object]]:
        """Reads the file, which was looked at as `seen`, and puts what it holds in the place of the project.

        What differs is given as `_differ` gives it. A file that cannot be read leaves everything as it was but
        `problem`, and nothing is given.
        """
        try:
            project = Project.load(self.path)
        except (OSError, RowSynthError) as error:
            self.problem = f"{self.path}: {error.strerror}" if isinstance(error, OSError) else str(error)
            return []
        found = _differ(self._project, project)
        with self._lock:
            self._project = project
            if found:
                self._loads += 1
                if self._picked is not None and self._picked[1] not in _table(project, self._picked[0]):
                    self._picked = None
                self._rendered.drop()
                self._found.drop()
                self._tones.clear()
        self._seen = seen
        self.problem = None
        return found

    def _render_job(self) -> tuple[tuple, tuple]:
        """What a render made now would be of, the load of the file and the picked thing, and what it is made from."""
        return (self._loads, self._picked), (self._project, self._picked, self.rate)

    def _findings_job(self) -> tuple[tuple, tuple]:
        """What findings made now would be of, the load of the file, and what they are made from."""
        return (self._loads,), (self._project, self.rate)
