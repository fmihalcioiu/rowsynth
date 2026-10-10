"""What the window knows, with no window in it: the project that is open, what is picked and what that renders to."""
from __future__ import annotations

import copy
import os
import threading
from pathlib import Path

from .. import checks
from ..bank import render_entry
from ..commands import mono
from ..errors import RowSynthError
from ..measure import brightness, measure
from ..note import DEFAULT_RATE
from ..project import Project
from ..song import render_loop

# The sections of the listing in the order they are shown, and the kind of the things each one holds.
KINDS = {"songs": "song", "sounds": "sound", "instruments": "instrument"}

# The note an instrument is heard by when it is the picked thing, and the ticks that note is held for.
TEST_NOTE = "c4"
TEST_TICKS = 12

# The numbers of a render, in the order they are shown.
NUMBERS = ("seconds", "peak_db", "rms_db", "dc", "clipped", "head_step", "tail_step", "above_4k", "centroid_hz")


def _table(project: Project, kind: str) -> dict:
    """The table of a project that holds the things of one kind. A kind that is none of the three is refused."""
    for section, held in KINDS.items():
        if held == kind:
            return getattr(project, section)
    raise RowSynthError(f"a thing of a project is a song, a sound or an instrument, not {kind!r}")


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


def _differ(old: Project, new: Project) -> set[tuple[str, str]]:
    """The things two projects do not share, each as (kind, name): what only one of them has, and what changed."""
    found = set()
    for section, kind in KINDS.items():
        before, after = getattr(old, section), getattr(new, section)
        found |= {(kind, name) for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
    return found


def _look(path: Path) -> tuple[int, int] | None:
    """The time a file was last written at and its size, or None for a file that cannot be looked at."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _make(project: Project, picked: tuple[str, str] | None, rate: int) -> dict:
    """One thing of a project as sound, and the numbers of that sound: {"render", "numbers"}."""
    if picked is None:
        raise RowSynthError("nothing is picked")
    kind, name = picked
    if kind == "song":
        played = render_loop(project.song(name), rate)
        sides = [played["left"], played["right"]]
    elif kind == "sound":
        sides = render_entry(project.sound(name), rate)[:1]
    else:
        voice = {"inst": copy.deepcopy(project.instruments[name]), "notes": [[0, TEST_NOTE, TEST_TICKS]]}
        sides = render_entry({"def": {"voices": [voice]}}, rate)[:1]
    row = mono(sides)
    told = {**measure(row, rate), **brightness(row, rate=rate)}
    return {"render": {"sides": sides, "rate": rate, "loop": kind == "song", "seconds": len(row) / rate},
            "numbers": {key: told[key] for key in NUMBERS}}


class Model:
    """A project file that is open: what it holds, the one thing of it that is picked, and what that renders to.

    One thread owns a model and calls it. Only `start_render` puts work on another thread, and what that thread
    makes is kept only if the thing it was made of is still the picked one, in the project as it was loaded then.
    """

    def __init__(self, path: str | Path, rate: int = DEFAULT_RATE) -> None:
        """Opens the project file at `path`, to be rendered at `rate`. What `Project.load` refuses is refused."""
        self.path = Path(path)
        self.rate = rate
        # What kept the last look at the file from going well, or None when it went well.
        self.problem: str | None = None
        # The file is looked at before it is read, so that a change made while it is read shows at the next look.
        # A file that cannot be looked at cannot be read either, and the reading says what is wrong with it.
        self._seen = _look(self.path)
        self._project = Project.load(self.path)
        self._picked = _first(self._project)
        # How many times the file has been loaded again. With the picked thing, it says what a render is of.
        self._loads = 0
        # Held wherever the picked thing, the project, the kept render or the threads at work are read or changed.
        self._lock = threading.Lock()
        # The last render that was made: (what it is of, the render and its numbers, the refusal). One of the
        # last two is None.
        self._kept: tuple | None = None
        self._workers: dict[tuple, threading.Thread] = {}
        self._findings: list[checks.Finding] | None = None

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
        """The picked thing as sound: {"sides", "rate", "loop", "seconds"}. Each side is a float32 array.

        A song is its seamless loop, in two sides, and "loop" is true. A sound is the first buffer of its entry.
        An instrument is one note of it, c4 held for 12 ticks, at the level the bank gives a sound. Those two
        have one side, and "loop" is false. The render is kept for as long as its thing stays picked and the file
        is not loaded again, so asking twice costs one render. One that is on its way is waited for. With nothing
        picked there is nothing to render, which is refused.
        """
        return self._made()["render"]

    def numbers(self) -> dict:
        """The numbers of the picked thing's render, measured on the mean of its sides.

        {"seconds", "peak_db", "rms_db", "dc", "clipped", "head_step", "tail_step", "above_4k", "centroid_hz"}, as
        `measure` and `brightness` give them. They are made with the render and kept with it.
        """
        return self._made()["numbers"]

    def findings(self) -> list[checks.Finding]:
        """What the checks find in the project, rendered at the model's rate. Kept until the file is loaded again."""
        if self._findings is None:
            self._findings = checks.run(self._project, self.rate)
        return self._findings

    def poll(self) -> set[tuple[str, str]]:
        """Looks at the file, loads it again when its time or its size has changed, and gives what differs.

        Each thing that differs is (kind, name): a song, a sound or an instrument that was added, removed or
        changed. With no change the set is empty. A file that cannot be read just now, as when another program
        is part-way through saving it, is no error: the model keeps what it had, the set is empty, and `problem`
        holds the message until a later look goes well. After a load the picked thing stays picked if the
        project still has it, and nothing is picked if it does not. The kept render, numbers and findings are
        dropped.
        """
        seen = _look(self.path)
        if seen is not None and seen == self._seen:
            self.problem = None
            return set()
        try:
            project = Project.load(self.path)
        except (OSError, RowSynthError) as error:
            self.problem = f"{self.path}: {error.strerror}" if isinstance(error, OSError) else str(error)
            return set()
        found = _differ(self._project, project)
        with self._lock:
            self._project = project
            self._loads += 1
            if self._picked is not None and self._picked[1] not in _table(project, self._picked[0]):
                self._picked = None
            self._kept = None
        self._seen = seen
        self._findings = None
        self.problem = None
        return found

    def start_render(self) -> None:
        """Begins the render of the picked thing on a thread of its own, unless it is kept or already on its way."""
        with self._lock:
            key = self._key()
            if key in self._workers or self._is_kept(key):
                return
            worker = threading.Thread(target=self._work, args=(key, self._project, self._picked),
                                      name="rowsynth-render", daemon=True)
            worker.start()
            self._workers[key] = worker

    def render_ready(self) -> bool:
        """Whether `render` and `numbers` would now answer at once, with the render or with its refusal."""
        with self._lock:
            return self._is_kept(self._key())

    def _key(self) -> tuple:
        """What a render made now would be of: the load of the file it is made from, and the picked thing."""
        return self._loads, self._picked

    def _is_kept(self, key: tuple) -> bool:
        """Whether the kept render is the one of `key`."""
        return self._kept is not None and self._kept[0] == key

    def _work(self, key: tuple, project: Project, picked: tuple[str, str] | None) -> None:
        """Renders what was picked as this was asked for, and keeps the render if that is still what is picked.

        A render the engine refuses is kept as its refusal, to be refused again each time it is asked for.
        Anything else that goes wrong is kept nowhere, and is left to whoever asks for the render next.
        """
        kept = None
        try:
            kept = (key, _make(project, picked, self.rate), None)
        except RowSynthError as error:
            kept = (key, None, str(error))
        finally:
            with self._lock:
                self._workers.pop(key, None)
                if kept is not None and key == self._key():
                    self._kept = kept

    def _made(self) -> dict:
        """What the picked thing came to: its render and its numbers. Unless kept, they are waited for or made now."""
        with self._lock:
            key, project, picked = self._key(), self._project, self._picked
            worker = self._workers.get(key)
        if worker is not None:
            worker.join()
        if not self.render_ready():
            self._work(key, project, picked)
        _, made, refusal = self._kept
        if refusal is not None:
            raise RowSynthError(refusal)
        return made
