"""What the window knows, with no window in it: the open project, what is picked, its render and the findings."""
from __future__ import annotations

import copy
import json
import math
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import TypeVar

import numpy as np

from .. import checks
from ..bank import render_entry
from ..commands import mono
from ..edit import NOTHING, change_file, change_file_many, delete_in, set_in, steps
from ..errors import RowSynthError
from ..measure import brightness, measure
from ..note import DEFAULT_RATE
from ..notes import midi
from ..project import Project
from ..song import mix_stems, pan_gains, render_stems

# What a write to the file answers.
_Told = TypeVar("_Told")
# Changes to whole things of a project, each as (where, was, now), with NOTHING for a thing that is not there.
_Changes = list[tuple[str, object, object]]

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

# The seconds of sound behind a moment that `Model.levels` reads unless it is told another stretch. It is the
# stretch the player's own level reads.
WINDOW = 0.03


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


def _tables(project: Project) -> dict[str, dict]:
    """The tables of a project by their names: the tables themselves, and no copies of them."""
    return {table: getattr(project, table) for table in _TABLES}


def _differ(old: Mapping[str, dict], new: Mapping[str, dict]) -> list[tuple[str, str, object, object]]:
    """What two projects do not share, as the changes that make the first of them into the second.

    Each project is given as its tables by their names, and a table that is not among them holds nothing.
    Each change is (table, name, was, now) for one whole thing: a song, a sound, an instrument or a chord that
    only one of the two has, or that changed. NOTHING is for a thing that is not there. With the chords every
    table of a project is looked at, so two projects that differ in nothing here are the same project.

    The changes come in an order they can be made in one at a time, the project holding after each, and taken
    back in from the last to the first: the instruments and chords that are new or changed, then the songs and
    sounds, then the instruments and chords that are gone. A song or a sound names instruments and chords, and
    none of those names a song or a sound.
    """
    found = []
    for table in _TABLES:
        before, after = old.get(table, {}), new.get(table, {})
        for name in [*before, *(name for name in after if name not in before)]:
            was, now = before.get(name, NOTHING), after.get(name, NOTHING)
            if was != now:
                found.append((table, name, was, now))
    return sorted(found, key=lambda change: (2 if change[3] is NOTHING else 0) if change[0] in _NAMED else 1)


def _unmade(data: dict, told: Sequence[dict]) -> dict:
    """The data of a project as it was before changes that are in it were made: each is unmade, the last first.

    `told` is what `change_file_many` answered for the changes. Each place gets what was there as its change
    came to it, and holds nothing again where nothing was. What was taken out of a list is put in where it sat.
    """
    for one in reversed(told):
        path = steps(one["where"])
        try:
            held = data
            for step in path[:-1]:
                held = held[step]
            if "was" not in one:
                del held[path[-1]]
            elif "now" in one or isinstance(held, dict):
                # The data has a value of its own, so that unmaking an earlier change in it leaves this one be.
                held[path[-1]] = copy.deepcopy(one["was"])
            else:
                held.insert(path[-1], copy.deepcopy(one["was"]))
        except (LookupError, TypeError):
            # The place is not as the change left it: another program wrote the file again since. It stays.
            pass
    return data


def _outside(old: Project, new: Project, told: Sequence[dict]) -> tuple[_Changes, _Changes]:
    """What another program changed in a file between the last look at it and a write made to it here.

    `old` is the project as the last look gave it, `new` the one the file holds after the write, and `told`
    what `change_file_many` answered for the write. The things changed outside are those that differ between
    `old` with the write made to it and `new`. A place the write went over is no part of that: there the
    write has the last word, whatever the file held.

    They are given twice, each time as (where, was, now) for each whole thing, in the order of `_differ`.
    First as they came before the write: each thing from what `old` has to what the file had as the write
    came to it, which is what a look made just before the write would have found. Then as they come after
    it: each from what the write alone would have left to what `new` has.
    """
    meant = old.to_dict()
    for one in told:
        try:
            if "now" in one:
                # The value as a file holds it, and as `new` so has it: a tuple is a list there.
                set_in(meant, one["where"], json.loads(json.dumps(one["now"])))
            else:
                delete_in(meant, one["where"])
        except RowSynthError:
            # A place the project here did not have. The thing it is in was changed outside, and is found below.
            pass
    found = _differ(meant, _tables(new))
    if not found:
        return [], []
    looked, came = _tables(old), _unmade(new.to_dict(), told)
    return ([(f"{table}.{name}", looked[table].get(name, NOTHING), came.get(table, {}).get(name, NOTHING))
             for table, name, _, _ in found],
            [(f"{table}.{name}", was, now) for table, name, was, now in found])


def things(changes: Iterable[tuple[str, object, object]]) -> set[tuple[str, str]]:
    """The things that changes to whole things are made to, each as (kind, name), the kind of a chord "chord".

    The changes are (where, was, now) as `Model.changes` holds them, with a place such as "sounds.zap".
    """
    return {(_TABLES[table], name) for table, name in (where.split(".", 1) for where, _, _ in changes)}


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


def _channels(project: Project, picked: tuple[str, str] | None) -> list[str] | None:
    """The names of the channels of the picked thing where it is a song the project has. None where it is not."""
    kind, name = picked or (None, None)
    return list(project.songs[name]["channels"]) if kind == "song" and name in project.songs else None


def _number(value: object) -> bool:
    """Whether a value is a number to count with: a real number that is finite, and no truth value."""
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _make(project: Project, picked: tuple[str, str] | None, rate: int) -> dict:
    """One thing of a project as sound, and the numbers of that sound: {"render", "numbers", "stems"}.

    The stems are those a song is the mix of, as `render_stems` gives them. A sound or an instrument has None.
    """
    if picked is None:
        raise RowSynthError("nothing is picked")
    kind, name = picked
    buffers = stems = None
    if kind == "song":
        # The song is played once, into its stems, and the two sides are all of those mixed. So the sides are
        # sample for sample those of `song.render_loop`, and any of the stems mixed again are the song with
        # only those channels in it.
        played = render_stems(project.song(name), rate)
        stems = played["stems"]
        sides = list(mix_stems(stems, played["loop_samples"]))
    elif kind == "sound":
        buffers = render_entry(project.sound(name), rate)
        sides = buffers[:1]
    else:
        sides = [_note(project.instruments[name], TEST_NOTE, TEST_TICKS, rate)]
    row = mono(sides)
    told = {**measure(row, rate), **brightness(row, rate=rate)}
    return {"render": {"sides": sides, "rate": rate, "loop": kind == "song", "seconds": len(row) / rate,
                       "buffers": [row] if buffers is None else buffers},
            "numbers": {key: told[key] for key in NUMBERS}, "stems": stems}


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
        return self.get(key, parts)

    def get(self, key: tuple, parts: tuple) -> object:
        """The result of `key`, as `take` gives the result of now, made from `parts` where it has to be made.

        This can be asked on any thread, and of a key that is the key of now no longer: the result is then
        given, and not kept.
        """
        with self._lock:
            worker = self._workers.get(key)
        if worker is not None:
            worker.join()
        with self._lock:
            kept = self._kept if self._has(key) else None
        _, made, wrong = kept or self._work(key, parts)
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

    def _work(self, key: tuple, parts: tuple) -> tuple:
        """Makes the result of `key`, keeps it if that is still the key of now, and gives it as it is kept.

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
        return kept


def _mix(rendered: _Kept, key: tuple, parts: tuple, heard: frozenset[str]) -> dict:
    """The render of a song with only the channels in `heard` to be heard: its sides are their stems mixed.

    The song's own render is that of `key` in `rendered`, made from `parts`. It is waited for where it is on
    its way, and where it is kept nothing of it is made again. Everything of the render but its sides stays
    that of the whole song. What the song's render went wrong with is what this goes wrong with.

    The sides are made the way that goes through fewer stems. With no more stems heard than not, the heard
    ones are mixed as they are, in their order, each at the level and the pan it has in the song. With more
    heard than not, the stems that are not heard are mixed and taken out of the song's own sides, which are
    left as they are: one channel muted of many costs that one stem, and not all the others. With every
    stem heard the sides are the song's own, the same arrays.

    Sides that had stems taken out are the heard stems added up to within rounding, as every sum on either
    way is rounded to float32. The two are no further apart than a step of float32 at the loudest the stems
    can add up to, for each stem of the song and one more. On the trailer piece with its 17 stems that is
    4.3e-6 of full scale, and the furthest apart that one muted channel put them there is 1.8e-7.
    """
    whole = rendered.get(key, parts)
    render = whole["render"]
    sides, size = render["sides"], len(render["sides"][0])
    kept = [stem for stem in whole["stems"] if stem["name"] in heard]
    gone = [stem for stem in whole["stems"] if stem["name"] not in heard]
    if len(gone) >= len(kept):
        sides = list(mix_stems(kept, size))
    elif gone:
        sides = [side - away for side, away in zip(sides, mix_stems(gone, size))]
    return {**render, "sides": sides}


class Model:
    """A project file that is open: what it holds, what is picked, what that renders to and what the checks find.

    One thread owns a model and calls it. Only `start_render` and `start_findings` put work on other threads. A
    render made there is kept only if the thing it was made of is still the picked one, and findings only if
    they are of the project as it is loaded now.

    A song is kept as the stems it was rendered into, a channel each. So some of its channels can be heard
    without the others, with nothing rendered again, and each has a level of its own to show.
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
        # can be made in one at a time. NOTHING is for a thing that is not there. A write made here reads the file
        # too, and leaves here what another program changed since the look before, as it came before the write.
        self.changes: _Changes = []
        # After a write made here, the same things as they come after the write: each from what the write alone
        # would have left to what the file holds.
        self.changes_after: _Changes = []
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
        # The channels of the picked song that are heard, or None for all of them.
        self._heard: frozenset[str] | None = None
        self._rendered = _Kept("rowsynth-render", self._lock, _make, self._render_job)
        # The render with only some channels heard. It is asked for only while not all of them are.
        self._mixed = _Kept("rowsynth-mix", self._lock, _mix, self._mix_job)
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

    @property
    def heard(self) -> frozenset[str] | None:
        """The channels of the picked song that are heard, by their names, or None for all of them."""
        return self._heard

    def pick(self, kind: str, name: str) -> None:
        """Selects a song, a sound or an instrument of the project. One the project does not have is refused.

        When it is another thing than the one that was picked, all its channels are heard.
        """
        if name not in _table(self._project, kind):
            raise RowSynthError(f"no {kind} named {name!r}")
        with self._lock:
            if (kind, name) != self._picked:
                self._heard = None
            self._picked = (kind, name)

    def hear(self, only: Iterable[str] | None = None) -> None:
        """Has only some channels of the picked song be heard, or all of them again: a way of listening.

        `only` names the channels that are heard, none of them for silence, and None is for all. The sides of
        the render are then those channels alone, each as loud and as far to a side as it is in the song.
        Nothing is rendered again for it, and nothing in the file changes: the stems the song was rendered
        into are kept, and those that are heard are mixed, on a thread of its own when `start_render` begins
        it. With most channels heard, the few that are not are taken out of the mix of all instead, which is
        the same sound to within rounding and far less work. With all channels heard the render is the one of
        the whole song, which is there at once.

        A name that is no channel of the picked song is refused, and so is anything but None when no song is
        picked, and then what is heard stays as it was.
        """
        if only is not None:
            channels = _channels(self._project, self._picked)
            if channels is None:
                raise RowSynthError("no song is picked, so there are no channels to hear")
            if isinstance(only, str) or not isinstance(only, Iterable):
                raise RowSynthError(f"the channels to hear are a set of names or None, not {only!r}")
            only = list(only)
            for name in only:
                if name not in channels:
                    raise RowSynthError(f"the song {self._picked[1]!r} has no channel named {name!r}")
            only = frozenset(only)
        with self._lock:
            self._heard = only

    def render(self) -> dict:
        """The picked thing as sound: {"sides", "rate", "loop", "seconds", "buffers"}. Each side is a float32 array.

        A song is its seamless loop, in two sides, and "loop" is true. A sound is the first buffer of its entry.
        An instrument is one note of it, c4 held for 12 ticks, at the level the bank gives a sound. Those two
        have one side, and "loop" is false. "seconds" is how long the sides are.

        "buffers" are all the buffers `render_entry` makes of a sound, in its order: one for one sound and for
        one of parts, one a variant, and for a held sound its start, its loop and its end. The first of them is
        the one side. For a song and for an instrument it is one row, the mean of the sides.

        The sides of a song are its stems mixed, sample for sample what `song.render_loop` gives. While `hear`
        has only some of its channels heard, they are the mix of the stems of those alone, to within rounding
        where most are heard, and everything else of the render stays that of the whole song.

        The render is kept for as long as its thing stays picked and the project stays as it is, so asking twice
        costs one render. One that is on its way is waited for. With nothing picked there is nothing to render,
        which is refused. A render that went wrong is tried once and no more: what the engine refused is refused
        again each time it is asked for, and an error the engine broke with is raised again as the error it was.
        """
        if self._heard is None:
            return self._rendered.take()["render"]
        return self._mixed.take()

    def numbers(self) -> dict:
        """The numbers of the picked thing's render, measured on the mean of its sides.

        {"seconds", "peak_db", "rms_db", "dc", "clipped", "head_step", "tail_step", "above_4k", "centroid_hz"}, as
        `measure` and `brightness` give them. They are made with the render and kept with it. Those of a song
        are of the whole song as it is in the file, whichever of its channels are heard.
        """
        return self._rendered.take()["numbers"]

    def levels(self, seconds: float, window: float = WINDOW) -> dict[str, float]:
        """How loud each channel of the picked song is in what is heard, at a moment of its loop: 0 to 1 by name.

        A channel reads the loudest sample its stem has over the last `window` seconds up to `seconds`, times
        the larger of the two shares its pan gives it of the sides: what it puts into the louder side of the
        mix. The stretch goes back round the end of the loop, and a moment past the loop is the same moment a
        round on. A channel that is not heard reads 0, and so does one that plays nothing in the song.

        There is a reading for every channel of a picked song once its render is kept, and what is heard is
        read as it is now, mixed or not yet. For anything else the answer is empty: a sound, an instrument,
        nothing picked, a render that is not ready or went wrong. This never waits. A moment or a stretch that
        is no number of seconds is refused, and so is a stretch below 0.
        """
        if not (_number(seconds) and _number(window) and window >= 0):
            raise RowSynthError("levels are read at a moment and over a stretch, both a number of seconds, not "
                                f"{seconds!r} and {window!r}")
        whole = self._rendered.peek()
        if whole is None or whole["stems"] is None:
            return {}
        render = whole["render"]
        count, rate = len(render["sides"][0]), render["rate"]
        last = min(int(seconds % render["seconds"] * rate), count - 1)
        first = last - min(int(window * rate), count - 1)
        # A stretch that begins before the loop does is the end of the round before, and then the start of this.
        spans = [slice(first, last + 1)] if first >= 0 else [slice(first, None), slice(0, last + 1)]
        read = dict.fromkeys(self._project.songs[self._picked[1]]["channels"], 0.0)
        for stem in whole["stems"]:
            if self._heard is None or stem["name"] in self._heard:
                loudest = max(float(np.abs(stem["samples"][span]).max()) for span in spans)
                read[stem["name"]] = min(1.0, loudest * max(pan_gains(stem["pan"])))
        return read

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
        """Begins the render of the picked thing on a thread of its own, unless it is kept or already on its way.

        While only some channels of a song are heard, the mix of those is begun as well, on a thread of its
        own that waits for the song's render where that is still on its way.
        """
        self._rendered.start()
        if self._heard is not None:
            self._mixed.start()

    def render_ready(self) -> bool:
        """Whether `render` would now answer at once, with the render or with what went wrong."""
        return self._rendered.ready() and (self._heard is None or self._mixed.ready())

    def numbers_ready(self) -> bool:
        """Whether `numbers` would now answer at once, with the numbers or with what went wrong.

        The numbers are of the whole song, so they are there while a mix of some of its channels is still made.
        """
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
        What is heard of a picked song stays as it was while the song still has the channels it had, by their
        names. When one was added, removed or named anew, or the song is gone, all channels are heard.

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

        The write reads the file as it is, so what another program changed since the last look is taken in
        with it. `changes` then holds those things as `poll` leaves them, and none where there are none: each
        whole thing in which something differs that this write did not go over, from what it was at the last
        look to what the file had as the write came to it. `changes_after` holds the same things from what
        the write alone would have left to what the file holds now. The first are for a write that is a step
        of its own, to come before it, and the second for one that is not, to come after it.
        """
        return self._write(lambda: [change_file(self.path, where, value, delete)])[0]

    def change_many(self, changes: Iterable[tuple[str, object]]) -> list[dict]:
        """Makes several changes to the file as one, as `change_file_many` makes them, and takes the file in.

        Each change is (where, value), with NOTHING for what is taken out. The answer is the one
        `change_file_many` gives. It is one write and one load, and everything else is as in `change`: what is
        refused leaves the file and the model as they were, a file another program has open is waited for,
        and what was changed outside since the last look is in `changes` and `changes_after`.
        """
        changes = list(changes)
        return self._write(lambda: change_file_many(self.path, changes))

    def _write(self, write: Callable[[], list[dict]]) -> list[dict]:
        """Makes a write to the file and takes the file in as it then is. What the write answers is given.

        A write that left the file as it was is followed by no load. After any other, `changes` and
        `changes_after` hold what another program changed since the last look, as `_outside` tells the two.
        """
        self.changes, self.changes_after = [], []
        told = self._saved(write)
        old = self._project
        if not all(one.get("same") for one in told) and self._load(_look(self.path)):
            self.changes, self.changes_after = _outside(old, self._project, told)
        return told

    def _saved(self, write: Callable[[], _Told]) -> _Told:
        """What a write to the file answers, tried again after each wait of `SAVE_WAITS` while the file is not free.

        A file another program has open is refused when the last try fails too, and a file that cannot be
        written for another reason is refused at once, both in plain words.
        """
        for pause in (*SAVE_WAITS, None):
            try:
                return write()
            except PermissionError as error:
                if pause is None:
                    raise RowSynthError(f"{self.path}: another program has the file open, so the change was not "
                                        "saved") from error
                self._wait(pause)
            except OSError as error:
                raise RowSynthError(f"{self.path}: {error.strerror}") from error

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
        found = _differ(_tables(self._project), _tables(project))
        channels = _channels(self._project, self._picked)
        with self._lock:
            self._project = project
            if found:
                self._loads += 1
                if self._picked is not None and self._picked[1] not in _table(project, self._picked[0]):
                    self._picked = None
                now = _channels(project, self._picked)
                if now is None or set(now) != set(channels or ()):
                    self._heard = None
                self._rendered.drop()
                self._mixed.drop()
                self._found.drop()
                self._tones.clear()
        self._seen = seen
        self.problem = None
        return found

    def _render_job(self) -> tuple[tuple, tuple]:
        """What a render made now would be of, the load of the file and the picked thing, and what it is made from."""
        return (self._loads, self._picked), (self._project, self._picked, self.rate)

    def _mix_job(self) -> tuple[tuple, tuple]:
        """What a mix made now would be of, the render of now and the channels heard, and what it is made from."""
        key, parts = self._render_job()
        return (*key, self._heard), (self._rendered, key, parts, self._heard)

    def _findings_job(self) -> tuple[tuple, tuple]:
        """What findings made now would be of, the load of the file, and what they are made from."""
        return (self._loads,), (self._project, self.rate)
