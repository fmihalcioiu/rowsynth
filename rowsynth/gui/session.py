"""What happens in the window, with no window in it: the model and the player made to work together."""
from __future__ import annotations

import copy
import re
import time
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from ..edit import NOTHING, get_in
from ..errors import RowSynthError
from .history import Change, History, same
from .model import KINDS, Model

if TYPE_CHECKING:
    from .audio import Player

# Seconds between two looks at the file.
POLL_EVERY = 0.25
# Seconds the glow of a thing that changed outside takes to fade.
GLOW_TIME = 1.5
# How many of the things that changed outside a message names before it counts the rest.
NAMED = 3
# What is said where there is no player: at the first tick, and each time something is to play.
NO_SOUND = "no sound output"
# What is said, after what it is called, of a thing another program changed in the moment it was changed here.
BOTH = "was changed outside at the same moment: this change stands, and undo brings the other back"

# A place that is an item of a list: the place of the list, and the number of the item.
_ITEM = re.compile(r"(.+)\[([0-9]+)\]")


def _told(changed: set[tuple[str, str]]) -> str:
    """What is said of the things that changed outside: their names in order, three at most, and how many more."""
    names = [name for name, _ in sorted((name, kind) for kind, name in changed)]
    more = len(names) - NAMED
    return "changed outside: " + ", ".join(names[:NAMED]) + (f" and {more} more" if more > 0 else "")


def _words(error: Exception) -> str:
    """What is said of an error: a refusal in its own words, anything else with the name of its class in front."""
    return str(error) if isinstance(error, RowSynthError) else f"{type(error).__name__}: {error}"


class Session:
    """A project that is open and the sound it is heard by: what plays, what changed outside, what there is to say.

    One thread owns a session and calls it, `tick` once a frame. Nothing here waits for a render or for the
    checks: what is not ready is taken up by a later `tick`. What plays is always the picked thing.

    The project is changed one thing at a time, with `change` and `remove`, and each change is in the file as
    soon as it is made. Every change is a step that `undo` takes back and `redo` makes again, and so is what
    another program changed in the file.
    """

    def __init__(self, model: Model, player: Player | None = None,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        """`player` is what the sound goes out through, or None for no sound. `clock` gives the time in seconds."""
        self.model = model
        self.player = player
        # Whether picking a sound or an instrument plays it.
        self.audition = True
        self._clock = clock
        self._loop = True
        # The last thing there was to say.
        self._said = ""
        # The moment each thing that changed outside began to glow at.
        self._lit: dict[tuple[str, str], float] = {}
        # The moment of the last look at the file, or None before the first.
        self._looked: float | None = None
        # The render of the picked thing that was taken up last, and whether it is the one the model has now.
        self._render: dict | None = None
        self._taken = False
        # Whether the findings the model has now were taken up.
        self._checked = False
        # Whether the picked thing is to play as soon as its render is ready.
        self._waiting = False
        # The steps made to the project, here and outside.
        self._history = History()

    @property
    def message(self) -> str:
        """The one line there is to say, "" for nothing.

        It is what keeps the file from being read, for as long as that lasts. Else it is the last of these: that
        there is no sound output, said at the first tick and each time something is to play; the things that
        changed outside; what went wrong with the render of the picked thing; what went wrong with the checks;
        why a change, or a step taken back or made again, was refused; that a change made here took the place
        of one made outside at the same moment. A pick clears it, and so does a change or a step that is made.
        """
        return self.model.problem or self._said

    @property
    def can_undo(self) -> bool:
        """Whether there is a step to take back: a change made here, or one that came from outside."""
        return self._history.can_undo

    @property
    def can_redo(self) -> bool:
        """Whether there is a step that was taken back, to make again."""
        return self._history.can_redo

    @property
    def undo_label(self) -> str:
        """What the step `undo` would take back is called, "" when there is none."""
        return self._history.undo_label

    @property
    def redo_label(self) -> str:
        """What the step `redo` would make again is called, "" when there is none."""
        return self._history.redo_label

    @property
    def loop(self) -> bool:
        """Whether a song goes round until it is stopped. A sound or an instrument plays once whatever this is.

        A song that plays takes a change at once and goes on from the place it is at: switched off, it plays to
        its end and stops there.
        """
        return self._loop

    @loop.setter
    def loop(self, on: bool) -> None:
        if bool(on) == self._loop:
            return
        self._loop = bool(on)
        if self._render is not None and self._render["loop"] and self._sounding():
            self.player.loop(self._loop)

    @property
    def playing(self) -> bool:
        """Whether the picked thing sounds, or is to as soon as its render is ready."""
        return self._waiting or self._sounding()

    @property
    def seconds(self) -> float:
        """How long the picked thing is, by the render of it taken up last. 0 while there is none."""
        return 0.0 if self._render is None else self._render["seconds"]

    @property
    def findings_refused(self) -> bool:
        """Whether the checks are through with no findings to give: they refused the project, or broke on it."""
        return self.model.findings_ready() and self.model.count() is None

    def tick(self) -> None:
        """A frame's share of the waiting work: a look at the file when one is due, and what the workers made."""
        now = self._clock()
        if self._looked is None or now - self._looked >= POLL_EVERY:
            first, self._looked = self._looked is None, now
            self._look(now, first)
        picked = self.model.picked is not None
        if not self._taken and picked and self.model.render_ready():
            self._take()
        # The findings wait for the render to be taken up: where both went wrong, it is always the words of the
        # checks that stand, whichever worker was through first.
        if not self._checked and (self._taken or not picked) and self.model.findings_ready():
            self._check()
        self._lit = {thing: lit for thing, lit in self._lit.items() if now - lit < GLOW_TIME}

    def pick(self, kind: str, name: str) -> None:
        """Picks a song, a sound or an instrument and starts its render. What was playing stops.

        A sound or an instrument is played once as soon as its render is ready, unless `audition` is off. What
        the project does not have is refused, and then nothing changes.
        """
        self.model.pick(kind, name)
        self._drop()
        self._said = ""
        self._waiting = self.audition and kind != "song" and self.player is not None
        self.model.start_render()

    def change(self, where: str, value: object, label: str, key: object = None) -> bool:
        """Puts a value at a place in the project and in its file, as a step that can be taken back.

        `where` is a place as `change_file` reads it, and `label` says in a few plain words what is changed
        there. Changes that follow one another with the same `key`, unless it is None, are one step: a knob that
        is dragged is taken back in one. A change the project's check refuses is not made: the answer is false,
        the message is the refusal, and no step is kept.

        The picked thing is rendered again and the checks are run again, as after a change from outside, and
        nothing glows. Where another program changed the same place a moment before, so that the file did not
        hold what the project here did, the change is made all the same and the message says so: taking the step
        back brings back what the other program wrote.
        """
        before = self.model.picked
        held = self._held(where)
        try:
            told = self.model.change(where, value, delete=value is NOTHING)
        except RowSynthError as error:
            self._said = str(error)
            return False
        self._said = ""
        if not told.get("same"):
            was = told.get("was", NOTHING)
            # The step has a value of its own, whatever becomes of the one that was handed in.
            now = copy.deepcopy(told["now"]) if "now" in told else NOTHING
            self._history.add([(where, was, now)], label, key)
            if not same(held, was):
                self._said = f"{label} {BOTH}"
            self._renew(before)
        return True

    def remove(self, where: str, label: str) -> bool:
        """Takes what is at a place out of the project and its file, as `change` puts a value there.

        An item of a list is taken out by putting the list there without it. Put back alone, the item would
        take the place of the one after it, so it is the whole list that the step puts back.
        """
        item = _ITEM.fullmatch(where)
        if item:
            holder, at = self._held(item[1]), int(item[2])
            if isinstance(holder, list) and at < len(holder):
                return self.change(item[1], holder[:at] + holder[at + 1:], label)
        return self.change(where, NOTHING, label)

    def undo(self) -> bool:
        """Takes the last step back, one made here or one that came from outside. False when there is none.

        The step's changes are made one by one, the last of them first. Where one is refused, those made before
        it are put back, the step stays the one to take back, the message is the refusal and the answer is false.
        """
        return self._replay(self._history.undo(), self._history.redo)

    def redo(self) -> bool:
        """Makes again the step taken back last. False when there is none, and when it is refused, as in `undo`."""
        return self._replay(self._history.redo(), self._history.undo)

    def play(self) -> None:
        """Plays the picked thing from its start: now if its render is ready, and else as soon as it is.

        With no player nothing sounds, and the message says so. With nothing picked nothing happens.
        """
        if self.player is None:
            self._said = NO_SOUND
        elif self.model.picked is not None:
            self._waiting = True
            if self.model.render_ready():
                self._take()
            else:
                self.model.start_render()

    def stop(self) -> None:
        """Stops what plays, and gives up a play that waited for its render."""
        self._waiting = False
        if self.player is not None:
            self.player.stop()

    def toggle(self) -> None:
        """Stops what plays, or plays the picked thing when nothing does."""
        if self.playing:
            self.stop()
        else:
            self.play()

    def position(self) -> float:
        """Seconds into what plays, as the player tells it. 0 with no player."""
        return 0.0 if self.player is None else self.player.position()

    def level(self) -> tuple[float, float]:
        """How loud the left side and the right are now, 0 to 1. Zeros with no player or with nothing playing."""
        return (0.0, 0.0) if self.player is None else self.player.level()

    def glow(self, kind: str, name: str) -> float:
        """How strongly a thing that changed outside glows now: 1 as it changes, fading to 0 over `GLOW_TIME`."""
        lit = self._lit.get((kind, name))
        return 0.0 if lit is None else max(0.0, 1.0 - (self._clock() - lit) / GLOW_TIME)

    def glows(self) -> dict[tuple[str, str], float]:
        """Every thing that glows now, as (kind, name), with how strongly: each above 0."""
        glows = {thing: self.glow(*thing) for thing in self._lit}
        return {thing: glow for thing, glow in glows.items() if glow > 0}

    def numbers(self) -> dict | None:
        """The numbers of the picked thing as the model gives them, or None while its render is not ready.

        None as well for a render that went wrong, refused or broken.
        """
        if not self.model.render_ready():
            return None
        try:
            return self.model.numbers()
        except Exception:
            # Whatever the render went wrong with, which the model raises again each time it is asked.
            return None

    def finding_count(self) -> int | None:
        """How many findings the checks have for the project, or None while the checks run.

        0 as well where the checks refused the project or broke on it, which `findings_refused` tells from a
        project with nothing found.
        """
        return 0 if self.findings_refused else self.model.count()

    def close(self) -> None:
        """Stops the sound and closes the player."""
        self.stop()
        if self.player is not None:
            self.player.close()

    def _sounding(self) -> bool:
        """Whether the player has a sound going."""
        return self.player is not None and self.player.playing

    def _drop(self) -> None:
        """Stops what plays and lets go of its render, as the picked thing is no longer what it was."""
        self.stop()
        self._render = None
        self._taken = False

    def _look(self, now: float, first: bool) -> None:
        """Looks at the file and takes up what changed in it. `first` is for the first look there is.

        What changed glows, is named in the message, and is a step that can be taken back. The render of the
        picked thing and the checks are started at the first look, and again after each change.
        """
        if first and self.player is None:
            self._said = NO_SOUND
        before = self.model.picked
        changed = self.model.poll()
        if changed:
            self._lit.update((thing, now) for thing in changed)
            self._said = _told(changed)
            self._history.add(self.model.changes, self._said)
        if changed or first:
            self._renew(before)

    def _renew(self, before: tuple[str, str] | None) -> None:
        """Starts the render of the picked thing and the checks, at the first look and after a change to the project.

        `before` is what was picked before the change. The model keeps a render and the findings for as long as
        nothing in the project differs, so with nothing let go of nothing is made again.
        """
        # What the model has let go of, with the project as it was, is to be taken up when it is there again.
        self._taken = self._taken and self.model.render_ready()
        self._checked = self._checked and self.model.findings_ready()
        if before is not None and self.model.picked is None:
            # The picked thing was removed: the first thing of the listing takes its place.
            self._drop()
            for section, things in self.model.listing():
                if things:
                    self.model.pick(KINDS[section], things[0][0])
                    break
        if self.model.picked is not None:
            self.model.start_render()
        self.model.start_findings()

    def _held(self, where: str) -> object:
        """What the project holds at a place, as it is known here. NOTHING for a place that holds nothing."""
        try:
            return get_in(self.model.project.to_dict(), where)
        except RowSynthError:
            return NOTHING

    def _put(self, where: str, value: object) -> str:
        """Puts a value at a place through the model, or with NOTHING takes out what is there.

        The answer is "" when that is done, and the refusal in its own words when it is not.
        """
        try:
            self.model.change(where, value, delete=value is NOTHING)
        except RowSynthError as error:
            return str(error)
        return ""

    def _replay(self, changes: Iterable[Change] | None, back: Callable[[], object]) -> bool:
        """Makes the changes of a step the history has just moved, one by one. None is for no step, and false.

        `back` moves the step to where it was. Where a change is refused, those made before it are put back,
        the step is moved back, the message is the refusal and the answer is false.
        """
        if changes is None:
            return False
        before = self.model.picked
        made: list[tuple[str, object]] = []
        refused = ""
        for where, was, now in changes:
            refused = self._put(where, now)
            if refused:
                break
            made.append((where, was))
        if refused:
            for where, was in reversed(made):
                self._put(where, was)
            back()
        self._said = refused
        if made:
            self._renew(before)
        return not refused

    def _take(self) -> None:
        """Takes up the render the model has ready.

        It is played if that was waited for, and else put in place of a loop that plays, which goes on. A render
        that went wrong is said, and so is what the player refuses, and either stops what plays.
        """
        play, self._waiting = self._waiting, False
        self._taken = True
        try:
            made = self.model.render()
        except Exception as error:
            # What the engine refused, or what it broke with on its worker, where the traceback was written.
            self._fail(error)
            return
        try:
            if play:
                self.player.play(made["sides"], self._loop and made["loop"])
            elif self._loop and made["loop"] and self._sounding():
                self.player.swap(made["sides"])
        except RowSynthError as error:
            self._fail(error)
            return
        self._render = made

    def _fail(self, error: Exception) -> None:
        """Says what kept the picked thing from being heard, and stops what plays."""
        self.stop()
        self._render = None
        self._said = _words(error)

    def _check(self) -> None:
        """Takes up the findings the model has ready: where the checks refused the project or broke, that is said."""
        self._checked = True
        try:
            self.model.findings()
        except Exception as error:
            self._said = _words(error)
