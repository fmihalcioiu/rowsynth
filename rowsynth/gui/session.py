"""What happens in the window, with no window in it: the model and the player made to work together."""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..errors import RowSynthError
from .model import KINDS, Model

if TYPE_CHECKING:
    from .audio import Player

# Seconds between two looks at the file.
POLL_EVERY = 0.25
# Seconds the glow of a thing that changed outside takes to fade.
GLOW_TIME = 1.5
# How many of the things that changed outside a message names before it counts the rest.
NAMED = 3
# What is said when something is to play and there is no player.
NO_SOUND = "no sound output"


def _told(changed: set[tuple[str, str]]) -> str:
    """What is said of the things that changed outside: their names in order, three at most, and how many more."""
    names = [name for name, _ in sorted((name, kind) for kind, name in changed)]
    more = len(names) - NAMED
    return "changed outside: " + ", ".join(names[:NAMED]) + (f" and {more} more" if more > 0 else "")


class Session:
    """A project that is open and the sound it is heard by: what plays, what changed outside, what there is to say.

    One thread owns a session and calls it, `tick` once a frame. Nothing here waits for a render or for the
    checks: what is not ready is taken up by a later `tick`. What plays is always the picked thing.
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
        # Whether the picked thing is to play as soon as its render is ready.
        self._waiting = False

    @property
    def message(self) -> str:
        """The one line there is to say, "" for nothing.

        It is what keeps the file from being read, for as long as that lasts. Else it is the last of these: the
        things that changed outside, what the engine refused, that there is no sound output. A pick clears it.
        """
        return self.model.problem or self._said

    @property
    def loop(self) -> bool:
        """Whether a song goes round until it is stopped. A sound or an instrument plays once whatever this is.

        A song that plays takes a change at once, and begins again from its start to do so.
        """
        return self._loop

    @loop.setter
    def loop(self, on: bool) -> None:
        if bool(on) == self._loop:
            return
        self._loop = bool(on)
        if self._render is not None and self._render["loop"] and self._sounding():
            # The player cannot change a sound it is playing, so it is handed the song again.
            self.player.play(self._render["sides"], self._loop)

    @property
    def playing(self) -> bool:
        """Whether the picked thing sounds, or is to as soon as its render is ready."""
        return self._waiting or self._sounding()

    @property
    def seconds(self) -> float:
        """How long the picked thing is, by the render of it taken up last. 0 while there is none."""
        return 0.0 if self._render is None else self._render["seconds"]

    def tick(self) -> None:
        """A frame's share of the waiting work: a look at the file when one is due, and a render that has come."""
        now = self._clock()
        if self._looked is None or now - self._looked >= POLL_EVERY:
            self._looked = now
            self._look(now)
        if not self._taken and self.model.picked is not None and self.model.render_ready():
            self._take()
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

        None as well for a render the engine refused.
        """
        if not self.model.render_ready():
            return None
        try:
            return self.model.numbers()
        except RowSynthError:
            return None

    def finding_count(self) -> int | None:
        """How many findings the checks have for the project, or None while they run or when they refuse it."""
        return self.model.count()

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

    def _look(self, now: float) -> None:
        """Looks at the file, takes up what changed in it, and has what the model no longer keeps made again."""
        before = self.model.picked
        changed = self.model.poll()
        if changed:
            self._lit.update((thing, now) for thing in changed)
            self._said = _told(changed)
        if before is not None and self.model.picked is None:
            # The picked thing was removed outside: the first thing of the listing takes its place.
            self._drop()
            for section, things in self.model.listing():
                if things:
                    self.model.pick(KINDS[section], things[0][0])
                    break
        # A file that was loaded again has no render kept and no findings, whether or not a thing in it differs.
        # Both starts do nothing while what they would make is kept or on its way, a refusal as much as a result.
        if self.model.picked is not None:
            self._taken = self._taken and self.model.render_ready()
            self.model.start_render()
        self.model.start_findings()

    def _take(self) -> None:
        """Takes up the render the model has ready.

        It is played if that was waited for, and else put in place of a loop that plays, which goes on. What the
        engine or the player refuses is said, and stops what plays.
        """
        play, self._waiting = self._waiting, False
        self._taken = True
        try:
            made = self.model.render()
            if play:
                self.player.play(made["sides"], self._loop and made["loop"])
            elif self._loop and made["loop"] and self._sounding():
                self.player.swap(made["sides"])
        except RowSynthError as error:
            self.stop()
            self._render = None
            self._said = str(error)
            return
        self._render = made
