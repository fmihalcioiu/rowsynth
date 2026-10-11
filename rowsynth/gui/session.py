"""What happens in the window, with no window in it: the model and the player made to work together."""
from __future__ import annotations

import copy
import math
import re
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from ..edit import NOTHING, delete_in, get_in, set_in
from ..errors import RowSynthError
from ..sound import tick_sample
from .history import Change, History, same
from .model import KINDS, Model, things

if TYPE_CHECKING:
    from .audio import Player

# Seconds between two looks at the file.
POLL_EVERY = 0.25
# Seconds the glow of a thing that changed outside takes to fade.
GLOW_TIME = 1.5
# How many of the things that changed outside a message names before it counts the rest.
NAMED = 3
# Seconds a notice of what another program did has the status line for, whatever is picked or said meanwhile.
# It is the time the status bar gives a message.
NOTICE_TIME = 3.0
# What is said where there is no player: at the first tick, and each time something is to play.
NO_SOUND = "no sound output"
# What is said, after what it is called, of a thing another program changed in the moment it was changed here.
BOTH = "was changed outside at the same moment: this change stands, and undo brings the other back"

# The seconds of its loop a held sound is given when `Session.hold` plays it. Held for longer, it is let go
# by itself.
HOLD = 10.0

# A place that is an item of a list: the place of the list, and the number of the item.
_ITEM = re.compile(r"(.+)\[([0-9]+)\]")

# The keys of the computer that play notes, by the names pygame has for them, each with its note: the row
# from z up and the keys above it are the twelve notes from c, as are the row from q up and the digits above
# it an octave higher, which runs on into a third octave from i.
_PIANO = {key: note for note, key in enumerate("zsxdcvgbhnjm" "q2w3er5t6y7u" "i9o0p")}


def piano(key: str) -> int | None:
    """The note a key of the computer plays, laid out as trackers lay a piano on the keys.

    `key` is pygame's name of the key, such as "z". The note is in semitones above the c of a base octave:
    0 to 11 for z s x d c v g b h n j m, 12 to 23 for q 2 w 3 e r 5 t 6 y 7 u, and 24 to 28 for i 9 o 0 p.
    None for a key that plays no note.
    """
    return _PIANO.get(key)


def joined(start: np.ndarray, loop: np.ndarray, end: np.ndarray, let_go: int, rounds: int = 1) -> np.ndarray:
    """The three buffers of a held sound as one row: held for `rounds` of its loop, and then let go.

    `let_go` is the samples the letting go takes. The row is the start, the loop `rounds` times over, and
    then the end with the loop under it, going round once more and brought down to nothing over `let_go`.
    """
    # How the three are joined for the mixer, which plays one buffer and can have only the whole of it go
    # round. A start played once before a loop that goes round cannot be asked of it, and a sound begun as
    # another ends meets it up to a block out of step. So the sound is laid out as it is heard when it is
    # held for a time and then let go, and that one row is played once: every join in it is to the sample.
    # The end is laid over the loop and not after it. The end is played when the sound is let go, and
    # "let_go" is the ticks a player takes to let the loop go: here the loop goes on round under the end,
    # down to nothing in a straight line. Set after a loop cut off at its full level, the end would open
    # with a step that is not heard in use.
    going = np.resize(loop, let_go) * np.linspace(1.0, 0.0, let_go, endpoint=False)
    tail = np.zeros(max(len(end), let_go), dtype=np.float32)
    tail[:len(end)] = end
    tail[:let_go] += going
    return np.concatenate([start, np.tile(loop, rounds), tail])


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
    checks: what is not ready is taken up by a later `tick`. What plays is always the picked thing, and single
    notes can be sounded over it.

    The project is changed one thing at a time, with `change` and `remove`, or several things as one with
    `change_many`, and each change is in the file as soon as it is made. Every change is a step that `undo`
    takes back and `redo` makes again, and so is what another program changed in the file.

    The channels of the picked song can be muted and soloed, with `mute` and `solo`, and the song can be heard
    in another key, with `key`. That is a way of listening and no change: nothing of it is in the file, it is
    no step, and another thing picked has none of it.
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
        # The last thing there was to say, and how many times there has been something to say.
        self._said = ""
        self._messages = 0
        # The notice of what another program did that has the status line, "" for none, and the moment it has
        # had its time by.
        self._notice = ""
        self._until = 0.0
        # The notices that wait for their turn: words, or the things that changed outside, which are named as
        # their turn comes.
        self._notices: deque[str | set[tuple[str, str]]] = deque()
        # Whether what was said last has yet to have the status line: it was said while a notice had it.
        self._unshown = False
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
        # How it is to play: the variant that was asked for, None for the one whose turn it is, and whether a
        # held sound is to be held.
        self._which: int | None = None
        self._holding = False
        # The variant of the picked sound whose turn it is next.
        self._turn = 0
        # What `release` lets a held sound go by while `hold` has it sounding: the seconds into what plays at
        # which it is let go by itself, the seconds its loop takes to die away, and its end as the sides of a sound.
        self._let: tuple[float, float, Sequence[np.ndarray]] | None = None
        # The channels of the picked song that are muted, and those that are soloed, by their names.
        self._muted: frozenset[str] = frozenset()
        self._soloed: frozenset[str] = frozenset()
        # The steps made to the project, here and outside.
        self._history = History()

    @property
    def message(self) -> str:
        """The one line there is to say, "" for nothing.

        It is what keeps the file from being read, for as long as that lasts. Else it is the last of these: that
        there is no sound output, said at the first tick and each time something is to play; the things that
        changed outside; what went wrong with the render of the picked thing; what went wrong with the checks;
        why a change, or a step taken back or made again, was refused; that a change made here took the place
        of one made outside at the same moment; why a single note did not sound; why a mute, a solo or a key
        was refused; what `say` was given. A pick clears it, and so does a change or a step that is made.
        """
        return self.model.problem or self._said

    @property
    def status(self) -> str:
        """The one line for a status bar, "" for nothing: `message`, but for the notices of what another program did.

        Two things said are such notices: the things that changed outside, and that a change made here took
        the place of one made outside at the same moment. A notice has the line for `NOTICE_TIME` seconds
        from the moment it is given, whatever is picked, said or cleared meanwhile. One that comes while
        another has the line waits, and they have their turns one after the other, each for as long. Those
        that wait to name what changed outside are one notice, which names it all. With the notices through,
        the line is `message` again, as it is by then. What keeps the file from being read comes before
        any notice.
        """
        return self.model.problem or self._notice or self._said

    @property
    def messages(self) -> int:
        """How many times there has been something to say: one more each time `message` is given words.

        Words that are the same as the last are counted as well, so whoever shows the message can tell that it
        was said again. What keeps the file from being read counts once for as long as it lasts.

        It is one more as well each time `status` is given something that had to wait for it: a notice whose
        turn has come, and what was said while the notices had the line, once they are through. So whoever
        shows `status` and gives a line its time from when this last grew shows each of them for that time.
        """
        return self._messages

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
    def muted(self) -> frozenset[str]:
        """The channels of the picked song that are muted, by their names."""
        return self._muted

    @property
    def soloed(self) -> frozenset[str]:
        """The channels of the picked song that are soloed, by their names."""
        return self._soloed

    @property
    def shift(self) -> int:
        """The semitones up from how it is written that the picked song is heard at, as `key` set them: 0 at first."""
        return self.model.shift

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
        self._notify(now)
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
        the project does not have is refused, and then nothing changes. Another thing than the one that was
        picked has nothing muted and nothing soloed, and is heard as it is written.
        """
        before = self.model.picked
        self.model.pick(kind, name)
        if self.model.picked != before:
            self._muted = self._soloed = frozenset()
        self._drop()
        self.say("")
        self._waiting = self.audition and kind != "song" and self.player is not None
        self.model.start_render()

    def say(self, words: str) -> None:
        """Puts words in `message`, as the last thing there is to say. With "" there is nothing to say.

        It is for whoever works with the session and has something to say that the session cannot know, as a
        panel that could not save what was done in it. Words that are not empty are one more of `messages`.
        """
        self._said = words
        if words:
            self._messages += 1
        self._unshown = bool(words and self._notice)

    def change(self, where: str, value: object, label: str, key: object = None) -> bool:
        """Puts a value at a place in the project and in its file, as a step that can be taken back.

        `where` is a place as `change_file` reads it, and `label` says in a few plain words what is changed
        there. Changes that follow one another with the same `key`, unless it is None, are one step: a knob that
        is dragged is taken back in one. A change the project's check refuses is not made: the answer is false,
        the message is the refusal, and no step is kept.

        The picked thing is rendered again and the checks are run again, as after a change from outside, and
        nothing of the change glows. Where another program changed the same place a moment before, so that the
        file did not hold what the project here did, the change is made all the same and the message says so:
        taking the step back brings back what the other program wrote.

        What another program changed elsewhere since the last look at the file comes in with the change, and
        is taken up as a look takes it up: it glows, it is named in the message, and it is a step of its own.
        That step comes before this one, so this change is taken back first and the other program's after it.
        """
        before = self.model.picked
        held = self._held(where)
        try:
            told = self.model.change(where, value, delete=value is NOTHING)
        except RowSynthError as error:
            self.say(str(error))
            return False
        self._made([told], [held], label, key, before)
        return True

    def change_many(self, changes: Iterable[tuple[str, object]], label: str) -> bool:
        """Makes several changes to the project and its file as one, and as one step that is taken back in one.

        Each change is (where, value), as `change` is given the two, with NOTHING for what is taken out. They
        are made in the order given and checked together, as `change_file_many` does it, so things that name
        one another can be changed where no one of the changes could be made alone. The step holds every one
        of the places. Refused, none of them is made: the answer is false, the message is the refusal, and no
        step is kept. Everything else is as in `change`.
        """
        changes = list(changes)
        before = self.model.picked
        # What is here at each place as its change comes to it, the changes before it made.
        data, held = self.model.project.to_dict(), []
        for where, value in changes:
            try:
                held.append(delete_in(data, where) if value is NOTHING
                            else set_in(data, where, copy.deepcopy(value)))
            except RowSynthError:
                held.append(NOTHING)
        try:
            told = self.model.change_many(changes)
        except RowSynthError as error:
            self.say(str(error))
            return False
        self._made(told, held, label, None, before)
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

        The step's changes are made as one, the last of them first, in one write to the file. Where that is
        refused nothing is changed, the step stays the one to take back, the message is the refusal and the
        answer is false.
        """
        return self._replay(self._history.undo(), self._history.redo)

    def redo(self) -> bool:
        """Makes again the step taken back last. False when there is none, and when it is refused, as in `undo`."""
        return self._replay(self._history.redo(), self._history.undo)

    def play(self, which: int | None = None) -> None:
        """Plays the picked thing from its start: now if its render is ready, and else as soon as it is.

        A sound with variants plays one of them: the one `which` counts to, from 0 and round again past the
        last, and with None the next one each time, so that playing again and again hears them all. A held
        sound plays its start, its loop once, and its end as it is let go: see `joined`. For anything else
        `which` changes nothing. With no player nothing sounds, and the message says so. With nothing picked
        nothing happens.
        """
        self._begin(which, False)

    def hold(self) -> None:
        """Plays the picked thing as `play` does, and a held sound as it is while it is held, until `release`.

        The held sound plays its start and then its loop, round and round. It is one row that is played, with
        `HOLD` seconds of the loop in it: held for longer, the sound is let go by itself.
        """
        self._begin(None, True)

    def release(self) -> None:
        """Lets go of the held sound that `hold` has sounding: its loop dies away while its end is heard.

        The loop takes the ticks of its "let_go", and the end is played over it at once. A hold that still
        waits for its render becomes a play. With no held sound held, nothing happens, and nothing either once
        the sound was held for so long that it was let go by itself.
        """
        let, self._let, self._holding = self._let, None, False
        if let is not None and self._sounding() and self.player.position() < let[0]:
            self.player.stop(let[1])
            self.player.also(let[2])

    def sound(self, inst: str | dict, pitch: str | float, song: str | None = None) -> None:
        """Sounds one note of an instrument at once, over whatever plays, which goes on.

        `inst` is a name, looked for as `Model.tone` looks: among the instruments of `song` first, if it is a
        song of the project. Or it is the instrument itself, a dict, as a voice of a sound has one that is
        written in place. With no player nothing happens. A note that is refused, as one of an instrument
        nobody has or of one the engine refuses, does not sound, and the message is the refusal.
        """
        if self.player is None:
            return
        try:
            self.player.also(self.model.tone(inst, pitch, song=song)["sides"])
        except RowSynthError as error:
            self.say(str(error))

    def drum(self, letter: str, song: str) -> None:
        """Sounds what a drum letter plays in a song, as `sound` does: the instrument and the note its drums give it.

        A letter the song has no drum for sounds nothing, and the message says so.
        """
        hit = self.model.project.songs.get(song, {}).get("drums", {}).get(letter)
        if hit is None:
            self.say(f"no drum {letter!r} in the song {song!r}")
        else:
            self.sound(hit["inst"], hit["note"], song)

    def mute(self, channel: str) -> bool:
        """Mutes a channel of the picked song, or has a muted one heard again: a way of listening.

        What is heard is as on a mixer. With any channel soloed it is the soloed ones and no others, and with
        none soloed it is all but the muted ones. Nothing in the file changes and nothing is rendered again:
        the model mixes what is heard from the stems it keeps, on a thread of its own. A loop that plays
        takes the mix up as soon as it is there, and goes on from the place it is at. With nothing playing, it
        is what the next play sounds like. A song that plays once is left to end as it was.

        What is no channel of the picked song is refused, and so is any channel while no song is picked: the
        answer is false, the message says why, and what is muted and what is heard stay as they were.
        """
        return self._press(channel, False)

    def solo(self, channel: str) -> bool:
        """Solos a channel of the picked song, or takes its solo off. It is heard and refused as a mute is.

        While any channel is soloed the mutes are kept and count for nothing: a channel that is both is
        heard. They count again when the last solo is taken off.
        """
        return self._press(channel, True)

    def hear_all(self) -> None:
        """Takes every mute and every solo off, so that all channels of the picked song are heard."""
        self._muted = self._soloed = frozenset()
        self._listen()

    def key(self, semis: int) -> bool:
        """Has the picked song heard in another key, so many semitones up from how it is written: a way of listening.

        0 is the song as written. The model renders the song in the key as a build does, its drum channels
        playing what they played, on a thread of its own. Nothing in the file changes, and what is muted and
        soloed stays so. The key is taken up as a mute is: a loop that plays goes on in it from the place it is
        at, as soon as its render is there, and with nothing playing it is what the next play sounds like.

        Refused are a key while no song is picked and one that is no whole number: the answer is false, the
        message says why, and the key stays as it was. What the engine refuses of the song in the key, as a
        note it has no room for, is said as any render that went wrong, and stops what plays.
        """
        before = self.model.shift
        try:
            self.model.shift = semis
        except RowSynthError as error:
            self.say(str(error))
            return False
        if self.model.shift != before:
            self._follow()
        return True

    def levels(self) -> dict[str, float]:
        """How loud each channel of the picked song is at the place it plays at, 0 to 1 by name: for the meters.

        The levels are read as `Model.levels` reads them: a channel that is not heard reads 0, from the moment
        it is muted. There are none, an empty answer, while the song does not sound, and for anything that is
        no song.
        """
        return self.model.levels(self.player.position()) if self._sounding() else {}

    def stop(self) -> None:
        """Stops what plays, and gives up a play that waited for its render. What `sound` began rings out."""
        self._waiting, self._which, self._holding, self._let = False, None, False, None
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

        Those of a song are of the whole song, whatever is muted, and they stay while a mix of what is heard
        is made. None as well for a render that went wrong, refused or broken.
        """
        if not self.model.numbers_ready():
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
        self._turn = 0

    def _channels(self) -> frozenset[str]:
        """The names of the channels of the picked song. None of them where what is picked is no song."""
        kind, name = self.model.picked or (None, None)
        return frozenset(self.model.project.songs[name]["channels"]) if kind == "song" else frozenset()

    def _press(self, channel: str, solo: bool) -> bool:
        """Flips whether a channel of the picked song is soloed, or else muted, and has that heard.

        False for what is refused, which the message then says: what is no channel of the picked song.
        """
        kind, name = self.model.picked or (None, None)
        if kind != "song":
            self.say("no song is picked, so there are no channels to mute or solo")
            return False
        if not isinstance(channel, str) or channel not in self._channels():
            self.say(f"the song {name!r} has no channel named {channel!r}")
            return False
        if solo:
            self._soloed ^= {channel}
        else:
            self._muted ^= {channel}
        self._listen()
        return True

    def _hear(self) -> bool:
        """Has the model hear the picked song by its mutes and solos. True when that is not what it heard before.

        A channel the picked song does not have is muted and soloed no longer: one that is gone from the file.
        """
        channels = self._channels()
        self._muted &= channels
        self._soloed &= channels
        only = self._soloed or (channels - self._muted if self._muted else None)
        if only == self.model.heard:
            return False
        self.model.hear(only)
        return True

    def _listen(self) -> None:
        """Has the mutes and solos heard: by the model now, and by what plays as soon as the mix is there.

        Where what is heard stays as it was, as with a channel muted while another is soloed, nothing is taken
        up again.
        """
        if self._hear():
            self._follow()

    def _follow(self) -> None:
        """Has what plays follow another way of listening to the picked song: other channels, or another key.

        The render of it is begun, and is taken up here where the model has it at once, and else by a later `tick`.
        """
        self._taken = False
        self.model.start_render()
        if self.model.render_ready():
            self._take()

    def _begin(self, which: int | None, holding: bool) -> None:
        """Has the picked thing play as soon as its render is there, in the way `play` or `hold` asked for."""
        if self.player is None:
            self.say(NO_SOUND)
        elif self.model.picked is not None:
            self._waiting, self._which, self._holding = True, which, holding
            if self.model.render_ready():
                self._take()
            else:
                self.model.start_render()

    def _heard(self, made: dict) -> tuple[Sequence[np.ndarray], bool]:
        """What the player is given of a render that is to play, and whether that is to go round."""
        self._let = None
        kind, name = self.model.picked
        entry = self.model.project.sounds[name] if kind == "sound" else {}
        buffers, rate = made["buffers"], made["rate"]
        if "defs" in entry:
            at = (self._turn if self._which is None else self._which) % len(buffers)
            self._turn = at + 1
            return buffers[at:at + 1], False
        if "held" in entry:
            let_go, rounds = tick_sample(entry["held"]["let_go"], rate), 1
            if self._holding:
                rounds = math.ceil(HOLD * rate / len(buffers[1]))
                held = (len(buffers[0]) + rounds * len(buffers[1])) / rate
                self._let = (held, let_go / rate, buffers[2:])
            return [joined(*buffers, let_go, rounds)], False
        return made["sides"], self._loop and made["loop"]

    def _look(self, now: float, first: bool) -> None:
        """Looks at the file and takes up what changed in it. `first` is for the first look there is.

        What changed glows, is named in the message, and is a step that can be taken back. The render of the
        picked thing and the checks are started at the first look, and again after each change. A file that
        cannot be read is the message through the model, and is counted as said when it begins to be so.
        """
        if first and self.player is None:
            self.say(NO_SOUND)
        before, problem = self.model.picked, self.model.problem
        self.model.poll()
        if self.model.problem and self.model.problem != problem:
            self._messages += 1
        if self._outside(self.model.changes, now) or first:
            self._renew(before)

    def _outside(self, changes: Sequence[Change], now: float) -> bool:
        """Takes up what the model found that another program changed, at a look or as a write here read the file.

        Each thing glows from `now`, they are named in a notice, and they are a step that can be taken back,
        called what the notice says. False where there is nothing to take up.
        """
        if not changes:
            return False
        changed = things(changes)
        self._lit.update((thing, now) for thing in changed)
        self._history.add(changes, _told(changed))
        self._tell(changed)
        return True

    def _tell(self, notice: str | set[tuple[str, str]]) -> None:
        """Gives notice of what another program did: words, or the things that changed outside, to be named.

        It is said as `say` says anything, and it has the status line as well: at once where no notice has
        it, and else when its turn comes. Things that changed outside are added to those that wait already,
        where they are the last to wait.
        """
        words = notice if isinstance(notice, str) else _told(notice)
        self.say(words)
        self._unshown = False
        now = self._clock()
        self._notify(now)
        if not self._notice:
            self._notice, self._until = words, now + NOTICE_TIME
        elif self._notices and isinstance(notice, set) and isinstance(self._notices[-1], set):
            self._notices[-1] |= notice
        else:
            self._notices.append(notice)

    def _notify(self, now: float) -> None:
        """Gives the status line to the next notice once the one that has it has had its time, at the moment `now`.

        With no notice left to wait, the line is the message again. A notice whose turn comes is one more of
        `messages`, and so is the message where it was said while the notices had the line.
        """
        if not self._notice or now < self._until:
            return
        if self._notices:
            notice = self._notices.popleft()
            self._notice, self._until = notice if isinstance(notice, str) else _told(notice), now + NOTICE_TIME
            self._messages += 1
        else:
            self._notice = ""
            if self._unshown:
                self._messages += 1
            self._unshown = False

    def _renew(self, before: tuple[str, str] | None) -> None:
        """Starts the render of the picked thing and the checks, at the first look and after a change to the project.

        `before` is what was picked before the change. The model keeps a render and the findings for as long as
        nothing in the project differs, so with nothing let go of nothing is made again. The picked song is
        heard by its mutes and solos as before, less those of channels it no longer has.
        """
        # What the model has let go of, with the project as it was, is to be taken up when it is there again.
        self._taken = self._taken and self.model.render_ready()
        self._checked = self._checked and self.model.findings_ready()
        if before is not None and self.model.picked is None:
            # The picked thing was removed: the first thing of the listing takes its place, with nothing muted.
            self._drop()
            self._muted = self._soloed = frozenset()
            for section, things in self.model.listing():
                if things:
                    self.model.pick(KINDS[section], things[0][0])
                    break
        # A song whose channels are not the ones it had is heard whole by the model, until it is told again.
        if self._hear():
            self._taken = False
        if self.model.picked is not None:
            self.model.start_render()
        self.model.start_findings()

    def _held(self, where: str) -> object:
        """What the project holds at a place, as it is known here. NOTHING for a place that holds nothing."""
        try:
            return get_in(self.model.project.to_dict(), where)
        except RowSynthError:
            return NOTHING

    def _made(self, told: Sequence[dict], held: Sequence[object], label: str, key: object,
              before: tuple[str, str] | None) -> None:
        """Takes up changes the model has made as one: `told` is what it answered for each, in their order.

        They are kept as one step, unless every one of them left its place as it was. `held` is what the
        project here had at each place, and where the file had something else, a notice says so. `before`
        is what was picked before the changes. What the write found changed outside in other places is
        taken up first, so that its step comes before this one.
        """
        self.say("")
        if all(one.get("same") for one in told):
            return
        self._outside(self.model.changes, self._clock())
        # The step has values of its own, whatever becomes of those that were handed in.
        step = [(one["where"], one.get("was", NOTHING), copy.deepcopy(one["now"]) if "now" in one else NOTHING)
                for one in told]
        self._history.add(step, label, key)
        if not all(same(had, was) for had, (_, was, _) in zip(held, step)):
            self._tell(f"{label} {BOTH}")
        self._renew(before)

    def _replay(self, changes: Iterable[Change] | None, back: Callable[[], object]) -> bool:
        """Makes the changes of a step the history has just moved, all as one write. None is for no step, and false.

        `back` moves the step to where it was. Where the changes are refused none of them is made, the step is
        moved back, the message is the refusal and the answer is false.

        What the write found changed outside in other places is taken up as after a look: it is a step on
        top of the one that was moved, and with it there is no step left to make again.
        """
        if changes is None:
            return False
        before = self.model.picked
        try:
            self.model.change_many([(where, now) for where, _, now in changes])
        except RowSynthError as error:
            back()
            self.say(str(error))
            return False
        self.say("")
        self._outside(self.model.changes_after, self._clock())
        self._renew(before)
        return True

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
                self.player.play(*self._heard(made))
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
        self.say(_words(error))

    def _check(self) -> None:
        """Takes up the findings the model has ready: where the checks refused the project or broke, that is said."""
        self._checked = True
        try:
            self.model.findings()
        except Exception as error:
            self.say(_words(error))
