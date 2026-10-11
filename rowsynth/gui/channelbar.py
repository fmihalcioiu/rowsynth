"""The settings of the picked channel of a song in one bar: its instrument, its level, its pan and its echo."""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..edit import NOTHING, get_in
from ..errors import RowSynthError
from . import theme
from .dials import Knob, Stepper, knob_parts, plain
from .font import ADVANCE, Font
from .layout import Rect
from .model import TEST_NOTE
from .panels import within
from .theme import Colour
from .ui import Label, Toggle

if TYPE_CHECKING:
    from ..project import Project
    from .panels import Panel
    from .session import Session

#: How high the bar is: a knob, with the line of its label above it and the line of its value below.
HEIGHT = 37
#: How high a small button is: the toggles of the bar are such buttons, and so is the field of the instrument.
BUTTON_H = 13
#: From one part of the bar to the next.
SPACE = 6
#: How wide the room for the channel's name is, the field of the instrument, and a knob with its two lines.
NAME_W = Font.width("m" * 15)
INST_W = 78
KNOB_W = 33

#: What the two toggles of the view are called, and the word after the name of a drum channel.
FOLLOW = "FOLLOW"
GHOSTS = "GHOSTS"
DRUMS = "DRUMS"
#: What the settings are called. The last two are the echo's: how many rows after a note it is played a second
#: time, and how loud that second time is.
INSTRUMENT = "INSTRUMENT"
LEVEL = "LEVEL"
PAN = "PAN"
ECHO = "ECHO"
AFTER = "AFTER"
LOUD = "LOUD"

#: The travel of each knob: its lowest value, its highest, and how many steps there are between them. A project
#: may hold any number for each of the four, so the travels are chosen here. A level goes in hundredths up to
#: twice the 1 a channel has when it says none, a pan in hundredths from all left to all right, the rows of an
#: echo in whole rows up to a bar of sixteen, and its share of the note's level in hundredths.
LEVELS = (0.0, 2.0, 200)
PANS = (-1.0, 1.0, 200)
AFTERS = (1.0, 16.0, 15)
LOUDS = (0.0, 1.0, 100)
#: What the engine takes for a setting that a channel does not have. A setting put back to it is taken out of the
#: file. The two settings of an echo are not here: an echo must have both.
ABSENT = {"gain": 1, "pan": 0}
#: The echo a channel gets when its echo is switched on and was not switched off here before: the one the
#: examples use.
USUAL_ECHO = {"rows": 3, "gain": 0.3}

#: The parts that give way in a bar too narrow for all of them, the first first. The two toggles of the view stay.
DROPPED = (("after", "loud"), ("pan",), ("level",), ("echo",), ("inst",), ("name",))
#: The parts that are knobs: they are as high as the bar, and the others as high as a button.
KNOBS = ("level", "pan", "after", "loud")


def pan_text(pan: float) -> str:
    """Where a pan puts a channel, in a few letters: "C" in the middle, and else the side and how far, as "L 30".

    How far is in hundredths of the way to that side, so "R 100" is all right.
    """
    far = math.floor(abs(pan) * 100 + 0.5)
    return "C" if far == 0 else f"{'L' if pan < 0 else 'R'} {far}"


def tidy(value: float) -> int | float:
    """The number a knob was left at as it is written in the file: to two places, and whole without its point."""
    value = round(value, 2)
    return int(value) if value == int(value) else value


def save(session: Session, where: str, value: object, label: str, absent: object = NOTHING) -> bool:
    """Put a value at a place of the project, as one step that is called `label`, and say whether the file is as asked.

    With NOTHING for the value what is there is taken out of the file. So it is with a value that is `absent`,
    which is what the engine takes for a setting that is left out: that is not written. What the file does not
    have is not there to take out, and is as was asked already. A change that is refused is not made, and the
    session's message is the refusal.
    """
    if value is not NOTHING and value != absent:
        return session.change(where, value, label)
    try:
        get_in(session.model.project.to_dict(), where)
    except RowSynthError:
        return True
    return session.remove(where, label)


def instruments(project: Project, song: str | None) -> list[str]:
    """The names of the instruments a song of a project can use: its own first, then the project's.

    A name that both have is there once, among the song's own: in the song it is the song's instrument.
    """
    own = project.songs.get(song, {}).get("instruments", {})
    return [*own, *(name for name in project.instruments if name not in own)]


def arrange(rect: Rect) -> dict[str, Rect]:
    """Where the parts of the bar go in its rectangle, each by what it is called, and each cut to the bar.

    From the left stand "name", "inst" with "heading" over it, "level", "pan", "echo", "after" and "loud", and
    at the right end "follow" and "ghosts". A bar too narrow for all of them drops parts in the order of
    `DROPPED` until the rest fit, so no two ever overlap. A part that was dropped has no width.
    """
    x, y, w, h = rect
    follow, ghosts, echo = (Font.width(text) + 2 * theme.PAD for text in (FOLLOW, GHOSTS, ECHO))
    widths = {"name": NAME_W, "inst": INST_W, "level": KNOB_W, "pan": KNOB_W, "echo": echo, "after": KNOB_W,
              "loud": KNOB_W}
    right = follow + theme.BUTTON_GAP + ghosts
    kept = list(widths)
    for group in DROPPED:
        if sum(widths[part] + SPACE for part in kept) + right <= w:
            break
        kept = [part for part in kept if part not in group]
    top = y + (h - BUTTON_H) // 2
    placed = {part: (x, y, 0, 0) for part in (*widths, "heading")}
    at = x
    for part in kept:
        placed[part] = (at, y, widths[part], h) if part in KNOBS else (at, top, widths[part], BUTTON_H)
        if part == "inst":
            # The heading is on the line the knobs have their labels on.
            placed["heading"] = knob_parts((at, y, widths[part], h))[0]
        at += widths[part] + SPACE
    placed["follow"] = (x + w - right, top, follow, BUTTON_H)
    placed["ghosts"] = (x + w - ghosts, top, ghosts, BUTTON_H)
    return {part: within(place, rect) for part, place in placed.items()}


def name_parts(room: Rect, name: str, drums: bool) -> tuple[Rect, Rect]:
    """Where a channel's name goes in the room for it, and the word that comes after the name of a drum channel.

    The name is at the left, no wider than it is written. The word is one glyph after it, and a name too long
    for the room is cut short to leave the word whole. A tuned channel has no word: its name has all the room,
    and the word no width.
    """
    x, y, w, h = room
    word, apart = (Font.width(DRUMS), ADVANCE) if drums else (0, 0)
    wide = max(0, min(Font.width(name), w - word - apart))
    return (x, y, wide, h), within((x + wide + apart, y, word, h), room)


class ChannelBar:
    """The bar over the view of a song panel: the settings of the picked channel, and the two toggles of the view.

    It is no panel of the frame. A song panel owns it, adds its widgets through itself, gives it a rectangle
    with `place`, a channel with `show` and a frame with `update`, and says with `seen` whether it is on the
    window. From the left it has the channel's name in its colour, with the word DRUMS after that of a drum
    channel, then `inst`, `level`, `pan` and `echo`, and while the echo is on `after` and `loud`: the rows
    after a note at which it is played a second time, and how loud that second time is as a share of the note.
    `follow` and `ghosts` are at the right end, and are the panel's to read.

    A knob shows its value as it is turned, and is saved when the hand has let go of it: one change to the
    file at the channel's place, and one step that can be taken back. A level or a pan put back to what the
    engine takes when a channel has none is taken out of the file. The instrument and the echo are saved at
    once. A new instrument is heard, one note of it. An echo that is switched off is taken out of the file,
    and its settings are kept here, so that switched on again it is the echo it was. Switched on for the
    first time it is `USUAL_ECHO`.

    After each change, and whenever the model has loaded the file again, the widgets show what the file
    holds. A change that was refused is not in it, so the widget is as it was, and the session says why.
    While a knob is under the hand nothing is read, and what changed outside meanwhile shows when the hand
    has let go.
    """

    def __init__(self, panel: Panel, clock: Callable[[], float] = time.perf_counter) -> None:
        """`panel` is the panel the widgets are added through. `clock` times a run of the wheel on a knob."""
        self.session = panel.session
        nowhere = (0, 0, 0, 0)

        def knob(text: str, travel: tuple[float, float, int], usual: float, key: str, words: str, help: str,
                 show: Callable[[float], str] = plain) -> Knob:
            """A knob of the bar: a right click puts it back to `usual`, and let go of, it is saved at `key`."""
            low, high, steps = travel
            return panel.add(Knob(nowhere, usual, low, high, default=usual, steps=steps, text=text, show=show,
                                  help=help, clock=clock,
                                  settled=lambda value: self._put(key, value, words, show)))

        self.name = panel.add(Label(nowhere, ""))
        self.word = panel.add(Label(nowhere, DRUMS))
        self.heading = panel.add(Label(nowhere, INSTRUMENT, align="centre"))
        self.inst = panel.add(Stepper(nowhere, ("",), changed=self._step,
                                      help="the instrument the channel plays: the song's own first, then the others"))
        self.level = knob(LEVEL, LEVELS, ABSENT["gain"], "gain", "level",
                          "how loud the channel is in the mix: a right click is 1 again")
        self.pan = knob(PAN, PANS, ABSENT["pan"], "pan", "pan",
                        "where the channel is from left to right: a right click is the middle again", pan_text)
        self.echo = panel.add(Toggle(nowhere, ECHO, changed=self._switch,
                                     help="every note of the channel is played a second time, later and quieter"))
        self.after = knob(AFTER, AFTERS, USUAL_ECHO["rows"], "echo.rows", "echo after",
                          "how many rows after a note it is played the second time")
        self.loud = knob(LOUD, LOUDS, USUAL_ECHO["gain"], "echo.gain", "echo loud",
                         "how loud the second time is, as a share of the note")
        self.follow = panel.add(Toggle(nowhere, FOLLOW, on=True,
                                       help="the pattern that is shown goes along with the playing"))
        self.ghosts = panel.add(Toggle(nowhere, GHOSTS, on=True,
                                       help="the notes of the other channels are drawn dim behind"))
        #: The names of the song and of the channel that are shown, and the colour of the channel.
        self.song: str | None = None
        self.channel: str | None = None
        self._colour: Colour = theme.TEXT
        #: The project the widgets were set from last, and what it holds for the channel: None for no channel.
        self._project: Project | None = None
        self._held: dict | None = None
        #: The settings of each echo that was switched off here, by its song and its channel.
        self._echoes: dict[tuple[str, str], dict] = {}
        #: The settings by the part of the bar each stands in, and those of them that are knobs.
        self._settings = {"heading": self.heading, "inst": self.inst, "level": self.level, "pan": self.pan,
                          "echo": self.echo, "after": self.after, "loud": self.loud}
        self._knobs = tuple(self._settings[part] for part in KNOBS)
        self._parts = arrange(nowhere)
        self._on = False
        self._see()

    @property
    def busy(self) -> bool:
        """Whether a knob is under the hand: dragged, or in a run of the wheel that has not ended."""
        return any(knob.busy for knob in self._knobs)

    def place(self, rect: Rect) -> None:
        """Take the bar's rectangle, and lay the widgets out in it as `arrange` has them."""
        self._parts = arrange(rect)
        self._lay()
        self._see()

    def seen(self, on: bool) -> None:
        """Bring the bar onto the window, or take it off. On it, a widget is seen by what there is to show."""
        self._on = on
        self._see()

    def show(self, song: str | None, channel: str | None, colour: Colour = theme.TEXT) -> None:
        """Show a channel of a song of the project, both by name, in the colour the channel has in its song.

        With None for the channel, or a channel the project does not have, no setting is shown. The channel
        that is shown already is left as it is while a knob is under the hand. Another channel takes the bar
        at once, and what the hand was doing to a knob is dropped, with nothing saved.
        """
        again = (song, channel) == (self.song, self.channel)
        self.song, self.channel, self._colour = song, channel, colour
        if again and self.busy:
            return
        for knob in self._knobs:
            knob.drop()
        self._read()

    def update(self) -> None:
        """Give the knobs their frame, and read the channel again where the model has loaded its file again.

        The frame of a knob ends a run of the wheel whose time is up, which saves it. While a knob is under
        the hand nothing is read.
        """
        for knob in self._knobs:
            knob.update()
        if not self.busy and self.session.model.project is not self._project:
            self._read()

    def _channel(self) -> dict | None:
        """What the project holds now for the channel that is shown, or None where it has no such channel."""
        return self.session.model.project.songs.get(self.song, {}).get("channels", {}).get(self.channel)

    def _read(self) -> None:
        """Set every widget from what the project holds for the channel now."""
        project = self._project = self.session.model.project
        held = self._held = self._channel()
        if held is not None:
            names = instruments(project, self.song)
            self.inst.options, self.inst.picked = tuple(names), names.index(held["inst"])
            self.level.value = held.get("gain", ABSENT["gain"])
            self.pan.value = held.get("pan", ABSENT["pan"])
            echo = held.get("echo")
            self.echo.is_on = echo is not None
            if echo is not None:
                self.after.value, self.loud.value = echo["rows"], echo["gain"]
            self.name.text, self.name.colour = self.channel, self._colour
            for knob in self._knobs:
                knob.colour = self._colour
        self._lay()
        self._see()

    def _lay(self) -> None:
        """Give each widget its rectangle: a part of the bar, and for the name and its word their share of one."""
        parts = self._parts
        for part, widget in self._settings.items():
            widget.rect = parts[part]
        self.follow.rect, self.ghosts.rect = parts["follow"], parts["ghosts"]
        drums = bool(self._held and self._held.get("drums"))
        self.name.rect, self.word.rect = name_parts(parts["name"], self.name.text, drums)

    def _see(self) -> None:
        """Have each widget seen or not: by whether the bar is on the window, what there is to show, and what fits.

        FOLLOW is seen with any song. The settings are seen with a channel, the knobs of the echo while it is
        on, the word DRUMS for a drum channel and GHOSTS for a tuned one: it is the roll that has ghosts.
        """
        held = self._held if self._on else None
        drums = bool(held and held.get("drums"))
        for part, widget in self._settings.items():
            widget.visible = held is not None and self._parts[part][2] > 0
        for knob in (self.after, self.loud):
            knob.visible = knob.visible and self.echo.is_on
        self.name.visible = held is not None and self._parts["name"][2] > 0
        self.word.visible = self.name.visible and drums
        self.follow.visible = self._on
        self.ghosts.visible = held is not None and not drums

    def _change(self, key: str, value: object, words: str) -> bool:
        """Put a value at a place of the channel that is shown, as one step, and show what the file then holds.

        `key` is the place in the channel, and `words` say what was done there. It is saved as `save` saves:
        with NOTHING for the value what is there is taken out of the file, and so it is with a value that is
        what the engine takes for a setting left out. The answer is whether the file is as was asked. A change
        that is refused is not made, the session's message is the refusal, and the widgets show what the file
        holds as before.
        """
        saved = save(self.session, f"songs.{self.song}.channels.{self.channel}.{key}", value,
                     f"{self.channel}: {words}", ABSENT.get(key, NOTHING))
        self._read()
        return saved

    def _put(self, key: str, value: float, words: str, text: Callable[[float], str]) -> None:
        """Save the value a knob was left at, written as `tidy` has it and called by how the knob shows it."""
        value = tidy(value)
        self._change(key, value, f"{words} {text(value)}")

    def _step(self, index: int) -> None:
        """Save the instrument the stepper was put on, and let one note of it be heard if it was saved."""
        name = self.inst.options[index]
        if self._change("inst", name, f"instrument {name}"):
            self.session.sound(name, TEST_NOTE, self.song)

    def _switch(self, on: bool) -> None:
        """Save the echo as it was switched: on, it has the settings it was switched off with here, or the usual."""
        kept = (self.song, self.channel)
        if on:
            self._change("echo", dict(self._echoes.get(kept, USUAL_ECHO)), "echo on")
            return
        echo = (self._channel() or {}).get("echo")
        if self._change("echo", NOTHING, "echo off") and echo is not None:
            self._echoes[kept] = echo
