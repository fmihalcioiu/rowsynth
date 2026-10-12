"""The settings bar of a song: the picked channel's instrument, volume, pan and echo, and the toggles of the view."""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from ..edit import NOTHING, get_in
from ..errors import RowSynthError
from . import theme
from .dials import Knob, Stepper, plain
from .font import ADVANCE, Font
from .kit import Bar, Slot, told
from .layout import Rect
from .model import TEST_NOTE
from .panels import within
from .theme import Colour
from .ui import Label, Toggle, Widget

if TYPE_CHECKING:
    from ..project import Project
    from .panels import Panel
    from .session import Session

#: How high the row of the bar's controls is: a knob, with the line of its name above it and the line of its value
#: below. How high a small button is, and from one control of the bar to the next.
HEIGHT = theme.BAR_ROW_H
BUTTON_H = theme.BUTTON_H
SPACE = theme.BAR_GAP

#: What the two toggles of the view are called, and the word after the name of a drum channel.
FOLLOW = "FOLLOW"
GHOSTS = "GHOSTS"
DRUMS = "DRUMS"
#: What the settings are called. The last two are the echo's: how many rows after a note it is played a second
#: time, and how loud that second time is. The switch of the echo reads one of the two words after them.
CHANNEL = "CHANNEL"
INSTRUMENT = "INSTRUMENT"
VOLUME = "VOLUME"
PAN = "PAN"
ECHO = "ECHO"
TIME = "TIME"
AMOUNT = "AMOUNT"
ON, OFF = "ON", "OFF"

#: The travel of each knob: its lowest value, its highest, and how many steps there are between them. A project
#: may hold any number for each of the four, so the travels are chosen here. A volume goes in hundredths up to
#: twice the 1 a channel has when it says none, a pan in hundredths from all left to all right, the time of an
#: echo in whole rows up to a bar of sixteen, and its amount, a share of the note's volume, in hundredths.
LEVELS = (0.0, 2.0, 200)
PANS = (-1.0, 1.0, 200)
TIMES = (1.0, 16.0, 15)
AMOUNTS = (0.0, 1.0, 100)
#: What the engine takes for a setting that a channel does not have. A setting put back to it is taken out of the
#: file. The two settings of an echo are not here: an echo must have both.
ABSENT = {"gain": 1, "pan": 0}
#: The echo a channel gets when its echo is switched on and was not switched off here before: the one the
#: examples use.
USUAL_ECHO = {"rows": 3, "gain": 0.3}


def pan_text(pan: float) -> str:
    """Where a pan puts a channel, in a few letters: "C" in the middle, and else the side and how far, as "L 30".

    How far is in hundredths of the way to that side, so "R 100" is all right.
    """
    far = math.floor(abs(pan) * 100 + 0.5)
    return "C" if far == 0 else f"{'L' if pan < 0 else 'R'} {far}"


def rows_text(rows: float) -> str:
    """A number of rows in words, as "1 row" and "3 rows"."""
    return f"{plain(rows)} {'row' if rows == 1 else 'rows'}"


#: How wide the room for the channel's name is, the field of the instrument, and the least a knob with its two
#: lines. A knob whose name or whose longest value is wider has the room of that, and a pixel free at each end.
NAME_W = Font.width("m" * 15)
INST_W = 78
KNOB_W = 33
VOLUME_W = max(KNOB_W, Font.width(VOLUME) + 2)
TIME_W = max(KNOB_W, Font.width(rows_text(TIMES[1])) + 2)
AMOUNT_W = max(KNOB_W, Font.width(AMOUNT) + 2)
#: How wide the switch of the echo is, and each of the two toggles of the view, which stand one over the other.
ECHO_W = Font.width(ECHO) + 2 * theme.PAD
TOGGLE_W = max(Font.width(text) for text in (FOLLOW, GHOSTS)) + 2 * theme.PAD


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


def toggle_parts(room: Rect) -> tuple[Rect, Rect]:
    """Where the two toggles of the view go in the room they share at the right end of the bar: one over the other.

    Each is as wide as the room and as high as a small button, with `theme.GAP` between the two, and the pair
    stands in the middle of the height. FOLLOW is the upper one.
    """
    x, y, w, h = room
    top = y + max(0, (h - 2 * BUTTON_H - theme.GAP) // 2)
    return within((x, top, w, BUTTON_H), room), within((x, top + BUTTON_H + theme.GAP, w, BUTTON_H), room)


class ChannelBar:
    """The settings bar of a song panel: the settings of the picked channel, and the two toggles of the view.

    It is no panel of the frame. A song panel owns it, adds its widgets through itself, gives it a rectangle
    with `place`, a channel with `show` and a frame with `update`, and says with `seen` whether it is on the
    window. `frame` is the bar every editor has, as `kit.Bar` draws it: it lays the controls out, and has the
    name of each over it. From the left stand CHANNEL, which is the channel's name in its colour, with the word
    DRUMS after that of a drum channel, then `inst`, `volume`, `pan` and `echo`, and while the echo is on
    `time` and `amount`: the rows after a note at which it is played a second time, and how loud that second
    time is as a share of the note. At the right end stand the settings its owner gives it with `place`, which
    are the song's own, and after them `follow` over `ghosts`, which are the panel's to read. In a bar too
    narrow for all of it the controls at the left give way, the last of them first.

    A knob shows its value as it is turned, and is saved when the hand has let go of it: one change to the
    file at the channel's place, and one step that can be taken back. A volume or a pan put back to what the
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

        def knob(travel: tuple[float, float, int], usual: float, key: str, words: str, what: str, back: str,
                 show: Callable[[float], str] = plain) -> Knob:
            """A knob of the bar: a right press puts it back to `usual`, and let go of, it is saved at `key`."""
            low, high, steps = travel
            help = told(f"drag up or down {what}, slower with shift", f"back to {back}", "steps it")
            return panel.add(Knob(nowhere, usual, low, high, default=usual, steps=steps, show=show, help=help,
                                  clock=clock, settled=lambda value: self._put(key, value, words, show)))

        # The bar itself is added first, so that it is drawn under its controls.
        self.frame = panel.add(Bar(nowhere))
        self.name = panel.add(Label(nowhere, ""))
        self.word = panel.add(Label(nowhere, DRUMS))
        self.inst = panel.add(Stepper(nowhere, ("",), changed=self._step, help=told(
            "an arrow steps to the next instrument, the song's own come first", wheel="steps through them")))
        self.volume = knob(LEVELS, ABSENT["gain"], "gain", "volume", "for how loud the channel is", "1")
        self.pan = knob(PANS, ABSENT["pan"], "pan", "pan", "to place the channel from left to right",
                        "the middle", pan_text)
        self.echo = panel.add(Toggle(nowhere, OFF, changed=self._switch, help=told(
            "switch the echo on or off: with it every note is played a second time, later and quieter")))
        self.time = knob(TIMES, USUAL_ECHO["rows"], "echo.rows", "echo time",
                         "for how many rows after a note its echo comes", rows_text(USUAL_ECHO["rows"]), rows_text)
        self.amount = knob(AMOUNTS, USUAL_ECHO["gain"], "echo.gain", "echo amount",
                           "for how loud the echo is, as a share of the note", plain(USUAL_ECHO["gain"]))
        self.follow = panel.add(Toggle(nowhere, FOLLOW, on=True, help=told(
            "have the pattern that is shown go along with the playing, or stay")))
        self.ghosts = panel.add(Toggle(nowhere, GHOSTS, on=True, help=told(
            "show or hide the notes of the other channels, drawn dim behind")))
        #: The names of the song and of the channel that are shown, and the colour of the channel.
        self.song: str | None = None
        self.channel: str | None = None
        self._colour: Colour = theme.TEXT
        #: The project the widgets were set from last, and what it holds for the channel: None for no channel.
        self._project: Project | None = None
        self._held: dict | None = None
        #: The settings of each echo that was switched off here, by its song and its channel.
        self._echoes: dict[tuple[str, str], dict] = {}
        #: The controls of the channel that are seen with any channel, and its knobs.
        self._settings: tuple[Widget, ...] = (self.inst, self.volume, self.pan, self.echo, self.time, self.amount)
        self._knobs = (self.volume, self.pan, self.time, self.amount)
        #: The settings of the owner that stand in the right group, and the room the name of the channel has.
        self._own: tuple[Slot, ...] = ()
        self._room: Rect = nowhere
        self._on = False
        self._see()

    @property
    def busy(self) -> bool:
        """Whether a knob is under the hand: dragged, or in a run of the wheel that has not ended."""
        return any(knob.busy for knob in self._knobs)

    def place(self, rect: Rect, own: Sequence[Slot] = ()) -> list[Rect]:
        """Take the bar's rectangle, and lay the controls out in it: the answer is the rectangles of `own`.

        `own` is the settings of the owner, as slots of a bar: they stand in the right group, in the order
        given, before the two toggles of the view. One the bar has no room for has no width.
        """
        self.frame.rect, self._own = rect, tuple(own)
        rects = self._lay()
        self._see()
        return rects

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
            self.volume.value = held.get("gain", ABSENT["gain"])
            self.pan.value = held.get("pan", ABSENT["pan"])
            echo = held.get("echo")
            self.echo.is_on, self.echo.text = echo is not None, OFF if echo is None else ON
            if echo is not None:
                self.time.value, self.amount.value = echo["rows"], echo["gain"]
            self.name.text, self.name.colour = self.channel, self._colour
            for knob in self._knobs:
                knob.colour = self._colour
        self._lay()
        self._see()

    def _lay(self) -> list[Rect]:
        """Give each widget its rectangle in the bar, and give back those of the owner's settings.

        The name of the channel and its word share the room under CHANNEL, and the two toggles of the view the
        room at the right end.
        """
        left = [Slot(CHANNEL, self.name, NAME_W, "left"), Slot(INSTRUMENT, self.inst, INST_W),
                Slot(VOLUME, self.volume, VOLUME_W), Slot(PAN, self.pan, KNOB_W), Slot(ECHO, self.echo, ECHO_W),
                Slot(TIME, self.time, TIME_W), Slot(AMOUNT, self.amount, AMOUNT_W)]
        rects = self.frame.lay(left, [*self._own, Slot("", self.follow, TOGGLE_W, tall=True)])
        self._room = rects[0]
        drums = bool(self._held and self._held.get("drums"))
        self.name.rect, self.word.rect = name_parts(self._room, self.name.text, drums)
        self.follow.rect, self.ghosts.rect = toggle_parts(rects[-1])
        return rects[len(left):-1]

    def _see(self) -> None:
        """Have each widget seen or not: by whether the bar is on the window, what there is to show, and what fits.

        The bar and FOLLOW are seen with any song. The settings are seen with a channel, the knobs of the echo
        while it is on, the word DRUMS for a drum channel and GHOSTS for a tuned one: it is the roll that has
        ghosts. What the bar has no room for is not seen.
        """
        held = self._held if self._on else None
        drums = bool(held and held.get("drums"))
        for widget in self._settings:
            widget.visible = held is not None and widget.rect[2] > 0
        for knob in (self.time, self.amount):
            knob.visible = knob.visible and self.echo.is_on
        self.name.visible = held is not None and self._room[2] > 0
        self.word.visible = self.name.visible and drums
        self.frame.visible = self._on
        self.follow.visible = self._on and self.follow.rect[2] > 0
        self.ghosts.visible = held is not None and not drums and self.ghosts.rect[2] > 0

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
