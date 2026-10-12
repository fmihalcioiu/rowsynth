"""The mixer of a song: a strip for each channel with its fader, meter, mute and solo, and one for the whole song."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from . import theme
from .channelbar import ABSENT, BUTTON_H, LEVELS, save, tidy
from .dials import Meter, Slider, plain
from .font import Font
from .kit import ZOOM, ScrollBar, told, wheel
from .layout import Rect
from .lists import DIMMED, mix
from .theme import Colour
from .ui import NOWHERE, Button, Event, Label, Pos, Toggle, Ui, Widget, inside

if TYPE_CHECKING:
    from ..project import Project
    from .canvas import Canvas
    from .panels import Panel

#: The least width of a channel's strip: its two toggles side by side, with air round them. Strips that would be
#: narrower keep it, and the mixer scrolls. With few channels a strip is no wider than the most.
STRIP_LEAST = 32
STRIP_MOST = 44
#: What is left free before the first drum channel, after the tuned ones.
GROUP_GAP = 8
#: How wide the strip of the whole song is, and what is left free before it: a line runs down the middle of that.
SONG_W = 36
APART = 9
#: How wide a fader is, a meter of one bar and a meter of two, and what is left free between a meter and its fader.
FADER_W = 12
METER_W = 6
SONG_METER_W = 11
PAIR_GAP = 3
#: How wide the toggle of a mute or of a solo is. It is as high as any small button.
TOGGLE_W = 13
#: From one part of a strip to the part under it.
SPACE = 2
#: What the scroll bar under the strips takes of the height of a mixer that scrolls: the bar, and what is left
#: free over it.
BAR_ROOM = theme.SCROLL_GAP + theme.SCROLL_W
#: The least height of a fader, and with it the least height of a mixer: the parts of a strip one under the other,
#: and the bar under them. A mixer lower than that, or too narrow for one channel beside the song, shows nothing.
FADER_LEAST = 24
HIGH_LEAST = 2 * Font.LINE + BUTTON_H + FADER_LEAST + 3 * SPACE + BAR_ROOM
WIDE_LEAST = STRIP_LEAST + APART + SONG_W
#: How many strips one step of the wheel moves a mixer that scrolls.
WHEEL_STRIPS = 2

#: The volume the engine takes for a song that says none, which is not written, and the travel of the song's
#: fader: its lowest value, its highest, and the steps between them. A project may hold any number, so the
#: travel is chosen here: in hundredths up to 1.
SONG_ABSENT = 0.35
SONG_LEVELS = (0.0, 1.0, 100)

#: What the two toggles of a strip read, and what the strip of the whole song is called.
MUTE = "M"
SOLO = "S"
ALL = "ALL"
#: What the wheel does over the mixer, as the help line says it. What the mixer says of itself where the mouse
#: is on none of its controls, and what the bar under its strips says.
WHEEL = "along the strips"
DESK_HELP = f"a strip for each channel, and at the right the whole song   {told(wheel=WHEEL)}"
BAR_HELP = told("drag the bar to go along the strips, or press beside it to jump", wheel=WHEEL)
#: The parts of a strip, as `strip_parts` calls them.
PARTS = ("number", "meter", "fader", "mute", "solo", "both", "name")


def ordered(channels: dict) -> list[tuple[int, str]]:
    """The channels of a song as a mixer has them: the tuned ones and then the drum channels, as the song lists them.

    `channels` is what a song holds for each channel by its name. Each is given as its place in the song, which
    its colour goes by, and its name.
    """
    return sorted(enumerate(channels), key=lambda found: bool(channels[found[1]].get("drums")))


def span(first: int, last: int, tuned: int, wide: int) -> int:
    """How wide the strips from `first` to `last` are together, each `wide`, of which the first `tuned` are tuned.

    The gap before the drums is counted where a tuned strip and a drum strip are both among them.
    """
    return (last - first + 1) * wide + (GROUP_GAP if first < tuned <= last else 0)


class Laid(NamedTuple):
    """Where the strips of a mixer stand, as `lay` has them.

    `wide` is the width of a channel's strip. `first` is the first channel in view, and `most` the furthest that
    can be: 0 while every strip is in view. `strips` has the rectangle of each channel in view, by its place
    among the channels. `song` is the strip of the whole song. `bar` is the room of the scroll bar under the
    strips, or None while every strip is in view.
    """

    wide: int
    first: int
    most: int
    strips: dict[int, Rect]
    song: Rect
    bar: Rect | None


def lay(rect: Rect, tuned: int, drums: int, first: int = 0) -> Laid:
    """Where the strips of so many tuned channels and so many drum channels stand in the rectangle of a mixer.

    The strip of the whole song is at the right end, `SONG_W` wide, with `APART` free before it. The channels
    share the room before that from the left, the tuned ones and then, after `GROUP_GAP`, the drums: each as
    wide as fits, from `STRIP_MOST` down to `STRIP_LEAST`. Where they do not all fit at the least width they
    keep it, and the mixer scrolls by whole strips: `first` is the first in view, kept between 0 and the least
    at which the last is in view, the strips after it are in view as far as they fit whole, and all are less
    high by the bar that then runs under them. The gap before the drums is there only between two strips in
    view. A rectangle under `WIDE_LEAST` by `HIGH_LEAST` has no strip in it.
    """
    x, y, w, h = rect
    count = tuned + drums
    if w < WIDE_LEAST or h < HIGH_LEAST:
        return Laid(0, 0, 0, {}, (x, y, 0, 0), None)
    room = w - APART - SONG_W
    wide = min(STRIP_MOST, (room - (GROUP_GAP if tuned and drums else 0)) // max(1, count))
    most, high = 0, h
    if wide < STRIP_LEAST:
        wide, high = STRIP_LEAST, h - BAR_ROOM
        most = next(start for start in range(count) if span(start, count - 1, tuned, wide) <= room)
    first = min(max(0, first), most)
    strips: dict[int, Rect] = {}
    at = x
    for index in range(first, count):
        at += GROUP_GAP if first < index == tuned else 0
        if at + wide > x + room:
            break
        strips[index] = (at, y, wide, high)
        at += wide
    return Laid(wide, first, most, strips, (x + w - SONG_W, y, SONG_W, high),
                (x, y + h - theme.SCROLL_W, room, theme.SCROLL_W) if most else None)


def along(laid: Laid, count: int) -> tuple[int, int, int]:
    """What the scroll bar under the strips of `count` channels is shown, in pixels, a strip counted as its width.

    They are how long all the strips are, how much of that is in view, and how much lies before the view:
    the bar is to its thumb what the strips are to those in view.
    """
    return count * laid.wide, (count - laid.most) * laid.wide, laid.first * laid.wide


def strip_parts(rect: Rect, meter: int = METER_W) -> dict[str, Rect]:
    """Where the parts of a strip go in its rectangle, each by what it is called.

    "number" runs along the top and "name" along the foot, each a line of text high. Over the name stand "mute"
    and "solo" side by side in the middle: "both" is the room of the two together. Between that and the number
    stand "meter", `meter` wide, and "fader" beside it, the pair in the middle. A strip with no room has parts
    with none.
    """
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return dict.fromkeys(PARTS, (x, y, 0, 0))
    foot = y + h - Font.LINE
    top = foot - SPACE - BUTTON_H
    left = x + (w - 2 * TOGGLE_W - theme.BUTTON_GAP) // 2
    pair = x + (w - meter - PAIR_GAP - FADER_W) // 2
    stand = y + Font.LINE + SPACE
    return {
        "number": (x, y, w, Font.LINE),
        "meter": (pair, stand, meter, top - SPACE - stand),
        "fader": (pair + meter + PAIR_GAP, stand, FADER_W, top - SPACE - stand),
        "mute": (left, top, TOGGLE_W, BUTTON_H),
        "solo": (left + TOGGLE_W + theme.BUTTON_GAP, top, TOGGLE_W, BUTTON_H),
        "both": (left, top, 2 * TOGGLE_W + theme.BUTTON_GAP, BUTTON_H),
        "name": (x, foot, w, Font.LINE),
    }


class Name(Button):
    """The name of a channel at the foot of its strip, in `colour`: pressed, it does what it was made for.

    The name is cut where the strip is too narrow for it. One that is `picked` stands on a lit ground.
    """

    def __init__(self, rect: Rect, action: Callable[[], object] | None = None) -> None:
        super().__init__(rect, action=action)
        self.colour: Colour = theme.TEXT
        self.picked = False

    def draw(self, canvas: Canvas) -> None:
        """Draw the name in the middle, on a lit ground when it is picked and a raised one under the mouse."""
        if self.picked or self.hot:
            canvas.box(self.rect, theme.LIT if self.picked else theme.RAISED)
        canvas.text_in(self.text, self.rect, self.colour, "centre")


class Fader(Slider):
    """The slider of a strip, which sets a volume. The wheel on it changes it, as on any slider."""

    def __init__(self, rect: Rect, value: float, travel: tuple[float, float, int],
                 changed: Callable[[float], object], settled: Callable[[float], object], colour: Colour,
                 clock: Callable[[], float], help: str) -> None:
        """`travel` is the lowest value, the highest, and the steps between them. A right press is `value` again."""
        low, high, steps = travel
        super().__init__(rect, value, low, high, changed, value, steps, colour, help, settled, clock)


class Desk(Widget):
    """What the strips of a mixer stand on: it takes the wheel for the mixer, and has what is no strip.

    The wheel is read by `kit.wheel`. The mixer goes one way, so the wheel alone moves along the strips, and so
    it does with shift: `wheel` is called with its steps, and answers whether they were used. With ctrl it is
    not used. `line` is the line before the strip of the whole song, which the mixer sets.

    `bar` is the scroll bar under the strips, a part of the desk: the desk draws it, hands it what the mouse
    does on it, and says its line of help there. The mixer gives it its rectangle, and has it seen only while
    not every strip is in view.
    """

    def __init__(self, rect: Rect, wheel: Callable[[int], bool], moved: Callable[[int], object] | None = None,
                 help: str = "") -> None:
        self.bar = ScrollBar((0, 0, 0, 0), moved, BAR_HELP)
        self.bar.visible = False
        self._mouse: Pos = NOWHERE
        super().__init__(rect, help)
        self.wheel = wheel
        self.line: Rect = (0, 0, 0, 0)

    @property
    def help(self) -> str:
        """What the bar says while the mouse is on it, and anywhere else the line the desk was made with."""
        return self.bar.help if self._on_bar(self._mouse) else self._help

    @help.setter
    def help(self, line: str) -> None:
        self._help = line

    def _on_bar(self, pos: Pos) -> bool:
        """Whether a position is on the bar, while there is one."""
        return self.bar.visible and inside(self.bar.rect, pos)

    def on(self, event: Event, ui: Ui) -> bool:
        """Use the wheel where the mixer has a use for it, and hand the bar a press on it and the drag after it."""
        if event.kind in ("key", "text"):
            return False
        self._mouse = event.pos
        if event.kind == "wheel":
            way, steps = wheel(event)
            return way != ZOOM and self.wheel(steps)
        if self.bar.busy or (event.kind == "down" and self._on_bar(event.pos)):
            return self.bar.on(event, ui)
        return False

    def draw(self, canvas: Canvas) -> None:
        """Draw the line before the strip of the song, and the bar while the mixer scrolls."""
        canvas.box(self.line, theme.LINE)
        if self.bar.visible:
            self.bar.hot = self.hot and self._on_bar(self._mouse)
            self.bar.draw(canvas)


@dataclass
class Strip:
    """The widgets of the strip of one channel, and the channel they show.

    `channel` is its name, or None while the strip shows none. `colour` is the colour the channel has in its
    song, and `dim` whether the channel is not heard just now.
    """

    number: Label
    meter: Meter
    fader: Fader
    mute: Toggle
    solo: Toggle
    name: Name
    channel: str | None = None
    colour: Colour = theme.TEXT
    dim: bool = False

    def parts(self) -> dict[str, Widget]:
        """The widgets by the part of the strip each stands in, as `strip_parts` calls them."""
        return {"number": self.number, "meter": self.meter, "fader": self.fader, "mute": self.mute,
                "solo": self.solo, "name": self.name}


class Mixer:
    """Every channel of a song side by side, to set their volumes against each other and to hear some alone.

    It is no panel of the frame. A song panel owns it, adds its widgets through itself, gives it a rectangle
    with `place`, a song with `show` and a frame with `update`, and says with `seen` whether it is on the
    window. `strips` has a strip for each channel, the tuned ones and then the drum channels, as `ordered` has
    them: more are made as a song needs them, and those a song has no channel for show none. A strip has, from
    the top, the channel's volume as a number, a meter beside the fader of that volume, the toggles of its mute
    and its solo, and its name in its colour: a press on the name calls `picked` with it. At the right end
    stands the strip of the whole song: `number`, `meter`, `fader` and `name`, and `clear`, which takes every
    mute and every solo off. Strips that do not fit are scrolled, `first` being the first in view: by the
    wheel anywhere on the mixer but on a fader, and by the scroll bar that then lies under the strips, which
    is the desk's. The wheel on a fader changes its volume, in a mixer that scrolls as in one that shows all.

    A fader shows its volume as it is moved, and is saved when the hand has let go of it, as a knob of the
    channel's bar is: one change to the file, and one step that can be taken back. A volume put back to what
    the engine takes when there is none is taken out of the file. After each change, and whenever the model
    has loaded the file again, the faders show what the file holds, but for one that is under the hand: that
    one shows it when the hand has let go.

    A mute and a solo are the session's, a way of listening that is not in the file. While the mixer is on the
    window its toggles are lit by what the session has muted and soloed, the strip of a channel that is not
    heard is dim, and the meters show how loud each channel and the whole song are now.
    """

    def __init__(self, panel: Panel, picked: Callable[[str], object] | None = None,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        """`panel` is the panel the widgets are added through. `clock` times a run of the wheel on a fader."""
        self.session = panel.session
        self.picked = picked
        self._panel, self._clock = panel, clock
        nowhere = (0, 0, 0, 0)
        self.desk = panel.add(Desk(nowhere, self._wheel, self._go, DESK_HELP))
        self.strips: list[Strip] = []
        self.number = panel.add(Label(nowhere, "", theme.TEXT, "centre"))
        self.meter = panel.add(Meter(nowhere))
        self.meter.levels = (0.0, 0.0)
        self.fader = panel.add(Fader(
            nowhere, SONG_ABSENT, SONG_LEVELS, lambda value: self._tell(), self._put_song, theme.TEXT, clock,
            told("drag up or down for the volume of the whole song, slower with shift",
                 f"back to {plain(SONG_ABSENT)}", "steps it")))
        self.clear = panel.add(Button(nowhere, icon="speaker", action=self._clear))
        self.name = panel.add(Label(nowhere, ALL, theme.TEXT, "centre"))
        #: The names of the song that is shown and of its picked channel, and the first strip in view.
        self.song: str | None = None
        self.channel: str | None = None
        self.first = 0
        #: How many strips show a channel, and how many of those a tuned one.
        self._count = self._tuned = 0
        #: The project the strips were set from last.
        self._project: Project | None = None
        self._rect: Rect = nowhere
        self._laid = lay(nowhere, 0, 0)
        self._on = False
        self._lay()

    @property
    def widgets(self) -> list[Widget]:
        """Every widget of the mixer: the desk, those of the song's strip, and those of each channel's."""
        own = [self.desk, self.number, self.meter, self.fader, self.clear, self.name]
        return own + [widget for strip in self.strips for widget in strip.parts().values()]

    @property
    def busy(self) -> bool:
        """Whether a fader is under the hand: dragged, or in a run of the wheel that has not ended."""
        return self.fader.busy or any(strip.fader.busy for strip in self.strips)

    @property
    def laid(self) -> Laid:
        """Where the strips stand now, as `lay` has them."""
        return self._laid

    def place(self, rect: Rect) -> None:
        """Take the mixer's rectangle, and lay the strips out in it as `lay` has them."""
        self._rect = rect
        self._lay()

    def seen(self, on: bool) -> None:
        """Bring the mixer onto the window, or take it off. Brought on, it shows how the song is listened to now."""
        self._on = on
        self._lay()
        if on:
            self._hear()

    def show(self, song: str | None, picked: str | None = None) -> None:
        """Show the channels of a song of the project, by its name, with the name of the picked one lit.

        A song the project does not have, and None, have no channels to show. The song that is shown already
        keeps its view, and a fader under the hand is left to the hand. Another song is shown from its first
        strip, and what the hand was doing to a fader is dropped, with nothing saved.
        """
        if song != self.song:
            self.first = 0
            for fader in (self.fader, *(strip.fader for strip in self.strips)):
                fader.drop()
        self.song, self.channel = song, picked
        self._read()

    def update(self, passed: float = 0.0) -> None:
        """Give the faders their frame, read the song again where the file was loaded again, and show what is heard.

        The frame of a fader ends a run of the wheel whose time is up, which saves it. While the mixer is on the
        window the toggles, the dim strips and the meters are set from the session: `passed` is the seconds
        since the frame before, for which the lines kept at the top of the meters fall.
        """
        for fader in (self.fader, *(strip.fader for strip in self.strips)):
            fader.update()
        if self.session.model.project is not self._project:
            self._read()
        if self._on:
            self._hear()
            levels = self.session.levels()
            for strip in self.strips[:self._count]:
                strip.meter.levels = (levels.get(strip.channel, 0.0),)
                strip.meter.fall(passed)
            self.meter.levels = self.session.level()
            self.meter.fall(passed)

    def _strip(self) -> Strip:
        """One more strip, its widgets added through the panel: it shows no channel yet."""
        add, nowhere = self._panel.add, (0, 0, 0, 0)
        strip = Strip(
            add(Label(nowhere, "", theme.TEXT, "centre")), add(Meter(nowhere)),
            add(Fader(nowhere, ABSENT["gain"], LEVELS, lambda value: self._paint(strip),
                      lambda value: self._put(strip, value), theme.TEXT, self._clock,
                      told("drag up or down for the volume of the channel, slower with shift",
                           f"back to {plain(ABSENT['gain'])}", "steps it"))),
            add(Toggle(nowhere, MUTE, colour=theme.MUSTARD, changed=lambda on: self._press(strip, False),
                       help=told("mute the channel, or hear it again"))),
            add(Toggle(nowhere, SOLO, colour=theme.GOOD, changed=lambda on: self._press(strip, True),
                       help=told("solo the channel, so that only the soloed ones are heard, or take its solo off"))),
            add(Name(nowhere, lambda: self._name(strip))))
        return strip

    def _read(self) -> None:
        """Set every strip from what the project holds for the song now, and lay them out.

        A strip that another channel takes lets go of what the hand was doing to its fader. A fader that is
        under the hand keeps the value the hand has given it.
        """
        project = self._project = self.session.model.project
        song = project.songs.get(self.song, {})
        channels = song.get("channels", {})
        found = ordered(channels)
        self._count, self._tuned = len(found), sum(not channels[name].get("drums") for name in channels)
        while len(self.strips) < len(found):
            self.strips.append(self._strip())
        for strip, (place, name) in zip(self.strips, found):
            if strip.channel != name:
                strip.fader.drop()
                strip.channel, strip.dim = name, False
            strip.colour = theme.CHANNELS[place % len(theme.CHANNELS)]
            if not strip.fader.busy:
                strip.fader.value = channels[name].get("gain", ABSENT["gain"])
            strip.name.text, strip.name.picked = name, name == self.channel
            strip.name.help = f"channel {name}, {channels[name]['inst']}   {told('show its notes')}"
            self._paint(strip)
        for strip in self.strips[len(found):]:
            strip.fader.drop()
            strip.channel = None
        if not self.fader.busy:
            self.fader.value = song.get("gain", SONG_ABSENT)
        self._tell()
        self._lay()

    def _tell(self) -> None:
        """Have the strip of the whole song show its volume as a number."""
        self.number.text = plain(self.fader.value)

    def _paint(self, strip: Strip) -> None:
        """Have a strip show its volume as a number, and be in its channel's colour, or dim where it is not heard."""
        colour = mix(theme.PANEL, strip.colour, DIMMED) if strip.dim else strip.colour
        strip.fader.colour = strip.name.colour = colour
        strip.number.text, strip.number.colour = plain(strip.fader.value), theme.DIM if strip.dim else theme.TEXT

    def _lay(self) -> None:
        """Give each widget its rectangle, and have it seen where the mixer is on the window and it has room."""
        laid = self._laid = lay(self._rect, self._tuned, self._count - self._tuned, self.first)
        self.first = laid.first
        nowhere = (self._rect[0], self._rect[1], 0, 0)
        placed = [(strip.parts(), strip_parts(laid.strips.get(index, nowhere)))
                  for index, strip in enumerate(self.strips)]
        whole = strip_parts(laid.song, SONG_METER_W)
        placed.append(({"number": self.number, "meter": self.meter, "fader": self.fader, "both": self.clear,
                        "name": self.name}, whole))
        for widgets, parts in placed:
            for part, widget in widgets.items():
                widget.rect, widget.visible = parts[part], self._on and parts[part][2] > 0
        x, y, w, h = laid.song
        self.desk.rect, self.desk.visible = self._rect, self._on
        self.desk.line = (x - APART // 2 - 1, y, 1, h) if w else nowhere
        bar = self.desk.bar
        bar.rect, bar.visible = laid.bar or nowhere, laid.bar is not None
        bar.show(*along(laid, self._count))

    def _hear(self) -> None:
        """Show how the session listens to the song: the toggles lit, the strips that are not heard dim.

        The button that clears is enabled where a channel is muted or soloed, and its help says what it does.
        """
        session = self.session
        heard, muted, soloed = session.model.heard, session.muted, session.soloed
        for strip in self.strips[:self._count]:
            strip.mute.is_on, strip.solo.is_on = strip.channel in muted, strip.channel in soloed
            strip.dim = heard is not None and strip.channel not in heard
            self._paint(strip)
        self.clear.enabled = bool(muted or soloed)
        self.clear.help = (told("hear every channel again: no mute and no solo") if self.clear.enabled
                           else "nothing is muted or soloed")

    def _wheel(self, steps: int) -> bool:
        """Move the view by steps of the wheel, up being towards the first strip. False where every strip is in view."""
        if not self._laid.most:
            return False
        self.first -= steps * WHEEL_STRIPS
        self._lay()
        return True

    def _go(self, offset: int) -> None:
        """Move the view to where the scroll bar was taken: the strip nearest to so many pixels is the first in view."""
        self.first = (2 * offset + self._laid.wide) // (2 * max(1, self._laid.wide))
        self._lay()

    def _press(self, strip: Strip, solo: bool) -> None:
        """Have the session flip the solo of a strip's channel, or else its mute, and show what it then has.

        What the session refuses stays as it was, and the session says why.
        """
        if solo:
            self.session.solo(strip.channel)
        else:
            self.session.mute(strip.channel)
        self._hear()

    def _clear(self) -> None:
        """Have every channel heard again."""
        self.session.hear_all()
        self._hear()

    def _name(self, strip: Strip) -> None:
        """Tell that the name of a strip's channel was pressed."""
        if self.picked is not None:
            self.picked(strip.channel)

    def _save(self, where: str, owner: str, value: float, absent: float) -> None:
        """Save the volume a fader was left at, at a place of the song, and show what the file then holds.

        It is written to two places after the point, called by whose volume it is and by the number, and taken
        out of the file where it is `absent`. A change that is refused is not made, and the session says why.
        """
        value = tidy(value)
        save(self.session, where, value, f"{owner}: volume {plain(value)}", absent)
        self._read()

    def _put(self, strip: Strip, value: float) -> None:
        """Save the volume the fader of a strip was left at, as the volume of its channel."""
        self._save(f"songs.{self.song}.channels.{strip.channel}.gain", strip.channel, value, ABSENT["gain"])

    def _put_song(self, value: float) -> None:
        """Save the volume the fader of the whole song was left at, as the volume of the song."""
        self._save(f"songs.{self.song}.gain", self.song, value, SONG_ABSENT)
