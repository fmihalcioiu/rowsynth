"""The settings bar of a sound: the voices of one sound of an entry as tabs, its instrument, its volume and its echo."""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..errors import RowSynthError
from . import soundedit, theme
from .channelbar import INST_W, tidy
from .dials import Knob, Stepper, plain
from .font import Font
from .kit import Bar, Plus, Slot, thing_menu, told
from .layout import Rect
from .panels import within
from .popups import Item, Menu, ask
from .theme import Colour
from .ui import Button, Choice, Event, Pos, Toggle, Ui, draw_button
from .wave import called, voice_colour

if TYPE_CHECKING:
    from .canvas import Canvas
    from .panels import Panel

#: What stands over the voices of the one sound of an entry, and what the settings are called. The last four are
#: the echo's: how many seconds after itself the sound comes back (its `time`), how loud that first return is
#: (its `mix`), how much of each return comes back once more (its `feedback`), and how bright the returns stay
#: (its `lp`, the hertz of the low-pass each goes through). The switch of the echo reads one of the two words
#: after them.
VOICE = "VOICE"
INSTRUMENT = "INSTRUMENT"
VOLUME = "VOLUME"
ECHO = "ECHO"
TIME = "TIME"
AMOUNT = "AMOUNT"
REPEATS = "REPEATS"
TONE = "TONE"
ON, OFF = "ON", "OFF"
#: What the two lengths of a held sound are called that have no handle on its clock: the ticks at the end of its
#: loop that are faded into what led up to it (its `join`), and the ticks its loop takes to die away once it is
#: let go (its `let_go`).
CROSSFADE = "CROSSFADE"
RELEASE = "RELEASE"
#: What the field of the instrument reads for one that is written in place in its voice, and the item of the
#: list of the instruments that makes such a one an instrument of the project.
OWN = "its own"
MAKE_SHARED = "Make shared..."

#: The travel of each knob: its lowest value, its highest, and how many steps there are between them. A project
#: may hold any number for each, so the travels are chosen here. A volume goes in half decibels up to the 3 that
#: bring a sound to full scale, the seconds of an echo in hundredths up to a second, its two shares in
#: hundredths, the second up to the most the engine lets come back, and its low-pass in hundreds of hertz. The
#: two lengths of a held sound go in whole ticks, up to half a second and up to a second.
VOLUMES = (-30.0, 3.0, 66)
CROSSFADES = (0.0, 30.0, 30)
RELEASES = (0.0, 60.0, 60)
TIMES = (0.01, 1.0, 99)
AMOUNTS = (0.0, 1.0, 100)
REPEATING = (0.0, 0.95, 95)
TONES = (200.0, 12000.0, 118)
#: The echo a sound gets when its echo is switched on and was not switched off here before: the one the example
#: uses. And what the engine takes for the settings an echo leaves out.
USUAL_ECHO = {"time": 0.09, "feedback": 0.35, "mix": 0.3, "lp": 4000}
ABSENT = {"feedback": 0.3, "mix": 0.3, "lp": 3000}
#: What the knob of the echo's seconds says of itself while the echo has several times.
SEVERAL_HELP = "this echo has several times, which are left as the file has them"


def seconds_text(value: float) -> str:
    """A number of seconds as a knob shows it."""
    return f"{plain(value)} s"


def hertz_text(value: float) -> str:
    """A number of hertz as a knob shows it."""
    return f"{plain(value)} Hz"


def db_text(value: float) -> str:
    """A volume in decibels as a knob shows it."""
    return f"{plain(value)} dB"


def ticks_text(value: float) -> str:
    """A number of ticks as a knob shows it, as "1 tick" and "3 ticks"."""
    return f"{plain(value)} {'tick' if value == 1 else 'ticks'}"


def knob_room(name: str, *values: str) -> int:
    """How wide the room of a knob is: `KNOB_W` at the least, and what its name and its widest values need."""
    return max(KNOB_W, *(Font.width(text) + 2 for text in (name, *values)))


#: How wide the tab of one voice is, the least room a knob has with its two lines, and the button that opens the
#: list of the instruments. The room of each knob is that of its name or of its widest value.
VOICE_W = Font.width("0") + 2 * theme.PAD
KNOB_W = 33
LIST_W = theme.BUTTON_H
VOLUME_W = knob_room(VOLUME, db_text(-29.5))
CROSSFADE_W = knob_room(CROSSFADE, ticks_text(CROSSFADES[1]))
RELEASE_W = knob_room(RELEASE, ticks_text(RELEASES[1]))
ECHO_W = Font.width(ECHO) + 2 * theme.PAD
TIME_W = knob_room(TIME, seconds_text(0.99))
AMOUNT_W = knob_room(AMOUNT, plain(0.99))
REPEATS_W = knob_room(REPEATS, plain(0.95))
TONE_W = knob_room(TONE, hertz_text(TONES[1]))


def voices_room(word: str, voices: int) -> int:
    """How wide the room of the voices is: what the word over them needs, or the tabs and the sign after them."""
    return max(Font.width(word), voices * VOICE_W + theme.BUTTON_GAP + theme.PLUS)


def voice_parts(room: Rect, voices: int) -> tuple[Rect, Rect]:
    """Where the tabs of so many voices go in the room they have in the bar, and the sign that adds one after them.

    The tabs stand from the left, each `VOICE_W` wide, and the sign close after the last, in the middle of the
    height. A room too narrow for them cuts the tabs, and a room of no width leaves neither any.
    """
    x, y, w, h = room
    wide = max(0, min(voices * VOICE_W, w - theme.BUTTON_GAP - theme.PLUS))
    sign = (x + wide + theme.BUTTON_GAP, y + (h - theme.PLUS) // 2, theme.PLUS, theme.PLUS)
    return (x, y, wide, h), within(sign, room)


class Voices(Choice):
    """The voices of a sound as a row of tabs: each has its number, from 1, in the colour of its bars.

    `colour` is that of the first voice, and the others follow as `wave.voice_colour` has them. A press of the
    left button picks a voice. One of the right button on a tab calls `asked` with the voice, counted from 0,
    and the position, for its owner to open the menu of that voice at.
    """

    def __init__(self, rect: Rect, changed: Callable[[int], object] | None = None,
                 asked: Callable[[int, Pos], object] | None = None, help: str = "") -> None:
        super().__init__(rect, ("1",), 0, changed, help)
        self.asked = asked
        self.colour: Colour = theme.SOUND

    def show(self, count: int, picked: int) -> None:
        """Have a tab for each of so many voices, with one of them on. Nothing is told of it."""
        self.options, self.picked = tuple(str(number + 1) for number in range(count)), picked

    def on(self, event: Event, ui: Ui) -> bool:
        """Tell of a press of the right button on a tab, and leave anything else to the row of tabs."""
        if event.kind != "down" or event.button != 3:
            return super().on(event, ui)
        voice = self.at(event.pos)
        if voice is None or self.asked is None or ui.held is not None:
            return False
        self.asked(voice, event.pos)
        return True

    def draw(self, canvas: Canvas) -> None:
        """Draw the voices as tabs that share the line between them. The one that is on is drawn last, whole."""
        cells = self.cells()
        for index in sorted(range(len(cells)), key=lambda index: index == self.picked):
            x, y, w, h = cells[index]
            ink = voice_colour(self.colour, index)
            lit = self.hot and index == self._over
            # All but the last reach one pixel into the next, so that the two outlines are one line.
            draw_button(canvas, (x, y, w + (index < len(cells) - 1), h), self.options[index], "", ink, lit,
                        lit and self.held, ink if index == self.picked else None)


class SoundBar:
    """The settings bar of a sound panel: the settings of one sound of the picked entry of the project.

    It is no panel of the frame. A sound panel owns it, adds its widgets through itself, gives it a rectangle
    with `place`, a sound with `show` and a frame with `update`, and says with `seen` whether it is on the
    window. `frame` is the bar every editor has, as `kit.Bar` draws it: it lays the controls out, and has the
    name of each over it. From the left stand `voices`, the voices of the sound as tabs, with `add` after the
    last, the sign that adds one: over them stands `word`, which is VOICE, or for an entry of several sounds
    the words for the one the bar is of. Then come `inst` under INSTRUMENT with `list` after it, the button
    that opens the list of the instruments, `volume`, for a held sound `crossfade` and `release`, `echo`, and
    while the echo is on its knobs `time`, `amount`, `repeats` and `tone`. In a bar too narrow for all of
    them the last give way first.

    `voice` is the picked voice: the one whose instrument `inst` shows. The right button on the tab of a voice
    opens its menu, in the order every such menu has: a voice has no name, it is copied with its notes as one
    more voice, and it is removed, but for the last voice of a sound.

    The bar saves nothing. What a widget makes of the sound is handed to `edited` as the new definition and a
    few words for the list of what can be undone, and a volume to `levelled` as the `db` of the entry, None for
    none, with such words. The two knobs of a held sound are its other two lengths, which have no handle on
    its clock: let go of, such a knob is handed to `lengthed` as the key of the length, its ticks, and such
    words. The owner saves, and shows the bar the sound again: an owner that answers false has not saved.
    `voiced` is called with the picked voice when one is picked here, and after a voice was added, copied or
    removed. A change that the rules of `soundedit` refuse is handed to nobody, and the session says why.

    A knob shows its value as it is turned, and is handed on when the hand has let go of it. A voice that is
    added plays the instrument of the picked one, and is then the picked one. The instrument is one of the
    project's, stepped to with `inst` or chosen from the list. A voice whose instrument is written in place
    shows `OWN`, and that is replaced only when another is stepped to or chosen. For such a voice the list has
    `MAKE_SHARED`: a box takes a name, and `shared` is called with the voice and the name, for the owner to
    make the instrument one of the project's under it. A volume put back to 0 is taken out of the file, and an
    entry of parts has no volume. An echo that is switched off is taken out of the file, and its settings are
    kept here, so that switched on again it is the echo it was. Switched on for the first time it is
    `USUAL_ECHO`. The knob of its seconds is not for an echo of several times.
    """

    def __init__(self, panel: Panel, edited: Callable[[dict, str], object],
                 levelled: Callable[[object, str], object], voiced: Callable[[int], object] | None = None,
                 clock: Callable[[], float] = time.perf_counter,
                 lengthed: Callable[[str, float, str], object] | None = None,
                 shared: Callable[[int, str], object] | None = None) -> None:
        """`panel` is the panel the widgets are added through. `clock` times a run of the wheel on a knob."""
        self.session = panel.session
        self.ui: Ui = panel.ui
        self.edited = edited
        self.levelled = levelled
        self.voiced = voiced
        self.lengthed = lengthed
        self.shared = shared
        nowhere = (0, 0, 0, 0)

        def knob(travel: tuple[float, float, int], usual: float | None, settled: Callable[[float], object],
                 what: str, show: Callable[[float], str] = plain) -> Knob:
            """A knob of the bar: a right press puts it back to `usual` where it has one, and let go of, it is
            `settled`."""
            low, high, steps = travel
            back = "" if usual is None else f"back to {show(usual)}"
            help = told(f"drag up or down {what}, slower with shift", back, "steps it")
            return panel.add(Knob(nowhere, low if usual is None else usual, low, high, default=usual, steps=steps,
                                  show=show, help=help, clock=clock, settled=settled))

        def echo(travel: tuple[float, float, int], key: str, words: str, what: str,
                 show: Callable[[float], str] = plain) -> Knob:
            """A knob of the echo: let go of, its value is the echo's setting `key`."""
            usual = ABSENT.get(key, USUAL_ECHO[key])
            return knob(travel, usual, lambda value: self._put(key, value, words, show), what, show)

        # The bar itself is added first, so that it is drawn under its controls.
        self.frame = panel.add(Bar(nowhere))
        self.voices = panel.add(Voices(nowhere, self._pick, self._ask_voice, told(
            "pick the voice that new notes go into, and whose instrument is shown", "its menu")))
        self.add = panel.add(Plus(nowhere, self._add, told("add a voice that plays the instrument of the picked one")))
        self.inst = panel.add(Stepper(nowhere, ("",), changed=self._step, help=told(
            "an arrow steps to the next instrument of the project", wheel="steps through them")))
        self.list = panel.add(Button(nowhere, action=self._ask_inst, arrow=True, help=told(
            "the list of the instruments, and what makes one that is written in place a shared one")))
        self.volume = knob(VOLUMES, 0, self._volume, "for how loud the sound is", db_text)
        self.crossfade = knob(CROSSFADES, None, lambda value: self._length("join", value, "crossfade"),
                              "for how many ticks the end of the loop fades into what came before it", ticks_text)
        self.release = knob(RELEASES, None, lambda value: self._length("let_go", value, "release"),
                            "for how many ticks the loop takes to die away once the sound is let go", ticks_text)
        self.echo = panel.add(Toggle(nowhere, OFF, changed=self._switch, help=told(
            "switch the echo on or off: with it the whole sound comes back after itself, quieter each time")))
        self.time = echo(TIMES, "time", "time", "for how long after itself the sound comes back", seconds_text)
        self.amount = echo(AMOUNTS, "mix", "amount", "for how loud the sound comes back the first time")
        self.repeats = echo(REPEATING, "feedback", "repeats", "for how much of each return comes back once more")
        self.tone = echo(TONES, "lp", "tone", "for how bright the returns stay", hertz_text)
        #: The name of the entry that is shown and which of its sounds, as `soundedit.sound_in` counts them.
        self.name: str | None = None
        self.index = 0
        #: The picked voice of the sound, counted from 0, and what stands over the voices.
        self.voice = 0
        self.word = VOICE
        self._colour: Colour = theme.SOUND
        #: What the project held for the entry when the widgets were set last, and its sound that is shown.
        self._entry: dict | None = None
        self._defn: dict | None = None
        #: The settings of each echo that was switched off here, by its entry and its sound.
        self._echoes: dict[tuple[str, int], dict] = {}
        #: The knobs, what the knob of the echo's seconds says while it is of use, and the controls that have a
        #: room in the bar as it is laid out now.
        self._knobs = (self.volume, self.crossfade, self.release, self.time, self.amount, self.repeats, self.tone)
        self._time_help = self.time.help
        self._roomed: tuple[object, ...] = ()
        self._on = False
        self._lay()
        self._see()

    @property
    def busy(self) -> bool:
        """Whether a knob is under the hand: dragged, or in a run of the wheel that has not ended."""
        return any(knob.busy for knob in self._knobs)

    def place(self, rect: Rect) -> None:
        """Take the bar's rectangle, and lay the controls out in it."""
        self.frame.rect = rect
        self._lay()
        self._see()

    def seen(self, on: bool) -> None:
        """Bring the bar onto the window, or take it off. On it, a widget is seen by what there is to show."""
        self._on = on
        self._see()

    def show(self, name: str | None, index: int = 0, colour: Colour = theme.SOUND) -> None:
        """Show one sound of an entry of the project's sounds, by the name of the entry and the place of the sound.

        `colour` is that of the first voice. With None for the name, or a name the project does not have,
        nothing is shown. Another entry than before has its first voice picked. What the hand was doing to a
        knob is dropped, with nothing handed on.
        """
        if name != self.name:
            self.voice = 0
        self.name, self.index, self._colour = name, index, colour
        for knob in self._knobs:
            knob.drop()
        self._read()

    def update(self) -> None:
        """Give the knobs their frame, which ends a run of the wheel whose time is up and hands its value on."""
        for knob in self._knobs:
            knob.update()

    def _read(self) -> None:
        """Set every widget from what the project holds now for the sound that is shown."""
        project = self.session.model.project
        entry = self._entry = project.sounds.get(self.name) if self.name is not None else None
        self._defn = None
        if entry is not None:
            defn = self._defn = soundedit.sound_in(entry, self.index)
            voices = defn["voices"]
            self.voice = min(self.voice, len(voices) - 1)
            self.voices.show(len(voices), self.voice)
            inst, names = voices[self.voice]["inst"], tuple(project.instruments)
            # A name is one of the project's. What is none of them stands before them: an instrument written in
            # place, under the words for one.
            named = inst in names if isinstance(inst, str) else False
            self.inst.options = names if named else (inst if isinstance(inst, str) else OWN, *names)
            self.inst.picked = names.index(inst) if named else 0
            level = entry.get("db", 0)
            self.volume.value = level[self.index] if isinstance(level, list) else level
            if "held" in entry:
                self.crossfade.value, self.release.value = entry["held"]["join"], entry["held"]["let_go"]
            echo = defn.get("echo")
            self.echo.is_on, self.echo.text = echo is not None, OFF if echo is None else ON
            # An echo of several times has no one number for the knob of its seconds.
            one = echo is None or isinstance(echo["time"], (int, float))
            self.time.enabled, self.time.help = one, self._time_help if one else SEVERAL_HELP
            if echo is not None:
                self.time.value = echo["time"] if one else self.time.value
                for knob, key in ((self.amount, "mix"), (self.repeats, "feedback"), (self.tone, "lp")):
                    knob.value = echo.get(key, ABSENT[key])
            self.word = called(soundedit.kind_of(entry), self.index).upper() or VOICE
            self.voices.colour = self._colour
            for knob in self._knobs:
                knob.colour = self._colour
        self._lay()
        self._see()

    def _lay(self) -> None:
        """Give each control its rectangle in the bar, laid out for the sound that is shown.

        The tabs of the voices and the sign after them share the room under the word for them. An entry of
        parts has no room for a volume, and only a held sound has one for its two lengths.
        """
        entry, defn = self._entry or {}, self._defn
        voices = len(defn["voices"]) if defn else 1
        left = [Slot(self.word, None, voices_room(self.word, voices), "left"), Slot(INSTRUMENT, self.inst, INST_W),
                Slot("", self.list, LIST_W)]
        if "parts" not in entry:
            left.append(Slot(VOLUME, self.volume, VOLUME_W))
        if "held" in entry:
            left += [Slot(CROSSFADE, self.crossfade, CROSSFADE_W), Slot(RELEASE, self.release, RELEASE_W)]
        left += [Slot(ECHO, self.echo, ECHO_W), Slot(TIME, self.time, TIME_W), Slot(AMOUNT, self.amount, AMOUNT_W),
                 Slot(REPEATS, self.repeats, REPEATS_W), Slot(TONE, self.tone, TONE_W)]
        rects = self.frame.lay(left)
        self.voices.rect, self.add.rect = voice_parts(rects[0], voices)
        self._roomed = (self.voices, self.add, *(slot.widget for slot in left[1:]))

    def _see(self) -> None:
        """Have each widget seen or not: by whether the bar is on the window, what there is to show, and what fits.

        The knobs of the echo are seen while it is on, and a control with no room in the bar is not seen.
        """
        on = self._on and self._defn is not None
        self.frame.visible = on
        for widget in (self.voices, self.add, self.inst, self.list, self.echo, *self._knobs):
            widget.visible = on and widget in self._roomed and widget.rect[2] > 0
        for knob in (self.time, self.amount, self.repeats, self.tone):
            knob.visible = knob.visible and self.echo.is_on

    def _make(self, change: Callable[[dict], dict], label: str) -> bool:
        """Hand on the sound as a change leaves it, and show what the project then holds. Whether it was saved.

        A change the rules refuse is handed to nobody: the session's message is the refusal.
        """
        try:
            defn = change(self._defn)
        except RowSynthError as error:
            self.session.say(f"{self.name}: {error}")
            saved = False
        else:
            saved = self.edited(defn, label) is not False
        self._read()
        return saved

    def _menu(self, items: list[Item | None], at: Pos) -> None:
        """Open a menu at a place, over everything else. Each item stands for what is done when it is chosen."""
        self.ui.open(Menu(items, at, chosen=lambda act: act()))

    def _pick(self, voice: int) -> None:
        """Take a voice as the picked one, show its instrument, and tell of it."""
        self.voice = voice
        self._read()
        if self.voiced is not None:
            self.voiced(self.voice)

    def _add(self) -> None:
        """Add a voice that plays the instrument of the picked one. Saved, it is the picked one."""
        voices = self._defn["voices"]
        inst = voices[self.voice]["inst"]
        if self._make(lambda defn: soundedit.voice_added(defn, inst), "voice added"):
            self._pick(len(voices))

    def _ask_voice(self, voice: int, pos: Pos) -> None:
        """Open the menu of a voice at the mouse: it is copied or removed. The last voice of a sound is not removed."""
        last = len(self._defn["voices"]) == 1
        self._menu(thing_menu(None, lambda: self._copy(voice), (), None if last else lambda: self._drop(voice)), pos)

    def _copy(self, voice: int) -> None:
        """Add a copy of a voice, with its instrument and its notes. Saved, it is the picked one."""
        voices = self._defn["voices"]
        if self._make(lambda defn: soundedit.voice_copied(defn, voice), f"voice {voice + 1} copied"):
            self._pick(len(voices))

    def _drop(self, voice: int) -> None:
        """Remove a voice. Saved, the voice that was picked still is: or, where it is the one gone, the voice that
        has its number, or else the last."""
        picked = self.voice - (voice < self.voice)
        if self._make(lambda defn: soundedit.voice_dropped(defn, voice), f"voice {voice + 1} removed"):
            self._pick(min(picked, len(self._defn["voices"]) - 1))

    def _step(self, index: int) -> None:
        """Have the picked voice play the instrument the stepper was put on."""
        self._name(self.inst.options[index])

    def _name(self, name: str) -> None:
        """Have the picked voice play an instrument of the project, by its name."""
        voice = self.voice
        self._make(lambda defn: soundedit.voice_inst(defn, voice, name), f"voice {voice + 1} plays {name}")

    def _ask_inst(self) -> None:
        """Open the list of the instruments under the field of the instrument: one that is chosen plays the voice.

        After them stands what makes a shared instrument of one that is written in place. For a voice that
        names its instrument it cannot be chosen.
        """
        inst = self._defn["voices"][self.voice]["inst"]
        items: list[Item | None] = [Item(name, lambda name=name: self._name(name), enabled=name != inst)
                                    for name in self.session.model.project.instruments]
        if items:
            items.append(None)
        items.append(Item(MAKE_SHARED, self._ask_shared, enabled=isinstance(inst, dict)))
        x, y, _, h = self.inst.rect
        self._menu(items, (x, y + h))

    def _ask_shared(self) -> None:
        """Open a box over the field of the instrument that takes a name: enter has the owner make the instrument
        of the picked voice one of the project's under it."""
        voice = self.voice

        def done(typed: str) -> None:
            if self.shared is not None:
                self.shared(voice, typed)
            self._read()

        ask(self.ui, self.inst.rect, "", done,
            help="the name the instrument gets in the project: enter takes it, escape leaves it")

    def _switch(self, on: bool) -> None:
        """Hand the echo on as it was switched: on, it has the settings it was switched off with here, or the usual."""
        kept = (self.name, self.index)
        if on:
            echo = dict(self._echoes.get(kept, USUAL_ECHO))
            self._make(lambda defn: soundedit.with_echo(defn, echo), "echo on")
            return
        echo = self._defn.get("echo")
        if self._make(lambda defn: soundedit.with_echo(defn, None), "echo off") and echo is not None:
            self._echoes[kept] = echo

    def _put(self, key: str, value: float, words: str, text: Callable[[float], str]) -> None:
        """Hand on the echo with the value a knob was left at, written as `tidy` has it, as its setting `key`."""
        value = tidy(value)
        self._make(lambda defn: soundedit.with_echo(defn, {**defn.get("echo", {}), key: value}),
                   f"echo {words} {text(value)}")

    def _length(self, key: str, value: float, words: str) -> None:
        """Hand on the ticks a knob of a held sound was left at, as its length `key`, and show what is then held."""
        if self.lengthed is not None:
            self.lengthed(key, tidy(value), f"{words} {ticks_text(tidy(value))}")
        self._read()

    def _volume(self, value: float) -> None:
        """Hand on the volume the knob was left at: the entry's own, or that of this variant where each has one."""
        value, level = tidy(value), self._entry.get("db")
        words = f"volume {db_text(value)}"
        if isinstance(level, list):
            words = f"{called(soundedit.kind_of(self._entry), self.index)} {words}"
            value = [*level[:self.index], value, *level[self.index + 1:]]
        self.levelled(value or None, words)
        self._read()
