"""The settings of one sound of an entry in one bar: its voices, the instrument of one, its level and its echo."""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..errors import RowSynthError
from . import channelbar, soundedit, theme
from .channelbar import tidy
from .dials import Knob, Stepper, knob_parts, plain
from .font import Font
from .layout import Rect
from .panels import within
from .theme import Colour
from .ui import Button, Choice, Label, Toggle, draw_button
from .wave import called, voice_colour

if TYPE_CHECKING:
    from .canvas import Canvas
    from .panels import Panel

#: How high the bar is, how high a small button of it, and from one part of it to the next: as in the bar of a
#: channel.
HEIGHT = channelbar.HEIGHT
BUTTON_H = channelbar.BUTTON_H
SPACE = channelbar.SPACE
#: How wide the room is for the words that say which sound of several the bar is of, the button of one voice,
#: the field of the instrument, and a knob with its two lines: the widest of those is the hertz of the low-pass.
WORD_W = Font.width("VARIANT 10")
VOICE_W = Font.width("0") + 2 * theme.PAD
INST_W = channelbar.INST_W
KNOB_W = Font.width("12000 HZ")

#: What the parts are called, the two buttons that add a voice and remove one, and what the field of the
#: instrument reads for one that is written in place in its voice.
VOICE = "VOICE"
ADD = "+"
DROP = "-"
INSTRUMENT = "INSTRUMENT"
OWN = "own"
LEVEL = "LEVEL"
ECHO = "ECHO"
#: What the settings of the echo are called, by what they do: how many seconds after itself the sound comes back
#: (its `time`), how loud that first return is (its `mix`), how much of each return comes back once more (its
#: `feedback`), and how bright the returns stay (its `lp`, the hertz of the low-pass each goes through).
AFTER = "AFTER"
LOUD = "LOUD"
AGAIN = "AGAIN"
BRIGHT = "BRIGHT"
#: What the two lengths of a held sound are called that have no handle on its clock: the ticks at the end of its
#: loop that are faded into what led up to it (its `join`), and the ticks its loop takes to die away once it is
#: let go (its `let_go`). And what each says of itself.
JOIN = "JOIN"
LET_GO = "LET GO"
JOIN_HELP = "how many ticks at the end of the loop fade into what came before it, so that it goes round without a click"
LET_GO_HELP = "how many ticks the loop takes to die away once the sound is let go"

#: The travel of each knob: its lowest value, its highest, and how many steps there are between them. A project
#: may hold any number for each, so the travels are chosen here. A level goes in half decibels up to the 3 that
#: bring a sound to full scale, the seconds of an echo in hundredths up to a second, its two shares in
#: hundredths, the second up to the most the engine lets come back, and its low-pass in hundreds of hertz. The
#: two lengths of a held sound go in whole ticks, up to half a second and up to a second.
LEVELS = (-30.0, 3.0, 66)
JOINS = (0.0, 30.0, 30)
LET_GOS = (0.0, 60.0, 60)
AFTERS = (0.01, 1.0, 99)
LOUDS = (0.0, 1.0, 100)
AGAINS = (0.0, 0.95, 95)
BRIGHTS = (200.0, 12000.0, 118)
#: The echo a sound gets when its echo is switched on and was not switched off here before: the one the example
#: uses. And what the engine takes for the settings an echo leaves out.
USUAL_ECHO = {"time": 0.09, "feedback": 0.35, "mix": 0.3, "lp": 4000}
ABSENT = {"feedback": 0.3, "mix": 0.3, "lp": 3000}

#: The parts that give way in a bar too narrow for all of them, the first first.
DROPPED = (("after", "loud", "again", "bright"), ("level",), ("word",), ("echo",), ("inst",), ("add", "drop"),
           ("voices",))
#: The two knobs of a held sound, which stand after the level. They give way after the knobs of the echo.
LENGTHS = ("join", "let_go")
#: The parts that are knobs: they are as high as the bar, and the others as high as a button.
KNOBS = ("level", *LENGTHS, "after", "loud", "again", "bright")
#: What the knob of the echo's seconds says of itself, and what it says while the echo has several times.
AFTER_HELP = "how many seconds after itself the sound comes back"
SEVERAL_HELP = "this echo has several times, which are left as the file has them"


def seconds_text(value: float) -> str:
    """A number of seconds as a knob shows it."""
    return f"{plain(value)} S"


def hertz_text(value: float) -> str:
    """A number of hertz as a knob shows it."""
    return f"{plain(value)} HZ"


def db_text(value: float) -> str:
    """A level in decibels as a knob shows it."""
    return f"{plain(value)} DB"


def ticks_text(value: float) -> str:
    """A number of ticks as a knob shows it."""
    return f"{plain(value)} TICKS"


def arrange(rect: Rect, voices: int = 1, several: bool = False, held: bool = False) -> dict[str, Rect]:
    """Where the parts of the bar go in its rectangle, each by what it is called, and each cut to the bar.

    From the left stand "word", for a sound that is one of `several`, then "voices" with a button for each of
    so many voices and "add" and "drop" close after it, with "voice" over the three, "inst" with "heading" over
    it, "level", "echo", and the knobs of the echo: "after", "loud", "again" and "bright". For a `held` sound
    "join" and "let_go" stand between the level and the echo, and for any other there are no such parts. A bar
    too narrow for all of them drops parts in the order of `DROPPED` until the rest fit, those two after the
    knobs of the echo, so no two ever overlap. A part that was dropped has no width, and neither has a heading
    with nothing under it.
    """
    x, y, w, h = rect
    widths = {"word": WORD_W, "voices": voices * VOICE_W, "add": VOICE_W, "drop": VOICE_W, "inst": INST_W,
              "level": KNOB_W, **({part: KNOB_W for part in LENGTHS} if held else {}),
              "echo": Font.width(ECHO) + 2 * theme.PAD, "after": KNOB_W, "loud": KNOB_W, "again": KNOB_W,
              "bright": KNOB_W}
    # The two buttons stand close after the voices they add to and take from.
    gaps = {"voices": theme.BUTTON_GAP, "add": theme.BUTTON_GAP}
    kept = [part for part in widths if several or part != "word"]
    for group in (DROPPED[0], LENGTHS, *DROPPED[1:]):
        # What is kept takes its widths, and after each part but the last the gap to the next.
        if sum(widths[part] for part in kept) + sum(gaps.get(part, SPACE) for part in kept[:-1]) <= w:
            break
        kept = [part for part in kept if part not in group]
    top = y + (h - BUTTON_H) // 2
    placed = {part: (x, y, 0, 0) for part in (*widths, "voice", "heading")}
    at = x
    for part in kept:
        placed[part] = (at, y, widths[part], h) if part in KNOBS else (at, top, widths[part], BUTTON_H)
        at += widths[part] + gaps.get(part, SPACE)
    # A heading is on the line the knobs have their labels on, over what it names.
    for heading, parts in (("voice", ("voices", "add", "drop")), ("heading", ("inst",))):
        under = [placed[part] for part in parts if placed[part][2]]
        if under:
            left, right = under[0][0], under[-1][0] + under[-1][2]
            placed[heading] = knob_parts((left, y, right - left, h))[0]
    return {part: within(place, rect) for part, place in placed.items()}


class Voices(Choice):
    """The voices of a sound as a row of joined toggles: each has its number, from 1, in the colour of its bars.

    `colour` is that of the first voice, and the others follow as `wave.voice_colour` has them.
    """

    def __init__(self, rect: Rect, changed: Callable[[int], object] | None = None, help: str = "") -> None:
        super().__init__(rect, ("1",), 0, changed, help)
        self.colour: Colour = theme.SOUND

    def show(self, count: int, picked: int) -> None:
        """Have a toggle for each of so many voices, with one of them on. Nothing is told of it."""
        self.options, self.picked = tuple(str(number + 1) for number in range(count)), picked

    def draw(self, canvas: Canvas) -> None:
        """Draw the voices as toggles that share the line between them. The one that is on is drawn last, whole."""
        cells = self.cells()
        for index in sorted(range(len(cells)), key=lambda index: index == self.picked):
            x, y, w, h = cells[index]
            ink = voice_colour(self.colour, index)
            lit = self.hot and index == self._over
            # All but the last reach one pixel into the next, so that the two outlines are one line.
            draw_button(canvas, (x, y, w + (index < len(cells) - 1), h), self.options[index], "", ink, lit,
                        lit and self.held, ink if index == self.picked else None)


class SoundBar:
    """The bar over the view of a sound panel: the settings of one sound of the picked entry of the project.

    It is no panel of the frame. A sound panel owns it, adds its widgets through itself, gives it a rectangle
    with `place`, a sound with `show` and a frame with `update`, and says with `seen` whether it is on the
    window. From the left it has, for an entry of several sounds, the words for the one it is of, then `voices`
    with `add` and `drop`, `inst`, `level`, `echo`, and while the echo is on its knobs `after`, `loud`, `again`
    and `bright`. `voice` is the picked voice: the one whose instrument `inst` shows, and that `drop` removes.
    A held sound has two knobs more after its level, `join` and `let_go`: its other two lengths are dragged on
    its clock. Let go of, such a knob is handed to `lengthed` as the key of the length, its ticks, and a few
    words for the list of what can be undone.

    The bar saves nothing. What a widget makes of the sound is handed to `edited` as the new definition and a
    few words for the list of what can be undone, and a level to `levelled` as the `db` of the entry, None for
    none, with such words. The owner saves, and shows the bar the sound again: an owner that answers false has
    not saved. `voiced` is called with the picked voice when one is picked here, and after a voice was added or
    removed. A change that the rules of `soundedit` refuse is handed to nobody, and the session says why.

    A knob shows its value as it is turned, and is handed on when the hand has let go of it. A voice that is
    added plays the instrument of the picked one, and is then the picked one. The last voice of a sound is not
    removed. The instrument is one of the project's: a voice whose instrument is written in place shows `OWN`,
    and that is replaced only when the stepper is stepped. A level put back to 0 is taken out of the file,
    and an entry of parts has no level. An echo that is switched off is taken out of the file, and its
    settings are kept here, so that switched on again it is the echo it was. Switched on for the first time it
    is `USUAL_ECHO`. The knob of its seconds is not for an echo of several times.
    """

    def __init__(self, panel: Panel, edited: Callable[[dict, str], object],
                 levelled: Callable[[object, str], object], voiced: Callable[[int], object] | None = None,
                 clock: Callable[[], float] = time.perf_counter,
                 lengthed: Callable[[str, float, str], object] | None = None) -> None:
        """`panel` is the panel the widgets are added through. `clock` times a run of the wheel on a knob."""
        self.session = panel.session
        self.edited = edited
        self.levelled = levelled
        self.voiced = voiced
        self.lengthed = lengthed
        nowhere = (0, 0, 0, 0)

        def knob(text: str, travel: tuple[float, float, int], usual: float | None,
                 settled: Callable[[float], object], help: str, show: Callable[[float], str] = plain) -> Knob:
            """A knob of the bar: a right click puts it back to `usual` where it has one, and let go of, it is
            `settled`."""
            low, high, steps = travel
            return panel.add(Knob(nowhere, low if usual is None else usual, low, high, default=usual, steps=steps,
                                  text=text, show=show, help=help, clock=clock, settled=settled))

        def echo(text: str, travel: tuple[float, float, int], key: str, words: str, help: str,
                 show: Callable[[float], str] = plain) -> Knob:
            """A knob of the echo: let go of, its value is the echo's setting `key`."""
            usual = ABSENT.get(key, USUAL_ECHO[key])
            return knob(text, travel, usual, lambda value: self._put(key, value, words, show), help, show)

        self.word = panel.add(Label(nowhere, ""))
        self.heading_voice = panel.add(Label(nowhere, VOICE, align="centre"))
        self.voices = panel.add(Voices(nowhere, changed=self._pick,
                                       help="the voice a new note goes into, and whose instrument is shown"))
        self.add = panel.add(Button(nowhere, ADD, action=self._add,
                                    help="add a voice that plays the instrument of the picked one"))
        self.drop = panel.add(Button(nowhere, DROP, action=self._drop,
                                     help="remove the picked voice and its notes"))
        self.heading = panel.add(Label(nowhere, INSTRUMENT, align="centre"))
        self.inst = panel.add(Stepper(nowhere, ("",), changed=self._step,
                                      help="the instrument the picked voice plays, one of the project's"))
        self.level = knob(LEVEL, LEVELS, 0, self._level,
                          "how loud the sound is, in decibels from the level every sound is brought to: a right "
                          "click is 0 again", db_text)
        self.join = knob(JOIN, JOINS, None, lambda value: self._length("join", value, "loop joined over"),
                         JOIN_HELP, ticks_text)
        self.let_go = knob(LET_GO, LET_GOS, None, lambda value: self._length("let_go", value, "let go in"),
                           LET_GO_HELP, ticks_text)
        self.echo = panel.add(Toggle(nowhere, ECHO, changed=self._switch,
                                     help="the whole sound comes back after itself, quieter each time"))
        self.after = echo(AFTER, AFTERS, "time", "after", AFTER_HELP, seconds_text)
        self.loud = echo(LOUD, LOUDS, "mix", "loud", "how loud the sound is the first time it comes back, as a "
                                                     "share of itself")
        self.again = echo(AGAIN, AGAINS, "feedback", "again", "how much of each return comes back once more: at 0 "
                                                              "the sound comes back once")
        self.bright = echo(BRIGHT, BRIGHTS, "lp", "bright", "how bright the returns stay: each goes through a "
                                                            "low-pass at this many hertz, so lower is darker",
                           hertz_text)
        #: The name of the entry that is shown and which of its sounds, as `soundedit.sound_in` counts them.
        self.name: str | None = None
        self.index = 0
        #: The picked voice of the sound, counted from 0.
        self.voice = 0
        self._colour: Colour = theme.SOUND
        #: What the project held for the entry when the widgets were set last, and its sound that is shown.
        self._entry: dict | None = None
        self._defn: dict | None = None
        #: The settings of each echo that was switched off here, by its entry and its sound.
        self._echoes: dict[tuple[str, int], dict] = {}
        #: The widgets by the part of the bar each stands in, and those of them that are knobs.
        self._widgets = {"word": self.word, "voice": self.heading_voice, "voices": self.voices, "add": self.add,
                         "drop": self.drop, "heading": self.heading, "inst": self.inst, "level": self.level,
                         "join": self.join, "let_go": self.let_go, "echo": self.echo, "after": self.after,
                         "loud": self.loud, "again": self.again, "bright": self.bright}
        self._knobs = tuple(self._widgets[part] for part in KNOBS)
        self._rect: Rect = nowhere
        self._parts: dict[str, Rect] = {}
        self._on = False
        self._lay()
        self._see()

    @property
    def busy(self) -> bool:
        """Whether a knob is under the hand: dragged, or in a run of the wheel that has not ended."""
        return any(knob.busy for knob in self._knobs)

    def place(self, rect: Rect) -> None:
        """Take the bar's rectangle, and lay the widgets out in it as `arrange` has them."""
        self._rect = rect
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
            # place, under the word for one.
            named = inst in names if isinstance(inst, str) else False
            self.inst.options = names if named else (inst if isinstance(inst, str) else OWN, *names)
            self.inst.picked = names.index(inst) if named else 0
            level = entry.get("db", 0)
            self.level.value = level[self.index] if isinstance(level, list) else level
            for key in LENGTHS if "held" in entry else ():
                self._widgets[key].value = entry["held"][key]
            echo = defn.get("echo")
            self.echo.is_on = echo is not None
            # An echo of several times has no one number for the knob of its seconds.
            one = echo is None or isinstance(echo["time"], (int, float))
            self.after.enabled, self.after.help = one, AFTER_HELP if one else SEVERAL_HELP
            if echo is not None:
                self.after.value = echo["time"] if one else self.after.value
                for knob, key in ((self.loud, "mix"), (self.again, "feedback"), (self.bright, "lp")):
                    knob.value = echo.get(key, ABSENT[key])
            self.word.text = called(soundedit.kind_of(entry), self.index).upper()
            self.voices.colour = self._colour
            for knob in self._knobs:
                knob.colour = self._colour
        self._lay()
        self._see()

    def _lay(self) -> None:
        """Give each widget its rectangle: its part of the bar, laid out for the voices of the sound that is shown."""
        voices = len(self._defn["voices"]) if self._defn else 1
        self._parts = arrange(self._rect, voices, bool(self.word.text), bool(self._defn) and "held" in self._entry)
        # A knob of a held sound has no part in the bar of any other, and no width.
        for part in LENGTHS:
            self._parts.setdefault(part, self._parts["word"][:2] + (0, 0))
        for part, widget in self._widgets.items():
            widget.rect = self._parts[part]

    def _see(self) -> None:
        """Have each widget seen or not: by whether the bar is on the window, what there is to show, and what fits.

        The knobs of the echo are seen while it is on, and the level for an entry that has one: any but parts.
        """
        on = self._on and self._defn is not None
        for part, widget in self._widgets.items():
            widget.visible = on and self._parts[part][2] > 0
        for knob in (self.after, self.loud, self.again, self.bright):
            knob.visible = knob.visible and self.echo.is_on
        self.level.visible = self.level.visible and soundedit.kind_of(self._entry) != "parts"

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

    def _drop(self) -> None:
        """Remove the picked voice. Saved, the voice that has its number is the picked one, or else the last."""
        voice = self.voice
        if self._make(lambda defn: soundedit.voice_dropped(defn, voice), f"voice {voice + 1} removed"):
            self._pick(self.voice)

    def _step(self, index: int) -> None:
        """Have the picked voice play the instrument the stepper was put on."""
        name, voice = self.inst.options[index], self.voice
        self._make(lambda defn: soundedit.voice_inst(defn, voice, name), f"voice {voice + 1} plays {name}")

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
            self.lengthed(key, tidy(value), f"{words} {ticks_text(tidy(value)).lower()}")
        self._read()

    def _level(self, value: float) -> None:
        """Hand on the level the knob was left at: the entry's own, or that of this variant where each has one."""
        value, level = tidy(value), self._entry.get("db")
        words = f"level {db_text(value)}"
        if isinstance(level, list):
            words = f"{called(soundedit.kind_of(self._entry), self.index)} {words}"
            value = [*level[:self.index], value, *level[self.index + 1:]]
        self.levelled(value or None, words)
        self._read()
