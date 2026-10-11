"""The middle of the window for a sound or an instrument: a heading, and under it its settings and its view."""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from ..bank import entry_kind, render_entry
from ..checks.sounds import _BRIGHT, _DC
from ..errors import RowSynthError
from ..note import DEFAULT_RATE
from . import soundbar, soundedit, soundview, theme
from .font import Font
from .layout import Rect
from .panels import Panel, within
from .popups import Item, Menu, ask
from .session import HOLD
from .soundbar import SoundBar
from .ui import Button, Event, Pos, Ui, draw_button
from .wave import KINDS, LIMIT, LIMIT_STEP, PART, RULER, SoundView, called, on_tick

if TYPE_CHECKING:
    import numpy as np

    from ..project import Project
    from .canvas import Canvas
    from .model import Model
    from .session import Session

#: From the sides of the panel to its two boxes, and between the two.
MARGIN = 4
#: How high the box of the heading is, from its sides to its words, and between two things it says.
HEAD_H = 17
HEAD_PAD = 5
HEAD_SPACE = 8
#: From the outline of the lower box to what is in it, and from the bar of the settings to the view under it.
PAD = 4
BAR_GAP = 2
#: The colour of each kind of thing the panel shows: of its name, and of its bars and its wave.
COLOURS = {"sound": theme.SOUND, "instrument": theme.INSTRUMENT}
#: What the view says of itself while the mouse is on no bar and not on the wave.
HELP = "ctrl and the wheel zoom the time, and the wheel alone moves along it"
#: What a sound of several holds, by the kind of its entry: the word for one of them, and for more.
COUNTED = {"defs": ("variant", "variants"), "parts": ("part", "parts")}
#: The word on the button that a held sound is played with, and what the button says of itself.
HOLD_WORD = "HOLD"
HOLD_HELP = "press and keep pressed to hear the sound held, and let go to hear its end"
#: What is said, after its name, of a sound whose edit ended after it was changed outside.
MEANWHILE = "was changed outside while it was edited here: this edit stands, and undo brings the other back"

#: What each kind of entry is called on the button of the heading and in its menu, what a sound is said to
#: have been made, and what the button says of itself.
KIND_WORDS = {"def": "ONE", "defs": "VARIANTS", "parts": "PARTS", "held": "HELD"}
MADE = {"def": "made one sound", "defs": "made variants", "parts": "made parts", "held": "made a held sound"}
KIND_HELP = "what kind of sound this is: press to make it one sound, variants, parts on a timeline or a held sound"
#: The word on the button of the limits, and what it says of itself.
LIMITS_WORD = "LIMITS"
LIMITS_HELP = "what the checks hold this sound to: how long it may last, how far off zero and how bright it may be"
#: The limits an entry can be held to, each as the key of the file, its plain words, the unit its number is in,
#: what the checks go by where the entry has none, and what the box says in which its number is typed.
LIMITS = (
    ("max_seconds", "Longest time", " s", None, "the longest the sound may last, in seconds: empty is no limit"),
    ("max_dc", "Furthest off zero", "", _DC,
     "how far the mean of the wave may be off zero before the checks say so: empty is the usual"),
    ("max_bright", "Brightest", "", _BRIGHT,
     "the share of the energy that may be above 4 kHz before the checks say so: empty is the usual"))
#: The least room the heading keeps for its words: one too narrow for that has neither of the two buttons.
HEAD_WORDS = 160
#: The characters a number is typed with, and how many of them a box takes.
NUMBER = "0123456789.-"
MOST_NUMBER = 10
#: What a border of a held sound is said to have been dragged to, before the ticks and after them.
BORDERS = {"start": ("loop starts at tick", ""), "loop": ("loop", " ticks long")}

# What stands for a render that has not come: it is no render, and not the None of one the engine refused.
_WAITING: dict = {}


def held(model: Model) -> Project:
    """The project a model has open. It is the same one until the file is loaded again."""
    return model.project


def words(kind: str, count: int) -> str:
    """What the heading says of a sound after its name: what its entry is, and how many sounds that holds.

    `kind` is as `bank.entry_kind` names it, and `count` how many shapes the entry has: "3 variants", "2 parts",
    "held". Of one sound it says nothing.
    """
    if kind in COUNTED:
        return f"{count} {COUNTED[kind][count != 1]}"
    return "held" if kind == "held" else ""


def lost(entry: dict, to: str) -> str:
    """What an entry lets go of when it is made an entry of another kind, in a few words: "" for nothing.

    One sound keeps the first of several, and a held sound the first two. A held sound that is made anything
    else has no loop any more, and parts that are made anything else all start together.
    """
    kind = entry_kind(entry)
    if kind == to:
        return ""
    count = {"def": 1, "held": 2}.get(kind) or len(entry[kind])
    keeps = {"def": 1, "held": 2}.get(to, count)
    if keeps < count:
        if kind == "held":
            return "keeps the note only"
        one, more = COUNTED[kind]
        return f"keeps {one} 1 only" if keeps == 1 else f"keeps {more} 1 and 2 only"
    if kind == "held":
        return "drops the loop"
    if kind == "parts" and any(at for at, _ in entry["parts"]):
        return "drops where the parts start"
    return ""


def converted(entry: dict, kind: str, rate: int = DEFAULT_RATE) -> dict:
    """An entry as an entry of another kind, as the rules of `soundedit` make one of it."""
    if kind == "held":
        return soundedit.as_held(entry, rate)
    return {"def": soundedit.as_one, "defs": soundedit.as_variants, "parts": soundedit.as_parts}[kind](entry)


def number(typed: str) -> int | float:
    """The number that was typed in a box, a whole one with nothing after a point. What is none is refused."""
    try:
        value = float(typed)
    except ValueError:
        raise RowSynthError(f"{typed!r} is no number") from None
    return int(value) if value == int(value) else value


def zone(at: float, lengths: Sequence[float], rounds: int = 1) -> tuple[int, float]:
    """Where a held sound is, at a place in the row it is played as: which zone, counted from 0, and how far into it.

    `lengths` are those of its start, its loop and its end, in the measure the place is in: samples, or seconds.
    The row is the start, the loop `rounds` times over, and then the end, as `session.joined` lays it. A border
    is the first place of the zone that starts there. In the loop the place is in the round that is going.
    """
    start, loop, _ = lengths
    if at < start:
        return 0, at
    into = at - start
    if into < rounds * loop:
        return 1, into % loop
    return 2, into - rounds * loop


def arrange(rect: Rect) -> dict[str, Rect]:
    """Where the things of the panel go in its rectangle, each by what it is called.

    "head" is the box of the heading, along the top, and "hold" the button at the right end of that box. Before
    it stand "limits" and before that "kind", the two buttons of a sound that can be changed: a heading that
    would keep less than `HEAD_WORDS` for its words has neither, and they have no width. "box" is the box under
    it, down to the foot. Inside that box, clear of its outline, "bar" runs along the top, which is where the
    settings of a sound go as `soundbar.arrange` lays them out, and "view" is what is left under it. Every one
    is cut to the panel.
    """
    x, y, w, h = rect
    wide, top = w - 2 * MARGIN, y + 2 * MARGIN + HEAD_H
    button = Font.width(HOLD_WORD) + 2 * theme.PAD
    hold = x + w - MARGIN - button
    limits = Font.width(LIMITS_WORD) + 2 * theme.PAD
    kind = max(Font.width(word) for word in KIND_WORDS.values()) + 2 * theme.PAD
    at = hold - HEAD_SPACE - limits
    if at - theme.BUTTON_GAP - kind - HEAD_SPACE - x - MARGIN < HEAD_WORDS:
        at, limits, kind = hold, 0, 0
    box = (x + MARGIN, top, wide, y + h - MARGIN - top)
    left, upper, inner = box[0] + PAD, box[1] + PAD, box[2] - 2 * PAD
    under = upper + soundbar.HEIGHT + BAR_GAP
    placed = {"head": (x + MARGIN, y + MARGIN, wide, HEAD_H),
              "hold": (hold, y + MARGIN, button, HEAD_H), "limits": (at, y + MARGIN, limits, HEAD_H),
              "kind": (at - (theme.BUTTON_GAP + kind if kind else 0), y + MARGIN, kind, HEAD_H), "box": box,
              "bar": (left, upper, inner, soundbar.HEIGHT),
              "view": (left, under, inner, box[1] + box[3] - PAD - under)}
    return {thing: within(place, rect) for thing, place in placed.items()}


class Hold(Button):
    """A button that is on for as long as it is kept pressed.

    `pressed` is called as the left button goes down on it, and `released` as that button comes up again,
    wherever the mouse is by then.
    """

    def __init__(self, rect: Rect, text: str, pressed: Callable[[], object], released: Callable[[], object],
                 help: str = "") -> None:
        super().__init__(rect, text, help=help)
        self.pressed = pressed
        self.released = released

    def on(self, event: Event, ui: Ui) -> bool:
        """Use a press of the left button, and tell of it and of its release."""
        if event.button != 1:
            return False
        if event.kind == "down":
            self.pressed()
            return True
        if event.kind == "up" and ui.held is self:
            self.released()
        return False

    def draw(self, canvas: Canvas) -> None:
        """Draw it raised, lit when hot, and sunk with its word lit for as long as it is kept pressed."""
        draw_button(canvas, self.rect, self.text, "", theme.ACTIVE if self.held else theme.TEXT, self.hot, self.held)


class SoundPanel(Panel):
    """The middle of the window while a sound or an instrument is picked: its kind and its name, and its view.

    A sound is its notes as bars over the waves of its render, and an instrument the wave of its test note alone.
    Until the first render is ready the wave is empty. With a song picked, or nothing, the panel shows nothing.

    A variant that is clicked in the view is played, and is the one the view marks. A held sound has a button,
    `hold`, in the heading: the sound is held for as long as the button is kept pressed.

    The notes of a sound are changed in `view`, and its voices, its echo and its level in `bar`, which has the
    settings of the sound of the entry that the hand is in. What either hands the panel is saved at once: the
    whole entry is put in the file, and that is one change, and one step that can be taken back. The sound is
    then read again as after any change, so the view and the bar show what the file holds: the notes at once,
    and the wave of before, dim, until the render of the saved sound has come. A change that is refused leaves
    them as they were, and the session says why. After a save the sound is played once, the variant that was
    changed where it has variants, unless the session's `audition` is off. While a gesture is under way in the
    view, or a knob of the bar is under the hand, nothing is read: what changed meanwhile, here or outside, is
    shown when the hand has let go. A note that is placed, or moved to another key, is heard where its voice
    names its instrument.

    What kind of entry a sound is, and what belongs to the kind, is changed here as well, each change one save
    of the whole entry and one step. `kinds`, in the heading, shows the kind and opens a menu of the four: an
    item says what the sound would let go of. The view tells of the rest. The sign after the labels adds a
    variant or a part, a copy of the one the hand is in, and a click in the ruler with ctrl held adds a part
    there. The right button on a label opens a menu: a variant is removed or moved, and a part is removed,
    copied, or given its level in a box. The label of a part is dragged to where the part is to start, the
    borders of a held sound to its start and its loop, and the line of the longest time to that. The right
    button in the ruler sets a longest time or takes it away. `limits`, in the heading, opens a menu of all
    three limits the checks go by, each typed as a number in a box: the bar has no room for them beside the
    knobs of a held sound with an echo. The other two lengths of a held sound are knobs of the bar.

    `heading` is the kind in capitals and the name of what is shown, or None for nothing, and `words` what the
    heading says after them of a sound of several. `kind` is what the entry of a sound is, as `bank.entry_kind`
    names it. `refusal` is what the engine says of a sound it cannot read, and "" for one it can: the words
    stand in place of the view.
    """

    def __init__(self, session: Session, ui: Ui) -> None:
        super().__init__(session, ui)
        self.view = self.add(SoundView((0, 0, 0, 0), HELP))
        self.view.chosen, self.view.edited = self._choose, self._save
        self.view.sounded, self.view.refused = self._sound, self._refuse
        self.view.added, self.view.asked = self._add, self._ask
        self.view.handled, self.view.alone = self._handle, self._alone
        self.hold = self.add(Hold((0, 0, 0, 0), HOLD_WORD, self._press, self._let_go, HOLD_HELP))
        self.kinds = self.add(Button((0, 0, 0, 0), KIND_WORDS[KINDS[0]], action=self._ask_kind, help=KIND_HELP))
        self.limits = self.add(Button((0, 0, 0, 0), LIMITS_WORD, action=self._ask_limits, help=LIMITS_HELP))
        self.bar = SoundBar(self, lambda defn, label: self._save(self.view.sound, defn, label), self._level,
                            self._voice, lengthed=self._length)
        # Nothing has been looked at yet, so there is nothing for the view to show.
        self.view.visible = self.hold.visible = self.kinds.visible = self.limits.visible = False
        self.heading: tuple[str, str] | None = None
        self.words = ""
        self.kind = KINDS[0]
        self.refusal = ""
        self.colour = theme.SOUND
        # What the shapes were last made of: the picked thing and the project. None before the first look.
        self._seen: tuple[tuple[str, str] | None, Project] | None = None
        self._shapes: list = []
        # The entry of the picked sound as the project held it then, or None where there is none to change.
        self._entry: dict | None = None
        # The render that was asked for last: one, `_WAITING` for one that had not come, None for one refused.
        self._made: dict | None = _WAITING
        # The render whose waves the view has, which can be that of the sound as it was. None for no waves.
        self._wave: dict | None = None
        self._places = arrange(self.rect)
        # The variant that was played last, or None where the panel does not know which that is.
        self._variant: int | None = None
        # Whether what plays is the held sound as the button has it played: with its loop going round.
        self._holding = False
        # The seconds since the held sound was let go, which its end is followed by. None before it is let go,
        # and once something plays again.
        self._ending: float | None = None
        # Whether the panel has asked for a play since the frame before, and what the session said in that
        # frame: whether the picked thing played, and how far into it.
        self._asked = False
        self._was = (False, 0.0)

    @property
    def showing(self) -> bool:
        """Whether there is something for the view to show: a sound it can read is picked, or an instrument."""
        return self.heading is not None and not self.refusal

    def show(self, on: bool) -> None:
        """Bring the panel onto the window, or take it off. The view comes back only with something to show."""
        super().show(on)
        self._turn()

    def place(self, rect: Rect) -> None:
        """Take the panel's rectangle, and give the view, the button and the bar what `arrange` leaves them."""
        super().place(rect)
        self._places = arrange(rect)
        self.view.rect, self.hold.rect = self._places["view"], self._places["hold"]
        self.kinds.rect, self.limits.rect = self._places["kind"], self._places["limits"]
        self.bar.place(self._places["bar"])

    def update(self, passed: float = 0.0) -> None:
        """Show what is picked as it is now: its shapes made again only for another pick or another project.

        The bar takes its frame first, which ends a run of the wheel whose time is up. While a gesture is under
        way in the view, or a knob of the bar is under the hand, nothing is read. Else the view is given the
        render when that has come, and the bar the sound that the hand is in. The view is told where the picked
        thing is while it plays.
        """
        self.bar.update()
        if not (self.view.busy or self.bar.busy):
            self._look()
            made = self._render() if self.showing else self._made
            if made is not self._made:
                self._take(made)
            elif self._entry is not None and self.view.sound != self.bar.index:
                self._settings()
        self._turn()
        if not self.showing:
            self.view.playhead = None
            return
        self.view.playhead = self._sounding(passed)
        self.view.played = self._variant if self.kind == "defs" else None

    def _turn(self) -> None:
        """Have the view on the window only with something to show, the button only with a held sound, and the
        bar and the two buttons of the kind and the limits only with a sound that can be changed."""
        self.view.visible = self.shown and self.showing
        self.hold.visible = self.view.visible and self.kind == "held"
        changed = self.view.visible and self._entry is not None
        self.bar.seen(changed)
        self.kinds.visible = self.limits.visible = changed and self._places["kind"][2] > 0
        self.kinds.text = KIND_WORDS[self.kind]

    def _look(self) -> None:
        """Read the picked thing where it, or the project, is another than the one that was read last."""
        model = self.session.model
        picked, project = model.picked, held(model)
        if self._seen is None or picked != self._seen[0] or project is not self._seen[1]:
            if self._seen is None or picked != self._seen[0]:
                # Another thing is shown whole, with the hand in its first sound and no wave of before.
                self.view.fit()
                self.view.sound = 0
                self._variant, self._holding, self._ending, self._wave = None, False, None, None
            self._seen = (picked, project)
            self._read(picked, project, model.rate)

    def _read(self, picked: tuple[str, str] | None, project: Project, rate: int) -> None:
        """Make the heading of a picked thing and, for a sound, its shapes: or keep what the engine says against it.

        The view is shown the shapes at once, over the waves it had, and the bar the sound the hand is in.
        """
        self.heading, self.words, self.kind, self.refusal = None, "", KINDS[0], ""
        self._shapes, self._entry = [], None
        if picked is not None and picked[0] in COLOURS:
            kind, name = picked
            self.heading, self.colour = (kind.upper(), name), COLOURS[kind]
            if kind == "sound":
                try:
                    self._shapes = soundview.shapes(project.sounds[name], rate, project.instruments)
                except RowSynthError as error:
                    self.refusal = str(error)
                else:
                    self._entry = project.sounds[name]
                    self.kind = entry_kind(self._entry)
                    self.words = words(self.kind, len(self._shapes))
        if self.showing:
            self._take(_WAITING)
        else:
            self._made = _WAITING
            self._settings()

    def _take(self, made: dict | None) -> None:
        """Show the view the shapes over the waves of a render, and the bar the sound that the hand is in.

        A render that was refused is an empty wave. While one has not come, the waves stay those of the render
        the view had, where they still fit the shapes, and are stale: for a thing that was only just picked
        there are none.
        """
        self._made = made
        waits = made is _WAITING
        shown = self._wave if waits else made
        rows = [] if shown is None else shown["buffers"]
        if waits and len(rows) not in (1, len(self._shapes)):
            rows = []
        rate = shown["rate"] if rows else self.session.model.rate
        self.view.show(self._shapes, rows, rate, self.colour, self.kind, self._entry, waits and bool(rows),
                       held(self.session.model).instruments)
        self._wave = shown if rows else None
        self._settings()

    def _settings(self) -> None:
        """Show the bar the sound of the entry that the hand is in, and give the view the voice that is picked there.

        With no sound to change the bar is shown none.
        """
        self.bar.show(self.heading[1] if self._entry is not None else None, self.view.sound, self.colour)
        self.view.voice = self.bar.voice

    def _render(self) -> dict | None:
        """The render of the picked thing, `_WAITING` while it is not ready, and None for one the engine refused."""
        model = self.session.model
        if not model.render_ready():
            return _WAITING
        try:
            return model.render()
        except RowSynthError:
            return None

    def _save(self, index: int, defn: dict, label: str) -> bool:
        """Save a definition in the place of one sound of the picked entry, and say whether it was saved.

        `index` is which sound of the entry, as `soundedit.sound_in` counts them, and `label` what was done to
        it. The definition is put in the entry as the project holds it now. What `soundedit` refuses of that is
        not saved, as a held sound whose note would be too short for its start and its loop, and the session
        is told why.

        Where that sound is another by now than the one the view was shown, it was changed outside while a
        gesture was under way here. What the hand made is saved all the same, and the session is told that it
        was, and that taking the step back gives the other.
        """
        model = self.session.model
        project, name = held(model), self.heading[1]
        try:
            entry = soundedit.with_sound(project.sounds[name], index, defn, project.instruments, model.rate)
            meanwhile = soundedit.sound_in(project.sounds[name], index) != soundedit.sound_in(self._entry, index)
        except RowSynthError as error:
            return self._refuse(str(error))
        saved = self._put(entry, label, index)
        if saved and meanwhile:
            self.session.say(f"{name} {MEANWHILE}")
        return saved

    def _level(self, level: object, label: str) -> bool:
        """Save a level as the `db` of the picked entry, or with None take it out, and say whether it was saved."""
        try:
            entry = soundedit.with_setting(held(self.session.model).sounds[self.heading[1]], "db", level)
        except RowSynthError as error:
            return self._refuse(str(error))
        return self._put(entry, label, self.view.sound)

    def _change(self, make: Callable[[dict], dict], label: str, hand: int | None = None) -> bool:
        """Save the picked entry as a rule makes it of what the project holds now, and say whether it was saved.

        `make` is given the entry and gives the new one, and `label` says what was done. What the rule refuses
        is not saved, and the session is told why. Saved, the hand is in the sound `hand` of the entry, where
        one is given. Where the entry is another by now than the one the view was shown, it was changed outside
        while a handle was dragged here: the session is told, as for a note.
        """
        name = self.heading[1]
        current = held(self.session.model).sounds.get(name)
        if current is None:
            return self._refuse("the project has no such sound any more")
        try:
            entry = make(current)
        except RowSynthError as error:
            return self._refuse(str(error))
        meanwhile = current != self._entry
        saved = self._put(entry, label, self.view.sound if hand is None else hand)
        if saved and hand is not None:
            self.view.sound = hand
            self._settings()
        if saved and meanwhile:
            self.session.say(f"{name} {MEANWHILE}")
        return saved

    def _add(self, at: float | None) -> None:
        """Save the entry with a variant or a part more, a copy of the one the hand is in, which the hand is then in.

        A part starts at a second of the clock, and with None on the tick nearest to where the sound ends.
        """
        hand, count = self.view.sound, len(self._shapes)
        if self.kind == "defs":
            self._change(lambda entry: soundedit.variant_added(entry, hand), f"variant {count + 1} added", count)
            return
        if at is None:
            at = on_tick(max(place + shape.seconds for _, place, shape in self._shapes))
        self._change(lambda entry: soundedit.part_added(entry, at, hand), f"part {count + 1} added at {at:g} s",
                     count)

    def _handle(self, kind: str, which: int, value: float) -> None:
        """Save what a handle of the view was dragged to: the start of a part, a border, or the longest time."""
        model = self.session.model
        instruments, rate = held(model).instruments, model.rate
        if kind == PART:
            self._change(lambda entry: soundedit.part_at(entry, which, value),
                         f"part {which + 1} moved to {value:g} s")
        elif kind == LIMIT:
            self._limit("max_seconds", value)
        else:
            before, after = BORDERS[kind]
            self._change(lambda entry: soundedit.held_with(entry, kind, value, instruments, rate),
                         f"{before} {value:g}{after}")

    def _length(self, key: str, value: float, label: str) -> bool:
        """Save a length of the held sound that a knob of the bar was left at, kept inside what its note has room for.

        Whether it was saved.
        """
        model = self.session.model
        instruments, rate = held(model).instruments, model.rate
        return self._change(lambda entry: soundedit.held_with(entry, key, value, instruments, rate), label)

    def _limit(self, key: str, value: float | None) -> bool:
        """Save a limit of the entry, or with None take it out of the file."""
        words, unit = next((words.lower(), unit) for name, words, unit, _, _ in LIMITS if name == key)
        none = "no longest time" if key == "max_seconds" else f"{words} as usual"
        return self._change(lambda entry: soundedit.with_setting(entry, key, value),
                            none if value is None else f"{words} {value:g}{unit}")

    def _alone(self, which: int) -> np.ndarray | None:
        """The wave of one part of the picked sound by itself, at its own level. None where the engine refuses it."""
        model = self.session.model
        try:
            return render_entry(held(model).sound(self.heading[1])["parts"][which][1], model.rate)[0]
        except (RowSynthError, LookupError):
            return None

    def _menu(self, items: Sequence[Item | None], at: Pos) -> None:
        """Open a menu at a place, over everything else. Each item stands for what is done when it is chosen."""
        self.ui.open(Menu(items, at, chosen=lambda act: act()))

    def _type(self, rect: Rect, value: float | None, entered: Callable[[float | None], object], help: str) -> None:
        """Open a box over a rectangle that holds a number, or nothing for None, of which `help` says what it is.

        On enter `entered` is given the number in the box, and None for an empty one. What is no number is
        handed to nobody, and the session says so.
        """
        def done(typed: str) -> None:
            try:
                entered(number(typed) if typed else None)
            except RowSynthError as error:
                self._refuse(str(error))

        ask(self.ui, rect, "" if value is None else f"{value:g}", done, allowed=NUMBER.__contains__,
            most=MOST_NUMBER, help=f"{help}, enter sets it, escape leaves it")

    def _ask_kind(self) -> None:
        """Open the menu of the four kinds under its button. An item says what the sound would let go of."""
        entry, rate = self._entry, self.session.model.rate

        def item(kind: str) -> Item:
            drops = lost(entry, kind)
            return Item(f"{KIND_WORDS[kind]} ({drops})" if drops else KIND_WORDS[kind],
                        lambda: self._change(lambda entry: converted(entry, kind, rate), MADE[kind], 0),
                        enabled=kind != self.kind)

        x, y, _, h = self.kinds.rect
        self._menu([item(kind) for kind in KINDS], (x, y + h))

    def _ask_limits(self) -> None:
        """Open the menu of the limits under its button: each item opens a box that holds the number of one."""
        entry = self._entry

        def item(key: str, words: str, unit: str, usual: float | None, help: str) -> Item:
            shown = f"{entry[key]:g}{unit}" if key in entry else "none" if usual is None else f"{usual:g}, the usual"
            return Item(f"{words} ({shown})...", lambda: self._type(
                self.limits.rect, entry.get(key), lambda value: self._limit(key, value), help))

        x, y, _, h = self.limits.rect
        self._menu([item(*limit) for limit in LIMITS], (x, y + h))

    def _ask(self, what: str, which: int, second: float, pos: Pos) -> None:
        """Open the menu the view asks for at the mouse: that of the ruler, or of the label of a variant or a part.

        The ruler sets the longest time to its second, in hundredths, or takes it away. A variant is removed
        or moved one place, and a part is removed, copied as a part that starts with it, or given its level in a
        box over its label. What cannot be done is not enabled: the last of them is not removed, and none is
        moved past the first or the last.
        """
        entry, count, name = self._entry, len(self._shapes), called(self.kind, which)
        removed = soundedit.variant_dropped if self.kind == "defs" else soundedit.part_dropped
        remove = Item("Remove", lambda: self._change(lambda entry: removed(entry, which), f"{name} removed"),
                      enabled=count > 1)
        if what == RULER:
            at = round(max(LIMIT_STEP, second), 2)
            items = [Item("Set a longest time here", lambda: self._limit("max_seconds", at)),
                     Item("No longest time", lambda: self._limit("max_seconds", None),
                          enabled="max_seconds" in entry)]
        elif self.kind == "defs":
            def move(way: str, to: int) -> Item:
                return Item(f"Move {way}", lambda: self._change(
                    lambda entry: soundedit.variant_moved(entry, which, to), f"{name} moved {way}", to),
                    enabled=0 <= to < count)

            items = [remove, move("left", which - 1), move("right", which + 1)]
        else:
            start, level = entry["parts"][which][0], entry["parts"][which][1].get("db")

            def levelled(db: float | None) -> None:
                self._change(lambda entry: soundedit.part_level(entry, which, db),
                             f"{name} level taken out" if db is None else f"{name} level {db:g} dB")

            items = [remove,
                     Item("Copy as a new part", lambda: self._change(
                         lambda entry: soundedit.part_added(entry, start, which),
                         f"{name} copied as part {count + 1}", count)),
                     Item(f"Level ({level or 0:g} dB)...", lambda: self._type(
                         self.view.tab(which), level, levelled,
                         f"the level of {name}, in decibels from the level every sound is brought to"))]
        self._menu(items, pos)

    def _put(self, entry: dict, label: str, index: int) -> bool:
        """Put an entry in the file in the place of the picked one, as one step, and show what the file then holds.

        The answer is whether it was saved. A change the project refuses is not made, and the session's message
        is the refusal. Saved, the sound is played once where there is a player and the session's `audition`
        is on: `index` is the variant that plays, of an entry of variants.
        """
        name = self.heading[1]
        saved = self.session.change(f"sounds.{name}", entry, f"{name}: {label}")
        self._look()
        if saved and self.session.audition and self.session.player is not None:
            self._choose(index)
        return saved

    def _refuse(self, words: str) -> bool:
        """Have the session say why a change to the picked sound was not made. The answer is false: not saved."""
        self.session.say(f"{self.heading[1]}: {words}")
        return False

    def _sound(self, inst: str | dict, pitch: float) -> None:
        """Let a note of the instrument of a voice be heard, over whatever plays.

        An instrument that is written in place in its voice has no name to ask the session for, and is not heard.
        """
        if isinstance(inst, str):
            self.session.sound(inst, pitch)

    def _voice(self, voice: int) -> None:
        """Take the voice that was picked in the bar as the one a new note of the view goes into."""
        self.view.voice = voice

    def _choose(self, index: int) -> None:
        """Play the picked sound from its start: of variants the one at an index, which is then the one played last."""
        self.session.play(index)
        self._variant, self._asked = index, True

    def _press(self) -> None:
        """Play the held sound and keep it held."""
        self.session.hold()
        self._holding, self._asked = True, True

    def _let_go(self) -> None:
        """Let go of the held sound."""
        sounding = self.session.playing
        self.session.release()
        if sounding and not self.session.playing:
            # The end is played over what is let go, and the session has no position for it: it is timed here.
            self._holding, self._ending = False, 0.0
        elif not self.session.model.render_ready():
            # A hold that still waits for its render is played through once, as any play of the sound is.
            self._holding = False

    def _sounding(self, passed: float) -> float | None:
        """The second of the view's clock that sounds now, or None for none. `passed` is the seconds of the frame.

        A variant goes by the stretch of the one that was played, and a held sound by its zones. A play the panel
        did not ask for is one of a variant the session does not name, and for that there is no second.
        """
        playing = self.session.playing
        at = self.session.position() if playing else 0.0
        if playing and not self._asked and (not self._was[0] or at < self._was[1]):
            self._variant, self._holding = None, False
        self._asked, self._was = False, (playing, at)
        if not playing:
            return self._end(passed)
        self._ending = None
        if self.kind == "defs":
            return None if self._variant is None else self.view.clock(self._variant, at)
        if self.kind == "held":
            return self._zone(at)
        return at

    def _zone(self, at: float) -> float | None:
        """The second of the view's clock a held sound is at, `at` seconds into the row it is played as."""
        if not self._made:
            return None
        rate = self._made["rate"]
        lengths = [len(row) for row in self._made["buffers"]]
        # As `Session.hold` has it played: with as many rounds of its loop as fill `HOLD` seconds.
        rounds = math.ceil(HOLD * rate / lengths[1]) if self._holding else 1
        index, into = zone(at * rate, lengths, rounds)
        return self.view.clock(index, into / rate)

    def _end(self, passed: float) -> float | None:
        """The second of the view's clock the end of a held sound is at, a frame on. None when it is through, and
        when the sound was not let go."""
        if self._ending is None:
            return None
        self._ending += passed
        return self.view.clock(2, self._ending)

    def draw(self, canvas: Canvas) -> None:
        """Draw the two boxes, the heading in the upper one, and in the lower one the refusal if there is one."""
        if self.heading is None:
            return
        places = self._places
        with canvas.clip(self.rect):
            for box in (places["head"], places["box"]):
                canvas.box(box, theme.PANEL)
                canvas.frame(box, theme.LINE)
            x, y, w, h = places["head"]
            # The words end before the first button there is.
            last = min([x + w - HEAD_PAD, *(button.rect[0] - HEAD_SPACE for button in (self.kinds, self.hold)
                                             if button.visible)])
            at = x + HEAD_PAD
            for text, colour in zip((*self.heading, self.words), (theme.DIM, self.colour, theme.DIM)):
                canvas.text_in(text, (at, y, max(0, last - at), h), colour)
                at += Font.width(text) + HEAD_SPACE
            if self.refusal:
                canvas.text_in(self.refusal, places["view"], theme.FAULT, "centre")
