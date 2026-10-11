"""The middle of the window for a sound or an instrument: a heading, and under it its settings and its view."""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from ..bank import entry_kind
from ..errors import RowSynthError
from . import soundbar, soundedit, soundview, theme
from .font import Font
from .layout import Rect
from .panels import Panel, within
from .session import HOLD
from .soundbar import SoundBar
from .ui import Button, Event, Ui, draw_button
from .wave import KINDS, SoundView

if TYPE_CHECKING:
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

    "head" is the box of the heading, along the top, and "hold" the button at the right end of that box. "box"
    is the box under it, down to the foot. Inside that box, clear of its outline, "bar" runs along the top,
    which is where the settings of a sound go as `soundbar.arrange` lays them out, and "view" is what is left
    under it. Every one is cut to the panel.
    """
    x, y, w, h = rect
    wide, top = w - 2 * MARGIN, y + 2 * MARGIN + HEAD_H
    button = Font.width(HOLD_WORD) + 2 * theme.PAD
    box = (x + MARGIN, top, wide, y + h - MARGIN - top)
    left, upper, inner = box[0] + PAD, box[1] + PAD, box[2] - 2 * PAD
    under = upper + soundbar.HEIGHT + BAR_GAP
    placed = {"head": (x + MARGIN, y + MARGIN, wide, HEAD_H),
              "hold": (x + w - MARGIN - button, y + MARGIN, button, HEAD_H), "box": box,
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
        self.hold = self.add(Hold((0, 0, 0, 0), HOLD_WORD, self._press, self._let_go, HOLD_HELP))
        self.bar = SoundBar(self, lambda defn, label: self._save(self.view.sound, defn, label), self._level,
                            self._voice)
        # Nothing has been looked at yet, so there is nothing for the view to show.
        self.view.visible = self.hold.visible = False
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
        bar only with a sound that can be changed."""
        self.view.visible = self.shown and self.showing
        self.hold.visible = self.view.visible and self.kind == "held"
        self.bar.seen(self.view.visible and self._entry is not None)

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
        self.view.show(self._shapes, rows, rate, self.colour, self.kind, self._entry, waits and bool(rows))
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
            # The words end before the button, where there is one.
            last = places["hold"][0] - HEAD_SPACE if self.hold.visible else x + w - HEAD_PAD
            at = x + HEAD_PAD
            for text, colour in zip((*self.heading, self.words), (theme.DIM, self.colour, theme.DIM)):
                canvas.text_in(text, (at, y, max(0, last - at), h), colour)
                at += Font.width(text) + HEAD_SPACE
            if self.refusal:
                canvas.text_in(self.refusal, places["view"], theme.FAULT, "centre")
