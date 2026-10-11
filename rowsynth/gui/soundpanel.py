"""The middle of the window for a sound or an instrument: what is picked as a heading, and its view under that."""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from ..bank import entry_kind
from ..errors import RowSynthError
from . import soundview, theme
from .font import Font
from .layout import Rect
from .panels import Panel, within
from .session import HOLD
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
#: From the outline of the lower box to the view in it.
PAD = 4
#: The colour of each kind of thing the panel shows: of its name, and of its bars and its wave.
COLOURS = {"sound": theme.SOUND, "instrument": theme.INSTRUMENT}
#: What the view says of itself while the mouse is on no bar and not on the wave.
HELP = "ctrl and the wheel zoom the time, and the wheel alone moves along it"
#: What a sound of several holds, by the kind of its entry: the word for one of them, and for more.
COUNTED = {"defs": ("variant", "variants"), "parts": ("part", "parts")}
#: The word on the button that a held sound is played with, and what the button says of itself.
HOLD_WORD = "HOLD"
HOLD_HELP = "press and keep pressed to hear the sound held, and let go to hear its end"

# What stands for the render that was shown last when nothing was shown yet: it is no render, and not None.
_UNSEEN: dict = {}


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
    is the box under it, down to the foot, and "view" what is inside that box, clear of its outline. Every one
    is cut to the panel.
    """
    x, y, w, h = rect
    wide, top = w - 2 * MARGIN, y + 2 * MARGIN + HEAD_H
    button = Font.width(HOLD_WORD) + 2 * theme.PAD
    box = (x + MARGIN, top, wide, y + h - MARGIN - top)
    placed = {"head": (x + MARGIN, y + MARGIN, wide, HEAD_H),
              "hold": (x + w - MARGIN - button, y + MARGIN, button, HEAD_H), "box": box,
              "view": (box[0] + PAD, box[1] + PAD, box[2] - 2 * PAD, box[3] - 2 * PAD)}
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
    Until the render is ready the wave is empty. With a song picked, or nothing, the panel shows nothing.

    A variant that is clicked in the view is played, and is the one the view marks. A held sound has a button,
    `hold`, in the heading: the sound is held for as long as the button is kept pressed.

    `heading` is the kind in capitals and the name of what is shown, or None for nothing, and `words` what the
    heading says after them of a sound of several. `kind` is what the entry of a sound is, as `bank.entry_kind`
    names it. `refusal` is what the engine says of a sound it cannot read, and "" for one it can: the words
    stand in place of the view.
    """

    def __init__(self, session: Session, ui: Ui) -> None:
        super().__init__(session, ui)
        self.view = self.add(SoundView((0, 0, 0, 0), HELP))
        self.view.chosen = self._choose
        self.hold = self.add(Hold((0, 0, 0, 0), HOLD_WORD, self._press, self._let_go, HOLD_HELP))
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
        # The render the view was last given, None for the empty wave of one that was not ready.
        self._made: dict | None = _UNSEEN
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
        """Take the panel's rectangle, and give the view and the button what `arrange` leaves them."""
        super().place(rect)
        self._places = arrange(rect)
        self.view.rect, self.hold.rect = self._places["view"], self._places["hold"]

    def update(self, passed: float = 0.0) -> None:
        """Show what is picked as it is now: its shapes made again only for another pick or another project.

        The view is given the render when that has come, and where the picked thing is while it plays.
        """
        model = self.session.model
        picked, project = model.picked, held(model)
        if self._seen is None or picked != self._seen[0] or project is not self._seen[1]:
            if self._seen is None or picked != self._seen[0]:
                self.view.fit()
                self._variant, self._holding, self._ending = None, False, None
            self._seen = (picked, project)
            self._read(picked, project, model.rate)
        self._turn()
        if not self.showing:
            self.view.playhead = None
            return
        made = self._render()
        if made is not self._made:
            self._made = made
            rows, rate = ([], model.rate) if made is None else (made["buffers"], made["rate"])
            self.view.show(self._shapes, rows, rate, self.colour, self.kind)
        self.view.playhead = self._sounding(passed)
        self.view.played = self._variant if self.kind == "defs" else None

    def _turn(self) -> None:
        """Have the view on the window only with something to show, and the button only with a held sound."""
        self.view.visible = self.shown and self.showing
        self.hold.visible = self.view.visible and self.kind == "held"

    def _read(self, picked: tuple[str, str] | None, project: Project, rate: int) -> None:
        """Make the heading of a picked thing and, for a sound, its shapes: or keep what the engine says against it."""
        self.heading, self.words, self.kind, self.refusal = None, "", KINDS[0], ""
        self._shapes, self._made = [], _UNSEEN
        if picked is None or picked[0] not in COLOURS:
            return
        kind, name = picked
        self.heading, self.colour = (kind.upper(), name), COLOURS[kind]
        if kind == "sound":
            try:
                self._shapes = soundview.shapes(project.sounds[name], rate, project.instruments)
            except RowSynthError as error:
                self.refusal = str(error)
                return
            self.kind = entry_kind(project.sounds[name])
            self.words = words(self.kind, len(self._shapes))

    def _render(self) -> dict | None:
        """The render of the picked thing, or None while it is not ready and for one the engine refused."""
        model = self.session.model
        if not model.render_ready():
            return None
        try:
            return model.render()
        except RowSynthError:
            return None

    def _choose(self, index: int) -> None:
        """Play one variant of the picked sound: it is the one that was played last."""
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
