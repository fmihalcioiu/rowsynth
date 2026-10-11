"""The middle of the window for a sound or an instrument: what is picked as a heading, and its view under that."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..commands import mono
from ..errors import RowSynthError
from . import soundview, theme
from .font import Font
from .layout import Rect
from .panels import Panel, within
from .ui import Ui
from .wave import SoundView

if TYPE_CHECKING:
    from ..project import Project
    from .canvas import Canvas
    from .model import Model
    from .session import Session

#: From the sides of the panel to its two boxes, and between the two.
MARGIN = 4
#: How high the box of the heading is, from its sides to its words, and between the kind and the name.
HEAD_H = 17
HEAD_PAD = 5
HEAD_SPACE = 8
#: From the outline of the lower box to the view in it.
PAD = 4
#: The colour of each kind of thing the panel shows: of its name, and of its bars and its wave.
COLOURS = {"sound": theme.SOUND, "instrument": theme.INSTRUMENT}
#: What the view says of itself while the mouse is on no bar and not on the wave.
HELP = "ctrl and the wheel zoom the time, and the wheel alone moves along it"

# What stands for the render that was shown last when nothing was shown yet: it is no render, and not None.
_UNSEEN: dict = {}


def held(model: Model) -> Project:
    """The project a model has open. It is the same one until the file is loaded again."""
    return model.project


def arrange(rect: Rect) -> dict[str, Rect]:
    """Where the things of the panel go in its rectangle, each by what it is called.

    "head" is the box of the heading, along the top. "box" is the box under it, down to the foot, and "view" what
    is inside that box, clear of its outline. Every one is cut to the panel.
    """
    x, y, w, h = rect
    wide, top = w - 2 * MARGIN, y + 2 * MARGIN + HEAD_H
    box = (x + MARGIN, top, wide, y + h - MARGIN - top)
    placed = {"head": (x + MARGIN, y + MARGIN, wide, HEAD_H), "box": box,
              "view": (box[0] + PAD, box[1] + PAD, box[2] - 2 * PAD, box[3] - 2 * PAD)}
    return {thing: within(place, rect) for thing, place in placed.items()}


class SoundPanel(Panel):
    """The middle of the window while a sound or an instrument is picked: its kind and its name, and its view.

    A sound is its notes as bars over the wave of its render, and an instrument the wave of its test note alone.
    Until the render is ready the wave is empty. With a song picked, or nothing, the panel shows nothing.

    `heading` is the kind in capitals and the name of what is shown, or None for nothing. `refusal` is what the
    engine says of a sound it cannot read, and "" for one it can: the words stand in place of the view.
    """

    def __init__(self, session: Session, ui: Ui) -> None:
        super().__init__(session, ui)
        self.view = self.add(SoundView((0, 0, 0, 0), HELP))
        # Nothing has been looked at yet, so there is nothing for the view to show.
        self.view.visible = False
        self.heading: tuple[str, str] | None = None
        self.refusal = ""
        self.colour = theme.SOUND
        # What the shapes were last made of: the picked thing and the project. None before the first look.
        self._seen: tuple[tuple[str, str] | None, Project] | None = None
        self._shapes: list = []
        # The render the view was last given, None for the empty wave of one that was not ready.
        self._made: dict | None = _UNSEEN
        self._places = arrange(self.rect)

    @property
    def showing(self) -> bool:
        """Whether there is something for the view to show: a sound it can read is picked, or an instrument."""
        return self.heading is not None and not self.refusal

    def show(self, on: bool) -> None:
        """Bring the panel onto the window, or take it off. The view comes back only with something to show."""
        super().show(on)
        self.view.visible = on and self.showing

    def place(self, rect: Rect) -> None:
        """Take the panel's rectangle, and give the view what `arrange` leaves it."""
        super().place(rect)
        self._places = arrange(rect)
        self.view.rect = self._places["view"]

    def update(self, passed: float = 0.0) -> None:
        """Show what is picked as it is now: its shapes made again only for another pick or another project.

        The view is given the render when that has come, and the session's position while the thing plays.
        """
        model = self.session.model
        picked, project = model.picked, held(model)
        if self._seen is None or picked != self._seen[0] or project is not self._seen[1]:
            if self._seen is None or picked != self._seen[0]:
                self.view.fit()
            self._seen = (picked, project)
            self._read(picked, project, model.rate)
        self.view.visible = self.shown and self.showing
        if not self.showing:
            self.view.playhead = None
            return
        made = self._render()
        if made is not self._made:
            self._made = made
            samples = np.zeros(0) if made is None else mono(made["sides"])
            self.view.show(self._shapes, samples, model.rate if made is None else made["rate"], self.colour)
        self.view.playhead = self.session.position() if self.session.playing else None

    def _read(self, picked: tuple[str, str] | None, project: Project, rate: int) -> None:
        """Make the heading of a picked thing and, for a sound, its shapes: or keep what the engine says against it."""
        self.heading, self.refusal, self._shapes, self._made = None, "", [], _UNSEEN
        if picked is None or picked[0] not in COLOURS:
            return
        kind, name = picked
        self.heading, self.colour = (kind.upper(), name), COLOURS[kind]
        if kind == "sound":
            try:
                self._shapes = soundview.shapes(project.sounds[name], rate, project.instruments)
            except RowSynthError as error:
                self.refusal = str(error)

    def _render(self) -> dict | None:
        """The render of the picked thing, or None while it is not ready and for one the engine refused."""
        model = self.session.model
        if not model.render_ready():
            return None
        try:
            return model.render()
        except RowSynthError:
            return None

    def draw(self, canvas: Canvas) -> None:
        """Draw the two boxes, the heading in the upper one, and in the lower one the refusal if there is one."""
        if self.heading is None:
            return
        places = self._places
        with canvas.clip(self.rect):
            for box in (places["head"], places["box"]):
                canvas.box(box, theme.PANEL)
                canvas.frame(box, theme.LINE)
            kind, name = self.heading
            x, y, w, h = places["head"]
            room = max(0, w - 2 * HEAD_PAD)
            along = Font.width(kind) + HEAD_SPACE
            canvas.text_in(kind, (x + HEAD_PAD, y, room, h), theme.DIM)
            canvas.text_in(name, (x + HEAD_PAD + along, y, max(0, room - along), h), self.colour)
            if self.refusal:
                canvas.text_in(self.refusal, places["view"], theme.FAULT, "centre")
