"""The middle of the window: the panels of a song, a sound and an instrument in one rectangle, one shown at a time."""
from __future__ import annotations

from typing import TYPE_CHECKING

from . import theme
from .instpanel import InstPanel
from .layout import Rect
from .panels import Panel
from .songpanel import SongPanel
from .soundpanel import SoundPanel
from .ui import Ui

if TYPE_CHECKING:
    from .canvas import Canvas
    from .session import Session

#: What the three panels are called: each by the kind of thing it shows.
SONG = "song"
SOUND = "sound"
INSTRUMENT = "instrument"
#: The one line that stands in the middle while nothing is picked.
EMPTY = "the project is empty"


def fitting(picked: tuple[str, str] | None) -> str | None:
    """Which panel fits what is picked: the one called by its kind, and None for nothing picked."""
    return None if picked is None else picked[0]


class Middle(Panel):
    """The middle of the window: what is picked, in the panel that fits it.

    `panels` is the three panels by what they are called, and `fits` is which of them fits what is picked now.
    That one is on the window and the others are off it. With nothing picked `fits` is None: none is on the
    window, and one dim line says that the project is empty.
    """

    def __init__(self, session: Session, ui: Ui) -> None:
        super().__init__(session, ui)
        self.panels: dict[str, Panel] = {SONG: SongPanel(session, ui), SOUND: SoundPanel(session, ui),
                                         INSTRUMENT: InstPanel(session, ui)}
        self.fits: str | None = None
        self._turn()

    def show(self, on: bool) -> None:
        """Bring the middle onto the window, or take it off: off, none of its panels is on the window."""
        super().show(on)
        self._turn()

    def place(self, rect: Rect) -> None:
        """Take the rectangle, and give all of it to each panel: those off the window are ready to come on."""
        super().place(rect)
        for panel in self.panels.values():
            panel.place(rect)

    def update(self, passed: float = 0.0) -> None:
        """Have the panel that fits what is picked now on the window, and have that panel take its frame."""
        self._turn()
        panel = self.panels.get(self.fits)
        if panel is not None and panel.shown:
            panel.update(passed)

    def draw(self, canvas: Canvas) -> None:
        """Draw what the panel on the window draws. With nothing picked, draw the line that says so."""
        panel = self.panels.get(self.fits)
        if panel is not None:
            panel.draw(canvas)
        else:
            canvas.text_in(EMPTY, self.rect, theme.DIM, "centre")

    def _turn(self) -> None:
        """Work out which panel fits what is picked, and have that one on the window and the others off it."""
        self.fits = fitting(self.session.model.picked)
        for name, panel in self.panels.items():
            on = self.shown and name == self.fits
            if panel.shown != on:
                panel.show(on)
