"""The left panel: what the project holds, as a list to pick a song, a sound or an instrument from."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from . import theme
from .layout import Rect
from .lists import List, Row
from .model import KINDS
from .panels import Panel, within
from .ui import Ui

if TYPE_CHECKING:
    from .session import Session

#: The colour of each section of a listing: of its heading, and of the square before each thing under it.
COLOURS = {"songs": theme.SONG, "sounds": theme.SOUND, "instruments": theme.INSTRUMENT}

#: A listing as a model gives it: each section with the names of its things, and the tag of each.
Listing = Sequence[tuple[str, Sequence[tuple[str, str]]]]


def rows_of(listing: Listing) -> list[Row]:
    """A listing as the rows of a list: each section a heading in its colour, and its things under it.

    A heading is known by its section, and reads as that in capitals: a section that holds nothing has its
    heading all the same. A thing is known by (kind, name), reads as its name, and has its tag at the right.
    """
    rows = []
    for section, things in listing:
        rows.append(Row(section, section.upper(), head=True, colour=COLOURS[section]))
        rows.extend(Row((KINDS[section], name), name, tag=tag) for name, tag in things)
    return rows


class Browser(Panel):
    """The panel down the left: the songs, the sounds and the instruments of the project, and which is picked.

    A click on one picks it in the session. What the session has picked is the picked row, whoever picked it.
    """

    def __init__(self, session: Session, ui: Ui) -> None:
        super().__init__(session, ui)
        self.list = ui.add(List((0, 0, 0, 0), changed=lambda key: session.pick(*key)))
        # The listing the rows were last made of.
        self._listing: Listing | None = None

    def place(self, rect: Rect) -> None:
        """Take the panel's rectangle, and give the list all of it but the air round it and the panel's line."""
        super().place(rect)
        x, y, w, h = rect
        pad, top = theme.BROWSER_PAD, theme.BROWSER_TOP
        # The panel's line runs down its right side.
        self.list.rect = within((x + pad, y + top, w - 2 * pad - 1, h - top - pad), rect)

    def update(self, passed: float = 0.0) -> None:
        """Show the project as it is now: its rows made again if the listing changed, the pick, and the glows."""
        model = self.session.model
        listing = model.listing()
        if listing != self._listing:
            self._listing = listing
            self.list.rows = rows_of(listing)
        self.list.picked = model.picked
        self.list.glow = self.session.glows()
