"""The left panel: what the project holds, as a list to pick a song, a sound or an instrument from, and to change."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from ..errors import RowSynthError
from . import projectedit, theme
from .layout import Rect
from .lists import List, Row
from .model import KINDS
from .panels import Panel, within
from .popups import Item, Menu, ask
from .projectedit import Change
from .ui import Event, Pos, Ui

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


def in_name(char: str) -> bool:
    """Whether the name of a song, a sound or an instrument can hold a character, as `projectedit.allowed` has it."""
    try:
        projectedit.allowed({}, "song", char)
    except RowSynthError:
        return False
    return True


class Things(List):
    """The list of the left panel: a press of the right button that is on no thing is told of as well.

    `asked` is then called with None for the key. That is a press on a heading, under the rows or on the bar.
    """

    def on(self, event: Event, ui: Ui) -> bool:
        """Use what a list uses, and a press of the right button anywhere else on the list."""
        if super().on(event, ui):
            return True
        if event.kind != "down" or event.button != 3 or self.asked is None:
            return False
        self.asked(None, event.pos)
        return True


class Browser(Panel):
    """The panel down the left: the songs, the sounds and the instruments of the project, and which is picked.

    A click on one picks it in the session. What the session has picked is the picked row, whoever picked it.

    The right button on a thing opens a menu at the mouse, to give it another name, to copy it and to delete
    it. On a heading or under the rows it opens one to make a new song, a new sound or a new instrument. Each
    is done as `projectedit` has it, to the project as it is then, and is one step that can be taken back,
    however many things it changes: an instrument has its new name in every song and sound that names it. A
    name is typed into a box that opens over the name. What is refused changes nothing, and the session says
    why: an instrument that is still used is not deleted, and a name that is refused leaves the box open, so
    that it can be mended. A new thing, a copy and a thing under its new name is the picked one afterwards.
    Where the picked thing is deleted, the session picks the first thing there is.
    """

    def __init__(self, session: Session, ui: Ui) -> None:
        super().__init__(session, ui)
        self.list = ui.add(Things((0, 0, 0, 0), changed=lambda key: session.pick(*key), asked=self._ask))
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

    def _ask(self, key: tuple[str, str] | None, pos: Pos) -> None:
        """Open the menu of a thing at the mouse, or with None for the thing the menu of what can be made new."""
        if key is None:
            items: Sequence[Item | None] = [Item(f"New {kind}", lambda kind=kind: self._add(kind))
                                            for kind in KINDS.values()]
        else:
            kind, name = key
            items = (Item("Rename...", lambda: self._rename(kind, name)),
                     Item("Copy", lambda: self._add(kind, name)),
                     None,
                     Item("Delete", lambda: self._do(lambda: projectedit.removed(self._data(), kind, name),
                                                     f"{kind} {name} deleted")))
        self.ui.open(Menu(items, pos, chosen=lambda act: act()))

    def _data(self) -> dict:
        """The project as it is now, as the plain data `projectedit` works on."""
        return self.session.model.project.to_dict()

    def _do(self, make: Callable[[], list[Change]], label: str, pick: tuple[str, str] | None = None) -> bool:
        """Make the changes a rule of `projectedit` gives, as one step called `label`, and say whether they were made.

        What the rule refuses is said through the session, and what the save refuses the session says itself:
        nothing is changed then. `pick` is the thing that is picked once the changes are made.
        """
        try:
            changes = make()
        except RowSynthError as error:
            self.session.say(str(error))
            return False
        if not self.session.change_many(changes, label):
            return False
        if pick is not None:
            self.session.pick(*pick)
        return True

    def _add(self, kind: str, like: str | None = None) -> None:
        """Make a new thing of a kind, or a copy of the one called `like`, under the next free name, and pick it."""
        data = self._data()
        name = projectedit.free(data, kind, like)
        self._do(lambda: projectedit.added(data, kind, name, like),
                 f"new {kind} {name}" if like is None else f"{kind} {like} copied as {name}", (kind, name))

    def _rename(self, kind: str, old: str, typed: str | None = None) -> None:
        """Open a box over the name of a thing: enter gives the thing that name everywhere, and picks it.

        A name that is refused is in the box again, which is open as before: the session has said why by then.
        `typed` is that name. None of it is picked then, and the caret stands after it, so that what is typed
        next is added to it.
        """
        def entered(new: str) -> None:
            if new != old and not self._do(lambda: projectedit.renamed(self._data(), kind, old, new),
                                           f"{kind} {old} renamed to {new}", (kind, new)):
                self._rename(kind, old, new)

        box = ask(self.ui, self.list.name_rect((kind, old)) or self.list.rect, old if typed is None else typed,
                  entered, allowed=in_name, help=f"the name of {kind} {old}: enter gives it the new one, escape "
                                                 "leaves it")
        if typed is not None:
            box.anchor = box.caret
