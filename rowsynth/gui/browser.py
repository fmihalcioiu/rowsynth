"""The left panel: what the project holds, as a list to pick a song, a sound or an instrument from, and to change."""
from __future__ import annotations

from collections.abc import Callable, Container, Hashable, Sequence
from typing import TYPE_CHECKING

from ..errors import RowSynthError
from . import projectedit, theme
from .kit import Plus, thing_menu, told
from .layout import Rect
from .lists import List, Row
from .model import KINDS, own
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

#: What the row reads that stands after the last thing of a section, to make one more. Its key is this word and
#: the kind of thing it makes: no kind is called so.
NEW = "new"
#: What the wheel does over the list, and a press of the right button that is on no thing, as the help line says
#: them. What the list says of itself where the mouse is on no row.
WHEEL = "along the list"
RIGHT = "a new song, sound or instrument"
LIST_HELP = f"what the project holds   {told(right=RIGHT, wheel=WHEEL)}"
#: What the first of the ready instruments reads, which is none of them, and the two things the menu of an
#: instrument has that no other menu has.
EMPTY = "Empty"
MAKE_SHARED = "Make shared"
INTO_SONG = "Copy into this song"


def rows_of(listing: Listing) -> list[Row]:
    """A listing as the rows of a list: each section a heading in its colour, and its things under it.

    A heading is known by its section, and reads as that in capitals: a section that holds nothing has its
    heading all the same. A thing is known by (kind, name), reads as its name, and has its tag at the right.
    An instrument with a tag is one of a song's own, listed by the name it is picked by: it reads as the name
    it has in its song, which is the tag. After the last thing of each section stands the row that makes one
    more, known by (`NEW`, kind).
    """
    rows = []
    for section, things in listing:
        rows.append(Row(section, section.upper(), head=True, colour=COLOURS[section]))
        rows.extend(Row((KINDS[section], name), own(name)[1] if tag and section == "instruments" else name, tag=tag)
                    for name, tag in things)
        rows.append(Row((NEW, KINDS[section]), NEW, plus=True))
    return rows


def adds(kind: str) -> str:
    """What a press does on the row that makes one more thing of a kind, as the help line says it."""
    return f"a new {kind}" + (", empty or one of the ready ones" if kind == "instrument" else "")


def row_help(row: Row, shared: Container[str] = ()) -> str:
    """The line of help of a row of the list: of a heading, of a thing by its kind and its name, of a row that adds.

    An instrument of a song's own is named with its song. `shared` is the names of the project's instruments,
    by which `model.own` knows one of those.
    """
    if row.head:
        return told(f"fold the {row.key} away or bring them back", RIGHT, WHEEL)
    kind, name = row.key
    if row.plus:
        return told(adds(name), RIGHT, WHEEL)
    song, name = own(name, shared) if kind == "instrument" else (None, name)
    said = f"{kind} {name}" + (f" of {song}" if song else "")
    return told(f"show {said}, twice to rename it", "its menu", WHEEL)


def plus_rect(row: Rect) -> Rect:
    """Where the button of a row that adds stands in the rectangle of its row: over the room of the square."""
    x, y, _, h = row
    return (x + theme.LIST_PAD - (theme.PLUS - theme.LIST_SQUARE) // 2, y + (h - theme.PLUS) // 2, theme.PLUS,
            theme.PLUS)


def in_name(char: str) -> bool:
    """Whether the name of a song, a sound or an instrument can hold a character, as `projectedit.allowed` has it."""
    try:
        projectedit.allowed({}, "song", char)
    except RowSynthError:
        return False
    return True


class Things(List):
    """The list of the left panel: a double press on a thing, and a right press on no thing, are told of as well.

    `named` is called with the key of the thing that the second of two presses of the left button is on, for
    its owner to open the box the name is typed in. The first of the two has picked the thing, as any press
    does. `asked` is called with None for the key at a press of the right button that is on no thing: on a
    heading, on a row that adds, under the rows or on the bar.
    """

    def __init__(self, rect: Rect, changed: Callable[[Hashable], object] | None = None, help: str = "",
                 asked: Callable[[Hashable, Pos], object] | None = None,
                 named: Callable[[Hashable], object] | None = None) -> None:
        super().__init__(rect, changed=changed, help=help, asked=asked)
        self.named = named

    def on(self, event: Event, ui: Ui) -> bool:
        """Tell of a double press on a thing, use what a list uses, and a right press anywhere else on the list."""
        row = self.at(event.pos) if event.kind == "down" and event.button == 1 and event.double else None
        if row is not None and not (row.head or row.plus) and self.named is not None:
            self.named(row.key)
            return True
        if super().on(event, ui):
            return True
        if event.kind != "down" or event.button != 3 or self.asked is None:
            return False
        self.asked(None, event.pos)
        return True


class Browser(Panel):
    """The panel down the left: the songs, the sounds and the instruments of the project, and which is picked.

    A click on one picks it in the session. What the session has picked is the picked row, whoever picked it.

    After the last thing of each section stands a row that makes one more, with its button in `plus` by the
    kind of thing: a new song, a new sound, and for an instrument a short list to choose from, `Empty` and the
    ready instruments of `projectedit.STARTERS` by name. A ready one comes under its own name, or the next
    free one after it.

    The right button on a thing opens its menu at the mouse, as `kit.thing_menu` orders it: to give it another
    name, to copy it and to remove it. A double press on a thing opens the box for its name as well. On a
    heading or under the rows the right button opens a menu to make a new song, a new sound or a new
    instrument. Each is done as `projectedit` has it, to the project as it is then, and is one step that can
    be taken back, however many things it changes: an instrument has its new name in every song and sound
    that names it. A name is typed into a box that opens over the name. What is refused changes nothing, and
    the session says why: an instrument that is still used is not removed, and a name that is refused leaves
    the box open, so that it can be mended. A new thing, a copy and a thing under its new name is the picked
    one afterwards. Where the picked thing is removed, the session picks the first thing there is.

    The menu of an instrument of the project has `Copy into this song` as well: the song that is picked, or
    was picked last, gets the instrument as one of its own under the same name. A song that has one of that
    name is refused, and with no such song the item is dim.

    An instrument of a song's own has a menu of its own. `Rename...` gives it another name in its song and
    nowhere else. It cannot be copied. `Make shared` moves it to the instruments of the project under its
    name, where the song goes on playing it, and picks it there: a project that has an instrument of that
    name already is refused. `Remove` takes it out of its song, unless the song still plays it.
    """

    def __init__(self, session: Session, ui: Ui) -> None:
        super().__init__(session, ui)
        nowhere = (0, 0, 0, 0)
        self.list = ui.add(Things(nowhere, changed=lambda key: session.pick(*key), help=LIST_HELP,
                                  asked=self._ask, named=lambda key: self._rename(*key)))
        self.list.told = lambda row: row_help(row, session.model.project.instruments)
        self.list.added = lambda key: self._plus(key[1])
        #: The button of each row that adds, by the kind of thing it makes. They stand over the list.
        self.plus = {kind: ui.add(Plus(nowhere, lambda kind=kind: self._plus(kind), told(adds(kind))))
                     for kind in KINDS.values()}
        # The listing the rows were last made of, and the song that is picked, or was picked last.
        self._listing: Listing | None = None
        self._song: str | None = None

    def place(self, rect: Rect) -> None:
        """Take the panel's rectangle, and give the list all of it but the air round it and the panel's line."""
        super().place(rect)
        x, y, w, h = rect
        pad, top = theme.BROWSER_PAD, theme.BROWSER_TOP
        # The panel's line runs down its right side.
        self.list.rect = within((x + pad, y + top, w - 2 * pad - 1, h - top - pad), rect)
        self._stand()

    def update(self, passed: float = 0.0) -> None:
        """Show the project as it is now: its rows made again if the listing changed, the pick, and the glows."""
        model = self.session.model
        listing = model.listing()
        if listing != self._listing:
            self._listing = listing
            self.list.rows = rows_of(listing)
        self.list.picked = model.picked
        self.list.glow = self.session.glows()
        if model.picked is not None and model.picked[0] == "song":
            self._song = model.picked[1]
        self._stand()

    def _stand(self) -> None:
        """Have the button of each row that adds stand in its row, as the list is scrolled now.

        One whose row is folded away, or not all in view, is not seen and has no room: what is in view of its
        row is pressed in its place.
        """
        x, y, _, _ = self.list.rect
        for kind, button in self.plus.items():
            row = self.list.row_rect((NEW, kind))
            rect = (x, y, 0, 0) if row is None else plus_rect(row)
            button.visible = row is not None and within(rect, self.list.rect) == rect
            button.rect = rect if button.visible else (x, y, 0, 0)

    def _plus(self, kind: str) -> None:
        """Make one more thing of a kind. For an instrument a list opens under the row, to choose what it is."""
        if kind != "instrument":
            self._add(kind)
            return
        x, y, _, h = self.list.row_rect((NEW, kind)) or self.list.rect
        items = [Item(EMPTY, lambda: self._add(kind)), None,
                 *(Item(name, lambda name=name: self._start(name)) for name in projectedit.STARTERS)]
        self.ui.open(Menu(items, (x, y + h), chosen=lambda act: act()))

    def _target(self) -> str | None:
        """The song an instrument is copied into: the one that is picked, or was picked last. None with no such."""
        picked = self.session.model.picked
        song = picked[1] if picked is not None and picked[0] == "song" else self._song
        return song if song in self.session.model.project.songs else None

    def _ask(self, key: tuple[str, str] | None, pos: Pos) -> None:
        """Open the menu of a thing at the mouse, or with None for the thing the menu of what can be made new."""
        if key is None:
            items: Sequence[Item | None] = [Item(f"New {kind}", lambda kind=kind: self._add(kind))
                                            for kind in KINDS.values()]
        else:
            kind, name = key
            song, inst = own(name, self.session.model.project.instruments) if kind == "instrument" else (None, name)
            said = name if song is None else f"{inst} of {song}"

            def rename() -> None:
                self._rename(kind, name)

            def remove() -> None:
                self._do(lambda: projectedit.removed(self._data(), kind, inst, song), f"{kind} {said} removed")

            if song is not None:
                items = thing_menu(rename, None, [Item(MAKE_SHARED, lambda: self._do(
                    lambda: projectedit.shared(self._data(), song, inst), f"{kind} {said} made shared",
                    (kind, inst)))], remove)
            else:
                into = self._target()
                special = [Item(INTO_SONG, lambda: self._do(
                    lambda: projectedit.own_copied(self._data(), into, name), f"{kind} {name} copied into {into}",
                    (kind, f"{into}.{name}")), enabled=into is not None)] if kind == "instrument" else []
                items = thing_menu(rename, lambda: self._add(kind, name), special, remove)
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

    def _start(self, starter: str) -> None:
        """Make a new instrument that is a copy of a ready one, under its name or the next free one, and pick it."""
        data = self._data()
        name = projectedit.fresh(data, "instrument", starter)
        self._do(lambda: projectedit.started(data, starter, name), f"new instrument {name}", ("instrument", name))

    def _rename(self, kind: str, picked: str, typed: str | None = None) -> None:
        """Open a box over the name of a thing: enter gives the thing that name everywhere, and picks it.

        `picked` is the name the thing is picked by. An instrument of a song's own has its new name in its
        song, and nowhere else. A name that is refused is in the box again, which is open as before: the
        session has said why by then. `typed` is that name. None of it is picked then, and the caret stands
        after it, so that what is typed next is added to it.
        """
        song, old = own(picked, self.session.model.project.instruments) if kind == "instrument" else (None, picked)
        said = f"{kind} {old}" + (f" of {song}" if song else "")

        def entered(new: str) -> None:
            if new == old:
                return
            if song is None:
                done = self._do(lambda: projectedit.renamed(self._data(), kind, old, new),
                                f"{said} renamed to {new}", (kind, new))
            else:
                done = self._do(lambda: projectedit.own_renamed(self._data(), song, old, new),
                                f"{said} renamed to {new}", (kind, f"{song}.{new}"))
            if not done:
                self._rename(kind, picked, new)

        box = ask(self.ui, self.list.name_rect((kind, picked)) or self.list.rect, old if typed is None else typed,
                  entered, allowed=in_name, help=f"the name of {said}: enter gives it the new one, escape leaves it")
        if typed is not None:
            box.anchor = box.caret
