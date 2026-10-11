"""The middle of the window for a song: its order, its channels, and a pattern as the roll of a channel or its drums."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from ..errors import RowSynthError
from ..note import TICK_HZ
from . import songview, theme
from .font import Font
from .layout import Rect
from .lists import List, Row
from .panels import Panel, within
from .roll import Roll
from .songview import Block, Channel, Pattern
from .steps import Lane, Steps, lanes
from .theme import Colour
from .ui import Choice, Label, Toggle, Ui, share

if TYPE_CHECKING:
    from ..project import Project
    from .canvas import Canvas
    from .session import Session

#: From the sides of the panel to its parts, and from one part to the next.
MARGIN = 4
#: How high a small button is: the row of the order is one such button high, and so are the toggles over the view.
BUTTON_H = 13
#: How wide the list of the channels is, its frame counted. The view takes what is left beside it.
CHANNELS_W = 128
#: From the frame of a part to what stands in it.
INSET = 3
#: From the top of the view's part to the roll or the steps: the room the toggles stand in.
HEAD_H = 20

#: What the two toggles over the view are called, and what the two headings of the list of channels read.
FOLLOW = "FOLLOW"
GHOSTS = "GHOSTS"
TUNED = "TUNED"
DRUMS = "DRUMS"


def place_at(seconds: float, row_ticks: float, blocks: Sequence[Block]) -> tuple[int, float] | None:
    """Where a song is, so many seconds into its playing: the place in its order, and the row of the pattern there.

    A row lasts `row_ticks` ticks, as the engine counts it. A moment at which a pattern ends is in the next one,
    and past the end of the song it has started again, as a loop does. None for a song that has no rows.
    """
    rows = sum(block.rows for block in blocks)
    if rows <= 0 or row_ticks <= 0:
        return None
    row = seconds * TICK_HZ / row_ticks % rows
    place = max(index for index, block in enumerate(blocks) if block.row <= row)
    return place, row - blocks[place].row


def colour_of(place: int) -> Colour:
    """The colour a channel takes by its place in its song: those of `theme.CHANNELS` in turn, round and round."""
    return theme.CHANNELS[place % len(theme.CHANNELS)]


def first_channel(channels: Sequence[Channel]) -> str | None:
    """The channel that is picked as a song is first shown: the first tuned one with notes in the pattern.

    With no such channel it is the first drum channel with notes, then the first tuned one, then the first of
    all. None for a song with no channels.
    """
    best = min(channels, key=lambda held: (not held.notes, held.drums), default=None)
    return None if best is None else best.name


def channel_rows(channels: Sequence[Channel]) -> list[Row]:
    """The channels of a song, as one pattern has them, as the rows of a list.

    The tuned ones stand under the heading TUNED and the drum channels under DRUMS, each heading there whether
    or not it has a channel. A channel is known by its name, and a heading by its text in a tuple, which no name
    is. A channel has its square in the colour of its place in the song, and its instrument as its tag. One the
    pattern has no text for is dim.
    """
    rows: list[Row] = []
    for drums, heading in ((False, TUNED), (True, DRUMS)):
        rows.append(Row((heading,), heading, head=True))
        rows.extend(Row(held.name, held.name, tag=held.inst, square=colour_of(place), dim=not held.written)
                    for place, held in enumerate(channels) if held.drums == drums)
    return rows


def arrange(rect: Rect) -> dict[str, Rect]:
    """Where the parts of the panel go in its rectangle, each by what it is called, and each cut to the panel.

    "order" runs along the top. Under it "channels" stands at the left, `CHANNELS_W` wide, with "list" inside its
    frame. "view" takes the rest: "follow" and "ghosts" side by side at its top, and under them "body", which is
    where the roll or the steps go.
    """
    x, y, w, h = rect
    top, high = y + 2 * MARGIN + BUTTON_H, h - 3 * MARGIN - BUTTON_H
    left, wide = x + 2 * MARGIN + CHANNELS_W, w - 3 * MARGIN - CHANNELS_W
    follow, ghosts = (Font.width(text) + 2 * theme.PAD for text in (FOLLOW, GHOSTS))
    placed = {
        "order": (x + MARGIN, y + MARGIN, w - 2 * MARGIN, BUTTON_H),
        "channels": (x + MARGIN, top, CHANNELS_W, high),
        "list": (x + MARGIN + INSET, top + INSET, CHANNELS_W - 2 * INSET, high - 2 * INSET),
        "view": (left, top, wide, high),
        "follow": (left + INSET, top + INSET, follow, BUTTON_H),
        "ghosts": (left + INSET + follow + theme.BUTTON_GAP, top + INSET, ghosts, BUTTON_H),
        "body": (left + INSET, top + HEAD_H, wide - 2 * INSET, high - HEAD_H - INSET),
    }
    return {part: within(place, rect) for part, place in placed.items()}


class Order(Choice):
    """The order of a song as a row of small buttons, one for each place in it, with the name of the pattern there.

    The picked place is lit. `pick` says which place that is, and `changed` is called with a place whenever
    another one becomes the picked one, by `pick` or by a click. The buttons stand from the left, each as wide as
    its name asks. Where they would be wider than the rectangle, they share it.
    """

    def __init__(self, rect: Rect, changed: Callable[[int], object] | None = None, help: str = "") -> None:
        # A choice is made with an option, and the row has no place to show before it is shown a song.
        super().__init__(rect, ("",), 0, changed, help)
        self.options = ()

    def show(self, names: Sequence[str], picked: int = 0) -> None:
        """Take the names of the patterns in the order they play, and which place is picked. Nothing is told."""
        self.options, self.picked = tuple(names), picked

    def cells(self) -> list[Rect]:
        """The rectangle of each place, side by side from the left."""
        x, y, w, h = self.rect
        widths = [Font.width(name) + 2 * theme.PAD for name in self.options]
        if sum(widths) > w:
            widths = share(w, widths)
        cells = []
        for width in widths:
            cells.append((x, y, width, h))
            x += width
        return cells


class SongPanel(Panel):
    """The middle of the window: the picked song as its order, its channels, and one pattern of it.

    `order.picked` is the picked place in the order, and `channel` the name of the picked channel. A tuned channel
    is shown in `roll`. With a drum channel picked, `steps` shows the drums of the pattern from all its drum
    channels together, the picked one's labels lit. One of the two is seen at a time. While a sound, an instrument
    or nothing is picked the panel shows nothing. A song the engine refuses is not shown: `fault` has the
    refusal's words. The song is read when another thing is picked and when the model has loaded its file again,
    and at no other time.
    """

    def __init__(self, session: Session, ui: Ui) -> None:
        super().__init__(session, ui)
        nowhere = (0, 0, 0, 0)
        # The row of the order is this one widget. Another that shows the order takes its place with the same
        # three things: `show` for the names, `pick` for the picked place, and `changed` for when it is another.
        self.order = self.add(Order(nowhere, changed=lambda place: self._view(),
                                    help="the patterns in the order they play: a click shows one"))
        self.list = self.add(List(nowhere, changed=self._pick))
        self.follow = self.add(Toggle(nowhere, FOLLOW, on=True,
                                      help="the pattern that is shown goes along with the playing"))
        self.ghosts = self.add(Toggle(nowhere, GHOSTS, on=True,
                                      help="the notes of the other channels are drawn dim behind"))
        self.roll = self.add(Roll(nowhere))
        self.steps = self.add(Steps(nowhere))
        self.fault = self.add(Label(nowhere, "", theme.FAULT, "centre"))
        self.channel: str | None = None
        # What was read last: the picked thing, and the project it was read from.
        self._picked: tuple[str, str] | None = None
        self._project: Project | None = None
        # The song that is shown, or None while none is, and what the views need of it.
        self._song: dict | None = None
        self._blocks: list[Block] = []
        self._patterns: dict[str, Pattern] = {}
        self._lanes: list[Lane] = []
        self._bar_rows = 16
        # Whether the channel that is shown is a drum channel. None while no channel is shown.
        self._drums: bool | None = None
        self._parts = arrange(nowhere)
        self._seen()

    def show(self, on: bool) -> None:
        """Bring the panel onto the window, or take it off. Brought back, it shows what it showed before."""
        super().show(on)
        self._seen()

    def place(self, rect: Rect) -> None:
        """Take the panel's rectangle, and lay the widgets out in it as `arrange` has them."""
        super().place(rect)
        self._parts = parts = arrange(rect)
        self.order.rect, self.list.rect = parts["order"], parts["list"]
        self.follow.rect, self.ghosts.rect = parts["follow"], parts["ghosts"]
        self.roll.rect = self.steps.rect = parts["body"]
        self.fault.rect = rect

    def update(self, passed: float = 0.0) -> None:
        """Read the picked thing if it or the project is another than before, and show where the playing is.

        `passed` is the seconds since the frame before, which the panel has no use for.
        """
        model = self.session.model
        # The model has no public name for its project. A file that was loaded again is another project.
        picked, project = model.picked, model.project
        if picked != self._picked or project is not self._project:
            fresh = picked != self._picked
            self._picked, self._project = picked, project
            self._read(fresh)
        self.roll.ghosts = self.ghosts.is_on
        self._play()

    def draw(self, canvas: Canvas) -> None:
        """Draw the two framed parts the list and the view stand in. With no song shown nothing is drawn."""
        if self._song is not None:
            for part in ("channels", "view"):
                canvas.box(self._parts[part], theme.PANEL)
                canvas.frame(self._parts[part], theme.LINE)

    def _read(self, fresh: bool) -> None:
        """Read the picked thing from the project. Only a song is shown, and one the engine refuses is told."""
        self._song, self.fault.text = None, ""
        if self._picked is not None and self._picked[0] == "song":
            try:
                self._take(self._project.song(self._picked[1]), fresh)
            except RowSynthError as error:
                self._song, self.fault.text = None, str(error)
        self._seen()

    def _take(self, song: dict, fresh: bool) -> None:
        """Take a song as it was read, and show it. Whatever the engine refuses in it is refused.

        `fresh` is for another song than before: it starts at the first place in its order, with the channel
        `first_channel` gives. A song read again keeps its place and its channel where it still has them.
        """
        blocks, _ = songview.blocks(song)
        self._patterns = {block.name: songview.pattern(song, block.name) for block in blocks}
        self._lanes = lanes(self._patterns.values(), songview.drum_rows(song))
        self._bar_rows = songview.bar_rows(song)
        self._blocks = blocks
        place = 0 if fresh or self.order.picked >= len(blocks) else self.order.picked
        self.order.show([block.name for block in blocks], place)
        if fresh or self.channel not in song["channels"]:
            self.channel = first_channel(self._channels())
        self._song = song
        self._view()

    def _channels(self) -> tuple[Channel, ...]:
        """The channels of the song as the pattern at the picked place has them. A song with no order has none."""
        return self._patterns[self._blocks[self.order.picked].name].channels if self._blocks else ()

    def _pick(self, channel: str) -> None:
        """Pick a channel of the song by its name, and show it."""
        self.channel = channel
        self._view()

    def _view(self) -> None:
        """Show the picked channel of the pattern at the picked place: in the list, and in the roll or the steps.

        A tuned channel is the roll's one channel. A drum channel is lit among all the drums of the pattern.
        """
        channels = self._channels()
        self.list.rows = channel_rows(channels)
        self.list.picked = self.channel
        place = next((place for place, held in enumerate(channels) if held.name == self.channel), None)
        self._drums = None if place is None else channels[place].drums
        if place is not None:
            pattern = self._patterns[self._blocks[self.order.picked].name]
            if self._drums:
                self.steps.show(pattern, self._lanes, self._bar_rows, self._bar_rows // 4)
                self.steps.picked = self.channel
            else:
                self.roll.show(pattern, self.channel, self._bar_rows, colour_of(place))
        self._seen()

    def _play(self) -> None:
        """Set the playhead of the views, and with FOLLOW on have the picked place go along with the playing.

        The playhead is there while the song plays and the pattern that plays is the one that is shown, at
        whatever place in the order it plays.
        """
        at = None
        if self._song is not None and self.session.playing:
            at = place_at(self.session.position(), self._song["row_ticks"], self._blocks)
        if at is not None and self.follow.is_on:
            self.order.pick(at[0])
        shown = at is not None and self._blocks[at[0]].name == self._blocks[self.order.picked].name
        self.roll.playhead = self.steps.playhead = at[1] if shown else None

    def _seen(self) -> None:
        """Have each widget seen or not: by whether the panel is on the window, and by what there is to show."""
        song = self.shown and self._song is not None
        for widget in (self.order, self.list, self.follow):
            widget.visible = song
        self.roll.visible = self.ghosts.visible = song and self._drums is False
        self.steps.visible = song and self._drums is True
        self.fault.visible = self.shown and bool(self.fault.text)
