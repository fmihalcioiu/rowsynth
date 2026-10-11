"""The middle of the window for a song: its order, its channels, and a pattern as the roll of a channel or its drums."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..errors import RowSynthError
from ..note import TICK_HZ
from . import songview, theme
from .font import Font
from .layout import Rect
from .lists import List, Row
from .panels import Panel, within
from .roll import Roll
from .session import piano
from .songview import Block, Channel, Note, Pattern
from .steps import Lane, Steps, lanes
from .strips import SongStrip
from .theme import Colour
from .ui import Event, Label, Toggle, Ui

if TYPE_CHECKING:
    from ..project import Project
    from .canvas import Canvas
    from .session import Session

#: From the sides of the panel to its parts, and from one part to the next.
MARGIN = 4
#: How high the song strip is. Under the names of its blocks that leaves 54 pixels for the lanes of the channels:
#: 3 for each of 18 channels, the least at which a note has three heights and two drum channels stand apart.
STRIP_H = 74
#: How high a small button is: the toggles over the view are such buttons.
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
#: What is said, after the channel and its pattern, of an edit that ended after the channel was changed outside.
MEANWHILE = "was changed outside while it was edited here: this edit stands, and undo brings the other back"


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


def lowest_c(pitches: range) -> int:
    """The lowest c among the pitches a roll has in view, which go up from the lowest of them.

    Where there is no c among them, it is the first c over them.
    """
    return -(-pitches.start // 12) * 12


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

    "strip" runs along the top, `STRIP_H` high. Under it "channels" stands at the left, `CHANNELS_W` wide, with
    "list" inside its frame. "view" takes the rest: "follow" and "ghosts" side by side at its top, and under them
    "body", which is where the roll or the steps go.
    """
    x, y, w, h = rect
    top, high = y + 2 * MARGIN + STRIP_H, h - 3 * MARGIN - STRIP_H
    left, wide = x + 2 * MARGIN + CHANNELS_W, w - 3 * MARGIN - CHANNELS_W
    follow, ghosts = (Font.width(text) + 2 * theme.PAD for text in (FOLLOW, GHOSTS))
    placed = {
        "strip": (x + MARGIN, y + MARGIN, w - 2 * MARGIN, STRIP_H),
        "channels": (x + MARGIN, top, CHANNELS_W, high),
        "list": (x + MARGIN + INSET, top + INSET, CHANNELS_W - 2 * INSET, high - 2 * INSET),
        "view": (left, top, wide, high),
        "follow": (left + INSET, top + INSET, follow, BUTTON_H),
        "ghosts": (left + INSET + follow + theme.BUTTON_GAP, top + INSET, ghosts, BUTTON_H),
        "body": (left + INSET, top + HEAD_H, wide - 2 * INSET, high - HEAD_H - INSET),
    }
    return {part: within(place, rect) for part, place in placed.items()}


class SongPanel(Panel):
    """The middle of the window: the picked song as its order, its channels, and one pattern of it.

    `strip` is the order as blocks. `strip.picked` is the picked place in it, which the panel keeps nowhere else:
    it is None for a song with no order. `channel` is the name of the picked channel. A tuned channel is shown in
    `roll`. With a drum channel picked, `steps` shows the drums of the pattern from all its drum channels
    together, the picked one's labels lit. One of the two is seen at a time. While a sound, an instrument or
    nothing is picked the panel shows nothing. A song the engine refuses is not shown: `fault` has the refusal's
    words. The song is read, and the strip shown it, when another thing is picked and when the model has loaded
    its file again, and at no other time.

    What the mouse and the keys make of the notes in the roll or the steps is saved at once: the channel's notes
    are written as its text for the pattern, and that is one change to the file, and one step that can be taken
    back. The song is then read again as after any change, so the views show what the file holds. A change that
    is refused leaves them as they were, and the session says why. While a gesture is under way in a view the
    song is not read, and the picked place does not go along with the playing: what changed meanwhile, here or
    outside, is shown when the gesture has ended. An edit that ends after its channel was changed outside is
    saved, and the session is told so. A note that is placed and saved is heard, and so is a key of the roll's
    strip that is pressed. With the focus on the roll the keys of the computer play the instrument of its
    channel as a piano, from the lowest c the roll has in view.
    """

    def __init__(self, session: Session, ui: Ui) -> None:
        super().__init__(session, ui)
        nowhere = (0, 0, 0, 0)
        self.strip = self.add(SongStrip(nowhere, changed=lambda place: self._view(),
                                        help="the patterns in the order they play: a click shows one"))
        self.list = self.add(List(nowhere, changed=self._pick))
        self.follow = self.add(Toggle(nowhere, FOLLOW, on=True,
                                      help="the pattern that is shown goes along with the playing"))
        self.ghosts = self.add(Toggle(nowhere, GHOSTS, on=True,
                                      help="the notes of the other channels are drawn dim behind"))
        self.roll = self.add(Roll(nowhere, edited=lambda notes, label: self._save(self.roll.channel, notes, label),
                                  sounded=self._sound, typed=self._key))
        self.steps = self.add(Steps(nowhere, edited=self._save, sounded=self._drum))
        self.fault = self.add(Label(nowhere, "", theme.FAULT, "centre"))
        self.channel: str | None = None
        # What was read last: the picked thing, and the project it was read from.
        self._picked: tuple[str, str] | None = None
        self._project: Project | None = None
        # The song that is shown, or None while none is, and what the views need of it.
        self._song: dict | None = None
        self._blocks: tuple[Block, ...] = ()
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
        self.strip.rect, self.list.rect = parts["strip"], parts["list"]
        self.follow.rect, self.ghosts.rect = parts["follow"], parts["ghosts"]
        self.roll.rect = self.steps.rect = parts["body"]
        self.fault.rect = rect

    def update(self, passed: float = 0.0) -> None:
        """Read the picked thing if it or the project is another than before, and show where the playing is.

        The steps take their frame first, which ends a run of the wheel whose time is up. While a gesture is
        under way in the roll or the steps nothing is read, so the notes do not change under the hand. `passed`
        is the seconds since the frame before, which the panel has no use for.
        """
        self.steps.update()
        if not self._busy():
            self._look()
        self.roll.ghosts = self.ghosts.is_on
        self._play()

    def draw(self, canvas: Canvas) -> None:
        """Draw the two framed parts the list and the view stand in. With no song shown nothing is drawn."""
        if self._song is not None:
            for part in ("channels", "view"):
                canvas.box(self._parts[part], theme.PANEL)
                canvas.frame(self._parts[part], theme.LINE)

    def _look(self) -> None:
        """Read the picked thing where it, or the project, is another than the one that was read last."""
        model = self.session.model
        # A file that was loaded again is another project.
        picked, project = model.picked, model.project
        if picked != self._picked or project is not self._project:
            fresh = picked != self._picked
            self._picked, self._project = picked, project
            self._read(fresh)

    def _busy(self) -> bool:
        """Whether a gesture is under way in the roll or in the steps."""
        return self.roll.busy or self.steps.busy

    def _text(self, project: Project, pattern: str, channel: str) -> object:
        """The text a project has for a channel in a pattern of the song that is shown, or None where it has none."""
        return project.songs.get(self._picked[1], {}).get("patterns", {}).get(pattern, {}).get(channel)

    def _save(self, channel: str, notes: Sequence[Note], label: str) -> bool:
        """Save the notes a view gave for a channel of the pattern that is shown, and show what the file then holds.

        The notes are written as the channel's text, filling the rows of the pattern, and put in the file as one
        step: `label` is what the view calls it. A channel the pattern had no text for gets its text. The answer
        is whether they were saved. A change the project refuses is not made, and the session's message is the
        refusal. Notes that cannot be written as text are not saved either, and the session is told why.

        Where the channel's text is another by now than the one the views were shown, it was changed outside
        while a gesture was under way here. The notes of the hand are saved all the same, and the session is told
        that they were, and that taking the step back gives the other text.
        """
        pattern = self._patterns[self._blocks[self.strip.picked].name]
        place = f"{channel} of pattern {pattern.name}"
        try:
            text = songview.text(notes, pattern.rows, channel, self._song)
        except RowSynthError as error:
            self.session.say(f"{place}: {error}")
            return False
        shown, held = (self._text(project, pattern.name, channel)
                       for project in (self._project, self.session.model.project))
        saved = self.session.change(f"songs.{self._picked[1]}.patterns.{pattern.name}.{channel}", text,
                                    f"{place}: {label}")
        if saved and held not in (shown, text):
            self.session.say(f"{place} {MEANWHILE}")
        self._look()
        return saved

    def _sound(self, pitch: float, inst: str) -> None:
        """Let a note of an instrument of the song be heard, over whatever plays."""
        self.session.sound(inst, pitch, self._picked[1])

    def _drum(self, channel: str, letter: str) -> None:
        """Let what a drum letter plays in the song be heard. The channel of the hit makes no difference to it."""
        self.session.drum(letter, self._picked[1])

    def _key(self, event: Event) -> bool:
        """Take a key the roll has the focus for and does not use, and say whether it was used.

        A key of the piano sounds its note with the instrument of the channel in the roll, counted from the
        lowest c the roll has in view. With ctrl held it is no key of the piano.
        """
        note = piano(event.key)
        if note is None or "ctrl" in event.mods:
            return False
        self._sound(lowest_c(self.roll.map.pitches()) + note, self._song["channels"][self.roll.channel]["inst"])
        return True

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
        `first_channel` gives. A song read again keeps its channel where it still has it. Its place is the strip's
        to keep: it stays while the order is as it was, and an order that is another starts at its first place,
        so the place is never past the end of an order that came back shorter.
        """
        self.strip.show(song)
        blocks = self.strip.map.blocks
        if fresh:
            self.strip.picked, self.strip.offset = (0 if blocks else None), 0
        self._patterns = {block.name: songview.pattern(song, block.name) for block in blocks}
        self._lanes = lanes(self._patterns.values(), songview.drum_rows(song))
        self._bar_rows = songview.bar_rows(song)
        self._blocks = blocks
        if fresh or self.channel not in song["channels"]:
            self.channel = first_channel(self._channels())
        self._song = song
        self._view()

    def _channels(self) -> tuple[Channel, ...]:
        """The channels of the song as the pattern at the picked place has them. A song with no order has none."""
        return self._patterns[self._blocks[self.strip.picked].name].channels if self._blocks else ()

    def _pick(self, channel: str) -> None:
        """Pick a channel of the song by its name, and show it."""
        self.channel = channel
        self._view()

    def _view(self) -> None:
        """Show the picked channel of the pattern at the picked place: in the list, and in the roll or the steps.

        A tuned channel is the roll's one channel. A drum channel is lit among all the drums of the pattern, which
        are given the drum table of the song, so that their hits can be changed.
        """
        channels = self._channels()
        self.list.rows = channel_rows(channels)
        self.list.picked = self.channel
        place = next((place for place, held in enumerate(channels) if held.name == self.channel), None)
        self._drums = None if place is None else channels[place].drums
        if place is not None:
            pattern = self._patterns[self._blocks[self.strip.picked].name]
            if self._drums:
                self.steps.show(pattern, self._lanes, self._bar_rows, self._bar_rows // 4,
                                self._song.get("drums") or {})
                self.steps.picked = self.channel
            else:
                self.roll.show(pattern, self.channel, self._bar_rows, colour_of(place))
        self._seen()

    def _play(self) -> None:
        """Set the playheads, and with FOLLOW on have the picked place go along with the playing.

        While the song plays, the playhead of the strip is the row of the whole song that plays. That of the views
        is the row of the pattern, and it is there only while the pattern that plays is the one that is shown, at
        whatever place in the order it plays. While a gesture is under way in a view, the picked place stays
        where it is, so that the pattern does not change under the hand: it goes along again when the gesture
        has ended.
        """
        at = None
        if self._song is not None and self.session.playing:
            at = place_at(self.session.position(), self._song["row_ticks"], self._blocks)
        if at is not None and self.follow.is_on and at[0] != self.strip.picked and not self._busy():
            self.strip.picked = at[0]
            self._view()
        shown = at is not None and self._blocks[at[0]].name == self._blocks[self.strip.picked].name
        self.roll.playhead = self.steps.playhead = at[1] if shown else None
        # The row of the song is the row its place starts at and the row of the pattern there, put together again.
        self.strip.playhead = None if at is None else self._blocks[at[0]].row + at[1]

    def _seen(self) -> None:
        """Have each widget seen or not: by whether the panel is on the window, and by what there is to show."""
        song = self.shown and self._song is not None
        for widget in (self.strip, self.list, self.follow):
            widget.visible = song
        self.roll.visible = self.ghosts.visible = song and self._drums is False
        self.steps.visible = song and self._drums is True
        self.fault.visible = self.shown and bool(self.fault.text)
