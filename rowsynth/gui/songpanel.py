"""The middle of the window for a song: its heading, its order, its settings, the notes of a channel or the mixer."""
from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Collection, Sequence
from typing import TYPE_CHECKING

from ..errors import RowSynthError
from ..note import TICK_HZ
from . import chordview, roll, songedit, songview, strips, theme
from .channelbar import ChannelBar, instruments
from .chordview import ChordBox, ChordList
from .dials import Stepper, plain
from .font import Font
from .kit import Heading, Plus, Slot, thing_menu, told
from .layerview import Layers
from .layout import Rect
from .lists import List, Row
from .mixer import Mixer
from .panels import Panel, within
from .popups import Item, Menu, ask
from .roll import Roll
from .session import piano
from .songview import Block, Channel, Note, Pattern
from .steps import PLUS_HELP, Lane, Steps, lanes
from .steps import WHEEL_HELP as STEPS_WHEEL
from .strips import SongStrip
from .theme import Colour
from .ui import Button, Choice, Event, Label, Pos, Ui, Widget

if TYPE_CHECKING:
    from ..project import Project
    from .canvas import Canvas
    from .session import Session

#: From the sides of the panel to its parts, and from one part to the next.
MARGIN = 4
GAP = 2
#: How high the song strip is. Under the names of its blocks that leaves 54 pixels for the lanes of the channels:
#: 3 for each of 18 channels, the least at which a note has three heights and two drum channels stand apart.
STRIP_H = 74
#: From the frame of the side column to the list of the channels in it, and from the frame the mixer and the
#: layers stand in to what stands in it.
LIST_INSET = 2
INSET = 3

#: What the song is headed as, and the three things the panel shows under its settings, one at a time: a pattern
#: as the notes of a channel, the mixer of all the channels, and the layers of the song with its intro. And what
#: the button reads that opens the list of the chords.
SONG = "SONG"
NOTES = "NOTES"
MIXER = "MIXER"
LAYERS = "LAYERS"
CHORDS = "CHORDS"
#: What the two headings of the list of channels read, and what the row of the list is known by with one of them
#: that the plus of that kind of channel stands in.
TUNED = "TUNED"
DRUMS = "DRUMS"
MORE = "+"
#: What the wheel does over the list of the channels, as the help line says it. What the list says of itself where
#: the mouse is on no row, the strip off its blocks, and the roll and the steps where a press would do nothing.
LIST_WHEEL = "along the channels"
LIST_HELP = f"a dim channel is not heard just now, or has nothing in this pattern   {told(wheel=LIST_WHEEL)}"
STRIP_HELP = f"the patterns in the order they play   {told(wheel=strips.WHEEL_HELP)}"
ROLL_HELP = told(wheel=roll.WHEEL_HELP)
STEPS_HELP = told(wheel=STEPS_WHEEL)
#: What the buttons that add say, and the one that opens the list of the chords.
BLOCK_HELP = told("a new empty pattern at the end of the song")
TUNED_HELP = told("a new tuned channel")
DRUMS_HELP = told("a new drum channel")
CHORDS_HELP = told("the chords of the project: a chord opens its menu, and the plus makes a new one")
#: What the plus of the drums says while the song has a drum for every letter there is.
NO_LETTER = "the song has a drum for every letter a drum can have"
#: What a new channel is called where the song has none of its kind yet to be named after.
NEW_TUNED = "tuned"
NEW_DRUMS = "drums"
#: The c the two rows of piano keys start at while a drum channel is picked.
DRUM_C = 48
#: What the things of a menu read that only one kind of thing has.
AGAIN = "Play it again after this"
EMPTY_AFTER = "New empty pattern after this"
MAKE_DRUMS = "Make it a drum channel"
MAKE_TUNED = "Make it a tuned channel"
INSTRUMENT = "Instrument"
NOTE = "Note..."
LENGTH = "Length..."
VELOCITY = "Velocity..."
CHANGE = "Change..."
#: What is said, after the channel and its pattern, of an edit that ended after the channel was changed outside.
MEANWHILE = "was changed outside while it was edited here: this edit stands, and undo brings the other back"
#: The song's own settings stand at the right end of the settings bar: its tempo, its grid, which is the rows of
#: a beat, and its key. These are what the three are called there, and how wide they are. The key is a small
#: button at each end, and between them the room of a count of two figures with its sign.
TEMPO = "TEMPO"
GRID = "GRID"
KEY = "KEY"
TEMPO_W = 56
GRID_W = 34
KEY_W = 2 * theme.BUTTON_H + Font.width("+12") + 2 * theme.GAP + 1
#: What is said, after the name of the song, where its order or its channels were changed outside while a block
#: was carried here, or a menu or a box was open.
ELSEWHERE = "was changed outside meanwhile, so nothing was done here"
#: What the two things at the end of the menu of a bar's chord read: the one that clears the bar, and the one
#: that opens the box a new chord is made in.
NO_CHORD = "none"
NEW_CHORD = "New chord..."
#: What a number of bars is typed with, and how many of them it can have.
DIGITS = "0123456789"
MOST_DIGITS = 3

# A name as what it starts with and the number it ends on, which may be none.
_COUNTED = re.compile(r"(.*?)(\d*)", re.DOTALL)


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


def tempo_steps(row_ticks: float, rows_per_beat: float = 4) -> list[tuple[float, float]]:
    """The tempos a song can be put at, the slowest first, each as the ticks of a row and the beats a minute.

    They are the ones `songedit.tempos` gives for the rows of a beat. A song whose rows last another number of
    ticks has its own tempo among them, at its place, so that it can be shown.
    """
    found = dict(songedit.tempos(rows_per_beat))
    found.setdefault(row_ticks, 60 * TICK_HZ / (rows_per_beat * row_ticks) if row_ticks > 0 else 0.0)
    return sorted(found.items(), reverse=True)


def tempo_text(beats: float) -> str:
    """A tempo in beats a minute as it is shown: to one place after the point, and whole without its point."""
    return plain(round(beats, 1))


def shift_text(semis: int) -> str:
    """How many semitones a song was moved, as it is shown: with its sign, and "0" for not at all."""
    return f"{semis:+d}" if semis else "0"


def bars_text(bars: float) -> str:
    """A number of bars in words, as "1 bar", "4 bars" and "0.5 bars"."""
    return f"{plain(bars)} {'bar' if bars == 1 else 'bars'}"


def free_name(taken: Collection[str], name: str) -> str:
    """The next free name after one: `lead` gives `lead2`, or `lead3` where `lead2` is taken, and `pad2` `pad3`."""
    stem, count = _COUNTED.fullmatch(name).groups()
    number = int(count or 1) + 1
    while f"{stem}{number}" in taken:
        number += 1
    return f"{stem}{number}"


def in_name(char: str) -> bool:
    """Whether the name of a pattern or of a channel can hold a character, as `songedit.allowed` has it."""
    try:
        songedit.allowed({"patterns": ()}, "pattern", char)
    except RowSynthError:
        return False
    return True


def step_to(channels: dict, name: str, by: int) -> int | None:
    """Where `songedit.channel_moved` puts a channel that goes one row up in the list of the channels, or one down.

    `channels` is what a song holds for each channel by its name. `by` is -1 for up and 1 for down. The list has
    the tuned channels and the drum channels apart, so a channel goes before the one of its own kind that comes
    before it in the song, or after the one that comes after it. None where there is none: the first of its
    kind does not go up, and the last does not go down.
    """
    names = list(channels)
    kind = [held for held in names if bool(channels[held].get("drums")) == bool(channels[name].get("drums"))]
    at = kind.index(name) + by
    if not 0 <= at < len(kind):
        return None
    return [held for held in names if held != name].index(kind[at]) + (by > 0)


def _state(song: dict) -> int:
    """A number for the texts of all the patterns of a song, which is another when any of them is."""
    return hash(json.dumps(song.get("patterns")))


def channel_rows(channels: Sequence[Channel], heard: Collection[str] | None = None) -> list[Row]:
    """The channels of a song, as one pattern has them, as the rows of a list.

    The tuned ones stand under the heading TUNED and the drum channels under DRUMS, each heading there whether
    or not it has a channel. A channel is known by its name, and a heading by its text in a tuple, which no name
    is. A channel has its square in the colour of its place in the song, and its instrument as its tag. One the
    pattern has no text for is dim. So is one that is not heard just now: `heard` has the names of those that
    are, and None is for all of them. After the last channel of each kind comes an empty row, known by `MORE`
    and the heading in a tuple: the plus that makes a channel of that kind stands in it.
    """
    rows: list[Row] = []
    for drums, heading in ((False, TUNED), (True, DRUMS)):
        rows.append(Row((heading,), heading, head=True))
        rows.extend(Row(held.name, held.name, tag=held.inst, square=colour_of(place),
                        dim=not held.written or (heard is not None and held.name not in heard))
                    for place, held in enumerate(channels) if held.drums == drums)
        rows.append(Row((MORE, heading), "", square=theme.PANEL))
    return rows


def is_more(row: Row) -> bool:
    """Whether a row of the list of the channels is the empty one that a plus stands in."""
    return isinstance(row.key, tuple) and row.key[0] == MORE


def channel_help(row: Row) -> str:
    """The line of help of a row of the list of the channels: of a heading, and of a channel by its name.

    The empty row of a plus has none of its own: the plus says its own line.
    """
    if row.head:
        return told("fold these channels away or bring them back", wheel=LIST_WHEEL)
    if is_more(row):
        return ""
    return told(f"show channel {row.key}, twice to rename it", "its menu", LIST_WHEEL)


def new_channel(song: dict, drums: bool, picked: str | None, shared: Sequence[str]) -> tuple[str, str]:
    """The name and the instrument of the channel that the plus of a kind makes in a song.

    It is named after the last channel of its kind, with the next free name: `lead` gives `lead2`. It plays
    the instrument of the picked channel where that is of its kind, and else that of the last of its kind. A
    song with no channel of the kind yet names it `NEW_TUNED` or `NEW_DRUMS`, and has it play the instrument of
    the picked channel, or of the first there is, or the first of `shared`, the instruments the song can use.
    With no instrument anywhere it is refused.
    """
    channels = song["channels"]
    kind = [name for name, held in channels.items() if bool(held.get("drums")) == drums]
    word = NEW_DRUMS if drums else NEW_TUNED
    name = free_name(channels, kind[-1]) if kind else word if word not in channels else free_name(channels, word)
    if picked in kind or (not kind and picked in channels):
        source = picked
    else:
        source = kind[-1] if kind else next(iter(channels), None)
    inst = channels[source]["inst"] if source is not None else next(iter(shared), None)
    if inst is None:
        raise RowSynthError("there is no instrument for a new channel to play: make one first")
    return name, inst


def facts(song: dict) -> str:
    """The few facts the heading has of a song: how many patterns it has, and how many channels."""
    counts = ((len(song.get("patterns", ())), "pattern"), (len(song.get("channels", ())), "channel"))
    return ", ".join(f"{count} {word}{'' if count == 1 else 's'}" for count, word in counts)


def arrange(rect: Rect) -> dict[str, Rect]:
    """Where the parts of the panel go in its rectangle, each by what it is called, and each cut to the panel.

    They are the skeleton every editor has, from the top: "head" is the heading, `theme.HEAD_H` high, "strip"
    the order of the song, `STRIP_H` high, with "more" after it, the plus that adds a pattern at its end, and
    "bar" the settings bar, `theme.BAR_H` high. "foot" is at the
    lower end, `theme.FOOT_H` high: the velocity lane stands in it. The main area is what is left between the
    bar and the foot: "side" is the column down its left, `theme.SIDE_W` wide, with "list", the list of the
    channels, inside its frame, and "body" is the rest, which is where the roll or the steps go.

    "mixer" is the main area and the foot together, where the mixer or the layers stand in place of the notes,
    and "desk" is the room inside its frame.
    """
    x, y, w, h = rect
    left, wide, end = x + MARGIN, w - 2 * MARGIN, y + h - MARGIN
    strip = y + MARGIN + theme.HEAD_H + GAP
    bar = strip + STRIP_H + GAP
    top = bar + theme.BAR_H + GAP
    foot = max(top, end - theme.FOOT_H)
    high = foot - GAP - top
    placed = {
        "head": (left, y + MARGIN, wide, theme.HEAD_H),
        "strip": (left, strip, wide - theme.PLUS - GAP, STRIP_H),
        "more": (left + wide - theme.PLUS, strip + (STRIP_H - theme.PLUS) // 2, theme.PLUS, theme.PLUS),
        "bar": (left, bar, wide, theme.BAR_H),
        "side": (left, top, theme.SIDE_W, high),
        "list": (left + LIST_INSET, top + LIST_INSET, theme.SIDE_W - 2 * LIST_INSET, high - 2 * LIST_INSET),
        "body": (left + theme.SIDE_W + GAP, top, wide - theme.SIDE_W - GAP, high),
        "foot": (left, foot, wide, theme.FOOT_H),
        "mixer": (left, top, wide, end - top),
        "desk": (left + INSET, top + INSET, wide - 2 * INSET, end - top - 2 * INSET),
    }
    return {part: within(place, rect) for part, place in placed.items()}


def key_parts(room: Rect) -> tuple[Rect, Rect, Rect]:
    """Where the three widgets of the song's key go in the room they have in the settings bar, from the left.

    A button that is as wide as it is high stands at each end, the one that moves the song lower first, and
    the count of the semitones it was moved has the room between them.
    """
    x, y, w, h = room
    side = min(h, w // 2)
    return (x, y, side, h), (x + side, y, w - 2 * side, h), (x + w - side, y, side, h)


class Channels(List):
    """The list of the channels of a song, down the side of its notes: a double press on a channel renames it.

    `named` is called with the name of the channel that the second of two presses is on, for its owner to
    open the box the name is typed in. The first of the two has picked the channel, as any press does.

    After the last channel of each kind stands a plus, in the empty row `channel_rows` gives it: `tuned` and
    `drums` are the two, parts that are widgets of their own. Their owner adds the `parts` to the Ui after the
    list, and says with `seen` whether the list is on the window. A press on one calls `added` with whether it
    is the one of the drum channels. The empty rows take no press and are never picked: the keys that move
    the pick pass them.
    """

    def __init__(self, rect: Rect, changed: Callable[[str], object] | None = None, help: str = "",
                 asked: Callable[[str, Pos], object] | None = None,
                 named: Callable[[str], object] | None = None,
                 added: Callable[[bool], object] | None = None) -> None:
        self.named = named
        # Kept under a name of its own: the list has an `added` too, for rows that are a plus themselves.
        self._more = added
        self.tuned = Plus((0, 0, 0, 0), lambda: self._add(False), TUNED_HELP)
        self.drums = Plus((0, 0, 0, 0), lambda: self._add(True), DRUMS_HELP)
        self._on = True
        super().__init__(rect, changed=changed, help=help, asked=asked)

    @property
    def parts(self) -> tuple[Widget, ...]:
        """The parts that are widgets of their own: the plus of the tuned channels, and that of the drum channels."""
        return self.tuned, self.drums

    def _add(self, drums: bool) -> None:
        """Tell the owner of the plus that was pressed."""
        if self._more is not None:
            self._more(drums)

    def seen(self, on: bool) -> None:
        """Bring the list onto the window with its two buttons, or take it off."""
        self._on = self.visible = on
        self.lay()

    def lay(self) -> None:
        """Put each plus on the row it has, as the list is scrolled now. The owner calls this every frame.

        A plus stands where the square of a channel would, and is seen while its row is whole in the list: not
        while the channels of its kind are folded away, nor while the row is scrolled out of view.
        """
        x, y, _, h = self.rect
        for heading, plus in ((TUNED, self.tuned), (DRUMS, self.drums)):
            room = self.name_rect((MORE, heading))
            whole = room is not None and y <= room[1] and room[1] + theme.PLUS <= y + h
            plus.rect = (within((x + theme.LIST_PAD, room[1], theme.PLUS, theme.PLUS), self.rect) if whole
                         else (x, y, 0, 0))
            plus.visible = self._on and whole

    def on(self, event: Event, ui: Ui) -> bool:
        """Tell of a double press of the left button on a channel, and leave anything else to the list.

        A press on the empty row of a plus is not used. The keys that move the pick are handed to the list
        with those rows out of it, so that the pick never comes to one: the rows are put back after, unless
        the owner gave the list other rows meanwhile, as it does when another channel is picked.
        """
        if event.kind == "key":
            rows, self.rows = self.rows, [row for row in self.rows if not is_more(row)]
            lent = self.rows
            try:
                return super().on(event, ui)
            finally:
                if self.rows is lent:
                    self.rows = rows
                room = self.name_rect(self.picked)
                if room is not None:
                    top = room[1] - self.rect[1] + self.scroll.offset
                    self.scroll.show(top, top + theme.LIST_ROW_H)
                self.lay()
        row = self.at(event.pos) if event.kind == "down" else None
        if row is not None and is_more(row):
            return False
        if row is None or row.head or event.button != 1 or not event.double or self.named is None:
            used = super().on(event, ui)
            self.lay()
            return used
        self.named(row.key)
        return True


class SongPanel(Panel):
    """The middle of the window: the picked song under its heading, as its order, its settings and its notes.

    It is laid out as every editor is. `heading` says what song it is, and how many patterns and channels it
    has. `strip` is the order as blocks. `strip.picked` is the picked place in it, which the panel keeps nowhere
    else: it is None for a song with no order. `bar` is the settings bar: the settings of the picked channel,
    which are changed there, at its right end the song's own, and the toggles FOLLOW and GHOSTS. `channel` is
    the name of the picked channel. `list` is the side column, the channels of the song. A tuned channel is
    shown in `roll`, between its keyboard and its scroll bars, with its velocity lane in the foot. With a drum
    channel picked, `steps` shows the drums of the pattern from all its drum channels together, the picked
    one's labels lit, with their own velocity lane in the foot. One of the two is seen at a time. While a
    sound, an instrument or nothing is picked the panel shows nothing. A song the engine refuses is not shown:
    `fault` has the refusal's words. The song is read, and the strip shown it, when another thing is picked and
    when the model has loaded its file again, and at no other time. Whenever the pattern that is shown is
    another, the session is told its name as `session.shown`, and None while no song is shown.

    `choice` stands at the right end of the heading, and is what stands under the settings bar: NOTES is the
    list of the channels with the view of the picked one and the lane, and MIXER is `mixer`, every channel side
    by side, in place of all three. It stays as it was chosen for as long as the song stays picked, and another
    song starts with its notes. A press on the name of a channel in the mixer picks that channel and shows its
    notes. In the list a channel that is not heard just now, muted or silenced by the solo of another, is dim,
    as one the pattern has no text for is. LAYERS is `layers`, in the room the mixer has: the groups and the
    versions of the song, and its intro.

    What the mouse and the keys make of the notes in the roll or the steps is saved at once: the channel's notes
    are written as its text for the pattern, and that is one change to the file, and one step that can be taken
    back. The song is then read again as after any change, so the views show what the file holds. A change that
    is refused leaves them as they were, and the session says why. While a gesture is under way in a view the
    song is not read, and the picked place does not go along with the playing: what changed meanwhile, here or
    outside, is shown when the gesture has ended. An edit that ends after its channel was changed outside is
    saved, and the session is told so. A note that is placed and saved is heard, and so is a key of the roll's
    strip that is pressed. With the focus on the roll or on the steps the keys of the computer play the
    instrument of the picked channel as a piano: in the roll from the lowest c it has in view.

    The song as a whole is changed here too, and each such change is the whole song saved as `songedit` leaves
    it: one change to the file, and one step that can be taken back, called by the name of the song and what
    was done. What `songedit` or the save refuses changes nothing, and the session says why. A block of the
    strip that is dragged is moved in the order, and with ctrl its pattern is copied to where it was let go:
    the block that was moved or made is then the picked one, and the view of the strip is where it was.

    Wherever there can be one more of something a plus stands after the last: `more` after the last block of
    the strip, which puts an empty pattern at the end of the order, one in the list after the last tuned
    channel and one after the last drum channel, and one in the steps under the last lane, which adds a drum
    to the song. A thing with a name has its menu on the right button, in the order `kit.thing_menu` gives
    every such menu: to rename it, to copy it, what only its kind has, and to remove it. A block can be played
    again, have an empty pattern put after it, and be made longer or shorter. A channel can be moved up or
    down, and made a drum channel or a tuned one while it has no notes. A drum, on the label of its lane, can
    be given another instrument, note, length and velocity: its letter is its name, so it is not renamed or
    copied, and those two are dim. A name, the bars of a pattern, the letter of a new drum and what a drum
    plays are typed into a box that opens over the thing: a double press on the name of a block opens it as
    well, and so does one on a channel of the list. A text that is refused leaves the box open, so that the
    reason can be read and the text mended. A new block, and a new channel, is the picked one. Anything else
    leaves picked what was, where the song still has it: a pattern and a channel also under a new name.

    The roll is shown the chords the song has written for the bars of its pattern, read as the chord check
    reads them: a song with none has an empty strip there. A press on a bar of that strip opens a menu at the
    mouse with the chords of the project by name, `none` and `New chord...`. A chord that is chosen is saved as
    the chord of the bar, and `none` leaves the bar with no chord: either is the song's chords for the pattern
    saved, one change to the file and one step that can be taken back. `New chord...` opens the box of
    `chordview.ChordBox`, where a chord is made of a root and a kind and named. It is added to the chords of the
    project and set on the bar as one: one write to the file, and one step that can be taken back.

    `chords` is the button CHORDS at the right end of the heading. It opens the list of the chords of the
    project, with a plus after the last that makes a new one in the same box. A chord that is pressed there
    opens its menu: to rename it, which renames it in the bars of every song, to change it in the box, opened
    on it, and to remove it, which is refused while a bar of any song names it. Each is the rule of that name
    in `chordview`, saved as one write to the file and one step that can be taken back.

    `tempo` and the two buttons `lower` and `higher` are the song's own settings. They stand at the right end
    of the settings bar, whatever is shown under it. The tempo is one of those `tempo_steps` gives, shown in
    beats a minute. The buttons move every tuned note of the song a semitone down or up, and `shifted` between
    them counts the semitones since the song was opened: a move that is taken back is counted back. Between
    the tempo and the key stands `grid`, the rows of a beat: it steps through those `songedit.offered` gives,
    at each of which `songedit.gridded` leaves the song sounding as it did.

    A press on a block of the strip counts as a gesture under way from the press to its release, and so does a
    menu or a box of the panel's for as long as it is open. Where the order or the channels of the song were
    changed outside meanwhile, what the hand asked for is not done, and the session says so.
    """

    def __init__(self, session: Session, ui: Ui) -> None:
        super().__init__(session, ui)
        nowhere = (0, 0, 0, 0)
        # The heading and the bar are added before what stands on them, so that they are drawn under it.
        self.heading = self.add(Heading(nowhere, SONG, theme.SONG))
        self.choice = self.add(Choice(nowhere, (NOTES, MIXER, LAYERS), changed=lambda index: self._seen(), help=told(
            "show the notes of one channel, the mixer of them all, or the layers and the intro")))
        self.chords = self.add(Button(nowhere, CHORDS, action=self._ask_chords, help=CHORDS_HELP, arrow=True))
        self.strip = self.add(SongStrip(nowhere, changed=lambda place: self._view(), help=STRIP_HELP,
                                        moved=self._moved, copied=self._copied, asked=self._ask_block,
                                        named=self._rename))
        for part in self.strip.parts:
            self.add(part)
        self.more = self.add(Plus(nowhere, self._new_block, BLOCK_HELP))
        self.bar = ChannelBar(self)
        self.tempo = self.add(Stepper(nowhere, ("",), changed=self._tempo, help=told(
            "an arrow steps the tempo, in beats a minute", wheel="steps it")))
        self.grid = self.add(Stepper(nowhere, ("",), changed=self._regrid, help=told(
            "an arrow steps the rows of a beat, and the song sounds as before", wheel="steps them")))
        self.lower = self.add(Button(nowhere, "-", action=lambda: self._shift(-1),
                                     help=told("every tuned note of the song a semitone lower")))
        self.shifted = self.add(Label(nowhere, shift_text(0), theme.TEXT, "centre"))
        self.higher = self.add(Button(nowhere, "+", action=lambda: self._shift(1),
                                      help=told("every tuned note of the song a semitone higher")))
        # The song's own settings, which stand in the settings bar.
        self._settings: tuple[Widget, ...] = (self.tempo, self.grid, self.lower, self.shifted, self.higher)
        self.list = self.add(Channels(nowhere, changed=self._pick, help=LIST_HELP, asked=self._ask_channel,
                                      named=self._rename_channel, added=self._new_channel))
        self.list.told = channel_help
        for part in self.list.parts:
            self.add(part)
        self.roll = self.add(Roll(nowhere, ROLL_HELP,
                                  edited=lambda notes, label: self._save(self.roll.channel, notes, label),
                                  sounded=self._sound, typed=self._key, chord_asked=self._ask_chord))
        for part in self.roll.parts:
            self.add(part)
        self.steps = self.add(Steps(nowhere, STEPS_HELP, edited=self._save, sounded=self._drum, typed=self._key,
                                    asked=self._ask_drum, added=self._new_drum))
        for part in self.steps.parts:
            self.add(part)
        self.fault = self.add(Label(nowhere, "", theme.FAULT, "centre"))
        self.layers = Layers(self)
        self.mixer = Mixer(self, picked=self._choose)
        self.channel: str | None = None
        # The ticks a row lasts at each of the tempos the stepper has, and the rows of a beat at each of its grids.
        self._ticks: list[float] = []
        self._grids: list[int] = []
        # How many semitones each song was moved since it was opened, by its name. And the count that goes with
        # the texts of a song as a move found them and as it left them, by the name and by `_state` of them.
        self._semis: dict[str, int] = {}
        self._states: dict[tuple[str, int], int] = {}
        # The menu or the box that the panel opened last.
        self._popup: Widget | None = None
        # What is picked when the song is read after a change to it as a whole: the place in the order, the view
        # of the strip and the channel. None for the place or the channel leaves the one there is.
        self._kept: tuple[int | None, int, str | None] = (None, 0, None)
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
        # The channels that were heard when the list was given its rows: None for all of them.
        self._heard: Collection[str] | None = None
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
        self.heading.rect, self.strip.rect, self.list.rect = parts["head"], parts["strip"], parts["list"]
        self.more.rect = parts["more"]
        self.heading.right([self.choice, self.chords])
        own = self.bar.place(parts["bar"], (Slot(TEMPO, self.tempo, TEMPO_W), Slot(GRID, self.grid, GRID_W),
                                            Slot(KEY, None, KEY_W)))
        self.lower.rect, self.shifted.rect, self.higher.rect = key_parts(own[2])
        self.roll.rect = self.steps.rect = parts["body"]
        self.roll.lane.rect = self.steps.lane.rect = parts["foot"]
        self.fault.rect = rect
        self.mixer.place(parts["desk"])
        self.layers.place(parts["desk"])
        self._seen()

    @property
    def mixing(self) -> bool:
        """Whether MIXER is chosen: the mixer stands under the strip, and not the notes."""
        return self.choice.options[self.choice.picked] == MIXER

    @property
    def layering(self) -> bool:
        """Whether LAYERS is chosen: the layers and the intro stand under the strip, and not the notes."""
        return self.choice.options[self.choice.picked] == LAYERS

    def update(self, passed: float = 0.0) -> None:
        """Read the picked thing if it or the project is another than before, and show where the playing is.

        The bar takes its frame first, which ends a run of the wheel whose time is up, and the strip, which
        moves its view under a block that is carried. While a gesture is under way nothing is read, so that
        nothing changes under the hand. The list is given its rows again when other channels are heard than
        it shows as heard, and puts its two buttons where its rows are now. The mixer takes its frame last:
        `passed` is the seconds since the frame before, which its meters go by.
        """
        self.bar.update()
        self.strip.update()
        if not self._busy():
            self._look()
        if self._song is not None and self.session.model.heard != self._heard:
            self._list()
        self.list.lay()
        self.roll.ghosts = self.bar.ghosts.is_on
        self._play()
        self.mixer.update(passed)
        self.layers.update()

    def draw(self, canvas: Canvas) -> None:
        """Draw the framed parts: the side column and the foot, or the one of the mixer. None with no song."""
        if self._song is not None:
            for part in ("mixer",) if self.mixing or self.layering else ("side", "foot"):
                canvas.box(self._parts[part], theme.PANEL)
                canvas.frame(self._parts[part], theme.LINE)

    def _look(self, again: bool = False) -> None:
        """Read the picked thing where it, or the project, is another than the one that was read last, or `again`."""
        model = self.session.model
        # A file that was loaded again is another project.
        picked, project = model.picked, model.project
        if again or picked != self._picked or project is not self._project:
            fresh = picked != self._picked
            self._picked, self._project = picked, project
            self._read(fresh)

    def _busy(self) -> bool:
        """Whether a gesture is under way: in the roll, in the steps, on a block of the strip, or in a menu or a box."""
        return (self.roll.busy or self.steps.busy or self.strip.busy
                or (self._popup is not None and self.ui.modal is self._popup))

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
        """Take a key the roll or the steps have the focus for and do not use, and say whether it was used.

        A key of the piano sounds its note with the instrument of the picked channel, counted from the lowest
        c the roll has in view, and with a drum channel picked from `DRUM_C`. With ctrl held it is no key of
        the piano.
        """
        note = piano(event.key)
        if note is None or "ctrl" in event.mods or self.channel not in self._song["channels"]:
            return False
        low = DRUM_C if self._drums else lowest_c(self.roll.map.pitches())
        self._sound(low + note, self._song["channels"][self.channel]["inst"])
        return True

    def _read(self, fresh: bool) -> None:
        """Read the picked thing from the project. Only a song is shown, and one the engine refuses is told."""
        self._song, self.fault.text = None, ""
        if self._picked is not None and self._picked[0] == "song":
            try:
                self._take(self._project.song(self._picked[1]), fresh)
            except RowSynthError as error:
                self._song, self.fault.text = None, str(error)
        if self._song is None:
            self.session.shown = None
        self._seen()

    def _take(self, song: dict, fresh: bool) -> None:
        """Take a song as it was read, and show it. Whatever the engine refuses in it is refused.

        `fresh` is for another song than before: it starts at the first place in its order, with the channel
        `first_channel` gives, and with its notes under the strip. A song read again keeps its channel where it
        still has it. Its place is the strip's to keep: it stays while the order is as it was, and an order that
        is another starts at its first place, so the place is never past the end of an order that came back
        shorter. A song read after it was changed here as a whole has what `_change` wants picked: its place,
        or the last where the order does not go that far, with the strip's view where it was, and its channel.
        """
        place, offset, channel = self._kept
        self.strip.show(song)
        blocks = self.strip.map.blocks
        if fresh:
            self.strip.picked, self.strip.offset = (0 if blocks else None), 0
            self.choice.picked = self.choice.options.index(NOTES)
        elif place is not None and blocks:
            self.strip.picked, self.strip.offset = min(place, len(blocks) - 1), offset
            self.strip.offset = self.strip.map.offset
        self._patterns = {block.name: songview.pattern(song, block.name) for block in blocks}
        self._bar_rows = songview.bar_rows(song)
        self._blocks = blocks
        if channel in song["channels"]:
            self.channel = channel
        if fresh or self.channel not in song["channels"]:
            self.channel = first_channel(self._channels())
        self._song = song
        self.heading.show(self._picked[1], facts(song))
        steps = tempo_steps(song["row_ticks"], song.get("rows_per_beat", 4))
        self._ticks = [ticks for ticks, _ in steps]
        self.tempo.options = tuple(tempo_text(beats) for _, beats in steps)
        self.tempo.picked = self._ticks.index(song["row_ticks"])
        self._grids = songedit.offered(song)
        self.grid.options = tuple(plain(rows) for rows in self._grids)
        self.grid.picked = self._grids.index(song.get("rows_per_beat", 4))
        self._count()
        self._view()

    def _count(self) -> None:
        """Show how many semitones the song was moved since it was opened.

        It is the count that goes with the texts of its patterns as they are, where a move found them so or
        left them so: that is how a move that is taken back is counted back. With any other texts it is the
        count there was, since no other change moves the song.
        """
        name = self._picked[1]
        self._semis[name] = self._states.get((name, _state(self._song)), self._semis.get(name, 0))
        self.shifted.text = shift_text(self._semis[name])

    def _change(self, make: Callable[[dict], dict], label: str | Callable[[dict], str], place: int | None = None,
                channel: str | None = None) -> dict | None:
        """Save the song as a change of `songedit` leaves it, as one step, and show what the file then holds.

        `make` is given the song as the project holds it and gives the new one. `label` is what the step is
        called after the name of the song: words, or a function that makes them of the new song. The answer is
        the new song, or None where it was not saved. What `make` refuses is not saved, and the session is told
        the refusal. A save the project refuses is told by the session itself.

        Where the order or the channels the project holds are not the ones that are shown, they were changed
        outside while a block was carried or a menu or a box was open, and a place or a name the hand chose
        may be another thing by now. Nothing is saved then, and the session is told so.

        The song is read again whatever came of it, so the widgets show what the file holds. `place` is the
        place in the order that is picked then, and `channel` the channel, where the song has them. With None
        it is the one that was. The view of the strip stays where it was.
        """
        name = self._picked[1]
        song = self.session.model.project.songs.get(name, {})
        made = None
        if (song.get("order") != [block.name for block in self._blocks]
                or list(song.get("channels", ())) != list(self._song["channels"])):
            self.session.say(f"{name} {ELSEWHERE}")
        else:
            try:
                made = make(song)
            except RowSynthError as error:
                self.session.say(str(error))
        if made is not None:
            words = label if isinstance(label, str) else label(made)
            if not self.session.change(f"songs.{name}", made, f"{name}: {words}"):
                made = None
        self._kept = (self.strip.picked if place is None else place, self.strip.offset, channel)
        self._look(again=True)
        self._kept = (None, 0, None)
        return made

    def _moved(self, place: int, to: int) -> None:
        """Save the order with the block of a place moved to another, where it is then the picked one."""
        self._change(lambda song: songedit.moved(song, place, to), f"pattern {self._blocks[place].name} moved", to)

    def _copied(self, place: int, to: int) -> None:
        """Save the song with a copy of the pattern of a place as a new pattern, played at another place and picked."""
        name = self._blocks[place].name
        self._change(lambda song: songedit.moved(songedit.copied(song, place), place + 1, to),
                     lambda made: f"pattern {name} copied as {made['order'][to]}", to)

    def _bars(self, pattern: str) -> float:
        """How many bars a pattern of the song that is shown is long, which need be no whole number."""
        return self._patterns[pattern].rows / self._bar_rows

    def _ask_block(self, place: int | None, pos: Pos) -> None:
        """Open the menu of the block at a place in the order, at the mouse. Past the last block there is none.

        It is the menu of a named thing, as `kit.thing_menu` has it: to rename the pattern, to copy it as a new
        one, then what only a block has, and to remove the block. A new pattern is as long as the one it comes
        after, in whole bars. What an item makes is the picked block afterwards. With the block taken out, the
        picked one is the one that was, or the one that took its place. The last block of an order cannot be
        taken out, and its item is dim.
        """
        if place is None:
            return
        name, after, picked = self._blocks[place].name, place + 1, self.strip.picked
        bars = self._bars(name)
        special = (
            Item(AGAIN, lambda: self._change(
                lambda song: songedit.repeated(song, place), f"pattern {name} played again", after)),
            Item(EMPTY_AFTER, lambda: self._change(
                lambda song: songedit.added(song, place, max(1, math.ceil(bars))),
                lambda made: f"new pattern {made['order'][after]} after {name}", after)),
            Item(f"Length ({bars_text(bars)})...", lambda: self._resize(place)),
        )
        self._menu(thing_menu(
            rename=lambda: self._rename(place),
            copy=lambda: self._change(lambda song: songedit.copied(song, place),
                                      lambda made: f"pattern {name} copied as {made['order'][after]}", after),
            special=special,
            remove=(lambda: self._change(lambda song: songedit.dropped(song, place),
                                         f"pattern {name} taken out of the order", picked - (picked > place)))
            if len(self._blocks) > 1 else None), pos)

    def _new_block(self) -> None:
        """Save the song with a new empty pattern at the end of its order, which is then the picked block.

        It is as long as the pattern it comes after, in whole bars, and named after it.
        """
        if not self._blocks:
            return
        last = len(self._blocks) - 1
        name = self._blocks[last].name
        self._change(lambda song: songedit.added(song, last, max(1, math.ceil(self._bars(name)))),
                     lambda made: f"new pattern {made['order'][-1]} at the end", last + 1)

    def _ask_channel(self, channel: str, pos: Pos) -> None:
        """Open the menu of a channel of the list, at the mouse.

        It is the menu of a named thing, as `kit.thing_menu` has it: to rename the channel, to copy it with
        its notes as the next channel, then what only a channel has, and to remove it. A channel moves among
        those of its own kind, as the list has them, and not past the first or the last of them. It is made a
        drum channel, or a tuned one, by `songedit.channel_kind`, which refuses one that has notes. A copy is
        the picked channel afterwards.
        """
        channels = self._song["channels"]
        new = free_name(channels, channel)
        drums = bool(channels[channel].get("drums"))

        def move(way: str, to: int | None) -> Item:
            return Item(f"Move {way}", lambda: self._change(
                lambda song: songedit.channel_moved(song, channel, to), f"channel {channel} moved {way}"),
                enabled=to is not None)

        kind = Item(MAKE_TUNED if drums else MAKE_DRUMS, lambda: self._change(
            lambda song: songedit.channel_kind(song, channel, not drums),
            f"channel {channel} made a {'tuned' if drums else 'drum'} channel"))
        self._menu(thing_menu(
            rename=lambda: self._rename_channel(channel),
            copy=lambda: self._change(lambda song: songedit.channel_copied(song, channel, new),
                                      f"channel {channel} copied as {new}", channel=new),
            special=(move("up", step_to(channels, channel, -1)), move("down", step_to(channels, channel, 1)), kind),
            remove=lambda: self._change(lambda song: songedit.channel_dropped(song, channel),
                                        f"channel {channel} removed")), pos)

    def _new_channel(self, drums: bool) -> None:
        """Save the song with one more channel of a kind, as `new_channel` names it, which is then the picked one."""
        try:
            name, inst = new_channel(self._song, drums, self.channel,
                                     instruments(self.session.model.project, self._picked[1]))
        except RowSynthError as error:
            self.session.say(str(error))
            return
        self._change(lambda song: songedit.channel_added(song, name, inst, drums),
                     f"new {'drum' if drums else 'tuned'} channel {name}", channel=name)

    def _ask_drum(self, place: int, pos: Pos) -> None:
        """Open the menu of the drum of a lane of the steps, at the mouse: what its letter plays in the song.

        It is the menu of a named thing, as `kit.thing_menu` has it. A drum is called by its letter, which is
        not changed, and is not copied, so those two are dim. Its instrument is picked from a second menu, of
        the instruments the song can use by name, the one it plays dim. Its note, the rows it is held for and
        its velocity are typed into a box over its label. Each is the song saved as `songedit` leaves it.
        """
        letter = self.steps.lanes[place].letter
        held = self._song["drums"][letter]

        def play(name: str) -> Item:
            return Item(name, lambda: self._change(lambda song: songedit.drum_set(song, letter, "inst", name),
                                                   f"drum {letter}: instrument {name}"), enabled=name != held["inst"])

        names = instruments(self.session.model.project, self._picked[1])
        special = (
            Item(INSTRUMENT, lambda: self._menu([play(name) for name in names], pos)),
            Item(NOTE, lambda: self._drum_box(place, letter, "note")),
            Item(LENGTH, lambda: self._drum_box(place, letter, "rows")),
            Item(VELOCITY, lambda: self._drum_box(place, letter, "vel")),
        )
        self._menu(thing_menu(special=special, remove=lambda: self._change(
            lambda song: songedit.drum_removed(song, letter), f"drum {letter} removed")), pos)

    def _drum_box(self, place: int, letter: str, key: str) -> None:
        """Open a box over the label of a lane, holding one thing of its drum: enter saves what was typed.

        `key` is "note", "rows" or "vel". The rows and the velocity are numbers, and a box left empty takes
        the one the drum has of its own away. What `songedit.drum_set` refuses leaves the box open.
        """
        held = self._song["drums"][letter].get(key)
        what, help = {
            "note": ("note", f"the note drum {letter} plays, as c2 or f#4: enter takes it, escape leaves it"),
            "rows": ("length", f"the rows a hit of drum {letter} is held for at the most, or nothing for as long as "
                               "its text says: enter takes it, escape leaves it"),
            "vel": ("velocity", f"how hard drum {letter} is struck, from 0 to 1, or nothing for 1: enter takes it, "
                                "escape leaves it"),
        }[key]

        def entered(typed: str) -> object:
            value: object = typed.strip() or None
            if key != "note" and value is not None:
                try:
                    value = float(typed)
                    value = int(value) if value == int(value) else value
                except (ValueError, OverflowError):
                    value = typed
            return self._change(lambda song: songedit.drum_set(song, letter, key, value),
                                f"drum {letter}: {what} {typed.strip() or 'as its text says'}")

        text = "" if held is None else str(held) if key == "note" else plain(held)
        fits = (lambda char: char.isalnum() or char in "#-") if key == "note" else (DIGITS + ".").__contains__
        self._ask(self.steps.name_rect(place), text, fits, entered, help, 8)

    def _new_drum(self) -> None:
        """Open a box at the plus of the steps that asks for the letter of a new drum: enter adds the drum.

        Only a letter the song has no drum for goes in. The drum plays the note `songedit.DRUM_NOTE` of the
        instrument of the picked channel, and a song with no drum table yet gets one with it.
        """
        free = [letter for letter in songedit.LETTERS if letter not in self._song.get("drums", {})]
        inst = self._song["channels"][self.channel]["inst"]
        self._ask(self.steps.plus.rect, "", free.__contains__,
                  lambda letter: self._change(lambda song: songedit.drum_added(song, letter, inst),
                                              f"new drum {letter}"),
                  f"the letter of the new drum, one of {' '.join(free)}: enter adds it, escape leaves it", 1)

    def _ask_chord(self, bar: int, pos: Pos) -> None:
        """Open the menu of a bar of the pattern that is shown, at the mouse: the chord the bar is to be in.

        It has the chords of the project by name, then what clears the bar, and what opens the box a new chord
        is made in. The pattern is the one that was shown when the menu opened, with the bars it had then.
        """
        at = self._blocks[self.strip.picked].name, len(self.roll.chords.bars), bar
        chords = [Item(name, lambda name=name: self._chord(*at, name)) for name in self.session.model.project.chords]
        self._menu((*chords, *([None] if chords else []), Item(NO_CHORD, lambda: self._chord(*at, None)),
                    Item(NEW_CHORD, lambda: self._new_chord(*at, pos))), pos)

    def _chord(self, pattern: str, bars: int, bar: int, name: str | None) -> bool:
        """Save a chord of the project as the chord of a bar of a pattern of so many bars, or with None no chord.

        It is the song's chords for the pattern that are saved, as `chordview.with_chord` leaves them: one
        change to the file, and one step that can be taken back. The answer is whether the bar is so now. What
        `chordview` or the project refuses changes nothing, and the session says why. A bar that is so already
        is left alone, and is no step. The song is read again whatever came of it.
        """
        song_name = self._picked[1]
        project = self.session.model.project
        song = project.songs.get(song_name, {})
        written = song.get("chords", {}).get(pattern, [])
        try:
            made = chordview.with_chord(project.chords, written, bars, bar, name)
        except RowSynthError as error:
            self.session.say(str(error))
            return False
        saved = made == list(written)
        if not saved:
            what = "no chord" if name is None else f"chord {name}"
            saved = self.session.change(*chordview.saving(song_name, song, pattern, made),
                                        f"bar {bar + 1} of pattern {pattern}: {what}")
        self._look()
        return saved

    def _new_chord(self, pattern: str, bars: int, bar: int, pos: Pos) -> None:
        """Open the box a new chord is made in, at a place: OK adds the chord and sets it on a bar of a pattern."""
        self._popup = self.ui.open(ChordBox(pos, lambda name, made: self._add_chord(pattern, bars, bar, name, made)))

    def _add_chord(self, pattern: str, bars: int, bar: int, name: str, made: dict) -> bool:
        """Add a chord to the chords of the project under a name, and set it on a bar of a pattern, the two as one.

        The chord goes into the table, and the song's chords for the pattern are saved as `chordview.with_chord`
        leaves them with the table as it then is: one write to the file, and one step that can be taken back.
        The answer is whether that was done. A name `chordview.named` refuses, a bar the pattern does not
        have, and what the project refuses change nothing, and the session says why. The song is read again
        once the two are saved.
        """
        song_name = self._picked[1]
        project = self.session.model.project
        song = project.songs.get(song_name, {})
        written = song.get("chords", {}).get(pattern, [])
        try:
            # Only a table the file has can be given one more chord: a file with no chords gets the table whole.
            added = chordview.adding(project.to_dict().get("chords"), name, made)
            strings = chordview.with_chord({**project.chords, name: made}, written, bars, bar, name)
        except RowSynthError as error:
            self.session.say(str(error))
            return False
        saved = self.session.change_many([added, chordview.saving(song_name, song, pattern, strings)],
                                         f"new chord {name} on bar {bar + 1} of pattern {pattern}")
        self._look()
        return saved

    def _ask_chords(self) -> None:
        """Open the list of the chords of the project under the button CHORDS, with a plus after the last.

        A chord that is chosen opens its menu there, and the plus the box a new chord is made in.
        """
        x, y, _, h = self.chords.rect
        at = (x, y + h)

        def chosen(name: object) -> None:
            if name == chordview.ADD:
                self._popup = self.ui.open(ChordBox(at, self._table_chord))
            else:
                self._ask_chord_of(name, at)

        self._popup = self.ui.open(ChordList(self.session.model.project.chords, at, chosen))

    def _ask_chord_of(self, name: str, at: Pos) -> None:
        """Open the menu of a chord of the project at a place: the menu of a named thing, as `kit.thing_menu` has it.

        A chord is renamed in a box over the button CHORDS, and changed in the box a chord is made in, opened
        on it. It is not copied, so that item is dim. What `chordview` refuses is said, as a chord that a bar
        of a song names is when it is to be removed.
        """
        project = self.session.model.project
        self._menu(thing_menu(
            rename=lambda: self._ask(
                self.chords.rect, name, chordview.in_name,
                lambda new: new == name or self._chords(
                    lambda: chordview.chord_renamed(project.chords, project.songs, name, new),
                    f"chord {name} renamed to {new}"),
                f"the name of chord {name}: enter gives it the new one in every bar, escape leaves it"),
            special=(Item(CHANGE, lambda: self._open_chord(name, at)),),
            remove=lambda: self._chords(lambda: chordview.chord_removed(project.chords, project.songs, name),
                                        f"chord {name} removed")), at)

    def _open_chord(self, name: str, at: Pos) -> None:
        """Open the box a chord is made in on a chord of the project: OK saves the chord as it is left there.

        A chord that was given another name in the box is renamed as well, in every bar, and the two are one
        step. One that is left as it was is no step.
        """
        project = self.session.model.project
        held = project.chords[name]

        def done(new: str, made: dict) -> bool:
            if new == name and made == held:
                return True
            if new == name:
                return self._chords(lambda: chordview.chord_changed(project.chords, name, made),
                                    f"chord {name} changed")
            return self._chords(lambda: [*chordview.chord_renamed(project.chords, project.songs, name, new),
                                         (f"chords.{new}", made)], f"chord {name} changed to {new}")

        self._popup = self.ui.open(ChordBox(at, done, name, held))

    def _table_chord(self, name: str, made: dict) -> bool:
        """Add a chord to the chords of the project under a name, as one step, and say whether it was added."""
        table = self.session.model.project.to_dict().get("chords")
        return self._chords(lambda: [chordview.adding(table, name, made)], f"new chord {name}")

    def _chords(self, make: Callable[[], list], label: str) -> bool:
        """Save what a rule of `chordview` gives for the chord table, as one step, and show what the file then holds.

        The answer is whether it was saved. What the rule refuses, and what the project refuses, changes
        nothing, and the session says why.
        """
        try:
            changes = make()
        except RowSynthError as error:
            self.session.say(str(error))
            return False
        saved = self.session.change_many(changes, label)
        self._look()
        return saved

    def _menu(self, items: Sequence[Item | None], at: Pos) -> None:
        """Open a menu at a place, over everything else. Each item stands for what is done when it is chosen."""
        self._popup = self.ui.open(Menu(items, at, chosen=lambda act: act()))

    def _ask(self, rect: Rect, text: str, fits: Callable[[str], object], entered: Callable[[str], object],
             help: str, most: int = 32, mend: bool = False) -> None:
        """Open a box that is typed into over a rectangle, holding a text, of which `help` says what it is.

        Only the characters `fits` says yes to go in, and no more than `most`. On enter `entered` is given the
        text, and says whether it was taken. One that was not is in the box again, which is open as before: the
        session has said why by then. `mend` is for that box: none of its text is picked, and the caret stands
        after it, so that what is typed next is added to it.
        """
        def done(typed: str) -> None:
            if not entered(typed):
                self._ask(rect, typed, fits, entered, help, most, True)

        box = self._popup = ask(self.ui, rect, text, done, allowed=fits, most=most, help=help)
        if mend:
            box.anchor = box.caret

    def _rename(self, place: int) -> None:
        """Open a box over the name of the block at a place: enter gives its pattern that name everywhere."""
        old = self._blocks[place].name
        self._ask(self.strip.name_rect(place), old, in_name,
                  lambda new: new == old or self._change(lambda song: songedit.renamed(song, old, new),
                                                         f"pattern {old} renamed to {new}"),
                  f"the name of pattern {old}: enter gives it the new one, escape leaves it")

    def _resize(self, place: int) -> None:
        """Open a box over the name of the block at a place, holding the bars of its pattern: enter makes it so long."""
        name = self._blocks[place].name

        def entered(typed: str) -> object:
            bars = int(typed or 0)
            return self._change(lambda song: songedit.resized(song, name, bars),
                                f"pattern {name} made {bars_text(bars)} long")

        self._ask(self.strip.name_rect(place), str(max(1, math.ceil(self._bars(name)))), DIGITS.__contains__,
                  entered, f"how many bars pattern {name} is long: enter cuts it or fills it, escape leaves it",
                  MOST_DIGITS)

    def _rename_channel(self, old: str) -> None:
        """Open a box over the name of a channel in the list: enter gives the channel that name everywhere.

        It is what the menu of the channel opens, and a double press on the channel.
        """
        self._ask(self.list.name_rect(old) or self.list.rect, old, in_name,
                  lambda new: new == old or self._change(lambda song: songedit.channel_renamed(song, old, new),
                                                         f"channel {old} renamed to {new}",
                                                         channel=new if self.channel == old else None),
                  f"the name of channel {old}: enter gives it the new one, escape leaves it")

    def _tempo(self, index: int) -> None:
        """Save the song at the tempo the stepper was put on, as one step."""
        ticks = self._ticks[index]
        self._change(lambda song: songedit.with_tempo(song, ticks), f"tempo {self.tempo.options[index]}")

    def _regrid(self, index: int) -> None:
        """Save the song with the rows to a beat the stepper was put on, as `songedit.gridded` leaves it: one step."""
        rows = self._grids[index]
        self._change(lambda song: songedit.gridded(song, rows), f"{rows} rows to a beat")

    def _shift(self, by: int) -> None:
        """Save the song with every tuned note so many semitones higher, as one step, and count the move."""
        name = self._picked[1]
        semis, was = self._semis.get(name, 0), _state(self.session.model.project.songs.get(name, {}))
        made = self._change(lambda song: songedit.transposed(song, by),
                            f"a semitone {'up' if by > 0 else 'down'}")
        if made is not None and _state(made) != was:
            self._states[name, was], self._states[name, _state(made)] = semis, semis + by
            self._count()

    def _channels(self) -> tuple[Channel, ...]:
        """The channels of the song as the pattern at the picked place has them. A song with no order has none."""
        return self._patterns[self._blocks[self.strip.picked].name].channels if self._blocks else ()

    def _pick(self, channel: str) -> None:
        """Pick a channel of the song by its name, and show it."""
        self.channel = channel
        self._view()

    def _choose(self, channel: str) -> None:
        """Pick the channel whose name was pressed in the mixer, and show its notes in place of the mixer."""
        self.choice.picked = self.choice.options.index(NOTES)
        self._pick(channel)

    def _list(self) -> None:
        """Give the list its rows: the channels of the pattern at the picked place, dim where one is not heard."""
        self._heard = self.session.model.heard
        self.list.rows = channel_rows(self._channels(), self._heard)
        self.list.picked = self.channel
        self.list.lay()

    def _view(self) -> None:
        """Show the picked channel of the pattern at the picked place: in the list, the bar, and the roll or the steps.

        A tuned channel is the roll's one channel. A drum channel is lit among all the drums of the pattern, which
        are given the drum table of the song, so that their hits can be changed. The bar is shown the channel
        in its colour, and no channel where the pattern has none of that name. The mixer is shown the song, with
        the name of the picked channel lit.
        """
        channels = self._channels()
        self._list()
        place = next((place for place, held in enumerate(channels) if held.name == self.channel), None)
        self._drums = None if place is None else channels[place].drums
        self.bar.show(self._picked[1], None if place is None else self.channel, colour_of(place or 0))
        self.mixer.show(self._picked[1], self.channel)
        self.layers.show(self._picked[1])
        self.session.shown = self._blocks[self.strip.picked].name if self._blocks else None
        if place is not None:
            pattern = self._patterns[self._blocks[self.strip.picked].name]
            if self._drums:
                drums = self._song.get("drums") or {}
                self._lanes = lanes(self._patterns.values(), songview.drum_rows(self._song), self.channel)
                self.steps.show(pattern, self._lanes, self._bar_rows, self._bar_rows // 4, drums)
                self.steps.picked = self.channel
                self.steps.lane.colour = colour_of(place)
                self.steps.plus.enabled = any(letter not in drums for letter in songedit.LETTERS)
                self.steps.plus.help = PLUS_HELP if self.steps.plus.enabled else NO_LETTER
            else:
                self.roll.show(pattern, self.channel, self._bar_rows, colour_of(place),
                               chordview.read(self._project.chords, self._song, pattern.name, pattern.rows))
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
        if at is not None and self.bar.follow.is_on and at[0] != self.strip.picked and not self._busy():
            self.strip.picked = at[0]
            self._view()
        shown = at is not None and self._blocks[at[0]].name == self._blocks[self.strip.picked].name
        self.roll.playhead = self.steps.playhead = at[1] if shown else None
        # The row of the song is the row its place starts at and the row of the pattern there, put together again.
        self.strip.playhead = None if at is None else self._blocks[at[0]].row + at[1]

    def _seen(self) -> None:
        """Have each widget seen or not: by whether the panel is on the window, and by what there is to show.

        The heading with the choice, the strip and the settings bar are seen with any song, the song's own
        settings as far as the bar has room for them. Under the bar it is the notes, the mixer or the layers
        that are seen, as is chosen: the widgets of the others are off the window, and take nothing from the
        mouse. The roll is seen with its four parts.
        """
        song = self.shown and self._song is not None
        notes = song and not self.mixing and not self.layering
        for widget in (self.heading, self.choice, self.chords, self.strip, *self.strip.parts, self.more):
            widget.visible = song
        self.more.enabled = bool(self._blocks)
        self.bar.seen(song)
        for widget in self._settings:
            widget.visible = song and widget.rect[2] > 0
        self.list.seen(notes)
        for widget in (self.roll, *self.roll.parts):
            widget.visible = notes and self._drums is False
        self.steps.seen(notes and self._drums is True)
        self.mixer.seen(song and self.mixing)
        self.layers.seen(song and self.layering)
        self.fault.visible = self.shown and bool(self.fault.text)
