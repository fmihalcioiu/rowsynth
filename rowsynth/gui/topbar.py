"""The top bar: the FILE button, the file's name, undo and redo, play, stop and loop, the time, a meter, findings."""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from . import theme
from .dials import Meter
from .font import Font
from .layout import Rect
from .panels import Panel, within
from .theme import Colour
from .ui import Button, Label, Pos, Toggle, Ui, Widget

if TYPE_CHECKING:
    from .canvas import Canvas
    from .session import Session

#: The widest the time is written, which its box has room for: both of its moments past ten minutes.
WIDEST = "00:00.0 / 00:00.0"
#: What stands in place of the count of findings while the checks run.
RUNNING = "..."
#: What the findings say of themselves, with the mouse on them, where the checks gave none.
REFUSED = "the checks could not run on this project"
#: What the button at the left end of the bar reads as: the one that opens the menu of what is done with files.
FILE = "FILE"
#: The buttons of the bar after the name, in the groups they stand in: those of a group stand side by side.
GROUPS = (("undo", "redo"), ("play", "stop", "loop"))


def offer(word: str, can: bool, label: str) -> str:
    """The line of help of undo or of redo: the word and what its step is called, as in "undo: zap: echo off".

    `can` is whether there is such a step. With none it says so, as in "nothing to undo".
    """
    return f"{word}: {label}" if can else f"nothing to {word}"


def moment(seconds: float) -> str:
    """A time as minutes, seconds and tenths of a second, as in "0:03.2". The tenth is rounded down."""
    tenths = max(0, int(seconds * 10))
    return f"{tenths // 600}:{tenths // 10 % 60:02d}.{tenths % 10}"


def timing(position: float, seconds: float) -> str:
    """Where the playing is and how long the thing is, as in "0:03.2 / 0:12.8"."""
    return f"{moment(position)} / {moment(seconds)}"


def verdict(count: int | None, refused: bool = False) -> tuple[str, str, Colour]:
    """How a count of findings is shown: the name of an icon, a text, and the colour of the two.

    None is for checks that still run: dim dots, and "" for no icon. 0 is a tick in the colour of what is as it
    should be. Anything more is the warning sign in the colour of what is wrong. `refused` is for checks that
    could not run, whatever the count: the warning sign in that colour, and "" for no number.
    """
    if refused:
        return "warn", "", theme.FAULT
    if count is None:
        return "", RUNNING, theme.DIM
    if count == 0:
        return "tick", "0", theme.GOOD
    return "warn", str(count), theme.FAULT


def arrange(rect: Rect, name: str) -> dict[str, Rect]:
    """Where the things of the bar go in its rectangle, each by what it is called.

    "found" stands at the right end. From the left stand "file", as wide as its word needs, then "name", then
    "undo" and "redo" side by side, then "play", "stop" and "loop" side by side, then "time" and then "meter".
    A bar too narrow for them all has no meter, and one still too narrow has no time either: what has gone has
    no width. The buttons all stay. The name has the width of its text, or what the others leave it in a bar
    too narrow even for that. Every thing is cut to the bar.
    """
    x, y, w, h = rect
    wide, high = theme.ICON_BUTTON
    top = y + (h - high) // 2
    found = x + w - theme.BAR_MARGIN - theme.FOUND_W
    left = x + theme.BAR_MARGIN
    file = (left, top, Font.width(FILE) + 2 * theme.PAD, high)
    left += file[2] + theme.GROUP_GAP
    # What stands after the buttons, each with its width. The last of them is the first to go.
    after = (("time", Font.width(WIDEST) + 2 * theme.PAD), ("meter", theme.BAR_METER_W))
    # Each group of buttons, with the gap that is before it.
    buttons = sum(theme.GROUP_GAP + len(group) * (wide + theme.BUTTON_GAP) - theme.BUTTON_GAP for group in GROUPS)
    room = found - left - buttons - theme.GROUP_GAP
    stay = len(after)
    while stay and Font.width(name) + sum(width + theme.GROUP_GAP for _, width in after[:stay]) > room:
        stay -= 1
    room -= sum(width + theme.GROUP_GAP for _, width in after[:stay])
    named = max(0, min(Font.width(name), room))
    placed = {"file": file, "name": (left, y, named, h), "found": (found, y, theme.FOUND_W, h)}
    left += named
    for group in GROUPS:
        left += theme.GROUP_GAP
        for button in group:
            placed[button] = (left, top, wide, high)
            left += wide + theme.BUTTON_GAP
        left -= theme.BUTTON_GAP
    left += theme.GROUP_GAP
    for index, (thing, width) in enumerate(after):
        placed[thing] = (left, top, width if index < stay else 0, high)
        left += width + theme.GROUP_GAP
    return {thing: within(place, rect) for thing, place in placed.items()}


class TopBar(Panel):
    """The bar along the top: what is open, the steps back and again, what plays it, and what the checks found.

    `file` is the button at the left end. A press on it calls `asked` with the place just under its left
    corner, for the owner to open the menu of what is done with files there: the bar opens none itself, and
    with no owner to tell the button does nothing.

    `undo` and `redo` are enabled while the session has a step for them, and the help of each names that step.
    `found` is how the findings are shown now, as `verdict` gives it. `sign` lies over them, and draws nothing:
    it has the line of help that says so when the checks could not run.
    """

    def __init__(self, session: Session, ui: Ui, asked: Callable[[Pos], object] | None = None) -> None:
        super().__init__(session, ui)
        nowhere = (0, 0, 0, 0)
        self.asked = asked
        self.file = ui.add(Button(nowhere, FILE, action=self._file,
                                  help="a new project, another file, a copy of this one, the export of files, "
                                       "or the first screen"))
        self.name = ui.add(Label(nowhere, session.model.path.name, theme.TEXT))
        self.undo = ui.add(Button(nowhere, icon="undo", action=session.undo))
        self.redo = ui.add(Button(nowhere, icon="redo", action=session.redo))
        # Play is no switch of its own: it is lit for as long as the session says something plays.
        self.play = ui.add(Toggle(nowhere, icon="play", changed=lambda on: session.play(), colour=theme.GOOD,
                                  help="play what is picked from its start: the space bar plays and stops"))
        self.stop = ui.add(Button(nowhere, icon="stop", action=session.stop, help="stop what plays"))
        self.loop = ui.add(Toggle(nowhere, icon="loop", on=session.loop, changed=self._loop,
                                  help="a song goes round until it is stopped"))
        self.time = ui.add(Label(nowhere, timing(0, 0), theme.TEXT, "centre"))
        # A bar for the left side and under it one for the right.
        self.meter = ui.add(Meter(nowhere))
        self.meter.levels = (0.0, 0.0)
        self.sign = ui.add(Widget(nowhere))
        self.found = verdict(None)
        self._places = arrange(nowhere, self.name.text)

    def _file(self) -> None:
        """Tell the owner where the menu of the FILE button opens: just under the button, at its left."""
        if self.asked is not None:
            x, y, _, h = self.file.rect
            self.asked((x, y + h))

    def _loop(self, on: bool) -> None:
        """Tell the session whether a song goes round."""
        self.session.loop = on

    def place(self, rect: Rect) -> None:
        """Take the bar's rectangle, and lay the widgets out in it as `arrange` has them."""
        super().place(rect)
        self._places = arrange(rect, self.name.text)
        for thing in ("file", "name", "undo", "redo", "play", "stop", "loop", "time", "meter"):
            getattr(self, thing).rect = self._places[thing]
        self.sign.rect = self._places["found"]
        # A meter that has gone has no width, and is not drawn.
        self.meter.visible = self._places["meter"][2] > 0

    def update(self, passed: float = 0.0) -> None:
        """Show what the session has now: its steps, what plays, the time, how loud it is, and the findings.

        Undo and redo are enabled by whether there is a step for them, and the help of each says which. `passed`
        is the seconds since the frame before: the kept lines of the meter fall for that long.
        """
        session = self.session
        self.undo.enabled, self.undo.help = session.can_undo, offer("undo", session.can_undo, session.undo_label)
        self.redo.enabled, self.redo.help = session.can_redo, offer("redo", session.can_redo, session.redo_label)
        self.play.is_on = session.playing
        self.loop.is_on = session.loop
        self.time.text = timing(session.position(), session.seconds)
        self.meter.levels = session.level()
        self.meter.fall(passed)
        refused = session.findings_refused
        self.found = verdict(session.finding_count(), refused)
        self.sign.help = REFUSED if refused else ""

    def draw(self, canvas: Canvas) -> None:
        """Draw the sunk box the time stands in, and the findings: the count at the right end, its icon before it.

        With no count the icon stands at the right end.
        """
        canvas.box(self._places["time"], theme.BACKGROUND)
        canvas.frame(self._places["time"], theme.LINE)
        icon, text, colour = self.found
        x, y, w, h = self._places["found"]
        left = x + w - Font.width(text)
        with canvas.clip(self._places["found"]):
            canvas.text_in(text, (left, y, Font.width(text), h), colour)
            if icon:
                wide, high = canvas.icon_size(icon)
                canvas.icon(icon, left - (theme.GAP if text else 0) - wide, y + (h - high) // 2, colour)
