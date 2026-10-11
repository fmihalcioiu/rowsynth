"""The top bar: the file's name, play, stop and loop, where the playing is and how loud, and the count of findings."""
from __future__ import annotations

from typing import TYPE_CHECKING

from . import theme
from .dials import Meter
from .font import Font
from .layout import Rect
from .panels import Panel, within
from .theme import Colour
from .ui import Button, Label, Toggle, Ui

if TYPE_CHECKING:
    from .canvas import Canvas
    from .session import Session

#: The widest the time is written, which its box has room for: both of its moments past ten minutes.
WIDEST = "00:00.0 / 00:00.0"
#: What stands in place of the count of findings while the checks run.
RUNNING = "..."


def moment(seconds: float) -> str:
    """A time as minutes, seconds and tenths of a second, as in "0:03.2". The tenth is rounded down."""
    tenths = max(0, int(seconds * 10))
    return f"{tenths // 600}:{tenths // 10 % 60:02d}.{tenths % 10}"


def timing(position: float, seconds: float) -> str:
    """Where the playing is and how long the thing is, as in "0:03.2 / 0:12.8"."""
    return f"{moment(position)} / {moment(seconds)}"


def verdict(count: int | None) -> tuple[str, str, Colour]:
    """How a count of findings is shown: the name of an icon, a text, and the colour of the two.

    None is for checks that still run: dim dots, and "" for no icon. 0 is a tick in the colour of what is as it
    should be. Anything more is the warning sign in the colour of what is wrong.
    """
    if count is None:
        return "", RUNNING, theme.DIM
    if count == 0:
        return "tick", "0", theme.GOOD
    return "warn", str(count), theme.FAULT


def arrange(rect: Rect, name: str) -> dict[str, Rect]:
    """Where the things of the bar go in its rectangle, each by what it is called.

    "found" stands at the right end. From the left stand "name", then "play", "stop" and "loop" side by side,
    then "time" and then "meter". A bar too narrow for them all has no meter, and one still too narrow has no
    time either: what has gone has no width. The name has the width of its text, or what the others leave it
    in a bar too narrow even for that. Every thing is cut to the bar.
    """
    x, y, w, h = rect
    wide, high = theme.ICON_BUTTON
    top = y + (h - high) // 2
    found = x + w - theme.BAR_MARGIN - theme.FOUND_W
    left = x + theme.BAR_MARGIN
    # What stands after the buttons, each with its width. The last of them is the first to go.
    after = (("time", Font.width(WIDEST) + 2 * theme.PAD), ("meter", theme.BAR_METER_W))
    room = found - left - 3 * wide - 2 * theme.BUTTON_GAP - 2 * theme.GROUP_GAP
    stay = len(after)
    while stay and Font.width(name) + sum(width + theme.GROUP_GAP for _, width in after[:stay]) > room:
        stay -= 1
    room -= sum(width + theme.GROUP_GAP for _, width in after[:stay])
    named = max(0, min(Font.width(name), room))
    placed = {"name": (left, y, named, h), "found": (found, y, theme.FOUND_W, h)}
    left += named + theme.GROUP_GAP
    for button in ("play", "stop", "loop"):
        placed[button] = (left, top, wide, high)
        left += wide + theme.BUTTON_GAP
    left += theme.GROUP_GAP - theme.BUTTON_GAP
    for index, (thing, width) in enumerate(after):
        placed[thing] = (left, top, width if index < stay else 0, high)
        left += width + theme.GROUP_GAP
    return {thing: within(place, rect) for thing, place in placed.items()}


class TopBar(Panel):
    """The bar along the top: what is open, what plays it, how far and how loud, and what the checks found.

    `found` is how the findings are shown now, as `verdict` gives it.
    """

    def __init__(self, session: Session, ui: Ui) -> None:
        super().__init__(session, ui)
        nowhere = (0, 0, 0, 0)
        self.name = ui.add(Label(nowhere, session.model.path.name, theme.TEXT))
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
        self.found = verdict(None)
        self._places = arrange(nowhere, self.name.text)

    def _loop(self, on: bool) -> None:
        """Tell the session whether a song goes round."""
        self.session.loop = on

    def place(self, rect: Rect) -> None:
        """Take the bar's rectangle, and lay the widgets out in it as `arrange` has them."""
        super().place(rect)
        self._places = arrange(rect, self.name.text)
        for thing in ("name", "play", "stop", "loop", "time", "meter"):
            getattr(self, thing).rect = self._places[thing]
        # A meter that has gone has no width, and is not drawn.
        self.meter.visible = self._places["meter"][2] > 0

    def update(self, passed: float = 0.0) -> None:
        """Show what plays, whether a song goes round, the time, how loud it is, and the findings, as they are now.

        `passed` is the seconds since the frame before: the kept lines of the meter fall for that long.
        """
        session = self.session
        self.play.is_on = session.playing
        self.loop.is_on = session.loop
        self.time.text = timing(session.position(), session.seconds)
        self.meter.levels = session.level()
        self.meter.fall(passed)
        self.found = verdict(session.finding_count())

    def draw(self, canvas: Canvas) -> None:
        """Draw the sunk box the time stands in, and the findings: the count at the right end, its icon before it."""
        canvas.box(self._places["time"], theme.BACKGROUND)
        canvas.frame(self._places["time"], theme.LINE)
        icon, text, colour = self.found
        x, y, w, h = self._places["found"]
        left = x + w - Font.width(text)
        with canvas.clip(self._places["found"]):
            canvas.text_in(text, (left, y, Font.width(text), h), colour)
            if icon:
                wide, high = canvas.icon_size(icon)
                canvas.icon(icon, left - theme.GAP - wide, y + (h - high) // 2, colour)
