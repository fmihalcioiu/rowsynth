"""The window's own title row, and the top bar: FILE, undo and redo, play, stop, loop and its two switches, the time."""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from . import theme
from .dials import Meter
from .font import Font
from .kit import told
from .layout import Rect
from .panels import Panel, within
from .theme import Colour
from .ui import NOWHERE, Button, Event, Label, Pos, Toggle, Ui, Widget, inside

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
#: What the two switches beside loop read as: one pattern played alone, and play with no echoes.
PATTERN = "PATTERN"
DRY = "DRY"
#: The buttons of the bar after FILE, in the groups they stand in: those of a group stand side by side.
GROUPS = (("undo", "redo"), ("play", "stop", "loop", "pattern", "dry"))
#: How wide the buttons are that show a word: each as wide as its word needs. Every other one shows an icon.
WORDS = {"pattern": Font.width(PATTERN) + 2 * theme.PAD, "dry": Font.width(DRY) + 2 * theme.PAD}

#: The program's name, as the title row has it, and the side of the small square before it.
PROGRAM = "rowsynth"
MARK = 5
#: The three buttons at the right end of the title row, from the left, and how wide each is. Each is as high as
#: the row. The sign on a button is drawn in a square of this side.
MINIMISE, MAXIMISE, CLOSE = "minimise", "maximise", "close"
BUTTONS = (MINIMISE, MAXIMISE, CLOSE)
BUTTON_W = 20
SIGN = 7
#: What the middle button does to a window that fills the desktop.
RESTORE = "restore"
#: What the title row says of itself where the mouse is on none of its buttons, and what each button says.
TITLE_HELP = told("drag to move the window, twice to have it fill the screen or to put it back")
BUTTON_HELP = {MINIMISE: told("take the window off the screen, down to the task bar"),
               MAXIMISE: told("have the window fill the screen"),
               RESTORE: told("put the window back where it stood before it filled the screen"),
               CLOSE: told("close the window")}


def act(button: str, big: bool) -> str:
    """What a button of the title row does to the window: `MINIMISE`, `MAXIMISE`, `RESTORE` or `CLOSE`.

    `big` is whether the window fills the desktop: the middle button then puts it back, and else has it fill
    the desktop. The other two do the one thing they are called.
    """
    return RESTORE if button == MAXIMISE and big else button


def title_parts(rect: Rect, name: str = "") -> dict[str, Rect]:
    """Where the things of the title row go in its rectangle, each by what it is called.

    From the left stand "mark", the small square, then "program", as wide as the program's name, then "name",
    the name of the open file, as wide as its text or as what the buttons leave it. At the right end stand
    the three of `BUTTONS`, the last of them in the corner. Every thing is cut to the row.
    """
    x, y, w, h = rect
    first = x + w - len(BUTTONS) * BUTTON_W
    left = x + theme.BAR_MARGIN
    placed = {"mark": (left, y + (h - MARK) // 2, MARK, MARK),
              "program": (left + MARK + theme.GAP + 1, y, Font.width(PROGRAM), h)}
    left += MARK + theme.GAP + 1 + Font.width(PROGRAM) + theme.GROUP_GAP
    placed["name"] = (left, y, max(0, min(Font.width(name), first - theme.GROUP_GAP - left)), h)
    placed.update((button, (first + place * BUTTON_W, y, BUTTON_W, h)) for place, button in enumerate(BUTTONS))
    return {thing: within(place, rect) for thing, place in placed.items()}


class Title(Widget):
    """The window's own title row, which stands in for the title bar of the system over a window with no frame.

    It shows a small square, the program's name and `name`, the name of the open file. At its right end are
    the three of `BUTTONS`. A press of the left button on one, let go on it, calls `acted` with what `act`
    says the button does. A press on an empty place of the row calls `grabbed`, for the owner to move the
    window with the mouse until the button is let go, and a double press there calls `acted` as the middle
    button does. `big` is whether the window fills the desktop, which the owner sets.
    """

    def __init__(self, rect: Rect, name: str = "", acted: Callable[[str], object] | None = None,
                 grabbed: Callable[[], object] | None = None) -> None:
        super().__init__(rect, TITLE_HELP)
        self.name = name
        self.acted = acted
        self.grabbed = grabbed
        self.big = False
        self._mouse: Pos = NOWHERE
        # The button the left press went down on whose release has not come.
        self._down: str | None = None

    @property
    def help(self) -> str:
        """The line of the button the mouse is on, and anywhere else on the row the row's own."""
        button = self.at(self._mouse)
        return self._help if button is None else BUTTON_HELP[act(button, self.big)]

    @help.setter
    def help(self, line: str) -> None:
        self._help = line

    def at(self, pos: Pos) -> str | None:
        """The button of the row a position is on, or None when it is on none of the three."""
        places = title_parts(self.rect)
        return next((button for button in BUTTONS if inside(places[button], pos)), None)

    def _act(self, button: str) -> None:
        """Tell the owner what a button does to the window as it is now."""
        if self.acted is not None:
            self.acted(act(button, self.big))

    def on(self, event: Event, ui: Ui) -> bool:
        """Use a press of the left button: on a button it is done on the release, and off them it grabs the row."""
        if event.kind in ("key", "text", "wheel"):
            return False
        self._mouse = event.pos
        over = self.at(event.pos)
        if event.kind == "down" and event.button == 1:
            self._down = over
            if over is None and event.double:
                self._act(MAXIMISE)
            elif over is None and self.grabbed is not None:
                self.grabbed()
            return True
        if event.kind == "up" and event.button == 1:
            down, self._down = self._down, None
            if down is not None and down == over:
                self._act(down)
        return False

    def draw(self, canvas: Canvas) -> None:
        """Draw the row on a dark ground with its line under it: the square, the two names and the three buttons.

        The button under the mouse is lit, and the one that closes is then in the colour of what is wrong.
        """
        places = title_parts(self.rect, self.name)
        x, y, w, h = self.rect
        with canvas.clip(self.rect):
            canvas.box(self.rect, theme.BACKGROUND)
            canvas.box(places["mark"], theme.ACTIVE)
            canvas.text_in(PROGRAM, places["program"], theme.TEXT)
            canvas.text_in(self.name, places["name"], theme.DIM)
            over = self.at(self._mouse) if self.hot else None
            for button in BUTTONS:
                if button == over:
                    canvas.box(places[button], theme.FAULT if button == CLOSE else theme.LIT)
                self._sign(canvas, places[button], act(button, self.big))
            canvas.hline(x, y + h - 1, w, theme.LINE)

    @staticmethod
    def _sign(canvas: Canvas, rect: Rect, what: str) -> None:
        """Draw the sign of what a button does in the middle of the button: a line, a box, two boxes or a cross."""
        x, y = rect[0] + (rect[2] - SIGN) // 2, rect[1] + (rect[3] - SIGN) // 2
        if what == MINIMISE:
            canvas.hline(x, y + SIGN - 2, SIGN, theme.TEXT)
        elif what == MAXIMISE:
            canvas.frame((x, y, SIGN, SIGN), theme.TEXT)
            canvas.hline(x, y + 1, SIGN, theme.TEXT)
        elif what == RESTORE:
            canvas.frame((x + 2, y, SIGN - 2, SIGN - 2), theme.TEXT)
            canvas.box((x, y + 2, SIGN - 2, SIGN - 2), theme.BACKGROUND)
            canvas.frame((x, y + 2, SIGN - 2, SIGN - 2), theme.TEXT)
        else:
            for step in range(SIGN):
                canvas.box((x + step, y + step, 1, 1), theme.TEXT)
                canvas.box((x + SIGN - 1 - step, y + step, 1, 1), theme.TEXT)


def offer(word: str, can: bool, label: str) -> str:
    """The line of help of undo or of redo: the word and what its step is called, as in "press: undo zap: echo off".

    `can` is whether there is such a step. With none it says so, as in "nothing to undo".
    """
    return told(f"{word} {label}") if can else f"nothing to {word}"


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


def arrange(rect: Rect) -> dict[str, Rect]:
    """Where the things of the bar go in its rectangle, each by what it is called.

    "found" stands at the right end. From the left stand "file", as wide as its word needs, then "undo" and
    "redo" side by side, then "play", "stop", "loop" and the two switches "pattern" and "dry" side by side,
    each of those two as wide as its word needs, then "time" and then "meter". A bar too narrow for them all
    has no meter, and one still too narrow has no time either: what has gone has no width. The buttons all
    stay. Every thing is cut to the bar.
    """
    x, y, w, h = rect
    wide, high = theme.ICON_BUTTON
    top = y + (h - high) // 2
    found = x + w - theme.BAR_MARGIN - theme.FOUND_W
    left = x + theme.BAR_MARGIN
    placed = {"file": (left, top, Font.width(FILE) + 2 * theme.PAD, high), "found": (found, y, theme.FOUND_W, h)}
    left += placed["file"][2]
    for group in GROUPS:
        left += theme.GROUP_GAP
        for button in group:
            placed[button] = (left, top, WORDS.get(button, wide), high)
            left += placed[button][2] + theme.BUTTON_GAP
        left -= theme.BUTTON_GAP
    # What stands after the buttons, each with its width. The last of them is the first to go.
    after = (("time", Font.width(WIDEST) + 2 * theme.PAD), ("meter", theme.BAR_METER_W))
    stay = len(after)
    while stay and left + sum(theme.GROUP_GAP + width for _, width in after[:stay]) + theme.GROUP_GAP > found:
        stay -= 1
    for index, (thing, width) in enumerate(after):
        left += theme.GROUP_GAP
        placed[thing] = (left, top, width if index < stay else 0, high)
        left += width
    return {thing: within(place, rect) for thing, place in placed.items()}


class TopBar(Panel):
    """The bar along the top: the steps back and again, what plays the picked thing, and what the checks found.

    `file` is the button at the left end. A press on it calls `asked` with the place just under its left
    corner, for the owner to open the menu of what is done with files there: the bar opens none itself, and
    with no owner to tell the button does nothing.

    `undo` and `redo` are enabled while the session has a step for them, and the help of each names that step.
    `pattern` and `dry` are the two switches beside `loop`: on, the first has the one pattern the song view
    shows play alone, and the second has what plays play with no echoes. `found` is how the findings are shown
    now, as `verdict` gives it. `sign` lies over them, and draws nothing: it has the line of help that says so
    when the checks could not run.
    """

    def __init__(self, session: Session, ui: Ui, asked: Callable[[Pos], object] | None = None) -> None:
        super().__init__(session, ui)
        nowhere = (0, 0, 0, 0)
        self.asked = asked
        self.file = ui.add(Button(nowhere, FILE, action=self._file,
                                  help=told("open the menu of files: a new project, another file, a copy, the "
                                            "export, a WAV file to measure, the first screen")))
        self.undo = ui.add(Button(nowhere, icon="undo", action=session.undo))
        self.redo = ui.add(Button(nowhere, icon="redo", action=session.redo))
        # Play is no switch of its own: it is lit for as long as the session says something plays.
        self.play = ui.add(Toggle(nowhere, icon="play", changed=lambda on: session.play(), colour=theme.GOOD,
                                  help=told("play what is picked from its start, as the space bar does")))
        self.stop = ui.add(Button(nowhere, icon="stop", action=session.stop, help=told("stop what plays")))
        self.loop = ui.add(Toggle(nowhere, icon="loop", on=session.loop, changed=self._loop,
                                  help=told("have a song go round until it is stopped, or play it once")))
        self.pattern = ui.add(Toggle(nowhere, PATTERN, on=session.pattern, changed=self._pattern,
                                     help=told("play and loop the one pattern the song view shows, or all the song")))
        self.dry = ui.add(Toggle(nowhere, DRY, on=session.dry, changed=self._dry,
                                 help=told("play with no echoes, or with them")))
        self.time = ui.add(Label(nowhere, timing(0, 0), theme.TEXT, "centre"))
        # A bar for the left side and under it one for the right.
        self.meter = ui.add(Meter(nowhere))
        self.meter.levels = (0.0, 0.0)
        self.sign = ui.add(Widget(nowhere))
        self.found = verdict(None)
        self._places = arrange(nowhere)

    def _file(self) -> None:
        """Tell the owner where the menu of the FILE button opens: just under the button, at its left."""
        if self.asked is not None:
            x, y, _, h = self.file.rect
            self.asked((x, y + h))

    def _loop(self, on: bool) -> None:
        """Tell the session whether a song goes round."""
        self.session.loop = on

    def _pattern(self, on: bool) -> None:
        """Tell the session whether the one pattern the song view shows plays alone."""
        self.session.pattern = on

    def _dry(self, on: bool) -> None:
        """Tell the session whether what plays has no echoes."""
        self.session.dry = on

    def place(self, rect: Rect) -> None:
        """Take the bar's rectangle, and lay the widgets out in it as `arrange` has them."""
        super().place(rect)
        self._places = arrange(rect)
        for thing in ("file", "undo", "redo", "play", "stop", "loop", "pattern", "dry", "time", "meter"):
            getattr(self, thing).rect = self._places[thing]
        self.sign.rect = self._places["found"]
        # A meter that has gone has no width, and is not drawn.
        self.meter.visible = self._places["meter"][2] > 0

    def update(self, passed: float = 0.0) -> None:
        """Show what the session has now: its steps, what plays and how, the time, how loud it is, the findings.

        Undo and redo are enabled by whether there is a step for them, and the help of each says which. `passed`
        is the seconds since the frame before: the kept lines of the meter fall for that long.
        """
        session = self.session
        self.undo.enabled, self.undo.help = session.can_undo, offer("undo", session.can_undo, session.undo_label)
        self.redo.enabled, self.redo.help = session.can_redo, offer("redo", session.can_redo, session.redo_label)
        self.play.is_on = session.playing
        self.loop.is_on = session.loop
        self.pattern.is_on = session.pattern
        self.dry.is_on = session.dry
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
