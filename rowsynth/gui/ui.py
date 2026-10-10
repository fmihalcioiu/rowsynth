"""The kit: what the user did as plain events, the widgets that take them and draw, and the Ui that holds them."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..errors import RowSynthError
from . import theme
from .font import Font, fit
from .layout import Rect
from .theme import Colour

if TYPE_CHECKING:
    import pygame

    from .canvas import Canvas

#: What an event can be: a mouse button going down or up, the mouse moving, a step of the wheel, a key going down.
KINDS = ("down", "up", "move", "wheel", "key")

#: Where the mouse is taken to be before it has moved: on no widget.
NOWHERE = (-1, -1)

Pos = tuple[int, int]


@dataclass(frozen=True)
class Event:
    """One thing the user did, which widgets are handed in place of a pygame event.

    `pos` is where the mouse is, in art pixels. `button` is 1 for the left, 2 for the middle and 3 for the right.
    `wheel` is in steps, and up is positive. `key` is pygame's name of the key, such as "space" or "z". `mods` holds
    those of "ctrl", "shift" and "alt" that are held.
    """

    kind: str
    pos: Pos = (0, 0)
    button: int = 0
    wheel: int = 0
    key: str = ""
    mods: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise RowSynthError(f"an event is one of {', '.join(KINDS)}, not {self.kind!r}")


def translate(event: pygame.event.Event, to_art: Callable[[Pos], Pos]) -> Event | None:
    """A pygame event as an `Event`, its position put through `to_art`. None for one the GUI has no use for."""
    import pygame

    names = (("ctrl", pygame.KMOD_CTRL), ("shift", pygame.KMOD_SHIFT), ("alt", pygame.KMOD_ALT))
    if event.type == pygame.KEYDOWN:
        return Event("key", key=pygame.key.name(event.key),
                     mods=frozenset(name for name, bit in names if event.mod & bit))
    kind = {pygame.MOUSEBUTTONDOWN: "down", pygame.MOUSEBUTTONUP: "up", pygame.MOUSEMOTION: "move",
            pygame.MOUSEWHEEL: "wheel"}.get(event.type)
    if kind is None:
        return None
    held = pygame.key.get_mods()
    mods = frozenset(name for name, bit in names if held & bit)
    if kind == "wheel":
        # A step of the wheel does not say where the mouse is.
        return Event(kind, to_art(pygame.mouse.get_pos()), wheel=event.y, mods=mods)
    if kind == "move":
        return Event(kind, to_art(event.pos), mods=mods)
    if event.button not in (1, 2, 3):
        # pygame also tells a step of the wheel as buttons 4 and 5.
        return None
    return Event(kind, to_art(event.pos), button=event.button, mods=mods)


def inside(rect: Rect, pos: Pos) -> bool:
    """Whether a position is on one of a rectangle's own pixels."""
    x, y, w, h = rect
    return x <= pos[0] < x + w and y <= pos[1] < y + h


def share(total: int, weights: Sequence[int]) -> list[int]:
    """`total` in whole parts, one for each weight and as near in proportion to it as whole parts can be.

    The parts add up to `total` exactly. The weights are above 0.
    """
    whole = sum(weights)
    edges, so_far = [0], 0
    for weight in weights:
        so_far += weight
        edges.append((2 * total * so_far + whole) // (2 * whole))
    return [after - before for before, after in zip(edges, edges[1:])]


class Widget:
    """A part of the screen with a rectangle: it is offered events, and draws itself. This one does neither.

    `help` is one plain line shown when the mouse rests on it. One that is not `enabled` or not `visible` is
    offered no events, and one that is not visible is not drawn. `hot` (the mouse is on it) and `held` (it is
    pressed) are set by the Ui just before the widget is drawn, for it to ask while drawing.
    """

    def __init__(self, rect: Rect, help: str = "") -> None:
        self.rect = rect
        self.help = help
        self.enabled = True
        self.visible = True
        self.hot = False
        self.held = False

    def hit(self, pos: Pos) -> bool:
        """Whether a position is on the widget."""
        return inside(self.rect, pos)

    def on(self, event: Event, ui: Ui) -> bool:
        """Take an event, and say whether it was used. One that was not is offered to what lies below."""
        return False

    def draw(self, canvas: Canvas) -> None:
        """Draw the widget as it is now."""


class Ui:
    """The widgets in the order they are drawn, the last on top, and who among them an event goes to.

    `keys` is from (key, mods) to a function, called when that key goes down and no widget used it.
    """

    def __init__(self) -> None:
        self.widgets: list[Widget] = []
        self.keys: dict[tuple[str, frozenset[str]], Callable[[], object]] = {}
        self.mouse: Pos = NOWHERE
        #: The widget that used a "down" whose "up" has not come yet.
        self.held: Widget | None = None
        self._button = 0

    def add(self, widget: Widget) -> Widget:
        """Put a widget on top of the others, and give it back."""
        self.widgets.append(widget)
        return widget

    def _live(self) -> list[Widget]:
        """The widgets that are offered events, the topmost first."""
        return [widget for widget in reversed(self.widgets) if widget.visible and widget.enabled]

    @property
    def hot(self) -> Widget | None:
        """The topmost visible, enabled widget under the mouse. While a widget is held, no other one is hot."""
        for widget in self._live():
            if widget.hit(self.mouse):
                return widget if self.held in (None, widget) else None
        return None

    @property
    def help(self) -> str:
        """The help line of the hot widget, and nothing when there is none."""
        hot = self.hot
        return hot.help if hot is not None else ""

    def handle(self, event: Event) -> bool:
        """Hand an event to the widgets, and say whether it was used.

        What the mouse does is offered to the widgets under it, the topmost first, until one uses it. A widget that
        used a "down" is held: it gets everything the mouse does up to the "up" of the same button, also when the
        mouse has left it. A key is offered to every widget, the topmost first, and then to `keys`.
        """
        if event.kind == "key":
            if any(widget.on(event, self) for widget in self._live()):
                return True
            action = self.keys.get((event.key, event.mods))
            if action is not None:
                action()
            return action is not None
        self.mouse = event.pos
        if self.held is not None:
            self.held.on(event, self)
            if event.kind == "up" and event.button == self._button:
                self.held = None
            return True
        for widget in self._live():
            if widget.hit(event.pos) and widget.on(event, self):
                if event.kind == "down":
                    self.held, self._button = widget, event.button
                return True
        return False

    def draw(self, canvas: Canvas) -> None:
        """Draw the visible widgets, the first at the bottom, each told whether it is hot and whether it is held."""
        hot = self.hot
        for widget in self.widgets:
            if widget.visible:
                widget.hot, widget.held = widget is hot, widget is self.held
                widget.draw(canvas)


class Label(Widget):
    """One line of text in a rectangle. The mouse goes through it to what lies below."""

    def __init__(self, rect: Rect, text: str, colour: Colour = theme.DIM, align: str = "left") -> None:
        super().__init__(rect)
        self.text = text
        self.colour = colour
        self.align = align

    def hit(self, pos: Pos) -> bool:
        """Never: a label takes nothing from the mouse."""
        return False

    def draw(self, canvas: Canvas) -> None:
        """Draw the text, cut if the rectangle is too narrow for it."""
        canvas.text_in(self.text, self.rect, self.colour, self.align)


def draw_button(canvas: Canvas, rect: Rect, text: str = "", icon: str = "", ink: Colour = theme.TEXT,
                hot: bool = False, pressed: bool = False, strip: Colour | None = None) -> None:
    """Draw a button: its face and outline, then its icon, its text, or its icon and then its text, in the middle.

    The face is lit when `hot`. When `pressed` it is sunk, and what is on it sits one pixel lower. `strip` is the
    colour of a strip along the lower edge.
    """
    x, y, w, h = rect
    canvas.box(rect, theme.BACKGROUND if pressed else theme.LIT if hot else theme.RAISED)
    if not pressed:
        # A raised button catches the light along its upper edge.
        canvas.hline(x + 1, y + 1, w - 2, theme.LIT)
    canvas.frame(rect, theme.LINE)
    if strip is not None:
        canvas.box((x, y + h - theme.STRIP, w, theme.STRIP), strip)
    icon_w, icon_h = canvas.icon_size(icon) if icon else (0, 0)
    gap = theme.GAP if icon and text else 0
    shown = fit(text, w - 2 * theme.PAD - icon_w - gap)
    left = x + (w - icon_w - gap - Font.width(shown)) // 2
    top = y + int(pressed)
    with canvas.clip(rect):
        if icon:
            canvas.icon(icon, left, top + (h - icon_h) // 2, ink)
        canvas.text_in(shown, (left + icon_w + gap, top, Font.width(shown), h), ink)


class Button(Widget):
    """A button with an icon, a text, or an icon and then a text. `action` is called when it is pressed.

    It is pressed by the left button, on the release, and only when the release is still on it.
    """

    def __init__(self, rect: Rect, text: str = "", icon: str = "", action: Callable[[], object] | None = None,
                 help: str = "") -> None:
        super().__init__(rect, help)
        self.text = text
        self.icon = icon
        self.action = action

    def on(self, event: Event, ui: Ui) -> bool:
        """Use a press of the left button, and fire on its release when that is still on the button."""
        if event.kind == "down":
            return event.button == 1
        if event.kind == "up" and event.button == 1 and ui.held is self and self.hit(event.pos):
            self.fire()
        return False

    def fire(self) -> None:
        """Do what the button is for."""
        if self.action is not None:
            self.action()

    def draw(self, canvas: Canvas) -> None:
        """Draw it raised, lit when hot, and sunk while it is held with the mouse on it. Dim when not enabled."""
        draw_button(canvas, self.rect, self.text, self.icon, theme.TEXT if self.enabled else theme.DIM,
                    self.hot, self.hot and self.held)


class Toggle(Button):
    """A button that stays on or off: `is_on`. `changed` is called with the new state when it flips.

    On, its text and its icon are in `colour`, and a strip of that colour runs along its lower edge.
    """

    def __init__(self, rect: Rect, text: str = "", icon: str = "", on: bool = False,
                 changed: Callable[[bool], object] | None = None, colour: Colour = theme.ACTIVE,
                 help: str = "") -> None:
        super().__init__(rect, text, icon, None, help)
        self.is_on = on
        self.changed = changed
        self.colour = colour

    def fire(self) -> None:
        """Flip it, and tell."""
        self.is_on = not self.is_on
        if self.changed is not None:
            self.changed(self.is_on)

    def draw(self, canvas: Canvas) -> None:
        """Draw it as a button, with its colour on what it shows and along its lower edge when it is on."""
        ink = theme.DIM if not self.enabled else self.colour if self.is_on else theme.TEXT
        draw_button(canvas, self.rect, self.text, self.icon, ink, self.hot, self.hot and self.held,
                    self.colour if self.is_on else None)


class Choice(Widget):
    """A row of joined toggles, one for each text of `options`, of which exactly one is on: `picked`.

    The rectangle is shared out between them by the width of their texts. `changed` is called with the index of
    the new one when another is picked.
    """

    def __init__(self, rect: Rect, options: Sequence[str], picked: int = 0,
                 changed: Callable[[int], object] | None = None, help: str = "") -> None:
        super().__init__(rect, help)
        self.options = tuple(options)
        if not self.options:
            raise RowSynthError("a choice needs at least one option")
        if not 0 <= picked < len(self.options):
            raise RowSynthError(f"a choice of {len(self.options)} options has no option {picked}")
        self.picked = picked
        self.changed = changed
        self._over: int | None = None

    def cells(self) -> list[Rect]:
        """The rectangle of each option, side by side from the left: together they are the whole rectangle."""
        x, y, _, h = self.rect
        cells = []
        for width in share(self.rect[2], [Font.width(text) + 2 * theme.PAD for text in self.options]):
            cells.append((x, y, width, h))
            x += width
        return cells

    def at(self, pos: Pos) -> int | None:
        """The index of the option a position is on, or None when it is on none."""
        return next((index for index, cell in enumerate(self.cells()) if inside(cell, pos)), None)

    def pick(self, index: int) -> None:
        """Turn one option on and the one that was on off. Picking the one that is on changes nothing."""
        if index != self.picked:
            self.picked = index
            if self.changed is not None:
                self.changed(index)

    def on(self, event: Event, ui: Ui) -> bool:
        """Use a press of the left button, and pick the option its release is on."""
        if event.kind == "key":
            return False
        self._over = self.at(event.pos)
        if event.kind == "down":
            return event.button == 1
        if event.kind == "up" and event.button == 1 and ui.held is self and self._over is not None:
            self.pick(self._over)
        return False

    def draw(self, canvas: Canvas) -> None:
        """Draw the options as toggles that share the line between them. The one that is on is drawn last, whole."""
        cells = self.cells()
        for index in sorted(range(len(cells)), key=lambda index: index == self.picked):
            x, y, w, h = cells[index]
            is_on = index == self.picked
            lit = self.hot and index == self._over
            ink = theme.DIM if not self.enabled else theme.ACTIVE if is_on else theme.TEXT
            # All but the last reach one pixel into the next, so that the two outlines are one line.
            draw_button(canvas, (x, y, w + (index < len(cells) - 1), h), self.options[index], "", ink, lit,
                        lit and self.held, theme.ACTIVE if is_on else None)
