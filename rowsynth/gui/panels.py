"""A part of the window: a panel has a rectangle, the widgets it lays out there, and a background of its own."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .layout import Rect
from .ui import Ui, Widget

if TYPE_CHECKING:
    from .canvas import Canvas
    from .session import Session


def within(rect: Rect, bounds: Rect) -> Rect:
    """The part of a rectangle that lies inside another. Where none of it does, it has no width or no height."""
    x, y, w, h = bounds
    left = min(max(rect[0], x), x + w)
    top = min(max(rect[1], y), y + h)
    right = max(left, min(rect[0] + rect[2], x + w))
    bottom = max(top, min(rect[1] + rect[3], y + h))
    return left, top, right - left, bottom - top


class Panel:
    """The base of every part of the window. This one has no widgets and draws nothing.

    A panel is made with the session and the Ui, and adds its widgets to the Ui. `place` is called at the start
    and whenever the window has another size, `update` every frame before anything is drawn, with the seconds
    that passed since the frame before, and `draw` every frame before the widgets are.

    A panel can be taken off the window and brought back with `show`: two panels can then share one part of
    the frame, one of them shown at a time. Only the widgets it added through `add` go with it.
    """

    def __init__(self, session: Session, ui: Ui) -> None:
        self.session = session
        self.ui = ui
        self.rect: Rect = (0, 0, 0, 0)
        self.shown = True
        self.widgets: list[Widget] = []

    def add(self, widget: Widget) -> Widget:
        """Add a widget to the Ui as one of the panel's own, and give it back."""
        self.widgets.append(widget)
        widget.visible = self.shown
        return self.ui.add(widget)

    def show(self, on: bool) -> None:
        """Bring the panel onto the window, or take it off: off, its widgets are not seen and take no mouse."""
        self.shown = on
        for widget in self.widgets:
            widget.visible = on

    def place(self, rect: Rect) -> None:
        """Take a rectangle as the panel's own, and lay the widgets out in it."""
        self.rect = rect

    def update(self, passed: float = 0.0) -> None:
        """Read the session, and set what the widgets show. `passed` is the seconds since the frame before."""

    def draw(self, canvas: Canvas) -> None:
        """Draw the panel's own background, and whatever of it is not a widget."""
