"""Where the parts of the frame go on an art surface, and where a window stands after a drag. Numbers in and out."""
from __future__ import annotations

from . import theme

Rect = tuple[int, int, int, int]

#: How high the window's own title row is, in art pixels, where the window has no frame of the system's.
TITLE_H = 14
#: The least width a side column keeps while its border is dragged, and the least that is then left to the middle.
LEAST_SIDE = 72
LEAST_MIDDLE = 320
#: How far the strip a border is taken by reaches to each side of the border's line.
BORDER_REACH = 2

#: How far into a window an edge takes the mouse, in window pixels, and how far along an edge its corner does.
EDGE_REACH = 4
CORNER_REACH = 12
#: The least size a window is dragged to, in window pixels.
LEAST_WINDOW = (640, 360)
#: What a drag is called that moves the whole window, where an edge or a corner is named by its compass letters.
MOVE = "move"
#: The shape the mouse pointer has over each edge and corner: two arrows up and down, across, or aslant.
POINTERS = {"n": "ns", "s": "ns", "w": "ew", "e": "ew", "nw": "nwse", "se": "nwse", "ne": "nesw", "sw": "nesw"}


def frame(size: tuple[int, int], title: int = 0, left: int | None = None, right: int | None = None) -> dict[str, Rect]:
    """The parts of the frame on an art surface of `size`, each by its name as (x, y, w, h).

    "title" is the window's own title row, `title` pixels high: with 0 there is none, and the part has no
    height. "top" and "status" are the bars, and all three run the whole width. "left" and "right" are the
    panels that stand between the bars, `left` and `right` pixels wide, or with None as wide as the theme has
    them, and "middle" is what is left. Together they cover the surface, and no two share a pixel. The rows
    and the panels keep their sizes, so a larger surface gives its room to the middle. On a surface too small
    for them they are cut to fit, down to nothing.
    """
    width, height = size
    title = min(max(0, title), height)
    top = min(theme.TOP_H, height - title)
    status = min(theme.STATUS_H, height - title - top)
    left = min(max(0, theme.LEFT_W if left is None else left), width)
    right = min(max(0, theme.RIGHT_W if right is None else right), width - left)
    under, between = title + top, height - title - top - status
    return {
        "title": (0, 0, width, title),
        "top": (0, title, width, top),
        "left": (0, under, left, between),
        "middle": (left, under, width - left - right, between),
        "right": (width - right, under, right, between),
        "status": (0, height - status, width, status),
    }


def edge(rect: Rect, side: str) -> Rect:
    """The strip one pixel thick just inside one side of a rectangle: "top", "bottom", "left" or "right"."""
    x, y, w, h = rect
    return {
        "top": (x, y, w, 1),
        "bottom": (x, y + h - 1, w, 1),
        "left": (x, y, 1, h),
        "right": (x + w - 1, y, 1, h),
    }[side]


def borders(parts: dict[str, Rect]) -> dict[str, Rect]:
    """The strip each border between two columns is taken by, in the parts `frame` gives: "left" and "right".

    "left" is the border between the left panel and the middle, and "right" the one between the middle and the
    right panel. A strip is as high as the columns, and reaches `BORDER_REACH` pixels to each side of the line
    of its panel: the last pixel of the left panel, and the first of the right one.
    """
    x, y, w, h = parts["middle"]
    wide = 2 * BORDER_REACH + 1
    return {"left": (x - 1 - BORDER_REACH, y, wide, h), "right": (x + w - BORDER_REACH, y, wide, h)}


def pulled(width: int, left: int, right: int, border: str, x: int) -> tuple[int, int]:
    """How wide the left panel and the right one are after a border was dragged to an x, on a surface of a width.

    `border` is "left" or "right", as `borders` names them, and `x` is where the line of that border is put.
    The panel of the border is then as wide as that makes it, and the other stays. It is never narrower than
    `LEAST_SIDE`, and never so wide that the middle has less than `LEAST_MIDDLE`: on a surface with no room for
    both, the panel keeps its least width.
    """
    if border == "left":
        return max(LEAST_SIDE, min(x + 1, width - right - LEAST_MIDDLE)), right
    return left, max(LEAST_SIDE, min(width - x, width - left - LEAST_MIDDLE))


def grip(size: tuple[int, int], pos: tuple[int, int]) -> str:
    """Which edge or corner of a window a position is on, both in window pixels, and "" for none of them.

    An edge is "n", "s", "w" or "e", and a corner is the two edges that meet in it, as in "nw". An edge takes
    the mouse `EDGE_REACH` pixels into the window. A corner takes it that far in from either of its edges, for
    `CORNER_REACH` pixels along that edge. A position off the window is on none.
    """
    (width, height), (x, y) = size, pos
    if not (0 <= x < width and 0 <= y < height):
        return ""
    at_side = x < EDGE_REACH or x >= width - EDGE_REACH
    at_end = y < EDGE_REACH or y >= height - EDGE_REACH
    if not (at_side or at_end):
        return ""
    down = CORNER_REACH if at_side else EDGE_REACH
    along = CORNER_REACH if at_end else EDGE_REACH
    return (("n" if y < down else "s" if y >= height - down else "")
            + ("w" if x < along else "e" if x >= width - along else ""))


def dragged(rect: Rect, grip: str, by: tuple[int, int], least: tuple[int, int] = LEAST_WINDOW) -> Rect:
    """Where a window stands and how big it is after a drag, as (x, y, w, h) in pixels of the desktop.

    `rect` is the window as the drag began, and `by` how far the mouse has gone since, across and down. With
    `MOVE` for the grip the whole window goes with the mouse, at the size it has. With an edge or a corner, as
    `grip` names them, that edge or those two go with the mouse and the others stay where they are. The window
    is never made smaller than `least`: an edge stops where the window would be.
    """
    (x, y, w, h), (across, down) = rect, by
    if grip == MOVE:
        return x + across, y + down, w, h
    if "e" in grip:
        w = max(least[0], w + across)
    elif "w" in grip:
        across = min(across, w - least[0])
        x, w = x + across, w - across
    if "s" in grip:
        h = max(least[1], h + down)
    elif "n" in grip:
        down = min(down, h - least[1])
        y, h = y + down, h - down
    return x, y, w, h


def filled(rect: Rect, kept: Rect | None, work: Rect) -> tuple[Rect, Rect | None]:
    """A window made to fill the desktop, or put back: where it then stands, and what is kept of where it stood.

    `rect` is the window as it stands. `kept` is None for a window that does not fill the desktop: it is then
    put on `work`, the part of the desktop a window may fill, and `rect` is kept. Any other `kept` is where
    the window stood before it filled the desktop: it is put back there, and nothing is kept.
    """
    return (work, rect) if kept is None else (kept, None)
