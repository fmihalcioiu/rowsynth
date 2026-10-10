"""Where the parts of the frame go on an art surface of a size. Numbers in, numbers out."""
from __future__ import annotations

from . import theme

Rect = tuple[int, int, int, int]


def frame(size: tuple[int, int]) -> dict[str, Rect]:
    """The parts of the frame on an art surface of `size`, each by its name as (x, y, w, h).

    "top" and "status" are the bars, and run the whole width. "left" and "right" are the panels that stand
    between them, and "middle" is what is left. Together they cover the surface, and no two share a pixel. The
    bars and the panels keep their sizes, so a larger surface gives its room to the middle. On a surface too
    small for them they are cut to fit, down to nothing.
    """
    width, height = size
    top = min(theme.TOP_H, height)
    status = min(theme.STATUS_H, height - top)
    left = min(theme.LEFT_W, width)
    right = min(theme.RIGHT_W, width - left)
    between = height - top - status
    return {
        "top": (0, 0, width, top),
        "left": (0, top, left, between),
        "middle": (left, top, width - left - right, between),
        "right": (width - right, top, right, between),
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
