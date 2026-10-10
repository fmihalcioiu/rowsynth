"""The graphical interface: a project file open in a window. It needs pygame, which is imported only when it runs."""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from ..errors import RowSynthError

if TYPE_CHECKING:
    from .app import App


def start(path: str | Path, scale: int | None = None) -> App:
    """The app on the project in a file, with its window open. `scale` is how large an art pixel is shown.

    Without pygame this is refused, with how to get it. A file that holds no project is refused with no window
    opened.
    """
    # pygame greets on standard output when it is imported, which is where a command prints its one document.
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    # Asked for by name before the modules that use it, so that its absence is told as what it is.
    try:
        import pygame
    except ImportError:
        raise RowSynthError("the GUI needs pygame, which is not installed: pip install rowsynth[gui]") from None
    from .app import App

    return App(path, scale)


def check_frames(frames: int | None) -> None:
    """Refuse a count of frames to run for that is under 1. None is no count: the window stays until it is closed."""
    if frames is not None and frames < 1:
        raise RowSynthError(f"a run of {frames} frames shows nothing: the count of frames is a whole number, "
                            "1 or more")


def run(path: str | Path, scale: int | None = None, frames: int | None = None,
        shot: str | Path | None = None) -> int:
    """Open the project in a file in a window, keep it until the window is closed, and return 0.

    With `frames` the window closes by itself after that many frames: a count under 1 is refused, before any
    window opens. With `shot` its last frame is written to that path as a PNG, at the size of the art surface.
    """
    check_frames(frames)
    start(path, scale).run(frames, shot)
    return 0
