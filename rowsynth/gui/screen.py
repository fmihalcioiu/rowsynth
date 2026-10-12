"""The window, and the small surface that is drawn on and scaled up whole to fill it."""
from __future__ import annotations

import ctypes
import sys
import warnings

import pygame

from ..errors import RowSynthError
from . import layout, theme, winframe
from .layout import Rect

#: What pygame's own handle of a window has to have for a window with no frame of the system's: the way to it,
#: where the window stands, how big it is, and the way to take it off the screen.
NEEDS = ("from_display_module", "position", "size", "minimize")
#: The pointers of the system by the shape they are asked for with, "" for the usual arrow: the names pygame has.
POINTERS = {"": "SYSTEM_CURSOR_ARROW", "ew": "SYSTEM_CURSOR_SIZEWE", "ns": "SYSTEM_CURSOR_SIZENS",
            "nwse": "SYSTEM_CURSOR_SIZENWSE", "nesw": "SYSTEM_CURSOR_SIZENESW"}
#: What Windows is asked with for the display nearest to a point.
_NEAREST = 2
# The handle of the window that pygame gave out last. Classic pygame leaves a bare pointer to it in the window
# and follows that pointer each time it tells what happened to the window, whether the handle is still there or
# not. So the last one is held on to for as long as the program runs, and a handle that is let go of too soon
# cannot bring the program down.
_HANDLES: list[object] = []


class _Point(ctypes.Structure):
    """A point of the desktop, as Windows takes one."""

    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class _Display(ctypes.Structure):
    """What Windows tells of a display: all of it, and the part the task bar leaves, each as its four sides."""

    _fields_ = [("size", ctypes.c_ulong), ("whole", ctypes.c_long * 4), ("work", ctypes.c_long * 4),
                ("flags", ctypes.c_ulong)]


def fit(window: tuple[int, int], scale: int | None = None) -> tuple[int, tuple[int, int]]:
    """The scale and the size of the art surface for a window of a size. The scale is a whole number.

    With no scale given it is the largest at which the art surface is no smaller than `theme.MIN_ART`, and 1 in
    a window too small for that. The art surface is the window's size divided by the scale, rounded down, so a
    larger window shows more and shows nothing larger. A scale under 1 is refused.
    """
    width, height = window
    if scale is None:
        scale = max(1, min(width // theme.MIN_ART[0], height // theme.MIN_ART[1]))
    elif scale < 1:
        raise RowSynthError(f"a scale of {scale} cannot be drawn: the scale is a whole number, 1 or more")
    return scale, (max(1, width // scale), max(1, height // scale))


def opening(desktop: tuple[int, int], scale: int | None = None) -> tuple[int, int]:
    """The size a window opens at on a desktop of a size, so that it fits there.

    The desktop's height is first less by what the title bar and the task bar take. The window is the size the
    layout is drawn for times a scale, cut down to what the desktop has left. The scale is the one given, or
    with none the largest that `fit` would still choose in what is left. A desktop too small for the layout at
    scale 1 gets a window of what it has left.
    """
    width, height = max(1, desktop[0]), max(1, desktop[1] - theme.DESKTOP_TAKEN_H)
    scale, _ = fit((width, height), scale)
    return min(theme.DESIGN[0] * scale, width), min(theme.DESIGN[1] * scale, height)


def dpi_aware() -> None:
    """On Windows, tell the system that this process draws its own pixels. Elsewhere there is nothing to tell.

    Untold, Windows stretches the whole window on a display that is scaled above 100%, which smooths the pixels.
    It has to be told before the display is started. Where it cannot be told, the window is opened all the same.
    """
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


def own_frame() -> type | None:
    """pygame's own handle of a window, where the running pygame has all that a window with no frame needs.

    None where it lacks any of `NEEDS`, or the flag that opens a window with no frame: the window then keeps
    the frame of the system. What is asked for is what classic pygame 2.3 and pygame-ce both have.
    """
    try:
        from pygame._sdl2.video import Window
    except ImportError:
        return None
    return Window if hasattr(pygame, "NOFRAME") and all(hasattr(Window, name) for name in NEEDS) else None


def work_area(at: tuple[int, int], desktop: tuple[int, int]) -> Rect:
    """The part of the desktop a window may fill, as (x, y, w, h): what the task bar leaves of a display.

    On Windows it is that of the display nearest to the point `at`. On any other system, and where Windows
    cannot be asked, it is all of a desktop of the size `desktop`, from its upper left corner.
    """
    if sys.platform == "win32":
        try:
            user32 = ctypes.WinDLL("user32")
            user32.MonitorFromPoint.restype = ctypes.c_void_p
            user32.MonitorFromPoint.argtypes = [_Point, ctypes.c_ulong]
            user32.GetMonitorInfoW.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Display)]
            told = _Display(size=ctypes.sizeof(_Display))
            if user32.GetMonitorInfoW(user32.MonitorFromPoint(_Point(*at), _NEAREST), ctypes.byref(told)):
                left, top, right, bottom = told.work
                return left, top, right - left, bottom - top
        except (AttributeError, OSError, TypeError, ValueError):
            pass
    return 0, 0, desktop[0], desktop[1]


class Screen:
    """The window and the art surface. Everything is drawn on the art surface, in art pixels.

    `frameless` is whether the window shows no frame of the system's, and whoever draws in it draws a title row
    of its own. On Windows the system still moves, sizes and snaps such a window, as it does any other: `frame`
    is the hold on the window by which it is told where the title row and the edges are, and `asked` is what
    it asks for a position of the window ("caption", "button" or "", which the owner of the title row sets).
    `during` is called again and again while the system moves or sizes the window, for its owner to draw it:
    the loop that draws waits for as long as that lasts.
    Anywhere else `frame` is None, and the window is moved, sized, taken off the screen and made to fill the
    desktop from here. Where the running pygame cannot do that either, the window keeps the system's frame.
    `kept` is where a window that fills the desktop stood before, and None for one that does not fill it.
    """

    def __init__(self, size: tuple[int, int] | None = None, scale: int | None = None) -> None:
        """Open a window that can be resized. Its scale is kept at every size: `scale`, or with none the one
        that `fit` chooses for the size the window opens at. A smaller window shows less, and nothing smaller.

        A `size` that is given is the window's, as it is. With none the window opens at the size that fits the
        first desktop, which is the main one.
        """
        self._given = scale
        # A scale that cannot be drawn is refused here, before the display is started.
        fit(theme.DESIGN, scale)
        dpi_aware()
        pygame.display.init()
        if size is None:
            size = opening(pygame.display.get_desktop_sizes()[0], scale)
        pygame.display.set_caption("rowsynth")
        self.frameless = False
        self.frame: winframe.Frame | None = None
        self.asked = lambda pos: ""
        self.during = lambda: None
        self.kept: Rect | None = None
        self._shape = ""
        self.window = self._open(size)
        # The window may not have come out at the size asked for, so its own size is the one that counts.
        self.scale, art = fit(self.window.get_size(), scale)
        self._given = self.scale
        self.art = pygame.Surface(art)
        if self.frame is not None:
            self.frame.least = self.least

    def _open(self, size: tuple[int, int]) -> pygame.Surface:
        """Open the window with no frame of the system's where pygame can stand in for the frame, and else with it.

        A window that came out with no frame and cannot be reached after all is opened again with the frame:
        it is never left as one that cannot be moved or closed.
        """
        window = self._native(size)
        if window is not None:
            return window
        if own_frame() is not None:
            try:
                window = pygame.display.set_mode(size, pygame.RESIZABLE | pygame.NOFRAME)
                handle = self._reach()
                self.frameless = len(tuple(handle.position)) == 2 and len(tuple(handle.size)) == 2
            except Exception:
                # Whatever it is that this pygame or this system does not do: the frame of the system does it.
                self.frameless = False
            if self.frameless:
                return window
            # The display is started anew, so that nothing is left of the window with no frame.
            pygame.display.quit()
            pygame.display.init()
            pygame.display.set_caption("rowsynth")
        return pygame.display.set_mode(size, pygame.RESIZABLE)

    def _native(self, size: tuple[int, int]) -> pygame.Surface | None:
        """Open the window as one the system itself moves and sizes, with no frame to be seen. None where it cannot.

        That is on Windows, with its own video driver. The window is opened with the system's frame and then
        covers all of it with its picture, so it is made as big as it was asked to be once more.
        """
        if sys.platform != "win32" or pygame.display.get_driver() != "windows":
            return None
        try:
            pygame.display.set_mode(size, pygame.RESIZABLE)
            self.frame = winframe.Frame(pygame.display.get_wm_info()["window"], lambda pos: self.asked(pos))
            self.frame.during = lambda: self.during()
            self.frame.fit(size)
            pygame.event.pump()
        except Exception:
            # Whatever the system refused: the window is opened again in one of the other ways.
            self.frame = None
            pygame.display.quit()
            pygame.display.init()
            pygame.display.set_caption("rowsynth")
            return None
        self.frameless = True
        return pygame.display.get_surface()

    @staticmethod
    def _reach() -> object:
        """pygame's own handle of the window that is open, by which it is moved and sized.

        It is asked for anew each time, so that it is never the handle of a window that is gone, and it takes
        the place of the one in `_HANDLES`, which pygame no longer points to once there is a newer one.
        """
        from pygame._sdl2.video import Window

        with warnings.catch_warnings():
            # pygame-ce says of this way to the window that it is an old one. It is the one both have.
            warnings.simplefilter("ignore", DeprecationWarning)
            handle = Window.from_display_module()
        _HANDLES[:] = [handle]
        return handle

    @property
    def least(self) -> tuple[int, int]:
        """The least size the window is dragged to, in window pixels: `theme.LEAST_ART` at the window's scale."""
        return theme.LEAST_ART[0] * self.scale, theme.LEAST_ART[1] * self.scale

    @property
    def big(self) -> bool:
        """Whether the window fills the desktop: as the system has it, or as it was made to here and not put back."""
        return self.frame.big if self.frame is not None else self.kept is not None

    def stand(self) -> Rect | None:
        """Where the window stands on the desktop and how big it is, as (x, y, w, h). None with the system's frame."""
        if not self.frameless:
            return None
        try:
            handle = self._reach()
            return (*map(int, handle.position), *map(int, handle.size))
        except pygame.error:
            return None

    def put(self, rect: Rect) -> None:
        """Have the window stand at a place of the desktop, at a size. With the system's frame, nothing happens."""
        if not self.frameless:
            return
        try:
            handle = self._reach()
            if tuple(handle.size) != tuple(rect[2:]):
                handle.size = rect[2:]
            if tuple(handle.position) != tuple(rect[:2]):
                handle.position = rect[:2]
        except pygame.error:
            pass

    def minimise(self) -> None:
        """Take the window off the screen, down to where the system keeps the windows that are open."""
        if self.frame is not None:
            self.frame.minimise()
        elif self.frameless:
            try:
                self._reach().minimize()
            except pygame.error:
                pass

    def fill(self) -> None:
        """Have the window fill the part of its display that a window may fill, or put it back where it stood."""
        if self.frame is not None:
            self.frame.fill()
            return
        stand = self.stand()
        if stand is not None:
            middle = (stand[0] + stand[2] // 2, stand[1] + stand[3] // 2)
            work = work_area(middle, pygame.display.get_desktop_sizes()[0])
            rect, self.kept = layout.filled(stand, self.kept, work)
            self.put(rect)

    def pointer(self, shape: str) -> None:
        """Give the mouse pointer a shape of `POINTERS`, "" for the usual arrow. Where there is none, nothing."""
        if shape == self._shape:
            return
        self._shape = shape
        try:
            pygame.mouse.set_cursor(getattr(pygame, POINTERS[shape]))
        except (pygame.error, AttributeError):
            pass

    def name(self, text: str) -> None:
        """Put the name of the open file after the program's in what the system calls the window. "" for none."""
        pygame.display.set_caption(f"rowsynth  {text}" if text else "rowsynth")

    def begin(self) -> pygame.Surface:
        """The art surface, to draw the next frame on. It follows the window: after a resize it has a new size.

        The scale stays what it was as the window opened.
        """
        self.window = pygame.display.get_surface()
        self.scale, art = fit(self.window.get_size(), self._given)
        if art != self.art.get_size():
            self.art = pygame.Surface(art)
        return self.art

    def present(self) -> None:
        """Show the art surface in the window: each of its pixels as a square of whole pixels, with no smoothing."""
        width, height = self.art.get_size()
        shown = (width * self.scale, height * self.scale)
        if shown != self.window.get_size():
            # The window is not a whole number of art pixels across or down, and a strip is left over at an edge.
            self.window.fill(theme.BACKGROUND)
        self.window.blit(self.art if self.scale == 1 else pygame.transform.scale(self.art, shown), (0, 0))
        pygame.display.flip()

    def to_art(self, pos: tuple[int, int]) -> tuple[int, int]:
        """A position in the window, as the mouse gives it, in pixels of the art surface."""
        return pos[0] // self.scale, pos[1] // self.scale

    def snapshot(self) -> pygame.Surface:
        """A copy of the art surface as it is now."""
        return self.art.copy()

    def close(self) -> None:
        """Close the window."""
        if self.frame is not None:
            self.frame.release()
            self.frame = None
        pygame.display.quit()
