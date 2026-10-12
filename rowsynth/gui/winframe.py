"""On Windows, a window that draws its own title row and is still moved, sized and snapped by the system."""
from __future__ import annotations

import ctypes
import sys
from collections.abc import Callable

from . import layout

Pos = tuple[int, int]

#: What the system is told a position of the window is: the window's own, or the row it is moved by.
CLIENT, CAPTION = 1, 2
#: And an edge or a corner, by the name `layout.grip` gives it.
EDGES = {"w": 10, "e": 11, "n": 12, "nw": 13, "ne": 14, "s": 15, "sw": 16, "se": 17}

_CALC_SIZE, _HIT_TEST, _CLOSE, _SIZES = 0x0083, 0x0084, 0x0010, 0x0024
_WNDPROC = -4
_SHOW_MAXIMISED, _SHOW_MINIMISED, _SHOW_RESTORED = 3, 6, 9
_NO_SIZE, _NO_MOVE, _NO_Z_ORDER, _FRAME_CHANGED = 0x0001, 0x0002, 0x0004, 0x0020
_NEAREST = 2
# The procedures handed to the system. It calls them for as long as their windows are there, so they are held
# on to for as long as the program runs: one that is let go of too soon brings the program down.
_KEPT: list[object] = []


def zone(size: tuple[int, int], pos: Pos, what: str, big: bool) -> int:
    """What the system is told a position of the window is, both in window pixels.

    `what` is what the window has there: "caption" for the part of its title row that moves it, "button" for a
    button of that row, and "" for anything else. A button is the window's own, wherever it stands. An edge or
    a corner, as `layout.grip` finds them, sizes the window, but for one that fills its display, which has
    none. The caption moves it, and the rest is the window's own.
    """
    if what == "button":
        return CLIENT
    grip = "" if big else layout.grip(size, pos)
    if grip:
        return EDGES[grip]
    return CAPTION if what == "caption" else CLIENT


class _Rect(ctypes.Structure):
    """Four sides, as Windows gives them."""

    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class _Display(ctypes.Structure):
    """What Windows tells of a display: all of it, and the part the task bar leaves."""

    _fields_ = [("size", ctypes.c_ulong), ("whole", _Rect), ("work", _Rect), ("flags", ctypes.c_ulong)]


class _Point(ctypes.Structure):
    """Two numbers, across and down, as Windows gives a point or a size."""

    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class _Sizes(ctypes.Structure):
    """What Windows asks a window of the sizes it may have: of them all, `least` is the one answered here."""

    _fields_ = [("kept", _Point), ("big", _Point), ("big_at", _Point), ("least", _Point), ("most", _Point)]


def _signed(value: int) -> int:
    """The low 16 bits of a number as a whole number with a sign, which is how a message holds an x or a y."""
    value &= 0xFFFF
    return value - 0x10000 if value & 0x8000 else value


class Frame:
    """The hold on one window of this process, by which the system is told where its title row and edges are.

    The window keeps the frame of the system as far as the system is concerned, so it is moved, sized,
    snapped to the sides of the screen, taken off the screen and brought back by the system, as any window
    is. Nothing of that frame is drawn: the window's own picture covers all of it. `asked` is asked, for a
    position of the window, what the window has there, in the words `zone` takes. `least` is the least size
    the system lets the window be dragged or snapped to, in window pixels.

    Only Windows has this. Anywhere else, and where the system refuses, making one raises OSError.
    """

    def __init__(self, hwnd: int, asked: Callable[[Pos], str]) -> None:
        if sys.platform != "win32":
            raise OSError("only Windows moves a window by a title row the window draws itself")
        self.hwnd = hwnd
        self.asked = asked
        self.least = (0, 0)
        user = self._user = ctypes.WinDLL("user32", use_last_error=True)
        pointer, word, long = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ssize_t
        self._kind = ctypes.WINFUNCTYPE(long, pointer, ctypes.c_uint, word, long)
        user.SetWindowLongPtrW.restype, user.SetWindowLongPtrW.argtypes = pointer, [pointer, ctypes.c_int, pointer]
        user.CallWindowProcW.restype = long
        user.CallWindowProcW.argtypes = [pointer, pointer, ctypes.c_uint, word, long]
        user.GetWindowRect.argtypes = [pointer, ctypes.POINTER(_Rect)]
        user.IsZoomed.argtypes = [pointer]
        user.ShowWindow.argtypes = [pointer, ctypes.c_int]
        user.SetWindowPos.argtypes = [pointer, pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_uint]
        user.MonitorFromWindow.restype, user.MonitorFromWindow.argtypes = pointer, [pointer, ctypes.c_ulong]
        user.GetMonitorInfoW.argtypes = [pointer, ctypes.POINTER(_Display)]
        user.PostMessageW.argtypes = [pointer, ctypes.c_uint, word, long]
        self._own = self._kind(self._told)
        _KEPT.append(self._own)
        self._before = user.SetWindowLongPtrW(hwnd, _WNDPROC, ctypes.cast(self._own, pointer))
        if not self._before:
            raise OSError("the window does not take a procedure of its own")
        self._released = False
        # The system works out anew what of the window is its frame, which is now nothing.
        user.SetWindowPos(hwnd, None, 0, 0, 0, 0, _NO_MOVE | _NO_SIZE | _NO_Z_ORDER | _FRAME_CHANGED)

    def _told(self, hwnd: int, message: int, first: int, second: int) -> int:
        """What the system is answered for one message to the window. Any but three goes on to the window as it
        was.

        Nothing may be raised here, where the system is the caller: whatever goes wrong, the message goes on.
        """
        try:
            if message == _SIZES:
                # The window as it was answers first, and its least size is then made the one of here.
                answer = self._user.CallWindowProcW(self._before, hwnd, message, first, second)
                sizes = ctypes.cast(second, ctypes.POINTER(_Sizes)).contents
                sizes.least.x, sizes.least.y = max(sizes.least.x, self.least[0]), max(sizes.least.y, self.least[1])
                return answer
            if message == _CALC_SIZE:
                # All of the window is the window's own. One that fills its display would hang over the edges
                # of it by the frame the system gives it, so it is set on what the task bar leaves.
                if first and self._user.IsZoomed(hwnd):
                    work = self._work()
                    if work is not None:
                        rect = ctypes.cast(second, ctypes.POINTER(_Rect)).contents
                        rect.left, rect.top, rect.right, rect.bottom = work
                return 0
            if message == _HIT_TEST:
                stands = _Rect()
                if self._user.GetWindowRect(hwnd, ctypes.byref(stands)):
                    pos = (_signed(second) - stands.left, _signed(second >> 16) - stands.top)
                    size = (stands.right - stands.left, stands.bottom - stands.top)
                    return zone(size, pos, self.asked(pos), bool(self._user.IsZoomed(hwnd)))
        except Exception:
            pass
        return self._user.CallWindowProcW(self._before, hwnd, message, first, second)

    def _work(self) -> tuple[int, int, int, int] | None:
        """The four sides of what the task bar leaves of the window's display, or None where it cannot be asked."""
        told = _Display(size=ctypes.sizeof(_Display))
        if self._user.GetMonitorInfoW(self._user.MonitorFromWindow(self.hwnd, _NEAREST), ctypes.byref(told)):
            return told.work.left, told.work.top, told.work.right, told.work.bottom
        return None

    @property
    def big(self) -> bool:
        """Whether the window fills its display."""
        return bool(self._user.IsZoomed(self.hwnd))

    def fit(self, size: tuple[int, int]) -> None:
        """Have the whole window be a size, where it stands. Its picture is then that size too."""
        self._user.SetWindowPos(self.hwnd, None, 0, 0, size[0], size[1], _NO_MOVE | _NO_Z_ORDER | _FRAME_CHANGED)

    def minimise(self) -> None:
        """Take the window off the screen, as the system does it."""
        self._user.ShowWindow(self.hwnd, _SHOW_MINIMISED)

    def fill(self) -> None:
        """Have the window fill its display, or put one that does back where it stood."""
        self._user.ShowWindow(self.hwnd, _SHOW_RESTORED if self.big else _SHOW_MAXIMISED)

    def release(self) -> None:
        """Give the window back the procedure it had. It is done before the window is closed."""
        if not self._released:
            self._user.SetWindowLongPtrW(self.hwnd, _WNDPROC, self._before)
            self._released = True
