"""Compiling the kernels, the loops over single samples, when Numba is there to do it."""
from __future__ import annotations

import os
from collections.abc import Callable
from typing import TypeVar

_Function = TypeVar("_Function", bound=Callable)


def _switched_off() -> bool:
    """True when the environment asks for no compiling: ROWSYNTH_NO_JIT is set, and to something other than "0"."""
    return os.environ.get("ROWSYNTH_NO_JIT", "") not in ("", "0")


def _compiler() -> Callable | None:
    """Numba's decorator for code that runs without the interpreter, or None when there is to be no compiling."""
    if _switched_off():
        return None
    try:
        from numba import njit
    except ImportError:
        # Without Numba the package loses its speed and nothing else.
        return None
    return njit


# Settled once, as this module is imported. A change to the environment after that changes nothing.
_NJIT = _compiler()

# True when the kernels are compiled, False when they run as the plain Python they are written as.
COMPILED = _NJIT is not None


def kernel(function: _Function) -> _Function:
    """Marks a loop over samples. Numba compiles it the first time it is called. With no compiling it stays as it is.

    Nothing is asked of the compiler that could move a result: no fast-math, so the samples are the ones the
    plain Python gives. Of this package's own functions, a kernel can call only other kernels.

    The compiled code is kept on disk, in the folder NUMBA_CACHE_DIR names or else in __pycache__ beside the
    source, so that only the first run has to wait for the compiler. The standalone program has no source on
    disk, and Numba keeps the code in its own cache folder of the user's. Numba goes by a kernel's own file
    to tell whether what it kept is still good. After a change to a kernel that a kernel of another module
    calls, empty that folder, or the calling kernel goes on running the old code.
    """
    if _NJIT is None:
        return function
    return _NJIT(cache=True)(function)
