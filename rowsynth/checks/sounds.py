"""The rules a sound is held to: its peak, its two ends, its offset, its brightness and its length."""
from __future__ import annotations

import numpy as np

from ..bank import PEAK_DB, RENDER_GAIN
from ..errors import RowSynthError
from ..measure import brightness, measure
from ..project import Project
from ..sound import KNEE, render_sound
from . import Finding

# A first or a last sample of this size or more is a jump out of silence or back into it, and clicks.
_STEP = 0.01

# The mean sample must sit closer to zero than this, in an entry with no "max_dc" of its own.
_DC = 0.002

# No more than this share of a sound's energy may lie above 4 kHz, in an entry with no "max_bright" of its own.
_BRIGHT = 0.3

# The samples in each frame the brightness is read from.
_FRAME = 2048


def _variants(name: str, entry: dict) -> list[tuple[str, dict]]:
    """The defs of an entry, each with the target its findings go by. A variant's target carries its index.

    An entry made of anything other than "def" or "defs" has none.
    """
    target = f"sounds.{name}"
    if "def" in entry:
        return [(target, entry["def"])]
    return [(f"{target}[{i}]", defn) for i, defn in enumerate(entry.get("defs", ()))]


def _levelled(samples: np.ndarray) -> np.ndarray:
    """The samples with the loudest of them at the bank's peak, in a new buffer. Silence is handed back as it is.

    Each sample is multiplied first and divided after, as a double, and stored as a float32.
    """
    top = float(np.abs(samples).max())
    if not top:
        return samples
    return (samples.astype(np.float64) * 10 ** (PEAK_DB / 20) / top).astype(np.float32)


def _breaches(target: str, entry: dict, samples: np.ndarray, rate: int) -> list[Finding]:
    """What one sound breaks of the rules. samples is the sound as the bank renders it, before it is levelled.

    The peak, the brightness and the length are read from samples as they are. The ends and the offset are
    read at the level the sound is played at, with its loudest sample at the bank's peak.
    """
    found = []

    def add(rule: str, message: str) -> None:
        found.append(Finding(rule, target, message))

    peak = float(np.abs(samples).max())
    # A sound that gets to the knee at the bank's gain no longer has the shape it was written with.
    if peak >= KNEE:
        add("peak", f"reaches {peak:.3f} at the bank's gain of {RENDER_GAIN}, and from {KNEE} up the output "
                    f"stage bends it")
    figures = measure(_levelled(samples), rate)
    if figures["head_step"] >= _STEP:
        add("ends", f"starts with a jump of {figures['head_step']} out of silence, and {_STEP} or more clicks")
    if figures["tail_step"] >= _STEP:
        add("ends", f"ends with a jump of {figures['tail_step']} into silence, and {_STEP} or more clicks")
    limit = entry.get("max_dc", _DC)
    if abs(figures["dc"]) >= limit:
        add("dc", f"its mean sample is {figures['dc']}, which is {limit} or more off zero")
    above = brightness(samples, _FRAME, rate)["above_4k"]
    limit = entry.get("max_bright", _BRIGHT)
    if above > limit:
        add("bright", f"{above} of its energy is above 4 kHz, which is more than {limit}")
    seconds = len(samples) / rate
    if "max_seconds" in entry and seconds > entry["max_seconds"]:
        add("length", f"lasts {seconds:.5g} seconds, which is over its limit of {entry['max_seconds']}")
    return found


def check(project: Project, rate: int) -> list[Finding]:
    """The findings on every sound of a project that is one def or a list of them, each def rendered once at `rate`.

    They come in the order of the sounds in the project, then of the variants of a sound, then of the rules:
    "peak", "ends" (the start before the end), "dc", "bright" and "length". A def that cannot be rendered is
    refused with its target in front.
    """
    found = []
    for name in project.sounds:
        entry = project.sound(name)
        for target, defn in _variants(name, entry):
            try:
                samples = render_sound(defn, rate, RENDER_GAIN)
            except RowSynthError as error:
                raise RowSynthError(f"{target}: {error}") from None
            found += _breaches(target, entry, samples, rate)
    return found
