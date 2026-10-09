"""The rules a sound is held to: its peak, its two ends, its offset, its brightness, its length and its notes."""
from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from ..bank import PEAK_DB, RENDER_GAIN, render_entry
from ..errors import RowSynthError
from ..measure import brightness, measure, pitch_at
from ..notes import hz, midi
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

# A note read back from a sound may be this many cents away from the note written for it, and no more.
_CENTS = 35

# The lowest and the highest pitch that is looked for where a note is read back, in Hz.
_LOW, _HIGH = 60.0, 4000.0


@dataclass(frozen=True)
class _Render:
    """One buffer to hold to the rules: the target its findings go by, what it answers to, and its samples.

    entry is the entry whose limits and "check" hold it. samples is the buffer as it is made, before its
    loudest sample is brought to the bank's peak. peak says whether the "peak" rule holds it.
    """

    target: str
    entry: dict
    samples: np.ndarray
    peak: bool = True


@contextmanager
def _at(target: str) -> Iterator[None]:
    """What is refused inside is refused with the target in front."""
    try:
        yield
    except RowSynthError as error:
        raise RowSynthError(f"{target}: {error}") from None


def _renders(target: str, entry: dict, rate: int) -> Iterator[_Render]:
    """The buffers an entry is held to the rules by, in the order they are checked.

    One def gives one, rendered as the bank renders it before it levels it. Variants give one each, with the
    index of the variant on its target. Parts give every part first, as an entry of its own with the target
    "TARGET.parts[K]", and then the one buffer the bank adds them up into. An entry of any other kind gives none.
    """
    def one(place: str, defn: dict) -> _Render:
        with _at(place):
            return _Render(place, entry, render_sound(defn, rate, RENDER_GAIN))

    if "def" in entry:
        yield one(target, entry["def"])
    elif "defs" in entry:
        for i, defn in enumerate(entry["defs"]):
            yield one(f"{target}[{i}]", defn)
    elif "parts" in entry:
        for k, (_, part) in enumerate(entry["parts"]):
            yield from _renders(f"{target}.parts[{k}]", part, rate)
        with _at(target):
            whole = render_entry(entry, rate)[0]
        # The whole is a sum of parts that each went through the output stage alone and were levelled after.
        # It was never one sound in front of the knee, and its size says nothing about the knee.
        yield _Render(target, entry, whole, peak=False)


def _levelled(samples: np.ndarray) -> np.ndarray:
    """The samples with the loudest of them at the bank's peak, in a new buffer. Silence is handed back as it is.

    Each sample is multiplied first and divided after, as a double, and stored as a float32.
    """
    top = float(np.abs(samples).max())
    if not top:
        return samples
    return (samples.astype(np.float64) * 10 ** (PEAK_DB / 20) / top).astype(np.float32)


def _breaches(render: _Render, rate: int) -> list[Finding]:
    """What one render breaks of the rules that hold it.

    The peak, the brightness, the length and the notes are read from its samples as they are. The ends and the
    offset are read at the level a sound is played at, with its loudest sample at the bank's peak.
    """
    entry, samples = render.entry, render.samples
    found = []

    def add(rule: str, message: str) -> None:
        found.append(Finding(rule, render.target, message))

    top = float(np.abs(samples).max())
    # A sound that gets to the knee at the bank's gain no longer has the shape it was written with.
    if render.peak and top >= KNEE:
        add("peak", f"reaches {top:.3f} at the bank's gain of {RENDER_GAIN}, and from {KNEE} up the output "
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
    for mark in entry.get("check", ()):
        at, note = mark["at"], mark["note"]
        want = hz(midi(note))
        got = pitch_at(samples, at, _LOW, _HIGH, rate=rate)
        if not got:
            add("pitch", f"has no note to be read at {at} seconds, where the note {note} is written")
            continue
        cents = 1200 * math.log2(got / want)
        if abs(cents) > _CENTS:
            add("pitch", f"is {abs(cents):.0f} cents {'above' if cents > 0 else 'below'} the note {note} written "
                         f"at {at} seconds, and more than {_CENTS} is out of tune")
    return found


def check(project: Project, rate: int) -> list[Finding]:
    """The findings on every sound of a project that is one def, a list of them or parts, rendered at `rate`.

    They come in the order of the sounds in the project, then of the buffers a sound is checked by (its
    variants, or its parts and then the whole of them), then of the rules: "peak", "ends" (the start before
    the end), "dc", "bright", "length" and "pitch" (one note of "check" after the other). The whole of an
    entry of parts is not held to "peak". What cannot be rendered is refused with its target in front.
    """
    found = []
    for name in project.sounds:
        for render in _renders(f"sounds.{name}", project.sound(name), rate):
            found += _breaches(render, rate)
    return found
