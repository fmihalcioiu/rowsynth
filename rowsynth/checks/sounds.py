"""The rules a sound is held to: its peak, ends, offset, brightness, length and notes, and a held sound's joins."""
from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from ..bank import PEAK_DB, RENDER_GAIN, held_parts, render_entry
from ..measure import brightness, measure, pitch_at
from ..notes import hz, midi
from ..project import Project
from ..sound import KNEE, render_sound
from . import Finding, _at

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

# A join of a held sound clicks when the step across it is this share of the largest step inside the loop, or more.
_JOIN = 0.25


@dataclass(frozen=True)
class _Render:
    """One buffer to hold to the rules: the target its findings go by, what it answers to, and its samples.

    entry is the entry whose limits and "check" hold it. samples is the buffer as it is made, before its
    loudest sample is brought to the bank's peak. The switches say which rules hold it: peak the "peak" rule,
    head and tail the "ends" rule at its first and at its last sample, and dc the "dc" rule. The rules with no
    switch hold every render.
    """

    target: str
    entry: dict
    samples: np.ndarray
    peak: bool = True
    head: bool = True
    tail: bool = True
    dc: bool = True


def _findings(target: str, entry: dict, rate: int) -> Iterator[Finding]:
    """The findings on one entry: on each buffer it is held to the rules by, in turn, and then on the whole of it.

    One def gives one buffer, rendered as the bank renders it before it levels it. Variants give one each, with
    the index of the variant on its target. Parts give every part first, as an entry of its own with the target
    "TARGET.parts[K]", and then the one buffer the bank adds them up into. A held sound gives its three parts as
    the bank cuts them before it levels them, "TARGET.start", "TARGET.loop" and "TARGET.end", and then what
    `_joins` finds between the start and the loop.
    """
    def held_to(place: str, samples: np.ndarray, **switches: bool) -> list[Finding]:
        return _breaches(_Render(place, entry, samples, **switches), rate)

    def one(place: str, defn: dict) -> list[Finding]:
        with _at(place):
            samples = render_sound(defn, rate, RENDER_GAIN)
        return held_to(place, samples)

    if "def" in entry:
        yield from one(target, entry["def"])
    elif "defs" in entry:
        for i, defn in enumerate(entry["defs"]):
            yield from one(f"{target}[{i}]", defn)
    elif "parts" in entry:
        for k, (_, part) in enumerate(entry["parts"]):
            yield from _findings(f"{target}.parts[{k}]", part, rate)
        with _at(target):
            whole = render_entry(entry, rate)[0]
        # The whole is a sum of parts that each went through the output stage alone and were levelled after.
        # It was never one sound in front of the knee, and its size says nothing about the knee.
        yield from held_to(target, whole, peak=False)
    elif "held" in entry:
        with _at(target):
            start, loop, end = held_parts(entry, rate)
        # A held sound touches silence twice: before its start and after its end. Its other three ends lie
        # against sound, and `_joins` holds them, and with them the offset of the start and the loop as one.
        if len(start):
            yield from held_to(f"{target}.start", start, tail=False, dc=False)
        # With no start it is the loop that comes out of silence, and its first sample is held as a start's is.
        yield from held_to(f"{target}.loop", loop, head=not len(start), tail=False, dc=False)
        yield from held_to(f"{target}.end", end)
        yield from _joins(target, entry, start, loop, rate)


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
    if render.head and figures["head_step"] >= _STEP:
        add("ends", f"starts with a jump of {figures['head_step']} out of silence, and {_STEP} or more clicks")
    if render.tail and figures["tail_step"] >= _STEP:
        add("ends", f"ends with a jump of {figures['tail_step']} into silence, and {_STEP} or more clicks")
    limit = entry.get("max_dc", _DC)
    if render.dc and abs(figures["dc"]) >= limit:
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


def _joins(target: str, entry: dict, start: np.ndarray, loop: np.ndarray, rate: int) -> list[Finding]:
    """What a held sound breaks of the rules on its start and its loop taken together: its joins and their offset.

    A join is where one buffer takes over from another while the sound goes on: the last sample of the start is
    followed by the first of the loop, and the last of the loop by its first as it comes round. The step across
    a join is set against the largest step between two neighbours inside the loop. A flat loop has no step
    inside it, and there a join clicks if it steps at all. With no start there is one join, and what comes
    before the loop is silence, which is for the "ends" rule. The offset is that of the start and the loop as
    one buffer, levelled as one.
    """
    found = []

    def add(rule: str, message: str) -> None:
        found.append(Finding(rule, target, message))

    wave = loop.astype(np.float64)
    inside = float(np.abs(np.diff(wave)).max(initial=0.0))
    joins = []
    if len(start):
        joins.append(("from its start into its loop", abs(wave[0] - float(start[-1]))))
    joins.append(("from the end of its loop round to its beginning", abs(wave[0] - wave[-1])))
    for where, step in joins:
        if inside and step >= _JOIN * inside:
            add("join", f"steps {where} by {step / inside:.3f} of the largest step inside the loop, and {_JOIN} "
                        f"or more clicks")
        elif step and not inside:
            add("join", f"steps {where} by {step:.3g}, and the loop is flat: any step into it clicks")
    dc = measure(_levelled(np.concatenate((start, loop))), rate)["dc"]
    limit = entry.get("max_dc", _DC)
    if abs(dc) >= limit:
        add("dc", f"the mean sample of its start and its loop together is {dc}, which is {limit} or more off zero")
    return found


def check(project: Project, rate: int) -> list[Finding]:
    """The findings on every sound of a project, rendered at `rate`.

    They come in the order of the sounds in the project, then of the buffers a sound is checked by (its
    variants, its parts and then the whole of them, or the start, the loop and the end of a held sound), then
    of the rules: "peak", "ends" (the start before the end), "dc", "bright", "length" and "pitch" (one note of
    "check" after the other). After the three buffers of a held sound come the findings on the sound itself:
    "join" (the start into the loop before the loop coming round) and the "dc" of its start and its loop
    together. The whole of an entry of parts is not held to "peak". Of a held sound only the end is held to
    every rule: the start is not held to "ends" at its last sample, the loop at neither, and neither of the
    two to "dc" on its own. A start of no samples is no buffer, and the loop after it is held to "ends" at its
    first sample. What cannot be rendered or cut is refused with its target in front.
    """
    found = []
    for name in project.sounds:
        found += _findings(f"sounds.{name}", project.sound(name), rate)
    return found
