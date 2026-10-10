"""The GUI's sound output: buffers rendered ahead of time, played through pygame's mixer."""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np
import pygame

from ..errors import RowSynthError

# Frames the mixer asks for at a time: about a hundredth of a second at 44100.
BUFFER = 512
# Seconds of sound behind the position that `Player.level` reads.
WINDOW = 0.03
# Full scale in 16 bits, as in a WAV file.
FULL = 32767.0


def to_pcm(sides: Sequence[np.ndarray]) -> np.ndarray:
    """One side or two as 16-bit frames, a row a frame, left then right. One side goes to both.

    Values past -1 to 1 are clamped, and a value that is not a number is silence.
    """
    try:
        rows = [np.asarray(side, dtype=np.float64) for side in sides]
    except (TypeError, ValueError):
        raise RowSynthError("the sides of a sound must be rows of numbers") from None
    for row in rows:
        if row.ndim != 1:
            raise RowSynthError(f"a side of a sound must be one row of numbers, got shape {row.shape}")
    if len(rows) not in (1, 2):
        raise RowSynthError(f"a sound has one side or two, left and right, got {len(rows)}")
    if len(rows[0]) != len(rows[-1]):
        raise RowSynthError(f"the two sides differ in length, {len(rows[0])} and {len(rows[1])} samples")
    if len(rows[0]) == 0:
        raise RowSynthError("no samples to play")
    # The sum `rowsynth.wav.save_wav` does, written out again: there it sits inside the function that
    # writes the file. A test holds the two together, so what is heard is what a WAV file would hold.
    frames = np.clip(np.stack([rows[0], rows[-1]], axis=1), -1.0, 1.0)
    frames[np.isnan(frames)] = 0.0
    return np.floor(frames * FULL + 0.5).astype(np.int16)


class Player:
    """Plays one sound at a time, once or round and round, and says how far into it the clock is."""

    def __init__(self, rate: int = 44100, clock: Callable[[], float] = time.perf_counter) -> None:
        """Opens the mixer at `rate`, 16-bit, two channels. `clock` gives the time in seconds."""
        if isinstance(rate, bool) or not isinstance(rate, (int, np.integer)) or rate < 1:
            raise RowSynthError(f"the rate must be an integer of 1 or more, got {rate!r}")
        # A mixer that is open already (pygame.init opens one) keeps its own rate until it is shut.
        pygame.mixer.quit()
        try:
            # With no changes allowed the mixer takes this format as it is and converts for the sound
            # card itself, so the frames are played at the rate they were rendered at.
            pygame.mixer.init(frequency=int(rate), size=-16, channels=2, buffer=BUFFER, allowedchanges=0)
        except (pygame.error, OverflowError) as error:
            raise RowSynthError(f"the sound output cannot be opened at {rate} Hz: {error}") from None
        self._rate = int(rate)
        self._clock = clock
        self._channel = pygame.mixer.Channel(0)
        # The frames of the sound that was played, until it is stopped. The origin is the moment by the
        # clock at which it began, or would have begun had it always been the render it is now.
        self._frames: np.ndarray | None = None
        self._loop = False
        self._length = 0.0
        self._origin = 0.0

    def close(self) -> None:
        """Stops the sound and shuts the mixer down."""
        self._frames = None
        pygame.mixer.quit()

    def _send(self, frames: np.ndarray, loop: bool) -> None:
        """Hands frames to the mixer, which drops what it was playing and starts on these."""
        self._channel.play(pygame.mixer.Sound(buffer=frames.tobytes()), loops=-1 if loop else 0)

    def play(self, sides: Sequence[np.ndarray], loop: bool = False) -> None:
        """Plays one side (heard on both) or two, in place of whatever was playing.

        A loop goes round without a gap until it is stopped.
        """
        frames = to_pcm(sides)
        self._send(frames, loop)
        self._frames, self._loop = frames, loop
        self._length = len(frames) / self._rate
        self._origin = self._clock()

    def stop(self) -> None:
        """Stops the sound. The position goes back to 0."""
        self._frames = None
        self._channel.stop()

    @property
    def playing(self) -> bool:
        """Whether a sound is going: a loop until it is stopped, a sound played once until its length has passed."""
        return self._frames is not None and (self._loop or self._clock() - self._origin < self._length)

    def position(self) -> float:
        """Seconds into the sound, by the clock: 0 when nothing was played or it was stopped.

        A loop wraps at its length. A sound played once stays at its length when it has ended.
        """
        if self._frames is None:
            return 0.0
        elapsed = self._clock() - self._origin
        return elapsed % self._length if self._loop else min(elapsed, self._length)

    def swap(self, sides: Sequence[np.ndarray]) -> None:
        """Puts a new render in place of a playing loop, going on from the place the old one had reached.

        When the new render has another length, the place is carried over as a share of the length.
        With no loop playing it does nothing.
        """
        if self._frames is None or not self._loop:
            return
        frames = to_pcm(sides)
        now = self._clock()
        length = len(frames) / self._rate
        # The rounds gone by, whole ones and the share of this one.
        turns = (now - self._origin) / self._length
        if length != self._length:
            self._origin = now - turns * length
        at = int(turns % 1.0 * len(frames)) % len(frames)
        # A Sound always starts at its first frame, and a Sound queued behind another is played once
        # and not again: the tail of the new render with the whole of it queued behind would stop after
        # one round unless something kept queueing. So the new render is turned round to begin at the
        # place reached, and that is looped. Frame for frame it is the tail and then the whole render
        # over and over, and nothing has to be kept fed.
        # What it costs: one more copy of the frames at each swap. The old sound is cut off and the new
        # one starts with the mixer's next block, so the two meet up to a block out of step, with a
        # click if the wave is not the same there. And the place is the clock's, not the sound card's,
        # so it is off by whatever the card delays.
        self._send(np.roll(frames, -at, axis=0), True)
        self._frames, self._length = frames, length

    def level(self) -> tuple[float, float]:
        """The loudest sample of the left side and of the right, 0 to 1, over the stretch just played.

        The stretch is the last `WINDOW` seconds up to the position, read from the frames and the clock.
        In a loop it goes back round the end, but never to before the sound began. 0 when nothing sounds.
        """
        if self._frames is None:
            return 0.0, 0.0
        elapsed = self._clock() - self._origin
        if not self._loop and elapsed >= self._length:
            return 0.0, 0.0
        count = len(self._frames)
        last = min(int(elapsed % self._length * self._rate), count - 1)
        behind = min(int(min(WINDOW, elapsed) * self._rate), count - 1)
        stretch = self._frames[np.arange(last - behind, last + 1) % count]
        left, right = np.abs(stretch).max(axis=0) / FULL
        return float(left), float(right)
