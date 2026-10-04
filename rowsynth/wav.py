"""Sample buffers to 16-bit PCM WAV files, and back."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .errors import RowSynthError

# The 44 bytes ahead of the samples: the RIFF size, a 16-byte format chunk
# (kind, channels, rate, bytes a second, bytes a frame, bits a sample) and the
# size of the data.
_HEADER = struct.Struct("<4sI4s4sIHHIIHH4sI")
_CHUNK = struct.Struct("<4sI")
_FORMAT = struct.Struct("<HHIIHH")
_PCM = 1
_BITS = 16
_FULL = 32767.0


def _channels(path: Path, samples: np.ndarray | list | tuple) -> list[np.ndarray]:
    """One channel or two out of `samples`, as float64 arrays of one length."""
    if isinstance(samples, np.ndarray):
        paired = samples.ndim == 2
    else:
        paired = (isinstance(samples, (list, tuple)) and len(samples) > 0
                  and isinstance(samples[0], (list, tuple, np.ndarray)))
    rows = list(samples) if paired else [samples]
    if paired and len(rows) != 2:
        raise RowSynthError(f"{path}: stereo is two channels, left and right, got {len(rows)}")
    try:
        channels = [np.asarray(row, dtype=np.float64) for row in rows]
    except (TypeError, ValueError):
        raise RowSynthError(f"{path}: the samples must all be numbers") from None
    for channel in channels:
        if channel.ndim != 1:
            raise RowSynthError(f"{path}: a channel must be one row of numbers, got shape {channel.shape}")
    if len(channels[0]) != len(channels[-1]):
        raise RowSynthError(f"{path}: the two channels differ in length, "
                            f"{len(channels[0])} and {len(channels[1])} samples")
    if len(channels[0]) == 0:
        raise RowSynthError(f"{path}: no samples to write")
    return channels


def save_wav(path: str | Path, samples: np.ndarray | list | tuple, rate: int = 44100) -> None:
    """Write one channel (mono) or a pair (stereo). Values past -1..1 are clamped."""
    path = Path(path)
    channels = _channels(path, samples)
    if not isinstance(rate, (int, np.integer)) or rate < 1:
        raise RowSynthError(f"{path}: the rate must be an integer of 1 or more, got {rate!r}")
    count = len(channels)
    # One row a frame, so the bytes come out left, right, left, right.
    frames = np.clip(np.stack(channels, axis=1), -1.0, 1.0)
    # A sample that is not a number is written as silence.
    frames[np.isnan(frames)] = 0.0
    data = np.floor(frames * _FULL + 0.5).astype("<i2").tobytes()
    header = _HEADER.pack(b"RIFF", 36 + len(data), b"WAVE", b"fmt ", 16, _PCM, count, rate,
                          rate * count * 2, count * 2, _BITS, b"data", len(data))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + data)


def load_wav(path: str | Path) -> tuple[list[np.ndarray], int]:
    """Read a 16-bit PCM file, mono or stereo: its channels as float32 in -1..1, and its rate."""
    path = Path(path)
    raw = path.read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise RowSynthError(f"{path}: not a RIFF/WAVE file")
    count = rate = 0
    at = 12
    while at + 8 <= len(raw):
        tag, size = _CHUNK.unpack_from(raw, at)
        at += 8
        if at + size > len(raw):
            name = tag.decode("latin-1").strip()
            raise RowSynthError(f"{path}: cut short, the {name} chunk says {size} bytes and {len(raw) - at} follow")
        if tag == b"fmt ":
            if size < _FORMAT.size:
                raise RowSynthError(f"{path}: cut short, the fmt chunk is {size} bytes of {_FORMAT.size}")
            kind, count, rate, _, _, bits = _FORMAT.unpack_from(raw, at)
            if kind != _PCM:
                raise RowSynthError(f"{path}: not PCM, its format is number {kind}")
            if bits != _BITS:
                raise RowSynthError(f"{path}: not 16-bit, it has {bits} bits a sample")
            if count not in (1, 2):
                raise RowSynthError(f"{path}: {count} channels, only mono and stereo are read")
        elif tag == b"data":
            if count == 0:
                raise RowSynthError(f"{path}: the data chunk comes before any fmt chunk")
            if size % (count * 2):
                raise RowSynthError(f"{path}: cut short, {size} bytes of data do not fill whole frames")
            frames = np.frombuffer(raw, dtype="<i2", count=size // 2, offset=at).reshape(-1, count)
            return [(frames[:, c] / _FULL).astype(np.float32) for c in range(count)], rate
        # A chunk of odd size has one byte of padding after it.
        at += size + (size & 1)
    raise RowSynthError(f"{path}: cut short, it ends before any data chunk")
