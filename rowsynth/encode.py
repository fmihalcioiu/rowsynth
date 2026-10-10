"""Sample buffers to audio files: WAV as it is, and MP3 and Opus through ffmpeg."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .errors import RowSynthError
from .wav import save_wav

# The formats a file can be written in. A file of a format goes by that ending, as a rule.
FORMATS = ("wav", "mp3", "opus")

# The samples a second that Opus takes.
_OPUS_RATES = (8000, 12000, 16000, 24000, 48000)

# What ffmpeg is told of the encoder of each format it makes: MP3 at a variable bit rate of good quality, and
# Opus at a variable one of about 96 kbit/s.
_ENCODERS = {
    "mp3": ("-c:a", "libmp3lame", "-q:a", "2"),
    "opus": ("-c:a", "libopus", "-b:a", "96k", "-vbr", "on"),
}

# What keeps tags out of a file: none is carried over, and none is written that says which ffmpeg made it. So
# the same samples make the same bytes from run to run.
_NO_TAGS = ("-map_metadata", "-1", "-fflags", "+bitexact", "-flags:a", "+bitexact")


def unwritable(fmt: str, rate: int) -> str | None:
    """Why no file of the format `fmt` can be written at `rate`, or None when one can.

    It goes by the format and the rate alone, so it can be asked before there are samples to write: a format
    that is none of `FORMATS` cannot be written at all, and Opus only at the rates it takes. Whether ffmpeg
    is there to make the file is not looked into.
    """
    if fmt not in FORMATS:
        return f"unknown format {fmt!r}: a file is written as {', '.join(FORMATS[:-1])} or {FORMATS[-1]}"
    if fmt == "opus" and rate not in _OPUS_RATES:
        rates = f"{', '.join(map(str, _OPUS_RATES[:-1]))} or {_OPUS_RATES[-1]}"
        return f"opus takes a rate of {rates}, got {rate!r}"
    return None


def save_audio(path: str | Path, samples: np.ndarray | list | tuple, rate: int, fmt: str = "wav") -> None:
    """Write one channel (mono) or a pair (stereo) as a file of the format `fmt`.

    "wav" is the file `save_wav` writes. "mp3" and "opus" are made from that file by ffmpeg, which has to be
    on the path: the WAV goes into a temporary folder, ffmpeg writes its file beside it, and the folder is
    gone again whatever happens. The file is written at `path` as it is given, whatever its ending, and
    nothing is written there unless ffmpeg has made all of it.

    Refused: a format at a rate that `unwritable` has a reason against, which is a format that is none of the
    three and a rate that Opus does not take, a format that needs ffmpeg when none is found, samples or a
    rate that `save_wav` refuses, and a run of ffmpeg that ends in an error, with what it printed.
    """
    why = unwritable(fmt, rate)
    if why is not None:
        raise RowSynthError(why)
    if fmt == "wav":
        save_wav(path, samples, rate)
        return
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RowSynthError(f"ffmpeg is needed to write {fmt}, and none is found on the path")
    path = Path(path)
    with tempfile.TemporaryDirectory() as folder:
        source, made = Path(folder, "in.wav"), Path(folder, f"out.{fmt}")
        try:
            save_wav(source, samples, rate)
        except RowSynthError as error:
            # The message starts with the file in the temporary folder. The caller knows the file it asked for.
            raise RowSynthError(f"{path}{str(error).removeprefix(str(source))}") from None
        # A list of arguments, so no shell reads them. ffmpeg is given nothing to read but the file, and what
        # it prints is kept as text, which it prints only of an error.
        done = subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(source), *_NO_TAGS, *_ENCODERS[fmt], str(made)],
                              stdin=subprocess.DEVNULL, capture_output=True, encoding="utf-8", errors="replace")
        if done.returncode:
            raise RowSynthError(f"{path}: ffmpeg ended with code {done.returncode}: {done.stderr.strip()}")
        encoded = made.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
