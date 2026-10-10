"""Tests for rowsynth.encode."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from rowsynth.encode import FORMATS, save_audio, unwritable
from rowsynth.errors import RowSynthError
from rowsynth.wav import load_wav, save_wav

# A rate that MP3 and Opus both take.
RATE = 8000

# The tests that run ffmpeg are left out where there is none.
with_ffmpeg = unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is not on the path")

# What ffmpeg is told between the file it reads and the encoder: no tags, and nothing that changes from run to run.
NO_TAGS = ["-map_metadata", "-1", "-fflags", "+bitexact", "-flags:a", "+bitexact"]
ENCODERS = {"mp3": ["-c:a", "libmp3lame", "-q:a", "2"], "opus": ["-c:a", "libopus", "-b:a", "96k", "-vbr", "on"]}


def tone(hz: float = 440.0) -> np.ndarray:
    """A tenth of a second of a sine at half of full scale."""
    return (0.5 * np.sin(2 * np.pi * hz * np.arange(RATE // 10) / RATE)).astype(np.float32)


def frame_of(raw: bytes) -> bytes:
    """The four bytes that head the first frame of an MP3 file, past the ID3 tag of a file that starts with one."""
    if raw[:3] != b"ID3":
        return raw[:4]
    # The size of a tag is four bytes of seven bits each, and leaves out the ten bytes that head it.
    start = 10 + (raw[6] << 21 | raw[7] << 14 | raw[8] << 7 | raw[9])
    return raw[start:start + 4]


def sides_of(raw: bytes, fmt: str) -> int:
    """How many sides a file of MP3 or of Opus says it has, after holding it to the start of its container."""
    if fmt == "mp3":
        head = frame_of(raw)
        # Eleven set bits start a frame. The top two bits of its fourth byte are 3 for one side.
        assert head[0] == 0xFF and head[1] & 0xE0 == 0xE0, raw[:32]
        return 1 if head[3] >> 6 == 3 else 2
    # The first page of an Ogg file is 27 bytes and a table of one entry, and then the head of the Opus stream.
    assert raw[:4] == b"OggS" and raw[28:36] == b"OpusHead", raw[:44]
    return raw[37]


class EncodeCase(unittest.TestCase):
    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)

    def refused(self, path: Path, samples: object, rate: int, fmt: str) -> str:
        """What `save_audio` says as it refuses. Nothing is at the path afterwards."""
        with self.assertRaises(RowSynthError) as caught:
            save_audio(path, samples, rate, fmt)
        self.assertFalse(path.exists())
        return str(caught.exception)


class SaveAudioTest(EncodeCase):
    def test_the_formats_are_wav_mp3_and_opus_and_wav_is_the_file_save_wav_writes(self) -> None:
        self.assertEqual(FORMATS, ("wav", "mp3", "opus"))
        save_wav(self.folder / "wanted.wav", tone(), RATE)
        save_audio(self.folder / "left out.wav", tone(), RATE)
        save_audio(self.folder / "named.wav", tone(), RATE, "wav")
        wanted = (self.folder / "wanted.wav").read_bytes()
        self.assertEqual((self.folder / "left out.wav").read_bytes(), wanted)
        self.assertEqual((self.folder / "named.wav").read_bytes(), wanted)

    @with_ffmpeg
    def test_an_mp3_file_starts_with_an_id3_tag_or_a_frame_and_has_the_one_side_it_was_given(self) -> None:
        path = self.folder / "tone.mp3"
        save_audio(path, tone(), RATE, "mp3")
        raw = path.read_bytes()
        self.assertTrue(raw[:3] == b"ID3" or (raw[0] == 0xFF and raw[1] & 0xE0 == 0xE0), raw[:16])
        self.assertEqual(sides_of(raw, "mp3"), 1)

    @with_ffmpeg
    def test_an_opus_file_starts_with_an_ogg_page_and_says_its_one_side_and_its_rate(self) -> None:
        path = self.folder / "tone.opus"
        save_audio(path, tone(), RATE, "opus")
        raw = path.read_bytes()
        self.assertEqual(raw[:4], b"OggS")
        self.assertEqual(sides_of(raw, "opus"), 1)
        # The rate of what went in is four bytes, low byte first, twelve into the head of the stream.
        self.assertEqual(int.from_bytes(raw[40:44], "little"), RATE)

    @with_ffmpeg
    def test_two_sides_make_a_stereo_file(self) -> None:
        for fmt in ("mp3", "opus"):
            path = self.folder / f"pair.{fmt}"
            save_audio(path, (tone(440.0), tone(660.0)), RATE, fmt)
            self.assertEqual(sides_of(path.read_bytes(), fmt), 2, fmt)

    @with_ffmpeg
    def test_the_same_samples_written_twice_are_the_same_bytes_and_other_samples_are_not(self) -> None:
        for fmt in ("mp3", "opus"):
            first, second, other = (self.folder / f"{name}.{fmt}" for name in ("first", "second", "other"))
            save_audio(first, tone(), RATE, fmt)
            save_audio(second, tone(), RATE, fmt)
            save_audio(other, tone(660.0), RATE, fmt)
            self.assertEqual(first.read_bytes(), second.read_bytes(), fmt)
            self.assertNotEqual(first.read_bytes(), other.read_bytes(), fmt)

    @with_ffmpeg
    def test_a_file_is_written_at_its_path_whatever_its_ending_and_over_what_is_there(self) -> None:
        # The folders it is in are made for it.
        path = self.folder / "made" / "for it" / "tone.sound"
        save_audio(path, tone(), RATE, "opus")
        self.assertEqual(sides_of(path.read_bytes(), "opus"), 1)
        save_audio(path, tone(), RATE, "mp3")
        self.assertEqual(sides_of(path.read_bytes(), "mp3"), 1)
        self.assertEqual([found.name for found in path.parent.iterdir()], ["tone.sound"])

    def test_ffmpeg_is_run_without_a_shell_on_a_wav_file_of_the_samples_and_what_it_makes_is_the_file(self) -> None:
        for fmt in ("mp3", "opus"):
            seen = []

            def run(args: list, **how: object) -> subprocess.CompletedProcess:
                seen.append((list(args), how, load_wav(args[5])))
                Path(args[-1]).write_bytes(b"made by ffmpeg")
                return subprocess.CompletedProcess(args, 0, "", "")

            path = self.folder / f"tone.{fmt}"
            with mock.patch("shutil.which", return_value="found/ffmpeg") as which, \
                    mock.patch("subprocess.run", side_effect=run):
                save_audio(path, (tone(), tone(660.0)), RATE, fmt)
            which.assert_called_once_with("ffmpeg")
            ((args, how, (sides, rate)),) = seen
            source, made = Path(args[5]), Path(args[-1])
            self.assertEqual(args, ["found/ffmpeg", "-v", "error", "-y", "-i", str(source), *NO_TAGS, *ENCODERS[fmt],
                                    str(made)])
            self.assertEqual((source.suffix, made.suffix, made.parent), (".wav", f".{fmt}", source.parent))
            self.assertNotIn("shell", how)
            self.assertEqual((len(sides), rate, len(sides[0])), (2, RATE, RATE // 10))
            self.assertEqual(path.read_bytes(), b"made by ffmpeg")
            self.assertFalse(source.parent.exists())


class RefusalTest(EncodeCase):
    def test_a_format_that_is_none_of_the_three_is_refused(self) -> None:
        for fmt in ("flac", "WAV", ""):
            self.assertEqual(self.refused(self.folder / "tone.flac", tone(), RATE, fmt),
                             f"unknown format {fmt!r}: a file is written as wav, mp3 or opus")

    def test_mp3_and_opus_are_refused_when_no_ffmpeg_is_on_the_path_and_wav_is_written_all_the_same(self) -> None:
        with mock.patch("shutil.which", return_value=None) as which:
            for fmt in ("mp3", "opus"):
                self.assertEqual(self.refused(self.folder / f"tone.{fmt}", tone(), RATE, fmt),
                                 f"ffmpeg is needed to write {fmt}, and none is found on the path")
                which.assert_called_with("ffmpeg")
            save_audio(self.folder / "tone.wav", tone(), RATE, "wav")
        self.assertEqual([found.name for found in self.folder.iterdir()], ["tone.wav"])

    def test_opus_refuses_a_rate_it_does_not_take_and_says_which_it_takes(self) -> None:
        for rate in (44100, 11025):
            self.assertEqual(self.refused(self.folder / "tone.opus", tone(), rate, "opus"),
                             f"opus takes a rate of 8000, 12000, 16000, 24000 or 48000, got {rate}")

    def test_samples_that_cannot_be_a_wav_file_are_refused_by_the_path_asked_for_and_no_folder_is_left(self) -> None:
        temporary = self.folder / "temporary"
        temporary.mkdir()
        path = self.folder / "tone.mp3"
        with mock.patch("shutil.which", return_value="found/ffmpeg"), \
                mock.patch.object(tempfile, "tempdir", str(temporary)):
            self.assertEqual(self.refused(path, [], RATE, "mp3"), f"{path}: no samples to write")
            self.assertEqual(self.refused(path, (tone(), tone()[:5]), RATE, "mp3"),
                             f"{path}: the two channels differ in length, 800 and 5 samples")
        self.assertEqual(list(temporary.iterdir()), [])

    def test_a_run_of_ffmpeg_that_ends_in_an_error_is_refused_with_what_it_printed_and_no_folder_is_left(self) -> None:
        temporary = self.folder / "temporary"
        temporary.mkdir()
        folders = []

        def run(args: list, **how: object) -> subprocess.CompletedProcess:
            # The folder of the WAV file is there for as long as ffmpeg runs, and half a file is left in it.
            folders.append(Path(args[5]).parent)
            Path(args[-1]).write_bytes(b"half")
            return subprocess.CompletedProcess(args, 8, "", "Unknown encoder 'libopus'\nEncoder not found\n")

        path = self.folder / "tone.opus"
        with mock.patch("shutil.which", return_value="found/ffmpeg"), mock.patch("subprocess.run", side_effect=run), \
                mock.patch.object(tempfile, "tempdir", str(temporary)):
            said = self.refused(path, tone(), RATE, "opus")
        self.assertEqual(said, f"{path}: ffmpeg ended with code 8: Unknown encoder 'libopus'\nEncoder not found")
        self.assertEqual([folder.parent for folder in folders], [temporary])
        self.assertEqual(list(temporary.iterdir()), [])


class UnwritableTest(EncodeCase):
    def test_wav_and_mp3_at_any_rate_and_opus_at_a_rate_it_takes_have_nothing_against_them_ffmpeg_or_none(self) -> None:
        with mock.patch("shutil.which", return_value=None) as which:
            for fmt, rates in (("wav", (8000, 11025, 44100)), ("mp3", (8000, 11025, 44100)),
                               ("opus", (8000, 12000, 16000, 24000, 48000))):
                for rate in rates:
                    self.assertIsNone(unwritable(fmt, rate), (fmt, rate))
        which.assert_not_called()

    def test_an_unknown_format_and_opus_at_another_rate_have_the_reason_save_audio_refuses_them_with(self) -> None:
        for fmt, rate, why in (("flac", RATE, "unknown format 'flac': a file is written as wav, mp3 or opus"),
                               ("flac", 44100, "unknown format 'flac': a file is written as wav, mp3 or opus"),
                               ("opus", 44100, "opus takes a rate of 8000, 12000, 16000, 24000 or 48000, got 44100")):
            self.assertEqual(unwritable(fmt, rate), why)
            self.assertEqual(self.refused(self.folder / "tone", tone(), rate, fmt), why)


if __name__ == "__main__":
    unittest.main()
