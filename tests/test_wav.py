"""Tests for rowsynth.wav."""
from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rowsynth.errors import RowSynthError
from rowsynth.wav import load_wav, save_wav

HEADER = "<4sI4s4sIHHIIHH4sI"
STEP = 1 / 32767


def chunk(tag: bytes, body: bytes) -> bytes:
    """A RIFF chunk: tag, size, bytes, and a byte of padding after an odd size."""
    return tag + struct.pack("<I", len(body)) + body + b"\x00" * (len(body) & 1)


def fmt(kind: int = 1, channels: int = 1, rate: int = 8000, bits: int = 16) -> bytes:
    """A 16-byte format chunk."""
    width = channels * bits // 8
    return chunk(b"fmt ", struct.pack("<HHIIHH", kind, channels, rate, rate * width, width, bits))


def data(*values: int) -> bytes:
    """A data chunk of 16-bit values."""
    return chunk(b"data", struct.pack(f"<{len(values)}h", *values))


def riff(*chunks: bytes) -> bytes:
    """A whole file out of its chunks."""
    body = b"WAVE" + b"".join(chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


class WavCase(unittest.TestCase):
    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)

    def written(self, samples, rate: int = 8000) -> list[int]:
        """The 16-bit values `save_wav` puts after the header."""
        path = self.folder / "written.wav"
        save_wav(path, samples, rate)
        return np.frombuffer(path.read_bytes(), dtype="<i2", offset=44).tolist()

    def refused(self, reason: str, call, path: Path, *rest) -> None:
        """`call` raises the package's error, and the message names the file and gives `reason`."""
        with self.assertRaises(RowSynthError) as caught:
            call(path, *rest)
        self.assertIn(path.name, str(caught.exception))
        self.assertIn(reason, str(caught.exception))


class SaveWavTest(WavCase):
    def test_header_fields_of_a_mono_file(self) -> None:
        path = self.folder / "mono.wav"
        save_wav(path, np.zeros(5, dtype=np.float32), 22050)
        raw = path.read_bytes()
        self.assertEqual(len(raw), 44 + 10)
        self.assertEqual(struct.unpack(HEADER, raw[:44]),
                         (b"RIFF", 36 + 10, b"WAVE", b"fmt ", 16, 1, 1, 22050, 44100, 2, 16, b"data", 10))

    def test_header_fields_of_a_stereo_file(self) -> None:
        path = self.folder / "stereo.wav"
        save_wav(path, [np.zeros(7, dtype=np.float32), np.zeros(7, dtype=np.float32)], 48000)
        raw = path.read_bytes()
        self.assertEqual(len(raw), 44 + 28)
        self.assertEqual(struct.unpack(HEADER, raw[:44]),
                         (b"RIFF", 36 + 28, b"WAVE", b"fmt ", 16, 1, 2, 48000, 192000, 4, 16, b"data", 28))

    def test_rate_is_44100_when_none_is_given(self) -> None:
        path = self.folder / "rate.wav"
        save_wav(path, [0.0, 0.5])
        self.assertEqual(struct.unpack(HEADER, path.read_bytes()[:44])[7], 44100)

    def test_bytes_of_a_mono_file(self) -> None:
        path = self.folder / "mono.wav"
        save_wav(path, np.array([0.0, 1.0, -1.0, 0.25, -0.25], dtype=np.float32), 8000)
        self.assertEqual(path.read_bytes(), bytes.fromhex(
            "52 49 46 46 2e 00 00 00 57 41 56 45 66 6d 74 20 10 00 00 00 01 00 01 00"
            "40 1f 00 00 80 3e 00 00 02 00 10 00 64 61 74 61 0a 00 00 00"
            "00 00 ff 7f 01 80 00 20 00 e0"))

    def test_bytes_of_a_stereo_file_go_left_then_right(self) -> None:
        path = self.folder / "stereo.wav"
        save_wav(path, [np.array([0.0, 0.5, 1.0]), np.array([-0.5, 1.5, -1.0])], 11025)
        self.assertEqual(path.read_bytes(), bytes.fromhex(
            "52 49 46 46 30 00 00 00 57 41 56 45 66 6d 74 20 10 00 00 00 01 00 02 00"
            "11 2b 00 00 44 ac 00 00 04 00 10 00 64 61 74 61 0c 00 00 00"
            "00 00 01 c0 00 40 ff 7f ff 7f 01 80"))

    def test_values_past_full_scale_are_clamped(self) -> None:
        loud = np.array([1.0000001, 2.0, 1e9, np.inf, -1.0000001, -2.0, -1e9, -np.inf])
        self.assertEqual(self.written(loud), [32767] * 4 + [-32767] * 4)

    def test_a_half_rounds_up(self) -> None:
        self.assertEqual(self.written(np.array([0.5, -0.5], dtype=np.float32)), [16384, -16383])
        halves = [(k + 0.5) / 32767 for k in (0, 1, 2, -1, -2, -3)]
        self.assertEqual(self.written(halves), [1, 2, 3, 0, -1, -2])

    def test_not_a_number_is_written_as_silence(self) -> None:
        self.assertEqual(self.written(np.array([np.nan, 0.3, np.nan, -0.3])), [0, 9830, 0, -9830])

    def test_a_list_of_numbers_is_mono(self) -> None:
        values = [0.0, 0.1, -0.2, 0.3]
        self.assertEqual(self.written(values), self.written(np.array(values)))
        self.assertEqual(self.written(tuple(values)), self.written(np.array(values)))

    def test_float32_and_float64_samples_write_the_same(self) -> None:
        values = np.linspace(-1.2, 1.2, 97).astype(np.float32)
        self.assertEqual(self.written(values), self.written(values.astype(np.float64)))

    def test_every_way_of_giving_two_channels_writes_the_same(self) -> None:
        left = np.linspace(-1.0, 1.0, 9)
        right = np.linspace(0.5, -0.5, 9)
        want = self.written([left, right])
        self.assertEqual(want[0::2], self.written(left))
        self.assertEqual(want[1::2], self.written(right))
        self.assertEqual(self.written((left, right)), want)
        self.assertEqual(self.written(np.stack([left, right])), want)
        self.assertEqual(self.written([left.tolist(), right.tolist()]), want)

    def test_parent_folders_are_created(self) -> None:
        path = self.folder / "a" / "b" / "deep.wav"
        save_wav(path, [0.0, 0.5], 8000)
        self.assertTrue(path.is_file())

    def test_path_may_be_a_string(self) -> None:
        path = self.folder / "text.wav"
        save_wav(str(path), [0.0, 0.5], 8000)
        channels, rate = load_wav(str(path))
        self.assertEqual((len(channels), len(channels[0]), rate), (1, 2, 8000))

    def test_an_old_file_is_replaced(self) -> None:
        path = self.folder / "again.wav"
        save_wav(path, np.zeros(50), 8000)
        save_wav(path, np.zeros(3), 8000)
        self.assertEqual(len(path.read_bytes()), 44 + 6)

    def test_three_channels_are_refused(self) -> None:
        path = self.folder / "three.wav"
        self.refused("two channels", save_wav, path, [np.zeros(4), np.zeros(4), np.zeros(4)])

    def test_one_channel_in_a_list_is_refused(self) -> None:
        path = self.folder / "one.wav"
        self.refused("two channels", save_wav, path, [np.zeros(4)])

    def test_channels_of_unequal_length_are_refused(self) -> None:
        path = self.folder / "unequal.wav"
        self.refused("differ in length, 4 and 5", save_wav, path, [np.zeros(4), np.zeros(5)])
        self.refused("differ in length, 2 and 1", save_wav, path, [[0.0, 0.1], [0.2]])

    def test_a_2d_array_that_is_not_two_rows_is_refused(self) -> None:
        path = self.folder / "rows.wav"
        for shape in ((3, 4), (1, 4), (4, 2), (0, 4)):
            with self.subTest(shape=shape):
                self.refused("two channels", save_wav, path, np.zeros(shape))

    def test_a_channel_that_is_not_one_row_is_refused(self) -> None:
        path = self.folder / "shape.wav"
        for samples in (np.zeros((2, 3, 4)), np.float32(0.5), 0.5, [np.zeros((2, 2)), np.zeros((2, 2))]):
            with self.subTest(samples=samples):
                self.refused("one row of numbers", save_wav, path, samples)

    def test_empty_samples_are_refused(self) -> None:
        path = self.folder / "empty.wav"
        for samples in ([], (), np.zeros(0, dtype=np.float32), [np.zeros(0), np.zeros(0)], np.zeros((2, 0))):
            with self.subTest(samples=samples):
                self.refused("no samples", save_wav, path, samples)

    def test_samples_that_are_not_numbers_are_refused(self) -> None:
        path = self.folder / "words.wav"
        for samples in (["a", "b"], "ab", {"a": 1.0}, [[0.0, 0.1], ["a", "b"]], [0.0, [0.1, 0.2]]):
            with self.subTest(samples=samples):
                self.refused("must all be numbers", save_wav, path, samples)

    def test_a_rate_below_one_or_not_an_integer_is_refused(self) -> None:
        path = self.folder / "speed.wav"
        for rate in (0, -8000, 8000.5, "8000"):
            with self.subTest(rate=rate):
                self.refused("the rate must be an integer of 1 or more", save_wav, path, [0.0, 0.5], rate)

    def test_a_refused_save_writes_nothing(self) -> None:
        path = self.folder / "new" / "none.wav"
        self.refused("no samples", save_wav, path, [])
        self.refused("the rate must be", save_wav, path, [0.0, 0.5], 0)
        self.assertFalse(path.parent.exists())


class LoadWavTest(WavCase):
    def load(self, raw: bytes, name: str = "made.wav") -> tuple[list[np.ndarray], int]:
        path = self.folder / name
        path.write_bytes(raw)
        return load_wav(path)

    def bad(self, reason: str, raw: bytes) -> None:
        """A file of these bytes is refused, and the message names it and gives `reason`."""
        path = self.folder / "bad.wav"
        path.write_bytes(raw)
        self.refused(reason, load_wav, path)

    def test_a_mono_save_comes_back_within_one_step(self) -> None:
        path = self.folder / "mono.wav"
        wave = (0.9 * np.sin(np.arange(400) * 0.37)).astype(np.float32)
        save_wav(path, wave, 11025)
        channels, rate = load_wav(path)
        self.assertEqual(rate, 11025)
        self.assertEqual(len(channels), 1)
        self.assertEqual(channels[0].dtype, np.float32)
        self.assertEqual(channels[0].shape, wave.shape)
        self.assertLess(float(np.max(np.abs(channels[0] - wave))), STEP)

    def test_a_stereo_save_comes_back_within_one_step(self) -> None:
        path = self.folder / "stereo.wav"
        left = (0.8 * np.sin(np.arange(300) * 0.21)).astype(np.float32)
        right = np.linspace(-1.0, 1.0, 300).astype(np.float32)
        save_wav(path, [left, right], 48000)
        channels, rate = load_wav(path)
        self.assertEqual(rate, 48000)
        self.assertEqual(len(channels), 2)
        for got, want in zip(channels, (left, right)):
            self.assertEqual(got.dtype, np.float32)
            self.assertEqual(got.shape, want.shape)
            self.assertLess(float(np.max(np.abs(got - want))), STEP)

    def test_clamped_values_come_back_as_full_scale(self) -> None:
        path = self.folder / "loud.wav"
        save_wav(path, [3.0, -3.0, 1.0, -1.0], 8000)
        channels, _ = load_wav(path)
        self.assertEqual(channels[0].tolist(), [1.0, -1.0, 1.0, -1.0])

    def test_values_are_the_16_bit_numbers_over_32767(self) -> None:
        values = (0, 1, -1, 16384, -16383, 32767, -32767)
        channels, rate = self.load(riff(fmt(rate=22050), data(*values)))
        self.assertEqual(rate, 22050)
        want = np.array([v / 32767 for v in values], dtype=np.float32)
        self.assertEqual(channels[0].tolist(), want.tolist())

    def test_stereo_frames_split_into_left_and_right(self) -> None:
        channels, _ = self.load(riff(fmt(channels=2), data(100, -100, 200, -200, 300, -300)))
        self.assertEqual([np.rint(c * 32767).tolist() for c in channels], [[100, 200, 300], [-100, -200, -300]])

    def test_loaded_channels_can_be_written_to(self) -> None:
        channels, _ = self.load(riff(fmt(channels=2), data(100, -100, 200, -200)))
        for channel in channels:
            channel *= 0.5
        self.assertAlmostEqual(float(channels[1][1]), -100 / 32767, places=6)

    def test_chunks_before_the_data_are_skipped(self) -> None:
        raw = riff(chunk(b"JUNK", b"\x00" * 6), fmt(), chunk(b"LIST", b"odd"), chunk(b"fact", b"\x01\x02\x03\x04"),
                   data(5, -5, 7), chunk(b"LIST", b"after"))
        channels, rate = self.load(raw)
        self.assertEqual(rate, 8000)
        self.assertEqual(np.rint(channels[0] * 32767).tolist(), [5, -5, 7])

    def test_a_longer_fmt_chunk_is_read(self) -> None:
        long_fmt = chunk(b"fmt ", struct.pack("<HHIIHHH", 1, 1, 8000, 16000, 2, 16, 0))
        channels, rate = self.load(riff(long_fmt, data(9, -9)))
        self.assertEqual(rate, 8000)
        self.assertEqual(np.rint(channels[0] * 32767).tolist(), [9, -9])

    def test_a_file_that_is_not_riff_wave_is_refused(self) -> None:
        good = riff(fmt(), data(1, 2, 3))
        for raw in (b"", b"RIFF", b"not a wave file at all, only some text", b"RIFX" + good[4:],
                    good[:8] + b"AVI " + good[12:]):
            with self.subTest(raw=raw[:12]):
                self.bad("not a RIFF/WAVE file", raw)

    def test_a_file_that_is_not_pcm_is_refused(self) -> None:
        for kind in (3, 6, 7, 0xFFFE):
            with self.subTest(kind=kind):
                self.bad(f"not PCM, its format is number {kind}", riff(fmt(kind=kind), data(1, 2)))

    def test_a_file_that_is_not_16_bit_is_refused(self) -> None:
        for bits in (8, 24, 32):
            with self.subTest(bits=bits):
                self.bad(f"not 16-bit, it has {bits} bits", riff(fmt(bits=bits), chunk(b"data", b"\x00" * 24)))

    def test_a_file_that_is_not_mono_or_stereo_is_refused(self) -> None:
        self.bad("3 channels", riff(fmt(channels=3), data(1, 2, 3)))
        self.bad("0 channels", riff(fmt(channels=0), data(1, 2, 3)))

    def test_a_file_whose_header_gives_a_rate_of_0_is_refused(self) -> None:
        self.bad("a rate of 0", riff(fmt(rate=0), data(1, 2)))

    def test_a_file_cut_short_is_refused(self) -> None:
        good = riff(fmt(), data(1, 2, 3, 4))
        self.assertEqual(len(good), 52)
        cuts = {
            "in the samples": (good[:49], "the data chunk says 8 bytes and 5 follow"),
            "at the first sample": (good[:44], "the data chunk says 8 bytes and 0 follow"),
            "in the data chunk's size": (good[:42], "ends before any data chunk"),
            "after the fmt chunk": (good[:36], "ends before any data chunk"),
            "in the fmt chunk": (good[:30], "the fmt chunk says 16 bytes and 10 follow"),
            "after the first twelve bytes": (good[:12], "ends before any data chunk"),
        }
        for where, (raw, reason) in cuts.items():
            with self.subTest(where=where):
                self.bad("cut short", raw)
                self.bad(reason, raw)

    def test_a_file_with_no_data_chunk_is_refused(self) -> None:
        self.bad("ends before any data chunk", riff(fmt(), chunk(b"LIST", b"note")))

    def test_a_fmt_chunk_too_small_is_refused(self) -> None:
        small = chunk(b"fmt ", struct.pack("<HHIIH", 1, 1, 8000, 16000, 2))
        self.bad("the fmt chunk is 14 bytes of 16", riff(small, data(1, 2)))

    def test_data_before_the_fmt_chunk_is_refused(self) -> None:
        self.bad("before any fmt chunk", riff(data(1, 2), fmt()))

    def test_data_that_does_not_fill_whole_frames_is_refused(self) -> None:
        self.bad("whole frames", riff(fmt(channels=2), data(1, 2, 3)))
        self.bad("whole frames", riff(fmt(), chunk(b"data", b"\x01\x02\x03")))


if __name__ == "__main__":
    unittest.main()
