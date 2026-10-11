"""Tests for rowsynth.gui.audio."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from rowsynth.errors import RowSynthError
from rowsynth.wav import save_wav

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
try:
    import pygame
    from rowsynth.gui import audio
except ImportError:
    pygame = None

RATE = 44100
START = 100.0


class Clock:
    """A clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> float:
        return self.now


def ramp(seconds: float = 1.0, top: float = 1.0) -> np.ndarray:
    """A line rising to `top` over that many seconds, so that no two stretches of it are alike."""
    return np.linspace(-top, top, int(seconds * RATE), dtype=np.float32)


def turned(sides: list[np.ndarray], at: int) -> bytes:
    """The frames of `sides` from frame `at` to the end, and then from the start up to it."""
    frames = audio.to_pcm(sides)
    return np.concatenate([frames[at:], frames[:at]]).tobytes()


def sounding() -> list[bytes]:
    """The frames of each sound the mixer is playing."""
    channels = [pygame.mixer.Channel(number) for number in range(pygame.mixer.get_num_channels())]
    return [channel.get_sound().get_raw() for channel in channels if channel.get_busy()]


@unittest.skipUnless(pygame, "pygame is not installed")
class ToPcmTest(unittest.TestCase):
    def test_frames_are_the_16_bit_values_save_wav_writes(self) -> None:
        halves = [(k + 0.5) / 32767 for k in (0, 1, 2, -1, -2, -3)]
        left = np.concatenate([np.linspace(-1.2, 1.2, 97), halves, [np.nan, np.inf, -np.inf]])
        right = left[::-1].astype(np.float32)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "same.wav"
            save_wav(path, [left, right], RATE)
            written = path.read_bytes()[44:]
        self.assertEqual(audio.to_pcm([left, right]).astype("<i2").tobytes(), written)

    def test_one_side_goes_to_both(self) -> None:
        side = ramp(0.01)
        frames = audio.to_pcm([side])
        self.assertEqual(frames.shape, (441, 2))
        self.assertEqual(frames.tobytes(), audio.to_pcm([side, side]).tobytes())

    def test_samples_past_full_scale_are_clamped(self) -> None:
        frames = audio.to_pcm([np.array([1.5, -1.5, 1.0, -1.0], dtype=np.float32)])
        self.assertEqual(frames.dtype, np.int16)
        self.assertEqual(frames[:, 0].tolist(), [32767, -32767, 32767, -32767])

    def test_sides_that_cannot_be_played_are_refused(self) -> None:
        row = np.zeros(4, dtype=np.float32)
        refused = {
            "the two sides differ in length, 4 and 5 samples": [row, np.zeros(5, dtype=np.float32)],
            "one side or two, left and right, got 3": [row, row, row],
            "one side or two, left and right, got 0": [],
            "no samples to play": [np.zeros(0, dtype=np.float32)],
            "one row of numbers, got shape (2, 2)": [np.zeros((2, 2))],
            "one row of numbers, got shape ()": row,
            "must be rows of numbers": [["a", "b"]],
        }
        for reason, sides in refused.items():
            with self.subTest(reason=reason):
                with self.assertRaises(RowSynthError) as caught:
                    audio.to_pcm(sides)
                self.assertIn(reason, str(caught.exception))


@unittest.skipUnless(pygame, "pygame is not installed")
class PlayerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = Clock()
        self.player = audio.Player(RATE, self.clock)
        self.addCleanup(self.player.close)

    def at(self, seconds: float) -> None:
        """Moves the clock to that long after the test began."""
        self.clock.now = START + seconds

    def test_it_opens_the_mixer_at_its_rate_in_16_bit_with_two_channels(self) -> None:
        self.assertEqual(pygame.mixer.get_init(), (RATE, -16, 2))
        other = audio.Player(22050)
        self.addCleanup(other.close)
        self.assertEqual(pygame.mixer.get_init(), (22050, -16, 2))

    def test_a_rate_that_cannot_be_opened_is_refused(self) -> None:
        for rate in (0, -44100, 44100.5, "44100", True):
            with self.subTest(rate=rate):
                with self.assertRaises(RowSynthError) as caught:
                    audio.Player(rate)
                self.assertIn("the rate must be an integer of 1 or more", str(caught.exception))
        with mock.patch.object(pygame.mixer, "init", side_effect=pygame.error("no such device")):
            with self.assertRaises(RowSynthError) as caught:
                audio.Player(48000)
        self.assertIn("cannot be opened at 48000 Hz: no such device", str(caught.exception))
        with self.assertRaises(RowSynthError) as caught:
            audio.Player(2 ** 40)
        self.assertIn("cannot be opened at 1099511627776 Hz", str(caught.exception))

    def test_nothing_plays_until_a_sound_is_played(self) -> None:
        self.assertFalse(self.player.playing)
        self.assertEqual(self.player.position(), 0.0)
        self.assertEqual(self.player.level(), (0.0, 0.0))
        self.assertEqual(sounding(), [])

    def test_play_hands_the_mixer_the_frames_in_place_of_what_was_playing(self) -> None:
        sides = [ramp(0.5), ramp(0.5, 0.25)]
        self.player.play([ramp()], loop=True)
        self.at(0.25)
        self.player.play(sides)
        self.assertEqual(sounding(), [audio.to_pcm(sides).tobytes()])
        self.assertEqual(self.player.position(), 0.0)

    def test_a_sound_played_once_plays_for_its_length(self) -> None:
        self.player.play([ramp(0.5)])
        for seconds, playing, place in ((0.0, True, 0.0), (0.25, True, 0.25), (0.5, False, 0.5), (3.0, False, 0.5)):
            with self.subTest(seconds=seconds):
                self.at(seconds)
                self.assertEqual(self.player.playing, playing)
                self.assertEqual(self.player.position(), place)

    def test_a_loop_wraps_at_its_length_and_plays_until_stopped(self) -> None:
        self.player.play([ramp(0.5)], loop=True)
        for seconds, place in ((0.0, 0.0), (0.25, 0.25), (0.5, 0.0), (0.625, 0.125), (7.75, 0.25)):
            with self.subTest(seconds=seconds):
                self.at(seconds)
                self.assertTrue(self.player.playing)
                self.assertEqual(self.player.position(), place)
        self.assertEqual(sounding(), [audio.to_pcm([ramp(0.5)]).tobytes()])

    def test_stop_ends_the_sound_and_puts_the_position_back_to_0(self) -> None:
        self.player.play([np.ones(RATE, dtype=np.float32)], loop=True)
        self.at(0.25)
        self.assertEqual(self.player.level(), (1.0, 1.0))
        self.player.stop()
        self.assertFalse(self.player.playing)
        self.assertEqual(self.player.position(), 0.0)
        self.assertEqual(self.player.level(), (0.0, 0.0))
        self.assertEqual(sounding(), [])

    def test_swap_goes_on_from_the_same_place(self) -> None:
        new = [ramp(1.0, 0.5), ramp(1.0, 0.25)]
        self.player.play([ramp()], loop=True)
        self.at(1.25)
        self.player.swap(new)
        self.assertEqual(self.player.position(), 0.25)
        self.assertEqual(sounding(), [turned(new, RATE // 4)])
        self.at(1.5)
        self.assertEqual(self.player.position(), 0.5)
        self.at(2.125)
        self.assertEqual(self.player.position(), 0.125)
        self.assertTrue(self.player.playing)

    def test_swap_carries_the_place_as_a_share_of_another_length(self) -> None:
        new = [ramp(2.0)]
        self.player.play([ramp()], loop=True)
        self.at(1.25)
        self.player.swap(new)
        self.assertEqual(self.player.position(), 0.5)
        self.assertEqual(sounding(), [turned(new, RATE // 2)])
        self.at(2.25)
        self.assertEqual(self.player.position(), 1.5)
        self.at(3.0)
        self.assertEqual(self.player.position(), 0.25)

    def test_swap_does_nothing_when_no_loop_is_playing(self) -> None:
        once = [ramp(0.5)]
        self.player.swap([ramp()])
        self.assertFalse(self.player.playing)
        self.assertEqual(sounding(), [])
        self.player.play(once)
        self.at(0.25)
        self.player.swap([ramp()])
        self.assertEqual(self.player.position(), 0.25)
        self.assertEqual(sounding(), [audio.to_pcm(once).tobytes()])
        self.player.stop()
        self.player.swap([ramp()])
        self.assertEqual(sounding(), [])

    def test_loop_switched_on_has_a_sound_played_once_go_round_from_the_place_it_is_at(self) -> None:
        sides = [ramp(), ramp(1.0, 0.5)]
        self.player.play(sides)
        self.at(0.25)
        self.player.loop(True)
        self.assertEqual(self.player.position(), 0.25)
        self.assertEqual(sounding(), [turned(sides, RATE // 4)])
        for seconds, place in ((0.5, 0.5), (1.0, 0.0), (3.75, 0.75)):
            with self.subTest(seconds=seconds):
                self.at(seconds)
                self.assertTrue(self.player.playing)
                self.assertEqual(self.player.position(), place)

    def test_loop_switched_off_has_a_loop_play_on_from_the_place_it_is_at_and_end_at_its_end(self) -> None:
        # Silence but for one sample three quarters of the way in, to read the level by.
        side = np.zeros(RATE, dtype=np.float32)
        side[RATE * 3 // 4] = 1.0
        self.player.play([side], loop=True)
        self.at(2.25)
        self.player.loop(False)
        self.assertEqual(self.player.position(), 0.25)
        self.assertEqual(sounding(), [audio.to_pcm([side])[RATE // 4:].tobytes()])
        for seconds, playing, place, level in ((2.5, True, 0.5, 0.0), (2.76, True, 0.76, 1.0),
                                               (3.0, False, 1.0, 0.0), (4.5, False, 1.0, 0.0)):
            with self.subTest(seconds=seconds):
                self.at(seconds)
                self.assertEqual(self.player.playing, playing)
                self.assertAlmostEqual(self.player.position(), place)
                self.assertEqual(self.player.level(), (level, level))

    def test_loop_does_nothing_with_nothing_playing_or_with_nothing_to_change(self) -> None:
        once = [ramp(0.5)]
        self.player.loop(True)
        self.assertEqual((self.player.playing, sounding()), (False, []))
        self.player.play(once)
        self.at(0.25)
        self.player.loop(False)
        self.assertEqual((self.player.position(), sounding()), (0.25, [audio.to_pcm(once).tobytes()]))
        # By the clock the sound has ended, so there is nothing to have go round.
        self.at(0.5)
        self.player.loop(True)
        self.assertEqual((self.player.playing, sounding()), (False, [audio.to_pcm(once).tobytes()]))
        self.player.play(once, loop=True)
        self.at(0.75)
        self.player.loop(True)
        self.assertEqual((self.player.position(), sounding()), (0.25, [audio.to_pcm(once).tobytes()]))

    def test_level_reads_the_stretch_behind_the_position_on_each_side(self) -> None:
        left = np.zeros(RATE, dtype=np.float32)
        right = np.zeros(RATE, dtype=np.float32)
        left[RATE // 4] = 0.5
        left[RATE // 4 - 100] = 0.25
        right[RATE // 2] = -1.0
        self.player.play([left, right])
        half = 16384 / 32767
        levels = ((0.0, (0.0, 0.0)), (0.24, (0.0, 0.0)), (0.25, (half, 0.0)), (0.27, (half, 0.0)),
                  (0.29, (0.0, 0.0)), (0.5, (0.0, 1.0)), (0.52, (0.0, 1.0)), (0.54, (0.0, 0.0)))
        for seconds, level in levels:
            with self.subTest(seconds=seconds):
                self.at(seconds)
                self.assertEqual(self.player.level(), level)

    def test_level_of_a_loop_reads_back_round_the_end_once_it_has_gone_round(self) -> None:
        side = np.zeros(RATE, dtype=np.float32)
        side[-1] = 1.0
        self.player.play([side], loop=True)
        for seconds, level in ((0.01, (0.0, 0.0)), (1.01, (1.0, 1.0)), (1.04, (0.0, 0.0))):
            with self.subTest(seconds=seconds):
                self.at(seconds)
                self.assertEqual(self.player.level(), level)

    def test_level_is_0_when_a_sound_played_once_has_ended(self) -> None:
        self.player.play([np.ones(RATE // 2, dtype=np.float32)])
        self.at(0.25)
        self.assertEqual(self.player.level(), (1.0, 1.0))
        self.at(0.5)
        self.assertEqual(self.player.level(), (0.0, 0.0))

    def test_close_shuts_the_mixer_down(self) -> None:
        self.player.play([ramp()], loop=True)
        self.player.close()
        self.assertIsNone(pygame.mixer.get_init())
        self.assertFalse(self.player.playing)
        self.assertEqual(self.player.position(), 0.0)
        self.player.close()

    def test_also_plays_over_the_sound_that_plays_and_leaves_it_where_it_is(self) -> None:
        main, note = [ramp(4.0)], [ramp(4.0, 0.5), ramp(4.0, 0.25)]
        self.player.play(main, loop=True)
        self.at(0.25)
        self.player.also(note)
        self.assertEqual(sounding(), [audio.to_pcm(main).tobytes(), audio.to_pcm(note).tobytes()])
        self.assertEqual((self.player.playing, self.player.position()), (True, 0.25))

    def test_the_position_and_the_level_know_nothing_of_what_also_plays(self) -> None:
        self.player.also([np.ones(RATE, dtype=np.float32)])
        self.at(0.25)
        self.assertEqual((self.player.playing, self.player.position(), self.player.level()), (False, 0.0, (0.0, 0.0)))
        self.player.play([np.full(RATE, 0.5, dtype=np.float32)])
        self.player.also([np.ones(RATE, dtype=np.float32)])
        self.at(0.5)
        half = 16384 / 32767
        self.assertEqual((self.player.position(), self.player.level()), (0.25, (half, half)))

    def test_a_new_voice_is_one_that_has_ended_and_with_none_the_one_that_began_longest_ago(self) -> None:
        self.assertEqual(audio.VOICES, 4)
        # The second is short, and the six others outlast the test.
        notes = [[ramp(1.0 if number == 1 else 4.0, (1 + number) / 8)] for number in range(7)]
        first, _, third, fourth, fifth, sixth, seventh = (audio.to_pcm(note).tobytes() for note in notes)
        for note in notes[:4]:
            self.player.also(note)
        # By the clock the second has ended, so its voice is free, and the first goes on though it is the oldest.
        self.at(2.0)
        self.player.also(notes[4])
        self.assertEqual(sounding(), [first, fifth, third, fourth])
        # With all four ringing, each new one takes the place of the one that began longest ago.
        self.player.also(notes[5])
        self.assertEqual(sounding(), [sixth, fifth, third, fourth])
        self.player.also(notes[6])
        self.assertEqual(sounding(), [sixth, fifth, seventh, fourth])

    def test_stop_leaves_what_also_played_to_ring_and_close_ends_it(self) -> None:
        note = [ramp(4.0)]
        self.player.play([ramp(4.0, 0.5)], loop=True)
        self.player.also(note)
        self.player.stop()
        self.assertEqual(sounding(), [audio.to_pcm(note).tobytes()])
        self.player.close()
        self.assertIsNone(pygame.mixer.get_init())

    def test_stop_with_a_fade_has_the_mixer_bring_the_sound_down_and_is_through_with_it_at_once(self) -> None:
        sides = [ramp(4.0)]
        self.player.play(sides, loop=True)
        self.at(0.25)
        self.player.stop(fade=30.0)
        self.assertEqual((self.player.playing, self.player.position(), self.player.level()), (False, 0.0, (0.0, 0.0)))
        # It was not cut off: the mixer has it still, on its way down.
        self.assertEqual(sounding(), [audio.to_pcm(sides).tobytes()])
        # The mixer does not tell how long a sound takes to die away, so the channel is stood in for.
        with mock.patch.object(self.player, "_channel") as channel:
            self.player.stop(fade=0.25)
            self.player.stop(fade=0.0001)
            self.player.stop()
        self.assertEqual(channel.method_calls, [mock.call.fadeout(250), mock.call.fadeout(1), mock.call.stop()])

    def test_sides_that_are_refused_leave_the_loop_playing(self) -> None:
        self.player.play([ramp()], loop=True)
        self.at(0.25)
        for call in (self.player.play, self.player.swap, self.player.also):
            with self.assertRaises(RowSynthError):
                call([])
        self.assertTrue(self.player.playing)
        self.assertEqual(self.player.position(), 0.25)
        self.assertEqual(sounding(), [audio.to_pcm([ramp()]).tobytes()])


if __name__ == "__main__":
    unittest.main()
