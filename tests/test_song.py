"""Tests for a song laid end to end and played."""
from __future__ import annotations

import copy
import math
import unittest
from unittest import mock

import numpy as np

from rowsynth.errors import RowSynthError
from rowsynth.note import render_note
from rowsynth.song import lay, render_song, render_stems
from rowsynth.sound import finish

# With three ticks to a row, a row is 400 samples at 8000 Hz, and the two sides run on for 4000 after the last.
RATE = 8000
# Noise at full level from its first tick to its last. Its first sample is never 0.
HISS = {"wave": "noise", "release": 0}
# A triangle that stays well under the knee.
HUM = {"wave": "tri", "vol": 12}
# That triangle, sounding on for nine ticks after it is let go: 1200 samples.
RING = {**HUM, "release": 9}

# Two tuned channels and a drum channel. The second pattern has no text for the bass, and its tune is half as
# long as its beat.
SONG = {
    "row_ticks": 3,
    "instruments": {
        "lead": {"vol": [15, 12, 10], "duty": 1, "gate": 0.5, "release": 1},
        "bass": {"wave": "tri", "vol": 12},
        "thud": {"wave": "noise", "vol": [15, 9, 4], "release": 0},
        "tick": {"wave": "noise", "short": True, "vol": [10, 3], "release": 0},
    },
    "drums": {"k": {"inst": "thud", "note": "c2"},
              "s": {"inst": "thud", "note": "d4", "rows": 1, "vel": 0.7},
              "h": {"inst": "tick", "note": 90, "vel": 0.5}},
    "channels": {"tune": {"inst": "lead", "pan": -0.5, "echo": {"rows": 1.5, "gain": 0.4}},
                 "low": {"inst": "bass", "gain": 0.8, "pan": 0.6},
                 "beat": {"inst": "thud", "drums": True}},
    "patterns": {"a": {"tune": "l8 c e g > c", "low": "o2 c2", "beat": "k4 h8 s8"},
                 "b": {"tune": "o5 c4", "beat": "l8 k h k h"}},
    "order": ["a", "b", "a"],
}

# Two channels, both all to the left. The second comes in two rows late and makes the pattern ten rows long.
DUO = {
    "row_ticks": 3,
    "instruments": {"hum": HUM, "hiss": HISS},
    "channels": {"high": {"inst": "hum", "pan": -1}, "deep": {"inst": "hiss", "pan": -1}},
    "patterns": {"p": {"high": "c4", "deep": "r8 c2"}},
    "order": ["p"],
}


def event(row: float, rows: float, note: float, inst: str = "lead", vel: float = 1) -> dict:
    """One event as lay gives it."""
    return {"row": row, "rows": rows, "note": note, "inst": inst, "vel": vel}


def tunes(patterns: dict, order: list, **keys: object) -> dict:
    """A song of tuned channels: one for every name its patterns have text for, each with an instrument of its name."""
    names = list(dict.fromkeys(name for pattern in patterns.values() for name in pattern))
    return {"row_ticks": 3, "instruments": {name: HUM for name in names},
            "channels": {name: {"inst": name} for name in names}, "patterns": patterns, "order": order, **keys}


def solo(text: str, inst: dict = HUM, row_ticks: int = 3, **settings: object) -> dict:
    """A song of one channel and one pattern. The channel is all to the left unless its settings say otherwise."""
    return {"row_ticks": row_ticks, "instruments": {"one": inst},
            "channels": {"voice": {"inst": "one", "pan": -1, **settings}},
            "patterns": {"p": {"voice": text}}, "order": ["p"]}


def left(song: dict, rate: int = RATE, **keys: object) -> np.ndarray:
    """The left side of a song."""
    return render_song(song, rate=rate, **keys)["left"]


def raw(notes: list, size: int, rate: int = RATE) -> np.ndarray:
    """Notes played by hand into one buffer of `size` samples: each is (sample, instrument, note, ticks, velocity)."""
    out = np.zeros(size, dtype=np.float32)
    for at, inst, note, ticks, velocity in notes:
        render_note(out, at, inst, note, ticks, velocity, rate)
    return out


def hand(notes: list, size: int, gain: float = 0.35, rate: int = RATE) -> np.ndarray:
    """Those notes as a side of a song has them, when their channel is all on that side at a gain of 1."""
    return finish(raw(notes, size, rate), gain, rate)


def stems(song: dict, rate: int = RATE) -> dict:
    """The stems of a song, each under the name of its channel."""
    return {found["name"]: found["samples"] for found in render_stems(song, rate)["stems"]}


def stem(song: dict, rate: int = RATE) -> np.ndarray:
    """The stem of a song of one channel."""
    return stems(song, rate)["voice"]


def looped(notes: list, size: int, loop: int, gain: float = 0.35, rate: int = RATE) -> np.ndarray:
    """Those notes as a stem of `loop` samples has them, when their channel's gain times the song's is `gain`.

    They are played by hand into `size` samples. Those go into the loop one at a time, each at the gain, and
    the ones past the end of the loop go round to its start. The loop is finished without the fade.
    """
    out = np.zeros(loop, dtype=np.float32)
    for at, sample in enumerate(raw(notes, size, rate)):
        out[at % loop] = float(out[at % loop]) + float(sample) * gain
    return finish(out, 1, rate, False)


def near(test: unittest.TestCase, got: np.ndarray, wanted: np.ndarray) -> None:
    """Holds that a side is what is wanted of it but for the rounding to float32, and that it is not silence."""
    np.testing.assert_allclose(got.astype(np.float64), wanted, rtol=0, atol=1e-6)
    test.assertGreater(float(np.abs(wanted).max()), 0.01)


class LayTest(unittest.TestCase):
    def test_patterns_follow_one_another_with_their_rows_moved_along(self):
        laid, rows = lay(SONG)
        self.assertEqual(laid["tune"], [event(0, 2, 60), event(2, 2, 64), event(4, 2, 67), event(6, 2, 72),
                                        event(8, 4, 72),
                                        event(16, 2, 60), event(18, 2, 64), event(20, 2, 67), event(22, 2, 72)])
        self.assertEqual(laid["low"], [event(0, 8, 36, "bass"), event(16, 8, 36, "bass")])
        self.assertEqual(laid["beat"], [event(0, 4, 36, "thud"), event(4, 2, 90, "tick", 0.5),
                                        event(6, 1, 62, "thud", 0.7),
                                        event(8, 2, 36, "thud"), event(10, 2, 90, "tick", 0.5),
                                        event(12, 2, 36, "thud"), event(14, 2, 90, "tick", 0.5),
                                        event(16, 4, 36, "thud"), event(20, 2, 90, "tick", 0.5),
                                        event(22, 1, 62, "thud", 0.7)])
        self.assertEqual(rows, 24)

    def test_one_pattern_gives_its_channels_as_they_are_written(self):
        laid, rows = lay({**SONG, "order": ["b"]})
        self.assertEqual(laid, {"tune": [event(0, 4, 72)],
                                "beat": [event(0, 2, 36, "thud"), event(2, 2, 90, "tick", 0.5),
                                         event(4, 2, 36, "thud"), event(6, 2, 90, "tick", 0.5)]})
        self.assertEqual(rows, 8)

    def test_pattern_lasts_as_long_as_the_longest_of_its_channels(self):
        song = tunes({"a": {"lead": "c4", "bass": "c1"}, "b": {"lead": "d"}, "c": {"bass": "e2", "lead": "e"}},
                     ["a", "b", "b", "c", "b"])
        laid, rows = lay(song)
        self.assertEqual(laid["lead"], [event(0, 4, 60), event(16, 1, 62), event(17, 1, 62), event(18, 1, 64),
                                        event(26, 1, 62)])
        self.assertEqual(laid["bass"], [event(0, 16, 60, "bass"), event(18, 8, 64, "bass")])
        self.assertEqual(rows, 27)

    def test_rests_at_the_end_of_a_channel_count_towards_its_pattern(self):
        laid, rows = lay(tunes({"a": {"lead": "c r2"}, "b": {"lead": "d"}}, ["a", "b"]))
        self.assertEqual(laid, {"lead": [event(0, 1, 60), event(9, 1, 62)]})
        self.assertEqual(rows, 10)

    def test_part_of_a_row_is_carried_into_the_patterns_after_it(self):
        laid, rows = lay(tunes({"a": {"lead": "c32"}, "b": {"lead": "d8."}}, ["a", "b", "a"]))
        self.assertEqual(laid, {"lead": [event(0, 0.5, 60), event(0.5, 3, 62), event(3.5, 0.5, 60)]})
        self.assertEqual(rows, 4)

    def test_channels_come_in_the_order_the_patterns_bring_them_in(self):
        song = tunes({"a": {"bass": "c"}, "b": {"lead": "d", "bass": "e", "pad": "f"}}, ["a", "b"])
        self.assertEqual(list(lay(song)[0]), ["bass", "lead", "pad"])
        song["order"] = ["b", "a"]
        self.assertEqual(list(lay(song)[0]), ["lead", "bass", "pad"])

    def test_channel_that_no_pattern_played_has_text_for_is_left_out(self):
        song = tunes({"a": {"lead": "c"}, "b": {"lead": "d", "bass": "e"}}, ["a", "a"])
        song["channels"]["spare"] = {"inst": "lead"}
        self.assertEqual(lay(song), ({"lead": [event(0, 1, 60), event(1, 1, 60)]}, 2))

    def test_channel_with_empty_text_has_no_events(self):
        self.assertEqual(lay(tunes({"a": {"lead": "", "bass": " | "}}, ["a"])), ({"lead": [], "bass": []}, 0))

    def test_pattern_with_no_channels_adds_nothing(self):
        self.assertEqual(lay(tunes({"a": {"lead": "c"}, "gap": {}}, ["gap", "a", "gap", "a"])),
                         ({"lead": [event(0, 1, 60), event(1, 1, 60)]}, 2))

    def test_order_with_no_patterns_gives_no_events_and_no_rows(self):
        self.assertEqual(lay({**SONG, "order": []}), ({}, 0))

    def test_pattern_played_twice_gives_events_of_its_own_each_time(self):
        laid, _ = lay(tunes({"a": {"lead": "c"}}, ["a", "a"]))
        first, second = laid["lead"]
        self.assertIsNot(first, second)
        first["note"] = 0
        self.assertEqual(second, event(1, 1, 60))

    def test_rows_per_beat_of_the_song_sets_every_length(self):
        patterns = {"a": {"lead": "c4 d8"}, "b": {"lead": "e4."}}
        self.assertEqual(lay(tunes(patterns, ["a", "b"], rows_per_beat=2)),
                         ({"lead": [event(0, 2, 60), event(2, 1, 62), event(3, 3, 64)]}, 6))
        self.assertEqual(lay(tunes(patterns, ["a", "b"], rows_per_beat=8)),
                         ({"lead": [event(0, 8, 60), event(8, 4, 62), event(12, 12, 64)]}, 24))

    def test_channel_starts_every_pattern_on_its_own_instrument_and_as_text_starts(self):
        song = tunes({"a": {"lead": "l8 o5 v6 @bass c"}, "b": {"lead": "c"}, "c": {"bass": "c"}}, ["a", "b", "c"])
        laid, rows = lay(song)
        self.assertEqual(laid, {"lead": [event(0, 2, 72, "bass", 0.4), event(2, 1, 60)],
                                "bass": [event(3, 1, 60, "bass")]})
        self.assertEqual(rows, 4)

    def test_drum_channel_plays_the_drums_of_the_song_and_notes_on_its_own_instrument(self):
        song = copy.deepcopy(SONG)
        song["patterns"] = {"a": {"beat": "k c s"}}
        song["order"] = ["a", "a"]
        self.assertEqual(lay(song), ({"beat": [event(0, 1, 36, "thud"), event(1, 1, 60, "thud"),
                                               event(2, 1, 62, "thud", 0.7), event(3, 1, 36, "thud"),
                                               event(4, 1, 60, "thud"), event(5, 1, 62, "thud", 0.7)]}, 6))

    def test_pattern_left_out_of_the_order_is_not_read(self):
        song = tunes({"a": {"lead": "c"}, "b": {"lead": "c ? d", "nobody": "c"}}, ["a"])
        del song["channels"]["nobody"]
        self.assertEqual(lay(song), ({"lead": [event(0, 1, 60)]}, 1))

    def test_instrument_that_is_named_and_plays_no_note_is_not_missed(self):
        self.assertEqual(lay(tunes({"a": {"lead": "c @organ r | @lead d @organ"}}, ["a"])),
                         ({"lead": [event(0, 1, 60), event(2, 1, 62)]}, 3))

    def test_song_is_left_as_it_was(self):
        song = copy.deepcopy(SONG)
        lay(song)
        self.assertEqual(song, SONG)


class NotesTest(unittest.TestCase):
    def test_note_lands_at_its_row(self):
        for text, start in (("c", 0), ("r c", 400), ("r4 c", 1600), ("r32 c", 200), ("r12 c", 533)):
            out = left(solo(text, HISS))
            self.assertFalse(out[:start].any(), text)
            self.assertNotEqual(float(out[start]), 0.0, text)

    def test_note_lands_where_its_pattern_has_got_to(self):
        song = solo("c", HISS)
        song["patterns"]["gap"] = {"voice": "r2"}
        song["order"] = ["gap", "gap", "p"]
        out = left(song)
        self.assertFalse(out[:6400].any())
        self.assertNotEqual(float(out[6400]), 0.0)

    def test_start_rounds_a_half_up(self):
        # A row of one tick is 183.75 samples at 11025 Hz, and two of them 367.5.
        out = left(solo("r8 c", HISS, 1), 11025)
        self.assertFalse(out[:368].any())
        self.assertNotEqual(float(out[368]), 0.0)

    def test_side_is_its_notes_added_into_one_buffer_and_finished(self):
        got = left(solo("l8 c e > c4"))
        wanted = hand([(0, HUM, 60, 5, 1.0), (800, HUM, 64, 5, 1.0), (1600, HUM, 72, 11, 1.0)], 7200)
        self.assertTrue(np.array_equal(got, wanted))

    def test_note_is_held_for_nine_tenths_of_its_rows(self):
        # Three ticks to a row: 12 ticks for a quarter note and 48 for a whole one, and 10.8 and 43.2 of them held.
        for text, ticks in (("c4", 11), ("c1", 43), ("c", 3)):
            got = left(solo(text))
            self.assertTrue(np.array_equal(got, hand([(0, HUM, 60, ticks, 1.0)], len(got))), text)

    def test_gate_of_its_instrument_is_the_share_of_its_rows_a_note_is_held_for(self):
        for gate, ticks in ((1, 12), (0.5, 6), (0.25, 3), (0.2, 2), (1.5, 18)):
            inst = {**HUM, "gate": gate}
            got = left(solo("c4", inst))
            self.assertTrue(np.array_equal(got, hand([(0, inst, 60, ticks, 1.0)], len(got))), gate)

    def test_ticks_a_note_is_held_for_round_a_half_up(self):
        inst = {**HUM, "gate": 0.5}
        for text, ticks in (("c", 2), ("c8.", 5)):
            got = left(solo(text, inst))
            self.assertTrue(np.array_equal(got, hand([(0, inst, 60, ticks, 1.0)], len(got))), text)

    def test_note_is_held_for_one_tick_at_the_least(self):
        # Half a row of one tick, and nine tenths of that.
        got = left(solo("c32", HUM, 1))
        self.assertTrue(np.array_equal(got, hand([(0, HUM, 60, 1, 1.0)], len(got))))
        for gate in (0, 0.01, -1):
            inst = {**HUM, "gate": gate}
            got = left(solo("c4", inst))
            self.assertTrue(np.array_equal(got, hand([(0, inst, 60, 1, 1.0)], len(got))), gate)

    def test_ticks_in_a_row_scale_what_a_note_is_held_for_and_where_it_lands(self):
        got = left(solo("c d8", HISS, 6))
        self.assertTrue(np.array_equal(got, hand([(0, HISS, 60, 5, 1.0), (800, HISS, 62, 11, 1.0)], len(got))))

    def test_volume_in_the_text_is_the_velocity_of_its_notes(self):
        got = left(solo("v6 c4 v15 d4"))
        wanted = hand([(0, HUM, 60, 11, 0.4), (1600, HUM, 62, 11, 1.0)], len(got))
        self.assertTrue(np.array_equal(got, wanted))

    def test_instrument_named_in_the_text_plays_the_notes_after_it(self):
        song = solo("c4 @two c4 @one c4")
        song["instruments"]["two"] = HISS
        got = left(song)
        wanted = hand([(0, HUM, 60, 11, 1.0), (1600, HISS, 60, 11, 1.0), (3200, HUM, 60, 11, 1.0)], len(got))
        self.assertTrue(np.array_equal(got, wanted))

    def test_drum_plays_its_instrument_at_its_note_for_no_longer_than_its_rows(self):
        song = solo("k4 s4 k16", drums=True)
        song["instruments"]["two"] = HISS
        song["drums"] = {"k": {"inst": "two", "note": "c2"}, "s": {"inst": "one", "note": 62.5, "rows": 1, "vel": 0.5}}
        got = left(song)
        wanted = hand([(0, HISS, 36.0, 11, 1.0), (1600, HUM, 62.5, 3, 0.5), (3200, HISS, 36.0, 3, 1.0)], len(got))
        self.assertTrue(np.array_equal(got, wanted))

    def test_tied_notes_are_held_as_one(self):
        got = left(solo("c4 & c4"))
        self.assertTrue(np.array_equal(got, hand([(0, HUM, 60, 22, 1.0)], len(got))))


class EchoTest(unittest.TestCase):
    def test_echo_plays_each_note_again_later_and_quieter(self):
        got = left(solo("c4 e4", echo={"rows": 2, "gain": 0.5}))
        wanted = hand([(0, HUM, 60, 11, 1.0), (800, HUM, 60, 11, 0.5), (1600, HUM, 64, 11, 1.0),
                       (2400, HUM, 64, 11, 0.5)], len(got))
        self.assertTrue(np.array_equal(got, wanted))

    def test_echo_can_come_part_of_a_row_later(self):
        got = left(solo("c4", echo={"rows": 0.75, "gain": 0.25}))
        self.assertTrue(np.array_equal(got, hand([(0, HUM, 60, 11, 1.0), (300, HUM, 60, 11, 0.25)], len(got))))

    def test_start_of_an_echo_rounds_a_half_up(self):
        # A row of one tick is 183.75 samples at 11025 Hz, and two of them 367.5.
        got = left(solo("c32", HUM, 1, echo={"rows": 2, "gain": 0.5}), 11025)
        wanted = hand([(0, HUM, 60, 1, 1.0), (368, HUM, 60, 1, 0.5)], len(got), rate=11025)
        self.assertTrue(np.array_equal(got, wanted))

    def test_gain_of_an_echo_scales_the_velocity_its_note_has(self):
        got = left(solo("v6 c4", echo={"rows": 1, "gain": 0.5}))
        self.assertTrue(np.array_equal(got, hand([(0, HUM, 60, 11, 0.4), (400, HUM, 60, 11, 0.4 * 0.5)], len(got))))

    def test_echo_of_a_drum_is_as_long_as_the_drum(self):
        song = solo("k4", HISS, drums=True, echo={"rows": 3, "gain": 0.5})
        song["drums"] = {"k": {"inst": "one", "note": 70, "rows": 2}}
        got = left(song)
        self.assertTrue(np.array_equal(got, hand([(0, HISS, 70.0, 5, 1.0), (1200, HISS, 70.0, 5, 0.5)], len(got))))

    def test_echo_runs_on_into_the_patterns_after_its_note(self):
        song = solo("r2 r4 r8 c8", echo={"rows": 4, "gain": 0.5})
        song["order"] = ["p", "p"]
        got = left(song)
        wanted = hand([(5600, HUM, 60, 5, 1.0), (7200, HUM, 60, 5, 0.5), (12000, HUM, 60, 5, 1.0),
                       (13600, HUM, 60, 5, 0.5)], len(got))
        self.assertTrue(np.array_equal(got, wanted))

    def test_echo_makes_a_song_no_longer_and_what_falls_past_its_end_is_dropped(self):
        plain = render_song(solo("c4"), rate=RATE)
        late = render_song(solo("c4", echo={"rows": 100, "gain": 0.5}), rate=RATE)
        self.assertEqual((late["rows"], late["seconds"]), (plain["rows"], plain["seconds"]))
        self.assertTrue(np.array_equal(late["left"], plain["left"]))

    def test_dry_leaves_the_echo_out(self):
        song = solo("c4 e4", echo={"rows": 2, "gain": 0.5})
        got = left(song, dry=True)
        self.assertTrue(np.array_equal(got, left(solo("c4 e4"))))
        self.assertFalse(np.array_equal(got, left(song)))
        self.assertTrue(np.array_equal(left(song, dry=False), left(song)))

    def test_dry_changes_nothing_in_a_song_without_an_echo(self):
        self.assertTrue(np.array_equal(left(DUO, dry=True), left(DUO)))

    def test_only_the_channel_with_an_echo_has_one(self):
        song = copy.deepcopy(DUO)
        song["channels"]["deep"]["echo"] = {"rows": 1, "gain": 0.5}
        self.assertTrue(np.array_equal(left(song, only=["high"]), left(DUO, only=["high"])))
        self.assertFalse(np.array_equal(left(song, only=["deep"]), left(DUO, only=["deep"])))


class MixTest(unittest.TestCase):
    NOTE = [(0, HUM, 60, 11, 1.0)]

    def test_pan_of_minus_1_puts_a_channel_all_to_the_left(self):
        out = render_song(solo("c4", pan=-1), rate=RATE)
        self.assertTrue(np.array_equal(out["left"], hand(self.NOTE, 5600)))
        self.assertFalse(out["right"].any())

    def test_pan_of_1_puts_a_channel_all_to_the_right(self):
        out = render_song(solo("c4", pan=1), rate=RATE)
        self.assertTrue(np.array_equal(out["right"], hand(self.NOTE, 5600)))
        self.assertLess(float(np.abs(out["left"]).max()), 1e-12)

    def test_pan_left_out_puts_a_channel_in_the_middle_with_each_side_3_decibels_down(self):
        song = solo("c4")
        del song["channels"]["voice"]["pan"]
        out = render_song(song, rate=RATE)
        wanted = hand(self.NOTE, 5600).astype(np.float64) * math.sqrt(0.5)
        near(self, out["left"], wanted)
        near(self, out["right"], wanted)
        middle = render_song(solo("c4", pan=0), rate=RATE)
        self.assertTrue(np.array_equal(out["left"], middle["left"]))
        self.assertTrue(np.array_equal(out["right"], middle["right"]))

    def test_pan_shares_a_channel_between_the_sides_and_keeps_its_power(self):
        whole = hand(self.NOTE, 5600).astype(np.float64)
        for pan in (-0.5, 0.25, 0.8):
            out = render_song(solo("c4", pan=pan), rate=RATE)
            angle = (pan + 1) * math.pi / 4
            near(self, out["left"], whole * math.cos(angle))
            near(self, out["right"], whole * math.sin(angle))
            power = out["left"].astype(np.float64) ** 2 + out["right"].astype(np.float64) ** 2
            np.testing.assert_allclose(power, whole ** 2, rtol=0, atol=1e-6)

    def test_pan_to_the_right_of_the_middle_is_louder_on_the_right(self):
        for pan, louder, quieter in ((0.3, "right", "left"), (-0.3, "left", "right")):
            out = render_song(solo("c4", pan=pan), rate=RATE)
            self.assertGreater(float(np.abs(out[louder]).max()), 1.5 * float(np.abs(out[quieter]).max()), pan)

    def test_gain_of_a_channel_scales_it_on_both_sides(self):
        whole = hand(self.NOTE, 5600).astype(np.float64)
        for gain in (0.5, 0.3, 1.7):
            near(self, left(solo("c4", gain=gain)), whole * gain)
            out = render_song(solo("c4", gain=gain, pan=0.5), rate=RATE)
            near(self, out["left"], whole * gain * math.cos(3 * math.pi / 8))
            near(self, out["right"], whole * gain * math.sin(3 * math.pi / 8))

    def test_channels_add_up_on_a_side(self):
        high = raw([(0, HUM, 60, 11, 1.0)], 8000)
        deep = raw([(800, HISS, 60, 22, 1.0)], 8000)
        both = (high.astype(np.float64) + deep).astype(np.float32)
        out = render_song(DUO, rate=RATE)
        self.assertTrue(np.array_equal(out["left"], finish(both, 0.35, RATE)))
        self.assertFalse(out["right"].any())

    def test_share_of_a_channel_is_added_to_a_side_in_doubles_and_the_sum_rounded_to_float32(self):
        song = copy.deepcopy(DUO)
        song["channels"]["deep"].update(gain=0.3, pan=-0.25)
        high = raw([(0, HUM, 60, 11, 1.0)], 8000)
        deep = raw([(800, HISS, 60, 22, 1.0)], 8000).astype(np.float64)
        angle = 0.75 * math.pi / 4
        out = render_song(song, rate=RATE)
        both = (high + deep * (0.3 * math.cos(angle))).astype(np.float32)
        self.assertTrue(np.array_equal(out["left"], finish(both, 0.35, RATE)))
        alone = (deep * (0.3 * math.sin(angle))).astype(np.float32)
        self.assertTrue(np.array_equal(out["right"], finish(alone, 0.35, RATE)))

    def test_each_channel_goes_where_its_own_pan_puts_it(self):
        song = copy.deepcopy(DUO)
        song["channels"]["deep"]["pan"] = 1
        out = render_song(song, rate=RATE)
        near(self, out["left"], hand([(0, HUM, 60, 11, 1.0)], 8000).astype(np.float64))
        self.assertTrue(np.array_equal(out["right"], hand([(800, HISS, 60, 22, 1.0)], 8000)))

    def test_gain_of_a_song_left_out_is_0_35(self):
        self.assertTrue(np.array_equal(left(solo("c4")), hand(self.NOTE, 5600, 0.35)))
        self.assertTrue(np.array_equal(left({**solo("c4"), "gain": 0.35}), left(solo("c4"))))

    def test_gain_of_a_song_is_the_gain_of_the_output_stage(self):
        for gain in (0.2, 1, 3.5):
            self.assertTrue(np.array_equal(left({**solo("c4"), "gain": gain}), hand(self.NOTE, 5600, gain)), gain)

    def test_loud_song_stays_under_full_scale_and_ends_on_silence(self):
        out = render_song({**solo("c1 c1 c1", HISS, gain=4, pan=0), "gain": 2}, rate=RATE)
        for side in ("left", "right"):
            self.assertGreater(float(np.abs(out[side]).max()), 0.98)
            self.assertLess(float(np.abs(out[side]).max()), 1.0)
            self.assertEqual(float(out[side][-1]), 0.0)


class OnlyTest(unittest.TestCase):
    def test_only_plays_the_channels_it_names(self):
        self.assertTrue(np.array_equal(left(DUO, only=["high"]), hand([(0, HUM, 60, 11, 1.0)], 8000)))
        self.assertTrue(np.array_equal(left(DUO, only=["deep"]), hand([(800, HISS, 60, 22, 1.0)], 8000)))

    def test_only_keeps_the_length_of_the_whole_song(self):
        whole = render_song(DUO, rate=RATE)
        part = render_song(DUO, only=["high"], rate=RATE)
        self.assertEqual((part["rows"], part["seconds"]), (whole["rows"], whole["seconds"]))
        self.assertEqual((len(part["left"]), len(part["right"])), (8000, 8000))

    def test_only_naming_every_channel_or_left_out_plays_the_whole_song(self):
        whole = render_song(SONG, rate=RATE)
        for only in (["tune", "low", "beat"], ["beat", "tune", "low", "nobody"], None):
            out = render_song(SONG, only=only, rate=RATE)
            self.assertTrue(np.array_equal(out["left"], whole["left"]), only)
            self.assertTrue(np.array_equal(out["right"], whole["right"]), only)

    def test_only_naming_none_of_the_channels_is_silence(self):
        for only in ([], ["nobody"]):
            out = render_song(DUO, only=only, rate=RATE)
            self.assertEqual(len(out["left"]), 8000, only)
            self.assertFalse(out["left"].any(), only)
            self.assertFalse(out["right"].any(), only)

    def test_tuple_does_for_a_list_of_names(self):
        self.assertTrue(np.array_equal(left(DUO, only=("deep",)), left(DUO, only=["deep"])))

    def test_only_and_dry_go_together(self):
        got = render_song(SONG, only=["tune"], dry=True, rate=RATE)
        song = copy.deepcopy(SONG)
        del song["channels"]["tune"]["echo"]
        wanted = render_song(song, only=["tune"], rate=RATE)
        self.assertTrue(np.array_equal(got["left"], wanted["left"]))
        self.assertTrue(np.array_equal(got["right"], wanted["right"]))


class LengthTest(unittest.TestCase):
    def test_rows_are_those_of_its_patterns_laid_end_to_end(self):
        self.assertEqual(render_song(SONG, rate=RATE)["rows"], 24)
        self.assertEqual(render_song(solo("c32"), rate=RATE)["rows"], 0.5)
        self.assertEqual(render_song(DUO, rate=RATE)["rows"], lay(DUO)[1])

    def test_seconds_are_the_time_its_rows_take_at_any_rate(self):
        # 24 rows of three ticks are 72 sixtieths of a second.
        self.assertEqual(render_song(SONG, rate=RATE)["seconds"], 1.2)
        self.assertEqual(render_song(SONG, only=[], rate=11025)["seconds"], 1.2)
        self.assertEqual(render_song(solo("c1", HUM, 6), rate=RATE)["seconds"], 1.6)
        self.assertEqual(render_song(solo("c1", HUM, 1), rate=RATE)["seconds"], 16 / 60)

    def test_sides_run_half_a_second_past_the_last_row(self):
        for text, size in (("c2", 7200), ("c", 4400), ("c32", 4200), ("r1 r1", 16800)):
            out = render_song(solo(text), rate=RATE)
            self.assertEqual((len(out["left"]), len(out["right"])), (size, size), text)

    def test_part_of_a_sample_rounds_the_length_up(self):
        # A row of one tick is 183.75 samples at 11025 Hz, and half a second 5512.5.
        self.assertEqual(len(left(solo("c", HUM, 1), 11025)), 5697)
        self.assertEqual(len(left(solo("c32", HUM, 1))), 4067)

    def test_song_that_plays_no_pattern_is_half_a_second_of_silence(self):
        out = render_song({**SONG, "order": []}, rate=RATE)
        self.assertEqual((out["rows"], out["seconds"]), (0, 0))
        self.assertEqual((len(out["left"]), len(out["right"])), (4000, 4000))
        self.assertFalse(out["left"].any() or out["right"].any())

    def test_rate_left_out_is_44100(self):
        song = solo("c32", HISS, 1, pan=0.4)
        out = render_song(song)
        self.assertEqual(len(out["left"]), 22418)
        at_44100 = render_song(song, rate=44100)
        self.assertTrue(np.array_equal(out["left"], at_44100["left"]))
        self.assertTrue(np.array_equal(out["right"], at_44100["right"]))

    def test_gives_the_two_sides_the_rows_and_the_seconds(self):
        out = render_song(SONG, rate=RATE)
        self.assertEqual(sorted(out), ["left", "right", "rows", "seconds"])
        for side in ("left", "right"):
            self.assertEqual(out[side].dtype, np.float32)
            self.assertEqual(out[side].shape, (13600,))
        self.assertIsNot(out["left"], out["right"])

    def test_song_is_left_as_it_was(self):
        song = copy.deepcopy(SONG)
        render_song(song, only=["tune", "beat"], rate=RATE)
        self.assertEqual(song, SONG)


class StemTest(unittest.TestCase):
    NOTE = [(0, HUM, 60, 11, 1.0)]

    def test_stem_is_the_notes_of_its_channel_at_the_gain_of_the_song_and_finished_without_the_fade(self):
        # Twelve rows are 4800 samples, and the last note is over well before they end.
        notes = [(0, HUM, 60, 5, 1.0), (800, HUM, 64, 5, 1.0), (1600, HUM, 72, 11, 1.0)]
        self.assertTrue(np.array_equal(stem(solo("l8 c e > c4 r4")), looped(notes, 4800, 4800)))

    def test_every_channel_has_a_stem_and_they_come_in_the_order_lay_gives_them(self):
        song = tunes({"a": {"bass": "c"}, "b": {"lead": "d", "bass": "e", "pad": "f"}}, ["a", "b"])
        song["channels"]["spare"] = {"inst": "lead"}
        self.assertEqual(list(stems(song)), ["bass", "lead", "pad"])
        song["order"] = ["b", "a"]
        self.assertEqual(list(stems(song)), ["lead", "bass", "pad"])
        self.assertEqual(list(stems(SONG)), list(lay(SONG)[0]))

    def test_stem_holds_its_own_channel_and_none_of_the_others(self):
        # The bass plays twice, 22 ticks and 2 of release each time, and its second note ends as the loop does.
        bass = SONG["instruments"]["bass"]
        wanted = looped([(0, bass, 36, 22, 1.0), (6400, bass, 36, 22, 1.0)], 9600, 9600, 0.8 * 0.35)
        self.assertTrue(np.array_equal(stems(SONG)["low"], wanted))

    def test_channel_that_only_rests_has_a_silent_stem_as_long_as_the_others(self):
        out = stems(tunes({"a": {"lead": "c4", "bass": "r8"}}, ["a"]))
        self.assertEqual((len(out["lead"]), len(out["bass"])), (1600, 1600))
        self.assertTrue(out["lead"].any())
        self.assertFalse(out["bass"].any())

    def test_stem_has_the_echo_of_its_channel(self):
        got = stem(solo("c8 r4.", echo={"rows": 3, "gain": 0.5}))
        self.assertTrue(np.array_equal(got, looped([(0, HUM, 60, 5, 1.0), (1200, HUM, 60, 5, 0.5)], 3200, 3200)))
        self.assertFalse(np.array_equal(got, stem(solo("c8 r4."))))

    def test_drum_channel_has_a_stem_as_a_tuned_one_has(self):
        song = solo("k4 s4", drums=True)
        song["instruments"]["two"] = HISS
        song["drums"] = {"k": {"inst": "two", "note": "c2"}, "s": {"inst": "one", "note": 62.5, "vel": 0.5}}
        # The second drum sounds for 13 ticks from sample 1600, which takes it 134 samples past the end.
        wanted = looped([(0, HISS, 36.0, 11, 1.0), (1600, HUM, 62.5, 11, 0.5)], 3334, 3200)
        self.assertTrue(np.array_equal(stem(song), wanted))

    def test_gain_of_a_song_left_out_is_0_35(self):
        self.assertTrue(np.array_equal(stem(solo("c4 r4")), looped(self.NOTE, 3200, 3200, 0.35)))
        self.assertTrue(np.array_equal(stem({**solo("c4 r4"), "gain": 0.35}), stem(solo("c4 r4"))))

    def test_level_of_a_stem_is_the_gain_of_its_channel_times_the_gain_of_the_song(self):
        self.assertTrue(np.array_equal(stem(solo("c4 r4", gain=0.8)), looped(self.NOTE, 3200, 3200, 0.8 * 0.35)))
        for channel, whole in ((1, 0.2), (0.5, 0.6), (1.7, 1), (0.3, 3.5)):
            got = stem({**solo("c4 r4", gain=channel), "gain": whole})
            self.assertTrue(np.array_equal(got, looped(self.NOTE, 3200, 3200, channel * whole)), (channel, whole))

    def test_loud_stem_stays_under_full_scale(self):
        out = stem({**solo("c1", HISS, gain=4), "gain": 2})
        self.assertGreater(float(np.abs(out).max()), 0.98)
        self.assertLess(float(np.abs(out).max()), 1.0)

    def test_stem_is_not_panned_and_the_pan_of_its_channel_comes_back_with_it(self):
        wanted = looped(self.NOTE, 3200, 3200)
        for pan in (-1, 1, 0.3, 0):
            out = render_stems(solo("c4 r4", pan=pan), RATE)["stems"][0]
            self.assertEqual(out["pan"], pan)
            self.assertTrue(np.array_equal(out["samples"], wanted), pan)

    def test_pan_left_out_comes_back_as_0(self):
        song = solo("c4 r4")
        del song["channels"]["voice"]["pan"]
        out = render_stems(song, RATE)["stems"][0]
        self.assertEqual(out["pan"], 0)
        self.assertTrue(np.array_equal(out["samples"], looped(self.NOTE, 3200, 3200)))


class JoinTest(unittest.TestCase):
    # The note starts at sample 1600 of 3200 and sounds for 20 ticks, 2667 samples: 1067 of them past the end.
    LATE = [(1600, RING, 60, 11, 1.0)]

    def test_release_that_runs_past_the_last_row_comes_back_on_at_the_start(self):
        self.assertTrue(np.array_equal(stem(solo("r4 c4", RING)), looped(self.LATE, 4267, 3200)))

    def test_start_of_a_stem_holds_what_sounds_past_its_end_and_nothing_is_cut(self):
        played = raw(self.LATE, 4267).astype(np.float64) * 0.35
        self.assertFalse(played[:1600].any())
        # Past the output stage its high-pass has been over every sample. Before it they can be told apart.
        with mock.patch("rowsynth.song.finish", side_effect=lambda out, *rest, **keys: out):
            got = stem(solo("r4 c4", RING))
        self.assertGreater(float(np.abs(got[:1067]).max()), 0.2)
        self.assertTrue(np.array_equal(got[:1067], played[3200:].astype(np.float32)))
        self.assertFalse(got[1067:1600].any())
        self.assertTrue(np.array_equal(got[1600:], played[1600:3200].astype(np.float32)))

    def test_wave_runs_on_over_the_join_in_steps_like_those_on_either_side_of_it(self):
        got = stem(solo("r4 c4", RING)).astype(np.float64)
        before, after = abs(got[-1] - got[-2]), abs(got[1] - got[0])
        self.assertGreater(min(before, after), 0.02)
        self.assertLess(abs(got[0] - got[-1]), max(before, after))
        self.assertGreater(abs(got[0] - got[-1]), min(before, after))

    def test_what_comes_back_is_added_to_what_plays_at_the_start(self):
        notes = [(0, RING, 60, 11, 1.0), (1600, RING, 67, 11, 1.0)]
        self.assertTrue(np.array_equal(stem(solo("c4 g4", RING)), looped(notes, 4267, 3200)))

    def test_echo_that_falls_past_the_last_row_comes_back_on_at_the_start(self):
        # The note is at sample 2400 of 3200. Its echo, four rows later, starts at 4000 and is over by 4934.
        notes = [(2400, HUM, 60, 5, 1.0), (4000, HUM, 60, 5, 0.5)]
        got = stem(solo("r4 r8 c8", echo={"rows": 4, "gain": 0.5}))
        self.assertTrue(np.array_equal(got, looped(notes, 4934, 3200)))
        self.assertFalse(np.array_equal(got, looped(notes[:1], 4934, 3200)))

    def test_tail_many_times_as_long_as_the_loop_goes_round_as_often_as_it_takes(self):
        # Half a row of one tick is 67 samples, and the note sounds for ten ticks, 1334 samples: twenty times round.
        got = stem(solo("c32", RING, 1))
        self.assertEqual(len(got), 67)
        self.assertTrue(np.array_equal(got, looped([(0, RING, 60, 1, 1.0)], 1334, 67)))

    def test_last_release_is_played_as_far_as_its_ticks_reach_and_part_of_a_sample_rounds_that_up(self):
        # Held for 3 ticks from sample 2800 and let go for 2.5, the note is 733.33 samples long and is played
        # as far as 3534. Its last tick, cut short there, would have run on to 3600.
        inst = {**HUM, "gate": 1, "release": 2.5}
        got = stem(solo("r4 r8 r16 c16", inst))
        self.assertTrue(np.array_equal(got, looped([(2800, inst, 60, 3, 1.0)], 3534, 3200)))
        self.assertFalse(np.array_equal(got, looped([(2800, inst, 60, 3, 1.0)], 3533, 3200)))
        self.assertFalse(np.array_equal(got, looped([(2800, inst, 60, 3, 1.0)], 3600, 3200)))

    def test_end_of_a_stem_is_not_faded(self):
        # Noise at full level, held right up to the last sample.
        out = stem(solo("c1", {**HISS, "gate": 1}))
        self.assertEqual(len(out), 6400)
        self.assertGreater(float(np.abs(out[-40:]).min()), 0.2)


class LoopTest(unittest.TestCase):
    def test_gives_the_stems_the_rows_the_loop_samples_and_the_seconds(self):
        out = render_stems(SONG, rate=RATE)
        self.assertEqual(sorted(out), ["loop_samples", "rows", "seconds", "stems"])
        self.assertEqual((out["rows"], out["loop_samples"], out["seconds"]), (24, 9600, 1.2))
        self.assertIsInstance(out["loop_samples"], int)
        self.assertEqual([(found["name"], found["pan"]) for found in out["stems"]],
                         [("tune", -0.5), ("low", 0.6), ("beat", 0)])
        for found in out["stems"]:
            self.assertEqual(sorted(found), ["name", "pan", "samples"])
            self.assertEqual(found["samples"].dtype, np.float32)
            self.assertEqual(found["samples"].shape, (9600,))

    def test_loop_is_as_long_as_the_rows_of_the_song_and_half_a_sample_rounds_it_up(self):
        # A row of one tick is 183.75 samples at 11025 Hz, and six of them 1102.5. At 8000 Hz it is 133.33.
        for text, rate, size in (("c", 11025, 184), ("c4", 11025, 735), ("c4.", 11025, 1103), ("c32", 8000, 67),
                                 ("c", 8000, 133), ("r1 r1", 8000, 4267)):
            out = render_stems(solo(text, HUM, 1), rate)
            self.assertEqual(out["loop_samples"], size, text)
            self.assertEqual(len(out["stems"][0]["samples"]), size, text)

    def test_seconds_are_how_long_the_loop_lasts_which_is_not_quite_what_its_rows_take(self):
        out = render_stems(solo("c4.", HUM, 1), 11025)
        self.assertEqual(out["seconds"], 1103 / 11025)
        self.assertEqual(out["rows"], 6)
        self.assertEqual(render_song(solo("c4.", HUM, 1), rate=11025)["seconds"], 0.1)

    def test_rows_are_those_of_its_patterns_laid_end_to_end(self):
        self.assertEqual(render_stems(solo("c32"), RATE)["rows"], 0.5)
        self.assertEqual(render_stems(DUO, RATE)["rows"], lay(DUO)[1])

    def test_rate_left_out_is_44100(self):
        song = solo("c32", HISS, 1)
        out = render_stems(song)
        self.assertEqual((out["loop_samples"], out["seconds"]), (368, 368 / 44100))
        self.assertTrue(np.array_equal(out["stems"][0]["samples"], stem(song, 44100)))

    def test_song_is_left_as_it_was(self):
        song = copy.deepcopy(SONG)
        render_stems(song, RATE)
        self.assertEqual(song, SONG)


class KnownSamplesTest(unittest.TestCase):
    def check(self, out: np.ndarray, known: tuple) -> None:
        spots = [spot for spot, sample in known]
        np.testing.assert_allclose(out[spots], [sample for spot, sample in known], rtol=0, atol=1e-6)

    def test_three_channels_with_an_echo_and_drums(self):
        # The second pattern starts at sample 3200, the first comes round again at 6400, and the rows end at 9600.
        out = render_song(SONG, rate=RATE)
        self.assertEqual(len(out["left"]), 13600)
        self.check(out["left"], ((0, 0.030075), (1, 0.096853), (2, 0.138692), (200, -0.166690), (400, 0.286487),
                                 (800, -0.085787), (801, -0.034899), (1000, 0.330939), (1600, 0.127224),
                                 (1601, 0.041194), (2400, -0.043740), (2401, 0.025360), (3200, 0.173273),
                                 (3201, 0.232579), (3400, 0.486157), (4800, 0.015415), (6400, 0.029408),
                                 (6401, 0.096202), (7200, -0.085787), (9000, 0.337913), (9599, 0.102389),
                                 (9600, 0.147550), (9800, 0.121645), (10000, 0.000321), (13599, 0.0)))
        self.check(out["right"], ((0, 0.033162), (1, 0.079706), (2, 0.114228), (200, -0.104561), (400, 0.049634),
                                  (800, -0.148719), (801, -0.124973), (1000, 0.133003), (1600, 0.133680),
                                  (1601, 0.098979), (2400, -0.058800), (2401, 0.000752), (3200, 0.058547),
                                  (3201, 0.091951), (3400, 0.186352), (4800, 0.015109), (6400, 0.032496),
                                  (6401, 0.079055), (7200, -0.148719), (9000, 0.079863), (9599, 0.048215),
                                  (9600, 0.038778), (9800, 0.050197), (10000, 0.000132), (13599, 0.0)))

    def test_two_of_those_channels_dry_at_11025_hz(self):
        # A row is 551.25 samples.
        out = render_song(SONG, only=["tune", "beat"], dry=True, rate=11025)
        self.assertEqual(len(out["left"]), 18743)
        self.check(out["left"], ((0, 0.018959), (1, 0.065961), (2, 0.095652), (500, -0.101053), (1103, 0.007479),
                                 (1104, 0.043750), (1500, -0.096163), (2205, 0.011676), (2206, 0.051518),
                                 (3308, 0.014888), (3400, -0.459200), (4410, 0.018966), (4411, 0.065968),
                                 (5000, 0.130787), (6616, 0.021567), (8000, -0.049865), (8820, 0.018317),
                                 (8821, 0.065330), (10000, -0.185606), (12000, -0.005739), (13230, 0.000006),
                                 (18742, 0.0)))
        self.check(out["right"], ((0, 0.014571), (1, 0.040326), (2, 0.058501), (500, -0.049595), (1103, 0.003098),
                                  (1104, 0.018122), (1500, -0.039832), (2205, 0.007278), (2206, 0.025872),
                                  (3308, 0.010499), (3400, -0.357816), (4410, 0.014572), (4411, 0.040327),
                                  (5000, 0.052511), (6616, 0.021567), (8000, -0.049865), (8820, 0.013929),
                                  (8821, 0.039695), (10000, -0.076880), (12000, -0.005740), (13230, 0.000001),
                                  (18742, 0.0)))

    def test_second_pattern_into_the_knee_at_44100_and_48000_hz(self):
        song = {**SONG, "order": ["b"], "gain": 0.6}
        # A row is 2205 samples.
        out = render_song(song)
        self.assertEqual(len(out["left"]), 39690)
        self.check(out["left"], ((0, 0.008381), (1, 0.029793), (2, 0.044313), (100, 0.798367), (1000, -0.268035),
                                 (4410, 0.064074), (4411, 0.066551), (6000, -0.137898), (6615, 0.226092),
                                 (6616, 0.083652), (8820, 0.004562), (8821, 0.009551), (9000, 0.228180),
                                 (11025, -0.009748), (13231, -0.001464), (15000, 0.013569), (17640, -0.001164),
                                 (18000, -0.000249), (39689, 0.0)))
        self.check(out["right"], ((0, 0.006441), (1, 0.018232), (2, 0.027119), (100, 0.468204), (1000, -0.148752),
                                  (4410, 0.024697), (4411, 0.026709), (6000, -0.099666), (6615, 0.101945),
                                  (6616, 0.042909), (8820, 0.004177), (8821, 0.009168), (9000, 0.228002),
                                  (11025, -0.009748), (13231, -0.001464), (15000, 0.013569), (17640, -0.001164),
                                  (18000, -0.000249), (39689, 0.0)))
        # And here 2400.
        out = render_song(song, rate=48000)
        self.assertEqual(len(out["left"]), 43200)
        self.check(out["left"], ((0, 0.007707), (1, 0.027412), (2, 0.040798), (100, 0.788918), (1000, -0.282287),
                                 (4800, 0.063277), (4801, 0.065560), (5000, -0.116006), (7200, 0.197128),
                                 (7201, -0.023241), (9600, 0.004108), (9601, 0.008701), (10000, 0.127665),
                                 (12000, -0.009758), (14401, -0.001737), (16000, 0.010295), (19200, -0.001205),
                                 (20000, -0.000052), (43199, 0.0)))
        self.check(out["right"], ((0, 0.005923), (1, 0.016775), (2, 0.024969), (100, 0.460958), (1000, -0.162627),
                                  (4800, 0.024285), (4801, 0.026138), (5000, 0.005131), (7200, 0.089503),
                                  (7201, -0.001807), (9600, 0.003726), (9601, 0.008321), (10000, 0.127585),
                                  (12000, -0.009758), (14401, -0.001737), (16000, 0.010295), (19200, -0.001205),
                                  (20000, -0.000052), (43199, 0.0)))

    def test_stems_of_three_channels_with_an_echo_and_drums(self):
        # The loop is 9600 samples. The echo of the tune's last note starts at 9400 and runs 333 past the end.
        out = stems(SONG)
        self.assertEqual([len(out[name]) for name in ("tune", "low", "beat")], [9600, 9600, 9600])
        self.check(out["tune"], ((0, 0.179789), (1, 0.228408), (2, 0.232516), (100, -0.215540), (332, -0.116887),
                                 (333, -0.091328), (400, 0.343067), (800, -0.046858), (801, 0.007139),
                                 (1000, 0.359862), (3400, 0.553694), (6400, 0.011002), (8800, -0.039444),
                                 (9000, 0.377148), (9400, 0.005273), (9401, 0.026303), (9599, 0.108467)))
        self.check(out["low"], ((0, -0.013058), (1, 0.000425), (2, 0.012742), (200, -0.025362), (1600, 0.094458),
                                (3199, 0.007054), (3200, -0.027141), (3300, -0.002501), (6400, 0.014082),
                                (6401, 0.026926), (7000, 0.148810), (9000, -0.084012), (9599, 0.007054)))
        self.check(out["beat"], ((0, 0.022001), (1, 0.042069), (2, 0.060339), (400, -0.007660), (1600, 0.007817),
                                 (1601, 0.014495), (2400, 0.014460), (2401, 0.028530), (4800, 0.021061),
                                 (8800, 0.014460), (9000, 0.021829), (9199, -0.101733), (9200, -0.035530),
                                 (9599, -0.000003)))

    def test_stems_of_the_second_pattern_at_44100_and_48000_hz(self):
        song = {**SONG, "order": ["b"], "gain": 0.6}
        out = stems(song, 44100)
        self.assertEqual((len(out["tune"]), len(out["beat"])), (17640, 17640))
        self.check(out["tune"], ((0, 0.003585), (1, 0.021363), (2, 0.031769), (100, 0.617557), (1000, -0.220407),
                                 (3307, 0.573444), (3308, 0.189602), (3309, -0.217694), (4410, 0.072759),
                                 (6000, -0.070644), (8820, 0.000711), (9000, 0.000329), (17639, 0.0)))
        self.check(out["beat"], ((0, 0.005522), (1, 0.012582), (2, 0.019526), (100, 0.326847), (1000, -0.091106),
                                 (4410, -0.004450), (4411, -0.002070), (6000, -0.102717), (8820, 0.005522),
                                 (9000, 0.322266), (13230, -0.004450), (15000, 0.019190), (17639, -0.001654)))
        out = stems(song, 48000)
        self.assertEqual((len(out["tune"]), len(out["beat"])), (19200, 19200))
        self.check(out["tune"], ((0, 0.003296), (1, 0.019655), (2, 0.029249), (100, 0.611451), (1000, -0.221104),
                                 (3599, 0.574578), (3600, 0.498441), (3601, -0.091187), (4800, 0.072048),
                                 (6000, 0.237519), (9600, 0.000705), (10000, 0.000146), (19199, 0.0)))
        self.check(out["beat"], ((0, 0.004887), (1, 0.011388), (2, 0.017791), (100, 0.319827), (1000, -0.110362),
                                 (4800, -0.004648), (4801, -0.002457), (6000, -0.004145), (9600, 0.004887),
                                 (10000, 0.180354), (14400, -0.004648), (16000, 0.014559), (19199, -0.001711)))

    def test_stem_of_a_late_note_whose_release_comes_back_on_at_the_start(self):
        out = stem(solo("r4 c4", RING))
        self.assertEqual(len(out), 3200)
        self.check(out, ((0, -0.100789), (1, -0.130219), (2, -0.158581), (500, -0.033602), (1066, 0.006283),
                         (1067, -0.002979), (1599, 0.0), (1600, 0.017603), (1601, 0.029114), (2000, 0.185909),
                         (3199, -0.070221)))

    def test_stem_that_goes_twenty_times_round_at_11025_hz(self):
        # Half a row of one tick is 92 samples, and the note sounds for ten ticks, 1838 samples.
        out = stem(solo("c32", RING, 1), 11025)
        self.assertEqual(len(out), 92)
        self.check(out, ((0, -0.136378), (1, -0.109076), (2, -0.143599), (10, -0.144184), (20, 0.010160),
                         (30, 0.210899), (45, -0.021518), (46, -0.031766), (60, -0.169593), (75, 0.222017),
                         (90, -0.125255), (91, -0.167074)))


class ErrorTest(unittest.TestCase):
    def refused(self, song: dict) -> str:
        """What the error says when a song is refused, which is the same laid out as played."""
        with self.assertRaises(RowSynthError, msg=repr(song)) as laid:
            lay(song)
        with self.assertRaises(RowSynthError, msg=repr(song)) as played:
            render_song(song, rate=RATE)
        self.assertEqual(str(played.exception), str(laid.exception))
        return str(laid.exception)

    def changed(self, pattern: str, channel: str, text: str) -> dict:
        """The song with the text of one channel of one pattern changed."""
        song = copy.deepcopy(SONG)
        song["patterns"][pattern][channel] = text
        return song

    def test_refuses_an_order_that_names_a_pattern_the_song_lacks(self):
        self.assertEqual(self.refused({**SONG, "order": ["a", "c", "b"]}), "order: no pattern called 'c'")
        self.assertEqual(self.refused({**SONG, "order": ["A"]}), "order: no pattern called 'A'")

    def test_refuses_a_pattern_with_text_for_a_channel_the_song_lacks(self):
        self.assertEqual(self.refused(self.changed("b", "pad", "c e g")), "pattern 'b': no channel called 'pad'")

    def test_refuses_text_that_cannot_be_read_by_its_pattern_and_channel(self):
        self.assertEqual(self.refused(self.changed("b", "tune", "c e ? g")),
                         "pattern 'b', channel 'tune': cannot read \"? g\" in a channel")
        self.assertEqual(self.refused(self.changed("a", "beat", "k4 h0")),
                         "pattern 'a', channel 'beat': cannot read \"h0\" in a channel")

    def test_refuses_a_drum_letter_in_a_channel_that_is_not_a_drum_channel(self):
        self.assertEqual(self.refused(self.changed("a", "low", "o2 c k")),
                         "pattern 'a', channel 'low': \"k\" is a drum, and this is not a drum channel")
        song = copy.deepcopy(SONG)
        song["channels"]["beat"]["drums"] = False
        self.assertEqual(self.refused(song),
                         "pattern 'a', channel 'beat': \"k\" is a drum, and this is not a drum channel")

    def test_refuses_a_drum_letter_the_song_has_no_drum_for(self):
        self.assertEqual(self.refused(self.changed("b", "beat", "k t")),
                         "pattern 'b', channel 'beat': \"t\" is a drum, and this is not a drum channel")
        song = copy.deepcopy(SONG)
        del song["drums"]
        self.assertEqual(self.refused(song),
                         "pattern 'a', channel 'beat': \"k\" is a drum, and this is not a drum channel")

    def test_refuses_a_drum_whose_note_is_not_a_note(self):
        song = copy.deepcopy(SONG)
        song["drums"]["h"]["note"] = "x9"
        self.assertEqual(self.refused(song), "pattern 'a', channel 'beat': not a note: 'x9'")

    def test_refuses_a_note_that_names_an_instrument_the_song_lacks(self):
        self.assertEqual(self.refused(self.changed("b", "tune", "c @organ d")),
                         "pattern 'b', channel 'tune': no instrument called 'organ'")

    def test_refuses_a_channel_whose_instrument_the_song_lacks(self):
        song = copy.deepcopy(SONG)
        song["channels"]["low"]["inst"] = "organ"
        self.assertEqual(self.refused(song), "pattern 'a', channel 'low': no instrument called 'organ'")

    def test_refuses_a_drum_whose_instrument_the_song_lacks(self):
        song = copy.deepcopy(SONG)
        del song["instruments"]["tick"]
        self.assertEqual(self.refused(song), "pattern 'a', channel 'beat': no instrument called 'tick'")

    def test_refuses_a_song_that_plays_notes_and_has_no_instruments(self):
        song = copy.deepcopy(SONG)
        del song["instruments"]
        self.assertEqual(self.refused(song), "pattern 'a', channel 'tune': no instrument called 'lead'")

    def test_song_is_refused_whichever_of_its_channels_are_to_play(self):
        song = self.changed("b", "tune", "c @organ d")
        with self.assertRaises(RowSynthError) as caught:
            render_song(song, only=["low"], rate=RATE)
        self.assertEqual(str(caught.exception), "pattern 'b', channel 'tune': no instrument called 'organ'")

    def test_refuses_a_rate_that_makes_no_sense(self):
        for rate in (0, -8000, 8000.5, "8000", None, True, float("inf")):
            with self.assertRaises(RowSynthError, msg=repr(rate)) as caught:
                render_song(SONG, rate=rate)
            self.assertTrue(str(caught.exception).startswith("rate: "), rate)

    def test_refuses_a_dry_that_is_neither_true_nor_false(self):
        for dry in (1, 0, "yes", None):
            with self.assertRaises(RowSynthError, msg=repr(dry)) as caught:
                render_song(SONG, dry=dry, rate=RATE)
            self.assertTrue(str(caught.exception).startswith("dry: "), dry)

    def test_nothing_is_played_when_a_song_is_refused(self):
        cases = ((self.changed("b", "tune", "c e ? g"), {}), (self.changed("b", "beat", "k @organ c"), {}),
                 ({**SONG, "order": ["a", "b", "c"]}, {}), (SONG, {"rate": 0}), (SONG, {"dry": None}))
        for song, keys in cases:
            with mock.patch("rowsynth.song.render_note") as played:
                with self.assertRaises(RowSynthError):
                    render_song(song, **{"rate": RATE, **keys})
            played.assert_not_called()

    def test_stems_are_refused_for_what_the_song_is_refused_for_when_it_is_laid_out(self):
        songs = ({**SONG, "order": ["a", "c", "b"]}, self.changed("b", "pad", "c e g"),
                 self.changed("b", "tune", "c e ? g"), self.changed("a", "low", "o2 c k"),
                 self.changed("b", "tune", "c @organ d"))
        for song in songs:
            with self.assertRaises(RowSynthError, msg=repr(song)) as laid:
                lay(song)
            with self.assertRaises(RowSynthError, msg=repr(song)) as played:
                render_stems(song, RATE)
            self.assertEqual(str(played.exception), str(laid.exception))

    def test_stems_refuse_a_rate_that_makes_no_sense(self):
        for rate in (0, -8000, 8000.5, "8000", None, True, float("inf")):
            with self.assertRaises(RowSynthError, msg=repr(rate)) as caught:
                render_stems(SONG, rate)
            self.assertTrue(str(caught.exception).startswith("rate: "), rate)

    def test_stems_refuse_a_song_with_no_rows(self):
        songs = ({**SONG, "order": []}, tunes({"a": {"lead": "", "bass": " | "}}, ["a"]),
                 tunes({"gap": {}}, ["gap", "gap"]))
        for song in songs:
            with self.assertRaises(RowSynthError, msg=repr(song)) as caught:
                render_stems(song, RATE)
            self.assertEqual(str(caught.exception), "the song has no rows, so there is nothing to loop")

    def test_stems_refuse_rows_that_come_to_less_than_half_a_sample(self):
        # A row of one tick is a third of a sample at 20 Hz.
        with self.assertRaises(RowSynthError) as caught:
            render_stems(solo("c", HUM, 1), 20)
        self.assertEqual(str(caught.exception),
                         "the song's rows come to no samples at a rate of 20, so there is nothing to loop")

    def test_nothing_is_played_when_stems_are_refused(self):
        cases = ((self.changed("b", "tune", "c e ? g"), RATE), ({**SONG, "order": ["a", "b", "c"]}, RATE), (SONG, 0),
                 ({**SONG, "order": []}, RATE), (solo("c", HUM, 1), 20))
        for song, rate in cases:
            with mock.patch("rowsynth.song.render_note") as played:
                with self.assertRaises(RowSynthError):
                    render_stems(song, rate)
            played.assert_not_called()


if __name__ == "__main__":
    unittest.main()
