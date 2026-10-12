"""Tests for rowsynth.gui.songedit: a song changed as a whole, and that what comes back still loads and plays."""
from __future__ import annotations

import json
import unittest
from collections.abc import Callable
from pathlib import Path

import numpy as np

from rowsynth.errors import RowSynthError
from rowsynth.gui import songview
from rowsynth.gui.songedit import (LETTERS, added, allowed, channel_added, channel_copied, channel_dropped,
                                   channel_kind, channel_moved, channel_renamed, copied, dropped, drum_added,
                                   drum_removed, drum_set, gridded, grids, moved, offered, renamed, repeated,
                                   resized, tempos, transposed, with_tempo)
from rowsynth.layers import render_groups
from rowsynth.project import Project
from rowsynth.song import render_loop, render_song, render_stems, up

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

# Each example with its song, and a channel that the song can do without. Every channel of the tune is all that
# its group has, so there is none.
SONGS = (("arcade.json", "tune", None), ("piece.json", "piece", "stamp"))

# A low rate, so that a loop is quick to play.
RATE = 2000

INSTRUMENTS = {"lead": {"vol": 12}, "bass": {"wave": "tri"}, "thud": {"wave": "noise"}}
CHORDS = {"C": {"root": 0, "tones": [0, 4, 7]}, "F": {"root": 5, "tones": [5, 9, 0]},
          "G": {"root": 7, "tones": [7, 11, 2]}}

# The c of octaves 2, 4 and 5 and the e of octave 4, as the numbers of those notes.
C2, C4, E4, C5 = 36, 60, 64, 72


def song(**keys: object) -> dict:
    """A song as a project holds it: two tuned channels and a drum channel, a pattern of one bar and one of two."""
    return {"row_ticks": 6,
            "drums": {"k": {"inst": "thud", "note": "c2", "rows": 2}, "t": {"inst": "thud", "note": "c5", "rows": 8}},
            "channels": {"tune": {"inst": "lead"}, "low": {"inst": "bass", "pan": -0.3},
                         "beat": {"inst": "thud", "drums": True}},
            "patterns": {"a": {"tune": "o4 l4 c e g e", "low": "o2 c1", "beat": "l4 k k k k"},
                         "b": {"tune": "o4 l4 f a > c < a | g2 e2", "low": "o2 f1 | g1"}},
            "order": ["a", "b", "a"],
            "chords": {"a": ["C"], "b": ["F", "G C"]},
            **keys}


def one(texts: dict, **keys: object) -> dict:
    """The song with one pattern, `x`, that has those texts, and no chords written."""
    return song(**{"patterns": {"x": texts}, "order": ["x"], "chords": {}, **keys})


def project(made: dict) -> Project:
    """The project a song is in, as `s`. The check that loading makes refuses a song that is wrong."""
    return Project.from_dict({"instruments": INSTRUMENTS, "chords": CHORDS, "songs": {"s": made}})


def loop(made: dict) -> dict:
    """The loop of a song, once it has passed the project's check."""
    return render_loop(project(made).song("s"), RATE)


def changed(test: unittest.TestCase, call: Callable[..., dict], given: dict, *more: object) -> dict:
    """What a function makes of a song, once the song handed in is seen to be as it was and the new one to play."""
    before = json.dumps(given)
    made = call(given, *more)
    test.assertEqual(json.dumps(given), before)
    loop(made)
    return made


def read(made: dict, name: str) -> dict[str, tuple[list[tuple], float]]:
    """A pattern as it is read back: for each channel it has text for, its notes as (row, rows, pitch), and its rows."""
    return {held.name: ([(note.row, note.rows, note.pitch) for note in held.notes], held.rows)
            for held in songview.pattern(project(made).song("s"), name).channels if held.written}


def refusal(test: unittest.TestCase, call: Callable[..., object], *given: object) -> str:
    """The message a call is refused with."""
    with test.assertRaises(RowSynthError) as caught:
        call(*given)
    return str(caught.exception)


def same(test: unittest.TestCase, one_loop: dict, other: dict) -> None:
    """Two loops are sample for sample the same."""
    for side in ("left", "right"):
        test.assertTrue(np.array_equal(one_loop[side], other[side]), side)


def edits(given: dict, spare: str | None) -> dict[str, dict]:
    """Every function once on a song, by what it is called. `spare` is a channel the song can do without.

    With none, the channel that is dropped is one that was added first.
    """
    order, channels = given["order"], list(given["channels"])
    last = len(order) - 1
    extra = channel_added(given, "extra", given["channels"][channels[0]]["inst"])
    return {
        "moved": moved(given, 0, last),
        "repeated": repeated(given, last),
        "copied": copied(given, 0),
        "added": added(given, last, 2),
        "dropped": dropped(given, 0),
        "renamed": renamed(given, order[0], "other"),
        "resized longer": resized(given, order[0], 9),
        "resized shorter": resized(given, order[0], 1),
        "channel_added": extra,
        "channel_dropped": channel_dropped(given, spare) if spare else channel_dropped(extra, "extra"),
        "channel_renamed": channel_renamed(given, channels[0], "other"),
        "channel_moved": channel_moved(given, channels[0], len(channels) - 1),
        "transposed": transposed(given, -3),
        "with_tempo": with_tempo(given, tempos(given.get("rows_per_beat", 4))[0][0]),
    }


def scribble(value: object) -> None:
    """Empty every dict and every list that a value holds, however deep."""
    if isinstance(value, (dict, list)):
        held = list(value.values()) if isinstance(value, dict) else list(value)
        value.clear()
        for part in held:
            scribble(part)


class Order(unittest.TestCase):
    def test_a_block_is_moved_to_another_place_and_the_rest_keep_their_order(self) -> None:
        given = song(order=["a", "b", "a", "b"])
        self.assertEqual(changed(self, moved, given, 0, 2)["order"], ["b", "a", "a", "b"])
        self.assertEqual(changed(self, moved, given, 3, 0)["order"], ["b", "a", "b", "a"])
        self.assertEqual(changed(self, moved, given, 1, 1), given)

    def test_a_pattern_is_played_once_more_right_after_its_place(self) -> None:
        made = changed(self, repeated, song(), 1)
        self.assertEqual(made["order"], ["a", "b", "b", "a"])
        self.assertEqual(made["patterns"], song()["patterns"])

    def test_a_place_the_order_does_not_have_is_refused_by_all_of_them(self) -> None:
        given = song()
        for call, more, wrong in ((moved, (3, 0), 3), (moved, (0, -1), -1), (repeated, (3,), 3),
                                  (copied, (1.0,), 1.0), (added, (True, 1), True), (dropped, ("0",), "0")):
            with self.subTest(call=call.__name__, more=more):
                self.assertEqual(refusal(self, call, given, *more),
                                 f"{wrong!r} is no place in the order, which has 3, counted from 0")

    def test_a_copy_is_a_new_pattern_under_the_next_free_name_played_right_after(self) -> None:
        once = changed(self, copied, song(), 0)
        self.assertEqual(once["order"], ["a", "a2", "b", "a"])
        self.assertEqual(once["patterns"]["a2"], once["patterns"]["a"])
        self.assertIsNot(once["patterns"]["a2"], once["patterns"]["a"])
        self.assertEqual(list(once["patterns"]), ["a", "b", "a2"])
        # The song has an a2 now, so the next copy of a is a3. A name that ends on a number counts on from it.
        self.assertEqual(copied(once, 0)["order"], ["a", "a3", "a2", "b", "a"])
        self.assertEqual(copied(renamed(song(), "b", "tune2"), 1)["order"], ["a", "tune2", "tune3", "a"])

    def test_a_copy_carries_the_chords_of_its_pattern_and_takes_the_name_it_is_given(self) -> None:
        made = changed(self, copied, song(), 1, "bridge")
        self.assertEqual(made["order"], ["a", "b", "bridge", "a"])
        self.assertEqual(made["chords"], {"a": ["C"], "b": ["F", "G C"], "bridge": ["F", "G C"]})
        self.assertIsNot(made["chords"]["bridge"], made["chords"]["b"])
        # A pattern with no chords written has none copied.
        self.assertEqual(copied(song(chords={"b": ["F", "G"]}), 0)["chords"], {"b": ["F", "G"]})

    def test_an_added_pattern_is_bars_of_rests_for_the_first_channel(self) -> None:
        made = changed(self, added, song(), 0, 2)
        self.assertEqual(made["order"], ["a", "a2", "b", "a"])
        self.assertEqual(made["patterns"]["a2"], {"tune": "r1 | r1"})
        self.assertEqual(read(made, "a2"), {"tune": ([], 32.0)})
        self.assertEqual(made["chords"], song()["chords"])
        named = changed(self, added, song(), 2, 1, "end")
        self.assertEqual((named["order"], named["patterns"]["end"]), (["a", "b", "a", "end"], {"tune": "r1"}))
        # A bar is four beats, whatever the rows of a beat.
        self.assertEqual(read(changed(self, added, song(rows_per_beat=2), 0, 2), "a2"), {"tune": ([], 16.0)})

    def test_bars_that_are_no_whole_number_and_a_song_with_no_channel_are_refused(self) -> None:
        for bars in (0, 1.5, True):
            for call in (added, resized):
                with self.subTest(call=call.__name__, bars=bars):
                    self.assertEqual(refusal(self, call, song(), 0 if call is added else "a", bars),
                                     f"the bars of a pattern are a whole number, 1 or more, not {bars!r}")
        self.assertEqual(refusal(self, added, song(channels={}, patterns={"a": {}}, order=["a"], chords={}), 0, 1),
                         "the song has no channel, and a pattern needs one to count its rows out in rests")

    def test_a_dropped_place_takes_the_pattern_no_one_plays_any_more_and_its_chords(self) -> None:
        made = changed(self, dropped, song(), 1)
        self.assertEqual(made["order"], ["a", "a"])
        self.assertEqual((list(made["patterns"]), made["chords"]), (["a"], {"a": ["C"]}))

    def test_a_pattern_that_is_still_played_or_that_the_intro_names_is_kept(self) -> None:
        still = changed(self, dropped, song(), 0)
        self.assertEqual(still["order"], ["b", "a"])
        self.assertEqual((still["patterns"], still["chords"]), (song()["patterns"], song()["chords"]))
        given = song(order=["a", "b"], intro={"order": ["b"]})
        named = changed(self, dropped, given, 1)
        self.assertEqual(named["order"], ["a"])
        self.assertEqual((named["patterns"], named["chords"], named["intro"]),
                         (given["patterns"], given["chords"], {"order": ["b"]}))

    def test_the_last_place_of_an_order_is_not_dropped(self) -> None:
        self.assertEqual(refusal(self, dropped, song(order=["b"]), 0),
                         "pattern 'b' is all the order has, and a song with none has nothing to play")

    def test_a_renamed_pattern_has_its_new_name_everywhere_and_keeps_its_place(self) -> None:
        given = song(intro={"order": ["a", "b"], "tail": 1.0})
        made = changed(self, renamed, given, "a", "verse")
        self.assertEqual(list(made["patterns"]), ["verse", "b"])
        self.assertEqual(made["patterns"]["verse"], given["patterns"]["a"])
        self.assertEqual(made["order"], ["verse", "b", "verse"])
        self.assertEqual(list(made["chords"].items()), [("verse", ["C"]), ("b", ["F", "G C"])])
        self.assertEqual(made["intro"], {"order": ["verse", "b"], "tail": 1.0})

    def test_a_pattern_the_song_does_not_have_is_refused(self) -> None:
        for call, *more in ((renamed, "z", "y"), (resized, "z", 2), (renamed, 3, "y")):
            with self.subTest(call=call.__name__):
                self.assertEqual(refusal(self, call, song(), *more), f"the song has no pattern called {more[0]!r}")


class Resize(unittest.TestCase):
    def test_a_note_that_crosses_the_new_end_is_cut_short_there_and_one_after_it_goes(self) -> None:
        given = one({"tune": "o4 c2 e1 g2", "low": "o2 c1 | c1", "beat": "l4 k k k k k k k k"})
        made = changed(self, resized, given, "x", 1)
        self.assertEqual(read(made, "x"), {
            "tune": ([(0.0, 8.0, C4), (8.0, 8.0, E4)], 16.0),
            "low": ([(0.0, 16.0, C2)], 16.0),
            "beat": ([(0.0, 2.0, C2), (4.0, 2.0, C2), (8.0, 2.0, C2), (12.0, 2.0, C2)], 16.0),
        })
        # The texts are the editor's.
        self.assertEqual((made["patterns"]["x"]["tune"], made["patterns"]["x"]["low"]), ("o4 c2 e2", "o2 c1"))

    def test_a_longer_pattern_is_filled_with_rests_in_every_channel_it_has_text_for(self) -> None:
        made = changed(self, resized, song(), "a", 3)
        was = read(song(), "a")
        self.assertEqual(read(made, "a"), {channel: (notes, 48.0) for channel, (notes, _) in was.items()})
        self.assertEqual(made["patterns"]["a"]["low"], "o2 c1 | r1 | r1")
        self.assertEqual(made["patterns"]["b"], song()["patterns"]["b"])
        self.assertEqual(read(resized(song(rows_per_beat=2), "a", 3), "a")["low"], ([(0.0, 8.0, C2)], 24.0))
        # A channel the pattern has no text for is given none, and a pattern with no text has the first channel rest.
        self.assertEqual(list(resized(song(), "b", 3)["patterns"]["b"]), ["tune", "low"])
        empty = song(patterns={**song()["patterns"], "e": {}})
        self.assertEqual(changed(self, resized, empty, "e", 2)["patterns"]["e"], {"tune": "r1 | r1"})

    def test_a_hit_that_crosses_the_new_end_is_cut_to_a_length_its_text_can_say(self) -> None:
        # The hit begins at row 11 and is held for 8 rows. Its 5 rows of room are no length, and 4 are.
        given = one({"low": "o2 c1 | c1", "beat": "r2 r8. t2 r2 r4 r16"})
        self.assertEqual(read(given, "x")["beat"], ([(11.0, 8.0, C5)], 32.0))
        made = changed(self, resized, given, "x", 1)
        self.assertEqual(read(made, "x")["beat"], ([(11.0, 4.0, C5)], 16.0))
        self.assertEqual(made["patterns"]["x"]["beat"], "r2 r8. t4 r16")

    def test_the_chords_are_cut_off_and_added_bars_have_the_chord_of_the_last(self) -> None:
        self.assertEqual(resized(song(), "b", 1)["chords"], {"a": ["C"], "b": ["F"]})
        self.assertEqual(resized(song(), "b", 4)["chords"], {"a": ["C"], "b": ["F", "G C", "G C", "G C"]})
        self.assertEqual(resized(song(chords={"a": ["C"]}), "b", 4)["chords"], {"a": ["C"]})
        self.assertEqual(resized(song(chords={"a": ["C"], "b": []}), "b", 4)["chords"], {"a": ["C"], "b": []})

    def test_a_pattern_made_as_long_as_it_is_keeps_every_text_as_it_was(self) -> None:
        # Neither text is as the editor would write it, and neither is written again.
        given = one({"tune": "o4 c4 c4 c4 c4 | c2 c2", "low": "o2 c1 o2 c1"})
        self.assertEqual(changed(self, resized, given, "x", 2), given)

    def test_a_pattern_off_the_grid_is_refused(self) -> None:
        self.assertEqual(refusal(self, resized, one({"tune": "o4 c12 c12 c12 r4 r2"}), "x", 2),
                         "pattern 'x' has notes off the grid of rows, and such a pattern is not made longer or shorter")


class Channels(unittest.TestCase):
    def test_an_added_channel_is_the_last_and_has_no_text_in_any_pattern(self) -> None:
        made = changed(self, channel_added, song(), "pad", "bass")
        self.assertEqual(list(made["channels"]), ["tune", "low", "beat", "pad"])
        self.assertEqual(made["channels"]["pad"], {"inst": "bass"})
        self.assertEqual(made["patterns"], song()["patterns"])
        self.assertEqual(changed(self, channel_added, song(), "hat", "thud", True)["channels"]["hat"],
                         {"inst": "thud", "drums": True})

    def test_a_dropped_channel_is_gone_from_the_channels_and_from_every_pattern(self) -> None:
        made = changed(self, channel_dropped, song(), "low")
        self.assertEqual(list(made["channels"]), ["tune", "beat"])
        self.assertEqual({name: list(texts) for name, texts in made["patterns"].items()},
                         {"a": ["tune", "beat"], "b": ["tune"]})

    def test_a_channel_with_the_only_text_of_a_pattern_is_not_dropped(self) -> None:
        given = song(patterns={**song()["patterns"], "solo": {"low": "o2 c1"}})
        self.assertEqual(refusal(self, channel_dropped, given, "low"),
                         "channel 'low' has the only text of pattern 'solo', which would be left with no rows")
        self.assertNotIn("beat", changed(self, channel_dropped, given, "beat")["channels"])

    def test_a_renamed_channel_has_its_new_name_everywhere_and_keeps_its_place(self) -> None:
        made = changed(self, channel_renamed, song(), "low", "deep")
        self.assertEqual(list(made["channels"].items()), [("tune", {"inst": "lead"}),
                                                          ("deep", {"inst": "bass", "pan": -0.3}),
                                                          ("beat", {"inst": "thud", "drums": True})])
        self.assertEqual(list(made["patterns"]["a"].items()),
                         [("tune", "o4 l4 c e g e"), ("deep", "o2 c1"), ("beat", "l4 k k k k")])
        self.assertEqual(list(made["patterns"]["b"]), ["tune", "deep"])

    def test_a_moved_channel_is_listed_at_its_new_place_and_the_song_sounds_the_same(self) -> None:
        given = song()
        made = changed(self, channel_moved, given, "beat", 0)
        self.assertEqual(list(made["channels"]), ["beat", "tune", "low"])
        self.assertEqual(list(changed(self, channel_moved, given, "tune", 2)["channels"]), ["low", "beat", "tune"])
        self.assertEqual((made["channels"], made["patterns"]), (given["channels"], given["patterns"]))
        # The stems are added up in the order the texts come to the channels, and that is as it was.
        stems = [stem["name"] for stem in render_stems(project(made).song("s"), RATE)["stems"]]
        self.assertEqual(stems, ["tune", "low", "beat"])
        same(self, loop(made), loop(given))

    def test_a_channel_the_song_does_not_have_and_a_place_that_is_none_are_refused(self) -> None:
        for call, *more in ((channel_dropped,), (channel_renamed, "new"), (channel_moved, 0)):
            with self.subTest(call=call.__name__):
                self.assertEqual(refusal(self, call, song(), "pad", *more), "the song has no channel called 'pad'")
        for to in (3, -1, True, "0"):
            with self.subTest(to=to):
                self.assertEqual(refusal(self, channel_moved, song(), "low", to),
                                 f"{to!r} is no place among the channels, of which there are 3, counted from 0")


class Kinds(unittest.TestCase):
    def test_a_copied_channel_has_the_settings_and_the_texts_of_its_channel_and_stands_after_it(self) -> None:
        given = song()
        made = changed(self, channel_copied, given, "low")
        self.assertEqual((list(made["channels"]), made["channels"]["low2"]),
                         (["tune", "low", "low2", "beat"], {"inst": "bass", "pan": -0.3}))
        self.assertEqual([list(texts) for texts in made["patterns"].values()],
                         [["tune", "low", "low2", "beat"], ["tune", "low", "low2"]])
        self.assertEqual((read(made, "b")["low2"], read(made, "a")["low2"]),
                         (read(given, "b")["low"], read(given, "a")["low"]))
        # The next copy has the next free name, a name can be given, and it is held to what a name can be.
        again = channel_copied(made, "low")
        self.assertEqual((list(again["channels"])[1:4], list(channel_copied(given, "beat", "hats")["channels"])[-1]),
                         (["low", "low3", "low2"], "hats"))
        self.assertEqual([refusal(self, channel_copied, given, *names) for names in (("low", "tune"), ("none",))],
                         ["the song has a channel called 'tune' already", "the song has no channel called 'none'"])

    def test_a_channel_with_no_notes_is_made_a_drum_channel_or_a_tuned_one_and_one_with_notes_is_refused(self) -> None:
        given = channel_added(channel_added(song(), "spare", "thud"), "hits", "thud", True)
        # Rests alone are read either way, so a channel that has only those changes its kind as well.
        given["patterns"]["a"].update(spare="r1", hits="r1")
        drums = changed(self, channel_kind, given, "spare", True)
        self.assertEqual((drums["channels"]["spare"], drums["patterns"]),
                         ({"inst": "thud", "drums": True}, given["patterns"]))
        tuned = changed(self, channel_kind, given, "hits", False)
        self.assertEqual((tuned["channels"]["hits"], channel_kind(given, "hits", True)), ({"inst": "thud"}, given))
        noted = "has notes in pattern 'a': a channel is made a"
        self.assertEqual([refusal(self, channel_kind, given, *asked)
                          for asked in (("tune", True), ("beat", False), ("none", True))],
                         [f"channel 'tune' {noted} drum channel while it has none",
                          f"channel 'beat' {noted} tuned channel while it has none",
                          "the song has no channel called 'none'"])


class Drums(unittest.TestCase):
    def test_an_added_drum_is_the_last_of_the_table_and_a_song_with_no_table_gets_one(self) -> None:
        self.assertEqual(LETTERS, "kshHt")
        given = song()
        made = changed(self, drum_added, given, "H", "bass")
        self.assertEqual((list(made["drums"]), made["drums"]["H"]), (["k", "t", "H"], {"inst": "bass", "note": "c4"}))
        self.assertEqual(drum_added(given, "s", "thud", "a2")["drums"]["s"], {"inst": "thud", "note": "a2"})
        bare = one({"tune": "o4 c1"})
        del bare["drums"]
        self.assertEqual(changed(self, drum_added, bare, "k", "thud")["drums"], {"k": {"inst": "thud", "note": "c4"}})
        self.assertEqual([refusal(self, drum_added, given, *asked)
                          for asked in (("x", "thud"), ("kk", "thud"), (None, "thud"), ("k", "thud"),
                                        ("s", "thud", "no"))],
                         ["a drum is called by one of the letters k s h H t, not 'x'",
                          "a drum is called by one of the letters k s h H t, not 'kk'",
                          "a drum is called by one of the letters k s h H t, not None",
                          "the song has a drum 'k' already", "not a note: 'no'"])

    def test_a_drum_has_its_instrument_its_note_its_rows_and_its_velocity_set_and_the_last_two_taken_away(
            self) -> None:
        given = song()
        made = given
        for key, value in (("inst", "bass"), ("note", "e2"), ("rows", 1), ("vel", 0.5)):
            made = changed(self, drum_set, made, "k", key, value)
        self.assertEqual((made["drums"], given["drums"]["k"]),
                         ({"k": {"inst": "bass", "note": "e2", "rows": 1, "vel": 0.5}, "t": given["drums"]["t"]},
                          {"inst": "thud", "note": "c2", "rows": 2}))
        # A hit is held for the rows its drum has, and for those of its text where the drum has none of its own.
        self.assertEqual([read(drum_set(given, "k", "rows", rows), "a")["beat"][0][0][1] for rows in (1, None)],
                         [1.0, 4.0])
        self.assertEqual(drum_set(drum_set(made, "k", "rows", None), "k", "vel", None)["drums"]["k"],
                         {"inst": "bass", "note": "e2"})
        self.assertEqual([refusal(self, drum_set, given, *asked)
                          for asked in (("s", "rows", 1), ("k", "gain", 1), ("k", "inst", None), ("k", "note", None),
                                        ("k", "note", "h9"), ("k", "inst", ""), ("k", "rows", 0), ("k", "rows", "2"),
                                        ("k", "vel", 1.5), ("k", "vel", True))],
                         ["the song has no drum 's'", "a drum has inst, note, rows, vel, and no 'gain'",
                          "drum 'k' cannot do without its inst", "drum 'k' cannot do without its note",
                          "not a note: 'h9'", "the instrument of a drum is a name, not ''",
                          "the rows a drum is held for are a number above 0, not 0",
                          "the rows a drum is held for are a number above 0, not '2'",
                          "the velocity of a drum is a number from 0 to 1, not 1.5",
                          "the velocity of a drum is a number from 0 to 1, not True"])

    def test_a_drum_that_no_pattern_plays_is_removed_and_the_last_takes_the_table_with_it(self) -> None:
        given = song()
        made = changed(self, drum_removed, given, "t")
        self.assertEqual(made["drums"], {"k": given["drums"]["k"]})
        self.assertEqual([refusal(self, drum_removed, given, letter) for letter in ("k", "s")],
                         ["drum 'k' is played in pattern 'a' by channel 'beat', and a drum that is played is not "
                          "removed", "the song has no drum 's'"])
        # With its hits rubbed out the last drum goes, and the song is one with no drum table.
        quiet = dict(made, patterns={**made["patterns"], "a": {**made["patterns"]["a"], "beat": "r1"}})
        self.assertNotIn("drums", changed(self, drum_removed, quiet, "k"))


class Grids(unittest.TestCase):
    #: A song of one pattern with an echo, a drum of rows of its own, a dotted note and triplets, at 4 rows a beat.
    SONG = {"row_ticks": 6,
            "drums": {"k": {"inst": "thud", "note": "c2", "rows": 2}, "t": {"inst": "thud", "note": "c5"}},
            "channels": {"tune": {"inst": "lead", "echo": {"rows": 3, "gain": 0.3}}, "low": {"inst": "bass"},
                         "beat": {"inst": "thud", "drums": True}},
            "patterns": {"a": {"tune": "o4 l8 c. e16 g4 l12 c e g > c4", "low": "o2 c2 g2", "beat": "l4 k t8 k8 k t"}},
            "order": ["a"], "chords": {"a": ["C"]}}

    def test_a_song_at_another_grid_sounds_sample_for_sample_as_it_did(self) -> None:
        given = self.SONG
        # At this rate a tick is a whole number of samples, as it is at the rate the window plays at.
        rate = 2400
        before = render_song(project(given).song("s"), rate=rate)
        self.assertEqual((grids(given), before["rows"], len(before["left"]) > rate),
                         ([1, 2, 3, 4, 6, 8, 12], 16.0, True))
        for rows in grids(given):
            with self.subTest(rows=rows):
                after = render_song(project(gridded(given, rows)).song("s"), rate=rate)
                same(self, after, before)
                self.assertAlmostEqual(after["seconds"], before["seconds"])
                self.assertAlmostEqual(after["rows"], 4.0 * rows)

    def test_the_ticks_of_a_row_the_echoes_and_the_rows_of_the_drums_go_with_the_grid_and_nothing_else_does(
            self) -> None:
        given = self.SONG
        made = changed(self, gridded, given, 8)
        self.assertEqual((made["rows_per_beat"], made["row_ticks"], made["channels"]["tune"]["echo"],
                          made["drums"]["k"], made["drums"]["t"]),
                         (8, 3, {"rows": 6, "gain": 0.3}, {"inst": "thud", "note": "c2", "rows": 4},
                          given["drums"]["t"]))
        self.assertEqual({key: value for key, value in made.items() if key in ("patterns", "order", "chords")},
                         {key: given[key] for key in ("patterns", "order", "chords")})
        # Rows that come out as no whole number are written as they are, and there and back is the song it was.
        self.assertEqual((gridded(given, 2)["channels"]["tune"]["echo"]["rows"], gridded(gridded(given, 3), 4)),
                         (1.5, {**given, "rows_per_beat": 4}))
        # The grid the song has is among those it can be given, whatever its ticks.
        odd = {**given, "row_ticks": 7, "rows_per_beat": 5}
        self.assertEqual((grids(odd), grids({**given, "row_ticks": 1})), ([1, 5, 7], [1, 2, 4]))

    def test_the_window_offers_only_the_grids_at_which_every_written_note_is_on_a_row(self) -> None:
        # Eighths and a quarter: on the rows from two rows a beat up, and between them at one row a beat.
        plain = {**self.SONG, "patterns": {"a": {"tune": "o4 l8 c e g4 c2", "low": "o2 c2 g2", "beat": "l4 k t k t"}}}
        self.assertEqual((grids(plain), offered(plain)), ([1, 2, 3, 4, 6, 8, 12], [2, 4, 6, 8, 12]))
        # A triplet is on the rows only where a beat has a number of rows that three goes into. The grid a song
        # has is offered whatever its notes do.
        self.assertEqual(offered(self.SONG), [4, 12])

    def test_a_grid_at_which_a_row_would_last_no_whole_number_of_ticks_is_refused_and_so_is_one_that_is_none(
            self) -> None:
        cannot = "and a row lasts a whole number of them: the song cannot sound the same at that grid"
        self.assertEqual([refusal(self, gridded, self.SONG, rows) for rows in (5, 16, 48, 0, 2.0, True)],
                         [f"with 5 rows to a beat a row would last 4.8 ticks, {cannot}",
                          f"with 16 rows to a beat a row would last 1.5 ticks, {cannot}",
                          f"with 48 rows to a beat a row would last 0.5 ticks, {cannot}",
                          "the rows of a beat are a whole number, 1 or more, not 0",
                          "the rows of a beat are a whole number, 1 or more, not 2.0",
                          "the rows of a beat are a whole number, 1 or more, not True"])


class Layers(unittest.TestCase):
    def setUp(self) -> None:
        self.data = Project.load(EXAMPLES / "arcade.json").to_dict()
        self.tune = self.data["songs"]["tune"]

    def groups(self, made: dict) -> dict:
        """The groups of the tune's layers after a change, once the changed tune is seen to mix into them."""
        loaded = Project.from_dict({**self.data, "songs": {"tune": made}})
        self.assertEqual(list(render_groups(loaded.song("tune"), RATE)["groups"]), list(made["layers"]["groups"]))
        return made["layers"]["groups"]

    def test_a_renamed_channel_is_renamed_in_its_group(self) -> None:
        made = channel_renamed(self.tune, "bass", "deep")
        self.assertEqual(self.groups(made), {"rhythm": ["kit"], "low": ["deep"], "middle": ["arp"], "top": ["lead"]})
        self.assertEqual(list(made["channels"]), ["lead", "arp", "deep", "kit"])
        self.assertEqual(made["layers"]["versions"], self.tune["layers"]["versions"])

    def test_a_dropped_channel_leaves_the_group_it_shares_with_another(self) -> None:
        self.tune["layers"] = {"groups": {"rhythm": ["kit"], "low": ["bass", "arp"], "top": ["lead"]},
                               "versions": {"beat": ["rhythm"], "thin": ["rhythm", "low"],
                                            "full": ["rhythm", "low", "top"]}}
        made = channel_dropped(self.tune, "arp")
        self.assertEqual(self.groups(made), {"rhythm": ["kit"], "low": ["bass"], "top": ["lead"]})
        self.assertEqual(list(made["channels"]), ["lead", "bass", "kit"])
        self.assertEqual([list(texts) for texts in made["patterns"].values()], [["lead", "bass", "kit"]] * 2)

    def test_a_channel_that_is_all_a_group_has_is_not_dropped_and_the_refusal_names_the_group(self) -> None:
        for channel, group in (("kit", "rhythm"), ("lead", "top")):
            with self.subTest(channel=channel):
                self.assertEqual(refusal(self, channel_dropped, self.tune, channel),
                                 f"channel {channel!r} is all that group {group!r} of the layers has, and the "
                                 "group would be left empty")

    def test_an_added_channel_is_in_no_group_and_a_moved_one_stays_in_its_own(self) -> None:
        self.assertEqual(channel_added(self.tune, "pad", "pluck")["layers"], self.tune["layers"])
        self.assertEqual(self.groups(channel_moved(self.tune, "kit", 0)), self.tune["layers"]["groups"])


class Whole(unittest.TestCase):
    def test_a_song_is_moved_to_another_key_as_up_moves_it_and_its_drums_stay(self) -> None:
        given = song()
        made = changed(self, transposed, given, 2)
        self.assertEqual(made, up(given, 2))
        was, now = read(given, "a"), read(made, "a")
        for channel in ("tune", "low"):
            self.assertEqual(now[channel][0], [(row, rows, pitch + 2) for row, rows, pitch in was[channel][0]])
        self.assertEqual(now["beat"], was["beat"])
        self.assertEqual(read(changed(self, transposed, given, -12), "b")["low"][0][0], (0.0, 16.0, C2 + 5 - 12))

    def test_a_move_that_is_no_whole_number_or_takes_a_note_out_of_the_octaves_is_refused(self) -> None:
        self.assertEqual(refusal(self, transposed, song(), 1.5),
                         "the semitones a song is moved by are a whole number, not 1.5")
        # The highest note of the song is the c of octave 5, which is 59 under the highest there is.
        self.assertEqual(refusal(self, transposed, song(), 60),
                         "pattern 'b', channel 'tune': 60 semitones up takes a note out of octaves 0 to 9, and the "
                         "text can write no other")
        self.assertEqual(refusal(self, transposed, song(), -25),
                         "pattern 'a', channel 'low': -25 semitones up takes a note out of octaves 0 to 9, and the "
                         "text can write no other")
        self.assertEqual(read(changed(self, transposed, song(), 59), "b")["tune"][0][2], (8.0, 4.0, C5 + 59))
        # A drum hit stays where it is, so the note its letter plays is not held to the octaves.
        high = one({"tune": "o4 c1", "beat": "t1"}, drums={"t": {"inst": "thud", "note": "b9"}})
        self.assertEqual(transposed(high, 12)["patterns"]["x"], {"tune": "o5 c1", "beat": "t1"})

    def test_the_tempos_are_whole_ticks_to_a_row_from_300_beats_a_minute_down_to_40(self) -> None:
        found = tempos(4)
        self.assertEqual([row_ticks for row_ticks, _ in found], list(range(3, 23)))
        self.assertEqual(found[:4], [(3, 300.0), (4, 225.0), (5, 180.0), (6, 150.0)])
        self.assertTrue(all(beats == 900 / row_ticks for row_ticks, beats in found))
        # With other rows to a beat the ticks of a beat are other ones: 3600 over them is the tempo.
        self.assertEqual((tempos(8)[0], tempos(3)[0], tempos(3)[-1]), ((2, 225.0), (4, 300.0), (30, 40.0)))
        for beat in (0, -4, "4", True):
            with self.subTest(beat=beat):
                self.assertEqual(refusal(self, tempos, beat), f"a beat has more than 0 rows, not {beat!r}")

    def test_a_song_at_another_tempo_has_the_ticks_of_its_rows_changed_and_nothing_else(self) -> None:
        made = changed(self, with_tempo, song(), 9)
        self.assertEqual(made, song(row_ticks=9))
        # Rows of 9 ticks for rows of 6: the loop is half as long again.
        self.assertEqual(loop(made)["loop_samples"] * 2, loop(song())["loop_samples"] * 3)
        for row_ticks in (0, 2.5, True):
            with self.subTest(row_ticks=row_ticks):
                self.assertEqual(refusal(self, with_tempo, song(), row_ticks),
                                 f"the ticks of a row are a whole number, 1 or more, not {row_ticks!r}")


class Names(unittest.TestCase):
    def test_a_new_name_is_held_to_one_check_by_every_function_that_takes_one(self) -> None:
        given = song()
        takers = {"pattern": (lambda name: copied(given, 0, name), lambda name: added(given, 0, 1, name),
                              lambda name: renamed(given, "a", name)),
                  "channel": (lambda name: channel_added(given, name, "lead"),
                              lambda name: channel_renamed(given, "tune", name))}
        for kind, taken in (("pattern", "b"), ("channel", "low")):
            said = {"": f"a {kind} needs a name, and '' is none",
                    7: f"a {kind} needs a name, and 7 is none",
                    "a/b": f"a {kind} cannot be called 'a/b': a name cannot hold '/'",
                    "new\n": f"a {kind} cannot be called 'new\\n': a name cannot hold '\\n'",
                    "v1.2": f"a {kind} cannot be called 'v1.2': a name cannot hold '.'",
                    "take[2]": f"a {kind} cannot be called 'take[2]': a name cannot hold '['",
                    taken: f"the song has a {kind} called {taken!r} already"}
            for name, words in said.items():
                with self.subTest(kind=kind, name=name):
                    self.assertEqual(refusal(self, allowed, given, kind, name), words)
                    for take in takers[kind]:
                        self.assertEqual(refusal(self, take, name), words)
            self.assertEqual(allowed(given, kind, "new one"), "new one")

    def test_a_name_that_is_its_own_already_is_taken(self) -> None:
        self.assertEqual(refusal(self, renamed, song(), "a", "a"), "the song has a pattern called 'a' already")
        self.assertEqual(refusal(self, channel_renamed, song(), "low", "low"),
                         "the song has a channel called 'low' already")


class Results(unittest.TestCase):
    def test_what_comes_back_shares_nothing_with_the_song_handed_in(self) -> None:
        given = song(intro={"order": ["a"]}, layers={"groups": {"all": ["tune", "low", "beat"]},
                                                     "versions": {"full": ["all"]}})
        before = json.dumps(given)
        for label, made in edits(given, "beat").items():
            with self.subTest(label=label):
                loop(made)
                scribble(made)
                self.assertEqual(json.dumps(given), before)


class Examples(unittest.TestCase):
    def test_every_change_to_an_example_song_passes_the_project_s_check_and_plays(self) -> None:
        for file, name, spare in SONGS:
            data = Project.load(EXAMPLES / file).to_dict()
            given = data["songs"][name]
            before = json.dumps(given)
            made = edits(given, spare)
            self.assertEqual(json.dumps(given), before)
            for label, new in made.items():
                with self.subTest(file=file, label=label):
                    loaded = Project.from_dict({**data, "songs": {**data["songs"], name: new}})
                    played = render_loop(loaded.song(name), RATE)
                    self.assertGreater(played["loop_samples"], 0)
                    self.assertTrue(np.isfinite(played["left"]).all() and np.isfinite(played["right"]).all())

    def test_a_change_that_leaves_the_notes_where_they_were_plays_sample_for_sample_the_same(self) -> None:
        for file, name, _ in SONGS:
            data = Project.load(EXAMPLES / file).to_dict()
            given = data["songs"][name]
            first, last = next(iter(given["channels"])), len(given["order"]) - 1
            there = moved(given, 0, last)
            self.assertNotEqual(there["order"], given["order"])
            quiet = {
                "moved there and back": moved(there, last, 0),
                "renamed": renamed(given, given["order"][0], "other"),
                "channel_renamed": channel_renamed(given, first, "other"),
                "copied and the copy dropped": dropped(copied(given, 0), 1),
                "channel_moved": channel_moved(given, first, len(given["channels"]) - 1),
            }
            self.assertEqual((quiet["moved there and back"], quiet["copied and the copy dropped"]), (given, given))
            base = render_loop(Project.from_dict(data).song(name), RATE)
            for label, new in quiet.items():
                with self.subTest(file=file, label=label):
                    loaded = Project.from_dict({**data, "songs": {**data["songs"], name: new}})
                    same(self, render_loop(loaded.song(name), RATE), base)


if __name__ == "__main__":
    unittest.main()
