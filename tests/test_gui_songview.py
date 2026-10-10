"""Tests for rowsynth.gui.songview: a song read into the notes, channels, patterns and blocks the song view shows."""
from __future__ import annotations

import copy
import unittest
from pathlib import Path

from rowsynth import layers
from rowsynth.errors import RowSynthError
from rowsynth.gui.songview import Block, DrumRow, Note, bar_rows, blocks, drum_rows, on_grid, pattern, span
from rowsynth.project import Project
from rowsynth.song import lay

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

INSTRUMENTS = {"lead": {"vol": 12}, "bass": {"wave": "tri"}, "thud": {"wave": "noise"},
               "tick": {"wave": "noise", "short": True}}

# One letter with no level of its own, one at half, and one that is silent whatever its text says.
DRUMS = {"k": {"inst": "thud", "note": "c2"},
         "h": {"inst": "tick", "note": 90, "vel": 0.5},
         "s": {"inst": "thud", "note": "d4", "rows": 1, "vel": 0}}


def song(patterns: dict, order: list | None = None, **keys: object) -> dict:
    """A song of two tuned channels and a drum channel. Its order is its patterns, each once, unless one is given."""
    return {"row_ticks": 3, "instruments": INSTRUMENTS, "drums": DRUMS,
            "channels": {"tune": {"inst": "lead"}, "low": {"inst": "bass", "pan": -0.3},
                         "beat": {"inst": "thud", "drums": True}},
            "patterns": patterns, "order": list(patterns) if order is None else order, **keys}


def read(text: str, channel: str = "tune") -> tuple[Note, ...]:
    """The notes of one channel, in a song whose one pattern has that text for it and for no other."""
    found = pattern(song({"a": {channel: text}}), "a")
    return next(held.notes for held in found.channels if held.name == channel)


def at(*pitches: float) -> list[Note]:
    """A note at each of those pitches."""
    return [Note(0.0, 1.0, pitch, 15, "lead") for pitch in pitches]


def refusal(test: unittest.TestCase, call: object, *given: object) -> str:
    """The message a call is refused with."""
    with test.assertRaises(RowSynthError) as caught:
        call(*given)
    return str(caught.exception)


class Notes(unittest.TestCase):
    def test_a_tie_is_one_note(self) -> None:
        self.assertEqual(read("c4 & c4 d"), (Note(0, 8, 60, 15, "lead"), Note(8, 1, 62, 15, "lead")))

    def test_a_rest_leaves_a_gap(self) -> None:
        self.assertEqual(read("c r8 d"), (Note(0, 1, 60, 15, "lead"), Note(3, 1, 62, 15, "lead")))

    def test_a_dotted_note_is_half_as_long_again(self) -> None:
        self.assertEqual(read("c4. d"), (Note(0, 6, 60, 15, "lead"), Note(6, 1, 62, 15, "lead")))

    def test_v_is_the_level_and_the_octave_marks_move_the_pitch_of_the_notes_after_them(self) -> None:
        self.assertEqual([note.level for note in read("c v9 d e v0 f v15 g")], [15, 9, 9, 0, 15])
        self.assertEqual([note.pitch for note in read("c > c < < c o2 c+")], [60, 72, 48, 37])

    def test_an_instrument_switch_shows_on_the_notes_after_it(self) -> None:
        made = song({"a": {"tune": "c @bass d e"}, "b": {"tune": "c"}})
        found = pattern(made, "a").channels[0]
        self.assertEqual([note.inst for note in found.notes], ["lead", "bass", "bass"])
        self.assertEqual(found.inst, "lead")
        # The next pattern starts the channel on its own instrument again.
        self.assertEqual(pattern(made, "b").channels[0].notes, (Note(0, 1, 60, 15, "lead"),))


class Drums(unittest.TestCase):
    def test_a_hit_has_its_letter_what_the_letter_plays_and_the_level_it_was_written_with(self) -> None:
        self.assertEqual(read("k h8 k2 s2 c", "beat"),
                         (Note(0, 1, 36, 15, "thud", "k"), Note(1, 2, 90, 15, "tick", "h"),
                          Note(3, 8, 36, 15, "thud", "k"), Note(11, 1, 62, 15, "thud", "s"),
                          Note(19, 1, 60, 15, "thud", "")))
        # What a letter has for a level of its own, a half or nothing at all, is not in the level of its hits.
        self.assertEqual([(note.drum, note.level) for note in read("v6 k h s v11 h v15 s", "beat")],
                         [("k", 6), ("h", 6), ("s", 6), ("h", 11), ("s", 15)])

    def test_drum_rows_are_the_letters_in_the_order_of_the_file(self) -> None:
        self.assertEqual(drum_rows(song({})), [DrumRow("k", "thud", 36), DrumRow("h", "tick", 90),
                                               DrumRow("s", "thud", 62)])
        letters = {"t": {"inst": "tick", "note": "f#5"}, "H": {"inst": "thud", "note": 40.5}}
        self.assertEqual(drum_rows(song({}, drums=letters)), [DrumRow("t", "tick", 78), DrumRow("H", "thud", 40.5)])
        plain = song({})
        del plain["drums"]
        self.assertEqual(drum_rows(plain), [])


class Patterns(unittest.TestCase):
    def test_every_channel_of_the_song_is_there_and_one_with_no_text_is_not_written(self) -> None:
        found = pattern(song({"a": {"beat": "k4", "tune": ""}}), "a")
        self.assertEqual(found.name, "a")
        self.assertEqual([(held.name, held.inst, held.drums) for held in found.channels],
                         [("tune", "lead", False), ("low", "bass", False), ("beat", "thud", True)])
        tune, low, beat = found.channels
        self.assertEqual((low.written, low.notes, low.rows), (False, (), 0))
        # Text that is empty is text all the same.
        self.assertEqual((tune.written, tune.notes, tune.rows), (True, (), 0))
        self.assertEqual((beat.written, len(beat.notes), beat.rows), (True, 1, 4))

    def test_the_rows_of_a_pattern_are_those_of_its_longest_channel(self) -> None:
        found = pattern(song({"a": {"tune": "c4", "low": "c1", "beat": "k2 r1"}}), "a")
        self.assertEqual([held.rows for held in found.channels], [4, 16, 24])
        self.assertEqual(found.rows, 24)
        self.assertEqual(pattern(song({"a": {}}), "a").rows, 0)

    def test_the_order_is_a_block_for_each_place_in_it(self) -> None:
        made = song({"a": {"tune": "c1"}, "b": {"tune": "c2", "low": "c4"}}, ["a", "b", "a"])
        self.assertEqual(blocks(made), ([Block("a", 0, 16), Block("b", 16, 8), Block("a", 24, 16)], 40))
        self.assertEqual(blocks({**made, "order": []}), ([], 0))

    def test_a_song_is_left_as_it_was(self) -> None:
        made = song({"a": {"tune": "c @bass d", "beat": "v9 k h s"}}, ["a", "a"])
        kept = copy.deepcopy(made)
        pattern(made, "a")
        blocks(made)
        drum_rows(made)
        self.assertEqual(made, kept)

    def test_rows_per_beat_sets_the_lengths_and_a_bar_is_four_beats_as_the_chord_check_counts_it(self) -> None:
        self.assertEqual(bar_rows(song({})), 16)
        for rows_per_beat in (3, 8):
            made = song({"a": {"tune": "c4 d8"}}, rows_per_beat=rows_per_beat)
            self.assertEqual(bar_rows(made), 4 * rows_per_beat)
            self.assertEqual(bar_rows(made), layers.bar_rows(made))
            self.assertEqual([(note.row, note.rows) for note in pattern(made, "a").channels[0].notes],
                             [(0, rows_per_beat), (rows_per_beat, rows_per_beat / 2)])


class Grid(unittest.TestCase):
    def test_notes_that_start_and_end_on_whole_rows_are_on_the_grid(self) -> None:
        self.assertTrue(on_grid(pattern(song({"a": {"tune": "c4. r8 d16 & d2", "beat": "k8 h s4"}}), "a")))
        self.assertTrue(on_grid(pattern(song({"a": {}}), "a")))
        # Six triplet rests leave the note after them a hair short of a whole row, and it is on that row.
        found = pattern(song({"a": {"tune": "r6 r6 r6 r6 r6 r6 c4"}}), "a")
        self.assertNotEqual(found.channels[0].notes[0].row, 16)
        self.assertTrue(on_grid(found))

    def test_a_triplet_length_or_half_a_row_is_off_the_grid(self) -> None:
        for text in ("c12 d12 e12", "c4 r12 d6", "c32 r32", "r32 c32"):
            with self.subTest(text=text):
                self.assertFalse(on_grid(pattern(song({"a": {"low": "c1", "tune": text}}), "a")))


class Span(unittest.TestCase):
    def test_notes_are_shown_from_the_lowest_to_the_highest_and_in_the_middle_of_the_least(self) -> None:
        self.assertEqual(span(at(72, 48, 84, 60)), (48, 84))
        self.assertEqual(span(at(60, 72)), (60, 72))
        self.assertEqual(span(iter(at(40.5, 61.25)), least=4), (40, 62))
        self.assertEqual(span(at(60)), (54, 66))
        self.assertEqual(span(at(60, 61)), (55, 67))
        self.assertEqual(span(at(60), least=24), (48, 72))
        self.assertEqual(span(at(-13), least=0), (-13, -13))

    def test_no_notes_are_an_octave_round_middle_c(self) -> None:
        self.assertEqual(span([]), (54, 66))
        self.assertEqual(span([], least=5), (54, 66))
        self.assertEqual(span([], least=24), (48, 72))


class Refusals(unittest.TestCase):
    def test_a_pattern_the_song_does_not_have_is_refused(self) -> None:
        made = song({"a": {"tune": "c"}}, ["a", "z"])
        self.assertEqual(refusal(self, pattern, made, "z"), "no pattern called 'z'")
        self.assertEqual(refusal(self, blocks, made), "order: no pattern called 'z'")
        self.assertEqual(refusal(self, lay, made), "order: no pattern called 'z'")

    def test_what_lay_refuses_in_a_pattern_is_refused_in_the_same_words(self) -> None:
        cases = [({"low": "c", "tune": "@organ c"}, "pattern 'a', channel 'tune': no instrument called 'organ'"),
                 ({"low": "c", "tune": "c ? d"}, "pattern 'a', channel 'tune': cannot read \"? d\" in a channel"),
                 ({"tune": "c", "nobody": "c"}, "pattern 'a': no channel called 'nobody'")]
        for texts, said in cases:
            with self.subTest(texts=texts):
                made = song({"a": texts})
                self.assertEqual(refusal(self, pattern, made, "a"), said)
                self.assertEqual(refusal(self, blocks, made), said)
                self.assertEqual(refusal(self, lay, made), said)


class Examples(unittest.TestCase):
    def test_the_notes_of_the_blocks_are_the_events_the_engine_lays(self) -> None:
        for file in ("arcade.json", "piece.json"):
            project = Project.load(EXAMPLES / file)
            for name in project.songs:
                with self.subTest(file=file, song=name):
                    made = project.song(name)
                    laid, rows = lay(made)
                    found, total = blocks(made)
                    self.assertEqual(total, rows)
                    placed: dict[str, list[tuple]] = {}
                    for block in found:
                        for held in pattern(made, block.name).channels:
                            if held.written:
                                # The level is held to the engine's as well, with the letter's own put back.
                                placed.setdefault(held.name, []).extend(
                                    (block.row + note.row, note.rows, note.pitch, note.inst, note.drum,
                                     note.level / 15 * made.get("drums", {}).get(note.drum, {}).get("vel", 1))
                                    for note in held.notes)
                    self.assertEqual(placed, {channel: [(event["row"], event["rows"], event["note"], event["inst"],
                                                         event.get("drum", ""), event["vel"]) for event in events]
                                              for channel, events in laid.items()})
                    self.assertTrue(placed)

    def test_the_piece_has_its_channels_and_patterns_and_every_note_of_it_is_on_the_grid(self) -> None:
        made = Project.load(EXAMPLES / "piece.json").song("piece")
        found = {name: pattern(made, name) for name in made["patterns"]}
        self.assertEqual(len(found), 9)
        for name, one in found.items():
            with self.subTest(pattern=name):
                self.assertEqual(len(one.channels), 18)
                self.assertEqual(sum(held.drums for held in one.channels), 8)
                self.assertTrue(on_grid(one))
        notes = {name: sum(len(held.notes) for held in one.channels) for name, one in found.items()}
        self.assertEqual(sum(notes.values()), 1785)
        self.assertEqual(sum(notes[block.name] for block in blocks(made)[0]), 1785)


if __name__ == "__main__":
    unittest.main()
