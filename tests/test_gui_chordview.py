"""Tests for rowsynth.gui.chordview: chords read as the check reads them, wrong notes, a bar's chord, new chords."""
from __future__ import annotations

import copy
import unittest
from pathlib import Path

from rowsynth import checks
from rowsynth.checks import chords as rule
from rowsynth.edit import NOTHING, delete_in, set_in
from rowsynth.errors import RowSynthError
from rowsynth.gui import chordview, songview
from rowsynth.gui.chordview import KINDS, Bar, ChordBox, Span, box_parts, key_at, keys
from rowsynth.gui.ui import Event, Ui, inside
from rowsynth.project import Project

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

AM, F, C = ({"root": 9, "tones": [9, 0, 4]}, {"root": 5, "tones": [5, 9, 0]}, {"root": 0, "tones": [0, 4, 7]})
#: A song of three bars. Its second bar is shared by two chords, and its third has none written. One channel ends
#: before the pattern does, and one plays drums: the check looks at neither.
SONG = {
    "row_ticks": 6,
    "drums": {"k": {"inst": "thud", "note": "c2"}},
    "channels": {"top": {"inst": "tone"}, "low": {"inst": "tone"}, "short": {"inst": "tone"},
                 "kit": {"inst": "thud", "drums": True}},
    "patterns": {"a": {"top": "o4 l4 a b >c< d | f g a8 b8 >c< | c d e f", "low": "o2 l1 a | f | c",
                       "short": "o3 l1 b", "kit": "l4 k k k k | k k k k | k k k k"}},
    "order": ["a"],
    "chords": {"a": ["Am", "F C"]},
}
PROJECT = {
    "instruments": {"tone": {"wave": "tri"}, "thud": {"wave": "noise"}},
    "chords": {"Am": AM, "F": F, "C": C},
    "songs": {"tune": SONG},
}


def here(project: Project, name: str) -> list[str]:
    """What is found wrong here in a song of a project, each note in the words the check has for it."""
    song, said = project.song(name), []
    for called in song.get("chords", {}):
        pattern = songview.pattern(song, called)
        chords = chordview.read(project.chords, song, called, pattern.rows)
        for channel in pattern.channels:
            for found in map(lambda note: chordview.wrong(chords, note), channel.notes):
                if found is not None and chordview.looked(pattern, channel):
                    bar, chord, tone = found
                    said.append(f"channel {channel.name!r} starts a beat of bar {bar + 1} on {rule._NOTES[tone]}, "
                                f"which is not in the chord {chord}")
    return sorted(said)


def there(found: list[checks.Finding]) -> list[str]:
    """What the chord check says of the notes of a project: its findings, but for those that count the bars."""
    return sorted(finding.message for finding in found
                  if finding.rule == "chords" and "starts a beat" in finding.message)


def put(data: dict, where: str, value: object) -> Project:
    """A project with a value put at a place of its data, or with NOTHING what is there taken out, checked whole."""
    data = copy.deepcopy(data)
    if value is NOTHING:
        delete_in(data, where)
    else:
        set_in(data, where, value)
    return Project.from_dict(data)


class Read(unittest.TestCase):
    def test_the_bars_of_a_pattern_have_the_chords_the_check_reads_for_them(self) -> None:
        read = chordview.read(PROJECT["chords"], SONG, "a", 48.0)
        # One chord has its bar, two share theirs evenly in the order they are written, and the last bar has none.
        self.assertEqual(read.bars, (
            Bar("Am", (Span(0.0, 16.0, "Am", frozenset({9, 0, 4})),)),
            Bar("F C", (Span(16.0, 8.0, "F", frozenset({5, 9, 0})), Span(24.0, 8.0, "C", frozenset({0, 4, 7})))),
            Bar("", ())))
        self.assertEqual((read.written, read.per_beat, read.per_bar), (("Am", "F C"), 4, 16))
        # Every row is under the chord that the check's own rule has it under.
        for row in range(48):
            under = rule.under(PROJECT["chords"], SONG["chords"]["a"], row, 16, "chords")
            spans = [span.name for bar in read.bars for span in bar.spans if span.row <= row < span.row + span.rows]
            self.assertEqual(spans, [] if under is None else [under[1]])
        arcade = Project.load(EXAMPLES / "arcade.json")
        tune = arcade.songs["tune"]
        for pattern, names in tune["chords"].items():
            bars = chordview.read(arcade.chords, tune, pattern, 64.0).bars
            self.assertEqual([bar.text for bar in bars], names)
            self.assertEqual([[(span.rows, set(span.tones)) for span in bar.spans] for bar in bars],
                             [[(16.0, set(arcade.chords[name]["tones"]))] for name in names])
        # A song with no chords has its bars all the same, each with none. A string the check refuses is none too.
        bare = {key: value for key, value in tune.items() if key != "chords"}
        self.assertEqual(chordview.read(arcade.chords, bare, "a", 64.0).bars, (Bar("", ()),) * 4)
        cleared = chordview.read(arcade.chords, {"chords": {"a": ["", " G  C "]}}, "a", 32.0)
        self.assertEqual([(bar.text, len(bar.spans)) for bar in cleared.bars], [("", 0), ("G C", 2)])

    def test_a_note_is_wrong_here_where_the_check_finds_it_and_nowhere_else(self) -> None:
        project = Project.from_dict(PROJECT)
        found = here(project, "tune")
        # The channel that ends early starts on b under Am, and the third bar has a d and an f: none is found.
        self.assertEqual(found, [f"channel 'top' starts a beat of bar {bar} on {tone}, which is not in the chord "
                                 f"{chord}" for bar, tone, chord in ((1, "b", "Am"), (1, "d", "Am"), (2, "a", "C"),
                                                                    (2, "g", "F"))])
        self.assertEqual(found, there(checks.run(project)))
        # The arcade tune keeps the rule, and with one note of its lead moved off its chord it breaks it once.
        arcade = Project.load(EXAMPLES / "arcade.json")
        self.assertEqual((here(arcade, "tune"), there(rule.check(arcade, 44100))), ([], []))
        data = arcade.to_dict()
        lead = data["songs"]["tune"]["patterns"]["a"]["lead"]
        data["songs"]["tune"]["patterns"]["a"]["lead"] = lead.replace("| >c< a f a8 g8 |", "| >c< a g a8 g8 |")
        moved = Project.from_dict(data)
        self.assertEqual(here(moved, "tune"),
                         ["channel 'lead' starts a beat of bar 2 on g, which is not in the chord F"])
        self.assertEqual(here(moved, "tune"), there(rule.check(moved, 44100)))


class Changed(unittest.TestCase):
    def test_a_bar_that_is_set_or_cleared_gives_strings_the_project_takes(self) -> None:
        table, written = PROJECT["chords"], SONG["chords"]["a"]
        made = {(bar, name): chordview.with_chord(table, written, 3, bar, name)
                for bar, name in ((0, "C"), (2, "F"), (1, None), (0, None))}
        # A bar past the strings gets its string, the last string cleared is left out, and one before it is empty.
        self.assertEqual(made, {(0, "C"): ["C", "F C"], (2, "F"): ["Am", "F C", "F"], (1, None): ["Am"],
                                (0, None): ["", "F C"]})
        self.assertEqual((chordview.with_chord(table, [], 3, 2, "Am"), chordview.with_chord(table, ["Am"], 3, 0, None)),
                         (["", "", "Am"], []))
        self.assertEqual(written, ["Am", "F C"])
        for strings in (*made.values(), ["", "", "Am"], []):
            where, value = chordview.saving("tune", SONG, "a", strings)
            self.assertEqual((where, value), ("songs.tune.chords.a", strings or NOTHING))
            self.assertEqual(put(PROJECT, where, value).songs["tune"]["chords"], {"a": strings} if strings else {})
        # A song with no chords at all gets them whole, since a place cannot be put into what is not there.
        bare = {**PROJECT, "songs": {"tune": {key: value for key, value in SONG.items() if key != "chords"}}}
        where, value = chordview.saving("tune", bare["songs"]["tune"], "a", ["", "C"])
        self.assertEqual((where, put(bare, where, value).songs["tune"]["chords"]),
                         ("songs.tune.chords", {"a": ["", "C"]}))
        for bar, name, said in ((3, "C", "3 is no bar of the pattern, which has 3, counted from 0"),
                                (-1, "C", "-1 is no bar of the pattern, which has 3, counted from 0"),
                                (0, "G", "the project has no chord called 'G'")):
            with self.assertRaises(RowSynthError) as refused:
                chordview.with_chord(table, written, 3, bar, name)
            self.assertEqual(str(refused.exception), said)

    def test_a_chord_of_a_root_and_a_kind_is_one_the_project_takes_under_a_name_that_can_be_given(self) -> None:
        self.assertEqual([chordview.chord(root, kind) for root, kind in ((9, "minor"), (5, "major"), (11, "7"))],
                         [AM, F, {"root": 11, "tones": [11, 3, 6, 9]}])
        self.assertEqual([chordview.chord_name(root, kind) for root, kind in ((9, "minor"), (0, "major"), (6, "dim"))],
                         ["Am", "C", "F#dim"])
        # Every chord that can be made is taken, each under a name of its own.
        table = {chordview.chord_name(root, kind): chordview.chord(root, kind) for root in range(12) for kind in KINDS}
        self.assertEqual(len(Project.from_dict({"chords": table}).chords), 12 * len(KINDS))
        # A file with chords gets one more, and a file with none gets its table.
        self.assertEqual((chordview.adding(PROJECT["chords"], "Dm", table["Dm"]), chordview.adding(None, "Dm", AM)),
                         (("chords.Dm", {"root": 2, "tones": [2, 5, 9]}), ("chords", {"Dm": AM})))
        for refused, said in ((lambda: chordview.chord(12, "major"), "the root of a chord is a whole number from 0 "
                                                                    "for c to 11 for b, not 12"),
                              (lambda: chordview.chord(0, "big"), f"a chord is one of {', '.join(KINDS)}, not 'big'"),
                              (lambda: chordview.named({}, ""), "a chord needs a name, and '' is none"),
                              (lambda: chordview.named({}, "A m"), "a chord cannot be called 'A m': a name cannot "
                                                                   "hold ' '"),
                              (lambda: chordview.named({}, "A.m"), "a chord cannot be called 'A.m': a name cannot "
                                                                   "hold '.'"),
                              (lambda: chordview.adding(PROJECT["chords"], "Am", AM), "the project has a chord called "
                                                                                    "'Am' already")):
            with self.assertRaises(RowSynthError) as error:
                refused()
            self.assertEqual(str(error.exception), said)


class Box(unittest.TestCase):
    def setUp(self) -> None:
        self.ui = Ui()
        self.done: list[tuple[str, dict]] = []
        self.taken = True

    def opened(self) -> ChordBox:
        """Open the box at a place it has room at, with an owner that keeps what it is told and answers `taken`."""
        def done(name: str, chord: dict) -> bool:
            self.done.append((name, chord))
            return self.taken

        return self.ui.open(ChordBox((40, 30), done))

    def press(self, pos: tuple[int, int]) -> None:
        """Press the left button at a position and let it go there."""
        self.ui.handle(Event("down", pos, button=1))
        self.ui.handle(Event("up", pos, button=1))

    def test_the_parts_lie_in_the_box_which_stays_on_the_surface_and_a_black_key_lies_over_the_white_ones(self) -> None:
        parts = box_parts((40, 30))
        x, y, w, h = box = parts["box"]
        self.assertEqual((x, y), (40, 30))
        for part, (left, top, wide, high) in parts.items():
            self.assertTrue(x <= left and y <= top and left + wide <= x + w and top + high <= y + h, part)
        self.assertEqual((box_parts((950, 530), (960, 540))["box"], box_parts((-5, -5), (960, 540))["box"]),
                         ((960 - w, 540 - h, w, h), (0, 0, w, h)))
        board = left, top, _, high = parts["keys"]
        found = keys(board)
        # The seven white keys fill the keyboard side by side, each as high as it.
        self.assertEqual([found[tone] for tone in (0, 2, 4, 5, 7, 9, 11)],
                         [(left + 14 * place, top, 14, high) for place in range(7)])
        # A black key is pressed down to its foot, and under its foot the white key on either side of the line.
        for tone in (1, 3, 6, 8, 10):
            x, y, wide, tall = found[tone]
            self.assertTrue(inside(board, (x, y)) and tall < high)
            self.assertEqual([key_at(board, pos) for pos in ((x + 1, y + tall - 1), (x + 1, y + tall),
                                                             (x + wide - 1, y + tall))], [tone, tone - 1, tone + 1])
        self.assertEqual((key_at(board, (left - 1, top)), key_at(board, (left, top + high))), (None, None))

    def test_a_key_and_a_kind_that_are_pressed_make_the_chord_and_its_name_until_one_is_typed(self) -> None:
        box = self.opened()
        self.assertEqual((self.ui.modal, self.ui.focus, box.root, box.kind, box.name.text, box.chord),
                         (box, box, 0, "major", "C", C))
        found = keys(box_parts((40, 30))["keys"])
        self.press((found[9][0] + 2, found[9][1] + found[9][3] - 2))
        self.press((box.kinds.cells()[1][0] + 2, box.kinds.rect[1] + 2))
        self.assertEqual((box.root, box.kind, box.name.text, box.chord), (9, "minor", "Am", AM))
        # The name is all picked, so what is typed takes its place, less what no name holds. It stays through
        # another root.
        self.ui.handle(Event("text", text="lo w."))
        self.press((found[1][0] + 2, found[1][1] + 2))
        self.assertEqual((box.root, box.name.text, box.chord["tones"]), (1, "low", [1, 4, 8]))
        # An owner that answers false leaves the box open, and enter tells it again.
        self.taken = False
        self.ui.handle(Event("key", key="return"))
        self.assertEqual((self.ui.modal, self.done), (box, [("low", {"root": 1, "tones": [1, 4, 8]})]))
        self.taken = True
        self.press((box.ok.rect[0] + 2, box.ok.rect[1] + 2))
        self.assertEqual((self.ui.modal, len(self.done)), (None, 2))
        # Escape, CANCEL and a press off the box close it with nothing told.
        for close in (lambda: self.ui.handle(Event("key", key="escape")),
                      lambda: self.press((box.cancel.rect[0] + 2, box.cancel.rect[1] + 2)),
                      lambda: self.press((box.rect[0] - 3, box.rect[1] - 3))):
            box = self.opened()
            close()
            self.assertEqual((self.ui.modal, len(self.done)), (None, 2))


if __name__ == "__main__":
    unittest.main()
