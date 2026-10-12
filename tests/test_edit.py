"""Tests for one change to a project: to a place in its data, and to its file."""
from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

from rowsynth.edit import NOTHING, change_file, change_file_many, delete_in, get_in, set_in, steps
from rowsynth.errors import RowSynthError
from rowsynth.project import Project

ARCADE = Path(__file__).resolve().parent / "data" / "arcade.json"


def band() -> dict:
    """Data with all a place can lead through or to: dicts, lists in them and in lists, a number, a string and so on."""
    return {
        "instruments": {"reed": {"wave": "pulse", "vol": [15, 12, 9], "release": 4, "short": False}},
        "sounds": {
            "ping": {
                "def": {"voices": [{"inst": "reed", "notes": [[0, "a5", 3], [3, "e6", 6]]}]},
                "meta": {"by": None},
            },
        },
    }


class Places(unittest.TestCase):
    def test_a_place_is_read_into_a_string_for_each_name_and_a_number_for_each_item(self) -> None:
        for where, read in (
            ("songs", ["songs"]),
            ("songs.tune.patterns.a.lead", ["songs", "tune", "patterns", "a", "lead"]),
            ("sounds.zaps[1].notes", ["sounds", "zaps", 1, "notes"]),
            ("instruments.chime.vol[3]", ["instruments", "chime", "vol", 3]),
            ("songs.tune.order[2]", ["songs", "tune", "order", 2]),
            # An item of a list that is itself an item of a list.
            ("sounds.ping.def.voices[0].notes[12][1]", ["sounds", "ping", "def", "voices", 0, "notes", 12, 1]),
            # Without brackets a number is a name, and so is whatever else holds no dot and no bracket.
            ("chords.7.F#m 7", ["chords", "7", "F#m 7"]),
        ):
            with self.subTest(where=where):
                self.assertEqual(steps(where), read)

    def test_a_place_that_cannot_be_read_is_refused_with_the_column_the_reading_stops_at(self) -> None:
        name = "a name is expected here"
        number = "a whole number, 0 or more, is expected after '['"
        close = "']' is expected after the number"
        after = "a dot, '[' or the end of the place is expected here"
        for where, column, expected in (
            ("", 1, name),
            (".songs", 1, name),
            ("[0]", 1, name),
            ("songs.", 7, name),
            ("songs..tune", 7, name),
            ("songs.[0]", 7, name),
            ("vol[", 5, number),
            ("vol[]", 5, number),
            ("vol[x]", 5, number),
            ("vol[-1]", 5, number),
            ("vol[1", 6, close),
            ("vol[1.5]", 6, close),
            ("vol[12x]", 7, close),
            ("vol[1]x", 7, after),
            ("vol]", 4, after),
        ):
            with self.subTest(where=where):
                with self.assertRaises(RowSynthError) as refused:
                    steps(where)
                self.assertEqual(str(refused.exception), f"{where!r}, column {column}: {expected}")


class InPlace(unittest.TestCase):
    def refusal(self, call: Callable, where: str, *value: object) -> str:
        """What a call at a place in data is refused with. The data must be left as it was."""
        data = band()
        with self.assertRaises(RowSynthError) as refused:
            call(data, where, *value)
        self.assertEqual(data, band())
        return str(refused.exception)

    def test_get_in_gives_what_is_at_a_place(self) -> None:
        data = band()
        self.assertEqual(get_in(data, "instruments.reed.release"), 4)
        self.assertEqual(get_in(data, "instruments.reed.vol[2]"), 9)
        self.assertEqual(get_in(data, "sounds.ping.def.voices[0].notes[1][1]"), "e6")
        self.assertIsNone(get_in(data, "sounds.ping.meta.by"))
        self.assertIs(get_in(data, "sounds"), data["sounds"])
        self.assertEqual(data, band())

    def test_set_in_puts_a_value_where_one_is_and_gives_the_one_that_was_there(self) -> None:
        data = band()
        self.assertEqual(set_in(data, "instruments.reed.wave", "saw"), "pulse")
        self.assertEqual(set_in(data, "instruments.reed.vol[1]", 13), 12)
        self.assertEqual(set_in(data, "sounds.ping.def.voices[0].notes[0]", [0, "c5", 2]), [0, "a5", 3])
        # A null that was there is given as it is: it is a value, and not the marker for nothing.
        self.assertIsNone(set_in(data, "sounds.ping.meta.by", "hand"))
        # A key that is set stays where it was among the others.
        self.assertEqual(list(data["instruments"]["reed"].items()),
                         [("wave", "saw"), ("vol", [15, 13, 9]), ("release", 4), ("short", False)])
        self.assertEqual(data["sounds"]["ping"], {
            "def": {"voices": [{"inst": "reed", "notes": [[0, "c5", 2], [3, "e6", 6]]}]},
            "meta": {"by": "hand"},
        })

    def test_set_in_adds_a_key_at_the_end_of_its_dict_and_the_next_item_of_a_list_and_gives_nothing(self) -> None:
        data = band()
        self.assertIs(set_in(data, "instruments.reed.duty", 1), NOTHING)
        self.assertEqual(list(data["instruments"]["reed"]), ["wave", "vol", "release", "short", "duty"])
        self.assertEqual(data["instruments"]["reed"]["duty"], 1)
        self.assertIs(set_in(data, "instruments.reed.vol[3]", 6), NOTHING)
        self.assertEqual(data["instruments"]["reed"]["vol"], [15, 12, 9, 6])
        # A table the data does not have yet is a key like any other.
        self.assertIs(set_in(data, "songs", {}), NOTHING)
        self.assertEqual(list(data.items())[2:], [("songs", {})])

    def test_delete_in_takes_out_a_key_or_an_item_and_gives_it(self) -> None:
        data = band()
        self.assertEqual(delete_in(data, "instruments.reed.release"), 4)
        self.assertEqual(delete_in(data, "instruments.reed.vol[0]"), 15)
        self.assertEqual(delete_in(data, "sounds.ping.def.voices[0].notes[1]"), [3, "e6", 6])
        self.assertEqual(delete_in(data, "sounds.ping.meta"), {"by": None})
        self.assertEqual(data, {
            "instruments": {"reed": {"wave": "pulse", "vol": [12, 9], "short": False}},
            "sounds": {"ping": {"def": {"voices": [{"inst": "reed", "notes": [[0, "a5", 3]]}]}}},
        })

    def test_a_step_that_leads_nowhere_is_refused_naming_what_is_there_and_what_is_not_and_changes_nothing(
            self) -> None:
        for where, message in (
            # Through a key or an item that is missing.
            ("sonngs.tune", "the data has no key 'sonngs'"),
            ("instruments.flute.vol[0]", "instruments has no key 'flute'"),
            ("sounds.ping.def.voices[2].inst", "sounds.ping.def.voices has no item [2]: it holds 1, counted from 0"),
            # Into what holds nothing.
            ("instruments.reed.release.long", "instruments.reed.release is a number, and has no key 'long'"),
            ("instruments.reed.wave[0]", "instruments.reed.wave is a string, and has no item [0]"),
            ("instruments.reed.short.very", "instruments.reed.short is true or false, and has no key 'very'"),
            ("sounds.ping.meta.by.whom", "sounds.ping.meta.by is null, and has no key 'whom'"),
            # A key asked of a list, and an item asked of a dict.
            ("instruments.reed.vol.first", "instruments.reed.vol is a list, and has no key 'first'"),
            ("instruments[0]", "instruments is a dict, and has no item [0]"),
            # A place that cannot be read leads nowhere either.
            ("instruments..reed", "'instruments..reed', column 13: a name is expected here"),
        ):
            for call, value in ((get_in, ()), (set_in, (5,)), (delete_in, ())):
                with self.subTest(where=where, call=call.__name__):
                    self.assertEqual(self.refusal(call, where, *value), message)

    def test_an_item_past_the_next_is_not_set_and_what_is_not_there_is_neither_read_nor_deleted(self) -> None:
        self.assertEqual(self.refusal(set_in, "instruments.reed.vol[4]", 5),
                         "instruments.reed.vol has no item [4]: it holds 3, counted from 0")
        for call in (get_in, delete_in):
            with self.subTest(call=call.__name__):
                self.assertEqual(self.refusal(call, "instruments.reed.vol[3]"),
                                 "instruments.reed.vol has no item [3]: it holds 3, counted from 0")
                self.assertEqual(self.refusal(call, "instruments.reed.duty"), "instruments.reed has no key 'duty'")


class ChangeFile(unittest.TestCase):
    """Changes to a copy of an example project, in a folder of its own."""

    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "arcade.json"
        shutil.copyfile(ARCADE, self.path)
        self.before = self.held()

    def held(self) -> dict:
        """What the file holds now. Only its data is looked at, never how its text is laid out."""
        return json.loads(self.path.read_text(encoding="utf-8"))

    def assert_holds(self, expected: dict) -> None:
        """The file holds this data, every key of it in this order."""
        self.assertEqual(json.dumps(self.held()), json.dumps(expected))

    def test_the_text_of_one_channel_is_changed_and_everything_else_in_the_file_is_as_it_was(self) -> None:
        where, text = "songs.tune.patterns.a.lead", "o5 l4 e a >c< a | >c< a f a | g >c e c< | b >d< b g"
        told = change_file(self.path, where, text)
        was = "o5 l4 e a >c< a8 b8 | >c< a f a8 g8 | g >c e c8< b8 | b >d< b g"
        self.assertEqual(list(told.items()),
                         [("file", str(self.path)), ("where", where), ("was", was), ("now", text)])
        expected = copy.deepcopy(self.before)
        expected["songs"]["tune"]["patterns"]["a"]["lead"] = text
        self.assert_holds(expected)

    def test_a_new_sound_is_added_after_the_others_and_nothing_is_said_of_what_was_there(self) -> None:
        zing = {"def": {"voices": [{"inst": "zap", "notes": [[0, "g6", 6]]}]}}
        self.assertEqual(change_file(self.path, "sounds.zing", zing),
                         {"file": str(self.path), "where": "sounds.zing", "now": zing})
        expected = copy.deepcopy(self.before)
        expected["sounds"]["zing"] = zing
        self.assert_holds(expected)
        self.assertEqual(list(Project.load(self.path).sounds),
                         ["coin", "jump", "laser", "explosion", "blip", "powerup", "zing"])

    def test_one_item_of_a_step_table_is_changed(self) -> None:
        self.assertEqual(change_file(self.path, "instruments.chime.vol[3]", 11),
                         {"file": str(self.path), "where": "instruments.chime.vol[3]", "was": 12, "now": 11})
        expected = copy.deepcopy(self.before)
        expected["instruments"]["chime"]["vol"] = [15, 14, 13, 11, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        self.assert_holds(expected)

    def test_delete_takes_out_a_key_or_an_item_and_says_what_it_was_and_no_more(self) -> None:
        jump = {"def": {"voices": [{"inst": "hop", "notes": [[0, "c4", 9]]}]}}
        self.assertEqual(change_file(self.path, "sounds.jump", delete=True),
                         {"file": str(self.path), "where": "sounds.jump", "was": jump})
        self.assertEqual(change_file(self.path, "songs.tune.layers.keys[1]", delete=True),
                         {"file": str(self.path), "where": "songs.tune.layers.keys[1]", "was": 2})
        expected = copy.deepcopy(self.before)
        del expected["sounds"]["jump"]
        expected["songs"]["tune"]["layers"]["keys"] = [0]
        self.assert_holds(expected)

    def test_a_change_the_project_s_check_refuses_leaves_the_bytes_of_the_file_as_they_were(self) -> None:
        written = self.path.read_bytes()
        for where, given, message in (
            # An instrument and a wave that do not exist.
            ("sounds.coin.def.voices[0].inst", {"value": "bell"},
             "sounds.coin.def.voices[0].inst: no instrument named 'bell'"),
            ("instruments.chime.wave", {"value": "square"},
             "instruments.chime.wave: must be one of 'pulse', 'saw', 'tri', 'noise', 'fm', not 'square'"),
            # A delete of what something else in the project names.
            ("instruments.chime", {"delete": True}, "sounds.coin.def.voices[0].inst: no instrument named 'chime'"),
            # A place that is not in the file.
            ("songs.march.order[0]", {"value": "a"}, "songs has no key 'march'"),
        ):
            with self.subTest(where=where):
                with self.assertRaises(RowSynthError) as refused:
                    change_file(self.path, where, **given)
                self.assertEqual(str(refused.exception), message)
                self.assertEqual(self.path.read_bytes(), written)

    def test_a_channel_text_that_cannot_be_read_is_refused_and_the_bytes_of_the_file_are_as_they_were(self) -> None:
        written = self.path.read_bytes()
        for value, message in (
            ("o5 zz", "songs.tune: pattern 'a', channel 'lead': cannot read \"zz\" in a channel"),
            ("o9 >>>>> c", "songs.tune: pattern 'a', channel 'lead': the note at row 0 is 180, and a note is "
                           "from -97 to 136"),
        ):
            with self.subTest(value=value):
                with self.assertRaises(RowSynthError) as refused:
                    change_file(self.path, "songs.tune.patterns.a.lead", value)
                self.assertEqual(str(refused.exception), message)
                self.assertEqual(self.path.read_bytes(), written)

    def test_a_song_that_could_not_be_read_before_a_change_can_still_be_changed(self) -> None:
        # Two faults written by hand: the file loads, and neither channel can be read.
        data = self.held()
        data["songs"]["tune"]["patterns"]["a"].update(lead="o5 zz", bass="o2 yy")
        self.path.write_text(json.dumps(data), encoding="utf-8")
        change_file(self.path, "songs.tune.patterns.a.lead", "o5 c1")
        self.assertEqual(self.held()["songs"]["tune"]["patterns"]["a"]["lead"], "o5 c1")
        # A pattern the order leaves out is read as well.
        change_file(self.path, "songs.tune.patterns.a.bass", "o2 c1")
        change_file(self.path, "songs.tune.patterns.spare", {"lead": "o5 c1"})
        with self.assertRaises(RowSynthError) as refused:
            change_file(self.path, "songs.tune.patterns.spare.lead", "o5 zz")
        self.assertEqual(str(refused.exception),
                         "songs.tune: pattern 'spare', channel 'lead': cannot read \"zz\" in a channel")

    def test_a_value_that_is_what_the_file_holds_is_not_written_and_the_answer_says_so(self) -> None:
        # The file is given a time long past, so that any writing of it would show.
        os.utime(self.path, ns=(10 ** 18, 10 ** 18))
        stamped, written = os.stat(self.path), self.path.read_bytes()
        layers = copy.deepcopy(self.before["songs"]["tune"]["layers"])
        for where, value in (("songs.tune.gain", 0.35), ("songs.tune.layers", layers)):
            with self.subTest(where=where):
                self.assertEqual(change_file(self.path, where, value),
                                 {"file": str(self.path), "where": where, "was": value, "now": value, "same": True})
                now = os.stat(self.path)
                self.assertEqual((now.st_mtime_ns, now.st_size), (stamped.st_mtime_ns, stamped.st_size))
                self.assertEqual(self.path.read_bytes(), written)
        # A value that is not what is there is written, and the time of the file moves on.
        self.assertNotIn("same", change_file(self.path, "songs.tune.gain", 0.3))
        self.assertGreater(os.stat(self.path).st_mtime_ns, stamped.st_mtime_ns)

    def test_true_where_1_is_and_the_same_keys_in_another_order_are_changes_though_python_calls_them_equal(
            self) -> None:
        self.assertEqual(self.before["instruments"]["hop"]["duty"], 1)
        with self.assertRaises(RowSynthError) as refused:
            change_file(self.path, "instruments.hop.duty", True)
        self.assertEqual(str(refused.exception), "instruments.hop.duty: must be a whole number from 0 to 3")
        groups = self.before["songs"]["tune"]["layers"]["groups"]
        turned = dict(reversed(groups.items()))
        self.assertEqual(turned, groups)
        self.assertNotIn("same", change_file(self.path, "songs.tune.layers.groups", turned))
        self.assertEqual(list(self.held()["songs"]["tune"]["layers"]["groups"]), ["top", "middle", "low", "rhythm"])

    def test_a_change_is_a_value_or_a_delete_and_both_or_neither_is_refused_before_the_file_is_read(self) -> None:
        missing = self.path.with_name("nowhere.json")
        for given in ({}, {"value": 0.3, "delete": True}):
            with self.subTest(given=given):
                with self.assertRaises(RowSynthError) as refused:
                    change_file(missing, "songs.tune.gain", **given)
                self.assertEqual(str(refused.exception),
                                 "songs.tune.gain: a change is a value to put there or a delete, and only one of "
                                 "the two")

    def test_several_changes_are_made_as_one_where_each_alone_would_leave_the_project_wrong(self) -> None:
        # The instrument of the lead under another name: in its table, where it keeps its place, and in the song.
        table = {"top" if name == "lead" else name: inst for name, inst in self.before["instruments"].items()}
        song = copy.deepcopy(self.before["songs"]["tune"])
        song["channels"]["lead"]["inst"] = "top"
        for where, value in (("instruments", table), ("songs.tune", song)):
            with self.subTest(where=where):
                with self.assertRaises(RowSynthError):
                    change_file(self.path, where, value)
        self.assert_holds(self.before)
        told = change_file_many(self.path, [("instruments", table), ("songs.tune", song), ("sounds.jump", NOTHING)])
        file = str(self.path)
        self.assertEqual(told, [
            {"file": file, "where": "instruments", "was": self.before["instruments"], "now": table},
            {"file": file, "where": "songs.tune", "was": self.before["songs"]["tune"], "now": song},
            {"file": file, "where": "sounds.jump", "was": self.before["sounds"]["jump"]}])
        expected = copy.deepcopy(self.before)
        expected["instruments"], expected["songs"]["tune"] = table, song
        del expected["sounds"]["jump"]
        self.assert_holds(expected)

    def test_several_changes_of_which_one_is_refused_or_all_are_there_already_write_nothing(self) -> None:
        os.utime(self.path, ns=(10 ** 18, 10 ** 18))
        stamped, written = os.stat(self.path).st_mtime_ns, self.path.read_bytes()
        for changes, message in (
            # The first is refused by the project's check once both are made, and the second by where it leads.
            ([("instruments.lead", NOTHING), ("songs.tune.gain", 0.3)],
             "songs.tune.channels.lead.inst: no instrument named 'lead'"),
            ([("songs.tune.gain", 0.3), ("songs.march.gain", 0.3)], "songs has no key 'march'"),
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(RowSynthError) as refused:
                    change_file_many(self.path, changes)
                self.assertEqual(str(refused.exception), message)
        told = change_file_many(self.path, [("songs.tune.gain", 0.35), ("songs.tune.row_ticks", 6)])
        self.assertEqual([one.get("same") for one in told], [True, True])
        self.assertEqual((os.stat(self.path).st_mtime_ns, self.path.read_bytes()), (stamped, written))


if __name__ == "__main__":
    unittest.main()
