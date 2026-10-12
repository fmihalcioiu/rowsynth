"""Tests for rowsynth.gui.projectedit: the things of a project added, renamed, copied and removed as changes."""
from __future__ import annotations

import copy
import json
import unittest
from collections.abc import Callable
from pathlib import Path

import numpy as np

from rowsynth.bank import render_entry
from rowsynth.edit import NOTHING, delete_in, set_in
from rowsynth.errors import RowSynthError
from rowsynth.gui.projectedit import (STARTERS, Change, added, allowed, free, fresh, own_added, own_copied, own_free,
                                      own_renamed, removed,
                                      renamed, shared, started, uses)
from rowsynth.project import Project
from rowsynth.song import render_loop

EXAMPLES = Path(__file__).resolve().parent / "data"

# A low rate, so that a thing is quick to render.
RATE = 2000


def voices(inst: str) -> dict:
    """A sound of one voice that plays one note of an instrument."""
    return {"voices": [{"inst": inst, "notes": [[0, "a4", 6]]}]}


# A project that names its instrument lead in every way there is: a channel, a drum, a switch in the middle of
# a text beside a switch to lead2, and a voice of a sound. The second song has an instrument of its own under
# the name, which its first channel plays, and a name of its own for the project's.
DATA = {
    "instruments": {"lead": {"wave": "pulse", "vol": 12}, "lead2": {"wave": "tri"}, "thud": {"wave": "noise"}},
    "sounds": {"zap": {"defs": [voices("lead"), voices("lead2")]}, "pop": {"def": voices("thud")}},
    "songs": {
        "tune": {"row_ticks": 6,
                 "drums": {"k": {"inst": "lead", "note": "c2"}, "s": {"inst": "thud", "note": "c4"}},
                 "channels": {"top": {"inst": "lead"}, "low": {"inst": "lead2"},
                              "kit": {"inst": "thud", "drums": True}},
                 "patterns": {"a": {"top": "o4 l4 c e g e", "low": "o3 l4 c @lead g  | @lead2 e @lead c",
                                    "kit": "l4 k s k s"}},
                 "order": ["a"]},
        "own": {"row_ticks": 6, "instruments": {"lead": {"wave": "saw"}, "soft": "lead"},
                "channels": {"x": {"inst": "lead"}, "y": {"inst": "soft"}, "z": {"inst": "thud"}},
                "patterns": {"a": {"x": "o4 c1", "y": "o3 e1", "z": "o2 c1"}}, "order": ["a"]},
    },
}


def with_buzz() -> dict:
    """DATA with two more instruments of the second song's own: buzz, which it plays, and spare, which it does not.

    One channel has buzz for its instrument, and the text of another switches to it.
    """
    data = copy.deepcopy(DATA)
    song = data["songs"]["own"]
    song["instruments"].update(buzz={"wave": "saw", "vol": 9}, spare={"wave": "tri"})
    song["channels"]["z"]["inst"] = "buzz"
    song["patterns"]["a"]["y"] = "o3 e2 @buzz e2"
    return data


def applied(data: dict, changes: list[Change]) -> dict:
    """A copy of the data with changes made to it in their order, as `change_file_many` makes them."""
    made = copy.deepcopy(data)
    for where, value in changes:
        if value is NOTHING:
            delete_in(made, where)
        else:
            set_in(made, where, copy.deepcopy(value))
    return made


def heard(data: dict, kind: str, name: str) -> np.ndarray:
    """A thing of a project as sound: the left side of a song, a sound, or one note of an instrument.

    The project is loaded first, so data that it refuses is refused here.
    """
    project = Project.from_dict(data)
    if kind == "song":
        return render_loop(project.song(name), RATE)["left"]
    if kind == "sound":
        return render_entry(project.sound(name), RATE)[0]
    return render_entry({"def": {"voices": [{"inst": project.instruments[name], "notes": [[0, "c4", 12]]}]}}, RATE)[0]


def refusal(call: Callable, *given: object, **named: object) -> str:
    """What a call is refused with."""
    try:
        call(*given, **named)
    except RowSynthError as error:
        return str(error)
    raise AssertionError(f"{call.__name__} refused nothing")


class Renaming(unittest.TestCase):
    def test_an_instrument_is_renamed_in_its_table_and_in_every_song_and_sound_that_names_it(self) -> None:
        began = copy.deepcopy(DATA)
        changes = renamed(DATA, "instrument", "lead", "top")
        self.assertEqual(DATA, began)
        self.assertEqual([where for where, _ in changes], ["instruments", "songs.tune", "songs.own", "sounds.zap"])
        made = applied(DATA, changes)
        Project.from_dict(made)
        # It keeps its place among the instruments.
        self.assertEqual(list(made["instruments"]), ["top", "lead2", "thud"])
        tune, own = made["songs"]["tune"], made["songs"]["own"]
        self.assertEqual([tune["channels"]["top"]["inst"], tune["drums"]["k"]["inst"]], ["top", "top"])
        # In the text only the name is another: the switch to lead2 and the two spaces are as they were.
        self.assertEqual(tune["patterns"]["a"], {"top": "o4 l4 c e g e", "low": "o3 l4 c @top g  | @lead2 e @top c",
                                                 "kit": "l4 k s k s"})
        self.assertEqual(made["sounds"]["zap"], {"defs": [voices("top"), voices("lead2")]})
        # The song's own instrument is none of the project's: only the name it gives the project's is another.
        self.assertEqual([own["instruments"], own["channels"]],
                         [{"lead": {"wave": "saw"}, "soft": "top"}, DATA["songs"]["own"]["channels"]])
        self.assertEqual(json.dumps(applied(made, renamed(made, "instrument", "top", "lead"))), json.dumps(DATA))
        # A song and a sound are named by their tables alone.
        (where, table), = renamed(DATA, "sound", "zap", "zing")
        self.assertEqual([where, list(table), table["zing"]], ["sounds", ["zing", "pop"], DATA["sounds"]["zap"]])

    def test_a_name_that_cannot_be_given_is_refused_and_the_refusal_says_why(self) -> None:
        unread = copy.deepcopy(DATA)
        unread["songs"]["tune"]["patterns"]["a"]["low"] = "o3 c ? @lead"
        for call, given, message in (
            (allowed, (DATA, "song", ""), "a song needs a name, and '' is none"),
            (allowed, (DATA, "instrument", "a.b"), "an instrument cannot be called 'a.b': a name cannot hold '.'"),
            (allowed, (DATA, "sound", "pop"), "the project has a sound called 'pop' already"),
            (allowed, (DATA, "chord", "C"), "a thing of a project is a song, a sound or an instrument, not 'chord'"),
            (renamed, (DATA, "song", "march", "walk"), "the project has no song called 'march'"),
            (renamed, (DATA, "instrument", "lead", "thud"), "the project has an instrument called 'thud' already"),
            (renamed, (DATA, "instrument", "lead", "my lead"),
             "the channel low in the pattern a of tune switches to the instrument as @lead, and a name written so "
             "holds only letters, digits and _, which 'my lead' does not"),
            (renamed, (DATA, "instrument", "thud", "soft"),
             "the song 'own' has an instrument of its own called 'soft', and would play that one in place of this"),
            (renamed, (unread, "instrument", "lead", "top"),
             'the channel low in the pattern a of tune: cannot read "? @lead" in a channel'),
            (added, (DATA, "song", "tune"), "the project has a song called 'tune' already"),
        ):
            with self.subTest(message=message):
                self.assertEqual(refusal(call, *given), message)

    def test_on_the_examples_every_instrument_renamed_and_renamed_back_gives_the_data_it_began_with(self) -> None:
        for file in ("arcade.json", "piece.json"):
            began = json.loads((EXAMPLES / file).read_text(encoding="utf-8"))
            for name in began["instruments"]:
                with self.subTest(file=file, name=name):
                    away = applied(began, renamed(began, "instrument", name, f"{name}_x"))
                    Project.from_dict(away)
                    self.assertNotIn(name, away["instruments"])
                    self.assertEqual(uses(away, "instrument", f"{name}_x"), uses(began, "instrument", name))
                    back = applied(away, renamed(away, "instrument", f"{name}_x", name))
                    self.assertEqual(json.dumps(back), json.dumps(began))


class Removing(unittest.TestCase):
    def test_an_instrument_that_is_still_used_is_not_removed_and_the_refusal_names_where(self) -> None:
        self.assertEqual(uses(DATA, "instrument", "lead"),
                         ["the channel top of tune", "the drum k of tune", "the channel low in the pattern a of tune",
                          "the instrument soft of own", "the sound zap"])
        self.assertEqual(refusal(removed, DATA, "instrument", "lead"),
                         "instrument 'lead' is still used by the channel top of tune, the drum k of tune, the "
                         "channel low in the pattern a of tune and 2 more")
        self.assertEqual(refusal(removed, DATA, "instrument", "lead2"),
                         "instrument 'lead2' is still used by the channel low of tune, the channel low in the "
                         "pattern a of tune, the sound zap")
        self.assertEqual(refusal(removed, DATA, "sound", "zing"), "the project has no sound called 'zing'")
        # Nothing names a song or a sound, so those go, and an instrument goes once nothing names it.
        self.assertEqual([uses(DATA, "song", "tune"), uses(DATA, "sound", "zap")], [[], []])
        gone = applied(DATA, [*removed(DATA, "song", "tune"), *removed(DATA, "sound", "zap")])
        self.assertEqual(removed(gone, "instrument", "lead2"), [("instruments.lead2", NOTHING)])
        left = Project.from_dict(applied(gone, removed(gone, "instrument", "lead2")))
        self.assertEqual([list(left.instruments), list(left.sounds), list(left.songs)],
                         [["lead", "thud"], ["pop"], ["own"]])


class Owned(unittest.TestCase):
    def test_an_own_instrument_made_shared_is_the_project_s_under_its_name_and_the_song_sounds_as_before(self) -> None:
        data = with_buzz()
        began = copy.deepcopy(data)
        changes = shared(data, "own", "buzz")
        self.assertEqual((data, changes), (began, [("instruments.buzz", {"wave": "saw", "vol": 9}),
                                                   ("songs.own.instruments.buzz", NOTHING)]))
        made = applied(data, changes)
        self.assertEqual([list(made["instruments"]), list(made["songs"]["own"]["instruments"])],
                         [["lead", "lead2", "thud", "buzz"], ["lead", "soft", "spare"]])
        np.testing.assert_array_equal(heard(made, "song", "own"), heard(data, "song", "own"))
        for given, message in (
            # The song's own lead is not the project's lead, which is there already.
            ((data, "own", "lead"), "the project has an instrument called 'lead' already"),
            ((data, "own", "soft"), "the song 'own' has no instrument of its own called 'soft'"),
            ((data, "tune", "lead"), "the song 'tune' has no instrument of its own called 'lead'"),
            ((data, "march", "buzz"), "the project has no song called 'march'"),
        ):
            with self.subTest(message=message):
                self.assertEqual(refusal(shared, *given), message)

    def test_the_only_thing_in_a_song_s_table_made_shared_takes_the_table_with_it(self) -> None:
        data = copy.deepcopy(DATA)
        song = data["songs"]["own"]
        song["instruments"] = {"buzz": {"wave": "saw", "vol": 9}}
        song["channels"]["x"]["inst"] = song["channels"]["y"]["inst"] = "buzz"
        changes = shared(data, "own", "buzz")
        self.assertEqual(changes, [("instruments.buzz", {"wave": "saw", "vol": 9}), ("songs.own.instruments", NOTHING)])
        made = applied(data, changes)
        # No empty table is left in the song, which plays the instrument as the project's from then on.
        self.assertEqual(("instruments" in made["songs"]["own"], list(made["instruments"])),
                         (False, ["lead", "lead2", "thud", "buzz"]))
        np.testing.assert_array_equal(heard(made, "song", "own"), heard(data, "song", "own"))

    def test_an_own_instrument_is_taken_out_of_its_song_unless_the_song_still_plays_it(self) -> None:
        data = with_buzz()
        # The song plays its own lead by the name the project has for another: only its own uses are found.
        self.assertEqual([uses(data, "instrument", "lead", "own"), uses(data, "instrument", "spare", "own")],
                         [["the channel x of own"], []])
        self.assertEqual(refusal(removed, data, "instrument", "buzz", "own"),
                         "instrument 'buzz' is still used by the channel z of own, the channel y in the pattern a of "
                         "own")
        self.assertEqual(refusal(removed, data, "instrument", "soft", "own"),
                         "the song 'own' has no instrument of its own called 'soft'")
        changes = removed(data, "instrument", "spare", "own")
        self.assertEqual(changes, [("songs.own.instruments.spare", NOTHING)])
        left = Project.from_dict(applied(data, changes))
        self.assertEqual([list(left.songs["own"]["instruments"]), list(left.instruments)],
                         [["lead", "soft", "buzz"], ["lead", "lead2", "thud"]])

    def test_an_instrument_of_the_project_is_copied_into_a_song_as_its_own_under_the_name_it_has(self) -> None:
        began = copy.deepcopy(DATA)
        # A song with no instruments of its own gets the table with it, and one that has some gets it at its place.
        changes = own_copied(DATA, "tune", "lead")
        self.assertEqual((DATA, changes), (began, [("songs.tune.instruments", {"lead": DATA["instruments"]["lead"]})]))
        self.assertIsNot(changes[0][1]["lead"], DATA["instruments"]["lead"])
        self.assertEqual(own_copied(DATA, "own", "thud"), [("songs.own.instruments.thud", {"wave": "noise"})])
        # The song plays its own copy from then on, which sounds as the project's did, and nothing else is changed.
        made = applied(DATA, changes)
        self.assertEqual({key: made[key] for key in ("instruments", "sounds")},
                         {key: DATA[key] for key in ("instruments", "sounds")})
        np.testing.assert_array_equal(heard(made, "song", "tune"), heard(DATA, "song", "tune"))
        named = copy.deepcopy(DATA)
        named["instruments"]["soft"] = {"wave": "tri"}
        for given, message in (
            ((DATA, "own", "lead"), "the song 'own' has an instrument called 'lead' already"),
            # A name the song gives to an instrument of the project is taken as well.
            ((named, "own", "soft"), "the song 'own' has an instrument called 'soft' already"),
            ((DATA, "own", "soft"), "the project has no instrument called 'soft'"),
            ((DATA, "march", "lead"), "the project has no song called 'march'"),
        ):
            with self.subTest(message=message):
                self.assertEqual(refusal(own_copied, *given), message)

    def test_a_song_gets_a_new_instrument_of_its_own_under_a_name_neither_it_nor_the_project_has(self) -> None:
        began = copy.deepcopy(DATA)
        # A song with no instruments of its own gets the table with it, and one that has some gets it at its place.
        self.assertEqual(own_added(DATA, "tune"), [("songs.tune.instruments", {"instrument": {"wave": "pulse"}})])
        self.assertEqual(own_added(DATA, "own"), [("songs.own.instruments.instrument", {"wave": "pulse"})])
        self.assertEqual(DATA, began)
        # The name is one that neither the song nor the project has, so the song sounds as it did.
        named = copy.deepcopy(DATA)
        named["instruments"]["instrument"] = {"wave": "tri"}
        named["songs"]["own"]["instruments"]["instrument2"] = {"wave": "saw"}
        self.assertEqual((own_free(DATA, "own"), own_free(named, "tune"), own_free(named, "own")),
                         ("instrument", "instrument2", "instrument3"))
        made = applied(named, own_added(named, "own"))
        self.assertEqual(list(made["songs"]["own"]["instruments"]), ["lead", "soft", "instrument2", "instrument3"])
        np.testing.assert_array_equal(heard(made, "song", "own"), heard(named, "song", "own"))
        # What is made holds: the project is one that loads.
        Project.from_dict(made)
        for given, message in (
            ((DATA, "own", "lead"), "the song 'own' has an instrument called 'lead' already"),
            ((DATA, "own", "a.b"), "an instrument cannot be called 'a.b': a name cannot hold '.'"),
            ((DATA, "march"), "the project has no song called 'march'"),
        ):
            with self.subTest(message=message):
                self.assertEqual(refusal(own_added, *given), message)
        self.assertEqual(refusal(own_free, DATA, "march"), "the project has no song called 'march'")

    def test_an_own_instrument_is_renamed_in_its_song_where_the_song_names_it_and_nowhere_else(self) -> None:
        data = with_buzz()
        began = copy.deepcopy(data)
        changes = own_renamed(data, "own", "buzz", "hum")
        self.assertEqual((data, [where for where, _ in changes]), (began, ["songs.own"]))
        made = applied(data, changes)
        song = made["songs"]["own"]
        # It keeps its place in the table, and the channel and the text that named it name it anew.
        self.assertEqual([list(song["instruments"]), song["channels"]["z"]["inst"], song["patterns"]["a"]],
                         [["lead", "soft", "hum", "spare"], "hum", {"x": "o4 c1", "y": "o3 e2 @hum e2", "z": "o2 c1"}])
        self.assertEqual([made["instruments"], made["sounds"], made["songs"]["tune"]],
                         [data["instruments"], data["sounds"], data["songs"]["tune"]])
        np.testing.assert_array_equal(heard(made, "song", "own"), heard(data, "song", "own"))
        # The song's own lead is not the project's: the name the song gives to that one stays as it is.
        song = applied(data, own_renamed(data, "own", "lead", "main"))["songs"]["own"]
        self.assertEqual([list(song["instruments"]), song["instruments"]["soft"], song["channels"]["x"]["inst"]],
                         [["main", "soft", "buzz", "spare"], "lead", "main"])
        # A name the project has for an instrument the song does not play can be given: the song's own is over it.
        Project.from_dict(applied(data, own_renamed(data, "own", "spare", "lead2")))

    def test_a_name_an_own_instrument_cannot_be_given_is_refused_and_the_refusal_says_why(self) -> None:
        data = with_buzz()
        for given, message in (
            ((data, "own", "buzz", "spare"), "the song 'own' has an instrument called 'spare' already"),
            ((data, "own", "buzz", "soft"), "the song 'own' has an instrument called 'soft' already"),
            ((data, "own", "buzz", ""), "an instrument needs a name, and '' is none"),
            ((data, "own", "buzz", "a.b"), "an instrument cannot be called 'a.b': a name cannot hold '.'"),
            ((data, "own", "buzz", "my hum"),
             "the channel y in the pattern a of own switches to the instrument as @buzz, and a name written so holds "
             "only letters, digits and _, which 'my hum' does not"),
            # The song plays the project's thud by that name, and would play its own in its place.
            ((DATA, "own", "lead", "thud"), "the channel z of own plays the instrument 'thud' of the project, and "
                                            "would play this one in place of that"),
            ((data, "own", "soft", "calm"), "the song 'own' has no instrument of its own called 'soft'"),
            ((data, "tune", "lead", "calm"), "the song 'tune' has no instrument of its own called 'lead'"),
            ((data, "march", "lead", "calm"), "the project has no song called 'march'"),
        ):
            with self.subTest(message=message):
                self.assertEqual(refusal(own_renamed, *given), message)


class Adding(unittest.TestCase):
    def test_a_new_thing_of_each_kind_is_accepted_by_the_project_and_can_be_heard(self) -> None:
        for data in (DATA, {}):
            for kind in ("song", "sound", "instrument"):
                with self.subTest(kind=kind, empty=not data):
                    # The name of a new thing is its kind, where that is free.
                    made = applied(data, added(data, kind))
                    self.assertGreater(float(np.abs(heard(made, kind, kind)).max()), 0.01)
        # Where the project has no instrument, one comes along first. A table the data has not is put there whole.
        self.assertEqual([[where for where, _ in added(data, "song", "march")]
                          for data in ({}, {"instruments": {}}, DATA)],
                         [["instruments", "songs"], ["instruments.instrument", "songs"], ["songs.march"]])
        # The new sound plays the first instrument of the project.
        self.assertEqual(added(DATA, "sound")[0][1]["def"]["voices"][0]["inst"], "lead")
        # A free name counts on from one that is taken.
        self.assertEqual([free(DATA, "song"), free({"songs": {"song": {}, "song2": {}}}, "song"),
                          free(DATA, "instrument", "lead")], ["song", "song3", "lead3"])

    def test_a_copy_has_the_next_free_name_and_renders_the_same_as_its_source(self) -> None:
        for kind, name, new in (("song", "tune", "tune2"), ("sound", "zap", "zap2"), ("instrument", "lead", "lead3")):
            with self.subTest(kind=kind):
                made = applied(DATA, added(DATA, kind, like=name))
                # It is the last of its kind, and none of the things there were is another.
                self.assertEqual(list(made[f"{kind}s"]), [*DATA[f"{kind}s"], new])
                self.assertEqual({key: made[f"{kind}s"][key] for key in DATA[f"{kind}s"]}, DATA[f"{kind}s"])
                np.testing.assert_array_equal(heard(made, kind, new), heard(DATA, kind, name))
        self.assertEqual(refusal(added, DATA, "song", like="march"), "the project has no song called 'march'")

    def test_the_ready_instruments_are_those_of_the_two_examples_and_each_is_accepted_and_can_be_heard(self) -> None:
        arcade, piece = (json.loads((EXAMPLES / name).read_text(encoding="utf-8"))["instruments"]
                         for name in ("arcade.json", "piece.json"))
        ready = {**{name: arcade[name] for name in ("pluck", "bass", "kick", "snare", "hat", "chime")},
                 **{name: piece[name] for name in ("fmbrass", "fmpad", "sub")}}
        # They are in the order they are offered, each as its example holds it, key for key.
        self.assertEqual(json.dumps(STARTERS), json.dumps(ready))
        for name in STARTERS:
            with self.subTest(name=name):
                made = applied(DATA, started(DATA, name))
                self.assertEqual(list(made["instruments"]), [*DATA["instruments"], name])
                self.assertGreater(float(np.abs(heard(made, "instrument", name)).max()), 0.01)

    def test_a_ready_instrument_comes_under_its_own_name_or_the_next_free_one_and_shares_nothing(self) -> None:
        # A project with no table of instruments gets the table with it.
        first = started({}, "bass")
        self.assertEqual(first, [("instruments", {"bass": STARTERS["bass"]})])
        taken = applied({}, first)
        again = started(taken, "bass")
        self.assertEqual(again, [("instruments.bass2", STARTERS["bass"])])
        self.assertIsNot(again[0][1]["vol"], STARTERS["bass"]["vol"])
        self.assertEqual([fresh(taken, "instrument", "bass"), fresh(taken, "instrument", "kick"),
                          fresh(applied(taken, again), "instrument", "bass")], ["bass2", "kick", "bass3"])
        # A name that is given is held to what any name is held to.
        self.assertEqual(started(taken, "kick", "deep"), [("instruments.deep", STARTERS["kick"])])
        self.assertEqual([refusal(started, taken, "kick", "bass"), refusal(started, DATA, "organ")],
                         ["the project has an instrument called 'bass' already",
                          "there is no ready instrument called 'organ'"])


if __name__ == "__main__":
    unittest.main()
