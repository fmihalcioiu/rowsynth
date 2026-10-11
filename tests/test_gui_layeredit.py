"""Tests for rowsynth.gui.layeredit: the layers and the intro of a song changed, each result one the project accepts."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from rowsynth.errors import RowSynthError
from rowsynth.gui import layeredit
from rowsynth.layers import mix_version, render_groups, render_intro
from rowsynth.project import Project

ARCADE = json.loads((Path(__file__).resolve().parents[1] / "examples" / "arcade.json").read_text(encoding="utf-8"))
#: The layered tune of the example: four groups of a channel each, four versions, and the keys 0 and 2.
TUNE = ARCADE["songs"]["tune"]
#: The rate the tune is rendered at: a low one, to be quick.
RATE = 8000


def held(song: dict) -> Project:
    """The example with a song in the place of its tune, once the project's check has passed it."""
    return Project.from_dict({**ARCADE, "songs": {"tune": song}})


class Rules(unittest.TestCase):
    def setUp(self) -> None:
        self.before = copy.deepcopy(TUNE)
        # No rule changes the song it is handed.
        self.addCleanup(lambda: self.assertEqual(TUNE, self.before))

    def layers(self, song: dict) -> dict:
        """The layers of a song the project's check has passed."""
        held(song)
        return song["layers"]

    def test_layers_switched_on_are_one_group_of_every_channel_in_one_version_and_off_the_song_is_as_it_began(
            self) -> None:
        bare = {key: value for key, value in TUNE.items() if key != "layers"}
        made = layeredit.layered(bare)
        self.assertEqual(self.layers(made), {"groups": {"all": ["lead", "arp", "bass", "kit"]},
                                             "versions": {"full": ["all"]}})
        self.assertEqual((layeredit.unlayered(made), list(layeredit.unlayered(made))), (bare, list(bare)))
        held(layeredit.unlayered(TUNE))

    def test_a_group_is_added_renamed_and_removed_and_a_channel_is_in_one_group_or_in_none(self) -> None:
        self.assertEqual(list(self.layers(layeredit.group_added(TUNE))["groups"].items())[-1], ("group", []))
        self.assertEqual(list(self.layers(layeredit.group_added(layeredit.group_added(TUNE)))["groups"])[-2:],
                         ["group", "group2"])
        renamed = self.layers(layeredit.group_renamed(TUNE, "low", "deep"))
        self.assertEqual((list(renamed["groups"]), renamed["versions"]["most"]),
                         (["rhythm", "deep", "middle", "top"], ["rhythm", "deep", "middle"]))
        removed = self.layers(layeredit.group_removed(TUNE, "low"))
        self.assertEqual((list(removed["groups"]), removed["versions"]["thin"], removed["versions"]["full"]),
                         (["rhythm", "middle", "top"], ["rhythm"], ["rhythm", "middle", "top"]))
        # A channel that is put in a group leaves the one it was in, and one that is taken out is in none.
        put = layeredit.channel_put(TUNE, "bass", "top")
        self.assertEqual(self.layers(put)["groups"], {"rhythm": ["kit"], "low": [], "middle": ["arp"],
                                                      "top": ["lead", "bass"]})
        self.assertEqual(self.layers(layeredit.channel_taken(put, "bass"))["groups"]["top"], ["lead"])

    def test_a_version_is_added_renamed_and_removed_and_a_group_is_switched_into_it_and_out(self) -> None:
        mixed = {**TUNE, "layers": {**TUNE["layers"], "premix": [{"version": "thin", "key": 0},
                                                                 {"version": "full", "key": 2}]}}
        self.assertEqual(list(self.layers(layeredit.version_added(TUNE))["versions"].items())[-1], ("version", []))
        renamed = self.layers(layeredit.version_renamed(mixed, "thin", "lean"))
        self.assertEqual((list(renamed["versions"]), renamed["premix"][0]),
                         (["beat", "lean", "most", "full"], {"version": "lean", "key": 0}))
        removed = self.layers(layeredit.version_removed(mixed, "thin"))
        self.assertEqual((list(removed["versions"]), removed["premix"]),
                         (["beat", "most", "full"], [{"version": "full", "key": 2}]))
        # A group goes out of a version and in again where the order of the groups has it.
        out = layeredit.switched(TUNE, "low", "full")
        self.assertEqual(self.layers(out)["versions"]["full"], ["rhythm", "middle", "top"])
        self.assertEqual((layeredit.switched(out, "low", "full"), layeredit.switched(TUNE, "low", "full", True)),
                         (TUNE, TUNE))
        self.assertEqual(self.layers(layeredit.switched(TUNE, "top", "beat"))["versions"]["beat"], ["rhythm", "top"])

    def test_a_key_is_added_and_removed_with_what_was_mixed_ahead_in_it(self) -> None:
        mixed = {**TUNE, "layers": {**TUNE["layers"], "premix": [{"version": "thin", "key": 0},
                                                                 {"version": "full", "key": 2}]}}
        self.assertEqual(self.layers(layeredit.key_added(TUNE, -3))["keys"], [0, 2, -3])
        removed = self.layers(layeredit.key_removed(mixed, 0))
        self.assertEqual((removed["keys"], removed["premix"]), ([2], [{"version": "full", "key": 2}]))
        # The one key a song is written in is what the engine takes for a song that says none, and is not written.
        self.assertNotIn("keys", self.layers(layeredit.key_removed(TUNE, 2)))
        self.assertEqual(self.layers(layeredit.key_added(layeredit.key_removed(TUNE, 2), 5))["keys"], [0, 5])

    def test_a_version_is_mixed_ahead_in_a_key_and_no_longer_and_the_last_one_out_leaves_no_list_of_them(
            self) -> None:
        full, thin = {"version": "full", "key": 2}, {"version": "thin", "key": 0}
        one = layeredit.premixed(TUNE, "full", 2)
        two = layeredit.premixed(one, "thin", 0, True)
        # They stand in the order they were put in, after the keys, and the project's check passes them.
        self.assertEqual((self.layers(one)["premix"], self.layers(two)["premix"], list(two["layers"])[-1]),
                         ([full], [full, thin], "premix"))
        # With no word on which, the one that is in goes out. One that is as asked already changes nothing.
        self.assertEqual((layeredit.premixed(two, "full", 2)["layers"]["premix"],
                          layeredit.premixed(two, "thin", 0, True), layeredit.premixed(two, "thin", 2, False)),
                         ([thin], two, two))
        self.assertEqual((layeredit.premixed(one, "full", 2), layeredit.premixed(one, "full", 2, False),
                          layeredit.premixed(TUNE, "thin", 0, False)), (TUNE, TUNE, TUNE))
        # A song that names no keys is heard in the one it is written in, and is mixed ahead in that one.
        plain = layeredit.key_removed(TUNE, 2)
        self.assertEqual(self.layers(layeredit.premixed(plain, "beat", 0))["premix"], [{"version": "beat", "key": 0}])

    def test_an_intro_is_switched_on_with_the_engine_s_lengths_each_of_its_settings_is_set_and_it_is_switched_off(
            self) -> None:
        made = layeredit.intro_on(TUNE)
        held(made)
        self.assertEqual((made["intro"], layeredit.intro_off(made)), ({"order": ["a"]}, TUNE))
        for change, intro in ((layeredit.intro_order(made, ["b", "a"]), {"order": ["b", "a"]}),
                              (layeredit.intro_tail(made, 0.8), {"order": ["a"], "tail": 0.8}),
                              (layeredit.intro_fade(made, 0.1), {"order": ["a"], "fade": 0.1})):
            held(change)
            self.assertEqual(change["intro"], intro)
        # A length put back to what the engine takes for none is taken out of the file.
        self.assertEqual(layeredit.intro_tail(layeredit.intro_tail(made, 0.8), 1.5), made)
        # In a song whose bar is shorter than that tail, the tail is a bar, which is all the engine plays of one.
        self.assertEqual(layeredit.intro_on({**TUNE, "row_ticks": 3})["intro"], {"order": ["a"], "tail": 0.8})

    def test_what_cannot_be_made_is_refused_in_plain_words(self) -> None:
        alone = layeredit.layered({key: value for key, value in TUNE.items() if key != "layers"})
        made = layeredit.intro_on(TUNE)
        refused = (
            (lambda: layeredit.layered(TUNE), "the song has layers already"),
            (lambda: layeredit.group_added(alone, "all"), "the layers have a group called 'all' already"),
            (lambda: layeredit.group_renamed(TUNE, "low", "a.b"),
             "a group cannot be called 'a.b': a name cannot hold '.'"),
            (lambda: layeredit.group_removed(alone, "all"),
             "group 'all' is the only one: layers need a group, and are switched off to have none"),
            (lambda: layeredit.version_removed(alone, "full"),
             "version 'full' is the only one, and a layered song is heard as one of its versions"),
            (lambda: layeredit.switched(TUNE, "low", "calm"), "the layers have no version called 'calm'"),
            (lambda: layeredit.channel_put(TUNE, "pad", "low"), "the song has no channel called 'pad'"),
            (lambda: layeredit.channel_taken(layeredit.channel_taken(TUNE, "arp"), "arp"),
             "channel 'arp' is in no group, so there is none to take it out of"),
            (lambda: layeredit.key_added(TUNE, 2), "the song is heard in key 2 already"),
            (lambda: layeredit.key_added(TUNE, 1.5), "a key is a whole number of semitones, not 1.5"),
            (lambda: layeredit.key_added(TUNE, 80), "pattern 'a', channel 'lead': 80 semitones up takes a note out of "
                                                    "octaves 0 to 9, and the text can write no other"),
            (lambda: layeredit.key_removed(alone, 0),
             "key 0 is the only one, and a song is heard in one key at the least"),
            (lambda: layeredit.premixed(TUNE, "calm", 0), "the layers have no version called 'calm'"),
            (lambda: layeredit.premixed(TUNE, "full", 5), "the song is not heard in key 5"),
            (lambda: layeredit.premixed(TUNE, "full", True), "the song is not heard in key True"),
            (lambda: layeredit.intro_on(made), "the song has an intro already"),
            (lambda: layeredit.intro_off(TUNE), "the song has no intro"),
            (lambda: layeredit.intro_order(made, ["a", "c"]), "the song has no pattern called 'c'"),
            (lambda: layeredit.intro_order(made, []), "an intro plays one pattern at the least"),
            (lambda: layeredit.intro_tail(made, 2),
             "a tail of 2 seconds is longer than a bar of the song, which lasts 1.6"),
            (lambda: layeredit.intro_fade(made, 2),
             "a fade of 2 seconds is longer than the tail it ends, which is 1.5"),
        )
        for make, words in refused:
            with self.subTest(words=words):
                with self.assertRaises(RowSynthError) as caught:
                    make()
                self.assertEqual(str(caught.exception), words)

    def test_the_arcade_tune_put_through_every_rule_still_renders_its_groups_and_its_intro(self) -> None:
        song = layeredit.layered(layeredit.unlayered(TUNE))
        rendered = render_groups(held(song).song("tune"), RATE)
        self.assertEqual((list(rendered["groups"]), rendered["keys"]), (["all"], [0]))
        # The four groups again, made by hand, with the lead in none of them, and one version of the drums alone.
        for name, channel in (("rhythm", "kit"), ("low", "bass"), ("mid", "arp")):
            song = layeredit.channel_put(layeredit.group_added(song, name), channel, name)
            song = layeredit.switched(song, name, "full")
        song = layeredit.group_removed(layeredit.channel_taken(song, "lead"), "all")
        song = layeredit.group_renamed(song, "mid", "middle")
        song = layeredit.switched(layeredit.version_added(song), "rhythm", "version")
        song = layeredit.version_removed(layeredit.version_renamed(layeredit.version_added(song, "spare"),
                                                                   "version", "beat"), "spare")
        song = layeredit.key_removed(layeredit.key_added(layeredit.key_added(song, 2), -2), 2)
        song = layeredit.intro_fade(layeredit.intro_tail(layeredit.intro_order(layeredit.intro_on(song), ["b", "a"]),
                                                         0.8), 0.1)
        self.assertEqual((song["layers"], song["intro"]), (
            {"groups": {"rhythm": ["kit"], "low": ["bass"], "middle": ["arp"]},
             "versions": {"full": ["rhythm", "low", "middle"], "beat": ["rhythm"]}, "keys": [0, -2]},
            {"order": ["b", "a"], "tail": 0.8, "fade": 0.1}))
        ready = held(song).song("tune")
        rendered = render_groups(ready, RATE)
        self.assertEqual((list(rendered["groups"]), rendered["keys"], rendered["groups"]["rhythm"]["keyless"]),
                         (["rhythm", "low", "middle"], [0, -2], True))
        left, _ = mix_version(rendered, ready, "full", -2)
        self.assertEqual((len(left), bool(left.any())), (rendered["loop_samples"], True))
        intro = render_intro(ready, RATE)
        self.assertEqual(intro["in_samples"] - intro["walk_samples"], round(0.8 * RATE))


if __name__ == "__main__":
    unittest.main()
