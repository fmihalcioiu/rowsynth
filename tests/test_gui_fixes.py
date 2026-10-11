"""Tests for rowsynth.gui.fixes: the safe repair of a finding, held to the checks run again on the repaired project."""
from __future__ import annotations

import copy
import unittest

from rowsynth import checks
from rowsynth.checks import Finding
from rowsynth.edit import set_in
from rowsynth.gui.fixes import Fix, fix
from rowsynth.project import Project

#: The rate the projects are checked at: a low one, to be quick.
RATE = 8000
#: A project that keeps every rule: a song of two channels that rest at both ends of its loop, and a sound.
SONG = {"row_ticks": 6, "gain": 0.2, "channels": {"lead": {"inst": "tone"}, "bass": {"inst": "tone"}},
        "patterns": {"a": {"lead": "o4 l4 r c e r", "bass": "o2 l4 r c g r"}}, "order": ["a"]}
ZAP = {"voices": [{"inst": "tone", "notes": [[2, "c4", 10, 2], [12, "e4", 10]]},
                  {"inst": "tone", "notes": [[2, "g4", 20]]}]}
PROJECT = {"instruments": {"tone": {"wave": "tri"}}, "sounds": {"zap": {"def": ZAP}}, "songs": {"tune": SONG}}
#: A sound that the output stage bends, and one that it does not.
LOUD = {"voices": [{"inst": "tone", "notes": [[2, "c3", 60, 9]]}]}
QUIET = {"voices": [{"inst": "tone", "notes": [[2, "c3", 4]]}]}
#: Layers for the song, with a version that names a group twice.
LAYERS = {"groups": {"top": ["lead"], "low": ["bass"]}, "versions": {"full": ["top", "low", "top"], "thin": ["low"]}}


def faulty(*changes: tuple[str, object]) -> dict:
    """The project with some places of it changed."""
    data = copy.deepcopy(PROJECT)
    for where, value in changes:
        set_in(data, where, copy.deepcopy(value))
    return data


def found(data: dict) -> list[Finding]:
    """What the checks find in a project, which has first to be one that loads."""
    return checks.run(Project.from_dict(data), RATE)


def mended(data: dict, *made: Fix) -> dict:
    """A project with repairs made in it."""
    data = copy.deepcopy(data)
    for one in made:
        for where, value in one.changes:
            set_in(data, where, value)
    return data


class Repairs(unittest.TestCase):
    def repaired(self, data: dict, rule: str) -> Fix:
        """The one repair that every finding of a faulty project has, once the project with it made keeps every rule.

        The project has findings of one rule and no other, and the repairs are asked for as those of one project.
        """
        before = found(data)
        self.assertEqual({one.rule for one in before}, {rule})
        shared: dict = {}
        made = [fix(data, one, RATE, shared) for one in before]
        self.assertIsNotNone(made[0])
        self.assertEqual(made, made[:1] * len(before))
        self.assertEqual(found(mended(data, made[0])), [])
        return made[0]

    def test_each_short_channel_of_a_pattern_is_filled_with_rests_to_the_rows_of_the_pattern(self) -> None:
        self.assertEqual(found(PROJECT), [])
        data = faulty(("songs.tune.channels.pad", {"inst": "tone"}),
                      ("songs.tune.patterns.a", {"lead": "o4 l4 r c e", "bass": "o2 l4 r c", "pad": "l1 r"}))
        before = found(data)
        self.assertEqual([(one.rule, one.target) for one in before], [("rows", "songs.tune.patterns.a")] * 2)
        # A finding says which channel only in its words, and each has the repair of its own channel.
        made = [fix(data, one, RATE) for one in before]
        self.assertEqual(made, [
            Fix("tune: lead of pattern a filled to 16 rows", (("songs.tune.patterns.a.lead", "o4 l4 r c e r4"),)),
            Fix("tune: bass of pattern a filled to 16 rows", (("songs.tune.patterns.a.bass", "o2 l4 r c r2"),))])
        self.assertEqual(found(mended(data, made[0])), before[1:])
        self.assertEqual(found(mended(data, *made)), [])

    def test_a_song_that_is_too_loud_has_its_gain_lowered_by_just_what_is_needed(self) -> None:
        data = faulty(("songs.tune.gain", 0.9))
        made = self.repaired(data, "level")
        (where, gain), = made.changes
        self.assertEqual((where, made.label), ("songs.tune.gain", f"tune: gain lowered to {gain}"))
        # The gain reads cleanly, and one hundredth more of it is too loud.
        self.assertEqual((gain, 0 < gain < 0.9), (round(gain, 2), True))
        self.assertEqual({one.rule for one in found(faulty(("songs.tune.gain", round(gain + 0.01, 2))))}, {"level"})

    def test_a_layered_song_has_its_gain_lowered_until_every_version_fits_in_every_key(self) -> None:
        layers = {**LAYERS, "versions": {"full": ["top", "low"], "thin": ["low"]}, "keys": [0, 3]}
        data = faulty(("songs.tune.gain", 0.9), ("songs.tune.layers", layers))
        self.assertEqual(len(found(data)), 4)
        (where, gain), = self.repaired(data, "level").changes
        self.assertEqual((where, 0 < gain < 0.9), ("songs.tune.gain", True))

    def test_a_sound_that_is_bent_has_every_note_set_at_one_share_of_its_level(self) -> None:
        made = self.repaired(faulty(("sounds.zap.def.voices[0].notes", [[2, "c4", 10, 9], [12, "e4", 10]])), "peak")
        share = made.changes[1][1][0][3]
        self.assertEqual(made, Fix(f"zap: every note at {share} of its level", (
            ("sounds.zap.def.voices[0].notes", [[2, "c4", 10, round(9 * share, 4)], [12, "e4", 10, share]]),
            ("sounds.zap.def.voices[1].notes", [[2, "g4", 20, share]]))))
        self.assertEqual((share, 0 < share < 1), (round(share, 2), True))

    def test_a_bent_sound_is_repaired_where_its_entry_has_it_and_a_held_note_once_for_its_start_and_its_loop(
            self) -> None:
        held = {"note": LOUD, "start": 8, "loop": 30, "join": 3, "end": QUIET, "let_go": 4}
        data = faulty(("sounds.hum", {"held": held}), ("sounds.zaps", {"defs": [ZAP, LOUD]}),
                      ("sounds.boom", {"parts": [[0, {"def": QUIET}], [0.1, {"def": LOUD}]]}))
        before = found(data)
        self.assertEqual([(one.rule, one.target) for one in before],
                         [("peak", f"sounds.{target}") for target in ("hum.start", "hum.loop", "zaps[1]",
                                                                      "boom.parts[1]")])
        made = [fix(data, one, RATE) for one in before]
        self.assertEqual(made[0], made[1])
        self.assertEqual([(one.label.split(":")[0], one.changes[0][0]) for one in made[1:]],
                         [("hum, held note", "sounds.hum.held.note.voices[0].notes"),
                          ("zaps, variant 2", "sounds.zaps.defs[1].voices[0].notes"),
                          ("boom, part 2", "sounds.boom.parts[1][1].def.voices[0].notes")])
        self.assertEqual(found(mended(data, *made)), [])

    def test_a_version_that_names_a_group_twice_names_each_group_once(self) -> None:
        made = self.repaired(faulty(("songs.tune.layers", LAYERS)), "layers")
        self.assertEqual(made, Fix("tune: version full names each group once",
                                   (("songs.tune.layers.versions.full", ["top", "low"]),)))

    def test_a_rule_whose_repair_would_be_a_choice_or_a_guess_has_none(self) -> None:
        # A limit that is broken, an offset, and a channel that is in no group: all else the "layers" rule finds.
        data = faulty(("sounds.zap.max_seconds", 0.1), ("sounds.zap.max_dc", 0),
                      ("songs.tune.layers", {"groups": {"top": ["lead"]}, "versions": {"full": ["top"]}}))
        before = found(data)
        self.assertEqual([one.rule for one in before], ["layers", "dc", "length"])
        self.assertEqual([fix(data, one, RATE) for one in before], [None] * 3)

    def test_a_finding_that_is_not_of_the_project_has_no_repair(self) -> None:
        # What a rule with a repair found in a project as it was before: a song that is gone, a pattern, a
        # level and a bend that were put right, a place that is no place.
        gone = [Finding("rows", "songs.gone.patterns.a", "channel 'bass' has 12 rows"),
                Finding("rows", "songs.tune.patterns.a", "channel 'bass' has 12 rows"),
                Finding("level", "songs.tune", "the left side of its mix peaks at 1.63 dBFS"),
                Finding("layers", "songs.tune", "version 'full' names a group twice"),
                Finding("peak", "sounds.zap", "reaches 0.970"), Finding("peak", "sounds.zap.loop", "reaches 0.970"),
                Finding("peak", "sounds..zap", "reaches 0.970")]
        self.assertEqual([fix(PROJECT, one, RATE) for one in gone], [None] * len(gone))


if __name__ == "__main__":
    unittest.main()
