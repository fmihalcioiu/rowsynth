"""Tests for the rules a song is held to."""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from rowsynth.checks import Finding, run, songs
from rowsynth.errors import RowSynthError
from rowsynth.project import Project
from rowsynth.song import parse

# With three ticks to a row, a row is 400 samples at 8000 Hz and a bar of sixteen rows is 6400.
RATE = 8000

INSTRUMENTS = {
    "pipe": {"vol": [13, 11, 9], "duty": 1, "release": 1},
    "hum": {"wave": "tri", "vol": 11},
    # Held for all of its rows and then cut dead. A note of it that runs to the end of the loop stops in mid wave.
    "drone": {"wave": "saw", "vol": 10, "gate": 1, "release": 0},
    "thud": {"wave": "noise", "vol": [15, 8, 3], "release": 0},
}

# One bar for each channel. Each opens on a rest and closes on one, so its stem starts and ends in silence.
LEAD = "l8 r e g > c < b a g r"
BASS = "o2 r8 c4 c4 g4 r8"
BEAT = "r8 k4 k4 k4 r8"
# A bar that closes on a low note of the drone, and one that closes on a hit of it.
HELD = "@drone r4 o1 c2."
RUNG = "r8 k4 k4 t4."
# Bars that are short of their sixteen rows: by two, and by a triplet's odd share.
BASS_SHORT = "o2 r8 c4 c4 g4"
BEAT_SHORT = "r8 k4 k4 k12"
# Bars of triplets for the lead: six to the bar, and three of those with six half as long after them. Each fills
# its sixteen rows, and added up as floats the first comes to a hair under them and the second to a hair over.
TRIPLETS = "l6 r e g > c < b r"
TRIPLETS_OVER = "l6 r e g l12 > c < b a g e r"

# The tuned channels as one group and the drums as another, heard as written and three semitones up.
LAYERS = {"groups": {"tune": ["lead", "bass"], "kit": ["beat"]},
          "versions": {"full": ["tune", "kit"], "bare": ["kit"]},
          "keys": [0, 3]}
# Every channel a group of its own that is heard alone, in the one key the song is written in.
ALONE = {"groups": {"top": ["lead"], "low": ["bass"], "kit": ["beat"]},
         "versions": {"high": ["top"], "deep": ["low"], "hits": ["kit"]}}


def song(gain: float = 0.3, layers: dict | None = None, **texts: str) -> dict:
    """A song of one pattern, a bar long, at that gain. `texts` gives channels other text than the bars above."""
    made = {
        "row_ticks": 3,
        "gain": gain,
        "drums": {"k": {"inst": "thud", "note": "c2"}, "t": {"inst": "drone", "note": "c1"}},
        "channels": {"lead": {"inst": "pipe", "pan": -0.4}, "bass": {"inst": "hum", "pan": 0.3},
                     "beat": {"inst": "thud", "drums": True}},
        "patterns": {"a": {"lead": LEAD, "bass": BASS, "beat": BEAT, **texts}},
        "order": ["a"],
    }
    if layers is not None:
        made["layers"] = layers
    return made


def layered(groups: dict, versions: dict, **texts: str) -> dict:
    """The song with those groups and versions, in the one key it is written in."""
    return song(layers={"groups": groups, "versions": versions}, **texts)


def project_of(**tunes: dict) -> Project:
    """A project with the instruments above and these songs, as loading takes it."""
    return Project.from_dict({"instruments": INSTRUMENTS, "songs": tunes})


def by_hand(tune: dict) -> Project:
    """A project with the instruments above and a song named tune, put in by hand past what loading refuses."""
    project = Project()
    project.instruments = INSTRUMENTS
    project.songs = {"tune": tune}
    return project


def found_in(tune: dict, rate: int = RATE) -> list[Finding]:
    """The findings on one song, named tune."""
    return songs.check(project_of(tune=tune), rate)


def on(rule: str, message: str, pattern: str | None = None) -> Finding:
    """A finding on the song named tune, or on one of its patterns."""
    return Finding(rule, "songs.tune" if pattern is None else f"songs.tune.patterns.{pattern}", message)


def short(channel: str, rows: str, pattern: str) -> Finding:
    """The "rows" finding on a channel with that many rows in a pattern of sixteen."""
    return on("rows", f"channel {channel!r} has {rows} rows, and the longest channel of the pattern has 16", pattern)


def clicks(what: str, step: str) -> Finding:
    """The "join" finding on a loop with that step where it comes round."""
    return on("join", f"{what} steps by {step} from its last sample round to its first, and 0.02 or more clicks")


def peaks(what: str, level: str) -> Finding:
    """The "level" finding on a mix whose loudest sample is at that level."""
    return on("level", f"{what} peaks at {level} dBFS, and -1 or over is too loud")


def full(what: str, count: int) -> Finding:
    """The "level" finding on a mix with that many samples at full scale."""
    return on("level", f"{what} is at full scale, 0.999 or over, for {count} of its samples")


def breaks(message: str) -> Finding:
    """A "layers" finding."""
    return on("layers", message)


def rules(found: list[Finding]) -> list[str]:
    """The rule of each finding."""
    return [finding.rule for finding in found]


def stem(*samples: float) -> dict:
    """What `render_stems` gives for a song of one channel, its stem written by hand and all of it on the left."""
    wave = np.array(samples, dtype=np.float32)
    return {"stems": [{"name": "lead", "pan": -1, "samples": wave}], "rows": 16.0, "loop_samples": len(wave),
            "seconds": len(wave) / RATE}


class RowsTest(unittest.TestCase):
    def test_rows_fires_on_each_channel_short_of_the_longest_of_its_pattern(self):
        tune = song()
        tune["patterns"]["b"] = {"lead": LEAD, "bass": BASS_SHORT, "beat": BEAT_SHORT}
        tune["order"] = ["a", "b"]
        self.assertEqual(found_in(tune), [short("bass", "14", "b"), short("beat", "11.333333333333334", "b")])

    def test_rows_reads_a_pattern_the_order_does_not_play(self):
        tune = song()
        tune["patterns"]["spare"] = {"lead": LEAD, "bass": BASS_SHORT}
        self.assertEqual(found_in(tune), [short("bass", "14", "spare")])

    def test_a_channel_of_triplets_is_as_long_as_the_channels_of_even_notes_beside_it(self):
        for lead, rows in ((TRIPLETS, 15.999999999999998), (TRIPLETS_OVER, 16.000000000000004)):
            self.assertEqual(parse(lead, "pipe")[1], rows)
            # With no "rows" finding the song is played, and it keeps the other rules too.
            self.assertEqual(found_in(song(lead=lead)), [])

    def test_rows_fires_on_a_channel_a_row_short_of_triplets_and_says_their_whole_count_whole(self):
        # The triplets are the longest channel of the pattern, and the bass ends a sixteenth before the bar does.
        for lead in (TRIPLETS, TRIPLETS_OVER):
            tune = song()
            tune["patterns"]["b"] = {"lead": lead, "bass": "o2 r8 c4 c4 g4 r16"}
            self.assertEqual(found_in(tune), [short("bass", "15", "b")])

    def test_a_song_with_a_rows_finding_is_not_played_and_its_layers_are_still_read(self):
        # Loud enough for the level and cut off at the join, were it played.
        self.assertEqual(found_in(song(0.9, beat=RUNG, bass=BASS_SHORT)), [short("bass", "14", "a")])
        tune = song(0.9, {**LAYERS, "groups": {"tune": ["lead"], "kit": ["beat"]}}, beat=RUNG, bass=BASS_SHORT)
        self.assertEqual(found_in(tune), [short("bass", "14", "a"), breaks("channel 'bass' is in no group")])


class PlayedTest(unittest.TestCase):
    def found_by_hand(self, *samples: float) -> list[Finding]:
        """The findings on a song whose one stem is those samples."""
        with mock.patch.object(songs, "render_stems", return_value=stem(*samples)):
            return found_in(song())

    def test_a_song_that_keeps_the_rules_gives_no_findings(self):
        for tune in (song(), song(layers=LAYERS), {**song(), "intro": {"order": ["a"]}}):
            self.assertEqual(found_in(tune), [])
        self.assertEqual(songs.check(Project(), RATE), [])

    def test_join_fires_on_the_stem_of_a_channel_cut_off_where_the_loop_ends(self):
        self.assertEqual(found_in(song(bass=HELD)), [clicks("the stem of channel 'bass'", "0.047")])

    def test_level_fires_on_the_side_of_the_mix_that_reaches_the_limit(self):
        # The lead is panned to the left, and the right side stays 2 dB under it.
        self.assertEqual(found_in(song(0.6)), [peaks("the left side of its mix", "-0.12")])

    def test_level_counts_the_samples_at_full_scale_on_each_side(self):
        self.assertEqual(found_in(song(0.9)), [
            peaks("the left side of its mix", "2.30"), full("the left side of its mix", 121),
            peaks("the right side of its mix", "0.65"), full("the right side of its mix", 8)])

    def test_the_song_is_played_at_the_rate_handed_in(self):
        self.assertEqual(found_in(song(0.6), 44100), [
            peaks("the left side of its mix", "0.04"), full("the left side of its mix", 3)])

    def test_a_step_is_set_against_the_limit_as_a_double(self):
        # A float32 holds 0.52 a little under it and 0.27 a little over it.
        self.assertEqual(self.found_by_hand(0.5, 0.52), [])
        self.assertEqual(self.found_by_hand(0.25, 0.27), [clicks("the stem of channel 'lead'", "0.020")])

    def test_a_sample_is_set_against_the_level_limits_as_a_double(self):
        # The float32 nearest to the level limit is under it, and the one nearest to 0.999 is over that.
        under = np.float32(10 ** (-1 / 20))
        self.assertEqual(self.found_by_hand(under, under), [])
        self.assertEqual(self.found_by_hand(under, np.nextafter(under, np.float32(1))),
                         [peaks("the left side of its mix", "-1.00")])
        top = np.float32(0.999)
        self.assertEqual(self.found_by_hand(top, np.nextafter(top, np.float32(0))),
                         [peaks("the left side of its mix", "-0.01"), full("the left side of its mix", 1)])
        self.assertEqual(self.found_by_hand(np.nextafter(top, np.float32(0))),
                         [peaks("the left side of its mix", "-0.01")])


class LayersTest(unittest.TestCase):
    def test_layers_fires_on_a_channel_in_no_group_and_on_one_that_a_group_has_already(self):
        versions = LAYERS["versions"]
        self.assertEqual(found_in(layered({"tune": ["lead"], "kit": ["beat"]}, versions)),
                         [breaks("channel 'bass' is in no group")])
        self.assertEqual(found_in(layered({"tune": ["lead", "bass"], "kit": ["beat", "bass"]}, versions)),
                         [breaks("channel 'bass' is in group 'tune' and again in group 'kit'")])
        self.assertEqual(found_in(layered({"tune": ["lead", "bass", "lead"], "kit": ["beat"]}, versions)),
                         [breaks("channel 'lead' is in group 'tune' and again in group 'tune'")])

    def test_layers_fires_on_a_version_that_names_a_group_twice(self):
        tune = layered(LAYERS["groups"], {"full": ["tune", "kit"], "bare": ["kit", "kit"]})
        self.assertEqual(found_in(tune), [breaks("version 'bare' names a group twice")])

    def test_layers_fires_on_a_group_with_no_channels_and_on_one_in_no_version(self):
        groups = {**LAYERS["groups"], "spare": []}
        self.assertEqual(found_in(layered(groups, {"full": ["tune", "kit"], "bare": ["kit"], "odd": ["spare"]})),
                         [breaks("group 'spare' has no channels")])
        self.assertEqual(found_in(layered(groups, LAYERS["versions"])),
                         [breaks("group 'spare' has no channels"), breaks("group 'spare' is in no version")])

    def test_layers_fires_on_a_group_in_exactly_the_versions_of_an_earlier_one(self):
        groups = {"top": ["lead"], "low": ["bass"], "kit": ["beat"]}
        same = "are in exactly the same versions, so they would be one group"
        self.assertEqual(found_in(layered(groups, {"full": ["top", "low", "kit"], "bare": ["kit"]})),
                         [breaks(f"groups 'top' and 'low' {same}")])
        self.assertEqual(found_in(layered(groups, {"full": ["kit", "low", "top"]})),
                         [breaks(f"groups 'top' and 'low' {same}"), breaks(f"groups 'low' and 'kit' {same}")])
        # Two groups that no version names are in the same versions too: in none.
        self.assertEqual(found_in(layered(groups, {"bare": ["kit"]})),
                         [breaks("group 'top' is in no version"), breaks("group 'low' is in no version"),
                          breaks(f"groups 'top' and 'low' {same}")])

    def test_a_song_with_a_layers_finding_has_its_stems_played_and_its_groups_left_unmixed(self):
        tune = layered({"tune": ["lead", "lead"], "kit": [], "low": []}, {"full": ["tune", "tune"]}, bass=HELD)
        self.assertEqual(found_in(tune), [
            clicks("the stem of channel 'bass'", "0.047"),
            breaks("channel 'lead' is in group 'tune' and again in group 'tune'"),
            breaks("group 'kit' has no channels"),
            breaks("group 'low' has no channels"),
            breaks("channel 'bass' is in no group"),
            breaks("channel 'beat' is in no group"),
            breaks("version 'full' names a group twice"),
            breaks("group 'kit' is in no version"),
            breaks("group 'low' is in no version"),
            breaks("groups 'kit' and 'low' are in exactly the same versions, so they would be one group")])

    def test_a_song_with_layers_is_not_held_to_the_level_of_all_its_channels_at_once(self):
        self.assertEqual(rules(found_in(song(0.6))), ["level"])
        self.assertEqual(found_in(song(0.6, ALONE)), [])

    def test_level_fires_on_a_version_over_the_limit_in_each_key_it_is_over_it_in(self):
        self.assertEqual(found_in(song(0.6, LAYERS)), [
            peaks("version 'full' in key 0", "-0.12"),
            peaks("version 'full' in key 3", "0.01"), full("version 'full' in key 3", 1)])

    def test_join_fires_on_a_group_at_the_shared_scale_and_a_group_of_drums_alone_has_no_key(self):
        self.assertEqual(found_in(song(layers=LAYERS, bass=HELD, beat=RUNG)), [
            clicks("the stem of channel 'bass'", "0.047"),
            clicks("the stem of channel 'beat'", "0.121"),
            clicks("group 'tune' in key 0, at the scale the groups share,", "0.093"),
            clicks("group 'tune' in key 3, at the scale the groups share,", "0.130"),
            clicks("group 'kit', at the scale the groups share,", "0.197")])


class CheckTest(unittest.TestCase):
    def refusal_of(self, project: Project) -> str:
        """What the check refuses a project with."""
        with self.assertRaises(RowSynthError) as caught:
            songs.check(project, RATE)
        return str(caught.exception)

    def test_findings_come_song_by_song_in_the_order_of_the_project(self):
        uneven = song(bass=BASS_SHORT)
        found = songs.check(project_of(zed=song(0.6, bass=HELD), ace=uneven, mid=song(0.6, LAYERS, bass=HELD)), RATE)
        self.assertEqual([(finding.target, finding.rule) for finding in found], [
            ("songs.zed", "join"), ("songs.zed", "level"),
            ("songs.ace.patterns.a", "rows"),
            ("songs.mid", "join"), ("songs.mid", "level"), ("songs.mid", "level"), ("songs.mid", "join"),
            ("songs.mid", "join")])

    def test_run_gives_the_findings_on_songs_among_its_own(self):
        project = project_of(tune=song(bass=HELD))
        own = songs.check(project, RATE)
        self.assertEqual(rules(own), ["join"])
        self.assertIn(own[0], run(project, RATE))

    def test_a_missing_channel_or_instrument_is_refused_as_the_renderer_refuses_it_with_the_song_in_front(self):
        self.assertEqual(self.refusal_of(by_hand(song(ghost="c4"))),
                         "songs.tune: pattern 'a': no channel called 'ghost'")
        tune = song()
        tune["channels"]["bass"] = {"inst": "nothing"}
        self.assertEqual(self.refusal_of(by_hand(tune)),
                         "songs.tune: pattern 'a', channel 'bass': no instrument called 'nothing'")

    def test_a_missing_name_in_the_layers_is_refused_as_the_renderer_refuses_it_with_the_song_in_front(self):
        tune = layered({"tune": ["lead", "bass", "ghost"], "kit": ["beat", "ghost"]}, LAYERS["versions"])
        self.assertEqual(self.refusal_of(by_hand(tune)), "songs.tune: group 'tune': no channel called 'ghost'")
        tune = layered(LAYERS["groups"], {"full": ["tune", "kit"], "bare": ["kit", "ghost"]})
        self.assertEqual(self.refusal_of(by_hand(tune)),
                         "songs.tune: version 'bare': group 'ghost' was not rendered")

    def test_a_loaded_song_that_cannot_be_read_or_played_is_refused_with_the_song_in_front(self):
        self.assertEqual(self.refusal_of(project_of(tune=song(lead="c4 ?"))),
                         "songs.tune: pattern 'a', channel 'lead': cannot read \"?\" in a channel")
        self.assertEqual(self.refusal_of(project_of(tune={**song(), "order": []})),
                         "songs.tune: the song has no rows, so there is nothing to loop")


if __name__ == "__main__":
    unittest.main()
