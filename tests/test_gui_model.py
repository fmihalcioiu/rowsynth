"""Tests for rowsynth.gui.model."""
from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import threading
import time
import traceback
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import numpy as np

from rowsynth import checks
from rowsynth.bank import render_entry
from rowsynth.edit import NOTHING
from rowsynth.errors import RowSynthError
from rowsynth.gui import model as gui_model
from rowsynth.gui.model import SAVE_WAITS, TEST_NOTE, TEST_TICKS, Model, inst_of, inst_place, own, things
from rowsynth.gui.model import SAVE_WAITS, TEST_NOTE, TEST_TICKS, Model, things
from rowsynth.layers import group_mix, render_groups
from rowsynth.measure import brightness, measure
from rowsynth.project import Project
from rowsynth.song import mix_stems, pan_gains, render_loop, render_stems, up

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

# With three ticks to a row, a row is 400 samples at 8000 Hz, and a song of one bar is 6400: room for five of the
# frames its brightness is read from.
RATE = 8000

# The time the first write of a test is stamped with, in nanoseconds. Every later write is stamped a second on.
START = 1_700_000_000 * 10**9


def sound(note: str, ticks: float = 4, inst: str = "beep") -> dict:
    """A def of one voice playing one note from the start."""
    return {"voices": [{"inst": inst, "notes": [[0, note, ticks]]}]}


def song(text: str) -> dict:
    """A song of one bar: a tune left of the middle, and one long note under it right of the middle."""
    return {"row_ticks": 3,
            "channels": {"tune": {"inst": "beep", "pan": -0.5}, "low": {"inst": "hum", "pan": 0.5}},
            "patterns": {"a": {"tune": text, "low": "o3 c1"}}, "order": ["a"]}


# The tables come in another order than the listing shows them in, and no table has its names in the order of
# the alphabet. The sounds are one of each form. The checks find three things in it.
PROJECT = {
    "instruments": {"hum": {"wave": "tri", "vol": 12}, "beep": {"vol": [15, 12, 9, 6]}},
    "sounds": {
        "ping": {"def": sound("a4")},
        "blip": {"defs": [sound("c5"), sound("e5"), sound("g5")]},
        "crash": {"parts": [[0, {"def": sound("c3")}], [0.05, {"def": sound("c4"), "db": -6}]]},
        "drone": {"held": {"note": sound("a3", 12, "hum"), "start": 2, "loop": 6, "join": 1,
                           "end": sound("a3", 1, "hum"), "let_go": 2}},
    },
    "songs": {"tune": song("l4 c e g > c"), "air": song("l2 e g")},
}

# A song the engine refuses to loop.
NO_ROWS = {"songs": {"still": {"row_ticks": 3, "channels": {}, "patterns": {}, "order": []}}}
NO_ROWS_SAID = "the song has no rows, so there is nothing to loop"

# One fault in a song, where "air" comes round, and one in a sound, the peak of the second variant of "zaps". The
# sound "zap" has no fault, and the name of the one that has begins with its name.
FAULTS = {
    "instruments": PROJECT["instruments"],
    "sounds": {"zap": {"def": sound("a4")},
               "zaps": {"defs": [sound("c5"), {"voices": [{"inst": {"gain": 8}, "notes": [[0, "a4", 4]]}]}]}},
    "songs": {"tune": song("l4 c e g > c"), "air": song("l2 e g")},
}

# A project that loads and that the checks refuse: its song has text they cannot read.
UNREAD = {"instruments": PROJECT["instruments"], "songs": {"odd": song("l4 c e x9")}}
UNREAD_SAID = "songs.odd: pattern 'a', channel 'tune': cannot read \"x9\" in a channel"

# The checks themselves, for a test that has put something else in their place.
RUN = checks.run

# PROJECT with one song more, which has an instrument of its own under a name the project has too, and a name of
# its own for an instrument of the project.
MINE = {**song("l4 c e"), "instruments": {"beep": {"wave": "saw", "vol": 9}, "low": "hum"}}
OWN = {**PROJECT, "songs": {**PROJECT["songs"], "mine": MINE}}

# A song of one bar to listen into: a channel in the middle, one all to the left, one all to the right at half
# its level, and a fourth that no pattern has text for, which so has no stem. With it a song of two channels
# and a sound.
TRIO = {"row_ticks": 3,
        "channels": {"mid": {"inst": "beep"}, "port": {"inst": "hum", "pan": -1},
                     "star": {"inst": "hum", "pan": 1, "gain": 0.5}, "spare": {"inst": "beep"}},
        "patterns": {"a": {"mid": "l4 c e g > c", "port": "o3 c2 r2", "star": "o5 r2 g2"}}, "order": ["a"]}
LISTEN = {"instruments": PROJECT["instruments"], "sounds": {"ping": PROJECT["sounds"]["ping"]},
          "songs": {"trio": TRIO, "tune": song("l4 c e g > c")}}

# A song of one bar with five channels that all play, each at a pan and a level of its own: enough of them for
# one or two to be the few that are not heard.
FIVE = {"row_ticks": 3,
        "channels": {"lead": {"inst": "beep", "pan": -0.3}, "second": {"inst": "beep", "pan": 0.4, "gain": 0.8},
                     "bass": {"inst": "hum"}, "high": {"inst": "hum", "pan": -1, "gain": 0.6},
                     "under": {"inst": "hum", "pan": 1, "gain": 0.7}},
        "patterns": {"a": {"lead": "l4 c e g > c", "second": "l4 e g > c e", "bass": "o3 c2 g2",
                           "high": "o5 l8 c e g e c e g e", "under": "o2 c1"}},
        "order": ["a"]}
QUINTET = {"instruments": PROJECT["instruments"], "songs": {"five": FIVE}}

# A layered song of one bar that is built in two keys: a group of two tuned channels, and a group of two drum
# channels, which a build mixes once for all its keys. With it a song with no layers and a sound.
KEYED = {"row_ticks": 3, "drums": {"k": {"inst": "beep", "note": "c3"}},
         "channels": {"tune": {"inst": "beep", "pan": -0.5}, "low": {"inst": "hum", "pan": 0.5},
                      "kit": {"inst": "beep", "drums": True}, "tap": {"inst": "beep", "drums": True, "pan": 0.3}},
         "patterns": {"a": {"tune": "l4 c e g > c", "low": "o3 c1", "kit": "l4 k k k k", "tap": "l8 r k r k r k r k"}},
         "order": ["a"],
         "layers": {"groups": {"notes": ["tune", "low"], "beat": ["kit", "tap"]},
                    "versions": {"full": ["notes", "beat"]}, "keys": [0, 3]}}
LAYERED = {"instruments": PROJECT["instruments"], "sounds": {"ping": PROJECT["sounds"]["ping"]},
           "songs": {"keyed": KEYED, "tune": song("l4 c e g > c")}}


def stems_of(data: dict, name: str = "trio", rate: int = RATE) -> dict[str, dict]:
    """The stems the engine makes of a song of a project, each under the name of its channel."""
    return {stem["name"]: stem for stem in render_stems(Project.from_dict(data).song(name), rate)["stems"]}


def near(stems: dict[str, dict]) -> float:
    """How far a mix that had stems taken out of the mix of all may be from the same stems added up.

    Every sum on either way is rounded to float32, once for each stem of the song at most, and once more as
    the stems are taken out. So it is a step of float32 at the loudest the stems can add up to, for each stem
    and one more.
    """
    loudest = sum(float(np.abs(stem["samples"]).max()) for stem in stems.values())
    return (len(stems) + 1) * float(np.spacing(np.float32(loudest)))


def found_in(data: dict) -> list[checks.Finding]:
    """What the checks find in a project."""
    return RUN(Project.from_dict(data), RATE)


def changed(**tables: dict) -> dict:
    """PROJECT with the names given for each table set to what they come with. A name that comes with None is gone."""
    data = copy.deepcopy(PROJECT)
    for table, entries in tables.items():
        for name, entry in entries.items():
            if entry is None:
                del data[table][name]
            else:
                data[table][name] = entry
    return data


class ModelCase(unittest.TestCase):
    """What the tests below share: a folder for one project file, and the file stamped with a time of the test's own."""

    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "project.json"
        self.stamp = START

    def write(self, data: dict | str, later: bool = True) -> None:
        """Writes the file: a project, or text as it is. It is stamped a second after the write before, if `later`."""
        text = data if isinstance(data, str) else json.dumps(data)
        self.path.write_text(text, encoding="utf-8", newline="\n")
        if later:
            self.stamp += 10**9
        os.utime(self.path, ns=(self.stamp, self.stamp))

    def opened(self, data: dict = PROJECT) -> Model:
        """A model of the file, holding `data`."""
        self.write(data)
        return Model(self.path, RATE)

    def assert_render(self, made: dict, sides: list[np.ndarray], loop: bool, within: float = 0.0) -> None:
        """Holds a render to its sides, sample for sample or no further off than `within`, and to what goes with it."""
        self.assertEqual([made["rate"], made["loop"], made["seconds"]], [RATE, loop, len(sides[0]) / RATE])
        self.assertEqual([side.dtype for side in made["sides"]], [np.float32] * len(sides))
        self.assertEqual(len(made["sides"]), len(sides))
        for side, wanted in zip(made["sides"], sides):
            if within:
                np.testing.assert_allclose(side, wanted, rtol=0, atol=within)
            else:
                np.testing.assert_array_equal(side, wanted)

    def assert_loop(self, made: dict, data: dict, name: str) -> None:
        """Holds a render to the loop the engine makes of a song of `data`."""
        loop = render_loop(Project.from_dict(data).song(name), RATE)
        self.assertEqual(made["seconds"], loop["seconds"])
        self.assert_render(made, [loop["left"], loop["right"]], True)

    def assert_mix(self, made: dict, stems: list[dict], within: float = 0.0) -> None:
        """Holds a render of a song to the mix of some stems alone, sample for sample or no further off than `within`.

        A mix that had stems taken out of the mix of all is held to `near` of the song's stems.
        """
        self.assert_render(made, list(mix_stems(stems, len(stems[0]["samples"]))), True, within)


class OpeningTest(ModelCase):
    def test_the_listing_is_always_songs_sounds_and_instruments_with_their_names_as_the_file_has_them(self):
        self.assertEqual(self.opened().listing(), [
            ("songs", [("tune", ""), ("air", "")]),
            ("sounds", [("ping", ""), ("blip", "3"), ("crash", "parts"), ("drone", "held")]),
            ("instruments", [("hum", ""), ("beep", "")]),
        ])
        self.assertEqual(self.opened({}).listing(), [("songs", []), ("sounds", []), ("instruments", [])])

    def test_the_first_song_is_picked_at_the_start_or_with_no_song_the_first_sound_or_nothing(self):
        no_songs = {table: PROJECT[table] for table in ("instruments", "sounds")}
        no_sounds = {"instruments": PROJECT["instruments"]}
        for data, picked in ((PROJECT, ("song", "tune")), (no_songs, ("sound", "ping")), (no_sounds, None)):
            with self.subTest(picked=picked):
                self.assertEqual(self.opened(data).picked, picked)

    def test_a_file_that_cannot_be_read_or_is_no_project_is_refused_as_loading_it_is(self):
        for text in (None, "{", '{"songs": 3}'):
            with self.subTest(text=text):
                if text is not None:
                    self.write(text)
                with self.assertRaises((OSError, RowSynthError)) as loading:
                    Project.load(self.path)
                with self.assertRaises(type(loading.exception)) as opening:
                    Model(self.path, RATE)
                self.assertEqual(str(opening.exception), str(loading.exception))


class PickingTest(ModelCase):
    def test_picking_makes_a_song_a_sound_or_an_instrument_the_picked_thing(self):
        model = self.opened()
        for picked in (("sound", "crash"), ("instrument", "beep"), ("song", "air")):
            model.pick(*picked)
            self.assertEqual(model.picked, picked)

    def test_picking_what_the_project_does_not_have_is_refused_and_the_picked_thing_stays(self):
        model = self.opened()
        for kind, name, said in (
            ("sound", "tune", "no sound named 'tune'"),
            ("songs", "tune", "a thing of a project is a song, a sound or an instrument, not 'songs'"),
        ):
            with self.subTest(kind=kind):
                with self.assertRaises(RowSynthError) as refused:
                    model.pick(kind, name)
                self.assertEqual(str(refused.exception), said)
        self.assertEqual(model.picked, ("song", "tune"))


class RenderTest(ModelCase):
    def test_a_song_is_rendered_as_its_loop_in_two_sides(self):
        made = self.opened().render()
        self.assert_loop(made, PROJECT, "tune")
        self.assertFalse(np.array_equal(*made["sides"]))

    def test_a_sound_is_rendered_as_the_first_buffer_of_its_entry(self):
        model = self.opened()
        model.pick("sound", "blip")
        first = render_entry(Project.from_dict(PROJECT).sound("blip"), RATE)[0]
        self.assert_render(model.render(), [first], False)

    def test_an_instrument_is_rendered_as_one_c4_of_twelve_ticks_through_the_bank(self):
        model = self.opened()
        model.pick("instrument", "hum")
        note = {"def": {"voices": [{"inst": PROJECT["instruments"]["hum"], "notes": [[0, "c4", 12]]}]}}
        self.assert_render(model.render(), render_entry(note, RATE), False)

    def test_the_render_of_a_song_is_sample_for_sample_the_loop_the_engine_makes_of_it(self):
        # The song of the first example at the rate a model has unless it is told, the long one at a low rate.
        for file, rate in (("arcade.json", 44100), ("piece.json", RATE)):
            with self.subTest(file=file):
                model = Model(EXAMPLES / file, rate)
                made = model.render()
                loop = render_loop(model.project.song(model.picked[1]), model.rate)
                self.assertEqual([model.picked[0], made["loop"], made["seconds"]], ["song", True, loop["seconds"]])
                for side, wanted in zip(made["sides"], (loop["left"], loop["right"]), strict=True):
                    self.assertEqual(side.dtype, np.float32)
                    np.testing.assert_array_equal(side, wanted)

    def test_a_render_is_made_once_for_as_long_as_its_thing_stays_picked(self):
        model = self.opened()
        with mock.patch.object(gui_model, "render_stems", wraps=render_stems) as engine:
            made = model.render()
            model.numbers()
            model.pick("song", "tune")
            self.assertIs(model.render(), made)
            self.assertEqual(engine.call_count, 1)
            model.pick("song", "air")
            self.assert_loop(model.render(), PROJECT, "air")
            self.assertEqual(engine.call_count, 2)

    def test_with_nothing_picked_there_is_nothing_to_render_or_to_measure(self):
        model = self.opened({})
        for ask in (model.render, model.numbers):
            with self.assertRaises(RowSynthError) as refused:
                ask()
            self.assertEqual(str(refused.exception), "nothing is picked")

    def test_a_render_the_engine_refuses_is_tried_once_and_refused_each_time_it_is_asked_for(self):
        model = self.opened(NO_ROWS)
        with mock.patch.object(gui_model, "render_stems", wraps=render_stems) as engine:
            for ask in (model.render, model.numbers, model.render):
                with self.assertRaises(RowSynthError) as refused:
                    ask()
                self.assertEqual(str(refused.exception), NO_ROWS_SAID)
        self.assertEqual(engine.call_count, 1)
        self.assertTrue(model.render_ready())

    def test_the_numbers_are_those_of_the_mean_of_the_sides(self):
        model = self.opened()
        for picked in (("song", "tune"), ("sound", "crash"), ("instrument", "beep")):
            with self.subTest(picked=picked):
                model.pick(*picked)
                row = np.mean(model.render()["sides"], axis=0, dtype=np.float64)
                numbers = model.numbers()
                self.assertEqual(numbers, {**measure(row, RATE), **brightness(row, rate=RATE)})
                self.assertEqual(list(numbers), ["seconds", "peak_db", "rms_db", "dc", "clipped", "head_step",
                                                 "tail_step", "above_4k", "centroid_hz"])
        # The song is long enough for its brightness to be read, so that number is not a zero for want of samples.
        model.pick("song", "tune")
        self.assertGreater(model.numbers()["centroid_hz"], 0)


    def test_the_buffers_of_a_sound_are_all_its_entry_renders_to_and_the_sides_are_the_first_of_them(self):
        model = self.opened()
        project = Project.from_dict(PROJECT)
        for name, count in (("ping", 1), ("blip", 3), ("crash", 1), ("drone", 3)):
            with self.subTest(sound=name):
                model.pick("sound", name)
                made, wanted = model.render(), render_entry(project.sound(name), RATE)
                self.assertEqual(len(made["buffers"]), count)
                for buffer, want in zip(made["buffers"], wanted):
                    np.testing.assert_array_equal(buffer, want)
                self.assertIs(made["sides"][0], made["buffers"][0])
                self.assert_render(made, wanted[:1], False)

    def test_the_buffers_of_a_song_and_of_an_instrument_are_the_one_row_of_the_mean_of_the_sides(self):
        model = self.opened()
        for picked in (("song", "tune"), ("instrument", "beep")):
            with self.subTest(picked=picked):
                model.pick(*picked)
                made = model.render()
                self.assertEqual(len(made["buffers"]), 1)
                np.testing.assert_array_equal(made["buffers"][0], np.mean(made["sides"], axis=0, dtype=np.float64))


class HearTest(ModelCase):
    def test_one_channel_heard_is_the_mix_of_its_stem_alone_two_are_their_sum_and_all_are_the_whole_render(self):
        model = self.opened(LISTEN)
        whole, stems = model.render(), stems_of(LISTEN)
        self.assertIsNone(model.heard)
        model.hear({"mid"})
        self.assertEqual(model.heard, {"mid"})
        self.assert_mix(model.render(), [stems["mid"]])
        # Two are their sum in the order the song has them, worked out in doubles and rounded to float32 after each.
        # They are two of three stems, so it is the third that was taken out, and the sum is met within rounding.
        model.hear(["star", "port"])
        wanted = [np.zeros(6400, dtype=np.float32), np.zeros(6400, dtype=np.float32)]
        for name in ("port", "star"):
            for side, gain in enumerate(pan_gains(stems[name]["pan"])):
                wanted[side] = (wanted[side] + stems[name]["samples"].astype(np.float64) * gain).astype(np.float32)
        self.assert_render(model.render(), wanted, True, near(stems))
        # Every channel by its name is the whole song sample for sample, and all of them is the render it was.
        model.hear(TRIO["channels"])
        self.assert_render(model.render(), whole["sides"], True)
        model.hear()
        self.assertIsNone(model.heard)
        self.assertIs(model.render(), whole)
        # A channel that plays nothing is silence, and so are no channels at all.
        for only in ({"spare"}, set()):
            model.hear(only)
            self.assert_render(model.render(), [np.zeros(6400, dtype=np.float32)] * 2, True)

    def test_a_channel_in_the_middle_one_on_the_left_and_one_on_the_right_keep_their_sides_when_heard_alone(self):
        model = self.opened(LISTEN)
        stems = {name: stem["samples"] for name, stem in stems_of(LISTEN).items()}
        sides = {}
        for name in ("mid", "port", "star"):
            model.hear({name})
            sides[name] = model.render()["sides"]
            self.assertGreater(float(np.abs(stems[name]).max()), 0.05)
        # All to one side, a channel is its stem there, at the level it has in the song, and nothing on the other.
        np.testing.assert_array_equal(sides["port"][0], stems["port"])
        self.assertFalse(sides["port"][1].any())
        np.testing.assert_array_equal(sides["star"][1], stems["star"])
        self.assertLess(float(np.abs(sides["star"][0]).max()), 1e-12)
        # In the middle, each side has it 3 dB down.
        for side in sides["mid"]:
            np.testing.assert_allclose(side, stems["mid"] * math.sqrt(0.5), rtol=0, atol=1e-7)

    def test_a_hear_renders_nothing_again_and_a_mix_is_made_once_while_the_same_channels_are_heard(self):
        model = self.opened(LISTEN)
        with mock.patch.object(gui_model, "render_stems", wraps=render_stems) as engine, \
                mock.patch.object(gui_model, "mix_stems", wraps=mix_stems) as mixer:
            whole = model.render()
            self.assertEqual([engine.call_count, mixer.call_count], [1, 1])
            for mixes, only in enumerate(({"mid"}, {"port", "star"}, {"mid"}), 2):
                model.hear(only)
                self.assertFalse(model.render_ready())
                made = model.render()
                self.assertTrue(model.render_ready())
                self.assertIs(model.render(), made)
                model.numbers()
                self.assertEqual([engine.call_count, mixer.call_count], [1, mixes])
            # The whole song is there as it was rendered, with nothing to mix for it.
            model.hear(None)
            self.assertTrue(model.render_ready())
            self.assertIs(model.render(), whole)
            self.assertEqual([engine.call_count, mixer.call_count], [1, 4])

    def test_the_numbers_and_the_buffers_stay_those_of_the_whole_song_whatever_is_heard(self):
        model = self.opened(LISTEN)
        whole, numbers = model.render(), model.numbers()
        model.hear({"port"})
        made = model.render()
        self.assertIs(model.numbers(), numbers)
        self.assertIs(made["buffers"], whole["buffers"])
        self.assertEqual([made["rate"], made["loop"], made["seconds"]], [RATE, True, whole["seconds"]])
        row = np.mean(whole["sides"], axis=0, dtype=np.float64)
        np.testing.assert_array_equal(made["buffers"][0], row)
        self.assertEqual(numbers, {**measure(row, RATE), **brightness(row, rate=RATE)})
        self.assertFalse(np.array_equal(np.mean(made["sides"], axis=0, dtype=np.float64), row))

    def test_what_is_no_channel_of_the_picked_song_is_refused_and_what_was_heard_stays(self):
        model = self.opened(LISTEN)
        model.hear({"mid"})
        for only, said in (
            ({"mid", "tune"}, "the song 'trio' has no channel named 'tune'"),
            ([["mid"]], "the song 'trio' has no channel named ['mid']"),
            ("mid", "the channels to hear are a set of names or None, not 'mid'"),
            (3, "the channels to hear are a set of names or None, not 3"),
        ):
            with self.subTest(only=only):
                with self.assertRaises(RowSynthError) as refused:
                    model.hear(only)
                self.assertEqual(str(refused.exception), said)
                self.assertEqual(model.heard, {"mid"})
        # A sound has no channels to hear and nor has nothing: all of them is all there is to ask for.
        model.pick("sound", "ping")
        for other in (model, self.opened({})):
            with self.assertRaises(RowSynthError) as refused:
                other.hear(set())
            self.assertEqual(str(refused.exception), "no song is picked, so there are no channels to hear")
            other.hear(None)
            self.assertIsNone(other.heard)

    def test_picking_another_thing_has_all_its_channels_heard_and_picking_the_same_thing_again_does_not(self):
        model = self.opened(LISTEN)
        model.hear({"mid"})
        model.pick("song", "trio")
        self.assertEqual(model.heard, {"mid"})
        model.pick("song", "tune")
        self.assertIsNone(model.heard)
        self.assert_loop(model.render(), LISTEN, "tune")
        model.pick("song", "trio")
        self.assertIsNone(model.heard)
        self.assert_loop(model.render(), LISTEN, "trio")

    def test_a_channel_removed_or_added_in_the_file_has_all_heard_again_and_any_other_change_does_not(self):
        model = self.opened(LISTEN)
        model.hear({"mid", "port"})
        # Other notes and another pan: the song has the channels it had, and the same two are heard of it.
        after = copy.deepcopy(LISTEN)
        after["songs"]["trio"]["patterns"]["a"]["mid"] = "l4 c e g a"
        after["songs"]["trio"]["channels"]["port"]["pan"] = -0.5
        self.write(after)
        self.assertEqual(model.poll(), {("song", "trio")})
        self.assertEqual([model.heard, model.render_ready()], [{"mid", "port"}, False])
        stems = stems_of(after)
        self.assert_mix(model.render(), [stems["mid"], stems["port"]], near(stems))
        del after["songs"]["trio"]["channels"]["spare"]
        self.write(after)
        model.poll()
        self.assertIsNone(model.heard)
        self.assert_loop(model.render(), after, "trio")
        # A change made here is taken in as one from outside is.
        model.hear({"mid"})
        model.change("songs.trio.channels.spare", {"inst": "beep"})
        self.assertIsNone(model.heard)
        model.hear({"mid"})
        model.change("songs.trio", delete=True)
        self.assertEqual([model.picked, model.heard], [None, None])

    def test_a_song_the_engine_refuses_is_refused_as_it_was_while_only_some_of_its_channels_are_heard(self):
        model = self.opened(NO_ROWS)
        model.hear(set())
        with mock.patch.object(gui_model, "render_stems", wraps=render_stems) as engine:
            for ask in (model.render, model.numbers, model.render):
                with self.assertRaises(RowSynthError) as refused:
                    ask()
                self.assertEqual(str(refused.exception), NO_ROWS_SAID)
        self.assertEqual(engine.call_count, 1)
        self.assertTrue(model.render_ready())
        self.assertEqual(model.levels(0.0), {})

    def test_with_most_channels_heard_the_few_others_are_taken_out_of_the_mix_of_all_to_within_rounding(self):
        model = self.opened(QUINTET)
        whole, stems = model.render(), stems_of(QUINTET, "five")
        own = [side.copy() for side in whole["sides"]]
        # How far apart the two ways may be: for this song, well under a millionth of full scale.
        bound = near(stems)
        self.assertLess(bound, 1e-6)
        with mock.patch.object(gui_model, "mix_stems", wraps=mix_stems) as mixer:
            # One muted, two muted, and all but one: each time the mixer goes through the fewer stems.
            for gone, through in ((["bass"], ["bass"]), (["lead", "under"], ["lead", "under"]),
                                  (["lead", "second", "bass", "under"], ["high"])):
                with self.subTest(gone=gone):
                    heard = [name for name in stems if name not in gone]
                    model.hear(heard)
                    made = model.render()
                    self.assertEqual([stem["name"] for stem in mixer.call_args[0][0]], through)
                    # Stems that were added up are met sample for sample, and stems taken out within the bound.
                    self.assert_mix(made, [stems[name] for name in heard], bound if through == gone else 0.0)
            self.assertEqual(mixer.call_count, 3)
        # Nothing was taken out of the song's own sides where they are kept.
        self.assert_render(whole, own, True)

    def test_with_every_channel_heard_again_the_render_is_the_kept_one_of_the_whole_song(self):
        model = self.opened(QUINTET)
        whole = model.render()
        model.hear([name for name in FIVE["channels"] if name != "bass"])
        self.assertIsNot(model.render(), whole)
        model.hear(None)
        self.assertTrue(model.render_ready())
        self.assertIs(model.render(), whole)
        self.assert_loop(whole, QUINTET, "five")
        # Every channel by its name leaves nothing to take out: the sides are the kept ones themselves.
        model.hear(FIVE["channels"])
        for side, kept in zip(model.render()["sides"], whole["sides"], strict=True):
            self.assertIs(side, kept)

    def test_the_numbers_are_ready_while_a_mix_is_on_its_way_and_the_render_is_not(self):
        model = self.opened(LISTEN)
        self.assertFalse(model.numbers_ready())
        numbers = model.numbers()
        go = threading.Event()
        # A test that fails before it lets the mix go leaves no thread waiting.
        self.addCleanup(go.set)

        def held(*args: object) -> tuple:
            go.wait(60)
            return mix_stems(*args)

        with mock.patch.object(gui_model, "mix_stems", side_effect=held):
            model.hear({"mid"})
            model.start_render()
            (mixing,) = model._mixed._workers.values()
            self.assertEqual([model.render_ready(), model.numbers_ready()], [False, True])
            self.assertIs(model.numbers(), numbers)
            go.set()
            mixing.join()
        self.assertEqual([model.render_ready(), model.numbers_ready()], [True, True])


class ShiftTest(ModelCase):
    def test_a_song_at_a_shift_is_group_for_group_what_a_build_mixes_of_it_in_that_key(self):
        model = self.opened(LAYERED)
        song = Project.from_dict(LAYERED).song("keyed")
        built = render_groups(song, RATE)
        self.assertEqual([model.shift, built["keys"], built["groups"]["beat"]["keyless"]], [0, [0, 3], True])
        model.shift = 3
        self.assertEqual(model.shift, 3)
        # The whole of it is the loop of the song as the engine moves it up, which is another than as written.
        moved = render_loop(up(song, 3), RATE)
        self.assert_render(model.render(), [moved["left"], moved["right"]], True)
        self.assertFalse(np.array_equal(moved["left"], render_loop(song, RATE)["left"]))
        # The tuned group is its mix in that key. The drums are the one mix a build has of them for every key.
        for group, channels in KEYED["layers"]["groups"].items():
            with self.subTest(group=group):
                model.hear(channels)
                self.assert_render(model.render(), list(group_mix(built["groups"][group], 3)), True)
        model.shift = 0
        self.assert_render(model.render(), list(group_mix(built["groups"]["beat"], 0)), True)
        # The numbers are those of the song where it is heard, and down is a shift below 0.
        model.hear(None)
        written = model.numbers()
        model.shift = -12
        self.assertLess(model.numbers()["centroid_hz"], written["centroid_hz"])

    def test_the_render_of_a_shift_is_made_once_on_the_thread_of_a_render_and_kept_with_its_mixes(self):
        model = self.opened(LAYERED)
        with mock.patch.object(gui_model, "render_stems", wraps=render_stems) as engine:
            model.render()
            model.shift = 3
            self.assertFalse(model.render_ready())
            model.start_render()
            (worker,) = model._rendered._workers.values()
            worker.join()
            self.assertEqual([worker.name, model.render_ready(), engine.call_count], ["rowsynth-render", True, 2])
            made = model.render()
            # The same shift set again, a mix of some channels and the numbers are all made of that one render.
            model.shift = 3
            model.numbers()
            model.hear({"tune"})
            self.assertIsNot(model.render(), made)
            model.hear(None)
            self.assertIs(model.render(), made)
            self.assertEqual(engine.call_count, 2)

    def test_another_thing_picked_is_heard_as_it_is_written_and_the_same_thing_picked_again_keeps_its_shift(self):
        model = self.opened(LAYERED)
        model.shift = 3
        model.pick("song", "keyed")
        self.assertEqual(model.shift, 3)
        for picked in (("song", "tune"), ("song", "keyed")):
            model.pick(*picked)
            self.assertEqual(model.shift, 0)
            self.assert_loop(model.render(), LAYERED, picked[1])
        # A picked song that is gone from the file leaves nothing picked, and no shift.
        model.shift = 3
        model.change("songs.keyed", delete=True)
        self.assertEqual([model.picked, model.shift], [None, 0])

    def test_a_shift_is_refused_while_no_song_is_picked_and_when_it_is_no_whole_number_and_stays_as_it_was(self):
        model = self.opened(LAYERED)
        model.shift = 3
        for semis in (1.5, "2", True, None):
            with self.subTest(semis=semis):
                with self.assertRaises(RowSynthError) as refused:
                    model.shift = semis
                self.assertEqual(str(refused.exception),
                                 f"a song is heard a whole number of semitones up, not {semis!r}")
        self.assertEqual(model.shift, 3)
        model.pick("sound", "ping")
        with self.assertRaises(RowSynthError) as refused:
            model.shift = 3
        self.assertEqual(str(refused.exception), "no song is picked, so there is nothing to hear in another key")
        self.assertEqual(model.shift, 0)

    def test_what_the_engine_refuses_of_a_song_moved_up_is_refused_as_any_render_that_went_wrong(self):
        model = self.opened(LAYERED)
        with self.assertRaises(RowSynthError) as engine_said:
            render_stems(up(Project.from_dict(LAYERED).song("keyed"), 80), RATE)
        self.assertIn("the note at row 0 is 140", str(engine_said.exception))
        model.shift = 80
        with mock.patch.object(gui_model, "render_stems", wraps=render_stems) as engine:
            for ask in (model.render, model.numbers, model.render):
                with self.assertRaises(RowSynthError) as refused:
                    ask()
                self.assertEqual(str(refused.exception), str(engine_said.exception))
        self.assertEqual([engine.call_count, model.render_ready(), model.levels(0.0)], [1, True, {}])
        # Brought back to a shift the engine takes, the song is heard again.
        model.shift = 0
        self.assert_loop(model.render(), LAYERED, "keyed")


class LevelsTest(ModelCase):
    # What the middle channel reads at its loudest: its one sample, 3 dB down.
    MIDDLE = 0.5 * math.cos(math.pi / 4)

    def by_hand(self) -> Model:
        """A model of LISTEN whose trio is rendered into stems written by hand: each silent but for one sample."""
        stems = []
        for name, pan, at, sample in (("mid", 0, 4000, 0.5), ("port", -1, 6390, -0.25), ("star", 1, 100, 0.75)):
            samples = np.zeros(6400, dtype=np.float32)
            samples[at] = sample
            stems.append({"name": name, "pan": pan, "samples": samples})
        model = self.opened(LISTEN)
        played = {"stems": stems, "rows": 16, "loop_samples": 6400, "seconds": 0.8}
        with mock.patch.object(gui_model, "render_stems", return_value=played):
            model.render()
        return model

    def test_a_channel_reads_the_loudest_sample_of_its_stem_over_the_stretch_times_its_share_of_the_louder_side(self):
        model = self.by_hand()
        # The stretch is the 0.03 seconds up to the moment, 240 samples at 8000 Hz, and the sample at each end.
        for seconds, mid in ((0.49, 0.0), (0.5, self.MIDDLE), (0.515, self.MIDDLE), (0.53, self.MIDDLE), (0.54, 0.0)):
            with self.subTest(seconds=seconds):
                self.assertEqual(model.levels(seconds), {"mid": mid, "port": 0.0, "star": 0.0, "spare": 0.0})
        # A stretch of no time is the one sample, and one longer than the loop is the whole of it.
        self.assertEqual([model.levels(0.5, 0)["mid"], model.levels(0.5005, 0)["mid"]], [self.MIDDLE, 0.0])
        self.assertEqual(model.levels(0.2, 5), {"mid": self.MIDDLE, "port": 0.25, "star": 0.75, "spare": 0.0})

    def test_the_stretch_goes_back_round_the_end_of_the_loop_and_a_moment_past_the_loop_is_that_moment_a_round_on(self):
        model = self.by_hand()
        # 0.01 seconds in, the stretch began 160 samples before the end of the round before.
        self.assertEqual(model.levels(0.01), {"mid": 0.0, "port": 0.25, "star": 0.0, "spare": 0.0})
        self.assertEqual(model.levels(0.02), {"mid": 0.0, "port": 0.25, "star": 0.75, "spare": 0.0})
        self.assertEqual(model.levels(0.035), {"mid": 0.0, "port": 0.0, "star": 0.75, "spare": 0.0})
        for seconds in (0.51 + 0.8, 0.51 + 8.0, 0.51 - 1.6):
            self.assertEqual(model.levels(seconds), model.levels(0.51))
        self.assertEqual(model.levels(0.51)["mid"], self.MIDDLE)

    def test_a_channel_that_is_not_heard_reads_0_at_once_and_the_others_read_as_they_did(self):
        model = self.by_hand()
        model.hear({"mid", "star"})
        # The mix of the two is not made yet: the levels are read off the stems, by what is heard now.
        self.assertFalse(model.render_ready())
        self.assertEqual(model.levels(0.2, 5), {"mid": self.MIDDLE, "port": 0.0, "star": 0.75, "spare": 0.0})
        model.hear(set())
        self.assertEqual(model.levels(0.2, 5), dict.fromkeys(TRIO["channels"], 0.0))
        model.hear(None)
        self.assertEqual(model.levels(0.2, 5), {"mid": self.MIDDLE, "port": 0.25, "star": 0.75, "spare": 0.0})

    def test_there_are_no_levels_but_of_a_picked_song_whose_render_is_kept(self):
        model = self.opened(LISTEN)
        self.assertEqual(model.levels(0.1), {})
        model.render()
        read = model.levels(0.1)
        self.assertEqual(list(read), list(TRIO["channels"]))
        self.assertGreater(read["port"], 0.05)
        for picked in (("sound", "ping"), ("instrument", "hum")):
            model.pick(*picked)
            model.render()
            self.assertEqual(model.levels(0.1), {})
        self.assertEqual(self.opened({}).levels(0.1), {})

    def test_a_moment_or_a_stretch_that_is_no_number_of_seconds_is_refused(self):
        model = self.by_hand()
        for seconds, window in (("0.5", 0.03), (None, 0.03), (True, 0.03), (float("nan"), 0.03), (0.5, -0.01),
                                (0.5, float("inf")), (0.5, [0.03])):
            with self.subTest(seconds=seconds, window=window):
                with self.assertRaises(RowSynthError) as refused:
                    model.levels(seconds, window)
                self.assertEqual(str(refused.exception), "levels are read at a moment and over a stretch, both a "
                                 f"number of seconds, not {seconds!r} and {window!r}")

    def test_the_levels_of_eighteen_channels_cost_far_less_than_a_frame_has(self):
        names = [f"voice{at}" for at in range(18)]
        wide = {"row_ticks": 3, "channels": {name: {"inst": "hum", "pan": at / 9 - 1} for at, name in enumerate(names)},
                "patterns": {"a": dict.fromkeys(names, "l1 c")}, "order": ["a"]}
        self.write({"instruments": PROJECT["instruments"], "songs": {"wide": wide}})
        model = Model(self.path, 44100)
        model.render()
        took = []
        for _ in range(3):
            start = time.perf_counter()
            read = model.levels(0.4)
            took.append(time.perf_counter() - start)
        self.assertEqual(list(read), names)
        self.assertGreater(min(read.values()), 0.01)
        self.assertLess(min(took), 0.002)


class ToneTest(ModelCase):
    def note(self, inst: dict, pitch: float, ticks: float = TEST_TICKS) -> np.ndarray:
        """One note of an instrument as the bank renders a sound of it."""
        return render_entry({"def": {"voices": [{"inst": inst, "notes": [[0, pitch, ticks]]}]}}, RATE)[0]

    def test_a_tone_of_the_test_note_is_the_render_of_the_instrument_and_any_other_is_levelled_as_that_is(self):
        model = self.opened()
        model.pick("instrument", "hum")
        made = model.tone("hum", TEST_NOTE)
        self.assertEqual([list(made), made["rate"], len(made["sides"])], [["sides", "rate"], RATE, 1])
        np.testing.assert_array_equal(made["sides"][0], model.render()["sides"][0])
        np.testing.assert_array_equal(model.tone("beep", "g4", 3)["sides"][0],
                                      self.note(PROJECT["instruments"]["beep"], 67, 3))

    def test_a_song_s_own_instrument_is_found_before_the_project_s_of_the_same_name(self):
        model = self.opened(OWN)
        own, shared = model.tone("beep", 60, song="mine"), model.tone("beep", 60)
        np.testing.assert_array_equal(own["sides"][0], self.note(OWN["songs"]["mine"]["instruments"]["beep"], 60))
        np.testing.assert_array_equal(shared["sides"][0], self.note(PROJECT["instruments"]["beep"], 60))
        self.assertFalse(np.array_equal(own["sides"][0], shared["sides"][0]))
        # A song with no such instrument of its own, and a song the project does not have, have the project's.
        for song in ("tune", "nowhere"):
            self.assertIs(model.tone("beep", 60, song=song), shared)
        # A name the song gives to an instrument of the project is that instrument.
        self.assertIs(model.tone("low", 60, song="mine"), model.tone("hum", 60))

    def test_a_tone_is_made_once_for_the_same_instrument_pitch_and_ticks(self):
        model = self.opened()
        with mock.patch.object(gui_model, "render_entry", wraps=render_entry) as engine:
            made = model.tone("hum", "c4", 6)
            for pitch, ticks in (("c4", 6), (60, 6), (60.0, 6.0)):
                self.assertIs(model.tone("hum", pitch, ticks), made)
            self.assertEqual(engine.call_count, 1)
            for asked in (("beep", 60, 6), ("hum", 61, 6), ("hum", 60, 7)):
                self.assertIsNot(model.tone(*asked), made)
            self.assertEqual(engine.call_count, 4)

    def test_only_the_tones_asked_for_last_are_kept(self):
        model = self.opened()
        with mock.patch.object(gui_model, "TONES", 2):
            first, second = model.tone("hum", 60), model.tone("hum", 62)
            self.assertIs(model.tone("hum", 60), first)
            model.tone("hum", 64)
            # The second was asked for longest ago, so it is the one that was let go of.
            self.assertIs(model.tone("hum", 60), first)
            self.assertIsNot(model.tone("hum", 62), second)

    def test_the_tones_are_dropped_when_the_file_is_loaded_again_with_something_changed(self):
        model = self.opened()
        made = model.tone("hum", 60)
        self.write(PROJECT)
        self.assertEqual(model.poll(), set())
        self.assertIs(model.tone("hum", 60), made)
        after = changed(instruments={"hum": {"wave": "saw", "vol": 12}})
        self.write(after)
        self.assertEqual(model.poll(), {("instrument", "hum")})
        np.testing.assert_array_equal(model.tone("hum", 60)["sides"][0], self.note(after["instruments"]["hum"], 60))
        self.assertFalse(np.array_equal(model.tone("hum", 60)["sides"][0], made["sides"][0]))

    def test_an_instrument_written_in_place_is_heard_as_it_is_written_and_its_tones_are_kept_by_its_json_text(self):
        model = self.opened()
        written = {"wave": "saw", "vol": [15, 9, 3]}
        with mock.patch.object(gui_model, "render_entry", wraps=render_entry) as engine:
            made = model.tone(written, 64, 3)
            np.testing.assert_array_equal(made["sides"][0], self.note(written, 64, 3))
            # Another dict that holds the same is the same instrument, and one that holds something else is not.
            self.assertIs(model.tone(copy.deepcopy(written), "e4", 3), made)
            self.assertIsNot(model.tone({**written, "wave": "tri"}, 64, 3), made)
            self.assertEqual(engine.call_count, 2)
        # The tone has an instrument of its own: the dict it was made of is another one once it is changed.
        written["vol"].append(0)
        np.testing.assert_array_equal(model.tone(written, 64, 3)["sides"][0], self.note(written, 64, 3))
        self.assertFalse(np.array_equal(model.tone(written, 64, 3)["sides"][0], made["sides"][0]))

    def test_a_tone_that_cannot_be_made_is_refused(self):
        model = self.opened()
        for said, asked in (
            ("no instrument named 'bell'", ("bell", 60)),
            ("no instrument named ['hum']", (["hum"], 60)),
            ("def: voices[0].inst: unknown key 'wav'", ({"wav": "saw"}, 60)),
            ("an instrument holds what a file can hold, and this one does not: {'vol': {15}}", ({"vol": {15}}, 60)),
            ("not a note: 'x9'", ("hum", "x9")),
            ("the ticks of a note must be a number, not [6]", ("hum", 60, [6])),
            ("def: voices[0].notes[0][2]: must be 0 or more, not -1", ("hum", 60, -1)),
        ):
            with self.subTest(asked=asked):
                with self.assertRaises(RowSynthError) as refused:
                    model.tone(*asked)
                self.assertEqual(str(refused.exception), said)


class OwnInstrumentTest(ModelCase):
    def test_a_name_with_a_dot_is_an_instrument_of_a_song_s_own_unless_the_project_has_one_called_so(self):
        self.assertEqual([own("beep"), own("mine.beep"), own("mine.beep", {"beep"}), own("mine.beep", {"mine.beep"})],
                         [(None, "beep"), ("mine", "beep"), ("mine", "beep"), (None, "mine.beep")])
        self.assertEqual([inst_place(None, "beep"), inst_place("mine", "beep")],
                         ["instruments.beep", "songs.mine.instruments.beep"])
        held = Project.from_dict(OWN)
        self.assertEqual([inst_of(held, None, "beep"), inst_of(held, "mine", "beep")],
                         [PROJECT["instruments"]["beep"], MINE["instruments"]["beep"]])
        # A name the song only gives to an instrument of the project is none of its own, and nor is one that
        # only the project has, one of a song there is not, or a name nobody has.
        self.assertEqual([inst_of(held, *where) for where in (("mine", "low"), ("mine", "hum"), ("nowhere", "beep"),
                                                              (None, "low"))], [None] * 4)

    def test_the_listing_has_the_own_instruments_of_each_song_after_the_project_s_tagged_with_their_song(self):
        self.assertEqual(self.opened(OWN).listing()[2],
                         ("instruments", [("hum", ""), ("beep", ""), ("mine.beep", "mine")]))
        # Names written by hand with a dot in them: an instrument of the project that is called as one of a
        # song's own is picked, which is then not listed, and nor is one whose song or own name has a dot.
        saw = {"wave": "saw"}
        by_hand = {"instruments": {**PROJECT["instruments"], "air.beep": {"wave": "tri", "vol": 3}},
                   "songs": {"mine": {**MINE, "instruments": {**MINE["instruments"], "c.d": saw}},
                             "air": {**song("l2 e g"), "instruments": {"beep": saw}},
                             "a.b": {**song("l4 c"), "instruments": {"beep": saw}}}}
        model = self.opened(by_hand)
        self.assertEqual(model.listing()[2],
                         ("instruments", [("hum", ""), ("beep", ""), ("air.beep", ""), ("mine.beep", "mine")]))
        model.pick("instrument", "air.beep")
        note = {"def": {"voices": [{"inst": by_hand["instruments"]["air.beep"], "notes": [[0, "c4", 12]]}]}}
        self.assert_render(model.render(), render_entry(note, RATE), False)

    def test_an_own_instrument_is_picked_by_its_song_and_its_name_and_rendered_as_its_own_test_note(self):
        model = self.opened(OWN)
        model.pick("instrument", "mine.beep")
        self.assertEqual(model.picked, ("instrument", "mine.beep"))
        note = {"def": {"voices": [{"inst": MINE["instruments"]["beep"], "notes": [[0, "c4", 12]]}]}}
        self.assert_render(model.render(), render_entry(note, RATE), False)
        np.testing.assert_array_equal(model.render()["sides"][0],
                                      model.tone("beep", TEST_NOTE, song="mine")["sides"][0])
        self.assertFalse(np.array_equal(model.render()["sides"][0], model.tone("beep", TEST_NOTE)["sides"][0]))
        # A name the song only gives to one of the project's is not picked, nor one of a song with none such.
        for name in ("mine.low", "tune.beep", "nowhere.beep"):
            with self.subTest(name=name):
                with self.assertRaises(RowSynthError) as refused:
                    model.pick("instrument", name)
                self.assertEqual(str(refused.exception), f"no instrument named {name!r}")
        self.assertEqual(model.picked, ("instrument", "mine.beep"))

    def test_an_own_instrument_stays_picked_when_it_is_changed_and_is_picked_no_longer_once_it_is_gone(self):
        model = self.opened(OWN)
        model.pick("instrument", "mine.beep")
        model.change("songs.mine.instruments.beep.vol", 5)
        self.assertEqual([model.picked, model.project.songs["mine"]["instruments"]["beep"]],
                         [("instrument", "mine.beep"), {"wave": "saw", "vol": 5}])
        # Outside, the song gives the name to an instrument of the project, or the song itself is removed.
        named = {**OWN, "songs": {**OWN["songs"], "mine": {**MINE, "instruments": {"beep": "hum"}}}}
        for after in (named, PROJECT):
            with self.subTest(songs=list(after["songs"])):
                model = self.opened(OWN)
                model.pick("instrument", "mine.beep")
                self.write(after)
                self.assertEqual([model.poll(), model.picked], [{("song", "mine")}, None])


class FindingsTest(ModelCase):
    def test_the_findings_are_those_of_the_checks_at_the_model_s_rate_and_are_made_once(self):
        model = self.opened()
        with mock.patch.object(checks, "run", wraps=checks.run) as run:
            found = model.findings()
            self.assertIs(model.findings(), found)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(found, found_in(PROJECT))
        self.assertEqual([finding.target for finding in found], ["songs.air", "sounds.drone.start", "sounds.drone.end"])

    def test_a_project_the_checks_refuse_is_tried_once_and_refused_each_time_its_findings_are_asked_for(self):
        model = self.opened(UNREAD)
        with mock.patch.object(checks, "run", wraps=checks.run) as run:
            for _ in range(2):
                with self.assertRaises(RowSynthError) as refused:
                    model.findings()
                self.assertEqual(str(refused.exception), UNREAD_SAID)
        self.assertEqual(run.call_count, 1)
        self.assertTrue(model.findings_ready())
        self.assertIsNone(model.count())
        self.assertEqual(model.findings_for("song", "odd"), [])

    def test_the_count_is_how_many_findings_there_are_once_they_are_ready(self):
        for data, count in ((PROJECT, 3), (FAULTS, 2), ({"instruments": PROJECT["instruments"]}, 0)):
            with self.subTest(count=count):
                model = self.opened(data)
                self.assertIsNone(model.count())
                model.findings()
                self.assertEqual(model.count(), count)

    def test_the_findings_of_one_thing_are_those_whose_place_is_that_thing_or_lies_in_it(self):
        model = self.opened(FAULTS)
        self.assertEqual(model.findings_for("song", "air"), [])
        join, peak = model.findings()
        self.assertEqual([join.target, peak.target], ["songs.air", "sounds.zaps[1]"])
        for kind, name, found in (("song", "air", [join]), ("sound", "zaps", [peak]), ("song", "tune", []),
                                  ("sound", "zap", []), ("instrument", "hum", []), ("sound", "gone", [])):
            with self.subTest(kind=kind, name=name):
                self.assertEqual(model.findings_for(kind, name), found)
        with self.assertRaises(RowSynthError) as refused:
            model.findings_for("sounds", "zaps")
        self.assertEqual(str(refused.exception),
                         "a thing of a project is a song, a sound or an instrument, not 'sounds'")
        # The parts of a held sound are places in it.
        model = self.opened()
        _, start, end = model.findings()
        self.assertEqual(model.findings_for("sound", "drone"), [start, end])


class PollTest(ModelCase):
    def test_a_look_at_a_file_that_was_not_written_again_finds_nothing_and_keeps_the_render(self):
        model = self.opened()
        made = model.render()
        self.assertEqual(model.poll(), set())
        self.assertIsNone(model.problem)
        self.assertIs(model.render(), made)

    def test_a_look_after_the_file_was_written_again_gives_what_was_added_removed_or_changed(self):
        model = self.opened()
        self.write(changed(sounds={"ping": {"def": sound("b4")}, "pong": {"def": sound("d4")}, "crash": None},
                           instruments={"hum": {"wave": "tri", "vol": 9}}, songs={"air": None}))
        self.assertEqual(model.poll(), {("sound", "ping"), ("sound", "pong"), ("sound", "crash"),
                                        ("instrument", "hum"), ("song", "air")})
        self.assertEqual(model.listing(), [
            ("songs", [("tune", "")]),
            ("sounds", [("ping", ""), ("blip", "3"), ("drone", "held"), ("pong", "")]),
            ("instruments", [("hum", ""), ("beep", "")]),
        ])
        self.assertEqual(model.poll(), set())

    def test_a_change_to_the_chords_alone_is_given_as_each_chord_added_removed_or_changed(self):
        chord = {"root": 0, "tones": [0, 4, 7]}
        model = self.opened({**PROJECT, "chords": {"C": chord, "F": {"root": 5, "tones": [5, 9, 0]}, "Am": chord}})
        listing = model.listing()
        model.render()
        model.findings()
        self.write({**PROJECT, "chords": {"C": {"root": 0, "tones": [0, 4, 7, 11]}, "G": chord, "Am": chord}})
        self.assertEqual(model.poll(), {("chord", "C"), ("chord", "F"), ("chord", "G")})
        self.assertEqual([model.listing(), model.picked], [listing, ("song", "tune")])
        # The chords are part of what the checks go by, so what was kept is of a project that is no more.
        self.assertEqual([model.render_ready(), model.findings_ready()], [False, False])

    def test_a_file_written_again_with_nothing_in_it_changed_gives_nothing_and_keeps_what_was_kept(self):
        model = self.opened()
        made, found = model.render(), model.findings()
        # The same things, each table with its names the other way round.
        self.write({table: dict(reversed(things.items())) for table, things in PROJECT.items()})
        self.assertEqual(model.poll(), set())
        self.assertEqual([model.problem, model.picked], [None, ("song", "tune")])
        self.assertIs(model.render(), made)
        self.assertIs(model.findings(), found)
        self.assertEqual(model.listing(), [
            ("songs", [("air", ""), ("tune", "")]),
            ("sounds", [("drone", "held"), ("crash", "parts"), ("blip", "3"), ("ping", "")]),
            ("instruments", [("beep", ""), ("hum", "")]),
        ])
        self.assertEqual(model.poll(), set())

    def test_the_project_is_the_one_object_until_a_load_puts_another_in_its_place(self):
        model = self.opened()
        before = model.project
        self.assertEqual(model.poll(), set())
        self.assertIs(model.project, before)
        after = changed(songs={"tune": song("l4 c e g a"), "air": None})
        self.write(after)
        model.poll()
        self.assertIsNot(model.project, before)
        self.assertEqual(model.project, Project.from_dict(after))

    def test_after_a_load_the_picked_thing_is_still_picked_and_what_was_kept_of_it_is_made_again(self):
        model = self.opened()
        model.render()
        numbers, found = model.numbers(), model.findings()
        after = changed(songs={"tune": song("l4 c e g a"), "air": None})
        self.write(after)
        self.assertEqual(model.poll(), {("song", "tune"), ("song", "air")})
        self.assertEqual(model.picked, ("song", "tune"))
        self.assertFalse(model.render_ready())
        self.assert_loop(model.render(), after, "tune")
        self.assertNotEqual(model.numbers(), numbers)
        self.assertNotEqual(model.findings(), found)
        self.assertEqual(model.findings(), checks.run(Project.from_dict(after), RATE))

    def test_after_a_load_a_picked_thing_that_is_gone_is_picked_no_longer(self):
        model = self.opened()
        model.pick("sound", "crash")
        self.write(changed(sounds={"crash": None}))
        model.poll()
        self.assertIsNone(model.picked)

    def test_a_file_that_is_half_written_leaves_the_model_as_it_was_and_is_its_problem_until_it_is_whole(self):
        model = self.opened()
        made, listing = model.render(), model.listing()
        whole = json.dumps(changed(sounds={"ping": None}))
        self.write(whole[:len(whole) // 2])
        with self.assertRaises(RowSynthError) as loading:
            Project.load(self.path)
        for _ in range(2):
            self.assertEqual(model.poll(), set())
            self.assertEqual(model.problem, str(loading.exception))
        self.assertEqual([model.listing(), model.picked], [listing, ("song", "tune")])
        self.assertIs(model.render(), made)
        self.write(whole)
        self.assertEqual(model.poll(), {("sound", "ping")})
        self.assertIsNone(model.problem)

    def test_a_file_that_is_gone_is_the_model_s_problem_until_it_is_back(self):
        model = self.opened()
        made = model.render()
        self.path.unlink()
        self.assertEqual(model.poll(), set())
        self.assertEqual(model.problem, f"{self.path}: No such file or directory")
        self.assertEqual(model.picked, ("song", "tune"))
        # Back as it was, to the time it was written at: there is nothing to load again.
        self.write(PROJECT, later=False)
        self.assertEqual(model.poll(), set())
        self.assertIsNone(model.problem)
        self.assertIs(model.render(), made)

    def test_the_changes_of_a_look_are_each_whole_thing_that_was_added_removed_or_changed(self):
        chord = {"root": 0, "tones": [0, 4, 7]}
        before = {**copy.deepcopy(PROJECT), "chords": {"C": chord, "Am": chord}}
        before["instruments"]["spare"] = {"wave": "saw"}
        model = self.opened(before)
        self.assertEqual(model.changes, [])
        after = copy.deepcopy(before)
        del after["instruments"]["spare"], after["sounds"]["crash"], after["songs"]["air"], after["chords"]["C"]
        after["instruments"].update(beep={"vol": 9}, bell={"wave": "tri"})
        after["sounds"].update(ping={"def": sound("b4", inst="bell")}, pong={"def": sound("d4")})
        after["chords"]["G"] = {"root": 7, "tones": [7, 11, 2]}
        self.write(after)
        self.assertEqual(len(model.poll()), 9)
        # What a song or a sound may name comes before them when it is new, and after them when it is gone.
        self.assertEqual(model.changes, [
            ("instruments.beep", {"vol": [15, 12, 9, 6]}, {"vol": 9}),
            ("instruments.bell", NOTHING, {"wave": "tri"}),
            ("chords.G", NOTHING, {"root": 7, "tones": [7, 11, 2]}),
            ("songs.air", PROJECT["songs"]["air"], NOTHING),
            ("sounds.ping", PROJECT["sounds"]["ping"], {"def": sound("b4", inst="bell")}),
            ("sounds.crash", PROJECT["sounds"]["crash"], NOTHING),
            ("sounds.pong", NOTHING, {"def": sound("d4")}),
            ("instruments.spare", {"wave": "saw"}, NOTHING),
            ("chords.C", chord, NOTHING),
        ])
        self.assertEqual([model.poll(), model.changes], [set(), []])
        # A look that cannot read the file has no changes either.
        self.write("{")
        self.assertEqual([model.poll(), model.changes], [set(), []])


class ChangeTest(ModelCase):
    def held(self) -> dict:
        """What the file holds now."""
        return json.loads(self.path.read_text(encoding="utf-8"))

    def blocked(self, times: int) -> mock.Mock:
        """Has the file refuse to be put in place `times` times, as it does while another program has it open."""
        replace, left = os.replace, [times]

        def blocked(source: str, target: Path) -> None:
            if left[0]:
                left[0] -= 1
                raise PermissionError(13, "Permission denied")
            replace(source, target)

        patch = mock.patch.object(os, "replace", side_effect=blocked)
        self.addCleanup(patch.stop)
        return patch.start()

    def test_a_change_is_in_the_file_and_in_the_project_and_the_next_look_finds_nothing_in_it(self):
        model = self.opened()
        before = model.project
        model.render()
        model.findings()
        where, after = "songs.tune.patterns.a.tune", changed(songs={"tune": song("l4 c e g a")})
        self.assertEqual(model.change(where, "l4 c e g a"),
                         {"file": str(self.path), "where": where, "was": "l4 c e g > c", "now": "l4 c e g a"})
        self.assertEqual(self.held(), after)
        self.assertIsNot(model.project, before)
        self.assertEqual(model.project, Project.from_dict(after))
        # What was kept is of a project that is no more, as after any load.
        self.assertEqual([model.render_ready(), model.findings_ready()], [False, False])
        made = model.render()
        self.assert_loop(made, after, "tune")
        self.assertEqual([model.poll(), model.changes, model.problem], [set(), [], None])
        self.assertIs(model.render(), made)
        # What is there already is not written, and the model stays as it is.
        project = model.project
        self.assertTrue(model.change(where, "l4 c e g a")["same"])
        self.assertIs(model.project, project)
        self.assertIs(model.render(), made)

    def test_a_change_leaves_what_another_program_changed_since_the_last_look_but_for_the_place_it_went_over(self):
        model = self.opened()
        # Outside, beep gets a wave and ping another note. Here, an item is taken out of the volume of beep.
        self.write(changed(instruments={"beep": {"vol": [15, 12, 9, 6], "wave": "saw"}},
                           sounds={"ping": {"def": sound("b4")}}))
        model.change("instruments.beep.vol[1]", delete=True)
        self.assertEqual(self.held()["instruments"]["beep"], {"vol": [15, 9, 6], "wave": "saw"})
        ping = ("sounds.ping", PROJECT["sounds"]["ping"], {"def": sound("b4")})
        # As they came before the change, each from what the last look gave, and as they come after it.
        self.assertEqual(model.changes, [
            ("instruments.beep", {"vol": [15, 12, 9, 6]}, {"vol": [15, 12, 9, 6], "wave": "saw"}), ping])
        self.assertEqual(model.changes_after, [
            ("instruments.beep", {"vol": [15, 9, 6]}, {"vol": [15, 9, 6], "wave": "saw"}), ping])
        self.assertEqual(things(model.changes), {("instrument", "beep"), ("sound", "ping")})
        self.assertEqual([model.poll(), model.changes], [set(), []])
        # What another program wrote at the very place is gone over, and is none of them.
        where = "songs.tune.patterns.a.tune"
        data = self.held()
        data["songs"]["tune"]["patterns"]["a"]["tune"] = "l8 c d e f"
        self.write(data)
        self.assertEqual(model.change(where, "l2 c e")["was"], "l8 c d e f")
        self.assertEqual([model.changes, model.changes_after], [[], []])
        # Neither is a value that the file holds in another form than it was given in, as a list for a tuple.
        model.change("sounds.ping.meta", {"tags": ("a", "b")})
        self.assertEqual([self.held()["sounds"]["ping"]["meta"], model.changes], [{"tags": ["a", "b"]}, []])

    def test_a_delete_takes_a_thing_out_of_the_file_and_a_picked_thing_that_is_gone_is_picked_no_longer(self):
        model = self.opened()
        model.pick("sound", "crash")
        self.assertEqual(model.change("sounds.crash", delete=True),
                         {"file": str(self.path), "where": "sounds.crash", "was": PROJECT["sounds"]["crash"]})
        self.assertEqual(self.held(), changed(sounds={"crash": None}))
        self.assertIsNone(model.picked)
        self.assertEqual(model.poll(), set())

    def test_a_change_that_is_refused_leaves_the_file_and_the_model_as_they_were(self):
        model = self.opened()
        project, made, written = model.project, model.render(), self.path.read_bytes()
        with self.assertRaises(RowSynthError) as refused:
            model.change("sounds.ping.def.voices[0].inst", "bell")
        self.assertEqual(str(refused.exception), "sounds.ping.def.voices[0].inst: no instrument named 'bell'")
        self.assertEqual(self.path.read_bytes(), written)
        self.assertIs(model.project, project)
        self.assertIs(model.render(), made)
        self.assertEqual(model.poll(), set())
        # A file that is gone is refused in plain words, and not waited for.
        self.path.unlink()
        with self.assertRaises(RowSynthError) as refused:
            model.change("songs.tune.gain", 0.3)
        self.assertEqual(str(refused.exception), f"{self.path}: No such file or directory")
        self.assertIs(model.project, project)

    def test_a_save_another_program_stands_in_the_way_of_is_tried_again_after_a_wait(self):
        self.write(PROJECT)
        waits: list[float] = []
        model = Model(self.path, RATE, wait=waits.append)
        replace = self.blocked(2)
        self.assertEqual(model.change("songs.tune.gain", 0.3)["now"], 0.3)
        self.assertEqual([replace.call_count, waits], [3, list(SAVE_WAITS[:2])])
        self.assertEqual([self.held()["songs"]["tune"]["gain"], model.project.songs["tune"]["gain"]], [0.3, 0.3])
        # No try has left a file of its own beside the project.
        self.assertEqual(os.listdir(self.path.parent), [self.path.name])
        self.assertEqual(model.poll(), set())

    def test_a_save_that_fails_at_the_last_try_too_is_refused_and_nothing_is_changed(self):
        self.write(PROJECT)
        waits: list[float] = []
        model = Model(self.path, RATE, wait=waits.append)
        project, written = model.project, self.path.read_bytes()
        replace = self.blocked(len(SAVE_WAITS) + 1)
        with self.assertRaises(RowSynthError) as refused:
            model.change("songs.tune.gain", 0.3)
        self.assertEqual(str(refused.exception),
                         f"{self.path}: another program has the file open, so the change was not saved")
        self.assertEqual([replace.call_count, waits], [len(SAVE_WAITS) + 1, list(SAVE_WAITS)])
        # The waits are about a fifth of a second in all.
        self.assertAlmostEqual(sum(SAVE_WAITS), 0.2)
        self.assertEqual(self.path.read_bytes(), written)
        self.assertIs(model.project, project)
        self.assertEqual(model.poll(), set())


class WorkerTest(ModelCase):
    """A render and the checks on threads of their own. An event holds them back, so each test says when they end."""

    def held(self, error: Exception | None = None) -> threading.Event:
        """Has the engine wait before it renders a song, and the checks before they run. Gives what lets them go on.

        With `error`, each raises it once it is let go, and does nothing else.
        """
        go = threading.Event()

        def waiting(real: Callable) -> Callable:
            def wait(*args: object) -> object:
                go.wait(60)
                if error is not None:
                    raise error
                return real(*args)
            return wait

        for owner, name, kept_as in ((gui_model, "render_stems", "engine"), (checks, "run", "checker")):
            patch = mock.patch.object(owner, name, side_effect=waiting(getattr(owner, name)))
            setattr(self, kept_as, patch.start())
            self.addCleanup(patch.stop)
        # A test that fails before it lets them go leaves no thread waiting.
        self.addCleanup(go.set)
        return go

    def started(self, model: Model) -> threading.Thread:
        """The thread that a render of the picked thing is begun on."""
        model.start_render()
        (worker,) = model._rendered._workers.values()
        return worker

    def checking(self, model: Model) -> threading.Thread:
        """The thread that the checks are begun on."""
        model.start_findings()
        (worker,) = model._found._workers.values()
        return worker

    def test_a_render_begun_on_its_own_thread_is_ready_when_the_thread_is_through(self):
        model = self.opened()
        go = self.held()
        worker = self.started(model)
        model.start_render()
        self.assertEqual(list(model._rendered._workers.values()), [worker])
        self.assertFalse(model.render_ready())
        go.set()
        worker.join()
        self.assertTrue(model.render_ready())
        self.assertEqual(model._rendered._workers, {})
        self.assert_loop(model.render(), PROJECT, "tune")
        model.start_render()
        self.assertEqual(model._rendered._workers, {})
        self.assertEqual(self.engine.call_count, 1)

    def test_a_pick_made_while_a_render_runs_is_not_given_that_render(self):
        model = self.opened()
        go = self.held()
        worker = self.started(model)
        model.pick("sound", "ping")
        go.set()
        worker.join()
        self.assertFalse(model.render_ready())
        ping = render_entry(Project.from_dict(PROJECT).sound("ping"), RATE)
        self.assert_render(model.render(), ping, False)

    def test_a_load_made_while_a_render_runs_is_not_given_that_render(self):
        model = self.opened()
        go = self.held()
        worker = self.started(model)
        after = changed(songs={"tune": song("l4 c e g a")})
        self.write(after)
        model.poll()
        go.set()
        worker.join()
        self.assertEqual(model.picked, ("song", "tune"))
        self.assertFalse(model.render_ready())
        self.assert_loop(model.render(), after, "tune")

    def test_a_mix_begun_on_its_own_thread_waits_for_the_render_and_is_ready_when_both_are_through(self):
        model = self.opened(LISTEN)
        go = self.held()
        model.hear({"mid"})
        rendering = self.started(model)
        (mixing,) = model._mixed._workers.values()
        self.assertEqual([rendering.name, mixing.name], ["rowsynth-render", "rowsynth-mix"])
        model.start_render()
        self.assertEqual([*model._rendered._workers.values(), *model._mixed._workers.values()], [rendering, mixing])
        self.assertEqual([model.render_ready(), model.levels(0.1)], [False, {}])
        go.set()
        mixing.join()
        self.assertFalse(rendering.is_alive())
        self.assertTrue(model.render_ready())
        self.assertEqual([model._rendered._workers, model._mixed._workers], [{}, {}])
        stems = stems_of(LISTEN)
        self.assert_mix(model.render(), [stems["mid"]])
        # Other channels are mixed from the stems that are kept, on a thread too, with nothing rendered again.
        threads = []

        def mixed(*args: object) -> tuple:
            threads.append(threading.current_thread().name)
            return mix_stems(*args)

        with mock.patch.object(gui_model, "mix_stems", side_effect=mixed):
            model.hear({"port", "star"})
            self.assertFalse(model.render_ready())
            model.start_render()
            self.assertEqual(model._rendered._workers, {})
            self.assert_mix(model.render(), [stems["port"], stems["star"]], near(stems))
        self.assertEqual([threads, self.engine.call_count], [["rowsynth-mix"], 1])

    def test_a_hear_made_while_a_mix_is_on_its_way_is_not_given_that_mix_and_renders_nothing_again(self):
        model = self.opened(LISTEN)
        go = self.held()
        model.hear({"mid"})
        self.started(model)
        (mixing,) = model._mixed._workers.values()
        model.hear({"port"})
        go.set()
        mixing.join()
        # The song's render is the one of now, and is kept. The mix was of other channels than are heard now.
        self.assertEqual([model._rendered.ready(), model.render_ready()], [True, False])
        self.assert_mix(model.render(), [stems_of(LISTEN)["port"]])
        self.assertEqual(self.engine.call_count, 1)

    def test_asking_for_a_render_that_is_on_its_way_waits_for_it(self):
        model = self.opened()
        go = self.held()
        worker = self.started(model)
        join = worker.join
        # The engine is let go at the moment the model starts to wait for the thread, and not before.
        with mock.patch.object(worker, "join", side_effect=lambda: (go.set(), join())) as waited:
            made = model.render()
        self.assertEqual(waited.call_count, 1)
        self.assertEqual(self.engine.call_count, 1)
        self.assert_loop(made, PROJECT, "tune")

    def test_a_render_that_breaks_on_the_thread_is_told_once_and_kept_as_a_refusal_is(self):
        model = self.opened()
        go = self.held(ZeroDivisionError("broken"))
        with mock.patch.object(threading, "excepthook") as told:
            worker = self.started(model)
            go.set()
            worker.join()
        # The thread ended in the error, which is how its traceback comes to be written, the once.
        self.assertEqual([told.call_count, told.call_args[0][0].exc_type], [1, ZeroDivisionError])
        self.assertTrue(model.render_ready())
        self.assertEqual(model._rendered._workers, {})
        depths = set()
        for ask in (model.render, model.numbers, model.render):
            with self.assertRaises(ZeroDivisionError) as broke:
                ask()
            self.assertEqual(str(broke.exception), "broken")
            depths.add(len(traceback.extract_tb(broke.exception.__traceback__)))
        # Raised again and again, the error carries no more traceback than it did.
        self.assertEqual(len(depths), 1)
        model.start_render()
        self.assertEqual([model._rendered._workers, self.engine.call_count], [{}, 1])
        # It is what one thing of one load broke with, and is not taken for what another would do.
        model.pick("song", "air")
        self.assertFalse(model.render_ready())

    def test_checks_that_break_on_the_thread_are_told_once_and_kept_as_a_refusal_is(self):
        model = self.opened()
        go = self.held(ZeroDivisionError("broken"))
        with mock.patch.object(threading, "excepthook") as told:
            worker = self.checking(model)
            go.set()
            worker.join()
        self.assertEqual(told.call_count, 1)
        self.assertEqual([model.findings_ready(), model.count(), model.findings_for("song", "air")],
                         [True, None, []])
        for _ in range(2):
            with self.assertRaises(ZeroDivisionError):
                model.findings()
        self.assertEqual(self.checker.call_count, 1)
        self.write(changed(songs={"air": None}))
        model.poll()
        self.assertFalse(model.findings_ready())

    def test_findings_begun_on_their_own_thread_are_ready_when_the_thread_is_through(self):
        model = self.opened()
        go = self.held()
        worker = self.checking(model)
        model.start_findings()
        self.assertEqual(list(model._found._workers.values()), [worker])
        self.assertEqual([model.findings_ready(), model.count(), model.findings_for("song", "air")],
                         [False, None, []])
        go.set()
        worker.join()
        self.assertTrue(model.findings_ready())
        self.assertEqual(model._found._workers, {})
        self.assertEqual(model.count(), 3)
        self.assertEqual(model.findings(), found_in(PROJECT))
        self.assertEqual(self.checker.call_count, 1)

    def test_a_pick_made_while_the_checks_run_does_not_lose_their_findings(self):
        model = self.opened()
        go = self.held()
        worker = self.checking(model)
        model.pick("sound", "ping")
        go.set()
        worker.join()
        self.assertTrue(model.findings_ready())
        self.assertEqual(model.findings(), found_in(PROJECT))

    def test_a_load_made_while_the_checks_run_is_not_given_their_findings(self):
        model = self.opened()
        go = self.held()
        worker = self.checking(model)
        after = changed(songs={"air": None})
        self.write(after)
        model.poll()
        go.set()
        worker.join()
        self.assertEqual([model.findings_ready(), model.count()], [False, None])
        self.assertEqual(model.findings(), found_in(after))
        self.assertEqual(model.count(), 2)

    def test_a_render_and_the_checks_run_at_the_same_time_and_both_come_out_right(self):
        model = self.opened()
        go = self.held()
        workers = [self.started(model), self.checking(model)]
        self.assertEqual([worker.is_alive() for worker in workers], [True, True])
        self.assertEqual([model.render_ready(), model.findings_ready()], [False, False])
        go.set()
        for worker in workers:
            worker.join()
        self.assertEqual([model.render_ready(), model.findings_ready()], [True, True])
        self.assert_loop(model.render(), PROJECT, "tune")
        self.assertEqual(model.findings(), found_in(PROJECT))
        self.assertEqual([self.engine.call_count, self.checker.call_count], [1, 1])


if __name__ == "__main__":
    unittest.main()
