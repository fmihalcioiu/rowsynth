"""Tests for rowsynth.gui.mixer: where its strips stand, what they show of a song, what they save and what they hear."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from collections.abc import Callable, Sequence
from pathlib import Path

from rowsynth.gui import dials, layout, theme
from rowsynth.gui.kit import ScrollBar
from rowsynth.gui.lists import DIMMED, mix
from rowsynth.gui.mixer import (APART, BAR_HELP, DESK_HELP, FADER_LEAST, HIGH_LEAST, SONG_METER_W, WIDE_LEAST, Laid,
                                Mixer, along, lay, ordered, strip_parts)
from rowsynth.gui.model import Model
from rowsynth.gui.panels import Panel
from rowsynth.gui.session import POLL_EVERY, Session
from rowsynth.gui.ui import Event, Ui

#: The room the mixer has in a song panel in the middle of the art surface, and in the least song panel.
DESK = (131, 122, 648, 397)
SMALL = (131, 122, 486, 193)
#: A room in which a fader travels 200 pixels: one for each step of a channel's level, and two for a step of the
#: song's. And a room of that height that is too narrow for more than two strips beside the song's.
ROOM = (131, 122, 648, 247)
NARROW = (131, 122, 109, 247)

#: A song that says its own level. Its drum channel is the second it lists, and its third says no level.
WALK = {
    "row_ticks": 6,
    "gain": 0.4,
    "drums": {"k": {"inst": "thud", "note": "c2"}},
    "channels": {"lead": {"inst": "tone", "gain": 0.8},
                 "kit": {"inst": "thud", "gain": 0.6, "drums": True},
                 "low": {"inst": "buzz"},
                 "a_long_name": {"inst": "tone", "gain": 0.5}},
    "patterns": {"a": {"lead": "o4 l4 c e g e", "kit": "l4 k k k k", "low": "o2 l2 c g", "a_long_name": "l1 r"}},
    "order": ["a"],
}
PLAIN = {"row_ticks": 6, "channels": {"air": {"inst": "tone"}}, "patterns": {"a": {"air": "o4 l1 c"}}, "order": ["a"]}
#: A song of three tuned channels.
TRIO = {"row_ticks": 6, "channels": {name: {"inst": "tone"} for name in ("one", "two", "three")},
        "patterns": {"a": {"one": "o4 l1 c"}}, "order": ["a"]}
PROJECT = {
    "instruments": {"tone": {"wave": "tri"}, "thud": {"wave": "noise"}, "buzz": {"wave": "saw"}},
    "songs": {"walk": WALK, "plain": PLAIN, "trio": TRIO},
}
#: The channels of the song as the mixer has them, and the colour each has by its place in the song.
ORDER = ["lead", "low", "a_long_name", "kit"]
COLOURS = [theme.SONG, theme.SALMON, theme.MUSTARD, theme.SOUND]
#: The rate the project is rendered at on the threads of its model: a low one, to be quick.
RATE = 8000
#: The time the first write from outside is stamped with, in nanoseconds: none the file gets by being saved.
STAMP = 1_700_000_000 * 10**9


class Speaker:
    """A stand-in for the player: it keeps what it was asked, plays nothing, and is as loud as the test says."""

    def __init__(self) -> None:
        self.asked: list[tuple] = []
        self.playing = False
        # The seconds into what plays, and how loud the two sides are, which a test can set.
        self.at = 0.0
        self.sides = (0.0, 0.0)

    def play(self, sides: Sequence, loop: bool = False) -> None:
        self.asked.append(("play", sides, loop))
        self.playing = True

    def also(self, sides: Sequence) -> None:
        self.asked.append(("also", sides))

    def stop(self, fade: float = 0.0) -> None:
        self.asked.append(("stop",))
        self.playing = False

    def swap(self, sides: Sequence) -> None:
        self.asked.append(("swap", sides))

    def position(self) -> float:
        return self.at

    def level(self) -> tuple[float, float]:
        return self.sides


def settle() -> None:
    """Wait for every render and every run of the checks that is still on a thread of a model's."""
    for thread in threading.enumerate():
        if thread.name.startswith("rowsynth-"):
            thread.join()


def differing(one: object, other: object, where: str = "") -> list[str]:
    """The places at which the data of two files differ, each spelt as a change spells its place."""
    if isinstance(one, dict) and isinstance(other, dict):
        return [place for key in dict.fromkeys([*one, *other])
                for place in (differing(one[key], other[key], f"{where}.{key}".lstrip("."))
                              if key in one and key in other else [f"{where}.{key}".lstrip(".")])]
    return [] if json.dumps(one) == json.dumps(other) else [where]


def middle(rect: layout.Rect) -> tuple[int, int]:
    """The pixel in the middle of a rectangle."""
    return rect[0] + rect[2] // 2, rect[1] + rect[3] // 2


def grip(fader: dials.Slider) -> tuple[int, int]:
    """The pixel of a fader's handle at which a press leaves the fader at the value it has."""
    x, y, w, h = fader.handle()
    return x + w // 2, y + h // 2 - 1


def click(ui: Ui, pos: tuple[int, int], button: int = 1) -> None:
    """Press a mouse button at a position and let it go there."""
    ui.handle(Event("down", pos, button=button))
    ui.handle(Event("up", pos, button=button))


def side_by_side(rects: Sequence[layout.Rect]) -> bool:
    """Whether each rectangle ends where the next begins or before it, from the left."""
    return all(x + w <= after for (x, _, w, _), (after, _, _, _) in zip(rects, rects[1:]))


class Sums(unittest.TestCase):
    def test_the_tuned_channels_come_first_and_then_the_drums_each_with_its_place_in_the_song(self) -> None:
        self.assertEqual(ordered(WALK["channels"]), [(0, "lead"), (2, "low"), (3, "a_long_name"), (1, "kit")])
        self.assertEqual((ordered(PLAIN["channels"]), ordered({})), ([(0, "air")], []))

    def test_strips_that_fit_share_the_room_before_the_strip_of_the_song_with_a_gap_before_the_drums(self) -> None:
        # The eighteen channels of the trailer piece in the song panel the layout is drawn for: all are in view.
        laid = lay(DESK, 10, 8)
        self.assertEqual((laid.wide, laid.first, laid.most, laid.bar, laid.song),
                         (33, 0, 0, None, (743, 122, 36, 397)))
        self.assertEqual(list(laid.strips), list(range(18)))
        self.assertEqual([laid.strips[index] for index in (0, 9, 10, 17)],
                         [(131, 122, 33, 397), (428, 122, 33, 397), (469, 122, 33, 397), (700, 122, 33, 397)])
        # Before the strip of the song the room ends, with what stands apart left free.
        self.assertLessEqual(700 + 33, laid.song[0] - APART)
        # Few channels are no wider than the most, and stand at the left.
        self.assertEqual(lay(DESK, 3, 1).strips, {0: (131, 122, 44, 397), 1: (175, 122, 44, 397),
                                                  2: (219, 122, 44, 397), 3: (271, 122, 44, 397)})
        # There is a gap only between a tuned channel and a drum channel, and a song with no channels has its own strip.
        self.assertEqual([[rect[0] for rect in lay(DESK, *counts).strips.values()] for counts in ((2, 0), (0, 2))],
                         [[131, 175], [131, 175]])
        self.assertEqual(lay(DESK, 0, 0), Laid(44, 0, 0, {}, (743, 122, 36, 397), None))

    def test_at_the_least_width_the_strips_still_fit_and_a_pixel_narrower_they_scroll(self) -> None:
        # Eighteen strips of 32 and the gap are 584 pixels, and the strip of the song with what is before it 45.
        laid = lay((0, 0, 629, 200), 10, 8)
        self.assertEqual((laid.wide, laid.most, laid.bar, len(laid.strips), laid.strips[17]),
                         (32, 0, None, 18, (552, 0, 32, 200)))
        laid = lay((0, 0, 628, 200), 10, 8)
        # The last does not fit whole, so it is not in view, and the strips are less high by the bar under them.
        self.assertEqual((laid.wide, laid.most, laid.bar, list(laid.strips), laid.strips[16], laid.song),
                         (32, 1, (0, 196, 583, 4), list(range(17)), (520, 0, 32, 195), (592, 0, 36, 195)))
        # The bar is as thick as the scroll bar of any view, and keeps as far from the strips.
        self.assertEqual((laid.bar[3], laid.bar[1] - 195), (theme.SCROLL_W, theme.SCROLL_GAP))

    def test_a_mixer_that_scrolls_shows_whole_strips_from_the_first_in_view_and_the_gap_between_two_of_them(
            self) -> None:
        # The trailer piece in the least song panel: 441 pixels for the channels.
        seen = {first: lay(SMALL, 10, 8, first) for first in range(-1, 8)}
        self.assertEqual({first: (laid.first, laid.most) for first, laid in seen.items()},
                         {-1: (0, 5), 0: (0, 5), 1: (1, 5), 2: (2, 5), 3: (3, 5), 4: (4, 5), 5: (5, 5), 6: (5, 5),
                          7: (5, 5)})
        self.assertEqual([list(seen[first].strips) for first in (0, 3, 5)],
                         [list(range(0, 13)), list(range(3, 16)), list(range(5, 18))])
        for laid in seen.values():
            rects = list(laid.strips.values())
            self.assertTrue(side_by_side(rects))
            self.assertTrue(all(131 <= x and x + w <= 131 + 441 and (w, h) == (32, 188) for x, _, w, h in rects))
            # Each strip begins where the one before it ends, but for the first drum channel.
            self.assertEqual([index for index in list(laid.strips)[1:]
                              if laid.strips[index][0] != laid.strips[index - 1][0] + 32], [10])
            self.assertEqual(laid.strips[10][0], laid.strips[9][0] + 40)
        # In a room for three strips the drums can be the first in view, and then nothing is left free before them.
        self.assertEqual([(laid.most, {index: rect[0] for index, rect in laid.strips.items()})
                          for laid in (lay((0, 0, 141, 200), 10, 8, first) for first in (9, 10))],
                         [(15, {9: 0, 10: 40}), (15, {10: 0, 11: 32, 12: 64})])

    def test_the_scroll_bar_is_shown_the_strips_in_pixels_so_its_thumb_is_what_those_in_view_are_of_them_all(
            self) -> None:
        # Thirteen of eighteen strips of 32 pixels are in view: the thumb is 13 eighteenths of the bar.
        seen = [lay(SMALL, 10, 8, first) for first in (0, 2, 5)]
        self.assertEqual([(laid.bar, along(laid, 18)) for laid in seen],
                         [((131, 311, 441, 4), (576, 416, 0)), ((131, 311, 441, 4), (576, 416, 64)),
                          ((131, 311, 441, 4), (576, 416, 160))])
        bar = ScrollBar(seen[0].bar)
        thumbs = []
        for laid in seen:
            bar.show(*along(laid, 18))
            thumbs.append(bar.thumb())
        self.assertEqual(thumbs, [(131, 311, 319, 4), (180, 311, 319, 4), (253, 311, 319, 4)])
        # With every strip in view all of them are in view, and there is no thumb.
        bar.show(*along(lay(DESK, 10, 8), 18))
        self.assertEqual((along(lay(DESK, 10, 8), 18), bar.thumb()), ((594, 594, 0), None))

    def test_a_rectangle_too_small_for_a_strip_beside_the_song_or_for_a_fader_has_no_strip_in_it(self) -> None:
        nothing = Laid(0, 0, 0, {}, (5, 7, 0, 0), None)
        self.assertEqual([lay((5, 7, WIDE_LEAST - 1, 300), 2, 1), lay((5, 7, 300, HIGH_LEAST - 1), 2, 1),
                          lay((5, 7, 0, 0), 2, 1)], [nothing] * 3)
        # At the least size one strip is in view, and under the bar its fader is the least a fader is.
        laid = lay((5, 7, WIDE_LEAST, HIGH_LEAST), 2, 1)
        self.assertEqual((laid.most, laid.strips, laid.song), (2, {0: (5, 7, 32, 65)}, (46, 7, 36, 65)))
        self.assertEqual(strip_parts(laid.strips[0])["fader"][3], FADER_LEAST)

    def test_the_parts_of_a_strip_stand_one_under_the_other_with_the_meter_beside_the_fader(self) -> None:
        self.assertEqual(strip_parts((131, 122, 44, 247)), {
            "number": (131, 122, 44, 11), "meter": (142, 135, 6, 206), "fader": (151, 135, 12, 206),
            "mute": (139, 343, 13, 13), "solo": (154, 343, 13, 13), "both": (139, 343, 28, 13),
            "name": (131, 358, 44, 11)})
        # The strip of the song has a meter of two bars, and one button in the room of the two toggles.
        whole = strip_parts((743, 122, 36, 247), SONG_METER_W)
        self.assertEqual((whole["meter"], whole["fader"], whole["both"]),
                         ((748, 135, 11, 206), (762, 135, 12, 206), (747, 343, 28, 13)))
        self.assertEqual(set(strip_parts((5, 7, 0, 0)).values()), {(5, 7, 0, 0)})


class Opened(unittest.TestCase):
    """A project file of the test's own, a real session on it with a stand-in player, and the mixer on a panel."""

    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        # No thread of the model is left to run into the next test, or into a folder that is gone.
        self.addCleanup(settle)
        self.path = Path(folder.name) / "project.json"
        self.path.write_text(json.dumps(PROJECT), encoding="utf-8")
        self.now, self.stamp = 100.0, STAMP
        self.speaker = Speaker()
        self.session = Session(Model(self.path, RATE), self.speaker, lambda: self.now)
        self.ui = Ui()
        self.picks: list[str] = []
        self.mixer = Mixer(Panel(self.session, self.ui), self.picks.append, lambda: self.now)
        self.mixer.place(ROOM)
        self.mixer.seen(True)
        self.mixer.show("walk", "low")
        self.lead, self.low, self.long, self.kit = self.mixer.strips

    def data(self) -> dict:
        """What the file holds now."""
        return json.loads(self.path.read_text(encoding="utf-8"))

    def song(self) -> dict:
        """What the file holds now for the song."""
        return self.data()["songs"]["walk"]

    def outside(self, change: Callable[[dict], object]) -> None:
        """Have another program change the song and write the file, and the session look at it."""
        project = self.data()
        change(project["songs"]["walk"])
        self.path.write_text(json.dumps(project), encoding="utf-8")
        # Stamped as no write before it was, so that the look cannot take the file for the one it knows.
        self.stamp += 10**9
        os.utime(self.path, ns=(self.stamp, self.stamp))
        self.now += POLL_EVERY
        self.session.tick()

    def drag(self, fader: dials.Slider, up: int, let_go: bool = True) -> tuple[int, int]:
        """Press the left button on the handle of a fader and move so many pixels up. Where the mouse then is."""
        x, y = grip(fader)
        self.ui.handle(Event("down", (x, y), button=1))
        self.ui.handle(Event("move", (x, y - up)))
        if let_go:
            self.ui.handle(Event("up", (x, y - up), button=1))
        return x, y - up

    def seen(self) -> list[str]:
        """The channels whose strips are on the window, by the names at their feet."""
        return [strip.name.text for strip in self.mixer.strips if strip.name.visible]

    def dim(self) -> list[str]:
        """The channels whose strips are dim."""
        return [strip.channel for strip in self.mixer.strips if strip.dim]


class Shown(Opened):
    def test_each_channel_has_a_strip_with_its_level_and_its_name_in_its_colour_and_the_song_has_one_too(self) -> None:
        mixer, strips = self.mixer, self.mixer.strips
        self.assertEqual(([strip.channel for strip in strips], self.seen()), (ORDER, ORDER))
        self.assertEqual([(strip.name.colour, strip.fader.colour) for strip in strips],
                         [(colour, colour) for colour in COLOURS])
        # A level the file does not say is the 1 the engine takes, and every fader has the travel of the bar's knob.
        self.assertEqual([(strip.number.text, strip.fader.value) for strip in strips],
                         [("0.8", 0.8), ("1", 1), ("0.5", 0.5), ("0.6", 0.6)])
        self.assertEqual({(strip.fader.low, strip.fader.high, strip.fader.steps, strip.fader.default)
                          for strip in strips}, {(0, 2, 200, 1)})
        self.assertEqual([strip.name.picked for strip in strips], [False, True, False, False])
        # The strips stand as they are laid out, the drums after a gap, and each part where the sums have it.
        self.assertEqual(mixer.laid, lay(ROOM, 3, 1))
        for strip, rect in zip(strips, mixer.laid.strips.values()):
            parts = strip_parts(rect)
            self.assertEqual({part: widget.rect for part, widget in strip.parts().items()},
                             {part: parts[part] for part in strip.parts()})
        self.assertEqual((mixer.number.text, mixer.name.text, mixer.fader.value, mixer.fader.low, mixer.fader.high,
                          mixer.fader.steps, mixer.fader.default), ("0.4", "ALL", 0.4, 0, 1, 100, 0.35))
        whole = strip_parts(mixer.laid.song, SONG_METER_W)
        self.assertEqual((mixer.meter.rect, mixer.fader.rect, mixer.clear.rect, len(mixer.meter.levels)),
                         (whole["meter"], whole["fader"], whole["both"], 2))
        self.assertTrue(all(widget.visible for widget in mixer.widgets))
        self.assertEqual((len(mixer.widgets), self.ui.widgets), (6 + 4 * 6, mixer.widgets))

    def test_the_help_line_has_the_whole_name_and_the_instrument_and_a_press_on_the_name_tells_the_channel(
            self) -> None:
        self.ui.handle(Event("move", middle(self.long.name.rect)))
        self.assertEqual((self.long.name.text, self.ui.help),
                         ("a_long_name", "channel a_long_name, tone   press: show its notes"))
        click(self.ui, middle(self.long.name.rect))
        click(self.ui, middle(self.kit.name.rect))
        self.assertEqual(self.picks, ["a_long_name", "kit"])
        # A press that is let go of elsewhere tells nothing.
        self.ui.handle(Event("down", middle(self.lead.name.rect), button=1))
        self.ui.handle(Event("up", middle(self.lead.number.rect), button=1))
        self.assertEqual(self.picks, ["a_long_name", "kit"])

    def test_another_song_has_its_own_strips_and_off_the_window_no_widget_is_seen_or_takes_the_mouse(self) -> None:
        mixer = self.mixer
        self.drag(self.lead.fader, 10, let_go=False)
        mixer.show("plain")
        # What the hand was doing to a fader is dropped, and the strips the song has no channel for show none.
        self.assertEqual((self.seen(), [strip.channel for strip in mixer.strips], mixer.busy),
                         (["air"], ["air", None, None, None], False))
        self.assertEqual((self.lead.fader.value, self.lead.name.colour, mixer.fader.value, mixer.number.text),
                         (1, theme.SONG, 0.35, "0.35"))
        self.ui.handle(Event("up", middle(self.lead.fader.rect), button=1))
        self.assertEqual((self.data(), self.session.can_undo), (PROJECT, False))
        mixer.show(None)
        self.assertEqual((self.seen(), mixer.number.visible), ([], True))
        mixer.show("walk")
        mixer.seen(False)
        self.assertEqual([widget for widget in mixer.widgets if widget.visible], [])
        for strip in mixer.strips:
            click(self.ui, middle(strip.mute.rect))
            click(self.ui, middle(strip.fader.rect))
        self.assertEqual((self.session.muted, self.data(), self.ui.handle(Event("wheel", middle(ROOM), wheel=1))),
                         (frozenset(), PROJECT, False))


class Heard(Opened):
    def test_the_toggles_have_the_session_mute_and_solo_and_are_lit_from_what_it_has(self) -> None:
        session = self.session
        click(self.ui, middle(self.lead.mute.rect))
        click(self.ui, middle(self.low.solo.rect))
        self.assertEqual((session.muted, session.soloed), ({"lead"}, {"low"}))
        self.assertEqual([(strip.mute.is_on, strip.solo.is_on) for strip in self.mixer.strips],
                         [(True, False), (False, True), (False, False), (False, False)])
        self.assertEqual({(strip.mute.text, strip.mute.colour, strip.solo.text, strip.solo.colour)
                          for strip in self.mixer.strips}, {("M", theme.MUSTARD, "S", theme.GOOD)})
        # What the session was told elsewhere is lit at the next frame, and a second press takes a mute off.
        session.mute("kit")
        session.solo("low")
        self.mixer.update()
        click(self.ui, middle(self.lead.mute.rect))
        self.assertEqual((session.muted, session.soloed), ({"kit"}, frozenset()))
        self.assertEqual([(strip.mute.is_on, strip.solo.is_on) for strip in self.mixer.strips],
                         [(False, False), (False, False), (False, False), (True, False)])
        # Nothing of it is in the file, and nothing is a step.
        self.assertEqual((self.data(), session.can_undo), (PROJECT, False))

    def test_a_strip_is_dim_when_it_is_muted_and_when_another_is_soloed_and_not_when_it_is_soloed_itself(self) -> None:
        click(self.ui, middle(self.lead.mute.rect))
        self.assertEqual(self.dim(), ["lead"])
        faint = mix(theme.PANEL, theme.SONG, DIMMED)
        self.assertEqual((self.lead.name.colour, self.lead.fader.colour, self.lead.number.colour),
                         (faint, faint, theme.DIM))
        self.assertEqual((self.low.name.colour, self.low.fader.colour, self.low.number.colour),
                         (theme.SALMON, theme.SALMON, theme.TEXT))
        click(self.ui, middle(self.low.solo.rect))
        self.assertEqual(self.dim(), ["lead", "a_long_name", "kit"])
        # A channel that is muted and soloed is heard.
        click(self.ui, middle(self.lead.solo.rect))
        self.assertEqual((self.dim(), self.lead.name.colour, self.session.model.heard),
                         (["a_long_name", "kit"], theme.SONG, {"lead", "low"}))

    def test_the_button_that_clears_is_enabled_only_with_a_mute_or_a_solo_and_takes_them_all_off(self) -> None:
        clear = self.mixer.clear
        self.ui.handle(Event("move", middle(clear.rect)))
        self.assertEqual((clear.enabled, clear.icon, self.ui.help), (False, "speaker", "nothing is muted or soloed"))
        click(self.ui, middle(clear.rect))
        # A solo alone is something to clear, and so is a mute alone.
        enabled = []
        for toggle in (self.low.solo, self.low.solo, self.kit.mute, self.low.solo):
            click(self.ui, middle(toggle.rect))
            enabled.append(clear.enabled)
        self.assertEqual(enabled, [True, False, True, True])
        self.ui.handle(Event("move", middle(clear.rect)))
        self.assertEqual((clear.enabled, self.ui.help),
                         (True, "press: hear every channel again: no mute and no solo"))
        # The two toggles of a strip say what a press on them does.
        for toggle, begins in ((self.low.mute, "press: mute the channel"), (self.low.solo, "press: solo the channel")):
            self.ui.handle(Event("move", middle(toggle.rect)))
            self.assertTrue(self.ui.help.startswith(begins), self.ui.help)
        click(self.ui, middle(clear.rect))
        self.assertEqual((self.session.muted, self.session.soloed, self.session.model.heard, self.dim(), clear.enabled),
                         (frozenset(), frozenset(), None, [], False))
        self.assertEqual({(strip.mute.is_on, strip.solo.is_on) for strip in self.mixer.strips}, {(False, False)})

    def test_a_solo_the_session_refuses_is_not_lit_and_the_session_says_why(self) -> None:
        def gone(song: dict) -> None:
            """Take a channel out of the song."""
            del song["channels"]["low"], song["patterns"]["a"]["low"]

        # The session has taken the file up, and the mixer has not had its frame: the strip still stands.
        self.outside(gone)
        click(self.ui, middle(self.low.solo.rect))
        self.assertEqual((self.low.solo.is_on, self.session.soloed, self.session.message),
                         (False, frozenset(), "the song 'walk' has no channel named 'low'"))
        self.mixer.update()
        self.assertEqual(self.seen(), ["lead", "a_long_name", "kit"])

    def test_the_meters_take_the_levels_of_the_session_and_their_lines_fall_with_the_time_that_passed(self) -> None:
        mixer, session, speaker = self.mixer, self.session, self.speaker
        session.play()
        settle()
        session.tick()
        # A fifth of a second in, the first note of the lead sounds, and the channel that only rests is silent.
        speaker.at, speaker.sides = 0.2, (0.5, 0.25)
        mixer.update(0.0)
        levels = session.levels()
        self.assertEqual((speaker.playing, sorted(levels)), (True, sorted(ORDER)))
        self.assertEqual([strip.meter.levels for strip in mixer.strips], [(levels[name],) for name in ORDER])
        self.assertGreater(levels["lead"], 0.1)
        self.assertEqual((levels["a_long_name"], self.long.meter.tops, mixer.meter.levels), (0.0, [0.0], (0.5, 0.25)))
        top, tops = self.lead.meter.tops[0], list(mixer.meter.tops)
        self.assertEqual((top, tops), (dials.meter_height(levels["lead"]), [dials.meter_height(0.5),
                                                                           dials.meter_height(0.25)]))
        # A song that has stopped has no levels: the bars are down, and the lines fall a quarter of the way a second.
        speaker.playing, speaker.sides = False, (0.0, 0.0)
        mixer.update(1.0)
        fall = theme.METER_FALL_DB / -theme.METER_FLOOR_DB
        self.assertEqual((self.lead.meter.levels, mixer.meter.levels, fall), ((0.0,), (0.0, 0.0), 0.25))
        self.assertAlmostEqual(self.lead.meter.tops[0], top - fall)
        self.assertAlmostEqual(mixer.meter.tops[0], tops[0] - fall)
        mixer.update(10.0)
        self.assertEqual((self.lead.meter.tops, mixer.meter.tops), ([0.0], [0.0, 0.0]))


class Saved(Opened):
    def test_a_fader_that_is_dragged_changes_nothing_in_the_file_until_it_is_let_go_and_then_its_one_place(
            self) -> None:
        fader, before = self.lead.fader, self.data()
        mouse = self.drag(fader, 10, let_go=False)
        # The fader and the number over it show where the hand has it.
        self.assertEqual((self.lead.number.text, self.mixer.busy), ("0.9", True))
        self.assertEqual((self.data(), self.session.can_undo), (before, False))
        self.ui.handle(Event("up", mouse, button=1))
        self.assertEqual(differing(before, self.data()), ["songs.walk.channels.lead.gain"])
        self.assertEqual((self.song()["channels"]["lead"]["gain"], fader.value, self.mixer.busy), (0.9, 0.9, False))
        # It is one step, called as the knob of the bar calls it, and taken back the fader shows that.
        self.assertEqual(self.session.undo_label, "lead: volume 0.9")
        self.assertTrue(self.session.undo())
        self.mixer.update()
        self.assertEqual((self.data(), fader.value, self.lead.number.text, self.session.can_undo),
                         (before, 0.8, "0.8", False))

    def test_a_level_put_back_to_what_the_engine_takes_without_it_is_taken_out_of_the_file(self) -> None:
        click(self.ui, middle(self.lead.fader.rect), button=3)
        self.assertEqual((self.song()["channels"]["lead"], self.session.undo_label, self.lead.fader.value),
                         ({"inst": "tone"}, "lead: volume 1", 1))
        # A channel that said none gets one when its fader is moved, and loses it when the fader is put back.
        self.drag(self.low.fader, -20)
        self.assertEqual((self.song()["channels"]["low"], self.session.undo_label),
                         ({"inst": "buzz", "gain": 0.8}, "low: volume 0.8"))
        self.drag(self.low.fader, 20)
        self.assertEqual((self.song()["channels"]["low"], self.session.undo_label, self.low.number.text),
                         ({"inst": "buzz"}, "low: volume 1", "1"))

    def test_the_fader_of_the_song_saves_the_level_of_the_song_which_is_not_written_at_what_the_engine_takes(
            self) -> None:
        mixer, before = self.mixer, self.data()
        # Two pixels are a hundredth: twenty up are a tenth.
        mouse = self.drag(mixer.fader, 20, let_go=False)
        self.assertEqual((mixer.number.text, mixer.busy, self.data()), ("0.5", True, before))
        self.ui.handle(Event("up", mouse, button=1))
        self.assertEqual(differing(before, self.data()), ["songs.walk.gain"])
        self.assertEqual((self.song()["gain"], mixer.fader.value, self.session.undo_label),
                         (0.5, 0.5, "walk: volume 0.5"))
        click(self.ui, middle(mixer.fader.rect), button=3)
        self.assertEqual(("gain" in self.song(), mixer.fader.value, mixer.number.text, self.session.undo_label),
                         (False, 0.35, "0.35", "walk: volume 0.35"))
        self.outside(lambda song: song.update(gain=0.2))
        mixer.update()
        self.assertEqual((mixer.fader.value, mixer.number.text), (0.2, "0.2"))

    def test_a_change_from_outside_is_shown_and_a_fader_that_is_dragged_waits_for_the_hand_to_let_go(self) -> None:
        mixer = self.mixer
        self.outside(lambda song: song["channels"]["lead"].update(gain=0.5))
        mixer.update()
        self.assertEqual((self.lead.fader.value, self.lead.number.text), (0.5, "0.5"))
        mouse = self.drag(self.kit.fader, 10, let_go=False)

        def more(song: dict) -> None:
            """Give two channels and the song other levels."""
            song["channels"]["lead"]["gain"], song["channels"]["kit"]["gain"], song["gain"] = 0.25, 0.3, 0.2

        self.outside(more)
        mixer.update()
        # Shown its song again, as a song panel does when the song was read, the mixer leaves the fader to the hand.
        mixer.show("walk", "low")
        self.assertEqual((self.lead.fader.value, mixer.fader.value, self.kit.fader.value, self.kit.number.text,
                          mixer.busy), (0.25, 0.2, 0.7, "0.7", True))
        # The release saves the level beside what was written outside, and the mixer then shows what the file holds.
        self.ui.handle(Event("up", mouse, button=1))
        self.assertEqual(([self.song()["channels"][name].get("gain") for name in ORDER], self.song()["gain"]),
                         ([0.25, None, 0.5, 0.7], 0.2))
        self.assertEqual((self.kit.fader.value, self.session.undo_label, self.session.message),
                         (0.7, "kit: volume 0.7", ""))

    def test_a_fader_whose_strip_another_channel_takes_is_let_go_of_and_nothing_is_saved(self) -> None:
        def gone(song: dict) -> None:
            """Take a channel out of the song."""
            del song["channels"]["low"], song["patterns"]["a"]["low"]

        mouse = self.drag(self.low.fader, 10, let_go=False)
        self.outside(gone)
        written = self.data()
        self.mixer.update()
        # The channel after it has the strip now, at its own level, and the first channel is where it was.
        self.assertEqual(([strip.channel for strip in self.mixer.strips], self.low.fader.value, self.mixer.busy),
                         (["lead", "a_long_name", "kit", None], 0.5, False))
        self.ui.handle(Event("up", mouse, button=1))
        self.assertEqual((self.data(), self.session.undo_label), (written, "changed outside: walk"))

    def test_the_wheel_on_a_fader_is_its_level_which_is_saved_when_the_run_of_the_wheel_has_ended(self) -> None:
        fader = self.lead.fader
        for _ in range(3):
            self.assertTrue(self.ui.handle(Event("wheel", middle(fader.rect), wheel=1)))
        self.mixer.update()
        self.assertEqual((self.lead.number.text, self.mixer.busy, self.session.can_undo, self.mixer.first),
                         ("0.83", True, False, 0))
        self.now += dials.WHEEL_WAIT
        self.mixer.update()
        self.assertEqual((self.mixer.busy, self.song()["channels"]["lead"]["gain"], self.session.undo_label),
                         (False, 0.83, "lead: volume 0.83"))
        # The fader says what the left press, the right press and the wheel do on it, in the word for how loud.
        self.ui.handle(Event("move", middle(fader.rect)))
        self.assertEqual(self.ui.help, "press: drag up or down for the volume of the channel, slower with shift   "
                                       "right press: back to 1   wheel: steps it")


class Scrolled(Opened):
    def test_the_wheel_moves_the_view_only_when_the_strips_do_not_fit(self) -> None:
        mixer, desk = self.mixer, self.mixer.desk
        over = middle(self.lead.number.rect)
        self.assertFalse(self.ui.handle(Event("wheel", over, wheel=-1)))
        self.assertEqual((mixer.first, self.seen(), desk.bar.visible, desk.bar.thumb()), (0, ORDER, False, None))
        # In a room for two strips beside the song's, a step of the wheel is two strips.
        mixer.place(NARROW)
        self.assertEqual((mixer.laid.most, self.seen(), desk.bar.visible, desk.bar.rect),
                         (3, ["lead", "low"], True, (131, 365, 64, 4)))
        self.assertEqual((desk.bar.thumb(), mixer.number.visible, mixer.fader.rect[3]), ((131, 365, 16, 4), True, 201))
        seen = []
        for wheel in (-1, -1, -1, 1, 5):
            self.assertTrue(self.ui.handle(Event("wheel", over, wheel=wheel)))
            seen.append((mixer.first, self.seen(), desk.bar.thumb()[0]))
        # The third channel has no room for the drum channel after it, which stands apart.
        self.assertEqual(seen, [(2, ["a_long_name"], 163), (3, ["kit"], 179), (3, ["kit"], 179),
                                (1, ["low", "a_long_name"], 147), (0, ["lead", "low"], 131)])
        # A strip out of view has no widget on the window, and one in view has all of its own.
        self.assertEqual([widget.visible for strip in (self.lead, self.kit) for widget in strip.parts().values()],
                         [True] * 6 + [False] * 6)
        # The mixer goes one way, so with shift the wheel moves along the strips as well. With ctrl it is not used.
        self.assertTrue(self.ui.handle(Event("wheel", over, wheel=-1, mods=frozenset({"shift"}))))
        self.assertFalse(self.ui.handle(Event("wheel", over, wheel=-1, mods=frozenset({"ctrl"}))))
        self.assertEqual(mixer.first, 2)
        self.ui.handle(Event("wheel", over, wheel=1))
        # The wheel on a fader sets its volume in a mixer that scrolls too, and moves no view.
        for fader in (self.lead.fader, mixer.fader):
            self.assertTrue(self.ui.handle(Event("wheel", middle(fader.rect), wheel=-1)))
        self.assertEqual((mixer.first, mixer.busy, self.lead.fader.value, mixer.fader.value), (0, True, 0.79, 0.39))
        self.now += dials.WHEEL_WAIT
        mixer.update()
        self.assertEqual((mixer.busy, self.song()["channels"]["lead"]["gain"], self.song()["gain"]),
                         (False, 0.79, 0.39))
        # The song that is shown keeps its view, and another starts at its first strip, scrolling or not.
        self.ui.handle(Event("wheel", over, wheel=-1))
        mixer.show("walk", "kit")
        self.assertEqual(mixer.first, 2)
        mixer.show("trio")
        self.assertEqual((mixer.first, mixer.laid.most, self.seen()), (0, 1, ["one", "two"]))
        mixer.show("plain")
        self.assertEqual((mixer.first, self.seen(), desk.bar.visible), (0, ["air"], False))

    def test_the_scroll_bar_under_the_strips_is_dragged_by_its_thumb_and_pressed_beside_it_and_says_its_line(
            self) -> None:
        mixer, bar = self.mixer, self.mixer.desk.bar
        self.ui.handle(Event("move", middle(self.lead.number.rect)))
        self.assertEqual((self.ui.help, DESK_HELP),
                         (DESK_HELP, "a strip for each channel, and at the right the whole song   wheel: along the "
                                     "strips"))
        mixer.place(NARROW)
        x, y, w, _ = bar.thumb()
        self.ui.handle(Event("move", (x + 2, y + 1)))
        self.assertEqual((self.ui.help, BAR_HELP),
                         (BAR_HELP, "press: drag the bar to go along the strips, or press beside it to jump   wheel: "
                                    "along the strips"))
        # The thumb has 48 pixels of way for the three strips that are out of view: 16 pixels a strip.
        self.ui.handle(Event("down", (x + 2, y + 1), button=1))
        seen = []
        for moved in (16, 30, 48, 100, -40):
            self.ui.handle(Event("move", (x + 2 + moved, y + 30)))
            seen.append((mixer.first, bar.busy, self.seen()))
        self.assertEqual(seen, [(1, True, ["low", "a_long_name"]), (2, True, ["a_long_name"]), (3, True, ["kit"]),
                                (3, True, ["kit"]), (0, True, ["lead", "low"])])
        self.ui.handle(Event("up", (x + 2, y + 30), button=1))
        self.ui.handle(Event("move", (x + 40, y + 1)))
        self.assertEqual((mixer.first, bar.busy), (0, False))
        # A press beside the thumb moves the view by what is in view, which is one strip here.
        click(self.ui, (x + w + 20, y + 1))
        self.assertEqual((mixer.first, bar.thumb()[0]), (1, x + 16))
        click(self.ui, (x + 2, y + 1))
        self.assertEqual((mixer.first, bar.thumb()[0], self.data(), self.session.can_undo), (0, x, PROJECT, False))


if __name__ == "__main__":
    unittest.main()
