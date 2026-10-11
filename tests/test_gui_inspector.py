"""Tests for the right panel of the GUI: the numbers of the picked thing in words, and the findings of the checks."""
from __future__ import annotations

import math
import os
import unittest

# No test opens a window that can be seen, or a sound device. This is said before pygame is first imported.
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from rowsynth.checks import Finding, songs, sounds
from rowsynth.errors import RowSynthError
from rowsynth.gui import layout, sprites, theme
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.font import Font
from rowsynth.gui.inspector import (AIR, ICON, LABELS, PAD, RULE_AIR, Block, Inspector, arrange, belongs, faulty,
                                    lay, where, worded)
from rowsynth.gui.ui import Event, Ui

try:
    import pygame
except ImportError:
    pygame = None

NEEDS = "pygame is not installed"
#: The right panel on the art surface the layout is drawn for, and the same panel in a lower window.
RIGHT, LOW = (786, 22, 174, 504), (786, 22, 174, 300)
#: The numbers of a sound that is as it should be.
COIN = {"seconds": 0.383, "peak_db": -3.0, "rms_db": -11.3, "dc": 0.0, "clipped": 0, "head_step": 0.0,
        "tail_step": 0.0, "above_4k": 0.06, "centroid_hz": 1846}
#: A finding in a sound, one in a variant of another sound, and one in a pattern of a song.
CLICK = Finding("ends", "sounds.coin", "ends with a jump of 0.412 into silence, and 0.01 or more clicks")
HARSH = Finding("bright", "sounds.zaps[1]", "0.41 of its energy is above 4 kHz, which is more than 0.3")
SHORT = Finding("rows", "songs.tune.patterns.a",
                "channel 'bass' has 28 rows, and the longest channel of the pattern has 32")
#: A colour the theme does not have, for a ground on which everything that is drawn shows.
GROUND = (1, 2, 3)


class Held:
    """A stand-in for the model: what is picked, and the findings the test sets. None is for checks that run."""

    def __init__(self) -> None:
        self.picked: tuple[str, str] | None = ("sound", "coin")
        self.found: list[Finding] | None = None

    def findings(self) -> list[Finding]:
        """The findings, in a list made anew each time."""
        return list(self.found)


class Stand:
    """A stand-in for the session: it tells what the test has set, and keeps what was picked through it."""

    def __init__(self) -> None:
        self.model = Held()
        self.told: dict | None = None
        self.asked: list[tuple[str, str]] = []

    def numbers(self) -> dict | None:
        return self.told

    def finding_count(self) -> int | None:
        return None if self.model.found is None else len(self.model.found)

    def pick(self, kind: str, name: str) -> None:
        self.asked.append((kind, name))
        self.model.picked = (kind, name)


def made(found: list[Finding] | None = None, rect: layout.Rect = RIGHT) -> tuple[Stand, Ui, Inspector]:
    """A stand-in session with findings, a Ui, and the panel made with the two, placed and brought up to date."""
    stand, ui = Stand(), Ui()
    stand.model.found = found
    panel = Inspector(stand, ui)
    panel.place(rect)
    panel.update()
    return stand, ui, panel


def spot(panel: Inspector, block: Block) -> tuple[int, int]:
    """A pixel on the first line of a block, as the findings are scrolled now."""
    x, y, _, _ = panel.list.rect
    return x + PAD, y + block.top - panel.list.scroll.offset


def click(ui: Ui, pos: tuple[int, int]) -> None:
    """Press the left button at a position and let it go there."""
    ui.handle(Event("down", pos, button=1))
    ui.handle(Event("up", pos, button=1))


def lies_in(rect: layout.Rect, bounds: layout.Rect) -> bool:
    """Whether a rectangle has no pixel outside another, and neither a width nor a height below 0."""
    x, y, w, h = rect
    return (w >= 0 and h >= 0 and bounds[0] <= x and x + w <= bounds[0] + bounds[2]
            and bounds[1] <= y and y + h <= bounds[1] + bounds[3])


def ink(surface: pygame.Surface, rect: layout.Rect, colour: theme.Colour) -> int:
    """How many pixels of a rectangle are in a colour."""
    x, y, w, h = rect
    return sum(surface.get_at((x + column, y + row))[:3] == colour for column in range(w) for row in range(h))


class Wording(unittest.TestCase):
    def test_each_number_is_said_in_its_unit(self) -> None:
        told = dict(COIN, seconds=0.42, rms_db=-14.2, dc=-0.012, head_step=0.005)
        self.assertEqual([worded(key, told[key]) for key in LABELS],
                         ["0.42 s", "-3.0 dB", "-14.2 dB", "-0.0120", "no", "0.005", "none", "6%", "1846 Hz"])
        self.assertEqual([worded("seconds", 12.8), worded("dc", 0.0031), worded("above_4k", 0.29)],
                         ["12.80 s", "+0.0031", "29%"])

    def test_zero_is_none_of_an_offset_and_of_a_step_and_no_of_clipping(self) -> None:
        self.assertEqual([worded(key, 0) for key in ("dc", "head_step", "tail_step", "clipped")],
                         ["none", "none", "none", "no"])
        self.assertEqual([worded(key, 0.0) for key in ("seconds", "peak_db", "above_4k")], ["0.00 s", "0.0 dB", "0%"])
        self.assertEqual(worded("centroid_hz", 0), "0 Hz")

    def test_a_level_of_minus_infinity_is_silent(self) -> None:
        self.assertEqual([worded("peak_db", -math.inf), worded("rms_db", float("-inf"))], ["silent", "silent"])

    def test_clipping_is_said_as_its_count_of_samples_and_a_large_step_as_its_size(self) -> None:
        self.assertEqual([worded("clipped", 1), worded("clipped", 12)], ["1 sample", "12 samples"])
        self.assertEqual([worded("head_step", 0.412), worded("tail_step", 1.0)], ["0.412", "1.000"])

    def test_a_key_that_is_none_of_the_numbers_is_refused(self) -> None:
        with self.assertRaises(RowSynthError) as refused:
            worded("tempo", 150)
        self.assertEqual(str(refused.exception), "a render has no number called 'tempo'")


class Faults(unittest.TestCase):
    def test_a_sound_is_at_fault_from_the_limits_the_checks_hold_a_sound_to(self) -> None:
        for key in ("head_step", "tail_step"):
            self.assertEqual([faulty("sound", key, sounds._STEP), faulty("sound", key, sounds._STEP - 0.001)],
                             [True, False], key)
        self.assertEqual([faulty("sound", "dc", value) for value in (sounds._DC, -sounds._DC, sounds._DC - 0.0001)],
                         [True, True, False])
        # A share above 4 kHz that is at the limit is not over it.
        self.assertEqual([faulty("sound", "above_4k", value) for value in (sounds._BRIGHT, sounds._BRIGHT + 0.01)],
                         [False, True])

    def test_a_song_is_at_fault_from_the_level_the_checks_hold_a_mix_to(self) -> None:
        self.assertEqual([faulty("song", "peak_db", value) for value in (songs._PEAK_DB, 0.0, songs._PEAK_DB - 0.1,
                                                                         -math.inf)],
                         [True, True, False, False])
        self.assertEqual([faulty("song", "clipped", 1), faulty("song", "clipped", 0)], [True, False])

    def test_what_the_checks_have_no_limit_for_is_never_at_fault(self) -> None:
        # A song goes round, a sound is brought to one peak, and an instrument is held to nothing.
        plain = [("song", "head_step", 0.5), ("song", "tail_step", 0.5), ("song", "dc", 0.1),
                 ("song", "above_4k", 0.9), ("sound", "peak_db", 0.0), ("sound", "clipped", 5),
                 ("sound", "seconds", 99.0), ("sound", "centroid_hz", 9000), ("instrument", "tail_step", 0.5),
                 ("instrument", "above_4k", 0.9), ("", "tail_step", 0.5)]
        self.assertEqual([case for case in plain if faulty(*case)], [])


class Places(unittest.TestCase):
    def test_a_target_belongs_to_the_song_the_sound_or_the_instrument_it_is_in(self) -> None:
        self.assertEqual([belongs(target) for target in ("sounds.coin", "sounds.zaps[1]", "songs.tune.patterns.a.lead",
                                                         "instruments.lead")],
                         [("sound", "coin"), ("sound", "zaps"), ("song", "tune"), ("instrument", "lead")])
        self.assertEqual([belongs(target) for target in ("chords.Am", "sounds", "sounds[1]", "", "sounds..coin")],
                         [None] * 5)

    def test_a_place_is_said_as_the_name_of_its_thing_and_the_steps_into_it_counted_from_1(self) -> None:
        said = {"sounds.coin": "coin", "sounds.zaps[1]": "zaps, variant 2", "sounds.boom.parts[0]": "boom, part 1",
                "sounds.hum.loop": "hum, loop", "songs.tune": "tune", "songs.tune.patterns.a": "tune, pattern a",
                "songs.tune.patterns.a.lead": "tune, pattern a, lead",
                "sounds.zaps[1].notes": "zaps, variant 2, notes",
                # A pattern can be called what a step is called, and so can a channel.
                "songs.tune.patterns.parts.patterns": "tune, pattern parts, patterns",
                "chords.Am": "chords.Am"}
        self.assertEqual({target: where(target) for target in said}, said)


class Laying(unittest.TestCase):
    def test_the_findings_of_the_picked_thing_come_first_and_the_rest_under_a_line(self) -> None:
        blocks, rule, height = lay([HARSH, CLICK, SHORT, CLICK], ("sound", "coin"), 162)
        self.assertEqual([block.thing for block in blocks],
                         [("sound", "coin"), ("sound", "coin"), ("sound", "zaps"), ("song", "tune")])
        message = ("ends with a jump of 0.412", "into silence, and 0.01 or", "more clicks")
        self.assertEqual(blocks[0], Block(AIR, "warn", ("coin",), message, ("sound", "coin")))
        self.assertEqual((blocks[2].head, blocks[3].head), (("zaps, variant 2",), ("tune, pattern a",)))
        # A block is its lines with air above and below, and the line has air of its own on each side.
        self.assertEqual(blocks[0].bottom, AIR + 4 * Font.LINE)
        self.assertEqual(blocks[1].top, blocks[0].bottom + 2 * AIR)
        self.assertEqual(rule, blocks[1].bottom + AIR + RULE_AIR)
        self.assertEqual(blocks[2].top, rule + 1 + RULE_AIR + AIR)
        self.assertEqual(height, blocks[3].bottom + AIR)
        # The place stands beside the icon, and is broken to what the icon leaves it.
        narrow = lay([SHORT], ("song", "tune"), 60)
        self.assertEqual((narrow.blocks[0].head, narrow.rule), (("tune,", "pattern", "a"), None))
        self.assertTrue(all(Font.width(line) <= 60 - ICON - theme.GAP for line in narrow.blocks[0].head))
        # With nothing picked every finding is one of the rest, under the line that says the picked thing has none.
        blocks, rule, _ = lay([CLICK], None, 162)
        self.assertEqual([(block.icon, block.head, block.thing) for block in blocks],
                         [("tick", ("nothing found",), None), ("warn", ("coin",), ("sound", "coin"))])
        self.assertEqual(rule, blocks[0].bottom + AIR + RULE_AIR)


class Shown(unittest.TestCase):
    def test_the_values_are_dim_dots_until_the_render_is_ready_and_then_the_numbers_in_words(self) -> None:
        stand, _, panel = made()
        self.assertEqual(panel.values, [(label, "...", theme.DIM) for label in LABELS.values()])
        stand.told = dict(COIN, tail_step=0.412)
        panel.update()
        fine = theme.TEXT
        self.assertEqual(panel.values, [
            ("LENGTH", "0.38 s", fine), ("PEAK", "-3.0 dB", fine), ("LOUDNESS", "-11.3 dB", fine),
            ("OFFSET", "none", fine), ("CLIPPED", "no", fine), ("START", "none", fine), ("END", "0.412", theme.FAULT),
            ("HIGHS", "6%", fine), ("BRIGHTNESS", "1846 Hz", fine)])
        # The same numbers are no fault in a song, which the checks hold to other rules.
        stand.model.picked = ("song", "tune")
        panel.update()
        self.assertEqual(panel.values[6], ("END", "0.412", fine))
        stand.model.picked, stand.told = None, None
        panel.update()
        self.assertEqual(panel.values[6], ("END", "...", theme.DIM))

    def test_the_findings_of_the_picked_thing_come_first_whatever_is_picked(self) -> None:
        stand, _, panel = made([CLICK, HARSH, SHORT])
        self.assertEqual([block.thing for block in panel.list.blocks],
                         [("sound", "coin"), ("sound", "zaps"), ("song", "tune")])
        stand.model.picked = ("song", "tune")
        panel.update()
        self.assertEqual([block.thing for block in panel.list.blocks],
                         [("song", "tune"), ("sound", "coin"), ("sound", "zaps")])
        # A thing with no finding of its own has the line that says so before the rest.
        stand.model.picked = ("instrument", "lead")
        panel.update()
        self.assertEqual([(block.icon, block.thing) for block in panel.list.blocks],
                         [("tick", None), ("warn", ("sound", "coin")), ("warn", ("sound", "zaps")),
                          ("warn", ("song", "tune"))])

    def test_with_nothing_found_there_is_the_one_line_that_says_so_and_none_while_the_checks_run(self) -> None:
        stand, _, panel = made()
        self.assertEqual((panel.list.found, panel.list.blocks), (None, ()))
        stand.model.found = []
        panel.update()
        self.assertEqual(panel.list.blocks, (Block(AIR, "tick", ("nothing found",), (), None),))
        # The file changed outside, and the checks run again.
        stand.model.found = None
        panel.update()
        self.assertEqual((panel.list.found, panel.list.blocks), (None, ()))

    def test_a_click_on_a_finding_picks_the_thing_it_is_in(self) -> None:
        stand, ui, panel = made([CLICK, HARSH, SHORT])
        click(ui, spot(panel, panel.list.blocks[2]))
        self.assertEqual(stand.asked, [("song", "tune")])
        panel.update()
        self.assertEqual(panel.list.blocks[0].thing, ("song", "tune"))
        # The last line of a finding is as good as its first, and the right button picks nothing.
        last = panel.list.blocks[2]
        x, y = spot(panel, last)
        ui.handle(Event("down", (x, y), button=3))
        click(ui, (x + 100, y + last.bottom - last.top - 1))
        self.assertEqual(stand.asked, [("song", "tune"), ("sound", "zaps")])
        # The line that says nothing was found picks nothing, and neither does the room under the findings.
        stand.model.found = []
        panel.update()
        click(ui, spot(panel, panel.list.blocks[0]))
        click(ui, (RIGHT[0] + 50, RIGHT[1] + RIGHT[3] - 1))
        self.assertEqual(len(stand.asked), 2)

    def test_the_wheel_scrolls_the_findings_only_when_they_do_not_fit(self) -> None:
        _, ui, panel = made([CLICK, HARSH, SHORT])
        over = (RIGHT[0] + 50, panel.list.rect[1] + 50)
        ui.handle(Event("wheel", over, wheel=-1))
        self.assertEqual((panel.list.scroll.most, panel.list.scroll.offset), (0, 0))
        stand, ui, panel = made([CLICK, HARSH, SHORT] * 10)
        scroll = panel.list.scroll
        ui.handle(Event("wheel", over, wheel=-1))
        self.assertEqual(scroll.offset, theme.LIST_WHEEL * Font.LINE)
        self.assertEqual(spot(panel, panel.list.blocks[0])[1], panel.list.rect[1] + AIR - scroll.offset)
        ui.handle(Event("wheel", over, wheel=1))
        self.assertEqual(scroll.offset, 0)
        ui.handle(Event("wheel", over, wheel=-1000))
        self.assertEqual((scroll.offset, scroll.most > 0), (scroll.most, True))
        # The wheel over the numbers is not the findings' to use.
        self.assertFalse(ui.handle(Event("wheel", (RIGHT[0] + 50, RIGHT[1] + 50), wheel=1)))
        # Another pick lays the findings out anew, from their top.
        stand.model.picked = ("song", "tune")
        panel.update()
        self.assertEqual((panel.list.blocks[0].thing, scroll.offset), (("song", "tune"), 0))

    def test_placed_at_two_heights_the_numbers_keep_their_room_and_the_findings_have_the_rest(self) -> None:
        _, ui, panel = made([CLICK, HARSH, SHORT] * 10)
        tall, low = arrange(RIGHT), arrange(LOW)
        self.assertEqual(tall, {"measured": (792, 26, 162, 11), "table": (792, 40, 162, 99),
                                "rule": (786, 147, 174, 1), "found": (792, 151, 162, 11),
                                "list": (786, 165, 174, 361)})
        self.assertEqual(low, dict(tall, list=(786, 165, 174, 157)))
        for rect in (RIGHT, LOW, (5, 5, 40, 30), (0, 0, 3, 2)):
            panel.place(rect)
            self.assertEqual((panel.rect, ui.widgets), (rect, [panel.list]))
            self.assertEqual(panel.list.rect, arrange(rect)["list"])
            self.assertTrue(all(lies_in(place, rect) for place in arrange(rect).values()), rect)


@unittest.skipUnless(pygame, NEEDS)
class Drawn(unittest.TestCase):
    def frame(self, ui: Ui, panel: Inspector, size: tuple[int, int] = (200, 300)) -> pygame.Surface:
        """A surface of one colour on which the panel has been brought up to date and drawn, and then its widgets."""
        surface = pygame.Surface(size)
        surface.fill(GROUND)
        canvas = Canvas(surface, Font(sprites.load("font")), sprites.load("icons"))
        panel.update()
        panel.draw(canvas)
        ui.draw(canvas)
        return surface

    def test_a_value_the_checks_would_find_fault_with_is_in_the_colour_of_what_is_wrong(self) -> None:
        rect = (10, 10, 174, 280)
        stand, ui, panel = made([], rect)
        stand.told = dict(COIN, tail_step=0.412)
        surface = self.frame(ui, panel)
        x, y, w, _ = arrange(rect)["table"]
        rows = [(x, y + row * Font.LINE, w, Font.LINE) for row in range(len(LABELS))]
        self.assertEqual([ink(surface, row, theme.FAULT) > 0 for row in rows], [key == "tail_step" for key in LABELS])
        # The value stands at the right end of its line and its label, dim, at the left.
        value = Font.width("0.412")
        self.assertEqual(ink(surface, (x, rows[6][1], w - value, Font.LINE), theme.FAULT), 0)
        self.assertGreater(ink(surface, (x, rows[6][1], Font.width("END"), Font.LINE), theme.DIM), 0)
        self.assertEqual([ink(surface, row, theme.TEXT) > 0 for row in rows], [key != "tail_step" for key in LABELS])
        # The two headings are dim, and the line between the two halves runs the width of the panel.
        places = arrange(rect)
        self.assertEqual(ink(surface, places["rule"], theme.LINE), rect[2])
        self.assertTrue(all(ink(surface, places[part], theme.DIM) > 0 for part in ("measured", "found")))

    def test_nothing_found_is_a_tick_in_the_colour_of_what_is_as_it_should_be_and_running_checks_are_dim_dots(
            self) -> None:
        rect = (10, 10, 174, 280)
        stand, ui, panel = made([], rect)
        surface = self.frame(ui, panel)
        view = arrange(rect)["list"]
        first = (view[0] + PAD, view[1] + AIR, ICON, Font.LINE)
        self.assertGreater(ink(surface, first, theme.GOOD), 0)
        # After the tick come the words, and nothing in the list is in another colour.
        words = (first[0] + ICON + theme.GAP, first[1], Font.width("nothing found"), Font.LINE)
        self.assertGreater(ink(surface, words, theme.GOOD), 0)
        self.assertEqual(ink(surface, view, theme.GOOD) + ink(surface, view, GROUND), view[2] * view[3])
        stand.model.found = None
        surface = self.frame(ui, panel)
        self.assertGreater(ink(surface, (first[0], first[1], Font.width("..."), Font.LINE), theme.DIM), 0)
        self.assertEqual(ink(surface, view, theme.DIM) + ink(surface, view, GROUND), view[2] * view[3])
        # A finding has the warning sign before its place, and its place is in the colour of its thing.
        stand.model.found = [CLICK]
        surface = self.frame(ui, panel)
        self.assertGreater(ink(surface, first, theme.FAULT), 0)
        self.assertEqual((ink(surface, view, theme.SOUND) > 0, ink(surface, view, theme.GOOD)), (True, 0))

    def test_nothing_is_drawn_outside_the_panel_with_many_findings_scrolled_part_of_the_way(self) -> None:
        size = (220, 330)
        for rect in ((20, 30, 174, 280), (20, 30, 174, 60), (20, 30, 40, 280)):
            stand, ui, panel = made([CLICK, HARSH, SHORT] * 10, rect)
            stand.told = dict(COIN, tail_step=0.412)
            panel.update()
            view = panel.list.rect
            # The mouse rests on a finding, which is drawn on a lit ground.
            over = (view[0] + view[2] // 2, view[1] + view[3] // 2)
            ui.handle(Event("wheel", over, wheel=-1))
            ui.handle(Event("move", over))
            surface = self.frame(ui, panel, size)
            x, y, w, h = rect
            outside = sum(surface.get_at((column, row))[:3] != GROUND for column in range(size[0])
                          for row in range(size[1]) if not (x <= column < x + w and y <= row < y + h))
            self.assertEqual(outside, 0, rect)
            if view[3]:
                self.assertEqual(panel.list.scroll.offset, theme.LIST_WHEEL * Font.LINE)
                self.assertGreater(ink(surface, view, theme.RAISED), 0)
                # The lit ground stays clear of the panel's line, and nothing of a finding is above the list.
                self.assertEqual(ink(surface, (x, view[1], 1, view[3]), theme.RAISED), 0)
                self.assertEqual(ink(surface, (x, y, w, view[1] - y), theme.RAISED), 0)


if __name__ == "__main__":
    unittest.main()
