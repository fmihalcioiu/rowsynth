"""Tests for the shell of the GUI: the top bar, the left panel, and the app that joins the panels to a session."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from collections.abc import Hashable
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

# No test opens a window that can be seen, or a sound device. This is said before pygame is first imported.
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from rowsynth import checks
from rowsynth.errors import RowSynthError
from rowsynth.gui import inspector, layout, popups, run, soundpanel, sprites, start, theme
from rowsynth.gui import model as gui_model
from rowsynth.gui import projectedit
from rowsynth.gui.browser import Browser, plus_rect, rows_of
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.dials import bars
from rowsynth.gui.font import Font
from rowsynth.gui.lists import List, Row
from rowsynth.gui.panels import within
from rowsynth.gui.session import Session
from rowsynth.gui.topbar import (BUTTON_HELP, CLOSE, MAXIMISE, MINIMISE, REFUSED, RESTORE, TITLE_HELP, Title, TopBar,
                                 act, arrange, moment, timing, title_parts)
from rowsynth.gui.ui import Event, Ui

try:
    import pygame
except ImportError:
    pygame = None

if TYPE_CHECKING:
    from rowsynth.gui.app import App

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
NEEDS = "pygame is not installed"
#: The top bar and the left panel on the art surface the layout is drawn for, and the two in a far smaller window.
BAR, SIDE = (0, 0, 960, 22), (0, 22, 124, 504)
SMALL_BAR, SMALL_SIDE = (40, 10, 280, 22), (40, 32, 60, 30)
#: The level, from 0 to 1, that is 24 decibels under the top: half the way along a meter.
HALF = 10 ** (-24 / 20)
#: What stands in the bar from its left end, up to the last of its buttons.
ROW = ("file", "undo", "redo", "play", "stop", "loop", "pattern", "dry")
#: The air between each of those and the one after it: the FILE button stands alone, the rest in two groups.
GAPS = [theme.GROUP_GAP, theme.BUTTON_GAP, theme.GROUP_GAP, *[theme.BUTTON_GAP] * 4]
#: The window's own title row on the art surface the layout is drawn for.
TITLE = (0, 0, 960, 14)
#: A small project: one instrument, and one song of one bar on one channel.
SMALL = {
    "instruments": {"beep": {"vol": [15, 12, 9, 6]}},
    "songs": {"tune": {"row_ticks": 3, "channels": {"tune": {"inst": "beep"}},
                       "patterns": {"a": {"tune": "l4 c e g > c"}}, "order": ["a"]}},
}
#: A project whose one instrument is named by a sound, by the channel of a song, and by a switch in its text.
THINGS = {
    "instruments": {"beep": {"vol": [15, 12, 9, 6]}},
    "sounds": {"zap": {"def": {"voices": [{"inst": "beep", "notes": [[0, "a4", 4]]}]}}},
    "songs": {"tune": {"row_ticks": 3, "channels": {"tune": {"inst": "beep"}},
                       "patterns": {"a": {"tune": "l4 c e @beep g > c"}}, "order": ["a"]}},
}


class Held:
    """A stand-in for the model: the path of a file, what is picked, and a listing the test changes."""

    def __init__(self) -> None:
        self.path = Path("songs") / "arcade.json"
        self.picked: tuple[str, str] | None = ("song", "tune")
        self.holds = {"songs": [("tune", "")], "sounds": [("coin", ""), ("blip", "3")],
                      "instruments": [("lead", "")]}

    def listing(self) -> list[tuple[str, list[tuple[str, str]]]]:
        """What it holds, in lists made anew each time, as a model gives its listing."""
        return [(section, list(things)) for section, things in self.holds.items()]


class Stand:
    """A stand-in for the session: it keeps what it was asked, in order, and tells what the test has set."""

    def __init__(self) -> None:
        self.model = Held()
        self.asked: list[tuple] = []
        self.playing = False
        self.loop = True
        self.pattern = False
        self.dry = False
        self.seconds = 0.0
        self.at = 0.0
        self.loud = (0.0, 0.0)
        self.count: int | None = None
        self.findings_refused = False
        self.lit: dict[tuple[str, str], float] = {}
        self.can_undo = False
        self.can_redo = False
        self.undo_label = ""
        self.redo_label = ""

    def undo(self) -> bool:
        self.asked.append(("undo",))
        return True

    def redo(self) -> bool:
        self.asked.append(("redo",))
        return True

    def play(self) -> None:
        self.asked.append(("play",))

    def stop(self) -> None:
        self.asked.append(("stop",))

    def pick(self, kind: str, name: str) -> None:
        self.asked.append(("pick", kind, name))
        self.model.picked = (kind, name)

    def position(self) -> float:
        return self.at

    def level(self) -> tuple[float, float]:
        return self.loud

    def finding_count(self) -> int | None:
        return self.count

    def glows(self) -> dict[tuple[str, str], float]:
        return dict(self.lit)


def bar_on(rect: layout.Rect = BAR) -> tuple[Stand, Ui, TopBar]:
    """A stand-in session, a Ui, and the top bar made with the two and placed in a rectangle."""
    stand, ui = Stand(), Ui()
    bar = TopBar(stand, ui)
    bar.place(rect)
    return stand, ui, bar


def side_on(rect: layout.Rect = SIDE) -> tuple[Stand, Ui, Browser]:
    """A stand-in session, a Ui, and the left panel made with the two, placed in a rectangle and brought up to date."""
    stand, ui = Stand(), Ui()
    side = Browser(stand, ui)
    side.place(rect)
    side.update()
    return stand, ui, side


def click(ui: Ui, pos: tuple[int, int]) -> None:
    """Press the left button at a position and let it go there."""
    ui.handle(Event("down", pos, button=1))
    ui.handle(Event("up", pos, button=1))


def middle(rect: layout.Rect) -> tuple[int, int]:
    """The pixel in the middle of a rectangle."""
    return rect[0] + rect[2] // 2, rect[1] + rect[3] // 2


def spot(rows: List, key: Hashable) -> tuple[int, int]:
    """The top-left corner of the row of a key, as the list is scrolled now."""
    x, y, _, h = rows.rect
    return next((x, down) for down in range(y, y + h) if getattr(rows.at((x, down)), "key", None) == key)


def lies_in(rect: layout.Rect, bounds: layout.Rect) -> bool:
    """Whether a rectangle has no pixel outside another, and neither a width nor a height below 0."""
    x, y, w, h = rect
    return (w >= 0 and h >= 0 and bounds[0] <= x and x + w <= bounds[0] + bounds[2]
            and bounds[1] <= y and y + h <= bounds[1] + bounds[3])


def gaps(places: dict[str, layout.Rect]) -> list[int]:
    """The air between each thing of the bar's row and the one after it, from the left."""
    row = [places[thing] for thing in ROW]
    return [after[0] - before[0] - before[2] for before, after in zip(row, row[1:])]


def ink(surface: pygame.Surface, rect: layout.Rect, colour: theme.Colour) -> int:
    """How many pixels of a rectangle are in a colour."""
    x, y, w, h = rect
    return sum(surface.get_at((x + column, y + row))[:3] == colour for column in range(w) for row in range(h))


def settle() -> None:
    """Waits for every render and every run of the checks that is still on a thread of a model's."""
    for thread in threading.enumerate():
        if thread.name.startswith("rowsynth-"):
            thread.join()


class Wording(unittest.TestCase):
    def test_a_time_under_a_minute_is_its_seconds_and_tenths_after_no_minutes(self) -> None:
        self.assertEqual([moment(3.2), moment(12.8), moment(59.9)], ["0:03.2", "0:12.8", "0:59.9"])
        self.assertEqual(timing(3.2, 12.8), "0:03.2 / 0:12.8")

    def test_a_time_over_a_minute_counts_its_minutes(self) -> None:
        self.assertEqual([moment(60), moment(72.5), moment(600), moment(3725.1)],
                         ["1:00.0", "1:12.5", "10:00.0", "62:05.1"])

    def test_zero_is_all_noughts(self) -> None:
        self.assertEqual([moment(0), timing(0.0, 0.0)], ["0:00.0", "0:00.0 / 0:00.0"])

    def test_a_tenth_is_rounded_down(self) -> None:
        self.assertEqual([moment(0.09), moment(3.29), moment(59.99), moment(119.96)],
                         ["0:00.0", "0:03.2", "0:59.9", "1:59.9"])


class Rows(unittest.TestCase):
    def test_a_listing_is_a_heading_for_each_section_and_its_things_under_it_with_their_tags(self) -> None:
        listing = [("songs", [("tune", "")]), ("sounds", [("coin", ""), ("blip", "3"), ("boom", "parts")]),
                   ("instruments", [])]
        self.assertEqual(rows_of(listing), [
            Row("songs", "SONGS", head=True, colour=theme.SONG),
            Row(("song", "tune"), "tune"),
            # After the last thing of a section stands the row that makes one more.
            Row(("new", "song"), "new", plus=True),
            Row("sounds", "SOUNDS", head=True, colour=theme.SOUND),
            Row(("sound", "coin"), "coin"),
            Row(("sound", "blip"), "blip", tag="3"),
            Row(("sound", "boom"), "boom", tag="parts"),
            Row(("new", "sound"), "new", plus=True),
            # A section that holds nothing has its heading all the same, and its row that adds.
            Row("instruments", "INSTRUMENTS", head=True, colour=theme.INSTRUMENT),
            Row(("new", "instrument"), "new", plus=True),
        ])


class Cutting(unittest.TestCase):
    def test_a_rectangle_is_cut_to_what_of_it_lies_in_another_down_to_nothing(self) -> None:
        bounds = (10, 20, 30, 40)
        self.assertEqual([within(rect, bounds) for rect in ((12, 22, 5, 6), (0, 0, 15, 25), (35, 50, 20, 20))],
                         [(12, 22, 5, 6), (10, 20, 5, 5), (35, 50, 5, 10)])
        # One that is outside it, and one whose width is below 0.
        self.assertEqual([within((50, 70, 5, 5), bounds), within((12, 22, -4, 6), bounds)],
                         [(40, 60, 0, 0), (12, 22, 0, 6)])


class Top(unittest.TestCase):
    def test_play_stop_loop_and_the_two_switches_call_the_session_and_each_has_a_line_of_help(self) -> None:
        stand, ui, bar = bar_on()
        click(ui, middle(bar.play.rect))
        click(ui, middle(bar.stop.rect))
        self.assertEqual(stand.asked, [("play",), ("stop",)])
        self.assertTrue(bar.loop.is_on)
        click(ui, middle(bar.loop.rect))
        self.assertFalse(stand.loop)
        click(ui, middle(bar.loop.rect))
        self.assertTrue(stand.loop)
        # The two switches beside loop read as words, are off to start with, and each tells the session.
        self.assertEqual((bar.pattern.text, bar.dry.text, bar.pattern.is_on, bar.dry.is_on),
                         ("PATTERN", "DRY", False, False))
        click(ui, middle(bar.pattern.rect))
        self.assertEqual((stand.pattern, stand.dry), (True, False))
        click(ui, middle(bar.dry.rect))
        click(ui, middle(bar.pattern.rect))
        self.assertEqual((stand.pattern, stand.dry), (False, True))
        # What the session has is what they show, whoever set it.
        stand.pattern, stand.dry = True, False
        bar.update()
        self.assertEqual((bar.pattern.is_on, bar.dry.is_on), (True, False))
        lines = [widget.help for widget in (bar.file, bar.play, bar.stop, bar.loop, bar.pattern, bar.dry)]
        self.assertEqual((len(set(lines)), all(line.startswith("press: ") for line in lines)), (6, True))

    def test_undo_and_redo_are_enabled_only_while_the_session_has_a_step_for_them(self) -> None:
        stand, _, bar = bar_on()
        self.assertEqual((bar.undo.icon, bar.redo.icon), ("undo", "redo"))
        steps = [(False, False), (True, False), (False, True), (True, True)]
        shown = []
        for can in steps:
            stand.can_undo, stand.can_redo = can
            bar.update()
            shown.append((bar.undo.enabled, bar.redo.enabled))
        self.assertEqual(shown, steps)

    def test_a_click_on_undo_or_on_redo_calls_the_session(self) -> None:
        stand, ui, bar = bar_on()
        stand.can_undo = stand.can_redo = True
        bar.update()
        click(ui, middle(bar.undo.rect))
        click(ui, middle(bar.redo.rect))
        click(ui, middle(bar.redo.rect))
        self.assertEqual(stand.asked, [("undo",), ("redo",), ("redo",)])

    def test_a_click_on_undo_or_on_redo_while_it_is_not_enabled_calls_nothing(self) -> None:
        stand, ui, bar = bar_on()
        bar.update()
        click(ui, middle(bar.undo.rect))
        click(ui, middle(bar.redo.rect))
        self.assertEqual(stand.asked, [])
        # Each goes by its own step: with one to take back and none to make again, only undo takes a click.
        stand.can_undo = True
        bar.update()
        click(ui, middle(bar.redo.rect))
        click(ui, middle(bar.undo.rect))
        self.assertEqual(stand.asked, [("undo",)])

    def test_the_help_of_undo_and_of_redo_names_the_step_and_says_so_when_there_is_none(self) -> None:
        stand, ui, bar = bar_on()
        bar.update()
        self.assertEqual((bar.undo.help, bar.redo.help), ("nothing to undo", "nothing to redo"))
        stand.can_undo, stand.undo_label = True, "lead of pattern a: note placed"
        stand.can_redo, stand.redo_label = True, "changed outside: zap"
        bar.update()
        self.assertEqual((bar.undo.help, bar.redo.help),
                         ("press: undo lead of pattern a: note placed", "press: redo changed outside: zap"))
        # With the mouse on the button, that is the line there is to show.
        ui.handle(Event("move", middle(bar.redo.rect)))
        self.assertEqual(ui.help, "press: redo changed outside: zap")

    def test_in_a_wide_bar_undo_and_redo_stand_after_the_file_button_with_a_gap_before_play(self) -> None:
        places = arrange(BAR)
        # The FILE button stands at the left end, as wide as its word needs, and each switch as wide as its own.
        self.assertEqual([places[thing] for thing in ROW],
                         [(6, 3, 31, 16), (49, 3, 17, 16), (68, 3, 17, 16), (97, 3, 17, 16), (116, 3, 17, 16),
                          (135, 3, 17, 16), (154, 3, 49, 16), (205, 3, 25, 16)])
        self.assertTrue(all(lies_in(places[thing], BAR) for thing in ROW))
        # Undo and redo are a group as play, stop, loop and the switches are: a gap before it, and less inside it.
        self.assertEqual(gaps(places), GAPS)

    def test_in_a_narrow_bar_undo_and_redo_stay_as_the_buttons_of_play_do_and_in_their_order(self) -> None:
        # A bar with no meter, one with no time either, and one that has just the room for its buttons.
        for rect, gone in (((0, 0, 577, 22), ["meter"]), ((0, 0, 503, 22), ["time", "meter"]),
                           (SMALL_BAR, ["time", "meter"])):
            places = arrange(rect)
            self.assertEqual([thing for thing in ("time", "meter") if not places[thing][2]], gone, rect)
            self.assertEqual([places[thing][2:] for thing in ROW[1:6]], [theme.ICON_BUTTON] * 5, rect)
            self.assertTrue(all(lies_in(places[thing], rect) for thing in ROW), rect)
            self.assertEqual(gaps(places), GAPS, rect)
            # The last of the buttons ends a gap before the findings, or sooner.
            self.assertLessEqual(places["dry"][0] + places["dry"][2] + theme.GROUP_GAP, places["found"][0], rect)

    def test_with_a_real_session_the_undo_button_and_then_the_redo_button_leave_the_file_as_it_should_be(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        # No thread of the model's is left to run into the next test.
        self.addCleanup(settle)
        path = Path(folder.name) / "small.json"
        path.write_text(json.dumps(SMALL), encoding="utf-8", newline="\n")
        session, ui = Session(gui_model.Model(path, 8000)), Ui()
        bar = TopBar(session, ui)
        bar.place(BAR)

        def held() -> dict:
            """What the file holds now."""
            return json.loads(path.read_text(encoding="utf-8"))

        def buttons() -> tuple[bool, bool]:
            """Of the bar brought up to date: whether undo is enabled, and whether redo is."""
            bar.update()
            return bar.undo.enabled, bar.redo.enabled

        self.assertEqual(buttons(), (False, False))
        self.assertTrue(session.change("songs.tune.patterns.a.tune", "l2 c e", "tune of pattern a: note removed"))
        after = json.loads(json.dumps(SMALL))
        after["songs"]["tune"]["patterns"]["a"]["tune"] = "l2 c e"
        self.assertEqual((held(), buttons(), bar.undo.help),
                         (after, (True, False), "press: undo tune of pattern a: note removed"))
        click(ui, middle(bar.undo.rect))
        self.assertEqual((held(), buttons(), bar.redo.help),
                         (SMALL, (False, True), "press: redo tune of pattern a: note removed"))
        click(ui, middle(bar.redo.rect))
        self.assertEqual((held(), buttons()), (after, (True, False)))

    def test_play_is_lit_while_something_plays_and_the_time_is_told_as_the_session_has_it(self) -> None:
        stand, ui, bar = bar_on()
        bar.update()
        self.assertEqual((bar.play.is_on, bar.time.text), (False, "0:00.0 / 0:00.0"))
        stand.playing, stand.at, stand.seconds, stand.loop = True, 3.26, 12.8, False
        bar.update()
        self.assertEqual((bar.play.is_on, bar.loop.is_on, bar.time.text), (True, False, "0:03.2 / 0:12.8"))
        # A press of play that the session does not take leaves it unlit.
        stand.playing = False
        click(ui, middle(bar.play.rect))
        bar.update()
        self.assertFalse(bar.play.is_on)

    def test_the_findings_are_dots_while_the_checks_run_then_a_tick_and_0_or_the_sign_and_a_count(self) -> None:
        stand, _, bar = bar_on()
        shown = []
        for count in (None, 0, 3):
            stand.count = count
            bar.update()
            shown.append(bar.found)
        self.assertEqual(shown, [("", "...", theme.DIM), ("tick", "0", theme.GOOD), ("warn", "3", theme.FAULT)])

    def test_checks_that_could_not_run_are_the_sign_with_no_number_and_the_line_of_help_says_so(self) -> None:
        stand, ui, bar = bar_on()
        # The session counts no findings where the checks gave none.
        stand.count = 0
        ui.handle(Event("move", middle(arrange(BAR)["found"])))
        shown = []
        for refused in (False, True, False):
            stand.findings_refused = refused
            bar.update()
            shown.append((bar.found, ui.help))
        self.assertEqual(shown, [(("tick", "0", theme.GOOD), ""), (("warn", "", theme.FAULT), REFUSED),
                                 (("tick", "0", theme.GOOD), "")])
        self.assertEqual((ui.hot, REFUSED), (bar.sign, "the checks could not run on this project"))

    def test_the_meter_lies_with_a_bar_for_each_side_and_takes_their_levels_from_the_session(self) -> None:
        stand, _, bar = bar_on()
        self.assertEqual((bar.meter.upright, bar.meter.levels, bar.meter.tops), (False, (0.0, 0.0), [0.0, 0.0]))
        for loud in ((0.5, 0.25), (0.0, 0.0)):
            stand.loud = loud
            bar.update()
            self.assertEqual(bar.meter.levels, loud)

    def test_the_kept_lines_of_the_meter_fall_by_the_seconds_passed_and_stay_when_none_did(self) -> None:
        stand, _, bar = bar_on()
        stand.loud = (1.0, HALF)
        bar.update(0.5)
        # Nothing plays any more. A line falls 12 decibels a second: an eighth of the meter in half a second.
        stand.loud = (0.0, 0.0)
        # A frame that is told no time, or nothing at all, leaves the lines where they are.
        for passed, tops in (((0.5,), (0.875, 0.375)), ((0.0,), (0.875, 0.375)), ((), (0.875, 0.375)),
                             ((1.0,), (0.625, 0.125)), ((60,), (0, 0))):
            bar.update(*passed)
            for top, there in zip(bar.meter.tops, tops, strict=True):
                self.assertAlmostEqual(top, there)
        self.assertEqual(bar.meter.levels, (0.0, 0.0))

    def test_a_bar_wide_enough_has_the_meter_after_the_time_as_high_as_the_buttons(self) -> None:
        places = arrange(BAR)
        self.assertEqual((places["time"], places["meter"], places["found"]),
                         ((242, 3, 109, 16), (363, 3, theme.BAR_METER_W, theme.ICON_BUTTON[1]), (925, 0, 29, 22)))
        # At 578 pixels everything just fits: the meter ends a gap before the two switches of the side panels.
        places = arrange((0, 0, 578, 22))
        self.assertEqual((places["meter"], places["list"], places["inspector"], places["found"]),
                         ((363, 3, 62, 16), (437, 3, 31, 16), (470, 3, 61, 16), (543, 0, 29, 22)))

    def test_the_two_switches_of_the_side_panels_stand_side_by_side_a_gap_before_the_findings(self) -> None:
        places = arrange(BAR)
        self.assertEqual((places["list"], places["inspector"], places["found"]),
                         ((819, 3, 31, 16), (852, 3, 61, 16), (925, 0, 29, 22)))
        self.assertEqual((places["inspector"][0] - places["list"][0] - places["list"][2],
                          places["found"][0] - places["inspector"][0] - places["inspector"][2]),
                         (theme.BUTTON_GAP, theme.GROUP_GAP))

    def test_a_switch_of_a_side_panel_is_on_while_its_panel_is_and_a_press_tells_the_owner_its_side(self) -> None:
        asked = []
        stand = Stand()
        ui = Ui()
        bar = TopBar(stand, ui, folded=asked.append)
        bar.place(BAR)
        bar.update()
        self.assertEqual((bar.list.text, bar.inspector.text, bar.list.is_on, bar.inspector.is_on),
                         ("LIST", "INSPECTOR", True, True))
        self.assertEqual((bar.list.help, bar.inspector.help),
                         ("press: show the list of songs, sounds and instruments, or hide it",
                          "press: show the numbers and the findings, or hide them"))
        click(ui, middle(bar.list.rect))
        click(ui, middle(bar.inspector.rect))
        self.assertEqual(asked, ["left", "right"])
        # What the owner says of the panels is what the switches show, whatever was pressed.
        for sides in ((False, True), (True, False), (False, False)):
            bar.sides = sides
            bar.update()
            self.assertEqual((bar.list.is_on, bar.inspector.is_on), sides)
        # With no owner to tell, a press does nothing.
        bar.folded = None
        click(ui, middle(bar.list.rect))
        self.assertEqual(asked, ["left", "right"])

    def test_a_bar_too_narrow_for_everything_has_no_meter_and_keeps_the_rest(self) -> None:
        places = arrange((0, 0, 577, 22))
        self.assertEqual(places["meter"][2], 0)
        self.assertEqual([places[thing] for thing in ("play", "stop", "loop", "dry", "time", "found")],
                         [(97, 3, 17, 16), (116, 3, 17, 16), (135, 3, 17, 16), (205, 3, 25, 16), (242, 3, 109, 16),
                          (542, 0, 29, 22)])
        # The time stays down to the width at which it ends a gap before the two switches.
        self.assertEqual(arrange((0, 0, 504, 22))["time"], (242, 3, 109, 16))

    def test_a_bar_too_narrow_for_the_time_as_well_has_neither_and_keeps_the_buttons_and_the_findings(self) -> None:
        places = arrange((0, 0, 503, 22))
        self.assertEqual((places["time"][2], places["meter"][2]), (0, 0))
        self.assertEqual([places[thing] for thing in ("play", "stop", "loop", "dry", "found")],
                         [(97, 3, 17, 16), (116, 3, 17, 16), (135, 3, 17, 16), (205, 3, 25, 16), (468, 0, 29, 22)])
        # The buttons keep their size and their places from the left end of the bar, wherever the bar is.
        self.assertEqual([arrange(SMALL_BAR)[thing] for thing in ("file", "play", "dry")],
                         [(46, 13, 31, 16), (137, 13, 17, 16), (245, 13, 25, 16)])

    def test_placed_at_two_sizes_every_widget_is_inside_the_bar(self) -> None:
        _, ui, bar = bar_on()
        places = arrange(BAR)
        self.assertEqual(places, {"file": (6, 3, 31, 16), "undo": (49, 3, 17, 16), "redo": (68, 3, 17, 16),
                                  "play": (97, 3, 17, 16), "stop": (116, 3, 17, 16), "loop": (135, 3, 17, 16),
                                  "pattern": (154, 3, 49, 16), "dry": (205, 3, 25, 16), "time": (242, 3, 109, 16),
                                  "meter": (363, 3, 62, 16), "list": (819, 3, 31, 16),
                                  "inspector": (852, 3, 61, 16), "found": (925, 0, 29, 22)})
        things = (*ROW, "time", "meter", "list", "inspector")
        self.assertEqual([getattr(bar, thing).rect for thing in things], [places[thing] for thing in things])
        self.assertEqual(bar.sign.rect, places["found"])
        # A wide bar, one too narrow for the meter, one too narrow for the time as well, and one for nothing at all.
        for rect in (BAR, (0, 0, 577, 22), SMALL_BAR, (0, 0, 30, 5)):
            bar.place(rect)
            # A meter that has gone is not drawn at all.
            self.assertEqual((len(ui.widgets), bar.meter.visible), (13, rect == BAR))
            self.assertTrue(all(lies_in(widget.rect, rect) for widget in ui.widgets), rect)
            self.assertTrue(all(lies_in(place, rect) for place in arrange(rect).values()), rect)


class TitleRow(unittest.TestCase):
    def test_the_row_has_the_square_the_two_names_and_three_buttons_and_the_middle_one_puts_a_big_window_back(
            self) -> None:
        places = title_parts(TITLE, "piece.json")
        self.assertEqual(places, {"mark": (6, 4, 5, 5), "program": (15, 0, 47, 14), "name": (74, 0, 59, 14),
                                  "minimise": (900, 0, 20, 14), "maximise": (920, 0, 20, 14),
                                  "close": (940, 0, 20, 14)})
        # A name too long for the row is cut a gap before the buttons, and everything is cut to the row.
        self.assertEqual(title_parts(TITLE, "n" * 200)["name"], (74, 0, 900 - theme.GROUP_GAP - 74, 14))
        self.assertTrue(all(lies_in(place, (0, 0, 50, 6)) for place in title_parts((0, 0, 50, 6), "a.json").values()))
        self.assertEqual([act(button, False) for button in (MINIMISE, MAXIMISE, CLOSE)],
                         ["minimise", "maximise", "close"])
        self.assertEqual([act(button, True) for button in (MINIMISE, MAXIMISE, CLOSE)],
                         ["minimise", "restore", "close"])

    def test_a_button_acts_on_its_release_and_an_empty_place_grabs_the_row_or_with_two_presses_fills_the_screen(
            self) -> None:
        told: list[str] = []
        now = [0.0]
        ui = Ui(clock=lambda: now[0])
        row = ui.add(Title(TITLE, "piece.json", told.append, lambda: told.append("grabbed")))
        places = title_parts(TITLE)
        for button in (MINIMISE, MAXIMISE, CLOSE):
            click(ui, middle(places[button]))
        self.assertEqual(told, ["minimise", "maximise", "close"])
        # A press that is let go off its button does nothing, and a window that fills the screen is put back.
        ui.handle(Event("down", middle(places[CLOSE]), button=1))
        ui.handle(Event("up", (400, 7), button=1))
        row.big = True
        click(ui, middle(places[MAXIMISE]))
        self.assertEqual(told[3:], ["restore"])
        # A press on an empty place grabs the row, and the second of two there acts as the middle button does.
        del told[:]
        now[0] = 10.0
        click(ui, (400, 7))
        now[0] = 10.2
        click(ui, (400, 7))
        row.big = False
        now[0] = 20.0
        click(ui, (400, 7))
        now[0] = 20.2
        click(ui, (401, 7))
        self.assertEqual(told, ["grabbed", "restore", "grabbed", "maximise"])
        # Each button says what it does to the window as it is, and the rest of the row what a drag does.
        ui.handle(Event("move", (400, 7)))
        self.assertEqual((ui.help, row.at((400, 7)), row.at(middle(places[CLOSE]))), (TITLE_HELP, None, CLOSE))
        said = []
        for big in (False, True):
            row.big = big
            for button in (MINIMISE, MAXIMISE, CLOSE):
                ui.handle(Event("move", middle(places[button])))
                said.append(ui.help)
        self.assertEqual(said, [BUTTON_HELP[what] for what in (MINIMISE, MAXIMISE, CLOSE, MINIMISE, RESTORE, CLOSE)])


class Left(unittest.TestCase):
    def test_a_click_on_a_row_picks_it_in_the_session(self) -> None:
        stand, ui, side = side_on()
        self.assertEqual(side.list.picked, ("song", "tune"))
        click(ui, spot(side.list, ("sound", "blip")))
        self.assertEqual(stand.asked, [("pick", "sound", "blip")])
        side.update()
        self.assertEqual(side.list.picked, ("sound", "blip"))

    def test_the_rows_are_made_again_only_when_the_listing_changes(self) -> None:
        stand, _, side = side_on()
        rows = side.list.rows
        self.assertEqual(rows, tuple(rows_of(stand.model.listing())))
        stand.lit, stand.model.picked = {("sound", "coin"): 1.0}, ("sound", "coin")
        side.update()
        side.update()
        self.assertIs(side.list.rows, rows)
        stand.model.holds["sounds"].append(("zap", "held"))
        side.update()
        self.assertIsNot(side.list.rows, rows)
        self.assertEqual(side.list.rows, tuple(rows_of(stand.model.listing())))
        self.assertIn(Row(("sound", "zap"), "zap", tag="held"), side.list.rows)

    def test_the_glows_of_the_session_reach_the_list_at_every_update(self) -> None:
        stand, _, side = side_on()
        self.assertEqual(side.list.glow, {})
        for lit in ({("sound", "coin"): 1.0}, {("sound", "coin"): 0.5, ("song", "tune"): 1.0}, {}):
            stand.lit = lit
            side.update()
            self.assertEqual(side.list.glow, lit)

    def test_a_pick_made_elsewhere_shows_as_the_picked_row(self) -> None:
        stand, _, side = side_on()
        stand.model.picked = ("instrument", "lead")
        side.update()
        self.assertEqual(side.list.picked, ("instrument", "lead"))
        # The picked thing is removed outside, and the session picks the first thing in its place.
        del stand.model.holds["instruments"][0]
        stand.model.picked = ("song", "tune")
        side.update()
        self.assertEqual(side.list.picked, ("song", "tune"))
        stand.model.picked = None
        side.update()
        self.assertIsNone(side.list.picked)

    def test_placed_at_two_sizes_the_list_is_inside_the_panel_and_clear_of_its_line(self) -> None:
        _, ui, side = side_on()
        self.assertEqual(side.list.rect, (2, 26, 119, 498))
        for rect in (SIDE, SMALL_SIDE, (0, 22, 3, 2)):
            side.place(rect)
            # Over the list stand the buttons of the rows that add.
            self.assertEqual(ui.widgets, [side.list, *side.plus.values()])
            self.assertTrue(all(lies_in(widget.rect, rect) for widget in ui.widgets), rect)
        side.place(SMALL_SIDE)
        self.assertEqual(side.list.rect, (42, 36, 55, 24))

    def test_the_button_of_a_row_that_adds_stands_in_its_row_while_all_of_the_row_is_in_view(self) -> None:
        _, ui, side = side_on()
        self.assertEqual(list(side.plus), ["song", "sound", "instrument"])
        for kind, button in side.plus.items():
            row = side.list.row_rect(("new", kind))
            self.assertEqual((button.visible, button.rect, side.list.at(middle(button.rect)).key),
                             (True, plus_rect(row), ("new", kind)))
            self.assertTrue(lies_in(button.rect, row))
        # The wheel on a button is the list's, and the mouse on it is told what a press there makes.
        ui.handle(Event("move", middle(side.plus["sound"].rect)))
        self.assertEqual(ui.help, "press: a new sound")
        # A row that is folded away has no button, and neither has one that is not all in view.
        click(ui, spot(side.list, "sounds"))
        side.update()
        self.assertEqual([button.visible for button in side.plus.values()], [True, False, True])
        side.place((0, 22, 124, theme.BROWSER_TOP + theme.BROWSER_PAD + 3 * theme.LIST_ROW_H + theme.LIST_ROW_H // 2))
        self.assertEqual([button.visible for button in side.plus.values()], [True, False, False])
        self.assertTrue(ui.handle(Event("wheel", middle(side.plus["song"].rect), wheel=-1)))
        side.update()
        self.assertEqual(side.plus["song"].visible, False)


class Changing(unittest.TestCase):
    """The left panel with a real session, on a small project in a folder of its own."""

    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        # Every render and the checks are refused at once, so that none costs its time.
        for owner, name in ((gui_model, "render_stems"), (gui_model, "render_entry"), (checks, "run")):
            patch = mock.patch.object(owner, name, side_effect=RowSynthError("not made in this test"))
            patch.start()
            self.addCleanup(patch.stop)
        # No thread of one test is left to run into the next.
        self.addCleanup(settle)
        self.path = Path(folder.name) / "things.json"
        self.path.write_text(json.dumps(THINGS), encoding="utf-8")
        self.session = Session(gui_model.Model(self.path))
        self.ui = Ui()
        self.side = Browser(self.session, self.ui)
        self.side.place(SIDE)
        self.side.update()

    def held(self) -> dict:
        """What the file holds now."""
        return json.loads(self.path.read_text(encoding="utf-8"))

    def menu(self, key: Hashable | None) -> list[str | None]:
        """Press the right button on the row of a key, or with None under the rows: the texts of the menu it opens."""
        x, y, _, h = self.side.list.rect
        pos = (x + 2, y + h - 2) if key is None else spot(self.side.list, key)
        self.ui.handle(Event("down", pos, button=3))
        self.ui.handle(Event("up", pos, button=3))
        return [item and item.text for item in self.ui.modal.items]

    def choose(self, text: str) -> None:
        """Click the item of the open menu that has a text, and bring the panel up to date."""
        menu = self.ui.modal
        click(self.ui, middle(menu.rows[[item and item.text for item in menu.items].index(text)]))
        self.side.update()

    def enter(self, text: str) -> None:
        """Type a text into the open box in place of what is picked there, press enter, and update the panel."""
        self.ui.handle(Event("text", text=text))
        self.ui.handle(Event("key", key="return"))
        self.side.update()

    def test_rename_opens_a_box_over_the_name_and_enter_gives_the_instrument_its_new_name_everywhere(self) -> None:
        # The menu of a thing is in the order every such menu has, and only an instrument has more than the three.
        self.assertEqual(self.menu(("song", "tune")), ["Rename...", "Copy", None, "Remove"])
        self.ui.handle(Event("key", key="escape"))
        self.assertEqual(self.menu(("instrument", "beep")),
                         ["Rename...", "Copy", "Copy into this song", None, "Remove"])
        self.choose("Rename...")
        box = self.ui.modal
        self.assertEqual((box.text, box.picked), ("beep", (0, 4)))
        self.assertEqual(box.rect, popups.over(self.side.list.name_rect(("instrument", "beep")), self.ui.size))
        # A name that is refused is said, and is in the box again to be mended. A dot is not typed at all.
        written = self.path.read_bytes()
        self.enter("my .bell")
        self.assertEqual(self.session.message,
                         "the channel tune in the pattern a of tune switches to the instrument as @beep, and a name "
                         "written so holds only letters, digits and _, which 'my bell' does not")
        box = self.ui.modal
        self.assertEqual((box.text, box.picked, self.path.read_bytes()), ("my bell", (7, 7), written))
        self.ui.handle(Event("key", key="a", mods=frozenset({"ctrl"})))
        self.enter("bell")
        self.assertIsNone(self.ui.modal)
        held = self.held()
        tune = held["songs"]["tune"]
        self.assertEqual([list(held["instruments"]), tune["channels"]["tune"]["inst"], tune["patterns"]["a"]["tune"],
                          held["sounds"]["zap"]["def"]["voices"][0]["inst"]],
                         [["bell"], "bell", "l4 c e @bell g > c", "bell"])
        # The press of the right button picked nothing, and the thing under its new name is the picked one.
        self.assertEqual([self.session.undo_label, self.session.message, self.session.model.picked,
                          self.side.list.picked],
                         ["instrument beep renamed to bell", "", ("instrument", "bell"), ("instrument", "bell")])

    def test_a_rename_is_taken_back_in_one_step_and_the_list_has_the_old_name_again(self) -> None:
        self.menu(("instrument", "beep"))
        self.choose("Rename...")
        self.enter("bell")
        self.assertIn(Row(("instrument", "bell"), "bell"), self.side.list.rows)
        self.assertTrue(self.session.undo())
        self.side.update()
        self.assertEqual(json.dumps(self.held()), json.dumps(THINGS))
        self.assertEqual([self.session.can_undo, self.session.redo_label], [False, "instrument beep renamed to bell"])
        self.assertIn(Row(("instrument", "beep"), "beep"), self.side.list.rows)
        # The picked thing is gone with its name, and the first thing there is takes its place.
        self.assertEqual(self.side.list.picked, ("song", "tune"))

    def test_copy_makes_a_thing_like_it_under_the_next_free_name_and_picks_it(self) -> None:
        self.menu(("song", "tune"))
        self.choose("Copy")
        held = self.held()
        self.assertEqual([list(held["songs"]), held["songs"]["tune2"]], [["tune", "tune2"], THINGS["songs"]["tune"]])
        self.assertEqual([self.session.undo_label, self.side.list.picked],
                         ["song tune copied as tune2", ("song", "tune2")])

    def test_remove_takes_a_thing_out_and_an_instrument_that_is_still_used_is_refused_and_said(self) -> None:
        written = self.path.read_bytes()
        self.menu(("instrument", "beep"))
        self.choose("Remove")
        self.assertEqual(self.session.message,
                         "instrument 'beep' is still used by the channel tune of tune, the channel tune in the "
                         "pattern a of tune, the sound zap")
        self.assertEqual([self.path.read_bytes(), self.session.can_undo], [written, False])
        # The picked thing removed, the first thing there is takes its place.
        self.menu(("song", "tune"))
        self.choose("Remove")
        self.assertEqual([self.held()["songs"], self.session.undo_label, self.side.list.picked],
                         [{}, "song tune removed", ("sound", "zap")])

    def test_copy_into_this_song_gives_the_song_that_was_picked_last_the_instrument_as_one_of_its_own(self) -> None:
        self.assertEqual(self.session.model.picked, ("song", "tune"))
        self.menu(("instrument", "beep"))
        self.choose("Copy into this song")
        held = self.held()
        # The song has a copy under the same name, which it plays from now on, and the project's is as it was.
        self.assertEqual([held["songs"]["tune"]["instruments"], held["instruments"], held["songs"]["tune"]["channels"]],
                         [THINGS["instruments"], THINGS["instruments"], THINGS["songs"]["tune"]["channels"]])
        self.assertEqual([self.session.undo_label, self.session.model.picked, self.side.list.picked],
                         ["instrument beep copied into tune", ("instrument", "tune.beep"), ("instrument", "tune.beep")])
        self.assertIn(Row(("instrument", "tune.beep"), "beep", tag="tune"), self.side.list.rows)
        # With an instrument picked it is still the song that was picked last, which has one of that name now.
        self.menu(("instrument", "beep"))
        self.choose("Copy into this song")
        self.assertEqual([self.session.message, self.session.undo_label],
                         ["the song 'tune' has an instrument called 'beep' already",
                          "instrument beep copied into tune"])
        # A project with no such song has nothing to copy into: the item is dim, and is not left out.
        self.assertTrue(self.session.change_many(projectedit.removed(self.held(), "song", "tune"), "song removed"))
        self.side.update()
        self.assertEqual(self.menu(("instrument", "beep")),
                         ["Rename...", "Copy", "Copy into this song", None, "Remove"])
        self.assertEqual([item.enabled for item in self.ui.modal.items if item], [True, True, False, True])

    def test_the_row_after_the_last_thing_of_a_section_and_its_button_make_one_more_of_that_kind(self) -> None:
        # A press on the row itself, and one on the button that stands in it.
        x, y, w, h = self.side.list.row_rect(("new", "song"))
        click(self.ui, (x + w // 2, y + h // 2))
        self.side.update()
        self.assertEqual([list(self.held()["songs"]), self.session.undo_label, self.side.list.picked],
                         [["tune", "song"], "new song song", ("song", "song")])
        click(self.ui, middle(self.side.plus["sound"].rect))
        self.side.update()
        self.assertEqual([list(self.held()["sounds"]), self.session.undo_label, self.side.list.picked],
                         [["zap", "sound"], "new sound sound", ("sound", "sound")])
        # The row is no thing: the keys pass it, and the right button on it is the one on no thing.
        click(self.ui, spot(self.side.list, ("sound", "sound")))
        self.ui.handle(Event("key", key="down"))
        self.side.update()
        self.assertEqual(self.side.list.picked, ("instrument", "beep"))
        pos = middle(self.side.list.name_rect(("new", "sound")))
        self.ui.handle(Event("down", pos, button=3))
        self.ui.handle(Event("up", pos, button=3))
        self.assertEqual([item.text for item in self.ui.modal.items], ["New song", "New sound", "New instrument"])

    def test_the_plus_of_the_instruments_opens_the_ready_ones_and_one_comes_under_its_name_or_the_next_free(
            self) -> None:
        button = self.side.plus["instrument"]
        made = []
        for text in ("bass", "bass", "Empty"):
            click(self.ui, middle(button.rect))
            menu = self.ui.modal
            self.assertEqual([item and item.text for item in menu.items],
                             ["Empty", "Empty in this song", None, *projectedit.STARTERS])
            # The list opens under the row of the button.
            self.assertEqual(menu.rect[1], button.rect[1] + theme.LIST_ROW_H)
            self.choose(text)
            made.append((self.session.undo_label, self.side.list.picked))
        self.assertEqual(made, [("new instrument bass", ("instrument", "bass")),
                                ("new instrument bass2", ("instrument", "bass2")),
                                ("new instrument instrument", ("instrument", "instrument"))])
        held = self.held()["instruments"]
        self.assertEqual([list(held), held["bass"], held["bass2"], held["instrument"]],
                         [["beep", "bass", "bass2", "instrument"], projectedit.STARTERS["bass"],
                          projectedit.STARTERS["bass"], {"wave": "pulse"}])
        # Each is one step: taken back, the project has the instruments it had before it.
        self.assertTrue(self.session.undo() and self.session.undo())
        self.assertEqual(list(self.held()["instruments"]), ["beep", "bass"])

    def test_empty_in_this_song_makes_an_instrument_of_the_song_s_own_that_was_picked_last(self) -> None:
        button = self.side.plus["instrument"]
        self.assertEqual(self.session.model.picked, ("song", "tune"))
        made = []
        for _ in range(2):
            click(self.ui, middle(button.rect))
            self.choose("Empty in this song")
            self.side.update()
            made.append((self.session.undo_label, self.session.model.picked, self.side.list.picked))
        # The second is made with an instrument picked: it is still the song that was picked last.
        self.assertEqual(made, [("new instrument instrument of tune", ("instrument", "tune.instrument"),
                                 ("instrument", "tune.instrument")),
                                ("new instrument instrument2 of tune", ("instrument", "tune.instrument2"),
                                 ("instrument", "tune.instrument2"))])
        held = self.held()
        self.assertEqual([held["songs"]["tune"]["instruments"], held["instruments"]],
                         [{"instrument": {"wave": "pulse"}, "instrument2": {"wave": "pulse"}}, THINGS["instruments"]])
        self.assertIn(Row(("instrument", "tune.instrument"), "instrument", tag="tune"), self.side.list.rows)
        # Each is one step that is taken back.
        self.assertTrue(self.session.undo())
        self.assertEqual(list(self.held()["songs"]["tune"]["instruments"]), ["instrument"])
        # A project with no such song has none to give it to: the item is dim, and is not left out.
        self.assertTrue(self.session.change_many(projectedit.removed(self.held(), "song", "tune"), "song removed"))
        self.side.update()
        click(self.ui, middle(button.rect))
        self.assertEqual([(item.text, item.enabled) for item in self.ui.modal.items[:2]],
                         [("Empty", True), ("Empty in this song", False)])

    def test_the_right_button_on_a_heading_or_under_the_rows_makes_a_new_thing_of_each_kind_and_picks_it(self) -> None:
        for key, kind in (("songs", "song"), (None, "sound"), (None, "instrument")):
            with self.subTest(kind=kind):
                self.assertEqual(self.menu(key), ["New song", "New sound", "New instrument"])
                self.choose(f"New {kind}")
                self.assertEqual([self.session.undo_label, self.side.list.picked], [f"new {kind} {kind}", (kind, kind)])
        held = self.held()
        # The press on the heading folded nothing away, and the new song and the new sound play the instrument there is.
        self.assertIsNotNone(self.side.list.name_rect(("song", "tune")))
        self.assertEqual([held["songs"]["song"]["channels"]["lead"]["inst"],
                          held["sounds"]["sound"]["def"]["voices"][0]["inst"], held["instruments"]["instrument"]],
                         ["beep", "beep", {"wave": "pulse"}])


@unittest.skipUnless(pygame, NEEDS)
class Drawn(unittest.TestCase):
    def test_the_bar_draws_the_findings_in_their_colour_and_the_time_in_a_sunk_box(self) -> None:
        rect = (0, 0, 520, theme.TOP_H)
        surface = pygame.Surface(rect[2:])
        canvas = Canvas(surface, Font(sprites.load("font")), sprites.load("icons"))
        stand, ui, bar = bar_on(rect)
        places = arrange(rect)
        x, y, w, h = places["found"]
        # The last is of checks that could not run.
        for count, refused, colour, has_icon in ((None, False, theme.DIM, False), (0, False, theme.GOOD, True),
                                                 (3, False, theme.FAULT, True), (0, True, theme.FAULT, True)):
            stand.count, stand.findings_refused = count, refused
            surface.fill(theme.PANEL)
            bar.update()
            bar.draw(canvas)
            ui.draw(canvas)
            # Nothing is there in another colour, and the icon stands before the text.
            self.assertEqual(ink(surface, places["found"], colour) + ink(surface, places["found"], theme.PANEL),
                             w * h)
            self.assertEqual(ink(surface, (x, y, w - Font.width(bar.found[1]), h), colour) > 0, has_icon)
        # With no number after it the sign stands at the right end.
        wide = canvas.icon_size("warn")[0]
        self.assertEqual(ink(surface, (x, y, w - wide, h), theme.FAULT), 0)
        x, y, w, h = places["time"]
        self.assertEqual(ink(surface, places["time"], theme.LINE), 2 * w + 2 * h - 4)
        self.assertGreater(ink(surface, places["time"], theme.TEXT), 0)
        self.assertEqual(ink(surface, places["time"], theme.BACKGROUND),
                         (w - 2) * (h - 2) - ink(surface, places["time"], theme.TEXT))

    def test_the_bar_draws_the_left_side_in_the_upper_bar_of_the_meter_and_no_meter_that_has_gone(self) -> None:
        rect = (0, 0, 578, theme.TOP_H)
        surface = pygame.Surface(rect[2:])
        canvas = Canvas(surface, Font(sprites.load("font")), sprites.load("icons"))
        stand, ui, bar = bar_on(rect)
        stand.loud = (1.0, 0.0)
        meter = arrange(rect)["meter"]
        upper, lower = bars(meter, 2, lying=True)
        shown = []
        for wide in (578, 577):
            bar.place((0, 0, wide, theme.TOP_H))
            surface.fill(theme.PANEL)
            bar.update()
            bar.draw(canvas)
            ui.draw(canvas)
            shown.append((ink(surface, upper, theme.BACKGROUND), ink(surface, lower, theme.BACKGROUND),
                          ink(surface, meter, theme.PANEL)))
        # The upper bar is lit from end to end and the lower one is empty. A pixel narrower, there is no meter.
        self.assertEqual(shown, [(0, lower[2] * lower[3], 0), (0, 0, meter[2] * meter[3])])


@unittest.skipUnless(pygame, NEEDS)
class InTheApp(unittest.TestCase):
    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        # No thread of one test is left to run into the next, and neither is a window or a sound output.
        self.addCleanup(settle)
        self.addCleanup(pygame.mixer.quit)
        self.addCleanup(pygame.display.quit)
        # The desktop is a full HD one, whatever the machine the tests are run on has.
        desktop = mock.patch("pygame.display.get_desktop_sizes", return_value=[(1920, 1080)])
        desktop.start()
        self.addCleanup(desktop.stop)

    def silent(self) -> App:
        """The app on the arcade example, on a machine whose sound output cannot be opened."""
        refusal = RowSynthError("the sound output cannot be opened at 44100 Hz: no such device")
        with mock.patch("rowsynth.gui.app.Player", side_effect=refusal) as asked:
            app = start(EXAMPLES / "arcade.json")
        asked.assert_called_once_with(app.model.rate)
        return app

    def held(self) -> threading.Event:
        """Has every render and the checks wait, and be refused when let go, so that none costs its time."""
        go = threading.Event()

        def refused(*args: object) -> None:
            go.wait(60)
            raise RowSynthError("not made in this test")

        for owner, name in ((gui_model, "render_stems"), (gui_model, "render_entry"), (checks, "run")):
            patch = mock.patch.object(owner, name, side_effect=refused)
            patch.start()
            self.addCleanup(patch.stop)
        # A test that fails before it lets them go leaves no thread waiting.
        self.addCleanup(go.set)
        return go

    def space(self, app: App) -> None:
        """Press the space bar in the window."""
        app.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE, mod=0))

    def test_three_frames_return_0_and_the_shot_holds_the_name_of_the_file_and_the_heading_of_the_songs(self) -> None:
        shot = self.folder / "shot.png"
        self.assertEqual(run(EXAMPLES / "arcade.json", frames=3, shot=shot), 0)
        # The end of the frames closed the window, and the session with its sound output.
        self.assertEqual((pygame.display.get_surface(), pygame.mixer.get_init()), (None, None))
        with open(shot, "rb") as file:
            drawn = pygame.image.load(file, ".png")
        parts = layout.frame(drawn.get_size(), layout.TITLE_H)
        # The title row has the square in the colour of what is switched on, the program's name and, dim, the
        # name of the file. Its three buttons stand at its right end.
        title = title_parts(parts["title"], "arcade.json")
        self.assertEqual((ink(drawn, title["mark"], theme.ACTIVE), title["name"][2]), (25, Font.width("arcade.json")))
        self.assertGreater(ink(drawn, title["program"], theme.TEXT), 0)
        self.assertGreater(ink(drawn, title["name"], theme.DIM), 0)
        self.assertTrue(all(ink(drawn, title[button], theme.TEXT) > 0 for button in (MINIMISE, MAXIMISE, CLOSE)))
        # The FILE button stands first in the top bar, and the name of the file is not there.
        places = arrange(parts["top"])
        x, y, w, h = places["file"]
        self.assertEqual(x, theme.BAR_MARGIN)
        self.assertGreater(ink(drawn, places["file"], theme.TEXT), 0)
        self.assertEqual(ink(drawn, (x + w, y, theme.GROUP_GAP, h), theme.PANEL), theme.GROUP_GAP * h)
        # The first row of the list in the left panel is the heading of the songs, in their colour.
        x, y, _, _ = parts["left"]
        heading = (x + theme.BROWSER_PAD + theme.LIST_PAD + theme.LIST_SQUARE + theme.LIST_SPACE,
                   y + theme.BROWSER_TOP, Font.width("SONGS"), theme.LIST_ROW_H)
        self.assertGreater(ink(drawn, heading, theme.SONG), 0)

    def test_three_frames_on_the_trailer_piece_return_0(self) -> None:
        self.held()
        self.assertEqual(run(EXAMPLES / "piece.json", frames=3), 0)

    def test_with_the_render_ready_a_frame_has_the_lead_in_the_middle_and_its_length_at_the_right(self) -> None:
        app = start(EXAMPLES / "arcade.json")
        # The first frame starts the render on its thread, and the model is asked for what it makes: that waits.
        app.update()
        numbers = app.model.numbers()
        app.update()
        art = app.screen.begin()
        app.draw(art)
        parts = app.parts(art.get_size())
        shown = app.panels["middle"]
        song = shown.panels["song"]
        self.assertEqual((shown.fits, song.channel, song.roll.colour), ("song", "lead", theme.SONG))
        first = song.roll.notes[0]
        bar = song.roll.map.note(first.row, first.rows, first.pitch)
        self.assertTrue(lies_in(bar, parts["middle"]))
        self.assertGreater(ink(art, bar, theme.SONG), bar[2] * bar[3] // 2)
        # The first of the numbers is the length: its label is dim at the left of its line, its value at the right.
        x, y, w, _ = inspector.arrange(parts["right"])["table"]
        value = Font.width(inspector.worded("seconds", numbers["seconds"]))
        self.assertGreater(ink(art, (x, y, Font.width("LENGTH"), Font.LINE), theme.DIM), 0)
        self.assertGreater(ink(art, (x + w - value, y, value, Font.LINE), theme.TEXT), 0)

    def test_a_click_on_a_sound_in_the_left_list_has_the_middle_show_that_sound(self) -> None:
        self.held()
        app = start(EXAMPLES / "arcade.json")
        art = app.screen.begin()
        shown, rows = app.panels["middle"], app.panels["left"].list
        head = soundpanel.arrange(app.parts(art.get_size())["middle"])["head"]

        def named() -> bool:
            """Of a frame made now: is something written in the colour of a sound where a sound has its name."""
            app.update()
            app.draw(art)
            return ink(art, head, theme.SOUND) > 0

        self.assertEqual((named(), shown.fits, app.model.picked), (False, "song", ("song", "tune")))
        at = tuple(part * app.screen.scale for part in spot(rows, ("sound", "powerup")))
        app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=at, button=1))
        app.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=at, button=1))
        self.assertEqual((named(), shown.fits, shown.panels["sound"].heading), (True, "sound", ("SOUND", "powerup")))
        self.assertEqual([widget for widget in shown.panels["song"].widgets if widget.visible], [])
        # A pick plays nothing. The list has the focus and has no use for the space bar: it plays the sound that
        # is picked, and stops it.
        self.assertEqual((app.ui.focus, app.session.playing), (rows, False))
        self.space(app)
        self.assertTrue(app.session.playing)
        self.space(app)
        self.assertFalse(app.session.playing)

    def test_the_space_bar_plays_what_is_picked_and_stops_it(self) -> None:
        self.held()
        app = start(EXAMPLES / "arcade.json")
        # The sound output is open at the rate the project is rendered at.
        self.assertEqual((pygame.mixer.get_init()[0], app.session.playing), (app.model.rate, False))
        self.space(app)
        self.assertTrue(app.session.playing)
        self.space(app)
        self.assertFalse(app.session.playing)

    def test_a_text_box_over_the_time_in_the_top_bar_takes_what_is_typed_and_the_space_bar_plays_nothing(self) -> None:
        self.held()
        with mock.patch("pygame.key.start_text_input") as asked:
            app = start(EXAMPLES / "arcade.json")
        # The window asks for what is typed to be told as text, apart from the keys.
        asked.assert_called_once_with()
        named: list[str] = []
        box = popups.ask(app.ui, app.panels["top"].time.rect, "arcade", named.append)
        # The window tells a key that types twice: as the key that went down, and then as its character.
        for key, typed in ((pygame.K_a, "a"), (pygame.K_SPACE, " "), (pygame.K_z, "z")):
            app.handle(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0, unicode=typed))
            app.handle(pygame.event.Event(pygame.TEXTINPUT, text=typed))
        self.assertEqual((box.text, app.session.playing, app.ui.modal), ("a z", False, box))
        # The box is drawn over the bar, where the time stands: its outline is in the colour of what is switched on.
        art = app.screen.begin()
        self.assertEqual(ink(art, box.rect, theme.ACTIVE), 0)
        app.draw(art)
        self.assertTrue(lies_in(box.rect, app.parts(art.get_size())["top"]))
        self.assertGreater(ink(art, box.rect, theme.ACTIVE), 0)
        app.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0))
        self.assertEqual((named, app.ui.modal), (["a z"], None))
        # With the box closed the space bar is the key that plays again, and its character is nothing.
        self.space(app)
        app.handle(pygame.event.Event(pygame.TEXTINPUT, text=" "))
        self.assertTrue(app.session.playing)

    def mouse(self, app: App, kind: int, pos: tuple[int, int]) -> None:
        """Tell the window what the mouse did at a position of the window: it moved, or its left button did."""
        more = {"rel": (0, 0), "buttons": (1, 0, 0)} if kind == pygame.MOUSEMOTION else {"button": 1}
        app.handle(pygame.event.Event(kind, pos=pos, **more))

    def test_the_title_row_moves_the_window_its_buttons_act_on_it_and_an_edge_resizes_it(self) -> None:
        self.held()
        app = self.silent()
        down, move, up = pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION, pygame.MOUSEBUTTONUP
        x, y, w, h = app.screen.stand()
        self.assertEqual((app.screen.frameless, app.title.name, app.title.rect, (w, h), app.screen.scale),
                         (True, "arcade.json", (0, 0, 960, 14), (1920, 984), 2))
        # A press on an empty place of the row takes hold of the window, which then goes with the mouse.
        self.mouse(app, down, (300, 10))
        self.mouse(app, move, (340, 35))
        app.update()
        self.assertEqual((app.screen.stand(), app.ui.held), ((x + 40, y + 25, w, h), app.title))
        # The window is under the mouse again as it was pressed: it stays, and the release lets go of it.
        self.mouse(app, move, (300, 10))
        self.mouse(app, up, (300, 10))
        self.mouse(app, move, (500, 300))
        app.update()
        self.assertEqual((app.screen.stand(), app.ui.held, app._shape()), ((x + 40, y + 25, w, h), None, ""))
        # An edge shows by the pointer which way it goes, and a press on it has that edge go with the mouse.
        self.mouse(app, move, (w - 2, 500))
        self.assertEqual(app._shape(), "ew")
        self.mouse(app, down, (w - 2, 500))
        self.mouse(app, move, (w - 102, 500))
        app.update()
        self.assertEqual((app.screen.stand(), app.ui.held, app._shape()), ((x + 40, y + 25, w - 100, h), None, "ew"))
        self.mouse(app, up, (w - 102, 500))
        # The narrower window shows less, at the scale it opened with.
        self.assertEqual((app.screen.begin().get_size(), app.screen.scale), (((w - 100) // 2, h // 2), 2))
        # No edge has hold of the mouse on a button of the row, in its corner: the middle one fills the desktop.
        work = (0, 0, 1920, 1032)
        at = tuple(part * app.screen.scale for part in middle(title_parts(app.title.rect)[MAXIMISE]))
        with mock.patch("rowsynth.gui.screen.work_area", return_value=work):
            self.mouse(app, down, at)
            self.mouse(app, up, at)
            app.update()
            self.assertEqual((app.screen.stand(), app.title.big), (work, True))
            # A window that fills the desktop is not moved by its row, and has no edges to take: it is put back.
            self.mouse(app, down, (300, 10))
            self.mouse(app, move, (340, 35))
            self.mouse(app, up, (340, 35))
            self.mouse(app, move, (1, 500))
            app.update()
            self.assertEqual((app.screen.stand(), app._shape()), (work, ""))
            app.frame_do(RESTORE)
        self.assertEqual((app.screen.stand(), app.screen.big, app.closed), ((x + 40, y + 25, w - 100, h), False, False))
        app.frame_do(CLOSE)
        self.assertTrue(app.closed)

    def test_the_title_row_moves_and_closes_the_window_while_a_menu_is_open_over_the_widgets(self) -> None:
        self.held()
        app = self.silent()
        down, move, up = pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION, pygame.MOUSEBUTTONUP
        x, y, w, h = app.screen.stand()
        app.file_menu((10, 40))
        menu = app.ui.modal
        self.assertIsNotNone(menu)
        # The menu has the mouse, and the row takes hold of the window all the same. The menu stays open.
        self.mouse(app, down, (300, 10))
        self.mouse(app, move, (340, 35))
        app.update()
        self.mouse(app, up, (340, 35))
        self.assertEqual((app.screen.stand(), app.ui.modal), ((x + 40, y + 25, w, h), menu))
        # A button of the row acts on its release.
        at = tuple(part * app.screen.scale for part in middle(title_parts(app.title.rect)[CLOSE]))
        self.mouse(app, down, at)
        self.mouse(app, up, at)
        self.assertTrue(app.closed)

    def test_the_app_says_for_a_place_of_the_window_whether_the_title_row_or_one_of_its_buttons_is_there(self) -> None:
        self.held()
        app = self.silent()
        scale = app.screen.scale
        close = tuple(part * scale for part in middle(title_parts(app.title.rect)[CLOSE]))
        self.assertEqual([app._asked(pos) for pos in ((300, 10), close, (300, 200))], ["caption", "button", ""])
        # The dummy driver has no window of the system's to take hold of, so the window is moved from here.
        self.assertEqual((app.screen.frame, app.screen.frameless), (None, True))

    def test_with_the_frame_of_the_system_kept_there_is_no_title_row_and_the_system_names_the_file(self) -> None:
        self.held()
        with mock.patch("rowsynth.gui.screen.own_frame", return_value=None):
            app = self.silent()
        size = app.screen.begin().get_size()
        self.assertEqual((app.screen.frameless, app.title, app.parts(size), app.panels["top"].rect),
                         (False, None, layout.frame(size), (0, 0, 960, 22)))
        self.assertEqual(pygame.display.get_caption()[0], "rowsynth  arcade.json")
        # The edges are the system's: a press on one takes hold of nothing here, and the buttons do nothing.
        self.mouse(app, pygame.MOUSEBUTTONDOWN, (1, 500))
        self.mouse(app, pygame.MOUSEMOTION, (40, 500))
        app.update()
        app.frame_do(MAXIMISE)
        self.assertEqual((app._drag, app._shape(), app.screen.big, pygame.display.get_surface().get_size()),
                         (None, "", False, (1920, 984)))
        app.leave()
        self.assertEqual(pygame.display.get_caption()[0], "rowsynth")

    def test_a_border_between_two_columns_is_dragged_and_stays_for_the_run_and_each_column_keeps_a_least_width(
            self) -> None:
        self.held()
        app = self.silent()
        ui, (left, right) = app.ui, (app.borders["left"], app.borders["right"])
        self.assertEqual((app.columns, left.line(), right.line()), ((124, 174), 123, 786))
        self.assertTrue(left.help.startswith("press: drag ") and right.help.startswith("press: drag "))
        # The mouse on a border has the pointer that points across, and a drag takes the line with it.
        ui.handle(Event("move", (124, 200)))
        self.assertEqual((ui.hot, app._shape()), (left, "ew"))
        ui.handle(Event("down", (124, 200), button=1))
        ui.handle(Event("move", (200, 210)))
        parts = app.parts(ui.size)
        self.assertEqual((app.columns, left.line(), parts["left"][2], parts["middle"][0]), ((200, 174), 199, 200, 200))
        self.assertEqual([app.panels[name].rect for name in ("left", "middle", "right")],
                         [parts[name] for name in ("left", "middle", "right")])
        # Each column keeps a least width, wherever the mouse goes.
        ui.handle(Event("move", (-40, 210)))
        self.assertEqual(app.columns, (layout.LEAST_SIDE, 174))
        ui.handle(Event("move", (900, 210)))
        self.assertEqual(app.columns, (960 - 174 - layout.LEAST_MIDDLE, 174))
        ui.handle(Event("move", (300, 210)))
        ui.handle(Event("up", (300, 210), button=1))
        ui.handle(Event("down", (786, 100), button=1))
        ui.handle(Event("move", (700, 100)))
        ui.handle(Event("up", (700, 100), button=1))
        self.assertEqual((app.columns, right.line(), app.panels["right"].rect[:3:2]), ((300, 260), 700, (700, 260)))
        # They stay as they were dragged for the run: on the first screen there are none, and in the next file
        # that is opened they are where they were.
        app.leave()
        self.assertEqual((app.borders, app.columns), ({}, (300, 260)))
        self.assertTrue(app.open(EXAMPLES / "arcade.json"))
        self.assertEqual((app.borders["left"].line(), app.borders["right"].line(), app.panels["left"].rect[2]),
                         (299, 700, 300))

    def wide(self, app: App, width: int) -> dict[str, layout.Rect]:
        """Have the window be a width, in window pixels, and draw a frame in it: the parts of the frame it then has."""
        x, y, _, h = app.screen.stand()
        app.screen.put((x, y, width, h))
        app.update()
        art = app.screen.begin()
        app.draw(art)
        # The frame after it has the switches of the top bar as the panels are now.
        app.update()
        return app.parts(art.get_size())

    def test_a_side_panel_is_off_a_window_too_narrow_for_it_and_comes_back_with_the_room(self) -> None:
        self.held()
        app = self.silent()
        top, (left, right) = app.panels["top"], (app.panels["left"], app.panels["right"])

        def on() -> tuple[bool, ...]:
            return (left.shown, app.borders["left"].visible, top.list.is_on,
                    right.shown, app.borders["right"].visible, top.inspector.is_on)

        parts = self.wide(app, 1920)
        self.assertEqual((app.screen.scale, parts["middle"], on()), (2, (124, 36, 662, 442), (True,) * 6))
        # A window with no room for both has no inspector, and the middle has what the inspector had.
        parts = self.wide(app, 1800)
        self.assertEqual((app.screen.scale, parts["left"][2], parts["middle"], parts["right"][2], on()),
                         (2, 124, (124, 36, 776, 442), 0, (True, True, True, False, False, False)))
        # None of what the inspector has is seen or takes the mouse, and nothing is drawn where it stood.
        self.assertFalse(any(widget.visible for widget in right.widgets))
        # Half of a desktop of 2560 across has neither, and the middle has all of the width at the same scale.
        parts = self.wide(app, 1280)
        self.assertEqual((app.screen.scale, parts["left"][2], parts["middle"], parts["right"][2], on()),
                         (2, 0, (0, 36, 640, 442), 0, (False,) * 6))
        self.assertFalse(any(widget.visible for widget in (*left.widgets, *right.widgets)))
        self.assertEqual(left.list.visible, False)
        # With room again, both are back by themselves, as wide as they were.
        parts = self.wide(app, 1920)
        self.assertEqual((parts["left"][2], parts["right"][2], on(), app.wishes),
                         (124, 174, (True,) * 6, {"left": None, "right": None}))
        self.assertTrue(left.list.visible)

    def test_the_switch_of_a_side_panel_brings_it_onto_a_narrow_window_and_takes_it_off_a_wide_one(self) -> None:
        self.held()
        app = self.silent()
        top = app.panels["top"]
        self.wide(app, 1280)
        # A press on LIST in a window with no room for the list: the list is there, and the middle is narrower.
        click(app.ui, middle(top.list.rect))
        parts = self.wide(app, 1280)
        self.assertEqual((app.wishes, parts["left"][2], parts["middle"], parts["right"][2], app.panels["left"].shown,
                          app.borders["left"].visible, (top.list.is_on, top.inspector.is_on)),
                         ({"left": True, "right": None}, 124, (124, 36, 516, 442), 0, True, True, (True, False)))
        # Its border is dragged as in any window, with the inspector off the window and out of the sum.
        app.pull("left", 199)
        self.assertEqual((app.columns, app.parts((640, 492))["middle"]), ((200, 174), (200, 36, 440, 442)))
        app.pull("left", 123)
        # Another press takes it off again, and that is what the window gives by itself: nothing is left asked.
        click(app.ui, middle(top.list.rect))
        parts = self.wide(app, 1280)
        self.assertEqual((app.wishes, parts["middle"], top.list.is_on), ({"left": None, "right": None},
                                                                         (0, 36, 640, 442), False))
        # What was asked onto a narrow window is over in a wide one, which has it anyway: narrow again, it is off.
        click(app.ui, middle(top.inspector.rect))
        self.assertEqual((app.wishes, self.wide(app, 1280)["right"][2]), ({"left": None, "right": True}, 174))
        self.assertEqual((self.wide(app, 1920)["right"][2], app.wishes), (174, {"left": None, "right": None}))
        self.assertEqual(self.wide(app, 1280)["right"][2], 0)
        # In a wide window a press takes a panel off, for the middle to have its room, and a press brings it back.
        self.wide(app, 1920)
        click(app.ui, middle(top.inspector.rect))
        parts = self.wide(app, 1920)
        self.assertEqual((app.wishes["right"], parts["middle"], app.panels["right"].shown, top.inspector.is_on),
                         (False, (124, 36, 836, 442), False, False))
        click(app.ui, middle(top.inspector.rect))
        self.assertEqual((app.wishes["right"], self.wide(app, 1920)["right"][2]), (None, 174))

    def test_a_frame_the_system_calls_for_while_it_has_the_window_is_drawn_on_the_window_as_it_then_is(self) -> None:
        self.held()
        app = self.silent()
        # It is what the screen is told to call while the system moves or sizes the window.
        self.assertEqual(app.screen.during, app._live)
        x, y, _, h = app.screen.stand()
        app.screen.put((x, y, 1280, h))
        updated = []
        with mock.patch.object(app, "update", side_effect=updated.append):
            app._live()
            app._live()
        # The panels are laid out for the window as it is now, and the picture is on the window.
        window = pygame.display.get_surface()
        self.assertEqual((app._size, window.get_size(), app.panels["left"].shown), ((640, 492), (1280, 984), False))
        # The right end of the top bar, which is in the panel's colour, is where the narrower window ends.
        self.assertEqual(window.get_at((1270, 40))[:3], theme.PANEL)
        # The first frame of a move has no time behind it, and the next has what passed since the first.
        self.assertEqual((updated[0], 0.0 <= updated[1] < 0.25), (0.0, True))

    def test_an_edge_of_the_window_is_not_dragged_further_in_than_the_least_size_at_its_scale(self) -> None:
        self.held()
        app = self.silent()
        x, y, w, h = app.screen.stand()
        self.assertEqual((app.screen.scale, app.screen.least), (2, (1280, 800)))
        self.mouse(app, pygame.MOUSEBUTTONDOWN, (w - 2, h - 2))
        self.mouse(app, pygame.MOUSEMOTION, (40, 30))
        app.update()
        self.mouse(app, pygame.MOUSEBUTTONUP, (40, 30))
        self.assertEqual(app.screen.stand(), (x, y, 1280, 800))
        self.assertEqual((app.screen.begin().get_size(), app.screen.scale), (theme.LEAST_ART, 2))

    def test_ctrl_and_z_is_undo_and_ctrl_and_y_or_ctrl_and_shift_and_z_is_redo_whatever_has_the_focus(self) -> None:
        self.held()
        asked = mock.Mock()
        # The app takes the two from the session as it is made.
        with mock.patch.object(Session, "undo", asked.undo), mock.patch.object(Session, "redo", asked.redo):
            app = self.silent()
        app.update()
        rows = app.panels["left"].list
        at = tuple(part * app.screen.scale for part in spot(rows, ("sound", "powerup")))
        app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=at, button=1))
        app.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=at, button=1))
        self.assertIs(app.ui.focus, rows)
        ctrl, shift, alt = pygame.KMOD_LCTRL, pygame.KMOD_RSHIFT, pygame.KMOD_LALT
        # The first two and the last are no step: the key alone, and the key with one thing too many held.
        for key, mod in ((pygame.K_z, 0), (pygame.K_y, shift), (pygame.K_z, ctrl), (pygame.K_y, ctrl),
                         (pygame.K_z, ctrl | shift), (pygame.K_z, ctrl | alt)):
            app.handle(pygame.event.Event(pygame.KEYDOWN, key=key, mod=mod))
        self.assertEqual(asked.mock_calls, [mock.call.undo(), mock.call.redo(), mock.call.redo()])

    def test_closing_the_window_while_a_render_runs_returns_and_waits_for_no_worker(self) -> None:
        self.held()
        app = start(EXAMPLES / "arcade.json")
        app.update()
        workers = [thread for thread in threading.enumerate() if thread.name.startswith("rowsynth-")]
        self.assertEqual(sorted(worker.name for worker in workers), ["rowsynth-checks", "rowsynth-render"])
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        self.assertEqual(app.run(frames=5), 0)
        self.assertEqual((pygame.display.get_surface(), pygame.mixer.get_init()), (None, None))
        # The workers are still at it, and none of them keeps the process from ending.
        self.assertTrue(all(worker.is_alive() and worker.daemon for worker in workers))

    def test_each_frame_tells_the_panels_the_seconds_the_frame_before_took(self) -> None:
        self.held()
        app = self.silent()
        clock = mock.Mock()
        clock.tick.return_value = 250
        with mock.patch("pygame.time.Clock", return_value=clock), \
                mock.patch.object(app.panels["top"], "update") as told:
            self.assertEqual(app.run(frames=3), 3)
        self.assertEqual(told.call_args_list, [mock.call(0.0), mock.call(0.25), mock.call(0.25)])

    def test_with_no_sound_output_the_session_has_no_player_and_says_so_when_asked_to_play(self) -> None:
        app = self.silent()
        self.assertEqual((app.session.player, app.session.message), (None, ""))
        self.space(app)
        self.assertEqual((app.session.playing, app.session.message), (False, "no sound output"))

    def test_the_status_bar_shows_the_help_under_the_mouse_or_else_the_message_and_what_is_picked(self) -> None:
        from rowsynth.gui.app import picked_words

        self.assertEqual([picked_words(("sound", "coin")), picked_words(None)], ["SOUND coin", ""])
        app = self.silent()
        art = app.screen.begin()
        x, y, w, h = layout.frame(art.get_size())["status"]
        picked = Font.width("SONG tune")
        right = (x + w - theme.BAR_MARGIN - picked, y + 1, picked, h - 1)
        left = (x, y + 1, right[0] - x, h - 1)

        def shown() -> tuple[bool, bool, bool]:
            """Of a frame drawn now: is the line there as help, is it there as a message, is the pick at the end."""
            app.draw(art)
            return ink(art, left, theme.DIM) > 0, ink(art, left, theme.TEXT) > 0, ink(art, right, theme.DIM) > 0

        self.assertEqual(shown(), (False, False, True))
        # The mouse is told in pixels of the window, each art pixel being a square of them.
        top = app.panels["top"]
        at = tuple(part * app.screen.scale for part in middle(top.play.rect))
        app.handle(pygame.event.Event(pygame.MOUSEMOTION, pos=at, rel=(1, 0), buttons=(0, 0, 0)))
        self.assertEqual((app.ui.hot, app.ui.help), (top.play, top.play.help))
        self.assertEqual(shown(), (True, False, True))
        # Off the widgets, as on the status bar itself, the line is what the session has to say.
        self.space(app)
        at = tuple(part * app.screen.scale for part in middle((x, y, w, h)))
        app.handle(pygame.event.Event(pygame.MOUSEMOTION, pos=at, rel=(1, 0), buttons=(0, 0, 0)))
        self.assertEqual((app.ui.help, shown()), ("", (False, True, True)))
        # A click in the window reaches the widget under it.
        at = tuple(part * app.screen.scale for part in middle(top.loop.rect))
        app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=at, button=1))
        app.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=at, button=1))
        self.assertFalse(app.session.loop)

    def test_the_line_of_the_status_bar_is_a_new_message_then_the_help_and_the_message_where_no_help_is(self) -> None:
        from rowsynth.gui.app import MESSAGE_TIME, status_line

        self.assertEqual(MESSAGE_TIME, 3.0)
        said, help = ("refused", theme.TEXT), ("press to play", theme.DIM)
        self.assertEqual([status_line("press to play", "refused", age) for age in (0.0, 2.99, 3.0, 60.0)],
                         [said, said, help, help])
        self.assertEqual([status_line("", "refused", age) for age in (0.0, 60.0)], [said, said])
        self.assertEqual([status_line("press to play", "", 0.0), status_line("", "", 0.0)], [help, ("", theme.DIM)])

    def test_what_the_session_says_has_the_status_bar_for_three_seconds_whatever_the_mouse_is_on(self) -> None:
        self.held()
        app = self.silent()
        art = app.screen.begin()
        x, y, w, h = layout.frame(art.get_size())["status"]
        left = (x, y + 1, w // 2, h - 1)
        at = tuple(part * app.screen.scale for part in middle(app.panels["top"].play.rect))
        app.handle(pygame.event.Event(pygame.MOUSEMOTION, pos=at, rel=(1, 0), buttons=(0, 0, 0)))

        def line(passed: float | None = None) -> str:
            """Of a frame made now, after so many seconds where they are given: which line the bar has."""
            if passed is not None:
                app.update(passed)
            app.draw(art)
            return "message" if ink(art, left, theme.TEXT) else "help" if ink(art, left, theme.DIM) else ""

        self.assertEqual((app.session.message, line()), ("", "help"))
        # The first frame says that there is no sound output. In the frame that finds it said, it is new.
        self.assertEqual((line(1.0), app.session.message), ("message", "no sound output"))
        self.assertEqual([line(2.5), line(0.5), line(0.0)], ["message", "help", "help"])
        # The same words said again are new again, and so before a frame has counted them.
        self.space(app)
        self.assertEqual((app.session.message, app.session.messages), ("no sound output", 2))
        self.assertEqual([line(), line(0.0), line(2.5)], ["message", "message", "message"])
        # Said anew before its time is up, a message has its time anew: it is 5 seconds since the space bar.
        app.session.say("another thing")
        self.assertEqual([line(0.0), line(2.5), line(0.5)], ["message", "message", "help"])
        # With nothing to say, the help line has the bar at once.
        app.session.say("yet another")
        app.session.say("")
        self.assertEqual(line(0.0), "help")

    def test_a_file_that_is_not_a_project_is_refused_before_any_window_and_named_as_it_was_given(self) -> None:
        (self.folder / "notes.json").write_text("o4 l8 c e g\n", encoding="utf-8")
        # A spelling of the path that is not the one a Path gives it.
        path = f"{self.folder}/./notes.json"
        with mock.patch("pygame.display.set_mode") as opened, self.assertRaises(RowSynthError) as refused:
            run(path, frames=1)
        self.assertEqual(str(refused.exception), f"{path}: line 1, column 1: Expecting value")
        opened.assert_not_called()


if __name__ == "__main__":
    unittest.main()
