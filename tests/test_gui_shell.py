"""Tests for the shell of the GUI: the top bar, the left panel, and the app that joins them to a session."""
from __future__ import annotations

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
from rowsynth.gui import layout, run, sprites, start, theme
from rowsynth.gui import model as gui_model
from rowsynth.gui.browser import Browser, rows_of
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.dials import bars
from rowsynth.gui.font import Font
from rowsynth.gui.lists import List, Row
from rowsynth.gui.panels import within
from rowsynth.gui.topbar import TopBar, arrange, moment, timing
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
SMALL_BAR, SMALL_SIDE = (40, 10, 200, 22), (40, 32, 60, 30)
#: The level, from 0 to 1, that is 24 decibels under the top: half the way along a meter.
HALF = 10 ** (-24 / 20)


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
        self.seconds = 0.0
        self.at = 0.0
        self.loud = (0.0, 0.0)
        self.count: int | None = None
        self.lit: dict[tuple[str, str], float] = {}

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
            Row("sounds", "SOUNDS", head=True, colour=theme.SOUND),
            Row(("sound", "coin"), "coin"),
            Row(("sound", "blip"), "blip", tag="3"),
            Row(("sound", "boom"), "boom", tag="parts"),
            # A section that holds nothing has its heading all the same.
            Row("instruments", "INSTRUMENTS", head=True, colour=theme.INSTRUMENT),
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
    def test_play_stop_and_loop_call_the_session_and_each_has_a_line_of_help(self) -> None:
        stand, ui, bar = bar_on()
        self.assertEqual(bar.name.text, "arcade.json")
        click(ui, middle(bar.play.rect))
        click(ui, middle(bar.stop.rect))
        self.assertEqual(stand.asked, [("play",), ("stop",)])
        self.assertTrue(bar.loop.is_on)
        click(ui, middle(bar.loop.rect))
        self.assertFalse(stand.loop)
        click(ui, middle(bar.loop.rect))
        self.assertTrue(stand.loop)
        lines = {bar.play.help, bar.stop.help, bar.loop.help}
        self.assertEqual((len(lines), "" in lines), (3, False))

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
        places = arrange(BAR, "arcade.json")
        self.assertEqual((places["time"], places["meter"], places["found"]),
                         ((150, 3, 109, 16), (271, 3, theme.BAR_METER_W, theme.ICON_BUTTON[1]), (925, 0, 29, 22)))
        # At 380 pixels everything just fits: the meter ends a gap before the findings.
        places = arrange((0, 0, 380, 22), "arcade.json")
        self.assertEqual((places["meter"], places["found"]), ((271, 3, 62, 16), (345, 0, 29, 22)))

    def test_a_bar_too_narrow_for_everything_has_no_meter_and_keeps_the_rest(self) -> None:
        places = arrange((0, 0, 379, 22), "arcade.json")
        self.assertEqual(places["meter"][2], 0)
        self.assertEqual([places[thing] for thing in ("name", "play", "stop", "loop", "time", "found")],
                         [(6, 0, 65, 22), (83, 3, 17, 16), (102, 3, 17, 16), (121, 3, 17, 16), (150, 3, 109, 16),
                          (344, 0, 29, 22)])
        # The time stays down to the width at which it ends a gap before the findings.
        self.assertEqual(arrange((0, 0, 306, 22), "arcade.json")["time"], (150, 3, 109, 16))

    def test_a_bar_too_narrow_for_the_time_as_well_has_neither_and_keeps_the_buttons_and_the_findings(self) -> None:
        places = arrange((0, 0, 305, 22), "arcade.json")
        self.assertEqual((places["time"][2], places["meter"][2]), (0, 0))
        self.assertEqual([places[thing] for thing in ("name", "play", "stop", "loop", "found")],
                         [(6, 0, 65, 22), (83, 3, 17, 16), (102, 3, 17, 16), (121, 3, 17, 16), (270, 0, 29, 22)])
        # Only after the two have gone does the name give way, and the buttons keep their size.
        self.assertEqual([arrange(SMALL_BAR, "arcade.json")[thing] for thing in ("name", "play")],
                         [(46, 10, 65, 22), (123, 13, 17, 16)])
        self.assertEqual([arrange((40, 10, 150, 22), "arcade.json")[thing] for thing in ("name", "play")],
                         [(46, 10, 30, 22), (88, 13, 17, 16)])

    def test_placed_at_two_sizes_every_widget_is_inside_the_bar(self) -> None:
        _, ui, bar = bar_on()
        places = arrange(BAR, "arcade.json")
        self.assertEqual(places, {"name": (6, 0, 65, 22), "play": (83, 3, 17, 16), "stop": (102, 3, 17, 16),
                                  "loop": (121, 3, 17, 16), "time": (150, 3, 109, 16), "meter": (271, 3, 62, 16),
                                  "found": (925, 0, 29, 22)})
        things = ("name", "play", "stop", "loop", "time", "meter")
        self.assertEqual([getattr(bar, thing).rect for thing in things], [places[thing] for thing in things])
        # A wide bar, one too narrow for the meter, one too narrow for the time as well, and one for nothing at all.
        for rect in (BAR, (0, 0, 379, 22), SMALL_BAR, (0, 0, 30, 5)):
            bar.place(rect)
            # A meter that has gone is not drawn at all.
            self.assertEqual((len(ui.widgets), bar.meter.visible), (6, rect == BAR))
            self.assertTrue(all(lies_in(widget.rect, rect) for widget in ui.widgets), rect)
            self.assertTrue(all(lies_in(place, rect) for place in arrange(rect, "arcade.json").values()), rect)


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
            self.assertEqual(ui.widgets, [side.list])
            self.assertTrue(lies_in(side.list.rect, rect), rect)
        side.place(SMALL_SIDE)
        self.assertEqual(side.list.rect, (42, 36, 55, 24))


@unittest.skipUnless(pygame, NEEDS)
class Drawn(unittest.TestCase):
    def test_the_bar_draws_the_findings_in_their_colour_and_the_time_in_a_sunk_box(self) -> None:
        rect = (0, 0, 400, theme.TOP_H)
        surface = pygame.Surface(rect[2:])
        canvas = Canvas(surface, Font(sprites.load("font")), sprites.load("icons"))
        stand, ui, bar = bar_on(rect)
        places = arrange(rect, "arcade.json")
        x, y, w, h = places["found"]
        for count, colour, has_icon in ((None, theme.DIM, False), (0, theme.GOOD, True), (3, theme.FAULT, True)):
            stand.count = count
            surface.fill(theme.PANEL)
            bar.update()
            bar.draw(canvas)
            ui.draw(canvas)
            # Nothing is there in another colour, and the icon stands before the text.
            self.assertEqual(ink(surface, places["found"], colour) + ink(surface, places["found"], theme.PANEL),
                             w * h)
            self.assertEqual(ink(surface, (x, y, w - Font.width(bar.found[1]), h), colour) > 0, has_icon)
        x, y, w, h = places["time"]
        self.assertEqual(ink(surface, places["time"], theme.LINE), 2 * w + 2 * h - 4)
        self.assertGreater(ink(surface, places["time"], theme.TEXT), 0)
        self.assertEqual(ink(surface, places["time"], theme.BACKGROUND),
                         (w - 2) * (h - 2) - ink(surface, places["time"], theme.TEXT))

    def test_the_bar_draws_the_left_side_in_the_upper_bar_of_the_meter_and_no_meter_that_has_gone(self) -> None:
        rect = (0, 0, 400, theme.TOP_H)
        surface = pygame.Surface(rect[2:])
        canvas = Canvas(surface, Font(sprites.load("font")), sprites.load("icons"))
        stand, ui, bar = bar_on(rect)
        stand.loud = (1.0, 0.0)
        meter = arrange(rect, "arcade.json")["meter"]
        upper, lower = bars(meter, 2, lying=True)
        shown = []
        for wide in (400, 379):
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

        for owner, name in ((gui_model, "render_loop"), (gui_model, "render_entry"), (checks, "run")):
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
        parts = layout.frame(drawn.get_size())
        x, y, _, h = parts["top"]
        name = (x + theme.BAR_MARGIN, y, Font.width("arcade.json"), h)
        self.assertGreater(ink(drawn, name, theme.TEXT), 0)
        # After the name there is air, up to the first button.
        self.assertEqual(ink(drawn, (name[0] + name[2], y, theme.GROUP_GAP, h - 1), theme.PANEL),
                         theme.GROUP_GAP * (h - 1))
        # The first row of the list in the left panel is the heading of the songs, in their colour.
        x, y, _, _ = parts["left"]
        heading = (x + theme.BROWSER_PAD + theme.LIST_PAD + theme.LIST_SQUARE + theme.LIST_SPACE,
                   y + theme.BROWSER_TOP, Font.width("SONGS"), theme.LIST_ROW_H)
        self.assertGreater(ink(drawn, heading, theme.SONG), 0)

    def test_the_space_bar_plays_what_is_picked_and_stops_it(self) -> None:
        self.held()
        app = start(EXAMPLES / "arcade.json")
        # The sound output is open at the rate the project is rendered at.
        self.assertEqual((pygame.mixer.get_init()[0], app.session.playing), (app.model.rate, False))
        self.space(app)
        self.assertTrue(app.session.playing)
        self.space(app)
        self.assertFalse(app.session.playing)

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
        self.space(app)
        self.assertEqual(shown(), (False, True, True))
        # The mouse is told in pixels of the window, each art pixel being a square of them.
        top = app.panels["top"]
        at = tuple(part * app.screen.scale for part in middle(top.play.rect))
        app.handle(pygame.event.Event(pygame.MOUSEMOTION, pos=at, rel=(1, 0), buttons=(0, 0, 0)))
        self.assertEqual((app.ui.hot, app.ui.help), (top.play, top.play.help))
        self.assertEqual(shown(), (True, False, True))
        # A click in the window reaches the widget under it.
        at = tuple(part * app.screen.scale for part in middle(top.loop.rect))
        app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=at, button=1))
        app.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=at, button=1))
        self.assertFalse(app.session.loop)

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
