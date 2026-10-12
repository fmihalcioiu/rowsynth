"""Tests for the line of help in the status bar: every control of the window says one, and each has its room."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import unittest
from collections.abc import Hashable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

# No test opens a window that can be seen, or a sound device. This is said before pygame is first imported.
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from rowsynth import checks
from rowsynth.errors import RowSynthError
from rowsynth.gui import browser, exports, files, fmview, inspector, instpanel, layerview, layout, mixer, songpanel
from rowsynth.gui import model as gui_model
from rowsynth.gui import soundpanel, start, theme, topbar
from rowsynth.gui import steps as drumsteps
from rowsynth.gui.font import Font
from rowsynth.gui.kit import VELOCITY_HELP
from rowsynth.gui.lists import List, Row
from rowsynth.gui.roll import ACROSS_HELP, ALONG_HELP, Roll
from rowsynth.gui.ui import Event, Ui, Widget, inside

try:
    import pygame
except ImportError:
    pygame = None

if TYPE_CHECKING:
    from rowsynth.gui.app import App

EXAMPLES = Path(__file__).resolve().parent / "data"
NEEDS = "pygame is not installed"
Pos = tuple[int, int]
#: How far apart the places are at which the mouse is rested all over the window.
STEP = 12
#: How wide the art surface of the smallest window is.
SMALLEST = theme.MIN_ART[0]
#: A held sound for the arcade example, which has none, with a longest time that has its line on the clock.
HUM = {"held": {"note": {"voices": [{"inst": "bass", "notes": [[0, "c3", 60]]}]}, "start": 6, "loop": 30, "join": 3,
                "end": {"voices": [{"inst": "bass", "notes": [[0, "c3", 2]]}]}, "let_go": 4},
       "db": -6, "max_seconds": 0.8}

#: The things of the widgets that draw several, each with how a line begins that is said on a thing of that kind:
#: at the places the layout of its widget gives for it, every one of the beginnings is that of a line.
KINDS: dict[str, tuple[str, ...]] = {
    "the left list, a heading": ("press: fold the songs away or bring them back   right press: a new song, sound",),
    "the left list, a song": ("press: show song tune, twice to rename it   right press: its menu   wheel: along",),
    "the left list, a sound": ("press: show sound coin, twice to rename it   right press: its menu",),
    "the left list, an instrument": ("press: show instrument lead, twice to rename it   right press: its menu",),
    "the left list, a row that adds": ("press: a new song   right press: ",
                                       "press: a new instrument, empty or one of the ready ones"),
    "the left list, under its rows": (browser.LIST_HELP,),
    "the strip, a block": ("press: show a, a drag moves it, ",),
    "the strip, off its blocks": ("the patterns in the order they play",),
    "the song, its buttons": (songpanel.BLOCK_HELP, songpanel.TUNED_HELP, songpanel.DRUMS_HELP,
                              songpanel.CHORDS_HELP),
    "the channels, a heading": ("press: fold these channels away",),
    "the channels, a channel": ("press: show channel lead, twice to rename it   right press: its menu   wheel: ",),
    "the channels, under the rows": (songpanel.LIST_HELP,),
    "the roll, a note": ("press: pick ",),
    "the roll, empty grid": ("press: put a note at ",),
    "the roll, a key": ("press: hear ",),
    "the roll, the ruler": (songpanel.ROLL_HELP,),
    "the roll, the chord of a bar": ("press: pick the chord of bar ",),
    "the roll, its scroll bars": (ALONG_HELP, ACROSS_HELP),
    "the roll, the velocity lane": (VELOCITY_HELP,),
    "the steps, the squares": ("press: pick ", "press: put "),
    "the steps, the names of the drums": ("press: show the velocity of k   right press: its menu   wheel: ",),
    "the steps, off its squares": (songpanel.STEPS_HELP,),
    "the steps, its parts": (drumsteps.ALONG_HELP, drumsteps.ACROSS_HELP, VELOCITY_HELP, drumsteps.PLUS_HELP),
    "the mixer, the name of a channel": ("channel lead, lead   press: show its notes",),
    "the mixer, a fader and its toggles": ("press: drag up or down for the volume of the channel, slower with shift   "
                                           "right press: back to 1   wheel: steps it",
                                           "press: mute the channel", "press: solo the channel"),
    "the mixer, between its strips": (mixer.DESK_HELP,),
    "the layers, a switch": ("press: take group rhythm out of version beat   wheel: along the groups, shift",),
    "the layers, a group": ("press: twice to rename group rhythm   right press: its menu   wheel: along",),
    "the layers, a version": ("press: twice to rename version beat   right press: its menu   wheel: along",
                              "press: hear version beat alone   right press: its menu   wheel: along"),
    "the layers, the buttons that add": ("press: a new version", "press: a new group", "press: one more key"),
    "the layers, off all of them": (layerview.TABLE_HELP,),
    "a sound, a note": ("press: pick ",),
    "a sound, empty grid": ("press: put a note at ",),
    "a sound, a key": ("press: hear ",),
    "a sound, the ruler": ("right press: a longest time, or a note to expect   wheel: ",),
    "a sound, the wave": (soundpanel.HELP,),
    "a sound, its scroll bars": ("press: drag the bar to go along the keys, ",
                                 "press: drag the bar to go along the time, "),
    "a sound, the velocity lane": (VELOCITY_HELP,),
    "variants, the labels": ("press: pick variant ", "press: add a variant, a copy of variant "),
    "variants, the ruler": ("right press: a longest time, or a note to expect   wheel: ",),
    "parts, the labels": ("press: pick part 1, a drag moves where it starts   right press: its menu",
                          "press: add a part at the end, a copy of part "),
    "parts, the ruler": ("press: ctrl adds a part here   right press: a longest time, or a note to expect",),
    "a held sound, the labels": ("press: drag where the loop starts", "press: drag where the loop ends",
                                 "press: pick the note", "press: pick the end"),
    "a held sound, the ruler": ("press: drag the longest time   right press: a longest time, or a note to expect",
                                "right press: a longest time, or a note to expect"),
    "a step table, a step": ("press: set step 1 of ",),
    "a step table, its handle": ("press: drag right for more steps, left for fewer   wheel: along the tables",),
    "a step table, the loop under its bars": ("press: loop the table from step 1   wheel: along the tables",),
    "a step table, off its steps": (f"{instpanel.TABLES[0].help}   wheel: along the tables",),
    "the keyboard, a key": ("press: hear c3",),
    "the instrument, between its parts": (f"press: {instpanel.PAGE_HELP}",
                                          f"press: {instpanel.PAGE_HELP}   wheel: along the tables"),
    "the instrument, its scroll bar": (f"press: {instpanel.BAR_HELP}   wheel: along the tables",),
    "the wirings, a picture": (f"press: pick algorithm 0: {fmview.told(0)}   wheel: along the operators",),
    "an envelope, a point": ("press: drag the attack: ",),
    "an envelope, off its points": (f"{instpanel.ENVELOPE_HELP}   wheel: along the operators",),
    "the export, its parts": tuple(exports.HELP[part]
                                   for part in ("tabs", "stems", "change", "files", "write", "close")),
    "the export, off its parts": (exports.HELP["window"],),
    "the export, a build": tuple(exports.HELP[part] for part in ("build_format", "rate", "songs")),
    "the export, the listen page": (exports.HELP["page_format"],),
    "the export, a sound of variants": (exports.HELP["variant"],),
    "the export, a held sound": (exports.HELP["part"],),
    "the export, the box of its folder": (exports.HELP["box"],),
    "a chooser, its rows": (files.UP_HELP, files.FOLDER_HELP, files.FILE_HELP),
    "a chooser, its foot": tuple(files.PART_HELP.values()),
    "a chooser, off its parts": (files.CHOOSER_HELP,),
    "a chooser for a new file, a file": (files.OLD_FILE_HELP,),
    "the first screen, its list": ("press: open an example as a copy of your own",),
    "the title row, its buttons": tuple(topbar.BUTTON_HELP[button] for button in topbar.BUTTONS),
    "the title row, off its buttons": (topbar.TITLE_HELP,),
    "the borders between the columns": ("press: drag to give the list ", "press: drag to give the numbers "),
    "the window of a WAV file": (inspector.WAV_HELP, inspector.CLOSE_HELP),
}


def settle() -> None:
    """Waits for every render and every run of the checks that is still on a thread of a model's."""
    for thread in threading.enumerate():
        if thread.name.startswith("rowsynth-"):
            thread.join()


def refused(*args: object, **more: object) -> None:
    """Stands in for a render and for the checks: refused at once, so that none is waited for."""
    raise RowSynthError("not made in this test")


def frame(app: App) -> None:
    """Have the window take two frames: the panels have then read what is picked, and laid their widgets out."""
    for _ in range(2):
        app.update()
        app.draw(app.screen.begin())


def over(ui: Ui, pos: Pos) -> Widget | None:
    """The widget whose line the status bar has with the mouse at a position, found as `Ui.help` finds it."""
    if ui.modal is not None:
        return ui.modal if ui.modal.hit(pos) else None
    return next((widget for widget in reversed(ui.widgets) if widget.visible and widget.hit(pos)), None)


def middle(rect: layout.Rect) -> Pos:
    """The pixel in the middle of a rectangle."""
    return rect[0] + rect[2] // 2, rect[1] + rect[3] // 2


def along(rect: layout.Rect, step: int = 2) -> list[Pos]:
    """Places from side to side of a rectangle, through its middle."""
    return [(x, rect[1] + rect[3] // 2) for x in range(rect[0], rect[0] + rect[2], step)]


def across(rect: layout.Rect, step: int = 4) -> list[Pos]:
    """Places all over a rectangle, row by row from its upper left corner."""
    x, y, w, h = rect
    return [(left, top) for top in range(y, y + h, step) for left in range(x, x + w, step)]


def spot(rows: List, key: Hashable) -> Pos:
    """A place on the row of a key, as the list is scrolled now."""
    x, y, _, h = rows.rect
    return next((x + 2, down + 2) for down in range(y, y + h) if getattr(rows.at((x + 2, down)), "key", None) == key)


def under(rows: List) -> Pos:
    """A place of a list that is under its last row."""
    x, y, _, h = rows.rect
    return next((x + 2, down) for down in range(y + h - 1, y, -1) if rows.at((x + 2, down)) is None)


@unittest.skipUnless(pygame, NEEDS)
class Walked(unittest.TestCase):
    """The window walked once through its states, the mouse rested all over each: what the status bar said there.

    The states are the trailer piece with its song as notes, drums, mixer and layers, a copy of the arcade example
    with its song in each of those, a sound of each kind, an instrument and an FM instrument, the window that
    writes files on each tab, the chooser for a file to open and for a new one, and the first screen.

    A widget needs a line where the mouse can rest on it: where it is the topmost one that is seen and takes
    the mouse. A label, a meter and the picture of an intro take nothing from the mouse, so they need none: the
    line under them is that of what they stand on. Of the widgets that do take the mouse one is let through
    with no line: the sign over the count of findings in the top bar, which the tests of the shell hold to no
    line unless the checks could not run.

    No render is waited for: every render and the checks are refused at once. So the wave of a sound, the
    scope and the spectrum have nothing to read out here, and no finding has its button. What those say is
    held by the tests of their own modules. The menus are not walked either: their items say what they do.
    """

    folder: Path
    #: The widgets that said nothing, each by its state and its kind, every line that was said, and the lines
    #: said on each kind of thing of `KINDS`.
    silent: list[str]
    lines: set[str]
    found: dict[str, set[str]]

    @classmethod
    def setUpClass(cls) -> None:
        folder = tempfile.TemporaryDirectory()
        cls.addClassCleanup(folder.cleanup)
        cls.folder = Path(folder.name)
        # No thread is left to run into another test, and neither is a window or a sound output.
        cls.addClassCleanup(settle)
        cls.addClassCleanup(pygame.mixer.quit)
        cls.addClassCleanup(pygame.display.quit)
        patches = [mock.patch("pygame.display.get_desktop_sizes", return_value=[(1920, 1080)]),
                   *(mock.patch.object(owner, name, side_effect=refused)
                     for owner, name in ((gui_model, "render_stems"), (gui_model, "render_entry"), (checks, "run")))]
        for patch in patches:
            patch.start()
            cls.addClassCleanup(patch.stop)
        cls.silent, cls.lines, cls.found = [], set(), {}
        (cls.folder / "sub").mkdir()
        shutil.copyfile(EXAMPLES / "arcade.json", cls.folder / "arcade.json")
        for walk, path in ((cls.piece, EXAMPLES / "piece.json"), (cls.arcade, cls.folder / "arcade.json"),
                           (cls.first, None)):
            app = start(path)
            try:
                walk(app)
            finally:
                app.close()

    @classmethod
    def say(cls, ui: Ui, pos: Pos) -> str:
        """Rest the mouse at a position, and give the line the status bar then has. It is one of `lines`."""
        ui.handle(Event("move", pos))
        line = ui.help
        cls.lines.add(line)
        return line

    @classmethod
    def at(cls, what: str, app: App, places: Iterable[Pos]) -> None:
        """Rest the mouse at each of some places of one kind of thing, and keep what was said there."""
        cls.found.setdefault(what, set()).update(cls.say(app.ui, pos) for pos in places)

    @classmethod
    def walk(cls, app: App, state: str) -> None:
        """Rest the mouse all over the window as it is now, and note each widget that said nothing somewhere.

        The mouse rests at every `STEP` pixels each way, and on each widget that takes the mouse and has none
        of those places, at the first place where it is the topmost. With a window open over the others, it is
        that one that is walked. A widget is noted once, at the first place where it said nothing.
        """
        frame(app)
        ui = app.ui
        # The widgets that are seen, the topmost first: the one a place is on is the first of them it hits.
        shown = [ui.modal] if ui.modal is not None else [widget for widget in reversed(ui.widgets) if widget.visible]
        sign = app.panels["top"].sign if app.panels else None
        reached: set[int] = set()
        noted: set[int] = {id(sign)}

        def rest(pos: Pos) -> Widget | None:
            widget = next((widget for widget in shown if widget.hit(pos)), None)
            if widget is not None:
                reached.add(id(widget))
                if not cls.say(ui, pos) and id(widget) not in noted:
                    noted.add(id(widget))
                    cls.silent.append(f"{state}: {type(widget).__name__} at {pos}")
            return widget

        width, height = ui.size
        for pos in across((0, 0, width, height), STEP):
            rest(pos)
        for widget in shown:
            if id(widget) not in reached and widget.hit(middle(widget.rect)):
                next((pos for pos in across(widget.rect, 1) if rest(pos) is widget), None)

    @classmethod
    def piece(cls, app: App) -> None:
        """Walk the trailer piece: the notes of a tuned channel, its drums, its mixer and its layers."""
        cls.walk(app, "the piece, its notes")
        song = app.panels["middle"].panels["song"]
        cls.press(app, spot(song.list, "kick"))
        cls.walk(app, "the piece, its drums")
        for index, state in ((1, "its mixer"), (2, "its layers")):
            song.choice.pick(index)
            cls.walk(app, f"the piece, {state}")

    @staticmethod
    def press(app: App, pos: Pos) -> None:
        """Press the left button at a position and let it go there."""
        app.ui.handle(Event("down", pos, button=1))
        app.ui.handle(Event("up", pos, button=1))

    @staticmethod
    def change(app: App, where: str, value: object) -> None:
        """Put a value at a place of the project that is open, as one step. One that is refused fails the walk."""
        if not app.session.change(where, value, f"{where} for the walk"):
            raise AssertionError(f"{where} was refused: {app.session.message}")

    @classmethod
    def arcade(cls, app: App) -> None:
        """Walk the copy of the arcade example: its song, its sounds, two instruments, and the windows over them."""
        ui = app.ui
        song = app.panels["middle"].panels["song"]
        cls.walk(app, "the song, its notes")
        things = app.panels["left"].list
        cls.at("the left list, a heading", app, [spot(things, "songs")])
        cls.at("the left list, a song", app, [spot(things, ("song", "tune"))])
        cls.at("the left list, a sound", app, [spot(things, ("sound", "coin"))])
        cls.at("the left list, an instrument", app, [spot(things, ("instrument", "lead"))])
        cls.at("the left list, a row that adds", app, [spot(things, ("new", "song")),
                                                       middle(app.panels["left"].plus["instrument"].rect)])
        cls.at("the left list, under its rows", app, [under(things)])
        seen = song.strip.map
        cls.at("the strip, a block", app, [middle(seen.block(0))])
        cls.at("the strip, off its blocks", app, [next(pos for pos in across(song.strip.rect, 1)
                                                      if seen.block_at(pos) is None)])
        cls.at("the channels, a heading", app, [spot(song.list, (songpanel.TUNED,))])
        cls.at("the channels, a channel", app, [spot(song.list, "lead")])
        cls.at("the channels, under the rows", app, [under(song.list)])
        cls.at("the song, its buttons", app, [middle(song.more.rect), *(middle(plus.rect) for plus in song.list.parts),
                                              middle(song.chords.rect)])
        cls.roll(app, song.roll)
        cls.press(app, spot(song.list, "kit"))
        cls.walk(app, "the song, its drums")
        grid = song.steps.grid
        squares = [middle(grid.square(drum, row)) for drum in grid.drums_in_view() for row in grid.rows_in_view()]
        cls.at("the steps, the squares", app, [pos for pos in squares if inside(grid.area, pos)])
        cls.at("the steps, the names of the drums", app, [(song.steps.rect[0] + 2, grid.y(0) + 2)])
        cls.at("the steps, off its squares", app, [(grid.x(4) - 1, grid.y(0) + 2), (grid.x(0), song.steps.rect[1] + 2)])
        cls.at("the steps, its parts", app, [middle(grid.along), middle(grid.across), *along(song.steps.lane.rect),
                                             middle(song.steps.plus.rect)])
        song.choice.pick(1)
        cls.walk(app, "the song, its mixer")
        strip = song.mixer.strips[0]
        cls.at("the mixer, the name of a channel", app, [middle(strip.name.rect)])
        cls.at("the mixer, a fader and its toggles", app, [middle(strip.fader.rect), middle(strip.mute.rect),
                                                           middle(strip.solo.rect)])
        cls.at("the mixer, between its strips", app, [next(pos for pos in across(song.mixer.desk.rect, 1)
                                                          if over(ui, pos) is song.mixer.desk)])
        cls.change(app, "songs.tune.intro", {"order": ["a"]})
        song.choice.pick(2)
        cls.walk(app, "the song, its layers")
        table = song.layers.table
        laid = table.laid
        button, version = layerview.head_parts(laid.heads[0])
        cls.at("the layers, a switch", app, [middle(layerview.cell(laid, 0, 0))])
        cls.at("the layers, a group", app, [middle(laid.rows[0])])
        cls.at("the layers, a version", app, [middle(version), middle(button)])
        cls.at("the layers, the buttons that add", app, [middle(laid.add_version), middle(laid.add_group),
                                                         middle(song.layers.key_add.rect)])
        cls.at("the layers, off all of them", app, [next(pos for pos in across(table.rect, 1)[::-1]
                                                        if layerview.hit(laid, pos) is None)])
        cls.sounds(app)
        cls.instruments(app)
        cls.windows(app)

    @classmethod
    def roll(cls, app: App, roll: Roll) -> None:
        """Rest the mouse on each kind of thing of the piano roll, as it shows the first channel of the song."""
        seen = roll.map
        bars = [seen.note(note.row, note.rows, note.pitch) for note in roll.notes]
        x, y, w, _ = next(bar for bar in bars if inside(seen.grid, bar[:2])
                          and inside(seen.grid, (bar[0] + bar[2] - 1, bar[1] + bar[3] - 1)))
        cls.at("the roll, a note", app, [(x, y + 1), (x + w - 1, y + 1)])
        cls.at("the roll, empty grid", app, [next(pos for pos in across(seen.grid, 3) if roll.note_at(pos) is None)])
        cls.at("the roll, a key", app, [middle(seen.keys)])
        cls.at("the roll, the ruler", app, [middle(seen.ruler)])
        cls.at("the roll, the chord of a bar", app, [middle(seen.chords)])
        cls.at("the roll, its scroll bars", app, [middle(seen.along), middle(seen.across)])
        cls.at("the roll, the velocity lane", app, along(roll.lane.rect))

    @classmethod
    def sounds(cls, app: App) -> None:
        """Walk a sound of each kind, and rest the mouse on each kind of thing of the view of each."""
        session = app.session
        view = app.panels["middle"].panels["sound"].view
        cls.change(app, "sounds.hum", HUM)
        for name, kind in (("powerup", "a sound"), ("blip", "variants"), ("explosion", "parts"),
                           ("hum", "a held sound")):
            session.pick("sound", name)
            cls.walk(app, f"{kind}, {name}")
            seen = view.map
            cls.at(f"{kind}, the ruler", app, along(seen.ruler, 1))
            if name != "powerup":
                cls.at(f"{kind}, the labels", app, along(seen.names, 1))
            else:
                # The first place that is on a note, and the places of that note from there to its right end.
                x, y = next(pos for pos in across(seen.grid, 3) if view.bar_at(pos) is not None)
                on = [(right, y) for right in range(x, x + seen.grid[2])
                      if view.bar_at((right, y)) == view.bar_at((x, y))]
                cls.at("a sound, a note", app, [on[0], on[-1]])
                cls.at("a sound, empty grid", app, [next(pos for pos in across(seen.grid, 3)
                                                         if view.bar_at(pos) is None)])
                cls.at("a sound, a key", app, [middle(seen.keys)])
                cls.at("a sound, the wave", app, [middle(seen.wave)])
                cls.at("a sound, its scroll bars", app, [middle(seen.along), middle(seen.across)])
                cls.at("a sound, the velocity lane", app, along(view.lane.rect))

    @classmethod
    def instruments(cls, app: App) -> None:
        """Walk an instrument of a plain wave and one of the FM wave, and rest the mouse on the things of each."""
        session = app.session
        panel = app.panels["middle"].panels["instrument"]
        session.pick("instrument", "lead")
        cls.walk(app, "an instrument")
        table = panel.tables["vol"]
        seen = table.map
        cls.at("a step table, a step", app, [(seen.x(0), middle(seen.zone)[1])])
        cls.at("a step table, its handle", app, [middle(seen.handle(len(table.steps)))])
        cls.at("a step table, the loop under its bars", app, [(seen.x(0), middle(seen.strip)[1])])
        cls.at("a step table, off its steps", app, [(seen.zone[0] + seen.zone[2] - 1, middle(seen.zone)[1])])
        cls.at("the keyboard, a key", app, [middle(instpanel.keyboard(panel.keys.rect, panel.keys.low)[0][1])])
        cls.at("the instrument, between its parts", app, [next(pos for pos in across(panel.page.rect, 1)
                                                              if over(app.ui, pos) is panel.page),
                                                         middle(panel.page.area)[:1] + (panel.page.area[1] + 1,)])
        cls.at("the instrument, its scroll bar", app, [middle(panel.along.rect)])
        drive = json.loads((EXAMPLES / "piece.json").read_text(encoding="utf-8"))["instruments"]["fmdrive"]
        cls.change(app, "instruments.fmdrive", drive)
        session.pick("instrument", "fmdrive")
        cls.walk(app, "an FM instrument")
        cls.at("the wirings, a picture", app, [middle(fmview.cells(panel.wiring.rect, panel.wiring.between)[0])])
        envelope = panel.operators[0]["env"]
        cls.at("an envelope, a point", app, [envelope.map.spots(fmview.settings(envelope.op))[0]])
        cls.at("an envelope, off its points", app, [next(pos for pos in across(envelope.rect, 1)
                                                        if envelope.at(pos) is None)])

    @classmethod
    def windows(cls, app: App) -> None:
        """Walk the window that writes files on each of its tabs, and the chooser as it opens a file and names one."""
        session, ui = app.session, app.ui
        session.pick("song", "tune")
        cls.at("the borders between the columns", app, [middle(border.rect) for border in app.borders.values()])
        numbers = ui.open(inspector.Measured("tone.wav", inspector.table("", None)))
        cls.walk(app, "the window of a WAV file")
        at = inspector.wav_parts(ui.size, len(numbers.rows))
        cls.at("the window of a WAV file", app, [middle(at["table"]), middle(at["close"])])
        ui.close(numbers)
        app.export()
        window = ui.modal
        at = exports.parts(ui.size)
        for index, (state, parts) in enumerate((("its parts", ("tabs", "row1", "change", "list", "write", "close")),
                                                ("a build", ("row0", "row1", "row2")),
                                                ("the listen page", ("row0",)))):
            window.tabs.pick(index)
            cls.walk(app, f"the export, {state}")
            cls.at(f"the export, {state}", app, [middle(at[part]) for part in parts])
        cls.at("the export, off its parts", app, [middle(at["folder"]), middle(at["said"]), middle(at["title"])])
        window.tabs.pick(0)
        for name, state in (("blip", "a sound of variants"), ("hum", "a held sound")):
            session.pick("sound", name)
            cls.walk(app, f"the export, {state}")
            cls.at(f"the export, {state}", app, [middle(at["row1"])])
        window.change.fire()
        cls.walk(app, "the export, the box of its folder")
        cls.at("the export, the box of its folder", app, [middle(at["folder"])])
        ui.handle(Event("key", key="escape"))
        window.close()
        app.browse()
        cls.walk(app, "a chooser")
        at, rows = files.arrange(ui.size), ui.modal.list
        cls.at("a chooser, its rows", app, [spot(rows, (files.FOLDER, files.UP)), spot(rows, (files.FOLDER, "sub")),
                                            spot(rows, (files.FILE, "arcade.json"))])
        cls.at("a chooser, its foot", app, [middle(at[part]) for part in files.PART_HELP])
        cls.at("a chooser, off its parts", app, [middle(at["title"]), middle(at["folder"]), under(rows)])
        ui.handle(Event("key", key="escape"))
        app.keep_copy()
        cls.walk(app, "a chooser for a new file")
        cls.at("a chooser for a new file, a file", app, [spot(ui.modal.list, (files.FILE, "arcade.json"))])
        ui.handle(Event("key", key="escape"))

    @classmethod
    def first(cls, app: App) -> None:
        """Walk the first screen, with a file that was opened before and is gone, and each kind of row of its list."""
        gone = str(cls.folder / "gone.json")
        app.first.show(files.examples(), [str(cls.folder / "arcade.json"), gone])
        cls.walk(app, "the first screen")
        at = topbar.title_parts(app.title.rect)
        cls.at("the title row, its buttons", app, [middle(at[button]) for button in topbar.BUTTONS])
        cls.at("the title row, off its buttons", app, [middle(at["program"]), middle(app.title.rect)])
        rows = app.first.list
        cls.at("the first screen, its list", app, [spot(rows, key) for key in ("examples", ("recent", gone))])

    def test_every_widget_the_mouse_can_rest_on_says_a_line_wherever_it_rests(self) -> None:
        self.assertEqual(self.silent, [])
        # The walk went through every state, and heard far more than the lines that are written out in the code.
        self.assertGreater(len(self.lines), 200)

    def test_each_kind_of_thing_in_a_widget_of_several_has_its_line(self) -> None:
        self.assertEqual(sorted(self.found), sorted(KINDS))
        wrong = {what: sorted(self.found[what]) for what, begins in KINDS.items()
                 if "" in self.found[what] or not all(any(line.startswith(begin) for line in self.found[what])
                                                      for begin in begins)}
        self.assertEqual(wrong, {})
        # An instrument of a song's own is named with its song, as the status bar names what is picked.
        self.assertEqual(browser.row_help(Row(("instrument", "tune.soft"), "soft", tag="tune"), ("lead",)),
                         "press: show instrument soft of tune, twice to rename it   right press: its menu   wheel: "
                         "along the list")

    def test_no_line_is_wider_than_the_room_the_status_bar_has_in_the_smallest_window(self) -> None:
        # At the right end of the bar stands what is picked: the longest that the two examples can have there.
        things = [(kind, name) for example in ("arcade.json", "piece.json")
                  for section, kind in gui_model.KINDS.items()
                  for name in json.loads((EXAMPLES / example).read_text(encoding="utf-8")).get(section, {})]
        picked = max((exports.pick_words(thing) for thing in things), key=len)
        self.assertEqual(picked, "INSTRUMENT fmdrive")
        # The room is what `App._status` leaves the line: the bar less its two margins, the pick and the gap.
        room = SMALLEST - 2 * theme.BAR_MARGIN - Font.width(picked) - theme.GROUP_GAP
        self.assertGreater(room, SMALLEST // 2)
        self.assertEqual(sorted(line for line in self.lines if Font.width(line) > room), [])

    def test_a_line_begins_with_no_capital_and_ends_with_no_full_stop(self) -> None:
        self.assertEqual(sorted(line for line in self.lines if line[:1].isupper() or line.endswith(".")), [])


if __name__ == "__main__":
    unittest.main()
