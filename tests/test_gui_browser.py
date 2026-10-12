"""Tests for rowsynth.gui.browser: the instruments of a song's own in the left panel, and what their menu does."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from collections.abc import Hashable
from pathlib import Path
from unittest import mock

# No test opens a window that can be seen, or a sound device. This is said before pygame is first imported.
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from rowsynth import checks
from rowsynth.errors import RowSynthError
from rowsynth.gui import layout, theme
from rowsynth.gui import model as gui_model
from rowsynth.gui.browser import LIST_HELP, NEW, Browser, plus_rect, row_help, rows_of
from rowsynth.gui.lists import List, Row
from rowsynth.gui.popups import TextBox
from rowsynth.gui.session import Session
from rowsynth.gui.ui import Event, Ui

try:
    import pygame
except ImportError:
    pygame = None

NEEDS = "pygame is not installed"
#: The left panel on the art surface the layout is drawn for.
SIDE = (0, 22, 124, 504)
#: A project whose song has instruments of its own: one that a channel plays and a text switches to, one that
#: nothing plays, and one called as the instrument of the project is. It has a name of its own for that one too.
OWNED = {
    "instruments": {"beep": {"vol": [15, 12, 9, 6]}},
    "songs": {"tune": {"row_ticks": 3,
                       "instruments": {"lead": {"wave": "saw"}, "spare": {"wave": "tri"}, "beep": {"wave": "noise"},
                                       "soft": "beep"},
                       "channels": {"tune": {"inst": "lead"}, "low": {"inst": "soft"}},
                       "patterns": {"a": {"tune": "l4 c e @lead g > c", "low": "o3 c1"}}, "order": ["a"]}},
}


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


def settle() -> None:
    """Waits for every render and every run of the checks that is still on a thread of a model's."""
    for thread in threading.enumerate():
        if thread.name.startswith("rowsynth-"):
            thread.join()


class Rows(unittest.TestCase):
    def test_an_own_instrument_reads_as_the_name_it_has_in_its_song_and_has_the_song_for_its_tag(self) -> None:
        listing = [("songs", [("tune", "")]), ("sounds", [("tune.zap", "3")]),
                   ("instruments", [("beep", ""), ("tune.lead", "tune")])]
        rows = rows_of(listing)
        self.assertEqual(rows[-3:], [Row(("instrument", "beep"), "beep"),
                                     Row(("instrument", "tune.lead"), "lead", tag="tune"),
                                     Row((NEW, "instrument"), "new", plus=True)])
        # A sound reads as it is called, whatever its name and its tag.
        self.assertIn(Row(("sound", "tune.zap"), "tune.zap", tag="3"), rows)

    def test_the_last_row_of_each_section_adds_one_more_and_its_button_stands_over_the_room_of_the_square(
            self) -> None:
        rows = rows_of([("songs", [("tune", "")]), ("sounds", []), ("instruments", [("beep", "")])])
        self.assertEqual([(row.key, row.text) for row in rows if row.plus],
                         [((NEW, "song"), "new"), ((NEW, "sound"), "new"), ((NEW, "instrument"), "new")])
        # Each stands after the last thing of its section: after the heading in a section that holds nothing.
        self.assertEqual([row.key for row in rows],
                         ["songs", ("song", "tune"), (NEW, "song"), "sounds", (NEW, "sound"), "instruments",
                          ("instrument", "beep"), (NEW, "instrument")])
        x, y, w, h = plus_rect((2, 50, 119, theme.LIST_ROW_H))
        square = 2 + theme.LIST_PAD
        self.assertEqual(((w, h), x + w // 2, y), ((theme.PLUS, theme.PLUS), square + theme.LIST_SQUARE // 2, 50))
        self.assertLess(x + w, square + theme.LIST_SQUARE + theme.LIST_SPACE)

    def test_the_help_of_a_row_says_what_the_left_press_the_right_press_and_the_wheel_do_on_it(self) -> None:
        rows = rows_of([("songs", [("tune", "")]), ("sounds", []), ("instruments", [("tune.lead", "tune")])])
        self.assertEqual([row_help(row, ("beep",)) for row in (rows[0], rows[1], rows[2], rows[-2], rows[-1])], [
            "press: fold the songs away or bring them back   right press: a new song, sound or instrument   "
            "wheel: along the list",
            "press: show song tune, twice to rename it   right press: its menu   wheel: along the list",
            "press: a new song   right press: a new song, sound or instrument   wheel: along the list",
            "press: show instrument lead of tune, twice to rename it   right press: its menu   wheel: along the list",
            "press: a new instrument, empty or one of the ready ones   right press: a new song, sound or "
            "instrument   wheel: along the list"])
        self.assertEqual(LIST_HELP, "what the project holds   right press: a new song, sound or instrument   "
                                    "wheel: along the list")


class Owned(unittest.TestCase):
    """The left panel with a real session, on a project whose song has instruments of its own."""

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
        self.path = Path(folder.name) / "owned.json"
        self.path.write_text(json.dumps(OWNED), encoding="utf-8")
        self.session = Session(gui_model.Model(self.path))
        # The time two presses are told apart by is the test's own.
        self.now = 0.0
        self.ui = Ui(lambda: self.now)
        self.side = Browser(self.session, self.ui)
        self.side.place(SIDE)
        self.side.update()

    def held(self) -> dict:
        """What the file holds now."""
        return json.loads(self.path.read_text(encoding="utf-8"))

    def enter(self, text: str) -> None:
        """Type a text into the open box in place of what is picked there, press enter, and update the panel."""
        self.ui.handle(Event("text", text=text))
        self.ui.handle(Event("key", key="return"))
        self.side.update()

    def choose(self, key: Hashable, text: str) -> list[str | None]:
        """Press the right button on the row of a key and click an item of the menu: the texts the menu had."""
        pos = spot(self.side.list, key)
        self.ui.handle(Event("down", pos, button=3))
        self.ui.handle(Event("up", pos, button=3))
        menu = self.ui.modal
        texts = [item and item.text for item in menu.items]
        click(self.ui, middle(menu.rows[texts.index(text)]))
        self.side.update()
        return texts

    def test_make_shared_moves_an_own_instrument_to_the_project_s_in_one_step_and_picks_it_there(self) -> None:
        # The menu is in the order every menu of a named thing has. An instrument of a song's own is not copied.
        self.assertEqual(self.choose(("instrument", "tune.lead"), "Make shared"),
                         ["Rename...", "Copy", "Make shared", None, "Remove"])
        pos = spot(self.side.list, ("instrument", "tune.spare"))
        self.ui.handle(Event("down", pos, button=3))
        self.ui.handle(Event("up", pos, button=3))
        self.assertEqual([item.enabled for item in self.ui.modal.items if item], [True, False, True, True])
        self.ui.handle(Event("key", key="escape"))
        held = self.held()
        tune = held["songs"]["tune"]
        # The song plays it by the name it had, which is now the project's.
        self.assertEqual([held["instruments"], list(tune["instruments"]), tune["channels"], tune["patterns"]],
                         [{**OWNED["instruments"], "lead": {"wave": "saw"}}, ["spare", "beep", "soft"],
                          OWNED["songs"]["tune"]["channels"], OWNED["songs"]["tune"]["patterns"]])
        self.assertEqual([self.session.undo_label, self.session.message, self.session.model.picked,
                          self.side.list.picked],
                         ["instrument lead of tune made shared", "", ("instrument", "lead"), ("instrument", "lead")])
        self.assertNotIn(("instrument", "tune.lead"), [row.key for row in self.side.list.rows])
        self.assertTrue(self.session.undo())
        self.side.update()
        self.assertEqual([self.held(), self.session.can_undo], [OWNED, False])
        self.assertIn(Row(("instrument", "tune.lead"), "lead", tag="tune"), self.side.list.rows)

    def test_what_cannot_be_done_to_an_own_instrument_is_refused_and_said_and_nothing_is_changed(self) -> None:
        written = self.path.read_bytes()
        for key, text, said in (
            (("instrument", "tune.beep"), "Make shared", "the project has an instrument called 'beep' already"),
            (("instrument", "tune.lead"), "Remove", "instrument 'lead' is still used by the channel tune of tune, "
                                                    "the channel tune in the pattern a of tune"),
            # The song has an instrument of its own called as the project's is.
            (("instrument", "beep"), "Copy into this song", "the song 'tune' has an instrument called 'beep' already"),
        ):
            with self.subTest(text=text):
                self.choose(key, text)
                self.assertEqual([self.session.message, self.path.read_bytes(), self.session.can_undo],
                                 [said, written, False])

    def test_an_own_instrument_is_listed_under_the_project_s_and_picked_and_one_nothing_plays_is_removed(
            self) -> None:
        self.assertEqual(self.side.list.rows[-5:-1], (
            Row(("instrument", "beep"), "beep"), Row(("instrument", "tune.lead"), "lead", tag="tune"),
            Row(("instrument", "tune.spare"), "spare", tag="tune"),
            Row(("instrument", "tune.beep"), "beep", tag="tune")))
        click(self.ui, spot(self.side.list, ("instrument", "tune.spare")))
        self.side.update()
        self.assertEqual([self.session.model.picked, self.side.list.picked], [("instrument", "tune.spare")] * 2)
        self.choose(("instrument", "tune.spare"), "Remove")
        self.assertEqual([list(self.held()["songs"]["tune"]["instruments"]), self.held()["instruments"]],
                         [["lead", "beep", "soft"], OWNED["instruments"]])
        # The picked thing removed, the first thing there is takes its place.
        self.assertEqual([self.session.undo_label, self.session.model.picked, self.side.list.picked],
                         ["instrument spare of tune removed", ("song", "tune"), ("song", "tune")])
        # The project's instrument has the menu of a named thing, with what copies it into the song between.
        self.assertEqual(self.choose(("instrument", "beep"), "Copy"),
                         ["Rename...", "Copy", "Copy into this song", None, "Remove"])

    def test_rename_gives_an_own_instrument_another_name_in_its_song_and_nowhere_else(self) -> None:
        self.choose(("instrument", "tune.lead"), "Rename...")
        box = self.ui.modal
        self.assertEqual((box.text, box.picked, box.help),
                         ("lead", (0, 4), "the name of instrument lead of tune: enter gives it the new one, escape "
                                          "leaves it"))
        # A name the song has for another of its instruments is refused and said, and is in the box to be mended.
        written = self.path.read_bytes()
        self.enter("spare")
        self.assertEqual((self.session.message, self.ui.modal.text, self.ui.modal.picked, self.path.read_bytes()),
                         ("the song 'tune' has an instrument called 'spare' already", "spare", (5, 5), written))
        self.ui.handle(Event("key", key="a", mods=frozenset({"ctrl"})))
        self.enter("main")
        self.assertIsNone(self.ui.modal)
        held = self.held()
        tune = held["songs"]["tune"]
        # It keeps its place in the song's table, and the channel and the text that named it name it anew.
        self.assertEqual([held["instruments"], list(tune["instruments"]), tune["channels"], tune["patterns"]],
                         [OWNED["instruments"], ["main", "spare", "beep", "soft"],
                          {"tune": {"inst": "main"}, "low": {"inst": "soft"}},
                          {"a": {"tune": "l4 c e @main g > c", "low": "o3 c1"}}])
        self.assertEqual([self.session.undo_label, self.session.message, self.session.model.picked,
                          self.side.list.picked],
                         ["instrument lead of tune renamed to main", "", ("instrument", "tune.main"),
                          ("instrument", "tune.main")])
        self.assertTrue(self.session.undo())
        self.assertEqual((self.held(), self.session.can_undo), (OWNED, False))

    def test_the_second_of_two_presses_on_a_thing_opens_the_box_for_its_name_and_a_slow_one_only_picks(
            self) -> None:
        pos = spot(self.side.list, ("instrument", "tune.spare"))
        click(self.ui, pos)
        self.now += theme.DOUBLE_PRESS + 0.1
        click(self.ui, pos)
        self.assertEqual((self.ui.modal, self.session.model.picked), (None, ("instrument", "tune.spare")))
        click(self.ui, pos)
        self.assertIsInstance(self.ui.modal, TextBox)
        self.enter("extra")
        self.assertEqual((list(self.held()["songs"]["tune"]["instruments"])[1], self.session.model.picked),
                         ("extra", ("instrument", "tune.extra")))
        # Two presses on a heading fold its rows away and bring them back, and open nothing.
        for _ in range(2):
            click(self.ui, spot(self.side.list, "songs"))
        self.assertEqual((self.ui.modal, self.side.list.name_rect(("song", "tune")) is None), (None, False))


@unittest.skipUnless(pygame, NEEDS)
class Status(unittest.TestCase):
    def test_the_status_bar_names_an_own_instrument_by_its_name_and_its_song(self) -> None:
        from rowsynth.gui.app import picked_words

        self.assertEqual([picked_words(("instrument", "tune.lead")), picked_words(("instrument", "beep"), {"beep"}),
                          picked_words(("song", "a.b"))], ["INSTRUMENT lead of tune", "INSTRUMENT beep", "SONG a.b"])
        # An instrument of the project whose name was written by hand with a dot in it is called as it is.
        self.assertEqual(picked_words(("instrument", "tune.lead"), {"tune.lead"}), "INSTRUMENT tune.lead")


if __name__ == "__main__":
    unittest.main()
