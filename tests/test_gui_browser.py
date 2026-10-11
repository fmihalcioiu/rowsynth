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
from rowsynth.gui import layout
from rowsynth.gui import model as gui_model
from rowsynth.gui.browser import Browser, rows_of
from rowsynth.gui.lists import List, Row
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
        self.assertEqual(rows[-2:], [Row(("instrument", "beep"), "beep"),
                                     Row(("instrument", "tune.lead"), "lead", tag="tune")])
        # A sound reads as it is called, whatever its name and its tag.
        self.assertIn(Row(("sound", "tune.zap"), "tune.zap", tag="3"), rows)


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
        self.ui = Ui()
        self.side = Browser(self.session, self.ui)
        self.side.place(SIDE)
        self.side.update()

    def held(self) -> dict:
        """What the file holds now."""
        return json.loads(self.path.read_text(encoding="utf-8"))

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
        self.assertEqual(self.choose(("instrument", "tune.lead"), "Make shared"), ["Make shared", None, "Delete"])
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
            (("instrument", "tune.lead"), "Delete", "instrument 'lead' is still used by the channel tune of tune, "
                                                    "the channel tune in the pattern a of tune"),
        ):
            with self.subTest(text=text):
                self.choose(key, text)
                self.assertEqual([self.session.message, self.path.read_bytes(), self.session.can_undo],
                                 [said, written, False])

    def test_an_own_instrument_is_listed_under_the_project_s_and_picked_and_one_nothing_plays_is_deleted(
            self) -> None:
        self.assertEqual(self.side.list.rows[-4:], (
            Row(("instrument", "beep"), "beep"), Row(("instrument", "tune.lead"), "lead", tag="tune"),
            Row(("instrument", "tune.spare"), "spare", tag="tune"),
            Row(("instrument", "tune.beep"), "beep", tag="tune")))
        click(self.ui, spot(self.side.list, ("instrument", "tune.spare")))
        self.side.update()
        self.assertEqual([self.session.model.picked, self.side.list.picked], [("instrument", "tune.spare")] * 2)
        self.choose(("instrument", "tune.spare"), "Delete")
        self.assertEqual([list(self.held()["songs"]["tune"]["instruments"]), self.held()["instruments"]],
                         [["lead", "beep", "soft"], OWNED["instruments"]])
        # The picked thing deleted, the first thing there is takes its place.
        self.assertEqual([self.session.undo_label, self.session.model.picked, self.side.list.picked],
                         ["instrument spare of tune deleted", ("song", "tune"), ("song", "tune")])
        # The project's instrument has the menu it had.
        self.assertEqual(self.choose(("instrument", "beep"), "Copy"), ["Rename...", "Copy", None, "Delete"])


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
