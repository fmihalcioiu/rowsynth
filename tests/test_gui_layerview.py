"""Tests for rowsynth.gui.layerview: where the table stands, what it shows of a song, what it saves and hears."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from collections.abc import Sequence
from pathlib import Path

from rowsynth.gui import dials, layout, sprites, theme
from rowsynth.gui.canvas import Canvas
from rowsynth.gui.font import Font
from rowsynth.gui.layerview import (ABOUT, Group, Laid, Layers, Version, arrange, cell, head_parts, hit, intro_parts,
                                    kept, lay, more, picture)
from rowsynth.gui.model import Model
from rowsynth.gui.panels import Panel
from rowsynth.gui.popups import Menu, TextBox
from rowsynth.gui.session import Session
from rowsynth.gui.ui import Event, Ui
from rowsynth.project import Project

# pygame reads these when it is imported: no test opens a window or plays a sound.
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
try:
    import pygame
except ImportError:
    pygame = None

#: The room the part has in a song panel in the middle of the art surface.
DESK = (131, 122, 648, 397)

LAYERS = {
    "groups": {"rhythm": ["kit"], "low": ["bass"], "middle": ["arp"], "top": ["lead"]},
    "versions": {"beat": ["rhythm"], "thin": ["rhythm", "low"], "most": ["rhythm", "low", "middle"],
                 "full": ["rhythm", "low", "middle", "top"]},
    "keys": [0, 2],
}
#: A layered song of two bars, laid out as the tune of the arcade example is, and a song with no layers.
TUNE = {
    "row_ticks": 6,
    "drums": {"k": {"inst": "thud", "note": "c2"}},
    "channels": {"lead": {"inst": "tone"}, "arp": {"inst": "tone", "pan": 0.4}, "bass": {"inst": "buzz"},
                 "kit": {"inst": "thud", "drums": True}},
    "patterns": {"a": {"lead": "o5 l4 e a g e", "arp": "o4 l8 a e a e a e a e", "bass": "o2 l2 a e",
                       "kit": "l4 k k k k"},
                 "b": {"lead": "o5 l2 a g", "arp": "o4 l1 f", "bass": "o2 l1 f", "kit": "l2 k k"}},
    "order": ["a", "b"],
    "layers": LAYERS,
}
PLAIN = {"row_ticks": 6, "channels": {"air": {"inst": "tone"}, "low": {"inst": "buzz"}},
         "patterns": {"a": {"air": "o4 l1 c", "low": "o2 l1 c"}}, "order": ["a"]}
PROJECT = {
    "instruments": {"tone": {"wave": "tri"}, "thud": {"wave": "noise"}, "buzz": {"wave": "saw"}},
    "songs": {"tune": TUNE, "plain": PLAIN},
}
#: The rate the project is rendered at on the threads of its model: a low one, to be quick.
RATE = 8000


class Speaker:
    """A stand-in for the player: it keeps what it was asked, and plays nothing."""

    def __init__(self) -> None:
        self.asked: list[tuple] = []
        self.playing = False

    def play(self, sides: Sequence, loop: bool = False) -> None:
        self.asked.append(("play", sides, loop))
        self.playing = True

    def stop(self, fade: float = 0.0) -> None:
        self.asked.append(("stop",))
        self.playing = False

    def swap(self, sides: Sequence) -> None:
        self.asked.append(("swap", sides))

    def position(self) -> float:
        return 0.0

    def level(self) -> tuple[float, float]:
        return 0.0, 0.0


def settle() -> None:
    """Wait for every render and every run of the checks that is still on a thread of a model's."""
    for thread in threading.enumerate():
        if thread.name.startswith("rowsynth-"):
            thread.join()


def middle(rect: layout.Rect) -> tuple[int, int]:
    """The pixel in the middle of a rectangle."""
    return rect[0] + rect[2] // 2, rect[1] + rect[3] // 2


class Sums(unittest.TestCase):
    def test_the_table_has_a_row_a_group_and_a_column_a_version_and_a_position_is_on_the_part_it_stands_in(
            self) -> None:
        laid = lay((131, 141, 648, 84), 4, 4, 79)
        self.assertEqual((laid.left, laid.wide, laid.heads[0], laid.heads[3], laid.rows[0], laid.rows[3]),
                         (96, 64, (227, 141, 64, 14), (419, 141, 64, 14), (131, 155, 96, 14), (131, 197, 96, 14)))
        self.assertEqual((laid.add_version, laid.add_group), ((483, 141, 13, 13), (131, 211, 13, 13)))
        self.assertEqual((cell(laid, 1, 0), head_parts(laid.heads[1])),
                         ((228, 169, 62, 13), ((292, 141, 13, 13), (307, 141, 47, 13))))
        found = [hit(laid, pos) for pos in ((230, 170), (293, 142), (320, 142), (140, 200), (485, 143), (133, 213),
                                            (600, 170), (10, 10))]
        self.assertEqual(found, [("cell", 1, 0), ("play", 1), ("version", 1), ("group", 3), ("add", "version"),
                                 ("add", "group"), None, None])
        # A wide name takes its room from the columns, down to their least width: then the last are not in view.
        self.assertEqual([(held.left, held.wide, len(held.heads), len(held.rows))
                          for held in (lay((0, 0, 648, 84), 4, 4, 500), lay((0, 0, 200, 84), 4, 4, 79),
                                       lay((0, 0, 648, 44), 4, 4, 79))],
                         [(379, 64, 4, 4), (96, 30, 3, 4), (96, 64, 4, 1)])
        self.assertEqual(lay((5, 7, 100, 84), 4, 4), Laid(0, 0, (), (), (5, 7, 0, 0), (5, 7, 0, 0)))

    def test_a_table_with_no_room_for_all_of_it_shows_it_from_a_first_group_and_a_first_version_kept_in_range(
            self) -> None:
        # There is room for two groups of the five, and for three versions of the six.
        rect = (0, 0, 200, 56)
        first, laid = lay(rect, 5, 6, 79), lay(rect, 5, 6, 79, 2, 1)
        self.assertEqual((first.heads, first.rows, first.add_version, first.add_group),
                         (((96, 0, 30, 14), (126, 0, 30, 14), (156, 0, 30, 14)), ((0, 14, 96, 14), (0, 28, 96, 14)),
                          (186, 0, 13, 13), (0, 42, 13, 13)))
        # What is in view stands where the first ones stood, and is known by its place among all there are.
        self.assertEqual(laid, first._replace(first_group=2, first_version=1))
        self.assertEqual((cell(laid, 3, 2), cell(first, 1, 1)), ((127, 28, 28, 13), (127, 28, 28, 13)))
        found = [hit(laid, pos) for pos in ((130, 30), (99, 2), (170, 2), (5, 16), (188, 2), (2, 44))]
        self.assertEqual(found, [("cell", 3, 2), ("play", 1), ("version", 3), ("group", 2), ("add", "version"),
                                 ("add", "group")])
        # A first one is never before the first there is, nor further than has the last one in view.
        self.assertEqual([(held.first_group, held.first_version) for held in (
            lay(rect, 5, 6, 79, 9, 9), lay(rect, 5, 6, 79, -1, -4), lay(rect, 2, 3, 79, 1, 1))],
            [(3, 3), (0, 0), (0, 0)])
        self.assertEqual([kept(*asked) for asked in ((4, 10, 3), (9, 10, 3), (-2, 10, 3), (2, 3, 5), (2, 3, 0))],
                         [4, 7, 0, 0, 0])

    def test_a_mark_stands_at_each_edge_of_the_table_past_which_there_are_more(self) -> None:
        rect = (0, 0, 200, 56)
        self.assertEqual(more(lay(rect, 5, 6, 79, 2, 1), 5, 6),
                         {"left": (89, 6), "right": (192, 20), "up": (6, 5), "down": (19, 49)})
        self.assertEqual((more(lay(rect, 5, 6, 79), 5, 6), more(lay(rect, 5, 6, 79, 3, 3), 5, 6)),
                         ({"right": (192, 20), "down": (19, 49)}, {"left": (89, 6), "up": (6, 5)}))
        # A table that shows all of it has no mark, and neither has a rectangle with no table in it.
        self.assertEqual((more(lay(rect, 2, 3, 79), 2, 3), more(lay((5, 7, 100, 84), 4, 4), 4, 4)), ({}, {}))

    def test_the_intro_is_laid_under_the_table_and_its_picture_has_the_tail_begin_where_the_loop_does(self) -> None:
        parts = arrange(DESK, 4)
        self.assertEqual((parts["use"], parts["all"], parts["keys"], parts["table"], parts["intro"]),
                         ((131, 122, 67, 13), (204, 122, 25, 13), (272, 122, 54, 13), (131, 141, 648, 84),
                          (131, 233, 648, 37)))
        # A song with no layers has a line in the place of the table. A table too high leaves the intro its room.
        self.assertEqual((arrange(DESK, None)["about"], arrange(DESK, None)["intro"], arrange(DESK, 40)["table"],
                          arrange(DESK, 40)["intro"]),
                         ((131, 141, 648, 11), (131, 160, 648, 37), (131, 141, 648, 333), (131, 482, 648, 37)))
        self.assertEqual(intro_parts(parts["intro"]), {
            "switch": (131, 245, 37, 13), "order": (174, 245, 78, 13), "heading": (174, 233, 78, 11),
            "tail": (258, 233, 37, 37), "fade": (301, 233, 37, 37), "picture": (344, 236, 240, 31)})
        self.assertEqual(picture((10, 20, 240, 31), [16.0, 8.0], 8.0, 16.0),
                         ([(10, 20, 96, 31), (106, 20, 48, 31)], (154, 20, 48, 31), (154, 20, 96, 31)))
        self.assertEqual(picture((10, 20, 240, 31), [], 0.0, 0.0), ([], (10, 20, 0, 0), (10, 20, 0, 0)))


class Opened(unittest.TestCase):
    """A project file of the test's own, a real session on it with a stand-in player, and the part on a panel."""

    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        # No thread of the model is left to run into the next test, or into a folder that is gone.
        self.addCleanup(settle)
        self.path = Path(folder.name) / "project.json"
        self.path.write_text(json.dumps(PROJECT), encoding="utf-8")
        self.now = 100.0
        self.speaker = Speaker()
        self.session = Session(Model(self.path, RATE), self.speaker, lambda: self.now)
        self.ui = Ui()
        self.layers = Layers(Panel(self.session, self.ui), lambda: self.now)
        self.layers.place(DESK)
        self.layers.seen(True)
        self.layers.show("tune")
        self.table = self.layers.table

    def song(self, name: str = "tune") -> dict:
        """What the file holds now for a song."""
        return json.loads(self.path.read_text(encoding="utf-8"))["songs"][name]

    def click(self, pos: tuple[int, int], button: int = 1) -> None:
        """Press a mouse button at a position and let it go there."""
        self.ui.handle(Event("down", pos, button=button))
        self.ui.handle(Event("up", pos, button=button))

    def choose(self, text: str) -> None:
        """Click the item of the open menu that has a text."""
        menu = self.ui.modal
        self.click(middle(menu.rows[[item and item.text for item in menu.items].index(text)]))

    def type(self, text: str) -> None:
        """Type a text into the open box, in the place of what is picked there, and press enter."""
        self.ui.handle(Event("text", text=text))
        self.ui.handle(Event("key", key="return"))

    def seen(self) -> list[object]:
        """The widgets of the part that are on the window."""
        return [widget for widget in self.layers.widgets if widget.visible]


class Shown(Opened):
    def test_the_switches_of_the_table_follow_the_file_and_a_press_on_one_is_saved_and_taken_back(self) -> None:
        layers, table = self.layers, self.table
        self.assertEqual(table.groups, (
            Group("rhythm", (("kit", theme.MUSTARD),)), Group("low", (("bass", theme.SALMON),)),
            Group("middle", (("arp", theme.SOUND),)), Group("top", (("lead", theme.SONG),))))
        self.assertEqual(table.versions, tuple(Version(name, frozenset(named))
                                               for name, named in LAYERS["versions"].items()))
        self.assertEqual((table.rect, table.laid, layers.keys.options, layers.use.is_on),
                         (arrange(DESK, 4)["table"], lay(arrange(DESK, 4)["table"], 4, 4, 79), ("0", "+2"), True))
        self.assertEqual(self.seen(), [layers.use, layers.all, layers.keys_word, layers.keys, layers.key_add,
                                       layers.key_drop, table, layers.intro])
        self.ui.handle(Event("move", middle(table.cell("top", "beat"))))
        self.assertEqual(self.ui.help, "group top in version beat: a click puts it in or takes it out")
        self.click(middle(table.cell("top", "beat")))
        self.assertEqual((self.song()["layers"]["versions"]["beat"], self.session.undo_label, table.versions[0]),
                         (["rhythm", "top"], "tune: group top in version beat",
                          Version("beat", frozenset({"rhythm", "top"}))))
        self.click(middle(table.cell("low", "full")))
        self.assertEqual((self.song()["layers"]["versions"]["full"], self.session.undo_label),
                         (["rhythm", "middle", "top"], "tune: group low out of version full"))
        # Each press is one step: taken back, the file is as it was, and the table shows it at its next frame.
        self.assertTrue(self.session.undo() and self.session.undo())
        layers.update()
        self.assertEqual((self.song(), table.versions[0], self.session.can_undo),
                         (TUNE, Version("beat", frozenset({"rhythm"})), False))

    def test_the_button_of_a_version_has_its_channels_heard_alone_and_the_song_played_and_all_gives_the_rest_back(
            self) -> None:
        session, model, table = self.session, self.session.model, self.table
        session.tick()
        settle()
        session.tick()
        self.click(middle(table.play("thin")))
        self.assertEqual((model.heard, session.soloed, session.playing),
                         (frozenset({"kit", "bass"}), frozenset({"kit", "bass"}), True))
        settle()
        session.tick()
        self.layers.update()
        # What plays is the mix of those two stems, which the model made of the song's render with none made again.
        self.assertEqual(([asked[0] for asked in self.speaker.asked], self.speaker.asked[-1][2], table.heard),
                         (["play"], True, frozenset({"thin"})))
        self.assertIs(self.speaker.asked[-1][1], model.render()["sides"])
        # The button of another version changes over while the song goes round: it is not played from its start.
        self.click(middle(table.play("beat")))
        settle()
        session.tick()
        self.layers.update()
        self.assertEqual((model.heard, [asked[0] for asked in self.speaker.asked], table.heard),
                         (frozenset({"kit"}), ["play", "swap"], frozenset({"beat"})))
        self.click(middle(self.layers.all.rect))
        self.layers.update()
        self.assertEqual((model.heard, session.soloed, table.heard, self.song()),
                         (None, frozenset(), frozenset(), TUNE))

    def test_the_channels_of_a_group_are_picked_in_the_list_its_menu_opens(self) -> None:
        self.click(middle(self.table.name_rect("group", "low")), button=3)
        self.assertIsInstance(self.ui.modal, Menu)
        self.assertEqual([item and item.text for item in self.ui.modal.items],
                         ["Rename...", "Channels...", None, "Remove"])
        self.choose("Channels...")
        self.assertEqual([(item.text, item.icon) for item in self.ui.modal.items],
                         [("lead (top)", ""), ("arp (middle)", ""), ("bass", "tick"), ("kit (rhythm)", "")])
        # A channel that is picked leaves the group it was in, and the list is open again as the file then has it.
        self.choose("arp (middle)")
        self.assertEqual((self.song()["layers"]["groups"], self.session.undo_label),
                         ({"rhythm": ["kit"], "low": ["bass", "arp"], "middle": [], "top": ["lead"]},
                          "tune: arp in group low"))
        self.assertEqual([(item.text, item.icon) for item in self.ui.modal.items],
                         [("lead (top)", ""), ("arp", "tick"), ("bass", "tick"), ("kit (rhythm)", "")])
        self.choose("bass")
        self.assertEqual((self.song()["layers"]["groups"]["low"], self.session.undo_label, self.table.groups[1]),
                         (["arp"], "tune: bass out of group low", Group("low", (("arp", theme.SOUND),))))
        self.ui.handle(Event("key", key="escape"))
        self.assertIsNone(self.ui.modal)

    def test_a_version_and_a_group_are_added_by_the_buttons_and_a_version_is_renamed_and_removed_by_its_menu(
            self) -> None:
        self.click(middle(self.table.laid.add_version))
        self.click(middle(self.table.laid.add_group))
        layers = self.song()["layers"]
        self.assertEqual((list(layers["versions"].items())[-1], list(layers["groups"].items())[-1],
                          self.session.undo_label), (("version", []), ("group", []), "tune: new group group"))
        self.click(middle(self.table.name_rect("version", "version")), button=3)
        self.assertEqual([item and item.text for item in self.ui.modal.items],
                         ["Rename...", None, "Remove", None, "premix in 0", "premix in +2"])
        self.choose("Rename...")
        self.assertIsInstance(self.ui.modal, TextBox)
        self.type("calm")
        self.assertEqual((list(self.song()["layers"]["versions"])[-1], self.session.undo_label, self.ui.modal),
                         ("calm", "tune: version version renamed to calm", None))
        # A name that is taken is refused: the file stays, and the session says why.
        self.click(middle(self.table.name_rect("version", "calm")), button=3)
        self.choose("Rename...")
        self.type("full")
        self.assertEqual((list(self.song()["layers"]["versions"])[-1], self.session.message),
                         ("calm", "the layers have a version called 'full' already"))
        self.click(middle(self.table.name_rect("version", "calm")), button=3)
        self.choose("Remove")
        self.assertEqual((list(self.song()["layers"]["versions"]), [version.name for version in self.table.versions]),
                         (list(LAYERS["versions"]), list(LAYERS["versions"])))

    def test_a_version_is_premixed_in_a_key_by_its_menu_and_the_help_of_its_heading_says_in_which(self) -> None:
        table = self.table
        heading = middle(table.name_rect("version", "thin"))
        self.click(heading, button=3)
        self.assertEqual([item and (item.text, item.icon) for item in self.ui.modal.items],
                         [("Rename...", ""), None, ("Remove", ""), None, ("premix in 0", ""), ("premix in +2", "")])
        self.choose("premix in +2")
        self.assertEqual((self.song()["layers"]["premix"], self.session.undo_label, table.versions[1].premix),
                         ([{"version": "thin", "key": 2}], "tune: premix of thin in +2 on", (2,)))
        self.ui.handle(Event("move", heading))
        self.assertEqual(self.ui.help, "version thin, premixed in +2: the right button renames or removes it, or "
                                       "premixes it in a key")
        # The key it is premixed in has a tick. Chosen again it is taken out, and the last one out leaves no list.
        self.click(heading, button=3)
        self.assertEqual([item.icon for item in self.ui.modal.items[-2:]], ["", "tick"])
        self.choose("premix in +2")
        self.assertEqual((self.song(), self.session.undo_label, table.versions[1].premix),
                         (TUNE, "tune: premix of thin in +2 off", ()))

    def test_a_table_with_no_room_for_all_of_it_is_moved_by_the_wheel_and_what_is_out_of_view_has_no_rectangle(
            self) -> None:
        layers, table = self.layers, self.table
        layers.place((131, 122, 200, 110))
        rows, heads = table.laid.rows, table.laid.heads
        nothing = (131, 141, 0, 0)
        # One group of the four is in view, and three versions.
        self.assertEqual((len(rows), len(heads), table.cell("low", "beat"), table.play("full"),
                          table.name_rect("group", "low"), table.name_rect("version", "full")),
                         (1, 3, nothing, nothing, table.rect, table.rect))
        # The wheel moves the groups by a row a step, and over the headings the versions by a column.
        self.assertTrue(self.ui.handle(Event("wheel", middle(rows[0]), wheel=-1)))
        self.assertTrue(self.ui.handle(Event("wheel", middle(heads[2]), wheel=-1)))
        self.assertEqual((table.first_group, table.first_version, table.laid.rows, table.laid.heads, self.ui.help),
                         (1, 1, rows, heads, "version full: the right button renames or removes it, or premixes it in "
                                             "a key"))
        # In view now are the group low and the versions from thin on, each where the first ones stood.
        self.assertEqual((table.cell("low", "thin"), table.play("full"), table.name_rect("group", "low")[:2],
                          table.cell("rhythm", "thin"), table.play("beat"), table.name_rect("version", "beat")),
                         ((heads[0][0] + 1, rows[0][1], 28, 13), head_parts(heads[2])[0],
                          (rows[0][0] + theme.PAD, rows[0][1]), nothing, nothing, table.rect))
        # A press on a switch is a press on the group and the version that are under it.
        self.click(middle(table.cell("low", "full")))
        self.assertEqual((self.song()["layers"]["versions"]["full"], table.first_group, table.first_version),
                         (["rhythm", "middle", "top"], 1, 1))
        # The view stops at the last, and with fewer to show it goes back as far as it takes to be in range.
        self.ui.handle(Event("wheel", middle(heads[0]), wheel=-5))
        self.ui.handle(Event("wheel", middle(rows[0]), wheel=-9))
        self.assertEqual((table.first_group, table.first_version), (3, 1))
        table.show(table.groups[:2], table.versions[:3])
        self.assertEqual((table.first_group, table.first_version), (1, 0))
        # Where all of them are in view the wheel is left to what lies under the table.
        self.assertFalse(self.ui.handle(Event("wheel", middle(heads[0]), wheel=-1)))
        self.assertTrue(self.ui.handle(Event("wheel", middle(rows[0]), wheel=2)))
        self.assertEqual(table.first_group, 0)

    def test_a_key_that_is_typed_is_added_and_shown_and_the_one_that_is_shown_is_removed(self) -> None:
        layers = self.layers
        self.click(middle(layers.key_add.rect))
        self.type("-3")
        self.assertEqual((self.song()["layers"]["keys"], layers.keys.options, layers.keys.picked,
                          self.session.undo_label), ([0, 2, -3], ("0", "+2", "-3"), 2, "tune: key -3 added"))
        self.click(middle(layers.key_drop.rect))
        self.assertEqual((self.song()["layers"]["keys"], layers.keys.options, self.session.undo_label),
                         ([0, 2], ("0", "+2"), "tune: key -3 removed"))
        self.click(middle(layers.key_add.rect))
        self.type("2")
        self.assertEqual((self.song()["layers"]["keys"], self.session.message),
                         ([0, 2], "the song is heard in key 2 already"))

    def test_a_press_on_the_keys_opens_a_menu_to_hear_the_song_in_one_of_them_which_the_field_then_shows(
            self) -> None:
        layers, session, model = self.layers, self.session, self.session.model
        self.click(middle(self.table.play("thin")))
        settle()
        session.tick()
        self.click(middle(layers.keys.rect))
        self.assertIsInstance(self.ui.modal, Menu)
        self.assertEqual(([(item.text, item.icon) for item in self.ui.modal.items], self.ui.modal.rect[:2]),
                         ([("hear in 0", "tick"), ("hear in +2", "")], (272, 135)))
        self.choose("hear in +2")
        self.assertEqual((session.shift, layers.keys.picked, self.ui.modal, self.song()), (2, 1, None, TUNE))
        # The loop goes on in that key when the render of it is there, the channels of the version alone as before.
        settle()
        session.tick()
        self.assertEqual(([asked[0] for asked in self.speaker.asked], model.heard),
                         (["play", "swap"], frozenset({"kit", "bass"})))
        self.assertIs(self.speaker.asked[-1][1], model.render()["sides"])
        self.click(middle(layers.keys.rect))
        self.assertEqual([item.icon for item in self.ui.modal.items], ["", "tick"])
        self.ui.handle(Event("key", key="escape"))
        # While it is another key than the first, the key that is heard stays the one shown, whatever the wheel does.
        self.assertTrue(self.ui.handle(Event("wheel", middle(layers.keys.rect), wheel=-1)))
        layers.update()
        self.assertEqual(layers.keys.picked, 1)
        # The button of a version and ALL leave the key as it is.
        self.click(middle(self.table.play("beat")))
        self.click(middle(layers.all.rect))
        self.assertEqual((session.shift, model.heard), (2, None))
        # Heard in its first key again, the song has the field free to show any of its keys.
        self.click(middle(layers.keys.rect))
        self.choose("hear in 0")
        self.assertEqual((session.shift, layers.keys.picked), (0, 0))
        self.ui.handle(Event("wheel", middle(layers.keys.rect), wheel=1))
        layers.update()
        self.assertEqual((session.shift, layers.keys.picked), (0, 1))

    def test_a_key_that_goes_from_the_song_while_the_song_is_heard_in_it_gives_way_to_the_first_key(self) -> None:
        layers, session = self.layers, self.session
        self.click(middle(layers.keys.rect))
        self.choose("hear in +2")
        self.click(middle(layers.key_drop.rect))
        self.assertEqual((self.song()["layers"].get("keys"), session.undo_label, session.shift, layers.keys.options),
                         (None, "tune: key +2 removed", 0, ("0",)))
        # The first key is one that can go as well: the song is then heard in the key that is the first after it.
        self.assertTrue(session.undo())
        layers.update()
        self.assertEqual((layers.keys.options, layers.keys.picked, session.shift), (("0", "+2"), 0, 0))
        self.click(middle(layers.key_drop.rect))
        self.assertEqual((self.song()["layers"]["keys"], session.shift, layers.keys.options, layers.keys.picked),
                         ([2], 2, ("+2",), 0))
        # Another song that is shown has lost no key: what is heard stays in the key it was heard in.
        layers.show("plain")
        self.assertEqual((session.shift, session.message), (2, ""))

    def test_another_song_is_seen_from_the_top_left_and_what_is_added_to_a_table_is_brought_into_view(self) -> None:
        layers, table = self.layers, self.table
        self.assertTrue(self.session.change("songs.twin", TUNE, "twin added"))
        # One group of the four is in view, and three versions.
        layers.place((131, 122, 200, 110))
        self.ui.handle(Event("wheel", middle(table.laid.rows[0]), wheel=-2))
        self.ui.handle(Event("wheel", middle(table.laid.heads[2]), wheel=-1))
        self.assertEqual((table.first_group, table.first_version), (2, 1))
        layers.show("twin")
        self.assertEqual((table.first_group, table.first_version, [group.name for group in table.groups]),
                         (0, 0, list(LAYERS["groups"])))
        # A version and a group that are added are the last of them, which the table then has in view.
        nothing = (table.rect[0], table.rect[1], 0, 0)
        self.click(middle(table.laid.add_version))
        self.assertEqual((table.first_version, len(table.laid.heads), table.versions[-1].name), (2, 3, "version"))
        self.assertNotEqual(table.play("version"), nothing)
        self.click(middle(table.laid.add_group))
        self.assertEqual((table.first_group, len(table.laid.rows), table.groups[-1].name), (4, 1, "group"))
        self.assertNotEqual(table.cell("group", "version"), nothing)
        # One that is added to a table that shows all of it leaves the table where it is.
        layers.place(DESK)
        self.click(middle(table.laid.add_group))
        self.assertEqual((table.first_group, table.first_version, len(table.laid.rows)), (0, 0, 6))

    def test_the_intro_is_switched_on_its_tail_and_its_patterns_are_set_and_it_is_switched_off(self) -> None:
        layers = self.layers
        self.assertEqual((layers.intro.is_on, layers.tail.visible, layers.picture.visible), (False, False, False))
        self.click(middle(layers.intro.rect))
        self.assertEqual((self.song()["intro"], self.session.undo_label, layers.intro.is_on, layers.order.text),
                         ({"order": ["a"]}, "tune: intro on", True, "a"))
        inner = intro_parts(arrange(DESK, 4)["intro"])
        self.assertEqual((layers.tail.rect, layers.fade.rect, layers.picture.rect, self.seen()[-5:]),
                         (inner["tail"], inner["fade"], inner["picture"],
                          [layers.heading, layers.order, layers.tail, layers.fade, layers.picture]))
        # The knob of the tail goes in hundredths of a second up to a bar of the song, which is 1.6 seconds here.
        self.assertEqual((layers.tail.value, layers.tail.high, layers.tail.steps, layers.fade.value),
                         (1.5, 1.6, 160, 0.05))
        self.assertEqual((layers.picture.names, layers.picture.rows, layers.picture.first, layers.picture.first_rows,
                          layers.picture.tail), (("a",), (16.0,), "a", 16.0, 15.0))
        self.ui.handle(Event("wheel", middle(layers.tail.rect), wheel=-10))
        self.assertNotIn("tail", self.song()["intro"])
        self.now += dials.WHEEL_WAIT
        layers.update()
        self.assertEqual((self.song()["intro"], self.session.undo_label, layers.picture.tail),
                         ({"order": ["a"], "tail": 1.4}, "tune: intro tail 1.4 s", 14.0))
        self.click(middle(layers.order.rect))
        self.assertEqual([item and (item.text, item.enabled) for item in self.ui.modal.items],
                         [("Add a at the end", True), ("Add b at the end", True), None, ("Take the last out", False)])
        self.choose("Add b at the end")
        self.assertEqual((self.song()["intro"]["order"], layers.order.text, layers.picture.names,
                          self.session.undo_label), (["a", "b"], "a b", ("a", "b"), "tune: b added to the intro"))
        self.click(middle(layers.intro.rect))
        self.assertEqual(("intro" in self.song(), self.session.undo_label, layers.tail.visible),
                         (False, "tune: intro off", False))

    def test_a_song_with_no_layers_has_the_line_and_the_switch_which_gives_it_layers_the_project_accepts(
            self) -> None:
        layers = self.layers
        layers.show("plain")
        self.assertEqual((self.seen(), layers.about.text, layers.about.rect, layers.use.is_on),
                         ([layers.use, layers.about, layers.intro], ABOUT, arrange(DESK, None)["about"], False))
        self.click(middle(layers.use.rect))
        self.assertEqual((Project.load(self.path).songs["plain"]["layers"], self.session.undo_label),
                         ({"groups": {"all": ["air", "low"]}, "versions": {"full": ["all"]}}, "plain: layers on"))
        self.assertEqual((layers.use.is_on, layers.about.visible, self.table.visible, self.table.groups,
                          self.table.versions),
                         (True, False, True, (Group("all", (("air", theme.SONG), ("low", theme.SOUND))),),
                          (Version("full", frozenset({"all"})),)))
        # Switched off again, the song is as it began. Off the window nothing of the part is seen.
        self.click(middle(layers.use.rect))
        self.assertEqual((self.song("plain"), self.seen()), (PLAIN, [layers.use, layers.about, layers.intro]))
        layers.seen(False)
        self.assertEqual(self.seen(), [])


@unittest.skipUnless(pygame, "pygame is not installed")
class Drawn(Opened):
    def test_a_switch_is_lit_where_a_group_is_in_a_version_and_the_tail_of_the_intro_falls_into_the_loop(self) -> None:
        surface = pygame.Surface((960, 540))
        surface.fill(theme.PANEL)
        self.click(middle(self.layers.intro.rect))
        self.ui.draw(Canvas(surface, Font(sprites.load("font")), sprites.load("icons")))

        def ink(rect: layout.Rect, colour: theme.Colour) -> int:
            x, y, w, h = rect
            return sum(surface.get_at((x + column, y + row))[:3] == colour for column in range(w) for row in range(h))

        table = self.table
        lit = [[bool(ink(table.cell(group, version), theme.ACTIVE)) for version in LAYERS["versions"]]
               for group in LAYERS["groups"]]
        self.assertEqual(lit, [[True, True, True, True], [False, True, True, True], [False, False, True, True],
                               [False, False, False, True]])
        # Each group is named with the square of its channel after its name, in the channel's colour.
        self.assertEqual([ink(row, colour) for row, colour in zip(table.laid.rows, (theme.MUSTARD, theme.SALMON,
                                                                                   theme.SOUND, theme.SONG))],
                         [theme.LIST_SQUARE ** 2] * 4)
        # The slope of the tail is highest where the loop begins and gone where the tail ends, 15 rows of 16 on.
        blocks, loop, tail = picture(self.layers.picture.rect, (16.0,), 16.0, 15.0)
        slope = theme.shade(theme.ACTIVE, theme.TRAVEL)
        x, y, w, h = tail
        self.assertEqual((loop[0], blocks[0][0] + blocks[0][2], w), (x, x, 113))
        self.assertGreater(ink((x + 1, y, 1, h), slope), ink((x + w // 2, y, 1, h), slope))
        self.assertEqual((ink((x + w // 2, y, 1, h), slope), ink((x + w + 1, y, 1, h), slope)), (14, 0))

    def test_a_small_arrow_is_drawn_at_each_edge_past_which_the_table_has_more_and_none_where_all_is_in_view(
            self) -> None:
        surface = pygame.Surface((960, 540))
        canvas = Canvas(surface, Font(sprites.load("font")), sprites.load("icons"))
        table = self.table

        def arrows() -> tuple[int, ...]:
            """The pixels in the colour of an arrow in the corner, after the first row, and beside the lower button."""
            surface.fill(theme.PANEL)
            self.ui.draw(canvas)
            laid = table.laid
            (right, top), (left, foot) = laid.add_version[:2], laid.add_group[:2]
            return tuple(sum(surface.get_at((x + column, y + row))[:3] == theme.DIM
                             for column in range(w) for row in range(h))
                         for x, y, w, h in ((left, top, laid.left, 14), (right, top + 14, 13, 13),
                                            (left + 13, foot, 13, 13)))

        self.assertEqual(arrows(), (0, 0, 0))
        # An arrow of three columns has nine pixels. Of four groups one is in view, and of four versions three.
        self.layers.place((131, 122, 200, 110))
        self.assertEqual(arrows(), (0, 9, 9))
        self.ui.handle(Event("wheel", middle(table.laid.rows[0]), wheel=-1))
        self.ui.handle(Event("wheel", middle(table.laid.heads[0]), wheel=-1))
        self.assertEqual(arrows(), (18, 0, 9))


if __name__ == "__main__":
    unittest.main()
