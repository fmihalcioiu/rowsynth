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
                                    lay, picture)
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
        self.assertEqual([item and item.text for item in self.ui.modal.items], ["Rename...", None, "Remove"])
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


if __name__ == "__main__":
    unittest.main()
