"""Tests for rowsynth.page."""
from __future__ import annotations

import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

import numpy as np

from rowsynth.bank import render_entry
from rowsynth.errors import RowSynthError
from rowsynth.measure import measure
from rowsynth.page import write_page
from rowsynth.project import Project
from rowsynth.wav import load_wav

# A low rate keeps the rendering short. A tick is 133 and a third samples.
RATE = 8000
# The size of one step of a 16-bit file.
STEP = 1 / 32767

INSTRUMENTS = {
    "reed": {"wave": "saw", "vol": [14, 10, 6], "release": 1},
    "tick": {"wave": "noise", "vol": [12, 5], "release": 0},
}


def note(inst: dict | str, pitch: str | float, ticks: float = 3) -> dict:
    """A def of one voice playing one note from the start."""
    return {"voices": [{"inst": inst, "notes": [[0, pitch, ticks]]}]}


# A sound of each kind, one with a name that HTML cannot hold as it is, and one that is silent. With its
# release and its tail, a note of the reed held for 3 ticks is 6 ticks long: a tenth of a second.
SOUNDS = {
    "chime": {"def": note("reed", "e5")},
    "steps": {"defs": [note("reed", "c4"), note("reed", "g4")], "db": [-2, -5]},
    "thump": {"parts": [[0, {"def": note("tick", 40)}], [0.05, {"def": note("reed", "c3"), "db": -6}]]},
    "hum": {"held": {"note": note("reed", "a2", 10), "start": 3, "loop": 4, "join": 1, "end": note("reed", "a2", 2),
                     "let_go": 2}, "db": -4},
    "a&b": {"def": note("tick", 60)},
    "hush": {"def": {"voices": []}},
}

# The files of SOUNDS in the order of the project: the file, its sound, and which one of that sound it is.
FILES = [
    ("chime.wav", "chime", ""),
    ("steps-0.wav", "steps", "variant 0"),
    ("steps-1.wav", "steps", "variant 1"),
    ("thump.wav", "thump", ""),
    ("hum-start.wav", "hum", "start"),
    ("hum-loop.wav", "hum", "loop"),
    ("hum-end.wav", "hum", "end"),
    ("a&b.wav", "a&b", ""),
    ("hush.wav", "hush", ""),
]


def project_of(**sounds: dict) -> Project:
    """A project with the instruments above and these sounds."""
    return Project.from_dict({"instruments": INSTRUMENTS, "sounds": sounds})


class Page(HTMLParser):
    """A page as it parses: its tags, its text, what it loads, and every row of a table that has a player.

    A row is {"name", "which", "player", "figures"}: the text of its heading outside and inside <small>, the
    attributes of its <audio>, and the text of each cell that has a data-label, by that label.
    """

    def __init__(self, text: str) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.text: list[str] = []
        self.loaded: list[tuple[str, str]] = []
        self.styled: list[str] = []
        self.rows: list[dict] = []
        self._row: dict = {}
        self._inside: list[str] = []
        self._label: str | None = None
        self.feed(text)
        self.close()

    def handle_starttag(self, tag: str, attrs: list) -> None:
        self.tags.append(tag)
        given = dict(attrs)
        self.loaded += [(tag, given[key]) for key in ("src", "href") if key in given]
        if "style" in given:
            self.styled.append(tag)
        if tag == "tr":
            self._row = {"name": "", "which": "", "player": None, "figures": {}}
        elif tag == "td":
            self._label = given.get("data-label")
        elif tag == "audio":
            self._row["player"] = given
        if tag in ("th", "td", "small"):
            self._inside.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("th", "td", "small"):
            self._inside.pop()
        elif tag == "tr" and self._row["player"] is not None:
            self._row["name"] = self._row["name"].strip()
            self.rows.append(self._row)

    def handle_data(self, data: str) -> None:
        self.text.append(data)
        inside = self._inside[-1] if self._inside else ""
        if inside == "small":
            self._row["which"] += data
        elif inside == "th":
            self._row["name"] += data
        elif inside == "td" and self._label:
            figures = self._row["figures"]
            figures[self._label] = figures.get(self._label, "") + data


def read(path: str | Path) -> Page:
    """The page in a file, parsed."""
    return Page(Path(path).read_text(encoding="utf-8"))


def sources(page: Page) -> list[str]:
    """The address of each player of a page, as it is written."""
    return [row["player"]["src"] for row in page.rows]


class WritePageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """One page of SOUNDS for the tests that only read it, in a folder that is not there beforehand."""
        folder = tempfile.TemporaryDirectory()
        cls.addClassCleanup(folder.cleanup)
        cls.project = project_of(**SOUNDS)
        cls.out = Path(folder.name, "site")
        cls.written = write_page(cls.project, cls.out, RATE)
        cls.raw = (cls.out / "index.html").read_bytes()
        cls.page = Page(cls.raw.decode("utf-8"))
        cls.buffers = [samples for name in SOUNDS for samples in render_entry(cls.project.sound(name), RATE)]

    def folder(self) -> Path:
        """A folder of the test's own, gone when the test is over."""
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        return Path(folder.name)

    def test_what_is_returned_is_the_path_of_the_page_and_of_each_audio_file_in_the_order_of_the_rows(self):
        self.assertEqual(self.written, {"page": str(self.out / "index.html"),
                                        "files": [str(self.out / "audio" / file) for file, _, _ in FILES]})
        for path in (self.written["page"], *self.written["files"]):
            self.assertTrue(Path(path).is_file(), path)

    def test_each_buffer_of_each_kind_of_entry_is_a_mono_file_under_its_name_and_reads_back_within_a_step(self):
        self.assertEqual(sorted(path.name for path in (self.out / "audio").iterdir()),
                         sorted(file for file, _, _ in FILES))
        self.assertEqual(len(self.buffers), len(FILES))
        for (file, _, _), samples in zip(FILES, self.buffers):
            channels, rate = load_wav(self.out / "audio" / file)
            self.assertEqual((len(channels), rate, len(channels[0])), (1, RATE, len(samples)), file)
            self.assertLessEqual(float(np.abs(channels[0] - samples).max()), STEP, file)

    def test_the_page_has_one_player_for_each_file_in_the_order_of_the_project(self):
        self.assertEqual(self.page.tags.count("audio"), len(FILES))
        self.assertEqual([unquote(src) for src in sources(self.page)], [f"audio/{file}" for file, _, _ in FILES])
        for row in self.page.rows:
            player = row["player"]
            self.assertIn("controls", player)
            self.assertEqual(player["preload"], "none")
            # The address leads from the page to the file.
            self.assertTrue((self.out / unquote(player["src"])).is_file(), player["src"])

    def test_only_the_loop_of_a_held_sound_goes_round(self):
        self.assertEqual([row["player"]["src"] for row in self.page.rows if "loop" in row["player"]],
                         ["audio/hum-loop.wav"])

    def test_a_row_names_its_sound_and_says_which_one_of_it_the_file_is(self):
        self.assertEqual([(row["name"], row["which"]) for row in self.page.rows],
                         [(name, which) for _, name, which in FILES])

    def test_a_name_is_escaped_as_text_and_coded_in_the_address(self):
        self.assertIn(b">a&amp;b<", self.raw)
        self.assertNotIn(b"a&b", self.raw)
        self.assertEqual(sources(self.page)[7], "audio/a%26b.wav")

    def test_a_name_outside_ascii_is_written_by_its_number_and_still_leads_to_its_file(self):
        out = self.folder()
        raw = Path(write_page(project_of(**{"caf\u00e9": {"def": note("tick", 60)}}), out, RATE)["page"]).read_bytes()
        self.assertTrue(raw.isascii())
        self.assertIn(b">caf&#233;<", raw)
        page = Page(raw.decode("ascii"))
        self.assertEqual(page.rows[0]["name"], "caf\u00e9")
        self.assertEqual(sources(page), ["audio/caf%C3%A9.wav"])
        self.assertTrue((out / "audio" / "caf\u00e9.wav").is_file())

    def test_the_figures_of_a_row_are_those_measure_reads_from_its_buffer(self):
        for row, samples in zip(self.page.rows, self.buffers):
            figures = measure(samples, RATE)
            shown = {label: text.split() for label, text in row["figures"].items()}
            self.assertEqual({label: float(number) for label, (number, _) in shown.items()},
                             {"Length": figures["seconds"], "Peak": figures["peak_db"], "Loudness": figures["rms_db"]})
            self.assertEqual({label: unit for label, (_, unit) in shown.items()},
                             {"Length": "s", "Peak": "dB", "Loudness": "dB"})
        # The second variant is set 5 decibels under the bank's peak of -3, and silence has no level at all.
        steps, hush = self.page.rows[2]["figures"], self.page.rows[8]["figures"]
        self.assertEqual((steps["Length"], steps["Peak"]), ("0.100 s", "-8.0 dB"))
        self.assertEqual(hush, {"Length": "0.033 s", "Peak": "-inf dB", "Loudness": "-inf dB"})

    def test_the_page_is_ascii_with_one_style_and_no_script_and_loads_only_its_audio(self):
        self.assertTrue(self.raw.isascii())
        self.assertTrue(self.raw.endswith(b"</html>\n"))
        self.assertNotIn(b"\r", self.raw)
        for banned in (b"<script", b"http"):
            self.assertNotIn(banned, self.raw.lower())
        self.assertEqual(self.page.tags.count("style"), 1)
        self.assertEqual(self.page.styled, [])
        self.assertEqual({"script", "img", "link", "iframe", "object", "embed"} & set(self.page.tags), set())
        self.assertEqual({tag for tag, _ in self.page.loaded}, {"audio"})
        self.assertEqual([src for _, src in self.page.loaded if not src.startswith("audio/")], [])

    def test_a_project_with_no_sounds_gets_a_page_that_says_so(self):
        for project in (Project(), project_of()):
            out = self.folder() / "none"
            written = write_page(project, out, RATE)
            self.assertEqual(written["files"], [])
            page = read(written["page"])
            text = " ".join(page.text)
            self.assertIn("Sounds", text)
            self.assertIn("This project has no sounds.", text)
            self.assertEqual({"audio", "table"} & set(page.tags), set())
            self.assertEqual([path.name for path in out.iterdir()], ["index.html"])

    def test_a_sound_that_cannot_render_is_refused_by_its_name_and_nothing_is_written(self):
        # Loading a project refuses a wave it does not know, so this one is put together by hand.
        project = Project()
        project.sounds = {"chime": {"def": note({"vol": 12}, "e5")},
                          "zap": {"defs": [note({"vol": 12}, "a4"), note({"wave": "sine"}, "a4")]}}
        out = self.folder()
        with self.assertRaises(RowSynthError) as caught:
            write_page(project, out / "site", RATE)
        self.assertEqual(str(caught.exception), "zap: defs[1]: voices[0].inst.wave: unknown wave 'sine'")
        self.assertEqual(list(out.iterdir()), [])

    def test_building_again_into_the_same_folder_replaces_the_page_and_the_files(self):
        out = self.folder()
        write_page(project_of(chime=SOUNDS["chime"], steps=SOUNDS["steps"]), out, RATE)
        longer = project_of(chime={"def": note("reed", "c3", 9)})
        written = write_page(longer, out, RATE)
        # The files of the steps are still in the folder. They are no part of what the second page wrote.
        self.assertEqual(sorted(path.name for path in (out / "audio").iterdir()),
                         ["chime.wav", "steps-0.wav", "steps-1.wav"])
        self.assertEqual(written["files"], [str(out / "audio" / "chime.wav")])
        page = read(written["page"])
        self.assertEqual(sources(page), ["audio/chime.wav"])
        self.assertEqual(page.rows[0]["figures"]["Length"], "0.200 s")
        # The first chime was 800 samples long.
        samples = render_entry(longer.sound("chime"), RATE)[0]
        written = load_wav(out / "audio" / "chime.wav")[0][0]
        self.assertEqual((len(written), len(samples)), (1600, 1600))
        self.assertLessEqual(float(np.abs(written - samples).max()), STEP)

    def test_a_held_sound_that_starts_at_0_has_no_file_and_no_row_for_its_start(self):
        held = {**SOUNDS["hum"]["held"], "start": 0, "join": 0}
        out = self.folder()
        written = write_page(project_of(hum={"held": held}), out, RATE)
        self.assertEqual(sorted(path.name for path in (out / "audio").iterdir()), ["hum-end.wav", "hum-loop.wav"])
        self.assertEqual(written["files"], [str(out / "audio" / file) for file in ("hum-loop.wav", "hum-end.wav")])
        page = read(written["page"])
        self.assertEqual([(row["which"], "loop" in row["player"]) for row in page.rows],
                         [("loop", True), ("end", False)])

    def test_a_rate_left_out_is_48000(self):
        out = self.folder()
        page = read(write_page(project_of(chime=SOUNDS["chime"]), out)["page"])
        channels, rate = load_wav(out / "audio" / "chime.wav")
        self.assertEqual((rate, len(channels[0])), (48000, 4800))
        self.assertEqual(page.rows[0]["figures"]["Length"], "0.100 s")


if __name__ == "__main__":
    unittest.main()
