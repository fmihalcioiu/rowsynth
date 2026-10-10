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
from rowsynth.layers import mix_version, render_groups, render_intro
from rowsynth.measure import measure
from rowsynth.page import write_page
from rowsynth.project import Project
from rowsynth.song import render_loop
from rowsynth.wav import load_wav

# A low rate keeps the rendering short. A tick is 133 and a third samples.
RATE = 8000
# The size of one step of a 16-bit file.
STEP = 1 / 32767
# A decibel under full scale, which is where the loudest sample among the files of a song is put.
PEAK = 10 ** (-1 / 20)

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


# A plain song, and one with layers and an intro. With three ticks to a row, a row is 400 samples: the march
# is 8 rows, the drift is 16, and its intro is 8 with a tail of a quarter of a second after them. The march
# stays well under full scale. The gain of the drift takes it well past.
SONGS = {
    "march": {
        "row_ticks": 3,
        "channels": {"high": {"inst": "reed", "pan": -0.5}, "deep": {"inst": "reed", "gain": 0.8, "pan": 0.4}},
        "patterns": {"a": {"high": "o4 l8 c e g e", "deep": "o2 l4 c g"}},
        "order": ["a"],
    },
    "drift": {
        "row_ticks": 3,
        "gain": 2.5,
        "drums": {"k": {"inst": "tick", "note": "c2"}},
        "channels": {"lead": {"inst": "reed", "pan": -0.6}, "bass": {"inst": "reed", "pan": 0.3},
                     "kit": {"inst": "tick", "drums": True, "gain": 0.5}},
        "patterns": {"in": {"lead": "o4 l4 g e", "kit": "l4 k k"},
                     "a": {"lead": "o4 l8 c e g e", "bass": "o2 l4 c g", "kit": "l4 k k"},
                     "b": {"lead": "o4 l8 d f a f", "bass": "o2 l4 d a", "kit": "l8 k k k k"}},
        "order": ["a", "b"],
        "layers": {"groups": {"tune": ["lead", "bass"], "beat": ["kit"]},
                   "versions": {"full": ["tune", "beat"], "bare": ["beat"]}, "keys": [0, 2, -5]},
        "intro": {"order": ["in"], "tail": 0.25, "fade": 0.05},
    },
}

# The files of SONGS in the order of the project, all of them in audio/songs/: the file, its song, and which
# one of that song it is.
SONG_FILES = [
    ("march.wav", "march", ""),
    ("drift.wav", "drift", ""),
    ("drift.full.0.wav", "drift", "full, key 0"),
    ("drift.full.2.wav", "drift", "full, key +2"),
    ("drift.full.-5.wav", "drift", "full, key -5"),
    ("drift.bare.0.wav", "drift", "bare, key 0"),
    ("drift.bare.2.wav", "drift", "bare, key +2"),
    ("drift.bare.-5.wav", "drift", "bare, key -5"),
    ("drift.intro.wav", "drift", "intro"),
]


def project_of(songs: dict | None = None, **sounds: dict) -> Project:
    """A project with the instruments above, these sounds, and these songs when it is given any."""
    return Project.from_dict({"instruments": INSTRUMENTS, "sounds": sounds, "songs": songs or {}})


def mixes_of(song: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """The two sides of everything a song has a row for, in the order of its rows, at the song's own level.

    Its loop comes first, then each version of its layers in each key, then its intro.
    """
    loop = render_loop(song, RATE)
    mixes = [(loop["left"], loop["right"])]
    if "layers" in song:
        rendered = render_groups(song, RATE)
        mixes += [mix_version(rendered, song, version, key)
                  for version in song["layers"]["versions"] for key in rendered["keys"]]
    if "intro" in song:
        intro = render_intro(song, RATE)
        mixes.append((intro["left"], intro["right"]))
    return mixes


def peak(sides: tuple | list) -> float:
    """The size of the loudest sample of the sides of a mix or of a file."""
    return max(float(np.abs(side).max()) for side in sides)


class Page(HTMLParser):
    """A page as it parses: its tags, its text, what it loads, and every row of a table that has a player.

    A row is {"section", "name", "which", "player", "figures"}: the heading its table stands under, the text of
    its own heading outside and inside <small>, the attributes of its <audio>, and the text of each cell that
    has a data-label, by that label.
    """

    def __init__(self, text: str) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.text: list[str] = []
        self.loaded: list[tuple[str, str]] = []
        self.styled: list[str] = []
        self.rows: list[dict] = []
        self._section = ""
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
        if tag == "h2":
            self._section = ""
        elif tag == "tr":
            self._row = {"section": self._section, "name": "", "which": "", "player": None, "figures": {}}
        elif tag == "td":
            self._label = given.get("data-label")
        elif tag == "audio":
            self._row["player"] = given
        if tag in ("h2", "th", "td", "small"):
            self._inside.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("h2", "th", "td", "small"):
            self._inside.pop()
        elif tag == "tr" and self._row["player"] is not None:
            self._row["name"] = self._row["name"].strip()
            self.rows.append(self._row)

    def handle_data(self, data: str) -> None:
        self.text.append(data)
        inside = self._inside[-1] if self._inside else ""
        if inside == "h2":
            self._section += data
        elif inside == "small":
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


class PageCase(unittest.TestCase):
    """What the tests of the two sections share."""

    def folder(self) -> Path:
        """A folder of the test's own, gone when the test is over."""
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        return Path(folder.name)


class WritePageTest(PageCase):
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


class SongRowsTest(PageCase):
    @classmethod
    def setUpClass(cls) -> None:
        """One page of two sounds and SONGS for the tests that only read it. One sound has the name of a song."""
        folder = tempfile.TemporaryDirectory()
        cls.addClassCleanup(folder.cleanup)
        cls.sounds = {"chime": SOUNDS["chime"], "march": SOUNDS["thump"]}
        cls.project = project_of(SONGS, **cls.sounds)
        cls.out = Path(folder.name, "site")
        cls.songs = cls.out / "audio" / "songs"
        cls.written = write_page(cls.project, cls.out, RATE)
        cls.page = read(cls.written["page"])
        cls.rows = [row for row in cls.page.rows if row["section"] == "Songs"]
        cls.mixes = {name: mixes_of(cls.project.song(name)) for name in SONGS}

    def files_of(self, name: str) -> list[list[np.ndarray]]:
        """The sides of each file of a song, in the order of its rows."""
        return [load_wav(self.songs / file)[0] for file, song, _ in SONG_FILES if song == name]

    def test_the_rows_of_the_songs_come_after_those_of_the_sounds_and_the_files_are_listed_as_the_rows_come(self):
        audio = self.out / "audio"
        files = [audio / "chime.wav", audio / "march.wav", *(self.songs / file for file, _, _ in SONG_FILES)]
        self.assertEqual(self.written["files"], [str(path) for path in files])
        self.assertEqual([row["section"] for row in self.page.rows], ["Sounds"] * 2 + ["Songs"] * len(SONG_FILES))
        # The address of each player leads from the page to the file of its row.
        self.assertEqual([self.out / unquote(src) for src in sources(self.page)], files)
        for path in files:
            self.assertTrue(path.is_file(), path)

    def test_a_plain_song_is_one_stereo_file_of_its_loop_times_its_factor_within_a_step(self):
        loop = render_loop(self.project.song("march"), RATE)
        made = (loop["left"], loop["right"])
        # The march is made under full scale, so its file is the louder of the two.
        factor = PEAK / peak(made)
        self.assertGreater(factor, 1.5)
        sides, rate = load_wav(self.songs / "march.wav")
        self.assertEqual((len(sides), rate, loop["loop_samples"]), (2, RATE, 3200))
        for side, wanted in zip(sides, made):
            self.assertEqual(len(side), 3200)
            self.assertLessEqual(float(np.abs(side - wanted.astype(np.float64) * factor).max()), STEP)

    def test_a_layered_song_has_a_row_for_its_loop_then_each_version_in_each_key_then_its_intro(self):
        self.assertEqual([(row["name"], row["which"]) for row in self.rows],
                         [(song, which) for _, song, which in SONG_FILES])
        self.assertEqual([row["player"]["src"] for row in self.rows],
                         [f"audio/songs/{file}" for file, _, _ in SONG_FILES])
        # Each file has the two sides of its mix: 16 rows for a loop, and 8 rows and the tail for the intro.
        for sides, mix in zip(self.files_of("drift"), self.mixes["drift"], strict=True):
            self.assertEqual([len(side) for side in sides], [len(mix[0])] * 2)
        self.assertEqual([len(mix[0]) for mix in self.mixes["drift"]], [6400] * 7 + [5200])

    def test_the_player_of_every_loop_of_a_song_goes_round_and_that_of_its_intro_does_not(self):
        self.assertEqual([row["which"] for row in self.rows if "loop" not in row["player"]], ["intro"])
        for row in self.rows:
            self.assertIn("controls", row["player"])
            self.assertEqual(row["player"]["preload"], "none")

    def test_the_files_of_a_song_share_the_one_factor_that_puts_the_loudest_a_decibel_under_full_scale(self):
        for name, mixes in self.mixes.items():
            factor = PEAK / max(peak(mix) for mix in mixes)
            files = self.files_of(name)
            for sides, mix in zip(files, mixes, strict=True):
                for side, wanted in zip(sides, mix):
                    self.assertLessEqual(float(np.abs(side - wanted.astype(np.float64) * factor).max()), STEP, name)
            self.assertAlmostEqual(max(peak(sides) for sides in files), PEAK, delta=STEP, msg=name)
        # The drift is made well past full scale, and no file of it clips. Its loop sets the level, and the full
        # version as written is the same channels added up. The other keys and the intro are under them, and
        # the drums on their own a long way under.
        self.assertGreater(max(peak(mix) for mix in self.mixes["drift"]), 1.8)
        loop, full, up, down, bare, _, _, intro = (peak(sides) for sides in self.files_of("drift"))
        self.assertAlmostEqual(loop, PEAK, delta=STEP)
        self.assertAlmostEqual(full, loop, delta=STEP)
        for lower in (up, down, intro):
            self.assertLess(lower, loop - 0.03)
        self.assertLess(bare, loop / 3)

    def test_the_figures_of_a_song_row_are_those_measure_reads_from_the_mean_of_its_sides_before_the_factor(self):
        mixes = [mix for name in SONGS for mix in self.mixes[name]]
        for row, (left, right) in zip(self.rows, mixes, strict=True):
            figures = measure((left.astype(np.float64) + right) / 2, RATE)
            shown = {label: text.split() for label, text in row["figures"].items()}
            self.assertEqual({label: float(number) for label, (number, _) in shown.items()},
                             {"Length": figures["seconds"], "Peak": figures["peak_db"], "Loudness": figures["rms_db"]})
            self.assertEqual({label: unit for label, (_, unit) in shown.items()},
                             {"Length": "s", "Peak": "dB", "Loudness": "dB"})
        self.assertEqual([row["figures"]["Length"] for row in self.rows], ["0.400 s"] + ["0.800 s"] * 7 + ["0.650 s"])
        # The row of the drift says how far past full scale the song is made, and its file is under it all the same.
        self.assertGreater(float(self.rows[1]["figures"]["Peak"].split()[0]), 3)
        self.assertLess(peak(load_wav(self.songs / "drift.wav")[0]), 0.9)

    def test_a_sound_and_a_song_of_one_name_have_a_file_each(self):
        audio = self.out / "audio"
        self.assertEqual(sorted(path.name for path in audio.iterdir()), ["chime.wav", "march.wav", "songs"])
        self.assertEqual(sorted(path.name for path in self.songs.iterdir()), sorted(file for file, _, _ in SONG_FILES))
        self.assertEqual([(row["section"], row["player"]["src"]) for row in self.page.rows if row["name"] == "march"],
                         [("Sounds", "audio/march.wav"), ("Songs", "audio/songs/march.wav")])
        # The file of the sound is still its one buffer, and that of the song has two sides.
        (buffer,) = render_entry(self.project.sound("march"), RATE)
        (written,), _ = load_wav(audio / "march.wav")
        self.assertEqual(len(written), len(buffer))
        self.assertLessEqual(float(np.abs(written - buffer).max()), STEP)
        self.assertEqual(len(load_wav(self.songs / "march.wav")[0]), 2)

    def test_the_section_of_the_sounds_is_written_the_same_with_songs_after_it_as_with_none(self):
        alone = Path(write_page(project_of(**self.sounds), self.folder(), RATE)["page"]).read_text(encoding="ascii")
        both = (self.out / "index.html").read_text(encoding="ascii")
        end = "</section>\n"
        self.assertEqual(both[:both.index(end)], alone[:alone.index(end)])
        self.assertIn("<h2>Sounds</h2>", both[:both.index(end)])

    def test_a_project_with_no_songs_gets_a_line_that_says_so_after_its_sounds(self):
        out = self.folder()
        written = write_page(project_of(chime=SOUNDS["chime"]), out, RATE)
        page = read(written["page"])
        text = " ".join(page.text)
        self.assertLess(text.index("Sounds"), text.index("Songs"))
        self.assertIn("This project has no songs.", text)
        self.assertNotIn("This project has no sounds.", text)
        self.assertEqual((page.tags.count("table"), page.tags.count("audio")), (1, 1))
        self.assertEqual(written["files"], [str(out / "audio" / "chime.wav")])
        self.assertEqual([path.name for path in (out / "audio").iterdir()], ["chime.wav"])

    def test_a_project_with_songs_and_no_sounds_says_so_of_its_sounds_and_has_the_rows_of_its_songs(self):
        out = self.folder() / "site"
        written = write_page(project_of({"march": SONGS["march"]}), out, RATE)
        page = read(written["page"])
        text = " ".join(page.text)
        self.assertIn("This project has no sounds.", text)
        self.assertNotIn("This project has no songs.", text)
        self.assertEqual([(row["section"], row["name"], row["which"]) for row in page.rows], [("Songs", "march", "")])
        self.assertEqual(written["files"], [str(out / "audio" / "songs" / "march.wav")])

    def test_a_silent_song_is_written_as_it_is_and_its_row_says_it_has_no_level(self):
        rests = {**SONGS["march"], "patterns": {"a": {"high": "r2", "deep": "r2"}}}
        out = self.folder()
        page = read(write_page(project_of({"hush": rests}), out, RATE)["page"])
        left, right = load_wav(out / "audio" / "songs" / "hush.wav")[0]
        self.assertEqual((len(left), len(right)), (3200, 3200))
        self.assertEqual(peak((left, right)), 0.0)
        self.assertEqual(page.rows[0]["figures"], {"Length": "0.400 s", "Peak": "-inf dB", "Loudness": "-inf dB"})

    def test_a_song_that_cannot_render_is_refused_with_its_place_in_front_and_nothing_is_written(self):
        # An order of no patterns has no rows to loop. Loading takes it: it holds a song to its shape and no more.
        songs = {"march": SONGS["march"], "still": {**SONGS["march"], "order": []}}
        out = self.folder()
        with self.assertRaises(RowSynthError) as caught:
            write_page(project_of(songs, chime=SOUNDS["chime"]), out / "site", RATE)
        self.assertEqual(str(caught.exception), "songs.still: the song has no rows, so there is nothing to loop")
        self.assertEqual(list(out.iterdir()), [])

    def test_two_rows_whose_files_have_one_name_with_case_set_aside_are_refused_and_nothing_is_written(self):
        # Two songs that differ only in case, and a sound with the name that a variant of another has its file by.
        twice = [(project_of({"March": SONGS["march"], "march": SONGS["march"]}),
                  "audio/songs/March.wav and audio/songs/march.wav"),
                 (project_of(**{"steps": SOUNDS["steps"], "steps-0": SOUNDS["chime"]}),
                  "audio/steps-0.wav and audio/steps-0.wav")]
        for project, files in twice:
            out = self.folder()
            with self.assertRaises(RowSynthError) as caught:
                write_page(project, out / "site", RATE)
            self.assertEqual(str(caught.exception), f"{files}: two rows of the page cannot have one file, and these "
                                                    "names are the same once case is set aside")
            self.assertEqual(list(out.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
