"""The window that writes files: the picked thing, a build or the listen page, each file named before it is made."""
from __future__ import annotations

import math
import os
import shutil
import threading
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from ..bank import held_spans
from ..build import BUILD_FORMATS, BUILD_RATE, _file as build_file, _layer_files, build, manifest_path
from ..commands import PARTS, Pick, render_sides
from ..commands.render import _file as write_file, _stems as write_stems, stem_file
from ..encode import FORMATS, unwritable
from ..errors import RowSynthError
from ..note import TICK_HZ
from ..page import _SONGS, PAGE_RATE, _file as page_file, _parts, write_page
from ..song import _TAIL, _row_samples, lay
from . import theme
from .dials import Stepper
from .font import ADVANCE, CUT
from .layout import Rect
from .lists import List, Row
from .popups import ask
from .theme import Colour
from .ui import Button, Choice, Event, Pos, Toggle, Ui, Widget, inside

if TYPE_CHECKING:
    from ..project import Project
    from .canvas import Canvas
    from .session import Session

#: The tabs of the window in the order they are shown, each by the command it stands for, and what each is called.
TABS = ("render", "build", "page")
TAB_WORDS = ("THIS", "BUILD", "PAGE")
#: The rates a build can be rendered at here. Opus takes five of them.
RATES = (8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000)
#: The first option of the songs a build builds, before the names of the songs.
EVERY = "every song"
#: The folder the files go to as the window opens, beside the project file.
OUT = "out"
#: The folder inside the folder of a page that its audio goes in, and the page itself.
AUDIO = "audio"
INDEX = "index.html"
#: How the mark of a file reads that takes the place of one that is there.
REPLACES = "replaces"
#: The bytes a WAV file has ahead of its samples, and the bytes one sample of one side takes.
WAV_HEAD = 44
WAV_SAMPLE = 2

#: The words on the window, and those before the rows of each tab, from the top.
WRITE = "WRITE"
CLOSE = "CLOSE"
CHANGE = "CHANGE"
FOLDER = "FOLDER"
STEMS = "ONE LOOP PER CHANNEL"
NOTHING = "NOTHING"
WORDS = {"render": ("PICKED", "", ""), "build": ("FORMAT", "RATE", "SONGS"), "page": ("FORMAT", "", "")}
#: Why the first tab can write nothing while no sound and no song is picked.
NO_PICK = "pick a sound or a song in the list at the left: THIS writes the picked one"

#: The help line of each part.
HELP = {
    "tabs": "what to write: the picked sound or song, the whole project for a program to play, or the listen page",
    "stems": "write the song as one mono loop a channel, each CHANNEL.wav, in place of the one stereo file",
    "build_format": "the format of the files: wav, or opus, which is made by ffmpeg",
    "rate": "the samples a second the songs are rendered at",
    "songs": "the songs to build: every song of the project, or one of them",
    "page_format": "the format of the audio of the page: wav, or mp3 or opus, which are made by ffmpeg",
    "change": "type another folder for the files to go to",
    "box": "the folder the files go to: enter takes it, and escape leaves it as it was",
    "files": "the files a press of WRITE makes, each with its size where that is known before it is rendered",
    "write": "write these files into the folder",
    "close": "close this window: a write that is under way goes on to its end",
}

#: How wide the window is, and from its sides to what it shows.
WIDE = 440
MARGIN = 8
#: How high a row of the window is, a button, a field or a line of words, and between two rows.
ROW_H = 15
SPACE = 4
#: How many rows of settings a tab has room for, and how many files the list shows before it scrolls.
ROWS = 3
LIST_ROWS = 13
#: How wide the words before a row are, the field of a row, and a button.
LABEL_W = 54
FIELD_W = 150
BUTTON_W = 52
#: How high the window is: the tabs, the settings, the folder, the list in its outline, the line and the buttons.
HIGH = 2 * MARGIN + (ROWS + 3) * (ROW_H + SPACE) + 2 + LIST_ROWS * theme.LIST_ROW_H + SPACE + ROW_H
#: The most characters the folder can have.
FOLDER_MOST = 260


@dataclass(frozen=True)
class File:
    """One file a write will make: its name, its size, and whether it takes the place of one that is there.

    `name` is where it goes in the folder, with a "/" after each folder on the way. `size` is in bytes, and
    None where it is not known before the file is rendered.
    """

    name: str
    size: int | None = None
    replaces: bool = False


@dataclass(frozen=True)
class Plan:
    """What a press of WRITE is to do, worked out before it is done.

    `tab` is the command it stands for, one of `TABS`, and `rate` and `fmt` are what the files are rendered at
    and written as. `picked` is the (kind, name) that "render" writes, and `stems` whether that is one loop a
    channel. `songs` names the songs "build" builds, None for every one. `files` are what will be written
    into `folder`, and `refusal` is why nothing can be, None when nothing is known against it.
    """

    tab: str
    folder: Path
    rate: int
    fmt: str = "wav"
    picked: tuple[str, str] | None = None
    stems: bool = False
    songs: tuple[str, ...] | None = None
    files: tuple[File, ...] = ()
    refusal: str | None = None


def wav_size(samples: int, sides: int) -> int:
    """The bytes of a WAV file of so many samples a side, at the 16 bits a sample the package writes."""
    return WAV_HEAD + samples * sides * WAV_SAMPLE


def size_words(size: int | None) -> str:
    """A size in bytes in a few characters, as in "812 B", "41 KB" and "1.4 MB". Nothing for one that is not known."""
    if size is None:
        return ""
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size / 1024 / 1024:.1f} MB"


def counted(count: int) -> str:
    """A number of files in words: "1 file", "3 files"."""
    return f"{count} file" + ("" if count == 1 else "s")


def tag(file: File) -> str:
    """What stands at the right of a file in the list: that it replaces one, and its size where that is known."""
    return " ".join(word for word in (REPLACES if file.replaces else "", size_words(file.size)) if word)


def tail(text: str, width: int) -> str:
    """`text` as it is shown in `width` pixels when its end is what tells it apart: cut at the front, after '..'."""
    room = max(0, (width + 1) // ADVANCE)
    if len(text) <= room:
        return text
    return (CUT + text[len(text) - max(0, room - len(CUT)):])[:room]


def _loop(song: dict, rate: int) -> int:
    """The samples the loop of a song has at a rate, counted as `render_stems` counts them, with no note played."""
    _, rows = lay(song)
    return math.floor(rows * song["row_ticks"] * rate / TICK_HZ + 0.5)


def render_files(project: Project, kind: str, name: str, rate: int, stems: bool = False) -> list[File]:
    """The files the first tab writes of a sound or a song, as the render command writes them.

    It is one file, NAME.wav: a sound in mono, whose size is not known before it is rendered, or a song in
    stereo, half a second longer than its rows. With `stems` a song is one mono loop for each channel that
    plays, named as the command names them.
    """
    if kind != "song":
        return [File(f"{name}.wav")]
    song = project.song(name)
    laid, rows = lay(song)
    if stems:
        size = wav_size(_loop(song, rate), 1)
        return [File(stem_file(channel), size) for channel in laid]
    return [File(f"{name}.wav", wav_size(math.ceil(rows * _row_samples(song, rate) + rate * _TAIL), 2))]


def build_files(project: Project, rate: int, fmt: str = "wav", songs: Sequence[str] | None = None) -> list[File]:
    """The files a build writes, named as `build` names them and before a note is played: the manifest is the last.

    `songs` names the songs to build, None for every song of the project. A file that holds a loop has its
    size where the format is WAV, and an intro has none. A song the build would refuse for its layers is
    refused here, with its place in the project in front.
    """
    files = []
    for name in project.songs if songs is None else songs:
        song = project.song(name)
        try:
            size = wav_size(_loop(song, rate), 2) if fmt == "wav" else None
            beside = [build_file(f"{name}.intro", fmt)] if "intro" in song else []
            if "layers" in song:
                _, mixes, premixes = _layer_files(song, name, fmt, beside)
                loops = [file for file, _, _ in mixes + premixes]
            else:
                loops = [build_file(name, fmt)]
        except RowSynthError as error:
            raise RowSynthError(f"songs.{name}: {error}") from None
        files += [File(file, size) for file in loops] + [File(file) for file in beside]
    return [*files, File(manifest_path("").name)]


def page_files(project: Project, rate: int, fmt: str = "wav") -> list[File]:
    """The files the listen page is written as, named as `write_page` names them: the page itself is the last.

    The audio of the sounds comes first, in the folder audio, and that of the songs after it, in audio/songs.
    A held sound whose start has no samples has no file for it. What is a loop has its size where the format
    is WAV: a song, its versions, and the start and the loop of a held sound.
    """
    wav = fmt == "wav"
    files = []
    for name, entry in project.sounds.items():
        parts = _parts(entry, len(entry.get("defs", ())))
        try:
            # The start and the loop of a held sound are cut to lengths that are known before its note is played.
            known = dict(zip(parts, held_spans(entry, rate)[:2])) if "held" in entry else {}
        except RowSynthError as error:
            raise RowSynthError(f"{name}: {error}") from None
        for part in parts:
            if known.get(part) != 0:
                size = wav_size(known[part], 1) if wav and part in known else None
                files.append(File(f"{AUDIO}/{page_file(name, part, fmt)}", size))
    for name in project.songs:
        song = project.song(name)
        try:
            size = wav_size(_loop(song, rate), 2) if wav else None
        except RowSynthError as error:
            raise RowSynthError(f"songs.{name}: {error}") from None
        mixes: list[str | None] = [None]
        if "layers" in song:
            keys = song["layers"].get("keys", [0])
            mixes += [f"{version}.{key}" for version in song["layers"].get("versions", {}) for key in keys]
        files += [File(f"{AUDIO}/{_SONGS}/{page_file(name, mix, fmt, '.')}", size) for mix in mixes]
        if "intro" in song:
            files.append(File(f"{AUDIO}/{_SONGS}/{page_file(name, 'intro', fmt, '.')}"))
    return [*files, File(INDEX)]


def marked(files: Sequence[File], folder: Path) -> list[File]:
    """The files with each marked that takes the place of something that is in the folder already."""
    return [replace(file, replaces=(folder / file.name).exists()) for file in files]


def no_format(fmt: str, rate: int) -> str | None:
    """Why no file of a format can be written at a rate on this machine, or None when one can.

    It is what `unwritable` has against the format and the rate, and after that a format ffmpeg makes while
    no ffmpeg is found on the path, in the words `save_audio` refuses it with.
    """
    why = unwritable(fmt, rate)
    if why is None and fmt != "wav" and shutil.which("ffmpeg") is None:
        why = f"ffmpeg is needed to write {fmt}, and none is found on the path"
    return why


def no_folder(folder: Path) -> str | None:
    """Why no file can be written into a folder, or None when nothing is known against it.

    The folder need not be there: the package makes it, and whatever is missing above it. In the way is a
    file where the folder or one above it is to be, and a folder that is there and cannot be written in.
    """
    there = next((path for path in (folder, *folder.parents) if path.exists()), None)
    if there is None:
        return f"{folder}: there is nowhere to make this folder"
    if not there.is_dir():
        return f"{there}: a file of this name is in the way of the folder"
    if not os.access(there, os.W_OK):
        return f"{there}: this folder cannot be written in"
    return None


def plan(project: Project, tab: str, folder: str | Path, rate: int, fmt: str = "wav",
         picked: tuple[str, str] | None = None, stems: bool = False,
         songs: Sequence[str] | None = None) -> Plan:
    """The plan of one press of WRITE: the files it makes in the folder, or why it can make none.

    The files are named as the command of the tab names them, and nothing is rendered for it. What is known
    against the write is said here and not found out by trying: nothing picked that the first tab could
    write, a project the package refuses to name the files of, a format that cannot be written, and a folder
    that cannot be written in. Where only the format or the folder is in the way the files are listed all
    the same. A tab that is none of `TABS` is refused.
    """
    if tab not in TABS:
        raise RowSynthError(f"files are written by one of {', '.join(TABS)}, not by {tab!r}")
    folder = Path(folder)
    stems = stems and picked is not None and picked[0] == "song"
    made = Plan(tab, folder, rate, fmt, picked, stems, None if songs is None else tuple(songs))
    try:
        if tab == "build":
            files = build_files(project, rate, fmt, songs)
        elif tab == "page":
            files = page_files(project, rate, fmt)
        elif picked is None or picked[0] not in ("song", "sound"):
            raise RowSynthError(NO_PICK)
        else:
            files = render_files(project, *picked, rate, stems)
    except RowSynthError as error:
        return replace(made, refusal=str(error))
    return replace(made, files=tuple(marked(files, folder)), refusal=no_format(fmt, rate) or no_folder(folder))


def write(project: Project, made: Plan) -> int:
    """Write the files of a plan with the package's own functions, called as its commands call them.

    The answer is how many files the plan names. What the package refuses is refused in its words, and
    nothing is made here but what the package makes: no folder ahead of it, and no file of this module's own.
    A plan that says why it can write nothing is refused with that.
    """
    if made.refusal is not None:
        raise RowSynthError(made.refusal)
    folder = str(made.folder)
    if made.tab == "build":
        build(project, folder, made.rate, made.fmt, None if made.songs is None else list(made.songs))
    elif made.tab == "page":
        write_page(project, folder, made.rate, made.fmt)
    elif made.stems:
        write_stems(project.song(made.picked[1]), folder, made.rate)
    else:
        kind, name = made.picked
        # A held sound is written as its start, which is the part the command takes when it is told none.
        held = kind == "sound" and "held" in project.sounds.get(name, {})
        picked = Pick(name, kind, 0, PARTS[0] if held else None)
        write_file(str(made.folder / made.files[0].name), picked, render_sides(project, picked, made.rate), made.rate)
    return len(made.files)


def doing(made: Plan) -> str:
    """What is said while the files of a plan are written: what it is, how many files, and where they go."""
    if made.tab == "render":
        what = made.picked[1] + (" as one loop a channel" if made.stems else "")
    else:
        what = "the build" if made.tab == "build" else "the listen page"
    return f"writing {what}: {counted(len(made.files))} to {made.folder}"


def summed(files: Sequence[File]) -> str:
    """What is said of the files of a plan while nothing else is: how many, and how many of them replace one."""
    old = sum(file.replaces for file in files)
    return f"{counted(len(files))} to write" + (f", {old} in place of what is there" if old else "")


def parts(size: tuple[int, int]) -> dict[str, Rect]:
    """Where the parts of the window go on an art surface of a size, each by its name as (x, y, w, h).

    "box" is the whole window, in the middle of the surface and no larger than it. From the top come "title"
    and "tabs", the rows of the settings ("word0" and "row0" and so on, and "picked" for the words of the
    first), the folder ("folder_word", its field "folder" with the room of its text "folder_name", and
    "change"), the outline of the list "list", the line "said", and the buttons "write" and "close".
    """
    wide, high = min(WIDE, size[0]), min(HIGH, size[1])
    x, y = (size[0] - wide) // 2, (size[1] - high) // 2
    left, right = x + MARGIN, x + wide - MARGIN
    field, step = left + LABEL_W, ROW_H + SPACE
    top = y + MARGIN
    at = {"box": (x, y, wide, high), "title": (left, top, LABEL_W, ROW_H), "tabs": (field, top, FIELD_W, ROW_H)}
    for row in range(ROWS):
        top += step
        at[f"word{row}"] = (left, top, LABEL_W, ROW_H)
        at[f"row{row}"] = (field, top, FIELD_W, ROW_H)
    at["picked"] = (field, at["row0"][1], right - field, ROW_H)
    top += step
    inset = theme.TEXT_PAD + 1
    at["folder_word"] = (left, top, LABEL_W, ROW_H)
    at["folder"] = (field, top, right - BUTTON_W - SPACE - field, ROW_H)
    at["folder_name"] = (field + inset, top, at["folder"][2] - 2 * inset, ROW_H)
    at["change"] = (right - BUTTON_W, top, BUTTON_W, ROW_H)
    foot = y + high - MARGIN - ROW_H
    at["write"] = (right - 2 * BUTTON_W - SPACE, foot, BUTTON_W, ROW_H)
    at["close"] = (right - BUTTON_W, foot, BUTTON_W, ROW_H)
    at["said"] = (left, foot - step, right - left, ROW_H)
    top += step
    at["list"] = (left, top, right - left, max(0, foot - step - SPACE - top))
    return at


class Export(Widget):
    """The window that writes files, open over everything else: `open_export` opens one.

    Its three tabs are the picked sound or song as a WAV file, the build of the project for a program, and
    the listen page. Under the settings of the tab are the folder the files go to, which a box that is typed
    into changes, and the list of the files a write will make there, each with its size where that is known
    and marked where it takes the place of a file that is there. `plan` is what a press of WRITE would do as
    things are now, and `said` the one line of the window.

    WRITE does the work on a thread of its own, `worker`, with the package's own functions, and is not
    enabled while that runs or while the plan says why nothing can be written. CLOSE and escape close the
    window. Closed while it writes, the write goes on to its end, and the session then says how it ended.

    Its parts are `tabs`, the settings `stems`, `build_format`, `rate`, `songs` and `page_format`, the
    button `change` of the folder, the list `files`, and the buttons `write` and `shut`.
    """

    def __init__(self, ui: Ui, session: Session, start: str = "render") -> None:
        """`ui` is the Ui the window is opened on, and `start` the tab it opens at, one of `TABS`."""
        if start not in TABS:
            raise RowSynthError(f"the window that writes files opens at one of {', '.join(TABS)}, not at {start!r}")
        super().__init__((0, 0, 0, 0))
        self.ui = ui
        self.session = session
        self.folder = session.model.path.parent / OUT
        #: The thread a write is on, from the press of WRITE until its end has been taken up.
        self.worker: threading.Thread | None = None
        #: Whether the window was closed while it wrote: it is then no window, and only waits for the end.
        self.gone = False
        # What is said while the worker runs, how its write ended as the worker left it, and the same once it
        # has been taken up: words and their colour.
        self._doing = ""
        self._ended: tuple[str, Colour] | None = None
        self._told: tuple[str, Colour] | None = None
        # The parts are widgets of a Ui of the window's own, which is handed everything the window is.
        self.inner = Ui()
        self.inner.keys[("escape", frozenset())] = self.close
        nowhere = (0, 0, 0, 0)
        self.tabs = Choice(nowhere, TAB_WORDS, TABS.index(start), self._changed, HELP["tabs"])
        self.stems = Toggle(nowhere, STEMS, changed=self._changed, help=HELP["stems"])
        self.build_format = Choice(nowhere, [fmt.upper() for fmt in BUILD_FORMATS], 0, self._changed,
                                   HELP["build_format"])
        self.rate = Stepper(nowhere, RATES, RATES.index(BUILD_RATE), self._changed, HELP["rate"])
        self.songs = Stepper(nowhere, (EVERY,), 0, self._changed, HELP["songs"])
        self.page_format = Choice(nowhere, [fmt.upper() for fmt in FORMATS], 0, self._changed, HELP["page_format"])
        self.change = Button(nowhere, CHANGE, action=self._ask, help=HELP["change"])
        self.files = List(nowhere, help=HELP["files"])
        self.write = Button(nowhere, WRITE, action=self.begin, help=HELP["write"])
        self.shut = Button(nowhere, CLOSE, action=self.close, help=HELP["close"])
        for part in (self.tabs, self.stems, self.build_format, self.rate, self.songs, self.page_format,
                     self.change, self.files, self.write, self.shut):
            self.inner.add(part)
        self.keep_in(theme.DESIGN)
        self.refresh()

    @property
    def takes_text(self) -> bool:
        """Whether what is typed is the window's: it is while the box of its folder is open."""
        return self.inner.modal is not None

    @property
    def busy(self) -> bool:
        """Whether a write is under way, or has ended and was not taken up yet."""
        return self.worker is not None

    def hit(self, pos: Pos) -> bool:
        """Whether a position is on the window. Never once it is closed and only waits for its write to end."""
        return not self.gone and inside(self.rect, pos)

    def keep_in(self, size: tuple[int, int]) -> None:
        """Take the size of the art surface, and lay the window out in the middle of it."""
        self.inner.size = size
        at = self._at = parts(size)
        self.rect = at["box"]
        x, y, w, h = at["list"]
        placed = {self.tabs: at["tabs"], self.stems: at["row1"], self.build_format: at["row0"],
                  self.rate: at["row1"], self.songs: at["row2"], self.page_format: at["row0"],
                  self.change: at["change"], self.files: (x + 1, y + 1, max(0, w - 2), max(0, h - 2)),
                  self.write: at["write"], self.shut: at["close"]}
        for widget, rect in placed.items():
            widget.rect = rect

    def refresh(self) -> None:
        """Work the plan out again from the project, the pick, the folder and the settings, and show it."""
        model = self.session.model
        project, picked = model.project, model.picked
        self._seen = (project, picked)
        tab = TABS[self.tabs.picked]
        if self.songs.options[1:] != tuple(project.songs):
            self.songs.options, self.songs.picked = (EVERY, *project.songs), 0
        shown = {self.stems: tab == "render" and picked is not None and picked[0] == "song",
                 self.build_format: tab == "build", self.rate: tab == "build", self.songs: tab == "build",
                 self.page_format: tab == "page"}
        for widget, here in shown.items():
            widget.visible = here
        if tab == "build":
            # The first option is every song, and each one after it is one song by its name.
            only = (self.songs.options[self.songs.picked],) if self.songs.picked else None
            self.plan = plan(project, tab, self.folder, RATES[self.rate.picked],
                             BUILD_FORMATS[self.build_format.picked], songs=only)
        elif tab == "page":
            self.plan = plan(project, tab, self.folder, PAGE_RATE, FORMATS[self.page_format.picked])
        else:
            self.plan = plan(project, tab, self.folder, model.rate, picked=picked, stems=self.stems.is_on)
        self.files.rows = [Row(index, file.name, tag=tag(file), square=theme.MUSTARD if file.replaces else theme.LINE)
                           for index, file in enumerate(self.plan.files)]
        self.write.enabled = not self.busy and self.plan.refusal is None
        # A button that does nothing says why, where the mouse rests on it.
        self.write.help = self.plan.refusal or HELP["write"]

    def said(self) -> tuple[str, Colour]:
        """The one line of the window, and the colour it is in.

        While it writes, it is what is being written. Else it is why nothing can be written, in
        `theme.FAULT`. Else it is how the last write ended, until something is changed: how many files it wrote
        and where, or the words it was refused with, in `theme.FAULT`. With none of those it counts the files.
        """
        self._take()
        if self.busy:
            return self._doing, theme.ACTIVE
        if self.plan.refusal is not None:
            return self.plan.refusal, theme.FAULT
        return self._told or (summed(self.plan.files), theme.DIM)

    def begin(self) -> None:
        """Start the write of the plan as things are now, on a thread of its own. Nothing while one is under way.

        The thread is not one that ends with the process: files that were begun are written to their end.
        """
        self._take()
        self.refresh()
        if self.busy or self.plan.refusal is not None:
            return
        self._told, self._doing = None, doing(self.plan)
        self.worker = threading.Thread(target=self._work, args=(self.session.model.project, self.plan),
                                       name="rowsynth-export")
        self.write.enabled = False
        self.worker.start()

    def close(self) -> None:
        """Close the window. A write that is under way goes on, and the session says how it ended when it has."""
        self.ui.close(self)
        if self.busy:
            # It stays among the widgets with nothing to show and nothing to take, and is looked at each frame.
            self.gone = True
            self.ui.add(self)

    def on(self, event: Event, ui: Ui) -> bool:
        """Use everything: it goes to the part it is for, or to the box of the folder while that is open."""
        self._take()
        self.inner.handle(event)
        self.help = self.inner.help
        return True

    def draw(self, canvas: Canvas) -> None:
        """Draw the window: its box, the words before its rows, the folder, the line that is said, and its parts."""
        self._take()
        if self.gone:
            return
        at, tab = self._at, TABS[self.tabs.picked]
        x, y, w, h = at["box"]
        canvas.box((x - 1, y - 1, w + 2, h + 2), theme.BACKGROUND)
        canvas.box(at["box"], theme.PANEL)
        canvas.frame(at["box"], theme.LINE)
        canvas.text_in(WRITE, at["title"], theme.DIM)
        for row, word in enumerate(WORDS[tab]):
            canvas.text_in(word, at[f"word{row}"], theme.DIM)
        if tab == "render":
            kind, name = self._seen[1] or (NOTHING, "")
            canvas.text_in(f"{kind.upper()} {name}".strip(), at["picked"], theme.TEXT)
        canvas.text_in(FOLDER, at["folder_word"], theme.DIM)
        canvas.box(at["folder"], theme.BACKGROUND)
        canvas.frame(at["folder"], theme.LINE)
        canvas.text_in(tail(str(self.folder), at["folder_name"][2]), at["folder_name"], theme.TEXT)
        canvas.frame(at["list"], theme.LINE)
        words, colour = self.said()
        canvas.text_in(words, at["said"], colour)
        self.inner.draw(canvas)
        self.help = self.inner.help

    def _changed(self, _: object = None) -> None:
        """Take a setting that was changed: the plan is another, and how the last write ended is said no longer."""
        self._told = None
        self.refresh()

    def _ask(self) -> None:
        """Open the box the folder is typed into, over the folder."""
        ask(self.inner, self._at["folder_name"], str(self.folder), self._typed, most=FOLDER_MOST, help=HELP["box"])

    def _typed(self, text: str) -> None:
        """Take the folder that was typed. With nothing typed, it stays the one it was."""
        if text.strip():
            self.folder = Path(text.strip())
            self._changed()

    def _work(self, project: Project, made: Plan) -> None:
        """Write the files of a plan, on the worker, and leave how that ended for the window's own thread."""
        try:
            ended = f"wrote {counted(write(project, made))} to {made.folder}", theme.GOOD
        except Exception as error:
            # A refusal in its own words, and anything else with the name of its class in front.
            words = str(error) if isinstance(error, RowSynthError) else f"{type(error).__name__}: {error}"
            ended = words, theme.FAULT
        self._ended = ended

    def _take(self) -> None:
        """Take up what happened since the last look: a write that ended, another project, another thing picked."""
        ended = self.worker is not None and self._ended is not None
        if ended:
            self._told, self._ended, self.worker = self._ended, None, None
        if self.gone:
            if ended:
                # There is no window to say it in, so the session does, on the thread that owns it.
                self.session.say(self._told[0])
                self.ui.widgets.remove(self)
            return
        model = self.session.model
        if ended:
            self.refresh()
        elif model.project is not self._seen[0] or model.picked != self._seen[1]:
            self._changed()


def open_export(ui: Ui, session: Session, start: str = "render") -> Export:
    """Open the window that writes files as the modal widget, at one of its tabs, and give it back.

    `start` is the command the tab stands for: "render" for the picked thing, "build" or "page".
    """
    window = Export(ui, session, start)
    ui.open(window)
    return window
