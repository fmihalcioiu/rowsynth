"""A listen page: every sound and every song of a project as WAV files, and one HTML page with a player for each."""
from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np

from .bank import _to_peak, render_bank
from .errors import RowSynthError
from .layers import mix_version, render_groups, render_intro
from .measure import measure
from .project import Project
from .song import render_loop
from .wav import save_wav

# The samples a second the audio of a page is rendered at, when no rate is asked for.
PAGE_RATE = 48000

# The buffers of a held sound, in the order the bank gives them.
_HELD = ("start", "loop", "end")

# The folder inside audio/ that the files of the songs go in. Those of the sounds are in audio/ itself, and a
# name has no "/" in it, so a sound and a song of one name have a file each.
_SONGS = "songs"

# Where the loudest sample among the files of one song is brought to, in decibels against full scale.
_PEAK_DB = -1.0

# The page down to its heading. Its colours are the browser's own, light or dark as the browser is. Under 36em
# a row is too narrow to be a line of a table: the name, the player and the three figures are stacked, and
# each figure says what it is, since the headings of the columns are gone.
_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Listen page</title>
<style>
:root { color-scheme: light dark; }
body { max-width: 40em; margin: 0 auto; padding: 1em; font: 1em/1.5 system-ui, sans-serif;
  color: CanvasText; background: Canvas; }
h1 { margin: 0.5em 0; font-size: 1.6em; }
h2 { margin: 1.5em 0 0.5em; font-size: 1.2em; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 0.4em 0.5em; text-align: left; white-space: nowrap; }
thead th, small { color: GrayText; font-size: 0.85em; font-weight: normal; }
tbody tr { border-top: 1px solid GrayText; }
td, thead th:nth-child(n+3) { text-align: right; font-variant-numeric: tabular-nums; }
td.player { width: 100%; }
audio { display: block; width: 100%; min-width: 12em; }
@media (max-width: 36em) {
  table, tbody { display: block; }
  thead { display: none; }
  tbody tr { display: flex; flex-wrap: wrap; padding: 0.4em 0; }
  th, td { padding: 0.2em 0; }
  tbody th, td.player { flex: 0 0 100%; white-space: normal; overflow-wrap: anywhere; }
  td[data-label] { flex: 1; text-align: left; }
  td[data-label]::before { content: attr(data-label); display: block; color: GrayText; font-size: 0.85em; }
}
</style>
</head>
<body>
<h1>Listen page</h1>
"""

_FOOT = """\
</body>
</html>
"""


@dataclass(frozen=True, eq=False)
class _Row:
    """One row of the page: one audio file, and what the page says of it.

    name is the sound or the song the file is of, and which says which of its files this is, with nothing to
    say when it is the only one or the thing itself. file is where the audio goes inside audio/, with a "/"
    after a folder. sides are the samples of the file: one side is written mono, and two are the left and the
    right. loop is whether the player goes round. figures are what `measure` read as the row was made.
    """

    name: str
    which: str
    file: str
    sides: tuple[np.ndarray, ...]
    loop: bool
    figures: dict


def _row(name: str, which: str, file: str, sides: tuple[np.ndarray, ...], loop: bool, rate: int) -> _Row:
    """A row with the figures of its sides as they are now: of its one side, or of the mean of its two."""
    return _Row(name, which, file, sides, loop, measure(np.mean(sides, axis=0, dtype=np.float64), rate))


def _parts(entry: dict, count: int) -> list[int | str | None]:
    """Which one each of the `count` buffers of an entry is.

    A buffer of a held sound goes by its name, a variant by its number, counted from 0, and the one buffer of
    any other sound by None.
    """
    if "held" in entry:
        return list(_HELD)
    if "defs" in entry:
        return list(range(count))
    return [None]


def _file(name: str, part: int | str | None, joint: str = "-") -> str:
    """The name of the file of a row: the name of its sound or its song, then, after `joint`, which one of it."""
    return f"{name}.wav" if part is None else f"{name}{joint}{part}.wav"


def _sound_rows(project: Project, rate: int) -> list[_Row]:
    """The rows of the sounds, in the order of the project: one for each buffer of the bank that has samples."""
    entries = {name: project.sound(name) for name in project.sounds}
    rows = []
    for name, sound in render_bank(entries, rate).items():
        buffers = sound["buffers"]
        for part, samples in zip(_parts(entries[name], len(buffers)), buffers):
            if len(samples):
                which = f"variant {part}" if isinstance(part, int) else part or ""
                rows.append(_row(name, which, _file(name, part), (samples,), part == "loop", rate))
    return rows


def _mixes(song: dict, rate: int) -> list[tuple[str | None, str, tuple[np.ndarray, np.ndarray], bool]]:
    """What there is to hear of a song, in the order of its rows: each mix as (part, which, sides, loop).

    The song as one loop comes first. After it a song with layers has each of its versions in each of its
    keys, and a song with an intro has that last. part is what a mix adds to the name of its file, None for
    the loop of the song itself. which is what its row says of it. Only the intro is not made to go round.
    """
    whole = render_loop(song, rate)
    mixes = [(None, "", (whole["left"], whole["right"]), True)]
    if "layers" in song:
        rendered = render_groups(song, rate)
        for version in song["layers"].get("versions", {}):
            for key in rendered["keys"]:
                # A key is a number of semitones up, and a row says which way it goes: "key +2", "key -5".
                which = f"{version}, key {key:+}" if key else f"{version}, key 0"
                mixes.append((f"{version}.{key}", which, mix_version(rendered, song, version, key), True))
    if "intro" in song:
        intro = render_intro(song, rate)
        mixes.append(("intro", "intro", (intro["left"], intro["right"]), False))
    return mixes


def _song_rows(project: Project, rate: int) -> list[_Row]:
    """The rows of the songs, in the order of the project: one for each mix of each song.

    The figures of a row are of its mix at the level the song's data gives it. The files are at another. Every
    mix of a song is multiplied by one factor, the one that brings the loudest sample among them to _PEAK_DB:
    no file clips then, and the mixes of a song are as loud beside each other as they were. A song that is
    all silence has no loudest sample to go by, and is left as it is. A song that cannot be rendered is
    refused with its place in the project in front.
    """
    rows = []
    for name in project.songs:
        try:
            made = [_row(name, which, f"{_SONGS}/{_file(name, part, '.')}", sides, loop, rate)
                    for part, which, sides, loop in _mixes(project.song(name), rate)]
        except RowSynthError as error:
            raise RowSynthError(f"songs.{name}: {error}") from None
        # The figures are read by now. What is scaled here, in place, is what goes into the files.
        _to_peak([side for row in made for side in row.sides], _PEAK_DB)
        rows += made
    return rows


def _apart(rows: list[_Row]) -> None:
    """Refuses rows of which two would have one file.

    Names that differ only in case are one name to a system that does not tell an "A" from an "a", and there
    the file written second would take the place of the first. A project is to make the same page wherever it
    is built, so such names are refused everywhere.
    """
    taken: dict[str, _Row] = {}
    for row in rows:
        first = taken.setdefault(row.file.casefold(), row)
        if first is not row:
            raise RowSynthError(f"audio/{first.file} and audio/{row.file}: two rows of the page cannot have one "
                                "file, and these names are the same once case is set aside")


def _html(row: _Row) -> str:
    """A row as a line of its table: a name, a player, and the length and the levels of its figures."""
    label = html.escape(row.name) + (f" <small>{html.escape(row.which)}</small>" if row.which else "")
    # A browser would read a "#" or a "%" in a file name as part of the address, so such a character goes in
    # as its code.
    src = html.escape(f"audio/{quote(row.file)}")
    loop = " loop" if row.loop else ""
    figures = row.figures
    return (f'<tr><th scope="row">{label}</th>'
            f'<td class="player"><audio controls{loop} preload="none" src="{src}"></audio></td>'
            f'<td data-label="Length">{figures["seconds"]:.3f} s</td>'
            f'<td data-label="Peak">{figures["peak_db"]:.1f} dB</td>'
            f'<td data-label="Loudness">{figures["rms_db"]:.1f} dB</td></tr>\n')


def _section(kind: str, rows: list[_Row]) -> str:
    """The section of the sounds or of the songs, as `kind` says: a table of its rows, or a line when it has none."""
    title = kind.capitalize()
    if not rows:
        return f"<section>\n<h2>{title}s</h2>\n<p>This project has no {kind}s.</p>\n</section>\n"
    columns = "".join(f'<th scope="col">{column}</th>' for column in (title, "Player", "Length", "Peak", "Loudness"))
    return (f"<section>\n<h2>{title}s</h2>\n<table>\n"
            f"<thead>\n<tr>{columns}</tr>\n</thead>\n"
            f"<tbody>\n{''.join(map(_html, rows))}</tbody>\n"
            "</table>\n</section>\n")


def write_page(project: Project, out_dir: str | Path, rate: int = PAGE_RATE) -> dict:
    """Writes a page to listen to a project by, `out_dir/index.html`, and says what it wrote.

    The page has a section of the sounds and after it one of the songs. Each has a row for every audio file,
    in the order of the project: the name of the sound or the song, which of its files this is, a player, and
    the length, the peak and the loudness that `measure` reads. A section with no rows has a line that says
    so. The page is one file of HTML, and loads nothing but the audio beside it.

    Every sound is rendered by the bank at `rate`, so that it is heard at the level it would be played at.
    Each of its buffers is a mono WAV file in `out_dir/audio/`: NAME.wav for one sound or a sound of parts,
    NAME-0.wav, NAME-1.wav and so on for variants, and NAME-start.wav, NAME-loop.wav and NAME-end.wav for a
    held sound. A buffer of no samples gets no file and no row. Only the start of a held sound can be one,
    when the sound starts at 0. The figures of a row are those of its buffer. The player of a held sound's
    loop goes round.

    Every song is a stereo WAV file in `out_dir/audio/songs/`, NAME.wav: the loop that `render_loop` makes of
    it. A song with layers has a file for each of its versions in each of its keys after that,
    NAME.VERSION.KEY.wav, whose row says "VERSION, key +2". A song with an intro has NAME.intro.wav last. The
    player of an intro stops at its end, and that of every other file of a song goes round. The files of one
    song are all written at one level: each is its mix times the one factor that brings the loudest sample
    among them to 1 dB under full scale, so that none of them clips. A song that is all silence is left as it
    is. The figures of a song's row are not those of its file: they are of the mix before that factor, at the
    level the song's data gives it, read from the mean of its two sides.

    Everything is rendered before anything is written. A sound that cannot be rendered is refused with its
    name in front, and a song with "songs." and its name. So are two rows whose files would have one name
    with case set aside. Files that are there already are written over.
    What comes back is {"page": path, "files": [path, ...]}: the page, and the audio files in the order of
    their rows, those of the sounds first. It holds what this call wrote and no more, so a file that an
    earlier page left in the folder is not in it.
    """
    sounds = _sound_rows(project, rate)
    songs = _song_rows(project, rate)
    _apart(sounds + songs)
    out_dir = Path(out_dir)
    files = []
    for row in sounds + songs:
        path = out_dir / "audio" / row.file
        # `save_wav` takes one side on its own and two as a pair.
        save_wav(path, row.sides[0] if len(row.sides) == 1 else row.sides, rate)
        files.append(str(path))
    # With no sounds and no songs, nothing has made the folder yet.
    out_dir.mkdir(parents=True, exist_ok=True)
    page = out_dir / "index.html"
    # Written as ASCII, a character outside it as its number, so the page reads the same whatever a browser
    # takes it to be written in.
    body = _section("sound", sounds) + _section("song", songs)
    page.write_text(_HEAD + body + _FOOT, encoding="ascii", errors="xmlcharrefreplace", newline="\n")
    return {"page": str(page), "files": files}
