"""A listen page: every sound of a project as a WAV file, and one HTML page with a player for each."""
from __future__ import annotations

import html
from pathlib import Path
from urllib.parse import quote

import numpy as np

from .bank import render_bank
from .measure import measure
from .project import Project
from .wav import save_wav

# The buffers of a held sound, in the order the bank gives them.
_HELD = ("start", "loop", "end")

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


def _file(name: str, part: int | str | None) -> str:
    """The name of the file a buffer goes in: the name of its sound, then which one of that sound it is."""
    return f"{name}.wav" if part is None else f"{name}-{part}.wav"


def _row(name: str, part: int | str | None, file: str, samples: np.ndarray, rate: int) -> str:
    """One row of the table of sounds: a name, a player, and the length and the levels `measure` reads."""
    which = f"variant {part}" if isinstance(part, int) else part
    label = html.escape(name) + (f" <small>{html.escape(which)}</small>" if which else "")
    # A browser would read a "#" or a "%" in a file name as part of the address, so such a character goes in
    # as its code.
    src = html.escape(f"audio/{quote(file)}")
    loop = " loop" if part == "loop" else ""
    figures = measure(samples, rate)
    return (f'<tr><th scope="row">{label}</th>'
            f'<td class="player"><audio controls{loop} preload="none" src="{src}"></audio></td>'
            f'<td data-label="Length">{figures["seconds"]:.3f} s</td>'
            f'<td data-label="Peak">{figures["peak_db"]:.1f} dB</td>'
            f'<td data-label="Loudness">{figures["rms_db"]:.1f} dB</td></tr>\n')


def _sounds(rows: list[str]) -> str:
    """The section of the sounds: a table of the rows, or a line in its place when there are none."""
    if not rows:
        return "<section>\n<h2>Sounds</h2>\n<p>This project has no sounds.</p>\n</section>\n"
    columns = "".join(f'<th scope="col">{column}</th>' for column in ("Sound", "Player", "Length", "Peak", "Loudness"))
    return ("<section>\n<h2>Sounds</h2>\n<table>\n"
            f"<thead>\n<tr>{columns}</tr>\n</thead>\n"
            f"<tbody>\n{''.join(rows)}</tbody>\n"
            "</table>\n</section>\n")


def write_page(project: Project, out_dir: str | Path, rate: int = 48000) -> str:
    """Writes a page to listen to a project by, `out_dir/index.html`, and gives the path of that page.

    Every sound is rendered by the bank at `rate`, so that it is heard at the level it would be played at.
    Each of its buffers is a mono WAV file in `out_dir/audio/`: NAME.wav for one sound or a sound of parts,
    NAME-0.wav, NAME-1.wav and so on for variants, and NAME-start.wav, NAME-loop.wav and NAME-end.wav for a
    held sound. A buffer of no samples gets no file and no row. Only the start of a held sound can be one,
    when the sound starts at 0.
    The page has a row for each file, in the order of the project: the name of the sound, which of its buffers
    this is, a player, and the length, the peak and the loudness `measure` reads from the buffer. The player
    of a held sound's loop goes round. The page is one file of HTML, and loads nothing but the audio beside it.
    Files that are there already are written over. A sound that cannot be rendered is refused with its name in
    front, before anything is written.
    """
    entries = {name: project.sound(name) for name in project.sounds}
    bank = render_bank(entries, rate)
    out_dir = Path(out_dir)
    rows = []
    for name, sound in bank.items():
        buffers = sound["buffers"]
        for part, samples in zip(_parts(entries[name], len(buffers)), buffers):
            if not len(samples):
                continue
            file = _file(name, part)
            save_wav(out_dir / "audio" / file, samples, rate)
            rows.append(_row(name, part, file, samples, rate))
    # With no sounds, nothing has made the folder yet.
    out_dir.mkdir(parents=True, exist_ok=True)
    page = out_dir / "index.html"
    # Written as ASCII, a character outside it as its number, so the page reads the same whatever a browser
    # takes it to be written in.
    page.write_text(_HEAD + _sounds(rows) + _FOOT, encoding="ascii", errors="xmlcharrefreplace", newline="\n")
    return str(page)
