"""A chip synth driven by text."""

# The version comes before the imports: the build stamps it into what it writes, and reads it from here.
__version__ = "0.1.0"

from .bank import render_bank, render_entry
from .build import build
from .errors import RowSynthError
from .measure import measure
from .page import write_page
from .project import Project
from .song import render_loop, render_song, render_stems
from .sound import render_sound
from .wav import load_wav, save_wav

__all__ = [
    "Project",
    "RowSynthError",
    "build",
    "load_wav",
    "measure",
    "render_bank",
    "render_entry",
    "render_loop",
    "render_song",
    "render_sound",
    "render_stems",
    "save_wav",
    "write_page",
]
