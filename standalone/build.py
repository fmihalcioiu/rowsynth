"""Builds the standalone program for the system this is run on: a folder that needs no Python, and a zip of it.

    python standalone/build.py

It needs PyInstaller and Pillow, which are no part of rowsynth: `pip install pyinstaller pillow`. The folder is
`dist/rowsynth`, and the zip beside it is what goes on a release. What is printed is what was written.
"""
from __future__ import annotations

import json
import platform
import shutil
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: The sides of the pictures in the icon of the program, in pixels, each with the sprite of the set `app` it is
#: made of: a whole number of times that sprite's side, so that no pixel of it is smoothed.
SIDES = ((16, "small"), (32, "icon"), (48, "small"), (64, "icon"), (128, "icon"), (256, "icon"))
#: What goes in the folder beside the two programs, each as (where it is in the repo, where it goes).
BESIDE = (("examples", "examples"), ("skills/rowsynth", "skill"), ("LICENSE", "LICENSE.txt"), ("README.md", "README.md"),
          # The picture the README shows, where the README looks for it.
          ("docs/window.png", "docs/window.png"))
#: What the systems are called in the name of the zip.
SYSTEMS = {"Windows": "windows", "Darwin": "macos", "Linux": "linux"}


def version() -> str:
    """The version of the package, as `pyproject.toml` has it."""
    with open(ROOT / "pyproject.toml", "rb") as file:
        return tomllib.load(file)["project"]["version"]


def icon(path: Path) -> Path:
    """Write the icon of the program: the picture the window has for its own, at each of `SIDES`, with no smoothing."""
    from PIL import Image

    from rowsynth.gui import sprites

    def picture(name: str) -> Image.Image:
        drawn = sprites.load("app").sprites[name]
        made = Image.new("RGBA", (len(drawn.rows[0]), len(drawn.rows)), (0, 0, 0, 0))
        for y, row in enumerate(drawn.rows):
            for x, ink in enumerate(row):
                if ink != sprites.EMPTY:
                    made.putpixel((x, y), (*drawn.palette[ink], 255))
        return made

    sized = [picture(name).resize((side, side), Image.NEAREST) for side, name in SIDES]
    path.parent.mkdir(parents=True, exist_ok=True)
    sized[-1].save(path, format="ICO", sizes=[(side, side) for side, _ in SIDES], append_images=sized[:-1])
    return path


def build() -> dict:
    """Make the icon, the folder and the zip, and say where each is and how big the zip came out."""
    import PyInstaller.__main__

    made = icon(ROOT / "build" / "rowsynth.ico")
    PyInstaller.__main__.run([str(ROOT / "standalone" / "rowsynth.spec"), "--noconfirm",
                              "--distpath", str(ROOT / "dist"), "--workpath", str(ROOT / "build")])
    folder = ROOT / "dist" / "rowsynth"
    for source, name in BESIDE:
        here, there = ROOT / source, folder / name
        if here.is_dir():
            shutil.copytree(here, there, dirs_exist_ok=True)
        else:
            there.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(here, there)
    system = SYSTEMS.get(platform.system(), platform.system().lower())
    packed = shutil.make_archive(str(ROOT / "dist" / f"rowsynth-{version()}-{system}"), "zip", folder.parent,
                                 folder.name)
    return {"icon": str(made), "folder": str(folder), "zip": packed,
            "megabytes": round(Path(packed).stat().st_size / 1e6, 1)}


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
