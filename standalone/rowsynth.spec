# What PyInstaller is told to make: one folder with two programs in it, which share everything else there.
# `build.py` beside this file runs it, after it has made the icon this names.
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

HERE = Path(SPECPATH)
ROOT = HERE.parent
ICON = str(ROOT / "build" / "rowsynth.ico")
# The commands and the checks are found by name as the program runs, so none of them is seen by reading the code.
HIDDEN = collect_submodules("rowsynth")
# The art of the window: the set files in the package, and nothing else that lies about in it, as what Numba has
# compiled for the machine that builds does.
DATA = collect_data_files("rowsynth", includes=["**/*.json"])
# What is installed beside the package on a machine that builds, and is no part of it.
LEFT_OUT = ["tkinter", "matplotlib", "scipy", "pandas", "PIL", "IPython", "pytest", "pip"]


def read(script):
    """What one of the two programs needs, found from the script it starts with."""
    return Analysis([str(HERE / script)], pathex=[str(ROOT)], hiddenimports=HIDDEN, datas=DATA, excludes=LEFT_OUT)


programs, shared = [], []
# The command line has a console. The window has none: a double click on it opens no black box behind it.
for script, name, console in (("cli.py", "rowsynth", True), ("window.py", "rowsynth-window", False)):
    found = read(script)
    programs.append(EXE(PYZ(found.pure), found.scripts, [], exclude_binaries=True, name=name, console=console,
                        icon=ICON, upx=False))
    shared += [found.binaries, found.datas]

COLLECT(*programs, *shared, name="rowsynth", upx=False)
