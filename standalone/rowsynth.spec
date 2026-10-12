# What PyInstaller is told to make: one folder with the window in it, and a folder in that with everything the
# window needs to run. `build.py` beside this file runs it, after it has made the icon this names, and then puts
# the command line where it belongs.
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

HERE = Path(SPECPATH)
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from rowsynth.program import COMMAND_LINE, HELD, WINDOW

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
# The window comes first: the folder is laid out as it asks, with what it needs in a folder of its own. It has the
# icon and no console, so a double click on it opens no black box behind it. The command line has a console and
# the plain picture the system gives a program, and it looks for what it needs beside itself: `build.py` moves it
# into the folder of the window's.
for script, name, console, icon, held in (("window.py", WINDOW, False, ICON, HELD),
                                          ("cli.py", COMMAND_LINE, True, "NONE", ".")):
    found = read(script)
    programs.append(EXE(PYZ(found.pure), found.scripts, [], exclude_binaries=True, name=name, console=console,
                        icon=icon, contents_directory=held, upx=False))
    shared += [found.binaries, found.datas]

COLLECT(*programs, *shared, name="rowsynth", upx=False)
