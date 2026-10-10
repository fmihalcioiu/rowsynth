"""The look of the GUI: its colours by name, and the sizes of its frame."""
from __future__ import annotations

Colour = tuple[int, int, int]


def _rgb(text: str) -> Colour:
    """A colour written as "#rrggbb", as its red, its green and its blue, each from 0 to 255."""
    return int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16)


# A warm hardware look: charcoal panels and cream ink. The panels come in four shades, from the darkest: what
# lies behind everything, a panel, a button on a panel, and what is picked or lit.
BACKGROUND = _rgb("#191816")
PANEL = _rgb("#262422")
RAISED = _rgb("#33302d")
LIT = _rgb("#44403b")
LINE = _rgb("#5b554d")

TEXT = _rgb("#efe6d5")
DIM = _rgb("#a0988a")

# What is switched on or playing, what is as it should be, and what is wrong.
ACTIVE = _rgb("#f08c2e")
GOOD = _rgb("#8fbf4f")
FAULT = _rgb("#e5483a")

# The three kinds of thing a project holds.
SONG = _rgb("#6aa5d8")
SOUND = _rgb("#4fb3a2")
INSTRUMENT = _rgb("#c9a36b")

SALMON = _rgb("#e58f7b")
MUSTARD = _rgb("#e9b44c")

# The channels of a song take these in turn, and start again after the last.
CHANNELS = (SONG, SOUND, SALMON, MUSTARD)
# The steps of a drum row are coloured in fours: the first four of a bar take the first colour, and so on.
STEPS = (FAULT, ACTIVE, MUSTARD, TEXT)

# The frame, in art pixels: the bars along the top and the bottom, and the panels down the two sides.
TOP_H = 22
STATUS_H = 14
LEFT_W = 124
RIGHT_W = 174

# The art size the layout is drawn for. A full HD window shows it at twice its size.
DESIGN = (960, 540)
# The least art size a scale is chosen at. It is less high than the layout is drawn for, so that a window filling
# a full HD desktop, where the title bar and the task bar take their share, is still shown at twice.
MIN_ART = (960, 480)
# What a desktop loses to the title bar of a window and to the task bar, in window pixels of height: an allowance.
DESKTOP_TAKEN_H = 80
