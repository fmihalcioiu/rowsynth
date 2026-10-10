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


def shade(colour: Colour, factor: float) -> Colour:
    """A colour made darker: each of its three parts times `factor`, which is from 0 to 1."""
    red, green, blue = colour
    return int(red * factor), int(green * factor), int(blue * factor)


# The frame, in art pixels: the bars along the top and the bottom, and the panels down the two sides.
TOP_H = 22
STATUS_H = 14
LEFT_W = 124
RIGHT_W = 174

# A button: from a side of it to what is on it, the outline counted; between its icon and its text; and how thick
# the strip along the lower edge of a toggle that is on.
PAD = 4
GAP = 3
STRIP = 2

# A dial is dragged: this many art pixels take it through its whole range, and with shift held it goes at this
# share of that speed. A step of the wheel is one of this many in the range, for a dial with no steps of its own.
DRAG_TRAVEL = 100
FINE = 0.1
WHEEL_STEPS = 100

# A knob: how many pixels across. It turns from KNOB_START, in degrees counted from three o'clock against the
# clock, through KNOB_SWEEP degrees with the clock: from seven o'clock over the top to five. Its pointer is a
# pixel at each of these distances from its centre. Its travel is this many marks, this far from its centre.
# Its label and its value are this far from the marks.
KNOB = 13
KNOB_START = 225
KNOB_SWEEP = 270
KNOB_POINTER = (2, 3, 4)
KNOB_MARKS = 11
KNOB_MARKS_AT = 8
KNOB_GAP = 1

# A slider: how thick its track, how long its handle along the track when it lies and when it stands upright,
# and how dark the track is from its low end to the handle, as a share of the slider's colour.
TRACK = 2
HANDLE_LYING = 4
HANDLE_UPRIGHT = 6
TRAVEL = 0.7

# A meter is even in decibels: its foot is at METER_FLOOR_DB and its top at 0 dB. Its bar turns mustard from
# METER_WARM_DB and red from METER_HOT_DB. Two bars stand METER_GAP apart. The line kept at the top of a bar
# falls back by METER_FALL_DB decibels a second.
METER_FLOOR_DB = -48
METER_WARM_DB = -12
METER_HOT_DB = -3
METER_GAP = 1
METER_FALL_DB = 12

# The arrow at each end of a stepper: how many pixels from its tip to its back. It is twice that less 1 high.
ARROW = 3

# The art size the layout is drawn for. A full HD window shows it at twice its size.
DESIGN = (960, 540)
# The least art size a scale is chosen at. It is less high than the layout is drawn for, so that a window filling
# a full HD desktop, where the title bar and the task bar take their share, is still shown at twice.
MIN_ART = (960, 480)
# What a desktop loses to the title bar of a window and to the task bar, in window pixels of height: an allowance.
# The window is centred on the desktop, so half of it is left free above and half below, and each half must
# clear a task bar of 48.
DESKTOP_TAKEN_H = 96

# A list that scrolls, in art pixels. A row is this high, and a heading that is not the first thing in the list
# has this much air above it.
LIST_ROW_H = 12
LIST_GAP = 8
# Along a row: from a side of the list to what the row shows, the side of the square before a row (the mark before
# a heading has the same room), from that to the text and from the text to the tag, and one step of indent.
LIST_PAD = 6
LIST_SQUARE = 6
LIST_SPACE = 5
LIST_INDENT = 8
# The mark before a heading is an arrowhead this long. The picked row has a strip this wide down its left side.
LIST_MARK = 5
LIST_STRIP = 2
# The bar at the right of a list whose rows do not fit: how wide it is, and the least length of its thumb.
LIST_BAR_W = 4
LIST_THUMB_MIN = 8
# How many rows one step of the wheel scrolls.
LIST_WHEEL = 3

# A bar, the one along the top or the one along the bottom: from an end of it to the first thing it shows.
BAR_MARGIN = 6
# The top bar. How wide and how high a button is that shows an icon and no text. Between two buttons that stand
# together, and between two groups of things. How wide the findings are: an icon, the gap, and a count of three
# figures.
ICON_BUTTON = (17, 16)
BUTTON_GAP = 2
GROUP_GAP = 12
FOUND_W = 29
# The left panel: from its sides and its foot to the list in it, the panel's line not counted, and from its top.
BROWSER_PAD = 2
BROWSER_TOP = 4
