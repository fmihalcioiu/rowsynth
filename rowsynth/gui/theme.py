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
# How wide the meter of the top bar is: its outline, and bars 60 pixels long. It is as high as a button.
BAR_METER_W = 62
# The left panel: from its sides and its foot to the list in it, the panel's line not counted, and from its top.
BROWSER_PAD = 2
BROWSER_TOP = 4

# A menu that opens at a place. A row of it is this high, and the strip that holds the line between two groups
# this high. From a side of its box to what a row shows, and from its top and its foot to the first and the last
# row, the outline counted both times. The side of the square an icon has before the texts, in a menu that has
# icons. A menu is never narrower than MENU_MIN_W.
MENU_ROW_H = 12
MENU_LINE_H = 5
MENU_PAD = 6
MENU_EDGE = 2
MENU_ICON = 9
MENU_MIN_W = 48
# A box that is typed into. From a side of it to where the caret stands before the first character, the outline
# counted. One that opens over a name is this high, and never narrower than TEXT_BOX_MIN_W.
TEXT_PAD = 3
TEXT_BOX_H = 13
TEXT_BOX_MIN_W = 72

# A press that follows another press on the same thing within this many seconds is a double press, where the
# mouse has strayed no more than DOUBLE_REACH pixels either way between the two.
DOUBLE_PRESS = 0.4
DOUBLE_REACH = 3

# The skeleton the editors of a song, a sound and an instrument share, from the top: a heading of one line, the bar
# of the settings, the main area with a column down its side, and a foot. How high the heading, the bar and the
# foot are, and how wide the side column is.
HEAD_H = 17
BAR_H = 40
FOOT_H = 72
SIDE_W = 124
# A small button, as a heading and a settings bar have them: how high it is. A button that opens a list has a small
# arrow after its text that points down: how wide the arrow is at its back. It is half that high, rounded up.
BUTTON_H = 13
DOWN_ARROW = 5
# A heading: from its kind to its name, and from its name to the facts after it.
HEAD_SPACE = 6
HEAD_FACTS = 12
# A settings bar. From its top to the row of its controls, and how high that row is: a knob with the line of its
# name over it and the line of its value under it. From one control to the next.
BAR_TOP = 2
BAR_ROW_H = 37
BAR_GAP = 6
# The upright keyboard beside a grid: how wide it is, how high a key is, and the share of its width a black key
# takes. A key lower than KEY_LINES_FROM has no outline. How much of its light a white key keeps under the mouse.
KEYS_W = 40
KEY_H = 11
KEY_BLACK = 0.6
KEY_LINES_FROM = 4
KEY_HOT = 0.85
# The scroll bar of a grid: how thick it is, and how far it keeps from the grid.
SCROLL_W = 4
SCROLL_GAP = 1
# The velocity lane: how wide the bar of a note is where the note is not narrower, and from the outline of the lane
# to the bars, the outline counted.
VELOCITY_W = 3
VELOCITY_PAD = 2
# The small button that adds one more of something: its side, and how long each of the two strokes of its plus.
PLUS = 11
PLUS_STROKE = 5
# The file chooser. How wide and how high its box is, and from a side of it to what it shows. How high the line
# of its title is, and each of the two under it: the folder it is in, and what it has to say. Between two of its
# parts. How high the row at its foot is, where the name is typed, and how wide each of the two buttons there.
CHOOSER = (340, 280)
CHOOSER_PAD = 8
CHOOSER_LINE_H = 12
CHOOSER_GAP = 5
CHOOSER_FOOT_H = 16
CHOOSER_BUTTON_W = 54
# The square before a folder in the chooser, and before a project file.
FOLDER = MUSTARD
PROJECT_FILE = SONG

# The first screen, which stands in place of the panels while no project is open. How many times its size the
# program's name is drawn, and how wide the column under it is. From the name to the two buttons, and from those
# to the list. How high a button is, and how high the list at the most.
FIRST_NAME_SCALE = 3
FIRST_W = 260
FIRST_GAP = 18
FIRST_BUTTON_H = 16
FIRST_LIST_H = 164
