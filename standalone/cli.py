"""What the command line of the standalone program starts with: rowsynth, or rowsynth.exe on Windows."""
import sys

from rowsynth.program import command_line

sys.exit(command_line())
