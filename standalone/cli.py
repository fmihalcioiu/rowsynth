"""What the command line of the standalone program starts with: rowsynth-cli, in the folder `bin`."""
import sys

from rowsynth.program import command_line

sys.exit(command_line())
