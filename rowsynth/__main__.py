"""What runs when the package itself is run: python -m rowsynth <command> ..."""
from __future__ import annotations

import sys

from .cli import main

sys.exit(main())
