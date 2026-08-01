"""Put `src/` on the path so the suite runs from a clean checkout without an
editable install. `pip install -e .` also works and takes precedence."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
