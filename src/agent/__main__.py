"""`PYTHONPATH=src python -m agent` and `python -m src.agent`."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
