"""`PYTHONPATH=src python -m backtest` and `python -m src.backtest`."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from backtest.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
