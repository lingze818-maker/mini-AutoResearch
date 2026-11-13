from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BARS_FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_daily_bars.csv"
FIXTURES = ROOT / "tests" / "fixtures"


def bars_from_closes(closes: list[float], start: str = "2023-01-02") -> pd.DataFrame:
    close = pd.Series(closes, dtype="float64")
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) * 1.001
    low = pd.concat([open_, close], axis=1).min(axis=1) * 0.999
    dates = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_.to_numpy(),
            "high": high.to_numpy(),
            "low": low.to_numpy(),
            "close": close.to_numpy(),
            "volume": 1_000_000,
        }
    )
