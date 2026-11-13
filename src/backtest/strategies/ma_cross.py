"""Long-only moving-average cross. Windows for the pipe come from config.yaml."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from config_loader import load_config


def _windows(short_window: int, long_window: int) -> tuple[int, int]:
    if isinstance(short_window, bool) or isinstance(long_window, bool):
        raise ValueError("short_window must be a positive integer shorter than long_window")
    try:
        short = int(short_window)
        long = int(long_window)
    except (TypeError, ValueError) as exc:
        raise ValueError("short_window must be a positive integer shorter than long_window") from exc
    if short != short_window or long != long_window:
        raise ValueError("short_window must be a positive integer shorter than long_window")
    if short < 1 or long < 1 or short >= long:
        raise ValueError("short_window must be a positive integer shorter than long_window")
    return short, long


def ma_cross(data: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    """Signal at the close of the cross. The runner fills the order on the next bar."""
    short, long = _windows(short_window, long_window)
    close = pd.Series(pd.to_numeric(data["close"]), index=data.index, dtype="float64")
    fast = close.rolling(short, min_periods=short).mean()
    slow = close.rolling(long, min_periods=long).mean()
    fast_prev = fast.shift(1)
    slow_prev = slow.shift(1)
    entries = ((fast > slow) & (fast_prev <= slow_prev)).fillna(False)
    exits = ((fast < slow) & (fast_prev >= slow_prev)).fillna(False)
    return pd.DataFrame(
        {"entries": entries.astype(bool), "exits": exits.astype(bool)},
        index=data.index,
    )


def ma_cross_from_config() -> Callable[[pd.DataFrame], pd.DataFrame]:
    windows = load_config()["strategy"]["ma_cross"]

    def strategy(data: pd.DataFrame) -> pd.DataFrame:
        return ma_cross(
            data,
            short_window=windows["short_window"],
            long_window=windows["long_window"],
        )

    return strategy
