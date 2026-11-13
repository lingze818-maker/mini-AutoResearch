"""One backtest entry point. Every strategy is simulated here with vectorbt."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import vectorbt as vbt

from eval.metrics import average_turnover, count_entries, max_drawdown, period_return

REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume")
Strategy = Callable[..., pd.DataFrame]


@dataclass(frozen=True, slots=True)
class BacktestResult:
    curve: pd.DataFrame
    trade_count: int
    turnover: float
    max_drawdown: float
    period_return: float


def validate_bars(data: pd.DataFrame) -> pd.DataFrame:
    """Require a clean daily table. Prices are used as given and are assumed qfq."""
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    out = data.loc[:, list(REQUIRED_COLUMNS)].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if getattr(out["date"].dt, "tz", None) is not None:
        out["date"] = out["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    out["date"] = out["date"].dt.normalize()
    if out["date"].isna().any():
        raise ValueError("dates must be parseable")
    if out.empty:
        raise ValueError("price data is empty")
    if out["date"].duplicated().any() or not out["date"].is_monotonic_increasing:
        raise ValueError("dates must be strictly increasing and unique")
    for column in ("open", "high", "low", "close", "volume"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if out[["open", "high", "low", "close", "volume"]].isna().any().any():
        raise ValueError("prices and volume must be numeric")
    if (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("prices must be positive")
    if (out["volume"] < 0).any():
        raise ValueError("volume must be non-negative")
    upper = out[["open", "close"]].max(axis=1)
    lower = out[["open", "close"]].min(axis=1)
    if (out["high"] + 1e-9 < upper).any() or (out["low"] - 1e-9 > lower).any():
        raise ValueError("high and low must contain open and close")
    return out.reset_index(drop=True)


def load_bars(path: str | Path) -> pd.DataFrame:
    return validate_bars(pd.read_csv(path))


def _as_series(value: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(value, pd.DataFrame):
        if value.shape[1] != 1:
            raise ValueError("expected a single asset")
        return value.iloc[:, 0]
    return pd.Series(value)


def _boolean_column(values: pd.Series) -> pd.Series:
    if values.isna().any():
        raise ValueError("strategy signals contain missing values")
    if values.dtype != bool:
        allowed = {0, 1, False, True, 0.0, 1.0}
        if not set(pd.unique(values)).issubset(allowed):
            raise ValueError("entries and exits must be boolean")
    return values.astype(bool)


def _signals(signals: pd.DataFrame, length: int) -> tuple[pd.Series, pd.Series]:
    if not isinstance(signals, pd.DataFrame) or not {"entries", "exits"}.issubset(signals.columns):
        raise ValueError("strategy must return entries and exits columns")
    if len(signals) != length:
        raise ValueError("strategy signals must have one row per bar")
    frame = signals.loc[:, ["entries", "exits"]].reset_index(drop=True)
    return _boolean_column(frame["entries"]), _boolean_column(frame["exits"])


def run_backtest(
    data: pd.DataFrame,
    strategy: Strategy,
    params: Mapping[str, object] | None = None,
) -> BacktestResult:
    """Simulate one long-only strategy.

    `strategy(data)` or `strategy(data, **params)` returns entries and exits known
    at that bar's close. vectorbt fills those orders at the next bar's close.
    The returned curve is a NAV that starts at 1 while the book is in cash.
    """
    if not callable(strategy):
        raise ValueError("strategy must be a function")
    bars = validate_bars(data)
    if params is None:
        signals = strategy(bars)
    else:
        signals = strategy(bars, **dict(params))
    raw_entries, raw_exits = _signals(signals, len(bars))
    close = pd.Series(bars["close"].to_numpy(dtype=float), index=pd.DatetimeIndex(bars["date"]))
    entries = pd.Series(raw_entries.to_numpy(), index=close.index).shift(1, fill_value=False).astype(bool)
    exits = pd.Series(raw_exits.to_numpy(), index=close.index).shift(1, fill_value=False).astype(bool)
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        init_cash=1.0,
        fees=0.0,
        slippage=0.0,
        freq="1D",
        direction="longonly",
    )
    equity = _as_series(portfolio.value()).reindex(close.index).astype(float)
    exposure = _as_series(portfolio.gross_exposure()).reindex(close.index).fillna(0.0).astype(float)
    position = (exposure > 1e-8).astype(float)
    returns = equity.pct_change().fillna(0.0)
    curve = pd.DataFrame(
        {
            "date": close.index,
            "equity": equity.to_numpy(),
            "return": returns.to_numpy(),
            "position": position.to_numpy(),
        }
    )
    return BacktestResult(
        curve=curve,
        trade_count=count_entries(curve["position"]),
        turnover=average_turnover(curve["position"]),
        max_drawdown=max_drawdown(curve["equity"]),
        period_return=period_return(curve["return"]),
    )
