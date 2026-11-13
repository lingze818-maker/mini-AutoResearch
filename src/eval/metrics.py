"""Period metrics and the fixed in-sample / out-of-sample split."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from config_loader import load_config

CURVE_COLUMNS = ("date", "equity", "return", "position")


@dataclass(frozen=True, slots=True)
class PeriodMetrics:
    period_return: float
    max_drawdown: float
    turnover: float
    trade_count: int

    def __post_init__(self) -> None:
        for name in ("period_return", "max_drawdown", "turnover"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number")
        if self.max_drawdown < 0 or self.turnover < 0:
            raise ValueError("max_drawdown and turnover must be non-negative")
        if isinstance(self.trade_count, bool) or int(self.trade_count) != self.trade_count or self.trade_count < 0:
            raise ValueError("trade_count must be a non-negative integer")
        object.__setattr__(self, "period_return", float(self.period_return))
        object.__setattr__(self, "max_drawdown", float(self.max_drawdown))
        object.__setattr__(self, "turnover", float(self.turnover))
        object.__setattr__(self, "trade_count", int(self.trade_count))

    def as_dict(self) -> dict[str, float | int]:
        return {
            "period_return": self.period_return,
            "max_drawdown": self.max_drawdown,
            "turnover": self.turnover,
            "trade_count": self.trade_count,
        }


@dataclass(frozen=True, slots=True)
class SplitMetrics:
    in_sample: PeriodMetrics
    out_of_sample: PeriodMetrics
    in_sample_end: str
    out_of_sample_start: str


def period_return(returns: pd.Series) -> float:
    values = pd.Series(returns, dtype="float64").reset_index(drop=True)
    if values.empty or not values.map(math.isfinite).all():
        raise ValueError("returns must be finite")
    return float((1.0 + values).prod() - 1.0)


def max_drawdown(equity: pd.Series, anchor: float | None = None) -> float:
    """Positive peak-to-trough fraction. `anchor` is the equity just before the slice."""
    values = pd.Series(equity, dtype="float64").reset_index(drop=True)
    if anchor is not None:
        values = pd.concat([pd.Series([float(anchor)], dtype="float64"), values], ignore_index=True)
    if values.empty or (values <= 0).any() or not values.map(math.isfinite).all():
        raise ValueError("equity must be positive and finite")
    drawdown = 1.0 - values / values.cummax()
    if anchor is not None:
        drawdown = drawdown.iloc[1:]
    if drawdown.empty:
        return 0.0
    return float(drawdown.max())


def average_turnover(position: pd.Series, mask: pd.Series | None = None) -> float:
    """Mean absolute daily change in the 0/1 position. The first bar changes from flat."""
    held = pd.Series(position, dtype="float64")
    change = (held - held.shift(1).fillna(0.0)).abs()
    if mask is not None:
        change = change.loc[mask]
    if change.empty:
        raise ValueError("empty period")
    return float(change.sum() / len(change))


def count_entries(position: pd.Series, mask: pd.Series | None = None) -> int:
    held = pd.Series(position, dtype="float64")
    previous = held.shift(1).fillna(0.0)
    entries = (previous <= 0.0) & (held > 0.0)
    if mask is not None:
        entries = entries.loc[mask]
    return int(entries.sum())


def _prepare_curve(curve: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in CURVE_COLUMNS if column not in curve.columns]
    if missing:
        raise ValueError(f"curve missing columns: {missing}")
    frame = curve.loc[:, list(CURVE_COLUMNS)].copy().reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if getattr(frame["date"].dt, "tz", None) is not None:
        frame["date"] = frame["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    frame["date"] = frame["date"].dt.normalize()
    if frame["date"].isna().any():
        raise ValueError("curve dates must be parseable")
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise ValueError("curve dates must be strictly increasing and unique")
    for column in ("equity", "return", "position"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["equity", "return", "position"]].isna().any().any():
        raise ValueError("curve values must be numeric")
    return frame


def evaluate(curve: pd.DataFrame) -> SplitMetrics:
    """Split `curve` with config.yaml's in_sample_end. That date is inclusive."""
    frame = _prepare_curve(curve)
    in_sample_end = pd.Timestamp(load_config()["dates"]["in_sample_end"]).normalize()
    out_of_sample_start = in_sample_end + pd.Timedelta(days=1)
    in_sample = frame["date"] <= in_sample_end
    out_of_sample = frame["date"] > in_sample_end
    if not bool(in_sample.any()):
        raise ValueError("in-sample period has no bars on or before in_sample_end")
    if not bool(out_of_sample.any()):
        raise ValueError("out-of-sample period has no bars after in_sample_end")

    def summarize(mask: pd.Series, anchor: float | None) -> PeriodMetrics:
        return PeriodMetrics(
            period_return=period_return(frame.loc[mask, "return"]),
            max_drawdown=max_drawdown(frame.loc[mask, "equity"], anchor=anchor),
            turnover=average_turnover(frame["position"], mask),
            trade_count=count_entries(frame["position"], mask),
        )

    in_sample_metrics = summarize(in_sample, None)
    anchor = float(frame.loc[in_sample, "equity"].iloc[-1])
    return SplitMetrics(
        in_sample=in_sample_metrics,
        out_of_sample=summarize(out_of_sample, anchor),
        in_sample_end=in_sample_end.strftime("%Y-%m-%d"),
        out_of_sample_start=out_of_sample_start.strftime("%Y-%m-%d"),
    )
