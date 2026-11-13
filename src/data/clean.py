"""Turn akshare raw tables into date, open, high, low, close, volume."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .fetch import CLEAN_COLUMNS, _project, raw_path, symbols


def load_config():
    return _project()[1]()


def _root() -> Path:
    return _project()[0]

COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "date": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}


class CleanError(RuntimeError):
    """Raw bars could not be turned into a clean table for one symbol."""


def clean_dir() -> Path:
    return _root() / load_config()["paths"]["clean"]


def clean_path(symbol: str, directory: Path | None = None) -> Path:
    return (directory or clean_dir()) / f"{symbol}.csv"


def to_clean_frame(raw: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Keep only rows the provider returned. Do not fill missing sessions."""
    renamed = raw.rename(columns={column: COLUMN_MAP[column] for column in raw.columns if column in COLUMN_MAP})
    missing = [column for column in CLEAN_COLUMNS if column not in renamed.columns]
    if missing:
        raise CleanError(f"raw bars missing columns: {missing}")
    frame = renamed.loc[:, list(CLEAN_COLUMNS)].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if getattr(frame["date"].dt, "tz", None) is not None:
        frame["date"] = frame["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    frame["date"] = frame["date"].dt.normalize()
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(CLEAN_COLUMNS))
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    frame = frame[(frame["date"] >= start_ts) & (frame["date"] <= end_ts)]
    frame = frame.drop_duplicates(subset=["date"], keep="last")
    prices = frame[["open", "high", "low", "close"]]
    usable = (
        (prices > 0).all(axis=1)
        & (frame["volume"] >= 0)
        & (frame["high"] + 1e-9 >= prices[["open", "close"]].max(axis=1))
        & (frame["low"] - 1e-9 <= prices[["open", "close"]].min(axis=1))
    )
    frame = frame.loc[usable].sort_values("date")
    if frame.empty:
        raise CleanError(f"no usable rows from {start} to {end}")
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return frame.reset_index(drop=True)


def clean_symbol(symbol: str, raw_directory: Path, clean_directory: Path, start: str, end: str) -> None:
    source = raw_path(symbol, raw_directory)
    if not source.is_file():
        raise CleanError(f"raw file missing for {symbol}: {source}")
    raw = pd.read_csv(source)
    cleaned = to_clean_frame(raw, start, end)
    clean_directory.mkdir(parents=True, exist_ok=True)
    target = clean_path(symbol, clean_directory)
    temporary = target.with_suffix(".csv.tmp")
    try:
        cleaned.to_csv(temporary, index=False)
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def clean_all() -> list[str]:
    config = load_config()
    start = config["dates"]["start"]
    end = config["dates"]["end"]
    raw_directory = _root() / config["paths"]["raw"]
    clean_directory = clean_dir()
    errors: list[str] = []
    for symbol in symbols():
        try:
            clean_symbol(symbol, raw_directory, clean_directory, start, end)
        except Exception as exc:
            errors.append(f"clean failed for {symbol} {start} to {end}: {exc}")
    return errors
