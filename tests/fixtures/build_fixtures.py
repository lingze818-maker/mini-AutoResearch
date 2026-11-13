"""Write synthetic fixtures. No market data and no network."""

from __future__ import annotations

import csv
import math
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
BARS_PATH = HERE / "synthetic_daily_bars.csv"
PASS_PATH = HERE / "curve_pass.csv"
REJECT_PATH = HERE / "curve_reject_drawdown.csv"
OVERFIT_PATH = HERE / "curve_suspected_overfit.csv"

BARS_START = date(2022, 1, 4)
BARS_END = date(2024, 6, 28)


def business_days(start: date, end: date) -> list[date]:
    days: list[date] = []
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def calendar_days(start: date, end: date) -> list[date]:
    days: list[date] = []
    cursor = start
    while cursor <= end:
        days.append(cursor)
        cursor += timedelta(days=1)
    return days


def linspace(start: float, end: float, count: int) -> list[float]:
    if count < 2:
        raise ValueError("count must be at least 2")
    step = (end - start) / (count - 1)
    return [start + step * i for i in range(count)]


def synthetic_closes(count: int) -> list[float]:
    closes = []
    for t in range(count):
        closes.append(3.0 + 0.002 * t + 0.5 * math.sin(2.0 * math.pi * t / 80.0))
    return closes


def sma_cross_count(closes: list[float], short: int, long: int) -> int:
    def sma(window: int) -> list[float | None]:
        out: list[float | None] = [None] * len(closes)
        total = 0.0
        for i, value in enumerate(closes):
            total += value
            if i >= window:
                total -= closes[i - window]
            if i >= window - 1:
                out[i] = total / window
        return out

    fast = sma(short)
    slow = sma(long)
    crosses = 0
    for i in range(1, len(closes)):
        if None in (fast[i], slow[i], fast[i - 1], slow[i - 1]):
            continue
        if fast[i] > slow[i] and fast[i - 1] <= slow[i - 1]:
            crosses += 1
    return crosses


def write_bars(path: Path) -> None:
    days = business_days(BARS_START, BARS_END)
    closes = synthetic_closes(len(days))
    if sma_cross_count(closes, 20, 60) < 1:
        raise RuntimeError("synthetic series never crosses 20/60; adjust the wave")
    rows = []
    previous = closes[0]
    for i, (day, close) in enumerate(zip(days, closes)):
        open_ = closes[0] if i == 0 else previous
        high = max(open_, close) * 1.001
        low = min(open_, close) * 0.999
        if not (low <= min(open_, close) <= max(open_, close) <= high):
            raise RuntimeError(f"invalid bar on {day.isoformat()}")
        rows.append(
            {
                "date": day.isoformat(),
                "open": f"{open_:.8f}",
                "high": f"{high:.8f}",
                "low": f"{low:.8f}",
                "close": f"{close:.8f}",
                "volume": "1000000",
            }
        )
        previous = close
    _write_csv(path, ["date", "open", "high", "low", "close", "volume"], rows)


def write_curve(path: Path, equity: list[float], days: list[date]) -> None:
    if len(equity) != len(days):
        raise ValueError("equity and days differ in length")
    returns = [0.0]
    for i in range(1, len(equity)):
        returns.append(equity[i] / equity[i - 1] - 1.0)
    rows = []
    for day, level, ret in zip(days, equity, returns):
        rows.append(
            {
                "date": day.isoformat(),
                "equity": repr(float(level)),
                "return": repr(float(ret)),
                "position": "1.0",
            }
        )
    _write_csv(path, ["date", "equity", "return", "position"], rows)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    write_bars(BARS_PATH)
    is_days = calendar_days(date(2023, 11, 1), date(2023, 12, 31))
    oos_days = calendar_days(date(2024, 1, 1), date(2024, 2, 15))
    write_curve(PASS_PATH, linspace(1.0, 1.05, len(is_days)) + linspace(1.05, 1.08, len(oos_days)), is_days + oos_days)
    write_curve(REJECT_PATH, [1.0] * len(is_days) + linspace(1.0, 0.6, len(oos_days)), is_days + oos_days)
    write_curve(OVERFIT_PATH, linspace(1.0, 1.10, len(is_days)) + linspace(1.10, 1.00, len(oos_days)), is_days + oos_days)
    print(f"wrote {BARS_PATH.name} and three judge curves")


if __name__ == "__main__":
    main()
