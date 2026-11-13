"""Pull qfq daily bars from akshare. A failed symbol does not replace its file."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

def _project():
    try:
        from config_loader import ROOT, load_config
    except ImportError:
        from src.config_loader import ROOT, load_config
    return ROOT, load_config


def load_config():
    return _project()[1]()

CLEAN_COLUMNS = ("date", "open", "high", "low", "close", "volume")


class FetchError(RuntimeError):
    """Akshare did not return a usable table for one symbol and window."""


def symbols() -> list[str]:
    return [str(row["symbol"]) for row in load_config()["universe"]]


def date_window() -> tuple[str, str, str]:
    config = load_config()
    return config["dates"]["start"], config["dates"]["end"], config["data"]["adjust"]


def raw_dir() -> Path:
    root, _load = _project()
    return root / _load()["paths"]["raw"]


def raw_path(symbol: str, directory: Path | None = None) -> Path:
    return (directory or raw_dir()) / f"{symbol}.csv"


def _failure(symbol: str, start: str, end: str, reason: BaseException | str) -> FetchError:
    return FetchError(
        f"akshare fund_etf_hist_em failed for {symbol} {start} to {end}: {reason}"
    )


def download_bars(
    symbol: str,
    start: str,
    end: str,
    adjust: str,
    *,
    attempts: int = 6,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Call akshare.fund_etf_hist_em. Retries disconnects; does not invent rows."""
    import akshare as ak

    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            frame = ak.fund_etf_hist_em(
                symbol=symbol,
                period="daily",
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
                adjust=adjust,
            )
        except Exception as exc:  # network, rate limit, or provider error
            last = exc
        else:
            if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
                last = RuntimeError("akshare returned no rows")
            else:
                return frame
        if attempt < attempts:
            sleep(min(60.0, 5.0 * (2 ** (attempt - 1))))
    raise _failure(symbol, start, end, last if last is not None else "unknown error")


def fetch_symbol(
    symbol: str,
    start: str,
    end: str,
    adjust: str,
    directory: Path,
    *,
    attempts: int = 6,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Write data/raw/<symbol>.csv only after a successful download."""
    frame = download_bars(symbol, start, end, adjust, attempts=attempts, sleep=sleep)
    directory.mkdir(parents=True, exist_ok=True)
    target = raw_path(symbol, directory)
    temporary = target.with_suffix(".csv.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def fetch_all(
    *,
    attempts: int = 6,
    sleep: Callable[[float], None] = time.sleep,
) -> list[str]:
    start, end, adjust = date_window()
    directory = raw_dir()
    errors: list[str] = []
    for symbol in symbols():
        try:
            fetch_symbol(symbol, start, end, adjust, directory, attempts=attempts, sleep=sleep)
        except Exception as exc:
            message = str(exc)
            if "akshare" not in message or symbol not in message or start not in message or end not in message:
                message = str(_failure(symbol, start, end, exc))
            errors.append(message)
    return errors
