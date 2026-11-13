"""Clean keeps provider rows and the six backtest columns."""

import pandas as pd
import pytest

from data.clean import CleanError, clean_symbol, to_clean_frame


def _raw():
    return pd.DataFrame(
        {
            "日期": ["2018-12-31", "2019-01-02", "2019-01-04", "2019-01-04", "2026-11-01"],
            "开盘": [1.0, 3.0, 3.2, 3.4, 4.0],
            "最高": [1.1, 3.3, 3.1, 3.5, 4.2],
            "最低": [0.9, 2.9, 3.0, 3.3, 3.9],
            "收盘": [1.0, 3.2, 3.0, 3.4, 4.1],
            "成交量": [10, 100, 90, 80, 70],
            "成交额": [1, 2, 3, 4, 5],
        }
    )


def test_clean_uses_the_backtest_columns_and_does_not_fill_days():
    cleaned = to_clean_frame(_raw(), "2019-01-01", "2026-10-31")
    assert list(cleaned.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert cleaned["date"].tolist() == ["2019-01-02", "2019-01-04"]
    assert cleaned.loc[1, "open"] == 3.4


def test_clean_drops_rows_that_cannot_be_bars():
    raw = _raw()
    raw.loc[1, "最高"] = 1.0
    cleaned = to_clean_frame(raw, "2019-01-01", "2026-10-31")
    assert cleaned["date"].tolist() == ["2019-01-04"]


def test_failed_clean_keeps_the_previous_file(tmp_path):
    previous = tmp_path / "510300.csv"
    previous.write_text("keep\n", encoding="utf-8")
    with pytest.raises(CleanError):
        clean_symbol("510300", tmp_path / "missing", tmp_path, "2019-01-01", "2026-10-31")
    assert previous.read_text(encoding="utf-8") == "keep\n"
