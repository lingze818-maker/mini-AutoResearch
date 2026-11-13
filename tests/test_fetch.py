"""Fetch writes a file only after akshare returns rows."""

import sys

import pandas as pd
import pytest

from data.fetch import FetchError, download_bars, fetch_symbol, symbols


def _frame():
    return pd.DataFrame(
        {
            "日期": ["2019-01-02", "2019-01-03"],
            "开盘": [3.0, 3.1],
            "收盘": [3.2, 3.0],
            "最高": [3.3, 3.2],
            "最低": [2.9, 2.8],
            "成交量": [100.0, 110.0],
        }
    )


def test_symbols_are_the_configured_etfs():
    assert symbols() == ["510300", "510500", "518880"]
    assert "159915" not in symbols()


def test_download_asks_akshare_for_qfq_daily(monkeypatch):
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return _frame()

    monkeypatch.setitem(sys.modules, "akshare", type("Ak", (), {"fund_etf_hist_em": staticmethod(fake)})())
    frame = download_bars("510300", "2019-01-01", "2026-10-31", "qfq", attempts=1)
    assert seen == {
        "symbol": "510300",
        "period": "daily",
        "start_date": "20190101",
        "end_date": "20261031",
        "adjust": "qfq",
    }
    assert len(frame) == 2


def test_failed_symbol_keeps_the_previous_file(tmp_path, monkeypatch):
    previous = tmp_path / "518880.csv"
    previous.write_text("keep\n", encoding="utf-8")

    def fake(**kwargs):
        raise ConnectionError("East Money disconnected")

    monkeypatch.setitem(sys.modules, "akshare", type("Ak", (), {"fund_etf_hist_em": staticmethod(fake)})())
    with pytest.raises(FetchError) as caught:
        fetch_symbol("518880", "2019-01-01", "2026-10-31", "qfq", tmp_path, attempts=1)
    message = str(caught.value)
    assert "akshare" in message
    assert "518880" in message
    assert "2019-01-01" in message
    assert "2026-10-31" in message
    assert previous.read_text(encoding="utf-8") == "keep\n"
    assert not (tmp_path / "518880.csv.tmp").exists()


def test_empty_akshare_result_is_not_written(tmp_path, monkeypatch):
    def fake(**kwargs):
        return _frame().iloc[0:0]

    monkeypatch.setitem(sys.modules, "akshare", type("Ak", (), {"fund_etf_hist_em": staticmethod(fake)})())
    with pytest.raises(FetchError) as caught:
        fetch_symbol("510500", "2019-01-01", "2026-10-31", "qfq", tmp_path, attempts=1)
    assert "akshare" in str(caught.value)
    assert "510500" in str(caught.value)
    assert not (tmp_path / "510500.csv").exists()
