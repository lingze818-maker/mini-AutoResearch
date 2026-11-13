from __future__ import annotations

import pandas as pd
import pytest

from backtest.runner import load_bars, run_backtest, validate_bars
from tests.conftest import BARS_FIXTURE, bars_from_closes


def _signals(entries_at: list[int], exits_at: list[int] | None = None):
    exits_at = exits_at or []

    def strategy(data: pd.DataFrame) -> pd.DataFrame:
        entries = pd.Series(False, index=data.index)
        exits = pd.Series(False, index=data.index)
        entries.iloc[entries_at] = True
        exits.iloc[exits_at] = True
        return pd.DataFrame({"entries": entries, "exits": exits})

    return strategy


def test_next_bar_close_fill_follows_the_price_path():
    closes = [100, 100, 100, 100, 100, 110, 90, 90]
    result = run_backtest(bars_from_closes(closes), _signals([1]))

    assert result.curve["position"].tolist() == [0, 0, 1, 1, 1, 1, 1, 1]
    assert result.curve["equity"].tolist() == pytest.approx([1, 1, 1, 1, 1, 1.1, 0.9, 0.9])
    assert result.curve["return"].tolist() == pytest.approx(
        result.curve["equity"].pct_change().fillna(0).tolist()
    )
    assert result.trade_count == 1
    assert result.turnover == pytest.approx(1 / 8)
    assert result.max_drawdown == pytest.approx(2 / 11)
    assert result.period_return == pytest.approx(0.9 / 1.0 - 1)


def test_exit_also_waits_for_the_next_bar():
    result = run_backtest(bars_from_closes([100, 100, 100, 150, 150, 150]), _signals([1], [3]))

    assert result.curve["position"].tolist() == [0, 0, 1, 1, 0, 0]
    assert result.curve["equity"].tolist() == pytest.approx([1, 1, 1, 1.5, 1.5, 1.5])
    assert result.trade_count == 1
    assert result.max_drawdown == pytest.approx(0)


def test_a_signal_on_the_last_bar_is_not_filled():
    result = run_backtest(bars_from_closes([100, 101, 102, 103]), _signals([3]))

    assert result.curve["position"].tolist() == [0, 0, 0, 0]
    assert result.trade_count == 0
    assert result.curve["equity"].tolist() == pytest.approx([1, 1, 1, 1])


def test_flat_strategy_stays_in_cash():
    def flat(data: pd.DataFrame) -> pd.DataFrame:
        false = pd.Series(False, index=data.index)
        return pd.DataFrame({"entries": false, "exits": false})

    result = run_backtest(bars_from_closes([100, 120, 80, 90]), flat)
    assert result.trade_count == 0
    assert result.turnover == pytest.approx(0)
    assert result.max_drawdown == pytest.approx(0)
    assert result.curve["equity"].tolist() == pytest.approx([1, 1, 1, 1])


def test_runner_simulates_with_vectorbt(monkeypatch):
    import vectorbt as vbt

    seen: dict[str, object] = {}
    original = vbt.Portfolio.from_signals

    def wrapped(close, entries, exits, *args, **kwargs):
        seen["entries"] = list(entries)
        seen["kwargs"] = kwargs
        return original(close, entries, exits, *args, **kwargs)

    monkeypatch.setattr(vbt.Portfolio, "from_signals", wrapped)
    run_backtest(bars_from_closes([100, 100, 100, 100]), _signals([1]))

    assert seen["entries"] == [False, False, True, False]
    assert seen["kwargs"]["direction"] == "longonly"
    assert seen["kwargs"]["fees"] == 0.0
    assert seen["kwargs"]["init_cash"] == 1.0


def test_params_call_matches_a_closed_strategy():
    def parameterized(data: pd.DataFrame, enter_at: int) -> pd.DataFrame:
        return _signals([enter_at])(data)

    bars = bars_from_closes([100, 100, 110, 90])
    via_params = run_backtest(bars, parameterized, params={"enter_at": 1})
    via_function = run_backtest(bars, _signals([1]))
    pd.testing.assert_frame_equal(via_params.curve, via_function.curve)


def test_fixture_bars_load_as_one_strict_daily_series():
    bars = load_bars(BARS_FIXTURE)
    assert list(bars.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert bars["date"].is_monotonic_increasing
    assert not bars["date"].duplicated().any()
    assert bars["date"].dt.day_of_week.max() <= 4
    assert (bars["date"] <= "2023-12-31").any()
    assert (bars["date"] > "2023-12-31").any()
    assert (bars["high"] + 1e-9 >= bars[["open", "close"]].max(axis=1)).all()
    assert (bars["low"] - 1e-9 <= bars[["open", "close"]].min(axis=1)).all()
    assert (bars["volume"] >= 0).all()
    assert (bars["close"] > 0).all()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda frame: frame.drop(columns=["volume"]),
        lambda frame: frame.assign(high=frame["close"] - 1),
        lambda frame: frame.assign(low=frame["close"] + 1),
        lambda frame: frame.assign(close=0),
        lambda frame: frame.assign(volume=-1),
        lambda frame: frame.assign(close=float("nan")),
        lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
    ],
)
def test_invalid_bars_are_rejected(mutate):
    bars = bars_from_closes([100, 101, 102, 103])
    with pytest.raises(ValueError):
        validate_bars(mutate(bars))


def test_unsorted_dates_are_rejected():
    bars = bars_from_closes([100, 101, 102])
    bars = bars.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError):
        validate_bars(bars)


def test_empty_bars_are_rejected():
    with pytest.raises(ValueError):
        validate_bars(bars_from_closes([100]).iloc[0:0])


def test_strategy_must_return_aligned_boolean_signals():
    bars = bars_from_closes([100, 101, 102])
    with pytest.raises(ValueError):
        run_backtest(bars, lambda data: pd.DataFrame({"entries": [True, False]}))
    with pytest.raises(ValueError):
        run_backtest(bars, lambda data: pd.DataFrame({"entries": [True, False, False]}))
    with pytest.raises(ValueError):
        run_backtest(bars, "ma_cross")
