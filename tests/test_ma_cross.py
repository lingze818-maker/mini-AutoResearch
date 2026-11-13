from __future__ import annotations

import pandas as pd

from backtest.runner import load_bars, run_backtest
from backtest.strategies.ma_cross import ma_cross, ma_cross_from_config
from config_loader import load_config
from tests.conftest import BARS_FIXTURE, bars_from_closes


def test_cross_uses_only_closes_up_to_the_signal_bar():
    closes = [5, 4, 3, 2, 1, 2, 4, 8, 10, 10]
    signals = ma_cross(bars_from_closes(closes), short_window=2, long_window=4)

    assert signals["entries"].tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert signals["exits"].tolist() == [False] * 10


def test_signal_does_not_change_when_a_future_close_changes():
    bars = load_bars(BARS_FIXTURE)
    original = ma_cross_from_config()(bars)
    tweaked = bars.copy()
    tweaked.loc[tweaked.index[-1], "close"] = tweaked.loc[tweaked.index[-1], "close"] * 3
    tweaked.loc[tweaked.index[-1], "high"] = tweaked.loc[tweaked.index[-1], ["open", "close", "high"]].max()
    changed = ma_cross_from_config()(tweaked)

    pd.testing.assert_series_equal(original["entries"].iloc[:-1], changed["entries"].iloc[:-1])
    pd.testing.assert_series_equal(original["exits"].iloc[:-1], changed["exits"].iloc[:-1])


def test_config_windows_are_the_only_pipe_default():
    config = load_config()
    short = config["strategy"]["ma_cross"]["short_window"]
    long = config["strategy"]["ma_cross"]["long_window"]
    assert (short, long) == (20, 60)

    bars = load_bars(BARS_FIXTURE)
    from_config = ma_cross_from_config()(bars)
    from_params = ma_cross(bars, short_window=short, long_window=long)
    other = ma_cross(bars, short_window=5, long_window=10)

    pd.testing.assert_frame_equal(from_config, from_params)
    assert from_config["entries"].any()
    assert not from_config["entries"].equals(other["entries"])


def test_ma_cross_goes_through_the_runner_on_the_next_bar():
    result = run_backtest(
        bars_from_closes([5, 4, 3, 2, 1, 2, 4, 8, 10, 10]),
        ma_cross,
        params={"short_window": 2, "long_window": 4},
    )
    assert result.curve["position"].tolist()[6] == 0
    assert result.curve["position"].tolist()[7] == 1
    assert result.trade_count == 1


def test_inverted_windows_are_rejected():
    bars = bars_from_closes([1, 2, 3, 4, 5, 6])
    import pytest

    with pytest.raises(ValueError):
        ma_cross(bars, short_window=60, long_window=20)
    with pytest.raises(ValueError):
        ma_cross(bars, short_window=0, long_window=10)
