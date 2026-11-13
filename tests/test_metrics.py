from __future__ import annotations

import inspect

import pandas as pd
import pytest

from eval.metrics import evaluate


def _curve(dates, equity, returns, position) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates,
            "equity": equity,
            "return": returns,
            "position": position,
        }
    )


def test_evaluate_reads_only_the_fixed_config_split():
    assert list(inspect.signature(evaluate).parameters) == ["curve"]


def test_in_sample_end_is_inclusive_and_out_of_sample_starts_the_next_day():
    metrics = evaluate(
        _curve(
            ["2023-12-31", "2024-01-01"],
            [1.0, 0.5],
            [0.0, -0.5],
            [1.0, 1.0],
        )
    )

    assert metrics.in_sample_end == "2023-12-31"
    assert metrics.out_of_sample_start == "2024-01-01"
    assert metrics.in_sample.period_return == pytest.approx(0.0)
    assert metrics.out_of_sample.period_return == pytest.approx(-0.5)
    assert metrics.in_sample.max_drawdown == pytest.approx(0.0)
    assert metrics.out_of_sample.max_drawdown == pytest.approx(0.5)
    assert metrics.in_sample.trade_count == 1
    assert metrics.out_of_sample.trade_count == 0
    assert metrics.in_sample.turnover == pytest.approx(1.0)
    assert metrics.out_of_sample.turnover == pytest.approx(0.0)


def test_out_of_sample_drawdown_keeps_the_boundary_and_drops_an_earlier_peak():
    crash_inside_sample = evaluate(
        _curve(
            ["2023-12-30", "2023-12-31", "2024-01-01", "2024-01-02"],
            [2.0, 1.0, 1.0, 1.0],
            [0.0, -0.5, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        )
    )
    crash_across_boundary = evaluate(
        _curve(
            ["2023-12-30", "2023-12-31", "2024-01-01", "2024-01-02"],
            [1.0, 1.0, 0.7, 0.7],
            [0.0, 0.0, -0.3, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        )
    )

    assert crash_inside_sample.in_sample.max_drawdown == pytest.approx(0.5)
    assert crash_inside_sample.out_of_sample.max_drawdown == pytest.approx(0.0)
    assert crash_across_boundary.out_of_sample.max_drawdown == pytest.approx(0.3)
    assert crash_across_boundary.in_sample.trade_count == 1
    assert crash_across_boundary.out_of_sample.trade_count == 0
    assert crash_across_boundary.out_of_sample.turnover == pytest.approx(0.5)


def test_a_curve_that_misses_one_side_of_the_split_fails():
    with pytest.raises(ValueError, match="in-sample"):
        evaluate(
            _curve(
                ["2024-01-02", "2024-01-03"],
                [1.0, 1.1],
                [0.0, 0.1],
                [1.0, 1.0],
            )
        )
    with pytest.raises(ValueError, match="out-of-sample"):
        evaluate(
            _curve(
                ["2023-12-28", "2023-12-29"],
                [1.0, 1.1],
                [0.0, 0.1],
                [1.0, 1.0],
            )
        )
