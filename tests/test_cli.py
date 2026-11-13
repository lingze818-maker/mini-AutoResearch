from __future__ import annotations

import json

import pandas as pd
import pytest

from backtest.cli import main
from backtest.runner import load_bars, run_backtest
from backtest.strategies.ma_cross import ma_cross
from config_loader import load_config
from eval.judge import judge
from eval.metrics import PeriodMetrics, SplitMetrics, evaluate
from tests.conftest import BARS_FIXTURE


def test_fixture_cli_runs_ma_cross_metrics_and_judge(capsys, tmp_path):
    out = tmp_path / "fixture_run"
    code = main(["--csv", str(BARS_FIXTURE), "--out", str(out)])
    printed = capsys.readouterr().out
    summary = json.loads((tmp_path / "fixture_run_summary.json").read_text(encoding="utf-8"))
    curve = pd.read_csv(tmp_path / "fixture_run_curve.csv")
    bars = load_bars(BARS_FIXTURE)

    assert code == 0
    for label in (
        "in_sample.period_return:",
        "in_sample.max_drawdown:",
        "in_sample.turnover:",
        "out_of_sample.period_return:",
        "out_of_sample.max_drawdown:",
        "out_of_sample.turnover:",
        "decision:",
        "equity_start:",
        "equity_end:",
    ):
        assert label in printed

    assert summary["in_sample_end"] == "2023-12-31"
    assert summary["out_of_sample_start"] == "2024-01-01"
    assert summary["decision"] in {"pass", "reject", "suspected_overfit"}
    metrics = SplitMetrics(
        in_sample=PeriodMetrics(**summary["in_sample"]),
        out_of_sample=PeriodMetrics(**summary["out_of_sample"]),
        in_sample_end=summary["in_sample_end"],
        out_of_sample_start=summary["out_of_sample_start"],
    )
    verdict = judge(metrics)
    assert summary["decision"] == verdict.decision
    assert summary["rule_id"] == verdict.rule_id
    assert summary["reason"] == verdict.reason

    config = load_config()
    direct = run_backtest(
        bars,
        ma_cross,
        params={
            "short_window": config["strategy"]["ma_cross"]["short_window"],
            "long_window": config["strategy"]["ma_cross"]["long_window"],
        },
    )
    direct_metrics = evaluate(direct.curve)
    assert summary["in_sample"]["period_return"] == pytest.approx(direct_metrics.in_sample.period_return)
    assert summary["out_of_sample"]["max_drawdown"] == pytest.approx(
        direct_metrics.out_of_sample.max_drawdown
    )
    assert summary["in_sample"]["trade_count"] + summary["out_of_sample"]["trade_count"] >= 1

    assert len(curve) == len(bars)
    assert list(curve.columns) == ["date", "equity", "return", "position"]
    assert curve["equity"].iloc[0] == pytest.approx(1.0)
    assert set(curve["position"].unique()).issubset({0.0, 1.0})
    assert (curve["date"] <= "2023-12-31").any()
    assert (curve["date"] > "2023-12-31").any()


def test_cli_does_not_accept_a_window_override():
    with pytest.raises(SystemExit) as caught:
        main(["--csv", str(BARS_FIXTURE), "--short-window", "5"])
    assert caught.value.code != 0


def test_backtest_and_eval_sources_do_not_fetch_or_trade_live():
    from pathlib import Path

    files = list(Path("src/backtest").rglob("*.py"))
    files += list(Path("src/eval").rglob("*.py"))
    files.append(Path("src/config_loader.py"))
    text = "\n".join(path.read_text(encoding="utf-8") for path in files)
    for banned in ("akshare", "backtrader", "urllib", "yfinance", "ccxt", "deepseek", "openai"):
        assert banned not in text.lower()
    assert "import requests" not in text
    assert "api_key" not in text.lower()
