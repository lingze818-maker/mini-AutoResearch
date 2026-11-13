"""Universe runs read local clean bars and do not touch the network."""

from __future__ import annotations

import json
import socket

import pandas as pd
import pytest

from backtest.cli import default_universe_dir, main, run_universe
from backtest.runner import load_bars
from config_loader import ROOT, load_config
from eval.judge import judge
from eval.metrics import PeriodMetrics, SplitMetrics, evaluate
from tests.conftest import BARS_FIXTURE


def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*_args, **_kwargs):
        raise AssertionError("network access is not allowed")

    monkeypatch.setattr(socket, "create_connection", blocked)

    def connect(self, address):
        raise AssertionError(f"network access is not allowed: {address}")

    monkeypatch.setattr(socket.socket, "connect", connect)


def _metrics(summary: dict) -> SplitMetrics:
    return SplitMetrics(
        in_sample=PeriodMetrics(**summary["in_sample"]),
        out_of_sample=PeriodMetrics(**summary["out_of_sample"]),
        in_sample_end=summary["in_sample_end"],
        out_of_sample_start=summary["out_of_sample_start"],
    )


def test_default_universe_dir_follows_config_reports():
    assert default_universe_dir() == ROOT / load_config()["paths"]["reports"] / "baseline"


def test_universe_cli_loads_symbol_csvs_and_writes_verdicts(tmp_path, monkeypatch, capsys):
    _block_network(monkeypatch)
    clean = tmp_path / "clean"
    clean.mkdir()
    payload = BARS_FIXTURE.read_bytes()
    for symbol in ("510300", "510500", "518880"):
        (clean / f"{symbol}.csv").write_bytes(payload)
    monkeypatch.setattr("backtest.cli.clean_directory", lambda: clean)

    out = tmp_path / "baseline"
    code = main(["--universe", "--out", str(out)])
    printed = capsys.readouterr().out
    assert code == 0
    assert printed.count("decision:") == 3

    config = load_config()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["strategy"] == "ma_cross"
    assert summary["short_window"] == config["strategy"]["ma_cross"]["short_window"] == 20
    assert summary["long_window"] == config["strategy"]["ma_cross"]["long_window"] == 60
    assert summary["fees"] == 0.0
    assert summary["in_sample_end"] == "2023-12-31"
    assert summary["out_of_sample_start"] == "2024-01-01"
    assert [row["symbol"] for row in summary["symbols"]] == ["510300", "510500", "518880"]

    for row in summary["symbols"]:
        curve = pd.read_csv(out / row["curve"])
        per_symbol = json.loads((out / row["summary"]).read_text(encoding="utf-8"))
        assert per_symbol["decision"] == row["decision"]
        assert per_symbol["rule_id"] == row["rule_id"]
        assert per_symbol["short_window"] == 20
        assert per_symbol["long_window"] == 60
        assert per_symbol["fees"] == 0.0
        verdict = judge(_metrics(row))
        assert row["decision"] == verdict.decision
        assert row["rule_id"] == verdict.rule_id
        assert row["reason"] == verdict.reason
        direct = evaluate(curve)
        assert row["in_sample"]["period_return"] == pytest.approx(direct.in_sample.period_return)
        assert row["out_of_sample"]["max_drawdown"] == pytest.approx(direct.out_of_sample.max_drawdown)
        assert row["out_of_sample"]["turnover"] == pytest.approx(direct.out_of_sample.turnover)
        assert len(curve) == len(load_bars(clean / f"{row['symbol']}.csv"))
        assert curve["equity"].iloc[0] == pytest.approx(1.0)
        assert set(curve["position"].unique()).issubset({0.0, 1.0})


def test_universe_refuses_a_missing_clean_file_without_fetching(tmp_path, monkeypatch, capsys):
    _block_network(monkeypatch)
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "510300.csv").write_bytes(BARS_FIXTURE.read_bytes())
    monkeypatch.setattr("backtest.cli.clean_directory", lambda: clean)

    code = main(["--universe", "--out", str(tmp_path / "out")])
    captured = capsys.readouterr()
    assert code == 1
    assert "510500" in captured.err
    assert "missing" in captured.err
    assert captured.out == ""
    assert not (tmp_path / "out").exists()


def test_cli_requires_exactly_one_input_mode():
    with pytest.raises(SystemExit) as neither:
        main([])
    assert neither.value.code != 0
    with pytest.raises(SystemExit) as both:
        main(["--csv", str(BARS_FIXTURE), "--universe"])
    assert both.value.code != 0


def test_universe_does_not_accept_a_window_override():
    with pytest.raises(SystemExit) as caught:
        main(["--universe", "--short-window", "5"])
    assert caught.value.code != 0


def test_real_clean_bars_run_offline_and_match_the_committed_baseline(tmp_path, monkeypatch):
    _block_network(monkeypatch)
    out = tmp_path / "baseline"
    assert main(["--universe", "--out", str(out)]) == 0

    committed_dir = ROOT / "reports" / "baseline"
    committed = json.loads((committed_dir / "summary.json").read_text(encoding="utf-8"))
    fresh = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert [row["symbol"] for row in fresh["symbols"]] == ["510300", "510500", "518880"]
    assert fresh["short_window"] == 20
    assert fresh["long_window"] == 60
    assert fresh["fees"] == 0.0
    assert fresh["in_sample_end"] == "2023-12-31"
    assert fresh["out_of_sample_start"] == "2024-01-01"
    assert committed["disclaimer"] == fresh["disclaimer"]

    for committed_row, fresh_row in zip(committed["symbols"], fresh["symbols"], strict=True):
        symbol = fresh_row["symbol"]
        bars = load_bars(ROOT / "data" / "clean" / f"{symbol}.csv")
        assert bars["date"].iloc[0] == pd.Timestamp("2019-01-02")
        assert bars["date"].iloc[-1] == pd.Timestamp("2026-10-30")
        assert len(bars) == 1897
        assert fresh_row["source"] == f"data/clean/{symbol}.csv"
        assert fresh_row["curve_rows"] == 1897
        assert fresh_row["bar_start"] == "2019-01-02"
        assert fresh_row["bar_end"] == "2026-10-30"

        fresh_curve = pd.read_csv(out / f"{symbol}_curve.csv")
        committed_curve = pd.read_csv(committed_dir / f"{symbol}_curve.csv")
        assert fresh_curve["date"].tolist() == bars["date"].dt.strftime("%Y-%m-%d").tolist()
        assert committed_curve["date"].tolist() == fresh_curve["date"].tolist()
        assert committed_curve["equity"].tolist() == pytest.approx(fresh_curve["equity"].tolist())
        assert committed_row["decision"] == fresh_row["decision"]
        assert committed_row["rule_id"] == fresh_row["rule_id"]
        assert committed_row["in_sample"]["period_return"] == pytest.approx(fresh_row["in_sample"]["period_return"])
        assert committed_row["in_sample"]["max_drawdown"] == pytest.approx(fresh_row["in_sample"]["max_drawdown"])
        assert committed_row["in_sample"]["turnover"] == pytest.approx(fresh_row["in_sample"]["turnover"])
        assert committed_row["out_of_sample"]["period_return"] == pytest.approx(
            fresh_row["out_of_sample"]["period_return"]
        )
        assert committed_row["out_of_sample"]["max_drawdown"] == pytest.approx(
            fresh_row["out_of_sample"]["max_drawdown"]
        )
        assert committed_row["out_of_sample"]["turnover"] == pytest.approx(fresh_row["out_of_sample"]["turnover"])
        verdict = judge(_metrics(fresh_row))
        assert fresh_row["decision"] == verdict.decision
        assert fresh_row["rule_id"] == verdict.rule_id


def test_run_universe_reads_the_same_clean_layout(tmp_path):
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "510300.csv").write_bytes(BARS_FIXTURE.read_bytes())
    (clean / "510500.csv").write_bytes(BARS_FIXTURE.read_bytes())
    (clean / "518880.csv").write_bytes(BARS_FIXTURE.read_bytes())
    out = tmp_path / "out"
    summary = run_universe(clean, out)
    assert (out / "510300_curve.csv").is_file()
    assert summary["symbols"][0]["source"].endswith("510300.csv")
