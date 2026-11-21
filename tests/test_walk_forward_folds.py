"""Pre-registered walk-forward folds. Dates are literals here and only in config.yaml."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from agent.limits import OFFLINE_BANNER
from agent.loop import run_bounded_experiment
from agent.offline import Candidate
from agent.tools import BacktestRecord, summarize_bars
from config_loader import load_config
from eval.judge import judge
from eval.metrics import evaluate
from eval.splits import active_split, registered_splits, use_split
from tests.conftest import ROOT

# Frozen windows. The runtime must read these from config.yaml, not from this table.
FROZEN = {
    "A": {
        "in_sample_start": "2019-01-01",
        "in_sample_end": "2020-12-31",
        "out_of_sample_start": "2021-01-01",
        "out_of_sample_end": "2021-12-31",
        "inside_oos": "2021-06-01",
        "day_after_oos": "2022-01-01",
    },
    "B": {
        "in_sample_start": "2019-01-01",
        "in_sample_end": "2021-12-31",
        "out_of_sample_start": "2022-01-01",
        "out_of_sample_end": "2022-12-31",
        "inside_oos": "2022-06-01",
        "day_after_oos": "2023-01-01",
    },
    "C": {
        "in_sample_start": "2019-01-01",
        "in_sample_end": "2022-12-31",
        "out_of_sample_start": "2023-01-01",
        "out_of_sample_end": "2023-12-31",
        "inside_oos": "2023-06-01",
        "day_after_oos": "2024-01-01",
    },
    "default": {
        "in_sample_start": "2019-01-01",
        "in_sample_end": "2023-12-31",
        "out_of_sample_start": "2024-01-01",
        "out_of_sample_end": "2026-10-31",
        "inside_oos": "2024-06-03",
        "day_after_oos": "2026-11-01",
    },
}

_BEFORE = "2018-12-31"
_SHARED_IN_SAMPLE = "2020-06-01"
_SOURCE = "def generate(bars):\n    return {'entry': False, 'exit': False}\n"
_DATE_LITERALS = (
    "2019-01-01",
    "2020-12-31",
    "2021-01-01",
    "2021-12-31",
    "2022-01-01",
    "2022-12-31",
    "2023-01-01",
    "2023-12-31",
    "2024-01-01",
    "2026-10-31",
)


def _curve(rows: list[tuple[str, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [row[0] for row in rows],
            "equity": [row[1] for row in rows],
            "return": [row[2] for row in rows],
            "position": [row[3] for row in rows],
        }
    )


def _bars(dates: list[str], closes: list[float]) -> pd.DataFrame:
    close = pd.Series(closes, dtype="float64")
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "open": close.to_numpy(),
            "high": (close * 1.001).to_numpy(),
            "low": (close * 0.999).to_numpy(),
            "close": close.to_numpy(),
            "volume": 1_000_000,
        }
    )


def _window_curve(spec: dict[str, str]) -> pd.DataFrame:
    return _curve(
        [
            (_BEFORE, 1.0, 0.40, 0.0),
            (spec["in_sample_end"], 1.4, 0.0, 0.0),
            (spec["out_of_sample_start"], 1.54, 0.10, 0.0),
            (spec["out_of_sample_end"], 1.848, 0.20, 0.0),
            (spec["day_after_oos"], 2.772, 0.50, 0.0),
        ]
    )


class _RecordingProvider:
    banner = OFFLINE_BANNER
    name = "scripted"
    calls = 0

    def __init__(self) -> None:
        self.seen: list[dict] = []
        self.backtests_at_propose = -1
        self.backtests = 0

    def propose(self, summaries):
        self.calls += 1
        self.backtests_at_propose = self.backtests
        self.seen.append({key: dict(value) for key, value in summaries.items()})
        assert all("out_of_sample" not in item for item in summaries.values())
        return [
            Candidate("flat_a", "flat-a", "in-sample only", _SOURCE),
            Candidate("flat_b", "flat-b", "in-sample only", _SOURCE),
        ]


def test_registered_splits_are_exactly_the_three_folds_and_the_default():
    windows = registered_splits()
    assert list(windows) == ["A", "B", "C", "default"]
    dates = load_config()["dates"]
    folds = dates["folds"]
    assert set(folds) == {"A", "B", "C"}
    for name, spec in FROZEN.items():
        window = windows[name]
        assert window.name == name
        assert window.in_sample_start == spec["in_sample_start"]
        assert window.in_sample_end == spec["in_sample_end"]
        assert window.out_of_sample_start == spec["out_of_sample_start"]
        assert window.out_of_sample_end == spec["out_of_sample_end"]
        if name == "default":
            assert window.in_sample_start == dates["start"]
            assert window.in_sample_end == dates["in_sample_end"]
            assert window.out_of_sample_end == dates["end"]
            continue
        assert dict(folds[name]) == {
            "in_sample_start": spec["in_sample_start"],
            "in_sample_end": spec["in_sample_end"],
            "out_of_sample_start": spec["out_of_sample_start"],
            "out_of_sample_end": spec["out_of_sample_end"],
        }
    with pytest.raises(TypeError):
        dates["folds"]["A"]["in_sample_end"] = "1999-01-01"


def test_fold_dates_are_not_hardcoded_beside_the_config():
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for stamp in _DATE_LITERALS:
            assert stamp not in text, f"{path.name} hardcodes {stamp}"


def test_selecting_default_matches_the_unselected_split():
    assert active_split().name == "default"
    curve = _curve(
        [
            ("2023-12-31", 1.0, 0.0, 1.0),
            ("2024-01-01", 0.5, -0.5, 1.0),
        ]
    )
    bare = evaluate(curve)
    with use_split("default") as window:
        selected = evaluate(curve)
        assert window.name == "default"
        assert active_split() == window
    assert active_split().name == "default"
    assert selected.in_sample_end == bare.in_sample_end == "2023-12-31"
    assert selected.out_of_sample_start == bare.out_of_sample_start == "2024-01-01"
    assert selected.out_of_sample_end == "2026-10-31"
    assert selected.in_sample.period_return == pytest.approx(bare.in_sample.period_return)
    assert selected.out_of_sample.period_return == pytest.approx(bare.out_of_sample.period_return)
    assert selected.out_of_sample.max_drawdown == pytest.approx(bare.out_of_sample.max_drawdown)
    assert selected.in_sample.turnover == pytest.approx(bare.in_sample.turnover)
    assert selected.out_of_sample.turnover == pytest.approx(bare.out_of_sample.turnover)
    assert selected.in_sample.period_return == pytest.approx(0.0)
    assert selected.out_of_sample.period_return == pytest.approx(-0.5)
    assert selected.out_of_sample.max_drawdown == pytest.approx(0.5)
    assert selected.in_sample.trade_count == 1
    assert selected.out_of_sample.trade_count == 0
    assert selected.in_sample.turnover == pytest.approx(1.0)
    assert selected.out_of_sample.turnover == pytest.approx(0.0)


def test_unknown_split_is_rejected_and_does_not_stick():
    with pytest.raises(ValueError, match="A"):
        with use_split("D"):
            pass
    assert active_split().name == "default"
    with use_split("B"):
        assert active_split().name == "B"
    assert active_split().name == "default"


@pytest.mark.parametrize("name", ["A", "B", "C", "default"])
def test_selected_window_keeps_its_ends_and_drops_outside_bars(name):
    spec = FROZEN[name]
    with use_split(name):
        metrics = evaluate(_window_curve(spec))
    assert metrics.in_sample_end == spec["in_sample_end"]
    assert metrics.out_of_sample_start == spec["out_of_sample_start"]
    assert metrics.out_of_sample_end == spec["out_of_sample_end"]
    assert metrics.in_sample.period_return == pytest.approx(0.0)
    assert metrics.out_of_sample.period_return == pytest.approx(1.10 * 1.20 - 1.0)


def test_folds_do_not_use_the_default_holdout_as_out_of_sample():
    for name in ("A", "B", "C"):
        spec = FROZEN[name]
        with use_split(name):
            with pytest.raises(ValueError, match="out-of-sample"):
                evaluate(
                    _curve(
                        [
                            (spec["in_sample_end"], 1.0, 0.0, 0.0),
                            ("2024-01-01", 0.4, -0.6, 0.0),
                            ("2026-10-31", 0.2, -0.5, 0.0),
                        ]
                    )
                )


@pytest.mark.parametrize("name", ["A", "B", "C", "default"])
def test_summary_is_clipped_to_the_selected_in_sample_end(name):
    spec = FROZEN[name]
    bars = _bars(
        [_BEFORE, _SHARED_IN_SAMPLE, spec["in_sample_end"], spec["out_of_sample_start"], spec["day_after_oos"]],
        [1.0, 100.0, 110.0, 50.0, 7.0],
    )
    with use_split(name):
        summary = summarize_bars("510300", bars)
        tweaked = bars.copy()
        outside = tweaked["date"] > pd.Timestamp(spec["in_sample_end"])
        tweaked.loc[outside, "close"] = tweaked.loc[outside, "close"] * 5
        tweaked.loc[outside, "open"] = tweaked.loc[outside, "close"]
        tweaked["high"] = tweaked[["open", "close"]].max(axis=1) * 1.001
        tweaked["low"] = tweaked[["open", "close"]].min(axis=1) * 0.999
        assert summarize_bars("510300", tweaked) == summary
    assert summary["in_sample_end"] == spec["in_sample_end"]
    assert summary["start"] == _SHARED_IN_SAMPLE
    assert summary["end"] == spec["in_sample_end"]
    assert summary["bar_count"] == 2
    assert summary["close_start"] == pytest.approx(100.0)
    assert summary["close_end"] == pytest.approx(110.0)
    assert summary["period_return"] == pytest.approx(0.1)
    assert "out_of_sample" not in summary
    assert spec["out_of_sample_start"] not in json.dumps(summary)


@pytest.mark.parametrize("name", ["A", "B", "C", "default"])
def test_judge_uses_the_same_rules_on_the_selected_window(name):
    spec = FROZEN[name]
    inside = spec["inside_oos"]
    after = spec["day_after_oos"]
    in_sample_end = spec["in_sample_end"]

    def verdict(rows):
        with use_split(name):
            metrics = evaluate(_curve(rows))
        return judge(metrics), metrics

    crashed_outside, outside_metrics = verdict(
        [
            (in_sample_end, 1.0, 0.02, 0.0),
            (inside, 1.02, 0.02, 0.0),
            (after, 0.40, -0.60, 1.0),
        ]
    )
    assert outside_metrics.out_of_sample.max_drawdown == pytest.approx(0.0)
    assert outside_metrics.out_of_sample.turnover == pytest.approx(0.0)
    assert crashed_outside.decision == "pass"
    assert crashed_outside.rule_id == "aligned_pass"

    equal_drawdown, equal_metrics = verdict(
        [
            (in_sample_end, 1.0, 0.02, 0.0),
            (inside, 0.75, 0.01, 0.0),
            (after, 0.10, -0.80, 0.0),
        ]
    )
    assert equal_metrics.out_of_sample.max_drawdown == pytest.approx(0.25)
    assert equal_drawdown.decision == "pass"
    assert equal_drawdown.rule_id == "aligned_pass"

    over_drawdown, over_metrics = verdict(
        [
            (in_sample_end, 1.0, 0.02, 0.0),
            (inside, 0.749999, 0.01, 0.0),
        ]
    )
    assert over_metrics.out_of_sample.max_drawdown > 0.25
    assert over_drawdown.decision == "reject"
    assert over_drawdown.rule_id == "oos_drawdown"

    equal_turnover, turnover_metrics = verdict(
        [
            (in_sample_end, 1.0, 0.02, 0.0),
            (inside, 1.0, 0.01, 0.20),
            (after, 1.0, 0.01, 1.0),
        ]
    )
    assert turnover_metrics.out_of_sample.turnover == pytest.approx(0.20)
    assert turnover_metrics.in_sample.turnover == pytest.approx(0.0)
    assert equal_turnover.decision == "pass"
    assert equal_turnover.rule_id == "aligned_pass"

    hot_turnover, hot_metrics = verdict(
        [
            (in_sample_end, 1.0, 0.02, 0.0),
            (inside, 1.0, 0.01, 0.20 + 1e-6),
        ]
    )
    assert hot_metrics.out_of_sample.turnover > 0.20
    assert hot_turnover.decision == "reject"
    assert hot_turnover.rule_id == "turnover"

    aligned, aligned_metrics = verdict(
        [
            (in_sample_end, 1.0, 0.10, 0.0),
            (inside, 0.98, -0.02, 0.0),
            (after, 1.50, 0.50, 0.0),
        ]
    )
    assert aligned_metrics.in_sample.period_return == pytest.approx(0.10)
    assert aligned_metrics.out_of_sample.period_return == pytest.approx(-0.02)
    assert aligned.decision == "suspected_overfit"
    assert aligned.rule_id == "suspected_overfit"


@pytest.mark.parametrize("name", ["A", "B", "C", "default"])
def test_one_run_proposes_once_on_the_selected_in_sample_summary(name, tmp_path, monkeypatch):
    spec = FROZEN[name]
    bars = _bars(
        [_BEFORE, _SHARED_IN_SAMPLE, spec["in_sample_end"], spec["out_of_sample_start"], spec["day_after_oos"]],
        [1.0, 100.0, 110.0, 50.0, 7.0],
    )
    provider = _RecordingProvider()

    def fake_backtest(source, frame, **kwargs):
        provider.backtests += 1
        with use_split(name):
            metrics = evaluate(_window_curve(spec))
        return BacktestRecord(
            ok=True,
            status="ok",
            detail="",
            verdict=judge(metrics),
            metrics=metrics,
            curve=None,
            elapsed_ms=1,
        )

    monkeypatch.setattr("agent.loop.backtest_source", fake_backtest)
    root = tmp_path / "repo"
    (root / "reports").mkdir(parents=True)
    demo = root / "reports" / "agent_demo"
    document = run_bounded_experiment(
        {"510300": bars},
        provider,
        demo,
        root=root,
        run_failure_probes=False,
        fold=name,
    )
    assert provider.calls == 1
    assert provider.backtests_at_propose == 0
    assert provider.backtests == 2
    assert document["stop"]["model_calls_after_oos"] == 0
    assert document["stop"]["provider_calls"] == 1
    seen = provider.seen[0]["510300"]
    assert seen["in_sample_end"] == spec["in_sample_end"]
    assert seen["end"] == spec["in_sample_end"]
    assert seen["start"] == _SHARED_IN_SAMPLE
    assert seen["bar_count"] == 2
    assert seen["close_end"] == pytest.approx(110.0)
    assert "out_of_sample" not in seen
    proposal = json.loads((demo / "proposal.json").read_text(encoding="utf-8"))
    assert proposal["summaries"] == [seen]
    assert proposal["holdout"]["fold"] == name
    assert proposal["holdout"]["in_sample_end"] == spec["in_sample_end"]
    assert proposal["holdout"]["out_of_sample_start"] == spec["out_of_sample_start"]
    assert proposal["holdout"]["out_of_sample_end"] == spec["out_of_sample_end"]
    assert document["results"][0]["in_sample_end"] == spec["in_sample_end"]
    assert document["results"][0]["out_of_sample_start"] == spec["out_of_sample_start"]
    assert document["results"][0]["out_of_sample_end"] == spec["out_of_sample_end"]
    with use_split(name):
        expected = judge(evaluate(_window_curve(spec)))
    assert expected.rule_id == "aligned_pass"
    assert document["results"][0]["decision"] == expected.decision
    assert document["results"][0]["rule_id"] == expected.rule_id
    assert active_split().name == "default"


def test_agent_cli_selects_one_fold_and_rejects_any_other(monkeypatch, capsys):
    from agent.cli import main

    seen = {}

    def fake_run(*args, **kwargs):
        seen["fold"] = kwargs["fold"]
        return {
            "stop": {"reason": "oos_verdict_disclosed", "provider_calls": 1},
            "failures": [],
            "results": [],
        }

    monkeypatch.setattr("agent.cli.run_bounded_experiment", fake_run)
    monkeypatch.setattr("agent.cli.load_universe_bars", lambda: {"510300": object()})
    assert main(["--demo", "--out", "reports/fold_default"]) == 0
    assert seen["fold"] == "default"
    assert main(["--demo", "--fold", "A", "--out", "reports/fold_a"]) == 0
    assert seen["fold"] == "A"
    with pytest.raises(SystemExit) as caught:
        main(["--demo", "--fold", "D", "--out", "reports/fold_d"])
    assert caught.value.code != 0
    captured = capsys.readouterr()
    assert "sk-" not in captured.out
    assert "sk-" not in captured.err


def test_readme_names_the_three_folds_without_replacing_the_default():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "--fold" in text
    assert "dates.folds" in text
    for stamp in ("2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31", "2024-01-01", "2026-10-31"):
        assert stamp in text
    assert "DEEPSEEK_API_KEY" not in text.split("## 样本切分", 1)[-1].split("## ", 1)[0]


def test_run_rejects_an_unknown_fold_before_propose(tmp_path):
    root = tmp_path / "repo"
    (root / "reports").mkdir(parents=True)
    provider = _RecordingProvider()
    with pytest.raises(ValueError, match="A"):
        run_bounded_experiment(
            {"510300": _bars(["2020-06-01"], [100.0])},
            provider,
            root / "reports" / "agent_demo",
            root=root,
            run_failure_probes=False,
            fold="2024",
        )
    assert provider.calls == 0
