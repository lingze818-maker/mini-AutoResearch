from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from agent.limits import HOLDOUT_NOTE, OFFLINE_BANNER, STOP_REASON
from agent.loop import run_bounded_experiment
from agent.model import LiveDeepSeekProvider, ModelError, build_provider
from agent.offline import Candidate, OfflineFixtureProvider
from agent.tools import (
    assert_reports_target,
    prepare_report_dir,
    resolve_run_directory,
    summarize_bars,
    summarize_in_sample,
)
from config_loader import load_config
from eval.judge import judge
from eval.metrics import evaluate
from tests.conftest import ROOT, bars_from_closes


def _spanning() -> pd.DataFrame:
    dates = pd.bdate_range("2023-11-01", periods=50)
    closes = [100 + index * 0.2 for index in range(len(dates))]
    frame = bars_from_closes(closes)
    frame["date"] = dates
    return frame


def _root(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    (root / "reports").mkdir(parents=True)
    return root, root / "reports" / "agent_demo"


class _Scripted:
    banner = OFFLINE_BANNER
    name = "scripted"
    calls = 0

    def __init__(self, candidates: list[Candidate]) -> None:
        self._candidates = candidates

    def propose(self, summaries):
        self.calls += 1
        assert all("out_of_sample" not in item for item in summaries.values())
        return list(self._candidates)


def test_summary_uses_only_in_sample_rows():
    bars = _spanning()
    original = summarize_bars("510300", bars)
    tweaked = bars.copy()
    out_of_sample = tweaked["date"] > pd.Timestamp("2023-12-31")
    tweaked.loc[out_of_sample, "close"] = tweaked.loc[out_of_sample, "close"] * 5
    tweaked["high"] = tweaked[["open", "close"]].max(axis=1) * 1.001
    tweaked["low"] = tweaked[["open", "close"]].min(axis=1) * 0.999
    assert summarize_bars("510300", tweaked) == original
    assert "out_of_sample" not in original
    assert original["end"] <= "2023-12-31"
    real = summarize_in_sample("510300")
    assert real["end"] <= load_config()["dates"]["in_sample_end"]
    assert real["bar_count"] < 1897


def test_report_paths_reject_traversal_and_rerun_is_local(tmp_path):
    root, demo = _root(tmp_path)
    baseline = root / "reports" / "baseline"
    baseline.mkdir()
    (baseline / "summary.json").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError):
        assert_reports_target(root / "reports" / "agent_demo" / ".." / "baseline", root=root)
    with pytest.raises(ValueError):
        resolve_run_directory("../baseline", root=root)
    with pytest.raises(ValueError):
        resolve_run_directory("a/b", root=root)
    created = prepare_report_dir(demo, root=root, rerun=False)
    (created / "old.txt").write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_report_dir(demo, root=root, rerun=False)
    assert (created / "old.txt").read_text(encoding="utf-8") == "old"
    prepare_report_dir(demo, root=root, rerun=True)
    assert not (created / "old.txt").exists()
    assert (baseline / "summary.json").read_text(encoding="utf-8") == "keep"
    with pytest.raises(ValueError):
        prepare_report_dir(baseline, root=root, rerun=True)


def test_pass_still_stops_and_failures_do_not_drop_later_candidates(tmp_path):
    root, demo = _root(tmp_path)
    bars = {"510300": _spanning()}
    flat = "def generate(bars):\n    return {'entry': False, 'exit': False}\n"
    broken = "def generate(bars)\n    return {'entry': False, 'exit': False}\n"
    provider = _Scripted(
        [
            Candidate("broken", "broken", OFFLINE_BANNER, broken),
            Candidate("flat", "flat", OFFLINE_BANNER, flat),
        ]
    )
    document = run_bounded_experiment(
        bars,
        provider,
        demo,
        root=root,
        run_failure_probes=True,
    )
    assert document["banner"] == OFFLINE_BANNER
    assert document["stop"]["stopped"] is True
    assert document["stop"]["reason"] == STOP_REASON
    assert document["stop"]["provider_calls"] == 1
    assert document["stop"]["model_calls_after_oos"] == 0
    assert provider.calls == 1
    assert document["holdout"]["kind"] == "already_inspected_historical_holdout"
    assert HOLDOUT_NOTE in document["holdout"]["note"]
    assert document["sentinel"]["unchanged"] is True
    statuses = {item["probe"]: item["status"] for item in document["failures"]}
    assert statuses["syntax_error"] == "syntax_error"
    assert statuses["timeout"] == "timeout"
    assert statuses["unauthorized_read"] != "ok"
    assert statuses["unauthorized_write"] != "ok"
    assert statuses["network"] != "ok"
    assert statuses["subprocess"] != "ok"
    assert statuses["malformed_output"] == "malformed_output"
    assert statuses["parent_memory"] != "ok"
    assert statuses["repository_config"] != "ok"
    assert [item["rule_id"] for item in document["failures"]] == ["pipeline_failure"] * len(document["failures"])
    assert [item["candidate_id"] for item in document["results"]] == ["broken", "flat"]
    assert document["results"][0]["decision"] == "failure"
    assert document["results"][1]["decision"] == "pass"
    locked = json.loads((demo / "candidates_locked.json").read_text(encoding="utf-8"))
    assert locked["phase"] == "locked_before_oos"
    assert locked["oos_evaluated_at_lock"] is False
    assert "oos_evaluated" not in locked
    state = json.loads((demo / "evaluation_state.json").read_text(encoding="utf-8"))
    assert state["phase"] == "oos_disclosed"
    assert state["oos_evaluated"] is True
    assert state["oos_disclosed"] is True
    assert state["model_calls_after_oos"] == 0
    assert "def generate(bars)\n" in locked["candidates"][0]["source"]
    assert (demo / "sentinel_host_only.txt").read_text(encoding="utf-8") == "HOST-SENTINEL-UNCHANGED\n"
    assert OFFLINE_BANNER in (demo / "summary.json").read_text(encoding="utf-8")


def test_locked_sources_are_what_the_runner_receives(tmp_path, monkeypatch):
    root, demo = _root(tmp_path)
    seen: list[str] = []
    original = __import__("agent.tools", fromlist=["backtest_source"]).backtest_source

    def wrapped(source, bars, **kwargs):
        seen.append(source)
        return original(source, bars, **kwargs)

    monkeypatch.setattr("agent.loop.backtest_source", wrapped)
    provider = OfflineFixtureProvider()
    run_bounded_experiment(
        {"510300": _spanning()},
        provider,
        demo,
        root=root,
        run_failure_probes=False,
    )
    locked = json.loads((demo / "candidates_locked.json").read_text(encoding="utf-8"))
    assert [item["source"] for item in locked["candidates"]] == seen
    assert provider.calls == 1


def test_backtest_verdict_comes_from_the_frozen_judge(tmp_path, monkeypatch):
    import vectorbt as vbt

    root, demo = _root(tmp_path)
    called = {"from_signals": 0}
    original = vbt.Portfolio.from_signals

    def wrapped(*args, **kwargs):
        called["from_signals"] += 1
        assert kwargs.get("fees") == 0.0
        return original(*args, **kwargs)

    monkeypatch.setattr(vbt.Portfolio, "from_signals", wrapped)
    document = run_bounded_experiment(
        {"510300": _spanning()},
        OfflineFixtureProvider(),
        demo,
        root=root,
        run_failure_probes=False,
    )
    assert called["from_signals"] == 3
    curve = pd.read_csv(demo / "momentum_20_510300_curve.csv")
    metrics = evaluate(curve)
    verdict = judge(metrics)
    row = document["results"][0]
    assert row["decision"] == verdict.decision
    assert row["rule_id"] == verdict.rule_id
    assert row["reason"] == verdict.reason


def test_live_cli_missing_key_does_not_write_a_fixture(tmp_path, monkeypatch, capsys):
    from agent.cli import main

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_MODEL", "configured-by-operator")
    code = main(["--demo", "--live", "--out", "reports/agent_demo"])
    captured = capsys.readouterr()
    assert code == 2
    assert "DEEPSEEK_API_KEY" in captured.err
    assert "offline fixture" in captured.err
    assert OFFLINE_BANNER not in captured.out
    assert not (tmp_path / "reports").exists()


def test_committed_offline_demo_records_the_stop_and_the_sentinel():
    demo = ROOT / "reports" / "agent_demo"
    summary = json.loads((demo / "summary.json").read_text(encoding="utf-8"))
    assert summary["banner"] == OFFLINE_BANNER
    assert summary["provider"] == "offline-fixture"
    assert summary["stop"] == {
        "stopped": True,
        "reason": STOP_REASON,
        "provider_calls": 1,
        "model_calls_after_oos": 0,
    }
    assert summary["holdout"]["kind"] == "already_inspected_historical_holdout"
    assert summary["sentinel"]["unchanged"] is True
    sentinel = (demo / "sentinel_host_only.txt").read_bytes()
    assert summary["sentinel"]["sha256_before"] == summary["sentinel"]["sha256_after"]
    assert hashlib.sha256(sentinel).hexdigest() == summary["sentinel"]["sha256_before"]
    statuses = {item["probe"]: item["status"] for item in summary["failure_statuses"]}
    assert statuses["syntax_error"] == "syntax_error"
    assert statuses["timeout"] == "timeout"
    assert statuses["malformed_output"] == "malformed_output"
    assert statuses["network"] != "ok"
    assert statuses["subprocess"] != "ok"
    assert statuses["unauthorized_read"] != "ok"
    assert statuses["unauthorized_write"] != "ok"
    locked = json.loads((demo / "candidates_locked.json").read_text(encoding="utf-8"))
    assert locked["phase"] == "locked_before_oos"
    assert locked["oos_evaluated_at_lock"] is False
    assert "oos_evaluated" not in locked
    state = json.loads((demo / "evaluation_state.json").read_text(encoding="utf-8"))
    assert state["phase"] == "oos_disclosed"
    assert state["oos_evaluated"] is True
    assert state["oos_disclosed"] is True
    assert state["model_calls_after_oos"] == 0
    assert [item["id"] for item in locked["candidates"]] == ["momentum_20", "volatility_20", "momentum_60"]
    assert "def generate" in locked["candidates"][0]["source"]


def test_a_failed_proposal_writes_a_record_and_does_not_lock_candidates(tmp_path):
    root, demo = _root(tmp_path)
    secret = "sk-test-secret-value"
    source = "def generate(bars):\n    return {'entry': False, 'exit': False}\n" + secret

    def transport(_url, _headers, _body, _timeout):
        body = {
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "submit_candidates",
                                    "arguments": '{"candidates":[{"source":"' + secret,
                                },
                            }
                        ]
                    },
                }
            ],
        }
        return __import__("agent.model", fromlist=["HTTPResult"]).HTTPResult(
            200, json.dumps(body).encode("utf-8")
        )

    provider = LiveDeepSeekProvider(api_key=secret, model="configured-by-operator", transport=transport)
    with pytest.raises(ModelError, match="tool arguments are not valid json"):
        run_bounded_experiment(
            {"510300": _spanning()},
            provider,
            demo,
            root=root,
            run_failure_probes=False,
        )
    assert demo.is_dir()
    names = sorted(path.name for path in demo.iterdir())
    assert names == ["proposal_failure.json"]
    record = json.loads((demo / "proposal_failure.json").read_text(encoding="utf-8"))
    assert record["record_type"] == "proposal_failure"
    assert record["oos_disclosed"] is False
    assert record["http_attempts"] == 1
    assert record["diagnostics"]["finish_reason"] == "length"
    assert record["diagnostics"]["argument_type"] == "str"
    assert "source" not in record["diagnostics"]
    text = (demo / "proposal_failure.json").read_text(encoding="utf-8")
    assert secret not in text
    assert "def generate" not in text
    assert not (demo / "candidates_locked.json").exists()


def test_build_provider_live_false_ignores_a_present_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-secret-value")
    monkeypatch.setenv("DEEPSEEK_MODEL", "configured-by-operator")
    provider = build_provider(False)
    assert provider.banner == OFFLINE_BANNER
    assert provider.propose({"510300": {"symbol": "510300", "period_return": 0.0}}).__len__() == 3
