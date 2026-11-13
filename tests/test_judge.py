from __future__ import annotations

import inspect

import pandas as pd
import pytest

from eval.judge import OOS_MAX_DRAWDOWN_LIMIT, TURNOVER_LIMIT, judge
from eval.metrics import PeriodMetrics, SplitMetrics, evaluate
from tests.conftest import FIXTURES, ROOT


def _metrics(
    is_return: float,
    oos_return: float,
    is_dd: float = 0.0,
    oos_dd: float = 0.0,
    is_turnover: float = 0.01,
    oos_turnover: float = 0.01,
) -> SplitMetrics:
    return SplitMetrics(
        in_sample=PeriodMetrics(is_return, is_dd, is_turnover, 1),
        out_of_sample=PeriodMetrics(oos_return, oos_dd, oos_turnover, 1),
        in_sample_end="2023-12-31",
        out_of_sample_start="2024-01-01",
    )


def test_thresholds_and_signature_are_fixed():
    assert OOS_MAX_DRAWDOWN_LIMIT == 0.25
    assert TURNOVER_LIMIT == 0.20
    names = list(inspect.signature(judge).parameters)
    assert "metrics" in names
    assert "pipeline_error" in names
    assert "baseline" in names
    assert "threshold" not in names
    assert "in_sample_end" not in names
    assert "limit" not in names


def test_drawdown_limit_is_strictly_above_the_threshold():
    assert judge(_metrics(0.02, 0.01, oos_dd=0.25)).decision == "pass"
    verdict = judge(_metrics(0.02, 0.01, oos_dd=0.25 + 1e-6))
    assert verdict.decision == "reject"
    assert verdict.rule_id == "oos_drawdown"


def test_turnover_limit_uses_the_higher_sample_and_is_strictly_above():
    assert judge(_metrics(0.02, 0.01, is_turnover=0.20, oos_turnover=0.01)).decision == "pass"
    in_sample = judge(_metrics(0.02, 0.01, is_turnover=0.20 + 1e-6, oos_turnover=0.0))
    out_of_sample = judge(_metrics(0.02, 0.01, is_turnover=0.0, oos_turnover=0.20 + 1e-6))
    assert in_sample.rule_id == "turnover"
    assert out_of_sample.rule_id == "turnover"
    assert in_sample.decision == "reject"


def test_drawdown_reject_wins_over_an_overfit_pattern():
    verdict = judge(_metrics(0.20, -0.10, oos_dd=0.40))
    assert verdict.rule_id == "oos_drawdown"
    assert verdict.decision == "reject"


def test_sign_disagreement_is_suspected_overfitting():
    verdict = judge(_metrics(0.10, -0.02, oos_dd=0.05))
    assert verdict.decision == "suspected_overfit"
    assert verdict.rule_id == "suspected_overfit"
    assert "样本内收益为正" in verdict.reason
    assert "样本外收益为负" in verdict.reason


def test_baseline_improvement_only_in_sample_is_suspected_overfitting():
    previous = _metrics(0.02, 0.05)
    current = _metrics(0.08, 0.01)
    verdict = judge(current, baseline=previous)
    assert verdict.decision == "suspected_overfit"
    assert "样本内变好" in verdict.reason
    assert "样本外变差" in verdict.reason


def test_baseline_does_not_flag_when_out_of_sample_does_not_get_worse():
    previous = _metrics(0.02, 0.05)
    unchanged = judge(_metrics(0.08, 0.05), baseline=previous)
    improved = judge(_metrics(0.08, 0.06), baseline=previous)
    assert unchanged.decision == "pass"
    assert improved.decision == "pass"


def test_same_sign_losses_can_be_kept():
    verdict = judge(_metrics(-0.04, -0.01, oos_dd=0.10, is_turnover=0.02, oos_turnover=0.02))
    assert verdict.decision == "pass"
    assert verdict.rule_id == "aligned_pass"


def test_opposite_sign_without_the_overfit_pattern_is_rejected():
    verdict = judge(_metrics(-0.04, 0.03))
    assert verdict.decision == "reject"
    assert verdict.rule_id == "not_aligned"


def test_pipeline_failure_is_its_own_decision():
    good = _metrics(0.05, 0.04)
    with_metrics = judge(good, pipeline_error="timeout")
    without_metrics = judge(None, pipeline_error="syntax error")
    assert with_metrics.decision == "failure"
    assert with_metrics.rule_id == "pipeline_failure"
    assert without_metrics.decision == "failure"
    assert without_metrics.rule_id == "pipeline_failure"
    assert "疑似过拟合" not in with_metrics.reason
    assert "淘汰" not in with_metrics.reason


def test_every_reason_is_one_sentence():
    cases = [
        judge(_metrics(0.02, 0.01)),
        judge(_metrics(0.02, 0.01, oos_dd=0.30)),
        judge(_metrics(0.02, 0.01, oos_turnover=0.50)),
        judge(_metrics(0.10, -0.02)),
        judge(_metrics(0.08, 0.01), baseline=_metrics(0.02, 0.05)),
        judge(_metrics(-0.02, 0.03)),
        judge(None, pipeline_error="timeout"),
    ]
    for verdict in cases:
        assert verdict.reason.count("。") == 1
        assert "\n" not in verdict.reason


def test_judge_ignores_the_split_labels_on_the_metrics():
    base = _metrics(0.05, 0.02)
    relabeled = SplitMetrics(
        in_sample=base.in_sample,
        out_of_sample=base.out_of_sample,
        in_sample_end="1999-01-01",
        out_of_sample_start="1999-01-02",
    )
    assert judge(base) == judge(relabeled)


def test_missing_input_is_an_error():
    with pytest.raises(ValueError):
        judge(None)
    with pytest.raises(ValueError):
        judge(pipeline_error="   ")


@pytest.mark.parametrize(
    ("filename", "decision", "rule_id"),
    [
        ("curve_pass.csv", "pass", "aligned_pass"),
        ("curve_reject_drawdown.csv", "reject", "oos_drawdown"),
        ("curve_suspected_overfit.csv", "suspected_overfit", "suspected_overfit"),
    ],
)
def test_curve_fixtures_exercise_the_fixed_rules(filename, decision, rule_id):
    curve = pd.read_csv(FIXTURES / filename)
    assert "2023-12-31" in set(curve["date"].astype(str))
    assert "2024-01-01" in set(curve["date"].astype(str))
    verdict = judge(evaluate(curve))
    assert verdict.decision == decision
    assert verdict.rule_id == rule_id


def test_readme_states_the_rules_and_the_fixture_command():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "vectorbt" in text
    assert "tests/fixtures/synthetic_daily_bars.csv" in text
    assert "2023-12-31" in text
    assert "2024-01-01" in text
    assert "25%" in text
    assert "20%" in text
    assert "前复权" in text
    assert "疑似过拟合" in text
    assert "管道失败" in text
