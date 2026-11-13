"""Hardcoded keep / reject rules. Thresholds are not function arguments."""

from __future__ import annotations

from dataclasses import dataclass

from eval.metrics import SplitMetrics

# Draft ceilings. Out-of-sample max drawdown or either sample's average daily
# turnover above these values rejects the strategy. Equal to the ceiling does not.
OOS_MAX_DRAWDOWN_LIMIT = 0.25
TURNOVER_LIMIT = 0.20


@dataclass(frozen=True, slots=True)
class JudgeResult:
    decision: str
    rule_id: str
    reason: str


def _clean_detail(pipeline_error: str) -> str:
    detail = " ".join(str(pipeline_error).replace("。", " ").split())
    if not detail:
        raise ValueError("pipeline_error must describe the failure")
    return detail


def _same_direction(left: float, right: float) -> bool:
    if left >= 0.0 and right >= 0.0:
        return True
    return left < 0.0 and right < 0.0


def judge(
    metrics: SplitMetrics | None = None,
    *,
    pipeline_error: str | None = None,
    baseline: SplitMetrics | None = None,
) -> JudgeResult:
    """Apply the fixed rules in order and return the first match.

    1. A pipeline error is a failure, recorded on its own.
    2. Out-of-sample max drawdown above 25% rejects.
    3. Average daily turnover above 20% in either sample rejects.
    4. A round that improves in-sample and deteriorates out-of-sample is suspected overfitting.
    5. Positive in-sample return and negative out-of-sample return is suspected overfitting.
    6. Same-sign returns with acceptable drawdown and turnover may take a small next change, or stop and keep the record.
    7. Any other sign split rejects.
    """
    if pipeline_error is not None:
        detail = _clean_detail(pipeline_error)
        return JudgeResult(
            decision="failure",
            rule_id="pipeline_failure",
            reason=f"管道失败（{detail}），本轮记录为失败。",
        )
    if not isinstance(metrics, SplitMetrics):
        raise ValueError("metrics are required when there is no pipeline error")
    if baseline is not None and not isinstance(baseline, SplitMetrics):
        raise TypeError("baseline must be SplitMetrics")

    out_of_sample = metrics.out_of_sample
    if out_of_sample.max_drawdown > OOS_MAX_DRAWDOWN_LIMIT:
        return JudgeResult(
            decision="reject",
            rule_id="oos_drawdown",
            reason=f"样本外最大回撤为 {out_of_sample.max_drawdown:.2%}，超过 25% 的阈值，淘汰。",
        )

    turnover = max(metrics.in_sample.turnover, out_of_sample.turnover)
    if turnover > TURNOVER_LIMIT:
        return JudgeResult(
            decision="reject",
            rule_id="turnover",
            reason=f"日均换手为 {turnover:.2%}，超过 20% 的阈值，淘汰。",
        )

    if baseline is not None:
        in_sample_improved = metrics.in_sample.period_return > baseline.in_sample.period_return
        out_of_sample_worse = out_of_sample.period_return < baseline.out_of_sample.period_return
        if in_sample_improved and out_of_sample_worse:
            return JudgeResult(
                decision="suspected_overfit",
                rule_id="suspected_overfit",
                reason="只在样本内变好、样本外变差，记为疑似过拟合，停止继续调参。",
            )

    if metrics.in_sample.period_return > 0.0 and out_of_sample.period_return < 0.0:
        return JudgeResult(
            decision="suspected_overfit",
            rule_id="suspected_overfit",
            reason="样本内收益为正、样本外收益为负，记为疑似过拟合，停止继续调参。",
        )

    if _same_direction(metrics.in_sample.period_return, out_of_sample.period_return):
        return JudgeResult(
            decision="pass",
            rule_id="aligned_pass",
            reason="样本内外收益方向一致，且回撤和换手未超过阈值，可以做一轮小改，或停止并保留记录。",
        )

    return JudgeResult(
        decision="reject",
        rule_id="not_aligned",
        reason="样本内外收益方向不一致，淘汰。",
    )
