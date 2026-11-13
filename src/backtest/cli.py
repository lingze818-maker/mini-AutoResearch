"""Run the MA-cross pipe on a local CSV of clean daily bars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest.runner import load_bars, run_backtest
from backtest.strategies.ma_cross import ma_cross
from config_loader import ROOT, load_config
from eval.judge import judge
from eval.metrics import evaluate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="用本地日线跑双均线、指标和写死的判定。")
    parser.add_argument("--csv", required=True, help="列：date, open, high, low, close, volume")
    parser.add_argument(
        "--out",
        default=str(ROOT / "reports" / "fixture_run"),
        help="输出前缀，写成 <out>_curve.csv 和 <out>_summary.json",
    )
    args = parser.parse_args(argv)

    windows = load_config()["strategy"]["ma_cross"]
    result = run_backtest(
        load_bars(args.csv),
        ma_cross,
        params={
            "short_window": windows["short_window"],
            "long_window": windows["long_window"],
        },
    )
    metrics = evaluate(result.curve)
    verdict = judge(metrics)

    out = Path(args.out)
    curve_path = Path(str(out) + "_curve.csv")
    summary_path = Path(str(out) + "_summary.json")
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve = result.curve.copy()
    curve["date"] = pd.to_datetime(curve["date"]).dt.strftime("%Y-%m-%d")
    curve.to_csv(curve_path, index=False)
    payload = {
        "in_sample_end": metrics.in_sample_end,
        "out_of_sample_start": metrics.out_of_sample_start,
        "in_sample": metrics.in_sample.as_dict(),
        "out_of_sample": metrics.out_of_sample.as_dict(),
        "decision": verdict.decision,
        "rule_id": verdict.rule_id,
        "reason": verdict.reason,
        "curve_rows": int(len(result.curve)),
        "equity_start": float(result.curve["equity"].iloc[0]),
        "equity_end": float(result.curve["equity"].iloc[-1]),
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"in_sample_end: {metrics.in_sample_end}")
    print(f"out_of_sample_start: {metrics.out_of_sample_start}")
    for name, period in (("in_sample", metrics.in_sample), ("out_of_sample", metrics.out_of_sample)):
        print(f"{name}.period_return: {period.period_return:.6f}")
        print(f"{name}.max_drawdown: {period.max_drawdown:.6f}")
        print(f"{name}.turnover: {period.turnover:.6f}")
        print(f"{name}.trade_count: {period.trade_count}")
    print(f"decision: {verdict.decision}")
    print(f"rule_id: {verdict.rule_id}")
    print(f"reason: {verdict.reason}")
    print(f"curve_rows: {len(result.curve)}")
    print(f"equity_start: {result.curve['equity'].iloc[0]:.6f}")
    print(f"equity_end: {result.curve['equity'].iloc[-1]:.6f}")
    print(f"wrote_curve: {curve_path}")
    print(f"wrote_summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
