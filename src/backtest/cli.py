"""Run the MA-cross pipe on local CSVs of clean daily bars."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from backtest.runner import load_bars, run_backtest
from backtest.strategies.ma_cross import ma_cross
from config_loader import ROOT, load_config
from eval.judge import judge
from eval.metrics import evaluate

# pass 只表示没有触发淘汰线。它不表示这条规则能赚钱，也不能拿去实盘。
DISCLAIMER = "pass 只表示没有触发淘汰线，不表示策略盈利，也不表示可以实盘。"


def clean_directory() -> Path:
    return ROOT / load_config()["paths"]["clean"]


def default_csv_prefix() -> Path:
    return ROOT / "reports" / "fixture_run"


def default_universe_dir() -> Path:
    return ROOT / load_config()["paths"]["reports"] / "baseline"


def _source_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _payload(metrics, verdict, curve: pd.DataFrame) -> dict:
    return {
        "in_sample_end": metrics.in_sample_end,
        "out_of_sample_start": metrics.out_of_sample_start,
        "in_sample": metrics.in_sample.as_dict(),
        "out_of_sample": metrics.out_of_sample.as_dict(),
        "decision": verdict.decision,
        "rule_id": verdict.rule_id,
        "reason": verdict.reason,
        "curve_rows": int(len(curve)),
        "equity_start": float(curve["equity"].iloc[0]),
        "equity_end": float(curve["equity"].iloc[-1]),
    }


def _write_curve(path: Path, curve: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = curve.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    frame.to_csv(path, index=False)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _print_run(metrics, verdict, curve: pd.DataFrame, curve_path: Path, summary_path: Path) -> None:
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
    print(f"curve_rows: {len(curve)}")
    print(f"equity_start: {curve['equity'].iloc[0]:.6f}")
    print(f"equity_end: {curve['equity'].iloc[-1]:.6f}")
    print(f"wrote_curve: {curve_path}")
    print(f"wrote_summary: {summary_path}")


def run_configured_ma_cross(data: pd.DataFrame):
    """Run the config-owned 20/60 cross through the single backtest entry."""
    windows = load_config()["strategy"]["ma_cross"]
    result = run_backtest(
        data,
        ma_cross,
        params={
            "short_window": windows["short_window"],
            "long_window": windows["long_window"],
        },
    )
    metrics = evaluate(result.curve)
    verdict = judge(metrics)
    return result, metrics, verdict, windows


def run_csv(csv_path: Path, out_prefix: Path) -> dict:
    bars = load_bars(csv_path)
    result, metrics, verdict, _windows = run_configured_ma_cross(bars)
    payload = _payload(metrics, verdict, result.curve)
    curve_path = Path(str(out_prefix) + "_curve.csv")
    summary_path = Path(str(out_prefix) + "_summary.json")
    _write_curve(curve_path, result.curve)
    _write_json(summary_path, payload)
    _print_run(metrics, verdict, result.curve, curve_path, summary_path)
    return payload


def run_universe(clean_dir: Path, out_dir: Path) -> dict:
    """Load `<clean_dir>/<symbol>.csv` for each configured symbol and write reports."""
    config = load_config()
    missing: list[str] = []
    queued: list[tuple[str, str, Path]] = []
    for item in config["universe"]:
        symbol = str(item["symbol"])
        name = str(item["name"]) if "name" in item else ""
        path = clean_dir / f"{symbol}.csv"
        if not path.is_file():
            missing.append(f"{symbol}: {path}")
        else:
            queued.append((symbol, name, path))
    if missing:
        joined = "; ".join(missing)
        raise FileNotFoundError(f"clean bars missing ({joined})")

    runs: list[dict] = []
    for symbol, name, path in queued:
        bars = load_bars(path)
        result, metrics, verdict, windows = run_configured_ma_cross(bars)
        payload = {
            "symbol": symbol,
            "name": name,
            "source": _source_label(path),
            "short_window": windows["short_window"],
            "long_window": windows["long_window"],
            "fees": 0.0,
            "bar_start": pd.to_datetime(result.curve["date"].iloc[0]).strftime("%Y-%m-%d"),
            "bar_end": pd.to_datetime(result.curve["date"].iloc[-1]).strftime("%Y-%m-%d"),
            **_payload(metrics, verdict, result.curve),
            "curve": f"{symbol}_curve.csv",
            "summary": f"{symbol}_summary.json",
        }
        runs.append({"payload": payload, "curve": result.curve, "metrics": metrics, "verdict": verdict})

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[dict] = []
    for run in runs:
        payload = run["payload"]
        symbol = payload["symbol"]
        curve_path = out_dir / f"{symbol}_curve.csv"
        summary_path = out_dir / f"{symbol}_summary.json"
        _write_curve(curve_path, run["curve"])
        _write_json(summary_path, payload)
        print(f"symbol: {symbol}")
        _print_run(run["metrics"], run["verdict"], run["curve"], curve_path, summary_path)
        print()
        written.append(payload)

    first = written[0]
    combined = {
        "strategy": "ma_cross",
        "short_window": first["short_window"],
        "long_window": first["long_window"],
        "fees": 0.0,
        "adjust": config["data"]["adjust"],
        "fill": "信号用当日收盘，下一根 bar 的收盘成交",
        "in_sample_end": first["in_sample_end"],
        "out_of_sample_start": first["out_of_sample_start"],
        "disclaimer": DISCLAIMER,
        "symbols": written,
    }
    summary_path = out_dir / "summary.json"
    _write_json(summary_path, combined)
    print(f"wrote_summary: {summary_path}")
    return combined


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="用本地日线跑双均线、指标和写死的判定。")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", help="单条序列，列：date, open, high, low, close, volume")
    group.add_argument(
        "--universe",
        action="store_true",
        help="按 config.yaml 的 universe 读取 data/clean/<symbol>.csv",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="--csv 时是输出前缀；--universe 时是目录。默认分别是 reports/fixture_run 和 reports/baseline",
    )
    args = parser.parse_args(argv)

    if args.csv:
        out = Path(args.out) if args.out else default_csv_prefix()
        run_csv(Path(args.csv), out)
        return 0

    out_dir = Path(args.out) if args.out else default_universe_dir()
    try:
        run_universe(clean_directory(), out_dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
