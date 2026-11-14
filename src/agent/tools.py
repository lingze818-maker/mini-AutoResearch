"""Host tools. The model does not receive paths, a shell, or the judge."""

from __future__ import annotations

import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from agent.limits import DEFAULT_TIMEOUT_S
from agent.sandbox import SandboxOutcome, replay_strategy, run_candidate
from backtest.cli import DISCLAIMER
from backtest.runner import load_bars, run_backtest, validate_bars
from config_loader import ROOT, load_config
from eval.judge import JudgeResult, judge
from eval.metrics import SplitMetrics, evaluate

_SYMBOL = re.compile(r"^[0-9A-Za-z._-]{1,16}$")
_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")
_FILENAME = re.compile(r"^[A-Za-z0-9._-]{1,80}$")


@dataclass(frozen=True, slots=True)
class BacktestRecord:
    ok: bool
    status: str
    detail: str
    verdict: JudgeResult
    metrics: SplitMetrics | None
    curve: pd.DataFrame | None
    elapsed_ms: int


def configured_symbols() -> tuple[str, ...]:
    return tuple(str(item["symbol"]) for item in load_config()["universe"])


def in_sample_slice(bars: pd.DataFrame) -> pd.DataFrame:
    frame = validate_bars(bars)
    end = pd.Timestamp(load_config()["dates"]["in_sample_end"]).normalize()
    kept = frame.loc[frame["date"] <= end].reset_index(drop=True)
    if kept.empty:
        raise ValueError("no in-sample bars")
    return kept


def summarize_bars(symbol: str, bars: pd.DataFrame) -> dict:
    """Describe only the in-sample window. Out-of-sample rows are dropped first."""
    if not _SYMBOL.fullmatch(symbol):
        raise ValueError("invalid symbol")
    frame = in_sample_slice(bars)
    close = frame["close"].astype(float)
    returns = close.pct_change().iloc[1:]
    deviation = float(returns.std(ddof=0)) if len(returns) else 0.0
    return {
        "symbol": symbol,
        "in_sample_end": str(load_config()["dates"]["in_sample_end"]),
        "bar_count": int(len(frame)),
        "start": pd.Timestamp(frame["date"].iloc[0]).strftime("%Y-%m-%d"),
        "end": pd.Timestamp(frame["date"].iloc[-1]).strftime("%Y-%m-%d"),
        "close_start": round(float(close.iloc[0]), 6),
        "close_end": round(float(close.iloc[-1]), 6),
        "period_return": round(float(close.iloc[-1] / close.iloc[0] - 1.0), 6),
        "daily_return_std": round(deviation, 6),
    }


def summarize_in_sample(symbol: str) -> dict:
    """Load one configured clean file. The caller cannot pass a path."""
    if symbol not in configured_symbols():
        raise ValueError("unknown symbol")
    path = ROOT / load_config()["paths"]["clean"] / f"{symbol}.csv"
    return summarize_bars(symbol, load_bars(path))


def backtest_source(source: str, bars: pd.DataFrame, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> BacktestRecord:
    """Sandbox the source, then simulate the validated signals with the one runner."""
    try:
        outcome = run_candidate(source, bars, timeout_s=timeout_s)
    except Exception as exc:
        outcome = SandboxOutcome(False, "isolation_unavailable", f"sandbox raised {type(exc).__name__}")
    if not outcome.ok:
        detail = f"{outcome.status}: {outcome.detail or outcome.status}"
        return BacktestRecord(
            ok=False,
            status=outcome.status,
            detail=outcome.detail,
            verdict=judge(pipeline_error=detail),
            metrics=None,
            curve=None,
            elapsed_ms=outcome.elapsed_ms,
        )
    try:
        result = run_backtest(bars, replay_strategy(outcome.entries, outcome.exits))
        metrics = evaluate(result.curve)
    except Exception as exc:
        detail = " ".join(f"runtime_error: {type(exc).__name__}: {exc}".split())[:400]
        return BacktestRecord(
            ok=False,
            status="runtime_error",
            detail=detail,
            verdict=judge(pipeline_error=detail),
            metrics=None,
            curve=None,
            elapsed_ms=outcome.elapsed_ms,
        )
    return BacktestRecord(
        ok=True,
        status="ok",
        detail="",
        verdict=judge(metrics),
        metrics=metrics,
        curve=result.curve,
        elapsed_ms=outcome.elapsed_ms,
    )


def assert_reports_target(path: Path, *, root: Path = ROOT) -> Path:
    resolved = path.resolve()
    reports = (root / "reports").resolve()
    baseline = (reports / "baseline").resolve()
    if resolved == baseline or _is_relative_to(resolved, baseline):
        raise ValueError("refusing to write the baseline report directory")
    if resolved != reports and not _is_relative_to(resolved, reports):
        raise ValueError("report output must stay under reports/")
    if resolved == reports:
        raise ValueError("refusing to write the reports root")
    return resolved


def resolve_run_directory(run_id: str, *, root: Path = ROOT) -> Path:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("invalid run id")
    return assert_reports_target(root / "reports" / "agent_runs" / run_id, root=root)


def prepare_report_dir(path: Path, *, root: Path = ROOT, rerun: bool = False) -> Path:
    """Create a run directory. ``rerun`` replaces only that directory."""
    resolved = assert_reports_target(path, root=root)
    if resolved.exists():
        if not rerun:
            raise FileExistsError(
                f"{resolved} already exists; rerun replaces only this run directory"
            )
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def write_named_files(directory: Path, files: Mapping[str, str]) -> None:
    root = directory.resolve()
    for name, text in files.items():
        if not isinstance(name, str) or not _FILENAME.fullmatch(name) or ".." in name:
            raise ValueError("invalid report filename")
        path = (root / name).resolve()
        if path.parent != root:
            raise ValueError("invalid report filename")
        path.write_text(text, encoding="utf-8")


def load_universe_bars() -> dict[str, pd.DataFrame]:
    clean = ROOT / load_config()["paths"]["clean"]
    loaded = {}
    for symbol in configured_symbols():
        loaded[symbol] = load_bars(clean / f"{symbol}.csv")
    return loaded


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


DISCLAIMER_TEXT = DISCLAIMER
