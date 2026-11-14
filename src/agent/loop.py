"""One bounded propose, sandbox, backtest, judge, and stop cycle."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from agent.limits import HOLDOUT_NOTE, OFFLINE_BANNER, STOP_REASON
from agent.model import ModelError, validate_candidates
from agent.offline import Candidate
from agent.probes import containment_probes
from agent.sandbox import run_candidate
from agent.tools import (
    DISCLAIMER_TEXT,
    BacktestRecord,
    backtest_source,
    in_sample_slice,
    prepare_report_dir,
    summarize_bars,
    write_named_files,
)
from config_loader import CONFIG_PATH, load_config
from eval.judge import judge

_SENTINEL_TEXT = "HOST-SENTINEL-UNCHANGED\n"


def run_bounded_experiment(
    bars_by_symbol: Mapping[str, pd.DataFrame],
    provider,
    output_dir: Path,
    *,
    root: Path,
    rerun: bool = False,
    run_failure_probes: bool = True,
) -> dict:
    """Lock proposals from in-sample summaries, judge them once, then stop.

    The provider is not called again after the first out-of-sample verdict.
    Failure probes are recorded and do not remove the locked candidates.
    """
    if not bars_by_symbol:
        raise ValueError("at least one symbol is required")
    directory = prepare_report_dir(output_dir, root=root, rerun=rerun)
    summaries = {symbol: summarize_bars(symbol, bars) for symbol, bars in bars_by_symbol.items()}
    try:
        candidates = validate_candidates(list(provider.propose(summaries)))
    except ModelError as exc:
        _write_json(directory / "proposal_failure.json", _proposal_failure(provider, exc))
        raise
    proposal = {
        "banner": getattr(provider, "banner", ""),
        "provider": getattr(provider, "name", type(provider).__name__),
        "holdout": _holdout(),
        "summaries": list(summaries.values()),
        "candidates": [_candidate_record(item) for item in candidates],
    }
    locked = {
        "locked": True,
        "phase": "locked_before_oos",
        "oos_evaluated_at_lock": False,
        "banner": proposal["banner"],
        "candidates": proposal["candidates"],
    }
    _write_json(directory / "proposal.json", proposal)
    lock_path = directory / "candidates_locked.json"
    _write_json(lock_path, locked)
    locked_bytes = lock_path.read_bytes()
    log = [
        {"step": "summarize_in_sample", "symbols": list(summaries)},
        {"step": "propose", "provider": proposal["provider"], "count": len(candidates), "banner": proposal["banner"]},
        {"step": "lock_candidates", "ids": [item.id for item in candidates]},
    ]

    sentinel = directory / "sentinel_host_only.txt"
    sentinel.write_text(_SENTINEL_TEXT, encoding="utf-8")
    sentinel_before = _sha256(sentinel)
    failures = []
    if run_failure_probes:
        probe_bars = in_sample_slice(next(iter(bars_by_symbol.values()))).iloc[:8].reset_index(drop=True)
        failures = _run_probes(probe_bars, sentinel, log)
    sentinel_after = _sha256(sentinel)
    sentinel_unchanged = sentinel_before == sentinel_after and sentinel.read_text(encoding="utf-8") == _SENTINEL_TEXT
    if not sentinel_unchanged:
        document = _document(
            proposal,
            failures,
            [],
            stop_reason="sentinel_modified",
            sentinel_before=sentinel_before,
            sentinel_after=sentinel_after,
            provider_calls=getattr(provider, "calls", None),
        )
        _write_outputs(directory, document, {}, log)
        return document

    results = []
    curves: dict[str, str] = {}
    for candidate in _load_locked(lock_path):
        for symbol, bars in bars_by_symbol.items():
            record = backtest_source(candidate.source, bars)
            row = _result_row(candidate.id, symbol, record)
            results.append(row)
            log.append(
                {
                    "step": "backtest",
                    "candidate": candidate.id,
                    "symbol": symbol,
                    "status": record.status,
                    "decision": record.verdict.decision,
                    "rule_id": record.verdict.rule_id,
                }
            )
            if record.curve is not None:
                name = f"{candidate.id}_{symbol}_curve.csv"
                curves[name] = _curve_csv(record.curve)
                row["curve"] = name
    if lock_path.read_bytes() != locked_bytes:
        raise RuntimeError("locked candidate definitions changed after the out-of-sample run")
    log.append({"step": "stop", "reason": STOP_REASON, "provider_calls": getattr(provider, "calls", None)})
    document = _document(
        proposal,
        failures,
        results,
        stop_reason=STOP_REASON,
        sentinel_before=sentinel_before,
        sentinel_after=sentinel_after,
        provider_calls=getattr(provider, "calls", None),
    )
    _write_outputs(directory, document, curves, log)
    return document


def _run_probes(bars: pd.DataFrame, sentinel: Path, log: list[dict]) -> list[dict]:
    secret = sentinel.read_text(encoding="utf-8")
    rows = []
    for probe in containment_probes(str(sentinel.resolve()), str(CONFIG_PATH.resolve())):
        outcome = run_candidate(probe.source, bars, timeout_s=probe.timeout_s)
        detail = _redact(outcome.detail, secret)
        verdict = judge(pipeline_error=f"{outcome.status}: {detail or outcome.status}")
        row = {
            "probe": probe.name,
            "status": outcome.status,
            "detail": detail,
            "decision": verdict.decision,
            "rule_id": verdict.rule_id,
            "reason": verdict.reason,
            "elapsed_ms": outcome.elapsed_ms,
        }
        rows.append(row)
        log.append({"step": "sandbox_failure_probe", "probe": probe.name, "status": outcome.status})
    return rows


def _load_locked(path: Path) -> list[Candidate]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "oos_evaluated" in payload:
        raise RuntimeError("lock file must record oos_evaluated_at_lock, not a current oos_evaluated flag")
    if payload.get("phase") != "locked_before_oos" or payload.get("oos_evaluated_at_lock") is not False:
        raise RuntimeError("refusing to backtest a lock file that is not a pre-evaluation snapshot")
    candidates = [
        Candidate(item["id"], item["name"], item["rationale"], item["source"])
        for item in payload["candidates"]
    ]
    return validate_candidates(candidates)


def _candidate_record(candidate: Candidate) -> dict:
    return {
        "id": candidate.id,
        "name": candidate.name,
        "rationale": candidate.rationale,
        "source": candidate.source,
        "source_sha256": hashlib.sha256(candidate.source.encode("utf-8")).hexdigest(),
        "mechanism": "prefix_only_generate",
    }


def _result_row(candidate_id: str, symbol: str, record: BacktestRecord) -> dict:
    row = {
        "candidate_id": candidate_id,
        "symbol": symbol,
        "status": record.status,
        "detail": record.detail,
        "decision": record.verdict.decision,
        "rule_id": record.verdict.rule_id,
        "reason": record.verdict.reason,
        "elapsed_ms": record.elapsed_ms,
        "fees": 0.0,
        "fill": "信号用当日收盘，下一根 bar 的收盘成交",
    }
    if record.metrics is not None:
        row["in_sample_end"] = record.metrics.in_sample_end
        row["out_of_sample_start"] = record.metrics.out_of_sample_start
        row["in_sample"] = record.metrics.in_sample.as_dict()
        row["out_of_sample"] = record.metrics.out_of_sample.as_dict()
    return row


def _holdout() -> dict:
    dates = load_config()["dates"]
    return {
        "kind": "already_inspected_historical_holdout",
        "in_sample_end": str(dates["in_sample_end"]),
        "out_of_sample_start": str(pd.Timestamp(dates["in_sample_end"]) + pd.Timedelta(days=1))[:10],
        "note": HOLDOUT_NOTE,
    }


def _document(proposal, failures, results, *, stop_reason, sentinel_before, sentinel_after, provider_calls) -> dict:
    return {
        "banner": proposal["banner"],
        "offline_banner_exact": OFFLINE_BANNER if proposal["banner"] == OFFLINE_BANNER else "",
        "provider": proposal["provider"],
        "holdout": proposal["holdout"],
        "disclaimer": DISCLAIMER_TEXT,
        "stop": {
            "stopped": True,
            "reason": stop_reason,
            "provider_calls": provider_calls,
            "model_calls_after_oos": 0,
        },
        "sentinel": {
            "sha256_before": sentinel_before,
            "sha256_after": sentinel_after,
            "unchanged": sentinel_before == sentinel_after,
        },
        "candidates": proposal["candidates"],
        "failures": failures,
        "results": results,
        "evaluation_state": _evaluation_state(stop_reason, provider_calls),
        "sandbox": {
            "mechanism": "linux user, mount, pid, net, ipc, and uts namespaces with pivot_root, dropped capabilities, and seccomp",
            "fail_closed": True,
            "candidate_execution": "isolated helper only; the host replays validated booleans through vectorbt",
            "process_creation": "denied in the worker after pandas import; cgroup pids controller is not writable",
            "writable_storage": "root is a read-only tmpfs; only /work, /tmp, and /dev are writable tmpfs",
        },
    }


def _evaluation_state(stop_reason: str, provider_calls) -> dict:
    disclosed = stop_reason == STOP_REASON
    return {
        "phase": "oos_disclosed" if disclosed else "stopped_before_oos",
        "oos_evaluated": disclosed,
        "oos_disclosed": disclosed,
        "stop_reason": stop_reason,
        "provider_calls": provider_calls,
        "model_calls_after_oos": 0,
        "lock_file": "candidates_locked.json",
        "lock_note": (
            "candidates_locked.json is the immutable pre-evaluation snapshot. "
            "oos_evaluated_at_lock false means that snapshot was sealed before any "
            "out-of-sample result. It is not the final status."
        ),
    }


def _write_outputs(directory: Path, document: dict, curves: dict[str, str], log: list[dict]) -> None:
    files = {
        "summary.json": _dumps(_summary(document)),
        "results.json": _dumps(document["results"]),
        "failures.json": _dumps(document["failures"]),
        "evaluation_state.json": _dumps(document["evaluation_state"]),
        "run_log.json": _dumps(log),
    }
    files.update(curves)
    write_named_files(directory, files)


def _summary(document: dict) -> dict:
    return {
        "banner": document["banner"],
        "provider": document["provider"],
        "holdout": document["holdout"],
        "disclaimer": document["disclaimer"],
        "stop": document["stop"],
        "evaluation_state": document["evaluation_state"],
        "sentinel": document["sentinel"],
        "sandbox": document["sandbox"],
        "candidate_ids": [item["id"] for item in document["candidates"]],
        "failure_statuses": [
            {"probe": item["probe"], "status": item["status"], "rule_id": item["rule_id"]}
            for item in document["failures"]
        ],
        "results": [
            {
                "candidate_id": item["candidate_id"],
                "symbol": item["symbol"],
                "decision": item["decision"],
                "rule_id": item["rule_id"],
                "reason": item["reason"],
                "in_sample": item.get("in_sample"),
                "out_of_sample": item.get("out_of_sample"),
                "curve": item.get("curve"),
            }
            for item in document["results"]
        ],
    }


def _curve_csv(curve: pd.DataFrame) -> str:
    frame = curve.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    return frame.to_csv(index=False)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(_dumps(payload), encoding="utf-8")


def _dumps(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _redact(text: str, secret: str) -> str:
    cleaned = secret.strip()
    if cleaned and cleaned in text:
        return text.replace(cleaned, "[redacted]")
    return text


def _proposal_failure(provider, exc: ModelError) -> dict:
    """Structured stop before any candidate is locked. No argument text and no secrets."""
    message = _scrub(" ".join(str(exc).split()))[:200]
    diagnostics = getattr(exc, "diagnostics", {}) or {}
    allowed = ("finish_reason", "tool_name", "argument_type", "argument_length", "json_error", "usage")
    safe = {key: diagnostics[key] for key in allowed if key in diagnostics}
    return {
        "record_type": "proposal_failure",
        "status": "proposal_failed",
        "error": message or "proposal failed",
        "oos_evaluated": False,
        "oos_disclosed": False,
        "provider": str(getattr(provider, "name", ""))[:80],
        "provider_calls": getattr(provider, "calls", None),
        "http_attempts": getattr(provider, "http_attempts", None),
        "http_responses": getattr(provider, "http_responses", None),
        "diagnostics": safe,
    }


def _scrub(text: str) -> str:
    text = re.sub(r"sk-[A-Za-z0-9_-]{6,}", "[redacted]", text)
    return re.sub(r"(?i)bearer\s+\S+", "bearer [redacted]", text)
