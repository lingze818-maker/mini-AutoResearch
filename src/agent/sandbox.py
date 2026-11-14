"""Run generated strategy code in an OS sandbox and validate its signals.

The host never imports or execs the candidate. It copies the source into a
private work directory, starts ``isolate.py``, and reads back a JSON signal
list. Those booleans are later replayed by the existing vectorbt runner.

Isolation is a new user, mount, pid, network, ipc, and uts namespace. The
sandboxed root is a fresh tree: read-only ``/usr``, a fresh procfs, and three
tmpfs mounts (``/work``, ``/tmp``, ``/dev``). ``/work`` is not a bind of the
host directory. The repository, data directory, config, judge, credentials,
``/etc``, and host sockets are not mounted. Capabilities are dropped,
``no_new_privs`` is set, and a seccomp denylist blocks network, mount,
namespace, ptrace, and related calls. After pandas is imported, the worker
also denies ``execve`` and every ``clone``/``fork``/``clone3``, so the
candidate cannot start a process or thread.

This is not a full syscall allowlist. ``/usr`` remains readable. ``RLIMIT_NPROC``
is not used, and a cgroup pids limit is not available here. Writable bytes are
the sum of the tmpfs sizes. A wall-clock kill covers the process group. If
isolation or the runtime libraries are unavailable, the result is
``isolation_unavailable`` with the helper's detail, and the candidate is not
run on the host.
"""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from agent.limits import (
    AGGREGATE_WRITABLE_BYTES,
    DEFAULT_TIMEOUT_S,
    FILE_BYTES,
    MAX_BARS,
    MAX_SOURCE_CHARS,
    MAX_STDERR_BYTES,
    MAX_STDOUT_BYTES,
    MAX_TIMEOUT_S,
    MEMORY_BYTES,
)
from backtest.runner import validate_bars

_INNER = Path(__file__).resolve().parent / "inner.py"
_ISOLATE = Path(__file__).resolve().parent / "isolate.py"
_STATUSES = {
    "ok",
    "syntax_error",
    "timeout",
    "runtime_error",
    "malformed_output",
    "isolation_unavailable",
    "output_limit",
    "rejected_input",
    "resource_limit",
}


@dataclass(frozen=True, slots=True)
class SandboxOutcome:
    ok: bool
    status: str
    detail: str
    entries: tuple[bool, ...] = ()
    exits: tuple[bool, ...] = ()
    elapsed_ms: int = 0


def replay_strategy(entries: Sequence[bool], exits: Sequence[bool]):
    """Trusted signal replay. It does not read prices or import candidate code."""
    locked_entries = tuple(bool(item) if type(item) is bool else item for item in entries)
    locked_exits = tuple(bool(item) if type(item) is bool else item for item in exits)
    if any(type(item) is not bool for item in locked_entries + locked_exits):
        raise ValueError("replay signals must be booleans")

    def strategy(data: pd.DataFrame) -> pd.DataFrame:
        if len(data) != len(locked_entries) or len(locked_entries) != len(locked_exits):
            raise ValueError("replay length does not match bars")
        return pd.DataFrame(
            {
                "entries": pd.Series(locked_entries, dtype="bool"),
                "exits": pd.Series(locked_exits, dtype="bool"),
            }
        )

    return strategy


def _outcome(status: str, detail: str, *, elapsed_ms: int = 0, ok: bool = False) -> SandboxOutcome:
    text = " ".join(detail.split())[:400]
    return SandboxOutcome(ok=ok, status=status, detail=text or status, elapsed_ms=elapsed_ms)


def _reject_input(source: str, bars: pd.DataFrame, timeout_s: float) -> SandboxOutcome | None:
    if not isinstance(source, str) or "\x00" in source:
        return _outcome("rejected_input", "source must be text without NUL bytes")
    if not source.strip():
        return _outcome("rejected_input", "source is empty")
    if len(source) > MAX_SOURCE_CHARS:
        return _outcome("rejected_input", "source exceeds the size limit")
    if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool):
        return _outcome("rejected_input", "timeout must be a positive number")
    if not math.isfinite(float(timeout_s)) or float(timeout_s) <= 0 or float(timeout_s) > MAX_TIMEOUT_S:
        return _outcome("rejected_input", "timeout is outside the allowed range")
    try:
        frame = validate_bars(bars)
    except ValueError as exc:
        return _outcome("rejected_input", str(exc))
    if len(frame) > MAX_BARS:
        return _outcome("rejected_input", "too many bars")
    values = frame[["open", "high", "low", "close", "volume"]]
    finite = values.map(lambda item: isinstance(item, (int, float)) and math.isfinite(float(item)))
    if not bool(finite.all().all()):
        return _outcome("rejected_input", "prices and volume must be finite")
    return None


def _payload(bars: pd.DataFrame) -> bytes:
    frame = validate_bars(bars)
    rows = []
    for record in frame.itertuples(index=False):
        rows.append(
            {
                "date": pd.Timestamp(record.date).strftime("%Y-%m-%d"),
                "open": float(record.open),
                "high": float(record.high),
                "low": float(record.low),
                "close": float(record.close),
                "volume": float(record.volume),
            }
        )
    return json.dumps({"bars": rows}, separators=(",", ":")).encode("utf-8")


def _safe_rmtree(path: str) -> None:
    root = os.path.realpath(tempfile.gettempdir())
    target = os.path.realpath(path)
    if target == root or not target.startswith(root + os.sep):
        raise RuntimeError("refusing to remove a path outside the temp directory")
    shutil.rmtree(target, ignore_errors=True)


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def _read_capped(stream, limit: int, proc: subprocess.Popen) -> tuple[bytes, bool]:
    chunks = []
    total = 0
    overflow = False
    while True:
        block = stream.read(65536)
        if not block:
            break
        total += len(block)
        if total > limit:
            overflow = True
            _terminate(proc)
            break
        chunks.append(block)
    return b"".join(chunks), overflow


def _spawn(command: list[str], payload: bytes, timeout_s: float) -> tuple[int, bytes, bytes, bool, bool]:
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"PATH": "/usr/bin", "LANG": "C.UTF-8"},
        start_new_session=True,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None

    def write_input() -> None:
        try:
            proc.stdin.write(payload)
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    boxes: dict[str, tuple[bytes, bool]] = {}

    def read_out() -> None:
        boxes["out"] = _read_capped(proc.stdout, MAX_STDOUT_BYTES, proc)

    def read_err() -> None:
        boxes["err"] = _read_capped(proc.stderr, MAX_STDERR_BYTES, proc)

    threads = [
        threading.Thread(target=write_input),
        threading.Thread(target=read_out),
        threading.Thread(target=read_err),
    ]
    for thread in threads:
        thread.start()
    timed_out = False
    deadline = time.monotonic() + timeout_s
    while proc.poll() is None:
        if time.monotonic() >= deadline:
            timed_out = True
            _terminate(proc)
            break
        time.sleep(0.02)
    if timed_out:
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            _terminate(proc)
    for thread in threads:
        thread.join(timeout=5)
    stdout, out_overflow = boxes.get("out", (b"", False))
    stderr, err_overflow = boxes.get("err", (b"", False))
    return proc.returncode if proc.returncode is not None else -9, stdout, stderr, timed_out, out_overflow or err_overflow


def _bool_list(value: object, length: int) -> tuple[bool, ...] | None:
    if not isinstance(value, list) or len(value) != length:
        return None
    items = []
    for item in value:
        if type(item) is not bool:
            return None
        items.append(item)
    return tuple(items)


def _json_object(raw: bytes) -> dict | None:
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _interpret(returncode: int, stdout: bytes, stderr: bytes, timed_out: bool, overflow: bool, length: int, elapsed_ms: int) -> SandboxOutcome:
    if timed_out:
        return _outcome("timeout", "wall clock limit exceeded", elapsed_ms=elapsed_ms)
    if overflow:
        return _outcome("output_limit", "sandbox output exceeded the byte limit", elapsed_ms=elapsed_ms)
    payload = _json_object(stdout)
    if payload is not None:
        status = payload.get("status")
        detail = payload.get("detail") if isinstance(payload.get("detail"), str) else ""
        if payload.get("ok") is True and status == "ok" and returncode == 0:
            entries = _bool_list(payload.get("entries"), length)
            exits = _bool_list(payload.get("exits"), length)
            if entries is None or exits is None:
                return _outcome("malformed_output", "signals must be boolean lists aligned to the bars", elapsed_ms=elapsed_ms)
            return SandboxOutcome(True, "ok", "", entries, exits, elapsed_ms)
        if isinstance(status, str) and status in _STATUSES and status != "ok":
            # A non-zero exit still carries the helper's structured reason.
            # Exit 96 used to be collapsed to "sandbox setup failed".
            return _outcome(status, detail or status, elapsed_ms=elapsed_ms)
    if returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        if returncode < 0:
            return _outcome("resource_limit", detail or f"sandbox process died from signal {-returncode}", elapsed_ms=elapsed_ms)
        snippet = " ".join(detail.split())[:300]
        return _outcome("isolation_unavailable", snippet or "sandbox setup failed", elapsed_ms=elapsed_ms)
    return _outcome("malformed_output", "sandbox status is missing", elapsed_ms=elapsed_ms)


def run_candidate(
    source: str,
    bars: pd.DataFrame,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    _runtime_failure: str | None = None,
) -> SandboxOutcome:
    """Execute ``source`` inside the sandbox. Never falls back to the host."""
    rejected = _reject_input(source, bars, timeout_s)
    if rejected is not None:
        return rejected
    if sys.platform != "linux":
        return _outcome("isolation_unavailable", "sandbox requires Linux namespaces")
    if platform.machine() != "x86_64":
        return _outcome("isolation_unavailable", "seccomp policy is implemented for x86_64 only")
    if not hasattr(os, "unshare"):
        return _outcome("isolation_unavailable", "os.unshare is not available")
    if not _ISOLATE.is_file() or not _INNER.is_file():
        return _outcome("isolation_unavailable", "sandbox helper files are missing")

    base = tempfile.mkdtemp(prefix="mini-ar-sandbox-")
    started = time.monotonic()
    try:
        work = os.path.join(base, "work")
        newroot = os.path.join(base, "newroot")
        os.makedirs(work, mode=0o700)
        os.makedirs(newroot, mode=0o700)
        shutil.copyfile(_INNER, os.path.join(work, "inner.py"))
        Path(os.path.join(work, "candidate.py")).write_text(source, encoding="utf-8")
        if _runtime_failure is not None:
            note = " ".join(str(_runtime_failure).split())[:200]
            if not note:
                return _outcome("rejected_input", "runtime failure note is empty")
            Path(os.path.join(work, "runtime_failure.txt")).write_text(note + "\n", encoding="utf-8")
        cpu_seconds = max(1, min(180, int(math.ceil(float(timeout_s))) + 5))
        command = [sys.executable, str(_ISOLATE), work, newroot, str(cpu_seconds)]
        try:
            returncode, stdout, stderr, timed_out, overflow = _spawn(command, _payload(bars), float(timeout_s))
        except Exception as exc:
            return _outcome("isolation_unavailable", f"sandbox did not start: {type(exc).__name__}: {exc}")
        elapsed_ms = int((time.monotonic() - started) * 1000)
        return _interpret(returncode, stdout, stderr, timed_out, overflow, len(bars), elapsed_ms)
    finally:
        _safe_rmtree(base)


# Re-export the memory constant so operators can see the enforced ceiling.
SANDBOX_MEMORY_BYTES = MEMORY_BYTES
SANDBOX_FILE_BYTES = FILE_BYTES
SANDBOX_AGGREGATE_WRITABLE_BYTES = AGGREGATE_WRITABLE_BYTES
