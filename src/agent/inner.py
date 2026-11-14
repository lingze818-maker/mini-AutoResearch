"""Sandbox helper copied into the isolated root. It refuses to run on the host.

The supervisor feeds one bar at a time to a worker process. The worker loads the
candidate and decides the current bar from that prefix only. This file must not
import the repository.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys

ALLOWED_ROOT = {"bin", "dev", "lib", "lib64", "proc", "tmp", "usr", "work"}
_MAX_LINE = 8000


def _emit(payload: dict, code: int) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()
    raise SystemExit(code)


def _fail(status: str, detail: str, code: int) -> None:
    text = " ".join(str(detail).split())[:400]
    _emit({"ok": False, "status": status, "detail": text or status}, code)


def _status_value(name: str) -> str:
    with open("/proc/self/status", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(name + ":"):
                return line.split(":", 1)[1].strip()
    return ""


def _assert_sandbox() -> None:
    try:
        entries = set(os.listdir("/"))
    except OSError as exc:
        _fail("isolation_unavailable", f"cannot list sandbox root: {exc}", 96)
    extra = entries - ALLOWED_ROOT
    if extra or not {"usr", "work", "proc"} <= entries:
        _fail("isolation_unavailable", f"unexpected root entries: {sorted(entries)}", 96)
    for path in ("/workspace", "/oldroot", "/home", "/etc"):
        if os.path.exists(path):
            _fail("isolation_unavailable", f"host path is visible: {path}", 96)
    usr_readonly = False
    try:
        mounts = open("/proc/mounts", encoding="utf-8", errors="replace").read().splitlines()
    except OSError as exc:
        _fail("isolation_unavailable", f"cannot read mounts: {exc}", 96)
    for line in mounts:
        parts = line.split()
        if len(parts) >= 4 and parts[1] == "/usr":
            options = set(parts[3].split(","))
            if "rw" in options:
                _fail("isolation_unavailable", "usr is mounted read-write", 96)
            if "ro" in options:
                usr_readonly = True
    if not usr_readonly:
        _fail("isolation_unavailable", "usr is not a read-only mount", 96)
    if _status_value("Seccomp") != "2":
        _fail("isolation_unavailable", "seccomp filter is not active", 96)
    try:
        import socket

        socket.socket()
    except OSError:
        return
    _fail("isolation_unavailable", "socket() was not blocked", 96)


def _nondumpable() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(4, 0, 0, 0, 0) != 0:
        _fail("isolation_unavailable", "PR_SET_DUMPABLE failed", 96)


def _block_exec() -> None:
    """Stack a filter that refuses execve. The worker has already started."""
    import struct

    pr_set_seccomp = 22
    seccomp_mode_filter = 2
    audit_arch = 0xC000003E
    ret_kill = 0x80000000
    ret_errno = 0x00050000
    ret_allow = 0x7FFF0000
    eperm = 1
    bpf_ld = 0x00
    bpf_w = 0x00
    bpf_abs = 0x20
    bpf_jmp = 0x05
    bpf_jeq = 0x10
    bpf_k = 0x00
    bpf_ret = 0x06

    def stmt(code: int, k: int) -> bytes:
        return struct.pack("<HBBI", code, 0, 0, k & 0xFFFFFFFF)

    def jump(k: int, jt: int, jf: int) -> bytes:
        return struct.pack("<HBBI", bpf_jmp | bpf_jeq | bpf_k, jt, jf, k & 0xFFFFFFFF)

    # execve=59, execveat=322
    blocked = (59, 322)
    blob = b""
    blob += stmt(bpf_ld | bpf_w | bpf_abs, 4)
    blob += jump(audit_arch, 1, 0)
    blob += stmt(bpf_ret | bpf_k, ret_kill)
    blob += stmt(bpf_ld | bpf_w | bpf_abs, 0)
    for number in blocked:
        blob += jump(number, 0, 1)
        blob += stmt(bpf_ret | bpf_k, ret_errno | eperm)
    blob += stmt(bpf_ret | bpf_k, ret_allow)

    class SockFilter(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_uint16),
            ("jt", ctypes.c_uint8),
            ("jf", ctypes.c_uint8),
            ("k", ctypes.c_uint32),
        ]

    class SockFprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(SockFilter))]

    count = len(blob) // 8
    filters = (SockFilter * count).from_buffer_copy(blob)
    program = SockFprog(count, ctypes.cast(filters, ctypes.POINTER(SockFilter)))
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(pr_set_seccomp, seccomp_mode_filter, ctypes.byref(program)) != 0:
        error = ctypes.get_errno()
        _fail("isolation_unavailable", f"could not block exec: {os.strerror(error)}", 96)


def _as_bool(value: object, name: str) -> bool:
    if type(value) is bool:
        return value
    raise TypeError(f"{name} must be a boolean")


def _decision(fn, frame) -> dict:
    try:
        raw = fn(frame)
    except Exception as exc:
        return {"ok": False, "status": "runtime_error", "detail": _short(exc)}
    if not isinstance(raw, dict):
        return {"ok": False, "status": "malformed_output", "detail": "generate() must return an object"}
    try:
        entry = _as_bool(raw.get("entry"), "entry")
        exit_ = _as_bool(raw.get("exit"), "exit")
    except TypeError as exc:
        return {"ok": False, "status": "malformed_output", "detail": str(exc)}
    return {"ok": True, "entry": entry, "exit": exit_}


def _short(exc: BaseException) -> str:
    return " ".join(f"{type(exc).__name__}: {exc}".split())[:400]


class _Lines:
    def __init__(self, stream) -> None:
        self._stream = stream
        self._buffer = b""

    def readline(self) -> bytes:
        while b"\n" not in self._buffer:
            chunk = self._stream.read(4096)
            if not chunk:
                break
            self._buffer += chunk
            if len(self._buffer) > _MAX_LINE * 8:
                raise ValueError("sandbox line is too long")
        if b"\n" not in self._buffer:
            raise EOFError("worker closed the pipe")
        line, self._buffer = self._buffer.split(b"\n", 1)
        if len(line) > _MAX_LINE:
            raise ValueError("sandbox line is too long")
        return line


def worker() -> None:
    _assert_sandbox()
    _nondumpable()
    try:
        import pandas as pd
    except Exception as exc:
        _emit({"ok": False, "status": "isolation_unavailable", "detail": _short(exc)}, 96)
    _block_exec()
    import importlib.util

    try:
        spec = importlib.util.spec_from_file_location("candidate", "/work/candidate.py")
        if spec is None or spec.loader is None:
            _fail("runtime_error", "candidate could not be loaded", 0)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except SyntaxError as exc:
        _emit(
            {
                "ok": False,
                "status": "syntax_error",
                "detail": _short(exc),
            },
            0,
        )
    except Exception as exc:
        _emit({"ok": False, "status": "runtime_error", "detail": _short(exc)}, 0)
    generate = getattr(module, "generate", None)
    if not callable(generate):
        _fail("runtime_error", "candidate must define generate(bars)", 0)
    sys.stdout.write(json.dumps({"ok": True, "status": "ready"}) + "\n")
    sys.stdout.flush()
    rows: list[dict] = []
    while True:
        line = sys.stdin.readline()
        if not line:
            return
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            _fail("malformed_output", "supervisor sent invalid json", 0)
        if message.get("op") == "end":
            return
        row = message.get("row")
        if not isinstance(row, dict):
            _fail("malformed_output", "bar row is missing", 0)
        rows.append(row)
        frame = pd.DataFrame.from_records(rows)
        result = _decision(generate, frame)
        sys.stdout.write(json.dumps(result, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        if not result.get("ok"):
            return


def supervisor() -> None:
    _assert_sandbox()
    _nondumpable()
    to_read, to_write = os.pipe()
    from_read, from_write = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(to_write)
        os.close(from_read)
        os.dup2(to_read, 0)
        os.dup2(from_write, 1)
        os.close(to_read)
        os.close(from_write)
        env = dict(os.environ)
        os.execve(
            "/usr/bin/python3",
            ["/usr/bin/python3", "-I", "-B", "/work/inner.py", "worker"],
            env,
        )
        os._exit(127)
    os.close(to_read)
    os.close(from_write)
    raw = sys.stdin.buffer.read(MAX_STDIN)
    if len(raw) >= MAX_STDIN:
        _fail("rejected_input", "bar payload is too large", 0)
    try:
        payload = json.loads(raw.decode("utf-8"))
        bars = payload["bars"]
    except (UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        _fail("rejected_input", _short(exc), 0)
    if not isinstance(bars, list) or not bars:
        _fail("rejected_input", "bars are empty", 0)
    reader = _Lines(os.fdopen(from_read, "rb", buffering=0))
    writer = os.fdopen(to_write, "wb", buffering=0)
    try:
        ready = json.loads(reader.readline().decode("utf-8"))
    except (EOFError, ValueError, json.JSONDecodeError) as exc:
        _fail("isolation_unavailable", _short(exc), 96)
    if ready.get("status") != "ready":
        status = ready.get("status") if isinstance(ready.get("status"), str) else "runtime_error"
        detail = ready.get("detail") if isinstance(ready.get("detail"), str) else "worker did not start"
        code = 96 if status == "isolation_unavailable" else 0
        _fail(status, detail, code)
    entries: list[bool] = []
    exits: list[bool] = []
    for row in bars:
        writer.write(json.dumps({"op": "bar", "row": row}, separators=(",", ":")).encode() + b"\n")
        try:
            message = json.loads(reader.readline().decode("utf-8"))
        except (EOFError, ValueError, json.JSONDecodeError) as exc:
            _fail("runtime_error", _short(exc), 0)
        if not message.get("ok"):
            status = message.get("status") if isinstance(message.get("status"), str) else "runtime_error"
            detail = message.get("detail") if isinstance(message.get("detail"), str) else status
            code = 96 if status == "isolation_unavailable" else 0
            _fail(status, detail, code)
        try:
            entries.append(_as_bool(message.get("entry"), "entry"))
            exits.append(_as_bool(message.get("exit"), "exit"))
        except TypeError as exc:
            _fail("malformed_output", str(exc), 0)
    writer.write(b'{"op":"end"}\n')
    if len(entries) != len(bars):
        _fail("malformed_output", "signal length does not match bars", 0)
    _emit({"ok": True, "status": "ok", "entries": entries, "exits": exits}, 0)


MAX_STDIN = 2_000_000


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker()
        return
    supervisor()


if __name__ == "__main__":
    main()
