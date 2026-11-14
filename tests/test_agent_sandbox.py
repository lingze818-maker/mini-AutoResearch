from __future__ import annotations

import socket
import subprocess
import sys
import threading
from pathlib import Path

import json

from agent.limits import (
    AGGREGATE_WRITABLE_BYTES,
    DEV_TMPFS_BYTES,
    FILE_BYTES,
    TMP_TMPFS_BYTES,
    WORK_TMPFS_BYTES,
)
from agent.sandbox import _interpret, replay_strategy, run_candidate
import agent.isolate as isolate_module
from backtest.runner import run_backtest
from tests.conftest import ROOT, bars_from_closes

_SECRET = "SENTINEL-CONTENTS-9f3a"


def _sentinel(tmp_path: Path) -> Path:
    directory = tmp_path / "host-only"
    directory.mkdir()
    path = directory / "secret.txt"
    path.write_text(_SECRET + "\n", encoding="utf-8")
    return path


def _good_source() -> str:
    return (
        "def generate(bars):\n"
        "    close = bars['close'].astype(float)\n"
        "    if len(close) < 2:\n"
        "        return {'entry': False, 'exit': False}\n"
        "    up = bool(float(close.iloc[-1]) > float(close.iloc[-2]))\n"
        "    return {'entry': up, 'exit': (not up)}\n"
    )


def test_normal_strategy_returns_aligned_booleans():
    bars = bars_from_closes([100, 101, 99, 103, 104])
    outcome = run_candidate(_good_source(), bars, timeout_s=20)
    assert outcome.ok
    assert outcome.status == "ok"
    assert outcome.entries == (False, True, False, True, True)
    assert len(outcome.exits) == len(bars)


def test_future_prices_do_not_change_past_signals():
    base = bars_from_closes([10, 11, 12, 13, 14, 15, 16, 17, 18])
    changed = base.copy()
    changed.loc[5:, "close"] = [90, 80, 70, 60]
    changed["high"] = changed[["open", "close"]].max(axis=1) * 1.001
    changed["low"] = changed[["open", "close"]].min(axis=1) * 0.999
    first = run_candidate(_good_source(), base, timeout_s=20)
    second = run_candidate(_good_source(), changed, timeout_s=20)
    assert first.ok and second.ok
    assert first.entries[:5] == second.entries[:5]
    assert first.entries != second.entries


def test_shift_and_memory_cannot_see_future_bars():
    source = (
        "def generate(bars):\n"
        "    import gc\n"
        "    import pandas as pd\n"
        "    longest = 0\n"
        "    for obj in gc.get_objects():\n"
        "        if isinstance(obj, pd.DataFrame):\n"
        "            longest = max(longest, len(obj))\n"
        "    if longest > len(bars):\n"
        "        raise RuntimeError('saw future')\n"
        "    if pd.notna(bars['close'].shift(-1).iloc[-1]):\n"
        "        raise RuntimeError('peeked')\n"
        "    return {'entry': False, 'exit': False}\n"
    )
    outcome = run_candidate(source, bars_from_closes([100, 101, 102, 103, 104, 105]), timeout_s=20)
    assert outcome.ok, outcome.detail
    assert "saw future" not in outcome.detail
    assert "peeked" not in outcome.detail


def test_syntax_error_is_a_structured_failure():
    outcome = run_candidate("def generate(bars)\n    return {'entry': False, 'exit': False}\n", bars_from_closes([100, 101, 102, 103]), timeout_s=20)
    assert not outcome.ok
    assert outcome.status == "syntax_error"


def test_timeout_is_a_structured_failure():
    source = "def generate(bars):\n    while True:\n        pass\n"
    outcome = run_candidate(source, bars_from_closes([100, 101, 102, 103]), timeout_s=2)
    assert not outcome.ok
    assert outcome.status == "timeout"
    assert outcome.elapsed_ms < 10_000


def test_unauthorized_read_and_write_leave_the_sentinel_unreadable(tmp_path, monkeypatch):
    sentinel = _sentinel(tmp_path)
    monkeypatch.setenv("SENTINEL_SECRET", "super-secret-value")
    before = sentinel.read_bytes()
    read_source = (
        "def generate(bars):\n"
        f"    data = open({str(sentinel)!r}, encoding='utf-8').read()\n"
        "    raise RuntimeError(data)\n"
    )
    read = run_candidate(read_source, bars_from_closes([100, 101, 102, 103]), timeout_s=20)
    write_source = (
        "def generate(bars):\n"
        f"    open({str(sentinel)!r}, 'w', encoding='utf-8').write('PWNED')\n"
        "    raise RuntimeError('write_returned')\n"
    )
    written = run_candidate(write_source, bars_from_closes([100, 101, 102, 103]), timeout_s=20)
    assert read.status == "runtime_error"
    assert written.status == "runtime_error"
    assert sentinel.read_bytes() == before
    assert _SECRET not in read.detail
    assert "write_returned" not in written.detail
    assert "PWNED" not in sentinel.read_text(encoding="utf-8")


def test_network_and_subprocess_cannot_escape(tmp_path):
    sentinel = _sentinel(tmp_path)
    hits: list[bool] = []
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    server.settimeout(0.5)
    port = server.getsockname()[1]

    def accept() -> None:
        try:
            connection, _address = server.accept()
            hits.append(True)
            connection.close()
        except OSError:
            hits.append(False)

    listener = threading.Thread(target=accept, daemon=True)
    listener.start()
    network = (
        "def generate(bars):\n"
        "    import socket\n"
        "    sock = socket.socket()\n"
        "    sock.settimeout(1)\n"
        f"    sock.connect(('127.0.0.1', {port}))\n"
        "    return {'entry': False, 'exit': False}\n"
    )
    net = run_candidate(network, bars_from_closes([100, 101, 102, 103]), timeout_s=20)
    process = (
        "def generate(bars):\n"
        "    import subprocess\n"
        f"    subprocess.check_output(['/usr/bin/cat', {str(sentinel)!r}], timeout=2)\n"
        "    return {'entry': False, 'exit': False}\n"
    )
    spawned = run_candidate(process, bars_from_closes([100, 101, 102, 103]), timeout_s=20)
    listener.join(timeout=2)
    server.close()
    assert net.status == "runtime_error"
    assert spawned.status == "runtime_error"
    assert hits == [False]
    assert sentinel.read_text(encoding="utf-8") == _SECRET + "\n"
    assert _SECRET not in spawned.detail


def test_malformed_output_is_rejected_and_the_next_candidate_still_runs():
    bars = bars_from_closes([100, 101, 102, 103])
    bad = run_candidate("def generate(bars):\n    return {'entry': 'long', 'exit': False}\n", bars, timeout_s=20)
    good = run_candidate(_good_source(), bars, timeout_s=20)
    assert bad.status == "malformed_output"
    assert good.ok


def test_sandbox_does_not_inherit_the_host_environment(monkeypatch):
    monkeypatch.setenv("SENTINEL_SECRET", "super-secret-value")
    source = (
        "def generate(bars):\n"
        "    import os\n"
        "    if os.environ.get('SENTINEL_SECRET'):\n"
        "        raise RuntimeError('saw secret')\n"
        "    return {'entry': False, 'exit': False}\n"
    )
    outcome = run_candidate(source, bars_from_closes([100, 101, 102, 103]), timeout_s=20)
    assert outcome.ok, outcome.detail
    assert "super-secret-value" not in outcome.detail


def test_host_paths_and_usr_are_not_writable(tmp_path):
    marker = tmp_path / "host-marker.txt"
    usr_probe = Path("/usr/local/mini_ar_host_write_probe")
    if usr_probe.exists():
        usr_probe.unlink()
    source = (
        f"open({str(marker)!r}, 'w', encoding='utf-8').write('ran')\n"
        "def generate(bars):\n"
        "    open('/usr/local/mini_ar_host_write_probe', 'w', encoding='utf-8').write('pwned')\n"
        "    open('/workspace/config.yaml', encoding='utf-8').read(10)\n"
        "    return {'entry': False, 'exit': False}\n"
    )
    outcome = run_candidate(source, bars_from_closes([100, 101, 102, 103]), timeout_s=20)
    assert not outcome.ok
    assert not marker.exists()
    assert not usr_probe.exists()
    assert "pwned" not in outcome.detail


def test_parent_memory_is_not_readable():
    source = (
        "def generate(bars):\n"
        "    import os\n"
        "    with open(f'/proc/{os.getppid()}/mem', 'rb') as handle:\n"
        "        handle.read(16)\n"
        "    raise RuntimeError('read_parent_mem')\n"
    )
    outcome = run_candidate(source, bars_from_closes([100, 101, 102, 103]), timeout_s=20)
    assert not outcome.ok
    assert "read_parent_mem" not in outcome.detail


def test_spawn_failure_does_not_execute_on_the_host(tmp_path, monkeypatch):
    marker = tmp_path / "marker.txt"
    source = (
        f"open({str(marker)!r}, 'w', encoding='utf-8').write('ran')\n"
        "def generate(bars):\n"
        "    return {'entry': False, 'exit': False}\n"
    )

    def boom(*_args, **_kwargs):
        raise OSError("unshare failed: EPERM")

    monkeypatch.setattr("agent.sandbox._spawn", boom)
    outcome = run_candidate(source, bars_from_closes([100, 101, 102, 103]), timeout_s=20)
    assert outcome.status == "isolation_unavailable"
    assert not marker.exists()


def test_non_x86_64_fails_closed_without_spawning(monkeypatch):
    called = {"spawn": 0}

    def boom(*_args, **_kwargs):
        called["spawn"] += 1
        raise AssertionError("spawned")

    monkeypatch.setattr("agent.sandbox._spawn", boom)
    monkeypatch.setattr("agent.sandbox.platform.machine", lambda: "arm64")
    outcome = run_candidate(_good_source(), bars_from_closes([100, 101, 102, 103]), timeout_s=20)
    assert outcome.status == "isolation_unavailable"
    assert "x86_64" in outcome.detail
    assert called["spawn"] == 0


def test_helper_refuses_to_run_outside_the_sandbox():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "src" / "agent" / "inner.py")],
        input=b'{"bars": []}',
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 96
    assert b"isolation_unavailable" in completed.stdout


def test_replay_uses_the_existing_runner_and_next_bar_fill(monkeypatch):
    import vectorbt as vbt

    bars = bars_from_closes([100, 100, 100, 110, 110])
    outcome = run_candidate(
        "def generate(bars):\n    return {'entry': len(bars) == 2, 'exit': False}\n",
        bars,
        timeout_s=20,
    )
    assert outcome.ok
    seen: dict[str, object] = {}
    original = vbt.Portfolio.from_signals

    def wrapped(close, entries, exits, *args, **kwargs):
        seen["fees"] = kwargs.get("fees")
        seen["entries"] = list(entries)
        return original(close, entries, exits, *args, **kwargs)

    monkeypatch.setattr(vbt.Portfolio, "from_signals", wrapped)
    result = run_backtest(bars, replay_strategy(outcome.entries, outcome.exits))
    assert seen["fees"] == 0.0
    assert seen["entries"][0] is False or seen["entries"][0] == False
    assert result.curve["position"].tolist()[2:] == [1, 1, 1]


def test_tmpfs_ceilings_match_and_cover_the_per_file_rlimit():
    assert isolate_module._WORK_BYTES == WORK_TMPFS_BYTES == FILE_BYTES
    assert isolate_module._TMP_BYTES == TMP_TMPFS_BYTES
    assert isolate_module._DEV_BYTES == DEV_TMPFS_BYTES
    assert WORK_TMPFS_BYTES + TMP_TMPFS_BYTES + DEV_TMPFS_BYTES == AGGREGATE_WRITABLE_BYTES
    assert "RLIMIT_NPROC" not in isolate_module._set_limits.__code__.co_names


def test_work_is_a_capped_tmpfs_and_two_files_cannot_fill_the_host_disk():
    source = (
        "def generate(bars):\n"
        "    mounts = open('/proc/mounts', encoding='utf-8').read().splitlines()\n"
        "    work = [line for line in mounts if len(line.split()) >= 2 and line.split()[1] == '/work']\n"
        "    if not work or not work[0].startswith('tmpfs '):\n"
        "        raise RuntimeError('work_not_tmpfs')\n"
        "    if 'size=2048k' not in work[0].split()[3]:\n"
        "        raise RuntimeError('work_size_missing')\n"
        "    blob = b'x' * (1536 * 1024)\n"
        "    with open('/work/a.bin', 'wb') as handle:\n"
        "        handle.write(blob)\n"
        "    try:\n"
        "        with open('/work/b.bin', 'wb') as handle:\n"
        "            handle.write(blob)\n"
        "    except OSError as exc:\n"
        "        if exc.errno != 28:\n"
        "            raise\n"
        "        raise RuntimeError('aggregate_capped')\n"
        "    raise RuntimeError('second_write_succeeded')\n"
    )
    outcome = run_candidate(source, bars_from_closes([100, 101, 102, 103]), timeout_s=20)
    assert outcome.status == "runtime_error"
    assert "aggregate_capped" in outcome.detail
    assert "second_write_succeeded" not in outcome.detail
    assert "work_not_tmpfs" not in outcome.detail


def test_candidate_cannot_fork_or_start_a_thread_after_pandas_import():
    source = (
        "def generate(bars):\n"
        "    import os\n"
        "    import threading\n"
        "    close = bars['close'].astype(float)\n"
        "    if len(close) > 1:\n"
        "        float(close.pct_change().iloc[1:].std())\n"
        "    try:\n"
        "        pid = os.fork()\n"
        "    except OSError as exc:\n"
        "        if exc.errno != 1:\n"
        "            raise\n"
        "    else:\n"
        "        if pid == 0:\n"
        "            os._exit(0)\n"
        "        raise RuntimeError('fork_succeeded')\n"
        "    try:\n"
        "        threading.Thread(target=lambda: None).start()\n"
        "    except RuntimeError as exc:\n"
        "        if 'thread' not in str(exc):\n"
        "            raise\n"
        "    else:\n"
        "        raise RuntimeError('thread_succeeded')\n"
        "    return {'entry': False, 'exit': False}\n"
    )
    outcome = run_candidate(source, bars_from_closes([100, 101, 102, 103]), timeout_s=20)
    assert outcome.ok, outcome.detail
    assert "fork_succeeded" not in outcome.detail
    assert "thread_succeeded" not in outcome.detail


def test_exit_96_keeps_the_missing_library_detail():
    raw = json.dumps(
        {
            "ok": False,
            "status": "isolation_unavailable",
            "detail": "missing runtime library: libblas.so.3: cannot open shared object file",
        }
    ).encode("utf-8")
    outcome = _interpret(96, raw, b"namespace setup failed: ignored", False, False, 4, 12)
    assert outcome.status == "isolation_unavailable"
    assert "libblas.so.3" in outcome.detail
    assert "sandbox setup failed" not in outcome.detail
    assert "ignored" not in outcome.detail


def test_injected_runtime_failure_is_reported_from_the_real_sandbox():
    outcome = run_candidate(
        "def generate(bars):\n    return {'entry': False, 'exit': False}\n",
        bars_from_closes([100, 101, 102, 103]),
        timeout_s=20,
        _runtime_failure="libblas.so.3: cannot open shared object file: No such file or directory",
    )
    assert outcome.status == "isolation_unavailable"
    assert "missing runtime library" in outcome.detail
    assert "libblas.so.3" in outcome.detail
    assert outcome.detail != "sandbox setup failed"
    assert outcome.entries == ()


def test_distro_libblas_symlink_fails_without_mounting_etc():
    source = (
        "def generate(bars):\n"
        "    import ctypes\n"
        "    ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libblas.so.3')\n"
        "    return {'entry': False, 'exit': False}\n"
    )
    outcome = run_candidate(source, bars_from_closes([100, 101, 102, 103]), timeout_s=20)
    assert not outcome.ok
    assert outcome.detail != "sandbox setup failed"
    assert "libblas" in outcome.detail.lower() or "No such file" in outcome.detail


def test_environment_keys_are_not_passed_to_the_child(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-should-not-leak")
    source = (
        "def generate(bars):\n"
        "    import os\n"
        "    blob = '\\n'.join(os.environ.values())\n"
        "    if 'sk-test-should-not-leak' in blob:\n"
        "        raise RuntimeError('leaked key')\n"
        "    return {'entry': False, 'exit': False}\n"
    )
    outcome = run_candidate(source, bars_from_closes([100, 101, 102]), timeout_s=20)
    assert outcome.ok, outcome.detail
    assert "sk-test-should-not-leak" not in outcome.detail
