"""Fixed bounds for the sandbox and the model client. Not search parameters."""

from __future__ import annotations

# Wall clock for one sandboxed candidate. The CPU rlimit sits a few seconds above this.
DEFAULT_TIMEOUT_S = 60.0
MAX_TIMEOUT_S = 120.0

MAX_SOURCE_CHARS = 12_000
MAX_BARS = 5_000
MAX_STDOUT_BYTES = 1_000_000
MAX_STDERR_BYTES = 64_000

# Address-space rlimit for the isolated interpreter. pandas/numpy stay well under this.
MEMORY_BYTES = 1024 * 1024 * 1024
# Per-file rlimit only. It is not the aggregate disk cap: two files under this
# size can still fill an unbounded host bind. Writable storage is the tmpfs
# sizes below, and every writable mount is a tmpfs.
FILE_BYTES = 2 * 1024 * 1024
WORK_TMPFS_BYTES = 2 * 1024 * 1024
TMP_TMPFS_BYTES = 4 * 1024 * 1024
DEV_TMPFS_BYTES = 1 * 1024 * 1024
AGGREGATE_WRITABLE_BYTES = WORK_TMPFS_BYTES + TMP_TMPFS_BYTES + DEV_TMPFS_BYTES
OPEN_FILES = 256
# Process and thread creation is denied by a seccomp filter in the worker
# after pandas has been imported (clone, fork, vfork, clone3). RLIMIT_NPROC is
# not used: it counts every thread of the host uid, so a small value is not an
# effective sandbox bound. The cgroup pids controller is not writable here.

# Live model client. The offline demo does not spend these.
MAX_MODEL_CALLS = 2
MAX_MODEL_TOKENS = 2048
MAX_MODEL_RETRIES = 1
MODEL_TIMEOUT_S = 30.0

# Only this origin is accepted. DEEPSEEK_BASE_URL cannot point anywhere else.
DEFAULT_BASE_URL = "https://api.deepseek.com"
OFFICIAL_API_HOST = "api.deepseek.com"

OFFLINE_BANNER = "offline simulated proposal, not a real model research run"

HOLDOUT_NOTE = (
    "The out-of-sample window that starts the day after in_sample_end is an "
    "already-inspected historical holdout: the frozen 20/60 moving-average baseline "
    "was already judged on these dates. It is not a genuinely untouched validation set. "
    "Pass or reject here is not an unbiased discovery claim, and pass does not mean "
    "the rule is profitable or ready for live trading."
)

STOP_REASON = "oos_verdict_disclosed"
