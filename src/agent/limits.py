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
# Candidate writes inside the sandbox, including the private work directory.
FILE_BYTES = 2 * 1024 * 1024
OPEN_FILES = 256
MAX_PROCESSES = 64

# Live model client. The offline demo does not spend these.
MAX_MODEL_CALLS = 2
MAX_MODEL_TOKENS = 2048
MAX_MODEL_RETRIES = 1
MODEL_TIMEOUT_S = 30.0

# Assumed vendor host. The request path is not a verified live contract.
DEFAULT_BASE_URL = "https://api.deepseek.com"

OFFLINE_BANNER = "offline simulated proposal, not a real model research run"

HOLDOUT_NOTE = (
    "The out-of-sample window that starts the day after in_sample_end is an "
    "already-inspected historical holdout: the frozen 20/60 moving-average baseline "
    "was already judged on these dates. It is not a genuinely untouched validation set. "
    "Pass or reject here is not an unbiased discovery claim, and pass does not mean "
    "the rule is profitable or ready for live trading."
)

STOP_REASON = "oos_verdict_disclosed"
