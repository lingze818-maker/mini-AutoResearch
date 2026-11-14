"""Hostile snippets used to show that a failed candidate does not stop the run."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Probe:
    name: str
    source: str
    timeout_s: float


def containment_probes(sentinel_path: str, config_path: str) -> list[Probe]:
    sentinel = json.dumps(sentinel_path)
    config = json.dumps(config_path)
    return [
        Probe(
            "syntax_error",
            "def generate(bars)\n    return {'entry': False, 'exit': False}\n",
            15,
        ),
        Probe(
            "timeout",
            "def generate(bars):\n    while True:\n        pass\n",
            2,
        ),
        Probe(
            "unauthorized_read",
            (
                "def generate(bars):\n"
                f"    data = open({sentinel}, 'r', encoding='utf-8').read()\n"
                "    if data:\n"
                "        raise RuntimeError('read_returned_data')\n"
                "    return {'entry': False, 'exit': False}\n"
            ),
            15,
        ),
        Probe(
            "unauthorized_write",
            (
                "def generate(bars):\n"
                f"    open({sentinel}, 'w', encoding='utf-8').write('PWNED')\n"
                "    raise RuntimeError('write_returned')\n"
            ),
            15,
        ),
        Probe(
            "network",
            (
                "def generate(bars):\n"
                "    import socket\n"
                "    sock = socket.socket()\n"
                "    sock.settimeout(1)\n"
                "    sock.connect(('192.0.2.1', 80))\n"
                "    return {'entry': False, 'exit': False}\n"
            ),
            15,
        ),
        Probe(
            "subprocess",
            (
                "def generate(bars):\n"
                "    import subprocess\n"
                f"    subprocess.check_output(['/usr/bin/cat', {sentinel}], timeout=2)\n"
                "    return {'entry': False, 'exit': False}\n"
            ),
            15,
        ),
        Probe(
            "malformed_output",
            "def generate(bars):\n    return {'entry': 'long', 'exit': False}\n",
            15,
        ),
        Probe(
            "parent_memory",
            (
                "def generate(bars):\n"
                "    import os\n"
                "    with open(f'/proc/{os.getppid()}/mem', 'rb') as handle:\n"
                "        handle.read(16)\n"
                "    raise RuntimeError('read_parent_mem')\n"
            ),
            15,
        ),
        Probe(
            "repository_config",
            (
                "def generate(bars):\n"
                f"    open({config}, 'r', encoding='utf-8').read(20)\n"
                "    return {'entry': False, 'exit': False}\n"
            ),
            15,
        ),
    ]
