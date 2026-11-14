"""Offline demo of one bounded research cycle. Live mode is opt-in."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent.loop import run_bounded_experiment
from agent.model import LiveConfigError, build_provider
from agent.tools import load_universe_bars
from config_loader import ROOT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="跑一轮锁定候选、沙箱回测和写死判定，然后停止。")
    parser.add_argument("--demo", action="store_true", help="用本地干净日线跑离线夹具，或在 --live 时请求模型")
    parser.add_argument("--live", action="store_true", help="显式使用环境变量里的模型密钥；缺密钥不会退回离线夹具")
    parser.add_argument("--rerun", action="store_true", help="只替换本次输出目录，不会动 reports/baseline")
    parser.add_argument(
        "--out",
        default="reports/agent_demo",
        help="reports/ 下的输出目录，默认 reports/agent_demo",
    )
    args = parser.parse_args(argv)
    if not args.demo:
        parser.error("需要 --demo")
    try:
        provider = build_provider(args.live)
    except LiveConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(provider.banner)
    if args.live:
        print("live integration is unverified and was not used to produce the committed offline demo")
    output = Path(args.out)
    if output.is_absolute():
        print("output must be a path under reports/", file=sys.stderr)
        return 2
    try:
        document = run_bounded_experiment(
            load_universe_bars(),
            provider,
            ROOT / output,
            root=ROOT,
            rerun=args.rerun,
            run_failure_probes=True,
        )
    except (FileExistsError, ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"stop: {document['stop']['reason']}")
    print(f"provider_calls: {document['stop']['provider_calls']}")
    print(f"failure_probes: {len(document['failures'])}")
    print(f"results: {len(document['results'])}")
    print(f"wrote: {ROOT / output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
