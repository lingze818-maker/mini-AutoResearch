"""Command line for fetch and clean. Paths and symbols come from config.yaml."""

from __future__ import annotations

import argparse
import sys

from .clean import clean_all
from .fetch import fetch_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="拉取并清洗 config.yaml 里的 akshare 日线。")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="用 akshare fund_etf_hist_em 拉前复权日线到 data/raw")
    sub.add_parser("clean", help="写成 date, open, high, low, close, volume 到 data/clean")
    args = parser.parse_args(argv)
    errors = fetch_all() if args.command == "fetch" else clean_all()
    for message in errors:
        print(message, file=sys.stderr)
    return 1 if errors else 0
