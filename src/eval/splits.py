"""The one map from a split name to its frozen dates.

`config.yaml` holds the dates. Callers select a name; they do not pass a window.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date, timedelta
from types import MappingProxyType

from config_loader import load_config

_FOLD_NAMES = ("A", "B", "C")
_FIELDS = ("in_sample_start", "in_sample_end", "out_of_sample_start", "out_of_sample_end")
_SELECTED: ContextVar[str] = ContextVar("sample_split", default="default")


@dataclass(frozen=True, slots=True)
class SampleSplit:
    name: str
    in_sample_start: str
    in_sample_end: str
    out_of_sample_start: str
    out_of_sample_end: str


def registered_splits() -> Mapping[str, SampleSplit]:
    """Return A, B, C, and the existing default. No other split is valid."""
    dates = load_config()["dates"]
    default = SampleSplit(
        name="default",
        in_sample_start=_parse(dates["start"]),
        in_sample_end=_parse(dates["in_sample_end"]),
        out_of_sample_start=_next_day(_parse(dates["in_sample_end"])),
        out_of_sample_end=_parse(dates["end"]),
    )
    _require_order(default)
    folds = dates["folds"]
    if set(folds) != set(_FOLD_NAMES):
        raise ValueError("config dates.folds must be exactly A, B, and C")
    windows: dict[str, SampleSplit] = {}
    for name in _FOLD_NAMES:
        row = folds[name]
        if set(row) != set(_FIELDS):
            raise ValueError(f"config dates.folds.{name} must list the four window dates")
        window = SampleSplit(
            name=name,
            in_sample_start=_parse(row["in_sample_start"]),
            in_sample_end=_parse(row["in_sample_end"]),
            out_of_sample_start=_parse(row["out_of_sample_start"]),
            out_of_sample_end=_parse(row["out_of_sample_end"]),
        )
        if window.in_sample_start != default.in_sample_start:
            raise ValueError(f"fold {name} in-sample start must match dates.start")
        if window.out_of_sample_start != _next_day(window.in_sample_end):
            raise ValueError(f"fold {name} out-of-sample start must be the day after in_sample_end")
        _require_order(window)
        if date.fromisoformat(window.out_of_sample_end) >= date.fromisoformat(default.out_of_sample_start):
            raise ValueError(f"fold {name} out-of-sample window must end before the default holdout")
        windows[name] = window
    windows["default"] = default
    return MappingProxyType(windows)


def active_split() -> SampleSplit:
    windows = registered_splits()
    name = _SELECTED.get()
    if name not in windows:
        raise ValueError(f"unknown split {name!r}; choose one of {', '.join(windows)}")
    return windows[name]


@contextmanager
def use_split(name: str) -> Iterator[SampleSplit]:
    """Bind one registered split for the current run, then restore the previous one."""
    windows = registered_splits()
    if name not in windows:
        raise ValueError(f"unknown split {name!r}; choose one of {', '.join(windows)}")
    token = _SELECTED.set(name)
    try:
        yield windows[name]
    finally:
        _SELECTED.reset(token)


def _parse(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("split dates must be YYYY-MM-DD strings")
    return date.fromisoformat(value).isoformat()


def _next_day(value: str) -> str:
    return (date.fromisoformat(value) + timedelta(days=1)).isoformat()


def _require_order(window: SampleSplit) -> None:
    start = date.fromisoformat(window.in_sample_start)
    in_end = date.fromisoformat(window.in_sample_end)
    out_start = date.fromisoformat(window.out_of_sample_start)
    out_end = date.fromisoformat(window.out_of_sample_end)
    if not start <= in_end < out_start <= out_end:
        raise ValueError(f"split {window.name} dates are out of order")
