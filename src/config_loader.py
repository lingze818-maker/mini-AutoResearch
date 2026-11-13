"""Load the repo's config.yaml. The sample split is not a function argument."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def load_config() -> MappingProxyType:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("config.yaml must be a mapping")
    config = _freeze(loaded)
    for key in ("start", "end", "in_sample_end"):
        if key not in config["dates"]:
            raise ValueError(f"config dates.{key} is missing")
    windows = config["strategy"]["ma_cross"]
    if "short_window" not in windows or "long_window" not in windows:
        raise ValueError("config strategy.ma_cross windows are missing")
    if "adjust" not in config["data"]:
        raise ValueError("config data.adjust is missing")
    return config
