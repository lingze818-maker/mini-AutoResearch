from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from config_loader import CONFIG_PATH, load_config
from tests.conftest import ROOT


def test_config_file_is_the_project_source_of_truth():
    config = load_config()
    assert config["dates"]["start"] == "2019-01-01"
    assert config["dates"]["end"] == "2026-10-31"
    assert config["dates"]["in_sample_end"] == "2023-12-31"
    assert config["data"]["adjust"] == "qfq"
    assert config["data"]["frequency"] == "daily"
    assert config["strategy"]["ma_cross"]["short_window"] == 20
    assert config["strategy"]["ma_cross"]["long_window"] == 60
    assert [row["symbol"] for row in config["universe"]] == ["510300", "510500", "518880"]
    assert config["paths"]["raw"] == "data/raw"
    assert config["paths"]["clean"] == "data/clean"
    assert config["paths"]["reports"] == "reports"


def test_split_loader_has_no_override_argument():
    assert list(inspect.signature(load_config).parameters) == []
    assert CONFIG_PATH == ROOT / "config.yaml"


def test_loaded_config_cannot_replace_the_split():
    config = load_config()
    with pytest.raises(TypeError):
        config["dates"]["in_sample_end"] = "2020-01-01"


def test_config_text_has_no_secret_material():
    text = Path(CONFIG_PATH).read_text(encoding="utf-8")
    lowered = text.lower()
    assert "sk-" not in lowered
    assert "api_key:" not in lowered
    assert "secret:" not in lowered


def test_requirements_stay_small_and_off_the_live_stack():
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for name in ("vectorbt", "pandas", "pyyaml", "pytest"):
        assert name in text
    assert "akshare" not in text
    assert "backtrader" not in text
