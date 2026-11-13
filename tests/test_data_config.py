"""The data layer reads the frozen universe and the qfq window from config.yaml."""

from config_loader import load_config
from data.fetch import date_window, symbols


def test_data_layer_uses_only_the_three_etfs():
    config = load_config()
    names = [row["symbol"] for row in config["universe"]]
    assert names == ["510300", "510500", "518880"]
    assert "159915" not in names
    assert symbols() == names
    assert config["data"]["source"] == "akshare"
    assert config["data"]["frequency"] == "daily"
    assert date_window() == ("2019-01-01", "2026-10-31", "qfq")
    assert config["paths"]["raw"] == "data/raw"
    assert config["paths"]["clean"] == "data/clean"
