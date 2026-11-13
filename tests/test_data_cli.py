"""python -m src.data fetch|clean reports provider failures and exits non-zero."""

from data.cli import main


def test_fetch_command_succeeds_when_every_symbol_is_written(monkeypatch):
    monkeypatch.setattr("data.cli.fetch_all", lambda: [])
    assert main(["fetch"]) == 0


def test_clean_command_succeeds_when_every_symbol_is_written(monkeypatch):
    monkeypatch.setattr("data.cli.clean_all", lambda: [])
    assert main(["clean"]) == 0


def test_fetch_command_prints_the_akshare_error(monkeypatch, capsys):
    message = "akshare fund_etf_hist_em failed for 518880 2019-01-01 to 2026-10-31: disconnected"
    monkeypatch.setattr("data.cli.fetch_all", lambda: [message])
    assert main(["fetch"]) == 1
    assert message in capsys.readouterr().err
