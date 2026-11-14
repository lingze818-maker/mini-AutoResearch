"""Deterministic stand-in for a model proposal.

The sources below are presets. They are not fit to the out-of-sample window.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from agent.limits import OFFLINE_BANNER

MOMENTUM_20 = '''\
def generate(bars):
    """20-bar momentum. `bars` is only the current prefix."""
    close = bars["close"].astype(float)
    if len(close) <= 20:
        return {"entry": False, "exit": False}
    ratio = float(close.iloc[-1] / close.iloc[-21]) - 1.0
    if ratio > 0.0:
        return {"entry": True, "exit": False}
    if ratio < 0.0:
        return {"entry": False, "exit": True}
    return {"entry": False, "exit": False}
'''

VOLATILITY_20 = '''\
def generate(bars):
    """Exit when 20-day realized volatility is above a preset 2% daily ceiling."""
    close = bars["close"].astype(float)
    if len(close) < 21:
        return {"entry": False, "exit": False}
    returns = close.iloc[-21:].pct_change().iloc[1:]
    volatility = float(returns.std(ddof=0))
    last = float(close.iloc[-1] / close.iloc[-2] - 1.0)
    if volatility > 0.02:
        return {"entry": False, "exit": True}
    if last > 0.0:
        return {"entry": True, "exit": False}
    return {"entry": False, "exit": False}
'''

MOMENTUM_60 = '''\
def generate(bars):
    """60-bar momentum. The window is a preset, not a searched parameter."""
    close = bars["close"].astype(float)
    if len(close) <= 60:
        return {"entry": False, "exit": False}
    ratio = float(close.iloc[-1] / close.iloc[-61]) - 1.0
    if ratio > 0.0:
        return {"entry": True, "exit": False}
    if ratio < 0.0:
        return {"entry": False, "exit": True}
    return {"entry": False, "exit": False}
'''


@dataclass(frozen=True, slots=True)
class Candidate:
    id: str
    name: str
    rationale: str
    source: str


class OfflineFixtureProvider:
    """Fixed proposals. This is not a model research run."""

    banner = OFFLINE_BANNER
    name = "offline-fixture"

    def __init__(self) -> None:
        self.calls = 0

    def propose(self, summaries: Mapping[str, Mapping[str, object]]) -> list[Candidate]:
        if self.calls:
            raise RuntimeError("offline provider was asked for another proposal")
        self.calls += 1
        quoted = ", ".join(
            f"{item.get('symbol')} in-sample return {item.get('period_return')}"
            for item in summaries.values()
        )
        context = (
            f"{OFFLINE_BANNER} In-sample summary used for context only: {quoted}. "
            "Lookbacks 20 and 60 and the 0.02 volatility ceiling are presets, not a search."
        )
        return [
            Candidate("momentum_20", "20-day momentum", context, MOMENTUM_20),
            Candidate("volatility_20", "20-day volatility ceiling", context, VOLATILITY_20),
            Candidate("momentum_60", "60-day momentum", context, MOMENTUM_60),
        ]
