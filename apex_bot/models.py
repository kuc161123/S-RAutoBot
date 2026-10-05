"""Serializable contracts shared by the data, analysis and execution services."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Candle:
    open_time: int  # Unix milliseconds, UTC; callers supply closed candles only.
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass(frozen=True)
class Opportunity:
    id: str
    symbol: str
    side: str  # Buy or Sell (Bybit vocabulary)
    setup: str
    state: str  # WAIT, READY or INVALID
    entry: float
    stop: float
    target1: float
    target2: float
    invalidation: float
    confirmation: float
    created_at: float
    expires_at: float
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    tier: int = 1
    bucket: str = "crypto"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Instrument:
    symbol: str
    qty_step: float
    min_qty: float
    max_qty: float
    tick_size: float
    min_notional: float
    funding_interval_minutes: int = 480
    max_leverage: float = 5.0
