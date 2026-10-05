"""Optional, disposable Redis cache for public market candles only."""

from __future__ import annotations

import json
import time
from dataclasses import asdict

from .models import Candle
from .bybit import validate_candles


class MarketCache:
    def __init__(self, url=""):
        self.client = None
        self.status = "disabled"
        if url:
            from redis.asyncio import Redis

            self.client = Redis.from_url(
                url, socket_timeout=2, socket_connect_timeout=2, decode_responses=True
            )
            self.status = "configured"

    async def candles(self, client, symbol, interval, limit=500, now=None):
        now = time.time() if now is None else now
        # Version + venue prevents mixing testnet/mainnet or older data schemas.
        venue = getattr(client, "base_url", "unknown")
        key = f"apex:v1:candles:{venue}:{symbol}:{interval}:{limit}"
        if self.client:
            try:
                raw = await self.client.get(key)
                if raw:
                    item = json.loads(raw)
                    if 0 <= now - item["fetched_at"] < 30:
                        self.status = "connected"
                        return validate_candles(
                            [Candle(**b) for b in item["bars"]],
                            interval,
                            int(now * 1000),
                        )
            except Exception:
                self.status = "unavailable; direct market reads"
        bars = await client.candles(symbol, interval, limit, now_ms=int(now * 1000))
        if self.client:
            try:
                await self.client.setex(
                    key,
                    30,
                    json.dumps(
                        {"fetched_at": now, "bars": [asdict(b) for b in bars]},
                        allow_nan=False,
                    ),
                )
                self.status = "connected"
            except Exception:
                self.status = "unavailable; direct market reads"
        return bars

    async def close(self):
        if self.client:
            await self.client.aclose()
