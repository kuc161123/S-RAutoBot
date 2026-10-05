"""Unsigned public-data connectivity check. Never reads credentials or storage."""

from __future__ import annotations
import argparse
import asyncio
import json
from pathlib import Path
import time
import aiohttp
from .bybit import BybitClient
from .context import ContextService


def failure(exc):
    return {
        "ok": False,
        "error_type": type(exc).__name__,
        "http_status": getattr(exc, "http_status", None),
        "ret_code": getattr(exc, "ret_code", None),
    }


async def probe():
    result = {
        "as_of": time.time(),
        "scope": "Public endpoints from this host, not Railway deployment proof",
    }
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=50), trust_env=False
    ) as session:
        client = BybitClient(session)
        try:
            instruments = await client.instruments()
            result["instruments"] = {"ok": True, "count": len(instruments)}
        except Exception as exc:
            result["instruments"] = failure(exc)
        daily = []
        for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            try:
                bars = await client.candles(symbol, "D", 500)
                execution = await client.candles(symbol, "240", 500)
                ticker = await client.ticker(symbol)
                result[symbol] = {
                    "ok": True,
                    "daily_count": len(bars),
                    "execution_count": len(execution),
                    "daily_last_open_ms": bars[-1].open_time,
                    "funding_available": ticker.get("fundingRate") is not None,
                }
                if symbol == "BTCUSDT":
                    daily = bars
            except Exception as exc:
                result[symbol] = failure(exc)
        context = await ContextService(session).build(daily, time.time())
        result["context"] = {
            k: context.get(k)
            for k in (
                "data_complete",
                "reason",
                "risk_state",
                "sources",
                "event_coverage",
                "as_of",
                "expires_at",
            )
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(probe())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
