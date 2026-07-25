#!/usr/bin/env python3
"""Top up cache_3yr_1h (ends 2026-05-25) with fresh 1H klines through today.

Writes the merged history to cache_ext/ so the validated cache is left untouched.
Public market endpoint only — no API key needed.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import aiohttp
import pandas as pd
import yaml

ROOT = Path(__file__).parent
SRC = ROOT / "cache_3yr_1h"
DST = ROOT / "cache_ext"
BASE = "https://api.bybit.com"
CONCURRENCY = 6          # polite: Bybit public limit is generous but not unlimited
RETRIES = 4

COLS = ["start", "open", "high", "low", "close", "volume", "turnover"]


async def fetch_chunk(sess, sem, symbol, start_ms, end_ms):
    params = {"category": "linear", "symbol": symbol, "interval": "60",
              "start": str(start_ms), "end": str(end_ms), "limit": "1000"}
    for attempt in range(RETRIES):
        try:
            async with sem:
                async with sess.get(f"{BASE}/v5/market/kline", params=params,
                                    timeout=aiohttp.ClientTimeout(total=30)) as r:
                    j = await r.json()
            if j.get("retCode") == 0:
                return j.get("result", {}).get("list", []) or []
            if attempt == RETRIES - 1:
                return []
        except Exception:
            if attempt == RETRIES - 1:
                return []
        await asyncio.sleep(0.5 * (2 ** attempt))
    return []


async def fetch_symbol(sess, sem, symbol, since_ms, now_ms):
    rows = []
    cur = since_ms
    while cur < now_ms:
        chunk_end = min(cur + 1000 * 3_600_000, now_ms)
        data = await fetch_chunk(sess, sem, symbol, cur, chunk_end)
        if not data:
            break
        rows.extend(data)
        oldest = min(int(r[0]) for r in data)
        newest = max(int(r[0]) for r in data)
        if newest <= cur:
            break
        cur = newest + 3_600_000
        if len(data) < 2:
            break
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=COLS)
    df["start"] = pd.to_datetime(df["start"].astype("int64"), unit="ms")
    for c in COLS[1:]:
        df[c] = df[c].astype(float)
    return df.sort_values("start").drop_duplicates("start").reset_index(drop=True)


async def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    symbols = [s for s, sc in (cfg.get("symbols") or {}).items()
               if (sc or {}).get("enabled", True)]
    symbols.sort()
    DST.mkdir(exist_ok=True)
    now_ms = int(time.time() * 1000)

    sem = asyncio.Semaphore(CONCURRENCY)
    ok = miss = 0
    async with aiohttp.ClientSession() as sess:
        for i, sym in enumerate(symbols, 1):
            src = SRC / f"{sym}.parquet"
            old = pd.read_parquet(src) if src.exists() else None
            if old is not None and not old.empty:
                since = int(old["start"].max().timestamp() * 1000) + 3_600_000
            else:
                since = now_ms - 400 * 24 * 3_600_000  # ~13mo if no base history

            new = await fetch_symbol(sess, sem, sym, since, now_ms)
            if new is None and old is None:
                miss += 1
                continue
            if old is None:
                merged = new
            elif new is None:
                merged = old
            else:
                merged = (pd.concat([old, new], ignore_index=True)
                          .drop_duplicates("start").sort_values("start")
                          .reset_index(drop=True))
            merged.to_parquet(DST / f"{sym}.parquet", index=False)
            ok += 1
            if i % 25 == 0:
                print(f"  {i}/{len(symbols)}  last={merged['start'].max()}", flush=True)

    print(f"[EXT] wrote {ok} symbols to {DST.name}, {miss} unavailable")
    # coverage report
    ends = []
    for f in DST.glob("*.parquet"):
        d = pd.read_parquet(f, columns=["start"])
        ends.append(d["start"].max())
    if ends:
        s = pd.Series(ends)
        print(f"[EXT] newest candle: min={s.min()}  median={s.median()}  max={s.max()}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
