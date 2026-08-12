#!/usr/bin/env python3
"""Pull full 1H history for the symbols the live config does NOT trade.

Why this is the load-bearing step of the symbol study: `cache_3yr_1h/` was built in two
passes. All 277 LIVE symbols run to 2026-07-25; all 232 non-live symbols stop at
2026-05-25 or earlier (median ~10 months of bars vs ~2.1 years). So today the repo cannot
ask "was the universe chosen well?" out-of-sample — the comparison symbols have no data in
the only untouched window, and any cross-symbol test spanning 2026-05-25 is confounded by
the cache boundary rather than by the market. STRATEGY_VERDICT_2026-08-11.md 2.5 flags
this and notes that re-pulling would give a free out-of-universe test.

Public `/v5/market/kline` endpoint — no API key, read-only market data.
Writes to cache_outside/ so the validated cache is left untouched.
"""
from __future__ import annotations

import asyncio
import sys
import time
import warnings
from pathlib import Path

import aiohttp
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "cache_3yr_1h"
DST = ROOT / "cache_outside"
BASE = "https://api.bybit.com"
CONCURRENCY = 6
RETRIES = 4
START = pd.Timestamp("2023-05-01")     # a month before the study window, for indicator warmup
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
    rows, cur = [], since_ms
    while cur < now_ms:
        chunk_end = min(cur + 1000 * 3_600_000, now_ms)
        data = await fetch_chunk(sess, sem, symbol, cur, chunk_end)
        if not data:
            break
        rows.extend(data)
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
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))["symbols"]
    live = {s for s, sc in (cfg or {}).items()
            if (sc or {}).get("enabled", True) and (sc or {}).get("configs")}

    targets = []
    for f in sorted(SRC.glob("*.parquet")):
        sym = f.stem
        if sym in live:
            continue
        if "-" in sym:                      # dated futures contracts, not perps
            continue
        targets.append(sym)

    DST.mkdir(exist_ok=True)
    now_ms = int(time.time() * 1000)
    since_ms = int(START.timestamp() * 1000)
    print(f"[OUT] pulling {len(targets)} non-live symbols, {START.date()} -> now", flush=True)

    sem = asyncio.Semaphore(CONCURRENCY)
    ok = miss = 0
    async with aiohttp.ClientSession() as sess:
        for i, sym in enumerate(targets, 1):
            if (DST / f"{sym}.parquet").exists():
                ok += 1
                continue
            new = await fetch_symbol(sess, sem, sym, since_ms, now_ms)
            if new is None or new.empty:
                miss += 1
                continue
            new.to_parquet(DST / f"{sym}.parquet", index=False)
            ok += 1
            if i % 20 == 0:
                print(f"  {i}/{len(targets)}  {sym} n={len(new):,} "
                      f"{new.start.min().date()}..{new.start.max().date()}", flush=True)

    print(f"[OUT] wrote {ok}, unavailable {miss}")
    ends, ns = [], []
    for f in DST.glob("*.parquet"):
        d = pd.read_parquet(f, columns=["start"])
        ends.append(d["start"].max()); ns.append(len(d))
    if ends:
        s = pd.Series(ends)
        print(f"[OUT] newest candle: min={s.min()} median={s.median()} max={s.max()}")
        print(f"[OUT] bars: median={pd.Series(ns).median():,.0f} "
              f"symbols with >=15000 bars: {(pd.Series(ns)>=15000).sum()}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
