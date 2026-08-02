#!/usr/bin/env python3
"""Download 5-minute klines for the live universe into cache_5m/.

Purpose: the 1H backtest cannot see the order of events inside an hour. When a bar
touches both the trailing stop and the take-profit, the resolver has to assume the
stop hit first. 5m bars resolve that ordering 12x more finely, which matters far more
for trailed exits (stop moves every bar) than for a fixed bracket.

Sharded so several copies can run in parallel:  --shard 0 --of 8
Resumable: a symbol whose file already covers the requested range is skipped.

Public market endpoint, no API key.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import aiohttp
import pandas as pd
import yaml

ROOT = Path(__file__).parent
DST = ROOT / "cache_5m"
BASE = "https://api.bybit.com"
INTERVAL = "5"
STEP_MS = 5 * 60_000
CHUNK = 1000
RETRIES = 5
COLS = ["start", "open", "high", "low", "close", "volume", "turnover"]
SINCE = pd.Timestamp("2025-01-01")


async def fetch_chunk(sess, sem, symbol, start_ms, end_ms):
    params = {"category": "linear", "symbol": symbol, "interval": INTERVAL,
              "start": str(start_ms), "end": str(end_ms), "limit": str(CHUNK)}
    for attempt in range(RETRIES):
        try:
            async with sem:
                async with sess.get(f"{BASE}/v5/market/kline", params=params,
                                    timeout=aiohttp.ClientTimeout(total=40)) as r:
                    if r.status == 429:
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    j = await r.json()
            if j.get("retCode") == 0:
                return j.get("result", {}).get("list", []) or []
            if j.get("retCode") in (10001,):        # bad symbol / delisted
                return []
        except Exception:
            pass
        await asyncio.sleep(0.4 * (2 ** attempt))
    return []


async def fetch_symbol(sess, sem, symbol, since_ms, now_ms):
    rows, cur = [], since_ms
    while cur < now_ms:
        end = min(cur + CHUNK * STEP_MS, now_ms)
        data = await fetch_chunk(sess, sem, symbol, cur, end)
        if not data:
            cur = end + STEP_MS                      # gap: skip forward, keep going
            if end >= now_ms:
                break
            continue
        rows.extend(data)
        newest = max(int(r[0]) for r in data)
        if newest <= cur:
            break
        cur = newest + STEP_MS
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=COLS)
    df["start"] = pd.to_datetime(df["start"].astype("int64"), unit="ms")
    for c in COLS[1:]:
        df[c] = df[c].astype(float)
    return df.sort_values("start").drop_duplicates("start").reset_index(drop=True)


def already_done(path, now_ms):
    if not path.exists():
        return False
    try:
        d = pd.read_parquet(path, columns=["start"])
    except Exception:
        return False
    if d.empty:
        return False
    last = pd.to_datetime(d["start"]).max()
    return (pd.Timestamp(now_ms, unit="ms") - last) < pd.Timedelta(hours=6)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--conc", type=int, default=8)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    symbols = sorted(s for s, sc in (cfg.get("symbols") or {}).items()
                     if (sc or {}).get("enabled", True))
    mine = [s for i, s in enumerate(symbols) if i % a.of == a.shard]
    DST.mkdir(exist_ok=True)
    now_ms = int(time.time() * 1000)
    since_ms = int(SINCE.timestamp() * 1000)

    # paging within a symbol is cursor-sequential, so the parallelism has to come from
    # running several SYMBOLS at once. `conc` = symbols in flight for this shard.
    sem = asyncio.Semaphore(a.conc * 2)          # request-level guard
    todo = [s for s in mine if not already_done(DST / f"{s}.parquet", now_ms)]
    skipped = len(mine) - len(todo)
    stats = {"ok": 0, "fail": 0, "n": 0}
    t0 = time.time()

    async def one(sess, sym):
        out = DST / f"{sym}.parquet"
        try:
            df = await fetch_symbol(sess, sem, sym, since_ms, now_ms)
        except Exception:
            df = None
        if df is None or len(df) < 500:
            stats["fail"] += 1
        else:
            df.to_parquet(out, index=False)
            stats["ok"] += 1
        stats["n"] += 1
        if stats["n"] % 3 == 0:
            el = max(time.time() - t0, 1)
            eta = (len(todo) - stats["n"]) / max(stats["n"] / el, 1e-9) / 60
            print(f"[shard {a.shard}/{a.of}] {stats['n']}/{len(todo)} "
                  f"ok={stats['ok']} fail={stats['fail']} eta={eta:.0f}m", flush=True)

    q = asyncio.Queue()
    for s in todo:
        q.put_nowait(s)

    async def worker(sess):
        while True:
            try:
                sym = q.get_nowait()
            except asyncio.QueueEmpty:
                return
            await one(sess, sym)

    conn = aiohttp.TCPConnector(limit=a.conc * 3)
    async with aiohttp.ClientSession(connector=conn) as sess:
        await asyncio.gather(*[worker(sess) for _ in range(a.conc)])

    print(f"[shard {a.shard}/{a.of}] DONE ok={stats['ok']} skip={skipped} "
          f"fail={stats['fail']} in {(time.time() - t0) / 60:.1f}m", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
