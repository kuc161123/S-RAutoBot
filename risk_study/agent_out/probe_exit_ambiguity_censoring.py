#!/usr/bin/env python3
"""Full-universe probe: same-bar entry-bar resolution, stop&target-same-bar
ambiguity, and right-censoring (unresolved trades dropped by the `for k`
loop with no `else`).

Reuses the exact BOS/CHOP(bos)/ATR(bos)/entry(o[e]) logic of
risk_study/build_risk_universe.py::replay(), instrumented with extra
counters instead of just returning resolved rows.
"""
from __future__ import annotations
import multiprocessing as mp
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import backtest_3yr_walkforward as bt  # noqa: E402
from backtest_halt_multiwindow import chop_series  # noqa: E402

CACHE = ROOT / "cache_3yr_1h"
GEN_FROM = pd.Timestamp("2023-06-01")
MAX_WAIT = bt.MAX_WAIT_CANDLES
CHOP_T = 52.0


def replay(args):
    sym, picks, cache_dir = args
    f = Path(cache_dir) / f"{sym}.parquet"
    if not f.exists():
        return None
    try:
        df = pd.read_parquet(f)
    except Exception:
        return None
    if df.empty or len(df) < 2000:
        return None
    df = bt.prepare_data(df)
    df["chop"] = chop_series(df)
    o = df.open.values; h = df.high.values; l = df.low.values; c = df.close.values
    atr = df.atr.values; ema = df.ema.values; ch = df.chop.values; ts = df.start.values
    n = len(c)
    sigs = {}
    for s in bt.detect_signals(df):
        sigs.setdefault(s["type"], []).append(s)

    n_entered = 0          # trades that passed all gates and got a fill at o[e]
    n_resolved = 0         # of those, resolved (stop or target hit before data end)
    n_censored = 0         # of those, NEVER resolved (right-censored, dropped silently)
    n_censored_last21 = 0  # censored AND entry within the last 21 days of THIS symbol's data
    n_same_bar_entry = 0   # resolved on bar k == e (the entry bar itself)
    n_ambiguous_total = 0  # resolved bars where BOTH stop and target conditions are true
    n_ambiguous_same_bar = 0  # of those, specifically on the entry bar itself
    entry_times = []
    censored_entry_times = []

    sym_last_ts = pd.Timestamp(ts[-1])
    cutoff21 = sym_last_ts - pd.Timedelta(days=21)

    for dt, (rr, am) in picks.items():
        for s in sigs.get(dt, []):
            conf, side, lvl = s["conf_idx"], s["side"], s["swing"]
            bos = None
            for i in range(1, MAX_WAIT + 1):
                idx = conf + i
                if idx >= n:
                    break
                if (side == "long" and c[idx] > lvl) or (side == "short" and c[idx] < lvl):
                    bos = idx
                    break
            if bos is None:
                continue
            e = bos + 1
            if e >= n or not np.isfinite(ema[bos]):
                continue
            if side == "long" and not c[bos] > ema[bos]:
                continue
            if side == "short" and not c[bos] < ema[bos]:
                continue
            if np.isfinite(ch[bos]) and ch[bos] >= CHOP_T:
                continue
            if not (np.isfinite(atr[bos]) and atr[bos] > 0):
                continue

            entry = o[e]; sl_d = atr[bos] * am
            sl = entry - sl_d if side == "long" else entry + sl_d
            tp = entry + sl_d * rr if side == "long" else entry - sl_d * rr

            n_entered += 1
            entry_ts = pd.Timestamp(ts[e])
            entry_times.append(entry_ts)

            resolved = False
            for k in range(e, n):
                if side == "long":
                    hs, ht = l[k] <= sl, h[k] >= tp
                else:
                    hs, ht = h[k] >= sl, l[k] <= tp
                if hs or ht:
                    resolved = True
                    n_resolved += 1
                    if hs and ht:
                        n_ambiguous_total += 1
                        if k == e:
                            n_ambiguous_same_bar += 1
                    if k == e:
                        n_same_bar_entry += 1
                    break
            if not resolved:
                n_censored += 1
                censored_entry_times.append(entry_ts)
                if entry_ts >= cutoff21:
                    n_censored_last21 += 1

    return dict(sym=sym, n_entered=n_entered, n_resolved=n_resolved, n_censored=n_censored,
                n_censored_last21=n_censored_last21, n_same_bar_entry=n_same_bar_entry,
                n_ambiguous_total=n_ambiguous_total, n_ambiguous_same_bar=n_ambiguous_same_bar,
                sym_last_ts=sym_last_ts)


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))["symbols"]
    jobs = []
    for s, sc in cfg.items():
        if not (sc or {}).get("enabled", True):
            continue
        picks = {c["divergence_type"]: (float(c["rr"]), float(c["atr_mult"]))
                 for c in (sc or {}).get("configs", []) or []}
        if picks and (CACHE / f"{s}.parquet").exists():
            jobs.append((s, picks, str(CACHE)))
    jobs.sort()
    print(f"[PROBE] {len(jobs)} symbols", flush=True)

    results = []
    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        for i, r in enumerate(pool.imap_unordered(replay, jobs, chunksize=4), 1):
            if r:
                results.append(r)
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    tot_entered = sum(r["n_entered"] for r in results)
    tot_resolved = sum(r["n_resolved"] for r in results)
    tot_censored = sum(r["n_censored"] for r in results)
    tot_censored_last21 = sum(r["n_censored_last21"] for r in results)
    tot_same_bar = sum(r["n_same_bar_entry"] for r in results)
    tot_ambig = sum(r["n_ambiguous_total"] for r in results)
    tot_ambig_same_bar = sum(r["n_ambiguous_same_bar"] for r in results)

    print("\n" + "=" * 70)
    print("EXIT RESOLUTION / RIGHT-CENSORING — full live-config universe")
    print("=" * 70)
    print(f"  symbols processed:                          {len(results)}")
    print(f"  total entered trades (passed all gates):    {tot_entered:,}")
    print(f"  resolved (stop or target hit before data end): {tot_resolved:,} ({tot_resolved/tot_entered:.2%})")
    print(f"  RIGHT-CENSORED (never resolved, silently dropped): {tot_censored:,} ({tot_censored/tot_entered:.2%})")
    print(f"    of which, entry within last 21 days of that symbol's data: {tot_censored_last21:,} "
          f"({tot_censored_last21/max(1,tot_censored):.2%} of censored)")
    print()
    print(f"  resolved on the ENTRY BAR itself (k == e):  {tot_same_bar:,} ({tot_same_bar/tot_resolved:.2%} of resolved)")
    print(f"  AMBIGUOUS resolved bars (stop AND target both true on the resolving bar): "
          f"{tot_ambig:,} ({tot_ambig/tot_resolved:.2%} of resolved)")
    print(f"    of which on the entry bar itself: {tot_ambig_same_bar:,} "
          f"({tot_ambig_same_bar/max(1,tot_same_bar):.2%} of same-bar-entry resolutions)")

    out = pd.DataFrame(results)
    out.to_csv(Path(__file__).parent / "exit_ambiguity_censoring_by_symbol.csv", index=False)
    print(f"\n  per-symbol detail written to exit_ambiguity_censoring_by_symbol.csv")


if __name__ == "__main__":
    main()
