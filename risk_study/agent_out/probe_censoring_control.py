#!/usr/bin/env python3
"""Control test: is the "last-21-days looks terrible" effect a pure
right-censoring artifact, or does it reflect genuine performance?

Pick an ARTIFICIAL cutoff deep in history (2024-12-01), well before the
independently-established June-2026 edge break (STRATEGY_VERDICT 1.2). Take
every symbol's data truncated at that cutoff and run the exact same
signal->BOS->entry->exit pipeline TWICE for the same set of entries in the
[cutoff-21d, cutoff] window:
  (a) TRUNCATED: exits can only be found within the truncated series (mimics
      what the real end-of-data right-censoring does)
  (b) FULL: exits are resolved against the FULL (untruncated) series, i.e.
      the "true" outcome with no artificial data-ending.

If (a) shows a cliff that (b) does not, the cliff is a pure resolution-
window artifact, not a real regime effect, because both draw from the exact
same underlying trades/market history — only the resolution horizon differs.
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
MAX_WAIT = bt.MAX_WAIT_CANDLES
CHOP_T = 52.0
FEE = 0.00341
CUTOFF = pd.Timestamp("2024-12-01")
WINDOW_START = CUTOFF - pd.Timedelta(days=21)


def resolve(side, entry, sl, tp, e, n, h, l):
    for k in range(e, n):
        if side == "long":
            hs, ht = l[k] <= sl, h[k] >= tp
        else:
            hs, ht = h[k] >= sl, l[k] <= tp
        if hs or ht:
            r = -1.0 if hs else None
            return k, hs, ht
    return None, None, None


def replay(args):
    sym, picks, cache_dir = args
    f = Path(cache_dir) / f"{sym}.parquet"
    if not f.exists():
        return []
    try:
        full_raw = pd.read_parquet(f)
    except Exception:
        return []
    if full_raw.empty or len(full_raw) < 2000:
        return []
    full_raw = full_raw.sort_values("start").reset_index(drop=True)
    if full_raw["start"].min() > WINDOW_START or full_raw["start"].max() < CUTOFF + pd.Timedelta(days=60):
        return []  # need real data on both sides of the artificial cutoff to make a fair control

    # Truncated series: only rows up to CUTOFF (mimics "data ends here")
    trunc_raw = full_raw[full_raw["start"] <= CUTOFF].reset_index(drop=True)
    if len(trunc_raw) < 2000:
        return []

    full_df = bt.prepare_data(full_raw)
    full_df["chop"] = chop_series(full_df)
    trunc_df = bt.prepare_data(trunc_raw)
    trunc_df["chop"] = chop_series(trunc_df)

    # Detect signals + BOS + entries on the TRUNCATED series (this is what a real
    # backtest "as of CUTOFF" would have seen) -- entry gating must match what was
    # knowable at the time, so we do NOT use the full series for signal detection.
    o_t = trunc_df.open.values; h_t = trunc_df.high.values; l_t = trunc_df.low.values; c_t = trunc_df.close.values
    atr_t = trunc_df.atr.values; ema_t = trunc_df.ema.values; ch_t = trunc_df.chop.values; ts_t = trunc_df.start.values
    n_t = len(c_t)

    sigs = {}
    for s in bt.detect_signals(trunc_df):
        sigs.setdefault(s["type"], []).append(s)

    # Full series arrays, for the "true outcome" resolution
    o_f = full_df.open.values; h_f = full_df.high.values; l_f = full_df.low.values; c_f = full_df.close.values
    ts_f = full_df.start.values
    n_f = len(c_f)
    # map: truncated index -> full index (they share the same prefix, so identity works
    # as long as the truncated series's rows are an exact prefix of the full series)
    assert (trunc_df["start"].values == full_df["start"].values[:n_t]).all(), sym

    out = []
    for dt, (rr, am) in picks.items():
        for s in sigs.get(dt, []):
            conf, side, lvl = s["conf_idx"], s["side"], s["swing"]
            bos = None
            for i in range(1, MAX_WAIT + 1):
                idx = conf + i
                if idx >= n_t:
                    break
                if (side == "long" and c_t[idx] > lvl) or (side == "short" and c_t[idx] < lvl):
                    bos = idx
                    break
            if bos is None:
                continue
            e = bos + 1
            if e >= n_t or not np.isfinite(ema_t[bos]):
                continue
            if side == "long" and not c_t[bos] > ema_t[bos]:
                continue
            if side == "short" and not c_t[bos] < ema_t[bos]:
                continue
            if np.isfinite(ch_t[bos]) and ch_t[bos] >= CHOP_T:
                continue
            if not (np.isfinite(atr_t[bos]) and atr_t[bos] > 0):
                continue

            entry_ts = pd.Timestamp(ts_t[e])
            if not (WINDOW_START <= entry_ts <= CUTOFF):
                continue  # only care about the 21-day window before the artificial cutoff

            entry = o_t[e]; sl_d = atr_t[bos] * am
            sl = entry - sl_d if side == "long" else entry + sl_d
            tp = entry + sl_d * rr if side == "long" else entry - sl_d * rr

            # (a) TRUNCATED resolution
            k_t, hs_t, ht_t = resolve(side, entry, sl, tp, e, n_t, h_t, l_t)
            trunc_resolved = k_t is not None
            trunc_r = None
            if trunc_resolved:
                r_result = -1.0 if hs_t else rr
                fee_r = FEE * entry / abs(entry - sl)
                trunc_r = r_result - fee_r

            # (b) FULL resolution (same e index maps directly since prefix-identical)
            k_f, hs_f, ht_f = resolve(side, entry, sl, tp, e, n_f, h_f, l_f)
            full_resolved = k_f is not None
            full_r = None
            if full_resolved:
                r_result = -1.0 if hs_f else rr
                fee_r = FEE * entry / abs(entry - sl)
                full_r = r_result - fee_r

            out.append((sym, entry_ts, trunc_resolved, trunc_r, full_resolved, full_r))
    return out


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

    rows = []
    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        for i, r in enumerate(pool.imap_unordered(replay, jobs, chunksize=4), 1):
            rows.extend(r)

    d = pd.DataFrame(rows, columns=["symbol", "entry_time", "trunc_resolved", "trunc_r",
                                     "full_resolved", "full_r"])
    d.to_parquet(Path(__file__).parent / "censoring_control_rows.parquet", index=False)

    print(f"Artificial cutoff: {CUTOFF.date()}  window: [{WINDOW_START.date()}, {CUTOFF.date()}]")
    print(f"total entries in window (across symbols with data spanning the cutoff): {len(d):,}")
    print(f"\n(a) TRUNCATED (as a real-time backtest would have seen, data ending {CUTOFF.date()}):")
    print(f"    resolved: {d.trunc_resolved.sum():,} ({d.trunc_resolved.mean():.1%})  "
          f"avg R of resolved: {d.loc[d.trunc_resolved,'trunc_r'].mean():+.4f}")
    print(f"\n(b) FULL (same trades, resolved against real future data -- the TRUE outcome):")
    print(f"    resolved: {d.full_resolved.sum():,} ({d.full_resolved.mean():.1%})  "
          f"avg R of resolved: {d.loc[d.full_resolved,'full_r'].mean():+.4f}")

    only_trunc_censored = d[(~d.trunc_resolved) & (d.full_resolved)]
    print(f"\nTrades that were right-censored under truncation but DID resolve with future data: "
          f"{len(only_trunc_censored):,}")
    if len(only_trunc_censored):
        print(f"  their TRUE avg R (from full data): {only_trunc_censored.full_r.mean():+.4f}")
        print(f"  win rate (R>0): {(only_trunc_censored.full_r > 0).mean():.1%}")


if __name__ == "__main__":
    main()
