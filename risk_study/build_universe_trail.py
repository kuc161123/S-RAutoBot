#!/usr/bin/env python3
"""Rebuild the study universe with the LIVE EXIT RULE — the s3_a1 trailing stop.

The first pass of this study resolved every trade against a fixed take-profit. That is what
the bot did until 2026-08-02; since then the stop TRAILS, and `config.yaml`
`risk.trailing_stop.enabled: true`. So the fixed-TP universe understates nothing about
sizing but does model the wrong exit. This rebuilds both arms from the same signals, same
bars, so `r_trail - r_fixed` is attributable to the exit and nothing else.

The rule, per CLAUDE.md 6.1 and `config.yaml risk.trailing_stop` (trigger_r 3.0, atr_mult 1.0):

  * Nothing happens until a CLOSED bar's own excursion reaches +3R.
  * Arming re-tests `mfe >= 3R` FRESH EVERY BAR against that bar's own extreme -- it is not
    a running peak. When price falls back the stop HOLDS rather than continuing to ratchet.
    A running-peak version is a different rule and was not the one validated.
  * From then the stop ratchets to `high - 1*ATR` (long) / `low + 1*ATR` (short), computed
    on CLOSED candles only, and NEVER widens: `stop = max(stop, candidate)` for a long.
  * There is no breakeven move, and one must not be added -- every `be*` variant lost to
    its non-`be` twin in every period tested.
  * `risk_dist` is carried explicitly rather than reconstructed from `|entry - stop|`,
    because the stop mutates and the round-trip loses the last ulp -- that turned an
    excursion of exactly 3.0R into 2.999999999999996 and skipped a ratchet.

Within a bar the stop in force is the one set at the close of the PREVIOUS bar; the trail
is updated only after the bar closes. Stop wins ties, as everywhere in this repo.

Output: risk_study/universe_trail.parquet with BOTH `r_fixed`/`exit_fixed` and
`r_trail`/`exit_trail` on every row.
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
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import backtest_3yr_walkforward as bt  # noqa: E402
from backtest_halt_multiwindow import chop_series  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = ROOT / "cache_3yr_1h"
GEN_FROM = pd.Timestamp("2023-06-01")
MAX_WAIT = bt.MAX_WAIT_CANDLES
CHOP_T = 52.0
TRIGGER_R = 3.0
TRAIL_ATR = 1.0


def resolve_trail(e, n, long_, entry, sl0, tp, risk_dist, h, l, atr):
    """Walk bars from e; return (r, exit_idx) under the s3_a1 trail. None if unresolved."""
    stop = sl0
    for k in range(e, n):
        # --- exit check against the stop in force (set at close of bar k-1) ---
        if long_:
            hit_stop = l[k] <= stop
            hit_tp = h[k] >= tp
        else:
            hit_stop = h[k] >= stop
            hit_tp = l[k] <= tp
        if hit_stop:                              # stop wins ties
            r = (stop - entry) / risk_dist if long_ else (entry - stop) / risk_dist
            return r, k
        if hit_tp:
            r = (tp - entry) / risk_dist if long_ else (entry - tp) / risk_dist
            return r, k

        # --- bar k has closed: update the trail from ITS OWN excursion ---
        a = atr[k]
        if not (np.isfinite(a) and a > 0):
            continue
        if long_:
            mfe = (h[k] - entry) / risk_dist
            if mfe >= TRIGGER_R:
                cand = h[k] - TRAIL_ATR * a
                if cand > stop:
                    stop = cand
        else:
            mfe = (entry - l[k]) / risk_dist
            if mfe >= TRIGGER_R:
                cand = l[k] + TRAIL_ATR * a
                if cand < stop:
                    stop = cand
    return None


def one(args):
    sym, picks, cache_dir = args
    f = Path(cache_dir) / f"{sym}.parquet"
    if not f.exists():
        return []
    try:
        df = pd.read_parquet(f)
    except Exception:
        return []
    if df.empty or len(df) < 2000:
        return []
    df = bt.prepare_data(df)
    df["chop"] = chop_series(df)
    o = df.open.values; h = df.high.values; l = df.low.values; c = df.close.values
    atr = df.atr.values; ema = df.ema.values; ch = df.chop.values; ts = df.start.values
    n = len(c)

    sigs = {}
    for s in bt.detect_signals(df):
        sigs.setdefault(s["type"], []).append(s)

    out = []
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
            if not (np.isfinite(atr[bos]) and atr[bos] > 0):
                continue

            long_ = side == "long"
            entry = o[e]
            risk_dist = atr[bos] * am                 # carried explicitly, never rebuilt
            sl = entry - risk_dist if long_ else entry + risk_dist
            tp = entry + risk_dist * rr if long_ else entry - risk_dist * rr
            if sl <= 0 or tp <= 0:
                continue

            # fixed-TP arm
            fixed = None
            for k in range(e, n):
                if long_:
                    hs, ht = l[k] <= sl, h[k] >= tp
                else:
                    hs, ht = h[k] >= sl, l[k] <= tp
                if hs or ht:
                    fixed = (-1.0 if hs else rr, k)
                    break
            if fixed is None:
                continue
            tr = resolve_trail(e, n, long_, entry, sl, tp, risk_dist, h, l, atr)
            if tr is None:
                continue

            out.append((ts[e], sym, dt, side, entry, sl, rr, am,
                        ch[bos] if np.isfinite(ch[bos]) else np.nan,
                        fixed[0], ts[fixed[1]], tr[0], ts[tr[1]]))
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
    print(f"[TRAIL] {len(jobs)} symbols, live config, fixed + s3_a1 arms", flush=True)

    rows = []
    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=4), 1):
            rows.extend(r)
            if i % 40 == 0:
                print(f"  {i}/{len(jobs)} · {len(rows):,}", flush=True)

    d = pd.DataFrame(rows, columns=[
        "entry_time", "symbol", "div_type", "side", "entry_price", "sl_price", "rr",
        "atr_mult", "chop_bos", "r_fixed", "exit_fixed", "r_trail", "exit_trail"])
    d["entry_time"] = pd.to_datetime(d["entry_time"])
    d["exit_fixed"] = pd.to_datetime(d["exit_fixed"])
    d["exit_trail"] = pd.to_datetime(d["exit_trail"])
    d = d[d.entry_time >= GEN_FROM].sort_values("entry_time").reset_index(drop=True)
    d["stop_frac"] = (d.entry_price - d.sl_price).abs() / d.entry_price

    b = pd.read_parquet(CACHE / "BTCUSDT.parquet").sort_values("start")
    b["ema200"] = b["close"].ewm(span=200, adjust=False).mean()
    bull = (b["close"] > b["ema200"]).shift(1).fillna(False); bull.index = b["start"]
    dd_ = b.set_index("start")["close"].resample("1D").last().dropna()
    imp = (dd_ / dd_.shift(30) - 1.0) > 0.10
    imp.index = imp.index + pd.Timedelta(days=1)
    imp_h = imp.reindex(pd.date_range(b["start"].min().floor("D"),
                                      b["start"].max().ceil("D") + pd.Timedelta(days=1),
                                      freq="h")).ffill().fillna(False)
    et = d.entry_time.dt.floor("h")
    d["btc_bull"] = et.map(bull.to_dict()).fillna(False).astype(bool)
    d["btc_impulse"] = et.map(imp_h.to_dict()).fillna(False).astype(bool)

    kept = d[~(d.chop_bos.notna() & (d.chop_bos >= CHOP_T))].reset_index(drop=True)
    kept.to_parquet(HERE / "universe_trail.parquet", index=False)

    print(f"[TRAIL] {len(kept):,} trades after ch[bos] gate "
          f"({d.entry_time.min().date()}..{d.entry_time.max().date()})")
    for nm, col in (("fixed", "r_fixed"), ("trail", "r_trail")):
        g = kept[col].mean()
        net = (kept[col] - 0.00242 / kept.stop_frac).mean()
        print(f"  {nm:>6}: gross {g:+.4f}  net@24.2bps {net:+.4f}  "
              f"WR {(kept[col] > 0).mean()*100:5.2f}%")
    dlt = (kept.r_trail - kept.r_fixed)
    print(f"  delta (trail - fixed): mean {dlt.mean():+.4f} R  "
          f"better {(dlt > 0).mean()*100:.1f}%  worse {(dlt < 0).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
