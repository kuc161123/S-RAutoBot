#!/usr/bin/env python3
"""How many parameters can this strategy actually support? Global vs buckets vs per-symbol.

Every rule below is FITTED ON DESIGN DATA ONLY (entries resolving before 2026-02-01) and
then scored, once, on untouched data — in dollars, through the production engine, from
$1,500. The live config is included as the base; note it saw data through 2026-05-25, so
window A is partly in-sample FOR IT and window B is fair to everyone.

The rules differ only in how much per-symbol freedom they get:

    rule                     free parameter sets
    ONE GLOBAL                        1
    BY SIDE                           2   (long / short)
    BY DIV TYPE                       4
    BY LIQUIDITY TIER                 3   (terciles of median turnover)
    BY VOLATILITY TIER                3   (terciles of median ATR%)
    PER-SYMBOL (shrunk)            ~700
    LIVE CONFIG (current)           728

If fewer parameters do better out-of-sample, the per-symbol layer is fitting noise.
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
import backtest_3yr_walkforward as bt
import backtest_production_correct as P
from backtest_shadow_gate import LIVE as LIVE_SIM

ROOT = Path(__file__).parent
CACHE = ROOT / "cache_3yr_1h"
HOLDOUT_START = pd.Timestamp("2026-02-01")
DESIGN_START = pd.Timestamp("2023-06-01")
RR_GRID = [3.0, 5.0, 8.0, 10.0]
ATR_GRID = [1.0, 1.5, 2.0]
DIV_TYPES = ["REG_BULL", "REG_BEAR", "HID_BULL", "HID_BEAR"]
MAX_WAIT = bt.MAX_WAIT_CANDLES
SLIP = 0.0003
CHOP_THRESHOLD = 52.0
START_BAL = 1500.0
MIN_TRADES_SELECT = 40
SHRINK_K = 60.0


def chop_series(df, period=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    hi = df["high"].rolling(period).max(); lo = df["low"].rolling(period).min()
    return (100 * np.log10(tr.rolling(period).sum() / (hi - lo).replace(0, np.nan))
            / math.log10(period))


def replay(args):
    """Grid replay emitting PRODUCTION columns (raw prices, raw r_result)."""
    symbol, cache_dir = args
    f = Path(cache_dir) / f"{symbol}.parquet"
    if not f.exists():
        return [], None
    try:
        df = pd.read_parquet(f)
    except Exception:
        return [], None
    if df.empty or len(df) < 3000:
        return [], None

    df = bt.prepare_data(df)
    df["chop"] = chop_series(df)
    o = df["open"].to_numpy(); h = df["high"].to_numpy(); lo_ = df["low"].to_numpy()
    c = df["close"].to_numpy(); atr_a = df["atr"].to_numpy(); ema = df["ema"].to_numpy()
    chop_a = df["chop"].to_numpy(); ts = df["start"].to_numpy()
    n = len(c)

    meta = dict(symbol=symbol,
                turnover=float(df["turnover"].tail(8000).median())
                if "turnover" in df else 0.0,
                atr_pct=float((df["atr"] / df["close"]).tail(8000).median()))

    sigs = {}
    for s in bt.detect_signals(df):
        sigs.setdefault(s["type"], []).append(s)

    rows = []
    for dt in DIV_TYPES:
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
            cv = chop_a[e]
            if np.isfinite(cv) and cv >= CHOP_THRESHOLD:
                continue
            atr = atr_a[bos]
            if not np.isfinite(atr) or atr <= 0:
                continue
            entry = o[e]                       # RAW — engine applies its own costs
            for am in ATR_GRID:
                sl_d = atr * am
                if sl_d <= 0:
                    continue
                sl = entry - sl_d if side == "long" else entry + sl_d
                for rr in RR_GRID:
                    tp = entry + sl_d * rr if side == "long" else entry - sl_d * rr
                    r = None; xi = None
                    for k in range(e, n):
                        if side == "long":
                            hs, ht = lo_[k] <= sl, h[k] >= tp
                        else:
                            hs, ht = h[k] >= sl, lo_[k] <= tp
                        if not (hs or ht):
                            continue
                        r = -1.0 if hs else rr
                        xi = k
                        break
                    if r is None:
                        continue
                    rows.append((ts[e], ts[xi], entry, sl, r, side, symbol, dt, rr, am))
    return rows, meta


def fit_rules(design, meta):
    """All rules fitted on DESIGN ONLY. Returns {name: (mapping_fn, n_params)}."""
    d = design.merge(meta, on="symbol", how="left")
    d["liq_tier"] = pd.qcut(d.turnover.rank(method="first"), 3, labels=["lo", "mid", "hi"])
    d["vol_tier"] = pd.qcut(d.atr_pct.rank(method="first"), 3, labels=["lo", "mid", "hi"])
    rules = {}

    def best_by(keys):
        g = d.groupby(keys + ["rr", "atr_mult"]).r_result.agg(["mean", "count"])
        g = g[g["count"] >= 200]
        out = {}
        for idx, row in g.iterrows():
            k = idx[:len(keys)] if len(keys) > 1 else idx[0]
            rr, am = idx[-2], idx[-1]
            if k not in out or row["mean"] > out[k][1]:
                out[k] = ((rr, am), row["mean"])
        return {k: v[0] for k, v in out.items()}

    g = d.groupby(["rr", "atr_mult"]).r_result.mean()
    rules["ONE GLOBAL"] = ({"__global__": g.idxmax()}, 1)
    rules["BY SIDE"] = (best_by(["side"]), 2)
    rules["BY DIV TYPE"] = (best_by(["div_type"]), 4)
    rules["BY LIQUIDITY TIER"] = (best_by(["liq_tier"]), 3)
    rules["BY VOLATILITY TIER"] = (best_by(["vol_tier"]), 3)

    gs = d.groupby(["symbol", "div_type", "rr", "atr_mult"]).r_result.agg(["mean", "count"])
    gs = gs[gs["count"] >= MIN_TRADES_SELECT]
    gm = d.r_result.mean()
    w = gs["count"] / (gs["count"] + SHRINK_K)
    gs["score"] = w * gs["mean"] + (1 - w) * gm
    ps = {}
    for (sym, dv, rr, am), row in gs.iterrows():
        k = (sym, dv)
        if k not in ps or row["score"] > ps[k][1]:
            ps[k] = ((rr, am), row["score"])
    rules["PER-SYMBOL (shrunk)"] = ({k: v[0] for k, v in ps.items()}, len(ps))
    return rules


def select(seg, name, mapping, meta):
    s = seg.merge(meta, on="symbol", how="left")
    if name == "ONE GLOBAL":
        rr, am = mapping["__global__"]
        return s[(s.rr == rr) & (s.atr_mult == am)]
    if name == "BY SIDE":
        keys = s.side
    elif name == "BY DIV TYPE":
        keys = s.div_type
    elif name == "BY LIQUIDITY TIER":
        keys = pd.qcut(s.turnover.rank(method="first"), 3, labels=["lo", "mid", "hi"])
    elif name == "BY VOLATILITY TIER":
        keys = pd.qcut(s.atr_pct.rank(method="first"), 3, labels=["lo", "mid", "hi"])
    else:
        keys = list(zip(s.symbol, s.div_type))
    want = [mapping.get(k, (None, None)) for k in keys]
    rr_w = np.array([x[0] for x in want], dtype=object)
    am_w = np.array([x[1] for x in want], dtype=object)
    return s[(s.rr.values == rr_w) & (s.atr_mult.values == am_w)]


def dollars(sel, chop, label):
    if sel.empty:
        return dict(final=START_BAL, roi=0.0, dd=0.0, trades=0, pf=0.0)
    cols = ["entry_time", "exit_time", "entry_price", "sl_price", "r_result",
            "side", "symbol", "btc_bull", "btc_impulse"]
    df = sel[cols].sort_values("entry_time").reset_index(drop=True)
    kw = dict(LIVE_SIM, starting_balance=START_BAL,
              btc_bull_col="btc_bull", btc_short_col="btc_impulse")
    P.STARTING_BALANCE = START_BAL
    r = P.run_simulation(df, chop, **kw)
    t = r["entered_trades"]
    gp = sum(x["pnl"] for x in t if x["pnl"] > 0)
    gl = abs(sum(x["pnl"] for x in t if x["pnl"] < 0))
    return dict(final=r["final_effective"], roi=r["final_effective"] / START_BAL - 1,
                dd=r["max_dd_pct"], trades=len(t), pf=(gp / gl) if gl else float("inf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config.yaml"))["symbols"]
    live_map = {}
    syms = []
    for s, sc in cfg.items():
        if not (sc or {}).get("enabled", True):
            continue
        for c in (sc or {}).get("configs", []) or []:
            live_map[(s, c["divergence_type"])] = (float(c["rr"]), float(c["atr_mult"]))
        if (CACHE / f"{s}.parquet").exists():
            syms.append(s)
    syms.sort()
    if a.limit:
        syms = syms[: a.limit]
    print(f"[BUCKET] live universe: {len(syms)} symbols · {len(live_map)} live picks", flush=True)

    rows, metas = [], []
    with mp.Pool(a.workers) as pool:
        for i, (r, m) in enumerate(pool.imap_unordered(
                replay, [(s, str(CACHE)) for s in syms], chunksize=4), 1):
            rows.extend(r)
            if m:
                metas.append(m)
            if i % 50 == 0:
                print(f"  {i}/{len(syms)} · {len(rows):,} trades", flush=True)

    d = pd.DataFrame(rows, columns=["entry_time", "exit_time", "entry_price", "sl_price",
                                    "r_result", "side", "symbol", "div_type", "rr",
                                    "atr_mult"])
    d["entry_time"] = pd.to_datetime(d["entry_time"])
    d["exit_time"] = pd.to_datetime(d["exit_time"])
    d = d[d.entry_time >= DESIGN_START]
    meta = pd.DataFrame(metas)

    # BTC overlay state (causal), needed by the production engine
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

    design = d[d.exit_time < HOLDOUT_START]
    print(f"[BUCKET] {len(d):,} grid trades · design {len(design):,}")

    rules = fit_rules(design, meta)
    rules["LIVE CONFIG (current)"] = (live_map, len(live_map))
    print("\nFITTED ON DESIGN DATA ONLY:")
    for nm, (mp_, npar) in rules.items():
        if nm == "ONE GLOBAL":
            print(f"  {nm:<24} {npar:>5} params  -> rr={mp_['__global__'][0]:.0f} "
                  f"atr={mp_['__global__'][1]:.1f}")
        elif nm in ("BY SIDE", "BY DIV TYPE"):
            print(f"  {nm:<24} {npar:>5} params  -> " +
                  ", ".join(f"{k}:rr{v[0]:.0f}/a{v[1]:.1f}" for k, v in sorted(mp_.items())))
        else:
            print(f"  {nm:<24} {npar:>5} params")

    chop = P.load_chop_data(sorted(d.symbol.unique()))
    for wlbl, lo, note in [
        ("A) 2026-02-01 -> today (6mo)", HOLDOUT_START,
         "generous to LIVE — it saw data to 2026-05-25"),
        ("B) 2026-05-25 -> today (2mo)", pd.Timestamp("2026-05-25"),
         "genuinely out-of-sample for EVERY rule")]:
        seg = d[d.entry_time >= lo]
        print(f"\n{'=' * 96}\n{wlbl}   ${START_BAL:,.0f} start   ({note})\n{'=' * 96}")
        print(f"  {'rule':<26}{'params':>7}{'final $':>10}{'ROI':>9}{'maxDD':>8}"
              f"{'PF':>7}{'trades':>8}")
        out = []
        for nm, (mp_, npar) in rules.items():
            res = dollars(select(seg, nm, mp_, meta), chop, nm)
            out.append((nm, npar, res))
            print(f"  {nm:<26}{npar:>7}{res['final']:>10,.0f}{res['roi']:>9.1%}"
                  f"{res['dd']:>7.1f}%{res['pf']:>7.2f}{res['trades']:>8,}", flush=True)
        print(f"  {'never trade':<26}{0:>7}{START_BAL:>10,.0f}{0.0:>9.1%}"
              f"{0.0:>7.1f}%{'—':>7}{0:>8}")
        base = next(r for n, _, r in out if n == "LIVE CONFIG (current)")
        print(f"\n  vs current base (${base['final']:,.0f}):")
        for nm, npar, r in sorted(out, key=lambda x: -x[2]["final"]):
            if nm == "LIVE CONFIG (current)":
                continue
            print(f"    {nm:<26} {r['final'] - base['final']:>+9,.0f}  "
                  f"({npar} params vs {base['trades'] and len(live_map)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
