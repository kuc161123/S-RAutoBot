#!/usr/bin/env python3
"""Protection-ablation matrix over a recent window, using the production engine.

Answers: is the bot losing because the STRATEGY's edge has gone, or because the
PROTECTION STACK is mis-tuned for a small account? Runs the same trade universe
through backtest_production_correct.run_simulation under many overlay combinations.

Every scenario shares one universe, so differences are attributable purely to the
overlay settings (and, with --arm, to BOS timing).

    python3 backtest_protection_matrix.py --universe universe_live_8mo.csv \
        --start 2025-11-25 --balance 1876
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import backtest_production_correct as P

ROOT = Path(__file__).parent

# Live config.yaml values as of 2026-07-25
LIVE_TAPER = [(1500, 0.003), (3000, 0.0028), (5000, 0.0025), (8000, 0.0022),
              (12000, 0.002), (20000, 0.0017), (40000, 0.0014)]
OLD_TAPER = [(1500, 0.007), (2000, 0.006), (3000, 0.0055), (5000, 0.005),
             (8000, 0.0045), (12000, 0.004), (20000, 0.0035), (40000, 0.003)]
LIVE_BASE_RISK = 0.003
OLD_BASE_RISK = 0.012


def scenarios():
    """(name, group, kwargs) — kwargs go straight to run_simulation."""
    live = dict(scenario="production", base_risk=LIVE_BASE_RISK, custom_taper=LIVE_TAPER,
                taper_basis="wallet", size_basis="equity",
                net_dir_cap=0.10, open_risk_cap=0.30,
                short_gate=True, long_boost=1.3, overlay_min_balance=0.0)
    S = []

    # --- the two reference points -------------------------------------------
    S.append(("LIVE (everything on, 0.3%)", "reference", dict(live)))
    S.append(("NAKED (no protections at all)", "reference",
              dict(scenario="no_filters", base_risk=LIVE_BASE_RISK,
                   custom_taper=LIVE_TAPER, taper_basis="wallet", size_basis="equity")))

    # --- remove ONE protection at a time from LIVE ---------------------------
    for name, patch in [
        ("- CHOP filter",        dict(scenario="production_nochop")),
        ("- regime sizing",      dict(scenario="chop_only")),
        ("- net-dir cap",        dict(net_dir_cap=None)),
        ("- gross risk cap",     dict(open_risk_cap=None)),
        ("- BTC short-gate",     dict(short_gate=False)),
        ("- long bull-boost",    dict(long_boost=1.0)),
    ]:
        k = dict(live); k.update(patch)
        S.append((f"LIVE {name}", "ablation", k))

    # --- add ONE protection at a time to NAKED -------------------------------
    naked = dict(scenario="no_filters", base_risk=LIVE_BASE_RISK,
                 custom_taper=LIVE_TAPER, taper_basis="wallet", size_basis="equity")
    for name, patch in [
        ("+ CHOP filter",     dict(scenario="chop_only")),
        ("+ regime sizing",   dict(scenario="production_nochop")),
        ("+ net-dir cap 10%", dict(net_dir_cap=0.10)),
        ("+ gross cap 30%",   dict(open_risk_cap=0.30)),
        ("+ short-gate",      dict(short_gate=True)),
    ]:
        k = dict(naked); k.update(patch)
        S.append((f"NAKED {name}", "buildup", k))

    # --- risk-level sweep on the live stack ----------------------------------
    for br, tap, lbl in [(OLD_BASE_RISK, OLD_TAPER, "1.2% (pre-Jul21)"),
                         (0.008, LIVE_TAPER, "0.8%"),
                         (0.005, LIVE_TAPER, "0.5%"),
                         (LIVE_BASE_RISK, LIVE_TAPER, "0.3% (current)"),
                         (0.0015, LIVE_TAPER, "0.15%")]:
        k = dict(live); k.update(base_risk=br, custom_taper=tap)
        S.append((f"LIVE @ risk {lbl}", "risk", k))

    # --- net-dir cap sweep ----------------------------------------------------
    for cap in [0.05, 0.10, 0.15, 0.25]:
        k = dict(live); k["net_dir_cap"] = cap
        S.append((f"LIVE @ net-dir cap {cap:.0%}", "netdir", k))

    # --- concurrency cap (the bot has NONE today) -----------------------------
    for mc in [10, 20, 30]:
        k = dict(live); k["max_concurrent"] = mc
        S.append((f"LIVE @ max {mc} concurrent", "concurrency", k))

    # --- daily loss halt (config key exists but is dead code) -----------------
    for dh in [-5.0, -10.0]:
        k = dict(live); k["daily_halt_r"] = dh
        S.append((f"LIVE @ daily halt {dh:.0f}R", "dailyhalt", k))

    return S


def metrics(res, start_bal):
    t = res["entered_trades"]
    n = len(t)
    if n == 0:
        return dict(trades=0, wr=0, avg_r=0, pnl=0, roi=0, pf=0,
                    dd=res["max_dd_pct"], dd_mtm=res["max_dd_mtm_pct"], final=start_bal)
    pnl = sum(x["pnl"] for x in t)
    gp = sum(x["pnl"] for x in t if x["pnl"] > 0)
    gl = abs(sum(x["pnl"] for x in t if x["pnl"] < 0))
    return dict(
        trades=n,
        wr=sum(1 for x in t if x["r_result"] > 0) / n,
        avg_r=float(np.mean([x["r_result"] for x in t])),
        pnl=pnl,
        roi=(res["final_effective"] - start_bal) / start_bal,
        pf=(gp / gl) if gl > 0 else float("inf"),
        dd=res["max_dd_pct"],
        dd_mtm=res["max_dd_mtm_pct"],
        final=res["final_effective"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--balance", type=float, default=1876.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--cache-dir", dest="cache_dir", default="cache_3yr_1h")
    ap.add_argument("--out", default="protection_matrix_results.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.universe, parse_dates=["entry_time", "exit_time"])
    if a.start:
        df = df[df.entry_time >= pd.Timestamp(a.start)]
    if a.end:
        df = df[df.entry_time <= pd.Timestamp(a.end)]
    df = df.sort_values("entry_time").reset_index(drop=True)
    if df.empty:
        print("no trades in window")
        return 1

    print(f"[MATRIX] {a.label or a.universe}")
    print(f"[MATRIX] {len(df)} signals · {df.entry_time.min()} .. {df.entry_time.max()}")
    print(f"[MATRIX] raw pre-cost edge: WR {(df.r_result>0).mean():.2%} "
          f"avgR {df.r_result.mean():+.4f}")
    print(f"[MATRIX] starting balance ${a.balance:,.0f}")

    syms = sorted(df.symbol.unique())
    P.STARTING_BALANCE = a.balance
    chop_map = P.load_chop_data(syms)

    # === BTC overlay state, mirroring the LIVE signals exactly ===
    #   btc_bull    : 1H close > 200 EMA         -> drives long_bull_boost (bot.py:1690)
    #   btc_impulse : trailing 30d daily return  -> drives btc_short_gate v2 (bot.py:706)
    df["btc_bull"] = False
    df["btc_impulse"] = False
    btc_f = ROOT / a.cache_dir / "BTCUSDT.parquet"
    if btc_f.exists():
        b = pd.read_parquet(btc_f).sort_values("start").reset_index(drop=True)
        b["ema200"] = b["close"].ewm(span=200, adjust=False).mean()
        # CAUSAL: a trade entering on candle i fills at open[i], so the newest BTC close
        # that exists at that moment is close[i-1]. Using close[i] would leak ~1h.
        b["bull"] = (b["close"] > b["ema200"]).shift(1).fillna(False)
        bull = b.set_index("start")["bull"]

        # CAUSAL: resample('1D') labels a bucket by its START but .last() is the price at
        # its END, so reindexing straight to hours would make day D's close visible at
        # 00:00 on day D (up to 23h of lookahead). Shift the label forward one day so a
        # day's close only takes effect from 00:00 the NEXT day — this mirrors the live
        # bot, which reads closes[1] and deliberately skips the unclosed current day
        # (autobot/core/bot.py:729-732).
        d_ = b.set_index("start")["close"].resample("1D").last().dropna()
        ret30 = (d_ / d_.shift(30) - 1.0)
        imp_d = (ret30 > 0.10)
        imp_d.index = imp_d.index + pd.Timedelta(days=1)
        imp_h = imp_d.reindex(
            pd.date_range(b["start"].min().floor("D"),
                          b["start"].max().ceil("D") + pd.Timedelta(days=1), freq="h")
        ).ffill().fillna(False)

        et = df.entry_time.dt.floor("h")
        df["btc_bull"] = et.map(bull.reindex(bull.index).to_dict()).fillna(False).astype(bool)
        df["btc_impulse"] = et.map(imp_h.to_dict()).fillna(False).astype(bool)
        print(f"[MATRIX] BTC state: bull on {df.btc_bull.mean():.1%} of entries, "
              f"impulse-bull on {df.btc_impulse.mean():.1%}")
    else:
        print("[MATRIX] WARNING: no BTCUSDT cache — short-gate/long-boost inert")

    rows = []
    for name, group, kw in scenarios():
        kw = dict(kw)
        kw.setdefault("starting_balance", a.balance)
        if kw.get("short_gate") or kw.get("long_boost", 1.0) != 1.0:
            kw["btc_bull_col"] = "btc_bull"        # long-boost trigger (1H EMA200)
            kw["btc_short_col"] = "btc_impulse"    # short-gate v2 trigger (30d return)
        try:
            res = P.run_simulation(df, chop_map, **kw)
        except TypeError as e:
            print(f"  SKIP {name}: {e}")
            continue
        except Exception as e:
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
            continue
        m = metrics(res, a.balance)
        m.update(name=name, group=group,
                 chop_blocked=len(res.get("chop_blocked") or []),
                 risk_capped=res.get("risk_capped", 0),
                 btc_blocked=res.get("btc_blocked", 0))
        rows.append(m)
        pf = "inf" if m["pf"] == float("inf") else f"{m['pf']:.2f}"
        print(f"  {name:<34} {m['trades']:>6}t  WR {m['wr']:>6.2%}  "
              f"avgR {m['avg_r']:>+7.3f}  ${m['final']:>10,.0f}  "
              f"ROI {m['roi']:>+8.1%}  PF {pf:>6}  DD {m['dd']:>5.1f}%", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print(f"\n[MATRIX] -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
