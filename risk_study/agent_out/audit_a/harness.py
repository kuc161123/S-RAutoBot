#!/usr/bin/env python3
"""AUDIT-A shared harness: run the production engine on the CURRENT global rr10/atr3
config with one parameter changed at a time, in-sample and holdout, with restart-fresh
compounding for each window (matches risk_study/search_roidd.py's sim()).

size_basis='wallet' throughout (methodology point 7 -- no lookahead through equity
marking). Cost 24.2bps. IS/holdout split 2026-05-25. Shadow halt 21d(-300/+200) applied
as the baseline overlay (matches search_roidd.py's own baseline definition) since that is
"the current configuration" per AUDIT_PROMPT.md.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path("/Users/lualakol/AutoTrading Bot")
RS = ROOT / "risk_study"
HERE = RS / "agent_out" / "audit_a"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RS))

import backtest_production_correct as P  # noqa: E402
from backtest_shadow_gate import LIVE as LIVE_SIM, LIVE_TAPER, gate_series  # noqa: E402
import compare_bots as C  # noqa: E402
from variations import with_halt  # noqa: E402

P.ROUND_TRIP_COST = 0.00242
START = 1500.0
IS_END = pd.Timestamp("2026-05-25")
COST = 0.00242

# base kwargs = the CURRENT live config (2026-08-12), size_basis forced to wallet
BASE_KW = dict(
    scenario="production",
    starting_balance=START, base_risk=0.003, custom_taper=LIVE_TAPER,
    btc_bull_col="btc_bull", btc_short_col="btc_impulse",
    taper_basis="wallet", size_basis="wallet",
    net_dir_cap=0.10, open_risk_cap=0.10, open_risk_mode="block",
    short_gate=False, long_boost=1.3, overlay_min_balance=0.0,
)

_CHOP_ORIG = dict(P.CHOP_THRESHOLDS)
_REGIME_ORIG = P.get_regime


def reset_globals():
    P.CHOP_THRESHOLDS = dict(_CHOP_ORIG)
    P.get_regime = _REGIME_ORIG


def make_get_regime(window=20, tiers=None):
    """tiers: list of (wr_min, avg_r_min_and, avg_r_min_or, label, mult) reproducing the
    live 4-tier cascade, parameterised by window length. Default tiers match bot.py."""
    if tiers is None:
        tiers = [
            (0.18, 0.15, 0.10, "favorable", 1.0, "cautious", 0.5),
            (0.18, None, 0.10, "cautious", 0.5, "adverse", 0.25),
        ]

    def get_regime(recent_trades):
        n = len(recent_trades)
        if n < 10:
            return "critical", 0.1
        w = recent_trades[-window:]
        wr = sum(1 for t in w if t["r"] > 0) / len(w)
        avg_r = sum(t["r"] for t in w) / len(w)
        if wr >= 0.18 and avg_r >= 0.15:
            return "favorable", 1.0
        elif wr >= 0.18 or avg_r >= 0.10:
            return "cautious", 0.5
        elif wr >= 0.10 or avg_r >= -0.5:
            return "adverse", 0.25
        else:
            return "critical", 0.1

    return get_regime


def load_chop_cache():
    uni = pd.read_parquet(RS / "universe_trail.parquet")
    return P.load_chop_data(sorted(uni.symbol.unique()))


def load_gated_halted():
    """The already-CHOP-gated (<52 flat, matches the 728 live pairs) file, with the
    21d shadow halt applied -- what search_roidd.py / variations.py call 'the baseline'."""
    d = with_halt(C.load("uni_glob_rr10_am3_same.parquet", "r_result", "exit_time"),
                  21, -300, 200)
    return d


def load_ungated_halted():
    """Ungated (all chop_bos values) global rr10/atr3 population, 728 live pairs, with
    the 21d shadow halt applied. Needed to test CHOP thresholds looser than the live 52,
    or no gate at all, through the engine -- lookup_chop() applies the gate internally so
    the input population must not already be pre-filtered."""
    d = pd.read_parquet(HERE / "ungated_glob_rr10_am3.parquet")
    d = with_halt(d, 21, -300, 200)
    return d


def sim(d, chop, t0=None, t1=None, **over):
    sub = d
    if t0 is not None:
        sub = sub[sub.entry_time >= t0]
    if t1 is not None:
        sub = sub[sub.entry_time < t1]
    if len(sub) < 50:
        return None
    kw = dict(BASE_KW)
    kw.update(over)
    P.STARTING_BALANCE = START
    cols = ["entry_time", "exit_time", "entry_price", "sl_price", "r_result",
            "side", "symbol", "btc_bull", "btc_impulse"]
    r = P.run_simulation(sub[cols].reset_index(drop=True), chop, **kw)
    t = pd.DataFrame(r["entered_trades"])
    if t.empty:
        return dict(final=START, roi=0.0, dd=0.0, n=0, avg_r=np.nan, roidd=np.nan)
    t = t.sort_values("exit_time")
    c = pd.Series((START + t.pnl.cumsum()).values, index=pd.to_datetime(t.exit_time.values))
    pk = c.cummax()
    dd = float(((pk - c) / pk).max() * 100)
    roi = (r["final_effective"] / START - 1) * 100
    return dict(final=r["final_effective"], roi=roi, dd=dd, n=len(t),
                avg_r=float(t.r_result.mean()),
                roidd=roi / dd if dd > 0 else np.nan)


def windows_6mo(d):
    start = d.entry_time.min().normalize().replace(day=1)
    end = d.entry_time.max()
    edges = pd.date_range(start, end + pd.Timedelta(days=180), freq="6MS")
    out = []
    for i in range(len(edges) - 1):
        w0, w1 = edges[i], edges[i + 1]
        if len(d[(d.entry_time >= w0) & (d.entry_time < w1)]) >= 150:
            out.append((w0, w1))
    return out


def run_variant(name, d, chop, cfg_diff, note="", **over):
    """Full record: IS, HOLD, and per-6mo-window avg_r, each a FRESH restart sim.

    NOTE: does NOT reset_globals() itself -- callers who need to monkeypatch
    P.CHOP_THRESHOLDS or P.get_regime must do so BEFORE calling this, and call
    reset_globals() themselves before and after their own sweep.
    """
    is_ = sim(d, chop, t1=IS_END, **over)
    ho = sim(d, chop, t0=IS_END, **over)
    row = dict(name=name, cfg_diff=cfg_diff, note=note,
               is_final=is_["final"], is_roi=is_["roi"], is_dd=is_["dd"], is_n=is_["n"],
               is_avg_r=is_["avg_r"], is_roidd=is_["roidd"],
               hold_final=ho["final"], hold_roi=ho["roi"], hold_dd=ho["dd"], hold_n=ho["n"],
               hold_avg_r=ho["avg_r"])
    wins = 0
    tot = 0
    win_avgrs = []
    for (w0, w1) in windows_6mo(d):
        r = sim(d, chop, t0=w0, t1=w1, **over)
        if r is None or r["n"] < 20:
            continue
        tot += 1
        win_avgrs.append(r["avg_r"])
        # 'wins' recorded relative to baseline outside this fn; store raw avg_r list
    row["window_avg_rs"] = ";".join(f"{x:.4f}" for x in win_avgrs)
    row["n_windows"] = tot
    print(f"  {name:<40} IS ${is_['final']:>10,.0f} roi{is_['roi']:>+8.1f}% dd{is_['dd']:>6.1f}%  "
          f"n={is_['n']:>6}  avgR{is_['avg_r']:>+.3f}  |  HOLD ${ho['final']:>9,.0f} "
          f"roi{ho['roi']:>+7.1f}% dd{ho['dd']:>6.1f}% n={ho['n']:>5} avgR{ho['avg_r']:>+.3f}",
          flush=True)
    reset_globals()
    return row
