#!/usr/bin/env python3
"""The bot's REAL round-trip cost, measured from `exec_log` (live fills).

`exec_log` records, per executed trade: intended vs actual entry price (so entry slippage
is measured, not assumed), the notional, the exchange `fee_usd`, `funding_usd`, and the
realized R. That is everything needed to compute cost in basis points of notional without
a single assumption about Bybit's fee tier.

Read-only. The connection string is read from a file and never printed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def pct(s, ps=(1, 5, 25, 50, 75, 95, 99)):
    return {f"p{p}": float(np.nanpercentile(s, p)) for p in ps}


def main():
    url = Path(sys.argv[1]).read_text().strip()
    conn = psycopg2.connect(url)
    cur = conn.cursor()
    cur.execute("select * from exec_log")
    cols = [d[0] for d in cur.description]
    e = pd.DataFrame(cur.fetchall(), columns=cols)
    conn.close()
    OUT.mkdir(parents=True, exist_ok=True)
    e.to_parquet(OUT / "live_exec_log.parquet", index=False)

    for c in ("intended_entry", "actual_entry", "slippage_frac", "notional", "qty",
              "risk_usd", "fee_usd", "funding_usd", "realized_pnl", "r_result",
              "actual_exit", "intended_sl", "hold_hours", "fill_lag_sec", "equity",
              "atr", "atr_mult", "rr", "leverage"):
        if c in e:
            e[c] = pd.to_numeric(e[c], errors="coerce")

    print(f"exec_log rows: {len(e):,}   closed (exit_ts not null): {e.exit_ts.notna().sum():,}")
    print(f"date range: {e.ts.min()} .. {e.ts.max()}")

    cl = e[e.exit_ts.notna() & e.notional.notna() & (e.notional > 0)].copy()
    print(f"\nusing {len(cl):,} CLOSED trades with a recorded notional\n")

    # ── 1. entry slippage, measured directly ───────────────────────────────────
    print("=" * 78)
    print("1. ENTRY SLIPPAGE  (|actual_entry - intended_entry| / intended_entry)")
    print("=" * 78)
    sl_meas = (cl.actual_entry - cl.intended_entry).abs() / cl.intended_entry
    # signed: positive = filled WORSE than intended (paid up on a long, sold low on a short)
    signed = np.where(cl.side.str.lower() == "long",
                      (cl.actual_entry - cl.intended_entry) / cl.intended_entry,
                      (cl.intended_entry - cl.actual_entry) / cl.intended_entry)
    print(f"  n={len(cl)}  mean |slip| {sl_meas.mean()*1e4:7.2f} bps   "
          f"median {sl_meas.median()*1e4:6.2f} bps")
    print(f"  SIGNED (adverse +): mean {np.nanmean(signed)*1e4:7.2f} bps   "
          f"median {np.nanmedian(signed)*1e4:6.2f} bps")
    print(f"  percentiles |slip| bps: " +
          "  ".join(f"{k}={v*1e4:.1f}" for k, v in pct(sl_meas).items()))
    print(f"  recorded slippage_frac col: mean {cl.slippage_frac.mean()*1e4:.2f} bps")
    print(f"  fill lag: median {cl.fill_lag_sec.median():.0f}s  p95 {cl.fill_lag_sec.quantile(.95):.0f}s")

    # ── 2. exchange fees, measured directly ────────────────────────────────────
    print()
    print("=" * 78)
    print("2. EXCHANGE FEES  (fee_usd / notional) — this is the ROUND TRIP as booked")
    print("=" * 78)
    fee_bps = cl.fee_usd / cl.notional
    print(f"  n={fee_bps.notna().sum()}  mean {fee_bps.mean()*1e4:7.2f} bps   "
          f"median {fee_bps.median()*1e4:6.2f} bps")
    print(f"  percentiles bps: " + "  ".join(f"{k}={v*1e4:.1f}" for k, v in pct(fee_bps.dropna()).items()))
    print(f"  implied per-side: {fee_bps.mean()*1e4/2:.2f} bps "
          f"(Bybit published taker for USDT perps is 5.5 bps/side)")

    fund_bps = cl.funding_usd / cl.notional
    print(f"\n  funding: mean {fund_bps.mean()*1e4:+7.2f} bps of notional over a "
          f"median {cl.hold_hours.median():.1f}h hold")

    # ── 3. total round trip ────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("3. TOTAL ROUND-TRIP COST")
    print("=" * 78)
    entry_slip = np.nanmean(signed)
    fees = fee_bps.mean()
    fund = fund_bps.mean()
    # exit slippage is not directly recorded (no intended-exit for a market stop fill),
    # so it is inferred below; here assume symmetry with entry as the central case.
    total_sym = fees + 2 * entry_slip + fund
    print(f"  exchange fees (round trip, booked) {fees*1e4:7.2f} bps")
    print(f"  entry slippage (measured)          {entry_slip*1e4:7.2f} bps")
    print(f"  exit slippage (assumed = entry)    {entry_slip*1e4:7.2f} bps")
    print(f"  funding (mean, signed)             {fund*1e4:+7.2f} bps")
    print(f"  {'-'*44}")
    print(f"  TOTAL round trip                   {total_sym*1e4:7.2f} bps")

    # ── 4. independent estimator: stop-outs should realise exactly -1.0 R ───────
    print()
    print("=" * 78)
    print("4. CROSS-CHECK — stop-outs realise -1.0 R minus cost")
    print("=" * 78)
    st = cl[(cl.outcome.astype(str).str.lower().str.contains("loss|stop", na=False))].copy()
    st = st[st.intended_sl.notna() & (st.intended_sl > 0)]
    if len(st) >= 10:
        st["stop_frac"] = (st.intended_entry - st.intended_sl).abs() / st.intended_entry
        # realized R already includes cost; the excess loss beyond -1 is cost in R
        st["excess_r"] = -(st.r_result + 1.0)          # >0 means worse than -1R
        st["implied_bps"] = st.excess_r * st.stop_frac
        good = st[st.implied_bps.between(-0.01, 0.02)]
        print(f"  n={len(st)} stop-outs ({len(good)} within a sane range)")
        print(f"  mean excess loss beyond -1.0R : {st.excess_r.mean():+.4f} R")
        print(f"  => implied round trip          : {good.implied_bps.mean()*1e4:7.2f} bps "
              f"(median {good.implied_bps.median()*1e4:.2f})")
        print(f"  mean stop_frac on these trades : {st.stop_frac.mean()*100:.2f}% of price")
    else:
        print(f"  only {len(st)} stop-outs with a recorded stop — too few, skipped")

    # ── 5. what this means in R ────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("5. TRANSLATION TO cost_R  (cost_R = round_trip / stop_frac)")
    print("=" * 78)
    uni = pd.read_parquet(HERE / "universe_chopBOS.parquet")
    inv = (1.0 / uni.stop_frac)
    print(f"  backtest universe mean 1/stop_frac = {inv.mean():.1f}   "
          f"(median {np.median(inv):.1f})")
    print(f"  live exec_log     mean 1/stop_frac = "
          f"{(cl.intended_entry/(cl.intended_entry-cl.intended_sl).abs()).replace([np.inf,-np.inf],np.nan).mean():.1f}")
    print()
    print(f"  {'round trip':>12} {'mean cost_R':>12}")
    for bps in (11, 18, 25, 34.1, 45, total_sym * 1e4):
        print(f"  {bps:>10.1f}bps {(bps/1e4*inv).mean():>12.4f}")

    rows = [dict(component="fees_round_trip_bps", value=fees * 1e4),
            dict(component="entry_slippage_bps", value=entry_slip * 1e4),
            dict(component="funding_bps", value=fund * 1e4),
            dict(component="total_symmetric_bps", value=total_sym * 1e4),
            dict(component="n_closed_trades", value=len(cl))]
    pd.DataFrame(rows).to_csv(OUT / "live_measured_cost.csv", index=False)
    print(f"\nsaved -> {OUT/'live_measured_cost.csv'}, {OUT/'live_exec_log.parquet'}")


if __name__ == "__main__":
    main()
