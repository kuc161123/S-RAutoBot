#!/usr/bin/env python3
"""
AGENT-CMP — independent reproduction of the six-way bot comparison.

Written from scratch. Does NOT import backtest_production_correct.py or anything
under risk_study/ other than the input parquet files themselves. This is a plain
event-driven portfolio simulator over pre-computed per-trade outcome rows.

INPUTS (risk_study/*.parquet), one row per signal that reached BOS + entry, already
filtered by the flat CHOP>=52 gate at generation time (see build_universe_trail.py /
build_global_trail.py headers — verified by inspection, not re-derived here):

  OLD-BASE    universe_trail.parquet            r_fixed  / exit_fixed
  CURRENT     universe_trail.parquet            r_trail  / exit_trail
  GLOBAL      uni_glob_rr10_am3_same.parquet    r_result / exit_time
  GLOBAL-FIX  uni_globfix_rr10_am3_same.parquet r_result / exit_time
  GLB-10/2    uni_glob_rr10_am2_same.parquet    r_result / exit_time
  GLB-8/2     uni_glob_rr8_am2_same.parquet     r_result / exit_time

All r_* columns are GROSS R (before trading cost). Cost is applied here as
    net_r = r - 0.00242 / stop_frac
per the live cost model (24.2bps round-trip friction expressed in R via the stop
distance fraction of price), matching the number quoted throughout this repo's
research artifacts (e.g. build_universe_trail.py's own printed net@24.2bps stat).

SIMULATOR MECHANICS — see module docstring sections below for the exact gates.
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RS = ROOT / "risk_study"
OUT = Path(__file__).resolve().parent

START_BALANCE = 1500.0
BASE_RISK = 0.003          # not directly used; taper table below is authoritative
WITHDRAW_CEILING = 50000.0
COST_BPS_R = 0.00242        # net_r = r - COST_BPS_R / stop_frac
NET_DIR_CAP = 0.10
GROSS_RISK_CAP = 0.30
LONG_BULL_BOOST = 1.3

TAPER = [
    (1500.0, 0.003), (3000.0, 0.0028), (5000.0, 0.0025), (8000.0, 0.0022),
    (12000.0, 0.002), (20000.0, 0.0017), (40000.0, 0.0014),
]


def taper_rate(wallet: float) -> float:
    if wallet < 1500.0:
        return 0.003
    rate = 0.003
    for thresh, r in TAPER:
        if wallet >= thresh:
            rate = r
        else:
            break
    return rate


def regime_mult(closed_r: deque) -> float:
    n = len(closed_r)
    if n < 10:
        return 0.1
    arr = np.fromiter(closed_r, dtype=float, count=n)
    wr = float((arr > 0).mean())
    avg_r = float(arr.mean())
    if wr >= 0.18 and avg_r >= 0.15:
        return 1.0
    if wr >= 0.18 or avg_r >= 0.10:
        return 0.5
    if wr >= 0.10 or avg_r >= -0.5:
        return 0.25
    return 0.1


ARMS = [
    ("OLD-BASE",   "universe_trail.parquet",            "r_fixed",  "exit_fixed"),
    ("CURRENT",    "universe_trail.parquet",            "r_trail",  "exit_trail"),
    ("GLOBAL",     "uni_glob_rr10_am3_same.parquet",     "r_result", "exit_time"),
    ("GLOBAL-FIX", "uni_globfix_rr10_am3_same.parquet",  "r_result", "exit_time"),
    ("GLB-10/2",   "uni_glob_rr10_am2_same.parquet",     "r_result", "exit_time"),
    ("GLB-8/2",    "uni_glob_rr8_am2_same.parquet",      "r_result", "exit_time"),
]


def load_trades(fname: str, r_col: str, exit_col: str) -> pd.DataFrame:
    df = pd.read_parquet(RS / fname)
    need = ["entry_time", "symbol", "div_type", "side", "stop_frac",
            "btc_bull", "btc_impulse", r_col, exit_col]
    df = df[need].copy()
    df.columns = ["entry_time", "symbol", "div_type", "side", "stop_frac",
                  "btc_bull", "btc_impulse", "r", "exit_time"]
    df = df.dropna(subset=["entry_time", "exit_time", "r", "stop_frac"])
    df = df[df["stop_frac"] > 0].reset_index(drop=True)
    df["trade_id"] = df.index
    return df


def simulate(df: pd.DataFrame):
    """Event-driven single pass. Returns (closed_trades_df, blocked_counts)."""
    n = len(df)

    ev_time = np.concatenate([df["entry_time"].values, df["exit_time"].values])
    ev_kind = np.concatenate([np.zeros(n, dtype=np.int8), np.ones(n, dtype=np.int8)])  # 0=entry,1=exit
    ev_tid = np.concatenate([df["trade_id"].values, df["trade_id"].values])
    # deterministic tiebreak key (symbol, side, div_type) beyond kind
    sym_key = np.concatenate([df["symbol"].values, df["symbol"].values])
    side_key = np.concatenate([df["side"].values, df["side"].values])

    order = np.lexsort((side_key, sym_key, ev_kind, ev_time))
    # lexsort: last key is primary -> primary = ev_time, then ev_kind (entries=0 before exits=1), then sym, side

    rows = df.set_index("trade_id")

    wallet = START_BALANCE
    cum_pnl = 0.0
    total_withdrawn = 0.0

    open_positions: dict[int, dict] = {}
    open_symbol_side: set[tuple[str, str]] = set()
    open_long_risk = 0.0
    open_short_risk = 0.0

    closed_r_window: deque = deque(maxlen=20)

    blocked = {"anti_pyramid": 0, "short_impulse_gate": 0, "net_dir_cap": 0, "gross_risk_cap": 0}
    n_opened = 0

    closed_records = []  # one row per CLOSED trade, in exit chronological order
    eps = 1e-9

    for idx in order:
        tid = int(ev_tid[idx])
        kind = int(ev_kind[idx])
        row = rows.loc[tid]

        if kind == 0:  # ENTRY
            symbol = row["symbol"]
            side = row["side"]

            if (symbol, side) in open_symbol_side:
                blocked["anti_pyramid"] += 1
                continue
            if side == "short" and bool(row["btc_impulse"]):
                blocked["short_impulse_gate"] += 1
                continue

            eq = wallet
            t_rate = taper_rate(wallet)
            r_mult = regime_mult(closed_r_window)
            risk_frac = t_rate * r_mult
            risk_usd = risk_frac * eq
            if side == "long" and bool(row["btc_bull"]):
                risk_usd *= LONG_BULL_BOOST

            new_long = open_long_risk + (risk_usd if side == "long" else 0.0)
            new_short = open_short_risk + (risk_usd if side == "short" else 0.0)
            if abs(new_long - new_short) > NET_DIR_CAP * eq + eps:
                blocked["net_dir_cap"] += 1
                continue

            gross = open_long_risk + open_short_risk + risk_usd
            if gross > GROSS_RISK_CAP * eq + eps:
                blocked["gross_risk_cap"] += 1
                continue

            open_positions[tid] = dict(
                risk_usd=risk_usd, side=side, symbol=symbol,
                r=float(row["r"]), stop_frac=float(row["stop_frac"]),
                entry_time=row["entry_time"],
            )
            open_symbol_side.add((symbol, side))
            if side == "long":
                open_long_risk += risk_usd
            else:
                open_short_risk += risk_usd
            n_opened += 1

        else:  # EXIT
            pos = open_positions.pop(tid, None)
            if pos is None:
                continue  # never opened (blocked at entry)

            open_symbol_side.discard((pos["symbol"], pos["side"]))
            if pos["side"] == "long":
                open_long_risk -= pos["risk_usd"]
            else:
                open_short_risk -= pos["risk_usd"]

            net_r = pos["r"] - COST_BPS_R / pos["stop_frac"]
            pnl = net_r * pos["risk_usd"]

            wallet += pnl
            cum_pnl += pnl
            withdrawn_now = 0.0
            if wallet > WITHDRAW_CEILING:
                withdrawn_now = wallet - WITHDRAW_CEILING
                total_withdrawn += withdrawn_now
                wallet = WITHDRAW_CEILING

            closed_r_window.append(net_r)

            closed_records.append(dict(
                trade_id=tid, symbol=pos["symbol"], side=pos["side"],
                entry_time=pos["entry_time"], exit_time=row["exit_time"],
                r_gross=pos["r"], net_r=net_r, risk_usd=pos["risk_usd"],
                pnl=pnl, wallet_after=wallet, withdrawn_now=withdrawn_now,
                total_withdrawn=total_withdrawn,
                trading_equity=START_BALANCE + cum_pnl,
            ))

    assert len(open_positions) == 0, (
        f"{len(open_positions)} positions never closed — event ordering bug"
    )

    closed_df = pd.DataFrame(closed_records)
    return closed_df, blocked, n_opened


def max_drawdown_pct(equity_series: pd.Series) -> float:
    """Peak-to-trough on a monotone-in-time equity curve, as a % of the running peak."""
    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak
    return float(-dd.min() * 100.0)


def month_end_table(closed_df: pd.DataFrame, months: pd.PeriodIndex) -> pd.Series:
    """trading_equity value as of each month end (last close <= month end), ffilled."""
    s = closed_df.set_index("exit_time")["trading_equity"].sort_index()
    out = {}
    last = START_BALANCE
    for m in months:
        end = m.end_time
        sub = s[s.index <= end]
        if len(sub) > 0:
            last = sub.iloc[-1]
        out[str(m)] = last
    return pd.Series(out)


def monthly_returns(monthly_eq: pd.Series) -> pd.Series:
    vals = monthly_eq.values.astype(float)
    prev = np.concatenate([[START_BALANCE], vals[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = (vals - prev) / prev
    return pd.Series(ret * 100.0, index=monthly_eq.index)


def main():
    months = pd.period_range("2023-06", "2026-07", freq="M")

    summary_rows = []
    monthly_cols = {}
    monthly_ret_cols = {}
    all_notes = []

    for name, fname, r_col, exit_col in ARMS:
        df = load_trades(fname, r_col, exit_col)
        closed, blocked, n_opened = simulate(df)
        closed = closed.sort_values("exit_time").reset_index(drop=True)

        eq_curve = closed["trading_equity"]
        final_eq = float(eq_curve.iloc[-1]) if len(eq_curve) else START_BALANCE
        roi_wealth_pct = (final_eq - START_BALANCE) / START_BALANCE * 100.0  # cum-pnl basis, ignores withdrawal cap
        final_wallet = float(closed["wallet_after"].iloc[-1]) if len(closed) else START_BALANCE
        total_withdrawn = float(closed["total_withdrawn"].iloc[-1]) if len(closed) else 0.0

        mdd = max_drawdown_pct(pd.concat([pd.Series([START_BALANCE]), eq_curve], ignore_index=True))
        wr = float((closed["r_gross"] > 0).mean()) * 100.0 if len(closed) else float("nan")
        mean_r = float(closed["r_gross"].mean()) if len(closed) else float("nan")
        mean_net_r = float(closed["net_r"].mean()) if len(closed) else float("nan")

        m_eq = month_end_table(closed, months)
        m_ret = monthly_returns(m_eq)
        monthly_cols[name] = m_eq
        monthly_ret_cols[name] = m_ret

        pos_months = int((m_ret > 0).sum())
        neg_months = int((m_ret < 0).sum())
        med_month = float(m_ret.median())
        mean_month = float(m_ret.mean())
        worst_month = float(m_ret.min())
        best_month = float(m_ret.max())

        # holdout: 2026-05-31 -> final
        cut = pd.Timestamp("2026-05-31 23:59:59")
        pre = closed[closed["exit_time"] <= cut]
        eq_at_cut = float(pre["trading_equity"].iloc[-1]) if len(pre) else START_BALANCE
        holdout_pct = (final_eq - eq_at_cut) / eq_at_cut * 100.0 if eq_at_cut > 0 else float("nan")

        # sanity: peak concurrent open-position count (approx, via reconstruction)
        opens = df.set_index("trade_id")
        entry_ev = pd.DataFrame({"t": df["entry_time"], "d": 1})
        exit_ev = pd.DataFrame({"t": df["exit_time"], "d": -1})
        conc = pd.concat([entry_ev, exit_ev]).sort_values("t")
        conc_signal_only = conc["d"].cumsum()  # ignores gating, upper bound only
        peak_signal_conc = int(conc_signal_only.max())

        summary_rows.append(dict(
            arm=name,
            trades_entered=n_opened,
            trades_available=len(df),
            blocked_anti_pyramid=blocked["anti_pyramid"],
            blocked_short_impulse=blocked["short_impulse_gate"],
            blocked_net_dir_cap=blocked["net_dir_cap"],
            blocked_gross_risk_cap=blocked["gross_risk_cap"],
            final_trading_equity=final_eq,
            final_wallet_capped=final_wallet,
            total_withdrawn=total_withdrawn,
            roi_pct_wealth_basis=roi_wealth_pct,
            max_drawdown_pct=mdd,
            win_rate_gross_pct=wr,
            mean_r_gross=mean_r,
            mean_r_net=mean_net_r,
            months_positive=pos_months,
            months_negative=neg_months,
            median_month_ret_pct=med_month,
            mean_month_ret_pct=mean_month,
            worst_month_ret_pct=worst_month,
            best_month_ret_pct=best_month,
            holdout_2026_05_to_final_pct=holdout_pct,
            holdout_eq_at_2026_05_31=eq_at_cut,
            peak_signal_level_concurrency_upper_bound=peak_signal_conc,
        ))

        closed.to_csv(OUT / f"trades_{name.replace('/', '-')}.csv", index=False)

    summary = pd.DataFrame(summary_rows).set_index("arm")
    summary.to_csv(OUT / "summary.csv")

    monthly = pd.DataFrame(monthly_cols)
    monthly.index.name = "month"
    monthly.to_csv(OUT / "monthly.csv")

    monthly_ret = pd.DataFrame(monthly_ret_cols)
    monthly_ret.index.name = "month"
    monthly_ret.to_csv(OUT / "monthly_returns.csv")

    print(summary.to_string())
    print()
    print(monthly.tail(8).to_string())


if __name__ == "__main__":
    main()
