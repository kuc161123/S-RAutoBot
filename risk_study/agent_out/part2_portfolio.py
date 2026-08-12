"""
AGENT-REP2 -- independent replication, PART 2.

Own portfolio simulator over the trade sets produced by part1_signals.py
(trades_A0.parquet / trades_A1.parquet). Written from scratch against the task
brief's spec; does not import backtest_production_correct.py, risk_study/sweep.py,
risk_study/monthly.py or build_trail_universe_wide.py.

$1500 start. Per-trade risk = taper(wallet) * regime_mult(last 20 closed), applied to
EQUITY (== wallet here since unrealized P&L is held at 0 by design -- using each open
trade's known outcome for unrealized would leak the future). Long BTC-bull boost 1.3x.
BTC short-impulse gate skips shorts. Net-directional (10%) and gross (30%) open-risk
caps. One open position per (symbol, side). Cost 0.00242/stop_frac subtracted from
gross R. Withdraw down to $50,000 whenever wallet exceeds it. Entries and exits are
processed as one chronological event stream (exits before entries on an exact tie).
"""
import os

import numpy as np
import pandas as pd

REPO = "/Users/lualakol/AutoTrading Bot"
OUT_DIR = os.path.join(REPO, "risk_study", "agent_out")
CACHE_DIR = os.path.join(REPO, "cache_3yr_1h")

START_BALANCE = 1500.0
WITHDRAW_TARGET = 50000.0
ROUND_TRIP_COST = 0.00242
NET_DIR_CAP = 0.10
GROSS_RISK_CAP = 0.30
LONG_BULL_BOOST = 1.3
BTC_SHORT_GATE_RET30 = 0.10

TAPER_SCHEDULE = [
    (1500, 0.003), (3000, 0.0028), (5000, 0.0025), (8000, 0.0022),
    (12000, 0.002), (20000, 0.0017), (40000, 0.0014),
]

HOLDOUT_START = pd.Timestamp("2026-05-31")


def taper_lookup(wallet):
    if wallet < 1500:
        return 0.003
    best = 0.003
    for thresh, rate in TAPER_SCHEDULE:
        if wallet >= thresh:
            best = rate
    return best


def regime_mult(closed_r_hist):
    n = len(closed_r_hist)
    if n < 10:
        return 0.1
    last20 = closed_r_hist[-20:]
    wr = sum(1 for r in last20 if r > 0) / len(last20)
    avg_r = sum(last20) / len(last20)
    if wr >= 0.18 and avg_r >= 0.15:
        return 1.0
    elif wr >= 0.18 or avg_r >= 0.10:
        return 0.5
    elif wr >= 0.10 or avg_r >= -0.5:
        return 0.25
    else:
        return 0.1


def build_btc_signals():
    df = pd.read_parquet(os.path.join(CACHE_DIR, "BTCUSDT.parquet"),
                          columns=["start", "close"])
    df = df.sort_values("start").drop_duplicates(subset="start").reset_index(drop=True)
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    df["bull"] = df["close"] > df["ema200"]
    df["bull_lagged"] = df["bull"].shift(1).fillna(False)
    hourly = df[["start", "bull_lagged"]].copy()

    daily = df.copy()
    daily["date"] = daily["start"].dt.floor("D")
    daily_close = daily.groupby("date")["close"].last().reset_index()
    daily_close["ret30"] = daily_close["close"].pct_change(30)
    daily_close["impulse"] = daily_close["ret30"] > BTC_SHORT_GATE_RET30
    daily_close["impulse_lagged"] = daily_close["impulse"].shift(1).fillna(False)
    daily_sig = daily_close[["date", "impulse_lagged"]].copy()

    return hourly, daily_sig


def lookup_btc_bull(hourly, entry_times):
    tmp = pd.DataFrame({"start": entry_times}).sort_values("start")
    merged = pd.merge_asof(tmp, hourly, on="start", direction="backward")
    merged = merged.set_index(tmp.index)
    return merged["bull_lagged"].reindex(range(len(entry_times))).fillna(False).to_numpy()


def lookup_btc_impulse(daily_sig, entry_times):
    dates = pd.Series(entry_times).dt.floor("D")
    tmp = pd.DataFrame({"date": dates}).sort_values("date")
    merged = pd.merge_asof(tmp, daily_sig, on="date", direction="backward")
    merged = merged.set_index(tmp.index)
    return merged["impulse_lagged"].reindex(range(len(entry_times))).fillna(False).to_numpy()


def simulate(trades, arm_name):
    trades = trades.copy().reset_index(drop=True)
    trades["trade_id"] = trades.index
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["exit_time"] = pd.to_datetime(trades["exit_time"])
    trades["stop_frac"] = trades["risk_dist"] / trades["entry_price"]
    trades["cost_r"] = ROUND_TRIP_COST / trades["stop_frac"]
    trades["r_net"] = trades["r_gross"] - trades["cost_r"]

    hourly, daily_sig = build_btc_signals()
    trades["btc_bull"] = lookup_btc_bull(hourly, trades["entry_time"])
    trades["btc_impulse"] = lookup_btc_impulse(daily_sig, trades["entry_time"])

    n = len(trades)
    r_net = trades["r_net"].to_numpy()
    side = trades["side"].to_numpy()
    symbol = trades["symbol"].to_numpy()
    entry_time = trades["entry_time"].to_numpy()
    exit_time = trades["exit_time"].to_numpy()
    btc_bull = trades["btc_bull"].to_numpy()
    btc_impulse = trades["btc_impulse"].to_numpy()

    # Tie-break: entries sort before exits at an identical timestamp.
    # This matters because ~10% of A0 trades (tight stops, atr_mult 1.0-1.5) resolve
    # within their own entry bar -- exit_time == entry_time for the SAME trade. An
    # "exits-first" convention (the more obvious portfolio-accounting default, and
    # what this script used on the first pass) processes that trade's own exit event
    # before its entry event ever ran, so the position is never found in
    # open_positions, is silently skipped, and then leaks open for the rest of the
    # run once the entry event does fire -- permanently consuming a (symbol,side)
    # slot and gross-risk headroom. Caught this via a directly measured mismatch:
    # candidate rows actually open at a sampled block timestamp = 11, but the
    # simulator's open_positions dict said 166. n_exit_matched (5,243) trailing
    # n_entered (5,682) by exactly the leaked count confirmed it.
    # Entries-before-exits fixes same-trade same-bar causality at the cost of a
    # mild, deliberate conservative bias: a DIFFERENT trade's exit at the exact same
    # timestamp is processed after the new entry, so that entry sees capital as not
    # yet freed. That trades a catastrophic, permanent lockup for a small amount of
    # extra blocking -- an acceptable and clearly-documented tradeoff.
    events = []
    for i in range(n):
        events.append((entry_time[i], 0, str(symbol[i]), i, "entry"))
        events.append((exit_time[i], 1, str(symbol[i]), i, "exit"))
    events.sort(key=lambda e: (e[0], e[1], e[2], e[3]))

    wallet = START_BALANCE
    open_positions = {}          # trade_id -> dict
    open_by_symbol_side = {}     # (symbol, side) -> trade_id
    open_long_risk = 0.0
    open_short_risk = 0.0
    open_total_risk = 0.0
    closed_r_hist = []

    equity_curve = [(pd.Timestamp("2023-06-01"), wallet)]
    entered = 0
    blocked_anti_pyramid = 0
    blocked_short_gate = 0
    blocked_net_dir = 0
    blocked_gross_cap = 0

    holdout_equity_at_start = None

    for ts, _prio, _sym, tid, kind in events:
        if kind == "exit":
            if tid not in open_positions:
                continue
            pos = open_positions.pop(tid)
            risk_dollars = pos["risk_dollars"]
            pnl = r_net[tid] * risk_dollars
            wallet += pnl
            if wallet > WITHDRAW_TARGET:
                wallet = WITHDRAW_TARGET
            if pos["side"] == "long":
                open_long_risk -= risk_dollars
            else:
                open_short_risk -= risk_dollars
            open_total_risk -= risk_dollars
            del open_by_symbol_side[(pos["symbol"], pos["side"])]
            closed_r_hist.append(r_net[tid])
            equity_curve.append((pd.Timestamp(ts), wallet))
            if holdout_equity_at_start is None and pd.Timestamp(ts) >= HOLDOUT_START:
                # equity just BEFORE this exit crossed the holdout boundary
                pass
        else:
            sym = str(symbol[tid])
            sd = side[tid]
            key = (sym, sd)
            if key in open_by_symbol_side:
                blocked_anti_pyramid += 1
                continue

            equity = wallet
            mult = regime_mult(closed_r_hist)
            base = taper_lookup(wallet)
            final_frac = base * mult
            risk_dollars = equity * final_frac

            if sd == "long" and bool(btc_bull[tid]):
                risk_dollars *= LONG_BULL_BOOST

            if sd == "short" and bool(btc_impulse[tid]):
                blocked_short_gate += 1
                continue

            if sd == "long":
                new_long = open_long_risk + risk_dollars
                if abs(new_long - open_short_risk) > NET_DIR_CAP * equity:
                    blocked_net_dir += 1
                    continue
            else:
                new_short = open_short_risk + risk_dollars
                if abs(open_long_risk - new_short) > NET_DIR_CAP * equity:
                    blocked_net_dir += 1
                    continue

            if open_total_risk + risk_dollars > GROSS_RISK_CAP * equity:
                blocked_gross_cap += 1
                continue

            open_positions[tid] = dict(symbol=sym, side=sd, risk_dollars=risk_dollars)
            open_by_symbol_side[key] = tid
            if sd == "long":
                open_long_risk += risk_dollars
            else:
                open_short_risk += risk_dollars
            open_total_risk += risk_dollars
            entered += 1

    # holdout equity: wallet value at the last realized close at/just before HOLDOUT_START
    eq_ts = [e[0] for e in equity_curve]
    eq_val = [e[1] for e in equity_curve]
    idx = np.searchsorted(np.array(eq_ts, dtype="datetime64[ns]"), np.datetime64(HOLDOUT_START), side="right") - 1
    idx = max(idx, 0)
    holdout_start_equity = eq_val[idx]
    final_equity = eq_val[-1]

    # max drawdown on realized curve
    peak = -np.inf
    max_dd = 0.0
    for _, w in equity_curve:
        peak = max(peak, w)
        dd = (peak - w) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    roi_pct = (final_equity / START_BALANCE - 1) * 100
    holdout_roi_pct = (final_equity / holdout_start_equity - 1) * 100 if holdout_start_equity > 0 else float("nan")

    total_candidates = n
    total_blocked = blocked_anti_pyramid + blocked_short_gate + blocked_net_dir + blocked_gross_cap

    print(f"\n=== {arm_name} ===")
    print(f"candidate trades: {total_candidates}")
    print(f"entered: {entered}  blocked: {total_blocked} "
          f"(anti_pyramid={blocked_anti_pyramid}, short_gate={blocked_short_gate}, "
          f"net_dir={blocked_net_dir}, gross_cap={blocked_gross_cap})")
    print(f"final equity: ${final_equity:,.2f}")
    print(f"ROI: {roi_pct:.1f}%")
    print(f"max drawdown (realized curve): {max_dd*100:.1f}%")
    print(f"holdout start ({HOLDOUT_START.date()}) equity: ${holdout_start_equity:,.2f}")
    print(f"holdout ROI (holdout_start -> final): {holdout_roi_pct:.1f}%")

    return dict(
        arm=arm_name, candidates=total_candidates, entered=entered, blocked=total_blocked,
        final_equity=final_equity, roi_pct=roi_pct, max_dd_pct=max_dd * 100,
        holdout_start_equity=holdout_start_equity, holdout_roi_pct=holdout_roi_pct,
        equity_curve=equity_curve,
    )


def main():
    a0 = pd.read_parquet(os.path.join(OUT_DIR, "trades_A0.parquet"))
    a1 = pd.read_parquet(os.path.join(OUT_DIR, "trades_A1.parquet"))

    res0 = simulate(a0, "A0 (LIVE per-symbol rr/atr_mult)")
    res1 = simulate(a1, "A1 (GLOBAL rr=10, atr_mult=3.0)")

    print("\n=== SIDE-BY-SIDE ===")
    print(f"{'metric':<30}{'A0':>18}{'A1':>18}")
    print(f"{'entered trades':<30}{res0['entered']:>18,}{res1['entered']:>18,}")
    print(f"{'final equity':<30}${res0['final_equity']:>16,.0f}${res1['final_equity']:>16,.0f}")
    print(f"{'ROI %':<30}{res0['roi_pct']:>17.1f}%{res1['roi_pct']:>17.1f}%")
    print(f"{'max drawdown %':<30}{res0['max_dd_pct']:>17.1f}%{res1['max_dd_pct']:>17.1f}%")
    print(f"{'holdout ROI %':<30}{res0['holdout_roi_pct']:>17.1f}%{res1['holdout_roi_pct']:>17.1f}%")

    for r in (res0, res1):
        ec = pd.DataFrame(r.pop("equity_curve"), columns=["timestamp", "wallet"])
        ec.to_csv(os.path.join(OUT_DIR, f"equity_curve_{r['arm'].split()[0]}.csv"), index=False)


if __name__ == "__main__":
    main()
