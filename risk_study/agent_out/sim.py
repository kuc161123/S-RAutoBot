"""
Independent, from-scratch replication of the live bot's portfolio sizing policy,
built WITHOUT reference to backtest_production_correct.py or risk_study/sweep.py.

Design choices made where the spec was silent (documented in REPLICATION.md too):

1. EQUITY vs WALLET: the input trade table gives only terminal R per trade (no
   intrabar price path), so there is no leakage-free way to mark open positions
   to market. I therefore treat "equity" as identically equal to "wallet"
   (starting capital + sum of realized net-R dollars from trades closed so far).
   This is an approximation of the live bot's true equity (which includes
   unrealized P&L from the exchange) — see REPLICATION.md for the implication.

2. Tie-breaking at identical timestamps: when an exit and an entry share the
   exact same timestamp, the exit is processed first (frees risk capacity
   before new risk is committed). Among entries that share a timestamp, they
   are processed in the row order they appear in the source parquet (stable
   sort) — this is an acknowledged source of the same intra-hour ordering
   noise the repo's own studies flag (~15% on some figures).

3. Regime rolling window (wr, avg_r over the last 20 CLOSED trades) is tracked
   on NET (post-cost) R of trades that were actually opened and have since
   closed, per-period (each of DEV/VAL/HOLDOUT is its own simulation starting
   from a blank slate — matching the fact that sweep_main.csv shows each
   window starting from start_balance=10000 independently).

4. Cost model: net_r = r_result - cost_bps / stop_frac, cost_bps = 0.00341,
   applied once at trade construction (fixed for the life of the trade).

5. Taper-schedule wallet-lookup: below the lowest rung ($1500) the taper never
   fires and the base f itself is used directly, per the documented live-bot
   quirk.

Everything below is a single self-contained script; no imports from the repo's
other backtest engines.
"""
import heapq
import pandas as pd
import numpy as np

DATA_PATH = "/Users/lualakol/AutoTrading Bot/risk_study/universe_chopBOS.parquet"
OUT_PATH = "/Users/lualakol/AutoTrading Bot/risk_study/agent_out/agent_sweep.csv"

COST_BPS = 0.00341
START_EQUITY = 10000.0

# taper schedule at baseline f = 0.003
BASE_F = 0.003
BASE_SCHEDULE = [
    (1500, 0.003),
    (3000, 0.0028),
    (5000, 0.0025),
    (8000, 0.0022),
    (12000, 0.002),
    (20000, 0.0017),
    (40000, 0.0014),
]

NET_DIR_CAP = 0.10
GROSS_CAP = 0.30
LONG_BULL_BOOST = 1.3

WINDOWS = {
    "DEV": (pd.Timestamp("2000-01-01"), pd.Timestamp("2025-07-01")),
    "VAL": (pd.Timestamp("2025-07-01"), pd.Timestamp("2026-05-25")),
    "HOLDOUT": (pd.Timestamp("2026-05-25"), pd.Timestamp("2100-01-01")),
}

F_VALUES = [0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03]


def taper_lookup(wallet, f):
    """Return the risk fraction selected by the wallet-based taper schedule,
    scaled proportionally so that at f=0.003 it reproduces BASE_SCHEDULE.
    Below the lowest rung, taper never fires -> return f itself (documented
    live-bot quirk)."""
    scale = f / BASE_F
    selected = f
    for thresh, val in BASE_SCHEDULE:
        if wallet >= thresh:
            selected = val * scale
        else:
            break
    return selected


def regime_mult(closed_r_window):
    """closed_r_window: list of net-R values of the last <=20 CLOSED trades,
    most-recent-last. Returns the regime multiplier per the live-bot policy."""
    n = len(closed_r_window)
    if n < 10:
        return 0.1
    wr = sum(1 for r in closed_r_window if r > 0) / n
    avg_r = sum(closed_r_window) / n
    if wr >= 0.18 and avg_r >= 0.15:
        return 1.0
    elif wr >= 0.18 or avg_r >= 0.10:
        return 0.5
    elif wr >= 0.10 or avg_r >= -0.5:
        return 0.25
    else:
        return 0.1


def max_drawdown_pct(equity_curve):
    """equity_curve: list of wallet values in exit-chronological order
    (including the starting value as index 0). Peak-to-trough % drawdown."""
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd * 100.0


def run_sim(df_period, f):
    """Event-driven interleaved entry/exit simulation for one (period, f) cell."""
    # candidate entries, stable-sorted by entry_time (ties keep source row order)
    entries = df_period.sort_values("entry_time", kind="stable").reset_index(drop=True)
    n_candidates = len(entries)

    wallet = START_EQUITY
    equity = START_EQUITY  # equity == wallet, see module docstring

    # open-position book
    open_heap = []  # (exit_time, uid, trade) min-heap by exit_time
    open_long_risk = 0.0
    open_short_risk = 0.0
    open_positions = {}  # (symbol, side) -> uid, for anti-pyramid
    open_risk_by_uid = {}  # uid -> (risk_usd, side)

    closed_r_window = []  # rolling last-20 net-R of trades that were opened & closed

    equity_curve = [wallet]  # realized equity curve in exit order
    n_opened = 0
    uid_counter = 0

    ei = 0  # pointer into entries
    while ei < n_candidates or open_heap:
        next_entry_t = entries.loc[ei, "entry_time"] if ei < n_candidates else None
        next_exit_t = open_heap[0][0] if open_heap else None

        do_exit = False
        if next_exit_t is not None:
            if next_entry_t is None or next_exit_t <= next_entry_t:
                do_exit = True

        if do_exit:
            exit_t, uid, trade = heapq.heappop(open_heap)
            risk_usd, side = open_risk_by_uid.pop(uid)
            net_r = trade["net_r"]
            pnl = risk_usd * net_r
            wallet += pnl
            equity = wallet
            if side == "long":
                open_long_risk -= risk_usd
            else:
                open_short_risk -= risk_usd
            del open_positions[(trade["symbol"], side)]

            closed_r_window.append(net_r)
            if len(closed_r_window) > 20:
                closed_r_window.pop(0)

            equity_curve.append(wallet)
            continue

        # process an entry candidate
        row = entries.loc[ei]
        ei += 1

        side = row["side"]
        symbol = row["symbol"]

        # btc short gate
        if side == "short" and bool(row["btc_impulse"]):
            continue

        # anti-pyramid: at most one open position per (symbol, side)
        if (symbol, side) in open_positions:
            continue

        # regime multiplier from last-20 closed trades (opened-and-closed only)
        rmult = regime_mult(closed_r_window)

        # taper lookup on wallet
        base_frac = taper_lookup(wallet, f)

        final_mult = base_frac * rmult
        if side == "long" and bool(row["btc_bull"]):
            final_mult *= LONG_BULL_BOOST

        risk_usd = equity * final_mult
        if risk_usd <= 0:
            continue

        # net-directional cap
        if side == "long":
            new_long = open_long_risk + risk_usd
            new_short = open_short_risk
        else:
            new_long = open_long_risk
            new_short = open_short_risk + risk_usd
        if abs(new_long - new_short) > NET_DIR_CAP * equity:
            continue

        # gross open-risk cap
        gross_now = open_long_risk + open_short_risk
        if gross_now + risk_usd > GROSS_CAP * equity:
            continue

        # trade opens
        net_r = row["r_result"] - COST_BPS / row["stop_frac"]
        trade = {"symbol": symbol, "net_r": net_r}
        uid = uid_counter
        uid_counter += 1
        open_risk_by_uid[uid] = (risk_usd, side)
        open_positions[(symbol, side)] = uid
        if side == "long":
            open_long_risk += risk_usd
        else:
            open_short_risk += risk_usd
        heapq.heappush(open_heap, (row["exit_time"], uid, trade))
        n_opened += 1

    final_equity = wallet
    net_roi_pct = (final_equity - START_EQUITY) / START_EQUITY * 100.0
    max_dd = max_drawdown_pct(equity_curve)
    roi_dd = net_roi_pct / max_dd if max_dd > 0 else float("nan")

    return {
        "n_trades": n_opened,
        "n_candidates": n_candidates,
        "final_equity": final_equity,
        "net_roi_pct": net_roi_pct,
        "max_dd_pct": max_dd,
        "roi_dd_ratio": roi_dd,
    }


def main():
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values("entry_time", kind="stable").reset_index(drop=True)

    rows = []
    for wname, (lo, hi) in WINDOWS.items():
        mask = (df["entry_time"] >= lo) & (df["entry_time"] < hi)
        dfp = df.loc[mask]
        for f in F_VALUES:
            res = run_sim(dfp, f)
            rows.append({
                "window": wname,
                "risk_pct": f * 100,  # express as percent to match sweep_main.csv convention
                "f": f,
                **res,
            })
            print(f"{wname:8s} f={f:<7} n_trades={res['n_trades']:6d} "
                  f"final_eq={res['final_equity']:14,.2f} "
                  f"roi%={res['net_roi_pct']:12.2f} "
                  f"maxdd%={res['max_dd_pct']:8.2f} "
                  f"roi/dd={res['roi_dd_ratio']:8.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print("\nWrote", OUT_PATH)


if __name__ == "__main__":
    main()
