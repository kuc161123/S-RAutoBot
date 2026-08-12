"""
CRITERION 4 — market benchmark controls (C2/C3/C4) for PROTOCOL_symbols.md.

Builds three unlevered benchmark equity curves over 2023-06-01 -> 2026-07-25,
each starting at $1500:

  C2  BTC buy-and-hold
  C3  long-only equal-weight basket of the 277 live symbols, rebalanced monthly
  C4  BTC 200-EMA trend follow (long/flat), 24.2bps per switch

Writes:
  risk_study/agent_out/MARKET_CONTROLS_equity.csv   (hourly equity curves, 4 cols + ts)
  risk_study/agent_out/MARKET_CONTROLS_stats.csv     (summary stats table)
"""
import os
import yaml
import numpy as np
import pandas as pd

ROOT = "/Users/lualakol/AutoTrading Bot"
CACHE = os.path.join(ROOT, "cache_3yr_1h")
OUT = os.path.join(ROOT, "risk_study", "agent_out")

START = pd.Timestamp("2023-06-01 00:00:00")
END = pd.Timestamp("2026-07-25 18:00:00")  # last bar available in BTC cache
HOLDOUT_START = pd.Timestamp("2026-05-25 00:00:00")
HOLDOUT_END = pd.Timestamp("2026-07-04 00:00:00")

INITIAL = 1500.0
ROUNDTRIP_BPS = 24.2 / 10000.0

ANNUALIZATION_DAYS = 365.0  # crypto trades 24/7


def load_symbol(sym, start=START, end=END):
    path = os.path.join(CACHE, f"{sym}.parquet")
    df = pd.read_parquet(path, columns=["start", "open", "high", "low", "close"])
    df["start"] = pd.to_datetime(df["start"])
    df = df[(df["start"] >= start) & (df["start"] <= end)].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# C2 — BTC buy and hold
# ---------------------------------------------------------------------------
def build_c2():
    btc = load_symbol("BTCUSDT")
    entry_open = btc["open"].iloc[0]
    equity = INITIAL * (btc["close"] / entry_open)
    s = pd.Series(equity.values, index=btc["start"].values, name="C2_btc_buyhold")
    return s


# ---------------------------------------------------------------------------
# C4 — BTC 200 EMA trend follow, long/flat, causal (lag 1 bar), 24.2bps/switch
# ---------------------------------------------------------------------------
def build_c4():
    btc = load_symbol("BTCUSDT")
    close = btc["close"]
    ema200 = close.ewm(span=200, adjust=False).mean()
    # causal signal: position at bar t decided using info as of bar t-1
    raw_signal = (close > ema200).astype(int)
    position = raw_signal.shift(1).fillna(0)  # lagged 1 bar -> causal

    ret = close.pct_change().fillna(0.0)
    strat_ret = position.values * ret.values

    switch = position.diff().fillna(position.iloc[0]).abs()  # 1 at bar of a position change
    # first bar: if position starts at 0 (since raw_signal shifted, first entry NaN->0), no switch charged
    switch.iloc[0] = 0.0
    cost = switch.values * ROUNDTRIP_BPS

    equity = np.empty(len(btc))
    eq = INITIAL
    for i in range(len(btc)):
        eq = eq * (1.0 + strat_ret[i])
        eq = eq * (1.0 - cost[i])
        equity[i] = eq

    n_switches = int(switch.sum())
    s = pd.Series(equity, index=btc["start"].values, name="C4_btc_ema200_trend")
    return s, n_switches


# ---------------------------------------------------------------------------
# C3 — long-only equal-weight basket of the 277 live symbols, monthly rebalance
# ---------------------------------------------------------------------------
def load_live_symbols():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    syms_cfg = cfg.get("symbols", {})
    live = sorted([s for s, v in syms_cfg.items() if v.get("enabled") and v.get("configs")])
    return live


def build_c3(live_symbols):
    # Build daily close panel (last close of each UTC day) per symbol, over full range.
    daily_closes = {}
    first_last = {}
    for sym in live_symbols:
        df = load_symbol(sym)
        if df.empty:
            continue
        df = df.set_index("start")
        daily = df["close"].resample("1D").last()
        daily_closes[sym] = daily
        first_last[sym] = (df.index.min(), df.index.max())

    panel = pd.DataFrame(daily_closes)
    panel = panel.sort_index()
    # full daily index across the study window
    full_idx = pd.date_range(START.normalize(), END.normalize(), freq="1D")
    panel = panel.reindex(full_idx)
    # forward-fill only WITHIN each symbol's own observed [first,last] range (no fabricating
    # pre-listing or extrapolating price); leave NaN outside.
    for sym in panel.columns:
        fl = first_last[sym]
        mask = (panel.index >= fl[0].normalize()) & (panel.index <= fl[1].normalize())
        panel.loc[mask, sym] = panel.loc[mask, sym].ffill()
        panel.loc[~mask, sym] = np.nan

    # Monthly rebalance dates: first available trading day of each calendar month in range.
    month_starts = pd.date_range(full_idx[0], full_idx[-1], freq="MS")
    # clip to range; also need the very first day if not exactly a month start
    rebal_dates = []
    for ms in month_starts:
        # first index date >= ms within panel
        candidates = panel.index[panel.index >= ms]
        if len(candidates) == 0:
            continue
        rebal_dates.append(candidates[0])
    if full_idx[0] not in rebal_dates:
        rebal_dates = [full_idx[0]] + rebal_dates
    rebal_dates = sorted(set(rebal_dates))

    equity_curve = pd.Series(index=panel.index, dtype=float)
    cash_units = None  # dict sym -> units held
    n_eligible_history = []

    for i, rdate in enumerate(rebal_dates):
        next_rdate = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else panel.index[-1] + pd.Timedelta(days=1)
        segment_idx = panel.index[(panel.index >= rdate) & (panel.index < next_rdate)]
        if len(segment_idx) == 0:
            continue
        prices_at_rebal = panel.loc[rdate]
        eligible = prices_at_rebal.dropna().index.tolist()
        n_eligible_history.append((rdate, len(eligible)))
        if len(eligible) == 0:
            # carry forward previous equity flat if truly nothing eligible (shouldn't happen)
            prev_val = equity_curve.loc[:rdate].dropna()
            val = prev_val.iloc[-1] if len(prev_val) else INITIAL
            for d in segment_idx:
                equity_curve.loc[d] = val
            continue

        port_val_at_rebal = equity_curve.loc[:rdate].dropna()
        capital = port_val_at_rebal.iloc[-1] if len(port_val_at_rebal) else INITIAL
        alloc_per_sym = capital / len(eligible)
        units = {sym: alloc_per_sym / prices_at_rebal[sym] for sym in eligible}

        for d in segment_idx:
            row = panel.loc[d, eligible]
            # if a symbol has NaN mid-segment (shouldn't, since ffill within its own range,
            # but guard anyway) value it at last known unit price -> use ffilled panel so it's fine
            val = 0.0
            for sym in eligible:
                px = row[sym]
                if pd.isna(px):
                    # symbol had no data at all this day even after ffill (delisted before window
                    # end is not possible for live universe per spec) -> hold cash at last price
                    px = prices_at_rebal[sym]
                val += units[sym] * px
            equity_curve.loc[d] = val

    equity_curve = equity_curve.ffill()
    equity_curve.name = "C3_basket_equalweight"
    return equity_curve, n_eligible_history


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def compute_stats(equity_hourly_or_daily, label, is_daily_native=False):
    s = equity_hourly_or_daily.dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if len(s) < 2:
        return None

    start_val = s.iloc[0]
    end_val = s.iloc[-1]
    roi_pct = (end_val / start_val - 1.0) * 100.0

    days = (s.index[-1] - s.index[0]).total_seconds() / 86400.0
    years = days / ANNUALIZATION_DAYS
    cagr_pct = ((end_val / start_val) ** (1.0 / years) - 1.0) * 100.0 if years > 0 else np.nan

    running_max = s.cummax()
    dd = s / running_max - 1.0
    maxdd_pct = -dd.min() * 100.0

    roi_over_maxdd = roi_pct / maxdd_pct if maxdd_pct > 0 else np.nan
    calmar = (cagr_pct / 100.0) / (maxdd_pct / 100.0) if maxdd_pct > 0 else np.nan

    # daily returns for Sharpe/Sortino
    if is_daily_native:
        daily = s
    else:
        daily = s.resample("1D").last().dropna()
    daily_ret = daily.pct_change().dropna()
    mean_d = daily_ret.mean()
    std_d = daily_ret.std()
    sharpe = (mean_d / std_d) * np.sqrt(ANNUALIZATION_DAYS) if std_d > 0 else np.nan
    downside = daily_ret[daily_ret < 0]
    down_std = downside.std()
    sortino = (mean_d / down_std) * np.sqrt(ANNUALIZATION_DAYS) if down_std and down_std > 0 else np.nan

    # longest time underwater (days), based on the same series used for drawdown
    underwater_days = longest_underwater(s)

    return {
        "label": label,
        "start": str(s.index[0]),
        "end": str(s.index[-1]),
        "start_val": round(start_val, 2),
        "end_val": round(end_val, 2),
        "roi_pct": round(roi_pct, 3),
        "cagr_pct": round(cagr_pct, 3),
        "maxdd_pct": round(maxdd_pct, 3),
        "roi_over_maxdd": round(roi_over_maxdd, 4) if not np.isnan(roi_over_maxdd) else np.nan,
        "calmar": round(calmar, 4) if not np.isnan(calmar) else np.nan,
        "sharpe_daily": round(sharpe, 4) if not np.isnan(sharpe) else np.nan,
        "sortino_daily": round(sortino, 4) if not np.isnan(sortino) else np.nan,
        "longest_underwater_days": underwater_days,
        "n_days": round(days, 1),
    }


def longest_underwater(s):
    """Longest stretch (in days) from a new equity peak until equity again reaches
    that peak. If never recovered by series end, that stretch counts up to the last
    observation and is flagged separately by the caller if needed."""
    peak = s.iloc[0]
    peak_time = s.index[0]
    longest = pd.Timedelta(0)
    in_dd = False
    for t, v in s.items():
        if v >= peak:
            if in_dd:
                dur = t - peak_time
                if dur > longest:
                    longest = dur
            peak = v
            peak_time = t
            in_dd = False
        else:
            in_dd = True
    if in_dd:
        dur = s.index[-1] - peak_time
        if dur > longest:
            longest = dur
    return round(longest.total_seconds() / 86400.0, 1)


def restrict(s, start, end):
    return s[(s.index >= start) & (s.index <= end)]


def main():
    os.makedirs(OUT, exist_ok=True)

    print("Building C2 (BTC buy-hold)...")
    c2 = build_c2()

    print("Building C4 (BTC 200EMA trend)...")
    c4, c4_switches = build_c4()

    print("Loading live symbol list...")
    live_symbols = load_live_symbols()
    print(f"  {len(live_symbols)} live symbols")

    print("Building C3 (equal-weight basket, monthly rebalance)... this may take a while")
    c3, elig_hist = build_c3(live_symbols)

    # Save equity curves (align on outer join; C3 is daily-native, C2/C4 hourly)
    eq_df = pd.DataFrame({
        "C2_btc_buyhold": c2,
    })
    eq_df["C4_btc_ema200_trend"] = c4
    eq_df = eq_df.sort_index()
    eq_df.index.name = "ts"
    eq_df.to_csv(os.path.join(OUT, "MARKET_CONTROLS_equity_hourly.csv"))

    c3_df = c3.to_frame()
    c3_df.index.name = "date"
    c3_df.to_csv(os.path.join(OUT, "MARKET_CONTROLS_equity_C3_daily.csv"))

    # eligibility history for C3 (for transparency in report)
    elig_df = pd.DataFrame(elig_hist, columns=["rebal_date", "n_eligible_symbols"])
    elig_df.to_csv(os.path.join(OUT, "C3_eligibility_by_rebalance.csv"), index=False)

    # ---- Stats: full period ----
    rows = []
    rows.append(compute_stats(c2, "C2 BTC buy-and-hold (full)"))
    rows.append(compute_stats(c4, "C4 BTC 200EMA trend (full)"))
    rows.append(compute_stats(c3, "C3 equal-weight basket (full)", is_daily_native=True))

    # ---- Stats: holdout window ----
    c2_h = restrict(c2, HOLDOUT_START, HOLDOUT_END)
    c4_h = restrict(c4, HOLDOUT_START, HOLDOUT_END)
    c3_h = restrict(c3, HOLDOUT_START, HOLDOUT_END)

    rows.append(compute_stats(c2_h, "C2 BTC buy-and-hold (holdout)"))
    rows.append(compute_stats(c4_h, "C4 BTC 200EMA trend (holdout)"))
    rows.append(compute_stats(c3_h, "C3 equal-weight basket (holdout)", is_daily_native=True))

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(os.path.join(OUT, "MARKET_CONTROLS_stats.csv"), index=False)

    # BTC own return over the window (context)
    btc_full_roi = (c2.iloc[-1] / c2.iloc[0] - 1.0) * 100.0
    btc_holdout_roi = (c2_h.iloc[-1] / c2_h.iloc[0] - 1.0) * 100.0 if len(c2_h) else np.nan

    print("\n=== SUMMARY ===")
    print(stats_df.to_string(index=False))
    print(f"\nC4 switches over full period: {c4_switches}")
    print(f"BTC own ROI full period: {btc_full_roi:.2f}%")
    print(f"BTC own ROI holdout: {btc_holdout_roi:.2f}%")
    print(f"C3 first-day eligible symbols: {elig_hist[0][1] if elig_hist else 'n/a'}, "
          f"last rebalance eligible: {elig_hist[-1][1] if elig_hist else 'n/a'}")

    with open(os.path.join(OUT, "market_controls_run_meta.txt"), "w") as f:
        f.write(f"c4_switches_full_period={c4_switches}\n")
        f.write(f"btc_roi_full_pct={btc_full_roi:.4f}\n")
        f.write(f"btc_roi_holdout_pct={btc_holdout_roi:.4f}\n")
        f.write(f"n_live_symbols={len(live_symbols)}\n")
        f.write(f"c3_first_rebal_eligible={elig_hist[0][1] if elig_hist else 'n/a'}\n")
        f.write(f"c3_last_rebal_eligible={elig_hist[-1][1] if elig_hist else 'n/a'}\n")
        f.write(f"c3_n_rebalances={len(elig_hist)}\n")


if __name__ == "__main__":
    main()
