"""
Build causal BTC state features + crowding/breadth features and attach to the
trade-level dataset. Save the enriched frame to short/trades_enriched.parquet
so downstream analysis scripts don't have to recompute this every run.
"""
import pandas as pd
import numpy as np

BASE = "/Users/lualakol/AutoTrading Bot"

df = pd.read_parquet(f"{BASE}/risk_study/uni_glob_rr10_am3_same.parquet")
btc = pd.read_parquet(f"{BASE}/cache_3yr_1h/BTCUSDT.parquet")

df = df.sort_values("entry_time").reset_index(drop=True)

# ---------------------------------------------------------------------------
# Cost model -> net R
# ---------------------------------------------------------------------------
df["net_r"] = df["r_result"] - 0.00242 / df["stop_frac"]

# ---------------------------------------------------------------------------
# BTC daily close series, and causal (t-1) state variables
# ---------------------------------------------------------------------------
btc = btc.sort_values("start").reset_index(drop=True)
btc["date"] = btc["start"].dt.floor("D")
daily = btc.groupby("date").agg(close=("close", "last"), high=("high", "max"), low=("low", "min")).reset_index()
daily = daily.sort_values("date").reset_index(drop=True)

def ret_n(s, n):
    return s.pct_change(n)

daily["ret_7d"] = ret_n(daily["close"], 7)
daily["ret_14d"] = ret_n(daily["close"], 14)
daily["ret_30d"] = ret_n(daily["close"], 30)
daily["ret_60d"] = ret_n(daily["close"], 60)

# realized vol: std of daily log returns over trailing N days (annualized-ish, just raw std)
logret = np.log(daily["close"] / daily["close"].shift(1))
daily["rvol_14d"] = logret.rolling(14).std()
daily["rvol_30d"] = logret.rolling(30).std()

# distance from 200-day EMA (on daily close)
ema200 = daily["close"].ewm(span=200, adjust=False).mean()
daily["dist_ema200"] = daily["close"] / ema200 - 1.0

# drawdown from trailing high (all-time-to-date and 90d rolling)
roll_max_all = daily["close"].cummax()
daily["dd_from_high_all"] = daily["close"] / roll_max_all - 1.0
roll_max_90 = daily["close"].rolling(90, min_periods=10).max()
daily["dd_from_high_90d"] = daily["close"] / roll_max_90 - 1.0

# Shift everything forward by 1 day: value "as of date D" only uses data through D-1's close,
# i.e. it's the state known at the start of date D (causal, matches btc_impulse's own construction).
state_cols = ["ret_7d", "ret_14d", "ret_30d", "ret_60d", "rvol_14d", "rvol_30d",
              "dist_ema200", "dd_from_high_all", "dd_from_high_90d"]
for c in state_cols:
    daily[c + "_causal"] = daily[c].shift(1)

causal_cols = [c + "_causal" for c in state_cols]
daily_small = daily[["date"] + causal_cols].copy()

df["entry_date"] = df["entry_time"].dt.floor("D")
df = df.merge(daily_small, left_on="entry_date", right_on="date", how="left")
df = df.drop(columns=["date"])

# Sanity check vs shipped btc_impulse: our ret_30d_causal > 0.10 should match closely
mine = (df["ret_30d_causal"] > 0.10)
match = (mine == df["btc_impulse"]).mean()
print(f"[check] my ret_30d_causal>0.10 vs shipped btc_impulse agreement: {match:.4f}")
print(f"[check] rows with NaN causal state (early history, dropped from state-based rules): {df['ret_30d_causal'].isna().sum()}")

# ---------------------------------------------------------------------------
# Crowding / density features: trailing entry counts, fully causal
# (uses only entries strictly before this trade's own entry_time)
# ---------------------------------------------------------------------------
all_entries = np.sort(df["entry_time"].values.astype("datetime64[ns]"))
short_entries = np.sort(df.loc[df.side == "short", "entry_time"].values.astype("datetime64[ns]"))

def trailing_count(query_times, event_times, window_hours):
    window = np.timedelta64(window_hours, "h")
    lo_idx = np.searchsorted(event_times, query_times - window, side="left")
    hi_idx = np.searchsorted(event_times, query_times, side="left")  # strictly before own time
    return hi_idx - lo_idx

qt = df["entry_time"].values.astype("datetime64[ns]")
df["density_all_24h"] = trailing_count(qt, all_entries, 24)
df["density_all_72h"] = trailing_count(qt, all_entries, 72)
df["density_short_24h"] = trailing_count(qt, short_entries, 24)
df["density_short_72h"] = trailing_count(qt, short_entries, 72)

# ---------------------------------------------------------------------------
# Breadth: concurrent OPEN short positions at the moment of entry (causal --
# only counts shorts that already entered and have not yet exited)
# ---------------------------------------------------------------------------
short_mask = df.side == "short"
s_entry = df.loc[short_mask, "entry_time"].values.astype("datetime64[ns]")
s_exit = df.loc[short_mask, "exit_time"].values.astype("datetime64[ns]")
sorted_entries = np.sort(s_entry)
sorted_exits = np.sort(s_exit)

def concurrency_at(query_times):
    # entries strictly before t (already open by t)
    n_started = np.searchsorted(sorted_entries, query_times, side="left")
    # exits at or before t (already closed by t)
    n_closed = np.searchsorted(sorted_exits, query_times, side="left")
    return n_started - n_closed

df["short_concurrency"] = np.nan
df.loc[short_mask, "short_concurrency"] = concurrency_at(s_entry)

out_path = f"{BASE}/risk_study/agent_out/short/trades_enriched.parquet"
df.to_parquet(out_path, index=False)
print(f"[ok] wrote {out_path}, shape={df.shape}")
print(df[["entry_time","side","net_r","ret_30d_causal","btc_impulse","density_short_24h","short_concurrency"]].tail(10))
