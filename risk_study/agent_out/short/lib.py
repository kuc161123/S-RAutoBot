"""Shared helpers for the short-tail study."""
import pandas as pd
import numpy as np

BASE = "/Users/lualakol/AutoTrading Bot"
SPLIT = pd.Timestamp("2026-05-25")


def load():
    df = pd.read_parquet(f"{BASE}/risk_study/agent_out/short/trades_enriched.parquet")
    df["exit_date"] = df["exit_time"].dt.floor("D")
    df["week"] = df["entry_time"].dt.to_period("W").astype(str)
    return df


def split(df):
    train = df[df.entry_time < SPLIT].copy()
    hold = df[df.entry_time >= SPLIT].copy()
    return train, hold


def worst_n_days(sub, n=10, by="exit_date"):
    """Return (sum_r_of_worst_n_days, day_table) grouping net_r by day."""
    if len(sub) == 0:
        return 0.0, pd.DataFrame()
    daily = sub.groupby(by)["net_r"].sum().sort_values()
    worst = daily.head(n)
    return worst.sum(), worst


def summarize(sub, label=""):
    if len(sub) == 0:
        return dict(label=label, n=0, mean_r=np.nan, total_r=np.nan, worst10=np.nan)
    total = sub["net_r"].sum()
    mean = sub["net_r"].mean()
    w10, _ = worst_n_days(sub, 10)
    return dict(label=label, n=len(sub), mean_r=mean, total_r=total, worst10=w10)


def week_block_bootstrap(sub, statistic_fn, n_boot=1000, seed=0):
    """Weekly block bootstrap CI for a scalar statistic_fn(df_subset)->float."""
    rng = np.random.default_rng(seed)
    weeks = sub["week"].unique()
    if len(weeks) < 2:
        val = statistic_fn(sub)
        return val, val, val
    by_week = {w: sub[sub.week == w] for w in weeks}
    stats = []
    nW = len(weeks)
    for _ in range(n_boot):
        chosen = rng.choice(weeks, size=nW, replace=True)
        parts = [by_week[w] for w in chosen]
        resampled = pd.concat(parts, ignore_index=True)
        stats.append(statistic_fn(resampled))
    stats = np.array(stats)
    return np.nanmean(stats), np.nanpercentile(stats, 2.5), np.nanpercentile(stats, 97.5)


def stat_total_r(sub):
    return sub["net_r"].sum()


def stat_mean_r(sub):
    return sub["net_r"].mean() if len(sub) else np.nan


def stat_worst10(sub):
    w10, _ = worst_n_days(sub, 10)
    return w10
