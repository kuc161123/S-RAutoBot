#!/usr/bin/env python3
"""RED-TEAM attacks #2 (rejected pairs), #3 (holdout window ranking + block bootstrap CI),
#5 (cost sensitivity), #6 (implementation risk: stop_frac / notional / holding time).

Reads only existing risk_study parquets -- no re-simulation.
"""
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
ROOT = Path("/Users/lualakol/AutoTrading Bot")
RS = ROOT / "risk_study"
OUT = RS / "agent_out" / "red2"
COST = 0.00242
KEY = ["symbol", "div_type", "entry_time"]


def load(path, rcol, xcol):
    d = pd.read_parquet(RS / path)
    d = d.rename(columns={rcol: "r", xcol: "x"})
    d = d.dropna(subset=["r", "x"]).copy()
    d["nr"] = d.r - COST / d.stop_frac
    return d


def dedup(d):
    return d.drop_duplicates(subset=KEY, keep="first")


print("=" * 90)
print("ATTACK 2 -- rejected (symbol, div_type) pairs: does GLOBAL's edge depend on the")
print("            walk-forward's SELECTION of which pairs to trade at all?")
print("=" * 90)

cfg = yaml.safe_load(open(ROOT / "config.yaml"))["symbols"]
selected = set()
for sym, sdata in cfg.items():
    if not (sdata or {}).get("enabled", False):
        continue
    for c in (sdata or {}).get("configs", []) or []:
        selected.add((sym, c["divergence_type"]))
print(f"selected (symbol,div_type) pairs in config.yaml: {len(selected)}")

grid = pd.read_parquet(RS / "grid_inuni.parquet")
grid["pair"] = list(zip(grid.symbol, grid.div_type))
all_pairs = set(grid["pair"].unique())
rejected_pairs = all_pairs - selected
print(f"pairs present in grid_inuni (in-universe symbols x 4 div types): {len(all_pairs)}")
print(f"  of which SELECTED by config.yaml: {len(all_pairs & selected)}")
print(f"  of which REJECTED by the walk-forward: {len(rejected_pairs)}")

# GLOBAL arm = rr10, atr_mult=3.0, FIXED exit (grid_inuni only has fixed-TP exit, not the
# trailing exit -- this is a real limitation, flagged in the report).
g = grid.dropna(subset=["r_10", "exit_10"]).copy()
g["nr"] = g.r_10 - COST / g.stop_frac
# apply the same CHOP<52 gate the headline numbers use
g = g[g.chop_bos < 52]
g_sel = g[g["pair"].isin(selected)]
g_rej = g[g["pair"].isin(rejected_pairs)]
print(f"\nGLOBAL (rr=10, atr_mult=3.0, FIXED exit, CHOP<52 gate) net R per trade:")
print(f"  on SELECTED pairs   n={len(g_sel):>7,}  mean netR={g_sel.nr.mean():+.4f}")
print(f"  on REJECTED pairs   n={len(g_rej):>7,}  mean netR={g_rej.nr.mean():+.4f}")
print(f"  delta (selected - rejected): {g_sel.nr.mean() - g_rej.nr.mean():+.4f}")

# Also: what does GLOBAL look like on ALL in-universe pairs (no selection at all -- the
# protocol's A5 arm, which was never computed)?
print(f"\nGLOBAL on ALL in-universe pairs (A5 proxy, fixed exit): n={len(g):,}  "
      f"mean netR={g.nr.mean():+.4f}")

# per-atr_mult too, since grid has it
print("\nGLOBAL rr=10 at each atr_mult, selected vs rejected pairs (fixed exit):")
for am in sorted(grid.atr_mult.unique()):
    gg = grid[(grid.atr_mult == am)].dropna(subset=["r_10", "exit_10"]).copy()
    gg["nr"] = gg.r_10 - COST / gg.stop_frac
    gg = gg[gg.chop_bos < 52]
    s = gg[gg["pair"].isin(selected)]
    r = gg[gg["pair"].isin(rejected_pairs)]
    print(f"  am={am:>4}  selected n={len(s):>6,} netR={s.nr.mean():+.4f}   "
          f"rejected n={len(r):>6,} netR={r.nr.mean():+.4f}   "
          f"delta={s.nr.mean()-r.nr.mean():+.4f}")

print()
print("=" * 90)
print("ATTACK 3 -- how special is the 6-week holdout? Rolling window rank + block-bootstrap CI")
print("=" * 90)

a0 = load("universe_trail.parquet", "r_trail", "exit_trail")
a1 = load("uni_glob_rr10_am3_same.parquet", "r_result", "exit_time")
m = dedup(a0).merge(dedup(a1), on=KEY, suffixes=("_0", "_1"), how="inner")
m = m.sort_values("entry_time").reset_index(drop=True)
m["diff"] = m.nr_1 - m.nr_0
print(f"paired trades: {len(m):,}  span {m.entry_time.min().date()} .. {m.entry_time.max().date()}")

# weekly block bootstrap CI on the actual holdout window
hold = m[(m.entry_time >= "2026-05-25") & (m.entry_time < "2026-07-05")]
print(f"\nActual holdout: n={len(hold)}  A1-A0 mean diff={hold['diff'].mean():+.4f}")
wk = pd.to_datetime(hold.entry_time).dt.to_period("W")
groups = [g.to_numpy() for _, g in pd.Series(np.arange(len(hold))).groupby(wk.to_numpy())]
print(f"  ({len(groups)} weekly blocks in the holdout)")
rng = np.random.default_rng(7)
diffarr = hold["diff"].to_numpy()
bs = np.empty(4000)
for b in range(4000):
    pick = rng.integers(0, len(groups), len(groups))
    idx = np.concatenate([groups[i] for i in pick])
    bs[b] = diffarr[idx].mean()
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"  weekly block-bootstrap 95% CI on holdout diff: [{lo:+.4f}, {hi:+.4f}]  "
      f"(includes zero: {lo <= 0 <= hi})")

# rolling 6-week (42-day) windows across the FULL history, stepped weekly
print("\nAll overlapping 6-week (42-day) windows across full history, stepped by 7 days:")
t0 = m.entry_time.min().normalize()
t1 = m.entry_time.max()
starts = pd.date_range(t0, t1 - pd.Timedelta(days=42), freq="7D")
wins = 0
tot = 0
diffs = []
for s in starts:
    e = s + pd.Timedelta(days=42)
    w = m[(m.entry_time >= s) & (m.entry_time < e)]
    if len(w) < 30:
        continue
    d = w["diff"].mean()
    diffs.append((s, len(w), d))
    tot += 1
    if d > 0:
        wins += 1
diffs_df = pd.DataFrame(diffs, columns=["window_start", "n", "diff"])
diffs_df.to_csv(OUT / "attack3_rolling_windows.csv", index=False)
print(f"  {tot} windows with >=30 paired trades; GLOBAL beats FITTED (diff>0) in "
      f"{wins}/{tot} = {wins/tot*100:.1f}% of them")
holddiff = hold["diff"].mean()
pct = (diffs_df["diff"] < holddiff).mean() * 100
print(f"  the actual pre-registered holdout's diff ({holddiff:+.4f}) ranks at the "
      f"{pct:.1f}th percentile of all rolling-window diffs")
print(f"  window diff distribution: mean={diffs_df['diff'].mean():+.4f} "
      f"median={diffs_df['diff'].median():+.4f} "
      f"[{diffs_df['diff'].quantile(.1):+.4f}, {diffs_df['diff'].quantile(.9):+.4f}] (10/90pct)")

print()
print("=" * 90)
print("ATTACK 3b -- walk-forward fold CIs: how many of the 4 folds exclude zero?")
print("=" * 90)
FOLDS = [("F1", "2024-06-01", "2024-12-01"), ("F2", "2024-12-01", "2025-06-01"),
         ("F3", "2025-06-01", "2025-12-01"), ("F4", "2025-12-01", "2026-05-25")]
excl_zero = 0
for nm, a, b in FOLDS:
    s = m[(m.entry_time >= a) & (m.entry_time < b)]
    dd = s["diff"].to_numpy()
    wk = pd.to_datetime(s.entry_time).dt.to_period("W")
    groups = [g.to_numpy() for _, g in pd.Series(np.arange(len(s))).groupby(wk.to_numpy())]
    rng = np.random.default_rng(3)
    bs = np.empty(2000)
    for b_ in range(2000):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        bs[b_] = dd[idx].mean()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    excludes_zero = not (lo <= 0 <= hi)
    print(f"  {nm}: mean diff {dd.mean():+.4f}  95% CI [{lo:+.4f},{hi:+.4f}]  "
          f"excludes zero: {excludes_zero}")
    if excludes_zero:
        excl_zero += 1
print(f"\n  {excl_zero}/4 folds have CIs that EXCLUDE zero (i.e. are individually significant)")
print("  binomial: P(>=4 of 4 sign-wins | true p=0.5, independent) = 0.0625 -- and folds are")
print("  NOT independent (overlapping training windows, shared bull/bear regime), so the")
print("  true p-value under a fair null is higher than 0.0625, not lower.")

print()
print("=" * 90)
print("ATTACK 5 -- cost sensitivity: recompute net R at 11/18/24.2/34.1/45 bps")
print("=" * 90)
for bps in (11, 18, 24.2, 34.1, 45):
    c = bps / 10000
    d0 = (m.r_0 - c / m.stop_frac_0)
    d1 = (m.r_1 - c / m.stop_frac_1)
    diff = (d1 - d0).mean()
    print(f"  cost={bps:>5.1f}bps   A0 netR={d0.mean():+.4f}   A1 netR={d1.mean():+.4f}   "
          f"A1-A0={diff:+.4f}")

print()
print("=" * 90)
print("ATTACK 6 -- implementation risk: stop distance / notional / holding time")
print("=" * 90)
print(f"  A0 (fitted, mix of atr_mult 1.0-2.0)  mean stop_frac={m.stop_frac_0.mean()*100:.3f}%  "
      f"median={m.stop_frac_0.median()*100:.3f}%")
print(f"  A1 (global atr_mult=3.0)              mean stop_frac={m.stop_frac_1.mean()*100:.3f}%  "
      f"median={m.stop_frac_1.median()*100:.3f}%")
ratio = m.stop_frac_1 / m.stop_frac_0
print(f"  ratio A1/A0 stop distance: mean={ratio.mean():.2f}x  median={ratio.median():.2f}x")
print(f"  => at fixed $ risk per trade, A1 position notional is ~{1/ratio.median():.2f}x of A0's"
      f" (smaller notional, same $ risk, wider stop)")

hold0 = (m.x_0 - m.entry_time).dt.total_seconds() / 3600
hold1 = (m.x_1 - m.entry_time).dt.total_seconds() / 3600
print(f"\n  A0 holding time (hours): mean={hold0.mean():.1f}  median={hold0.median():.1f}  "
      f"p90={hold0.quantile(.9):.1f}")
print(f"  A1 holding time (hours): mean={hold1.mean():.1f}  median={hold1.median():.1f}  "
      f"p90={hold1.quantile(.9):.1f}")
print(f"  A1 holds positions {hold1.mean()/hold0.mean():.2f}x as long on average "
      f"(more funding-fee exposure, more overnight/weekend gap risk)")

print("\ndone")
