#!/usr/bin/env python3
"""AUDIT-C analysis: fold stats, weekly-block-bootstrap CIs, transfer (train-rank vs
test-rank Spearman) across 4 anchored walk-forward folds, and the top-5-by-IS table.

Reads risk_study/agent_out/audit_c/universes/*.parquet (one per swept cell, built by
build_param_universe.py). Writes results.csv and AUDIT_C.md into the same audit_c/ dir.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
UNI_DIR = HERE / "universes"

COST = 0.00242
IS_HOLDOUT_SPLIT = pd.Timestamp("2026-05-25")
TRIM_DAYS = 21

FOLD_BOUNDARIES = [pd.Timestamp("2024-06-30"), pd.Timestamp("2024-12-31"),
                   pd.Timestamp("2025-06-30"), pd.Timestamp("2025-12-31")]

# cell name -> {dimension, description, param value}
CELL_META = {
    "dist0":     ("sanity", "min_pivot_dist=0 (bt.detect_signals unconstrained; validates repro)"),
    "current":   ("baseline", "pivot(3,3) dist=3 stale=10 rsi=14 wait=12 keep-2nd-EMA (CURRENT live config)"),
    "piv_2_2":   ("pivot_width", "pivot (2,2)"),
    "piv_4_4":   ("pivot_width", "pivot (4,4)"),
    "piv_5_5":   ("pivot_width", "pivot (5,5)"),
    "piv_3_2":   ("pivot_width", "pivot (3,2) asymmetric"),
    "piv_2_3":   ("pivot_width", "pivot (2,3) asymmetric"),
    "wait_4":    ("max_wait", "max_wait_candles=4"),
    "wait_8":    ("max_wait", "max_wait_candles=8"),
    "wait_18":   ("max_wait", "max_wait_candles=18"),
    "wait_24":   ("max_wait", "max_wait_candles=24"),
    "stale_5":   ("pivot_stale", "pivot staleness limit=5"),
    "stale_15":  ("pivot_stale", "pivot staleness limit=15"),
    "stale_20":  ("pivot_stale", "pivot staleness limit=20"),
    "dist_2":    ("min_pivot_dist", "MIN_PIVOT_DISTANCE=2"),
    "dist_5":    ("min_pivot_dist", "MIN_PIVOT_DISTANCE=5"),
    "dist_8":    ("min_pivot_dist", "MIN_PIVOT_DISTANCE=8"),
    "rsi_7":     ("rsi_period", "RSI period=7"),
    "rsi_21":    ("rsi_period", "RSI period=21"),
    "dropema":   ("bos_ema_gate", "drop the second (BOS-time) EMA-200 gate"),
}
# cells that enter the coordinate-search sweep proper (excludes the dist0 sanity-only cell)
SWEEP_CELLS = [c for c in CELL_META if c != "dist0"]


def load_cell(name):
    f = UNI_DIR / f"{name}.parquet"
    d = pd.read_parquet(f)
    if d.empty:
        return d
    d["net_r"] = d.r_result - COST / d.stop_frac
    d["iso_year"] = d.entry_time.dt.isocalendar().year
    d["iso_week"] = d.entry_time.dt.isocalendar().week
    return d


def weekly_block_bootstrap(df, n_draws=1500, seed=0):
    """Resample ISO weeks with replacement; return (mean, lo2.5, hi97.5) of net_r mean."""
    if df.empty:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    wk = df.groupby(["iso_year", "iso_week"])["net_r"].apply(list)
    weeks = wk.index.to_list()
    vals = wk.values
    nW = len(weeks)
    if nW < 3:
        return df.net_r.mean(), np.nan, np.nan
    means = np.empty(n_draws)
    for b in range(n_draws):
        idx = rng.integers(0, nW, size=nW)
        pooled = np.concatenate([vals[i] for i in idx])
        means[b] = pooled.mean()
    return df.net_r.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    cells = {}
    for name in CELL_META:
        f = UNI_DIR / f"{name}.parquet"
        if not f.exists():
            print(f"[warn] missing {f}, skipping", file=sys.stderr)
            continue
        cells[name] = load_cell(name)

    # ------------------------------------------------------------------
    # trim: common cutoff = min over cells of (max entry_time) - 21 days
    # ------------------------------------------------------------------
    max_times = [d.entry_time.max() for d in cells.values() if not d.empty]
    global_cutoff = min(max_times) - pd.Timedelta(days=TRIM_DAYS)
    print(f"[trim] common cutoff (min max-entry across cells - 21d): {global_cutoff}")

    for name in cells:
        cells[name] = cells[name][cells[name].entry_time <= global_cutoff].reset_index(drop=True)

    # ------------------------------------------------------------------
    # folds: anchored (train from GEN_FROM to boundary), test = boundary -> next boundary
    # fold4 test end = global_cutoff (no 5th boundary available)
    # ------------------------------------------------------------------
    folds = []
    for i, b in enumerate(FOLD_BOUNDARIES):
        test_end = FOLD_BOUNDARIES[i + 1] if i + 1 < len(FOLD_BOUNDARIES) else global_cutoff
        folds.append((f"F{i+1}", b, test_end))
    print("[folds]", folds)

    # ------------------------------------------------------------------
    # per-cell summary rows
    # ------------------------------------------------------------------
    rows = []
    fold_train = {f[0]: {} for f in folds}  # fold_label -> cell -> mean net R
    fold_test = {f[0]: {} for f in folds}

    for name, d in cells.items():
        dim, desc = CELL_META[name]
        row = {"cell": name, "dimension": dim, "description": desc, "n_total": len(d)}

        is_d = d[d.entry_time < IS_HOLDOUT_SPLIT]
        ho_d = d[d.entry_time >= IS_HOLDOUT_SPLIT]
        is_mean, is_lo, is_hi = weekly_block_bootstrap(is_d, seed=hash(name) % 10000)
        ho_mean, ho_lo, ho_hi = weekly_block_bootstrap(ho_d, seed=(hash(name) + 1) % 10000)
        row.update({
            "n_is": len(is_d), "net_r_is": is_mean, "net_r_is_lo": is_lo, "net_r_is_hi": is_hi,
            "n_holdout": len(ho_d), "net_r_holdout": ho_mean, "net_r_holdout_lo": ho_lo,
            "net_r_holdout_hi": ho_hi,
            "gross_r_total": d.r_result.mean() if len(d) else np.nan,
            "net_r_total": d.net_r.mean() if len(d) else np.nan,
            "wr_total": (d.r_result > 0).mean() if len(d) else np.nan,
        })

        for label, b, test_end in folds:
            tr = d[d.entry_time < b]
            te = d[(d.entry_time >= b) & (d.entry_time < test_end)]
            row[f"{label}_train_n"] = len(tr)
            row[f"{label}_train_netr"] = tr.net_r.mean() if len(tr) else np.nan
            row[f"{label}_test_n"] = len(te)
            row[f"{label}_test_netr"] = te.net_r.mean() if len(te) else np.nan
            fold_train[label][name] = row[f"{label}_train_netr"]
            fold_test[label][name] = row[f"{label}_test_netr"]

        rows.append(row)

    res = pd.DataFrame(rows).sort_values("net_r_is", ascending=False).reset_index(drop=True)
    res.to_csv(HERE / "results.csv", index=False)
    print(f"[write] {HERE / 'results.csv'} ({len(res)} rows)")

    # ------------------------------------------------------------------
    # transfer statistic: per-fold Spearman(train rank, test rank) across SWEEP_CELLS only
    # (exclude the dist0 sanity cell -- it's not part of the coordinate search)
    # ------------------------------------------------------------------
    per_fold_rho = []
    pooled_train_rank = []
    pooled_test_rank = []
    fold_rho_rows = []
    for label, b, test_end in folds:
        names = [n for n in SWEEP_CELLS if n in fold_train[label]]
        tr_vals = pd.Series({n: fold_train[label][n] for n in names})
        te_vals = pd.Series({n: fold_test[label][n] for n in names})
        mask = tr_vals.notna() & te_vals.notna()
        tr_vals, te_vals = tr_vals[mask], te_vals[mask]
        if len(tr_vals) < 4:
            continue
        rho, p = spearmanr(tr_vals, te_vals)
        per_fold_rho.append(rho)
        fold_rho_rows.append({"fold": label, "n_cells": len(tr_vals), "spearman_rho": rho, "p": p})
        pooled_train_rank.extend(tr_vals.rank().values)
        pooled_test_rank.extend(te_vals.rank().values)

    fold_rho_df = pd.DataFrame(fold_rho_rows)
    mean_rho = np.mean(per_fold_rho) if per_fold_rho else np.nan
    median_rho = np.median(per_fold_rho) if per_fold_rho else np.nan
    pooled_rho, pooled_p = spearmanr(pooled_train_rank, pooled_test_rank) if pooled_train_rank else (np.nan, np.nan)

    fold_rho_df.to_csv(HERE / "transfer_fold_rho.csv", index=False)
    print("[transfer] per-fold rho:", fold_rho_rows)
    print(f"[transfer] mean={mean_rho:.3f} median={median_rho:.3f} pooled={pooled_rho:.3f} (p={pooled_p:.3g})")

    # per-dimension transfer rho (secondary diagnostic): within each dimension's own cells
    # (including 'current' as the dimension's baseline point), pooled across folds
    dim_rho_rows = []
    dims = sorted(set(v[0] for v in CELL_META.values()) - {"sanity"})
    for dim in dims:
        names = [n for n, (d, _) in CELL_META.items() if d == dim or n == "current"]
        names = [n for n in names if n in cells]
        if len(names) < 4:
            continue
        ptr, pte = [], []
        for label, b, test_end in folds:
            tr_vals = pd.Series({n: fold_train[label][n] for n in names if n in fold_train[label]})
            te_vals = pd.Series({n: fold_test[label][n] for n in names if n in fold_test[label]})
            mask = tr_vals.notna() & te_vals.notna()
            tr_vals, te_vals = tr_vals[mask], te_vals[mask]
            if len(tr_vals) < 3:
                continue
            ptr.extend(tr_vals.rank().values)
            pte.extend(te_vals.rank().values)
        if len(ptr) >= 6:
            rho, p = spearmanr(ptr, pte)
            dim_rho_rows.append({"dimension": dim, "n_cells": len(names), "pooled_rho": rho, "p": p})
    dim_rho_df = pd.DataFrame(dim_rho_rows)
    dim_rho_df.to_csv(HERE / "transfer_dim_rho.csv", index=False)
    print("[transfer by dimension]\n", dim_rho_df)

    # ------------------------------------------------------------------
    # write everything needed for AUDIT_C.md as a side-channel text file the
    # main process will read to compose the final report
    # ------------------------------------------------------------------
    summary = {
        "global_cutoff": str(global_cutoff),
        "n_cells": len(res),
        "mean_fold_rho": mean_rho,
        "median_fold_rho": median_rho,
        "pooled_rho": pooled_rho,
        "pooled_p": pooled_p,
    }
    import json
    (HERE / "transfer_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("[done]")


if __name__ == "__main__":
    main()
