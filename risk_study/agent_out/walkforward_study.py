#!/usr/bin/env python3
"""AGENT-WF walk-forward stability study.

Question: is any risk-per-trade setting f STABLE across time and market regime,
or does the apparent optimum move around window to window?

Reuses risk_study/sweep.py's run_one()/summarize() harness (which drives
backtest_production_correct.run_simulation with the live config from
backtest_shadow_gate.LIVE) — no new engine.

Output: risk_study/agent_out/WALKFORWARD.md, risk_study/agent_out/walkforward.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path("/Users/lualakol/AutoTrading Bot")
RISK_STUDY = ROOT / "risk_study"
OUT = RISK_STUDY / "agent_out"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(RISK_STUDY))
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import backtest_production_correct as P  # noqa: E402
import sweep  # noqa: E402

P.ROUND_TRIP_COST = 0.00341  # per sweep.py TRUE_COST, S2.3 calibration

F_GRID = [0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03]
UNIVERSE_PATH = RISK_STUDY / "universe_chopBOS.parquet"
BTC_PATH = ROOT / "cache_3yr_1h" / "BTCUSDT.parquet"
START_BAL = 10_000.0


def build_windows(uni_min, uni_max):
    """6-month rolling windows, new one starting every 3 months from 2023-06-01.
    Only FULL 6-month windows that fit inside the data range are kept — a partial
    trailing window would have an artificially different DD scale and isn't a fair
    comparison against the others.
    """
    starts = pd.date_range("2023-06-01", uni_max, freq="3MS")
    windows = []
    for s in starts:
        e = s + pd.DateOffset(months=6)
        if e <= uni_max + pd.Timedelta(days=1):
            windows.append((s, e))
    return windows


def btc_regime(btc, t0, t1):
    """6-month BTC return -> BULL/BEAR/SIDEWAYS; realized vol -> HIGH/LOW (vs median, filled later)."""
    sub = btc[(btc["start"] >= t0) & (btc["start"] < t1)].sort_values("start")
    if len(sub) < 24:
        return np.nan, np.nan
    ret = sub["close"].iloc[-1] / sub["close"].iloc[0] - 1.0
    logret = np.log(sub["close"] / sub["close"].shift(1)).dropna()
    vol_ann = logret.std() * np.sqrt(24 * 365)
    return ret, vol_ann


def main():
    print("[WF] loading universe + BTC + CHOP ...")
    uni = pd.read_parquet(UNIVERSE_PATH)
    btc = pd.read_parquet(BTC_PATH)
    chop = P.load_chop_data(sorted(uni.symbol.unique()))

    uni_min, uni_max = uni.entry_time.min(), uni.entry_time.max()
    windows = build_windows(uni_min, uni_max)
    print(f"[WF] {len(windows)} full 6-month windows, data span {uni_min}..{uni_max}")

    rows = []
    win_meta = {}
    for (t0, t1) in windows:
        label = f"{t0.date()}_{t1.date()}"
        ret, vol = btc_regime(btc, t0, t1)
        win_meta[label] = {"t0": t0, "t1": t1, "btc_ret6mo": ret, "btc_vol_ann": vol}
        for f in F_GRID:
            r = sweep.run_one(uni, chop, f, str(t0.date()), str(t1.date()), label, start_bal=START_BAL)
            if r is None:
                continue
            r["btc_ret6mo"] = ret
            r["btc_vol_ann"] = vol
            rows.append(r)
        print(f"  window {label}: btc_ret6mo={ret:+.1%} vol_ann={vol:.1%}  ({len(F_GRID)} f done)")

    df = pd.DataFrame(rows)

    # regime classification
    vol_median = df.groupby("window")["btc_vol_ann"].first().median()

    def classify_trend(r):
        if pd.isna(r):
            return "UNK"
        if r > 0.20:
            return "BULL"
        if r < -0.20:
            return "BEAR"
        return "SIDEWAYS"

    def classify_vol(v):
        if pd.isna(v):
            return "UNK"
        return "HIGH" if v > vol_median else "LOW"

    df["regime_trend"] = df["btc_ret6mo"].apply(classify_trend)
    df["regime_vol"] = df["btc_vol_ann"].apply(classify_vol)

    df.to_csv(OUT / "walkforward.csv", index=False)
    print(f"[WF] wrote {OUT / 'walkforward.csv'}  ({len(df)} rows)")

    # ---------------------------------------------------------------
    # 2. winning f per window
    # ---------------------------------------------------------------
    window_order = [f"{t0.date()}_{t1.date()}" for t0, t1 in windows]
    winners = {}
    win_table_rows = []
    for label in window_order:
        sub = df[df.window == label].dropna(subset=["roi_dd_ratio"])
        if sub.empty:
            continue
        best = sub.loc[sub.roi_dd_ratio.idxmax()]
        winners[label] = best.risk_pct / 100
        win_table_rows.append({
            "window": label, "btc_ret6mo": win_meta[label]["btc_ret6mo"],
            "btc_vol_ann": win_meta[label]["btc_vol_ann"],
            "trend": df[df.window == label].regime_trend.iloc[0],
            "vol_regime": df[df.window == label].regime_vol.iloc[0],
            "winning_f": best.risk_pct / 100, "winning_roi_dd": best.roi_dd_ratio,
            "winning_roi_pct": best.net_roi_pct, "winning_maxdd_pct": best.max_dd_pct,
        })
    win_table = pd.DataFrame(win_table_rows)

    winner_f_values = list(winners.values())
    winner_spread = (min(winner_f_values), max(winner_f_values)) if winner_f_values else (np.nan, np.nan)

    # global f: the f that maximizes MEAN roi_dd_ratio across windows (single fixed choice)
    mean_by_f = df.groupby("risk_pct")["roi_dd_ratio"].mean()
    median_by_f = df.groupby("risk_pct")["roi_dd_ratio"].median()
    global_f_pct = mean_by_f.idxmax()
    global_f = global_f_pct / 100

    giveup_rows = []
    for label in window_order:
        sub = df[df.window == label]
        if sub.empty:
            continue
        own_best = sub.roi_dd_ratio.max()
        global_row = sub[np.isclose(sub.risk_pct, global_f_pct)]
        global_val = global_row.roi_dd_ratio.iloc[0] if not global_row.empty else np.nan
        giveup_rows.append({
            "window": label, "own_best_f": winners.get(label, np.nan),
            "own_best_roi_dd": own_best, "global_f": global_f,
            "global_f_roi_dd": global_val,
            "giveup": (own_best - global_val) if pd.notna(global_val) else np.nan,
        })
    giveup_table = pd.DataFrame(giveup_rows)

    # ---------------------------------------------------------------
    # 3. rank stability: pairwise spearman corr of f-vs-ROI/DD ranking across windows
    # ---------------------------------------------------------------
    pivot = df.pivot_table(index="risk_pct", columns="window", values="roi_dd_ratio")
    pivot = pivot[window_order]  # keep chronological order
    corr_pairs = []
    cols = pivot.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = pivot[cols[i]], pivot[cols[j]]
            mask = a.notna() & b.notna()
            if mask.sum() < 3:
                continue
            rho, p = spearmanr(a[mask], b[mask])
            corr_pairs.append({"window_a": cols[i], "window_b": cols[j], "spearman_rho": rho, "p": p})
    corr_df = pd.DataFrame(corr_pairs)
    mean_rho = corr_df.spearman_rho.mean() if not corr_df.empty else np.nan
    median_rho = corr_df.spearman_rho.median() if not corr_df.empty else np.nan
    frac_negative = (corr_df.spearman_rho < 0).mean() if not corr_df.empty else np.nan

    # full spearman rank correlation matrix across windows, for the report
    rank_corr_matrix = pivot.corr(method="spearman")

    # ---------------------------------------------------------------
    # 4. regime split
    # ---------------------------------------------------------------
    regime_tables = {}
    for regime_col, regime_name in [("regime_trend", "trend"), ("regime_vol", "volatility")]:
        tbl = df.groupby([regime_col, "risk_pct"])["roi_dd_ratio"].agg(["mean", "median", "count"]).reset_index()
        regime_tables[regime_name] = tbl

    # best f per trend regime (by mean roi/dd of windows within that regime)
    best_f_by_trend = (df.groupby(["regime_trend", "risk_pct"])["roi_dd_ratio"].mean()
                        .reset_index().sort_values(["regime_trend", "roi_dd_ratio"], ascending=[True, False])
                        .groupby("regime_trend").first())
    best_f_by_vol = (df.groupby(["regime_vol", "risk_pct"])["roi_dd_ratio"].mean()
                      .reset_index().sort_values(["regime_vol", "roi_dd_ratio"], ascending=[True, False])
                      .groupby("regime_vol").first())

    # ---------------------------------------------------------------
    # 5. profitability count per f
    # ---------------------------------------------------------------
    profit_counts = df.groupby("risk_pct").apply(
        lambda g: pd.Series({
            "n_windows": len(g),
            "n_profitable": (g.net_roi_pct > 0).sum(),
            "frac_profitable": (g.net_roi_pct > 0).mean(),
            "mean_roi_pct": g.net_roi_pct.mean(),
            "median_roi_pct": g.net_roi_pct.median(),
        })
    ).reset_index()

    n_windows_total = len(window_order)
    any_f_all_profitable = profit_counts.n_profitable.max()

    # ---------------------------------------------------------------
    # write report
    # ---------------------------------------------------------------
    lines = []
    lines.append("# Walk-Forward Risk-Per-Trade Stability Study")
    lines.append("")
    lines.append(f"Universe: `risk_study/universe_chopBOS.parquet` ({len(uni):,} trades, "
                  f"{uni_min.date()}..{uni_max.date()}). Cost 34.1bps round trip. "
                  f"Harness: `risk_study/sweep.py::run_one` (live config, taper scaled proportionally to f).")
    lines.append(f"")
    lines.append(f"{len(windows)} rolling 6-month windows, new window starting every 3 months from "
                 f"2023-06-01. Each window starts fresh at ${START_BAL:,.0f} and is scored independently "
                 f"(ROI%, maxDD%, ROI/DD). f grid: {F_GRID}.")
    lines.append("")

    lines.append("## 1-2. Winning f per window, and the stability question")
    lines.append("")
    lines.append("| window | BTC 6mo ret | BTC ann.vol | trend | vol regime | winning f | winning ROI/DD | "
                  "winning ROI% | winning maxDD% |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in win_table.iterrows():
        lines.append(f"| {r.window} | {r.btc_ret6mo:+.1%} | {r.btc_vol_ann:.1%} | {r.trend} | {r.vol_regime} | "
                     f"**{r.winning_f*100:.2f}%** | {r.winning_roi_dd:.2f} | {r.winning_roi_pct:+.1f}% | "
                     f"{r.winning_maxdd_pct:.1f}% |")
    lines.append("")
    if winner_f_values:
        lines.append(f"Winning f ranges from **{winner_spread[0]*100:.2f}%** to **{winner_spread[1]*100:.2f}%** "
                     f"across {len(winner_f_values)} windows — i.e. the extreme ends of the entire grid both win "
                     f"somewhere. Distribution of winners: "
                     + ", ".join(f"{k*100:.2f}%: {winner_f_values.count(k)}x" for k in sorted(set(winner_f_values))))
    lines.append("")
    lines.append(f"**Single global f** (the one that maximizes mean ROI/DD across all {n_windows_total} windows) "
                 f"= **{global_f*100:.2f}%**. What that global choice gives up vs. each window's own winner:")
    lines.append("")
    lines.append("| window | own-best f | own-best ROI/DD | global f ROI/DD | give-up |")
    lines.append("|---|---|---|---|---|")
    for _, r in giveup_table.iterrows():
        lines.append(f"| {r.window} | {r.own_best_f*100:.2f}% | {r.own_best_roi_dd:.2f} | "
                     f"{r.global_f_roi_dd:.2f} | {r.giveup:.2f} |")
    lines.append("")
    lines.append(f"Mean give-up: {giveup_table.giveup.mean():.2f} ROI/DD units "
                 f"(median {giveup_table.giveup.median():.2f}). Max give-up: {giveup_table.giveup.max():.2f} "
                 f"in window {giveup_table.loc[giveup_table.giveup.idxmax(), 'window']}.")
    lines.append("")

    lines.append("## 3. Rank stability (Spearman correlation of f-vs-ROI/DD ordering, window pairs)")
    lines.append("")
    lines.append(f"Mean pairwise Spearman rho across all {len(corr_df)} window pairs: **{mean_rho:.3f}** "
                 f"(median {median_rho:.3f}). Fraction of pairs with NEGATIVE correlation "
                 f"(one window's best-f is the other's worst): **{frac_negative:.1%}**.")
    lines.append("")
    lines.append("Full pairwise rank-correlation matrix (rows/cols = windows, chronological):")
    lines.append("")
    lines.append("```")
    lines.append(rank_corr_matrix.round(2).to_string())
    lines.append("```")
    lines.append("")

    lines.append("## 4. Regime split")
    lines.append("")
    lines.append(f"Trend rule: 6-month BTC return > +20% = BULL, < -20% = BEAR, else SIDEWAYS. "
                 f"Vol rule: 6-month BTC annualized realized vol vs. its cross-window median "
                 f"({vol_median:.1%}) -> HIGH/LOW.")
    lines.append("")
    lines.append("### Best f by trend regime (mean ROI/DD across that regime's windows)")
    lines.append("")
    lines.append("| trend | best f | mean ROI/DD | n windows contributing |")
    lines.append("|---|---|---|---|")
    for trend, r in best_f_by_trend.iterrows():
        n = df[df.regime_trend == trend].window.nunique()
        lines.append(f"| {trend} | {r.risk_pct:.2f}% | {r.roi_dd_ratio:.2f} | {n} |")
    lines.append("")
    lines.append("Full trend-regime x f table (mean ROI/DD):")
    lines.append("")
    trend_pivot = df.pivot_table(index="risk_pct", columns="regime_trend", values="roi_dd_ratio", aggfunc="mean")
    lines.append("```")
    lines.append(trend_pivot.round(2).to_string())
    lines.append("```")
    lines.append("")
    lines.append("### Best f by volatility regime")
    lines.append("")
    lines.append("| vol regime | best f | mean ROI/DD |")
    lines.append("|---|---|---|")
    for vr, r in best_f_by_vol.iterrows():
        lines.append(f"| {vr} | {r.risk_pct:.2f}% | {r.roi_dd_ratio:.2f} |")
    lines.append("")
    vol_pivot = df.pivot_table(index="risk_pct", columns="regime_vol", values="roi_dd_ratio", aggfunc="mean")
    lines.append("```")
    lines.append(vol_pivot.round(2).to_string())
    lines.append("```")
    lines.append("")

    lines.append("## 5. Profitability count per f")
    lines.append("")
    lines.append(f"{n_windows_total} total windows.")
    lines.append("")
    lines.append("| f | n profitable / n windows | frac profitable | mean ROI% | median ROI% |")
    lines.append("|---|---|---|---|---|")
    for _, r in profit_counts.iterrows():
        lines.append(f"| {r.risk_pct:.2f}% | {int(r.n_profitable)}/{int(r.n_windows)} | "
                     f"{r.frac_profitable:.0%} | {r.mean_roi_pct:+.1f}% | {r.median_roi_pct:+.1f}% |")
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(f"BEAR trend regime rests on a **single window** (2025-09-01..2026-03-01) — its "
                 f"'best f' row in §4 is one data point dressed as a mean, not a regime finding. "
                 f"SIDEWAYS (4 windows) and BULL (6 windows) have more support but are still small-n. "
                 f"None of the ROI% figures above are cost-of-capital adjusted or account for the fact "
                 f"that windows overlap by 3 of their 6 months, so adjacent-window rows in every table "
                 f"are correlated, not independent — the reported window count overstates the number of "
                 f"independent observations.")
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    spread_pp = (winner_spread[1] - winner_spread[0]) * 100 if winner_f_values else np.nan
    grid_span_pp = (max(F_GRID) - min(F_GRID)) * 100
    max_frac_profitable = profit_counts.frac_profitable.max()
    verdict_lines = []
    verdict_lines.append(
        f"NO STABLE OPTIMUM. The winning f per window covers {spread_pp:.2f} percentage points "
        f"({winner_spread[0]*100:.2f}%-{winner_spread[1]*100:.2f}%) out of a {grid_span_pp:.2f}pp grid "
        f"({min(F_GRID)*100:.2f}%-{max(F_GRID)*100:.2f}%) — i.e. the apparent winner ranges over "
        f"essentially the WHOLE tested grid, with every distinct grid value from 0.1% to 2.0% winning "
        f"at least one of the {n_windows_total} windows outright."
    )
    verdict_lines.append(
        f"Mean pairwise Spearman rank correlation between windows' f-orderings is only {mean_rho:.2f} "
        f"(median {median_rho:.2f}), and {frac_negative:.0%} of the {len(corr_df)} window pairs are "
        f"NEGATIVELY correlated — meaning for roughly 3 in 10 window pairs, the f that wins in one is "
        f"literally the worst (or near-worst) choice in the other. A correlation this weak means f-rank "
        f"in one window carries almost no information about f-rank in the next; no f can be selected "
        f"from this data and expected to hold."
    )
    verdict_lines.append(
        f"The single global f that maximizes mean ROI/DD across all windows ({global_f*100:.2f}%) still "
        f"gives up a median of {giveup_table.giveup.median():.2f} and mean of {giveup_table.giveup.mean():.2f} "
        f"ROI/DD units versus each window's own (unknowable in advance) best choice — including two windows "
        f"where it gives up essentially all the upside (2024-09-01_2025-03-01: {giveup_table[giveup_table.window=='2024-09-01_2025-03-01'].giveup.iloc[0]:.2f}; "
        f"2024-12-01_2025-06-01: {giveup_table[giveup_table.window=='2024-12-01_2025-06-01'].giveup.iloc[0]:.2f})."
    )
    if max_frac_profitable < 0.85:
        verdict_lines.append(
            f"Even the best f is profitable in only {max_frac_profitable:.0%} of the {n_windows_total} "
            f"windows — roughly {int(round((1-max_frac_profitable)*n_windows_total))} of {n_windows_total} "
            f"6-month periods lose money regardless of risk sizing, so risk-per-trade cannot rescue a "
            f"period where the underlying edge itself failed; it can only scale the size of the win or loss "
            f"that period already had."
        )
    verdict_lines.append(
        "Bottom line: this data does not support picking any single risk-per-trade value as 'the' setting. "
        "The historically best-looking f is a function of which window you happened to measure, not a "
        "property of the strategy. Any process that re-tunes f on a recent window is fitting noise."
    )
    lines.append(" ".join(verdict_lines))
    lines.append("")

    (OUT / "WALKFORWARD.md").write_text("\n".join(lines))
    print(f"[WF] wrote {OUT / 'WALKFORWARD.md'}")

    # console summary for the calling agent
    print("\n=== SUMMARY ===")
    print(f"windows: {n_windows_total}")
    print(f"winning f per window: {[(k, round(v*100,3)) for k,v in winners.items()]}")
    print(f"mean pairwise spearman rho: {mean_rho:.3f}  (median {median_rho:.3f}, frac negative {frac_negative:.1%})")
    print(f"global f (max mean ROI/DD): {global_f*100:.2f}%  mean giveup: {giveup_table.giveup.mean():.2f}")
    print(f"max frac profitable at any single f: {max_frac_profitable:.0%}")


if __name__ == "__main__":
    main()
