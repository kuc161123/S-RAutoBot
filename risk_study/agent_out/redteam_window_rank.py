#!/usr/bin/env python3
"""AGENT-RED probe 2: is the HOLDOUT 2-month window just an unlucky draw?

Compute mean net R (and gross R) over EVERY overlapping 2-month window (61-day, stepped
weekly) in the full 2023-06-01..2026-07-26 history, then rank the actual HOLDOUT
(2026-05-25..2026-07-26) window against that empirical distribution. If many historical
2-month windows were equally bad, "the edge is dead" over-claims -- it could be a normal
bad patch that the strategy has recovered from before.

Also do the NON-overlapping-window version, since overlapping windows are pseudo-replicates
(the step is small vs the window), and report both.
"""
import numpy as np
import pandas as pd

UNI = "/Users/lualakol/AutoTrading Bot/risk_study/universe_chopBOS.parquet"
OUT = "/Users/lualakol/AutoTrading Bot/risk_study/agent_out"
COST = 0.00341
WINDOW_DAYS = 61  # ~2 months, matches HOLDOUT span (2026-05-25 to 2026-07-26 = 62 days)


def net_r(d, cost):
    return d.r_result - cost / d.stop_frac


def main():
    d = pd.read_parquet(UNI)
    d["net_r"] = net_r(d, COST)
    t_start, t_end = d.entry_time.min(), d.entry_time.max()
    print(f"universe span: {t_start} .. {t_end}")

    hold_lo, hold_hi = pd.Timestamp("2026-05-25"), pd.Timestamp("2026-07-26")
    hold = d[(d.entry_time >= hold_lo) & (d.entry_time < hold_hi)]
    hold_gross, hold_net = hold.r_result.mean(), hold.net_r.mean()
    print(f"HOLDOUT actual: n={len(hold)}  gross_mean={hold_gross:+.4f}  net_mean={hold_net:+.4f}")

    # ---- overlapping windows, stepped weekly ----
    starts = pd.date_range(t_start, t_end - pd.Timedelta(days=WINDOW_DAYS), freq="7D")
    rows = []
    for s in starts:
        e = s + pd.Timedelta(days=WINDOW_DAYS)
        sub = d[(d.entry_time >= s) & (d.entry_time < e)]
        if len(sub) < 20:
            continue
        rows.append(dict(start=s, end=e, n=len(sub),
                          gross_mean=sub.r_result.mean(), net_mean=sub.net_r.mean(),
                          gross_sum=sub.r_result.sum(), net_sum=sub.net_r.sum()))
    wf = pd.DataFrame(rows)
    wf.to_csv(f"{OUT}/window_scan_overlapping.csv", index=False)

    print(f"\n{len(wf)} overlapping 61-day windows (weekly step)")
    for col, label, hv in [("gross_mean", "gross", hold_gross), ("net_mean", "net@34.1bps", hold_net)]:
        rank = (wf[col] <= hv).sum()
        frac = rank / len(wf) * 100
        print(f"  {label}: HOLDOUT rank {rank}/{len(wf)} from the bottom "
              f"({frac:.1f}% of windows are AS BAD OR WORSE)")
        print(f"    distribution: min={wf[col].min():+.4f}  p10={wf[col].quantile(.1):+.4f}  "
              f"p25={wf[col].quantile(.25):+.4f}  median={wf[col].median():+.4f}  "
              f"p75={wf[col].quantile(.75):+.4f}  max={wf[col].max():+.4f}")
        n_neg = (wf[col] < 0).sum()
        print(f"    {n_neg}/{len(wf)} windows ({n_neg/len(wf)*100:.1f}%) have negative mean R at all")

    # ---- non-overlapping windows (true independent replicates) ----
    print("\n" + "=" * 90)
    print("NON-OVERLAPPING 61-day windows (independent, no pseudo-replication)")
    print("=" * 90)
    nrows = []
    s = t_start
    while s + pd.Timedelta(days=WINDOW_DAYS) <= t_end + pd.Timedelta(days=WINDOW_DAYS):
        e = s + pd.Timedelta(days=WINDOW_DAYS)
        sub = d[(d.entry_time >= s) & (d.entry_time < e)]
        if len(sub) >= 20:
            nrows.append(dict(start=s, end=e, n=len(sub),
                               gross_mean=sub.r_result.mean(), net_mean=sub.net_r.mean()))
        s = e
    nf = pd.DataFrame(nrows)
    nf.to_csv(f"{OUT}/window_scan_nonoverlapping.csv", index=False)
    print(nf.to_string())
    for col, label, hv in [("gross_mean", "gross", hold_gross), ("net_mean", "net@34.1bps", hold_net)]:
        n_worse = (nf[col] <= hv).sum()
        print(f"\n  {label}: {n_worse}/{len(nf)} non-overlapping windows are as bad or worse than "
              f"HOLDOUT ({hv:+.4f})")
        n_neg = (nf[col] < 0).sum()
        print(f"    {n_neg}/{len(nf)} of the {len(nf)} independent 2-month windows in the full "
              f"3yr history are net-negative at all")

    # Recovery check: for each historical window as bad as HOLDOUT, what happened in the
    # FOLLOWING 2 months?
    print("\n" + "=" * 90)
    print("RECOVERY CHECK: for overlapping windows at least as bad as HOLDOUT, what did the")
    print("NEXT 61 days (non-overlapping, immediately following) look like?")
    print("=" * 90)
    bad = wf[wf.net_mean <= hold_net].copy()
    rec_rows = []
    for _, r in bad.iterrows():
        nxt_lo, nxt_hi = r["end"], r["end"] + pd.Timedelta(days=WINDOW_DAYS)
        nxt = d[(d.entry_time >= nxt_lo) & (d.entry_time < nxt_hi)]
        if len(nxt) < 10:
            continue
        rec_rows.append(dict(bad_start=r["start"], bad_net_mean=r["net_mean"],
                              next_n=len(nxt), next_net_mean=nxt.net_r.mean()))
    rec = pd.DataFrame(rec_rows)
    if len(rec):
        rec.to_csv(f"{OUT}/window_recovery_check.csv", index=False)
        print(rec.to_string())
        print(f"\n  Of {len(rec)} bad-window instances (net_mean <= HOLDOUT's {hold_net:+.4f}), "
              f"{(rec.next_net_mean > 0).sum()}/{len(rec)} were followed by a POSITIVE next "
              f"61-day net mean ({(rec.next_net_mean > 0).mean()*100:.1f}%).")
    else:
        print("  no comparably-bad windows with 61 days of follow-up data found")


if __name__ == "__main__":
    main()
