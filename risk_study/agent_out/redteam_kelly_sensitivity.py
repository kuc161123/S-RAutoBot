#!/usr/bin/env python3
"""AGENT-RED probe 3: is the cluster-Kelly f* sensitive to the (arbitrary) bucket width?

kelly.py used 24h buckets. Recompute at 6h, 12h, 24h, 48h, 168h (1 week) and see how much
f* moves. Also compare against the simulator's own best-ROI/DD f from sweep_main.csv.
"""
import numpy as np
import pandas as pd

UNI = "/Users/lualakol/AutoTrading Bot/risk_study/universe_chopBOS.parquet"
OUT = "/Users/lualakol/AutoTrading Bot/risk_study/agent_out"
SWEEP = "/Users/lualakol/AutoTrading Bot/risk_study/results/sweep_main.csv"
COST = 0.00341

PERIODS = {
    "DEV": ("2023-06-01", "2025-07-01"),
    "VAL": ("2025-07-01", "2026-05-25"),
    "HOLDOUT": ("2026-05-25", "2026-07-26"),
    "FULL": ("2023-06-01", "2026-07-26"),
}
BUCKETS_H = [6, 12, 24, 48, 168]


def net_r(d, cost):
    return d.r_result - cost / d.stop_frac


def kelly_f(r, lo=1e-6, hi=0.5):
    r = np.asarray(r, dtype=float)
    if r.mean() <= 0:
        return 0.0
    worst = r.min()
    cap = (0.999 / -worst) if worst < 0 else hi
    hi_ = min(hi, cap)

    def g(f):
        v = 1.0 + f * r
        if np.any(v <= 0):
            return -np.inf
        return np.mean(np.log(v))

    gr = (np.sqrt(5) - 1) / 2
    a, b = lo, hi_
    for _ in range(200):
        c, dd = b - gr * (b - a), a + gr * (b - a)
        if g(c) < g(dd):
            a = c
        else:
            b = dd
    return (a + b) / 2


def cluster_returns(d, cost, hours):
    """Aggregate net R by exit-time bucket of width `hours`, floor-based (epoch-anchored).

    NOTE: the original kelly.py used pandas .resample(f"{hours}h") intersected with
    .dt.floor(f"{hours}h") via .isin() -- those two use DIFFERENT anchor points
    (resample anchors at the first timestamp in the series; floor anchors at the Unix
    epoch), so for any bucket width that isn't a clean divisor of the data's start-offset
    (confirmed broken for 48h and 168h, silently returns zero overlapping buckets -> NaN
    Kelly f*), the two never intersect. 6h/12h/24h happened to work by coincidence (they
    evenly divide a day and the data starts at midnight). This rewrite groups directly by
    the floor value with no resample/isin dance, so every bucket width is handled the same
    way and none silently drops all buckets.
    """
    x = d.copy()
    x["nr"] = net_r(x, cost)
    x["bucket"] = x.exit_time.dt.floor(f"{hours}h")
    return x.groupby("bucket")["nr"].sum().to_numpy()


def main():
    d = pd.read_parquet(UNI)
    rows = []
    print(f"{'period':<9}{'f*_single':>11}" + "".join(f"{'f*_'+str(h)+'h':>10}" for h in BUCKETS_H))
    for name, (t0, t1) in PERIODS.items():
        s = d[(d.entry_time >= t0) & (d.entry_time < t1)]
        r = net_r(s, COST).to_numpy()
        fs = kelly_f(r)
        line = f"{name:<9}{fs*100:>10.3f}%"
        rec = dict(period=name, kelly_single_pct=fs * 100)
        for h in BUCKETS_H:
            cl = cluster_returns(s, COST, h)
            fc = kelly_f(cl) if len(cl) > 20 else np.nan
            line += f"{(fc*100 if fc==fc else float('nan')):>9.3f}%"
            rec[f"kelly_{h}h_pct"] = fc * 100 if fc == fc else np.nan
        print(line)
        rows.append(rec)
    kdf = pd.DataFrame(rows)
    kdf.to_csv(f"{OUT}/kelly_bucket_sensitivity.csv", index=False)

    # spread of f* across bucket choices, per period
    print("\nSpread of cluster-Kelly f* across bucket widths (6h..168h), per period:")
    for _, r in kdf.iterrows():
        vals = [r[f"kelly_{h}h_pct"] for h in BUCKETS_H if pd.notna(r[f"kelly_{h}h_pct"])]
        if vals:
            print(f"  {r.period:<9} min={min(vals):.3f}%  max={max(vals):.3f}%  "
                  f"ratio max/min={max(vals)/max(min(vals),1e-9):.2f}x")

    # compare to simulator's own best-ROI/DD f
    print("\n" + "=" * 90)
    print("Simulator's own argmax ROI/DD f, vs cluster-Kelly range, per window")
    print("=" * 90)
    sw = pd.read_csv(SWEEP)
    for name in ["DEV", "VAL", "HOLDOUT", "FULL"]:
        sub = sw[sw.window == name]
        if sub.empty:
            continue
        best = sub.loc[sub.roi_dd_ratio.idxmax()]
        krow = kdf[kdf.period == name].iloc[0]
        cl_vals = [krow[f"kelly_{h}h_pct"] for h in BUCKETS_H if pd.notna(krow[f"kelly_{h}h_pct"])]
        print(f"{name:<9} simulator argmax f = {best.risk_pct:.2f}%  (ROI/DD={best.roi_dd_ratio:.2f}, "
              f"maxDD={best.max_dd_pct:.1f}%)   cluster-Kelly range = "
              f"[{min(cl_vals) if cl_vals else float('nan'):.3f}%, "
              f"{max(cl_vals) if cl_vals else float('nan'):.3f}%]")


if __name__ == "__main__":
    main()
