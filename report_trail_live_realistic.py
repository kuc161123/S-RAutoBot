#!/usr/bin/env python3
"""Trailing vs base — run exactly as the live bot runs, no extra strictness.

The deliberate framing here is different from the gauntlet. The gauntlet asked
"is trailing a robust universal improvement?" and weighted 2023-2024 as heavily as now.
This asks the practical question instead:

    if I run THIS bot, with THIS config, THESE symbols, THESE protections,
    over the period that resembles the market I'm actually in — what happens?

Everything live is kept: 0.30% base risk, live taper schedule, net_directional_cap 0.10,
gross_open_risk_cap 0.30, btc_short_gate, long_bull_boost 1.3, regime sizing, chop filter.
Only the exit rule changes between columns, against identical entries.
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from build_trail_universe_wide import VARIANTS

START = 1500.0
BASE_RT = 2 * (0.0006 + 0.0003)
UNI = ROOT / "trail_wide_universe.parquet"
OOS = pd.Timestamp("2026-05-25")
END = pd.Timestamp("2026-07-25")
NAMES = [v[0] for v in VARIANTS]

_P = None
_CHOP = None
_D = None


def _init():
    global _P, _CHOP, _D
    import backtest_production_correct as P
    from backtest_shadow_gate import LIVE as LIVE_KW
    _P = (P, LIVE_KW)
    _D = pd.read_parquet(UNI)
    _D["entry_time"] = pd.to_datetime(_D["entry_time"])
    _CHOP = P.load_chop_data(sorted(_D.symbol.unique()))


def _frame(d, name):
    out = d[["entry_time", "entry_price", "sl_price", "side", "symbol",
             "btc_bull", "btc_impulse"]].copy()
    out["exit_time"] = pd.to_datetime(d[f"x_{name}"])
    out["r_result"] = d[f"r_{name}"]
    return out.dropna(subset=["exit_time", "r_result"])


def _run(name, t0, t1, cost, placebo, start):
    P, LIVE_KW = _P
    d = _D[_D.placebo == placebo]
    d = d[(d.entry_time >= t0) & (d.entry_time < t1)]
    if len(d) < 20:
        return None
    sub = _frame(d, name)
    P.ROUND_TRIP_COST = BASE_RT * cost
    P.STARTING_BALANCE = start
    kw = dict(LIVE_KW, starting_balance=start,
              btc_bull_col="btc_bull", btc_short_col="btc_impulse")
    r = P.run_simulation(sub.sort_values("entry_time").reset_index(drop=True), _CHOP, **kw)
    t = r["entered_trades"]
    if not t:
        return None
    gp = sum(x["pnl"] for x in t if x["pnl"] > 0)
    gl = abs(sum(x["pnl"] for x in t if x["pnl"] < 0))
    hold = np.mean([(pd.Timestamp(x["exit_time"]) - pd.Timestamp(x["entry_time"]))
                    .total_seconds() / 3600 for x in t])
    ev = []
    for x in t:
        ev.append((pd.Timestamp(x["entry_time"]), 1))
        ev.append((pd.Timestamp(x["exit_time"]), -1))
    ev.sort()
    cur = peak = 0
    for _, dl in ev:
        cur += dl
        peak = max(peak, cur)
    wins = sum(1 for x in t if x["pnl"] > 0)
    return dict(final=r["final_effective"], roi=r["final_effective"] / start - 1,
                dd=r["max_dd_pct"], n=len(t), wr=wins / len(t),
                pf=(gp / gl) if gl else float("inf"), hold=hold, peak=peak,
                monthly=dict(r["monthly_pnl"]))


def job(a):
    key, name, t0, t1, cost, placebo, start = a
    try:
        return key, name, _run(name, t0, t1, cost, placebo, start)
    except Exception as e:                                    # noqa: BLE001
        return key, name, None


def table(res, keys, title, note=""):
    print("\n" + "=" * 112)
    print(title)
    if note:
        print(note)
    print("=" * 112)
    print(f"  {'exit rule':<14}{'final $':>11}{'ROI':>9}{'maxDD':>8}{'ROI/DD':>8}"
          f"{'PF':>7}{'WR':>7}{'hold h':>8}{'trades':>8}{'peak':>6}{'vs base':>11}")
    base = res.get((keys, "base"))
    rows = []
    for nm in NAMES:
        r = res.get((keys, nm))
        if r:
            rows.append((nm, r))
    rows.sort(key=lambda x: -x[1]["final"])
    for nm, r in rows:
        ratio = r["roi"] * 100 / r["dd"] if r["dd"] else 0.0
        vs = (r["final"] - base["final"]) if base else 0.0
        mark = " *" if nm == "base" else ""
        print(f"  {nm + mark:<14}{r['final']:>11,.0f}{r['roi']:>9.0%}{r['dd']:>7.1f}%"
              f"{ratio:>8.1f}{r['pf']:>7.2f}{r['wr']:>7.1%}{r['hold']:>8.1f}"
              f"{r['n']:>8,}{r['peak']:>6}{vs:>+11,.0f}")
    print(f"  {'never trade':<14}{START:>11,.0f}{0.0:>9.0%}{0.0:>7.1f}%{0.0:>8}"
          f"{'—':>7}{'—':>7}{'—':>8}{0:>8}{0:>6}"
          f"{(START - base['final']) if base else 0:>+11,.0f}")


def main():
    d = pd.read_parquet(UNI)
    d["entry_time"] = pd.to_datetime(d["entry_time"])
    real = d[~d.placebo]
    print(f"[LIVE-REAL] {len(real):,} signals · {real.symbol.nunique()} symbols · "
          f"{real.entry_time.min():%Y-%m-%d} .. {real.entry_time.max():%Y-%m-%d}")
    print(f"[LIVE-REAL] {len(VARIANTS)} exit rules · live config, live protections, "
          f"${START:,.0f} start\n", flush=True)

    periods = {
        "15mo": (END - pd.DateOffset(months=15), END),
        "10mo": (END - pd.DateOffset(months=10), END),
        "6mo": (END - pd.DateOffset(months=6), END),
        "oos": (OOS, END),
        "full": (pd.Timestamp("2023-06-01"), END),
    }
    jobs = []
    for k, (t0, t1) in periods.items():
        for nm in NAMES:
            jobs.append((k, nm, t0, t1, 1.0, False, START))
    # cost band + placebo on the 15-month window
    for nm in NAMES:
        jobs.append(("15mo_c15", nm, *periods["15mo"], 1.5, False, START))
        jobs.append(("15mo_c20", nm, *periods["15mo"], 2.0, False, START))
        jobs.append(("15mo_plc", nm, *periods["15mo"], 1.0, True, START))
    # rolling overlapping 12-month windows, monthly starts
    rolls = pd.date_range("2023-07-01", END - pd.DateOffset(months=12), freq="MS")
    for s0 in rolls:
        for nm in NAMES:
            jobs.append((f"roll_{s0:%Y-%m}", nm, s0, s0 + pd.DateOffset(months=12),
                         1.0, False, START))
    # rolling 6-month windows, monthly starts — more windows, and reaches into now
    r6 = pd.date_range("2023-07-01", END - pd.DateOffset(months=6), freq="MS")
    for s0 in r6:
        for nm in NAMES:
            jobs.append((f"r6_{s0:%Y-%m}", nm, s0, s0 + pd.DateOffset(months=6),
                         1.0, False, START))

    print(f"[LIVE-REAL] {len(jobs):,} simulations across "
          f"{max(1, mp.cpu_count() - 1)} workers...", flush=True)
    res = {}
    with mp.Pool(max(1, mp.cpu_count() - 1), initializer=_init) as pool:
        for i, (k, nm, r) in enumerate(pool.imap_unordered(job, jobs, chunksize=2), 1):
            if r:
                res[(k, nm)] = r
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    table(res, "15mo", "LAST 15 MONTHS — the bot as it runs today  (2025-04-25 -> 2026-07-25)",
          "  live config · live protections · $1,500 start · bot's own cost model (1x)")
    table(res, "10mo", "LAST 10 MONTHS  (2025-09-25 -> 2026-07-25)")
    table(res, "6mo", "LAST 6 MONTHS  (2026-01-25 -> 2026-07-25)")
    table(res, "oos", "CLEAN OUT-OF-SAMPLE — after the config's fitting window (2026-05-25 ->)",
          "  the only slice no fitting touched")
    table(res, "full", "FULL HISTORY 2023-06 -> 2026-07  (context: includes two bull runs)")

    # ---- rolling overlapping windows ----
    def rolling(prefix, title, lo=None, hi=None):
        rkeys = sorted({k for k, _ in res if k.startswith(prefix)})
        if lo or hi:
            def ok(k):
                s = k.split("_", 1)[1]
                return (not lo or s >= lo) and (not hi or s <= hi)
            rkeys = [k for k in rkeys if ok(k)]
        if not rkeys:
            return
        print("\n" + "=" * 112)
        print(title)
        print("=" * 112)
        print(f"  {'exit rule':<14}{'windows':>9}{'beat base':>12}{'median $':>12}"
              f"{'base median':>13}{'median DD':>11}{'base DD':>10}")
        bmed = np.median([res[(k, "base")]["final"] for k in rkeys if (k, "base") in res])
        bdd = np.median([res[(k, "base")]["dd"] for k in rkeys if (k, "base") in res])
        summary = []
        for nm in NAMES:
            vals, dds, w, t = [], [], 0, 0
            for k in rkeys:
                a, b = res.get((k, nm)), res.get((k, "base"))
                if not a or not b:
                    continue
                vals.append(a["final"])
                dds.append(a["dd"])
                t += 1
                w += 1 if a["final"] > b["final"] else 0
            if t:
                summary.append((nm, t, w, float(np.median(vals)), float(np.median(dds))))
        summary.sort(key=lambda x: -x[3])
        for nm, t, w, med, mdd in summary:
            mark = " *" if nm == "base" else ""
            print(f"  {nm + mark:<14}{t:>9}{f'{w}/{t}':>12}{med:>12,.0f}{bmed:>13,.0f}"
                  f"{mdd:>10.1f}%{bdd:>9.1f}%")

    rolling("roll_", "ROLLING 12-MONTH WINDOWS, MONTHLY STARTS — all history")
    rolling("r6_", "ROLLING 6-MONTH WINDOWS, MONTHLY STARTS — all history")
    rolling("r6_", "ROLLING 6-MONTH WINDOWS — OLD REGIME ONLY (starts 2023-07 .. 2024-12)",
            hi="2024-12")
    rolling("r6_", "ROLLING 6-MONTH WINDOWS — RECENT REGIME ONLY (starts 2025-01 onward)",
            lo="2025-01")

    # ---- cost band + placebo ----
    print("\n" + "=" * 112)
    print("COST SENSITIVITY + PLACEBO — 15-month window")
    print("=" * 112)
    print(f"  {'exit rule':<14}{'1.0x $':>12}{'1.5x $':>12}{'2.0x $':>12}"
          f"{'placebo $':>12}{'edge vs placebo':>18}")
    order = sorted(NAMES, key=lambda n: -(res.get(("15mo", n), {}).get("final", 0)))
    for nm in order:
        a = res.get(("15mo", nm))
        if not a:
            continue
        b = res.get(("15mo_c15", nm))
        c = res.get(("15mo_c20", nm))
        p = res.get(("15mo_plc", nm))
        mark = " *" if nm == "base" else ""
        print(f"  {nm + mark:<14}{a['final']:>12,.0f}"
              f"{(b['final'] if b else 0):>12,.0f}{(c['final'] if c else 0):>12,.0f}"
              f"{(p['final'] if p else 0):>12,.0f}"
              f"{(a['final'] - (p['final'] if p else 0)):>+18,.0f}")

    # ---- month by month, base vs best few ----
    top = [n for n, _ in sorted(
        [(n, res[("15mo", n)]["final"]) for n in NAMES if ("15mo", n) in res],
        key=lambda x: -x[1])][:3]
    picks = ["base"] + [t for t in top if t != "base"][:3]
    months = sorted({m for n in picks for m in res[("15mo", n)]["monthly"]})
    print("\n" + "=" * 112)
    print("MONTH BY MONTH — 15 months from $1,500 (balance path)")
    print("=" * 112)
    hdr = f"  {'month':<10}"
    for n in picks:
        hdr += f"{n:>16}"
    print(hdr)
    bal = {n: START for n in picks}
    for m in months:
        row = f"  {m:<10}"
        for n in picks:
            bal[n] += res[("15mo", n)]["monthly"].get(m, 0.0)
            row += f"{bal[n]:>16,.0f}"
        print(row)

    # ---- when does trailing win? condition on BTC trend ----
    print("\n" + "=" * 112)
    print("WHEN DOES TRAILING WIN? — each rolling 6-month window vs BTC's move in it")
    print("=" * 112)
    b = pd.read_parquet(ROOT / "cache_3yr_1h" / "BTCUSDT.parquet").sort_values("start")
    b["start"] = pd.to_datetime(b["start"])
    bs = b.set_index("start")["close"]
    cands = ["s3_a1", "s2_a1", "s1_a1", "be1_s2_a1"]
    print(f"  {'window':<12}{'BTC 6mo':>10}{'base $':>12}"
          + "".join(f"{c:>12}" for c in cands))
    diffs = {c: [] for c in cands}
    btcr = []
    for s0 in r6:
        k = f"r6_{s0:%Y-%m}"
        if (k, "base") not in res:
            continue
        s1 = s0 + pd.DateOffset(months=6)
        w = bs[(bs.index >= s0) & (bs.index < s1)]
        if len(w) < 100:
            continue
        ret = float(w.iloc[-1] / w.iloc[0] - 1)
        btcr.append(ret)
        base_f = res[(k, "base")]["final"]
        row = f"  {s0:%Y-%m}      {ret:>+9.0%}{base_f:>12,.0f}"
        for c in cands:
            v = res.get((k, c))
            f = v["final"] if v else float("nan")
            diffs[c].append(f - base_f)
            row += f"{f:>12,.0f}"
        print(row)
    print()
    for c in cands:
        d_ = np.array(diffs[c], dtype=float)
        m = np.isfinite(d_)
        if m.sum() > 3:
            r_ = float(np.corrcoef(np.array(btcr)[m], d_[m])[0, 1])
            print(f"  corr(BTC 6mo return, {c} minus base) = {r_:+.2f}   "
                  f"[negative = trailing helps most when BTC is weak]")

    # ---- R distribution: what does the trail actually cost? ----
    print("\n" + "=" * 112)
    print("WHERE THE DIFFERENCE COMES FROM — R distribution, last 15 months")
    print("=" * 112)
    seg15 = real[real.entry_time >= END - pd.DateOffset(months=15)]
    print(f"  {'exit rule':<14}{'mean R':>9}{'WR':>8}{'p90 R':>8}{'p99 R':>8}"
          f"{'max R':>9}{'R>5 %':>8}{'R>10 %':>8}{'sum top1%':>11}")
    for nm in ["base", "s4_a2", "s3_a1", "s2_a1", "s1_a1", "be1", "be1_s2_a1"]:
        v = seg15[f"r_{nm}"].dropna().values
        if not len(v):
            continue
        cut = np.quantile(v, 0.99)
        print(f"  {nm:<14}{v.mean():>+9.3f}{(v > 0).mean():>8.1%}"
              f"{np.quantile(v, 0.90):>8.2f}{cut:>8.2f}{v.max():>9.1f}"
              f"{(v > 5).mean():>8.2%}{(v > 10).mean():>8.2%}"
              f"{v[v >= cut].sum() / max(1e-9, v.sum()):>10.0%}")

    # ---- intrabar ambiguity (the 5m question) ----
    print("\n" + "=" * 112)
    print("INTRABAR AMBIGUITY — how much would 5-MINUTE data change any of this?")
    print("=" * 112)
    seg = real[real.entry_time >= END - pd.DateOffset(months=15)]
    for nm in ["base", "s2_a1", "be1_s2_a1", "s3_a2"]:
        col = f"r_{nm}"
        if col not in seg:
            continue
        losers = (seg[col] < -0.9).mean()
        print(f"  {nm:<14} full-stop rate {losers:>6.1%}   "
              f"mean R {seg[col].mean():>+7.3f}   median hold "
              f"{(pd.to_datetime(seg[f'x_{nm}']) - seg.entry_time).median()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
