#!/usr/bin/env python3
"""Search for a high ROI/DD configuration by attacking the ACTUAL source of drawdown.

Diagnosis first, then treatment. On the ten worst days of the recommended configuration,
shorts lost 1,025 R while longs made +21 R, with 100-165 trades resolving on a single day.
The drawdown is not a slow bleed and it is not bad luck on individual trades — it is a
one-sided, heavily correlated short book being run over in a rally.

That tells you which knobs can possibly work. Sizing down (tested) scales return and
drawdown together and leaves ROI/DD flat. Drawdown throttles (tested) cut risk AFTER the
damage and give back 96% of the equity because the recovery is where the money is. Neither
touches the cause.

What might: bounding the CORRELATED EXPOSURE itself.
  net_dir_cap     |long risk - short risk| <= x * equity. This is the direct instrument.
                  Live value 0.10 was validated years ago against a different config.
  open_risk_cap   total open risk <= x * equity.
  mode 'scale'    when a cap binds, SHRINK the new trade to fit instead of rejecting it.
                  Rejecting is a lottery on arrival order — the trades that get blocked are
                  whichever happened to be processed last, not the worst ones. Scaling keeps
                  the same trades at lower size, which is a smoother and less arbitrary
                  version of the same budget.

Everything is fitted on DEV+VAL and judged on the untouched HOLDOUT. A configuration that
only wins in-sample is a negative result, not a candidate.
"""
from __future__ import annotations

import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import backtest_production_correct as P  # noqa: E402
from backtest_shadow_gate import LIVE as LIVE_SIM, LIVE_TAPER  # noqa: E402
import compare_bots as C  # noqa: E402
from variations import with_halt  # noqa: E402

START = 1500.0
IS_END = "2026-05-25"          # everything at or after this is the untouched holdout


def sim(d, chop, t0=None, t1=None, base=0.003, **over):
    sub = d
    if t0 is not None:
        sub = sub[sub.entry_time >= t0]
    if t1 is not None:
        sub = sub[sub.entry_time < t1]
    if len(sub) < 100:
        return None
    sc = base / 0.003
    kw = dict(LIVE_SIM)
    kw.update(starting_balance=START, base_risk=base,
              custom_taper=[(t, r * sc) for t, r in LIVE_TAPER],
              btc_bull_col="btc_bull", btc_short_col="btc_impulse",
              taper_basis="wallet", size_basis="wallet")
    kw.update(over)
    P.STARTING_BALANCE = START
    cols = ["entry_time", "exit_time", "entry_price", "sl_price", "r_result",
            "side", "symbol", "btc_bull", "btc_impulse"]
    r = P.run_simulation(sub[cols].reset_index(drop=True), chop, **kw)
    t = pd.DataFrame(r["entered_trades"])
    if t.empty:
        return None
    t = t.sort_values("exit_time")
    c = pd.Series((START + t.pnl.cumsum()).values, index=pd.to_datetime(t.exit_time.values))
    pk = c.cummax()
    dd = float(((pk - c) / pk).max() * 100)
    roi = (r["final_effective"] / START - 1) * 100
    return dict(final=r["final_effective"], roi=roi, dd=dd, n=len(t),
                roidd=roi / dd if dd > 0 else np.nan)


def main():
    P.ROUND_TRIP_COST = 0.00242
    uni = pd.read_parquet(HERE / "universe_trail.parquet")
    chop = P.load_chop_data(sorted(uni.symbol.unique()))
    d = with_halt(C.load("uni_glob_rr10_am3_same.parquet", "r_result", "exit_time"),
                  21, -300, 200)

    base = sim(d, chop, t1=IS_END)
    bh = sim(d, chop, t0=IS_END)
    print(f"BASELINE (recommended config)  in-sample ROI {base['roi']:,.0f}%  "
          f"DD {base['dd']:.1f}%  ROI/DD {base['roidd']:.1f}   |   holdout ROI {bh['roi']:+.1f}%")
    print()
    print("=" * 104)
    print("EXPOSURE-BUDGET SEARCH — fitted on 2023-06..2026-05-25, judged on the holdout")
    print("=" * 104)
    print(f"{'net_dir':>8}{'gross':>8}{'mode':>7}{'  IS ROI%':>11}{'IS DD%':>8}"
          f"{'IS ROI/DD':>11}{'HOLD ROI%':>11}{'HOLD DD%':>10}{'trades':>8}")

    rows = []
    grid = itertools.product([0.02, 0.03, 0.05, 0.075, 0.10, 0.20, None],
                             [0.10, 0.20, 0.30, 0.60],
                             ["block", "scale"])
    for nd, gr, mode in grid:
        over = dict(net_dir_cap=nd, open_risk_cap=gr, open_risk_mode=mode)
        a = sim(d, chop, t1=IS_END, **over)
        b = sim(d, chop, t0=IS_END, **over)
        if a is None or b is None:
            continue
        rows.append(dict(net_dir=nd, gross=gr, mode=mode,
                         is_roi=a["roi"], is_dd=a["dd"], is_roidd=a["roidd"],
                         hold_roi=b["roi"], hold_dd=b["dd"], n=a["n"]))
        print(f"{str(nd):>8}{gr:>8.2f}{mode:>7}{a['roi']:>11,.0f}{a['dd']:>8.1f}"
              f"{a['roidd']:>11.1f}{b['roi']:>11.1f}{b['dd']:>10.1f}{a['n']:>8,}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "results" / "search_roidd.csv", index=False)

    print()
    print("=" * 104)
    print("CANDIDATES — must beat baseline ROI/DD in-sample AND stay positive on the holdout")
    print("=" * 104)
    ok = df[(df.is_roidd > base["roidd"]) & (df.hold_roi > 0)].sort_values("is_roidd",
                                                                          ascending=False)
    if ok.empty:
        print("  none. Exposure budgeting alone does not produce a better ROI/DD that survives.")
    else:
        print(ok.head(12).to_string(index=False, float_format=lambda x: f"{x:8.2f}"))


if __name__ == "__main__":
    main()
