#!/usr/bin/env python3
"""Is turning the regime multiplier OFF a real improvement, or just more leverage?

Turning it off looks spectacular: GLOBAL rr10/atr3 goes $17,493 -> $60,396. But the regime
multiplier is 0.1 / 0.25 / 0.5 / 1.0, and it spends most of its life well below 1.0 — so
"regime off" silently multiplies the average deployed risk by ~3x. Comparing it to regime-on
at the same NOMINAL base risk compares two different bet sizes, not two different policies.

This equalises them. Step 1 measures the realised average multiplier. Step 2 re-runs
regime-off with base risk divided by that factor, so both arms deploy the same average risk
per trade. If regime-off still wins at matched risk it is a genuine feature. If the advantage
disappears, it was leverage all along — and the sizing study already established that this
bot is at or above its growth-optimal risk fraction, so buying growth with leverage here is
the one thing the evidence most clearly rules out.
"""
from __future__ import annotations

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
CONFIGS = [
    ("GLOBAL rr10a3+trail", "uni_glob_rr10_am3_same.parquet", "r_result", "exit_time"),
    ("CURRENT fitted+trail", "universe_trail.parquet", "r_trail", "exit_trail"),
]


def run(d, chop, base=0.003, taper=None, **over):
    kw = dict(LIVE_SIM)
    kw.update(starting_balance=START, base_risk=base,
              custom_taper=taper if taper is not None else LIVE_TAPER,
              btc_bull_col="btc_bull", btc_short_col="btc_impulse",
              taper_basis="wallet", size_basis="wallet")
    kw.update(over)
    P.STARTING_BALANCE = START
    cols = ["entry_time", "exit_time", "entry_price", "sl_price", "r_result",
            "side", "symbol", "btc_bull", "btc_impulse"]
    r = P.run_simulation(d[cols].reset_index(drop=True), chop, **kw)
    t = pd.DataFrame(r["entered_trades"])
    if t.empty:
        return None
    t = t.sort_values("exit_time")
    c = pd.Series((START + t.pnl.cumsum()).values, index=pd.to_datetime(t.exit_time.values))
    pk = c.cummax()
    dd = float(((pk - c) / pk).max() * 100)
    h = c[c.index >= pd.Timestamp("2026-05-31")]
    ho = (h.iloc[-1] / h.iloc[0] - 1) * 100 if len(h) > 2 else np.nan
    mult = float(t.regime_mult.mean()) if "regime_mult" in t else np.nan
    # risk actually deployed, as a fraction of the balance at the time of each entry
    rf = float((t.risk_usd / t.balance_after.shift().fillna(START)).mean())
    return dict(final=r["final_effective"], dd=dd, n=len(t), hold=ho,
                ratio=r["final_effective"] / dd if dd > 0 else np.nan,
                mult=mult, riskfrac=rf * 100)


def main():
    P.ROUND_TRIP_COST = 0.00242
    uni = pd.read_parquet(HERE / "universe_trail.parquet")
    chop = P.load_chop_data(sorted(uni.symbol.unique()))

    print("=" * 100)
    print("STEP 1 — how much leverage does 'regime off' actually add?")
    print("=" * 100)
    print(f"{'arm':<34}{'avg mult':>10}{'avg risk%':>11}{'final $':>11}{'maxDD%':>9}{'hold%':>9}")
    facts = {}
    for cname, p, rc, xc in CONFIGS:
        d = C.load(p, rc, xc)
        on = run(d, chop)
        off = run(d, chop, scenario="chop_only")
        facts[cname] = (d, on, off)
        print(f"{cname + '  regime ON':<34}{on['mult']:>10.3f}{on['riskfrac']:>11.4f}"
              f"{on['final']:>11,.0f}{on['dd']:>9.1f}{on['hold']:>9.1f}")
        print(f"{cname + '  regime OFF':<34}{off['mult']:>10.3f}{off['riskfrac']:>11.4f}"
              f"{off['final']:>11,.0f}{off['dd']:>9.1f}{off['hold']:>9.1f}")
        print(f"{'':>34}{'':>10}  -> regime-off deploys "
              f"{off['riskfrac']/on['riskfrac']:.2f}x the risk\n")

    print("=" * 100)
    print("STEP 2 — regime OFF at MATCHED average risk (base scaled down by the factor above)")
    print("=" * 100)
    print(f"{'arm':<34}{'base%':>8}{'avg risk%':>11}{'final $':>11}{'maxDD%':>9}"
          f"{'$/DDpt':>9}{'hold%':>9}")
    rows = []
    for cname, *_ in CONFIGS:
        d, on, off = facts[cname]
        k = on["riskfrac"] / off["riskfrac"]
        print(f"{cname + '  regime ON':<34}{0.3:>8.3f}{on['riskfrac']:>11.4f}"
              f"{on['final']:>11,.0f}{on['dd']:>9.1f}{on['ratio']:>9.0f}{on['hold']:>9.1f}")
        rows.append(dict(arm=cname, variant="regime ON", **on))
        for label, scale in (("matched", k), ("half-way", (1 + k) / 2)):
            b = 0.003 * scale
            m = run(d, chop, base=b, taper=[(t, r * scale) for t, r in LIVE_TAPER],
                    scenario="chop_only")
            print(f"{cname + f'  regime OFF ({label})':<34}{b*100:>8.3f}{m['riskfrac']:>11.4f}"
                  f"{m['final']:>11,.0f}{m['dd']:>9.1f}{m['ratio']:>9.0f}{m['hold']:>9.1f}")
            rows.append(dict(arm=cname, variant=f"regime OFF {label}", **m))
        print()

    print("=" * 100)
    print("STEP 3 — the same question asked as a pure risk sweep, regime OFF")
    print("=" * 100)
    print("If 'regime off' is only leverage, its curve should sit ON the regime-on risk curve,")
    print("just shifted along it — not above it.\n")
    d, on, _ = facts["GLOBAL rr10a3+trail"]
    print(f"{'base risk':<12}{'regime ON: $':>14}{'DD%':>7}{'hold%':>8}"
          f"{'  |  regime OFF: $':>20}{'DD%':>7}{'hold%':>8}")
    for b in (0.001, 0.002, 0.003, 0.005, 0.0075, 0.01):
        sc = b / 0.003
        tp = [(t, r * sc) for t, r in LIVE_TAPER]
        a = run(d, chop, base=b, taper=tp)
        o = run(d, chop, base=b, taper=tp, scenario="chop_only")
        print(f"{b*100:<12.2f}{a['final']:>14,.0f}{a['dd']:>7.1f}{a['hold']:>8.1f}"
              f"{o['final']:>20,.0f}{o['dd']:>7.1f}{o['hold']:>8.1f}")
        rows.append(dict(arm="GLOBAL sweep", variant=f"ON f={b}", **a))
        rows.append(dict(arm="GLOBAL sweep", variant=f"OFF f={b}", **o))

    pd.DataFrame(rows).to_csv(HERE / "results" / "regime_test.csv", index=False)
    print(f"\nwrote results/regime_test.csv")


if __name__ == "__main__":
    main()
