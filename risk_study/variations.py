#!/usr/bin/env python3
"""Full variation matrix: every feature toggle x every parameter/exit configuration.

All runs are LEAK-FREE (size_basis='wallet'). The engine's 'equity' basis marks open
positions by interpolating toward their already-known final PnL, which leaks the future
into the sizing of concurrent trades; it inflates final equity by 1.2-1.6x and inflates it
MOST for the arms with the biggest winners, so it is not safe for cross-arm comparison.
The live bot does size on real equity, so 'wallet' is a conservative lower bound and the
truth sits between — but only 'wallet' is honest for ranking.

Feature toggles, applied one at a time so each row is attributable:
  REGIME OFF   scenario='chop_only' — keeps CHOP and the balance taper, drops ONLY the
               last-20-trade regime multiplier, so risk stays at the tapered base
  NO CAPS      net_directional_cap and gross_open_risk_cap both disabled
  NO BOOST     long_bull_boost 1.3 -> 1.0
  NO GATE      BTC impulse-bull short gate off
  FLAT RISK    taper schedule removed (flat 0.3% at every balance)
  ALL OFF      regime off + no caps + no boost + no gate (pure signal + taper)
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
from backtest_shadow_gate import LIVE as LIVE_SIM, LIVE_TAPER, gate_series  # noqa: E402
import compare_bots as C  # noqa: E402

START = 1500.0
FLAT = [(0, 0.003)]
COST = 0.00242


def with_halt(d, window_d, stop_r, start_r):
    """Apply the shadow-R halt as a trade filter.

    `gate_series` builds an hourly on/off state machine from the trailing sum of RESOLVED
    net R with hysteresis (halt at stop_r, resume at start_r). Live this is ADVISORY —
    config.yaml sets mode: advisory, so it only tells the operator to run /stop or /start;
    it does not block a trade in code. Modelling it as a hard filter is therefore the
    OPTIMISTIC reading: it assumes the operator acts instantly every time. The repo's own
    note says acting a week late costs most of the benefit.
    """
    x = d.copy()
    x["r_net"] = x.r_result - COST / x.stop_frac
    g = gate_series(x, window_d, stop_r, start_r)
    allow = x.entry_time.dt.floor("h").map(g.to_dict()).fillna(True).astype(bool)
    return x[allow].reset_index(drop=True)

CONFIGS = [
    ("OLD-BASE fitted+fixed", "universe_trail.parquet", "r_fixed", "exit_fixed"),
    ("CURRENT  fitted+trail", "universe_trail.parquet", "r_trail", "exit_trail"),
    ("GLOBAL   rr10a3+trail", "uni_glob_rr10_am3_same.parquet", "r_result", "exit_time"),
    ("GLB-10/2 rr10a2+trail", "uni_glob_rr10_am2_same.parquet", "r_result", "exit_time"),
]

VARIANTS = [
    ("live defaults", {}),
    ("REGIME OFF", dict(scenario="chop_only")),
    ("NO CAPS", dict(net_dir_cap=None, open_risk_cap=None)),
    ("NO BOOST", dict(long_boost=1.0)),
    ("NO GATE", dict(short_gate=False)),
    ("FLAT RISK (no taper)", dict(custom_taper=FLAT)),
    ("REGIME OFF + NO CAPS", dict(scenario="chop_only", net_dir_cap=None, open_risk_cap=None)),
    ("ALL OFF", dict(scenario="chop_only", net_dir_cap=None, open_risk_cap=None,
                     long_boost=1.0, short_gate=False)),
    # shadow-R halt. Thresholds are config.yaml's: 7d window -300/+400, 21d -300/+200.
    ("HALT 7d", {}, ("halt", 7, -300, 400)),
    ("HALT 21d", {}, ("halt", 21, -300, 200)),
    ("HALT 7d + REGIME OFF", dict(scenario="chop_only"), ("halt", 7, -300, 400)),
    ("HALT 21d + REGIME OFF", dict(scenario="chop_only"), ("halt", 21, -300, 200)),
]
VARIANTS = [(v[0], v[1], (v[2] if len(v) > 2 else None)) for v in VARIANTS]


def run(d, chop, **over):
    kw = dict(LIVE_SIM)
    kw.update(starting_balance=START, base_risk=0.003, custom_taper=LIVE_TAPER,
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
    # profit concentration: share of total net R from the top 1% of trades
    rr = np.sort(t.pnl.to_numpy())[::-1]
    conc = rr[:max(1, len(rr) // 100)].sum() / rr.sum() * 100 if rr.sum() > 0 else np.nan
    return dict(final=r["final_effective"], dd=dd, n=len(t), hold=ho,
                ratio=r["final_effective"] / dd if dd > 0 else np.nan, conc=conc)


def main():
    P.ROUND_TRIP_COST = 0.00242
    uni = pd.read_parquet(HERE / "universe_trail.parquet")
    chop = P.load_chop_data(sorted(uni.symbol.unique()))
    data = {nm: C.load(p, rc, xc) for nm, p, rc, xc in CONFIGS}

    rows = []
    for vname, over, halt in VARIANTS:
        print(f"\n{'=' * 96}\n{vname}\n{'=' * 96}")
        print(f"{'config':<24}{'trades':>8}{'final $':>11}{'maxDD%':>9}{'$/DDpt':>9}"
              f"{'holdout%':>10}{'top1%R':>9}")
        for cname, *_ in CONFIGS:
            d = data[cname]
            if halt is not None:
                d = with_halt(d, halt[1], halt[2], halt[3])
            r = run(d, chop, **over)
            if r is None:
                continue
            rows.append(dict(variant=vname, config=cname, **r))
            print(f"{cname:<24}{r['n']:>8,}{r['final']:>11,.0f}{r['dd']:>9.1f}"
                  f"{r['ratio']:>9.0f}{r['hold']:>10.1f}{r['conc']:>9.1f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "results" / "variations.csv", index=False)

    print(f"\n{'=' * 96}\nSUMMARY — best variant per config, by \\$ per drawdown point\n{'=' * 96}")
    for cname, *_ in CONFIGS:
        s = df[df.config == cname].sort_values("ratio", ascending=False)
        b = s.iloc[0]
        print(f"  {cname:<24} best={b.variant:<22} ${b.final:>9,.0f}  DD {b.dd:>5.1f}%  "
              f"hold {b.hold:>+6.1f}%")

    print(f"\n{'=' * 96}\nEFFECT OF EACH TOGGLE (median across the 4 configs, vs live defaults)\n{'=' * 96}")
    base = df[df.variant == "live defaults"].set_index("config")
    for vname, _, _h in VARIANTS[1:]:
        s = df[df.variant == vname].set_index("config")
        rel = (s.final / base.final).median()
        dd = (s.dd - base.dd).median()
        ho = (s.hold - base.hold).median()
        print(f"  {vname:<24} equity x{rel:>5.2f}   DD {dd:>+6.1f} pts   holdout {ho:>+6.1f} pts")


if __name__ == "__main__":
    main()
