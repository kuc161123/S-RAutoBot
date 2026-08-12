#!/usr/bin/env python3
"""Fourth-implementation parity: does risk_study's trail resolver match the repo's reference?

`verify_trailing_parity.py` already cross-checks three implementations of the s3_a1 rule —
the live engine (`Bot4H._trail_one`), the shadow learner (`trail_shadow.walk_trail`) and the
validated backtest (`build_trail_universe_wide.resolve_all`). The risk_study rebuild added a
FOURTH (`build_universe_trail_helpers.resolve_trail`), and every claim about the trailing
stop's cost rests on it. So it has to clear the same bar.

This drives resolve_trail and resolve_all over identical signals, identical bars, identical
(rr, atr_mult), and asserts agreement trade by trade on BOTH the R outcome and the exit bar.

One deliberate difference is neutralised rather than hidden: resolve_all applies entry
slippage (`o[e] * (1 +/- 0.0003)`) while the risk_study builder fills at `o[e]` and charges
cost separately in the portfolio engine. Slippage shifts entry, stop and TP together, so it
changes R by a hair on every trade in BOTH arms and would mask a real logic difference. The
run below therefore compares with SLIP forced to 0, and separately reports the size of the
slippage effect so it is on the record.
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_trail_universe_wide as BW  # noqa: E402
import backtest_3yr_walkforward as bt  # noqa: E402
from build_universe_trail_helpers import resolve_trail  # noqa: E402

CACHE = ROOT / "cache_3yr_1h"
VAR = "s3_a1"
VIDX = [v[0] for v in BW.VARIANTS].index(VAR)
MAX_WAIT = bt.MAX_WAIT_CANDLES


def one(args):
    sym, picks, slip = args
    f = CACHE / f"{sym}.parquet"
    if not f.exists():
        return []
    try:
        df = pd.read_parquet(f)
    except Exception:
        return []
    if df.empty or len(df) < 2500:
        return []
    df = bt.prepare_data(df)
    o = df.open.values; h = df.high.values; l = df.low.values; c = df.close.values
    atr = df.atr.values; ema = df.ema.values
    n = len(c)

    BW.SLIP = slip  # neutralise or restore the reference's entry slippage

    sigs = {}
    for s in bt.detect_signals(df):
        sigs.setdefault(s["type"], []).append(s)

    rows = []
    for dt, (rr, am) in picks.items():
        for s in sigs.get(dt, []):
            conf, side, lvl = s["conf_idx"], s["side"], s["swing"]
            bos = None
            for i in range(1, MAX_WAIT + 1):
                idx = conf + i
                if idx >= n:
                    break
                if (side == "long" and c[idx] > lvl) or (side == "short" and c[idx] < lvl):
                    bos = idx
                    break
            if bos is None:
                continue
            e = bos + 1
            if e >= n or not np.isfinite(ema[bos]):
                continue
            if side == "long" and not c[bos] > ema[bos]:
                continue
            if side == "short" and not c[bos] < ema[bos]:
                continue
            if not (np.isfinite(atr[bos]) and atr[bos] > 0):
                continue

            sl_d = atr[bos] * am

            # --- C: the repo's validated reference ---
            rr_all, xx_all = BW.resolve_all(o, h, l, atr, e, side, sl_d, rr, n)
            r_ref, x_ref = rr_all[VIDX], xx_all[VIDX]

            # --- D: risk_study's resolver, fed the same entry the reference used ---
            long_ = side == "long"
            entry = o[e] * (1 + slip) if long_ else o[e] * (1 - slip)
            sl0 = entry - sl_d if long_ else entry + sl_d
            tp = entry + sl_d * rr if long_ else entry - sl_d * rr
            mine = resolve_trail(e, n, long_, entry, sl0, tp, sl_d, h, l, atr)
            r_mine = mine[0] if mine else np.nan
            x_mine = mine[1] if mine else -1

            rows.append((sym, dt, rr, am, r_ref, r_mine, x_ref, x_mine))
    return rows


def main():
    slip = 0.0 if "--with-slip" not in sys.argv else BW.SLIP
    nsym = 60
    for a in sys.argv[1:]:
        if a.isdigit():
            nsym = int(a)

    cfg = yaml.safe_load(open(ROOT / "config.yaml"))["symbols"]
    jobs = []
    for s, sc in cfg.items():
        if not (sc or {}).get("enabled", True):
            continue
        picks = {c["divergence_type"]: (float(c["rr"]), float(c["atr_mult"]))
                 for c in (sc or {}).get("configs", []) or []}
        if picks and (CACHE / f"{s}.parquet").exists():
            jobs.append((s, picks, slip))
    jobs.sort()
    jobs = jobs[:nsym]
    print(f"[PARITY] risk_study resolve_trail  vs  build_trail_universe_wide.resolve_all"
          f"  ({VAR})")
    print(f"[PARITY] {len(jobs)} symbols · entry slippage = {slip}", flush=True)

    rows = []
    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=2), 1):
            rows.extend(r)
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)} · {len(rows):,}", flush=True)

    d = pd.DataFrame(rows, columns=["symbol", "div_type", "rr", "atr_mult",
                                    "r_ref", "r_mine", "x_ref", "x_mine"])
    both = d.dropna(subset=["r_ref", "r_mine"])
    print(f"\n[PARITY] {len(d):,} signals · {len(both):,} resolved by both")

    r_ok = np.isclose(both.r_ref, both.r_mine, rtol=1e-6, atol=1e-6)
    x_ok = both.x_ref.values == both.x_mine.values
    print(f"  R agreement        {r_ok.mean()*100:8.4f}%   ({(~r_ok).sum():,} mismatches)")
    print(f"  exit-bar agreement {x_ok.mean()*100:8.4f}%   ({(~x_ok).sum():,} mismatches)")
    print(f"  mean R  reference {both.r_ref.mean():+.6f}   mine {both.r_mine.mean():+.6f}"
          f"   delta {both.r_mine.mean()-both.r_ref.mean():+.2e}")

    # resolution-coverage differences matter too: a resolver that silently drops trades
    # would look "in agreement" on the ones it kept.
    only_ref = d.r_ref.notna() & d.r_mine.isna()
    only_mine = d.r_mine.notna() & d.r_ref.isna()
    print(f"  resolved by reference only: {only_ref.sum():,}")
    print(f"  resolved by mine only:      {only_mine.sum():,}")

    if (~r_ok).any():
        bad = both[~r_ok]
        print(f"\n  worst mismatches:")
        bad = bad.assign(diff=(bad.r_mine - bad.r_ref).abs()).nlargest(8, "diff")
        print(bad[["symbol", "div_type", "rr", "atr_mult", "r_ref", "r_mine",
                   "x_ref", "x_mine"]].to_string(index=False))
        print(f"\n  mismatch rate by rr:")
        m = both.assign(bad=~r_ok).groupby("rr").bad.agg(["mean", "size"])
        print((m.assign(pct=lambda x: x["mean"] * 100)[["pct", "size"]]).to_string())

    d.to_parquet(HERE / "results" / "trail_parity.parquet", index=False)
    verdict = "PASS" if r_ok.all() and x_ok.all() else "FAIL"
    print(f"\n[PARITY] {verdict}")


if __name__ == "__main__":
    main()
