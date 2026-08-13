#!/usr/bin/env python3
"""AUDIT-A item 2b: same CHOP-schedule comparison as item 2, but WITHOUT the shadow
halt overlay. The halt is advisory-only in production (a human must act on /stop), so
mixing it with the CHOP test muddies which feature is doing the work. This isolates the
CHOP-schedule effect alone on the raw ungated population.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as H  # noqa: E402

VARIANTS = [
    ("current (live, inverted)", None, "CHOP_THRESHOLDS unchanged: {fav:52,caut:45,adv:52,crit:55}"),
    ("flat-45", {"favorable": 45, "cautious": 45, "adverse": 45, "critical": 45}, "flat 45 all regimes"),
    ("flat-52", {"favorable": 52, "cautious": 52, "adverse": 52, "critical": 52}, "flat 52 all regimes"),
    ("corrected (tighter-when-worse)", {"favorable": 55, "cautious": 48, "adverse": 42, "critical": 35},
     "inversion flipped: looser when favorable, tighter when critical"),
    ("no gate", {"favorable": 999, "cautious": 999, "adverse": 999, "critical": 999}, "CHOP filter removed entirely"),
]


def main():
    chop = H.load_chop_cache()
    d = pd.read_parquet(HERE / "ungated_glob_rr10_am3.parquet")
    cutoff = d.entry_time.max() - pd.Timedelta(days=21)
    d = d[d.entry_time <= cutoff].reset_index(drop=True)
    print(f"ungated, NO halt, 21d-trimmed population: {len(d)} rows, "
          f"{d.entry_time.min()}..{d.entry_time.max()}")

    rows = []
    for name, thresholds, note in VARIANTS:
        H.reset_globals()
        if thresholds is not None:
            H.P.CHOP_THRESHOLDS = thresholds
        row = H.run_variant(f"CHOP[{name}]", d, chop, cfg_diff=str(thresholds), note=note)
        rows.append(row)
        H.reset_globals()
        pd.DataFrame(rows).to_csv(HERE / "item2b_chop_nohalt_results.csv", index=False)

    print("\nwrote item2b_chop_nohalt_results.csv")


if __name__ == "__main__":
    main()
