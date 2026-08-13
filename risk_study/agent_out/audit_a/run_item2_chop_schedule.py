#!/usr/bin/env python3
"""AUDIT-A item 2: regime-aware CHOP schedule, tested through the production engine
on the ungated global rr10/atr3 population (so the engine's own internal gate controls
admission at every level tested, including looser-than-52 and no-gate).
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
    d = H.load_ungated_halted()
    print(f"ungated+halted population: {len(d)} rows, {d.entry_time.min()}..{d.entry_time.max()}")

    rows = []
    for name, thresholds, note in VARIANTS:
        H.reset_globals()
        if thresholds is not None:
            H.P.CHOP_THRESHOLDS = thresholds
        row = H.run_variant(f"CHOP[{name}]", d, chop, cfg_diff=str(thresholds), note=note)
        rows.append(row)
        H.reset_globals()
        pd.DataFrame(rows).to_csv(HERE / "item2_chop_schedule_results.csv", index=False)

    print("\nwrote item2_chop_schedule_results.csv")


if __name__ == "__main__":
    main()
