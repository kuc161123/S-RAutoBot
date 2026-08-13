#!/usr/bin/env python3
"""AUDIT-A item 5: taper schedule, current descending schedule vs flat 0.3% at every
balance. Corrects the same short_gate=True staleness noted in item 4/harness.py.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as H  # noqa: E402

FLAT = [(0, 0.003)]


def main():
    chop = H.load_chop_cache()
    d = H.load_gated_halted()
    print(f"gated+halted population: {len(d)} rows")

    rows = []
    row = H.run_variant("taper=current (live)", d, chop,
                         cfg_diff="none (baseline)", note="live schedule",
                         custom_taper=H.LIVE_TAPER)
    rows.append(row)
    row = H.run_variant("taper=flat 0.3%", d, chop,
                         cfg_diff="taper_schedule -> flat 0.3% at all balances",
                         note="descending schedule removed",
                         custom_taper=FLAT)
    rows.append(row)
    pd.DataFrame(rows).to_csv(HERE / "item5_taper_results.csv", index=False)
    print("\nwrote item5_taper_results.csv")


if __name__ == "__main__":
    main()
