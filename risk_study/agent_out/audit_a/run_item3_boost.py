#!/usr/bin/env python3
"""AUDIT-A item 3: long_bull_boost sweep, 1.3 (live) vs {1.0, 1.1, 1.5, 2.0}.
Uses the already CHOP-gated (<52, current schedule) population -- boost doesn't touch
CHOP admission, only long-side risk sizing when BTC>EMA200, so the gated file is correct
and faster.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as H  # noqa: E402

VALUES = [1.0, 1.1, 1.3, 1.5, 2.0]


def main():
    chop = H.load_chop_cache()
    d = H.load_gated_halted()
    print(f"gated+halted population: {len(d)} rows")

    rows = []
    for v in VALUES:
        tag = "live" if v == 1.3 else ("no boost" if v == 1.0 else f"boost={v}")
        row = H.run_variant(f"long_boost={v} ({tag})", d, chop,
                             cfg_diff=f"long_boost: 1.3 -> {v}", note=tag,
                             long_boost=v)
        rows.append(row)
        pd.DataFrame(rows).to_csv(HERE / "item3_boost_results.csv", index=False)

    print("\nwrote item3_boost_results.csv")


if __name__ == "__main__":
    main()
